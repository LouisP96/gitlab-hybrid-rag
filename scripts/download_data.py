#!/usr/bin/env python3
import requests
import os
import subprocess
import time
import json
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional


# Constants
RATE_LIMIT_DELAY = 0.5
DEFAULT_TIMEOUT = 30
DEFAULT_PER_PAGE = 100
PROGRESS_REPORT_INTERVAL = 10


@dataclass
class DownloadConfig:
    """Configuration for GitLab data download"""

    gitlab_url: str
    access_token: str
    output_dir: str
    pipeline_limit: int = 3
    timeout: int = DEFAULT_TIMEOUT
    per_page: int = DEFAULT_PER_PAGE


@dataclass
class CloneStats:
    """Statistics for repository cloning"""

    successful: int = 0
    failed: int = 0
    skipped: int = 0


class ProgressTracker:
    """Handles progress tracking and reporting"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def report_project_progress(self, current: int, total: int, project_name: str):
        """Report progress for project processing"""
        percentage = (current / total) * 100
        self.logger.info(
            f"Processing project: {project_name} - {current}/{total} ({percentage:.1f}%)"
        )

    def report_clone_progress(self, current: int, total: int, stats: CloneStats):
        """Report cloning progress summary"""
        if total > 1 and (current % PROGRESS_REPORT_INTERVAL == 0):
            percentage = (current / total) * 100
            self.logger.info("\n--- CLONING PROGRESS SUMMARY ---")
            self.logger.info(f"Completed: {current}/{total} ({percentage:.1f}%)")
            self.logger.info(f"Successful: {stats.successful}")
            self.logger.info(f"Failed: {stats.failed}")
            self.logger.info(f"Skipped: {stats.skipped}")
            self.logger.info(f"Remaining: {total - current}")
            self.logger.info("-----------------------\n")

    def report_final_summary(self, total_projects: int, stats: CloneStats):
        """Report final cloning summary"""
        self.logger.info("\n=== FINAL CLONING SUMMARY ===")
        self.logger.info(f"Total projects: {total_projects}")
        self.logger.info(f"Successful clones: {stats.successful}")
        self.logger.info(f"Failed clones: {stats.failed}")
        self.logger.info(f"Skipped clones: {stats.skipped}")

        if stats.failed > 0:
            self.logger.info(
                "Some clones failed. Check 'failed_clones.txt' to retry them later."
            )


class GitLabAPI:
    """Handles GitLab API interactions with error handling and retries"""

    def __init__(self, config: DownloadConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def get_all_pages(self, url: str, params: Optional[Dict] = None) -> List[Dict]:
        """Get all pages of a paginated API response with consistent error handling"""
        if params is None:
            params = {}

        headers = {"PRIVATE-TOKEN": self.config.access_token}
        all_items = []
        page = 1

        while True:
            page_params = {**params, "page": page, "per_page": self.config.per_page}

            try:
                response = requests.get(
                    url,
                    params=page_params,
                    headers=headers,
                    timeout=self.config.timeout,
                )

                if response.status_code != 200:
                    self.logger.error(
                        f"Error fetching {url}: {response.status_code} - {response.text}"
                    )
                    break

                items = response.json()
                if not items or (isinstance(items, dict) and not items.get("data", [])):
                    break

                # Handle both array responses and object responses with data field
                if isinstance(items, list):
                    all_items.extend(items)
                elif isinstance(items, dict) and "data" in items:
                    all_items.extend(items["data"])

                page += 1
                time.sleep(RATE_LIMIT_DELAY)

            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout fetching {url}, stopping pagination")
                break
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                break

        return all_items

    def get_single_resource(self, url: str) -> Optional[Dict]:
        """Get a single resource with error handling"""
        headers = {"PRIVATE-TOKEN": self.config.access_token}

        try:
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None


class GitLabDownloader:
    """Main class for downloading GitLab project data"""

    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.api = GitLabAPI(config, self.logger)
        self.progress = ProgressTracker(self.logger)
        self.clone_stats = CloneStats()

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(self.config.output_dir, "download_log.txt")
                ),
                logging.StreamHandler(),
            ],
        )
        return logging.getLogger(__name__)

    def get_projects(self, specific_project: Optional[str] = None) -> List[Dict]:
        """Get projects to download - either specific project or all public/internal"""
        if specific_project:
            return self._get_specific_project(specific_project)
        else:
            return self._get_all_projects()

    def _get_specific_project(self, project_path: str) -> List[Dict]:
        """Get a specific project by path"""
        self.logger.info(f"Fetching specific project: {project_path}")

        url = f"{self.config.gitlab_url}/api/v4/projects/{project_path.replace('/', '%2F')}"
        project = self.api.get_single_resource(url)

        if project:
            self.logger.info(f"Found project: {project['name']} (ID: {project['id']})")
            return [project]
        else:
            self.logger.error(f"Project '{project_path}' not found.")
            self.logger.error(
                "The format should be exactly as shown in GitLab, e.g., 'group/project'"
            )
            return []

    def _get_all_projects(self) -> List[Dict]:
        """Get all public and internal projects"""
        self.logger.info("Fetching all projects")
        all_projects = []

        for visibility in ["public", "internal"]:
            projects = self.api.get_all_pages(
                f"{self.config.gitlab_url}/api/v4/projects",
                params={"visibility": visibility},
            )
            all_projects.extend(projects)
            self.logger.info(f"Found {len(projects)} {visibility} projects")

        return all_projects

    def download_project_data(
        self, project: Dict, project_index: int, total_projects: int
    ):
        """Download all data for a single project"""
        project_id = project["id"]
        project_name = project["path_with_namespace"].replace("/", "_")

        self.progress.report_project_progress(
            project_index, total_projects, project["name"]
        )

        # Clone repository
        self._clone_repository(project, project_name, project_index, total_projects)

        # Download metadata
        self._download_metadata(project, project_name, project_id)

    def _clone_repository(
        self, project: Dict, project_name: str, project_index: int, total_projects: int
    ):
        """Handle repository cloning"""
        project_dir = os.path.join(self.config.output_dir, project_name)

        self.logger.info(f"Cloning {project['name']}...")

        # Get project size in MB
        size_mb = project.get("statistics", {}).get("repository_size", 0) / 1024 / 1024
        self.logger.info(f"Repository size: {size_mb:.2f} MB")

        if os.path.exists(project_dir):
            self.logger.info(
                f"Directory {project_dir} already exists, skipping clone..."
            )
            self.clone_stats.skipped += 1
        else:
            self._perform_clone(project, project_dir, project_name)

        # Report progress
        self.progress.report_clone_progress(
            project_index, total_projects, self.clone_stats
        )

    def _perform_clone(self, project: Dict, project_dir: str, project_name: str):
        """Perform the actual git clone operation"""
        os.makedirs(project_dir, exist_ok=True)
        project_url = project["http_url_to_repo"].replace(
            "https://", f"https://oauth2:{self.config.access_token}@"
        )

        try:
            start_time = time.time()
            subprocess.run(["git", "clone", project_url, project_dir], check=True)
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Successfully cloned {project['name']} in {elapsed_time:.1f} seconds"
            )
            self.clone_stats.successful += 1
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone {project['name']}: {e}")
            self.clone_stats.failed += 1
            self._log_failed_clone(project, project_name, project_url)

    def _log_failed_clone(self, project: Dict, project_name: str, project_url: str):
        """Log failed clone for later retry"""
        failed_clones_file = os.path.join(self.config.output_dir, "failed_clones.txt")
        with open(failed_clones_file, "a") as f:
            f.write(f"{project['id']},{project_name},{project_url}\n")

    def _download_metadata(self, project: Dict, project_name: str, project_id: int):
        """Download all metadata for a project"""
        metadata_dir = os.path.join(self.config.output_dir, f"{project_name}_metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Check if metadata already exists
        if self._metadata_exists(metadata_dir):
            self.logger.info(
                f"Metadata for {project['name']} already exists, skipping..."
            )
            return

        # Download different types of metadata
        self._download_issues(project_id, metadata_dir, project["name"])
        self._download_merge_requests(project_id, metadata_dir, project["name"])
        self._download_wikis(project, project_id, metadata_dir, project["name"])
        self._download_milestones(project_id, metadata_dir, project["name"])
        self._download_releases(project_id, metadata_dir, project["name"])
        self._download_ci_data(project_id, metadata_dir, project["name"])

    def _metadata_exists(self, metadata_dir: str) -> bool:
        """Check if essential metadata files already exist"""
        return os.path.exists(
            os.path.join(metadata_dir, "issues.json")
        ) and os.path.exists(os.path.join(metadata_dir, "merge_requests.json"))

    def _download_and_save(
        self,
        endpoint: str,
        filename: str,
        project_id: int,
        metadata_dir: str,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Generic method for downloading and saving JSON data"""
        url = f"{self.config.gitlab_url}/api/v4/projects/{project_id}/{endpoint}"
        data = self.api.get_all_pages(url, params)

        filepath = os.path.join(metadata_dir, f"{filename}.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return data

    def _download_issues(self, project_id: int, metadata_dir: str, project_name: str):
        """Download issues and their discussions"""
        self.logger.info(f"Downloading issues for {project_name}...")

        issues = self._download_and_save(
            "issues", "issues", project_id, metadata_dir, params={"state": "all"}
        )
        self.logger.info(f"Downloaded {len(issues)} issues for {project_name}")

        # Download discussions for each issue
        for issue in issues:
            issue_iid = issue["iid"]
            discussions = self._download_and_save(
                f"issues/{issue_iid}/discussions",
                f"issue_{issue_iid}_discussions",
                project_id,
                metadata_dir,
            )

    def _download_merge_requests(
        self, project_id: int, metadata_dir: str, project_name: str
    ):
        """Download merge requests and their discussions"""
        self.logger.info(f"Downloading merge requests for {project_name}...")

        merge_requests = self._download_and_save(
            "merge_requests",
            "merge_requests",
            project_id,
            metadata_dir,
            params={"state": "all"},
        )
        self.logger.info(
            f"Downloaded {len(merge_requests)} merge requests for {project_name}"
        )

        # Download discussions for each merge request
        for mr in merge_requests:
            mr_iid = mr["iid"]
            discussions = self._download_and_save(
                f"merge_requests/{mr_iid}/discussions",
                f"mr_{mr_iid}_discussions",
                project_id,
                metadata_dir,
            )

    def _download_wikis(
        self, project: Dict, project_id: int, metadata_dir: str, project_name: str
    ):
        """Download wiki content if enabled"""
        if not project.get("wiki_enabled", False):
            return

        self.logger.info(f"Downloading wiki content for {project_name}...")
        wiki_dir = os.path.join(metadata_dir, "wiki")
        os.makedirs(wiki_dir, exist_ok=True)

        try:
            # Get list of wiki pages
            url = f"{self.config.gitlab_url}/api/v4/projects/{project_id}/wikis"
            wiki_pages_response = self.api.get_single_resource(url)

            if not wiki_pages_response:
                self.logger.info(f"No wiki pages found for {project_name}")
                return

            wiki_pages = (
                wiki_pages_response if isinstance(wiki_pages_response, list) else []
            )

            if not wiki_pages:
                self.logger.info(f"No wiki pages found for {project_name}")
                return

            # Save wiki pages list
            with open(os.path.join(wiki_dir, "wiki_pages_list.json"), "w") as f:
                json.dump(wiki_pages, f, indent=2)

            # Download each wiki page
            successful_downloads = 0
            for page in wiki_pages:
                if self._download_wiki_page(project_id, wiki_dir, page):
                    successful_downloads += 1
                time.sleep(RATE_LIMIT_DELAY)

            self.logger.info(
                f"Downloaded {successful_downloads}/{len(wiki_pages)} wiki pages for {project_name}"
            )

        except Exception as e:
            self.logger.error(f"Error processing wikis for {project_name}: {str(e)}")

    def _download_wiki_page(self, project_id: int, wiki_dir: str, page: Dict) -> bool:
        """Download a single wiki page"""
        try:
            page_slug = page["slug"]
            url = f"{self.config.gitlab_url}/api/v4/projects/{project_id}/wikis/{page_slug}"
            page_content = self.api.get_single_resource(url)

            if page_content:
                # Save JSON
                with open(os.path.join(wiki_dir, f"{page_slug}.json"), "w") as f:
                    json.dump(page_content, f, indent=2)

                # Save markdown content
                content = page_content.get("content", "")
                with open(
                    os.path.join(wiki_dir, f"{page_slug}.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(content)

                return True

        except Exception as e:
            self.logger.warning(
                f"Error downloading wiki page {page.get('slug', 'unknown')}: {str(e)}"
            )

        return False

    def _download_milestones(
        self, project_id: int, metadata_dir: str, project_name: str
    ):
        """Download milestones"""
        self.logger.info(f"Downloading milestones for {project_name}...")
        milestones = self._download_and_save(
            "milestones", "milestones", project_id, metadata_dir
        )
        self.logger.info(f"Downloaded {len(milestones)} milestones for {project_name}")

    def _download_releases(self, project_id: int, metadata_dir: str, project_name: str):
        """Download releases"""
        self.logger.info(f"Downloading releases for {project_name}...")
        releases = self._download_and_save(
            "releases", "releases", project_id, metadata_dir
        )
        self.logger.info(f"Downloaded {len(releases)} releases for {project_name}")

    def _download_ci_data(self, project_id: int, metadata_dir: str, project_name: str):
        """Download CI/CD pipelines and their details"""
        self.logger.info(f"Downloading CI/CD pipelines for {project_name}...")

        pipelines = self._download_and_save(
            "pipelines", "pipelines", project_id, metadata_dir
        )
        self.logger.info(f"Downloaded {len(pipelines)} pipelines for {project_name}")

        # Get detailed data for recent pipelines
        for pipeline in pipelines[: self.config.pipeline_limit]:
            pipeline_id = pipeline["id"]

            # Get pipeline details
            pipeline_details = self.api.get_single_resource(
                f"{self.config.gitlab_url}/api/v4/projects/{project_id}/pipelines/{pipeline_id}"
            )
            if pipeline_details:
                filepath = os.path.join(
                    metadata_dir, f"pipeline_{pipeline_id}_details.json"
                )
                with open(filepath, "w") as f:
                    json.dump(pipeline_details, f, indent=2)

            # Get pipeline jobs
            jobs = self._download_and_save(
                f"pipelines/{pipeline_id}/jobs",
                f"pipeline_{pipeline_id}_jobs",
                project_id,
                metadata_dir,
            )

    def download_all(self, specific_project: Optional[str] = None):
        """Main method to download all project data"""
        # Get projects
        projects = self.get_projects(specific_project)
        if not projects:
            self.logger.error("No projects found to download")
            return

        self.logger.info(f"Found {len(projects)} projects")

        # Process each project
        for project_index, project in enumerate(projects, 1):
            self.download_project_data(project, project_index, len(projects))

        self.logger.info("Download completed successfully!")
        self.progress.report_final_summary(len(projects), self.clone_stats)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Download GitLab data and metadata")
    parser.add_argument(
        "--gitlab-url",
        type=str,
        required=True,
        help="GitLab instance URL (e.g., https://gitlab.example.com)",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        required=True,
        help="GitLab access token for API authentication",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/gitlab_data",
        help="Output directory for downloaded data (default: data/gitlab_data)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Specific project path with namespace to download (e.g., group/project)",
    )
    parser.add_argument(
        "--pipeline-limit",
        type=int,
        default=1,
        help="Number of most recent pipelines to download detailed data for",
    )
    args = parser.parse_args()

    # Create configuration
    config = DownloadConfig(
        gitlab_url=args.gitlab_url.rstrip("/"),
        access_token=args.access_token,
        output_dir=args.output_dir,
        pipeline_limit=args.pipeline_limit,
    )

    # Create downloader and run
    downloader = GitLabDownloader(config)
    downloader.download_all(args.project)


if __name__ == "__main__":
    main()
