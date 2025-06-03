#!/usr/bin/env python3
import requests
import os
import subprocess
import time
import json
import argparse
import logging


def get_all_pages(url, params=None, headers=None, access_token=None):
    """Helper function to get all pages of a paginated API response"""
    if params is None:
        params = {}
    if headers is None:
        headers = {}

    headers["PRIVATE-TOKEN"] = access_token

    all_items = []
    page = 1
    per_page = 100

    while True:
        page_params = {**params, "page": page, "per_page": per_page}
        response = requests.get(url, params=page_params, headers=headers)

        if response.status_code != 200:
            logging.error(
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
        time.sleep(0.5)  # Rate limiting to be nice to the server

    return all_items


def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Download GitLab data and metadata")
    parser.add_argument(
        "--gitlab_url",
        type=str,
        required=True,
        help="GitLab instance URL (e.g., https://gitlab.example.com)",
    )
    parser.add_argument(
        "--access_token",
        type=str,
        required=True,
        help="GitLab access token for API authentication",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="gitlab_data",
        help="Output directory for downloaded data (default: gitlab_data)",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Specific project path with namespace to download (e.g., group/project)",
    )
    parser.add_argument(
        "--clone_repo",
        action="store_true",
        help="Clone the repository as well (default: skip cloning)",
    )
    args = parser.parse_args()

    # Configuration from arguments
    gitlab_url = args.gitlab_url.rstrip("/")
    access_token = args.access_token
    output_directory = args.output_directory

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_directory, "download_log.txt")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Initialize counters for repo cloning
    successful_clones = 0
    failed_clones = 0
    skipped_clones = 0

    # Get projects based on command line arguments
    all_projects = []

    if args.project:
        # If a specific project is requested, get only that project
        logger.info(f"Fetching specific project: {args.project}")
        response = requests.get(
            f"{gitlab_url}/api/v4/projects/{args.project.replace('/', '%2F')}",
            headers={"PRIVATE-TOKEN": access_token},
        )

        if response.status_code == 200:
            project = response.json()
            logger.info(f"Found project: {project['name']} (ID: {project['id']})")
            all_projects.append(project)
        else:
            logger.info(f"Direct lookup failed with status: {response.status_code}")

        if not all_projects:
            logger.error(
                f"Project '{args.project}' not found. Please check the path format."
            )
            logger.error(
                "The format should be exactly as shown in GitLab, e.g., 'group/project'"
            )
            exit(1)
    else:
        # Otherwise get all projects as before
        logger.info("Fetching all projects")
        page = 1
        per_page = 100

        # Valid visibility values are: public, internal, or private
        for visibility in ["public", "internal"]:
            page = 1

            while True:
                response = requests.get(
                    f"{gitlab_url}/api/v4/projects",
                    params={
                        "page": page,
                        "per_page": per_page,
                        "visibility": visibility,
                    },
                    headers={"PRIVATE-TOKEN": access_token},
                )

                if response.status_code != 200:
                    logger.error(
                        f"Error for {visibility} projects: {response.status_code}"
                    )
                    logger.error(response.text)
                    break

                projects = response.json()
                if not projects:
                    break

                all_projects.extend(projects)
                page += 1
                time.sleep(0.5)  # Rate limiting to be nice to the server

    logger.info(f"Found {len(all_projects)} projects")

    # Save all projects metadata
    with open(os.path.join(output_directory, "all_projects.json"), "w") as f:
        json.dump(all_projects, f, indent=2)

    # Process each project
    total_projects = len(all_projects)
    for project_index, project in enumerate(all_projects, 1):
        project_id = project["id"]
        project_name = project["path_with_namespace"].replace("/", "_")
        metadata_dir = os.path.join(output_directory, f"{project_name}_metadata")

        logger.info(
            f"Processing project: {project['name']} (ID: {project_id}) - {project_index}/{total_projects} ({(project_index / total_projects) * 100:.1f}%)"
        )

        # Create metadata directory
        os.makedirs(metadata_dir, exist_ok=True)

        # Handle repository cloning if requested
        if args.clone_repo:
            project_dir = os.path.join(output_directory, project_name)
            logger.info(
                f"[{project_index}/{total_projects}] {(project_index / total_projects) * 100:.1f}% - Cloning {project['name']}..."
            )

            # Get project size in MB
            size_mb = (
                project.get("statistics", {}).get("repository_size", 0) / 1024 / 1024
            )
            logger.info(f"Repository size: {size_mb:.2f} MB")

            if os.path.exists(project_dir):
                logger.info(
                    f"Directory {project_dir} already exists, skipping clone..."
                )
                skipped_clones += 1
            else:
                os.makedirs(project_dir, exist_ok=True)
                project_url = project["http_url_to_repo"].replace(
                    "https://", f"https://oauth2:{access_token}@"
                )

                try:
                    start_time = time.time()
                    subprocess.run(
                        ["git", "clone", project_url, project_dir], check=True
                    )
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Successfully cloned {project['name']} in {elapsed_time:.1f} seconds"
                    )
                    successful_clones += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to clone {project['name']}: {e}")
                    failed_clones += 1

                    # Write failed clones to a file for later retry
                    with open(
                        os.path.join(output_directory, "failed_clones.txt"), "a"
                    ) as f:
                        f.write(f"{project['id']},{project_name},{project_url}\n")

            # Print summary every 10 projects if cloning multiple projects
            if total_projects > 1 and (project_index % 10 == 0):
                logger.info("\n--- CLONING PROGRESS SUMMARY ---")
                logger.info(
                    f"Completed: {project_index}/{total_projects} ({project_index / total_projects * 100:.1f}%)"
                )
                logger.info(f"Successful: {successful_clones}")
                logger.info(f"Failed: {failed_clones}")
                logger.info(f"Skipped: {skipped_clones}")
                logger.info(f"Remaining: {total_projects - project_index}")
                logger.info("-----------------------\n")
        else:
            logger.info(f"Skipping repository clone for {project['name']}")

        # Check if metadata already exists for this project
        if os.path.exists(os.path.join(metadata_dir, "issues.json")) and os.path.exists(
            os.path.join(metadata_dir, "merge_requests.json")
        ):
            logger.info(f"Metadata for {project['name']} already exists, skipping...")
            continue

        # 1. Get issues
        logger.info(f"Downloading issues for {project['name']}...")
        issues = get_all_pages(
            f"{gitlab_url}/api/v4/projects/{project_id}/issues",
            params={"state": "all"},
            access_token=access_token,
        )

        with open(os.path.join(metadata_dir, "issues.json"), "w") as f:
            json.dump(issues, f, indent=2)
        logger.info(f"Downloaded {len(issues)} issues for {project['name']}")

        # Get issue discussions for each issue
        for issue in issues:
            issue_iid = issue["iid"]
            discussions = get_all_pages(
                f"{gitlab_url}/api/v4/projects/{project_id}/issues/{issue_iid}/discussions",
                access_token=access_token,
            )

            if discussions:
                with open(
                    os.path.join(metadata_dir, f"issue_{issue_iid}_discussions.json"),
                    "w",
                ) as f:
                    json.dump(discussions, f, indent=2)

        # 2. Get merge requests
        logger.info(f"Downloading merge requests for {project['name']}...")
        merge_requests = get_all_pages(
            f"{gitlab_url}/api/v4/projects/{project_id}/merge_requests",
            params={"state": "all"},
            access_token=access_token,
        )

        with open(os.path.join(metadata_dir, "merge_requests.json"), "w") as f:
            json.dump(merge_requests, f, indent=2)
        logger.info(
            f"Downloaded {len(merge_requests)} merge requests for {project['name']}"
        )

        # Get MR discussions for each merge request
        for mr in merge_requests:
            mr_iid = mr["iid"]
            discussions = get_all_pages(
                f"{gitlab_url}/api/v4/projects/{project_id}/merge_requests/{mr_iid}/discussions",
                access_token=access_token,
            )

            if discussions:
                with open(
                    os.path.join(metadata_dir, f"mr_{mr_iid}_discussions.json"), "w"
                ) as f:
                    json.dump(discussions, f, indent=2)

        # 3. Get wiki content via API with better error handling and timeouts
        if project.get("wiki_enabled", False):
            logger.info(f"Downloading wiki content for {project['name']}...")
            wiki_dir = os.path.join(metadata_dir, "wiki")
            os.makedirs(wiki_dir, exist_ok=True)

            try:
                # Get list of all wiki pages with timeout
                wiki_pages = []
                try:
                    # First try with a short timeout to catch hanging requests
                    response = requests.get(
                        f"{gitlab_url}/api/v4/projects/{project_id}/wikis",
                        headers={"PRIVATE-TOKEN": access_token},
                        timeout=30,  # 30 second timeout
                    )

                    if response.status_code == 200:
                        wiki_pages = response.json()
                    else:
                        logger.warning(
                            f"Failed to get wiki pages for {project['name']}: {response.status_code} - {response.text}"
                        )
                except requests.exceptions.Timeout:
                    logger.warning(
                        f"Timeout getting wiki pages for {project['name']}, skipping wiki"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error getting wiki pages for {project['name']}: {str(e)}"
                    )

                if wiki_pages:
                    with open(os.path.join(wiki_dir, "wiki_pages_list.json"), "w") as f:
                        json.dump(wiki_pages, f, indent=2)

                    # Download content of each wiki page with timeout and error handling
                    successful_downloads = 0
                    for page in wiki_pages:
                        try:
                            page_slug = page["slug"]
                            response = requests.get(
                                f"{gitlab_url}/api/v4/projects/{project_id}/wikis/{page_slug}",
                                headers={"PRIVATE-TOKEN": access_token},
                                timeout=30,  # 30 second timeout
                            )

                            if response.status_code == 200:
                                page_content = response.json()

                                with open(
                                    os.path.join(wiki_dir, f"{page_slug}.json"), "w"
                                ) as f:
                                    json.dump(page_content, f, indent=2)

                                # Also save the raw content for easier access
                                content = page_content.get("content", "")
                                with open(
                                    os.path.join(wiki_dir, f"{page_slug}.md"),
                                    "w",
                                    encoding="utf-8",
                                ) as f:
                                    f.write(content)

                                successful_downloads += 1
                            else:
                                logger.warning(
                                    f"Failed to get wiki page {page_slug}: {response.status_code}"
                                )
                        except requests.exceptions.Timeout:
                            logger.warning(
                                f"Timeout getting wiki page {page_slug} for {project['name']}, skipping"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error getting wiki page {page_slug} for {project['name']}: {str(e)}"
                            )

                        # Add a brief pause between requests
                        time.sleep(0.5)

                    logger.info(
                        f"Downloaded {successful_downloads}/{len(wiki_pages)} wiki pages for {project['name']}"
                    )
                else:
                    logger.info(f"No wiki pages found for {project['name']}")
            except Exception as e:
                logger.error(f"Error processing wikis for {project['name']}: {str(e)}")
                # Continue with the next project

        # 4. Get milestones
        logger.info(f"Downloading milestones for {project['name']}...")
        milestones = get_all_pages(
            f"{gitlab_url}/api/v4/projects/{project_id}/milestones",
            access_token=access_token,
        )

        with open(os.path.join(metadata_dir, "milestones.json"), "w") as f:
            json.dump(milestones, f, indent=2)
        logger.info(f"Downloaded {len(milestones)} milestones for {project['name']}")

        # 5. Get releases
        logger.info(f"Downloading releases for {project['name']}...")
        releases = get_all_pages(
            f"{gitlab_url}/api/v4/projects/{project_id}/releases",
            access_token=access_token,
        )

        with open(os.path.join(metadata_dir, "releases.json"), "w") as f:
            json.dump(releases, f, indent=2)
        logger.info(f"Downloaded {len(releases)} releases for {project['name']}")

        # 6. Get CI/CD pipelines
        logger.info(f"Downloading CI/CD pipelines for {project['name']}...")
        pipelines = get_all_pages(
            f"{gitlab_url}/api/v4/projects/{project_id}/pipelines",
            access_token=access_token,
        )

        with open(os.path.join(metadata_dir, "pipelines.json"), "w") as f:
            json.dump(pipelines, f, indent=2)
        logger.info(f"Downloaded {len(pipelines)} pipelines for {project['name']}")

        # Get pipeline details for each pipeline (limiting to most recent x to avoid too many API calls)
        for pipeline in pipelines[:5]:
            pipeline_id = pipeline["id"]
            pipeline_details = requests.get(
                f"{gitlab_url}/api/v4/projects/{project_id}/pipelines/{pipeline_id}",
                headers={"PRIVATE-TOKEN": access_token},
            ).json()

            with open(
                os.path.join(metadata_dir, f"pipeline_{pipeline_id}_details.json"), "w"
            ) as f:
                json.dump(pipeline_details, f, indent=2)

            # Get pipeline jobs
            jobs = get_all_pages(
                f"{gitlab_url}/api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs",
                access_token=access_token,
            )

            if jobs:
                with open(
                    os.path.join(metadata_dir, f"pipeline_{pipeline_id}_jobs.json"), "w"
                ) as f:
                    json.dump(jobs, f, indent=2)

    logger.info("Download completed successfully!")

    # Final summary for repository cloning if applicable
    if args.clone_repo:
        logger.info("\n=== FINAL CLONING SUMMARY ===")
        logger.info(f"Total projects: {total_projects}")
        logger.info(f"Successful clones: {successful_clones}")
        logger.info(f"Failed clones: {failed_clones}")
        logger.info(f"Skipped clones: {skipped_clones}")

        if failed_clones > 0:
            logger.info(
                "Some clones failed. Check 'failed_clones.txt' to retry them later."
            )


if __name__ == "__main__":
    main()
