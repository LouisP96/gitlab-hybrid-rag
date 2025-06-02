from setuptools import setup, find_packages

setup(
    name="gitlab-rag",
    version="0.1.0",
    description="GitLab RAG System",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "accelerate==1.7.0",
        "anthropic==0.52.0",
        "faiss_gpu==1.7.2",
        "Flask==3.1.1",
        "flask_cors==5.0.1",
        "nltk==3.9.1",
        "numpy==1.26.4",
        "rank_bm25==0.2.2",
        "Requests==2.32.3",
        "sentence_transformers==4.0.2",
        "torch==2.6.0",
        "transformers==4.51.0",
        "Werkzeug==3.1.3",
    ],
)