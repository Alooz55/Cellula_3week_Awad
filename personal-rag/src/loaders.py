"""Document loaders: CV, local folders (projects / certifications), GitHub."""
import os
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

SUPPORTED = {".pdf", ".txt", ".md"}


def _load_file(path: Path, doc_type: str):
    """Load one pdf/txt/md file and tag every page with source + type metadata."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = PyPDFLoader(str(path)).load()
    elif ext in (".txt", ".md"):
        pages = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []
    for p in pages:
        label = path.name
        if "page" in p.metadata:
            label = f"{path.name} (page {p.metadata['page'] + 1})"
        p.metadata["source"] = label
        p.metadata["type"] = doc_type
    return pages


def load_cv(path: str):
    """Load the CV / resume (PDF, txt or md)."""
    p = Path(path)
    if not p.exists():
        print(f"[loaders] CV not found: {path}")
        return []
    return _load_file(p, "cv")


def load_folder(folder: str, doc_type: str):
    """Recursively load every supported file inside a folder."""
    root = Path(folder)
    if not root.is_dir():
        print(f"[loaders] folder not found: {folder}")
        return []
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                docs.extend(_load_file(path, doc_type))
            except Exception as e:
                print(f"[loaders] skipped {path.name}: {e}")
    return docs


def load_github(user: str, repos=None):
    """Load repo metadata + README for a GitHub user (public repos only).

    Set GITHUB_TOKEN in the environment to avoid API rate limits.
    """
    if not user:
        return []
    api = "https://api.github.com"
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        if repos:
            data = [requests.get(f"{api}/repos/{user}/{r}", headers=headers, timeout=20).json() for r in repos]
        else:
            data = requests.get(f"{api}/users/{user}/repos?per_page=100", headers=headers, timeout=20).json()
    except Exception as e:
        print(f"[loaders] GitHub request failed: {e}")
        return []

    docs = []
    for repo in data:
        if not isinstance(repo, dict) or "name" not in repo:
            continue
        name = repo["name"]
        url = repo.get("html_url", f"https://github.com/{user}/{name}")
        info = (
            f"GitHub repository: {name}\n"
            f"Description: {repo.get('description')}\n"
            f"Main language: {repo.get('language')}\n"
            f"Stars: {repo.get('stargazers_count')}\n"
            f"Topics: {', '.join(repo.get('topics', []))}\n"
        )
        docs.append(Document(page_content=info, metadata={"source": f"GitHub: {url}", "type": "github"}))
        try:
            r = requests.get(
                f"{api}/repos/{user}/{name}/readme",
                headers={**headers, "Accept": "application/vnd.github.raw"},
                timeout=20,
            )
            if r.status_code == 200 and r.text.strip():
                docs.append(
                    Document(page_content=r.text, metadata={"source": f"GitHub README: {url}", "type": "github"})
                )
        except Exception as e:
            print(f"[loaders] README failed for {name}: {e}")
    return docs


def load_all(data_dir="data", github_user="", github_repos=None):
    """Load everything: data/cv.pdf, data/projects/, data/certifications/, GitHub."""
    data = Path(data_dir)
    docs = []
    docs += load_cv(str(data / "cv.pdf"))
    docs += load_folder(str(data / "projects"), "project")
    docs += load_folder(str(data / "certifications"), "certification")
    docs += load_github(github_user, github_repos)
    return docs
