"""Document loaders: CVs/resumes, LinkedIn, portfolios, projects, certifications,
professional experience, and GitHub."""
import csv
import os
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".csv"}

# sub-folder of data/ -> document type (stored in metadata["type"])
FOLDERS = {
    "cv": "cv",
    "resumes": "cv",
    "linkedin": "linkedin",
    "portfolio": "portfolio",
    "projects": "project",
    "certifications": "certification",
    "experience": "experience",
}

# LinkedIn data-export CSVs that contain other people's data: never ingest them
SKIP_CSV = {"connections", "messages", "invitations", "contacts", "recommendations_received"}


def _load_csv(path: Path, doc_type: str):
    """One Document per row (works for LinkedIn export files such as Positions.csv)."""
    if path.stem.lower() in SKIP_CSV:
        return []
    docs = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if k and v and v.strip())
            if text:
                docs.append(
                    Document(
                        page_content=f"{path.stem}\n{text}",
                        metadata={"source": path.name, "type": doc_type},
                    )
                )
    return docs


def _load_file(path: Path, doc_type: str):
    """Load one pdf/docx/txt/md/csv file and tag it with source + type metadata."""
    ext = path.suffix.lower()
    if ext == ".csv":
        return _load_csv(path, doc_type)
    if ext == ".pdf":
        pages = PyPDFLoader(str(path)).load()
    elif ext == ".docx":
        pages = Docx2txtLoader(str(path)).load()
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


def load_folder(folder: str, doc_type: str):
    """Recursively load every supported file inside a folder (missing folder = skipped)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    docs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                docs.extend(_load_file(path, doc_type))
            except Exception as e:
                print(f"[loaders] skipped {path.name}: {e}")
    return docs


def load_web(urls):
    """Load portfolio / personal website pages as plain text."""
    from bs4 import BeautifulSoup

    docs = []
    for url in urls or []:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "personal-rag"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            if text:
                docs.append(Document(page_content=text, metadata={"source": f"Web: {url}", "type": "portfolio"}))
        except Exception as e:
            print(f"[loaders] failed to load {url}: {e}")
    return docs


def load_github(user: str, repos=None):
    """Load profile + repo metadata + README for a GitHub user (public repos only).

    Set GITHUB_TOKEN in the environment to avoid API rate limits.
    """
    if not user:
        return []
    api = "https://api.github.com"
    headers = {}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

    docs = []
    try:
        # profile
        prof = requests.get(f"{api}/users/{user}", headers=headers, timeout=20).json()
        if isinstance(prof, dict) and "login" in prof:
            info = (
                f"GitHub profile: {prof.get('login')}\n"
                f"Name: {prof.get('name')}\n"
                f"Bio: {prof.get('bio')}\n"
                f"Company: {prof.get('company')}\n"
                f"Location: {prof.get('location')}\n"
                f"Website: {prof.get('blog')}\n"
                f"Public repos: {prof.get('public_repos')}\n"
            )
            docs.append(Document(page_content=info, metadata={"source": f"GitHub: {prof.get('html_url')}", "type": "github"}))

        # repositories (paginated)
        if repos:
            data = [requests.get(f"{api}/repos/{user}/{r}", headers=headers, timeout=20).json() for r in repos]
        else:
            data, page = [], 1
            while True:
                batch = requests.get(
                    f"{api}/users/{user}/repos?per_page=100&page={page}", headers=headers, timeout=20
                ).json()
                if not isinstance(batch, list):
                    print(f"[loaders] GitHub: {batch.get('message', batch)}")
                    break
                if not batch:
                    break
                data.extend(batch)
                page += 1
    except Exception as e:
        print(f"[loaders] GitHub request failed: {e}")
        return docs

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
            f"Fork: {repo.get('fork')}\n"
            f"Topics: {', '.join(repo.get('topics', []))}\n"
            f"Homepage: {repo.get('homepage')}\n"
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


def load_all(data_dir="data", github_user="", github_repos=None, urls=None):
    """Load everything under data/ (see FOLDERS), loose data/cv.* / data/resume.* files,
    portfolio URLs and GitHub."""
    data = Path(data_dir)
    docs = []

    # loose files in data/ named cv.* or resume.*
    if data.is_dir():
        for path in sorted(data.iterdir()):
            if path.is_file() and path.stem.lower() in ("cv", "resume") and path.suffix.lower() in SUPPORTED:
                try:
                    docs += _load_file(path, "cv")
                except Exception as e:
                    print(f"[loaders] skipped {path.name}: {e}")

    for folder, doc_type in FOLDERS.items():
        docs += load_folder(str(data / folder), doc_type)

    docs += load_web(urls)
    docs += load_github(github_user, github_repos)
    return docs
