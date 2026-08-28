import base64
import json
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
from github import Github, Auth
from langchain_core.documents import Document
from src.config import config

SUPPORTED_EXTENSIONS = (
    ".py", ".ipynb", ".js", ".jsx", ".ts", ".tsx", ".java",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh", ".bash", ".sql",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".md",
    ".rst", ".txt"
)

IGNORED_DIRS = {
    ".git", ".github", "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".idea", ".vscode", "dist",
    "build", ".mypy_cache", ".tox", ".eggs", "coverage", ".next"
}

IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "composer.lock", "flake.lock",
    ".DS_Store", "Thumbs.db"
}

IGNORED_FILE_SUFFIXES = (
    ".min.js", ".min.css", ".map", ".bundle.js", ".chunk.js",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot"
)

import re

def parse_github_url(repo_url: str) -> Tuple[str, str]:
    """Extracts owner and repo name from full GitHub URL with strict input sanitization."""
    clean_url = repo_url.strip().rstrip("/").replace("https://github.com/", "").replace("http://github.com/", "")
    parts = clean_url.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL format. Expected: https://github.com/owner/repo")
    
    owner, repo = parts[0].strip(), parts[1].strip()
    # Strip any trailing .git
    if repo.endswith(".git"):
        repo = repo[:-4]
        
    # Strict validation against GitHub valid naming characters (anti-SSRF / injection)
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", owner) or not re.match(r"^[a-zA-Z0-9_\-\.]+$", repo):
        raise ValueError("Invalid characters detected in repository owner or name.")
    return owner, repo

def parse_jupyter_notebook(raw_json_str: str) -> str:
    """
    Parses a Jupyter Notebook (.ipynb) JSON content into clean code and markdown text,
    stripping execution outputs, base64 images, and noisy metadata.
    """
    try:
        data = json.loads(raw_json_str)
        cells = data.get("cells", [])
        extracted_sections = []
        
        for idx, cell in enumerate(cells, 1):
            cell_type = cell.get("cell_type", "")
            source_content = cell.get("source", "")
            
            if isinstance(source_content, list):
                source_text = "".join(source_content).strip()
            else:
                source_text = str(source_content).strip()
                
            if not source_text:
                continue
                
            if cell_type == "code":
                extracted_sections.append(f"# [Notebook Cell {idx} - Code]\n{source_text}\n")
            elif cell_type == "markdown":
                extracted_sections.append(f"<!-- [Notebook Cell {idx} - Markdown] -->\n{source_text}\n")
            else:
                extracted_sections.append(f"# [Notebook Cell {idx} - {cell_type}]\n{source_text}\n")
                
        if extracted_sections:
            return "\n".join(extracted_sections)
    except Exception:
        pass
    
    # Fallback to raw content if parsing fails
    return raw_json_str

def is_ignored_path(path: str) -> bool:
    """Checks if the path or filename is inside an ignored directory or matches ignored files."""
    clean_path = path.replace("\\", "/")
    filename = clean_path.split("/")[-1]
    
    # Check ignored filenames and suffixes
    if filename in IGNORED_FILENAMES or filename.endswith(IGNORED_FILE_SUFFIXES):
        return True
        
    parts = clean_path.split("/")
    return any(p in IGNORED_DIRS or p.startswith(".") for p in parts[:-1])

def fetch_single_file(repo, file_path: str, html_url: str, owner: str, repo_name: str, sha: Optional[str] = None) -> Optional[Document]:
    """Fetches and processes a single file content from GitHub."""
    try:
        if sha:
            blob = repo.get_git_blob(sha)
            if blob.encoding == "base64":
                raw_data = base64.b64decode(blob.content).decode("utf-8", errors="replace")
            else:
                raw_data = blob.content
        else:
            fc = repo.get_contents(file_path)
            raw_data = fc.decoded_content.decode("utf-8", errors="replace")

        # Process Jupyter Notebooks cleanly
        if file_path.endswith(".ipynb"):
            processed_content = parse_jupyter_notebook(raw_data)
            doc_type = "notebook"
        elif file_path.endswith((".md", ".rst", ".txt")):
            processed_content = raw_data
            doc_type = "documentation"
        else:
            processed_content = raw_data
            doc_type = "code"

        return Document(
            page_content=processed_content,
            metadata={
                "source": file_path,
                "type": doc_type,
                "html_url": html_url,
                "repo": f"{owner}/{repo_name}"
            }
        )
    except Exception:
        return None

def resolve_repo_branch(repo, requested_branch: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Resolves valid branch name and commit SHA, automatically detecting default branch (main/master/etc)."""
    if requested_branch and requested_branch.strip():
        try:
            b = repo.get_branch(requested_branch.strip())
            return b.name, b.commit.sha
        except Exception:
            pass
            
    # Try repository default branch
    try:
        def_name = repo.default_branch
        if def_name:
            b = repo.get_branch(def_name)
            return b.name, b.commit.sha
    except Exception:
        pass
        
    # Try common defaults
    for name in ("main", "master", "trunk", "develop"):
        try:
            b = repo.get_branch(name)
            return b.name, b.commit.sha
        except Exception:
            continue
            
    return repo.default_branch or "main", None

def fetch_repository_data(
    repo_url: str,
    max_issues: int = 50,
    max_files: int = 300,
    branch: Optional[str] = None,
    allowed_extensions: Optional[Tuple[str, ...]] = None
) -> List[Document]:
    """
    Ingests source code, notebooks, and issues from a public/private GitHub repository.
    Uses Git Tree API and multi-threading for fast, scalable processing.
    """
    owner, repo_name = parse_github_url(repo_url)
    
    auth = Auth.Token(config.GITHUB_TOKEN) if config.GITHUB_TOKEN else None
    gh = Github(auth=auth) if auth else Github()
    
    repo = gh.get_repo(f"{owner}/{repo_name}")
    documents: List[Document] = []
    active_extensions = allowed_extensions or SUPPORTED_EXTENSIONS
    
    # Resolve exact working branch and tree SHA
    target_branch, commit_sha = resolve_repo_branch(repo, branch)
    tree_ref = commit_sha or target_branch
    
    # 1. Fetch Repository Code & Markdown Files via Git Tree API
    file_tasks = []
    try:
        git_tree = repo.get_git_tree(tree_ref, recursive=True)
        
        for item in git_tree.tree:
            if item.type == "blob":
                if item.path.endswith(active_extensions) and not is_ignored_path(item.path):
                    # Skip oversized single files (> 1.5MB)
                    if getattr(item, "size", 0) and item.size > 1_500_000:
                        continue
                    file_url = f"https://github.com/{owner}/{repo_name}/blob/{target_branch}/{item.path}"
                    file_tasks.append((item.path, file_url, item.sha))
                    if len(file_tasks) >= max_files:
                        break
    except Exception:
        # Fallback to get_contents directory crawl if Git Tree fails
        try:
            contents = repo.get_contents("", ref=target_branch)
        except Exception:
            contents = repo.get_contents("")
            
        while contents and len(file_tasks) < max_files:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                if not is_ignored_path(file_content.path):
                    try:
                        contents.extend(repo.get_contents(file_content.path, ref=target_branch))
                    except Exception:
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                        except Exception:
                            pass
            else:
                if file_content.path.endswith(active_extensions) and not is_ignored_path(file_content.path):
                    file_tasks.append((file_content.path, file_content.html_url, None))

    # Fetch file contents in parallel
    if file_tasks:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_file = {
                executor.submit(fetch_single_file, repo, path, url, owner, repo_name, sha): path
                for path, url, sha in file_tasks
            }
            for future in as_completed(future_to_file):
                doc = future.result()
                if doc and doc.page_content.strip():
                    documents.append(doc)
                    
    # 2. Fetch Issues and Resolutions
    if max_issues > 0:
        try:
            issues = list(itertools.islice(repo.get_issues(state="all"), max_issues))
            for issue in issues:
                if issue.pull_request:
                    continue  # Skip pull requests
                    
                issue_body = issue.body or "No description provided."
                try:
                    comments = [c.body for c in itertools.islice(issue.get_comments(), 3)]
                except Exception:
                    comments = []
                
                full_issue_text = f"Issue Title: {issue.title}\nState: {issue.state}\nBody:\n{issue_body}\n"
                if comments:
                    full_issue_text += "\nTop Comments/Resolutions:\n" + "\n---\n".join(filter(None, comments))
                    
                doc = Document(
                    page_content=full_issue_text,
                    metadata={
                        "source": f"issue-#{issue.number}",
                        "type": "issue",
                        "html_url": issue.html_url,
                        "title": issue.title,
                        "repo": f"{owner}/{repo_name}"
                    }
                )
                documents.append(doc)
        except Exception:
            pass
        
    return documents