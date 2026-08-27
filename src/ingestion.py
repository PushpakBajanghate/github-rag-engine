import base64
import itertools
from typing import List, Tuple
from github import Github, Auth
from langchain_core.documents import Document
from src.config import config

SUPPORTED_EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".go", ".md")

def parse_github_url(repo_url: str) -> Tuple[str, str]:
    """Extracts owner and repo name from full GitHub URL."""
    clean_url = repo_url.rstrip("/").replace("https://github.com/", "")
    parts = clean_url.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL format. Expected: https://github.com/owner/repo")
    return parts[0], parts[1]

def fetch_repository_data(repo_url: str, max_issues: int = 30) -> List[Document]:
    """
    Ingests source code and issues from a public/private GitHub repository.
    """
    owner, repo_name = parse_github_url(repo_url)
    
    auth = Auth.Token(config.GITHUB_TOKEN) if config.GITHUB_TOKEN else None
    gh = Github(auth=auth) if auth else Github()
    
    repo = gh.get_repo(f"{owner}/{repo_name}")
    documents: List[Document] = []
    
    # 1. Fetch Repository Code & Markdown Files
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            # Skip hidden and cache directories
            if not file_content.path.startswith(".") and not "test" in file_content.path:
                contents.extend(repo.get_contents(file_content.path))
        else:
            if file_content.path.endswith(SUPPORTED_EXTENSIONS):
                try:
                    raw_data = file_content.decoded_content.decode("utf-8")
                    doc = Document(
                        page_content=raw_data,
                        metadata={
                            "source": file_content.path,
                            "type": "code" if not file_content.path.endswith(".md") else "documentation",
                            "html_url": file_content.html_url,
                            "repo": f"{owner}/{repo_name}"
                        }
                    )
                    documents.append(doc)
                except Exception:
                    continue  # Skip un-decodable binary files
                    
    # 2. Fetch Issues and Resolutions
    issues = itertools.islice(repo.get_issues(state="all"), max_issues)
    for issue in issues:
        if issue.pull_request:
            continue  # Skip pull requests
            
        issue_body = issue.body or "No description provided."
        comments = [c.body for c in itertools.islice(issue.get_comments(), 3)]
        
        full_issue_text = f"Issue Title: {issue.title}\nState: {issue.state}\nBody:\n{issue_body}\n"
        if comments:
            full_issue_text += "\nTop Comments/Resolutions:\n" + "\n---\n".join(comments)
            
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
        
    return documents