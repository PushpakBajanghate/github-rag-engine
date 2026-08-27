"""GitHub API client & file/issue extractor."""

import os
import re
from typing import List, Optional, Dict, Any
from github import Github, Auth
from langchain_core.documents import Document
from src.config import settings

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".kt", ".scala", ".sh", ".sql", ".html", ".css", ".json",
    ".yaml", ".yml", ".toml", ".md", ".rst", ".txt"
}

IGNORE_DIRS = {
    ".git", ".github", "node_modules", "dist", "build", ".venv", "venv",
    "__pycache__", ".next", ".idea", ".vscode", "vendor", "target"
}


class GitHubIngestion:
    """GitHub API extractor for repository code, documentation, and issues."""

    def __init__(self, token: Optional[str] = None):
        auth_token = token or settings.github_token
        if auth_token:
            auth = Auth.Token(auth_token)
            self.client = Github(auth=auth)
        else:
            self.client = Github()

    @staticmethod
    def parse_repo_url(repo_input: str) -> str:
        cleaned = repo_input.strip().rstrip("/")
        match = re.search(r"github\.com[/:]([^/]+/[^/\s.]+)(?:\.git)?", cleaned)
        if match:
            return match.group(1)
        if "/" in cleaned and len(cleaned.split("/")) == 2:
            return cleaned
        raise ValueError(f"Invalid GitHub repository format: {repo_input}")

    def fetch_repo_files(
        self,
        repo_name: str,
        branch: Optional[str] = None,
        max_files: int = 150
    ) -> List[Document]:
        target_repo = self.parse_repo_url(repo_name)
        repo = self.client.get_repo(target_repo)
        target_branch = branch or repo.default_branch

        git_tree = repo.get_git_tree(sha=target_branch, recursive=True)
        documents: List[Document] = []
        files_indexed = 0

        for item in git_tree.tree:
            if files_indexed >= max_files:
                break

            if item.type != "blob":
                continue

            path_parts = item.path.split("/")
            if any(ignored in path_parts for ignored in IGNORE_DIRS):
                continue

            _, ext = os.path.splitext(item.path)
            if ext.lower() not in CODE_EXTENSIONS:
                continue

            try:
                content_file = repo.get_contents(item.path, ref=target_branch)
                if content_file.encoding == "base64":
                    content = content_file.decoded_content.decode("utf-8", errors="replace")
                else:
                    content = content_file.content or ""

                if not content.strip() or "\x00" in content:
                    continue

                html_url = f"{repo.html_url}/blob/{target_branch}/{item.path}"
                metadata: Dict[str, Any] = {
                    "source": item.path,
                    "file_path": item.path,
                    "file_name": path_parts[-1],
                    "extension": ext.lower(),
                    "repo": target_repo,
                    "branch": target_branch,
                    "html_url": html_url,
                    "type": "code" if ext.lower() != ".md" else "documentation"
                }

                documents.append(Document(page_content=content, metadata=metadata))
                files_indexed += 1
            except Exception as e:
                print(f"Warning: unable to fetch {item.path}: {e}")
                continue

        return documents

    def fetch_repo_issues(
        self,
        repo_name: str,
        state: str = "open",
        max_issues: int = 30
    ) -> List[Document]:
        target_repo = self.parse_repo_url(repo_name)
        repo = self.client.get_repo(target_repo)
        issues = repo.get_issues(state=state)

        documents: List[Document] = []
        for count, issue in enumerate(issues):
            if count >= max_issues:
                break
            if issue.pull_request:
                continue

            body = issue.body or ""
            content = f"Issue #{issue.number}: {issue.title}\nState: {issue.state}\n\n{body}"

            metadata: Dict[str, Any] = {
                "source": f"Issue #{issue.number}",
                "file_path": f"issues/{issue.number}",
                "file_name": f"issue_{issue.number}.md",
                "extension": ".md",
                "repo": target_repo,
                "branch": "issues",
                "html_url": issue.html_url,
                "type": "issue",
                "issue_number": issue.number
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents
