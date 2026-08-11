"""
Validation helpers for CodeRAG.

Responsible for making sure the user-supplied input is actually a
usable, public GitHub repository URL before we try to clone it.
"""

import re
from dataclasses import dataclass
from typing import Optional

import requests

GITHUB_URL_PATTERN = re.compile(
    r"^https?://(www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(\.git)?/?$"
)


@dataclass
class RepoIdentity:
    owner: str
    repo: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"


class ValidationError(Exception):
    """Raised for any user-facing validation failure. `.user_message`
    is safe to show directly in the Streamlit UI."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


def parse_github_url(url: str) -> RepoIdentity:
    """Parse and structurally validate a GitHub repo URL.

    Raises ValidationError with a friendly message on failure.
    """
    if not url or not url.strip():
        raise ValidationError("❌ Please enter a valid public GitHub repository URL.")

    url = url.strip()
    match = GITHUB_URL_PATTERN.match(url)
    if not match:
        raise ValidationError("❌ Please enter a valid public GitHub repository URL.")

    owner = match.group("owner")
    repo = match.group("repo")

    # Guard against accidentally matching things like github.com/settings
    if not owner or not repo:
        raise ValidationError("❌ Please enter a valid public GitHub repository URL.")

    return RepoIdentity(owner=owner, repo=repo)


def check_repository_exists_and_public(
    identity: RepoIdentity, timeout: int = 10
) -> Optional[dict]:
    """Hit the GitHub REST API to confirm the repo exists and is public.

    Returns the parsed JSON repo metadata on success.
    Raises ValidationError for not-found / private / rate-limited cases.
    """
    api_url = f"https://api.github.com/repos/{identity.owner}/{identity.repo}"

    try:
        response = requests.get(
            api_url,
            timeout=timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
    except requests.RequestException:
        raise ValidationError(
            "❌ Could not reach GitHub right now. Please check your connection and try again."
        )

    if response.status_code == 404:
        raise ValidationError("❌ Repository could not be found.")

    if response.status_code == 403:
        raise ValidationError(
            "❌ GitHub API rate limit reached. Please try again in a few minutes."
        )

    if response.status_code != 200:
        raise ValidationError("❌ Repository could not be found.")

    data = response.json()

    if data.get("private", False):
        raise ValidationError("❌ Private repositories are not supported in this version.")

    return data
