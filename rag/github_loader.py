"""
Repository ingestion: clones a public GitHub repository into a
temporary directory and reports basic metadata (commit sha, size).

We shell out to GitPython for the clone and never execute anything
from inside the cloned tree.
"""

import os
import shutil
import tempfile
from dataclasses import dataclass

from git import Repo, GitCommandError

from utils.validation import RepoIdentity, ValidationError
from utils.file_filters import MAX_TOTAL_FILES, MAX_TOTAL_SIZE_BYTES


@dataclass
class ClonedRepository:
    identity: RepoIdentity
    local_path: str
    commit_sha: str

    def cleanup(self):
        shutil.rmtree(self.local_path, ignore_errors=True)


def clone_repository(identity: RepoIdentity) -> ClonedRepository:
    """Shallow-clone the repo into a fresh temp dir.

    Raises ValidationError on failure (not found, too large, network
    issue, etc.) so the caller can show a clean message in the UI.
    """
    temp_dir = tempfile.mkdtemp(prefix="coderag_")

    try:
        repo = Repo.clone_from(
            identity.clone_url,
            temp_dir,
            depth=1,             # shallow clone: history isn't needed for RAG
            single_branch=True,
            no_tags=True,
        )
    except GitCommandError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValidationError("❌ Repository could not be found.")
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValidationError(
            "❌ Something went wrong while fetching the repository. Please try again."
        )

    try:
        commit_sha = repo.head.commit.hexsha
    except Exception:
        commit_sha = "unknown"

    _enforce_size_limits(temp_dir)

    return ClonedRepository(identity=identity, local_path=temp_dir, commit_sha=commit_sha)


def _enforce_size_limits(root_dir: str):
    """Bail out early if the repo is clearly too large to process,
    before we spend time chunking/embedding it."""
    total_files = 0
    total_size = 0

    for current_dir, subdirs, files in os.walk(root_dir):
        subdirs[:] = [d for d in subdirs if d != ".git"]
        for filename in files:
            total_files += 1
            try:
                total_size += os.path.getsize(os.path.join(current_dir, filename))
            except OSError:
                pass

            if total_files > MAX_TOTAL_FILES or total_size > MAX_TOTAL_SIZE_BYTES:
                shutil.rmtree(root_dir, ignore_errors=True)
                raise ValidationError(
                    "❌ This repository is too large to process in this version "
                    f"(limit: {MAX_TOTAL_FILES} files / "
                    f"{MAX_TOTAL_SIZE_BYTES // (1024 * 1024)} MB)."
                )
