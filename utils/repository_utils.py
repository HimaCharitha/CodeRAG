"""
Small helpers shared across the ingestion pipeline: turning a repo
identity + commit sha into a stable collection name for ChromaDB, and
sanitizing file paths so nothing outside the temp clone can be touched.
"""

import hashlib
import os
import re


def make_collection_name(owner: str, repo: str, commit_sha: str = "") -> str:
    """Build a deterministic, Chroma-safe collection name.

    Including the commit sha means a repo that has since been updated
    gets a fresh collection instead of silently reusing stale chunks,
    while an unchanged repo hits the cache.
    """
    raw = f"{owner}/{repo}@{commit_sha or 'unknown'}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    safe_owner = re.sub(r"[^a-zA-Z0-9_-]", "-", owner.lower())
    safe_repo = re.sub(r"[^a-zA-Z0-9_-]", "-", repo.lower())

    name = f"{safe_owner}-{safe_repo}-{digest}"
    # Chroma collection names must be 3-63 chars.
    return name[:63]


def sanitize_relative_path(root_dir: str, absolute_path: str) -> str:
    """Return a path relative to root_dir, guaranteed not to escape it.

    Raises ValueError if the resolved path is outside root_dir, which
    would indicate a symlink or path traversal attempt in the
    (untrusted) cloned repository.
    """
    real_root = os.path.realpath(root_dir)
    real_path = os.path.realpath(absolute_path)

    if not real_path.startswith(real_root + os.sep) and real_path != real_root:
        raise ValueError(f"Path escapes repository root: {absolute_path}")

    return os.path.relpath(real_path, real_root).replace(os.sep, "/")


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
