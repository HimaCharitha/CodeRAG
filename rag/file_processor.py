"""
Walks a cloned repository, applies the file filters, and reads each
surviving file into memory as a ProcessedFile ready for chunking.
"""

import os
from dataclasses import dataclass
from typing import List

from utils.file_filters import iter_candidate_files
from utils.repository_utils import sanitize_relative_path


@dataclass
class ProcessedFile:
    relative_path: str
    absolute_path: str
    file_type: str      # extension without the dot, e.g. "py"
    content: str
    line_count: int


def _read_text_file(path: str) -> str:
    """Best-effort text read; skips files that turn out to be binary."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                text = f.read()
            # Heuristic: lots of NUL bytes means this wasn't really text.
            if "\x00" in text:
                return ""
            return text
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def process_repository(root_dir: str) -> List[ProcessedFile]:
    processed: List[ProcessedFile] = []

    for absolute_path in iter_candidate_files(root_dir):
        try:
            relative_path = sanitize_relative_path(root_dir, absolute_path)
        except ValueError:
            # Path traversal / symlink escape — skip, treat repo as untrusted.
            continue

        content = _read_text_file(absolute_path)
        if not content or not content.strip():
            continue

        _, ext = os.path.splitext(relative_path)
        file_type = ext.lstrip(".").lower() or "text"

        processed.append(
            ProcessedFile(
                relative_path=relative_path,
                absolute_path=absolute_path,
                file_type=file_type,
                content=content,
                line_count=content.count("\n") + 1,
            )
        )

    return processed
