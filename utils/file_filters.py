"""
File filtering rules for CodeRAG.

Decides which files in a cloned repository are worth reading, chunking
and embedding, and which should be ignored outright.
"""

import os

# Directories we never descend into.
IGNORED_DIR_NAMES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "target",
    ".next",
    ".nuxt",
    "vendor",
    "site-packages",
}

# Extensions we treat as text/code worth indexing.
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".rs",
    ".html",
    ".css",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".sql",
    ".txt",
}

# Filenames that are never indexed even if their extension looks fine,
# because they commonly hold secrets or are pure lockfile noise.
BLOCKED_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    "credentials.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_rsa.pub",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
}

# Substrings in a filename that flag it as likely to hold credentials.
BLOCKED_NAME_HINTS = ("secret", "credential", "password", "apikey", "api_key", ".pem", ".key")

# Hard safety caps.
MAX_FILE_SIZE_BYTES = 512 * 1024          # skip any single file over 512 KB
MAX_TOTAL_FILES = 2000                    # repository-level file count cap
MAX_TOTAL_SIZE_BYTES = 150 * 1024 * 1024  # 150 MB repository-level cap


def is_ignored_dir(dirname: str) -> bool:
    return dirname in IGNORED_DIR_NAMES or dirname.startswith(".")


def is_supported_file(filename: str) -> bool:
    if filename in BLOCKED_FILENAMES:
        return False

    lower_name = filename.lower()
    if any(hint in lower_name for hint in BLOCKED_NAME_HINTS):
        return False

    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_EXTENSIONS


def iter_candidate_files(root_dir: str):
    """Yield absolute paths of files under root_dir that pass filtering,
    pruning ignored directories as we walk for efficiency."""
    for current_dir, subdirs, files in os.walk(root_dir):
        subdirs[:] = [d for d in subdirs if not is_ignored_dir(d)]

        for filename in files:
            if not is_supported_file(filename):
                continue
            full_path = os.path.join(current_dir, filename)
            try:
                if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
                    continue
                if os.path.getsize(full_path) == 0:
                    continue
            except OSError:
                continue
            yield full_path
