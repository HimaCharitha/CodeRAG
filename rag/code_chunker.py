"""
Code-aware chunking.

Strategy:
  * Python files: parse with `ast` and cut chunks at function/class
    boundaries, so each chunk is a semantically complete unit.
  * C-family languages (JS/TS/Java/C/C++/C#/Go/etc.): a brace-depth
    regex scan that finds top-level function/class/method blocks.
  * Everything else (README, config, markup, plain text): a
    LangChain RecursiveCharacterTextSplitter, with Markdown-aware
    separators for .md files.

Every chunk carries the metadata required by the spec: repository,
file_path, file_type, start_line, end_line, chunk_id.
"""

import ast
import re
import uuid
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.file_processor import ProcessedFile

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

C_FAMILY_EXTENSIONS = {
    "js", "jsx", "ts", "tsx", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "php",
}

FUNCTION_LIKE_PATTERN = re.compile(
    r"^[ \t]*(export\s+)?(public|private|protected|static|async|final|abstract)?\s*"
    r"(function\s+\w+|class\s+\w+|\w[\w<>\[\], ]*\s+\w+\s*\([^;{]*\)\s*\{|"
    r"func\s+\w+|def\s+\w+)",
    re.MULTILINE,
)


@dataclass
class CodeChunk:
    chunk_id: str
    repository: str
    file_path: str
    file_type: str
    start_line: int
    end_line: int
    content: str
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "repository": self.repository,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_id": self.chunk_id,
        }


def chunk_file(processed_file: ProcessedFile, repository: str) -> List[CodeChunk]:
    if processed_file.file_type == "py":
        chunks = _chunk_python(processed_file, repository)
    elif processed_file.file_type in C_FAMILY_EXTENSIONS:
        chunks = _chunk_c_family(processed_file, repository)
    else:
        chunks = _chunk_generic(processed_file, repository)

    # Safety net: if a structural strategy produced nothing useful
    # (e.g. a Python file that fails to parse), fall back to generic.
    if not chunks:
        chunks = _chunk_generic(processed_file, repository)

    return chunks


def _make_chunk(repository, file_path, file_type, start_line, end_line, content) -> CodeChunk:
    return CodeChunk(
        chunk_id=str(uuid.uuid4()),
        repository=repository,
        file_path=file_path,
        file_type=file_type,
        start_line=start_line,
        end_line=end_line,
        content=content.strip(),
    )


def _chunk_python(pf: ProcessedFile, repository: str) -> List[CodeChunk]:
    try:
        tree = ast.parse(pf.content)
    except SyntaxError:
        return []

    lines = pf.content.splitlines()
    chunks: List[CodeChunk] = []
    top_level_nodes = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    if not top_level_nodes:
        return []

    covered_ranges = []
    for node in top_level_nodes:
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        covered_ranges.append((start, end))
        snippet = "\n".join(lines[start - 1:end])
        if not snippet.strip():
            continue
        chunks.append(
            _make_chunk(repository, pf.relative_path, pf.file_type, start, end, snippet)
        )

    # Capture module-level code (imports, constants, docstring) that
    # sits outside any function/class, as its own chunk if non-trivial.
    covered_lines = set()
    for start, end in covered_ranges:
        covered_lines.update(range(start, end + 1))

    leftover_lines = [
        (i + 1, line) for i, line in enumerate(lines) if (i + 1) not in covered_lines
    ]
    leftover_text = "\n".join(line for _, line in leftover_lines).strip()
    if leftover_text and len(leftover_text) > 40:
        first_ln = leftover_lines[0][0] if leftover_lines else 1
        last_ln = leftover_lines[-1][0] if leftover_lines else len(lines)
        chunks.append(
            _make_chunk(
                repository, pf.relative_path, pf.file_type, first_ln, last_ln, leftover_text
            )
        )

    return _split_oversized(chunks, repository, pf)


def _chunk_c_family(pf: ProcessedFile, repository: str) -> List[CodeChunk]:
    lines = pf.content.splitlines()
    matches = list(FUNCTION_LIKE_PATTERN.finditer(pf.content))
    if not matches:
        return []

    # Map character offsets to line numbers.
    offsets = []
    running = 0
    for line in lines:
        offsets.append(running)
        running += len(line) + 1

    def offset_to_line(offset: int) -> int:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    starts = sorted({offset_to_line(m.start()) for m in matches})
    chunks: List[CodeChunk] = []
    for i, start_line in enumerate(starts):
        end_line = (starts[i + 1] - 1) if i + 1 < len(starts) else len(lines)
        snippet = "\n".join(lines[start_line - 1:end_line])
        if not snippet.strip():
            continue
        chunks.append(
            _make_chunk(repository, pf.relative_path, pf.file_type, start_line, end_line, snippet)
        )

    return _split_oversized(chunks, repository, pf)


def _chunk_generic(pf: ProcessedFile, repository: str) -> List[CodeChunk]:
    if pf.file_type == "md":
        separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
    else:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
    )

    pieces = splitter.split_text(pf.content)
    chunks: List[CodeChunk] = []

    # Approximate line ranges by locating each piece's first line in
    # the original content (best-effort, fine for display purposes).
    search_from = 0
    lines = pf.content.splitlines()
    for piece in pieces:
        if not piece.strip():
            continue
        piece_first_line = piece.splitlines()[0].strip() if piece.splitlines() else ""
        start_line = 1
        for idx in range(search_from, len(lines)):
            if piece_first_line and piece_first_line in lines[idx]:
                start_line = idx + 1
                search_from = idx
                break
        end_line = min(start_line + piece.count("\n"), pf.line_count)
        chunks.append(
            _make_chunk(repository, pf.relative_path, pf.file_type, start_line, end_line, piece)
        )

    return chunks


def _split_oversized(chunks: List[CodeChunk], repository: str, pf: ProcessedFile) -> List[CodeChunk]:
    """A structurally-found chunk (e.g. a huge function) can still be
    too large for the embedding model's effective context. Re-split
    any oversized chunk with the generic splitter while keeping the
    original line offset for metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    result: List[CodeChunk] = []
    for chunk in chunks:
        if len(chunk.content) <= CHUNK_SIZE * 1.5:
            result.append(chunk)
            continue
        sub_pieces = splitter.split_text(chunk.content)
        span = max(chunk.end_line - chunk.start_line, 1)
        for i, sub in enumerate(sub_pieces):
            if not sub.strip():
                continue
            approx_start = chunk.start_line + int(span * i / max(len(sub_pieces), 1))
            approx_end = chunk.start_line + int(span * (i + 1) / max(len(sub_pieces), 1))
            result.append(
                _make_chunk(
                    repository, pf.relative_path, pf.file_type,
                    approx_start, max(approx_end, approx_start), sub
                )
            )
    return result


def chunk_repository(processed_files, repository: str) -> List[CodeChunk]:
    all_chunks: List[CodeChunk] = []
    for pf in processed_files:
        all_chunks.extend(chunk_file(pf, repository))
    return all_chunks
