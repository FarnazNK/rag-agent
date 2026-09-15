from __future__ import annotations

import re
from pathlib import Path

from rag_agent.agent.models import ContextFile

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

DEFAULT_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "vendor",
    }
)

DEFAULT_TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".dart",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)

_ALWAYS_TEXT_FILENAMES = frozenset(
    {
        "Dockerfile",
        "Gemfile",
        "Makefile",
        "Procfile",
        "Rakefile",
    }
)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(value) if len(token) > 1}


class RepositoryContextBuilder:
    """Select a compact set of repository files relevant to a developer task."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        max_file_bytes: int = 200_000,
        max_context_chars: int = 40_000,
        ignored_dirs: frozenset[str] = DEFAULT_IGNORED_DIRS,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.repo_root}")
        self.max_file_bytes = max_file_bytes
        self.max_context_chars = max_context_chars
        self.ignored_dirs = ignored_dirs

    def resolve_path(self, relative_path: str | Path) -> Path:
        candidate = (self.repo_root / relative_path).resolve()
        if not candidate.is_relative_to(self.repo_root):
            raise ValueError("Path escapes the repository root.")
        return candidate

    def iter_candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo_root)
            if any(part in self.ignored_dirs for part in relative.parts):
                continue
            is_supported = (
                path.name in _ALWAYS_TEXT_FILENAMES
                or path.suffix.lower() in DEFAULT_TEXT_SUFFIXES
            )
            if not is_supported:
                continue
            try:
                if path.stat().st_size > self.max_file_bytes:
                    continue
            except OSError:
                continue
            candidates.append(path)
        return sorted(candidates)

    def _read(self, path: Path) -> tuple[str, bool]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        max_chars = min(self.max_file_bytes, self.max_context_chars)
        if len(raw) <= max_chars:
            return raw, False
        return raw[:max_chars], True

    def build(self, task: str, *, limit: int = 12) -> list[ContextFile]:
        query_tokens = _tokens(task)
        ranked: list[ContextFile] = []

        for path in self.iter_candidate_paths():
            relative = path.relative_to(self.repo_root).as_posix()
            try:
                content, truncated = self._read(path)
            except OSError:
                continue

            path_tokens = _tokens(relative.replace("/", " "))
            content_tokens = _tokens(content[:20_000])
            path_overlap = len(query_tokens & path_tokens)
            content_overlap = len(query_tokens & content_tokens)

            score = float(path_overlap * 4 + content_overlap)
            if path.name.lower() in {"readme.md", "pyproject.toml", "package.json", "gemfile"}:
                score += 0.25
            if score <= 0:
                continue

            ranked.append(
                ContextFile(
                    path=relative,
                    score=score,
                    content=content,
                    truncated=truncated,
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.path))

        selected: list[ContextFile] = []
        used_chars = 0
        for item in ranked:
            if len(selected) >= limit:
                break
            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break
            if len(item.content) > remaining:
                item = item.model_copy(
                    update={
                        "content": item.content[:remaining],
                        "truncated": True,
                    }
                )
            selected.append(item)
            used_chars += len(item.content)

        return selected
