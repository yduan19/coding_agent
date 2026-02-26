"""Local file and shell tools exposed to the model.

The agent uses deterministic local tools via provider tool-calling APIs.
Each tool is intentionally small and explicit to keep behavior auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import hashlib
import json
import shutil
import os
from pathlib import Path
import subprocess
from typing import Any

from .changes import ChangeTracker


_IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules"}

_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_project_info",
        "description": "Get basic repository information such as root path and file count.",
        "schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_files",
        "description": "List project files, optionally filtered by glob pattern.",
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern like '*.py' or 'src/**/*.ts'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum files to return.",
                    "default": 200,
                },
            },
        },
    },
    {
        "name": "search_text",
        "description": "Search for text across project files.",
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text pattern to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter, e.g. '*.py' or 'tests/**/*.py'.",
                    "default": "*",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "default": False,
                },
                "max_results": {
                    "type": "integer",
                    "default": 200,
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": "Read text from a project file.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file_chunk",
        "description": "Read a line range from a UTF-8 text file for efficient large-file inspection.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {
                    "type": "integer",
                    "description": "1-indexed start line.",
                    "default": 1,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return.",
                    "default": 200,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write full content to a project file.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "append_file",
        "description": "Append text to a project file. Creates file if it does not exist.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "make_dir",
        "description": "Create a directory (and parent directories) under project root.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_exists",
        "description": "Check whether a file or directory exists.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "delete_path",
        "description": "Delete a file or directory under project root.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {
                    "type": "boolean",
                    "description": "Required for deleting non-empty directories.",
                    "default": False,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "move_path",
        "description": "Move or rename a file/directory within project root.",
        "schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "copy_path",
        "description": "Copy a file/directory within project root.",
        "schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "compute_file_hash",
        "description": "Compute a file hash to verify exact content state.",
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "description": "Hash algorithm name such as sha256, sha1, md5.",
                    "default": "sha256",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "update_plan",
        "description": "Update task plan status with optional explanation.",
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["step", "status"],
                    },
                },
            },
            "required": ["plan"],
        },
    },
    {
        "name": "get_plan",
        "description": "Read the current in-memory task plan.",
        "schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "run_shell",
        "description": "Run a non-interactive shell command in project root for validation.",
        "schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "default": 60},
            },
            "required": ["command"],
        },
    },
]


@dataclass
class ToolContext:
    """Execution context shared across all tool calls."""

    root: Path
    changes: ChangeTracker
    plan: dict[str, Any] | None = None
    current_turn_undo: dict[str, str | None] = field(default_factory=dict)
    undo_history: list[dict[str, str | None]] = field(default_factory=list)


class ToolRunner:
    """Dispatch function-call tool invocations to concrete Python code."""

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx


    def begin_turn(self) -> None:
        """Start collecting undo information for a new turn."""
        self.ctx.current_turn_undo = {}

    def commit_turn(self) -> None:
        """Persist undo data for the last completed turn if there were file mutations."""
        if self.ctx.current_turn_undo:
            self.ctx.undo_history.append(dict(self.ctx.current_turn_undo))
        self.ctx.current_turn_undo = {}

    def _record_undo(self, abs_path: Path) -> None:
        """Record pre-change file content once per path for current turn."""
        rel = abs_path.relative_to(self.ctx.root.resolve()).as_posix()
        if rel in self.ctx.current_turn_undo:
            return
        if abs_path.exists() and abs_path.is_file():
            self.ctx.current_turn_undo[rel] = abs_path.read_text(encoding="utf-8")
        else:
            self.ctx.current_turn_undo[rel] = None

    def revert_last_turn(self) -> list[str]:
        """Revert file mutations from the most recent committed turn."""
        if not self.ctx.undo_history:
            return []

        snapshot = self.ctx.undo_history.pop()
        reverted = sorted(snapshot.keys())
        for rel_str, before in snapshot.items():
            abs_path = (self.ctx.root / rel_str).resolve()
            if before is None:
                if abs_path.exists():
                    if abs_path.is_dir():
                        shutil.rmtree(abs_path)
                    else:
                        abs_path.unlink()
            else:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(before, encoding="utf-8")
        return reverted

    @staticmethod
    def tool_schemas(provider: str) -> list[dict[str, Any]]:
        """Return provider-specific tool schemas."""
        if provider == "openai":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["schema"],
                    },
                }
                for spec in _TOOL_SPECS
            ]

        if provider == "anthropic":
            return [
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "input_schema": spec["schema"],
                }
                for spec in _TOOL_SPECS
            ]

        raise ValueError(f"Unsupported provider: {provider}")

    def run(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name from serialized JSON arguments."""
        args = json.loads(arguments_json or "{}")

        handlers = {
            "get_project_info": self.get_project_info,
            "list_files": self.list_files,
            "search_text": self.search_text,
            "read_file": self.read_file,
            "read_file_chunk": self.read_file_chunk,
            "write_file": self.write_file,
            "append_file": self.append_file,
            "make_dir": self.make_dir,
            "file_exists": self.file_exists,
            "delete_path": self.delete_path,
            "move_path": self.move_path,
            "copy_path": self.copy_path,
            "compute_file_hash": self.compute_file_hash,
            "update_plan": self.update_plan,
            "get_plan": self.get_plan,
            "run_shell": self.run_shell,
        }
        if name not in handlers:
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"})

        try:
            result = handlers[name](**args)
            return json.dumps({"ok": True, "result": result})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": str(exc)})

    def get_project_info(self) -> dict[str, Any]:
        """Return basic information about the current project root."""
        file_count = 0
        for current_dir, dirs, files in os.walk(self.ctx.root, topdown=True):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            file_count += len(files)
        return {
            "root": str(self.ctx.root),
            "file_count": file_count,
        }

    def list_files(self, pattern: str = "*", max_results: int = 200) -> list[str]:
        """List files under root.

        Args:
            pattern: Glob pattern matched against relative path.
            max_results: Maximum number of files in output.
        """
        out: list[str] = []
        for current_dir, dirs, files in os.walk(self.ctx.root, topdown=True):
            dirs[:] = [d for d in dirs if d not in _IGNORE_DIRS]
            base = Path(current_dir)
            for filename in files:
                rel = (base / filename).relative_to(self.ctx.root)
                rel_str = rel.as_posix()
                if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(filename, pattern):
                    out.append(rel_str)
                if len(out) >= max_results:
                    return out
        return out

    def search_text(
        self,
        pattern: str,
        glob: str = "*",
        case_sensitive: bool = False,
        max_results: int = 200,
    ) -> list[dict[str, Any]]:
        """Search for a text pattern in project files.

        Returns:
            List of match entries with file path, line number and line text.
        """
        needle = pattern if case_sensitive else pattern.lower()
        results: list[dict[str, Any]] = []

        for rel_str in self.list_files(pattern=glob, max_results=20000):
            abs_path = self.ctx.root / rel_str
            try:
                text = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            for idx, line in enumerate(text.splitlines(), start=1):
                hay = line if case_sensitive else line.lower()
                if needle in hay:
                    results.append(
                        {
                            "path": rel_str,
                            "line_number": idx,
                            "line": line[:500],
                        }
                    )
                if len(results) >= max_results:
                    return results

        return results

    def _resolve_path(self, path: str) -> tuple[Path, Path]:
        """Resolve and validate a project-relative path."""
        rel = Path(path)
        abs_path = (self.ctx.root / rel).resolve()
        root = self.ctx.root.resolve()
        if root not in abs_path.parents and abs_path != root:
            raise ValueError("Path escapes project root")
        return rel, abs_path

    def read_file(self, path: str) -> str:
        """Read UTF-8 text from a relative file path."""
        _, abs_path = self._resolve_path(path)
        return abs_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        """Write a full file and register change tracking."""
        _, abs_path = self._resolve_path(path)

        rel_norm = abs_path.relative_to(self.ctx.root.resolve())
        self._record_undo(abs_path)
        self.ctx.changes.write_file(rel_norm, content)
        return f"Wrote {rel_norm.as_posix()} ({len(content)} chars)"

    def read_file_chunk(self, path: str, start_line: int = 1, max_lines: int = 200) -> dict[str, Any]:
        """Read a chunk of a text file by line range."""
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")

        _, abs_path = self._resolve_path(path)
        text = abs_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        start_idx = start_line - 1
        end_idx = min(start_idx + max_lines, len(lines))
        chunk_lines = lines[start_idx:end_idx]

        return {
            "path": path,
            "start_line": start_line,
            "end_line": end_idx,
            "total_lines": len(lines),
            "content": "\n".join(chunk_lines),
        }

    def append_file(self, path: str, content: str) -> str:
        """Append content to a file while preserving change tracking."""
        _, abs_path = self._resolve_path(path)
        prior = abs_path.read_text(encoding="utf-8") if abs_path.exists() else ""
        self.write_file(path=path, content=prior + content)
        return f"Appended {len(content)} chars to {path}"

    def make_dir(self, path: str) -> str:
        """Create a directory and its parents under project root."""
        _, abs_path = self._resolve_path(path)
        abs_path.mkdir(parents=True, exist_ok=True)
        return f"Created directory {abs_path.relative_to(self.ctx.root.resolve()).as_posix()}"

    def file_exists(self, path: str) -> dict[str, bool]:
        """Check if a path exists and whether it is a file or directory."""
        _, abs_path = self._resolve_path(path)
        return {
            "exists": abs_path.exists(),
            "is_file": abs_path.is_file(),
            "is_dir": abs_path.is_dir(),
        }

    def delete_path(self, path: str, recursive: bool = False) -> str:
        """Delete a file or directory in project root."""
        _, abs_path = self._resolve_path(path)
        if not abs_path.exists():
            return f"Path not found: {path}"

        if abs_path.is_file() or abs_path.is_symlink():
            rel_norm = abs_path.relative_to(self.ctx.root.resolve())
            self.ctx.changes.record_original(rel_norm)
            self._record_undo(abs_path)
            abs_path.unlink()
            return f"Deleted file {path}"

        if not recursive and any(abs_path.iterdir()):
            raise ValueError("Directory is not empty; set recursive=true to delete.")

        for child in abs_path.rglob("*"):
            if child.is_file() or child.is_symlink():
                rel_norm = child.relative_to(self.ctx.root.resolve())
                self.ctx.changes.record_original(rel_norm)
                self._record_undo(child)

        abs_path.rmdir() if not recursive else shutil.rmtree(abs_path)
        return f"Deleted directory {path}"

    def move_path(self, src: str, dst: str) -> str:
        """Move/rename a file or directory in project root."""
        _, src_abs = self._resolve_path(src)
        _, dst_abs = self._resolve_path(dst)
        if not src_abs.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        self._record_undo(src_abs)
        self._record_undo(dst_abs)
        shutil.move(str(src_abs), str(dst_abs))
        return f"Moved {src} -> {dst}"

    def copy_path(self, src: str, dst: str) -> str:
        """Copy a file or directory in project root."""
        _, src_abs = self._resolve_path(src)
        _, dst_abs = self._resolve_path(dst)
        if not src_abs.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dst_abs.exists():
            raise ValueError("Destination already exists")
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        self._record_undo(dst_abs)
        if src_abs.is_dir():
            shutil.copytree(src_abs, dst_abs)
        else:
            shutil.copy2(src_abs, dst_abs)
        return f"Copied {src} -> {dst}"

    def compute_file_hash(self, path: str, algorithm: str = "sha256") -> dict[str, str]:
        """Compute digest for a file using hashlib."""
        _, abs_path = self._resolve_path(path)
        if not abs_path.is_file():
            raise ValueError("Path must be a file")

        hasher = hashlib.new(algorithm)
        with abs_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return {
            "path": path,
            "algorithm": algorithm,
            "digest": hasher.hexdigest(),
        }

    def update_plan(self, plan: list[dict[str, str]], explanation: str = "") -> dict[str, Any]:
        """Store/update an in-memory task plan for visibility and progress tracking."""
        in_progress_count = sum(1 for item in plan if item.get("status") == "in_progress")
        if in_progress_count > 1:
            raise ValueError("At most one plan item can be in_progress")
        self.ctx.plan = {
            "explanation": explanation,
            "plan": plan,
        }
        return self.ctx.plan

    def get_plan(self) -> dict[str, Any]:
        """Return the current plan if one has been set."""
        return self.ctx.plan or {"explanation": "", "plan": []}

    def run_shell(self, command: str, timeout_seconds: int = 60) -> dict[str, Any]:
        """Execute a shell command for checks like tests or linters."""
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=str(self.ctx.root),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-6000:],
            "stderr": completed.stderr[-6000:],
        }
