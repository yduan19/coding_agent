"""Change tracking and diff utilities.

This module tracks file contents before edits so the agent can generate
unified patches and open visual diffs in editors such as VS Code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict


@dataclass
class ChangeTracker:
    """Track original file state and provide diff helpers.

    Attributes:
        root: Absolute project root path.
        originals: Map of relative file path -> original file contents.
            `None` means the file did not exist before the first write.
    """

    root: Path
    originals: Dict[Path, str | None] = field(default_factory=dict)

    def record_original(self, rel_path: Path) -> None:
        """Store the initial file content once, before any write.

        Args:
            rel_path: Path relative to `root`.
        """
        rel_path = Path(rel_path)
        if rel_path in self.originals:
            return

        abs_path = self.root / rel_path
        if abs_path.exists() and abs_path.is_file():
            self.originals[rel_path] = abs_path.read_text(encoding="utf-8")
        else:
            self.originals[rel_path] = None

    def write_file(self, rel_path: Path, content: str) -> None:
        """Write a file while preserving original content for diffing.

        Args:
            rel_path: Path relative to `root`.
            content: New full file content.
        """
        rel_path = Path(rel_path)
        self.record_original(rel_path)
        abs_path = self.root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")

    def changed_files(self) -> list[Path]:
        """Return tracked files whose current content differs from original."""
        changed: list[Path] = []
        for rel_path, before in self.originals.items():
            abs_path = self.root / rel_path
            if before is None and abs_path.exists():
                changed.append(rel_path)
                continue
            if before is not None:
                after = abs_path.read_text(encoding="utf-8") if abs_path.exists() else None
                if after != before:
                    changed.append(rel_path)
        return sorted(changed)

    def unified_diff(self, rel_path: Path) -> str:
        """Create a unified diff for one tracked file.

        Args:
            rel_path: Path relative to `root`.

        Returns:
            Unified diff string. Empty string means no change.
        """
        rel_path = Path(rel_path)
        before = self.originals.get(rel_path)
        abs_path = self.root / rel_path
        after = abs_path.read_text(encoding="utf-8") if abs_path.exists() else None

        if before == after:
            return ""

        before_lines = [] if before is None else before.splitlines(keepends=True)
        after_lines = [] if after is None else after.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )

    def write_patch(self, output_file: Path) -> Path:
        """Write a combined unified patch for all changed files.

        Args:
            output_file: Where to save the patch.

        Returns:
            The absolute patch path.
        """
        output_file = output_file if output_file.is_absolute() else self.root / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        chunks = [self.unified_diff(p) for p in self.changed_files()]
        output_file.write_text("\n".join(c for c in chunks if c), encoding="utf-8")
        return output_file

    def open_vscode_diffs(self) -> tuple[bool, str]:
        """Open side-by-side diffs in VS Code for each changed file.

        Returns:
            Tuple of `(success, message)`.
        """
        if shutil.which("code") is None:
            return False, "VS Code CLI 'code' not found in PATH."

        changed = self.changed_files()
        if not changed:
            return True, "No changed files to diff."

        temp_dir = Path(tempfile.mkdtemp(prefix="liteagent-before-"))
        for rel_path in changed:
            before = self.originals.get(rel_path)
            before_file = temp_dir / rel_path
            before_file.parent.mkdir(parents=True, exist_ok=True)
            before_file.write_text(before or "", encoding="utf-8")
            after_file = self.root / rel_path
            subprocess.Popen(["code", "--diff", str(before_file), str(after_file)])

        return True, f"Opened {len(changed)} diff view(s) in VS Code."
