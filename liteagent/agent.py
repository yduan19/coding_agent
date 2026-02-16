"""LLM-driven coding loop.

The agent sends user instructions to either OpenAI (default) or Anthropic,
allows the model to call local tools, and stops when the model returns a
plain-text response.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from .changes import ChangeTracker
from .tools import ToolContext, ToolRunner

SYSTEM_PROMPT = """You are a focused coding agent operating on a local repository.

Rules:
- Inspect files before editing them.
- Prefer minimal, correct edits.
- Use `write_file` with complete updated file content.
- Run lightweight checks with `run_shell` when useful.
- When complete, reply with a concise summary of what changed.
"""


@dataclass
class AgentResult:
    """Outcome for one agent run."""

    summary: str
    changed_files: list[str]


class CodingAgent:
    """Minimal coding agent with OpenAI and Anthropic provider support."""

    def __init__(
        self,
        root: Path,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create an agent bound to a project root.

        Args:
            root: Repository directory where tools can read/write.
            model: Provider model name.
            provider: `openai` (default) or `anthropic`.
            api_key: Optional API key override. Falls back to environment.
            base_url: Optional provider-compatible API base URL.
        """
        provider = provider.lower()
        if provider not in {"openai", "anthropic"}:
            raise ValueError("provider must be 'openai' or 'anthropic'")

        self.root = root.resolve()
        self.model = model
        self.provider = provider
        self.changes = ChangeTracker(root=self.root)
        self.tools = ToolRunner(ToolContext(root=self.root, changes=self.changes))
        self.messages: list[dict[str, Any]] = []

        if provider == "openai":
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "openai package is required. Install dependencies with `pip install -e .`."
                ) from exc
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            return

        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "anthropic package is required. Install dependencies with `pip install -e .`."
            ) from exc
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    @staticmethod
    def _changed_files(changes: ChangeTracker) -> list[str]:
        return [p.as_posix() for p in changes.changed_files()]

    def _text_from_anthropic_blocks(self, blocks: list[Any]) -> str:
        """Extract plain text from Anthropic content blocks."""
        parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    def run(self, instruction: str, max_steps: int = 20) -> AgentResult:
        """Backward-compatible alias for `run_turn`."""
        return self.run_turn(instruction=instruction, max_steps=max_steps)

    def run_turn(
        self,
        instruction: str,
        max_steps: int = 20,
        stream: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> AgentResult:
        """Execute one instruction with iterative tool calling."""
        if self.provider == "openai":
            return self._run_turn_openai(
                instruction=instruction,
                max_steps=max_steps,
                stream=stream,
                on_text_delta=on_text_delta,
            )
        return self._run_turn_anthropic(instruction=instruction, max_steps=max_steps)

    def _run_turn_openai(
        self,
        instruction: str,
        max_steps: int,
        stream: bool,
        on_text_delta: Callable[[str], None] | None,
    ) -> AgentResult:
        """Execute one turn using OpenAI Chat Completions tool calling."""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Project root: {self.root}\n"
                    f"Task: {instruction}\n"
                    "Do the work now."
                ),
            }
        )

        for _ in range(max_steps):
            assistant_content = ""
            tool_calls: list[dict[str, Any]] = []

            if stream:
                streamed_tool_calls: dict[int, dict[str, Any]] = {}
                stream_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.messages],
                    tools=self.tools.tool_schemas("openai"),
                    tool_choice="auto",
                    stream=True,
                )
                for chunk in stream_resp:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        assistant_content += delta.content
                        if on_text_delta:
                            on_text_delta(delta.content)
                    for tc in delta.tool_calls or []:
                        call = streamed_tool_calls.setdefault(
                            tc.index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.id:
                            call["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                call["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                call["function"]["arguments"] += tc.function.arguments

                tool_calls = [streamed_tool_calls[i] for i in sorted(streamed_tool_calls)]
            else:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, *self.messages],
                    tools=self.tools.tool_schemas("openai"),
                    tool_choice="auto",
                )
                msg = resp.choices[0].message
                assistant_content = msg.content or ""
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]

            if not tool_calls:
                summary = (assistant_content or "Task completed.").strip()
                self.messages.append({"role": "assistant", "content": assistant_content})
                return AgentResult(summary=summary, changed_files=self._changed_files(self.changes))

            self.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )

            for idx, tc in enumerate(tool_calls):
                tool_id = tc.get("id") or f"tool_call_{idx}"
                function = tc.get("function", {})
                tool_name = function.get("name", "")
                tool_args = function.get("arguments", "{}")
                tool_output = self.tools.run(tool_name, tool_args)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_output,
                    }
                )

        return AgentResult(
            summary="Stopped after reaching max steps. Increase --max-steps if needed.",
            changed_files=self._changed_files(self.changes),
        )

    def _run_turn_anthropic(self, instruction: str, max_steps: int) -> AgentResult:
        """Execute one turn using Anthropic Messages tool use API."""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    f"Project root: {self.root}\n"
                    f"Task: {instruction}\n"
                    "Do the work now."
                ),
            }
        )

        for _ in range(max_steps):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    system=SYSTEM_PROMPT,
                    max_tokens=2048,
                    messages=self.messages,
                    tools=self.tools.tool_schemas("anthropic"),
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(self._format_anthropic_error(exc)) from exc

            blocks = list(resp.content)
            tool_uses = [b for b in blocks if getattr(b, "type", None) == "tool_use"]

            if not tool_uses:
                summary = self._text_from_anthropic_blocks(blocks) or "Task completed."
                self.messages.append({"role": "assistant", "content": blocks})
                return AgentResult(summary=summary, changed_files=self._changed_files(self.changes))

            self.messages.append({"role": "assistant", "content": blocks})

            tool_result_blocks: list[dict[str, Any]] = []
            for call in tool_uses:
                name = getattr(call, "name")
                tool_input = getattr(call, "input", {})
                call_id = getattr(call, "id")
                tool_output = self.tools.run(name, arguments_json=json.dumps(tool_input))
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": tool_output,
                    }
                )

            self.messages.append({"role": "user", "content": tool_result_blocks})

        return AgentResult(
            summary="Stopped after reaching max steps. Increase --max-steps if needed.",
            changed_files=self._changed_files(self.changes),
        )

    def _format_anthropic_error(self, err: Exception) -> str:
        """Build a user-friendly Anthropic error message with model hints."""
        error_type = type(err).__name__
        raw = str(err)
        if "NotFoundError" not in error_type and "not_found_error" not in raw:
            return f"Anthropic request failed: {raw}"

        available = self._anthropic_available_models()
        available_text = ", ".join(available) if available else "(unable to fetch model list)"
        return (
            f"Anthropic model not found: '{self.model}'. "
            f"Use --model with one of your available models: {available_text}"
        )

    def _anthropic_available_models(self, limit: int = 12) -> list[str]:
        """Best-effort lookup of model IDs available to the current Anthropic account."""
        try:
            page = self.client.models.list(limit=limit)
        except Exception:  # noqa: BLE001
            return []

        ids: list[str] = []
        data = getattr(page, "data", [])
        for item in data:
            model_id = getattr(item, "id", None)
            if isinstance(model_id, str):
                ids.append(model_id)
        return ids
