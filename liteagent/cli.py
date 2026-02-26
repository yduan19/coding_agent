"""Command-line interface for liteagent.

Examples:
    liteagent
    liteagent chat --provider anthropic
    liteagent run "Add unit tests for parser module" --open-diff

Azure OpenAI example:
    export AZURE_OPENAI_API_KEY="..."
    liteagent run "hello" \
      --provider azure_openai \
      --base-url "https://<resource>.openai.azure.com/" \
      --azure-api-version "2024-02-15-preview" \
      --model "<deployment_name>"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
import json

from .agent import CodingAgent

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "azure_openai": "gpt-4.1-mini",  # interpreted as deployment name
    "anthropic": "claude-sonnet-4-20250514",
}
MODEL_CHOICES = {
    "openai": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
    ],
    "azure_openai": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-5-sonnet-latest",
    ],
}


def _add_common_options(cmd: argparse.ArgumentParser) -> None:
    """Attach shared run/chat flags."""
    cmd.add_argument("--root", default=".", help="Project root directory.")
    cmd.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["openai", "azure_openai", "anthropic"],
        help="LLM provider.",
    )
    cmd.add_argument("--model", default=None, help="Model name for selected provider.")
    cmd.add_argument("--api-key", default=None, help="Optional API key override.")
    cmd.add_argument(
        "--base-url",
        default=None,
        help=(
            "Optional provider-compatible API base URL. "
            "For azure_openai, this must be your Azure endpoint like https://<resource>.openai.azure.com/"
        ),
    )
    cmd.add_argument(
        "--azure-api-version",
        default=None,
        help="Azure OpenAI api-version (only used when --provider azure_openai).",
    )
    cmd.add_argument("--max-steps", type=int, default=20, help="Maximum tool-call rounds.")
    cmd.add_argument(
        "--patch-out",
        default=".liteagent/last.patch",
        help="Where to write unified diff patch after run.",
    )
    cmd.add_argument(
        "--open-diff",
        action="store_true",
        help="Open side-by-side diffs in VS Code after changes.",
    )
    cmd.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream assistant text output live while generating.",
    )
    cmd.add_argument(
        "--auto-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run a lightweight verification command after file changes.",
    )
    cmd.add_argument(
        "--verify-command",
        default=None,
        help="Override automatic verification command (runs in project root).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="liteagent",
        description="Lightweight Python coding agent using OpenAI by default, with optional Anthropic Claude and Azure OpenAI support.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    run_cmd = sub.add_parser("run", help="Run one instruction through the coding agent.")
    run_cmd.add_argument("instruction", help="Instruction to execute.")
    _add_common_options(run_cmd)

    chat_cmd = sub.add_parser("chat", help="Start interactive coding session.")
    _add_common_options(chat_cmd)

    return parser


def _load_api_keys() -> dict[str, str | None]:
    """Load provider API keys from system environment."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "azure_openai": os.getenv("AZURE_OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    }


def _resolve_api_key(provider: str, explicit_api_key: str | None, env_keys: dict[str, str | None]) -> str:
    """Resolve API key for selected provider or raise a clear error."""
    if explicit_api_key:
        return explicit_api_key

    env_key_name = {
        "openai": "OPENAI_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }[provider]

    key = env_keys.get(provider)
    if key:
        return key

    raise SystemExit(
        f"No API key detected for provider '{provider}'. "
        f"Set {env_key_name} in your environment or pass --api-key."
    )


def _select_model_interactive(provider: str) -> str:
    """Prompt the user to choose a model for interactive sessions."""
    models = MODEL_CHOICES[provider]
    default_model = DEFAULT_MODELS[provider]

    print(f"\nSelect {provider} model:")
    for idx, name in enumerate(models, start=1):
        default_tag = " (default)" if name == default_model else ""
        print(f"{idx}. {name}{default_tag}")
    print(f"{len(models) + 1}. custom")

    raw = input("Model [press Enter for default]: ").strip()
    if not raw:
        return default_model

    if raw.isdigit():
        pos = int(raw)
        if 1 <= pos <= len(models):
            return models[pos - 1]
        if pos == len(models) + 1:
            custom = input("Enter custom model id: ").strip()
            if custom:
                return custom
            return default_model

    return raw


def _create_agent(
    args: argparse.Namespace,
    interactive_select_model: bool,
    env_keys: dict[str, str | None],
) -> CodingAgent:
    """Build agent from parsed CLI args."""
    provider = args.provider
    api_key = _resolve_api_key(provider=provider, explicit_api_key=args.api_key, env_keys=env_keys)

    model = args.model
    if not model:
        model = _select_model_interactive(provider) if interactive_select_model else DEFAULT_MODELS[provider]

    root = Path(args.root).resolve()
    return CodingAgent(
        root=root,
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=args.base_url,
        azure_api_version=getattr(args, "azure_api_version", None),
    )




def _verification_command(changed_files: list[str], explicit: str | None) -> str | None:
    """Return a lightweight verification command based on changed files."""
    if explicit:
        return explicit

    py_changed = [f for f in changed_files if f.endswith(".py")]
    if py_changed:
        quoted = " ".join(shlex.quote(f) for f in py_changed)
        return f"python -m py_compile {quoted}"

    return None


def _run_auto_verify(agent: CodingAgent, changed_files: list[str], args: argparse.Namespace) -> None:
    """Run best-effort post-change verification and print outcome."""
    if not args.auto_verify or not changed_files:
        return

    command = _verification_command(changed_files=changed_files, explicit=args.verify_command)
    if not command:
        print("No automatic verification command inferred for changed files.")
        return

    print("\n=== Verification ===")
    print(f"$ {command}")
    result_json = agent.tools.run("run_shell", json.dumps({"command": command, "timeout_seconds": 120}))
    result = json.loads(result_json)
    if not result.get("ok"):
        print(f"Verification failed to run: {result.get('error')}")
        return

    payload = result.get("result", {})
    if payload.get("stdout"):
        print(payload["stdout"].rstrip())
    if payload.get("stderr"):
        print(payload["stderr"].rstrip())

    exit_code = payload.get("exit_code", 1)
    if exit_code == 0:
        print("Verification passed.")
    else:
        print(f"Verification failed with exit code {exit_code}.")

def _finalize_turn(agent: CodingAgent, patch_out: str, open_diff: bool) -> None:
    """Write patch and optionally open visual diffs."""
    patch_path = agent.changes.write_patch(Path(patch_out))
    print(f"Patch written to: {patch_path}")

    if open_diff:
        ok, message = agent.changes.open_vscode_diffs()
        print(message)
        if not ok:
            raise SystemExit(2)


def _run_once(args: argparse.Namespace, env_keys: dict[str, str | None]) -> None:
    """Handle one-shot execution mode."""
    agent = _create_agent(args, interactive_select_model=False, env_keys=env_keys)
    if args.stream and agent.provider == "anthropic":
        print("Streaming is currently supported for OpenAI/Azure OpenAI provider; Anthropic uses buffered output.")
    if args.stream:
        print("\nagent> ", end="", flush=True)
    try:
        result = agent.run_turn(
            instruction=args.instruction,
            max_steps=args.max_steps,
            stream=args.stream,
            on_text_delta=(lambda s: print(s, end="", flush=True)) if args.stream else None,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.stream:
        print()

    if not args.stream:
        print("\n=== Agent Summary ===")
        print(result.summary)
    print("\n=== Changed Files ===")
    if result.changed_files:
        for fpath in result.changed_files:
            print(f"- {fpath}")
    else:
        print("(none)")

    _run_auto_verify(agent, changed_files=result.changed_files, args=args)
    _finalize_turn(agent, patch_out=args.patch_out, open_diff=args.open_diff)


def _chat_loop(args: argparse.Namespace, env_keys: dict[str, str | None]) -> None:
    """Handle interactive multi-turn terminal session."""
    agent = _create_agent(args, interactive_select_model=True, env_keys=env_keys)

    print("\nInteractive session started")
    print(f"Root: {agent.root}")
    print(f"Provider: {agent.provider}")
    print(f"Model: {agent.model}")
    if args.stream and agent.provider == "anthropic":
        print("Streaming is currently supported for OpenAI/Azure OpenAI provider; Anthropic uses buffered output.")
    print("Type /exit to quit. Type /diff to open current VS Code diffs. Type /revert to undo last turn changes.\n")

    while True:
        try:
            instruction = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if not instruction:
            continue

        if instruction in {"/exit", "exit", "quit"}:
            print("Session ended.")
            break

        if instruction == "/diff":
            ok, message = agent.changes.open_vscode_diffs()
            print(message)
            continue

        if instruction == "/revert":
            reverted = agent.revert_last_turn()
            if not reverted:
                print("No recent turn changes to revert.")
                continue
            print("Reverted files from latest turn:")
            for rel in reverted:
                print(f"- {rel}")
            _finalize_turn(agent, patch_out=args.patch_out, open_diff=args.open_diff)
            continue

        if args.stream:
            print("\nagent> ", end="", flush=True)
        try:
            result = agent.run_turn(
                instruction=instruction,
                max_steps=args.max_steps,
                stream=args.stream,
                on_text_delta=(lambda s: print(s, end="", flush=True)) if args.stream else None,
            )
        except RuntimeError as exc:
            print(f"error> {exc}")
            continue
        if args.stream:
            print("\n")
        else:
            print(f"\nagent> {result.summary}\n")

        changed = result.changed_files
        if changed:
            print("Changed files:")
            for fpath in changed:
                print(f"- {fpath}")
            _run_auto_verify(agent, changed_files=changed, args=args)
            _finalize_turn(agent, patch_out=args.patch_out, open_diff=args.open_diff)
        else:
            print("No file changes.")


def main() -> None:
    """Entry point for the `liteagent` command."""
    parser = build_parser()
    env_keys = _load_api_keys()

    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] in {"run", "chat"}:
        args = parser.parse_args(raw_args)
    elif raw_args and raw_args[0] in {"-h", "--help"}:
        args = parser.parse_args(raw_args)
    else:
        args = parser.parse_args(["chat", *raw_args])
    command = args.command

    if command == "run":
        _run_once(args, env_keys=env_keys)
        return

    if command == "chat":
        _chat_loop(args, env_keys=env_keys)
        return

    raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
