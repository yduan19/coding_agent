# liteagent

A lightweight coding agent written entirely in Python for learning purposes.
It runs in your terminal, edits files in the current folder, and shows diffs in
VS Code.

## Usage

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Default provider: OpenAI
export OPENAI_API_KEY="your_openai_key"

# Optional provider: Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Interactive mode (default)

```bash
liteagent
```

- Defaults to `openai` provider.
- Uses current terminal directory as project root.
- Streaming is on by default.
- Prompts for model selection unless `--model` is provided.

Interactive commands:

- `/diff` open current VS Code diffs
- `/exit` quit session

### One-shot mode

```bash
liteagent run "refactor parser to use dataclass"
```

### Provider selection

```bash
# OpenAI (default)
liteagent --provider openai --model gpt-5.2

# Anthropic
liteagent --provider anthropic --model claude-sonnet-4-5-20250929
```

### Streaming

```bash
# default behavior
liteagent run "add type hints"

# explicit streaming
liteagent run "add type hints" --stream

# disable streaming
liteagent run "add type hints" --no-stream
```

### Common flags

- `--provider`: `openai` (default) or `anthropic`
- `--model`: model id for the selected provider
- `--root`: project root (default `.`)
- `--max-steps`: max tool-call rounds per turn (default `20`)
- `--patch-out`: patch output file (default `.liteagent/last.patch`)
- `--open-diff`: open side-by-side VS Code diffs after changes
- `--api-key`: override env key for selected provider
- `--base-url`: provider-compatible API base URL override

## How It Works

### Architecture

- `liteagent/cli.py`: argument parsing, interactive loop, provider/model selection
- `liteagent/agent.py`: provider-specific LLM loop and tool-calling orchestration
- `liteagent/tools.py`: deterministic local tools:
  - `get_project_info`
  - `list_files`
  - `search_text`
  - `read_file`
  - `write_file`
  - `append_file`
  - `make_dir`
  - `file_exists`
  - `run_shell`
- `liteagent/changes.py`: before/after tracking, unified patch generation, VS Code diff

### Context used for coding

The coding context is the conversation state held in `CodingAgent.messages` plus
the fixed system prompt in `liteagent/agent.py`.

Each turn adds this user task envelope:

```text
Project root: <absolute_path>
Task: <your_instruction>
Do the work now.
```

The model then receives and produces context in this sequence:

1. User task message is appended.
2. Assistant response is appended.
3. If tool calls are requested, tool outputs are appended.
4. Steps 2-3 repeat until final text response or `--max-steps` limit.

What is included in coding context:

- System prompt with coding rules (inspect files first, minimal edits, use tools).
- Current instruction and prior instructions in the same session.
- Tool transcripts:
  - file listings
  - file reads
  - file write confirmations
  - shell command stdout/stderr snippets
- Project root path for every turn.

Provider-specific message shape:

- OpenAI: `system + messages` with function tool calls.
- Anthropic: `system + messages` with `tool_use` / `tool_result` blocks.

Implications:

- Interactive mode accumulates coding context across turns in one running process.
- One-shot `run` has context only for that invocation.
- Context is in memory only; it is not persisted across separate CLI launches.

### Provider behavior

- OpenAI path uses Chat Completions with function tools.
- Anthropic path uses Messages API with `tool_use` / `tool_result` blocks.
- If selected provider key is missing, startup fails with a clear message:
  - `OPENAI_API_KEY` for `openai`
  - `ANTHROPIC_API_KEY` for `anthropic`
- If Anthropic model is invalid, liteagent returns a friendly error and tries to list
  available model ids.

### Change tracking and diffs

- All writes go through `ChangeTracker`.
- Original file state is captured once before first write.
- Combined unified patch is written to `--patch-out`.
- `--open-diff` opens `code --diff <before> <after>` for changed files.

## Notes

- This is a learning implementation, not a hardened sandbox.
- `run_shell` executes local shell commands from project root; use trusted prompts.
- Install VS Code CLI (`code`) to use visual diff opening.
