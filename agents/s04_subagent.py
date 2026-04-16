#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents

Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.

    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ---------->| while tool_use:  |
    |   prompt="..."   |            |   call tools     |
    |   description="" |            |   append results |
    |                  |  summary   |                  |
    |   result = "..." | <--------- | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.

Key insight: "Process isolation gives context isolation for free."
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL"),
)
MODEL = os.environ["MODEL_ID"]

WORKDIR = Path.cwd()

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


# -- Tool implementations shared by parent and child --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

def parse_tool_arguments(arguments: str | None) -> tuple[dict[str, Any] | None, str | None]:
    raw_arguments = arguments or "{}"
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return None, f"Error: Invalid tool arguments: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool arguments must decode to a JSON object"
    return parsed, None


def compress_tool_output(tool_name: str, args: dict[str, Any], output: str) -> str:
    if output.startswith("Error:"):
        return output[:2000]

    if tool_name == "read_file":
        path = args.get("path", "<unknown>")
        lines = output.splitlines()
        summary: list[str] = [f"Read {path} ({len(lines)} lines)."]

        docstring_line = ""
        for line in lines[:40]:
            stripped = line.strip().strip("\"'")
            if stripped and not stripped.startswith("#"):
                docstring_line = stripped
                break
        if docstring_line:
            summary.append(f"Top text: {docstring_line[:160]}")

        definitions = []
        for line in lines:
            match = re.match(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if match:
                definitions.append(f"{match.group(1)} {match.group(2)}")
            if len(definitions) >= 20:
                break
        if definitions:
            summary.append("Symbols: " + ", ".join(definitions))

        preview = [line for line in lines[:20] if line.strip()]
        if preview:
            summary.append("Preview:\n" + "\n".join(preview[:12]))
        return "\n".join(summary)[:2500]

    if tool_name == "bash":
        command = args.get("command", "")
        lines = output.splitlines()
        if "rg --files" in command or "find " in command:
            preview = "\n".join(lines[:80])
            return f"Command `{command}` returned {len(lines)} lines.\n{preview}"[:2500]
        return f"Command `{command}` output:\n{output[:2200]}"

    return output[:2000]


# Child gets all base tools except task (no recursive spawning)
CHILD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUBAGENT_SYSTEM}, *sub_messages],
            tools=CHILD_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            sanitized_tool_calls = []
            invalid_tool_calls = []
            for tool_call in message.tool_calls:
                args, error = parse_tool_arguments(tool_call.function.arguments)
                if error:
                    invalid_tool_calls.append(f"{tool_call.function.name}: {error}")
                    continue
                sanitized_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.dumps(args, ensure_ascii=True),
                        },
                    }
                )
            if sanitized_tool_calls:
                assistant_message["tool_calls"] = sanitized_tool_calls
            if invalid_tool_calls:
                error_text = "\n".join(invalid_tool_calls)
                assistant_message["content"] = (
                    (assistant_message["content"] + "\n") if assistant_message["content"] else ""
                ) + error_text
        sub_messages.append(assistant_message)

        if not message.tool_calls:
            return message.content or "(no summary)"

        tool_messages = []
        for tool_call in message.tool_calls:
            args, error = parse_tool_arguments(tool_call.function.arguments)
            if error:
                output = error
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                output = handler(**args) if handler else f"Error: Unknown tool {tool_call.function.name}"
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": compress_tool_output(tool_call.function.name, args or {}, str(output)),
                }
            )
        sub_messages.extend(tool_messages)
    # Only the final text returns to the parent -- child context is discarded
    return "(subagent stopped after 30 steps)"


# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def agent_loop(messages: list) -> str:
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=PARENT_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            sanitized_tool_calls = []
            invalid_tool_calls = []
            for tool_call in message.tool_calls:
                args, error = parse_tool_arguments(tool_call.function.arguments)
                if error:
                    invalid_tool_calls.append(f"{tool_call.function.name}: {error}")
                    continue
                sanitized_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": json.dumps(args, ensure_ascii=True),
                        },
                    }
                )
            if sanitized_tool_calls:
                assistant_message["tool_calls"] = sanitized_tool_calls
            if invalid_tool_calls:
                error_text = "\n".join(invalid_tool_calls)
                assistant_message["content"] = (
                    (assistant_message["content"] + "\n") if assistant_message["content"] else ""
                ) + error_text
        messages.append(assistant_message)

        if not message.tool_calls:
            return message.content or ""

        tool_messages = []
        for tool_call in message.tool_calls:
            args, error = parse_tool_arguments(tool_call.function.arguments)
            if error:
                output = error
            elif tool_call.function.name == "task":
                desc = args.get("description", "subtask")
                print(f"> task ({desc}): {args['prompt'][:80]}")
                output = run_subagent(args["prompt"])
            else:
                handler = TOOL_HANDLERS.get(tool_call.function.name)
                output = handler(**args) if handler else f"Error: Unknown tool {tool_call.function.name}"
            print(f"  {str(output)[:200]}")
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )
        messages.extend(tool_messages)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
