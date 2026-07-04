"""The four core tools: read, write, edit, bash.

Frontier models are heavily RL-trained on coding-agent workflows; these four
tools are sufficient for effective coding work. Anything else the agent needs
it can get through bash.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ToolResult:
    """Outcome of a tool invocation, fed back to the model verbatim."""

    output: str
    is_error: bool = False


class Tool(ABC):
    """A callable capability exposed to the model.

    Subclasses define ``name``, ``description``, and ``parameters`` (a JSON
    schema for the arguments) as class attributes.
    """

    name: str
    description: str
    parameters: dict

    @abstractmethod
    def run(self, args: dict) -> ToolResult: ...


class ToolRegistry:
    """Holds the tools available to one agent and dispatches calls to them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def run(self, name: str, args: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(output=f"Unknown tool: {name}", is_error=True)
        try:
            return tool.run(args)
        except Exception as exc:  # errors go back to the model, never crash the loop
            return ToolResult(output=f"{type(exc).__name__}: {exc}", is_error=True)


class _WorkdirTool(Tool):
    """Base for tools that resolve relative paths against a working directory."""

    def __init__(self, workdir: Path | str) -> None:
        self.workdir = Path(workdir)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.workdir / p


class ReadTool(_WorkdirTool):
    name = "read"
    description = (
        "Read a file. Returns its contents. Use offset (1-based line number) "
        "and limit to page through large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute"},
            "offset": {"type": "integer", "description": "1-based first line to read"},
            "limit": {"type": "integer", "description": "Max number of lines to read"},
        },
        "required": ["path"],
    }

    MAX_LINES = 2000

    def run(self, args: dict) -> ToolResult:
        target = self._resolve(args["path"])
        if not target.is_file():
            return ToolResult(output=f"File not found: {args['path']}", is_error=True)
        lines = target.read_text(errors="replace").splitlines(keepends=True)
        offset = max(args.get("offset", 1), 1)
        limit = args.get("limit", self.MAX_LINES)
        selected = lines[offset - 1 : offset - 1 + limit]
        output = "".join(selected)
        if len(lines) > offset - 1 + limit:
            remaining = len(lines) - (offset - 1 + limit)
            plural = "s" if remaining != 1 else ""
            output += f"\n[truncated: {remaining} more line{plural}, re-read with offset={offset + limit}]"
        return ToolResult(output=output)


class WriteTool(_WorkdirTool):
    name = "write"
    description = "Write content to a file, creating it (and parent directories) or overwriting it."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute"},
            "content": {"type": "string", "description": "Full file content to write"},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict) -> ToolResult:
        target = self._resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        return ToolResult(output=f"Wrote {len(args['content'])} chars to {args['path']}")


class EditTool(_WorkdirTool):
    name = "edit"
    description = (
        "Replace old_string with new_string in a file. old_string must match "
        "exactly one location unless replace_all is true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative or absolute"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, args: dict) -> ToolResult:
        target = self._resolve(args["path"])
        if not target.is_file():
            return ToolResult(output=f"File not found: {args['path']}", is_error=True)
        content = target.read_text()
        count = content.count(args["old_string"])
        if count == 0:
            return ToolResult(
                output=f"old_string not found in {args['path']}", is_error=True
            )
        if count > 1 and not args.get("replace_all"):
            return ToolResult(
                output=(
                    f"old_string matches {count} locations in {args['path']}; "
                    "add more context to make it unique, or set replace_all"
                ),
                is_error=True,
            )
        target.write_text(content.replace(args["old_string"], args["new_string"]))
        return ToolResult(output=f"Replaced {count} occurrence(s) in {args['path']}")


class BashTool(_WorkdirTool):
    name = "bash"
    description = (
        "Run a shell command in the working directory. Returns combined "
        "stdout/stderr. Use for anything the other tools don't cover: "
        "search, git, tests, package managers, process control."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120)"},
        },
        "required": ["command"],
    }

    MAX_OUTPUT_CHARS = 50_000
    DEFAULT_TIMEOUT = 120

    def run(self, args: dict) -> ToolResult:
        timeout = args.get("timeout", self.DEFAULT_TIMEOUT)
        try:
            proc = subprocess.run(
                args["command"],
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Command timed out after {timeout}s", is_error=True
            )
        output = proc.stdout + proc.stderr
        if len(output) > self.MAX_OUTPUT_CHARS:
            output = output[: self.MAX_OUTPUT_CHARS] + "\n[truncated]"
        if proc.returncode != 0:
            return ToolResult(
                output=f"{output}\n[exit code {proc.returncode}]", is_error=True
            )
        return ToolResult(output=output or "[no output]")


def default_registry(workdir: Path | str) -> ToolRegistry:
    """The standard four-tool registry rooted at ``workdir``."""
    registry = ToolRegistry()
    for tool_cls in (ReadTool, WriteTool, EditTool, BashTool):
        registry.register(tool_cls(workdir))
    return registry
