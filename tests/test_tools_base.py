"""Tests for the tool protocol and registry."""

import pytest

from bridle.tools import Tool, ToolRegistry, ToolResult, default_registry


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the input."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, args: dict) -> ToolResult:
        return ToolResult(output=args["text"])


def test_tool_result_defaults_to_success():
    result = ToolResult(output="hi")
    assert result.output == "hi"
    assert result.is_error is False


def test_registry_registers_and_runs_tool():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = registry.run("echo", {"text": "hello"})
    assert result.output == "hello"
    assert result.is_error is False


def test_registry_unknown_tool_returns_error_result():
    registry = ToolRegistry()
    result = registry.run("nope", {})
    assert result.is_error is True
    assert "nope" in result.output


def test_registry_tool_exception_returns_error_result():
    class Boom(Tool):
        name = "boom"
        description = "Always fails."
        parameters = {"type": "object", "properties": {}}

        def run(self, args: dict) -> ToolResult:
            raise ValueError("kaput")

    registry = ToolRegistry()
    registry.register(Boom())
    result = registry.run("boom", {})
    assert result.is_error is True
    assert "kaput" in result.output


def test_registry_schemas_lists_all_tools():
    registry = ToolRegistry()
    registry.register(EchoTool())
    schemas = registry.schemas()
    assert schemas == [
        {
            "name": "echo",
            "description": "Echo back the input.",
            "input_schema": EchoTool.parameters,
        }
    ]


def test_default_registry_has_four_core_tools(tmp_path):
    registry = default_registry(tmp_path)
    names = {schema["name"] for schema in registry.schemas()}
    assert names == {"read", "write", "edit", "bash"}
