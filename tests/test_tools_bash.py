"""Tests for the bash tool."""

from bridle.tools import BashTool


def test_runs_command_and_captures_stdout(tmp_path):
    result = BashTool(tmp_path).run({"command": "echo hello"})
    assert result.is_error is False
    assert "hello" in result.output


def test_runs_in_workdir(tmp_path):
    (tmp_path / "marker.txt").write_text("here")
    result = BashTool(tmp_path).run({"command": "ls"})
    assert "marker.txt" in result.output


def test_nonzero_exit_is_error_with_stderr(tmp_path):
    result = BashTool(tmp_path).run({"command": "ls /definitely-not-a-real-dir-xyz"})
    assert result.is_error is True
    assert "exit" in result.output.lower()


def test_timeout_is_error(tmp_path):
    result = BashTool(tmp_path).run({"command": "sleep 5", "timeout": 1})
    assert result.is_error is True
    assert "timed out" in result.output.lower()


def test_output_is_capped(tmp_path):
    result = BashTool(tmp_path).run({"command": "yes | head -c 1000000"})
    assert len(result.output) <= BashTool.MAX_OUTPUT_CHARS + 100
    assert "truncated" in result.output
