"""Tests for the read, write, and edit tools."""

import pytest

from bridle.tools import ReadTool, WriteTool, EditTool


@pytest.fixture
def workdir(tmp_path):
    return tmp_path


class TestRead:
    def test_reads_file_contents(self, workdir):
        (workdir / "a.txt").write_text("hello\nworld\n")
        result = ReadTool(workdir).run({"path": "a.txt"})
        assert result.is_error is False
        assert result.output == "hello\nworld\n"

    def test_missing_file_is_error(self, workdir):
        result = ReadTool(workdir).run({"path": "nope.txt"})
        assert result.is_error is True
        assert "nope.txt" in result.output

    def test_offset_and_limit_select_lines(self, workdir):
        (workdir / "a.txt").write_text("l1\nl2\nl3\nl4\n")
        result = ReadTool(workdir).run({"path": "a.txt", "offset": 2, "limit": 2})
        assert result.output.startswith("l2\nl3\n")
        assert "1 more line" in result.output  # paging hint for the model

    def test_long_file_is_truncated_with_notice(self, workdir):
        (workdir / "big.txt").write_text("x\n" * 5000)
        result = ReadTool(workdir).run({"path": "big.txt"})
        assert result.output.count("\n") <= ReadTool.MAX_LINES + 1
        assert "truncated" in result.output

    def test_absolute_path_is_allowed(self, workdir):
        target = workdir / "abs.txt"
        target.write_text("abs")
        result = ReadTool(workdir).run({"path": str(target)})
        assert result.output == "abs"


class TestWrite:
    def test_writes_file(self, workdir):
        result = WriteTool(workdir).run({"path": "out.txt", "content": "data"})
        assert result.is_error is False
        assert (workdir / "out.txt").read_text() == "data"

    def test_creates_parent_directories(self, workdir):
        WriteTool(workdir).run({"path": "a/b/c.txt", "content": "deep"})
        assert (workdir / "a/b/c.txt").read_text() == "deep"

    def test_overwrites_existing_file(self, workdir):
        (workdir / "x.txt").write_text("old")
        WriteTool(workdir).run({"path": "x.txt", "content": "new"})
        assert (workdir / "x.txt").read_text() == "new"


class TestEdit:
    def test_replaces_unique_string(self, workdir):
        (workdir / "f.py").write_text("a = 1\nb = 2\n")
        result = EditTool(workdir).run(
            {"path": "f.py", "old_string": "b = 2", "new_string": "b = 3"}
        )
        assert result.is_error is False
        assert (workdir / "f.py").read_text() == "a = 1\nb = 3\n"

    def test_missing_old_string_is_error(self, workdir):
        (workdir / "f.py").write_text("a = 1\n")
        result = EditTool(workdir).run(
            {"path": "f.py", "old_string": "zzz", "new_string": "y"}
        )
        assert result.is_error is True
        assert "not found" in result.output

    def test_ambiguous_old_string_is_error(self, workdir):
        (workdir / "f.py").write_text("x\nx\n")
        result = EditTool(workdir).run(
            {"path": "f.py", "old_string": "x", "new_string": "y"}
        )
        assert result.is_error is True
        assert "2" in result.output  # occurrence count aids the model

    def test_replace_all_replaces_every_occurrence(self, workdir):
        (workdir / "f.py").write_text("x\nx\n")
        result = EditTool(workdir).run(
            {
                "path": "f.py",
                "old_string": "x",
                "new_string": "y",
                "replace_all": True,
            }
        )
        assert result.is_error is False
        assert (workdir / "f.py").read_text() == "y\ny\n"

    def test_missing_file_is_error(self, workdir):
        result = EditTool(workdir).run(
            {"path": "ghost.py", "old_string": "a", "new_string": "b"}
        )
        assert result.is_error is True
