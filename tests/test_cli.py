"""Tests for the event-times CLI."""

import json
from pathlib import Path

import pytest

from event_times.cli import main

DATA = Path(__file__).parent / "data"


def run_cli(capsys, argv: list[str]) -> list[dict]:
    """Run main() and return parsed JSON Lines output."""
    main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    return [json.loads(line) for line in lines] if lines else []


class TestBasicOnOff:
    def test_two_events(self, capsys):
        events = run_cli(capsys, [str(DATA / "basic_on_off.txt")])
        assert len(events) == 2
        assert events[0]["start_max"] == "2024-01-01T10:00:05.000000000"
        assert events[0]["stop_max"] == "2024-01-01T10:00:15.000000000"
        assert events[1]["start_max"] == "2024-01-01T10:00:50.000000000"
        assert events[1]["stop_max"] == "2024-01-01T10:00:55.000000000"

    def test_case_insensitive(self, capsys):
        events = run_cli(capsys, [str(DATA / "case_insensitive.txt")])
        assert len(events) == 1
        assert events[0]["start_max"] == "2024-01-01T10:00:05.000000000"

    def test_extra_columns_ignored(self, capsys):
        events = run_cli(capsys, [str(DATA / "extra_columns.txt")])
        assert len(events) == 1
        assert events[0]["start_max"] == "2024-01-01T10:00:05.000000000"


class TestAllOn:
    def test_all_on(self, capsys):
        events = run_cli(capsys, ["--all-on", str(DATA / "all_on.txt")])
        assert len(events) == 1
        assert events[0]["start_max"] == "2024-01-01T10:00:00.000000000"
        assert events[0]["stop_min"] == "2024-01-01T10:00:10.000000000"

    def test_all_on_extra_columns(self, capsys):
        events = run_cli(capsys, ["--all-on", str(DATA / "all_on_extra_columns.txt")])
        assert len(events) == 1
        assert events[0]["start_max"] == "2024-01-01T10:00:00.000000000"
        assert events[0]["stop_min"] == "2024-01-01T10:00:05.000000000"


class TestCommentsAndBlanks:
    def test_comments_and_blanks_skipped(self, capsys):
        events = run_cli(capsys, [str(DATA / "comments_and_blanks.txt")])
        assert len(events) == 1
        assert events[0]["start_max"] == "2024-01-01T10:00:00.000000000"
        assert events[0]["stop_min"] == "2024-01-01T10:00:05.000000000"

    def test_empty_input_no_output(self, capsys):
        main([str(DATA / "empty.txt")])
        assert capsys.readouterr().out == ""


class TestMetadata:
    def test_description(self, capsys):
        events = run_cli(
            capsys,
            ["--description", "sensor A", str(DATA / "basic_on_off.txt")],
        )
        assert all(e["description"] == "sensor A" for e in events)

    def test_color(self, capsys):
        events = run_cli(
            capsys,
            ["--color", "#ff0000", str(DATA / "basic_on_off.txt")],
        )
        assert all(e["color"] == "#ff0000" for e in events)

    def test_description_and_color(self, capsys):
        events = run_cli(
            capsys,
            [
                "--description",
                "test",
                "--color",
                "red",
                str(DATA / "basic_on_off.txt"),
            ],
        )
        assert events[0]["description"] == "test"
        assert events[0]["color"] == "red"

    def test_defaults_to_null(self, capsys):
        events = run_cli(capsys, [str(DATA / "basic_on_off.txt")])
        assert events[0]["description"] is None
        assert events[0]["color"] is None


class TestMaxGap:
    def test_default_gap_splits(self, capsys):
        events = run_cli(capsys, [str(DATA / "gap.txt")])
        assert len(events) == 2

    def test_large_gap_merges(self, capsys):
        events = run_cli(capsys, ["--max-gap", "600", str(DATA / "gap.txt")])
        assert len(events) == 1


class TestBatchSize:
    def test_small_batch_same_result(self, capsys):
        events_default = run_cli(capsys, [str(DATA / "basic_on_off.txt")])
        events_small = run_cli(
            capsys, ["--batch-size", "2", str(DATA / "basic_on_off.txt")]
        )
        assert events_default == events_small


class TestErrors:
    def test_backwards_timestamps(self):
        with pytest.raises(SystemExit):
            main([str(DATA / "backwards.txt")])

    def test_missing_keyword(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("2024-01-01T10:00:00\n")
        with pytest.raises(SystemExit):
            main([str(f)])

    def test_invalid_keyword(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("2024-01-01T10:00:00 MAYBE\n")
        with pytest.raises(SystemExit):
            main([str(f)])

    def test_invalid_timestamp(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("not-a-date ON\n")
        with pytest.raises(SystemExit):
            main([str(f)])
