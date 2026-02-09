"""CLI for converting timestamped ON/OFF data into events."""

import argparse
import sys
from typing import Iterator, TextIO

import numpy as np
from dateutil.parser import parse as parse_datetime

from event_times.state import OnOffStateProcessor

_DEFAULT_BATCH_SIZE = 5_000


def _parse_line(
    lineno: int, line: str, all_on: bool
) -> tuple[np.datetime64, bool] | None:
    """Parse a single input line. Returns None for blank/comment lines."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    ts_str = parts[0]

    try:
        dt = np.datetime64(parse_datetime(ts_str), "ns")
    except (ValueError, OverflowError) as exc:
        print(f"line {lineno}: {exc}", file=sys.stderr)
        sys.exit(1)

    if all_on:
        return dt, True

    if len(parts) < 2 or parts[1].upper() not in ("ON", "OFF"):
        print(
            f"line {lineno}: expected '<timestamp> ON|OFF'",
            file=sys.stderr,
        )
        sys.exit(1)

    return dt, parts[1].upper() == "ON"


def _iter_batches(
    stream: TextIO, all_on: bool, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (time_arr, state_arr) batches from *stream*.

    Validates that timestamps are non-decreasing across the entire stream.
    """
    prev_time: np.datetime64 | None = None
    times: list[np.datetime64] = []
    states: list[bool] = []

    for lineno, raw in enumerate(stream, 1):
        result = _parse_line(lineno, raw, all_on)
        if result is None:
            continue

        t, s = result
        if prev_time is not None and t < prev_time:
            print(
                f"line {lineno}: timestamp goes backwards ({t} < {prev_time})",
                file=sys.stderr,
            )
            sys.exit(1)
        prev_time = t
        times.append(t)
        states.append(s)

        if len(times) >= batch_size:
            yield (
                np.array(times, dtype="datetime64[ns]"),
                np.array(states, dtype=bool),
            )
            times = []
            states = []

    if times:
        yield (
            np.array(times, dtype="datetime64[ns]"),
            np.array(states, dtype=bool),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="event-times",
        description="Convert timestamped ON/OFF data into events.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help="Input file (default: stdin).",
    )
    parser.add_argument(
        "--all-on",
        action="store_true",
        help="Treat every timestamp as ON (no keyword column).",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=60.0,
        help="Max gap in seconds between samples (default: 60).",
    )
    parser.add_argument(
        "--description",
        help="Description to apply to all generated events.",
    )
    parser.add_argument(
        "--color",
        help="Color to apply to all generated events.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Lines per processing batch (default: {_DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args(argv)

    proc = OnOffStateProcessor(
        description=args.description,
        color=args.color,
        max_gap=args.max_gap,
    )

    if args.file == "-":
        stream = sys.stdin
    else:
        stream = open(args.file)

    try:
        for time_arr, state_arr in _iter_batches(stream, args.all_on, args.batch_size):
            proc(time_arr, state_arr)
            for event in proc.get_events():
                print(event.model_dump_json())
    finally:
        if stream is not sys.stdin:
            stream.close()

    proc.finalize()
    for event in proc.get_events():
        print(event.model_dump_json())
