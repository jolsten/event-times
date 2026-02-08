#!/usr/bin/env python3
"""Command-line interface for event-times package.

Reads timestamp and state pairs from stdin or a file, analyzes events,
and outputs event timing information.
"""

import argparse
import sys
from typing import List, TextIO, Tuple

import numpy as np

from .core import Event, on_off_times


def parse_line(line: str) -> Tuple[str, bool]:
    """Parse a line of input into timestamp and state.

    Args:
        line: Input line with format: "<iso8601-timestamp> ON|OFF [extra text...]"

    Returns:
        Tuple of (timestamp_string, state_boolean)

    Raises:
        ValueError: If line format is invalid.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None, None

    parts = line.split(None, 2)  # Split on whitespace, max 3 parts
    if len(parts) < 2:
        raise ValueError(f"Invalid line format: {line}")

    timestamp_str = parts[0]
    state_str = parts[1].upper()

    if state_str not in ("ON", "OFF"):
        raise ValueError(f"State must be ON or OFF, got: {state_str}")

    state = state_str == "ON"
    return timestamp_str, state


def read_input(file_handle: TextIO) -> Tuple[List[str], List[bool]]:
    """Read and parse input lines.

    Args:
        file_handle: File handle to read from (stdin or file).

    Returns:
        Tuple of (timestamps_list, states_list).
    """
    timestamps = []
    states = []

    for line_num, line in enumerate(file_handle, 1):
        try:
            timestamp_str, state = parse_line(line)
            if timestamp_str is not None:
                timestamps.append(timestamp_str)
                states.append(state)
        except ValueError as e:
            print(f"Warning: Line {line_num}: {e}", file=sys.stderr)
            continue

    return timestamps, states


def format_event(event: Event, index: int) -> str:
    """Format an event for output.

    Args:
        event: Event object to format.
        index: Event number (1-indexed).

    Returns:
        Formatted string representation.
    """
    lines = [f"Event {index}:"]
    lines.append(f"  Start:    {event.start}")
    lines.append(f"  Stop:     {event.stop}")
    lines.append(f"  Duration: {event.duration:.1f}s")

    if event.last_off is not None:
        lines.append(f"  Last OFF: {event.last_off}")
    if event.first_on is not None:
        lines.append(f"  First ON: {event.first_on}")
    if event.last_on is not None:
        lines.append(f"  Last ON:  {event.last_on}")
    if event.first_off is not None:
        lines.append(f"  First OFF: {event.first_off}")

    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze boolean time series data and extract event timing information.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input format:
  Each line: <ISO8601-timestamp> ON|OFF [extra text...]
  
  Example:
    2024-01-01T10:00:00 OFF
    2024-01-01T10:00:30 ON started
    2024-01-01T10:01:00 ON still running
    2024-01-01T10:01:30 OFF stopped

  Lines starting with # are ignored.
  Any text after the ON/OFF keyword is ignored.
        """,
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input file (default: stdin)",
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum time gap in seconds between events (default: 60.0)",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Show only summary statistics"
    )

    parser.add_argument("--csv", action="store_true", help="Output in CSV format")

    args = parser.parse_args()

    try:
        # Read input
        timestamp_strs, states_list = read_input(args.input_file)

        if not timestamp_strs:
            print("Error: No valid input data", file=sys.stderr)
            sys.exit(1)

        # Convert to numpy arrays
        timestamps = np.array(timestamp_strs, dtype="datetime64")
        states = np.array(states_list, dtype=bool)

        # Analyze events
        events = on_off_times(timestamps, states, max_gap=args.max_gap)

        if not events:
            print("No events detected")
            sys.exit(0)

        # Output results
        if args.csv:
            print("event_num,start,stop,duration_s,last_off,first_on,last_on,first_off")
            for i, event in enumerate(events, 1):
                print(
                    f"{i},{event.start},{event.stop},{event.duration:.1f},"
                    f"{event.last_off or ''},{event.first_on or ''},"
                    f"{event.last_on or ''},{event.first_off or ''}"
                )
        elif args.summary:
            total_duration = sum(e.duration for e in events)
            print(f"Total events: {len(events)}")
            print(f"Total duration: {total_duration:.1f}s")
            if events:
                print(f"First event start: {events[0].start}")
                print(f"Last event stop: {events[-1].stop}")
                avg_duration = total_duration / len(events)
                print(f"Average duration: {avg_duration:.1f}s")
        else:
            for i, event in enumerate(events, 1):
                print(format_event(event, i))
                print()

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.input_file is not sys.stdin:
            args.input_file.close()


if __name__ == "__main__":
    main()
