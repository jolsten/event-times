#!/usr/bin/env python3
"""
Comprehensive demonstration of the onofftimes package.

This script demonstrates all key features and edge cases of the package.
"""

import sys

import numpy as np

sys.path.insert(0, "/mnt/user-data/outputs/onofftimes")
from event_times import on_off_times


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_usage():
    """Demonstrate basic usage."""
    print_section("1. Basic Usage")

    timestamps = np.array(
        [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:30",
            "2024-01-01T10:01:00",
            "2024-01-01T10:01:30",
            "2024-01-01T10:02:00",
        ],
        dtype="datetime64",
    )

    states = np.array([False, True, True, True, False])

    print("\nInput:")
    for t, s in zip(timestamps, states):
        print(f"  {t}: {'ON' if s else 'OFF'}")

    events = on_off_times(timestamps, states)

    print(f"\nDetected {len(events)} event(s):")
    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(f"  start_min:  {event.start_min}")
        print(f"  start_max:  {event.start_max}")
        print(f"  stop_min:   {event.stop_min}")
        print(f"  stop_max: {event.stop_max}")


def demo_event_properties():
    """Demonstrate event properties."""
    print_section("2. Event Properties")

    timestamps = np.array(
        [
            "2024-01-01T14:00:00",
            "2024-01-01T14:00:30",
            "2024-01-01T14:01:00",
            "2024-01-01T14:01:30",
        ],
        dtype="datetime64",
    )

    states = np.array([False, True, True, False])
    events = on_off_times(timestamps, states)
    event = events[0]

    print("\nEvent timestamps:")
    print(f"  start_min:  {event.start_min}")
    print(f"  start_max:  {event.start_max}")
    print(f"  stop_min:   {event.stop_min}")
    print(f"  stop_max: {event.stop_max}")

    print("\nEvent properties:")
    print(f"  start:          {event.start}")
    print(f"  stop:           {event.stop}")
    print(f"  inner_interval: {event.inner_interval}")
    print(f"  outer_interval: {event.outer_interval}")

    inner_duration = event.inner_interval[1] - event.inner_interval[0]
    outer_duration = event.outer_interval[1] - event.outer_interval[0]

    print("\nDurations:")
    print(f"  Inner (guaranteed on): {inner_duration}")
    print(f"  Outer (full span):     {outer_duration}")


def demo_time_gaps():
    """Demonstrate time gap handling."""
    print_section("3. Time Gap Handling")

    timestamps = np.array(
        [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:30",
            "2024-01-01T10:05:00",  # 4.5 minute gap
            "2024-01-01T10:05:30",
        ],
        dtype="datetime64",
    )

    states = np.array([True, True, True, True])

    print("\nAll states are True, but there's a large time gap:")
    for i, t in enumerate(timestamps):
        if i > 0:
            gap = timestamps[i] - timestamps[i - 1]
            print(f"  {timestamps[i - 1]} -> {t} (gap: {gap})")

    # With default 60-second threshold
    events_default = on_off_times(timestamps, states)
    print(f"\nWith default 60s threshold: {len(events_default)} events")
    for i, event in enumerate(events_default, 1):
        print(f"  Event {i}: {event.start_max} to {event.stop_min}")

    # With large threshold
    events_large = on_off_times(timestamps, states, max_gap=np.timedelta64(10, "m"))
    print(f"\nWith 10-minute threshold: {len(events_large)} event(s)")
    for i, event in enumerate(events_large, 1):
        print(f"  Event {i}: {event.start_max} to {event.stop_min}")


def demo_edge_cases():
    """Demonstrate edge cases."""
    print_section("4. Edge Cases")

    timestamps = np.array(
        [
            "2024-01-01T12:00:00",
            "2024-01-01T12:01:00",
            "2024-01-01T12:02:00",
        ],
        dtype="datetime64",
    )

    # Case 1: Series starts with True
    print("\nCase 1: Series starts in ON state")
    states1 = np.array([True, True, False])
    events1 = on_off_times(timestamps, states1)
    print(f"  start_min:  {events1[0].start_min} <- None (no prior OFF)")
    print(f"  start_max:  {events1[0].start_max}")

    # Case 2: Series ends with True
    print("\nCase 2: Series ends in ON state")
    states2 = np.array([False, True, True])
    events2 = on_off_times(timestamps, states2)
    print(f"  stop_min:   {events2[0].stop_min}")
    print(f"  stop_max: {events2[0].stop_max} <- None (no subsequent OFF)")

    # Case 3: All False
    print("\nCase 3: All states are OFF")
    states3 = np.array([False, False, False])
    events3 = on_off_times(timestamps, states3)
    print(f"  Events: {events3} <- Empty list")

    # Case 4: All True (no gaps)
    print("\nCase 4: All states are ON (no large gaps)")
    states4 = np.array([True, True, True])
    events4 = on_off_times(timestamps, states4)
    print(f"  Events: {len(events4)} event")
    print(f"  Spans: {events4[0].start_max} to {events4[0].stop_min}")
    print(f"  start_min:  {events4[0].start_min}")
    print(f"  stop_max: {events4[0].stop_max}")


def demo_multiple_events():
    """Demonstrate multiple events."""
    print_section("5. Multiple Events")

    timestamps = np.array(
        [
            "2024-01-01T08:00:00",
            "2024-01-01T08:00:30",
            "2024-01-01T08:01:00",
            "2024-01-01T08:01:30",
            "2024-01-01T08:02:00",
            "2024-01-01T08:02:30",
            "2024-01-01T08:03:00",
        ],
        dtype="datetime64",
    )

    states = np.array([False, True, False, True, True, False, True])

    print("\nTimeline:")
    for t, s in zip(timestamps, states):
        print(f"  {t}: {'█' if s else '░'} ({'ON' if s else 'OFF'})")

    events = on_off_times(timestamps, states)

    print(f"\nDetected {len(events)} events:")
    for i, event in enumerate(events, 1):
        duration = (
            (event.stop_min - event.start_max)
            if event.start_max and event.stop_min
            else None
        )
        print(f"\nEvent {i}:")
        print(f"  Period: {event.start_max} to {event.stop_min}")
        print(f"  Duration: {duration}")
        print(f"  Boundaries: {event.start_min} -> {event.stop_max}")


def demo_validation():
    """Demonstrate input validation."""
    print_section("6. Input Validation")

    print("\nTesting empty arrays:")
    try:
        on_off_times(np.array([]), np.array([]))
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")

    print("\nTesting mismatched lengths:")
    try:
        timestamps = np.array(
            ["2024-01-01T10:00:00", "2024-01-01T10:01:00"], dtype="datetime64"
        )
        states = np.array([True])
        on_off_times(timestamps, states)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")

    print("\nTesting unsorted timestamps:")
    try:
        timestamps = np.array(
            ["2024-01-01T10:01:00", "2024-01-01T10:00:00"], dtype="datetime64"
        )
        states = np.array([True, False])
        on_off_times(timestamps, states)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised: {e}")


def demo_real_world_scenario():
    """Demonstrate a real-world scenario."""
    print_section("7. Real-World Scenario: Server Monitoring")

    # Simulate server health check data (every 10 seconds)
    base = np.datetime64("2024-01-15T10:00:00")
    timestamps = base + np.arange(11) * np.timedelta64(10, "s")

    # Server responding: ON for 30s, OFF for 20s, ON for 60s
    states = np.array(
        [
            True,
            True,
            True,  # 0-30s: responding (3 samples)
            False,
            False,  # 30-50s: not responding (2 samples)
            True,
            True,
            True,  # 50-110s: responding (6 samples)
            True,
            True,
            True,
        ]
    )

    print("\nServer health check timeline (10s intervals):")
    for i, (t, s) in enumerate(zip(timestamps, states)):
        status = "✓ Responding" if s else "✗ Down"
        print(f"  {t}: {status}")

    events = on_off_times(timestamps, states)

    print(f"\nDetected {len(events)} uptime period(s):")
    total_uptime = np.timedelta64(0, "s")

    for i, event in enumerate(events, 1):
        start = event.start_max
        stop = event.stop_min
        duration = stop - start
        total_uptime += duration

        print(f"\nUptime Period {i}:")
        print(f"  Start:    {start}")
        print(f"  End:      {stop}")
        print(f"  Duration: {duration}")

    print(f"\nTotal uptime: {total_uptime}")
    total_time = timestamps[-1] - timestamps[0]
    uptime_pct = (total_uptime / total_time) * 100
    print(f"Uptime percentage: {uptime_pct:.1f}%")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ONOFFTIMES PACKAGE - COMPREHENSIVE DEMONSTRATION".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    demo_basic_usage()
    demo_event_properties()
    demo_time_gaps()
    demo_edge_cases()
    demo_multiple_events()
    demo_validation()
    demo_real_world_scenario()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
