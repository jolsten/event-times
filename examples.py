"""
Examples for using the onofftimes package.
"""

import numpy as np

from event_times import on_off_times


def example_basic_usage():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    timestamps = np.array(
        [
            "2024-01-01T09:00:00",
            "2024-01-01T09:00:30",
            "2024-01-01T09:01:00",
            "2024-01-01T09:01:30",
            "2024-01-01T09:02:00",
        ],
        dtype="datetime64",
    )

    states = np.array([False, True, True, True, False])

    events = on_off_times(timestamps, states)

    print(f"\nNumber of events: {len(events)}")
    for i, event in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(f"  Last off:  {event.last_off}")
        print(f"  First on:  {event.first_on}")
        print(f"  Last on:   {event.last_on}")
        print(f"  First off: {event.first_off}")
        print(f"  Start:     {event.start}")
        print(f"  Stop:      {event.stop}")


def example_machine_monitoring():
    """Example: Monitoring machine on/off times."""
    print("\n" + "=" * 60)
    print("Example 2: Machine Monitoring")
    print("=" * 60)

    # Simulate machine state data sampled every 30 seconds
    base_time = np.datetime64("2024-01-15T08:00:00")
    timestamps = base_time + np.arange(30) * np.timedelta64(
        30, "s"
    )  # Every 30 seconds, 30 samples = 15 minutes

    # Machine is on for first 10 samples (5 min), off for 10 samples (5 min), on for last 10 samples (5 min)
    states = np.concatenate(
        [
            np.ones(10, dtype=bool),  # On for 5 minutes
            np.zeros(10, dtype=bool),  # Off for 5 minutes
            np.ones(10, dtype=bool),  # On for 5 minutes
        ]
    )

    events = on_off_times(timestamps, states)

    print(f"\nMachine had {len(events)} operational periods:")
    for i, event in enumerate(events, 1):
        duration = event.last_on - event.first_on
        print(f"\nPeriod {i}:")
        print(f"  Started: {event.first_on}")
        print(f"  Stopped: {event.last_on}")
        print(f"  Duration: {duration}")


def example_time_gaps():
    """Example: Handling time gaps in data."""
    print("\n" + "=" * 60)
    print("Example 3: Time Gaps in Data")
    print("=" * 60)

    # Simulate data with a large gap (e.g., data collection interruption)
    timestamps = np.array(
        [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:30",
            "2024-01-01T10:01:00",
            "2024-01-01T10:10:00",  # 9-minute gap
            "2024-01-01T10:10:30",
            "2024-01-01T10:11:00",
        ],
        dtype="datetime64",
    )

    states = np.array([True, True, True, True, True, True])

    # With default 60-second threshold, the gap splits the event
    events = on_off_times(timestamps, states)

    print(f"\nWith default 60s threshold: {len(events)} events")
    for i, event in enumerate(events, 1):
        print(f"  Event {i}: {event.first_on} to {event.last_on}")

    # With a larger threshold, it's treated as one continuous event
    events_large = on_off_times(timestamps, states, max_gap=np.timedelta64(10, "m"))

    print(f"\nWith 10-minute threshold: {len(events_large)} event")
    for i, event in enumerate(events_large, 1):
        print(f"  Event {i}: {event.first_on} to {event.last_on}")


def example_sensor_data():
    """Example: Analyzing sensor trigger events."""
    print("\n" + "=" * 60)
    print("Example 4: Sensor Trigger Events")
    print("=" * 60)

    # Simulate a motion sensor
    timestamps = np.array(
        [
            "2024-01-01T14:00:00",
            "2024-01-01T14:00:05",
            "2024-01-01T14:00:10",
            "2024-01-01T14:00:15",
            "2024-01-01T14:01:00",  # Long gap
            "2024-01-01T14:01:05",
            "2024-01-01T14:01:10",
            "2024-01-01T14:01:15",
            "2024-01-01T14:01:20",
        ],
        dtype="datetime64",
    )

    # Sensor triggered at various times
    states = np.array([False, True, True, False, False, True, False, True, False])

    events = on_off_times(timestamps, states)

    print(f"\nDetected {len(events)} trigger events:")
    for i, event in enumerate(events, 1):
        inner_start, inner_stop = event.inner_interval
        outer_start, outer_stop = event.outer_interval

        print(f"\nEvent {i}:")
        print(f"  Definitely active: {inner_start} to {inner_stop}")
        print(f"  Possibly active: {outer_start} to {outer_stop}")


def example_edge_cases():
    """Example: Edge cases and special scenarios."""
    print("\n" + "=" * 60)
    print("Example 5: Edge Cases")
    print("=" * 60)

    # Case 1: Series starts with True
    print("\nCase 1: Series starts with True state")
    timestamps1 = np.array(
        [
            "2024-01-01T12:00:00",
            "2024-01-01T12:01:00",
            "2024-01-01T12:02:00",
        ],
        dtype="datetime64",
    )
    states1 = np.array([True, True, False])

    events1 = on_off_times(timestamps1, states1)
    print(f"  last_off: {events1[0].last_off} (None because series starts on)")
    print(f"  first_on: {events1[0].first_on}")

    # Case 2: Series ends with True
    print("\nCase 2: Series ends with True state")
    states2 = np.array([False, True, True])
    events2 = on_off_times(timestamps1, states2)
    print(f"  last_on: {events2[0].last_on}")
    print(f"  first_off: {events2[0].first_off} (None because series ends on)")

    # Case 3: All False
    print("\nCase 3: All states are False")
    states3 = np.array([False, False, False])
    events3 = on_off_times(timestamps1, states3)
    print(f"  Number of events: {len(events3)} (empty list)")

    # Case 4: All True
    print("\nCase 4: All states are True")
    states4 = np.array([True, True, True])
    events4 = on_off_times(timestamps1, states4)
    print(f"  Number of events: {len(events4)}")
    print(f"  Spans entire series: {events4[0].first_on} to {events4[0].last_on}")


def example_using_event_properties():
    """Example: Using Event properties for analysis."""
    print("\n" + "=" * 60)
    print("Example 6: Using Event Properties")
    print("=" * 60)

    timestamps = np.array(
        [
            "2024-01-01T08:00:00",
            "2024-01-01T08:00:30",
            "2024-01-01T08:01:00",
            "2024-01-01T08:01:30",
            "2024-01-01T08:02:00",
        ],
        dtype="datetime64",
    )

    states = np.array([False, True, True, True, False])
    events = on_off_times(timestamps, states)

    event = events[0]

    print("\nEvent timing analysis:")
    print(f"  Start time: {event.start}")
    print(f"  Stop time: {event.stop}")

    inner_start, inner_stop = event.inner_interval
    print("\n  Guaranteed active period:")
    print(f"    From: {inner_start}")
    print(f"    To:   {inner_stop}")
    print(f"    Duration: {inner_stop - inner_start}")

    outer_start, outer_stop = event.outer_interval
    print("\n  Maximum possible period:")
    print(f"    From: {outer_start}")
    print(f"    To:   {outer_stop}")
    print(f"    Duration: {outer_stop - outer_start}")


if __name__ == "__main__":
    example_basic_usage()
    example_machine_monitoring()
    example_time_gaps()
    example_sensor_data()
    example_edge_cases()
    example_using_event_properties()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
