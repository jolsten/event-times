"""
Core functionality for onofftimes package.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Event:
    """Represents a single on/off event with timing information.

    Attributes:
        last_off: The last timestamp before the event turned on (None if series starts on).
        first_on: The first timestamp when the event turned on.
        last_on: The last timestamp when the event was on.
        first_off: The first timestamp when the event turned off (None if series ends on).

    Raises:
        ValueError: If both last_off and first_on are None, or if both last_on and first_off are None.
    """

    last_off: Optional[np.datetime64]
    first_on: Optional[np.datetime64]
    last_on: Optional[np.datetime64]
    first_off: Optional[np.datetime64]

    def __post_init__(self):
        """Validate that the event has meaningful start and stop information."""
        if self.last_off is None and self.first_on is None:
            raise ValueError(
                "Invalid Event: at least one of last_off or first_on must be not None"
            )
        if self.last_on is None and self.first_off is None:
            raise ValueError(
                "Invalid Event: at least one of last_on or first_off must be not None"
            )

    @property
    def inner_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]:
        """Returns the interval where the state is definitely on.

        Returns:
            Tuple of (first_on, last_on).
        """
        return (self.first_on, self.last_on)

    @property
    def outer_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]:
        """Returns the interval spanning from last off to first off.

        Returns:
            Tuple of (last_off, first_off).
        """
        return (self.last_off, self.first_off)

    @property
    def start(self) -> np.datetime64:
        """Returns the start time of the event.

        Returns first_on if available, otherwise last_off.

        Returns:
            The start timestamp of the event (guaranteed non-None due to validation).
        """
        if self.first_on is not None:
            return self.first_on
        # Due to __post_init__ validation, if first_on is None, last_off must not be None
        assert self.last_off is not None
        return self.last_off

    @property
    def stop(self) -> np.datetime64:
        """Returns the stop time of the event.

        Returns last_on if available, otherwise first_off.

        Returns:
            The stop timestamp of the event (guaranteed non-None due to validation).
        """
        if self.last_on is not None:
            return self.last_on
        # Due to __post_init__ validation, if last_on is None, first_off must not be None
        assert self.first_off is not None
        return self.first_off

    @property
    def duration(self) -> float:
        """Returns the duration of the event in seconds.

        Based on the inner interval (first_on to last_on).

        Returns:
            Duration in seconds as a float. Returns 0 if inner interval cannot be computed.
        """
        if self.first_on is not None and self.last_on is not None:
            delta = self.last_on - self.first_on
            # Convert timedelta64 to seconds
            return float(delta / np.timedelta64(1, "s"))
        return 0.0


def on_off_times(
    timestamps: np.ndarray, states: Optional[np.ndarray] = None, max_gap: float = 60.0
) -> List[Event]:
    """Convert a time series of boolean states to a summary of start and stop times.

    This function analyzes a time series of boolean states and identifies distinct
    "on" events, including their boundaries. Events are split if there is a time
    gap larger than max_gap between consecutive timestamps.

    Args:
        timestamps: Array of datetime64 timestamps (must be sorted).
        states: Array of boolean states corresponding to each timestamp. If None,
            all timestamps are assumed to represent ON times (all True).
        max_gap: Maximum allowed time gap in seconds between consecutive samples.
            Events spanning gaps larger than this are treated as independent events.
            Default is 60.0 seconds.

    Returns:
        List of Event objects, each containing:
            - last_off: Last timestamp before turning on (None if starts on)
            - first_on: First timestamp when turned on
            - last_on: Last timestamp when on
            - first_off: First timestamp after turning off (None if ends on)

    Raises:
        ValueError: If arrays are empty, have mismatched lengths, or timestamps
            are not sorted in ascending order.

    Examples:
        >>> timestamps = np.array(['2024-01-01T00:00:00', '2024-01-01T00:01:00',
        ...                        '2024-01-01T00:02:00'], dtype='datetime64')
        >>> states = np.array([False, True, True])
        >>> events = on_off_times(timestamps, states)
        >>> events[0].first_on
        numpy.datetime64('2024-01-01T00:01:00')

        >>> # Without states - all timestamps are ON times
        >>> on_times = np.array(['2024-01-01T00:01:00',
        ...                      '2024-01-01T00:02:00'], dtype='datetime64')
        >>> events = on_off_times(on_times)
        >>> len(events)
        1

        >>> # Custom gap threshold (5 minutes)
        >>> events = on_off_times(timestamps, states, max_gap=300.0)
    """
    # Input validation
    if len(timestamps) == 0:
        raise ValueError("Input arrays cannot be empty")

    # If states not provided, assume all timestamps are ON times
    if states is None:
        states = np.ones(len(timestamps), dtype=bool)

    if len(timestamps) != len(states):
        raise ValueError(
            f"Array length mismatch: timestamps has {len(timestamps)} elements, "
            f"states has {len(states)} elements"
        )

    # Check if timestamps are sorted
    if len(timestamps) > 1 and not np.all(timestamps[1:] >= timestamps[:-1]):
        raise ValueError("Timestamps must be sorted in ascending order")

    # Convert max_gap from seconds to timedelta64
    max_gap = np.timedelta64(
        int(max_gap * 1_000_000), "us"
    )  # microseconds for precision

    # Convert max_gap to timedelta64 if it's not already
    if not isinstance(max_gap, np.timedelta64):
        max_gap = np.timedelta64(int(max_gap), "s")

    # Convert states to boolean array if needed
    states = np.asarray(states, dtype=bool)
    timestamps = np.asarray(timestamps, dtype="datetime64")

    # Handle all False case
    if not np.any(states):
        return []

    events = []

    # Find indices where state is True
    on_indices = np.where(states)[0]

    # Find gaps in the time series (between consecutive timestamps)
    time_diffs = np.diff(timestamps)
    gap_indices = np.where(time_diffs > max_gap)[0]

    # Split on_indices into groups based on:
    # 1. State transitions (gaps in on_indices sequence)
    # 2. Time gaps that exceed max_gap

    current_event_indices = []

    for i, idx in enumerate(on_indices):
        # Check if this is the start of a new event
        start_new_event = False

        if i == 0:
            # First on index always starts a new event
            current_event_indices = [idx]
        else:
            prev_idx = on_indices[i - 1]

            # Check if there's a gap in the on_indices (state went False in between)
            if idx != prev_idx + 1:
                start_new_event = True
            # Check if there's a large time gap between this and previous on timestamp
            elif prev_idx in gap_indices:
                start_new_event = True

            if start_new_event:
                # Process the completed event
                events.append(_create_event(current_event_indices, timestamps, states))
                current_event_indices = [idx]
            else:
                current_event_indices.append(idx)

        # If this is the last on_index, close the current event
        if i == len(on_indices) - 1:
            events.append(_create_event(current_event_indices, timestamps, states))

    return events


def _create_event(
    on_indices: List[int], timestamps: np.ndarray, states: np.ndarray
) -> Event:
    """Create an Event object from a group of consecutive on indices.

    Args:
        on_indices: List of indices where state is True.
        timestamps: Full array of timestamps.
        states: Full array of states.

    Returns:
        Event object with appropriate boundary times.
    """
    first_on_idx = on_indices[0]
    last_on_idx = on_indices[-1]

    # Determine last_off (last False before first_on)
    # Only set if there's a previous timestamp AND it's in the False state
    if first_on_idx == 0:
        last_off = None
    elif not states[first_on_idx - 1]:
        last_off = timestamps[first_on_idx - 1]
    else:
        # Previous timestamp is True (event was split by time gap)
        last_off = None

    # Determine first_off (first False after last_on)
    # Only set if there's a next timestamp AND it's in the False state
    if last_on_idx == len(timestamps) - 1:
        first_off = None
    elif not states[last_on_idx + 1]:
        first_off = timestamps[last_on_idx + 1]
    else:
        # Next timestamp is True (event was split by time gap)
        first_off = None

    return Event(
        last_off=last_off,
        first_on=timestamps[first_on_idx],
        last_on=timestamps[last_on_idx],
        first_off=first_off,
    )
