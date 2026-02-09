"""Core functionality for event-times package."""

from typing import Optional, cast

import numpy as np

from event_times.event import Event


def events_from_state(
    time: np.ndarray,
    state: np.ndarray,
    max_gap: float = 60.0,
) -> list[Event]:
    """Convert time series state data into a list of Event objects.

    Scans a boolean state time series and identifies contiguous "on" periods,
    mapping each run to an Event with appropriate boundary timestamps.

    Data gaps larger than *max_gap* act as hard breaks: events cannot span them,
    and boundaries at gap edges are set to None.

    Args:
        time: 1-D array of sample timestamps (np.datetime64, sorted ascending).
        state: 1-D boolean array; True means the event is occurring at that sample.
        max_gap: Maximum allowed gap in seconds between consecutive samples.
            Gaps larger than this value split the series into independent segments.

    Returns:
        List of Event objects, one per contiguous on-period. Returns an empty list
        if the input is empty or no on-periods are found.

    Raises:
        ValueError: If time and state have different lengths, or if time is not
            sorted in ascending order.
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    state = np.asarray(state, dtype=bool)

    if len(time) == 0:
        return []

    if len(time) != len(state):
        raise ValueError(
            f"time and state must have the same length ({len(time)} vs {len(state)})"
        )

    if len(time) > 1 and not np.all(time[1:] >= time[:-1]):
        raise ValueError("time must be sorted in ascending order")

    if not np.any(state):
        return []

    n = len(time)
    max_gap_td = np.timedelta64(round(max_gap * 1_000_000_000), "ns")

    # Identify large-gap positions: gap_after[i] is True when the gap between
    # time[i] and time[i+1] exceeds max_gap.
    if n > 1:
        gap_after = np.diff(time) > max_gap_td
    else:
        gap_after = np.zeros(0, dtype=bool)

    # Compute segment boundaries. Each segment is a contiguous block of samples
    # with no large gaps.
    seg_starts = np.concatenate([[0], np.where(gap_after)[0] + 1])
    seg_ends = np.concatenate([np.where(gap_after)[0], [n - 1]])

    events: list[Event] = []

    for seg_start, seg_end in zip(seg_starts, seg_ends):
        seg_time = time[seg_start : seg_end + 1]
        seg_state = state[seg_start : seg_end + 1]

        # Pad with False on both sides to capture runs at segment boundaries.
        padded = np.concatenate([[False], seg_state, [False]])
        d = np.diff(padded.astype(np.int8))
        run_starts = np.where(d == 1)[0]  # local index of first on-sample
        run_ends = np.where(d == -1)[0] - 1  # local index of last on-sample

        seg_len = len(seg_time)
        for rs, re in zip(run_starts, run_ends):
            last_off: Optional[np.datetime64] = (
                None if rs == 0 else cast(np.datetime64, seg_time[rs - 1])
            )
            first_off: Optional[np.datetime64] = (
                None if re == seg_len - 1 else cast(np.datetime64, seg_time[re + 1])
            )
            events.append(
                Event(
                    last_off=last_off,
                    first_on=cast(np.datetime64, seg_time[rs]),
                    last_on=cast(np.datetime64, seg_time[re]),
                    first_off=first_off,
                )
            )

    return events
