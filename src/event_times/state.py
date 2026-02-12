"""Stateful batch processor for converting boolean state time series into Events."""

from typing import Optional, cast

import numpy as np
import numpy.typing as npt

from event_times.event import Event


class OnOffStateProcessor:
    """Stateful batch processor for converting boolean state time series into Events.

    Processes batches of (time, state) data via ``__call__``.  Events that span
    batch boundaries are correctly handled by maintaining internal state between
    calls.  Completed events accumulate internally and are drained via
    ``get_events()``.

    Args:
        description: Description applied to all generated events.
        color: Color applied to all generated events.
        max_gap: Maximum allowed gap in seconds between consecutive samples.
            Gaps larger than this split the series into independent segments.
    """

    def __init__(
        self,
        description: Optional[str] = None,
        color: Optional[str] = None,
        max_gap: float = 60.0,
    ) -> None:
        self.description = description
        self.color = color
        self.max_gap = max_gap

        self._events: list[Event] = []
        self._pending: Optional[dict[str, Optional[np.datetime64]]] = None
        self._last_time: Optional[np.datetime64] = None
        self._last_state: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        time: npt.NDArray[np.datetime64],
        state: npt.NDArray[np.bool_],
    ) -> None:
        """Process one batch of time-series state data.

        Args:
            time: 1-D array of sample timestamps (sorted ascending).
            state: 1-D boolean array; True means the event is occurring.

        Raises:
            ValueError: If *time* and *state* differ in length or *time* is
                not sorted ascending.
        """
        time = np.asarray(time, dtype="datetime64[ns]")
        state = np.asarray(state, dtype=bool)

        if len(time) == 0:
            return

        if len(time) != len(state):
            raise ValueError(
                f"time and state must have the same length "
                f"({len(time)} vs {len(state)})"
            )

        if len(time) > 1 and not np.all(time[1:] >= time[:-1]):
            raise ValueError("time must be sorted in ascending order")

        max_gap_td = np.timedelta64(round(self.max_gap * 1_000_000_000), "ns")

        # --- Cross-batch boundary handling --------------------------------
        if self._last_time is not None:
            gap = time[0] - self._last_time
            is_large_gap = gap > max_gap_td

            if self._pending is not None:
                if is_large_gap:
                    self._close_pending(stop_max=None)
                elif not state[0]:
                    self._close_pending(stop_max=cast(np.datetime64, time[0]))
                # else: small gap & state[0] is True → pending stays open
        else:
            is_large_gap = True  # no previous batch → treat as discontinuity

        # --- Segment by internal large gaps --------------------------------
        n = len(time)
        if n > 1:
            gap_after = np.diff(time) > max_gap_td
        else:
            gap_after = np.zeros(0, dtype=bool)

        seg_starts = np.concatenate([[0], np.where(gap_after)[0] + 1])
        seg_ends = np.concatenate([np.where(gap_after)[0], [n - 1]])

        # --- Process each gap-free segment --------------------------------
        for seg_idx in range(len(seg_starts)):
            seg_s = int(seg_starts[seg_idx])
            seg_e = int(seg_ends[seg_idx])

            # Close pending on internal segment boundaries
            if seg_idx > 0 and self._pending is not None:
                self._close_pending(stop_max=None)

            seg_time = time[seg_s : seg_e + 1]
            seg_state = state[seg_s : seg_e + 1]

            # Detect runs via pad-and-diff
            padded = np.concatenate([[False], seg_state, [False]])
            d = np.diff(padded.astype(np.int8))
            run_starts = np.where(d == 1)[0]
            run_ends = np.where(d == -1)[0] - 1

            seg_len = len(seg_time)

            for rs, re in zip(run_starts, run_ends):
                starts_at_edge = rs == 0
                ends_at_edge = re == seg_len - 1

                # --- Run starts at the left edge of the segment -----------
                if starts_at_edge:
                    if self._pending is not None:
                        # Continue the pending event
                        self._pending["stop_min"] = cast(np.datetime64, seg_time[re])
                        if not ends_at_edge:
                            self._close_pending(
                                stop_max=cast(np.datetime64, seg_time[re + 1])
                            )
                        continue

                    # No pending – determine start_min from cross-batch info
                    if seg_idx == 0 and not is_large_gap:
                        start_min = self._last_time
                    else:
                        start_min = None
                else:
                    start_min = cast(np.datetime64, seg_time[rs - 1])

                start_max = cast(np.datetime64, seg_time[rs])
                stop_min = cast(np.datetime64, seg_time[re])

                if ends_at_edge:
                    # Run reaches the right edge – keep as pending
                    self._pending = {
                        "start_min": start_min,
                        "start_max": start_max,
                        "stop_min": stop_min,
                    }
                else:
                    stop_max = cast(np.datetime64, seg_time[re + 1])
                    self._make_event(start_min, start_max, stop_min, stop_max)

        # --- Update cross-batch tracking ----------------------------------
        self._last_time = cast(np.datetime64, time[-1])
        self._last_state = bool(state[-1])

    def get_events(self) -> list[Event]:
        """Return accumulated completed events and clear the internal buffer."""
        events = self._events
        self._events = []
        return events

    def finalize(self) -> None:
        """Close any dangling pending event.

        Call after the last batch has been processed.  If the state was True at
        the end of the final batch, the pending event is closed with
        ``stop_max=None``.
        """
        self._close_pending(stop_max=None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_pending(self, stop_max: Optional[np.datetime64]) -> None:
        """Close the pending event with the given *stop_max* and store it."""
        if self._pending is None:
            return
        self._make_event(
            start_min=self._pending["start_min"],
            start_max=cast(np.datetime64, self._pending["start_max"]),
            stop_min=cast(np.datetime64, self._pending["stop_min"]),
            stop_max=stop_max,
        )
        self._pending = None

    def _make_event(
        self,
        start_min: Optional[np.datetime64],
        start_max: np.datetime64,
        stop_min: np.datetime64,
        stop_max: Optional[np.datetime64],
    ) -> None:
        """Create an Event with instance properties and append to buffer."""
        self._events.append(
            Event(
                start_min=start_min,
                start_max=start_max,
                stop_min=stop_min,
                stop_max=stop_max,
                description=self.description,
                color=self.color,
            )
        )
