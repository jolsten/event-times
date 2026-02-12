"""
Unit tests for events_from_state function.
"""

import numpy as np
import pytest

from event_times import events_from_state


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_arrays_return_empty_list(self):
        """Test that empty arrays return an empty list."""
        result = events_from_state(
            np.array([], dtype="datetime64"), np.array([], dtype=bool), max_gap=60.0
        )
        assert result == []

    def test_mismatched_lengths_raise_error(self):
        """Test that mismatched array lengths raise ValueError."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        states = np.array([True])
        with pytest.raises(ValueError, match="same length"):
            events_from_state(timestamps, states, max_gap=60.0)

    def test_unsorted_timestamps_raise_error(self):
        """Test that unsorted timestamps raise ValueError."""
        timestamps = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:01:00", "2024-01-01T00:03:00"],
            dtype="datetime64",
        )
        states = np.array([True, False, True])
        with pytest.raises(ValueError, match="sorted"):
            events_from_state(timestamps, states, max_gap=60.0)

    def test_single_element_arrays(self):
        """Test handling of single-element arrays."""
        timestamps = np.array(["2024-01-01T00:00:00"], dtype="datetime64")
        states = np.array([True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min is None
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[0]
        assert events[0].stop_max is None


class TestBasicFunctionality:
    """Tests for basic events_from_state functionality."""

    def test_all_false_returns_empty_list(self):
        """Test that all False states return an empty list."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([False, False, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert events == []

    def test_all_true_returns_single_event(self):
        """Test that all True states return a single event with None boundaries."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([True, True, True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min is None
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[-1]
        assert events[0].stop_max is None

    def test_single_true_in_middle(self):
        """Test single True state surrounded by False states."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([False, True, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min == timestamps[0]
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[1]
        assert events[0].stop_max == timestamps[2]

    def test_starts_true_ends_false(self):
        """Test series starting with True and ending with False."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([True, True, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min is None
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[1]
        assert events[0].stop_max == timestamps[2]

    def test_starts_false_ends_true(self):
        """Test series starting with False and ending with True."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([False, True, True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min == timestamps[0]
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[2]
        assert events[0].stop_max is None

    def test_multiple_events(self):
        """Test multiple distinct on/off events."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
            ],
            dtype="datetime64",
        )
        states = np.array([False, True, False, True, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 2

        # First event
        assert events[0].start_min == timestamps[0]
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[1]
        assert events[0].stop_max == timestamps[2]

        # Second event
        assert events[1].start_min == timestamps[2]
        assert events[1].start_max == timestamps[3]
        assert events[1].stop_min == timestamps[3]
        assert events[1].stop_max == timestamps[4]


class TestTimeGapHandling:
    """Tests for time gap handling."""

    def test_large_gap_splits_on_event(self):
        """Test that large time gaps split on events."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:05:00",  # 4-minute gap (exceeds 60s default)
                "2024-01-01T00:06:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, True, True, True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 2

        # First event: starts at series start (start_min=None), split by gap (stop_max=None)
        assert events[0].start_min is None
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[1]
        assert events[0].stop_max is None

        # Second event: after gap (start_min=None), ends at series end (stop_max=None)
        assert events[1].start_min is None
        assert events[1].start_max == timestamps[2]
        assert events[1].stop_min == timestamps[3]
        assert events[1].stop_max is None

    def test_custom_max_time_gap(self):
        """Test using a custom max_gap parameter."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:03:00",  # 2-minute gap
                "2024-01-01T00:04:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, True, True, True])

        # With 3-minute threshold, should be one event
        events = events_from_state(timestamps, states, max_gap=3 * 60.0)
        assert len(events) == 1

        # With 1-minute threshold, should be two events
        events = events_from_state(timestamps, states, max_gap=1 * 60.0)
        assert len(events) == 2

    def test_gap_during_off_period(self):
        """Test that a gap during off period creates two segments."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:10:00",  # Large gap while False
                "2024-01-01T00:11:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, False, False, True])
        events = events_from_state(timestamps, states, max_gap=60.0)

        assert len(events) == 2
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_max == timestamps[1]
        # After the gap, the off sample within the new segment is used as start_min
        assert events[1].start_min == timestamps[2]
        assert events[1].start_max == timestamps[3]
        assert events[1].stop_max is None

    def test_exact_threshold_gap(self):
        """Test behavior at exactly the threshold."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([True, True, True])

        # Gap is exactly 60 seconds (at threshold, not exceeded)
        # Using > comparison, so 60s gap with 60s threshold should NOT split
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1

        # Gap is 60 seconds but threshold is 59 seconds (gap exceeds threshold)
        # Both gaps exceed threshold, so we get 3 separate events
        events = events_from_state(timestamps, states, max_gap=59.0)
        assert len(events) == 3  # Each timestamp becomes its own event

    def test_threshold_comparison_is_strict_greater_than(self):
        """Test that gap splitting uses strict > comparison (not >=)."""
        # Create timestamps with exactly 60-second gaps
        base = np.datetime64("2024-01-01T00:00:00")
        timestamps = np.array(
            [base, base + np.timedelta64(60, "s"), base + np.timedelta64(120, "s")]
        )
        states = np.array([True, True, True])

        # With threshold = 60s, gaps of exactly 60s should NOT cause splits
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1, "Gaps equal to threshold should not split events"

        # With threshold = 61s, gaps of 60s should NOT cause splits
        events = events_from_state(timestamps, states, max_gap=61.0)
        assert len(events) == 1

        # With threshold = 59s, gaps of 60s SHOULD cause splits
        events = events_from_state(timestamps, states, max_gap=59.0)
        assert len(events) == 3, "Gaps exceeding threshold should split events"

    def test_gap_split_events_have_none_boundaries(self):
        """Test that events split by time gaps have None for gap-side boundaries."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:05:00",  # Large gap, still True
                "2024-01-01T00:06:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, True, True, True])

        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 2

        # First event: starts at beginning (start_min=None), split by gap (stop_max=None)
        assert events[0].start_min is None
        assert events[0].stop_max is None

        # Second event: split by gap (start_min=None), ends at end (stop_max=None)
        assert events[1].start_min is None
        assert events[1].stop_max is None

        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[1]
        assert events[1].start_max == timestamps[2]
        assert events[1].stop_min == timestamps[3]


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_alternating_single_states(self):
        """Test alternating single True/False states."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, False, True, False, True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 3

        for event in events:
            assert event.start_max == event.stop_min  # Single-timestamp events

    def test_long_on_period(self):
        """Test a long continuous on period."""
        timestamps = np.array(
            [f"2024-01-01T00:{i:02d}:00" for i in range(10)], dtype="datetime64"
        )
        states = np.array([False] + [True] * 8 + [False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[8]

    def test_equal_first_and_stop_min(self):
        """Test case where start_max equals stop_min (single True timestamp)."""
        timestamps = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        states = np.array([False, True, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_max == events[0].stop_min == timestamps[1]

    def test_multiple_consecutive_true_at_start(self):
        """Test multiple True states at the beginning."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
            ],
            dtype="datetime64",
        )
        states = np.array([True, True, True, False])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min is None
        assert events[0].start_max == timestamps[0]
        assert events[0].stop_min == timestamps[2]
        assert events[0].stop_max == timestamps[3]

    def test_multiple_consecutive_true_at_end(self):
        """Test multiple True states at the end."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
            ],
            dtype="datetime64",
        )
        states = np.array([False, True, True, True])
        events = events_from_state(timestamps, states, max_gap=60.0)
        assert len(events) == 1
        assert events[0].start_min == timestamps[0]
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[3]
        assert events[0].stop_max is None


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_mixed_events_with_gaps(self):
        """Test complex scenario with multiple events and gaps."""
        timestamps = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:10:00",  # Large gap
                "2024-01-01T00:11:00",
                "2024-01-01T00:12:00",
            ],
            dtype="datetime64",
        )
        states = np.array([False, True, True, False, True, True, False])
        events = events_from_state(timestamps, states, max_gap=5 * 60.0)

        assert len(events) == 2

        # First event
        assert events[0].start_min == timestamps[0]
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[2]
        assert events[0].stop_max == timestamps[3]

        # Second event (after gap — start_min is None because segment starts on)
        assert events[1].start_min is None
        assert events[1].start_max == timestamps[4]
        assert events[1].stop_min == timestamps[5]
        assert events[1].stop_max == timestamps[6]

    def test_very_short_timestamps(self):
        """Test with timestamps in milliseconds."""
        base = np.datetime64("2024-01-01T00:00:00.000")
        timestamps = base + np.arange(5) * np.timedelta64(100, "ms")
        states = np.array([False, True, True, True, False])
        events = events_from_state(
            timestamps, states, max_gap=0.2
        )  # 200 milliseconds = 0.2 seconds

        assert len(events) == 1
        assert events[0].start_max == timestamps[1]
        assert events[0].stop_min == timestamps[3]
