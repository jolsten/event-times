"""
Property-based tests using hypothesis for onofftimes package.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from event_times import Event, on_off_times


# Custom strategies for datetime64 arrays
@st.composite
def sorted_datetime_arrays(draw, min_size=1, max_size=100):
    """Generate sorted datetime64 arrays."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate random timestamps
    base_time = np.datetime64("2024-01-01T00:00:00")
    # Generate increasing time deltas in seconds
    deltas = draw(
        st.lists(st.integers(min_value=1, max_value=3600), min_size=size, max_size=size)
    )

    # Create cumulative sum for sorted timestamps
    cumulative = np.cumsum([0] + deltas[:-1])
    timestamps = base_time + cumulative * np.timedelta64(1, "s")

    return timestamps


@st.composite
def datetime_state_pairs(draw, min_size=1, max_size=50):
    """Generate pairs of sorted timestamps and boolean states."""
    timestamps = draw(sorted_datetime_arrays(min_size=min_size, max_size=max_size))
    size = len(timestamps)
    states = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    return np.array(timestamps), np.array(states, dtype=bool)


class TestPropertyBasedValidation:
    """Property-based tests for input validation."""

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_mismatched_lengths_always_raise(self, size):
        """Test that mismatched lengths always raise ValueError."""
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps = base_time + np.arange(size) * np.timedelta64(1, "s")
        states = np.ones(size + 1, dtype=bool)

        with pytest.raises(ValueError, match="Array length mismatch"):
            on_off_times(timestamps, states)

    @given(st.integers(min_value=2, max_value=50))
    @settings(max_examples=50)
    def test_unsorted_timestamps_always_raise(self, size):
        """Test that unsorted timestamps always raise ValueError."""
        # Create sorted then reverse to ensure unsorted
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps = (base_time + np.arange(size) * np.timedelta64(1, "s"))[::-1]
        states = np.ones(size, dtype=bool)

        with pytest.raises(ValueError, match="Timestamps must be sorted"):
            on_off_times(timestamps, states)


class TestPropertyBasedOutputStructure:
    """Property-based tests for output structure and consistency."""

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_output_is_always_list_of_events(self, data):
        """Test that output is always a list of Event objects."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, Event)

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_events_have_valid_temporal_ordering(self, data):
        """Test that event timestamps are temporally ordered."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            # If both are not None, first_on should come before or equal to last_on
            if event.first_on is not None and event.last_on is not None:
                assert event.first_on <= event.last_on

            # If all four are present, check full ordering
            if (
                event.last_off is not None
                and event.first_on is not None
                and event.last_on is not None
                and event.first_off is not None
            ):
                # Now Pylance knows these are not None
                assert event.last_off < event.first_on
                assert event.first_on <= event.last_on
                assert event.last_on < event.first_off

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_event_timestamps_are_in_input_range(self, data):
        """Test that all event timestamps are from the input timestamps."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        timestamp_set = set(timestamps)

        for event in events:
            if event.last_off is not None:
                assert event.last_off in timestamp_set
            if event.first_on is not None:
                assert event.first_on in timestamp_set
            if event.last_on is not None:
                assert event.last_on in timestamp_set
            if event.first_off is not None:
                assert event.first_off in timestamp_set

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_no_overlapping_events(self, data):
        """Test that events don't overlap in time."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        if len(events) <= 1:
            return  # Nothing to check

        for i in range(len(events) - 1):
            curr_event = events[i]
            next_event = events[i + 1]

            # Current event's last_on should come before next event's first_on
            if curr_event.last_on is not None and next_event.first_on is not None:
                assert curr_event.last_on < next_event.first_on


class TestPropertyBasedStateConsistency:
    """Property-based tests for state consistency."""

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_all_false_always_returns_empty(self, size):
        """Test that all False states always return empty list."""
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps = base_time + np.arange(size) * np.timedelta64(1, "s")
        states = np.zeros(size, dtype=bool)

        events = on_off_times(timestamps, states)
        assert events == []

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=50)
    def test_all_true_always_returns_single_event(self, size):
        """Test that all True states always return single event."""
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps = base_time + np.arange(size) * np.timedelta64(1, "s")
        states = np.ones(size, dtype=bool)

        events = on_off_times(timestamps, states)
        assert len(events) == 1
        assert events[0].last_off is None
        assert events[0].first_off is None
        assert events[0].first_on == timestamps[0]
        assert events[0].last_on == timestamps[-1]

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_at_least_one_true_produces_events(self, data):
        """Test that if there's at least one True, we get events."""
        timestamps, states = data
        assume(np.any(states))  # Only test when there's at least one True

        events = on_off_times(timestamps, states)
        assert len(events) > 0

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_first_on_corresponds_to_true_state(self, data):
        """Test that first_on timestamps correspond to True states."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.first_on is not None:
                idx = np.where(timestamps == event.first_on)[0][0]
                assert states[idx]

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_last_on_corresponds_to_true_state(self, data):
        """Test that last_on timestamps correspond to True states."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.last_on is not None:
                idx = np.where(timestamps == event.last_on)[0][0]
                assert states[idx]

    @given(datetime_state_pairs(min_size=2, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_last_off_corresponds_to_false_state(self, data):
        """Test that last_off timestamps correspond to False states."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.last_off is not None:
                idx = np.where(timestamps == event.last_off)[0][0]
                assert not states[idx]

    @given(datetime_state_pairs(min_size=2, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_first_off_corresponds_to_false_state(self, data):
        """Test that first_off timestamps correspond to False states."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.first_off is not None:
                idx = np.where(timestamps == event.first_off)[0][0]
                assert not states[idx]


class TestPropertyBasedEventProperties:
    """Property-based tests for Event properties."""

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_start_property_consistency(self, data):
        """Test that start property is consistent with its definition."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.first_on is not None:
                assert event.start == event.first_on
            else:
                assert event.start == event.last_off

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_stop_property_consistency(self, data):
        """Test that stop property is consistent with its definition."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            if event.last_on is not None:
                assert event.stop == event.last_on
            else:
                assert event.stop == event.first_off

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_inner_interval_is_subset_of_outer(self, data):
        """Test that inner interval is within outer interval."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        for event in events:
            inner = event.inner_interval
            outer = event.outer_interval

            # If all components exist, inner should be within outer
            if (
                outer[0] is not None
                and inner[0] is not None
                and inner[1] is not None
                and outer[1] is not None
            ):
                # Now Pylance knows these are not None
                assert outer[0] <= inner[0]
                assert inner[1] <= outer[1]


class TestPropertyBasedTimeGaps:
    """Property-based tests for time gap handling."""

    @given(
        st.integers(min_value=2, max_value=20), st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, deadline=1000)
    def test_large_gaps_create_separate_events(self, size, gap_seconds):
        """Test that large gaps create separate events."""
        # Create timestamps with a large gap in the middle
        half = size // 2
        base_time = np.datetime64("2024-01-01T00:00:00")
        timestamps1 = base_time + np.arange(half) * np.timedelta64(1, "s")
        timestamps2 = (
            timestamps1[-1]
            + np.timedelta64(gap_seconds * 10, "s")
            + np.arange(1, size - half + 1) * np.timedelta64(1, "s")
        )
        timestamps = np.concatenate([timestamps1, timestamps2])

        # All True states
        states = np.ones(size, dtype=bool)

        # Use gap threshold smaller than the actual gap
        events = on_off_times(timestamps, states, max_gap=float(gap_seconds))

        # Should split into at least 2 events
        assert len(events) >= 2

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_increasing_gap_threshold_reduces_or_maintains_events(self, data):
        """Test that increasing gap threshold doesn't increase event count."""
        timestamps, states = data
        assume(np.any(states))  # Need at least one True

        events_small = on_off_times(timestamps, states, max_gap=1.0)
        events_large = on_off_times(timestamps, states, max_gap=1000.0)

        # Larger threshold should result in same or fewer events
        assert len(events_large) <= len(events_small)


class TestPropertyBasedInvariants:
    """Property-based tests for system invariants."""

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_total_on_samples_equals_sum_of_event_on_samples(self, data):
        """Test that total True states equals sum across all events."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        total_true = np.sum(states)

        # Count True samples covered by events
        event_true_count = 0
        for event in events:
            if event.first_on is not None and event.last_on is not None:
                first_idx = np.where(timestamps == event.first_on)[0][0]
                last_idx = np.where(timestamps == event.last_on)[0][0]
                event_true_count += last_idx - first_idx + 1

        assert event_true_count == total_true

    @given(datetime_state_pairs(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=1000)
    def test_events_are_disjoint(self, data):
        """Test that events represent disjoint time periods."""
        timestamps, states = data
        events = on_off_times(timestamps, states)

        # Collect all indices covered by each event
        event_indices = []
        for event in events:
            if event.first_on is not None and event.last_on is not None:
                first_idx = np.where(timestamps == event.first_on)[0][0]
                last_idx = np.where(timestamps == event.last_on)[0][0]
                indices = set(range(first_idx, last_idx + 1))
                event_indices.append(indices)

        # Check that all index sets are disjoint
        for i in range(len(event_indices)):
            for j in range(i + 1, len(event_indices)):
                assert event_indices[i].isdisjoint(event_indices[j])
