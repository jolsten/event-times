"""Unit tests for OnOffStateProcessor."""

import numpy as np
import pytest

from event_times import OnOffStateProcessor, events_from_state


def _ts(spec: str) -> np.datetime64:
    """Shorthand for creating a datetime64 timestamp."""
    return np.datetime64(spec, "ns")


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_array_is_noop(self):
        proc = OnOffStateProcessor()
        proc(np.array([], dtype="datetime64"), np.array([], dtype=bool))
        proc.finalize()
        assert proc.get_events() == []

    def test_mismatched_lengths_raise_error(self):
        proc = OnOffStateProcessor()
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state = np.array([True])
        with pytest.raises(ValueError, match="same length"):
            proc(time, state)

    def test_unsorted_timestamps_raise_error(self):
        proc = OnOffStateProcessor()
        time = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state = np.array([True, False])
        with pytest.raises(ValueError, match="sorted"):
            proc(time, state)


class TestSingleBatchEquivalence:
    """StateProcessor with a single batch + finalize should match events_from_state."""

    def test_all_false(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state = np.array([False, False, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        assert proc.get_events() == []

    def test_all_true(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state = np.array([True, True, True])

        expected = events_from_state(time, state, max_gap=60.0)
        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.start_min == e.start_min
            assert r.start_max == e.start_max
            assert r.stop_min == e.stop_min
            assert r.stop_max == e.stop_max

    def test_single_event_in_middle(self):
        time = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
            ],
            dtype="datetime64",
        )
        state = np.array([False, True, True, True, False])

        expected = events_from_state(time, state, max_gap=60.0)
        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min == expected[0].start_min
        assert result[0].start_max == expected[0].start_max
        assert result[0].stop_min == expected[0].stop_min
        assert result[0].stop_max == expected[0].stop_max

    def test_multiple_events(self):
        time = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:02:00",
                "2024-01-01T00:03:00",
                "2024-01-01T00:04:00",
                "2024-01-01T00:05:00",
                "2024-01-01T00:06:00",
            ],
            dtype="datetime64",
        )
        state = np.array([False, True, False, False, True, True, False])

        expected = events_from_state(time, state, max_gap=60.0)
        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.start_min == e.start_min
            assert r.start_max == e.start_max
            assert r.stop_min == e.stop_min
            assert r.stop_max == e.stop_max

    def test_single_batch_with_internal_gap(self):
        time = np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:01:00",
                "2024-01-01T00:05:00",  # 4-minute gap (> 2 min max_gap)
                "2024-01-01T00:06:00",
            ],
            dtype="datetime64",
        )
        state = np.array([True, True, True, True])

        expected = events_from_state(time, state, max_gap=120.0)
        proc = OnOffStateProcessor(max_gap=120.0)
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.start_min == e.start_min
            assert r.start_max == e.start_max
            assert r.stop_min == e.stop_min
            assert r.stop_max == e.stop_max


class TestCrossBatchContinuation:
    """Tests for events spanning batch boundaries."""

    def test_event_spans_two_batches(self):
        """Batch1 ends True, Batch2 starts True, small gap → one event."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min == _ts("2024-01-01T00:00:00")
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:02:00")
        assert result[0].stop_max == _ts("2024-01-01T00:03:00")

    def test_event_spans_three_batches(self):
        """Event continues across three consecutive batches."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, True])

        time3 = np.array(
            ["2024-01-01T00:04:00", "2024-01-01T00:05:00"], dtype="datetime64"
        )
        state3 = np.array([True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc(time3, state3)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min == _ts("2024-01-01T00:00:00")
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:04:00")
        assert result[0].stop_max == _ts("2024-01-01T00:05:00")

    def test_pending_closed_by_false_start(self):
        """Batch1 ends True, Batch2 starts False, small gap → event closed."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([False, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min == _ts("2024-01-01T00:00:00")
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:01:00")
        assert result[0].stop_max == _ts("2024-01-01T00:02:00")

    def test_large_gap_between_batches(self):
        """Large gap breaks pending event, second batch starts fresh."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        time2 = np.array(
            ["2024-01-01T01:00:00", "2024-01-01T01:01:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 2
        # First event: closed with stop_max=None (gap boundary)
        assert result[0].start_min == _ts("2024-01-01T00:00:00")
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:01:00")
        assert result[0].stop_max is None
        # Second event: start_min=None (gap boundary)
        assert result[1].start_min is None
        assert result[1].start_max == _ts("2024-01-01T01:00:00")
        assert result[1].stop_min == _ts("2024-01-01T01:00:00")
        assert result[1].stop_max == _ts("2024-01-01T01:01:00")


class TestCrossBatchLastOff:
    """Tests that start_min is correctly inherited from previous batch."""

    def test_start_min_from_previous_batch(self):
        """New True run at batch start gets start_min from previous batch's last sample."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([True, False])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 2
        # Second event should have start_min = last sample of batch1
        assert result[1].start_min == _ts("2024-01-01T00:01:00")

    def test_start_min_none_after_large_gap(self):
        """New run at batch start after large gap gets start_min=None."""
        time1 = np.array(["2024-01-01T00:00:00"], dtype="datetime64")
        state1 = np.array([False])

        time2 = np.array(
            ["2024-01-01T01:00:00", "2024-01-01T01:01:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min is None


class TestEventProperties:
    """Tests that description and color are passed to events."""

    def test_description_and_color(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state = np.array([False, True, False])

        proc = OnOffStateProcessor(
            description="test event", color="#ff0000", max_gap=60.0
        )
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].description == "test event"
        assert result[0].color == "#ff0000"

    def test_properties_on_cross_batch_event(self):
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc = OnOffStateProcessor(description="spanning", color="blue", max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].description == "spanning"
        assert result[0].color == "blue"

    def test_no_properties(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state = np.array([False, True, False])

        proc = OnOffStateProcessor()
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].description is None
        assert result[0].color is None


class TestDrainSemantics:
    """Tests for get_events() drain behavior."""

    def test_get_events_clears_buffer(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state = np.array([False, True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()

        first_call = proc.get_events()
        second_call = proc.get_events()

        assert len(first_call) == 1
        assert len(second_call) == 0

    def test_get_events_between_batches(self):
        """Events completed mid-stream can be drained before more batches."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
            dtype="datetime64",
        )
        state1 = np.array([False, True, False])

        time2 = np.array(
            ["2024-01-01T00:03:00", "2024-01-01T00:04:00", "2024-01-01T00:05:00"],
            dtype="datetime64",
        )
        state2 = np.array([False, True, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        batch1_events = proc.get_events()

        proc(time2, state2)
        proc.finalize()
        batch2_events = proc.get_events()

        assert len(batch1_events) == 1
        assert len(batch2_events) == 1


class TestFinalize:
    """Tests for finalize() behavior."""

    def test_finalize_no_pending(self):
        """Finalize with no pending event is a no-op."""
        proc = OnOffStateProcessor()
        proc.finalize()  # should not raise
        assert proc.get_events() == []

    def test_finalize_closes_pending(self):
        time = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state = np.array([False, True])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)

        # Before finalize, the event is pending (not in completed buffer)
        assert proc.get_events() == []

        proc.finalize()
        result = proc.get_events()
        assert len(result) == 1
        assert result[0].stop_max is None

    def test_processing_after_finalize(self):
        """Processing more batches after finalize works correctly."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc.finalize()
        events1 = proc.get_events()

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])

        proc(time2, state2)
        proc.finalize()
        events2 = proc.get_events()

        assert len(events1) == 1
        assert len(events2) == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_true_sample(self):
        time = np.array(["2024-01-01T00:00:00"], dtype="datetime64")
        state = np.array([True])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_min is None
        assert result[0].start_max == _ts("2024-01-01T00:00:00")
        assert result[0].stop_min == _ts("2024-01-01T00:00:00")
        assert result[0].stop_max is None

    def test_single_false_sample(self):
        time = np.array(["2024-01-01T00:00:00"], dtype="datetime64")
        state = np.array([False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time, state)
        proc.finalize()
        assert proc.get_events() == []

    def test_all_false_batch_closes_pending(self):
        """All-False batch after a pending event closes it."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([True, True])

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([False, False])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].stop_max == _ts("2024-01-01T00:02:00")

    def test_alternating_single_sample_batches(self):
        """Each batch is a single sample, alternating True/False."""
        proc = OnOffStateProcessor(max_gap=60.0)

        proc(np.array(["2024-01-01T00:00:00"], dtype="datetime64"), np.array([False]))
        proc(np.array(["2024-01-01T00:01:00"], dtype="datetime64"), np.array([True]))
        proc(np.array(["2024-01-01T00:02:00"], dtype="datetime64"), np.array([False]))
        proc(np.array(["2024-01-01T00:03:00"], dtype="datetime64"), np.array([True]))
        proc(np.array(["2024-01-01T00:04:00"], dtype="datetime64"), np.array([False]))
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 2
        # First event
        assert result[0].start_min == _ts("2024-01-01T00:00:00")
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:01:00")
        assert result[0].stop_max == _ts("2024-01-01T00:02:00")
        # Second event
        assert result[1].start_min == _ts("2024-01-01T00:02:00")
        assert result[1].start_max == _ts("2024-01-01T00:03:00")
        assert result[1].stop_min == _ts("2024-01-01T00:03:00")
        assert result[1].stop_max == _ts("2024-01-01T00:04:00")

    def test_empty_batch_between_batches(self):
        """Empty batch does not disrupt state tracking."""
        time1 = np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64"
        )
        state1 = np.array([False, True])

        proc = OnOffStateProcessor(max_gap=60.0)
        proc(time1, state1)
        proc(np.array([], dtype="datetime64"), np.array([], dtype=bool))

        time2 = np.array(
            ["2024-01-01T00:02:00", "2024-01-01T00:03:00"], dtype="datetime64"
        )
        state2 = np.array([True, False])
        proc(time2, state2)
        proc.finalize()
        result = proc.get_events()

        assert len(result) == 1
        assert result[0].start_max == _ts("2024-01-01T00:01:00")
        assert result[0].stop_min == _ts("2024-01-01T00:02:00")
