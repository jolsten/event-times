"""Comprehensive test suite for Event class."""

import datetime

import numpy as np
import pytest

# Import the Event class from the event module
from event_times import Event


class TestEventCreation:
    """Test Event instantiation and validation."""

    def test_create_event_with_inner_interval(self):
        """Test creating event with first_on and last_on."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.duration == 3600.0

    def test_create_event_with_outer_interval(self):
        """Test creating event with last_off and first_off."""
        event = Event(
            last_off="2024-01-01T09:00:00",
            first_off="2024-01-01T12:00:00",
        )
        assert event.last_off is not None
        assert event.first_off is not None
        assert event.duration == 3 * 3600.0

    def test_create_event_with_all_boundaries(self):
        """Test creating event with all four boundaries."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.last_off is not None
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.first_off is not None

    def test_create_event_with_description_and_color(self):
        """Test creating event with metadata."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            description="Test event",
            color="#FF5733",
        )
        assert event.description == "Test event"
        assert event.color == "#FF5733"

    def test_create_event_with_numpy_datetime(self):
        """Test creating event with numpy datetime64 objects."""
        start = np.datetime64("2024-01-01T10:00:00", "ns")
        stop = np.datetime64("2024-01-01T11:00:00", "ns")
        event = Event(first_on=start, last_on=stop)
        assert event.first_on == start
        assert event.last_on == stop

    def test_create_event_with_python_datetime(self):
        """Test creating event with Python datetime objects."""
        start = datetime.datetime(2024, 1, 1, 10, 0, 0)
        stop = datetime.datetime(2024, 1, 1, 11, 0, 0)
        event = Event(first_on=start, last_on=stop)
        assert event.duration == 3600.0


class TestEventValidation:
    """Test Event validation rules."""

    def test_missing_start_times_raises_error(self):
        """Test that event without start times raises ValueError."""
        with pytest.raises(ValueError, match="at least one start time"):
            Event(last_on="2024-01-01T11:00:00")

    def test_missing_stop_times_raises_error(self):
        """Test that event without stop times raises ValueError."""
        with pytest.raises(ValueError, match="at least one stop time"):
            Event(first_on="2024-01-01T10:00:00")

    def test_negative_duration_raises_error(self):
        """Test that stop before start raises ValueError."""
        with pytest.raises(ValueError, match="first_on must precede or equal last_on"):
            Event(
                first_on="2024-01-01T11:00:00",
                last_on="2024-01-01T10:00:00",
            )

    def test_last_off_after_first_on_raises_error(self):
        """Test that last_off > first_on raises ValueError."""
        with pytest.raises(ValueError, match="last_off must precede or equal first_on"):
            Event(
                last_off="2024-01-01T10:30:00",
                first_on="2024-01-01T10:00:00",
                last_on="2024-01-01T11:00:00",
            )

    def test_first_on_after_last_on_raises_error(self):
        """Test that first_on > last_on raises ValueError."""
        with pytest.raises(ValueError, match="first_on must precede or equal last_on"):
            Event(
                first_on="2024-01-01T11:00:00",
                last_on="2024-01-01T10:00:00",
            )

    def test_last_on_after_first_off_raises_error(self):
        """Test that last_on > first_off raises ValueError."""
        with pytest.raises(ValueError, match="last_on must precede or equal first_off"):
            Event(
                first_on="2024-01-01T10:00:00",
                last_on="2024-01-01T11:30:00",
                first_off="2024-01-01T11:00:00",
            )

    def test_invalid_datetime_string_raises_error(self):
        """Test that invalid datetime string raises error."""
        with pytest.raises((ValueError, TypeError)):
            Event(first_on="not a date", last_on="2024-01-01T11:00:00")


class TestEventProperties:
    """Test Event properties and computed values."""

    def test_start_property_prefers_first_on(self):
        """Test that start prefers first_on over last_off."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.start == event.first_on

    def test_start_property_uses_last_off_when_first_on_missing(self):
        """Test that start uses last_off when first_on is None."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.start == event.last_off

    def test_stop_property_prefers_last_on(self):
        """Test that stop prefers last_on over first_off."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.stop == event.last_on

    def test_stop_property_uses_first_off_when_last_on_missing(self):
        """Test that stop uses first_off when last_on is None."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.stop == event.first_off

    def test_times_property(self):
        """Test that times returns all four boundaries."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        times = event.times
        assert times[0] == event.last_off
        assert times[1] == event.first_on
        assert times[2] == event.last_on
        assert times[3] == event.first_off

    def test_inner_interval_property(self):
        """Test inner_interval returns (first_on, last_on)."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        inner = event.inner_interval
        assert inner[0] == event.first_on
        assert inner[1] == event.last_on

    def test_outer_interval_property(self):
        """Test outer_interval returns (last_off, first_off)."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        outer = event.outer_interval
        assert outer[0] == event.last_off
        assert outer[1] == event.first_off

    def test_duration_in_seconds(self):
        """Test duration calculation in seconds."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:30:00",
        )
        assert event.duration == 5400.0  # 1.5 hours

    def test_get_duration_various_units(self):
        """Test get_duration with different units."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.get_duration("h") == 1.0
        assert event.get_duration("m") == 60.0
        assert event.get_duration("s") == 3600.0
        assert event.get_duration("ms") == 3600000.0

    def test_midpoint_property(self):
        """Test midpoint calculation."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T12:00:00",
        )
        expected_midpoint = np.datetime64("2024-01-01T11:00:00", "ns")
        assert event.midpoint == expected_midpoint

    def test_uncertainty_start(self):
        """Test start uncertainty calculation."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.uncertainty_start == 300.0  # 5 minutes

    def test_uncertainty_start_none_when_missing(self):
        """Test start uncertainty is None when boundaries missing."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.uncertainty_start is None

    def test_uncertainty_stop(self):
        """Test stop uncertainty calculation."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.uncertainty_stop == 300.0  # 5 minutes

    def test_uncertainty_stop_none_when_missing(self):
        """Test stop uncertainty is None when boundaries missing."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.uncertainty_stop is None

    def test_total_uncertainty(self):
        """Test total uncertainty calculation."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        assert event.total_uncertainty == 600.0  # 10 minutes total

    def test_total_uncertainty_partial(self):
        """Test total uncertainty with only one boundary."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.total_uncertainty == 300.0  # Only start uncertainty

    def test_is_point_event_true(self):
        """Test is_point_event for very short event."""
        event = Event(
            first_on="2024-01-01T10:00:00.000000",
            last_on="2024-01-01T10:00:00.000001",  # 1 microsecond
        )
        assert event.is_point_event()  # Pythonic: assert truthy value

    def test_is_point_event_false(self):
        """Test is_point_event for regular event."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert not event.is_point_event()  # Pythonic: assert falsy value

    def test_is_point_event_custom_threshold(self):
        """Test is_point_event with custom threshold."""
        event = Event(
            first_on="2024-01-01T10:00:00.000",
            last_on="2024-01-01T10:00:00.100",  # 100 milliseconds
        )
        # 100ms is not a point event with 1μs threshold
        assert not event.is_point_event()
        # But it is with 1 second threshold
        assert event.is_point_event(threshold=1.0)
        # And not with 10ms threshold
        assert not event.is_point_event(threshold=0.01)


class TestEventMethods:
    """Test Event methods."""

    def test_contains_timestamp_inside(self):
        """Test contains returns True for timestamp inside event."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        timestamp = np.datetime64("2024-01-01T10:30:00", "ns")
        assert event.contains(timestamp) is True

    def test_contains_timestamp_outside(self):
        """Test contains returns False for timestamp outside event."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        timestamp = np.datetime64("2024-01-01T12:00:00", "ns")
        assert event.contains(timestamp) is False

    def test_contains_timestamp_at_boundary(self):
        """Test contains returns True for timestamp at boundary."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.contains(event.start) is True
        assert event.contains(event.stop) is True

    def test_definitely_contains_inside_inner_interval(self):
        """Test definitely_contains for timestamp in inner interval."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        timestamp = np.datetime64("2024-01-01T10:30:00", "ns")
        assert event.definitely_contains(timestamp) is True

    def test_definitely_contains_in_uncertainty_region(self):
        """Test definitely_contains for timestamp in uncertainty region."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
        )
        timestamp = np.datetime64("2024-01-01T09:57:00", "ns")
        assert event.definitely_contains(timestamp) is False

    def test_definitely_contains_no_inner_interval(self):
        """Test definitely_contains when inner interval not defined."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_off="2024-01-01T11:05:00",
        )
        timestamp = np.datetime64("2024-01-01T10:30:00", "ns")
        assert event.definitely_contains(timestamp) is False

    def test_overlaps_true(self):
        """Test overlaps returns True for overlapping events."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
        )
        assert event1.overlaps(event2) is True
        assert event2.overlaps(event1) is True

    def test_overlaps_false(self):
        """Test overlaps returns False for non-overlapping events."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        assert event1.overlaps(event2) is False
        assert event2.overlaps(event1) is False

    def test_overlaps_touching_events(self):
        """Test overlaps for events that touch but don't overlap."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T11:00:00",
            last_on="2024-01-01T12:00:00",
        )
        # Should not overlap since one stops exactly when other starts
        assert event1.overlaps(event2) is False

    def test_gap_between_separated_events(self):
        """Test gap_between for separated events."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        assert event1.gap_between(event2) == 3600.0  # 1 hour gap
        assert event2.gap_between(event1) == 3600.0

    def test_gap_between_overlapping_events(self):
        """Test gap_between for overlapping events (negative gap)."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
        )
        gap = event1.gap_between(event2)
        assert gap < 0  # Negative indicates overlap
        assert abs(gap) == 1800.0  # 30 minutes overlap

    def test_gap_between_touching_events(self):
        """Test gap_between for events that touch."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T11:00:00",
            last_on="2024-01-01T12:00:00",
        )
        assert event1.gap_between(event2) == 0.0

    def test_merge_overlapping_events(self):
        """Test merging overlapping events."""
        event1 = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
            description="Event 1",
        )
        event2 = Event(
            last_off="2024-01-01T10:25:00",
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
            first_off="2024-01-01T11:35:00",
            description="Event 2",
        )
        merged = event1.merge(event2)
        assert merged is not None
        assert merged.last_off == event1.last_off  # Earlier
        assert merged.first_on == event1.first_on  # Earlier
        assert merged.last_on == event2.last_on  # Later
        assert merged.first_off == event2.first_off  # Later

    def test_merge_non_overlapping_events_returns_none(self):
        """Test merging non-overlapping events returns None."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        assert event1.merge(event2) is None

    def test_merge_with_custom_description(self):
        """Test merge with custom description."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            description="Event 1",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
            description="Event 2",
        )
        merged = event1.merge(event2, description="Merged event")
        assert merged is not None
        assert merged.description == "Merged event"

    def test_merge_with_custom_color(self):
        """Test merge with custom color."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            color="#FF0000",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
            color="#00FF00",
        )
        merged = event1.merge(event2, color="#0000FF")
        assert merged is not None
        assert merged.color == "#0000FF"

    def test_merge_preserves_first_color_by_default(self):
        """Test merge preserves first event's color by default."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            color="#FF0000",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
            color="#00FF00",
        )
        merged = event1.merge(event2)
        assert merged is not None
        assert merged.color == "#FF0000"

    def test_merge_asymmetric_boundaries(self):
        """Test merge when only one event has outer boundaries."""
        event1 = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T10:30:00",
            last_on="2024-01-01T11:30:00",
            first_off="2024-01-01T11:35:00",
        )
        merged = event1.merge(event2)
        assert merged is not None
        assert merged.last_off == event1.last_off
        assert merged.first_on == event1.first_on
        assert merged.last_on == event2.last_on
        assert merged.first_off == event2.first_off

    def test_overlaps_contained_event(self):
        """Test overlaps when one event is entirely within another."""
        outer = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T13:00:00",
        )
        inner = Event(
            first_on="2024-01-01T11:00:00",
            last_on="2024-01-01T12:00:00",
        )
        assert outer.overlaps(inner) is True
        assert inner.overlaps(outer) is True

    def test_gap_between_when_self_is_after_other(self):
        """Test gap_between when self comes after other."""
        event1 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event1.gap_between(event2) == 3600.0


class TestEventFromInterval:
    """Test Event.from_interval class method."""

    def test_from_interval_inner_default(self):
        """Test from_interval with default inner interval."""
        event = Event.from_interval(
            "2024-01-01T10:00:00",
            "2024-01-01T11:00:00",
        )
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.last_off is None
        assert event.first_off is None

    def test_from_interval_inner_explicit(self):
        """Test from_interval with explicit inner interval."""
        event = Event.from_interval(
            "2024-01-01T10:00:00",
            "2024-01-01T11:00:00",
            interval_type="inner",
        )
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.last_off is None
        assert event.first_off is None

    def test_from_interval_outer(self):
        """Test from_interval with outer interval."""
        event = Event.from_interval(
            "2024-01-01T09:55:00",
            "2024-01-01T11:05:00",
            interval_type="outer",
        )
        assert event.last_off is not None
        assert event.first_off is not None
        assert event.first_on is None
        assert event.last_on is None

    def test_from_interval_with_description_and_color(self):
        """Test from_interval with metadata."""
        event = Event.from_interval(
            "2024-01-01T10:00:00",
            "2024-01-01T11:00:00",
            description="Test event",
            color="#FF5733",
        )
        assert event.description == "Test event"
        assert event.color == "#FF5733"

    def test_from_interval_with_numpy_datetime(self):
        """Test from_interval with numpy datetime64 inputs."""
        start = np.datetime64("2024-01-01T10:00:00", "ns")
        stop = np.datetime64("2024-01-01T11:00:00", "ns")
        event = Event.from_interval(start, stop)
        assert event.first_on == start
        assert event.last_on == stop


class TestEventFromDuration:
    """Test Event.from_duration class method."""

    def test_from_duration_seconds_default(self):
        """Test from_duration with default seconds unit."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            3600,  # 1 hour in seconds
        )
        assert event.duration == 3600.0
        assert event.first_on is not None
        assert event.last_on is not None

    def test_from_duration_hours(self):
        """Test from_duration with hours unit."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            2,
            duration_units="h",
        )
        assert event.duration == 7200.0  # 2 hours in seconds

    def test_from_duration_minutes(self):
        """Test from_duration with minutes unit."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            30,
            duration_units="m",
        )
        assert event.duration == 1800.0  # 30 minutes in seconds

    def test_from_duration_days(self):
        """Test from_duration with days unit."""
        event = Event.from_duration(
            "2024-01-01T00:00:00",
            1,
            duration_units="D",
        )
        assert event.get_duration("D") == 1.0

    def test_from_duration_milliseconds(self):
        """Test from_duration with milliseconds unit."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            5000,
            duration_units="ms",
        )
        assert event.duration == 5.0  # 5000 ms = 5 seconds

    def test_from_duration_inner_interval(self):
        """Test from_duration creates inner interval by default."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            3600,
        )
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.last_off is None
        assert event.first_off is None

    def test_from_duration_outer_interval(self):
        """Test from_duration with outer interval type."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            3600,
            interval_type="outer",
        )
        assert event.last_off is not None
        assert event.first_off is not None
        assert event.first_on is None
        assert event.last_on is None

    def test_from_duration_with_description_and_color(self):
        """Test from_duration with metadata."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            3600,
            description="1-hour meeting",
            color="#FF5733",
        )
        assert event.description == "1-hour meeting"
        assert event.color == "#FF5733"

    def test_from_duration_negative_raises_error(self):
        """Test from_duration raises error for negative duration."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            Event.from_duration("2024-01-01T10:00:00", -3600)

    def test_from_duration_zero_duration(self):
        """Test from_duration with zero duration."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            0,
        )
        assert event.duration == 0.0
        assert event.is_point_event()  # Pythonic

    def test_from_duration_fractional(self):
        """Test from_duration with fractional duration."""
        event = Event.from_duration(
            "2024-01-01T10:00:00",
            90,  # 90 minutes = 1.5 hours
            duration_units="m",
        )
        assert event.duration == 5400.0  # 90 minutes in seconds

    def test_from_duration_preserves_fractional_seconds(self):
        """Test from_duration preserves fractional duration via ns promotion."""
        event = Event.from_duration("2024-01-01T10:00:00", 1.9, "s")
        assert event.duration == pytest.approx(1.9)

    def test_from_duration_fractional_hours(self):
        """Test from_duration with fractional hours (1.5 h = 5400 s)."""
        event = Event.from_duration("2024-01-01T10:00:00", 1.5, "h")
        assert event.duration == pytest.approx(5400.0)


class TestEventFromMerlin:
    """Test Event.from_merlin class method."""

    def test_from_merlin_same_day(self):
        """Test from_merlin with same-day event."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00",
            "11:00:00",
        )
        assert event.duration == 3600.0
        assert event.first_on is not None
        assert event.last_on is not None

    def test_from_merlin_spanning_midnight(self):
        """Test from_merlin with event spanning midnight."""
        event = Event.from_merlin(
            "2024-01-01",
            "23:00:00",
            "01:00:00",
        )
        assert event.duration == 7200.0  # 2 hours
        # Check that end is on next day
        expected_end = np.datetime64("2024-01-02T01:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_compact_format(self):
        """Test from_merlin with compact YYYYMMDD HHMMSS format."""
        event = Event.from_merlin(
            "20240101",
            "100000",
            "110000",
        )
        assert event.duration == 3600.0
        expected_start = np.datetime64("2024-01-01T10:00:00", "ns")
        expected_end = np.datetime64("2024-01-01T11:00:00", "ns")
        assert event.first_on == expected_start
        assert event.last_on == expected_end

    def test_from_merlin_compact_spanning_midnight(self):
        """Test from_merlin compact format spanning midnight."""
        event = Event.from_merlin(
            "20240101",
            "230000",
            "010000",
        )
        assert event.duration == 7200.0  # 2 hours
        expected_end = np.datetime64("2024-01-02T01:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_with_subseconds(self):
        """Test from_merlin with subsecond precision."""
        event = Event.from_merlin(
            "20240101",
            "100000.123",
            "110000.456",
        )
        # Duration should account for subseconds
        expected_duration = 3600.0 + 0.456 - 0.123  # 3600.333 seconds
        assert abs(event.duration - expected_duration) < 0.001

    def test_from_merlin_with_microseconds(self):
        """Test from_merlin with microsecond precision."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00.123456",
            "11:00:00.789012",
        )
        # Check microsecond precision is preserved
        expected_duration = 3600.0 + (789012 - 123456) / 1e6
        assert abs(event.duration - expected_duration) < 0.000001

    def test_from_merlin_date_with_slashes(self):
        """Test from_merlin with date using slashes."""
        event = Event.from_merlin(
            "2024/01/01",
            "10:00:00",
            "11:00:00",
        )
        assert event.duration == 3600.0

    def test_from_merlin_mixed_delimiters(self):
        """Test from_merlin with mixed delimiter styles."""
        event = Event.from_merlin(
            "2024-01-01",
            "100000",  # Compact time
            "11:00:00",  # Standard time
        )
        assert event.duration == 3600.0

    def test_from_merlin_hhmm_format(self):
        """Test from_merlin with HHMM format (no seconds)."""
        event = Event.from_merlin(
            "20240101",
            "1030",
            "1145",
        )
        assert event.duration == 4500.0  # 1 hour 15 minutes

    def test_from_merlin_short_time_format(self):
        """Test from_merlin with HH:MM format."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:30",
            "11:45",
        )
        assert event.duration == 4500.0  # 1 hour 15 minutes

    def test_from_merlin_with_seconds(self):
        """Test from_merlin with HH:MM:SS format."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00",
            "10:30:45",
        )
        assert event.duration == 1845.0  # 30 minutes 45 seconds

    def test_from_merlin_with_description_and_color(self):
        """Test from_merlin with metadata."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00",
            "11:00:00",
            description="Morning meeting",
            color="#FF5733",
        )
        assert event.description == "Morning meeting"
        assert event.color == "#FF5733"

    def test_from_merlin_midnight_exact(self):
        """Test from_merlin ending exactly at midnight."""
        event = Event.from_merlin(
            "2024-01-01",
            "22:00:00",
            "00:00:00",
        )
        # 00:00:00 should be interpreted as next day
        assert event.duration == 7200.0  # 2 hours
        expected_end = np.datetime64("2024-01-02T00:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_same_time_wraps_to_next_day(self):
        """Test from_merlin with same start and end time wraps to next day."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00",
            "10:00:00",
        )
        # Since end equals start, it should wrap to next day
        assert event.duration == 86400.0  # 24 hours

    def test_from_merlin_leap_year(self):
        """Test from_merlin spanning midnight on leap year."""
        event = Event.from_merlin(
            "2024-02-28",  # 2024 is a leap year
            "23:00:00",
            "01:00:00",
        )
        expected_end = np.datetime64("2024-02-29T01:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_year_boundary(self):
        """Test from_merlin spanning New Year's Eve."""
        event = Event.from_merlin(
            "2023-12-31",
            "23:00:00",
            "01:00:00",
        )
        expected_end = np.datetime64("2024-01-01T01:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_year_boundary_compact(self):
        """Test from_merlin spanning New Year's Eve in compact format."""
        event = Event.from_merlin(
            "20231231",
            "230000",
            "010000",
        )
        expected_end = np.datetime64("2024-01-01T01:00:00", "ns")
        assert event.last_on == expected_end

    def test_from_merlin_creates_inner_interval(self):
        """Test that from_merlin creates inner interval."""
        event = Event.from_merlin(
            "2024-01-01",
            "10:00:00",
            "11:00:00",
        )
        assert event.first_on is not None
        assert event.last_on is not None
        assert event.last_off is None
        assert event.first_off is None

    def test_from_merlin_subseconds_different_lengths(self):
        """Test from_merlin with different subsecond precision."""
        # 3 digits (milliseconds)
        event1 = Event.from_merlin("20240101", "100000.123", "110000")
        assert event1.first_on is not None

        # 6 digits (microseconds)
        event2 = Event.from_merlin("20240101", "100000.123456", "110000")
        assert event2.first_on is not None

        # 9 digits (nanoseconds - should truncate to microseconds)
        event3 = Event.from_merlin("20240101", "100000.123456789", "110000")
        assert event3.first_on is not None

    def test_from_merlin_invalid_date_format(self):
        """Test from_merlin with invalid date format."""
        with pytest.raises(ValueError, match="Invalid date"):
            Event.from_merlin("20241301", "100000", "110000")  # Month 13 doesn't exist

    def test_from_merlin_invalid_time_format(self):
        """Test from_merlin with invalid time format."""
        with pytest.raises(ValueError, match="Invalid time format"):
            Event.from_merlin("20240101", "10000", "110000")  # Only 5 digits

    def test_from_merlin_invalid_hour(self):
        """Test from_merlin with invalid hour."""
        with pytest.raises(ValueError, match="Invalid hour"):
            Event.from_merlin("20240101", "250000", "110000")  # Hour 25

    def test_from_merlin_invalid_minute(self):
        """Test from_merlin with invalid minute."""
        with pytest.raises(ValueError, match="Invalid minute"):
            Event.from_merlin("20240101", "106000", "110000")  # Minute 60

    def test_from_merlin_invalid_second(self):
        """Test from_merlin with invalid second."""
        with pytest.raises(ValueError, match="Invalid second"):
            Event.from_merlin("20240101", "100060", "110000")  # Second 60

    def test_from_merlin_compact_with_milliseconds_spanning_midnight(self):
        """Test compact format with subseconds spanning midnight."""
        event = Event.from_merlin(
            "20240101",
            "235959.999",
            "000000.001",
        )
        # Should span midnight
        # .999 becomes .999000 microseconds, .001 becomes .001000 microseconds
        expected_end = np.datetime64("2024-01-02T00:00:00.001000", "ns")
        assert event.last_on == expected_end
        # Duration should be about 0.002 seconds
        assert abs(event.duration - 0.002) < 0.001


class TestEventDunderMethods:
    """Test Event dunder methods."""

    def test_contains_operator(self):
        """Test 'in' operator for timestamp containment."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        timestamp_inside = np.datetime64("2024-01-01T10:30:00", "ns")
        timestamp_outside = np.datetime64("2024-01-01T12:00:00", "ns")

        assert timestamp_inside in event
        assert timestamp_outside not in event

    def test_less_than_operator(self):
        """Test '<' operator for event comparison."""
        event1 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        assert event1 < event2
        assert not (event2 < event1)

    def test_sorting_events(self):
        """Test that events can be sorted by start time."""
        event1 = Event(
            first_on="2024-01-01T12:00:00",
            last_on="2024-01-01T13:00:00",
        )
        event2 = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        event3 = Event(
            first_on="2024-01-01T14:00:00",
            last_on="2024-01-01T15:00:00",
        )

        events = [event1, event2, event3]
        sorted_events = sorted(events)

        assert sorted_events[0] == event2
        assert sorted_events[1] == event1
        assert sorted_events[2] == event3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_duration_event(self):
        """Test event with zero duration."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T10:00:00",
        )
        assert event.duration == 0.0
        assert event.is_point_event()  # Pythonic

    def test_event_with_only_start_boundaries(self):
        """Test event with only start and one stop boundary."""
        event = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            first_off="2024-01-01T11:00:00",
        )
        assert event.last_off is not None
        assert event.first_on is not None
        assert event.last_on is None
        assert event.first_off is not None

    def test_event_spanning_midnight(self):
        """Test event that spans midnight."""
        event = Event(
            first_on="2024-01-01T23:00:00",
            last_on="2024-01-02T01:00:00",
        )
        assert event.duration == 2 * 3600.0  # 2 hours

    def test_event_spanning_year_boundary(self):
        """Test event that spans year boundary."""
        event = Event(
            first_on="2023-12-31T23:00:00",
            last_on="2024-01-01T01:00:00",
        )
        assert event.duration == 2 * 3600.0  # 2 hours

    def test_very_long_event(self):
        """Test event spanning multiple days."""
        event = Event(
            first_on="2024-01-01T00:00:00",
            last_on="2024-01-10T00:00:00",
        )
        assert event.get_duration("D") == 9.0  # 9 days

    def test_microsecond_precision(self):
        """Test event with microsecond precision."""
        event = Event(
            first_on="2024-01-01T10:00:00.000000",
            last_on="2024-01-01T10:00:00.123456",
        )
        duration_us = event.get_duration("us")
        assert abs(duration_us - 123456.0) < 1.0  # Allow small floating point error

    def test_contains_with_different_datetime_unit(self):
        """Test contains with np.datetime64 of a different unit than stored (ns)."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        assert event.contains(np.datetime64("2024-01-01T10:30:00", "s")) is True
        assert event.contains(np.datetime64("2024-01-01T12:00:00", "s")) is False


class TestPydanticIntegration:
    """Test Pydantic-specific functionality."""

    def test_model_dump(self):
        """Test that event can be dumped to dict."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            description="Test",
        )
        data = event.model_dump()
        assert "first_on" in data
        assert "last_on" in data
        assert "description" in data

    def test_model_dump_json(self):
        """Test that event can be dumped to JSON."""
        event = Event(
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
        )
        json_str = event.model_dump_json()
        assert isinstance(json_str, str)
        assert "first_on" in json_str

    def test_model_validate(self):
        """Test that event can be validated from dict."""
        data = {
            "first_on": "2024-01-01T10:00:00",
            "last_on": "2024-01-01T11:00:00",
        }
        event = Event.model_validate(data)
        assert event.first_on is not None
        assert event.last_on is not None

    def test_serialization_round_trip(self):
        """Test that event can be serialized and deserialized."""
        original = Event(
            last_off="2024-01-01T09:55:00",
            first_on="2024-01-01T10:00:00",
            last_on="2024-01-01T11:00:00",
            first_off="2024-01-01T11:05:00",
            description="Test event",
            color="#FF5733",
        )

        # Serialize to dict and back
        data = original.model_dump()
        restored = Event.model_validate(data)

        assert restored.times == original.times
        assert restored.description == original.description
        assert restored.color == original.color

    def test_serialization_truncates_sub_microsecond_precision(self):
        """Serialization round-trip loses sub-microsecond (ns) precision.

        str(np.datetime64(..., "ns")) emits nanoseconds, but dateutil.parser
        only parses up to microseconds, so sub-microsecond digits are dropped.
        """
        original = Event(
            first_on=np.datetime64("2024-01-01T10:00:00.123456789", "ns"),
            last_on=np.datetime64("2024-01-01T11:00:00.987654321", "ns"),
        )
        data = original.model_dump()
        restored = Event.model_validate(data)
        # Precision is truncated to microseconds after round-trip
        assert restored.first_on == np.datetime64("2024-01-01T10:00:00.123456000", "ns")
        assert restored.last_on == np.datetime64("2024-01-01T11:00:00.987654000", "ns")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
