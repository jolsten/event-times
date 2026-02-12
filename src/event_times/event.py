"""Event time representation with uncertain boundaries."""

import datetime
import re
from typing import Annotated, Literal, Optional, Union

import numpy as np
from dateutil.parser import parse as parse_datetime
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    model_validator,
)

__all__ = [
    "Event",
    "validate_datetime",
    "DateTime",
    "DateTimeLike",
    "DurationUnits",
    "IntervalType",
    # Note: _parse_date and _parse_time are private (not exported)
]


# Type alias for inputs that can be converted to datetime
DateTimeLike = Union[str, np.datetime64, datetime.datetime]


def validate_datetime(value: DateTimeLike) -> np.datetime64:
    """Validate and convert various datetime formats to numpy datetime64.

    Args:
        value: A datetime value as string, np.datetime64, or datetime.datetime.

    Returns:
        A numpy datetime64 object with nanosecond precision.

    Raises:
        TypeError: If the value cannot be converted to a datetime.
    """
    if isinstance(value, str):
        return np.datetime64(parse_datetime(value), "ns")
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    if isinstance(value, datetime.datetime):
        return np.datetime64(value, "ns")
    msg = f"invalid datetime: {value!r}"
    raise TypeError(msg)


DateTime = Annotated[
    np.datetime64,
    BeforeValidator(validate_datetime),
    PlainSerializer(str),
]

DurationUnits = Literal["D", "h", "m", "s", "ms", "us", "ns"]
IntervalType = Literal["inner", "outer"]

_NS_PER_UNIT: dict[DurationUnits, int] = {
    "D": 86_400_000_000_000,
    "h": 3_600_000_000_000,
    "m": 60_000_000_000,
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}


def _parse_date(date_str: str) -> tuple[int, int, int]:
    """Parse flexible date format to (year, month, day).

    Supports formats:
    - YYYYMMDD (compact)
    - YYYY-MM-DD (ISO with dashes)
    - YYYY/MM/DD (with slashes)

    Args:
        date_str: Date string in various formats.

    Returns:
        Tuple of (year, month, day) as integers.

    Raises:
        ValueError: If date format is invalid.
    """
    # Remove common delimiters
    date_clean = re.sub(r"[-/\s.]", "", date_str)

    # Try YYYYMMDD format
    if len(date_clean) == 8 and date_clean.isdigit():
        year = int(date_clean[0:4])
        month = int(date_clean[4:6])
        day = int(date_clean[6:8])

        # Validate date components
        try:
            datetime.datetime(year, month, day)
        except ValueError as e:
            raise ValueError(f"Invalid date: {date_str} - {e}") from e

        return (year, month, day)

    # Try parsing with dateutil as fallback
    try:
        dt = datetime.datetime.fromisoformat(date_str)
        return (dt.year, dt.month, dt.day)
    except ValueError:
        pass

    raise ValueError(f"Invalid date format: {date_str}")


def _parse_time(time_str: str) -> tuple[int, int, int, int]:
    """Parse flexible time format to (hour, minute, second, microsecond).

    Supports formats:
    - HHMMSS (compact 6 digits)
    - HHMM (compact 4 digits)
    - HH:MM:SS (with colons)
    - HH:MM (with colons, no seconds)
    - Any of the above with .f+ subseconds

    Args:
        time_str: Time string in various formats.

    Returns:
        Tuple of (hour, minute, second, microsecond) as integers.

    Raises:
        ValueError: If time format is invalid.
    """
    # Split on decimal point to separate seconds from subseconds
    parts = time_str.split(".")
    time_part = parts[0]
    subsec_part = parts[1] if len(parts) > 1 else "0"

    # Remove common delimiters from time part
    time_clean = re.sub(r"[:\s]", "", time_part)

    # Try HHMMSS format
    if len(time_clean) == 6 and time_clean.isdigit():
        hour = int(time_clean[0:2])
        minute = int(time_clean[2:4])
        second = int(time_clean[4:6])
    elif len(time_clean) == 4 and time_clean.isdigit():
        # HHMM format
        hour = int(time_clean[0:2])
        minute = int(time_clean[2:4])
        second = 0
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    # Validate time components
    if not (0 <= hour <= 23):
        raise ValueError(f"Invalid hour: {hour}")
    if not (0 <= minute <= 59):
        raise ValueError(f"Invalid minute: {minute}")
    if not (0 <= second <= 59):
        raise ValueError(f"Invalid second: {second}")

    # Parse subseconds (convert to microseconds)
    if subsec_part and subsec_part != "0":
        # Pad or truncate to 6 digits for microseconds
        subsec_clean = subsec_part.ljust(6, "0")[:6]
        microsecond = int(subsec_clean)
    else:
        microsecond = 0

    return (hour, minute, second, microsecond)


class Event(BaseModel):
    """Represents an event with uncertain start and stop times.

    An Event captures the temporal boundaries of an occurrence where the exact
    start and stop times may be uncertain. The event is bounded by four optional
    timestamps that define inner and outer intervals:

    - start_min: Latest time the event was definitely not occurring (before start)
    - start_max: Earliest time the event was definitely occurring (start bound)
    - stop_min: Latest time the event was definitely occurring (stop bound)
    - stop_max: Earliest time the event was definitely not occurring (after stop)

    Timeline visualization:
        start_min <= [uncertain] <= start_max <= stop_min <= [uncertain] <= stop_max

    Attributes:
        start_min: Latest timestamp before the event definitely started.
            Accepts string, np.datetime64, or datetime.datetime.
        start_max: Earliest timestamp when the event was definitely active.
            Accepts string, np.datetime64, or datetime.datetime.
        stop_min: Latest timestamp when the event was definitely active.
            Accepts string, np.datetime64, or datetime.datetime.
        stop_max: Earliest timestamp after the event definitely stopped.
            Accepts string, np.datetime64, or datetime.datetime.
        description: Optional human-readable description of the event.
        color: Optional color code for visualization (e.g., hex color).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start_min: Optional[DateTime] = None
    start_max: Optional[DateTime] = None
    stop_min: Optional[DateTime] = None
    stop_max: Optional[DateTime] = None
    description: Optional[str] = None
    color: Optional[str] = None

    @model_validator(mode="after")
    def check_valid_event(self) -> "Event":
        """Validate that the event has valid temporal boundaries.

        Returns:
            The validated Event instance.

        Raises:
            ValueError: If the event lacks required start/stop times or has
                invalid time ordering.
        """
        if (self.start_min is None) and (self.start_max is None):
            raise ValueError(
                "Event must have at least one start time (start_min or start_max)"
            )
        if (self.stop_min is None) and (self.stop_max is None):
            raise ValueError(
                "Event must have at least one stop time (stop_min or stop_max)"
            )

        # Check internal ordering constraints
        if self.start_min is not None and self.start_max is not None:
            if self.start_min > self.start_max:
                raise ValueError("start_min must precede or equal start_max")

        if self.start_max is not None and self.stop_min is not None:
            if self.start_max > self.stop_min:
                raise ValueError("start_max must precede or equal stop_min")

        if self.stop_min is not None and self.stop_max is not None:
            if self.stop_min > self.stop_max:
                raise ValueError("stop_min must precede or equal stop_max")

        # Check overall validity
        if self.duration < 0:
            raise ValueError("Invalid interval: stop time precedes start time")

        return self

    @property
    def times(
        self,
    ) -> tuple[
        Optional[np.datetime64],
        Optional[np.datetime64],
        Optional[np.datetime64],
        Optional[np.datetime64],
    ]:
        """Get all four boundary timestamps as a tuple.

        Returns:
            A tuple of (start_min, start_max, stop_min, stop_max).
        """
        return (self.start_min, self.start_max, self.stop_min, self.stop_max)

    @property
    def start(self) -> np.datetime64:
        """Get the best estimate of the event start time.

        Prefers start_max (earliest definite occurrence) over start_min.

        Returns:
            The start timestamp of the event.
        """
        if self.start_max is not None:
            return self.start_max
        return self.start_min  # type: ignore - object is invalid if both starts are None

    @property
    def stop(self) -> np.datetime64:
        """Get the best estimate of the event stop time.

        Prefers stop_min (latest definite occurrence) over stop_max.

        Returns:
            The stop timestamp of the event.
        """
        if self.stop_min is not None:
            return self.stop_min
        return self.stop_max  # type: ignore - object is invalid if both stops are None

    @property
    def duration(self) -> float:
        """Duration of the event in seconds.

        Returns:
            The duration between start and stop in seconds.
        """
        return self.get_duration(units="s")

    @property
    def inner_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]:
        """Get the interval when the event was definitely occurring.

        Returns:
            A tuple of (start_max, stop_min) representing the guaranteed active period.
        """
        return (self.start_max, self.stop_min)

    @property
    def outer_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]:
        """Get the interval that fully bounds the event including uncertainty.

        Returns:
            A tuple of (start_min, stop_max) representing the outer boundary.
        """
        return (self.start_min, self.stop_max)

    @property
    def uncertainty_start(self) -> Optional[float]:
        """Get the temporal uncertainty at the event start in seconds.

        Returns:
            The duration between start_min and start_max, or None if either is missing.
        """
        if self.start_min is not None and self.start_max is not None:
            return (self.start_max - self.start_min) / np.timedelta64(1, "s")
        return None

    @property
    def uncertainty_stop(self) -> Optional[float]:
        """Get the temporal uncertainty at the event stop in seconds.

        Returns:
            The duration between stop_min and stop_max, or None if either is missing.
        """
        if self.stop_min is not None and self.stop_max is not None:
            return (self.stop_max - self.stop_min) / np.timedelta64(1, "s")
        return None

    @property
    def total_uncertainty(self) -> float:
        """Get the total temporal uncertainty in seconds.

        Returns:
            The sum of start and stop uncertainties (0 if unknown).
        """
        start_unc = (
            self.uncertainty_start if self.uncertainty_start is not None else 0.0
        )
        stop_unc = self.uncertainty_stop if self.uncertainty_stop is not None else 0.0
        return start_unc + stop_unc

    @property
    def midpoint(self) -> np.datetime64:
        """Get the temporal midpoint of the event.

        Returns:
            The timestamp halfway between start and stop.
        """
        duration_ns = self.stop - self.start
        return self.start + duration_ns / 2

    def is_point_event(self, threshold: float = 100e-6) -> bool:
        """Check if this is effectively a point event (zero or near-zero duration).

        Args:
            threshold: Duration threshold in seconds. Events with duration less than
                or equal to this value are considered point events.
                Default is 100 microseconds (100e-6).

        Returns:
            True if duration is less than or equal to the threshold.

        Examples:
            >>> event = Event.from_duration("2024-01-01T10:00:00", 0.0001)
            >>> event.is_point_event()  # Uses default 1 microsecond
            False
            >>> event.is_point_event(threshold=0.001)  # 1 millisecond threshold
            True
        """
        return bool(self.duration <= threshold)

    def get_duration(self, units: DurationUnits = "s") -> float:
        """Get the event duration in specified units.

        Args:
            units: Time unit ('D', 'h', 'm', 's', 'ms', 'us', 'ns').

        Returns:
            The duration in the requested units.
        """
        return (self.stop - self.start) / np.timedelta64(1, units)

    def overlaps(self, other: "Event") -> bool:
        """Check if this event overlaps with another event.

        Args:
            other: Another Event instance to check overlap with.

        Returns:
            True if the events overlap (considering outer intervals).
        """
        return bool(self.start < other.stop and other.start < self.stop)

    def contains(self, timestamp: np.datetime64) -> bool:
        """Check if a timestamp falls within the event's outer interval.

        Args:
            timestamp: The timestamp to check.

        Returns:
            True if the timestamp is between start and stop.
        """
        return bool(self.start <= timestamp <= self.stop)

    def definitely_contains(self, timestamp: np.datetime64) -> bool:
        """Check if a timestamp is definitely within the event (inner interval).

        Args:
            timestamp: The timestamp to check.

        Returns:
            True if the timestamp is in the inner interval, False otherwise.
        """
        if self.start_max is None or self.stop_min is None:
            return False
        return bool(self.start_max <= timestamp <= self.stop_min)

    def gap_between(self, other: "Event") -> float:
        """Calculate the temporal gap between this event and another.

        Args:
            other: Another Event instance.

        Returns:
            Gap duration in seconds. Returns 0 if events overlap or touch,
            positive value if there's a gap, negative value if they overlap.
        """
        if self.overlaps(other):
            # Calculate overlap amount (negative gap)
            overlap_start = max(self.start, other.start)
            overlap_stop = min(self.stop, other.stop)
            return -float((overlap_stop - overlap_start) / np.timedelta64(1, "s"))

        # Events don't overlap - calculate gap
        if self.stop <= other.start:
            return float((other.start - self.stop) / np.timedelta64(1, "s"))
        else:
            return float((self.start - other.stop) / np.timedelta64(1, "s"))

    def merge(
        self,
        other: "Event",
        description: Optional[str] = None,
        color: Optional[str] = None,
        max_gap: float = 0.0,
    ) -> Optional["Event"]:
        """Merge this event with another overlapping or adjacent event.

        Args:
            other: Another Event instance to merge with.
            description: Custom description for merged event. If None, combines
                both descriptions.
            color: Color for merged event. If None, uses this event's color.
            max_gap: Maximum gap in seconds between events to allow merging.
                If 0 (default), only overlapping events are merged. If > 0,
                events separated by up to max_gap seconds can be merged.

        Returns:
            A new Event representing the union, or None if events don't overlap
            and the gap between them exceeds max_gap, or if there's a definite
            OFF state between the events.
        """
        # Determine temporal order
        if self.start <= other.start:
            first, second = self, other
        else:
            first, second = other, self

        # Check for contradictory outer boundaries
        # If an outer boundary (definite OFF) falls within the other event's
        # inner interval (definite ON), the events are contradictory
        if (
            first.stop_max is not None
            and second.start_max is not None
            and second.stop_min is not None
        ):
            if second.start_max <= first.stop_max <= second.stop_min:
                return None
        if (
            first.start_min is not None
            and second.start_max is not None
            and second.stop_min is not None
        ):
            if second.start_max <= first.start_min <= second.stop_min:
                return None
        if (
            second.stop_max is not None
            and first.start_max is not None
            and first.stop_min is not None
        ):
            if first.start_max <= second.stop_max <= first.stop_min:
                return None
        if (
            second.start_min is not None
            and first.start_max is not None
            and first.stop_min is not None
        ):
            if first.start_max <= second.start_min <= first.stop_min:
                return None

        # Check if events overlap or are within max_gap
        if not self.overlaps(other):
            gap = self.gap_between(other)
            if gap > max_gap:
                return None

            # Don't merge if there's a definite OFF between events
            # first.stop_max means we know the state was OFF after first event
            # second.start_min means we know the state was OFF before second event
            if first.stop_max is not None or second.start_min is not None:
                return None

        if description is not None:
            merged_description = description
        elif self.description is not None or other.description is not None:
            merged_description = " + ".join(
                d for d in [self.description, other.description] if d is not None
            )
        else:
            merged_description = None
        merged_color = color if color is not None else self.color

        # Compute merged inner interval
        merged_start_max = (
            min(t for t in [self.start_max, other.start_max] if t is not None)
            if self.start_max is not None or other.start_max is not None
            else None
        )
        merged_stop_min = (
            max(t for t in [self.stop_min, other.stop_min] if t is not None)
            if self.stop_min is not None or other.stop_min is not None
            else None
        )

        # Only use outer boundaries that are actually outside the merged inner interval
        valid_start_mins = [
            t
            for t in [self.start_min, other.start_min]
            if t is not None and (merged_start_max is None or t < merged_start_max)
        ]
        merged_start_min = min(valid_start_mins) if valid_start_mins else None

        valid_stop_maxs = [
            t
            for t in [self.stop_max, other.stop_max]
            if t is not None and (merged_stop_min is None or t > merged_stop_min)
        ]
        merged_stop_max = max(valid_stop_maxs) if valid_stop_maxs else None

        return Event(
            start_min=merged_start_min,
            start_max=merged_start_max,
            stop_min=merged_stop_min,
            stop_max=merged_stop_max,
            description=merged_description,
            color=merged_color,
        )

    @classmethod
    def from_interval(
        cls,
        start: DateTimeLike,
        stop: DateTimeLike,
        interval_type: IntervalType = "inner",
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Event":
        """Create an Event from a start-stop interval.

        Args:
            start: Start timestamp (any format accepted by validate_datetime).
            stop: Stop timestamp (any format accepted by validate_datetime).
            interval_type: Whether the provided interval represents the 'inner'
                (definite) interval or 'outer' (bounding) interval. Default is 'inner'.
            description: Optional event description.
            color: Optional color code.

        Returns:
            A new Event instance.

        Examples:
            >>> # Create event with known definite interval
            >>> event = Event.from_interval(
            ...     "2024-01-01T10:00:00",
            ...     "2024-01-01T11:00:00",
            ...     interval_type="inner"
            ... )
            >>> # Create event with outer bounds only
            >>> event = Event.from_interval(
            ...     "2024-01-01T09:55:00",
            ...     "2024-01-01T11:05:00",
            ...     interval_type="outer"
            ... )
        """
        start_dt = validate_datetime(start)
        stop_dt = validate_datetime(stop)

        if interval_type == "inner":
            return cls(
                start_max=start_dt,
                stop_min=stop_dt,
                description=description,
                color=color,
            )
        else:  # outer
            return cls(
                start_min=start_dt,
                stop_max=stop_dt,
                description=description,
                color=color,
            )

    @classmethod
    def from_duration(
        cls,
        start: DateTimeLike,
        duration: float,
        duration_units: DurationUnits = "s",
        interval_type: IntervalType = "inner",
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Event":
        """Create an Event from a start time and duration.

        Args:
            start: Start timestamp (any format accepted by validate_datetime).
            duration: Duration value (must be non-negative).
            duration_units: Time unit for duration ('D', 'h', 'm', 's', 'ms', 'us', 'ns').
                Default is 's' (seconds).
            interval_type: Whether to set the 'inner' (definite) interval or 'outer'
                (bounding) interval. Default is 'inner'.
            description: Optional event description.
            color: Optional color code.

        Returns:
            A new Event instance.

        Raises:
            ValueError: If duration is negative.

        Examples:
            >>> # Create 1-hour event
            >>> event = Event.from_duration("2024-01-01T10:00:00", 3600)
            >>> # Create 2-hour event using hour units
            >>> event = Event.from_duration("2024-01-01T10:00:00", 2, duration_units="h")
            >>> # Create event with outer interval
            >>> event = Event.from_duration(
            ...     "2024-01-01T10:00:00",
            ...     1,
            ...     duration_units="h",
            ...     interval_type="outer"
            ... )
        """
        if duration < 0:
            raise ValueError("Duration must be non-negative")

        start_dt = validate_datetime(start)

        # Convert to nanoseconds with overflow protection
        duration_ns = duration * _NS_PER_UNIT[duration_units]

        # Check for overflow (int64 max is ~9.2e18, giving ~292 years in ns)
        MAX_NS = 2**63 - 1  # int64 max
        if duration_ns > MAX_NS:
            raise ValueError(
                f"Duration too large: {duration} {duration_units} exceeds ~292 year limit"
            )

        # Check for sub-nanosecond precision (would be truncated to 0 or 1)
        if 0 < duration_ns < 1:
            raise ValueError(
                f"Duration too small: {duration} {duration_units} is sub-nanosecond "
                f"and would be truncated to zero"
            )

        duration_td = np.timedelta64(round(duration_ns), "ns")
        stop_dt = start_dt + duration_td

        return cls.from_interval(
            start=start_dt,
            stop=stop_dt,
            interval_type=interval_type,
            description=description,
            color=color,
        )

    @classmethod
    def from_merlin(
        cls,
        start_date: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Event":
        """Create an Event from Merlin-style date and time strings.

        Assumes the event occurs within a single 24-hour period. If end_time is
        earlier than start_time (e.g., start='23:00', end='01:00'), the end is
        assumed to occur on the next day.

        Supports flexible formats:
        - Dates: YYYYMMDD, YYYY-MM-DD, YYYY/MM/DD, etc.
        - Times: HHMMSS, HH:MM:SS, HHMMSS.fff, HH:MM:SS.ffffff, etc.

        Args:
            start_date: Date string (e.g., '20240101', '2024-01-01', '2024/01/01').
            start_time: Start time string (e.g., '103000', '10:30:00', '103000.123').
            end_time: End time string (e.g., '114500', '11:45:00', '114500.456').
            description: Optional event description.
            color: Optional color code.

        Returns:
            A new Event instance with inner interval set.

        Examples:
            >>> # ISO format
            >>> event = Event.from_merlin('2024-01-01', '10:00:00', '11:00:00')
            >>> # Compact format
            >>> event = Event.from_merlin('20240101', '100000', '110000')
            >>> # Mixed format with subseconds
            >>> event = Event.from_merlin('2024-01-01', '100000.123', '110000.456')
            >>> # Event spanning midnight
            >>> event = Event.from_merlin('20240101', '230000', '010000')
        """
        # Parse start date and time
        year, month, day = _parse_date(start_date)
        start_h, start_m, start_s, start_us = _parse_time(start_time)

        start_dt_obj = datetime.datetime(
            year, month, day, start_h, start_m, start_s, start_us
        )
        start_dt = validate_datetime(start_dt_obj)

        # Parse end time (same day first)
        end_h, end_m, end_s, end_us = _parse_time(end_time)
        end_dt_same_day_obj = datetime.datetime(
            year, month, day, end_h, end_m, end_s, end_us
        )
        end_dt_same_day = validate_datetime(end_dt_same_day_obj)

        # If end is before start, assume it's the next day
        if end_dt_same_day < start_dt:
            end_dt = end_dt_same_day + np.timedelta64(1, "D")
        else:
            end_dt = end_dt_same_day

        return cls.from_interval(
            start=start_dt,
            stop=end_dt,
            interval_type="inner",
            description=description,
            color=color,
        )

    def __contains__(self, timestamp: np.datetime64) -> bool:
        """Enable 'timestamp in event' syntax.

        Args:
            timestamp: The timestamp to check.

        Returns:
            True if the timestamp is within the event.
        """
        return self.contains(timestamp)

    def __lt__(self, other: "Event") -> bool:
        """Compare events by start time for sorting.

        Args:
            other: Another Event instance.

        Returns:
            True if this event starts before the other.
        """
        return bool(self.start < other.start)
