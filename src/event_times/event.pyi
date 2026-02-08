"""Type stub file for Event class - provides better IDE type checking support."""

import datetime
from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel

# Type alias for datetime-like inputs
DateTimeLike = Union[str, np.datetime64, datetime.datetime]

DurationUnits = Literal["D", "h", "m", "s", "ms", "us", "ns"]
IntervalType = Literal["inner", "outer"]

def validate_datetime(value: DateTimeLike) -> np.datetime64: ...

class Event(BaseModel):
    """Event with uncertain temporal boundaries."""
    
    last_off: Optional[np.datetime64]
    first_on: Optional[np.datetime64]
    last_on: Optional[np.datetime64]
    first_off: Optional[np.datetime64]
    description: Optional[str]
    color: Optional[str]
    
    # Single signature accepting DateTimeLike types
    def __init__(
        self,
        *,
        last_off: Optional[DateTimeLike] = None,
        first_on: Optional[DateTimeLike] = None,
        last_on: Optional[DateTimeLike] = None,
        first_off: Optional[DateTimeLike] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None: ...
    
    @property
    def times(self) -> tuple[
        Optional[np.datetime64],
        Optional[np.datetime64],
        Optional[np.datetime64],
        Optional[np.datetime64],
    ]: ...
    
    @property
    def start(self) -> np.datetime64: ...
    
    @property
    def stop(self) -> np.datetime64: ...
    
    @property
    def duration(self) -> float: ...
    
    @property
    def inner_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]: ...
    
    @property
    def outer_interval(self) -> tuple[Optional[np.datetime64], Optional[np.datetime64]]: ...
    
    @property
    def uncertainty_start(self) -> Optional[float]: ...
    
    @property
    def uncertainty_stop(self) -> Optional[float]: ...
    
    @property
    def total_uncertainty(self) -> float: ...
    
    @property
    def midpoint(self) -> np.datetime64: ...
    
    def is_point_event(self, threshold: float = 1e-6) -> bool: ...
    
    def get_duration(self, units: DurationUnits = "s") -> float: ...
    
    def overlaps(self, other: Event) -> bool: ...
    
    def contains(self, timestamp: np.datetime64) -> bool: ...
    
    def definitely_contains(self, timestamp: np.datetime64) -> bool: ...
    
    def gap_between(self, other: Event) -> float: ...
    
    def merge(
        self,
        other: Event,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Optional[Event]: ...
    
    @classmethod
    def from_interval(
        cls,
        start: DateTimeLike,
        stop: DateTimeLike,
        interval_type: IntervalType = "inner",
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Event: ...
    
    @classmethod
    def from_duration(
        cls,
        start: DateTimeLike,
        duration: float,
        duration_units: DurationUnits = "s",
        interval_type: IntervalType = "inner",
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Event: ...
    
    @classmethod
    def from_merlin(
        cls,
        start_date: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Event: ...
    
    def __contains__(self, timestamp: np.datetime64) -> bool: ...
    
    def __lt__(self, other: Event) -> bool: ...

