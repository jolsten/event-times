"""
event-times: A package for analyzing boolean time series data.

This package provides tools to convert time series of boolean states (on/off)
into a summary of start and stop times for each event.
"""

from event_times.core import events_from_state
from event_times.event import Event

__version__ = "0.1.0"
__all__ = ["Event", "events_from_state"]
