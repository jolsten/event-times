# event-times Usage Guide

## Introduction

The `event-times` package provides a robust solution for analyzing boolean time series data. It converts sequences of on/off states into structured event information with precise timing boundaries.

## Installation

```bash
pip install event-times
```

Or for development:

```bash
git clone https://github.com/yourusername/event-times.git
cd onofftimes
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from eventtimes import on_off_times

# Example 1: With explicit states
timestamps = np.array([
    '2024-01-01T10:00:00',
    '2024-01-01T10:00:30',
    '2024-01-01T10:01:00',
    '2024-01-01T10:01:30',
], dtype='datetime64')

states = np.array([False, True, True, False])

# Analyze events
events = on_off_times(timestamps, states)

# Access event information
event = events[0]
print(f"Started: {event.first_on}")
print(f"Stopped: {event.last_on}")
print(f"Duration: {event.duration} seconds")

# Example 2: With only ON times (states is optional)
on_times = np.array([
    '2024-01-01T10:00:00',
    '2024-01-01T10:00:30',
    '2024-01-01T10:01:00',
], dtype='datetime64')

# When states is not provided, all timestamps are treated as ON
events = on_off_times(on_times)
print(f"Duration: {events[0].duration} seconds")
```

## Core Concepts

### Event Structure

Each event detected by `on_off_times` contains four key timestamps:

1. **last_off**: The last timestamp *before* the event turned on
   - `None` if the time series starts in the "on" state
   
2. **first_on**: The first timestamp when the event turned on
   - The beginning of the guaranteed "on" period
   
3. **last_on**: The last timestamp when the event was on
   - The end of the guaranteed "on" period
   
4. **first_off**: The first timestamp *after* the event turned off
   - `None` if the time series ends in the "on" state

**Event Validation**: The `Event` dataclass enforces integrity constraints to ensure events are meaningful:
- At least one of `last_off` or `first_on` must be not None (must have a start reference)
- At least one of `last_on` or `first_off` must be not None (must have a stop reference)
- Attempting to create an invalid Event raises `ValueError`

This validation ensures that every event has both meaningful start and stop information, preventing malformed events from being created.

### Time Gap Handling

Events are automatically split when the time between consecutive samples exceeds the `max_gap` threshold (default: 60 seconds). This handles:

- Data collection interruptions
- Natural breaks in activity
- Sensor dropout periods

```python
# Custom gap threshold
events = on_off_times(timestamps, states, max_gap=np.timedelta64(5, 'm'))
```

**Important**: When events are split by time gaps (rather than actual state changes to False), the boundary fields that would normally reference the adjacent timestamp are set to `None`:
- If an event ends due to a gap (not a state change), `first_off` will be `None`
- If an event starts after a gap (not a state change), `last_off` will be `None`

Example:
```python
timestamps = [t0, t1, t5, t6]  # Large gap between t1 and t5
states = [True, True, True, True]  # All True

# With 60s threshold, splits into 2 events:
# Event 1: last_off=None, first_on=t0, last_on=t1, first_off=None (split by gap)
# Event 2: last_off=None (split by gap), first_on=t5, last_on=t6, first_off=None
```

## Event Properties

The `Event` class provides convenient properties:

### `start` Property
Returns the event's start time:
- Returns `first_on` if available (guaranteed on time)
- Falls back to `last_off` if `first_on` is None

```python
event.start  # Best estimate of when event began
```

### `stop` Property
Returns the event's stop time:
- Returns `last_on` if available (guaranteed on time)
- Falls back to `first_off` if `last_on` is None

```python
event.stop  # Best estimate of when event ended
```

### `inner_interval` Property
Returns `(first_on, last_on)` - the period where the state is definitely "on":

```python
start, end = event.inner_interval
guaranteed_duration = end - start
```

### `outer_interval` Property
Returns `(last_off, first_off)` - the full span including boundary transitions:

```python
start, end = event.outer_interval
maximum_duration = end - start
```

## Input Requirements

### Required Format

1. **Timestamps**: `np.ndarray` with dtype `datetime64`
   - Must be sorted in ascending order
   - Can be any datetime64 resolution (seconds, milliseconds, etc.)

2. **States**: `np.ndarray` with dtype `bool` (or convertible to bool)
   - Must have the same length as timestamps
   - True = on, False = off

3. **max_gap**: `np.timedelta64` (optional)
   - Default: 60 seconds
   - Events spanning gaps larger than this are split

### Input Validation

The function performs strict validation:

```python
# ✓ Valid inputs
timestamps = np.array(['2024-01-01T10:00:00', '2024-01-01T10:01:00'], dtype='datetime64')
states = np.array([True, False])

# ✗ Raises ValueError: Empty arrays
on_off_times(np.array([]), np.array([]))

# ✗ Raises ValueError: Length mismatch
on_off_times(timestamps, np.array([True]))

# ✗ Raises ValueError: Unsorted timestamps
timestamps_bad = np.array(['2024-01-01T10:01:00', '2024-01-01T10:00:00'], dtype='datetime64')
on_off_times(timestamps_bad, states)
```

## Common Use Cases

### 1. Machine Uptime Monitoring

```python
# Collect machine state every minute
base = np.datetime64('2024-01-01T08:00:00')
timestamps = base + np.arange(480) * np.timedelta64(1, 'm')  # 8 hours

# Machine states (True = running, False = stopped)
states = load_machine_states()  # Your data source

# Analyze uptime events
events = on_off_times(timestamps, states)

# Calculate total uptime
total_uptime = sum(
    (event.last_on - event.first_on) 
    for event in events
)
print(f"Total uptime: {total_uptime}")
```

### 2. Sensor Trigger Analysis

```python
# Motion sensor data
timestamps = load_sensor_timestamps()
triggered = load_sensor_states()  # True when motion detected

# Find trigger events with 5-second grouping
events = on_off_times(timestamps, triggered, max_gap=np.timedelta64(5, 's'))

# Analyze each trigger
for i, event in enumerate(events, 1):
    inner = event.inner_interval
    outer = event.outer_interval
    
    print(f"Trigger {i}:")
    print(f"  Definitely detected: {inner[0]} to {inner[1]}")
    print(f"  Possibly detected: {outer[0]} to {outer[1]}")
```

### 3. System State Transitions

```python
# Server response states
timestamps = np.array([...], dtype='datetime64')
responding = np.array([...], dtype=bool)  # True = server responding

# Find downtime events (invert the logic)
events = on_off_times(timestamps, ~responding)  # Note the ~

# Report outages
for event in events:
    if event.first_on is not None and event.last_on is not None:
        duration = event.last_on - event.first_on
        print(f"Outage: {event.first_on} to {event.last_on} ({duration})")
```

### 4. Handling Missing Data

```python
# Data with gaps from sensor dropout
timestamps = np.array([
    '2024-01-01T10:00:00',
    '2024-01-01T10:00:30',
    # 10-minute gap - sensor offline
    '2024-01-01T10:10:30',
    '2024-01-01T10:11:00',
], dtype='datetime64')

states = np.array([True, True, True, True])

# Gap threshold smaller than the actual gap
# This treats the data before and after the gap as separate events
events = on_off_times(timestamps, states, max_gap=np.timedelta64(1, 'm'))

print(f"Detected {len(events)} separate events")
# Output: Detected 2 separate events
```

## Edge Cases

### Series Starting On

```python
# First sample is True
timestamps = np.array(['2024-01-01T10:00:00', '2024-01-01T10:01:00'], dtype='datetime64')
states = np.array([True, False])

events = on_off_times(timestamps, states)
assert events[0].last_off is None  # No "off" state before
```

### Series Ending On

```python
# Last sample is True
timestamps = np.array(['2024-01-01T10:00:00', '2024-01-01T10:01:00'], dtype='datetime64')
states = np.array([False, True])

events = on_off_times(timestamps, states)
assert events[0].first_off is None  # No "off" state after
```

### All States False

```python
states = np.array([False, False, False])
events = on_off_times(timestamps, states)
assert events == []  # Returns empty list
```

### All States True

```python
states = np.array([True, True, True])
events = on_off_times(timestamps, states)
assert len(events) >= 1  # At least one event (may be split by gaps)
```

## Performance Considerations

The function is optimized for:
- Array sizes up to millions of samples
- Efficient numpy operations
- Minimal memory overhead

For very large datasets (>10M samples), consider:
- Processing data in chunks
- Using appropriate time resolution (e.g., seconds vs nanoseconds)
- Pre-filtering obvious noise

## Testing

The package includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=onofftimes

# Run property-based tests only
pytest tests/test_properties.py
```

## API Reference

### `on_off_times(timestamps, states, max_gap=np.timedelta64(60, 's'))`

Convert time series of boolean states to event summaries.

**Parameters:**
- `timestamps` (np.ndarray): Sorted datetime64 array
- `states` (np.ndarray): Boolean array of states
- `max_gap` (np.timedelta64): Maximum gap before splitting events

**Returns:**
- `List[Event]`: List of detected events

**Raises:**
- `ValueError`: If inputs are invalid

### `Event` Class

**Attributes:**
- `last_off` (Optional[np.datetime64]): Last timestamp before on
- `first_on` (Optional[np.datetime64]): First timestamp when on
- `last_on` (Optional[np.datetime64]): Last timestamp when on
- `first_off` (Optional[np.datetime64]): First timestamp after on

**Properties:**
- `start` → np.datetime64: Event start time (always non-None, guaranteed by validation)
- `stop` → np.datetime64: Event stop time (always non-None, guaranteed by validation)
- `duration` → float: Duration in seconds (based on inner interval)
- `inner_interval` → Tuple[Optional[np.datetime64], Optional[np.datetime64]]: Guaranteed on period
- `outer_interval` → Tuple[Optional[np.datetime64], Optional[np.datetime64]]: Full event span

## Troubleshooting

### "Array length mismatch" Error

Ensure timestamps and states have the same length:

```python
assert len(timestamps) == len(states)
```

### "Timestamps must be sorted" Error

Sort your timestamps before calling:

```python
sort_idx = np.argsort(timestamps)
timestamps = timestamps[sort_idx]
states = states[sort_idx]
```

### Unexpected Event Splits

Check your time gaps - default is 60 seconds:

```python
time_diffs = np.diff(timestamps)
print(f"Max gap: {time_diffs.max()}")

# Adjust threshold if needed
events = on_off_times(timestamps, states, max_gap=time_diffs.max() + 1)
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.