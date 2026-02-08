# Getting Started with event-times

Welcome to `event-times`! This guide will help you get up and running quickly.

## What is event-times?

`event-times` is a Python package that analyzes boolean time series data (sequences of True/False or on/off states) and converts them into structured event information with precise timing boundaries.

## Quick Installation

```bash
# From the package directory
pip install -e .

# With test dependencies
pip install -e ".[test]"
```

## 5-Minute Tutorial

### 1. Basic Example

```python
import numpy as np
from eventtimes import on_off_times

# Create sample data - a machine that turns on and off
timestamps = np.array([
    '2024-01-01T10:00:00',  # OFF
    '2024-01-01T10:00:30',  # ON
    '2024-01-01T10:01:00',  # ON
    '2024-01-01T10:01:30',  # OFF
], dtype='datetime64')

states = np.array([False, True, True, False])

# Analyze the on/off events
events = on_off_times(timestamps, states)

# You get one event
event = events[0]
print(f"Machine started: {event.first_on}")
# Output: Machine started: 2024-01-01T10:00:30

print(f"Machine stopped: {event.last_on}")
# Output: Machine stopped: 2024-01-01T10:01:00
```

### 2. Understanding Events

Each event has four key timestamps:

```python
event = events[0]

# The four timestamps
event.last_off   # Last time it was OFF before turning ON
event.first_on   # First time it turned ON
event.last_on    # Last time it was ON before turning OFF
event.first_off  # First time it turned OFF after being ON

# Convenient properties
event.start      # Best estimate of start (first_on or last_off)
event.stop       # Best estimate of stop (last_on or first_off)

# Intervals
inner_start, inner_stop = event.inner_interval  # Guaranteed ON period
outer_start, outer_stop = event.outer_interval  # Full event span
```

### 3. Multiple Events

```python
timestamps = np.array([
    '2024-01-01T10:00:00',  # OFF
    '2024-01-01T10:00:30',  # ON - Event 1 starts
    '2024-01-01T10:01:00',  # OFF - Event 1 ends
    '2024-01-01T10:01:30',  # ON - Event 2 starts
    '2024-01-01T10:02:00',  # OFF - Event 2 ends
], dtype='datetime64')

states = np.array([False, True, False, True, False])

events = on_off_times(timestamps, states)
print(f"Found {len(events)} events")
# Output: Found 2 events
```

### 4. Time Gap Handling

Events are automatically split when time gaps are too large:

```python
timestamps = np.array([
    '2024-01-01T10:00:00',  # ON
    '2024-01-01T10:00:30',  # ON
    '2024-01-01T10:05:00',  # ON (4.5 min gap!)
    '2024-01-01T10:05:30',  # ON
], dtype='datetime64')

states = np.array([True, True, True, True])

# Default gap threshold is 60 seconds
events = on_off_times(timestamps, states)
print(f"Events: {len(events)}")
# Output: Events: 2  (split due to large gap)

# Use custom threshold
events = on_off_times(timestamps, states, max_gap=np.timedelta64(10, 'm'))
print(f"Events: {len(events)}")
# Output: Events: 1  (not split with larger threshold)
```

## Common Patterns

### Pattern 1: Calculate Total On-Time

```python
events = on_off_times(timestamps, states)

total_time = sum(
    (event.last_on - event.first_on) 
    for event in events
    if event.first_on and event.last_on
)

print(f"Total on-time: {total_time}")
```

### Pattern 2: Find Longest Event

```python
events = on_off_times(timestamps, states)

longest = max(
    events,
    key=lambda e: (e.last_on - e.first_on) if (e.first_on and e.last_on) else np.timedelta64(0)
)

duration = longest.last_on - longest.first_on
print(f"Longest event: {duration} from {longest.first_on} to {longest.last_on}")
```

### Pattern 3: Count Events Per Hour

```python
from collections import Counter

events = on_off_times(timestamps, states)

# Group by hour
hours = [event.first_on.astype('datetime64[h]') for event in events if event.first_on]
counts = Counter(hours)

for hour, count in sorted(counts.items()):
    print(f"{hour}: {count} events")
```

## Edge Cases to Know

### Series Starting ON

```python
states = np.array([True, True, False])
events = on_off_times(timestamps, states)

# last_off will be None (no prior OFF state)
assert events[0].last_off is None
```

### Series Ending ON

```python
states = np.array([False, True, True])
events = on_off_times(timestamps, states)

# first_off will be None (no subsequent OFF state)
assert events[0].first_off is None
```

### All States OFF

```python
states = np.array([False, False, False])
events = on_off_times(timestamps, states)

# Returns empty list
assert events == []
```

### All States ON (No Gaps)

```python
states = np.array([True, True, True])
events = on_off_times(timestamps, states)

# Returns one event spanning entire series
assert len(events) == 1
assert events[0].last_off is None
assert events[0].first_off is None
```

## Run Examples

The package includes several example scripts:

```bash
# Simple test runner
python test_runner.py

# Comprehensive examples
python examples.py

# Full demonstration
python demo.py
```

## Run Tests

```bash
# Run all tests (requires pytest and hypothesis)
pytest

# Run specific test file
pytest tests/test_onofftimes.py

# Run with verbose output
pytest -v

# Run property-based tests only
pytest tests/test_properties.py
```

## Next Steps

- Read **USAGE.md** for comprehensive documentation
- Check **examples.py** for real-world usage patterns
- Run **demo.py** for an interactive demonstration
- Review **PACKAGE_SUMMARY.md** for technical details

## Common Mistakes to Avoid

1. **Unsorted timestamps** - Always ensure your timestamps are sorted
2. **Mismatched array lengths** - timestamps and states must have the same length
3. **Wrong datetime type** - Use numpy datetime64, not Python datetime
4. **Forgetting gaps** - Consider your sampling rate when setting max_gap

## Getting Help

If you encounter issues:

1. Check that your inputs meet the requirements (sorted timestamps, matching lengths)
2. Review the examples in **examples.py**
3. Run **demo.py** to see expected behavior
4. Look at the test files for edge case handling

## Quick Reference

```python
# Import
from eventtimes import on_off_times, Event

# Basic call
events = on_off_times(timestamps, states)

# With custom gap threshold
events = on_off_times(timestamps, states, max_gap=np.timedelta64(5, 'm'))

# Access event data
event.last_off    # Optional[np.datetime64]
event.first_on    # Optional[np.datetime64]
event.last_on     # Optional[np.datetime64]
event.first_off   # Optional[np.datetime64]
event.start       # Property
event.stop        # Property
event.inner_interval  # Property -> tuple
event.outer_interval  # Property -> tuple
```

Happy analyzing! 🎉