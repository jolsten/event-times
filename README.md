# event-times

A Python package for analyzing boolean time series data and extracting event timing information.

## Overview

`event-times` converts time series of boolean states (on/off) into a summary of start and stop times for each distinct event. It's particularly useful for analyzing sensor data, system states, or any time-based binary signals.

## Features

- Robust handling of edge cases (series starting/ending in "on" state)
- Automatic event splitting for large time gaps
- Rich event metadata with inner/outer intervals and duration
- Comprehensive input validation
- Event integrity validation (ensures events have meaningful start/stop information)
- Fully typed with dataclasses
- Optional states parameter (treat timestamps as ON times only)
- Command-line interface for quick analysis

## Installation

```bash
pip install event-times
```

For development with testing dependencies:

```bash
pip install -e ".[test]"
```

## Usage

### Python API

```python
import numpy as np
from eventtimes import on_off_times, Event

# Create sample data with explicit states
timestamps = np.array([
    '2024-01-01T00:00:00',
    '2024-01-01T00:01:00',
    '2024-01-01T00:02:00',
    '2024-01-01T00:03:00',
    '2024-01-01T00:04:00',
], dtype='datetime64')

states = np.array([False, True, True, False, True])

# Analyze events (max_gap is in seconds)
events = on_off_times(timestamps, states, max_gap=60.0)

# Access event properties
for event in events:
    print(f"Event started: {event.start}")
    print(f"Event stopped: {event.stop}")
    print(f"Duration: {event.duration}s")
    print(f"Inner interval: {event.inner_interval}")
    print(f"Outer interval: {event.outer_interval}")

# Or use with just ON times (states parameter is optional)
on_times = np.array([
    '2024-01-01T10:00:00',
    '2024-01-01T10:00:30',
    '2024-01-01T10:01:00',
], dtype='datetime64')

# When states is omitted, all timestamps are treated as ON
events = on_off_times(on_times)
print(f"Duration: {events[0].duration} seconds")
```

### Command-Line Interface

```bash
# From a file
event-times data.txt

# From stdin
cat data.txt | event-times

# With custom gap threshold (5 minutes)
event-times --max-gap 300 data.txt

# Summary mode
event-times --summary data.txt

# CSV output
event-times --csv data.txt > events.csv
```

**Input format:**
```
# Comments start with #
2024-01-01T10:00:00 OFF
2024-01-01T10:00:30 ON started
2024-01-01T10:01:00 ON still running
2024-01-01T10:01:30 OFF stopped
```

Each line: `<ISO8601-timestamp> ON|OFF [optional text...]`

## Event Structure

Each `Event` object contains:

- `last_off`: Last timestamp before the event turned on (None if series starts on)
- `first_on`: First timestamp when the event turned on
- `last_on`: Last timestamp when the event was on
- `first_off`: First timestamp after turning off (None if series ends on)

### Event Properties

- `start`: Returns `first_on` if available, otherwise `last_off` (always non-None)
- `stop`: Returns `last_on` if available, otherwise `first_off` (always non-None)
- `duration`: Returns the event duration in seconds (based on inner interval)
- `inner_interval`: Tuple of `(first_on, last_on)` - guaranteed on period
- `outer_interval`: Tuple of `(last_off, first_off)` - full event span

## Time Gap Handling

Events are automatically split when consecutive timestamps have a gap larger than `max_gap` (default: 60 seconds). This applies regardless of the state, treating large gaps as potential data collection interruptions.

## Requirements

- Python >= 3.9
- numpy >= 1.20.0

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT License
