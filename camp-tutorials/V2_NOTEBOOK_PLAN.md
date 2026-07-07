# V2 Notebook Plan: Open Data, X-Maze Behavior, and Spatial Firing

This plan guides agents building the second version of the first tutorial notebook. The notebook should be participant-facing, interactive through scientific choices, and clear enough for students to understand the mechanisms. Explain concepts at important points, especially why occupancy correction matters. Use equations when they improve understanding; avoid equations that add formality without clarifying the analysis.

## 1. Stream From DANDI

Goal: everyone starts from the same dataset and file.

Use DANDI `001701`, version `0.260120.0303`.

Start with one default file:

```python
asset_path = "sub-AppleBottom/sub-AppleBottom_ses-AppleBottom-DY01-g1_behavior+ecephys.nwb"
```

Students can later change `asset_path` manually.

Brief framing:
- This dataset comes from Aery Jones et al. 2026.
- Mice ran an X-maze task.
- Days 1-10: match-to-sample.
- Days 11-20: nonmatch-to-sample.
- Recordings include MEC and CA1 Neuropixels data.

Activity:

> From the filename, what animal and day are we looking at?

## 2. Explore The NWB File Structure

Goal: students learn that NWB files are organized and searchable.

Use `h5glance(h5_file)`.

Students find paths for:

| Data | Path |
| --- | --- |
| Position | `_____` |
| Head direction | `_____` |
| Units | `_____` |
| Electrodes | `_____` |
| LFP | `_____` |

Then provide a check cell using HDF5 paths, not PyNWB syntax:

```python
paths_to_check = {
    "position": "processing/behavior/Position/position",
    "head_direction": "processing/behavior/CompassDirection/head direction",
    "units": "units",
    "electrodes": "general/extracellular_ephys/electrodes",
}
```

Then show PyNWB equivalents:

```python
position_series = nwb.processing["behavior"]["Position"]["position"]
units = nwb.units
electrodes = nwb.electrodes
```

Key concept:

> The browser helps us find the data. PyNWB helps us analyze it.

## 3. Understand The Task Through Behavior

Goal: use position to understand the X-maze task structure.

Extract all position data.

Show:

```python
position_series.data.shape
position_series.rate
position_series.starting_time
```

Students answer:

> Is position stored as `(time, 2)` or `(2, time)`?

Build:

```python
position_df = pd.DataFrame({"time": time, "x": x, "y": y})
```

Plot full 2D trajectory.

Questions:

> Can you see the X-maze structure?
> Where do you think the rewarded arm ends are?
> Does the animal visit all parts of the maze equally?

Optional plot:
- trajectory colored by time

This section is not for choosing an analysis window. It is for understanding behavior and task geometry.

## 4. Movement, Pausing, And Occupancy

Goal: connect behavior to task-relevant locations.

Compute speed over the full session:

```python
dx = np.diff(x)
dy = np.diff(y)
dt = np.diff(time)
speed = np.sqrt(dx**2 + dy**2) / dt
```

Students fill in parts of the formula.

Plot:
- speed over time
- trajectory points colored by speed
- low-speed points over the maze

Questions:

> Where does the animal pause? Are pauses concentrated near likely reward locations?

Introduce movement threshold:

```python
speed_threshold = 5
is_moving = speed > speed_threshold
```

Decision:

> Should spatial firing analyses use all samples or only movement periods?

Then compute occupancy over the full session:

```python
n_bins = 40
occupancy, x_edges, y_edges = np.histogram2d(
    x_for_occupancy,
    y_for_occupancy,
    bins=n_bins,
    weights=np.full_like(x_for_occupancy, dt)
)
```

Explain the concept:

> Occupancy is the amount of time the animal spent in each spatial bin. Raw spike counts are not enough because an animal that spends more time in one place has more opportunity to spike there. A firing-rate map divides spike count by time spent.

Helpful equation:

```text
firing rate at position = spikes at position / time spent at position
```

Questions:

> Which locations are oversampled?
> Why will raw spike counts be misleading without occupancy?

Ahead challenge:
- derive acceleration
- compare occupancy for all samples vs movement-only samples

## 5. Inspect Units And Brain Regions

Goal: understand what was recorded.

Load:

```python
units_df = nwb.units.to_dataframe()
electrodes_df = nwb.electrodes.to_dataframe()
```

Show:
- number of units
- electrode locations
- units per brain region

Question:

> This is an MEC/CA1 experiment. Why do some units appear in visual cortex or other regions?

Students choose a region manually:

```python
region_filter = "Entorhinal area medial part dorsal zone"
```

or:

```python
region_filter = "CA1"
```

If CA1 is absent in the selected file, use MEC.

Decision:

> Which brain region should we inspect first, and why?

## 6. Raster Plot For A Short Time Window

Goal: inspect spike timing without overwhelming the plot.

This is where time windows are useful.

Students choose:

```python
raster_start_time = 300
raster_stop_time = 360
```

Plot raster for units in the selected region.

Questions:

> Are spikes evenly distributed across time?
> Do some cells fire more than others?
> Is this window useful for visualizing activity?

This does not define the place-field analysis window.

## 7. Choose One Unit

Goal: make unit selection an explicit decision.

Show a table for selected-region units:

| unit_id | region | total_spikes | spikes_during_movement |
| --- | --- | ---: | ---: |

Students choose:

```python
unit_id = _____
```

Selection criteria:
- enough spikes
- selected brain region
- not obviously too sparse
- preferably spikes during movement

Prompt:

> Before plotting spike locations, predict whether this unit will fire everywhere or in specific parts of the maze.

## 8. Spike Locations On The Maze

Goal: connect spikes to behavior.

Use all-session data, optionally movement-only.

Interpolate position at spike times:

```python
spike_x = np.interp(spike_times, time, x)
spike_y = np.interp(spike_times, time, y)
```

Explain the concept:

> Spike times and position samples are measured on different time grids. Interpolation estimates where the animal was at each spike time.

Plot:
- full trajectory in gray
- spike locations on top

Questions:

> Are spike locations clustered?
> Could clustering be explained just by where the animal spent more time?

This motivates rate maps.

## 9. Rate Maps

Goal: turn qualitative spike-position plots into occupancy-normalized firing maps.

Compute:

```python
spike_counts, _, _ = np.histogram2d(spike_x, spike_y, bins=[x_edges, y_edges])
rate_map = spike_counts / occupancy
```

Mask low-occupancy bins.

Explain the concept:

> A rate map estimates how strongly a neuron fires at each location. It is more interpretable than raw spike locations because it corrects for unequal sampling of the maze.

Plot:
- occupancy map
- spike count map
- firing rate map

Questions:

> Did the rate map change your interpretation?
> Does this unit look spatially modulated?
> What would make this analysis more convincing?

Optional:
- smooth rate map
- compare all-samples vs movement-only rate map
- try another unit

## 10. LFP As A Preview

Only if time remains.

Goal: show that the file also contains LFP, but do not overdo theta analysis.

Students find LFP path from earlier:

```python
processing/probe_0_channel_159/LFP/LFP
```

Explain:
- LFP is from selected channels, not every probe channel.
- In the default file, one LFP trace is available.

Plot a short LFP window:

```python
lfp_start_time = 300
lfp_stop_time = 310
```

Question:

> What would we need to study theta or phase precession properly?

## Recommended One-Hour Core

Must complete:

1. Stream file.
2. Find position/units/electrodes with `h5glance`.
3. Plot full position.
4. Compute speed and occupancy.
5. Inspect units by region.
6. Pick one unit.
7. Plot spikes on position.
8. Explain occupancy correction.

If time:

9. Compute rate map.
10. Preview LFP.

## Fast Student Side Quests

- Change the asset path to a different day.
- Compare match day `DY01-DY10` vs nonmatch day `DY11-DY20`.
- Compute acceleration.
- Compare occupancy with and without movement threshold.
- Try a different brain region.
- Find a unit with clearer spatial firing.
- Inspect the selected LFP channel.

## Class-Wide Interaction

Use a shared table:

| Group | Subject/session | Region | Unit ID | Movement threshold | Spatial? | Evidence |
| --- | --- | --- | ---: | ---: | --- | --- |

End with:

> Across the class, we screened multiple units from an open dataset and asked whether their firing relates to task-structured space.
