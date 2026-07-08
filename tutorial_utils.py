import numpy as np
from scipy.ndimage import gaussian_filter


EXPECTED_NWB_PATHS = {
    "position": {"neurodata_type": "SpatialSeries", "required_children": {"data", "starting_time"}},
    "units": {"neurodata_type": "Units", "required_children": {"spike_times"}},
    "lfp": {"neurodata_type": "LFP", "required_children": {"LFP"}},
}


def check_nwb_path(h5_file, path, neurodata_type=None, required_children=None):
    """Check whether an NWB/HDF5 path exists and matches a simple expected structure."""
    if not path or path not in h5_file:
        return False

    obj = h5_file[path]

    if neurodata_type is not None and obj.attrs.get("neurodata_type", "") != neurodata_type:
        return False

    if required_children:
        if not hasattr(obj, "keys"):
            return False
        if not set(required_children).issubset(set(obj.keys())):
            return False

    return True


def get_unit_row(unit_ids, unit_id):
    """Return the row number for a unit id."""
    matches = (unit_ids == unit_id).nonzero()[0]
    if len(matches) == 0:
        raise KeyError(f"unit_id {unit_id} not found")
    return int(matches[0])


def get_unit_spike_times(units_table, unit_ids, unit_id):
    """Fetch spike times for one unit without materializing the full units table."""
    return units_table["spike_times"][get_unit_row(unit_ids, unit_id)][:]


def unit_region_from_row(units_table, row_i, electrodes_df):
    """Infer a unit's brain region from the electrode location linked to that row."""
    try:
        electrodes = units_table["electrodes"][row_i]
        if hasattr(electrodes, "columns"):
            locations = electrodes["location"].dropna().astype(str)
        else:
            electrode_ids = electrodes[:]
            locations = electrodes_df.loc[electrode_ids, "location"].dropna().astype(str)
        if len(locations) == 0:
            return "unknown"
        return locations.mode().iloc[0]
    except Exception:
        return "unknown"


def compute_occupancy_map(x, y, dt, n_bins):
    """Compute a 2D occupancy map after dropping non-finite samples."""
    xy_valid = np.isfinite(x[:-1]) & np.isfinite(y[:-1]) & np.isfinite(dt)
    occupancy_x = x[:-1][xy_valid]
    occupancy_y = y[:-1][xy_valid]
    occupancy_dt = dt[xy_valid]
    return np.histogram2d(
        occupancy_x,
        occupancy_y,
        bins=n_bins,
        weights=occupancy_dt,
    )


def map_spikes_to_position_frames(unit_spike_times, time, x, y):
    """Assign spikes to tracked position samples using the position frame bins."""
    time = np.asarray(time)
    x = np.asarray(x)
    y = np.asarray(y)
    unit_spike_times = np.asarray(unit_spike_times)

    if len(time) == 0:
        return {
            "spike_hist": np.array([], dtype=int),
            "spike_x": np.array([], dtype=float),
            "spike_y": np.array([], dtype=float),
        }

    if len(time) == 1:
        spike_hist = np.array([np.sum(unit_spike_times == time[0])], dtype=int)
    else:
        timestep = np.median(np.diff(time))
        bins = np.append(time, time[-1] + timestep)
        spike_hist, _ = np.histogram(unit_spike_times, bins=bins)

    spike_mask = spike_hist > 0
    spike_x = x[spike_mask]
    spike_y = y[spike_mask]
    spike_valid = np.isfinite(spike_x) & np.isfinite(spike_y)

    return {
        "spike_hist": spike_hist,
        "spike_x": spike_x[spike_valid],
        "spike_y": spike_y[spike_valid],
    }


def compute_rate_map(unit_spike_times, time, x, y, x_edges, y_edges, occupancy, min_occupancy=0.1):
    """Compute spike counts and an unsmoothed rate map for one unit."""
    spike_positions = map_spikes_to_position_frames(unit_spike_times, time, x, y)
    spike_x = spike_positions["spike_x"]
    spike_y = spike_positions["spike_y"]
    spike_counts, _, _ = np.histogram2d(
        spike_x,
        spike_y,
        bins=[x_edges, y_edges],
    )
    rate_map = np.full_like(occupancy, np.nan, dtype=float)
    valid = occupancy > min_occupancy
    rate_map[valid] = spike_counts[valid] / occupancy[valid]
    return {
        "spike_x": spike_x,
        "spike_y": spike_y,
        "spike_counts": spike_counts,
        "rate_map": rate_map,
    }


def smooth_rate_map(spike_counts, occupancy, min_occupancy=0.1, smoothing_sigma=1):
    """Smooth spike counts and occupancy separately, then divide."""
    smoothed_spike_counts = gaussian_filter(spike_counts, sigma=smoothing_sigma)
    smoothed_occupancy = gaussian_filter(occupancy, sigma=smoothing_sigma)
    smoothed_rate_map = np.full_like(occupancy, np.nan, dtype=float)
    valid_smoothed = smoothed_occupancy > min_occupancy
    smoothed_rate_map[valid_smoothed] = (
        smoothed_spike_counts[valid_smoothed] / smoothed_occupancy[valid_smoothed]
    )
    return smoothed_rate_map
