import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from scipy.ndimage import gaussian_filter


COLORS = {
    "ink": "#1f2937",
    "teal": "#14b8a6",
    "blue": "#3b82f6",
    "orange": "#f59e0b",
    "magenta": "#d946ef",
    "gray": "#cbd5e1",
}


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


def hanning_smooth_1d(values, window_size_frames=15):
    """Smooth a 1D signal with an edge-padded Hanning window."""
    values = np.asarray(values, dtype=float)
    if window_size_frames <= 1:
        return values.copy()

    if window_size_frames % 2 == 0:
        window_size_frames += 1

    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    window = np.hanning(window_size_frames)
    window = window / window.sum()
    pad_width = window_size_frames // 2
    padded = np.pad(filled, pad_width=pad_width, mode="edge")
    return np.convolve(padded, window, mode="valid")


def smooth_linear_rate_map(rate_map, sigma_bins=1.5):
    """Gaussian-smooth each contiguous valid linear-track segment separately."""
    radius = int(np.ceil(4 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()

    smoothed = np.full_like(rate_map, np.nan, dtype=float)
    finite = np.isfinite(rate_map)
    if not np.any(finite):
        return smoothed

    padded = np.pad(finite.astype(int), (1, 1), constant_values=0)
    starts = np.where(np.diff(padded) == 1)[0]
    stops = np.where(np.diff(padded) == -1)[0]
    pad_width = len(kernel) // 2

    for start_i, stop_i in zip(starts, stops):
        segment = rate_map[start_i:stop_i]
        if len(segment) == 0:
            continue
        padded_segment = np.pad(segment, pad_width=pad_width, mode="edge")
        smoothed[start_i:stop_i] = np.convolve(padded_segment, kernel, mode="valid")

    return smoothed


def make_time_bins(start_time, stop_time, bin_size_s):
    """Create evenly spaced decoding time bins across the session."""
    duration = stop_time - start_time
    n_bins = int(np.floor(duration / bin_size_s))
    bin_edges = start_time + np.arange(n_bins + 1) * bin_size_s
    bin_centers = bin_edges[:-1] + bin_size_s / 2
    return bin_edges, bin_centers


def bin_spike_counts(units_table, unit_ids, candidate_unit_ids, bin_edges):
    """Count spikes from each unit in each decoding bin."""
    spike_count_list = []

    for unit_id in candidate_unit_ids:
        spike_times = np.asarray(get_unit_spike_times(units_table, unit_ids, unit_id))
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        spike_count_list.append(counts)

    spike_count_matrix = np.column_stack(spike_count_list)
    return spike_count_matrix


def bin_linear_position(linear_position_df, bin_edges):
    """Assign one linear position to each decoding bin, dropping bins that cross segments."""
    sample_times = linear_position_df["time"].to_numpy()
    sample_positions = linear_position_df["linear_position"].to_numpy()
    sample_segments = linear_position_df["track_segment_id"].to_numpy()

    bin_index = np.digitize(sample_times, bin_edges) - 1
    n_bins = len(bin_edges) - 1

    valid = (
        (bin_index >= 0)
        & (bin_index < n_bins)
        & np.isfinite(sample_positions)
        & pd.notna(sample_segments)
    )

    mean_position = np.full(n_bins, np.nan, dtype=float)
    sample_count = np.zeros(n_bins, dtype=int)
    binned_segment_id = np.full(n_bins, np.nan, dtype=float)
    ambiguous_bin = np.zeros(n_bins, dtype=bool)

    if not np.any(valid):
        return mean_position, sample_count, binned_segment_id, ambiguous_bin

    valid_bins = bin_index[valid]
    valid_positions = sample_positions[valid]
    valid_segments = sample_segments[valid]

    unique_bins, inverse_bin = np.unique(valid_bins, return_inverse=True)
    sample_count[unique_bins] = np.bincount(inverse_bin, minlength=len(unique_bins))
    position_sums = np.bincount(inverse_bin, weights=valid_positions, minlength=len(unique_bins))
    mean_position[unique_bins] = position_sums / sample_count[unique_bins]

    segment_group_df = pd.DataFrame({
        "bin": valid_bins,
        "segment": valid_segments,
    })
    segment_nunique = segment_group_df.groupby("bin")["segment"].nunique()
    first_segment = segment_group_df.groupby("bin")["segment"].first()

    ambiguous_bins = segment_nunique.index.to_numpy()[segment_nunique.to_numpy() != 1]
    ambiguous_bin[ambiguous_bins] = True

    unambiguous_bins = first_segment.index.to_numpy()[segment_nunique.to_numpy() == 1]
    binned_segment_id[unambiguous_bins] = first_segment.loc[unambiguous_bins].to_numpy(dtype=float)
    mean_position[ambiguous_bin] = np.nan

    return mean_position, sample_count, binned_segment_id, ambiguous_bin


def split_train_test_by_time(n_bins, train_fraction=0.7, block_size_bins=100):
    """Split bins into interleaved time blocks so train and test both span the session."""
    block_index = np.arange(n_bins) // block_size_bins
    cycle_length = 10
    n_train_blocks = int(np.round(train_fraction * cycle_length))

    train_mask = (block_index % cycle_length) < n_train_blocks
    test_mask = ~train_mask
    return train_mask, test_mask


def _gaussian_kernel_1d(sigma_bins):
    """Create a normalized 1D Gaussian kernel."""
    if sigma_bins <= 0:
        return np.array([1.0])

    radius = int(np.ceil(4 * sigma_bins))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()


def smooth_1d_values_by_valid_segments(values, valid, sigma_bins=1.5):
    """Gaussian-smooth a 1D vector within contiguous valid segments."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    smoothed = np.full_like(values, np.nan, dtype=float)

    if not np.any(valid):
        return smoothed

    kernel = _gaussian_kernel_1d(sigma_bins)
    pad_width = len(kernel) // 2
    padded_valid = np.pad(valid.astype(int), (1, 1), constant_values=0)
    starts = np.where(np.diff(padded_valid) == 1)[0]
    stops = np.where(np.diff(padded_valid) == -1)[0]

    for start_i, stop_i in zip(starts, stops):
        segment = values[start_i:stop_i]
        if len(segment) == 0:
            continue

        padded_segment = np.pad(segment, pad_width=pad_width, mode="edge")
        smoothed[start_i:stop_i] = np.convolve(padded_segment, kernel, mode="valid")

    return smoothed


def compute_1d_tuning_curves_from_fine_data(
    units_table,
    unit_ids,
    candidate_unit_ids,
    linear_position_df,
    included_decode_bin_mask,
    decode_bin_edges,
    position_edges,
    training_sample_mask=None,
    min_occupancy_s=0.2,
    smoothing_sigma_bins=1.5,
    shrinkage_s=1.0,
    rate_floor_hz=1e-3,
):
    """Estimate regularized tuning curves from fine position samples in training bins."""
    sample_times = linear_position_df["time"].to_numpy()
    sample_positions = linear_position_df["linear_position"].to_numpy()
    sample_segments = linear_position_df["track_segment_id"].to_numpy()

    if len(sample_times) == 0:
        raise ValueError("linear_position_df has no samples to train tuning curves from.")

    if len(sample_times) == 1:
        sample_dt = np.array([0.0], dtype=float)
    else:
        median_dt = np.median(np.diff(sample_times))
        sample_dt = np.diff(sample_times, append=sample_times[-1] + median_dt)

    if training_sample_mask is None:
        training_sample_mask = np.ones(len(sample_times), dtype=bool)
    else:
        training_sample_mask = np.asarray(training_sample_mask, dtype=bool)
        if training_sample_mask.shape != sample_times.shape:
            raise ValueError("training_sample_mask must match the fine position sample shape.")

    sample_decode_bin_index = np.digitize(sample_times, decode_bin_edges) - 1
    n_decode_bins = len(included_decode_bin_mask)
    n_position_bins = len(position_edges) - 1

    sample_in_decode_range = (
        (sample_decode_bin_index >= 0)
        & (sample_decode_bin_index < n_decode_bins)
    )
    sample_in_training_bins = np.zeros(len(sample_times), dtype=bool)
    sample_in_training_bins[sample_in_decode_range] = included_decode_bin_mask[
        sample_decode_bin_index[sample_in_decode_range]
    ]

    valid_sample = (
        sample_in_decode_range
        & sample_in_training_bins
        & training_sample_mask
        & np.isfinite(sample_positions)
        & pd.notna(sample_segments)
        & np.isfinite(sample_dt)
        & (sample_dt > 0)
    )

    occupancy_s = np.zeros(n_position_bins, dtype=float)
    tuning_curves = np.full((len(candidate_unit_ids), n_position_bins), np.nan, dtype=float)

    if not np.any(valid_sample):
        return tuning_curves, occupancy_s

    valid_sample_positions = sample_positions[valid_sample]
    valid_sample_dt = sample_dt[valid_sample]
    occupancy_bin_index = np.digitize(valid_sample_positions, position_edges) - 1
    occupancy_in_range = (occupancy_bin_index >= 0) & (occupancy_bin_index < n_position_bins)

    occupancy_s = np.bincount(
        occupancy_bin_index[occupancy_in_range],
        weights=valid_sample_dt[occupancy_in_range],
        minlength=n_position_bins,
    )
    smoothable = occupancy_s > 0
    occupied = occupancy_s >= min_occupancy_s
    smoothed_occupancy_s = smooth_1d_values_by_valid_segments(
        occupancy_s,
        smoothable,
        sigma_bins=smoothing_sigma_bins,
    )

    for unit_i, unit_id in enumerate(candidate_unit_ids):
        spike_times = np.asarray(get_unit_spike_times(units_table, unit_ids, unit_id))
        spike_sums = np.zeros(n_position_bins, dtype=float)

        if len(spike_times) > 0:
            sample_ind = np.searchsorted(sample_times, spike_times, side="right") - 1
            valid_spike = (sample_ind >= 0) & (sample_ind < len(sample_times))

            if np.any(valid_spike):
                sample_ind = sample_ind[valid_spike]
                spike_positions = sample_positions[sample_ind]
                spike_decode_bin_index = sample_decode_bin_index[sample_ind]
                spike_segments = sample_segments[sample_ind]

                spike_in_decode_range = (
                    (spike_decode_bin_index >= 0)
                    & (spike_decode_bin_index < n_decode_bins)
                )
                spike_in_training_bins = np.zeros(len(spike_decode_bin_index), dtype=bool)
                spike_in_training_bins[spike_in_decode_range] = included_decode_bin_mask[
                    spike_decode_bin_index[spike_in_decode_range]
                ]

                keep_spike = (
                    spike_in_decode_range
                    & spike_in_training_bins
                    & training_sample_mask[sample_ind]
                    & np.isfinite(spike_positions)
                    & pd.notna(spike_segments)
                )

                if np.any(keep_spike):
                    spike_bin_index = np.digitize(spike_positions[keep_spike], position_edges) - 1
                    spike_in_range = (spike_bin_index >= 0) & (spike_bin_index < n_position_bins)

                    if np.any(spike_in_range):
                        spike_sums = np.bincount(
                            spike_bin_index[spike_in_range],
                            minlength=n_position_bins,
                        ).astype(float)

        smoothed_spike_sums = smooth_1d_values_by_valid_segments(
            spike_sums,
            smoothable,
            sigma_bins=smoothing_sigma_bins,
        )

        total_occupancy_s = occupancy_s[smoothable].sum()
        global_rate_hz = spike_sums[smoothable].sum() / total_occupancy_s if total_occupancy_s > 0 else 0.0
        regularized_rate = (
            smoothed_spike_sums + shrinkage_s * global_rate_hz
        ) / (
            smoothed_occupancy_s + shrinkage_s
        )
        regularized_rate = np.maximum(regularized_rate, rate_floor_hz)
        tuning_curves[unit_i, occupied] = regularized_rate[occupied]

    return tuning_curves, occupancy_s


def smooth_1d_tuning_curves(tuning_curves, sigma_bins=1.5):
    """Gaussian-smooth each contiguous valid 1D segment separately."""
    smoothed = np.full_like(tuning_curves, np.nan, dtype=float)

    for unit_i, curve in enumerate(tuning_curves):
        finite = np.isfinite(curve)
        if not np.any(finite):
            continue

        smoothed[unit_i] = smooth_1d_values_by_valid_segments(
            curve,
            finite,
            sigma_bins=sigma_bins,
        )

    return smoothed


def make_structural_position_mask(linear_position_df, position_centers):
    """Mark position bins that lie on a real track segment, using track_segment_id."""
    valid_rows = linear_position_df[["linear_position", "track_segment_id"]].dropna().copy()
    structural_mask = np.zeros(len(position_centers), dtype=bool)

    for _, segment_df in valid_rows.groupby("track_segment_id"):
        segment_min = segment_df["linear_position"].min()
        segment_max = segment_df["linear_position"].max()
        on_segment = (position_centers >= segment_min) & (position_centers <= segment_max)
        structural_mask |= on_segment

    return structural_mask


def centers_to_edges(position_centers):
    """Convert ordered bin centers into edges, allowing nonuniform gaps between segments."""
    position_centers = np.asarray(position_centers, dtype=float)
    if len(position_centers) == 0:
        return np.array([], dtype=float)
    if len(position_centers) == 1:
        half_width = 0.5
        return np.array([position_centers[0] - half_width, position_centers[0] + half_width], dtype=float)

    midpoint_edges = 0.5 * (position_centers[1:] + position_centers[:-1])
    first_edge = position_centers[0] - (midpoint_edges[0] - position_centers[0])
    last_edge = position_centers[-1] + (position_centers[-1] - midpoint_edges[-1])
    return np.r_[first_edge, midpoint_edges, last_edge]


def build_segment_position_bin_table(
    linear_position_df,
    track_graph,
    position_bin_size_cm=2.0,
    stacked_gap_cm=12.0,
    edge_spacing_cm=15.0,
):
    """Create one decoding bin table over graph edges using the full track geometry."""
    edge_order = list(track_graph.edges)
    if len(edge_order) == 0:
        return pd.DataFrame()

    if np.isscalar(edge_spacing_cm):
        edge_spacing_values = np.full(len(edge_order) - 1, float(edge_spacing_cm), dtype=float)
    else:
        edge_spacing_values = np.asarray(edge_spacing_cm, dtype=float)
        if len(edge_spacing_values) != len(edge_order) - 1:
            raise ValueError("edge_spacing_cm must have length len(edge_order) - 1 when passed as a sequence.")

    records = []
    segment_linear_start = 0.0
    segment_stacked_start = 0.0

    for segment_order, edge in enumerate(edge_order):
        edge_id = float(track_graph.edges[edge]["edge_id"])
        segment_length_cm = float(track_graph.edges[edge]["distance"])
        local_edges = np.arange(0.0, segment_length_cm + position_bin_size_cm, position_bin_size_cm)

        if local_edges[-1] < segment_length_cm:
            local_edges = np.r_[local_edges, segment_length_cm]

        if len(local_edges) < 2:
            local_edges = np.array([0.0, max(segment_length_cm, position_bin_size_cm)], dtype=float)

        local_centers = 0.5 * (local_edges[:-1] + local_edges[1:])
        linear_centers = segment_linear_start + local_centers
        stacked_centers = segment_stacked_start + local_centers
        start_node_id, end_node_id = edge

        for segment_bin_index, (linear_center, local_center, stacked_center) in enumerate(
            zip(linear_centers, local_centers, stacked_centers)
        ):
            records.append({
                "segment_id": edge_id,
                "segment_order": segment_order,
                "segment_bin_index": segment_bin_index,
                "linear_position": float(linear_center),
                "local_position": float(local_center),
                "stacked_position": float(stacked_center),
                "segment_linear_min": float(segment_linear_start),
                "segment_stacked_min": float(segment_stacked_start),
                "start_node_id": int(start_node_id),
                "end_node_id": int(end_node_id),
                "segment_length_cm": float(segment_length_cm),
            })

        segment_linear_start += segment_length_cm
        segment_stacked_start += segment_length_cm
        if segment_order < len(edge_order) - 1:
            segment_linear_start += edge_spacing_values[segment_order]
            segment_stacked_start += stacked_gap_cm

    return pd.DataFrame(records)


def compute_segment_aware_tuning_curves_from_fine_data(
    units_table,
    unit_ids,
    candidate_unit_ids,
    linear_position_df,
    included_decode_bin_mask,
    decode_bin_edges,
    position_bin_table,
    training_sample_mask=None,
    min_occupancy_s=0.2,
    smoothing_sigma_bins=1.5,
    shrinkage_s=1.0,
    rate_floor_hz=1e-3,
):
    """Estimate tuning curves separately within each track segment."""
    sample_times = linear_position_df["time"].to_numpy()
    sample_positions = linear_position_df["linear_position"].to_numpy()
    sample_segments = linear_position_df["track_segment_id"].to_numpy()

    if len(sample_times) == 0:
        raise ValueError("linear_position_df has no samples to train tuning curves from.")

    if len(sample_times) == 1:
        sample_dt = np.array([0.0], dtype=float)
    else:
        median_dt = np.median(np.diff(sample_times))
        sample_dt = np.diff(sample_times, append=sample_times[-1] + median_dt)

    if training_sample_mask is None:
        training_sample_mask = np.ones(len(sample_times), dtype=bool)
    else:
        training_sample_mask = np.asarray(training_sample_mask, dtype=bool)
        if training_sample_mask.shape != sample_times.shape:
            raise ValueError("training_sample_mask must match the fine position sample shape.")

    sample_decode_bin_index = np.digitize(sample_times, decode_bin_edges) - 1
    n_decode_bins = len(included_decode_bin_mask)
    n_position_bins = len(position_bin_table)

    sample_in_decode_range = (
        (sample_decode_bin_index >= 0)
        & (sample_decode_bin_index < n_decode_bins)
    )
    sample_in_training_bins = np.zeros(len(sample_times), dtype=bool)
    sample_in_training_bins[sample_in_decode_range] = included_decode_bin_mask[
        sample_decode_bin_index[sample_in_decode_range]
    ]

    valid_sample = (
        sample_in_decode_range
        & sample_in_training_bins
        & training_sample_mask
        & np.isfinite(sample_positions)
        & pd.notna(sample_segments)
        & np.isfinite(sample_dt)
        & (sample_dt > 0)
    )

    tuning_curves = np.full((len(candidate_unit_ids), n_position_bins), np.nan, dtype=float)
    occupancy_s = np.zeros(n_position_bins, dtype=float)

    if not np.any(valid_sample):
        return tuning_curves, occupancy_s

    sample_position_bin_index = np.full(len(sample_times), -1, dtype=int)
    for segment_id, segment_bin_df in position_bin_table.groupby("segment_id", sort=False):
        segment_edges = centers_to_edges(segment_bin_df["linear_position"].to_numpy())
        segment_mask = valid_sample & (sample_segments == segment_id)
        if not np.any(segment_mask):
            continue

        segment_positions = sample_positions[segment_mask]
        segment_local_bin_index = np.digitize(segment_positions, segment_edges) - 1
        in_range = (
            (segment_local_bin_index >= 0)
            & (segment_local_bin_index < len(segment_bin_df))
        )

        segment_sample_indices = np.flatnonzero(segment_mask)
        sample_position_bin_index[segment_sample_indices[in_range]] = (
            segment_bin_df.index.to_numpy()[segment_local_bin_index[in_range]]
        )

    occupied_sample_mask = sample_position_bin_index >= 0
    occupancy_s = np.bincount(
        sample_position_bin_index[occupied_sample_mask],
        weights=sample_dt[occupied_sample_mask],
        minlength=n_position_bins,
    ).astype(float)

    smoothed_occupancy_s = np.full(n_position_bins, np.nan, dtype=float)
    occupied = occupancy_s >= min_occupancy_s
    for _, segment_bin_df in position_bin_table.groupby("segment_id", sort=False):
        segment_indices = segment_bin_df.index.to_numpy()
        segment_occupancy = occupancy_s[segment_indices]
        smoothable = segment_occupancy > 0
        if not np.any(smoothable):
            continue

        smoothed_occupancy_s[segment_indices] = smooth_1d_values_by_valid_segments(
            segment_occupancy,
            smoothable,
            sigma_bins=smoothing_sigma_bins,
        )

    for unit_i, unit_id in enumerate(candidate_unit_ids):
        spike_times = np.asarray(get_unit_spike_times(units_table, unit_ids, unit_id))
        spike_sums = np.zeros(n_position_bins, dtype=float)

        if len(spike_times) > 0:
            sample_ind = np.searchsorted(sample_times, spike_times, side="right") - 1
            valid_spike = (sample_ind >= 0) & (sample_ind < len(sample_times))
            if np.any(valid_spike):
                sample_ind = sample_ind[valid_spike]
                spike_bin_index = sample_position_bin_index[sample_ind]
                spike_in_range = spike_bin_index >= 0
                if np.any(spike_in_range):
                    spike_sums = np.bincount(
                        spike_bin_index[spike_in_range],
                        minlength=n_position_bins,
                    ).astype(float)

        smoothed_spike_sums = np.full(n_position_bins, np.nan, dtype=float)
        for _, segment_bin_df in position_bin_table.groupby("segment_id", sort=False):
            segment_indices = segment_bin_df.index.to_numpy()
            segment_occupancy = occupancy_s[segment_indices]
            smoothable = segment_occupancy > 0
            if not np.any(smoothable):
                continue

            smoothed_spike_sums[segment_indices] = smooth_1d_values_by_valid_segments(
                spike_sums[segment_indices],
                smoothable,
                sigma_bins=smoothing_sigma_bins,
            )

        total_occupancy_s = occupancy_s.sum()
        global_rate_hz = spike_sums.sum() / total_occupancy_s if total_occupancy_s > 0 else 0.0
        regularized_rate = (
            smoothed_spike_sums + shrinkage_s * global_rate_hz
        ) / (
            smoothed_occupancy_s + shrinkage_s
        )
        regularized_rate = np.maximum(regularized_rate, rate_floor_hz)
        tuning_curves[unit_i, occupied] = regularized_rate[occupied]

    return tuning_curves, occupancy_s


def map_linear_positions_to_segment_stacked(linear_positions, segment_ids, position_bin_table):
    """Map global linear positions into the segment-stacked display coordinate."""
    linear_positions = np.asarray(linear_positions, dtype=float)
    segment_ids = np.asarray(segment_ids, dtype=float)
    stacked_positions = np.full_like(linear_positions, np.nan, dtype=float)

    segment_display_df = (
        position_bin_table.groupby("segment_id", sort=False)
        .agg(
            segment_linear_min=("segment_linear_min", "first"),
            segment_stacked_min=("segment_stacked_min", "first"),
        )
    )

    for segment_id, row in segment_display_df.iterrows():
        segment_mask = (
            np.isfinite(linear_positions)
            & np.isfinite(segment_ids)
            & (segment_ids == float(segment_id))
        )
        if not np.any(segment_mask):
            continue

        stacked_positions[segment_mask] = (
            row["segment_stacked_min"] + (linear_positions[segment_mask] - row["segment_linear_min"])
        )

    return stacked_positions


def build_graph_distance_matrix(position_bin_table, track_graph):
    """Compute shortest-path distance between all candidate bins on the track graph."""
    import networkx as nx

    if len(position_bin_table) == 0:
        return np.zeros((0, 0), dtype=float)

    node_distance_lookup = dict(nx.all_pairs_dijkstra_path_length(track_graph, weight="distance"))
    segment_ids = position_bin_table["segment_id"].to_numpy()
    local_positions = position_bin_table["local_position"].to_numpy()
    start_nodes = position_bin_table["start_node_id"].to_numpy(dtype=int)
    end_nodes = position_bin_table["end_node_id"].to_numpy(dtype=int)
    segment_lengths = position_bin_table["segment_length_cm"].to_numpy()

    n_bins = len(position_bin_table)
    graph_distances = np.zeros((n_bins, n_bins), dtype=float)

    for i in range(n_bins):
        for j in range(i, n_bins):
            if segment_ids[i] == segment_ids[j]:
                distance_ij = abs(local_positions[i] - local_positions[j])
            else:
                distance_ij = min(
                    local_positions[i]
                    + node_distance_lookup[start_nodes[i]][start_nodes[j]]
                    + local_positions[j],
                    local_positions[i]
                    + node_distance_lookup[start_nodes[i]][end_nodes[j]]
                    + (segment_lengths[j] - local_positions[j]),
                    (segment_lengths[i] - local_positions[i])
                    + node_distance_lookup[end_nodes[i]][start_nodes[j]]
                    + local_positions[j],
                    (segment_lengths[i] - local_positions[i])
                    + node_distance_lookup[end_nodes[i]][end_nodes[j]]
                    + (segment_lengths[j] - local_positions[j]),
                )

            graph_distances[i, j] = distance_ij
            graph_distances[j, i] = distance_ij

    return graph_distances


def build_graph_position_transition_matrix(position_bin_table, track_graph, sigma_cm=15.0):
    """Build a Gaussian transition matrix using track-graph shortest-path distance."""
    sigma_cm = float(sigma_cm)
    if sigma_cm <= 0:
        raise ValueError("sigma_cm must be positive.")

    graph_distances = build_graph_distance_matrix(position_bin_table, track_graph)
    transition = np.exp(-0.5 * (graph_distances / sigma_cm) ** 2)
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    return transition / row_sums


def subset_transition_matrix(full_transition_matrix, position_mask):
    """Restrict a transition matrix to the currently decodable position bins."""
    subset = full_transition_matrix[np.ix_(position_mask, position_mask)].copy()
    row_sums = subset.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    return subset / row_sums


def _plot_posterior_layout(
    window_times,
    true_positions,
    decoded_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    layout_column,
    ylabel,
    decoded_label,
    time_limits,
):
    window_position_mask = ~np.all(np.isnan(window_posterior), axis=0)
    if not np.any(window_position_mask):
        raise ValueError("No posterior values are available in this window. Choose a different time range.")

    plot_bin_df = position_bin_table.loc[window_position_mask].copy()
    plot_positions = plot_bin_df[layout_column].to_numpy()
    plot_posterior = window_posterior[:, window_position_mask]
    time_step = np.median(np.diff(window_times)) if len(window_times) > 1 else bin_size_s
    window_time_edges = np.r_[window_times - time_step / 2, window_times[-1] + time_step / 2]
    window_position_edges = centers_to_edges(plot_positions)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    mesh = ax.pcolormesh(
        window_time_edges,
        window_position_edges,
        plot_posterior.T,
        shading="auto",
        cmap="bone_r",
    )

    ax.scatter(
        window_times,
        true_positions,
        color=COLORS["magenta"],
        s=10,
        alpha=0.9,
        label="True position",
    )

    if decoded_positions is not None and decoded_label is not None:
        ax.scatter(
            window_times,
            decoded_positions,
            color=COLORS["orange"],
            s=18,
            alpha=0.9,
            label=decoded_label,
        )

    if layout_column == "stacked_position":
        segment_maxima = plot_bin_df.groupby("segment_id", sort=False)[layout_column].max().to_numpy()
        segment_minima = plot_bin_df.groupby("segment_id", sort=False)[layout_column].min().to_numpy()
        for upper_segment_max, lower_segment_min in zip(segment_maxima[:-1], segment_minima[1:]):
            ax.axhspan(upper_segment_max, lower_segment_min, color="white", alpha=0.95, zorder=2)
            ax.axhline(upper_segment_max, color=COLORS["gray"], linewidth=0.8, alpha=0.8)

    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*time_limits)
    ax.set_ylim(window_position_edges[0], window_position_edges[-1])
    ax.legend(frameon=False, loc="upper right")
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("posterior probability")
    sns.despine(ax=ax)
    plt.show()


def plot_posterior_linear_layout(
    window_times,
    true_linear_positions,
    decoded_linear_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    decoded_label,
    time_limits,
):
    """Plot decoded posterior on the original linearized-track display axis."""
    _plot_posterior_layout(
        window_times=window_times,
        true_positions=true_linear_positions,
        decoded_positions=decoded_linear_positions,
        window_posterior=window_posterior,
        position_bin_table=position_bin_table,
        bin_size_s=bin_size_s,
        layout_column="linear_position",
        ylabel="linear position (cm)",
        decoded_label=decoded_label,
        time_limits=time_limits,
    )


def plot_posterior_segment_stacked_layout(
    window_times,
    true_stacked_positions,
    decoded_stacked_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    decoded_label,
    time_limits,
):
    """Plot decoded posterior with one vertical band per track segment."""
    _plot_posterior_layout(
        window_times=window_times,
        true_positions=true_stacked_positions,
        decoded_positions=decoded_stacked_positions,
        window_posterior=window_posterior,
        position_bin_table=position_bin_table,
        bin_size_s=bin_size_s,
        layout_column="stacked_position",
        ylabel="segment-stacked position (cm)",
        decoded_label=decoded_label,
        time_limits=time_limits,
    )


def _downsample_posterior_for_plotly(
    window_times,
    plot_posterior,
    max_time_bins=2000,
):
    """Reduce the posterior heatmap's time resolution while preserving posterior peaks.

    Only the heatmap is downsampled here. Position markers stay at full resolution
    since Scattergl (WebGL) renders hundreds of thousands of points without lag -
    the heatmap, not the position trace, is what makes the figure heavy.
    """
    n_time = len(window_times)
    if n_time == 0:
        return window_times, plot_posterior, 1

    block_size = max(1, int(np.ceil(n_time / max_time_bins)))
    if block_size == 1:
        return window_times, plot_posterior, block_size

    downsampled_times = []
    downsampled_posterior = []

    for start_ind in range(0, n_time, block_size):
        stop_ind = min(start_ind + block_size, n_time)
        downsampled_times.append(np.nanmean(window_times[start_ind:stop_ind]))
        downsampled_posterior.append(np.nanmax(plot_posterior[start_ind:stop_ind], axis=0))

    return (
        np.asarray(downsampled_times),
        np.vstack(downsampled_posterior),
        block_size,
    )


def _plot_posterior_layout_plotly(
    window_times,
    true_positions,
    decoded_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    layout_column,
    ylabel,
    decoded_label,
    time_limits,
    max_plot_time_bins=2000,
):
    window_position_mask = ~np.all(np.isnan(window_posterior), axis=0)
    if not np.any(window_position_mask):
        raise ValueError("No posterior values are available in this window. Choose a different time range.")

    plot_bin_df = position_bin_table.loc[window_position_mask].copy()
    plot_positions = plot_bin_df[layout_column].to_numpy()
    plot_posterior = window_posterior[:, window_position_mask]
    window_position_edges = centers_to_edges(plot_positions)

    (
        heatmap_times,
        heatmap_posterior,
        plot_block_size,
    ) = _downsample_posterior_for_plotly(
        window_times,
        plot_posterior,
        max_time_bins=max_plot_time_bins,
    )
    if plot_block_size > 1:
        print(
            f"Posterior heatmap downsampled by {plot_block_size}x "
            f"({len(window_times)} -> {len(heatmap_times)} time bins) for responsiveness. "
            f"Position markers below are still plotted at full resolution ({len(window_times)} samples)."
        )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=heatmap_times,
            y=plot_positions,
            z=heatmap_posterior.T,
            colorscale="Greys",
            zmin=0.0,
            colorbar=dict(title="posterior probability"),
            hovertemplate="time=%{x:.3f} s<br>position=%{y:.2f} cm<br>posterior=%{z:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scattergl(
            x=window_times,
            y=true_positions,
            mode="markers",
            marker=dict(color=COLORS["magenta"], size=3, opacity=0.6),
            name="True position",
            hovertemplate="time=%{x:.3f} s<br>true position=%{y:.2f} cm<extra></extra>",
        )
    )

    if decoded_positions is not None and decoded_label is not None:
        fig.add_trace(
            go.Scattergl(
                x=window_times,
                y=decoded_positions,
                mode="markers",
                marker=dict(color=COLORS["orange"], size=4, opacity=0.6),
                name=decoded_label,
                hovertemplate="time=%{x:.3f} s<br>decoded position=%{y:.2f} cm<extra></extra>",
            )
        )

    if layout_column == "stacked_position":
        segment_maxima = plot_bin_df.groupby("segment_id", sort=False)[layout_column].max().to_numpy()
        segment_minima = plot_bin_df.groupby("segment_id", sort=False)[layout_column].min().to_numpy()
        for upper_segment_max, lower_segment_min in zip(segment_maxima[:-1], segment_minima[1:]):
            fig.add_hrect(
                y0=float(upper_segment_max),
                y1=float(lower_segment_min),
                line_width=0,
                fillcolor="white",
                opacity=0.95,
            )
            fig.add_hline(
                y=float(upper_segment_max),
                line=dict(color=COLORS["gray"], width=1),
                opacity=0.8,
            )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=500,
        legend=dict(x=0.8, y=1.0),
        margin=dict(l=70, r=30, t=30, b=60),
        xaxis=dict(title="time (s)", range=list(time_limits)),
        yaxis=dict(
            title=ylabel,
            range=[float(window_position_edges[0]), float(window_position_edges[-1])],
        ),
    )
    fig.show()


def plot_posterior_linear_layout_plotly(
    window_times,
    true_linear_positions,
    decoded_linear_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    decoded_label,
    time_limits,
    max_plot_time_bins=2000,
):
    """Plot decoded posterior on the original linearized-track display axis with Plotly."""
    _plot_posterior_layout_plotly(
        window_times=window_times,
        true_positions=true_linear_positions,
        decoded_positions=decoded_linear_positions,
        window_posterior=window_posterior,
        position_bin_table=position_bin_table,
        bin_size_s=bin_size_s,
        layout_column="linear_position",
        ylabel="linear position (cm)",
        decoded_label=decoded_label,
        time_limits=time_limits,
        max_plot_time_bins=max_plot_time_bins,
    )


def plot_posterior_segment_stacked_layout_plotly(
    window_times,
    true_stacked_positions,
    decoded_stacked_positions,
    window_posterior,
    position_bin_table,
    bin_size_s,
    decoded_label,
    time_limits,
    max_plot_time_bins=2000,
):
    """Plot decoded posterior with one vertical band per track segment using Plotly."""
    _plot_posterior_layout_plotly(
        window_times=window_times,
        true_positions=true_stacked_positions,
        decoded_positions=decoded_stacked_positions,
        window_posterior=window_posterior,
        position_bin_table=position_bin_table,
        bin_size_s=bin_size_s,
        layout_column="stacked_position",
        ylabel="segment-stacked position (cm)",
        decoded_label=decoded_label,
        time_limits=time_limits,
        max_plot_time_bins=max_plot_time_bins,
    )


def compute_poisson_log_likelihood(spike_count_matrix, expected_spike_count_matrix):
    """Score each candidate position using an independent Poisson model."""
    safe_expected = np.clip(expected_spike_count_matrix, 1e-12, None)
    log_expected = np.log(safe_expected)

    log_likelihood = spike_count_matrix @ log_expected - safe_expected.sum(axis=0)
    return log_likelihood


def log_likelihood_to_posterior(log_likelihood, log_prior=None):
    """Convert log-likelihoods into normalized posterior probabilities."""
    if log_prior is None:
        n_positions = log_likelihood.shape[1]
        log_prior = np.full(n_positions, -np.log(n_positions), dtype=float)

    log_posterior = log_likelihood + log_prior
    log_posterior = log_posterior - np.max(log_posterior, axis=1, keepdims=True)

    posterior = np.exp(log_posterior)
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    return posterior


def apply_position_transition_prior(log_likelihood, transition_matrix):
    """Convert per-bin likelihoods into a causal posterior with a fixed transition matrix."""
    n_time, n_positions = log_likelihood.shape
    transition_matrix = np.asarray(transition_matrix, dtype=float)
    if transition_matrix.shape != (n_positions, n_positions):
        raise ValueError("transition_matrix must have shape (n_positions, n_positions).")

    stabilized_likelihood = np.exp(log_likelihood - np.max(log_likelihood, axis=1, keepdims=True))

    posterior = np.zeros((n_time, n_positions), dtype=float)
    posterior[0] = stabilized_likelihood[0]
    posterior[0] = posterior[0] / posterior[0].sum()

    for time_i in range(1, n_time):
        prior = posterior[time_i - 1] @ transition_matrix
        prior = np.clip(prior, 1e-12, None)
        posterior[time_i] = stabilized_likelihood[time_i] * prior
        posterior[time_i] = posterior[time_i] / posterior[time_i].sum()

    return posterior
