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
