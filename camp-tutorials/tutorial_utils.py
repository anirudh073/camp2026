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
