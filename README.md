# CAMP Tutorials

Working materials for the CAMP tutorials on open neuroscience data, NWB/DANDI, hippocampal spatial coding, theta-related analyses, and position decoding.

These tutorials were inspired by course material developed by Eric Denovellis, UCSF. Course material: https://github.com/edeno/ncbs-neural-circuits-navigation 

## Tutorial Notebooks

The Colab badge opens the participant version of the notebook, with exercise answers left blank.

| Tutorial | Notebook | Colab |
| --- | --- | --- |
| Open data exploration with DANDI and NWB | [`01-open-data-exploration.ipynb`](01-open-data-exploration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anirudh073/camp2026/blob/main/01-open-data-exploration.ipynb) |
| Open data exploration (offline fallback, local NWB file) | [`01-open-data-exploration-local.ipynb`](01-open-data-exploration-local.ipynb) | — |

Solved reference copies are available as [`01-open-data-exploration-solved.ipynb`](01-open-data-exploration-solved.ipynb) and [`01-open-data-exploration-local-solved.ipynb`](01-open-data-exploration-local-solved.ipynb). For the streamed solved copy, this is the Colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anirudh073/camp2026/blob/main/01-open-data-exploration-solved.ipynb)

## Install

These notebooks use the package manager [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

1. Install uv (one-time, pick your OS):

   **macOS / Linux** : open Terminal and run:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows** : open PowerShell and run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Then restart your terminal. (Full instructions: [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).)

2. Download the repository:

   Go to [github.com/anirudh073/camp2026](https://github.com/anirudh073/camp2026), click the green **Code** button, then **Download ZIP**. Unzip it, then open a terminal (macOS/Linux) or PowerShell (Windows) and `cd` into the unzipped folder, e.g.:

   ```bash
   cd path/to/camp2026-main
   ```

3. Install the dependencies (uv also downloads the right Python version automatically, no separate Python install needed):

   ```bash
   uv sync
   ```

4. Launch Jupyter:

   ```bash
   uv run jupyter lab
   ```

5. Open a notebook and select the `camp2026 (uv)` kernel if your editor or Jupyter asks. If the kernel isn't listed, register it once with:

   ```bash
   uv run python -m ipykernel install --user --name camp2026 --display-name "camp2026 (uv)"
   ```

### Requirements

- No pre-installed Python, conda, or Linux experience needed — uv handles everything above.
- The notebooks stream NWB files from DANDI, so an internet connection is required while running them.
- No local dataset download is required for the tutorial notebooks.

### If streaming doesn't work

[`01-open-data-exploration-local.ipynb`](01-open-data-exploration-local.ipynb) is an offline fallback copy of the open-data-exploration notebook. It reads the NWB file from a local copy ([`sub-Lovelace_ses-Lovelace-DY01-g1_behavior+ecephys.nwb`](sub-Lovelace_ses-Lovelace-DY01-g1_behavior+ecephys.nwb), included in this repository) instead of streaming it from DANDI. Use it if DANDI streaming is slow or unavailable on your network.
