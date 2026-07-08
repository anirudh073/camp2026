# CAMP Tutorials

Working materials for the CAMP tutorials on open neuroscience data, NWB/DANDI, hippocampal spatial coding, theta-related analyses, and position decoding.

## Install

These notebooks are set up to run from the shared conda environment defined in [`environment.yml`](environment.yml).

1. Clone the repository:

   ```bash
   git clone https://github.com/anirudh073/camp2026.git
   cd camp2026
   ```

2. Create the environment:

   ```bash
   conda env create -f environment.yml
   ```

   If you already have the environment and want to refresh it after updates:

   ```bash
   conda env update -n camp2026 -f environment.yml --prune
   ```

3. Activate it:

   ```bash
   conda activate camp2026
   ```

4. Launch Jupyter:

   ```bash
   jupyter lab
   ```

5. Open a notebook and select the `camp2026` Python environment or kernel if your editor asks.

### Requirements

- Install a conda-compatible distribution first, such as Miniforge, Mambaforge, Anaconda, or Miniconda.
- The notebooks stream NWB files from DANDI, so an internet connection is required while running them.
- No local dataset download is required for the tutorial notebooks.

## Tutorial Notebooks

| Tutorial | Notebook | Colab |
| --- | --- | --- |
| Open data exploration with DANDI and NWB | [`01-open-data-exploration.ipynb`](01-open-data-exploration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anirudh073/camp2026/blob/main/01-open-data-exploration.ipynb) |
