# CAMP Tutorials

Working materials for the CAMP tutorials on open neuroscience data, NWB/DANDI, hippocampal spatial coding, theta-related analyses, and position decoding.

## Install

These notebooks use [uv](https://docs.astral.sh/uv/) to manage Python and dependencies — no prior Python or Linux experience needed.

1. Install uv (one-time, pick your OS):

   **macOS / Linux** — open Terminal and run:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows** — open PowerShell and run:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Then restart your terminal. (Full instructions: [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).)

2. Install git, if you don't already have it:

   **macOS** — open Terminal and run:

   ```bash
   git --version
   ```

   If it's not installed, this prompts you to install the Xcode Command Line Tools — follow the popup.

   **Windows** — download and run the installer from [git-scm.com/download/win](https://git-scm.com/download/win), keeping the default options. This also adds a "Git Bash" terminal you can use for the commands below.

   **Ubuntu / Linux** — open a terminal and run:

   ```bash
   sudo apt install git
   ```

3. Clone the repository:

   ```bash
   git clone https://github.com/anirudh073/camp2026.git
   cd camp2026
   ```

4. Install the dependencies (uv also downloads the right Python version automatically, no separate Python install needed):

   ```bash
   uv sync
   ```

5. Launch Jupyter:

   ```bash
   uv run jupyter lab
   ```

6. Open a notebook and select the `camp2026 (uv)` kernel if your editor or Jupyter asks. If the kernel isn't listed, register it once with:

   ```bash
   uv run python -m ipykernel install --user --name camp2026 --display-name "camp2026 (uv)"
   ```

### Requirements

- No pre-installed Python, conda, or Linux experience needed — uv handles everything above.
- The notebooks stream NWB files from DANDI, so an internet connection is required while running them.
- No local dataset download is required for the tutorial notebooks.

## Tutorial Notebooks

| Tutorial | Notebook | Colab |
| --- | --- | --- |
| Open data exploration with DANDI and NWB | [`01-open-data-exploration.ipynb`](01-open-data-exploration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anirudh073/camp2026/blob/main/01-open-data-exploration.ipynb) |
