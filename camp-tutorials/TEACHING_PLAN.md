# CAMP Tutorial Teaching Plan

## Overview

Build two Google Colab-friendly tutorials that teach open neuroscience data through active exploration, not passive cell-running. Students should type key choices such as DANDI asset paths, time windows, region filters, and unit IDs. Avoid dropdown-heavy interfaces except where a real tool adds value, such as `h5glance` for browsing NWB/HDF5 structure.

Tutorial 1 uses DANDI `001701`, from Aery Jones et al. 2026, Nature Neuroscience: mice performed an X-maze spatial match-to-sample/nonmatch-to-sample task while Neuropixels probes recorded MEC and CA1. Tutorial 2 focuses on decoding position from hippocampal spikes, using a dataset/session chosen for clean behavior and enough units.

## Tutorial 1: Open Data, NWB, Position, Spikes, and LFP

Goal: students should understand how to go from a DANDI dataset to interpretable behavioral and neural variables.

Use DANDI `001701`, version `0.260120.0303`.

Paper framing:
- Mice ran an X-maze task with rewarded ports at arm ends.
- Days 1-10: match-to-sample rule.
- Days 11-20: nonmatch-to-sample rule.
- Neuropixels recordings targeted superficial MEC and, in some mice, dorsal hippocampus/CA1.
- Data include 2D position, head direction, sorted units, selected-channel LFP, and electrode metadata.

One-hour live flow:
- 0-8 min: dataset context from paper and DANDI page.
- 8-15 min: students inspect NWB structure with `h5glance` and find position.
- 15-25 min: extract and plot position for a typed time window.
- 25-35 min: compute speed and occupancy; decide what counts as analyzable behavior.
- 35-48 min: inspect units, filter by brain region, type a unit ID, overlay spikes on position.
- 48-55 min: compare group findings using a shared table.
- 55-60 min: discuss what is needed for LFP/theta, phase precession, and decoding.

Interactive structure:
- Each group chooses or is assigned a different subject/session path.
- Students type `asset_path`, `start_time`, `stop_time`, `region_filter`, and `unit_id`.
- Add prediction prompts before plots: expected position shape, high-occupancy regions, whether a unit should look spatial.
- Use a shared board with columns: group, subject, session, time window, region, unit ID, evidence for spatial modulation.

Notebook implementation rules:
- Use simple typed variables instead of dropdowns.
- Keep helper functions small and visible.
- Show mechanisms once: time reconstruction, speed formula, occupancy histogram, spike-position interpolation.
- Use `h5glance` to show that existing file-inspection tools exist.
- Do not use `nwbwidgets`; it caused PyNWB/HDMF namespace conflicts.
- Do not overclaim anatomical location for selected LFP channels; inspect electrode metadata first.

## Tutorial 2: Decoding Position from Spikes

Goal: students should understand how spike trains can be used to infer position, and what assumptions decoding requires.

Recommended structure:
- Start from position and units already introduced in Tutorial 1.
- Select a session with enough hippocampal/CA1 units and good maze coverage.
- Restrict to a clean time window and movement periods.
- Build spatial tuning curves/place fields.
- Split data into training and testing windows.
- Decode position from binned spikes using a simple Bayesian or template-matching decoder.
- Compare decoded position to actual position.

Student-facing decisions:
- Choose a time window.
- Choose movement threshold.
- Choose region filter.
- Choose bin size.
- Choose spike bin width.
- Compare decoding quality when choices change.

Keep code transparent:
- Show spike binning explicitly.
- Show occupancy-normalized firing rate maps.
- Show the decoder equation or scoring rule.
- Use helper functions only after the core mechanism has been shown.

## Cross-Tutorial Design Rules

- Every section should answer a scientific question, not only produce a plot.
- Keep narration audience-facing; no instructor notes in notebooks.
- Include short “Your turn” prompts every 10-15 minutes.
- Prefer typed variables over widgets for core choices.
- Use group variation to create different outputs across the room.
- Include optional side quests for fast students.
- Keep notebooks robust in Colab and local `camp2026` kernel.
- Clear notebook outputs before committing.
- Avoid unnecessary package complexity.

## Acceptance Criteria

Tutorial 1 is successful if students can:
- identify what the dataset is and what experiment generated it,
- stream one NWB file from DANDI,
- locate position in the NWB structure,
- extract and plot position over a chosen interval,
- compute speed or occupancy,
- inspect units and overlay spikes on behavior,
- state what evidence would support spatial coding.

Tutorial 2 is successful if students can:
- explain why decoding needs position, spikes, and training data,
- compute or use place fields,
- decode position from spike counts,
- compare decoded and true position,
- explain how bin size, time window, and unit selection affect decoding.

## Assumptions

- Tutorial 1 remains centered on DANDI `001701`.
- Spaceflight DANDI `001754` can remain a homework/live exercise for place fields, but not the main live dataset for theta/LFP.
- The second tutorial may use `001701` if a clean CA1-rich session is selected; otherwise choose another DANDI dataset with clearer hippocampal place-cell decoding suitability.
- Colab compatibility is more important than polished UI.
