# Repository Guidelines

## Primary Role

Help build participant-friendly CAMP tutorials on open neuroscience data, NWB/DANDI, hippocampal spatial coding, theta-related analyses, and position decoding. Prefer Google Colab-compatible notebooks.

## Communication

Keep responses concise. Avoid repeating the same recommendation in multiple forms.

## Tutorial Style

Notebook narration must be pedagogical and audience-facing. Do not include instructor-facing design notes, teaching strategy notes, or reminders about how the class should be run inside participant-facing notebooks.

Tutorial explanations should be easy to read. Avoid dense one-sentence explanations for multi-step ideas; break them into short bullets or small paragraphs, and make the logic between steps explicit.

Build interactivity around scientific interpretation rather than only cell execution. Students should predict, choose sessions or units, compare plots, and interpret results live.

Use helper functions to keep notebooks manageable, but do not make the analysis opaque. Expose enough code, equations, and intermediate variables for students to understand NWB access, spike-position alignment, occupancy maps, filtering, decoding, and related analyses.

When useful, maintain paired notebook versions: an unsolved participant notebook with blanks/questions and a solved reference notebook with completed code. The V2 open-data notebook may be treated as a solved version.

Do not use the word "detective" in tutorial titles or narration.

## Project Structure

Use `camp-tutorials/` for notebooks and supporting tutorial materials. Keep `ncbs-neural-circuits-navigation/` as an ignored reference folder; do not track it in git.
