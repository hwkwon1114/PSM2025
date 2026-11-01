# Current joint-calibration figure set

Editable source for the associated manuscript: `docs/calibration/bayesian_calibration/calibration_manuscript_draft.tex` (a rendered PDF is alongside it). Each figure is available as a vector/text PDF and a PNG; see `figure_captions.md` for methods, selection, axes and limitations.

- `optimization_history.pdf`: the broader joint-search history, including the parallel comparison arms, warm-start boundaries and a final-stage detail. Earlier saved scores were partly rescored with the revised objective; the stages are not one uninterrupted search.
- `optimization_history_final_stage_original.pdf`: the earlier final-stage-only figure, preserved for comparison (with `figure_data_before_full_history.json` as its provenance).
- `gp_sensitivity.pdf`: four conditional, post-hoc GP response slices, not parameter confidence intervals.
- `shape_fit_maps.pdf`: all three cases, height fields, signed height residuals and scan-to-model distance distributions.
- `shape_fit_sections.pdf`: matched x–y sections at the three fixed z locations.
- `shape_fit_sections_data.npz`: numerical section curves.
- `figure_data_and_provenance.json`: selected values, plots, GP diagnostics and SHA-256 inputs.

**Public labels:** Run 1 is historical `run6` (joint fit), Run 2 is historical `run9` (exploratory check), Run 3 is historical `run10` (joint fit). `calibration/case_labels.py` is the central mapping for current code; `python -m calibration.case_labels` lists it. Historical identifiers and files have not been renamed, because they anchor the archived evaluation and cache identities. Do not replace identifiers in old configurations or sequence data.

Run 2 was known during development and was evaluated after a failed short-limit solve and an approved longer-limit retry. It is not a pristine, one-shot, unseen test. Its 0.641 mm symmetric RMS and the old 1.250 mm value use different metrics and must not be compared directly. The earlier Attempt-209 parameters rescore to 0.678 mm with the revised metric.

To regenerate, from the repository root in the DHD environment, run on a compute node:

```bash
~/bin/srun-here --cpus 2 --mem 6G --time 00:10:00 --env DHD -- python -u calibration/analysis/current_calibration_paper.py
```

This reads and checks existing saved results; it does not run the shell solver or optimize parameters. Rebuilding requires the original run directories referenced in the provenance file. The generator checks case identities, common frozen parameters, stored distances and input hashes.
