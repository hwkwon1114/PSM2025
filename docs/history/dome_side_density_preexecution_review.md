# Side-density delta review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers for this one fresh collection run.** This does not authorize execution, retries, additional candidates or dome qualification.

## Scope and authority

Read:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 1859.
- `docs/reports/dome_toolpath_investigation.md` — especially **Side-density contrast — predeclared** and the frozen scoring/stopping protocol.
- `docs/reports/dome_baseline_preexecution_review.md`, `docs/reports/dome_density_preexecution_review.md`, and `docs/reports/dome_length168_preexecution_review.md` — unchanged numerical/test evidence reused within their stated limits.

Reviewed command, **not executed**:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/middle_n18_sides_n14.json \
  --output run/dome/middle_n18_sides_n14_v1
```

The output directory was absent in the inspected listing.

## Checks performed

**Exact delta:** Manual comparison against `calibration/configs/dome/middle_n18_repeat2.json` found only B/C angle **2.86→1.430891319512728°** and count **8→14**, at `calibration/configs/dome/middle_n18_sides_n14.json:57–58,85–86`. Calibration, hardening, profiles, widths, lengths, placement, rotations, order, repeats **2/3/3**, and disabled BCs remain identical.

**Footprints:** `scripts/prepare_dome_side_density.py:10–19` correctly preserves
\[
W=(N-2)L_v\tan\alpha
\]
and checks equal envelopes and exact restoration after reversing the declared changes. This agrees with native construction and placement checks in `src/libshell/ZigZagGrowth.hpp:199–245,663–735`. Assertions were inspected, not rerun.

Unchanged width-inflated bounds, in mm:

| Path | u | v |
|---|---|---|
| A | ±93.998231 | ±74 |
| B | ±100 | 86.624745 to 149.375255 |
| C | ±100 | −149.375255 to −86.624745 |

All fit the 200×300 mm sheet.

**Execution:** Inspected `scripts/run_dome.py` retains eight accumulation cycles before **one terminal free equilibrium**, no intermediate solves, adaptive continuation or restart. Gradient ≤**5e−14**, positive-pivot acceptance, endpoint-proof/history checks, existing-output refusal, native cooperative **600 s** and external **1500 s** limits remain. Unchanged CPU-double/SPD/solver evidence is reused, not re-established.

**Relevance:** Retained metrics and execution records corroborate accepted N18-repeat2 coverage **52.817708%**, versus length168 **52.598958%**, both nonqualifying. Spatial densification is distinct from repeating existing bands; improved continuity and worsened anisotropy are credible rival outcomes. Preserve failed equilibrium as unscored/unresolved. No material improvement should trigger summary and researcher consultation before broader changes—not a grid.

## Recorded identities

Inspected `run/dome/checks/{side_density,density,baseline}_identity.sha256`; comparator execution manifests corroborate the retained comparator/runner/binary identities.

| Artifact | SHA-256 |
|---|---|
| `scripts/prepare_dome_side_density.py` | `4c2e911b64e6deb136a9634ee7d5d9fda981c1590dea7c155f2082e244eb8002` |
| `calibration/configs/dome/middle_n18_sides_n14.json` | `f345c4687d5b1e46dd0a13783ce62a39bae78c27f77b665062ce77d320758203` |
| `calibration/configs/dome/middle_n18_repeat2.json` | `ad3f2730396f0b10c04f8bc54aad5e9dc4e20494a25953d6f727350953925960` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

**Recorded hashes were not independently recomputed.** Review applies only to these identities and inspected contents; consequential drift requires reassessment.

## Launch-blocking findings

None.

## Nonblocking caveats and limits

- Side inclined-strip pitch halves from approximately **9.792 to 4.896 mm**, below the frozen **15 mm** estimator radius. Apparent smoothing may conceal finer saddle structure. Equal envelopes establish neither discrete hit coverage nor treatment continuity.
- Count, angle, treated area, overlap and shared hardening remain coupled. This is a matched layout intervention, not equal-dose mechanism identification.
- Collection acceptance is not qualification. Retain the fixed ROI/reference-area denominator, separate coherent signs, invalid-fit failures, weaker curvature ≥0.1 m⁻¹, ratio ≥0.1 and 80% coverage; retain spatial/full-shape vetoes, director disagreement handling, sensitivity reporting and tighter-tolerance/finer-mesh verification.
- Prior width/profile calibration-transfer, endpoint-audit and direction-mapping limitations remain. Results are model-conditional local equilibria—not physical validation, global optimality, cadence equivalence or restart evidence.

No simulations, tests, builds, installations, edits or delegation were performed. Existing working-tree changes were preserved. No historical cumulative quota was imported; runtime enforcement is not claimed.