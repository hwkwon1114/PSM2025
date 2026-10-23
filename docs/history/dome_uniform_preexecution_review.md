# Uniform middle-profile diagnostic — delta review

## Verdict: `APPROVED_WITH_CAVEATS`

**No unresolved scientific or operational launch blockers for this one fresh collection run.** Approval is limited to the reviewed slice; it does not grant execution permission, retries, profile optimization or dome qualification.

## Scope and authority

Repository: `/gpfs/projects/p33082/PSM2025-zigzag`; paths below are repository-relative.

Authority inspected:
- `.pi/goals/active_goal_2026101722293411_mvd9obky-2qqxhb.md` — version 3, revision 2045.
- `docs/reports/dome_toolpath_investigation.md:142–148` — **Researcher-authorized profile diagnostic**, alongside the frozen scoring/verification protocol.

The explicit researcher selection supersedes the earlier profile-freeze **only for this diagnostic**.

Reviewed command, **not executed**:

```bash
srun-here --cpus 8 --mem 32G --time 00:30:00 \
  /gpfs/home/pxl1051/miniforge/envs/smcpp_vtk38/bin/python \
  scripts/run_dome.py \
  --sequence calibration/configs/dome/middle_n18_uniform.json \
  --output run/dome/middle_n18_uniform_v1
```

The proposed output directory was absent in the inspected listing.

## Recorded identities

Inspected `run/dome/checks/{uniform,density,baseline}_identity.sha256`; the retained comparator’s `execution.json` corroborates its configuration, runner and binary identities.

| Artifact | SHA-256 |
|---|---|
| `scripts/prepare_dome_uniform.py` | `02ce9b0111c834c15688aa23adeddc7653813990309b056a3f32dd2cc66c5d05` |
| `calibration/configs/dome/middle_n18_uniform.json` | `8ffe4a0c82ab29bee16f9b372da9f51933ad4f00d8eae97438b0bdea69678a69` |
| `calibration/configs/dome/middle_n18_repeat2.json` | `ad3f2730396f0b10c04f8bc54aad5e9dc4e20494a25953d6f727350953925960` |
| `scripts/run_dome.py` | `7426d6c51dbeb3c95509dc0e1130e417f5c207f32a713e2e316f314bd998df82` |
| `bin/shell` | `11069e3a1fbf180cabe7a8c87cfac4a96f0cb3b4850c727ad64535c0648f0c8c` |

**Hashes were inspected, not independently recomputed.** Approval is bounded to these identities and reviewed contents; consequential drift requires reassessment.

## Checks performed

### Exact delta and effective semantics

Manual comparison found only A’s added operation profile at `calibration/configs/dome/middle_n18_uniform.json:49–51`. Geometry, repeats **2/3/3**, Attempt209 scalar coefficients, hardening, ordering and disabled BCs remain identical. B/C already override to uniform.

`scripts/prepare_dome_uniform.py:9–14` checks absence of an existing A override, verifies exact restoration after removing the addition, and refuses output overwrite. Assertions were inspected, not rerun.

Native behavior is unambiguous:

- `src/libshell/ZigZagSequenceGrowth.hpp:296–324,634–639` inherits defaults, then overrides the supplied mode. Inherited end ratio **0.4** and power **2** remain stored but inactive under uniform.
- `src/simulations/Sim_Bilayer_Growth.cpp:2381–2384` transfers that effective profile to hit construction.
- `src/libshell/ZigZagGrowth.hpp:825–840` leaves the uniform multiplier at **1**, applies it only to `gtop`, and leaves `gbot` unchanged.

Thus A changes from
\[
p(s)=0.4+0.6\sin^2(\pi s)
\]
to \(p(s)=1\), not a mixture with the inherited center peak.

Hit membership/order remain geometric. Both profiles retain positive growth, so plastic-hit counting and \(q(n)=e^{-\beta n}=0.8^n\) remain unchanged (`src/libshell/ZigZagSequenceGrowth.hpp:653–670`). The supplied orthotropies give positive directional increments; uniform introduces no nonpositive stretch. Unchanged multiplicative/SPD safeguards remain applicable.

### Execution and reused evidence

Inspected `scripts/run_dome.py` retains:
- **Eight accumulated cycles before one terminal free equilibrium**; no intermediate solve or adaptive continuation.
- Zero slope feedback, multiplicative updates, disabled checkpoint/control routes and seed escape.
- Gradient ≤**5e−14**, positive-pivot acceptance, endpoint-proof/history checks.
- Eight threads, native cooperative **600 s**, external **1500 s**, existing-output refusal and no automatic restart.

Reused unchanged numerical evidence from:
- `docs/reports/dome_baseline_preexecution_review.md`
- `docs/reports/dome_density_preexecution_review.md`
- `docs/reports/dome_side_density_preexecution_review.md`

Saved logs report three baseline-input and two runner tests passing. Comparator result/history confirm eight cycles and one accepted solve, gradient **1.8478921932612982e−14**. No new solver route or C++ change is introduced by this slice.

### Scientific relevance and decisions

Retained comparator metrics show **52.817708%** geometric qualifying coverage, **29.5625%** director coverage and **39.744792%** saddle area at the fixed 15 mm scale. This supports testing whether the profile intervention reduces saddle structure without assuming its cause.

- Improvement without qualification: report and discuss; **no automatic tuning**.
- No improvement: retain the diagnostic outcome, not proof of impossibility.
- Failed equilibrium: preserve as numerically unresolved and unscored; no restart.
- Apparent qualification: proceed through existing verification gates, not immediate success.

## Launch-blocking findings

**None.**

## Nonblocking caveats and approval limits

1. **Not equal-dose identification.** Uniform increases integrated top treatment as well as changing its distribution. Any difference is a combined profile intervention, not evidence that modulation alone caused the saddle.
2. **Collection is not qualification.** Keep the fixed ROI, reference-area denominator, separate coherent signs, weaker curvature ≥0.1 m⁻¹, ratio ≥0.1 and 80% coverage. Invalid fits remain failures. Preserve director/geometric disagreement handling, spatial/full-shape vetoes, sensitivity reporting and tighter-tolerance/finer-mesh verification.
3. **Unchanged resolution and transfer limits.** The 15 mm estimator can smooth finer structure. Widths 8/4 mm still extrapolate beyond the calibrated 2 mm band. Results concern model-conditional local equilibria—not physical validation, global optimality or cadence equivalence.
4. **Evidence limits.** No simulations, tests, builds, installations, edits, delegation, independent hash calculations or numerical recomputations were performed. Prior numerical evidence does not guarantee convergence of this candidate. The native limit is cooperative; runtime enforcement is not claimed. Existing dirty-tree changes were preserved; no historical cumulative quota was imported.