#!/usr/bin/env python3
"""Serial solver-characteristics study for the target-metric shell model.

Four families, all executed strictly one case after another under a single
exclusive lock:

  path_dependence         Forward vs reverse execution of one fixed path set,
                          compared at the final accumulated target metric, plus
                          same-endpoint sparse-equilibrium-cadence controls.
  hlbfgs_reproducibility  Byte-level run-to-run comparison of repeated
                          identical HLBFGS runs at a fixed thread count, and
                          the same inputs at a single thread. This measures
                          run-to-run variability of the corrector under
                          identical inputs (OpenMP reduction order, thread
                          count); it is NOT an initial-perturbation study and
                          no perturbation is ever applied.
  shrinking_crown         Broad-to-central shrinking zigzag footprint recipe,
                          with a central-to-broad reverse control and three
                          constant-footprint controls.
  prescribed_deformation  Impose a deformation mid-sequence, then continue with
                          the next path, either re-equilibrated first or not.
                          Requires the -prescribed_* CLI support in bin/shell;
                          gated behind --enable-prescribed and a binary
                          capability probe. Without support the case records a
                          blocked.json and never a synthetic result.

Every case directory receives sequence.json, manifest.json, result.json,
sequence_convergence.csv and the per-cycle VTP files written by the solver.
A case is only considered finished when the atomic completed.json marker is in
place; a failing case records its failure and does not stop unrelated cases.

Nothing here asserts physical accuracy: all starts are geometry-only, so the
outputs are solver characteristics, not validated physics.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "bin" / "shell"
RUN_ROOT = ROOT / "run" / "solver_characteristics"
LOCK_PATH = RUN_ROOT / ".solver_characteristics.lock"

# Plate geometry. IMPORTANT: -lx / -ly are HALF extents, not full lengths.
# RectangularPlate_RightAngle (src/libshell/Geometry.hpp:465-499) stores them as
# halfEdgeX / halfEdgeY and lays vertices over [-halfEdgeX, +halfEdgeX] x
# [-halfEdgeY, +halfEdgeY], and Sim_Bilayer_Growth.cpp passes -lx/-ly straight
# in for -geometry rectangle. The flag values below are kept identical to the
# completed equilibrium/continuation study for comparability, so the actual
# plate is 254 mm x 304.8 mm with material coordinates spanning
# u in [-127, +127] mm and v in [-152.4, +152.4] mm.
PLATE_HALF_X_M = 0.127          # value passed to -lx
PLATE_HALF_Y_M = 0.1524         # value passed to -ly
PLATE_FULL_X_M = 2.0 * PLATE_HALF_X_M
PLATE_FULL_Y_M = 2.0 * PLATE_HALF_Y_M
PLATE_H_TOTAL_M = 0.0006

# Mesh resolutions reused from the completed equilibrium/continuation study.
# RectangularPlate_RightAngle takes edgeLength = 2 * lx * res, so:
#   res 0.03  -> edge 7.62 mm, 35 x 41 vertices, 2720 faces  (dX 7.47, dY 7.62)
#   res 0.015 -> edge 3.81 mm, 67 x 81 vertices, 10560 faces (dX 3.85, dY 3.81)
# Both reproduce the vertex/face counts of the existing study runs exactly.
RES_STANDARD = 0.03
RES_FINE = 0.015

TOL = 1e-12
GRAD_TOL = 10.0 * TOL
MIN_STEP = 0.015625
STEP_GROWTH = 2.0
MAX_RETRIES = 8
MAX_ITER = 50000

FAMILIES = (
    "path_dependence",
    "hlbfgs_reproducibility",
    "shrinking_crown",
    "prescribed_deformation",
)

PRESCRIBED_PROBE_FLAG = b"-prescribed_at_cycle"


# --------------------------------------------------------------------------
# Toolpath library
# --------------------------------------------------------------------------
def zigzag(path_id: str, *, lv_mm: float, alpha_deg: float, n_strips: int,
           width_mm: float, center_uv_mm=(0.0, 0.0), rotation_deg: float = 0.0,
           gtop: float = 0.0, gbot: float = 0.0, ortho: float = 0.0) -> dict:
    """One toolpath entry in the zigzag_sequence schema (mm / deg units)."""
    return {
        "id": path_id,
        "enabled": True,
        "repeat": 1,
        "operation": {
            "type": "zigzag",
            "lv_mm": lv_mm,
            "alpha_deg": alpha_deg,
            "n_strips": n_strips,
            "width_mm": width_mm,
            "center_uv_mm": [float(center_uv_mm[0]), float(center_uv_mm[1])],
            "rotation_deg": rotation_deg,
            "gtop": gtop,
            "gbot": gbot,
            "ortho": ortho,
        },
    }


# Eight distinct paths that deliberately overlap over the plate centre, so the
# accumulated target metric is order sensitive under multiplicative
# composition. Path parameters are kept exactly in the style of the completed
# equilibrium/continuation study for comparability; on the actual 254 x 304.8 mm
# plate their footprints (nominally 180 x 140 mm at lv 140 mm, alpha 9.13 deg,
# 10 strips) sit inside the panel without clipping and work the central region
# only, which is what these families need - they probe corrector behaviour, not
# a crown recipe. id, rotation_deg, center_uv_mm, n_strips, width_mm, gtop, ortho.
_DEPENDENCE_SPEC = (
    ("D1_center_long", 0.0, (0.0, 0.0), 10, 10.0, 1.5e-4, 0.0),
    ("D2_center_trans", 90.0, (0.0, 0.0), 10, 10.0, 1.5e-4, 0.0),
    ("D3_diag_pos", 45.0, (0.0, 0.0), 8, 12.0, 1.2e-4, 0.2),
    ("D4_diag_neg", -45.0, (0.0, 0.0), 8, 12.0, 1.2e-4, 0.2),
    ("D5_offset_east", 0.0, (30.0, 0.0), 8, 10.0, 1.0e-4, 0.0),
    ("D6_offset_west", 0.0, (-30.0, 0.0), 8, 10.0, 1.0e-4, 0.0),
    ("D7_offset_north", 90.0, (0.0, 40.0), 8, 10.0, 1.0e-4, 0.0),
    ("D8_offset_south", 90.0, (0.0, -40.0), 8, 10.0, 1.0e-4, 0.0),
)

DEPENDENCE_PATHS = tuple(
    zigzag(path_id, lv_mm=140.0, alpha_deg=9.13, n_strips=n_strips,
           width_mm=width_mm, center_uv_mm=center, rotation_deg=rotation,
           gtop=gtop, ortho=ortho)
    for path_id, rotation, center, n_strips, width_mm, gtop, ortho
    in _DEPENDENCE_SPEC
)

# Nested-box crown recipe.
#
# Recipe source: Eastwood English-wheel crowning demonstration
# (https://www.youtube.com/watch?v=GjzI7aAjno8), as transcribed by the study
# owner: stay off the panel edges, start with one broad box, then work
# progressively smaller nested boxes with shorter strokes toward the centre;
# holding coverage constant in the centre yields a less graded crown;
# cross-wheeling at 90 deg balances the two curvatures; staggering the turn
# points keeps reversal ridges from stacking on one line.
#
# Model mapping: one nested box is one zigzag toolpath. The zigzag footprint is
# Lv (stroke length) by (n_strips - 2) * Lv * tan(alpha) (box width), so
# prescribing the wanted half-extents fixes alpha exactly at a constant strip
# count. Strokes therefore shorten monotonically toward the centre while the
# deposited strain per pass is held constant, so the crown grading comes purely
# from nested-box overlap.
#
# Not calibrated: Rossi & Nicholas (2018) wheel 250 x 250 mm, 1.5 mm aluminium
# with 5-35 mm tracking spacing. The plate, thickness and per-pass strain used
# here are model-side choices for solver characterisation only; no parameter is
# matched to that publication and no physical accuracy is claimed.
CROWN_N_STRIPS = 8
CROWN_GTOP = 1.2e-4
# Half-extents over the plate's material domain (see the plate block above).
# The outermost box is inset 17 mm / 16.4 mm from the plate edge (23 mm /
# 24.4 mm once the strip half-width is counted), i.e. the recipe's "avoid the
# edges" rule, and the ladder runs broad-to-central across the whole panel.
CROWN_HALF_EXTENTS_MM = (
    (110.0, 136.0),
    (92.0, 114.0),
    (74.0, 92.0),
    (56.0, 70.0),
    (38.0, 48.0),
    (20.0, 26.0),
)
# Square nested ladder used by the cross-wheeled case, so a 90 deg stage still
# fits inside the plate without clipping.
CROWN_SQUARE_HALF_EXTENTS_MM = tuple(
    (u_half, u_half) for u_half, _ in CROWN_HALF_EXTENTS_MM)
# Strip width is tied to the strip pitch so the boxes stay self-similar, but it
# is floored at 8 mm so every strip spans at least two elements of the res 0.015
# mesh (3.81 mm edge) and capped at 16 mm. The two innermost boxes hit the floor,
# so their strips overlap: that is the recipe's dense central working and it is
# recorded per cycle as max_hits_on_one_face_this_cycle, not hidden.
CROWN_WIDTH_MIN_MM = 8.0
CROWN_WIDTH_MAX_MM = 16.0
CROWN_WIDTH_PITCH_FRACTION = 0.55


def crown_path(index: int, path_id: str, *, ladder=CROWN_HALF_EXTENTS_MM,
               stagger: bool = False, rotation_deg: float = 0.0) -> dict:
    """Zigzag toolpath for nested box ``index`` of ``ladder``.

    ``stagger`` offsets every second box by half a strip pitch in u and a
    quarter of the stroke-length decrement in v, so its reversal points fall
    between the neighbouring boxes' reversal lines instead of on them. Box
    sizes are untouched, so the nesting order is identical either way.
    """
    u_half, v_half = ladder[index]
    lv_mm = 2.0 * v_half
    pitch_mm = 2.0 * u_half / (CROWN_N_STRIPS - 2)
    width_mm = min(CROWN_WIDTH_MAX_MM,
                   max(CROWN_WIDTH_MIN_MM,
                       CROWN_WIDTH_PITCH_FRACTION * pitch_mm))
    center = [0.0, 0.0]
    if stagger and index % 2 == 1:
        center = [round(0.5 * pitch_mm, 6),
                  round(0.25 * (ladder[index - 1][1] - v_half), 6)]
    return zigzag(path_id, lv_mm=round(lv_mm, 6),
                  alpha_deg=round(math.degrees(math.atan2(pitch_mm, lv_mm)), 6),
                  n_strips=CROWN_N_STRIPS, width_mm=round(width_mm, 6),
                  center_uv_mm=center, rotation_deg=rotation_deg,
                  gtop=CROWN_GTOP)


def crown_nested_paths(*, stagger: bool = True, crossed: bool = False,
                       reverse: bool = False) -> list[dict]:
    """Broad-to-central nested boxes; optionally cross-wheeled or reversed."""
    ladder = CROWN_SQUARE_HALF_EXTENTS_MM if crossed else CROWN_HALF_EXTENTS_MM
    order = range(len(ladder) - 1, -1, -1) if reverse else range(len(ladder))
    paths = []
    for stage, index in enumerate(order):
        rotation = 90.0 if (crossed and stage % 2 == 1) else 0.0
        label = f"u{int(ladder[index][0])}"
        if crossed:
            label += f"_rot{int(rotation)}"
        paths.append(crown_path(index, f"B{stage + 1}_box_{label}",
                                ladder=ladder, stagger=stagger,
                                rotation_deg=rotation))
    return paths


def crown_constant_paths(index: int, label: str) -> list[dict]:
    """One box size repeated once per stage, distinct ids, kept ordered.

    This is the recipe's constant-coverage counterexample: the same footprint
    is worked for the same number of ordered paths, so the total number of
    passes matches the nested recipe while the coverage never changes.
    """
    count = len(CROWN_HALF_EXTENTS_MM)
    return [crown_path(index, f"K{k + 1}_{label}") for k in range(count)]


def sequence_config(paths: list[dict]) -> dict:
    ids = [path["id"] for path in paths]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate toolpath ids: {ids}")
    return {
        "schema_version": 1,
        "units": {"length": "mm", "angle": "deg",
                  "growth": "engineering_strain"},
        "hardening": {"model": "none"},
        "defaults": {"start_mode": "left_bottom_up",
                     "profile": {"mode": "uniform"}},
        "toolpaths": list(paths),
    }


# --------------------------------------------------------------------------
# Case enumeration
# --------------------------------------------------------------------------
def make_case(*, family: str, name: str, purpose: str, paths: list[dict],
              res: float = RES_STANDARD, metric: str = "multiplicative",
              solver: str = "hlbfgs", adaptive: bool = True,
              initial_step: float = 0.5, minimize_every: int = 1,
              threads: int = 8, group: str | None = None,
              role: str = "reference", control_of: str | None = None,
              compare_with: tuple[str, ...] = (), replica: int = 1,
              endpoint_reference: str | None = None,
              endpoint_identical_abar: bool | None = None,
              endpoint_basis: str = "",
              extra_flags: tuple[str, ...] = (),
              requires_prescribed: bool = False) -> dict:
    """One executable case contract.

    ``role`` is one of reference / contrast / control / replica and, together
    with ``control_of`` and ``compare_with``, names the intended comparison so
    downstream analysis never has to guess pairings.

    ``endpoint_*`` declare whether the accumulated target metric endpoint is
    expected to be identical to ``endpoint_reference``. The declaration is a
    contract for the analyser's endpoint-matched assertion (elementwise abar
    comparison against the raw rest second fundamental form and pass history),
    not a measured result.
    """
    if adaptive and minimize_every != 1:
        raise ValueError(
            f"{name}: the solver rejects adaptive continuation with "
            "-sequence_minimize_every != 1")
    if role not in ("reference", "contrast", "control", "replica"):
        raise ValueError(f"{name}: unknown role {role!r}")
    if endpoint_reference is not None and endpoint_identical_abar is None:
        raise ValueError(f"{name}: endpoint_reference needs an explicit claim")
    return {
        "family": family,
        "name": name,
        "purpose": purpose,
        "path_order": [path["id"] for path in paths],
        "paths": paths,
        "res": res,
        "metric": metric,
        "solver": solver,
        "adaptive": adaptive,
        "initial_step": initial_step,
        "minimize_every": minimize_every,
        "threads": threads,
        "group": group or name,
        "role": role,
        "control_of": control_of,
        "compare_with": list(compare_with),
        "replica": replica,
        "endpoint": {
            "reference_case": endpoint_reference,
            "expected_identical_abar": endpoint_identical_abar,
            "basis": endpoint_basis,
            # Metric-only model: the rest second fundamental form is held fixed
            # by the solver and hardening/passE are off, so the endpoint
            # assertion may also require raw rest bbar and pass history to be
            # unchanged between the compared cases.
            "expected_identical_rest_bbar": True,
            "expected_identical_pass_history": True,
        },
        "extra_flags": list(extra_flags),
        "requires_prescribed": requires_prescribed,
        "tol": TOL,
        "grad_tol": GRAD_TOL,
    }


def path_dependence_cases() -> list[dict]:
    forward = list(DEPENDENCE_PATHS)
    reverse = list(reversed(DEPENDENCE_PATHS))
    cases: list[dict] = []

    # Order effect with the production adaptive corrector. The comparison is
    # made on the final accumulated target metric (and on the final shape) of
    # the forward and reverse runs.
    for order_name, paths in (("forward", forward), ("reverse", reverse)):
        is_reverse = order_name == "reverse"
        cases.append(make_case(
            family="path_dependence",
            name=f"order_{order_name}_multiplicative",
            purpose=(
                f"{order_name} execution of the 8-path set, multiplicative "
                "metric composition, adaptive corrector, equilibrium after "
                "every path; forward/reverse pair is compared at the final "
                "target metric"),
            paths=paths, metric="multiplicative", adaptive=True,
            minimize_every=1, group="order_multiplicative",
            role="contrast" if is_reverse else "reference",
            control_of=("order_forward_multiplicative" if is_reverse
                        else None),
            compare_with=("order_forward_multiplicative",
                          "order_reverse_multiplicative"),
            endpoint_reference=("order_forward_multiplicative" if is_reverse
                                else None),
            endpoint_identical_abar=False if is_reverse else None,
            endpoint_basis=(
                "multiplicative composition of overlapping increments does "
                "not commute, so the accumulated abar endpoints are expected "
                "to differ; rest bbar and pass history stay identical"
                if is_reverse else "")))

    # Same order pair under reference-additive linearised composition, where
    # the accumulated target metric is order independent by construction. Any
    # residual forward/reverse difference is therefore mechanical (state path),
    # not metric bookkeeping.
    for order_name, paths in (("forward", forward), ("reverse", reverse)):
        is_reverse = order_name == "reverse"
        cases.append(make_case(
            family="path_dependence",
            name=f"order_{order_name}_additive",
            purpose=(
                f"{order_name} execution with reference_additive_linearized "
                "composition; the accumulated target metric is order "
                "independent by construction, isolating mechanical path "
                "dependence"),
            paths=paths, metric="reference_additive_linearized",
            adaptive=True, minimize_every=1, group="order_additive",
            role="contrast" if is_reverse else "reference",
            control_of="order_forward_additive" if is_reverse else None,
            compare_with=("order_forward_additive", "order_reverse_additive"),
            endpoint_reference=("order_forward_additive" if is_reverse
                                else None),
            endpoint_identical_abar=True if is_reverse else None,
            endpoint_basis=(
                "reference-additive linearised increments sum, so the "
                "accumulated abar endpoint must match elementwise to solver "
                "round-off; any final-shape difference is mechanical"
                if is_reverse else "")))

    # Same-endpoint sparse-cadence controls. The adaptive corrector is refused
    # by the solver when minimize_every != 1, so the whole cadence block runs
    # with adaptive continuation off and a single full load step per minimised
    # cycle; cadence is then the only factor that varies inside the block.
    for order_name, paths in (("forward", forward), ("reverse", reverse)):
        for cadence in (1, 2, len(forward)):
            label = "end" if cadence == len(forward) else f"{cadence}"
            dense = cadence == 1
            cases.append(make_case(
                family="path_dependence",
                name=f"cadence_{order_name}_every{label}",
                purpose=(
                    f"{order_name} order, identical endpoint growth, "
                    f"equilibrium enforced every {cadence} path(s) "
                    f"({'only at the end' if cadence == len(forward) else 'sparse' if cadence > 1 else 'dense reference'}); "
                    "fixed full load step, adaptive continuation off"),
                paths=paths, metric="multiplicative", adaptive=False,
                initial_step=1.0, minimize_every=cadence,
                group=f"cadence_{order_name}",
                role="reference" if dense else "control",
                control_of=(None if dense
                            else f"cadence_{order_name}_every1"),
                compare_with=(f"cadence_{order_name}_every1",),
                endpoint_reference=(None if dense
                                    else f"cadence_{order_name}_every1"),
                endpoint_identical_abar=None if dense else True,
                endpoint_basis=(
                    "" if dense else
                    "identical path order and identical growth increments; "
                    "only the equilibrium cadence differs, so the accumulated "
                    "abar endpoint must match elementwise")))
    return cases


def reproducibility_cases(threads_fixed: int, replicas_fixed: int,
                          replicas_single: int) -> list[dict]:
    """Repeated identical HLBFGS runs; inputs are bitwise identical."""
    cases: list[dict] = []
    anchor = f"repeat_t{threads_fixed}_r01"
    for replica in range(1, replicas_fixed + 1):
        cases.append(make_case(
            family="hlbfgs_reproducibility",
            name=f"repeat_t{threads_fixed}_r{replica:02d}",
            purpose=(
                f"replica {replica} of {replicas_fixed} identical HLBFGS runs "
                f"at OMP_NUM_THREADS={threads_fixed}; no input is changed "
                "between replicas, so any difference is run-to-run corrector "
                "variability at fixed inputs"),
            paths=list(DEPENDENCE_PATHS), threads=threads_fixed,
            replica=replica, group=f"repeat_t{threads_fixed}",
            role="reference" if replica == 1 else "replica",
            control_of=None if replica == 1 else anchor,
            compare_with=() if replica == 1 else (anchor,),
            endpoint_reference=None if replica == 1 else anchor,
            endpoint_identical_abar=None if replica == 1 else True,
            endpoint_basis=(
                "" if replica == 1 else
                "bitwise identical inputs and identical thread count, so the "
                "accumulated abar endpoint must match elementwise; only the "
                "equilibrated state may differ")))
    for replica in range(1, replicas_single + 1):
        cases.append(make_case(
            family="hlbfgs_reproducibility",
            name=f"repeat_t1_r{replica:02d}",
            purpose=(
                f"replica {replica} of {replicas_single} identical HLBFGS runs "
                "at OMP_NUM_THREADS=1; single-thread comparison for the "
                f"{threads_fixed}-thread replica group"),
            paths=list(DEPENDENCE_PATHS), threads=1, replica=replica,
            group="repeat_t1",
            role="control" if replica == 1 else "replica",
            control_of=anchor if replica == 1 else "repeat_t1_r01",
            compare_with=(anchor,) if replica == 1 else ("repeat_t1_r01",),
            endpoint_reference=anchor if replica == 1 else "repeat_t1_r01",
            endpoint_identical_abar=True,
            endpoint_basis=(
                "identical inputs at a single thread; the accumulated abar "
                "endpoint must match elementwise, so any state difference is "
                "corrector/thread-order variability only")))
    return cases


def shrinking_crown_cases() -> list[dict]:
    nested = crown_nested_paths(stagger=True)
    last = len(CROWN_HALF_EXTENTS_MM) - 1
    middle = len(CROWN_HALF_EXTENTS_MM) // 2
    main = "crown_nested_broad_to_central"
    differing_footprint = (
        "the toolpath footprints themselves differ, so the accumulated abar "
        "endpoint is expected to differ; rest bbar and pass history stay "
        "identical")
    return [
        make_case(
            family="shrinking_crown", name=main,
            purpose=(
                "recipe under test: broad box first, then progressively "
                "smaller nested boxes with shorter strokes toward the centre, "
                "edges avoided, turn points staggered, constant deposited "
                "strain per pass"),
            paths=nested, res=RES_FINE, group="crown_nested",
            role="reference",
            compare_with=("crown_nested_central_to_broad",
                          "crown_nested_aligned_turns",
                          "crown_nested_crossed",
                          "crown_constant_coverage_central")),
        make_case(
            family="shrinking_crown", name="crown_nested_central_to_broad",
            purpose=(
                "reverse control: the same nested boxes worked central-to-"
                "broad, so only the nesting order changes"),
            paths=crown_nested_paths(stagger=True, reverse=True),
            res=RES_FINE, group="crown_nested", role="contrast",
            control_of=main, compare_with=(main,),
            endpoint_reference=main, endpoint_identical_abar=False,
            endpoint_basis=(
                "identical box set but reversed order under multiplicative "
                "composition, which does not commute on the overlapping "
                "central region")),
        make_case(
            family="shrinking_crown", name="crown_nested_aligned_turns",
            purpose=(
                "turn-point control: identical nested boxes, concentric and "
                "unstaggered, so reversal points stack on the same lines"),
            paths=crown_nested_paths(stagger=False), res=RES_FINE,
            group="crown_turns", role="control", control_of=main,
            compare_with=(main,), endpoint_reference=main,
            endpoint_identical_abar=False,
            endpoint_basis=(
                "the staggered boxes sit at shifted centres, so the covered "
                "faces and the abar endpoint differ; the comparison targets "
                "reversal-ridge structure, not endpoint equality")),
        make_case(
            family="shrinking_crown", name="crown_nested_crossed",
            purpose=(
                "cross-wheeled variant: square nested ladder with every "
                "second box rotated 90 deg, testing whether alternating "
                "direction balances the two curvatures"),
            paths=crown_nested_paths(stagger=True, crossed=True),
            res=RES_FINE, group="crown_crossed", role="contrast",
            control_of=main, compare_with=(main,), endpoint_reference=main,
            endpoint_identical_abar=False,
            endpoint_basis=differing_footprint),
        make_case(
            family="shrinking_crown", name="crown_constant_coverage_broad",
            purpose=(
                "constant-coverage control: the broadest box worked for the "
                "same number of ordered paths, coverage never shrinks"),
            paths=crown_constant_paths(0, "broad"), res=RES_FINE,
            group="crown_constant", role="control", control_of=main,
            compare_with=(main,), endpoint_reference=main,
            endpoint_identical_abar=False,
            endpoint_basis=differing_footprint),
        make_case(
            family="shrinking_crown", name="crown_constant_coverage_mid",
            purpose=(
                "constant-coverage control at the mid box size, same number "
                "of ordered paths"),
            paths=crown_constant_paths(middle, "mid"), res=RES_FINE,
            group="crown_constant", role="control", control_of=main,
            compare_with=(main,), endpoint_reference=main,
            endpoint_identical_abar=False,
            endpoint_basis=differing_footprint),
        make_case(
            family="shrinking_crown", name="crown_constant_coverage_central",
            purpose=(
                "constant-coverage control at the central box size; the "
                "recipe predicts a less graded crown than the nested run"),
            paths=crown_constant_paths(last, "central"), res=RES_FINE,
            group="crown_constant", role="control", control_of=main,
            compare_with=(main,), endpoint_reference=main,
            endpoint_identical_abar=False,
            endpoint_basis=differing_footprint),
    ]


# Prescribed-deformation family. Flag spellings follow the CLI contract owned
# by the C++ side: -prescribed_at_cycle, -prescribed_shape,
# -prescribed_amplitude, -prescribed_reference, -prescribed_reequilibrate,
# -prescribed_disp_csv, -prescribed_tag.
PRESCRIBED_AT_CYCLE = 4
PRESCRIBED_SHAPE = "crown"
PRESCRIBED_AMPLITUDE_M = 0.004


def prescribed_cases() -> list[dict]:
    forward = list(DEPENDENCE_PATHS)
    common = (
        "-prescribed_at_cycle", str(PRESCRIBED_AT_CYCLE),
        "-prescribed_shape", PRESCRIBED_SHAPE,
        "-prescribed_amplitude", repr(PRESCRIBED_AMPLITUDE_M),
        "-prescribed_reference", "rest",
    )
    baseline = "prescribed_control_none"
    retained = (
        "the imposition retains abar, the rest second fundamental form and "
        "the pass history per the C++ contract, so the accumulated abar "
        "endpoint must match the baseline elementwise while the equilibrated "
        "state differs")
    return [
        make_case(
            family="prescribed_deformation", name=baseline,
            purpose=(
                "no imposition; in-family baseline for the prescribed pair, "
                "identical in every other respect"),
            paths=forward, group="prescribed", role="reference",
            compare_with=("prescribed_crown_reequilibrated",
                          "prescribed_crown_direct_next_path")),
        make_case(
            family="prescribed_deformation",
            name="prescribed_crown_reequilibrated",
            purpose=(
                f"impose the analytic {PRESCRIBED_SHAPE} field "
                f"({PRESCRIBED_AMPLITUDE_M} m, from rest) after cycle "
                f"{PRESCRIBED_AT_CYCLE}, re-equilibrate at the imposed state, "
                "then run the next path"),
            paths=forward, group="prescribed", role="contrast",
            control_of=baseline,
            extra_flags=common + ("-prescribed_reequilibrate", "true",
                                  "-prescribed_tag", "prescribed_reeq"),
            requires_prescribed=True,
            compare_with=("prescribed_crown_direct_next_path", baseline),
            endpoint_reference=baseline, endpoint_identical_abar=True,
            endpoint_basis=retained),
        make_case(
            family="prescribed_deformation",
            name="prescribed_crown_direct_next_path",
            purpose=(
                f"impose the same {PRESCRIBED_SHAPE} field after cycle "
                f"{PRESCRIBED_AT_CYCLE} and go straight into the next path "
                "without an intermediate re-equilibration"),
            paths=forward, group="prescribed", role="contrast",
            control_of="prescribed_crown_reequilibrated",
            extra_flags=common + ("-prescribed_reequilibrate", "false",
                                  "-prescribed_tag", "prescribed_direct"),
            requires_prescribed=True,
            compare_with=("prescribed_crown_reequilibrated", baseline),
            endpoint_reference=baseline, endpoint_identical_abar=True,
            endpoint_basis=retained),
    ]


def merge_flags(base, overrides) -> list[str]:
    """Flag/value pairs from ``base``, with ``overrides`` replacing by name.

    An override for an unknown flag is appended, so the prescribed family can
    be extended (for example to a per-vertex CSV field) without ever passing
    the same flag twice.
    """
    pairs: list[list[str]] = []
    for source in (list(base), list(overrides)):
        if len(source) % 2:
            raise SystemExit(f"flag list is not flag/value paired: {source}")
        for index in range(0, len(source), 2):
            flag, value = source[index], source[index + 1]
            for pair in pairs:
                if pair[0] == flag:
                    pair[1] = value
                    break
            else:
                pairs.append([flag, value])
    return [token for pair in pairs for token in pair]


def all_cases(*, threads_fixed: int, replicas_fixed: int,
              replicas_single: int,
              prescribed_extra: tuple[str, ...] = ()) -> list[dict]:
    cases = (path_dependence_cases()
             + reproducibility_cases(threads_fixed, replicas_fixed,
                                     replicas_single)
             + shrinking_crown_cases()
             + prescribed_cases())
    if prescribed_extra:
        for case in cases:
            if case["family"] == "prescribed_deformation" \
                    and case["requires_prescribed"]:
                case["extra_flags"] = merge_flags(case["extra_flags"],
                                                  prescribed_extra)
    names = [case["name"] for case in cases]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise RuntimeError(f"duplicate case names: {sorted(duplicates)}")
    return cases


# --------------------------------------------------------------------------
# Command construction and hashing
# --------------------------------------------------------------------------
def build_command(case: dict, sequence_path: Path) -> list[str]:
    command = [
        str(SHELL),
        "-sim", "bilayer_growth", "-case", "custom",
        "-geometry", "rectangle",
        "-lx", repr(PLATE_HALF_X_M), "-ly", repr(PLATE_HALF_Y_M),
        "-res", repr(case["res"]), "-h_total", repr(PLATE_H_TOTAL_M),
        "-growth_type", "zigzag_sequence",
        "-cycle_file", str(sequence_path.resolve()),
        "-sequence_warm_start", "true",
        "-metric_update", case["metric"],
        "-sequence_minimize_every", str(case["minimize_every"]),
        "-sequence_adaptive", "true" if case["adaptive"] else "false",
        "-sequence_initial_step", repr(case["initial_step"]),
        "-sequence_max_step", repr(case["initial_step"]),
        "-sequence_min_step", repr(MIN_STEP),
        "-sequence_step_growth", repr(STEP_GROWTH),
        "-sequence_max_retries", str(MAX_RETRIES),
        "-equilibrium_solver", case["solver"],
        "-equilibrium_grad_tol", repr(case["grad_tol"]),
        "-tol", repr(case["tol"]),
        "-max_iter", str(MAX_ITER),
        "-sequence_stability", "false",
        "-enable_passE", "false",
        "-nsteps", "1",
        "-seed_escape", "false",
        "-basename", case["name"],
        "-export_stl", "false",
    ]
    command.extend(case["extra_flags"])
    return command


def normalized_command(command: list[str]) -> list[str]:
    """Command with per-case-only tokens masked, for replica identity hashes."""
    masked = list(command)
    masked[0] = "<shell>"
    for flag, placeholder in (("-cycle_file", "<sequence>"),
                              ("-basename", "<basename>")):
        if flag in masked:
            masked[masked.index(flag) + 1] = placeholder
    return masked


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def relative(path: Path) -> str:
    """Repo-relative path for logs and metadata, absolute if outside ROOT."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False).stdout.strip()


def write_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent),
                                         prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: dict) -> None:
    write_atomic(path, json.dumps(payload, indent=2) + "\n")


def binary_supports_prescribed() -> bool:
    """Probe bin/shell for the prescribed-deformation flag literal."""
    if not SHELL.exists():
        return False
    tail = b""
    with SHELL.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            if PRESCRIBED_PROBE_FLAG in tail + block:
                return True
            tail = block[-len(PRESCRIBED_PROBE_FLAG):]
    return False


# --------------------------------------------------------------------------
# Result verification
# --------------------------------------------------------------------------
def expected_accepted_cycles(case: dict) -> list[int]:
    total = len(case["paths"])
    cadence = case["minimize_every"]
    return sorted({cycle for cycle in range(1, total + 1)
                   if cycle % cadence == 0 or cycle == total})


def prescribed_reequilibration_requested(case: dict) -> bool | None:
    """True/False when the case asks for re-equilibration, None if disabled."""
    flags = case["extra_flags"]
    if "-prescribed_at_cycle" not in flags:
        return None
    if "-prescribed_reequilibrate" not in flags:
        return False
    value = flags[flags.index("-prescribed_reequilibrate") + 1]
    return value.strip().lower() in ("1", "true", "yes", "on")


def summarize(case: dict, case_dir: Path) -> dict:
    with (case_dir / "sequence_convergence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # The prescribed-deformation branch writes state_kind prescribed_imposed
    # (a deliberately non-equilibrated diagnostic state) and prescribed_reeq
    # (an equilibrium claim) at the boundary cycle. Both are excluded from the
    # physical toolpath statistics and reported separately, so a prescription
    # can never perturb the physical accepted-cycle set or residual set.
    physical = [row for row in rows
                if not row["state_kind"].startswith("prescribed")]
    prescribed_rows = [row for row in rows
                       if row["state_kind"].startswith("prescribed")]
    attempted = [row for row in physical
                 if row["minimization_performed"] == "1"]
    accepted = [row for row in attempted if row["equilibrium_accepted"] == "1"]
    rejected = [row for row in attempted if row["equilibrium_accepted"] == "0"]
    accepted_residuals = [float(row["final_gradient_norm"]) for row in accepted]
    accepted_cycles = sorted({int(row["executed_cycle_index"])
                              for row in accepted})
    expected = expected_accepted_cycles(case)
    residual_bound = case["grad_tol"]
    convergence = {
        "attempts": len(attempted),
        "accepted_substeps": len(accepted),
        "rejected_substeps": len(rejected),
        "accepted_cycles": accepted_cycles,
        "expected_accepted_cycles": expected,
        "all_expected_cycles_accepted": accepted_cycles == expected,
        "accepted_residual_bound": residual_bound,
        "max_accepted_residual": max(accepted_residuals, default=None),
        "min_accepted_residual": min(accepted_residuals, default=None),
        "accepted_residuals": accepted_residuals,
        "residuals_within_bound": bool(accepted_residuals) and all(
            residual <= residual_bound for residual in accepted_residuals),
        "final_energy": (float(accepted[-1]["recomputed_energy"])
                         if accepted else None),
        "prescribed_rows_excluded": len(prescribed_rows),
    }

    requested = prescribed_reequilibration_requested(case)
    imposed = [row for row in prescribed_rows
               if row["state_kind"] == "prescribed_imposed"]
    reeq = [row for row in prescribed_rows
            if row["state_kind"] == "prescribed_reeq"]
    reeq_residuals = [float(row["final_gradient_norm"]) for row in reeq]
    prescribed = {
        "enabled": requested is not None,
        "reequilibration_requested": requested,
        "boundary_cycles": sorted({int(row["executed_cycle_index"])
                                   for row in prescribed_rows}),
        # Free-DOF residual of the imposed, deliberately non-equilibrated
        # state; never counted as an accepted equilibrium.
        "imposed_states": len(imposed),
        "imposed_free_dof_residuals": [float(row["final_gradient_norm"])
                                       for row in imposed],
        "reequilibrated_states": len(reeq),
        "reequilibrated_accepted": [row["equilibrium_accepted"] == "1"
                                    for row in reeq],
        "reequilibrated_residuals": reeq_residuals,
        "reequilibrated_within_bound": all(
            residual <= residual_bound for residual in reeq_residuals),
    }
    if requested is None:
        prescribed["satisfied"] = not prescribed_rows
    elif requested:
        prescribed["satisfied"] = (
            len(imposed) >= 1 and len(reeq) >= 1
            and all(prescribed["reequilibrated_accepted"])
            and prescribed["reequilibrated_within_bound"])
    else:
        prescribed["satisfied"] = len(imposed) >= 1 and not reeq

    order: dict = {"expected": list(case["path_order"])}
    per_cycle: list[dict] = []
    prescribed_events: list[dict] = []
    summaries = sorted(case_dir.glob("*_summary.csv"))
    summary_path = summaries[0] if summaries else None
    if summary_path is not None:
        with summary_path.open(newline="") as stream:
            summary_rows = [row for row in csv.DictReader(stream)
                            if int(row["executed_cycle_index"]) >= 1]
        # The prescribed-deformation branch adds PRESCRIBED_IMPOSED /
        # PRESCRIBED_REEQ rows to the same summary file. They are events, not
        # toolpaths, so they are reported separately and excluded from the
        # executed-order check.
        toolpath_rows = [row for row in summary_rows
                         if not row["toolpath_id"].startswith("PRESCRIBED")]
        order["executed"] = [row["toolpath_id"] for row in toolpath_rows]
        order["matches"] = order["executed"] == order["expected"]
        for row in toolpath_rows:
            per_cycle.append({
                "cycle": int(row["executed_cycle_index"]),
                "toolpath_id": row["toolpath_id"],
                "covered_face_count": int(row["covered_face_count"]),
                "hit_event_count": int(row["hit_event_count"]),
                "max_displacement_m": float(row["max_displacement"]),
                "min_U3_m": float(row["min_U3"]),
                "max_U3_m": float(row["max_U3"]),
                "total_energy_J": float(row["total_energy"]),
                "final_vtp": row.get("final_file", ""),
            })
        for row in summary_rows:
            if not row["toolpath_id"].startswith("PRESCRIBED"):
                continue
            prescribed_events.append({
                "after_cycle": int(row["executed_cycle_index"]),
                "event": row["toolpath_id"],
                "max_displacement_m": float(row["max_displacement"]),
                "min_U3_m": float(row["min_U3"]),
                "max_U3_m": float(row["max_U3"]),
                "total_energy_J": float(row["total_energy"]),
                "retained_max_total_face_hit_count":
                    int(row["max_total_face_hit_count"]),
                "vtp": row.get("final_file", ""),
            })
        order["summary_csv"] = summary_path.name
    else:
        order["executed"] = None
        order["matches"] = False

    cycle_vtp = sorted(
        path.name for path in case_dir.glob("*_final.vtp")
        if "mapping" not in path.name)
    pending_vtp = sorted(path.name for path in case_dir.glob("*_pending.vtp"))
    final_vtp = cycle_vtp[-1] if cycle_vtp else None
    fingerprints = {
        "sequence_convergence_csv_sha256":
            sha256_file(case_dir / "sequence_convergence.csv"),
        "final_vtp": final_vtp,
        "final_vtp_sha256": (sha256_file(case_dir / final_vtp)
                             if final_vtp else None),
    }
    if summary_path is not None:
        fingerprints["summary_csv_sha256"] = sha256_file(summary_path)
    # Prescribed-deformation artefacts, named
    # <tag>_<prescribed_tag>_after_cycle_<KKK>_imposed{,_vertices}.* and
    # ..._reequilibrated{,_vertices}.* by the solver.
    prescribed_vtp = sorted(
        path.name for path in case_dir.glob("*_imposed.vtp"))
    prescribed_vtp += sorted(
        path.name for path in case_dir.glob("*_reequilibrated.vtp"))
    prescribed_vertex_csv = sorted(
        path.name for path in case_dir.glob("*_vertices.csv"))
    return {
        "convergence": convergence,
        "prescribed": prescribed,
        "path_order": order,
        "per_cycle": per_cycle,
        "prescribed_events": prescribed_events,
        # Flat keys downstream analysis reads directly.
        "final_vtp": final_vtp,
        "cycle_vtp": cycle_vtp,
        "pending_vtp": pending_vtp,
        "prescribed_vtp": prescribed_vtp,
        "prescribed_vertex_csv": prescribed_vertex_csv,
        "cycle_vtp_count": len(cycle_vtp),
        "pending_vtp_count": len(pending_vtp),
        "fingerprints": fingerprints,
    }


def acceptance(case: dict, report: dict) -> dict:
    convergence = report["convergence"]
    checks = {
        "all_expected_cycles_accepted":
            convergence["all_expected_cycles_accepted"],
        "residuals_within_bound": convergence["residuals_within_bound"],
        "path_order_preserved": bool(report["path_order"]["matches"]),
        "per_cycle_vtp_written":
            report["cycle_vtp_count"] + report["pending_vtp_count"]
            >= len(case["paths"]),
    }
    # Only asserted for the prescribed family: the imposed state must exist and,
    # when re-equilibration was requested, that solve must be accepted within
    # the same residual bound as a physical cycle.
    if report["prescribed"]["enabled"]:
        checks["prescribed_states_as_requested"] = \
            bool(report["prescribed"]["satisfied"])
    return {"checks": checks, "ok": all(checks.values())}


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def case_hashes(case: dict, command: list[str], sequence_text: str) -> dict:
    spec = {key: case[key] for key in
            ("family", "name", "res", "metric", "solver", "adaptive",
             "initial_step", "minimize_every", "tol", "grad_tol",
             "path_order", "extra_flags")}
    sequence_sha256 = sha256_text(sequence_text)
    binary_sha256 = sha256_file(SHELL)
    return {
        "spec_sha256": sha256_text(canonical(spec)),
        "sequence_sha256": sequence_sha256,
        "binary_sha256": binary_sha256,
        "command_sha256": sha256_text(canonical(command)),
        "identity_sha256": sha256_text(canonical({
            "command": normalized_command(command),
            "sequence": sequence_sha256,
            "binary": binary_sha256,
        })),
    }


def cached(case_dir: Path, hashes: dict) -> bool:
    marker = case_dir / "completed.json"
    result = case_dir / "result.json"
    if not (marker.exists() and result.exists()):
        return False
    try:
        completed = json.loads(marker.read_text())
        finished = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (completed.get("status") == "completed"
            and completed.get("hashes") == hashes
            and finished.get("return_code") == 0
            and finished.get("acceptance", {}).get("ok") is True)


def run_case(case: dict, *, timeout: int, force: bool,
             prescribed_enabled: bool, prescribed_available: bool,
             prepare_only: bool = False) -> dict:
    case_dir = RUN_ROOT / case["family"] / case["name"]
    sequence_path = case_dir / "sequence.json"
    sequence_text = json.dumps(sequence_config(case["paths"]), indent=2) + "\n"
    command = build_command(case, sequence_path)
    hashes = case_hashes(case, command, sequence_text)

    if case["requires_prescribed"]:
        blocker = None
        if not prescribed_enabled:
            blocker = ("prescribed-deformation family is gated; rerun with "
                       "--enable-prescribed once bin/shell carries the "
                       "-prescribed_* flags")
        elif not prescribed_available:
            blocker = ("bin/shell does not contain the "
                       f"{PRESCRIBED_PROBE_FLAG.decode()} flag literal; the "
                       "C++ prescribed-deformation support is not built yet")
        if blocker is not None:
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(case_dir / "blocked.json", {
                "case": case["name"],
                "family": case["family"],
                "status": "blocked",
                "reason": blocker,
                "command_that_would_run": command,
                "hashes": hashes,
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"BLOCKED {case['name']}: {blocker}", flush=True)
            return {"name": case["name"], "family": case["family"],
                    "status": "blocked", "reason": blocker}

    if not force and not prepare_only and cached(case_dir, hashes):
        print(f"SKIP {case['name']} (matching completed marker)", flush=True)
        return {"name": case["name"], "family": case["family"],
                "status": "cached"}
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(sequence_path, sequence_text)

    manifest = {
        # Flat keys downstream analysis reads directly.
        "family": case["family"],
        "case_name": case["name"],
        "purpose": case["purpose"],
        "group": case["group"],
        "role": case["role"],
        "control_of": case["control_of"],
        "compare_with": case["compare_with"],
        "replica": case["replica"],
        "endpoint": case["endpoint"],
        "case": {key: value for key, value in case.items() if key != "paths"},
        "toolpaths": case["paths"],
        "path_order": case["path_order"],
        "path_count": len(case["paths"]),
        "plate": {
            # -lx / -ly are half extents; both forms are given explicitly so
            # no downstream consumer has to guess.
            "lx_flag_is_half_extent": True,
            "half_extent_x_m": PLATE_HALF_X_M,
            "half_extent_y_m": PLATE_HALF_Y_M,
            "full_length_x_m": PLATE_FULL_X_M,
            "full_length_y_m": PLATE_FULL_Y_M,
            "material_u_range_mm": [-1e3 * PLATE_HALF_X_M,
                                    1e3 * PLATE_HALF_X_M],
            "material_v_range_mm": [-1e3 * PLATE_HALF_Y_M,
                                    1e3 * PLATE_HALF_Y_M],
            "h_total_m": PLATE_H_TOTAL_M,
        },
        "command": command,
        "omp_num_threads": case["threads"],
        "hashes": hashes,
        "expected_accepted_cycles": expected_accepted_cycles(case),
        "accepted_residual_bound": case["grad_tol"],
        "git_revision": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(case_dir / "manifest.json", manifest)
    if prepare_only:
        print(f"PREPARED {case['family']}/{case['name']} "
              f"-> {relative(case_dir)}", flush=True)
        return {"name": case["name"], "family": case["family"],
                "group": case["group"], "status": "prepared",
                "identity_sha256": hashes["identity_sha256"]}

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(case["threads"])
    env["OMP_DYNAMIC"] = "false"
    env["MKL_NUM_THREADS"] = str(case["threads"])
    env["OPENBLAS_NUM_THREADS"] = str(case["threads"])

    print(f"RUN {case['family']}/{case['name']} "
          f"(threads={case['threads']}, paths={len(case['paths'])})",
          flush=True)
    started = time.monotonic()
    with (case_dir / "run.log").open("w") as log:
        try:
            completed = subprocess.run(command, cwd=case_dir, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
            return_code, timed_out = completed.returncode, False
        except subprocess.TimeoutExpired:
            return_code, timed_out = 124, True

    result = {
        "case": case["name"],
        "case_name": case["name"],
        "family": case["family"],
        "group": case["group"],
        "role": case["role"],
        "control_of": case["control_of"],
        "compare_with": case["compare_with"],
        "endpoint": case["endpoint"],
        "replica": case["replica"],
        "omp_num_threads": case["threads"],
        "manifest": "manifest.json",
        "sequence": "sequence.json",
        "return_code": return_code,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "hashes": hashes,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    failure = None
    if return_code != 0:
        failure = (f"solver exited with code {return_code}"
                   + (" (timeout)" if timed_out else ""))
    elif not (case_dir / "sequence_convergence.csv").exists():
        failure = "solver produced no sequence_convergence.csv"
    else:
        try:
            result.update(summarize(case, case_dir))
            result["acceptance"] = acceptance(case, result)
            if not result["acceptance"]["ok"]:
                failed = [name for name, ok
                          in result["acceptance"]["checks"].items() if not ok]
                failure = "acceptance checks failed: " + ", ".join(failed)
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            failure = f"post-processing failed: {error!r}"

    result["status"] = "completed" if failure is None else "failed"
    if failure is not None:
        result["failure"] = failure
    write_json_atomic(case_dir / "result.json", result)
    (case_dir / "completed.json").unlink(missing_ok=True)
    if failure is None:
        write_json_atomic(case_dir / "completed.json", {
            "case": case["name"], "family": case["family"],
            "status": "completed", "hashes": hashes,
            "wall_seconds": result["wall_seconds"],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"OK {case['name']} "
              f"({result['wall_seconds']:.1f}s, "
              f"max accepted residual "
              f"{result['convergence']['max_accepted_residual']:.3e})",
              flush=True)
    else:
        print(f"FAIL {case['name']}: {failure}", flush=True)
    return {"name": case["name"], "family": case["family"],
            "status": result["status"], "failure": failure,
            "wall_seconds": result["wall_seconds"],
            "group": case["group"], "role": case["role"],
            "control_of": case["control_of"],
            "compare_with": case["compare_with"],
            "case_dir": relative(case_dir),
            "final_vtp": result.get("final_vtp"),
            "identity_sha256": hashes["identity_sha256"],
            "fingerprints": result.get("fingerprints")}


def group_report(records: list[dict]) -> dict:
    """Per-group input-identity and output-fingerprint comparison."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get("fingerprints") is None:
            continue
        groups.setdefault(record["group"], []).append(record)
    report = {}
    for group, members in sorted(groups.items()):
        identities = {member["identity_sha256"] for member in members}
        vtp = {member["fingerprints"].get("final_vtp_sha256")
               for member in members}
        csv_hash = {member["fingerprints"].get(
            "sequence_convergence_csv_sha256") for member in members}
        report[group] = {
            "members": [member["name"] for member in members],
            "identical_inputs": len(identities) == 1,
            "identical_final_vtp": len(vtp) == 1,
            "identical_convergence_csv": len(csv_hash) == 1,
            "final_vtp_sha256": {member["name"]:
                                 member["fingerprints"].get("final_vtp_sha256")
                                 for member in members},
        }
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def describe(cases: list[dict]) -> str:
    lines = []
    family = None
    for case in cases:
        if case["family"] != family:
            family = case["family"]
            lines.append(f"\n[{family}]")
        flags = " ".join(case["extra_flags"])
        lines.append(
            f"  {case['name']}\n"
            f"      purpose : {case['purpose']}\n"
            f"      role    : {case['role']}"
            + (f" of {case['control_of']}" if case['control_of'] else "")
            + f" (group {case['group']})\n"
            f"      order   : {' -> '.join(case['path_order'])}\n"
            f"      solver  : {case['solver']} res={case['res']} "
            f"metric={case['metric']} adaptive={case['adaptive']} "
            f"step={case['initial_step']} "
            f"minimize_every={case['minimize_every']} "
            f"threads={case['threads']}\n"
            f"      residual: accepted iff final_gradient_norm <= "
            f"{case['grad_tol']:g} on every minimised cycle "
            f"{expected_accepted_cycles(case)}\n"
            f"      endpoint: "
            + (f"abar identical to {case['endpoint']['reference_case']}: "
               f"{case['endpoint']['expected_identical_abar']} "
               f"({case['endpoint']['basis']})"
               if case['endpoint']['reference_case'] else "reference case")
            + f"\n      compare : {', '.join(case['compare_with']) or '-'}"
            + (f"\n      extra   : {flags}" if flags else ""))
    return "\n".join(lines)


def parse_prescribed_flags(items: list[str]) -> tuple[str, ...]:
    extra: list[str] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"--prescribed-flag expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit("--prescribed-flag key must not be empty")
        extra.extend([key if key.startswith("-") else f"-{key}", value])
    return tuple(extra)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", action="append", choices=(*FAMILIES, "all"),
                        help="restrict to a family (repeatable, default all)")
    parser.add_argument("--case", action="append",
                        help="restrict to a case name (repeatable)")
    parser.add_argument("--list", action="store_true",
                        help="print the case contracts and exit")
    parser.add_argument("--print-commands", action="store_true",
                        help="print the exact solver command per case and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and check the selection without writing "
                             "or running anything")
    parser.add_argument("--prepare-only", action="store_true",
                        help="write sequence.json and manifest.json for each "
                             "case without invoking the solver")
    parser.add_argument("--force", action="store_true",
                        help="discard existing case directories and rerun")
    parser.add_argument("--timeout", type=int, default=10800,
                        help="per-case wall-clock limit in seconds")
    parser.add_argument("--threads", type=int, default=8,
                        help="fixed thread count for the repeatability group")
    parser.add_argument("--replicas", type=int, default=5,
                        help="identical runs at the fixed thread count (>=3)")
    parser.add_argument("--single-thread-replicas", type=int, default=3,
                        help="identical runs at one thread (>=3)")
    parser.add_argument("--enable-prescribed", action="store_true",
                        help="run the prescribed-deformation cases; requires "
                             "the -prescribed_* flags in bin/shell")
    parser.add_argument("--prescribed-flag", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="append an extra solver flag to the prescribed "
                             "cases, e.g. prescribed_disp_csv=/path/field.csv")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop at the first failing case instead of "
                             "continuing with unrelated cases")
    args = parser.parse_args()

    if args.replicas < 3 or args.single_thread_replicas < 3:
        raise SystemExit("at least 3 replicas are required per thread setting")

    cases = all_cases(threads_fixed=args.threads,
                      replicas_fixed=args.replicas,
                      replicas_single=args.single_thread_replicas,
                      prescribed_extra=parse_prescribed_flags(
                          args.prescribed_flag))
    families = set(args.family or ["all"])
    if "all" in families:
        families = set(FAMILIES)
    wanted = set(args.case or [])
    unknown = wanted - {case["name"] for case in cases}
    if unknown:
        raise SystemExit(f"unknown case name(s): {sorted(unknown)}")
    selected = [case for case in cases
                if case["family"] in families
                and (not wanted or case["name"] in wanted)]
    if not selected:
        raise SystemExit("no matching cases")

    if args.list:
        print(f"{len(selected)} case(s) selected of {len(cases)} defined")
        print(describe(selected))
        return
    if args.print_commands:
        for case in selected:
            sequence_path = (RUN_ROOT / case["family"] / case["name"]
                             / "sequence.json")
            print(f"# {case['family']}/{case['name']} "
                  f"(OMP_NUM_THREADS={case['threads']})")
            print(" ".join(build_command(case, sequence_path)))
        return
    if args.dry_run:
        for case in selected:
            sequence_config(case["paths"])
            print(f"DRY {case['family']}/{case['name']} "
                  f"paths={len(case['paths'])} "
                  f"expected_cycles={expected_accepted_cycles(case)}")
        print(f"{len(selected)} case(s) would run serially; nothing written")
        return

    if args.prepare_only:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        prescribed_available = binary_supports_prescribed()
        for case in selected:
            run_case(case, timeout=args.timeout, force=True,
                     prescribed_enabled=True,
                     prescribed_available=prescribed_available,
                     prepare_only=True)
        print(f"{len(selected)} case input set(s) written under "
              f"{relative(RUN_ROOT)}; solver not invoked")
        return

    if not SHELL.exists():
        raise SystemExit(f"solver binary not found: {SHELL}")
    prescribed_available = binary_supports_prescribed()
    if args.enable_prescribed and not prescribed_available:
        print("WARNING: --enable-prescribed given but bin/shell lacks "
              f"{PRESCRIBED_PROBE_FLAG.decode()}; those cases will be "
              "recorded as blocked", flush=True)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    started = datetime.now(timezone.utc)
    with LOCK_PATH.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(
                f"another solver-characteristics run holds {LOCK_PATH}; "
                "cases must execute strictly serially") from None
        lock.write(f"pid={os.getpid()} host={platform.node()} "
                   f"started={started.isoformat()}\n")
        lock.flush()
        for case in selected:
            try:
                record = run_case(
                    case, timeout=args.timeout, force=args.force,
                    prescribed_enabled=args.enable_prescribed,
                    prescribed_available=prescribed_available)
            except Exception as error:  # noqa: BLE001 - isolate one case
                record = {"name": case["name"], "family": case["family"],
                          "group": case["group"], "status": "failed",
                          "failure": f"runner error: {error!r}"}
                write_json_atomic(
                    RUN_ROOT / case["family"] / case["name"] / "result.json",
                    {"case": case["name"], "family": case["family"],
                     "status": "failed", "failure": record["failure"],
                     "finished_utc": datetime.now(timezone.utc).isoformat()})
                print(f"FAIL {case['name']}: {record['failure']}", flush=True)
            records.append(record)
            if args.fail_fast and record["status"] == "failed":
                print("stopping early (--fail-fast)", flush=True)
                break

    failures = [record for record in records if record["status"] == "failed"]
    blocked = [record for record in records if record["status"] == "blocked"]
    study = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "binary_sha256": sha256_file(SHELL),
        "git_revision": git_value("rev-parse", "HEAD"),
        "selection": {"families": sorted(families),
                      "cases": [case["name"] for case in selected]},
        "cases": records,
        "groups": group_report(records),
        "counts": {
            "total": len(records),
            "completed": sum(1 for r in records if r["status"] == "completed"),
            "cached": sum(1 for r in records if r["status"] == "cached"),
            "failed": len(failures),
            "blocked": len(blocked),
        },
    }
    write_json_atomic(RUN_ROOT / "study_summary.json", study)
    print(json.dumps(study["counts"]), flush=True)
    for group, report in study["groups"].items():
        print(f"GROUP {group}: identical_inputs="
              f"{report['identical_inputs']} "
              f"identical_final_vtp={report['identical_final_vtp']} "
              f"identical_convergence_csv="
              f"{report['identical_convergence_csv']}", flush=True)
    if failures:
        print("failed cases: "
              + ", ".join(f"{r['name']} ({r['failure']})" for r in failures),
              flush=True)
    if failures or blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
