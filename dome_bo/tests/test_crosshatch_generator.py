#!/usr/bin/env python3
"""Tests of the cross-hatch generator against the EMITTED treatment field and the real simulator parser."""
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dome_bo'))

from crosshatch_generator import (PitchGradedCrossHatch, conventional_zigzag_ripple, decode_hit_field,
                                  decode_treatment_field, invert_voce_dose, operation_footprint, operation_hits,
                                  raw_from_recipe, treatment_monotonicity_violation, voce_effective_dose,
                                  BETA_VOCE, SHEET_HALF_U_MM, SHEET_HALF_V_MM, A_E_MM, TILE_GAP_MM)

SHELL = ROOT / 'bin/shell'


def grid(step=1.0):
    u = np.arange(-SHEET_HALF_U_MM, SHEET_HALF_U_MM + step / 2, step)
    v = np.arange(-SHEET_HALF_V_MM, SHEET_HALF_V_MM + step / 2, step)
    return np.meshgrid(u, v)


def simulator_operation_keys():
    """Allowed operation keys, read from the C++ parser so the test cannot drift from it."""
    text = (ROOT / 'src/libshell/ZigZagSequenceGrowth.hpp').read_text()
    block = text[text.index('operation,\n'):]
    block = block[:block.index('context + ".operation")')]
    return set(re.findall(r'"([a-z_]+)"', block))


class TestVoce(unittest.TestCase):
    def test_inversion_roundtrip_and_unreachable_dose(self):
        for n in range(1, 12):
            self.assertEqual(invert_voce_dose(voce_effective_dose(n)), n)
        with self.assertRaises(ValueError):
            invert_voce_dose(10.0)


class TestEmittedRecipe(unittest.TestCase):
    """Parser compatibility and the separated-stroke (tile) layout, which is explicitly selected here."""
    @classmethod
    def setUpClass(cls):
        cls.gen = PitchGradedCrossHatch(layout='stroke')
        cls.recipe = cls.gen.generate_recipe()
        cls.U, cls.V = grid()
        cls.hits = {o: decode_hit_field(cls.recipe, cls.U, cls.V, o) for o in 'AB'}

    def test_schema_matches_the_simulator_parser(self):
        self.assertEqual(set(self.recipe),
                         {'schema_version', 'units', 'hardening', 'defaults', 'boundary_conditions', 'toolpaths'})
        allowed = simulator_operation_keys()
        self.assertTrue({'lv_mm', 'width_mm', 'center_uv_mm', 'rotation_deg'} <= allowed)
        for tp in self.recipe['toolpaths']:
            self.assertLessEqual(set(tp['operation']), allowed)
            self.assertGreaterEqual(tp['operation']['n_strips'], 2)
            self.assertGreaterEqual(tp['repeat'], 1)

    def test_every_footprint_is_inside_the_sheet(self):
        for tp in self.recipe['toolpaths']:
            u0, u1, v0, v1 = operation_footprint(tp['operation'])
            self.assertGreaterEqual(u0, -SHEET_HALF_U_MM)
            self.assertLessEqual(u1, SHEET_HALF_U_MM)
            self.assertGreaterEqual(v0, -SHEET_HALF_V_MM)
            self.assertLessEqual(v1, SHEET_HALF_V_MM)

    def test_centre_hits_follow_the_voce_dose(self):
        """Every sweep contributes two simulator hits (two coincident strips); the centre reaches the
        requested effective Voce dose."""
        c = np.unravel_index(np.argmin(np.hypot(self.U, self.V)), self.U.shape)
        for o, sweeps in (('A', self.gen.sweeps_a), ('B', self.gen.sweeps_b)):
            self.assertEqual(self.hits[o][c], 2 * sweeps)
            self.assertEqual(self.hits[o].max(), 2 * sweeps)
        total = self.hits['A'][c] + self.hits['B'][c]
        self.assertEqual(total, self.gen.hits_centre)
        self.assertGreaterEqual(voce_effective_dose(int(total)), self.gen.total_dose - 1e-9)
        f = decode_treatment_field(self.recipe, np.array([0.0]), np.array([0.0]))
        self.assertEqual(f['hits'][0], total)

    # ---- default layout: spatially separated, pitch-graded hatch strokes ----------------------
    def test_strokes_are_spatially_separated_tiles_whose_pitch_grows_outward(self):
        """Distinct strokes at distinct positions; each tiles its pitch (no overlap, no gap beyond the
        1 um tile gap); the pitch (stroke width = spacing of adjacent strokes) GROWS toward the rim."""
        for o in 'AB':
            st = self.gen.stroke_table()[o]
            self.assertGreaterEqual(len(st), 4)
            positions = [s_['pos'] for s_ in st]
            self.assertEqual(len(set(positions)), len(positions))
            widths = [s_['width'] for s_ in st]
            self.assertTrue(all(b >= a - 1e-9 for a, b in zip(widths, widths[1:])), widths)
            self.assertGreater(widths[-1], 1.3 * widths[0])                       # real grading
            for inner, outer in zip(st, st[1:]):                                   # exact tiling across the pitch axis
                self.assertAlmostEqual(inner['pos'] + 0.5 * inner['width'], outer['x_in'], places=9)
                self.assertAlmostEqual(outer['x_in'] + 0.5 * outer['width'], outer['pos'], places=9)
            self.assertAlmostEqual(st[-1]['x_in'] + st[-1]['width'], A_E_MM if o == 'A' else min(142.0, 1.25 * A_E_MM))
        shifts = {(tp['operation']['rotation_deg'], *tp['operation']['shift_uv_mm']) for tp in self.recipe['toolpaths']}
        self.assertEqual(len(shifts), len(self.recipe['toolpaths']))                # no two strokes coincide

    def test_stroke_repeats_and_chords_decay_monotonically_outward(self):
        for o in 'AB':
            st = self.gen.stroke_table()[o]
            reps = [s_['repeat'] for s_ in st]
            self.assertTrue(all(b <= a for a, b in zip(reps, reps[1:])), reps)
            self.assertLess(reps[-1], reps[0])
            half = [s_['half_extent'] for s_ in st]
            self.assertTrue(all(b < a for a, b in zip(half, half[1:])), half)      # outer strokes nested in the inner chord
            for inner, outer in zip(st, st[1:]):
                self.assertLessEqual(outer['half_extent'], inner['half_extent'] - 0.5 * inner['width'] + 1e-9)

    def test_stroke_hits_take_only_the_stroke_levels_and_never_double_count(self):
        for o in 'AB':
            reps = {2.0 * s_['repeat'] for s_ in self.gen.stroke_table()[o]}
            self.assertTrue(set(np.unique(self.hits[o])) <= ({0.0} | reps), (o, np.unique(self.hits[o]), reps))

    def _tile(self, o, c):
        """Stroke tile (stroke_table entry) containing the across-stroke coordinate magnitude c."""
        for st in self.gen.stroke_table()[o]:
            if st['k'] == 0 and c < 0.5 * st['width']:
                return st
            if st['k'] and st['x_in'] <= c < st['x_in'] + st['width']:
                return st
        return self.gen.stroke_table()[o][-1]

    def test_hit_levels_decay_outward_and_increases_only_occur_in_end_cap_notches(self):
        """What 'monotonic edge decay' means for separated strokes, measured on the simulator-faithful hit
        field along dense rays: the stroke LEVEL (2 x repeat of the tile) never increases outward, hits are
        always the level or zero, and the decoded hits increase outward ONLY where a ray re-enters a
        capsule through the round end cap of a shorter, outer stroke - i.e. at along-stroke coordinates
        beyond the straight part of its tile (the notch between neighbouring end caps at the rim of that
        stroke). The nested-terrace layout has no such notches (see TestTerraceLayout)."""
        s = np.linspace(0.0, 1.0, 1201)
        n_checked = 0
        for deg in np.arange(0.0, 360.0, 5.0):
            t = np.radians(deg)
            r = 0.0123456 + s * 160.0
            U, V = r * np.cos(t), r * np.sin(t)
            keep = (np.abs(U) <= SHEET_HALF_U_MM) & (np.abs(V) <= SHEET_HALF_V_MM)
            U, V = U[keep], V[keep]
            for o in ('A', 'B'):
                h = decode_hit_field(self.recipe, U, V, o)
                across = np.abs(U) if o == 'A' else np.abs(V)
                along = np.abs(V) if o == 'A' else np.abs(U)
                levels = np.array([2.0 * self._tile(o, c)['repeat'] for c in across])
                self.assertTrue(np.all(np.diff(levels) <= 0), (deg, o))
                self.assertTrue(np.all((h == 0) | (h == levels)), (deg, o))
                edges = [st['x_in'] for st in self.gen.stroke_table()[o] if st['k']]
                for i in np.where(np.diff(h) > 0)[0]:
                    if min(abs(across[i] - e) for e in edges) <= TILE_GAP_MM:       # sample inside a 1 um tile gap
                        continue
                    tile = self._tile(o, across[i + 1])
                    self.assertGreater(along[i + 1], 0.5 * tile['lv'] - 1e-9, (deg, o, U[i + 1], V[i + 1]))
                    n_checked += 1
        self.assertGreater(n_checked, 0)

    def test_pitch_grading_and_shape_parameters_change_the_emitted_toolpaths(self):
        base = json.dumps(self.recipe, sort_keys=True)
        for kw in ({'edge_pitch_factor': 3.0}, {'aspect_ratio': 1.0}, {'decay_power': 1.5},
                   {'center_pitch_a': 15.0}, {'total_dose': 2.0}, {'dose_balance': 0.3}):
            other = json.dumps(PitchGradedCrossHatch(**kw).generate_recipe(), sort_keys=True)
            self.assertNotEqual(base, other, kw)

    def test_invalid_parameters_are_rejected(self):
        for kw in ({'decay_power': 0.5}, {'aspect_ratio': 0.8}, {'edge_pitch_factor': 0.5}, {'center_pitch_a': 0.0},
                   {'dose_balance': 0.0}, {'dose_balance': 1.0}, {'profile_mode': 'bogus'}, {'total_dose': 5.0}):
            with self.assertRaises(ValueError):
                PitchGradedCrossHatch(**kw)

    def test_prescreen_is_computed_from_the_emitted_recipe(self):
        a = self.gen.compute_prescreen(grid_size=61)
        self.assertEqual(set(a), {'synclastic_fraction', 'saddle_fraction',
                                  'prescreen_score', 'target_K_pos_flux', 'target_K_neg_flux'})
        for kw in ({'total_dose': 1.0}, {'edge_pitch_factor': 3.0}, {'center_pitch_b': 12.0}):
            b = PitchGradedCrossHatch(**kw).compute_prescreen(grid_size=61)
            self.assertNotEqual(a['prescreen_score'], b['prescreen_score'], kw)


class TestDefaultLayout(unittest.TestCase):
    """The default emission is the nested terrace (layer-cake of the pitch-graded stroke staircase)."""
    def test_default_is_terrace_and_exactly_monotone(self):
        gen = PitchGradedCrossHatch()
        self.assertEqual(gen.layout, 'terrace')
        v = treatment_monotonicity_violation(gen.generate_recipe())
        self.assertEqual(v['hits'], 0.0)
        self.assertEqual(v['top_trace'], 0.0)
        self.assertEqual(v['bottom_trace'], 0.0)

    def test_terrace_hit_field_is_a_pitch_graded_staircase(self):
        """Layer-cake: nested centred strokes with repeats r_j give a staircase whose step positions are the
        terrace edges. Along each axis the decoded hits equal sum_j 2 r_j 1[|x| <= edge_j], take >= 3 distinct
        levels, never increase, and the step widths (the hatch pitch) do not shrink toward the rim."""
        gen = PitchGradedCrossHatch(layout='terrace')
        recipe = gen.generate_recipe()
        for o in 'AB':
            layers = gen.layer_table()[o]                       # outermost first
            edges = np.array([lay['cross'] for lay in layers[::-1]])
            steps = np.diff(edges, prepend=0.0)
            self.assertTrue(np.all(np.diff(steps) >= -1e-9), steps)       # pitch grows (or stays) outward
            self.assertGreater(steps[-1], steps[0])
            x = np.linspace(0.0, edges[-1] - 1e-6, 400)
            zeros = np.zeros_like(x)
            u, v = (x, zeros) if o == 'A' else (zeros, x)
            h = decode_hit_field(recipe, u, v, o)
            expected = sum(2 * lay['repeat'] * (x <= lay['cross']) for lay in layers)
            self.assertTrue(np.array_equal(h, expected))
            self.assertTrue(np.all(np.diff(h) <= 0))
            self.assertGreaterEqual(len(np.unique(h)), 3)

    def test_separated_strokes_have_a_notch_violation_so_they_are_not_the_default(self):
        v = treatment_monotonicity_violation(PitchGradedCrossHatch(layout='stroke').generate_recipe())
        self.assertGreater(v['hits'], 0.0)


class TestTerraceLayout(unittest.TestCase):
    """The nested-terrace layout (the default): exactly monotone by construction (centred convex capsules)."""
    @classmethod
    def setUpClass(cls):
        cls.gen = PitchGradedCrossHatch(layout='terrace')
        cls.recipe = cls.gen.generate_recipe()
        cls.U, cls.V = grid()
        cls.hits = {o: decode_hit_field(cls.recipe, cls.U, cls.V, o) for o in 'AB'}

    def test_layers_are_nested_centred_convex_capsules(self):
        for o, layers in self.gen.layer_table().items():
            for outer, inner in zip(layers, layers[1:]):
                self.assertGreater(outer['cross'], inner['cross'])
                self.assertGreaterEqual(outer['along'], inner['along'])
            for lay in layers:
                self.assertGreaterEqual(lay['along'], lay['cross'])          # lv >= 0: a true capsule
                self.assertGreaterEqual(lay['repeat'], 1)
        for tp in self.recipe['toolpaths']:
            self.assertEqual(tp['operation']['center_uv_mm'], [0.0, 0.0])
            self.assertEqual(tp['operation']['shift_uv_mm'], [0.0, 0.0])
            self.assertEqual(tp['operation']['n_strips'], 2)

    def test_terrace_hits_are_non_increasing_along_every_ray_from_the_centre(self):
        """Simulator-faithful hit field on dense rays (not only a grid): exactly non-increasing, so no
        gap followed by renewed deposition, overlap band or pitch reversal exists in the emitted field."""
        s = np.linspace(0.0, 1.0, 1201)
        for deg in np.arange(0.0, 360.0, 5.0):
            t = np.radians(deg)
            r = s * 160.0
            U, V = r * np.cos(t), r * np.sin(t)
            keep = (np.abs(U) <= SHEET_HALF_U_MM) & (np.abs(V) <= SHEET_HALF_V_MM)
            for o in ('A', 'B', None):
                h = decode_hit_field(self.recipe, U[keep], V[keep], o)
                self.assertTrue(np.all(np.diff(h) <= 0), (deg, o))
        for o in 'AB':                       # no untreated hole inside the outermost footprint
            lay = self.gen.layer_table()[o][0]
            along_u = lay['cross'] if o == 'A' else lay['along']
            along_v = lay['along'] if o == 'A' else lay['cross']
            inside = (np.abs(self.U) <= 0.7 * along_u) & (np.abs(self.V) <= 0.7 * along_v)
            self.assertEqual(int(((self.hits[o] == 0) & inside).sum()), 0)

    def test_hits_are_constant_inside_each_terrace(self):
        """Faithful counting gives a clean staircase: hits take only multiples of 2 x repeat sums."""
        for o in 'AB':
            reps = [l['repeat'] for l in self.gen.layer_table()[o]]
            allowed = {0.0} | {2.0 * sum(reps[k:]) for k in range(len(reps))}
            self.assertTrue(set(np.unique(self.hits[o])) <= allowed, (o, np.unique(self.hits[o]), allowed))

    def test_terrace_spacing_grows_outward_by_the_requested_factor(self):
        for f in (1.0, 2.0, 3.0):
            gen = PitchGradedCrossHatch(total_dose=4.5, center_pitch_a=8.0, center_pitch_b=8.0,
                                        edge_pitch_factor=f, n_layers=5)
            layers = gen.layer_table()['A']
            spacing = [l['spacing'] for l in layers]                  # outermost first
            self.assertTrue(all(a >= b - 1e-12 for a, b in zip(spacing, spacing[1:])), spacing)
            self.assertAlmostEqual(spacing[0] / spacing[-1], f, places=9)
            radii = [l['cross'] for l in layers][::-1]
            self.assertTrue(np.allclose(np.diff(radii), spacing[::-1][1:]))   # radial gaps are the spacings


class TestTreatmentAndCoordinates(unittest.TestCase):
    """Profile- and hardening-weighted treatment, and recipe-consistent surrogate coordinates."""

    def test_uniform_profile_treatment_is_exactly_non_increasing_outward(self):
        """The cumulative Voce-hardened growth (not just hit counts) never increases outward, for the
        default and for graded / strongly unbalanced variants, with the recipe ordered A-layers then B."""
        for kw in ({}, {'edge_pitch_factor': 3.0, 'decay_power': 3.0}, {'dose_balance': 0.2},
                   {'dose_balance': 0.8, 'aspect_ratio': 1.4}, {'total_dose': 4.5, 'n_layers': 4}):
            v = treatment_monotonicity_violation(PitchGradedCrossHatch(layout='terrace', **kw).generate_recipe())
            for key, val in v.items():
                self.assertLessEqual(val, 1e-12, (kw, key, v))

    def test_treatment_equals_voce_sum_of_the_total_hit_count(self):
        """With a uniform profile and equal coefficients every hit has the same weight, so the cumulative
        top growth is g * sum_{j<H} exp(-beta j) with H the TOTAL (A and B) hit count - not per orientation."""
        from crosshatch_generator import G_TOP_DEFAULT, ORTHO_TOP_DEFAULT
        recipe = PitchGradedCrossHatch().generate_recipe()
        U, V = grid(step=5.0)
        f = decode_treatment_field(recipe, U, V)
        expected = 2.0 * G_TOP_DEFAULT * np.vectorize(lambda h: voce_effective_dose(int(h)))(f['hits'])
        self.assertTrue(np.allclose(f['top_trace'], expected, rtol=1e-10, atol=1e-15))

    def test_centre_dose_is_shared_not_per_orientation(self):
        """B's contribution at the centre is smaller than a fresh B because A already hardened the face."""
        gen = PitchGradedCrossHatch(total_dose=4.5)
        both = decode_treatment_field(gen.generate_recipe(), np.array([0.0]), np.array([0.0]))
        b_only = gen.generate_recipe()
        b_only['toolpaths'] = [t for t in b_only['toolpaths'] if '_B_' in t['id']]
        alone = decode_treatment_field(b_only, np.array([0.0]), np.array([0.0]))
        self.assertLess(both['along_B'][0] + both['across_B'][0], alone['along_B'][0] + alone['across_B'][0] - 1e-9)

    def test_center_peak_profile_can_break_monotonicity_and_is_measured(self):
        """The calibrated sin-profile weights hits unequally, so hardening can make treatment rise outward;
        the check detects it (this is why 'uniform' is the default and the monotone-by-construction mode)."""
        v = treatment_monotonicity_violation(PitchGradedCrossHatch(layout='terrace', profile_mode='center_peak').generate_recipe())
        self.assertGreater(v['top_trace'], 0.0)
        self.assertLessEqual(v['hits'], 1e-12)             # hit counts remain monotone regardless

    def test_conventional_overlapping_zigzag_has_ripple(self):
        """Why terraces and not a literal graded-pitch zigzag: with w > p overlapping strips each count as
        a hit, giving non-monotone hit counts in this simulator."""
        self.assertGreater(conventional_zigzag_ripple(), 0.05)

    def test_realized_coordinates_are_measured_on_the_emitted_recipe(self):
        gen = PitchGradedCrossHatch(total_dose=3.5, dose_balance=0.5, center_pitch_a=22.0, center_pitch_b=14.0,
                                    aspect_ratio=1.25, decay_power=2.5, layout='stroke')
        recipe = gen.generate_recipe()
        raw = gen.realized_raw(recipe)
        self.assertTrue(np.allclose(raw, raw_from_recipe(recipe, gen.decay_power)))
        strokes = gen.stroke_table()
        self.assertAlmostEqual(raw[4], strokes['A'][0]['width'], places=2)       # centre-stroke pitch of A
        self.assertAlmostEqual(raw[5], strokes['B'][0]['width'], places=2)
        self.assertEqual(raw[1], gen.hits_a / gen.hits_centre)
        self.assertAlmostEqual(raw[0], voce_effective_dose(gen.hits_centre))
        # request differs from realisation (integer sweeps / sheet fitting): the surrogate must use `raw`
        self.assertFalse(np.allclose(raw, gen.requested_raw()))
        a = PitchGradedCrossHatch(center_pitch_a=10.0).realized_raw()
        b = PitchGradedCrossHatch(center_pitch_a=20.0).realized_raw()
        self.assertNotEqual(a[4], b[4])


@unittest.skipUnless(SHELL.exists(), 'bin/shell not built')
class TestSimulatorAcceptsRecipe(unittest.TestCase):
    def test_coarse_mesh_run_applies_every_hit(self):
        """Real parser + accumulation on a coarse mesh (parse/accounting check, not an accuracy claim)."""
        recipe = PitchGradedCrossHatch().generate_recipe()
        expected = sum(tp['repeat'] for tp in recipe['toolpaths'])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'sequence.json').write_text(json.dumps(recipe))
            opts = {'-sim': 'bilayer_growth', '-case': 'custom', '-geometry': 'rectangle', '-lx': '.1',
                    '-ly': '.15', '-res': '.04', '-h_total': '.0007700000000000001',
                    '-growth_type': 'zigzag_sequence', '-sequence_solve_mode': 'final_only',
                    '-metric_update': 'multiplicative', '-enable_passE': 'false', '-nsteps': '1',
                    '-equilibrium_solver': 'native_hybrid', '-native_hybrid_grad_tol': '5e-14',
                    '-equilibrium_grad_tol': '5e-14', '-native_hybrid_warmup_attempts': '20',
                    '-native_hybrid_seconds': '60', '-hessian_threads': '1', '-sequence_adaptive': 'true',
                    '-sequence_initial_step': '1.0', '-sequence_max_step': '1.0',
                    '-sequence_min_step': '.015625', '-sequence_max_retries': '8',
                    '-native_hybrid_export_state': 'true', '-cycle_write_state': 'false',
                    '-export_stl': 'false', '-cycle_file': str(tmp / 'sequence.json')}
            cmd = [str(SHELL)] + [x for kv in opts.items() for x in kv]
            env = dict(os.environ, OMP_NUM_THREADS='1')
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stdout[-800:] + proc.stderr[-800:])
            with (tmp / 'sequence_history.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), expected)
            state = json.loads((tmp / 'bilayer_zigzag_sequence_native_hybrid_state.json').read_text())
            uv = np.asarray(state['material_uv'], float) * 1000.0
            faces = np.asarray(state['faces'], int)
            cen = uv[faces].mean(axis=1)
            # Per-cycle simulator hit events must equal the decoded per-strip hits at the mesh centroids.
            cycle = 0
            running = np.zeros(len(cen))
            for tp in recipe['toolpaths']:
                for _ in range(tp['repeat']):
                    per_cycle = operation_hits(tp['operation'], cen[:, 0], cen[:, 1])
                    running += per_cycle
                    self.assertEqual(int(rows[cycle]['hit_events']), int(per_cycle.sum()), (tp['id'], cycle))
                    self.assertEqual(int(rows[cycle]['max_total_hits']), int(running.max()), (tp['id'], cycle))
                    cycle += 1
            self.assertGreater(running.max(), 0)


if __name__ == '__main__':
    unittest.main()
