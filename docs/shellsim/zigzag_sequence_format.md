# JSON-Controlled Recurring Zigzag Toolpaths

## Overview

The `zigzag_sequence` growth type runs an ordered series of zigzag toolpaths defined in a JSON file. It is intended for recurring forming or reworking simulations in which:

1. A complete toolpath is applied.
2. The shell is released and allowed to reach elastic equilibrium.
3. The resulting geometry becomes the starting geometry for the next physical cycle.
4. The accumulated top and bottom target metrics and face-level treatment history are preserved.
5. The reference curvature \(b_r\) remains equal to its original value.

Each JSON item defines a **toolpath recipe**. Its `repeat` value determines how many physical cycles use that recipe.

For example:

```json
{
  "id": "toolpath_A",
  "repeat": 3,
  "operation": {
    "type": "zigzag"
  }
}
```

executes:

```text
toolpath_A, repeat 1 -> release/springback
toolpath_A, repeat 2 -> release/springback
toolpath_A, repeat 3 -> release/springback
```

Each repeat is mapped to the latest released geometry and receives its own minimization, VTP outputs, and row in the sequence summary CSV.

---

## Required implementation files

The sequence implementation uses:

```text
Sim_Bilayer_Growth.cpp
ZigZagGrowth.hpp
ZigZagSequenceGrowth.hpp
GrowthHelper.hpp
```

The build must also provide the existing `nlohmann::json` dependency used elsewhere in the repository.

---

## Quick start

### Flat starting panel

```bash
time ./shell \
  -sim bilayer_growth \
  -case custom \
  -geometry rectangle \
  -lx 0.13 \
  -ly 0.16 \
  -res 0.01 \
  -h_total 0.0005 \
  -growth_type zigzag_sequence \
  -cycle_file zigzag_sequence.json \
  -enable_passE false \
  -nsteps 1 \
  -tol 3e-12 \
  -export_stl true
```

### Curved starting panel

```bash
time ./shell \
  -sim bilayer_growth \
  -case custom \
  -geometry curved_rectangle \
  -lx 0.13 \
  -ly 0.16 \
  -res 0.01 \
  -curve_radius 0.50 \
  -curve_sign 1 \
  -h_total 0.0005 \
  -growth_type zigzag_sequence \
  -cycle_file zigzag_sequence.json \
  -enable_passE false \
  -nsteps 1 \
  -tol 3e-12 \
  -export_stl true
```

For `zigzag_sequence`, the zigzag dimensions, position, rotation, growth values, profile, repeats, and hardening settings are read from the JSON file rather than from individual zigzag command-line arguments.

---

## Required command-line settings

| Parameter | Meaning |
|---|---|
| `-sim bilayer_growth` | Select the bilayer shell simulation. |
| `-case custom` | Select custom growth-pattern handling. |
| `-growth_type zigzag_sequence` | Activate the JSON-controlled recurring zigzag sequence. |
| `-cycle_file <file.json>` | Path to the JSON sequence file. |
| `-enable_passE false` | Required. The sequence uses eigenstrain-decay hardening rather than modulus-based `passE`. |
| `-nsteps 1` | Required. The sequence owns the physical loading/release loop. |

### Common geometry and solver parameters

| Parameter | Meaning |
|---|---|
| `-geometry rectangle` | Flat structured panel. |
| `-geometry curved_rectangle` | Analytical cylindrical starting panel. |
| `-lx`, `-ly` | Panel half-lengths in meters. A 260 mm × 320 mm panel uses `-lx 0.13 -ly 0.16`. |
| `-res` | Mesh-resolution parameter used by the structured geometry generator. |
| `-h_total` | Total shell thickness in meters. |
| `-curve_radius` | Initial cylindrical midsurface radius in meters; only for `curved_rectangle`. |
| `-curve_sign` | Curvature direction, typically `1` or `-1`; only for `curved_rectangle`. |
| `-margin_x`, `-margin_y` | No-growth material-coordinate margins in meters. |
| `-tol` | Minimization tolerance. |
| `-max_iter` | Optional maximum number of minimizer iterations. |
| `-dump_iters` | Optional comma-separated iteration numbers for intermediate dumps. |
| `-stepwise` | Optional stepwise minimization flag. |
| `-cycle_write_state` | If `true`, write an internal mesh state after every physical cycle. |
| `-export_stl` | If `true`, write the final deformed geometry as STL. |
| `-stl_ascii` | If `true`, write ASCII STL instead of the default format. |

---

# JSON file structure

A sequence file contains five main sections:

```json
{
  "schema_version": 1,
  "units": {},
  "hardening": {},
  "defaults": {},
  "toolpaths": []
}
```

A complete example is provided later in this document.

---

## 1. `schema_version`

```json
"schema_version": 1
```

The current implementation supports only schema version `1`.

---

## 2. `units`

```json
"units": {
  "length": "mm",
  "angle": "deg",
  "growth": "engineering_strain"
}
```

The currently supported values are fixed:

| Field | Supported value |
|---|---|
| `length` | `"mm"` |
| `angle` | `"deg"` |
| `growth` | `"engineering_strain"` |

Growth values are dimensionless. For example:

```text
0.003 = 0.3% engineering growth
```

---

## 3. Global hardening settings

```json
"hardening": {
  "model": "voce_decay",
  "history_variable": "hit_count",
  "beta": 0.223143551,
  "floor": 0.0,
  "scope": "face_shared",
  "count_rule": "plastic_hits"
}
```

### Supported hardening models

```json
"model": "voce_decay"
```

uses a face-based exponential decay:

\[
q(N_f)
=
q_{\infty}
+
\left(1-q_{\infty}\right)
\exp\left(-\beta N_f\right)
\]

and

\[
\Delta g_{\mathrm{effective}}
=
q(N_f)\Delta g_{\mathrm{base}}.
\]

Here:

- \(N_f\) is the number of previous qualifying hits on face \(f\);
- `beta` is \(\beta\), the decay rate;
- `floor` is \(q_{\infty}\), the long-term retained fraction.

The history count is evaluated before each hit and updated after each qualifying hit.

```json
"model": "none"
```

disables the decay and uses:

\[
q(N_f)=1.
\]

### Hardening parameter reference

| Key | Allowed value | Meaning |
|---|---|---|
| `model` | `"voce_decay"` or `"none"` | Select the hardening rule. |
| `history_variable` | `"hit_count"` | Current face-level history variable. |
| `beta` | Number \(\ge 0\) | Exponential decay rate. `0` gives no decay. |
| `floor` | Number in \([0,1]\) | Minimum retained treatment effectiveness. |
| `scope` | `"face_shared"` | One face history is shared by top and bottom layers. |
| `count_rule` | `"plastic_hits"` or `"all_tool_hits"` | Determines which events increase the face history count. |

### Count rules

```json
"count_rule": "plastic_hits"
```

counts a hit only when its prescribed top or bottom growth is nonzero.

```json
"count_rule": "all_tool_hits"
```

counts every geometric strip/face intersection, including zero-growth trial passes.

### Example: 80% retained effectiveness per previous hit

```json
"beta": 0.223143551,
"floor": 0.0
```

because:

\[
\exp(-0.223143551)\approx 0.8.
\]

For a face hit once per physical cycle and base growth \(g_0=0.003\):

| Hit | Previous hits | Factor | Effective growth |
|---:|---:|---:|---:|
| 1 | 0 | 1.000 | 0.003000 |
| 2 | 1 | 0.800 | 0.002400 |
| 3 | 2 | 0.640 | 0.001920 |
| 4 | 3 | 0.512 | 0.001536 |

Overlapping strips are separate hit events. A face covered twice during one toolpath is hardened between the first and second hit.

---

## 4. Global defaults

```json
"defaults": {
  "start_mode": "left_bottom_up",
  "profile": {
    "mode": "center_peak",
    "top_end_ratio": 0.4,
    "top_profile_power": 2.0
  }
}
```

### `start_mode`

The currently supported value is:

```json
"start_mode": "left_bottom_up"
```

The canonical zigzag begins at the left-bottom point, travels upward along the left vertical segment, traverses the inclined segments, and ends with the right vertical segment.

### Default profile

The profile in `defaults` is inherited by every toolpath unless that toolpath provides its own `profile` block.

---

# Toolpath definitions

The `toolpaths` array defines the loading order:

```json
"toolpaths": [
  {
    "id": "toolpath_A",
    "enabled": true,
    "repeat": 3,
    "operation": {}
  },
  {
    "id": "toolpath_B",
    "enabled": true,
    "repeat": 1,
    "operation": {}
  }
]
```

Toolpaths execute in array order.

## Toolpath-level parameters

| Key | Required? | Default | Meaning |
|---|---:|---:|---|
| `id` | Recommended | Generated name | Unique toolpath identifier used in filenames and the CSV summary. |
| `enabled` | No | `true` | Execute or skip this toolpath definition. |
| `repeat` | No | `1` | Number of separate physical cycles using this recipe. |
| `operation` | Yes | — | Zigzag operation definition. |

### `enabled`

```json
"enabled": true
```

executes the toolpath.

```json
"enabled": false
```

skips it completely. It produces no hit updates, minimization, VTP files, or CSV rows.

### `repeat`

```json
"repeat": 3
```

creates three physical cycles. Each repetition:

1. Starts from the previous released geometry.
2. Remaps the same material-space toolpath to that geometry.
3. Uses the updated face hit history.
4. Updates the target metrics.
5. Runs a new minimization.
6. Writes separate outputs.

`repeat` must be an integer greater than or equal to `1`.

---

# Zigzag operation parameters

```json
"operation": {
  "type": "zigzag",
  "lv_mm": 140.0,
  "alpha_deg": 9.13,
  "n_strips": 10,
  "width_mm": 10.0,
  "center_uv_mm": [0.0, 0.0],
  "shift_uv_mm": [0.0, 30.0],
  "shift_frame": "material",
  "rotation_deg": 0.0,
  "gtop": 0.003,
  "gbot": 0.0,
  "ortho": 0.0,
  "profile": {
    "mode": "uniform"
  }
}
```

## Geometry parameters

| Key | Required? | Default | Meaning |
|---|---:|---:|---|
| `type` | No | `"zigzag"` | Only `"zigzag"` is currently supported. |
| `lv_mm` | No | `40.0` | Vertical span of the canonical zigzag in millimeters. |
| `alpha_deg` | No | `15.0` | Inclined-segment angle relative to the vertical direction. |
| `n_strips` | No | `6` | Total segment/strip count, including the two vertical end strips. Must be at least `2`. |
| `width_mm` | No | `2.0` | Full treated-band width around each segment centerline. Must be positive. |
| `rotation_deg` | No | `0.0` | Counterclockwise rigid rotation of the complete pattern in the material \(u\)-\(v\) plane. |

For `n_strips = N`, the path contains:

- one left vertical strip;
- \(N-2\) inclined strips;
- one right vertical strip.

---

## Material-coordinate placement

The zigzag is defined in fixed material coordinates \((u,v)\). These coordinates remain attached to the material even when the current panel becomes curved.

For a centered rectangular panel:

- \(u=0,v=0\) is the panel center;
- positive \(u\) follows the original panel \(+x\) direction;
- positive \(v\) follows the original panel \(+y\) direction.

### `center_uv_mm`

```json
"center_uv_mm": [20.0, -10.0]
```

defines the absolute nominal pattern center and rotation pivot:

\[
(u_c,v_c)=(20,-10)\ \mathrm{mm}.
\]

If `center_uv_mm` is omitted, the code uses the material-coordinate bounding-box center.

### `shift_uv_mm`

```json
"shift_uv_mm": [5.0, 0.0]
```

adds a translation to the pattern.

### `shift_frame: "material"`

```json
"shift_frame": "material"
```

keeps the shift aligned with the fixed material axes:

\[
\mathbf p_{uv}
=
\mathbf c
+
\mathbf R(\phi)\mathbf p_{\mathrm{local}}
+
\mathbf s.
\]

With:

```json
"center_uv_mm": [20.0, -10.0],
"shift_uv_mm": [5.0, 0.0],
"rotation_deg": 30.0
```

the nominal pattern location is:

\[
(20,-10)+(5,0)=(25,-10)\ \mathrm{mm},
\]

and the footprint is rotated by \(30^\circ\).

### `shift_frame: "pattern"`

```json
"shift_frame": "pattern"
```

rotates the shift with the zigzag:

\[
\mathbf p_{uv}
=
\mathbf c
+
\mathbf R(\phi)
\left(
\mathbf p_{\mathrm{local}}+\mathbf s
\right).
\]

For most controlled panel-position trials, `"material"` is the clearer choice.

### Footprint boundary check

The complete rotated and shifted zigzag footprint, including half the strip width, must remain inside the material-coordinate panel bounds. Otherwise the code stops with a placement error.

---

# Growth inputs

## List-based form

The canonical form specifies every strip explicitly:

```json
"n_strips": 4,
"gtop": [0.0030, 0.0028, 0.0026, 0.0024],
"gbot": [0.0, 0.0, 0.0, 0.0],
"ortho": [0.0, 0.0, 0.0, 0.0]
```

Every array must contain exactly `n_strips` entries. The parser does not silently pad or truncate a list.

Strip indexing follows the toolpath order:

```text
index 0             : left vertical strip
indices 1 to N - 2  : inclined strips
index N - 1         : right vertical strip
```

## Scalar form

A scalar is automatically broadcast to all strips:

```json
"n_strips": 4,
"gtop": 0.003,
"gbot": 0.0,
"ortho": 0.0
```

is equivalent to:

```json
"gtop": [0.003, 0.003, 0.003, 0.003],
"gbot": [0.0, 0.0, 0.0, 0.0],
"ortho": [0.0, 0.0, 0.0, 0.0]
```

The formats may be mixed:

```json
"gtop": [0.0030, 0.0028, 0.0026, 0.0024],
"gbot": 0.0,
"ortho": 0.0
```

No separate input-mode key is needed.

## Required and optional growth fields

| Key | Required? | Default if omitted | Meaning |
|---|---:|---:|---|
| `gtop` | Yes | — | Base top-layer engineering growth per strip. |
| `gbot` | No | `0.0` | Base bottom-layer engineering growth per strip. |
| `ortho` | No | `0.0` | Orthotropy coefficient per strip. Must lie in \([-1,1]\). |

## Orthotropic directional growth

For scalar growth \(g\) and orthotropy coefficient \(\eta\):

\[
g_1=g(1+\eta),
\qquad
g_2=g(1-\eta).
\]

Examples:

| `ortho` | Direction-1 growth | Direction-2 growth |
|---:|---:|---:|
| `0.0` | \(g\) | \(g\) |
| `0.2` | \(1.2g\) | \(0.8g\) |
| `-0.2` | \(0.8g\) | \(1.2g\) |

The direction frame follows the local zigzag-segment direction, including the global pattern rotation.

---

# Along-segment growth profiles

The profile modifies `gtop` along each individual segment.

## Uniform profile

```json
"profile": {
  "mode": "uniform"
}
```

uses:

\[
P(s)=1,
\]

so top growth is constant from segment start to segment end:

\[
g_{\mathrm{top}}(s)=g_{\mathrm{top,strip}}.
\]

`top_end_ratio` and `top_profile_power` are not needed for a uniform profile.

## Center-peak profile

```json
"profile": {
  "mode": "center_peak",
  "top_end_ratio": 0.4,
  "top_profile_power": 2.0
}
```

uses:

\[
P(s)
=
r+
(1-r)\sin^{p}(\pi s),
\qquad 0\le s\le1,
\]

where:

- \(s\) is normalized distance along the segment;
- \(r\) is `top_end_ratio`;
- \(p\) is `top_profile_power`.

The local top growth is:

\[
g_{\mathrm{top}}(s)
=
P(s)g_{\mathrm{top,strip}}.
\]

For `top_end_ratio = 0.4`, `top_profile_power = 2.0`, and `gtop = 0.003`:

| Segment position | Profile multiplier | Top growth |
|---|---:|---:|
| Start/end | 0.4 | 0.0012 |
| Quarter point | 0.7 | 0.0021 |
| Center | 1.0 | 0.0030 |

The current profile modifies top growth only. Bottom growth remains equal to its specified strip value.

A toolpath-level profile overrides the profile in `defaults`.

---

# Physical execution and history

## Toolpath order

Toolpaths execute in the order in which they appear in the JSON array.

## Every repeat is a physical cycle

For:

```json
{
  "id": "toolpath_A",
  "repeat": 3
}
```

the global sequence is:

```text
cycle 1: toolpath_A, repeat 1
cycle 2: toolpath_A, repeat 2
cycle 3: toolpath_A, repeat 3
```

The geometry is released after every repeat.

## History preserved between cycles

After cycle \(n\):

\[
\mathbf X_{\mathrm{start}}^{n+1}
=
\mathbf X_{\mathrm{final}}^{n}.
\]

The top and bottom target metrics and face hit counts remain accumulated. The code does not redefine the previous final geometry as a new stress-free state.

## Reference curvature

The treatment is represented through changes in the top and bottom target metrics. The shared reference curvature remains fixed:

\[
\mathbf b_r^{n+1}
=
\mathbf b_r^0.
\]

For a flat initial panel, \(\mathbf b_r^0=0\). For an initially curved stress-free panel, \(\mathbf b_r^0\) is the initial cylindrical reference curvature.

## Overlapping strips

Every strip/face intersection is an independent hit. There is no `last_wins` reduction in the sequence case.

If one face is covered by two strips during one toolpath:

1. The first hit uses the current hardening factor.
2. The target metric is updated.
3. The face count is updated if the hit qualifies.
4. The second hit uses the newly hardened state.

---

# Complete JSON example

```json
{
  "schema_version": 1,

  "units": {
    "length": "mm",
    "angle": "deg",
    "growth": "engineering_strain"
  },

  "hardening": {
    "model": "voce_decay",
    "history_variable": "hit_count",
    "beta": 0.223143551,
    "floor": 0.0,
    "scope": "face_shared",
    "count_rule": "plastic_hits"
  },

  "defaults": {
    "start_mode": "left_bottom_up",
    "profile": {
      "mode": "center_peak",
      "top_end_ratio": 0.4,
      "top_profile_power": 2.0
    }
  },

  "toolpaths": [
    {
      "id": "toolpath_A",
      "enabled": true,
      "repeat": 3,

      "operation": {
        "type": "zigzag",

        "lv_mm": 140.0,
        "alpha_deg": 9.13,
        "n_strips": 10,
        "width_mm": 10.0,

        "center_uv_mm": [0.0, 0.0],
        "shift_uv_mm": [0.0, 30.0],
        "shift_frame": "material",
        "rotation_deg": 0.0,

        "gtop": [
          0.003, 0.003, 0.003, 0.003, 0.003,
          0.003, 0.003, 0.003, 0.003, 0.003
        ],

        "gbot": [
          0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0
        ],

        "ortho": [
          0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0
        ]
      }
    },

    {
      "id": "toolpath_B_scalar_trial",
      "enabled": true,
      "repeat": 1,

      "operation": {
        "type": "zigzag",

        "lv_mm": 120.0,
        "alpha_deg": 12.0,
        "n_strips": 8,
        "width_mm": 8.0,

        "center_uv_mm": [20.0, -10.0],
        "shift_uv_mm": [5.0, 0.0],
        "shift_frame": "material",
        "rotation_deg": 30.0,

        "gtop": 0.0025,
        "gbot": 0.0,
        "ortho": 0.0,

        "profile": {
          "mode": "uniform"
        }
      }
    }
  ]
}
```

This file executes four physical cycles:

| Executed cycle | Toolpath | Repeat |
|---:|---|---:|
| 1 | `toolpath_A` | 1 of 3 |
| 2 | `toolpath_A` | 2 of 3 |
| 3 | `toolpath_A` | 3 of 3 |
| 4 | `toolpath_B_scalar_trial` | 1 of 1 |

---

# Common case templates

## Repeat one identical toolpath

```json
{
  "schema_version": 1,
  "hardening": {
    "model": "voce_decay",
    "beta": 0.223143551,
    "floor": 0.0,
    "count_rule": "plastic_hits"
  },
  "toolpaths": [
    {
      "id": "repeated_centered_zigzag",
      "repeat": 5,
      "operation": {
        "type": "zigzag",
        "lv_mm": 140.0,
        "alpha_deg": 9.13,
        "n_strips": 10,
        "width_mm": 10.0,
        "center_uv_mm": [0.0, 0.0],
        "shift_uv_mm": [0.0, 30.0],
        "shift_frame": "material",
        "rotation_deg": 0.0,
        "gtop": 0.003,
        "gbot": 0.0,
        "ortho": 0.0
      }
    }
  ]
}
```

## Disable hardening

```json
"hardening": {
  "model": "none",
  "history_variable": "hit_count",
  "beta": 0.0,
  "floor": 0.0,
  "scope": "face_shared",
  "count_rule": "plastic_hits"
}
```

## Shift and rotate a later toolpath

```json
{
  "id": "shifted_rotated_pass",
  "repeat": 1,
  "operation": {
    "type": "zigzag",
    "lv_mm": 120.0,
    "alpha_deg": 10.0,
    "n_strips": 8,
    "width_mm": 8.0,
    "center_uv_mm": [20.0, -10.0],
    "shift_uv_mm": [5.0, 0.0],
    "shift_frame": "material",
    "rotation_deg": 30.0,
    "gtop": 0.0025,
    "gbot": 0.0,
    "ortho": 0.0
  }
}
```

## Vary growth by strip

```json
{
  "n_strips": 6,
  "gtop": [0.0015, 0.0020, 0.0025, 0.0025, 0.0020, 0.0015],
  "gbot": 0.0,
  "ortho": 0.0
}
```

---

# Output files

The run tag is:

```text
bilayer_zigzag_sequence
```

## Sequence summary

Every run writes:

```text
bilayer_zigzag_sequence_summary.csv
```

One row is flushed after every completed physical cycle.

Columns are:

```text
executed_cycle_index
toolpath_id
toolpath_sequence_index
repeat_index
repeat_count
operation_count
hit_event_count
covered_face_count
max_hits_on_one_face_this_cycle
max_total_face_hit_count
hardening_factor_min
hardening_factor_mean
hardening_factor_max
max_abs_top_increment
max_abs_bottom_increment
max_displacement
min_U3
max_U3
total_energy
max_tangency_error
max_bbar_drift
mapping_file
final_file
```

Important checks:

- `max_total_face_hit_count` should increase as expected.
- `hardening_factor_mean` should decrease in repeatedly treated regions when `voce_decay` is active.
- `max_tangency_error` should remain near numerical zero.
- `max_bbar_drift` should remain near zero.
- `mapping_file` and `final_file` provide direct links between the summary row and its VTP outputs.

## Mapping output

Each physical cycle writes a mapping file such as:

```text
bilayer_zigzag_sequence_cycle_001_toolpath_A_r001_mapping.vtp
```

It contains the start-of-cycle geometry and diagnostic fields such as:

```text
material_u
material_v
executed_cycle_index
hits_this_cycle
total_hits_before_cycle
growth_dir_3d_hit_01
hit_active_01
growth_angle_material_hit_01
strip_index_hit_01
base_gtop_hit_01
base_gbot_hit_01
base_ortho_hit_01
```

Additional `hit_02`, `hit_03`, and higher-rank fields are written when a face receives overlapping passes.

## Final-state output

Each physical cycle writes a final file such as:

```text
bilayer_zigzag_sequence_cycle_001_toolpath_A_r001_final.vtp
```

It includes:

```text
U_from_cycle0
U3_from_cycle0
Umag_from_cycle0
hits_this_cycle
total_hit_count
hardening_factor_first_hit
hardening_factor_last_hit
hardening_factor_next_hit
effective_increment_g1_top_sum
effective_increment_g2_top_sum
effective_increment_g1_bot_sum
effective_increment_g2_bot_sum
abar_top_11
abar_top_12
abar_top_22
abar_bot_11
abar_bot_12
abar_bot_22
bbar_ref_11
bbar_ref_12
bbar_ref_22
bbar_reference_drift_norm
gauss
mean
```

The accumulated target metric components are the authoritative treatment-history fields. The summed incremental growth fields are convenient diagnostics but do not replace the full tensor history when different hit directions overlap.

## Optional internal state files

With:

```bash
-cycle_write_state true
```

the code also writes an internal mesh state after every physical cycle.

## Final STL

With:

```bash
-export_stl true
```

the final deformed sequence geometry is written as:

```text
bilayer_zigzag_sequence_final_deformed.stl
```

---

# Common errors

## `enable_passE must be false`

Use:

```bash
-enable_passE false
```

The JSON sequence already represents hardening through reduced incremental eigenstrain.

## `use -nsteps 1`

Use:

```bash
-nsteps 1
```

The JSON sequence itself controls the number of physical cycles.

## Growth-list size mismatch

If:

```json
"n_strips": 8
```

then every array-form `gtop`, `gbot`, or `ortho` must contain exactly eight values.

## Unknown JSON key

The parser rejects unknown keys. Check spelling and nesting carefully.

## Footprint outside material bounds

Reduce the pattern dimensions, rotation, center offset, or shift so that the complete footprint remains inside the panel.

## No enabled toolpaths

At least one entry must have:

```json
"enabled": true
```

or omit the `enabled` field, whose default is `true`.

## No covered faces

Check the panel dimensions, toolpath center, shift, rotation, strip width, and mesh resolution.

---

# Current limitations

The current schema supports:

- one zigzag operation per toolpath recipe;
- `left_bottom_up` start mode;
- millimeter length input;
- degree angle input;
- engineering-strain growth input;
- one shared face-level hardening count for top and bottom layers;
- one elastic release after every toolpath repeat;
- fixed reference curvature during the treatment sequence.

Future extensions may add multiple operations per toolpath, release control between operations, other path types, separate top/bottom history variables, and stress-based plasticity laws.
