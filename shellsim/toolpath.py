"""Build a zigzag toolpath sequence (the JSON the shell solver reads). Lengths in mm, angles in degrees."""
import json
from pathlib import Path

VOCE = {'model': 'voce_decay', 'history_variable': 'hit_count', 'beta': 0.2231435513142097, 'floor': 0,
        'scope': 'face_shared', 'count_rule': 'plastic_hits'}


def zigzag(lv_mm, alpha_deg, n_strips, width_mm, gtop, gbot=0.0, ortho=None, center_uv_mm=(0.0, 0.0),
           rotation_deg=0.0, **extra):
    """One zigzag wheel pass: stroke length `lv_mm`, strip angle `alpha_deg`, `n_strips` strips of `width_mm`,
    growth strains `gtop`/`gbot`, and orthotropy: one `ortho`, or both `ortho_top` and `ortho_bottom`. Other keys (profile, ...) pass through."""
    op = {'type': 'zigzag', 'lv_mm': lv_mm, 'alpha_deg': alpha_deg, 'n_strips': n_strips, 'width_mm': width_mm,
          'center_uv_mm': list(center_uv_mm), 'rotation_deg': rotation_deg, 'gtop': gtop, 'gbot': gbot}
    if ortho is not None:                    # either one `ortho`, or both ortho_top and ortho_bottom (via **extra)
        op['ortho'] = ortho
    op.update(extra)
    return op


def sequence(operations, hardening=None):
    """`operations`: list of (id, operation[, repeat]) applied in order, accumulated in order (the solver releases to equilibrium once, after the last pass, in `final_only` mode)."""
    toolpaths = [{'id': o[0], 'enabled': True, 'repeat': o[2] if len(o) > 2 else 1, 'operation': o[1]}
                 for o in operations]
    return {'schema_version': 1, 'units': {'length': 'mm', 'angle': 'deg', 'growth': 'engineering_strain'},
            'hardening': hardening or {'model': 'none'}, 'defaults': {'start_mode': 'left_bottom_up'},
            'boundary_conditions': {'enabled': False}, 'toolpaths': toolpaths}


def save(seq, path):
    Path(path).write_text(json.dumps(seq, indent=2) + '\n')
    return Path(path)
