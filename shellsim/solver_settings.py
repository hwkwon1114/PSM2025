"""The one solver configuration (shellsim/configs/default_adaptive_cutback.json) for every driver."""
import json
from pathlib import Path

SOLVER_CONFIG = Path(__file__).resolve().parent / 'configs/default_adaptive_cutback.json'


def solver_options(tolerance=None, threads=None, seconds=None, fixture=False, config=SOLVER_CONFIG):
    """Shared solver options as command-line strings. `tolerance` tightens both gradient gates (e.g. '1e-14');
    `seconds` overrides the wall-clock watchdog; `fixture` is the short smoke-test setting."""
    options = dict(json.loads(Path(config).read_text())['options'])
    if tolerance is not None:
        options.update({'-native_hybrid_grad_tol': str(tolerance), '-equilibrium_grad_tol': str(tolerance)})
    if threads is not None:
        options['-hessian_threads'] = str(threads)
    if seconds is not None:
        options['-native_hybrid_seconds'] = str(seconds)
    if fixture:
        options.update({'-native_hybrid_warmup_attempts': '5', '-native_hybrid_seconds': '20'})
    return options


def typed(value):
    """'true'/'false'/numbers from the JSON strings, for configs that keep typed arguments."""
    if value in ('true', 'false'):
        return value == 'true'
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value
