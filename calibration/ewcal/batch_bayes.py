"""qLogEI proposals and the fixed four-worker campaign controller.

Controller phases: load/migrate, recover, check stopping rules, propose/reserve,
execute, finalize. Integrity checks live in batch_integrity.py; the driver owns
CLI setup. The v16 corrections require runtime verification and independent
reassessment; this is not cleared for scientific execution.
"""
import copy
import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from calibration.ewcal.common import resolve, save_json
from calibration.ewcal.progress import message
from calibration.ewcal.runner import Evaluator
from calibration.ewcal.batch_integrity import (
    DOMAIN, SCORING, METADATA, validate_campaign_ledger,
    IMMUTABLE_MAX_TOTAL_BUDGET, IMMUTABLE_MAX_NEW_EVALUATIONS,
    IMMUTABLE_BATCH_SIZE, IMMUTABLE_MAX_WAVES, MAX_CONSECUTIVE_FAILURES,
    DEFAULT_DEADLINE_RESERVE_SECONDS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDECESSOR_MANIFEST = PROJECT_ROOT / "calibration/data/processed/reconciled_training_manifest_v1.json"

# Imported names remain available for existing callers and tests.
DEFAULT_BASE_SEED = METADATA["base_seed"]


class ExclusiveLock:
    """Acquires and holds an exclusive non-blocking file lock for the campaign (B6)."""

    def __init__(self, lock_path):
        self.lock_path = Path(lock_path)
        self.fd = None

    def acquire(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_path, "a+")
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.seek(0)
            self.fd.truncate()
            self.fd.write(f"pid={os.getpid()} host={os.uname().nodename} time={time.time()}\n")
            self.fd.flush()
        except (IOError, OSError) as e:
            self.fd.close()
            self.fd = None
            raise RuntimeError(f"Campaign directory is locked by another process: {self.lock_path}") from e

    def release(self):
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
            except Exception:
                pass
            self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def validate_observation_record(rec, expected_attempt_id=None, expected_normalized_x=None, tolerance=1e-4):
    """Shared strict observation / publication validator (R2).

    Enforces typed attempt identity (positive int), shape (4,), bounds [0, 1]^4,
    non-Boolean numeric scalar objective for valid, rejection of valid carrying
    failure_class or error, non-empty string error for invalid, and recognized failure taxonomy.
    """
    if not isinstance(rec, dict):
        raise ValueError(f"Observation record must be a dict, got {type(rec)}")

    att_id = rec.get("attempt_id")
    if not isinstance(att_id, int) or isinstance(att_id, bool) or att_id <= 0:
        raise ValueError(f"Record attempt_id must be a positive integer, got {att_id!r} ({type(att_id)})")
    if expected_attempt_id is not None and att_id != expected_attempt_id:
        raise ValueError(f"Attempt ID mismatch: record has {att_id}, expected {expected_attempt_id}")

    norm_p = rec.get("normalized_parameters")
    if norm_p is None:
        raise ValueError(f"Record attempt {att_id} missing normalized_parameters")
    if not isinstance(norm_p, (list, tuple, np.ndarray)) or any(isinstance(x, (bool, np.bool_)) for x in norm_p):
        raise ValueError("Normalized parameters must be four non-Boolean numbers")
    norm_arr = np.asarray(norm_p, dtype=float)
    if norm_arr.shape != (4,):
        raise ValueError(f"Record attempt {att_id} normalized_parameters must have shape (4,), got {norm_arr.shape}")
    if not np.all(np.isfinite(norm_arr)):
        raise ValueError(f"Record attempt {att_id} normalized_parameters contains non-finite values: {norm_p}")
    if np.any(norm_arr < 0.0) or np.any(norm_arr > 1.0):
        raise ValueError(f"Record attempt {att_id} normalized_parameters strictly outside [0, 1]^4: {norm_p}")
    if expected_normalized_x is not None:
        exp_arr = np.asarray(expected_normalized_x, dtype=float)
        if exp_arr.shape != (4,) or not np.allclose(norm_arr, exp_arr, rtol=0.0, atol=tolerance):
            raise ValueError(f"Record attempt {att_id} coordinates mismatch: {norm_arr} != expected {exp_arr}")

    val = rec.get("valid")
    if not isinstance(val, bool):
        raise ValueError(f"Record attempt {att_id} must have boolean 'valid', got {val!r} ({type(val)})")

    if val:
        obj = rec.get("objective_mm")
        if not isinstance(obj, (int, float)) or isinstance(obj, bool) or not np.isfinite(obj) or obj <= 0.0:
            raise ValueError(
                f"Valid record attempt {att_id} must have positive finite numeric scalar objective_mm, got {obj!r}"
            )
        if rec.get("failure_class") is not None:
            raise ValueError(
                f"Valid record attempt {att_id} contradiction: carries failure_class={rec.get('failure_class')!r}"
            )
        if rec.get("error") is not None and rec.get("error") != "":
            raise ValueError(f"Valid record attempt {att_id} contradiction: carries error={rec.get('error')!r}")
    else:
        fc = rec.get("failure_class")
        if fc not in ("solver", "registration", "infrastructure", "timeout"):
            raise ValueError(
                f"Invalid record attempt {att_id} failure_class must be in ('solver', 'registration', 'infrastructure', 'timeout'), got {fc!r}"
            )
        err = rec.get("error")
        if not isinstance(err, str) or len(err.strip()) == 0:
            raise ValueError(f"Invalid record attempt {att_id} must have a non-empty string 'error', got {err!r}")

    return True


def validate_warm_start_state(state, expected_evaluator_identity=None, config=None, predecessor_manifest_path=None, history=None):
    from calibration.ewcal.batch_integrity import validate_predecessor
    return validate_predecessor(state, expected_evaluator_identity, config,
                                predecessor_manifest_path, validate_observation_record, history)


def is_admissible(pt, failures, valid_obs, batch_so_far, failure_radius=0.35, min_separation=0.08):
    """Check geometric admissibility: shape (4,), finiteness, bounds [0, 1], failure exclusion, and candidate separation (B2)."""
    pt = np.asarray(pt, dtype=float)
    if pt.shape != (4,):
        return False
    if not np.all(np.isfinite(pt)):
        return False
    if np.any(pt < 0.0) or np.any(pt > 1.0):
        return False

    for f in failures:
        f = np.asarray(f, dtype=float)
        if f.shape != (4,) or not np.all(np.isfinite(f)):
            continue
        dist = np.linalg.norm(pt - f)
        if np.isnan(dist) or dist < failure_radius:
            return False

    for v in valid_obs:
        v = np.asarray(v, dtype=float)
        if v.shape != (4,) or not np.all(np.isfinite(v)):
            continue
        dist = np.linalg.norm(pt - v)
        if np.isnan(dist) or dist < min_separation:
            return False

    for b in batch_so_far:
        b = np.asarray(b, dtype=float)
        if b.shape != (4,) or not np.all(np.isfinite(b)):
            continue
        dist = np.linalg.norm(pt - b)
        if np.isnan(dist) or dist < min_separation:
            return False

    return True


def fit_gp_model(valid_observations):
    """Fit a BoTorch SingleTaskGP model with Matérn-5/2 ARD covariance on normalized coordinates and -log(RMS)."""
    X_list = [r["normalized_parameters"] for r in valid_observations]
    Y_list = [-np.log(r["objective_mm"]) for r in valid_observations]

    train_X = torch.tensor(X_list, dtype=torch.double)
    train_Y = torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)

    d = train_X.shape[-1]
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    gp = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp, train_Y.max()


DEFAULT_REGISTRATION_SIDECAR = PROJECT_ROOT / "run/stage2_bayesian_optimization/registration_metrics_sidecar.json"
EXPECTED_SIDECAR_SHA256 = "90f3955337196a5bc9e71a9766aa1a60d427855cce25029f73c25cdfc09e4838"


def load_registration_sidecar(sidecar_path=None, expected_sha256=EXPECTED_SIDECAR_SHA256, required_attempts=20):
    """
    Load and strictly validate the schema-2 registration metrics sidecar.
    Fails closed on missing file, SHA256 mismatch, duplicate attempts, or margin discrepancy.
    """
    if sidecar_path is None:
        sidecar_path = DEFAULT_REGISTRATION_SIDECAR
    p = Path(sidecar_path)
    if not p.exists():
        raise FileNotFoundError(f"Registration metrics sidecar missing at {p}")

    raw_bytes = p.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError(
            f"Registration metrics sidecar SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )

    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Registration sidecar must have schema_version=2")

    records = payload.get("records")
    if not isinstance(records, list) or len(records) != required_attempts:
        raise ValueError(
            f"Registration sidecar must contain exactly {required_attempts} records, found {len(records) if isinstance(records, list) else 'non-list'}"
        )

    lookup = {}
    for r in records:
        att = r.get("attempt_id")
        if type(att) is not int or att <= 0 or att in lookup:
            raise ValueError(f"Duplicate or invalid attempt_id {att} in registration sidecar")

        for k in ("valid", "training_eligible", "source_file", "source_file_sha256"):
            if k not in r:
                raise ValueError(f"Missing required key {k} in registration sidecar record for attempt {att}")

        if type(r["valid"]) is not bool or type(r["training_eligible"]) is not bool:
            raise ValueError("Sidecar labels must be boolean")
        if r["training_eligible"]:
            if r.get("failure_class") in ("timeout", "solver", "infrastructure"):
                raise ValueError(
                    f"Contradictory sidecar record for attempt {att}: marked training_eligible with mechanics failure_class {r.get('failure_class')}"
                )

            um = r.get("u_max")
            cm = r.get("computed_margin")
            if um is None or cm is None or not np.isfinite(um) or not np.isfinite(cm):
                raise ValueError(f"Non-finite registration metrics for eligible attempt {att}")

            pose = np.asarray(r.get("pose"), dtype=float)
            rot = r.get("rotation_angle_deg")
            if pose.shape != (6,) or not np.all(np.isfinite(pose)) or rot is None or not np.isfinite(rot):
                raise ValueError(f"Missing registration pose for attempt {att}")
            measured_um = max(abs(float(rot)) / 5.0, max(np.abs(pose[3:]) / [5.0, 5.0, 20.0]))
            if not np.isclose(um, measured_um, rtol=0, atol=1e-10):
                raise ValueError(f"Sidecar utilization mismatch for attempt {att}")
            qual = bool(r.get("valid"))
            near_bound = bool(r.get("pose_near_bound"))
            stop_reason = r.get("stop_reason")

            if qual:
                expected_margin = min(um - 0.95, -0.01)
            else:
                if near_bound and stop_reason == "iteration_limit":
                    expected_margin = max(um - 0.95, 0.05)
                elif near_bound:
                    expected_margin = max(um - 0.95, 0.05)
                elif stop_reason == "iteration_limit":
                    expected_margin = max(um - 0.95, 0.08)
                else:
                    expected_margin = 0.10

            if not np.isclose(cm, expected_margin, atol=1e-7):
                raise ValueError(f"Margin mismatch for attempt {att}: recorded {cm}, recomputed {expected_margin}")

        elif r.get("failure_class") not in ("timeout", "solver", "infrastructure"):
            raise ValueError(f"Missing registration data for non-mechanics attempt {att}")
        lookup[att] = r

    if set(lookup) != set(range(1, required_attempts + 1)):
        raise ValueError("Incomplete sidecar attempt coverage")
    return lookup


def is_registration_eligible(rec, sidecar_lookup=None):
    """
    Determine if an observation reached registration and has valid registration metrics.
    Strictly excludes mechanics failures (timeout, solver divergence, numerical breakdown)
    and infrastructure failures.
    """
    att_id = rec.get("attempt_id")
    fc = rec.get("failure_class")
    if fc in ("timeout", "solver", "infrastructure"):
        return False

    if sidecar_lookup and att_id in sidecar_lookup:
        s_rec = sidecar_lookup[att_id]
        if s_rec.get("failure_class") in ("timeout", "solver", "infrastructure"):
            raise ValueError(f"Contradictory sidecar record for attempt {att_id}: has mechanics failure {s_rec.get('failure_class')}")
        return bool(s_rec.get("training_eligible", False))

    if rec.get("valid") or rec.get("registration_qualified"):
        return True

    # Registration rejections must have registration stop reason
    sr = rec.get("stop_reason")
    if not sr and "metrics" in rec and rec["metrics"]:
        sr = rec["metrics"].get("stop_reason")
    if fc == "registration" or sr in ("pose_near_bound", "iteration_limit", "objective_stagnation"):
        return True

    return False


def compute_registration_margin(rec, sidecar_lookup=None):
    """
    Compute continuous constraint margin g(x) <= 0 for registration feasibility.
    g(x) <= 0 indicates feasible registration within bounds.
    g(x) > 0 indicates bound violation or non-convergence.
    Fails closed if invoked on a mechanics or infrastructure failure.
    """
    att_id = rec.get("attempt_id")
    fc = rec.get("failure_class")
    if fc in ("timeout", "solver", "infrastructure"):
        raise ValueError(
            f"Attempt {att_id} is a mechanics/infrastructure failure ({fc}) "
            f"and cannot have a registration margin computed"
        )

    if sidecar_lookup and att_id in sidecar_lookup:
        s_rec = sidecar_lookup[att_id]
        if s_rec.get("failure_class") in ("timeout", "solver", "infrastructure"):
            raise ValueError(
                f"Contradictory sidecar record for attempt {att_id}: has mechanics failure {s_rec.get('failure_class')}"
            )
        if not s_rec.get("training_eligible"):
            raise ValueError(
                f"Attempt {att_id} is marked training_eligible=False in sidecar and has no valid registration margin"
            )
        margin = s_rec.get("computed_margin")
        if margin is None or not np.isfinite(margin):
            raise ValueError(f"Attempt {att_id} has invalid/non-finite computed_margin in sidecar")
        return float(margin)

    is_valid = bool(rec.get("valid") or rec.get("gp_eligible") or rec.get("registration_qualified"))
    m = None
    if "metrics" in rec and rec["metrics"]:
        m = rec["metrics"]
    elif "cases" in rec and rec["cases"]:
        m = rec["cases"][0].get("metrics")
    elif "rejected_case" in rec and rec["rejected_case"]:
        m = rec["rejected_case"].get("metrics")

    rot_deg = rec.get("rotation_angle_deg")
    near_bound = rec.get("pose_near_bound")
    stop_reason = rec.get("stop_reason")

    if not m or m.get("rotation_angle_deg", rot_deg) is None or m.get("pose") is None:
        raise ValueError(f"Missing registration measurements for attempt {att_id}")
    u_t = 0.0
    if m:
        rot_deg = m.get("rotation_angle_deg", rot_deg)
        near_bound = m.get("pose_near_bound", near_bound)
        stop_reason = m.get("stop_reason", stop_reason)
        pose = m.get("pose")
        if len(pose) != 6 or not np.all(np.isfinite(np.asarray(pose, dtype=float))):
            raise ValueError(f"Invalid registration pose for attempt {att_id}")
        tx, ty, tz = pose[3:6]
        u_t = max(abs(float(tx)) / 5.0, abs(float(ty)) / 5.0, abs(float(tz)) / 20.0)

    if not np.isfinite(rot_deg):
        raise ValueError(f"Invalid registration rotation for attempt {att_id}")
    u_rot = abs(float(rot_deg)) / 5.0
    u_max = max(u_rot, u_t)

    if is_valid:
        return min(u_max - 0.95, -0.01)
    else:
        if near_bound and stop_reason == "iteration_limit":
            return max(u_max - 0.95, 0.05)
        elif near_bound:
            return max(u_max - 0.95, 0.05)
        elif stop_reason == "iteration_limit":
            return max(u_max - 0.95, 0.08)
        else:
            return 0.10



def fit_constrained_gp_model(records, sidecar_lookup=None):
    """
    Fit a BoTorch ModelListGP([model_f, model_g]) with Matérn-5/2 ARD covariance:
      model_f: objective GP on valid observations f(x) = -log(RMS)
      model_g: constraint GP on registration feasibility margin g(x) <= 0.
               Trained only on observations that completed mechanics and reached registration.
               (Timeouts and solver failures are strictly excluded from g(x)).
               Historical endpoints P03, P06, P07 are included as infeasible points (g > 0).
    """
    valid_obs = [r for r in records if r.get("valid", False)]
    if len(valid_obs) < 2:
        raise ValueError(f"Need at least 2 valid observations to fit GP, found {len(valid_obs)}")

    X_list = [r["normalized_parameters"] for r in valid_obs]
    Y_list = [-np.log(r["objective_mm"]) for r in valid_obs]
    train_X = torch.tensor(X_list, dtype=torch.double)
    train_Y = torch.tensor(Y_list, dtype=torch.double).unsqueeze(-1)

    d = train_X.shape[-1]
    covar_f = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    model_f = SingleTaskGP(train_X, train_Y, covar_module=covar_f, outcome_transform=Standardize(m=1))
    mll_f = ExactMarginalLogLikelihood(model_f.likelihood, model_f)
    fit_gpytorch_mll(mll_f)

    # Constraint model: trained strictly on observations that reached registration (pure registration data)
    reg_records = [r for r in records if is_registration_eligible(r, sidecar_lookup=sidecar_lookup)]
    if len(reg_records) < 2:
        raise ValueError(
            f"Need at least 2 registration-eligible observations to fit constraint GP, found {len(reg_records)}"
        )

    X_reg = [r["normalized_parameters"] for r in reg_records]
    g_reg = [compute_registration_margin(r, sidecar_lookup=sidecar_lookup) for r in reg_records]
    train_X_reg = torch.tensor(X_reg, dtype=torch.double)
    train_g_reg = torch.tensor(g_reg, dtype=torch.double).unsqueeze(-1)

    covar_g = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    model_g = SingleTaskGP(train_X_reg, train_g_reg, covar_module=covar_g, outcome_transform=Standardize(m=1))
    mll_g = ExactMarginalLogLikelihood(model_g.likelihood, model_g)
    fit_gpytorch_mll(mll_g)

    model_list = ModelListGP(model_f, model_g)
    return model_list, train_Y.max()


def propose_q_batch(
    records,
    q=4,
    failure_radius=0.35,
    min_separation=0.08,
    seed=2026,
    num_restarts=10,
    raw_samples=512,
    max_rejection_samples=5000,
    use_constrained_gp=False,
    sidecar_lookup=None,
):
    """Propose a batch of q points jointly optimizing qLogEI with registration boundary constraints."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Every attempted point imposes separation, including timeouts. Only physical
    # solver/registration failures impose the larger exclusion sphere below.
    valid_obs_pts = [np.array(r["normalized_parameters"], dtype=float) for r in records]
    valid_obs = [r for r in records if r.get("valid", False)]
    if len(valid_obs) < 2:
        raise ValueError(f"Need at least 2 valid observations to fit GP, found {len(valid_obs)}")

    if use_constrained_gp:
        # FAIL CLOSED: no silent try/except fallback. Any failure during model fitting
        # or acquisition construction raises immediately before candidate dispatch.
        model_list, best_f = fit_constrained_gp_model(records, sidecar_lookup=sidecar_lookup)
        qLogEI = qLogExpectedImprovement(
            model=model_list,
            best_f=best_f,
            objective=GenericMCObjective(objective=lambda Y, X=None: Y[..., 0]),
            constraints=[lambda Y: Y[..., 1]],  # evaluates g(x) <= 0
        )
        # When using constrained GP, registration feasibility is natively enforced by acquisition.
        # Failure exclusion radius is eliminated (effective_failure_radius = 0.0), while
        # min_separation between points is preserved.
        effective_failure_radius = 0.0
        acquisition_mode = "constrained_qlogei"
    else:
        gp, best_f = fit_gp_model(valid_obs)
        qLogEI = qLogExpectedImprovement(model=gp, best_f=best_f)
        effective_failure_radius = failure_radius
        acquisition_mode = "single_objective_qlogei"

    bounds = torch.stack([torch.zeros(4, dtype=torch.double), torch.ones(4, dtype=torch.double)])

    # Optimize acquisition function
    candidates, acq_val = optimize_acqf(
        qLogEI,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    raw_batch = candidates.detach().cpu().numpy()


    if not np.all(np.isfinite(raw_batch)):
        raise RuntimeError("qLogEI optimization produced non-finite candidates")

    raw_acq_float = float(acq_val.item()) if torch.is_tensor(acq_val) else float(acq_val)
    if not np.isfinite(raw_acq_float):
        raise RuntimeError(f"qLogEI optimization produced non-finite acquisition value: {raw_acq_float}")

    # Extract failure points (solver or registration - timeouts explicitly DO NOT form exclusion spheres) (B2)
    failures = [
        np.array(r["normalized_parameters"], dtype=float)
        for r in records
        if not r.get("valid", False) and r.get("failure_class") in ("solver", "registration")
    ]

    cleaned_batch = []
    clipped_indices = []
    random_replaced_indices = []
    repaired_indices = []
    repair_types = {}
    repair_attempts = 0

    for i in range(q):
        raw_pt = np.asarray(raw_batch[i], dtype=float)
        clipped_pt = np.clip(raw_pt, 0.0, 1.0)
        # Exact clipping detection (B2)
        was_clipped = not np.array_equal(clipped_pt, raw_pt)
        if was_clipped:
            clipped_indices.append(i)

        if not was_clipped and is_admissible(raw_pt, failures, valid_obs_pts, cleaned_batch, effective_failure_radius, min_separation):
            cleaned_batch.append(raw_pt)
        elif was_clipped and is_admissible(clipped_pt, failures, valid_obs_pts, cleaned_batch, effective_failure_radius, min_separation):
            cleaned_batch.append(clipped_pt)
            repaired_indices.append(i)
            repair_types[i] = "clipping"
        else:
            repaired_indices.append(i)
            random_replaced_indices.append(i)
            repair_types[i] = "clipping_and_random_replacement" if was_clipped else "random_replacement"
            found = False
            for _ in range(max_rejection_samples):
                repair_attempts += 1
                rnd_pt = np.random.uniform(0.0, 1.0, size=4)
                if is_admissible(rnd_pt, failures, valid_obs_pts, cleaned_batch, effective_failure_radius, min_separation):
                    cleaned_batch.append(rnd_pt)
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Constraint satisfaction failure: unable to find candidate satisfying failure exclusion "
                    f"(r={effective_failure_radius}) and separation (min_sep={min_separation}) within {max_rejection_samples} draws."
                )

    # Final hard re-assertion across the full batch (B2) - unconditional raises
    final_batch = np.array(cleaned_batch)
    for i in range(q):
        other_batch = [final_batch[j] for j in range(q) if j != i]
        if not is_admissible(final_batch[i], failures, valid_obs_pts, other_batch, effective_failure_radius, min_separation):
            raise RuntimeError(f"Candidate {i} violates final admissibility criteria unconditionally")


    diag = {
        "acquisition_mode": acquisition_mode,
        "raw_acq_value": raw_acq_float,
        "raw_proposals": raw_batch.tolist(),
        "final_proposals": final_batch.tolist(),
        "clipped_indices": clipped_indices,
        "random_replaced_indices": random_replaced_indices,
        "repaired_indices": repaired_indices,
        "repair_types": {str(k): v for k, v in repair_types.items()},
        "repaired_candidates": len(repaired_indices),
        "repair_attempts": repair_attempts,
        "valid_count": len(valid_obs),
        "registration_training_count": len([r for r in records if is_registration_eligible(r, sidecar_lookup=sidecar_lookup)]) if use_constrained_gp else None,
        "failures_count": len(failures),
    }
    return final_batch, diag


def get_epoch_start_attempt(state):
    """Retrieve the starting attempt ID of the active stopping epoch (defaults to 1)."""
    if isinstance(state, dict):
        epoch = state.get("stopping_epoch")
        if isinstance(epoch, dict):
            return int(epoch.get("epoch_start_attempt", 1))
    return 1


def reconstruct_proposal_order_streak(observations, max_failures=MAX_CONSECUTIVE_FAILURES, epoch_start_attempt=1):
    """
    Reconstruct failure streak strictly along the contiguous completed prefix
    in proposal order (sorted by attempt_id), latching if the threshold (>= 6)
    was ever crossed within the active stopping epoch (attempt_id >= epoch_start_attempt).
    Stops at unresolved attempt gaps to prevent out-of-order false latching.
    """
    by_att = {o["attempt_id"]: o for o in observations if isinstance(o, dict) and "attempt_id" in o}
    if not by_att:
        return 0, False

    current_streak = 0
    latched = False
    curr_att = epoch_start_attempt
    while curr_att in by_att:
        obs = by_att[curr_att]
        if not obs.get("valid", False):
            current_streak += 1
            if current_streak >= max_failures:
                latched = True
        else:
            current_streak = 0
        curr_att += 1

    return current_streak, latched



def _read_evaluations_log(path, durable, require_in_ledger=False, record_comparator=None):
    """Read-only validation; the checkpoint is the canonical charged ledger."""
    from calibration.ewcal.batch_integrity import same_record
    logged = {}
    if not path.exists():
        return logged
    lines = path.read_text().splitlines(keepends=True)
    for idx, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as error:
            if idx == len(lines) - 1 and not raw.endswith("\n") and error.msg != "Extra data":
                break
            raise ValueError("Interior or complete-line JSONL corruption") from error
        validate_observation_record(entry)
        att = entry["attempt_id"]
        if require_in_ledger and att not in durable:
            raise ValueError(f"JSONL outcome has no reconciled charged reservation: attempt {att}")
        for previous in (logged.get(att), durable.get(att)):
            if previous is not None and not (record_comparator or same_record)(entry, previous, validate_observation_record):
                raise ValueError(f"Conflicting JSONL record for attempt {att}")
        logged[att] = entry
    return logged


def sync_evaluations_log(evaluations_log_path, observations):
    """Atomic replay; only an unterminated, undecodable final line is repairable."""
    from calibration.ewcal.batch_integrity import unique_records, same_record
    path = Path(evaluations_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExclusiveLock(path.parent / ".evaluations_log.lock"):
        durable = unique_records(observations, validate_observation_record)
        logged = _read_evaluations_log(path, durable)
        added = len(set(durable) - set(logged))
        records = dict(logged)
        records.update(durable)
        fd, temporary = tempfile.mkstemp(prefix="evaluations_", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                for att in sorted(records):
                    f.write(json.dumps(records[att], allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        return added


def reconcile_reservations(base, state, expected_evaluator_identity=None):
    """Validate every existing publication, including already-checkpointed results."""
    from calibration.ewcal.batch_integrity import positive_id, unique_records, same_record
    records = state.setdefault("observations", [])
    completed = state.setdefault("completed_attempts", [])
    indexed = unique_records(records, validate_observation_record)
    recovered, infra = 0, 0
    seen = set()
    for res in state.setdefault("reserved_attempts", []):
        att = positive_id(res.get("attempt_id"))
        if att in seen:
            raise ValueError("Duplicate reservation attempt ID")
        seen.add(att)
        if att <= 12 and state.get("campaign_type") == "batch_qlogei":
            # Frozen historical publications have their own identity, paths and
            # the explicit P06/P07 error reclassification (not a new solve).
            publication = state.get("historical_publications", {}).get(str(att))
            if publication is None:
                raise ValueError("Historical reservation has no explicit publication binding")
            expected_res = publication["reservation"]
            if res != expected_res:
                raise ValueError("Historical reservation identity/path/metadata mismatch")
            primary = Path(res["result_file"])
            if hashlib.sha256(primary.read_bytes()).hexdigest() != publication["sha256"]:
                raise ValueError(f"Historical publication changed: attempt {att}")
            original = json.loads(primary.read_text())
            validate_observation_record(original, expected_attempt_id=att, expected_normalized_x=res["x"], tolerance=1e-10)
            canonical = copy.deepcopy(original)
            # Only the two explicitly documented taxonomy corrections are
            # permitted; both source bytes and canonical observations are bound.
            if att in (4, 5) and canonical.get("error") == "out_of_domain":
                canonical["error"] = publication["observation"]["error"]
            if att not in indexed or not same_record(canonical, indexed[att], validate_observation_record):
                raise ValueError(f"Historical publication/observation conflict: attempt {att}")
            alternate = primary.with_name("trial_result.json")
            if alternate.exists():
                other = json.loads(alternate.read_text())
                if not same_record(original, other, validate_observation_record):
                    raise ValueError(f"Conflicting historical publication: attempt {att}")
            continue
        validate_observation_record({"attempt_id": att, "normalized_parameters": res.get("x"),
                                     "valid": False, "failure_class": "infrastructure", "error": "reserved"})
        expected_dir = Path(base) / "evaluations" / f"attempt_{att:04d}"
        pub_dir = Path(res.get("publication_dir", ""))
        lineage = state.get("evaluator_lineage")
        expected_eval_identity = expected_evaluator_identity
        if lineage:
            from calibration.ewcal.batch_integrity import get_lineage_evaluator_identity
            expected_eval_identity = get_lineage_evaluator_identity(lineage, att, expected_evaluator_identity)
        mismatch = (expected_eval_identity is not None and res.get("evaluator_identity") != expected_eval_identity)
        mismatch = mismatch or pub_dir.resolve() != expected_dir.resolve()
        if res.get("result_file") is not None and Path(res["result_file"]).resolve() != (expected_dir / "result.json").resolve():
            mismatch = True
        if mismatch:
            raise ValueError("Reservation identity or publication directory mismatch")
        publications = []
        for name in ("trial_result.json", "result.json"):
            path = pub_dir / name
            if not path.exists():
                continue
            try:
                rec = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                # Never discard contradictory complete JSON followed by garbage.
                # Even a truncated publication is held for explicit reconciliation;
                # absence and malformed bytes are deliberately different states.
                raise ValueError(f"Corrupt or incomplete publication: {path}") from error
            validate_observation_record(rec, expected_attempt_id=att, expected_normalized_x=res["x"], tolerance=1e-10)
            if expected_eval_identity is not None and rec.get("identity") != expected_eval_identity:
                raise ValueError("Publication evaluator identity mismatch")
            publications.append(rec)
        existing = indexed.get(att)
        candidates = publications + ([existing] if existing is not None else [])
        for candidate in candidates[1:]:
            if not same_record(candidates[0], candidate, validate_observation_record):
                raise ValueError(f"Conflicting publications for attempt {att}")
        if existing is not None:
            validate_observation_record(existing, expected_attempt_id=att, expected_normalized_x=res["x"], tolerance=1e-10)
            if not publications and existing.get("failure_class") != "infrastructure":
                raise ValueError("Checkpointed observation has no verified publication")
            continue
        if publications:
            rec = copy.deepcopy(publications[0])
            res["status"] = "completed"
        else:
            rec = {"attempt_id": att, "normalized_parameters": res["x"],
                   "identity": expected_evaluator_identity, "valid": False,
                   "failure_class": "infrastructure",
                   "error": "Interrupted execution during uncompleted wave; no complete verified result on disk"}
            res["status"] = "completed_interrupted"
        records.append(rec)
        completed.append(att)
        indexed[att] = rec
        if rec.get("failure_class") == "infrastructure":
            infra += 1
        else:
            recovered += 1
    return recovered, infra


def get_remaining_allocation_seconds(start_time=None, total_time_limit_seconds=None):
    """Calculate remaining wall-clock time in Slurm allocation or configured limit (B7)."""
    import subprocess

    if start_time is not None:
        st_val = float(start_time)
        if not np.isfinite(st_val):
            raise ValueError(f"start_time must be finite: {start_time}")
    if total_time_limit_seconds is not None:
        limit_val = float(total_time_limit_seconds)
        if not np.isfinite(limit_val) or limit_val <= 0:
            raise ValueError(f"total_time_limit_seconds must be positive finite: {total_time_limit_seconds}")

    slurm_remaining = None
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        try:
            res = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                time_str = res.stdout.strip()
                days = 0
                if "-" in time_str:
                    days_part, time_str = time_str.split("-", 1)
                    days = int(days_part)
                parts = [int(p) for p in time_str.split(":")]
                if len(parts) == 3:
                    h, m, s = parts
                elif len(parts) == 2:
                    h, m, s = 0, parts[0], parts[1]
                elif len(parts) == 1:
                    h, m, s = 0, 0, parts[0]
                else:
                    parts = None
                if parts is not None:
                    total_s = days * 86400 + h * 3600 + m * 60 + s
                    if np.isfinite(total_s) and total_s >= 0:
                        slurm_remaining = float(total_s)
        except Exception:
            pass

    caller_remaining = None
    if total_time_limit_seconds is not None and start_time is not None:
        elapsed = time.time() - float(start_time)
        caller_remaining = max(0.0, float(total_time_limit_seconds) - elapsed)

    if slurm_remaining is not None and caller_remaining is not None:
        # Conservative minimum of Slurm and caller allocation remaining time (B7)
        return min(slurm_remaining, caller_remaining)
    elif slurm_remaining is not None:
        return slurm_remaining
    elif caller_remaining is not None:
        if job_id:
            # We are inside a Slurm job, but squeue failed to establish remaining allocation time!
            # Fail closed rather than falling back to caller clock (B7)
            raise RuntimeError(
                f"Running inside Slurm job {job_id} but unable to query authoritative remaining allocation time from squeue; failing closed"
            )
        return caller_remaining

    # Fail closed: cannot establish remaining wall-clock time
    raise RuntimeError("Cannot establish remaining wall-clock allocation time from Slurm or caller limits (fail-closed)")


def _validate_invocation(iterations, batch_size, max_total_evaluations, base_seed, deadline_reserve_seconds):
    """Check the existing fixed campaign limits before touching its files."""
    # Validate typed invocation limits before opening the campaign.
    if type(base_seed) is not int or base_seed != DEFAULT_BASE_SEED:
        raise ValueError("Base seed must remain 2026")
    if type(iterations) is not int or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if type(max_total_evaluations) is not int or max_total_evaluations < 12:
        raise ValueError("Total ceiling must include the twelve predecessors")
    # Enforce immutable protocol ceilings (B4)
    if max_total_evaluations > IMMUTABLE_MAX_TOTAL_BUDGET:
        raise ValueError(
            f"max_total_evaluations ({max_total_evaluations}) exceeds immutable campaign ceiling ({IMMUTABLE_MAX_TOTAL_BUDGET})"
        )
    if batch_size != IMMUTABLE_BATCH_SIZE:
        raise ValueError(f"batch_size must be exactly {IMMUTABLE_BATCH_SIZE}, got {batch_size}")
    if iterations > IMMUTABLE_MAX_WAVES:
        raise ValueError(f"iterations ({iterations}) exceeds maximum campaign waves ({IMMUTABLE_MAX_WAVES})")
    if deadline_reserve_seconds is None or not np.isfinite(float(deadline_reserve_seconds)):
        raise ValueError(f"deadline_reserve_seconds must be a finite number, got {deadline_reserve_seconds}")
    if float(deadline_reserve_seconds) < DEFAULT_DEADLINE_RESERVE_SECONDS:
        raise ValueError(
            f"deadline_reserve_seconds must be >= {DEFAULT_DEADLINE_RESERVE_SECONDS}s, got {deadline_reserve_seconds}"
        )


def _load_campaign(state_file, base, evaluator, predecessor_manifest_path, base_seed, resume_option_b=False):
    """Prepare a migration/restart candidate entirely in memory; no writes."""
    if not state_file.exists():
        raise FileNotFoundError(f"Missing bayes_state.json at {state_file}")

    state_bytes = state_file.read_bytes()
    state = json.loads(state_bytes)

    # Option B protocol transition candidate in memory (no writes before validation)
    if resume_option_b and state.get("campaign_type") == "batch_qlogei" and state.get("status") in ("consecutive_failures", "stopped_consecutive_failures"):
        epoch = state.get("stopping_epoch")
        if not epoch or not epoch.get("transition_applied"):
            AUTHORIZED_PRIOR_EVALUATOR = "630b8877d0c2d7dd725846a44ff8a15010d146bf487a2ea9446f1756f65edfe5"
            AUTHORIZED_PRIOR_DIGEST = "7a2d66011bbf062986c120f94500de2a280996dc4c3ee87c89b9689f093d930d"

            if len(state.get("completed_attempts", [])) != 20:
                raise ValueError(f"Option B transition requires exactly 20 completed attempts, found {len(state.get('completed_attempts', []))}")
            if len(state.get("observations", [])) != 20:
                raise ValueError(f"Option B transition requires exactly 20 observations, found {len(state.get('observations', []))}")
            if len(state.get("reserved_attempts", [])) != 20:
                raise ValueError(f"Option B transition requires exactly 20 reserved attempts, found {len(state.get('reserved_attempts', []))}")
            if state.get("evaluator_identity") != AUTHORIZED_PRIOR_EVALUATOR:
                raise ValueError(f"Option B transition requires prior evaluator {AUTHORIZED_PRIOR_EVALUATOR}, found {state.get('evaluator_identity')}")
            if state.get("historical_compatibility_digest") != AUTHORIZED_PRIOR_DIGEST:
                raise ValueError(f"Option B transition requires prior digest {AUTHORIZED_PRIOR_DIGEST}, found {state.get('historical_compatibility_digest')}")

            if any(key in state for key in ("evaluator_lineage", "stopping_epoch", "historical_import_transition", "option_b_resumed")):
                raise ValueError("Option B origin already contains transition metadata")
            validate_campaign_ledger(state, AUTHORIZED_PRIOR_EVALUATOR, base_seed, validate_observation_record)
            if (set(state["completed_attempts"]) != set(range(1, 21))
                    or state["charged_attempts"] != 20 or state["waves_reserved"] != 2
                    or not reconstruct_proposal_order_streak(state["observations"])[1]):
                raise ValueError("Option B requires the approved halted two-wave origin")
            prior_id = AUTHORIZED_PRIOR_EVALUATOR
            new_id = evaluator.identity
            lineage = {
                "schema_version": 2,
                "transition_reason": "option_b_constrained_gp",
                "transitions": [
                    {
                        "boundary_attempt": 21,
                        "prior_evaluator_identity": prior_id,
                        "new_evaluator_identity": new_id,
                    }
                ],
            }
            state["evaluator_lineage"] = lineage
            state["evaluator_identity"] = new_id
            state["historical_import_transition"] = {
                "prior_digest": AUTHORIZED_PRIOR_DIGEST,
                "new_digest": hashlib.sha256((PROJECT_ROOT / "run/optimization_review/historical_import_v16.json").read_bytes()).hexdigest(),
                "authorized_boundary": 21,
                "transition_reason": "option_b_constrained_gp",
            }
            state["stopping_epoch"] = {
                "epoch_start_attempt": 21,
                "protocol": "option_b_constrained_gp",
                "transition_applied": True,
            }
            state["status"] = "in_progress"
            state["proposal_order_failure_streak"] = 0
            state["option_b_resumed"] = True
            state["historical_compatibility_digest"] = state["historical_import_transition"]["new_digest"]

    from calibration.ewcal.batch_integrity import validate_option_b_transition
    validate_option_b_transition(state, evaluator.identity)
    if resume_option_b and not state.get("option_b_resumed"):
        raise ValueError("Option B flag requires a valid authorized transition")

    # B5: Validate imported state schema and provenance
    if predecessor_manifest_path is None:
        if DEFAULT_PREDECESSOR_MANIFEST.exists():
            predecessor_manifest_path = DEFAULT_PREDECESSOR_MANIFEST
        elif (base.parents[1] / "calibration/data/processed/reconciled_training_manifest_v1.json").exists():
            predecessor_manifest_path = base.parents[1] / "calibration/data/processed/reconciled_training_manifest_v1.json"
    elif not Path(predecessor_manifest_path).is_absolute() and not Path(predecessor_manifest_path).exists():
        if (PROJECT_ROOT / predecessor_manifest_path).exists():
            predecessor_manifest_path = PROJECT_ROOT / predecessor_manifest_path
        elif (base.parents[1] / predecessor_manifest_path).exists():
            predecessor_manifest_path = base.parents[1] / predecessor_manifest_path

    from calibration.ewcal.batch_integrity import validate_history_binding
    history = validate_history_binding(state, evaluator.identity,
                                       getattr(evaluator, "identity_record", None),
                                       state_bytes, predecessor_manifest_path)
    predecessor_digest, predecessor_sources = validate_warm_start_state(
        state,
        expected_evaluator_identity=evaluator.identity,
        config=getattr(evaluator, "config", None),
        predecessor_manifest_path=predecessor_manifest_path,
        history=history,
    )

    records = state.setdefault("observations", [])
    reserved = state.setdefault("reserved_attempts", [])
    completed = state.setdefault("completed_attempts", [])

    # B3: Authorized predecessor campaign migration
    if state.get("campaign_type") != "batch_qlogei":
        if len(records) == 12 and state.get("status") in ("budget_complete", "in_progress"):
            state["observations"][11]["tuple_id"] = predecessor_sources[11]["tuple_id"]
            state["schema_version"] = 2
            state["domain_definition"] = DOMAIN
            state["scoring_compatibility"] = SCORING
            state["predecessor_sources"] = predecessor_sources
            state["completed_attempts"] = list(range(1, 13))
            completed = state["completed_attempts"]
            state["charged_attempts"] = 12
            state["waves_reserved"] = 0
            state["evaluator_identity"] = evaluator.identity
            state["campaign_type"] = "batch_qlogei"
            state["max_evaluations"] = IMMUTABLE_MAX_TOTAL_BUDGET
            state["batch_size"] = IMMUTABLE_BATCH_SIZE
            state["budget_new"] = IMMUTABLE_MAX_NEW_EVALUATIONS
            state["status"] = "in_progress"
            state["proposal_order_failure_streak"] = 0
            state["campaign_metadata"] = {
                "base_seed": base_seed,
                "max_total_evaluations": IMMUTABLE_MAX_TOTAL_BUDGET,
                "batch_size": IMMUTABLE_BATCH_SIZE,
                "max_waves": IMMUTABLE_MAX_WAVES,
            }
            state["migration"] = {
                "predecessor": "stage2_sequential_warm_start_12",
                "predecessor_manifest": str(predecessor_manifest_path) if predecessor_manifest_path else None,
                "migrated_observations": 12,
                "target_total_budget": IMMUTABLE_MAX_TOTAL_BUDGET,
                "target_new_evaluations": IMMUTABLE_MAX_NEW_EVALUATIONS,
                "timestamp": time.time(),
                "predecessor_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            }
            if predecessor_manifest_path:
                pred_digest = hashlib.sha256(Path(predecessor_manifest_path).read_bytes()).hexdigest()
                state["predecessor_manifest_digest"] = pred_digest
                state["migration"]["predecessor_manifest_digest"] = pred_digest
            if history:
                state["historical_compatibility_digest"] = history["digest"]
                state["historical_publications"] = history["publications"]

    return state


def _recover_campaign(base, state_file, eval_log, state, evaluator, base_seed):
    """Validate a recovery candidate, then commit once; rejection preserves bytes."""
    from calibration.ewcal.batch_integrity import unique_records
    candidate = copy.deepcopy(state)
    validate_campaign_ledger(candidate, evaluator.identity, base_seed, validate_observation_record)
    recovered, infra = reconcile_reservations(base, candidate, evaluator.identity)
    validate_campaign_ledger(candidate, evaluator.identity, base_seed, validate_observation_record)
    durable = unique_records(candidate["observations"], validate_observation_record)
    _read_evaluations_log(Path(eval_log), durable, require_in_ledger=True)
    if recovered + infra:
        streak, crossed = reconstruct_proposal_order_streak(
            candidate["observations"], max_failures=MAX_CONSECUTIVE_FAILURES, epoch_start_attempt=get_epoch_start_attempt(candidate),
        )
        candidate["proposal_order_failure_streak"] = streak
        if crossed:
            candidate["status"] = "consecutive_failures"
    # No validation rejection beyond this point. Checkpoint-before-log remains
    # crash-recoverable; a log-only charge is never silently admitted/reissued.
    save_json(state_file, candidate)
    state.clear()
    state.update(candidate)
    sync_evaluations_log(eval_log, candidate["observations"])
    return infra


def _stop_before_launch(state, state_file, reconciled_infra, max_total_evaluations):
    """Apply the existing startup halt rules, before proposing any new wave."""
    records = state["observations"]
    # B3: Immediate zero-launch halt on recovered infrastructure failure
    if reconciled_infra > 0:
        state["status"] = "stopped_infrastructure_failure"
        save_json(state_file, state)
        message("BATCH-BO", f"Halted: recovered {reconciled_infra} uncompleted reservations as infrastructure failure. Zero launches.")
        return True

    # Check existing terminal state (B3, B4)
    terminal_statuses = (
        "budget_complete",
        "consecutive_failures",
        "stopped_consecutive_failures",
        "stopped_proposal_failures",
        "stopped_infrastructure_failure",
        "deadline_stop",
    )
    if state.get("status") in terminal_statuses:
        message("BATCH-BO", f"Campaign is already in terminal state '{state['status']}'. Halting with zero launches.")
        return True

    # B3: Check if ANY recorded observation is an infrastructure failure
    for r in records:
        if r.get("failure_class") == "infrastructure":
            state["status"] = "stopped_infrastructure_failure"
            save_json(state_file, state)
            message("BATCH-BO", f"Found existing infrastructure failure at attempt {r.get('attempt_id')}; halting with zero launches.")
            return True

    # B4: Deterministically reconstruct proposal-order failure streak from durable records
    reconstructed_streak, crossed = reconstruct_proposal_order_streak(
        records, max_failures=MAX_CONSECUTIVE_FAILURES, epoch_start_attempt=get_epoch_start_attempt(state)
    )
    streak = reconstructed_streak
    state["proposal_order_failure_streak"] = streak
    if crossed or streak >= MAX_CONSECUTIVE_FAILURES or state.get("status") == "consecutive_failures":
        state["status"] = "consecutive_failures"
        save_json(state_file, state)
        message("BATCH-BO", f"Durable proposal-order failure streak ({streak} >= 6) crossed threshold; halting with zero launches.")
        return True

    # B4: Enforce max_total_evaluations ceiling
    if len(records) >= max_total_evaluations:
        state["status"] = "budget_complete"
        save_json(state_file, state)
        message("BATCH-BO", f"Evaluation count ({len(records)}) >= ceiling ({max_total_evaluations}). Halting.")
        return True

    return False


def _execute_wave(evaluator, wave_reservations, state, state_file, eval_log, base, base_seed, state_lock,
                  ledger_validator=None):
    """Execute an already-reserved wave and checkpoint each completion."""
    current_q = len(wave_reservations)
    records = state["observations"]
    reserved = state["reserved_attempts"]
    completed = state["completed_attempts"]
    # Worker function with per-result checkpointing (B3, B6)
    infra_encountered = False

    def solve_worker(prop_idx, att_id, norm_x):
        nonlocal infra_encountered
        message("WORKER", f"[Worker {prop_idx}] Starting Attempt {att_id:04d}")
        try:
            # B6: write_to_evaluations_log=False prevents double/uncoordinated logging
            rec = evaluator.evaluate(norm_x, attempt_id=att_id, write_to_evaluations_log=False)
        except Exception as e:
            rec = {
                "attempt_id": att_id,
                "normalized_parameters": norm_x.tolist(),
                "identity": evaluator.identity,
                "valid": False,
                "failure_class": "infrastructure",
                "error": str(e),
            }

        validate_observation_record(rec, expected_attempt_id=att_id, expected_normalized_x=norm_x, tolerance=1e-10)
        if rec.get("identity") != evaluator.identity:
            raise ValueError("Live worker evaluator identity mismatch")
        rec["proposal_idx"] = prop_idx

        # Thread-safe checkpointing per result (B3, B6)
        with state_lock:
            reservation = next(r for r in reserved if r["attempt_id"] == att_id)
            reconcile_reservations(base, {"observations": [rec], "reserved_attempts": [reservation],
                                          "completed_attempts": []}, evaluator.identity)
            records.append(rec)
            completed.append(att_id)
            for r in reserved:
                if r["attempt_id"] == att_id:
                    r["status"] = "completed"

            state["observations"] = records
            state["completed_attempts"] = completed
            state["reserved_attempts"] = reserved
            (ledger_validator or validate_campaign_ledger)(state, evaluator.identity, base_seed, validate_observation_record)

            # Transactionally reconstruct failure streak and latched stopping (B4)
            cur_streak, hit_halt = reconstruct_proposal_order_streak(
                records, max_failures=MAX_CONSECUTIVE_FAILURES, epoch_start_attempt=get_epoch_start_attempt(state)
            )
            state["proposal_order_failure_streak"] = cur_streak
            if hit_halt:
                state["status"] = "consecutive_failures"

            save_json(state_file, state)

            # B6: Recoverable idempotent sync to evaluations.jsonl
            sync_evaluations_log(eval_log, records)

        if not rec.get("valid", False) and rec.get("failure_class") == "infrastructure":
            infra_encountered = True

        status_str = "VALID" if rec.get("valid") else f"FAILED ({rec.get('failure_class')})"
        obj_str = f"{rec.get('objective_mm', float('nan')):.4f} mm" if rec.get("valid") else "N/A"
        message("WORKER", f"[Worker {prop_idx}] Finished Attempt {att_id:04d}: {status_str}, RMS={obj_str}")
        return prop_idx, att_id, rec

    # Launch concurrent wave evaluations
    with ThreadPoolExecutor(max_workers=current_q) as executor:
        futures = [
            executor.submit(solve_worker, wave_reservations[i][3], wave_reservations[i][0], wave_reservations[i][1])
            for i in range(current_q)
        ]
        for future in as_completed(futures):
            future.result()

    return infra_encountered


def run_batch_bayes(
    evaluator,
    iterations=5,
    batch_size=4,
    max_total_evaluations=32,
    output_dir=None,
    base_seed=2026,
    num_restarts=10,
    raw_samples=512,
    start_time=None,
    total_time_limit_seconds=None,
    deadline_reserve_seconds=DEFAULT_DEADLINE_RESERVE_SECONDS,
    campaign_lock_acquired=False,
    predecessor_manifest_path=None,
    resume_option_b=False,
):
    """Coordinate startup, recovery, and complete reserved qLogEI waves."""
    _validate_invocation(iterations, batch_size, max_total_evaluations, base_seed, deadline_reserve_seconds)

    # Set PyTorch thread limits
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    base = resolve(output_dir) if output_dir else evaluator.base
    state_file = base / "bayes_state.json"
    lock_file = base / ".campaign.lock"  # B6: Shared campaign lock with sequential BO
    eval_log = base / "evaluations.jsonl"

    if start_time is None:
        start_time = time.time()

    # B6: Acquire exclusive directory lock if not already held by caller
    lock = None
    if not campaign_lock_acquired:
        lock = ExclusiveLock(lock_file)
        lock.acquire()

    try:
        # Startup remains separate from proposals and worker execution.
        state = _load_campaign(state_file, base, evaluator, predecessor_manifest_path, base_seed, resume_option_b=resume_option_b)
        use_cgp = state.get("option_b_resumed", False)
        sidecar_lookup = load_registration_sidecar() if use_cgp else {}
        if use_cgp:
            for rec in state["observations"][:20]:
                row = sidecar_lookup[rec["attempt_id"]]
                if rec["valid"] != row["valid"] or rec.get("failure_class") != row.get("failure_class"):
                    raise ValueError("Sidecar/observation registration label mismatch")
        reconciled_infra = _recover_campaign(base, state_file, eval_log, state, evaluator, base_seed)

        records = state["observations"]
        reserved = state["reserved_attempts"]
        completed = state["completed_attempts"]
        if _stop_before_launch(state, state_file, reconciled_infra, max_total_evaluations):
            return records

        message("BATCH-BO", f"Starting batch BO: {len(records)} existing observations ({sum(r.get('valid', False) for r in records)} valid).")
        message("BATCH-BO", f"Ceiling: {max_total_evaluations} total evaluations. Planned: {iterations} waves of batch_size={batch_size}.")

        state_lock = threading.Lock()

        for wave_step in range(1, iterations + 1):
            next_attempt = validate_campaign_ledger(state, evaluator.identity, base_seed, validate_observation_record)
            if len(records) != next_attempt - 1:
                raise ValueError("Unresolved charged attempts prohibit a new wave")
            # Check remaining budget
            if len(records) >= max_total_evaluations:
                state["status"] = "budget_complete"
                save_json(state_file, state)
                message("BATCH-BO", f"Hit max_total_evaluations ceiling ({max_total_evaluations}). Budget complete.")
                break

            current_q = min(batch_size, max_total_evaluations - len(records))
            if current_q != IMMUTABLE_BATCH_SIZE:
                raise ValueError(
                    f"Partial waves are not authorized by campaign contract: current_q={current_q} != {IMMUTABLE_BATCH_SIZE}"
                )

            # B7: Pre-acquisition deadline check
            rem_sec = get_remaining_allocation_seconds(start_time, total_time_limit_seconds)
            if rem_sec < deadline_reserve_seconds:
                state["status"] = "deadline_stop"
                save_json(state_file, state)
                message(
                    "BATCH-BO",
                    f"Remaining wall time ({rem_sec:.1f}s) < {deadline_reserve_seconds}s reserve. "
                    f"Clean deadline stop before wave {wave_step}.",
                )
                break

            # Persistent wave index derived from completed new evaluations (B1, B3, R3)
            completed_new = max(0, len(records) - 12)
            wave_idx = (completed_new // IMMUTABLE_BATCH_SIZE) + 1
            wave_seed = base_seed + wave_idx
            # Preserve historical attempts: new waves start at 13, 17, 21, 25, 29 (R3)
            expected_wave_start = 12 + ((wave_idx - 1) * IMMUTABLE_BATCH_SIZE) + 1
            wave_start_attempt = next_attempt
            if wave_start_attempt != expected_wave_start or wave_idx > IMMUTABLE_MAX_WAVES:
                raise ValueError("Charged ledger and wave start disagree")

            message("BATCH-BO", f"=== WAVE {wave_idx} (Step {wave_step}/{iterations}, Attempts {wave_start_attempt} to {wave_start_attempt + current_q - 1}) ===")

            # B2: Propose batch with strict constraint assertions
            batch_pts, diag = propose_q_batch(
                records,
                q=current_q,
                seed=wave_seed,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                use_constrained_gp=use_cgp,
                sidecar_lookup=sidecar_lookup,
            )
            message(
                "BATCH-BO",
                f"Proposed batch of {current_q} points with qLogEI={diag['raw_acq_value']:.6g} (repaired={diag['repaired_candidates']})",
            )
            # B2: Durably persist proposal and repair diagnostics
            state.setdefault("proposals", []).append(diag)
            state.setdefault("wave_diagnostics", []).append({
                "wave_idx": wave_idx,
                "wave_seed": wave_seed,
                "diagnostics": diag,
                "timestamp": time.time(),
            })
            save_json(state_file, state)

            # B7: Re-check deadline immediately before launching parallel workers
            rem_sec_prelaunch = get_remaining_allocation_seconds(start_time, total_time_limit_seconds)
            if rem_sec_prelaunch < deadline_reserve_seconds:
                state["status"] = "deadline_stop"
                save_json(state_file, state)
                message(
                    "BATCH-BO",
                    f"Remaining wall time ({rem_sec_prelaunch:.1f}s) < {deadline_reserve_seconds}s reserve immediately before launch. "
                    f"Clean deadline stop.",
                )
                break

            # B3: Reserve attempts in state before execution
            wave_reservations = []
            for i in range(current_q):
                att_id = wave_start_attempt + i
                pt = batch_pts[i]
                pub_dir = base / "evaluations" / f"attempt_{att_id:04d}"
                res = {
                    "attempt_id": att_id,
                    "proposal_idx": i,
                    "wave_idx": wave_idx,
                    "evaluator_identity": evaluator.identity,
                    "x": pt.tolist(),
                    "publication_dir": str(pub_dir),
                    "result_file": str(pub_dir / "result.json"),
                    "status": "reserved",
                    "created_at": time.time(),
                }
                reserved.append(res)
                wave_reservations.append((att_id, pt, res, i))

            state["reserved_attempts"] = reserved
            state["charged_attempts"] = wave_start_attempt + current_q - 1
            state["waves_reserved"] = wave_idx
            validate_campaign_ledger(state, evaluator.identity, base_seed, validate_observation_record)
            save_json(state_file, state)

            infra_encountered = _execute_wave(
                evaluator, wave_reservations, state, state_file, eval_log, base, base_seed, state_lock,
            )

            valid_count = sum(r.get("valid", False) for r in records)
            best_valid = min([r for r in records if r.get("valid", False)], key=lambda r: r["objective_mm"], default=None)
            best_att = best_valid.get("attempt_id", "initial") if best_valid else "N/A"
            best_str = f"{best_valid['objective_mm']:.4f} mm (Attempt {best_att})" if best_valid else "None"
            message("BATCH-BO", f"Wave {wave_idx} complete! Valid total: {valid_count}/{len(records)}; Current Best: {best_str}")

            # B4: Infrastructure failure stopping rule
            if infra_encountered:
                state["status"] = "stopped_infrastructure_failure"
                save_json(state_file, state)
                message("BATCH-BO", "Infrastructure failure encountered during wave. Stopping further launches.")
                break

            # B4: Proposal-order consecutive failures stopping rule check
            streak, hit_streak_halt = reconstruct_proposal_order_streak(
                records, max_failures=MAX_CONSECUTIVE_FAILURES, epoch_start_attempt=get_epoch_start_attempt(state),
            )
            state["proposal_order_failure_streak"] = streak
            if hit_streak_halt:
                state["status"] = "consecutive_failures"
            save_json(state_file, state)

            if hit_streak_halt:
                message("BATCH-BO", "6 consecutive failures in proposal order encountered. Latched halt.")
                break

        # R3: Terminal publication check when all evaluations complete normally
        terminal_statuses = (
            "consecutive_failures",
            "stopped_consecutive_failures",
            "stopped_proposal_failures",
            "stopped_infrastructure_failure",
            "deadline_stop",
        )
        if len(records) >= max_total_evaluations and state.get("status") not in terminal_statuses:
            state["status"] = "budget_complete"
            save_json(state_file, state)
            message("BATCH-BO", f"Campaign completed all {max_total_evaluations} evaluations. Status set to 'budget_complete'.")

        return records

    finally:
        if lock:
            lock.release()
