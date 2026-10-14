"""Pure integrity checks for the fixed 12-seed / five four-attempt-wave campaign.

No solver calls, checkpoint edits, or budget renewal. Legacy migration and a
bound batch restart deliberately have different provenance requirements.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

DOMAIN = {"gtop": [0.001, 0.01], "gbot": [1e-5, 0.0003],
          "ortho_top": [-1.0, 1.0], "ortho_bottom": [-1.0, 1.0]}
SCORING = {"metric": "rms_symmetric_mm",
           "formula": "sqrt(0.5 * (rms_forward^2 + rms_reverse_weighted^2))",
           "target_scan": "calibration/data/processed/run6_oriented.npz"}
# One source for the fixed campaign limits, shared by controller and CLI.
IMMUTABLE_MAX_TOTAL_BUDGET = 32
IMMUTABLE_MAX_NEW_EVALUATIONS = 20
IMMUTABLE_BATCH_SIZE = 4
IMMUTABLE_MAX_WAVES = 5
MAX_CONSECUTIVE_FAILURES = 6
DEFAULT_DEADLINE_RESERVE_SECONDS = 3900.0

METADATA = {
    "base_seed": 2026,
    "max_total_evaluations": IMMUTABLE_MAX_TOTAL_BUDGET,
    "batch_size": IMMUTABLE_BATCH_SIZE,
    "max_waves": IMMUTABLE_MAX_WAVES,
}


def positive_id(value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Attempt ID must be a positive integer, got {value!r}")
    return value


def unique_records(records, validate):
    if not isinstance(records, list):
        raise ValueError("Observation ledger must be a list")
    indexed = {}
    for rec in records:
        validate(rec)
        att = positive_id(rec["attempt_id"])
        if att in indexed:
            raise ValueError(f"Duplicate durable attempt ID {att}")
        indexed[att] = rec
    return indexed


def record_identity(rec, validate):
    """Canonical content, excluding timestamps, paths and controller annotations."""
    validate(rec)
    identity = rec.get("identity")
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ValueError("Malformed publication evaluator identity")
    core = [rec["attempt_id"], rec["normalized_parameters"], identity, rec["valid"]]
    core += ([rec["objective_mm"]] if rec["valid"] else [rec["failure_class"], rec["error"]])
    return core


def same_record(left, right, validate):
    a, b = record_identity(left, validate), record_identity(right, validate)
    return (a[0] == b[0] and a[2:] == b[2:] and
            np.allclose(a[1], b[1], rtol=0.0, atol=1e-12))


def source_map(rows):
    keys = ("attempt_id", "tuple_id", "request_json", "state_file", "state_sha256",
            "mesh_file", "mesh_sha256")
    mapped = []
    for row in rows:
        positive_id(row.get("attempt_id"))
        for key in keys[1:]:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing predecessor source mapping {key}")
        mapped.append({key: row[key] for key in keys})
    return mapped


def campaign_kind(state):
    """A damaged batch checkpoint must never be interpreted as legacy history."""
    if not isinstance(state, dict):
        raise ValueError("bayes_state.json must be a JSON object")
    schema = state.get("schema_version")
    kind = state.get("campaign_type")
    if type(schema) is int and schema == 2 and kind == "batch_qlogei":
        return "batch"
    batch_fields = {"migration", "campaign_metadata", "evaluator_identity",
                    "predecessor_manifest_digest", "predecessor_sources",
                    "charged_attempts", "waves_reserved", "historical_compatibility_digest",
                    "historical_publications", "budget_new", "batch_size"}
    if (type(schema) is int and schema == 1 and kind is None
            and not batch_fields.intersection(state) and state.get("pending") is None):
        return "legacy"
    raise ValueError("Unknown or damaged campaign discriminator/schema; migration denied")


def validate_history_binding(state, identity, identity_inputs, state_bytes, manifest_path):
    """Explicit frozen import contract; history and current execution keep separate IDs.

    Source/environment differences are enumerated, not silently erased from the
    evaluator identity. This function reads evidence only and never writes it.
    """
    kind = campaign_kind(state)
    records = state.get("observations", [])
    if not isinstance(records, list) or len(records) < 12 or not isinstance(records[11], dict):
        raise ValueError("Campaign origin requires twelve canonical predecessor observations")
    historical_identity = records[11].get("identity")
    if historical_identity == identity and not isinstance(identity_inputs, dict):
        return None  # No cross-identity import; ordinary provenance checks still apply.
    root = Path(__file__).resolve().parents[2]
    policy_path = root / "run/optimization_review/historical_import_v16.json"
    raw = policy_path.read_bytes()
    policy = json.loads(raw)
    policy_digest = hashlib.sha256(raw).hexdigest()
    if (type(policy.get("schema_version")) is not int or policy["schema_version"] != 1
            or historical_identity != policy.get("historical_identity")):
        raise ValueError("Historical identity is not authorized by the import contract")
    if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != policy["manifest_sha256"]:
        raise ValueError("Historical import manifest changed")
    history_raw = (root / policy["checkpoint_path"]).read_bytes()
    if hashlib.sha256(history_raw).hexdigest() != policy["checkpoint_sha256"]:
        raise ValueError("Preserved historical checkpoint changed")
    if kind == "legacy" and hashlib.sha256(state_bytes).hexdigest() != policy["checkpoint_sha256"]:
        raise ValueError("Legacy input is not the authorized byte-preserved checkpoint")
    authorized_prior_digest = "7a2d66011bbf062986c120f94500de2a280996dc4c3ee87c89b9689f093d930d"
    trans = state.get("historical_import_transition")
    if trans:
        if (trans.get("prior_digest") != authorized_prior_digest
                or trans.get("authorized_boundary") != 21
                or trans.get("transition_reason") != "option_b_constrained_gp"):
            raise ValueError("Unauthorized historical import transition schema")
        if (trans.get("new_digest") != policy_digest
                or state.get("historical_compatibility_digest") != policy_digest):
            raise ValueError("Historical import transition digest mismatch")
        if state.get("historical_publications") != policy["publications"]:
            raise ValueError("Historical publications changed during transition")
    elif kind == "batch":
        if (state.get("historical_compatibility_digest") != policy_digest
                or state.get("historical_publications") != policy["publications"]):
            raise ValueError("Missing or changed historical import binding on restart")
    historical_raw = (root / policy["identity_path"]).read_bytes()
    if hashlib.sha256(historical_raw).hexdigest() != policy["identity_file_sha256"]:
        raise ValueError("Historical evaluator identity evidence changed")
    historical = json.loads(historical_raw)
    if hashlib.sha256(json.dumps(historical, sort_keys=True).encode()).hexdigest() != historical_identity:
        raise ValueError("Historical evaluator identity does not match its payload")
    if not isinstance(identity_inputs, dict):
        raise ValueError("Real current evaluator identity inputs are required for historical import")
    if hashlib.sha256(json.dumps(identity_inputs, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("Current evaluator identity does not match its payload")
    for key in ("executable", "targets"):
        if identity_inputs.get(key) != historical.get(key):
            raise ValueError(f"Scientific identity changed: {key}")
    for key in ("python_sources", "versions"):
        if identity_inputs.get(key) != policy["current_" + key]:
            raise ValueError(f"Unreviewed current identity inputs: {key}")
    environment = identity_inputs.get("environment")
    if (not isinstance(environment, dict) or set(environment) != set(policy["allowed_environment"])
            or any(value not in policy["allowed_environment"][key] for key, value in environment.items())):
        raise ValueError("Unreviewed current evaluator environment")
    old_config = json.loads(json.dumps(historical["config"]))
    new_config = json.loads(json.dumps(identity_inputs["config"]))
    if old_config["optimization"].pop("max_evaluations") != 31 or new_config["optimization"].pop("max_evaluations") != 32:
        raise ValueError("Historical import requires the declared 31-to-32 budget amendment")
    if (old_config.get("registration", {}).pop("icp_iterations", None) != 30
            or new_config.get("registration", {}).pop("icp_iterations", None) != 60):
        raise ValueError("Historical import requires the declared 30-to-60 registration iteration amendment")
    for cfg in (old_config, new_config):
        for key in ("output_dir", "report_dir"):
            cfg.pop(key, None)  # Artifact locations, not the scientific recipe.
    if new_config != old_config:
        raise ValueError("Historical import scientific configuration changed")
    original = json.loads(history_raw)
    if state.get("reserved_attempts", [])[:12] != original["reserved_attempts"]:
        raise ValueError("Historical reservation identity/path/metadata changed")
    if any(r.get("attempt_id", 0) <= 12 for r in state.get("reserved_attempts", [])[12:]):
        raise ValueError("Duplicate historical reservation")
    return {"digest": policy_digest, "identity": historical_identity,
            "publications": policy["publications"]}


def validate_option_b_transition(state, identity):
    """Only the approved single transition may reset the stopping epoch."""
    keys = ("evaluator_lineage", "stopping_epoch", "historical_import_transition", "option_b_resumed")
    if not any(key in state for key in keys):
        return False
    if not set(range(1, 21)).issubset({r.get("attempt_id") for r in state.get("observations", [])}):
        raise ValueError("Option B transition requires all twenty origin observations")
    expected_lineage = {
        "schema_version": 2, "transition_reason": "option_b_constrained_gp",
        "transitions": [{"boundary_attempt": 21,
                         "prior_evaluator_identity": "630b8877d0c2d7dd725846a44ff8a15010d146bf487a2ea9446f1756f65edfe5",
                         "new_evaluator_identity": identity}],
    }
    if (state.get("evaluator_lineage") != expected_lineage
            or state.get("stopping_epoch") != {"epoch_start_attempt": 21,
                "protocol": "option_b_constrained_gp", "transition_applied": True}
            or state.get("option_b_resumed") is not True
            or state.get("evaluator_identity") != identity):
        raise ValueError("Malformed or unauthorized Option B transition")
    trans = state.get("historical_import_transition")
    if (not isinstance(trans, dict) or set(trans) != {"prior_digest", "new_digest", "authorized_boundary", "transition_reason"}
            or trans.get("prior_digest") != "7a2d66011bbf062986c120f94500de2a280996dc4c3ee87c89b9689f093d930d"
            or trans.get("new_digest") != state.get("historical_compatibility_digest")
            or not isinstance(trans.get("new_digest"), str)
            or trans.get("authorized_boundary") != 21
            or trans.get("transition_reason") != "option_b_constrained_gp"):
        raise ValueError("Malformed historical import transition")
    return True


def get_lineage_evaluator_identity(lineage, attempt_id, default_identity):
    """Resolve expected evaluator identity for attempt_id using lineage."""
    if not lineage:
        return default_identity
    transitions = []
    if isinstance(lineage, dict):
        transitions = lineage.get("transitions", [])
    elif isinstance(lineage, list):
        transitions = lineage
    if not transitions:
        return default_identity
    last_trans = transitions[-1]
    boundary = last_trans.get("boundary_attempt", last_trans.get("transition_attempt", 21))
    if attempt_id < boundary:
        return last_trans.get("prior_evaluator_identity")
    else:
        return last_trans.get("new_evaluator_identity", default_identity)


def validate_predecessor(state, expected_identity, config, manifest_path, validate, history=None):
    """Validate without mutating either the caller's state or the manifest."""
    if not isinstance(state, dict):
        raise ValueError("bayes_state.json must be a JSON object")
    records = state.get("observations")
    indexed = unique_records(records, validate)
    if len(records) < 12 or [r["attempt_id"] for r in records[:12]] != list(range(1, 13)):
        raise ValueError("Campaign origin requires the twelve canonical predecessor attempts 1..12")
    if not manifest_path:
        raise ValueError("predecessor_manifest_path is required")
    raw = Path(manifest_path).read_bytes()
    manifest_digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
        raise ValueError("Predecessor manifest must have schema_version=2")
    if manifest.get("domain_definition", {}).get("authorized_domain") != DOMAIN:
        raise ValueError("Predecessor manifest domain mismatch")
    scoring = manifest.get("scoring_compatibility", {})
    if any(scoring.get(k) != v for k, v in SCORING.items()):
        raise ValueError("Predecessor manifest scoring mismatch")
    rows = manifest.get("observations")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Predecessor manifest must have exactly twelve observations")
    sources = source_map(rows)
    batch = campaign_kind(state) == "batch"
    historical_identity = history["identity"] if history else expected_identity
    if batch:
        if type(state.get("schema_version")) is not int or state["schema_version"] != 2:
            raise ValueError("Batch restart requires schema_version=2")
        lineage = state.get("evaluator_lineage", [])
        if lineage:
            expected_current = get_lineage_evaluator_identity(lineage, 21, expected_identity)
            if not expected_identity or state.get("evaluator_identity") != expected_current:
                raise ValueError("Missing or changed batch evaluator_identity")
        else:
            if not expected_identity or state.get("evaluator_identity") != expected_identity:
                raise ValueError("Missing or changed batch evaluator_identity")
        if state.get("predecessor_manifest_digest") != manifest_digest:
            raise ValueError("Missing or changed predecessor_manifest_digest")
        if state.get("domain_definition") != DOMAIN or state.get("scoring_compatibility") != SCORING:
            raise ValueError("Batch restart compatibility mismatch")
        if state.get("predecessor_sources") != sources:
            raise ValueError("Batch restart historical source mappings mismatch")
    else:
        if len(records) != 12 or state.get("status") not in ("budget_complete", "in_progress"):
            raise ValueError("Only the terminal twelve-point legacy predecessor may migrate")
        if state.get("evaluator_identity") not in (None, expected_identity):
            raise ValueError("Legacy evaluator identity mismatch")
    for i, (row, rec) in enumerate(zip(rows, records[:12]), 1):
        if positive_id(row.get("attempt_id")) != i:
            raise ValueError("Predecessor manifest attempt_id mismatch")
        # Only legacy Attempt 12 omitted tuple_id; match all remaining provenance
        # before admitting this narrow metadata migration. Assignment happens later.
        if rec.get("tuple_id") != row["tuple_id"]:
            if batch or i != 12 or rec.get("tuple_id") is not None or row["tuple_id"] != "Attempt_12":
                raise ValueError(f"Predecessor tuple_id mismatch at attempt {i}")
        if not np.allclose(rec["normalized_parameters"], row.get("normalized_parameters"), rtol=0, atol=1e-10):
            raise ValueError(f"Predecessor coordinate mismatch at attempt {i}")
        if type(row.get("gp_eligible")) is not bool or rec["valid"] != row["gp_eligible"]:
            raise ValueError(f"Predecessor validity mismatch at attempt {i}")
        physical = rec.get("parameters")
        if not isinstance(physical, dict) or set(physical) != set(DOMAIN):
            raise ValueError(f"Missing predecessor physical parameters at attempt {i}")
        for key in DOMAIN:
            if not np.isclose(physical[key], row["parameters"][key], rtol=1e-10, atol=1e-14):
                raise ValueError(f"Predecessor physical parameters mismatch at attempt {i}")
        if i < 12:
            for key in ("state_file", "mesh_file"):
                if rec.get(key) != row[key]:
                    raise ValueError(f"Predecessor source mapping mismatch for {key} at attempt {i}")
        else:
            if rec.get("identity") != historical_identity:
                raise ValueError("Legacy Attempt 12 evaluator identity mismatch")
            if Path(rec.get("directory", "")).resolve() != Path(row["directory"]).resolve():
                raise ValueError("Attempt 12 source directory mismatch")
        if rec["valid"]:
            loss = row.get("rms_symmetric_mm")
            if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not np.isfinite(loss) or loss <= 0:
                raise ValueError("Malformed predecessor objective")
            if not np.isclose(rec["objective_mm"], loss, rtol=0, atol=1e-10):
                raise ValueError("Predecessor objective mismatch")
        else:
            reason = row.get("ineligibility_reason")
            if not isinstance(reason, str):
                raise ValueError("Missing predecessor failure mapping")
            if reason.startswith(("pose_near_bound", "registration_unconverged", "Registration failed")):
                failure = "registration"
            elif reason == "solver_failure":
                failure = "solver"
            elif reason == "solver_timeout":
                failure = "timeout"
            else:
                raise ValueError(f"Unknown predecessor failure mapping: {reason!r}")
            if rec["failure_class"] != failure:
                raise ValueError("Predecessor failure class mismatch")
    if config is not None:
        from calibration.ewcal.common import Parameters
        if {k: config["parameters"][k]["bounds"] for k in DOMAIN} != DOMAIN:
            raise ValueError("Caller domain differs from authorized domain")
        if config["registration"]["metric"] != "symmetric" or config["registration"]["reject_at_bound"] is not True:
            raise ValueError("Caller scoring differs from authorized scoring")
        if [c["target"] for c in config["cases"]] != [SCORING["target_scan"]]:
            raise ValueError("Caller target differs from authorized target")
        decoder = Parameters(config)
        for rec in records:
            if "parameters" in rec:
                expected = decoder.decode(rec["normalized_parameters"])
                if set(rec["parameters"]) != set(DOMAIN) or any(not np.isclose(rec["parameters"][k], expected[k], rtol=1e-8, atol=1e-14) for k in DOMAIN):
                    raise ValueError("Physical/normalized parameters mismatch")
    return manifest_digest, sources


def validate_campaign_ledger(state, identity, base_seed, validate):
    """Count unique reserved charges, not completion count or completion order."""
    if state.get("status") not in (
        "in_progress", "budget_complete", "consecutive_failures",
        "stopped_consecutive_failures", "stopped_proposal_failures",
        "stopped_infrastructure_failure", "deadline_stop",
    ):
        raise ValueError("Missing or unknown campaign status")
    streak = state.get("proposal_order_failure_streak", 0)
    if type(streak) is not int or streak < 0:
        raise ValueError("Proposal-order failure streak must be a nonnegative integer")
    validate_option_b_transition(state, identity)
    indexed = unique_records(state["observations"], validate)
    for key, value in {"max_evaluations": IMMUTABLE_MAX_TOTAL_BUDGET, "batch_size": IMMUTABLE_BATCH_SIZE,
                       "budget_new": IMMUTABLE_MAX_NEW_EVALUATIONS}.items():
        if type(state.get(key)) is not int or state[key] != value:
            raise ValueError(f"Immutable campaign metadata mismatch: {key}")
    if base_seed != 2026 or state.get("campaign_metadata") != METADATA:
        raise ValueError("Immutable campaign_metadata mismatch")
    complete = state.get("completed_attempts")
    if not isinstance(complete, list) or any(type(x) is not int or x <= 0 for x in complete) or len(set(complete)) != len(complete) or set(complete) != set(indexed):
        raise ValueError("Completed attempts do not match unique observations")
    reservations = state.get("reserved_attempts")
    if not isinstance(reservations, list):
        raise ValueError("Reservation ledger must be a list")
    reserved = {}
    for res in reservations:
        if not isinstance(res, dict):
            raise ValueError("Reservation must be a dict")
        att = positive_id(res.get("attempt_id"))
        if att in reserved or att > 32:
            raise ValueError("Duplicate or over-ceiling reserved attempt")
        validate({"attempt_id": att, "normalized_parameters": res.get("x"), "valid": False,
                  "failure_class": "infrastructure", "error": "reserved"})
        reserved[att] = res
        if att <= 12:
            # Historical charged seeds are bound to the manifest; not BO waves.
            if att not in indexed or not np.allclose(res.get("x"), indexed[att]["normalized_parameters"], rtol=0, atol=1e-10):
                raise ValueError("Historical reservation coordinate mismatch")
            continue
        lineage = state.get("evaluator_lineage")
        expected_att_identity = get_lineage_evaluator_identity(lineage, att, identity)
        if res.get("evaluator_identity") != expected_att_identity:
            raise ValueError("Reservation evaluator identity mismatch")
        wave, proposal = (att - 13) // 4 + 1, (att - 13) % 4
        if type(res.get("wave_idx")) is not int or res["wave_idx"] != wave or type(res.get("proposal_idx")) is not int or res["proposal_idx"] != proposal:
            raise ValueError("Reservation wave/proposal index mismatch")
        if att in indexed and res.get("status") not in ("completed", "completed_interrupted"):
            raise ValueError("Completed observation has unfinished reservation")
    new = sorted(att for att in reserved if att > 12)
    if new != list(range(13, 13 + len(new))) or len(new) % 4 or len(new) > 20:
        raise ValueError("Charged attempts must form contiguous complete four-attempt waves")
    if set(indexed) - set(range(1, 13)) - set(new):
        raise ValueError("Observation without a charged reservation")
    if type(state.get("charged_attempts")) is not int or state["charged_attempts"] != 12 + len(new):
        raise ValueError("Charged attempt count mismatch")
    if type(state.get("waves_reserved")) is not int or state["waves_reserved"] != len(new) // 4:
        raise ValueError("Cumulative wave count mismatch")
    # Only the last wave may have incomplete results at a crash boundary.
    if new and any(att not in indexed for att in new[:-4]):
        raise ValueError("Unresolved gap in an earlier wave")
    lineage = state.get("evaluator_lineage")
    for att in new:
        if att in indexed:
            expected_att_identity = get_lineage_evaluator_identity(lineage, att, identity)
            if indexed[att].get("identity") != expected_att_identity or not np.allclose(indexed[att]["normalized_parameters"], reserved[att]["x"], rtol=0, atol=1e-10):
                raise ValueError("Observation/reservation identity mismatch")
    return 13 + len(new)
