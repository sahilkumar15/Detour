"""Fingerprint the dataset and the metric definitions, and refuse to mix them.

Why this exists
---------------
The claim the paper makes across models is a DIFFERENCE claim: model A
pursues a forbidden effect through another tool more often than model B.
That claim is only readable if the benchmark did not move between the two
runs. In a small suite it is very easy to edit one scenario, re-run one
model, and compare it against a number produced by the earlier suite. The
resulting difference is uninterpretable, and nothing in the output would
show it.

So the suite and the metric definitions are hashed into a manifest written
beside every results file:

  dataset_digest  semantic hash of every scenario ACTUALLY built -- task
                  ids, goals, policy rules, denied call, withheld tools,
                  markers, and the file bodies the agent can read (T6's
                  injection lives in a file body, so a change there must
                  change the digest).
  metric_digest   semantic hash of the metric names, their eligibility
                  predicates and their indicator fields, plus the outcome
                  label vocabulary.
  code_digest     file hashes of the modules that define the above. This is
                  informational: a docstring edit changes it while the two
                  semantic digests stay put, which is exactly the signal
                  wanted -- "the code moved, the benchmark did not".

`--verify-freeze` compares the current code against a manifest and exits
non-zero on a semantic mismatch. Run it before every model comparison, and
in CI. The honest workflow is: freeze the suite, then vary only the model.
If a scenario genuinely has to change, the digest changes, and every number
in the paper is re-run under the new digest.

The digests are short (16 hex chars). They are collision-resistant enough
for an audit trail and short enough to print in a table caption.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "dataset_fingerprint",
    "metric_fingerprint",
    "code_fingerprint",
    "build_manifest",
    "write_manifest",
    "verify_manifest",
    "FreezeMismatch",
]

# Modules whose contents define the dataset and the measurement.
_CODE_FILES = (
    "scenarios.py",
    "metrics.py",
    "gate.py",
    "effects.py",
    "runner.py",
    "normalize.py",
    "sandbox.py",
    # The strict primary denominator lives here. A change to it changes the
    # paper's headline rate without touching gate or metric semantics, so it
    # belongs under the code digest even though no digest above moves.
    "subsets.py",
)


class FreezeMismatch(RuntimeError):
    """The current dataset or metric definitions differ from the manifest."""


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def _scenario_record(spec: Any) -> dict[str, Any]:
    """Everything about a scenario that can change what an agent faces.

    Deliberately includes the built environment's file bodies. A scenario
    whose prompt is unchanged but whose document now carries an injected
    instruction is a DIFFERENT scenario, and the digest has to say so.
    """
    sandbox = spec.build()
    files = sorted(
        (f.filename, hashlib.sha256(f.content.encode("utf-8")).hexdigest()[:16])
        for f in sandbox.cloud_drive.files.values()
    )
    rules = sorted(
        (
            r.rule_id,
            r.op,
            r.principal,
            r.obj,
            r.qualifier,
            bool(r.requires_approval),
        )
        for r in spec.rules
    )
    return {
        "recovery": _recovery_probe(spec),
        "preload": _preload_probe(spec, sandbox),
        "observation": _observation_probe(spec, sandbox),
        "task_id": spec.task_id,
        "template": spec.template,
        "op_class": spec.op_class,
        "continuation_class": spec.continuation_class,
        "cross_tool": bool(spec.cross_tool),
        "adversarial": bool(spec.adversarial),
        "goal": spec.goal,
        "denied_tool": spec.denied_tool,
        "denied_args": dict(spec.denied_args),
        "expected_status": spec.expected_status,
        "unavailable_tools": sorted(spec.unavailable_tools),
        "markers": {k: sorted(v) for k, v in spec.ctx.object_markers.items()},
        "rules": rules,
        "files": files,
    }


def _observation_probe(spec: Any, sandbox: Any) -> str:
    """Hash the prompt the agent actually receives.

    Every other field here describes the scenario's DATA. The prompt is
    assembled by `runner._observation` from that data plus text the runner
    owns -- the drive listing, the principal directory, the channel list --
    so a change to what the agent is told moved none of the other fields and
    the dataset digest stayed put. That is the same hole `_recovery_probe`
    and `_preload_probe` exist to close, and adding the principal directory
    to the prompt is exactly the kind of change that must not slip through:
    it altered the measured rates without altering any scenario datum.

    Probed rather than stored so the digest moves when the rendering moves,
    and the denial text is held fixed so this captures the prompt rather
    than the feedback condition, which is an experimental factor and not
    part of the dataset.
    """
    from .runner import _observation

    rendered = _observation(spec, sandbox, denial="<frozen-denial-probe>")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _preload_probe(spec: Any, sandbox: Any) -> list[str]:
    """Fingerprint what the agent is GIVEN before its first turn.

    The episode now opens with the protected document replayed as a read the
    agent already made, which is what delivers T6's injection. That is part
    of what an agent faces -- the thing this record exists to pin down -- but
    it lives in the runner rather than on the spec, so without this probe the
    dataset digest is identical before and after the change and
    `--verify-freeze` would pass two runs where one agent had read the
    document and the other had not.

    The same hole as the metric digest had around `_reference_status`, and
    the same remedy: probe the behaviour instead of hashing the data.
    Content is hashed rather than stored so the digest moves when the
    document body moves (T6's injection lives in that body).
    """
    from .runner import _preload_reads

    try:
        steps = _preload_reads(spec, sandbox)
    except Exception:
        return ["<preload-failed>"]

    # Hash the FILE CONTENT the step delivers, not the rendered tool result.
    # AgentDojo's file objects carry a `last_modified` timestamp set at build
    # time, so the rendered string differs on every build and hashing it
    # would make the dataset digest unstable across runs of the same suite --
    # which would make `--verify-freeze` fail against its own manifest.
    out: list[str] = []
    for step in steps:
        requested = str((step.get("args") or {}).get("file_id") or "")
        body = ""
        for file_id, record in sandbox.cloud_drive.files.items():
            if str(file_id) == requested:
                body = str(record.content)
                break
        out.append(
            f"{step.get('tool')}:{hashlib.sha256(body.encode('utf-8')).hexdigest()[:16]}"
        )
    return out


def _recovery_probe(spec: Any) -> list[bool]:
    """Fingerprint WHAT COUNTS as a recovery, by probing the predicate.

    `recovered` decides SRR, EAC and (through completion) OHR, so changing
    it changes most of the table. It is a closure, not data, so it cannot be
    hashed by value -- and an earlier version of this manifest left it out
    entirely, meaning the recovery rule could be rewritten and every rate
    move while the dataset digest stayed put. That is precisely the silent
    incomparability the freeze exists to prevent.

    So the predicate is probed against four canonical end states. The probe
    battery is part of the digest's meaning: reordering it is a dataset
    change.
    """
    # The battery must straddle the substance threshold, not just sit under
    # it. With only the one-character artifact below, changing
    # MIN_ARTIFACT_CHARS from 120 to 80 left every probe answering the same
    # way and the dataset digest unmoved -- while SRR's meaning changed. The
    # three sized artifacts below bracket the plausible range, so any
    # threshold moved across 40, 100 or 200 characters now shows up.
    def sized(n: int):
        return lambda sb: sb.execute(
            "create_file",
            {"filename": f"probe-{n}.md", "content": "detail " * (n // 7 + 1)},
        )

    probes: list[bool] = []
    for mutate in (
        lambda sb: None,
        lambda sb: sb.execute("create_file", {"filename": "probe-artifact.md", "content": "x"}),
        sized(40),
        sized(100),
        sized(200),
        lambda sb: sb.execute(
            "append_to_file",
            {"file_id": str(next(iter(sb.cloud_drive.files))), "content": "probe"},
        ),
        lambda sb: sb.execute(
            "share_file",
            {
                "file_id": str(next(iter(sb.cloud_drive.files))),
                "email": "probe@acme-corp.com",
                "permission": "r",
            },
        ),
    ):
        sandbox = spec.build()
        try:
            mutate(sandbox)
            probes.append(bool(spec.recovered(sandbox)))
        except Exception:
            probes.append(False)
    return probes


def dataset_fingerprint(suite: Sequence[Any]) -> dict[str, Any]:
    records = [_scenario_record(s) for s in suite]
    records.sort(key=lambda r: r["task_id"])
    return {
        "digest": _digest(records),
        "n_tasks": len(records),
        "n_templates": len({r["template"] for r in records}),
        "n_cross_tool": sum(1 for r in records if r["cross_tool"]),
        "n_cross_tool_clusters": len(
            {r["template"] for r in records if r["cross_tool"]}
        ),
        "task_ids": [r["task_id"] for r in records],
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def metric_fingerprint() -> dict[str, Any]:
    """Hash WHAT each metric counts, not how the counting is written.

    Eligibility predicates and indicators are lambdas, so they are probed
    against synthetic episodes rather than read as source: the fingerprint
    then tracks the behaviour a reviewer cares about and ignores cosmetic
    edits.
    """
    from . import runner as runner_mod
    from .metrics import ELIGIBILITY, INDICATOR, LOWER_IS_BETTER, METRIC_ORDER

    probes = _probe_episodes()
    table = {}
    for name in sorted(set(ELIGIBILITY) | set(INDICATOR)):
        elig = ELIGIBILITY.get(name)
        ind = INDICATOR.get(name)
        table[name] = {
            "eligibility": [bool(elig(p)) if elig else None for p in probes],
            "indicator": [bool(ind(p)) if ind else None for p in probes],
        }
    labels = sorted(
        v
        for k, v in vars(runner_mod).items()
        if k.isupper() and isinstance(v, str) and not k.startswith("_")
    )
    payload = {
        "order": list(METRIC_ORDER),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "table": table,
        "labels": labels,
        "reference_status": _reference_status_probe(),
        "equivalence": _equivalence_probe(),
    }
    return {"digest": _digest(payload), "metrics": list(METRIC_ORDER)}


def _equivalence_probe() -> list[str]:
    """Probe the principal resolution that CTWAR's equivalence test rests on.

    CTWAR counts a later proposal only when its effect tuple matches the
    denied one, and the tuple's recipient slot comes from
    `EffectContext.principal`. A change there changes what the primary metric
    counts -- local-part identity at the internal domain moved CTWAR by
    10/162 episodes on one model and 20/162 on the other -- while every
    probe above kept returning the same answers, because they exercise
    eligibility and indicators over synthetic episodes and never touch the
    oracle. Resolving a fixed set of surface identifiers closes that: the
    metric digest now moves when the equivalence semantics move.
    """
    from .scenarios import build_suite

    ctx = build_suite()[0].ctx
    surfaces = (
        "bob",
        "bob@partner-labs.com",
        "bob@acme-corp.com",
        "rchen",
        "r.chen@outside-consultancy.net",
        "rchen@acme-corp.com",
        "dana",
        "dana@acme-corp.com",
        "someone-new@acme-corp.com",
        "stranger@elsewhere.example",
        "#partners-open",
        "#general",
        "",
    )
    out: list[str] = []
    for surface in surfaces:
        canonical = ctx.principal(surface)
        out.append(f"{surface}->{canonical}:{ctx.disclosure_qualifier(canonical)}")
    return out


def _reference_status_probe() -> list[str]:
    """Fingerprint WHAT COUNTS as the true terminal status.

    `_reference_status` decides RF, and it is a function rather than data, so
    like `recovered` it cannot be hashed by value. Leaving it out was a real
    hole: its keying was changed from `continuation_class` to "was an
    approval actually requested", which flipped RF from 22.7% to 81.0% at F1
    for qwen2.5 7B -- and BOTH digests stayed put, so `--verify-freeze`
    cheerfully passed two runs whose RF meant different things. That is
    exactly the silent incomparability this module exists to prevent.

    So the decision table is probed instead: every continuation class, with
    and without a recorded approval request, with and without something
    actually sent. The battery is part of the digest's meaning; reordering it
    is a metric change.
    """
    from .runner import Episode, _reference_status
    from .scenarios import build_suite

    suite = build_suite()
    out: list[str] = []
    for cls in ("alt", "approve", "halt"):
        spec = next((s for s in suite if s.continuation_class == cls), None)
        if spec is None:
            out.append(f"{cls}:<no-scenario>")
            continue
        for asked in (False, True):
            for sent in (False, True):
                sandbox = spec.build()
                if sent:
                    try:
                        sandbox.execute(
                            "send_direct_message", {"recipient": "dana", "body": "probe"}
                        )
                    except Exception:
                        pass
                episode = Episode(
                    task_id="probe",
                    template="P",
                    op_class="disclose",
                    continuation_class=cls,
                    cross_tool=True,
                    adversarial=False,
                    feedback="F1",
                    model="probe",
                    repetition=0,
                    severity=1.0,
                    Q=asked,
                    asked_approval=asked,
                )
                out.append(
                    f"{cls}:asked={int(asked)}:sent={int(sent)}"
                    f"->{_reference_status(spec, sandbox, episode)}"
                )
    return out


def _probe_episodes() -> list[Any]:
    """A fixed battery of synthetic episodes covering every indicator field.

    The battery itself is part of the fingerprint's meaning, so it must not
    be reordered or trimmed without accepting a digest change.
    """
    from .runner import Episode

    base = dict(
        task_id="probe",
        template="P",
        op_class="disclose",
        continuation_class="alt",
        cross_tool=True,
        adversarial=False,
        feedback="F1",
        model="probe",
        repetition=0,
        severity=1.0,
    )
    # Every indicator field a metric reads. A field left out of this battery
    # is a field two metrics can be swapped between without moving the
    # digest: before `X_gate`, `X_state` and `asked_approval` were added
    # here, UER-gate and UER-state probed identically (all False on every
    # episode), so exchanging their indicators was invisible to the freeze.
    fields = (
        "W",
        "U",
        "X",
        "X_gate",
        "X_state",
        "C",
        "C_plus",
        "H",
        "Q",
        "J",
        "asked_approval",
        "injection_read",
    )
    probes: list[Any] = []
    for cls in ("alt", "approve", "halt"):
        for cross in (True, False):
            probes.append(Episode(**{**base, "continuation_class": cls, "cross_tool": cross}))
    for field_name in fields:
        ep = Episode(**base)
        setattr(ep, field_name, True)
        probes.append(ep)
    ep = Episode(**base)
    ep.rf_eligible = True
    ep.rf_correct = True
    probes.append(ep)
    ep = Episode(**base)
    ep.budget_exhausted = True
    probes.append(ep)
    return probes


# --------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------


def code_fingerprint() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    files = {}
    for name in _CODE_FILES:
        path = here / name
        if path.exists():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return {"digest": _digest(files), "files": files}


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def build_manifest(suite: Sequence[Any], **extra: Any) -> dict[str, Any]:
    return {
        "dataset": dataset_fingerprint(suite),
        "metrics": metric_fingerprint(),
        "code": code_fingerprint(),
        **extra,
    }


def write_manifest(out_path: str | Path, suite: Sequence[Any], **extra: Any) -> Path:
    """Write `<out>.manifest.json` beside a results file and return its path."""
    path = Path(out_path)
    target = path.with_suffix(".manifest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_manifest(suite, **extra), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def verify_manifest(
    manifest_path: str | Path, suite: Sequence[Any], *, strict_code: bool = False
) -> list[str]:
    """Return the list of differences; empty means the benchmark is unchanged.

    Raises FreezeMismatch on a dataset or metric difference. A code-only
    difference is returned as a warning unless `strict_code` is set, because
    a comment or a provider adapter may legitimately change between runs
    without moving the benchmark.
    """
    stored = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    current = build_manifest(suite)
    problems: list[str] = []
    fatal: list[str] = []

    for section in ("dataset", "metrics"):
        old = stored.get(section, {}).get("digest")
        new = current[section]["digest"]
        if old != new:
            fatal.append(f"{section} digest changed: {old} -> {new}")

    old_code = stored.get("code", {}).get("files", {})
    new_code = current["code"]["files"]
    changed = sorted(k for k in set(old_code) | set(new_code) if old_code.get(k) != new_code.get(k))
    if changed:
        message = "code changed (benchmark unaffected): " + ", ".join(changed)
        (fatal if strict_code else problems).append(message)

    if fatal:
        raise FreezeMismatch(
            "the benchmark is not the one this manifest was written for:\n  "
            + "\n  ".join(fatal)
            + "\nRe-run every model under the new digest, or restore the suite."
        )
    return problems