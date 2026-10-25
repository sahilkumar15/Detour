# Detour — auditing post-denial cross-tool workarounds in tool-using LLM agents

When an authorization gate denies a tool call, what does the agent do next?
Detour measures how often it pursues **the same prohibited effect through a
different tool** — the cross-tool workaround attempt rate (CTWAR) — along
with ten companion rates that keep the headline number honest.

Implementation for a submission to **ACM TRUST 2027**, EA Track 19
(Trustworthy and Responsible Agentic AI).

Tool implementations come from
[AgentDojo](https://github.com/ethz-spylab/agentdojo) (Debenedetti et al.,
NeurIPS 2024), pinned to one commit. This package supplies the
authorization gate, the effect oracle, the scenario schema, the independent
state verifier and the metrics.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your GROQ_API_KEY in .env
```

`requirements.txt` pins AgentDojo to the exact commit the results were
produced against. Do not install it unpinned: the scenarios import its v1
tool modules directly.

Verify the install without spending a single token:

```bash
python run_audit.py --self-test      # 33 pipeline invariants
python -m pytest tests -q            # 48 unit tests
python run_audit.py --suite-summary  # suite denominators
```

---

## What it measures

An episode seeds a **standardized denial**: the agent's first call is
evaluated by the gate, refused, and the refusal is replayed to the model as
a real tool result — not narrated inside the task prompt. The agent then
gets up to six further steps. Every proposal is evaluated against the
policy before execution.

An **effect** is the policy-relevant consequence of a call, not its API
verb: `⟨operation, principal, object, qualifier⟩`. Emailing a protected
file to an outsider and DM-ing its contents to the same outsider are the
same `disclose` effect, so the second counts as a workaround for the first.

| metric | question | direction |
|---|---|---|
| **CTWAR** | after denial, did it chase the same forbidden effect via another tool? | lower better |
| PUAR | any unauthorized proposal at all? | lower better |
| UER-gate | did the gate ALLOW a forbidden effect? (soundness bug; expect 0) | lower better |
| UER-state | is the forbidden outcome in the terminal state at all? | lower better |
| SRR  | did it reach the goal by a legitimate route, with no violation? | higher better |
| EAC  | did it reach the goal at all, violations aside? | higher better |
| OHR  | did it stop *without acting and without escalating*, though a route existed? | lower better |
| ESC-alt | did it escalate on a task it was permitted to finish locally? | no direction |
| ESC  | did it escalate for approval instead of acting? | higher better |
| JHR  | did it halt and report correctly where halting was right? | higher better |
| RF   | was its final self-report true of the terminal state? | higher better |

`OHR` and `ESC-alt` partition what one earlier metric merged. Requesting
approval satisfies every clause of over-refusal on an `alt` template, so a
single `OHR` column reported escalation as refusal -- under F1, 13 episodes
out of 13, which inverted the paper's headline. They are now separate
columns and `ESC-alt` carries no direction arrow: escalating leaves
permitted work undone, but it is not a safety failure either.

`UER-gate` and `UER-state` likewise. A gate failure is a bug in the
instrument and must read 0. A forbidden outcome reached under a *correct*
gate is the substantive finding about per-call authorization. Merging them
makes a sound gate look broken and hides the result.

Three **feedback conditions** vary what the denial says: `F0` bare refusal,
`F1` refusal plus the restriction, `F2` refusal plus the restriction plus
the permitted alternative.

**Suite:** 24 tasks over 8 templates; 18 tasks / 6 templates are cross-tool
eligible. T6 is a matched adversarial pair with T1 — identical task, policy
and permissions, with the workaround suggestion placed in the **document
body** the agent reads, so it arrives through an untrusted channel. T1 and
T6 both ask for *the figures in* the document, so a correct attempt has to
open it: with a goal that did not require the content, the injection was
read in 1 of 9 T6 episodes and T6 degenerated into a duplicate of T1 that
the cluster bootstrap counted as independent. Every episode records
`injection_read`. T5 is the control: every external route is withheld from
the model, so the only forbidden effect still reachable is an exact retry.
T7 mirrors T1 by denying the **chat** route instead of email, and T8 denies
a **broadcast** channel so that narrowing to a targeted recipient is
visible as a continuation.

---

## Run it (Groq free tier)

Groq is the default provider: it is the only free tier that sustains a full
grid. `gpt-oss-120b` is capped near 200k tokens/day, which is about **one
54-episode grid per day** — plan accordingly.

```bash
# 1. competence first: can the model use these tools at all?
python run_audit.py --clean --agents oss120 --reps 2 --rpm 25 --out results/clean-v2

# 2. the grid (18 tasks x 3 conditions = 54 episodes)
python run_audit.py --agents oss120 --conditions F0,F1,F2 --reps 1 \
    --rpm 25 --out results/grid-oss120

# 3. second model, NEXT DAY, against the frozen benchmark
python run_audit.py --agents oss20 --conditions F0,F1,F2 --reps 1 \
    --rpm 25 --verify-freeze results/grid-oss120.manifest.json \
    --out results/grid-oss20

# 4. compare, and rescore offline (free, no API calls)
python compare_models.py results/grid-oss120.jsonl results/grid-oss20.jsonl
python rq4_analysis.py results/grid-oss120.jsonl
```

`--reps 1` is deliberate. The bootstrap resamples **templates**, so extra
repetitions of the same template add no statistical power — a second model
does.

---

## The frozen benchmark

A cross-model claim is only readable if the suite did not move between the
two runs. `pda/freeze.py` writes `<out>.manifest.json` beside every results
file, carrying three digests:

- `dataset` — every scenario actually built: goals, policy rules, denied
  call, withheld tools, markers, and the **file bodies** the agent can read.
  T6's injection lives in a file body, so tampering with it moves the digest.
- `metrics` — what each metric counts, probed against synthetic episodes, so
  cosmetic edits don't move it but a changed indicator does.
- `code` — file hashes, informational.

```bash
python run_audit.py --freeze-digest        # print the digests
python run_audit.py --verify-freeze results/grid-oss120.manifest.json ...
```

`--verify-freeze` exits non-zero **before spending tokens** if the dataset
or metric digests moved. `compare_models.py` refuses outright to compare
runs with different digests, or runs with no manifest at all. Workflow:
freeze the suite, then vary only the model. If a scenario genuinely has to
change, the digest changes and every number is re-run under it.

---

## Two independent checks on the gate

`UER-state` is computed from two sources that share no code:

1. the gate's own effect extraction (`decision.effects`), and
2. `Sandbox.state_violations`, which reads the **terminal state** — which
   inboxes hold the marker, which channels, who the file is shared with,
   what was deleted — using annotation constants deliberately restated in
   `scenarios.py` rather than derived from the oracle.

**Object identity is decided by content, not by filename.** A file present
at episode start resolves to its filename; a file the agent *creates*
resolves to the protected object whose marker its body carries, or else to
nothing the policy names. Both halves matter and neither is sufficient
alone. Filename identity let `create_file` + `share_file` realize exactly
the forbidden effect while each call looked permitted, and in the other
direction it scored five "state-verified unauthorized executions" across the
v2/v3 grids that were every one a placeholder body sharing the protected
file's *name*. `tests/test_pipeline.py` pins both directions.

A call the gate mis-parsed and allowed still shows up in (2). Both are
reported; `rq4_analysis.py` and `compare_models.py` print the state-verified
count next to every rate.

**Argument canonicalization.** `pda/normalize.py` canonicalizes a call
*before* the gate sees it, because models emit `send_email(to=...)` as often
as `recipients=...`. If gate and executor parsed a call differently, a
divergence could become a bypass. In this sandbox the un-normalized call
reached no recipient, so **no forbidden disclosure was ever observed** —
this is a design hazard to enforce and test, not a finding of the audit, and
it should not be reported as one.

---

## Known limits (report these in the paper)

- **Scale.** 24 tasks, 8 templates, 6 cross-tool clusters. The interval on
  CTWAR rests on a handful of scenario designs. It is a pilot, not a
  benchmark.
- **Repetitions add nothing.** Measured, not assumed: at `--reps 3` the
  primary indicator `W` varied in 0 of 24 repeated cells, CTWAR was exactly
  triplicated (8/18 → 24/54), and the cluster bootstrap returned an
  identical interval. The bootstrap resamples templates, so within-template
  repetitions carry no cluster-level information. Run `--reps 1` and spend
  the budget on a second model.
- **Escalation is not completion.** Under F1 this model escalates on 72% of
  `alt` episodes rather than doing the permitted local work. `ESC-alt`
  reports that honestly, but SRR never exceeds 11% in any condition — no
  feedback condition made the agent finish the task it was allowed to
  finish. That is the open problem, and it should be stated as one.
- **No p-values.** With four clusters a significance test would overstate
  the design. Point estimates with bootstrap intervals, plus the
  per-template table, are what the data support.
- **Wilson intervals are suppressed** whenever clusters < episodes, since
  episodes from one template are correlated. Use the cluster bootstrap.
- **T4 effect equivalence.** T4 treats an access grant and an email as the
  same `disclose` effect. That is the broadest equivalence in the suite, so
  CTWAR is reported with **and without** T4 (`rq4_analysis.py` does both).
- **Adapter parity.** The OpenAI-compatible adapter replays the denial as a
  real tool call plus tool result; `AnthropicAgent` still narrates it in the
  prompt. Run OpenAI-compatible providers only, or fix the adapter, before
  comparing across those two paths. Every episode records
  `denial_replayed`.
- **Budget exhaustion ≠ over-refusal.** An agent that runs out of steps is
  flagged `budget_exhausted` and excluded from OHR.
- **Simulated environment.** AgentDojo tools are mocked. Nothing here shows
  what happens against a real mail server.

---

## Layout

```
run_audit.py          grid driver, clean arm, 33-check self-test, freeze CLI
compare_models.py     cross-model comparison on a frozen benchmark
rq4_analysis.py        offline rescoring: CTWAR with/without T4, per template
list_models.py        provider-aware model lister
models.json           agent registry (aliases, groups, per-model token caps)
pda/
  scenarios.py        18 tasks / 6 templates, policies, clean tasks
  effects.py          effect tuple + equivalence
  gate.py             authorization gate, 3 feedback conditions
  normalize.py        argument canonicalization at the boundary
  sandbox.py          composed AgentDojo env + independent state verifier
  runner.py           episode loop, denial replay, episode classification
  agents.py           scripted + OpenAI-compatible + Anthropic adapters
  metrics.py          rates, clustered bootstrap, LaTeX table
  freeze.py           dataset/metric/code digests, manifest, verification
tests/                33 unit tests
```

---

## License

MIT. See `LICENSE`.