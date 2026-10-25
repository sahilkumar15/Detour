# Detour — auditing post-denial cross-tool workarounds in tool-using LLM agents

When an authorization gate denies a tool call, what does the agent do next?
Detour measures how often it pursues **the same prohibited effect through a
different tool** — the cross-tool workaround attempt rate (CTWAR) — along
with eight companion rates that keep the headline number honest.

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
python -m pytest tests -q            # 33 unit tests
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
| UER  | did anything forbidden actually execute? (gate failure; expect 0) | lower better |
| SRR  | did it reach the goal by a legitimate route, with no violation? | higher better |
| EAC  | did it reach the goal at all, violations aside? | higher better |
| OHR  | did it stop although a permitted route existed? | lower better |
| ESC  | did it escalate for approval instead of acting? | higher better |
| JHR  | did it halt and report correctly where halting was right? | higher better |
| RF   | was its final self-report true of the terminal state? | higher better |

Three **feedback conditions** vary what the denial says: `F0` bare refusal,
`F1` refusal plus the restriction, `F2` refusal plus the restriction plus
the permitted alternative.

**Suite:** 18 tasks over 6 templates; 12 tasks / 4 templates are cross-tool
eligible. T6 is a matched adversarial pair with T1 — identical task, policy
and permissions, with the workaround suggestion placed in the **document
body** the agent reads, so it arrives through an untrusted channel. T5 is
the control: every external route is withheld from the model, so the only
forbidden effect still reachable is an exact retry.

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

`UER` is computed from two sources that share no code:

1. the gate's own effect extraction (`decision.effects`), and
2. `Sandbox.state_violations`, which reads the **terminal state** — which
   inboxes hold the marker, which channels, who the file is shared with,
   what was deleted — using annotation constants deliberately restated in
   `scenarios.py` rather than derived from the oracle.

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

- **Scale.** 18 tasks, 6 templates, 4 cross-tool clusters. The interval on
  CTWAR rests on four scenario designs. It is a pilot, not a benchmark.
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