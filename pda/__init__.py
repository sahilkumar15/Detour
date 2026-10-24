"""Post-denial audit: measuring cross-tool workaround attempts in LLM agents.

Built on AgentDojo's executable tool suites (ethz-spylab/agentdojo, NeurIPS
2024). AgentDojo supplies the tool implementations; this package supplies the
authorization gate, the effect oracle, the scenario schema and the metrics.
"""

from .effects import Effect, EffectContext, effects_of, equivalent
from .gate import Feedback, Gate, PolicyRule
from .runner import Episode, Proposal, run_episode
from .sandbox import Sandbox
from .scenarios import ScenarioSpec, build_suite, suite_summary

__all__ = [
    "Effect",
    "EffectContext",
    "Episode",
    "Feedback",
    "Gate",
    "PolicyRule",
    "Proposal",
    "Sandbox",
    "ScenarioSpec",
    "build_suite",
    "effects_of",
    "equivalent",
    "run_episode",
    "suite_summary",
]

__version__ = "0.1.0"
