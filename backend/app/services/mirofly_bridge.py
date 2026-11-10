"""
MiroFly bridge — gives MiroFish agents REAL fly-brain affect cores.

Each agent gets a batch element of a spiking circuit built from the actual
FlyWire v783 connectome (sugar + bitter gustatory channels). The bridge
translates post/event text into sensory drives, ticks the agent's circuit,
and returns an internal-state clause for the agent's prompt. Affect ticks
cost ZERO LLM tokens.

HONESTY (project spine — do not delete):
- The fly brain does not understand text. The LLM does all semantics; the
  circuit contributes mechanistic affect dynamics (habituation, alarm,
  novelty, learned valence/trust via the MB-inspired layer).
- Measured-wiring results so far (see Poom5741/MiroFly README): real
  topology discriminates sweet/bitter and gives societies coordinated,
  contained, on-time outrage — rewired controls fail these.

Setup:
  export MIROFLY_FLY_BRAIN_DIR=/path/to/fly-brain   # data files
  pip install numpy pandas pyarrow                  # bridge deps

Usage sketch inside an OASIS/Twitter step loop:

    bridge = MiroFlyBridge.from_env(n_agents=len(agents))
    ...
    state = bridge.affect(agent_id, post_text_being_read)
    user_message += " " + MiroFlyBridge.state_clause(state)
    # after the agent acts:
    bridge.learn(agent_id, post_text_being_read, outcome=+1 if action=="like" else -1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from mirofly import build_dual_core, SpikeCore, PreferenceLayer  # noqa: E402
from mirofly.encoder import encode  # noqa: E402

DEFAULT_FLY_BRAIN = str(Path.home() / "fly-brain")


class MiroFlyBridge:
    def __init__(self, n_agents: int = 40, fly_brain_dir: str | None = None,
                 gain: float = 16.0, learn: bool = True,
                 topics: list[str] | None = None, citizen_seed: int = 7):
        fly_brain_dir = fly_brain_dir or os.environ.get(
            "MIROFLY_FLY_BRAIN_DIR", DEFAULT_FLY_BRAIN)
        self.circuit = build_dual_core(Path(fly_brain_dir))
        self.n_agents = n_agents
        self.gain = gain
        self.core = SpikeCore(self.circuit, n_agents, seed=citizen_seed,
                              gain=gain)
        self.pref = PreferenceLayer(
            n_agents, topics=topics or ["general"], seed=5) if learn else None
        self._last = {}

    @classmethod
    def from_env(cls, n_agents: int = 40, **kw) -> "MiroFlyBridge":
        return cls(n_agents=n_agents, **kw)

    # ---------- per-step affect ----------
    def affect(self, agent_id: int, text: str, settle: int = 3) -> dict:
        """Tick one agent's circuit on `text` and return its internal state.

        `settle`: run the circuit a few ticks on the stimulus (a burst, not
        an instant) so rate EMAs and baseline deviations respond. Costs no
        LLM tokens; each tick is ~10ms on CPU for 40 agents.
        """
        drive = encode(text)
        drives = np_zeros(self.n_agents)
        drives[agent_id] = drive
        ro = {}
        out = {"novelty": [0.0] * self.n_agents}
        for _ in range(max(1, settle)):
            out = self.core.tick(drives)
            ro = self.core.readouts()
        st = {k: float(v[agent_id]) for k, v in ro.items()}
        st["novelty"] = float(out["novelty"][agent_id])
        self._last = {"agent_id": agent_id, "text": text, "state": st}
        return st

    def learn(self, agent_id: int, text: str, outcome: float):
        """Dopamine-gated valence/trust update from an observed outcome.

        outcome: +1 good for the agent, -1 bad, 0 neutral. Learned valence
        of the topic moves toward this sign (bad event -> negative valence).
        """
        if self.pref is None:
            return
        topic = text[:24]
        ti = self.pref.t_idx.get(topic)
        if ti is None:
            self.pref.topics.append(topic)
            self.pref.t_idx[topic] = ti = len(self.pref.topics) - 1
            self.pref.valence = _hstack_topic(self.pref.valence, self.n_agents)
        outcome_valence = float(np.clip(outcome, -1, 1))
        dopamine = float(self._last.get("state", {}).get("novelty", 0.0))
        self.pref.update(np.array([ti]), np.full(self.n_agents, outcome_valence),
                         np.full(self.n_agents, dopamine))

    # ---------- prompt surface ----------
    def quiet(self, n: int = 15):
        """World time passing: silence + sleep-like synaptic renormalization.

        The real circuit holds reverberant activity once ignited (an emergent
        working-memory property). Flies downscale synapses during sleep
        (Tononi's synaptic homeostasis); quiet ticks apply the same: extra
        membrane leak + activity decay, so scenes start from a rested brain.
        """
        for _ in range(n):
            self.core.tick(np_zeros(self.n_agents))
            self.core.v *= 0.50   # sleep downscaling: collapse reverberation
            self.core.rate *= 0.85
            # renormalize the reference too, or phantom gaps pollute readouts
            if hasattr(self.core, "_base_a"):
                self.core._base_a *= 0.85
                self.core._base_b *= 0.85

    @staticmethod
    def state_clause(state: dict, floor: float = 0.03, margin: float = 0.01) -> str:
        """Scale-free, dominance-based: the most-deviating channel names the
        feeling; if the top two are within `margin`, the honest answer is
        mixed feelings (real cross-channel coupling exists)."""
        a, v, l = state["approach"], state["avoid"], state["alarm"]
        ranked = sorted([("alarm", l), ("avoid", v), ("approach", a)],
                        key=lambda x: -x[1])
        kind, val = ranked[0]
        runner = ranked[1][1]
        if val < floor:
            feeling = "flat, nothing grabs you"
        elif val - runner < margin:
            feeling = "mixed feelings, pulled both ways"
        elif kind == "alarm":
            feeling = "a spike of alarm"
        elif kind == "avoid":
            feeling = "irritation and disapproval"
        else:
            feeling = "it feels good, you are drawn to it"
        parts = [feeling]
        if state["novelty"] > 0.1:
            parts.append("this feels surprisingly new")
        return "[Internal state: " + "; ".join(parts) + ".]"

    def agent_clause(self, agent_id: int, text: str, settle: int = 5) -> str:
        """Prompt fragment for one agent. Scales per-neuron deviations (~0.1
        range) to the clause thresholds tuned on v0.1 group readouts."""
        st = self.affect(agent_id, text, settle=settle)
        scaled = {k: v * 6.0 for k, v in st.items()}
        return self.state_clause(scaled)

    # ---------- reporting ----------
    def valence_report(self, top_n: int = 10) -> dict:
        if self.pref is None:
            return {}
        import numpy as np
        mean_v = self.pref.valence.mean(axis=0)
        idx = np.argsort(mean_v)
        topics = self.pref.topics
        return {
            "most_negative": [(topics[i], float(mean_v[i])) for i in idx[:top_n]],
            "most_positive": [(topics[i], float(mean_v[i])) for i in idx[-top_n:]],
        }


def np_zeros(n):
    return np.zeros(n)


def _hstack_topic(valence, n):
    return np.hstack([valence, np.zeros((n, 1))])
