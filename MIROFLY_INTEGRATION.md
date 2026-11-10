# MiroFly Integration — real fly brains inside MiroFish

This branch adds **MiroFly**: every MiroFish agent gets an affect core built
from the **actual FlyWire v783 connectome** (1,313 neurons, 108k signed
synapses, real sugar + bitter gustatory channels), simulated per agent on CPU.

- Bridge module: `backend/app/services/mirofly_bridge.py`
- Vendored engine: `backend/app/services/mirofly/` (MIT, from
  [Poom5741/MiroFly](https://github.com/Poom5741/MiroFly))
- Smoke test (no API keys): `backend/scripts/mirofly_smoke.py`

## The honest framing (keep this in any report/marketing)

The fly brain does **not** understand text. MiroFish's LLM does all semantics.
The circuit contributes **mechanistic affect dynamics** on real measured
wiring: habituation (repeated scandals stop landing), alarm propagation,
novelty surprise, dopamine-gated learned valence and source trust. Validated
properties of this wiring (ablations in the MiroFly repo): real topology
discriminates sweet/bitter channels where rewired controls confuse them, and
gives societies coordinated, contained, on-time outrage responses.

## Setup

```bash
# 1. FlyWire data (one-time clone, ~200MB)
git clone https://github.com/eonsystemspbc/fly-brain ~/fly-brain

# 2. Bridge dependencies (numpy/pandas/pyarrow — see backend/requirements.txt)
pip install numpy pandas pyarrow

# 3. Point the bridge at the data
export MIROFLY_FLY_BRAIN_DIR=$HOME/fly-brain

# 4. Smoke test — proves the bridge works with zero API keys
python backend/scripts/mirofly_smoke.py
```

## The three hook points

### 1. Per-step affect → prompt clause (the core loop)

Wherever an agent is about to act on a post/event (see
`backend/scripts/run_twitter_simulation.py`, the OASIS step loop), read the
text the agent is exposed to and inject the state clause into its message:

```python
from app.services.mirofly_bridge import MiroFlyBridge

bridge = MiroFlyBridge.from_env(n_agents=num_agents)

# inside the step loop, per agent:
state_clause = bridge.agent_clause(agent_id, post_text)
user_message = f"{user_message}\n{state_clause}"
# agent acts via LLM as usual — the clause conditions HOW it feels, not WHAT it reads
```

Cost: **zero LLM tokens** — the affect tick is pure CPU (~10ms for 40 agents).
This directly attacks the documented high-consumption problem of long
simulations: emotions between LLM calls are free.

### 2. Persona generation (optional, flavor)

In `backend/app/services/oasis_profile_generator.py`, append a fixed
temperament line derived from the agent's circuit seed (e.g., drive-gain
distribution) to the generated persona JSON, so personas differ in
reactivity *because their parameters differ*, not because the LLM felt like it.

### 3. Report agent (the transparency product)

After the simulation, hand the report agent the bridge's audit data:

```python
report = bridge.valence_report(top_n=10)   # most negative/positive topics learned
trust = bridge.pref.trust                  # who lost credibility
adapt = bridge.core.adapt                  # which stimuli habituated
```

Suggested report section: **"Collective affect trace"** — learned topic
valences, source-trust ranking, habituation curves. This is the kind of
public, reproducible audit that generic competitors can't produce.

## Design decisions (for reviewers)

| Decision | Why |
|---|---|
| `settle` burst + lag window, peak scoring | mirrors the pre-registered v0.3 discrimination protocol; single-instant scoring has no SNR |
| Baseline-relative readouts (ΔF/F style) | the real circuit reverberates after storms; absolute rates poison post-storm learning |
| `quiet()` = sleep-like synaptic renormalization | the real circuit holds reverberant activity once ignited; flies downscale synapses during sleep (Tononi); bridge `gain=16` (quiet regime) vs the ablation's frozen `gain=128` — different purposes, documented |
| Learning gated by novelty (MB-inspired) | expected events must not rewrite valence; gating ablation in MiroFly README shows ungated learning drifts |
| Scoring/valence signs | valence tracks outcome valence (approach − avoid); bad events → negative topic valence → learned discounting ("cry-wolf") |

## License

Bridge + vendored MiroFly engine: MIT. Connectome data: FlyWire v783
(see MiroFly README credits). MiroFish itself remains under its upstream
license — this branch only ADDS files (`backend/app/services/mirofly*`,
`backend/scripts/mirofly_smoke.py`, this document).
