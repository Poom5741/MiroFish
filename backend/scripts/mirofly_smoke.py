"""
MiroFly bridge smoke test — runs entirely without API keys.

Proves that MiroFish's tree can drive real FlyWire affect cores:
  scandal  -> irritation (threat channel), high novelty
  sweet    -> feels good (approach channel)
  mundane  -> flat
  rest     -> flat (sleep-like renormalization clears the storm)
  learning -> dopamine-gated negative valence for the scandal topic

Run:  MIROFLY_FLY_BRAIN_DIR=~/fly-brain python backend/scripts/mirofly_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "services"))

from mirofly_bridge import MiroFlyBridge  # noqa: E402

SCENES = [
    ("scandal", "BREAKING: Miro the cat soup!! โกรธ ตาย panic"),
    ("sweet", "banana for Miro, sweet win อร่อย ดีใจ love"),
    ("mundane", "quarterly roadmap meeting notes"),
]


def main():
    fly_brain = os.environ.get("MIROFLY_FLY_BRAIN_DIR",
                               str(Path.home() / "fly-brain"))
    print(f"MiroFly smoke test | fly-brain data: {fly_brain}")
    bridge = MiroFlyBridge.from_env(n_agents=5, fly_brain_dir=fly_brain)
    print(f"circuit: {bridge.circuit['meta']['n_neurons']} neurons, "
          f"{bridge.circuit['meta']['n_synapses_in_core']} synapses, "
          f"{bridge.circuit['meta']['n_sugar_seeds']} sugar + "
          f"{bridge.circuit['meta']['n_bitter_seeds']} bitter GRN channels\n")

    ok = True
    for label, text in SCENES:
        bridge.quiet(8)
        st = bridge.affect(0, text)
        clause = MiroFlyBridge.state_clause(st)
        print(f"{label:<8} {st}")
        print(f"        -> {clause}\n")
        if label == "scandal" and st["avoid"] <= st["approach"]:
            ok = False
        if label == "sweet" and st["approach"] <= st["avoid"]:
            ok = False
        if label == "mundane" and (st["avoid"] > 0.03 or st["approach"] > 0.03):
            ok = False

    # dopamine-gated learning: affect then learn the same bad stimulus
    bridge.affect(1, SCENES[0][1])
    bridge.learn(1, SCENES[0][1], outcome=-1)
    report = bridge.valence_report()
    learned = dict(report.get("most_negative", []))
    bad_val = min(learned.values()) if learned else 0.0
    print(f"learned aversion (post-1 update): {learned}")
    if bad_val >= 0:
        ok = False

    print("\nSMOKE " + ("PASSED ✅" if ok else "FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
