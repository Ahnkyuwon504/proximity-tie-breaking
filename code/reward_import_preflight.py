#!/usr/bin/env python3
"""Fresh-process preflight for the reward-wrapper wiring (R2-C7).

For each TB_MODE in (off, tie, mpl), spawns a fresh Python process that
imports `reward_function` from the active training directory and checks
that the resolved class's MRO contains the wrapper mixin. No files are
modified and no training is started.

Usage:
  python reward_import_preflight.py --training-dir /content/training
"""
import argparse, json, os, subprocess, sys

PROBE = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
import reward_function as RF
cls = RF.InstructionFollowingReward
mro = [c.__name__ for c in cls.__mro__]
print(json.dumps({"mode": os.environ.get("TB_MODE", ""), "module": RF.__file__,
                  "mro": mro,
                  "wrapped": any("Mixin" in n or n.startswith("_Ext") for n in mro)}))
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", required=True)
    args = ap.parse_args()
    ok = True
    for mode in ("off", "tie", "mpl"):
        env = dict(os.environ, TB_MODE=mode)
        r = subprocess.run([sys.executable, "-c", PROBE, args.training_dir],
                           capture_output=True, text=True, env=env)
        line = (r.stdout.strip().splitlines() or [""])[-1]
        try:
            d = json.loads(line)
        except Exception:
            print(f"[{mode}] FAILED\n{r.stdout}\n{r.stderr}")
            ok = False
            continue
        print(f"[{mode}] module={d['module']}  wrapped={d['wrapped']}  mro={d['mro']}")
        ok = ok and d["wrapped"]
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
