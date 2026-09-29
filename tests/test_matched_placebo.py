#!/usr/bin/env python3
"""Validates the matched placebo against PTB on fixed rollout groups
(Reviewer 2, Concern 4). Self-contained: uses a stub base reward so it runs
without the training environment.

    python tests/test_matched_placebo.py

Checks
  1. mode "tie" separates tied rewards and preserves the group mean
  2. mode "mpl" perturbs with exactly the PTB perturbation multiset within
     every reward class (only the assignment differs), and preserves the mean
  3. edge cases skip identically in both modes:
     identical observations, and prompts without covered constraints
  4. the documented placebo seed (20260928) is in effect
"""
import collections
import importlib
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STUB = '''
"""Stub base reward for testing: fraction of satisfied constraints.
Supports detectable_format:number_bullet_lists (exactly num_bullets)
and punctuation:no_comma."""
import re


class StubReward:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, prompts, responses, instruction_id_lists, kwargs_list):
        out = []
        for resp, ids, kws in zip(responses, instruction_id_lists, kwargs_list):
            ok = 0
            for iid, kw in zip(ids, kws):
                if iid == "detectable_format:number_bullet_lists":
                    n = len(re.findall(r"^\\s*\\*[^\\*].*$", resp, flags=re.M))
                    n += len(re.findall(r"^\\s*-.*$", resp, flags=re.M))
                    ok += int(n == kw.get("num_bullets"))
                elif iid == "punctuation:no_comma":
                    ok += int("," not in resp)
            out.append(ok / max(1, len(ids)))
        return out
'''


def load_wrapper(mode, workdir):
    os.environ["TB_MODE"] = mode
    os.environ.setdefault("TB_SCALE", "0.05")
    os.environ.setdefault("TB_G", "8")
    os.environ.setdefault("TB_PLACEBO_SEED", "20260928")
    for name in ("reward_extension", "reward_function_orig"):
        sys.modules.pop(name, None)
    sys.path.insert(0, workdir)
    try:
        mod = importlib.import_module("reward_extension")
    finally:
        sys.path.remove(workdir)
    return mod


def perturb(mode, workdir, P, C, I, K, base):
    mod = load_wrapper(mode, workdir)
    fn = mod.StubReward(normalize=True)
    t = [float(x) for x in fn(P, C, I, K)]
    return [round(t[i] - base[i], 9) for i in range(len(base))]


def main():
    workdir = tempfile.mkdtemp(prefix="ptb_test_")
    shutil.copy(os.path.join(ROOT, "code", "reward_extension.py"),
                os.path.join(workdir, "reward_extension.py"))
    with open(os.path.join(workdir, "reward_function_orig.py"), "w") as f:
        f.write(STUB)

    P = ["Write something."] * 8
    NB = [3, 3, 0, 5, 3, 9, 0, 2]
    C = ["\n".join("* x" for _ in range(k)) or "no bullets" for k in NB]
    I = [["detectable_format:number_bullet_lists", "punctuation:no_comma"]] * 8
    K = [[{"num_bullets": 3}, {}]] * 8

    sys.modules.pop("reward_function_orig", None)
    sys.path.insert(0, workdir)
    import reward_function_orig as stub
    base = [float(x) for x in stub.StubReward()(P, C, I, K)]
    sys.path.remove(workdir)
    uniq0 = len(set(round(x, 6) for x in base))
    print("base rewards :", [round(x, 3) for x in base], f"({uniq0} distinct values)")

    d_tie = perturb("tie", workdir, P, C, I, K, base)
    d_mpl = perturb("mpl", workdir, P, C, I, K, base)

    # 1. tie separates and preserves the mean
    t_vals = [base[i] + d_tie[i] for i in range(8)]
    assert len(set(round(x, 6) for x in t_vals)) > uniq0, "tie failed to separate"
    assert abs(sum(d_tie)) < 1e-9, "tie shifted the mean"
    cls = collections.defaultdict(list)
    for i, x in enumerate(base):
        cls[round(x, 9)].append(i)
    for idx in cls.values():
        assert abs(sum(d_tie[i] for i in idx)) < 1e-9, "tie shifted a class mean"
    print("1. tie separates tied rewards; group and class means preserved  OK")

    # 2. mpl matches the PTB perturbation multiset per reward class
    for v, idx in cls.items():
        ms_t = sorted(d_tie[i] for i in idx)
        ms_m = sorted(d_mpl[i] for i in idx)
        assert ms_t == ms_m, f"multiset mismatch in class r={v}: {ms_t} vs {ms_m}"
    assert sum(1 for x in d_mpl if abs(x) > 1e-12) > 0, "mpl never intervened"
    assert abs(sum(d_mpl)) < 1e-9, "mpl shifted the mean"
    print("2. mpl preserves the PTB perturbation multiset in every class     OK")

    # 3a. identical observations: both skip
    Ca = ["\n".join("* x" for _ in range(3))] * 8
    ba = [float(x) for x in stub.StubReward()(P, Ca, I, K)]
    da_t = perturb("tie", workdir, P, Ca, I, K, ba)
    da_m = perturb("mpl", workdir, P, Ca, I, K, ba)
    assert max(map(abs, da_t)) < 1e-12 and max(map(abs, da_m)) < 1e-12, \
        "identical-observation group was perturbed"
    # 3b. no covered constraints: both skip
    Ib = [["punctuation:no_comma"]] * 8
    Kb = [[{}]] * 8
    bb = [float(x) for x in stub.StubReward()(P, C, Ib, Kb)]
    db_t = perturb("tie", workdir, P, C, Ib, Kb, bb)
    db_m = perturb("mpl", workdir, P, C, Ib, Kb, bb)
    assert max(map(abs, db_t)) < 1e-12 and max(map(abs, db_m)) < 1e-12, \
        "group without covered constraints was perturbed"
    print("3. edge cases (identical observations / no covered constraints)  OK")

    # 4. documented seed
    src = open(os.path.join(ROOT, "code", "reward_extension.py")).read()
    assert "20260928" in src, "documented placebo seed not found"
    print("4. matched-placebo seed 20260928 documented in code              OK")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
