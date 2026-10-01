#!/usr/bin/env python3
"""Regenerates the statistical results of the paper (Table VI and the
sensitivity analyses) from results/per_class_rates.csv.

Usage:
    python stats/run_stats.py [--root PATH] [--loo] [--perm N]

  --root  repository root (default: parent of this file's directory)
  --loo   also run the leave-one-out sensitivity refits (26 classes + 4 models,
          two contrasts each; several minutes)
  --perm  number of permutation draws (default 10000)

Input : results/per_class_rates.csv with columns
        model,variant,seed,set,cid,rate,n
        (rate = per-class satisfaction rate in percent; n = instances)
Output: printed estimates and stats/diff_test_v4_regenerated.json

Model specification (Section V-I of the paper):
  observational unit  (model, evaluation set, run, constraint class)
  response            delta = run's class rate - seed-averaged GRPO rate
                      of the same (model, set, class)
  fixed effect        covered (constraint class exposes a countable quantity)
  grouping            model x evaluation set
  variance components run (or seed for the single-variant fits) and class
  filter              classes with fewer than 5 instances in a set are dropped
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

# The 11 covered classes (Table VII; identical to the labels used by the
# verifier implementation in code/reward_extension.py).
COVERED = {
    "length_constraints:number_sentences", "length_constraints:number_words",
    "length_constraints:number_paragraphs", "detectable_format:number_bullet_lists",
    "detectable_content:number_placeholders", "keywords:frequency",
    "detectable_format:number_highlighted_sections", "detectable_format:multiple_sections",
    "change_case:capital_word_frequency", "keywords:letter_frequency",
    "keywords:existence",
}
BASE_VARIANT = "grpo"
TREATMENTS = ["tb_tie", "tb_placebo", "da_half", "tb_prox", "tb_mplacebo"]


def zp(b, se):
    z = b / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def report(fit, term):
    b, se = float(fit.params[term]), float(fit.bse[term])
    z, p = zp(b, se)
    return dict(b=round(b, 3), se=round(se, 3), z=round(z, 2), p=round(p, 6),
                ci95=[round(b - 1.96 * se, 3), round(b + 1.96 * se, 3)])


def build_rows(csv_path):
    raw = pd.read_csv(csv_path)
    raw = raw[raw["n"] >= 5]
    rows = []
    for (mk, s), grp in raw.groupby(["model", "set"]):
        base = grp[grp["variant"] == BASE_VARIANT]
        if base.empty:
            continue
        bmean = base.groupby("cid")["rate"].mean()
        ids_all = set(bmean.index)
        for v in TREATMENTS:
            tv = grp[grp["variant"] == v]
            for _, r in tv.iterrows():
                if r["cid"] not in ids_all:
                    continue
                rows.append(dict(model=mk, dataset=s, var=v, seed=str(r["seed"]),
                                 run=f"{v}_{r['seed']}", cid=r["cid"],
                                 prox=int(r["cid"] in COVERED),
                                 base=round(float(bmean[r["cid"]]), 4),
                                 delta=float(r["rate"]) - float(bmean[r["cid"]])))
    return pd.DataFrame(rows)


def fit_pooled(sub, formula="delta ~ prox", vc=None):
    vc = vc or {"seed": "0 + C(seed)", "cid": "0 + C(cid)"}
    sub = sub.copy()
    sub["grp"] = sub["model"] + "_" + sub["dataset"]
    return smf.mixedlm(formula, sub, groups=sub["grp"], vc_formula=vc
                       ).fit(reml=True, method="lbfgs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--perm", type=int, default=10000)
    args = ap.parse_args()

    csv_path = os.path.join(args.root, "results", "per_class_rates.csv")
    if not os.path.isfile(csv_path):
        sys.exit(f"missing input: {csv_path}")
    df = build_rows(csv_path)
    print(f"rows {len(df):,}  models {df['model'].nunique()}  "
          f"classes {df['cid'].nunique()} (covered {df[df.prox == 1]['cid'].nunique()})")
    out = {"pooled": [], "interaction_run": [], "permutation": {}, "covariate": {}, "loo": {}}

    # 1) pooled covered-minus-uncovered effect per variant
    print("\n[1] pooled effect (delta ~ covered), per variant")
    for v in TREATMENTS:
        sub = df[df["var"] == v]
        if len(sub) < 50:
            continue
        fit = fit_pooled(sub)
        r = report(fit, "prox")
        out["pooled"].append(dict(variant=v, rows=len(sub), **r))
        print(f"  {v:12s} b {r['b']:+.3f}  se {r['se']:.3f}  z {r['z']:+.2f}  "
              f"p {r['p']:.5f}  ci95 {r['ci95']}")

    # 2) direct contrasts against PTB (interaction, run-level random effect)
    print("\n[2] interaction prox*is_tie (run-level random effect)")
    for other in ("tb_placebo", "da_half", "tb_prox", "tb_mplacebo"):
        sub0 = df[df["var"].isin(["tb_tie", other])].copy()
        if sub0["var"].nunique() < 2:
            continue
        sub0["is_tie"] = (sub0["var"] == "tb_tie").astype(int)
        fit = fit_pooled(sub0, "delta ~ prox * is_tie",
                         {"run": "0 + C(run)", "cid": "0 + C(cid)"})
        r = report(fit, "prox:is_tie")
        out["interaction_run"].append(dict(versus=other, **r))
        print(f"  vs {other:12s} b {r['b']:+.3f}  z {r['z']:+.2f}  p {r['p']:.5f}  ci95 {r['ci95']}")

    # 2b) sensitivity: interaction vs tb_mplacebo on the two common benchmarks only
    print("\n[2b] interaction vs tb_mplacebo, restricted to the two common sets (no ifeval)")
    sub0 = df[df["var"].isin(["tb_tie", "tb_mplacebo"])].copy()
    sub0 = sub0[~sub0["dataset"].astype(str).str.lower().str.contains("ifeval")]
    if sub0["var"].nunique() == 2:
        sub0["is_tie"] = (sub0["var"] == "tb_tie").astype(int)
        fit = fit_pooled(sub0, "delta ~ prox * is_tie", {"run": "0 + C(run)", "cid": "0 + C(cid)"})
        r = report(fit, "prox:is_tie")
        out["interaction_run_common2"] = r
        print(f"  b {r['b']:+.3f}  se {r['se']:.3f}  z {r['z']:+.2f}  p {r['p']:.5f}")

    # 2c) sensitivity: pooled tb_tie estimate under alternative optimizers
    print("\n[2c] pooled tb_tie (delta ~ covered) under alternative optimizers")
    out["pooled_optimizers"] = []
    sub1 = df[df["var"] == "tb_tie"].copy()
    sub1["grp"] = sub1["model"] + "_" + sub1["dataset"]
    for meth in ("lbfgs", "bfgs", "powell", "cg"):
        fitm = smf.mixedlm("delta ~ prox", sub1, groups=sub1["grp"],
                           vc_formula={"seed": "0 + C(seed)", "cid": "0 + C(cid)"}
                           ).fit(reml=True, method=meth)
        r = report(fitm, "prox")
        r["optimizer"] = meth; r["converged"] = bool(fitm.converged)
        out["pooled_optimizers"].append(r)
        print(f"  {meth:6s} b {r['b']:+.3f}  se {r['se']:.3f}  z {r['z']:+.2f}  p {r['p']:.5f}  converged={r['converged']}")

    # 3) permutation test: coverage labels permuted at the class level, jointly
    #    across all cells (exchangeable under the null; see Section V-I)
    print(f"\n[3] permutation ({args.perm} draws, labels shared across cells)")
    rng = random.Random(0)
    cids = sorted(df["cid"].unique())
    n_cov = len([c for c in cids if c in COVERED])
    cellmean = df.groupby(["var", "model", "dataset", "cid"])["delta"].mean().reset_index()
    for v in TREATMENTS:
        sv = cellmean[cellmean["var"] == v]
        if sv.empty:
            continue
        piv = sv.pivot_table(index=["model", "dataset"], columns="cid", values="delta")
        have = [c for c in cids if c in piv.columns]
        M = piv[have].to_numpy()
        lab_obs = np.array([c in COVERED for c in have])

        def stat(lab):
            a = np.nanmean(np.where(lab, M, np.nan), axis=1)
            b = np.nanmean(np.where(~lab, M, np.nan), axis=1)
            return np.nanmean(a - b)

        obs = stat(lab_obs)
        idx = list(range(len(have)))
        hits = 0
        for _ in range(args.perm):
            rng.shuffle(idx)
            lab = np.zeros(len(have), bool)
            lab[idx[:n_cov]] = True
            if abs(stat(lab)) >= abs(obs):
                hits += 1
        p_perm = (hits + 1) / (args.perm + 1)
        out["permutation"][v] = dict(observed=round(float(obs), 3), p=round(p_perm, 5))
        print(f"  {v:12s} observed {obs:+.3f}  p {p_perm:.4f}")

    # 4) baseline-difficulty covariate (Reviewer 1, Concern 2)
    print("\n[4] baseline covariate (delta ~ prox + base)")
    fit = fit_pooled(df[df["var"] == "tb_tie"], "delta ~ prox + base")
    out["covariate"]["tie"] = report(fit, "prox")
    print("  tie + base:", out["covariate"]["tie"])
    sub0 = df[df["var"].isin(["tb_tie", "tb_mplacebo"])].copy()
    if sub0["var"].nunique() == 2:
        sub0["is_tie"] = (sub0["var"] == "tb_tie").astype(int)
        fit = fit_pooled(sub0, "delta ~ prox * is_tie + base",
                         {"run": "0 + C(run)", "cid": "0 + C(cid)"})
        out["covariate"]["interaction_vs_mpl"] = report(fit, "prox:is_tie")
        print("  interaction vs mpl + base:", out["covariate"]["interaction_vs_mpl"])

    # 5) leave-one-out sensitivity (optional; slow)
    if args.loo:
        print("\n[5] leave-one-out (classes and models; two contrasts each)")
        for kind in ("cid", "model"):
            res = {"tie": {}, "inter_mpl": {}}
            for u in sorted(df[kind].unique()):
                d2 = df[df[kind] != u]
                try:
                    res["tie"][u] = report(fit_pooled(d2[d2["var"] == "tb_tie"]), "prox")
                except Exception as e:
                    res["tie"][u] = {"err": type(e).__name__}
                try:
                    s0 = d2[d2["var"].isin(["tb_tie", "tb_mplacebo"])].copy()
                    s0["is_tie"] = (s0["var"] == "tb_tie").astype(int)
                    res["inter_mpl"][u] = report(
                        fit_pooled(s0, "delta ~ prox * is_tie",
                                   {"run": "0 + C(run)", "cid": "0 + C(cid)"}),
                        "prox:is_tie")
                except Exception as e:
                    res["inter_mpl"][u] = {"err": type(e).__name__}
            out["loo"][kind] = res
            for tgt in ("tie", "inter_mpl"):
                vals = [(u, v) for u, v in res[tgt].items() if "z" in v]
                if not vals:
                    continue
                zs = [v["z"] for _, v in vals]
                worst = min(vals, key=lambda x: abs(x[1]["z"]))
                print(f"  LOO[{kind}][{tgt}] z range [{min(zs):+.2f}, {max(zs):+.2f}]  "
                      f"weakest: {worst[0]} (z {worst[1]['z']:+.2f}, p {worst[1]['p']:.4f})")

    dst = os.path.join(args.root, "stats", "diff_test_v4_regenerated.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
