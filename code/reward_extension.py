# reward_extension.py -- the exact reward wrapper used in all experiments.
# Comments and log strings are in English; the executable logic is the module
# generated and imported by the training runs (all TB_MODE variants included).
# Modes reported in the paper: off (GRPO), tie (PTB), placebo, mpl (matched
# placebo), nz ("Nonzero only" ablation), vc (verifier-consistent ablation).
# Epsilon-sensitivity runs use mode "tie" with TB_SCALE in {0.025, 0.05, 0.10}.
"""Reward-function extension. Inherits from the original (reward_function_orig)
without modifying its logic.

train.py calls it with positional arguments:
    reward_fn(prompts, responses, instruction_id_lists, kwargs_list)

TB_MODE selects the behavior.
  off      original reward, unchanged (= GRPO baseline)
  tie      PTB: proximity-rank tie-breaking applied to every tied subset
  nz       PTB restricted to tied subsets with nonzero reward ("Nonzero only" ablation)
  placebo  random-rank tie-breaking on fully collapsed groups (legacy placebo)
  mpl      matched placebo: computes g under the identical PTB eligibility rule,
           then shuffles that vector within the reward class (perturbation
           multiset preserved; only the assignment is randomized)
  vc       verifier-consistent ablation: verifier-satisfied responses are pinned
           to the top proximity value; order among unsatisfied ones is preserved
  prox     earlier PTB variant acting on collapsed groups only
  v2       prox + unsatisfied constraints only + per-constraint rank averaging + split direction
  part     partial credit for exact-type constraints
  supp     negative uniform shift on saturated groups
  adapt    recomputes the reward as a per-constraint weighted mean (no tie-breaking)

For every mode except adapt, z-normalization follows, so the zero mean of
advantages within a group is maintained.
"""
import os, re, sys, time, json, random, hashlib, collections
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_function_orig as _ORIG

MODE      = os.environ.get("TB_MODE", "off")
SCALE     = float(os.environ.get("TB_SCALE", "0.05"))
G         = int(os.environ.get("TB_G", "8"))
EMA       = float(os.environ.get("TB_ADAPT_EMA", "0.9"))
WMIN      = float(os.environ.get("TB_ADAPT_MIN", "0.5"))
LOG_PATH  = os.environ.get("TB_LOG", "")
SUPP      = float(os.environ.get("TB_SUPP", "0.03"))
_rng      = random.Random(12345)
# Dedicated RNG for the matched placebo. Seed fixed and documented: 20260928.
_rng_pl   = random.Random(int(os.environ.get("TB_PLACEBO_SEED", "20260928")))

_SRC = os.environ.get("TB_SRC_DIR", "/content/src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    import instructions_util as _IU
except Exception:
    _IU = None
try:
    import instructions_registry as _REGMOD
    REG = _REGMOD.INSTRUCTION_DICT
except Exception:
    REG = {}


# ---------------- satisfaction of a single constraint (calls the IFEval checker class) ----------------
def _follow(iid, kw, prompt, resp):
    cls = REG.get(iid)
    if cls is None or not str(resp).strip():
        return None
    try:
        inst = cls(iid)
        keys = inst.get_instruction_args_keys() or []
        f = {k: v for k, v in kw.items() if k in keys}
        if "prompt" in keys:
            f["prompt"] = prompt
        inst.build_description(**f)
        return bool(inst.check_following(resp))
    except Exception:
        return None


# ---------------- the quantity the verifier counts ----------------
def _observe(iid, kw, resp):
    try:
        if iid == "length_constraints:number_words" and _IU:
            return _IU.count_words(resp), kw.get("num_words"), kw.get("relation")
        if iid == "length_constraints:number_sentences" and _IU:
            return _IU.count_sentences(resp), kw.get("num_sentences"), kw.get("relation")
        if iid == "length_constraints:number_paragraphs":
            ps = re.split(r"\s?\*\*\*\s?", resp)
            return len([p for p in ps if p.strip()]), kw.get("num_paragraphs"), "exactly"
        if iid == "detectable_format:number_bullet_lists":
            n = len(re.findall(r"^\s*\*[^\*].*$", resp, flags=re.M))
            n += len(re.findall(r"^\s*-.*$", resp, flags=re.M))
            return n, kw.get("num_bullets"), "exactly"
        if iid == "detectable_content:number_placeholders":
            return len(re.findall(r"\[.*?\]", resp)), kw.get("num_placeholders"), "at least"
        if iid == "detectable_format:number_highlighted_sections":
            n = len([x for x in re.findall(r"\*[^\n\*]*\*", resp) if x.strip("*").strip()])
            n += len([x for x in re.findall(r"\*\*[^\n\*]*\*\*", resp) if x.strip("*").strip()])
            return n, kw.get("num_highlights"), "at least"
        if iid == "detectable_format:multiple_sections":
            sp = kw.get("section_spliter", "Section")
            return len(re.split(re.escape(sp) + r"\s?\d+", resp)) - 1, kw.get("num_sections"), "at least"
        if iid == "change_case:capital_word_frequency":
            n = sum(1 for w in re.findall(r"\b[A-Z]+\b", resp) if len(w) > 1)
            return n, kw.get("capital_frequency"), kw.get("capital_relation")
        if iid == "keywords:frequency":
            k = str(kw.get("keyword", ""))
            return ((len(re.findall(re.escape(k), resp, flags=re.I)) if k else None),
                    kw.get("frequency"), kw.get("relation"))
        if iid == "keywords:letter_frequency":
            l = str(kw.get("letter", ""))
            return ((resp.lower().count(l.lower()) if l else None),
                    kw.get("let_frequency"), kw.get("let_relation"))
        if iid == "keywords:existence":
            ks = kw.get("keywords") or []
            return sum(1 for k in ks if re.search(re.escape(k), resp, flags=re.I)), len(ks), "at least"
    except Exception:
        return None
    return None


def _prox(obs, tgt, rel, split_dir):
    """Proximity in [0, 1]; larger is closer to the target. No attempt (obs = 0) ranks lowest."""
    if obs is None or tgt is None:
        return None
    o, t = float(obs), float(tgt)
    if t <= 0:
        return 1.0 / (1.0 + o)
    if o == 0:
        return 0.0
    r = (rel or "exactly").lower()
    if r in ("at least", "more than"):
        return min(o / t, 1.0)
    if r in ("up to", "less than"):
        return min(t / o, 1.0)
    if split_dir:
        # split direction: undershoot maps to 0.5-1.0, overshoot to 0-0.5; the two never overlap
        return (0.5 + 0.5 * (o / t)) if o < t else (0.5 * (t / o))
    return 1.0 / (1.0 + abs(o - t) / t)


def _to_rank(vals):
    n = len(vals)
    if n < 2:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for t in range(i, j + 1):
            rank[order[t]] = avg
        i = j + 1
    return [r / (n - 1) for r in rank]


# ---------------- g vector for a tied subset ----------------
def _group_g(prompt, comps, ids, kws, common_r):
    """Returns (g vector, diagnostic string) or (None, reason)."""
    n = len(comps)
    if MODE == "placebo":
        v = list(range(n)); _rng.shuffle(v)
        return [x / max(1, n - 1) for x in v], "placebo"
    if common_r >= 1.0 - 1e-9 and MODE not in ("tie", "mpl", "vc"):
        return None, "saturated"     # tie/mpl/vc pass through: proximity cannot separate a saturated group

    unsat = None
    if MODE == "v2":
        # use unsatisfied constraints only; the group is collapsed, so the first completion suffices
        unsat = []
        for j, (iid, kw) in enumerate(zip(ids, kws)):
            f = _follow(iid, kw, prompt, str(comps[0]))
            if f is False:
                unsat.append(j)
        if not unsat:
            return None, "no_unsatisfied"

    use = unsat if unsat is not None else list(range(len(ids)))
    per_con = []                                  # per constraint: [proximity per completion]
    for j in use:
        iid, kw = ids[j], kws[j]
        col = []
        for c in comps:
            o = _observe(iid, kw, str(c))
            p = _prox(o[0], o[1], o[2], MODE == "v2") if o else None
            if MODE == "vc" and p is not None:
                # verifier-consistent: satisfied completions pinned to 1.0; unsatisfied keep their proximity order below 1.0
                sat = _follow(iid, kw, prompt, str(c))
                if sat is True:
                    p = 1.0
                elif sat is False:
                    p = min(p, 0.999)
            col.append(p)
        if any(x is None for x in col):
            continue
        if len(set(round(x, 9) for x in col)) < 2:
            continue                              # this constraint does not separate the subset
        per_con.append(col)
    if not per_con:
        return None, "identical_observations"

    if MODE == "v2":
        # Borda: rank per constraint first, then average (removes dilution)
        ranks = [_to_rank(col) for col in per_con]
        raw = [st.mean(r[i] for r in ranks) for i in range(n)]
    else:
        raw = [st.mean(col[i] for col in per_con) for i in range(n)]
    if len(set(round(x, 9) for x in raw)) < 2:
        return None, "identical_ranks"
    g = _to_rank(raw)
    if MODE == "mpl":
        # === matched placebo ===
        # 1) compute g (normalized average ranks) under the identical PTB eligibility
        #    rule -- everything above is shared with mode "tie"
        # 2) shuffle that g vector within the reward class; the perturbation is
        #    SCALE*(g - mean(g)), so its magnitude, variance, and multiset are
        #    exactly those of PTB and only the assignment is randomized
        # 3) RNG: random.Random(TB_PLACEBO_SEED=20260928), seeded once per process
        g = list(g)
        _rng_pl.shuffle(g)
        return g, f"con{len(per_con)}|mpl"
    return g, f"con{len(per_con)}"


# ---------------- adaptive weighting ----------------
class _W:
    hit = {}          # iid -> moving-average satisfaction rate
    steps = 0


def _weight(iid):
    h = _W.hit.get(iid, 0.5)
    return WMIN + (1.0 - WMIN) * (1.0 - h)


class _S:
    calls = groups = collapsed = applied = tied = 0
    last = 0.0
    why = {}


def _log(obj):
    if not LOG_PATH:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _report():
    d = dict(sorted(_S.why.items(), key=lambda x: -x[1])[:4])
    print(f"[rw:{MODE}] calls {_S.calls}  groups {_S.groups}  "
          f"collapsed {_S.collapsed} ({_S.collapsed/max(1,_S.groups)*100:.1f}%)  "
          f"applied {_S.applied} ({_S.applied/max(1,_S.collapsed)*100:.1f}% of collapsed)  "
          f"tied rollouts {_S.tied}  skip reasons {d}", flush=True)
    if MODE == "adapt" and _W.hit:
        top = sorted(_W.hit.items(), key=lambda x: x[1])
        lo = ", ".join(f"{k.split(':')[1][:14]}={v:.2f}" for k, v in top[:3])
        hi = ", ".join(f"{k.split(':')[1][:14]}={v:.2f}" for k, v in top[-3:])
        print(f"[rw:adapt] lowest satisfaction {lo} | highest {hi}", flush=True)


class RewardMixin:
    def __call__(self, *a, **kw):
        base = super().__call__(*a, **kw)
        if MODE == "off":
            return base
        try:
            prompts = a[0] if len(a) > 0 else kw.get("prompts")
            comps   = a[1] if len(a) > 1 else kw.get("responses")
            ids     = a[2] if len(a) > 2 else kw.get("instruction_id_lists")
            kws     = a[3] if len(a) > 3 else kw.get("kwargs_list")
            _S.calls += 1
            n = len(base)
            if any(x is None for x in (prompts, comps, ids, kws)) or n % G != 0 \
               or len({len(comps), len(ids), len(kws), n}) != 1:
                if _S.calls == 1:
                    print(f"[rw:{MODE}] insufficient information -- returning original rewards", flush=True)
                return base

            # ---------- part: partial proximity credit for exact-type constraints ----------
            if MODE == "part":
                out = []
                for i in range(n):
                    tot = 0.0
                    for iid, kwv in zip(ids[i], kws[i]):
                        f = _follow(iid, kwv, prompts[i], str(comps[i]))
                        if f:
                            tot += 1.0; continue
                        o = _observe(iid, kwv, str(comps[i]))
                        rel = (o[2] if o else None) or ""
                        if o and str(rel).lower() in ("exactly", ""):
                            p = _prox(o[0], o[1], o[2], False)
                            tot += (p if p is not None else 0.0) * 0.5   # at most 0.5 partial credit
                    out.append(tot / max(1, len(ids[i])))
                _S.groups += n // G
                now = time.time()
                if _S.calls <= 3 or now - _S.last > 120:
                    _S.last = now; _report()
                return out

            # ---------- adapt: recompute the reward itself ----------
            if MODE == "adapt":
                out, cnt = [], {}
                for i in range(n):
                    num = den = 0.0
                    for iid, kwv in zip(ids[i], kws[i]):
                        f = _follow(iid, kwv, prompts[i], str(comps[i]))
                        if f is None:
                            continue
                        w = _weight(iid)
                        num += w * (1.0 if f else 0.0); den += w
                        c = cnt.setdefault(iid, [0, 0]); c[0] += 1; c[1] += int(f)
                    out.append(num / den if den > 0 else float(base[i]))
                for iid, (t, s) in cnt.items():          # update moving averages
                    _W.hit[iid] = EMA * _W.hit.get(iid, 0.5) + (1 - EMA) * (s / t)
                _W.steps += 1
                _S.groups += n // G
                now = time.time()
                if _S.calls <= 3 or now - _S.last > 120:
                    _S.last = now; _report()
                    _log({"t": "adapt", "call": _S.calls,
                          "hit": {k: round(v, 4) for k, v in _W.hit.items()}})
                return out

            # ---------- tie-breaking family ----------
            out = [float(x) for x in base]
            for s in range(0, n, G):
                seg = out[s:s + G]
                _S.groups += 1
                vals = collections.Counter(round(x, 9) for x in seg)
                collapsed = len(vals) == 1
                if collapsed:
                    _S.collapsed += 1
                # tie/nz/mpl/vc act on every tied subset; other modes only on fully collapsed groups
                if MODE not in ("tie", "nz", "mpl", "vc") and not collapsed:
                    continue
                if MODE == "supp":
                    # negative uniform shift on saturated groups (all rewards at maximum)
                    if collapsed and seg[0] >= 1.0 - 1e-9:
                        for k in range(G):
                            out[s + k] = seg[k] - SUPP
                        _S.applied += 1
                    continue

                targets = [v for v, c in vals.items() if c >= 2] \
                          if MODE in ("tie", "nz", "mpl", "vc") else [round(seg[0], 9)]
                if MODE == "nz":
                    _d = [v for v in targets if v <= 1e-9]
                    if _d:
                        _S.why["zero_reward_excluded"] = _S.why.get("zero_reward_excluded", 0) + len(_d)
                    targets = [v for v in targets if v > 1e-9]
                for v in targets:
                    idx = [i for i in range(G) if round(seg[i], 9) == v]
                    if len(idx) < 2:
                        continue
                    sub_comps = [comps[s + i] for i in idx]
                    g, why = _group_g(prompts[s], sub_comps, ids[s], kws[s], v)
                    if g is None:
                        _S.why[why] = _S.why.get(why, 0) + 1
                        continue
                    mg = sum(g) / len(g)
                    for j, i in enumerate(idx):
                        out[s + i] = seg[i] + SCALE * (g[j] - mg)
                    _S.applied += 1
                    _S.tied += len(idx)
                    _log({"t": "hit", "call": _S.calls,
                          "ph": hashlib.md5(str(prompts[s]).encode()).hexdigest()[:10],
                          "C": len(ids[s]), "r": round(v, 4), "k": len(idx),
                          "why": why, "collapsed": int(collapsed),
                          "g": [round(x, 3) for x in g]})
            now = time.time()
            if _S.calls <= 3 or now - _S.last > 120:
                _S.last = now; _report()
            return out
        except Exception as e:
            print(f"[rw:{MODE}] exception -- returning original rewards: {type(e).__name__}: {e}", flush=True)
            return base


# Dynamically wrap the first reward class defined in reward_function_orig,
# mirroring how the training notebooks generated this module.
import inspect as _inspect
_BASE_NAME = re.findall(r"^class\s+(\w+)", _inspect.getsource(_ORIG), re.M)[0]
_BASE = getattr(_ORIG, _BASE_NAME)


class _Extended(RewardMixin, _BASE):
    pass


globals()[_BASE_NAME] = _Extended
for _n in dir(_ORIG):
    if not _n.startswith("_") and _n != _BASE_NAME and _n not in globals():
        globals()[_n] = getattr(_ORIG, _n)
