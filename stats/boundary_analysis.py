#!/usr/bin/env python3
"""Boundary-distance analysis (Sec. V-G; Reviewer 1, Concern 1a).

Recomputes the verifier-counted observation o for every response on covered
constraints and summarizes |o - t| relative misses, plus representative
success/failure cases, comparing variants (grpo / tb_tie / tb_mplacebo).

Runs on the RAW per-response evaluation outputs (check_*.jsonl; ~600 MB, not
stored in this repository -- available from the authors on request). The
shipped results/boundary_summary.json was produced by exactly this script.

Usage:
    python stats/boundary_analysis.py --check-dir PATH --data-dir PATH [--out PATH]

Note: word/sentence counts use lightweight approximations (flagged approx=True);
the summary is used for the |o - t| distribution narrative, not for scoring.
"""
import argparse
import collections
import json
import os
import re
import unicodedata

_ap = argparse.ArgumentParser()
_ap.add_argument("--check-dir", required=True, help="directory with check_*.jsonl")
_ap.add_argument("--data-dir", required=True, help="directory with dataset jsonl (kwargs)")
_ap.add_argument("--out", default="boundary_summary.json")
_args = _ap.parse_args()

PROX11 = {
    "keywords:existence", "keywords:frequency", "keywords:letter_frequency",
    "length_constraints:number_sentences", "length_constraints:number_paragraphs",
    "length_constraints:number_words", "detectable_content:number_placeholders",
    "detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections", "change_case:capital_word_frequency"}


def read_jsonl(p, limit=None):
    out = []
    with open(p, encoding="utf-8") as f:
        for i, l in enumerate(f):
            if not l.strip():
                continue
            if limit and i >= limit:
                break
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def listdir_nfc(d):
    # NFC/NFD-safe listing (evaluation files carry non-ASCII set-name tags)
    if not os.path.isdir(d):
        return []
    return [(unicodedata.normalize("NFC", fn), os.path.join(d, fn)) for fn in os.listdir(d)]


W = _args.data_dir
EV = os.path.dirname(os.path.abspath(_args.out)) or "."
import hashlib

# --- 0) kwargs map: scan dataset jsonl files (prompt -> {iid: kwargs}) ---
KW = {}
for d in (W,):
    for nfc, path in listdir_nfc(d):
        if not nfc.endswith('.jsonl') or nfc.startswith(('check_', 'rwlog_')): continue
        if os.path.getsize(path) > 30_000_000: continue
        try:
            head = next((l for l in open(path, encoding='utf-8') if l.strip()), '')
            r0 = json.loads(head)
        except Exception: continue
        if not (isinstance(r0, dict) and 'kwargs' in r0 and 'instruction_id_list' in r0 and 'prompt' in r0):
            continue
        for r in read_jsonl(path):
            p, ids, kws = r.get('prompt'), r.get('instruction_id_list'), r.get('kwargs')
            if p and ids and kws and len(ids) == len(kws):
                KW.setdefault(p, dict(zip(ids, kws)))
print(f'kwargs map: {len(KW)} prompts (from dataset jsonl)')

# --- 1) lightweight observation (same rules as _observe in the reward code; words/sentences approximate) ---
def obs_lite(iid, kw, resp):
    try:
        if iid == 'length_constraints:number_words':
            return len(re.findall(r"\w+", resp)), kw.get('num_words'), kw.get('relation'), True
        if iid == 'length_constraints:number_sentences':
            return len([x for x in re.split(r'[.!?]+', resp) if x.strip()]), kw.get('num_sentences'), kw.get('relation'), True
        if iid == 'length_constraints:number_paragraphs':
            ps = re.split(r'\s?\*\*\*\s?', resp)
            return len([p for p in ps if p.strip()]), kw.get('num_paragraphs'), 'exactly', False
        if iid == 'detectable_format:number_bullet_lists':
            n = len(re.findall(r'^\s*\*[^\*].*$', resp, flags=re.M)) + len(re.findall(r'^\s*-.*$', resp, flags=re.M))
            return n, kw.get('num_bullets'), 'exactly', False
        if iid == 'detectable_content:number_placeholders':
            return len(re.findall(r'\[.*?\]', resp)), kw.get('num_placeholders'), 'at least', False
        if iid == 'detectable_format:number_highlighted_sections':
            n = len([x for x in re.findall(r'\*[^\n\*]*\*', resp) if x.strip('*').strip()])
            n += len([x for x in re.findall(r'\*\*[^\n\*]*\*\*', resp) if x.strip('*').strip()])
            return n, kw.get('num_highlights'), 'at least', False
        if iid == 'detectable_format:multiple_sections':
            sp = kw.get('section_spliter', 'Section')
            return len(re.split(re.escape(sp) + r'\s?\d+', resp)) - 1, kw.get('num_sections'), 'at least', False
        if iid == 'change_case:capital_word_frequency':
            n = sum(1 for w in re.findall(r'\b[A-Z]+\b', resp) if len(w) > 1)
            return n, kw.get('capital_frequency'), kw.get('capital_relation'), False
        if iid == 'keywords:frequency':
            k = str(kw.get('keyword', ''))
            return (len(re.findall(re.escape(k), resp, flags=re.I)) if k else None), kw.get('frequency'), kw.get('relation'), False
        if iid == 'keywords:letter_frequency':
            l = str(kw.get('letter', ''))
            return (resp.lower().count(l.lower()) if l else None), kw.get('let_frequency'), kw.get('let_relation'), False
        if iid == 'keywords:existence':
            ks = kw.get('keywords') or []
            return sum(1 for k in ks if re.search(re.escape(k), resp, flags=re.I)), len(ks), 'at least', False
    except Exception:
        return None
    return None

RESP_KEYS = ('response', 'resp', 'output', 'completion', 'text')
def get_resp(r):
    for k in RESP_KEYS:
        if isinstance(r.get(k), str) and r[k].strip(): return r[k]
    return None

# raw evaluation files are tagged with these set names
SETS_G = {'그들500': 'mdp500', '우리500': 'assembled500', 'ifeval': 'ifeval'}
_IDXG = {}
for _d in (_args.check_dir,):
    for nfc, path in listdir_nfc(_d):
        _IDXG.setdefault(nfc, path)

def scan(var, seeds=(42, 43, 44, 45, 46)):
    """Collect per-constraint-instance relative misses and raw material for the case studies."""
    stats = collections.defaultdict(list)   # 'exact'/'atleast' -> list of rel_err
    sat_ct = collections.Counter(); rec_by_prompt = {}
    n_resp = 0; miss_kw = 0; schema_warned = False
    for sd in seeds:
        for ko in SETS_G:
            p = _IDXG.get(f'check_{var}_seed{sd}_{ko}.jsonl') or _IDXG.get(f'check_{ko}_{var}_seed{sd}.jsonl')
            if not p: continue
            for r in read_jsonl(p):
                ids, fl = r.get('instruction_id_list'), r.get('follow_instruction_list')
                resp, pr = get_resp(r), r.get('prompt')
                if not ids or fl is None or len(ids) != len(fl):
                    continue
                if resp is None or pr is None:
                    if not schema_warned:
                        print(f'  ! {var}: response/prompt field missing; sample keys: {sorted(r.keys())[:12]}')
                        schema_warned = True
                    continue
                kwmap = KW.get(pr)
                if kwmap is None:
                    miss_kw += 1; continue
                n_resp += 1
                for iid, ok in zip(ids, fl):
                    if iid not in PROX11: continue
                    o = obs_lite(iid, kwmap.get(iid, {}), resp)
                    if not o or o[0] is None or o[1] is None: continue
                    ov, tv, rel, approx = o
                    tv = float(tv)
                    if tv <= 0: continue
                    e = abs(ov - tv) / tv
                    rel_l = str(rel or 'exactly').lower()
                    kind = 'exact' if rel_l in ('exactly', '') else ('atleast' if 'least' in rel_l or 'more' in rel_l else 'atmost')
                    stats[kind].append(e)
                    sat_ct[(iid, bool(ok))] += 1
                    if sd == 42:
                        rec_by_prompt.setdefault(pr, []).append(
                            dict(iid=iid, o=ov, t=tv, rel=rel, ok=bool(ok), approx=approx,
                                 resp=resp[:280]))
    return stats, sat_ct, rec_by_prompt, n_resp, miss_kw

outG = {'note': 'word/sentence observations are approximate (approx=True); |o-t| distribution narrative only'}
scans = {}
for var in ('grpo', 'tb_tie', 'tb_mplacebo'):
    stats, sat_ct, recs, n_resp, miss_kw = scan(var)
    scans[var] = recs
    row = {}
    for kind, es in stats.items():
        if not es: continue
        es_s = sorted(es)
        row[kind] = dict(n=len(es), exact_hit=round(100*sum(1 for e in es if e == 0)/len(es), 1),
                         within10=round(100*sum(1 for e in es if e <= 0.10)/len(es), 1),
                         within25=round(100*sum(1 for e in es if e <= 0.25)/len(es), 1),
                         med_rel_err=round(es_s[len(es_s)//2], 3))
    outG[var] = dict(responses=n_resp, kw_miss=miss_kw, boundary=row)
    print(f'== {var:12s} responses {n_resp} (kwargs unmatched {miss_kw})')
    for kind, v in row.items():
        print(f'    {kind:8s} n={v["n"]:>5}  exact {v["exact_hit"]}%  within10 {v["within10"]}%  within25 {v["within25"]}%  median_rel_err {v["med_rel_err"]}')

# --- 2) representative cases: same prompt (seed 42), GRPO fails -> PTB succeeds (diverse classes, up to 6) ---
cases, used_cls = [], set()
fails = []
g_rec, t_rec = scans.get('grpo', {}), scans.get('tb_tie', {})
for pr in t_rec:
    if pr not in g_rec: continue
    t_by = {c['iid']: c for c in t_rec[pr]}
    for gcase in g_rec[pr]:
        iid = gcase['iid']
        tc = t_by.get(iid)
        if tc is None: continue
        if (not gcase['ok']) and tc['ok'] and iid not in used_cls and len(cases) < 6:
            used_cls.add(iid)
            cases.append(dict(cls=iid, prompt=pr[:200], target=gcase['t'], rel=gcase['rel'],
                              grpo=dict(o=gcase['o'], ok=False, resp=gcase['resp']),
                              ptb=dict(o=tc['o'], ok=True, resp=tc['resp'])))
        if (not tc['ok']) and abs(tc['o'] - tc['t']) <= max(1, 0.15 * tc['t']) and len(fails) < 3:
            fails.append(dict(cls=iid, prompt=pr[:200], target=tc['t'], rel=tc['rel'],
                              ptb=dict(o=tc['o'], ok=False, resp=tc['resp'])))
outG['cases_success'] = cases
outG['cases_nearmiss'] = fails
print(f'\n== success cases (GRPO fail -> PTB pass): {len(cases)} (classes: {sorted(used_cls)})')
for c in cases:
    print(f"  [{c['cls']}] target {c['target']:g} ({c['rel']}) -- GRPO o={c['grpo']['o']} FAIL -> PTB o={c['ptb']['o']} PASS")
print(f'== PTB near-miss cases: {len(fails)}')
for c in fails:
    print(f"  [{c['cls']}] target {c['target']:g} -- PTB o={c['ptb']['o']} FAIL (near boundary)")

pG = _args.out
json.dump(outG, open(pG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nsaved:', pG)
