#!/usr/bin/env python3
"""Response-length (whitespace words) and truncation summaries for PTB and GRPO.
Runs on the raw per-response evaluation outputs (check_*.jsonl; available on request).
Truncation is counted as generated length >= 1024 tokens under each model's own tokenizer.
Usage: python stats/length_truncation.py --check-dir DIR [--out PATH]
"""
import argparse, json, os, re, statistics, unicodedata
ap=argparse.ArgumentParser(); ap.add_argument('--check-dir',required=True); ap.add_argument('--out',default='length_truncation_summary.json')
A=ap.parse_args()
from transformers import AutoTokenizer
MODELS={'':'google/gemma-2-2b-it','gemma3_1b':'google/gemma-3-1b-it','llama':'meta-llama/Llama-3.2-3B-Instruct','llama1b':'meta-llama/Llama-3.2-1B-Instruct'}
TOKS={k:AutoTokenizer.from_pretrained(v, token=os.environ.get('HF_TOKEN')) for k,v in MODELS.items()}
SETS={'그들500':'mdp500','우리500':'assembled500'}
RESP=('response','resp','output','completion','text')
def rows(p):
    for l in open(p,encoding='utf-8'):
        if l.strip():
            try: yield json.loads(l)
            except Exception: pass
out={}
agg={}
for fn in sorted(os.listdir(A.check_dir)):
    n=unicodedata.normalize('NFC',fn)
    m=re.match(r'^check_(?:([a-z0-9_]+)_)?(tb_tie|grpo)_seed(\d+)_(그들500|우리500)\.jsonl$',n)
    if not m: continue
    mk,var,sd,ko=m.group(1) or '',m.group(2),int(m.group(3)),m.group(4)
    if mk not in MODELS: continue
    tok=TOKS[mk]
    for r in rows(os.path.join(A.check_dir,fn)):
        resp=next((r[k] for k in RESP if isinstance(r.get(k),str) and r[k].strip()),None)
        if resp is None: continue
        w=len(resp.split()); t=len(tok(resp,add_special_tokens=False)['input_ids'])
        key=(var,SETS[ko]); agg.setdefault(key,{'w':[],'t':[]}); agg[key]['w'].append(w); agg[key]['t'].append(t)
        key2=(var,SETS[ko],mk or 'gemma'); agg.setdefault(key2,{'w':[],'t':[]}); agg[key2]['w'].append(w); agg[key2]['t'].append(t)
for key,d in sorted(agg.items(), key=lambda x:str(x[0])):
    W,T=d['w'],d['t']
    out['|'.join(map(str,key))]=dict(n=len(W), words_mean=round(sum(W)/len(W),1), words_median=statistics.median(W),
        under20w_pct=round(100*sum(1 for x in W if x<20)/len(W),1), tokens_mean=round(sum(T)/len(T),1),
        truncated_pct=round(100*sum(1 for x in T if x>=1024)/len(T),2))
json.dump(out,open(A.out,'w'),indent=1)
for k in ('tb_tie|mdp500','tb_tie|assembled500','grpo|mdp500','grpo|assembled500'):
    print(k,out.get(k))
print('saved',A.out)
