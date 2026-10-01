#!/usr/bin/env python3
"""Export the per-instance boundary table (results/boundary_instances.csv.gz) and the
failed-only breakdown (results/boundary_failed_breakdown.json) from the raw per-response
evaluation outputs (check_*.jsonl; ~600 MB, available from the authors on request).

Scope: Gemma-2-2B; variants grpo, tb_tie, tb_mplacebo; seeds 42-46; benchmarks
MDP-500 and Assembled-500 (the two sets shared by all three variants). Counts use the
approximate recomputation obs_lite_v1_approx, identical to stats/boundary_analysis.py.

Usage: python stats/export_boundary_instances.py --check-dir DIR --data-dir DIR \
          [--out-csv PATH] [--out-breakdown PATH]
"""
import argparse, collections, csv, gzip, hashlib, json, os, re, sys, unicodedata

ap=argparse.ArgumentParser()
ap.add_argument('--check-dir', required=True)
ap.add_argument('--data-dir', required=True)
ap.add_argument('--out-csv', default='boundary_instances.csv.gz')
ap.add_argument('--out-breakdown', default='boundary_failed_breakdown.json')
A=ap.parse_args()
for d in (A.check_dir, A.data_dir):
    if not os.path.isdir(d): sys.exit(f'input directory not found: {d}')

def read_jsonl(p):
    out=[]
    for l in open(p, encoding='utf-8'):
        if l.strip():
            try: out.append(json.loads(l))
            except Exception: pass
    return out

def listing(d):
    return {unicodedata.normalize('NFC',fn): os.path.join(d,fn) for fn in os.listdir(d)}

# kwargs map: any jsonl in data-dir carrying prompt/instruction_id_list/kwargs
KW={}
for nfc,p in listing(A.data_dir).items():
    if not nfc.endswith('.jsonl') or nfc.startswith(('check_','rwlog_')): continue
    try: r0=json.loads(next(l for l in open(p,encoding='utf-8') if l.strip()))
    except Exception: continue
    if not (isinstance(r0,dict) and 'kwargs' in r0 and 'instruction_id_list' in r0 and 'prompt' in r0): continue
    for r in read_jsonl(p):
        pr,ids,kws=r.get('prompt'),r.get('instruction_id_list'),r.get('kwargs')
        if pr and ids and kws and len(ids)==len(kws): KW.setdefault(pr,dict(zip(ids,kws)))
if not KW: sys.exit('no kwargs source found in --data-dir')

PROX11={"keywords:existence","keywords:frequency","keywords:letter_frequency",
 "length_constraints:number_sentences","length_constraints:number_paragraphs",
 "length_constraints:number_words","detectable_content:number_placeholders",
 "detectable_format:number_bullet_lists","detectable_format:number_highlighted_sections",
 "detectable_format:multiple_sections","change_case:capital_word_frequency"}

def obs_lite(iid,kw,resp):
    try:
        if iid=='length_constraints:number_words': return len(re.findall(r"\w+",resp)),kw.get('num_words'),kw.get('relation')
        if iid=='length_constraints:number_sentences': return len([x for x in re.split(r'[.!?]+',resp) if x.strip()]),kw.get('num_sentences'),kw.get('relation')
        if iid=='length_constraints:number_paragraphs': return len([q for q in re.split(r'\s?\*\*\*\s?',resp) if q.strip()]),kw.get('num_paragraphs'),'exactly'
        if iid=='detectable_format:number_bullet_lists':
            n=len(re.findall(r'^\s*\*[^\*].*$',resp,flags=re.M))+len(re.findall(r'^\s*-.*$',resp,flags=re.M)); return n,kw.get('num_bullets'),'exactly'
        if iid=='detectable_content:number_placeholders': return len(re.findall(r'\[.*?\]',resp)),kw.get('num_placeholders'),'at least'
        if iid=='detectable_format:number_highlighted_sections':
            n=len([x for x in re.findall(r'\*[^\n\*]*\*',resp) if x.strip('*').strip()])+len([x for x in re.findall(r'\*\*[^\n\*]*\*\*',resp) if x.strip('*').strip()]); return n,kw.get('num_highlights'),'at least'
        if iid=='detectable_format:multiple_sections':
            sp=kw.get('section_spliter','Section'); return len(re.split(re.escape(sp)+r'\s?\d+',resp))-1,kw.get('num_sections'),'at least'
        if iid=='change_case:capital_word_frequency':
            return sum(1 for w in re.findall(r'\b[A-Z]+\b',resp) if len(w)>1),kw.get('capital_frequency'),kw.get('capital_relation')
        if iid=='keywords:frequency':
            k=str(kw.get('keyword','')); return (len(re.findall(re.escape(k),resp,flags=re.I)) if k else None),kw.get('frequency'),kw.get('relation')
        if iid=='keywords:letter_frequency':
            l=str(kw.get('letter','')); return (resp.lower().count(l.lower()) if l else None),kw.get('let_frequency'),kw.get('let_relation')
        if iid=='keywords:existence':
            ks=kw.get('keywords') or []; return sum(1 for k in ks if re.search(re.escape(k),resp,flags=re.I)),len(ks),'at least'
    except Exception: return None
    return None

SETS={'그들500':'mdp500','우리500':'assembled500'}
RESP=('response','resp','output','completion','text')
idx=listing(A.check_dir)
rows=[]; drop=collections.Counter(); files=0
for var in ('grpo','tb_tie','tb_mplacebo'):
    for sd in (42,43,44,45,46):
        for ko,bn in SETS.items():
            p=idx.get(f'check_{var}_seed{sd}_{ko}.jsonl') or idx.get(f'check_{ko}_{var}_seed{sd}.jsonl')
            if not p: continue
            files+=1
            for r in read_jsonl(p):
                ids,fl,pr=r.get('instruction_id_list'),r.get('follow_instruction_list'),r.get('prompt')
                resp=next((r[k] for k in RESP if isinstance(r.get(k),str) and r[k].strip()),None)
                if not ids or fl is None or len(ids)!=len(fl): drop['schema']+=1; continue
                if resp is None: drop['empty_response']+=1; continue
                kwm=KW.get(pr)
                if kwm is None: drop['no_kwargs']+=1; continue
                ph=hashlib.md5(pr.encode()).hexdigest()[:10]
                for iid,ok in zip(ids,fl):
                    if iid not in PROX11: continue
                    o=obs_lite(iid,kwm.get(iid,{}),resp)
                    if not o or o[0] is None or o[1] is None: drop['no_count']+=1; continue
                    rows.append(('gemma-2-2b',var,sd,bn,ph,iid,str(o[2] or 'exactly'),float(o[1]),float(o[0]),int(bool(ok))))
if files==0: sys.exit('no check files matched in --check-dir')
with gzip.open(A.out_csv,'wt',newline='') as f:
    w=csv.writer(f); w.writerow(['model','variant','seed','benchmark','prompt_hash','constraint_id','relation','target','observed_count','verifier_pass','counter_version'])
    for r in rows: w.writerow(list(r)+['obs_lite_v1_approx'])

def kindof(rel):
    rl=rel.lower(); return 'exact' if rl in('exactly','') else ('atleast' if ('least' in rl or 'more' in rl) else 'atmost')
def med(v):
    v=sorted(v); n=len(v)
    return (v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2)
def summ(es):
    n=len(es)
    return dict(n=n, exact_hit=round(100*sum(1 for e in es if e==0)/n,2), within25=round(100*sum(1 for e in es if e<=0.25)/n,2), med=round(med(es),4)) if n else {}
BD={'scope':'mdp500+assembled500 (the two benchmarks shared by all three variants)',
    'counter_version':'obs_lite_v1_approx','files_read':files,'export_drops':dict(drop)}
for var in ('grpo','tb_tie','tb_mplacebo'):
    BD[var]={}
    for fo in ('all','failed'):
        agg=collections.defaultdict(list); per_seed=collections.defaultdict(list)
        for (_m,v,sd,bn,_p,_c,rel,t,o,ok) in rows:
            if v!=var or t<=0: continue
            if fo=='failed' and ok: continue
            e=abs(o-t)/t; k=kindof(rel); agg[k].append(e); per_seed[(k,sd)].append(e)
        BD[var][fo]={k:summ(x) for k,x in agg.items()}
        BD[var][fo]['per_seed_median']={f'{k}_s{sd}':round(med(x),4) for (k,sd),x in sorted(per_seed.items())}
json.dump(BD,open(A.out_breakdown,'w'),indent=1)
print(f'rows={len(rows)} files={files} drops={dict(drop)}')
for var in ('grpo','tb_tie','tb_mplacebo'):
    for k in ('exact','atleast','atmost'):
        s=BD[var]['failed'].get(k,{})
        if s: print(f'  FAILED {var:12s} {k:8s} n={s["n"]:>5} med={s["med"]} w25={s["within25"]}%')
