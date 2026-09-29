# Generates Figs. 2-5 of the paper from the run outputs.
# =========================================================
# §K  FIG.2 ~ FIG.5 일괄 생성 → ahn2~ahn5 (PNG+PDF) → Drive
#     이 셀 하나로 끝. 몇 번을 다시 돌려도 안전합니다.
# =========================================================
import os, sys, json, glob, shutil, subprocess, collections
from pathlib import Path
import numpy as np

# ---- Drive ---------------------------------------------------------
try:
    pass
except Exception as e:
    print('[drive]', e)

# ---- 폰트 + matplotlib 규격 ----------------------------------------
subprocess.run('apt-get -qq install -y fonts-liberation', shell=True, capture_output=True)
import matplotlib, matplotlib.pyplot as plt, matplotlib.font_manager as fm
try: fm._load_fontmanager(try_read_cache=False)
except Exception: fm.fontManager = fm.FontManager()
_have = {f.name for f in fm.fontManager.ttflist}
_PICKF = next((n for n in ['Arial','Liberation Sans','Helvetica','DejaVu Sans'] if n in _have),
              'DejaVu Sans')

COL_W_IN, PAGE_W_IN, BASE_PT = 3.31, 6.93, 8      # 8.4cm / 17.6cm / 8pt
matplotlib.rcParams.update({
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'font.family':'sans-serif', 'font.sans-serif':[_PICKF],
    'font.size':BASE_PT, 'axes.labelsize':BASE_PT, 'axes.titlesize':BASE_PT,
    'xtick.labelsize':BASE_PT, 'ytick.labelsize':BASE_PT, 'legend.fontsize':BASE_PT,
    'axes.linewidth':.6, 'grid.linewidth':.4,
    'xtick.major.width':.6, 'ytick.major.width':.6,
    'lines.linewidth':1.1, 'lines.markersize':3.0,
    'axes.spines.top':False, 'axes.spines.right':False,
    'figure.dpi':130, 'savefig.dpi':600,
    'savefig.bbox':'tight', 'savefig.pad_inches':.02,
})

# ---- 경로 ----------------------------------------------------------
_DEFAULT = 'WORK_ROOT  # set to your working directory'
ROOT = Path(_DEFAULT)
if not ROOT.exists():
    _h = glob.glob('.', recursive=True)
    ROOT = Path(_h[0]) if _h else None
assert ROOT is not None, 'mdp-grpo 폴더를 찾지 못했습니다'
RUNS, EVAL = ROOT/'runs', ROOT/'eval'
OUT = Path('/content/fig_out'); OUT.mkdir(exist_ok=True)
print('ROOT:', ROOT, '| 폰트:', _PICKF)

# ---- 색 (FIG.1 과 통일) --------------------------------------------
_AMB, _TEA, _DAF, _PLC = '#E08A2E', '#2E7D96', '#5A5A5A', '#BDBDBD'
_LBL = {'grpo':'GRPO','da_half':'DA-fixed','tb_placebo':'Placebo','tb_tie':'PTB'}
_CLR = {'grpo':_AMB,'da_half':_DAF,'tb_placebo':_PLC,'tb_tie':_TEA}
_STY = {'grpo':'-','da_half':'--','tb_tie':'-.'}
_MRK = {'tb_tie':'o','tb_placebo':'s','da_half':'^'}
_BAND = 'range'            # FIG.3 밴드: 'range'(min-max) 또는 'sd'

def _load(p):
    try: return json.load(open(p, encoding='utf-8'))
    except Exception: return None

def _save(fig, tag):
    for e in ('pdf','png'): fig.savefig(OUT/f'ahn{tag}.{e}')
    w,h = fig.get_size_inches(); print(f'  ahn{tag}   {w*2.54:.2f} x {h*2.54:.2f} cm')
    plt.show(); plt.close(fig)

_agg = {n: _load(EVAL/n) for n in
        ('diff_test_for_paper.json','eval_all.json') if (EVAL/n).exists()}
_ev, _dt = _agg.get('eval_all.json') or {}, _agg.get('diff_test_for_paper.json') or {}

_PREFIX = ['gemma3_1b','gemma3_4b','llama1b','qwen3b','llama','qwen','gemma']
def _parse(tag):
    t, model = tag, 'gemma'
    for m in _PREFIX:
        if t.startswith(m+'_'): model, t = m, t[len(m)+1:]; break
        if t == m: model, t = m, ''; break
    seed = None
    if '_seed' in t:
        t, s = t.rsplit('_seed', 1); seed = int(s) if s.isdigit() else s
    return model, t, seed

def _cells(model, var, bench, field='ssr_macro'):
    out = []
    for k, d in _ev.items():
        p3 = k.split('|')
        if len(p3) < 3 or p3[1] != bench or p3[2] != 'greedy': continue
        m, v, _ = _parse(p3[0])
        if m == model and v == var and isinstance(d, dict) and field in d:
            out.append(d[field])
    return np.array(out, float)

# ================================================== FIG.2
print('='*66); print('FIG.2  C별 동점률 / 갈림 비율'); print('='*66)
_C     = np.arange(1, 7)
_TIED  = np.array([98.2, 96.8, 97.1, 95.2, 95.2, 95.3])
_SPLIT = np.array([16.4, 21.4, 31.8, 42.9, 60.9, 73.8])
fig, ax = plt.subplots(figsize=(COL_W_IN, 1.85)); w = .38
ax.bar(_C-w/2, _TIED,  w, color='#C9C9C9', edgecolor='#6E6E6E', lw=.5, label='tied')
ax.bar(_C+w/2, _SPLIT, w, color=_TEA, edgecolor=_TEA, lw=.5, hatch='///', label='separable')
ax.set_xlabel('Constraints per prompt'); ax.set_ylabel('Rollouts (%)')
ax.set_xticks(_C); ax.set_ylim(0, 122)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=2,
          handlelength=1.2, borderpad=.2, columnspacing=1.0, handletextpad=.4)
ax.grid(axis='y', color='#DDDDDD', lw=.4); ax.set_axisbelow(True)
_save(fig, 2)

# ================================================== FIG.3
print(); print('='*66); print('FIG.3  엔트로피 / KL  (세 계열 n=3)'); print('='*66)
_PICK = {'grpo':[42,45,46], 'da_half':[42,43,44], 'tb_tie':[42,45,46]}
def _clean(p):
    d = _load(p)
    return d if (isinstance(d, dict) and isinstance(d.get('epoch'), list)
                 and len(d['epoch']) == 76) else None
_SER = {}
for var, seeds in _PICK.items():
    got = [(s, d) for s in seeds if (d := _clean(RUNS/f'metrics_{var}_seed{s}.json')) is not None]
    _SER[var] = got; print(f'  {var:10s} n={len(got)}  seeds={[s for s,_ in got]}')
def _curve(d, key):
    v = d.get(key)
    if not isinstance(v, list): return None, None
    return (np.array([p[0] for p in v], float), np.array([p[1] for p in v], float))
fig, axes = plt.subplots(1, 2, figsize=(PAGE_W_IN, 2.35))
for ax, key, ylab in zip(axes, ['entropy','kl'], ['Token entropy','KL to reference policy']):
    for var in ('grpo','da_half','tb_tie'):
        xs, ys = [], []
        for s, d in _SER[var]:
            x, y = _curve(d, key)
            if x is not None: xs.append(x); ys.append(y)
        if not ys: continue
        L = min(len(y) for y in ys)
        x = xs[0][:L]; Y = np.vstack([y[:L] for y in ys]); m = Y.mean(0)
        lo, hi = (Y.min(0), Y.max(0)) if _BAND=='range' else (m-Y.std(0,ddof=1), m+Y.std(0,ddof=1))
        ax.fill_between(x, lo, hi, color=_CLR[var], alpha=.18, lw=0)
        ax.plot(x, m, _STY[var], color=_CLR[var], label=_LBL[var])
    ax.set_xlabel('Training step'); ax.set_ylabel(ylab)
    ax.grid(color='#EEEEEE', lw=.4); ax.set_axisbelow(True)
axes[0].legend(frameon=False, handlelength=1.8, borderpad=.2, labelspacing=.3)
fig.tight_layout(pad=.2); _save(fig, 3)
for var in ('grpo','da_half','tb_tie'):
    for key in ('entropy','kl'):
        v = [_curve(d, key)[1][-1] for _, d in _SER[var]]
        print(f'    {_LBL[var]:9s} {key:8s} 종점 {np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}')

# ================================================== FIG.4
print(); print('='*66); print('FIG.4  셀별 차등값 분포'); print('='*66)
_cw     = _dt.get('셀별') or []
_pooled = {r['변형']: r['차이'] for r in (_dt.get('통합검정') or [])}
_ORDER  = [('tb_tie','PTB',_TEA,'o'), ('tb_placebo','Placebo',_PLC,'s'),
           ('da_half','DA-fixed',_DAF,'^')]
_rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(COL_W_IN, 2.3))
ax.axhline(0, color='#555555', lw=.8, zorder=1)
for i, (key, lab, col, mk) in enumerate(_ORDER):
    v = np.array([r['차이'] for r in _cw if r['변형']==key], float)
    ax.scatter(i + _rng.uniform(-.13,.13,len(v)), v, s=13, marker=mk,
               facecolor=col, edgecolor='#444444', linewidth=.4, zorder=3)
    p = _pooled.get(key)
    if p is not None: ax.plot([i-.28, i+.28], [p, p], color=col, lw=2.0, zorder=4)
    print(f'  {lab:9s} n={len(v):2d}  양수 {int((v>0).sum())}/{len(v)}  pooled {p:+.2f}')
ax.set_xticks(range(3)); ax.set_xticklabels([l for _,l,_,_ in _ORDER])
ax.set_ylabel('Covered - uncovered\n(percentage points)'); ax.set_xlim(-.55, 2.55)
ax.grid(axis='y', color='#EEEEEE', lw=.4); ax.set_axisbelow(True)
fig.tight_layout(pad=.2); _save(fig, 4)

# ================================================== FIG.5
print(); print('='*66); print('FIG.5  12칸 Δ + 표준오차'); print('='*66)
_MR = [('gemma','Gemma-2-2B'), ('gemma3_1b','Gemma-3-1B'),
       ('llama','Llama-3.2-3B'), ('llama1b','Llama-3.2-1B')]
_BR = [('theirs','Their set'), ('ours','Ours'), ('ifeval','IFEval')]
_xs = np.array([j*3.6+i for j in range(4) for i in range(3)], float)
_labs = [b for _ in _MR for _, b in _BR]
fig, ax = plt.subplots(figsize=(PAGE_W_IN, 2.7))
ax.axhline(0, color=_AMB, lw=1.2, zorder=1)
for var, off in (('tb_tie',-.17), ('tb_placebo',0.), ('da_half',.17)):
    D, E = [], []
    for mk_, _ in _MR:
        for bk, _ in _BR:
            g, t = _cells(mk_,'grpo',bk), _cells(mk_,var,bk)
            D.append(t.mean()-g.mean() if len(g) and len(t) else np.nan)
            E.append(np.hypot(g.std(ddof=1)/np.sqrt(len(g)), t.std(ddof=1)/np.sqrt(len(t)))
                     if len(g)>1 and len(t)>1 else np.nan)
    ax.errorbar(_xs+off, D, yerr=E, fmt=_MRK[var], ms=3.4, lw=0, elinewidth=.8,
                capsize=1.6, color=_CLR[var], ecolor=_CLR[var], label=_LBL[var],
                markeredgecolor='#444444', markeredgewidth=.4, zorder=3)
    print(f'  {_LBL[var]:9s}', np.round(D, 2))
ax.set_xticks(_xs); ax.set_xticklabels(_labs, rotation=38, ha='right')
ax.set_ylabel('SSR difference vs GRPO\n(percentage points)')
ax.set_xlim(-.9, 3*3.6+2.9)
_lo, _hi = ax.get_ylim(); ax.set_ylim(_lo, _hi + (_hi-_lo)*.12)
for j, (_, ml) in enumerate(_MR):
    ax.text(j*3.6+1, ax.get_ylim()[1]-(_hi-_lo)*.03, ml, ha='center', va='top', fontsize=BASE_PT)
    if j: ax.axvline(j*3.6-1.3, color='#DDDDDD', lw=.6)
_h, _l = ax.get_legend_handles_labels()
_h.append(plt.Line2D([0],[0], color=_AMB, lw=1.2)); _l.append('GRPO baseline')
ax.legend(_h, _l, frameon=False, ncol=4, handlelength=1.2, borderpad=.2,
          columnspacing=1.4, loc='lower center', bbox_to_anchor=(.5, 1.04))
ax.grid(axis='y', color='#EEEEEE', lw=.4); ax.set_axisbelow(True)
fig.tight_layout(pad=.2); _save(fig, 5)

# ================================================== Drive
print(); print('='*66); print('Drive 저장'); print('='*66)
_DEST = ROOT/'figures'; _DEST.mkdir(exist_ok=True)
for p in list(_DEST.iterdir()):
    if p.name.startswith('kim') or p.stem in ('ahn3a','ahn3b'): p.unlink()
_n = 0
for p in sorted(OUT.iterdir()):
    if p.suffix.lower() in ('.pdf','.png') and p.name.startswith('ahn'):
        shutil.copy2(p, _DEST/p.name); _n += 1
print(f'  {_n}개 → {_DEST}')
for p in sorted(_DEST.iterdir()):
    print(f'    {p.name:14s} {p.stat().st_size:>9,} B')
print('\n  Word 삽입 = PNG (한 단 8.4cm / 페이지폭 17.6cm) · 투고 = PDF')
print('  ※ ahn1 은 draw.io 산출물 — 직접 figures/ 에 올려두십시오.')
