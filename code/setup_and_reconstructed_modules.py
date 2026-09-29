# Reconstructed instruction registry / constants and run configuration.
# Extracted from the Colab notebooks used for all experiments in the paper.
import os, sys, json, time, re, shutil, subprocess, collections
import statistics as st

print("=" * 70)
print("  NOTEBOOK: mdpgrpo_train  r12   [A100 80GB · 최대 12시간]")
print("=" * 70)

WORK_ROOT = "WORK_ROOT  # set to your working directory"
REPO      = WORK_ROOT + "/MDP-GRPO-main"
DATA_DIR  = WORK_ROOT + "/data"
OUT_DIR   = WORK_ROOT + "/runs"          # 판번호와 무관. 모든 run 이 여기 쌓인다
THEIR_EVAL = REPO + "/data/data_test.jsonl"

LOCAL_SRC, LOCAL_TRN, LOCAL_RUN = "/content/src", "/content/training", "/content/run"

# ══════════════════ 여기 두 줄만 고치십시오 ══════════════════

MODEL_KEY = "llama1b"
RUNS = [("tb_placebo", 43), ("tb_placebo", 44)]

BUDGET_HOURS = 48.0        # 이 시간을 넘기면 새 run 을 시작하지 않는다

# 모델별 설정. gen_batch 는 VRAM 에 맞춘 값이고 나머지는 전 모델 공통이다.

PRESETS = {
    "gemma": dict(model_id="google/gemma-2-2b-it",
                  gen_batch=128, est_h=3.4, prefix=""),
    "qwen":  dict(model_id="Qwen/Qwen2.5-1.5B-Instruct",
                  gen_batch=192, est_h=4.2, prefix="qwen_"),
    "llama": dict(model_id="meta-llama/Llama-3.2-3B-Instruct",
                  gen_batch=96, est_h=5.0, prefix="llama_"),
    "llama1b": dict(model_id="meta-llama/Llama-3.2-1B-Instruct",
                    gen_batch=256, est_h=2.5, prefix="llama1b_"),
    "qwen3b":  dict(model_id="Qwen/Qwen2.5-3B-Instruct",
                    gen_batch=96, est_h=5.0, prefix="qwen3b_"),
    "gemma3_1b": dict(model_id="google/gemma-3-1b-it",
                      gen_batch=256, est_h=2.5, prefix="gemma3_1b_"),
    "gemma3_4b": dict(model_id="google/gemma-3-4b-it",
                      gen_batch=96, est_h=5.5, prefix="gemma3_4b_"),
    "qwen3_17b": dict(model_id="Qwen/Qwen3-1.7B",
                      gen_batch=192, est_h=3.0, prefix="qwen3_17b_"),
}

assert MODEL_KEY in PRESETS, f"MODEL_KEY 는 {list(PRESETS)} 중 하나"
PS = PRESETS[MODEL_KEY]
MODEL_ID = PS["model_id"]
PREFIX   = PS["prefix"]          # gemma 는 기존 산출물과 이름을 맞추려 접두사 없음
print(f"\n모델  {MODEL_KEY}  ({MODEL_ID})")
print(f"산출물 접두사  '{PREFIX}'    run 당 예상 {PS['est_h']:.1f}시간")

# 아래는 참고용 — 다른 변형을 돌릴 때 RUNS 에 넣으면 된다
# ("tb_prox", 42)     붕괴 그룹만 (v1)
# ("tb_placebo", 43)  위약군
# ("da", 42) ("pt", 42) ("mt", 42)   논문 처방

# 변형별로 바뀌는 인자만 적는다. 나머지는 BASE_ARGS 공통
VARIANTS = {
    # 아래 셋은 train.py 인자가 GRPO 와 동일하다. 개입은 §C 의 보상 함수에서 일어난다.
    "tb_nz":      {},
    "tb_tie":     {},
    "tb_nz":      {},
    "tb_part":    {},
    "tb_supp":    {},
    "w_adapt":    {},
    "tb_v2":      {},
    "tb_placebo": {},
    "tb_prox":    {},
    "grpo":  {},                                              # 표준 GRPO
    "mt":    {"temperature_list": "0.1,0.4,0.7,1.0,0.1,0.4,0.7,1.0"},   # 다중 온도(G=8)
    "da":    {"DA_ALPHA": 0.2},
    "da_half": {"DA_ALPHA": 0.2, "DA_GOAL_MU_MODE": "half"},                         # 논문 본문 Eq.4 — 앵커 0.5 고정
                                                              #   (§B 하단에서 grpo_trainer 를 패치하고
                                                              #    DA_HALF 환경변수로 run 마다 전환한다)
    "pt":    {"DO_PROSPECT": True, "prospect_lambda_pos": 1.25,
              "prospect_lambda_neg": 2.0, "prospect_beta": 0.8},   # 논문 §5.5 값
    "da_pt": {"DA_ALPHA": 0.2, "DO_PROSPECT": True, "prospect_lambda_pos": 1.25,
              "prospect_lambda_neg": 2.0, "prospect_beta": 0.8},
    "akl":   {"beta_by_adv_sign": True, "beta_pos": 0.01, "beta_neg": 0.025},
}

BASE_ARGS = {
    "per_device_train_batch_size": 8, "gradient_accumulation_steps": 4,
    "num_generations": 8, "num_train_epochs": 1.0,
    "learning_rate": 1e-5, "temperature": 1.0, "top_p": 0.9, "top_k": 50,
    "max_prompt_length": 1024, "max_completion_length": 1024,
    "beta": 0.01, "epsilon": 0.2, "epsilon_high": 0.2, "num_iterations": 1,
    "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
    "DA_ALPHA": 0.0, "DO_PROSPECT": False, "beta_by_adv_sign": False,
    "importance_sampling_level": "token",
    "logging_steps": 10, "save_steps": 500, "save_total_limit": 1,
    "report_to": "mlflow",
}

GEN_BATCH_SIZE, MAX_MODEL_LEN, MAX_TOKENS = PS["gen_batch"], 2048, 1024
GPU_MEM_UTIL, DTYPE, EVAL_MODE = 0.9, "bfloat16", "strict"

assert os.path.isdir(REPO), f"저장소 없음: {REPO}"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOCAL_RUN, exist_ok=True)

import torch, transformers, peft, trl, dataclasses
print(f"torch {torch.__version__} | transformers {transformers.__version__} | "
      f"peft {peft.__version__} | trl {trl.__version__}")
assert torch.cuda.is_available(), "GPU 런타임이 아님"
_p = torch.cuda.get_device_properties(0)
_vram = _p.total_memory / 1e9
print(f"{_p.name}  VRAM {_vram:.0f}GB")
if _vram < 60:
    print("  ⚠ 40GB 에서는 OOM. 80GB 런타임을 쓰거나 배치를 8/4 → 4/8 로 바꿀 것")

from trl import GRPOConfig
_f = {x.name for x in dataclasses.fields(GRPOConfig)}
assert not ({"temperature_list", "advantage_mode", "prospect_enable",
             "beta_by_adv_sign"} - _f), "표준 TRL 이 깔려 있다 — §A 를 다시 볼 것"
print("TRL 포크 확인")

_r = subprocess.run([sys.executable, "-c", "import vllm;print(vllm.__version__)"],
                    capture_output=True, text=True)
assert _r.returncode == 0, "vllm import 실패:\n" + (_r.stderr or "")[-800:]
print(f"vllm {_r.stdout.strip()}")

# ---------------- 저장소 복사 + 없어진 두 모듈 복원 ----------------
shutil.rmtree(LOCAL_SRC, ignore_errors=True); shutil.copytree(os.path.join(REPO, "src"), LOCAL_SRC)
shutil.rmtree(LOCAL_TRN, ignore_errors=True); shutil.copytree(os.path.join(REPO, "training"), LOCAL_TRN)

CONSTANTS_SRC = '''"""instructions.py 가 요구하는 상수. 원본 저장소에 없어 복원했다."""
import collections
_EN = {
    "COMPARISON_RELATION": ("less than", "at least", "more than", "up to", "exactly"),
    "CONSTRAINED_RESPONSE_OPTIONS": (
        "My answer is yes.", "My answer is no.", "My answer is maybe."),
    "STARTER_OPTIONS": (
        "I would say", "My answer is", "I believe", "In my opinion", "I think",
        "I reckon", "I feel", "From my perspective", "As I see it", "According to me",
        "As far as I am concerned", "To my understanding", "In my view",
        "My take on it is", "As per my perception"),
    "ENDING_OPTIONS": ("Any other questions?", "Is there anything else I can help with?"),
    "SECTION_SPLITER": ("Section", "SECTION"),
    "POSTSCRIPT_MARKER": ("P.S.", "P.P.S"),
}
_Consts = collections.namedtuple("_Consts", sorted(_EN.keys()))
def get_instruction_constants(language="en"):
    if language != "en":
        raise NotImplementedError("복원본은 영어만 지원한다.")
    return _Consts(**_EN)
'''
REGISTRY_SRC = '''"""instruction_id -> Instruction class. 원본 저장소에 없어 복원했다."""
import instructions
INSTRUCTION_DICT = {
    "keywords:existence": instructions.KeywordChecker,
    "keywords:frequency": instructions.KeywordFrequencyChecker,
    "keywords:forbidden_words": instructions.ForbiddenWords,
    "keywords:letter_frequency": instructions.LetterFrequencyChecker,
    "language:response_language": instructions.ResponseLanguageChecker,
    "length_constraints:number_sentences": instructions.NumberOfSentences,
    "length_constraints:number_paragraphs": instructions.ParagraphChecker,
    "length_constraints:number_words": instructions.NumberOfWords,
    "length_constraints:nth_paragraph_first_word": instructions.ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": instructions.PlaceholderChecker,
    "detectable_content:postscript": instructions.PostscriptChecker,
    "detectable_format:number_bullet_lists": instructions.BulletListChecker,
    "detectable_format:constrained_response": instructions.ConstrainedResponseChecker,
    "detectable_format:number_highlighted_sections": instructions.HighlightSectionChecker,
    "detectable_format:multiple_sections": instructions.SectionChecker,
    "detectable_format:json_format": instructions.JsonFormat,
    "detectable_format:xml_format": instructions.XmlFormat,
    "detectable_format:title": instructions.TitleChecker,
    "combination:two_responses": instructions.TwoResponsesChecker,
    "combination:repeat_prompt": instructions.RepeatPromptThenAnswer,
    "startend:end_checker": instructions.EndChecker,
    "startend:quotation": instructions.QuotationChecker,
    "change_case:capital_word_frequency": instructions.CapitalWordFrequencyChecker,
    "change_case:english_capital": instructions.CapitalLettersEnglishChecker,
    "change_case:english_lowercase": instructions.LowercaseLettersEnglishChecker,
    "punctuation:no_comma": instructions.CommaChecker,
}
'''
for _n, _s in [("instruction_constants.py", CONSTANTS_SRC),
               ("instructions_registry.py", REGISTRY_SRC)]:
    with open(os.path.join(LOCAL_SRC, _n), "w", encoding="utf-8") as f:
        f.write(_s)
for _p2 in (LOCAL_SRC, LOCAL_TRN):
    if _p2 not in sys.path:
        sys.path.insert(0, _p2)
import instructions, instructions_registry
REG = instructions_registry.INSTRUCTION_DICT
print(f"복원 모듈 import 통과. 등록된 제약 {len(REG)}종")
os.environ["PYTHONPATH"] = LOCAL_SRC + ":" + LOCAL_TRN + ":" + os.environ.get("PYTHONPATH", "")

# ---------------- 변형 인자가 train.py 에 실재하는지 검증 ----------------
_src = open(os.path.join(LOCAL_TRN, "train.py"), encoding="utf-8").read()
_known = set(re.findall(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*:", _src, re.M))
_bad = []
for _v, _ov in VARIANTS.items():
    for _k in list(_ov) :
        if _k not in _known:
            _bad.append(f"{_v}/{_k}")
for _k in BASE_ARGS:
    if _k not in _known:
        _bad.append(f"BASE/{_k}")
print(f"\ntrain.py 인자 {len(_known)}개 확인")
if _bad:
    print(f"  ⚠ train.py 에 없는 인자: {_bad}")
    print("  → 해당 변형은 실행 시 죽습니다. VARIANTS 를 고치십시오.")
else:
    print("  모든 변형 인자가 train.py 에 존재")

# ---------------- HF 토큰 ----------------
_tok = None
try:
    from google.colab import userdata
    _tok = userdata.get("HF_TOKEN")
except Exception:
    pass
if not _tok:
    try:
        from huggingface_hub import get_token
        _tok = get_token()
    except Exception:
        pass
print(f"HF 토큰 {'있음' if _tok else '없음 — gemma 를 받지 못합니다'}")
if _tok:
    os.environ["HF_TOKEN"] = _tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok

# ---------------- 데이터 ----------------
def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

PATHS = {k: os.path.join(DATA_DIR, f) for k, f in
         [("train", "train_3000.jsonl"), ("ourtest", "ourtest_500.jsonl"),
          ("spare", "spare_1500.jsonl")]}
for k, p in PATHS.items():
    assert os.path.exists(p), f"없음: {p}"
    print(f"  {k:8s} {len(read_jsonl(p)):5d}개")
THEIR_ROWS = read_jsonl(THEIR_EVAL)
TRAIN_FILE, OUR_EVAL = PATHS["train"], PATHS["ourtest"]

N_STEPS = len(read_jsonl(TRAIN_FILE)) * BASE_ARGS["num_generations"] // (
    BASE_ARGS["per_device_train_batch_size"] * BASE_ARGS["gradient_accumulation_steps"])
print(f"\n계획: {len(RUNS)} run  예상 {len(RUNS)*PS['est_h']:.1f}시간 (예산 {BUDGET_HOURS}시간)")
for v, s in RUNS:
    print(f"  {v}_seed{s:<4d} {VARIANTS.get(v, '없는 변형!')}")
print(f"run 당 {N_STEPS} 스텝")

# ---------------- da_half 용 앵커 패치 ----------------
# 저자 코드는 goal_mu = max(0.5, 그룹평균) 이라 학습 후 δ 가 0 이 된다.
# 논문 본문 Eq.4 는 0.5 고정. DA_HALF=1 일 때만 0.5 를 쓰도록 바꾼다.
import trl, inspect
GT_PATH = os.path.join(os.path.dirname(inspect.getfile(trl)), "trainer", "grpo_trainer.py")
_gt = open(GT_PATH, encoding="utf-8").read()

if "__DA_HALF_PATCH__" in _gt:
    print("da_half 패치 이미 적용됨")
else:
    _cands = [(i + 1, l) for i, l in enumerate(_gt.split("\n")) if "goal_mu" in l]
    print("goal_mu 관련 줄:")
    for i, l in _cands[:12]:
        print(f"  {i:5d} | {l.strip()[:120]}")
    _asn = [(i, l) for i, l in _cands
            if re.search(r"goal_mu\s*=", l) and "def " not in l and "self.goal_mu" not in l]
    if not _asn:
        _asn = [(i, l) for i, l in _cands if re.search(r"goal_mu\s*=", l)]
    assert _asn, ("grpo_trainer.py 에서 goal_mu 대입문을 못 찾았다. "
                  "위 목록을 보고 패치 대상을 지정해야 한다")
    _ln, _src = _asn[0]
    _indent = _src[:len(_src) - len(_src.lstrip())]
    _new = (f'{_indent}# __DA_HALF_PATCH__\n'
            f'{_indent}if __import__("os").environ.get("DA_HALF") == "1":\n'
            f'{_indent}    goal_mu = 0.5\n'
            f'{_indent}else:\n'
            f'{_indent}    ' + _src.strip())
    _lines = _gt.split("\n"); _lines[_ln - 1] = _new
    open(GT_PATH, "w", encoding="utf-8").write("\n".join(_lines))
    print(f"\n패치 적용 (줄 {_ln})")
    print("  전:", _src.strip()[:110])
    print("  후: DA_HALF=1 이면 goal_mu = 0.5, 아니면 원본")

_chk = subprocess.run([sys.executable, "-c",
    "import trl, inspect, os;"
    "p=os.path.join(os.path.dirname(inspect.getfile(trl)),'trainer','grpo_trainer.py');"
    "s=open(p,encoding='utf-8').read();"
    "print('PATCHED' if '__DA_HALF_PATCH__' in s else 'NOPATCH');"
    "import ast; ast.parse(s); print('SYNTAX_OK')"], capture_output=True, text=True)
print(" ", (_chk.stdout or _chk.stderr[-400:]).strip().replace("\n", " / "))
assert "PATCHED" in _chk.stdout and "SYNTAX_OK" in _chk.stdout, \
    "패치 후 grpo_trainer.py 가 깨졌다 — §A 부터 다시"

# ---------------- 모델 호환 사전 검증 ----------------
print("\n" + "=" * 70)
print("  모델 호환 검증")
print("=" * 70)
_probe = f'''
import sys, json
sys.path.insert(0, {LOCAL_TRN!r})
from transformers import AutoTokenizer, AutoConfig
mid = {MODEL_ID!r}
cfg = AutoConfig.from_pretrained(mid)
tok = AutoTokenizer.from_pretrained(mid)
print("ARCH", cfg.architectures[0] if cfg.architectures else "?")
print("LAYERS", getattr(cfg, "num_hidden_layers", "?"))
print("VOCAB", getattr(cfg, "vocab_size", "?"))
print("HAS_TEMPLATE", bool(getattr(tok, "chat_template", None)))
# 시스템 메시지를 받는가
ok_sys = True
try:
    tok.apply_chat_template([{{"role":"system","content":"s"}},
                             {{"role":"user","content":"u"}}],
                            tokenize=False, add_generation_prompt=True)
except Exception:
    ok_sys = False
print("SYSTEM_OK", ok_sys)
t = tok.apply_chat_template([{{"role":"user","content":"HELLO"}}],
                            tokenize=False, add_generation_prompt=True)
print("TEMPLATE_SAMPLE", json.dumps(t[:220]))
'''
_r = subprocess.run([sys.executable, "-c", _probe], capture_output=True, text=True)
print((_r.stdout or "") + ((_r.stderr or "")[-600:] if _r.returncode else ""))
assert _r.returncode == 0, "모델 로드 실패 — HF 토큰·라이선스 승인 확인"
assert "HAS_TEMPLATE True" in _r.stdout, \
    "채팅 템플릿이 없다. do_inference.py 가 템플릿에 의존하므로 이 모델은 쓸 수 없다"

# LoRA target_modules 가 이 모델에 실재하는가
_tm = re.search(r"target_modules\s*=\s*\[([^\]]*)\]",
                open(os.path.join(LOCAL_TRN, "train.py"), encoding="utf-8").read())
assert _tm, "train.py 에서 target_modules 를 못 찾음"
TARGET_MODULES = [x.strip().strip("\"'") for x in _tm.group(1).split(",") if x.strip()]
print(f"\nLoRA target_modules {TARGET_MODULES}")
_probe2 = f'''
from transformers import AutoModelForCausalLM
import torch
m = AutoModelForCausalLM.from_pretrained({MODEL_ID!r}, dtype=torch.bfloat16,
                                         device_map="cpu")
names = {{n.split(".")[-1] for n, _ in m.named_modules()}}
want = {TARGET_MODULES!r}
print("MISSING", [w for w in want if w not in names])
'''
_r2 = subprocess.run([sys.executable, "-c", _probe2], capture_output=True, text=True)
_miss = re.search(r"MISSING (\[.*\])", _r2.stdout or "")
if _miss:
    _ml = eval(_miss.group(1))
    print(f"  이 모델에 없는 모듈: {_ml if _ml else '없음 ✅'}")
    assert not _ml, f"LoRA 대상 모듈 {_ml} 이 이 모델에 없다 — train.py 수정 필요"
else:
    print("  ⚠ 모듈 확인 실패(메모리 부족일 수 있음). 학습 중 오류가 나면 여기를 의심할 것")
    print("   ", (_r2.stderr or "")[-400:])
print("\n✅ 모델 호환 확인 완료")