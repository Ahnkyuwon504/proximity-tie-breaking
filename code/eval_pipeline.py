# Merge + vLLM inference + strict/loose scoring pipeline (wraps the released MDP-GRPO scripts).
import signal

# set tags kept as used in all artifact filenames: 그들500 = MDP-500, 우리500 = Assembled-500
EVALSETS = [("그들500", THEIR_EVAL), ("우리500", OUR_EVAL)]
# Never set TQDM_DISABLE: vLLM derives throughput from the progress-bar elapsed
# time and divides by zero inside llm.generate without it (observed empirically).
# TB_MODE changes per run, so the environment is read at call time.
def _child_env():
    return dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", PYTHONUNBUFFERED="1")
NOISY = ("Processed prompts", "it/s, est. speed", "Adding requests",
         "Capturing CUDA graph", "torch.compile", "Loading safetensors")
ABORT_ON = ["vLLM init failed", "VLLM not available", "falling back to Transformers",
            "Batch error", "CUDA out of memory"]
PROG = re.compile(r"(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+)")

def run_stream(cmd, cwd=None, abort_on=None, tag="", on_line=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=_child_env())
    for line in p.stdout:
        line = line.rstrip()
        if on_line is not None:
            on_line(line)
        elif line and not any(n in line for n in NOISY):
            print("   |", line[:170])
        if abort_on and any(a.lower() in line.lower() for a in abort_on):
            p.terminate()
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
            raise RuntimeError(f"[{tag}] aborted: {line[:300]}")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"[{tag}] exit code {p.returncode}")

class _TO(Exception):
    pass

def _alarm(sig, frm):
    raise _TO()

def safe_copytree(src, dst, sec=600):
    old = signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(sec))
    try:
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("checkpoint-*", "runs", "*.pt"))
        return True
    except Exception as e:
        print(f"  [copy failed] {type(e).__name__}"); return False
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def collect_mlflow(root):
    out = {}
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) != "metrics":
            continue
        for fn in filenames:
            pts = []
            try:
                for ln in open(os.path.join(dirpath, fn), encoding="utf-8"):
                    q = ln.split()
                    if len(q) >= 3:
                        pts.append((int(q[2]), float(q[1])))
            except Exception:
                continue
            if pts:
                out.setdefault(fn, []).extend(sorted(pts))
    return out

def run_eval(model_path, label):
    """Evaluates both sets with the prior work's do_inference.py + check.py."""
    from transformers import AutoTokenizer
    slug = label.replace("/", "-")
    parts, counts = [], []
    for name, path in EVALSETS:
        rows = read_jsonl(path); parts.extend(rows); counts.append((name, len(rows)))
    cin, cout = f"/content/ev_in_{slug}.jsonl", f"/content/ev_out_{slug}.jsonl"
    write_jsonl(cin, parts)
    t0 = time.time()
    run_stream([sys.executable, os.path.join(LOCAL_SRC, "do_inference.py"),
                f"--data_path={cin}", f"--res_path={cout}", f"--model_path={model_path}",
                "--batch_size", str(GEN_BATCH_SIZE), "--max_model_len", str(MAX_MODEL_LEN),
                "--max_tokens", str(MAX_TOKENS),
                "--gpu_memory_utilization", str(GPU_MEM_UTIL), "--dtype", DTYPE],
               cwd="/content", abort_on=ABORT_ON, tag=label)
    out_rows = read_jsonl(cout)
    assert len(out_rows) == len(parts), f"response count mismatch {len(out_rows)} != {len(parts)}"
    bad = [r for r in out_rows if str(r.get("response", "")).startswith(("Batch error", "Error:"))]
    uniq = len({r.get("response", "") for r in out_rows})
    print(f"  generation {(time.time()-t0)/60:.1f} min  error strings {len(bad)}  distinct {uniq}")
    if bad:
        raise RuntimeError(f"generation failures present: {bad[0]['response'][:120]}")
    if uniq <= 1:
        raise RuntimeError("all responses identical")

    tok = AutoTokenizer.from_pretrained(model_path)
    res, off = {}, 0
    for (name, n), (_, orig) in zip(counts, EVALSETS):
        sub = out_rows[off:off + n]; off += n
        rp = os.path.join(OUT_DIR, f"resp_{slug}_{name}.jsonl"); write_jsonl(rp, sub)
        base = f"check_{slug}_{name}"
        run_stream([sys.executable, os.path.join(LOCAL_SRC, "check.py"),
                    f"--input_data={orig}", f"--input_response_data={rp}",
                    f"--output_dir={OUT_DIR}", f"--output_file_name={base}",
                    f"--evaluation_mode={EVAL_MODE}", "--language=en"],
                   cwd=LOCAL_SRC, tag=base)
        m = json.load(open(os.path.join(OUT_DIR, base + "_metrics.json"), encoding="utf-8"))
        graded = read_jsonl(os.path.join(OUT_DIR, base + ".jsonl"))
        macro = st.mean(st.mean(g["follow_instruction_list"]) for g in graded
                        if g["follow_instruction_list"])
        ntok = [len(tok(r.get("response", ""), add_special_tokens=False)["input_ids"]) for r in sub]
        res[name] = {"ssr_macro": macro, "ssr_micro": m["instruction_accuracy"],
                     "hsr": m["prompt_accuracy"], "ntok_mean": st.mean(ntok),
                     "difficulty": m.get("difficulty", {}), "tier1": m.get("tier1", {})}
        print(f"  [{label}/{name}] SSR macro {macro*100:.2f} micro "
              f"{m['instruction_accuracy']*100:.2f}  HSR {m['prompt_accuracy']*100:.2f}"
              f"  tokens {st.mean(ntok):.0f}")
    del tok
    return res

# ---------------- baseline (once; reused if present) ----------------
BASE_PATH = os.path.join(OUT_DIR, f"result_{PREFIX}baseline.json")
if os.path.exists(BASE_PATH):
    BASELINE = json.load(open(BASE_PATH, encoding="utf-8"))
    print("=" * 70)
    print("  baseline reused (already measured)")
    for k, v in BASELINE.items():
        print(f"    {k}  SSR macro {v['ssr_macro']*100:.2f}  HSR {v['hsr']*100:.2f}")
else:
    print("=" * 70); print(f"  baseline evaluation -- {MODEL_ID}"); print("=" * 70)
    BASELINE = run_eval(MODEL_ID, "baseline")
    with open(BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(BASELINE, f, ensure_ascii=False, indent=2)
print("=" * 70)