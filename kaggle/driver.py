# dynamic-couple | Kaggle — couple loop on SWE-bench Lite (in-process, Docker-free)
# Gemma 4 12B = PLANNER, Qwythos-9B = DOER. Roles SWAP on double-fail.
# CHECK-A = run the instance's FAIL_TO_PASS tests in the installed repo (objective gate).
# No API key needed: SWE-bench Lite is a HuggingFace dataset.
# GPU auto-detect: T4x2 -> one model per GPU; single GPU -> both on GPU0.
import os, re, subprocess, sys, json, time, shutil, pathlib

# ---------------- CONFIG (tune here) ----------------
# CONTEXT LENGTH REALITY on 15GB T4: KV cache grows linearly with ctx. 256k is impossible
# (needs ~hundreds of GB). Gemma OOMs above ~4k, Qwythos handles ~8k. Raise only if you
# have a bigger GPU (A100/H100 80GB can do 128k-256k). These are per-model:
QWY_CTX   = int(os.environ.get("QWY_CTX", "8192"))
GEM_CTX   = int(os.environ.get("GEM_CTX", "4096"))
N_INST    = int(os.environ.get("N_INST", "3"))
QWY_REPO  = os.environ.get("QWY_REPO",   "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO= os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")
# ----------------------------------------------------

def sh(c, **k): print("$", c, flush=True); return subprocess.run(c, shell=True, **k)

print("=== GPU ===", flush=True)
gpus = sh("nvidia-smi -L", capture_output=True, text=True).stdout
n_gpu = len([l for l in gpus.splitlines() if l.strip()])
print(gpus, f"-> {n_gpu} GPU(s)", flush=True)
QWY_GPU, GEM_GPU = (0, 1) if n_gpu >= 2 else (0, 0)

print("\n=== install ===", flush=True)
sh("pip -q install huggingface_hub datasets")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")

from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama
from datasets import load_dataset

WORK = pathlib.Path("/kaggle/working/repos"); WORK.mkdir(parents=True, exist_ok=True)

def pick_gguf(repo, prefer="q4_k_m"):
    fs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    pref = [f for f in fs if prefer in f.lower()] or [f for f in fs if "q4" in f.lower()]
    pick = (pref or fs)[0]; print("  ", repo, "->", pick, flush=True)
    return hf_hub_download(repo, pick)

print(f"\n=== load models (Qwythos GPU{QWY_GPU} ctx{QWY_CTX} | Gemma GPU{GEM_GPU} ctx{GEM_CTX}) ===", flush=True)
QWY = Llama(model_path=pick_gguf(QWY_REPO),   n_gpu_layers=-1, n_ctx=QWY_CTX, main_gpu=QWY_GPU, verbose=False)
GEM = Llama(model_path=pick_gguf(GEMMA_REPO), n_gpu_layers=-1, n_ctx=GEM_CTX, main_gpu=GEM_GPU, verbose=False)
MODELS = {"qwythos": QWY, "gemma": GEM}

def chat(key, system, user, max_tokens=1024, temp=0.2):
    o = MODELS[key].create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temp)
    return o["choices"][0]["message"]["content"]

# --- edit format: SEARCH/REPLACE blocks (reliable for small models, unlike raw diffs) ---
PLANNER_SYS = ("You are the PLANNER in a two-model coding couple fixing a real bug. "
               "Given an issue and the relevant file, output a SHORT numbered plan (3-5 steps) "
               "naming the exact function/lines to change. No code.")
DOER_SYS = ("You are the DOER. Produce ONE edit as a SEARCH/REPLACE block, exactly:\n"
            "<<<<<<< SEARCH\n<exact existing code>\n=======\n<new code>\n>>>>>>> REPLACE\n"
            "The SEARCH text must match the file byte-for-byte. Output only the block.")

def apply_search_replace(file_text, block):
    m = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", block, re.DOTALL)
    if not m: return None, "no SEARCH/REPLACE block produced"
    search, replace = m.group(1), m.group(2)
    if search not in file_text: return None, "SEARCH text not found in file (no byte match)"
    return file_text.replace(search, replace, 1), None

def read(p):
    try: return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception: return ""

def target_file_from_patch(gold_patch):
    m = re.search(r"^\+\+\+ b/(.+)$", gold_patch, re.MULTILINE)
    return m.group(1) if m else None

def setup_repo(inst):
    rid = inst["instance_id"].replace("/", "__"); d = WORK / rid
    if d.exists(): shutil.rmtree(d)
    sh(f"git clone -q https://github.com/{inst['repo']}.git {d}", check=True, timeout=300)
    sh(f"git -C {d} checkout -q {inst['base_commit']}", check=True, timeout=120)
    sh(f"pip -q install -e {d} 2>/dev/null || true", timeout=600)
    return d

def run_tests(repo_dir, fail_to_pass):
    tests = json.loads(fail_to_pass) if isinstance(fail_to_pass, str) else fail_to_pass
    if not tests: return False, "no FAIL_TO_PASS tests"
    r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", *tests],
                       cwd=repo_dir, capture_output=True, text=True, timeout=600)
    return (r.returncode == 0), ("" if r.returncode == 0 else (r.stdout + r.stderr)[-600:])

def solve(inst, repo_dir):
    tgt = target_file_from_patch(inst["patch"])
    if not tgt: return False, [{"err": "no target file in gold patch"}]
    fpath = repo_dir / tgt; issue = inst["problem_statement"][:3000]
    planner, doer = "gemma", "qwythos"; fails, hint, trace = 0, None, []
    plan = chat(planner, PLANNER_SYS, f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:4000]}", 400)
    while True:
        u = f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:6000]}\n\nPlan:\n{plan}"
        if hint: u += f"\n\nPrevious edit FAILED:\n{hint}\nFix it."
        block = chat(doer, DOER_SYS, u, 800)
        orig = read(fpath); new, aerr = apply_search_replace(orig, block)
        if new is None:
            ok, err = False, aerr
        else:
            fpath.write_text(new, encoding="utf-8")
            ok, err = run_tests(repo_dir, inst["FAIL_TO_PASS"])
            if not ok: fpath.write_text(orig, encoding="utf-8")
        trace.append({"doer": doer, "planner": planner, "ok": ok, "err": err[:150]})
        if ok: return True, trace
        fails += 1; hint = err
        if fails == 1: continue
        if fails == 2:
            planner, doer = doer, planner
            plan = chat(planner, PLANNER_SYS, f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:4000]}", 400)
            continue
        return False, trace

print(f"\n=== couple on {N_INST} SWE-bench Lite instances ===", flush=True)
ds = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
solved = env_fail = swaps = 0; rows = []; t0 = time.time()
for i in range(min(N_INST, len(ds))):
    inst = ds[i]; iid = inst["instance_id"]
    try:
        rd = setup_repo(inst)
    except Exception as e:
        env_fail += 1; rows.append({"id": iid, "result": "ENV_FAIL", "why": str(e)[:120]})
        print(f"  {iid}: ENV_FAIL ({str(e)[:80]})", flush=True); continue
    try:
        ok, trace = solve(inst, rd)
    except Exception as e:
        ok, trace = False, [{"err": f"solve crash: {e}"}]
    used_swap = any(t.get("planner") == "qwythos" for t in trace)
    swaps += int(used_swap); solved += int(ok)
    rows.append({"id": iid, "result": "PASS" if ok else "FAIL",
                 "attempts": len(trace), "swapped": used_swap})
    print(f"  {iid}: {'PASS' if ok else 'FAIL'} ({len(trace)} attempts"
          f"{', SWAPPED' if used_swap else ''})", flush=True)

dur = time.time() - t0; attempted = N_INST - env_fail
print(f"\n*** SWE-bench Lite: solved {solved}/{attempted} attempted "
      f"({env_fail} env-setup failures) | swaps on {swaps} | {dur:.0f}s ***", flush=True)
with open("/kaggle/working/swe_results.json", "w") as f:
    json.dump({"solved": solved, "attempted": attempted, "env_fail": env_fail,
               "swaps": swaps, "seconds": dur, "rows": rows}, f, indent=2)
print("saved /kaggle/working/swe_results.json", flush=True)
