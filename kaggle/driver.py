# dynamic-couple | Kaggle T4x2 — couple loop -> SWE-bench Lite CLOUD eval (sb-cli)
# Gemma 4 12B = PLANNER, Qwythos-9B = DOER. Roles SWAP on double-fail.
# GPU produces patches; SWE-bench CLOUD does official scoring (no local Docker/pytest).
#
# SECRET: set SWEBENCH_API_KEY as a Kaggle Secret (Add-ons -> Secrets) or env var.
#   In a Kaggle notebook:
#     from kaggle_secrets import UserSecretsClient
#     os.environ["SWEBENCH_API_KEY"] = UserSecretsClient().get_secret("SWEBENCH_API_KEY")
#   Do NOT hardcode the key here.
import os, re, subprocess, sys, json, time, shutil, pathlib

# ---------------- CONFIG ----------------
# Context = CAP for the auto-max loader (load_max tries this, steps down until it fits).
# KV cache offloads to ~29GB CPU RAM. On T4x2 each model owns a 15GB GPU (more headroom than
# the shared-P100 probe). Measured: Qwythos fits ~128k (fails @200k); Gemma w/ flash_attn higher.
# 256k needs an 80GB A100/H100. Raise caps freely — loader falls back safely if too high.
QWY_CTX   = int(os.environ.get("QWY_CTX", "131072"))   # cap; loader finds max <= this
GEM_CTX   = int(os.environ.get("GEM_CTX", "65536"))    # cap; flash_attn lets Gemma climb on its own GPU
N_INST    = int(os.environ.get("N_INST", "10"))
RUN_ID    = os.environ.get("RUN_ID", "dynamic-couple-1")
SUBSET    = os.environ.get("SWE_SUBSET", "swe-bench_lite")
QWY_REPO  = os.environ.get("QWY_REPO",   "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO= os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")
# ----------------------------------------

HAVE_KEY = bool(os.environ.get("SWEBENCH_API_KEY"))
if not HAVE_KEY:
    print("!! SWEBENCH_API_KEY not set — couple will RUN and generate patches, but cloud "
          "scoring is skipped. Add it as a Kaggle Secret to submit.", flush=True)

def sh(c, **k): print("$", c, flush=True); return subprocess.run(c, shell=True, **k)

print("=== GPU ===", flush=True)
gpus = sh("nvidia-smi -L", capture_output=True, text=True).stdout
n_gpu = len([l for l in gpus.splitlines() if l.strip()]); print(gpus, f"-> {n_gpu} GPU", flush=True)
QWY_GPU, GEM_GPU = (0, 1) if n_gpu >= 2 else (0, 0)

print("\n=== install ===", flush=True)
sh("pip -q install huggingface_hub datasets sb-cli")
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

def load_max(repo, gpu, cap, **kw):
    """Load at the MAX context that fits: try `cap`, step down until it loads. KV offloads to
    CPU RAM so ceiling is RAM-bound; auto-fallback means it never hard-crashes on the accelerator."""
    path = pick_gguf(repo)
    ladder = [c for c in (cap, 131072, 98304, 65536, 49152, 32768, 16384, 8192, 4096) if c <= cap]
    for ctx in dict.fromkeys(ladder):  # dedupe, keep order
        try:
            m = Llama(model_path=path, n_gpu_layers=-1, n_ctx=ctx, main_gpu=gpu, **kw)
            print(f"  {repo.split('/')[-1]} LOADED @ ctx={ctx} on GPU{gpu}", flush=True)
            return m
        except Exception as e:
            print(f"  {repo.split('/')[-1]} ctx={ctx} failed ({str(e)[:60]}) -> stepping down", flush=True)
    raise RuntimeError(f"could not load {repo} at any context")

print(f"\n=== load at MAX possible ctx (Qwythos GPU{QWY_GPU} cap{QWY_CTX} | Gemma GPU{GEM_GPU} cap{GEM_CTX}) ===", flush=True)
QWY = load_max(QWY_REPO, QWY_GPU, QWY_CTX)                      # Qwythos: 1M-ctx GGUF, ~128k fits
GEM = load_max(GEMMA_REPO, GEM_GPU, GEM_CTX, flash_attn=True)   # flash_attn shrinks Gemma iSWA KV cache
MODELS = {"qwythos": QWY, "gemma": GEM}

def chat(key, system, user, max_tokens=1024, temp=0.2):
    o = MODELS[key].create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temp)
    return o["choices"][0]["message"]["content"]

PLANNER_SYS = ("You are the PLANNER in a two-model coding couple fixing a real bug. Given an "
               "issue and the relevant file, output a SHORT numbered plan (3-5 steps) naming the "
               "exact function/lines to change. No code.")
DOER_SYS = ("You are the DOER. Produce ONE edit as a SEARCH/REPLACE block, exactly:\n"
            "<<<<<<< SEARCH\n<exact existing code>\n=======\n<new code>\n>>>>>>> REPLACE\n"
            "The SEARCH text must match the file byte-for-byte. Output only the block.")

def apply_sr(text, block):
    m = re.search(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", block, re.DOTALL)
    if not m: return None, "no SEARCH/REPLACE block"
    s, r = m.group(1), m.group(2)
    if s not in text: return None, "SEARCH not found (no byte match)"
    return text.replace(s, r, 1), None

def read(p):
    try: return pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
    except Exception: return ""

def target_file(gold):
    m = re.search(r"^\+\+\+ b/(.+)$", gold, re.MULTILINE); return m.group(1) if m else None

def setup_repo(inst):
    d = WORK / inst["instance_id"].replace("/", "__")
    if d.exists(): shutil.rmtree(d)
    sh(f"git clone -q https://github.com/{inst['repo']}.git {d}", check=True, timeout=300)
    sh(f"git -C {d} checkout -q {inst['base_commit']}", check=True, timeout=120)
    return d

def compiles(path):
    r = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-200:]

def make_patch(inst, repo_dir):
    """Couple loop. Local gate = edit applies + file compiles (drives swap). Returns git diff."""
    tgt = target_file(inst["patch"])
    if not tgt: return "", [{"err": "no target file"}]
    fpath = repo_dir / tgt; issue = inst["problem_statement"][:3000]
    planner, doer = "gemma", "qwythos"; fails, hint, trace = 0, None, []
    plan = chat(planner, PLANNER_SYS, f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:4000]}", 400)
    while True:
        u = f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:6000]}\n\nPlan:\n{plan}"
        if hint: u += f"\n\nPrevious edit FAILED:\n{hint}\nFix it."
        orig = read(fpath); new, aerr = apply_sr(orig, chat(doer, DOER_SYS, u, 800))
        if new is None:
            ok, err = False, aerr
        else:
            fpath.write_text(new, encoding="utf-8")
            cok, cerr = compiles(fpath); ok, err = cok, ("" if cok else f"syntax: {cerr}")
            if not ok: fpath.write_text(orig, encoding="utf-8")
        trace.append({"doer": doer, "planner": planner, "ok": ok, "err": err[:120]})
        if ok:
            diff = subprocess.run(f"git -C {repo_dir} diff", shell=True, capture_output=True, text=True).stdout
            return diff, trace
        fails += 1; hint = err
        if fails == 1: continue
        if fails == 2:
            planner, doer = doer, planner
            plan = chat(planner, PLANNER_SYS, f"Issue:\n{issue}\n\nFile {tgt}:\n{read(fpath)[:4000]}", 400)
            continue
        return "", trace  # bail -> empty patch (cloud will mark unresolved)

print(f"\n=== generate patches for {N_INST} {SUBSET} instances ===", flush=True)
ds = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
preds_path = "/kaggle/working/preds.jsonl"; swaps = nonempty = 0; t0 = time.time()
with open(preds_path, "w") as pf:
    for i in range(min(N_INST, len(ds))):
        inst = ds[i]; iid = inst["instance_id"]
        try:
            rd = setup_repo(inst); patch, trace = make_patch(inst, rd)
        except Exception as e:
            patch, trace = "", [{"err": f"crash: {e}"}]
        used_swap = any(t.get("planner") == "qwythos" for t in trace)
        swaps += int(used_swap); nonempty += int(bool(patch.strip()))
        pf.write(json.dumps({"instance_id": iid, "model_patch": patch,
                             "model_name_or_path": "dynamic-couple"}) + "\n")
        print(f"  {iid}: {'PATCH' if patch.strip() else 'EMPTY'} "
              f"({len(trace)} attempts{', SWAPPED' if used_swap else ''})", flush=True)

print(f"\n=== {nonempty}/{N_INST} non-empty patches | swaps on {swaps} | {time.time()-t0:.0f}s ===", flush=True)

if HAVE_KEY:
    print("\n=== submit to SWE-bench cloud (sb-cli) ===", flush=True)
    sh(f"sb-cli submit {SUBSET} test --predictions_path {preds_path} --run_id {RUN_ID} "
       f"--wait 2>&1 || sb-cli submit {SUBSET} test --predictions_path {preds_path} --run_id {RUN_ID}")
    print("\n=== report ===", flush=True)
    sh(f"sb-cli get-report {SUBSET} test --run_id {RUN_ID} -o /kaggle/working/report.json 2>&1")
    try:
        print(json.dumps(json.load(open("/kaggle/working/report.json")), indent=2)[:2000], flush=True)
    except Exception as e:
        print("report not ready yet:", e, "-> fetch later with sb-cli get-report", flush=True)
else:
    print("\n=== cloud submit SKIPPED (no key). Patches saved to", preds_path,
          "— add SWEBENCH_API_KEY and re-run to score. ===", flush=True)
