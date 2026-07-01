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
# Context (MEASURED on Kaggle P100-16GB; KV cache offloads to ~29GB CPU RAM):
#   Qwythos loads @ 131072 (128k) ✓, fails @ 200000.  Gemma loads @ 16384 (16k) ✓, fails @ 32768.
# These are the proven ceilings. 256k needs an 80GB A100/H100.
QWY_CTX   = int(os.environ.get("QWY_CTX", "131072"))
GEM_CTX   = int(os.environ.get("GEM_CTX", "16384"))
N_INST    = int(os.environ.get("N_INST", "10"))
RUN_ID    = os.environ.get("RUN_ID", "dynamic-couple-1")
SUBSET    = os.environ.get("SWE_SUBSET", "swe-bench_lite")
QWY_REPO  = os.environ.get("QWY_REPO",   "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO= os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")
# ----------------------------------------

assert os.environ.get("SWEBENCH_API_KEY"), "Set SWEBENCH_API_KEY (Kaggle Secret) before running."

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

print(f"\n=== load (Qwythos GPU{QWY_GPU} ctx{QWY_CTX} | Gemma GPU{GEM_GPU} ctx{GEM_CTX}) ===", flush=True)
QWY = Llama(model_path=pick_gguf(QWY_REPO),   n_gpu_layers=-1, n_ctx=QWY_CTX, main_gpu=QWY_GPU, verbose=False)
GEM = Llama(model_path=pick_gguf(GEMMA_REPO), n_gpu_layers=-1, n_ctx=GEM_CTX, main_gpu=GEM_GPU, verbose=False)
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

print("\n=== submit to SWE-bench cloud (sb-cli) ===", flush=True)
sh(f"sb-cli submit {SUBSET} test --predictions_path {preds_path} --run_id {RUN_ID} "
   f"--wait 2>&1 || sb-cli submit {SUBSET} test --predictions_path {preds_path} --run_id {RUN_ID}")
print("\n=== report ===", flush=True)
sh(f"sb-cli get-report {SUBSET} test --run_id {RUN_ID} -o /kaggle/working/report.json 2>&1")
try:
    print(json.dumps(json.load(open("/kaggle/working/report.json")), indent=2)[:2000], flush=True)
except Exception as e:
    print("report not ready yet:", e, "-> fetch later with sb-cli get-report", flush=True)
