# dynamic-couple | Kaggle T4x2 — REAL couple loop (plan -> do -> check -> swap)
# Gemma 4 12B = PLANNER (GPU1), Qwythos-9B = DOER (GPU0). Roles are hats that SWAP on
# double-fail. CHECK-A = run the candidate against HumanEval unit tests (objective gate).
# CHECK-B (GitNexus KG) is local/MCP, not available on Kaggle -> tests are the gate here.
# Self-contained (no package import) so it runs as a single Kaggle script kernel.
import os, re, subprocess, sys, tempfile, json, time

def sh(c): print("$", c, flush=True); return subprocess.run(c, shell=True)

print("=== GPU ===", flush=True)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv || true")

print("\n=== install (prebuilt cu124 wheel) ===", flush=True)
sh("pip -q install huggingface_hub datasets")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")

from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama
from datasets import load_dataset

QWY_REPO = os.environ.get("QWY_REPO", "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF")
GEMMA_REPO = os.environ.get("GEMMA_REPO", "ggml-org/gemma-4-12B-it-GGUF")
N_PROBLEMS = int(os.environ.get("N_PROBLEMS", "10"))

def pick_gguf(repo, prefer="q4_k_m"):
    fs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    pref = [f for f in fs if prefer in f.lower()] or [f for f in fs if "q4" in f.lower()]
    pick = (pref or fs)[0]; print("  ", repo, "->", pick, flush=True)
    return hf_hub_download(repo, pick)

print("\n=== load both models ===", flush=True)
# n_ctx=4096: Gemma's SWA/iswa cache pads V cache and OOMs a 15GB T4 at 8192 (proven 4096 OK)
QWY = Llama(model_path=pick_gguf(QWY_REPO), n_gpu_layers=-1, n_ctx=4096, main_gpu=0, verbose=False)
GEM = Llama(model_path=pick_gguf(GEMMA_REPO), n_gpu_layers=-1, n_ctx=4096, main_gpu=1, verbose=False)
MODELS = {"qwythos": QWY, "gemma": GEM}

def chat(model_key, system, user, max_tokens=1024, temp=0.2):
    out = MODELS[model_key].create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temp)
    return out["choices"][0]["message"]["content"]

def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()

# --- roles (hats), default: Gemma plans, Qwythos does ---
PLANNER_SYS = ("You are the PLANNER in a two-model coding couple. Given a function stub, "
               "produce a SHORT numbered implementation plan (3-5 steps). No code, just the plan.")
DOER_SYS = ("You are the DOER in a two-model coding couple. Implement the function. "
            "Return ONLY a Python code block with the complete function. No prose.")

def plan(planner_key, prompt):
    return chat(planner_key, PLANNER_SYS, f"Function to implement:\n{prompt}", max_tokens=400)

def do(doer_key, prompt, plan_text, hint):
    u = f"Function stub:\n{prompt}\n\nPlan:\n{plan_text}"
    if hint: u += f"\n\nThe previous attempt FAILED with:\n{hint}\nFix it."
    return extract_code(chat(doer_key, DOER_SYS, u, max_tokens=1024))

def check_A(candidate, test_code, entry_point):
    """Objective gate: run candidate + HumanEval test in a subprocess."""
    src = candidate + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=20)
        ok = (r.returncode == 0)
        return ok, ("" if ok else (r.stderr.strip()[-400:] or "assertion failed"))
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)

# --- the loop: plan -> do -> check -> swap ---
def solve(problem):
    prompt, test, ep = problem["prompt"], problem["test"], problem["entry_point"]
    planner, doer = "gemma", "qwythos"          # starting hats
    p = plan(planner, prompt)
    fails, hint, trace = 0, None, []
    while True:
        cand = do(doer, prompt, p, hint)
        ok, err = check_A(cand, test, ep)
        trace.append({"doer": doer, "planner": planner, "ok": ok, "err": err[:120]})
        if ok:
            return True, trace
        fails += 1; hint = err
        if fails == 1:
            continue                             # retry, same hats, with hint
        if fails == 2:
            planner, doer = doer, planner        # SWAP hats
            p = plan(planner, prompt)            # incremental re-plan from new planner
            continue
        return False, trace                      # bail after 3

print(f"\n=== couple run on {N_PROBLEMS} HumanEval problems ===", flush=True)
ds = load_dataset("openai_humaneval")["test"]
passed, swaps_used, rows = 0, 0, []
t0 = time.time()
for i in range(min(N_PROBLEMS, len(ds))):
    ok, trace = solve(ds[i])
    used_swap = any(t["planner"] == "qwythos" for t in trace)
    swaps_used += int(used_swap)
    passed += int(ok)
    rows.append({"task": ds[i]["task_id"], "ok": ok, "attempts": len(trace), "swapped": used_swap})
    print(f"  {ds[i]['task_id']}: {'PASS' if ok else 'FAIL'} "
          f"({len(trace)} attempts{', SWAPPED' if used_swap else ''})", flush=True)

dur = time.time() - t0
print(f"\n*** COUPLE pass@1: {passed}/{N_PROBLEMS} = {100*passed/N_PROBLEMS:.1f}%  "
      f"| swaps used on {swaps_used} problems | {dur:.0f}s ***", flush=True)
with open("/kaggle/working/couple_results.json", "w") as f:
    json.dump({"passed": passed, "n": N_PROBLEMS, "swaps_used": swaps_used,
               "seconds": dur, "rows": rows}, f, indent=2)
print("saved /kaggle/working/couple_results.json", flush=True)
