# ctx probe: does Qwythos load at 200k context on Kaggle T4x2? Measure, don't guess.
import os, subprocess, time
def sh(c, **k): print("$", c, flush=True); return subprocess.run(c, shell=True, **k)
sh("nvidia-smi -L")
sh("pip -q install huggingface_hub")
sh("pip -q install 'llama-cpp-python[server]' --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama

def pick(repo, prefer="q4_k_m"):
    fs=[f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    p=[f for f in fs if prefer in f.lower()] or [f for f in fs if "q4" in f.lower()]
    x=(p or fs)[0]; print("  ",repo,"->",x,flush=True); return hf_hub_download(repo,x)

QWY="empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF"
GEM="ggml-org/gemma-4-12B-it-GGUF"
qpath=pick(QWY); gpath=pick(GEM)

for ctx in [200000, 131072, 65536, 32768]:
    print(f"\n=== try Qwythos GPU0 @ n_ctx={ctx} ===", flush=True)
    try:
        t=time.time()
        m=Llama(model_path=qpath, n_gpu_layers=-1, n_ctx=ctx, main_gpu=0, verbose=False)
        print(f"  QWYTHOS LOADED @ {ctx} in {time.time()-t:.0f}s", flush=True)
        sh("nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv")
        sh("free -g | head -2")
        out=m.create_chat_completion(messages=[{"role":"user","content":"say OK"}], max_tokens=8)
        print("  gen:", out["choices"][0]["message"]["content"], flush=True)
        del m; QWY_OK=ctx; break
    except Exception as e:
        print(f"  FAILED @ {ctx}: {str(e)[:200]}", flush=True)

print("\n=== try Gemma GPU1 @ escalating ctx ===", flush=True)
for ctx in [32768, 16384, 8192, 4096]:
    try:
        m=Llama(model_path=gpath, n_gpu_layers=-1, n_ctx=ctx, main_gpu=1, verbose=False)
        print(f"  GEMMA LOADED @ {ctx}", flush=True); del m; break
    except Exception as e:
        print(f"  gemma FAILED @ {ctx}: {str(e)[:150]}", flush=True)
print("\n=== PROBE DONE ===", flush=True)
