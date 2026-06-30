# Dynamic Couple Architecture — Qwythos-9B × Gemma 4 12B

> **One-line:** Two peer models share a 3-namespace GitNexus graph **and** a direct
> model-to-model pipeline. Roles (PLANNER / DOER) are dynamic hats that swap on repeated
> failure. Checking is objective (tests + KG). Runs on Kaggle T4×2. Eval = SWE-bench.

---

## Models & runtime

| Role (a hat, not an identity) | Default holder | Why |
|---|---|---|
| **PLANNER** (decompose, check) | Qwythos-9B | stronger reasoner (+34 MMLU, +30 GSM8K), 1M ctx, Python/tool-tuned, uncensored |
| **DOER** (implement edits) | Gemma 4 12B | excellent JSON + multi-step tool calls, multimodal, native thinking |

Roles **swap on double-fail**; incoming model rehydrates from the **shared** graph namespace.

**Runtime = Kaggle T4×2 (GGUF + llama.cpp — proven by qwythos-solo kernel):**
- Models are **GGUF**, downloaded from **HuggingFace at runtime** (`enable_internet`), loaded
  with `llama-cpp-python` on GPU. NOT transformers/bitsandbytes, NOT Kaggle Model mounts.
- **Q4 GGUF** fits one T4 (16GB). **One model per GPU:** Qwythos→GPU0, Gemma→GPU1
  (also enables edit #4 parallelism).
- GGUF/llama.cpp sidesteps T4 Turing (sm_75) limits (no bf16 / flash-attn 2 needed).
- ~12h session cap + weekly quota → **session checkpoint/resume** (edit #2).
- Repos:
  - Qwythos-9B: `empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF` ✅ (proven)
  - Gemma 4 12B: **needs a GGUF HF repo id** ⛔ (same download path — NO Kaggle upload needed)
- Kaggle kernel: `sujitnarrayanm/dynamic-couple-driver` (private, GPU, internet).

---

## Connection — TWO channels at once (the whole idea)

```
        ┌──────────────────────────────────────────────────┐
        │      SHARED GRAPH  (slow path · durable)          │
        │   GitNexus code-KG · agreed plan · commits ·      │
        │   results   +  planner.ns / doer.ns (private)     │
        └──────────────────────────────────────────────────┘
              ▲                                    ▲
              │ promote / read (structured)        │
        ┌───────────┐  ◀═══ DIRECT PIPELINE ═══▶  ┌───────────┐
        │ (1) PLANNER│   fast path · tight couple  │ (2) DOER  │
        │  Qwythos   │   token/state stream         │  Gemma    │
        └───────────┘                              └───────────┘
```

- **Direct pipeline (fast/ephemeral):** proven Phase-D coupling. Live nudges, "stuck"
  signals, fast dialectic on disagreement.
- **Shared graph (durable/structured):** 3-namespace KG = source of truth, clean swaps.
- Rule of thumb: **pipeline carries live thinking; graph carries settled truth.**

### 3 namespaces
1. **planner.ns** (private) — candidate decompositions, reasoning, step hypotheses.
2. **doer.ns** (private) — edit attempts, local notes.
3. **shared.ns** — GitNexus code-KG + agreed plan + committed patches + results.
   Handoff = promote private → shared.

---

## The loop (plan → do → check → swap)

```
plan = PLANNER.make_steps(task, graph.query(task), graph.context(...))  # graph-grounded
for step in shared.ns.steps:
    fails = 0
    while True:
        blast = graph.impact(step.symbol, "upstream")     # DO guardrail
        if blast.risk in (HIGH, CRITICAL): warn / re-scope
        patch = DOER.implement(step, blast, hint=last_reason)
        a = run_tests(patch)                  # CHECK A: compile + tests
        b = graph.detect_changes(base="main") # CHECK B: scope vs planned step
        if a.ok and b.scope_ok: commit -> shared.ns; break
        fails += 1; last_reason = verdict.reason
        if fails == 1: continue               # retry with hint
        if fails == 2: swap(PLANNER, DOER)    # rehydrate from shared.ns
        if fails >= 3: mark_failed(step); break
```

**CHECK = A + B only.** A = tests/compile (SWE-bench FAIL_TO_PASS). B = `detect_changes` +
`impact` (scope/blast). No model-judgment layer C for now.

---

## The 9 architecture edits (all adopted)

**Essential (skeleton):**
1. Pipeline health-check + graph-only fallback (pipeline is flaky → accelerator, not SPOF).
2. Loop budget guard: max steps/swaps/tokens → stop, return best-so-far. + session checkpoint.
3. Planner generates a check/test per step (so CHECK-A always has something objective).

**High-value (speed):**
4. Speculative parallelism on T4×2: DOER starts step i+1 if `impact` says symbols independent.
5. Content-hash verify cache: hash diff → cache verdict; kills redundant re-checks.
6. Incremental replan on swap: patch only the failed sub-plan from shared.ns, not from scratch.

**Learning (the lift):**
7. Write failures+fixes back to graph: "symbol X: approach A failed, B worked." KG = experience.

**Optional (max coupling):**
8. Confidence-gated pipeline: fire expensive pipeline only on low confidence / disagreement.
9. Reconciler for simultaneous shared-graph writes (needed once #4 is on).

Build order: 1,2,3 in skeleton → 4,5 speed → 7 lift → 8,9 last.

---

## Eval — SWE-bench (not HumanEval)
HumanEval is single-function; never exercises the graph. SWE-bench is multi-file repo work —
exactly what shared-graph + impact is for. **Constraint:** standard SWE-bench harness uses
Docker per instance; Kaggle forbids Docker → use a **Docker-free / in-process** variant
(checkout repo at base commit in the notebook, apply patch, run FAIL_TO_PASS in-process).

## Localization (both)
- **Local/Kaggle deploy:** the runtime layer above (4-bit, in-process, T4×2).
- **i18n:** locale detection + language→model routing at the **edges only**; the
  plan→do→check loop stays language-neutral (code is code).

## Open decisions
- [x] Starting roles: **Gemma = PLANNER**, **Qwythos = DOER** (swap on double-fail).
- [x] Localization = both (local-first Kaggle deploy + i18n).
- [x] Eval = SWE-bench (Docker-free variant).
- [ ] Pick a Gemma 4 12B GGUF HuggingFace repo (loads at runtime; no Kaggle upload needed).
