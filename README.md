# dynamic-couple

**Qwythos-9B × Gemma 4 12B** — a dynamic peer couple for software engineering.

Two models, neither a fixed planner or doer. They share a **3-namespace GitNexus graph**
and a **direct model-to-model pipeline**. Roles are dynamic hats that **swap on repeated
failure**. Checking is objective: tests + KG impact. Target eval: **SWE-bench** (real
multi-file repo tasks — what the graph is actually for).

Runs on **Kaggle T4×2**, 4-bit, in-process (one model per GPU).

See [docs/COUPLE_ARCHITECTURE.md](docs/COUPLE_ARCHITECTURE.md) for the full spec.

## Layout

```
src/couple/
  models/   model backends (Qwythos cuda:0, Gemma cuda:1)
  graph/    3-namespace GitNexus KG (planner.ns / doer.ns / shared.ns) + learning loop
  channel/  two channels: direct pipeline + private→shared promote
  roles/    dynamic PLANNER / DOER hats + swap arbiter
  check/    A: tests   B: KG scope/impact   + content-hash cache
  loop/     plan→do→check→swap controller + budget guard + T4×2 scheduler
  runtime/  Kaggle: 4-bit loader, device map, health, session checkpoint
  i18n/     locale detection + language→model routing (edges only)
eval/       SWE-bench couple runner (Docker-free / in-process variant)
kaggle/     kernel-metadata + driver notebook + push script
```

## Status
Scaffold. Build order: skeleton (edits 1,2,3) → SWE eval → speed (4,5) → lift (7) → max (8,9).
