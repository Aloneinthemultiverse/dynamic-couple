"""SWE-bench couple runner — Docker-free / in-process (Kaggle forbids Docker).

Per instance: checkout repo at base_commit, hand task to the couple controller, apply the
produced patch, run FAIL_TO_PASS + PASS_TO_PASS in-process, score. Crash-resumable
(checkpoint per instance) to survive Kaggle's ~12h session cap.

Baselines to report: Qwythos solo | Gemma solo | dynamic couple (the lift).
"""
from __future__ import annotations


def run(dataset="princeton-nlp/SWE-bench_Lite", split="test", checkpoint_dir="checkpoints/"):
    raise NotImplementedError(
        "load instances; for each: checkout base_commit, run_task(...), apply patch, "
        "run FAIL_TO_PASS in-process, record pass/fail + checkpoint."
    )
