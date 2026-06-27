"""CHECK = A (tests) + B (KG scope/impact). Edit #5: content-hash verdict cache.

No model-judgment layer C for now — both layers are objective.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass


@dataclass
class Verdict:
    ok: bool
    layer_failed: str | None = None   # "tests" | "kg" | None
    reason: str = ""


class Check:
    def __init__(self, graph, cache: dict | None = None):
        self.graph = graph
        self.cache = cache if cache is not None else {}

    @staticmethod
    def _key(patch, step) -> str:
        h = hashlib.sha256()
        h.update(repr(patch).encode())
        h.update(repr(getattr(step, "symbol", step)).encode())
        return h.hexdigest()

    def run(self, patch, step, graph) -> Verdict:
        k = self._key(patch, step)             # edit #5
        if k in self.cache:
            return self.cache[k]

        # A: tests / compile (SWE-bench FAIL_TO_PASS, run in-process — see eval/)
        a = run_tests(patch, step)
        if not a.ok:
            v = Verdict(False, "tests", a.reason)
        else:
            # B: KG scope — did patch touch only the planned symbol? callers ok?
            changed = graph.detect_changes(base="main")
            if not scope_ok(changed, step):
                v = Verdict(False, "kg", "patch touched symbols outside the planned step")
            else:
                v = Verdict(True)
        self.cache[k] = v
        return v


# --- to be implemented against the SWE-bench in-process harness + GitNexus ---
def run_tests(patch, step):  # noqa: D401
    raise NotImplementedError("wire to eval/ in-process SWE-bench runner")


def scope_ok(changed, step) -> bool:
    raise NotImplementedError("compare detect_changes() symbols against step.symbol")
