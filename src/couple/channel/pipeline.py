"""Direct model<->model pipeline (fast/ephemeral). Edit #1: health-check + graph-only
fallback. Edit #8 (later): confidence-gated firing.

If the pipeline stalls/desyncs, callers fall back to graph-only handoffs and keep going.
The pipeline is an ACCELERATOR, never a single point of failure.
"""
from __future__ import annotations


class Pipeline:
    def __init__(self, planner_model, doer_model):
        self.planner_model = planner_model
        self.doer_model = doer_model
        self.degraded = False  # True => graph-only fallback

    def available(self) -> bool:
        # edit #1: in-process liveness of both ends
        ok = self.planner_model.healthy() and self.doer_model.healthy()
        self.degraded = not ok
        return ok

    def nudge(self, from_model, to_model, msg: str) -> str | None:
        """Live coupling turn. Returns None when degraded (caller uses graph instead)."""
        if not self.available():
            return None
        # token/state stream exchange (the proven Phase-D coupling) goes here
        raise NotImplementedError("wire live token/state exchange")
