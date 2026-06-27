"""The plan -> do -> check -> swap loop. Skeleton with edits 1,2,3 wired.

This is the orchestration core. graph / channel / roles / check are injected so each
layer stays swappable (and testable without GPUs).
"""
from __future__ import annotations
from .budget import Budget
from ..roles.arbiter import Roles, decide


def run_task(task, *, graph, planner, doer, check, pipeline, budget: Budget | None = None):
    budget = budget or Budget()
    roles = Roles(planner=planner.name, doer=doer.name)

    # PLAN — graph-grounded; edit #3: planner also emits a per-step test
    plan = planner.make_steps(task, ctx=graph.query(task))
    graph.shared.put_plan(plan)

    results = []
    for step in plan.steps:
        fails, last_reason = 0, None
        while True:
            if budget.exhausted():
                return _finish(results, reason="budget")
            budget.tick_step()

            # DO — impact guardrail before editing (CLAUDE.md rule)
            blast = graph.impact(step.symbol, "upstream")
            patch = doer.implement(step, blast=blast, hint=last_reason,
                                   pipeline=pipeline)  # edit #1: pipeline may be in fallback

            # CHECK A+B
            verdict = check.run(patch, step, graph)
            if verdict.ok:
                graph.shared.commit(patch)
                results.append((step, "ok"))
                break

            fails += 1
            last_reason = verdict.reason
            roles, action = decide(roles, fails)
            if action == "retry":
                continue
            if action == "swap":
                budget.tick_swap()
                planner, doer = doer, planner          # hats flip
                # incoming planner rehydrates from shared.ns (clean, not chat history)
                continue
            results.append((step, f"bail:{verdict.reason}"))  # edit #7 hook: record fail->fix
            break
    return _finish(results, reason="done")


def _finish(results, reason):
    return {"reason": reason, "results": results}
