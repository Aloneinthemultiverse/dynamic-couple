"""3-namespace KG: planner.ns / doer.ns (private) + shared.ns (consensus).

Wraps GitNexus for the code-KG (impact / detect_changes / query / context). The private
namespaces keep half-baked reasoning from polluting the other model (old dualbrain failure).
Handoff = promote(private -> shared).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Namespace:
    name: str
    nodes: dict[str, Any] = field(default_factory=dict)

    def put(self, key: str, value: Any) -> None: self.nodes[key] = value
    def get(self, key: str, default=None): return self.nodes.get(key, default)


@dataclass
class SharedNamespace(Namespace):
    def put_plan(self, plan) -> None: self.put("plan", plan)
    def commit(self, patch) -> None: self.nodes.setdefault("commits", []).append(patch)


class GraphCouple:
    """Facade over the three namespaces + GitNexus code-KG calls."""

    def __init__(self, gitnexus):
        self.planner = Namespace("planner.ns")
        self.doer = Namespace("doer.ns")
        self.shared = SharedNamespace("shared.ns")
        self._gn = gitnexus  # GitNexus MCP / CLI client

    # --- GitNexus code-KG (shared.ns is grounded in real code) ---
    def query(self, q: str): return self._gn.query(query=q)
    def context(self, name: str): return self._gn.context(name=name)
    def impact(self, target: str, direction: str = "upstream"):
        return self._gn.impact(target=target, direction=direction)
    def detect_changes(self, base: str = "main"):
        return self._gn.detect_changes(scope="compare", base_ref=base)

    def promote(self, src: Namespace, key: str) -> None:
        """Handoff: copy a settled artifact from a private ns into shared.ns."""
        self.shared.put(key, src.get(key))
