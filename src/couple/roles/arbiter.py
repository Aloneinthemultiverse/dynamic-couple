"""The swap rule — NOT a separate model. Just the referee that decides role hats.

retry-once → swap → bail. This is the whole "dynamic" mechanism (edit: roles are hats).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Roles:
    planner: str  # model name currently wearing PLANNER hat
    doer: str     # model name currently wearing DOER hat

    def swapped(self) -> "Roles":
        return Roles(planner=self.doer, doer=self.planner)


def decide(roles: Roles, fails: int) -> tuple[Roles, str]:
    """Return (roles_for_next_attempt, action).

    fails == 1 -> same doer retries with hint
    fails == 2 -> SWAP hats; incoming model rehydrates from shared.ns
    fails >= 3 -> bail this step
    """
    if fails <= 1:
        return roles, "retry"
    if fails == 2:
        return roles.swapped(), "swap"
    return roles, "bail"
