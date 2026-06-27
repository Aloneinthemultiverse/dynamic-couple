"""Edit #2 — loop budget guard. Prevents runaway/thrash on the T4×2 weekly quota."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Budget:
    max_steps: int = 40
    max_swaps: int = 12
    max_tokens: int = 400_000
    _steps: int = field(default=0, init=False)
    _swaps: int = field(default=0, init=False)
    _tokens: int = field(default=0, init=False)

    def tick_step(self) -> None: self._steps += 1
    def tick_swap(self) -> None: self._swaps += 1
    def add_tokens(self, n: int) -> None: self._tokens += n

    def exhausted(self) -> bool:
        return (self._steps >= self.max_steps
                or self._swaps >= self.max_swaps
                or self._tokens >= self.max_tokens)
