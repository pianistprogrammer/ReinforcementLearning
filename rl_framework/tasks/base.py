# tasks/base.py — task interface
from __future__ import annotations
from typing import Protocol, Tuple, List, Dict, Any


class RLTask(Protocol):
    def make_prompt(self, tokenizer) -> str: ...
    def reward(self, completion: str) -> Tuple[float, Dict[str, Any]]: ...
