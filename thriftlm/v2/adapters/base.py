"""Abstract interface for V2 plan cache clients."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BasePlanCache(ABC):

    @abstractmethod
    def lookup(self, task: str, context: dict, runtime_caps: dict) -> dict:
        """
        Canonicalize task, fetch candidates, adapt and validate.

        Returns a dict with at minimum a "status" key ("hit" or "miss").
        """

    @abstractmethod
    def store(self, plan: dict) -> dict:
        """
        Persist a plan template.

        Returns a dict with at minimum a "status" key.
        """
