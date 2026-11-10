"""CacheGen endpoint/edge proxy with codec and network timing separated."""
from __future__ import annotations
import time
from typing import Any, Callable, Mapping
from .common import PhaseTimer, budget_report, normalize_budget_ratio, recursive_nbytes


class CacheGenAdapter:
    def __init__(self, *, prefill: Callable[[str], Any] | None = None,
                 encode: Callable[..., Any] | None = None,
                 decode: Callable[[Any], Any] | None = None,
                 generate: Callable[[str, Any], Any] | None = None,
                 transfer: Callable[[Any], Any] | None = None,
                 bandwidth_mbps: float = 100.0, budget_ratio: float | str = .25,
                 codec_static_state: Any = None, sleep_for_network: bool = False,
                 dry_run: bool = False, strict_budget: bool = False) -> None:
        if bandwidth_mbps <= 0: raise ValueError("bandwidth_mbps must be positive")
        self._prefill, self._encode, self._decode = prefill, encode, decode
        self._generate, self._transfer = generate, transfer
        self.bandwidth_mbps, self.sleep_for_network = float(bandwidth_mbps), sleep_for_network
        self.budget_ratio = normalize_budget_ratio(budget_ratio)
        self._codec_static = codec_static_state
        self.dry_run, self.strict_budget = dry_run, strict_budget
        self._timer = PhaseTimer(); self.reset()

    @property
    def timings_ms(self): return dict(self._timer.ms)

    def reset(self) -> None:
        self._edge_archive = None
        self._client_payload = None
        self._dense_workspace = None
        self.last_budget: dict[str, Any] = {}
        self.peak_temporary_bytes = 0
        self._timer.reset()

    def prefill_archive(self, text: str, segment_id: str | None = None) -> dict[str, Any]:
        if self.dry_run:
            dense = bytearray(max(8, len(text.encode()) * 16))
            self._client_payload = bytearray(int(len(dense) * self.budget_ratio))
        else:
            if self._prefill is None or self._encode is None:
                raise RuntimeError("prefill and encode callbacks are required")
            dense = self._prefill(text)
            target = int(recursive_nbytes(dense) * self.budget_ratio)
            with self._timer.measure("compress"):
                self._client_payload = self._encode(
                    dense, budget_bytes=target, budget_ratio=self.budget_ratio
                )
        self.peak_temporary_bytes = max(self.peak_temporary_bytes,
                                        recursive_nbytes(self._client_payload))
        if self._transfer is not None and not self.dry_run:
            with self._timer.measure("network"):
                self._edge_archive = self._transfer(self._client_payload)
        else:
            network_ms = recursive_nbytes(self._client_payload) * 8 / (self.bandwidth_mbps * 1_000_000) * 1000
            if self.sleep_for_network and not self.dry_run:
                with self._timer.measure("network"): time.sleep(network_ms / 1000)
            else:
                self._timer.ms["network"] += network_ms
            self._edge_archive = self._client_payload
        self._client_payload = None
        self.last_budget = budget_report(self.budget_ratio, dense, self._edge_archive)
        self.last_budget.update({"bandwidth_mbps": self.bandwidth_mbps, "segment_id": segment_id})
        if self.strict_budget and not self.last_budget["within_budget"]:
            raise MemoryError(f"CacheGen stream exceeds byte budget: {self.last_budget}")
        return dict(self.last_budget)

    def query(self, text: str, **kwargs: Any) -> Any:
        if self._edge_archive is None: raise RuntimeError("prefill_archive must be called first")
        if self.dry_run: return "314159"
        if self._decode is None or self._generate is None:
            raise RuntimeError("decode and generate callbacks are required")
        with self._timer.measure("recover"):
            self._dense_workspace = self._decode(self._edge_archive)
        self.peak_temporary_bytes = max(self.peak_temporary_bytes,
                                        recursive_nbytes(self._dense_workspace))
        try:
            with self._timer.measure("generate"):
                return self._generate(text, self._dense_workspace, **kwargs)
        finally:
            self._dense_workspace = None

    def persistent_state(self) -> Mapping[str, Any]:
        return {"edge_encoded_archive": self._edge_archive}
    def static_extra_state(self) -> Mapping[str, Any]:
        return {"codec_tables": self._codec_static}
    def temporary_state(self) -> Mapping[str, Any]:
        return {"client_encoded_payload": self._client_payload,
                "edge_restored_dense_kv": self._dense_workspace}


def make_cachegen_engine(model_id: str) -> Any:
    """Delayed official import so the adapter core works without LMCache/torch."""
    from src.cachegen_interface import CacheGenEngine
    return CacheGenEngine(model_id)
