"""Tests for parallel provider ingestion (Phase 6 – Step 17).

Six tests covering:
  1. Wave-1 providers execute concurrently when max_workers > 1.
  2. A provider failure does not cancel surviving providers.
  3. Parallel and sequential runs produce equivalent output.
  4. Per-provider checkpoint files are written during wave-1.
  5. Coordinator checkpoint is written after each completed provider.
  6. ThreadPoolExecutor is never constructed when max_workers is None.
"""

import threading
import time
from pathlib import Path

import pytest

import citation_ingestion as ci


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper provider
# ─────────────────────────────────────────────────────────────────────────────


class _SimpleL2Provider(ci.CitationProvider):
    """Minimal provider that registers one L2 paper and optionally raises."""

    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

    def __init__(self, name: str, *, raise_on_l2: bool = False, delay: float = 0.0):
        self.name = name
        self.raise_on_l2 = raise_on_l2
        self.delay = delay

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1_papers[0]: ci.IngestionPaper(
                paper_id=l1_papers[0],
                source_ids={self.name: f"seed-{self.name}"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        if self.delay:
            time.sleep(self.delay)
        if self.raise_on_l2:
            raise RuntimeError(f"{self.name} simulated L2 failure")
        seed = l1_papers[0]
        l2_id = f"{self.name}:L2A"
        return (
            {l2_id: {seed}},
            {l2_id: ci.IngestionPaper(paper_id=l2_id, title=f"{self.name} L2A", year=2022, citations=1)},
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=None):
        return {}, {}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – concurrent wave-1 execution
# ─────────────────────────────────────────────────────────────────────────────


def test_wave1_providers_execute_concurrently(monkeypatch, tmp_path):
    """When max_workers >= 2, two providers' fetch_l2_and_metadata calls must overlap."""
    tracker = {"active": 0, "overlap": False, "lock": threading.Lock()}

    class _BarrierProvider(ci.CitationProvider):
        capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=False)

        def __init__(self, name: str):
            self.name = name

        def fetch_seed_metadata(self, l1_papers):
            return {
                l1_papers[0]: ci.IngestionPaper(
                    paper_id=l1_papers[0],
                    source_ids={self.name: f"seed-{self.name}"},
                )
            }

        def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
            with tracker["lock"]:
                tracker["active"] += 1
                if tracker["active"] >= 2:
                    tracker["overlap"] = True
            time.sleep(0.06)
            with tracker["lock"]:
                tracker["active"] -= 1

            seed = l1_papers[0]
            l2_id = f"{self.name}:L2A"
            return (
                {l2_id: {seed}},
                {l2_id: ci.IngestionPaper(paper_id=l2_id, year=2022, citations=1)},
            )

        def fetch_l3_references(self, l2_paper_ids, max_l3=None):
            return {}, {}

    p1 = _BarrierProvider("wave1_a")
    p2 = _BarrierProvider("wave1_b")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p1, p2])

    ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["wave1_a", "wave1_b"],
        depth="l2",
        cache_dir=tmp_path,
        refresh=True,
        max_workers=2,
    )

    assert tracker["overlap"] is True, "Wave-1 providers did not execute concurrently"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – one provider failure does not abort others
# ─────────────────────────────────────────────────────────────────────────────


def test_provider_failure_does_not_cancel_others(monkeypatch, tmp_path):
    """A failing wave-1 provider must not prevent surviving providers from completing."""
    failing = _SimpleL2Provider("fail_a", raise_on_l2=True)
    ok_b = _SimpleL2Provider("ok_b")
    ok_c = _SimpleL2Provider("ok_c")
    monkeypatch.setattr(ci, "build_providers", lambda _: [failing, ok_b, ok_c])

    result = ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["fail_a", "ok_b", "ok_c"],
        depth="l2",
        cache_dir=tmp_path,
        refresh=True,
        max_workers=3,
    )

    assert "ok_b:L2A" in result.citation_data
    assert "ok_c:L2A" in result.citation_data
    assert "fail_a:L2A" not in result.citation_data
    assert result.metadata["provider_stats"]["fail_a"]["status"] == "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – parallel and sequential produce identical output
# ─────────────────────────────────────────────────────────────────────────────


def test_parallel_vs_sequential_equivalence(monkeypatch, tmp_path):
    """max_workers=1 (sequential) and max_workers=3 (parallel) must yield identical results."""

    def _providers(_):
        return [_SimpleL2Provider("equiv_a"), _SimpleL2Provider("equiv_b")]

    monkeypatch.setattr(ci, "build_providers", _providers)
    sequential = ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["equiv_a", "equiv_b"],
        depth="l2",
        cache_dir=tmp_path / "seq-cache",
        refresh=True,
        max_workers=1,
    )

    monkeypatch.setattr(ci, "build_providers", _providers)
    parallel = ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["equiv_a", "equiv_b"],
        depth="l2",
        cache_dir=tmp_path / "par-cache",
        refresh=True,
        max_workers=3,
    )

    assert sequential.citation_data == parallel.citation_data
    assert sequential.papers_data == parallel.papers_data


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – per-provider checkpoint files are written
# ─────────────────────────────────────────────────────────────────────────────


def test_per_provider_checkpoint_files_written(monkeypatch, tmp_path):
    """_write_provider_checkpoint_state must be called for each wave-1 provider."""
    calls: list[str] = []
    call_lock = threading.Lock()
    original_write = ci._write_provider_checkpoint_state

    def _tracking_write(checkpoint_root, key, provider_name, **kwargs):
        with call_lock:
            calls.append(provider_name)
        original_write(checkpoint_root, key, provider_name, **kwargs)

    monkeypatch.setattr(ci, "_write_provider_checkpoint_state", _tracking_write)

    p1 = _SimpleL2Provider("ckpt_a")
    p2 = _SimpleL2Provider("ckpt_b")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p1, p2])

    ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["ckpt_a", "ckpt_b"],
        depth="l2",
        cache_dir=tmp_path,
        refresh=True,
        max_workers=2,
    )

    assert "ckpt_a" in calls
    assert "ckpt_b" in calls


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – coordinator checkpoint is written after each provider completes
# ─────────────────────────────────────────────────────────────────────────────


def test_coordinator_checkpoint_written_after_each_provider(monkeypatch, tmp_path):
    """_write_coordinator_checkpoint_state must be called at least once per completed provider."""
    call_count = {"n": 0}
    count_lock = threading.Lock()
    original_write = ci._write_coordinator_checkpoint_state

    def _tracking_write(*args, **kwargs):
        with count_lock:
            call_count["n"] += 1
        original_write(*args, **kwargs)

    monkeypatch.setattr(ci, "_write_coordinator_checkpoint_state", _tracking_write)

    p1 = _SimpleL2Provider("coord_a")
    p2 = _SimpleL2Provider("coord_b")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p1, p2])

    ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["coord_a", "coord_b"],
        depth="l2",
        cache_dir=tmp_path,
        refresh=True,
        max_workers=2,
    )

    # One write per successfully completed wave-1 provider.
    assert call_count["n"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – max_workers=None never constructs a ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────


def test_max_workers_none_is_sequential(monkeypatch, tmp_path):
    """ThreadPoolExecutor must never be constructed when max_workers is None."""

    def _assert_not_called(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor should not be constructed when max_workers is None")

    monkeypatch.setattr(ci, "ThreadPoolExecutor", _assert_not_called)

    p = _SimpleL2Provider("seq_only")
    monkeypatch.setattr(ci, "build_providers", lambda _: [p])

    result = ci.ingest_from_internet(
        theory_name="TAM",
        l1_papers=["10.1000/xyz"],
        sources=["seq_only"],
        depth="l2",
        cache_dir=tmp_path,
        refresh=True,
        max_workers=None,
    )

    assert "seq_only:L2A" in result.citation_data
