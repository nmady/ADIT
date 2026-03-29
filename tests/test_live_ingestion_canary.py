"""Optional live internet-ingestion canary tests.

These tests are skipped by default and only run when explicitly enabled via
environment variables. They are intended for manual/nightly checks against live
provider APIs, not for regular PR gating.
"""

import os
from pathlib import Path

import pytest

import citation_ingestion as ci


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


RUN_LIVE = _env_flag("ADIT_RUN_LIVE_CANARY", default=False)


pytestmark = pytest.mark.skipif(
    not RUN_LIVE,
    reason="Set ADIT_RUN_LIVE_CANARY=1 to run live ingestion canary tests.",
)


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def test_live_ingestion_completeness_invariants(tmp_path):
    """Live canary: validate completeness metadata invariants in exhaustive mode.

    Environment knobs:
    - ADIT_CANARY_THEORY_NAME: optional theory label used for query context
    - ADIT_CANARY_L1_PAPERS: comma-separated seed IDs/DOIs
    - ADIT_CANARY_SOURCES: comma-separated providers (default: openalex,semantic_scholar)
    - ADIT_CANARY_DEPTH: l2 or l2l3 (default: l2)
    - ADIT_CANARY_CACHE_DIR: optional explicit cache path
    """

    theory_name = os.getenv("ADIT_CANARY_THEORY_NAME", "Technology Acceptance Model")
    l1_papers = _csv_env("ADIT_CANARY_L1_PAPERS", "10.2307/249008")
    sources = _csv_env("ADIT_CANARY_SOURCES", "openalex,semantic_scholar")
    depth = os.getenv("ADIT_CANARY_DEPTH", "l2").strip().lower()

    if depth not in {"l2", "l2l3"}:
        raise AssertionError("ADIT_CANARY_DEPTH must be 'l2' or 'l2l3'.")

    cache_dir = Path(os.getenv("ADIT_CANARY_CACHE_DIR", str(tmp_path / "live_canary_cache")))

    result = ci.ingest_from_internet(
        theory_name=theory_name,
        l1_papers=l1_papers,
        sources=sources,
        depth=depth,
        cache_dir=cache_dir,
        refresh=True,
        max_l2=200,
        max_l3=None,
        exhaustive=True,
        verbose=False,
    )

    assert "completeness" in result.metadata
    assert isinstance(result.metadata.get("completeness"), dict)
    assert isinstance(result.metadata.get("provider_stats"), dict)
    assert result.metadata.get("paper_count", 0) >= len(l1_papers)

    completeness = result.metadata["completeness"]

    # Validate core invariants across all reported L1/provider entries.
    for _l1_id, provider_entries in completeness.items():
        for provider_name, entry in provider_entries.items():
            status = entry.get("status")
            fetched = int(entry.get("fetched") or 0)
            expected = entry.get("expected")

            assert status in {"complete", "partial", "failed", "skipped"}
            assert fetched >= 0

            if expected is not None:
                expected_int = int(expected)
                assert expected_int >= 0

                # Impossible state: reporting complete with fewer items than expected.
                assert not (status == "complete" and fetched < expected_int)

                # For partial responses with known totals, fetched should be lower.
                if status == "partial":
                    assert fetched < expected_int

            # When complete and expected is known, fetched should match expected.
            if status == "complete" and expected is not None:
                assert fetched == int(expected)

            # Provider stats should include every provider observed in completeness.
            assert provider_name in result.metadata["provider_stats"]
