import io
import logging
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock

import citation_ingestion as ci


class _FakeProvider(ci.CitationProvider):
    name = "fake"
    capabilities = ci.ProviderCapabilities(True, True, True)

    def __init__(self):
        self.calls = 0
        self.seed_calls = 0

    def fetch_seed_metadata(self, l1_papers):
        self.seed_calls += 1
        return {
            "doi:10.1000/xyz1": ci.IngestionPaper(
                paper_id="doi:10.1000/xyz1",
                title="Foundational Paper",
                abstract="Canonical theory text.",
                citations=100,
                year=2000,
                doi="10.1000/xyz1",
                source_ids={"fake": "seed:doi:10.1000/xyz1"},
            )
        }

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        self.calls += 1
        return (
            {
                "openalex:W100": {"doi:10.1000/xyz1"},
                "semantic_scholar:abc123": {"doi:10.1000/xyz1"},
            },
            {
                "openalex:W100": ci.IngestionPaper(
                    paper_id="openalex:W100",
                    title="Theory Application Study",
                    year=2022,
                    citations=8,
                    doi="10.1000/abc",
                    source_ids={"openalex": "https://openalex.org/W100"},
                ),
                "semantic_scholar:abc123": ci.IngestionPaper(
                    paper_id="semantic_scholar:abc123",
                    title="Theory Application Study",
                    year=2022,
                    citations=12,
                    doi="10.1000/abc",
                    source_ids={"semantic_scholar": "abc123"},
                ),
            },
        )

    def fetch_l3_references(self, l2_paper_ids, max_l3=500):
        return (
            {"openalex:W100": {"doi:10.1000/l3a"}},
            {
                "doi:10.1000/l3a": ci.IngestionPaper(
                    paper_id="doi:10.1000/l3a",
                    title="L3 supporting work",
                    year=2018,
                    citations=5,
                    doi="10.1000/l3a",
                )
            },
        )


def test_normalize_identifier_handles_doi_and_openalex():
    assert ci.normalize_identifier("10.1234/ABC") == "doi:10.1234/abc"
    assert ci.normalize_identifier("https://doi.org/10.5555/xyz") == "doi:10.5555/xyz"
    assert ci.normalize_identifier("W12345") == "openalex:W12345"
    assert ci.normalize_identifier("https://openalex.org/W999") == "openalex:W999"


def test_build_providers_ignores_unknown_sources():
    providers = ci.build_providers(["openalex", "not-real", "crossref"])
    names = [p.name for p in providers]
    assert names == ["openalex", "crossref"]


def test_ingest_from_internet_dedupes_and_uses_cache(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result1 = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert provider.seed_calls == 1
    assert result1.metadata["paper_count"] >= 3
    assert result1.metadata["edge_count"] >= 2

    # Duplicate L2 candidates should collapse into a single citing node.
    assert len(result1.citation_data) == 1
    citing_id = next(iter(result1.citation_data.keys()))
    assert len(result1.citation_data[citing_id]) == 2
    merged_entry = result1.papers_data[citing_id]
    assert merged_entry["source_ids"]["openalex"] == "https://openalex.org/W100"
    assert merged_entry["source_ids"]["semantic_scholar"] == "abc123"

    # Second run should come from cache and avoid calling provider again.
    result2 = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2l3",
        cache_dir=Path(tmp_path),
        refresh=False,
        max_l2=10,
        max_l3=10,
    )

    assert provider.calls == 1
    assert provider.seed_calls == 1
    assert result2.metadata["cache_key"] == result1.metadata["cache_key"]


def test_ingest_from_internet_hydrates_l1_metadata_from_seed_lookup(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=0,
    )

    l1_entry = result.papers_data["doi:10.1000/xyz1"]
    assert l1_entry["title"] == "Foundational Paper"
    assert l1_entry["abstract"] == "Canonical theory text."
    assert l1_entry["citations"] == 100
    assert l1_entry["year"] == 2000
    assert l1_entry["source_ids"] == {"fake": "seed:doi:10.1000/xyz1"}


def test_ingest_from_internet_metadata_includes_fetch_stats(monkeypatch, tmp_path):
    provider = _FakeProvider()

    def fake_build_providers(_sources):
        return [provider]

    monkeypatch.setattr(ci, "build_providers", fake_build_providers)

    result = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["fake"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        max_l2=10,
        max_l3=0,
    )

    assert "fetch_stats" in result.metadata
    stats = result.metadata["fetch_stats"]
    assert isinstance(stats["total_requests"], int)
    assert isinstance(stats["total_failures"], int)
    assert isinstance(stats["per_provider_failures"], dict)
    assert stats["per_provider_failures"]["fake"] == 0


# ---------------------------------------------------------------------------
# Cited-by traversal: exhaustive pagination
# ---------------------------------------------------------------------------


class _PagedCitedByProvider(ci.CitationProvider):
    """Fake provider that supports cited-by traversal and returns 2 pages of citers."""

    name = "paged"
    capabilities = ci.ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def __init__(self):
        self.citers_calls: list = []

    def fetch_seed_metadata(self, l1_papers):
        return {
            l1: ci.IngestionPaper(
                paper_id=l1,
                title="Foundational",
                citations=50,
                year=2000,
                doi="10.1000/xyz1",
                source_ids={"paged": "PAGED001"},
            )
            for l1 in l1_papers
        }

    def fetch_citers_for_l1(self, l1_provider_id, max_results=None):
        self.citers_calls.append(l1_provider_id)
        page1 = {
            "paged:P001": ci.IngestionPaper(
                paper_id="paged:P001", title="Citer One", year=2021, citations=5
            ),
            "paged:P002": ci.IngestionPaper(
                paper_id="paged:P002", title="Citer Two", year=2022, citations=3
            ),
        }
        page2 = {
            "paged:P003": ci.IngestionPaper(
                paper_id="paged:P003", title="Citer Three", year=2023, citations=1
            ),
        }
        all_papers = {**page1, **page2}
        if max_results is not None:
            truncated = dict(list(all_papers.items())[:max_results])
            status = "partial" if len(truncated) < len(all_papers) else "complete"
            return truncated, len(all_papers), status
        return all_papers, len(all_papers), "complete"

    def fetch_l2_and_metadata(self, l1_papers, theory_name, key_constructs=None, max_l2=200):
        return {}, {}

    def fetch_l3_references(self, l2_paper_ids, max_l3=500):
        return {}, {}


def test_ingest_exhaustive_fetches_all_pages(monkeypatch, tmp_path):
    """Exhaustive mode should collect all citers across provider pages."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    # All 3 citers must appear in citation_data.
    all_citing = set(result.citation_data.keys())
    assert len(all_citing) == 3

    # fetch_citers_for_l1 was called once, for the resolved L1 ID.
    assert len(provider.citers_calls) == 1


def test_ingest_sample_mode_caps_results(monkeypatch, tmp_path):
    """Non-exhaustive mode should cap citers at max_l2."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=False,
        max_l2=2,
    )

    assert len(result.citation_data) <= 2


def test_ingest_completeness_in_metadata(monkeypatch, tmp_path):
    """metadata['completeness'] should report status per L1 per provider."""
    provider = _PagedCitedByProvider()

    monkeypatch.setattr(ci, "build_providers", lambda _: [provider])

    result = ci.ingest_from_internet(
        theory_name="Technology Acceptance Model",
        l1_papers=["10.1000/xyz1"],
        sources=["paged"],
        depth="l2",
        cache_dir=Path(tmp_path),
        refresh=True,
        exhaustive=True,
    )

    assert "completeness" in result.metadata
    completeness = result.metadata["completeness"]
    # Should have an entry for the L1 paper.
    assert len(completeness) == 1
    l1_entry = next(iter(completeness.values()))
    assert "paged" in l1_entry
    assert l1_entry["paged"]["status"] == "complete"
    assert l1_entry["paged"]["fetched"] == 3
    assert l1_entry["paged"]["expected"] == 3


# ---------------------------------------------------------------------------
# _safe_get: retry on transient errors, stop on permanent errors
# ---------------------------------------------------------------------------


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.com", code=code, msg="err", hdrs=None, fp=None
    )


def test_safe_get_retries_on_429_then_succeeds(monkeypatch):
    """_safe_get should retry on HTTP 429 and return the response when the retry succeeds."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            raise _make_http_error(429)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")
    assert result == {"ok": True}
    assert call_count[0] == 3


def test_safe_get_stops_immediately_on_permanent_failure(monkeypatch):
    """_safe_get should not retry on HTTP 403 and should return None immediately."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        raise _make_http_error(403)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")
    assert result is None
    assert call_count[0] == 1  # No retries for permanent failures.
    assert ci._INGEST_STATS["total_failures"] == 1


def test_safe_get_exhausts_retries_and_returns_none(monkeypatch):
    """_safe_get should return None after exhausting all retries on transient errors."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        raise _make_http_error(500)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test", max_retries=3)
    assert result is None
    assert call_count[0] == 3
    assert ci._INGEST_STATS["total_failures"] == 1


def test_safe_get_logs_http_error_body_on_permanent_failure(monkeypatch, caplog):
    """_safe_get should include the server response body in permanent failure logs."""

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            url="https://example.com/test",
            code=400,
            msg="bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"offset + limit must be < 10000"}'),
        )

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    ci._reset_ingest_stats(["test"])

    with caplog.at_level(logging.WARNING):
        result = ci._safe_get("https://example.com/test", provider="test")

    assert result is None
    assert "offset + limit must be < 10000" in caplog.text


def test_semantic_citers_caps_limit_before_api_boundary(monkeypatch):
    """Semantic Scholar pagination should cap page size so offset + limit stays below 10000."""
    provider = ci.SemanticScholarProvider()
    request_pairs = []

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        offset = int(params["offset"][0])
        limit = int(params["limit"][0])
        request_pairs.append((offset, limit))
        assert offset + limit < 10000

        data = []
        for index in range(limit):
            paper_index = offset + index
            data.append(
                {
                    "citingPaper": {
                        "paperId": f"paper-{paper_index}",
                        "title": f"Paper {paper_index}",
                        "year": 2020,
                        "citationCount": 1,
                        "externalIds": {},
                        "abstract": "",
                    }
                }
            )

        return {
            "total": 12000,
            "data": data,
            "next": "has-more",
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers, expected_count, status = provider.fetch_citers_for_l1("semantic_scholar:seed")

    assert request_pairs[-1] == (9000, 999)
    assert len(request_pairs) == 10
    assert len(papers) == 9999
    assert expected_count == 12000
    assert status == "partial"


def test_semantic_citers_stop_without_requesting_at_offset_9999(monkeypatch):
    """Semantic Scholar pagination should stop cleanly once the next request would cross the API ceiling."""
    provider = ci.SemanticScholarProvider()
    request_count = [0]
    request_offsets = []

    def fake_safe_get(url, timeout=20, provider=None, max_retries=ci._SAFE_GET_MAX_RETRIES):
        request_count[0] += 1
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        offset = int(params["offset"][0])
        limit = int(params["limit"][0])
        request_offsets.append(offset)
        assert request_count[0] <= 10
        assert offset + limit < 10000
        return {
            "total": 10050,
            "data": [
                {
                    "citingPaper": {
                        "paperId": f"paper-{offset + index}",
                        "title": f"Paper {offset + index}",
                        "year": 2020,
                        "citationCount": 1,
                        "externalIds": {},
                        "abstract": "",
                    }
                }
                for index in range(limit)
            ],
            "next": "has-more",
        }

    monkeypatch.setattr(ci, "_safe_get", fake_safe_get)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)

    papers, expected_count, status = provider.fetch_citers_for_l1("semantic_scholar:seed")

    assert request_count[0] == 10
    assert request_offsets[-1] == 9000
    assert len(papers) == 9999
    assert expected_count == 10050
    assert status == "partial"


# ---------------------------------------------------------------------------
# Verbose progress output
# ---------------------------------------------------------------------------


def test_verbose_off_produces_no_output(monkeypatch, capsys):
    """_vprint and _countdown_sleep should produce no output when verbose is off."""
    ci.set_verbose(False)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._vprint("should not appear")
    ci._countdown_sleep(2.0, "label")
    captured = capsys.readouterr()
    assert captured.err == ""


def test_verbose_countdown_writes_ticks_to_stderr(monkeypatch, capsys):
    """_countdown_sleep should write countdown ticks to stderr and clear the line."""
    ci.set_verbose(True)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._countdown_sleep(2.0, "test-provider attempt 1/3")
    captured = capsys.readouterr()
    ci.set_verbose(False)
    assert "retrying in" in captured.err
    # Final clear sequence leaves line ending with \r
    assert captured.err.endswith("\r")


def test_verbose_safe_get_prints_retry_message(monkeypatch, capsys):
    """When verbose, _safe_get should print retry messages to stderr on transient failures."""
    call_count = [0]

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 2:
            raise _make_http_error(429)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci._reset_ingest_stats(["test"])
    ci.set_verbose(True)

    result = ci._safe_get("https://example.com/test", provider="test", max_retries=3)
    captured = capsys.readouterr()
    ci.set_verbose(False)

    assert result == {"ok": True}
    assert "attempt" in captured.err
    assert "test" in captured.err


def test_verbose_safe_get_prints_permanent_failure(monkeypatch, capsys):
    """When verbose, _safe_get should print a skip message on permanent HTTP failures."""

    def fake_urlopen(req, timeout=None):
        raise _make_http_error(404)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    ci._reset_ingest_stats(["test"])
    ci.set_verbose(True)

    result = ci._safe_get("https://example.com/not-found", provider="test")
    captured = capsys.readouterr()
    ci.set_verbose(False)

    assert result is None
    assert "404" in captured.err
    assert "skipping" in captured.err.lower()


# ---------------------------------------------------------------------------
# Retry-After header handling tests
# ---------------------------------------------------------------------------


def test_safe_get_429_retry_after_seconds_header(monkeypatch):
    """_safe_get should honor Retry-After header with integer seconds on 429."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "7"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    assert call_count[0] == 3
    # Both retries should use 7s from Retry-After header
    assert sleep_calls[0] == 7  # Actually _countdown_sleep, but monkeypatch handles both


def test_safe_get_429_retry_after_respects_cap(monkeypatch):
    """_safe_get should cap Retry-After at _SAFE_GET_RETRY_AFTER_MAX_SECONDS."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 2:
            # Server requests 999 seconds, but we should cap at 300
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "999"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # Should use capped value (300 seconds), not the 999 requested
    assert sleep_calls[0] == ci._SAFE_GET_RETRY_AFTER_MAX_SECONDS


def test_safe_get_429_retry_after_invalid_falls_back(monkeypatch):
    """_safe_get should fall back to exponential backoff when Retry-After is invalid."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            # Invalid header format
            err = urllib.error.HTTPError(
                url="https://example.com",
                code=429,
                msg="Too Many Requests",
                hdrs={"Retry-After": "not-a-valid-value"},
                fp=None,
            )
            raise err
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # Should use exponential backoff: 1.0 to 1.2 (first attempt), ~2.0-2.4 (second)
    # Just verify that it's not the invalid header value and retried successfully
    assert call_count[0] == 3


def test_safe_get_non_429_uses_exponential_backoff(monkeypatch):
    """_safe_get should use exponential backoff for 5xx errors, not Retry-After."""
    call_count = [0]
    sleep_calls = []

    def fake_urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] < 3:
            # 503 Service Unavailable should use exponential backoff
            raise _make_http_error(503)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = b'{"ok": true}'
        return mock_resp

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(ci.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ci.time, "sleep", fake_sleep)
    ci._reset_ingest_stats(["test"])

    result = ci._safe_get("https://example.com/test", provider="test")

    assert result == {"ok": True}
    # First retry: 1.0-1.2s range (with jitter)
    # Second retry: 2.0-2.4s range (with jitter)
    # Just verify retries happened and weren't instant
    assert len(sleep_calls) == 2
    assert all(s > 0 for s in sleep_calls)
