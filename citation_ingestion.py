import hashlib
import json
import logging
import math
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    supports_doi_lookup: bool
    supports_reference_expansion: bool
    supports_citation_counts: bool
    supports_cited_by_traversal: bool = False


@dataclass
class IngestionPaper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    keywords: str = ""
    citations: int = 0
    year: Optional[int] = None
    doi: Optional[str] = None
    source_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class IngestionResult:
    citation_data: Dict[str, List[str]]
    papers_data: Dict[str, Dict[str, object]]
    metadata: Dict[str, object]


class CitationProvider:
    name: str = "base"
    capabilities = ProviderCapabilities(False, False, False)

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        raise NotImplementedError

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: int = 500,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        raise NotImplementedError

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        return {}

    def fetch_citers_for_l1(
        self,
        l1_provider_id: str,
        max_results: Optional[int] = None,
    ) -> Tuple[Dict[str, "IngestionPaper"], int, str]:
        """Fetch all papers that cite this L1 paper via provider-native cited-by traversal.

        Args:
            l1_provider_id: Normalized provider ID (e.g. ``"openalex:W123"``).
            max_results: Cap on returned citers; ``None`` exhausts all pages.

        Returns:
            ``(papers, expected_count, status)`` where *status* is one of
            ``"complete"``, ``"partial"``, ``"failed"``, or ``"skipped"``.
        """
        raise NotImplementedError


_CACHE_SCHEMA_VERSION = 5

# HTTP codes that are safe to retry vs. those that indicate a permanent failure.
_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
_PERMANENT_FAILURE_HTTP_CODES = frozenset({400, 401, 403, 404})
_SAFE_GET_MAX_RETRIES = 5
_SAFE_GET_INITIAL_DELAY = 1.0
_SAFE_GET_MAX_DELAY = 60.0
_SAFE_GET_BACKOFF_FACTOR = 2.0
_SAFE_GET_RETRY_AFTER_MAX_SECONDS = 300  # cap on server-specified Retry-After waits

# ---------------------------------------------------------------------------
# Verbose terminal output
# ---------------------------------------------------------------------------

_VERBOSE: bool = False
_VERBOSE_CLEAR_WIDTH = 80  # column width used when clearing a transient line


def set_verbose(flag: bool) -> None:
    """Enable or disable transient terminal progress messages."""
    global _VERBOSE
    _VERBOSE = bool(flag)


def _vprint(msg: str) -> None:
    """Print a permanent progress line to stderr when verbose mode is on."""
    if not _VERBOSE:
        return
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _retry_after_seconds(exc: urllib.error.HTTPError) -> Optional[float]:
    """Parse Retry-After header from HTTP response as seconds to wait.

    Supports both formats:
    - integer seconds (e.g. "120")
    - HTTP-date format (RFC 7231, e.g. "Sun, 06 Nov 2022 08:49:37 GMT")

    Returns None if header is absent or unparseable. If parsed and valid,
    returns the float seconds to wait (caller will clamp to max cap).
    """
    try:
        retry_after = exc.headers.get("Retry-After")
        if not retry_after:
            return None

        # Try parsing as integer seconds first
        try:
            return float(int(retry_after))
        except ValueError:
            pass

        # Try parsing as HTTP-date (RFC 7231)
        # Format: "Sun, 06 Nov 2022 08:49:37 GMT"
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone

            retry_datetime = parsedate_to_datetime(retry_after)
            now = datetime.now(timezone.utc)
            delta_seconds = (retry_datetime - now).total_seconds()
            # Ensure we don't return negative or zero
            return max(0.1, delta_seconds) if delta_seconds > 0 else None
        except (ValueError, TypeError, AttributeError):
            return None
    except Exception:
        return None


def _countdown_sleep(seconds: float, label: str) -> None:
    """Sleep for *seconds*, ticking a live countdown to stderr each second.

    The countdown is transient — each tick overwrites the previous line with
    ``\\r`` so it does not clutter the terminal history.  The line is cleared
    on completion.  Falls back to a plain ``time.sleep`` when verbose mode is
    off or the delay is very short.
    """
    if not _VERBOSE or seconds < 1.0:
        time.sleep(seconds)
        return

    remaining = seconds
    while remaining > 0.0:
        display_secs = math.ceil(remaining)
        sys.stderr.write(f"\r  \u23f3 {label} \u2014 retrying in {display_secs}s...  ")
        sys.stderr.flush()
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk

    # Clear the transient line
    sys.stderr.write("\r" + " " * _VERBOSE_CLEAR_WIDTH + "\r")
    sys.stderr.flush()


_INGEST_STATS = {
    "total_requests": 0,
    "total_failures": 0,
    "per_provider_failures": {},
}


def _http_error_body(exc: urllib.error.HTTPError) -> Optional[str]:
    try:
        body = exc.read()
    except Exception:
        return None

    if not body:
        return None
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace").strip() or None
    return str(body).strip() or None


def _reset_ingest_stats(source_list: Sequence[str]) -> None:
    _INGEST_STATS["total_requests"] = 0
    _INGEST_STATS["total_failures"] = 0
    _INGEST_STATS["per_provider_failures"] = {source: 0 for source in source_list}


def _safe_get(
    url: str,
    timeout: int = 20,
    provider: Optional[str] = None,
    max_retries: int = _SAFE_GET_MAX_RETRIES,
) -> Optional[dict]:
    """Fetch a URL, retrying on transient errors with exponential backoff + jitter.

    Retryable: network/timeout errors and HTTP 429/5xx.
    Permanent failures: HTTP 400/401/403/404 — these return None immediately.
    Exhausted retries: also return None and record to _INGEST_STATS.
    """
    _INGEST_STATS["total_requests"] += 1
    req = urllib.request.Request(url, headers={"User-Agent": "ADIT/0.1 ingestion"})
    delay = _SAFE_GET_INITIAL_DELAY
    last_error: Optional[Exception] = None
    last_error_body: Optional[str] = None

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = _http_error_body(exc)
            last_error_body = error_body
            if exc.code in _PERMANENT_FAILURE_HTTP_CODES:
                _INGEST_STATS["total_failures"] += 1
                if provider:
                    failures = _INGEST_STATS["per_provider_failures"]
                    failures[provider] = int(failures.get(provider, 0)) + 1
                logger.warning(
                    "Permanent HTTP failure: provider=%s status=%s url=%s body=%s",
                    provider,
                    exc.code,
                    url,
                    error_body,
                )
                _vprint(f"  [{provider or 'unknown'}] Permanent HTTP {exc.code} — skipping request")
                return None
            last_error = exc
        except Exception as exc:
            last_error = exc
            last_error_body = None

        if attempt < max_retries - 1:
            # Check if this is a 429 with a Retry-After header; otherwise use exponential backoff
            sleep_time: Optional[float] = None
            retry_strategy: str = "exponential backoff"

            if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
                retry_after = _retry_after_seconds(last_error)
                if retry_after is not None:
                    sleep_time = min(retry_after, _SAFE_GET_RETRY_AFTER_MAX_SECONDS)
                    retry_strategy = f"Retry-After (server requested {retry_after:.0f}s, capped to {sleep_time:.0f}s)"

            # Fall back to exponential backoff if Retry-After was not used
            if sleep_time is None:
                jitter = random.uniform(0.8, 1.2)
                sleep_time = min(delay * jitter, _SAFE_GET_MAX_DELAY)

            logger.debug(
                "Retryable failure (attempt %d/%d): provider=%s error=%s strategy=%s sleeping=%.1fs",
                attempt + 1,
                max_retries,
                provider,
                last_error,
                retry_strategy,
                sleep_time,
            )
            _vprint(
                f"  [{provider or 'unknown'}] HTTP error "
                f"{getattr(last_error, 'code', '?')} — "
                f"attempt {attempt + 1}/{max_retries}, "
                f"waiting {sleep_time:.0f}s ({retry_strategy}) before retry"
            )
            _countdown_sleep(
                sleep_time,
                f"[{provider or 'unknown'}] attempt {attempt + 2}/{max_retries}",
            )
            delay *= _SAFE_GET_BACKOFF_FACTOR

    _INGEST_STATS["total_failures"] += 1
    if provider:
        failures = _INGEST_STATS["per_provider_failures"]
        failures[provider] = int(failures.get(provider, 0)) + 1
    logger.warning(
        "Request failed after %d attempts: provider=%s url=%s final_error=%s body=%s",
        max_retries,
        provider,
        url,
        last_error,
        last_error_body,
    )
    _vprint(f"  [{provider or 'unknown'}] All {max_retries} retries exhausted — skipping request")
    return None


def _norm_title(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", cleaned)


def normalize_identifier(raw_id: str, source: Optional[str] = None) -> str:
    value = (raw_id or "").strip()
    if not value:
        return ""

    lower = value.lower()
    if lower.startswith("https://doi.org/"):
        return f"doi:{lower.replace('https://doi.org/', '')}"
    if lower.startswith("doi:"):
        return f"doi:{lower.replace('doi:', '')}"
    if re.match(r"^10\.\d{4,9}/", lower):
        return f"doi:{lower}"

    # OpenAlex ids often appear as W1234 or https://openalex.org/W1234
    if "openalex.org/" in lower:
        token = lower.split("openalex.org/")[-1].upper()
        return f"openalex:{token}"
    if re.match(r"^[Ww]\d+$", value):
        return f"openalex:{value.upper()}"

    if source:
        return f"{source}:{value}"
    return value


def _canonical_merge_key(paper: IngestionPaper) -> str:
    if paper.doi:
        return normalize_identifier(paper.doi)
    title_key = _norm_title(paper.title)
    if title_key and paper.year:
        return f"titleyear:{title_key}:{paper.year}"
    return f"id:{paper.paper_id}"


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> Optional[dict]:
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _paper_to_output_dict(paper: IngestionPaper) -> Dict[str, object]:
    output = {
        "title": paper.title,
        "abstract": paper.abstract,
        "keywords": paper.keywords,
        "citations": int(paper.citations or 0),
        "year": int(paper.year) if paper.year is not None else None,
    }
    if paper.doi:
        output["doi"] = paper.doi
    if paper.source_ids:
        output["source_ids"] = dict(paper.source_ids)
    return output


def _merge_papers(existing: IngestionPaper, incoming: IngestionPaper) -> IngestionPaper:
    if incoming.title and len(incoming.title) > len(existing.title):
        existing.title = incoming.title
    if incoming.abstract and len(incoming.abstract) > len(existing.abstract):
        existing.abstract = incoming.abstract
    if incoming.keywords and len(incoming.keywords) > len(existing.keywords):
        existing.keywords = incoming.keywords
    if incoming.citations > existing.citations:
        existing.citations = incoming.citations
    if incoming.year and (not existing.year or existing.year == 2010):
        existing.year = incoming.year
    if incoming.doi and not existing.doi:
        existing.doi = incoming.doi
    existing.source_ids.update(incoming.source_ids)
    return existing


def _query_terms(theory_name: str, key_constructs: Optional[Sequence[str]]) -> str:
    terms = [theory_name.strip()]
    if key_constructs:
        terms.extend([k.strip() for k in key_constructs if k.strip()])
    return " ".join([t for t in terms if t])


def _doi_from_identifier(identifier: str) -> Optional[str]:
    if not identifier:
        return None
    if identifier.startswith("doi:"):
        return identifier.split(":", 1)[1]
    return None


def _paper_from_semantic_reference(ref: dict) -> Optional[IngestionPaper]:
    ext_ids = ref.get("externalIds") or {}
    doi = ext_ids.get("DOI")
    paper_id_raw = ref.get("paperId")

    if doi:
        paper_id = normalize_identifier(doi)
    elif paper_id_raw:
        paper_id = normalize_identifier(paper_id_raw, source="semantic_scholar")
    else:
        return None

    source_ids = {}
    if paper_id_raw:
        source_ids["semantic_scholar"] = str(paper_id_raw)

    return IngestionPaper(paper_id=paper_id, doi=doi, source_ids=source_ids)


def _reconstruct_openalex_abstract(inv_idx: dict) -> str:
    if not inv_idx:
        return ""

    reconstructed = []
    for token, positions in inv_idx.items():
        for pos in positions:
            reconstructed.append((int(pos), token))
    return " ".join(token for _, token in sorted(reconstructed, key=lambda x: x[0]))


def _openalex_linked_l1(item: dict, l1_norm: Set[str], theory_name: str) -> List[str]:
    refs = item.get("referenced_works", []) or []
    ref_norm = {normalize_identifier(ref, source="openalex") for ref in refs if ref}
    linked_l1 = sorted(ref_norm.intersection(l1_norm))
    if linked_l1:
        return linked_l1

    title = " ".join(item.get("title", "").split())
    if theory_name.lower() in title.lower():
        return []
    return []


def _should_keep_openalex_item(item: dict, linked_l1: Sequence[str], theory_name: str) -> bool:
    if linked_l1:
        return True
    title = " ".join(item.get("title", "").split())
    return theory_name.lower() in title.lower()


def _paper_from_openalex_item(item: dict, paper_id: str) -> IngestionPaper:
    return IngestionPaper(
        paper_id=paper_id,
        title=item.get("title") or "",
        abstract=_reconstruct_openalex_abstract(item.get("abstract_inverted_index") or {}),
        keywords="",
        citations=int(item.get("cited_by_count") or 0),
        year=int(item.get("publication_year")) if item.get("publication_year") else None,
        doi=item.get("doi"),
        source_ids={"openalex": str(item.get("id", ""))},
    )


def _semantic_linked_l1(item: dict, l1_norm: Set[str]) -> Set[str]:
    linked_l1: Set[str] = set()
    for ref in item.get("references", []) or []:
        ref_pid = ref.get("paperId")
        if ref_pid:
            candidate = normalize_identifier(ref_pid, source="semantic_scholar")
            if candidate in l1_norm:
                linked_l1.add(candidate)
        ext_ids = ref.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        if doi:
            doi_norm = normalize_identifier(doi)
            if doi_norm in l1_norm:
                linked_l1.add(doi_norm)
    return linked_l1


def _should_keep_semantic_item(item: dict, linked_l1: Set[str], theory_name: str) -> bool:
    return bool(linked_l1) or theory_name.lower() in (item.get("title") or "").lower()


def _paper_from_semantic_item(item: dict, paper_id: str) -> IngestionPaper:
    ext_ids = item.get("externalIds") or {}
    return IngestionPaper(
        paper_id=paper_id,
        title=item.get("title") or "",
        abstract=item.get("abstract") or "",
        year=int(item.get("year")) if item.get("year") else None,
        citations=int(item.get("citationCount") or 0),
        doi=ext_ids.get("DOI"),
        source_ids={"semantic_scholar": str(item.get("paperId", ""))},
    )


def _seed_l1_papers(l1_papers: Sequence[str]) -> tuple[List[str], Dict[str, IngestionPaper]]:
    normalized = [normalize_identifier(value) for value in l1_papers if value]
    papers = {}
    for raw, norm in zip(l1_papers, normalized):
        # If the raw L1 looks like a DOI, capture it so canonical merge keys match provider DOIs.
        raw_lower = (raw or "").strip().lower()
        doi_val = None
        if raw_lower.startswith("https://doi.org/"):
            doi_val = raw_lower.replace("https://doi.org/", "")
        elif raw_lower.startswith("doi:"):
            doi_val = raw_lower.replace("doi:", "")
        elif re.match(r"^10\.\d{4,9}/", raw_lower):
            doi_val = raw_lower

        # Create a minimal placeholder with no title and no default year so incoming metadata wins.
        papers[norm] = IngestionPaper(paper_id=norm, title="", year=None, doi=doi_val)
    return normalized, papers


def _merge_provider_outputs(
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    edges: Dict[str, Set[str]],
    papers: Dict[str, IngestionPaper],
) -> None:
    for citing, cited in edges.items():
        all_edges.setdefault(citing, set()).update(cited)
    for pid, paper in papers.items():
        if pid in all_papers:
            all_papers[pid] = _merge_papers(all_papers[pid], paper)
        else:
            all_papers[pid] = paper


def _fetch_provider_graph(
    provider: CitationProvider,
    l1_norm: Sequence[str],
    l1_papers_resolved: Dict[str, IngestionPaper],
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    exhaustive: bool,
    max_l2: int,
    max_l3: int,
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], Dict[str, object]]:
    """Fetch L2 (and optionally L3) papers for a provider.

    When the provider supports cited-by traversal, each L1 seed is queried
    individually using its resolved provider-native ID so all citers are
    returned (no title/keyword gate).  Falls back to keyword search for
    providers that don't support direct cited-by lookup (e.g. Crossref).
    """
    l2_edges: Dict[str, Set[str]] = {}
    l2_papers: Dict[str, IngestionPaper] = {}
    completeness: Dict[str, Dict[str, object]] = {}
    max_results: Optional[int] = None if exhaustive else max_l2

    if provider.capabilities.supports_cited_by_traversal:
        for l1_id in l1_norm:
            seed = l1_papers_resolved.get(l1_id)
            raw_provider_id = seed.source_ids.get(provider.name) if seed else None
            if not raw_provider_id:
                _vprint(f"  [{provider.name}] Skipping {l1_id} — no provider ID resolved")
                completeness[l1_id] = {
                    provider.name: {
                        "status": "skipped",
                        "reason": "no_provider_id",
                        "fetched": 0,
                        "expected": 0,
                    }
                }
                continue

            l1_native_id = normalize_identifier(raw_provider_id, source=provider.name)
            _vprint(f"  [{provider.name}] Fetching citers for seed: {l1_id}")
            papers_for_l1, expected_count, status = provider.fetch_citers_for_l1(
                l1_native_id, max_results=max_results
            )
            _vprint(
                f"  [{provider.name}] \u2192 {len(papers_for_l1)}"
                f"/{expected_count or '?'} papers ({status})"
            )
            completeness.setdefault(l1_id, {})[provider.name] = {
                "status": status,
                "fetched": len(papers_for_l1),
                "expected": expected_count,
            }
            for paper_id, paper in papers_for_l1.items():
                l2_edges.setdefault(paper_id, set()).add(l1_id)
                if paper_id in l2_papers:
                    l2_papers[paper_id] = _merge_papers(l2_papers[paper_id], paper)
                else:
                    l2_papers[paper_id] = paper
    else:
        # Keyword-search fallback: used for Crossref and any provider lacking cited-by support.
        search_edges, search_papers = provider.fetch_l2_and_metadata(
            l1_papers=l1_norm,
            theory_name=theory_name,
            key_constructs=key_constructs,
            max_l2=max_l2,
        )
        l2_edges.update(search_edges)
        l2_papers.update(search_papers)

    all_edges = dict(l2_edges)
    all_papers = dict(l2_papers)
    added_l3_edges = 0

    if depth.lower() in {"l2l3", "l3", "2"}:
        l3_edges, l3_papers = provider.fetch_l3_references(
            l2_paper_ids=list(l2_edges.keys()),
            max_l3=max_l3,
        )
        _merge_provider_outputs(all_edges, all_papers, l3_edges, l3_papers)
        added_l3_edges = sum(len(cited) for cited in l3_edges.values())

    stats: Dict[str, object] = {
        "l2_nodes": len(l2_edges),
        "l2_edges": sum(len(v) for v in l2_edges.values()),
        "l3_edges": added_l3_edges,
        "papers": len(l2_papers),
        "completeness": completeness,
    }
    return all_edges, all_papers, stats


def _merge_seed_metadata(
    all_papers: Dict[str, IngestionPaper],
    seed_papers: Dict[str, IngestionPaper],
) -> None:
    for pid, paper in seed_papers.items():
        if pid in all_papers:
            all_papers[pid] = _merge_papers(all_papers[pid], paper)
        else:
            all_papers[pid] = paper


def _request_payload(
    theory_name: str,
    l1_papers: Sequence[str],
    key_constructs: Optional[Sequence[str]],
    sources: Sequence[str],
    depth: str,
    max_l2: int,
    max_l3: int,
    exhaustive: bool,
) -> dict:
    return {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "theory_name": theory_name,
        "l1_papers": list(l1_papers),
        "key_constructs": list(key_constructs or []),
        "sources": list(sources),
        "depth": depth,
        "max_l2": int(max_l2),
        "max_l3": int(max_l3),
        "exhaustive": bool(exhaustive),
    }


def _load_cached_result(cache_root: Path, key: str, refresh: bool) -> Optional[IngestionResult]:
    if refresh:
        return None
    cached = _read_cache(cache_root, key)
    if not cached:
        return None
    return IngestionResult(
        citation_data=cached.get("citation_data", {}),
        papers_data=cached.get("papers_data", {}),
        metadata=cached.get("metadata", {}),
    )


def _build_metadata(
    source_list: Sequence[str],
    depth: str,
    provider_stats: Dict[str, object],
    cache_key: str,
    alias_map: Dict[str, str],
    papers_data: Dict[str, Dict[str, object]],
    citation_data: Dict[str, List[str]],
    completeness: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    return {
        "sources": list(source_list),
        "depth": depth,
        "provider_stats": provider_stats,
        "fetch_stats": {
            "total_requests": int(_INGEST_STATS["total_requests"]),
            "total_failures": int(_INGEST_STATS["total_failures"]),
            "per_provider_failures": dict(_INGEST_STATS["per_provider_failures"]),
        },
        "completeness": completeness,
        "cache_key": cache_key,
        "alias_count": len(alias_map),
        "paper_count": len(papers_data),
        "edge_count": sum(len(values) for values in citation_data.values()),
    }


class OpenAlexProvider(CitationProvider):
    name = "openalex"
    capabilities = ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        papers: Dict[str, IngestionPaper] = {}
        for paper_id in l1_papers:
            payload = None
            doi = _doi_from_identifier(paper_id)
            if doi:
                query = urllib.parse.urlencode({"filter": f"doi:{doi}", "per-page": 1})
                result = _safe_get(f"https://api.openalex.org/works?{query}", provider=self.name)
                matches = (result or {}).get("results", [])
                payload = matches[0] if matches else None
            elif paper_id.startswith("openalex:"):
                token = paper_id.split(":", 1)[1]
                payload = _safe_get(f"https://api.openalex.org/works/{token}", provider=self.name)

            if not payload:
                continue

            papers[paper_id] = _paper_from_openalex_item(payload, paper_id)
            time.sleep(0.03)
        return papers

    def fetch_citers_for_l1(
        self,
        l1_provider_id: str,
        max_results: Optional[int] = None,
    ) -> Tuple[Dict[str, IngestionPaper], int, str]:
        """Fetch all papers citing this OpenAlex work via cursor-paginated cited-by traversal."""
        papers: Dict[str, IngestionPaper] = {}
        # Strip "openalex:" prefix to get bare work ID for the API filter.
        work_id = l1_provider_id.split(":", 1)[-1] if ":" in l1_provider_id else l1_provider_id
        cursor = "*"
        expected_count: Optional[int] = None

        while True:
            per_page = 200
            if max_results is not None:
                remaining = max_results - len(papers)
                if remaining <= 0:
                    break
                per_page = min(per_page, remaining)

            params = urllib.parse.urlencode(
                {
                    "filter": f"cites:{work_id}",
                    "per-page": per_page,
                    "cursor": cursor,
                    "select": "id,title,publication_year,cited_by_count,doi,abstract_inverted_index",
                }
            )
            payload = _safe_get(f"https://api.openalex.org/works?{params}", provider=self.name)
            if not payload:
                status = "failed" if not papers else "partial"
                return papers, expected_count or 0, status

            if expected_count is None:
                meta = payload.get("meta") or {}
                expected_count = int(meta.get("count") or 0)

            results = payload.get("results") or []
            for item in results:
                oa_id = item.get("id") or ""
                paper_id = normalize_identifier(oa_id, source=self.name)
                if paper_id:
                    papers[paper_id] = _paper_from_openalex_item(item, paper_id)

            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if not next_cursor or not results:
                break
            cursor = next_cursor
            time.sleep(0.03)

        fetched = len(papers)
        status = "complete" if (expected_count is None or fetched >= expected_count) else "partial"
        return papers, expected_count or fetched, status

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        citation_edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        query = urllib.parse.quote(_query_terms(theory_name, key_constructs))
        url = f"https://api.openalex.org/works?search={query}&per-page={max(1, min(max_l2, 200))}"
        payload = _safe_get(url, provider=self.name)
        if not payload:
            return citation_edges, papers

        l1_norm = {normalize_identifier(v) for v in l1_papers}
        for item in payload.get("results", []):
            openalex_id = item.get("id", "")
            paper_id = normalize_identifier(openalex_id, source=self.name)
            if not paper_id:
                continue

            linked_l1 = _openalex_linked_l1(item, l1_norm, theory_name)
            if not _should_keep_openalex_item(item, linked_l1, theory_name):
                continue

            citation_edges.setdefault(paper_id, set()).update(linked_l1)
            papers[paper_id] = _paper_from_openalex_item(item, paper_id)

            # Minimal metadata entries for referenced L1 papers if missing.
            for ref in linked_l1:
                papers.setdefault(ref, IngestionPaper(paper_id=ref, doi=_doi_from_identifier(ref)))

            time.sleep(0.03)

        return citation_edges, papers

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: int = 500,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        budget = max(0, max_l3)

        for pid in l2_paper_ids:
            if budget <= 0:
                break
            if not pid.startswith("openalex:"):
                continue

            openalex_token = pid.split(":", 1)[1]
            url = f"https://api.openalex.org/works/{openalex_token}?select=referenced_works"
            payload = _safe_get(url, provider=self.name)
            if not payload:
                continue

            refs = payload.get("referenced_works", []) or []
            for ref in refs:
                if budget <= 0:
                    break
                ref_id = normalize_identifier(ref, source=self.name)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                token = ref_id.split(":", 1)[1] if ref_id.startswith("openalex:") else ""
                source_id = f"https://openalex.org/{token}" if token else str(ref)
                papers.setdefault(
                    ref_id,
                    IngestionPaper(
                        paper_id=ref_id,
                        doi=_doi_from_identifier(ref_id),
                        source_ids={"openalex": source_id},
                    ),
                )
                budget -= 1
            time.sleep(0.03)

        # Identity hydration pass: keep the first step lightweight (edge discovery),
        # then fetch only minimal identity fields needed for cross-provider dedup.
        for ref_id, existing in list(papers.items()):
            if not ref_id.startswith("openalex:"):
                continue

            token = ref_id.split(":", 1)[1]
            url = f"https://api.openalex.org/works/{token}?select=id,doi,title,publication_year"
            payload = _safe_get(url, provider=self.name)
            if not payload:
                continue

            year_raw = payload.get("publication_year")
            hydrated = IngestionPaper(
                paper_id=ref_id,
                title=payload.get("title") or "",
                year=int(year_raw) if year_raw else None,
                doi=payload.get("doi"),
                source_ids={"openalex": str(payload.get("id") or f"https://openalex.org/{token}")},
            )
            papers[ref_id] = _merge_papers(existing, hydrated)
            time.sleep(0.03)

        return edges, papers


class SemanticScholarProvider(CitationProvider):
    name = "semantic_scholar"
    capabilities = ProviderCapabilities(True, True, True, supports_cited_by_traversal=True)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        papers: Dict[str, IngestionPaper] = {}
        for paper_id in l1_papers:
            payload = None
            doi = _doi_from_identifier(paper_id)
            if doi:
                encoded = urllib.parse.quote(doi, safe="")
                payload = _safe_get(
                    "https://api.semanticscholar.org/graph/v1/paper/DOI:"
                    f"{encoded}?fields=paperId,title,abstract,year,citationCount,externalIds",
                    provider=self.name,
                )
            elif paper_id.startswith("semantic_scholar:"):
                token = paper_id.split(":", 1)[1]
                payload = _safe_get(
                    "https://api.semanticscholar.org/graph/v1/paper/"
                    f"{token}?fields=paperId,title,abstract,year,citationCount,externalIds",
                    provider=self.name,
                )

            if not payload:
                continue

            papers[paper_id] = _paper_from_semantic_item(payload, paper_id)
            time.sleep(0.05)
        return papers

    def fetch_citers_for_l1(
        self,
        l1_provider_id: str,
        max_results: Optional[int] = None,
    ) -> Tuple[Dict[str, IngestionPaper], int, str]:
        """Fetch all papers citing this S2 paper via offset-paginated citations endpoint."""
        papers: Dict[str, IngestionPaper] = {}
        token = l1_provider_id.split(":", 1)[-1] if ":" in l1_provider_id else l1_provider_id
        offset = 0
        limit = 1000
        expected_count: Optional[int] = None

        while True:
            batch_limit = limit
            if max_results is not None:
                remaining = max_results - len(papers)
                if remaining <= 0:
                    break
                batch_limit = min(limit, remaining)

            # Semantic Scholar enforces `offset + limit < 10000` and returns
            # HTTP 400 if that boundary is crossed ("offset + limit must be < 10000").
            # Compute the maximum allowed `limit` for the next request so we
            # never issue a request where `offset + limit >= 10000`.
            max_allowed = 9999 - offset
            if max_allowed <= 0:
                fetched = len(papers)
                status = (
                    "complete"
                    if expected_count is not None and fetched >= expected_count
                    else "partial"
                )
                return papers, expected_count or fetched, status

            batch_limit = min(batch_limit, max_allowed)
            if batch_limit <= 0:
                fetched = len(papers)
                status = (
                    "complete"
                    if expected_count is not None and fetched >= expected_count
                    else "partial"
                )
                return papers, expected_count or fetched, status

            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/{token}/citations"
                f"?fields=paperId,title,year,citationCount,externalIds,abstract"
                f"&limit={batch_limit}&offset={offset}"
            )
            payload = _safe_get(url, provider=self.name)
            if not payload:
                status = "failed" if not papers else "partial"
                return papers, expected_count or 0, status

            data = payload.get("data") or []
            if expected_count is None and "total" in payload:
                expected_count = int(payload.get("total") or 0)

            for item_wrapper in data:
                item = item_wrapper.get("citingPaper") or {}
                pid_raw = item.get("paperId")
                if not pid_raw:
                    continue
                paper_id = normalize_identifier(pid_raw, source=self.name)
                papers[paper_id] = _paper_from_semantic_item(item, paper_id)

            # S2 provides a "next" URL when more pages exist; absence or short page means done.
            if not data or not payload.get("next"):
                break
            offset += len(data)
            time.sleep(0.05)

        fetched = len(papers)
        status = "complete" if (expected_count is None or fetched >= expected_count) else "partial"
        return papers, expected_count or fetched, status

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        query = urllib.parse.quote(_query_terms(theory_name, key_constructs))
        limit = max(1, min(max_l2, 100))
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={query}&limit={limit}&fields=paperId,title,year,citationCount,references.paperId,references.externalIds"
        )
        payload = _safe_get(url, provider=self.name)
        if not payload:
            return edges, papers

        l1_norm = {normalize_identifier(v) for v in l1_papers}
        for item in payload.get("data", []):
            paper_id_raw = item.get("paperId")
            if not paper_id_raw:
                continue
            paper_id = normalize_identifier(paper_id_raw, source=self.name)

            linked_l1 = _semantic_linked_l1(item, l1_norm)
            if not _should_keep_semantic_item(item, linked_l1, theory_name):
                continue

            edges.setdefault(paper_id, set()).update(linked_l1)
            papers[paper_id] = _paper_from_semantic_item(item, paper_id)
            for ref in linked_l1:
                papers.setdefault(ref, IngestionPaper(paper_id=ref, doi=_doi_from_identifier(ref)))
            time.sleep(0.05)

        return edges, papers

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: int = 500,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        budget = max(0, max_l3)

        for pid in l2_paper_ids:
            if budget <= 0:
                break
            if not pid.startswith("semantic_scholar:"):
                continue

            token = pid.split(":", 1)[1]
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"{token}?fields=references.paperId,references.externalIds"
            )
            payload = _safe_get(url, provider=self.name)
            if not payload:
                continue

            for ref in payload.get("references", []) or []:
                if budget <= 0:
                    break
                paper = _paper_from_semantic_reference(ref)
                if not paper:
                    continue
                ref_id = paper.paper_id
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, paper)
                budget -= 1
            time.sleep(0.05)

        return edges, papers


class CrossrefProvider(CitationProvider):
    name = "crossref"
    capabilities = ProviderCapabilities(True, False, False)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        papers: Dict[str, IngestionPaper] = {}
        for paper_id in l1_papers:
            doi = _doi_from_identifier(paper_id)
            if not doi:
                continue

            payload = _safe_get(
                f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                provider=self.name,
            )
            item = (payload or {}).get("message")
            if not item:
                continue

            titles = item.get("title") or []
            title = titles[0] if titles else ""
            issued = item.get("issued", {}).get("date-parts", [[]])
            year = issued[0][0] if issued and issued[0] else None
            citations = int(item.get("is-referenced-by-count") or 0)
            papers[paper_id] = IngestionPaper(
                paper_id=paper_id,
                title=title,
                year=int(year) if year else None,
                citations=citations,
                doi=doi,
                source_ids={self.name: doi},
            )
            time.sleep(0.03)
        return papers

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        # Crossref offers metadata but does not provide full cited-by/reference graph richness.
        # We still surface candidates for supplemental metadata coverage.
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        query = urllib.parse.quote(_query_terms(theory_name, key_constructs))
        rows = max(1, min(max_l2, 100))
        url = f"https://api.crossref.org/works?query={query}&rows={rows}"
        payload = _safe_get(url, provider=self.name)
        if not payload:
            return edges, papers

        for item in payload.get("message", {}).get("items", []):
            doi = item.get("DOI")
            if not doi:
                continue
            paper_id = normalize_identifier(doi)
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            issued = item.get("issued", {}).get("date-parts", [[]])
            year = issued[0][0] if issued and issued[0] else None
            citations = int(item.get("is-referenced-by-count") or 0)
            papers[paper_id] = IngestionPaper(
                paper_id=paper_id,
                title=title,
                year=int(year) if year else None,
                citations=citations,
                doi=doi,
                source_ids={self.name: doi},
            )
            # No edges added by default due to limited cited-by graph support.
            time.sleep(0.03)

        return edges, papers

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: int = 500,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        return {}, {}


_PROVIDER_REGISTRY = {
    "openalex": OpenAlexProvider,
    "semantic_scholar": SemanticScholarProvider,
    "crossref": CrossrefProvider,
}


def build_providers(sources: Sequence[str]) -> List[CitationProvider]:
    providers: List[CitationProvider] = []
    for source in sources:
        key = source.strip().lower()
        provider_cls = _PROVIDER_REGISTRY.get(key)
        if provider_cls is None:
            continue
        providers.append(provider_cls())
    return providers


def _dedupe_and_materialize(
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, object]], Dict[str, str]]:
    merged_by_key: Dict[str, IngestionPaper] = {}
    key_to_final_id: Dict[str, str] = {}

    for paper in all_papers.values():
        key = _canonical_merge_key(paper)
        if key in merged_by_key:
            merged_by_key[key] = _merge_papers(merged_by_key[key], paper)
        else:
            merged_by_key[key] = paper

    papers_out: Dict[str, Dict[str, object]] = {}
    alias_to_final: Dict[str, str] = {}

    for key, paper in merged_by_key.items():
        final_id = paper.paper_id
        if final_id in papers_out:
            final_id = key
            paper.paper_id = final_id
        papers_out[final_id] = _paper_to_output_dict(paper)
        key_to_final_id[key] = final_id

    for paper in all_papers.values():
        alias_to_final[paper.paper_id] = key_to_final_id[_canonical_merge_key(paper)]

    citation_out: Dict[str, List[str]] = {}
    for citing, cited_set in all_edges.items():
        citing_final = alias_to_final.get(citing, citing)
        normalized_cited = sorted({alias_to_final.get(c, c) for c in cited_set if c})
        citation_out.setdefault(citing_final, [])
        citation_out[citing_final] = sorted(set(citation_out[citing_final]).union(normalized_cited))

    return citation_out, papers_out, alias_to_final


def ingest_from_internet(
    theory_name: str,
    l1_papers: Sequence[str],
    key_constructs: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    depth: str = "l2l3",
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    max_l2: int = 200,
    max_l3: int = 500,
    exhaustive: bool = True,
    verbose: bool = False,
) -> IngestionResult:
    """Ingest citation data from internet providers.

    Args:
        exhaustive: When ``True`` (default), providers that support cited-by
            traversal paginate until all citers are fetched.  When ``False``,
            retrieval is capped at *max_l2* results per L1/provider.
        verbose: When ``True``, print live progress messages to stderr
            (per-seed status, retry countdowns).
    """
    set_verbose(verbose)
    source_list = ["openalex", "semantic_scholar", "crossref"] if not sources else list(sources)
    _reset_ingest_stats(source_list)
    providers = build_providers(source_list)
    _vprint(
        f"[ADIT] Starting ingestion: theory='{theory_name}', "
        f"seeds={len(l1_papers)}, providers={source_list}"
    )

    cache_root = cache_dir or Path(".cache") / "adit_ingestion"
    request_payload = _request_payload(
        theory_name,
        l1_papers,
        key_constructs,
        source_list,
        depth,
        max_l2,
        max_l3,
        exhaustive,
    )
    key = _cache_key(request_payload)

    cached = _load_cached_result(cache_root, key, refresh)
    if cached:
        return cached

    all_edges: Dict[str, Set[str]] = {}
    l1_norm, all_papers = _seed_l1_papers(l1_papers)
    provider_stats: Dict[str, object] = {}
    combined_completeness: Dict[str, Dict[str, object]] = {}

    for provider in providers:
        _vprint(f"\n[ADIT] Querying provider: {provider.name}")
        _merge_seed_metadata(all_papers, provider.fetch_seed_metadata(l1_norm))
        provider_edges, provider_papers, stats = _fetch_provider_graph(
            provider=provider,
            l1_norm=l1_norm,
            l1_papers_resolved=all_papers,
            theory_name=theory_name,
            key_constructs=key_constructs,
            depth=depth,
            exhaustive=exhaustive,
            max_l2=max_l2,
            max_l3=max_l3,
        )
        _merge_provider_outputs(all_edges, all_papers, provider_edges, provider_papers)
        provider_stats[provider.name] = stats
        for l1_id, l1_completeness in stats.get("completeness", {}).items():
            combined_completeness.setdefault(l1_id, {}).update(l1_completeness)

    citation_data, papers_data, alias_map = _dedupe_and_materialize(all_edges, all_papers)

    metadata = _build_metadata(
        source_list,
        depth,
        provider_stats,
        key,
        alias_map,
        papers_data,
        citation_data,
        combined_completeness,
    )

    result_payload = {
        "citation_data": citation_data,
        "papers_data": papers_data,
        "metadata": metadata,
    }
    _write_cache(cache_root, key, result_payload)
    _vprint(f"\n[ADIT] Done: {metadata['paper_count']} papers, {metadata['edge_count']} edges")

    return IngestionResult(citation_data=citation_data, papers_data=papers_data, metadata=metadata)
