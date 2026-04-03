import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    """Declares which ingestion features a provider implementation supports."""

    supports_doi_lookup: bool
    supports_reference_expansion: bool
    supports_citation_counts: bool
    supports_cited_by_traversal: bool = False
    supports_l3_outgoing: bool = False


@dataclass
class IngestionPaper:
    """Canonical in-memory representation of a paper across all providers."""

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
    """Final normalized graph, paper metadata table, and run metadata."""

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
        """Return L2 citation edges and metadata for discovered papers."""
        raise NotImplementedError

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Return L3 reference edges and metadata for provided L2 papers."""
        raise NotImplementedError

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        """Resolve provider metadata for the seed papers when available."""
        return {}

    def fetch_citers_for_l1(
        self,
        l1_provider_id: str,
        max_results: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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

    def fetch_l3_outgoing_references(
        self,
        l3_paper_ids: Sequence[str],
        max_edges: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Return outgoing references from L3 papers for L3-to-L3 edge discovery.

        Default no-op for providers that do not declare
        ``supports_l3_outgoing``.  Edges and paper stubs returned here will
        be filtered by the orchestrator to retain only targets already present
        in the L3 membership set.
        """
        return {}, {}


_CACHE_SCHEMA_VERSION = 5
_CHECKPOINT_SCHEMA_VERSION = 4
_COORDINATOR_CHECKPOINT_SCHEMA_VERSION = 2
_PROVIDER_CHECKPOINT_SCHEMA_VERSION = 2

# HTTP codes that are safe to retry vs. those that indicate a permanent failure.
_PERMANENT_FAILURE_HTTP_CODES = frozenset({400, 401, 403, 404})
_SAFE_GET_MAX_RETRIES = 5
_SAFE_GET_INITIAL_DELAY = 1.0
_SAFE_GET_MAX_DELAY = 60.0
_SAFE_GET_BACKOFF_FACTOR = 2.0
_SAFE_GET_RETRY_AFTER_MAX_SECONDS = 300  # cap on server-specified Retry-After waits
_PAGINATION_STATE_MAX_AGE_SECONDS = 6 * 60 * 60
_FAILURE_SUMMARY_REQUEST_INTERVAL = 50
_FAILURE_SUMMARY_SECONDS_INTERVAL = 15.0
_DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPTS = 5
_DEFAULT_TRANSIENT_RETRY_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_TRANSIENT_RETRY_MAX_ATTEMPTS = _DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPTS
_TRANSIENT_RETRY_MAX_AGE_SECONDS = _DEFAULT_TRANSIENT_RETRY_MAX_AGE_SECONDS

_SENSITIVE_HEADER_NAMES = frozenset({"authorization", "proxy-authorization", "cookie", "x-api-key"})

# ---------------------------------------------------------------------------
# Verbose terminal output
# ---------------------------------------------------------------------------

_VERBOSE: bool = False
_QUIET: bool = False
_DEBUG_HTTP: bool = False
_VERBOSE_CLEAR_WIDTH = 80  # column width used when clearing a transient line
_TRANSIENT_PROGRESS_ACTIVE: bool = False
_TRANSIENT_PROGRESS_LAST_LEN: int = 0
_PROVIDER_TQDM_ACTIVE: bool = False
_PROVIDER_TQDM_DEFAULT_NCOLS = 110
_PROVIDER_TQDM_MIN_NCOLS = 96
_PROVIDER_TQDM_MAX_NCOLS = 130
_PROVIDER_TQDM_DESC_MIN_WIDTH = 24
_PROVIDER_TQDM_DESC_MAX_WIDTH = 34


def _provider_tqdm_ncols() -> int:
    """Return a stable width that fits the active terminal when possible."""
    terminal_cols = shutil.get_terminal_size((
        _PROVIDER_TQDM_DEFAULT_NCOLS,
        20,
    )).columns
    preferred_cols = max(_PROVIDER_TQDM_MIN_NCOLS, terminal_cols - 2)
    return min(_PROVIDER_TQDM_MAX_NCOLS, preferred_cols)


def _provider_tqdm_desc(provider_name: str, phase_label: str) -> str:
    """Create a fixed-width provider label to keep bars visually aligned."""
    return f"[{provider_name}] {phase_label}"


def _provider_tqdm_desc_width(provider_names: Sequence[str], phase_label: str) -> int:
    """Compute one shared desc width so bars in a block remain column-aligned."""
    if not provider_names:
        return _PROVIDER_TQDM_DESC_MIN_WIDTH

    longest = max(len(_provider_tqdm_desc(name, phase_label)) for name in provider_names)
    return max(_PROVIDER_TQDM_DESC_MIN_WIDTH, min(_PROVIDER_TQDM_DESC_MAX_WIDTH, longest))


def _create_provider_tqdm(
    provider_name: str,
    phase_label: str,
    total: int,
    position: int,
    unit: str,
    desc_width: int,
) -> tqdm:
    """Create a provider progress bar with consistent sizing and layout."""
    ncols = _provider_tqdm_ncols()
    # Reserve width for static fields and use the remainder for the visual bar.
    bar_width = max(14, min(34, ncols - (desc_width + 48)))
    return tqdm(
        total=max(1, int(total or 1)),
        desc=_provider_tqdm_desc(provider_name, phase_label),
        position=position,
        leave=True,
        dynamic_ncols=False,
        ncols=ncols,
        bar_format=(
            f"{{desc:<{desc_width}}} "
            f"|{{bar:{bar_width}}}| "
            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
        unit=unit,
        file=sys.stderr,
    )
_PROGRESS_LOCK = threading.RLock()
_STATS_LOCK = threading.Lock()
_TRANSIENT_FAILURES_LOCK = threading.Lock()


def set_verbose(flag: bool) -> None:
    """Enable or disable transient terminal progress messages."""
    global _VERBOSE
    _VERBOSE = bool(flag)


def set_quiet(flag: bool) -> None:
    """Enable or disable standard ingestion progress messages."""
    global _QUIET
    _QUIET = bool(flag)


def set_debug_http(flag: bool) -> None:
    """Enable or disable HTTP response-body diagnostics for failed requests."""
    global _DEBUG_HTTP
    _DEBUG_HTTP = bool(flag)


def _stderr_is_tty() -> bool:
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _clear_transient_progress_line() -> None:
    """Clear active in-place progress output so permanent lines stay readable."""
    global _TRANSIENT_PROGRESS_ACTIVE, _TRANSIENT_PROGRESS_LAST_LEN
    with _PROGRESS_LOCK:
        if not _TRANSIENT_PROGRESS_ACTIVE:
            return
        if _stderr_is_tty():
            clear_width = max(_VERBOSE_CLEAR_WIDTH, _TRANSIENT_PROGRESS_LAST_LEN)
            sys.stderr.write("\r" + " " * clear_width + "\r")
            sys.stderr.flush()
        _TRANSIENT_PROGRESS_ACTIVE = False
        _TRANSIENT_PROGRESS_LAST_LEN = 0


def _stderr_supports_color() -> bool:
    if _QUIET:
        return False
    if os.getenv("NO_COLOR"):
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _progress(msg: str) -> None:
    """Print always-on milestone progress lines to stderr unless quiet mode is enabled."""
    if _QUIET:
        return
    with _PROGRESS_LOCK:
        _clear_transient_progress_line()
        if _PROVIDER_TQDM_ACTIVE:
            tqdm.write(msg, file=sys.stderr)
        else:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()


def _progress_done(msg: str) -> None:
    """Print completed milestone progress lines with a checkmark marker."""
    if _QUIET:
        return
    with _PROGRESS_LOCK:
        _clear_transient_progress_line()
        if _stderr_supports_color():
            prefix = "\033[32m✓\033[0m "
        else:
            prefix = "✓ "
        if _PROVIDER_TQDM_ACTIVE:
            tqdm.write(prefix + msg, file=sys.stderr)
        else:
            sys.stderr.write(prefix + msg + "\n")
            sys.stderr.flush()


def _progress_inline(msg: str) -> None:
    """Print transient in-place progress updates for interactive terminals."""
    global _TRANSIENT_PROGRESS_ACTIVE, _TRANSIENT_PROGRESS_LAST_LEN
    if _QUIET or _PROVIDER_TQDM_ACTIVE:
        return
    if not _stderr_is_tty():
        _progress(msg)
        return

    with _PROGRESS_LOCK:
        padded_msg = msg
        if _TRANSIENT_PROGRESS_ACTIVE and _TRANSIENT_PROGRESS_LAST_LEN > len(msg):
            padded_msg = msg + " " * (_TRANSIENT_PROGRESS_LAST_LEN - len(msg))

        sys.stderr.write("\r" + padded_msg)
        sys.stderr.flush()
        _TRANSIENT_PROGRESS_ACTIVE = True
        _TRANSIENT_PROGRESS_LAST_LEN = len(msg)


def _vprint(msg: str) -> None:
    """Print a permanent progress line to stderr when verbose mode is on."""
    if _QUIET or not _VERBOSE:
        return
    with _PROGRESS_LOCK:
        if _PROVIDER_TQDM_ACTIVE:
            tqdm.write(msg, file=sys.stderr)
        else:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()


def _update_provider_tqdm_bar(
    provider_bars: Dict[str, tqdm],
    provider_name: str,
    completed: int,
    total: int,
    status: Optional[str],
    lock: threading.Lock,
) -> None:
    """Safely update one provider progress bar from sequential/parallel workers."""
    bar = provider_bars.get(provider_name)
    if bar is None:
        return

    bounded_total = max(int(total or 1), 1)
    bounded_completed = max(0, min(int(completed), bounded_total))
    with lock:
        if bar.total != bounded_total:
            bar.total = bounded_total
        bar.n = bounded_completed
        if status:
            bar.set_postfix_str(status)
        bar.refresh()


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
            from datetime import datetime, timezone
            from email.utils import parsedate_to_datetime

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
    if _QUIET or not _VERBOSE or seconds < 1.0:
        time.sleep(seconds)
        return

    remaining = seconds
    while remaining > 0.0:
        display_secs = math.ceil(remaining)
        with _PROGRESS_LOCK:
            sys.stderr.write(f"\r  \u23f3 {label} \u2014 retrying in {display_secs}s...  ")
            sys.stderr.flush()
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk

    # Clear the transient line
    with _PROGRESS_LOCK:
        sys.stderr.write("\r" + " " * _VERBOSE_CLEAR_WIDTH + "\r")
        sys.stderr.flush()


_INGEST_STATS = {
    "total_requests": 0,
    "total_failures": 0,
    "per_provider_failures": {},
    "per_status_failures": {},
    "last_summary_request_count": 0,
    "last_summary_timestamp": 0.0,
}

_TRANSIENT_REQUEST_FAILURES: List[Dict[str, Any]] = []


def _sanitize_retry_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Drop sensitive headers before persisting request replay metadata."""
    if not isinstance(headers, dict):
        return {}
    sanitized: Dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.lower() in _SENSITIVE_HEADER_NAMES:
            continue
        sanitized[key] = value
    return sanitized


def _transient_failure_key(record: Dict[str, Any]) -> str:
    """Build a deterministic identity for one transient retry record."""
    op = str(record.get("op") or "safe_get")
    provider = str(record.get("provider") or "unknown")
    target = str(record.get("target_id") or "")
    return f"{provider}:{op}:{target}"


def _record_transient_request_failure(
    provider: Optional[str],
    url: str,
    timeout: int,
    max_retries: int,
    headers: Optional[Dict[str, str]],
    last_error: Optional[Exception],
    last_error_body: Optional[str],
) -> None:
    """Store retry-exhausted transient request metadata for checkpoint replay."""
    if not provider:
        return

    retry_after_seconds: Optional[float] = None
    status_code: Optional[int] = None
    error_type = "unknown"
    if isinstance(last_error, urllib.error.HTTPError):
        status_code = int(last_error.code)
        error_type = "http_error"
        retry_after_seconds = _retry_after_seconds(last_error)
        if retry_after_seconds is not None:
            retry_after_seconds = min(retry_after_seconds, _SAFE_GET_RETRY_AFTER_MAX_SECONDS)
    elif last_error is not None:
        error_type = type(last_error).__name__

    payload: Dict[str, Any] = {
        "op": "safe_get",
        "provider": provider,
        "target_id": url,
        "resume_state": {
            "timeout": int(timeout),
            "max_retries": int(max_retries),
            "headers": _sanitize_retry_headers(headers),
        },
        "error_code": status_code,
        "error_type": error_type,
        "attempts": 1,
        "last_attempt_ts": time.time(),
        "server_retry_after": retry_after_seconds,
    }
    if _DEBUG_HTTP and last_error_body:
        payload["error_hint"] = re.sub(r"\s+", " ", str(last_error_body)).strip()[:1024]

    with _TRANSIENT_FAILURES_LOCK:
        _TRANSIENT_REQUEST_FAILURES.append(payload)


def _drain_transient_request_failures(provider: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pop queued transient request failures, optionally filtering by provider."""
    with _TRANSIENT_FAILURES_LOCK:
        if provider is None:
            drained = list(_TRANSIENT_REQUEST_FAILURES)
            _TRANSIENT_REQUEST_FAILURES.clear()
            return drained

        kept: List[Dict[str, Any]] = []
        drained: List[Dict[str, Any]] = []
        for payload in _TRANSIENT_REQUEST_FAILURES:
            if payload.get("provider") == provider:
                drained.append(payload)
            else:
                kept.append(payload)
        _TRANSIENT_REQUEST_FAILURES.clear()
        _TRANSIENT_REQUEST_FAILURES.extend(kept)
        return drained


def _http_error_body(exc: urllib.error.HTTPError) -> Optional[str]:
    """Extract and normalize text body from an HTTPError response."""
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
    """Reset per-run request and failure counters for metadata reporting."""
    _INGEST_STATS["total_requests"] = 0
    _INGEST_STATS["total_failures"] = 0
    _INGEST_STATS["per_provider_failures"] = dict.fromkeys(source_list, 0)
    _INGEST_STATS["per_status_failures"] = {}
    _INGEST_STATS["last_summary_request_count"] = 0
    _INGEST_STATS["last_summary_timestamp"] = time.time()
    _drain_transient_request_failures()


def _record_request_failure(provider: Optional[str], status_code: Optional[int] = None) -> None:
    """Increment global and per-provider failure counters."""
    code_key = str(status_code) if status_code is not None else "unknown"
    with _STATS_LOCK:
        _INGEST_STATS["total_failures"] += 1
        per_status = _INGEST_STATS["per_status_failures"]
        per_status[code_key] = int(per_status.get(code_key, 0)) + 1
        if provider:
            failures = _INGEST_STATS["per_provider_failures"]
            failures[provider] = int(failures.get(provider, 0)) + 1


def _emit_failure_summary_if_due(force: bool = False) -> None:
    """Emit periodic aggregate failure counts to keep long runs readable."""
    with _STATS_LOCK:
        total_failures = int(_INGEST_STATS["total_failures"])
        if total_failures <= 0:
            return

        total_requests = int(_INGEST_STATS["total_requests"])
        now = time.time()
        last_count = int(_INGEST_STATS["last_summary_request_count"])
        last_ts = float(_INGEST_STATS["last_summary_timestamp"])
        requests_delta = total_requests - last_count
        time_delta = now - last_ts

        if not force and requests_delta < _FAILURE_SUMMARY_REQUEST_INTERVAL:
            if time_delta < _FAILURE_SUMMARY_SECONDS_INTERVAL:
                return

        _INGEST_STATS["last_summary_request_count"] = total_requests
        _INGEST_STATS["last_summary_timestamp"] = now
        per_status = dict(_INGEST_STATS["per_status_failures"])

    status_tokens = [f"{code}={count}" for code, count in sorted(per_status.items())]
    status_summary = ", ".join(status_tokens) if status_tokens else "none"
    summary_msg = (
        "[ADIT] HTTP failures so far: "
        f"{total_failures}/{total_requests} request(s) failed ({status_summary})"
    )
    if force:
        _progress(summary_msg)
    else:
        _progress_inline(summary_msg)


def _compute_retry_sleep(last_error: Exception, delay: float) -> Tuple[float, str]:
    """Choose retry sleep duration and strategy label for logging."""
    if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
        retry_after = _retry_after_seconds(last_error)
        if retry_after is not None:
            sleep_time = min(retry_after, _SAFE_GET_RETRY_AFTER_MAX_SECONDS)
            strategy = (
                f"Retry-After (server requested {retry_after:.0f}s, capped to {sleep_time:.0f}s)"
            )
            return sleep_time, strategy

    jitter = random.uniform(0.8, 1.2)
    return min(delay * jitter, _SAFE_GET_MAX_DELAY), "exponential backoff"


def _safe_get(
    url: str,
    timeout: int = 20,
    provider: Optional[str] = None,
    max_retries: int = _SAFE_GET_MAX_RETRIES,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[dict]:
    """Fetch a URL, retrying on transient errors with exponential backoff + jitter.

    Retryable: network/timeout errors and HTTP 429/5xx.
    Permanent failures: HTTP 400/401/403/404 — these return None immediately.
    Exhausted retries: also return None, record to _INGEST_STATS, and queue
    provider-scoped transient retry metadata for checkpoint resume.
    """
    with _STATS_LOCK:
        _INGEST_STATS["total_requests"] += 1
    request_headers = {"User-Agent": "ADIT/0.1 ingestion"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, headers=request_headers)
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
                _record_request_failure(provider, status_code=exc.code)
                log_body = error_body if _DEBUG_HTTP else "<suppressed; enable --debug-http>"
                if exc.code == 404 and not _DEBUG_HTTP:
                    logger.debug(
                        "Permanent HTTP failure: provider=%s status=%s url=%s body=%s",
                        provider,
                        exc.code,
                        url,
                        log_body,
                    )
                else:
                    logger.warning(
                        "Permanent HTTP failure: provider=%s status=%s url=%s body=%s",
                        provider,
                        exc.code,
                        url,
                        log_body,
                    )
                _vprint(f"  [{provider or 'unknown'}] Permanent HTTP {exc.code} — skipping request")
                if _DEBUG_HTTP and error_body:
                    body_preview = re.sub(r"\s+", " ", error_body).strip()[:240]
                    _vprint(f"    body: {body_preview}")
                _emit_failure_summary_if_due()
                return None
            last_error = exc
        except Exception as exc:
            last_error = exc
            last_error_body = None

        if attempt < max_retries - 1:
            sleep_time, retry_strategy = _compute_retry_sleep(last_error, delay)

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

    _record_request_failure(provider, status_code=getattr(last_error, "code", None))
    _record_transient_request_failure(
        provider=provider,
        url=url,
        timeout=timeout,
        max_retries=max_retries,
        headers=request_headers,
        last_error=last_error,
        last_error_body=last_error_body,
    )
    log_body = last_error_body if _DEBUG_HTTP else "<suppressed; enable --debug-http>"
    logger.warning(
        "Request failed after %d attempts: provider=%s url=%s final_error=%s body=%s",
        max_retries,
        provider,
        url,
        last_error,
        log_body,
    )
    _vprint(f"  [{provider or 'unknown'}] All {max_retries} retries exhausted — skipping request")
    if _DEBUG_HTTP and last_error_body:
        body_preview = re.sub(r"\s+", " ", last_error_body).strip()[:240]
        _vprint(f"    body: {body_preview}")
    _emit_failure_summary_if_due()
    return None


def _norm_title(value: str) -> str:
    """Normalize title text for fuzzy dedup merge keys."""
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", cleaned)


def normalize_identifier(raw_id: str, source: Optional[str] = None) -> str:
    """Canonicalize raw IDs (DOI/OpenAlex/etc.) into stable prefixed forms."""
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
    """Build a provider-agnostic key used to merge duplicate papers."""
    if paper.doi:
        return normalize_identifier(paper.doi)
    title_key = _norm_title(paper.title)
    if title_key and paper.year:
        return f"titleyear:{title_key}:{paper.year}"
    return f"id:{paper.paper_id}"


def _cache_key(payload: dict) -> str:
    """Create deterministic hash key for cache/checkpoint files."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> Optional[dict]:
    """Read cached request result payload, returning None on miss/corruption."""
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    """Write request result payload to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Atomically write JSON by replacing target via temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _checkpoint_file(checkpoint_root: Path, key: str) -> Path:
    """Compute checkpoint path for a specific request key."""
    return checkpoint_root / f"{key}.checkpoint.json"


def _coordinator_checkpoint_file(checkpoint_root: Path, key: str) -> Path:
    """Compute path for the coordinator checkpoint payload."""
    return checkpoint_root / f"{key}.coordinator.checkpoint.json"


def _provider_checkpoint_file(checkpoint_root: Path, key: str, provider_name: str) -> Path:
    """Compute path for a provider-specific checkpoint payload."""
    safe_provider_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", provider_name)
    return checkpoint_root / f"{key}.{safe_provider_name}.provider.checkpoint.json"


def _is_pagination_state_stale(
    seed_state: Dict[str, Any],
    now_ts: Optional[float] = None,
    max_age_seconds: int = _PAGINATION_STATE_MAX_AGE_SECONDS,
) -> bool:
    """Return True when saved pagination state is older than staleness window."""
    if not isinstance(seed_state, dict):
        return True

    updated_at = seed_state.get("updated_at")
    if updated_at is None:
        return False

    try:
        updated_ts = float(updated_at)
    except (TypeError, ValueError):
        return True

    current = time.time() if now_ts is None else float(now_ts)
    return (current - updated_ts) > max_age_seconds


def _serialize_edges(edges: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Convert set-based edge map to JSON-friendly sorted lists."""
    return {citing: sorted(cited) for citing, cited in edges.items()}


def _deserialize_edges(raw_edges: Any) -> Dict[str, Set[str]]:
    """Normalize persisted edges into in-memory set representation."""
    if not isinstance(raw_edges, dict):
        return {}

    output: Dict[str, Set[str]] = {}
    for citing, cited in raw_edges.items():
        if not isinstance(citing, str):
            continue
        if isinstance(cited, list):
            output[citing] = {str(item) for item in cited if isinstance(item, str)}
        else:
            output[citing] = set()
    return output


def _serialize_papers(papers: Dict[str, IngestionPaper]) -> Dict[str, Dict[str, object]]:
    """Convert paper dataclasses into output-ready dictionaries."""
    return {paper_id: _paper_to_output_dict(paper) for paper_id, paper in papers.items()}


def _serialize_provider_pagination_state(
    state: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Prepare provider L2 pagination resume state for checkpoint persistence."""
    serialized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for provider_name, provider_state in state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_state, dict):
            continue

        provider_serialized: Dict[str, Dict[str, Any]] = {}
        for seed_id, seed_state in provider_state.items():
            if not isinstance(seed_id, str) or not isinstance(seed_state, dict):
                continue

            clean_state = dict(seed_state)
            papers = clean_state.get("papers")
            if isinstance(papers, dict):
                paper_objs = _deserialize_papers(papers)
                clean_state["papers"] = _serialize_papers(paper_objs)

            provider_serialized[seed_id] = clean_state

        serialized[provider_name] = provider_serialized

    return serialized


def _deserialize_papers(raw_papers: Any) -> Dict[str, IngestionPaper]:
    """Parse persisted paper dictionaries into IngestionPaper objects."""
    if not isinstance(raw_papers, dict):
        return {}

    output: Dict[str, IngestionPaper] = {}
    for paper_id, payload in raw_papers.items():
        if not isinstance(paper_id, str) or not isinstance(payload, dict):
            continue

        year_raw = payload.get("year")
        year: Optional[int] = None
        if year_raw is not None:
            try:
                year = int(year_raw)
            except (TypeError, ValueError):
                year = None

        citations_raw = payload.get("citations")
        try:
            citations = int(citations_raw or 0)
        except (TypeError, ValueError):
            citations = 0

        source_ids_raw = payload.get("source_ids")
        source_ids: Dict[str, str] = {}
        if isinstance(source_ids_raw, dict):
            source_ids = {
                str(k): str(v)
                for k, v in source_ids_raw.items()
                if isinstance(k, str) and isinstance(v, str)
            }

        doi_raw = payload.get("doi")
        doi = str(doi_raw) if isinstance(doi_raw, str) and doi_raw else None

        output[paper_id] = IngestionPaper(
            paper_id=paper_id,
            title=str(payload.get("title") or ""),
            abstract=str(payload.get("abstract") or ""),
            keywords=str(payload.get("keywords") or ""),
            citations=citations,
            year=year,
            doi=doi,
            source_ids=source_ids,
        )

    return output


def _deserialize_provider_pagination_state(
    raw_state: Any,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load and sanitize provider L2 pagination state from checkpoint JSON."""
    if not isinstance(raw_state, dict):
        return {}

    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for provider_name, provider_state in raw_state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_state, dict):
            continue

        provider_output: Dict[str, Dict[str, Any]] = {}
        for seed_id, seed_state in provider_state.items():
            if not isinstance(seed_id, str) or not isinstance(seed_state, dict):
                continue

            clean_state = dict(seed_state)
            papers = clean_state.get("papers")
            if isinstance(papers, dict):
                paper_objs = _deserialize_papers(papers)
                clean_state["papers"] = _serialize_papers(paper_objs)

            provider_output[seed_id] = clean_state

        output[provider_name] = provider_output

    return output


def _serialize_provider_l3_state(
    state: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Prepare provider L3 resume state (edges/papers/index/budget) for JSON."""
    serialized: Dict[str, Dict[str, Any]] = {}
    for provider_name, provider_state in state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_state, dict):
            continue

        clean_state = dict(provider_state)
        edges_raw = clean_state.get("edges")
        if isinstance(edges_raw, dict):
            clean_state["edges"] = _serialize_edges(_deserialize_edges(edges_raw))

        papers_raw = clean_state.get("papers")
        if isinstance(papers_raw, dict):
            clean_state["papers"] = _serialize_papers(_deserialize_papers(papers_raw))

        serialized[provider_name] = clean_state

    return serialized


def _deserialize_provider_l3_state(raw_state: Any) -> Dict[str, Dict[str, Any]]:
    """Load and sanitize provider L3 resume state from checkpoint JSON."""
    if not isinstance(raw_state, dict):
        return {}

    output: Dict[str, Dict[str, Any]] = {}
    for provider_name, provider_state in raw_state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_state, dict):
            continue

        clean_state = dict(provider_state)
        edges_raw = clean_state.get("edges")
        if isinstance(edges_raw, dict):
            clean_state["edges"] = _serialize_edges(_deserialize_edges(edges_raw))

        papers_raw = clean_state.get("papers")
        if isinstance(papers_raw, dict):
            clean_state["papers"] = _serialize_papers(_deserialize_papers(papers_raw))

        output[provider_name] = clean_state

    return output


def _serialize_transient_failures(
    state: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Prepare provider transient retry records for checkpoint persistence."""
    serialized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for provider_name, provider_failures in state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_failures, dict):
            continue

        provider_out: Dict[str, Dict[str, Any]] = {}
        for failure_key, failure in provider_failures.items():
            if not isinstance(failure_key, str) or not isinstance(failure, dict):
                continue

            clean = dict(failure)
            resume_state = clean.get("resume_state")
            if not isinstance(resume_state, dict):
                clean["resume_state"] = {}
            else:
                headers = _sanitize_retry_headers(resume_state.get("headers"))
                clean["resume_state"] = {
                    "timeout": _parse_optional_int(resume_state.get("timeout"), default=20) or 20,
                    "max_retries": _parse_optional_int(
                        resume_state.get("max_retries"),
                        default=_SAFE_GET_MAX_RETRIES,
                    )
                    or _SAFE_GET_MAX_RETRIES,
                    "headers": headers,
                }

            clean["attempts"] = _parse_optional_int(clean.get("attempts"), default=0) or 0
            updated_ts = clean.get("last_attempt_ts")
            try:
                clean["last_attempt_ts"] = float(updated_ts) if updated_ts is not None else 0.0
            except (TypeError, ValueError):
                clean["last_attempt_ts"] = 0.0

            retry_after = clean.get("server_retry_after")
            try:
                clean["server_retry_after"] = (
                    float(retry_after) if retry_after is not None else None
                )
            except (TypeError, ValueError):
                clean["server_retry_after"] = None

            provider_out[failure_key] = clean

        if provider_out:
            serialized[provider_name] = provider_out

    return serialized


def _deserialize_transient_failures(raw_state: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load and sanitize transient retry records from checkpoint payloads."""
    if not isinstance(raw_state, dict):
        return {}

    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for provider_name, provider_failures in raw_state.items():
        if not isinstance(provider_name, str) or not isinstance(provider_failures, dict):
            continue

        provider_out: Dict[str, Dict[str, Any]] = {}
        for failure_key, failure in provider_failures.items():
            if not isinstance(failure_key, str) or not isinstance(failure, dict):
                continue

            clean = dict(failure)
            clean.setdefault("provider", provider_name)
            clean.setdefault("op", "safe_get")
            clean.setdefault("target_id", "")
            clean["attempts"] = _parse_optional_int(clean.get("attempts"), default=0) or 0

            updated_ts = clean.get("last_attempt_ts")
            try:
                clean["last_attempt_ts"] = float(updated_ts) if updated_ts is not None else 0.0
            except (TypeError, ValueError):
                clean["last_attempt_ts"] = 0.0

            retry_after = clean.get("server_retry_after")
            try:
                clean["server_retry_after"] = (
                    float(retry_after) if retry_after is not None else None
                )
            except (TypeError, ValueError):
                clean["server_retry_after"] = None

            resume_state = clean.get("resume_state")
            if not isinstance(resume_state, dict):
                clean["resume_state"] = {}
            else:
                clean["resume_state"] = {
                    "timeout": _parse_optional_int(resume_state.get("timeout"), default=20) or 20,
                    "max_retries": _parse_optional_int(
                        resume_state.get("max_retries"),
                        default=_SAFE_GET_MAX_RETRIES,
                    )
                    or _SAFE_GET_MAX_RETRIES,
                    "headers": _sanitize_retry_headers(resume_state.get("headers")),
                }

            provider_out[failure_key] = clean

        if provider_out:
            output[provider_name] = provider_out

    return output


def _merge_provider_transient_failures(
    existing: Dict[str, Dict[str, Any]],
    incoming: Sequence[Dict[str, Any]],
) -> None:
    """Merge newly observed transient failures into provider-local retry state."""
    for payload in incoming:
        if not isinstance(payload, dict):
            continue
        failure = dict(payload)
        failure.setdefault("op", "safe_get")
        failure.setdefault("provider", "unknown")
        failure.setdefault("target_id", "")
        failure["attempts"] = _parse_optional_int(failure.get("attempts"), default=0) or 0
        failure.setdefault("last_attempt_ts", time.time())
        key = _transient_failure_key(failure)
        previous = existing.get(key)
        if isinstance(previous, dict):
            previous_attempts = _parse_optional_int(previous.get("attempts"), default=0) or 0
            failure["attempts"] = max(previous_attempts + 1, failure["attempts"])
        existing[key] = failure


def _transient_retry_wait_seconds(record: Dict[str, Any], now_ts: Optional[float] = None) -> float:
    """Return remaining wait time before a transient record can be retried."""
    now = time.time() if now_ts is None else float(now_ts)
    attempts = _parse_optional_int(record.get("attempts"), default=0) or 0
    last_attempt = record.get("last_attempt_ts")
    try:
        last_ts = float(last_attempt) if last_attempt is not None else 0.0
    except (TypeError, ValueError):
        last_ts = 0.0

    backoff_delay = min(
        _SAFE_GET_INITIAL_DELAY * (_SAFE_GET_BACKOFF_FACTOR ** max(attempts, 0)),
        _SAFE_GET_MAX_DELAY,
    )
    retry_after = record.get("server_retry_after")
    try:
        retry_after_delay = float(retry_after) if retry_after is not None else 0.0
    except (TypeError, ValueError):
        retry_after_delay = 0.0

    wait_until = max(last_ts + backoff_delay, last_ts + retry_after_delay)
    return max(0.0, wait_until - now)


def _prune_provider_transient_failures(
    provider_name: str,
    failures: Dict[str, Dict[str, Any]],
    checkpoint_stats: Optional[Dict[str, object]] = None,
) -> None:
    """Drop transient retry records that are too old or exhausted."""
    now = time.time()
    for failure_key, record in list(failures.items()):
        attempts = _parse_optional_int(record.get("attempts"), default=0) or 0
        last_attempt = record.get("last_attempt_ts")
        try:
            last_ts = float(last_attempt) if last_attempt is not None else 0.0
        except (TypeError, ValueError):
            last_ts = 0.0
        too_old = last_ts > 0.0 and (now - last_ts) > _TRANSIENT_RETRY_MAX_AGE_SECONDS
        exhausted = attempts >= _TRANSIENT_RETRY_MAX_ATTEMPTS
        if not too_old and not exhausted:
            continue
        failures.pop(failure_key, None)
        if checkpoint_stats is None:
            continue
        checkpoint_stats["transient_failures_pruned"] = (
            int(checkpoint_stats.get("transient_failures_pruned", 0)) + 1
        )
        _vprint(f"  [{provider_name}] Pruned transient retry record: {failure_key}")


def _transient_failure_summary(
    transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """Build compact coordinator-friendly summary for transient retry queues."""
    summary: Dict[str, Dict[str, int]] = {}
    for provider_name, provider_records in transient_failures.items():
        if not isinstance(provider_name, str) or not isinstance(provider_records, dict):
            continue
        queued = len(provider_records)
        max_attempts = 0
        for record in provider_records.values():
            attempts = _parse_optional_int((record or {}).get("attempts"), default=0) or 0
            max_attempts = max(max_attempts, attempts)
        summary[provider_name] = {"queued": queued, "max_attempts": max_attempts}
    return summary


def _load_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    reset_checkpoints: bool,
) -> Optional[Dict[str, Any]]:
    """Load validated checkpoint payload for request key, if available."""
    checkpoint_path = _checkpoint_file(checkpoint_root, key)
    if reset_checkpoints and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            return None

    if not checkpoint_path.exists():
        return None

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("checkpoint_schema_version") not in {3, _CHECKPOINT_SCHEMA_VERSION}:
        return None
    if payload.get("request_key") != key:
        return None
    return payload


def _load_coordinator_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    reset_checkpoints: bool,
) -> Optional[Dict[str, Any]]:
    """Load validated coordinator checkpoint payload for request key, if available."""
    checkpoint_path = _coordinator_checkpoint_file(checkpoint_root, key)
    if reset_checkpoints and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            return None

    if not checkpoint_path.exists():
        return None

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("coordinator_checkpoint_schema_version") not in {
        1,
        _COORDINATOR_CHECKPOINT_SCHEMA_VERSION,
    }:
        return None
    if payload.get("request_key") != key:
        return None
    return payload


def _write_coordinator_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    completed_providers: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, object],
    combined_completeness: Dict[str, Dict[str, object]],
    transient_failure_summary: Optional[Dict[str, Dict[str, int]]] = None,
    l3_to_l3_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ingestion_phase: str = "l2_to_l3",
) -> None:
    """Persist coordinator checkpoint snapshot used for crash-safe resume."""
    checkpoint_path = _coordinator_checkpoint_file(checkpoint_root, key)
    payload = {
        "coordinator_checkpoint_schema_version": _COORDINATOR_CHECKPOINT_SCHEMA_VERSION,
        "request_key": key,
        "completed_providers": sorted(completed_providers),
        "all_edges": _serialize_edges(all_edges),
        "all_papers": _serialize_papers(all_papers),
        "provider_stats": provider_stats,
        "combined_completeness": combined_completeness,
        "transient_failure_summary": transient_failure_summary or {},
        "l3_to_l3_state": _serialize_provider_l3_state(l3_to_l3_state or {}),
        "ingestion_phase": ingestion_phase,
    }
    _write_json_atomic(checkpoint_path, payload)


def _load_provider_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    provider_name: str,
    reset_checkpoints: bool,
) -> Optional[Dict[str, Any]]:
    """Load validated provider-specific checkpoint payload, if available."""
    checkpoint_path = _provider_checkpoint_file(checkpoint_root, key, provider_name)
    if reset_checkpoints and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            return None

    if not checkpoint_path.exists():
        return None

    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("provider_checkpoint_schema_version") not in {
        1,
        _PROVIDER_CHECKPOINT_SCHEMA_VERSION,
    }:
        return None
    if payload.get("request_key") != key:
        return None
    if payload.get("provider_name") != provider_name:
        return None
    return payload


def _write_provider_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    provider_name: str,
    provider_pagination_state: Optional[Dict[str, Dict[str, Any]]] = None,
    provider_l3_state: Optional[Dict[str, Any]] = None,
    transient_failures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Persist provider-specific checkpoint state for in-flight resume."""
    checkpoint_path = _provider_checkpoint_file(checkpoint_root, key, provider_name)
    payload = {
        "provider_checkpoint_schema_version": _PROVIDER_CHECKPOINT_SCHEMA_VERSION,
        "request_key": key,
        "provider_name": provider_name,
        "provider_pagination_state": _serialize_provider_pagination_state(
            {provider_name: provider_pagination_state or {}}
        ).get(provider_name, {}),
        "provider_l3_state": _serialize_provider_l3_state(
            {provider_name: provider_l3_state or {}}
        ).get(provider_name, {}),
        "transient_failures": _serialize_transient_failures(
            {provider_name: transient_failures or {}}
        ).get(provider_name, {}),
    }
    _write_json_atomic(checkpoint_path, payload)


def _write_checkpoint_state(
    checkpoint_root: Path,
    key: str,
    completed_providers: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, object],
    combined_completeness: Dict[str, Dict[str, object]],
    provider_pagination_state: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    provider_l3_state: Optional[Dict[str, Dict[str, Any]]] = None,
    provider_transient_failures: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    l3_to_l3_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ingestion_phase: str = "l2_to_l3",
) -> None:
    """Persist the current ingestion snapshot used for crash-safe resume."""
    checkpoint_path = _checkpoint_file(checkpoint_root, key)
    payload = {
        "checkpoint_schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "request_key": key,
        "completed_providers": sorted(completed_providers),
        "all_edges": _serialize_edges(all_edges),
        "all_papers": _serialize_papers(all_papers),
        "provider_stats": provider_stats,
        "combined_completeness": combined_completeness,
        "provider_pagination_state": _serialize_provider_pagination_state(
            provider_pagination_state or {}
        ),
        "provider_l3_state": _serialize_provider_l3_state(provider_l3_state or {}),
        "provider_transient_failures": _serialize_transient_failures(
            provider_transient_failures or {}
        ),
        "transient_failure_summary": _transient_failure_summary(provider_transient_failures or {}),
        "l3_to_l3_state": _serialize_provider_l3_state(l3_to_l3_state or {}),
        "ingestion_phase": ingestion_phase,
    }
    _write_json_atomic(checkpoint_path, payload)


def _paper_to_output_dict(paper: IngestionPaper) -> Dict[str, object]:
    """Convert internal paper model into output/cache JSON shape."""
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
    """Merge two paper records, preferring richer fields from incoming.

    ``existing.paper_id`` is always preserved as the identity of the merged
    result.  Callers that intentionally merge records with different IDs (e.g.
    ``_dedupe_and_materialize``) rely on this guarantee.

    Returns a new ``IngestionPaper`` without mutating either argument.

    Uncertain merges (cases where non-empty existing data is overwritten or
    silently discarded) are logged at DEBUG level so that callers can surface
    them when diagnosing unexpected metadata.
    """
    pid = existing.paper_id

    # ------------------------------------------------------------------
    # Title: prefer the longer non-empty string (more complete text).
    # Log when we discard a non-empty existing title in favour of incoming.
    # ------------------------------------------------------------------
    if incoming.title and len(incoming.title) > len(existing.title):
        if existing.title:
            logger.debug(
                "merge[%s] title overwritten: %r -> %r",
                pid,
                existing.title,
                incoming.title,
            )
        best_title = incoming.title
    else:
        best_title = existing.title

    # ------------------------------------------------------------------
    # Abstract: same length-preference logic as title.
    # ------------------------------------------------------------------
    if incoming.abstract and len(incoming.abstract) > len(existing.abstract):
        if existing.abstract:
            logger.debug(
                "merge[%s] abstract overwritten (%d chars -> %d chars)",
                pid,
                len(existing.abstract),
                len(incoming.abstract),
            )
        best_abstract = incoming.abstract
    else:
        best_abstract = existing.abstract

    # ------------------------------------------------------------------
    # Keywords: union-merge comma-separated entries so neither source loses data.
    # ------------------------------------------------------------------
    if incoming.keywords and existing.keywords:
        existing_kw = {k.strip() for k in existing.keywords.split(",") if k.strip()}
        incoming_kw = {k.strip() for k in incoming.keywords.split(",") if k.strip()}
        merged_keywords = ", ".join(sorted(existing_kw | incoming_kw))
    else:
        merged_keywords = incoming.keywords or existing.keywords

    # ------------------------------------------------------------------
    # Citations: take the higher count.  Both values may be stale snapshots
    # from different fetch times; without timestamps we cannot determine which
    # is fresher.  Log whenever two non-zero counts diverge so the caller can
    # see which value was discarded.
    # ------------------------------------------------------------------
    best_citations = max(existing.citations, incoming.citations)
    if (
        existing.citations > 0
        and incoming.citations > 0
        and existing.citations != incoming.citations
    ):
        discarded = min(existing.citations, incoming.citations)
        logger.debug(
            "merge[%s] citation count conflict: keeping %d, discarding %d",
            pid,
            best_citations,
            discarded,
        )

    # ------------------------------------------------------------------
    # Year: accept incoming only when we have no year yet.  Log when both
    # records carry a year but disagree — incoming's value is silently
    # dropped and the caller may want to investigate.
    # ------------------------------------------------------------------
    best_year = existing.year or incoming.year
    if existing.year is not None and incoming.year is not None and existing.year != incoming.year:
        logger.debug(
            "merge[%s] year conflict: keeping existing %d, discarding incoming %d",
            pid,
            existing.year,
            incoming.year,
        )

    return IngestionPaper(
        paper_id=existing.paper_id,
        title=best_title,
        abstract=best_abstract,
        keywords=merged_keywords,
        citations=best_citations,
        year=best_year,
        doi=existing.doi or incoming.doi,
        source_ids={**existing.source_ids, **incoming.source_ids},
    )


def _query_terms(theory_name: str, key_constructs: Optional[Sequence[str]]) -> str:
    """Build provider search query text from theory and constructs."""
    terms = [theory_name.strip()]
    if key_constructs:
        terms.extend([k.strip() for k in key_constructs if k.strip()])
    return " ".join([t for t in terms if t])


def _core_auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Build CORE API auth headers if key is configured."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _semantic_scholar_auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    """Build Semantic Scholar API auth headers if key is configured."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _doi_from_identifier(identifier: str) -> Optional[str]:
    """Extract bare DOI from a normalized identifier when possible."""
    if not identifier:
        return None
    if identifier.startswith("doi:"):
        return identifier.split(":", 1)[1]
    return None


def _paper_from_semantic_reference(ref: dict) -> Optional[IngestionPaper]:
    """Create a minimal paper model from a Semantic Scholar reference object."""
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
    """Rebuild plain-text abstract from OpenAlex inverted index payload."""
    if not inv_idx:
        return ""

    reconstructed = []
    for token, positions in inv_idx.items():
        for pos in positions:
            reconstructed.append((int(pos), token))
    return " ".join(token for _, token in sorted(reconstructed, key=lambda x: x[0]))


def _openalex_linked_l1(item: dict, l1_norm: Set[str], theory_name: str) -> List[str]:
    """Return normalized L1 papers explicitly cited by an OpenAlex result item."""
    refs = item.get("referenced_works", []) or []
    ref_norm = {normalize_identifier(ref, source="openalex") for ref in refs if ref}
    linked_l1 = sorted(ref_norm.intersection(l1_norm))
    return linked_l1


def _should_keep_openalex_item(item: dict, linked_l1: Sequence[str], theory_name: str) -> bool:
    """Gate OpenAlex candidates to only those with explicit links to seed papers."""
    return bool(linked_l1)


def _paper_from_openalex_item(item: dict, paper_id: str) -> IngestionPaper:
    """Map an OpenAlex work payload to the shared paper model."""
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


def _openalex_reference_stub(ref_id: str, ref: str) -> IngestionPaper:
    """Create lightweight placeholder metadata for OpenAlex references."""
    token = ref_id.split(":", 1)[1] if ref_id.startswith("openalex:") else ""
    source_id = f"https://openalex.org/{token}" if token else str(ref)
    return IngestionPaper(
        paper_id=ref_id,
        doi=_doi_from_identifier(ref_id),
        source_ids={"openalex": source_id},
    )


def _openalex_hydrated_paper(ref_id: str, payload: dict) -> IngestionPaper:
    """Create enriched OpenAlex paper identity fields from hydration payload."""
    year_raw = payload.get("publication_year")
    token = ref_id.split(":", 1)[1]
    return IngestionPaper(
        paper_id=ref_id,
        title=payload.get("title") or "",
        year=int(year_raw) if year_raw else None,
        doi=payload.get("doi"),
        source_ids={"openalex": str(payload.get("id") or f"https://openalex.org/{token}")},
    )


def _semantic_fetch_status(
    papers: Dict[str, IngestionPaper],
    expected_count: Optional[int],
) -> Tuple[int, str]:
    """Compute fetch status label from observed and expected counts."""
    fetched = len(papers)
    status = "complete" if expected_count is not None and fetched >= expected_count else "partial"
    return fetched, status


def _semantic_batch_limit(
    offset: int,
    limit: int,
    papers: Dict[str, IngestionPaper],
    max_results: Optional[int],
) -> Optional[int]:
    """Compute next S2 page size while honoring API and caller limits."""
    batch_limit = limit
    if max_results is not None:
        remaining = max_results - len(papers)
        if remaining <= 0:
            return None
        batch_limit = min(limit, remaining)

    max_allowed = 9999 - offset
    if max_allowed <= 0:
        return None

    batch_limit = min(batch_limit, max_allowed)
    return batch_limit if batch_limit > 0 else None


def _restore_traversal_state(
    resume_state: Optional[Dict[str, Any]],
    index_key: str,
    max_edges: Optional[int],
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], int, Optional[int]]:
    """Restore edge traversal state from a checkpoint snapshot."""
    state = resume_state or {}
    edges = _deserialize_edges(state.get("edges"))
    papers = _deserialize_papers(state.get("papers"))

    start_index_raw = state.get(index_key)
    try:
        start_index = int(start_index_raw) if start_index_raw is not None else 0
    except (TypeError, ValueError):
        start_index = 0

    if max_edges is None:
        budget = None
    elif state.get("budget_remaining") is not None:
        try:
            budget = max(0, int(state.get("budget_remaining")))
        except (TypeError, ValueError):
            budget = max(0, max_edges)
    else:
        budget = max(0, max_edges)

    return edges, papers, start_index, budget


def _parse_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely parse an optional integer value with fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _emit_traversal_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    status: str,
    index_key: str,
    index_value: int,
    budget_remaining: Optional[int],
    edges: Dict[str, Set[str]],
    papers: Dict[str, IngestionPaper],
) -> None:
    """Emit a serialized traversal checkpoint update when requested."""
    if not progress_callback:
        return
    progress_callback(
        {
            "status": status,
            index_key: index_value,
            "budget_remaining": budget_remaining,
            "edges": _serialize_edges(edges),
            "papers": _serialize_papers(papers),
            "updated_at": time.time(),
        }
    )


def _emit_citers_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    status: str,
    position_key: str,
    position_value: Any,
    expected_count: Optional[int],
    papers: Dict[str, IngestionPaper],
    fetched_count: Optional[int] = None,
) -> None:
    """Emit citer traversal checkpoint updates when requested."""
    if not progress_callback:
        return
    progress_callback(
        {
            "status": status,
            position_key: position_value,
            "expected_count": expected_count,
            "fetched_count": len(papers) if fetched_count is None else fetched_count,
            "papers": _serialize_papers(papers),
            "updated_at": time.time(),
        }
    )


def _semantic_linked_l1(item: dict, l1_norm: Set[str]) -> Set[str]:
    """Find seed papers cited by a Semantic Scholar candidate item."""
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
    """Gate Semantic Scholar candidates to explicit L1-citation matches."""
    return bool(linked_l1)


def _crossref_enrichment_targets(
    l2_paper_ids: Sequence[str],
    all_papers: Dict[str, IngestionPaper],
) -> List[str]:
    """Select DOI-based papers eligible for Crossref metadata enrichment."""
    targets: Set[str] = set()
    for paper_id in l2_paper_ids:
        paper = all_papers.get(paper_id)
        doi = paper.doi if paper else _doi_from_identifier(paper_id)
        if doi:
            targets.add(normalize_identifier(doi))
    return sorted(targets)


def _paper_from_semantic_item(item: dict, paper_id: str) -> IngestionPaper:
    """Map a Semantic Scholar paper payload to the shared paper model."""
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


def _core_reference_candidates(ref: dict) -> List[str]:
    """Extract normalized candidate identifiers from a CORE reference entry."""
    candidates: List[str] = []
    doi = ref.get("doi")
    if doi:
        candidates.append(normalize_identifier(str(doi)))

    ref_id = ref.get("id")
    if ref_id is not None:
        candidates.append(normalize_identifier(str(ref_id), source="core"))
    return [candidate for candidate in candidates if candidate]


def _paper_from_core_item(item: dict, paper_id: str) -> IngestionPaper:
    """Map a CORE work payload to the shared paper model."""
    doi = item.get("doi")
    core_id = item.get("id")
    year_raw = item.get("yearPublished")
    if year_raw is None:
        year_raw = item.get("year_published")

    citations_raw = item.get("citationCount")
    if citations_raw is None:
        citations_raw = item.get("citation_count")

    subjects = item.get("subjects") or []
    if isinstance(subjects, list):
        keywords = ", ".join(str(subject) for subject in subjects if subject)
    else:
        keywords = str(subjects or "")

    source_ids: Dict[str, str] = {}
    if core_id is not None:
        source_ids["core"] = str(core_id)

    return IngestionPaper(
        paper_id=paper_id,
        title=str(item.get("title") or ""),
        abstract=str(item.get("abstract") or ""),
        keywords=keywords,
        citations=int(citations_raw or 0),
        year=int(year_raw) if year_raw else None,
        doi=str(doi) if doi else None,
        source_ids=source_ids,
    )


def _seed_l1_papers(l1_papers: Sequence[str]) -> tuple[List[str], Dict[str, IngestionPaper]]:
    """Normalize seeds and create placeholder records merged later with provider metadata."""
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
    """Merge provider edge/paper outputs into cumulative ingestion state."""
    for citing, cited in edges.items():
        all_edges.setdefault(citing, set()).update(cited)
    for pid, paper in papers.items():
        if pid in all_papers:
            all_papers[pid] = _merge_papers(all_papers[pid], paper)
        else:
            all_papers[pid] = paper


def _record_seed_without_provider_id(
    provider_name: str,
    seed_index: int,
    total_seeds: int,
    l1_id: str,
    completeness: Dict[str, Dict[str, object]],
) -> None:
    """Record skipped seed state when provider-native ID is unavailable."""
    _progress(
        f"  [{provider_name}] Seed {seed_index}/{total_seeds}: "
        f"skipped {l1_id} (no provider ID resolved)"
    )
    completeness[l1_id] = {
        provider_name: {
            "status": "skipped",
            "reason": "no_provider_id",
            "fetched": 0,
            "expected": 0,
        }
    }


def _build_seed_progress_callback(
    provider_name: str,
    seed_id: str,
    seed_index: int,
    total_seeds: int,
    seed_progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Build per-seed progress callback for cited-by traversal."""
    if seed_progress_callback is None and _QUIET:
        return None

    last_progress_fetched: Optional[int] = None

    def _on_progress(state: Dict[str, Any]) -> None:
        nonlocal last_progress_fetched
        if seed_progress_callback is not None:
            seed_progress_callback(seed_id, state)
        if _QUIET or str(state.get("status") or "") != "in_progress":
            return

        fetched_raw = state.get("fetched_count")
        if not isinstance(fetched_raw, int) or fetched_raw == last_progress_fetched:
            return

        expected_raw = state.get("expected_count")
        expected_display = (
            str(expected_raw) if isinstance(expected_raw, int) and expected_raw > 0 else "?"
        )
        _progress_inline(
            f"  [{provider_name}] Seed {seed_index}/{total_seeds}: "
            f"citers {fetched_raw}/{expected_display}"
        )
        last_progress_fetched = fetched_raw

    return _on_progress


def _fetch_l2_via_cited_by_traversal(
    provider: CitationProvider,
    l1_norm: Sequence[str],
    l1_papers_resolved: Dict[str, IngestionPaper],
    max_results: Optional[int],
    provider_seed_state: Optional[Dict[str, Dict[str, Any]]],
    seed_progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], Dict[str, Dict[str, object]]]:
    """Collect provider-local L2 edges/papers by traversing citers for each seed."""
    l2_edges: Dict[str, Set[str]] = {}
    l2_papers: Dict[str, IngestionPaper] = {}
    completeness: Dict[str, Dict[str, object]] = {}
    total_seeds = len(l1_norm)

    for seed_index, l1_id in enumerate(l1_norm, start=1):
        seed = l1_papers_resolved.get(l1_id)
        raw_provider_id = seed.source_ids.get(provider.name) if seed else None
        if not raw_provider_id:
            _record_seed_without_provider_id(
                provider_name=provider.name,
                seed_index=seed_index,
                total_seeds=total_seeds,
                l1_id=l1_id,
                completeness=completeness,
            )
            if provider_progress_callback is not None:
                provider_progress_callback(seed_index, total_seeds, "skipped")
            continue

        _progress(
            f"  [{provider.name}] Seed {seed_index}/{total_seeds}: fetching citers for {l1_id}"
        )
        l1_native_id = normalize_identifier(raw_provider_id, source=provider.name)
        resume_state = None
        if provider_seed_state and isinstance(provider_seed_state.get(l1_id), dict):
            resume_state = provider_seed_state[l1_id]

        call_kwargs: Dict[str, Any] = {"max_results": max_results}
        signature = inspect.signature(provider.fetch_citers_for_l1)
        if "resume_state" in signature.parameters:
            call_kwargs["resume_state"] = resume_state
        progress_fn = _build_seed_progress_callback(
            provider_name=provider.name,
            seed_id=l1_id,
            seed_index=seed_index,
            total_seeds=total_seeds,
            seed_progress_callback=seed_progress_callback,
        )
        if "progress_callback" in signature.parameters and progress_fn is not None:
            call_kwargs["progress_callback"] = progress_fn

        papers_for_l1, expected_count, status = provider.fetch_citers_for_l1(
            l1_native_id, **call_kwargs
        )
        _progress_done(
            f"  [{provider.name}] Seed {seed_index}/{total_seeds}: "
            f"fetched {len(papers_for_l1)}/{expected_count or '?'} papers ({status})"
        )
        if provider_progress_callback is not None:
            provider_progress_callback(seed_index, total_seeds, status)
        completeness.setdefault(l1_id, {})[provider.name] = {
            "status": status,
            "fetched": len(papers_for_l1),
            "expected": expected_count,
        }
        for paper_id, paper in papers_for_l1.items():
            l2_edges.setdefault(paper_id, set()).add(l1_id)
            l2_papers[paper_id] = (
                _merge_papers(l2_papers[paper_id], paper) if paper_id in l2_papers else paper
            )

    return l2_edges, l2_papers, completeness


def _build_l3_progress_callback(
    provider_name: str,
    l2_parent_ids: Sequence[str],
    l3_progress_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> Optional[Callable[[Dict[str, Any]], None]]:
    """Build progress callback used during L2->L3 hydration."""
    if l3_progress_callback is None and _QUIET:
        return None

    total_l2_parents = len(l2_parent_ids)
    last_l3_parent_index: Optional[int] = None

    def _on_l3_progress(state: Dict[str, Any]) -> None:
        nonlocal last_l3_parent_index
        if l3_progress_callback is not None:
            l3_progress_callback(state)
        if _QUIET or str(state.get("status") or "") != "in_progress" or total_l2_parents <= 0:
            return

        parent_index_raw = state.get("next_l2_index")
        if not isinstance(parent_index_raw, int):
            return
        parent_index = min(max(parent_index_raw, 1), total_l2_parents)
        if parent_index == last_l3_parent_index:
            return

        refs_display = "?"
        edges_raw = state.get("edges")
        if isinstance(edges_raw, dict):
            parent_targets = edges_raw.get(l2_parent_ids[parent_index - 1])
            if isinstance(parent_targets, list):
                refs_display = str(len(parent_targets))

        _progress_inline(
            f"  [{provider_name}] L2 node {parent_index}/{total_l2_parents}: "
            f"references {refs_display}"
        )
        last_l3_parent_index = parent_index

    return _on_l3_progress


def _fetch_l3_for_provider(
    provider: CitationProvider,
    l2_parent_ids: Sequence[str],
    max_l3: Optional[int],
    provider_l3_state: Optional[Dict[str, Any]],
    l3_progress_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], int]:
    """Fetch L3 reference edges and papers for a provider."""
    l3_call_kwargs: Dict[str, Any] = {"l2_paper_ids": list(l2_parent_ids), "max_l3": max_l3}
    signature = inspect.signature(provider.fetch_l3_references)
    if "resume_state" in signature.parameters:
        l3_call_kwargs["resume_state"] = provider_l3_state

    progress_fn = _build_l3_progress_callback(provider.name, l2_parent_ids, l3_progress_callback)
    if "progress_callback" in signature.parameters and progress_fn is not None:
        l3_call_kwargs["progress_callback"] = progress_fn

    l3_edges, l3_papers = provider.fetch_l3_references(**l3_call_kwargs)
    return l3_edges, l3_papers, sum(len(cited) for cited in l3_edges.values())


def _fetch_provider_graph(
    provider: CitationProvider,
    l1_norm: Sequence[str],
    l1_papers_resolved: Dict[str, IngestionPaper],
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    exhaustive: bool,
    max_l2: int,
    max_l3: Optional[int],
    provider_seed_state: Optional[Dict[str, Dict[str, Any]]] = None,
    seed_progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
    provider_l3_state: Optional[Dict[str, Any]] = None,
    l3_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    include_l3: bool = True,
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], Dict[str, object]]:
    """Fetch L2 (and optionally L3) papers for a provider.

    When the provider supports cited-by traversal, each L1 seed is queried
    individually using its resolved provider-native ID so all citers are
    returned (no title/keyword gate). Providers without cited-by traversal
    can still use keyword-based candidate retrieval, but candidates are only
    admitted when an explicit L1 citation link is present.
    """
    max_results: Optional[int] = None if exhaustive else max_l2

    if provider.capabilities.supports_cited_by_traversal:
        l2_edges, l2_papers, completeness = _fetch_l2_via_cited_by_traversal(
            provider=provider,
            l1_norm=l1_norm,
            l1_papers_resolved=l1_papers_resolved,
            max_results=max_results,
            provider_seed_state=provider_seed_state,
            seed_progress_callback=seed_progress_callback,
            provider_progress_callback=provider_progress_callback,
        )
    else:
        # Keyword-search fallback: used for Crossref and any provider lacking cited-by support.
        search_edges, search_papers = provider.fetch_l2_and_metadata(
            l1_papers=l1_norm,
            theory_name=theory_name,
            key_constructs=key_constructs,
            max_l2=max_l2,
        )
        l2_edges, l2_papers = dict(search_edges), dict(search_papers)
        completeness = {}
        if provider_progress_callback is not None:
            provider_progress_callback(1, 1, "complete")

    _progress(
        f"  [{provider.name}] Total L2 nodes collected across all seeds (provider-local): {len(l2_edges)}"
    )

    all_edges = dict(l2_edges)
    all_papers = dict(l2_papers)
    added_l3_edges = 0

    if include_l3 and depth.lower() in {"l2l3", "l3", "2"}:
        l3_edges, l3_papers, added_l3_edges = _fetch_l3_for_provider(
            provider=provider,
            l2_parent_ids=list(l2_edges.keys()),
            max_l3=max_l3,
            provider_l3_state=provider_l3_state,
            l3_progress_callback=l3_progress_callback,
        )
        _merge_provider_outputs(all_edges, all_papers, l3_edges, l3_papers)
        _progress_done(f"  [{provider.name}] L3 hydration complete: {added_l3_edges} edges")

    stats: Dict[str, object] = {
        "l2_nodes": len(l2_edges),
        "l2_edges": sum(len(v) for v in l2_edges.values()),
        "l3_edges": added_l3_edges,
        "papers": len(l2_papers),
        "completeness": completeness,
    }
    return all_edges, all_papers, stats


class ProviderIngestionError(Exception):
    """Raised when a provider worker fails during parallel ingestion."""

    def __init__(self, provider_name: str, cause: BaseException):
        super().__init__(f"provider={provider_name} failed: {cause}")
        self.provider_name = provider_name
        self.cause = cause


def _run_provider_wave1_worker(
    provider: CitationProvider,
    l1_norm: Sequence[str],
    all_papers_snapshot: Dict[str, IngestionPaper],
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    exhaustive: bool,
    max_l2: int,
    max_l3: Optional[int],
    provider_seed_state: Optional[Dict[str, Dict[str, Any]]],
    provider_l3_state: Optional[Dict[str, Any]],
    provider_transient_failures: Optional[Dict[str, Dict[str, Any]]],
    checkpoint_root: Path,
    request_key: str,
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[
    Dict[str, IngestionPaper],
    Dict[str, Set[str]],
    Dict[str, IngestionPaper],
    Dict[str, object],
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
]:
    """Run one provider ingestion unit for parallel wave-1 execution."""
    try:
        local_seed_state: Dict[str, Dict[str, Any]] = dict(provider_seed_state or {})
        local_l3_state: Dict[str, Any] = dict(provider_l3_state or {})
        local_transient_failures: Dict[str, Dict[str, Any]] = dict(
            provider_transient_failures or {}
        )

        _replay_provider_transient_failures(provider.name, local_transient_failures, {})

        seed_metadata = provider.fetch_seed_metadata(l1_norm)
        local_papers_resolved = dict(all_papers_snapshot)
        _merge_seed_metadata(local_papers_resolved, seed_metadata)

        def _seed_progress(seed_id: str, state: Dict[str, Any]) -> None:
            local_seed_state[seed_id] = dict(state)
            _write_provider_checkpoint_state(
                checkpoint_root,
                request_key,
                provider.name,
                provider_pagination_state=local_seed_state,
                provider_l3_state=local_l3_state,
                transient_failures=local_transient_failures,
            )

        def _l3_progress(state: Dict[str, Any]) -> None:
            local_l3_state.clear()
            local_l3_state.update(dict(state))
            _write_provider_checkpoint_state(
                checkpoint_root,
                request_key,
                provider.name,
                provider_pagination_state=local_seed_state,
                provider_l3_state=local_l3_state,
                transient_failures=local_transient_failures,
            )

        provider_edges, provider_papers, stats = _fetch_provider_graph(
            provider=provider,
            l1_norm=l1_norm,
            l1_papers_resolved=local_papers_resolved,
            theory_name=theory_name,
            key_constructs=key_constructs,
            depth=depth,
            exhaustive=exhaustive,
            max_l2=max_l2,
            max_l3=max_l3,
            provider_seed_state=local_seed_state,
            seed_progress_callback=_seed_progress,
            provider_progress_callback=provider_progress_callback,
            provider_l3_state=local_l3_state,
            l3_progress_callback=_l3_progress,
            include_l3=False,
        )

        drained_failures = _drain_transient_request_failures(provider.name)
        _merge_provider_transient_failures(local_transient_failures, drained_failures)
        _prune_provider_transient_failures(provider.name, local_transient_failures)

        return (
            seed_metadata,
            provider_edges,
            provider_papers,
            stats,
            local_seed_state,
            local_l3_state,
            local_transient_failures,
        )
    except Exception as exc:  # pragma: no cover - exercised by parallel failure tests
        raise ProviderIngestionError(provider.name, exc) from exc


def _run_provider_l2_to_l3_worker(
    provider: CitationProvider,
    l2_parent_ids: Sequence[str],
    max_l3: Optional[int],
    provider_l3_state: Optional[Dict[str, Any]],
    provider_transient_failures: Optional[Dict[str, Dict[str, Any]]],
    checkpoint_root: Path,
    request_key: str,
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, IngestionPaper],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
]:
    """Run L2->L3 expansion for one provider with provider-local checkpoint updates."""
    try:
        local_l3_state: Dict[str, Any] = dict(provider_l3_state or {})
        local_transient_failures: Dict[str, Dict[str, Any]] = dict(
            provider_transient_failures or {}
        )
        _replay_provider_transient_failures(provider.name, local_transient_failures, {})

        def _l3_progress(state: Dict[str, Any]) -> None:
            local_l3_state.clear()
            local_l3_state.update(dict(state))
            if provider_progress_callback is not None:
                progress_raw = state.get("next_l2_index")
                progress_index = progress_raw if isinstance(progress_raw, int) else 0
                provider_progress_callback(
                    progress_index,
                    len(l2_parent_ids),
                    str(state.get("status") or "in_progress"),
                )
            _write_provider_checkpoint_state(
                checkpoint_root,
                request_key,
                provider.name,
                provider_pagination_state={},
                provider_l3_state=local_l3_state,
                transient_failures=local_transient_failures,
            )

        l3_call_kwargs: Dict[str, Any] = {
            "l2_paper_ids": list(l2_parent_ids),
            "max_l3": max_l3,
        }
        l3_signature = inspect.signature(provider.fetch_l3_references)
        if "resume_state" in l3_signature.parameters:
            l3_call_kwargs["resume_state"] = local_l3_state
        if "progress_callback" in l3_signature.parameters:
            l3_call_kwargs["progress_callback"] = _l3_progress

        l3_edges, l3_papers = provider.fetch_l3_references(**l3_call_kwargs)
        if provider_progress_callback is not None:
            provider_progress_callback(len(l2_parent_ids), len(l2_parent_ids), "complete")
        drained_failures = _drain_transient_request_failures(provider.name)
        _merge_provider_transient_failures(local_transient_failures, drained_failures)
        _prune_provider_transient_failures(provider.name, local_transient_failures)
        return l3_edges, l3_papers, local_l3_state, local_transient_failures
    except Exception as exc:  # pragma: no cover - exercised by parallel failure tests
        raise ProviderIngestionError(provider.name, exc) from exc


def _merge_seed_metadata(
    all_papers: Dict[str, IngestionPaper],
    seed_papers: Dict[str, IngestionPaper],
) -> None:
    """Ensure seed paper placeholders/metadata are present in global paper map."""
    for pid, paper in seed_papers.items():
        if pid in all_papers:
            all_papers[pid] = _merge_papers(all_papers[pid], paper)
        else:
            all_papers[pid] = paper


def _paper_has_provider_identity(
    paper: Optional[IngestionPaper],
    provider_name: str,
    paper_id: str,
) -> bool:
    """Return True when a deduped paper can be queried by the given provider."""
    if paper and provider_name in paper.source_ids:
        return True

    if provider_name == "openalex":
        return paper_id.startswith("openalex:")
    if provider_name == "semantic_scholar":
        return paper_id.startswith("semantic_scholar:")
    if provider_name == "core":
        return paper_id.startswith("core:")
    if provider_name == "crossref":
        return _doi_from_identifier(paper_id) is not None
    if paper_id.startswith(f"{provider_name}:"):
        return True
    return False


def _provider_l2_parents_for_l3(
    provider_name: str,
    l2_parent_ids: Sequence[str],
    all_papers: Dict[str, IngestionPaper],
) -> List[str]:
    """Select deduped L2 parents that can be expanded by a specific provider."""
    if provider_name not in {"openalex", "semantic_scholar", "core", "crossref"}:
        return list(l2_parent_ids)

    provider_l2_ids: List[str] = []
    for parent_id in l2_parent_ids:
        if _paper_has_provider_identity(all_papers.get(parent_id), provider_name, parent_id):
            provider_l2_ids.append(parent_id)
    return provider_l2_ids


def _request_payload(
    theory_name: str,
    l1_papers: Sequence[str],
    key_constructs: Optional[Sequence[str]],
    sources: Sequence[str],
    depth: str,
    max_l2: int,
    max_l3: Optional[int],
    exhaustive: bool,
) -> dict:
    """Build stable request payload used to derive cache/checkpoint key."""
    return {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "theory_name": theory_name,
        "l1_papers": list(l1_papers),
        "key_constructs": list(key_constructs or []),
        "sources": list(sources),
        "depth": depth,
        "max_l2": int(max_l2),
        "max_l3": int(max_l3) if max_l3 is not None else None,
        "exhaustive": bool(exhaustive),
    }


def _load_cached_result(cache_root: Path, key: str, refresh: bool) -> Optional[IngestionResult]:
    """Return cached ingestion result unless refresh is explicitly requested."""
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
    transient_failure_summary: Optional[Dict[str, Dict[str, int]]] = None,
    checkpoint_stats: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Assemble final run metadata for output payload and cache."""
    metadata = {
        "sources": list(source_list),
        "depth": depth,
        "provider_stats": provider_stats,
        "fetch_stats": {
            "total_requests": int(_INGEST_STATS["total_requests"]),
            "total_failures": int(_INGEST_STATS["total_failures"]),
            "per_provider_failures": dict(_INGEST_STATS["per_provider_failures"]),
            "per_status_failures": dict(_INGEST_STATS["per_status_failures"]),
        },
        "completeness": completeness,
        "cache_key": cache_key,
        "alias_count": len(alias_map),
        "paper_count": len(papers_data),
        "edge_count": sum(len(values) for values in citation_data.values()),
        "transient_failure_summary": transient_failure_summary or {},
    }
    if checkpoint_stats is not None:
        metadata["checkpoint_stats"] = checkpoint_stats
    return metadata


class OpenAlexProvider(CitationProvider):
    name = "openalex"
    capabilities = ProviderCapabilities(
        True, True, True, supports_cited_by_traversal=True, supports_l3_outgoing=True
    )

    def _collect_l3_reference_edges(
        self,
        l2_paper_ids: Sequence[str],
        budget: Optional[int],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], Optional[int]]:
        """Collect L3 reference edges from OpenAlex works while respecting budget."""
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}

        for pid in l2_paper_ids:
            if budget is not None and budget <= 0:
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
                if budget is not None and budget <= 0:
                    break
                ref_id = normalize_identifier(ref, source=self.name)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, _openalex_reference_stub(ref_id, ref))
                if budget is not None:
                    budget -= 1
            time.sleep(0.03)

        return edges, papers, budget

    def _hydrate_l3_reference_papers(self, papers: Dict[str, IngestionPaper]) -> None:
        """Hydrate discovered OpenAlex L3 stubs with identity metadata fields."""
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

            hydrated = _openalex_hydrated_paper(ref_id, payload)
            papers[ref_id] = _merge_papers(existing, hydrated)
            time.sleep(0.03)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        """Resolve OpenAlex metadata for seed identifiers (DOI or OpenAlex ID)."""
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
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, IngestionPaper], int, str]:
        """Fetch all papers citing this OpenAlex work via cursor-paginated cited-by traversal."""
        state = resume_state or {}
        papers = _deserialize_papers(state.get("papers"))
        # Strip "openalex:" prefix to get bare work ID for the API filter.
        work_id = l1_provider_id.split(":", 1)[-1] if ":" in l1_provider_id else l1_provider_id
        cursor = str(state.get("cursor") or "*")
        expected_count = _parse_optional_int(state.get("expected_count"))

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
                _emit_citers_progress(
                    progress_callback=progress_callback,
                    status=status,
                    position_key="cursor",
                    position_value=cursor,
                    expected_count=expected_count,
                    papers=papers,
                )
                return papers, expected_count or 0, status

            if expected_count is None:
                meta = payload.get("meta") or {}
                expected_count = _parse_optional_int(meta.get("count"), default=0)

            results = payload.get("results") or []
            for item in results:
                oa_id = item.get("id") or ""
                paper_id = normalize_identifier(oa_id, source=self.name)
                if paper_id:
                    papers[paper_id] = _paper_from_openalex_item(item, paper_id)

            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if not next_cursor or not results:
                break

            _emit_citers_progress(
                progress_callback=progress_callback,
                status="in_progress",
                position_key="cursor",
                position_value=next_cursor,
                expected_count=expected_count,
                papers=papers,
            )
            cursor = next_cursor
            time.sleep(0.03)

        fetched = len(papers)
        status = "complete" if (expected_count is None or fetched >= expected_count) else "partial"
        _emit_citers_progress(
            progress_callback=progress_callback,
            status=status,
            position_key="cursor",
            position_value=None,
            expected_count=expected_count or fetched,
            papers=papers,
            fetched_count=fetched,
        )
        return papers, expected_count or fetched, status

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Search OpenAlex and retain only candidates that cite an L1 seed."""
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
        max_l3: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Expand L2 OpenAlex works to referenced papers with checkpoint-resume support."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l2_index",
            max_edges=max_l3,
        )

        for idx in range(start_index, len(l2_paper_ids)):
            pid = l2_paper_ids[idx]
            if budget is not None and budget <= 0:
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
                if budget is not None and budget <= 0:
                    break
                ref_id = normalize_identifier(ref, source=self.name)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, _openalex_reference_stub(ref_id, ref))
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l2_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        self._hydrate_l3_reference_papers(papers)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l2_index",
            index_value=len(l2_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers

    def fetch_l3_outgoing_references(
        self,
        l3_paper_ids: Sequence[str],
        max_edges: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Fetch outgoing references from L3 papers for L3-to-L3 edge discovery."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l3_parent_index",
            max_edges=max_edges,
        )

        for idx in range(start_index, len(l3_paper_ids)):
            pid = l3_paper_ids[idx]
            if budget is not None and budget <= 0:
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
                if budget is not None and budget <= 0:
                    break
                ref_id = normalize_identifier(ref, source=self.name)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, _openalex_reference_stub(ref_id, ref))
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l3_parent_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l3_parent_index",
            index_value=len(l3_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers


class SemanticScholarProvider(CitationProvider):
    name = "semantic_scholar"
    capabilities = ProviderCapabilities(
        True, True, True, supports_cited_by_traversal=True, supports_l3_outgoing=True
    )

    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider with optional API key for higher rate limits."""
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        """Return request headers for Semantic Scholar calls."""
        return _semantic_scholar_auth_headers(self.api_key)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        """Resolve Semantic Scholar metadata for DOI or S2 seed IDs."""
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
                    headers=self._headers(),
                )
            elif paper_id.startswith("semantic_scholar:"):
                token = paper_id.split(":", 1)[1]
                payload = _safe_get(
                    "https://api.semanticscholar.org/graph/v1/paper/"
                    f"{token}?fields=paperId,title,abstract,year,citationCount,externalIds",
                    provider=self.name,
                    headers=self._headers(),
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
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, IngestionPaper], int, str]:
        """Fetch all papers citing this S2 paper via offset-paginated citations endpoint."""
        state = resume_state or {}
        papers = _deserialize_papers(state.get("papers"))
        token = l1_provider_id.split(":", 1)[-1] if ":" in l1_provider_id else l1_provider_id
        offset = _parse_optional_int(state.get("offset"), default=len(papers)) or len(papers)
        limit = 1000
        expected_count = _parse_optional_int(state.get("expected_count"))

        while True:
            batch_limit = _semantic_batch_limit(offset, limit, papers, max_results)
            if batch_limit is None:
                fetched, status = _semantic_fetch_status(papers, expected_count)
                return papers, expected_count or fetched, status

            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/{token}/citations"
                f"?fields=paperId,title,year,citationCount,externalIds,abstract"
                f"&limit={batch_limit}&offset={offset}"
            )
            payload = _safe_get(url, provider=self.name, headers=self._headers())
            if not payload:
                status = "failed" if not papers else "partial"
                _emit_citers_progress(
                    progress_callback=progress_callback,
                    status=status,
                    position_key="offset",
                    position_value=offset,
                    expected_count=expected_count,
                    papers=papers,
                )
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
            _emit_citers_progress(
                progress_callback=progress_callback,
                status="in_progress",
                position_key="offset",
                position_value=offset,
                expected_count=expected_count,
                papers=papers,
            )
            time.sleep(0.05)

        fetched = len(papers)
        status = "complete" if (expected_count is None or fetched >= expected_count) else "partial"
        _emit_citers_progress(
            progress_callback=progress_callback,
            status=status,
            position_key="offset",
            position_value=offset,
            expected_count=expected_count or fetched,
            papers=papers,
            fetched_count=fetched,
        )
        return papers, expected_count or fetched, status

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Search Semantic Scholar and keep only papers citing L1 seeds."""
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        query = urllib.parse.quote(_query_terms(theory_name, key_constructs))
        limit = max(1, min(max_l2, 100))
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={query}&limit={limit}&fields=paperId,title,year,citationCount,references.paperId,references.externalIds"
        )
        payload = _safe_get(url, provider=self.name, headers=self._headers())
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
        max_l3: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Expand L2 Semantic papers to references with checkpoint-resume support."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l2_index",
            max_edges=max_l3,
        )

        for idx in range(start_index, len(l2_paper_ids)):
            pid = l2_paper_ids[idx]
            if budget is not None and budget <= 0:
                break
            if not pid.startswith("semantic_scholar:"):
                continue

            token = pid.split(":", 1)[1]
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"{token}?fields=references.paperId,references.externalIds"
            )
            payload = _safe_get(url, provider=self.name, headers=self._headers())
            if not payload:
                continue

            for ref in payload.get("references", []) or []:
                if budget is not None and budget <= 0:
                    break
                paper = _paper_from_semantic_reference(ref)
                if not paper:
                    continue
                ref_id = paper.paper_id
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, paper)
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l2_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.05)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l2_index",
            index_value=len(l2_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers

    def fetch_l3_outgoing_references(
        self,
        l3_paper_ids: Sequence[str],
        max_edges: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Fetch outgoing references from L3 papers for L3-to-L3 edge discovery."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l3_parent_index",
            max_edges=max_edges,
        )

        for idx in range(start_index, len(l3_paper_ids)):
            pid = l3_paper_ids[idx]
            if budget is not None and budget <= 0:
                break
            if not pid.startswith("semantic_scholar:"):
                continue

            token = pid.split(":", 1)[1]
            url = (
                "https://api.semanticscholar.org/graph/v1/paper/"
                f"{token}?fields=references.paperId,references.externalIds"
            )
            payload = _safe_get(url, provider=self.name, headers=self._headers())
            if not payload:
                continue

            for ref in payload.get("references", []) or []:
                if budget is not None and budget <= 0:
                    break
                paper = _paper_from_semantic_reference(ref)
                if not paper:
                    continue
                ref_id = paper.paper_id
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(ref_id, paper)
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l3_parent_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.05)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l3_parent_index",
            index_value=len(l3_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers


class CrossrefProvider(CitationProvider):
    name = "crossref"
    capabilities = ProviderCapabilities(True, True, False, supports_l3_outgoing=True)

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        """Fetch Crossref metadata for DOI seeds or DOI-normalized inputs."""
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
        # L2 contract: all admitted L2 papers must explicitly cite L1.
        # Crossref keyword search cannot provide this guarantee reliably,
        # so Crossref does not contribute L2 graph candidates.
        return {}, {}

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Expand L2 papers to L3 references via Crossref DOI-based lookups.

        Only references that themselves carry a DOI are admitted as new L3
        nodes (DOI-gated).  References without DOIs are silently dropped.
        """
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l2_index",
            max_edges=max_l3,
        )

        for idx in range(start_index, len(l2_paper_ids)):
            pid = l2_paper_ids[idx]
            if budget is not None and budget <= 0:
                break

            doi = _doi_from_identifier(pid)
            if not doi:
                continue

            payload = _safe_get(
                f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                provider=self.name,
            )
            item = (payload or {}).get("message")
            if not item:
                continue

            for ref in item.get("reference", []) or []:
                if budget is not None and budget <= 0:
                    break
                ref_doi = ref.get("DOI")
                if not ref_doi:
                    continue
                ref_id = normalize_identifier(ref_doi)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(
                    ref_id,
                    IngestionPaper(
                        paper_id=ref_id,
                        title=ref.get("article-title") or "",
                        year=(
                            int(ref["year"])
                            if ref.get("year") and str(ref["year"]).isdigit()
                            else None
                        ),
                        doi=ref_doi.lower(),
                        source_ids={self.name: ref_doi.lower()},
                    ),
                )
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l2_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l2_index",
            index_value=len(l2_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers

    def fetch_l3_outgoing_references(
        self,
        l3_paper_ids: Sequence[str],
        max_edges: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Fetch outgoing references from L3 papers for L3-to-L3 edge discovery.

        Reuses the Crossref /works/{doi} API; only L3 parents with DOIs are
        queried.
        """
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l3_parent_index",
            max_edges=max_edges,
        )

        for idx in range(start_index, len(l3_paper_ids)):
            pid = l3_paper_ids[idx]
            if budget is not None and budget <= 0:
                break

            doi = _doi_from_identifier(pid)
            if not doi:
                continue

            payload = _safe_get(
                f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='')}",
                provider=self.name,
            )
            item = (payload or {}).get("message")
            if not item:
                continue

            for ref in item.get("reference", []) or []:
                if budget is not None and budget <= 0:
                    break
                ref_doi = ref.get("DOI")
                if not ref_doi:
                    continue
                ref_id = normalize_identifier(ref_doi)
                if not ref_id:
                    continue
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(
                    ref_id,
                    IngestionPaper(
                        paper_id=ref_id,
                        title=ref.get("article-title") or "",
                        year=(
                            int(ref["year"])
                            if ref.get("year") and str(ref["year"]).isdigit()
                            else None
                        ),
                        doi=ref_doi.lower(),
                        source_ids={self.name: ref_doi.lower()},
                    ),
                )
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l3_parent_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l3_parent_index",
            index_value=len(l3_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers


class CoreProvider(CitationProvider):
    name = "core"
    capabilities = ProviderCapabilities(
        True, True, True, supports_cited_by_traversal=False, supports_l3_outgoing=True
    )

    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider with optional CORE API key."""
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        """Return request headers for CORE API calls."""
        return _core_auth_headers(self.api_key)

    def _search_works(self, query: str, limit: int) -> List[dict]:
        """Search CORE works endpoint with bounded page size."""
        params = urllib.parse.urlencode({"q": query, "limit": max(1, min(limit, 100))})
        payload = _safe_get(
            f"https://api.core.ac.uk/v3/search/works/?{params}",
            provider=self.name,
            headers=self._headers(),
        )
        return list((payload or {}).get("results") or [])

    def _work_by_identifier(self, identifier: str) -> Optional[dict]:
        """Fetch CORE work directly by provider-native identifier."""
        encoded = urllib.parse.quote(identifier, safe=":/.")
        return _safe_get(
            f"https://api.core.ac.uk/v3/works/{encoded}",
            provider=self.name,
            headers=self._headers(),
        )

    def _lookup_work(self, paper_id: str) -> Optional[dict]:
        """Resolve a canonical paper identifier to a CORE work payload."""
        doi = _doi_from_identifier(paper_id)
        if doi:
            matches = self._search_works(f'doi:"{doi}"', limit=1)
            return matches[0] if matches else None
        if paper_id.startswith("core:"):
            return self._work_by_identifier(paper_id.split(":", 1)[1])
        return None

    def _reference_to_paper(self, ref: dict) -> Optional[IngestionPaper]:
        """Map CORE reference object to canonical paper model when identifiable."""
        ref_doi = ref.get("doi")
        ref_id = ref.get("id")
        if ref_doi:
            normalized_ref = normalize_identifier(str(ref_doi))
        elif ref_id is not None:
            normalized_ref = normalize_identifier(str(ref_id), source=self.name)
        else:
            normalized_ref = ""

        if not normalized_ref:
            return None

        return IngestionPaper(
            paper_id=normalized_ref,
            title=str(ref.get("title") or ""),
            doi=str(ref_doi) if ref_doi else None,
            source_ids={"core": str(ref_id)} if ref_id is not None else {},
        )

    def fetch_seed_metadata(
        self,
        l1_papers: Sequence[str],
    ) -> Dict[str, IngestionPaper]:
        """Resolve CORE metadata for DOI or CORE-ID seeds."""
        papers: Dict[str, IngestionPaper] = {}
        for paper_id in l1_papers:
            doi = _doi_from_identifier(paper_id)
            payload: Optional[dict] = None

            if doi:
                matches = self._search_works(f'doi:"{doi}"', limit=1)
                payload = matches[0] if matches else None
            elif paper_id.startswith("core:"):
                payload = self._work_by_identifier(paper_id.split(":", 1)[1])

            if not payload:
                continue

            papers[paper_id] = _paper_from_core_item(payload, paper_id)
            time.sleep(0.03)
        return papers

    def fetch_l2_and_metadata(
        self,
        l1_papers: Sequence[str],
        theory_name: str,
        key_constructs: Optional[Sequence[str]] = None,
        max_l2: int = 200,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Search CORE and keep results that explicitly reference L1 seeds."""
        edges: Dict[str, Set[str]] = {}
        papers: Dict[str, IngestionPaper] = {}
        l1_norm = {normalize_identifier(v) for v in l1_papers}

        items = self._search_works(_query_terms(theory_name, key_constructs), limit=max_l2)
        for item in items:
            doi = item.get("doi")
            core_id = item.get("id")
            paper_id = ""
            if doi:
                paper_id = normalize_identifier(str(doi))
            elif core_id is not None:
                paper_id = normalize_identifier(str(core_id), source=self.name)
            if not paper_id:
                continue

            linked_l1: Set[str] = set()
            for ref in item.get("references") or []:
                for candidate in _core_reference_candidates(ref):
                    if candidate in l1_norm:
                        linked_l1.add(candidate)

            if not linked_l1:
                continue

            edges.setdefault(paper_id, set()).update(linked_l1)
            papers[paper_id] = _paper_from_core_item(item, paper_id)
            for ref in linked_l1:
                papers.setdefault(ref, IngestionPaper(paper_id=ref, doi=_doi_from_identifier(ref)))
            time.sleep(0.03)

        return edges, papers

    def fetch_l3_references(
        self,
        l2_paper_ids: Sequence[str],
        max_l3: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Expand CORE L2 papers to references with checkpoint-resume support."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l2_index",
            max_edges=max_l3,
        )

        for idx in range(start_index, len(l2_paper_ids)):
            pid = l2_paper_ids[idx]
            if budget is not None and budget <= 0:
                break

            payload = self._lookup_work(pid)
            if not payload:
                continue

            for ref in payload.get("references") or []:
                if budget is not None and budget <= 0:
                    break

                paper = self._reference_to_paper(ref)
                if not paper:
                    continue

                edges.setdefault(pid, set()).add(paper.paper_id)
                papers.setdefault(paper.paper_id, paper)
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l2_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l2_index",
            index_value=len(l2_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers

    def fetch_l3_outgoing_references(
        self,
        l3_paper_ids: Sequence[str],
        max_edges: Optional[int] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]:
        """Fetch outgoing references from L3 papers for L3-to-L3 edge discovery."""
        edges, papers, start_index, budget = _restore_traversal_state(
            resume_state=resume_state,
            index_key="next_l3_parent_index",
            max_edges=max_edges,
        )

        for idx in range(start_index, len(l3_paper_ids)):
            pid = l3_paper_ids[idx]
            if budget is not None and budget <= 0:
                break

            payload = self._lookup_work(pid)
            if not payload:
                continue

            for ref in payload.get("references") or []:
                if budget is not None and budget <= 0:
                    break

                paper = self._reference_to_paper(ref)
                if not paper:
                    continue

                edges.setdefault(pid, set()).add(paper.paper_id)
                papers.setdefault(paper.paper_id, paper)
                if budget is not None:
                    budget -= 1

            _emit_traversal_progress(
                progress_callback=progress_callback,
                status="in_progress",
                index_key="next_l3_parent_index",
                index_value=idx + 1,
                budget_remaining=budget,
                edges=edges,
                papers=papers,
            )
            time.sleep(0.03)

        _emit_traversal_progress(
            progress_callback=progress_callback,
            status="complete",
            index_key="next_l3_parent_index",
            index_value=len(l3_paper_ids),
            budget_remaining=budget,
            edges=edges,
            papers=papers,
        )

        return edges, papers


_PROVIDER_REGISTRY = {
    "openalex": OpenAlexProvider,
    "semantic_scholar": SemanticScholarProvider,
    "crossref": CrossrefProvider,
    "core": CoreProvider,
}


def build_providers(sources: Sequence[str]) -> List[CitationProvider]:
    """Instantiate provider objects for requested source names."""
    providers: List[CitationProvider] = []
    for source in sources:
        key = source.strip().lower()
        provider_cls = _PROVIDER_REGISTRY.get(key)
        if provider_cls is None:
            continue
        if key == "core":
            providers.append(provider_cls(api_key=os.getenv("CORE_API_KEY")))
        elif key == "semantic_scholar":
            providers.append(provider_cls(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")))
        else:
            providers.append(provider_cls())
    return providers


def _dedupe_and_materialize(
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, object]], Dict[str, str]]:
    """Deduplicate papers across providers and materialize final output structures."""
    merged_by_key: Dict[str, IngestionPaper] = {}
    key_to_final_id: Dict[str, str] = {}

    for paper in tqdm(all_papers.values(), disable=_QUIET, leave=False):
        key = _canonical_merge_key(paper)
        if key in merged_by_key:
            merged_by_key[key] = _merge_papers(merged_by_key[key], paper)
        else:
            merged_by_key[key] = paper

    # Map final canonical paper_id → materialized metadata dict (output-ready paper records)
    papers_out: Dict[str, Dict[str, object]] = {}

    # Map each original/alias paper ID to its canonical final ID for edge normalization.
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


def _default_checkpoint_stats() -> Dict[str, object]:
    """Create the default checkpoint statistics payload."""
    return {
        "hit": False,
        "miss": True,
        "cache_short_circuit": False,
        "providers_skipped": 0,
        "skipped_provider_names": [],
        "providers_executed": 0,
        "executed_provider_names": [],
        "stale_state_ignored_count": 0,
        "stale_state_ignored_seeds": [],
        "l3_stale_state_ignored_count": 0,
        "l3_stale_state_ignored_providers": [],
        "l3_resumed_providers": [],
        "l3_resumed_parent_count": 0,
        "l3_to_l3_edges_added": 0,
        "l3_to_l3_parent_scanned_count": 0,
        "l3_to_l3_resumed_providers": [],
        "transient_failures_retried": 0,
        "transient_failures_resumed_success": 0,
        "transient_failures_exhausted": 0,
        "transient_failures_pruned": 0,
        "transient_failures_queued": 0,
    }


def _cached_result_with_checkpoint_stats(cached: IngestionResult) -> IngestionResult:
    """Attach standard checkpoint stats to cached short-circuit responses."""
    metadata = dict(cached.metadata or {})
    existing_checkpoint_stats = metadata.get("checkpoint_stats")
    if not isinstance(existing_checkpoint_stats, dict):
        existing_checkpoint_stats = {}

    merged_stats = dict(existing_checkpoint_stats)
    merged_stats.update(_default_checkpoint_stats())
    merged_stats["cache_short_circuit"] = True
    metadata["checkpoint_stats"] = merged_stats
    cached.metadata = metadata
    return cached


def _initialize_runtime_state(
    l1_papers: Sequence[str],
) -> tuple[
    List[str],
    Dict[str, IngestionPaper],
    Dict[str, Set[str]],
    Dict[str, object],
    Dict[str, Dict[str, object]],
    Set[str],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    str,
]:
    """Create default in-memory state containers for ingestion runtime."""
    l1_norm, seed_papers = _seed_l1_papers(l1_papers)
    all_edges: Dict[str, Set[str]] = {}
    all_papers: Dict[str, IngestionPaper] = dict(seed_papers)
    provider_stats: Dict[str, object] = {}
    combined_completeness: Dict[str, Dict[str, object]] = {}
    completed_providers: Set[str] = set()
    provider_pagination_state: Dict[str, Dict[str, Dict[str, Any]]] = {}
    provider_l3_state: Dict[str, Dict[str, Any]] = {}
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]] = {}
    l3_to_l3_state: Dict[str, Dict[str, Any]] = {}
    ingestion_phase = "l2_to_l3"
    return (
        l1_norm,
        all_papers,
        all_edges,
        provider_stats,
        combined_completeness,
        completed_providers,
        provider_pagination_state,
        provider_l3_state,
        provider_transient_failures,
        l3_to_l3_state,
        ingestion_phase,
    )


def _restore_runtime_state_from_checkpoint(
    checkpoint_state: Dict[str, Any],
    seed_papers: Dict[str, IngestionPaper],
    checkpoint_stats: Dict[str, object],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, object],
    combined_completeness: Dict[str, Dict[str, object]],
    completed_providers: Set[str],
    provider_pagination_state: Dict[str, Dict[str, Dict[str, Any]]],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    l3_to_l3_state: Dict[str, Dict[str, Any]],
    ingestion_phase: str,
) -> tuple[
    Dict[str, Set[str]],
    Dict[str, IngestionPaper],
    Dict[str, object],
    Dict[str, Dict[str, object]],
    Set[str],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    str,
]:
    """Restore runtime state dictionaries from checkpoint payload."""
    checkpoint_stats["hit"] = True
    checkpoint_stats["miss"] = False

    all_edges = _deserialize_edges(checkpoint_state.get("all_edges"))
    all_papers = _deserialize_papers(checkpoint_state.get("all_papers"))
    _merge_seed_metadata(all_papers, seed_papers)

    raw_provider_stats = checkpoint_state.get("provider_stats")
    if isinstance(raw_provider_stats, dict):
        provider_stats = raw_provider_stats

    raw_completeness = checkpoint_state.get("combined_completeness")
    if isinstance(raw_completeness, dict):
        combined_completeness = raw_completeness

    raw_completed = checkpoint_state.get("completed_providers")
    if isinstance(raw_completed, list):
        completed_providers = {value for value in raw_completed if isinstance(value, str)}

    provider_pagination_state = _deserialize_provider_pagination_state(
        checkpoint_state.get("provider_pagination_state")
    )
    provider_l3_state = _deserialize_provider_l3_state(checkpoint_state.get("provider_l3_state"))
    provider_transient_failures = _deserialize_transient_failures(
        checkpoint_state.get("provider_transient_failures")
    )
    l3_to_l3_state = _deserialize_provider_l3_state(checkpoint_state.get("l3_to_l3_state"))
    raw_phase = checkpoint_state.get("ingestion_phase")
    if isinstance(raw_phase, str) and raw_phase in ("l2_to_l3", "l3_to_l3"):
        ingestion_phase = raw_phase

    _progress(f"[ADIT] Resuming from checkpoint: {len(completed_providers)} completed provider(s)")
    return (
        all_edges,
        all_papers,
        provider_stats,
        combined_completeness,
        completed_providers,
        provider_pagination_state,
        provider_l3_state,
        provider_transient_failures,
        l3_to_l3_state,
        ingestion_phase,
    )


def _drop_stale_seed_resume_state(
    provider_name: str,
    provider_state: Dict[str, Dict[str, Any]],
    checkpoint_stats: Dict[str, object],
    max_age_seconds: int,
) -> bool:
    """Drop stale per-seed pagination state for a provider."""
    stale_removed = False
    for seed_id, seed_state in list(provider_state.items()):
        if not _is_pagination_state_stale(seed_state, max_age_seconds=max_age_seconds):
            continue
        provider_state.pop(seed_id, None)
        stale_removed = True
        stale_seeds = checkpoint_stats.get("stale_state_ignored_seeds")
        if isinstance(stale_seeds, list):
            stale_seeds.append(f"{provider_name}:{seed_id}")
        checkpoint_stats["stale_state_ignored_count"] = (
            int(checkpoint_stats.get("stale_state_ignored_count", 0)) + 1
        )
        _vprint(f"  [{provider_name}] Ignoring stale pagination checkpoint for seed: {seed_id}")
    return stale_removed


def _drop_stale_provider_l3_state(
    provider_name: str,
    provider_l3_state: Dict[str, Dict[str, Any]],
    checkpoint_stats: Dict[str, object],
    max_age_seconds: int,
) -> bool:
    """Drop stale provider-level L3 checkpoint state."""
    run_state = provider_l3_state.get(provider_name)
    if not _is_pagination_state_stale(run_state or {}, max_age_seconds=max_age_seconds):
        return False

    provider_l3_state.pop(provider_name, None)
    l3_stale_providers = checkpoint_stats.get("l3_stale_state_ignored_providers")
    if isinstance(l3_stale_providers, list):
        l3_stale_providers.append(provider_name)
    checkpoint_stats["l3_stale_state_ignored_count"] = (
        int(checkpoint_stats.get("l3_stale_state_ignored_count", 0)) + 1
    )
    _vprint(f"  [{provider_name}] Ignoring stale L3 checkpoint state")
    return True


def _record_l3_resume_progress(
    provider_name: str,
    provider_l3_run_state: Dict[str, Any],
    checkpoint_stats: Dict[str, object],
) -> None:
    """Record checkpoint resume progress metadata for L3 expansion."""
    resumed_l3_index = provider_l3_run_state.get("next_l2_index")
    if not isinstance(resumed_l3_index, int) or resumed_l3_index <= 0:
        return

    resumed_providers = checkpoint_stats.get("l3_resumed_providers")
    if isinstance(resumed_providers, list) and provider_name not in resumed_providers:
        resumed_providers.append(provider_name)
    checkpoint_stats["l3_resumed_parent_count"] = (
        int(checkpoint_stats.get("l3_resumed_parent_count", 0)) + resumed_l3_index
    )


def _replay_provider_transient_failures(
    provider_name: str,
    failures: Dict[str, Dict[str, Any]],
    checkpoint_stats: Dict[str, object],
) -> None:
    """Replay eligible transient request failures for one provider before ingestion resumes."""
    _prune_provider_transient_failures(provider_name, failures, checkpoint_stats)
    if not failures:
        return

    attempted = 0
    succeeded = 0
    exhausted = 0
    now = time.time()
    for failure_key, record in list(failures.items()):
        wait_seconds = _transient_retry_wait_seconds(record, now_ts=now)
        if wait_seconds > 0.0:
            continue

        attempts = _parse_optional_int(record.get("attempts"), default=0) or 0
        if attempts >= _TRANSIENT_RETRY_MAX_ATTEMPTS:
            failures.pop(failure_key, None)
            exhausted += 1
            continue

        attempted += 1
        resume_state = (
            record.get("resume_state") if isinstance(record.get("resume_state"), dict) else {}
        )
        timeout = _parse_optional_int(resume_state.get("timeout"), default=20) or 20
        max_retries = (
            _parse_optional_int(
                resume_state.get("max_retries"),
                default=_SAFE_GET_MAX_RETRIES,
            )
            or _SAFE_GET_MAX_RETRIES
        )
        headers = _sanitize_retry_headers(resume_state.get("headers"))
        target_url = str(record.get("target_id") or "")
        if not target_url:
            failures.pop(failure_key, None)
            exhausted += 1
            continue

        failures.pop(failure_key, None)
        result = _safe_get(
            target_url,
            timeout=timeout,
            provider=provider_name,
            max_retries=max_retries,
            headers=headers,
        )
        drained = _drain_transient_request_failures(provider_name)
        _merge_provider_transient_failures(failures, drained)
        if result is not None:
            succeeded += 1
            continue

        if failure_key not in failures:
            exhausted += 1

    if attempted:
        checkpoint_stats["transient_failures_retried"] = (
            int(checkpoint_stats.get("transient_failures_retried", 0)) + attempted
        )
        checkpoint_stats["transient_failures_resumed_success"] = (
            int(checkpoint_stats.get("transient_failures_resumed_success", 0)) + succeeded
        )
        checkpoint_stats["transient_failures_exhausted"] = (
            int(checkpoint_stats.get("transient_failures_exhausted", 0)) + exhausted
        )
        _vprint(
            f"  [{provider_name}] Replay transient retries: "
            f"attempted={attempted}, succeeded={succeeded}, exhausted={exhausted}, queued={len(failures)}"
        )


def _build_provider_progress_callbacks(
    provider_state: Dict[str, Dict[str, Any]],
    provider_l3_run_state: Dict[str, Any],
    persist_callback: Callable[[], None],
) -> tuple[Callable[[str, Dict[str, Any]], None], Callable[[Dict[str, Any]], None]]:
    """Create state-persisting callbacks for provider L1 and L3 progress."""

    def _seed_progress(seed_id: str, state: Dict[str, Any]) -> None:
        provider_state[seed_id] = dict(state)
        persist_callback()

    def _l3_progress(state: Dict[str, Any]) -> None:
        provider_l3_run_state.clear()
        provider_l3_run_state.update(dict(state))
        persist_callback()

    return _seed_progress, _l3_progress


def _collect_l2_to_l3_jobs(
    l2_parent_ids: List[str],
    l3_providers: List[CitationProvider],
    provider_l3_state: Dict[str, Dict[str, Any]],
    all_papers: Dict[str, IngestionPaper],
    effective_staleness_seconds: int,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
) -> Tuple[Set[str], List[Tuple[CitationProvider, int, List[str], Dict[str, Any]]]]:
    """Collect L2->L3 expansion jobs, checking completion status and staleness."""
    l2_to_l3_completed: Set[str] = set()
    raw_l2_to_l3_completed = provider_l3_state.get("__completed_providers")
    if isinstance(raw_l2_to_l3_completed, dict):
        names = raw_l2_to_l3_completed.get("names")
        if isinstance(names, list):
            l2_to_l3_completed = {n for n in names if isinstance(n, str)}

    l3_jobs: List[Tuple[CitationProvider, int, List[str], Dict[str, Any]]] = []
    l3_provider_total = len(l3_providers)

    for l3_provider_index, provider in enumerate(l3_providers, start=1):
        if provider.name in l2_to_l3_completed:
            _progress_done(
                f"  [{provider.name}] L2->L3 provider {l3_provider_index}/{l3_provider_total}: "
                "skipped (checkpoint complete)"
            )
            continue

        provider_l3_run_state = provider_l3_state.get(provider.name)
        if _is_pagination_state_stale(
            provider_l3_run_state or {},
            max_age_seconds=effective_staleness_seconds,
        ):
            provider_l3_state.pop(provider.name, None)
            l3_stale_providers = checkpoint_stats.get("l3_stale_state_ignored_providers")
            if isinstance(l3_stale_providers, list):
                l3_stale_providers.append(provider.name)
            checkpoint_stats["l3_stale_state_ignored_count"] = (
                int(checkpoint_stats.get("l3_stale_state_ignored_count", 0)) + 1
            )
            persist_callback()

        provider_l3_run_state = provider_l3_state.setdefault(provider.name, {})
        resumed_l3_index = provider_l3_run_state.get("next_l2_index")
        if isinstance(resumed_l3_index, int) and resumed_l3_index > 0:
            resumed_providers = checkpoint_stats.get("l3_resumed_providers")
            if isinstance(resumed_providers, list) and provider.name not in resumed_providers:
                resumed_providers.append(provider.name)
            checkpoint_stats["l3_resumed_parent_count"] = (
                int(checkpoint_stats.get("l3_resumed_parent_count", 0)) + resumed_l3_index
            )

        provider_l2_parent_ids = _provider_l2_parents_for_l3(
            provider.name,
            l2_parent_ids,
            all_papers,
        )
        _progress(
            f"  [{provider.name}] L2->L3 provider {l3_provider_index}/{l3_provider_total}: "
            f"expanding {len(provider_l2_parent_ids)}/{len(l2_parent_ids)} parents"
        )
        if not provider_l2_parent_ids:
            _progress_done(
                f"  [{provider.name}] L2->L3 provider {l3_provider_index}/{l3_provider_total}: "
                "skipped (no provider-linked deduped parents)"
            )
            l2_to_l3_completed.add(provider.name)
            provider_l3_state["__completed_providers"] = {"names": sorted(l2_to_l3_completed)}
            provider_l3_state.pop(provider.name, None)
            persist_callback()
            continue

        l3_jobs.append((provider, l3_provider_index, provider_l2_parent_ids, provider_l3_run_state))

    return l2_to_l3_completed, l3_jobs


def _execute_l2_to_l3_jobs(
    l3_jobs: List[Tuple[CitationProvider, int, List[str], Dict[str, Any]]],
    l2_to_l3_completed: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    max_workers: Optional[int],
    max_l3: int,
    checkpoint_root: Path,
    key: str,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
    provider_progress_callbacks: Optional[Dict[str, Callable[[int, int, str], None]]] = None,
) -> None:
    """Execute L2->L3 jobs (parallel or sequential based on max_workers)."""

    if max_workers is not None and max_workers > 1 and len(l3_jobs) > 1:
        _execute_l2_to_l3_parallel_jobs(
            l3_jobs=l3_jobs,
            l2_to_l3_completed=l2_to_l3_completed,
            all_edges=all_edges,
            all_papers=all_papers,
            provider_stats=provider_stats,
            provider_l3_state=provider_l3_state,
            provider_transient_failures=provider_transient_failures,
            max_workers=max_workers,
            max_l3=max_l3,
            checkpoint_root=checkpoint_root,
            key=key,
            checkpoint_stats=checkpoint_stats,
            persist_callback=persist_callback,
            provider_progress_callbacks=provider_progress_callbacks,
        )
        return

    _execute_l2_to_l3_sequential_jobs(
        l3_jobs=l3_jobs,
        l2_to_l3_completed=l2_to_l3_completed,
        all_edges=all_edges,
        all_papers=all_papers,
        provider_stats=provider_stats,
        provider_l3_state=provider_l3_state,
        provider_transient_failures=provider_transient_failures,
        max_l3=max_l3,
        checkpoint_root=checkpoint_root,
        key=key,
        checkpoint_stats=checkpoint_stats,
        persist_callback=persist_callback,
        provider_progress_callbacks=provider_progress_callbacks,
    )


def _finalize_l2_to_l3_provider(
    provider: CitationProvider,
    l3_provider_index: int,
    l3_job_total: int,
    l3_edges: Dict[str, Set[str]],
    l3_papers: Dict[str, IngestionPaper],
    l2_to_l3_completed: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    provider_transient_state: Dict[str, Dict[str, Any]],
    checkpoint_root: Path,
    key: str,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
) -> None:
    _merge_provider_outputs(all_edges, all_papers, l3_edges, l3_papers)

    l3_edges_added = sum(len(cited) for cited in l3_edges.values())
    provider_stat = provider_stats.get(provider.name)
    if not isinstance(provider_stat, dict):
        provider_stat = {
            "l2_nodes": 0,
            "l2_edges": 0,
            "l3_edges": 0,
            "papers": 0,
            "completeness": {},
        }
    provider_stat["l3_edges"] = int(provider_stat.get("l3_edges", 0)) + l3_edges_added
    provider_stats[provider.name] = provider_stat

    _progress_done(
        f"  [{provider.name}] L2->L3 provider {l3_provider_index}/{l3_job_total}: "
        f"added {l3_edges_added} edges"
    )
    provider_transient_failures[provider.name] = dict(provider_transient_state)
    if provider_transient_state:
        checkpoint_stats["transient_failures_queued"] = int(
            checkpoint_stats.get("transient_failures_queued", 0)
        ) + len(provider_transient_state)
    else:
        l2_to_l3_completed.add(provider.name)
        provider_l3_state["__completed_providers"] = {"names": sorted(l2_to_l3_completed)}
        provider_transient_failures.pop(provider.name, None)
    provider_l3_state.pop(provider.name, None)
    _write_provider_checkpoint_state(
        checkpoint_root,
        key,
        provider.name,
        provider_pagination_state={},
        provider_l3_state={},
        transient_failures=provider_transient_state,
    )
    persist_callback()


def _restore_failed_l2_to_l3_state(
    provider_name: str,
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    checkpoint_root: Path,
    key: str,
    persist_callback: Callable[[], None],
) -> None:
    provider_checkpoint = _load_provider_checkpoint_state(
        checkpoint_root, key, provider_name, False
    )
    if not provider_checkpoint:
        return
    checkpoint_l3_state = provider_checkpoint.get("provider_l3_state")
    if isinstance(checkpoint_l3_state, dict):
        provider_l3_state[provider_name] = checkpoint_l3_state
    checkpoint_transient = provider_checkpoint.get("transient_failures")
    if isinstance(checkpoint_transient, dict):
        provider_transient_failures[provider_name] = _deserialize_transient_failures(
            {provider_name: checkpoint_transient}
        ).get(provider_name, {})
    if isinstance(checkpoint_l3_state, dict) or isinstance(checkpoint_transient, dict):
        persist_callback()


def _execute_l2_to_l3_parallel_jobs(
    l3_jobs: List[Tuple[CitationProvider, int, List[str], Dict[str, Any]]],
    l2_to_l3_completed: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    max_workers: int,
    max_l3: int,
    checkpoint_root: Path,
    key: str,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
    provider_progress_callbacks: Optional[Dict[str, Callable[[int, int, str], None]]] = None,
) -> None:
    effective_workers = min(max_workers, len(l3_jobs))
    merge_lock = threading.Lock()
    l3_job_total = len(l3_jobs)
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for provider, l3_provider_index, provider_l2_parent_ids, provider_l3_run_state in l3_jobs:
            provider_progress_callback = None
            if provider_progress_callbacks is not None:
                provider_progress_callback = provider_progress_callbacks.get(provider.name)
            futures[
                executor.submit(
                    _run_provider_l2_to_l3_worker,
                    provider,
                    provider_l2_parent_ids,
                    max_l3,
                    provider_l3_run_state,
                    provider_transient_failures.get(provider.name, {}),
                    checkpoint_root,
                    key,
                    provider_progress_callback,
                )
            ] = (provider, l3_provider_index, len(provider_l2_parent_ids))

        for future in as_completed(futures):
            provider, l3_provider_index, provider_parent_total = futures[future]
            try:
                l3_edges, l3_papers, local_l3_state, local_transient_state = future.result()
            except ProviderIngestionError as exc:
                if provider_progress_callbacks is not None:
                    progress_cb = provider_progress_callbacks.get(exc.provider_name)
                    if progress_cb is not None:
                        progress_cb(provider_parent_total, provider_parent_total, "failed")
                _restore_failed_l2_to_l3_state(
                    provider_name=exc.provider_name,
                    provider_l3_state=provider_l3_state,
                    provider_transient_failures=provider_transient_failures,
                    checkpoint_root=checkpoint_root,
                    key=key,
                    persist_callback=persist_callback,
                )
                provider_stats[exc.provider_name] = {
                    "status": "failed",
                    "error": str(exc.cause),
                }
                _progress(
                    f"  [{exc.provider_name}] L2->L3 provider {l3_provider_index}: "
                    f"failed ({exc.cause}) — continuing"
                )
                continue

            if provider_progress_callbacks is not None:
                progress_cb = provider_progress_callbacks.get(provider.name)
                if progress_cb is not None:
                    progress_cb(provider_parent_total, provider_parent_total, "done")

            with merge_lock:
                provider_l3_state[provider.name] = dict(local_l3_state)
                provider_transient_failures[provider.name] = dict(local_transient_state)
                _finalize_l2_to_l3_provider(
                    provider=provider,
                    l3_provider_index=l3_provider_index,
                    l3_job_total=l3_job_total,
                    l3_edges=l3_edges,
                    l3_papers=l3_papers,
                    l2_to_l3_completed=l2_to_l3_completed,
                    all_edges=all_edges,
                    all_papers=all_papers,
                    provider_stats=provider_stats,
                    provider_l3_state=provider_l3_state,
                    provider_transient_failures=provider_transient_failures,
                    provider_transient_state=local_transient_state,
                    checkpoint_root=checkpoint_root,
                    key=key,
                    checkpoint_stats=checkpoint_stats,
                    persist_callback=persist_callback,
                )


def _execute_l2_to_l3_sequential_jobs(
    l3_jobs: List[Tuple[CitationProvider, int, List[str], Dict[str, Any]]],
    l2_to_l3_completed: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    max_l3: int,
    checkpoint_root: Path,
    key: str,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
    provider_progress_callbacks: Optional[Dict[str, Callable[[int, int, str], None]]] = None,
) -> None:
    l3_job_total = len(l3_jobs)
    for provider, l3_provider_index, provider_l2_parent_ids, provider_l3_run_state in l3_jobs:
        provider_progress_callback = None
        if provider_progress_callbacks is not None:
            provider_progress_callback = provider_progress_callbacks.get(provider.name)
        try:
            l3_edges, l3_papers, local_l3_state, local_transient_state = (
                _run_provider_l2_to_l3_worker(
                    provider,
                    provider_l2_parent_ids,
                    max_l3,
                    provider_l3_run_state,
                    provider_transient_failures.get(provider.name, {}),
                    checkpoint_root,
                    key,
                    provider_progress_callback,
                )
            )
        except ProviderIngestionError as exc:
            if provider_progress_callback is not None:
                provider_progress_callback(
                    len(provider_l2_parent_ids),
                    len(provider_l2_parent_ids),
                    "failed",
                )
            _restore_failed_l2_to_l3_state(
                provider_name=provider.name,
                provider_l3_state=provider_l3_state,
                provider_transient_failures=provider_transient_failures,
                checkpoint_root=checkpoint_root,
                key=key,
                persist_callback=persist_callback,
            )
            raise exc.cause from exc
        if provider_progress_callback is not None:
            provider_progress_callback(
                len(provider_l2_parent_ids),
                len(provider_l2_parent_ids),
                "done",
            )
        provider_l3_state[provider.name] = dict(local_l3_state)
        provider_transient_failures[provider.name] = dict(local_transient_state)
        _finalize_l2_to_l3_provider(
            provider=provider,
            l3_provider_index=l3_provider_index,
            l3_job_total=l3_job_total,
            l3_edges=l3_edges,
            l3_papers=l3_papers,
            l2_to_l3_completed=l2_to_l3_completed,
            all_edges=all_edges,
            all_papers=all_papers,
            provider_stats=provider_stats,
            provider_l3_state=provider_l3_state,
            provider_transient_failures=provider_transient_failures,
            provider_transient_state=local_transient_state,
            checkpoint_root=checkpoint_root,
            key=key,
            checkpoint_stats=checkpoint_stats,
            persist_callback=persist_callback,
        )


def _compute_l3_member_set(
    l1_norm: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
) -> Set[str]:
    """Compute L3 membership: papers that are NOT L1 seeds and NOT L2 citers."""
    l1_set = set(l1_norm)
    l2_set: Set[str] = set()
    for citing, cited in all_edges.items():
        if cited & l1_set:
            l2_set.add(citing)
    return set(all_papers.keys()) - l1_set - l2_set


def _scan_l3_provider_edges(
    provider_name: str,
    l3_provider_index: int,
    l3_parent_ids: List[str],
    l3_member_set: Set[str],
    max_l3: int,
    l3_to_l3_state: Dict[str, Dict[str, Any]],
    effective_staleness_seconds: int,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
    provider_fetch_func: Callable[..., Tuple[Dict[str, Set[str]], Dict[str, IngestionPaper]]],
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Dict[str, Set[str]], int]:
    """Scan one L3 provider for outgoing references. Returns retained edges and count."""
    run_state = l3_to_l3_state.get(provider_name)
    if _is_pagination_state_stale(
        run_state or {},
        max_age_seconds=effective_staleness_seconds,
    ):
        l3_to_l3_state.pop(provider_name, None)
        persist_callback()

    run_state = l3_to_l3_state.setdefault(provider_name, {})

    resumed_idx = run_state.get("next_l3_parent_index")
    if isinstance(resumed_idx, int) and resumed_idx > 0:
        resumed = checkpoint_stats.get("l3_to_l3_resumed_providers")
        if isinstance(resumed, list) and provider_name not in resumed:
            resumed.append(provider_name)

    def _l3_to_l3_progress(state: Dict[str, Any], _rs: Dict[str, Any] = run_state) -> None:
        _rs.clear()
        _rs.update(dict(state))
        if provider_progress_callback is not None:
            progress_raw = state.get("next_l3_parent_index")
            progress_index = progress_raw if isinstance(progress_raw, int) else 0
            provider_progress_callback(
                progress_index,
                len(l3_parent_ids),
                str(state.get("status") or "in_progress"),
            )
        persist_callback()

    raw_edges, raw_papers = provider_fetch_func(
        l3_paper_ids=l3_parent_ids,
        max_edges=max_l3,
        resume_state=run_state,
        progress_callback=_l3_to_l3_progress,
    )
    if provider_progress_callback is not None:
        provider_progress_callback(len(l3_parent_ids), len(l3_parent_ids), "complete")

    retained_edges: Dict[str, Set[str]] = {}
    for parent, targets in raw_edges.items():
        kept = targets & l3_member_set
        if kept:
            retained_edges[parent] = kept

    return retained_edges, sum(len(v) for v in retained_edges.values())


def _run_l2_to_l3_pass(
    depth: str,
    l1_norm: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    providers: List[CitationProvider],
    max_workers: Optional[int],
    max_l3: int,
    checkpoint_root: Path,
    key: str,
    effective_staleness_seconds: int,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
) -> None:
    """Run L2->L3 reference expansion pass for all providers."""
    if depth.lower() not in {"l2l3", "l3", "2"}:
        return

    l1_set = set(l1_norm)
    l2_parent_ids = sorted({citing for citing, cited in all_edges.items() if cited & l1_set})
    _progress(f"[ADIT] L2->L3 pass: {len(l2_parent_ids)} deduped L2 parent(s)")

    l3_providers = [
        provider for provider in providers if provider.capabilities.supports_reference_expansion
    ]

    l2_to_l3_completed, l3_jobs = _collect_l2_to_l3_jobs(
        l2_parent_ids=l2_parent_ids,
        l3_providers=l3_providers,
        provider_l3_state=provider_l3_state,
        all_papers=all_papers,
        effective_staleness_seconds=effective_staleness_seconds,
        checkpoint_stats=checkpoint_stats,
        persist_callback=persist_callback,
    )

    bars_lock = threading.Lock()
    provider_bars: Dict[str, tqdm] = {}
    provider_progress_callbacks: Dict[str, Callable[[int, int, str], None]] = {}

    global _PROVIDER_TQDM_ACTIVE
    _PROVIDER_TQDM_ACTIVE = bool(not _QUIET and _stderr_is_tty() and l3_jobs)
    if _PROVIDER_TQDM_ACTIVE:
        desc_width = _provider_tqdm_desc_width([provider.name for provider, *_ in l3_jobs], "l2->l3")
        for bar_position, (provider, _, provider_l2_parent_ids, _) in enumerate(l3_jobs):
            provider_bars[provider.name] = _create_provider_tqdm(
                provider_name=provider.name,
                phase_label="l2->l3",
                total=max(1, len(provider_l2_parent_ids)),
                position=bar_position,
                unit="parent",
                desc_width=desc_width,
            )
            provider_progress_callbacks[provider.name] = (
                lambda completed, total, status, provider_name=provider.name: _update_provider_tqdm_bar(
                    provider_bars,
                    provider_name,
                    completed,
                    total,
                    status,
                    bars_lock,
                )
            )

    try:
        _execute_l2_to_l3_jobs(
            l3_jobs=l3_jobs,
            l2_to_l3_completed=l2_to_l3_completed,
            all_edges=all_edges,
            all_papers=all_papers,
            provider_stats=provider_stats,
            provider_l3_state=provider_l3_state,
            provider_transient_failures=provider_transient_failures,
            max_workers=max_workers,
            max_l3=max_l3,
            checkpoint_root=checkpoint_root,
            key=key,
            checkpoint_stats=checkpoint_stats,
            persist_callback=persist_callback,
            provider_progress_callbacks=provider_progress_callbacks or None,
        )
    finally:
        for bar in provider_bars.values():
            bar.close()
        _PROVIDER_TQDM_ACTIVE = False


def _run_l3_to_l3_pass(
    depth: str,
    l1_norm: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    l3_to_l3_state: Dict[str, Dict[str, Any]],
    providers: List[CitationProvider],
    max_l3: int,
    checkpoint_root: Path,
    key: str,
    effective_staleness_seconds: int,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
) -> int:
    """Run L3->L3 outgoing reference discovery pass. Returns total edges added."""
    if depth.lower() not in {"l2l3", "l3", "2"}:
        return 0

    l3_member_set = _compute_l3_member_set(l1_norm, all_edges, all_papers)
    l3_parent_ids = sorted(l3_member_set)
    _progress(f"[ADIT] L3->L3 pass: {len(l3_parent_ids)} parent(s) to scan")

    l3_to_l3_completed = _load_completed_provider_names(l3_to_l3_state)

    total_l3_to_l3_edges = 0
    l3_providers = [p for p in providers if p.capabilities.supports_l3_outgoing]
    l3_provider_total = len(l3_providers)
    bars_lock = threading.Lock()
    provider_bars: Dict[str, tqdm] = {}
    provider_progress_callbacks: Dict[str, Callable[[int, int, str], None]] = {}

    global _PROVIDER_TQDM_ACTIVE
    _PROVIDER_TQDM_ACTIVE = bool(not _QUIET and _stderr_is_tty() and l3_providers)
    if _PROVIDER_TQDM_ACTIVE:
        desc_width = _provider_tqdm_desc_width([provider.name for provider in l3_providers], "l3->l3")
        for bar_position, provider in enumerate(l3_providers):
            provider_bars[provider.name] = _create_provider_tqdm(
                provider_name=provider.name,
                phase_label="l3->l3",
                total=max(1, len(l3_parent_ids)),
                position=bar_position,
                unit="parent",
                desc_width=desc_width,
            )
            provider_progress_callbacks[provider.name] = (
                lambda completed, total, status, provider_name=provider.name: _update_provider_tqdm_bar(
                    provider_bars,
                    provider_name,
                    completed,
                    total,
                    status,
                    bars_lock,
                )
            )

    try:
        for l3_provider_index, provider in enumerate(l3_providers, start=1):
            if provider.name in l3_to_l3_completed:
                if _PROVIDER_TQDM_ACTIVE:
                    _update_provider_tqdm_bar(
                        provider_bars,
                        provider.name,
                        max(1, len(l3_parent_ids)),
                        max(1, len(l3_parent_ids)),
                        "skipped",
                        bars_lock,
                    )
                _progress_done(
                    f"  [{provider.name}] L3->L3 provider {l3_provider_index}/{l3_provider_total}: "
                    "skipped (checkpoint complete)"
                )
                continue

            retained_count = _run_l3_to_l3_provider(
                provider=provider,
                l3_provider_index=l3_provider_index,
                l3_provider_total=l3_provider_total,
                l3_parent_ids=l3_parent_ids,
                l3_member_set=l3_member_set,
                all_edges=all_edges,
                l3_to_l3_completed=l3_to_l3_completed,
                l3_to_l3_state=l3_to_l3_state,
                max_l3=max_l3,
                effective_staleness_seconds=effective_staleness_seconds,
                checkpoint_stats=checkpoint_stats,
                persist_callback=persist_callback,
                provider_progress_callback=provider_progress_callbacks.get(provider.name),
            )
            total_l3_to_l3_edges += retained_count
    finally:
        for bar in provider_bars.values():
            bar.close()
        _PROVIDER_TQDM_ACTIVE = False

    checkpoint_stats["l3_to_l3_edges_added"] = total_l3_to_l3_edges
    checkpoint_stats["l3_to_l3_parent_scanned_count"] = len(l3_parent_ids)

    return total_l3_to_l3_edges


def _load_completed_provider_names(provider_state: Dict[str, Dict[str, Any]]) -> Set[str]:
    completed_names: Set[str] = set()
    raw_completed = provider_state.get("__completed_providers")
    if not isinstance(raw_completed, dict):
        return completed_names
    names = raw_completed.get("names")
    if isinstance(names, list):
        completed_names = {name for name in names if isinstance(name, str)}
    return completed_names


def _mark_provider_completed(
    provider_name: str,
    completed_names: Set[str],
    provider_state: Dict[str, Dict[str, Any]],
    persist_callback: Callable[[], None],
) -> None:
    completed_names.add(provider_name)
    provider_state["__completed_providers"] = {"names": sorted(completed_names)}
    provider_state.pop(provider_name, None)
    persist_callback()


def _run_l3_to_l3_provider(
    provider: CitationProvider,
    l3_provider_index: int,
    l3_provider_total: int,
    l3_parent_ids: List[str],
    l3_member_set: Set[str],
    all_edges: Dict[str, Set[str]],
    l3_to_l3_completed: Set[str],
    l3_to_l3_state: Dict[str, Dict[str, Any]],
    max_l3: int,
    effective_staleness_seconds: int,
    checkpoint_stats: Dict[str, object],
    persist_callback: Callable[[], None],
    provider_progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    _progress(
        f"  [{provider.name}] L3->L3 provider {l3_provider_index}/{l3_provider_total}: "
        "scanning outgoing references"
    )
    retained_edges, retained_count = _scan_l3_provider_edges(
        provider_name=provider.name,
        l3_provider_index=l3_provider_index,
        l3_parent_ids=l3_parent_ids,
        l3_member_set=l3_member_set,
        max_l3=max_l3,
        l3_to_l3_state=l3_to_l3_state,
        effective_staleness_seconds=effective_staleness_seconds,
        checkpoint_stats=checkpoint_stats,
        persist_callback=persist_callback,
        provider_fetch_func=provider.fetch_l3_outgoing_references,
        provider_progress_callback=provider_progress_callback,
    )
    if provider_progress_callback is not None:
        provider_progress_callback(len(l3_parent_ids), len(l3_parent_ids), "done")
    _progress_done(
        f"  [{provider.name}] L3->L3 provider {l3_provider_index}/{l3_provider_total}: "
        f"retained {retained_count} edges"
    )
    for parent, targets in retained_edges.items():
        all_edges.setdefault(parent, set()).update(targets)
    _mark_provider_completed(
        provider_name=provider.name,
        completed_names=l3_to_l3_completed,
        provider_state=l3_to_l3_state,
        persist_callback=persist_callback,
    )
    return retained_count


def _run_parallel_wave1_providers(
    providers: List[CitationProvider],
    completed_providers: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    combined_completeness: Dict[str, Any],
    provider_pagination_state: Dict[str, Dict[str, Any]],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    l3_to_l3_state: Dict[str, Dict[str, Any]],
    ingestion_phase: str,
    checkpoint_stats: Dict[str, object],
    l1_norm: Set[str],
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    exhaustive: bool,
    max_l2: int,
    max_l3: Optional[int],
    max_workers: int,
    checkpoint_root: Path,
    key: str,
    reset_checkpoints: bool,
    persist_callback: Callable[[], None],
    wave1_desc_width: Optional[int] = None,
) -> Tuple[Set[str], Set[str]]:
    """Run parallel wave-1 ingestion across non-crossref providers.

    Returns (parallel_completed, parallel_attempted) name sets.
    """
    parallel_completed: Set[str] = set()
    parallel_attempted: Set[str] = set()
    parallel_candidates = [
        p for p in providers if p.name != "crossref" and p.name not in completed_providers
    ]
    if not parallel_candidates:
        return parallel_completed, parallel_attempted

    provider_total = len(providers)
    provider_index_map = {p.name: idx for idx, p in enumerate(providers, start=1)}
    merge_lock = threading.Lock()
    effective_workers = min(max_workers, len(parallel_candidates))
    bars_lock = threading.Lock()
    provider_bars: Dict[str, tqdm] = {}

    global _PROVIDER_TQDM_ACTIVE
    _PROVIDER_TQDM_ACTIVE = bool(not _QUIET and _stderr_is_tty() and parallel_candidates)
    if _PROVIDER_TQDM_ACTIVE:
        desc_width = wave1_desc_width or _provider_tqdm_desc_width(
            [provider.name for provider in parallel_candidates],
            "wave-1",
        )
        for bar_position, provider in enumerate(parallel_candidates):
            total = max(1, len(l1_norm)) if provider.capabilities.supports_cited_by_traversal else 1
            unit = "seed" if provider.capabilities.supports_cited_by_traversal else "step"
            provider_bars[provider.name] = _create_provider_tqdm(
                provider_name=provider.name,
                phase_label="wave-1",
                total=total,
                position=bar_position,
                unit=unit,
                desc_width=desc_width,
            )

    try:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for provider in parallel_candidates:
                parallel_attempted.add(provider.name)
                provider_index = provider_index_map.get(provider.name, 0)
                executed_names = checkpoint_stats["executed_provider_names"]
                if isinstance(executed_names, list):
                    executed_names.append(provider.name)
                checkpoint_stats["providers_executed"] = int(checkpoint_stats["providers_executed"]) + 1
                if not _PROVIDER_TQDM_ACTIVE:
                    _progress(
                        f"[ADIT] Provider {provider_index}/{provider_total}: {provider.name} "
                        "(parallel wave-1)"
                    )

                provider_resume_state = provider_pagination_state.get(provider.name, {})
                provider_l3_resume_state = provider_l3_state.get(provider.name, {})
                provider_transient_state = provider_transient_failures.get(provider.name, {})
                provider_checkpoint = _load_provider_checkpoint_state(
                    checkpoint_root, key, provider.name, reset_checkpoints
                )
                if provider_checkpoint:
                    checkpoint_seed_state = provider_checkpoint.get("provider_pagination_state")
                    if isinstance(checkpoint_seed_state, dict):
                        provider_resume_state = checkpoint_seed_state
                    checkpoint_l3_state = provider_checkpoint.get("provider_l3_state")
                    if isinstance(checkpoint_l3_state, dict):
                        provider_l3_resume_state = checkpoint_l3_state
                    checkpoint_transient_state = provider_checkpoint.get("transient_failures")
                    if isinstance(checkpoint_transient_state, dict):
                        provider_transient_state = checkpoint_transient_state

                provider_transient_state = _deserialize_transient_failures(
                    {provider.name: provider_transient_state}
                ).get(provider.name, {})

                provider_progress_callback: Optional[Callable[[int, int, str], None]] = None
                if _PROVIDER_TQDM_ACTIVE:
                    provider_progress_callback = (
                        lambda completed, total, status, provider_name=provider.name: _update_provider_tqdm_bar(
                            provider_bars,
                            provider_name,
                            completed,
                            total,
                            status,
                            bars_lock,
                        )
                    )

                future = executor.submit(
                    _run_provider_wave1_worker,
                    provider,
                    l1_norm,
                    dict(all_papers),
                    theory_name,
                    key_constructs,
                    depth,
                    exhaustive,
                    max_l2,
                    max_l3,
                    provider_resume_state,
                    provider_l3_resume_state,
                    provider_transient_state,
                    checkpoint_root,
                    key,
                    provider_progress_callback,
                )
                futures[future] = (provider, provider_index)

            for future in as_completed(futures):
                provider, provider_index = futures[future]
                try:
                    (
                        seed_metadata,
                        provider_edges,
                        provider_papers,
                        stats,
                        local_seed_state,
                        local_l3_state,
                        local_transient_failures,
                    ) = future.result()
                except ProviderIngestionError as exc:
                    if _PROVIDER_TQDM_ACTIVE:
                        _update_provider_tqdm_bar(
                            provider_bars,
                            exc.provider_name,
                            max(1, len(l1_norm)),
                            max(1, len(l1_norm)),
                            "failed",
                            bars_lock,
                        )
                    provider_stats[exc.provider_name] = {"status": "failed", "error": str(exc.cause)}
                    _progress(
                        f"[ADIT] Provider {provider_index}/{provider_total}: "
                        f"{exc.provider_name} failed ({exc.cause}) — continuing"
                    )
                    continue

                with merge_lock:
                    _merge_seed_metadata(all_papers, seed_metadata)
                    _merge_provider_outputs(all_edges, all_papers, provider_edges, provider_papers)
                    provider_stats[provider.name] = stats
                    for l1_id, l1_completeness in stats.get("completeness", {}).items():
                        combined_completeness.setdefault(l1_id, {}).update(l1_completeness)

                    provider_pagination_state[provider.name] = dict(local_seed_state)
                    provider_l3_state[provider.name] = dict(local_l3_state)
                    provider_transient_failures[provider.name] = dict(local_transient_failures)
                    if not local_transient_failures:
                        completed_providers.add(provider.name)
                    parallel_completed.add(provider.name)
                    if _PROVIDER_TQDM_ACTIVE:
                        _update_provider_tqdm_bar(
                            provider_bars,
                            provider.name,
                            max(1, len(l1_norm)),
                            max(1, len(l1_norm)),
                            "done",
                            bars_lock,
                        )
                    if not _PROVIDER_TQDM_ACTIVE:
                        _progress_done(
                            f"[ADIT] Provider {provider_index}/{provider_total}: {provider.name} "
                            f"completed (papers={stats.get('papers', 0)}, "
                            f"l2_edges={stats.get('l2_edges', 0)}, l3_edges={stats.get('l3_edges', 0)})"
                        )
                    _write_coordinator_checkpoint_state(
                        checkpoint_root,
                        key,
                        completed_providers,
                        all_edges,
                        all_papers,
                        provider_stats,
                        combined_completeness,
                        _transient_failure_summary(provider_transient_failures),
                        l3_to_l3_state,
                        ingestion_phase,
                    )
                    persist_callback()

                    provider_pagination_state.pop(provider.name, None)
                    provider_l3_state.pop(provider.name, None)
                    if not local_transient_failures:
                        provider_transient_failures.pop(provider.name, None)
                        _write_provider_checkpoint_state(
                            checkpoint_root,
                            key,
                            provider.name,
                            provider_pagination_state={},
                            provider_l3_state={},
                            transient_failures={},
                        )
    finally:
        for bar in provider_bars.values():
            bar.close()
        _PROVIDER_TQDM_ACTIVE = False

    return parallel_completed, parallel_attempted


def _run_sequential_providers(
    providers: List[CitationProvider],
    parallel_attempted: Set[str],
    parallel_completed: Set[str],
    completed_providers: Set[str],
    all_edges: Dict[str, Set[str]],
    all_papers: Dict[str, IngestionPaper],
    provider_stats: Dict[str, Any],
    combined_completeness: Dict[str, Any],
    provider_pagination_state: Dict[str, Dict[str, Any]],
    provider_l3_state: Dict[str, Dict[str, Any]],
    provider_transient_failures: Dict[str, Dict[str, Dict[str, Any]]],
    checkpoint_stats: Dict[str, object],
    l1_norm: Set[str],
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    exhaustive: bool,
    max_l2: int,
    max_l3: Optional[int],
    checkpoint_root: Path,
    key: str,
    effective_staleness_seconds: int,
    persist_callback: Callable[[], None],
    wave1_desc_width: Optional[int] = None,
) -> None:
    """Run sequential wave ingestion for providers not handled in parallel."""
    pending_providers = [
        provider
        for provider in providers
        if provider.name not in parallel_attempted and provider.name not in parallel_completed
    ]
    bars_lock = threading.Lock()
    provider_bars: Dict[str, tqdm] = {}

    global _PROVIDER_TQDM_ACTIVE
    _PROVIDER_TQDM_ACTIVE = bool(not _QUIET and _stderr_is_tty() and pending_providers)
    if _PROVIDER_TQDM_ACTIVE:
        desc_width = wave1_desc_width or _provider_tqdm_desc_width(
            [provider.name for provider in pending_providers],
            "wave-1",
        )
        for bar_position, provider in enumerate(pending_providers):
            total = max(1, len(l1_norm)) if provider.capabilities.supports_cited_by_traversal else 1
            unit = "seed" if provider.capabilities.supports_cited_by_traversal else "step"
            provider_bars[provider.name] = _create_provider_tqdm(
                provider_name=provider.name,
                phase_label="wave-1",
                total=total,
                position=bar_position,
                unit=unit,
                desc_width=desc_width,
            )

    provider_total = len(providers)
    try:
        for provider_index, provider in enumerate(providers, start=1):
            if provider.name in parallel_attempted or provider.name in parallel_completed:
                continue
            if provider.name in completed_providers:
                if not _PROVIDER_TQDM_ACTIVE:
                    _progress_done(
                        f"[ADIT] Provider {provider_index}/{provider_total}: "
                        f"{provider.name} skipped (checkpoint complete)"
                    )
                if _PROVIDER_TQDM_ACTIVE:
                    _update_provider_tqdm_bar(
                        provider_bars,
                        provider.name,
                        max(1, len(l1_norm)),
                        max(1, len(l1_norm)),
                        "skipped",
                        bars_lock,
                    )
                skipped_names = checkpoint_stats["skipped_provider_names"]
                if isinstance(skipped_names, list):
                    skipped_names.append(provider.name)
                checkpoint_stats["providers_skipped"] = int(checkpoint_stats["providers_skipped"]) + 1
                continue

            executed_names = checkpoint_stats["executed_provider_names"]
            if isinstance(executed_names, list):
                executed_names.append(provider.name)
            checkpoint_stats["providers_executed"] = int(checkpoint_stats["providers_executed"]) + 1

            if not _PROVIDER_TQDM_ACTIVE:
                _progress(f"[ADIT] Provider {provider_index}/{provider_total}: {provider.name}")
            _merge_seed_metadata(all_papers, provider.fetch_seed_metadata(l1_norm))

            if provider.name == "crossref":
                targets = _crossref_enrichment_targets(list(all_edges.keys()), all_papers)
                crossref_papers = provider.fetch_seed_metadata(targets) if targets else {}
                _merge_provider_outputs(all_edges, all_papers, {}, crossref_papers)
                provider_stats[provider.name] = {
                    "l2_nodes": 0,
                    "l2_edges": 0,
                    "l3_edges": 0,
                    "papers": 0,
                    "metadata_enriched": len(crossref_papers),
                    "completeness": {},
                }
                if _PROVIDER_TQDM_ACTIVE:
                    _update_provider_tqdm_bar(
                        provider_bars,
                        provider.name,
                        1,
                        1,
                        "done",
                        bars_lock,
                    )
                if not _PROVIDER_TQDM_ACTIVE:
                    _progress_done(
                        f"[ADIT] Provider {provider_index}/{provider_total}: {provider.name} completed "
                        f"(enriched={len(crossref_papers)})"
                    )
                completed_providers.add(provider.name)
                provider_pagination_state.pop(provider.name, None)
                persist_callback()
                continue

            provider_state = provider_pagination_state.setdefault(provider.name, {})
            provider_transient_state = provider_transient_failures.setdefault(provider.name, {})
            _prune_provider_transient_failures(
                provider.name, provider_transient_state, checkpoint_stats
            )
            _replay_provider_transient_failures(
                provider.name, provider_transient_state, checkpoint_stats
            )
            if _drop_stale_seed_resume_state(
                provider_name=provider.name,
                provider_state=provider_state,
                checkpoint_stats=checkpoint_stats,
                max_age_seconds=effective_staleness_seconds,
            ):
                persist_callback()
            if _drop_stale_provider_l3_state(
                provider_name=provider.name,
                provider_l3_state=provider_l3_state,
                checkpoint_stats=checkpoint_stats,
                max_age_seconds=effective_staleness_seconds,
            ):
                persist_callback()

            provider_l3_run_state = provider_l3_state.setdefault(provider.name, {})
            _record_l3_resume_progress(
                provider_name=provider.name,
                provider_l3_run_state=provider_l3_run_state,
                checkpoint_stats=checkpoint_stats,
            )

            _seed_progress, _l3_progress = _build_provider_progress_callbacks(
                provider_state=provider_state,
                provider_l3_run_state=provider_l3_run_state,
                persist_callback=persist_callback,
            )

            provider_progress_callback: Optional[Callable[[int, int, str], None]] = None
            if _PROVIDER_TQDM_ACTIVE:
                provider_progress_callback = (
                    lambda completed, total, status, provider_name=provider.name: _update_provider_tqdm_bar(
                        provider_bars,
                        provider_name,
                        completed,
                        total,
                        status,
                        bars_lock,
                    )
                )

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
                provider_seed_state=provider_state,
                seed_progress_callback=_seed_progress,
                provider_progress_callback=provider_progress_callback,
                provider_l3_state=provider_l3_run_state,
                l3_progress_callback=_l3_progress,
                include_l3=False,
            )
            _merge_provider_outputs(all_edges, all_papers, provider_edges, provider_papers)
            provider_stats[provider.name] = stats
            if _PROVIDER_TQDM_ACTIVE:
                _update_provider_tqdm_bar(
                    provider_bars,
                    provider.name,
                    max(1, len(l1_norm)),
                    max(1, len(l1_norm)),
                    "done",
                    bars_lock,
                )
            if not _PROVIDER_TQDM_ACTIVE:
                _progress_done(
                    f"[ADIT] Provider {provider_index}/{provider_total}: {provider.name} completed "
                    f"(papers={stats.get('papers', 0)}, "
                    f"l2_edges={stats.get('l2_edges', 0)}, l3_edges={stats.get('l3_edges', 0)})"
                )
            for l1_id, l1_completeness in stats.get("completeness", {}).items():
                combined_completeness.setdefault(l1_id, {}).update(l1_completeness)

            drained_failures = _drain_transient_request_failures(provider.name)
            _merge_provider_transient_failures(provider_transient_state, drained_failures)
            _prune_provider_transient_failures(
                provider.name, provider_transient_state, checkpoint_stats
            )

            if provider_transient_state:
                checkpoint_stats["transient_failures_queued"] = int(
                    checkpoint_stats.get("transient_failures_queued", 0)
                ) + len(provider_transient_state)
                _vprint(
                    f"  [{provider.name}] Deferred transient retries queued: {len(provider_transient_state)}"
                )
            else:
                completed_providers.add(provider.name)
                provider_transient_failures.pop(provider.name, None)
            provider_pagination_state.pop(provider.name, None)
            persist_callback()
    finally:
        for bar in provider_bars.values():
            bar.close()
        _PROVIDER_TQDM_ACTIVE = False


def ingest_from_internet(
    theory_name: str,
    l1_papers: Sequence[str],
    key_constructs: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    depth: str = "l2l3",
    cache_dir: Optional[Path] = None,
    refresh: bool = False,
    max_l2: int = 200,
    max_l3: Optional[int] = None,
    exhaustive: bool = True,
    verbose: bool = False,
    quiet: bool = False,
    debug_http: bool = False,
    max_workers: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    reset_checkpoints: bool = False,
    checkpoint_staleness_seconds: Optional[int] = None,
    transient_retry_max_attempts: Optional[int] = None,
    transient_retry_max_age_seconds: Optional[int] = None,
) -> IngestionResult:
    """Ingest citation data from internet providers.

    Args:
        exhaustive: When ``True`` (default), providers that support cited-by
            traversal paginate until all citers are fetched.  When ``False``,
            retrieval is capped at *max_l2* results per L1/provider.
        verbose: When ``True``, print live progress messages to stderr
            (per-seed status, retry countdowns).
        quiet: When ``True``, suppress standard and verbose progress messages.
        debug_http: When ``True``, include failed-response bodies in diagnostics.
        checkpoint_staleness_seconds: Optional staleness window in seconds for
            checkpoint resume state. When unset, defaults to
            ``_PAGINATION_STATE_MAX_AGE_SECONDS``.
        transient_retry_max_attempts: Optional cap on resume retry attempts
            per transient record before it is pruned.
        transient_retry_max_age_seconds: Optional max age in seconds for
            transient retry records before pruning.
    """
    global _TRANSIENT_RETRY_MAX_ATTEMPTS, _TRANSIENT_RETRY_MAX_AGE_SECONDS

    if checkpoint_staleness_seconds is not None and checkpoint_staleness_seconds <= 0:
        raise ValueError("checkpoint_staleness_seconds must be a positive integer")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    if transient_retry_max_attempts is not None and transient_retry_max_attempts <= 0:
        raise ValueError("transient_retry_max_attempts must be a positive integer")
    if transient_retry_max_age_seconds is not None and transient_retry_max_age_seconds <= 0:
        raise ValueError("transient_retry_max_age_seconds must be a positive integer")

    _TRANSIENT_RETRY_MAX_ATTEMPTS = (
        int(transient_retry_max_attempts)
        if transient_retry_max_attempts is not None
        else _DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPTS
    )
    _TRANSIENT_RETRY_MAX_AGE_SECONDS = (
        int(transient_retry_max_age_seconds)
        if transient_retry_max_age_seconds is not None
        else _DEFAULT_TRANSIENT_RETRY_MAX_AGE_SECONDS
    )

    set_verbose(verbose)
    set_quiet(quiet)
    set_debug_http(debug_http)
    source_list = (
        ["openalex", "semantic_scholar", "crossref", "core"] if not sources else list(sources)
    )
    _reset_ingest_stats(source_list)
    providers = build_providers(source_list)
    _progress(
        f"[ADIT] Ingesting theory '{theory_name}': "
        f"{len(l1_papers)} seed(s), {len(providers)} provider(s)"
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

    checkpoint_stats: Dict[str, object] = _default_checkpoint_stats()

    cached = _load_cached_result(cache_root, key, refresh)
    if cached:
        _progress_done(
            "[ADIT] Loaded cached ingestion result: "
            f"{cached.metadata.get('paper_count', 0)} papers, "
            f"{cached.metadata.get('edge_count', 0)} edges"
        )
        return _cached_result_with_checkpoint_stats(cached)

    (
        l1_norm,
        all_papers,
        all_edges,
        provider_stats,
        combined_completeness,
        completed_providers,
        provider_pagination_state,
        provider_l3_state,
        provider_transient_failures,
        l3_to_l3_state,
        ingestion_phase,
    ) = _initialize_runtime_state(l1_papers)
    seed_papers = {paper_id: all_papers[paper_id] for paper_id in l1_norm if paper_id in all_papers}

    checkpoint_root = checkpoint_dir or (cache_root / "checkpoints")
    effective_staleness_seconds = (
        checkpoint_staleness_seconds
        if checkpoint_staleness_seconds is not None
        else _PAGINATION_STATE_MAX_AGE_SECONDS
    )
    checkpoint_state = _load_checkpoint_state(checkpoint_root, key, reset_checkpoints)
    if checkpoint_state:
        (
            all_edges,
            all_papers,
            provider_stats,
            combined_completeness,
            completed_providers,
            provider_pagination_state,
            provider_l3_state,
            provider_transient_failures,
            l3_to_l3_state,
            ingestion_phase,
        ) = _restore_runtime_state_from_checkpoint(
            checkpoint_state=checkpoint_state,
            seed_papers=seed_papers,
            checkpoint_stats=checkpoint_stats,
            all_edges=all_edges,
            all_papers=all_papers,
            provider_stats=provider_stats,
            combined_completeness=combined_completeness,
            completed_providers=completed_providers,
            provider_pagination_state=provider_pagination_state,
            provider_l3_state=provider_l3_state,
            provider_transient_failures=provider_transient_failures,
            l3_to_l3_state=l3_to_l3_state,
            ingestion_phase=ingestion_phase,
        )

    def _persist_checkpoint_snapshot() -> None:
        _write_checkpoint_state(
            checkpoint_root,
            key,
            completed_providers,
            all_edges,
            all_papers,
            provider_stats,
            combined_completeness,
            provider_pagination_state,
            provider_l3_state,
            provider_transient_failures,
            l3_to_l3_state,
            ingestion_phase,
        )

    parallel_completed: Set[str] = set()
    parallel_attempted: Set[str] = set()
    wave1_desc_width = _provider_tqdm_desc_width([provider.name for provider in providers], "wave-1")
    if max_workers is not None and max_workers > 1:
        parallel_completed, parallel_attempted = _run_parallel_wave1_providers(
            providers=providers,
            completed_providers=completed_providers,
            all_edges=all_edges,
            all_papers=all_papers,
            provider_stats=provider_stats,
            combined_completeness=combined_completeness,
            provider_pagination_state=provider_pagination_state,
            provider_l3_state=provider_l3_state,
            provider_transient_failures=provider_transient_failures,
            l3_to_l3_state=l3_to_l3_state,
            ingestion_phase=ingestion_phase,
            checkpoint_stats=checkpoint_stats,
            l1_norm=l1_norm,
            theory_name=theory_name,
            key_constructs=key_constructs,
            depth=depth,
            exhaustive=exhaustive,
            max_l2=max_l2,
            max_l3=max_l3,
            max_workers=max_workers,
            checkpoint_root=checkpoint_root,
            key=key,
            reset_checkpoints=reset_checkpoints,
            persist_callback=_persist_checkpoint_snapshot,
            wave1_desc_width=wave1_desc_width,
        )

    _run_sequential_providers(
        providers=providers,
        parallel_attempted=parallel_attempted,
        parallel_completed=parallel_completed,
        completed_providers=completed_providers,
        all_edges=all_edges,
        all_papers=all_papers,
        provider_stats=provider_stats,
        combined_completeness=combined_completeness,
        provider_pagination_state=provider_pagination_state,
        provider_l3_state=provider_l3_state,
        provider_transient_failures=provider_transient_failures,
        checkpoint_stats=checkpoint_stats,
        l1_norm=l1_norm,
        theory_name=theory_name,
        key_constructs=key_constructs,
        depth=depth,
        exhaustive=exhaustive,
        max_l2=max_l2,
        max_l3=max_l3,
        checkpoint_root=checkpoint_root,
        key=key,
        effective_staleness_seconds=effective_staleness_seconds,
        persist_callback=_persist_checkpoint_snapshot,
        wave1_desc_width=wave1_desc_width,
    )

    # Deduplicate after wave-1 so L2->L3 expands canonical L2 parents only.
    _progress("[ADIT] Wave-1 complete; deduplicating before L2->L3 expansion")
    wave1_citation_data, wave1_papers_data, wave1_alias_map = _dedupe_and_materialize(
        all_edges, all_papers
    )
    all_edges = _deserialize_edges(wave1_citation_data)
    all_papers = _deserialize_papers(wave1_papers_data)
    l1_norm = [wave1_alias_map.get(pid, pid) for pid in l1_norm]
    _persist_checkpoint_snapshot()

    _run_l2_to_l3_pass(
        depth=depth,
        l1_norm=l1_norm,
        all_edges=all_edges,
        all_papers=all_papers,
        provider_stats=provider_stats,
        provider_l3_state=provider_l3_state,
        provider_transient_failures=provider_transient_failures,
        providers=providers,
        max_workers=max_workers,
        max_l3=max_l3,
        checkpoint_root=checkpoint_root,
        key=key,
        effective_staleness_seconds=effective_staleness_seconds,
        checkpoint_stats=checkpoint_stats,
        persist_callback=_persist_checkpoint_snapshot,
    )

    # ── Second pass: L3→L3 edge discovery ──────────────────────────────
    # After the first pass completes (L2→L3), build the L3 membership set
    # and query each provider's outgoing references for L3 parents.  Only
    # edges whose target is already in the membership set are retained.
    ingestion_phase = "l3_to_l3"
    _run_l3_to_l3_pass(
        depth=depth,
        l1_norm=l1_norm,
        all_edges=all_edges,
        all_papers=all_papers,
        provider_stats=provider_stats,
        l3_to_l3_state=l3_to_l3_state,
        providers=providers,
        max_l3=max_l3,
        checkpoint_root=checkpoint_root,
        key=key,
        effective_staleness_seconds=effective_staleness_seconds,
        checkpoint_stats=checkpoint_stats,
        persist_callback=_persist_checkpoint_snapshot,
    )

    _progress("[ADIT] Deduplicating and materializing graph")
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
        _transient_failure_summary(provider_transient_failures),
        checkpoint_stats,
    )

    result_payload = {
        "citation_data": citation_data,
        "papers_data": papers_data,
        "metadata": metadata,
    }
    _write_cache(cache_root, key, result_payload)
    _emit_failure_summary_if_due(force=True)
    _progress_done(f"[ADIT] Cached ingestion result: key={key}")
    _progress_done(
        f"[ADIT] Ingestion complete: {metadata['paper_count']} papers, {metadata['edge_count']} edges"
    )

    return IngestionResult(citation_data=citation_data, papers_data=papers_data, metadata=metadata)
