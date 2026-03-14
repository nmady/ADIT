import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class ProviderCapabilities:
    supports_doi_lookup: bool
    supports_reference_expansion: bool
    supports_citation_counts: bool


@dataclass
class IngestionPaper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    keywords: str = ""
    citations: int = 0
    year: int = 2010
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


def _safe_get(url: str, timeout: int = 20) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "ADIT/0.1 ingestion"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
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
    return {
        "title": paper.title,
        "abstract": paper.abstract,
        "keywords": paper.keywords,
        "citations": int(paper.citations or 0),
        "year": int(paper.year or 2010),
    }


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
        year=int(item.get("publication_year") or 2010),
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
    return IngestionPaper(
        paper_id=paper_id,
        title=item.get("title") or "",
        year=int(item.get("year") or 2010),
        citations=int(item.get("citationCount") or 0),
        source_ids={"semantic_scholar": str(item.get("paperId", ""))},
    )


def _seed_l1_papers(l1_papers: Sequence[str]) -> tuple[List[str], Dict[str, IngestionPaper]]:
    normalized = [normalize_identifier(value) for value in l1_papers if value]
    papers = {
        paper_id: IngestionPaper(paper_id=paper_id, title=paper_id, year=2010)
        for paper_id in normalized
    }
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
    theory_name: str,
    key_constructs: Optional[Sequence[str]],
    depth: str,
    max_l2: int,
    max_l3: int,
) -> tuple[Dict[str, Set[str]], Dict[str, IngestionPaper], Dict[str, int]]:
    l2_edges, l2_papers = provider.fetch_l2_and_metadata(
        l1_papers=l1_norm,
        theory_name=theory_name,
        key_constructs=key_constructs,
        max_l2=max_l2,
    )

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

    stats = {
        "l2_nodes": len(l2_edges),
        "l2_edges": sum(len(values) for values in l2_edges.values()),
        "l3_edges": added_l3_edges,
        "papers": len(l2_papers),
    }
    return all_edges, all_papers, stats


def _request_payload(
    theory_name: str,
    l1_papers: Sequence[str],
    key_constructs: Optional[Sequence[str]],
    sources: Sequence[str],
    depth: str,
    max_l2: int,
    max_l3: int,
) -> dict:
    return {
        "theory_name": theory_name,
        "l1_papers": list(l1_papers),
        "key_constructs": list(key_constructs or []),
        "sources": list(sources),
        "depth": depth,
        "max_l2": int(max_l2),
        "max_l3": int(max_l3),
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
    provider_stats: Dict[str, Dict[str, int]],
    cache_key: str,
    alias_map: Dict[str, str],
    papers_data: Dict[str, Dict[str, object]],
    citation_data: Dict[str, List[str]],
) -> Dict[str, object]:
    return {
        "sources": list(source_list),
        "depth": depth,
        "provider_stats": provider_stats,
        "cache_key": cache_key,
        "alias_count": len(alias_map),
        "paper_count": len(papers_data),
        "edge_count": sum(len(values) for values in citation_data.values()),
    }


class OpenAlexProvider(CitationProvider):
    name = "openalex"
    capabilities = ProviderCapabilities(True, True, True)

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
        payload = _safe_get(url)
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
                papers.setdefault(ref, IngestionPaper(paper_id=ref, title=ref, year=2010))

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
            url = f"https://api.openalex.org/works/{openalex_token}"
            payload = _safe_get(url)
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
                papers.setdefault(ref_id, IngestionPaper(paper_id=ref_id, title=ref_id, year=2010))
                budget -= 1
            time.sleep(0.03)

        return edges, papers


class SemanticScholarProvider(CitationProvider):
    name = "semantic_scholar"
    capabilities = ProviderCapabilities(True, True, True)

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
        payload = _safe_get(url)
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
                papers.setdefault(ref, IngestionPaper(paper_id=ref, title=ref, year=2010))
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
                f"{token}?fields=references.paperId,references.title,references.year"
            )
            payload = _safe_get(url)
            if not payload:
                continue

            for ref in payload.get("references", []) or []:
                if budget <= 0:
                    break
                rid = ref.get("paperId")
                if not rid:
                    continue
                ref_id = normalize_identifier(rid, source=self.name)
                edges.setdefault(pid, set()).add(ref_id)
                papers.setdefault(
                    ref_id,
                    IngestionPaper(
                        paper_id=ref_id,
                        title=ref.get("title") or ref_id,
                        year=int(ref.get("year") or 2010),
                    ),
                )
                budget -= 1
            time.sleep(0.05)

        return edges, papers


class CrossrefProvider(CitationProvider):
    name = "crossref"
    capabilities = ProviderCapabilities(True, False, False)

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
        payload = _safe_get(url)
        if not payload:
            return edges, papers

        for item in payload.get("message", {}).get("items", []):
            doi = item.get("DOI")
            if not doi:
                continue
            paper_id = normalize_identifier(doi)
            titles = item.get("title") or []
            title = titles[0] if titles else ""
            issued = item.get("issued", {}).get("date-parts", [[2010]])
            year = issued[0][0] if issued and issued[0] else 2010
            citations = int(item.get("is-referenced-by-count") or 0)
            papers[paper_id] = IngestionPaper(
                paper_id=paper_id,
                title=title,
                year=int(year or 2010),
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
) -> IngestionResult:
    source_list = ["openalex", "semantic_scholar", "crossref"] if not sources else list(sources)
    providers = build_providers(source_list)

    cache_root = cache_dir or Path(".cache") / "adit_ingestion"
    request_payload = _request_payload(
        theory_name,
        l1_papers,
        key_constructs,
        source_list,
        depth,
        max_l2,
        max_l3,
    )
    key = _cache_key(request_payload)

    cached = _load_cached_result(cache_root, key, refresh)
    if cached:
        return cached

    all_edges: Dict[str, Set[str]] = {}
    l1_norm, all_papers = _seed_l1_papers(l1_papers)
    provider_stats: Dict[str, Dict[str, int]] = {}

    for provider in providers:
        provider_edges, provider_papers, stats = _fetch_provider_graph(
            provider=provider,
            l1_norm=l1_norm,
            theory_name=theory_name,
            key_constructs=key_constructs,
            depth=depth,
            max_l2=max_l2,
            max_l3=max_l3,
        )
        _merge_provider_outputs(all_edges, all_papers, provider_edges, provider_papers)
        provider_stats[provider.name] = stats

    citation_data, papers_data, alias_map = _dedupe_and_materialize(all_edges, all_papers)

    metadata = _build_metadata(
        source_list,
        depth,
        provider_stats,
        key,
        alias_map,
        papers_data,
        citation_data,
    )

    result_payload = {
        "citation_data": citation_data,
        "papers_data": papers_data,
        "metadata": metadata,
    }
    _write_cache(cache_root, key, result_payload)

    return IngestionResult(citation_data=citation_data, papers_data=papers_data, metadata=metadata)
