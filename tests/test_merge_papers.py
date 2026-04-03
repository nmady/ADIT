"""Unit tests for the _merge_papers helper."""

import pytest

from citation_ingestion import IngestionPaper, _merge_papers


def _paper(**kwargs) -> IngestionPaper:
    """Construct an IngestionPaper with sensible defaults for testing."""
    defaults = dict(
        paper_id="pid",
        title="",
        abstract="",
        keywords="",
        citations=0,
        year=None,
        doi=None,
        source_ids={},
    )
    defaults.update(kwargs)
    return IngestionPaper(**defaults)


# ---------------------------------------------------------------------------
# Non-mutation (pure function)
# ---------------------------------------------------------------------------


class TestPureFunction:
    def test_returns_new_object(self):
        a = _paper(paper_id="a", title="Short")
        b = _paper(paper_id="a", title="A much longer title")
        result = _merge_papers(a, b)
        assert result is not a
        assert result is not b

    def test_does_not_mutate_existing(self):
        a = _paper(paper_id="a", title="Short", citations=5)
        b = _paper(paper_id="a", title="A longer title", citations=10)
        _merge_papers(a, b)
        assert a.title == "Short"
        assert a.citations == 5

    def test_does_not_mutate_incoming(self):
        a = _paper(paper_id="a", year=2020)
        b = _paper(paper_id="a", year=2019)
        _merge_papers(a, b)
        assert b.year == 2019


# ---------------------------------------------------------------------------
# paper_id identity
# ---------------------------------------------------------------------------


class TestPaperIdIdentity:
    def test_existing_paper_id_always_wins(self):
        a = _paper(paper_id="doi:10.1/foo")
        b = _paper(paper_id="openalex:W99")
        result = _merge_papers(a, b)
        assert result.paper_id == "doi:10.1/foo"

    def test_same_paper_id_preserved(self):
        a = _paper(paper_id="x")
        b = _paper(paper_id="x")
        result = _merge_papers(a, b)
        assert result.paper_id == "x"


# ---------------------------------------------------------------------------
# Title / abstract: prefer longer
# ---------------------------------------------------------------------------


class TestTitleAbstract:
    def test_longer_incoming_title_wins(self):
        a = _paper(title="Short")
        b = _paper(title="A much longer title here")
        assert _merge_papers(a, b).title == "A much longer title here"

    def test_shorter_incoming_title_loses(self):
        a = _paper(title="A much longer title here")
        b = _paper(title="Short")
        assert _merge_papers(a, b).title == "A much longer title here"

    def test_empty_incoming_title_keeps_existing(self):
        a = _paper(title="Existing title")
        b = _paper(title="")
        assert _merge_papers(a, b).title == "Existing title"

    def test_longer_incoming_abstract_wins(self):
        a = _paper(abstract="Short abstract.")
        b = _paper(abstract="Much longer and more detailed abstract text.")
        assert _merge_papers(a, b).abstract == "Much longer and more detailed abstract text."


# ---------------------------------------------------------------------------
# Keywords: union merge (not length comparison)
# ---------------------------------------------------------------------------


class TestKeywords:
    def test_keywords_are_union_merged(self):
        a = _paper(keywords="NLP, IR")
        b = _paper(keywords="machine learning, NLP")
        result = _merge_papers(a, b)
        kws = {k.strip() for k in result.keywords.split(",")}
        assert kws == {"NLP", "IR", "machine learning"}

    def test_shorter_incoming_not_discarded(self):
        """A shorter incoming keyword string must not be dropped wholesale."""
        a = _paper(keywords="natural_language_processing")
        b = _paper(keywords="NLP, IR")
        result = _merge_papers(a, b)
        kws = {k.strip() for k in result.keywords.split(",")}
        assert "NLP" in kws
        assert "IR" in kws
        assert "natural_language_processing" in kws

    def test_empty_existing_takes_incoming(self):
        a = _paper(keywords="")
        b = _paper(keywords="deep learning")
        assert _merge_papers(a, b).keywords == "deep learning"

    def test_empty_incoming_keeps_existing(self):
        a = _paper(keywords="deep learning")
        b = _paper(keywords="")
        assert _merge_papers(a, b).keywords == "deep learning"

    def test_both_empty_stays_empty(self):
        a = _paper(keywords="")
        b = _paper(keywords="")
        assert _merge_papers(a, b).keywords == ""


# ---------------------------------------------------------------------------
# Citations: take the higher count
# ---------------------------------------------------------------------------


class TestCitations:
    def test_higher_incoming_wins(self):
        a = _paper(citations=5)
        b = _paper(citations=20)
        assert _merge_papers(a, b).citations == 20

    def test_higher_existing_wins(self):
        a = _paper(citations=20)
        b = _paper(citations=5)
        assert _merge_papers(a, b).citations == 20

    def test_equal_counts_preserved(self):
        a = _paper(citations=7)
        b = _paper(citations=7)
        assert _merge_papers(a, b).citations == 7


# ---------------------------------------------------------------------------
# Year: no magic sentinel — accept incoming only when existing is falsy
# ---------------------------------------------------------------------------


class TestYear:
    def test_existing_year_preserved(self):
        a = _paper(year=2015)
        b = _paper(year=2019)
        assert _merge_papers(a, b).year == 2015

    def test_missing_existing_year_filled_by_incoming(self):
        a = _paper(year=None)
        b = _paper(year=2018)
        assert _merge_papers(a, b).year == 2018

    def test_year_2010_not_overwritten_by_incoming(self):
        """2010 is a valid publication year and must NOT be treated as a sentinel."""
        a = _paper(year=2010)
        b = _paper(year=2021)
        assert _merge_papers(a, b).year == 2010

    def test_both_none_stays_none(self):
        a = _paper(year=None)
        b = _paper(year=None)
        assert _merge_papers(a, b).year is None


# ---------------------------------------------------------------------------
# DOI: prefer existing, fall back to incoming
# ---------------------------------------------------------------------------


class TestDoi:
    def test_existing_doi_preserved(self):
        a = _paper(doi="10.1/a")
        b = _paper(doi="10.1/b")
        assert _merge_papers(a, b).doi == "10.1/a"

    def test_missing_doi_filled_from_incoming(self):
        a = _paper(doi=None)
        b = _paper(doi="10.1/b")
        assert _merge_papers(a, b).doi == "10.1/b"

    def test_both_none_stays_none(self):
        a = _paper(doi=None)
        b = _paper(doi=None)
        assert _merge_papers(a, b).doi is None


# ---------------------------------------------------------------------------
# source_ids: union
# ---------------------------------------------------------------------------


class TestSourceIds:
    def test_source_ids_merged(self):
        a = _paper(source_ids={"openalex": "W1"})
        b = _paper(source_ids={"semantic_scholar": "S2"})
        result = _merge_papers(a, b)
        assert result.source_ids == {"openalex": "W1", "semantic_scholar": "S2"}

    def test_incoming_source_id_overwrites_on_conflict(self):
        """incoming wins for duplicate keys (dict merge semantics)."""
        a = _paper(source_ids={"openalex": "W1"})
        b = _paper(source_ids={"openalex": "W2"})
        result = _merge_papers(a, b)
        assert result.source_ids["openalex"] == "W2"

    def test_source_ids_not_shared_with_originals(self):
        """Mutating the result's source_ids must not affect the originals."""
        a = _paper(source_ids={"openalex": "W1"})
        b = _paper(source_ids={"semantic_scholar": "S2"})
        result = _merge_papers(a, b)
        result.source_ids["new"] = "X"
        assert "new" not in a.source_ids
        assert "new" not in b.source_ids
