"""Unit tests for the _merge_papers helper."""

import logging
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


# ---------------------------------------------------------------------------
# Logging: uncertain merges
# ---------------------------------------------------------------------------


class TestMergeLogging:
    """Verify that uncertain overwrites are surfaced via logger.debug."""

    def test_title_overwrite_is_logged(self, caplog):
        a = _paper(paper_id="p1", title="Original title")
        b = _paper(paper_id="p1", title="A longer replacement title")
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert any("title overwritten" in r.message for r in caplog.records)

    def test_no_title_log_when_existing_is_empty(self, caplog):
        """Filling an empty title from incoming is expected — not a warning."""
        a = _paper(paper_id="p1", title="")
        b = _paper(paper_id="p1", title="New title")
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("title overwritten" in r.message for r in caplog.records)

    def test_no_title_log_when_existing_wins(self, caplog):
        """When existing title is kept (it's longer), nothing is discarded."""
        a = _paper(paper_id="p1", title="A longer existing title")
        b = _paper(paper_id="p1", title="Short")
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("title overwritten" in r.message for r in caplog.records)

    def test_abstract_overwrite_is_logged(self, caplog):
        a = _paper(paper_id="p2", abstract="Short abstract.")
        b = _paper(paper_id="p2", abstract="A much longer abstract with more detail.")
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert any("abstract overwritten" in r.message for r in caplog.records)

    def test_no_abstract_log_when_existing_is_empty(self, caplog):
        a = _paper(paper_id="p2", abstract="")
        b = _paper(paper_id="p2", abstract="New abstract.")
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("abstract overwritten" in r.message for r in caplog.records)

    def test_citation_conflict_is_logged(self, caplog):
        a = _paper(paper_id="p3", citations=10)
        b = _paper(paper_id="p3", citations=25)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert any("citation count conflict" in r.message for r in caplog.records)

    def test_no_citation_log_when_counts_are_equal(self, caplog):
        a = _paper(paper_id="p3", citations=15)
        b = _paper(paper_id="p3", citations=15)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("citation count conflict" in r.message for r in caplog.records)

    def test_no_citation_log_when_existing_is_zero(self, caplog):
        """Zero citations means no prior data — filling it is unambiguous."""
        a = _paper(paper_id="p3", citations=0)
        b = _paper(paper_id="p3", citations=12)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("citation count conflict" in r.message for r in caplog.records)

    def test_year_conflict_is_logged(self, caplog):
        a = _paper(paper_id="p4", year=2015)
        b = _paper(paper_id="p4", year=2019)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert any("year conflict" in r.message for r in caplog.records)

    def test_no_year_log_when_existing_is_none(self, caplog):
        """Filling a missing year from incoming is unambiguous — not a conflict."""
        a = _paper(paper_id="p4", year=None)
        b = _paper(paper_id="p4", year=2018)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("year conflict" in r.message for r in caplog.records)

    def test_no_year_log_when_years_match(self, caplog):
        a = _paper(paper_id="p4", year=2020)
        b = _paper(paper_id="p4", year=2020)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        assert not any("year conflict" in r.message for r in caplog.records)

    def test_log_messages_include_paper_id(self, caplog):
        """paper_id should appear in every log message for traceability."""
        a = _paper(paper_id="doi:10.1/test", title="Short", year=2010, citations=5)
        b = _paper(paper_id="doi:10.1/test", title="A longer replacement title", year=2020, citations=50)
        with caplog.at_level(logging.DEBUG, logger="citation_ingestion"):
            _merge_papers(a, b)
        for record in caplog.records:
            assert "doi:10.1/test" in record.message
