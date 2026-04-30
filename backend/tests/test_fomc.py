"""Test del modulo FOMC: fetcher parsing + analyzer logic + service trend."""
from datetime import date, datetime

import pytest

from app.services.fomc.analyzer import (
    FOMCAnalysis,
    _safe_parse_json,
    _truncate_text,
    _validate_analysis,
)
from app.services.fomc.fetcher import FOMCDocument, _classify_doc, _strip_html
from app.services.fomc.service import _classify_trend


class TestFetcherClassification:
    def test_classify_statement(self):
        assert _classify_doc("Federal Reserve issues FOMC statement") == "statement"
        assert _classify_doc("Fed issues FOMC statement Q4") == "statement"

    def test_classify_minutes(self):
        assert _classify_doc("Minutes of the Federal Open Market Committee, March 2026") == "minutes"

    def test_classify_other(self):
        assert _classify_doc("Speech by Chair Powell on banking") == "other"
        assert _classify_doc("Beige Book") == "other"


class TestStripHtml:
    def test_removes_tags(self):
        out = _strip_html("<p>Hello <b>world</b></p>")
        assert out == "Hello world"

    def test_removes_script(self):
        out = _strip_html("<script>evil()</script>Real text<p>more</p>")
        assert "evil" not in out
        assert "Real text" in out

    def test_decodes_entities(self):
        out = _strip_html("Federal&nbsp;Reserve&amp;Co")
        assert "Federal Reserve&Co" in out or "Federal Reserve" in out


class TestJSONParser:
    def test_strips_markdown_fences(self):
        raw = '```json\n{"hawkish_dovish_score": 0.5}\n```'
        parsed = _safe_parse_json(raw)
        assert parsed == {"hawkish_dovish_score": 0.5}

    def test_extracts_first_json_block(self):
        raw = "Some preamble {\"score\": 0.3} trailing text"
        parsed = _safe_parse_json(raw)
        assert parsed == {"score": 0.3}

    def test_invalid_returns_none(self):
        assert _safe_parse_json("not json at all") is None
        assert _safe_parse_json("") is None


class TestValidateAnalysis:
    def _doc(self) -> FOMCDocument:
        return FOMCDocument(
            doc_type="statement",
            title="FOMC statement",
            published_date=date(2026, 3, 18),
            url="https://example.com/x",
            text="..." * 1000,
            fetched_at=datetime.now(),
        )

    def test_clamps_score_in_range(self):
        analysis = _validate_analysis(
            {"hawkish_dovish_score": 5.0, "confidence": 0.9, "summary": "x"},
            self._doc(), "groq",
        )
        assert analysis.hawkish_dovish_score == 1.0

    def test_clamps_negative(self):
        analysis = _validate_analysis(
            {"hawkish_dovish_score": -10.0, "confidence": 0.5},
            self._doc(), "groq",
        )
        assert analysis.hawkish_dovish_score == -1.0

    def test_filters_topics_to_strings(self):
        analysis = _validate_analysis(
            {"key_topics": ["inflation", "labor", 42, None]},
            self._doc(), "groq",
        )
        assert "inflation" in analysis.key_topics
        # 42 e None convertiti in stringa
        assert all(isinstance(t, str) for t in analysis.key_topics)

    def test_regime_implication_clamped(self):
        analysis = _validate_analysis({
            "regime_implication": {
                "reflation": 5.0,
                "stagflation": -2.0,
                "deflation": 0.1,
                "goldilocks": -0.05,
            }
        }, self._doc(), "groq")
        assert analysis.regime_implication["reflation"] == 0.2
        assert analysis.regime_implication["stagflation"] == -0.2
        assert analysis.regime_implication["deflation"] == 0.1
        assert analysis.regime_implication["goldilocks"] == -0.05

    def test_default_regime_implication_zero(self):
        analysis = _validate_analysis({}, self._doc(), "groq")
        for r in ("reflation", "stagflation", "deflation", "goldilocks"):
            assert analysis.regime_implication[r] == 0.0


class TestTrendClassification:
    def test_insufficient_for_short_series(self):
        assert _classify_trend([0.5]) == "insufficient"
        assert _classify_trend([]) == "insufficient"

    def test_stable_when_small_delta(self):
        assert _classify_trend([0.10, 0.05, 0.12]) == "stable"

    def test_hawkening_when_score_increases(self):
        assert _classify_trend([-0.3, 0.0, 0.4]) == "hawkening"

    def test_dovening_when_score_decreases(self):
        assert _classify_trend([0.5, 0.2, -0.2]) == "dovening"


class TestTruncateText:
    def test_truncates_long_text(self):
        long = "x" * 50000
        out = _truncate_text(long, max_chars=10000)
        assert len(out) == 10000

    def test_short_text_unchanged(self):
        out = _truncate_text("hello world", max_chars=1000)
        assert out == "hello world"
