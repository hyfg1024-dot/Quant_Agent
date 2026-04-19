from __future__ import annotations

from shared.inflection_detector import InflectionDetector


def test_revenue_inflection_trigger() -> None:
    detector = InflectionDetector()
    result = detector.detect_revenue_inflection([100, 95, 90, 92, 98])
    assert result["triggered"] is True


def test_gross_margin_improvement_trigger() -> None:
    detector = InflectionDetector()
    result = detector.detect_margin_improvement([30, 28, 29, 31, 33])
    assert result["triggered"] is True


def test_none_inputs_do_not_crash() -> None:
    detector = InflectionDetector()
    revenue_result = detector.detect_revenue_inflection([None, None, None, None, None])
    margin_result = detector.detect_margin_improvement([None, None, None, None, None])

    assert isinstance(revenue_result, dict)
    assert isinstance(margin_result, dict)
    assert revenue_result["triggered"] is False
    assert margin_result["triggered"] is False
