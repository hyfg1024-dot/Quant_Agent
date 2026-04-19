from __future__ import annotations

import pytest

from shared.valuation_engine import ValuationEngine, compute_full_valuation


def test_dcf_valuation_fixed_input_matches_expected_value() -> None:
    data = {
        "ocf_per_share": 1.0,
        "total_shares": 100.0,
        "current_price": 10.0,
        "revenue_growth": 0.0,
        "profit_growth": 0.0,
        "debt_ratio": 0.0,
    }
    report = ValuationEngine.dcf_valuation(
        data=data,
        wacc=0.10,
        terminal_growth=0.03,
        projection_years=5,
    )

    # base_fcf = 100, growth=0, shares=100
    pv_sum = sum(100.0 / ((1.0 + 0.10) ** t) for t in range(1, 6))
    terminal_value = 100.0 * (1.0 + 0.03) / (0.10 - 0.03)
    pv_terminal = terminal_value / ((1.0 + 0.10) ** 5)
    expected = (pv_sum + pv_terminal) / 100.0

    assert report["intrinsic_value_per_share"] == pytest.approx(expected, rel=1e-6)
    assert report["scenarios"]["neutral"] == pytest.approx(expected, rel=1e-6)
    assert report["safety_margin_pct"] == pytest.approx((expected - 10.0) / 10.0 * 100.0, rel=1e-6)


def test_compute_full_valuation_with_none_inputs_is_safe() -> None:
    report = compute_full_valuation({}, current_price=None)
    assert isinstance(report, dict)
    assert report["dcf"]["intrinsic_value_per_share"] is None
    assert report["ddm"]["ddm_value"] is None
    assert report["fcf"]["fcf"] is None
    assert report["payout"]["sustainable"] is False


@pytest.mark.parametrize(
    "ocf,capex,market_cap,expected_quality",
    [
        (109.0, 9.0, 1000.0, "优秀"),  # 10%
        (106.0, 46.0, 1000.0, "良好"),  # 6%
        (104.0, 64.0, 1000.0, "一般"),  # 4%
        (102.0, 82.0, 1000.0, "差"),  # 2%
    ],
)
def test_fcf_yield_quality_grading(
    ocf: float,
    capex: float,
    market_cap: float,
    expected_quality: str,
) -> None:
    report = ValuationEngine.fcf_yield(ocf=ocf, capex=capex, market_cap=market_cap)
    assert report["fcf_quality"] == expected_quality
    assert report["fcf"] == pytest.approx(ocf - capex, rel=1e-9)
    assert report["fcf_yield_pct"] == pytest.approx((ocf - capex) / market_cap * 100.0, rel=1e-9)
