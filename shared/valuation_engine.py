from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DCFConfig:
    """DCF 参数配置。"""

    wacc: float = 0.10
    terminal_growth: float = 0.03
    projection_years: int = 5


class ValuationEngine:
    """价值投资估值引擎。

    提供 DCF、DDM、FCF 收益率与分红可持续性分析。
    所有函数都做了缺失值防御：遇到关键输入缺失时返回 None，而不是抛异常。
    """

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        """将输入转为 float；无效值返回 None。"""
        if value is None:
            return None
        if isinstance(value, (int, float, np.floating, np.integer)):
            if np.isnan(value):
                return None
            return float(value)
        text = str(value).strip().replace(",", "")
        if text in {"", "--", "-", "None", "nan", "NaN"}:
            return None
        try:
            return float(text)
        except Exception:
            return None

    @classmethod
    def _normalize_rate(cls, rate: Any, default: float = 0.0) -> float:
        """归一化增速/收益率输入。

        - 传入 12 表示 12%，会转成 0.12
        - 传入 0.12 保持不变
        - 缺失时返回 default
        """
        val = cls._to_float(rate)
        if val is None:
            return default
        if abs(val) > 1.0:
            val = val / 100.0
        return float(val)

    @classmethod
    def _estimate_total_shares(cls, data: Dict[str, Any], current_price: Optional[float]) -> Optional[float]:
        """估算总股本。

        优先 `total_shares`；否则用 `total_mv/current_price` 反推。
        """
        total_shares = cls._to_float(data.get("total_shares"))
        if total_shares and total_shares > 0:
            return total_shares

        total_mv = cls._to_float(data.get("total_mv"))
        if total_mv is None or total_mv <= 0 or current_price is None or current_price <= 0:
            return None
        return total_mv / current_price

    @classmethod
    def _estimate_eps(cls, data: Dict[str, Any], current_price: Optional[float]) -> Optional[float]:
        """估算 EPS：优先直接输入，否则使用价格/PE 反推。"""
        eps = cls._to_float(data.get("eps"))
        if eps is not None:
            return eps

        pe_ttm = cls._to_float(data.get("pe_ttm"))
        if pe_ttm is None or pe_ttm <= 0 or current_price is None or current_price <= 0:
            return None
        return current_price / pe_ttm

    @classmethod
    def _estimate_dividend_per_share(cls, data: Dict[str, Any], current_price: Optional[float]) -> Optional[float]:
        """估算每股分红。

        优先 `dividend_per_share`；否则用 `dividend_yield * current_price` 反推。
        """
        dps = cls._to_float(data.get("dividend_per_share"))
        if dps is not None:
            return dps

        dy = cls._to_float(data.get("dividend_yield"))
        if dy is None or current_price is None or current_price <= 0:
            return None
        if dy > 1.0:
            dy = dy / 100.0
        return current_price * dy

    @classmethod
    def _terminal_value(cls, final_cashflow: float, wacc: float, terminal_growth: float) -> Optional[float]:
        """计算终值（Gordon 增长）。"""
        if wacc <= terminal_growth:
            return None
        return final_cashflow * (1.0 + terminal_growth) / (wacc - terminal_growth)

    @classmethod
    def dcf_valuation(
        cls,
        data: Dict[str, Any],
        wacc: float = 0.10,
        terminal_growth: float = 0.03,
        projection_years: int = 5,
    ) -> Dict[str, Any]:
        """两阶段 DCF 估值。

        参数
        ----------
        data : dict
            需要包含（可缺省部分，会自动估算）：
            - net_profit: 归母净利润
            - revenue_growth: 营收增速
            - profit_growth: 净利增速
            - ocf_per_share: 每股经营现金流
            - total_shares: 总股本
            - debt_ratio: 资产负债率（用于保守折扣）
            - current_price: 当前价格（可选，用于安全边际）

        返回
        ----------
        dict
            {
              "intrinsic_value_per_share": float|None,
              "scenarios": {"optimistic": float|None, "neutral": float|None, "pessimistic": float|None},
              "safety_margin_pct": float|None,
              "current_price": float|None
            }
        """
        current_price = cls._to_float(data.get("current_price"))
        shares = cls._estimate_total_shares(data, current_price)
        ocf_per_share = cls._to_float(data.get("ocf_per_share"))
        net_profit = cls._to_float(data.get("net_profit"))
        debt_ratio = cls._to_float(data.get("debt_ratio"))
        if debt_ratio is not None and abs(debt_ratio) > 1.0:
            debt_ratio = debt_ratio / 100.0

        growth_profit = cls._normalize_rate(data.get("profit_growth"), default=np.nan)
        growth_revenue = cls._normalize_rate(data.get("revenue_growth"), default=0.0)
        growth = growth_profit if not np.isnan(growth_profit) else growth_revenue

        wacc_n = cls._normalize_rate(wacc, default=0.10)
        tg_n = cls._normalize_rate(terminal_growth, default=0.03)
        years = int(cls._to_float(projection_years) or 5)
        years = max(1, min(years, 15))

        # 以 OCF 为主、净利润为辅估算自由现金流基数
        base_fcf = None
        if ocf_per_share is not None and shares is not None and shares > 0:
            base_fcf = ocf_per_share * shares
        elif net_profit is not None:
            base_fcf = net_profit

        if base_fcf is None or shares is None or shares <= 0:
            return {
                "intrinsic_value_per_share": None,
                "scenarios": {"optimistic": None, "neutral": None, "pessimistic": None},
                "safety_margin_pct": None,
                "current_price": current_price,
            }

        # 负债率越高，给一个轻微保守折扣（最多 20%）
        debt_haircut = 1.0
        if debt_ratio is not None and debt_ratio > 0:
            debt_haircut = max(0.8, 1.0 - max(0.0, debt_ratio - 0.5) * 0.4)

        def _run_dcf(one_wacc: float) -> Optional[float]:
            one_wacc = max(0.03, one_wacc)
            if one_wacc <= tg_n:
                return None

            pv_sum = 0.0
            fcf = base_fcf
            for t in range(1, years + 1):
                fcf = fcf * (1.0 + growth)
                pv_sum += fcf / ((1.0 + one_wacc) ** t)

            tv = cls._terminal_value(fcf, one_wacc, tg_n)
            if tv is None:
                return None
            pv_tv = tv / ((1.0 + one_wacc) ** years)

            enterprise_value = (pv_sum + pv_tv) * debt_haircut
            return enterprise_value / shares if shares > 0 else None

        optimistic = _run_dcf(max(0.03, wacc_n - 0.02))
        neutral = _run_dcf(wacc_n)
        pessimistic = _run_dcf(wacc_n + 0.02)

        safety_margin = None
        if neutral is not None and current_price is not None and current_price > 0:
            safety_margin = (neutral - current_price) / current_price * 100.0

        return {
            "intrinsic_value_per_share": neutral,
            "scenarios": {
                "optimistic": optimistic,
                "neutral": neutral,
                "pessimistic": pessimistic,
            },
            "safety_margin_pct": safety_margin,
            "current_price": current_price,
        }

    @classmethod
    def ddm_valuation(
        cls,
        dividend_per_share: float,
        growth_rate: float,
        required_return: float = 0.10,
    ) -> Dict[str, Any]:
        """Gordon 增长模型估值。"""
        dps = cls._to_float(dividend_per_share)
        g = cls._normalize_rate(growth_rate, default=0.0)
        r = cls._normalize_rate(required_return, default=0.10)

        if dps is None or dps < 0 or r <= g:
            return {"ddm_value": None, "dividend_yield_implied": None}

        ddm_value = dps * (1.0 + g) / (r - g)
        implied_yield = (dps / ddm_value * 100.0) if ddm_value > 0 else None
        return {
            "ddm_value": ddm_value,
            "dividend_yield_implied": implied_yield,
        }

    @classmethod
    def fcf_yield(cls, ocf: float, capex: float, market_cap: float) -> Dict[str, Any]:
        """计算 FCF 收益率与质量分级。"""
        ocf_v = cls._to_float(ocf)
        capex_v = cls._to_float(capex)
        mkt = cls._to_float(market_cap)

        if ocf_v is None or capex_v is None:
            return {"fcf": None, "fcf_yield_pct": None, "fcf_quality": "差"}

        fcf = ocf_v - capex_v
        if mkt is None or mkt <= 0:
            return {"fcf": fcf, "fcf_yield_pct": None, "fcf_quality": "差"}

        y = fcf / mkt * 100.0
        if y > 8:
            q = "优秀"
        elif y > 5:
            q = "良好"
        elif y > 3:
            q = "一般"
        else:
            q = "差"

        return {
            "fcf": fcf,
            "fcf_yield_pct": y,
            "fcf_quality": q,
        }

    @classmethod
    def payout_sustainability(
        cls,
        dividend_per_share: float,
        eps: float,
        fcf_per_share: float,
    ) -> Dict[str, Any]:
        """分红可持续性评估。

        - payout_ratio = DPS / EPS
        - fcf_coverage = FCFPS / DPS
        - sustainable 判定：派息率 <= 80% 且 FCF 覆盖 >= 1.0
        """
        dps = cls._to_float(dividend_per_share)
        eps_v = cls._to_float(eps)
        fcfps = cls._to_float(fcf_per_share)

        payout_ratio = None
        if dps is not None and eps_v is not None and eps_v > 0:
            payout_ratio = dps / eps_v

        fcf_coverage = None
        if dps is not None and dps > 0 and fcfps is not None:
            fcf_coverage = fcfps / dps

        sustainable = False
        if payout_ratio is not None and fcf_coverage is not None:
            sustainable = payout_ratio <= 0.80 and fcf_coverage >= 1.0

        return {
            "payout_ratio": payout_ratio,
            "fcf_coverage": fcf_coverage,
            "sustainable": sustainable,
        }


def compute_full_valuation(fundamental_data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """整合估值方法，返回完整估值报告。

    参数
    ----------
    fundamental_data : dict
        来自基础面引擎的字段字典。
    current_price : float
        当前股价。

    返回
    ----------
    dict
        包含 dcf/ddm/fcf/payout 四大模块与关键假设。
    """
    engine = ValuationEngine()
    data = dict(fundamental_data or {})
    data["current_price"] = current_price

    price = engine._to_float(current_price)
    shares = engine._estimate_total_shares(data, price)
    eps = engine._estimate_eps(data, price)
    dps = engine._estimate_dividend_per_share(data, price)

    # OCF 与 CapEx 口径
    ocf_sum_3y = engine._to_float(data.get("ocf_sum_3y"))
    ocf_latest = (ocf_sum_3y / 3.0) if ocf_sum_3y is not None else None
    capex = engine._to_float(data.get("capex"))
    if capex is None:
        # 无 CapEx 时给保守估算：取 OCF 的 30%
        capex = (ocf_latest * 0.30) if ocf_latest is not None else None

    market_cap = engine._to_float(data.get("total_mv"))
    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares

    profit_growth = data.get("profit_growth")
    if profit_growth is None:
        profit_growth = data.get("profit_cagr_5y")

    dcf_input = {
        "net_profit": data.get("net_profit"),
        "revenue_growth": data.get("revenue_growth"),
        "profit_growth": profit_growth,
        "ocf_per_share": data.get("ocf_per_share"),
        "total_shares": shares,
        "debt_ratio": data.get("debt_ratio"),
        "total_mv": market_cap,
        "current_price": price,
    }
    dcf_report = engine.dcf_valuation(dcf_input)

    ddm_growth = data.get("profit_cagr_5y")
    if ddm_growth is None:
        ddm_growth = data.get("revenue_cagr_5y")
    ddm_report = engine.ddm_valuation(
        dividend_per_share=dps,
        growth_rate=ddm_growth,
        required_return=0.10,
    )

    fcf_report = engine.fcf_yield(ocf=ocf_latest, capex=capex, market_cap=market_cap)

    fcf_per_share = None
    fcf_val = fcf_report.get("fcf")
    if fcf_val is not None and shares is not None and shares > 0:
        fcf_per_share = fcf_val / shares

    payout_report = engine.payout_sustainability(
        dividend_per_share=dps,
        eps=eps,
        fcf_per_share=fcf_per_share,
    )

    return {
        "input_snapshot": {
            "current_price": price,
            "total_shares": shares,
            "market_cap": market_cap,
            "eps": eps,
            "dividend_per_share": dps,
            "ocf_latest_estimate": ocf_latest,
            "capex_estimate": capex,
        },
        "dcf": dcf_report,
        "ddm": ddm_report,
        "fcf": fcf_report,
        "payout": payout_report,
    }


if __name__ == "__main__":
    # 中国神华(601088)示例（公开口径近似值，仅用于演示流程）
    example_data = {
        "code": "601088",
        "name": "中国神华",
        "current_price": 43.20,
        "pe_ttm": 13.5,
        "pb": 1.75,
        "dividend_yield": 6.5,
        "roe": 13.2,
        "gross_margin": 35.0,
        "net_margin": 19.0,
        "debt_ratio": 27.0,
        "ocf_sum_3y": 260_000_000_000.0,
        "ocf_per_share": 4.35,
        "revenue_growth": 3.0,
        "profit_growth": 5.0,
        "revenue_cagr_5y": 4.0,
        "profit_cagr_5y": 6.0,
        "total_mv": 860_000_000_000.0,
        "retained_eps": 14.0,
        # 可选输入
        "total_shares": 19_900_000_000.0,
        "net_profit": 65_000_000_000.0,
        "capex": 28_000_000_000.0,
    }

    report = compute_full_valuation(example_data, current_price=example_data["current_price"])
    print("=== Valuation Report (601088 中国神华) ===")
    for k, v in report.items():
        print(f"{k}: {v}")
