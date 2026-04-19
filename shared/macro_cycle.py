from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

import pandas as pd
try:
    from shared.cross_asset_signals import CrossAssetAnalyzer
except Exception:  # pragma: no cover
    CrossAssetAnalyzer = None  # type: ignore[assignment]


class MacroCycleAnalyzer:
    """A股宏观经济周期定位器。

    核心维度：
    1) PMI 景气趋势
    2) M1-M2 信用剪刀差
    3) 沪深300估值分位 + 全A破净率
    """

    _UNAVAILABLE = "数据不可用"

    def __init__(self, cache_ttl_hours: float = 24.0) -> None:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            ak = None
        self.ak = ak
        self.cache_ttl_seconds = max(60.0, float(cache_ttl_hours) * 3600.0)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._cross_asset: Any = None
        if CrossAssetAnalyzer is not None:
            try:
                self._cross_asset = CrossAssetAnalyzer(cache_ttl_hours=12.0)
            except Exception:
                self._cross_asset = None

    @contextmanager
    def _without_proxy_env(self):
        keys = [
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        ]
        backup = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            yield
        finally:
            for k, v in backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _ak_call(self, fn_name: str, **kwargs) -> pd.DataFrame:
        if self.ak is None or not hasattr(self.ak, fn_name):
            raise AttributeError(f"akshare missing function: {fn_name}")
        with self._without_proxy_env():
            fn = getattr(self.ak, fn_name)
            out = fn(**kwargs)
        if out is None or not isinstance(out, pd.DataFrame):
            raise ValueError(f"akshare empty frame: {fn_name}")
        return out.copy()

    def _cache_key(self, fn_name: str, **kwargs) -> str:
        if not kwargs:
            return fn_name
        items = ",".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
        return f"{fn_name}|{items}"

    def _cached_ak_df(self, fn_name: str, **kwargs) -> pd.DataFrame:
        key = self._cache_key(fn_name, **kwargs)
        now = time.time()
        hit = self._cache.get(key)
        if hit is not None:
            ts, cached_df = hit
            if (now - ts) <= self.cache_ttl_seconds:
                return cached_df.copy()
        fresh = self._ak_call(fn_name, **kwargs)
        self._cache[key] = (now, fresh.copy())
        return fresh

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return float(value)
        text = str(value).strip()
        if text in {"", "-", "--", "None", "nan", "NaN"}:
            return None
        text = text.replace(",", "").replace("%", "")
        m = re.search(r"[-+]?\d+(\.\d+)?", text)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    @staticmethod
    def _safe_round(value: Optional[float], digits: int = 2) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), digits)

    @staticmethod
    def _clamp(value: float, left: float, right: float) -> float:
        return max(left, min(right, float(value)))

    @staticmethod
    def _find_col(columns: list[str], include_tokens: list[str]) -> Optional[str]:
        for col in columns:
            text = str(col).upper()
            if all(token.upper() in text for token in include_tokens):
                return col
        return None

    @staticmethod
    def _parse_month(value: Any) -> pd.Timestamp:
        if value is None:
            return pd.NaT
        text = str(value).strip()
        if not text:
            return pd.NaT

        m = re.search(r"(\d{4})\D+(\d{1,2})", text)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            if 1 <= month <= 12:
                return pd.Timestamp(year=year, month=month, day=1)

        dt = pd.to_datetime(text, errors="coerce")
        if pd.isna(dt):
            return pd.NaT
        return pd.Timestamp(year=int(dt.year), month=int(dt.month), day=1)

    @classmethod
    def _to_series(cls, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "month": df[date_col].map(cls._parse_month),
                "value": pd.to_numeric(df[value_col], errors="coerce"),
            }
        )
        out = out.dropna(subset=["month", "value"]).sort_values("month")
        out = out.drop_duplicates(subset=["month"], keep="last")
        return out.reset_index(drop=True)

    def get_pmi_trend(self) -> Dict[str, Any]:
        """PMI趋势定位。

        Returns
        -------
        dict
            {
              "latest_pmi": float | None,
              "trend": "expanding"/"contracting"/"turning_up"/"turning_down"/"数据不可用",
              "months_above_50": int
            }
        """
        try:
            df = self._cached_ak_df("macro_china_pmi")
            if df.empty:
                raise ValueError("PMI empty")

            cols = [str(c) for c in df.columns]
            date_col = "月份" if "月份" in df.columns else cols[0]
            pmi_col = self._find_col(cols, ["制造业", "指数"]) or self._find_col(cols, ["PMI"])
            if pmi_col is None:
                raise KeyError("PMI column missing")

            series_df = self._to_series(df=df, date_col=date_col, value_col=pmi_col)
            if series_df.empty:
                raise ValueError("PMI series empty")

            values = series_df["value"].astype(float).tolist()
            latest = float(values[-1])

            months_above_50 = 0
            for val in reversed(values):
                if float(val) > 50.0:
                    months_above_50 += 1
                else:
                    break

            improving_3m = False
            weakening_3m = False
            if len(values) >= 4:
                last4 = values[-4:]
                diffs = [last4[i + 1] - last4[i] for i in range(3)]
                improving_3m = all(d > 0 for d in diffs)
                weakening_3m = all(d < 0 for d in diffs)

            if improving_3m:
                trend = "expanding" if latest > 50.0 else "turning_up"
            elif weakening_3m:
                trend = "turning_down" if latest > 50.0 else "contracting"
            else:
                trend = "expanding" if latest > 50.0 else "contracting"

            return {
                "latest_pmi": self._safe_round(latest, 2),
                "trend": trend,
                "months_above_50": int(months_above_50),
            }
        except Exception:
            return {"latest_pmi": None, "trend": self._UNAVAILABLE, "months_above_50": 0}

    def get_credit_cycle(self) -> Dict[str, Any]:
        """信用周期定位：M1-M2 剪刀差。"""
        try:
            df = self._cached_ak_df("macro_china_money_supply")
            if df.empty:
                raise ValueError("money supply empty")

            cols = [str(c) for c in df.columns]
            date_col = "月份" if "月份" in df.columns else cols[0]
            m1_col = self._find_col(cols, ["M1", "同比"])
            m2_col = self._find_col(cols, ["M2", "同比"])
            if m1_col is None or m2_col is None:
                raise KeyError("M1/M2 yoy column missing")

            work = pd.DataFrame(
                {
                    "month": df[date_col].map(self._parse_month),
                    "m1": pd.to_numeric(df[m1_col], errors="coerce"),
                    "m2": pd.to_numeric(df[m2_col], errors="coerce"),
                }
            )
            work = work.dropna(subset=["month", "m1", "m2"]).sort_values("month")
            work = work.drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)
            if work.empty:
                raise ValueError("credit series empty")

            m1 = float(work["m1"].iloc[-1])
            m2 = float(work["m2"].iloc[-1])
            gap = m1 - m2

            prev_gap = gap
            if len(work) >= 2:
                prev_gap = float(work["m1"].iloc[-2] - work["m2"].iloc[-2])

            if gap > 0 and gap >= prev_gap:
                trend = "activating"
            elif gap > 0 and gap < prev_gap:
                trend = "active_but_cooling"
            elif gap <= 0 and gap > prev_gap:
                trend = "improving"
            elif gap < 0 and gap <= prev_gap:
                trend = "tightening"
            else:
                trend = "neutral"

            return {
                "m1_growth": self._safe_round(m1, 2),
                "m2_growth": self._safe_round(m2, 2),
                "scissors_gap": self._safe_round(gap, 2),
                "trend": trend,
            }
        except Exception:
            return {
                "m1_growth": None,
                "m2_growth": None,
                "scissors_gap": None,
                "trend": self._UNAVAILABLE,
            }

    def _compute_percentile(self, values: pd.Series, current: float) -> Optional[float]:
        s = pd.to_numeric(values, errors="coerce").dropna()
        if s.empty:
            return None
        n = int(s.shape[0])
        if n <= 1:
            return 50.0
        pct = 100.0 * float((s <= current).sum()) / float(n)
        return self._clamp(pct, 0.0, 100.0)

    def _valuation_assessment(
        self,
        pe_pct: Optional[float],
        below_book_ratio: Optional[float],
    ) -> str:
        if pe_pct is None and below_book_ratio is None:
            return self._UNAVAILABLE

        low_score = 0
        high_score = 0

        if pe_pct is not None:
            if pe_pct <= 25:
                low_score += 1
            elif pe_pct >= 75:
                high_score += 1

        if below_book_ratio is not None:
            if below_book_ratio >= 8:
                low_score += 1
            elif below_book_ratio <= 3:
                high_score += 1

        if low_score >= 2:
            return "估值显著偏低"
        if high_score >= 2:
            return "估值显著偏高"
        if low_score == 1 and high_score == 0:
            return "估值偏低"
        if high_score == 1 and low_score == 0:
            return "估值偏高"
        return "估值中性"

    def get_market_valuation_context(self) -> Dict[str, Any]:
        """估值背景：沪深300 PE 分位 + 全A破净率。"""
        try:
            csi300_pe: Optional[float] = None
            csi300_pe_percentile: Optional[float] = None
            below_book_ratio: Optional[float] = None

            try:
                pe_df = self._cached_ak_df("stock_index_pe_lg", symbol="沪深300")
                if not pe_df.empty:
                    cols = [str(c) for c in pe_df.columns]
                    date_col = "日期" if "日期" in pe_df.columns else ("date" if "date" in pe_df.columns else cols[0])
                    pe_col = None
                    for candidate in ("滚动市盈率", "等权滚动市盈率", "ttmPe", "PE"):
                        if candidate in pe_df.columns:
                            pe_col = candidate
                            break
                    if pe_col is None:
                        pe_col = (
                            self._find_col(cols, ["TTM", "PE"])
                            or self._find_col(cols, ["PE"])
                        )
                    if pe_col is not None:
                        work = pd.DataFrame(
                            {
                                "date": pd.to_datetime(pe_df[date_col], errors="coerce"),
                                "pe": pd.to_numeric(pe_df[pe_col], errors="coerce"),
                            }
                        ).dropna(subset=["date", "pe"]).sort_values("date")
                        if not work.empty:
                            csi300_pe = float(work["pe"].iloc[-1])
                            csi300_pe_percentile = self._compute_percentile(work["pe"], csi300_pe)
            except Exception:
                pass

            try:
                pb_df = self._cached_ak_df("stock_a_below_net_asset_statistics", symbol="全部A股")
                if not pb_df.empty:
                    cols = [str(c) for c in pb_df.columns]
                    date_col = "date" if "date" in pb_df.columns else ("日期" if "日期" in pb_df.columns else cols[0])
                    ratio_col = (
                        "below_net_asset_ratio"
                        if "below_net_asset_ratio" in pb_df.columns
                        else self._find_col(cols, ["破净", "比例"])
                    )
                    if ratio_col is not None:
                        work = pd.DataFrame(
                            {
                                "date": pd.to_datetime(pb_df[date_col], errors="coerce"),
                                "ratio": pd.to_numeric(pb_df[ratio_col], errors="coerce"),
                            }
                        ).dropna(subset=["date", "ratio"]).sort_values("date")
                        if not work.empty:
                            latest_ratio = float(work["ratio"].iloc[-1])
                            if latest_ratio <= 1.0:
                                latest_ratio = latest_ratio * 100.0
                            below_book_ratio = latest_ratio
            except Exception:
                pass

            assessment = self._valuation_assessment(
                pe_pct=csi300_pe_percentile,
                below_book_ratio=below_book_ratio,
            )
            return {
                "csi300_pe": self._safe_round(csi300_pe, 2),
                "csi300_pe_percentile": self._safe_round(csi300_pe_percentile, 2),
                "below_book_ratio": self._safe_round(below_book_ratio, 2),
                "assessment": assessment,
            }
        except Exception:
            return {
                "csi300_pe": None,
                "csi300_pe_percentile": None,
                "below_book_ratio": None,
                "assessment": self._UNAVAILABLE,
            }

    def _score_pmi(self, pmi: Dict[str, Any]) -> int:
        trend = str(pmi.get("trend", "") or "")
        if trend == "turning_up":
            return 1
        if trend == "expanding":
            return 1
        if trend == "turning_down":
            return -1
        if trend == "contracting":
            return -1
        return 0

    def _score_credit(self, credit: Dict[str, Any]) -> int:
        gap = self._safe_float(credit.get("scissors_gap"))
        trend = str(credit.get("trend", "") or "")
        if gap is None:
            return 0
        if gap > 0:
            return 1
        if trend == "improving" and gap > -1.0:
            return 0
        return -1

    def _score_valuation(self, valuation: Dict[str, Any]) -> int:
        pe_pct = self._safe_float(valuation.get("csi300_pe_percentile"))
        below_ratio = self._safe_float(valuation.get("below_book_ratio"))
        if pe_pct is None and below_ratio is None:
            return 0
        low = (
            (pe_pct is not None and pe_pct <= 35.0)
            or (below_ratio is not None and below_ratio >= 8.0)
        )
        high = (
            (pe_pct is not None and pe_pct >= 70.0)
            and (below_ratio is not None and below_ratio <= 4.0)
        )
        if low and not high:
            return 1
        if high and not low:
            return -1
        return 0

    @staticmethod
    def _neutral_cross_asset() -> Dict[str, Any]:
        return {
            "environment": "neutral",
            "favored_sectors": [],
            "avoid_sectors": [],
            "confidence": 0.0,
            "summary": "💱 跨资产：Neutral | 数据不可用",
            "signals": {
                "bond": {"yield_10y": None, "trend": "neutral", "equity_implication": "中性", "score": 0.0},
                "commodity": {"copper_trend": "neutral", "steel_trend": "neutral", "cycle_signal": "neutral", "score": 0.0},
                "fx": {"usdcny": None, "trend": "neutral", "equity_implication": "中性", "score": 0.0},
                "total_score": 0.0,
            },
        }

    def _score_cross_asset(self, cross_asset: Dict[str, Any]) -> int:
        env = str(cross_asset.get("environment", "neutral") or "neutral").lower()
        conf = self._safe_float(cross_asset.get("confidence"))
        if conf is not None and conf < 35:
            return 0
        if env == "risk_on":
            return 1
        if env == "risk_off":
            return -1
        return 0

    def _build_summary(
        self,
        phase: str,
        stance: str,
        pmi: Dict[str, Any],
        credit: Dict[str, Any],
        valuation: Dict[str, Any],
        cross_asset: Optional[Dict[str, Any]] = None,
    ) -> str:
        if phase == self._UNAVAILABLE:
            if isinstance(cross_asset, dict):
                cross_summary = str(cross_asset.get("summary", "") or "").strip()
                if cross_summary:
                    return f"宏观数据不可用（仅适用于A股市场），建议暂以风险控制为主。{cross_summary}"
            return "宏观数据不可用（仅适用于A股市场），建议暂以风险控制为主。"

        latest_pmi = self._safe_float(pmi.get("latest_pmi"))
        gap = self._safe_float(credit.get("scissors_gap"))
        pe_pct = self._safe_float(valuation.get("csi300_pe_percentile"))

        pmi_text = f"PMI {latest_pmi:.1f}" if latest_pmi is not None else "PMI 数据缺失"
        gap_text = f"剪刀差 {gap:+.1f}%" if gap is not None else "剪刀差缺失"
        pe_text = f"PE分位 {pe_pct:.0f}%" if pe_pct is not None else "PE分位缺失"
        base = f"{pmi_text}、{gap_text}、{pe_text} 综合判断为{phase}，建议{stance}。"
        if isinstance(cross_asset, dict):
            cross_summary = str(cross_asset.get("summary", "") or "").strip()
            if cross_summary:
                return f"{base} {cross_summary}"
        return base

    def get_cycle_position(self) -> Dict[str, Any]:
        """输出宏观周期位置（仅适用于 A 股）。"""
        try:
            pmi = self.get_pmi_trend()
            credit = self.get_credit_cycle()
            valuation = self.get_market_valuation_context()
            cross_asset = self._neutral_cross_asset()
            if self._cross_asset is not None and hasattr(self._cross_asset, "cross_asset_dashboard"):
                try:
                    one = self._cross_asset.cross_asset_dashboard()
                    if isinstance(one, dict) and one:
                        cross_asset = one
                except Exception:
                    cross_asset = self._neutral_cross_asset()

            details: Dict[str, Any] = {
                "pmi": pmi,
                "credit": credit,
                "valuation": valuation,
                "cross_asset": cross_asset,
                "applicable_market": "A股",
            }

            if (
                str(pmi.get("trend", "")) == self._UNAVAILABLE
                and str(credit.get("trend", "")) == self._UNAVAILABLE
                and str(valuation.get("assessment", "")) == self._UNAVAILABLE
                and str(cross_asset.get("environment", "neutral")).lower() == "neutral"
            ):
                cross_asset_bar = str(cross_asset.get("summary", "") or "💱 跨资产：Neutral | 数据不可用")
                return {
                    "cycle_phase": self._UNAVAILABLE,
                    "investment_stance": "持有观望",
                    "confidence": 0.0,
                    "details": details,
                    "cross_asset_bar": cross_asset_bar,
                    "summary": self._build_summary(
                        phase=self._UNAVAILABLE,
                        stance="持有观望",
                        pmi=pmi,
                        credit=credit,
                        valuation=valuation,
                        cross_asset=cross_asset,
                    ),
                }

            pmi_score = self._score_pmi(pmi)
            credit_score = self._score_credit(credit)
            valuation_score = self._score_valuation(valuation)
            cross_asset_score = self._score_cross_asset(cross_asset)
            total_score = pmi_score + credit_score + valuation_score + cross_asset_score
            details["signals"] = {
                "pmi_score": pmi_score,
                "credit_score": credit_score,
                "valuation_score": valuation_score,
                "cross_asset_score": cross_asset_score,
                "total_score": total_score,
            }

            latest_pmi = self._safe_float(pmi.get("latest_pmi"))
            pmi_trend = str(pmi.get("trend", "") or "")
            months_above_50 = int(pmi.get("months_above_50", 0) or 0)
            valuation_is_low = valuation_score > 0
            valuation_is_high = valuation_score < 0
            credit_is_expanding = credit_score > 0
            credit_is_tightening = credit_score < 0
            cross_risk_on = cross_asset_score > 0
            cross_risk_off = cross_asset_score < 0
            pmi_recovering = pmi_trend in {"turning_up"} or (
                pmi_trend == "expanding" and months_above_50 <= 4
            )
            pmi_high_and_cooling = (
                latest_pmi is not None
                and latest_pmi >= 50.0
                and pmi_trend in {"turning_down"}
            )

            if pmi_recovering and credit_is_expanding and (valuation_is_low or cross_risk_on):
                cycle_phase = "复苏早期"
                stance = "积极加仓"
            elif pmi_high_and_cooling and credit_is_tightening and (valuation_is_high or cross_risk_off):
                cycle_phase = "过热晚期"
                stance = "减仓防守"
            elif latest_pmi is not None and latest_pmi < 50 and not credit_is_expanding and valuation_score >= 0 and not cross_risk_off:
                cycle_phase = "衰退底部"
                stance = "逐步建仓"
            elif total_score >= 2:
                cycle_phase = "扩张中期"
                stance = "持有观望"
            elif total_score <= -2:
                cycle_phase = "过热晚期" if valuation_is_high else "衰退底部"
                stance = "减仓防守" if cycle_phase == "过热晚期" else "逐步建仓"
            elif valuation_is_high and credit_is_tightening:
                cycle_phase = "过热晚期"
                stance = "减仓防守"
            elif valuation_is_low and (pmi_score <= 0 or credit_score <= 0):
                cycle_phase = "衰退底部"
                stance = "逐步建仓"
            elif cross_risk_on and not valuation_is_high:
                cycle_phase = "复苏早期"
                stance = "积极加仓"
            elif cross_risk_off and valuation_is_high:
                cycle_phase = "过热晚期"
                stance = "减仓防守"
            else:
                cycle_phase = "扩张中期"
                stance = "持有观望"

            available_dims = int(
                (str(pmi.get("trend", "")) != self._UNAVAILABLE)
                + (str(credit.get("trend", "")) != self._UNAVAILABLE)
                + (str(valuation.get("assessment", "")) != self._UNAVAILABLE)
                + (str(cross_asset.get("environment", "neutral")).lower() in {"risk_on", "risk_off"})
            )
            decisive_dims = int((pmi_score != 0) + (credit_score != 0) + (valuation_score != 0) + (cross_asset_score != 0))
            confidence = 30.0 + 12.0 * available_dims + 8.0 * decisive_dims + 6.0 * abs(total_score)
            if cycle_phase in {"复苏早期", "过热晚期"} and decisive_dims >= 3:
                confidence += 6.0
            confidence = self._clamp(confidence, 0.0, 100.0)

            summary = self._build_summary(
                phase=cycle_phase,
                stance=stance,
                pmi=pmi,
                credit=credit,
                valuation=valuation,
                cross_asset=cross_asset,
            )
            cross_asset_bar = str(cross_asset.get("summary", "") or "💱 跨资产：Neutral | 数据不可用")
            return {
                "cycle_phase": cycle_phase,
                "investment_stance": stance,
                "confidence": round(float(confidence), 2),
                "details": details,
                "cross_asset_bar": cross_asset_bar,
                "summary": summary,
            }
        except Exception:
            cross_asset = self._neutral_cross_asset()
            return {
                "cycle_phase": self._UNAVAILABLE,
                "investment_stance": "持有观望",
                "confidence": 0.0,
                "details": {
                    "pmi": {"latest_pmi": None, "trend": self._UNAVAILABLE, "months_above_50": 0},
                    "credit": {"m1_growth": None, "m2_growth": None, "scissors_gap": None, "trend": self._UNAVAILABLE},
                    "valuation": {
                        "csi300_pe": None,
                        "csi300_pe_percentile": None,
                        "below_book_ratio": None,
                        "assessment": self._UNAVAILABLE,
                    },
                    "cross_asset": cross_asset,
                    "applicable_market": "A股",
                },
                "cross_asset_bar": str(cross_asset.get("summary", "") or "💱 跨资产：Neutral | 数据不可用"),
                "summary": "宏观数据不可用（仅适用于A股市场），建议暂以风险控制为主。💱 跨资产：Neutral | 数据不可用",
            }
