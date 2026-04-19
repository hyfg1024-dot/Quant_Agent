from __future__ import annotations

import logging
import math
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_code(code: str) -> str:
    text = str(code or "").strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    if len(digits) == 5:
        return digits
    return digits


def _is_a_share(code: str) -> bool:
    c = _normalize_code(code)
    return c.isdigit() and len(c) == 6


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)

    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "None", "nan", "NaN"}:
        return None

    unit_mul = 1.0
    if "亿" in text:
        unit_mul = 1.0
    elif "万" in text:
        unit_mul = 0.0001

    text = text.replace("%", "")
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0)) * unit_mul
    except Exception:
        return None


def _to_yi_amount(value: Any) -> Optional[float]:
    raw = _safe_float(value)
    if raw is None:
        return None
    abs_v = abs(raw)
    if abs_v >= 1_000_000:
        return raw / 100_000_000.0
    if 10_000 <= abs_v < 1_000_000:
        return raw / 10_000.0
    return raw


def _find_col(columns: Iterable[Any], keywords: List[str]) -> Optional[str]:
    cols = [str(c) for c in columns]
    for kw in keywords:
        for col in cols:
            if kw.lower() in col.lower():
                return col
    return None


def _consecutive_from_tail(series: pd.Series, predicate) -> int:
    cnt = 0
    for v in reversed(series.tolist()):
        if predicate(v):
            cnt += 1
        else:
            break
    return cnt


def _clip_0_100(value: float) -> float:
    return float(max(0.0, min(100.0, value)))


class CapitalFlowAnalyzer:
    def __init__(self, request_interval_sec: float = 1.0) -> None:
        self.request_interval_sec = max(0.0, float(request_interval_sec))
        self._last_call_ts = 0.0
        self._lock = threading.Lock()
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover
            ak = None
            logger.warning("akshare unavailable for capital flow analyzer: %s", exc)
        self.ak = ak

    def _empty_northbound(self) -> Dict[str, Any]:
        return {
            "trend": "neutral",
            "consecutive_buy_days": 0,
            "total_net_buy_yi": 0.0,
            "score": 0.0,
        }

    def _empty_block_trade(self) -> Dict[str, Any]:
        return {
            "avg_premium_pct": 0.0,
            "trade_count": 0,
            "total_volume_yi": 0.0,
            "score": 0.0,
        }

    def _empty_margin(self) -> Dict[str, Any]:
        return {
            "margin_balance_change_pct": 0.0,
            "trend": "neutral",
            "score": 0.0,
        }

    def _empty_composite(self) -> Dict[str, Any]:
        return {
            "capital_score": 0.0,
            "grade": "流出",
            "signals": [],
            "northbound": self._empty_northbound(),
            "block_trade": self._empty_block_trade(),
            "margin": self._empty_margin(),
        }

    def _throttle(self) -> None:
        if self.request_interval_sec <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_sec = self.request_interval_sec - (now - self._last_call_ts)
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_ts = time.monotonic()

    def _ak_call_try(self, fn_name: str, kwargs_candidates: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        if self.ak is None or not hasattr(self.ak, fn_name):
            return None
        fn = getattr(self.ak, fn_name)
        last_exc: Optional[Exception] = None

        for kwargs in kwargs_candidates:
            try:
                self._throttle()
                df = fn(**kwargs)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return df
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            logger.debug("capital flow call failed %s: %s", fn_name, last_exc)
        return None

    def _filter_by_code(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        code_col = _find_col(df.columns, ["代码", "证券代码", "股票代码", "symbol", "code"])
        if not code_col:
            return df.copy()
        out = df.copy()
        out[code_col] = out[code_col].map(_normalize_code)
        matched = out[out[code_col] == code]
        return matched.copy()

    def _filter_recent_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        date_col = _find_col(out.columns, ["交易日期", "日期", "date", "time"])
        if not date_col:
            return out.tail(max(1, int(days)))
        dt = pd.to_datetime(out[date_col], errors="coerce")
        out = out.assign(__date=dt).dropna(subset=["__date"]).sort_values("__date")
        if out.empty:
            return out
        end_dt = datetime.now().date()
        start_dt = end_dt - timedelta(days=max(1, int(days)) + 1)
        out = out[(out["__date"].dt.date >= start_dt) & (out["__date"].dt.date <= end_dt)]
        return out if not out.empty else pd.DataFrame()

    def northbound_trend(self, code: str, days: int = 10) -> Dict[str, Any]:
        base = self._empty_northbound()
        try:
            normalized = _normalize_code(code)
            if not _is_a_share(normalized):
                return base

            df = None
            for fn_name in ["stock_hsgt_individual_em", "stock_hsgt_hold_stock_em", "stock_hsgt_stock_statistics_em"]:
                df = self._ak_call_try(
                    fn_name,
                    [
                        {"symbol": normalized},
                        {"stock": normalized},
                        {"code": normalized},
                        {},
                    ],
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    break
            if df is None or df.empty:
                return base

            df = self._filter_by_code(df, normalized)
            df = self._filter_recent_days(df, days)
            if df.empty:
                return base

            net_col = _find_col(
                df.columns,
                [
                    "净买入额",
                    "净买入",
                    "净流入",
                    "持股市值变化",
                    "增持市值",
                    "持股变化",
                    "持股数量变化",
                    "增减",
                ],
            )
            if not net_col:
                return base

            ser = pd.to_numeric(df[net_col].map(_to_yi_amount), errors="coerce").dropna()
            if ser.empty:
                return base

            ser = ser.tail(max(1, int(days)))
            consecutive_buy = _consecutive_from_tail(ser, lambda x: float(x) > 0)
            consecutive_sell = _consecutive_from_tail(ser, lambda x: float(x) < 0)
            total_net_buy_yi = float(ser.sum())

            trend = "neutral"
            if consecutive_buy >= 3 or total_net_buy_yi > 1.0:
                trend = "accumulating"
            elif consecutive_sell >= 3 or total_net_buy_yi < -1.0:
                trend = "distributing"

            score = 50.0
            score += min(30.0, consecutive_buy * 6.0)
            score -= min(30.0, consecutive_sell * 6.0)
            score += max(-20.0, min(20.0, total_net_buy_yi * 2.0))

            return {
                "trend": trend,
                "consecutive_buy_days": int(consecutive_buy),
                "total_net_buy_yi": round(total_net_buy_yi, 4),
                "score": round(_clip_0_100(score), 2),
            }
        except Exception as exc:
            logger.debug("northbound_trend failed for %s: %s", code, exc)
            return base

    def block_trade_premium(self, code: str, days: int = 30) -> Dict[str, Any]:
        base = self._empty_block_trade()
        try:
            normalized = _normalize_code(code)
            if not _is_a_share(normalized):
                return base

            end_dt = datetime.now().strftime("%Y%m%d")
            start_dt = (datetime.now() - timedelta(days=max(1, int(days)) + 5)).strftime("%Y%m%d")

            df = None
            for fn_name in ["stock_dzjy_mrmx", "stock_dzjy_mrtj"]:
                df = self._ak_call_try(
                    fn_name,
                    [
                        {"symbol": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"code": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"stock": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"symbol": normalized},
                        {"code": normalized},
                        {"start_date": start_dt, "end_date": end_dt},
                        {},
                    ],
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    break

            if df is None or df.empty:
                return base

            df = self._filter_by_code(df, normalized)
            df = self._filter_recent_days(df, days)
            if df.empty:
                return base

            premium_col = _find_col(df.columns, ["溢价率", "折溢率", "溢价"])
            premium_ser: pd.Series
            if premium_col:
                premium_ser = pd.to_numeric(df[premium_col].map(_safe_float), errors="coerce")
            else:
                price_col = _find_col(df.columns, ["成交价", "均价", "价格"])
                close_col = _find_col(df.columns, ["收盘价", "当日收盘价", "收盘"])
                if not price_col or not close_col:
                    return base
                p1 = pd.to_numeric(df[price_col].map(_safe_float), errors="coerce")
                p2 = pd.to_numeric(df[close_col].map(_safe_float), errors="coerce")
                premium_ser = (p1 - p2) / p2 * 100.0

            premium_ser = premium_ser.replace([math.inf, -math.inf], pd.NA).dropna()
            if premium_ser.empty:
                return base

            amt_col = _find_col(df.columns, ["成交额", "成交金额", "金额"])
            if amt_col:
                volume_yi = pd.to_numeric(df[amt_col].map(_to_yi_amount), errors="coerce").dropna().sum()
            else:
                volume_yi = 0.0

            trade_count = int(len(premium_ser))
            avg_premium_pct = float(premium_ser.mean())

            score = 40.0
            score += max(-30.0, min(36.0, avg_premium_pct * 4.0))
            score += min(14.0, trade_count * 1.4)
            score += min(10.0, float(volume_yi) * 1.2)

            return {
                "avg_premium_pct": round(avg_premium_pct, 4),
                "trade_count": trade_count,
                "total_volume_yi": round(float(volume_yi), 4),
                "score": round(_clip_0_100(score), 2),
            }
        except Exception as exc:
            logger.debug("block_trade_premium failed for %s: %s", code, exc)
            return base

    def margin_trend(self, code: str, days: int = 10) -> Dict[str, Any]:
        base = self._empty_margin()
        try:
            normalized = _normalize_code(code)
            if not _is_a_share(normalized):
                return base

            end_dt = datetime.now().strftime("%Y%m%d")
            start_dt = (datetime.now() - timedelta(days=max(1, int(days)) + 8)).strftime("%Y%m%d")

            fn_candidates: List[str]
            if normalized.startswith("6"):
                fn_candidates = ["stock_margin_detail_sse", "stock_margin_detail_szse", "stock_margin_detail_em"]
            else:
                fn_candidates = ["stock_margin_detail_szse", "stock_margin_detail_sse", "stock_margin_detail_em"]

            df = None
            for fn_name in fn_candidates:
                df = self._ak_call_try(
                    fn_name,
                    [
                        {"symbol": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"code": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"stock": normalized, "start_date": start_dt, "end_date": end_dt},
                        {"symbol": normalized},
                        {"code": normalized},
                        {"start_date": start_dt, "end_date": end_dt},
                        {},
                    ],
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    break

            if df is None or df.empty:
                return base

            df = self._filter_by_code(df, normalized)
            df = self._filter_recent_days(df, days)
            if df.empty:
                return base

            margin_col = _find_col(df.columns, ["融资余额", "融資餘額", "融资融券余额", "rzye", "margin"])
            if not margin_col:
                return base

            ser = pd.to_numeric(df[margin_col].map(_safe_float), errors="coerce").dropna()
            if ser.empty or len(ser) < 2:
                return base

            ser = ser.tail(max(2, int(days)))
            first = float(ser.iloc[0])
            last = float(ser.iloc[-1])
            if first <= 0:
                return base

            change_pct = (last - first) / first * 100.0
            diffs = ser.diff().dropna()
            consecutive_up = _consecutive_from_tail(diffs, lambda x: float(x) > 0)
            consecutive_down = _consecutive_from_tail(diffs, lambda x: float(x) < 0)

            trend = "neutral"
            if consecutive_up >= 3 and change_pct > 0:
                trend = "increasing"
            elif consecutive_down >= 3 and change_pct < 0:
                trend = "decreasing"

            score = 50.0
            score += max(-28.0, min(28.0, change_pct * 1.8))
            score += min(16.0, consecutive_up * 4.0)
            score -= min(16.0, consecutive_down * 4.0)

            return {
                "margin_balance_change_pct": round(float(change_pct), 4),
                "trend": trend,
                "score": round(_clip_0_100(score), 2),
            }
        except Exception as exc:
            logger.debug("margin_trend failed for %s: %s", code, exc)
            return base

    def composite_capital_signal(self, code: str) -> Dict[str, Any]:
        base = self._empty_composite()
        try:
            northbound = self.northbound_trend(code=code, days=10)
            block_trade = self.block_trade_premium(code=code, days=30)
            margin = self.margin_trend(code=code, days=10)

            north_score = _safe_float(northbound.get("score")) or 0.0
            block_score = _safe_float(block_trade.get("score")) or 0.0
            margin_score = _safe_float(margin.get("score")) or 0.0

            capital_score = _clip_0_100(north_score * 0.5 + block_score * 0.3 + margin_score * 0.2)
            if capital_score >= 75:
                grade = "强流入"
            elif capital_score >= 60:
                grade = "弱流入"
            elif capital_score >= 45:
                grade = "中性"
            else:
                grade = "流出"

            return {
                "capital_score": round(capital_score, 2),
                "grade": grade,
                "signals": [
                    {"type": "northbound", **northbound},
                    {"type": "block_trade", **block_trade},
                    {"type": "margin", **margin},
                ],
                "northbound": northbound,
                "block_trade": block_trade,
                "margin": margin,
            }
        except Exception as exc:
            logger.debug("composite_capital_signal failed for %s: %s", code, exc)
            return base
