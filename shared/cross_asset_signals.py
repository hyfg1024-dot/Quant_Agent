from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd


class CrossAssetAnalyzer:
    """跨资产信号传导分析器。"""

    _UNAVAILABLE = "数据不可用"

    def __init__(self, cache_ttl_hours: float = 12.0) -> None:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            ak = None
        self.ak = ak
        self.cache_ttl_seconds = max(300.0, float(cache_ttl_hours) * 3600.0)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._state_cache: dict[str, tuple[float, Any]] = {}

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

    def _cache_key(self, fn_name: str, **kwargs) -> str:
        if not kwargs:
            return fn_name
        items = ",".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
        return f"{fn_name}|{items}"

    def _ak_call(self, fn_name: str, **kwargs) -> pd.DataFrame:
        if self.ak is None or not hasattr(self.ak, fn_name):
            raise AttributeError(f"akshare missing function: {fn_name}")
        with self._without_proxy_env():
            fn = getattr(self.ak, fn_name)
            out = fn(**kwargs)
        if out is None or not isinstance(out, pd.DataFrame):
            raise ValueError(f"akshare empty frame: {fn_name}")
        return out.copy()

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
    def _neutral_bond() -> Dict[str, Any]:
        return {
            "yield_10y": None,
            "trend": "neutral",
            "equity_implication": "中性（国债收益率数据不可用）",
            "score": 0.0,
        }

    @staticmethod
    def _neutral_commodity() -> Dict[str, Any]:
        return {
            "copper_trend": "neutral",
            "steel_trend": "neutral",
            "cycle_signal": "neutral",
            "score": 0.0,
        }

    @staticmethod
    def _neutral_fx() -> Dict[str, Any]:
        return {
            "usdcny": None,
            "trend": "neutral",
            "equity_implication": "中性（汇率数据不可用）",
            "score": 0.0,
        }

    def bond_equity_signal(self) -> Dict[str, Any]:
        """国债收益率 -> 高股息股映射。"""
        try:
            now = datetime.now().date()
            windows = [
                (now - timedelta(days=365), now),
                (now - timedelta(days=730), now - timedelta(days=365)),
                (now - timedelta(days=1095), now - timedelta(days=730)),
            ]
            dfs: list[pd.DataFrame] = []
            for start, end in windows:
                df = self._cached_ak_df(
                    "bond_china_yield",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                if isinstance(df, pd.DataFrame) and not df.empty:
                    dfs.append(df)
            if not dfs:
                return self._neutral_bond()

            raw = pd.concat(dfs, ignore_index=True)
            cols = [str(c) for c in raw.columns]
            date_col = "日期" if "日期" in raw.columns else (cols[0] if cols else "")
            y10_col = "10年" if "10年" in raw.columns else self._find_col(cols, ["10", "年"])
            if not date_col or not y10_col:
                return self._neutral_bond()

            work = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw[date_col], errors="coerce"),
                    "y10": pd.to_numeric(raw[y10_col], errors="coerce"),
                }
            ).dropna(subset=["date", "y10"]).sort_values("date")
            work = work.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            if len(work) < 30:
                return self._neutral_bond()

            latest = float(work["y10"].iloc[-1])
            last3 = work["y10"].tail(3).astype(float).tolist()
            down_3 = len(last3) == 3 and (last3[2] < last3[1] < last3[0])
            p30 = float(work["y10"].quantile(0.3))
            low_zone = latest <= p30

            score = 0.0
            trend = "neutral"
            implication = "中性"
            if down_3 and low_zone:
                trend = "down_low"
                score = 80.0
                implication = "10Y收益率连续下行且处于近3年低位，利好高股息与类债资产"
            elif down_3:
                trend = "down"
                score = 60.0
                implication = "10Y收益率连续下行，债股跷跷板偏向高股息"
            elif low_zone:
                trend = "low"
                score = 50.0
                implication = "收益率处于低位，红利资产相对优势仍在"
            elif len(last3) == 3 and (last3[2] > last3[1] > last3[0]):
                trend = "up"
                score = -30.0
                implication = "收益率走高压制高估值板块，偏防守"
            else:
                trend = "neutral"
                score = 10.0
                implication = "收益率方向不明，股债配置中性"

            return {
                "yield_10y": round(latest, 4),
                "trend": trend,
                "equity_implication": implication,
                "score": round(score, 2),
            }
        except Exception:
            return self._neutral_bond()

    def _commodity_one_trend(self, symbol_candidates: list[str]) -> Dict[str, Any]:
        now = datetime.now().date()
        start = (now - timedelta(days=260)).strftime("%Y%m%d")
        end = now.strftime("%Y%m%d")
        for symbol in symbol_candidates:
            try:
                df = self._cached_ak_df("futures_main_sina", symbol=symbol, start_date=start, end_date=end)
                if df is None or df.empty:
                    continue
                cols = [str(c) for c in df.columns]
                date_col = "日期" if "日期" in df.columns else (cols[0] if cols else "")
                close_col = "收盘价" if "收盘价" in df.columns else self._find_col(cols, ["收盘"])
                if not date_col or not close_col:
                    continue
                work = pd.DataFrame(
                    {
                        "date": pd.to_datetime(df[date_col], errors="coerce"),
                        "close": pd.to_numeric(df[close_col], errors="coerce"),
                    }
                ).dropna(subset=["date", "close"]).sort_values("date")
                work = work.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
                if len(work) < 40:
                    continue

                close = work["close"].astype(float)
                latest = float(close.iloc[-1])
                ma20 = float(close.tail(20).mean())
                ma60 = float(close.tail(60).mean()) if len(close) >= 60 else float(close.mean())
                bottom20 = float(close.tail(60).min()) if len(close) >= 60 else float(close.min())

                if latest > ma20 > ma60 and latest >= bottom20 * 1.06:
                    trend = "rebound"
                    score = 70.0
                elif latest > ma20 and ma20 >= ma60:
                    trend = "up"
                    score = 55.0
                elif latest < ma20 < ma60:
                    trend = "down"
                    score = -40.0
                else:
                    trend = "neutral"
                    score = 5.0
                return {
                    "trend": trend,
                    "score": score,
                    "latest": round(latest, 3),
                    "symbol": symbol,
                }
            except Exception:
                continue
        return {"trend": "neutral", "score": 0.0, "latest": None, "symbol": ""}

    def commodity_cycle_signal(self) -> Dict[str, Any]:
        """工业品价格周期信号。"""
        try:
            copper = self._commodity_one_trend(["CU0", "CU", "HG0"])
            steel = self._commodity_one_trend(["RB0", "RB"])
            oil = self._commodity_one_trend(["SC0", "CL0", "OIL0"])

            score = 0.55 * float(copper.get("score", 0.0) or 0.0) + 0.30 * float(steel.get("score", 0.0) or 0.0)
            score += 0.15 * float(oil.get("score", 0.0) or 0.0)

            copper_trend = str(copper.get("trend", "neutral") or "neutral")
            steel_trend = str(steel.get("trend", "neutral") or "neutral")
            if copper_trend in {"rebound", "up"} and steel_trend in {"rebound", "up"}:
                cycle_signal = "recovery"
            elif copper_trend == "down" and steel_trend == "down":
                cycle_signal = "slowdown"
            else:
                cycle_signal = "neutral"

            return {
                "copper_trend": copper_trend,
                "steel_trend": steel_trend,
                "cycle_signal": cycle_signal,
                "score": round(float(score), 2),
            }
        except Exception:
            return self._neutral_commodity()

    def usd_cny_signal(self) -> Dict[str, Any]:
        """美元兑人民币信号（人民币升值通常对应风险偏好改善）。"""
        try:
            df = self._cached_ak_df("fx_spot_quote")
            if df is None or df.empty:
                return self._neutral_fx()

            pair_col = "货币对" if "货币对" in df.columns else self._find_col([str(c) for c in df.columns], ["货币"])
            bid_col = "买报价" if "买报价" in df.columns else self._find_col([str(c) for c in df.columns], ["买"])
            ask_col = "卖报价" if "卖报价" in df.columns else self._find_col([str(c) for c in df.columns], ["卖"])
            if not pair_col or (not bid_col and not ask_col):
                return self._neutral_fx()

            work = df.copy()
            work[pair_col] = work[pair_col].astype(str).str.upper().str.replace(" ", "", regex=False)
            mask = work[pair_col].str.contains("USD/CNY|USDCNY|美元/人民币|美元人民币", regex=True, na=False)
            target = work[mask]
            if target.empty:
                return self._neutral_fx()

            bid = self._safe_float(target.iloc[0].get(bid_col)) if bid_col else None
            ask = self._safe_float(target.iloc[0].get(ask_col)) if ask_col else None
            if bid is None and ask is None:
                return self._neutral_fx()
            usdcny = float((bid if bid is not None else ask) + (ask if ask is not None else bid)) / 2.0

            now = time.time()
            prev = self._state_cache.get("usdcny")
            self._state_cache["usdcny"] = (now, usdcny)

            trend = "neutral"
            score = 0.0
            implication = "汇率中性，对A股影响有限"
            if prev is not None and (now - float(prev[0])) <= self.cache_ttl_seconds * 2.0:
                prev_val = self._safe_float(prev[1])
                if prev_val is not None:
                    diff = usdcny - float(prev_val)
                    if diff <= -0.03:
                        trend = "cny_appreciating"
                        score = 65.0
                        implication = "人民币升值趋势，外资流入预期改善，利好港股/核心资产"
                    elif diff >= 0.03:
                        trend = "cny_depreciating"
                        score = -35.0
                        implication = "人民币走弱，外资风险偏好下降，成长风格承压"

            if trend == "neutral":
                if usdcny <= 6.95:
                    trend = "cny_strong"
                    score = 55.0
                    implication = "人民币处于偏强区间，利好风险资产"
                elif usdcny >= 7.25:
                    trend = "cny_weak"
                    score = -30.0
                    implication = "人民币偏弱，市场偏防守"
                else:
                    trend = "range"
                    score = 10.0
                    implication = "汇率区间震荡，股市影响中性"

            return {
                "usdcny": round(usdcny, 4),
                "trend": trend,
                "equity_implication": implication,
                "score": round(score, 2),
            }
        except Exception:
            return self._neutral_fx()

    def cross_asset_dashboard(self) -> Dict[str, Any]:
        """跨资产综合环境判断。"""
        try:
            bond = self.bond_equity_signal()
            commodity = self.commodity_cycle_signal()
            fx = self.usd_cny_signal()

            all_unavailable = (
                self._safe_float(bond.get("yield_10y")) is None
                and self._safe_float(fx.get("usdcny")) is None
                and str(commodity.get("copper_trend", "neutral")) == "neutral"
                and str(commodity.get("steel_trend", "neutral")) == "neutral"
                and abs(float(self._safe_float(commodity.get("score")) or 0.0)) < 1e-9
            )
            if all_unavailable:
                return {
                    "environment": "neutral",
                    "favored_sectors": [],
                    "avoid_sectors": [],
                    "confidence": 0.0,
                    "summary": "💱 跨资产：Neutral | 数据不可用",
                    "signals": {
                        "bond": bond,
                        "commodity": commodity,
                        "fx": fx,
                        "total_score": 0.0,
                    },
                }

            bond_score = float(self._safe_float(bond.get("score")) or 0.0)
            commodity_score = float(self._safe_float(commodity.get("score")) or 0.0)
            fx_score = float(self._safe_float(fx.get("score")) or 0.0)

            total = 0.35 * bond_score + 0.40 * commodity_score + 0.25 * fx_score
            if total >= 35:
                environment = "risk_on"
            elif total <= -25:
                environment = "risk_off"
            else:
                environment = "neutral"

            favored: list[str] = []
            avoid: list[str] = []

            if bond_score >= 50:
                favored.append("高股息")
            if commodity_score >= 35:
                favored.extend(["周期", "资源品"])
            if fx_score >= 45:
                favored.append("港股")
            if environment == "risk_on" and "成长" not in favored:
                favored.append("成长")

            if environment == "risk_off":
                avoid.extend(["高估值成长", "高杠杆"])
            if fx_score <= -20:
                avoid.append("外资敏感板块")
            if commodity_score <= -20:
                avoid.append("强周期")

            favored = list(dict.fromkeys(favored))
            avoid = list(dict.fromkeys(avoid))

            available_dims = int(
                (str(bond.get("trend", "neutral")) != "neutral")
                + (str(commodity.get("cycle_signal", "neutral")) != "neutral")
                + (str(fx.get("trend", "neutral")) not in {"neutral", "range"})
            )
            confidence = 42.0 + 14.0 * available_dims + 0.35 * abs(total)
            confidence = self._clamp(confidence, 0.0, 100.0)

            bond_desc = "国债收益率下行" if bond_score >= 50 else ("国债收益率上行" if bond_score < -10 else "国债收益率震荡")
            commodity_desc = "铜价回升" if str(commodity.get("copper_trend")) in {"rebound", "up"} else "铜价未企稳"
            env_text = "Risk-On" if environment == "risk_on" else ("Risk-Off" if environment == "risk_off" else "Neutral")

            if environment == "risk_on":
                conclusion = "利好周期+高股息"
            elif environment == "risk_off":
                conclusion = "防守优先，回避高波动成长"
            else:
                conclusion = "风格均衡，结构性配置"

            summary = f"💱 跨资产：{env_text} | {commodity_desc} | {bond_desc} | {conclusion}"

            return {
                "environment": environment,
                "favored_sectors": favored,
                "avoid_sectors": avoid,
                "confidence": round(float(confidence), 2),
                "summary": summary,
                "signals": {
                    "bond": bond,
                    "commodity": commodity,
                    "fx": fx,
                    "total_score": round(float(total), 2),
                },
            }
        except Exception:
            return {
                "environment": "neutral",
                "favored_sectors": [],
                "avoid_sectors": [],
                "confidence": 0.0,
                "summary": "💱 跨资产：Neutral | 数据不可用",
                "signals": {
                    "bond": self._neutral_bond(),
                    "commodity": self._neutral_commodity(),
                    "fx": self._neutral_fx(),
                    "total_score": 0.0,
                },
            }
