"""Signal-driven inflection strategy for backtest engine."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.inflection_detector import InflectionDetector
from shared.technical_structure import TechnicalStructureAnalyzer
from shared.valuation_percentile import ValuationPercentile


class InflectionStrategy:
    """Dynamic stock-selection strategy based on multi-factor inflection signals."""

    def __init__(
        self,
        *,
        market: str = "A",
        pool_source: str = "filter",
        pool_size: int = 100,
        min_score: float = 50.0,
        max_valuation_percentile: float = 40.0,
        top_n: int = 5,
        fallback_pool: Optional[Sequence[Tuple[str, str]]] = None,
        logger=print,
    ) -> None:
        self.market = str(market or "A").strip().upper()
        self.pool_source = str(pool_source or "filter").strip().lower()
        self.pool_size = max(1, int(pool_size))
        self.min_score = float(min_score)
        self.max_valuation_percentile = float(max_valuation_percentile)
        self.top_n = max(1, int(top_n))
        self.fallback_pool = list(fallback_pool or [])
        self.logger = logger

        self.inflection = InflectionDetector(enable_propagation_tip=False)
        self.technical = TechnicalStructureAnalyzer()
        self.valuation = ValuationPercentile()

        self._candidate_cache: Optional[List[Tuple[str, str]]] = None
        self._inflection_cache: Dict[str, float] = {}
        self._valuation_cache: Dict[str, Optional[float]] = {}
        self._bottom_cache: Dict[Tuple[str, str], float] = {}

    def get_candidates(self) -> List[Tuple[str, str]]:
        if self._candidate_cache is not None:
            return list(self._candidate_cache)

        if self.pool_source == "filter":
            candidates = self._load_from_filter_snapshot()
        else:
            candidates = []
        if not candidates:
            candidates = list(self.fallback_pool)

        dedup: Dict[str, str] = {}
        for code, name in candidates:
            c = str(code or "").strip().upper()
            if not c:
                continue
            if c not in dedup:
                dedup[c] = str(name or c)
        out = [(c, n) for c, n in dedup.items()]
        self._candidate_cache = out[: self.pool_size]
        return list(self._candidate_cache)

    def build_rebalance_plan(
        self,
        *,
        rebalance_date: pd.Timestamp,
        aligned_map: Dict[str, pd.DataFrame],
        current_shares: Dict[str, float],
    ) -> Dict[str, Any]:
        active_codes = set(aligned_map.keys())
        candidates = [(c, n) for c, n in self.get_candidates() if c in active_codes]

        score_rows: List[Dict[str, Any]] = []
        for code, name in candidates:
            inflection_score = self._get_inflection_score(code=code, name=name)
            bottom_score = self._get_bottom_score(code=code, dt=rebalance_date, aligned_df=aligned_map[code])
            valuation_pct = self._get_valuation_percentile(code=code)
            valuation_input = 100.0 if valuation_pct is None else float(np.clip(valuation_pct, 0.0, 100.0))
            composite_score = (
                float(inflection_score) * 0.4
                + float(bottom_score) * 0.2
                + (100.0 - float(valuation_input)) * 0.4
            )
            score_rows.append(
                {
                    "code": code,
                    "name": name,
                    "inflection_score": round(float(inflection_score), 2),
                    "bottom_score": round(float(bottom_score), 2),
                    "valuation_percentile": round(float(valuation_input), 2),
                    "composite_score": round(float(np.clip(composite_score, 0.0, 100.0)), 2),
                }
            )

        score_df = pd.DataFrame(score_rows)
        if score_df.empty:
            return {
                "target_weights": {c: 0.0 for c in active_codes},
                "selected_codes": [],
                "score_table": score_df,
                "forced_exit_codes": sorted(c for c, sh in current_shares.items() if sh > 0),
            }

        score_df["eligible"] = (
            (score_df["inflection_score"] >= self.min_score)
            & (score_df["valuation_percentile"] <= self.max_valuation_percentile)
        )
        score_df = score_df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        selected_codes = (
            score_df.loc[score_df["eligible"], "code"]
            .head(self.top_n)
            .astype(str)
            .tolist()
        )

        held_codes = {c for c, sh in current_shares.items() if float(sh) > 0.0}
        score_map = {str(r["code"]): float(r["composite_score"]) for r in score_df.to_dict(orient="records")}
        forced_exit = sorted(c for c in held_codes if score_map.get(c, 0.0) < 30.0)
        selected_codes = [c for c in selected_codes if c not in set(forced_exit)]

        target_weights = {c: 0.0 for c in active_codes}
        if selected_codes:
            w = 1.0 / float(len(selected_codes))
            for code in selected_codes:
                target_weights[code] = w

        self.logger(
            f"[INFLECTION] {rebalance_date.date()} 候选={len(score_df)} 入选={len(selected_codes)} "
            f"强制卖出={len(forced_exit)}"
        )
        return {
            "target_weights": target_weights,
            "selected_codes": selected_codes,
            "score_table": score_df,
            "forced_exit_codes": forced_exit,
        }

    def _load_from_filter_snapshot(self) -> List[Tuple[str, str]]:
        db_path = PROJECT_ROOT / "apps" / "filter" / "data" / "filter_market.db"
        if not db_path.exists():
            return []

        market_clause = {
            "A": "market = 'A'",
            "HK": "market = 'HK'",
            "AH": "market IN ('A', 'HK')",
        }.get(self.market, "market = 'A'")

        sql = f"""
        SELECT market, code, name
        FROM market_snapshot
        WHERE {market_clause}
        ORDER BY (total_mv IS NULL) ASC, total_mv DESC, code
        LIMIT ?
        """
        out: List[Tuple[str, str]] = []
        try:
            with sqlite3.connect(str(db_path)) as conn:
                df = pd.read_sql_query(sql, conn, params=[int(self.pool_size)])
            for _, row in df.iterrows():
                market = str(row.get("market", "")).strip().upper()
                code = self._to_trade_code(str(row.get("code", "")), market=market)
                name = str(row.get("name", "") or code).strip()
                if code:
                    out.append((code, name))
        except Exception:
            return []
        return out

    def _get_inflection_score(self, code: str, name: str) -> float:
        key = str(code).upper()
        if key in self._inflection_cache:
            return self._inflection_cache[key]
        signal_code = self._to_signal_code(code)
        try:
            payload = self.inflection.compute_inflection_score(signal_code, name)
            score = float(payload.get("inflection_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        score = float(np.clip(score, 0.0, 100.0))
        self._inflection_cache[key] = score
        return score

    def _get_bottom_score(self, code: str, dt: pd.Timestamp, aligned_df: pd.DataFrame) -> float:
        day_key = pd.Timestamp(dt).date().isoformat()
        cache_key = (str(code).upper(), day_key)
        if cache_key in self._bottom_cache:
            return self._bottom_cache[cache_key]

        hist = aligned_df.loc[aligned_df.index <= pd.Timestamp(dt)].copy()
        hist = hist.dropna(subset=["adj_close"], how="all")
        if hist.empty:
            self._bottom_cache[cache_key] = 0.0
            return 0.0

        close_col = "adj_close" if "adj_close" in hist.columns else "close"
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(hist.index),
                "open": pd.to_numeric(hist.get("open"), errors="coerce"),
                "high": pd.to_numeric(hist.get("high"), errors="coerce"),
                "low": pd.to_numeric(hist.get("low"), errors="coerce"),
                "close": pd.to_numeric(hist.get(close_col), errors="coerce"),
                "volume": pd.to_numeric(hist.get("volume"), errors="coerce"),
            }
        ).dropna(subset=["date", "close"])
        try:
            payload = self.technical.analyze_bottom_structure(daily)
            score = float(payload.get("bottom_score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        score = float(np.clip(score, 0.0, 100.0))
        self._bottom_cache[cache_key] = score
        return score

    def _get_valuation_percentile(self, code: str) -> Optional[float]:
        key = str(code).upper()
        if key in self._valuation_cache:
            return self._valuation_cache[key]

        market = "HK" if key.endswith(".HK") else "A"
        valuation_code = self._to_signal_code(key) if market == "A" else key
        try:
            payload = self.valuation.compute_percentile(
                market=market,
                code=valuation_code,
                metric="pe_ttm",
                lookback_years=5,
            )
            val = payload.get("percentile")
            out = float(val) if val is not None else None
            if out is not None:
                out = float(np.clip(out, 0.0, 100.0))
        except Exception:
            out = None
        self._valuation_cache[key] = out
        return out

    @staticmethod
    def _to_signal_code(code: str) -> str:
        raw = str(code or "").strip().upper()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if raw.endswith(".HK") or len(digits) <= 5:
            return digits.zfill(5)
        return digits.zfill(6)

    @staticmethod
    def _to_trade_code(raw_code: str, market: str) -> str:
        code = str(raw_code or "").strip().upper()
        if not code:
            return ""
        if code.endswith((".HK", ".SS", ".SZ")):
            return code

        digits = "".join(ch for ch in code if ch.isdigit())
        if not digits:
            return ""

        mk = str(market or "").strip().upper()
        if mk == "HK" or len(digits) <= 5:
            return f"{digits[-4:].zfill(4)}.HK"

        six = digits[-6:].zfill(6)
        suffix = ".SS" if six.startswith(("5", "6", "9")) else ".SZ"
        return f"{six}{suffix}"
