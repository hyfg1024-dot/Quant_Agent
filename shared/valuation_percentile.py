from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from shared.db_manager import connect_db, normalize_position_symbol


@dataclass
class PercentileConfig:
    """估值分位参数配置。"""

    lookback_years: int = 5
    min_points_warning: int = 50


class ValuationPercentile:
    """估值历史百分位计算器。

    基于 `daily_fundamental` 表计算单只股票的估值历史分位（0=最便宜，100=最贵）。
    当 DuckDB 历史数据不足时，会自动尝试 AkShare 回退抓取。
    """

    SUPPORTED_METRICS = {"pe_ttm", "pb", "dividend_yield"}
    # 该类指标数值越大通常越便宜，因此需要反向转为“贵/便宜分位”
    REVERSED_METRICS = {"dividend_yield"}

    def __init__(self, config: Optional[PercentileConfig] = None) -> None:
        self.config = config or PercentileConfig()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            x = float(value)
            if pd.isna(x):
                return None
            return x
        except Exception:
            return None

    @staticmethod
    def _clamp_0_100(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return max(0.0, min(100.0, float(value)))

    def _normalize_symbol(self, market: str, code: str) -> tuple[str, str, str]:
        mk_raw = str(market or "").strip().upper()
        if mk_raw in {"CN", "SH", "SZ"}:
            mk_raw = "A"
        elif mk_raw in {"H"}:
            mk_raw = "HK"
        mk, db_code, display = normalize_position_symbol(code, market=mk_raw or None)
        return mk, db_code, display

    def _assessment(self, percentile: Optional[float], data_points: int) -> str:
        if percentile is None:
            base = "无数据"
        elif percentile < 20:
            base = "极度低估"
        elif percentile < 40:
            base = "偏低"
        elif percentile < 60:
            base = "中性"
        elif percentile < 80:
            base = "偏高"
        else:
            base = "极度高估"

        if data_points < self.config.min_points_warning:
            return f"{base}(数据不足)"
        return base

    def _compute_from_series(self, metric: str, series: pd.Series) -> Dict[str, Any]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {
                "metric": metric,
                "current_value": None,
                "percentile": None,
                "min": None,
                "max": None,
                "median": None,
                "mean": None,
                "data_points": 0,
                "assessment": "无数据(数据不足)",
            }

        current = self._safe_float(s.iloc[-1])
        n = int(s.shape[0])
        min_v = self._safe_float(s.min())
        max_v = self._safe_float(s.max())
        median_v = self._safe_float(s.median())
        mean_v = self._safe_float(s.mean())

        pct: Optional[float]
        if n <= 1 or current is None:
            pct = 50.0 if current is not None else None
        else:
            if metric in self.REVERSED_METRICS:
                # 值越高越便宜 => “贵分位”用大于当前值的数量衡量
                cnt_more_expensive = int((s > current).sum())
                pct = 100.0 * cnt_more_expensive / (n - 1)
            else:
                cnt_more_expensive = int((s < current).sum())
                pct = 100.0 * cnt_more_expensive / (n - 1)

        pct = self._clamp_0_100(pct)
        return {
            "metric": metric,
            "current_value": current,
            "percentile": pct,
            "min": min_v,
            "max": max_v,
            "median": median_v,
            "mean": mean_v,
            "data_points": n,
            "assessment": self._assessment(pct, n),
        }

    def _query_history_series(
        self,
        market: str,
        code: str,
        metric: str,
        lookback_years: int,
    ) -> pd.Series:
        years = max(1, min(int(lookback_years), 20))
        sql = f"""
        WITH latest AS (
            SELECT trade_date AS latest_date
            FROM daily_fundamental
            WHERE market = ? AND code = ? AND {metric} IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT 1
        )
        SELECT CAST(f.{metric} AS DOUBLE) AS v
        FROM daily_fundamental f
        JOIN latest l ON TRUE
        WHERE f.market = ? AND f.code = ?
          AND f.{metric} IS NOT NULL
          AND f.trade_date BETWEEN l.latest_date - INTERVAL '{years}' YEAR AND l.latest_date
        ORDER BY f.trade_date ASC
        """
        with connect_db(read_only=True) as conn:
            df = conn.execute(sql, [market, code, market, code]).df()
        if df.empty or "v" not in df.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(df["v"], errors="coerce").dropna()

    def _fallback_from_akshare(self, code: str, metric: str) -> pd.Series:
        """AkShare 回退获取估值历史序列。

        参数
        ----------
        code : str
            股票代码。A股如 600036，港股可传 00700 或 00700.HK。
        metric : str
            估值字段，仅支持 pe_ttm/pb/dividend_yield。

        返回
        ----------
        pd.Series
            按时间升序的估值序列；若失败则返回空序列。
        """
        try:
            import akshare as ak
        except Exception:
            return pd.Series(dtype="float64")

        metric = str(metric).strip().lower()
        indicator_map = {
            "pe_ttm": ["市盈率(TTM)", "市盈率TTM", "市盈率"],
            "pb": ["市净率"],
            "dividend_yield": ["股息率", "股息率(%)"],
        }
        periods = ["近五年", "近5年", "近十年", "近一年", "全部"]
        indicators = indicator_map.get(metric, [])
        if not indicators:
            return pd.Series(dtype="float64")

        code_text = str(code or "").strip().upper()
        is_hk = code_text.endswith(".HK") or (len("".join(ch for ch in code_text if ch.isdigit())) <= 5)
        digits = "".join(ch for ch in code_text if ch.isdigit())
        if not digits:
            return pd.Series(dtype="float64")
        symbol = digits.zfill(5) if is_hk else digits.zfill(6)

        fetchers = []
        if is_hk:
            fn = getattr(ak, "stock_hk_valuation_baidu", None)
            if callable(fn):
                fetchers.append(lambda ind, per: fn(symbol=symbol, indicator=ind, period=per))
        else:
            fn = getattr(ak, "stock_zh_valuation_baidu", None)
            if callable(fn):
                fetchers.append(lambda ind, per: fn(symbol=symbol, indicator=ind, period=per))

        for fetch in fetchers:
            for ind in indicators:
                for per in periods:
                    try:
                        df = fetch(ind, per)
                        if df is None or df.empty:
                            continue
                        cols = [str(c) for c in df.columns]
                        date_col = next((c for c in cols if ("date" in c.lower() or "日期" in c)), cols[0])
                        value_col = None
                        for c in cols:
                            cl = c.lower()
                            if cl in {"value", "估值", "数值"} or ("value" in cl):
                                value_col = c
                                break
                        if value_col is None:
                            numeric_cols = [
                                c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
                            ]
                            value_col = numeric_cols[-1] if numeric_cols else None
                        if value_col is None:
                            continue

                        out = pd.DataFrame(
                            {
                                "date": pd.to_datetime(df[date_col], errors="coerce"),
                                "value": pd.to_numeric(df[value_col], errors="coerce"),
                            }
                        ).dropna(subset=["date", "value"]).sort_values("date")
                        if not out.empty:
                            return out["value"].reset_index(drop=True)
                    except Exception:
                        continue
        return pd.Series(dtype="float64")

    def compute_percentile(
        self,
        market: str,
        code: str,
        metric: str = "pe_ttm",
        lookback_years: int = 5,
    ) -> Dict[str, Any]:
        """计算单指标估值历史百分位。

        返回字段：
        metric/current_value/percentile/min/max/median/mean/data_points/assessment
        """
        metric = str(metric or "").strip().lower()
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"不支持的 metric: {metric}，仅支持 {sorted(self.SUPPORTED_METRICS)}")

        mk, db_code, display = self._normalize_symbol(market=market, code=code)
        series = self._query_history_series(market=mk, code=db_code, metric=metric, lookback_years=lookback_years)

        if series.shape[0] == 0:
            series = self._fallback_from_akshare(display if mk == "HK" else db_code, metric)

        result = self._compute_from_series(metric=metric, series=series)
        result["market"] = mk
        result["code"] = db_code
        result["lookback_years"] = int(lookback_years)
        return result

    def compute_multi_percentile(
        self,
        market: str,
        code: str,
        lookback_years: int = 5,
    ) -> Dict[str, Any]:
        """计算 PE/PB/股息率三维历史分位，并给出综合估值分位。"""
        pe = self.compute_percentile(market=market, code=code, metric="pe_ttm", lookback_years=lookback_years)
        pb = self.compute_percentile(market=market, code=code, metric="pb", lookback_years=lookback_years)
        dy = self.compute_percentile(
            market=market, code=code, metric="dividend_yield", lookback_years=lookback_years
        )

        pe_pct = self._safe_float(pe.get("percentile"))
        pb_pct = self._safe_float(pb.get("percentile"))
        if pe_pct is not None and pb_pct is not None:
            composite = (pe_pct + pb_pct) / 2.0
            composite_assessment = self._assessment(
                composite,
                min(int(pe.get("data_points", 0) or 0), int(pb.get("data_points", 0) or 0)),
            )
        else:
            composite = None
            composite_assessment = "无数据(数据不足)"

        return {
            "market": pe.get("market"),
            "code": pe.get("code"),
            "lookback_years": int(lookback_years),
            "pe_ttm": pe,
            "pb": pb,
            "dividend_yield": dy,
            "composite_percentile": self._clamp_0_100(composite),
            "composite_assessment": composite_assessment,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def batch_screen_undervalued(
        self,
        percentile_threshold: float = 30.0,
        min_data_points: int = 200,
    ) -> pd.DataFrame:
        """批量扫描全市场低估标的（PE+PB 综合分位）。

        参数
        ----------
        percentile_threshold : float
            综合分位阈值，小于该值即认为偏低估。
        min_data_points : int
            PE 与 PB 最少历史点数下限。
        """
        threshold = float(percentile_threshold)
        min_pts = max(1, int(min_data_points))
        years = max(1, min(int(self.config.lookback_years), 20))

        sql = f"""
        WITH latest AS (
            SELECT market, code, trade_date AS latest_date, pe_ttm AS current_pe, pb AS current_pb
            FROM (
                SELECT market, code, trade_date, pe_ttm, pb,
                       ROW_NUMBER() OVER (PARTITION BY market, code ORDER BY trade_date DESC) AS rn
                FROM daily_fundamental
                WHERE pe_ttm IS NOT NULL OR pb IS NOT NULL
            ) t
            WHERE rn = 1
        ),
        hist AS (
            SELECT
                f.market,
                f.code,
                f.pe_ttm,
                f.pb,
                l.current_pe,
                l.current_pb
            FROM daily_fundamental f
            JOIN latest l
              ON f.market = l.market AND f.code = l.code
            WHERE f.trade_date BETWEEN l.latest_date - INTERVAL '{years}' YEAR AND l.latest_date
        ),
        agg AS (
            SELECT
                market,
                code,
                MAX(current_pe) AS current_pe,
                MAX(current_pb) AS current_pb,
                COUNT(pe_ttm) AS pe_points,
                SUM(CASE WHEN pe_ttm < current_pe THEN 1 ELSE 0 END) AS pe_lt,
                MIN(pe_ttm) AS pe_min,
                MAX(pe_ttm) AS pe_max,
                MEDIAN(pe_ttm) AS pe_median,
                AVG(pe_ttm) AS pe_mean,
                COUNT(pb) AS pb_points,
                SUM(CASE WHEN pb < current_pb THEN 1 ELSE 0 END) AS pb_lt,
                MIN(pb) AS pb_min,
                MAX(pb) AS pb_max,
                MEDIAN(pb) AS pb_median,
                AVG(pb) AS pb_mean
            FROM hist
            GROUP BY market, code
        ),
        calc AS (
            SELECT
                *,
                CASE WHEN pe_points > 1 THEN 100.0 * pe_lt / (pe_points - 1) END AS pe_percentile,
                CASE WHEN pb_points > 1 THEN 100.0 * pb_lt / (pb_points - 1) END AS pb_percentile
            FROM agg
        )
        SELECT
            c.market,
            c.code,
            COALESCE(b.name, '') AS name,
            c.current_pe AS pe_ttm,
            c.current_pb AS pb,
            c.pe_percentile,
            c.pb_percentile,
            (c.pe_percentile + c.pb_percentile) / 2.0 AS composite_percentile,
            c.pe_points,
            c.pb_points,
            c.pe_min,
            c.pe_max,
            c.pe_median,
            c.pe_mean,
            c.pb_min,
            c.pb_max,
            c.pb_median,
            c.pb_mean
        FROM calc c
        LEFT JOIN stock_basic b
          ON c.market = b.market AND c.code = b.code
        WHERE c.pe_points >= ?
          AND c.pb_points >= ?
          AND c.pe_percentile IS NOT NULL
          AND c.pb_percentile IS NOT NULL
          AND (c.pe_percentile + c.pb_percentile) / 2.0 < ?
        ORDER BY composite_percentile ASC, c.market, c.code
        """
        with connect_db(read_only=True) as conn:
            df = conn.execute(sql, [min_pts, min_pts, threshold]).df()

        if df.empty:
            return df

        def _assess(row: pd.Series) -> str:
            pct = self._safe_float(row.get("composite_percentile"))
            dpts = int(min(row.get("pe_points", 0), row.get("pb_points", 0)))
            return self._assessment(pct, dpts)

        df["composite_assessment"] = df.apply(_assess, axis=1)
        return df

