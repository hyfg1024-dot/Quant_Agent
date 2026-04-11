"""Performance metrics and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine, BacktestResult, clone_config_with_borrow_rate


@dataclass
class MetricsReport:
    """Structured metrics payload used by console and report rendering."""

    summary_table: pd.DataFrame
    monthly_returns: pd.DataFrame
    annual_returns: pd.DataFrame
    drawdown_table: pd.DataFrame
    asset_table: pd.DataFrame
    extras: Dict[str, Any]


class MetricsCalculator:
    """Compute return/risk/cost metrics for backtest outputs."""

    def calculate_all(
        self,
        result: BacktestResult,
        benchmark: Optional[pd.Series],
        rf_rate: float = 0.03,
    ) -> MetricsReport:
        """Calculate complete metrics for long/short/portfolio/benchmark."""
        series_map: Dict[str, pd.Series] = {
            "long": self._clean_series(result.daily_long_value),
            "short": self._clean_series(result.daily_short_value),
            "portfolio": self._clean_series(result.daily_portfolio_value),
        }
        if benchmark is not None and not benchmark.empty:
            bmk = self._clean_series(benchmark)
            if not bmk.empty:
                base = float(bmk.iloc[0])
                series_map["benchmark"] = bmk / base if base != 0 else bmk

        metric_rows: List[Dict[str, Any]] = []
        monthly_dict: Dict[str, pd.Series] = {}
        annual_dict: Dict[str, pd.Series] = {}
        dd_rows: List[Dict[str, Any]] = []

        portfolio_ret = series_map["portfolio"].pct_change().dropna()
        benchmark_ret = series_map.get("benchmark", pd.Series(dtype=float)).pct_change().dropna()

        long_short_corr = np.nan
        if not series_map["long"].empty and not series_map["short"].empty:
            long_ret = series_map["long"].pct_change().dropna()
            short_ret = series_map["short"].pct_change().dropna()
            idx = long_ret.index.intersection(short_ret.index)
            if len(idx) > 10:
                long_short_corr = float(long_ret.loc[idx].corr(short_ret.loc[idx]))

        portfolio_beta = self._beta(portfolio_ret, benchmark_ret)
        portfolio_ir = self._information_ratio(portfolio_ret, benchmark_ret)

        for name, s in series_map.items():
            m = self._compute_series_metrics(s, rf_rate)
            metric_rows.append({"series": name, **m})
            monthly_dict[name] = self._monthly_returns(s)
            annual_dict[name] = self._annual_returns(s)
            dd_rows.append(
                {
                    "series": name,
                    "max_drawdown_pct": m["max_drawdown_pct"],
                    "max_drawdown_amt": m["max_drawdown_amt"],
                    "max_drawdown_days": m["max_drawdown_days"],
                }
            )

        summary_table = pd.DataFrame(metric_rows).set_index("series")

        if "portfolio" in summary_table.index:
            total_cost = float(result.costs.get("total", 0.0))
            start_val = float(series_map["portfolio"].iloc[0]) if not series_map["portfolio"].empty else np.nan
            end_val = float(series_map["portfolio"].iloc[-1]) if not series_map["portfolio"].empty else np.nan
            net_pnl = end_val - start_val if np.isfinite(start_val) and np.isfinite(end_val) else np.nan
            gross_pnl = net_pnl + total_cost if np.isfinite(net_pnl) else np.nan
            cost_drag = total_cost / gross_pnl if np.isfinite(gross_pnl) and abs(gross_pnl) > 1e-9 else np.nan
            summary_table.loc["portfolio", "total_trade_cost"] = float(result.costs.get("commission_long", 0.0) + result.costs.get("commission_short", 0.0))
            summary_table.loc["portfolio", "total_borrow_cost"] = float(result.costs.get("borrow_fee", 0.0))
            summary_table.loc["portfolio", "total_cost"] = total_cost
            summary_table.loc["portfolio", "cost_drag_ratio"] = cost_drag
            summary_table.loc["portfolio", "beta_vs_hsi"] = portfolio_beta
            summary_table.loc["portfolio", "information_ratio"] = portfolio_ir
            summary_table.loc["portfolio", "long_short_corr"] = long_short_corr

        monthly_returns = self._concat_series_dict(monthly_dict)
        annual_returns = self._concat_series_dict(annual_dict)
        drawdown_table = pd.DataFrame(dd_rows).set_index("series")
        asset_table = self._build_asset_table(result)

        extras = {
            "long_short_corr": long_short_corr,
            "portfolio_beta": portfolio_beta,
            "portfolio_ir": portfolio_ir,
        }

        return MetricsReport(
            summary_table=summary_table,
            monthly_returns=monthly_returns,
            annual_returns=annual_returns,
            drawdown_table=drawdown_table,
            asset_table=asset_table,
            extras=extras,
        )

    def sensitivity_analysis(
        self,
        engine: BacktestEngine,
        param: str,
        values: List[float],
    ) -> pd.DataFrame:
        """Run sensitivity tests for selected parameter values."""
        if param != "short_borrow_rate":
            raise ValueError("当前仅支持 short_borrow_rate 敏感性分析")

        rows: List[Dict[str, Any]] = []
        for v in values:
            cfg = clone_config_with_borrow_rate(engine.config, float(v))
            tmp_engine = BacktestEngine(
                config=cfg,
                data_manager=engine.dm,
                benchmark_codes=list(engine.benchmark_codes),
                logger=lambda *_args, **_kwargs: None,
            )
            out = tmp_engine.run()
            s = self._clean_series(out.daily_portfolio_value)
            if s.empty:
                continue
            cumulative = float(s.iloc[-1] / s.iloc[0] - 1.0)
            rows.append(
                {
                    "borrow_rate": float(v),
                    "final_value": float(s.iloc[-1]),
                    "cumulative_return": cumulative,
                    "total_cost": float(out.costs.get("total", 0.0)),
                    "borrow_cost": float(out.costs.get("borrow_fee", 0.0)),
                }
            )
        return pd.DataFrame(rows).sort_values("borrow_rate").reset_index(drop=True)

    def _compute_series_metrics(self, series: pd.Series, rf_rate: float) -> Dict[str, float]:
        s = self._clean_series(series)
        if s.empty:
            return self._empty_metrics()

        daily_ret = s.pct_change().dropna()
        cumulative = float(s.iloc[-1] / s.iloc[0] - 1.0) if s.iloc[0] != 0 else np.nan

        n_days = len(s)
        years = n_days / 252.0 if n_days > 1 else np.nan
        if np.isfinite(years) and years > 0 and s.iloc[0] > 0 and s.iloc[-1] > 0:
            annual_ret = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)
        else:
            annual_ret = np.nan

        ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else np.nan
        downside = daily_ret[daily_ret < 0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else np.nan

        sharpe = (annual_ret - rf_rate) / ann_vol if np.isfinite(annual_ret) and np.isfinite(ann_vol) and ann_vol > 0 else np.nan
        sortino = (annual_ret - rf_rate) / downside_vol if np.isfinite(annual_ret) and np.isfinite(downside_vol) and downside_vol > 0 else np.nan

        dd, max_dd_pct, max_dd_amt, max_dd_days = self._drawdown_stats(s)
        calmar = annual_ret / abs(max_dd_pct) if np.isfinite(annual_ret) and np.isfinite(max_dd_pct) and max_dd_pct < 0 else np.nan

        var95 = float(np.nanquantile(daily_ret.values, 0.05)) if len(daily_ret) > 1 else np.nan
        cvar95 = float(daily_ret[daily_ret <= var95].mean()) if len(daily_ret) > 1 and np.isfinite(var95) else np.nan

        monthly = self._monthly_returns(s)
        best_month = float(monthly.max()) if not monthly.empty else np.nan
        worst_month = float(monthly.min()) if not monthly.empty else np.nan
        win_rate = float((monthly > 0).mean()) if not monthly.empty else np.nan

        return {
            "cumulative_return": cumulative,
            "annual_return": annual_ret,
            "annual_volatility": ann_vol,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_amt": max_dd_amt,
            "max_drawdown_days": float(max_dd_days),
            "downside_volatility": downside_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "var_95": var95,
            "cvar_95": cvar95,
            "best_month": best_month,
            "worst_month": worst_month,
            "monthly_win_rate": win_rate,
        }

    def _drawdown_stats(self, series: pd.Series) -> tuple[pd.Series, float, float, int]:
        running_max = series.cummax()
        dd = series / running_max - 1.0
        max_dd_pct = float(dd.min()) if not dd.empty else np.nan

        if dd.empty:
            return dd, np.nan, np.nan, 0

        trough_dt = dd.idxmin()
        peak_val = float(running_max.loc[trough_dt])
        trough_val = float(series.loc[trough_dt])
        max_dd_amt = trough_val - peak_val

        # max drawdown duration as consecutive below-high-water days
        dur = 0
        max_dur = 0
        for v in dd.values:
            if v < 0:
                dur += 1
                max_dur = max(max_dur, dur)
            else:
                dur = 0

        return dd, max_dd_pct, float(max_dd_amt), int(max_dur)

    def _monthly_returns(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=float)
        return series.resample("ME").last().pct_change().dropna()

    def _annual_returns(self, series: pd.Series) -> pd.Series:
        if series.empty:
            return pd.Series(dtype=float)
        return series.resample("YE").last().pct_change().dropna()

    def _beta(self, target_ret: pd.Series, benchmark_ret: pd.Series) -> float:
        if target_ret.empty or benchmark_ret.empty:
            return np.nan
        idx = target_ret.index.intersection(benchmark_ret.index)
        if len(idx) < 30:
            return np.nan
        x = benchmark_ret.loc[idx].values
        y = target_ret.loc[idx].values
        var = np.var(x)
        if var <= 0:
            return np.nan
        cov = np.cov(y, x)[0, 1]
        return float(cov / var)

    def _information_ratio(self, target_ret: pd.Series, benchmark_ret: pd.Series) -> float:
        if target_ret.empty or benchmark_ret.empty:
            return np.nan
        idx = target_ret.index.intersection(benchmark_ret.index)
        if len(idx) < 30:
            return np.nan
        active = target_ret.loc[idx] - benchmark_ret.loc[idx]
        te = active.std()
        if te <= 0:
            return np.nan
        return float(active.mean() / te * np.sqrt(252))

    def _concat_series_dict(self, d: Dict[str, pd.Series]) -> pd.DataFrame:
        if not d:
            return pd.DataFrame()
        cols = []
        for k, s in d.items():
            cols.append(s.rename(k))
        return pd.concat(cols, axis=1)

    def _build_asset_table(self, result: BacktestResult) -> pd.DataFrame:
        pos = result.daily_positions.copy()
        if pos.empty:
            return pd.DataFrame(columns=["code", "start_price", "end_price", "return", "pnl_contribution"]).set_index("code")

        pos["date"] = pd.to_datetime(pos["date"])
        rows: List[Dict[str, Any]] = []
        total_contrib = float(result.code_contribution.sum()) if not result.code_contribution.empty else 0.0

        for code, grp in pos.groupby("code"):
            gp = grp.sort_values("date")
            valid = gp[pd.notna(gp["price"]) & (gp["price"] > 0)]
            if valid.empty:
                continue
            p0 = float(valid.iloc[0]["price"])
            p1 = float(valid.iloc[-1]["price"])
            ret = p1 / p0 - 1.0 if p0 > 0 else np.nan
            contrib = float(result.code_contribution.get(code, 0.0))
            contrib_ratio = contrib / total_contrib if abs(total_contrib) > 1e-9 else np.nan
            rows.append(
                {
                    "code": code,
                    "start_price": p0,
                    "end_price": p1,
                    "return": ret,
                    "pnl_contribution": contrib,
                    "contribution_ratio": contrib_ratio,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["code", "start_price", "end_price", "return", "pnl_contribution", "contribution_ratio"]).set_index("code")

        return pd.DataFrame(rows).set_index("code").sort_values("pnl_contribution", ascending=False)

    def _clean_series(self, s: pd.Series) -> pd.Series:
        out = pd.to_numeric(s, errors="coerce").dropna()
        out.index = pd.to_datetime(out.index)
        return out.sort_index()

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "cumulative_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "max_drawdown_pct": np.nan,
            "max_drawdown_amt": np.nan,
            "max_drawdown_days": np.nan,
            "downside_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "var_95": np.nan,
            "cvar_95": np.nan,
            "best_month": np.nan,
            "worst_month": np.nan,
            "monthly_win_rate": np.nan,
        }
