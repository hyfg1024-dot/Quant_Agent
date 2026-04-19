"""Core long-short backtest engine."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config_loader import PositionConfig, StrategyConfig
from .data_manager import DataManager


@dataclass
class BacktestResult:
    """Backtest output container."""

    daily_portfolio_value: pd.Series
    daily_long_value: pd.Series
    daily_short_value: pd.Series
    daily_positions: pd.DataFrame
    trades: pd.DataFrame
    costs: Dict[str, float]
    stop_loss_events: List[Dict[str, Any]]
    benchmark_nav: Dict[str, pd.Series]
    code_contribution: pd.Series
    daily_costs: pd.DataFrame
    warnings: List[str]


class BacktestEngine:
    """Run configurable long-short backtest on daily close data."""

    def __init__(
        self,
        config: StrategyConfig,
        data_manager: DataManager,
        benchmark_codes: Optional[List[str]] = None,
        logger=print,
    ) -> None:
        self.config = config
        self.dm = data_manager
        self.logger = logger
        self.benchmark_codes = benchmark_codes or ["^HSI"]

    def run(self) -> BacktestResult:
        """Execute backtest from config and return full result payload."""
        start = str(self.config.backtest.start_date)
        end = str(self.config.backtest.end_date)
        strategy_type = str(getattr(self.config, "strategy_type", "static") or "static").strip().lower()
        inflection_runner = None
        inflection_score_logs: Dict[str, pd.DataFrame] = {}

        if strategy_type == "inflection":
            try:
                from strategies.inflection_strategy import InflectionStrategy
            except Exception:
                from apps.backtest.strategies.inflection_strategy import InflectionStrategy

            inflection_runner = InflectionStrategy(
                market=str(self.config.signal_universe.market),
                pool_source=str(self.config.signal_universe.pool_source),
                pool_size=int(self.config.signal_universe.pool_size),
                min_score=float(self.config.inflection.min_score),
                max_valuation_percentile=float(self.config.inflection.max_valuation_percentile),
                top_n=int(self.config.inflection.top_n),
                fallback_pool=list(self.config.candidate_pool),
                logger=self.logger,
            )
            all_codes = sorted({code for code, _ in inflection_runner.get_candidates()})
            if not all_codes:
                raise RuntimeError("拐点策略候选池为空，无法回测")
        else:
            all_codes = sorted({p.code for p in [*self.config.long_positions, *self.config.short_positions]})
            if not all_codes:
                raise RuntimeError("策略未配置标的，无法回测")

        self.logger(f"[BT] 加载价格数据: {len(all_codes)} 个标的")
        raw_map: Dict[str, pd.DataFrame] = {}
        warnings: List[str] = []

        for code in all_codes:
            df = self.dm.fetch_stock_data(code, start=start, end=end)
            if df.empty:
                warnings.append(f"{code} 在回测区间内无数据，已跳过")
                continue
            raw_map[code] = df
            cov = f"{df.index.min().date()} -> {df.index.max().date()}"
            if df.index.min().date() > self.config.backtest.start_date or df.index.max().date() < self.config.backtest.end_date:
                warnings.append(f"{code} 数据覆盖不完整: {cov}")

        active_codes = sorted(raw_map.keys())
        if not active_codes:
            raise RuntimeError("无可用标的数据，无法回测")

        bench_raw: Dict[str, pd.DataFrame] = {}
        for b in self.benchmark_codes:
            try:
                bdf = self.dm.fetch_index_data(b, start=start, end=end)
                if not bdf.empty:
                    bench_raw[b] = bdf
            except Exception as exc:
                warnings.append(f"基准 {b} 拉取失败: {exc}")

        calendar = self._build_calendar(raw_map, bench_raw)
        if calendar.empty:
            raise RuntimeError("回测交易日历为空")

        aligned: Dict[str, pd.DataFrame] = {}
        for code in active_codes:
            prepared = self.dm.prepare_for_calendar(raw_map[code], calendar=calendar, max_suspend_days=30)
            aligned[code] = prepared.aligned
            if bool((prepared.aligned["suspended"]).any()):
                max_streak = int(prepared.aligned["missing_streak"].max())
                warnings.append(f"{code} 存在长期停牌/缺失，最长连续缺失 {max_streak} 天（>30天期间冻结交易）")

        bench_nav = self._build_benchmark_nav(bench_raw, calendar)

        if strategy_type == "inflection":
            long_weights = {c: 0.0 for c in active_codes}
            short_weights: Dict[str, float] = {}
        else:
            long_weights, short_weights = self._resolve_weights(aligned, active_codes)

        shares: Dict[str, float] = {c: 0.0 for c in active_codes}
        entry_price: Dict[str, float] = {c: np.nan for c in active_codes}
        short_enabled: Dict[str, bool] = {c: True for c in short_weights.keys()}

        long_codes = set(active_codes) if strategy_type == "inflection" else set(long_weights.keys())
        short_codes = set(short_weights.keys())
        cash = float(self.config.capital.total_hkd)
        init_long_cap = float(self.config.capital.total_hkd * self.config.capital.long_pct)
        init_short_cap = float(self.config.capital.total_hkd * self.config.capital.short_pct)

        long_pnl_cum = 0.0
        short_pnl_cum = 0.0
        long_fee_cum = 0.0
        short_fee_cum = 0.0
        borrow_fee_cum = 0.0

        trade_rows: List[Dict[str, Any]] = []
        stop_events: List[Dict[str, Any]] = []
        pos_rows: List[Dict[str, Any]] = []
        code_contrib: Dict[str, float] = {c: 0.0 for c in active_codes}

        long_stop_triggered: Set[str] = set()
        short_stop_triggered: Set[str] = set()
        strategy_terminated = False

        rebalance_dates = self._get_rebalance_dates(calendar)
        prev_price: Dict[str, float] = {c: np.nan for c in active_codes}

        nav_rows: List[Tuple[pd.Timestamp, float, float, float]] = []
        cost_rows: List[Dict[str, Any]] = []

        for i, dt in enumerate(calendar):
            px_adj = {c: float(aligned[c].at[dt, "adj_close"]) if pd.notna(aligned[c].at[dt, "adj_close"]) else np.nan for c in active_codes}
            # Use adjusted close consistently for mark-to-market and trading to avoid
            # artificial PnL jumps from corporate-action adjustment gaps.
            px_exec = {c: float(aligned[c].at[dt, "adj_close"]) if pd.notna(aligned[c].at[dt, "adj_close"]) else np.nan for c in active_codes}
            tradable = {c: bool(aligned[c].at[dt, "tradable"]) and not bool(aligned[c].at[dt, "suspended"]) for c in active_codes}

            if i == 0:
                day_trade_fee = 0.0
                day_borrow_fee = 0.0
                if not strategy_terminated:
                    if inflection_runner is not None:
                        plan = inflection_runner.build_rebalance_plan(
                            rebalance_date=dt,
                            aligned_map=aligned,
                            current_shares=shares,
                        )
                        long_weights = dict(plan.get("target_weights", {}))
                        inflection_score_logs[str(dt.date())] = plan.get("score_table", pd.DataFrame())
                        if not plan.get("selected_codes"):
                            warnings.append(f"{dt.date()} 拐点策略无入选标的，调仓后为空仓")
                    cash, fee_l, fee_s = self._rebalance_to_target(
                        dt=dt,
                        shares=shares,
                        cash=cash,
                        px_exec=px_exec,
                        tradable=tradable,
                        long_weights=long_weights,
                        short_weights=short_weights,
                        short_enabled=short_enabled,
                        trade_rows=trade_rows,
                        reason="init",
                    )
                    long_fee_cum += fee_l
                    short_fee_cum += fee_s
                    day_trade_fee += fee_l + fee_s
                    for c in active_codes:
                        if shares[c] != 0.0 and np.isfinite(px_adj[c]):
                            entry_price[c] = px_adj[c]
                equity = self._equity(cash, shares, px_adj)
                long_value = init_long_cap + long_pnl_cum - long_fee_cum
                short_value = init_short_cap + short_pnl_cum - short_fee_cum - borrow_fee_cum
                nav_rows.append((dt, equity, long_value, short_value))
                cost_rows.append(
                    {
                        "date": dt,
                        "trade_fee": day_trade_fee,
                        "borrow_fee": day_borrow_fee,
                        "total_fee": day_trade_fee + day_borrow_fee,
                        "cum_trade_fee": long_fee_cum + short_fee_cum,
                        "cum_borrow_fee": borrow_fee_cum,
                        "cum_total_fee": long_fee_cum + short_fee_cum + borrow_fee_cum,
                    }
                )
                self._append_pos_rows(pos_rows, dt, shares, px_adj, tradable, aligned)
                prev_price = px_adj
                continue

            # Daily mark-to-market PnL from previous close
            day_trade_fee = 0.0
            for c in active_codes:
                p0 = prev_price.get(c)
                p1 = px_adj.get(c)
                if not np.isfinite(p0) or not np.isfinite(p1):
                    continue
                pnl = shares[c] * (p1 - p0)
                code_contrib[c] += pnl
                if c in long_codes:
                    long_pnl_cum += pnl
                else:
                    short_pnl_cum += pnl

            # Borrow fee (daily accrual)
            short_notional = sum(abs(shares[c]) * px_adj[c] for c in short_codes if shares[c] < 0 and np.isfinite(px_adj[c]))
            borrow_fee = short_notional * float(self.config.costs.short_borrow_rate) / 252.0
            cash -= borrow_fee
            borrow_fee_cum += borrow_fee
            day_borrow_fee = borrow_fee

            # Single-name stop loss
            if not strategy_terminated:
                for c in sorted(long_codes):
                    if c in long_stop_triggered:
                        continue
                    if shares[c] <= 0:
                        continue
                    if not np.isfinite(entry_price[c]) or not np.isfinite(px_adj[c]):
                        continue
                    ret = px_adj[c] / entry_price[c] - 1.0
                    if ret <= float(self.config.stop_loss.single_long_stop):
                        if tradable[c]:
                            action = self.config.stop_loss.single_long_action
                            target = shares[c] * 0.5 if action == "halve" else 0.0
                            cash, fee = self._trade_to_target(
                                dt=dt,
                                code=c,
                                target_shares=target,
                                shares=shares,
                                cash=cash,
                                px_exec=px_exec,
                                trade_rows=trade_rows,
                                reason="single_long_stop",
                            )
                            long_fee_cum += fee
                            day_trade_fee += fee
                            long_stop_triggered.add(c)
                            stop_events.append({"date": str(dt.date()), "code": c, "side": "long", "trigger": ret, "action": action})

                for c in sorted(short_codes):
                    if c in short_stop_triggered:
                        continue
                    if shares[c] >= 0:
                        continue
                    if not short_enabled.get(c, True):
                        continue
                    if not np.isfinite(entry_price[c]) or not np.isfinite(px_adj[c]):
                        continue
                    adverse = px_adj[c] / entry_price[c] - 1.0
                    if adverse >= float(self.config.stop_loss.single_short_stop):
                        if tradable[c]:
                            action = self.config.stop_loss.single_short_action
                            if action == "close":
                                cash, fee = self._trade_to_target(
                                    dt=dt,
                                    code=c,
                                    target_shares=0.0,
                                    shares=shares,
                                    cash=cash,
                                    px_exec=px_exec,
                                    trade_rows=trade_rows,
                                    reason="single_short_stop",
                                )
                                short_fee_cum += fee
                                day_trade_fee += fee
                                short_enabled[c] = False
                            short_stop_triggered.add(c)
                            stop_events.append({"date": str(dt.date()), "code": c, "side": "short", "trigger": adverse, "action": action})

            # Portfolio-level stop
            equity_before_reb = self._equity(cash, shares, px_adj)
            total_ret = equity_before_reb / float(self.config.capital.total_hkd) - 1.0
            if (not strategy_terminated) and total_ret <= float(self.config.stop_loss.portfolio_stop):
                if self.config.stop_loss.portfolio_action == "close_all":
                    cash, fee_l, fee_s = self._close_all(
                        dt=dt,
                        shares=shares,
                        cash=cash,
                        px_exec=px_exec,
                        tradable=tradable,
                        long_codes=long_codes,
                        short_codes=short_codes,
                        trade_rows=trade_rows,
                        reason="portfolio_stop",
                    )
                    long_fee_cum += fee_l
                    short_fee_cum += fee_s
                    day_trade_fee += fee_l + fee_s
                    strategy_terminated = True
                    stop_events.append({
                        "date": str(dt.date()),
                        "code": "ALL",
                        "side": "portfolio",
                        "trigger": total_ret,
                        "action": self.config.stop_loss.portfolio_action,
                    })

            # Rebalance (monthly/weekly/daily/quarterly)
            if (not strategy_terminated) and (dt in rebalance_dates):
                long_stop_triggered.clear()
                if inflection_runner is not None:
                    plan = inflection_runner.build_rebalance_plan(
                        rebalance_date=dt,
                        aligned_map=aligned,
                        current_shares=shares,
                    )
                    long_weights = dict(plan.get("target_weights", {}))
                    inflection_score_logs[str(dt.date())] = plan.get("score_table", pd.DataFrame())
                    selected = list(plan.get("selected_codes", []))
                    forced_exit = list(plan.get("forced_exit_codes", []))
                    if not selected:
                        warnings.append(f"{dt.date()} 拐点策略无入选标的，调仓后为空仓")
                    if forced_exit:
                        warnings.append(f"{dt.date()} 拐点策略触发低分卖出: {', '.join(forced_exit[:8])}")
                cash, fee_l, fee_s = self._rebalance_to_target(
                    dt=dt,
                    shares=shares,
                    cash=cash,
                    px_exec=px_exec,
                    tradable=tradable,
                    long_weights=long_weights,
                    short_weights=short_weights,
                    short_enabled=short_enabled,
                    trade_rows=trade_rows,
                    reason="rebalance",
                )
                long_fee_cum += fee_l
                short_fee_cum += fee_s
                day_trade_fee += fee_l + fee_s
                for c in active_codes:
                    if shares[c] != 0.0 and np.isfinite(px_adj[c]):
                        entry_price[c] = px_adj[c]

            equity = self._equity(cash, shares, px_adj)
            long_value = init_long_cap + long_pnl_cum - long_fee_cum
            short_value = init_short_cap + short_pnl_cum - short_fee_cum - borrow_fee_cum
            nav_rows.append((dt, equity, long_value, short_value))
            cost_rows.append(
                {
                    "date": dt,
                    "trade_fee": day_trade_fee,
                    "borrow_fee": day_borrow_fee,
                    "total_fee": day_trade_fee + day_borrow_fee,
                    "cum_trade_fee": long_fee_cum + short_fee_cum,
                    "cum_borrow_fee": borrow_fee_cum,
                    "cum_total_fee": long_fee_cum + short_fee_cum + borrow_fee_cum,
                }
            )
            self._append_pos_rows(pos_rows, dt, shares, px_adj, tradable, aligned)
            prev_price = px_adj

        nav_df = pd.DataFrame(nav_rows, columns=["date", "portfolio", "long", "short"]).set_index("date")
        cost_df = pd.DataFrame(cost_rows).set_index("date") if cost_rows else pd.DataFrame()
        costs = {
            "commission_long": float(long_fee_cum),
            "commission_short": float(short_fee_cum),
            "borrow_fee": float(borrow_fee_cum),
            "total": float(long_fee_cum + short_fee_cum + borrow_fee_cum),
        }

        return BacktestResult(
            daily_portfolio_value=nav_df["portfolio"],
            daily_long_value=nav_df["long"],
            daily_short_value=nav_df["short"],
            daily_positions=pd.DataFrame(pos_rows),
            trades=pd.DataFrame(trade_rows),
            costs=costs,
            stop_loss_events=stop_events,
            benchmark_nav=bench_nav,
            code_contribution=pd.Series(code_contrib).sort_values(ascending=False),
            daily_costs=cost_df,
            warnings=warnings + ([f"拐点策略调仓记录: {len(inflection_score_logs)} 次"] if inflection_score_logs else []),
        )

    def _build_calendar(
        self,
        prices: Dict[str, pd.DataFrame],
        benchmarks: Dict[str, pd.DataFrame],
    ) -> pd.DatetimeIndex:
        if "^HSI" in benchmarks and not benchmarks["^HSI"].empty:
            base = pd.DatetimeIndex(benchmarks["^HSI"].index)
        else:
            idxs = [df.index for df in prices.values() if not df.empty]
            if not idxs:
                return pd.DatetimeIndex([])
            base = idxs[0]
            for i in idxs[1:]:
                base = base.union(i)
        base = pd.DatetimeIndex(base).sort_values().unique()
        start = pd.Timestamp(self.config.backtest.start_date)
        end = pd.Timestamp(self.config.backtest.end_date)
        return base[(base >= start) & (base <= end)]

    def _build_benchmark_nav(self, bench_raw: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        out: Dict[str, pd.Series] = {}
        for code, df in bench_raw.items():
            if df.empty:
                continue
            s = pd.to_numeric(df["adj_close"], errors="coerce").reindex(calendar).ffill()
            if s.notna().sum() == 0:
                continue
            base = float(s.dropna().iloc[0])
            if base == 0:
                continue
            out[code] = s / base
        return out

    def _resolve_weights(
        self,
        aligned: Dict[str, pd.DataFrame],
        active_codes: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        long_codes = [p.code for p in self.config.long_positions if p.code in active_codes]
        short_codes = [p.code for p in self.config.short_positions if p.code in active_codes]
        mode = self.config.weighting_mode

        if mode == "equal":
            lw = {c: 1.0 / len(long_codes) for c in long_codes}
            sw = {c: 1.0 / len(short_codes) for c in short_codes}
            return lw, sw

        if mode == "inverse_volatility":
            def inv_vol(codes: List[str]) -> Dict[str, float]:
                vals: Dict[str, float] = {}
                for c in codes:
                    ret = aligned[c]["adj_close"].pct_change().dropna().tail(120)
                    vol = float(ret.std()) if not ret.empty else np.nan
                    vals[c] = 1.0 / vol if vol and np.isfinite(vol) and vol > 0 else 0.0
                total = sum(vals.values())
                if total <= 0:
                    return {c: 1.0 / len(codes) for c in codes}
                return {c: vals[c] / total for c in codes}

            return inv_vol(long_codes), inv_vol(short_codes)

        lw = {p.code: float(p.weight) for p in self.config.long_positions if p.code in active_codes}
        sw = {p.code: float(p.weight) for p in self.config.short_positions if p.code in active_codes}
        return lw, sw

    def _get_rebalance_dates(self, calendar: pd.DatetimeIndex) -> Set[pd.Timestamp]:
        freq = self.config.rebalance.frequency
        nth = max(1, int(self.config.rebalance.day))
        s = pd.Series(calendar, index=calendar)

        if freq == "daily":
            return set(calendar)

        if freq == "weekly":
            groups = s.groupby([calendar.isocalendar().year, calendar.isocalendar().week])
        elif freq == "quarterly":
            groups = s.groupby([calendar.year, calendar.quarter])
        else:  # monthly
            groups = s.groupby([calendar.year, calendar.month])

        out: Set[pd.Timestamp] = set()
        for _, g in groups:
            idx = min(nth - 1, len(g) - 1)
            out.add(pd.Timestamp(g.iloc[idx]))
        return out

    def _equity(self, cash: float, shares: Dict[str, float], prices: Dict[str, float]) -> float:
        val = float(cash)
        for c, sh in shares.items():
            px = prices.get(c, np.nan)
            if np.isfinite(px):
                val += sh * px
        return val

    def _rebalance_to_target(
        self,
        dt: pd.Timestamp,
        shares: Dict[str, float],
        cash: float,
        px_exec: Dict[str, float],
        tradable: Dict[str, bool],
        long_weights: Dict[str, float],
        short_weights: Dict[str, float],
        short_enabled: Dict[str, bool],
        trade_rows: List[Dict[str, Any]],
        reason: str,
    ) -> Tuple[float, float, float]:
        fee_long = 0.0
        fee_short = 0.0

        equity_now = self._equity(cash, shares, px_exec)
        long_cap = equity_now * float(self.config.capital.long_pct)
        short_cap = equity_now * float(self.config.capital.short_pct)

        # Long side targets
        for c, w in long_weights.items():
            px = px_exec.get(c, np.nan)
            if not np.isfinite(px) or px <= 0:
                continue
            tgt_sh = (long_cap * w) / px
            if tradable.get(c, False):
                cash, fee = self._trade_to_target(dt, c, tgt_sh, shares, cash, px_exec, trade_rows, reason)
                fee_long += fee

        # Short side targets (disabled names stay at current size)
        enabled_codes = [c for c, w in short_weights.items() if short_enabled.get(c, True)]
        if enabled_codes:
            w_sum = sum(short_weights[c] for c in enabled_codes)
            for c in enabled_codes:
                px = px_exec.get(c, np.nan)
                if not np.isfinite(px) or px <= 0:
                    continue
                w = short_weights[c] / w_sum if w_sum > 0 else 1.0 / len(enabled_codes)
                tgt_sh = -(short_cap * w) / px
                if tradable.get(c, False):
                    cash, fee = self._trade_to_target(dt, c, tgt_sh, shares, cash, px_exec, trade_rows, reason)
                    fee_short += fee

        return cash, fee_long, fee_short

    def _close_all(
        self,
        dt: pd.Timestamp,
        shares: Dict[str, float],
        cash: float,
        px_exec: Dict[str, float],
        tradable: Dict[str, bool],
        long_codes: Set[str],
        short_codes: Set[str],
        trade_rows: List[Dict[str, Any]],
        reason: str,
    ) -> Tuple[float, float, float]:
        fee_long = 0.0
        fee_short = 0.0
        for c, sh in list(shares.items()):
            if sh == 0.0:
                continue
            if not tradable.get(c, False):
                continue
            cash, fee = self._trade_to_target(dt, c, 0.0, shares, cash, px_exec, trade_rows, reason)
            if c in long_codes:
                fee_long += fee
            else:
                fee_short += fee
        return cash, fee_long, fee_short

    def _trade_to_target(
        self,
        dt: pd.Timestamp,
        code: str,
        target_shares: float,
        shares: Dict[str, float],
        cash: float,
        px_exec: Dict[str, float],
        trade_rows: List[Dict[str, Any]],
        reason: str,
    ) -> Tuple[float, float]:
        px = float(px_exec.get(code, np.nan))
        if not np.isfinite(px) or px <= 0:
            return cash, 0.0

        cur = float(shares.get(code, 0.0))
        delta = float(target_shares - cur)
        if abs(delta) < 1e-10:
            return cash, 0.0

        slip = float(self.config.costs.slippage)
        fee_rate = float(self.config.costs.commission_rate)

        if delta > 0:  # buy / cover
            px_trade = px * (1.0 + slip)
            notional = delta * px_trade
            fee = notional * fee_rate
            cash -= (notional + fee)
            action = "BUY" if cur >= 0 else "COVER"
        else:  # sell / short
            qty = abs(delta)
            px_trade = px * (1.0 - slip)
            notional = qty * px_trade
            fee = notional * fee_rate
            cash += (notional - fee)
            action = "SELL" if cur > 0 else "SHORT"

        shares[code] = cur + delta
        trade_rows.append(
            {
                "date": str(dt.date()),
                "code": code,
                "action": action,
                "delta_shares": delta,
                "target_shares": target_shares,
                "price": px_trade,
                "notional": notional,
                "fee": fee,
                "reason": reason,
            }
        )
        return cash, fee

    def _append_pos_rows(
        self,
        out: List[Dict[str, Any]],
        dt: pd.Timestamp,
        shares: Dict[str, float],
        prices: Dict[str, float],
        tradable: Dict[str, bool],
        aligned: Dict[str, pd.DataFrame],
    ) -> None:
        for c, sh in shares.items():
            px = prices.get(c, np.nan)
            out.append(
                {
                    "date": str(dt.date()),
                    "code": c,
                    "shares": float(sh),
                    "price": float(px) if np.isfinite(px) else np.nan,
                    "value": float(sh * px) if np.isfinite(px) else np.nan,
                    "side": "long" if sh > 0 else ("short" if sh < 0 else "flat"),
                    "tradable": bool(tradable.get(c, False)),
                    "suspended": bool(aligned[c].at[dt, "suspended"]),
                    "missing_streak": int(aligned[c].at[dt, "missing_streak"]),
                }
            )


def clone_config_with_borrow_rate(config: StrategyConfig, borrow_rate: float) -> StrategyConfig:
    """Create a deep copy with modified borrow rate for sensitivity tests."""
    cfg = copy.deepcopy(config)
    cfg.costs.short_borrow_rate = float(borrow_rate)
    return cfg
