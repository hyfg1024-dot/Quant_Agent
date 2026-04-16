"""Backtrader-backed long-short engine with YAML-config bridge."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import backtrader as bt
import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult
from src.config_loader import StrategyConfig
from src.data_manager import DataManager


DEFAULT_BOARD_LOTS: Dict[str, int] = {
    "1109.HK": 2000,
    "0688.HK": 1000,
    "0960.HK": 1000,
    "2007.HK": 1000,
    "1918.HK": 1000,
    "1908.HK": 1000,
    "3383.HK": 1000,
    "3377.HK": 1000,
    "3883.HK": 1000,
    "0883.HK": 1000,
    "1088.HK": 500,
    "0941.HK": 500,
    "0939.HK": 1000,
    "9866.HK": 100,
    "9868.HK": 100,
    "0241.HK": 1000,
}


class HKPandasData(bt.feeds.PandasData):
    """Pandas data feed with tradable/suspended lines."""

    lines = ("tradable", "suspended", "missing_streak")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("tradable", "tradable"),
        ("suspended", "suspended"),
        ("missing_streak", "missing_streak"),
    )


class HKCostCommissionInfo(bt.CommInfoBase):
    """Custom commission/balance cost model for HK long-short."""

    params = (
        ("commission", 0.0015),
        ("borrow_rate", 0.15),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(float(size)) * float(price) * float(self.p.commission)

    def daily_borrow_fee(self, size: float, price: float) -> float:
        if size >= 0:
            return 0.0
        return abs(float(size)) * float(price) * float(self.p.borrow_rate) / 252.0


class VolumeShareFiller:
    """Fill at most X% of current bar volume to simulate partial fills."""

    def __init__(self, volume_pct: float = 0.2) -> None:
        self.volume_pct = float(max(0.01, min(1.0, volume_pct)))

    def __call__(self, order: bt.Order, price: float, ago: int) -> int:
        vol = float(order.data.volume[ago]) if len(order.data.volume) > 0 else 0.0
        if not np.isfinite(vol) or vol <= 0:
            return 0
        remaining = abs(float(order.created.size) - float(order.executed.size))
        if remaining <= 0:
            return 0
        cap = max(1, int(vol * self.volume_pct))
        return int(min(remaining, float(cap)))


@dataclass
class PreparedInput:
    """Preloaded input data used by runner."""

    calendar: pd.DatetimeIndex
    aligned_map: Dict[str, pd.DataFrame]
    benchmark_nav: Dict[str, pd.Series]
    long_weights: Dict[str, float]
    short_weights: Dict[str, float]
    board_lots: Dict[str, int]
    warnings: List[str]


class ConfigurableLongShortStrategy(bt.Strategy):
    """General long-short strategy reading config params from YAML bridge."""

    params = (
        ("long_weights", None),
        ("short_weights", None),
        ("board_lots", None),
        ("rebalance_dates", None),
        ("capital_total_hkd", 1_000_000.0),
        ("long_pct", 0.5),
        ("short_pct", 0.2),
        ("single_long_stop", -0.15),
        ("single_long_action", "halve"),
        ("single_short_stop", 0.2),
        ("single_short_action", "close"),
        ("portfolio_stop", -0.10),
        ("portfolio_action", "close_all"),
    )

    def __init__(self) -> None:
        self.long_weights = dict(self.p.long_weights or {})
        self.short_weights = dict(self.p.short_weights or {})
        self.board_lots = {str(k): max(1, int(v)) for k, v in dict(self.p.board_lots or {}).items()}
        self.rebalance_dates = set(self.p.rebalance_dates or [])

        self.code2data: Dict[str, bt.LineSeries] = {str(d._name): d for d in self.datas}
        self.long_codes = set(self.long_weights.keys())
        self.short_codes = set(self.short_weights.keys())
        self.short_enabled: Dict[str, bool] = {c: True for c in self.short_codes}

        self.initial_value = float(self.broker.getvalue())
        self.entry_price: Dict[str, float] = {c: np.nan for c in self.code2data}
        self.long_stop_triggered: Set[str] = set()
        self.short_stop_triggered: Set[str] = set()
        self.terminated = False
        self.inited = False

        self.trade_rows: List[Dict[str, Any]] = []
        self.position_rows: List[Dict[str, Any]] = []
        self.stop_events: List[Dict[str, Any]] = []
        self.warning_rows: List[str] = []
        self.daily_nav_rows: List[Dict[str, Any]] = []
        self.daily_cost_rows: List[Dict[str, Any]] = []

        self.pending_meta: Dict[int, Dict[str, Any]] = {}
        self.trade_fee_long = 0.0
        self.trade_fee_short = 0.0
        self.borrow_fee_cum = 0.0
        self.day_trade_fee = 0.0
        self.day_borrow_fee = 0.0

        self.long_pnl_cum = 0.0
        self.short_pnl_cum = 0.0
        self.code_contrib: Dict[str, float] = {c: 0.0 for c in self.code2data}
        self.prev_close: Dict[str, float] = {c: np.nan for c in self.code2data}
        self.prev_size: Dict[str, float] = {c: 0.0 for c in self.code2data}
        self.last_dt: Optional[date] = None

        comm = self.broker.getcommissioninfo(self.datas[0]) if self.datas else None
        self.cost_info = comm if isinstance(comm, HKCostCommissionInfo) else HKCostCommissionInfo()

    def next(self) -> None:
        if not self.datas:
            return
        dt = bt.num2date(self.datas[0].datetime[0]).date()
        if self.last_dt == dt:
            return

        self.day_trade_fee = 0.0
        self.day_borrow_fee = 0.0

        # Daily MTM PnL contribution (yesterday close -> today close).
        if self.inited:
            for code, data in self.code2data.items():
                p0 = self.prev_close.get(code, np.nan)
                p1 = float(data.close[0]) if np.isfinite(float(data.close[0])) else np.nan
                sh = float(self.prev_size.get(code, 0.0))
                if np.isfinite(p0) and np.isfinite(p1):
                    pnl = sh * (p1 - p0)
                    self.code_contrib[code] += pnl
                    if code in self.long_codes:
                        self.long_pnl_cum += pnl
                    elif code in self.short_codes:
                        self.short_pnl_cum += pnl

        # Borrow fee accrual.
        borrow_fee = 0.0
        for code in self.short_codes:
            data = self.code2data.get(code)
            if data is None:
                continue
            pos = self.getposition(data)
            px = float(data.close[0]) if np.isfinite(float(data.close[0])) else np.nan
            if np.isfinite(px):
                borrow_fee += self.cost_info.daily_borrow_fee(float(pos.size), px)
        if borrow_fee > 0:
            self.broker.add_cash(-borrow_fee)
            self.borrow_fee_cum += borrow_fee
            self.day_borrow_fee = borrow_fee

        if not self.terminated:
            self._check_single_name_stops(dt)
            self._check_portfolio_stop(dt)

        # Rebalance on first bar and schedule.
        if (not self.terminated) and ((not self.inited) or (dt in self.rebalance_dates)):
            self.long_stop_triggered.clear()
            reason = "init" if not self.inited else "rebalance"
            self._rebalance(reason=reason)
            self.inited = True

        self._snapshot_day(dt)
        self.last_dt = dt

    def notify_order(self, order: bt.Order) -> None:
        if order.status in [order.Submitted, order.Accepted]:
            return
        meta = self.pending_meta.get(int(order.ref), {})
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            code = str(order.data._name)
            self.warning_rows.append(f"{code} 订单失败: status={order.getstatusname()} reason={meta.get('reason','')}")
            self.pending_meta.pop(int(order.ref), None)
            return
        if order.status != order.Completed:
            return

        code = str(order.data._name)
        exec_size = float(order.executed.size)
        exec_price = float(order.executed.price)
        exec_comm = float(order.executed.comm)
        notional = abs(exec_size) * exec_price
        prev_size = float(meta.get("prev_size", 0.0))
        reason = str(meta.get("reason", "rebalance"))
        side = str(meta.get("side", "long"))

        if exec_size > 0 and prev_size >= 0:
            action = "BUY"
        elif exec_size > 0 and prev_size < 0:
            action = "COVER"
        elif exec_size < 0 and prev_size > 0:
            action = "SELL"
        else:
            action = "SHORT"

        if side == "short":
            self.trade_fee_short += exec_comm
        else:
            self.trade_fee_long += exec_comm
        self.day_trade_fee += exec_comm

        # refresh entry anchor after position update
        pos = self.getposition(order.data)
        if abs(float(pos.size)) > 0 and np.isfinite(exec_price):
            self.entry_price[code] = exec_price

        dt = bt.num2date(order.data.datetime[0]).date()
        self.trade_rows.append(
            {
                "date": str(dt),
                "code": code,
                "action": action,
                "delta_shares": exec_size,
                "target_shares": float(meta.get("target_shares", prev_size + exec_size)),
                "board_lot": int(meta.get("board_lot", self._lot(code))),
                "price": exec_price,
                "notional": notional,
                "fee": exec_comm,
                "reason": reason,
            }
        )
        self.pending_meta.pop(int(order.ref), None)

    def _snapshot_day(self, dt: date) -> None:
        long_mv = 0.0
        short_mv = 0.0
        for code, data in self.code2data.items():
            pos = self.getposition(data)
            px = float(data.close[0]) if np.isfinite(float(data.close[0])) else np.nan
            v = float(pos.size) * px if np.isfinite(px) else np.nan
            if np.isfinite(v):
                if pos.size >= 0:
                    long_mv += v
                else:
                    short_mv += abs(v)
            self.position_rows.append(
                {
                    "date": str(dt),
                    "code": code,
                    "shares": float(pos.size),
                    "price": float(px) if np.isfinite(px) else np.nan,
                    "value": float(v) if np.isfinite(v) else np.nan,
                    "side": "long" if pos.size > 0 else ("short" if pos.size < 0 else "flat"),
                    "tradable": bool(int(data.tradable[0])) if np.isfinite(float(data.tradable[0])) else False,
                    "suspended": bool(int(data.suspended[0])) if np.isfinite(float(data.suspended[0])) else False,
                    "missing_streak": int(float(data.missing_streak[0])) if np.isfinite(float(data.missing_streak[0])) else 0,
                }
            )
            self.prev_close[code] = float(px) if np.isfinite(px) else np.nan
            self.prev_size[code] = float(pos.size)

        portfolio = float(self.broker.getvalue())
        long_nav = float(self.p.capital_total_hkd * self.p.long_pct + self.long_pnl_cum - self.trade_fee_long)
        short_nav = float(self.p.capital_total_hkd * self.p.short_pct + self.short_pnl_cum - self.trade_fee_short - self.borrow_fee_cum)
        self.daily_nav_rows.append({"date": pd.Timestamp(dt), "portfolio": portfolio, "long": long_nav, "short": short_nav})

        cum_trade_fee = self.trade_fee_long + self.trade_fee_short
        self.daily_cost_rows.append(
            {
                "date": pd.Timestamp(dt),
                "trade_fee": float(self.day_trade_fee),
                "borrow_fee": float(self.day_borrow_fee),
                "total_fee": float(self.day_trade_fee + self.day_borrow_fee),
                "cum_trade_fee": float(cum_trade_fee),
                "cum_borrow_fee": float(self.borrow_fee_cum),
                "cum_total_fee": float(cum_trade_fee + self.borrow_fee_cum),
            }
        )

    def _rebalance(self, reason: str) -> None:
        equity_now = float(self.broker.getvalue())
        long_cap = equity_now * float(self.p.long_pct)
        short_cap = equity_now * float(self.p.short_pct)

        # Long target
        for code, w in self.long_weights.items():
            data = self.code2data.get(code)
            if data is None or (not self._tradable(data)):
                continue
            px = float(data.close[0])
            if not np.isfinite(px) or px <= 0:
                continue
            target = (long_cap * float(w)) / px
            self._order_to_target(code=code, target_shares=target, reason=reason, side="long")

        # Short target (disable names keep current size).
        enabled = [c for c in self.short_weights.keys() if self.short_enabled.get(c, True)]
        if enabled:
            w_sum = sum(float(self.short_weights[c]) for c in enabled)
            if w_sum <= 0:
                w_sum = float(len(enabled))
            for code in enabled:
                data = self.code2data.get(code)
                if data is None or (not self._tradable(data)):
                    continue
                px = float(data.close[0])
                if not np.isfinite(px) or px <= 0:
                    continue
                w = float(self.short_weights[code]) / float(w_sum)
                target = -(short_cap * w) / px
                self._order_to_target(code=code, target_shares=target, reason=reason, side="short")

    def _check_single_name_stops(self, dt: date) -> None:
        for code in sorted(self.long_codes):
            if code in self.long_stop_triggered:
                continue
            data = self.code2data.get(code)
            if data is None:
                continue
            pos = self.getposition(data)
            if float(pos.size) <= 0:
                continue
            px = float(data.close[0])
            ent = float(self.entry_price.get(code, np.nan))
            if not np.isfinite(px) or not np.isfinite(ent) or ent <= 0:
                continue
            ret = px / ent - 1.0
            if ret <= float(self.p.single_long_stop) and self._tradable(data):
                action = str(self.p.single_long_action).lower()
                target = float(pos.size) * 0.5 if action == "halve" else 0.0
                self._order_to_target(code=code, target_shares=target, reason="single_long_stop", side="long")
                self.long_stop_triggered.add(code)
                self.stop_events.append({"date": str(dt), "code": code, "side": "long", "trigger": ret, "action": action})

        for code in sorted(self.short_codes):
            if code in self.short_stop_triggered:
                continue
            if not self.short_enabled.get(code, True):
                continue
            data = self.code2data.get(code)
            if data is None:
                continue
            pos = self.getposition(data)
            if float(pos.size) >= 0:
                continue
            px = float(data.close[0])
            ent = float(self.entry_price.get(code, np.nan))
            if not np.isfinite(px) or not np.isfinite(ent) or ent <= 0:
                continue
            adverse = px / ent - 1.0
            if adverse >= float(self.p.single_short_stop) and self._tradable(data):
                action = str(self.p.single_short_action).lower()
                if action == "close":
                    self._order_to_target(code=code, target_shares=0.0, reason="single_short_stop", side="short")
                    self.short_enabled[code] = False
                self.short_stop_triggered.add(code)
                self.stop_events.append({"date": str(dt), "code": code, "side": "short", "trigger": adverse, "action": action})

    def _check_portfolio_stop(self, dt: date) -> None:
        ret = float(self.broker.getvalue()) / float(self.initial_value) - 1.0
        if ret > float(self.p.portfolio_stop):
            return
        action = str(self.p.portfolio_action).lower()
        if action != "close_all":
            return
        for code, data in self.code2data.items():
            if not self._tradable(data):
                continue
            pos = self.getposition(data)
            if abs(float(pos.size)) <= 1e-10:
                continue
            side = "short" if code in self.short_codes else "long"
            self._order_to_target(code=code, target_shares=0.0, reason="portfolio_stop", side=side)
        self.terminated = True
        self.stop_events.append({"date": str(dt), "code": "ALL", "side": "portfolio", "trigger": ret, "action": action})

    def _order_to_target(self, code: str, target_shares: float, reason: str, side: str) -> None:
        data = self.code2data.get(code)
        if data is None:
            return
        cur = float(self.getposition(data).size)
        lot = self._lot(code)
        target = self._round_to_lot(float(target_shares), lot)
        delta = float(target - cur)
        if abs(delta) < 1e-10:
            return
        if delta > 0:
            o = self.buy(data=data, size=abs(delta))
        else:
            o = self.sell(data=data, size=abs(delta))
        self.pending_meta[int(o.ref)] = {
            "code": code,
            "reason": reason,
            "side": side,
            "target_shares": target,
            "prev_size": cur,
            "board_lot": lot,
        }

    def _lot(self, code: str) -> int:
        return int(self.board_lots.get(code, DEFAULT_BOARD_LOTS.get(code, 1000)))

    def _round_to_lot(self, target_shares: float, lot: int) -> float:
        q = abs(float(target_shares))
        if q < 1e-10:
            return 0.0
        units = int(q // float(lot))
        if units <= 0:
            return 0.0
        qty = float(units * lot)
        return qty if target_shares >= 0 else -qty

    def _tradable(self, data: bt.LineSeries) -> bool:
        tradable = bool(int(data.tradable[0])) if np.isfinite(float(data.tradable[0])) else False
        suspended = bool(int(data.suspended[0])) if np.isfinite(float(data.suspended[0])) else False
        return tradable and (not suspended)


def run_backtrader_backtest(
    config: StrategyConfig,
    data_manager: DataManager,
    benchmark_codes: Optional[List[str]] = None,
    logger=print,
    board_lot_overrides: Optional[Dict[str, int]] = None,
    volume_fill_pct: float = 0.2,
) -> BacktestResult:
    """Run one backtest via Backtrader and return legacy-compatible result."""
    prepared = _prepare_inputs(
        config=config,
        data_manager=data_manager,
        benchmark_codes=benchmark_codes or ["^HSI"],
        logger=logger,
        board_lot_overrides=board_lot_overrides,
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(config.capital.total_hkd))
    cerebro.broker.set_coc(True)  # close-on-close execution for deterministic daily rebalance
    cerebro.broker.set_slippage_perc(float(config.costs.slippage), slip_open=True, slip_match=True)
    if hasattr(cerebro.broker, "set_filler"):
        cerebro.broker.set_filler(VolumeShareFiller(volume_pct=float(volume_fill_pct)))

    commission = HKCostCommissionInfo(
        commission=float(config.costs.commission_rate),
        borrow_rate=float(config.costs.short_borrow_rate),
    )
    cerebro.broker.addcommissioninfo(commission)

    for code, aligned in prepared.aligned_map.items():
        df_feed = aligned.copy()
        df_feed["close"] = pd.to_numeric(df_feed["adj_close"], errors="coerce")
        df_feed = df_feed[["open", "high", "low", "close", "volume", "tradable", "suspended", "missing_streak"]]
        data = HKPandasData(dataname=df_feed, name=code)
        cerebro.adddata(data, name=code)

    rebalance_dates = _get_rebalance_dates(prepared.calendar, config.rebalance.frequency, config.rebalance.day)
    cerebro.addstrategy(
        ConfigurableLongShortStrategy,
        long_weights=prepared.long_weights,
        short_weights=prepared.short_weights,
        board_lots=prepared.board_lots,
        rebalance_dates=rebalance_dates,
        capital_total_hkd=float(config.capital.total_hkd),
        long_pct=float(config.capital.long_pct),
        short_pct=float(config.capital.short_pct),
        single_long_stop=float(config.stop_loss.single_long_stop),
        single_long_action=str(config.stop_loss.single_long_action),
        single_short_stop=float(config.stop_loss.single_short_stop),
        single_short_action=str(config.stop_loss.single_short_action),
        portfolio_stop=float(config.stop_loss.portfolio_stop),
        portfolio_action=str(config.stop_loss.portfolio_action),
    )

    out = cerebro.run()
    if not out:
        raise RuntimeError("Backtrader 未返回策略实例")
    strat: ConfigurableLongShortStrategy = out[0]

    nav_df = pd.DataFrame(strat.daily_nav_rows)
    if nav_df.empty:
        raise RuntimeError("Backtrader 回测结果为空")
    nav_df = nav_df.set_index("date").sort_index()

    daily_costs = pd.DataFrame(strat.daily_cost_rows)
    if not daily_costs.empty:
        daily_costs = daily_costs.set_index("date").sort_index()

    costs = {
        "commission_long": float(strat.trade_fee_long),
        "commission_short": float(strat.trade_fee_short),
        "borrow_fee": float(strat.borrow_fee_cum),
        "total": float(strat.trade_fee_long + strat.trade_fee_short + strat.borrow_fee_cum),
    }

    warnings = list(prepared.warnings) + list(strat.warning_rows)
    code_contrib = pd.Series(strat.code_contrib).sort_values(ascending=False)

    return BacktestResult(
        daily_portfolio_value=nav_df["portfolio"],
        daily_long_value=nav_df["long"],
        daily_short_value=nav_df["short"],
        daily_positions=pd.DataFrame(strat.position_rows),
        trades=pd.DataFrame(strat.trade_rows),
        costs=costs,
        stop_loss_events=list(strat.stop_events),
        benchmark_nav=prepared.benchmark_nav,
        code_contribution=code_contrib,
        daily_costs=daily_costs if isinstance(daily_costs, pd.DataFrame) else pd.DataFrame(),
        warnings=warnings,
    )


def run_backtrader_sensitivity(
    config: StrategyConfig,
    data_manager: DataManager,
    benchmark_codes: Optional[List[str]],
    borrow_rates: List[float],
    logger=print,
    board_lot_overrides: Optional[Dict[str, int]] = None,
    volume_fill_pct: float = 0.2,
) -> pd.DataFrame:
    """Run borrow-rate sensitivity with Backtrader engine."""
    rows: List[Dict[str, Any]] = []
    for rate in borrow_rates:
        cfg = copy.deepcopy(config)
        cfg.costs.short_borrow_rate = float(rate)
        result = run_backtrader_backtest(
            config=cfg,
            data_manager=data_manager,
            benchmark_codes=benchmark_codes,
            logger=lambda *_args, **_kwargs: None,
            board_lot_overrides=board_lot_overrides,
            volume_fill_pct=volume_fill_pct,
        )
        s = pd.to_numeric(result.daily_portfolio_value, errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "borrow_rate": float(rate),
                "final_value": float(s.iloc[-1]),
                "cumulative_return": float(s.iloc[-1] / s.iloc[0] - 1.0),
                "total_cost": float(result.costs.get("total", 0.0)),
                "borrow_cost": float(result.costs.get("borrow_fee", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["borrow_rate", "final_value", "cumulative_return", "total_cost", "borrow_cost"])
    return pd.DataFrame(rows).sort_values("borrow_rate").reset_index(drop=True)


def _prepare_inputs(
    config: StrategyConfig,
    data_manager: DataManager,
    benchmark_codes: List[str],
    logger=print,
    board_lot_overrides: Optional[Dict[str, int]] = None,
) -> PreparedInput:
    start = str(config.backtest.start_date)
    end = str(config.backtest.end_date)
    all_codes = sorted({p.code for p in [*config.long_positions, *config.short_positions]})
    warnings: List[str] = []

    raw_map: Dict[str, pd.DataFrame] = {}
    for code in all_codes:
        try:
            df = data_manager.fetch_stock_data(code, start=start, end=end)
        except Exception as exc:
            warnings.append(f"{code} 数据拉取失败，已跳过: {exc}")
            continue
        if df.empty:
            warnings.append(f"{code} 在回测区间内无数据，已跳过")
            continue
        raw_map[code] = df
        cov_start = df.index.min().date()
        cov_end = df.index.max().date()
        if cov_start > config.backtest.start_date or cov_end < config.backtest.end_date:
            warnings.append(f"{code} 数据覆盖不完整: {cov_start} -> {cov_end}")

    active_codes = sorted(raw_map.keys())
    if not active_codes:
        raise RuntimeError("无可用标的数据，无法回测")

    bench_raw: Dict[str, pd.DataFrame] = {}
    for b in benchmark_codes:
        try:
            bdf = data_manager.fetch_index_data(b, start=start, end=end)
            if not bdf.empty:
                bench_raw[b] = bdf
        except Exception as exc:
            warnings.append(f"基准 {b} 拉取失败: {exc}")

    calendar = _build_calendar(config=config, prices=raw_map, benchmarks=bench_raw)
    if calendar.empty:
        raise RuntimeError("回测交易日历为空")

    aligned_map: Dict[str, pd.DataFrame] = {}
    for code in active_codes:
        prepared = data_manager.prepare_for_calendar(raw_map[code], calendar=calendar, max_suspend_days=30)
        aligned_map[code] = prepared.aligned
        if bool((prepared.aligned["suspended"]).any()):
            max_streak = int(prepared.aligned["missing_streak"].max())
            warnings.append(f"{code} 存在长期停牌/缺失，最长连续缺失 {max_streak} 天（>30天期间冻结交易）")

    benchmark_nav = _build_benchmark_nav(bench_raw, calendar)
    long_weights, short_weights = _resolve_weights(config=config, aligned=aligned_map, active_codes=active_codes)

    board_lots = {c: int(DEFAULT_BOARD_LOTS.get(c, 1000)) for c in active_codes}
    for k, v in dict(board_lot_overrides or {}).items():
        code = str(k).strip().upper()
        if code in board_lots:
            board_lots[code] = max(1, int(v))

    return PreparedInput(
        calendar=calendar,
        aligned_map=aligned_map,
        benchmark_nav=benchmark_nav,
        long_weights=long_weights,
        short_weights=short_weights,
        board_lots=board_lots,
        warnings=warnings,
    )


def _build_calendar(
    config: StrategyConfig,
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
    start = pd.Timestamp(config.backtest.start_date)
    end = pd.Timestamp(config.backtest.end_date)
    return base[(base >= start) & (base <= end)]


def _build_benchmark_nav(bench_raw: Dict[str, pd.DataFrame], calendar: pd.DatetimeIndex) -> Dict[str, pd.Series]:
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
    config: StrategyConfig,
    aligned: Dict[str, pd.DataFrame],
    active_codes: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    long_codes = [p.code for p in config.long_positions if p.code in active_codes]
    short_codes = [p.code for p in config.short_positions if p.code in active_codes]
    mode = config.weighting_mode

    if mode == "equal":
        return (
            ({c: 1.0 / len(long_codes) for c in long_codes} if long_codes else {}),
            ({c: 1.0 / len(short_codes) for c in short_codes} if short_codes else {}),
        )

    if mode == "inverse_volatility":
        def inv_vol(codes: List[str]) -> Dict[str, float]:
            if not codes:
                return {}
            vals: Dict[str, float] = {}
            for c in codes:
                ret = aligned[c]["adj_close"].pct_change().dropna().tail(120)
                vol = float(ret.std()) if not ret.empty else np.nan
                vals[c] = 1.0 / vol if np.isfinite(vol) and vol > 0 else 0.0
            total = sum(vals.values())
            if total <= 0:
                return {c: 1.0 / len(codes) for c in codes}
            return {c: vals[c] / total for c in codes}

        return inv_vol(long_codes), inv_vol(short_codes)

    return (
        {p.code: float(p.weight) for p in config.long_positions if p.code in active_codes},
        {p.code: float(p.weight) for p in config.short_positions if p.code in active_codes},
    )


def _get_rebalance_dates(calendar: pd.DatetimeIndex, frequency: str, day: int) -> Set[date]:
    freq = str(frequency).lower()
    nth = max(1, int(day))
    s = pd.Series(calendar, index=calendar)

    if freq == "daily":
        return {pd.Timestamp(x).date() for x in calendar}
    if freq == "weekly":
        groups = s.groupby([calendar.isocalendar().year, calendar.isocalendar().week])
    elif freq == "quarterly":
        groups = s.groupby([calendar.year, calendar.quarter])
    else:  # monthly
        groups = s.groupby([calendar.year, calendar.month])

    out: Set[date] = set()
    for _, g in groups:
        idx = min(nth - 1, len(g) - 1)
        out.add(pd.Timestamp(g.iloc[idx]).date())
    return out


def parse_board_lot_overrides(strategy_path: Any) -> Dict[str, int]:
    """Read board-lot overrides from YAML strategy file."""
    try:
        import yaml
    except Exception:
        return {}

    try:
        path = strategy_path if isinstance(strategy_path, str) else str(strategy_path)
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, int] = {}
    top = raw.get("board_lots", {})
    if isinstance(top, dict):
        for k, v in top.items():
            try:
                out[str(k).strip().upper()] = max(1, int(v))
            except Exception:
                pass
    pt = raw.get("paper_trade", {})
    if isinstance(pt, dict):
        b2 = pt.get("board_lots", {})
        if isinstance(b2, dict):
            for k, v in b2.items():
                try:
                    out[str(k).strip().upper()] = max(1, int(v))
                except Exception:
                    pass
    return out


def clone_config_with_borrow_rate(config: StrategyConfig, borrow_rate: float) -> StrategyConfig:
    """Clone config with overridden borrow rate."""
    cfg = copy.deepcopy(config)
    cfg.costs.short_borrow_rate = float(borrow_rate)
    return cfg
