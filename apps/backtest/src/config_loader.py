"""Configuration and universe loading/validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


class ConfigError(ValueError):
    """Raised when config/universe content is invalid."""


@dataclass
class PositionConfig:
    """Single long/short position configuration."""

    code: str
    weight: float


@dataclass
class BacktestWindow:
    """Backtest start/end window."""

    start_date: date
    end_date: date


@dataclass
class CapitalConfig:
    """Capital and exposure configuration."""

    total_rmb: float
    rmb_to_hkd_rate: float
    long_pct: float
    short_pct: float
    cash_buffer_pct: float

    @property
    def total_hkd(self) -> float:
        """Total capital in HKD."""
        return self.total_rmb * self.rmb_to_hkd_rate


@dataclass
class RebalanceConfig:
    """Rebalance settings."""

    frequency: str
    day: int


@dataclass
class CostConfig:
    """Transaction and short borrow costs."""

    commission_rate: float
    slippage: float
    short_borrow_rate: float


@dataclass
class StopLossConfig:
    """Stop-loss settings."""

    single_long_stop: float
    single_long_action: str
    single_short_stop: float
    single_short_action: str
    portfolio_stop: float
    portfolio_action: str


@dataclass
class SensitivityConfig:
    """Sensitivity-analysis parameters."""

    borrow_rates: List[float] = field(default_factory=list)


@dataclass
class EventMarker:
    """Optional timeline event marker."""

    date: date
    label: str


@dataclass
class StrategyConfig:
    """Whole strategy configuration."""

    strategy_name: str
    description: str
    sector: Optional[str]
    backtest: BacktestWindow
    capital: CapitalConfig
    long_positions: List[PositionConfig]
    short_positions: List[PositionConfig]
    weighting_mode: str
    rebalance: RebalanceConfig
    costs: CostConfig
    stop_loss: StopLossConfig
    sensitivity: SensitivityConfig
    events: List[EventMarker] = field(default_factory=list)


@dataclass
class UniverseStock:
    """Single stock record in universe."""

    code: str
    name: str
    tags: List[str] = field(default_factory=list)


@dataclass
class UniverseGroup:
    """Stock group under a sector."""

    key: str
    name: str
    stocks: List[UniverseStock] = field(default_factory=list)


@dataclass
class UniverseSector:
    """Sector definition."""

    key: str
    name: str
    description: str
    sector_benchmark: Optional[str]
    groups: List[UniverseGroup] = field(default_factory=list)


@dataclass
class UniverseConfig:
    """Universe config root."""

    benchmarks: List[Dict[str, str]] = field(default_factory=list)
    sectors: List[UniverseSector] = field(default_factory=list)

    def all_stock_codes(self) -> List[str]:
        """Return all stock codes from all sectors/groups."""
        out: List[str] = []
        for sector in self.sectors:
            for group in sector.groups:
                for stock in group.stocks:
                    out.append(stock.code)
        return out

    def to_raw(self) -> Dict[str, Any]:
        """Serialize back to raw dict for YAML writing."""
        sectors_out: Dict[str, Any] = {}
        for sec in self.sectors:
            groups_out: Dict[str, Any] = {}
            for grp in sec.groups:
                groups_out[grp.key] = {
                    "name": grp.name,
                    "stocks": [
                        {"code": s.code, "name": s.name, "tags": s.tags} for s in grp.stocks
                    ],
                }
            sectors_out[sec.key] = {
                "name": sec.name,
                "description": sec.description,
                "sector_benchmark": sec.sector_benchmark,
                "groups": groups_out,
            }
        return {"benchmarks": self.benchmarks, "sectors": sectors_out}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 解析失败: {path} -> {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置根节点必须是字典: {path}")
    return data


def _parse_date(v: Any) -> date:
    text = str(v).strip().lower()
    if text == "today":
        return date.today()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ConfigError(f"日期格式错误: {v}, 需要 YYYY-MM-DD 或 today") from exc


def _normalize_code(code: str) -> str:
    return str(code).strip().upper()


def _sum_weights(items: Iterable[PositionConfig]) -> float:
    return sum(float(i.weight) for i in items)


def _assert_close_1(value: float, name: str, tol: float = 1e-6) -> None:
    if abs(value - 1.0) > tol:
        raise ConfigError(f"{name} 必须等于 1.0，当前为 {value:.6f}")


def load_universe(universe_path: Path) -> UniverseConfig:
    """Load and validate universe YAML."""
    raw = _load_yaml(universe_path)
    benches = raw.get("benchmarks", [])
    sectors_raw = raw.get("sectors", {})

    if not isinstance(benches, list):
        raise ConfigError("benchmarks 必须是列表")
    if not isinstance(sectors_raw, dict):
        raise ConfigError("sectors 必须是字典")

    sectors: List[UniverseSector] = []
    seen_codes: Dict[str, str] = {}

    for sec_key, sec_obj in sectors_raw.items():
        if not isinstance(sec_obj, dict):
            raise ConfigError(f"sector {sec_key} 必须是字典")
        groups_raw = sec_obj.get("groups", {})
        if not isinstance(groups_raw, dict):
            raise ConfigError(f"sector {sec_key}.groups 必须是字典")

        groups: List[UniverseGroup] = []
        for grp_key, grp_obj in groups_raw.items():
            if not isinstance(grp_obj, dict):
                raise ConfigError(f"group {sec_key}/{grp_key} 必须是字典")
            stocks_raw = grp_obj.get("stocks", [])
            if not isinstance(stocks_raw, list):
                raise ConfigError(f"group {sec_key}/{grp_key}.stocks 必须是列表")

            stocks: List[UniverseStock] = []
            for st in stocks_raw:
                if not isinstance(st, dict):
                    raise ConfigError(f"group {sec_key}/{grp_key} 存在非法 stock 记录")
                code = _normalize_code(st.get("code", ""))
                name = str(st.get("name", "")).strip()
                if not code or not name:
                    raise ConfigError(f"group {sec_key}/{grp_key} 存在空 code/name")
                tags = st.get("tags", [])
                if tags is None:
                    tags = []
                if not isinstance(tags, list):
                    raise ConfigError(f"{code} tags 必须是列表")
                if code in seen_codes:
                    raise ConfigError(f"股票代码重复: {code} 同时出现在 {seen_codes[code]} 与 {sec_key}/{grp_key}")
                seen_codes[code] = f"{sec_key}/{grp_key}"
                stocks.append(UniverseStock(code=code, name=name, tags=[str(t) for t in tags]))

            groups.append(
                UniverseGroup(
                    key=str(grp_key),
                    name=str(grp_obj.get("name", grp_key)).strip(),
                    stocks=stocks,
                )
            )

        sectors.append(
            UniverseSector(
                key=str(sec_key),
                name=str(sec_obj.get("name", sec_key)).strip(),
                description=str(sec_obj.get("description", "")).strip(),
                sector_benchmark=(
                    str(sec_obj.get("sector_benchmark", "")).strip() or None
                ),
                groups=groups,
            )
        )

    return UniverseConfig(benchmarks=benches, sectors=sectors)


def save_universe(universe: UniverseConfig, universe_path: Path) -> None:
    """Save universe config back to YAML file."""
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with universe_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            universe.to_raw(),
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def load_strategy(strategy_path: Path, universe: UniverseConfig) -> StrategyConfig:
    """Load and validate strategy YAML."""
    raw = _load_yaml(strategy_path)

    strategy_name = str(raw.get("strategy_name", "")).strip()
    if not strategy_name:
        raise ConfigError("strategy_name 不能为空")

    back_raw = raw.get("backtest", {}) or {}
    start = _parse_date(back_raw.get("start_date", ""))
    end = _parse_date(back_raw.get("end_date", "today"))
    if start >= end:
        raise ConfigError("backtest.start_date 必须早于 end_date")

    cap_raw = raw.get("capital", {}) or {}
    capital = CapitalConfig(
        total_rmb=float(cap_raw.get("total", 0.0)),
        rmb_to_hkd_rate=float(cap_raw.get("rmb_to_hkd_rate", 1.0)),
        long_pct=float(cap_raw.get("long_pct", 0.0)),
        short_pct=float(cap_raw.get("short_pct", 0.0)),
        cash_buffer_pct=float(cap_raw.get("cash_buffer_pct", 0.0)),
    )
    _assert_close_1(
        capital.long_pct + capital.short_pct + capital.cash_buffer_pct,
        "capital.long_pct + short_pct + cash_buffer_pct",
    )

    long_raw = raw.get("long_positions", []) or []
    short_raw = raw.get("short_positions", []) or []
    if not isinstance(long_raw, list) or not isinstance(short_raw, list):
        raise ConfigError("long_positions/short_positions 必须是列表")

    long_positions = [
        PositionConfig(code=_normalize_code(x["code"]), weight=float(x["weight"])) for x in long_raw
    ]
    short_positions = [
        PositionConfig(code=_normalize_code(x["code"]), weight=float(x["weight"])) for x in short_raw
    ]
    if not long_positions:
        raise ConfigError("long_positions 不能为空")
    if not short_positions:
        raise ConfigError("short_positions 不能为空")

    _assert_close_1(_sum_weights(long_positions), "long_positions 权重和")
    _assert_close_1(_sum_weights(short_positions), "short_positions 权重和")

    all_universe_codes = set(universe.all_stock_codes())
    for p in [*long_positions, *short_positions]:
        if p.code not in all_universe_codes:
            raise ConfigError(f"策略使用了不在 universe 中的代码: {p.code}")

    reb_raw = raw.get("rebalance", {}) or {}
    rebalance = RebalanceConfig(
        frequency=str(reb_raw.get("frequency", "monthly")).lower(),
        day=int(reb_raw.get("day", 1)),
    )

    if rebalance.frequency not in {"daily", "weekly", "monthly", "quarterly"}:
        raise ConfigError("rebalance.frequency 仅支持 daily/weekly/monthly/quarterly")

    costs_raw = raw.get("costs", {}) or {}
    costs = CostConfig(
        commission_rate=float(costs_raw.get("commission_rate", 0.0015)),
        slippage=float(costs_raw.get("slippage", 0.001)),
        short_borrow_rate=float(costs_raw.get("short_borrow_rate", 0.15)),
    )

    sl_raw = raw.get("stop_loss", {}) or {}
    stop_loss = StopLossConfig(
        single_long_stop=float(sl_raw.get("single_long_stop", -0.15)),
        single_long_action=str(sl_raw.get("single_long_action", "halve")).lower(),
        single_short_stop=float(sl_raw.get("single_short_stop", 0.20)),
        single_short_action=str(sl_raw.get("single_short_action", "close")).lower(),
        portfolio_stop=float(sl_raw.get("portfolio_stop", -0.10)),
        portfolio_action=str(sl_raw.get("portfolio_action", "close_all")).lower(),
    )

    sens_raw = raw.get("sensitivity", {}) or {}
    sensitivity = SensitivityConfig(
        borrow_rates=[float(x) for x in sens_raw.get("borrow_rates", [0.15])]
    )

    events: List[EventMarker] = []
    for ev in raw.get("events", []) or []:
        if isinstance(ev, dict) and ev.get("date") and ev.get("label"):
            events.append(EventMarker(date=_parse_date(ev["date"]), label=str(ev["label"])))

    return StrategyConfig(
        strategy_name=strategy_name,
        description=str(raw.get("description", "")).strip(),
        sector=(str(raw.get("sector", "")).strip() or None),
        backtest=BacktestWindow(start_date=start, end_date=end),
        capital=capital,
        long_positions=long_positions,
        short_positions=short_positions,
        weighting_mode=str(raw.get("weighting_mode", "manual")).lower(),
        rebalance=rebalance,
        costs=costs,
        stop_loss=stop_loss,
        sensitivity=sensitivity,
        events=events,
    )
