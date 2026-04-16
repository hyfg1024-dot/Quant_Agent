"""Entry script for configurable backtests using Backtrader execution layer."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from src.config_loader import ConfigError, load_strategy, load_universe

if TYPE_CHECKING:
    from src.config_loader import StrategyConfig, UniverseConfig
    from src.data_manager import DataManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="港股多空对冲回测系统（Backtrader）")
    parser.add_argument("--config", default="config/strategies/realestate_example.yaml", help="策略配置文件路径")
    parser.add_argument("--universe", default="config/universe.yaml", help="标的池配置文件路径")
    parser.add_argument("--output", default="reports", help="报告输出目录")

    parser.add_argument("--validate-universe", action="store_true", help="仅验证标的池数据可用性")
    parser.add_argument("--update-data-only", action="store_true", help="仅更新数据缓存，不运行回测")
    parser.add_argument("--sector", default="", help="验证或更新时限定某个板块 key")
    parser.add_argument("--start", default="2021-01-01", help="validate/update 模式起始日期")
    parser.add_argument("--end", default="today", help="validate/update 模式结束日期")

    parser.add_argument("--rf-rate", type=float, default=0.03, help="无风险利率，默认 3%%")
    parser.add_argument("--volume-fill-pct", type=float, default=0.20, help="单日成交量可撮合比例，默认 20%%")
    parser.add_argument("--no-browser", action="store_true", help="生成报告后不自动打开浏览器")
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def resolve_end_date(text: str) -> str:
    t = str(text).strip().lower()
    return str(date.today()) if t == "today" else str(pd.Timestamp(t).date())


def get_codes_from_universe(universe: "UniverseConfig", sector: str = "") -> List[str]:
    codes: List[str] = []
    sec_filter = sector.strip()
    for sec in universe.sectors:
        if sec_filter and sec.key != sec_filter:
            continue
        for grp in sec.groups:
            for st in grp.stocks:
                codes.append(st.code)
    return sorted(set(codes))


def get_sector_benchmark(universe: "UniverseConfig", sector_key: Optional[str]) -> Optional[str]:
    if not sector_key:
        return None
    for sec in universe.sectors:
        if sec.key == sector_key:
            return sec.sector_benchmark
    return None


def update_data_cache(dm: "DataManager", universe: "UniverseConfig", start: str, end: str, sector: str = "") -> None:
    codes = get_codes_from_universe(universe, sector)
    print(f"[DATA] 更新股票缓存: {len(codes)} 支")
    for code in codes:
        try:
            df = dm.fetch_stock_data(code, start=start, end=end)
            print(f"  [OK] {code:<10} rows={len(df)}")
        except Exception as exc:
            print(f"  [ERR] {code:<10} {exc}")

    bench_codes = [str(x.get("code", "")).strip() for x in universe.benchmarks if str(x.get("code", "")).strip()]
    bench_codes = sorted(set(bench_codes))
    print(f"[DATA] 更新基准缓存: {bench_codes}")
    for b in bench_codes:
        try:
            df = dm.fetch_index_data(b, start=start, end=end)
            print(f"  [OK] {b:<10} rows={len(df)}")
        except Exception as exc:
            print(f"  [ERR] {b:<10} {exc}")


def validate_universe(dm: "DataManager", universe: "UniverseConfig", start: str, end: str, sector: str = "") -> int:
    codes = get_codes_from_universe(universe, sector)
    print(f"[CHECK] 校验标的数量: {len(codes)}")
    report = dm.validate_universe(codes, start=start, end=end)

    ok = 0
    for code in codes:
        item = report.get(code, {})
        status = str(item.get("status", "unknown"))
        rows = int(item.get("rows", 0) or 0)
        s = str(item.get("start", ""))
        e = str(item.get("end", ""))
        msg = str(item.get("message", ""))
        print(f"{code:<10} | {status:<6} | rows={rows:<5} | {s} -> {e} | {msg}")
        if status == "ok":
            ok += 1
    print(f"[CHECK] 完成: ok={ok}, total={len(codes)}")
    return 0 if ok > 0 else 2


def print_summary(
    cfg: "StrategyConfig",
    summary: pd.DataFrame,
    report_path: Path,
    costs: Dict[str, float],
) -> None:
    print("\n=== 回测完成（Backtrader）===")
    print(f"策略: {cfg.strategy_name}")
    print(f"区间: {cfg.backtest.start_date} -> {cfg.backtest.end_date}")

    if "portfolio" in summary.index:
        p = summary.loc["portfolio"]
        print(f"累计收益: {p.get('cumulative_return', float('nan')):.2%}")
        print(f"年化收益: {p.get('annual_return', float('nan')):.2%}")
        print(f"年化波动: {p.get('annual_volatility', float('nan')):.2%}")
        print(f"最大回撤: {p.get('max_drawdown_pct', float('nan')):.2%}")
        print(f"夏普: {p.get('sharpe', float('nan')):.4f}")
        print(f"索提诺: {p.get('sortino', float('nan')):.4f}")
        print(f"卡尔玛: {p.get('calmar', float('nan')):.4f}")
        print(f"信息比率: {p.get('information_ratio', float('nan')):.4f}")
        print(f"Beta(HSI): {p.get('beta_vs_hsi', float('nan')):.4f}")
        print(f"多空相关性: {p.get('long_short_corr', float('nan')):.4f}")

    print("成本:")
    print(f"  交易成本: {costs.get('commission_long', 0.0) + costs.get('commission_short', 0.0):,.2f} HKD")
    print(f"  融券费用: {costs.get('borrow_fee', 0.0):,.2f} HKD")
    print(f"  总成本:   {costs.get('total', 0.0):,.2f} HKD")
    print(f"报告: {report_path}")


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        universe_path = resolve_path(base_dir, args.universe)
        config_path = resolve_path(base_dir, args.config)
        output_dir = resolve_path(base_dir, args.output)

        from src.data_manager import DataManager
        from src.metrics import MetricsCalculator
        from src.visualizer import ReportVisualizer
        from strategies.long_short_bt import (
            parse_board_lot_overrides,
            run_backtrader_backtest,
            run_backtrader_sensitivity,
        )

        universe = load_universe(universe_path)
        dm = DataManager(data_dir=base_dir / "data", logger=print)

        start = str(pd.Timestamp(args.start).date())
        end = resolve_end_date(args.end)

        if args.validate_universe:
            return validate_universe(dm, universe, start=start, end=end, sector=args.sector)
        if args.update_data_only:
            update_data_cache(dm, universe, start=start, end=end, sector=args.sector)
            return 0

        cfg = load_strategy(config_path, universe)

        bench_codes = ["^HSI"]
        sector_bmk = get_sector_benchmark(universe, cfg.sector)
        if sector_bmk and sector_bmk != "^HSI":
            try:
                bdf = dm.fetch_index_data(
                    sector_bmk,
                    start=str(cfg.backtest.start_date),
                    end=str(cfg.backtest.end_date),
                )
                if not bdf.empty:
                    bench_codes.append(sector_bmk)
                else:
                    print(f"[WARN] 行业基准 {sector_bmk} 无数据，回退仅使用 ^HSI")
                    sector_bmk = None
            except Exception as exc:
                print(f"[WARN] 行业基准 {sector_bmk} 不可用({exc})，回退仅使用 ^HSI")
                sector_bmk = None
        bench_codes = list(dict.fromkeys(bench_codes))

        board_lots = parse_board_lot_overrides(config_path)
        result = run_backtrader_backtest(
            config=cfg,
            data_manager=dm,
            benchmark_codes=bench_codes,
            logger=print,
            board_lot_overrides=board_lots,
            volume_fill_pct=float(args.volume_fill_pct),
        )

        benchmark_for_metrics = None
        if sector_bmk and sector_bmk in result.benchmark_nav:
            benchmark_for_metrics = result.benchmark_nav[sector_bmk]
        elif "^HSI" in result.benchmark_nav:
            benchmark_for_metrics = result.benchmark_nav["^HSI"]
        elif result.benchmark_nav:
            benchmark_for_metrics = next(iter(result.benchmark_nav.values()))

        mc = MetricsCalculator()
        metrics = mc.calculate_all(result=result, benchmark=benchmark_for_metrics, rf_rate=float(args.rf_rate))

        sensitivity_df = pd.DataFrame()
        borrow_rates = list(cfg.sensitivity.borrow_rates or [])
        if borrow_rates:
            sensitivity_df = run_backtrader_sensitivity(
                config=cfg,
                data_manager=dm,
                benchmark_codes=bench_codes,
                borrow_rates=borrow_rates,
                logger=print,
                board_lot_overrides=board_lots,
                volume_fill_pct=float(args.volume_fill_pct),
            )

        visualizer = ReportVisualizer(logger=print)
        report_path = visualizer.generate_report(
            config=cfg,
            result=result,
            metrics=metrics,
            sensitivity_df=sensitivity_df,
            output_dir=output_dir,
        )

        print_summary(cfg, metrics.summary_table, report_path, result.costs)
        if not args.no_browser:
            try:
                webbrowser.open(report_path.as_uri())
            except Exception as exc:
                print(f"[WARN] 自动打开浏览器失败: {exc}")
        return 0

    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        return 2
    except ImportError as exc:
        print(f"[IMPORT ERROR] {exc}")
        print("请先安装 backtrader 依赖：pip install -r apps/backtest/requirements.txt")
        return 2
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

