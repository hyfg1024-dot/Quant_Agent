"""CLI entry for strategy parameter optimization and walk-forward validation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from src.strategy_optimizer import StrategyOptimizer


DEFAULT_PARAM_SPACE: Dict[str, Any] = {
    "inflection_min_score": [30, 80],
    "max_valuation_percentile": [20, 60],
    "top_n": [3, 10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="策略参数自动优化 + Walk-Forward 验证")
    parser.add_argument("--config", default="config/strategies/inflection_example.yaml", help="策略配置 YAML 路径")
    parser.add_argument("--universe", default="config/universe.yaml", help="Universe 配置路径")
    parser.add_argument("--generations", type=int, default=20, help="遗传算法迭代代数")
    parser.add_argument("--population", type=int, default=30, help="遗传算法种群大小")
    parser.add_argument("--splits", type=int, default=4, help="Walk-Forward 切分段数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fast", action="store_true", help="快速模式：缩短回测区间并压缩 GA 规模")
    parser.add_argument(
        "--param-space",
        default="",
        help='参数空间 JSON 字符串，例如: \'{"inflection_min_score":[30,80],"top_n":[3,10]}\'',
    )
    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_param_space(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return dict(DEFAULT_PARAM_SPACE)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--param-space 必须是 JSON 对象")
    return obj


def load_strategy_name(config_path: Path) -> str:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        name = str(payload.get("strategy_name", "")).strip()
        if name:
            return name
    except Exception:
        pass
    return config_path.stem


def sanitize_name(text: str) -> str:
    out = re.sub(r"\s+", "_", str(text).strip())
    out = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "strategy"


def clamp_ga_params(generations: int, population: int, fast: bool) -> tuple[int, int]:
    g = max(1, int(generations))
    p = max(4, int(population))
    if fast:
        g = min(g, 8)
        p = min(p, 12)
    return g, p


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        config_path = resolve_path(base_dir, args.config)
        universe_path = resolve_path(base_dir, args.universe)
        param_space = load_param_space(args.param_space)
        generations, population = clamp_ga_params(args.generations, args.population, bool(args.fast))

        if args.fast:
            print(f"[FAST] 已启用：generations={generations}, population={population}")

        optimizer = StrategyOptimizer(
            base_dir=base_dir,
            universe_path=universe_path,
            fast_mode=bool(args.fast),
            seed=int(args.seed),
            logger=print,
        )

        print("[RUN] 启动遗传算法优化...")
        optimization_result = optimizer.optimize(
            strategy_yaml=str(config_path),
            param_space=param_space,
            n_generations=generations,
            population_size=population,
        )

        print("[RUN] 启动 Walk-Forward 验证...")
        walkforward_result = optimizer.walk_forward_validate(
            strategy_yaml=str(config_path),
            params=param_space,
            n_splits=max(2, int(args.splits)),
        )

        report = optimizer.generate_report(optimization_result, walkforward_result)
        print("")
        print(report)

        strategy_name = sanitize_name(load_strategy_name(config_path))
        out_dir = (base_dir / "data" / "optimization").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}_{strategy_name}.json"

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "universe_path": str(universe_path),
            "fast_mode": bool(args.fast),
            "generations": generations,
            "population": population,
            "splits": max(2, int(args.splits)),
            "seed": int(args.seed),
            "param_space": param_space,
            "optimization_result": optimization_result,
            "walkforward_result": walkforward_result,
            "report": report,
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] 优化结果已保存: {out_file}")
        return 0

    except json.JSONDecodeError as exc:
        print(f"[ERROR] --param-space JSON 解析失败: {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

