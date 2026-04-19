"""Genetic optimization and walk-forward validation for backtest strategies."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backtest_engine import BacktestEngine
from .config_loader import StrategyConfig, UniverseConfig, load_strategy, load_universe
from .data_manager import DataManager
from .metrics import MetricsCalculator


@dataclass
class ParamBound:
    """Search bound definition for one parameter."""

    name: str
    low: float
    high: float
    is_int: bool


class StrategyOptimizer:
    """Auto-optimize strategy params with GA and validate with walk-forward."""

    _PARAM_TARGETS: Dict[str, Tuple[str, ...]] = {
        "inflection_min_score": ("inflection", "min_score"),
        "min_score": ("inflection", "min_score"),
        "max_valuation_percentile": ("inflection", "max_valuation_percentile"),
        "inflection_max_valuation_percentile": ("inflection", "max_valuation_percentile"),
        "top_n": ("inflection", "top_n"),
        "inflection_top_n": ("inflection", "top_n"),
    }

    def __init__(
        self,
        base_dir: Path | str,
        universe_path: Path | str,
        *,
        fast_mode: bool = False,
        seed: Optional[int] = None,
        logger=print,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.universe_path = Path(universe_path)
        if not self.universe_path.is_absolute():
            self.universe_path = (self.base_dir / self.universe_path).resolve()

        self.fast_mode = bool(fast_mode)
        self.logger = logger
        self.random = random.Random(seed)

        self.universe: UniverseConfig = load_universe(self.universe_path)
        self.data_manager = DataManager(data_dir=self.base_dir / "data", logger=logger)
        self.metrics_calculator = MetricsCalculator()

        self._fitness_cache: Dict[Tuple[Any, ...], float] = {}
        self._last_generations = 20
        self._last_population = 30

    def optimize(
        self,
        strategy_yaml: str,
        param_space: dict,
        n_generations: int = 20,
        population_size: int = 30,
    ) -> dict:
        """Run GA optimization and return best parameters + evolution history."""
        return self._optimize_internal(
            strategy_yaml=strategy_yaml,
            param_space=param_space,
            n_generations=n_generations,
            population_size=population_size,
            window=None,
            log_prefix="[GA]",
            log_generation=True,
        )

    def walk_forward_validate(
        self,
        strategy_yaml: str,
        params: dict,
        n_splits: int = 4,
    ) -> dict:
        """Run walk-forward optimization/validation across time splits."""
        strategy_path = self._resolve_path(self.base_dir, str(strategy_yaml))
        base_cfg = load_strategy(strategy_path, self.universe)

        if n_splits < 2:
            raise ValueError("n_splits 必须 >= 2")

        seg_bounds = self._build_segment_bounds(
            start=base_cfg.backtest.start_date,
            end=base_cfg.backtest.end_date,
            n_splits=n_splits,
        )

        wf_generations = max(4, int(self._last_generations * 0.5))
        wf_population = max(8, int(self._last_population * 0.5))
        if self.fast_mode:
            wf_generations = min(wf_generations, 6)
            wf_population = min(wf_population, 10)

        splits: List[Dict[str, Any]] = []
        train_sharpes: List[float] = []
        test_sharpes: List[float] = []

        for i in range(1, len(seg_bounds) - 1):
            train_start = seg_bounds[0]
            train_end = seg_bounds[i] - timedelta(days=1)
            test_start = seg_bounds[i]
            test_end = seg_bounds[i + 1] - timedelta(days=1)
            if i == len(seg_bounds) - 2:
                test_end = seg_bounds[i + 1]

            if train_end <= train_start or test_end <= test_start:
                continue

            self.logger(
                f"[WF] Split {i}/{len(seg_bounds)-2}: "
                f"train={train_start}~{train_end}, test={test_start}~{test_end}"
            )

            train_opt = self._optimize_internal(
                strategy_yaml=str(strategy_path),
                param_space=params,
                n_generations=wf_generations,
                population_size=wf_population,
                window=(train_start, train_end),
                log_prefix=f"[WF-{i}]",
                log_generation=False,
            )
            train_best = dict(train_opt["best_params"])
            train_sharpe = float(train_opt["best_sharpe"])
            test_sharpe = self._evaluate_individual(
                base_cfg=base_cfg,
                individual=train_best,
                window=(test_start, test_end),
            )

            train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)
            splits.append(
                {
                    "train_period": f"{train_start} -> {train_end}",
                    "test_period": f"{test_start} -> {test_end}",
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_sharpe,
                }
            )

        in_sample = float(np.nanmean(train_sharpes)) if train_sharpes else float("nan")
        out_sample = float(np.nanmean(test_sharpes)) if test_sharpes else float("nan")
        if np.isfinite(in_sample) and abs(in_sample) > 1e-9 and np.isfinite(out_sample):
            degradation = float(out_sample / in_sample)
        else:
            degradation = float("nan")
        is_robust = bool(np.isfinite(degradation) and degradation >= 0.5)

        return {
            "in_sample_sharpe": in_sample,
            "out_of_sample_sharpe": out_sample,
            "degradation_ratio": degradation,
            "is_robust": is_robust,
            "splits": splits,
        }

    def generate_report(self, optimization_result, walkforward_result) -> str:
        """Generate text diagnostics report for optimization + walk-forward."""
        best_params = optimization_result.get("best_params", {})
        best_sharpe = optimization_result.get("best_sharpe", float("nan"))
        history = optimization_result.get("evolution_history", []) or []

        in_sample = walkforward_result.get("in_sample_sharpe", float("nan"))
        out_sample = walkforward_result.get("out_of_sample_sharpe", float("nan"))
        degradation = walkforward_result.get("degradation_ratio", float("nan"))
        is_robust = bool(walkforward_result.get("is_robust", False))
        splits = walkforward_result.get("splits", []) or []

        lines: List[str] = []
        lines.append("=== Strategy Optimization Report ===")
        lines.append(f"Best Sharpe: {self._fmt(best_sharpe)}")
        lines.append(f"Best Params: {best_params}")
        lines.append("")
        lines.append("Evolution Chart Data (generation, best_sharpe, avg_sharpe):")
        if history:
            for row in history:
                lines.append(
                    f"{row.get('generation', '-')}, "
                    f"{self._fmt(row.get('best_sharpe', float('nan')))}, "
                    f"{self._fmt(row.get('avg_sharpe', float('nan')))}"
                )
        else:
            lines.append("N/A")

        lines.append("")
        lines.append("Walk-Forward Diagnostics:")
        lines.append(f"In-sample Sharpe: {self._fmt(in_sample)}")
        lines.append(f"Out-of-sample Sharpe: {self._fmt(out_sample)}")
        lines.append(f"Degradation Ratio (OOS/IS): {self._fmt(degradation)}")
        lines.append(f"Is Robust: {is_robust}")
        if np.isfinite(float(degradation)) and float(degradation) > 0.5:
            lines.append("⚠️ 策略可能过拟合")

        lines.append("")
        lines.append("Walk-Forward Splits:")
        if splits:
            for i, item in enumerate(splits, start=1):
                lines.append(
                    f"Split {i}: train[{item.get('train_period')}] "
                    f"sharpe={self._fmt(item.get('train_sharpe', float('nan')))} | "
                    f"test[{item.get('test_period')}] "
                    f"sharpe={self._fmt(item.get('test_sharpe', float('nan')))}"
                )
        else:
            lines.append("N/A")

        return "\n".join(lines)

    def _optimize_internal(
        self,
        *,
        strategy_yaml: str,
        param_space: dict,
        n_generations: int,
        population_size: int,
        window: Optional[Tuple[date, date]],
        log_prefix: str,
        log_generation: bool,
    ) -> dict:
        strategy_path = self._resolve_path(self.base_dir, str(strategy_yaml))
        base_cfg = load_strategy(strategy_path, self.universe)
        target_window = self._normalize_window(window, base_cfg)

        bounds = self._parse_param_space(param_space)
        if not bounds:
            raise ValueError("param_space 不能为空")

        n_generations = max(1, int(n_generations))
        population_size = max(4, int(population_size))
        self._last_generations = n_generations
        self._last_population = population_size

        population = [self._random_individual(bounds) for _ in range(population_size)]
        evolution_history: List[Dict[str, Any]] = []
        best_params: Dict[str, Any] = {}
        best_sharpe = -1e9

        for gen in range(1, n_generations + 1):
            fitness = [
                self._evaluate_individual(
                    base_cfg=base_cfg,
                    individual=individual,
                    window=target_window,
                )
                for individual in population
            ]

            gen_best_idx = int(np.argmax(fitness))
            gen_best_sharpe = float(fitness[gen_best_idx])
            gen_best_params = dict(population[gen_best_idx])
            gen_avg_sharpe = float(np.mean(fitness))

            if gen_best_sharpe > best_sharpe:
                best_sharpe = gen_best_sharpe
                best_params = dict(gen_best_params)

            evolution_history.append(
                {
                    "generation": gen,
                    "best_sharpe": gen_best_sharpe,
                    "avg_sharpe": gen_avg_sharpe,
                    "best_params": gen_best_params,
                }
            )

            if log_generation:
                self.logger(
                    f"{log_prefix} Generation {gen}/{n_generations} "
                    f"best_sharpe={gen_best_sharpe:.4f}, avg_sharpe={gen_avg_sharpe:.4f}"
                )

            if gen == n_generations:
                break

            elite = dict(gen_best_params)
            next_population: List[Dict[str, Any]] = [elite]
            while len(next_population) < population_size:
                p1 = self._tournament_select(population, fitness, k=3)
                p2 = self._tournament_select(population, fitness, k=3)
                c1, c2 = self._uniform_crossover(p1, p2, prob=0.5)
                c1 = self._gaussian_mutation(c1, bounds, gene_mutation_prob=0.25, sigma_scale=0.10)
                c2 = self._gaussian_mutation(c2, bounds, gene_mutation_prob=0.25, sigma_scale=0.10)
                next_population.append(c1)
                if len(next_population) < population_size:
                    next_population.append(c2)

            population = next_population

        return {
            "best_params": best_params,
            "best_sharpe": float(best_sharpe),
            "evolution_history": evolution_history,
        }

    def _evaluate_individual(
        self,
        *,
        base_cfg: StrategyConfig,
        individual: Dict[str, Any],
        window: Optional[Tuple[date, date]],
    ) -> float:
        cache_key = self._build_cache_key(base_cfg, individual, window)
        if cache_key in self._fitness_cache:
            return self._fitness_cache[cache_key]

        cfg = copy.deepcopy(base_cfg)
        self._apply_params(cfg, individual)
        if window is not None:
            cfg.backtest.start_date, cfg.backtest.end_date = window

        try:
            engine = BacktestEngine(
                config=cfg,
                data_manager=self.data_manager,
                benchmark_codes=["^HSI"],
                logger=lambda *_args, **_kwargs: None,
            )
            result = engine.run()
            benchmark = result.benchmark_nav.get("^HSI")
            if benchmark is None and result.benchmark_nav:
                benchmark = next(iter(result.benchmark_nav.values()))
            metrics = self.metrics_calculator.calculate_all(result=result, benchmark=benchmark, rf_rate=0.03)

            sharpe = float(metrics.summary_table.loc["portfolio", "sharpe"])
            if not np.isfinite(sharpe):
                sharpe = -1e6
        except Exception as exc:
            self.logger(f"[WARN] Backtest failed for params={individual}: {exc}")
            sharpe = -1e6

        self._fitness_cache[cache_key] = sharpe
        return sharpe

    def _build_cache_key(
        self,
        base_cfg: StrategyConfig,
        individual: Dict[str, Any],
        window: Optional[Tuple[date, date]],
    ) -> Tuple[Any, ...]:
        win_key = (
            str(window[0]) if window is not None else str(base_cfg.backtest.start_date),
            str(window[1]) if window is not None else str(base_cfg.backtest.end_date),
        )
        params_key = tuple(sorted((str(k), float(v)) for k, v in individual.items()))
        return (base_cfg.strategy_name, win_key, params_key)

    def _apply_params(self, cfg: StrategyConfig, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            target = self._PARAM_TARGETS.get(str(k))
            if target is None:
                continue
            obj: Any = cfg
            for attr in target[:-1]:
                obj = getattr(obj, attr)
            leaf = target[-1]
            current = getattr(obj, leaf)
            if isinstance(current, int):
                setattr(obj, leaf, int(round(float(v))))
            else:
                setattr(obj, leaf, float(v))

    def _parse_param_space(self, param_space: Dict[str, Any]) -> List[ParamBound]:
        out: List[ParamBound] = []
        for name, bound in dict(param_space or {}).items():
            if str(name) not in self._PARAM_TARGETS:
                raise ValueError(
                    f"未知参数: {name}。当前支持: {sorted(self._PARAM_TARGETS.keys())}"
                )
            if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                raise ValueError(f"param_space[{name}] 必须是 (min, max)")
            low = float(bound[0])
            high = float(bound[1])
            if high < low:
                low, high = high, low
            is_int = bool(isinstance(bound[0], int) and isinstance(bound[1], int))
            out.append(ParamBound(name=str(name), low=low, high=high, is_int=is_int))
        return out

    def _random_individual(self, bounds: List[ParamBound]) -> Dict[str, Any]:
        x: Dict[str, Any] = {}
        for b in bounds:
            v = self.random.uniform(b.low, b.high)
            x[b.name] = int(round(v)) if b.is_int else float(v)
        return x

    def _tournament_select(self, population: List[Dict[str, Any]], fitness: List[float], k: int = 3) -> Dict[str, Any]:
        k = max(2, int(k))
        candidates = [self.random.randrange(0, len(population)) for _ in range(k)]
        best_idx = max(candidates, key=lambda i: fitness[i])
        return dict(population[best_idx])

    def _uniform_crossover(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        prob: float = 0.5,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        c1 = dict(p1)
        c2 = dict(p2)
        keys = sorted(set(p1.keys()) | set(p2.keys()))
        for k in keys:
            if self.random.random() < prob:
                c1[k], c2[k] = c2.get(k, c1.get(k)), c1.get(k, c2.get(k))
        return c1, c2

    def _gaussian_mutation(
        self,
        individual: Dict[str, Any],
        bounds: List[ParamBound],
        gene_mutation_prob: float = 0.25,
        sigma_scale: float = 0.10,
    ) -> Dict[str, Any]:
        out = dict(individual)
        bound_map = {b.name: b for b in bounds}
        for name, b in bound_map.items():
            if self.random.random() >= gene_mutation_prob:
                continue
            span = max(1e-9, b.high - b.low)
            sigma = span * float(sigma_scale)
            base = float(out.get(name, b.low))
            mutated = base + self.random.gauss(0.0, sigma)
            clipped = min(max(mutated, b.low), b.high)
            out[name] = int(round(clipped)) if b.is_int else float(clipped)
        return out

    def _normalize_window(
        self,
        window: Optional[Tuple[date, date]],
        base_cfg: StrategyConfig,
    ) -> Optional[Tuple[date, date]]:
        if window is None:
            window = (base_cfg.backtest.start_date, base_cfg.backtest.end_date)
        start, end = window
        if self.fast_mode:
            fast_days = 365
            if (end - start).days > fast_days:
                start = end - timedelta(days=fast_days)
        if end <= start:
            start = end - timedelta(days=30)
        return (start, end)

    def _build_segment_bounds(self, start: date, end: date, n_splits: int) -> List[date]:
        total_days = max(2, (end - start).days)
        bounds = [start + timedelta(days=round(total_days * i / n_splits)) for i in range(n_splits + 1)]
        bounds[0] = start
        bounds[-1] = end
        out: List[date] = [bounds[0]]
        for d in bounds[1:]:
            if d <= out[-1]:
                d = out[-1] + timedelta(days=1)
            if d > end:
                d = end
            out.append(d)
        return out

    @staticmethod
    def _resolve_path(base_dir: Path, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base_dir / p).resolve()

    @staticmethod
    def _fmt(v: Any) -> str:
        try:
            f = float(v)
        except Exception:
            return "nan"
        return f"{f:.4f}" if np.isfinite(f) else "nan"
