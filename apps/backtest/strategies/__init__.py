"""Backtrader strategy package for quant backtests."""

from .inflection_strategy import InflectionStrategy


def prepare_backtrader_input(*args, **kwargs):
    from .long_short_bt import prepare_backtrader_input as _prepare_backtrader_input

    return _prepare_backtrader_input(*args, **kwargs)


def run_backtrader_backtest(*args, **kwargs):
    from .long_short_bt import run_backtrader_backtest as _run_backtrader_backtest

    return _run_backtrader_backtest(*args, **kwargs)


def run_backtrader_sensitivity(*args, **kwargs):
    from .long_short_bt import run_backtrader_sensitivity as _run_backtrader_sensitivity

    return _run_backtrader_sensitivity(*args, **kwargs)


__all__ = [
    "InflectionStrategy",
    "prepare_backtrader_input",
    "run_backtrader_backtest",
    "run_backtrader_sensitivity",
]
