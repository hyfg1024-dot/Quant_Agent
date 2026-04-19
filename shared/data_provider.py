from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import pandas as pd
from shared.utils import to_float


class DataProviderError(RuntimeError):
    """数据源基类异常。"""


class ProviderUnavailableError(DataProviderError):
    """数据源不可用（模块缺失 / 环境缺失 / 无返回）。"""


class ProviderTimeoutError(DataProviderError):
    """数据源超时。"""


_to_float = to_float


def _safe_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _is_hk_symbol(symbol: str) -> bool:
    s = str(symbol).strip()
    return s.isdigit() and len(s) == 5


def _to_qmt_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if _is_hk_symbol(s):
        return f"{s}.HK"
    return f"{s}.SH" if s.startswith(("5", "6", "9")) else f"{s}.SZ"


def _empty_intraday_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])


def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    out = df.copy()
    if "date" not in out.columns:
        if "time" in out.columns:
            out["date"] = pd.to_datetime(out["time"], errors="coerce")
        elif out.index.name is not None:
            out = out.reset_index().rename(columns={out.index.name: "date"})
        else:
            out["date"] = pd.NaT

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out[["date", "open", "high", "low", "close", "volume"]]


class BaseDataProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        pass

    @abstractmethod
    def get_intraday_flow(self, symbol: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_kline(self, symbol: str, count: int = 320) -> pd.DataFrame:
        pass


class QMTDataProvider(BaseDataProvider):
    def __init__(self, timeout_sec: float = 1.2, enabled: bool = True) -> None:
        self.enabled = enabled
        self.timeout_sec = timeout_sec
        self._subscribed_tick = set()
        try:
            from xtquant import xtdata  # type: ignore

            self.xtdata = xtdata
            self._init_error = None
        except Exception as exc:
            self.xtdata = None
            self._init_error = exc

    def _ensure_ready(self) -> None:
        if not self.enabled:
            raise ProviderUnavailableError("QMT disabled by config")
        if self.xtdata is None:
            raise ProviderUnavailableError(f"xtquant.xtdata unavailable: {self._init_error}")

    def _timed_call(self, fn: Callable[[], Any], action: str) -> Any:
        start = time.monotonic()
        result = fn()
        elapsed = time.monotonic() - start
        if elapsed > self.timeout_sec:
            raise ProviderTimeoutError(f"QMT {action} timeout: {elapsed:.2f}s")
        return result

    def _subscribe_tick_if_needed(self, qmt_symbol: str) -> None:
        if qmt_symbol in self._subscribed_tick:
            return
        try:
            self.xtdata.subscribe_quote(qmt_symbol, period="tick")  # type: ignore[attr-defined]
            self._subscribed_tick.add(qmt_symbol)
        except Exception:
            # 部分环境不要求显式订阅，忽略即可
            pass

    def get_quote(self, symbol: str) -> Dict:
        self._ensure_ready()
        qmt_symbol = _to_qmt_symbol(symbol)
        self._subscribe_tick_if_needed(qmt_symbol)

        payload = self._timed_call(
            lambda: self.xtdata.get_full_tick([qmt_symbol]),  # type: ignore[attr-defined]
            "get_full_tick",
        )
        tick = (payload or {}).get(qmt_symbol) or {}
        if not tick:
            raise ProviderUnavailableError(f"QMT empty tick for {qmt_symbol}")

        current_price = _to_float(tick.get("lastPrice") or tick.get("last_price"))
        prev_close = _to_float(tick.get("lastClose") or tick.get("preClose") or tick.get("pre_close"))
        open_price = _to_float(tick.get("open"))
        high_price = _to_float(tick.get("high"))
        low_price = _to_float(tick.get("low"))
        volume = _to_float(tick.get("volume"))
        amount = _to_float(tick.get("amount"))
        vwap = _to_float(tick.get("avgPrice") or tick.get("avg_price"))
        if vwap is None and volume and volume > 0 and amount is not None:
            vwap = amount / volume

        change_amount = None
        change_pct = None
        if current_price is not None and prev_close is not None:
            change_amount = current_price - prev_close
            if prev_close != 0:
                change_pct = change_amount / prev_close * 100

        bid_prices = _safe_list(tick.get("bidPrice") or tick.get("bid_price"))
        ask_prices = _safe_list(tick.get("askPrice") or tick.get("ask_price"))
        bid_vols = _safe_list(tick.get("bidVol") or tick.get("bid_vol"))
        ask_vols = _safe_list(tick.get("askVol") or tick.get("ask_vol"))

        bids_5 = []
        asks_5 = []
        for i in range(5):
            bids_5.append(
                {
                    "level": i + 1,
                    "price": _to_float(bid_prices[i]) if i < len(bid_prices) else None,
                    "volume_lot": _to_float(bid_vols[i]) if i < len(bid_vols) else None,
                }
            )
            asks_5.append(
                {
                    "level": i + 1,
                    "price": _to_float(ask_prices[i]) if i < len(ask_prices) else None,
                    "volume_lot": _to_float(ask_vols[i]) if i < len(ask_vols) else None,
                }
            )

        buy_10 = bids_5 + [{"level": i + 6, "price": None, "volume_lot": None} for i in range(5)]
        sell_10 = asks_5 + [{"level": i + 6, "price": None, "volume_lot": None} for i in range(5)]

        return {
            "symbol": str(symbol).strip(),
            "name": str(tick.get("stockName") or tick.get("name") or symbol),
            "current_price": current_price,
            "prev_close": prev_close,
            "open": open_price,
            "change_amount": change_amount,
            "change_pct": change_pct,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "amount": amount,
            "turnover_rate": _to_float(tick.get("turnoverRate") or tick.get("turnover_rate")),
            "turnover_rate_estimated": False,
            "amplitude_pct": _to_float(tick.get("amplitude")),
            "float_market_value_yi": _to_float(tick.get("floatMarketValue")),
            "total_market_value_yi": _to_float(tick.get("totalMarketValue")),
            "volume_ratio": _to_float(tick.get("volumeRatio") or tick.get("volRatio")),
            "order_diff": _to_float(tick.get("orderDiff")),
            "vwap": vwap,
            "premium_pct": ((current_price - vwap) / vwap * 100) if (current_price is not None and vwap not in {None, 0}) else None,
            "quote_time": tick.get("time"),
            "is_trading_data": bool(volume and volume > 0),
            "pe_dynamic": _to_float(tick.get("pe") or tick.get("peDynamic")),
            "pe_ttm": _to_float(tick.get("pe_ttm") or tick.get("peTTM")),
            "pb": _to_float(tick.get("pb")),
            "order_book_5": {"buy": bids_5, "sell": asks_5},
            "order_book_10": {"buy": buy_10, "sell": sell_10},
            "error": None,
        }

    def get_intraday_flow(self, symbol: str) -> pd.DataFrame:
        self._ensure_ready()
        qmt_symbol = _to_qmt_symbol(symbol)
        fields = ["time", "close", "volume", "amount"]

        payload = self._timed_call(
            lambda: self.xtdata.get_market_data_ex(  # type: ignore[attr-defined]
                field_list=fields,
                stock_list=[qmt_symbol],
                period="1m",
                count=240,
                fill_data=True,
            ),
            "get_market_data_ex_intraday",
        )

        data = (payload or {}).get(qmt_symbol)
        if data is None:
            raise ProviderUnavailableError(f"QMT intraday empty for {qmt_symbol}")
        df = pd.DataFrame(data)
        if df.empty:
            return _empty_intraday_df()

        time_col = "time" if "time" in df.columns else None
        price_col = "close" if "close" in df.columns else ("lastPrice" if "lastPrice" in df.columns else None)
        vol_col = "volume" if "volume" in df.columns else None
        amt_col = "amount" if "amount" in df.columns else None
        if not time_col or not price_col:
            raise ProviderUnavailableError("QMT intraday columns missing")

        out = pd.DataFrame()
        out["time"] = pd.to_datetime(df[time_col], errors="coerce", unit="ms")
        out["price"] = pd.to_numeric(df[price_col], errors="coerce")
        vol_series = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.Series([pd.NA] * len(df))
        amt_series = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else pd.Series([pd.NA] * len(df))
        vol_delta = vol_series.diff().fillna(vol_series).clip(lower=0)
        amt_delta = amt_series.diff().fillna(amt_series).clip(lower=0)
        if not _is_hk_symbol(symbol):
            vol_delta = vol_delta / 100.0
        out["volume_lot"] = vol_delta
        out["amount"] = amt_delta
        out = out.dropna(subset=["time"])
        return out[["time", "price", "volume_lot", "amount"]] if not out.empty else _empty_intraday_df()

    def get_kline(self, symbol: str, count: int = 320) -> pd.DataFrame:
        self._ensure_ready()
        qmt_symbol = _to_qmt_symbol(symbol)
        fields = ["time", "open", "high", "low", "close", "volume"]
        payload = self._timed_call(
            lambda: self.xtdata.get_market_data_ex(  # type: ignore[attr-defined]
                field_list=fields,
                stock_list=[qmt_symbol],
                period="1d",
                count=count,
                fill_data=True,
            ),
            "get_market_data_ex_kline",
        )
        data = (payload or {}).get(qmt_symbol)
        if data is None:
            raise ProviderUnavailableError(f"QMT kline empty for {qmt_symbol}")
        df = pd.DataFrame(data)
        if df.empty:
            raise ProviderUnavailableError(f"QMT kline empty frame for {qmt_symbol}")
        df = df.rename(columns={"time": "date"})
        return _normalize_kline_df(df)


class AkshareDataProvider(BaseDataProvider):
    """
    兼容现有免费爬虫接口（腾讯/AkShare），通过回调复用原实现。
    """

    def __init__(
        self,
        quote_fetcher: Callable[[str], Dict],
        intraday_fetcher: Callable[[str], pd.DataFrame],
        kline_fetcher: Callable[[str, int], pd.DataFrame],
    ) -> None:
        self._quote_fetcher = quote_fetcher
        self._intraday_fetcher = intraday_fetcher
        self._kline_fetcher = kline_fetcher

    def get_quote(self, symbol: str) -> Dict:
        return self._quote_fetcher(symbol)

    def get_intraday_flow(self, symbol: str) -> pd.DataFrame:
        return self._intraday_fetcher(symbol)

    def get_kline(self, symbol: str, count: int = 320) -> pd.DataFrame:
        return self._kline_fetcher(symbol, count)


class FallbackDataProvider(BaseDataProvider):
    """
    瀑布流降级：Primary失败时自动回退到Fallback。
    """

    def __init__(self, primary: BaseDataProvider, fallback: BaseDataProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_provider = "fallback"
        self.last_fallback_reason = ""

    @staticmethod
    def _can_fallback(exc: Exception) -> bool:
        return isinstance(
            exc,
            (
                ProviderUnavailableError,
                ProviderTimeoutError,
                TimeoutError,
                ModuleNotFoundError,
                ImportError,
            ),
        )

    def _run_with_fallback(self, method_name: str, *args, **kwargs):
        method_primary = getattr(self.primary, method_name)
        method_fallback = getattr(self.fallback, method_name)
        try:
            result = method_primary(*args, **kwargs)
            self.last_provider = "primary"
            self.last_fallback_reason = ""
            return result
        except Exception as exc:
            if not self._can_fallback(exc):
                raise
            result = method_fallback(*args, **kwargs)
            self.last_provider = "fallback"
            self.last_fallback_reason = str(exc)
            return result

    def get_quote(self, symbol: str) -> Dict:
        return self._run_with_fallback("get_quote", symbol)

    def get_intraday_flow(self, symbol: str) -> pd.DataFrame:
        return self._run_with_fallback("get_intraday_flow", symbol)

    def get_kline(self, symbol: str, count: int = 320) -> pd.DataFrame:
        return self._run_with_fallback("get_kline", symbol, count=count)
