from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd
import requests
import os
import sys
from pathlib import Path
try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_provider import AkshareDataProvider, FallbackDataProvider, QMTDataProvider

if str(os.getenv("DISABLE_STREAMLIT_RUNTIME_CACHE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
    st = None  # type: ignore[assignment]
else:
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - 非 Streamlit 环境兜底
        st = None  # type: ignore[assignment]

TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q={exchange}{symbol}"
TENCENT_MINUTE_URL = "https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={exchange}{symbol}"
TENCENT_DAILY_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={exchange}{symbol},day,,,{count},qfq"


def _cache_data(ttl: int = 3600):
    def _decorator(func):
        if st is None:
            return func
        return st.cache_data(ttl=ttl, show_spinner=False)(func)

    return _decorator


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "--"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol).strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit())

    if raw.startswith("hk"):
        if not digits:
            return raw.replace("hk", "").strip()
        return digits[-5:].zfill(5)

    if len(digits) == 5:
        return digits
    if len(digits) >= 6:
        return digits[-6:]
    return str(symbol).strip()


def _resolve_exchange(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    if normalized.isdigit() and len(normalized) == 5:
        return "hk"
    if normalized.isdigit() and len(normalized) == 6:
        return "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    raw = str(symbol).strip().lower()
    if raw.startswith("hk"):
        return "hk"
    return "sh" if raw.startswith("sh") else "sz"


def _resolve_market(symbol: str) -> Tuple[str, str]:
    normalized = _normalize_symbol(symbol)
    return _resolve_exchange(normalized), normalized


def _build_order_book_10(bids_5: List[Dict], asks_5: List[Dict]) -> Dict[str, List[Dict]]:
    buy = []
    sell = []

    for i in range(10):
        level = i + 1
        if i < len(bids_5):
            buy.append(bids_5[i])
        else:
            buy.append({"level": level, "price": None, "volume_lot": None})

        if i < len(asks_5):
            sell.append(asks_5[i])
        else:
            sell.append({"level": level, "price": None, "volume_lot": None})

    return {"buy": buy, "sell": sell}


def _parse_tencent_fields(symbol: str, fields: List[str]) -> Dict:
    if len(fields) < 60:
        raise ValueError("Malformed Tencent payload")

    name = fields[1].strip() or symbol
    current_price = _to_float(fields[3])
    prev_close = _to_float(fields[4])
    is_hk = str(symbol).isdigit() and len(str(symbol)) == 5
    volume_raw = _to_float(fields[36])
    amount_raw = _to_float(fields[37])
    quote_time = fields[30].strip()

    change_amount = _to_float(fields[31])
    change_pct = _to_float(fields[32])

    if current_price is None and prev_close is not None:
        current_price = prev_close

    # 口径差异：
    # A股: f36=手, f37=万元
    # 港股: f36=股, f37=成交额(港币)
    if is_hk:
        volume = volume_raw
        amount = amount_raw
    else:
        volume = volume_raw * 100 if volume_raw is not None else None
        amount = amount_raw * 10000 if amount_raw is not None else None

    vwap = _to_float(fields[51])
    if vwap is None:
        if volume and volume > 0 and amount and amount > 0:
            vwap = amount / volume
        else:
            vwap = current_price

    premium_pct = None
    if current_price is not None and vwap is not None and vwap > 0:
        premium_pct = (current_price - vwap) / vwap * 100

    bids_5: List[Dict] = []
    asks_5: List[Dict] = []
    for i in range(5):
        bid_price = _to_float(fields[9 + i * 2])
        bid_vol = _to_float(fields[10 + i * 2])
        ask_price = _to_float(fields[19 + i * 2])
        ask_vol = _to_float(fields[20 + i * 2])

        if bid_price is not None and bid_price <= 0:
            bid_price = None
        if ask_price is not None and ask_price <= 0:
            ask_price = None
        if bid_vol is not None and bid_vol <= 0:
            bid_vol = None
        if ask_vol is not None and ask_vol <= 0:
            ask_vol = None

        bids_5.append({"level": i + 1, "price": bid_price, "volume_lot": bid_vol})
        asks_5.append({"level": i + 1, "price": ask_price, "volume_lot": ask_vol})

    pe_dynamic = _to_float(fields[52])
    pe_ttm = _to_float(fields[53])
    if pe_dynamic is not None and pe_dynamic <= 0:
        pe_dynamic = None
    if pe_ttm is not None and pe_ttm <= 0:
        pe_ttm = None

    turnover_rate = _to_float(fields[38])
    turnover_rate_estimated = False
    float_mv_yi = _to_float(fields[44])
    if (
        (turnover_rate is None or turnover_rate <= 0)
        and is_hk
        and current_price is not None
        and current_price > 0
        and volume is not None
        and volume > 0
        and float_mv_yi is not None
        and float_mv_yi > 0
    ):
        float_shares = float_mv_yi * 1e8 / current_price
        if float_shares > 0:
            turnover_rate = volume / float_shares * 100
            turnover_rate_estimated = True

    return {
        "symbol": symbol,
        "name": name,
        "current_price": current_price,
        "prev_close": prev_close,
        "open": _to_float(fields[5]),
        "change_amount": change_amount,
        "change_pct": change_pct,
        "high": _to_float(fields[33]),
        "low": _to_float(fields[34]),
        "volume": volume,
        "amount": amount,
        "turnover_rate": turnover_rate,
        "turnover_rate_estimated": turnover_rate_estimated,
        "amplitude_pct": _to_float(fields[43]),
        "float_market_value_yi": float_mv_yi,
        "total_market_value_yi": _to_float(fields[45]),
        "volume_ratio": _to_float(fields[49]),
        "order_diff": _to_float(fields[50]),
        "vwap": vwap,
        "premium_pct": premium_pct,
        "quote_time": quote_time,
        "is_trading_data": bool(volume and volume > 0),
        "pe_dynamic": pe_dynamic,
        "pe_ttm": pe_ttm,
        "pb": _to_float(fields[46]),
        "order_book_5": {"buy": bids_5, "sell": asks_5},
        "order_book_10": _build_order_book_10(bids_5, asks_5),
        "error": None,
    }


def _fetch_tencent_quote(symbol: str) -> Dict:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_QUOTE_URL.format(exchange=exchange, symbol=normalized)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
    resp.raise_for_status()
    resp.encoding = "gbk"

    text = resp.text
    if '"' not in text or '~' not in text:
        raise ValueError("No Tencent quote payload")

    payload = text.split('"', 1)[1].rsplit('"', 1)[0]
    fields = payload.split("~")
    return _parse_tencent_fields(normalized, fields)


async def _fetch_tencent_quote_async(
    session: aiohttp.ClientSession,
    symbol: str,
    sem: Optional[asyncio.Semaphore] = None,
) -> Dict:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_QUOTE_URL.format(exchange=exchange, symbol=normalized)
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if sem is None:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                body = await resp.read()
        else:
            async with sem:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    body = await resp.read()
        text = body.decode("gbk", errors="ignore")
        if '"' not in text or '~' not in text:
            raise ValueError("No Tencent quote payload")
        payload = text.split('"', 1)[1].rsplit('"', 1)[0]
        fields = payload.split("~")
        out = _parse_tencent_fields(normalized, fields)
        out["provider_used"] = "fallback_async"
        out["provider_fallback_reason"] = ""
        out["qmt_enabled"] = _qmt_enabled_from_env()
        return out
    except Exception as exc:
        return {
            "symbol": normalized,
            "name": normalized,
            "current_price": None,
            "prev_close": None,
            "open": None,
            "change_amount": None,
            "change_pct": None,
            "high": None,
            "low": None,
            "volume": None,
            "amount": None,
            "turnover_rate": None,
            "turnover_rate_estimated": False,
            "amplitude_pct": None,
            "float_market_value_yi": None,
            "total_market_value_yi": None,
            "volume_ratio": None,
            "order_diff": None,
            "vwap": None,
            "premium_pct": None,
            "quote_time": None,
            "is_trading_data": False,
            "pe_dynamic": None,
            "pe_ttm": None,
            "pb": None,
            "order_book_5": {"buy": [], "sell": []},
            "order_book_10": {
                "buy": [{"level": i + 1, "price": None, "volume_lot": None} for i in range(10)],
                "sell": [{"level": i + 1, "price": None, "volume_lot": None} for i in range(10)],
            },
            "provider_used": "error",
            "provider_fallback_reason": "",
            "qmt_enabled": _qmt_enabled_from_env(),
            "error": str(exc),
        }


_DATA_PROVIDER = None


def _qmt_enabled_from_env() -> bool:
    raw = str(os.getenv("QMT_ENABLED", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _get_data_provider():
    global _DATA_PROVIDER
    if _DATA_PROVIDER is None:
        qmt_enabled = _qmt_enabled_from_env()
        _DATA_PROVIDER = FallbackDataProvider(
            primary=QMTDataProvider(timeout_sec=1.2, enabled=qmt_enabled),
            fallback=AkshareDataProvider(
                quote_fetcher=_fetch_tencent_quote,
                intraday_fetcher=_fetch_tencent_intraday_flow,
                kline_fetcher=_fetch_tencent_kline,
            ),
        )
    return _DATA_PROVIDER


def fetch_realtime_quote(symbol: str) -> Dict:
    normalized = _normalize_symbol(symbol)
    try:
        provider = _get_data_provider()
        quote = provider.get_quote(normalized)
        if "order_book_10" not in quote and "order_book_5" in quote:
            ob5 = quote.get("order_book_5", {"buy": [], "sell": []})
            quote["order_book_10"] = _build_order_book_10(
                ob5.get("buy", []),
                ob5.get("sell", []),
            )
        quote["symbol"] = normalized
        quote["provider_used"] = getattr(provider, "last_provider", "fallback")
        quote["provider_fallback_reason"] = getattr(provider, "last_fallback_reason", "")
        quote["qmt_enabled"] = _qmt_enabled_from_env()
        return quote
    except Exception as exc:
        return {
            "symbol": normalized,
            "name": normalized,
            "current_price": None,
            "prev_close": None,
            "open": None,
            "change_amount": None,
            "change_pct": None,
            "high": None,
            "low": None,
            "volume": None,
            "amount": None,
            "turnover_rate": None,
            "turnover_rate_estimated": False,
            "amplitude_pct": None,
            "float_market_value_yi": None,
            "total_market_value_yi": None,
            "volume_ratio": None,
            "order_diff": None,
            "vwap": None,
            "premium_pct": None,
            "quote_time": None,
            "is_trading_data": False,
            "pe_dynamic": None,
            "pe_ttm": None,
            "pb": None,
            "order_book_5": {"buy": [], "sell": []},
            "order_book_10": {
                "buy": [{"level": i + 1, "price": None, "volume_lot": None} for i in range(10)],
                "sell": [{"level": i + 1, "price": None, "volume_lot": None} for i in range(10)],
            },
            "provider_used": "error",
            "provider_fallback_reason": "",
            "qmt_enabled": _qmt_enabled_from_env(),
            "error": str(exc),
        }


async def _fetch_realtime_quotes_batch_async(
    symbols: List[str],
    timeout_sec: float = 1.2,
    max_concurrency: int = 12,
) -> Dict[str, Dict]:
    if aiohttp is None:
        return {sym: fetch_realtime_quote(sym) for sym in symbols}
    normalized_symbols = [_normalize_symbol(s) for s in symbols if str(s).strip()]
    if not normalized_symbols:
        return {}

    connector = aiohttp.TCPConnector(limit=max(4, int(max_concurrency)))
    timeout = aiohttp.ClientTimeout(total=max(0.6, float(timeout_sec)))
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))
    out: Dict[str, Dict] = {}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [_fetch_tencent_quote_async(session, sym, sem) for sym in normalized_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for sym, item in zip(normalized_symbols, results):
            if isinstance(item, Exception):
                out[sym] = fetch_realtime_quote(sym)
            else:
                out[sym] = item
    return out


def fetch_realtime_quotes_batch(
    symbols: List[str],
    timeout_sec: float = 1.2,
    max_concurrency: int = 12,
) -> Dict[str, Dict]:
    """
    批量实时行情（并发）。
    - 默认走 aiohttp + asyncio.gather（免费接口）
    - 当 QMT 开关开启时，为保持口径一致，退化为逐只 provider 调用
    """
    normalized_symbols = [_normalize_symbol(s) for s in symbols if str(s).strip()]
    if not normalized_symbols:
        return {}

    if _qmt_enabled_from_env():
        return {sym: fetch_realtime_quote(sym) for sym in normalized_symbols}
    if aiohttp is None:
        return {sym: fetch_realtime_quote(sym) for sym in normalized_symbols}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _fetch_realtime_quotes_batch_async(
                normalized_symbols,
                timeout_sec=timeout_sec,
                max_concurrency=max_concurrency,
            )
        )
    finally:
        loop.close()


def _fetch_tencent_intraday_flow(symbol: str) -> pd.DataFrame:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_MINUTE_URL.format(exchange=exchange, symbol=normalized)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
    resp.raise_for_status()
    payload = resp.json()

    code_key = f"{exchange}{normalized}"
    target = payload.get("data", {}).get(code_key, {})
    data_obj = target.get("data", {})
    raw_lines = data_obj.get("data", [])
    trade_date = str(data_obj.get("date", ""))

    rows = []
    for line in raw_lines:
        parts = str(line).split()
        if len(parts) < 4:
            continue
        hhmm, price_text, vol_text, amount_text = parts[:4]
        time_text = f"{trade_date}{hhmm}" if trade_date else hhmm
        rows.append(
            {
                "time": pd.to_datetime(time_text, format="%Y%m%d%H%M", errors="coerce"),
                "price": _to_float(price_text),
                "volume_lot_cum": _to_float(vol_text),
                "amount": _to_float(amount_text),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])

    df = pd.DataFrame(rows).dropna(subset=["time"]).reset_index(drop=True)
    df["volume_lot"] = df["volume_lot_cum"].diff().fillna(df["volume_lot_cum"])
    df["volume_lot"] = df["volume_lot"].clip(lower=0)
    return df[["time", "price", "volume_lot", "amount"]]


async def _fetch_tencent_intraday_flow_async(session: aiohttp.ClientSession, symbol: str) -> pd.DataFrame:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_MINUTE_URL.format(exchange=exchange, symbol=normalized)
    async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
        resp.raise_for_status()
        payload = await resp.json(content_type=None)

    code_key = f"{exchange}{normalized}"
    target = payload.get("data", {}).get(code_key, {})
    data_obj = target.get("data", {})
    raw_lines = data_obj.get("data", [])
    trade_date = str(data_obj.get("date", ""))

    rows = []
    for line in raw_lines:
        parts = str(line).split()
        if len(parts) < 4:
            continue
        hhmm, price_text, vol_text, amount_text = parts[:4]
        time_text = f"{trade_date}{hhmm}" if trade_date else hhmm
        rows.append(
            {
                "time": pd.to_datetime(time_text, format="%Y%m%d%H%M", errors="coerce"),
                "price": _to_float(price_text),
                "volume_lot_cum": _to_float(vol_text),
                "amount": _to_float(amount_text),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])

    df = pd.DataFrame(rows).dropna(subset=["time"]).reset_index(drop=True)
    df["volume_lot"] = df["volume_lot_cum"].diff().fillna(df["volume_lot_cum"])
    df["volume_lot"] = df["volume_lot"].clip(lower=0)
    return df[["time", "price", "volume_lot", "amount"]]


def fetch_intraday_flow(symbol: str) -> pd.DataFrame:
    normalized = _normalize_symbol(symbol)
    try:
        df = _get_data_provider().get_intraday_flow(normalized)
        if df is None or df.empty:
            return pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])
        keep_cols = ["time", "price", "volume_lot", "amount"]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df[keep_cols]
    except Exception:
        return pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])


def _calc_rsi(close: pd.Series, period: int = 6) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # 使用 Wilder/SMA 递推口径（SMA(X, N, 1)），更接近主流券商终端 RSI 结果
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    gain_only = (avg_gain > 0) & (avg_loss == 0)
    loss_only = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(gain_only, 100.0)
    rsi = rsi.mask(loss_only, 0.0)
    return rsi


def _calc_rsi_set(close: pd.Series) -> Dict[str, Optional[float]]:
    close = pd.to_numeric(close, errors="coerce").dropna().reset_index(drop=True)
    if close.empty:
        return {"rsi6": None, "rsi12": None, "rsi24": None}
    rsi6 = _calc_rsi(close, period=6)
    rsi12 = _calc_rsi(close, period=12)
    rsi24 = _calc_rsi(close, period=24)
    return {
        "rsi6": _to_float(rsi6.iloc[-1]),
        "rsi12": _to_float(rsi12.iloc[-1]),
        "rsi24": _to_float(rsi24.iloc[-1]),
    }


def _calc_indicator_set_from_close(close: pd.Series) -> Dict[str, Optional[float]]:
    close = pd.to_numeric(close, errors="coerce").dropna().reset_index(drop=True)
    if close.empty:
        return {
            "rsi6": None,
            "rsi12": None,
            "rsi24": None,
            "ma5": None,
            "ma10": None,
            "ma20": None,
            "ma60": None,
            "macd_hist": None,
            "boll_mid": None,
            "boll_upper": None,
            "boll_lower": None,
            "boll_pct_b": None,
            "boll_bandwidth": None,
        }

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_hist = (dif - dea) * 2

    rsi6 = _calc_rsi(close, period=6)
    rsi12 = _calc_rsi(close, period=12)
    rsi24 = _calc_rsi(close, period=24)
    ma5 = close.rolling(5, min_periods=5).mean()
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    boll_mid = ma20
    boll_std = close.rolling(20, min_periods=20).std(ddof=0)
    boll_upper = boll_mid + 2 * boll_std
    boll_lower = boll_mid - 2 * boll_std

    latest_close = _to_float(close.iloc[-1]) if not close.empty else None
    latest_upper = _to_float(boll_upper.iloc[-1]) if not boll_upper.empty else None
    latest_lower = _to_float(boll_lower.iloc[-1]) if not boll_lower.empty else None
    boll_pct_b = None
    boll_bandwidth = None
    if latest_close is not None and latest_upper is not None and latest_lower is not None:
        spread = latest_upper - latest_lower
        if spread > 0:
            boll_pct_b = (latest_close - latest_lower) / spread * 100
        mid = _to_float(boll_mid.iloc[-1])
        if mid is not None and mid != 0:
            boll_bandwidth = spread / mid * 100

    return {
        "macd_hist": _to_float(macd_hist.iloc[-1]),
        "rsi6": _to_float(rsi6.iloc[-1]),
        "rsi12": _to_float(rsi12.iloc[-1]),
        "rsi24": _to_float(rsi24.iloc[-1]),
        "ma5": _to_float(ma5.iloc[-1]),
        "ma10": _to_float(ma10.iloc[-1]),
        "ma20": _to_float(ma20.iloc[-1]),
        "ma60": _to_float(ma60.iloc[-1]),
        "boll_mid": _to_float(boll_mid.iloc[-1]),
        "boll_upper": latest_upper,
        "boll_lower": latest_lower,
        "boll_pct_b": boll_pct_b,
        "boll_bandwidth": boll_bandwidth,
    }


def _calc_indicators_from_ohlcv(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    close = pd.to_numeric(df["close"], errors="coerce")
    close = close.dropna().reset_index(drop=True)
    if close.empty:
        raise ValueError("No valid close prices")
    return _calc_indicator_set_from_close(close)


async def _fetch_tencent_kline_async(
    session: aiohttp.ClientSession,
    symbol: str,
    count: int = 320,
) -> pd.DataFrame:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_DAILY_URL.format(exchange=exchange, symbol=normalized, count=count)
    async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
        resp.raise_for_status()
        payload = await resp.json(content_type=None)
    code_key = f"{exchange}{normalized}"
    kline_data = payload.get("data", {}).get(code_key, {}).get("qfqday", [])
    if not kline_data:
        raise ValueError("No daily kline data")
    normalized_rows = [row[:6] for row in kline_data if len(row) >= 6]
    if not normalized_rows:
        raise ValueError("Invalid daily kline payload")
    cols = ["date", "open", "close", "high", "low", "volume"]
    df = pd.DataFrame(normalized_rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if df.empty:
        raise ValueError("No daily kline data")
    return df[["date", "open", "high", "low", "close", "volume"]]


@_cache_data(ttl=3600)
def _fetch_tencent_kline_cached(symbol: str, count: int = 320) -> pd.DataFrame:
    exchange, normalized = _resolve_market(symbol)
    url = TENCENT_DAILY_URL.format(exchange=exchange, symbol=normalized, count=count)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
    resp.raise_for_status()
    payload = resp.json()
    code_key = f"{exchange}{normalized}"
    kline_data = payload.get("data", {}).get(code_key, {}).get("qfqday", [])
    if not kline_data and exchange == "hk":
        hk = _fetch_hk_daily_ohlcv(normalized).copy()
        hk = hk.reset_index().rename(columns={"index": "date"})
        return hk[["date", "open", "high", "low", "close", "volume"]]
    if not kline_data:
        raise ValueError("No daily kline data")

    normalized_rows = [row[:6] for row in kline_data if len(row) >= 6]
    if not normalized_rows:
        raise ValueError("Invalid daily kline payload")
    cols = ["date", "open", "close", "high", "low", "volume"]
    df = pd.DataFrame(normalized_rows, columns=cols)
    df = df.rename(columns={"close": "close", "high": "high", "low": "low", "open": "open", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df[["date", "open", "high", "low", "close", "volume"]]


def _fetch_tencent_kline(symbol: str, count: int = 320) -> pd.DataFrame:
    return _fetch_tencent_kline_cached(symbol, count=count).copy()


@_cache_data(ttl=3600)
def _fetch_daily_close_series_cached(symbol: str, count: int = 320) -> pd.Series:
    normalized = _normalize_symbol(symbol)
    kline_df = _get_data_provider().get_kline(normalized, count=count)
    if kline_df is None or kline_df.empty:
        raise ValueError("No daily close series")
    df = kline_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).set_index("date").sort_index()
    if df.empty:
        raise ValueError("No daily close series")
    return df["close"]


def _fetch_daily_close_series(symbol: str, count: int = 320) -> pd.Series:
    return _fetch_daily_close_series_cached(symbol, count=count).copy()


def fetch_multi_timeframe_rsi(symbol: str, intraday_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Optional[float]]]:
    result = {
        "day": {"rsi6": None, "rsi12": None, "rsi24": None},
        "week": {"rsi6": None, "rsi12": None, "rsi24": None},
        "month": {"rsi6": None, "rsi12": None, "rsi24": None},
        "intraday": {"rsi6": None, "rsi12": None, "rsi24": None},
    }
    try:
        daily_close = _fetch_daily_close_series(symbol, count=320)
        result["day"] = _calc_rsi_set(daily_close)
        result["week"] = _calc_rsi_set(daily_close.resample("W-FRI").last().dropna())
        result["month"] = _calc_rsi_set(daily_close.resample("ME").last().dropna())
    except Exception:
        pass

    try:
        if intraday_df is not None and not intraday_df.empty and "price" in intraday_df.columns:
            intra_close = pd.to_numeric(intraday_df["price"], errors="coerce").dropna()
            if not intra_close.empty:
                result["intraday"] = _calc_rsi_set(intra_close)
    except Exception:
        pass

    return result


def fetch_multi_timeframe_indicators(symbol: str, intraday_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Optional[float]]]:
    empty = _calc_indicator_set_from_close(pd.Series(dtype=float))
    result: Dict[str, Dict[str, Optional[float]]] = {
        "day": empty.copy(),
        "week": empty.copy(),
        "month": empty.copy(),
        "intraday": empty.copy(),
    }
    try:
        daily_close = _fetch_daily_close_series(symbol, count=320)
        result["day"] = _calc_indicator_set_from_close(daily_close)
        result["week"] = _calc_indicator_set_from_close(daily_close.resample("W-FRI").last().dropna())
        result["month"] = _calc_indicator_set_from_close(daily_close.resample("ME").last().dropna())
    except Exception:
        pass

    try:
        if intraday_df is not None and not intraday_df.empty and "price" in intraday_df.columns:
            intra_close = pd.to_numeric(intraday_df["price"], errors="coerce").dropna()
            if not intra_close.empty:
                result["intraday"] = _calc_indicator_set_from_close(intra_close)
    except Exception:
        pass

    return result


@_cache_data(ttl=3600)
def _fetch_hk_daily_ohlcv(symbol: str) -> pd.DataFrame:
    df = ak.stock_hk_daily(symbol=str(symbol).zfill(5), adjust="qfq")
    if df is None or df.empty:
        raise ValueError("No HK daily data")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        raise ValueError("HK daily payload missing OHLCV fields")
    return df[["open", "high", "low", "close", "volume"]].copy()


def fetch_technical_indicators(symbol: str, count: int = 120) -> Dict[str, Optional[float]]:
    normalized = _normalize_symbol(symbol)
    df = _get_data_provider().get_kline(normalized, count=count)
    if df is None or df.empty:
        raise ValueError("No daily kline data")
    return _calc_indicators_from_ohlcv(df[["open", "high", "low", "close", "volume"]])


async def _fetch_fast_panel_bundle_async(symbol: str) -> Tuple[Dict, pd.DataFrame, Dict[str, Optional[float]], List[str]]:
    if aiohttp is None:
        raise RuntimeError("aiohttp_not_installed")
    errors: List[str] = []
    timeout = aiohttp.ClientTimeout(total=2.4)
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        quote_task = _fetch_tencent_quote_async(session, symbol)
        intraday_task = _fetch_tencent_intraday_flow_async(session, symbol)
        kline_task = _fetch_tencent_kline_async(session, symbol, count=120)
        quote_res, intraday_res, kline_res = await asyncio.gather(
            quote_task,
            intraday_task,
            kline_task,
            return_exceptions=True,
        )

    if isinstance(quote_res, Exception):
        quote = fetch_realtime_quote(symbol)
        errors.append(f"quote_async: {quote_res}")
    else:
        quote = quote_res

    if isinstance(intraday_res, Exception):
        intraday_df = pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])
        errors.append(f"intraday_async: {intraday_res}")
    else:
        intraday_df = intraday_res

    if isinstance(kline_res, Exception):
        indicators = {
            "macd_hist": None,
            "rsi6": None,
            "rsi12": None,
            "rsi24": None,
            "ma5": None,
            "ma10": None,
            "ma20": None,
            "ma60": None,
            "boll_mid": None,
            "boll_upper": None,
            "boll_lower": None,
            "boll_pct_b": None,
            "boll_bandwidth": None,
        }
        errors.append(f"kline_async: {kline_res}")
    else:
        indicators = _calc_indicators_from_ohlcv(kline_res[["open", "high", "low", "close", "volume"]])

    return quote, intraday_df, indicators, errors


def fetch_fast_panel(symbol: str) -> Dict:
    normalized = _normalize_symbol(symbol)
    quote: Optional[Dict] = None

    intraday_df = pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])
    indicators = {
        "macd_hist": None,
        "rsi6": None,
        "rsi12": None,
        "rsi24": None,
        "ma5": None,
        "ma10": None,
        "ma20": None,
        "ma60": None,
        "boll_mid": None,
        "boll_upper": None,
        "boll_lower": None,
        "boll_pct_b": None,
        "boll_bandwidth": None,
    }
    errors = []

    if not _qmt_enabled_from_env():
        loop = asyncio.new_event_loop()
        try:
            quote, intraday_df, indicators, async_errors = loop.run_until_complete(
                _fetch_fast_panel_bundle_async(normalized)
            )
            errors.extend(async_errors)
        except Exception as exc:
            errors.append(f"async_bundle: {exc}")
        finally:
            loop.close()

    if quote is None:
        quote = fetch_realtime_quote(normalized)

    if intraday_df is None or intraday_df.empty:
        try:
            intraday_df = fetch_intraday_flow(normalized)
        except Exception as exc:
            errors.append(f"intraday: {exc}")

    if indicators.get("rsi6") is None and indicators.get("ma20") is None:
        try:
            indicators = fetch_technical_indicators(normalized)
        except Exception as exc:
            errors.append(f"indicators: {exc}")

    tf_indicators = fetch_multi_timeframe_indicators(normalized, intraday_df=intraday_df)
    rsi_multi = {
        k: {"rsi6": v.get("rsi6"), "rsi12": v.get("rsi12"), "rsi24": v.get("rsi24")}
        for k, v in tf_indicators.items()
    }

    if quote.get("error"):
        errors.append(f"quote: {quote['error']}")

    ob5 = quote.get("order_book_5", {"buy": [], "sell": []})
    non_null_levels = 0
    for side in ("buy", "sell"):
        for row in ob5.get(side, []):
            if row.get("price") is not None and row.get("volume_lot") is not None:
                non_null_levels += 1

    exchange = _resolve_exchange(normalized)
    if exchange == "hk" and non_null_levels <= 2:
        depth_note = "港股免费接口通常仅稳定提供买1/卖1，买2-买5与卖2-卖5可能为空。"
    else:
        depth_note = "当前使用公开免费接口的买5卖5盘口数据。"

    return {
        "symbol": normalized,
        "quote": quote,
        "indicators": indicators,
        "intraday": intraday_df,
        "order_book_5": ob5,
        "order_book_10": quote.get("order_book_10", {"buy": [], "sell": []}),
        "rsi_multi": rsi_multi,
        "tf_indicators": tf_indicators,
        "depth_note": depth_note,
        "error": " | ".join(errors) if errors else None,
    }


def run_realtime_demo(symbol: str = "601088") -> None:
    panel = fetch_fast_panel(symbol)
    quote = panel["quote"]
    print(
        f"[FastEngine] {quote.get('symbol')} {quote.get('name')} "
        f"price={quote.get('current_price')} vwap={quote.get('vwap')} "
        f"macd={panel['indicators'].get('macd_hist')} rsi6={panel['indicators'].get('rsi6')}"
    )
    print("[FastEngine] orderbook buy levels:", panel["order_book_10"].get("buy", [])[:5])


if __name__ == "__main__":
    run_realtime_demo("601088")
