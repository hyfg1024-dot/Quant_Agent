from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd

import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.data_provider import AkshareDataProvider, FallbackDataProvider, QMTDataProvider


@lru_cache(maxsize=1)
def _get_data_provider() -> FallbackDataProvider:
    """基本面模块专用数据源代理：优先 QMT，失败自动回退 Akshare。"""
    return FallbackDataProvider(
        primary=QMTDataProvider(timeout_sec=1.2, enabled=True),
        fallback=AkshareDataProvider(),
    )


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol).strip().lower()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if raw.startswith("hk"):
        return digits[-5:].zfill(5) if digits else raw.replace("hk", "").strip()
    if len(digits) == 5:
        return digits
    if len(digits) >= 6:
        return digits[-6:]
    return str(symbol).strip()


def fetch_realtime_quote(symbol: str) -> Dict:
    """返回与交易模块一致风格的实时快照 Dict。"""
    normalized = _normalize_symbol(symbol)
    try:
        quote = _get_data_provider().get_quote(normalized)
    except Exception as exc:  # pragma: no cover
        return {
            "symbol": normalized,
            "name": normalized,
            "current_price": None,
            "pe_dynamic": None,
            "pe_ttm": None,
            "pb": None,
            "error": str(exc),
        }
    quote.setdefault("symbol", normalized)
    quote.setdefault("name", normalized)
    quote.setdefault("error", None)
    return quote


def fetch_daily_kline(symbol: str, count: int = 320) -> pd.DataFrame:
    """返回统一列的日线 DataFrame：date/open/high/low/close/volume。"""
    normalized = _normalize_symbol(symbol)
    try:
        df = _get_data_provider().get_kline(normalized, count=int(count or 320))
    except Exception:  # pragma: no cover
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    out = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    return out[["date", "open", "high", "low", "close", "volume"]]
