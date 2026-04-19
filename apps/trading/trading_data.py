import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from fast_engine import _get_data_provider, fetch_fast_panel

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_PREFS_PATH = "data/local_user_prefs.json"
ANALYSIS_CACHE_PATH = "data/deepseek_analysis_cache.json"

FUNDAMENTAL_DIR = CURRENT_DIR.parent / "fundamental"
if str(FUNDAMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(FUNDAMENTAL_DIR))

from fundamental_engine import analyze_watchlist as analyze_fundamental_watchlist

from shared.valuation_percentile import ValuationPercentile


def _load_trading_slow_engine():
    """强制按文件路径加载 trading/slow_engine.py，避免同名模块冲突。"""
    module_path = CURRENT_DIR / "slow_engine.py"
    module_name = "quant_agent_trading_slow_engine"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载交易慢引擎: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    required = [
        "add_stock_by_query",
        "fetch_live_valuation_snapshot",
        "get_stock_pool",
        "get_latest_fundamental_snapshot",
        "get_stock_group_map",
        "init_db",
        "remove_stock_from_pool",
        "update_fundamental_data",
    ]
    for attr in required:
        if not hasattr(module, attr):
            raise ImportError(f"交易慢引擎缺少属性 {attr}: {module_path}")
    return module


trading_slow_engine = _load_trading_slow_engine()
add_stock_by_query = trading_slow_engine.add_stock_by_query
fetch_live_valuation_snapshot = trading_slow_engine.fetch_live_valuation_snapshot
get_stock_pool = trading_slow_engine.get_stock_pool
get_latest_fundamental_snapshot = trading_slow_engine.get_latest_fundamental_snapshot
get_stock_group_map = trading_slow_engine.get_stock_group_map
init_db = trading_slow_engine.init_db
remove_stock_from_pool = trading_slow_engine.remove_stock_from_pool
update_fundamental_data = trading_slow_engine.update_fundamental_data

def _load_local_prefs() -> dict:
    try:
        if not os.path.exists(LOCAL_PREFS_PATH):
            return {}
        with open(LOCAL_PREFS_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _save_local_prefs(username: str, api_key: str) -> None:
    os.makedirs(os.path.dirname(LOCAL_PREFS_PATH), exist_ok=True)
    payload = {
        "deepseek_user": (username or "").strip(),
        "deepseek_api_key": (api_key or "").strip(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(LOCAL_PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_analysis_cache() -> dict:
    try:
        if not os.path.exists(ANALYSIS_CACHE_PATH):
            return {}
        with open(ANALYSIS_CACHE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _save_analysis_cache(cache_obj: dict) -> None:
    os.makedirs(os.path.dirname(ANALYSIS_CACHE_PATH), exist_ok=True)
    # 控制缓存大小，避免无限增长
    items = list(cache_obj.items())
    if len(items) > 120:
        items = items[-120:]
    with open(ANALYSIS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(items), f, ensure_ascii=False, indent=2)

def _load_json_file(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _save_json_file(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _dict_delta(curr, prev):
    if isinstance(curr, dict) and isinstance(prev, dict):
        out = {}
        for k, v in curr.items():
            d = _dict_delta(v, prev.get(k))
            if d is not None:
                out[k] = d
        return out if out else None
    if isinstance(curr, list) and isinstance(prev, list):
        return curr if curr != prev else None
    return curr if curr != prev else None

def _format_display_time(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    text = str(v).strip()
    if not text:
        return None
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(text, format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(dt):
        return None
    return dt.strftime("%m-%d %H:%M:%S")

def _infer_market_for_percentile(code: str, row: dict) -> str:
    market_raw = str(row.get("market") or "").strip().upper()
    if market_raw in {"HK", "A"}:
        return market_raw
    code_text = str(code or "").strip().upper()
    if code_text.endswith(".HK"):
        return "HK"
    digits = "".join(ch for ch in code_text if ch.isdigit())
    return "HK" if len(digits) <= 5 else "A"

@st.cache_data(ttl=1800, show_spinner=False)
def _cached_valuation_percentile(market: str, code: str, lookback_years: int = 5) -> dict:
    vp = ValuationPercentile()
    return vp.compute_multi_percentile(market=market, code=code, lookback_years=lookback_years)

def _shared_watchlist_rows():
    pool_rows = get_stock_pool()
    group_map = get_stock_group_map()
    out = []
    for code, name in pool_rows:
        out.append(
            {
                "code": str(code),
                "name": str(name).strip() or str(code),
                "type": "持仓" if group_map.get(str(code), "watch") == "holding" else "观察",
            }
        )
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_analyze_watchlist_rows(watchlist_payload: str) -> list:
    try:
        watchlist = json.loads(watchlist_payload)
    except Exception:
        watchlist = []
    if not isinstance(watchlist, list):
        watchlist = []
    return analyze_fundamental_watchlist(watchlist, force_refresh=False)

def _ensure_fundamental_state(force_refresh: bool = False):
    if "fnd_deepseek_reports" not in st.session_state:
        st.session_state["fnd_deepseek_reports"] = {}
    watchlist = _shared_watchlist_rows()
    hash_text = json.dumps(watchlist, ensure_ascii=False, sort_keys=True)
    wl_hash = hashlib.md5(hash_text.encode("utf-8")).hexdigest()

    stale = (
        force_refresh
        or "fnd_rows" not in st.session_state
        or st.session_state.get("fnd_watchlist_hash", "") != wl_hash
    )
    if stale:
        if force_refresh:
            st.session_state["fnd_rows"] = analyze_fundamental_watchlist(watchlist, force_refresh=True)
        else:
            st.session_state["fnd_rows"] = _cached_analyze_watchlist_rows(hash_text)
        st.session_state["fnd_watchlist_hash"] = wl_hash
        if st.session_state["fnd_rows"]:
            valid_codes = {str(x.get("code", "")) for x in st.session_state["fnd_rows"]}
            if st.session_state.get("fnd_selected_code") not in valid_codes:
                st.session_state["fnd_selected_code"] = st.session_state["fnd_rows"][0]["code"]
        else:
            st.session_state["fnd_selected_code"] = ""
    return watchlist, st.session_state.get("fnd_rows", [])

def _safe_str(v: object) -> str:
    return str(v).strip() if v is not None else ""

def _safe_int(v: object) -> int:
    try:
        return int(float(str(v)))
    except Exception:
        return 0

def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out = out.replace({None: "-", "None": "-", "nan": "-", "NaN": "-", "N/A": "-"})
    out = out.fillna("-")
    return out

def _is_hk_code(code: str) -> bool:
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    return len(digits) == 5

def _is_market_open(code: str) -> bool:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if now.weekday() >= 5:
        return False

    t = now.time()
    if _is_hk_code(code):
        # 港股常规交易时段（简化口径）
        return (time(9, 30) <= t <= time(12, 0)) or (time(13, 0) <= t <= time(16, 0))

    # A股常规交易时段
    return (time(9, 30) <= t <= time(11, 30)) or (time(13, 0) <= t <= time(15, 0))

def _calc_display_change_pct(quote: dict) -> float:
    current_price = quote.get("current_price")
    prev_close = quote.get("prev_close")
    api_change_pct = quote.get("change_pct")
    if (
        current_price is not None
        and prev_close is not None
        and prev_close > 0
    ):
        return float((current_price - prev_close) / prev_close * 100)
    return float(api_change_pct or 0.0)

@st.cache_data(ttl=180, show_spinner=False)
def _fetch_chart_ohlcv(symbol: str, count: int = 260) -> pd.DataFrame:
    provider = _get_data_provider()
    df = provider.get_kline(str(symbol), count=int(max(120, count)))
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    data = df.copy()
    if "date" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={"index": "date"})
        else:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    for col in ("open", "high", "low", "close", "volume"):
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").tail(count)
    if data.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    return data[["date", "open", "high", "low", "close", "volume"]].copy()

def _load_fast_panel_snapshot_once(selected_code: str) -> dict:
    cache_key = f"fast_panel_cache_{selected_code}"
    market_open = _is_market_open(selected_code)
    if market_open:
        panel_now = fetch_fast_panel(selected_code)
        st.session_state[cache_key] = panel_now
        return panel_now
    cached = st.session_state.get(cache_key)
    if isinstance(cached, dict):
        return cached
    panel_now = fetch_fast_panel(selected_code)
    st.session_state[cache_key] = panel_now
    return panel_now
