from __future__ import annotations

import copy
import io
import json
import os
import random
import re
import sqlite3
import socket
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd
import requests

CURRENT_DIR = Path(__file__).resolve().parent
FUNDAMENTAL_DIR = CURRENT_DIR.parent / "fundamental"
if str(FUNDAMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(FUNDAMENTAL_DIR))

from fundamental_engine import analyze_fundamental

APP_VERSION = "FLT-20260418-03"
DATA_DIR = CURRENT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "filter_market.db"
TEMPLATE_FILE = DATA_DIR / "filter_templates.json"
MANUAL_FLAGS_FILE = DATA_DIR / "manual_flags.json"

DEFAULT_SUNSET_INDUSTRIES = [
    "传统燃油汽车整车",
    "传统纸质媒体",
    "功能手机及相关",
    "胶卷及相关",
    "煤炭开采",
    "传统钢铁冶炼",
]

INDUSTRY_KEYWORD_ALIAS_MAP: Dict[str, List[str]] = {
    # 港股地产常见表达
    "房地产": ["地产", "物业", "reits", "reit"],
    "地产": ["房地产", "物业", "reits", "reit"],
    "物业": ["房地产", "地产", "reits", "reit"],
    "reits": ["reit", "房地产", "地产", "物业"],
    "reit": ["reits", "房地产", "地产", "物业"],
}

DEFAULT_FILTER_CONFIG: Dict[str, Any] = {
    "missing_policy": "ignore",  # ignore / exclude
    "risk": {
        "market_scope": "all",  # all / A / HK
        "industry_include_enabled": False,
        "industry_include_keywords": "",
        "exclude_st": True,
        "exclude_investigation": True,
        "exclude_penalty": True,
        "exclude_fund_occupation": True,
        "exclude_illegal_reduce": True,
        "require_standard_audit": False,
        "exclude_sunset_industry": False,
        "sunset_industries": "，".join(DEFAULT_SUNSET_INDUSTRIES),
        "pledge_ratio_max_enabled": False,
        "pledge_ratio_max": 80.0,
        "audit_change_max_enabled": False,
        "audit_change_max": 2,
        "exclude_no_dividend_5y": False,
    },
    "quality": {
        "ocf_3y_min_enabled": False,
        "ocf_3y_min": 0.0,
        "asset_liability_max_enabled": False,
        "asset_liability_max": 80.0,
        "interest_debt_asset_max_enabled": False,
        "interest_debt_asset_max": 20.0,
        "roe_min_enabled": False,
        "roe_min": 5.0,
        "gross_margin_min_enabled": False,
        "gross_margin_min": 20.0,
        "net_margin_min_enabled": False,
        "net_margin_min": 8.0,
        "receivable_ratio_max_enabled": False,
        "receivable_ratio_max": 50.0,
        "goodwill_ratio_max_enabled": False,
        "goodwill_ratio_max": 30.0,
    },
    "valuation": {
        "pe_ttm_min_enabled": False,
        "pe_ttm_min": 0.0,
        "pe_ttm_max_enabled": False,
        "pe_ttm_max": 25.0,
        "pb_max_enabled": False,
        "pb_max": 3.0,
        "ev_ebitda_max_enabled": False,
        "ev_ebitda_max": 18.0,
        "dividend_min_enabled": False,
        "dividend_min": 3.0,
        "dividend_max_enabled": False,
        "dividend_max": 12.0,
    },
    "growth_liquidity": {
        "revenue_growth_min_enabled": False,
        "revenue_growth_min": 0.0,
        "profit_growth_min_enabled": False,
        "profit_growth_min": 0.0,
        "market_cap_min_enabled": False,
        "market_cap_min": 100.0,  # 亿
        "market_cap_max_enabled": False,
        "market_cap_max": 5000.0,
        "turnover_min_enabled": False,
        "turnover_min": 0.2,
        "turnover_max_enabled": False,
        "turnover_max": 15.0,
        "volume_ratio_min_enabled": False,
        "volume_ratio_min": 0.5,
        "volume_ratio_max_enabled": False,
        "volume_ratio_max": 3.0,
        "amount_min_enabled": False,
        "amount_min": 100000000.0,  # 1亿
    },
    "rearview_5y": {
        "revenue_cagr_5y_min_enabled": False,
        "revenue_cagr_5y_min": 3.0,
        "profit_cagr_5y_min_enabled": False,
        "profit_cagr_5y_min": 3.0,
        "roe_avg_5y_min_enabled": False,
        "roe_avg_5y_min": 8.0,
        "ocf_positive_years_5y_min_enabled": False,
        "ocf_positive_years_5y_min": 4,
        "debt_ratio_change_5y_max_enabled": False,
        "debt_ratio_change_5y_max": 8.0,
        "gross_margin_change_5y_min_enabled": False,
        "gross_margin_change_5y_min": -6.0,
    },
}

DISPLAY_COLUMNS = [
    "market",
    "code",
    "name",
    "industry",
    "pe_ttm",
    "pb",
    "dividend_yield",
    "roe",
    "asset_liability_ratio",
    "turnover_ratio",
    "volume_ratio",
    "total_mv",
    "revenue_cagr_5y",
    "profit_cagr_5y",
    "roe_avg_5y",
    "ocf_positive_years_5y",
    "debt_ratio_change_5y",
    "gross_margin_change_5y",
    "data_quality",
    "exclude_reasons",
    "missing_fields",
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    ensure_dirs()
    return sqlite3.connect(DB_PATH)


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return float(v)
    text = str(v).strip().replace(",", "")
    if text in {"", "-", "--", "nan", "NaN", "None"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_mv_100m(v: Any) -> Optional[float]:
    num = _to_float(v)
    if num is None:
        return None
    # 东方财富现货一般是“元”口径，这里统一转换到“亿”
    if abs(num) > 1_000_000:
        return num / 100_000_000
    return num


def _safe_str(v: Any) -> str:
    return str(v).strip() if v is not None else ""


_ENV_MISSING = object()
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)
_MARKET_NETWORK_MODE_CACHE: Dict[str, Any] = {"mode": "auto", "ts": 0.0}
_DEFAULT_SURGE_PROXY = "http://127.0.0.1:6152"
_LOCAL_PROXY_CANDIDATES = (
    "http://127.0.0.1:6152",  # Surge 常见默认
    "http://127.0.0.1:7890",  # Clash 常见默认
    "http://127.0.0.1:9090",  # 其他代理常见端口
)
_LOCAL_PROXY_CACHE: Dict[str, Any] = {"url": _DEFAULT_SURGE_PROXY, "ok": False, "ts": 0.0}


def _local_proxy_url() -> str:
    """
    代码层本机代理入口：
    1) 优先读取 QUANT_LOCAL_HTTP_PROXY；
    2) 否则自动探测常见本机端口。
    """
    env_url = _safe_str(os.getenv("QUANT_LOCAL_HTTP_PROXY", ""))
    if env_url:
        return env_url
    now_ts = time.time()
    if now_ts - float(_LOCAL_PROXY_CACHE.get("ts", 0.0) or 0.0) < 60:
        return _safe_str(_LOCAL_PROXY_CACHE.get("url", _DEFAULT_SURGE_PROXY)) or _DEFAULT_SURGE_PROXY
    env_candidates: List[str] = []
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        v = _safe_str(os.getenv(k, ""))
        if v:
            env_candidates.append(v)
    candidates = tuple(dict.fromkeys([*env_candidates, *_LOCAL_PROXY_CANDIDATES]))
    for u in candidates:
        if _is_proxy_url_reachable(u):
            _LOCAL_PROXY_CACHE["url"] = u
            _LOCAL_PROXY_CACHE["ok"] = True
            _LOCAL_PROXY_CACHE["ts"] = now_ts
            return u
    _LOCAL_PROXY_CACHE["url"] = _DEFAULT_SURGE_PROXY
    _LOCAL_PROXY_CACHE["ok"] = False
    _LOCAL_PROXY_CACHE["ts"] = now_ts
    return _DEFAULT_SURGE_PROXY


def _force_surge_local_proxy() -> bool:
    return _safe_str(os.getenv("QUANT_FORCE_SURGE_PROXY", "0")).lower() in {"1", "true", "yes", "on"}


def _is_proxy_url_reachable(proxy_url: str, timeout: float = 0.6) -> bool:
    """
    探测本机代理端口是否可用；用于在 Surge 增强模式下自动切换到本地代理通道。
    """
    url = _safe_str(proxy_url)
    m = re.match(r"^https?://([^:/]+):(\d+)", url)
    if not m:
        return False
    host = _safe_str(m.group(1)) or "127.0.0.1"
    try:
        port = int(m.group(2))
    except Exception:
        return False


def _is_local_proxy_reachable(proxy_url: Optional[str] = None, timeout: float = 0.6) -> bool:
    url = _safe_str(proxy_url or _local_proxy_url())
    return _is_proxy_url_reachable(url, timeout=timeout)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _market_probe_request(mode: str = "direct") -> bool:
    """
    轻量探测东财接口可达性。
    mode:
      - direct: 不读取系统代理
      - proxy: 读取系统代理（Surge 场景）
    """
    hosts = [
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
        "http://82.push2.eastmoney.com/api/qt/clist/get",
        "http://push2.eastmoney.com/api/qt/clist/get",
    ]
    params = {
        "pn": "1",
        "pz": "1",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f12,f14",
    }
    proxy_url = _local_proxy_url()
    for url in hosts:
        try:
            ses = requests.Session()
            if mode == "surge_local":
                ses.trust_env = False
                ses.proxies.update({"http": proxy_url, "https": proxy_url})
            else:
                ses.trust_env = bool(mode == "proxy")
            resp = ses.get(
                url,
                params=params,
                timeout=6,
                headers={"User-Agent": "Mozilla/5.0", "Connection": "close", "Accept": "application/json,text/plain,*/*"},
            )
            resp.raise_for_status()
            obj = resp.json() or {}
            diff = ((obj.get("data") or {}).get("diff") or [])
            if isinstance(diff, list):
                return True
        except Exception:
            continue
    return False


def _resolve_market_network_mode(force: bool = False) -> str:
    """
    自动判定更新通道：
      - direct: 本机直连可用
      - proxy: 需经系统代理（Surge）可用
      - none: 两者都不可用
    """
    now_ts = time.time()
    if (not force) and (now_ts - float(_MARKET_NETWORK_MODE_CACHE.get("ts", 0.0) or 0.0) < 120):
        return _safe_str(_MARKET_NETWORK_MODE_CACHE.get("mode", "auto")) or "auto"

    force_surge = _force_surge_local_proxy()
    local_proxy_ok = _is_local_proxy_reachable()

    mode = "none"
    if force_surge and local_proxy_ok:
        mode = "surge_local"
    elif local_proxy_ok:
        mode = "surge_local"
    elif _market_probe_request("direct"):
        mode = "direct"
    elif _market_probe_request("proxy"):
        mode = "proxy"

    _MARKET_NETWORK_MODE_CACHE["mode"] = mode
    _MARKET_NETWORK_MODE_CACHE["ts"] = now_ts
    return mode


def _ak_call_with_proxy_fallback(func, *args, **kwargs):
    last_exc: Optional[Exception] = None

    # 第一轮：按当前环境直接请求
    for _ in range(2):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.8)

    # 第二轮：临时关闭代理环境变量后重试
    backup = {k: os.environ.get(k, _ENV_MISSING) for k in _PROXY_ENV_KEYS}
    try:
        for k in _PROXY_ENV_KEYS:
            os.environ.pop(k, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        for _ in range(2):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                time.sleep(0.8)
    finally:
        for k, v in backup.items():
            if v is _ENV_MISSING:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    # 第三轮：强制注入本机代理（Surge 增强模式场景）
    if _force_surge_local_proxy():
        proxy_url = _local_proxy_url()
        backup2 = {k: os.environ.get(k, _ENV_MISSING) for k in _PROXY_ENV_KEYS}
        try:
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            os.environ["http_proxy"] = proxy_url
            os.environ["https_proxy"] = proxy_url
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)
            for _ in range(2):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.8)
        finally:
            for k, v in backup2.items():
                if v is _ENV_MISSING:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = str(v)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("AKShare 请求失败")


def _pick_series(df: pd.DataFrame, names: List[str]) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index)


def _clean_text_series(s: pd.Series) -> pd.Series:
    def _one(v: Any) -> str:
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        t = str(v).strip()
        return "" if t in {"None", "nan", "NaN"} else t

    return s.map(_one)


def _normalize_hk_company_code(v: Any) -> str:
    """
    港股公司口径：RMB 柜台(8xxxx)并入对应 HKD 柜台(0xxxx)，避免同一公司双柜台重复计数。
    """
    text = re.sub(r"\D+", "", _safe_str(v)).zfill(5)
    if len(text) != 5:
        return ""
    if text.startswith("8"):
        return "0" + text[1:]
    return text


def _snapshot_meta_set(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO snapshot_meta(meta_key, meta_value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()


def get_snapshot_meta() -> Dict[str, str]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT meta_key, meta_value FROM snapshot_meta").fetchall()
    return {str(k): str(v) for k, v in rows}


def get_weekly_update_status(market_scope: str = "AH") -> Dict[str, Any]:
    scope = _safe_str(market_scope).upper() or "AH"
    meta = get_snapshot_meta()
    key = f"last_weekly_update_{scope}"
    last_text = _safe_str(meta.get(key, ""))
    now = datetime.now()
    if not last_text:
        return {"scope": scope, "due": True, "last": "", "next_due": "", "remaining_hours": 0.0}
    last_dt: Optional[datetime] = None
    try:
        last_dt = datetime.fromisoformat(last_text)
    except Exception:
        try:
            last_dt = datetime.strptime(last_text, "%Y-%m-%d %H:%M:%S")
        except Exception:
            last_dt = None
    if last_dt is None:
        return {"scope": scope, "due": True, "last": last_text, "next_due": "", "remaining_hours": 0.0}
    next_due_dt = last_dt + timedelta(days=7)
    remaining_hours = max(0.0, (next_due_dt - now).total_seconds() / 3600.0)
    return {
        "scope": scope,
        "due": remaining_hours <= 0,
        "last": last_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "next_due": next_due_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "remaining_hours": remaining_hours,
    }


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshot (
                market TEXT,
                code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                is_st INTEGER,
                close_price REAL,
                price_change_pct REAL,
                amount REAL,
                pe_dynamic REAL,
                pe_static REAL,
                pe_ttm REAL,
                pb REAL,
                dividend_yield REAL,
                total_mv REAL,
                float_mv REAL,
                turnover_ratio REAL,
                volume_ratio REAL,
                roe REAL,
                gross_margin REAL,
                net_margin REAL,
                asset_liability_ratio REAL,
                current_ratio REAL,
                operating_cashflow_3y REAL,
                receivable_revenue_ratio REAL,
                goodwill_equity_ratio REAL,
                interest_debt_asset_ratio REAL,
                ev_ebitda REAL,
                revenue_growth REAL,
                profit_growth REAL,
                revenue_cagr_5y REAL,
                profit_cagr_5y REAL,
                roe_avg_5y REAL,
                debt_ratio_avg_5y REAL,
                gross_margin_avg_5y REAL,
                debt_ratio_change_5y REAL,
                gross_margin_change_5y REAL,
                ocf_positive_years_5y REAL,
                investigation_flag INTEGER,
                penalty_flag INTEGER,
                fund_occupation_flag INTEGER,
                illegal_reduce_flag INTEGER,
                pledge_ratio REAL,
                no_dividend_5y_flag INTEGER,
                audit_change_count INTEGER,
                audit_opinion TEXT,
                sunset_industry_flag INTEGER,
                total_score REAL,
                conclusion TEXT,
                coverage_ratio REAL,
                data_quality TEXT,
                enriched_at TEXT,
                updated_at TEXT,
                source_note TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_meta (
                meta_key TEXT PRIMARY KEY,
                meta_value TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT,
                row_count INTEGER,
                enriched_count INTEGER,
                enrich_start INTEGER,
                enrich_end INTEGER,
                fallback INTEGER,
                error_brief TEXT,
                cache_hit INTEGER,
                cache_miss INTEGER,
                enrich_mode TEXT
            )
            """
        )
        conn.commit()


def default_filter_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_FILTER_CONFIG)


def load_templates() -> Dict[str, Dict[str, Any]]:
    ensure_dirs()
    if not TEMPLATE_FILE.exists():
        return {}
    try:
        obj = json.loads(TEMPLATE_FILE.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_template(name: str, config: Dict[str, Any]) -> None:
    key = _safe_str(name)
    if not key:
        raise ValueError("模板名不能为空")
    all_tpl = load_templates()
    all_tpl[key] = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
    }
    TEMPLATE_FILE.write_text(json.dumps(all_tpl, ensure_ascii=False, indent=2), encoding="utf-8")


def get_template_config(name: str) -> Dict[str, Any]:
    all_tpl = load_templates()
    one = all_tpl.get(name, {})
    cfg = one.get("config") if isinstance(one, dict) else {}
    return cfg if isinstance(cfg, dict) else default_filter_config()


def _load_manual_flags() -> Dict[str, Dict[str, Any]]:
    ensure_dirs()
    if not MANUAL_FLAGS_FILE.exists():
        MANUAL_FLAGS_FILE.write_text("{}", encoding="utf-8")
        return {}
    try:
        obj = json.loads(MANUAL_FLAGS_FILE.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _cache_file(code: str) -> Path:
    return CACHE_DIR / f"enrich_{code}.json"


def _load_enrich_cache(code: str, ttl_days: int = 7) -> Optional[Dict[str, Any]]:
    p = _cache_file(code)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(str(obj.get("cached_at")))
        if datetime.now() - ts <= timedelta(days=ttl_days):
            return obj
    except Exception:
        return None
    return None


def _save_enrich_cache(code: str, payload: Dict[str, Any]) -> None:
    p = _cache_file(code)
    data = dict(payload)
    data["cached_at"] = datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_sunset(industry: str, name: str, keywords: Optional[List[str]] = None) -> bool:
    words = keywords or DEFAULT_SUNSET_INDUSTRIES
    text = f"{_safe_str(industry)} {_safe_str(name)}"
    return any(w and w in text for w in words)


def _enrich_one(code: str, name: str, force_refresh: bool = False) -> Tuple[Dict[str, Any], str, Optional[str]]:
    if not force_refresh:
        cached = _load_enrich_cache(code)
        if cached:
            return cached, "cache", _safe_str(cached.get("cached_at")) or None

    result = analyze_fundamental(code=code, name=name, force_refresh=force_refresh, cache_ttl_hours=168)
    payload = {
        "pe_dynamic": _to_float(result.get("pe_dynamic")),
        "pe_static": _to_float(result.get("pe_static")),
        "pe_ttm": _to_float(result.get("pe_ttm")),
        "pb": _to_float(result.get("pb")),
        "dividend_yield": _to_float(result.get("dividend_yield")),
        "total_mv": _to_mv_100m(result.get("total_mv")),
        "roe": _to_float(result.get("roe")),
        "gross_margin": _to_float(result.get("gross_margin")),
        "net_margin": _to_float(result.get("net_margin")),
        "asset_liability_ratio": _to_float(result.get("debt_ratio")),
        "current_ratio": _to_float(result.get("current_ratio")),
        "operating_cashflow_3y": (
            (_to_float(result.get("ocf_sum_3y")) / 100000000) if _to_float(result.get("ocf_sum_3y")) is not None else None
        ),
        "receivable_revenue_ratio": _to_float(result.get("receivable_days")),
        "goodwill_equity_ratio": _to_float(result.get("goodwill_ratio_pct")),
        "interest_debt_asset_ratio": None,
        "ev_ebitda": None,
        "revenue_growth": _to_float(result.get("revenue_growth")),
        "profit_growth": _to_float(result.get("profit_growth")),
        "revenue_cagr_5y": _to_float(result.get("revenue_cagr_5y")),
        "profit_cagr_5y": _to_float(result.get("profit_cagr_5y")),
        "roe_avg_5y": _to_float(result.get("roe_avg_5y")),
        "debt_ratio_avg_5y": _to_float(result.get("debt_ratio_avg_5y")),
        "gross_margin_avg_5y": _to_float(result.get("gross_margin_avg_5y")),
        "debt_ratio_change_5y": _to_float(result.get("debt_ratio_change_5y")),
        "gross_margin_change_5y": _to_float(result.get("gross_margin_change_5y")),
        "ocf_positive_years_5y": _to_float(result.get("ocf_positive_years_5y")),
        "total_score": _to_float(result.get("total_score")),
        "conclusion": _safe_str(result.get("conclusion")) or "观察",
        "coverage_ratio": _to_float(result.get("coverage_ratio")),
        "audit_opinion": "标准无保留意见",
    }
    _save_enrich_cache(code, payload)
    return payload, "live", datetime.now().isoformat(timespec="seconds")


def _build_universe_from_spot(spot_df: pd.DataFrame, market: str) -> pd.DataFrame:
    mkt = _safe_str(market).upper()
    if mkt == "HK":
        raw_code = _pick_series(spot_df, ["代码", "证券代码", "symbol", "Symbol"]).astype(str).str.extract(r"(\d+)")[0].fillna("")
        code = raw_code.map(_normalize_hk_company_code)
    else:
        code = _pick_series(spot_df, ["代码"]).astype(str).str.strip().str.zfill(6)
    name = _clean_text_series(_pick_series(spot_df, ["名称", "股票名称", "简称"]))
    industry = _clean_text_series(_pick_series(spot_df, ["所处行业", "所属行业", "行业", "industry"]))

    df = pd.DataFrame(
        {
            "market": mkt,
            "code": code,
            "name": name,
            "industry": industry,
            "is_st": name.str.contains("ST", na=False).astype(int),
            "close_price": _pick_series(spot_df, ["最新价", "最新", "收盘"]).map(_to_float),
            "price_change_pct": _pick_series(spot_df, ["涨跌幅", "涨跌幅(%)"]).map(_to_float),
            "amount": _pick_series(spot_df, ["成交额"]).map(_to_float),
            "pe_dynamic": _pick_series(spot_df, ["市盈率-动态", "市盈率动态", "市盈率"]).map(_to_float),
            "pb": _pick_series(spot_df, ["市净率"]).map(_to_float),
            "dividend_yield": _pick_series(spot_df, ["股息率", "股息率(%)"]).map(_to_float),
            "total_mv": _pick_series(spot_df, ["总市值"]).map(_to_mv_100m),
            "float_mv": _pick_series(spot_df, ["流通市值"]).map(_to_mv_100m),
            "turnover_ratio": _pick_series(spot_df, ["换手率"]).map(_to_float),
            "volume_ratio": _pick_series(spot_df, ["量比"]).map(_to_float),
        }
    )
    df = df[df["code"].astype(str).str.len() > 0].copy()
    if mkt == "HK":
        # 同公司双柜台去重，优先保留成交额/市值更完整的一条。
        df["_amt_sort"] = pd.to_numeric(df["amount"], errors="coerce")
        df["_mv_sort"] = pd.to_numeric(df["total_mv"], errors="coerce")
        df = (
            df.sort_values(by=["_amt_sort", "_mv_sort", "code"], ascending=[False, False, True], na_position="last")
            .drop_duplicates(subset=["code"], keep="first")
            .drop(columns=["_amt_sort", "_mv_sort"], errors="ignore")
            .reset_index(drop=True)
        )
    df["pe_ttm"] = df["pe_dynamic"]
    df["pe_static"] = None
    df["roe"] = None
    df["gross_margin"] = None
    df["net_margin"] = None
    df["asset_liability_ratio"] = None
    df["current_ratio"] = None
    df["operating_cashflow_3y"] = None
    df["receivable_revenue_ratio"] = None
    df["goodwill_equity_ratio"] = None
    df["interest_debt_asset_ratio"] = None
    df["ev_ebitda"] = None
    df["revenue_growth"] = None
    df["profit_growth"] = None
    df["revenue_cagr_5y"] = None
    df["profit_cagr_5y"] = None
    df["roe_avg_5y"] = None
    df["debt_ratio_avg_5y"] = None
    df["gross_margin_avg_5y"] = None
    df["debt_ratio_change_5y"] = None
    df["gross_margin_change_5y"] = None
    df["ocf_positive_years_5y"] = None
    df["total_score"] = None
    df["coverage_ratio"] = None
    df["conclusion"] = "观察"

    df["investigation_flag"] = 0
    df["penalty_flag"] = 0
    df["fund_occupation_flag"] = 0
    df["illegal_reduce_flag"] = 0
    df["pledge_ratio"] = None
    df["no_dividend_5y_flag"] = 0
    df["audit_change_count"] = 0
    df["audit_opinion"] = "标准无保留意见"

    df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["source_note"] = f"ak.spot_{mkt.lower()} + fundamental_enrich"
    return df


def _fetch_hk_spot_ak(network_mode: str = "auto") -> pd.DataFrame:
    # 港股口径固定为“主板+GEM公司”，直连失败时由上层走快照回退，不降级为全证券口径。
    return _fetch_hk_spot_em_direct(network_mode=network_mode)


def _fetch_hk_spot_em_direct(network_mode: str = "auto") -> pd.DataFrame:
    """
    港股东方财富直连（禁用系统代理），优先获取市值等字段，避免 AKShare 精简字段导致大量空值。
    """
    hosts = [
        "https://72.push2.eastmoney.com/api/qt/clist/get",
        "https://81.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
        "http://72.push2.eastmoney.com/api/qt/clist/get",
        "http://81.push2.eastmoney.com/api/qt/clist/get",
        "http://push2.eastmoney.com/api/qt/clist/get",
    ]
    base_params = {
        "pn": "1",
        "pz": "200",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        # 主板 + GEM（上市公司口径），不混入其他证券类型
        "fs": "m:128 t:3,m:128 t:4",
        "fields": ",".join(
            [
                "f2",   # 最新价
                "f3",   # 涨跌幅
                "f6",   # 成交额
                "f8",   # 换手率
                "f9",   # 市盈率-动态
                "f10",  # 量比
                "f12",  # 代码
                "f14",  # 名称
                "f20",  # 总市值
                "f21",  # 流通市值
                "f23",  # 市净率
                "f100", # 行业
                "f127", # 行业(备用)
            ]
        ),
    }

    mode = _safe_str(network_mode).lower() or "auto"
    if mode == "auto":
        mode = _resolve_market_network_mode(force=False)
    if mode in {"direct", "proxy", "surge_local"}:
        mode_order = [mode]
    else:
        mode_order = (
            ["surge_local", "proxy", "direct"]
            if (_force_surge_local_proxy() or _is_local_proxy_reachable())
            else ["direct", "proxy"]
        )
    proxy_url = _local_proxy_url()

    last_exc: Optional[Exception] = None
    for req_mode in mode_order:
        for url in hosts:
            for _ in range(2):
                try:
                    rows: List[Dict[str, Any]] = []
                    seen_codes: set[str] = set()
                    total_hint: Optional[int] = None
                    ses = requests.Session()
                    if req_mode == "surge_local":
                        ses.trust_env = False
                        ses.proxies.update({"http": proxy_url, "https": proxy_url})
                    else:
                        ses.trust_env = bool(req_mode == "proxy")
                    for page in range(1, 120):
                        params = dict(base_params)
                        params["pn"] = str(page)
                        try:
                            resp = ses.get(
                                url,
                                params=params,
                                timeout=18,
                                headers={
                                    "User-Agent": "Mozilla/5.0",
                                    "Connection": "close",
                                    "Accept": "application/json,text/plain,*/*",
                                },
                            )
                            resp.raise_for_status()
                            obj = resp.json()
                            data_obj = (obj or {}).get("data") or {}
                            if total_hint is None:
                                try:
                                    total_hint = int(_to_float(data_obj.get("total")) or 0)
                                except Exception:
                                    total_hint = None
                            diff = data_obj.get("diff") or []
                        except Exception as page_exc:
                            last_exc = page_exc
                            # 分页中断时保留已抓到的数据，避免整批回退
                            break

                        if not diff:
                            break

                        page_new = 0
                        for it in diff:
                            raw = it or {}
                            code = str(raw.get("f12", "")).strip()
                            if (not code) or (code in seen_codes):
                                continue
                            seen_codes.add(code)
                            page_new += 1
                            industry = _safe_str(raw.get("f100")) or _safe_str(raw.get("f127"))
                            rows.append(
                                {
                                    "代码": code,
                                    "名称": str(raw.get("f14", "")).strip(),
                                    "所处行业": industry,
                                    "最新价": _to_float(raw.get("f2")),
                                    "涨跌幅": _to_float(raw.get("f3")),
                                    "成交额": _to_float(raw.get("f6")),
                                    "市盈率-动态": _to_float(raw.get("f9")),
                                    "市净率": _to_float(raw.get("f23")),
                                    "股息率": None,
                                    "总市值": _to_float(raw.get("f20")),
                                    "流通市值": _to_float(raw.get("f21")),
                                    "换手率": _to_float(raw.get("f8")),
                                    "量比": _to_float(raw.get("f10")),
                                }
                            )
                        if page_new == 0:
                            break
                        if total_hint and len(rows) >= total_hint:
                            break

                    out = pd.DataFrame(rows)
                    if out.empty:
                        raise RuntimeError("港股直连结果为空")
                    _MARKET_NETWORK_MODE_CACHE["mode"] = req_mode
                    _MARKET_NETWORK_MODE_CACHE["ts"] = time.time()
                    return out
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.8)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("港股直连失败")


def _build_base_universe(market_scope: str = "A", network_mode: str = "auto") -> pd.DataFrame:
    scope = _safe_str(market_scope).upper() or "A"
    use_a = scope in {"A", "AH", "ALL"}
    use_hk = scope in {"HK", "AH", "ALL"}
    frames: List[pd.DataFrame] = []
    errors: List[str] = []

    if use_a:
        try:
            # 优先走直连端点，速度更稳定，也避免 AKShare 分页进度阻塞。
            spot_a = _fetch_a_spot_em_direct(network_mode=network_mode)
            if spot_a is None or spot_a.empty:
                errors.append("A股快照为空")
            else:
                frames.append(_build_universe_from_spot(spot_a, market="A"))
        except Exception as exc:
            errors.append(f"A股拉取失败: {exc}")

    if use_hk:
        try:
            spot_hk = _fetch_hk_spot_ak(network_mode=network_mode)
            if spot_hk is None or spot_hk.empty:
                errors.append("港股快照为空")
            else:
                frames.append(_build_universe_from_spot(spot_hk, market="HK"))
        except Exception as exc:
            errors.append(f"港股拉取失败: {exc}")

    if not frames:
        err = "；".join(errors) if errors else "未获取到市场快照"
        raise RuntimeError(err)

    out = pd.concat(frames, ignore_index=True)
    return out


def _build_base_universe_legacy() -> pd.DataFrame:
    try:
        spot_df = _ak_call_with_proxy_fallback(ak.stock_zh_a_spot_em)
    except Exception:
        spot_df = _fetch_a_spot_em_direct()
    if spot_df is None or spot_df.empty:
        raise RuntimeError("未获取到全市场快照，请稍后重试")

    code = _pick_series(spot_df, ["代码"]).astype(str).str.strip().str.zfill(6)
    name = _pick_series(spot_df, ["名称"]).astype(str).str.strip()
    industry = _pick_series(spot_df, ["所处行业", "所属行业", "行业"]).astype(str).str.strip()

    df = pd.DataFrame(
        {
            "code": code,
            "name": name,
            "industry": industry,
            "is_st": name.str.contains("ST", na=False).astype(int),
            "close_price": _pick_series(spot_df, ["最新价", "最新", "收盘"]).map(_to_float),
            "price_change_pct": _pick_series(spot_df, ["涨跌幅"]).map(_to_float),
            "amount": _pick_series(spot_df, ["成交额"]).map(_to_float),
            "pe_dynamic": _pick_series(spot_df, ["市盈率-动态", "市盈率动态", "市盈率"]).map(_to_float),
            "pb": _pick_series(spot_df, ["市净率"]).map(_to_float),
            "dividend_yield": _pick_series(spot_df, ["股息率", "股息率(%)"]).map(_to_float),
            "total_mv": _pick_series(spot_df, ["总市值"]).map(_to_mv_100m),
            "float_mv": _pick_series(spot_df, ["流通市值"]).map(_to_mv_100m),
            "turnover_ratio": _pick_series(spot_df, ["换手率"]).map(_to_float),
            "volume_ratio": _pick_series(spot_df, ["量比"]).map(_to_float),
        }
    )
    df["pe_ttm"] = df["pe_dynamic"]
    df["pe_static"] = None
    df["roe"] = None
    df["gross_margin"] = None
    df["net_margin"] = None
    df["asset_liability_ratio"] = None
    df["current_ratio"] = None
    df["operating_cashflow_3y"] = None
    df["receivable_revenue_ratio"] = None
    df["goodwill_equity_ratio"] = None
    df["interest_debt_asset_ratio"] = None
    df["ev_ebitda"] = None
    df["revenue_growth"] = None
    df["profit_growth"] = None
    df["revenue_cagr_5y"] = None
    df["profit_cagr_5y"] = None
    df["roe_avg_5y"] = None
    df["debt_ratio_avg_5y"] = None
    df["gross_margin_avg_5y"] = None
    df["debt_ratio_change_5y"] = None
    df["gross_margin_change_5y"] = None
    df["ocf_positive_years_5y"] = None
    df["total_score"] = None
    df["coverage_ratio"] = None
    df["conclusion"] = "观察"

    df["investigation_flag"] = 0
    df["penalty_flag"] = 0
    df["fund_occupation_flag"] = 0
    df["illegal_reduce_flag"] = 0
    df["pledge_ratio"] = None
    df["no_dividend_5y_flag"] = 0
    df["audit_change_count"] = 0
    df["audit_opinion"] = "标准无保留意见"

    df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["source_note"] = "ak.stock_zh_a_spot_em + fundamental_enrich"

    return df


def _fetch_a_spot_em_direct(network_mode: str = "auto") -> pd.DataFrame:
    """
    东方财富直连兜底（禁用系统代理），避免 ProxyError 导致整次更新失败。
    仅提供筛选所需核心字段。
    """
    hosts = [
        "https://82.push2.eastmoney.com/api/qt/clist/get",
        "https://push2.eastmoney.com/api/qt/clist/get",
        "https://71.push2.eastmoney.com/api/qt/clist/get",
        "http://82.push2.eastmoney.com/api/qt/clist/get",
        "http://push2.eastmoney.com/api/qt/clist/get",
        "http://71.push2.eastmoney.com/api/qt/clist/get",
    ]
    base_params = {
        "pn": "1",
        "pz": "200",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": ",".join(
            [
                "f2",   # 最新价
                "f3",   # 涨跌幅
                "f5",   # 成交量
                "f6",   # 成交额
                "f8",   # 换手率
                "f9",   # 市盈率-动态
                "f10",  # 量比
                "f12",  # 代码
                "f14",  # 名称
                "f20",  # 总市值
                "f21",  # 流通市值
                "f23",  # 市净率
                "f100", # 行业（可能为空）
            ]
        ),
    }

    mode = _safe_str(network_mode).lower() or "auto"
    if mode == "auto":
        mode = _resolve_market_network_mode(force=False)
    if mode in {"direct", "proxy", "surge_local"}:
        mode_order = [mode]
    else:
        mode_order = (
            ["surge_local", "proxy", "direct"]
            if (_force_surge_local_proxy() or _is_local_proxy_reachable())
            else ["direct", "proxy"]
        )
    proxy_url = _local_proxy_url()

    last_exc: Optional[Exception] = None
    for req_mode in mode_order:
        for url in hosts:
            for _ in range(2):
                try:
                    rows: List[Dict[str, Any]] = []
                    seen_codes: set[str] = set()
                    total_hint: Optional[int] = None
                    ses = requests.Session()
                    if req_mode == "surge_local":
                        ses.trust_env = False
                        ses.proxies.update({"http": proxy_url, "https": proxy_url})
                    else:
                        ses.trust_env = bool(req_mode == "proxy")

                    for page in range(1, 120):
                        params = dict(base_params)
                        params["pn"] = str(page)
                        try:
                            resp = ses.get(
                                url,
                                params=params,
                                timeout=15,
                                headers={
                                    "User-Agent": "Mozilla/5.0",
                                    "Connection": "close",
                                    "Accept": "application/json,text/plain,*/*",
                                },
                            )
                            resp.raise_for_status()
                            obj = resp.json()
                            data_obj = (obj or {}).get("data") or {}
                            if total_hint is None:
                                try:
                                    total_hint = int(_to_float(data_obj.get("total")) or 0)
                                except Exception:
                                    total_hint = None
                            diff = data_obj.get("diff") or []
                        except Exception as page_exc:
                            last_exc = page_exc
                            # 分页中断时保留已抓到的数据，避免整次更新回退
                            break

                        if not diff:
                            break

                        page_new = 0
                        for it in diff:
                            raw = it or {}
                            code = str(raw.get("f12", "")).strip()
                            if (not code) or (code in seen_codes):
                                continue
                            seen_codes.add(code)
                            page_new += 1
                            rows.append(
                                {
                                    "代码": code,
                                    "名称": str(raw.get("f14", "")).strip(),
                                    "所处行业": str(raw.get("f100", "")).strip(),
                                    "最新价": _to_float(raw.get("f2")),
                                    "涨跌幅": _to_float(raw.get("f3")),
                                    "成交额": _to_float(raw.get("f6")),
                                    "市盈率-动态": _to_float(raw.get("f9")),
                                    "市净率": _to_float(raw.get("f23")),
                                    "股息率": None,
                                    "总市值": _to_float(raw.get("f20")),
                                    "流通市值": _to_float(raw.get("f21")),
                                    "换手率": _to_float(raw.get("f8")),
                                    "量比": _to_float(raw.get("f10")),
                                }
                            )
                        if page_new == 0:
                            break
                        if total_hint and len(rows) >= total_hint:
                            break
                    if not rows:
                        raise RuntimeError("东方财富直连返回空数据")
                    _MARKET_NETWORK_MODE_CACHE["mode"] = req_mode
                    _MARKET_NETWORK_MODE_CACHE["ts"] = time.time()
                    return pd.DataFrame(rows)
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.8)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("东方财富直连失败")


def _normalize_dt_text(v: Any) -> str:
    text = _safe_str(v)
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return text


def _log_snapshot_run(
    row_count: int,
    enriched_count: int,
    enrich_start: int,
    enrich_end: int,
    fallback: bool,
    error_brief: str,
    cache_hit: int,
    cache_miss: int,
    enrich_mode: str,
) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO snapshot_runs(
                run_at, row_count, enriched_count, enrich_start, enrich_end,
                fallback, error_brief, cache_hit, cache_miss, enrich_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                int(row_count),
                int(enriched_count),
                int(enrich_start),
                int(enrich_end),
                1 if fallback else 0,
                _safe_str(error_brief)[:360],
                int(cache_hit),
                int(cache_miss),
                _safe_str(enrich_mode) or "top",
            ),
        )
        conn.commit()


def _classify_error_type(text: Any) -> str:
    t = _safe_str(text).lower()
    if not t:
        return "none"
    if (
        "name resolution" in t
        or "failed to resolve" in t
        or "nodename nor servname" in t
        or "not known" in t
    ):
        return "dns"
    if "proxy" in t:
        return "proxy"
    if "timeout" in t:
        return "timeout"
    if "ssl" in t:
        return "ssl"
    if "connection" in t or "connect" in t:
        return "connection"
    if "rate" in t or "429" in t or "too many requests" in t:
        return "rate_limit"
    return "other"


def check_market_data_dns(hosts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    检查行情源 DNS 是否可解析。只要有一个东财主机可解析即视为可执行更新。
    """
    target_hosts = hosts or [
        "push2.eastmoney.com",
        "82.push2.eastmoney.com",
        "81.push2.eastmoney.com",
        "72.push2.eastmoney.com",
        "71.push2.eastmoney.com",
    ]
    detail: Dict[str, Dict[str, Any]] = {}
    ok_hosts: List[str] = []
    fail_hosts: List[str] = []

    for host in target_hosts:
        h = _safe_str(host)
        if not h:
            continue
        try:
            infos = socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)
            ips = sorted({str(x[4][0]) for x in infos if isinstance(x, tuple) and len(x) >= 5 and x[4]})
            detail[h] = {"ok": True, "ips": ips[:3]}
            ok_hosts.append(h)
        except Exception as exc:
            detail[h] = {"ok": False, "error": _safe_str(exc)}
            fail_hosts.append(h)

    network_mode = _resolve_market_network_mode(force=False)
    transport_ok = network_mode in {"direct", "proxy", "surge_local"}
    return {
        "ok": len(ok_hosts) > 0,
        "ok_hosts": ok_hosts,
        "fail_hosts": fail_hosts,
        "detail": detail,
        "network_mode": network_mode,
        "transport_ok": transport_ok,
    }


def get_baseline_build_progress(market_scope: str = "AH") -> Dict[str, Any]:
    """
    返回首轮建库进度。
    说明：当前深补仅针对 A 股，因此 A+H 口径下统计的是 A 股深补进度。
    """
    scope = _safe_str(market_scope).upper() or "AH"
    df = load_snapshot()
    if df is None or df.empty:
        return {
            "scope": scope,
            "effective_scope": "A" if scope in {"AH", "ALL"} else scope,
            "snapshot_total": 0,
            "eligible_total": 0,
            "enriched_done": 0,
            "remaining": 0,
            "progress_ratio": 0.0,
            "progress_pct": 0.0,
        }

    if "market" not in df.columns:
        df["market"] = "A"
    markets = df["market"].astype(str).str.upper()
    if scope in {"AH", "ALL", "A"}:
        effective_scope = "A"
        eligible_mask = (markets == "A")
    elif scope == "HK":
        effective_scope = "HK"
        eligible_mask = (markets == "HK")
    else:
        effective_scope = "A"
        eligible_mask = (markets == "A")

    eligible_df = df[eligible_mask].copy()
    eligible_total = int(len(eligible_df))
    if "data_quality" in eligible_df.columns:
        dq = eligible_df["data_quality"].astype(str).str.lower().str.strip()
        enriched_done = int(((dq == "full") | (dq == "partial")).sum())
    elif "enriched_at" in eligible_df.columns:
        enriched_series = eligible_df["enriched_at"].astype(str).str.strip()
        enriched_done = int((enriched_series != "").sum())
    else:
        enriched_done = 0
    remaining = max(0, eligible_total - enriched_done)
    ratio = (enriched_done / eligible_total) if eligible_total > 0 else 0.0
    return {
        "scope": scope,
        "effective_scope": effective_scope,
        "snapshot_total": int(len(df)),
        "eligible_total": eligible_total,
        "enriched_done": enriched_done,
        "remaining": remaining,
        "progress_ratio": ratio,
        "progress_pct": ratio * 100.0,
    }


def run_baseline_build_once(
    market_scope: str = "AH",
    enrich_batch: int = 600,
    safe_mode: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    首轮建库推进一次：前置 DNS 检查，通过后按轮转补充一批。
    """
    scope = _safe_str(market_scope).upper() or "AH"
    dns = check_market_data_dns()
    before = get_baseline_build_progress(scope)

    stats = refresh_market_snapshot(
        max_stocks=0,
        enrich_top_n=max(0, int(enrich_batch)),
        force_refresh=bool(force_refresh),
        rotate_enrich=True,
        market_scope=scope,
        weekly_mode=False,
        safe_mode=bool(safe_mode),
    )
    after = get_baseline_build_progress(scope)
    added = max(0, int(after.get("enriched_done", 0)) - int(before.get("enriched_done", 0)))
    completed = bool(after.get("eligible_total", 0) > 0 and after.get("remaining", 0) == 0)

    _snapshot_meta_set("baseline_build_scope", scope)
    _snapshot_meta_set("baseline_build_target", str(int(after.get("eligible_total", 0) or 0)))
    _snapshot_meta_set("baseline_build_done", str(int(after.get("enriched_done", 0) or 0)))
    _snapshot_meta_set("baseline_build_completed", "1" if completed else "0")

    out = dict(stats)
    out.update(
        {
            "dns": dns,
            "network_mode": _safe_str(out.get("network_mode")) or _safe_str(_MARKET_NETWORK_MODE_CACHE.get("mode")) or "auto",
            "progress_before": before,
            "progress_after": after,
            "added": added,
            "completed": completed,
        }
    )
    return out


def refresh_market_snapshot(
    max_stocks: int = 0,
    enrich_top_n: int = 300,
    force_refresh: bool = False,
    rotate_enrich: bool = True,
    market_scope: str = "A",
    weekly_mode: bool = False,
    safe_mode: bool = True,
) -> Dict[str, Any]:
    init_db()
    scope = _safe_str(market_scope).upper() or "A"
    dns = check_market_data_dns()
    network_mode = _resolve_market_network_mode(force=True)
    attempt_mode = network_mode if network_mode in {"direct", "proxy", "surge_local"} else "auto"
    meta0 = get_snapshot_meta()
    if weekly_mode:
        last_weekly = _safe_str(meta0.get(f"last_weekly_update_{scope}", ""))
        if last_weekly:
            try:
                last_dt = datetime.fromisoformat(last_weekly)
            except Exception:
                try:
                    last_dt = datetime.strptime(last_weekly, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    last_dt = None
            if last_dt is not None:
                delta_days = (datetime.now() - last_dt).total_seconds() / 86400.0
                if delta_days < 7:
                    return {
                        "skipped": True,
                        "reason": f"周更间隔未满7天（已过 {delta_days:.1f} 天）",
                        "row_count": int(_to_float(meta0.get("row_count")) or 0),
                        "enriched_count": 0,
                        "market_scope": scope,
                    }
    enrich_mode = "rotate" if rotate_enrich else "top"
    base_fallback = False
    base_error = ""
    try:
        df = _build_base_universe(market_scope=scope, network_mode=attempt_mode).copy()
    except Exception as exc:
        # 网络/代理异常时兜底：优先复用本地快照继续深补推进
        existed = load_snapshot()
        network_mode_used = _safe_str(_MARKET_NETWORK_MODE_CACHE.get("mode")) or attempt_mode
        if existed is not None and not existed.empty:
            df = existed.copy()
            base_fallback = True
            base_error = str(exc)
        else:
            raise RuntimeError(
                f"未能拉取市场快照（可能是代理/VPN导致连接被拒绝）：{exc}"
            ) from exc
    if "market" not in df.columns:
        df["market"] = "A"
    df["market"] = df["market"].astype(str).str.upper()
    df = df[
        ((df["market"] == "A") & df["code"].astype(str).str.fullmatch(r"\d{6}", na=False))
        | ((df["market"] == "HK") & df["code"].astype(str).str.fullmatch(r"\d{5}", na=False))
    ].copy()
    # 强制去重，避免同一代码在回退快照路径被重复累积
    df = (
        df.drop_duplicates(subset=["market", "code"], keep="first")
        .sort_values(by=["total_mv", "code"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )

    if max_stocks and max_stocks > 0:
        df = df.head(int(max_stocks)).copy()

    manual_flags = _load_manual_flags()

    enrich_n = max(0, min(int(enrich_top_n), len(df)))
    total_count = len(df)
    start_idx = 0
    target_indices: List[int] = []
    cache_hit = 0
    cache_miss = 0

    # 基本面深补目前仅对A股执行，港股先走行情快照，避免接口无效调用与过度请求
    eligible_indices = [int(i) for i, m in enumerate(df["market"].tolist()) if _safe_str(m).upper() == "A"]
    eligible_total = len(eligible_indices)
    enrich_n = max(0, min(enrich_n, eligible_total))
    cursor_key = f"enrich_cursor_index_{scope}"

    if enrich_n > 0 and eligible_total > 0:
        if rotate_enrich:
            meta = get_snapshot_meta()
            try:
                start_idx = int(meta.get(cursor_key, meta.get("enrich_cursor_index", "0")))
            except Exception:
                start_idx = 0
            start_idx = start_idx % eligible_total
            target_indices = [eligible_indices[int((start_idx + step) % eligible_total)] for step in range(enrich_n)]
        else:
            target_indices = eligible_indices[:enrich_n]

    df["enriched_at"] = None
    consecutive_fail = 0
    for idx in target_indices:
        code = str(df.at[idx, "code"])
        name = str(df.at[idx, "name"])
        try:
            ext, source, cached_at = _enrich_one(code, name, force_refresh=force_refresh)
            for k, v in ext.items():
                if k in df.columns and v is not None:
                    df.at[idx, k] = v
            df.at[idx, "enriched_at"] = _normalize_dt_text(cached_at) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if source == "cache":
                cache_hit += 1
            else:
                cache_miss += 1
            consecutive_fail = 0
        except Exception:
            consecutive_fail += 1

        # 防封节流：随机抖动 + 连续失败熔断
        if safe_mode:
            time.sleep(random.uniform(0.25, 0.65))
            if consecutive_fail >= 15:
                break
        if idx > 0 and idx % 40 == 0:
            time.sleep(0.2 if safe_mode else 0.05)

    for i in range(len(df)):
        code = str(df.at[i, "code"])
        flags = manual_flags.get(code, {}) if isinstance(manual_flags, dict) else {}
        df.at[i, "investigation_flag"] = int(bool(flags.get("investigation", False)))
        df.at[i, "penalty_flag"] = int(bool(flags.get("penalty", False)))
        df.at[i, "fund_occupation_flag"] = int(bool(flags.get("fund_occupation", False)))
        df.at[i, "illegal_reduce_flag"] = int(bool(flags.get("illegal_reduce", False)))
        df.at[i, "pledge_ratio"] = _to_float(flags.get("pledge_ratio"))
        df.at[i, "no_dividend_5y_flag"] = int(bool(flags.get("no_dividend_5y", False)))
        df.at[i, "audit_change_count"] = int(_to_float(flags.get("audit_change_count")) or 0)
        if _safe_str(flags.get("audit_opinion")):
            df.at[i, "audit_opinion"] = _safe_str(flags.get("audit_opinion"))

    df["sunset_industry_flag"] = df.apply(
        lambda r: int(_detect_sunset(str(r.get("industry", "")), str(r.get("name", "")))),
        axis=1,
    )

    # 数据质量分级
    key_cols = [
        "pe_ttm",
        "pb",
        "dividend_yield",
        "roe",
        "gross_margin",
        "net_margin",
        "asset_liability_ratio",
        "operating_cashflow_3y",
    ]

    def _quality(row: pd.Series) -> str:
        cnt = sum(1 for c in key_cols if _to_float(row.get(c)) is not None)
        if cnt >= 6:
            return "full"
        if cnt >= 3:
            return "partial"
        return "missing"

    df["data_quality"] = df.apply(_quality, axis=1)

    cols = [
        "market",
        "code",
        "name",
        "industry",
        "is_st",
        "close_price",
        "price_change_pct",
        "amount",
        "pe_dynamic",
        "pe_static",
        "pe_ttm",
        "pb",
        "dividend_yield",
        "total_mv",
        "float_mv",
        "turnover_ratio",
        "volume_ratio",
        "roe",
        "gross_margin",
        "net_margin",
        "asset_liability_ratio",
        "current_ratio",
        "operating_cashflow_3y",
        "receivable_revenue_ratio",
        "goodwill_equity_ratio",
        "interest_debt_asset_ratio",
        "ev_ebitda",
        "revenue_growth",
        "profit_growth",
        "revenue_cagr_5y",
        "profit_cagr_5y",
        "roe_avg_5y",
        "debt_ratio_avg_5y",
        "gross_margin_avg_5y",
        "debt_ratio_change_5y",
        "gross_margin_change_5y",
        "ocf_positive_years_5y",
        "investigation_flag",
        "penalty_flag",
        "fund_occupation_flag",
        "illegal_reduce_flag",
        "pledge_ratio",
        "no_dividend_5y_flag",
        "audit_change_count",
        "audit_opinion",
        "sunset_industry_flag",
        "total_score",
        "conclusion",
        "coverage_ratio",
        "data_quality",
        "enriched_at",
        "updated_at",
        "source_note",
    ]
    save_df = df[cols].copy()

    with _connect() as conn:
        save_df.to_sql("market_snapshot", conn, if_exists="replace", index=False)
        conn.commit()

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _snapshot_meta_set("last_update", now_text)
    _snapshot_meta_set("row_count", str(len(save_df)))
    _snapshot_meta_set("enriched_count", str(enrich_n))
    _snapshot_meta_set("app_version", APP_VERSION)
    _snapshot_meta_set("last_refresh_fallback", "1" if base_fallback else "0")
    _snapshot_meta_set("last_refresh_error", base_error if base_fallback else "")
    _snapshot_meta_set("last_refresh_error_at", now_text if base_fallback else "")
    _snapshot_meta_set("last_scope", scope)
    if rotate_enrich and enrich_n > 0 and eligible_total > 0:
        _snapshot_meta_set(cursor_key, str((start_idx + enrich_n) % eligible_total))
        _snapshot_meta_set("enrich_cursor_index", str((start_idx + enrich_n) % eligible_total))
    if weekly_mode:
        _snapshot_meta_set(f"last_weekly_update_{scope}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    enrich_start = int(start_idx + 1) if enrich_n > 0 else 0
    enrich_end = int(((start_idx + enrich_n - 1) % eligible_total) + 1) if enrich_n > 0 and eligible_total > 0 else 0
    _log_snapshot_run(
        row_count=len(save_df),
        enriched_count=enrich_n,
        enrich_start=enrich_start,
        enrich_end=enrich_end,
        fallback=bool(base_fallback),
        error_brief=base_error if base_fallback else "",
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        enrich_mode=enrich_mode,
    )
    network_mode_used = _safe_str(_MARKET_NETWORK_MODE_CACHE.get("mode")) or attempt_mode

    return {
        "row_count": len(save_df),
        "enriched_count": enrich_n,
        "updated_at": now_text,
        "enrich_mode": enrich_mode,
        "enrich_start": enrich_start,
        "enrich_end": enrich_end,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "market_scope": scope,
        "weekly_mode": bool(weekly_mode),
        "network_mode": network_mode_used,
        "base_fallback": bool(base_fallback),
        "base_error": base_error if base_fallback else "",
    }


def load_snapshot() -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM market_snapshot ORDER BY total_mv DESC, code ASC", conn)
        except Exception:
            return pd.DataFrame()
    return df


def get_snapshot_health_report(days: int = 7, top_n: int = 20) -> Dict[str, Any]:
    init_db()
    meta = get_snapshot_meta()
    df = load_snapshot()

    total = int(len(df))
    quality_counts = {"full": 0, "partial": 0, "missing": 0}
    if total > 0 and "data_quality" in df.columns:
        vc = df["data_quality"].value_counts(dropna=False).to_dict()
        quality_counts = {
            "full": int(vc.get("full", 0)),
            "partial": int(vc.get("partial", 0)),
            "missing": int(vc.get("missing", 0)),
        }
    covered = int(quality_counts["full"] + quality_counts["partial"])
    coverage_ratio = (covered / total) if total > 0 else 0.0

    now = datetime.now()
    freshness = {
        "0_1d": 0,
        "1_3d": 0,
        "3_7d": 0,
        "7d_plus": 0,
        "never": 0,
    }

    enriched_series = df["enriched_at"] if ("enriched_at" in df.columns) else pd.Series([None] * total, index=df.index)
    parsed_dt: List[Optional[datetime]] = []
    for v in enriched_series:
        text = _safe_str(v)
        if not text:
            parsed_dt.append(None)
            freshness["never"] += 1
            continue
        dt_val: Optional[datetime] = None
        try:
            dt_val = datetime.fromisoformat(text)
        except Exception:
            try:
                dt_val = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_val = None
        parsed_dt.append(dt_val)
        if dt_val is None:
            freshness["never"] += 1
            continue
        delta_days = (now - dt_val).total_seconds() / 86400.0
        if delta_days < 1:
            freshness["0_1d"] += 1
        elif delta_days < 3:
            freshness["1_3d"] += 1
        elif delta_days < 7:
            freshness["3_7d"] += 1
        else:
            freshness["7d_plus"] += 1

    key_cols = [
        "pe_ttm",
        "pb",
        "dividend_yield",
        "roe",
        "gross_margin",
        "net_margin",
        "asset_liability_ratio",
        "operating_cashflow_3y",
    ]
    available_key_cols = [c for c in key_cols if c in df.columns]
    if available_key_cols:
        missing_count_series = df[available_key_cols].isna().sum(axis=1)
    else:
        missing_count_series = pd.Series([0] * total, index=df.index)

    enrich_df = pd.DataFrame(
        {
            "code": df.get("code", pd.Series([], dtype=object)),
            "name": df.get("name", pd.Series([], dtype=object)),
            "enriched_at": [_normalize_dt_text(v.isoformat() if isinstance(v, datetime) else (v if v is not None else "")) for v in parsed_dt],
            "missing_fields_count": missing_count_series,
        }
    )

    oldest_df = enrich_df.copy()
    oldest_df["sort_dt"] = parsed_dt
    oldest_df = oldest_df[oldest_df["sort_dt"].notna()].sort_values(by=["sort_dt", "missing_fields_count"], ascending=[True, False]).head(int(top_n))
    oldest_df = oldest_df.drop(columns=["sort_dt"]).reset_index(drop=True)

    newest_df = enrich_df.copy()
    newest_df["sort_dt"] = parsed_dt
    newest_df = newest_df[newest_df["sort_dt"].notna()].sort_values(by=["sort_dt", "missing_fields_count"], ascending=[False, True]).head(int(top_n))
    newest_df = newest_df.drop(columns=["sort_dt"]).reset_index(drop=True)

    missing_rank = []
    for col in available_key_cols:
        missing_rank.append({"field": col, "missing_count": int(df[col].isna().sum())})
    missing_rank = sorted(missing_rank, key=lambda x: x["missing_count"], reverse=True)

    with _connect() as conn:
        trend_df = pd.read_sql_query(
            """
            SELECT substr(run_at, 1, 10) AS run_date,
                   SUM(enriched_count) AS enriched_total,
                   SUM(CASE WHEN fallback=1 THEN 1 ELSE 0 END) AS fallback_count,
                   COUNT(*) AS run_count
            FROM snapshot_runs
            WHERE run_at >= datetime('now', ?)
            GROUP BY substr(run_at, 1, 10)
            ORDER BY run_date ASC
            """,
            conn,
            params=(f"-{int(days)} days",),
        )
        latest_run_df = pd.read_sql_query(
            "SELECT * FROM snapshot_runs ORDER BY run_id DESC LIMIT 1",
            conn,
        )
        fail_df = pd.read_sql_query(
            """
            SELECT run_at, fallback, error_brief, enrich_mode
            FROM snapshot_runs
            WHERE fallback=1 OR length(trim(coalesce(error_brief, ''))) > 0
            ORDER BY run_id DESC
            LIMIT 5
            """,
            conn,
        )
        runs_df = pd.read_sql_query(
            """
            SELECT run_at, row_count, enriched_count, enrich_start, enrich_end,
                   fallback, cache_hit, cache_miss, enrich_mode, error_brief
            FROM snapshot_runs
            ORDER BY run_id DESC
            LIMIT 50
            """,
            conn,
        )

    latest_run = latest_run_df.iloc[0].to_dict() if not latest_run_df.empty else {}
    cache_hit = int(latest_run.get("cache_hit", 0) or 0)
    cache_miss = int(latest_run.get("cache_miss", 0) or 0)

    fail_type_df = pd.DataFrame(columns=["error_type", "count"])
    if isinstance(runs_df, pd.DataFrame) and (not runs_df.empty):
        tmp = runs_df.copy()
        tmp["error_type"] = tmp["error_brief"].map(_classify_error_type)
        tmp = tmp[tmp["error_type"] != "none"]
        if not tmp.empty:
            fail_type_df = (
                tmp.groupby("error_type", as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values(by="count", ascending=False)
                .reset_index(drop=True)
            )

    last_scope = _safe_str(meta.get("last_scope", "A")) or "A"
    cursor_idx = int(_to_float(meta.get(f"enrich_cursor_index_{last_scope}", meta.get("enrich_cursor_index"))) or 0)
    cursor_pos = (cursor_idx + 1) if total > 0 else 0

    return {
        "meta": meta,
        "last_scope": last_scope,
        "total": total,
        "quality_counts": quality_counts,
        "covered": covered,
        "coverage_ratio": coverage_ratio,
        "freshness": freshness,
        "oldest_df": oldest_df,
        "newest_df": newest_df,
        "missing_rank": missing_rank,
        "trend_df": trend_df,
        "fail_df": fail_df,
        "fail_type_df": fail_type_df,
        "runs_df": runs_df,
        "latest_run": latest_run,
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "cursor_pos": cursor_pos,
    }


def export_snapshot_health_excel(days: int = 30, top_n: int = 50) -> bytes:
    report = get_snapshot_health_report(days=days, top_n=top_n)
    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    total = int(report.get("total", 0) or 0)
    qc = report.get("quality_counts", {}) if isinstance(report, dict) else {}
    fresh = report.get("freshness", {}) if isinstance(report, dict) else {}

    summary_df = pd.DataFrame(
        [
            {"item": "last_update", "value": _safe_str(meta.get("last_update", ""))},
            {"item": "last_refresh_fallback", "value": _safe_str(meta.get("last_refresh_fallback", ""))},
            {"item": "last_refresh_error_at", "value": _safe_str(meta.get("last_refresh_error_at", ""))},
            {"item": "enrich_cursor_index", "value": _safe_str(meta.get("enrich_cursor_index", ""))},
            {"item": "row_count", "value": str(total)},
            {"item": "quality_full", "value": str(int(qc.get("full", 0) or 0))},
            {"item": "quality_partial", "value": str(int(qc.get("partial", 0) or 0))},
            {"item": "quality_missing", "value": str(int(qc.get("missing", 0) or 0))},
            {"item": "fresh_0_1d", "value": str(int(fresh.get("0_1d", 0) or 0))},
            {"item": "fresh_1_3d", "value": str(int(fresh.get("1_3d", 0) or 0))},
            {"item": "fresh_3_7d", "value": str(int(fresh.get("3_7d", 0) or 0))},
            {"item": "fresh_7d_plus", "value": str(int(fresh.get("7d_plus", 0) or 0))},
            {"item": "fresh_never", "value": str(int(fresh.get("never", 0) or 0))},
        ]
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        pd.DataFrame(report.get("trend_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="trend_7d")
        pd.DataFrame(report.get("runs_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="runs_latest50")
        pd.DataFrame(report.get("fail_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="fails_latest5")
        pd.DataFrame(report.get("fail_type_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="fail_types")
        pd.DataFrame(report.get("missing_rank", [])).to_excel(writer, index=False, sheet_name="missing_rank")
        pd.DataFrame(report.get("oldest_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="oldest_enriched")
        pd.DataFrame(report.get("newest_df", pd.DataFrame())).to_excel(writer, index=False, sheet_name="newest_enriched")
    return output.getvalue()


def _num(v: Any) -> Optional[float]:
    return _to_float(v)


def _check_missing(enabled: bool, value: Optional[float], name: str, cfg: Dict[str, Any], reasons: List[str], missing: List[str]) -> bool:
    if not enabled:
        return False
    if value is not None:
        return False
    if str(cfg.get("missing_policy", "ignore")) == "exclude":
        reasons.append(f"{name}:缺失")
    else:
        missing.append(name)
    return True


def _check_min(enabled: bool, value: Optional[float], floor: float, name: str, cfg: Dict[str, Any], reasons: List[str], missing: List[str]) -> None:
    if not enabled:
        return
    if _check_missing(True, value, name, cfg, reasons, missing):
        return
    if value is not None and value < floor:
        reasons.append(f"{name}<{floor:g} (当前 {value:.2f})")


def _check_max(enabled: bool, value: Optional[float], ceil: float, name: str, cfg: Dict[str, Any], reasons: List[str], missing: List[str]) -> None:
    if not enabled:
        return
    if _check_missing(True, value, name, cfg, reasons, missing):
        return
    if value is not None and value > ceil:
        reasons.append(f"{name}>{ceil:g} (当前 {value:.2f})")


def _check_range(
    min_enabled: bool,
    max_enabled: bool,
    value: Optional[float],
    floor: float,
    ceil: float,
    name: str,
    cfg: Dict[str, Any],
    reasons: List[str],
    missing: List[str],
) -> None:
    if not min_enabled and not max_enabled:
        return
    if _check_missing(True, value, name, cfg, reasons, missing):
        return
    if value is None:
        return
    if min_enabled and value < floor:
        reasons.append(f"{name}<{floor:g} (当前 {value:.2f})")
    if max_enabled and value > ceil:
        reasons.append(f"{name}>{ceil:g} (当前 {value:.2f})")


def _split_keywords(text: str) -> List[str]:
    raw = str(text or "")
    parts = re.split(r"[,，;；\n]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _expand_industry_keywords(kws: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def _add(v: str) -> None:
        key = _safe_str(v).lower()
        if not key:
            return
        if key in seen:
            return
        seen.add(key)
        out.append(v)

    for one in kws:
        raw = _safe_str(one)
        _add(raw)
        for alias in INDUSTRY_KEYWORD_ALIAS_MAP.get(raw.lower(), []):
            _add(alias)
    return out


def apply_filters(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    if df is None or df.empty:
        empty = pd.DataFrame(columns=DISPLAY_COLUMNS)
        return empty, empty, empty, {"total": 0, "passed": 0, "rejected": 0, "missing": 0}

    cfg = copy.deepcopy(config or default_filter_config())
    r = cfg.get("risk", {})
    q = cfg.get("quality", {})
    v = cfg.get("valuation", {})
    g = cfg.get("growth_liquidity", {})
    d5 = cfg.get("rearview_5y", {})

    out_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        reasons: List[str] = []
        missing: List[str] = []

        scope = _safe_str(r.get("market_scope", "all")).upper()
        row_market = _safe_str(row.get("market")).upper() or "A"
        if scope in {"A", "HK"} and row_market != scope:
            reasons.append(f"市场不匹配({scope})")

        if bool(r.get("industry_include_enabled", False)):
            kws = _expand_industry_keywords(_split_keywords(r.get("industry_include_keywords", "")))
            if kws:
                search_text = f"{_safe_str(row.get('industry'))} {_safe_str(row.get('name'))}".lower()
                if not any(k.lower() in search_text for k in kws):
                    reasons.append("行业不匹配")

        is_st = int(_num(row.get("is_st")) or 0)
        if r.get("exclude_st", True) and is_st == 1:
            reasons.append("ST/*ST")

        if r.get("exclude_investigation", True) and int(_num(row.get("investigation_flag")) or 0) == 1:
            reasons.append("存在立案调查")
        if r.get("exclude_penalty", True) and int(_num(row.get("penalty_flag")) or 0) == 1:
            reasons.append("存在重大处罚")
        if r.get("exclude_fund_occupation", True) and int(_num(row.get("fund_occupation_flag")) or 0) == 1:
            reasons.append("存在资金占用")
        if r.get("exclude_illegal_reduce", True) and int(_num(row.get("illegal_reduce_flag")) or 0) == 1:
            reasons.append("存在违规减持")

        if r.get("require_standard_audit", False):
            audit = _safe_str(row.get("audit_opinion"))
            if not audit:
                _check_missing(True, None, "审计意见", cfg, reasons, missing)
            elif "标准无保留" not in audit:
                reasons.append(f"审计意见异常: {audit}")

        _check_max(
            bool(r.get("pledge_ratio_max_enabled", False)),
            _num(row.get("pledge_ratio")),
            float(r.get("pledge_ratio_max", 80.0)),
            "实控人质押率",
            cfg,
            reasons,
            missing,
        )

        _check_max(
            bool(r.get("audit_change_max_enabled", False)),
            _num(row.get("audit_change_count")),
            float(r.get("audit_change_max", 2)),
            "审计所更换次数(3年)",
            cfg,
            reasons,
            missing,
        )

        if r.get("exclude_no_dividend_5y", False) and int(_num(row.get("no_dividend_5y_flag")) or 0) == 1:
            reasons.append("近5年未分红")

        if r.get("exclude_sunset_industry", False):
            kws = _split_keywords(r.get("sunset_industries", "")) or DEFAULT_SUNSET_INDUSTRIES
            text = f"{_safe_str(row.get('industry'))} {_safe_str(row.get('name'))}"
            if any(k in text for k in kws):
                reasons.append("夕阳行业")

        _check_min(
            bool(q.get("ocf_3y_min_enabled", False)),
            _num(row.get("operating_cashflow_3y")),
            float(q.get("ocf_3y_min", 0.0)),
            "近3年经营现金流(亿)",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(q.get("asset_liability_max_enabled", False)),
            _num(row.get("asset_liability_ratio")),
            float(q.get("asset_liability_max", 80.0)),
            "资产负债率(%)",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(q.get("interest_debt_asset_max_enabled", False)),
            _num(row.get("interest_debt_asset_ratio")),
            float(q.get("interest_debt_asset_max", 20.0)),
            "有息负债/总资产(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(q.get("roe_min_enabled", False)),
            _num(row.get("roe")),
            float(q.get("roe_min", 5.0)),
            "ROE(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(q.get("gross_margin_min_enabled", False)),
            _num(row.get("gross_margin")),
            float(q.get("gross_margin_min", 20.0)),
            "毛利率(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(q.get("net_margin_min_enabled", False)),
            _num(row.get("net_margin")),
            float(q.get("net_margin_min", 8.0)),
            "净利率(%)",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(q.get("receivable_ratio_max_enabled", False)),
            _num(row.get("receivable_revenue_ratio")),
            float(q.get("receivable_ratio_max", 50.0)),
            "应收代理指标",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(q.get("goodwill_ratio_max_enabled", False)),
            _num(row.get("goodwill_equity_ratio")),
            float(q.get("goodwill_ratio_max", 30.0)),
            "商誉/净资产(%)",
            cfg,
            reasons,
            missing,
        )

        _check_range(
            bool(v.get("pe_ttm_min_enabled", False)),
            bool(v.get("pe_ttm_max_enabled", False)),
            _num(row.get("pe_ttm")),
            float(v.get("pe_ttm_min", 0.0)),
            float(v.get("pe_ttm_max", 25.0)),
            "PE(TTM)",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(v.get("pb_max_enabled", False)),
            _num(row.get("pb")),
            float(v.get("pb_max", 3.0)),
            "PB",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(v.get("ev_ebitda_max_enabled", False)),
            _num(row.get("ev_ebitda")),
            float(v.get("ev_ebitda_max", 18.0)),
            "EV/EBITDA",
            cfg,
            reasons,
            missing,
        )
        _check_range(
            bool(v.get("dividend_min_enabled", False)),
            bool(v.get("dividend_max_enabled", False)),
            _num(row.get("dividend_yield")),
            float(v.get("dividend_min", 3.0)),
            float(v.get("dividend_max", 12.0)),
            "股息率(%)",
            cfg,
            reasons,
            missing,
        )

        _check_min(
            bool(g.get("revenue_growth_min_enabled", False)),
            _num(row.get("revenue_growth")),
            float(g.get("revenue_growth_min", 0.0)),
            "营收增速(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(g.get("profit_growth_min_enabled", False)),
            _num(row.get("profit_growth")),
            float(g.get("profit_growth_min", 0.0)),
            "利润增速(%)",
            cfg,
            reasons,
            missing,
        )
        _check_range(
            bool(g.get("market_cap_min_enabled", False)),
            bool(g.get("market_cap_max_enabled", False)),
            _num(row.get("total_mv")),
            float(g.get("market_cap_min", 100.0)),
            float(g.get("market_cap_max", 5000.0)),
            "总市值(亿)",
            cfg,
            reasons,
            missing,
        )
        _check_range(
            bool(g.get("turnover_min_enabled", False)),
            bool(g.get("turnover_max_enabled", False)),
            _num(row.get("turnover_ratio")),
            float(g.get("turnover_min", 0.2)),
            float(g.get("turnover_max", 15.0)),
            "换手率(%)",
            cfg,
            reasons,
            missing,
        )
        _check_range(
            bool(g.get("volume_ratio_min_enabled", False)),
            bool(g.get("volume_ratio_max_enabled", False)),
            _num(row.get("volume_ratio")),
            float(g.get("volume_ratio_min", 0.5)),
            float(g.get("volume_ratio_max", 3.0)),
            "量比",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(g.get("amount_min_enabled", False)),
            _num(row.get("amount")),
            float(g.get("amount_min", 100000000.0)),
            "成交额(元)",
            cfg,
            reasons,
            missing,
        )

        _check_min(
            bool(d5.get("revenue_cagr_5y_min_enabled", False)),
            _num(row.get("revenue_cagr_5y")),
            float(d5.get("revenue_cagr_5y_min", 3.0)),
            "营收5年CAGR(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(d5.get("profit_cagr_5y_min_enabled", False)),
            _num(row.get("profit_cagr_5y")),
            float(d5.get("profit_cagr_5y_min", 3.0)),
            "净利5年CAGR(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(d5.get("roe_avg_5y_min_enabled", False)),
            _num(row.get("roe_avg_5y")),
            float(d5.get("roe_avg_5y_min", 8.0)),
            "ROE5年均值(%)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(d5.get("ocf_positive_years_5y_min_enabled", False)),
            _num(row.get("ocf_positive_years_5y")),
            float(d5.get("ocf_positive_years_5y_min", 4)),
            "经营现金流为正年数(5年)",
            cfg,
            reasons,
            missing,
        )
        _check_max(
            bool(d5.get("debt_ratio_change_5y_max_enabled", False)),
            _num(row.get("debt_ratio_change_5y")),
            float(d5.get("debt_ratio_change_5y_max", 8.0)),
            "负债率5年变化(百分点)",
            cfg,
            reasons,
            missing,
        )
        _check_min(
            bool(d5.get("gross_margin_change_5y_min_enabled", False)),
            _num(row.get("gross_margin_change_5y")),
            float(d5.get("gross_margin_change_5y_min", -6.0)),
            "毛利率5年变化(百分点)",
            cfg,
            reasons,
            missing,
        )

        out = {
            "market": _safe_str(row.get("market")) or "A",
            "code": _safe_str(row.get("code")),
            "name": _safe_str(row.get("name")),
            "industry": _safe_str(row.get("industry")),
            "pe_ttm": _num(row.get("pe_ttm")),
            "pb": _num(row.get("pb")),
            "dividend_yield": _num(row.get("dividend_yield")),
            "roe": _num(row.get("roe")),
            "asset_liability_ratio": _num(row.get("asset_liability_ratio")),
            "turnover_ratio": _num(row.get("turnover_ratio")),
            "volume_ratio": _num(row.get("volume_ratio")),
            "total_mv": _num(row.get("total_mv")),
            "revenue_cagr_5y": _num(row.get("revenue_cagr_5y")),
            "profit_cagr_5y": _num(row.get("profit_cagr_5y")),
            "roe_avg_5y": _num(row.get("roe_avg_5y")),
            "ocf_positive_years_5y": _num(row.get("ocf_positive_years_5y")),
            "debt_ratio_change_5y": _num(row.get("debt_ratio_change_5y")),
            "gross_margin_change_5y": _num(row.get("gross_margin_change_5y")),
            "data_quality": _safe_str(row.get("data_quality")) or "partial",
            "exclude_reasons": "；".join(reasons),
            "missing_fields": "、".join(missing),
            "passed": 1 if len(reasons) == 0 else 0,
        }
        out_rows.append(out)

    res_df = pd.DataFrame(out_rows)
    passed_df = res_df[res_df["passed"] == 1].copy()
    rejected_df = res_df[res_df["passed"] == 0].copy()
    missing_df = res_df[res_df["missing_fields"].astype(str).str.len() > 0].copy()

    for d in (passed_df, rejected_df, missing_df):
        for c in DISPLAY_COLUMNS:
            if c not in d.columns:
                d[c] = None

    passed_df = passed_df[DISPLAY_COLUMNS].sort_values(by=["total_mv", "code"], ascending=[False, True], na_position="last")
    rejected_df = rejected_df[DISPLAY_COLUMNS].sort_values(by=["code"])
    missing_df = missing_df[DISPLAY_COLUMNS].sort_values(by=["code"])

    stats = {
        "total": int(len(res_df)),
        "passed": int(len(passed_df)),
        "rejected": int(len(rejected_df)),
        "missing": int(len(missing_df)),
    }
    return passed_df, rejected_df, missing_df, stats


def export_results_excel(passed_df: pd.DataFrame, rejected_df: pd.DataFrame, missing_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        passed_df.to_excel(writer, index=False, sheet_name="通过池")
        rejected_df.to_excel(writer, index=False, sheet_name="排除池")
        missing_df.to_excel(writer, index=False, sheet_name="缺失项")
    return output.getvalue()


def build_ai_quick_config(prompt: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config or default_filter_config())
    text = _safe_str(prompt).lower()

    if any(k in text for k in ["高股息", "分红", "红利"]):
        cfg["valuation"]["dividend_min_enabled"] = True
        cfg["valuation"]["dividend_min"] = max(float(cfg["valuation"]["dividend_min"]), 4.0)

    if any(k in text for k in ["低估值", "便宜", "价值"]):
        cfg["valuation"]["pe_ttm_max_enabled"] = True
        cfg["valuation"]["pe_ttm_max"] = min(float(cfg["valuation"]["pe_ttm_max"]), 18.0)
        cfg["valuation"]["pb_max_enabled"] = True
        cfg["valuation"]["pb_max"] = min(float(cfg["valuation"]["pb_max"]), 2.2)

    if any(k in text for k in ["低负债", "稳健", "防御"]):
        cfg["quality"]["asset_liability_max_enabled"] = True
        cfg["quality"]["asset_liability_max"] = min(float(cfg["quality"]["asset_liability_max"]), 60.0)
        cfg["quality"]["roe_min_enabled"] = True
        cfg["quality"]["roe_min"] = max(float(cfg["quality"]["roe_min"]), 8.0)

    if any(k in text for k in ["现金流", "经营现金流"]):
        cfg["quality"]["ocf_3y_min_enabled"] = True
        cfg["quality"]["ocf_3y_min"] = max(float(cfg["quality"]["ocf_3y_min"]), 0.0)

    if any(k in text for k in ["大市值", "大盘", "龙头"]):
        cfg["growth_liquidity"]["market_cap_min_enabled"] = True
        cfg["growth_liquidity"]["market_cap_min"] = max(float(cfg["growth_liquidity"]["market_cap_min"]), 300.0)

    if any(k in text for k in ["五年", "长期", "后视镜", "复合增长"]):
        cfg["rearview_5y"]["revenue_cagr_5y_min_enabled"] = True
        cfg["rearview_5y"]["revenue_cagr_5y_min"] = max(float(cfg["rearview_5y"]["revenue_cagr_5y_min"]), 3.0)
        cfg["rearview_5y"]["profit_cagr_5y_min_enabled"] = True
        cfg["rearview_5y"]["profit_cagr_5y_min"] = max(float(cfg["rearview_5y"]["profit_cagr_5y_min"]), 3.0)
        cfg["rearview_5y"]["ocf_positive_years_5y_min_enabled"] = True
        cfg["rearview_5y"]["ocf_positive_years_5y_min"] = max(int(cfg["rearview_5y"]["ocf_positive_years_5y_min"]), 4)

    if any(k in text for k in ["排雷", "风险", "避雷"]):
        cfg["risk"]["exclude_st"] = True
        cfg["risk"]["exclude_investigation"] = True
        cfg["risk"]["exclude_penalty"] = True
        cfg["risk"]["exclude_fund_occupation"] = True
        cfg["risk"]["exclude_illegal_reduce"] = True

    return cfg
