import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import re
import sys
import time as pytime
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from streamlit.components.v1 import html
from urllib3.util.retry import Retry

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPENAI_AVAILABLE = True
OPENAI_IMPORT_ERROR = None
try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
except Exception as _exc:  # pragma: no cover - 依赖缺失时兜底
    OPENAI_AVAILABLE = False
    OPENAI_IMPORT_ERROR = _exc
    OpenAI = None  # type: ignore[assignment]

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    class RateLimitError(Exception):
        pass

from shared.multi_agent_analyzer import MultiAgentAnalyzer
from trading_data import (
    _cached_valuation_percentile,
    _infer_market_for_percentile,
    _load_json_file,
    _load_local_prefs,
    _save_json_file,
)

FUNDAMENTAL_DIR = CURRENT_DIR.parent / "fundamental"
if str(FUNDAMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(FUNDAMENTAL_DIR))

from fundamental_engine import analyze_fundamental as analyze_fundamental_single

try:
    from shared.valuation_engine import compute_full_valuation
except Exception:
    compute_full_valuation = None  # type: ignore[assignment]

FUND_DEEPSEEK_PROMPT = """你是专业基本面分析师。基于输入 JSON 做结构化输出：
1) 总结（不超过120字）
2) 八维点评（每维1句）
3) 关键风险（3条）
4) 跟踪清单（3条）
5) 结论：通过 / 观察 / 谨慎（给出理由）
要求：数据驱动、简洁、中文输出。"""

ANALYSIS_CACHE_PATH = "data/deepseek_analysis_cache.json"
ANALYSIS_JOB_DIR = "data/analysis_jobs"
ANALYSIS_DELTA_CACHE_PATH = "data/deepseek_delta_cache.json"
ANALYSIS_COOLDOWN_PATH = "data/deepseek_cooldown.json"
DEEP_COOLDOWN_MINUTES = 5
ANALYSIS_ENGINE_VERSION = "multi_agent_v1"

MULTI_AGENT_ANALYZER = MultiAgentAnalyzer()


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
    items = list(cache_obj.items())
    if len(items) > 120:
        items = items[-120:]
    with open(ANALYSIS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(items), f, ensure_ascii=False, indent=2)


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

def _extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

def _json_safe(v):
    if isinstance(v, dict):
        return {k: _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return _json_safe(v.to_dict(orient="records"))
    if isinstance(v, pd.Series):
        return _json_safe(v.to_dict())
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        try:
            return _json_safe(v.item())
        except Exception:
            pass
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v

def _build_analysis_payload(payload: dict) -> dict:
    """构造给 DeepSeek 的精简快照，尽量压缩 token。"""
    safe = _json_safe(payload)
    slow = safe.get("slow_engine", {}) or {}
    fe = safe.get("fast_engine", {}) or {}
    quote = fe.get("quote", {}) or {}
    compact = dict(fe.get("compact_metrics", {}) or {})
    compact.pop("cards_snapshot", None)

    out = {
        "meta": {
            "generated_at": (safe.get("meta", {}) or {}).get("generated_at"),
            "app": (safe.get("meta", {}) or {}).get("app"),
            "analysis_user": (safe.get("meta", {}) or {}).get("analysis_user"),
        },
        "stock": safe.get("stock", {}),
        "slow_engine": {
            "date": slow.get("date"),
            "pe_dynamic": slow.get("pe_dynamic"),
            "pe_static": slow.get("pe_static"),
            "pe_rolling": slow.get("pe_rolling"),
            "pb": slow.get("pb"),
            "dividend_yield": slow.get("dividend_yield"),
            "boll_index": slow.get("boll_index"),
        },
        "fast_engine": {},
    }

    out["fast_engine"]["quote"] = {
        "current_price": quote.get("current_price"),
        "change_pct": quote.get("change_pct"),
        "change_amount": quote.get("change_amount"),
        "open": quote.get("open"),
        "prev_close": quote.get("prev_close"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "volume": quote.get("volume"),
        "amount": quote.get("amount"),
        "turnover_rate": quote.get("turnover_rate"),
        "volume_ratio": quote.get("volume_ratio"),
        "vwap": quote.get("vwap"),
        "premium_pct": quote.get("premium_pct"),
        "quote_time": quote.get("quote_time"),
    }
    out["fast_engine"]["compact_metrics"] = compact
    out["fast_engine"]["order_book_5"] = fe.get("order_book_5")
    out["fast_engine"]["depth_note"] = fe.get("depth_note")
    out["fast_engine"]["error"] = fe.get("error")

    intraday = fe.get("intraday", [])
    if isinstance(intraday, list):
        # 仅保留最近24条，压缩 token
        out["fast_engine"]["intraday_recent"] = intraday[-24:]
        out["fast_engine"]["intraday_count"] = len(intraday)
    return out

def _build_quick_payload(payload: dict, stock_code: str) -> dict:
    safe = _json_safe(payload)
    fe = safe.get("fast_engine", {}) or {}
    compact = fe.get("compact_metrics", {}) or {}
    snap = {
        "price": (compact.get("snapshot", {}) or {}).get("current_price"),
        "change_pct": (compact.get("snapshot", {}) or {}).get("change_pct"),
        "high": (compact.get("snapshot", {}) or {}).get("high"),
        "low": (compact.get("snapshot", {}) or {}).get("low"),
        "volume": (compact.get("trading", {}) or {}).get("volume"),
        "amount": (compact.get("trading", {}) or {}).get("amount"),
        "volume_ratio": (compact.get("trading", {}) or {}).get("volume_ratio"),
        "turnover_rate": (compact.get("trading", {}) or {}).get("turnover_rate"),
        "vwap": (compact.get("trading", {}) or {}).get("vwap"),
        "premium_pct": (compact.get("trading", {}) or {}).get("premium_pct"),
        "amplitude_pct": (compact.get("trading", {}) or {}).get("amplitude_pct"),
        "imbalance_bid_ask": (compact.get("order_book_summary", {}) or {}).get("imbalance_bid_ask"),
        "spread": (compact.get("order_book_summary", {}) or {}).get("spread"),
        "order_diff": (compact.get("order_book_summary", {}) or {}).get("order_diff"),
        "pe_dynamic": (compact.get("valuation", {}) or {}).get("pe_dynamic"),
        "pe_rolling": (compact.get("valuation", {}) or {}).get("pe_rolling"),
        "pb": (compact.get("valuation", {}) or {}).get("pb"),
        "dividend_yield": (compact.get("valuation", {}) or {}).get("dividend_yield"),
        "rsi6": (compact.get("technical", {}) or {}).get("rsi6"),
        "macd_hist": (compact.get("technical", {}) or {}).get("macd_hist"),
    }
    snap = _json_safe(snap)

    delta_cache = _load_json_file(ANALYSIS_DELTA_CACHE_PATH)
    prev_snap = delta_cache.get(str(stock_code), {})
    delta = _dict_delta(snap, prev_snap) or {}
    delta_cache[str(stock_code)] = snap
    _save_json_file(ANALYSIS_DELTA_CACHE_PATH, delta_cache)

    return {
        "meta": safe.get("meta", {}),
        "stock": safe.get("stock", {}),
        "snapshot": snap,
        "delta": delta,
        "has_delta": bool(delta),
        "delta_keys": list(delta.keys()) if isinstance(delta, dict) else [],
    }

def _trigger_rules(analysis_payload: dict, quick_struct: dict) -> dict:
    fe = ((analysis_payload or {}).get("fast_engine", {}) or {}).get("compact_metrics", {}) or {}
    snapshot = fe.get("snapshot", {}) or {}
    trading = fe.get("trading", {}) or {}
    valuation = fe.get("valuation", {}) or {}
    tech = fe.get("technical", {}) or {}
    ob = fe.get("order_book_summary", {}) or {}

    risk = str((quick_struct or {}).get("risk_level", "")).lower()
    quick_need = bool((quick_struct or {}).get("need_full_analysis", False))
    premium = trading.get("premium_pct")
    imbalance = ob.get("imbalance_bid_ask")
    pe_dyn = valuation.get("pe_dynamic")
    dy = valuation.get("dividend_yield")
    rsi6 = tech.get("rsi6")
    macd = tech.get("macd_hist")
    chg = snapshot.get("change_pct")

    cond_risk = risk in {"medium", "high"}
    cond_conflict = (
        (pe_dyn is not None and pe_dyn <= 15 and ((rsi6 is not None and rsi6 < 35) or (macd is not None and macd < 0)))
        or (dy is not None and dy >= 4.0 and rsi6 is not None and rsi6 < 35)
    )
    cond_vwap = premium is not None and abs(float(premium)) >= 1.5
    cond_ob = imbalance is not None and (float(imbalance) >= 2.2 or float(imbalance) <= 0.45)
    cond_jump = chg is not None and abs(float(chg)) >= 3.0

    reasons = []
    if quick_need:
        reasons.append("快筛建议深析")
    if cond_risk:
        reasons.append(f"风险等级={risk or 'unknown'}")
    if cond_conflict:
        reasons.append("估值与动量冲突")
    if cond_vwap:
        reasons.append("现价偏离VWAP过大")
    if cond_ob:
        reasons.append("盘口失衡异常")
    if cond_jump:
        reasons.append("涨跌幅波动较大")

    should_deep = bool(reasons)
    return {"should_deep": should_deep, "reasons": reasons}

def _resolve_deepseek_api_key() -> str:
    raw = ""
    if st.session_state.get("deepseek_api_key_input"):
        raw = str(st.session_state["deepseek_api_key_input"])
    if not raw:
        prefs = _load_local_prefs()
        if prefs.get("deepseek_api_key"):
            raw = str(prefs.get("deepseek_api_key", ""))
    if not raw:
        try:
            secret_key = st.secrets.get("DEEPSEEK_API_KEY", "")
            if secret_key:
                raw = str(secret_key)
        except Exception:
            # 没有 secrets.toml 时，Streamlit 会抛出异常；这里静默回退到环境变量
            pass
    if not raw:
        raw = os.getenv("DEEPSEEK_API_KEY", "")

    key = raw.strip().split()[0] if raw.strip() else ""
    # 常见粘贴错误：中文引号/括号/注释混入
    key = key.strip("“”\"'`")
    return key

def _validate_api_key(key: str) -> None:
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。请在侧栏填写，或设置环境变量 DEEPSEEK_API_KEY。")
    if not key.startswith("sk-"):
        raise RuntimeError("API Key 格式异常：应以 sk- 开头。")
    if not re.fullmatch(r"sk-[A-Za-z0-9._-]+", key):
        raise RuntimeError("API Key 包含非法字符（可能混入中文符号/空格）。请重新粘贴纯 key。")

def _call_deepseek_with_prompt(
    user_content: str,
    system_prompt: str,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> tuple[str, dict, float, float]:
    if (not OPENAI_AVAILABLE) or (OpenAI is None):
        hint = "缺少 openai 依赖，请先安装：cd /Users/wellthen/Desktop/TEST/Quant_System/apps/trading && source venv/bin/activate && pip install -r requirements.txt"
        if OPENAI_IMPORT_ERROR is not None:
            raise RuntimeError(f"{hint}；原始错误: {OPENAI_IMPORT_ERROR}")
        raise RuntimeError(hint)

    api_key = _resolve_deepseek_api_key()
    _validate_api_key(api_key)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60.0, max_retries=0)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    t0 = pytime.time()
    last_conn_error = None
    response = None

    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            break
        except (APIConnectionError, APITimeoutError) as exc:
            last_conn_error = exc
            if attempt < 4:
                pytime.sleep(0.8 * attempt)
                continue
        except Exception:
            raise

    if response is None:
        # 兜底直连请求，规避 SDK 在个别网络环境下的连接异常
        url = "https://api.deepseek.com/v1/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        try:
            session = requests.Session()
            retry = Retry(
                total=4,
                connect=4,
                read=4,
                backoff_factor=0.8,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["POST"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            r = session.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                    "Accept-Encoding": "identity",
                },
                json=payload,
                timeout=(20, 90),
            )
            r.raise_for_status()
            raw = r.json()
        except requests.exceptions.RequestException as req_exc:
            raise RuntimeError(f"网络重试与直连兜底均失败: {req_exc}; 上次SDK异常: {last_conn_error}") from req_exc

        report = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not report:
            raise RuntimeError("DeepSeek 未返回有效分析内容（直连兜底）")

        usage_raw = raw.get("usage") or {}
        cache_hit_tokens = int(usage_raw.get("prompt_cache_hit_tokens") or 0)
        cache_miss_tokens = int(usage_raw.get("prompt_cache_miss_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        elapsed = pytime.time() - t0
    else:
        elapsed = pytime.time() - t0

        usage = response.usage
        cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        report = (response.choices[0].message.content or "").strip()

    cost = (
        cache_hit_tokens / 1_000_000 * 0.028
        + cache_miss_tokens / 1_000_000 * 0.28
        + completion_tokens / 1_000_000 * 0.42
    )

    if not report:
        raise RuntimeError("DeepSeek 未返回有效分析内容")

    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (prompt_tokens + completion_tokens),
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
    }
    return report, usage_dict, cost, elapsed

def _run_async_blocking(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None or (not loop.is_running()):
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(lambda: asyncio.run(coro))
        return fut.result()

def _extract_current_price_from_analysis_json(json_text: str) -> Optional[float]:
    try:
        payload = json.loads(str(json_text or ""))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    candidates = [
        (((payload.get("fast_engine") or {}).get("quote") or {}).get("current_price")),
        (((payload.get("fast_engine") or {}).get("compact_metrics") or {}).get("snapshot") or {}).get("current_price"),
        ((payload.get("stock") or {}).get("current_price")),
    ]
    for value in candidates:
        num = _to_float(value)
        if num is not None:
            return float(num)
    return None

def _build_multi_agent_context(
    json_text: str,
    stock_code: str = "",
    stock_name: str = "",
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    code = str(stock_code or "").strip()
    name = str(stock_name or "").strip()
    if not code:
        return None, None, None

    try:
        fundamental_data = analyze_fundamental_single(code=code, name=name or code, force_refresh=False)
    except Exception:
        return None, None, None
    if not isinstance(fundamental_data, dict) or not fundamental_data:
        return None, None, None

    current_price = _to_float(fundamental_data.get("current_price"))
    if current_price is None:
        current_price = _extract_current_price_from_analysis_json(json_text)

    valuation_data: Optional[dict] = None
    if callable(compute_full_valuation) and current_price is not None:
        try:
            valuation_data = compute_full_valuation(fundamental_data, float(current_price))
        except Exception:
            valuation_data = None
    if (not isinstance(valuation_data, dict) or not valuation_data) and isinstance(
        fundamental_data.get("valuation_report"), dict
    ):
        valuation_data = copy.deepcopy(fundamental_data.get("valuation_report") or {})

    try:
        market = _infer_market_for_percentile(code=code, row=fundamental_data)
        percentile_report = _cached_valuation_percentile(market=market, code=code, lookback_years=5)
    except Exception:
        percentile_report = {}

    if isinstance(percentile_report, dict) and percentile_report:
        if not isinstance(valuation_data, dict):
            valuation_data = {}
        valuation_data = dict(valuation_data)
        valuation_data["valuation_percentile"] = percentile_report
        pe_pct = _to_float(((percentile_report.get("pe_ttm") or {}).get("percentile")))
        if pe_pct is not None:
            valuation_data["pe_percentile"] = float(pe_pct)

    news_context = {
        "news_catalysts": str(fundamental_data.get("news_catalysts") or "").strip(),
        "research_summary": str(fundamental_data.get("research_summary") or "").strip(),
    }
    telegraph_items = fundamental_data.get("telegraph_items")
    research_items = fundamental_data.get("research_items")
    if isinstance(telegraph_items, list) and telegraph_items:
        news_context["telegraph_items"] = telegraph_items
    if isinstance(research_items, list) and research_items:
        news_context["research_items"] = research_items
    has_news_context = any(
        [
            bool(news_context.get("news_catalysts")),
            bool(news_context.get("research_summary")),
            bool(news_context.get("telegraph_items")),
            bool(news_context.get("research_items")),
        ]
    )
    return (
        fundamental_data,
        valuation_data if isinstance(valuation_data, dict) and valuation_data else None,
        news_context if has_news_context else None,
    )

def _call_multi_agent_analysis(
    json_text: str,
    stock_code: str = "",
    stock_name: str = "",
) -> tuple[str, dict, float, float]:
    def _llm_call(
        user_content: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, dict, float, float]:
        return _call_deepseek_with_prompt(
            user_content=user_content,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    fundamental_data, valuation_data, news_context = _build_multi_agent_context(
        json_text=json_text,
        stock_code=stock_code,
        stock_name=stock_name,
    )
    result = _run_async_blocking(
        MULTI_AGENT_ANALYZER.analyze(
            user_content=json_text,
            llm_call=_llm_call,
            fundamental_data=fundamental_data,
            valuation_data=valuation_data,
            news_context=news_context,
            stock_code=stock_code,
            stock_name=stock_name,
        )
    )
    report = str(result.get("final_text", "") or "").strip()
    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
    if isinstance(usage, dict):
        usage["_multi_agent_meta"] = result.get("meta", {}) if isinstance(result.get("meta"), dict) else {}
    total_cost = float(result.get("total_cost", 0.0) or 0.0)
    elapsed = float(result.get("elapsed", 0.0) or 0.0)
    return report, usage, total_cost, elapsed

def _call_deepseek_analysis(
    json_text: str, stock_code: str = "", stock_name: str = ""
) -> tuple[str, dict, float, float]:
    return _call_multi_agent_analysis(json_text=json_text, stock_code=stock_code, stock_name=stock_name)

def _sanitize_deepseek_report(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s

    # 去掉模型常见抬头（标题型）
    s = re.sub(r"^\s*#{1,6}\s*DeepSeek[^\n]*\n+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*DeepSeek[^\n]*\n+", "", s, flags=re.IGNORECASE)

    # 去掉常见开场客套句
    s = re.sub(
        r"^\s*(好的|当然|明白了|收到)[，,。!\s]*作为[^\n。！？]*[。！？]?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # 若仍以“我将/我会...进行分析”开头，继续剥离一次
    s = re.sub(r"^\s*我将[^\n。！？]*[。！？]?\s*", "", s)
    s = re.sub(r"^\s*我会[^\n。！？]*[。！？]?\s*", "", s)

    return s.strip()

def _call_deepseek_fundamental(json_text: str) -> tuple[str, dict, float, float]:
    return _call_deepseek_with_prompt(
        user_content=json_text,
        system_prompt=FUND_DEEPSEEK_PROMPT,
        max_tokens=1200,
        temperature=0.3,
        top_p=0.9,
    )

def _clean_text_no_na(text: str) -> str:
    s = str(text or "")
    for bad in ["N/A", "n/a", "nan", "None", "--", "null"]:
        s = s.replace(bad, "")
    return re.sub(r"\s+", " ", s).strip(" ，。；、")

def _split_sentences(text: str):
    clean = _clean_text_no_na(text)
    if not clean:
        return []
    parts = re.split(r"[。！？；\n]+", clean)
    return [p.strip() for p in parts if p.strip()]

def _format_card_desc_lines(text: str, max_lines: int = 3) -> str:
    raw = _clean_text_no_na(text)
    if not raw:
        lines = []
    else:
        parts = re.split(r"[\/／|]+", raw)
        lines = []
        for part in parts:
            sub = _split_sentences(part)
            if sub:
                lines.extend(sub)
        lines = [x.strip() for x in lines if x.strip()]
    lines = lines[:max_lines]
    while len(lines) < max_lines:
        lines.append("")
    html_lines = []
    for one in lines:
        if one:
            html_lines.append(f"<span class='line'>{one}</span>")
        else:
            html_lines.append("<span class='line line-empty'>.</span>")
    return "".join(html_lines)

def _clean_text_keep_lines(text: str) -> str:
    s = str(text or "")
    for bad in ["N/A", "n/a", "nan", "None", "null"]:
        s = s.replace(bad, "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in s.split("\n")]
    lines = [x for x in lines if x]
    return "\n".join(lines).strip()

def _format_rag_text_block(text: str) -> str:
    raw = _clean_text_keep_lines(text)
    if not raw:
        return ""
    s = raw
    s = s.replace("】-", "】\n- ").replace("）-", "）\n- ")
    s = re.sub(r"\s+【", "\n【", s)
    s = re.sub(r"\s*-\s*(?=(\d{2}-\d{2}\s\d{2}:\d{2}|\d{4}-\d{2}-\d{2}|\[\d{4}-\d{2}-\d{2}\]))", "\n- ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _format_prompt_text_block(text: str) -> str:
    raw = _clean_text_keep_lines(text)
    if not raw:
        return ""
    s = raw
    for marker in ["【最新新闻催化】", "【机构研报摘要】", "输出要求：", "6) 催化剂研判（必须包含）：", "要求："]:
        s = s.replace(marker, f"\n{marker}")
    s = re.sub(r"\s*-\s*", "\n- ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _to_float(value):
    if value is None:
        return None
    try:
        num = float(value)
        if math.isnan(num):
            return None
        return num
    except Exception:
        return None

def _fmt_num(value, digits: int = 2, suffix: str = "") -> str:
    num = _to_float(value)
    if num is None:
        return "--"
    return f"{num:,.{digits}f}{suffix}"

def _fmt_pct_point(value, digits: int = 2) -> str:
    num = _to_float(value)
    if num is None:
        return "--"
    return f"{num:.{digits}f}%"

def _fmt_pct_ratio(value, digits: int = 2) -> str:
    num = _to_float(value)
    if num is None:
        return "--"
    return f"{num * 100:.{digits}f}%"

def _analysis_job_file(job_id: str) -> str:
    return os.path.join(ANALYSIS_JOB_DIR, f"{job_id}.json")

def _create_analysis_job(
    stock_code: str,
    stock_name: str,
    mode: str,
    quick_json: str,
    deep_json: str,
    quick_hash: str,
    deep_hash: str,
) -> str:
    os.makedirs(ANALYSIS_JOB_DIR, exist_ok=True)
    job_id = f"{stock_code}_{int(pytime.time() * 1000)}"
    job = {
        "job_id": job_id,
        "created_at": datetime.now().strftime("%m-%d %H:%M:%S"),
        "stock_code": str(stock_code),
        "stock_name": str(stock_name),
        "analysis_engine": "multi_agent_v1",
        "mode": str(mode),
        "status": "pending",
        "quick_json": quick_json,
        "deep_json": deep_json,
        "quick_hash": quick_hash,
        "deep_hash": deep_hash,
    }
    _save_json_file(_analysis_job_file(job_id), job)
    return job_id

def _upsert_live_analysis_job(
    stock_code: str,
    stock_name: str,
    quick_json: str,
    deep_json: str,
    quick_hash: str,
    deep_hash: str,
) -> str:
    os.makedirs(ANALYSIS_JOB_DIR, exist_ok=True)
    job_id = f"live_{stock_code}"
    path = _analysis_job_file(job_id)
    current = _load_json_file(path)
    now_text = datetime.now().strftime("%m-%d %H:%M:%S")

    if current and isinstance(current, dict):
        current["stock_name"] = str(stock_name)
        current["quick_json"] = quick_json
        current["deep_json"] = deep_json
        current["quick_hash"] = quick_hash
        current["deep_hash"] = deep_hash
        current["updated_at"] = now_text
        if str(current.get("analysis_engine", "")) != "multi_agent_v1":
            current["analysis_engine"] = "multi_agent_v1"
            current["mode"] = "idle"
            current["status"] = "pending"
            current.pop("final_text", None)
            current.pop("stats", None)
            current.pop("trigger_alert", None)
        if "created_at" not in current:
            current["created_at"] = now_text
        if "mode" not in current:
            current["mode"] = "idle"
        if "status" not in current:
            current["status"] = "pending"
        _save_json_file(path, current)
        return job_id

    job = {
        "job_id": job_id,
        "created_at": now_text,
        "updated_at": now_text,
        "stock_code": str(stock_code),
        "stock_name": str(stock_name),
        "analysis_engine": "multi_agent_v1",
        "mode": "idle",
        "status": "pending",
        "quick_json": quick_json,
        "deep_json": deep_json,
        "quick_hash": quick_hash,
        "deep_hash": deep_hash,
    }
    _save_json_file(path, job)
    return job_id

def _normalize_quick_result(quick_raw: str) -> dict:
    obj = _extract_json_object(quick_raw)
    risk = str(obj.get("risk_level", "medium")).lower()
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    conclusions = obj.get("conclusions", [])
    if not isinstance(conclusions, list):
        conclusions = []
    conclusions = [str(x) for x in conclusions][:3]
    if not conclusions:
        conclusions = ["快筛未返回标准结论，建议人工复核。"]

    need_full = obj.get("need_full_analysis", False)
    if isinstance(need_full, str):
        need_full = need_full.strip().lower() in {"1", "true", "yes", "y", "需要", "是"}
    else:
        need_full = bool(need_full)

    trigger_reasons = obj.get("trigger_reasons", [])
    if not isinstance(trigger_reasons, list):
        trigger_reasons = []
    trigger_reasons = [str(x) for x in trigger_reasons][:4]

    return {
        "risk_level": risk,
        "conclusions": conclusions,
        "need_full_analysis": need_full,
        "trigger_reasons": trigger_reasons,
        "raw": quick_raw,
    }

def _render_multi_agent_meta(stats: dict) -> None:
    meta = stats.get("multi_agent_meta", {}) if isinstance(stats, dict) else {}
    if not isinstance(meta, dict) or not meta:
        return
    bull_prob = int(meta.get("bull_prob", 0) or 0)
    bear_prob = int(meta.get("bear_prob", 0) or 0)
    disagreement = float(meta.get("disagreement_score", 0.0) or 0.0) * 100
    agreement = float(meta.get("agreement_score", 0.0) or 0.0) * 100
    consensus = str(meta.get("consensus_side", "--"))
    event_side = str(meta.get("event_side", "neutral"))
    event_cn = {"bull": "偏多", "bear": "偏空", "neutral": "中性/不明"}.get(event_side, event_side)

    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("做多胜率", f"{bull_prob}%")
    c2.metric("做空胜率", f"{bear_prob}%")
    c3.metric("专家分歧度", f"{disagreement:.1f}%")
    c4.metric("一致性", f"{agreement:.1f}%")
    st.caption(f"专家投票共识: {consensus} ｜ 事件面倾向: {event_cn}")

def _render_final_report_block(job_id: str, job_obj: dict, key_suffix: str, height: int = 560) -> None:
    final_text = str(job_obj.get("final_text", "") or "")
    done_mark = str(job_obj.get("done_at", "") or "")
    text_key = f"job_text_{job_id}_{key_suffix}_{hashlib.md5((final_text + done_mark).encode('utf-8')).hexdigest()[:10]}"
    stats = job_obj.get("stats", {}) or {}
    analyzed_at = str(job_obj.get("done_at") or job_obj.get("updated_at") or job_obj.get("created_at") or "--")
    st.markdown(
        f"<div class='analysis-time-badge'>分析时间: {analyzed_at}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"深析输入: {stats.get('deep_prompt_tokens', 0)} | 深析输出: {stats.get('deep_completion_tokens', 0)} | "
        f"预估总成本: {float(stats.get('total_cost', 0) or 0):.4f} 元"
    )
    _render_multi_agent_meta(stats)
    report_text = _sanitize_deepseek_report(final_text)
    if report_text:
        def _chunk_gen(text: str, step: int = 80):
            for i in range(0, len(text), max(20, int(step))):
                yield text[i : i + max(20, int(step))]

        should_stream = "_new_" in str(key_suffix) or str(key_suffix).startswith("new")
        stream_once_key = f"_md_stream_done_{job_id}_{hashlib.md5((report_text + key_suffix).encode('utf-8')).hexdigest()[:10]}"
        if should_stream and (not st.session_state.get(stream_once_key)):
            md_stream = getattr(st, "markdown_stream", None)
            if callable(md_stream):
                md_stream(_chunk_gen(report_text))
            else:
                slot = st.empty()
                acc = ""
                for piece in _chunk_gen(report_text):
                    acc += piece
                    slot.markdown(acc)
            st.session_state[stream_once_key] = True
        else:
            st.markdown(report_text)
    else:
        st.info("暂无分析文本。")
    st.text_area("分析文本（可复制）", value=report_text, height=min(height, 280), key=text_key)
    final_b64 = base64.b64encode(final_text.encode("utf-8")).decode("ascii")
    html(
        f"""
        <div style="margin-top:0.2rem;">
          <button id="copy-job-doc-{job_id}-{key_suffix}"
            style="height:38px;padding:0 0.9rem;border-radius:8px;border:1px solid #a8c2e8;background:#dbeafe;color:#0f2a52;font-size:0.95rem;font-weight:700;cursor:pointer;">
            复制分析文档
          </button>
          <span id="copy-job-msg-{job_id}-{key_suffix}" style="margin-left:0.55rem;color:rgba(239,229,216,0.82);font-size:0.86rem;"></span>
        </div>
        <script>
          const b = document.getElementById("copy-job-doc-{job_id}-{key_suffix}");
          const m = document.getElementById("copy-job-msg-{job_id}-{key_suffix}");
          const t = decodeURIComponent(escape(window.atob("{final_b64}")));
          b.onclick = async function () {{
            try {{
              await navigator.clipboard.writeText(t);
              m.textContent = "已复制";
            }} catch (e) {{
              m.textContent = "复制失败";
            }}
          }};
        </script>
        """,
        height=64,
    )

def _execute_analysis_job(job_id: str, mode: str, ui_prefix: str = "", force_refresh: bool = False) -> dict:
    job = _load_json_file(_analysis_job_file(job_id))
    if not job:
        raise RuntimeError("分析任务不存在或已失效。")

    stock_code = str(job.get("stock_code", ""))
    stock_name = str(job.get("stock_name", ""))
    mode = "deep"

    job["mode"] = mode
    job["status"] = "running"
    job["started_at"] = datetime.now().strftime("%m-%d %H:%M:%S")
    job.pop("error", None)
    _save_json_file(_analysis_job_file(job_id), job)

    progress = st.progress(5, text=f"{ui_prefix}准备任务...")
    cache_store = _load_analysis_cache()
    cooldown_store = _load_json_file(ANALYSIS_COOLDOWN_PATH)
    now_ts = datetime.now().timestamp()

    deep_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    total_cost = 0.0
    deep_report = ""
    deep_source = "未执行"

    deep_json = str(job.get("deep_json", "") or "")
    deep_hash = str(job.get("deep_hash", "") or "")

    try:
        if force_refresh:
            progress.progress(36, text=f"{ui_prefix}强制刷新分析...")
            deep_report, d_usage, d_cost, _ = _call_deepseek_analysis(
                json_text=deep_json,
                stock_code=stock_code,
                stock_name=stock_name,
            )
            deep_usage = d_usage
            total_cost += float(d_cost or 0.0)
            deep_source = "手动刷新"
            cache_store[deep_hash] = {
                "stage": "deep",
                "analysis_engine": ANALYSIS_ENGINE_VERSION,
                "result": deep_report,
                "usage": deep_usage,
                "saved_at": datetime.now().strftime("%m-%d %H:%M:%S"),
                "stock_code": stock_code,
                "stock_name": stock_name,
            }
            _save_analysis_cache(cache_store)
            cooldown_store[stock_code] = {
                "last_ts": now_ts,
                "last_time": datetime.now().strftime("%m-%d %H:%M:%S"),
                "last_deep_hash": deep_hash,
            }
            _save_json_file(ANALYSIS_COOLDOWN_PATH, cooldown_store)
        else:
            progress.progress(28, text=f"{ui_prefix}检查缓存...")
            d_cached = cache_store.get(deep_hash) if deep_hash else None
            if d_cached and isinstance(d_cached, dict) and str(d_cached.get("analysis_engine", "")) == ANALYSIS_ENGINE_VERSION:
                deep_report = str(d_cached.get("result", "") or "")
                deep_usage = d_cached.get("usage", {}) or deep_usage
                deep_source = "同快照复用"
            else:
                deep_cd = cooldown_store.get(stock_code, {})
                last_ts = float(deep_cd.get("last_ts", 0) or 0)
                in_cooldown = (now_ts - last_ts) < (DEEP_COOLDOWN_MINUTES * 60)
                cached_hash = str(deep_cd.get("last_deep_hash", ""))
                if in_cooldown and cached_hash:
                    cd_cached = cache_store.get(cached_hash)
                    if cd_cached and isinstance(cd_cached, dict):
                        deep_report = str(cd_cached.get("result", "") or "")
                        deep_usage = cd_cached.get("usage", {}) or deep_usage
                        deep_source = f"冷却期复用({DEEP_COOLDOWN_MINUTES}分钟)"

                if not deep_report:
                    progress.progress(75, text=f"{ui_prefix}执行深析...")
                    deep_report, d_usage, d_cost, _ = _call_deepseek_analysis(
                        json_text=deep_json,
                        stock_code=stock_code,
                        stock_name=stock_name,
                    )
                    deep_usage = d_usage
                    total_cost += float(d_cost or 0.0)
                    cache_store[deep_hash] = {
                        "stage": "deep",
                        "analysis_engine": ANALYSIS_ENGINE_VERSION,
                        "result": deep_report,
                        "usage": deep_usage,
                        "saved_at": datetime.now().strftime("%m-%d %H:%M:%S"),
                        "stock_code": stock_code,
                        "stock_name": stock_name,
                    }
                    _save_analysis_cache(cache_store)
                    deep_source = "实时调用"

                cooldown_store[stock_code] = {
                    "last_ts": now_ts,
                    "last_time": datetime.now().strftime("%m-%d %H:%M:%S"),
                    "last_deep_hash": deep_hash,
                }
                _save_json_file(ANALYSIS_COOLDOWN_PATH, cooldown_store)

        progress.progress(100, text=f"{ui_prefix}分析完成")
        final_text = _sanitize_deepseek_report(deep_report)

        stats = {
            "deep_prompt_tokens": int(deep_usage.get("prompt_tokens", 0) or 0),
            "deep_completion_tokens": int(deep_usage.get("completion_tokens", 0) or 0),
            "total_cost": float(total_cost),
            "deep_source": deep_source,
            "multi_agent_meta": deep_usage.get("_multi_agent_meta", {}) if isinstance(deep_usage, dict) else {},
        }

        job["status"] = "done"
        job["done_at"] = datetime.now().strftime("%m-%d %H:%M:%S")
        job["final_text"] = final_text
        job["stats"] = stats
        _save_json_file(_analysis_job_file(job_id), job)
        progress.empty()
        return job
    except Exception as exc:
        job["status"] = "failed"
        job["failed_at"] = datetime.now().strftime("%m-%d %H:%M:%S")
        job["error"] = f"{type(exc).__name__}: {exc}"
        _save_json_file(_analysis_job_file(job_id), job)
        progress.empty()
        raise

def _render_analysis_window(job_id: str, embedded: bool = False, auto_mode: str = "") -> None:
    job = _load_json_file(_analysis_job_file(job_id))
    if not job:
        st.error("分析任务不存在或已失效。")
        return

    stock_code = str(job.get("stock_code", ""))
    stock_name = str(job.get("stock_name", ""))
    mode = str(job.get("mode", "idle"))

    if not embedded:
        safe_title = f"{stock_name}({stock_code}) - Quant".replace("\\", "\\\\").replace("'", "\\'")
        html(
            f"""
            <script>
              try {{
                document.title = '{safe_title}';
                if (window.parent && window.parent.document) {{
                  window.parent.document.title = '{safe_title}';
                }}
              }} catch (e) {{}}
            </script>
            """,
            height=0,
        )
        st.title(f"DeepSeek 分析窗口 · {stock_name} ({stock_code})")
        st.caption(f"任务ID: {job_id} | 上次模式: {mode} | 创建时间: {job.get('created_at', '--')}")
    else:
        st.subheader(f"DeepSeek分析文档 · {stock_name} ({stock_code})")
        st.caption(f"任务ID: {job_id} | 上次模式: {mode}")

    run_mode = ""
    force_refresh = False
    if embedded:
        run_mode = "deep" if auto_mode else ""
    else:
        btn_cols = st.columns(2)
        if btn_cols[0].button("DeepSeek分析", key=f"analysis_window_deep_{job_id}", use_container_width=True):
            run_mode = "deep"
        if btn_cols[1].button("刷新", key=f"analysis_window_refresh_{job_id}", use_container_width=True):
            run_mode = "deep"
            force_refresh = True

    if not run_mode:
        if job.get("status") == "done":
            if not embedded:
                st.success("上次分析已完成。可点击上方按钮重新分析。")
            _render_final_report_block(job_id, job, "saved", height=420 if embedded else 560)
        elif job.get("status") == "failed":
            st.error(f"上次分析失败: {job.get('error', '未知错误')}")
            if not embedded:
                st.info("请点击上方按钮重新执行。")
        else:
            st.info("点击“DeepSeek分析”开始执行。")
        return

    try:
        done_job = _execute_analysis_job(
            job_id=job_id,
            mode="deep",
            ui_prefix=f"{stock_name} ",
            force_refresh=force_refresh,
        )
        if not embedded:
            st.success("深析已完成。")
        else:
            st.success("分析完成。")
        _render_final_report_block(job_id, done_job, "new", height=420 if embedded else 560)
    except Exception as exc:
        st.error(f"分析失败: {type(exc).__name__}: {exc}")
