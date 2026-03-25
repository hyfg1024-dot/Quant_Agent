import base64
import hashlib
import json
import math
import os
import re
import sys
import time as pytime
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import altair as alt
import pandas as pd
import requests
import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from streamlit.components.v1 import html

from fast_engine import fetch_fast_panel
from slow_engine import (
    add_stock_by_query,
    get_stock_pool,
    get_latest_fundamental_snapshot,
    get_stock_group_map,
    init_db,
    remove_stock_from_pool,
    update_fundamental_data,
)

CURRENT_DIR = Path(__file__).resolve().parent
FUNDAMENTAL_DIR = CURRENT_DIR.parent / "fundamental"
if str(FUNDAMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(FUNDAMENTAL_DIR))

from fundamental_engine import (
    APP_VERSION as FUND_APP_VERSION,
    analyze_watchlist as analyze_fundamental_watchlist,
    build_overview_table as build_fundamental_overview_table,
    format_pct as format_fundamental_pct,
)

st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")
APP_VERSION = "QDB-20260323-DSWIN-03"
LOCAL_PREFS_PATH = "data/local_user_prefs.json"
ANALYSIS_CACHE_PATH = "data/deepseek_analysis_cache.json"
ANALYSIS_JOB_DIR = "data/analysis_jobs"
ANALYSIS_DELTA_CACHE_PATH = "data/deepseek_delta_cache.json"
ANALYSIS_COOLDOWN_PATH = "data/deepseek_cooldown.json"
DEEP_COOLDOWN_MINUTES = 5
DEEPSEEK_SYSTEM_PROMPT = """你是一个专业的股票分析师。必须严格按照【五维分析框架】分析：

【五维分析框架】
一、核心数据摘要表格
- 行情指标：现价/涨跌幅/高低点/量比/换手率
- 资金指标：委差/买卖失衡比/盘口结构
- 估值指标：PE/PB/股息率/市值
- 技术指标：RSI/MACD/均线/布林带

二、五组数据交叉分析
1. 量价关系：价格涨跌 + 量比 + 委差 + 成交量
2. 多周期共振：日线/周线/月线/日内RSI和MACD对比
3. 估值与股息：PE + 股息率 + PB + 市值
4. 均线与布林带：现价 + MA5/MA10/MA20/MA60 + 布林带位置
5. 盘口与日内：委差 + 高低点 + 收盘价

三、综合结论与三种情景概率
- 3个关键结论要点
- 乐观情景(概率+条件+目标)
- 中性情景(概率+条件+区间)
- 悲观情景(概率+条件+支撑)

四、操作策略建议表格
五、数据潜力挖掘说明

要求：简洁、数据驱动、每部分控制在200字以内"""
DEEPSEEK_QUICK_PROMPT = """你是量化交易快筛分析器。请基于输入JSON执行低成本快筛：
1) 给出 risk_level: low/medium/high
2) 给出 3 条简短结论（每条不超过25字）
3) 给出 need_full_analysis: true/false
4) 给出 trigger_reasons 数组（最多4条）
输出必须是 JSON 对象，不要输出任何额外文字。"""
FUND_DEEPSEEK_PROMPT = """你是专业基本面分析师。基于输入 JSON 做结构化输出：
1) 总结（不超过120字）
2) 八维点评（每维1句）
3) 关键风险（3条）
4) 跟踪清单（3条）
5) 结论：通过 / 观察 / 谨慎（给出理由）
要求：数据驱动、简洁、中文输出。"""


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

st.markdown(
    """
    <style>
    :root {
        --bg-main: #edf3fa;
        --text-strong: #15253f;
        --text-normal: #1f334f;
        --text-muted: #536985;
    }
    .stApp {
        background: linear-gradient(180deg, var(--bg-main) 0%, #e8eff8 100%);
        color: var(--text-normal);
    }
    [data-testid="stSidebar"] {
        background: #1e2432;
        color: #e8eef8;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e8eef8 !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="tertiary"]) {
        background: #dbeafe !important;
        color: #0f2a52 !important;
        border: 1px solid #a8c2e8 !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="tertiary"]) span,
    [data-testid="stSidebar"] .stButton > button:not([kind="tertiary"]) p,
    [data-testid="stSidebar"] .stButton > button:not([kind="tertiary"]) div {
        color: #0f2a52 !important;
        -webkit-text-fill-color: #0f2a52 !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="tertiary"]):hover {
        background: #c7ddfb !important;
        color: #0b2346 !important;
    }
    h1, h2, h3, h4 { color: var(--text-strong) !important; }
    .stButton > button:not([kind="tertiary"]) {
        background: #dbeafe;
        color: #0f2a52;
        border: 1px solid #a8c2e8;
        font-weight: 600;
    }
    .stButton > button:not([kind="tertiary"]):hover {
        background: #c7ddfb;
        color: #0b2346;
    }
    [data-testid="stMetricLabel"] div { color: #5b6f89 !important; }
    [data-testid="stMetricValue"] div { color: #15253f !important; }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #f7fbff !important;
        border: 1px solid #b8cdea !important;
        color: #0f2a52 !important;
    }
    [data-testid="stToggle"] label p,
    [data-testid="stSelectbox"] label p {
        color: #1f334f !important;
        font-weight: 700 !important;
    }
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] label span {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] label span,
    div[data-testid="stToggle"] label p,
    div[data-testid="stToggle"] label [data-testid="stMarkdownContainer"],
    div[data-testid="stToggle"] label [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }
    .engine-divider {
        margin: 2.4rem 0 2rem 0;
        border-top: 4px solid #b8c9de;
        position: relative;
    }
    .engine-divider span {
        position: relative;
        top: -1.45rem;
        background: #edf3fa;
        padding: 0 0.8rem;
        color: #15253f;
        font-weight: 800;
        font-size: 2.05rem;
        line-height: 1.1;
    }
    .section-title {
        color: #15253f;
        font-size: 2.05rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.9rem 0 0.8rem 0;
    }
    .fast-head-title {
        color: #324760;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .fast-price-line {
        display: flex;
        align-items: baseline;
        gap: 0.8rem;
        margin: 0.3rem 0 0.7rem 0;
    }
    .price-num {
        font-size: 2.9rem;
        font-weight: 800;
        line-height: 1;
    }
    .chg-num {
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1;
    }
    .a-up { color: #d14343; }
    .a-down { color: #1fab63; }
    .fast-card {
        background: #f5f7fb;
        border: 1px solid #d9e2ef;
        border-radius: 10px;
        padding: 0.62rem 0.78rem;
        height: 156px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .fast-card .t {
        color: #5f738f;
        font-size: 0.94rem;
        font-weight: 700;
    }
    .fast-card .rows {
        margin-top: 0.25rem;
        display: grid;
        gap: 0.14rem;
        flex: 1;
        overflow: hidden;
    }
    .fast-card .krow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.4rem;
        line-height: 1.2;
        font-size: 0.82rem;
    }
    .fast-card .k {
        color: #7689a2;
        font-weight: 600;
    }
    .fast-card .vv {
        color: #1f2d42;
        font-weight: 800;
        text-align: right;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.1px;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .fast-card .d {
        color: #8a98ac;
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .ob-title {
        font-size: 1.95rem;
        color: #23364f;
        font-weight: 800;
    }
    .panel-title {
        font-size: 2.7rem;
        color: #1e3450;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
        letter-spacing: 0.2px;
    }
    .panel-title .unit-sub {
        display: block;
        font-size: 1.15rem;
        line-height: 1.15;
        font-weight: 700;
        color: #5f738f;
        margin-top: 0.12rem;
    }
    .fast-panels-gap {
        height: 0.75rem;
    }
    .subsection-divider {
        margin: 0.9rem 0 1.1rem 0;
        border-top: 3px solid #c4d3e6;
    }
    .ob-block { margin-top: 0.3rem; }
    .ob-row {
        display: grid;
        grid-template-columns: 44px 78px 1fr 56px;
        gap: 0.5rem;
        align-items: center;
        margin: 0.18rem 0;
    }
    .ob-lab {
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }
    .ob-price {
        font-weight: 700;
        font-size: 1.05rem;
        text-align: right;
        padding-right: 4px;
    }
    .ob-bar-wrap {
        height: 24px;
        background: rgba(207, 221, 236, 0.38);
        border-radius: 4px;
        position: relative;
        overflow: hidden;
    }
    .ob-bar {
        height: 100%;
        border-radius: 4px;
    }
    .ob-bar.sell { background: rgba(59, 180, 107, 0.25); }
    .ob-bar.buy { background: rgba(231, 98, 98, 0.28); }
    .ob-vol {
        text-align: right;
        color: #2f4059;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.2px;
    }
    .ob-sell { color: #2f9f5d; }
    .ob-buy { color: #d84f4f; }
    .ob-sep {
        border-top: 1px solid #d7e0ec;
        margin: 0.5rem 0;
    }
    .stock-open-wrap div.stButton > button {
        min-height: 58px !important;
        border-radius: 10px !important;
        white-space: pre-line !important;
        line-height: 1.12 !important;
        font-size: 0.97rem !important;
        font-weight: 800 !important;
        padding: 0.14rem 0.22rem !important;
    }
    .stock-open-wrap div[data-testid="stButton"],
    .stock-del-inline-wrap div[data-testid="stButton"] {
        margin-bottom: 0.06rem !important;
    }
    .stock-open-wrap div.stButton > button * {
        white-space: pre-line !important;
    }
    .stock-open-wrap div.stButton > button p {
        margin: 0 !important;
        text-align: center !important;
    }
    .stock-open-wrap div.stButton > button p:last-child {
        font-size: 0.86rem !important;
        letter-spacing: 0.5px !important;
        font-variant-numeric: tabular-nums !important;
    }
    .stock-del-inline-wrap div.stButton > button {
        min-height: 58px !important;
        border-radius: 10px !important;
        border: none !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        color: #5d708a !important;
        font-size: 1.05rem !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    .stock-del-inline-wrap div.stButton > button:hover,
    .stock-del-inline-wrap div.stButton > button:focus,
    .stock-del-inline-wrap div.stButton > button:active {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        color: #1f334f !important;
        border: none !important;
        box-shadow: none !important;
    }
    .watch-split-divider {
        min-height: 0;
        border-left: 2px solid #c7d3e3;
        margin: 0.2rem auto 0 auto;
        width: 1px;
    }
    .group-title {
        color: #15253f;
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 0.45rem 0;
    }
    .rsi-switch .stButton > button {
        height: 32px !important;
        padding: 0 0.45rem !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        font-weight: 800 !important;
    }
    .rsi-switch .stButton > button[kind="primary"] {
        background: #89addd !important;
        border: 1px solid #5f89c3 !important;
        color: #0f2a52 !important;
        box-shadow: inset 0 0 0 2px #3f6ea8 !important;
    }
    .rsi-switch-day .stButton > button { background: #dbeafe !important; color: #1e3a8a !important; border: 1px solid #93c5fd !important; }
    .rsi-switch-week .stButton > button { background: #dcfce7 !important; color: #166534 !important; border: 1px solid #86efac !important; }
    .rsi-switch-month .stButton > button { background: #fef3c7 !important; color: #92400e !important; border: 1px solid #fcd34d !important; }
    .rsi-switch-intra .stButton > button { background: #fee2e2 !important; color: #991b1b !important; border: 1px solid #fca5a5 !important; }
    .score-panel {
        border: 1px solid rgba(80,120,180,.25);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(240,245,255,0.75);
        min-height: 112px;
    }
    .score-panel .label {
        color: #5a7090;
        font-size: 1.02rem;
        font-weight: 700;
    }
    .score-panel .value {
        color: #15253f;
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 8px;
    }
    .fnd-card {
        border: 1px solid rgba(80,120,180,.25);
        border-radius: 14px;
        padding: 12px 14px;
        min-height: 208px;
        background: rgba(240,245,255,0.55);
        display: flex;
        flex-direction: column;
    }
    .fnd-card h4 {
        margin: 0 0 6px 0;
        font-size: 1.45rem;
    }
    .fnd-card .score {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 2px 0 6px 0;
    }
    .fnd-card .desc {
        color: #5c6e89;
        font-size: 1.0rem;
        line-height: 1.42;
        min-height: 4.26em;
    }
    .fnd-card .desc .line {
        display: block;
        min-height: 1.42em;
    }
    .fnd-card .desc .line-empty {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "trading"

st.sidebar.markdown("---")
st.sidebar.subheader("股票池管理")
new_query = st.sidebar.text_input(
    "新增股票（代码或名称）", value="", placeholder="例如 600036 / 00700 / 腾讯控股"
)

add_cols = st.sidebar.columns(2)
add_holding = add_cols[0].button("加入持仓", use_container_width=True)
add_watch = add_cols[1].button("加入观察", use_container_width=True)
if add_holding or add_watch:
    pool_group = "holding" if add_holding else "watch"
    group_text = "持仓" if pool_group == "holding" else "观察"
    try:
        code, name = add_stock_by_query(new_query, pool_group=pool_group)
        update_fundamental_data([(code, name, pool_group)])
        st.sidebar.success(f"已加入{group_text}: {code} - {name}")
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"添加失败: {exc}")

pool_rows_for_sidebar = get_stock_pool()
pool_name_map = {code: name for code, name in pool_rows_for_sidebar}
if pool_rows_for_sidebar:
    remove_code = st.sidebar.selectbox(
        "删除股票",
        options=[code for code, _ in pool_rows_for_sidebar],
        format_func=lambda c: f"{pool_name_map.get(c, c)} ({c})",
    )
    if st.sidebar.button("删除选中", use_container_width=True):
        remove_stock_from_pool(remove_code)
        if st.session_state.get("fast_selected_code") == remove_code:
            st.session_state.pop("fast_selected_code", None)
            st.session_state.pop("fast_selected_name", None)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("DeepSeek API")
if "_prefs_loaded" not in st.session_state:
    _prefs = _load_local_prefs()
    if "deepseek_user_input" not in st.session_state:
        st.session_state["deepseek_user_input"] = _prefs.get("deepseek_user", "")
    if "deepseek_api_key_input" not in st.session_state:
        st.session_state["deepseek_api_key_input"] = _prefs.get("deepseek_api_key", "")
    st.session_state["_last_saved_prefs"] = {
        "deepseek_user": st.session_state.get("deepseek_user_input", ""),
        "deepseek_api_key": st.session_state.get("deepseek_api_key_input", ""),
    }
    st.session_state["_prefs_loaded"] = True

analysis_user_input = st.sidebar.text_input(
    "用户名（用于区分不同使用者）",
    value=st.session_state.get("deepseek_user_input", ""),
    key="deepseek_user_input",
)
analysis_api_key_input = st.sidebar.text_input(
    "API Key（可留空，读取环境变量）",
    value=st.session_state.get("deepseek_api_key_input", ""),
    type="password",
    key="deepseek_api_key_input",
)

_curr_user = (analysis_user_input or "").strip()
_curr_key = (analysis_api_key_input or "").strip()
_last = st.session_state.get("_last_saved_prefs", {})
if _curr_user != _last.get("deepseek_user", "") or _curr_key != _last.get("deepseek_api_key", ""):
    _save_local_prefs(_curr_user, _curr_key)
    st.session_state["_last_saved_prefs"] = {
        "deepseek_user": _curr_user,
        "deepseek_api_key": _curr_key,
    }

st.sidebar.markdown("---")
st.sidebar.subheader("页面切换")
if st.sidebar.button("基本面", use_container_width=True, type="primary" if st.session_state["active_page"] == "fundamental" else "secondary"):
    st.session_state["active_page"] = "fundamental"
if st.sidebar.button("交易面", use_container_width=True, type="primary" if st.session_state["active_page"] == "trading" else "secondary"):
    st.session_state["active_page"] = "trading"

st.title("基本面" if st.session_state.get("active_page") == "fundamental" else "股票观察面板")

rows = get_latest_fundamental_snapshot()
if st.session_state.get("active_page") == "trading" and not rows:
    st.info("数据库暂无慢引擎快照，请先在左侧添加股票。")
    st.stop()

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


def _call_deepseek_analysis(json_text: str) -> tuple[str, dict, float, float]:
    return _call_deepseek_with_prompt(
        user_content=json_text,
        system_prompt=DEEPSEEK_SYSTEM_PROMPT,
        max_tokens=1500,
        temperature=0.3,
        top_p=0.9,
    )


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
        st.session_state["fnd_rows"] = analyze_fundamental_watchlist(watchlist, force_refresh=force_refresh)
        st.session_state["fnd_watchlist_hash"] = wl_hash
        if st.session_state["fnd_rows"]:
            valid_codes = {str(x.get("code", "")) for x in st.session_state["fnd_rows"]}
            if st.session_state.get("fnd_selected_code") not in valid_codes:
                st.session_state["fnd_selected_code"] = st.session_state["fnd_rows"][0]["code"]
        else:
            st.session_state["fnd_selected_code"] = ""
    return watchlist, st.session_state.get("fnd_rows", [])


def _render_fundamental_page():
    st.caption(f"版本号: {FUND_APP_VERSION}")
    top_cols = st.columns([1, 5], vertical_alignment="center")
    if top_cols[0].button("刷新基本面", use_container_width=True, key="refresh_fundamental_now"):
        _ensure_fundamental_state(force_refresh=True)
        st.rerun()

    watchlist, rows_fnd = _ensure_fundamental_state(force_refresh=False)
    if not watchlist:
        st.warning("当前股票池为空，请先在左侧添加股票。")
        return
    if not rows_fnd:
        st.info("正在生成基本面数据，请稍后刷新。")
        return

    st.subheader("股票列表")
    df = build_fundamental_overview_table(rows_fnd).copy()
    st.dataframe(df, use_container_width=True, hide_index=True)

    options = [str(r.get("code", "")) for r in rows_fnd]
    current = str(st.session_state.get("fnd_selected_code", options[0] if options else ""))
    idx = options.index(current) if current in options else 0
    chosen = st.selectbox(
        "打开评分板",
        options=options,
        index=idx,
        format_func=lambda c: next((f"{x.get('name','')} ({x.get('code','')})" for x in rows_fnd if str(x.get("code","")) == c), c),
        key="fnd_open_score_select",
    )
    if chosen != current:
        st.session_state["fnd_selected_code"] = chosen
        st.rerun()

    st.divider()
    row = next((x for x in rows_fnd if str(x.get("code", "")) == str(st.session_state.get("fnd_selected_code", ""))), rows_fnd[0])
    st.subheader(f"基本面评分板：{row.get('name', '')}（{row.get('code', '')}）")
    score = float(row.get("total_score", 0.0) or 0.0)
    conclusion = _clean_text_no_na(str(row.get("conclusion", "观察")))
    coverage = format_fundamental_pct(float((row.get("coverage_ratio") or 0.0) * 100.0))
    sp_cols = st.columns(3, gap="small")
    for col, (label, value) in zip(sp_cols, [("总分", f"{score:.1f}"), ("结论", conclusion), ("覆盖率", coverage)]):
        col.markdown(
            f"""
<div class="score-panel">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    dims = row.get("dimensions", []) or []
    if dims:
        st.subheader("八维评分")
        for i in range(0, len(dims), 4):
            cols = st.columns(4, gap="small")
            for j, card in enumerate(dims[i : i + 4]):
                title = _clean_text_no_na(card.get("title", ""))
                score_txt = _clean_text_no_na(f"{card.get('score', 0)} / {card.get('max_score', 5)}")
                desc = _format_card_desc_lines(str(card.get("comment", "")))
                with cols[j]:
                    st.markdown(
                        f"""
<div class="fnd-card">
  <h4>{title}</h4>
  <div class="score">{score_txt}</div>
  <div class="desc">{desc}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("总结性文本")
    lines = _split_sentences(str(row.get("summary_text", "")))
    if lines:
        st.markdown(
            "<div style='line-height:1.8;color:#243a58;font-size:1.05rem;'>"
            + "<br>".join(lines)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("暂无总结。")

    code = str(row.get("code", ""))
    json_payload = json.dumps(row, ensure_ascii=False, indent=2)
    json_b64 = base64.b64encode(json_payload.encode("utf-8")).decode("ascii")
    btn1, btn2 = st.columns([1, 1], gap="small")
    with btn1:
        html(
            f"""
            <div style="margin-top:0.1rem;">
              <button id="fnd-copy-json-{code}"
                style="width:100%;height:42px;border-radius:10px;border:1px solid #a8c2e8;background:#dbeafe;color:#0f2a52;font-size:1rem;font-weight:700;cursor:pointer;">
                复制JSON
              </button>
              <div id="fnd-copy-msg-{code}" style="margin-top:0.35rem;color:#2e4b6e;font-size:0.86rem;"></div>
            </div>
            <script>
              const btn = document.getElementById("fnd-copy-json-{code}");
              const msg = document.getElementById("fnd-copy-msg-{code}");
              const text = decodeURIComponent(escape(window.atob("{json_b64}")));
              btn.onclick = async function () {{
                try {{
                  await navigator.clipboard.writeText(text);
                  msg.textContent = "已复制";
                }} catch(e) {{
                  msg.textContent = "复制失败，请重试";
                }}
              }};
            </script>
            """,
            height=88,
        )
    with btn2:
        if st.button("DeepSeek分析", key=f"fnd_deepseek_{code}", use_container_width=True):
            progress = st.progress(0, text="正在准备分析任务...")
            pytime.sleep(0.08)
            progress.progress(35, text="正在压缩数据...")
            pytime.sleep(0.08)
            progress.progress(70, text="正在连接 DeepSeek...")
            try:
                report, usage, cost, elapsed = _call_deepseek_fundamental(json_payload)
                progress.progress(100, text="分析完成")
                pytime.sleep(0.1)
                progress.empty()
                st.session_state["fnd_deepseek_reports"][code] = {
                    "report": (report or "").strip(),
                    "usage": usage,
                    "cost": cost,
                    "elapsed": elapsed,
                    "at": datetime.now().strftime("%m-%d %H:%M:%S"),
                }
            except Exception as exc:
                progress.empty()
                st.error(f"DeepSeek 分析失败: {exc}")

    deep = st.session_state.get("fnd_deepseek_reports", {}).get(code)
    if deep:
        st.divider()
        st.subheader("DeepSeek分析结果")
        st.caption(
            f"分析时间: {deep.get('at','')} ｜耗时: {deep.get('elapsed',0):.2f}s ｜"
            f"Tokens: {deep.get('usage',{}).get('total_tokens',0)} ｜"
            f"预估成本: {deep.get('cost',0):.4f} 元"
        )
        report_text = (deep.get("report", "") or "").strip()
        st.markdown(report_text)
        st.text_area("分析文本（可复制）", value=report_text, height=260, key=f"fnd_report_{code}")


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
        "analysis_engine": "deep_only_v2",
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
        if str(current.get("analysis_engine", "")) != "deep_only_v2":
            current["analysis_engine"] = "deep_only_v2"
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
        "analysis_engine": "deep_only_v2",
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


def _render_final_report_block(job_id: str, job_obj: dict, key_suffix: str, height: int = 560) -> None:
    final_text = str(job_obj.get("final_text", "") or "")
    done_mark = str(job_obj.get("done_at", "") or "")
    text_key = f"job_text_{job_id}_{key_suffix}_{hashlib.md5((final_text + done_mark).encode('utf-8')).hexdigest()[:10]}"
    stats = job_obj.get("stats", {}) or {}
    st.caption(
        f"深析输入: {stats.get('deep_prompt_tokens', 0)} | 深析输出: {stats.get('deep_completion_tokens', 0)} | "
        f"预估总成本: {float(stats.get('total_cost', 0) or 0):.4f} 元"
    )
    report_text = final_text.strip()
    if report_text:
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
          <span id="copy-job-msg-{job_id}-{key_suffix}" style="margin-left:0.55rem;color:#2e4b6e;font-size:0.86rem;"></span>
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


def _execute_analysis_job(job_id: str, mode: str, ui_prefix: str = "") -> dict:
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
        progress.progress(28, text=f"{ui_prefix}检查缓存...")
        d_cached = cache_store.get(deep_hash) if deep_hash else None
        if d_cached and isinstance(d_cached, dict):
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
                deep_report, d_usage, d_cost, _ = _call_deepseek_analysis(json_text=deep_json)
                deep_usage = d_usage
                total_cost += float(d_cost or 0.0)
                cache_store[deep_hash] = {
                    "stage": "deep",
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
        final_text = f"## DeepSeek 深析结果（{deep_source}）\n\n{deep_report}"

        stats = {
            "deep_prompt_tokens": int(deep_usage.get("prompt_tokens", 0) or 0),
            "deep_completion_tokens": int(deep_usage.get("completion_tokens", 0) or 0),
            "total_cost": float(total_cost),
            "deep_source": deep_source,
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
    if embedded:
        run_mode = "deep" if auto_mode else ""
    else:
        if st.button("DeepSeek分析", key=f"analysis_window_deep_{job_id}", use_container_width=True):
            run_mode = "deep"

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
        done_job = _execute_analysis_job(job_id=job_id, mode="deep", ui_prefix=f"{stock_name} ")
        if not embedded:
            st.success("深析已完成。")
        else:
            st.success("分析完成。")
        _render_final_report_block(job_id, done_job, "new", height=420 if embedded else 560)
    except Exception as exc:
        st.error(f"分析失败: {type(exc).__name__}: {exc}")


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


if st.session_state.get("active_page") == "fundamental":
    _render_fundamental_page()
    st.stop()


if "fast_selected_code" not in st.session_state:
    st.session_state["fast_selected_code"] = rows[0]["code"]
    st.session_state["fast_selected_name"] = rows[0]["name"]

selected_code_for_ctrl = st.session_state["fast_selected_code"]
market_open_for_ctrl = _is_market_open(selected_code_for_ctrl)

header_cols = st.columns([2.4, 0.8, 0.6, 0.9], vertical_alignment="bottom")
auto_refresh_on = header_cols[1].checkbox("自动刷新", value=False, key="fast_auto_refresh_on")
auto_refresh_sec = header_cols[2].selectbox(
    "刷新间隔(秒)",
    options=[15, 30, 60, 90, 120],
    index=2,
    key="fast_auto_refresh_sec",
)
if header_cols[3].button("立即刷新", use_container_width=True, disabled=not market_open_for_ctrl):
    st.rerun()

group_map = get_stock_group_map()
holding_rows = [r for r in rows if group_map.get(str(r["code"]), "watch") == "holding"]
watch_rows = [r for r in rows if group_map.get(str(r["code"]), "watch") != "holding"]


def _stock_grid_cols(total: int) -> int:
    if total <= 1:
        return 1
    if total <= 4:
        return 2
    if total <= 9:
        return 3
    return 4


def _render_stock_group(stock_rows, group_key_prefix: str) -> None:
    if not stock_rows:
        st.caption("暂无标的")
        return

    grid_cols = _stock_grid_cols(len(stock_rows))
    for start in range(0, len(stock_rows), grid_cols):
        row_cols = st.columns(grid_cols)
        chunk = stock_rows[start : start + grid_cols]
        for idx, row in enumerate(chunk):
            col = row_cols[idx]
            with col:
                open_col, del_col = st.columns([5.2, 1], vertical_alignment="center")
                with open_col:
                    st.markdown('<div class="stock-open-wrap">', unsafe_allow_html=True)
                    if st.button(
                        f"{row['name']}\n{row['code']}",
                        key=f"open_fast_{group_key_prefix}_{row['code']}",
                        use_container_width=True,
                    ):
                        st.session_state["fast_selected_code"] = row["code"]
                        st.session_state["fast_selected_name"] = row["name"]
                    st.markdown("</div>", unsafe_allow_html=True)
                with del_col:
                    st.markdown('<div class="stock-del-inline-wrap">', unsafe_allow_html=True)
                    if st.button(
                        "🗑️",
                        key=f"mini_del_{group_key_prefix}_{row['code']}",
                        use_container_width=True,
                        type="tertiary",
                        help=f"删除 {row['name']}",
                    ):
                        remove_stock_from_pool(row["code"])
                        if st.session_state.get("fast_selected_code") == row["code"]:
                            st.session_state.pop("fast_selected_code", None)
                            st.session_state.pop("fast_selected_name", None)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)


holding_rows_needed = math.ceil(len(holding_rows) / max(_stock_grid_cols(len(holding_rows)), 1)) if holding_rows else 1
watch_rows_needed = math.ceil(len(watch_rows) / max(_stock_grid_cols(len(watch_rows)), 1)) if watch_rows else 1
divider_height = max(110, max(holding_rows_needed, watch_rows_needed) * 94 + 16)

group_cols = st.columns([1, 0.02, 1], vertical_alignment="top")
with group_cols[0]:
    st.markdown('<div class="group-title">持仓</div>', unsafe_allow_html=True)
    _render_stock_group(holding_rows, "holding")
with group_cols[1]:
    st.markdown(
        f'<div class="watch-split-divider" style="height:{divider_height}px;"></div>',
        unsafe_allow_html=True,
    )
with group_cols[2]:
    st.markdown('<div class="group-title">观察</div>', unsafe_allow_html=True)
    _render_stock_group(watch_rows, "watch")

def _render_fast_panel(selected_code: str, selected_name: str, panel=None):
    if panel is None:
        panel = fetch_fast_panel(selected_code)
    quote = panel["quote"]
    ind = panel["indicators"]
    intraday_df = panel["intraday"]
    order_book_5 = panel["order_book_5"]

    if panel.get("error") and not quote.get("current_price"):
        st.warning(f"快引擎数据拉取失败: {panel['error']}")
        return

    selected_slow = next((r for r in rows if str(r.get("code")) == str(selected_code)), {})
    sell_lv_for_json = sorted(order_book_5.get("sell", []), key=lambda x: int(x.get("level", 0)))
    buy_lv_for_json = sorted(order_book_5.get("buy", []), key=lambda x: int(x.get("level", 0)))
    sell_total_for_json = sum(float(r.get("volume_lot") or 0) for r in sell_lv_for_json)
    buy_total_for_json = sum(float(r.get("volume_lot") or 0) for r in buy_lv_for_json)
    ofi_for_json = (buy_total_for_json / sell_total_for_json) if sell_total_for_json > 0 else None
    ask1_for_json = next((r.get("price") for r in sell_lv_for_json if int(r.get("level", 0)) == 1), None)
    bid1_for_json = next((r.get("price") for r in buy_lv_for_json if int(r.get("level", 0)) == 1), None)
    spread_for_json = (
        float(ask1_for_json) - float(bid1_for_json)
        if (ask1_for_json is not None and bid1_for_json is not None)
        else None
    )

    fast_compact_metrics = {
        "snapshot": {
            "current_price": quote.get("current_price"),
            "change_pct": quote.get("change_pct"),
            "change_amount": quote.get("change_amount"),
            "open": quote.get("open"),
            "prev_close": quote.get("prev_close"),
            "high": quote.get("high"),
            "low": quote.get("low"),
        },
        "trading": {
            "volume": quote.get("volume"),
            "amount": quote.get("amount"),
            "turnover_rate": quote.get("turnover_rate"),
            "amplitude_pct": quote.get("amplitude_pct"),
            "volume_ratio": quote.get("volume_ratio"),
            "vwap": quote.get("vwap"),
            "premium_pct": quote.get("premium_pct"),
        },
        "order_book_summary": {
            "buy_total_lot": buy_total_for_json,
            "sell_total_lot": sell_total_for_json,
            "imbalance_bid_ask": ofi_for_json,
            "spread": spread_for_json,
            "order_diff": quote.get("order_diff"),
        },
        "valuation": {
            "pe_dynamic": selected_slow.get("pe_dynamic"),
            "pe_static": selected_slow.get("pe_static"),
            "pe_rolling": selected_slow.get("pe_rolling"),
            "pb": selected_slow.get("pb"),
            "dividend_yield": selected_slow.get("dividend_yield"),
            "total_market_value_yi": quote.get("total_market_value_yi"),
            "float_market_value_yi": quote.get("float_market_value_yi"),
        },
        "technical": {
            "macd_hist": ind.get("macd_hist"),
            "rsi6": ind.get("rsi6"),
            "rsi12": ind.get("rsi12"),
            "rsi24": ind.get("rsi24"),
            "rsi_multi": panel.get("rsi_multi", {}),
            "tf_indicators": panel.get("tf_indicators", {}),
            "ma5": ind.get("ma5"),
            "ma10": ind.get("ma10"),
            "ma20": ind.get("ma20"),
            "ma60": ind.get("ma60"),
            "boll_mid": ind.get("boll_mid"),
            "boll_upper": ind.get("boll_upper"),
            "boll_lower": ind.get("boll_lower"),
            "boll_pct_b": ind.get("boll_pct_b"),
            "boll_bandwidth": ind.get("boll_bandwidth"),
            "rsi_method": "Wilder(SMA, N,1)",
        },
    }

    price_now = quote.get("current_price")
    prev_close_for_pct = quote.get("prev_close")
    api_change_pct = quote.get("change_pct")
    calc_change_pct = None
    if (
        price_now is not None
        and prev_close_for_pct is not None
        and prev_close_for_pct > 0
    ):
        calc_change_pct = (price_now - prev_close_for_pct) / prev_close_for_pct * 100

    # 以现价/昨收重算为主，避免接口涨跌幅字段偶发异常导致颜色反向
    change_pct = calc_change_pct if calc_change_pct is not None else api_change_pct
    is_down = change_pct is not None and change_pct < 0
    price_class = "a-down" if is_down else "a-up"
    fast_compact_metrics["snapshot"]["display_change_pct"] = change_pct

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    head_left, head_right = st.columns([3.2, 1], vertical_alignment="center")
    copy_slot = None
    if price_now is not None:
        with head_left:
            st.markdown(
                f"""
                <div class="fast-head-title">{selected_name} ({selected_code})</div>
                <div class="fast-price-line">
                    <span class="price-num {price_class}">{price_now:.2f}</span>
                    <span class="chg-num {price_class}">{(change_pct or 0):+.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            q_time = _format_display_time(quote.get("quote_time"))
            st.caption(f"更新时间: {q_time if q_time else 'N/A'}")
    with head_right:
        copy_slot = st.container()

    def _fmt(v, nd=2):
        return "N/A" if v is None else f"{v:.{nd}f}"

    def _fmt_pct(v, nd=2):
        return "N/A" if v is None else f"{v:.{nd}f}%"

    def _fmt_signed(v, nd=2):
        return "N/A" if v is None else f"{v:+.{nd}f}"

    def _fmt_signed_pct(v, nd=2):
        return "N/A" if v is None else f"{v:+.{nd}f}%"

    def _fmt_lot(v):
        if v is None:
            return "N/A"
        return f"{int(v):,}"

    def _fmt_amount_yuan(v):
        if v is None:
            return "N/A"
        n = float(v)
        if abs(n) >= 1e8:
            return f"{n/1e8:.2f}亿"
        if abs(n) >= 1e4:
            return f"{n/1e4:.2f}万"
        return f"{n:.0f}"

    def _find_level(rows_data, level):
        for r in rows_data:
            if int(r.get("level", 0)) == int(level):
                return r
        return {}

    def _fmt_price_list(rows_data):
        vals = []
        for lv in range(1, 6):
            r = _find_level(rows_data, lv)
            vals.append(_fmt(r.get("price"), 2) if r else "N/A")
        return " / ".join(vals)

    def _fmt_vol_list(rows_data):
        vals = []
        for lv in range(1, 6):
            r = _find_level(rows_data, lv)
            vv = r.get("volume_lot") if r else None
            vals.append("--" if vv is None else str(int(float(vv))))
        return " / ".join(vals)

    def _rows_html(rows_data):
        return "".join(
            f'<div class="krow"><span class="k">{k}</span><span class="vv">{v}</span></div>'
            for k, v in rows_data
        )

    def _card_html(title, rows_data, desc=""):
        rows_html = _rows_html(rows_data)
        desc_html = f'<div class="d">{desc}</div>' if desc else ""
        return f'<div class="fast-card"><div class="t">{title}</div><div class="rows">{rows_html}</div>{desc_html}</div>'

    macd_val = ind.get("macd_hist")
    rsi_multi = panel.get("rsi_multi", {}) or {}
    rsi_tf_state = f"rsi_tf_key_{selected_code}"
    if st.session_state.get(rsi_tf_state) not in {"day", "week", "month", "intraday"}:
        st.session_state[rsi_tf_state] = "day"

    tf_cols = st.columns([0.42, 0.42, 0.42, 0.62, 2.12])
    tf_conf = [
        ("day", "日", "rsi-switch-day"),
        ("week", "周", "rsi-switch-week"),
        ("month", "月", "rsi-switch-month"),
        ("intraday", "分时", "rsi-switch-intra"),
    ]
    for idx, (tf_key, tf_label, tf_cls) in enumerate(tf_conf):
        is_active = st.session_state[rsi_tf_state] == tf_key
        with tf_cols[idx]:
            st.markdown(f'<div class="rsi-switch {tf_cls}">', unsafe_allow_html=True)
            if st.button(
                tf_label,
                key=f"rsi_tf_btn_{selected_code}_{tf_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state[rsi_tf_state] = tf_key
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    active_tf = st.session_state[rsi_tf_state]
    tf_indicators = panel.get("tf_indicators", {}) if isinstance(panel.get("tf_indicators", {}), dict) else {}
    active_ind = tf_indicators.get(active_tf, {}) if isinstance(tf_indicators, dict) else {}
    active_rsi = rsi_multi.get(active_tf, {}) if isinstance(rsi_multi, dict) else {}
    rsi_val = active_rsi.get("rsi6", active_ind.get("rsi6", ind.get("rsi6")))
    rsi12_val = active_rsi.get("rsi12", active_ind.get("rsi12", ind.get("rsi12")))
    rsi24_val = active_rsi.get("rsi24", active_ind.get("rsi24", ind.get("rsi24")))
    ma5_val = active_ind.get("ma5", ind.get("ma5"))
    ma10_val = active_ind.get("ma10", ind.get("ma10"))
    ma20_val = active_ind.get("ma20", ind.get("ma20"))
    ma60_val = active_ind.get("ma60", ind.get("ma60"))
    boll_mid_val = active_ind.get("boll_mid", ind.get("boll_mid"))
    boll_pct_b_fast = active_ind.get("boll_pct_b", ind.get("boll_pct_b"))
    boll_bw = active_ind.get("boll_bandwidth", ind.get("boll_bandwidth"))
    ref_val = quote.get("prev_close")
    boll_val = selected_slow.get("boll_index")
    pe_dynamic = selected_slow.get("pe_dynamic")
    pe_static = selected_slow.get("pe_static")
    pe_rolling = selected_slow.get("pe_rolling")
    pb_val = selected_slow.get("pb")
    dy_val = selected_slow.get("dividend_yield")

    open_val = quote.get("open")
    high_val = quote.get("high")
    low_val = quote.get("low")
    change_amt = quote.get("change_amount")
    vwap_val = quote.get("vwap")
    premium_pct = quote.get("premium_pct")
    amplitude_pct = quote.get("amplitude_pct")
    turnover_rate = quote.get("turnover_rate")
    volume_ratio = quote.get("volume_ratio")
    total_mv = quote.get("total_market_value_yi")
    float_mv = quote.get("float_market_value_yi")
    order_diff = quote.get("order_diff")

    volume_shares = quote.get("volume")
    volume_lot = (float(volume_shares) / 100.0) if volume_shares is not None else None
    amount_yuan = quote.get("amount")

    macd_tf_val = active_ind.get("macd_hist", macd_val)
    macd_desc = "趋势偏强" if (macd_tf_val is not None and macd_tf_val > 0) else "趋势偏弱"
    rsi_desc = "超买区间" if (rsi_val is not None and rsi_val >= 70) else ("超卖区间" if (rsi_val is not None and rsi_val <= 30) else "强弱指标")
    tf_name_map = {"day": "日线", "week": "周线", "month": "月线", "intraday": "分时"}
    rsi_desc = f"{tf_name_map.get(active_tf, '日线')} · {rsi_desc}"
    tf_caption = tf_name_map.get(active_tf, "日线")

    sell_lv = sorted(order_book_5.get("sell", []), key=lambda x: int(x.get("level", 0)))
    buy_lv = sorted(order_book_5.get("buy", []), key=lambda x: int(x.get("level", 0)))
    sell_total = sum(float(r.get("volume_lot") or 0) for r in sell_lv)
    buy_total = sum(float(r.get("volume_lot") or 0) for r in buy_lv)
    ofi = (buy_total / sell_total) if sell_total > 0 else None
    ask1 = _find_level(sell_lv, 1).get("price")
    bid1 = _find_level(buy_lv, 1).get("price")
    spread = (float(ask1) - float(bid1)) if (ask1 is not None and bid1 is not None) else None

    cards = [
        (
            "实时快照",
            [
                ("现价", _fmt(price_now, 2)),
                ("涨跌幅", _fmt_signed_pct(change_pct, 2)),
                ("涨跌额", _fmt_signed(change_amt, 2)),
            ],
            "Now / Pct / Chg",
        ),
        (
            "日内区间",
            [
                ("今开", _fmt(open_val, 2)),
                ("昨收", _fmt(ref_val, 2)),
                ("最高", _fmt(high_val, 2)),
                ("最低", _fmt(low_val, 2)),
            ],
            "",
        ),
        (
            "成交活跃",
            [
                ("成交量(手)", _fmt_lot(volume_lot)),
                ("成交额(元)", _fmt_amount_yuan(amount_yuan)),
                ("量比", _fmt(volume_ratio, 2)),
                ("换手率", _fmt_pct(turnover_rate, 2)),
            ],
            "",
        ),
        (
            "波动与均价",
            [
                ("VWAP", _fmt(vwap_val, 2)),
                ("偏离", _fmt_signed_pct(premium_pct, 2)),
                ("振幅", _fmt_pct(amplitude_pct, 2)),
            ],
            "",
        ),
        (
            "盘口结构",
            [
                ("买总量", _fmt_lot(buy_total)),
                ("卖总量", _fmt_lot(sell_total)),
                ("失衡比(B/A)", _fmt(ofi, 2)),
                ("买卖价差", _fmt(spread, 3)),
                ("委差", _fmt_signed(order_diff, 0)),
            ],
            "",
        ),
        (
            "PE 三口径",
            [
                ("PE(动)", _fmt(pe_dynamic, 2)),
                ("PE(静)", _fmt(pe_static, 2)),
                ("PE(滚)", _fmt(pe_rolling, 2)),
            ],
            "Eastmoney 口径",
        ),
        (
            "估值与规模",
            [
                ("PB", _fmt(pb_val, 2)),
                ("股息率", _fmt_pct(dy_val, 2)),
                ("总市值(亿)", _fmt(total_mv, 2)),
                ("流通市值(亿)", _fmt(float_mv, 2)),
            ],
            "",
        ),
        (
            "RSI 组合",
            [
                ("RSI(6)", _fmt(rsi_val, 2)),
                ("RSI(12)", _fmt(rsi12_val, 2)),
                ("RSI(24)", _fmt(rsi24_val, 2)),
            ],
            rsi_desc,
        ),
        (
            "均线组合",
            [
                ("MA5", _fmt(ma5_val, 2)),
                ("MA10", _fmt(ma10_val, 2)),
                ("MA20", _fmt(ma20_val, 2)),
                ("MA60", _fmt(ma60_val, 2)),
            ],
            f"{tf_caption}口径",
        ),
        (
            "MACD",
            [
                ("MACD柱", _fmt(macd_tf_val, 3)),
            ],
            f"{tf_caption} · {macd_desc}",
        ),
        (
            "BOLL",
            [
                ("BOLL %B", _fmt(boll_pct_b_fast if boll_pct_b_fast is not None else boll_val, 2)),
                ("BOLL带宽", _fmt_pct(boll_bw, 2)),
                ("BOLL中轨", _fmt(boll_mid_val, 2)),
            ],
            f"{tf_caption} · 布林带",
        ),
    ]

    cards_snapshot = {
        "timeframe_selected": active_tf,
        "timeframe_label": tf_caption,
        "snapshot": {
            "current_price": price_now,
            "change_pct_display": change_pct,
            "change_amount": change_amt,
            "open": open_val,
            "prev_close": ref_val,
            "high": high_val,
            "low": low_val,
            "quote_time": quote.get("quote_time"),
        },
        "trading": {
            "volume_lot": volume_lot,
            "amount_yuan": amount_yuan,
            "volume_ratio": volume_ratio,
            "turnover_rate": turnover_rate,
            "vwap": vwap_val,
            "premium_pct": premium_pct,
            "amplitude_pct": amplitude_pct,
        },
        "order_book_summary": {
            "buy_total_lot": buy_total,
            "sell_total_lot": sell_total,
            "imbalance_bid_ask": ofi,
            "spread": spread,
            "order_diff": order_diff,
        },
        "valuation": {
            "pe_dynamic": pe_dynamic,
            "pe_static": pe_static,
            "pe_rolling": pe_rolling,
            "pb": pb_val,
            "dividend_yield": dy_val,
            "total_market_value_yi": total_mv,
            "float_market_value_yi": float_mv,
        },
        "technical_current_tf": {
            "macd_hist": macd_tf_val,
            "rsi6": rsi_val,
            "rsi12": rsi12_val,
            "rsi24": rsi24_val,
            "ma5": ma5_val,
            "ma10": ma10_val,
            "ma20": ma20_val,
            "ma60": ma60_val,
            "boll_pct_b": boll_pct_b_fast if boll_pct_b_fast is not None else boll_val,
            "boll_bandwidth": boll_bw,
            "boll_mid": boll_mid_val,
        },
        "cards": {title: {k: v for k, v in kv_rows} for title, kv_rows, _ in cards},
    }

    fast_compact_metrics["ui_state"] = {
        "selected_timeframe": active_tf,
        "selected_timeframe_label": tf_caption,
    }
    fast_compact_metrics["cards_snapshot"] = cards_snapshot

    export_payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "app": "Quant",
            "analysis_user": (st.session_state.get("deepseek_user_input", "") or "").strip(),
        },
        "stock": {"code": selected_code, "name": selected_name},
        "slow_engine": selected_slow,
        "fast_engine": {
            "quote": quote,
            "indicators": ind,
            "rsi_multi": panel.get("rsi_multi", {}),
            "tf_indicators": panel.get("tf_indicators", {}),
            "order_book_5": order_book_5,
            "intraday": intraday_df,
            "depth_note": panel.get("depth_note"),
            "error": panel.get("error"),
            "compact_metrics": fast_compact_metrics,
            "cards_snapshot": cards_snapshot,
        },
    }
    export_json = json.dumps(_json_safe(export_payload), ensure_ascii=False, indent=2)
    analysis_payload = _build_analysis_payload(export_payload)
    analysis_json = json.dumps(analysis_payload, ensure_ascii=True, separators=(",", ":"))
    json_b64 = base64.b64encode(export_json.encode("utf-8")).decode("ascii")
    deep_json = analysis_json
    deep_hash = hashlib.sha256(f"deep:{deep_json}".encode("utf-8")).hexdigest()
    live_job_id = _upsert_live_analysis_job(
        stock_code=selected_code,
        stock_name=selected_name,
        quick_json="",
        deep_json=deep_json,
        quick_hash="",
        deep_hash=deep_hash,
    )
    run_analysis_now = False

    if copy_slot is not None:
        with copy_slot:
            html(
                f"""
                <div style="margin:0.1rem 0 0.45rem 0;">
                  <button id="copy-json-btn-{selected_code}"
                    style="width:100%;height:44px;padding:0 0.95rem;border-radius:10px;border:1px solid #a8c2e8;background:#dbeafe;color:#0f2a52;font-size:1.05rem;font-weight:700;cursor:pointer;white-space:nowrap;">
                    复制JSON
                  </button>
                  <div id="copy-json-msg-{selected_code}" style="margin-top:0.35rem;color:#2e4b6e;font-size:0.88rem;"></div>
                </div>
                <script>
                  const btn = document.getElementById("copy-json-btn-{selected_code}");
                  const msg = document.getElementById("copy-json-msg-{selected_code}");
                  const b64 = "{json_b64}";
                  const text = decodeURIComponent(escape(window.atob(b64)));
                  btn.onclick = async function () {{
                    try {{
                      await navigator.clipboard.writeText(text);
                      msg.textContent = "已复制";
                    }} catch (e) {{
                      msg.textContent = "复制失败，请重试";
                    }}
                  }};
                </script>
                """,
                height=96,
            )
            st.markdown('<div style="margin-top:0.22rem;"></div>', unsafe_allow_html=True)
            run_analysis_now = st.button("DeepSeek分析", key=f"run_inline_analysis_{selected_code}", use_container_width=True)

    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (title, kv_rows, desc) in zip(cols, cards[i : i + 4]):
            col.markdown(_card_html(title, kv_rows, desc), unsafe_allow_html=True)

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fast-panels-gap"></div>', unsafe_allow_html=True)
    left, right = st.columns([2, 1], vertical_alignment="top")
    with left:
        st.markdown('<div class="panel-title">资金分时</div>', unsafe_allow_html=True)
        if intraday_df.empty:
            st.info("暂无分时资金数据")
        else:
            chart_df = intraday_df.set_index("time")
            area_df = chart_df.reset_index()
            # A股配色: 涨红跌绿, 平盘中性灰
            area_color = "#ef4444" if (change_pct or 0) > 0 else ("#22c55e" if (change_pct or 0) < 0 else "#94a3b8")
            chart = (
                alt.Chart(area_df)
                .mark_area(color=area_color, opacity=0.9)
                .encode(
                    x=alt.X("time:T", title="time"),
                    y=alt.Y("volume_lot:Q", title="vol"),
                )
                .properties(height=330)
                .configure_view(strokeOpacity=0)
                .configure_axis(gridColor="#dbe4f0", labelColor="#4a5f7c", titleColor="#4a5f7c")
            )
            st.altair_chart(chart, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">实时盘口<span class="unit-sub">单位：手</span></div>', unsafe_allow_html=True)
        sell_df = pd.DataFrame(order_book_5.get("sell", []))
        buy_df = pd.DataFrame(order_book_5.get("buy", []))

        if sell_df.empty or buy_df.empty:
            st.info("暂无盘口数据")
        else:
            sell_df = sell_df.sort_values("level", ascending=False).copy()
            buy_df = buy_df.sort_values("level", ascending=True).copy()
            vol_max = max(
                1.0,
                max(pd.to_numeric(sell_df["volume_lot"], errors="coerce").fillna(0).max(), pd.to_numeric(buy_df["volume_lot"], errors="coerce").fillna(0).max()),
            )

            def _ob_rows(df: pd.DataFrame, side: str) -> str:
                rows_html = ""
                for _, r in df.iterrows():
                    lvl = int(r.get("level", 0))
                    price = r.get("price")
                    vol = r.get("volume_lot")
                    vol_num = float(vol) if vol is not None and pd.notna(vol) else 0.0
                    width = int((vol_num / vol_max) * 100)
                    width = max(width, 1 if vol_num > 0 else 0)
                    lab_class = "ob-sell" if side == "sell" else "ob-buy"
                    side_txt = "卖" if side == "sell" else "买"
                    bar_class = "sell" if side == "sell" else "buy"
                    p_txt = f"{float(price):.2f}" if price is not None and pd.notna(price) else "--"
                    v_txt = f"{int(vol_num)}" if vol_num > 0 else "--"
                    rows_html += (
                        f'<div class="ob-row">'
                        f'<div class="ob-lab {lab_class}">{side_txt}{lvl}</div>'
                        f'<div class="ob-price {lab_class}">{p_txt}</div>'
                        f'<div class="ob-bar-wrap"><div class="ob-bar {bar_class}" style="width:{width}%"></div></div>'
                        f'<div class="ob-vol">{v_txt}</div>'
                        f"</div>"
                    )
                return rows_html

            html_text = (
                '<div class="ob-block">'
                + _ob_rows(sell_df, "sell")
                + '<div class="ob-sep"></div>'
                + _ob_rows(buy_df, "buy")
                + "</div>"
            )
            st.markdown(html_text, unsafe_allow_html=True)

    st.caption(panel.get("depth_note", ""))
    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.subheader(f"DeepSeek分析文档 · {selected_name} ({selected_code})")

    try:
        if run_analysis_now:
            done_job = _execute_analysis_job(job_id=live_job_id, mode="deep", ui_prefix=f"{selected_name} ")
            _render_final_report_block(live_job_id, done_job, f"inline_new_{selected_code}", height=520)
        else:
            live_job_obj = _load_json_file(_analysis_job_file(live_job_id))
            if isinstance(live_job_obj, dict) and live_job_obj.get("status") == "done":
                _render_final_report_block(live_job_id, live_job_obj, f"inline_saved_{selected_code}", height=520)
            elif isinstance(live_job_obj, dict) and live_job_obj.get("status") == "failed":
                st.error(f"上次分析失败: {live_job_obj.get('error', '未知错误')}")
            else:
                st.caption("点击上方“DeepSeek分析”开始生成文档。")
    except Exception as exc:
        st.error(f"分析失败: {type(exc).__name__}: {exc}")


def _render_fast_panel_fragment():
    selected_code = st.session_state.get("fast_selected_code", rows[0]["code"])
    selected_name = st.session_state.get("fast_selected_name", rows[0]["name"])
    market_open = _is_market_open(selected_code)
    cache_key = f"fast_panel_cache_{selected_code}"

    panel = None
    if market_open:
        panel = fetch_fast_panel(selected_code)
        st.session_state[cache_key] = panel
    else:
        panel = st.session_state.get(cache_key)
        if panel is None:
            # 闭市时允许抓取一次静态快照用于查看，但不进入自动刷新循环
            panel = fetch_fast_panel(selected_code)
            st.session_state[cache_key] = panel

    _render_fast_panel(selected_code, selected_name, panel=panel)

if auto_refresh_on and market_open_for_ctrl:
    @st.fragment(run_every=f"{int(auto_refresh_sec)}s")
    def _auto_fast_panel_fragment():
        _render_fast_panel_fragment()

    _auto_fast_panel_fragment()
else:
    _render_fast_panel_fragment()
