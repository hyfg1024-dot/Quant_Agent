import base64
import asyncio
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time as pytime
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.components.v1 import html

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

from fast_engine import _get_data_provider, fetch_fast_panel
from slow_engine import (
    add_stock_by_query,
    fetch_live_valuation_snapshot,
    get_stock_pool,
    get_latest_fundamental_snapshot,
    get_stock_group_map,
    init_db,
    remove_stock_from_pool,
    update_fundamental_data,
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FUNDAMENTAL_DIR = CURRENT_DIR.parent / "fundamental"
if str(FUNDAMENTAL_DIR) not in sys.path:
    sys.path.insert(0, str(FUNDAMENTAL_DIR))

from fundamental_engine import (
    APP_VERSION as FUND_APP_VERSION,
    analyze_watchlist as analyze_fundamental_watchlist,
    build_overview_table as build_fundamental_overview_table,
    format_pct as format_fundamental_pct,
)

FILTER_DIR = CURRENT_DIR.parent / "filter"
if str(FILTER_DIR) not in sys.path:
    sys.path.insert(0, str(FILTER_DIR))

from filter_engine import (
    APP_VERSION as FILTER_APP_VERSION,
    DISPLAY_COLUMNS as FILTER_DISPLAY_COLUMNS,
    apply_filters as filter_apply_filters,
    build_ai_quick_config as filter_build_ai_quick_config,
    check_market_data_dns as filter_check_market_data_dns,
    default_filter_config as filter_default_filter_config,
    export_snapshot_health_excel as filter_export_snapshot_health_excel,
    export_results_excel as filter_export_results_excel,
    get_baseline_build_progress as filter_get_baseline_build_progress,
    get_snapshot_health_report as filter_get_snapshot_health_report,
    get_snapshot_meta as filter_get_snapshot_meta,
    get_template_config as filter_get_template_config,
    get_weekly_update_status as filter_get_weekly_update_status,
    load_snapshot as filter_load_snapshot,
    load_templates as filter_load_templates,
    refresh_market_snapshot as filter_refresh_market_snapshot,
    run_baseline_build_once as filter_run_baseline_build_once,
    save_template as filter_save_template,
)
from shared.multi_agent_analyzer import MultiAgentAnalyzer
from shared.ui_shell import render_app_shell, render_section_intro, render_status_row, render_top_nav

st.set_page_config(page_title="Quant Dashboard", page_icon="📊", layout="wide")
APP_VERSION = "QDB-20260327-FLT5Y-01"
BACKTEST_APP_VERSION = "BT-20260411-01"
PAPER_APP_VERSION = "PT-20260411-01"
PORTFOLIO_APP_VERSION = "PF-20260416-01"
LOCAL_PREFS_PATH = "data/local_user_prefs.json"
ANALYSIS_CACHE_PATH = "data/deepseek_analysis_cache.json"
ANALYSIS_JOB_DIR = "data/analysis_jobs"
ANALYSIS_DELTA_CACHE_PATH = "data/deepseek_delta_cache.json"
ANALYSIS_COOLDOWN_PATH = "data/deepseek_cooldown.json"
DEEP_COOLDOWN_MINUTES = 5
ANALYSIS_ENGINE_VERSION = "multi_agent_v1"
BACKTEST_DIR = CURRENT_DIR.parent / "backtest"
BACKTEST_RUNNER = BACKTEST_DIR / "run_backtest.py"
BACKTEST_PAPER_RUNNER = BACKTEST_DIR / "paper_trade.py"
BACKTEST_STRATEGY_DIR = BACKTEST_DIR / "config" / "strategies"
BACKTEST_STRATEGY_TRASH_DIR = BACKTEST_STRATEGY_DIR / "_trash"
BACKTEST_REPORT_DIR = BACKTEST_DIR / "reports"
BACKTEST_PAPER_DIR = BACKTEST_DIR / "paper_trades"
BACKTEST_PAPER_ACTIVE = BACKTEST_PAPER_DIR / ".active"
BACKTEST_PAPER_DASHBOARD = BACKTEST_PAPER_DIR / "dashboard.html"
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

MULTI_AGENT_ANALYZER = MultiAgentAnalyzer()


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
    .engine-divider {
        margin: 2.4rem 0 2rem 0;
        border-top: 4px solid rgba(255, 255, 255, 0.14);
        position: relative;
    }
    .engine-divider span {
        position: relative;
        top: -1.45rem;
        background: rgba(13, 24, 38, 0.96);
        padding: 0 0.8rem;
        color: rgba(255, 248, 241, 0.98);
        font-weight: 800;
        font-size: 2.05rem;
        line-height: 1.1;
    }
    .section-title {
        color: rgba(255, 248, 241, 0.98);
        font-size: 2.05rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.9rem 0 0.8rem 0;
    }
    .analysis-time-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.72rem;
        border-radius: 10px;
        border: 1px solid rgba(252, 211, 77, 0.7);
        background: linear-gradient(135deg, rgba(253, 224, 71, 0.26) 0%, rgba(251, 191, 36, 0.22) 100%);
        color: #fff4bf;
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.08;
        margin: 0.12rem 0 0.55rem 0;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    }
    .fast-head-title {
        color: rgba(255, 248, 241, 0.98);
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
        background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 0.62rem 0.78rem;
        height: 156px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    .fast-card .t {
        color: rgba(232, 223, 210, 0.88);
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
        color: rgba(232, 223, 210, 0.88);
        font-weight: 600;
    }
    .fast-card .vv {
        color: rgba(255, 248, 241, 0.98);
        font-weight: 800;
        text-align: right;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.1px;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .fast-card .d {
        color: rgba(232, 223, 210, 0.88);
        font-size: 0.78rem;
        margin-top: 0.25rem;
    }
    .ob-title {
        font-size: 1.95rem;
        color: rgba(255, 248, 241, 0.98);
        font-weight: 800;
    }
    .panel-title {
        font-size: 2.7rem;
        color: rgba(255, 248, 241, 0.98);
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
        color: rgba(232, 223, 210, 0.88);
        margin-top: 0.12rem;
    }
    .fast-panels-gap {
        height: 0.75rem;
    }
    .subsection-divider {
        margin: 0.9rem 0 1.1rem 0;
        border-top: 3px solid rgba(255, 255, 255, 0.14);
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
        background: rgba(255, 255, 255, 0.08);
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
        color: rgba(255, 248, 241, 0.98);
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.2px;
    }
    .ob-sell { color: #8fe3c3; }
    .ob-buy { color: #ff9f8e; }
    .ob-sep {
        border-top: 1px solid rgba(255, 255, 255, 0.12);
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
        color: rgba(232, 223, 210, 0.78) !important;
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
        color: rgba(255, 248, 241, 0.98) !important;
        border: none !important;
        box-shadow: none !important;
    }
    .watch-split-divider {
        min-height: 0;
        border-left: 2px solid rgba(255, 255, 255, 0.14);
        margin: 0.2rem auto 0 auto;
        width: 1px;
    }
    .group-title {
        color: rgba(255, 248, 241, 0.98);
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
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
        min-height: 112px;
    }
    .score-panel .label {
        color: rgba(232, 223, 210, 0.88);
        font-size: 1.02rem;
        font-weight: 700;
    }
    .score-panel .value {
        color: rgba(255, 248, 241, 0.98);
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.25;
        margin-top: 8px;
    }
    .fnd-focus-title {
        margin: 0 0 1.35rem 0;
        color: rgba(255, 248, 241, 0.98);
        font-size: 2.35rem;
        line-height: 1.1;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-weight: 700;
    }
    .fnd-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 12px 14px;
        min-height: 208px;
        background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
        display: flex;
        flex-direction: column;
    }
    .fnd-card h4 {
        margin: 0 0 6px 0;
        font-size: 1.45rem;
        color: rgba(255, 248, 241, 0.98);
    }
    .fnd-card .score {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 2px 0 6px 0;
        color: rgba(255, 248, 241, 0.98);
    }
    .fnd-card .desc {
        color: rgba(232, 223, 210, 0.88);
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
    .fnd-overview-head {
        color: rgba(230, 221, 208, 0.72);
        font-size: 1rem;
        font-weight: 700;
        padding: 0 0 0.45rem 0;
    }
    .fnd-overview-cell {
        color: rgba(246, 239, 229, 0.94);
        font-size: 1rem;
        line-height: 1.2;
        padding-top: 0.45rem;
    }
    .fnd-overview-cell.is-muted {
        color: rgba(222, 214, 202, 0.78);
    }
    .fnd-overview-row-divider {
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin: 0;
    }
    [class*="st-key-tr_fnd_name_wrap_"] {
        padding-top: 0.35rem;
    }
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton {
        margin: 0 !important;
    }
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button,
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button:hover,
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button:focus,
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button:active {
        min-height: 1.9rem !important;
        height: 1.9rem !important;
        padding: 0 1rem !important;
        border-radius: 999px !important;
        border: 1px solid rgba(196, 235, 232, 0.72) !important;
        background: linear-gradient(180deg, rgba(139, 198, 194, 0.98), rgba(101, 160, 157, 1)) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 18px rgba(0,0,0,0.10) !important;
        justify-content: center !important;
        color: #122033 !important;
        -webkit-text-fill-color: #122033 !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
    }
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button span,
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button p,
    [class*="st-key-tr_fnd_name_wrap_"] div.stButton > button div {
        color: #122033 !important;
        -webkit-text-fill-color: #122033 !important;
    }
    [class*="st-key-tr_fnd_name_wrap_active_"] div.stButton > button {
        border-color: rgba(232, 246, 244, 0.92) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.14), 0 0 0 1px rgba(255,255,255,0.10), 0 10px 20px rgba(0,0,0,0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "trading"
if "flt_cfg" not in st.session_state:
    st.session_state["flt_cfg"] = filter_default_filter_config()
if "flt_result" not in st.session_state:
    st.session_state["flt_result"] = None
if "flt_show_ops_panel" not in st.session_state:
    st.session_state["flt_show_ops_panel"] = False

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
    key="deepseek_user_input",
)
analysis_api_key_input = st.sidebar.text_input(
    "API Key（可留空，读取环境变量）",
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

_allowed_pages = {"trading", "fundamental", "filter", "backtest", "paper", "portfolio"}
try:
    _qp_page = str(st.query_params.get("page", "")).strip().lower()
except Exception:
    _qp_page = ""
if _qp_page not in _allowed_pages:
    _qp_page = "trading"
if st.session_state.get("active_page") not in _allowed_pages:
    st.session_state["active_page"] = _qp_page

_active_page = st.session_state.get("active_page", "trading")
_nav_selected = render_top_nav(_active_page)
if _nav_selected != _active_page:
    st.session_state["active_page"] = _nav_selected
    try:
        st.query_params["page"] = _nav_selected
    except Exception:
        pass
    st.rerun()
_active_page = st.session_state.get("active_page", "trading")
try:
    if str(st.query_params.get("page", "")).strip().lower() != _active_page:
        st.query_params["page"] = _active_page
except Exception:
    pass
_group_map = get_stock_group_map()
_holding_count = sum(1 for code, _name in pool_rows_for_sidebar if _group_map.get(code) == "holding")
_watch_count = sum(1 for code, _name in pool_rows_for_sidebar if _group_map.get(code) != "holding")
render_app_shell(
    _active_page,
    version={
        "fundamental": FUND_APP_VERSION,
        "trading": APP_VERSION,
        "filter": FILTER_APP_VERSION,
        "backtest": BACKTEST_APP_VERSION,
        "paper": PAPER_APP_VERSION,
        "portfolio": PORTFOLIO_APP_VERSION,
    }.get(_active_page, APP_VERSION),
    badges={
        "fundamental": ("八维评分", "观察名单", "结构化结论"),
        "trading": ("实时盘口", "分时结构", "DeepSeek 分析"),
        "filter": ("两段筛选", "快照更新", "结果导出"),
        "backtest": ("多空回测", "成本建模", "HTML报告"),
        "paper": ("策略入口", "逐日更新", "模拟看板"),
        "portfolio": ("仓位管理", "浮动盈亏", "ATR风控"),
    }.get(_active_page, ("实时盘口", "分时结构", "DeepSeek 分析")),
    metrics={
        "fundamental": (
            ("当前关注", f"{len(pool_rows_for_sidebar)} 只"),
            ("研究视角", "评分 + 文本 + AI"),
            ("池子结构", f"持仓 {_holding_count} / 观察 {_watch_count}"),
        ),
        "trading": (
            ("当前关注", f"{len(pool_rows_for_sidebar)} 只"),
            ("池子结构", f"持仓 {_holding_count} / 观察 {_watch_count}"),
            ("工作流", "盘口 -> 研判 -> 决策"),
        ),
        "filter": (
            ("股票池联动", f"{len(pool_rows_for_sidebar)} 只"),
            ("筛选方式", "两段排雷"),
            ("工作流", "快照 -> 条件 -> 导出"),
        ),
        "backtest": (
            ("策略配置", "YAML可编辑"),
            ("回测区间", "2021-01-01 -> today"),
            ("工作流", "更新数据 -> 执行 -> 复盘"),
        ),
        "paper": (
            ("执行模式", "逐日模拟"),
            ("交易日口径", "港股实际交易日"),
            ("工作流", "启动 -> 更新 -> 看板"),
        ),
        "portfolio": (
            ("执行模式", "真实仓位"),
            ("风险口径", "单笔风险 1%"),
            ("工作流", "录入 -> 监控 -> 调整"),
        ),
    }.get(
        _active_page,
        (
            ("当前关注", f"{len(pool_rows_for_sidebar)} 只"),
            ("池子结构", f"持仓 {_holding_count} / 观察 {_watch_count}"),
            ("工作流", "盘口 -> 研判 -> 决策"),
        ),
    ),
)

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


def _call_multi_agent_analysis(json_text: str) -> tuple[str, dict, float, float]:
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

    result = _run_async_blocking(MULTI_AGENT_ANALYZER.analyze(user_content=json_text, llm_call=_llm_call))
    report = str(result.get("final_text", "") or "").strip()
    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
    if isinstance(usage, dict):
        usage["_multi_agent_meta"] = result.get("meta", {}) if isinstance(result.get("meta"), dict) else {}
    total_cost = float(result.get("total_cost", 0.0) or 0.0)
    elapsed = float(result.get("elapsed", 0.0) or 0.0)
    return report, usage, total_cost, elapsed


def _call_deepseek_analysis(json_text: str) -> tuple[str, dict, float, float]:
    return _call_multi_agent_analysis(json_text=json_text)


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

    row = next(
        (
            x
            for x in rows_fnd
            if str(x.get("code", "")) == str(st.session_state.get("fnd_selected_code", ""))
        ),
        rows_fnd[0],
    )
    render_section_intro(
        "研究名单",
        "保持列表总览和评分板之间的切换距离足够短，让你能先扫一遍股票池，再快速下钻到单只标的。",
        kicker="Overview",
        pills=("股票池总览", "打开评分板", "支持快速切换"),
    )
    render_status_row(
        (
            ("名单规模", f"{len(rows_fnd)} 只"),
            ("当前标的", f"{row.get('name', '')} ({row.get('code', '')})"),
            ("当前结论", _clean_text_no_na(str(row.get("conclusion", "观察")))),
        )
    )
    st.subheader("股票列表")
    df = build_fundamental_overview_table(rows_fnd).copy()
    widths = [1.0, 1.15, 0.75, 0.75, 0.85, 1.35]
    header_cols = st.columns(widths, gap="small")
    for col, header in zip(header_cols, df.columns):
        col.markdown(f"<div class='fnd-overview-head'>{header}</div>", unsafe_allow_html=True)
    st.markdown("<div class='fnd-overview-row-divider'></div>", unsafe_allow_html=True)

    selected_code = str(st.session_state.get("fnd_selected_code", rows_fnd[0].get("code", "") if rows_fnd else ""))
    for row in rows_fnd:
        cols = st.columns(widths, gap="small")
        cols[0].markdown(f"<div class='fnd-overview-cell'>{row.get('code', '')}</div>", unsafe_allow_html=True)
        with cols[1]:
            wrap_key = (
                f"tr_fnd_name_wrap_active_{row.get('code', '')}"
                if str(row.get("code", "")) == selected_code
                else f"tr_fnd_name_wrap_{row.get('code', '')}"
            )
            with st.container(key=wrap_key):
                if st.button(str(row.get("name", "")), key=f"tr_fnd_name_pick_{row.get('code', '')}", type="tertiary"):
                    st.session_state["fnd_selected_code"] = row.get("code", "")
                    st.rerun()
        cols[2].markdown(f"<div class='fnd-overview-cell'>{row.get('total_score', '')}</div>", unsafe_allow_html=True)
        cols[3].markdown(f"<div class='fnd-overview-cell'>{row.get('type', '')}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div class='fnd-overview-cell'>{format_fundamental_pct(row.get('dividend_yield'))}</div>", unsafe_allow_html=True)
        cols[5].markdown(f"<div class='fnd-overview-cell is-muted'>{row.get('analysis_at', '')}</div>", unsafe_allow_html=True)
        st.markdown("<div class='fnd-overview-row-divider'></div>", unsafe_allow_html=True)

    st.divider()
    row = next((x for x in rows_fnd if str(x.get("code", "")) == str(st.session_state.get("fnd_selected_code", ""))), rows_fnd[0])
    st.markdown(
        f"<div class='fnd-focus-title'>{row.get('name', '')}（{row.get('code', '')}）</div>",
        unsafe_allow_html=True,
    )
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
        render_section_intro(
            "八维拆解",
            "把各维度拆成独立卡片，便于你快速发现强项、短板和需要二次验证的地方。",
            kicker="Dimensions",
            pills=("八维卡片", "便于横向比较", "统一句长"),
        )
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
    render_section_intro(
        "总结与输出",
        "把总结、复制 JSON 和 DeepSeek 深挖放在一起，形成研究闭环。",
        kicker="Narrative",
        pills=("总结文本", "复制 JSON", "DeepSeek 深挖"),
    )

    news_catalysts = _format_rag_text_block(str(row.get("news_catalysts", "") or ""))
    research_summary = _format_rag_text_block(str(row.get("research_summary", "") or ""))
    st.subheader("新闻催化与研报摘要")
    ns1, ns2 = st.columns(2, gap="large")
    with ns1:
        st.markdown("**最新新闻催化**")
        if news_catalysts:
            st.text_area(
                "news_catalysts",
                value=news_catalysts,
                height=220,
                key=f"tr_fnd_news_catalysts_{row.get('code', '')}",
            )
        else:
            st.info("当前标的暂无 news_catalysts（请点侧栏“刷新全部”后再看）")
    with ns2:
        st.markdown("**机构研报摘要**")
        if research_summary:
            st.text_area(
                "research_summary",
                value=research_summary,
                height=220,
                key=f"tr_fnd_research_summary_{row.get('code', '')}",
            )
        else:
            st.info("当前标的暂无 research_summary（请点侧栏“刷新全部”后再看）")

    with st.expander("查看动态 Prompt（含 news_catalysts/research_summary）", expanded=False):
        dyn_prompt = _format_prompt_text_block(str(row.get("fund_deepseek_prompt", "") or ""))
        if dyn_prompt:
            st.text_area(
                "fund_deepseek_prompt",
                value=dyn_prompt,
                height=280,
                key=f"tr_fnd_dyn_prompt_{row.get('code', '')}",
            )
        else:
            st.caption("当前缓存行未生成动态 Prompt。")

    st.divider()
    st.subheader("总结性文本")
    lines = _split_sentences(str(row.get("summary_text", "")))
    if lines:
        st.markdown(
            "<div style='line-height:1.8;color:rgba(242,235,225,0.94);font-size:1.05rem;'>"
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
                style="width:100%;height:42px;border-radius:999px;border:1px solid rgba(255,255,255,0.14);background:linear-gradient(135deg,#d9ece7 0%,#76b6b5 100%);color:#11202f;font-size:1rem;font-weight:800;cursor:pointer;box-shadow:0 12px 24px rgba(0,0,0,0.18);">
                复制JSON
              </button>
              <div id="fnd-copy-msg-{code}" style="margin-top:0.35rem;color:rgba(239,229,216,0.82);font-size:0.86rem;"></div>
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


def _has_rearview_enabled(cfg: dict) -> bool:
    d5 = cfg.get("rearview_5y", {}) if isinstance(cfg, dict) else {}
    return any(bool(v) for k, v in d5.items() if str(k).endswith("_enabled"))


def _build_stage1_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    d5 = out.get("rearview_5y", {})
    for k in list(d5.keys()):
        if str(k).endswith("_enabled"):
            d5[k] = False
    return out


def _build_stage2_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    risk = out.get("risk", {})
    for key in [
        "exclude_st",
        "exclude_investigation",
        "exclude_penalty",
        "exclude_fund_occupation",
        "exclude_illegal_reduce",
        "require_standard_audit",
        "exclude_sunset_industry",
        "exclude_no_dividend_5y",
        "pledge_ratio_max_enabled",
        "audit_change_max_enabled",
    ]:
        if key in risk:
            risk[key] = False

    for group in ["quality", "valuation", "growth_liquidity"]:
        g = out.get(group, {})
        for k in list(g.keys()):
            if str(k).endswith("_enabled"):
                g[k] = False
    return out


def _concat_dedup(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1 is None or df1.empty:
        merged = df2.copy() if isinstance(df2, pd.DataFrame) else pd.DataFrame()
    elif df2 is None or df2.empty:
        merged = df1.copy()
    else:
        merged = pd.concat([df1, df2], ignore_index=True)
    if isinstance(merged, pd.DataFrame) and (not merged.empty) and ("code" in merged.columns):
        merged = merged.drop_duplicates(subset=["code"], keep="first")
    return merged


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


def _render_filter_ops_panel() -> None:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], vertical_alignment="bottom")
    market_scope = c1.selectbox(
        "市场范围",
        options=["A", "HK", "AH"],
        index=2,
        format_func=lambda x: {"A": "A股", "HK": "港股", "AH": "A+H"}[x],
        key="flt_ops_market_scope",
    )
    max_stocks = c2.number_input("更新股票数（0=全部）", min_value=0, max_value=12000, value=0, step=200, key="flt_ops_max_stocks")
    enrich_n = c3.number_input("深度补充（A股）", min_value=0, max_value=2000, value=800, step=100, key="flt_ops_enrich_n")
    safe_mode = c4.checkbox("安全模式（防封）", value=True, key="flt_ops_safe_mode")
    rotate_enrich = st.checkbox("深度补充采用轮转增量（推荐）", value=True, key="flt_ops_rotate")
    force_refresh = st.checkbox("忽略缓存强制重抓", value=False, key="flt_ops_force")
    bootstrap_batch = st.number_input(
        "首轮建库每次推进数量（A股，断点续跑）",
        min_value=100,
        max_value=2000,
        value=800,
        step=100,
        key="flt_ops_bootstrap_batch",
    )

    weekly = filter_get_weekly_update_status("AH")
    weekly_state = "已到期可执行" if bool(weekly.get("due")) else f"未到期（剩余{float(weekly.get('remaining_hours', 0.0)):.1f}小时）"
    st.caption(
        f"周更(A+H)上次: {weekly.get('last') or '--'} ｜ 下次到期: {weekly.get('next_due') or '立即可执行'} ｜ 当前状态: {weekly_state}"
    )

    u1, u2, u3, u4 = st.columns(4)
    if u1.button("运行一次更新（运维台）", use_container_width=True, key="flt_ops_run_once"):
        with st.spinner("正在执行运维更新..."):
            try:
                stats = filter_refresh_market_snapshot(
                    max_stocks=int(max_stocks),
                    enrich_top_n=int(enrich_n),
                    force_refresh=bool(force_refresh),
                    rotate_enrich=bool(rotate_enrich),
                    market_scope=str(market_scope),
                    weekly_mode=False,
                    safe_mode=bool(safe_mode),
                )
                if bool(stats.get("fallback", False)):
                    st.warning("本次未连通接口，已回退本地快照。")
                else:
                    st.success(
                        f"更新完成：{stats.get('row_count', 0)} 只，深补 {stats.get('enriched_count', 0)} 只（区间 {int(stats.get('enrich_start', 0) or 0)} -> {int(stats.get('enrich_end', 0) or 0)}）"
                    )
                    st.caption(f"缓存命中: {int(stats.get('cache_hit', 0) or 0)} ｜ 重抓: {int(stats.get('cache_miss', 0) or 0)}")
                st.session_state["flt_result"] = None
            except Exception as exc:
                st.error(f"更新失败: {exc}")

    if u2.button("执行周更（A+H，7天一次）", use_container_width=True, key="flt_ops_weekly"):
        with st.spinner("正在执行周更..."):
            try:
                stats = filter_refresh_market_snapshot(
                    max_stocks=0,
                    enrich_top_n=int(enrich_n),
                    force_refresh=False,
                    rotate_enrich=True,
                    market_scope="AH",
                    weekly_mode=True,
                    safe_mode=True,
                )
                if bool(stats.get("skipped", False)):
                    st.info(_safe_str(stats.get("reason", "周更间隔未到，本次跳过")))
                elif bool(stats.get("fallback", False)):
                    st.warning("周更回退到本地快照。")
                else:
                    st.success(
                        f"周更完成：{stats.get('row_count', 0)} 只，深补 {stats.get('enriched_count', 0)} 只（区间 {int(stats.get('enrich_start', 0) or 0)} -> {int(stats.get('enrich_end', 0) or 0)}）"
                    )
                st.session_state["flt_result"] = None
            except Exception as exc:
                st.error(f"周更失败: {exc}")

    if u3.button("首轮建库推进（断点续跑）", use_container_width=True, key="flt_ops_bootstrap_once"):
        with st.spinner("正在推进首轮建库..."):
            try:
                result = filter_run_baseline_build_once(
                    market_scope="AH",
                    enrich_batch=int(bootstrap_batch),
                    safe_mode=bool(safe_mode),
                    force_refresh=False,
                )
                if bool(result.get("skipped", False)):
                    st.error(_safe_str(result.get("reason", "本次未执行首轮建库推进。")))
                    dns = result.get("dns", {}) if isinstance(result, dict) else {}
                    fail_hosts = dns.get("fail_hosts", []) if isinstance(dns, dict) else []
                    if fail_hosts:
                        st.caption(f"DNS失败主机: {', '.join([_safe_str(x) for x in fail_hosts])}")
                elif bool(result.get("fallback", False)):
                    st.warning("本次推进发生回退，未完成有效深补。请先处理网络后重试。")
                else:
                    after = result.get("progress_after", {}) if isinstance(result, dict) else {}
                    st.success(
                        f"推进完成：新增深补 {int(result.get('added', 0) or 0)} 只，"
                        f"累计 {int(after.get('enriched_done', 0) or 0)}/{int(after.get('eligible_total', 0) or 0)}"
                    )
                    if bool(result.get("completed", False)):
                        st.success("首轮建库已完成。")
                st.session_state["flt_result"] = None
            except Exception as exc:
                st.error(f"首轮建库推进失败: {exc}")

    if u4.button("检测数据源DNS", use_container_width=True, key="flt_ops_dns_check"):
        dns = filter_check_market_data_dns()
        if bool(dns.get("ok")):
            st.success(f"DNS可用：{', '.join([_safe_str(x) for x in dns.get('ok_hosts', [])])}")
        else:
            st.error("DNS不可用：当前无法连通东财行情主机。")
            fail_hosts = dns.get("fail_hosts", []) if isinstance(dns, dict) else []
            if fail_hosts:
                st.caption(f"失败主机: {', '.join([_safe_str(x) for x in fail_hosts])}")

    build_prog = filter_get_baseline_build_progress("AH")
    st.caption(
        f"A股首轮深补进度: {int(build_prog.get('enriched_done', 0) or 0)} / "
        f"{int(build_prog.get('eligible_total', 0) or 0)} ｜ "
        f"剩余 {int(build_prog.get('remaining', 0) or 0)}"
    )
    st.progress(float(build_prog.get("progress_ratio", 0.0) or 0.0), text=f"首轮建库进度 {float(build_prog.get('progress_pct', 0.0) or 0.0):.1f}%")

    report = filter_get_snapshot_health_report(days=7, top_n=20)
    qc = report.get("quality_counts", {})
    total = int(report.get("total", 0) or 0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总样本", total)
    m2.metric("full", int(qc.get("full", 0) or 0))
    m3.metric("partial", int(qc.get("partial", 0) or 0))
    m4.metric("missing", int(qc.get("missing", 0) or 0))
    coverage_ratio = float(report.get("coverage_ratio", 0.0) or 0.0)
    st.progress(coverage_ratio, text=f"覆盖率 {coverage_ratio * 100:.1f}%")

    diag_xlsx = filter_export_snapshot_health_excel(days=30, top_n=50)
    st.download_button(
        "导出体检报告（Excel）",
        data=diag_xlsx,
        file_name=f"filter_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="flt_ops_export_health",
    )

    with st.expander("查看体检详情", expanded=False):
        st.dataframe(report.get("trend_df", pd.DataFrame()), use_container_width=True, hide_index=True)
        st.dataframe(report.get("runs_df", pd.DataFrame()), use_container_width=True, hide_index=True)


def _render_filter_page():
    st.caption(f"版本号: {FILTER_APP_VERSION} ｜ 侧边栏统一，筛选参数在主区顶部")

    cfg = st.session_state.get("flt_cfg", filter_default_filter_config())
    meta = filter_get_snapshot_meta()
    render_section_intro(
        "数据运维台",
        "统一处理更新、周更与体检。点击下方按钮即可展开运维区；不展开时保持页面简洁。",
        kicker="Workflow",
        pills=("A+H更新", "周更", "体检", "导出"),
    )
    st.markdown(
        """
        <style>
        .st-key-flt_ops_toggle_in_filter_header button {
          font-size: 1.5rem !important;
          min-height: 3.2rem !important;
          padding: 0.55rem 1.6rem !important;
          white-space: nowrap !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    toggle_label = "打开数据运维台" if not bool(st.session_state.get("flt_show_ops_panel", False)) else "收起数据运维台"
    if st.button(toggle_label, use_container_width=False, key="flt_ops_toggle_in_filter_header"):
        st.session_state["flt_show_ops_panel"] = not bool(st.session_state.get("flt_show_ops_panel", False))
        st.rerun()
    render_status_row(
        (
            ("快照状态", meta.get("last_update", "尚未更新") if meta else "尚未更新"),
            ("样本数量", f"{int(meta.get('row_count', 0) or 0)} 只" if meta else "0 只"),
            ("深度补充", f"{int(meta.get('enriched_count', 0) or 0)} 只" if meta else "0 只"),
        )
    )
    if bool(st.session_state.get("flt_show_ops_panel", False)):
        _render_filter_ops_panel()
        st.markdown("---")

    if bool(st.session_state.get("flt_show_ops_panel", False)):
        st.subheader("过滤器参数")
        top_cols = st.columns([1.1, 1.1, 0.8, 1.0], vertical_alignment="bottom")
        max_stocks = top_cols[0].number_input("本次更新股票数（0=全部）", min_value=0, max_value=6000, value=2000, step=100, key="flt_max_stocks_main")
        enrich_n = top_cols[1].number_input("深度补充数量（调用基本面引擎）", min_value=0, max_value=2000, value=300, step=50, key="flt_enrich_n_main")
        force_refresh = top_cols[2].checkbox("忽略缓存强制重抓", value=False, key="flt_force_refresh_main")
        if top_cols[3].button("更新全市场数据", use_container_width=True, key="flt_refresh_market_main"):
            with st.spinner("正在更新市场数据，请稍候..."):
                try:
                    stats = filter_refresh_market_snapshot(
                        max_stocks=int(max_stocks),
                        enrich_top_n=int(enrich_n),
                        force_refresh=bool(force_refresh),
                    )
                    if bool(stats.get("fallback", False)):
                        st.warning(
                            "本次未连通东财接口，已回退为本地快照（未覆盖旧数据）。"
                            "请检查系统代理/VPN后再重试。"
                        )
                    else:
                        st.success(f"更新完成：{stats['row_count']} 只，深度补充 {stats['enriched_count']} 只")
                    st.session_state["flt_result"] = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"更新失败: {exc}")

        if meta:
            st.caption(
                f"最近更新: {meta.get('last_update', '--')} ｜ "
                f"样本数量: {meta.get('row_count', '--')} ｜ "
                f"深度补充: {meta.get('enriched_count', '--')}"
            )

        tpl_cols = st.columns([1.1, 0.65, 1.0, 0.65], vertical_alignment="bottom")
        all_tpl = filter_load_templates()
        tpl_names = sorted(all_tpl.keys())
        selected_tpl = tpl_cols[0].selectbox("模板", options=["(无)"] + tpl_names, key="flt_tpl_select_main")
        if tpl_cols[1].button("读取模板", use_container_width=True, key="flt_tpl_load_main"):
            if selected_tpl and selected_tpl != "(无)":
                st.session_state["flt_cfg"] = filter_get_template_config(selected_tpl)
                st.success(f"已加载模板: {selected_tpl}")
                st.rerun()

        save_tpl_name = tpl_cols[2].text_input("保存为模板名", value="", key="flt_tpl_save_main")
        if tpl_cols[3].button("保存模板", use_container_width=True, key="flt_tpl_save_btn_main"):
            try:
                filter_save_template(save_tpl_name, cfg)
                st.success("模板已保存")
            except Exception as exc:
                st.error(str(exc))

    render_section_intro(
        "条件矩阵",
        "筛选条件被拆成四层，从硬排除到五年后视镜，帮助你先做风险清洗，再做估值与长期验证。",
        kicker="Configuration",
        pills=("A 硬排除", "B 估值质量", "C 行业规模", "D 五年后视镜"),
    )
    st.markdown(
        """
        <style>
        [data-testid="stExpander"] details > summary {
          background: linear-gradient(180deg, rgba(142, 168, 107, 0.38), rgba(116, 146, 87, 0.34)) !important;
          border: 1px solid rgba(191, 217, 157, 0.30) !important;
          border-radius: 999px !important;
        }
        [data-testid="stExpander"] details > summary:hover {
          background: linear-gradient(180deg, rgba(156, 182, 119, 0.46), rgba(126, 156, 95, 0.42)) !important;
        }
        [data-testid="stExpander"] details > summary p,
        [data-testid="stExpander"] details > summary span,
        [data-testid="stExpander"] details > summary div,
        [data-testid="stExpander"] details > summary svg {
          color: rgba(250, 248, 241, 0.98) !important;
          fill: rgba(250, 248, 241, 0.98) !important;
          font-weight: 800 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    mode = st.radio("筛选模式", options=["手动筛选", "AI辅助设定"], horizontal=True, key="flt_mode_main")
    if mode == "AI辅助设定":
        render_section_intro(
            "AI 条件草拟",
            "先用自然语言生成一版初稿条件，再继续手动微调，适合从模糊目标快速进入筛选状态。",
            kicker="Assist",
            pills=("自然语言", "自动生成", "可继续微调"),
        )
        prompt = st.text_area("输入你的目标", value="高股息、低估值、低负债、防御型", height=80, key="flt_ai_prompt_main")
        if st.button("生成条件并应用", type="primary", key="flt_ai_apply_main"):
            cfg = filter_build_ai_quick_config(prompt, cfg)
            st.session_state["flt_cfg"] = cfg
            st.success("已根据描述生成条件，你可继续微调后执行筛选。")
            st.rerun()
    st.subheader("筛选条件（支持手动开关，像电商筛选一样）")

    with st.expander("A. 财务健康度与硬排除", expanded=True):
        c1, c2, c3 = st.columns(3)
        cfg["missing_policy"] = c1.selectbox("缺失数据处理", options=["ignore", "exclude"], index=0 if cfg.get("missing_policy") == "ignore" else 1)

        r = cfg["risk"]
        q = cfg["quality"]

        r["exclude_st"] = c1.checkbox("排除 ST/*ST", value=bool(r.get("exclude_st", True)))
        r["exclude_investigation"] = c1.checkbox("排除立案调查", value=bool(r.get("exclude_investigation", True)))
        r["exclude_penalty"] = c1.checkbox("排除重大处罚", value=bool(r.get("exclude_penalty", True)))

        r["exclude_fund_occupation"] = c2.checkbox("排除资金占用", value=bool(r.get("exclude_fund_occupation", True)))
        r["exclude_illegal_reduce"] = c2.checkbox("排除违规减持", value=bool(r.get("exclude_illegal_reduce", True)))
        r["require_standard_audit"] = c2.checkbox("审计意见必须标准无保留", value=bool(r.get("require_standard_audit", False)))

        q["ocf_3y_min_enabled"] = c3.checkbox("启用近3年经营现金流下限(亿)", value=bool(q.get("ocf_3y_min_enabled", False)))
        q["ocf_3y_min"] = c3.number_input("经营现金流下限(亿)", value=float(q.get("ocf_3y_min", 0.0)), step=1.0)
        q["asset_liability_max_enabled"] = c3.checkbox("启用资产负债率上限(%)", value=bool(q.get("asset_liability_max_enabled", False)))
        q["asset_liability_max"] = c3.number_input("资产负债率上限(%)", value=float(q.get("asset_liability_max", 80.0)), step=1.0)

    with st.expander("B. 估值与质量", expanded=True):
        c1, c2, c3 = st.columns(3)
        q = cfg["quality"]
        v = cfg["valuation"]

        q["roe_min_enabled"] = c1.checkbox("启用 ROE 下限(%)", value=bool(q.get("roe_min_enabled", False)))
        q["roe_min"] = c1.number_input("ROE 下限(%)", value=float(q.get("roe_min", 5.0)), step=0.5)
        q["gross_margin_min_enabled"] = c1.checkbox("启用毛利率下限(%)", value=bool(q.get("gross_margin_min_enabled", False)))
        q["gross_margin_min"] = c1.number_input("毛利率下限(%)", value=float(q.get("gross_margin_min", 20.0)), step=0.5)
        q["net_margin_min_enabled"] = c1.checkbox("启用净利率下限(%)", value=bool(q.get("net_margin_min_enabled", False)))
        q["net_margin_min"] = c1.number_input("净利率下限(%)", value=float(q.get("net_margin_min", 8.0)), step=0.5)

        q["receivable_ratio_max_enabled"] = c2.checkbox("启用应收代理指标上限", value=bool(q.get("receivable_ratio_max_enabled", False)))
        q["receivable_ratio_max"] = c2.number_input("应收代理指标上限", value=float(q.get("receivable_ratio_max", 50.0)), step=1.0)
        q["goodwill_ratio_max_enabled"] = c2.checkbox("启用商誉/净资产上限(%)", value=bool(q.get("goodwill_ratio_max_enabled", False)))
        q["goodwill_ratio_max"] = c2.number_input("商誉/净资产上限(%)", value=float(q.get("goodwill_ratio_max", 30.0)), step=1.0)
        q["interest_debt_asset_max_enabled"] = c2.checkbox("启用有息负债/总资产上限(%)", value=bool(q.get("interest_debt_asset_max_enabled", False)))
        q["interest_debt_asset_max"] = c2.number_input("有息负债/总资产上限(%)", value=float(q.get("interest_debt_asset_max", 20.0)), step=1.0)

        v["pe_ttm_min_enabled"] = c3.checkbox("启用 PE(TTM) 下限", value=bool(v.get("pe_ttm_min_enabled", False)))
        v["pe_ttm_min"] = c3.number_input("PE(TTM) 下限", value=float(v.get("pe_ttm_min", 0.0)), step=1.0)
        v["pe_ttm_max_enabled"] = c3.checkbox("启用 PE(TTM) 上限", value=bool(v.get("pe_ttm_max_enabled", False)))
        v["pe_ttm_max"] = c3.number_input("PE(TTM) 上限", value=float(v.get("pe_ttm_max", 25.0)), step=1.0)
        v["pb_max_enabled"] = c3.checkbox("启用 PB 上限", value=bool(v.get("pb_max_enabled", False)))
        v["pb_max"] = c3.number_input("PB 上限", value=float(v.get("pb_max", 3.0)), step=0.1)

    with st.expander("C. 行业、分红、流动性与规模", expanded=True):
        c1, c2, c3 = st.columns(3)
        r = cfg["risk"]
        v = cfg["valuation"]
        g = cfg["growth_liquidity"]

        scope_map = {"all": "全部市场", "A": "仅A股", "HK": "仅港股"}
        raw_scope = _safe_str(r.get("market_scope", "all")).upper()
        scope_value = "all" if raw_scope not in {"A", "HK"} else raw_scope
        r["market_scope"] = c1.selectbox(
            "筛选市场范围",
            options=["all", "A", "HK"],
            index=["all", "A", "HK"].index(scope_value),
            format_func=lambda x: scope_map.get(x, x),
        )
        r["industry_include_enabled"] = c1.checkbox("启用行业关键词包含", value=bool(r.get("industry_include_enabled", False)))
        r["industry_include_keywords"] = c1.text_input(
            "行业关键词（包含，逗号分隔）",
            value=str(r.get("industry_include_keywords", "")),
        )
        if str(r.get("industry_include_keywords", "")).strip():
            r["industry_include_enabled"] = True
        r["exclude_sunset_industry"] = c1.checkbox("排除夕阳行业", value=bool(r.get("exclude_sunset_industry", False)))
        r["sunset_industries"] = c1.text_area("夕阳行业关键词（逗号分隔）", value=str(r.get("sunset_industries", "")), height=120)
        r["pledge_ratio_max_enabled"] = c1.checkbox("启用质押率上限(%)", value=bool(r.get("pledge_ratio_max_enabled", False)))
        r["pledge_ratio_max"] = c1.number_input("质押率上限(%)", value=float(r.get("pledge_ratio_max", 80.0)), step=1.0)

        v["dividend_min_enabled"] = c2.checkbox("启用股息率下限(%)", value=bool(v.get("dividend_min_enabled", False)))
        v["dividend_min"] = c2.number_input("股息率下限(%)", value=float(v.get("dividend_min", 3.0)), step=0.1)
        v["dividend_max_enabled"] = c2.checkbox("启用股息率上限(%)", value=bool(v.get("dividend_max_enabled", False)))
        v["dividend_max"] = c2.number_input("股息率上限(%)", value=float(v.get("dividend_max", 12.0)), step=0.1)
        r["exclude_no_dividend_5y"] = c2.checkbox("排除近5年未分红", value=bool(r.get("exclude_no_dividend_5y", False)))

        g["market_cap_min_enabled"] = c3.checkbox("启用总市值下限(亿)", value=bool(g.get("market_cap_min_enabled", False)))
        g["market_cap_min"] = c3.number_input("总市值下限(亿)", value=float(g.get("market_cap_min", 100.0)), step=10.0)
        g["market_cap_max_enabled"] = c3.checkbox("启用总市值上限(亿)", value=bool(g.get("market_cap_max_enabled", False)))
        g["market_cap_max"] = c3.number_input("总市值上限(亿)", value=float(g.get("market_cap_max", 5000.0)), step=10.0)

        g["turnover_min_enabled"] = c3.checkbox("启用换手率下限(%)", value=bool(g.get("turnover_min_enabled", False)))
        g["turnover_min"] = c3.number_input("换手率下限(%)", value=float(g.get("turnover_min", 0.2)), step=0.1)
        g["turnover_max_enabled"] = c3.checkbox("启用换手率上限(%)", value=bool(g.get("turnover_max_enabled", False)))
        g["turnover_max"] = c3.number_input("换手率上限(%)", value=float(g.get("turnover_max", 15.0)), step=0.1)

        g["volume_ratio_min_enabled"] = c3.checkbox("启用量比下限", value=bool(g.get("volume_ratio_min_enabled", False)))
        g["volume_ratio_min"] = c3.number_input("量比下限", value=float(g.get("volume_ratio_min", 0.5)), step=0.1)
        g["volume_ratio_max_enabled"] = c3.checkbox("启用量比上限", value=bool(g.get("volume_ratio_max_enabled", False)))
        g["volume_ratio_max"] = c3.number_input("量比上限", value=float(g.get("volume_ratio_max", 3.0)), step=0.1)

    with st.expander("D. 五年后视镜（先看长期，再做当前筛选）", expanded=False):
        st.caption("建议先开这一层做长期体检，再叠加 A/B/C 做当下过滤。")
        d1, d2, d3 = st.columns(3)
        d5 = cfg.setdefault("rearview_5y", {})

        d5["revenue_cagr_5y_min_enabled"] = d1.checkbox(
            "启用营收5年CAGR下限(%)", value=bool(d5.get("revenue_cagr_5y_min_enabled", False))
        )
        d5["revenue_cagr_5y_min"] = d1.number_input(
            "营收5年CAGR下限(%)", value=float(d5.get("revenue_cagr_5y_min", 3.0)), step=0.5
        )
        d5["profit_cagr_5y_min_enabled"] = d1.checkbox(
            "启用净利5年CAGR下限(%)", value=bool(d5.get("profit_cagr_5y_min_enabled", False))
        )
        d5["profit_cagr_5y_min"] = d1.number_input(
            "净利5年CAGR下限(%)", value=float(d5.get("profit_cagr_5y_min", 3.0)), step=0.5
        )

        d5["roe_avg_5y_min_enabled"] = d2.checkbox(
            "启用ROE 5年均值下限(%)", value=bool(d5.get("roe_avg_5y_min_enabled", False))
        )
        d5["roe_avg_5y_min"] = d2.number_input(
            "ROE 5年均值下限(%)", value=float(d5.get("roe_avg_5y_min", 8.0)), step=0.5
        )
        d5["ocf_positive_years_5y_min_enabled"] = d2.checkbox(
            "启用经营现金流为正年数下限", value=bool(d5.get("ocf_positive_years_5y_min_enabled", False))
        )
        d5["ocf_positive_years_5y_min"] = int(
            d2.number_input(
                "经营现金流为正年数下限(0-5)",
                min_value=0,
                max_value=5,
                value=int(d5.get("ocf_positive_years_5y_min", 4)),
                step=1,
            )
        )

        d5["debt_ratio_change_5y_max_enabled"] = d3.checkbox(
            "启用负债率5年变化上限(百分点)", value=bool(d5.get("debt_ratio_change_5y_max_enabled", False))
        )
        d5["debt_ratio_change_5y_max"] = d3.number_input(
            "负债率5年变化上限(百分点)", value=float(d5.get("debt_ratio_change_5y_max", 8.0)), step=0.5
        )
        d5["gross_margin_change_5y_min_enabled"] = d3.checkbox(
            "启用毛利率5年变化下限(百分点)", value=bool(d5.get("gross_margin_change_5y_min_enabled", False))
        )
        d5["gross_margin_change_5y_min"] = d3.number_input(
            "毛利率5年变化下限(百分点)", value=float(d5.get("gross_margin_change_5y_min", -6.0)), step=0.5
        )

    st.session_state["flt_cfg"] = cfg

    render_section_intro(
        "执行与结果",
        "执行区负责跑规则，结果区负责解释结果和导出表格，让批量筛选的收尾动作更集中。",
        kicker="Execution",
        pills=("执行筛选", "二段筛选", "结果导出"),
    )
    run_col1, run_col2, run_col3 = st.columns([1, 1.4, 2])
    run_now = run_col1.button("执行筛选", type="primary", use_container_width=True, key="flt_run_filter_btn")
    two_stage = run_col2.checkbox("二段筛选（先 A/B/C，再 D 五年后视镜）", value=True, key="flt_two_stage_main")
    run_col3.caption("注：部分标的5年数据可能缺失；可通过“缺失数据处理”决定是否直接排除。")

    if run_now:
        snap = filter_load_snapshot()
        if snap.empty:
            st.error("还没有市场快照。请先在上方点击“更新全市场数据”。")
        else:
            with st.spinner("正在执行筛选..."):
                if two_stage:
                    stage1_cfg = _build_stage1_config(cfg)
                    p1, r1, m1, s1 = filter_apply_filters(snap, stage1_cfg)

                    if _has_rearview_enabled(cfg):
                        stage2_cfg = _build_stage2_config(cfg)
                        p2, r2, m2, _s2 = filter_apply_filters(p1, stage2_cfg)
                        passed_df = p2
                        rejected_df = _concat_dedup(r1, r2)
                        missing_df = _concat_dedup(m1, m2)
                    else:
                        passed_df, rejected_df, missing_df = p1, r1, m1

                    stats = {
                        "total": int(len(snap)),
                        "passed": int(len(passed_df)),
                        "rejected": int(len(rejected_df)),
                        "missing": int(len(missing_df)),
                        "stage1_passed": int(s1.get("passed", len(p1))),
                        "stage2_passed": int(len(passed_df)),
                        "stage_mode": "two",
                    }
                else:
                    passed_df, rejected_df, missing_df, stats = filter_apply_filters(snap, cfg)
                    stats["stage_mode"] = "single"
                st.session_state["flt_result"] = {
                    "passed": passed_df,
                    "rejected": rejected_df,
                    "missing": missing_df,
                    "stats": stats,
                    "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

    res = st.session_state.get("flt_result")
    if not res:
        st.info("请先更新市场快照，然后点击“执行筛选”。")
        return

    stats = res["stats"]
    render_status_row(
        (
            ("总样本", str(stats.get("total", 0))),
            ("通过池", str(stats.get("passed", 0))),
            ("排除池", str(stats.get("rejected", 0))),
            ("含缺失项", str(stats.get("missing", 0))),
        )
    )
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val in [
        (k1, "总样本", stats.get("total", 0)),
        (k2, "通过池", stats.get("passed", 0)),
        (k3, "排除池", stats.get("rejected", 0)),
        (k4, "含缺失项", stats.get("missing", 0)),
    ]:
        col.markdown(f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{val}</div></div>", unsafe_allow_html=True)

    if str(stats.get("stage_mode", "")) == "two":
        st.caption(
            f"二段筛选结果：第一段(A/B/C) 保留 {int(stats.get('stage1_passed', 0))} 只 ｜ "
            f"第二段(D) 后最终保留 {int(stats.get('stage2_passed', 0))} 只"
        )

    passed_df = res["passed"]
    rejected_df = res["rejected"]
    missing_df = res["missing"]
    st.caption(f"筛选时间: {res.get('run_at', '--')}")

    xlsx_bytes = filter_export_results_excel(passed_df, rejected_df, missing_df)
    st.download_button(
        "导出 Excel（通过池/排除池/缺失项）",
        data=xlsx_bytes,
        file_name=f"filter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    tab1, tab2, tab3 = st.tabs(["通过池", "排除池", "缺失项"])
    with tab1:
        t1, t2, t3 = st.columns([1.2, 1.2, 3.6], vertical_alignment="bottom")
        top_n = int(
            t1.number_input(
                "TopN（按总市值）",
                min_value=0,
                max_value=2000,
                value=50,
                step=10,
                key="flt_topn_mv_main",
            )
        )
        mv_desc = t2.checkbox("市值降序", value=True, key="flt_mv_desc_main")

        passed_view = passed_df.copy()
        mv_effective = False
        if "total_mv" in passed_view.columns:
            passed_view["_mv_sort"] = pd.to_numeric(passed_view["total_mv"], errors="coerce")
            mv_effective = bool(passed_view["_mv_sort"].notna().any())
            passed_view = passed_view.sort_values("_mv_sort", ascending=not mv_desc, na_position="last")
            passed_view = passed_view.drop(columns=["_mv_sort"], errors="ignore")
        if (not mv_effective) and ("market" in passed_view.columns):
            hk_only = passed_view["market"].astype(str).str.upper().eq("HK").all()
            if hk_only:
                st.warning("当前港股快照缺少总市值字段，市值排序暂不可用。先执行一次“更新全市场数据（运维台）”后再试。")
        if top_n > 0:
            passed_view = passed_view.head(top_n)

        t3.caption(f"展示数量: {len(passed_view)} / {len(passed_df)}")
        show1 = _display_df(passed_view)
        cols1 = [c for c in FILTER_DISPLAY_COLUMNS if c in show1.columns]
        st.dataframe(show1[cols1] if cols1 else show1, use_container_width=True, hide_index=True)
    with tab2:
        cols2 = [c for c in FILTER_DISPLAY_COLUMNS if c in rejected_df.columns]
        show2 = _display_df(rejected_df[cols2] if cols2 else rejected_df)
        st.dataframe(show2, use_container_width=True, hide_index=True)
    with tab3:
        cols3 = [c for c in FILTER_DISPLAY_COLUMNS if c in missing_df.columns]
        show3 = _display_df(missing_df[cols3] if cols3 else missing_df)
        st.dataframe(show3, use_container_width=True, hide_index=True)


def _render_backtest_page():
    render_section_intro(
        "回测系统",
        "这是 Quant 的第4个功能：通用港股多空对冲回测。支持策略配置、数据更新、回测执行与HTML报告产出。",
        kicker="Backtest",
        pills=("Universe管理", "策略配置", "一键回测", "交互报告"),
    )

    if "bt_console_output" not in st.session_state:
        st.session_state["bt_console_output"] = ""

    if not BACKTEST_RUNNER.exists():
        st.error(f"未找到回测入口: {BACKTEST_RUNNER}")
        return

    if "bt_preview_latest_open" not in st.session_state:
        st.session_state["bt_preview_latest_open"] = False

    BACKTEST_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_STRATEGY_TRASH_DIR.mkdir(parents=True, exist_ok=True)
    strategy_files = sorted(
        [p for p in BACKTEST_STRATEGY_DIR.glob("*.yaml") if p.is_file()],
        key=lambda p: (p.stat().st_mtime, p.name.lower()),
        reverse=True,
    )
    trash_files = sorted([p for p in BACKTEST_STRATEGY_TRASH_DIR.glob("*.yaml") if p.is_file()])
    if not strategy_files:
        st.info("策略库为空：可直接在下方输入框粘贴策略，然后保存到策略库。")

    st.markdown(
        """
        <style>
        [class*="st-key-bt_"] div.stButton > button {
          min-height: 2.2rem !important;
          padding: 0.25rem 0.75rem !important;
          font-size: 0.9rem !important;
          border-radius: 12px !important;
        }
        [class*="st-key-bt_load_strategy_"] div.stButton > button {
          min-height: 1.65rem !important;
          padding: 0.08rem 0.5rem !important;
          border-radius: 10px !important;
          font-size: 0.84rem !important;
        }
        [class*="st-key-bt_close_strategy_"] div.stButton > button,
        [class*="st-key-bt_lock_strategy_"] div.stButton > button {
          min-height: 1.3rem !important;
          min-width: 1.35rem !important;
          padding: 0rem 0.28rem !important;
          border-radius: 10px !important;
          font-size: 0.8rem !important;
        }
        [class*="st-key-bt_toggle_latest_preview"] div.stButton > button,
        [class*="st-key-bt_download_latest"] div.stDownloadButton > button {
          min-height: 2.0rem !important;
          border-radius: 10px !important;
          font-size: 0.98rem !important;
          font-weight: 650 !important;
          padding: 0.22rem 0.72rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    strategy_names = [p.name for p in strategy_files]
    if "bt_strategy_text" not in st.session_state:
        st.session_state["bt_strategy_text"] = ""
    if "bt_strategy_source" not in st.session_state:
        st.session_state["bt_strategy_source"] = "__inline__"

    selected_cfg = str(st.session_state.get("bt_selected_strategy", ""))
    if selected_cfg not in strategy_names:
        selected_cfg = ""
        st.session_state["bt_selected_strategy"] = ""

    strategy_source = str(st.session_state.get("bt_strategy_source", "") or "")
    if strategy_source and strategy_source not in {"__inline__", "__manual__"} and strategy_source not in strategy_names:
        st.session_state["bt_strategy_text"] = ""
        st.session_state["bt_strategy_source"] = "__inline__"

    def _run_cmd(cmd: list[str], label: str) -> bool:
        with st.spinner(f"{label} 执行中..."):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(BACKTEST_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                st.session_state["bt_console_output"] = output.strip()
                if proc.returncode == 0:
                    st.success(f"{label} 完成")
                    return True
                st.error(f"{label} 失败（exit={proc.returncode}）")
                return False
            except Exception as exc:
                st.error(f"{label} 异常: {exc}")
                return False

    def _extract_strategy_codes(strategy_text: str) -> list[str]:
        text = (strategy_text or "").strip()
        if not text:
            return []

        codes: list[str] = []
        try:
            import yaml  # type: ignore

            obj = yaml.safe_load(text) or {}
            if isinstance(obj, dict):
                for key in ("long_positions", "short_positions"):
                    items = obj.get(key, []) or []
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and item.get("code"):
                                codes.append(str(item.get("code", "")).strip().upper())
        except Exception:
            # 回退到正则提取，容忍非严格 YAML 的中间态。
            codes = [m.strip().upper() for m in re.findall(r'^\s*code\s*:\s*["\']?([A-Za-z0-9.^_-]+)["\']?\s*$', text, flags=re.M)]

        seen: set[str] = set()
        out: list[str] = []
        for code in codes:
            if code and code not in seen:
                seen.add(code)
                out.append(code)
        return out

    def _autofill_universe_by_codes(codes: list[str]) -> tuple[list[str], str]:
        if not codes:
            return [], ""
        try:
            import yaml  # type: ignore
        except Exception as exc:
            return [], f"自动补全失败：缺少 PyYAML 依赖（{exc}）"

        try:
            if BACKTEST_STRATEGY_DIR.exists():
                # 保证目录存在时，universe 也可写。
                BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
            universe_path = BACKTEST_DIR / "config" / "universe.yaml"
            universe_path.parent.mkdir(parents=True, exist_ok=True)
            if universe_path.exists():
                raw = yaml.safe_load(universe_path.read_text(encoding="utf-8")) or {}
            else:
                raw = {}
            if not isinstance(raw, dict):
                raw = {}

            sectors = raw.get("sectors")
            if not isinstance(sectors, dict):
                sectors = {}
                raw["sectors"] = sectors

            # 收集全局已有 code，避免重复。
            existing_codes: set[str] = set()
            for sec_obj in sectors.values():
                if not isinstance(sec_obj, dict):
                    continue
                groups = sec_obj.get("groups", {})
                if not isinstance(groups, dict):
                    continue
                for grp_obj in groups.values():
                    if not isinstance(grp_obj, dict):
                        continue
                    stocks = grp_obj.get("stocks", [])
                    if not isinstance(stocks, list):
                        continue
                    for st_obj in stocks:
                        if isinstance(st_obj, dict) and st_obj.get("code"):
                            existing_codes.add(str(st_obj.get("code")).strip().upper())

            auto_sector = sectors.get("auto_import")
            if not isinstance(auto_sector, dict):
                auto_sector = {
                    "name": "自动补全",
                    "description": "由策略校验按钮自动补齐的标的",
                    "sector_benchmark": "^HSI",
                    "groups": {},
                }
                sectors["auto_import"] = auto_sector

            groups = auto_sector.get("groups")
            if not isinstance(groups, dict):
                groups = {}
                auto_sector["groups"] = groups

            auto_group = groups.get("from_strategy")
            if not isinstance(auto_group, dict):
                auto_group = {
                    "name": "来自策略",
                    "stocks": [],
                }
                groups["from_strategy"] = auto_group

            stocks = auto_group.get("stocks")
            if not isinstance(stocks, list):
                stocks = []
                auto_group["stocks"] = stocks

            added_codes: list[str] = []
            for code in codes:
                c = str(code).strip().upper()
                if not c or c in existing_codes:
                    continue
                stocks.append(
                    {
                        "code": c,
                        "name": c,
                        "tags": ["auto-added", "from-strategy"],
                    }
                )
                existing_codes.add(c)
                added_codes.append(c)

            if added_codes:
                universe_path.write_text(
                    yaml.safe_dump(
                        raw,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    ),
                    encoding="utf-8",
                )
            return added_codes, ""
        except Exception as exc:
            return [], f"自动补全失败：{exc}"

    st.markdown("### 策略库管理")
    st.caption("点击策略名载入；右上角小 × 删除会进入回收站。")

    def _move_strategy_to_trash(src: Path) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{src.stem}__{ts}{src.suffix}"
        dst = BACKTEST_STRATEGY_TRASH_DIR / base_name
        idx = 1
        while dst.exists():
            dst = BACKTEST_STRATEGY_TRASH_DIR / f"{src.stem}__{ts}_{idx}{src.suffix}"
            idx += 1
        shutil.move(str(src), str(dst))
        return dst
    per_row = 3
    if not strategy_files:
        st.caption("暂无策略文件。可在下方填写后点击“保存到策略库”。")
    for start in range(0, len(strategy_files), per_row):
        chunk = strategy_files[start : start + per_row]
        cols = st.columns(per_row, vertical_alignment="top")
        for idx, p in enumerate(chunk):
            name = p.name
            with cols[idx]:
                b1, b2 = st.columns([8.8, 1.2], vertical_alignment="top")
                if b1.button(name, use_container_width=True, key=f"bt_load_strategy_{name}"):
                    try:
                        st.session_state["bt_selected_strategy"] = name
                        st.session_state["bt_strategy_text"] = p.read_text(encoding="utf-8")
                        st.session_state["bt_strategy_source"] = name
                        st.rerun()
                    except Exception as exc:
                        st.error(f"读取策略失败: {exc}")
                if b2.button("✕", use_container_width=True, key=f"bt_close_strategy_{name}"):
                    try:
                        _move_strategy_to_trash(p)
                        if st.session_state.get("bt_selected_strategy") == name:
                            st.session_state["bt_selected_strategy"] = ""
                            st.session_state["bt_strategy_text"] = ""
                            st.session_state["bt_strategy_source"] = "__inline__"
                        st.success(f"已移入回收站: {name}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败: {exc}")

    st.markdown("#### 回收站")
    trash_files = sorted([p for p in BACKTEST_STRATEGY_TRASH_DIR.glob("*.yaml") if p.is_file()], reverse=True)
    if trash_files:
        t1, t2, t3 = st.columns([2.4, 1.0, 1.0], vertical_alignment="bottom")
        trash_name = t1.selectbox("回收站文件", options=[p.name for p in trash_files], key="bt_trash_selected")
        if t2.button("恢复", use_container_width=True, key="bt_trash_restore"):
            src = BACKTEST_STRATEGY_TRASH_DIR / trash_name
            raw_name = trash_name.split("__", 1)[0] + ".yaml"
            dst = BACKTEST_STRATEGY_DIR / raw_name
            if dst.exists():
                dst = BACKTEST_STRATEGY_DIR / f"{dst.stem}_restored_{datetime.now().strftime('%H%M%S')}{dst.suffix}"
            try:
                shutil.move(str(src), str(dst))
                st.success(f"已恢复: {dst.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"恢复失败: {exc}")
        if t3.button("清空回收站", use_container_width=True, key="bt_trash_clear"):
            err = None
            for fp in trash_files:
                try:
                    fp.unlink(missing_ok=True)
                except Exception as exc:
                    err = exc
            if err:
                st.error(f"清空失败: {err}")
            else:
                st.success("回收站已清空。")
                st.rerun()
    else:
        st.caption("回收站为空。")

    st.markdown("### 策略输入（直接粘贴即可）")
    st.caption("把你的完整 YAML 直接贴进来，点击“粘贴策略并运行”即可，无需手动改文件。")
    st.text_area(
        "策略YAML",
        height=420,
        key="bt_strategy_text",
    )
    s1, s2 = st.columns([1.2, 2.2], vertical_alignment="bottom")
    save_name = s1.text_input("保存文件名（可选）", value="", key="bt_save_name")
    if s2.button("保存到策略库", use_container_width=True, key="bt_save_strategy_text"):
        text = (st.session_state.get("bt_strategy_text") or "").strip()
        if not text:
            st.error("策略文本为空，无法保存。")
        else:
            raw = (save_name or "").strip()
            safe = raw or "my_strategy.yaml"
            # 允许中文等Unicode文件名，仅替换文件系统常见非法字符。
            safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', safe)
            safe = safe.strip().strip(".")
            if not safe:
                safe = "my_strategy.yaml"
            if not safe.lower().endswith((".yaml", ".yml")):
                safe = f"{safe}.yaml"
            target = BACKTEST_STRATEGY_DIR / safe
            try:
                target.write_text(text, encoding="utf-8")
                st.success(f"已保存策略: {target.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"保存失败: {exc}")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("安装回测依赖", use_container_width=True, key="bt_install_deps"):
        _run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "安装依赖")
    if c2.button("校验Universe", use_container_width=True, key="bt_validate_universe"):
        # 自动补齐当前策略中的缺失代码，减少手工维护 universe。
        strategy_text_for_fill = (st.session_state.get("bt_strategy_text") or "").strip()
        if not strategy_text_for_fill and selected_cfg:
            try:
                strategy_text_for_fill = (BACKTEST_STRATEGY_DIR / selected_cfg).read_text(encoding="utf-8")
            except Exception:
                strategy_text_for_fill = ""

        if strategy_text_for_fill:
            codes = _extract_strategy_codes(strategy_text_for_fill)
            if codes:
                added_codes, err = _autofill_universe_by_codes(codes)
                if err:
                    st.warning(err)
                elif added_codes:
                    preview = ", ".join(added_codes[:8])
                    suffix = " ..." if len(added_codes) > 8 else ""
                    st.success(f"已自动补齐 {len(added_codes)} 个代码到 universe（auto_import/from_strategy）：{preview}{suffix}")
        _run_cmd([sys.executable, "run_backtest.py", "--validate-universe"], "校验Universe")
    if c3.button("更新缓存数据", use_container_width=True, key="bt_update_data"):
        _run_cmd([sys.executable, "run_backtest.py", "--update-data-only", "--start", "2021-01-01", "--end", "today"], "更新缓存")
    if c4.button("运行选中文件", use_container_width=True, key="bt_run", disabled=not bool(selected_cfg)):
        if not selected_cfg:
            st.error("当前没有策略文件，请先在下方保存一个策略到策略库。")
        else:
            _run_cmd(
                [
                    sys.executable,
                    "run_backtest.py",
                    "--config",
                    f"config/strategies/{selected_cfg}",
                    "--output",
                    "reports",
                    "--no-browser",
                ],
                "运行回测",
            )
    if st.button("粘贴策略并运行", type="primary", use_container_width=True, key="bt_run_pasted"):
        strategy_text = (st.session_state.get("bt_strategy_text") or "").strip()
        if not strategy_text:
            st.error("请先粘贴策略 YAML。")
        else:
            pasted_path = BACKTEST_STRATEGY_DIR / "_pasted_strategy_from_ui.yaml"
            try:
                pasted_path.write_text(strategy_text, encoding="utf-8")
                _run_cmd(
                    [
                        sys.executable,
                        "run_backtest.py",
                        "--config",
                        f"config/strategies/{pasted_path.name}",
                        "--output",
                        "reports",
                        "--no-browser",
                    ],
                    "运行粘贴策略",
                )
            except Exception as exc:
                st.error(f"写入粘贴策略失败: {exc}")

    console_output = st.session_state.get("bt_console_output", "")
    if console_output:
        st.caption("最近一次命令输出")
        st.code(console_output, language="bash")

    reports = sorted(BACKTEST_REPORT_DIR.glob("report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        latest = reports[0]
        st.success(f"最新报告: {latest.name}")
        preview_label = "打开页面内预览最新报告" if not st.session_state.get("bt_preview_latest_open", False) else "关闭页面内预览最新报告"
        pb1, pb2 = st.columns(2)
        with pb1:
            if st.button(preview_label, use_container_width=True, key="bt_toggle_latest_preview"):
                st.session_state["bt_preview_latest_open"] = not bool(st.session_state.get("bt_preview_latest_open", False))
                st.rerun()
        with pb2:
            st.download_button(
                "下载最新报告HTML",
                data=latest.read_bytes(),
                file_name=latest.name,
                mime="text/html",
                use_container_width=True,
                key="bt_download_latest",
            )
        if st.session_state.get("bt_preview_latest_open", False):
            try:
                html(latest.read_text(encoding="utf-8"), height=920, scrolling=True)
            except Exception as exc:
                st.warning(f"预览失败: {exc}")

        with st.expander("查看报告列表", expanded=False):
            rows = []
            for p in reports[:30]:
                stat = p.stat()
                rows.append(
                    {
                        "文件名": p.name,
                        "修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "大小(KB)": round(stat.st_size / 1024.0, 1),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂未找到回测报告，请先点击“运行回测”。")


def _render_paper_page():
    render_section_intro(
        "模拟实盘",
        "按策略逐日推进模拟持仓，记录交易与快照；入口即策略，先启动，后续逐日更新。",
        kicker="Paper",
        pills=("策略入口", "逐日更新", "看板输出"),
    )

    if "paper_console_output" not in st.session_state:
        st.session_state["paper_console_output"] = ""

    BACKTEST_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    strategy_files = sorted(
        [p for p in BACKTEST_STRATEGY_DIR.glob("*.yaml") if p.is_file()],
        key=lambda p: (p.stat().st_mtime, p.name.lower()),
        reverse=True,
    )
    if not strategy_files:
        st.info("当前没有策略文件，请先在回测系统里保存策略。")
        return
    if not BACKTEST_PAPER_RUNNER.exists():
        st.error(f"未找到模拟实盘入口: {BACKTEST_PAPER_RUNNER}")
        return

    st.markdown(
        """
        <style>
        .paper-card-wrap {
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 14px 14px 10px 14px;
          background: rgba(14, 25, 42, 0.38);
          min-height: 196px;
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        .paper-card-title {
          font-size: 1.12rem;
          font-weight: 700;
          line-height: 1.25;
          color: rgba(245, 248, 255, 0.96);
          min-height: 2.6rem;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .paper-card-desc {
          color: rgba(210, 218, 232, 0.82);
          font-size: 0.95rem;
          line-height: 1.45;
          min-height: 2.8rem;
          max-height: 2.8rem;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .paper-card-meta {
          color: rgba(186, 198, 220, 0.86);
          font-size: 0.92rem;
          min-height: 1.5rem;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        [class*="st-key-paper_action_"] div.stButton > button {
          min-height: 2.15rem !important;
          border-radius: 10px !important;
          font-size: 1.02rem !important;
          font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _paper_run_id_from_strategy(filename: str) -> str:
        stem = Path(filename).stem
        safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", stem)
        safe = re.sub(r"\s+", "", safe).strip("._")
        return f"pt_{safe or 'strategy'}"

    def _run_paper_cmd(cmd: list[str], label: str) -> bool:
        with st.spinner(f"{label} 执行中..."):
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(BACKTEST_DIR),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1800,
                )
                output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
                st.session_state["paper_console_output"] = output.strip()
                if proc.returncode == 0:
                    st.success(f"{label} 完成")
                    return True
                st.error(f"{label} 失败（exit={proc.returncode}）")
                return False
            except Exception as exc:
                st.error(f"{label} 异常: {exc}")
                return False

    def _auto_sync_dashboard(strategy_paths: list[Path]) -> None:
        """Auto rebuild paper dashboard when strategy files changed."""
        if not BACKTEST_PAPER_RUNNER.exists():
            return
        need_sync = False
        if not BACKTEST_PAPER_DASHBOARD.exists():
            need_sync = True
        else:
            try:
                dash_mtime = BACKTEST_PAPER_DASHBOARD.stat().st_mtime
                watched = [BACKTEST_PAPER_ACTIVE, *strategy_paths]
                for p in watched:
                    if p.exists() and p.stat().st_mtime > dash_mtime:
                        need_sync = True
                        break
            except Exception:
                need_sync = True
        if not need_sync:
            return
        try:
            proc = subprocess.run(
                [sys.executable, "paper_trade.py", "dashboard"],
                cwd=str(BACKTEST_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            if proc.returncode != 0:
                st.session_state["paper_console_output"] = output.strip()
                st.warning("自动同步模拟看板失败，请手动点击“重建模拟看板”。")
        except Exception:
            st.warning("自动同步模拟看板异常，请手动点击“重建模拟看板”。")

    def _load_paper_active_map() -> dict:
        if not BACKTEST_PAPER_ACTIVE.exists():
            return {}
        try:
            obj = json.loads(BACKTEST_PAPER_ACTIVE.read_text(encoding="utf-8"))
            if not isinstance(obj, list):
                return {}
            out = {}
            for item in obj:
                if isinstance(item, dict):
                    rid = str(item.get("run_id", "")).strip()
                    if rid:
                        out[rid] = item
            return out
        except Exception:
            return {}

    def _strategy_meta(path: Path) -> tuple[str, str]:
        strategy_name = path.stem
        description = ""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            m1 = re.search(r'^\s*strategy_name\s*:\s*["\']?(.*?)["\']?\s*$', raw, flags=re.MULTILINE)
            if m1:
                strategy_name = (m1.group(1) or "").strip() or strategy_name
            m2 = re.search(r'^\s*description\s*:\s*["\']?(.*?)["\']?\s*$', raw, flags=re.MULTILINE)
            if m2:
                description = (m2.group(1) or "").strip()
        except Exception:
            pass
        return strategy_name, description

    def _paper_summary(run_id: str, row) -> str:
        if not row:
            return f"未创建（run_id={run_id}）"
        last_date = str(row.get("last_update_date", "")).strip() or "-"
        status = str(row.get("status", "active")).strip() or "active"
        run_dir = Path(str(row.get("run_dir", "")).strip())
        snap = run_dir / "snapshots.csv"
        if snap.exists():
            try:
                sdf = pd.read_csv(snap)
                if not sdf.empty:
                    last = sdf.iloc[-1]
                    eq = float(last.get("equity", float("nan")))
                    cr = float(last.get("cum_return", float("nan")))
                    eq_txt = "-" if not np.isfinite(eq) else f"{eq:,.0f} HKD"
                    cr_txt = "-" if not np.isfinite(cr) else f"{cr:.2%}"
                    return f"{status} | {last_date} | 权益 {eq_txt} | 收益 {cr_txt}"
            except Exception:
                pass
        return f"{status} | 最新 {last_date}"

    active_map = _load_paper_active_map()
    _auto_sync_dashboard(strategy_files)

    active_rows = []
    for rid, row in active_map.items():
        run_dir = Path(str(row.get("run_dir", "")).strip())
        eq = float("nan")
        cr = float("nan")
        snap = run_dir / "snapshots.csv"
        if snap.exists():
            try:
                sdf = pd.read_csv(snap)
                if not sdf.empty:
                    last = sdf.iloc[-1]
                    eq = float(last.get("equity", float("nan")))
                    cr = float(last.get("cum_return", float("nan")))
            except Exception:
                pass
        active_rows.append((rid, row, eq, cr))

    total_equity = float(sum(x[2] for x in active_rows if np.isfinite(x[2])))
    weighted_initial = float(sum((x[2] / (1.0 + x[3])) for x in active_rows if np.isfinite(x[2]) and np.isfinite(x[3]) and (1.0 + x[3]) != 0.0))
    total_cum = (total_equity / weighted_initial - 1.0) if weighted_initial > 0 else float("nan")

    h1, h2 = st.columns(2)
    if h1.button("更新全部活跃模拟盘", use_container_width=True, key="paper_update_all"):
        _run_paper_cmd([sys.executable, "paper_trade.py", "update", "--all", "--as-of", "today"], "更新全部活跃模拟盘")
        st.rerun()
    if h2.button("刷新状态", use_container_width=True, key="paper_refresh_state"):
        st.rerun()

    render_status_row(
        (
            ("活跃策略", str(len(active_rows))),
            ("总权益(HKD)", "-" if not np.isfinite(total_equity) else f"{total_equity:,.2f}"),
            ("总收益", "-" if not np.isfinite(total_cum) else f"{total_cum:.2%}"),
        )
    )

    with st.expander("策略入口（单动作）", expanded=False):
        per_row = 3
        for start in range(0, len(strategy_files), per_row):
            chunk = strategy_files[start : start + per_row]
            cols = st.columns(per_row, vertical_alignment="top")
            for idx, p in enumerate(chunk):
                name = p.name
                run_id = _paper_run_id_from_strategy(name)
                row = active_map.get(run_id)
                sname, sdesc = _strategy_meta(p)
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="paper-card-wrap">
                          <div class="paper-card-title">{sname}</div>
                          <div class="paper-card-desc">{sdesc or "无策略说明"}</div>
                          <div class="paper-card-meta">{_paper_summary(run_id, row)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    action_label = "更新数据" if row else "启动模拟"
                    if st.button(action_label, use_container_width=True, key=f"paper_action_{name}"):
                        if row:
                            _run_paper_cmd(
                                [
                                    sys.executable,
                                    "paper_trade.py",
                                    "update",
                                    "--run-id",
                                    run_id,
                                    "--as-of",
                                    "today",
                                ],
                                f"模拟更新({name})",
                            )
                        else:
                            _run_paper_cmd(
                                [
                                    sys.executable,
                                    "paper_trade.py",
                                    "start",
                                    "--config",
                                    f"config/strategies/{name}",
                                    "--run-id",
                                    run_id,
                                    "--as-of",
                                    "today",
                                ],
                                f"模拟启动({name})",
                            )
                        st.rerun()

    with st.expander("低频操作", expanded=False):
        if st.button("重建模拟看板", use_container_width=True, key="paper_build_dashboard"):
            _run_paper_cmd([sys.executable, "paper_trade.py", "dashboard"], "重建模拟看板")
            st.rerun()
        render_status_row(
            (
                ("模块目录", str(BACKTEST_DIR)),
                ("模拟盘目录", str(BACKTEST_PAPER_DIR)),
                ("Python", sys.executable),
            )
        )

    if BACKTEST_PAPER_DASHBOARD.exists():
        with st.expander("模拟盘管理中心", expanded=False):
            try:
                html(BACKTEST_PAPER_DASHBOARD.read_text(encoding="utf-8"), height=1500, scrolling=True)
            except Exception as exc:
                st.warning(f"看板预览失败: {exc}")
        st.download_button(
            "下载模拟看板HTML",
            data=BACKTEST_PAPER_DASHBOARD.read_bytes(),
            file_name=BACKTEST_PAPER_DASHBOARD.name,
            mime="text/html",
            use_container_width=True,
            key="paper_download_dashboard",
        )
    else:
        if st.button("生成模拟看板", use_container_width=True, key="paper_build_dashboard_init"):
            _run_paper_cmd([sys.executable, "paper_trade.py", "dashboard"], "生成模拟看板")
            st.rerun()

    console_output = st.session_state.get("paper_console_output", "")
    if console_output:
        st.caption("最近一次模拟命令输出")
        st.code(console_output, language="bash")


def _load_portfolio_renderer():
    app_path = CURRENT_DIR.parent / "portfolio" / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"未找到仓位模块入口: {app_path}")
    module_name = "_quant_portfolio_app"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, str(app_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载仓位模块: {app_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    render_fn = getattr(module, "render_portfolio_page", None)
    if not callable(render_fn):
        raise RuntimeError("仓位模块缺少 render_portfolio_page()")
    return render_fn


def _render_portfolio_page():
    try:
        render_fn = _load_portfolio_renderer()
    except Exception as exc:
        st.error(f"仓位风控模块加载失败: {exc}")
        return
    try:
        render_fn()
    except Exception as exc:
        st.error(f"仓位风控模块运行失败: {exc}")


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
            deep_report, d_usage, d_cost, _ = _call_deepseek_analysis(json_text=deep_json)
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
                    deep_report, d_usage, d_cost, _ = _call_deepseek_analysis(json_text=deep_json)
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
if st.session_state.get("active_page") == "filter":
    _render_filter_page()
    st.stop()
if st.session_state.get("active_page") == "backtest":
    _render_backtest_page()
    st.stop()
if st.session_state.get("active_page") == "paper":
    _render_paper_page()
    st.stop()
if st.session_state.get("active_page") == "portfolio":
    _render_portfolio_page()
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
render_section_intro(
    "标的工作台",
    "上方保留自动刷新和即时刷新，下方把股票池拆成持仓与观察两栏，减少切换负担，方便快速定位当前关注标的。",
    kicker="Deck",
    pills=("自动刷新", "持仓 / 观察", "单击切换标的"),
)
render_status_row(
    (
        ("当前标的", f"{st.session_state.get('fast_selected_name', '')} ({selected_code_for_ctrl})"),
        ("市场状态", "交易时段内" if market_open_for_ctrl else "非交易时段"),
        ("刷新策略", f"{auto_refresh_sec} 秒自动刷新" if auto_refresh_on else "手动刷新"),
        ("股票池结构", f"持仓 {_holding_count} / 观察 {_watch_count}"),
    )
)

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


def _calc_ai_support_resistance(kline_df: pd.DataFrame, panel: dict):
    if kline_df.empty:
        return None, None

    df = kline_df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    boll_upper = ma20 + 2.0 * std20
    boll_lower = ma20 - 2.0 * std20

    recent = df.tail(60).copy()
    low_20 = pd.to_numeric(recent["low"], errors="coerce").tail(20).min()
    high_20 = pd.to_numeric(recent["high"], errors="coerce").tail(20).max()
    ma20_last = float(ma20.iloc[-1]) if len(ma20) and pd.notna(ma20.iloc[-1]) else None
    bu_last = float(boll_upper.iloc[-1]) if len(boll_upper) and pd.notna(boll_upper.iloc[-1]) else None
    bl_last = float(boll_lower.iloc[-1]) if len(boll_lower) and pd.notna(boll_lower.iloc[-1]) else None

    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else 0.0
    if not np.isfinite(atr14):
        atr14 = 0.0

    indicators = panel.get("indicators", {}) if isinstance(panel, dict) else {}
    ma5_now = indicators.get("ma5")
    ma20_now = indicators.get("ma20")
    macd_now = indicators.get("macd_hist")
    trend_factor = 0.0
    if ma5_now is not None and ma20_now is not None:
        trend_factor += 0.6 if float(ma5_now) >= float(ma20_now) else -0.6
    if macd_now is not None:
        trend_factor += 0.4 if float(macd_now) >= 0 else -0.4

    support_candidates = [x for x in (low_20, ma20_last, bl_last) if x is not None and np.isfinite(x)]
    resistance_candidates = [x for x in (high_20, ma20_last, bu_last) if x is not None and np.isfinite(x)]
    if not support_candidates or not resistance_candidates:
        return None, None

    support = float(np.average(support_candidates, weights=np.linspace(1.0, 1.4, len(support_candidates))))
    resistance = float(np.average(resistance_candidates, weights=np.linspace(1.0, 1.4, len(resistance_candidates))))

    # 简易 AI 风险调整：顺势上移支撑、逆势下移阻力（按 ATR 微调）
    support += trend_factor * 0.12 * atr14
    resistance += trend_factor * 0.08 * atr14
    if support >= resistance:
        mid = float((support + resistance) / 2.0)
        support = mid - max(atr14 * 0.6, mid * 0.01)
        resistance = mid + max(atr14 * 0.6, mid * 0.01)
    return support, resistance


def _build_lw_payload(kline_df: pd.DataFrame, panel: dict) -> dict:
    if kline_df.empty:
        return {}

    df = kline_df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    boll_mid = ma20
    boll_std = close.rolling(20, min_periods=20).std(ddof=0)
    boll_upper = boll_mid + 2.0 * boll_std
    boll_lower = boll_mid - 2.0 * boll_std
    support, resistance = _calc_ai_support_resistance(df, panel=panel)

    def _time_str(ts) -> str:
        return pd.to_datetime(ts).strftime("%Y-%m-%d")

    candles = [
        {
            "time": _time_str(r.date),
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
        }
        for r in df.itertuples(index=False)
    ]

    volumes = []
    for r in df.itertuples(index=False):
        is_up = float(r.close) >= float(r.open)
        volumes.append(
            {
                "time": _time_str(r.date),
                "value": float(r.volume) if pd.notna(r.volume) else 0.0,
                "color": "rgba(209,67,67,0.78)" if is_up else "rgba(31,171,99,0.78)",
            }
        )

    def _line_series(series: pd.Series) -> list[dict]:
        out: list[dict] = []
        for t, v in zip(df["date"], series):
            if pd.isna(v):
                continue
            out.append({"time": _time_str(t), "value": float(v)})
        return out

    overlays = {
        "ma5": _line_series(ma5),
        "ma20": _line_series(ma20),
        "boll_mid": _line_series(boll_mid),
        "boll_upper": _line_series(boll_upper),
        "boll_lower": _line_series(boll_lower),
    }
    if support is not None and resistance is not None:
        t0 = candles[0]["time"]
        t1 = candles[-1]["time"]
        overlays["ai_support"] = [{"time": t0, "value": float(support)}, {"time": t1, "value": float(support)}]
        overlays["ai_resistance"] = [{"time": t0, "value": float(resistance)}, {"time": t1, "value": float(resistance)}]

    return {"candles": candles, "volumes": volumes, "overlays": overlays}


def _render_lightweight_kline_chart(selected_code: str, selected_name: str, panel: dict) -> None:
    kline_df = _fetch_chart_ohlcv(selected_code, count=260)
    if kline_df is None or kline_df.empty:
        st.info("暂无可用日线K线数据，无法渲染交互K线。")
        return

    payload = _build_lw_payload(kline_df, panel=panel if isinstance(panel, dict) else {})
    if not payload or not payload.get("candles"):
        st.info("K线数据不足，暂不可渲染。")
        return

    render_section_intro(
        "专业K线盯盘",
        "主图为日线K线，叠加 MA5/MA20、布林带与 AI 支撑/阻力；下方同步成交量柱状图。",
        kicker="Lightweight Charts",
        pills=("Candles", "MA/BOLL Overlay", "AI 支撑阻力", "Volume Histogram"),
    )

    chart_payload = json.dumps(payload, ensure_ascii=False)
    safe_id = re.sub(r"[^0-9A-Za-z_]", "_", str(selected_code))
    html(
        f"""
        <div id="lw_wrap_{safe_id}" style="width:100%;min-height:740px;">
          <div style="padding:8px 10px 4px 10px;color:#e8e3d2;font-weight:700;">
            {selected_name} ({selected_code}) · 日线K线
          </div>
          <div id="lw_main_{safe_id}" style="width:100%;height:540px;"></div>
          <div id="lw_vol_{safe_id}" style="width:100%;height:170px;margin-top:10px;"></div>
        </div>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <script>
        (function() {{
          const payload = {chart_payload};
          const mainEl = document.getElementById("lw_main_{safe_id}");
          const volEl = document.getElementById("lw_vol_{safe_id}");
          if (!mainEl || !volEl || !window.LightweightCharts) return;
          mainEl.innerHTML = "";
          volEl.innerHTML = "";
          const width = Math.max(mainEl.clientWidth || 980, 680);

          const commonLayout = {{
            layout: {{
              background: {{ color: '#0f1f34' }},
              textColor: '#d9e2f0',
            }},
            grid: {{
              vertLines: {{ color: 'rgba(255,255,255,0.06)' }},
              horzLines: {{ color: 'rgba(255,255,255,0.06)' }},
            }},
            localization: {{
              locale: 'zh-CN',
            }},
            rightPriceScale: {{
              borderColor: 'rgba(255,255,255,0.15)',
            }},
            timeScale: {{
              borderColor: 'rgba(255,255,255,0.15)',
            }},
            crosshair: {{
              mode: 0
            }},
          }};

          const mainChart = LightweightCharts.createChart(mainEl, {{
            width,
            height: 540,
            ...commonLayout,
          }});
          const candle = mainChart.addCandlestickSeries({{
            upColor: '#d14343',
            downColor: '#1fab63',
            borderVisible: false,
            wickUpColor: '#d14343',
            wickDownColor: '#1fab63',
          }});
          candle.setData(payload.candles || []);

          const addLine = (data, color, width=1, style=0) => {{
            if (!Array.isArray(data) || !data.length) return null;
            const s = mainChart.addLineSeries({{
              color,
              lineWidth: width,
              lineStyle: style,
              priceLineVisible: false,
              lastValueVisible: false,
            }});
            s.setData(data);
            return s;
          }};

          addLine(payload.overlays?.ma5 || [], '#ffb454', 2, 0);
          addLine(payload.overlays?.ma20 || [], '#58a6ff', 2, 0);
          addLine(payload.overlays?.boll_mid || [], 'rgba(173,216,230,0.65)', 1, 1);
          addLine(payload.overlays?.boll_upper || [], 'rgba(173,216,230,0.5)', 1, 2);
          addLine(payload.overlays?.boll_lower || [], 'rgba(173,216,230,0.5)', 1, 2);
          addLine(payload.overlays?.ai_support || [], 'rgba(80,200,120,0.95)', 2, 2);
          addLine(payload.overlays?.ai_resistance || [], 'rgba(255,120,120,0.95)', 2, 2);

          const volChart = LightweightCharts.createChart(volEl, {{
            width,
            height: 170,
            ...commonLayout,
            rightPriceScale: {{
              scaleMargins: {{ top: 0.1, bottom: 0 }},
              borderColor: 'rgba(255,255,255,0.15)',
            }},
          }});
          const vol = volChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: '',
          }});
          vol.priceScale().applyOptions({{ scaleMargins: {{ top: 0.15, bottom: 0 }} }});
          vol.setData(payload.volumes || []);

          const syncTimeRange = (source, target) => {{
            source.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
              if (range) target.timeScale().setVisibleLogicalRange(range);
            }});
          }};
          syncTimeRange(mainChart, volChart);
          syncTimeRange(volChart, mainChart);

          const fit = () => {{
            mainChart.timeScale().fitContent();
            volChart.timeScale().fitContent();
          }};
          fit();

          const resize = () => {{
            const w = Math.max(mainEl.clientWidth || 980, 680);
            mainChart.applyOptions({{ width: w }});
            volChart.applyOptions({{ width: w }});
          }};
          window.addEventListener('resize', resize, {{ passive: true }});
        }})();
        </script>
        """,
        height=760,
        scrolling=False,
    )


def _render_intraday_orderbook_core(panel: dict, change_pct: float, orderbook_unit: str) -> None:
    intraday_df = panel.get("intraday") if isinstance(panel, dict) else pd.DataFrame()
    if intraday_df is None or not isinstance(intraday_df, pd.DataFrame):
        intraday_df = pd.DataFrame(columns=["time", "price", "volume_lot", "amount"])
    order_book_5 = panel.get("order_book_5", {"buy": [], "sell": []}) if isinstance(panel, dict) else {"buy": [], "sell": []}

    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fast-panels-gap"></div>', unsafe_allow_html=True)
    render_section_intro(
        "盘中结构",
        "左侧保留分时强弱，右侧保留五档盘口，用双栏视角把成交节奏和挂单深度放在一起读。",
        kicker="Intraday Structure",
        pills=("资金分时", "五档盘口", "同屏观察"),
    )
    left, right = st.columns([2, 1], vertical_alignment="top")
    with left:
        st.markdown('<div class="panel-title">资金分时</div>', unsafe_allow_html=True)
        if intraday_df.empty:
            st.info("暂无分时资金数据")
        else:
            chart_df = intraday_df.set_index("time")
            area_df = chart_df.reset_index()
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
        st.markdown(f'<div class="panel-title">实时盘口<span class="unit-sub">单位：{orderbook_unit}</span></div>', unsafe_allow_html=True)
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


def _render_intraday_orderbook_fragment(
    selected_code: str,
    initial_panel: dict,
    orderbook_unit: str,
    auto_refresh: bool,
    auto_refresh_seconds: int,
) -> None:
    cache_key = f"fast_panel_cache_{selected_code}"

    def _load_live_panel() -> dict:
        market_open = _is_market_open(selected_code)
        if market_open:
            panel_live = fetch_fast_panel(selected_code)
            st.session_state[cache_key] = panel_live
            return panel_live
        cached = st.session_state.get(cache_key)
        if isinstance(cached, dict):
            return cached
        if isinstance(initial_panel, dict):
            return initial_panel
        panel_live = fetch_fast_panel(selected_code)
        st.session_state[cache_key] = panel_live
        return panel_live

    if auto_refresh and _is_market_open(selected_code):
        @st.fragment(run_every=f"{int(auto_refresh_seconds)}s")
        def _live_intraday_fragment():
            panel_live = _load_live_panel()
            quote_live = panel_live.get("quote", {}) if isinstance(panel_live, dict) else {}
            live_change_pct = _calc_display_change_pct(quote_live) if isinstance(quote_live, dict) else 0.0
            _render_intraday_orderbook_core(panel_live, live_change_pct, orderbook_unit)

        _live_intraday_fragment()
    else:
        panel_now = initial_panel if isinstance(initial_panel, dict) else _load_live_panel()
        quote_now = panel_now.get("quote", {}) if isinstance(panel_now, dict) else {}
        now_change_pct = _calc_display_change_pct(quote_now) if isinstance(quote_now, dict) else 0.0
        _render_intraday_orderbook_core(panel_now, now_change_pct, orderbook_unit)


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
    live_val = fetch_live_valuation_snapshot(selected_code)
    quote_pe_dynamic = quote.get("pe_dynamic")
    quote_pb = quote.get("pb")
    quote_pe_ttm = quote.get("pe_ttm")
    pe_dynamic_live = (
        quote_pe_dynamic
        if quote_pe_dynamic is not None
        else live_val.get("pe_dynamic")
        if isinstance(live_val, dict)
        else selected_slow.get("pe_dynamic")
    )
    pe_static_live = (
        (live_val.get("pe_static") if isinstance(live_val, dict) else None)
        if (isinstance(live_val, dict) and live_val.get("pe_static") is not None)
        else (quote_pe_ttm if quote_pe_ttm is not None else selected_slow.get("pe_static"))
    )
    pe_rolling_live = (
        (live_val.get("pe_rolling") if isinstance(live_val, dict) else None)
        if (isinstance(live_val, dict) and live_val.get("pe_rolling") is not None)
        else (quote_pe_ttm if quote_pe_ttm is not None else selected_slow.get("pe_rolling"))
    )
    pb_live = (
        quote_pb
        if quote_pb is not None
        else live_val.get("pb")
        if isinstance(live_val, dict)
        else selected_slow.get("pb")
    )
    dy_live = (
        live_val.get("dividend_yield")
        if isinstance(live_val, dict) and live_val.get("dividend_yield") is not None
        else selected_slow.get("dividend_yield")
    )
    sell_lv_for_json = sorted(order_book_5.get("sell", []), key=lambda x: int(x.get("level", 0)))
    buy_lv_for_json = sorted(order_book_5.get("buy", []), key=lambda x: int(x.get("level", 0)))
    sell_total_for_json = sum(float(r.get("volume_lot") or 0) for r in sell_lv_for_json)
    buy_total_for_json = sum(float(r.get("volume_lot") or 0) for r in buy_lv_for_json)
    ofi_for_json = (buy_total_for_json / sell_total_for_json) if sell_total_for_json > 0 else None
    weibi_for_json = ((buy_total_for_json - sell_total_for_json) / (buy_total_for_json + sell_total_for_json) * 100) if (buy_total_for_json + sell_total_for_json) > 0 else None
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
            "weibi_pct": weibi_for_json,
            "spread": spread_for_json,
            "order_diff": quote.get("order_diff"),
        },
        "valuation": {
            "pe_dynamic": pe_dynamic_live,
            "pe_static": pe_static_live,
            "pe_rolling": pe_rolling_live,
            "pb": pb_live,
            "dividend_yield": dy_live,
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
    q_time = _format_display_time(quote.get("quote_time"))
    provider_used = str(quote.get("provider_used") or "").strip().lower()
    qmt_enabled = bool(quote.get("qmt_enabled", False))
    if provider_used == "primary":
        provider_label = "QMT"
    elif provider_used == "fallback":
        provider_label = "免费接口(降级)"
    elif provider_used == "error":
        provider_label = "数据源异常"
    else:
        provider_label = "免费接口"
    provider_hint = f"当前数据源: {provider_label}"
    if not qmt_enabled:
        provider_hint += " ｜ QMT开关: OFF"
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
            st.caption(f"更新时间: {q_time if q_time else 'N/A'}")
            st.caption(provider_hint)
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
    pe_dynamic = pe_dynamic_live
    pe_static = pe_static_live
    pe_rolling = pe_rolling_live
    pb_val = pb_live
    dy_val = dy_live

    open_val = quote.get("open")
    high_val = quote.get("high")
    low_val = quote.get("low")
    change_amt = quote.get("change_amount")
    vwap_val = quote.get("vwap")
    premium_pct = quote.get("premium_pct")
    amplitude_pct = quote.get("amplitude_pct")
    turnover_rate = quote.get("turnover_rate")
    turnover_rate_estimated = bool(quote.get("turnover_rate_estimated", False))
    volume_ratio = quote.get("volume_ratio")
    total_mv = quote.get("total_market_value_yi")
    float_mv = quote.get("float_market_value_yi")
    order_diff = quote.get("order_diff")

    volume_shares = quote.get("volume")
    is_hk_code = str(selected_code).isdigit() and len(str(selected_code)) == 5
    if volume_shares is not None:
        volume_display = float(volume_shares) if is_hk_code else (float(volume_shares) / 100.0)
    else:
        volume_display = None
    volume_label = "成交量(股)" if is_hk_code else "成交量(手)"
    amount_label = "成交额(HKD)" if is_hk_code else "成交额(元)"
    turnover_label = "换手率(估算)" if turnover_rate_estimated else "换手率"
    volume_ratio_label = "量比(接口口径)" if is_hk_code else "量比"
    orderbook_unit = "档位量(接口口径)" if is_hk_code else "手"
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
    weibi = ((buy_total - sell_total) / (buy_total + sell_total) * 100) if (buy_total + sell_total) > 0 else None
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
                (volume_label, _fmt_lot(volume_display)),
                (amount_label, _fmt_amount_yuan(amount_yuan)),
                (volume_ratio_label, _fmt(volume_ratio, 2)),
                (turnover_label, _fmt_pct(turnover_rate, 2)),
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
                ("买卖比(B/A)", _fmt(ofi, 2)),
                ("委比", _fmt_signed_pct(weibi, 2)),
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
            "volume_lot": volume_display,
            "volume_shares": volume_shares,
            "volume_display_label": volume_label,
            "amount_display_label": amount_label,
            "amount_yuan": amount_yuan,
            "volume_ratio_label": volume_ratio_label,
            "volume_ratio": volume_ratio,
            "turnover_label": turnover_label,
            "turnover_rate": turnover_rate,
            "turnover_rate_estimated": turnover_rate_estimated,
            "vwap": vwap_val,
            "premium_pct": premium_pct,
            "amplitude_pct": amplitude_pct,
        },
        "order_book_summary": {
            "buy_total_lot": buy_total,
            "sell_total_lot": sell_total,
            "imbalance_bid_ask": ofi,
            "weibi_pct": weibi,
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
    render_status_row(
        (
            ("当前周期", tf_caption),
            ("更新时间", q_time if q_time else "N/A"),
            ("数据源", provider_label),
            ("盘口失衡", _fmt(ofi, 2)),
            ("估值口径", f"PB {_fmt(pb_val, 2)} / 股息 {_fmt_pct(dy_val, 2)}"),
        )
    )

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
    deep_hash = hashlib.sha256(f"{ANALYSIS_ENGINE_VERSION}:{deep_json}".encode("utf-8")).hexdigest()
    live_job_id = _upsert_live_analysis_job(
        stock_code=selected_code,
        stock_name=selected_name,
        quick_json="",
        deep_json=deep_json,
        quick_hash="",
        deep_hash=deep_hash,
    )
    run_analysis_now = False
    refresh_analysis_now = False

    if copy_slot is not None:
        with copy_slot:
            html(
                f"""
                <div style="margin:0.1rem 0 0.45rem 0;">
                  <button id="copy-json-btn-{selected_code}"
                    style="width:100%;height:44px;padding:0 0.95rem;border-radius:999px;border:1px solid rgba(255,255,255,0.14);background:linear-gradient(135deg,#f0d7b0 0%,#c99859 100%);color:#111827;font-size:1.02rem;font-weight:800;cursor:pointer;white-space:nowrap;box-shadow:0 12px 24px rgba(0,0,0,0.22);">
                    复制JSON
                  </button>
                  <div id="copy-json-msg-{selected_code}" style="margin-top:0.35rem;color:rgba(239,229,216,0.82);font-size:0.88rem;"></div>
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

    render_section_intro(
        "快照矩阵",
        "把实时价格、成交、盘口、估值和技术指标压进一组可快速扫描的卡片，让盘中判断尽量停留在同一屏里。",
        kicker="Snapshot Matrix",
        pills=("实时报价", "盘口结构", "估值尺度", "技术状态"),
    )
    for i in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (title, kv_rows, desc) in zip(cols, cards[i : i + 4]):
            col.markdown(_card_html(title, kv_rows, desc), unsafe_allow_html=True)

    _render_lightweight_kline_chart(
        selected_code=selected_code,
        selected_name=selected_name,
        panel=panel if isinstance(panel, dict) else {},
    )

    _render_intraday_orderbook_fragment(
        selected_code=selected_code,
        initial_panel=panel if isinstance(panel, dict) else {},
        orderbook_unit=orderbook_unit,
        auto_refresh=bool(auto_refresh_on),
        auto_refresh_seconds=int(auto_refresh_sec),
    )
    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
    doc_cols = st.columns([5, 1], vertical_alignment="center")
    with doc_cols[0]:
        st.subheader(f"DeepSeek分析文档 · {selected_name} ({selected_code})")
    with doc_cols[1]:
        refresh_analysis_now = st.button("刷新", key=f"refresh_inline_analysis_{selected_code}", use_container_width=True)

    try:
        if run_analysis_now or refresh_analysis_now:
            done_job = _execute_analysis_job(
                job_id=live_job_id,
                mode="deep",
                ui_prefix=f"{selected_name} ",
                force_refresh=bool(refresh_analysis_now),
            )
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


selected_code_for_panel = st.session_state.get("fast_selected_code", rows[0]["code"])
selected_name_for_panel = st.session_state.get("fast_selected_name", rows[0]["name"])
panel_snapshot = _load_fast_panel_snapshot_once(selected_code_for_panel)
_render_fast_panel(selected_code_for_panel, selected_name_for_panel, panel=panel_snapshot)
