from __future__ import annotations

import json
import os
import re
import time as pytime
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st
from openai import APIConnectionError, APITimeoutError, OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fundamental_engine import (
    APP_VERSION,
    analyze_watchlist,
    build_overview_table,
    delete_watch_item,
    format_pct,
    load_watchlist,
    upsert_watch_item,
)


LOCAL_PREFS_PATH = "data/local_user_prefs.json"
DEEPSEEK_PROMPT = """你是专业基本面分析师。基于输入 JSON 做结构化输出：
1) 总结（不超过120字）
2) 八维点评（每维1句）
3) 关键风险（3条）
4) 跟踪清单（3条）
5) 结论：通过 / 观察 / 谨慎（给出理由）
要求：数据驱动、简洁、中文输出。"""


st.set_page_config(page_title="基本面板块", page_icon="📊", layout="wide")

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
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button * {
  background: #dbeafe !important;
  color: #0f2a52 !important;
  border: 1px solid #a8c2e8 !important;
}
.stButton > button:not([kind="tertiary"]) {
  background: #dbeafe;
  color: #0f2a52;
  border: 1px solid #a8c2e8;
  font-weight: 700;
}
.stButton > button:not([kind="tertiary"]):hover {
  background: #c7ddfb;
  color: #0b2346;
}
h1, h2, h3, h4 { color: var(--text-strong) !important; }
div[data-testid="stMetricValue"] { font-size: 1.7rem; }
.fnd-card {
  border: 1px solid rgba(80,120,180,.25);
  border-radius: 14px;
  padding: 14px 16px;
  min-height: 132px;
  background: rgba(240,245,255,0.55);
}
.fnd-card h4 {
  margin: 0 0 8px 0;
  font-size: 1.45rem;
}
.fnd-card .score {
  font-size: 1.9rem;
  font-weight: 800;
  margin: 4px 0 8px 0;
}
.fnd-card .desc {
  color: #5c6e89;
  font-size: 1.0rem;
}
</style>
""",
    unsafe_allow_html=True,
)


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
            sec = st.secrets.get("DEEPSEEK_API_KEY", "")
            if sec:
                raw = str(sec)
        except Exception:
            pass
    if not raw:
        raw = os.getenv("DEEPSEEK_API_KEY", "")
    key = raw.strip().split()[0] if raw.strip() else ""
    key = key.strip("“”\"'`")
    return key


def _validate_api_key(key: str) -> None:
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在侧栏填写。")
    if not key.startswith("sk-"):
        raise RuntimeError("API Key 格式异常，应以 sk- 开头。")
    if not re.fullmatch(r"sk-[A-Za-z0-9._-]+", key):
        raise RuntimeError("API Key 包含非法字符，请重新粘贴。")


def _call_deepseek_analysis(json_text: str) -> tuple[str, dict, float, float]:
    api_key = _resolve_deepseek_api_key()
    _validate_api_key(api_key)

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=60.0, max_retries=0)
    messages = [
        {"role": "system", "content": DEEPSEEK_PROMPT},
        {"role": "user", "content": json_text},
    ]

    t0 = pytime.time()
    response = None
    last_exc = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3,
                max_tokens=1200,
                top_p=0.9,
            )
            break
        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            if attempt < 3:
                pytime.sleep(0.8 * attempt)
                continue
        except Exception:
            raise

    if response is None:
        url = "https://api.deepseek.com/v1/chat/completions"
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1200,
            "top_p": 0.9,
        }
        try:
            session = requests.Session()
            retry = Retry(
                total=3,
                connect=3,
                read=3,
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
            raise RuntimeError(f"DeepSeek 连接失败：{req_exc}; SDK异常: {last_exc}") from req_exc

        report = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        usage_raw = raw.get("usage") or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens") or 0)
        completion_tokens = int(usage_raw.get("completion_tokens") or 0)
        cache_hit_tokens = int(usage_raw.get("prompt_cache_hit_tokens") or 0)
        cache_miss_tokens = int(usage_raw.get("prompt_cache_miss_tokens") or 0)
    else:
        report = (response.choices[0].message.content or "").strip()
        usage_obj = response.usage
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        cache_hit_tokens = int(getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0)
        cache_miss_tokens = int(getattr(usage_obj, "prompt_cache_miss_tokens", 0) or 0)

    if not report:
        raise RuntimeError("DeepSeek 未返回有效分析文本。")

    elapsed = pytime.time() - t0
    cost = (
        cache_hit_tokens / 1_000_000 * 0.028
        + cache_miss_tokens / 1_000_000 * 0.28
        + completion_tokens / 1_000_000 * 0.42
    )
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
    }
    return report, usage, cost, elapsed


def _init_state() -> None:
    if "fnd_watchlist" not in st.session_state:
        st.session_state["fnd_watchlist"] = load_watchlist()
    if "fnd_rows" not in st.session_state:
        st.session_state["fnd_rows"] = analyze_watchlist(st.session_state["fnd_watchlist"], force_refresh=False)
    if "fnd_selected_code" not in st.session_state:
        st.session_state["fnd_selected_code"] = st.session_state["fnd_rows"][0]["code"] if st.session_state["fnd_rows"] else ""
    if "fnd_deepseek_reports" not in st.session_state:
        st.session_state["fnd_deepseek_reports"] = {}

    if "_fnd_prefs_loaded" not in st.session_state:
        prefs = _load_local_prefs()
        st.session_state["deepseek_user_input"] = prefs.get("deepseek_user", "")
        st.session_state["deepseek_api_key_input"] = prefs.get("deepseek_api_key", "")
        st.session_state["_fnd_last_saved_prefs"] = {
            "deepseek_user": st.session_state.get("deepseek_user_input", ""),
            "deepseek_api_key": st.session_state.get("deepseek_api_key_input", ""),
        }
        st.session_state["_fnd_prefs_loaded"] = True


def _refresh_rows(force_refresh: bool = False) -> None:
    st.session_state["fnd_rows"] = analyze_watchlist(st.session_state["fnd_watchlist"], force_refresh=force_refresh)
    if st.session_state["fnd_rows"] and not st.session_state["fnd_selected_code"]:
        st.session_state["fnd_selected_code"] = st.session_state["fnd_rows"][0]["code"]


def _selected_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    code = st.session_state.get("fnd_selected_code", "")
    for row in rows:
        if row.get("code") == code:
            return row
    return rows[0] if rows else {}


def _render_overview(rows: List[Dict[str, Any]]) -> None:
    st.subheader("股票列表")
    header = st.columns([1, 2, 1, 1, 1, 1], gap="small")
    header[0].markdown("**代码**")
    header[1].markdown("**名称**")
    header[2].markdown("**评分**")
    header[3].markdown("**类型**")
    header[4].markdown("**股息率**")
    header[5].markdown("**打开**")

    for row in rows:
        cols = st.columns([1, 2, 1, 1, 1, 1], gap="small")
        cols[0].write(row.get("code", ""))
        cols[1].write(row.get("name", ""))
        cols[2].write(row.get("total_score", "N/A"))
        cols[3].write(row.get("type", "观察"))
        cols[4].write(format_pct(row.get("dividend_yield")))
        if cols[5].button("打开", key=f"fnd_open_{row.get('code')}"):
            st.session_state["fnd_selected_code"] = row.get("code", "")
            st.rerun()


def _render_dimension_cards(row: Dict[str, Any]) -> None:
    dimensions = row.get("dimensions", [])
    if not dimensions:
        st.warning("暂无可用评分数据。")
        return
    st.subheader("八维评分")
    for i in range(0, len(dimensions), 4):
        cols = st.columns(4, gap="small")
        for j, card in enumerate(dimensions[i : i + 4]):
            with cols[j]:
                st.markdown(
                    f"""
<div class="fnd-card">
  <h4>{card.get("title", "")}</h4>
  <div class="score">{card.get("score", "N/A")} / {card.get("max_score", 5)}</div>
  <div class="desc">{card.get("comment", "")}</div>
</div>
""",
                    unsafe_allow_html=True,
                )


def _render_summary(row: Dict[str, Any]) -> None:
    code = row.get("code", "")
    st.subheader("总结性文本")
    st.info(row.get("summary_text", "暂无总结。"))
    json_payload = json.dumps(row, ensure_ascii=False, indent=2)
    st.code(json_payload, language="json")

    btn1, btn2 = st.columns([1, 1], gap="small")
    if btn1.button("复制JSON", key=f"fnd_copy_json_{code}", use_container_width=True):
        st.toast("已生成 JSON，复制上方代码块即可。")

    if btn2.button("DeepSeek分析", key=f"fnd_deepseek_{code}", use_container_width=True):
        progress = st.progress(0, text="正在准备分析任务...")
        pytime.sleep(0.08)
        progress.progress(22, text="正在压缩数据...")
        pytime.sleep(0.08)
        progress.progress(44, text="正在连接 DeepSeek...")
        try:
            report, usage, cost, elapsed = _call_deepseek_analysis(json_payload)
            progress.progress(82, text="正在生成报告...")
            pytime.sleep(0.08)
            st.session_state["fnd_deepseek_reports"][code] = {
                "report": report,
                "usage": usage,
                "cost": cost,
                "elapsed": elapsed,
                "at": datetime.now().strftime("%m-%d %H:%M:%S"),
            }
            progress.progress(100, text="分析完成")
            pytime.sleep(0.1)
            progress.empty()
            st.success("DeepSeek 分析完成，结果已显示在下方。")
        except Exception as exc:
            progress.empty()
            st.error(f"DeepSeek 分析失败: {exc}")

    deep_result = st.session_state.get("fnd_deepseek_reports", {}).get(code)
    if deep_result:
        st.divider()
        st.subheader("DeepSeek分析结果")
        st.caption(
            f"分析时间: {deep_result.get('at','')} ｜耗时: {deep_result.get('elapsed', 0):.2f}s ｜"
            f"Tokens: {deep_result.get('usage', {}).get('total_tokens', 0)} ｜"
            f"预估成本: {deep_result.get('cost', 0):.4f} 元"
        )
        st.text_area(
            "分析文本（可复制）",
            value=deep_result.get("report", ""),
            height=360,
            key=f"fnd_report_text_{code}",
        )


def _render_page() -> None:
    _init_state()

    st.title("基本面")
    st.caption(f"版本号: {APP_VERSION}")

    with st.sidebar:
        st.header("股票池管理")
        input_code = st.text_input("股票代码", placeholder="例如 600007")
        input_name = st.text_input("股票名称(可选)", placeholder="可留空")
        item_type = st.segmented_control("类型", options=["持仓", "观察"], default="观察")
        c1, c2 = st.columns(2, gap="small")
        if c1.button("加入", use_container_width=True):
            st.session_state["fnd_watchlist"] = upsert_watch_item(input_code, input_name, item_type or "观察")
            _refresh_rows(force_refresh=False)
            st.rerun()
        if c2.button("刷新全部", use_container_width=True):
            _refresh_rows(force_refresh=True)
            st.rerun()

        if st.session_state["fnd_watchlist"]:
            remove_code = st.selectbox(
                "删除股票",
                options=[x["code"] for x in st.session_state["fnd_watchlist"]],
                format_func=lambda c: next(
                    (
                        f"{x['name']} ({x['code']})"
                        for x in st.session_state["fnd_watchlist"]
                        if x["code"] == c
                    ),
                    c,
                ),
            )
            if st.button("删除", use_container_width=True):
                st.session_state["fnd_watchlist"] = delete_watch_item(remove_code)
                if st.session_state.get("fnd_selected_code") == remove_code:
                    st.session_state["fnd_selected_code"] = ""
                _refresh_rows(force_refresh=False)
                st.rerun()

        st.markdown("---")
        st.subheader("DeepSeek API")
        user_input = st.text_input("用户名", value=st.session_state.get("deepseek_user_input", ""), key="deepseek_user_input")
        api_key_input = st.text_input(
            "API Key（可留空，读取环境变量）",
            value=st.session_state.get("deepseek_api_key_input", ""),
            type="password",
            key="deepseek_api_key_input",
        )
        current = {
            "deepseek_user": (user_input or "").strip(),
            "deepseek_api_key": (api_key_input or "").strip(),
        }
        last = st.session_state.get("_fnd_last_saved_prefs", {})
        if current != last:
            _save_local_prefs(current["deepseek_user"], current["deepseek_api_key"])
            st.session_state["_fnd_last_saved_prefs"] = current

    rows: List[Dict[str, Any]] = st.session_state["fnd_rows"]
    if not rows:
        st.warning("当前股票池为空，请先添加股票代码。")
        return

    _render_overview(rows)
    st.divider()

    row = _selected_row(rows)
    st.subheader(f"基本面评分板：{row.get('name', '')}（{row.get('code', '')}）")
    m1, m2, m3 = st.columns(3, gap="small")
    m1.metric("总分", row.get("total_score", "N/A"))
    m2.metric("结论", row.get("conclusion", "N/A"))
    m3.metric("覆盖率", format_pct((row.get("coverage_ratio") or 0) * 100))

    if row.get("data_warnings"):
        st.warning("；".join(row.get("data_warnings", [])))

    _render_dimension_cards(row)
    st.divider()
    _render_summary(row)

    with st.expander("原始表格预览"):
        st.dataframe(build_overview_table(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    _render_page()

