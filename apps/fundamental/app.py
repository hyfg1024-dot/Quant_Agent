from __future__ import annotations

import base64
import json
import os
import re
import sys
import time as pytime
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from streamlit.components.v1 import html
from urllib3.util.retry import Retry

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPENAI_AVAILABLE = True
OPENAI_IMPORT_ERROR = None
try:
    from openai import APIConnectionError, APITimeoutError, OpenAI
except Exception as _exc:  # pragma: no cover - 依赖缺失时兜底
    OPENAI_AVAILABLE = False
    OPENAI_IMPORT_ERROR = _exc
    OpenAI = None  # type: ignore[assignment]

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

from fundamental_engine import (
    APP_VERSION,
    analyze_watchlist,
    build_overview_table,
    delete_watch_item,
    format_pct,
    load_watchlist,
    upsert_watch_item_by_query,
)
from shared.ui_shell import render_app_shell, render_section_intro, render_status_row


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
    return key.strip("“”\"'`")


def _validate_api_key(key: str) -> None:
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在侧栏填写。")
    if not key.startswith("sk-"):
        raise RuntimeError("API Key 格式异常，应以 sk- 开头。")
    if not re.fullmatch(r"sk-[A-Za-z0-9._-]+", key):
        raise RuntimeError("API Key 包含非法字符，请重新粘贴。")


def _call_deepseek_analysis(json_text: str) -> tuple[str, dict, float, float]:
    if (not OPENAI_AVAILABLE) or (OpenAI is None):
        hint = "缺少 openai 依赖，请先安装：cd /Users/wellthen/Desktop/TEST/Quant_System/apps/fundamental && source venv/bin/activate && pip install -r requirements.txt"
        if OPENAI_IMPORT_ERROR is not None:
            raise RuntimeError(f"{hint}；原始错误: {OPENAI_IMPORT_ERROR}")
        raise RuntimeError(hint)

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
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cache_hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        cache_miss_tokens = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)

    if not report:
        raise RuntimeError("DeepSeek 未返回有效分析文本。")

    elapsed = pytime.time() - t0
    cost = (
        cache_hit_tokens / 1_000_000 * 0.028
        + cache_miss_tokens / 1_000_000 * 0.28
        + completion_tokens / 1_000_000 * 0.42
    )
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
    }
    return report, usage_dict, cost, elapsed


def _clean_text_no_na(text: str) -> str:
    out = str(text or "")
    out = re.sub(r"\bN/?A\b", "0.00", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnan\b", "0.00", out, flags=re.IGNORECASE)
    return out


def _split_sentences(text: str) -> List[str]:
    raw = _clean_text_no_na(text).replace("\n", "")
    parts = [x.strip() for x in re.split(r"(?<=[。！？；])", raw) if x.strip()]
    return parts if parts else ([raw] if raw else [])


def _format_card_desc_lines(text: str, max_lines: int = 3) -> str:
    cleaned = _clean_text_no_na(text).replace("／", "/")
    parts: List[str] = []
    for piece in cleaned.split("\n"):
        p = piece.strip()
        if not p:
            continue
        sub = [x.strip() for x in p.split("/") if x.strip()]
        parts.extend(sub if sub else [p])
    parts = parts[:max_lines]
    while len(parts) < max_lines:
        parts.append("")
    html_lines = []
    for p in parts:
        if p:
            html_lines.append(f"<span class='line'>{p}</span>")
        else:
            html_lines.append("<span class='line line-empty'>占位</span>")
    return "".join(html_lines)


def _init_state() -> None:
    if "fnd_watchlist" not in st.session_state:
        st.session_state["fnd_watchlist"] = load_watchlist()
    if "fnd_rows" not in st.session_state:
        st.session_state["fnd_rows"] = analyze_watchlist(st.session_state["fnd_watchlist"], force_refresh=False)
    if "fnd_selected_code" not in st.session_state:
        st.session_state["fnd_selected_code"] = st.session_state["fnd_rows"][0]["code"] if st.session_state["fnd_rows"] else ""
    if "fnd_deepseek_reports" not in st.session_state:
        st.session_state["fnd_deepseek_reports"] = {}

    rows = st.session_state.get("fnd_rows", [])
    if rows:
        need_refresh = False
        for row in rows:
            if str(row.get("app_version", "")) != APP_VERSION:
                need_refresh = True
                break
            if "N/A" in json.dumps(row, ensure_ascii=False) or "nan" in json.dumps(row, ensure_ascii=False).lower():
                need_refresh = True
                break
        if need_refresh:
            st.session_state["fnd_rows"] = analyze_watchlist(st.session_state["fnd_watchlist"], force_refresh=True)

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
    df = build_overview_table(rows).copy()
    st.dataframe(df, use_container_width=True, hide_index=True)

    options = [r.get("code", "") for r in rows]
    current = st.session_state.get("fnd_selected_code", options[0] if options else "")
    index = options.index(current) if current in options else 0
    chosen = st.selectbox(
        "打开评分板",
        options=options,
        index=index,
        format_func=lambda c: next((f"{x.get('name','')} ({x.get('code','')})" for x in rows if x.get("code") == c), c),
    )
    if chosen != current:
        st.session_state["fnd_selected_code"] = chosen
        st.rerun()


def _render_score_panels(row: Dict[str, Any]) -> None:
    score = float(row.get("total_score", 0.0) or 0.0)
    conclusion = _clean_text_no_na(str(row.get("conclusion", "观察")))
    coverage = format_pct(float((row.get("coverage_ratio") or 0.0) * 100.0))
    cols = st.columns(3, gap="small")
    items = [("总分", f"{score:.1f}"), ("结论", conclusion), ("覆盖率", coverage)]
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f"""
<div class="score-panel">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_dimension_cards(row: Dict[str, Any]) -> None:
    dimensions = row.get("dimensions", [])
    if not dimensions:
        st.warning("暂无可用评分数据。")
        return
    st.subheader("八维评分")
    for i in range(0, len(dimensions), 4):
        cols = st.columns(4, gap="small")
        for j, card in enumerate(dimensions[i : i + 4]):
            title = _clean_text_no_na(card.get("title", ""))
            score = _clean_text_no_na(f"{card.get('score', 0)} / {card.get('max_score', 5)}")
            desc = _format_card_desc_lines(str(card.get("comment", "")))
            with cols[j]:
                st.markdown(
                    f"""
<div class="fnd-card">
  <h4>{title}</h4>
  <div class="score">{score}</div>
  <div class="desc">{desc}</div>
</div>
""",
                    unsafe_allow_html=True,
                )


def _render_summary(row: Dict[str, Any]) -> None:
    code = row.get("code", "")
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
            progress.progress(25, text="正在压缩数据...")
            pytime.sleep(0.08)
            progress.progress(50, text="正在连接 DeepSeek...")
            try:
                report, usage, cost, elapsed = _call_deepseek_analysis(json_payload)
                progress.progress(85, text="正在生成报告...")
                pytime.sleep(0.08)
                st.session_state["fnd_deepseek_reports"][code] = {
                    "report": _clean_text_no_na(report),
                    "usage": usage,
                    "cost": cost,
                    "elapsed": elapsed,
                    "at": datetime.now().strftime("%m-%d %H:%M:%S"),
                }
                progress.progress(100, text="分析完成")
                pytime.sleep(0.1)
                progress.empty()
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
        st.text_area("分析文本（可复制）", value=deep.get("report", ""), height=360, key=f"fnd_report_{code}")


def _render_page() -> None:
    _init_state()
    watchlist = st.session_state.get("fnd_watchlist", [])
    render_app_shell(
        "fundamental",
        version=APP_VERSION,
        badges=("八维评分", "观察名单", "DeepSeek 研判"),
        metrics=(
            ("当前股票池", f"{len(watchlist)} 只"),
            ("研究视角", "八维拆解"),
            ("输出方式", "摘要 + 结构化结论"),
        ),
    )

    with st.sidebar:
        st.header("股票池管理")
        input_query = st.text_input("输入代码或名称", placeholder="例如 600007 / 中国国贸 / 00700")
        item_type = st.segmented_control("类型", options=["持仓", "观察"], default="观察")
        c1, c2 = st.columns(2, gap="small")
        if c1.button("加入", use_container_width=True):
            try:
                st.session_state["fnd_watchlist"] = upsert_watch_item_by_query(input_query, item_type or "观察")
                _refresh_rows(force_refresh=True)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if c2.button("刷新全部", use_container_width=True):
            _refresh_rows(force_refresh=True)
            st.rerun()

        if st.session_state["fnd_watchlist"]:
            remove_code = st.selectbox(
                "删除股票",
                options=[x["code"] for x in st.session_state["fnd_watchlist"]],
                format_func=lambda c: next((f"{x['name']} ({x['code']})" for x in st.session_state["fnd_watchlist"] if x["code"] == c), c),
            )
            if st.button("删除", use_container_width=True):
                st.session_state["fnd_watchlist"] = delete_watch_item(remove_code)
                if st.session_state.get("fnd_selected_code") == remove_code:
                    st.session_state["fnd_selected_code"] = ""
                _refresh_rows(force_refresh=True)
                st.rerun()

        st.markdown("---")
        st.subheader("DeepSeek API")
        user_input = st.text_input("用户名", key="deepseek_user_input")
        api_key_input = st.text_input(
            "API Key（可留空，读取环境变量）",
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

    rows: List[Dict[str, Any]] = st.session_state.get("fnd_rows", [])
    if not rows:
        st.warning("当前股票池为空，请先添加股票代码。")
        return

    row = _selected_row(rows)
    render_section_intro(
        "研究名单",
        "从股票池总览进入单只标的，先筛选，再展开评分板和结论文本，让研究路径更连续。",
        kicker="Overview",
        pills=("股票池总览", "打开评分板", "支持快速切换"),
    )
    render_status_row(
        (
            ("名单规模", f"{len(rows)} 只"),
            ("当前标的", f"{row.get('name', '')} ({row.get('code', '')})"),
            ("当前结论", _clean_text_no_na(str(row.get("conclusion", "观察")))),
        )
    )
    _render_overview(rows)
    st.divider()

    render_section_intro(
        "评分概览",
        "把总分、结论和覆盖率提到前面，先读结论，再看八维拆解，会更接近真实研究流程。",
        kicker="Scoreboard",
        pills=("总分", "结论", "覆盖率"),
    )
    st.subheader(f"基本面评分板：{row.get('name', '')}（{row.get('code', '')}）")
    _render_score_panels(row)

    if row.get("data_warnings"):
        st.warning("；".join(_clean_text_no_na(str(x)) for x in row.get("data_warnings", [])))

    render_section_intro(
        "八维拆解",
        "每个维度保留独立卡片，方便你一眼分辨优势项、短板项和需要进一步验证的部分。",
        kicker="Dimensions",
        pills=("独立维度卡", "统一阅读节奏", "便于横向比较"),
    )
    _render_dimension_cards(row)
    st.divider()
    render_section_intro(
        "总结与输出",
        "把摘要文本和后续分析动作留在同一区域，便于你在研究结束时直接复制或继续深挖。",
        kicker="Narrative",
        pills=("总结文本", "复制 JSON", "DeepSeek 深挖"),
    )
    _render_summary(row)


if __name__ == "__main__":
    _render_page()
