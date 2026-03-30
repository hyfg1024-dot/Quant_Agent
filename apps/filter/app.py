from __future__ import annotations

import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filter_engine import (
    APP_VERSION,
    DISPLAY_COLUMNS,
    apply_filters,
    build_ai_quick_config,
    default_filter_config,
    export_results_excel,
    get_snapshot_meta,
    get_template_config,
    load_snapshot,
    load_templates,
    refresh_market_snapshot,
    save_template,
)
from shared.ui_shell import render_app_shell, render_section_intro, render_status_row

st.set_page_config(page_title="大过滤器", page_icon="🧪", layout="wide")

st.markdown(
    """
<style>
.kpi {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 10px 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
}
.kpi .label { color: rgba(232, 223, 210, 0.88); font-size: 0.95rem; font-weight: 700; }
.kpi .value { color: rgba(255, 248, 241, 0.98); font-size: 1.75rem; font-weight: 800; margin-top: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

LOCAL_PREFS_PATH = "data/local_user_prefs.json"


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


if "flt_cfg" not in st.session_state:
    st.session_state["flt_cfg"] = default_filter_config()
if "flt_result" not in st.session_state:
    st.session_state["flt_result"] = None
if "deepseek_user_input" not in st.session_state:
    _prefs = _load_local_prefs()
    st.session_state["deepseek_user_input"] = _prefs.get("deepseek_user", "")
if "deepseek_api_key_input" not in st.session_state:
    _prefs = _load_local_prefs()
    st.session_state["deepseek_api_key_input"] = _prefs.get("deepseek_api_key", "")


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

meta = get_snapshot_meta()
render_app_shell(
    "filter",
    version=APP_VERSION,
    badges=("全市场快照", "模板筛选", "Excel 导出"),
    metrics=(
        ("快照样本", f"{int(meta.get('row_count', 0) or 0)} 只" if meta else "尚未更新"),
        ("深度补充", f"{int(meta.get('enriched_count', 0) or 0)} 只" if meta else "等待抓取"),
        ("筛选流程", "配置 -> 执行 -> 导出"),
    ),
)
render_section_intro(
    "筛选作业流",
    "先更新快照，再配置模板和条件，最后执行筛选并导出结果。这一页现在按真实工作流排布，而不是把控件平铺满屏。",
    kicker="Workflow",
    pills=("更新快照", "保存模板", "执行筛选", "导出 Excel"),
)
render_status_row(
    (
        ("快照状态", meta.get("last_update", "尚未更新") if meta else "尚未更新"),
        ("样本数量", f"{int(meta.get('row_count', 0) or 0)} 只" if meta else "0 只"),
        ("深度补充", f"{int(meta.get('enriched_count', 0) or 0)} 只" if meta else "0 只"),
    )
)

# Sidebar: update
st.sidebar.markdown("---")
st.sidebar.subheader("数据更新")
max_stocks = st.sidebar.number_input("本次更新股票数（0=全部）", min_value=0, max_value=6000, value=2000, step=100)
enrich_n = st.sidebar.number_input("深度补充数量（调用基本面引擎）", min_value=0, max_value=2000, value=300, step=50)
force_refresh = st.sidebar.checkbox("忽略缓存强制重抓", value=False)

if st.sidebar.button("更新全市场数据", use_container_width=True):
    with st.spinner("正在更新市场数据，请稍候..."):
        try:
            stats = refresh_market_snapshot(
                max_stocks=int(max_stocks),
                enrich_top_n=int(enrich_n),
                force_refresh=bool(force_refresh),
            )
            if bool(stats.get("fallback", False)):
                st.sidebar.warning("本次未连通东财接口，已回退到本地快照（未覆盖旧数据）。请检查代理/VPN后重试。")
            else:
                st.sidebar.success(f"更新完成：{stats['row_count']} 只，深度补充 {stats['enriched_count']} 只")
            meta = get_snapshot_meta()
        except Exception as exc:
            st.sidebar.error(f"更新失败: {exc}")

if meta:
    st.sidebar.caption(
        f"最近更新: {meta.get('last_update', '--')}\n"
        f"样本数量: {meta.get('row_count', '--')}\n"
        f"深度补充: {meta.get('enriched_count', '--')}"
    )

# Sidebar: templates
st.sidebar.markdown("---")
st.sidebar.subheader("模板")
all_tpl = load_templates()
tpl_names = sorted(all_tpl.keys())
selected_tpl = st.sidebar.selectbox("加载模板", options=["(无)"] + tpl_names)
if st.sidebar.button("读取模板", use_container_width=True):
    if selected_tpl and selected_tpl != "(无)":
        st.session_state["flt_cfg"] = get_template_config(selected_tpl)
        st.sidebar.success(f"已加载模板: {selected_tpl}")

save_tpl_name = st.sidebar.text_input("保存为模板名", value="")
if st.sidebar.button("保存当前模板", use_container_width=True):
    try:
        save_template(save_tpl_name, st.session_state["flt_cfg"])
        st.sidebar.success("模板已保存")
    except Exception as exc:
        st.sidebar.error(str(exc))

st.sidebar.markdown("---")
st.sidebar.subheader("DeepSeek API")
user_input = st.sidebar.text_input(
    "用户名（用于区分不同使用者）",
    key="deepseek_user_input",
)
api_key_input = st.sidebar.text_input(
    "API Key（可留空，读取环境变量）",
    type="password",
    key="deepseek_api_key_input",
)
_current_prefs = {
    "deepseek_user": (user_input or "").strip(),
    "deepseek_api_key": (api_key_input or "").strip(),
}
_saved_prefs = _load_local_prefs()
if (
    _current_prefs.get("deepseek_user", "") != str(_saved_prefs.get("deepseek_user", ""))
    or _current_prefs.get("deepseek_api_key", "") != str(_saved_prefs.get("deepseek_api_key", ""))
):
    _save_local_prefs(_current_prefs["deepseek_user"], _current_prefs["deepseek_api_key"])
st.sidebar.caption("已本地保存，不会上传到 GitHub。")

st.sidebar.markdown("---")
st.sidebar.subheader("筛选模式")
mode = st.sidebar.radio("模式", options=["手动筛选", "AI辅助设定"], index=0)

cfg = st.session_state["flt_cfg"]

if mode == "AI辅助设定":
    render_section_intro(
        "AI 条件草拟",
        "用一句目标描述生成一版初稿条件，然后你再手动微调，适合先粗筛再精修。",
        kicker="Assist",
        pills=("自然语言转条件", "生成后可继续修改"),
    )
    st.subheader("AI辅助设定（自然语言）")
    prompt = st.text_area("输入你的目标", value="高股息、低估值、低负债、防御型", height=90)
    if st.button("生成条件并应用", type="primary"):
        cfg = build_ai_quick_config(prompt, cfg)
        st.session_state["flt_cfg"] = cfg
        st.success("已根据描述生成条件，你可继续微调后执行筛选。")

render_section_intro(
    "条件矩阵",
    "把筛选条件拆成四个层次，从硬排除到五年后视镜，帮助你先做风险清洗，再叠加估值、质量和长期验证。",
    kicker="Configuration",
    pills=("A 硬排除", "B 估值质量", "C 行业规模", "D 五年后视镜"),
)
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
    "执行区负责运行筛选逻辑，结果区集中展示样本分布、二段筛选说明和导出能力。",
    kicker="Execution",
    pills=("执行筛选", "二段筛选", "结果导出"),
)
run_col1, run_col2, run_col3 = st.columns([1, 1.4, 2])
run_now = run_col1.button("执行筛选", type="primary", use_container_width=True)
two_stage = run_col2.checkbox("二段筛选（先 A/B/C，再 D 五年后视镜）", value=True)
run_col3.caption("注：部分标的5年数据可能缺失；可通过“缺失数据处理”决定是否直接排除。")

if run_now:
    snap = load_snapshot()
    if snap.empty:
        st.error("还没有市场快照。请先在左侧点击“更新全市场数据”。")
    else:
        with st.spinner("正在执行筛选..."):
            if two_stage:
                stage1_cfg = _build_stage1_config(cfg)
                p1, r1, m1, s1 = apply_filters(snap, stage1_cfg)

                if _has_rearview_enabled(cfg):
                    stage2_cfg = _build_stage2_config(cfg)
                    p2, r2, m2, _s2 = apply_filters(p1, stage2_cfg)
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
                passed_df, rejected_df, missing_df, stats = apply_filters(snap, cfg)
                stats["stage_mode"] = "single"

            st.session_state["flt_result"] = {
                "passed": passed_df,
                "rejected": rejected_df,
                "missing": missing_df,
                "stats": stats,
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

res = st.session_state.get("flt_result")

if res:
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

    xlsx_bytes = export_results_excel(passed_df, rejected_df, missing_df)
    st.download_button(
        "导出 Excel（通过池/排除池/缺失项）",
        data=xlsx_bytes,
        file_name=f"filter_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    tab1, tab2, tab3 = st.tabs(["通过池", "排除池", "缺失项"])
    with tab1:
        st.dataframe(passed_df[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(rejected_df[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(missing_df[DISPLAY_COLUMNS], use_container_width=True, hide_index=True)
else:
    st.info("请先更新市场快照，然后点击“执行筛选”。")
