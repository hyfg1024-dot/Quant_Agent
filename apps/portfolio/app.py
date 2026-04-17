from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.db_manager import (
    delete_position,
    get_atr20,
    get_position_flows,
    get_position_pnl,
    init_duckdb,
    upsert_position,
)
from shared.ui_shell import render_app_shell, render_section_intro, render_status_row


APP_VERSION = "PF-20260416-01"


def _fmt_pct(v: Optional[float]) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "-"


def _fmt_num(v: Optional[float], d: int = 2) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        return f"{float(v):,.{d}f}"
    except Exception:
        return "-"


def _render_pie(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("暂无持仓，无法绘制资金占比。")
        return
    pie_df = df.copy()
    pie_df["label"] = pie_df["symbol"] + " " + pie_df["name"].fillna("")
    pie_df["market_value"] = pd.to_numeric(pie_df["market_value"], errors="coerce").fillna(0.0)
    pie_df = pie_df[pie_df["market_value"] > 0]
    if pie_df.empty:
        st.info("当前持仓市值均为 0，暂无饼图。")
        return
    try:
        import plotly.express as px

        fig = px.pie(
            pie_df,
            values="market_value",
            names="label",
            hole=0.38,
            title="持仓占总资金分布",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # fallback
        st.bar_chart(pie_df.set_index("label")["market_value"], use_container_width=True)


def _render_position_form() -> None:
    st.markdown("#### 仓位录入 / 更新")
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    symbol = c1.text_input("股票代码", value="00700.HK", key="pf_form_symbol").strip().upper()
    market = c2.selectbox("市场", options=["HK", "A"], index=0, key="pf_form_market")
    open_date = c3.date_input("开仓日期", value=datetime.now().date(), key="pf_form_open_date")

    n1, n2, n3, n4 = st.columns(4)
    avg_cost = n1.number_input("买入平均成本", min_value=0.0, value=100.0, step=0.1, key="pf_form_avg_cost")
    quantity = n2.number_input("持仓数量", min_value=0.0, value=0.0, step=100.0, key="pf_form_quantity")
    stop_loss = n3.number_input("预期止损价", min_value=0.0, value=0.0, step=0.1, key="pf_form_stop_loss")
    take_profit = n4.number_input("预期止盈价", min_value=0.0, value=0.0, step=0.1, key="pf_form_take_profit")
    note = st.text_input("流水备注（可选）", value="manual", key="pf_form_note")

    b1, b2 = st.columns(2)
    if b1.button("保存/更新仓位", use_container_width=True, key="pf_btn_save"):
        try:
            upsert_position(
                code=symbol,
                market=market,
                avg_cost=float(avg_cost),
                quantity=float(quantity),
                stop_loss_price=(None if float(stop_loss) <= 0 else float(stop_loss)),
                take_profit_price=(None if float(take_profit) <= 0 else float(take_profit)),
                open_date=open_date,
                note=note,
            )
            st.success("仓位已保存。")
            st.rerun()
        except Exception as exc:
            st.error(f"保存失败: {exc}")

    if b2.button("删除该仓位", use_container_width=True, key="pf_btn_delete"):
        try:
            ok = delete_position(symbol, market=market, note=note or "manual delete")
            if ok:
                st.success("仓位已删除。")
            else:
                st.warning("未找到该仓位。")
            st.rerun()
        except Exception as exc:
            st.error(f"删除失败: {exc}")


def _render_atr_sizer(total_capital: float) -> None:
    st.markdown("#### 仓单规模建议器（ATR20 + 单笔风险 1%）")
    c1, c2, c3, c4 = st.columns(4)
    code = c1.text_input("标的代码", value="00700.HK", key="pf_atr_code").strip().upper()
    market = c2.selectbox("市场", options=["HK", "A"], index=0, key="pf_atr_market")
    risk_pct = c3.number_input("单笔风险上限(%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, key="pf_atr_risk_pct")
    atr_mult = c4.number_input("止损倍数(ATR)", min_value=0.5, max_value=5.0, value=1.0, step=0.1, key="pf_atr_mult")

    c5, c6, c7 = st.columns(3)
    account_equity = c5.number_input("账户净值(HKD)", min_value=1.0, value=float(total_capital), step=1000.0, key="pf_atr_equity")
    lot_size = int(c6.number_input("每手股数", min_value=1.0, value=100.0, step=1.0, key="pf_atr_lot_size"))
    entry_price_override = c7.number_input(
        "入场价(可覆盖)",
        min_value=0.0,
        value=0.0,
        step=0.1,
        key="pf_atr_entry_override",
    )

    info = get_atr20(market=market, code=code, lookback=120, window=20)
    atr = info.get("atr")
    latest_close = info.get("close")
    trade_date = info.get("trade_date")

    if atr is None or not np.isfinite(float(atr)) or float(atr) <= 0:
        st.warning("ATR20 暂不可用（历史数据不足或缺失），请先更新该标的数据。")
        return

    entry_price = float(entry_price_override) if float(entry_price_override) > 0 else float(latest_close or 0.0)
    if entry_price <= 0:
        st.warning("无法获取有效入场价，请手动输入入场价。")
        return

    risk_budget = float(account_equity) * float(risk_pct) / 100.0
    per_share_risk = float(atr) * float(atr_mult)
    suggested_shares = int(math.floor(risk_budget / per_share_risk)) if per_share_risk > 0 else 0
    suggested_lots = int(math.floor(suggested_shares / lot_size)) if lot_size > 0 else 0
    executable_shares = suggested_lots * lot_size
    est_position_value = executable_shares * entry_price

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ATR20", _fmt_num(float(atr), 4))
    m2.metric("风险预算(HKD)", _fmt_num(risk_budget, 2))
    m3.metric("建议股数", f"{suggested_shares:,}")
    m4.metric("建议手数", f"{suggested_lots:,}")

    st.caption(
        f"口径: 最新价={_fmt_num(latest_close, 3)}（{trade_date}） | "
        f"入场价={_fmt_num(entry_price, 3)} | 单股风险=ATR×倍数={_fmt_num(per_share_risk, 4)}"
    )
    st.info(
        "建议下单规模（按整手）: "
        f"{executable_shares:,} 股（{suggested_lots:,} 手），"
        f"预计仓位市值 {_fmt_num(est_position_value, 2)} HKD。"
    )


def render_portfolio_page() -> None:
    """Render portfolio tracker panel (embedded in trading app)."""
    init_duckdb()
    total_capital = st.number_input(
        "组合总资金(HKD)",
        min_value=1.0,
        value=1_000_000.0,
        step=10_000.0,
        key="pf_total_capital",
    )

    _render_position_form()

    pnl_df = get_position_pnl(total_capital=float(total_capital))
    total_market_value = float(pd.to_numeric(pnl_df.get("market_value"), errors="coerce").fillna(0).sum()) if not pnl_df.empty else 0.0
    total_pnl = float(pd.to_numeric(pnl_df.get("unrealized_pnl"), errors="coerce").fillna(0).sum()) if not pnl_df.empty else 0.0
    exposure = total_market_value / float(total_capital) if total_capital > 0 else 0.0

    render_section_intro(
        "仓位与风控看板",
        "这里聚合仓位分布、浮动盈亏和 ATR 风险头寸建议，用于执行前后的一致性风控。",
        kicker="Portfolio",
        pills=("持仓占比", "浮动盈亏", "ATR20 建议手数"),
    )
    render_status_row(
        (
            ("持仓标的数", f"{len(pnl_df)} 只"),
            ("持仓市值", f"{_fmt_num(total_market_value, 2)} HKD"),
            ("组合敞口", _fmt_pct(exposure)),
            ("浮动盈亏", f"{_fmt_num(total_pnl, 2)} HKD"),
        )
    )

    left, right = st.columns([1.2, 1.6], vertical_alignment="top")
    with left:
        _render_pie(pnl_df)
    with right:
        st.markdown("#### 持仓 PnL 明细")
        if pnl_df.empty:
            st.info("暂无持仓记录。先在上方录入仓位。")
        else:
            show_cols = [
                "symbol",
                "name",
                "avg_cost",
                "close",
                "quantity",
                "market_value",
                "unrealized_pnl",
                "unrealized_pnl_pct",
                "weight_pct",
                "stop_loss_price",
                "take_profit_price",
                "open_date",
                "trade_date",
            ]
            table = pnl_df[show_cols].copy()
            table = table.rename(
                columns={
                    "symbol": "代码",
                    "name": "名称",
                    "avg_cost": "成本价",
                    "close": "现价",
                    "quantity": "持仓数量",
                    "market_value": "持仓市值",
                    "unrealized_pnl": "浮动盈亏",
                    "unrealized_pnl_pct": "盈亏%",
                    "weight_pct": "仓位%",
                    "stop_loss_price": "止损价",
                    "take_profit_price": "止盈价",
                    "open_date": "开仓日期",
                    "trade_date": "价格日期",
                }
            )
            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "成本价": st.column_config.NumberColumn(format="%.4f"),
                    "现价": st.column_config.NumberColumn(format="%.4f"),
                    "持仓数量": st.column_config.NumberColumn(format="%.2f"),
                    "持仓市值": st.column_config.NumberColumn(format="%.2f"),
                    "浮动盈亏": st.column_config.NumberColumn(format="%.2f"),
                    "盈亏%": st.column_config.NumberColumn(format="%.2f%%"),
                    "仓位%": st.column_config.NumberColumn(format="%.2f%%"),
                    "止损价": st.column_config.NumberColumn(format="%.4f"),
                    "止盈价": st.column_config.NumberColumn(format="%.4f"),
                },
            )

    _render_atr_sizer(total_capital=float(total_capital))

    st.markdown("#### 持仓流水（最近200条）")
    flows = get_position_flows(limit=200)
    if flows.empty:
        st.caption("暂无流水记录。")
    else:
        st.dataframe(flows, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="仓位与风控", page_icon="🧭", layout="wide")
    render_app_shell(
        "portfolio",
        version=APP_VERSION,
        badges=("仓位管理", "PnL", "ATR风控"),
        metrics=(
            ("核心视角", "仓位 + 风控"),
            ("执行口径", "单笔风险 1%"),
            ("工作流", "录入 -> 监控 -> 调整"),
        ),
    )
    render_portfolio_page()


if __name__ == "__main__":
    main()

