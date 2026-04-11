"""Plotly report rendering for backtest outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest_engine import BacktestResult
from .config_loader import StrategyConfig
from .metrics import MetricsReport


class ReportVisualizer:
    """Generate interactive standalone HTML report."""

    def __init__(self, logger=print) -> None:
        self.logger = logger

    def generate_report(
        self,
        config: StrategyConfig,
        result: BacktestResult,
        metrics: MetricsReport,
        sensitivity_df: Optional[pd.DataFrame],
        output_dir: Path,
    ) -> Path:
        """Render HTML report and return output path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in config.strategy_name])
        report_path = output_dir / f"report_{safe_name}_{ts}.html"

        nav_df = self._build_nav_df(result)
        fig_nav = self._fig_nav(nav_df, config)
        fig_heat = self._fig_monthly_heatmap(metrics.monthly_returns)
        fig_dd = self._fig_drawdown(nav_df)
        fig_month_bar = self._fig_monthly_compare(metrics.monthly_returns)
        fig_roll = self._fig_rolling(nav_df)
        fig_contrib = self._fig_contribution(result.code_contribution)
        fig_cost = self._fig_cost_curve(result.daily_costs)
        fig_sens = self._fig_sensitivity(sensitivity_df)

        summary_cards = self._build_summary_cards(metrics.summary_table)
        stops_html = self._stops_table(result.stop_loss_events)
        warnings_html = self._warnings_block(result.warnings)

        sections = []
        sections.append(self._section("策略概览", summary_cards))
        sections.append(self._section("净值曲线", fig_nav.to_html(full_html=False, include_plotlyjs="cdn")))
        sections.append(self._section("收益分析", fig_heat.to_html(full_html=False, include_plotlyjs=False) + fig_month_bar.to_html(full_html=False, include_plotlyjs=False)))
        sections.append(self._section("风险分析", fig_dd.to_html(full_html=False, include_plotlyjs=False) + fig_roll.to_html(full_html=False, include_plotlyjs=False)))
        sections.append(self._section("持仓与交易", fig_contrib.to_html(full_html=False, include_plotlyjs=False) + fig_cost.to_html(full_html=False, include_plotlyjs=False)))
        sections.append(self._section("敏感性分析", fig_sens.to_html(full_html=False, include_plotlyjs=False)))
        sections.append(self._section("数据汇总", self._tables_html(metrics, result, stops_html, warnings_html)))

        html = f"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{config.strategy_name} 回测报告</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 0; background: #0d1423; color: #eaf0ff; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
  .header {{ background: linear-gradient(135deg, #1f2f4a, #15253f); border: 1px solid #2b3f61; border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
  .muted {{ color: #aab8d0; }}
  .sec {{ margin: 18px 0; background: #121d33; border: 1px solid #273a5b; border-radius: 14px; padding: 14px; }}
  .sec h2 {{ margin: 2px 0 12px 0; font-size: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px; }}
  .card {{ background: #1b2a46; border: 1px solid #314a73; border-radius: 10px; padding: 10px; }}
  .card .k {{ color: #9bb0d4; font-size: 12px; text-transform: uppercase; }}
  .card .v {{ font-size: 22px; font-weight: 700; margin-top: 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ border: 1px solid #2f446a; padding: 6px 8px; font-size: 12px; }}
  th {{ background: #1f3150; }}
  .warn {{ background: #3b2b1d; border: 1px solid #72522f; padding: 10px; border-radius: 8px; margin-top: 8px; }}
  .risk {{ background: #2f2434; border: 1px solid #64416f; padding: 10px; border-radius: 8px; margin-top: 8px; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>{config.strategy_name}</h1>
      <div>{config.description or '-'}</div>
      <div class="muted">回测区间: {config.backtest.start_date} 至 {config.backtest.end_date} | 板块: {config.sector or '未指定'} | 报告时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
      <div class="risk">风险提示：本报告回测基于历史数据和事后可得标的，存在后视偏差（look-ahead bias）和生存者偏差，结果不代表未来收益。</div>
    </div>
    {''.join(sections)}
  </div>
</body>
</html>
"""

        report_path.write_text(html, encoding="utf-8")
        self.logger(f"[REPORT] 已生成: {report_path}")
        return report_path

    def _build_nav_df(self, result: BacktestResult) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "long": result.daily_long_value,
                "short": result.daily_short_value,
                "portfolio": result.daily_portfolio_value,
            }
        ).dropna(how="all")

        for k, s in result.benchmark_nav.items():
            df[k] = s.reindex(df.index).ffill()

        for c in ["long", "short", "portfolio"]:
            if c in df.columns and not df[c].dropna().empty:
                base = float(df[c].dropna().iloc[0])
                if base != 0:
                    df[c] = df[c] / base
        return df

    def _fig_nav(self, nav_df: pd.DataFrame, config: StrategyConfig) -> go.Figure:
        fig = go.Figure()
        for col in nav_df.columns:
            fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[col], mode="lines", name=col))

        for ev in config.events:
            x = pd.Timestamp(ev.date)
            if nav_df.index.min() <= x <= nav_df.index.max():
                fig.add_vline(x=x, line_dash="dot", line_color="#ffb85c")
                fig.add_annotation(x=x, y=1.02, yref="paper", text=ev.label, showarrow=False, font={"size": 10})

        fig.update_layout(title="净值曲线（基准=1.0）", template="plotly_dark", height=480)
        return fig

    def _fig_monthly_heatmap(self, monthly: pd.DataFrame) -> go.Figure:
        series = monthly.get("portfolio", pd.Series(dtype=float)).dropna()
        if series.empty:
            return self._empty_fig("月度收益热力图（无数据）")

        df = series.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        piv = df.pivot(index="year", columns="month", values="ret").sort_index()

        fig = go.Figure(
            data=go.Heatmap(
                z=piv.values,
                x=[int(c) for c in piv.columns],
                y=[int(i) for i in piv.index],
                colorscale="RdYlGn",
                colorbar={"title": "return"},
                zmid=0,
            )
        )
        fig.update_layout(title="月度收益热力图（组合）", template="plotly_dark", height=380)
        return fig

    def _fig_drawdown(self, nav_df: pd.DataFrame) -> go.Figure:
        if "portfolio" not in nav_df.columns or nav_df["portfolio"].dropna().empty:
            return self._empty_fig("回撤曲线（无数据）")
        s = nav_df["portfolio"].dropna()
        dd = s / s.cummax() - 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name="drawdown"))
        fig.update_layout(title="组合回撤曲线", template="plotly_dark", height=320, yaxis_tickformat=".1%")
        return fig

    def _fig_monthly_compare(self, monthly: pd.DataFrame) -> go.Figure:
        if monthly.empty:
            return self._empty_fig("月度多空收益对比（无数据）")
        fig = go.Figure()
        x = monthly.index
        for c in ["long", "short", "portfolio"]:
            if c in monthly.columns:
                fig.add_trace(go.Bar(x=x, y=monthly[c], name=c))
        fig.update_layout(title="月度收益对比", template="plotly_dark", barmode="group", height=360, yaxis_tickformat=".1%")
        return fig

    def _fig_rolling(self, nav_df: pd.DataFrame) -> go.Figure:
        if not {"portfolio", "long", "short"}.issubset(set(nav_df.columns)):
            return self._empty_fig("滚动指标（无数据）")

        p = nav_df["portfolio"].pct_change()
        l = nav_df["long"].pct_change()
        s = nav_df["short"].pct_change()

        roll_sharpe = (p.rolling(60).mean() / p.rolling(60).std()) * np.sqrt(252)
        roll_corr = l.rolling(60).corr(s)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, name="rolling_sharpe_60d"), secondary_y=False)
        fig.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, name="long_short_corr_60d"), secondary_y=True)
        fig.update_layout(title="60日滚动夏普与多空相关性", template="plotly_dark", height=380)
        return fig

    def _fig_contribution(self, contrib: pd.Series) -> go.Figure:
        s = pd.to_numeric(contrib, errors="coerce").dropna()
        if s.empty:
            return self._empty_fig("个股收益贡献（无数据）")
        s = s.sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=s.index.tolist(), y=s.values.tolist(), marker_color=["#7bc96f" if v >= 0 else "#d96c6c" for v in s.values]))
        fig.update_layout(title="个股收益贡献分解（HKD）", template="plotly_dark", height=360)
        return fig

    def _fig_cost_curve(self, costs: pd.DataFrame) -> go.Figure:
        if costs is None or costs.empty:
            return self._empty_fig("交易成本与融券费用（无数据）")
        fig = go.Figure()
        for c in ["cum_trade_fee", "cum_borrow_fee", "cum_total_fee"]:
            if c in costs.columns:
                fig.add_trace(go.Scatter(x=costs.index, y=costs[c], mode="lines", name=c))
        fig.update_layout(title="累计成本曲线", template="plotly_dark", height=320)
        return fig

    def _fig_sensitivity(self, df: Optional[pd.DataFrame]) -> go.Figure:
        if df is None or df.empty:
            return self._empty_fig("融券费率敏感性（无数据）")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["borrow_rate"], y=df["final_value"], mode="lines+markers", name="final_value"))
        fig.update_layout(title="融券费率敏感性", template="plotly_dark", height=320, xaxis_title="borrow_rate", yaxis_title="final_value(HKD)")
        return fig

    def _build_summary_cards(self, summary: pd.DataFrame) -> str:
        if summary.empty or "portfolio" not in summary.index:
            return "<div class='cards'><div class='card'><div class='k'>状态</div><div class='v'>无指标</div></div></div>"

        p = summary.loc["portfolio"]
        cards = [
            ("累计收益", self._fmt_pct(p.get("cumulative_return"))),
            ("年化收益", self._fmt_pct(p.get("annual_return"))),
            ("夏普", self._fmt_num(p.get("sharpe"))),
            ("最大回撤", self._fmt_pct(p.get("max_drawdown_pct"))),
            ("Beta(HSI)", self._fmt_num(p.get("beta_vs_hsi"))),
            ("信息比率", self._fmt_num(p.get("information_ratio"))),
            ("总成本(HKD)", self._fmt_num(p.get("total_cost"))),
            ("成本拖累", self._fmt_pct(p.get("cost_drag_ratio"))),
        ]

        html = ["<div class='cards'>"]
        for k, v in cards:
            html.append(f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div></div>")
        html.append("</div>")
        return "".join(html)

    def _tables_html(self, metrics: MetricsReport, result: BacktestResult, stops_html: str, warnings_html: str) -> str:
        summary_tbl = metrics.summary_table.round(6).to_html(classes="tbl") if not metrics.summary_table.empty else "<p>无 summary</p>"
        asset_tbl = metrics.asset_table.round(6).to_html(classes="tbl") if not metrics.asset_table.empty else "<p>无个股明细</p>"
        annual_tbl = metrics.annual_returns.round(6).to_html(classes="tbl") if not metrics.annual_returns.empty else "<p>无年度收益</p>"

        return (
            "<h3>绩效指标汇总</h3>" + summary_tbl +
            "<h3>个股表现明细</h3>" + asset_tbl +
            "<h3>年度收益拆解</h3>" + annual_tbl +
            "<h3>止损事件</h3>" + stops_html +
            "<h3>数据警告</h3>" + warnings_html
        )

    def _stops_table(self, stops: list[dict]) -> str:
        if not stops:
            return "<p>无止损触发记录</p>"
        return pd.DataFrame(stops).to_html(index=False)

    def _warnings_block(self, warnings: list[str]) -> str:
        if not warnings:
            return "<p>无数据警告</p>"
        lines = "".join([f"<li>{w}</li>" for w in warnings])
        return f"<div class='warn'><ul>{lines}</ul></div>"

    def _section(self, title: str, body_html: str) -> str:
        return f"<section class='sec'><h2>{title}</h2>{body_html}</section>"

    def _empty_fig(self, title: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_dark", height=260)
        return fig

    def _fmt_pct(self, value: object) -> str:
        try:
            v = float(value)
            if np.isfinite(v):
                return f"{v * 100:.2f}%"
        except Exception:
            pass
        return "-"

    def _fmt_num(self, value: object) -> str:
        try:
            v = float(value)
            if np.isfinite(v):
                return f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.2f}"
        except Exception:
            pass
        return "-"
