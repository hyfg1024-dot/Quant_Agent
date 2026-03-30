from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Sequence

import streamlit as st


@dataclass(frozen=True)
class ShellMeta:
    label: str
    nav_title: str
    title: str
    subtitle: str
    accent: str
    accent_soft: str
    glow: str
    metrics: Sequence[tuple[str, str]]


_SHELLS: dict[str, ShellMeta] = {
    "filter": ShellMeta(
        label="Signal Sieve",
        nav_title="大过滤器",
        title="筛选指挥台",
        subtitle="先排雷，再精选。把复杂参数折叠成更有节奏的筛选流程，同时保留专业用户需要的控制力度。",
        accent="#7fa36b",
        accent_soft="#d7e6c9",
        glow="rgba(127, 163, 107, 0.22)",
        metrics=(
            ("主视角", "先排雷后精选"),
            ("信息层级", "配置 + 执行 + 结果"),
            ("工作方式", "批量筛选与导出"),
        ),
    ),
    "fundamental": ShellMeta(
        label="Fundamental Atlas",
        nav_title="基本面分析",
        title="基本面研究台",
        subtitle="用更清晰的层次组织八维评分、摘要结论和观察名单，让研究过程更像一块分析桌面而不是表单堆叠。",
        accent="#5da6a7",
        accent_soft="#bfe3df",
        glow="rgba(93, 166, 167, 0.24)",
        metrics=(
            ("主视角", "八维评分"),
            ("信息层级", "概览 + 维度 + 结论"),
            ("工作方式", "长期观察与复盘"),
        ),
    ),
    "trading": ShellMeta(
        label="Trading Deck",
        nav_title="交易面分析",
        title="交易总控台",
        subtitle="把股票池、盘口、分时、快照和 DeepSeek 研判收拢进一张真正可操作的交易桌面。",
        accent="#d9a766",
        accent_soft="#f3d7b0",
        glow="rgba(217, 167, 102, 0.22)",
        metrics=(
            ("主视角", "实时盯盘"),
            ("信息层级", "盘口 + 分时 + AI"),
            ("工作方式", "围绕持仓/观察池"),
        ),
    ),
}

_NAV_ORDER = ("filter", "fundamental", "trading")


def _shell_style(meta: ShellMeta, active_page: str) -> str:
    return f"""
    <style>
    :root {{
      --qs-bg-top: #07111f;
      --qs-bg-mid: #0f1f34;
      --qs-bg-bot: #15263d;
      --qs-paper: rgba(248, 243, 231, 0.86);
      --qs-paper-soft: rgba(255, 249, 240, 0.62);
      --qs-ink: #101b2d;
      --qs-muted: #526278;
      --qs-line: rgba(255, 255, 255, 0.12);
      --qs-panel: rgba(10, 18, 31, 0.54);
      --qs-panel-2: rgba(255, 250, 241, 0.08);
      --qs-accent: {meta.accent};
      --qs-accent-soft: {meta.accent_soft};
      --qs-glow: {meta.glow};
      --qs-display: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      --qs-sans: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    }}
    .stApp {{
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.06), transparent 24%),
        radial-gradient(circle at top right, var(--qs-glow), transparent 26%),
        linear-gradient(160deg, var(--qs-bg-top) 0%, var(--qs-bg-mid) 48%, var(--qs-bg-bot) 100%);
      color: rgba(255, 247, 238, 0.92);
      font-family: var(--qs-sans);
    }}
    .block-container {{
      padding-top: 2.2rem;
      padding-bottom: 4rem;
    }}
    [data-testid="stHeader"] {{
      background: transparent;
    }}
    [data-testid="stSidebar"] {{
      background:
        linear-gradient(180deg, rgba(6, 12, 22, 0.96) 0%, rgba(14, 24, 40, 0.98) 100%);
      border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    [data-testid="stSidebar"] * {{
      font-family: var(--qs-sans);
    }}
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
      color: rgba(245, 239, 229, 0.94) !important;
      -webkit-text-fill-color: rgba(245, 239, 229, 0.94) !important;
    }}
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button[kind="secondary"],
    .stButton > button:not([kind="tertiary"]),
    .stDownloadButton > button {{
      border-radius: 999px !important;
      border: 1px solid rgba(255, 255, 255, 0.18) !important;
      background:
        linear-gradient(135deg, color-mix(in srgb, var(--qs-accent) 84%, white 8%) 0%, color-mix(in srgb, var(--qs-accent) 56%, #7b5832 44%) 100%) !important;
      color: #101521 !important;
      font-weight: 800 !important;
      letter-spacing: 0.02em;
      box-shadow: 0 12px 24px rgba(0, 0, 0, 0.18);
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
      transform: translateY(-1px);
      box-shadow: 0 18px 28px rgba(0, 0, 0, 0.24);
    }}
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button div,
    .stButton > button span,
    .stButton > button p,
    .stButton > button div,
    .stDownloadButton > button span {{
      color: #101521 !important;
      -webkit-text-fill-color: #101521 !important;
    }}
    h1, h2, h3, h4, h5 {{
      color: rgba(250, 244, 236, 0.96) !important;
      font-family: var(--qs-display) !important;
      font-weight: 700 !important;
      letter-spacing: 0.01em;
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stCaptionContainer"] {{
      color: rgba(235, 228, 218, 0.88) !important;
    }}
    [data-testid="stMetric"] {{
      border-radius: 22px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
      padding: 0.9rem 1rem;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    [data-testid="stMetricLabel"] div {{
      color: rgba(237, 224, 206, 0.72) !important;
      font-weight: 700 !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.72rem;
    }}
    [data-testid="stMetricValue"] div {{
      color: rgba(255, 248, 240, 0.96) !important;
      font-family: var(--qs-display) !important;
      font-size: 2rem !important;
    }}
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea {{
      background: rgba(255, 249, 241, 0.09) !important;
      color: rgba(250, 244, 236, 0.96) !important;
      border-radius: 16px !important;
      border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }}
    [data-baseweb="tag"] {{
      background: rgba(255, 255, 255, 0.08) !important;
    }}
    [data-baseweb="select"] * {{
      color: rgba(250, 244, 236, 0.96) !important;
      -webkit-text-fill-color: rgba(250, 244, 236, 0.96) !important;
    }}
    [data-testid="stTabs"] [role="tablist"] {{
      gap: 0.7rem;
    }}
    [data-testid="stTabs"] [role="tab"] {{
      background: rgba(255, 255, 255, 0.06) !important;
      border: 1px solid rgba(255, 255, 255, 0.1) !important;
      border-radius: 999px !important;
      color: rgba(244, 238, 228, 0.72) !important;
      padding: 0.3rem 1rem !important;
      font-weight: 700 !important;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
      background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.09)) !important;
      color: rgba(255, 250, 243, 0.98) !important;
      border-color: color-mix(in srgb, var(--qs-accent) 70%, white 30%) !important;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.04), 0 14px 30px rgba(0,0,0,0.16);
    }}
    [data-testid="stExpander"] details {{
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.09);
      background: rgba(255, 251, 245, 0.05);
      backdrop-filter: blur(12px);
    }}
    [data-testid="stExpander"] details > summary {{
      background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04)) !important;
      padding-top: 0.35rem !important;
      padding-bottom: 0.35rem !important;
    }}
    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary svg {{
      color: rgba(255, 248, 240, 0.96) !important;
      fill: rgba(255, 248, 240, 0.96) !important;
      font-weight: 800 !important;
    }}
    [data-testid="stExpander"] details > div {{
      background: rgba(5, 11, 19, 0.14);
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }}
    [data-testid="stDataFrame"] {{
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.10);
      box-shadow: 0 16px 32px rgba(0, 0, 0, 0.16);
    }}
    [data-testid="stDataFrame"] [role="grid"] {{
      background: rgba(10, 17, 27, 0.74);
    }}
    [data-testid="stDataFrame"] * {{
      color: rgba(248, 242, 233, 0.94) !important;
    }}
    .stAlert {{
      border-radius: 18px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255, 251, 244, 0.08);
    }}
    .stDivider {{
      border-top-color: rgba(255, 255, 255, 0.08) !important;
    }}
    .qs-hero {{
      position: relative;
      overflow: hidden;
      border-radius: 32px;
      border: 1px solid rgba(255, 255, 255, 0.10);
      background:
        linear-gradient(145deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.04) 24%, rgba(0,0,0,0.16) 100%),
        radial-gradient(circle at top left, rgba(255,255,255,0.14), transparent 32%),
        linear-gradient(135deg, rgba(8,15,26,0.92), rgba(14,27,44,0.88));
      padding: 1.6rem 1.7rem;
      margin: 0 0 1.35rem 0;
      box-shadow:
        0 24px 60px rgba(0, 0, 0, 0.22),
        inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .qs-hero::after {{
      content: "";
      position: absolute;
      inset: auto -20% -45% 48%;
      height: 220px;
      background: radial-gradient(circle, var(--qs-glow) 0%, transparent 68%);
      pointer-events: none;
    }}
    .qs-hero-grid {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) minmax(280px, 1fr);
      gap: 1.25rem;
      align-items: end;
    }}
    .qs-kicker {{
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
      border-radius: 999px;
      padding: 0.35rem 0.8rem;
      background: rgba(255, 249, 239, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: rgba(245, 231, 214, 0.88);
      font-size: 0.76rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      width: fit-content;
    }}
    .qs-kicker::before {{
      content: "";
      width: 0.55rem;
      height: 0.55rem;
      border-radius: 999px;
      background: var(--qs-accent);
      box-shadow: 0 0 18px color-mix(in srgb, var(--qs-accent) 68%, white 32%);
    }}
    .qs-title {{
      margin: 0.95rem 0 0.7rem 0;
      color: rgba(255, 247, 239, 0.98);
      font-size: clamp(2.4rem, 6vw, 4.6rem);
      line-height: 0.98;
      font-family: var(--qs-display);
      font-weight: 700;
      max-width: 10ch;
    }}
    .qs-subtitle {{
      max-width: 50rem;
      color: rgba(244, 238, 229, 0.92) !important;
      -webkit-text-fill-color: rgba(244, 238, 229, 0.92) !important;
      font-size: 1.02rem;
      line-height: 1.75;
      margin: 0;
      text-shadow: 0 1px 0 rgba(0, 0, 0, 0.18);
    }}
    .qs-meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      margin-top: 1rem;
    }}
    .qs-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.42rem 0.8rem;
      background: rgba(255, 255, 255, 0.10);
      border: 1px solid rgba(255, 255, 255, 0.12);
      color: rgba(248, 242, 233, 0.94) !important;
      -webkit-text-fill-color: rgba(248, 242, 233, 0.94) !important;
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .qs-badge.qs-version {{
      color: #1a2333;
      background: linear-gradient(135deg, var(--qs-accent-soft), #f6ecdb);
      border-color: rgba(255, 255, 255, 0.12);
    }}
    .qs-side-column {{
      display: grid;
      gap: 0.8rem;
    }}
    .qs-mini-card {{
      border-radius: 22px;
      padding: 1rem 1.05rem;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04)),
        rgba(255,255,255,0.02);
      backdrop-filter: blur(10px);
    }}
    .qs-mini-label {{
      color: rgba(238, 229, 216, 0.78) !important;
      -webkit-text-fill-color: rgba(238, 229, 216, 0.78) !important;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.7rem;
      font-weight: 800;
    }}
    .qs-mini-value {{
      margin-top: 0.42rem;
      color: rgba(255, 248, 241, 0.98);
      font-size: 1.15rem;
      line-height: 1.4;
      font-weight: 700;
    }}
    .qs-top-nav {{
      display: flex;
      gap: 1rem;
      align-items: center;
      margin: 1.05rem 0 0.95rem 0;
      flex-wrap: wrap;
    }}
    .qs-top-nav-marker {{
      display: none;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker) div[data-testid="stButton"] {{
      margin: 0 !important;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker) div.stButton > button {{
      min-height: 3.55rem !important;
      width: 100% !important;
      border-radius: 999px !important;
      border: 1.5px solid rgba(255, 255, 255, 0.18) !important;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.08),
        0 8px 18px rgba(0,0,0,0.12) !important;
      color: #122033 !important;
      -webkit-text-fill-color: #122033 !important;
      font-size: 0.98rem !important;
      font-weight: 800 !important;
      letter-spacing: 0.01em;
      padding: 0 1.75rem !important;
      opacity: 0.96;
      transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease, opacity 160ms ease;
      transform: none !important;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker) div.stButton > button span,
    div[data-testid="column"]:has(.qs-top-nav-marker) div.stButton > button p,
    div[data-testid="column"]:has(.qs-top-nav-marker) div.stButton > button div {{
      color: #122033 !important;
      -webkit-text-fill-color: #122033 !important;
    }}
    .st-key-qs_top_nav_filter div.stButton > button {{
      background:
        linear-gradient(180deg, rgba(175, 193, 131, 0.98), rgba(145, 162, 106, 1)) !important;
      border-color: rgba(218, 229, 188, 0.72) !important;
    }}
    .st-key-qs_top_nav_fundamental div.stButton > button {{
      background:
        linear-gradient(180deg, rgba(139, 198, 194, 0.98), rgba(101, 160, 157, 1)) !important;
      border-color: rgba(196, 235, 232, 0.72) !important;
    }}
    .st-key-qs_top_nav_trading div.stButton > button {{
      background:
        linear-gradient(180deg, rgba(225, 193, 138, 0.98), rgba(195, 154, 93, 1)) !important;
      border-color: rgba(245, 221, 180, 0.76) !important;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker) div.stButton > button:hover {{
      opacity: 1;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.10),
        0 12px 24px rgba(0,0,0,0.15) !important;
      transform: none !important;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker.is-active) div.stButton > button {{
      border-color: rgba(255, 248, 236, 0.92) !important;
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.12),
        0 0 0 1px rgba(255,255,255,0.12),
        0 14px 30px rgba(0,0,0,0.18) !important;
      opacity: 1;
    }}
    div[data-testid="column"]:has(.qs-top-nav-marker.is-active) div.stButton > button span,
    div[data-testid="column"]:has(.qs-top-nav-marker.is-active) div.stButton > button p,
    div[data-testid="column"]:has(.qs-top-nav-marker.is-active) div.stButton > button div {{
      color: #122033 !important;
      -webkit-text-fill-color: #122033 !important;
    }}
    .qs-section-intro {{
      display: grid;
      gap: 0.85rem;
      margin: 1.4rem 0 0.85rem 0;
      padding: 1rem 1.15rem 1.05rem;
      border-radius: 24px;
      border: 1px solid rgba(255, 255, 255, 0.09);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
        rgba(8, 14, 24, 0.24);
      box-shadow: 0 16px 34px rgba(0, 0, 0, 0.16);
    }}
    .qs-section-kicker {{
      color: rgba(236, 226, 212, 0.78) !important;
      -webkit-text-fill-color: rgba(236, 226, 212, 0.78) !important;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.7rem;
      font-weight: 800;
    }}
    .qs-section-title {{
      margin: 0.35rem 0 0 0;
      color: rgba(255, 248, 241, 0.98);
      font-size: clamp(1.5rem, 3vw, 2.1rem);
      line-height: 1.04;
      font-family: var(--qs-display);
    }}
    .qs-section-desc {{
      margin: 0.45rem 0 0 0;
      color: rgba(241, 234, 224, 0.92) !important;
      -webkit-text-fill-color: rgba(241, 234, 224, 0.92) !important;
      line-height: 1.72;
      font-size: 0.96rem;
      max-width: 60rem;
      text-shadow: 0 1px 0 rgba(0, 0, 0, 0.14);
    }}
    .qs-inline-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .qs-inline-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 2rem;
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(255,255,255,0.08);
      color: rgba(245, 238, 228, 0.94) !important;
      -webkit-text-fill-color: rgba(245, 238, 228, 0.94) !important;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .qs-status-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
      margin: 0.65rem 0 1rem 0;
    }}
    .qs-status-card {{
      border-radius: 20px;
      padding: 0.9rem 1rem;
      border: 1px solid rgba(255, 255, 255, 0.10);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)),
        rgba(255,255,255,0.02);
      box-shadow: 0 14px 28px rgba(0, 0, 0, 0.15);
    }}
    .qs-status-label {{
      color: rgba(236, 226, 213, 0.78) !important;
      -webkit-text-fill-color: rgba(236, 226, 213, 0.78) !important;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.68rem;
      font-weight: 800;
    }}
    [data-testid="stAppViewContainer"] .qs-kicker,
    [data-testid="stAppViewContainer"] .qs-subtitle,
    [data-testid="stAppViewContainer"] .qs-badge,
    [data-testid="stAppViewContainer"] .qs-mini-label,
    [data-testid="stAppViewContainer"] .qs-nav-pill,
    [data-testid="stAppViewContainer"] .qs-section-kicker,
    [data-testid="stAppViewContainer"] .qs-section-desc,
    [data-testid="stAppViewContainer"] .qs-inline-pill,
    [data-testid="stAppViewContainer"] .qs-status-label,
    [data-testid="stAppViewContainer"] .qs-status-value,
    [data-testid="stAppViewContainer"] .qs-mini-value {{
      color: inherit;
    }}
    [data-testid="stAppViewContainer"] .qs-kicker {{
      color: rgba(245, 231, 214, 0.92) !important;
      -webkit-text-fill-color: rgba(245, 231, 214, 0.92) !important;
    }}
    [data-testid="stAppViewContainer"] .qs-subtitle,
    [data-testid="stAppViewContainer"] .qs-section-desc {{
      color: rgba(242, 235, 225, 0.94) !important;
      -webkit-text-fill-color: rgba(242, 235, 225, 0.94) !important;
    }}
    [data-testid="stAppViewContainer"] .qs-badge:not(.qs-version),
    [data-testid="stAppViewContainer"] .qs-inline-pill,
    [data-testid="stAppViewContainer"] .qs-nav-pill {{
      color: rgba(245, 238, 228, 0.94) !important;
      -webkit-text-fill-color: rgba(245, 238, 228, 0.94) !important;
    }}
    [data-testid="stAppViewContainer"] .qs-mini-label,
    [data-testid="stAppViewContainer"] .qs-status-label,
    [data-testid="stAppViewContainer"] .qs-section-kicker {{
      color: rgba(237, 227, 214, 0.80) !important;
      -webkit-text-fill-color: rgba(237, 227, 214, 0.80) !important;
    }}
    [data-testid="stAppViewContainer"] .qs-mini-value,
    [data-testid="stAppViewContainer"] .qs-status-value {{
      color: rgba(255, 248, 241, 0.98) !important;
      -webkit-text-fill-color: rgba(255, 248, 241, 0.98) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-kicker,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-kicker span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-subtitle,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-subtitle p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-badge,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-badge span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-mini-label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-mini-value,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-nav-pill,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-nav-pill span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-kicker,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-title span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-desc,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-desc p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-inline-pill,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-inline-pill span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-status-label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-status-value {{
      opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-kicker,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-kicker span {{
      color: rgba(245, 231, 214, 0.92) !important;
      -webkit-text-fill-color: rgba(245, 231, 214, 0.92) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-subtitle,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-subtitle p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-desc,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-desc p {{
      color: rgba(242, 235, 225, 0.94) !important;
      -webkit-text-fill-color: rgba(242, 235, 225, 0.94) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-badge:not(.qs-version),
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-badge:not(.qs-version) span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-inline-pill,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-inline-pill span,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-nav-pill,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-nav-pill span {{
      color: rgba(245, 238, 228, 0.94) !important;
      -webkit-text-fill-color: rgba(245, 238, 228, 0.94) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-mini-label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-status-label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-kicker {{
      color: rgba(237, 227, 214, 0.80) !important;
      -webkit-text-fill-color: rgba(237, 227, 214, 0.80) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-mini-value,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-status-value,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .qs-section-title span {{
      color: rgba(255, 248, 241, 0.98) !important;
      -webkit-text-fill-color: rgba(255, 248, 241, 0.98) !important;
    }}
    .qs-status-value {{
      margin-top: 0.4rem;
      color: rgba(255, 248, 241, 0.98);
      font-size: 1.08rem;
      line-height: 1.45;
      font-weight: 700;
      font-family: var(--qs-display);
    }}
    .score-panel,
    .fnd-card,
    .kpi,
    .fast-card,
    .ob-block {{
      border-radius: 22px !important;
      border: 1px solid rgba(255, 255, 255, 0.10) !important;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03)) !important;
      box-shadow: 0 18px 34px rgba(0, 0, 0, 0.18);
      backdrop-filter: blur(12px);
    }}
    .score-panel .label,
    .kpi .label,
    .fast-card .t,
    .fast-card .k,
    .fnd-card .desc,
    .fast-card .d {{
      color: rgba(231, 221, 207, 0.68) !important;
    }}
    .score-panel .value,
    .kpi .value,
    .fnd-card .score,
    .fast-card .vv,
    .price-num,
    .chg-num,
    .ob-price,
    .ob-vol {{
      color: rgba(255, 249, 242, 0.98) !important;
      font-family: var(--qs-display) !important;
    }}
    .fnd-card h4,
    .panel-title,
    .ob-title,
    .section-title,
    .group-title,
    .fast-head-title {{
      color: rgba(255, 246, 236, 0.96) !important;
      font-family: var(--qs-display) !important;
    }}
    .panel-title {{
      font-size: 2.15rem !important;
      line-height: 1.05 !important;
      margin-bottom: 0.8rem !important;
    }}
    .section-title,
    .group-title {{
      font-size: 1.8rem !important;
    }}
    .engine-divider,
    .subsection-divider,
    .watch-split-divider {{
      border-top-color: rgba(255, 255, 255, 0.14) !important;
    }}
    .unit-sub,
    .label,
    .desc,
    .line,
    .k,
    .d,
    .vv,
    .stock-open-wrap,
    .stock-del-inline-wrap {{
      color: rgba(239, 229, 216, 0.82) !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .score-panel .label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .kpi .label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fast-card .t,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fast-card .k,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fast-card .d,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fnd-card .desc,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fnd-card .line,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .unit-sub,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-lab,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-sep,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .label,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .desc,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .line,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .k {{
      color: rgba(232, 223, 210, 0.88) !important;
      -webkit-text-fill-color: rgba(232, 223, 210, 0.88) !important;
      opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .score-panel .value,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .kpi .value,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fast-card .vv,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fnd-card h4,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fnd-card .score,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-price,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-vol,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .panel-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .section-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .group-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .fast-head-title,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .price-num,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .chg-num,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .vv {{
      color: rgba(255, 248, 241, 0.98) !important;
      -webkit-text-fill-color: rgba(255, 248, 241, 0.98) !important;
      opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-sell {{
      color: #ff9f8e !important;
      -webkit-text-fill-color: #ff9f8e !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] .ob-buy {{
      color: #8fe3c3 !important;
      -webkit-text-fill-color: #8fe3c3 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] label,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] span,
    [data-testid="stAppViewContainer"] .stNumberInput label,
    [data-testid="stAppViewContainer"] .stNumberInput label p,
    [data-testid="stAppViewContainer"] .stTextInput label,
    [data-testid="stAppViewContainer"] .stTextInput label p,
    [data-testid="stAppViewContainer"] .stTextArea label,
    [data-testid="stAppViewContainer"] .stTextArea label p,
    [data-testid="stAppViewContainer"] .stSelectbox label,
    [data-testid="stAppViewContainer"] .stSelectbox label p,
    [data-testid="stAppViewContainer"] .stMultiSelect label,
    [data-testid="stAppViewContainer"] .stMultiSelect label p,
    [data-testid="stAppViewContainer"] .stRadio label,
    [data-testid="stAppViewContainer"] .stRadio label p,
    [data-testid="stAppViewContainer"] .stCheckbox label,
    [data-testid="stAppViewContainer"] .stCheckbox label p,
    [data-testid="stAppViewContainer"] .stCheckbox label span,
    [data-testid="stAppViewContainer"] div[data-testid="stToggle"] label,
    [data-testid="stAppViewContainer"] div[data-testid="stToggle"] label span,
    [data-testid="stAppViewContainer"] div[data-testid="stToggle"] label p {{
      color: rgba(241, 234, 224, 0.92) !important;
      -webkit-text-fill-color: rgba(241, 234, 224, 0.92) !important;
      opacity: 1 !important;
      font-weight: 700 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stAppViewContainer"] .stNumberInput [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stTextInput [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stTextArea [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stSelectbox [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stMultiSelect [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stRadio [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] .stCheckbox [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] div[data-testid="stToggle"] label [data-testid="stMarkdownContainer"] p {{
      color: rgba(241, 234, 224, 0.92) !important;
      -webkit-text-fill-color: rgba(241, 234, 224, 0.92) !important;
      opacity: 1 !important;
      font-weight: 700 !important;
    }}
    [data-testid="stAppViewContainer"] [aria-disabled="true"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [aria-disabled="true"] [data-testid="stWidgetLabel"] span,
    [data-testid="stAppViewContainer"] [aria-disabled="true"] [data-testid="stWidgetLabel"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stNumberInput"][aria-disabled="true"] label,
    [data-testid="stAppViewContainer"] [data-testid="stNumberInput"][aria-disabled="true"] label p,
    [data-testid="stAppViewContainer"] [data-testid="stTextInput"][aria-disabled="true"] label,
    [data-testid="stAppViewContainer"] [data-testid="stTextInput"][aria-disabled="true"] label p {{
      color: rgba(199, 206, 216, 0.82) !important;
      -webkit-text-fill-color: rgba(199, 206, 216, 0.82) !important;
      opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stNumberInput"] input:disabled,
    [data-testid="stAppViewContainer"] [data-testid="stTextInput"] input:disabled {{
      color: rgba(233, 237, 244, 0.90) !important;
      -webkit-text-fill-color: rgba(233, 237, 244, 0.90) !important;
      opacity: 1 !important;
    }}
    [data-testid="stAppViewContainer"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stAppViewContainer"] [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
      background: rgba(255, 249, 241, 0.10) !important;
      border: 1px solid rgba(255, 255, 255, 0.14) !important;
      color: rgba(246, 240, 231, 0.96) !important;
      -webkit-text-fill-color: rgba(246, 240, 231, 0.96) !important;
    }}
    .a-up {{
      color: #ff9f8e !important;
    }}
    .a-down {{
      color: #8fe3c3 !important;
    }}
    [data-testid="stRadio"] {{
      border-radius: 20px;
      padding: 0.45rem 0.55rem;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
    }}
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span {{
      color: rgba(244, 238, 228, 0.88) !important;
      font-weight: 700 !important;
    }}
    [data-testid="stCheckbox"] label p,
    [data-testid="stCheckbox"] label span {{
      color: rgba(245, 237, 226, 0.86) !important;
    }}
    [data-testid="stSelectbox"] label p,
    [data-testid="stNumberInput"] label p,
    [data-testid="stTextInput"] label p,
    [data-testid="stTextArea"] label p {{
      color: rgba(241, 232, 218, 0.82) !important;
      font-weight: 700 !important;
    }}
    [data-testid="stSpinner"] * {{
      color: rgba(245, 237, 227, 0.92) !important;
    }}
    .qs-hero,
    .qs-section-intro,
    .qs-status-card,
    .score-panel,
    .fnd-card,
    .kpi,
    .fast-card,
    .ob-block {{
      animation: qs-rise 520ms ease-out both;
    }}
    @keyframes qs-rise {{
      from {{
        opacity: 0;
        transform: translateY(10px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    @media (max-width: 980px) {{
      .qs-hero-grid {{
        grid-template-columns: 1fr;
      }}
      .qs-title {{
        max-width: none;
        font-size: clamp(2.2rem, 10vw, 3.4rem);
      }}
    }}
    </style>
    """


def render_app_shell(
    active_page: str,
    *,
    version: str,
    badges: Sequence[str] | None = None,
    metrics: Sequence[tuple[str, str]] | None = None,
) -> None:
    meta = _SHELLS.get(active_page, _SHELLS["trading"])
    card_metrics = list(metrics or meta.metrics)
    badge_html = "".join(
        f"<span class='qs-badge'>{escape(item)}</span>" for item in (badges or ())
    )
    metric_html = "".join(
        (
            "<div class='qs-mini-card'>"
            f"<div class='qs-mini-label'>{escape(label)}</div>"
            f"<div class='qs-mini-value'>{escape(value)}</div>"
            "</div>"
        )
        for label, value in card_metrics
    )
    shell_html = f"""
    {_shell_style(meta, active_page)}
    <section class="qs-hero">
      <div class="qs-hero-grid">
        <div>
          <div class="qs-kicker">Quant System / {escape(meta.label)}</div>
          <h1 class="qs-title">{escape(meta.title)}</h1>
          <p class="qs-subtitle">{escape(meta.subtitle)}</p>
          <div class="qs-meta-row">
            <span class="qs-badge qs-version">版本 {escape(version)}</span>
            {badge_html}
          </div>
        </div>
        <div class="qs-side-column">
          {metric_html}
        </div>
      </div>
    </section>
    """
    st.markdown(shell_html, unsafe_allow_html=True)


def render_top_nav(active_page: str) -> str:
    cols = st.columns([1.05, 1.05, 1.05, 4.2], vertical_alignment="center")
    selected = active_page
    for idx, key in enumerate(_NAV_ORDER):
        with cols[idx]:
            marker_class = "qs-top-nav-marker is-active" if key == active_page else "qs-top-nav-marker"
            st.markdown(f"<div class='{marker_class}'></div>", unsafe_allow_html=True)
            if st.button(_SHELLS[key].nav_title, key=f"qs_top_nav_{key}", use_container_width=True):
                selected = key
    return selected


def render_section_intro(
    title: str,
    description: str,
    *,
    kicker: str = "",
    pills: Sequence[str] | None = None,
) -> None:
    pill_html = "".join(
        f"<span class='qs-inline-pill'>{escape(item)}</span>" for item in (pills or ())
    )
    st.markdown(
        f"""
        <section class="qs-section-intro">
          <div class="qs-section-copy">
            <div class="qs-section-kicker">{escape(kicker)}</div>
            <h2 class="qs-section-title">{escape(title)}</h2>
            <p class="qs-section-desc">{escape(description)}</p>
          </div>
          <div class="qs-inline-pills">{pill_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(items: Sequence[tuple[str, str]]) -> None:
    cards = "".join(
        (
            "<div class='qs-status-card'>"
            f"<div class='qs-status-label'>{escape(label)}</div>"
            f"<div class='qs-status-value'>{escape(value)}</div>"
            "</div>"
        )
        for label, value in items
    )
    st.markdown(f"<div class='qs-status-row'>{cards}</div>", unsafe_allow_html=True)
