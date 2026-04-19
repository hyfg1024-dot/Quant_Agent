import json
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html

from fast_engine import fetch_fast_panel
from shared.ui_shell import render_section_intro
from trading_data import _calc_display_change_pct, _fetch_chart_ohlcv, _is_market_open

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

