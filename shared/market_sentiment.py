from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "").replace("%", "")
    unit_mul = 1.0
    if text.endswith("亿"):
        unit_mul = 1.0
        text = text[:-1]
    elif text.endswith("万"):
        unit_mul = 0.0001
        text = text[:-1]
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0)) * unit_mul
    except Exception:
        return None


def _to_int(value: Any) -> int:
    f = _to_float(value)
    if f is None:
        return 0
    return int(round(f))


def _normalize_flow_to_yi(value: Any) -> Optional[float]:
    """
    Normalize fund flow to unit "亿".
    Heuristic:
      - with explicit 亿/万 text -> handled by _to_float
      - large plain number (likely 元) -> divide by 1e8
      - medium plain number (likely 万) -> divide by 1e4
    """
    raw = _to_float(value)
    if raw is None:
        return None
    abs_v = abs(raw)
    if abs_v >= 1_000_000:
        return raw / 100_000_000.0
    if abs_v >= 10_000 and abs_v < 1_000_000:
        return raw / 10_000.0
    return raw


@dataclass
class MarketSentimentSnapshot:
    score: float
    state: str
    state_text: str
    message: str
    warning_banner: str
    up_count: int
    down_count: int
    adl: int
    flow_value_yi: float
    flow_source: str
    limit_up_counts_3d: List[int]
    board_heights_3d: List[int]
    limit_up_avg_3d: float
    max_board_3d: int
    updated_at: str
    fetch_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketSentimentEngine:
    """
    Global market sentiment calculator.
    Score range: 0 ~ 100
      - > 70: overheat
      - < 30: panic
    """

    def __init__(self) -> None:
        try:
            import akshare as ak  # type: ignore
        except Exception:
            ak = None
        self.ak = ak

    def get_snapshot(self) -> MarketSentimentSnapshot:
        errors: List[str] = []

        up_count, down_count = self._fetch_adv_dec(errors)
        adl = int(up_count - down_count)
        total = max(up_count + down_count, 1)
        up_ratio = up_count / total

        flow_value_yi, flow_source = self._fetch_flow(errors)
        limit_ups, board_heights = self._fetch_limitup_3d(errors)

        score = self._calc_score(
            up_count=up_count,
            down_count=down_count,
            adl=adl,
            flow_value_yi=flow_value_yi,
            limit_ups=limit_ups,
            board_heights=board_heights,
        )

        if score > 70:
            state = "overheat"
            state_text = "过热"
            message = "市场情绪偏热，短线追高性价比下降。"
            warning_banner = "过热警示：情绪分数超过 70，建议降低追涨仓位并提高止盈纪律。"
        elif score < 30:
            state = "panic"
            state_text = "冰点/恐慌"
            message = "市场情绪明显偏弱，防守优先。"
            warning_banner = "当前大盘极度弱势或恐慌，系统建议强制降低预警个股 50% 的建议买入仓位。"
        else:
            state = "neutral"
            state_text = "中性"
            message = "市场情绪中性，建议以结构性机会为主。"
            warning_banner = ""

        return MarketSentimentSnapshot(
            score=round(float(score), 2),
            state=state,
            state_text=state_text,
            message=message,
            warning_banner=warning_banner,
            up_count=int(up_count),
            down_count=int(down_count),
            adl=int(adl),
            flow_value_yi=round(float(flow_value_yi), 2),
            flow_source=flow_source,
            limit_up_counts_3d=[int(x) for x in limit_ups],
            board_heights_3d=[int(x) for x in board_heights],
            limit_up_avg_3d=round(float(np.mean(limit_ups)) if limit_ups else 0.0, 2),
            max_board_3d=int(max(board_heights) if board_heights else 0),
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            fetch_error="; ".join(errors[:3]),
        )

    def _fetch_adv_dec(self, errors: List[str]) -> tuple[int, int]:
        if self.ak is None:
            errors.append("akshare unavailable")
            return 0, 0
        funcs = ("stock_zh_a_spot_em", "stock_zh_a_spot")
        for fn in funcs:
            if not hasattr(self.ak, fn):
                continue
            try:
                df = getattr(self.ak, fn)()
                if df is None or df.empty:
                    continue
                pct_col = None
                for c in ("涨跌幅", "涨跌幅(%)", "涨跌幅%", "涨跌幅(%)"):
                    if c in df.columns:
                        pct_col = c
                        break
                if pct_col is not None:
                    pct = pd.to_numeric(df[pct_col], errors="coerce")
                else:
                    close_col = next((c for c in ("最新价", "最新", "现价", "收盘") if c in df.columns), None)
                    pre_col = next((c for c in ("昨收", "昨收价", "前收盘", "昨日收盘价") if c in df.columns), None)
                    if close_col and pre_col:
                        close_s = pd.to_numeric(df[close_col], errors="coerce")
                        pre_s = pd.to_numeric(df[pre_col], errors="coerce")
                        pct = (close_s - pre_s) / pre_s * 100.0
                    else:
                        continue
                up = int((pct > 0).sum())
                down = int((pct < 0).sum())
                if up + down > 0:
                    return up, down
            except Exception as exc:
                errors.append(f"adv/dec[{fn}]: {exc}")
        return 0, 0

    def _fetch_flow(self, errors: List[str]) -> tuple[float, str]:
        if self.ak is None:
            return 0.0, "unavailable"

        # Priority 1: 北向资金净流入（亿）
        try:
            if hasattr(self.ak, "stock_hsgt_north_net_flow_in_em"):
                df = self.ak.stock_hsgt_north_net_flow_in_em()
                val = self._extract_last_flow_value(df)
                if val is not None:
                    return float(val), "northbound"
        except Exception as exc:
            errors.append(f"northbound: {exc}")

        # Priority 2: 行业主力净流入（亿）作为替代
        try:
            if hasattr(self.ak, "stock_sector_fund_flow_rank"):
                df = self.ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
                if df is not None and (not df.empty):
                    col = next((c for c in df.columns if "主力净流入" in c), None)
                    if col:
                        vals = pd.to_numeric(df[col].map(_normalize_flow_to_yi), errors="coerce")
                        vals = vals.dropna()
                        if not vals.empty:
                            # 用前10行业净流入均值做温度代理
                            return float(vals.head(10).mean()), "industry_flow"
        except Exception as exc:
            errors.append(f"industry_flow: {exc}")

        return 0.0, "fallback"

    def _extract_last_flow_value(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        candidates = [c for c in df.columns if ("净流入" in c or "净买入" in c)]
        if not candidates:
            return None
        col = candidates[0]
        series = df[col].dropna()
        if series.empty:
            return None
        return _normalize_flow_to_yi(series.iloc[-1])

    def _recent_trade_days(self, n: int = 3) -> List[str]:
        if self.ak is not None:
            try:
                if hasattr(self.ak, "tool_trade_date_hist_sina"):
                    df = self.ak.tool_trade_date_hist_sina()
                    if df is not None and (not df.empty):
                        col = "trade_date" if "trade_date" in df.columns else df.columns[0]
                        dates = pd.to_datetime(df[col], errors="coerce").dropna()
                        dates = dates[dates <= pd.Timestamp(datetime.now().date())]
                        if not dates.empty:
                            return [d.strftime("%Y%m%d") for d in dates.tail(n)]
            except Exception:
                pass

        # Fallback: 近 n 个工作日
        days: List[str] = []
        cur = datetime.now().date()
        while len(days) < n:
            if cur.weekday() < 5:
                days.append(cur.strftime("%Y%m%d"))
            cur = cur - timedelta(days=1)
        return list(reversed(days))

    def _fetch_limitup_3d(self, errors: List[str]) -> tuple[List[int], List[int]]:
        if self.ak is None:
            return [0, 0, 0], [0, 0, 0]

        days = self._recent_trade_days(3)
        limit_ups: List[int] = []
        board_heights: List[int] = []

        for day in days:
            try:
                if not hasattr(self.ak, "stock_zt_pool_em"):
                    limit_ups.append(0)
                    board_heights.append(0)
                    continue
                df = self.ak.stock_zt_pool_em(date=day)
                if df is None or df.empty:
                    limit_ups.append(0)
                    board_heights.append(0)
                    continue
                limit_ups.append(int(len(df)))
                board_heights.append(self._extract_board_height(df))
            except Exception as exc:
                errors.append(f"limitup[{day}]: {exc}")
                limit_ups.append(0)
                board_heights.append(0)

        while len(limit_ups) < 3:
            limit_ups.insert(0, 0)
            board_heights.insert(0, 0)
        return limit_ups[-3:], board_heights[-3:]

    def _extract_board_height(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        for col in ("连板数", "连板高度"):
            if col in df.columns:
                vals = pd.to_numeric(df[col].map(_to_float), errors="coerce").dropna()
                if not vals.empty:
                    return int(vals.max())
        for col in df.columns:
            if "几天几板" in str(col):
                vals = []
                for item in df[col].fillna("").astype(str):
                    m = re.search(r"(\d+)\s*天\s*(\d+)\s*板", item)
                    if m:
                        vals.append(int(m.group(2)))
                if vals:
                    return int(max(vals))
        return 0

    def _calc_score(
        self,
        *,
        up_count: int,
        down_count: int,
        adl: int,
        flow_value_yi: float,
        limit_ups: List[int],
        board_heights: List[int],
    ) -> float:
        total_raw = up_count + down_count
        total = max(total_raw, 1)
        if total_raw <= 0:
            # 主广度数据不可用时，以中性口径处理，避免误触发恐慌/过热。
            breadth_final = 50.0
        else:
            up_ratio = up_count / total
            breadth_score = float(np.clip(up_ratio * 100.0, 0.0, 100.0))
            adl_ratio = adl / total
            adl_score = float(np.clip(50.0 + adl_ratio * 100.0, 0.0, 100.0))
            breadth_final = 0.7 * breadth_score + 0.3 * adl_score

        # flow_value_yi 单位：亿。[-200, 200] 映射到 [0, 100]
        flow_score = float(np.clip((flow_value_yi + 200.0) / 4.0, 0.0, 100.0))

        avg_zt = float(np.mean(limit_ups)) if limit_ups else 0.0
        max_board = float(max(board_heights)) if board_heights else 0.0
        zt_score = float(np.clip(avg_zt / 120.0 * 100.0, 0.0, 100.0))
        board_score = float(np.clip(max_board / 10.0 * 100.0, 0.0, 100.0))
        heat_score = 0.65 * zt_score + 0.35 * board_score

        score = 0.5 * breadth_final + 0.25 * flow_score + 0.25 * heat_score
        return float(np.clip(score, 0.0, 100.0))
