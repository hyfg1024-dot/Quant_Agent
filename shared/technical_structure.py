from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TechnicalStructureAnalyzer:
    """多周期技术结构分析器（底部结构）。"""

    WEIGHTS: Dict[str, float] = {
        "weekly_rsi_divergence": 25.0,
        "monthly_macd_cross": 20.0,
        "daily_volume_stable": 20.0,
        "ma_convergence": 15.0,
        "bollinger_support": 10.0,
        "bottom_volume_burst": 10.0,
    }

    def analyze_bottom_structure(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        df = self._prepare_daily(daily_df)
        if df.empty:
            return {
                "bottom_score": 0.0,
                "structure_grade": "无底部结构",
                "signals": [],
                "summary": "日线数据为空，无法识别底部结构。",
            }

        signals: List[Dict[str, Any]] = []
        weighted_sum = 0.0
        available_weight = 0.0

        # a) 周线 RSI 底背离
        a_score, a_note = self._signal_weekly_rsi_divergence(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="weekly_rsi_divergence",
            name="周线RSI底背离",
            signal_score=a_score,
            note=a_note,
        )

        # b) 月线 MACD 金叉或即将金叉（不足120日线时降级跳过）
        b_score, b_note = self._signal_monthly_macd(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="monthly_macd_cross",
            name="月线MACD",
            signal_score=b_score,
            note=b_note,
        )

        # c) 日线缩量企稳
        c_score, c_note = self._signal_daily_volume_stable(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="daily_volume_stable",
            name="日线缩量企稳",
            signal_score=c_score,
            note=c_note,
        )

        # d) 均线收敛
        d_score, d_note = self._signal_ma_convergence(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="ma_convergence",
            name="均线收敛",
            signal_score=d_score,
            note=d_note,
        )

        # e) 布林带下轨支撑
        e_score, e_note = self._signal_bollinger_support(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="bollinger_support",
            name="布林带下轨支撑",
            signal_score=e_score,
            note=e_note,
        )

        # f) 成交量底部放量
        f_score, f_note = self._signal_bottom_volume_burst(df)
        weighted_sum, available_weight = self._accumulate_signal(
            signals,
            weighted_sum,
            available_weight,
            key="bottom_volume_burst",
            name="成交量底部放量",
            signal_score=f_score,
            note=f_note,
        )

        if available_weight > 0:
            bottom_score = float(np.clip(weighted_sum * 100.0 / available_weight, 0.0, 100.0))
        else:
            bottom_score = 0.0
        bottom_score = round(bottom_score, 2)

        if bottom_score > 70:
            grade = "强底部"
        elif bottom_score > 50:
            grade = "弱底部"
        else:
            grade = "无底部结构"

        hit_count = sum(1 for s in signals if s.get("hit"))
        checked_count = sum(1 for s in signals if not s.get("skipped"))
        skipped = [s["name"] for s in signals if s.get("skipped")]
        summary = f"触发 {hit_count}/{checked_count} 项信号，底部结构评分 {bottom_score:.2f}，判定：{grade}。"
        if skipped:
            summary += f" 已降级跳过：{'、'.join(skipped)}。"

        return {
            "bottom_score": bottom_score,
            "structure_grade": grade,
            "signals": signals,
            "summary": summary,
        }

    @staticmethod
    def _accumulate_signal(
        signals: List[Dict[str, Any]],
        weighted_sum: float,
        available_weight: float,
        key: str,
        name: str,
        signal_score: Optional[float],
        note: str,
    ) -> Tuple[float, float]:
        weight = float(TechnicalStructureAnalyzer.WEIGHTS[key])
        if signal_score is None:
            signals.append(
                {
                    "key": key,
                    "name": name,
                    "weight": weight,
                    "score": None,
                    "hit": False,
                    "skipped": True,
                    "note": note,
                }
            )
            return weighted_sum, available_weight

        score = float(np.clip(signal_score, 0.0, 100.0))
        weighted_sum += score * weight / 100.0
        available_weight += weight
        signals.append(
            {
                "key": key,
                "name": name,
                "weight": weight,
                "score": round(score, 2),
                "hit": score > 0,
                "skipped": False,
                "note": note,
            }
        )
        return weighted_sum, available_weight

    @staticmethod
    def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        out = df.copy()
        if "date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index().rename(columns={"index": "date"})
            else:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return out[["date", "open", "high", "low", "close", "volume"]]

    def _signal_weekly_rsi_divergence(self, df: pd.DataFrame) -> Tuple[float, str]:
        if len(df) < 60:
            return 0.0, "样本不足60日，未见周线底背离。"

        cutoff_date = df["date"].iloc[-60]
        weekly = self._daily_to_weekly(df)
        if weekly.empty or len(weekly) < 3:
            return 0.0, "周线样本不足，未见底背离。"
        weekly["rsi14"] = self._compute_rsi(weekly["close"], period=14)

        recent = weekly[weekly["date"] >= cutoff_date].copy()
        if recent.empty or len(recent) < 2:
            return 0.0, "最近60日对应周线不足，未见底背离。"

        latest_close = float(recent["close"].iloc[-1])
        prev_price_low = recent["close"].iloc[:-1].min()
        latest_rsi = recent["rsi14"].iloc[-1]
        prev_rsi_low = recent["rsi14"].iloc[:-1].min()

        if pd.notna(prev_price_low) and latest_close < float(prev_price_low):
            if pd.notna(latest_rsi) and pd.notna(prev_rsi_low) and float(latest_rsi) > float(prev_rsi_low):
                return 85.0, "价格创新低但RSI14未创新低，周线底背离成立。"

        return 0.0, "最近60日未形成周线RSI底背离。"

    def _signal_monthly_macd(self, df: pd.DataFrame) -> Tuple[Optional[float], str]:
        if len(df) < 120:
            return None, "日线不足120条，按规则跳过月线MACD信号。"

        monthly = self._daily_to_monthly(df)
        if monthly.empty or len(monthly) < 2:
            return 0.0, "月线样本不足，未触发MACD信号。"

        dif, dea, _ = self._compute_macd(monthly["close"], fast=12, slow=26, signal=9)
        last_dif, last_dea = dif.iloc[-1], dea.iloc[-1]
        prev_dif, prev_dea = dif.iloc[-2], dea.iloc[-2]
        if pd.isna(last_dif) or pd.isna(last_dea) or pd.isna(prev_dif) or pd.isna(prev_dea):
            return 0.0, "月线MACD数据无效，未触发信号。"

        if float(prev_dif) <= float(prev_dea) and float(last_dif) > float(last_dea):
            return 90.0, "月线DIF上穿DEA，MACD金叉。"

        current_gap = abs(float(last_dif) - float(last_dea))
        prev_gap = abs(float(prev_dif) - float(prev_dea))
        close_ref = abs(float(df["close"].iloc[-1]))
        threshold = close_ref * 0.005
        if current_gap < threshold and current_gap < prev_gap:
            return 60.0, "月线DIF/DEA差值收敛至0.5%股价以内，接近金叉。"

        return 0.0, "月线MACD未金叉，且未达到临近金叉条件。"

    def _signal_daily_volume_stable(self, df: pd.DataFrame) -> Tuple[float, str]:
        if len(df) < 65:
            return 0.0, "样本不足65日，无法完整判断缩量企稳。"

        vol5 = df["volume"].tail(5).mean()
        vol_ma60 = df["volume"].rolling(60, min_periods=60).mean().iloc[-1]
        if pd.isna(vol5) or pd.isna(vol_ma60) or vol_ma60 <= 0:
            return 0.0, "成交量样本不足，未触发缩量企稳。"

        recent5_low = df["low"].tail(5).min()
        prev20_low = df["low"].iloc[-25:-5].min()
        no_new_low = pd.notna(recent5_low) and pd.notna(prev20_low) and float(recent5_low) >= float(prev20_low)
        low_volume = float(vol5) < float(vol_ma60) * 0.5

        if low_volume and no_new_low:
            return 80.0, "近5日均量显著低于60日均量，且价格未再创新低。"
        return 0.0, "未同时满足缩量与企稳条件。"

    def _signal_ma_convergence(self, df: pd.DataFrame) -> Tuple[float, str]:
        close = df["close"]
        ma5 = close.rolling(5, min_periods=5).mean().iloc[-1]
        ma10 = close.rolling(10, min_periods=10).mean().iloc[-1]
        ma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
        ma60 = close.rolling(60, min_periods=60).mean().iloc[-1]
        latest_close = close.iloc[-1] if not close.empty else np.nan

        if any(pd.isna(x) for x in [ma5, ma10, ma20, ma60, latest_close]) or float(latest_close) == 0:
            return 0.0, "均线样本不足，未触发收敛信号。"

        spread = max(float(ma5), float(ma10), float(ma20), float(ma60)) - min(float(ma5), float(ma10), float(ma20), float(ma60))
        if spread < abs(float(latest_close)) * 0.05:
            return 75.0, "MA5/10/20/60间距小于5%股价，均线收敛。"
        return 0.0, "均线间距仍偏大，未形成收敛。"

    def _signal_bollinger_support(self, df: pd.DataFrame) -> Tuple[float, str]:
        close = df["close"]
        _, _, lower = self._compute_bollinger(close, period=20, std_dev=2)
        work = df.copy()
        work["bb_lower"] = lower
        work = work.dropna(subset=["bb_lower"])
        if len(work) < 3:
            return 0.0, "布林带样本不足，未触发下轨支撑。"

        recent = work.tail(10)
        touched = recent[(recent["low"] <= recent["bb_lower"]) | (recent["close"] <= recent["bb_lower"])]
        if touched.empty:
            return 0.0, "最近未触及布林带下轨。"

        touch_idx = touched.index[-1]
        latest_idx = work.index[-1]
        if latest_idx <= touch_idx:
            return 0.0, "触轨后尚未形成有效反弹。"

        touch_close = float(work.loc[touch_idx, "close"])
        latest_close = float(work.iloc[-1]["close"])
        prev_close = float(work.iloc[-2]["close"])
        if latest_close > touch_close and latest_close > prev_close:
            return 70.0, "触及布林带下轨后出现价格反弹。"
        return 0.0, "触轨后反弹力度不足。"

    def _signal_bottom_volume_burst(self, df: pd.DataFrame) -> Tuple[float, str]:
        if len(df) < 6:
            return 0.0, "样本不足，无法判断底部放量。"

        today = df.iloc[-1]
        vol_ma5_prev = df["volume"].shift(1).rolling(5, min_periods=5).mean().iloc[-1]
        if pd.isna(vol_ma5_prev) or float(vol_ma5_prev) <= 0:
            return 0.0, "5日均量基准不足，未触发放量信号。"

        vol_ok = float(today["volume"]) > float(vol_ma5_prev) * 2.0
        bullish = pd.notna(today["open"]) and pd.notna(today["close"]) and float(today["close"]) > float(today["open"])
        if vol_ok and bullish:
            return 80.0, "今日放量超过5日均量2倍且收阳线。"
        return 0.0, "未满足放量阳线条件。"

    @staticmethod
    def _daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date")
        if data.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        wk = (
            data.set_index("date")
            .resample("W-FRI")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
            .reset_index()
        )
        return wk[["date", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def _daily_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date")
        if data.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        mo = (
            data.set_index("date")
            .resample("ME")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
            .reset_index()
        )
        return mo[["date", "open", "high", "low", "close", "volume"]]

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        close = pd.to_numeric(series, errors="coerce")
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        both_zero = (avg_gain == 0) & (avg_loss == 0)
        gain_only = (avg_gain > 0) & (avg_loss == 0)
        loss_only = (avg_gain == 0) & (avg_loss > 0)
        rsi = rsi.mask(both_zero, 50.0)
        rsi = rsi.mask(gain_only, 100.0)
        rsi = rsi.mask(loss_only, 0.0)
        return rsi

    @staticmethod
    def _compute_macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        close = pd.to_numeric(series, errors="coerce")
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea
        return dif, dea, hist

    @staticmethod
    def _compute_bollinger(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        close = pd.to_numeric(series, errors="coerce")
        mid = close.rolling(period, min_periods=period).mean()
        sigma = close.rolling(period, min_periods=period).std(ddof=0)
        upper = mid + std_dev * sigma
        lower = mid - std_dev * sigma
        return mid, upper, lower
