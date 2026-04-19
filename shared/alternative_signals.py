from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# 兼容从不同工作目录运行时的导入路径
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from apps.fundamental.fundamental_engine import (  # noqa: E402
    normalize_code,
    parse_cn_number,
    retry_call,
)


class AlternativeSignalAnalyzer:
    """另类数据拐点信号分析器。

    目标：在财报滞后前提供前瞻线索。
    """

    def __init__(self, min_interval_sec: float = 1.0) -> None:
        self.min_interval_sec = max(1.0, float(min_interval_sec))
        self._last_api_call_ts = 0.0

    # ---------- 基础工具 ----------
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            v = parse_cn_number(value)
            if v is None:
                return None
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None

    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
        cols = {str(c).strip().lower(): str(c) for c in df.columns}
        for name in candidates:
            hit = cols.get(str(name).strip().lower())
            if hit:
                return hit
        return ""

    @staticmethod
    def _is_hk_code(norm_code: str) -> bool:
        return len(str(norm_code or "")) == 5

    @staticmethod
    def _score_grade(score: float) -> str:
        if score >= 75:
            return "强"
        if score >= 60:
            return "较强"
        if score >= 40:
            return "中性"
        if score >= 20:
            return "偏弱"
        return "负向"

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_api_call_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self._last_api_call_ts = time.monotonic()

    def _call_ak(self, func):
        self._throttle()
        return retry_call(func, max_retries=2)

    # ---------- 信号1：董监高增减持 ----------
    def insider_trading_signal(self, code: str, days: int = 180) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "net_amount": 0.0,
            "buy_count": 0,
            "sell_count": 0,
            "score": 0.0,
            "detail": "",
        }

        norm_code = normalize_code(code)
        if self._is_hk_code(norm_code):
            out["detail"] = f"{norm_code} 为港股代码，接口不可用，已跳过"
            return out

        try:
            import akshare as ak
        except Exception:
            out["detail"] = "AkShare 不可用"
            return out

        if not hasattr(ak, "stock_inner_trade_xq"):
            out["detail"] = "接口 stock_inner_trade_xq 不可用"
            return out

        try:
            # 不同 akshare 版本签名不同：新版本通常无参数
            try:
                raw_df = self._call_ak(lambda: ak.stock_inner_trade_xq(symbol=norm_code))
            except Exception:
                raw_df = self._call_ak(lambda: ak.stock_inner_trade_xq())
        except Exception as exc:
            out["detail"] = f"董监高交易接口调用失败: {type(exc).__name__}"
            return out

        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            out["detail"] = "未获取到董监高交易数据"
            return out

        df = raw_df.copy()

        code_col = self._pick_col(df, ["股票代码", "证券代码", "代码", "symbol"])
        if code_col:
            df[code_col] = df[code_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
            df = df[df[code_col] == norm_code]

        date_col = self._pick_col(df, ["变动日期", "交易日期", "公告日期", "披露日期", "date", "DATE"])
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=max(1, int(days)))
            df = df[df[date_col].notna()]
            df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

        if df.empty:
            out["detail"] = f"近{int(days)}天无董监高交易记录"
            return out

        person_col = self._pick_col(df, ["变动人", "姓名", "高管名称", "股东名称", "holder"])
        direction_col = self._pick_col(df, ["变动方向", "变动类型", "增减", "买卖方向", "方向"])
        amount_col = self._pick_col(df, ["变动金额", "成交金额", "金额", "增减持金额"])
        shares_col = self._pick_col(df, ["变动股数", "变动数量", "成交数量", "数量", "增减持股数"])
        price_col = self._pick_col(df, ["成交均价", "均价", "价格", "成交价"])

        net_amount = 0.0
        buy_people: set[str] = set()
        sell_people: set[str] = set()

        for idx, row in df.iterrows():
            shares_val = self._safe_float(row.get(shares_col)) if shares_col else None
            amount_val = self._safe_float(row.get(amount_col)) if amount_col else None
            if amount_val is None:
                price_val = self._safe_float(row.get(price_col)) if price_col else None
                if shares_val is not None and price_val is not None:
                    amount_val = abs(shares_val) * price_val
                elif shares_val is not None:
                    amount_val = abs(shares_val)
            if amount_val is None:
                continue

            sign = 1.0
            if shares_val is not None and shares_val < 0:
                sign = -1.0
            direction_text = str(row.get(direction_col, "")).strip() if direction_col else ""
            direction_lower = direction_text.lower()
            if any(k in direction_text for k in ["减持", "卖出"]):
                sign = -1.0
            elif any(k in direction_text for k in ["增持", "买入"]):
                sign = 1.0
            elif any(k in direction_lower for k in ["sell", "decrease"]):
                sign = -1.0
            elif any(k in direction_lower for k in ["buy", "increase"]):
                sign = 1.0

            net_amount += sign * abs(float(amount_val))

            person = str(row.get(person_col, "")).strip() if person_col else ""
            if not person:
                person = f"row_{idx}"
            if sign > 0:
                buy_people.add(person)
            else:
                sell_people.add(person)

        buy_count = len(buy_people)
        sell_count = len(sell_people)

        if net_amount > 5_000_000:
            score = 80.0
        elif net_amount > 1_000_000:
            score = 60.0
        elif net_amount < 0:
            score = -40.0
        else:
            score = 20.0

        ratio_text = "inf" if sell_count == 0 and buy_count > 0 else f"{(buy_count / max(1, sell_count)):.2f}"

        out.update(
            {
                "net_amount": float(net_amount),
                "buy_count": int(buy_count),
                "sell_count": int(sell_count),
                "score": float(score),
                "detail": f"近{int(days)}天净增持{net_amount / 1e4:.2f}万元，增持人数{buy_count}，减持人数{sell_count}，增/减比{ratio_text}",
            }
        )
        return out

    # ---------- 信号2：股权质押解除 ----------
    def pledge_release_signal(self, code: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "pledge_ratio": None,
            "score": 0.0,
            "detail": "",
        }

        norm_code = normalize_code(code)
        if self._is_hk_code(norm_code):
            out["detail"] = f"{norm_code} 为港股代码，接口不可用，已跳过"
            return out

        try:
            import akshare as ak
        except Exception:
            out["detail"] = "AkShare 不可用"
            return out

        pledge_df = pd.DataFrame()
        pledge_source = ""

        # 优先使用股权质押明细（更贴近需求）
        if hasattr(ak, "stock_gpzy_pledge_ratio_detail_em"):
            try:
                one = self._call_ak(lambda: ak.stock_gpzy_pledge_ratio_detail_em())
                if isinstance(one, pd.DataFrame) and not one.empty:
                    pledge_df = one.copy()
                    pledge_source = "stock_gpzy_pledge_ratio_detail_em"
            except Exception:
                pass

        # 回退使用需求中提到的接口（对外担保，作为风险代理）
        if pledge_df.empty and hasattr(ak, "stock_cg_guarantee_cninfo"):
            try:
                start_date = (datetime.now().date() - timedelta(days=730)).strftime("%Y%m%d")
                end_date = datetime.now().date().strftime("%Y%m%d")
                one = self._call_ak(
                    lambda: ak.stock_cg_guarantee_cninfo(symbol="全部", start_date=start_date, end_date=end_date)
                )
                if isinstance(one, pd.DataFrame) and not one.empty:
                    pledge_df = one.copy()
                    pledge_source = "stock_cg_guarantee_cninfo"
            except Exception:
                pass

        if pledge_df.empty:
            out["detail"] = "质押/担保接口数据不可用"
            return out

        code_col = self._pick_col(pledge_df, ["股票代码", "证券代码", "代码"])
        if not code_col:
            out["detail"] = f"{pledge_source} 缺少代码列"
            return out

        df = pledge_df.copy()
        df[code_col] = df[code_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
        df = df[df[code_col] == norm_code]
        if df.empty:
            out["detail"] = "未找到该股票质押相关记录"
            return out

        if pledge_source == "stock_gpzy_pledge_ratio_detail_em":
            ratio_col = self._pick_col(df, ["占总股本比例", "质押比例", "质押率"])
            date_col = self._pick_col(df, ["公告日期", "变动日期", "质押开始日期", "日期"])
        else:
            ratio_col = self._pick_col(df, ["担保金融占净资产比例", "担保比例"])
            date_col = self._pick_col(df, ["公告统计区间", "公告日期", "日期"])

        if not ratio_col:
            out["detail"] = f"{pledge_source} 缺少比例字段"
            return out

        ratios: List[Tuple[pd.Timestamp, float]] = []
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
        else:
            dates = pd.to_datetime(pd.Series([datetime.now().date()] * len(df)), errors="coerce")

        for i, r in df.iterrows():
            ratio = self._safe_float(r.get(ratio_col))
            dt = dates.loc[i] if i in dates.index else pd.NaT
            if ratio is None:
                continue
            if pd.isna(dt):
                dt = pd.Timestamp(datetime.now().date())
            ratios.append((pd.Timestamp(dt), float(ratio)))

        if not ratios:
            out["detail"] = "质押比例字段无有效数值"
            return out

        ratios.sort(key=lambda x: x[0])
        ratio_values = [x[1] for x in ratios]
        latest_ratio = float(ratio_values[-1])

        decline_streak = len(ratio_values) >= 3 and (ratio_values[-1] < ratio_values[-2] < ratio_values[-3])

        if latest_ratio > 50:
            score = -30.0
        elif decline_streak:
            score = 80.0
        elif latest_ratio < 20:
            score = 70.0
        else:
            score = 20.0

        out.update(
            {
                "pledge_ratio": latest_ratio,
                "score": score,
                "detail": f"当前质押相关比例约{latest_ratio:.2f}%，{'连续下降' if decline_streak else '暂无连续下降'}（来源:{pledge_source}）",
            }
        )
        return out

    # ---------- 信号3：机构持仓变化 ----------
    @staticmethod
    def _recent_quarters(n: int = 8) -> List[str]:
        now = datetime.now()
        q = (now.month - 1) // 3 + 1
        year = now.year
        out: List[str] = []
        for _ in range(max(2, n)):
            out.append(f"{year}{q}")
            q -= 1
            if q == 0:
                q = 4
                year -= 1
        return out

    def _institution_stats_from_detail(self, df: pd.DataFrame) -> Tuple[float, set[str]]:
        if df is None or df.empty:
            return 0.0, set()
        ratio_col = self._pick_col(df, ["最新持股比例", "持股比例", "占流通股比例", "最新占流通股比例"])
        inst_col = self._pick_col(df, ["持股机构代码", "持股机构简称", "持股机构全称", "机构名称"])

        ratio_sum = 0.0
        if ratio_col:
            series = pd.to_numeric(df[ratio_col], errors="coerce")
            ratio_sum = float(series.fillna(0.0).sum())

        inst_set: set[str] = set()
        if inst_col:
            for one in df[inst_col].astype(str).tolist():
                text = str(one).strip()
                if text and text not in {"nan", "None"}:
                    inst_set.add(text)
        return ratio_sum, inst_set

    def institutional_position_signal(self, code: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "qoq_change": 0.0,
            "new_institutions": 0,
            "score": 0.0,
            "detail": "",
        }

        norm_code = normalize_code(code)
        if self._is_hk_code(norm_code):
            out["detail"] = f"{norm_code} 为港股代码，接口不可用，已跳过"
            return out

        try:
            import akshare as ak
        except Exception:
            out["detail"] = "AkShare 不可用"
            return out

        quarter_candidates = self._recent_quarters(10)
        snapshots: List[Tuple[str, float, set[str], str]] = []

        # 优先需求接口（不同版本可能不存在或签名不一致）
        if hasattr(ak, "stock_institute_hold_detail_em"):
            for q in quarter_candidates:
                try:
                    df = self._call_ak(lambda q=q: ak.stock_institute_hold_detail_em(symbol=norm_code, quarter=q))
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        ratio, inst_set = self._institution_stats_from_detail(df)
                        snapshots.append((q, ratio, inst_set, "stock_institute_hold_detail_em"))
                except Exception:
                    continue
                if len(snapshots) >= 2:
                    break

        # 回退到本机常见接口
        if len(snapshots) < 2 and hasattr(ak, "stock_institute_hold_detail"):
            for q in quarter_candidates:
                try:
                    df = self._call_ak(lambda q=q: ak.stock_institute_hold_detail(stock=norm_code, quarter=q))
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        ratio, inst_set = self._institution_stats_from_detail(df)
                        snapshots.append((q, ratio, inst_set, "stock_institute_hold_detail"))
                except Exception:
                    continue
                if len(snapshots) >= 2:
                    break

        # 再回退到机构持股一览（按季度全市场）
        if len(snapshots) < 2 and hasattr(ak, "stock_institute_hold"):
            for q in quarter_candidates:
                try:
                    df = self._call_ak(lambda q=q: ak.stock_institute_hold(symbol=q))
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    code_col = self._pick_col(df, ["证券代码", "股票代码", "代码"])
                    ratio_col = self._pick_col(df, ["持股比例", "占流通股比例"])
                    inst_num_col = self._pick_col(df, ["机构数", "机构数量"])
                    if not code_col:
                        continue
                    temp = df.copy()
                    temp[code_col] = temp[code_col].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
                    temp = temp[temp[code_col] == norm_code]
                    if temp.empty:
                        continue
                    ratio = self._safe_float(temp.iloc[0].get(ratio_col)) if ratio_col else 0.0
                    inst_num = int(self._safe_float(temp.iloc[0].get(inst_num_col)) or 0) if inst_num_col else 0
                    snapshots.append((q, float(ratio or 0.0), {f"inst_{i}" for i in range(inst_num)}, "stock_institute_hold"))
                except Exception:
                    continue
                if len(snapshots) >= 2:
                    break

        if len(snapshots) < 2:
            out["detail"] = "机构持仓数据不足（需至少2个季度）"
            return out

        curr_q, curr_ratio, curr_set, source = snapshots[0]
        prev_q, prev_ratio, prev_set, _ = snapshots[1]

        qoq_change = float(curr_ratio - prev_ratio)
        new_institutions = max(0, len(curr_set - prev_set))

        score = 20.0
        if qoq_change > 2:
            score = max(score, 75.0)
        if new_institutions > 3:
            score = max(score, 80.0)
        if qoq_change < -2:
            score = -20.0

        out.update(
            {
                "qoq_change": qoq_change,
                "new_institutions": int(new_institutions),
                "score": score,
                "detail": (
                    f"机构持仓季度变化 {prev_q}->{curr_q}: {qoq_change:+.2f}pct, "
                    f"新进机构{new_institutions}家（来源:{source}）"
                ),
            }
        )
        return out

    # ---------- 组合得分 ----------
    def composite_alternative_score(self, code: str) -> Dict[str, Any]:
        insider = self.insider_trading_signal(code=code)
        pledge = self.pledge_release_signal(code=code)
        institutional = self.institutional_position_signal(code=code)

        insider_score = float(self._safe_float(insider.get("score")) or 0.0)
        pledge_score = float(self._safe_float(pledge.get("score")) or 0.0)
        institutional_score = float(self._safe_float(institutional.get("score")) or 0.0)

        alt_score = insider_score * 0.40 + pledge_score * 0.20 + institutional_score * 0.40
        alt_score = max(-100.0, min(100.0, alt_score))

        signals = [
            {
                "name": "高管增减持",
                "score": round(insider_score, 2),
                "detail": str(insider.get("detail", "")),
            },
            {
                "name": "股权质押",
                "score": round(pledge_score, 2),
                "detail": str(pledge.get("detail", "")),
            },
            {
                "name": "机构持仓",
                "score": round(institutional_score, 2),
                "detail": str(institutional.get("detail", "")),
            },
        ]

        return {
            "alt_score": round(alt_score, 2),
            "grade": self._score_grade(alt_score),
            "signals": signals,
        }
