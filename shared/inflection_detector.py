from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# 兼容从不同工作目录运行时的导入路径
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from apps.fundamental.fundamental_engine import (  # noqa: E402
    _read_abstract,
    _read_annual_indicator,
    analyze_fundamental,
    normalize_code,
    parse_cn_number,
    retry_call,
)
try:
    from shared.knowledge_graph import IndustryKnowledgeGraph
except Exception:  # pragma: no cover
    IndustryKnowledgeGraph = None  # type: ignore[assignment]
try:
    from shared.alternative_signals import AlternativeSignalAnalyzer
except Exception:  # pragma: no cover
    AlternativeSignalAnalyzer = None  # type: ignore[assignment]


SignalResult = Dict[str, Any]


class InflectionDetector:
    """面向价值投资的盈利拐点检测器。

    说明：
    - 拐点强调“由坏转好”，并优先识别趋势改变而不是绝对水平。
    - 所有序列均要求时间正序（旧 -> 新）。
    - 任一维度数据获取失败时，该维度记 0 分，不影响其它维度计算。
    """

    SIGNAL_WEIGHTS: Dict[str, float] = {
        "revenue": 0.17,
        "margin": 0.1275,
        "fcf": 0.17,
        "leverage": 0.1275,
        "receivable": 0.085,
        "inventory": 0.085,
        "insider": 0.085,
        "alternative": 0.15,
    }

    def __init__(self, enable_propagation_tip: bool = True, propagation_top_n: int = 3) -> None:
        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}
        self._profit_growth_proxy_series: List[float] = []
        self.enable_propagation_tip = bool(enable_propagation_tip)
        self.propagation_top_n = max(1, int(propagation_top_n))
        self._kg: Any = None
        self._alt: Any = None
        if self.enable_propagation_tip and (IndustryKnowledgeGraph is not None):
            try:
                self._kg = IndustryKnowledgeGraph()
            except Exception:
                self._kg = None
        if AlternativeSignalAnalyzer is not None:
            try:
                self._alt = AlternativeSignalAnalyzer(min_interval_sec=1.0)
            except Exception:
                self._alt = None

    # ---------- 基础工具 ----------
    @staticmethod
    def _ok_signal(triggered: bool, score: float, detail: str) -> SignalResult:
        return {
            "triggered": bool(triggered),
            "score": float(max(0.0, min(100.0, score))),
            "detail": str(detail),
        }

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """将输入转为 float，失败返回 None。"""
        try:
            val = parse_cn_number(value)
            if val is None:
                return None
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None

    @classmethod
    def _clean_series(cls, series: Sequence[Any]) -> List[float]:
        """清洗序列，保留有效数字并维持原顺序。"""
        out: List[float] = []
        for x in series:
            v = cls._safe_float(x)
            if v is not None:
                out.append(v)
        return out

    @staticmethod
    def _consecutive_change_count(series: Sequence[float], improve_when: str = "up") -> int:
        """统计末端连续改善期数（相邻比较）。

        参数:
        - improve_when='up'：后值 > 前值 视为改善
        - improve_when='down'：后值 < 前值 视为改善
        """
        arr = list(series)
        if len(arr) < 2:
            return 0
        cnt = 0
        for i in range(len(arr) - 1, 0, -1):
            left = arr[i - 1]
            right = arr[i]
            improved = right > left if improve_when == "up" else right < left
            if improved:
                cnt += 1
            else:
                break
        return cnt

    @staticmethod
    def _pct_growth_series(level_series: Sequence[float]) -> List[float]:
        """由水平值序列生成增速序列（%），并保持时间正序。"""
        levels = list(level_series)
        growth: List[float] = []
        for i in range(1, len(levels)):
            prev = levels[i - 1]
            curr = levels[i]
            if prev == 0:
                continue
            growth.append((curr - prev) / abs(prev) * 100.0)
        return growth

    def _get_fundamental_snapshot(self, code: str, name: str) -> Dict[str, Any]:
        """读取基础面快照（用于总市值等信息兜底）。"""
        norm = normalize_code(code)
        if norm in self._fundamental_cache:
            return self._fundamental_cache[norm]
        try:
            snapshot = retry_call(lambda: analyze_fundamental(norm, name=name), max_retries=2) or {}
        except Exception:
            snapshot = {}
        self._fundamental_cache[norm] = snapshot if isinstance(snapshot, dict) else {}
        return self._fundamental_cache[norm]

    # ---------- 数据提取 ----------
    @staticmethod
    def _date_cols(df: pd.DataFrame) -> List[str]:
        """提取财务摘要中的日期列并按时间正序排序。"""
        cols = [str(c) for c in df.columns if re.fullmatch(r"\d{8}", str(c))]
        cols.sort()
        return cols

    @classmethod
    def _read_abstract_series(cls, df: pd.DataFrame, indicator_names: Sequence[str]) -> List[float]:
        """按指标名从财务摘要提取时间正序序列。"""
        if df is None or df.empty or "指标" not in df.columns:
            return []
        date_cols = cls._date_cols(df)
        if not date_cols:
            return []

        row = pd.DataFrame()
        for name in indicator_names:
            hit = df[df["指标"].astype(str).str.strip() == str(name).strip()]
            if not hit.empty:
                row = hit.iloc[[0]]
                break
        if row.empty:
            return []

        values = [cls._safe_float(row.iloc[0].get(col)) for col in date_cols]
        return cls._clean_series(values)

    @classmethod
    def _read_abstract_sum_series(cls, df: pd.DataFrame, indicator_names: Sequence[str]) -> List[float]:
        """将多个指标按期求和并返回时间正序序列。"""
        if df is None or df.empty or "指标" not in df.columns:
            return []
        date_cols = cls._date_cols(df)
        if not date_cols:
            return []

        rows: List[pd.Series] = []
        for name in indicator_names:
            hit = df[df["指标"].astype(str).str.strip() == str(name).strip()]
            if not hit.empty:
                rows.append(hit.iloc[0])
        if not rows:
            return []

        out: List[float] = []
        for col in date_cols:
            period_sum = 0.0
            has_value = False
            for row in rows:
                v = cls._safe_float(row.get(col))
                if v is not None:
                    period_sum += float(v)
                    has_value = True
            if has_value:
                out.append(period_sum)
        return out

    @classmethod
    def _read_indicator_series(cls, df: pd.DataFrame, candidate_cols: Sequence[str]) -> List[float]:
        """按列名从年度指标表提取时间正序序列。"""
        if df is None or df.empty:
            return []
        hit_col = ""
        for col in candidate_cols:
            if col in df.columns:
                hit_col = col
                break
        if not hit_col:
            return []

        temp = df.copy()
        if "报告期" in temp.columns:
            temp["报告期"] = pd.to_datetime(temp["报告期"], errors="coerce")
            temp = temp.sort_values("报告期", ascending=True)
        values = [cls._safe_float(v) for v in temp[hit_col].tolist()]
        return cls._clean_series(values)

    # ---------- 七维信号 ----------
    def detect_revenue_inflection(self, revenue_series: List[float]) -> SignalResult:
        """检测营收拐点：同比增速由负转正，或连续两期加速增长。"""
        revenue = self._clean_series(revenue_series)
        if len(revenue) < 3:
            return self._ok_signal(False, 0, "营收序列不足3期，无法判断拐点")

        growth = self._pct_growth_series(revenue)
        if len(growth) < 2:
            return self._ok_signal(False, 0, "营收增速序列不足，无法判断拐点")

        neg_to_pos = growth[-1] > 0 and growth[-2] <= 0
        accel_two = len(growth) >= 3 and growth[-1] > growth[-2] > growth[-3]
        decel_but_positive = len(growth) >= 2 and growth[-1] > 0 and growth[-1] < growth[-2]

        if neg_to_pos:
            return self._ok_signal(True, 80, f"营收增速由负转正：{growth[-2]:.2f}% -> {growth[-1]:.2f}%")
        if accel_two:
            return self._ok_signal(True, 60, f"营收增速连续2期加速：{growth[-3]:.2f}% -> {growth[-2]:.2f}% -> {growth[-1]:.2f}%")
        if decel_but_positive:
            return self._ok_signal(False, 30, f"营收仍增长但边际放缓：{growth[-2]:.2f}% -> {growth[-1]:.2f}%")
        return self._ok_signal(False, 0, "未观察到营收拐点信号")

    def detect_margin_improvement(self, margin_series: List[float]) -> SignalResult:
        """检测毛利率改善：连续2期环比改善。"""
        margin = self._clean_series(margin_series)
        if len(margin) < 3:
            return self._ok_signal(False, 0, "毛利率序列不足3期，无法判断趋势")

        improve_cnt = self._consecutive_change_count(margin, improve_when="up")
        if improve_cnt >= 3:
            return self._ok_signal(True, 90, f"毛利率连续{improve_cnt}期改善，最新{margin[-1]:.2f}%")
        if improve_cnt >= 2:
            return self._ok_signal(True, 70, f"毛利率连续2期改善，最新{margin[-1]:.2f}%")
        if improve_cnt >= 1:
            return self._ok_signal(False, 40, f"毛利率单期改善，最新{margin[-2]:.2f}% -> {margin[-1]:.2f}%")
        return self._ok_signal(False, 0, "毛利率未出现改善")

    def detect_fcf_turnaround(self, ocf_series: List[float], capex_series: Optional[List[float]]) -> SignalResult:
        """检测自由现金流拐点：FCF(经营现金流-资本开支)由负转正。

        无 capex 时，使用经营现金流 OCF 作为近似口径。
        """
        ocf = self._clean_series(ocf_series)
        if len(ocf) < 3:
            return self._ok_signal(False, 0, "经营现金流序列不足3期，无法判断趋势")

        use_capex = capex_series is not None and len(self._clean_series(capex_series)) >= 2
        if use_capex:
            capex = self._clean_series(capex_series or [])
            m = min(len(ocf), len(capex))
            if m < 3:
                return self._ok_signal(False, 0, "FCF序列不足3期，无法判断趋势")
            fcf = [ocf[-m + i] - abs(capex[-m + i]) for i in range(m)]
            basis = "FCF"
        else:
            fcf = ocf
            basis = "OCF代理"

        if len(fcf) < 3:
            return self._ok_signal(False, 0, "现金流序列不足3期，无法判断趋势")

        if fcf[-1] > 0 and fcf[-2] <= 0:
            return self._ok_signal(True, 85, f"{basis}由负转正：{fcf[-2]:.2f} -> {fcf[-1]:.2f}")
        if len(fcf) >= 3 and fcf[-1] > fcf[-2] > fcf[-3]:
            return self._ok_signal(False, 50, f"{basis}连续改善但未发生本期转正")
        return self._ok_signal(False, 0, f"{basis}未出现由负转正")

    def detect_operating_leverage(
        self,
        revenue_growth_series: List[float],
        expense_growth_series: Optional[List[float]],
    ) -> SignalResult:
        """检测经营杠杆释放：营收增速 > 费用增速。

        若费用增速缺失，自动使用“利润增速 > 营收增速”作为代理判断。
        """
        rev_g = self._clean_series(revenue_growth_series)
        if len(rev_g) < 2:
            return self._ok_signal(False, 0, "营收增速序列不足（至少需要3期收入数据）")

        exp_g = self._clean_series(expense_growth_series or [])
        if exp_g:
            m = min(len(rev_g), len(exp_g))
            if m == 0:
                return self._ok_signal(False, 0, "经营杠杆序列不足")
            diff = rev_g[-m:][-1] - exp_g[-m:][-1]
            if diff > 0:
                score = 80 if diff >= 5 else 65
                return self._ok_signal(True, score, f"营收增速高于费用增速：{rev_g[-1]:.2f}% > {exp_g[-1]:.2f}%")
            return self._ok_signal(False, 0, f"费用增速高于营收增速：{exp_g[-1]:.2f}% >= {rev_g[-1]:.2f}%")

        proxy = self._clean_series(self._profit_growth_proxy_series)
        if not proxy:
            return self._ok_signal(False, 0, "费用与利润增速均缺失，无法判断经营杠杆")

        m = min(len(rev_g), len(proxy))
        if m == 0:
            return self._ok_signal(False, 0, "经营杠杆代理序列不足")
        diff = proxy[-m:][-1] - rev_g[-m:][-1]
        if diff > 0:
            score = 70 if diff >= 5 else 55
            return self._ok_signal(True, score, f"利润增速代理高于营收增速：{proxy[-1]:.2f}% > {rev_g[-1]:.2f}%")
        return self._ok_signal(False, 0, f"利润增速代理未跑赢营收增速：{proxy[-1]:.2f}% <= {rev_g[-1]:.2f}%")

    def detect_receivable_improvement(self, receivable_days_series: List[float]) -> SignalResult:
        """检测应收账款周转改善：应收天数连续下降。"""
        days = self._clean_series(receivable_days_series)
        if len(days) < 3:
            return self._ok_signal(False, 0, "应收周转天数序列不足3期，无法判断趋势")

        improve_cnt = self._consecutive_change_count(days, improve_when="down")
        if improve_cnt >= 3:
            return self._ok_signal(True, 85, f"应收天数连续{improve_cnt}期下降，最新{days[-1]:.2f}天")
        if improve_cnt >= 2:
            return self._ok_signal(True, 70, f"应收天数连续2期下降，最新{days[-1]:.2f}天")
        if improve_cnt >= 1:
            return self._ok_signal(False, 40, f"应收天数单期下降：{days[-2]:.2f} -> {days[-1]:.2f}天")
        return self._ok_signal(False, 0, "应收天数未改善")

    def detect_inventory_clearing(self, inventory_revenue_ratio_series: List[float]) -> SignalResult:
        """检测库存去化：存货/营收比连续下降。"""
        ratio = self._clean_series(inventory_revenue_ratio_series)
        if len(ratio) < 3:
            return self._ok_signal(False, 0, "存货/营收比序列不足3期，无法判断趋势")

        improve_cnt = self._consecutive_change_count(ratio, improve_when="down")
        if improve_cnt >= 3:
            return self._ok_signal(True, 85, f"存货/营收比连续{improve_cnt}期下降，最新{ratio[-1]:.4f}")
        if improve_cnt >= 2:
            return self._ok_signal(True, 70, f"存货/营收比连续2期下降，最新{ratio[-1]:.4f}")
        if improve_cnt >= 1:
            return self._ok_signal(False, 40, f"存货/营收比单期下降：{ratio[-2]:.4f} -> {ratio[-1]:.4f}")
        return self._ok_signal(False, 0, "存货/营收比未改善")

    def detect_insider_activity(self, code: str) -> SignalResult:
        """检测近6个月董监高增减持。

        - 优先尝试 `ak.stock_inner_trade_xq`；
        - 失败时再尝试 `ak.stock_ggcg_em`；
        - 若最终不可用，返回默认空信号。
        """
        try:
            import akshare as ak
        except Exception:
            return self._ok_signal(False, 0, "数据不可用")

        norm_code = normalize_code(code)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=183)

        fetchers = []
        if hasattr(ak, "stock_inner_trade_xq"):
            fetchers.append(lambda: ak.stock_inner_trade_xq(symbol=norm_code))
        if hasattr(ak, "stock_ggcg_em"):
            fetchers.append(lambda: ak.stock_ggcg_em(symbol=norm_code))

        raw_df = pd.DataFrame()
        for fn in fetchers:
            try:
                one = retry_call(fn, max_retries=2)
                if isinstance(one, pd.DataFrame) and not one.empty:
                    raw_df = one.copy()
                    break
            except Exception:
                continue
        if raw_df.empty:
            return self._ok_signal(False, 0, "数据不可用")

        date_col = self._pick_col(
            raw_df,
            ["变动日期", "日期", "交易日期", "公告日期", "披露日期", "DATE", "date"],
        )
        direction_col = self._pick_col(
            raw_df,
            ["变动方向", "变动类型", "增减", "增减持方向", "买卖方向", "方向"],
        )
        amount_col = self._pick_col(
            raw_df,
            ["变动金额", "成交金额", "金额", "增减持金额", "交易金额"],
        )
        shares_col = self._pick_col(
            raw_df,
            ["变动股数", "变动数量", "成交数量", "数量", "增减持股数", "增减持数量"],
        )
        price_col = self._pick_col(raw_df, ["成交均价", "均价", "价格", "成交价"])

        if not date_col:
            return self._ok_signal(False, 0, "数据不可用")

        df = raw_df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
        df = df[df[date_col].notna()]
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        if df.empty:
            return self._ok_signal(False, 0, "近6个月无董监高交易记录")

        net_amount = 0.0
        for _, row in df.iterrows():
            amount = self._safe_float(row.get(amount_col)) if amount_col else None
            if amount is None:
                shares = self._safe_float(row.get(shares_col)) if shares_col else None
                price = self._safe_float(row.get(price_col)) if price_col else None
                if shares is not None and price is not None:
                    amount = abs(shares) * price
                elif shares is not None:
                    amount = abs(shares)
            if amount is None:
                continue

            sign = 1.0
            direction = str(row.get(direction_col, "")).strip() if direction_col else ""
            direction_lower = direction.lower()
            if any(k in direction for k in ["减持", "卖出"]) or any(k in direction_lower for k in ["sell", "decrease"]):
                sign = -1.0
            elif any(k in direction for k in ["增持", "买入"]) or any(k in direction_lower for k in ["buy", "increase"]):
                sign = 1.0
            else:
                shares_value = self._safe_float(row.get(shares_col)) if shares_col else None
                if shares_value is not None and shares_value < 0:
                    sign = -1.0
            net_amount += sign * abs(float(amount))

        if net_amount <= 0:
            return self._ok_signal(False, 0, f"近6个月净减持/持平，净额约{net_amount / 1e8:.2f}亿元")

        snapshot = self._get_fundamental_snapshot(norm_code, norm_code)
        total_mv = self._safe_float(snapshot.get("total_mv")) if isinstance(snapshot, dict) else None
        if total_mv and total_mv > 0:
            ratio = net_amount / total_mv
            if ratio >= 0.01:
                score = 100
            elif ratio >= 0.005:
                score = 90
            elif ratio >= 0.002:
                score = 75
            elif ratio >= 0.001:
                score = 60
            else:
                score = 40
            detail = f"近6个月净增持约{net_amount / 1e8:.2f}亿元，占总市值约{ratio * 100:.3f}%"
            return self._ok_signal(True, score, detail)

        return self._ok_signal(False, 0, "数据不可用")

    def detect_alternative_data_signal(self, code: str) -> SignalResult:
        """检测另类数据综合信号。"""
        if self._alt is None:
            return self._ok_signal(False, 0, "另类数据模块不可用")

        try:
            result = self._alt.composite_alternative_score(code=normalize_code(code))
        except Exception as exc:
            return self._ok_signal(False, 0, f"另类数据抓取失败: {type(exc).__name__}")

        score = self._safe_float(result.get("alt_score")) or 0.0
        score = float(max(-100.0, min(100.0, score)))
        grade = str(result.get("grade", "") or "")

        parts: List[str] = []
        signals = result.get("signals", [])
        if isinstance(signals, list):
            for one in signals:
                if not isinstance(one, dict):
                    continue
                name = str(one.get("name", "") or "")
                sc = self._safe_float(one.get("score"))
                detail = str(one.get("detail", "") or "")
                if name:
                    if sc is None:
                        parts.append(f"{name}: {detail}")
                    else:
                        parts.append(f"{name}{sc:.1f}分，{detail}")

        detail = f"另类数据综合分 {score:.2f}"
        if grade:
            detail += f"（{grade}）"
        if parts:
            detail += "；" + " | ".join(parts)

        return {
            "triggered": bool(score >= 60.0),
            "score": score,
            "detail": detail,
        }

    # ---------- 综合评分 ----------
    @staticmethod
    def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
        cols = {str(c).strip().lower(): str(c) for c in df.columns}
        for c in candidates:
            hit = cols.get(str(c).strip().lower())
            if hit:
                return hit
        return ""

    @staticmethod
    def _extract_trigger_signal_name(signals: Any) -> str:
        if not isinstance(signals, list) or not signals:
            return "拐点信号"
        best = None
        best_score = -1.0
        for one in signals:
            if not isinstance(one, dict):
                continue
            sc = 0.0
            try:
                sc = float(one.get("score", 0.0) or 0.0)
            except Exception:
                sc = 0.0
            if sc > best_score:
                best_score = sc
                best = one
        if isinstance(best, dict):
            return str(best.get("name", "拐点信号") or "拐点信号")
        return "拐点信号"

    def _build_propagation_tip(self, code: str, trigger_signal: str) -> str:
        if self._kg is None:
            return ""
        norm_code = normalize_code(code)
        if not norm_code:
            return ""
        try:
            return str(
                self._kg.format_propagation_tip(
                    trigger_code=norm_code,
                    trigger_signal=str(trigger_signal or "拐点信号"),
                    top_n=self.propagation_top_n,
                )
                or ""
            ).strip()
        except Exception:
            return ""

    @classmethod
    def _inventory_revenue_ratio_series(cls, abstract_df: pd.DataFrame) -> List[float]:
        inventory = cls._read_abstract_series(abstract_df, ["存货", "存货净额"])
        revenue = cls._read_abstract_series(abstract_df, ["营业总收入", "营业收入"])
        if not inventory or not revenue:
            return []
        m = min(len(inventory), len(revenue))
        ratios: List[float] = []
        inv_cut = inventory[-m:]
        rev_cut = revenue[-m:]
        for i in range(m):
            if rev_cut[i] == 0:
                continue
            ratios.append(inv_cut[i] / abs(rev_cut[i]))
        return ratios

    def compute_inflection_score(self, code: str, name: str) -> Dict[str, Any]:
        """计算单只股票的盈利拐点评分。"""
        norm_code = normalize_code(code)
        stock_name = str(name or norm_code)

        abstract_df = pd.DataFrame()
        indicator_df = pd.DataFrame()
        try:
            abstract_df = retry_call(lambda: _read_abstract(norm_code), max_retries=2)
        except Exception:
            abstract_df = pd.DataFrame()
        try:
            indicator_df = retry_call(lambda: _read_annual_indicator(norm_code), max_retries=2)
        except Exception:
            indicator_df = pd.DataFrame()

        snapshot = self._get_fundamental_snapshot(norm_code, stock_name)
        if isinstance(snapshot, dict) and snapshot.get("name"):
            stock_name = str(snapshot.get("name"))

        revenue_series = self._read_abstract_series(abstract_df, ["营业总收入", "营业收入"])
        ocf_series = self._read_abstract_series(abstract_df, ["经营现金流量净额", "经营活动产生的现金流量净额"])
        capex_series = self._read_abstract_series(
            abstract_df,
            [
                "购建固定资产、无形资产和其他长期资产支付的现金",
                "资本开支",
                "购建固定资产无形资产和其他长期资产所支付的现金",
            ],
        )
        profit_series = self._read_abstract_series(abstract_df, ["归母净利润", "净利润", "归属于母公司股东的净利润"])
        margin_series = self._read_indicator_series(indicator_df, ["销售毛利率", "销售毛利率(%)", "毛利率", "毛利率(%)"])
        receivable_days_series = self._read_indicator_series(indicator_df, ["应收账款周转天数"])
        inventory_ratio_series = self._inventory_revenue_ratio_series(abstract_df)

        expense_series = self._read_abstract_sum_series(
            abstract_df,
            ["销售费用", "管理费用", "研发费用", "财务费用", "营业总成本"],
        )

        revenue_growth_series = self._pct_growth_series(revenue_series)
        profit_growth_series = self._pct_growth_series(profit_series)
        expense_growth_series = self._pct_growth_series(expense_series) if expense_series else []
        self._profit_growth_proxy_series = profit_growth_series

        signals: List[Dict[str, Any]] = []

        revenue_sig = self.detect_revenue_inflection(revenue_series)
        margin_sig = self.detect_margin_improvement(margin_series)
        fcf_sig = self.detect_fcf_turnaround(ocf_series, capex_series if capex_series else None)
        lev_sig = self.detect_operating_leverage(
            revenue_growth_series,
            expense_growth_series if expense_growth_series else None,
        )
        recv_sig = self.detect_receivable_improvement(receivable_days_series)
        inv_sig = self.detect_inventory_clearing(inventory_ratio_series)
        insider_sig = self.detect_insider_activity(norm_code)
        alt_sig = self.detect_alternative_data_signal(norm_code)

        signal_map: List[Tuple[str, str, SignalResult]] = [
            ("revenue", "营收拐点", revenue_sig),
            ("margin", "毛利率改善", margin_sig),
            ("fcf", "FCF拐点", fcf_sig),
            ("leverage", "经营杠杆释放", lev_sig),
            ("receivable", "应收改善", recv_sig),
            ("inventory", "库存去化", inv_sig),
            ("insider", "董监高增减持", insider_sig),
            ("alternative", "另类数据", alt_sig),
        ]

        weighted_score = 0.0
        for key, title, sig in signal_map:
            w = self.SIGNAL_WEIGHTS[key]
            sc = float(sig.get("score", 0.0) or 0.0)
            weighted_score += sc * w
            signals.append(
                {
                    "name": title,
                    "triggered": bool(sig.get("triggered", False)),
                    "score": round(sc, 2),
                    "weight": float(w),
                    "detail": str(sig.get("detail", "")),
                }
            )

        inflection_score = round(max(0.0, min(100.0, weighted_score)), 2)
        if inflection_score > 70:
            grade = "强拐点"
        elif inflection_score > 50:
            grade = "弱拐点"
        else:
            grade = "无明显拐点"

        triggered_count = sum(1 for s in signals if s["triggered"])
        top_signal = max(signals, key=lambda x: float(x.get("score", 0.0))) if signals else None
        if top_signal:
            summary = (
                f"{stock_name}({norm_code}) 拐点评分 {inflection_score:.2f}，"
                f"{grade}；触发{triggered_count}/8项，最强信号为{top_signal['name']}。"
            )
        else:
            summary = f"{stock_name}({norm_code}) 拐点评分 {inflection_score:.2f}，{grade}。"

        return {
            "code": norm_code,
            "name": stock_name,
            "inflection_score": inflection_score,
            "inflection_grade": grade,
            "signals": signals,
            "summary": summary,
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def batch_scan(self, codes: List[Tuple[str, str]], min_score: float = 50) -> pd.DataFrame:
        """批量扫描拐点股票。

        参数:
        - codes: [(code, name), ...]
        - min_score: 最低拐点评分过滤阈值
        """
        rows: List[Dict[str, Any]] = []
        for code, name in codes:
            try:
                one = self.compute_inflection_score(code, name)
            except Exception as exc:
                one = {
                    "code": normalize_code(code),
                    "name": str(name or code),
                    "inflection_score": 0.0,
                    "inflection_grade": "无明显拐点",
                    "signals": [],
                    "summary": f"计算失败: {type(exc).__name__}: {exc}",
                    "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            rows.append(one)

        if not rows:
            return pd.DataFrame(columns=["code", "name", "inflection_score", "inflection_grade", "summary", "computed_at"])

        df = pd.DataFrame(rows)
        if "inflection_score" in df.columns:
            df["inflection_score"] = pd.to_numeric(df["inflection_score"], errors="coerce").fillna(0.0)
            df = df[df["inflection_score"] >= float(min_score)]
            df = df.sort_values("inflection_score", ascending=False).reset_index(drop=True)

        if self.enable_propagation_tip and (not df.empty):
            rows: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                one = row.to_dict()
                trigger_signal = self._extract_trigger_signal_name(one.get("signals", []))
                tip = self._build_propagation_tip(str(one.get("code", "")), trigger_signal)
                one["propagation_tip"] = tip
                one["summary_with_propagation"] = (
                    f"{one.get('summary', '')}\n\n【产业链传导提示】\n{tip}" if tip else str(one.get("summary", "") or "")
                )
                rows.append(one)
            df = pd.DataFrame(rows)
        return df

    def build_daily_report(self, codes: List[Tuple[str, str]], min_score: float = 50) -> str:
        """生成拐点日报（附产业链传导提示）。"""
        df = self.batch_scan(codes=codes, min_score=min_score)
        if df.empty:
            return "今日无达到阈值的拐点标的。"

        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"## 拐点日报 ({now_text})", ""]
        for i, (_, row) in enumerate(df.iterrows(), 1):
            code = str(row.get("code", ""))
            name = str(row.get("name", ""))
            score = float(row.get("inflection_score", 0.0) or 0.0)
            grade = str(row.get("inflection_grade", "无明显拐点"))
            summary = str(row.get("summary", "") or "")
            tip = str(row.get("propagation_tip", "") or "").strip()
            lines.append(f"### {i}. {name}({code})")
            lines.append(f"- 拐点评分: {score:.2f} | 等级: {grade}")
            if summary:
                lines.append(f"- 概要: {summary}")
            if tip:
                lines.append("- 产业链传导提示:")
                lines.append(tip)
            lines.append("")
        return "\n".join(lines).strip()


if __name__ == "__main__":
    """示例测试：
    1) 纯函数信号测试（不依赖外部数据）
    2) 单票拐点评分测试（依赖 AKShare 网络环境）
    """
    detector = InflectionDetector()

    print("=== 纯函数信号测试 ===")
    print("营收拐点:", detector.detect_revenue_inflection([100, 95, 98, 110]))
    print("毛利率改善:", detector.detect_margin_improvement([18.0, 17.5, 18.2, 19.1]))
    print("FCF拐点:", detector.detect_fcf_turnaround([1.2e8, -0.5e8, 0.3e8], [2.0e8, 1.5e8, 0.8e8]))
    print("经营杠杆:", detector.detect_operating_leverage([5, 8, 12], [9, 10, 8]))
    print("应收改善:", detector.detect_receivable_improvement([88, 83, 79, 72]))
    print("库存去化:", detector.detect_inventory_clearing([0.42, 0.39, 0.35, 0.31]))

    print("\n=== 单票综合评分示例（需联网） ===")
    try:
        one = detector.compute_inflection_score("600519", "贵州茅台")
        print(one["summary"])
        print("总分:", one["inflection_score"], "等级:", one["inflection_grade"])
    except Exception as e:
        print("示例失败（通常为数据源不可用）:", e)
