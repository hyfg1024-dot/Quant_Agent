from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd


APP_VERSION = "FND-20260323-02"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


DEFAULT_WATCHLIST = [
    {"code": "600007", "name": "中国国贸", "type": "观察"},
    {"code": "601088", "name": "中国神华", "type": "持仓"},
    {"code": "603871", "name": "嘉友国际", "type": "观察"},
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_code(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) == 5:  # 港股 5 位
        return digits
    if len(digits) >= 6:
        return digits[-6:]
    return digits


def parse_cn_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if text in {"", "-", "--", "nan", "None", "False"}:
        return None

    negative = text.startswith("-")
    text = text.replace(",", "").replace("+", "")
    text = text.replace("亿元", "亿").replace("万元", "万")

    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("万亿"):
        multiplier = 1e12
        text = text[:-2]
    elif text.endswith("亿"):
        multiplier = 1e8
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 1e4
        text = text[:-1]

    try:
        num = float(text)
    except ValueError:
        return None

    if negative:
        num = -abs(num)
    return num * multiplier


def format_num(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def format_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}%"


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in {None, 0}:
        return None
    return a / b


def retry_call(func, max_retries: int = 3):
    for i in range(max_retries):
        try:
            return func()
        except Exception:
            if i == max_retries - 1:
                raise
            time.sleep((2 ** i) + random.uniform(0.1, 0.6))
    return None


def _cache_file(code: str) -> Path:
    return CACHE_DIR / f"analysis_{code}.json"


def load_watchlist() -> List[Dict[str, str]]:
    ensure_dirs()
    if not WATCHLIST_FILE.exists():
        WATCHLIST_FILE.write_text(
            json.dumps(DEFAULT_WATCHLIST, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return DEFAULT_WATCHLIST.copy()
    try:
        rows = json.loads(WATCHLIST_FILE.read_text(encoding="utf-8"))
        cleaned: List[Dict[str, str]] = []
        for item in rows:
            code = normalize_code(str(item.get("code", "")))
            if not code:
                continue
            cleaned.append(
                {
                    "code": code,
                    "name": str(item.get("name", "")).strip() or code,
                    "type": str(item.get("type", "观察")).strip() or "观察",
                }
            )
        return cleaned or DEFAULT_WATCHLIST.copy()
    except Exception:
        return DEFAULT_WATCHLIST.copy()


def save_watchlist(rows: List[Dict[str, str]]) -> None:
    ensure_dirs()
    WATCHLIST_FILE.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def upsert_watch_item(code: str, name: str, item_type: str) -> List[Dict[str, str]]:
    rows = load_watchlist()
    norm_code = normalize_code(code)
    if not norm_code:
        return rows
    found = False
    for item in rows:
        if item["code"] == norm_code:
            item["name"] = name or item["name"]
            item["type"] = item_type
            found = True
            break
    if not found:
        rows.append({"code": norm_code, "name": name or norm_code, "type": item_type})
    save_watchlist(rows)
    return rows


def delete_watch_item(code: str) -> List[Dict[str, str]]:
    norm_code = normalize_code(code)
    rows = [x for x in load_watchlist() if x["code"] != norm_code]
    save_watchlist(rows)
    return rows


def _latest_annual_columns(df: pd.DataFrame, n: int = 6) -> List[str]:
    cols = [c for c in df.columns if re.fullmatch(r"\d{8}", str(c))]
    cols = [c for c in cols if str(c).endswith("1231")]
    cols.sort(reverse=True)
    return cols[:n]


def _read_abstract(symbol: str) -> pd.DataFrame:
    def _fetch():
        return ak.stock_financial_abstract(symbol=symbol)

    return retry_call(_fetch)


def _read_annual_indicator(symbol: str) -> pd.DataFrame:
    def _fetch():
        return ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")

    return retry_call(_fetch)


def _read_profile(symbol: str) -> Optional[Dict[str, Any]]:
    def _fetch():
        return ak.stock_individual_info_em(symbol=symbol)

    try:
        df = retry_call(_fetch)
        profile = dict(zip(df["item"].astype(str), df["value"]))
        return profile
    except Exception:
        return None


def _extract_row_values(df: pd.DataFrame, indicator_name: str, cols: List[str]) -> List[Optional[float]]:
    if "指标" not in df.columns:
        return []
    row = df[df["指标"] == indicator_name]
    if row.empty:
        return []
    one = row.iloc[0]
    return [parse_cn_number(one.get(col)) for col in cols]


def _extract_latest_from_indicator(df: pd.DataFrame, col: str) -> Optional[float]:
    if df.empty or col not in df.columns:
        return None
    temp = df.copy()
    if "报告期" in temp.columns:
        temp["报告期"] = temp["报告期"].astype(str)
        temp = temp.sort_values("报告期", ascending=False)
    return parse_cn_number(temp.iloc[0].get(col))


def _extract_latest_str_from_indicator(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return "N/A"
    temp = df.copy()
    if "报告期" in temp.columns:
        temp["报告期"] = temp["报告期"].astype(str)
        temp = temp.sort_values("报告期", ascending=False)
    val = temp.iloc[0].get(col)
    return str(val) if val not in [None, ""] else "N/A"


def _growth_from_series(values: List[Optional[float]]) -> Optional[float]:
    if len(values) < 2 or values[0] is None or values[1] in {None, 0}:
        return None
    prev = float(values[1])
    if prev == 0:
        return None
    return (float(values[0]) - prev) / abs(prev) * 100.0


@dataclass
class DimensionScore:
    key: str
    title: str
    score: float
    max_score: float
    comment: str


def _score_business_quality(gross_margin: Optional[float], net_margin: Optional[float]) -> DimensionScore:
    score = 0.0
    if gross_margin is not None:
        score += 2.5 if gross_margin >= 35 else 2.0 if gross_margin >= 25 else 1.3 if gross_margin >= 15 else 0.7
    if net_margin is not None:
        score += 2.5 if net_margin >= 15 else 2.0 if net_margin >= 10 else 1.3 if net_margin >= 5 else 0.7
    comment = f"毛利率 {format_pct(gross_margin)} / 净利率 {format_pct(net_margin)}"
    return DimensionScore("business_quality", "生意质量", min(score, 5.0), 5.0, comment)


def _score_profitability(roe: Optional[float], pe_ttm: Optional[float]) -> DimensionScore:
    score = 0.0
    if roe is not None:
        score += 3.0 if roe >= 18 else 2.4 if roe >= 12 else 1.7 if roe >= 8 else 0.8
    if pe_ttm is not None:
        score += 2.0 if pe_ttm <= 15 else 1.4 if pe_ttm <= 22 else 0.9 if pe_ttm <= 30 else 0.4
    comment = f"ROE {format_pct(roe)} / PE(滚) {format_num(pe_ttm)}"
    return DimensionScore("profitability", "盈利能力", min(score, 5.0), 5.0, comment)


def _score_cashflow(ocf_sum_3y: Optional[float], ocf_per_share: Optional[float]) -> DimensionScore:
    score = 0.0
    if ocf_sum_3y is not None:
        score += 3.0 if ocf_sum_3y > 0 else 1.0
    if ocf_per_share is not None:
        score += 2.0 if ocf_per_share >= 1 else 1.5 if ocf_per_share >= 0.5 else 0.8 if ocf_per_share > 0 else 0.3
    comment = f"近3年经营现金流 {format_num(ocf_sum_3y / 1e8 if ocf_sum_3y else None, 2, '亿')} / 每股经营现金流 {format_num(ocf_per_share)}"
    return DimensionScore("cashflow_quality", "现金流质量", min(score, 5.0), 5.0, comment)


def _score_balance_safety(debt_ratio: Optional[float], current_ratio: Optional[float], goodwill_ratio: Optional[float]) -> DimensionScore:
    score = 0.0
    if debt_ratio is not None:
        score += 2.2 if debt_ratio <= 45 else 1.7 if debt_ratio <= 60 else 1.0 if debt_ratio <= 75 else 0.4
    if current_ratio is not None:
        score += 1.8 if current_ratio >= 1.8 else 1.3 if current_ratio >= 1.2 else 0.8 if current_ratio >= 1.0 else 0.3
    if goodwill_ratio is not None:
        score += 1.0 if goodwill_ratio <= 10 else 0.6 if goodwill_ratio <= 20 else 0.2
    comment = f"负债率 {format_pct(debt_ratio)} / 流动比率 {format_num(current_ratio)} / 商誉占净资产 {format_pct(goodwill_ratio)}"
    return DimensionScore("balance_safety", "资产负债安全", min(score, 5.0), 5.0, comment)


def _score_growth(revenue_growth: Optional[float], profit_growth: Optional[float]) -> DimensionScore:
    score = 0.0
    if revenue_growth is not None:
        score += 2.5 if revenue_growth >= 15 else 2.0 if revenue_growth >= 8 else 1.3 if revenue_growth >= 0 else 0.6
    if profit_growth is not None:
        score += 2.5 if profit_growth >= 18 else 2.0 if profit_growth >= 10 else 1.3 if profit_growth >= 0 else 0.4
    comment = f"营收增长 {format_pct(revenue_growth)} / 净利增长 {format_pct(profit_growth)}"
    return DimensionScore("growth_quality", "增长质量", min(score, 5.0), 5.0, comment)


def _score_management(dividend_yield: Optional[float], retained_eps: Optional[float]) -> DimensionScore:
    score = 1.0
    if dividend_yield is not None:
        score += 2.2 if dividend_yield >= 5 else 1.7 if dividend_yield >= 3 else 1.2 if dividend_yield >= 1 else 0.5
    if retained_eps is not None:
        score += 1.8 if retained_eps >= 3 else 1.3 if retained_eps >= 1 else 0.9 if retained_eps >= 0 else 0.3
    comment = f"股息率 {format_pct(dividend_yield)} / 每股未分配利润 {format_num(retained_eps)}"
    return DimensionScore("management", "管理层配置", min(score, 5.0), 5.0, comment)


def _score_valuation(pe_dynamic: Optional[float], pe_static: Optional[float], pb: Optional[float]) -> DimensionScore:
    score = 0.0
    pe_candidates = [x for x in [pe_dynamic, pe_static] if x is not None]
    if pe_candidates:
        pe = min(pe_candidates)
        score += 3.0 if pe <= 12 else 2.2 if pe <= 18 else 1.4 if pe <= 25 else 0.7
    if pb is not None:
        score += 2.0 if pb <= 1.2 else 1.5 if pb <= 1.8 else 1.0 if pb <= 2.5 else 0.4
    comment = f"PE(动/静) {format_num(pe_dynamic)}/{format_num(pe_static)} / PB {format_num(pb)}"
    return DimensionScore("valuation", "估值安全边际", min(score, 5.0), 5.0, comment)


def _score_risk_control(volatility_proxy: Optional[float], receivable_days: Optional[float], debt_ratio: Optional[float]) -> DimensionScore:
    score = 0.0
    if volatility_proxy is not None:
        score += 2.0 if volatility_proxy <= 25 else 1.4 if volatility_proxy <= 35 else 0.9 if volatility_proxy <= 50 else 0.5
    if receivable_days is not None:
        score += 1.5 if receivable_days <= 35 else 1.1 if receivable_days <= 60 else 0.7 if receivable_days <= 90 else 0.3
    if debt_ratio is not None:
        score += 1.5 if debt_ratio <= 55 else 1.1 if debt_ratio <= 70 else 0.7 if debt_ratio <= 80 else 0.2
    comment = f"波动代理 {format_pct(volatility_proxy)} / 应收周转天数 {format_num(receivable_days)} / 负债率 {format_pct(debt_ratio)}"
    return DimensionScore("risk_control", "风险控制", min(score, 5.0), 5.0, comment)


def _build_summary(code: str, name: str, total_score: float, conclusion: str, dimensions: List[DimensionScore]) -> str:
    top = sorted(dimensions, key=lambda x: x.score, reverse=True)[:2]
    low = sorted(dimensions, key=lambda x: x.score)[:2]
    lines = [
        f"{name}({code}) 基本面总分 {total_score:.1f}/100，结论：{conclusion}。",
        f"优势：{top[0].title}({top[0].score:.1f}/5)、{top[1].title}({top[1].score:.1f}/5)。",
        f"短板：{low[0].title}({low[0].score:.1f}/5)、{low[1].title}({low[1].score:.1f}/5)。",
        "建议：优先跟踪现金流与负债安全，再结合估值位置决定仓位节奏。",
    ]
    return "\n".join(lines)


def analyze_fundamental(code: str, name: str = "", force_refresh: bool = False, cache_ttl_hours: int = 12) -> Dict[str, Any]:
    ensure_dirs()
    code = normalize_code(code)
    if not code:
        return {"code": code, "name": name or "未知", "error": "股票代码为空"}

    cache_file = _cache_file(code)
    if (not force_refresh) and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(payload.get("cached_at"))
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            if age_hours <= cache_ttl_hours:
                payload["from_cache"] = True
                return payload
        except Exception:
            pass

    result: Dict[str, Any] = {
        "code": code,
        "name": name or code,
        "analysis_at": now_str(),
        "from_cache": False,
        "app_version": APP_VERSION,
    }

    abstract_df = pd.DataFrame()
    indicator_df = pd.DataFrame()
    profile: Optional[Dict[str, Any]] = None
    data_warnings: List[str] = []

    try:
        abstract_df = _read_abstract(code)
    except Exception as e:
        data_warnings.append(f"财务摘要抓取失败: {e}")

    try:
        indicator_df = _read_annual_indicator(code)
    except Exception as e:
        data_warnings.append(f"年度指标抓取失败: {e}")

    try:
        profile = _read_profile(code)
    except Exception as e:
        data_warnings.append(f"个股资料抓取失败: {e}")

    if profile and not name:
        result["name"] = str(profile.get("股票简称", code))

    annual_cols = _latest_annual_columns(abstract_df, n=6) if not abstract_df.empty else []
    ocf_series = _extract_row_values(abstract_df, "经营现金流量净额", annual_cols)
    profit_series = _extract_row_values(abstract_df, "归母净利润", annual_cols)
    equity_series = _extract_row_values(abstract_df, "股东权益合计(净资产)", annual_cols)
    goodwill_series = _extract_row_values(abstract_df, "商誉", annual_cols)
    revenue_series = _extract_row_values(abstract_df, "营业总收入", annual_cols)

    ocf_sum_3y = sum(x for x in ocf_series[:3] if x is not None) if ocf_series else None
    revenue_growth = _growth_from_series(revenue_series[:2]) if revenue_series else None
    profit_growth = _growth_from_series(profit_series[:2]) if profit_series else None

    gross_margin = _extract_latest_from_indicator(indicator_df, "销售毛利率")
    net_margin = _extract_latest_from_indicator(indicator_df, "销售净利率")
    roe = _extract_latest_from_indicator(indicator_df, "净资产收益率")
    debt_ratio = _extract_latest_from_indicator(indicator_df, "资产负债率")
    current_ratio = _extract_latest_from_indicator(indicator_df, "流动比率")
    receivable_days = _extract_latest_from_indicator(indicator_df, "应收账款周转天数")
    ocf_per_share = _extract_latest_from_indicator(indicator_df, "每股经营性现金流(元)")
    retained_eps = _extract_latest_from_indicator(indicator_df, "每股未分配利润(元)")
    volatility_proxy = _extract_latest_from_indicator(indicator_df, "资产负债率")

    latest_goodwill = goodwill_series[0] if goodwill_series else None
    latest_equity = equity_series[0] if equity_series else None
    goodwill_ratio = safe_div(latest_goodwill, latest_equity)
    goodwill_ratio_pct = goodwill_ratio * 100 if goodwill_ratio is not None else None

    pe_dynamic = parse_cn_number((profile or {}).get("市盈率(动态)"))
    pe_static = parse_cn_number((profile or {}).get("市盈率(静)"))
    pe_ttm = parse_cn_number((profile or {}).get("市盈率(TTM)"))
    pb = parse_cn_number((profile or {}).get("市净率"))
    dividend_yield = parse_cn_number((profile or {}).get("股息率"))
    total_mv = parse_cn_number((profile or {}).get("总市值"))

    dimensions = [
        _score_business_quality(gross_margin, net_margin),
        _score_profitability(roe, pe_ttm),
        _score_cashflow(ocf_sum_3y, ocf_per_share),
        _score_balance_safety(debt_ratio, current_ratio, goodwill_ratio_pct),
        _score_growth(revenue_growth, profit_growth),
        _score_management(dividend_yield, retained_eps),
        _score_valuation(pe_dynamic, pe_static, pb),
        _score_risk_control(volatility_proxy, receivable_days, debt_ratio),
    ]

    total_score = sum(d.score for d in dimensions) / 40.0 * 100.0
    conclusion = "通过"
    if total_score < 70:
        conclusion = "观察"
    if total_score < 55:
        conclusion = "谨慎"

    available_fields = [
        gross_margin,
        net_margin,
        roe,
        debt_ratio,
        current_ratio,
        revenue_growth,
        profit_growth,
        pe_dynamic,
        pb,
        dividend_yield,
    ]
    coverage_ratio = sum(x is not None for x in available_fields) / len(available_fields)

    result.update(
        {
            "pe_dynamic": pe_dynamic,
            "pe_static": pe_static,
            "pe_ttm": pe_ttm,
            "pb": pb,
            "dividend_yield": dividend_yield,
            "total_mv": total_mv,
            "gross_margin": gross_margin,
            "net_margin": net_margin,
            "roe": roe,
            "debt_ratio": debt_ratio,
            "current_ratio": current_ratio,
            "receivable_days": receivable_days,
            "ocf_sum_3y": ocf_sum_3y,
            "revenue_growth": revenue_growth,
            "profit_growth": profit_growth,
            "coverage_ratio": coverage_ratio,
            "total_score": round(total_score, 1),
            "conclusion": conclusion,
            "dimensions": [
                {
                    "key": d.key,
                    "title": d.title,
                    "score": round(d.score, 2),
                    "max_score": d.max_score,
                    "comment": d.comment,
                }
                for d in dimensions
            ],
            "summary_text": _build_summary(code, result["name"], total_score, conclusion, dimensions),
            "data_warnings": data_warnings,
        }
    )

    try:
        cache_file.write_text(
            json.dumps({**result, "cached_at": datetime.now().isoformat()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
    return result


def analyze_watchlist(watchlist: List[Dict[str, str]], force_refresh: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in watchlist:
        one = analyze_fundamental(item["code"], item.get("name", ""), force_refresh=force_refresh)
        one["type"] = item.get("type", "观察")
        rows.append(one)
    return rows


def build_overview_table(analysis_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    data = []
    for x in analysis_rows:
        data.append(
            {
                "代码": x.get("code", ""),
                "名称": x.get("name", ""),
                "评分": x.get("total_score"),
                "类型": x.get("type", "观察"),
                "股息率": format_pct(x.get("dividend_yield")),
                "更新时间": x.get("analysis_at", ""),
            }
        )
    return pd.DataFrame(data)
