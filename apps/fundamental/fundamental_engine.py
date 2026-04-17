from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd
import requests
try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


APP_VERSION = "FND-20260416-03"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"

FUND_DEEPSEEK_PROMPT = """你是专业基本面分析师。请先基于财务与估值数据完成结构化研判，再结合事件催化进行短周期情绪判断。

【最新新闻催化】
{news_catalysts}

【机构研报摘要】
{research_summary}

输出要求：
1) 总结（不超过120字）
2) 八维点评（每维1句）
3) 关键风险（3条）
4) 跟踪清单（3条）
5) 结论：通过 / 观察 / 谨慎（给出理由）
6) 催化剂研判（必须包含）：
   - 请结合以上的最新新闻事件，判断目前的估值状态是否已被市场计价，
   - 以及未来一周潜在的利好/利空情绪反应方向。
要求：数据驱动、简洁、中文输出。"""


def _cache_data(ttl: int = 3600):
    def _decorator(func):
        if st is None:
            return func
        return st.cache_data(ttl=ttl, show_spinner=False)(func)

    return _decorator


def _clean_text(value: Any, max_len: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    return text[:max_len]


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    if df is None or df.empty:
        return ""
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        hit = cols.get(cand.lower())
        if hit:
            return hit
    return ""


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts)
        text = str(value).strip()
        if not text:
            return None
        dt = pd.to_datetime(text, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def _to_bullets(items: List[str], prefix: str = "- ") -> str:
    if not items:
        return "- 暂无可用素材"
    return "\n".join(f"{prefix}{x}" for x in items)


def _build_stock_keywords(code: str, name: str) -> List[str]:
    code_text = normalize_code(code)
    keys: List[str] = []
    if code_text:
        keys.append(code_text)
        keys.append(code_text.lstrip("0") or code_text)
        if len(code_text) == 6:
            keys.extend([f"SH{code_text}", f"SZ{code_text}"])
        elif len(code_text) == 5:
            keys.extend([f"{code_text}.HK", f"{code_text}HK"])
    clean_name = _clean_text(name, max_len=32).replace(" ", "")
    if clean_name:
        keys.append(clean_name)
    uniq: List[str] = []
    for x in keys:
        if x and x not in uniq:
            uniq.append(x)
    return uniq


def _contains_keywords(text: str, keywords: List[str]) -> bool:
    t = str(text or "")
    if not t:
        return False
    for k in keywords:
        if k and k in t:
            return True
    return False


def build_fund_deepseek_prompt(news_catalysts: str, research_summary: str) -> str:
    return FUND_DEEPSEEK_PROMPT.format(
        news_catalysts=(news_catalysts or "暂无可用催化剂"),
        research_summary=(research_summary or "暂无可用研报摘要"),
    )


@_cache_data(ttl=600)
def _read_cls_telegraph() -> pd.DataFrame:
    fn = getattr(ak, "stock_info_global_cls", None)
    if fn is None:
        return pd.DataFrame()
    try:
        df = retry_call(fn, max_retries=2)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@_cache_data(ttl=1200)
def _read_em_stock_news(symbol: str) -> pd.DataFrame:
    fn = getattr(ak, "stock_news_em", None)
    if fn is None:
        return pd.DataFrame()
    try:
        df = retry_call(lambda: fn(symbol=str(symbol)), max_retries=2)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@_cache_data(ttl=3600)
def _read_em_research(symbol: str) -> pd.DataFrame:
    fn = getattr(ak, "stock_research_report_em", None)
    if fn is None:
        return pd.DataFrame()
    try:
        df = retry_call(lambda: fn(symbol=str(symbol)), max_retries=2)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _fetch_news_research_context(code: str, name: str) -> Dict[str, Any]:
    now = datetime.now()
    start_3d = now - timedelta(days=3)
    keywords = _build_stock_keywords(code, name)

    telegraph_items: List[str] = []
    news_items: List[str] = []
    research_items: List[str] = []
    warns: List[str] = []
    details: Dict[str, Any] = {
        "telegraph_total": 0,
        "telegraph_matched": 0,
        "news_total": 0,
        "news_recent": 0,
        "news_selected": 0,
        "research_total": 0,
        "research_selected": 0,
    }

    tele_df = _read_cls_telegraph()
    if tele_df.empty:
        warns.append("财联社电报: 接口返回为空或抓取失败")
    else:
        details["telegraph_total"] = int(len(tele_df))
        t_time = _pick_col(tele_df, ["发布时间", "时间", "日期", "pub_time", "datetime"])
        t_title = _pick_col(tele_df, ["标题", "title", "摘要"])
        t_body = _pick_col(tele_df, ["内容", "正文", "快讯内容", "detail"])
        for _, row in tele_df.iterrows():
            title = _clean_text(row.get(t_title) if t_title else "", max_len=60)
            body = _clean_text(row.get(t_body) if t_body else "", max_len=90)
            merged = f"{title} {body}".strip()
            if not merged:
                continue
            if keywords and not _contains_keywords(merged.replace(" ", ""), keywords):
                continue
            dt = _parse_dt(row.get(t_time) if t_time else None)
            if dt is not None and dt.date() != now.date():
                continue
            details["telegraph_matched"] = int(details["telegraph_matched"]) + 1
            stamp = dt.strftime("%H:%M") if dt else "--:--"
            telegraph_items.append(f"{stamp} {title or body}")
            if len(telegraph_items) >= 5:
                break

    news_df = _read_em_stock_news(code)
    if news_df.empty:
        warns.append("近3日个股新闻: 接口返回为空或抓取失败")
    else:
        details["news_total"] = int(len(news_df))
        n_time = _pick_col(news_df, ["发布时间", "时间", "日期", "pub_time", "datetime"])
        n_title = _pick_col(news_df, ["新闻标题", "标题", "title", "文章标题"])
        n_src = _pick_col(news_df, ["文章来源", "来源", "媒体名称", "source"])
        picked: List[Tuple[datetime, str]] = []
        for _, row in news_df.iterrows():
            title = _clean_text(row.get(n_title) if n_title else "", max_len=70)
            if not title:
                continue
            dt = _parse_dt(row.get(n_time) if n_time else None)
            if dt is not None and dt < start_3d:
                continue
            details["news_recent"] = int(details["news_recent"]) + 1
            src = _clean_text(row.get(n_src) if n_src else "", max_len=14)
            label = f"{dt.strftime('%m-%d %H:%M') if dt else '--'} {title}"
            if src:
                label += f" ({src})"
            picked.append((dt or datetime(1970, 1, 1), label))
        for _, item in sorted(picked, key=lambda x: x[0], reverse=True)[:6]:
            news_items.append(item)
        details["news_selected"] = int(len(news_items))

    report_df = _read_em_research(code)
    if report_df.empty:
        warns.append("机构研报摘要: 接口返回为空或抓取失败")
    else:
        details["research_total"] = int(len(report_df))
        r_time = _pick_col(report_df, ["日期", "发布时间", "报告日期", "publish_date"])
        r_title = _pick_col(report_df, ["报告名称", "研报标题", "标题", "title"])
        r_org = _pick_col(report_df, ["机构", "机构名称", "评级机构", "研究机构"])
        r_rate = _pick_col(report_df, ["评级", "最新评级", "投资评级", "评级变动"])
        r_sum = _pick_col(report_df, ["摘要", "报告摘要", "核心观点", "观点"])
        picked_reports: List[Tuple[datetime, str]] = []
        for _, row in report_df.iterrows():
            title = _clean_text(row.get(r_title) if r_title else "", max_len=46)
            summary = _clean_text(row.get(r_sum) if r_sum else "", max_len=66)
            if not title and not summary:
                continue
            dt = _parse_dt(row.get(r_time) if r_time else None)
            org = _clean_text(row.get(r_org) if r_org else "", max_len=16)
            rate = _clean_text(row.get(r_rate) if r_rate else "", max_len=10)
            head = " / ".join(x for x in [org, rate] if x)
            core = summary or title
            line = f"{dt.strftime('%Y-%m-%d') if dt else '--'}"
            if head:
                line += f" [{head}]"
            line += f" {core}"
            picked_reports.append((dt or datetime(1970, 1, 1), line))
        for _, item in sorted(picked_reports, key=lambda x: x[0], reverse=True)[:2]:
            research_items.append(item)
        details["research_selected"] = int(len(research_items))

    telegraph_lines = telegraph_items[:]
    if not telegraph_lines:
        if tele_df.empty:
            telegraph_lines = ["财联社电报：接口返回为空或抓取失败"]
        else:
            telegraph_lines = [
                f"当日未匹配到该股（扫描{details['telegraph_total']}条电报，匹配{details['telegraph_matched']}条）"
            ]

    news_lines = news_items[:]
    if not news_lines:
        if news_df.empty:
            news_lines = ["个股新闻：接口返回为空或抓取失败"]
        elif int(details["news_recent"]) == 0:
            news_lines = [f"近3天时间窗内无新闻（原始{details['news_total']}条）"]
        else:
            news_lines = [f"有近3天新闻但筛选后为空（近3天{details['news_recent']}条）"]

    research_lines = research_items[:]
    if not research_lines:
        if report_df.empty:
            research_lines = ["机构研报：接口返回为空或抓取失败"]
        else:
            research_lines = [f"该股暂无可用研报摘要（原始{details['research_total']}篇）"]

    news_catalysts = (
        "【财联社电报（当日）】\n"
        + _to_bullets(telegraph_lines)
        + "\n【近3日核心新闻】\n"
        + _to_bullets(news_lines)
    )
    research_summary = "【最近2篇机构研报摘要】\n" + _to_bullets(research_lines)
    return {
        "telegraph_items": telegraph_items,
        "news_items": news_items,
        "research_items": research_items,
        "news_catalysts": news_catalysts,
        "research_summary": research_summary,
        "warns": warns,
        "details": details,
    }


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
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
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
        return f"--{suffix}"
    try:
        if pd.isna(value):
            return f"--{suffix}"
    except Exception:
        pass
    return f"{value:.{digits}f}{suffix}"


def format_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        if pd.isna(value):
            return "--"
    except Exception:
        pass
    return f"{value:.{digits}f}%"


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in {None, 0} or pd.isna(a) or pd.isna(b):
            return None
    except Exception:
        if a is None or b in {None, 0}:
            return None
    if b == 0:
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


def _normalize_name(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().upper()


def resolve_stock_identity(query: str) -> Tuple[str, str]:
    q = str(query or "").strip()
    if not q:
        raise ValueError("请输入股票代码或名称")

    code_candidate = normalize_code(q)
    rows = load_watchlist()

    # 先从本地观察池匹配（代码/名称）
    if code_candidate and len(code_candidate) in {5, 6}:
        for item in rows:
            if str(item.get("code", "")) == code_candidate:
                name = str(item.get("name", "")).strip() or code_candidate
                if name == code_candidate:
                    tx_name = _fetch_name_from_tencent(code_candidate)
                    if tx_name:
                        name = tx_name
                return code_candidate, name
    for item in rows:
        if str(item.get("name", "")).strip() == q:
            return str(item.get("code", "")).strip(), str(item.get("name", "")).strip()

    # 代码输入：优先腾讯名称兜底，再尝试 AKShare 列表
    if code_candidate and len(code_candidate) in {5, 6}:
        tx_name = _fetch_name_from_tencent(code_candidate)
        if tx_name:
            return code_candidate, tx_name

        if len(code_candidate) == 6:
            try:
                a_df = ak.stock_info_a_code_name()
                if a_df is not None and not a_df.empty:
                    a_df = a_df.copy()
                    a_df["code"] = a_df["code"].astype(str).str.strip()
                    a_df["name"] = a_df["name"].astype(str).str.strip()
                    one = a_df[a_df["code"] == code_candidate]
                    if not one.empty:
                        row = one.iloc[0]
                        return code_candidate, str(row["name"]).strip()
            except Exception:
                pass
        else:
            try:
                hk_df = ak.stock_hk_spot()
                if hk_df is not None and not hk_df.empty:
                    hk_df = hk_df.copy()
                    hk_df["代码"] = hk_df["代码"].astype(str).str.strip().str.zfill(5)
                    name_col = "中文名称" if "中文名称" in hk_df.columns else "名称"
                    hk_df[name_col] = hk_df[name_col].astype(str).str.strip()
                    one = hk_df[hk_df["代码"] == code_candidate]
                    if not one.empty:
                        row = one.iloc[0]
                        return code_candidate, str(row[name_col]).strip()
            except Exception:
                pass
        raise ValueError(f"未找到代码为 {code_candidate} 的标的")

    # 名称输入：先A股，再港股（支持模糊包含）
    q_norm = _normalize_name(q)
    try:
        a_df = ak.stock_info_a_code_name()
        if a_df is not None and not a_df.empty:
            a_df = a_df.copy()
            a_df["code"] = a_df["code"].astype(str).str.strip()
            a_df["name"] = a_df["name"].astype(str).str.strip()
            a_df["name_norm"] = a_df["name"].map(_normalize_name)
            exact = a_df[a_df["name_norm"] == q_norm]
            if not exact.empty:
                row = exact.iloc[0]
                return str(row["code"]), str(row["name"])
            fuzzy = a_df[a_df["name_norm"].str.contains(q_norm, na=False)]
            if not fuzzy.empty:
                row = fuzzy.iloc[0]
                return str(row["code"]), str(row["name"])
    except Exception:
        pass

    try:
        hk_df = ak.stock_hk_spot()
        if hk_df is not None and not hk_df.empty:
            hk_df = hk_df.copy()
            hk_df["代码"] = hk_df["代码"].astype(str).str.strip().str.zfill(5)
            name_col = "中文名称" if "中文名称" in hk_df.columns else "名称"
            hk_df[name_col] = hk_df[name_col].astype(str).str.strip()
            hk_df["name_norm"] = hk_df[name_col].map(_normalize_name)
            exact = hk_df[hk_df["name_norm"] == q_norm]
            if not exact.empty:
                row = exact.iloc[0]
                return str(row["代码"]), str(row[name_col])
            fuzzy = hk_df[hk_df["name_norm"].str.contains(q_norm, na=False)]
            if not fuzzy.empty:
                row = fuzzy.iloc[0]
                return str(row["代码"]), str(row[name_col])
    except Exception:
        pass

    raise ValueError(f"未找到名称为 {q} 的A股/港股标的")


def upsert_watch_item_by_query(query: str, item_type: str) -> List[Dict[str, str]]:
    code, name = resolve_stock_identity(query)
    return upsert_watch_item(code, name, item_type)


def _latest_annual_columns(df: pd.DataFrame, n: int = 6) -> List[str]:
    cols = [c for c in df.columns if re.fullmatch(r"\d{8}", str(c))]
    cols = [c for c in cols if str(c).endswith("1231")]
    cols.sort(reverse=True)
    return cols[:n]


@_cache_data(ttl=3600)
def _read_abstract(symbol: str) -> pd.DataFrame:
    def _fetch():
        return ak.stock_financial_abstract(symbol=symbol)

    return retry_call(_fetch)


@_cache_data(ttl=3600)
def _read_annual_indicator(symbol: str) -> pd.DataFrame:
    def _fetch():
        return ak.stock_financial_abstract_ths(symbol=symbol, indicator="按年度")

    return retry_call(_fetch)


@_cache_data(ttl=3600)
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


def _extract_latest_from_indicator_multi(df: pd.DataFrame, cols: List[str]) -> Optional[float]:
    for col in cols:
        val = _extract_latest_from_indicator(df, col)
        if val is not None:
            return val
    return None


def _extract_indicator_series_multi(df: pd.DataFrame, cols: List[str], n: int = 5) -> List[float]:
    if df.empty:
        return []
    hit_col = ""
    for col in cols:
        if col in df.columns:
            hit_col = col
            break
    if not hit_col:
        return []

    temp = df.copy()
    if "报告期" in temp.columns:
        temp["报告期"] = temp["报告期"].astype(str)
        temp = temp[temp["报告期"].str.contains("12-31|1231", na=False)]
        temp = temp.sort_values("报告期", ascending=False)

    out: List[float] = []
    for _, r in temp.iterrows():
        v = parse_cn_number(r.get(hit_col))
        if v is not None:
            out.append(float(v))
        if len(out) >= max(1, int(n)):
            break
    return out


def _pick_profile_number(profile: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in profile:
            val = parse_cn_number(profile.get(key))
            if val is not None:
                return val
    return None


def _coalesce_number(*values: Optional[float], default: Optional[float] = None) -> Optional[float]:
    for value in values:
        try:
            if value is not None and not pd.isna(value):
                return float(value)
        except Exception:
            if value is not None:
                return float(value)
    if default is None:
        return None
    return float(default)


def _extract_latest_str_from_indicator(df: pd.DataFrame, col: str) -> str:
    if df.empty or col not in df.columns:
        return "--"
    temp = df.copy()
    if "报告期" in temp.columns:
        temp["报告期"] = temp["报告期"].astype(str)
        temp = temp.sort_values("报告期", ascending=False)
    val = temp.iloc[0].get(col)
    return str(val) if val not in [None, ""] else "--"


def _is_hk_symbol(symbol: str) -> bool:
    text = str(symbol or "").strip()
    return bool(re.fullmatch(r"\d{5}", text))


def _normalize_pe_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    v = float(value)
    if abs(v) < 1e-12:
        return None
    if abs(v) > 1e6:
        return None
    return v


def _normalize_pb_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    v = float(value)
    if v <= 0 or v > 1e5:
        return None
    return v


def _normalize_dividend_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    v = float(value)
    if v < 0:
        return None
    # 某些接口给小数口径 0.0492 => 4.92%
    if 0 < v < 0.2:
        v *= 100
    return v


def _fetch_metrics_from_eastmoney_direct(symbol: str) -> Dict[str, Optional[float]]:
    secid = ("1." if str(symbol).startswith("6") else "0.") + str(symbol)
    fields = "f57,f58,f162,f163,f164,f167"
    urls = [
        "https://push2.eastmoney.com/api/qt/stock/get",
        "http://push2.eastmoney.com/api/qt/stock/get",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com",
    }
    for _ in range(3):
        for url in urls:
            try:
                session = requests.Session()
                session.trust_env = False
                resp = session.get(
                    url,
                    params={"invt": "2", "fltt": "2", "secid": secid, "fields": fields},
                    headers=headers,
                    timeout=8,
                )
                resp.raise_for_status()
                data = (resp.json() or {}).get("data") or {}
                return {
                    "pe_dynamic": _normalize_pe_value(parse_cn_number(data.get("f162"))),
                    "pe_static": _normalize_pe_value(parse_cn_number(data.get("f163"))),
                    "pe_ttm": _normalize_pe_value(parse_cn_number(data.get("f164"))),
                    "pb": _normalize_pb_value(parse_cn_number(data.get("f167"))),
                }
            except Exception:
                continue
    return {"pe_dynamic": None, "pe_static": None, "pe_ttm": None, "pb": None}


def _fetch_metrics_from_tencent(symbol: str) -> Dict[str, Optional[float]]:
    symbol_text = str(symbol).strip()
    if _is_hk_symbol(symbol_text):
        exchange = "hk"
    elif symbol_text.startswith(("5", "6", "9")):
        exchange = "sh"
    else:
        exchange = "sz"
    url = f"https://qt.gtimg.cn/q={exchange}{symbol_text}"
    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        resp.raise_for_status()
        resp.encoding = "gbk"
        text = resp.text
        if '"' not in text or "~" not in text:
            return {"pe_dynamic": None, "pe_ttm": None, "pb": None}
        payload = text.split('"', 1)[1].rsplit('"', 1)[0]
        fields = payload.split("~")
        pe_dynamic = parse_cn_number(fields[52]) if len(fields) > 52 else None
        pe_ttm = parse_cn_number(fields[53]) if len(fields) > 53 else None
        pb = parse_cn_number(fields[46]) if len(fields) > 46 else None
        return {
            "pe_dynamic": _normalize_pe_value(pe_dynamic),
            "pe_ttm": _normalize_pe_value(pe_ttm),
            "pb": _normalize_pb_value(pb),
        }
    except Exception:
        return {"pe_dynamic": None, "pe_ttm": None, "pb": None}


def _fetch_name_from_tencent(symbol: str) -> Optional[str]:
    symbol_text = str(symbol).strip()
    if _is_hk_symbol(symbol_text):
        exchange = "hk"
    elif symbol_text.startswith(("5", "6", "9")):
        exchange = "sh"
    else:
        exchange = "sz"
    url = f"https://qt.gtimg.cn/q={exchange}{symbol_text}"
    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        resp.raise_for_status()
        resp.encoding = "gbk"
        text = resp.text
        if '"' not in text or "~" not in text:
            return None
        payload = text.split('"', 1)[1].rsplit('"', 1)[0]
        fields = payload.split("~")
        if len(fields) < 2:
            return None
        name = str(fields[1]).strip()
        if not name or name in {"-", "--", "None", "nan", "NaN"}:
            return None
        return name
    except Exception:
        return None


def _fetch_dividend_yield_from_em(symbol: str) -> Optional[float]:
    try:
        df = ak.stock_fhps_detail_em(symbol=str(symbol))
        if df is None or df.empty or "现金分红-股息率" not in df.columns:
            return None
        tmp = df.copy()
        tmp["现金分红-股息率"] = pd.to_numeric(tmp["现金分红-股息率"], errors="coerce")
        tmp = tmp.dropna(subset=["现金分红-股息率"])
        if tmp.empty:
            return None
        annual = tmp[tmp["报告期"].astype(str).str.endswith("12-31")]
        chosen = annual if not annual.empty else tmp
        val = parse_cn_number(chosen.iloc[-1]["现金分红-股息率"])
        return _normalize_dividend_value(val)
    except Exception:
        return None


def _growth_from_series(values: List[Optional[float]]) -> Optional[float]:
    if len(values) < 2 or values[0] is None or values[1] in {None, 0}:
        return None
    prev = float(values[1])
    if prev == 0:
        return None
    return (float(values[0]) - prev) / abs(prev) * 100.0


def _cagr_from_series(values: List[Optional[float]], years: int = 5) -> Optional[float]:
    need_n = max(2, int(years))
    valid = [float(x) for x in values[:need_n] if x is not None]
    if len(valid) < need_n:
        return None
    latest = valid[0]
    oldest = valid[need_n - 1]
    periods = need_n - 1
    if oldest <= 0 or latest <= 0 or periods <= 0:
        return None
    try:
        return (pow(latest / oldest, 1.0 / periods) - 1.0) * 100.0
    except Exception:
        return None


def _avg_from_series(values: List[Optional[float]], n: int = 5) -> Optional[float]:
    valid = [float(x) for x in values[: max(1, int(n))] if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _positive_years(values: List[Optional[float]], n: int = 5) -> int:
    return sum(1 for x in values[: max(1, int(n))] if x is not None and float(x) > 0)


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
    stale_cache: Dict[str, Any] = {}
    if cache_file.exists():
        try:
            stale_cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            stale_cache = {}

    if (not force_refresh) and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(payload.get("cached_at"))
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            cache_ver = str(payload.get("app_version", ""))
            if age_hours <= cache_ttl_hours and cache_ver == APP_VERSION:
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

    if not name or str(name).strip() == code:
        tx_name = _fetch_name_from_tencent(code)
        if tx_name:
            result["name"] = tx_name

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
    revenue_cagr_5y = _cagr_from_series(revenue_series, years=5) if revenue_series else None
    profit_cagr_5y = _cagr_from_series(profit_series, years=5) if profit_series else None
    ocf_positive_years_5y = _positive_years(ocf_series, n=5) if ocf_series else 0

    gross_margin = _extract_latest_from_indicator_multi(indicator_df, ["销售毛利率", "销售毛利率(%)"])
    net_margin = _extract_latest_from_indicator_multi(indicator_df, ["销售净利率", "销售净利率(%)"])
    roe = _extract_latest_from_indicator_multi(indicator_df, ["净资产收益率", "净资产收益率(%)", "加权净资产收益率(%)"])
    debt_ratio = _extract_latest_from_indicator_multi(indicator_df, ["资产负债率", "资产负债率(%)"])
    current_ratio = _extract_latest_from_indicator_multi(indicator_df, ["流动比率"])
    receivable_days = _extract_latest_from_indicator(indicator_df, "应收账款周转天数")
    ocf_per_share = _extract_latest_from_indicator_multi(indicator_df, ["每股经营性现金流(元)", "每股经营现金流(元)", "每股经营现金流"])
    retained_eps = _extract_latest_from_indicator_multi(indicator_df, ["每股未分配利润(元)", "每股未分配利润"])
    volatility_proxy = _extract_latest_from_indicator(indicator_df, "资产负债率")
    roe_5y_series = _extract_indicator_series_multi(indicator_df, ["净资产收益率", "净资产收益率(%)", "加权净资产收益率(%)"], n=5)
    gross_margin_5y_series = _extract_indicator_series_multi(indicator_df, ["销售毛利率", "销售毛利率(%)"], n=5)
    debt_ratio_5y_series = _extract_indicator_series_multi(indicator_df, ["资产负债率", "资产负债率(%)"], n=5)
    roe_avg_5y = _avg_from_series(roe_5y_series, n=5)
    gross_margin_avg_5y = _avg_from_series(gross_margin_5y_series, n=5)
    debt_ratio_avg_5y = _avg_from_series(debt_ratio_5y_series, n=5)
    gross_margin_change_5y = None
    debt_ratio_change_5y = None
    if len(gross_margin_5y_series) >= 5:
        gross_margin_change_5y = float(gross_margin_5y_series[0]) - float(gross_margin_5y_series[4])
    if len(debt_ratio_5y_series) >= 5:
        debt_ratio_change_5y = float(debt_ratio_5y_series[0]) - float(debt_ratio_5y_series[4])

    latest_goodwill = goodwill_series[0] if goodwill_series else None
    latest_equity = equity_series[0] if equity_series else None
    goodwill_ratio = safe_div(latest_goodwill, latest_equity)
    goodwill_ratio_pct = goodwill_ratio * 100 if goodwill_ratio is not None else None

    p = profile or {}
    pe_dynamic = _normalize_pe_value(
        _pick_profile_number(p, ["市盈率(动态)", "市盈率-动态", "动态市盈率"])
    )
    pe_static = _normalize_pe_value(
        _pick_profile_number(p, ["市盈率(静)", "市盈率-静态", "静态市盈率"])
    )
    pe_ttm = _normalize_pe_value(
        _pick_profile_number(p, ["市盈率(TTM)", "市盈率TTM", "市盈率(滚动)", "滚动市盈率"])
    )
    pb = _normalize_pb_value(_pick_profile_number(p, ["市净率", "市净率MRQ"]))
    dividend_yield = _normalize_dividend_value(_pick_profile_number(p, ["股息率", "股息率(%)"]))
    total_mv = _pick_profile_number(p, ["总市值", "总市值(元)"])

    # 多源估值兜底：东方财富直连 + 腾讯快照 + 分红详情
    em_metrics = _fetch_metrics_from_eastmoney_direct(code) if not _is_hk_symbol(code) else {}
    tx_metrics = _fetch_metrics_from_tencent(code)
    dy_em = _fetch_dividend_yield_from_em(code) if not _is_hk_symbol(code) else None

    pe_dynamic = _coalesce_number(
        pe_dynamic,
        em_metrics.get("pe_dynamic"),
        tx_metrics.get("pe_dynamic"),
        _normalize_pe_value(stale_cache.get("pe_dynamic")),
    )
    pe_static = _coalesce_number(
        pe_static,
        em_metrics.get("pe_static"),
        _normalize_pe_value(stale_cache.get("pe_static")),
        pe_ttm,
        pe_dynamic,
    )
    pe_ttm = _coalesce_number(
        pe_ttm,
        em_metrics.get("pe_ttm"),
        tx_metrics.get("pe_ttm"),
        _normalize_pe_value(stale_cache.get("pe_ttm")),
        pe_dynamic,
        pe_static,
    )
    pb = _coalesce_number(
        pb,
        em_metrics.get("pb"),
        tx_metrics.get("pb"),
        _normalize_pb_value(stale_cache.get("pb")),
    )
    dividend_yield = _coalesce_number(
        dividend_yield,
        dy_em,
        _normalize_dividend_value(stale_cache.get("dividend_yield")),
    )
    total_mv = _coalesce_number(total_mv, stale_cache.get("total_mv"))

    gross_margin = _coalesce_number(gross_margin, stale_cache.get("gross_margin"))
    net_margin = _coalesce_number(net_margin, stale_cache.get("net_margin"))
    roe = _coalesce_number(roe, stale_cache.get("roe"))
    debt_ratio = _coalesce_number(debt_ratio, stale_cache.get("debt_ratio"))
    current_ratio = _coalesce_number(current_ratio, stale_cache.get("current_ratio"))
    receivable_days = _coalesce_number(receivable_days, stale_cache.get("receivable_days"))
    ocf_per_share = _coalesce_number(ocf_per_share, stale_cache.get("ocf_per_share"))
    retained_eps = _coalesce_number(retained_eps, stale_cache.get("retained_eps"))
    volatility_proxy = _coalesce_number(volatility_proxy, debt_ratio)
    ocf_sum_3y = _coalesce_number(ocf_sum_3y, stale_cache.get("ocf_sum_3y"))
    revenue_growth = _coalesce_number(revenue_growth, stale_cache.get("revenue_growth"))
    profit_growth = _coalesce_number(profit_growth, stale_cache.get("profit_growth"))
    revenue_cagr_5y = _coalesce_number(revenue_cagr_5y, stale_cache.get("revenue_cagr_5y"))
    profit_cagr_5y = _coalesce_number(profit_cagr_5y, stale_cache.get("profit_cagr_5y"))
    roe_avg_5y = _coalesce_number(roe_avg_5y, stale_cache.get("roe_avg_5y"))
    gross_margin_avg_5y = _coalesce_number(gross_margin_avg_5y, stale_cache.get("gross_margin_avg_5y"))
    debt_ratio_avg_5y = _coalesce_number(debt_ratio_avg_5y, stale_cache.get("debt_ratio_avg_5y"))
    gross_margin_change_5y = _coalesce_number(gross_margin_change_5y, stale_cache.get("gross_margin_change_5y"))
    debt_ratio_change_5y = _coalesce_number(debt_ratio_change_5y, stale_cache.get("debt_ratio_change_5y"))
    ocf_positive_years_5y = int(_coalesce_number(ocf_positive_years_5y, stale_cache.get("ocf_positive_years_5y"), default=0) or 0)
    goodwill_ratio_pct = _coalesce_number(goodwill_ratio_pct, stale_cache.get("goodwill_ratio_pct"))

    catalyst_ctx = _fetch_news_research_context(code, result.get("name", name or code))
    news_catalysts = str(catalyst_ctx.get("news_catalysts") or "").strip()
    research_summary = str(catalyst_ctx.get("research_summary") or "").strip()
    if not news_catalysts:
        news_catalysts = str(stale_cache.get("news_catalysts", "")).strip() or "暂无可用催化剂"
    if not research_summary:
        research_summary = str(stale_cache.get("research_summary", "")).strip() or "暂无可用研报摘要"
    for warn in catalyst_ctx.get("warns", []):
        if warn and warn not in data_warnings:
            data_warnings.append(str(warn))

    if pe_dynamic is None:
        data_warnings.append("PE(动) 暂不可用")
    if pe_static is None:
        data_warnings.append("PE(静) 暂不可用")
    if pe_ttm is None:
        data_warnings.append("PE(滚) 暂不可用")
    if dividend_yield is None:
        data_warnings.append("股息率 暂不可用")

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
            "ocf_per_share": ocf_per_share,
            "retained_eps": retained_eps,
            "goodwill_ratio_pct": goodwill_ratio_pct,
            "revenue_growth": revenue_growth,
            "profit_growth": profit_growth,
            "revenue_cagr_5y": revenue_cagr_5y,
            "profit_cagr_5y": profit_cagr_5y,
            "roe_avg_5y": roe_avg_5y,
            "gross_margin_avg_5y": gross_margin_avg_5y,
            "debt_ratio_avg_5y": debt_ratio_avg_5y,
            "gross_margin_change_5y": gross_margin_change_5y,
            "debt_ratio_change_5y": debt_ratio_change_5y,
            "ocf_positive_years_5y": ocf_positive_years_5y,
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
            "news_catalysts": news_catalysts,
            "research_summary": research_summary,
            "telegraph_items": catalyst_ctx.get("telegraph_items", []),
            "news_items": catalyst_ctx.get("news_items", []),
            "research_items": catalyst_ctx.get("research_items", []),
            "news_fetch_details": catalyst_ctx.get("details", {}),
            "fund_deepseek_prompt": build_fund_deepseek_prompt(news_catalysts, research_summary),
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
