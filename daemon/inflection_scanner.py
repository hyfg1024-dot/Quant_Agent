from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daemon.alert_worker import AlertWorker
from shared.capital_flow_signals import CapitalFlowAnalyzer
from shared.db_manager import connect_db, init_duckdb, normalize_position_symbol
from shared.inflection_detector import InflectionDetector
from shared.technical_structure import TechnicalStructureAnalyzer
from shared.valuation_percentile import ValuationPercentile


logger = logging.getLogger("inflection_scanner")


DEFAULT_CONFIG: Dict[str, Any] = {
    "scan": {
        "schedule": "18:00",
        "auto_expand_from_filter": True,
        "filter_min_score": 60,
        "composite_threshold": 60,
        "max_candidates": 200,
    },
    "watchlist": [],
    "weights": {
        "inflection": 0.35,
        "bottom": 0.20,
        "valuation": 0.30,
        "capital": 0.15,
    },
    "push": {
        "channel": "telegram",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def _normalize_market_code(code: str, market: Optional[str] = None) -> Tuple[str, str, str]:
    try:
        mk, db_code, display = normalize_position_symbol(code=code, market=market)
        return mk, db_code, display
    except Exception:
        digits = "".join(ch for ch in str(code or "") if ch.isdigit())
        if (market or "").upper() in {"HK", "H"} or len(digits) <= 5:
            c = digits.zfill(5)
            return "HK", c, f"{c}.HK"
        c = digits.zfill(6)
        return "A", c, c


def _parse_hhmm(text: str) -> Tuple[int, int]:
    parts = str(text or "18:00").strip().split(":")
    if len(parts) != 2:
        return 18, 0
    return int(parts[0]), int(parts[1])


class InflectionScanner:
    def __init__(self, config_path: Path) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        self.valuation_analyzer = ValuationPercentile()
        self.technical_analyzer = TechnicalStructureAnalyzer()
        self.capital_analyzer = CapitalFlowAnalyzer(request_interval_sec=0.3)

        scan_cfg = (self.config.get("scan", {}) or {})
        self.composite_threshold = float(scan_cfg.get("composite_threshold", 60) or 60)
        tz_name = str(scan_cfg.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
        self.tz = ZoneInfo(tz_name)

        self.filter_db_path = PROJECT_ROOT / "apps" / "filter" / "data" / "filter_market.db"
        self.alert_config_path = PROJECT_ROOT / "config" / "alert_rules.yaml"

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(obj, dict):
            raise ValueError("配置文件格式错误，顶层应为 YAML 对象")
        return _deep_merge(DEFAULT_CONFIG, obj)

    def _load_watchlist_candidates(self) -> List[Dict[str, str]]:
        rows = self.config.get("watchlist", [])
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, str]] = []
        for one in rows:
            if not isinstance(one, dict):
                continue
            code = str(one.get("code", "") or "").strip()
            name = str(one.get("name", "") or "").strip()
            market = str(one.get("market", "") or "").strip().upper() or None
            if not code:
                continue
            mk, db_code, _display = _normalize_market_code(code, market)
            out.append(
                {
                    "market": mk,
                    "code": db_code,
                    "name": name or db_code,
                    "source": "watchlist",
                }
            )
        return out

    def _load_filter_candidates(self, min_score: float, max_rows: int) -> List[Dict[str, str]]:
        if not self.filter_db_path.exists():
            logger.warning("filter snapshot db not found: %s", self.filter_db_path)
            return []

        queries = [
            (
                "SELECT market, code, name, total_score FROM market_snapshot "
                "WHERE total_score IS NOT NULL AND total_score > ? "
                "ORDER BY total_score DESC LIMIT ?",
                True,
            ),
            (
                "SELECT code, name, total_score FROM market_snapshot "
                "WHERE total_score IS NOT NULL AND total_score > ? "
                "ORDER BY total_score DESC LIMIT ?",
                False,
            ),
        ]

        with sqlite3.connect(str(self.filter_db_path)) as conn:
            for sql, has_market in queries:
                try:
                    df = pd.read_sql_query(sql, conn, params=[float(min_score), int(max_rows)])
                    if df is None or df.empty:
                        return []
                    out: List[Dict[str, str]] = []
                    for _, row in df.iterrows():
                        code = str(row.get("code", "") or "").strip()
                        name = str(row.get("name", "") or "").strip()
                        market = str(row.get("market", "") or "").strip().upper() if has_market else ""
                        mk_hint = market if market in {"A", "HK", "H", "CN", "SH", "SZ"} else None
                        if not code:
                            continue
                        mk, db_code, _display = _normalize_market_code(code, mk_hint)
                        out.append(
                            {
                                "market": mk,
                                "code": db_code,
                                "name": name or db_code,
                                "source": "filter",
                                "filter_total_score": _safe_float(row.get("total_score"), 0.0),
                            }
                        )
                    return out
                except Exception:
                    continue
        return []

    @staticmethod
    def _merge_candidates(
        watchlist: List[Dict[str, str]],
        expanded: List[Dict[str, str]],
        max_candidates: int,
    ) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        seen: set[str] = set()
        for rows in (watchlist, expanded):
            for one in rows:
                market = str(one.get("market", "A") or "A").upper()
                code = str(one.get("code", "") or "").strip()
                if not code:
                    continue
                key = f"{market}:{code}"
                if key in seen:
                    continue
                seen.add(key)
                merged.append(one)
                if len(merged) >= max(1, int(max_candidates)):
                    return merged
        return merged

    def _load_daily_bars_from_duckdb(self, market: str, code: str, lookback_days: int = 400) -> pd.DataFrame:
        try:
            with connect_db(read_only=True) as conn:
                latest_row = conn.execute(
                    "SELECT MAX(trade_date) FROM daily_kline WHERE market = ? AND code = ?",
                    [market, code],
                ).fetchone()
                latest_date = latest_row[0] if latest_row else None
                if latest_date is None:
                    return pd.DataFrame()
                if isinstance(latest_date, str):
                    latest_date = pd.to_datetime(latest_date, errors="coerce").date()
                if latest_date is None:
                    return pd.DataFrame()
                start_date = latest_date - timedelta(days=max(60, int(lookback_days)))
                df = conn.execute(
                    """
                    SELECT trade_date, open, high, low, close, volume
                    FROM daily_kline
                    WHERE market = ? AND code = ? AND trade_date >= ?
                    ORDER BY trade_date ASC
                    """,
                    [market, code, start_date],
                ).df()
            if df is None or df.empty:
                return pd.DataFrame()
            out = df.rename(columns={"trade_date": "date"}).copy()
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=["date", "close"]).reset_index(drop=True)
            return out[["date", "open", "high", "low", "close", "volume"]]
        except Exception:
            return pd.DataFrame()

    def _load_daily_bars_from_akshare(self, market: str, code: str, lookback_days: int = 400) -> pd.DataFrame:
        if market != "A":
            return pd.DataFrame()
        try:
            import akshare as ak
        except Exception:
            return pd.DataFrame()

        start_date = (datetime.now().date() - timedelta(days=max(120, int(lookback_days)))).strftime("%Y%m%d")
        end_date = datetime.now().date().strftime("%Y%m%d")
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="")
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()

        mapper = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        cols = {k: v for k, v in mapper.items() if k in df.columns}
        if len(cols) < 6:
            return pd.DataFrame()
        out = df[list(cols.keys())].rename(columns=cols).copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        return out[["date", "open", "high", "low", "close", "volume"]]

    def _load_daily_bars(self, market: str, code: str) -> pd.DataFrame:
        df = self._load_daily_bars_from_duckdb(market=market, code=code)
        if not df.empty:
            return df
        return self._load_daily_bars_from_akshare(market=market, code=code)

    @staticmethod
    def _extract_inflection_note(inflection_result: Dict[str, Any]) -> str:
        signals = inflection_result.get("signals", []) if isinstance(inflection_result, dict) else []
        picked: List[Tuple[float, str]] = []
        if isinstance(signals, list):
            for one in signals:
                if not isinstance(one, dict):
                    continue
                if not bool(one.get("triggered", False)):
                    continue
                score = _safe_float(one.get("score"), 0.0)
                detail = str(one.get("detail", "") or "").strip()
                if detail:
                    picked.append((score, detail))
        if picked:
            picked.sort(key=lambda x: x[0], reverse=True)
            details = [d for _, d in picked[:2]]
            return "，".join(details)
        summary = str((inflection_result or {}).get("summary", "") or "").strip()
        return summary if summary else "无显著拐点细节"

    def _scan_one(self, item: Dict[str, str], weights: Dict[str, float]) -> Tuple[Dict[str, Any], List[str]]:
        market = str(item.get("market", "A") or "A").upper()
        code = str(item.get("code", "") or "").strip()
        name = str(item.get("name", code) or code)

        warnings_list: List[str] = []
        inflection_score = 0.0
        bottom_score = 0.0
        valuation_percentile = 50.0
        capital_score = 0.0
        capital_grade = "中性"
        inflection_note = ""

        try:
            detector = InflectionDetector(enable_propagation_tip=False)
            inflection_result = detector.compute_inflection_score(code=code, name=name)
            inflection_score = _safe_float(inflection_result.get("inflection_score"), 0.0)
            inflection_note = self._extract_inflection_note(inflection_result)
        except Exception as exc:
            warnings_list.append(f"{name}({code}) inflection failed: {type(exc).__name__}: {exc}")

        try:
            daily_df = self._load_daily_bars(market=market, code=code)
            if daily_df.empty:
                warnings_list.append(f"{name}({code}) missing daily bars")
            else:
                tech = self.technical_analyzer.analyze_bottom_structure(daily_df)
                bottom_score = _safe_float(tech.get("bottom_score"), 0.0)
        except Exception as exc:
            warnings_list.append(f"{name}({code}) technical failed: {type(exc).__name__}: {exc}")

        try:
            val = self.valuation_analyzer.compute_multi_percentile(
                market=market,
                code=code,
                lookback_years=5,
            )
            vp = val.get("composite_percentile")
            if vp is None:
                warnings_list.append(f"{name}({code}) valuation percentile missing, fallback 50")
                valuation_percentile = 50.0
            else:
                valuation_percentile = _safe_float(vp, 50.0)
        except Exception as exc:
            warnings_list.append(f"{name}({code}) valuation failed: {type(exc).__name__}: {exc}")
            valuation_percentile = 50.0

        try:
            if market == "A" and code.isdigit() and len(code) == 6:
                cap = self.capital_analyzer.composite_capital_signal(code)
                capital_score = _safe_float(cap.get("capital_score"), 0.0)
                capital_grade = str(cap.get("grade", "中性") or "中性")
            else:
                capital_score = 0.0
                capital_grade = "中性"
        except Exception as exc:
            warnings_list.append(f"{name}({code}) capital failed: {type(exc).__name__}: {exc}")

        composite = (
            inflection_score * _safe_float(weights.get("inflection"), 0.35)
            + bottom_score * _safe_float(weights.get("bottom"), 0.20)
            + (100.0 - valuation_percentile) * _safe_float(weights.get("valuation"), 0.30)
            + capital_score * _safe_float(weights.get("capital"), 0.15)
        )

        row = {
            "market": market,
            "code": code,
            "name": name,
            "inflection_score": round(inflection_score, 2),
            "bottom_score": round(bottom_score, 2),
            "valuation_percentile": round(valuation_percentile, 2),
            "capital_score": round(capital_score, 2),
            "capital_grade": capital_grade,
            "composite_score": round(composite, 2),
            "inflection_note": inflection_note,
            "source": str(item.get("source", "") or ""),
        }
        return row, warnings_list

    def _build_daily_report_text(
        self,
        report_date: date,
        hits: List[Dict[str, Any]],
        total_candidates: int,
        elapsed_seconds: float,
        threshold: float,
    ) -> str:
        strong = [x for x in hits if _safe_float(x.get("composite_score"), 0.0) > 80]
        watch = [x for x in hits if threshold < _safe_float(x.get("composite_score"), 0.0) <= 80]

        lines: List[str] = []
        lines.append(f"📡 拐点雷达日报 {report_date.strftime('%Y-%m-%d')}")
        lines.append("🔴 强信号（综合 > 80）")

        idx = 1
        if strong:
            for row in strong:
                lines.append(
                    f"{idx}. {row['name']}({row['code']}) | 综合{_safe_float(row['composite_score']):.0f} | "
                    f"拐点{_safe_float(row['inflection_score']):.0f} | 底部{_safe_float(row['bottom_score']):.0f} | "
                    f"估值分位{_safe_float(row['valuation_percentile']):.0f}% | 资金{row.get('capital_grade', '中性')}"
                )
                note = str(row.get("inflection_note", "") or "").strip() or "无显著拐点细节"
                lines.append(f"   ▸ {note}")
                idx += 1
        else:
            lines.append("暂无")

        lines.append("🟡 关注信号（综合 60-80）")
        if watch:
            for row in watch:
                lines.append(
                    f"{idx}. {row['name']}({row['code']}) | 综合{_safe_float(row['composite_score']):.0f} | "
                    f"拐点{_safe_float(row['inflection_score']):.0f} | 底部{_safe_float(row['bottom_score']):.0f} | "
                    f"估值分位{_safe_float(row['valuation_percentile']):.0f}% | 资金{row.get('capital_grade', '中性')}"
                )
                note = str(row.get("inflection_note", "") or "").strip() or "无显著拐点细节"
                lines.append(f"   ▸ {note}")
                idx += 1
        else:
            lines.append("暂无")

        lines.append("────────────────")
        lines.append(
            f"扫描范围: {total_candidates}只 | 命中: {len(hits)}只 | 耗时: {elapsed_seconds / 60.0:.1f}分钟"
        )
        return "\n".join(lines)

    def _push_report(self, title: str, markdown: str) -> bool:
        channel_cfg = str(((self.config.get("push", {}) or {}).get("channel", "") or "")).strip().lower()
        try:
            worker = AlertWorker(config_path=self.alert_config_path)
        except Exception as exc:
            logger.warning("push init failed: %s", exc)
            return False

        push_cfg = worker.config.get("push", {}) if isinstance(worker.config, dict) else {}
        default_channel = str(push_cfg.get("channel", "telegram") or "telegram").strip().lower()
        channel = channel_cfg or default_channel

        try:
            if channel == "serverchan":
                return bool(worker._push_serverchan(title, markdown))
            if channel == "pushplus":
                return bool(worker._push_pushplus(title, markdown))
            return bool(worker._push_telegram(title, markdown))
        except Exception as exc:
            logger.warning("push failed via %s: %s", channel, exc)
            return False

    def _save_report_json(
        self,
        report_date: date,
        all_rows: List[Dict[str, Any]],
        hits: List[Dict[str, Any]],
        warnings_list: List[str],
        report_text: str,
        elapsed_seconds: float,
    ) -> Path:
        out_dir = PROJECT_ROOT / "data" / "inflection_reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{report_date.strftime('%Y-%m-%d')}.json"

        payload = {
            "report_date": report_date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": str(self.config_path),
            "stats": {
                "candidates": len(all_rows),
                "hits": len(hits),
                "elapsed_seconds": round(float(elapsed_seconds), 3),
            },
            "hits": hits,
            "results": all_rows,
            "warnings": warnings_list,
            "report_text": report_text,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def _upsert_duckdb(self, report_date: date, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        init_duckdb()

        db_rows: List[Dict[str, Any]] = []
        now_ts = datetime.now(self.tz)
        for row in rows:
            db_rows.append(
                {
                    "trade_date": report_date,
                    "market": str(row.get("market", "A") or "A"),
                    "code": str(row.get("code", "") or ""),
                    "name": str(row.get("name", "") or ""),
                    "inflection_score": _safe_float(row.get("inflection_score"), 0.0),
                    "bottom_score": _safe_float(row.get("bottom_score"), 0.0),
                    "valuation_percentile": _safe_float(row.get("valuation_percentile"), 50.0),
                    "capital_score": _safe_float(row.get("capital_score"), 0.0),
                    "composite_score": _safe_float(row.get("composite_score"), 0.0),
                    "capital_grade": str(row.get("capital_grade", "中性") or "中性"),
                    "inflection_note": str(row.get("inflection_note", "") or ""),
                    "hit": int(_safe_float(row.get("composite_score"), 0.0) > self.composite_threshold),
                    "updated_at": now_ts,
                }
            )

        df = pd.DataFrame(db_rows)
        with connect_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inflection_daily (
                    trade_date DATE,
                    market TEXT,
                    code TEXT,
                    name TEXT,
                    inflection_score DOUBLE,
                    bottom_score DOUBLE,
                    valuation_percentile DOUBLE,
                    capital_score DOUBLE,
                    composite_score DOUBLE,
                    capital_grade TEXT,
                    inflection_note TEXT,
                    hit INTEGER,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (trade_date, market, code)
                )
                """
            )
            conn.register("scan_rows", df)
            conn.execute(
                """
                MERGE INTO inflection_daily AS t
                USING scan_rows AS s
                ON t.trade_date = s.trade_date AND t.market = s.market AND t.code = s.code
                WHEN MATCHED THEN UPDATE SET
                    name = s.name,
                    inflection_score = s.inflection_score,
                    bottom_score = s.bottom_score,
                    valuation_percentile = s.valuation_percentile,
                    capital_score = s.capital_score,
                    composite_score = s.composite_score,
                    capital_grade = s.capital_grade,
                    inflection_note = s.inflection_note,
                    hit = s.hit,
                    updated_at = s.updated_at
                WHEN NOT MATCHED THEN INSERT (
                    trade_date,
                    market,
                    code,
                    name,
                    inflection_score,
                    bottom_score,
                    valuation_percentile,
                    capital_score,
                    composite_score,
                    capital_grade,
                    inflection_note,
                    hit,
                    updated_at
                ) VALUES (
                    s.trade_date,
                    s.market,
                    s.code,
                    s.name,
                    s.inflection_score,
                    s.bottom_score,
                    s.valuation_percentile,
                    s.capital_score,
                    s.composite_score,
                    s.capital_grade,
                    s.inflection_note,
                    s.hit,
                    s.updated_at
                )
                """
            )

    def run_once(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        scan_cfg = (self.config.get("scan", {}) or {})
        max_candidates = int(scan_cfg.get("max_candidates", 200) or 200)
        filter_min_score = float(scan_cfg.get("filter_min_score", 60) or 60)
        auto_expand = bool(scan_cfg.get("auto_expand_from_filter", True))
        threshold = float(scan_cfg.get("composite_threshold", 60) or 60)
        weights = (self.config.get("weights", {}) or {})

        watch = self._load_watchlist_candidates()
        expanded = self._load_filter_candidates(filter_min_score, max_candidates) if auto_expand else []
        candidates = self._merge_candidates(watch, expanded, max_candidates=max_candidates)

        if not candidates:
            logger.warning("candidate universe empty")
            report_date = datetime.now(self.tz).date()
            empty_text = (
                f"📡 拐点雷达日报 {report_date.strftime('%Y-%m-%d')}\n"
                "暂无候选股票（watchlist + auto_expand 均为空）。"
            )
            saved = self._save_report_json(
                report_date=report_date,
                all_rows=[],
                hits=[],
                warnings_list=["candidate universe empty"],
                report_text=empty_text,
                elapsed_seconds=(time.perf_counter() - t0),
            )
            return {
                "report_date": report_date.strftime("%Y-%m-%d"),
                "candidates": 0,
                "hits": 0,
                "report_path": str(saved),
                "pushed": False,
            }

        logger.info("scan started, candidates=%d", len(candidates))
        rows: List[Dict[str, Any]] = []
        warns: List[str] = []
        total = len(candidates)
        progress_step = max(1, total // 20)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._scan_one, one, weights): one for one in candidates}
            done = 0
            for fut in as_completed(futs):
                item = futs[fut]
                done += 1
                try:
                    row, one_warns = fut.result()
                    rows.append(row)
                    warns.extend(one_warns)
                except Exception as exc:
                    warns.append(f"{item.get('name')}({item.get('code')}) worker failed: {type(exc).__name__}: {exc}")
                if done == 1 or done % progress_step == 0 or done == total:
                    logger.info("scan progress %d/%d (%.1f%%)", done, total, done * 100.0 / total)

        rows = sorted(rows, key=lambda x: _safe_float(x.get("composite_score"), 0.0), reverse=True)
        hits = [x for x in rows if _safe_float(x.get("composite_score"), 0.0) > threshold]

        elapsed = time.perf_counter() - t0
        report_date = datetime.now(self.tz).date()
        report_text = self._build_daily_report_text(
            report_date=report_date,
            hits=hits,
            total_candidates=len(rows),
            elapsed_seconds=elapsed,
            threshold=threshold,
        )

        title = f"拐点雷达日报 {report_date.strftime('%Y-%m-%d')}"
        pushed = self._push_report(title=title, markdown=report_text)
        if pushed:
            logger.info("daily report pushed, hits=%d", len(hits))
        else:
            logger.warning("daily report push failed or disabled")

        report_path = self._save_report_json(
            report_date=report_date,
            all_rows=rows,
            hits=hits,
            warnings_list=warns,
            report_text=report_text,
            elapsed_seconds=elapsed,
        )
        self._upsert_duckdb(report_date=report_date, rows=rows)

        logger.info(
            "scan finished: candidates=%d hits=%d elapsed=%.2fs json=%s",
            len(rows),
            len(hits),
            elapsed,
            report_path,
        )

        return {
            "report_date": report_date.strftime("%Y-%m-%d"),
            "candidates": len(rows),
            "hits": len(hits),
            "warnings": len(warns),
            "elapsed_seconds": round(float(elapsed), 3),
            "report_path": str(report_path),
            "pushed": bool(pushed),
        }

    def start(self) -> None:
        schedule = str(((self.config.get("scan", {}) or {}).get("schedule", "18:00") or "18:00")).strip()
        hh, mm = _parse_hhmm(schedule)
        scheduler = BlockingScheduler(timezone=self.tz)
        scheduler.add_job(
            self.run_once,
            trigger="cron",
            hour=hh,
            minute=mm,
            id="inflection_daily_scan",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        logger.info("inflection scanner started: daily at %02d:%02d (%s)", hh, mm, self.tz)
        scheduler.start()


def run_daily_scan(config_path: str = "config/inflection_scan.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    scanner = InflectionScanner(config_path=path)
    return scanner.run_once()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quant inflection daily scanner")
    p.add_argument("--config", default=str(PROJECT_ROOT / "config" / "inflection_scan.yaml"), help="scan config yaml path")
    p.add_argument("--once", action="store_true", help="run once then exit")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    scanner = InflectionScanner(config_path=config_path)
    if args.once:
        result = scanner.run_once()
        logger.info("once result: %s", json.dumps(result, ensure_ascii=False))
        return
    scanner.start()


if __name__ == "__main__":
    main()
