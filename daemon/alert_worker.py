from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
os.environ.setdefault("DISABLE_STREAMLIT_RUNTIME_CACHE", "1")

import requests
import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADING_DIR = PROJECT_ROOT / "apps" / "trading"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRADING_DIR) not in sys.path:
    sys.path.insert(0, str(TRADING_DIR))

from fast_engine import fetch_intraday_flow, fetch_realtime_quote, fetch_technical_indicators
from slow_engine import get_stock_group_map, get_stock_pool, init_db
from shared.capital_flow_signals import CapitalFlowAnalyzer


logger = logging.getLogger("alert_worker")


@dataclass
class AlertItem:
    rule: str
    code: str
    name: str
    group: str
    title: str
    body: str
    dedup_key: str
    level: str = "warn"


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return float(v)
    except Exception:
        return None


def _parse_hhmm(text: str) -> Tuple[int, int]:
    hh, mm = str(text).strip().split(":")
    return int(hh), int(mm)


class AlertWorker:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config()
        general = self.config.get("general", {}) if isinstance(self.config, dict) else {}
        tz_name = str(general.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
        self.tz = ZoneInfo(tz_name)
        self.state_path = PROJECT_ROOT / str(general.get("state_file", "daemon/.alert_state.json"))
        self.cooldown_minutes = int(general.get("dedup_cooldown_minutes", 20) or 20)
        self.capital_flow_analyzer = CapitalFlowAnalyzer(request_interval_sec=1.0)

    @staticmethod
    def _is_a_share_code(code: str) -> bool:
        c = str(code or "").strip()
        return c.isdigit() and len(c) == 6

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        obj = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(obj, dict):
            raise ValueError("配置文件格式错误，顶层应为 YAML 对象")
        return obj

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"sent": {}}
        try:
            obj = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {"sent": {}}
            if "sent" not in obj or not isinstance(obj["sent"], dict):
                obj["sent"] = {}
            return obj
        except Exception:
            return {"sent": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def should_run_now(self) -> bool:
        now = datetime.now(self.tz)
        if now.weekday() >= 5:
            return False
        sessions = (self.config.get("general", {}) or {}).get("trading_sessions", [])
        now_min = now.hour * 60 + now.minute
        for s in sessions:
            if not isinstance(s, dict):
                continue
            try:
                sh, sm = _parse_hhmm(s.get("start", "09:30"))
                eh, em = _parse_hhmm(s.get("end", "11:30"))
            except Exception:
                continue
            start_min = sh * 60 + sm
            end_min = eh * 60 + em
            if start_min <= now_min <= end_min:
                return True
        return False

    def _load_watchlist(self) -> List[Dict[str, str]]:
        init_db()
        rows = get_stock_pool()
        group_map = get_stock_group_map()
        out: List[Dict[str, str]] = []
        for code, name in rows:
            c = str(code).strip()
            n = str(name).strip()
            if not c:
                continue
            out.append(
                {
                    "code": c,
                    "name": n or c,
                    "group": str(group_map.get(c, "watch")),
                }
            )
        return out

    def _check_one_symbol(self, item: Dict[str, str]) -> List[AlertItem]:
        code = item["code"]
        name = item["name"]
        group = item["group"]
        rules = self.config.get("rules", {}) if isinstance(self.config, dict) else {}
        capital_cfg = rules.get("capital_flow", {}) or {}
        need_capital = bool(capital_cfg.get("enabled", False)) and self._is_a_share_code(code)

        with ThreadPoolExecutor(max_workers=4 if need_capital else 3) as tp:
            f_quote = tp.submit(fetch_realtime_quote, code)
            f_intra = tp.submit(fetch_intraday_flow, code)
            f_tech = tp.submit(fetch_technical_indicators, code)
            f_capital = (
                tp.submit(self.capital_flow_analyzer.composite_capital_signal, code)
                if need_capital
                else None
            )
            quote = f_quote.result()
            intraday = f_intra.result()
            tech = f_tech.result()
            capital_signal = f_capital.result() if f_capital is not None else {}

        alerts: List[AlertItem] = []
        alerts.extend(self._rule_price_breakout(code, name, group, quote, rules.get("price_breakout", {})))
        alerts.extend(self._rule_orderbook_dump(code, name, group, quote, intraday, rules.get("orderbook_dump", {})))
        alerts.extend(self._rule_technical_warning(code, name, group, tech, rules.get("technical_warning", {})))
        alerts.extend(self._rule_capital_flow(code, name, group, capital_signal, capital_cfg))
        return alerts

    # Backward compatibility for older call sites.
    def _scan_symbol(self, item: Dict[str, str]) -> List[AlertItem]:
        return self._check_one_symbol(item)

    def _rule_price_breakout(
        self,
        code: str,
        name: str,
        group: str,
        quote: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> List[AlertItem]:
        if not bool(cfg.get("enabled", False)):
            return []
        cur = _safe_float((quote or {}).get("current_price"))
        prev_close = _safe_float((quote or {}).get("prev_close"))
        if cur is None:
            return []

        out: List[AlertItem] = []
        level_cfg = ((cfg.get("symbol_levels", {}) or {}).get(str(code), {}) or {})
        above = _safe_float(level_cfg.get("above"))
        below = _safe_float(level_cfg.get("below"))
        move_pct = _safe_float(((cfg.get("default", {}) or {}).get("move_pct")))
        pct = None
        if prev_close and prev_close > 0:
            pct = (cur - prev_close) / prev_close * 100.0

        if above is not None and cur >= above:
            out.append(
                AlertItem(
                    rule="A",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 突破心理价位",
                    body=f"现价 {cur:.2f} 已上破设定价位 {above:.2f}",
                    dedup_key=f"A:{code}:above:{above}",
                )
            )
        if below is not None and cur <= below:
            out.append(
                AlertItem(
                    rule="A",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 跌破心理价位",
                    body=f"现价 {cur:.2f} 已跌破设定价位 {below:.2f}",
                    dedup_key=f"A:{code}:below:{below}",
                )
            )
        if move_pct is not None and pct is not None and abs(pct) >= move_pct:
            out.append(
                AlertItem(
                    rule="A",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 波动超阈值",
                    body=f"当前涨跌幅 {pct:+.2f}% 超过阈值 {move_pct:.2f}%",
                    dedup_key=f"A:{code}:pct:{int(move_pct*100)}:{'up' if pct >= 0 else 'down'}",
                )
            )
        return out

    def _rule_orderbook_dump(
        self,
        code: str,
        name: str,
        group: str,
        quote: Dict[str, Any],
        intraday: pd.DataFrame,
        cfg: Dict[str, Any],
    ) -> List[AlertItem]:
        if not bool(cfg.get("enabled", False)):
            return []

        min_sell_lot = float(cfg.get("min_sell_lot", 10000) or 10000)
        sell_buy_ratio = float(cfg.get("sell_buy_ratio", 3.0) or 3.0)
        spike_ratio = float(cfg.get("intraday_volume_spike_ratio", 2.5) or 2.5)
        lookback_points = int(cfg.get("lookback_points", 30) or 30)

        ob5 = (quote or {}).get("order_book_5", {}) or {}
        sells = ob5.get("sell", []) or []
        buys = ob5.get("buy", []) or []
        sell1 = next((x for x in sells if int(x.get("level", 0) or 0) == 1), {})
        buy1 = next((x for x in buys if int(x.get("level", 0) or 0) == 1), {})
        sell1_vol = _safe_float(sell1.get("volume_lot")) or 0.0
        buy1_vol = _safe_float(buy1.get("volume_lot")) or 0.0
        ratio = (sell1_vol / buy1_vol) if buy1_vol > 0 else (10.0 if sell1_vol > 0 else 0.0)

        vol_spike = False
        latest_lot = None
        base_lot = None
        if isinstance(intraday, pd.DataFrame) and (not intraday.empty) and ("volume_lot" in intraday.columns):
            ser = pd.to_numeric(intraday["volume_lot"], errors="coerce").dropna()
            if len(ser) >= 5:
                latest_lot = float(ser.iloc[-1])
                base_lot = float(ser.tail(max(lookback_points, 5)).median())
                if base_lot > 0 and latest_lot / base_lot >= spike_ratio:
                    vol_spike = True

        if sell1_vol >= min_sell_lot and ratio >= sell_buy_ratio and vol_spike:
            body = (
                f"卖1挂单 {sell1_vol:,.0f}（阈值 {min_sell_lot:,.0f}），"
                f"卖买比 {ratio:.2f}（阈值 {sell_buy_ratio:.2f}），"
                f"分时量突增 {latest_lot or 0:,.0f}/{base_lot or 0:,.0f}。"
            )
            return [
                AlertItem(
                    rule="B",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 盘口疑似大砸单",
                    body=body,
                    dedup_key=f"B:{code}:dump",
                    level="high",
                )
            ]
        return []

    def _rule_technical_warning(
        self,
        code: str,
        name: str,
        group: str,
        tech: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> List[AlertItem]:
        if not bool(cfg.get("enabled", False)):
            return []
        rsi_overbought = float(cfg.get("rsi_overbought", 80) or 80)
        rsi_oversold = float(cfg.get("rsi_oversold", 20) or 20)
        macd_flip_check = bool(cfg.get("macd_flip_check", True))

        rsi6 = _safe_float((tech or {}).get("rsi6"))
        macd = _safe_float((tech or {}).get("macd_hist"))
        out: List[AlertItem] = []
        if rsi6 is None:
            return out

        if rsi6 >= rsi_overbought:
            body = f"RSI6={rsi6:.2f} 已进入超买区（阈值 {rsi_overbought:.2f}）"
            if macd_flip_check and macd is not None and macd < 0:
                body += "，且 MACD 柱转负。"
            out.append(
                AlertItem(
                    rule="C",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 技术面超买警惕",
                    body=body,
                    dedup_key=f"C:{code}:overbought",
                )
            )
        if rsi6 <= rsi_oversold:
            body = f"RSI6={rsi6:.2f} 已进入超卖区（阈值 {rsi_oversold:.2f}）"
            if macd_flip_check and macd is not None and macd > 0:
                body += "，且 MACD 柱转正。"
            out.append(
                AlertItem(
                    rule="C",
                    code=code,
                    name=name,
                    group=group,
                    title=f"{name}({code}) 技术面超卖警惕",
                    body=body,
                    dedup_key=f"C:{code}:oversold",
                )
            )
        return out

    def _rule_capital_flow(
        self,
        code: str,
        name: str,
        group: str,
        signal: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> List[AlertItem]:
        if not bool(cfg.get("enabled", False)):
            return []
        if not self._is_a_share_code(code):
            return []

        score_threshold = float(cfg.get("composite_score_alert", 70) or 70)
        northbound_min_days = int(cfg.get("northbound_min_days", 3) or 3)
        northbound_alert_days = max(5, northbound_min_days)
        block_trade_premium_min = float(cfg.get("block_trade_premium_min", 0) or 0)

        signal = signal or {}
        northbound = (signal.get("northbound", {}) or {})
        block_trade = (signal.get("block_trade", {}) or {})
        capital_score = _safe_float(signal.get("capital_score")) or 0.0
        grade = str(signal.get("grade", "中性") or "中性")
        consecutive_buy_days = int(_safe_float(northbound.get("consecutive_buy_days")) or 0)
        avg_premium_pct = _safe_float(block_trade.get("avg_premium_pct")) or 0.0

        if (capital_score <= score_threshold) and (consecutive_buy_days < northbound_alert_days):
            return []
        if avg_premium_pct < block_trade_premium_min:
            avg_premium_pct = block_trade_premium_min

        message = (
            f"📊 {name} 资金信号：{grade}，北向连续{consecutive_buy_days}天净买入，"
            f"大宗近期溢价{avg_premium_pct:.2f}%"
        )
        dedup_key = (
            f"D:{code}:score:{int(capital_score)}:"
            f"nb:{consecutive_buy_days}:bp:{int(round(avg_premium_pct * 100))}"
        )
        return [
            AlertItem(
                rule="D",
                code=code,
                name=name,
                group=group,
                title=f"{name}({code}) 资金行为拐点",
                body=message,
                dedup_key=dedup_key,
                level="high" if (capital_score >= 85 or consecutive_buy_days >= 7) else "warn",
            )
        ]

    def _dedup_and_mark(self, alerts: List[AlertItem]) -> List[AlertItem]:
        state = self._load_state()
        sent_map = state.get("sent", {}) if isinstance(state, dict) else {}
        now_ts = datetime.now(self.tz).timestamp()
        cooldown_sec = self.cooldown_minutes * 60

        keep: List[AlertItem] = []
        for a in alerts:
            last = sent_map.get(a.dedup_key)
            if isinstance(last, (int, float)) and (now_ts - float(last) < cooldown_sec):
                continue
            keep.append(a)
            sent_map[a.dedup_key] = now_ts

        # 清理过旧 key（2天）
        expire = now_ts - 2 * 24 * 3600
        sent_map = {k: v for k, v in sent_map.items() if isinstance(v, (int, float)) and float(v) >= expire}
        state["sent"] = sent_map
        self._save_state(state)
        return keep

    def _format_markdown(self, alerts: List[AlertItem]) -> str:
        ts = datetime.now(self.tz).strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"## Quant 自动预警 ({ts})",
            "",
            f"- 告警数量: **{len(alerts)}**",
            "",
        ]
        for a in alerts:
            lines.extend(
                [
                    f"### [{a.rule}] {a.title}",
                    f"- 分组: `{a.group}`",
                    f"- 说明: {a.body}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _push_serverchan(self, title: str, markdown: str) -> bool:
        cfg = ((self.config.get("push", {}) or {}).get("serverchan", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return False
        sendkey = str(cfg.get("sendkey", "") or "").strip()
        if not sendkey:
            return False
        url = f"https://sctapi.ftqq.com/{sendkey}.send"
        resp = requests.post(url, data={"title": title, "desp": markdown}, timeout=12)
        return resp.ok

    def _push_pushplus(self, title: str, markdown: str) -> bool:
        cfg = ((self.config.get("push", {}) or {}).get("pushplus", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return False
        token = str(cfg.get("token", "") or "").strip()
        if not token:
            return False
        payload = {
            "token": token,
            "title": title,
            "content": markdown,
            "template": "markdown",
            "topic": str(cfg.get("topic", "") or "").strip(),
        }
        resp = requests.post("http://www.pushplus.plus/send", json=payload, timeout=12)
        return resp.ok

    def _push_telegram(self, title: str, markdown: str) -> bool:
        cfg = ((self.config.get("push", {}) or {}).get("telegram", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return False
        bot_token = str(cfg.get("bot_token", "") or "").strip()
        chat_id = str(cfg.get("chat_id", "") or "").strip()
        api_base = str(cfg.get("api_base", "https://api.telegram.org") or "https://api.telegram.org").rstrip("/")
        if not bot_token or not chat_id:
            return False
        url = f"{api_base}/bot{bot_token}/sendMessage"
        text = f"*{title}*\n\n{markdown}"
        resp = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=12,
        )
        return resp.ok

    def _push_alerts(self, alerts: List[AlertItem]) -> bool:
        if not alerts:
            return True
        markdown = self._format_markdown(alerts)
        title = f"Quant 预警 x{len(alerts)}"

        push_cfg = self.config.get("push", {}) if isinstance(self.config, dict) else {}
        channel = str(push_cfg.get("channel", "telegram") or "telegram").strip().lower()

        try:
            if channel == "serverchan":
                return self._push_serverchan(title, markdown)
            if channel == "pushplus":
                return self._push_pushplus(title, markdown)
            return self._push_telegram(title, markdown)
        except Exception as exc:
            logger.exception("push failed: %s", exc)
            return False

    def _print_dry_run_signals(self, alerts: List[AlertItem]) -> None:
        capital_alerts = [a for a in alerts if str(a.rule).upper() == "D"]
        if not capital_alerts:
            logger.info("dry-run: no capital flow signals")
            return
        logger.info("dry-run capital flow signals: %d", len(capital_alerts))
        print(self._format_markdown(capital_alerts))

    def run_monitors(self, force: bool = False, dry_run: bool = False) -> int:
        if (not force) and (not self.should_run_now()):
            logger.info("skip monitor: not in trading sessions")
            return 0

        watchlist = self._load_watchlist()
        if not watchlist:
            logger.info("watchlist empty")
            return 0

        max_workers = int((self.config.get("general", {}) or {}).get("max_workers", 8) or 8)
        alerts: List[AlertItem] = []
        with ThreadPoolExecutor(max_workers=max(2, max_workers)) as ex:
            futs = {ex.submit(self._check_one_symbol, item): item for item in watchlist}
            for fut in as_completed(futs):
                item = futs[fut]
                try:
                    alerts.extend(fut.result())
                except Exception as exc:
                    logger.warning("scan failed %s(%s): %s", item.get("name"), item.get("code"), exc)

        alerts = self._dedup_and_mark(alerts)
        if not alerts:
            logger.info("no new alerts")
            return 0

        if dry_run:
            self._print_dry_run_signals(alerts)
            logger.info("dry-run enabled: push skipped")
            return len(alerts)

        ok = self._push_alerts(alerts)
        if ok:
            logger.info("alerts pushed: %d", len(alerts))
        else:
            logger.warning("alerts generated but push failed: %d", len(alerts))
        return len(alerts)

    def start(self) -> None:
        general = self.config.get("general", {}) if isinstance(self.config, dict) else {}
        interval = int(general.get("interval_minutes", 3) or 3)
        scheduler = BlockingScheduler(timezone=self.tz)
        scheduler.add_job(
            self.run_monitors,
            trigger="interval",
            minutes=max(1, interval),
            id="run_monitors",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
        )
        logger.info("alert worker started: every %d minutes", interval)
        scheduler.start()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quant alert daemon worker")
    p.add_argument("--config", default=str(PROJECT_ROOT / "config" / "alert_rules.yaml"), help="alert rules yaml path")
    p.add_argument("--once", action="store_true", help="run once then exit")
    p.add_argument("--force", action="store_true", help="run monitor even outside trading sessions")
    p.add_argument("--dry-run", action="store_true", help="run monitor without push, print capital flow signals only")
    p.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARN/ERROR")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    worker = AlertWorker(config_path=Path(args.config))
    if args.once:
        cnt = worker.run_monitors(force=bool(args.force), dry_run=bool(args.dry_run))
        logger.info("once finished, new alerts: %d", cnt)
        return
    if args.dry_run:
        logger.warning("--dry-run is only effective with --once in current implementation")
    worker.start()


if __name__ == "__main__":
    main()
