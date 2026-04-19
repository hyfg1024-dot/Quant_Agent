from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from shared.db_manager import connect_db, init_duckdb


class AgentMemory:
    REVIEW_HORIZON_TRADE_DAYS = 20

    def __init__(self) -> None:
        self._init_table()

    def _init_table(self) -> None:
        init_duckdb()
        with connect_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_predictions (
                    code TEXT NOT NULL,
                    name TEXT,
                    date DATE NOT NULL,
                    bull_score DOUBLE,
                    bear_score DOUBLE,
                    judge_verdict TEXT,
                    price DOUBLE,
                    target_action TEXT,
                    confidence DOUBLE,
                    analysis_result_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )

    @staticmethod
    def _normalize_code(code: str) -> str:
        text = str(code or "").strip()
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 6:
            return digits[-6:]
        if len(digits) == 5:
            return digits
        return text

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            out = float(value)
            if out != out:
                return None
            return out
        except Exception:
            return None

    @staticmethod
    def _to_date(value: Any) -> date:
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        text = str(value or "").strip()
        if not text:
            return datetime.now().date()
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
            try:
                return datetime.strptime(text, fmt).date()
            except Exception:
                continue
        try:
            return pd.to_datetime(text, errors="coerce").date()
        except Exception:
            return datetime.now().date()

    @staticmethod
    def _extract_verdict(judge_text: str) -> str:
        text = str(judge_text or "")
        m = re.search(r"主结论[：:\s]*([偏多偏空]+)", text)
        if m:
            v = m.group(1)
            if "偏多" in v:
                return "偏多"
            if "偏空" in v:
                return "偏空"
        if "偏多" in text and "偏空" not in text:
            return "偏多"
        if "偏空" in text and "偏多" not in text:
            return "偏空"
        return "未知"

    @classmethod
    def _extract_prediction_fields(cls, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        result = analysis_result if isinstance(analysis_result, dict) else {}
        meta = result.get("meta", {}) if isinstance(result.get("meta"), dict) else {}
        judge = result.get("judge", {}) if isinstance(result.get("judge"), dict) else {}
        judge_text = str(judge.get("text", "") or "")

        bull = cls._to_float(meta.get("bull_prob"))
        bear = cls._to_float(meta.get("bear_prob"))
        if bull is None or bear is None:
            m_bull = re.search(r"做多胜率[：:\s]*([0-9]{1,3})", judge_text)
            m_bear = re.search(r"做空胜率[：:\s]*([0-9]{1,3})", judge_text)
            if m_bull:
                bull = float(max(0, min(100, int(m_bull.group(1)))))
            if m_bear:
                bear = float(max(0, min(100, int(m_bear.group(1)))))
            if bull is None and bear is not None:
                bull = 100.0 - bear
            if bear is None and bull is not None:
                bear = 100.0 - bull

        bull = float(bull if bull is not None else 50.0)
        bear = float(bear if bear is not None else max(0.0, 100.0 - bull))
        if bull + bear != 100 and (bull + bear) > 0:
            bull = round(100.0 * bull / (bull + bear), 2)
            bear = round(100.0 - bull, 2)

        verdict = cls._extract_verdict(judge_text)
        if verdict == "偏多":
            action = "买入"
        elif verdict == "偏空":
            action = "卖出"
        else:
            if abs(bull - bear) <= 8:
                action = "观望"
            else:
                action = "买入" if bull > bear else "卖出"

        confidence = max(bull, bear)
        return {
            "bull_score": bull,
            "bear_score": bear,
            "judge_verdict": verdict,
            "target_action": action,
            "confidence": confidence,
        }

    def save_prediction(
        self,
        code: str,
        name: str,
        analysis_result: Dict[str, Any],
        price_at_analysis: Any,
        date: Any,
    ) -> None:
        normalized = self._normalize_code(code)
        if not normalized:
            return

        fields = self._extract_prediction_fields(analysis_result if isinstance(analysis_result, dict) else {})
        dt = self._to_date(date)
        px = self._to_float(price_at_analysis)
        try:
            payload = json.dumps(analysis_result if isinstance(analysis_result, dict) else {}, ensure_ascii=False)
        except Exception:
            payload = "{}"

        with connect_db() as conn:
            conn.execute(
                """
                INSERT INTO agent_predictions
                (code, name, date, bull_score, bear_score, judge_verdict, price, target_action, confidence, analysis_result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    normalized,
                    str(name or "").strip(),
                    dt,
                    float(fields.get("bull_score", 50.0)),
                    float(fields.get("bear_score", 50.0)),
                    str(fields.get("judge_verdict", "未知")),
                    None if px is None else float(px),
                    str(fields.get("target_action", "观望")),
                    float(fields.get("confidence", 50.0)),
                    payload,
                    datetime.now(),
                ],
            )

    @staticmethod
    def _prediction_label(action: str) -> str:
        if str(action) == "买入":
            return "看多"
        if str(action) == "卖出":
            return "看空"
        return "观望"

    @staticmethod
    def _direction_correct(action: str, actual_return: float) -> bool:
        a = str(action)
        if a == "买入":
            return actual_return > 0
        if a == "卖出":
            return actual_return < 0
        return abs(actual_return) <= 2.0

    @classmethod
    def _error_analysis(cls, action: str, actual_return: float, correct: bool) -> str:
        if correct:
            return "方向判断与实际走势一致。"
        a = str(action)
        if a == "买入" and actual_return <= 0:
            return "错判：看多但后续下跌，可能低估了政策/行业逆风。"
        if a == "卖出" and actual_return >= 0:
            return "错判：看空但后续上涨，可能高估了短期利空冲击。"
        return "错判：观望但后续波动显著，节奏把握不足。"

    def _fetch_prediction(self, code: str, prediction_date: Any) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_code(code)
        dt = self._to_date(prediction_date)
        with connect_db(read_only=True) as conn:
            row = conn.execute(
                """
                SELECT code, name, date, bull_score, bear_score, judge_verdict, price, target_action, confidence, analysis_result_json
                FROM agent_predictions
                WHERE code = ? AND date = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                [normalized, dt],
            ).fetchone()
        if not row:
            return None
        return {
            "code": row[0],
            "name": row[1],
            "date": row[2],
            "bull_score": row[3],
            "bear_score": row[4],
            "judge_verdict": row[5],
            "price": row[6],
            "target_action": row[7],
            "confidence": row[8],
            "analysis_result_json": row[9],
        }

    def review_prediction(self, code: str, prediction_date: Any, current_price: Any) -> Dict[str, Any]:
        row = self._fetch_prediction(code, prediction_date)
        if not row:
            return {
                "prediction": "未知",
                "actual_return": 0.0,
                "correct": False,
                "error_analysis": "未找到对应预测记录。",
            }

        old_price = self._to_float(row.get("price"))
        now_price = self._to_float(current_price)
        if old_price is None or old_price <= 0 or now_price is None:
            return {
                "prediction": self._prediction_label(str(row.get("target_action", "观望"))),
                "actual_return": 0.0,
                "correct": False,
                "error_analysis": "价格数据不足，无法复盘。",
            }

        actual_return = (now_price - old_price) / old_price * 100.0
        action = str(row.get("target_action", "观望"))
        correct = self._direction_correct(action, actual_return)
        return {
            "prediction": self._prediction_label(action),
            "actual_return": round(actual_return, 4),
            "correct": bool(correct),
            "error_analysis": self._error_analysis(action, actual_return, bool(correct)),
        }

    def _get_price_after_n_trade_days(self, code: str, prediction_date: Any, n: int = REVIEW_HORIZON_TRADE_DAYS) -> Optional[float]:
        normalized = self._normalize_code(code)
        dt = self._to_date(prediction_date)
        with connect_db(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT close
                FROM daily_kline
                WHERE code = ? AND trade_date > ?
                ORDER BY trade_date ASC
                LIMIT ?
                """,
                [normalized, dt, int(n)],
            ).fetchall()
        if not rows or len(rows) < int(n):
            return None
        try:
            return float(rows[-1][0])
        except Exception:
            return None

    def batch_review(self, lookback_days: int = 90) -> pd.DataFrame:
        cutoff = datetime.now().date() - timedelta(days=max(1, int(lookback_days)))
        with connect_db(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT code, name, date, price, target_action, confidence
                FROM agent_predictions
                WHERE date >= ?
                ORDER BY date ASC, created_at ASC
                """,
                [cutoff],
            ).fetchall()

        details: List[Dict[str, Any]] = []
        for code, name, dt, price, action, confidence in rows:
            old_price = self._to_float(price)
            future_price = self._get_price_after_n_trade_days(str(code), dt, self.REVIEW_HORIZON_TRADE_DAYS)
            if old_price is None or old_price <= 0 or future_price is None:
                details.append(
                    {
                        "code": str(code),
                        "name": str(name or ""),
                        "date": self._to_date(dt),
                        "prediction": self._prediction_label(str(action)),
                        "target_action": str(action),
                        "confidence": self._to_float(confidence),
                        "actual_return": None,
                        "correct": None,
                        "error_analysis": "待验证",
                    }
                )
                continue

            actual_return = (future_price - old_price) / old_price * 100.0
            correct = self._direction_correct(str(action), actual_return)
            details.append(
                {
                    "code": str(code),
                    "name": str(name or ""),
                    "date": self._to_date(dt),
                    "prediction": self._prediction_label(str(action)),
                    "target_action": str(action),
                    "confidence": self._to_float(confidence),
                    "actual_return": round(actual_return, 4),
                    "correct": bool(correct),
                    "error_analysis": self._error_analysis(str(action), actual_return, bool(correct)),
                }
            )

        df = pd.DataFrame(details)
        if df.empty:
            return pd.DataFrame(
                [
                    {
                        "total_predictions": 0,
                        "accuracy_pct": 0.0,
                        "avg_return_pct": 0.0,
                        "max_miss_return_pct": 0.0,
                    }
                ]
            )

        verified = df[df["correct"].notna()].copy()
        total = int(len(verified))
        if total <= 0:
            summary = {
                "total_predictions": int(len(df)),
                "accuracy_pct": 0.0,
                "avg_return_pct": 0.0,
                "max_miss_return_pct": 0.0,
            }
        else:
            acc = float(verified["correct"].astype(bool).mean() * 100.0)
            avg_ret = float(pd.to_numeric(verified["actual_return"], errors="coerce").mean())
            wrong = verified[verified["correct"] == False]  # noqa: E712
            if wrong.empty:
                max_miss = 0.0
            else:
                max_miss = float(pd.to_numeric(wrong["actual_return"], errors="coerce").abs().max())
            summary = {
                "total_predictions": total,
                "accuracy_pct": round(acc, 4),
                "avg_return_pct": round(avg_ret, 4),
                "max_miss_return_pct": round(max_miss, 4),
            }

        for k, v in summary.items():
            df[k] = v
        return df

    def _fetch_recent_predictions(self, code: str, months: int = 6) -> List[Dict[str, Any]]:
        normalized = self._normalize_code(code)
        cutoff = datetime.now().date() - timedelta(days=max(1, int(months * 30)))
        with connect_db(read_only=True) as conn:
            rows = conn.execute(
                """
                SELECT code, name, date, price, target_action, confidence, analysis_result_json
                FROM agent_predictions
                WHERE code = ? AND date >= ?
                ORDER BY date ASC, created_at ASC
                """,
                [normalized, cutoff],
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "code": r[0],
                    "name": r[1],
                    "date": self._to_date(r[2]),
                    "price": self._to_float(r[3]),
                    "target_action": str(r[4] or "观望"),
                    "confidence": self._to_float(r[5]),
                    "analysis_result_json": str(r[6] or ""),
                }
            )
        return out

    def get_track_record(self, code: str) -> str:
        rows = self._fetch_recent_predictions(code, months=6)
        if not rows:
            return f"过去6个月对{self._normalize_code(code)}暂无历史分析。"

        display_name = next((str(x.get("name") or "").strip() for x in reversed(rows) if str(x.get("name") or "").strip()), "")
        if not display_name:
            display_name = self._normalize_code(code)

        lines = [f"过去6个月对{display_name}做过{len(rows)}次分析："]
        for one in rows:
            dt = self._to_date(one.get("date"))
            action = str(one.get("target_action", "观望"))
            label = self._prediction_label(action)
            conf = int(round(float(one.get("confidence") or 0.0)))
            old_price = self._to_float(one.get("price"))
            future_price = self._get_price_after_n_trade_days(str(one.get("code")), dt, self.REVIEW_HORIZON_TRADE_DAYS)
            if old_price is None or old_price <= 0 or future_price is None:
                lines.append(f"{dt.strftime('%Y-%m')}: {label}(置信度{conf}%) → 待验证")
                continue

            actual_return = (future_price - old_price) / old_price * 100.0
            correct = self._direction_correct(action, actual_return)
            mark = "✅" if correct else "❌"
            if correct:
                lines.append(f"{dt.strftime('%Y-%m')}: {label}(置信度{conf}%) → 实际{actual_return:+.1f}% {mark}")
            else:
                reason = self._error_analysis(action, actual_return, False).replace("错判：", "")
                lines.append(f"{dt.strftime('%Y-%m')}: {label}(置信度{conf}%) → 实际{actual_return:+.1f}% {mark}（错判：{reason}）")
        return "\n".join(lines)

    def generate_self_correction_prompt(self, code: str) -> str:
        rows = self._fetch_recent_predictions(code, months=6)
        if not rows:
            return ""

        reviewed: List[Dict[str, Any]] = []
        for one in rows:
            old_price = self._to_float(one.get("price"))
            dt = self._to_date(one.get("date"))
            future_price = self._get_price_after_n_trade_days(str(one.get("code")), dt, self.REVIEW_HORIZON_TRADE_DAYS)
            if old_price is None or old_price <= 0 or future_price is None:
                continue
            ret = (future_price - old_price) / old_price * 100.0
            action = str(one.get("target_action", "观望"))
            correct = self._direction_correct(action, ret)
            reviewed.append(
                {
                    "action": action,
                    "return": ret,
                    "correct": correct,
                    "error": self._error_analysis(action, ret, correct),
                    "analysis_result_json": str(one.get("analysis_result_json") or ""),
                }
            )

        if not reviewed:
            return ""

        total = len(reviewed)
        correct_cnt = sum(1 for x in reviewed if bool(x.get("correct")))
        accuracy = round(correct_cnt / max(1, total) * 100.0)
        wrong = [x for x in reviewed if not bool(x.get("correct"))]
        if not wrong:
            last_error = "暂无显著错误模式，但仍需警惕外部变量冲击。"
        else:
            wrong_actions = Counter(str(x.get("action")) for x in wrong)
            if wrong_actions.get("买入", 0) >= max(wrong_actions.get("卖出", 0), wrong_actions.get("观望", 0)):
                last_error = "过度依赖短期技术信号，低估了下行风险或外部利空。"
            elif wrong_actions.get("卖出", 0) >= max(wrong_actions.get("买入", 0), wrong_actions.get("观望", 0)):
                last_error = "对利空过度敏感，忽视了趋势延续和反转力量。"
            else:
                last_error = "观望阈值设置偏保守，错过了方向性波段。"

        return (
            f"【历史复盘警示】你过去对该股票的预测正确率为 {accuracy}%。\n"
            f"上次的主要错误是：{last_error}\n"
            "本次分析请特别注意：1) 政策面变化 2) 行业竞争格局变动。"
        )
