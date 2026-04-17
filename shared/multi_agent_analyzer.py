from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


LLMCall = Callable[[str, str, int, float, float], Tuple[str, Dict[str, Any], float, float]]


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    max_tokens: int = 1200
    temperature: float = 0.3
    top_p: float = 0.9


class BaseAgent:
    def __init__(self, config: AgentConfig):
        self.config = config

    async def run(self, user_content: str, llm_call: LLMCall) -> Dict[str, Any]:
        started = time.time()
        try:
            text, usage, cost, elapsed = await asyncio.to_thread(
                llm_call,
                user_content,
                self.config.system_prompt,
                self.config.max_tokens,
                self.config.temperature,
                self.config.top_p,
            )
            return {
                "agent": self.config.name,
                "ok": True,
                "text": str(text or "").strip(),
                "usage": usage if isinstance(usage, dict) else {},
                "cost": float(cost or 0.0),
                "elapsed": float(elapsed or 0.0),
                "started_at": started,
            }
        except Exception as exc:
            return {
                "agent": self.config.name,
                "ok": False,
                "text": f"{self.config.name} 调用失败: {type(exc).__name__}: {exc}",
                "usage": {},
                "cost": 0.0,
                "elapsed": time.time() - started,
                "started_at": started,
                "error": f"{type(exc).__name__}: {exc}",
            }


class FundamentalAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            AgentConfig(
                name="FundamentalAgent",
                system_prompt=(
                    "你是 FundamentalAgent（基本面空头专家）。你的职责是挑刺，不做中立总结。\n"
                    "请仅基于输入数据，优先从以下角度构建看空链条：\n"
                    "1) 高估值风险：高PE/高PB/低股息与增长错配；\n"
                    "2) 质量风险：现金流弱、杠杆高、盈利波动；\n"
                    "3) 安全边际不足：估值已透支预期。\n"
                    "输出格式：\n"
                    "- 空头核心论点（3-5条）\n"
                    "- 触发条件（2条）\n"
                    "- 未来一周看空概率（0-100）与置信度说明（50字内）\n"
                    "禁止和稀泥，禁止给双向建议。"
                ),
                max_tokens=900,
                temperature=0.25,
                top_p=0.9,
            )
        )


class TechnicalAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            AgentConfig(
                name="TechnicalAgent",
                system_prompt=(
                    "你是 TechnicalAgent（技术面多头专家）。你的职责是寻找可执行的看多结构，不做中立总结。\n"
                    "请仅基于输入数据，优先从以下角度构建看多链条：\n"
                    "1) 均线结构与趋势延续；\n"
                    "2) RSI/MACD/量价共振；\n"
                    "3) 资金净流入和盘口强弱。\n"
                    "输出格式：\n"
                    "- 多头核心论点（3-5条）\n"
                    "- 失效条件（2条）\n"
                    "- 未来一周看多概率（0-100）与置信度说明（50字内）\n"
                    "禁止和稀泥，禁止给双向建议。"
                ),
                max_tokens=900,
                temperature=0.25,
                top_p=0.9,
            )
        )


class EventAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            AgentConfig(
                name="EventAgent",
                system_prompt=(
                    "你是 EventAgent（消息面冲击分析专家）。请专注事件催化，不做财务细节复述。\n"
                    "请基于输入中的新闻/事件线索，判断：\n"
                    "1) 事件强度（弱/中/强）与持续性（天/周）；\n"
                    "2) 市场情绪偏向（偏多/偏空）与是否已计价；\n"
                    "3) 一周内可能触发的二次传播风险或增量利好。\n"
                    "输出格式：\n"
                    "- 事件要点（3条）\n"
                    "- 已计价/未计价判断（必须明确）\n"
                    "- 一周情绪方向：偏多或偏空（二选一）+ 概率（0-100）\n"
                    "禁止和稀泥。"
                ),
                max_tokens=900,
                temperature=0.3,
                top_p=0.9,
            )
        )


class JudgeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            AgentConfig(
                name="JudgeAgent",
                system_prompt=(
                    "你是 JudgeAgent（法官智能体）。你将收到三个专家的结论：\n"
                    "- FundamentalAgent（空头）\n"
                    "- TechnicalAgent（多头）\n"
                    "- EventAgent（事件）\n"
                    "你的任务是对抗式裁决：整合正反方核心论点并给出明确胜率。\n"
                    "强制要求：\n"
                    "1) 给出做多胜率%与做空胜率%，两者相加必须=100；\n"
                    "2) 必须给出唯一主结论：偏多或偏空（二选一）；\n"
                    "3) 不能输出“观望/中性/五五开”等和稀泥结论；\n"
                    "4) 给出未来一周的关键触发位与失效位。\n"
                    "输出格式：\n"
                    "## 法官裁决\n"
                    "- 做多胜率：xx%\n"
                    "- 做空胜率：xx%\n"
                    "- 主结论：偏多/偏空（择一）\n"
                    "- 裁决理由：3条\n"
                    "- 一周关键位：触发位、失效位\n"
                    "- 执行建议：仓位与风控（不超过80字）"
                ),
                max_tokens=1200,
                temperature=0.2,
                top_p=0.85,
            )
        )


class MultiAgentAnalyzer:
    def __init__(self):
        self.fund_agent = FundamentalAgent()
        self.tech_agent = TechnicalAgent()
        self.event_agent = EventAgent()
        self.judge_agent = JudgeAgent()

    @staticmethod
    def _merge_usage(items: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
        }
        for one in items:
            usage = one.get("usage", {}) if isinstance(one, dict) else {}
            for k in out.keys():
                try:
                    out[k] += int(usage.get(k, 0) or 0)
                except Exception:
                    continue
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
        return out

    @staticmethod
    def _build_judge_input(raw_payload: str, experts: List[Dict[str, Any]]) -> str:
        expert_map = {str(x.get("agent", "")): str(x.get("text", "")) for x in experts if isinstance(x, dict)}
        obj = {
            "input_snapshot": raw_payload,
            "experts": {
                "fundamental_bear": expert_map.get("FundamentalAgent", ""),
                "technical_bull": expert_map.get("TechnicalAgent", ""),
                "event_impact": expert_map.get("EventAgent", ""),
            },
            "instruction": "请作为法官做出二选一裁决并给出做多/做空胜率。",
        }
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _compose_final_markdown(experts: List[Dict[str, Any]], judge: Dict[str, Any]) -> str:
        parts: List[str] = ["## 多智能体对抗结果"]
        order = ["FundamentalAgent", "TechnicalAgent", "EventAgent"]
        title_map = {
            "FundamentalAgent": "### FundamentalAgent（基本面空头）",
            "TechnicalAgent": "### TechnicalAgent（技术面多头）",
            "EventAgent": "### EventAgent（消息面）",
        }
        by_name = {str(x.get("agent", "")): x for x in experts if isinstance(x, dict)}
        for name in order:
            one = by_name.get(name, {})
            parts.append(title_map[name])
            parts.append(str(one.get("text", "无输出")))
        parts.append("### JudgeAgent（法官裁决）")
        parts.append(str(judge.get("text", "无裁决输出")))
        return "\n\n".join(parts).strip()

    @staticmethod
    def _detect_event_side(text: str) -> str:
        t = str(text or "")
        if "偏多" in t and "偏空" not in t:
            return "bull"
        if "偏空" in t and "偏多" not in t:
            return "bear"
        return "neutral"

    @staticmethod
    def _extract_judge_probs(text: str) -> Dict[str, int]:
        t = str(text or "")
        bull = None
        bear = None
        m_bull = re.search(r"做多胜率[：:\s]*([0-9]{1,3})\s*%", t)
        m_bear = re.search(r"做空胜率[：:\s]*([0-9]{1,3})\s*%", t)
        if m_bull:
            bull = max(0, min(100, int(m_bull.group(1))))
        if m_bear:
            bear = max(0, min(100, int(m_bear.group(1))))
        if bull is None and bear is None:
            return {}
        if bull is None:
            bull = max(0, min(100, 100 - int(bear)))
        if bear is None:
            bear = max(0, min(100, 100 - int(bull)))
        s = int(bull) + int(bear)
        if s != 100 and s > 0:
            bull = int(round(100 * bull / s))
            bear = 100 - bull
        return {"bull_prob": int(bull), "bear_prob": int(bear)}

    @classmethod
    def _build_meta(cls, experts: List[Dict[str, Any]], judge: Dict[str, Any]) -> Dict[str, Any]:
        by_name = {str(x.get("agent", "")): x for x in experts if isinstance(x, dict)}
        event_text = str((by_name.get("EventAgent") or {}).get("text", ""))
        event_side = cls._detect_event_side(event_text)

        bull_votes = 1  # TechnicalAgent
        bear_votes = 1  # FundamentalAgent
        if event_side == "bull":
            bull_votes += 1
        elif event_side == "bear":
            bear_votes += 1
        total_votes = bull_votes + bear_votes
        agreement = max(bull_votes, bear_votes) / max(1, total_votes)
        disagreement = 1.0 - agreement
        consensus = "偏多" if bull_votes > bear_votes else "偏空" if bear_votes > bull_votes else "分歧"

        judge_probs = cls._extract_judge_probs(str(judge.get("text", "")))
        out = {
            "event_side": event_side,
            "bull_votes": bull_votes,
            "bear_votes": bear_votes,
            "agreement_score": round(agreement, 4),
            "disagreement_score": round(disagreement, 4),
            "consensus_side": consensus,
        }
        out.update(judge_probs)
        return out

    async def analyze(self, user_content: str, llm_call: LLMCall) -> Dict[str, Any]:
        t0 = time.time()
        experts = await asyncio.gather(
            self.fund_agent.run(user_content, llm_call),
            self.tech_agent.run(user_content, llm_call),
            self.event_agent.run(user_content, llm_call),
        )
        judge_input = self._build_judge_input(user_content, experts)
        judge = await self.judge_agent.run(judge_input, llm_call)

        merged_items = [x for x in experts if isinstance(x, dict)] + [judge]
        usage = self._merge_usage(merged_items)
        total_cost = float(sum(float((x or {}).get("cost", 0.0) or 0.0) for x in merged_items))
        elapsed = time.time() - t0
        final_text = self._compose_final_markdown(experts, judge)
        meta = self._build_meta(experts, judge)
        return {
            "final_text": final_text,
            "experts": experts,
            "judge": judge,
            "usage": usage,
            "total_cost": total_cost,
            "elapsed": elapsed,
            "meta": meta,
        }
