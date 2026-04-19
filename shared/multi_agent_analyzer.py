from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.agent_memory import AgentMemory


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

    async def run(
        self,
        user_content: str,
        llm_call: LLMCall,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        try:
            system_prompt = (
                str(system_prompt_override)
                if system_prompt_override is not None
                else str(self.config.system_prompt)
            )
            text, usage, cost, elapsed = await asyncio.to_thread(
                llm_call,
                user_content,
                system_prompt,
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
                    "你将收到该股票的基本面数据（ROE、现金流、估值分位等），请基于这些真实数据构建看空论点，而非仅依赖价格数据。\n"
                    "请仅基于输入数据，优先从以下角度构建看空链条：\n"
                    "1) 高估值风险：高PE/高PB/低股息与增长错配；\n"
                    "2) 质量风险：现金流弱、杠杆高、盈利波动；\n"
                    "3) 安全边际不足：估值已透支预期。\n"
                    "输出格式：\n"
                    "- 空头核心论点（3-5条）\n"
                    "- 触发条件（2条）\n"
                    "- 未来一周看空概率（0-100）与置信度说明（50字内）\n"
                    "必须引用具体数据数字（如“ROE仅8.2%，低于行业均值15%”），禁止泛泛而谈。\n"
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
                    "你将收到该股票的最新新闻和研报摘要，请基于这些事件线索判断催化方向。\n"
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
        self.memory = AgentMemory()

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
    def _try_parse_json(text: str) -> Any:
        try:
            return json.loads(str(text or ""))
        except Exception:
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            out = float(value)
            if out != out:  # NaN
                return None
            return out
        except Exception:
            return None

    @classmethod
    def _extract_code_name_from_payload(
        cls,
        user_content: str,
        stock_code: str = "",
        stock_name: str = "",
        fundamental_data: dict | None = None,
    ) -> Tuple[str, str]:
        code = str(stock_code or "").strip()
        name = str(stock_name or "").strip()
        if code and name:
            return code, name

        fd = fundamental_data if isinstance(fundamental_data, dict) else {}
        if not code:
            code = str(fd.get("code") or "").strip()
        if not name:
            name = str(fd.get("name") or "").strip()
        if code and name:
            return code, name

        parsed = cls._try_parse_json(user_content)
        if isinstance(parsed, dict):
            if not code:
                code = str(
                    parsed.get("code")
                    or parsed.get("symbol")
                    or parsed.get("stock_code")
                    or ""
                ).strip()
            if not name:
                name = str(parsed.get("name") or parsed.get("stock_name") or "").strip()
        return code, name

    @classmethod
    def _extract_price_from_payload(
        cls,
        user_content: str,
        price_at_analysis: Any = None,
    ) -> float | None:
        direct = cls._to_float(price_at_analysis)
        if direct is not None and direct > 0:
            return direct
        parsed = cls._try_parse_json(user_content)
        if not isinstance(parsed, dict):
            return None
        for k in ["current_price", "price", "latest_price", "close"]:
            px = cls._to_float(parsed.get(k))
            if px is not None and px > 0:
                return px
        return None

    @classmethod
    def _build_fundamental_input(
        cls,
        raw_payload: str,
        fundamental_data: dict | None = None,
        valuation_data: dict | None = None,
    ) -> str:
        fd = fundamental_data if isinstance(fundamental_data, dict) else {}
        vd = valuation_data if isinstance(valuation_data, dict) else {}
        if not fd and not vd:
            return raw_payload

        parsed_snapshot = cls._try_parse_json(raw_payload)
        market_snapshot = parsed_snapshot if isinstance(parsed_snapshot, dict) else raw_payload

        dimensions = fd.get("dimensions", [])
        if not isinstance(dimensions, list):
            dimensions = []
        eight_dims = []
        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            eight_dims.append(
                {
                    "key": dim.get("key"),
                    "title": dim.get("title"),
                    "score": dim.get("score"),
                    "max_score": dim.get("max_score"),
                    "comment": dim.get("comment"),
                }
            )

        valuation_block = vd if vd else (fd.get("valuation_report") if isinstance(fd.get("valuation_report"), dict) else {})

        obj = {
            "market_snapshot": market_snapshot,
            "fundamental_context": {
                "code": fd.get("code"),
                "name": fd.get("name"),
                "total_score": fd.get("total_score"),
                "conclusion": fd.get("conclusion"),
                "coverage_ratio": fd.get("coverage_ratio"),
                "eight_dimension_scores": eight_dims,
                "profitability_quality": {
                    "roe": fd.get("roe"),
                    "gross_margin": fd.get("gross_margin"),
                    "net_margin": fd.get("net_margin"),
                    "debt_ratio": fd.get("debt_ratio"),
                    "current_ratio": fd.get("current_ratio"),
                    "ocf_sum_3y": fd.get("ocf_sum_3y"),
                    "ocf_per_share": fd.get("ocf_per_share"),
                },
                "profit_trend": {
                    "revenue_growth": fd.get("revenue_growth"),
                    "profit_growth": fd.get("profit_growth"),
                    "revenue_cagr_5y": fd.get("revenue_cagr_5y"),
                    "profit_cagr_5y": fd.get("profit_cagr_5y"),
                    "roe_avg_5y": fd.get("roe_avg_5y"),
                    "gross_margin_avg_5y": fd.get("gross_margin_avg_5y"),
                    "debt_ratio_avg_5y": fd.get("debt_ratio_avg_5y"),
                    "gross_margin_change_5y": fd.get("gross_margin_change_5y"),
                    "debt_ratio_change_5y": fd.get("debt_ratio_change_5y"),
                    "ocf_positive_years_5y": fd.get("ocf_positive_years_5y"),
                },
                "valuation_data": valuation_block,
            },
            "instruction": "请优先基于基本面和估值数字构建空头论证，且必须引用明确数值。",
        }
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _build_event_input(cls, raw_payload: str, news_context: dict | None = None) -> str:
        nc = news_context if isinstance(news_context, dict) else {}
        if not nc:
            return raw_payload
        parsed_snapshot = cls._try_parse_json(raw_payload)
        market_snapshot = parsed_snapshot if isinstance(parsed_snapshot, dict) else raw_payload
        obj = {
            "market_snapshot": market_snapshot,
            "news_context": nc,
            "instruction": "请基于新闻与研报摘要判断一周催化方向及是否已计价。",
        }
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _extract_judge_metrics(
        cls, fundamental_data: dict | None = None, valuation_data: dict | None = None
    ) -> Dict[str, float]:
        fd = fundamental_data if isinstance(fundamental_data, dict) else {}
        vd = valuation_data if isinstance(valuation_data, dict) else {}
        out: Dict[str, float] = {}

        pe_pct = cls._to_float(vd.get("pe_percentile"))
        if pe_pct is None:
            pe_pct = cls._to_float(((vd.get("valuation_percentile") or {}).get("pe_ttm") or {}).get("percentile"))
        if pe_pct is None:
            pe_pct = cls._to_float(((vd.get("pe_ttm") or {}).get("percentile")))
        if pe_pct is not None:
            out["pe_percentile"] = pe_pct

        dcf_margin = cls._to_float(((vd.get("dcf") or {}).get("safety_margin_pct")))
        if dcf_margin is None:
            dcf_margin = cls._to_float(((fd.get("valuation_report") or {}).get("dcf") or {}).get("safety_margin_pct"))
        if dcf_margin is None:
            dcf_margin = cls._to_float(((vd.get("full_valuation") or {}).get("dcf") or {}).get("safety_margin_pct"))
        if dcf_margin is not None:
            out["dcf_safety_margin_pct"] = dcf_margin

        total_score = cls._to_float(fd.get("total_score"))
        if total_score is not None:
            out["eight_dim_total_score"] = total_score
        return out

    @classmethod
    def _build_key_fundamental_summary(cls, fundamental_data: dict | None = None) -> Dict[str, Any]:
        fd = fundamental_data if isinstance(fundamental_data, dict) else {}
        if not fd:
            return {}
        dimensions = fd.get("dimensions", [])
        scored: List[Dict[str, Any]] = []
        if isinstance(dimensions, list):
            for dim in dimensions:
                if not isinstance(dim, dict):
                    continue
                score = cls._to_float(dim.get("score"))
                if score is None:
                    continue
                scored.append(
                    {
                        "title": dim.get("title") or dim.get("key"),
                        "score": score,
                        "max_score": dim.get("max_score"),
                        "comment": dim.get("comment"),
                    }
                )
        weakest = sorted(scored, key=lambda x: x["score"])[:2]
        strongest = sorted(scored, key=lambda x: x["score"], reverse=True)[:2]
        return {
            "total_score": fd.get("total_score"),
            "conclusion": fd.get("conclusion"),
            "coverage_ratio": fd.get("coverage_ratio"),
            "growth_trend": {
                "revenue_growth": fd.get("revenue_growth"),
                "profit_growth": fd.get("profit_growth"),
                "revenue_cagr_5y": fd.get("revenue_cagr_5y"),
                "profit_cagr_5y": fd.get("profit_cagr_5y"),
            },
            "quality_risk": {
                "roe": fd.get("roe"),
                "ocf_sum_3y": fd.get("ocf_sum_3y"),
                "ocf_per_share": fd.get("ocf_per_share"),
                "debt_ratio": fd.get("debt_ratio"),
            },
            "weakest_dimensions": weakest,
            "strongest_dimensions": strongest,
        }

    @classmethod
    def _build_judge_input(
        cls,
        raw_payload: str,
        experts: List[Dict[str, Any]],
        fundamental_data: dict | None = None,
        valuation_data: dict | None = None,
    ) -> str:
        expert_map = {str(x.get("agent", "")): str(x.get("text", "")) for x in experts if isinstance(x, dict)}
        if not isinstance(fundamental_data, dict) and not isinstance(valuation_data, dict):
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

        judge_metrics = cls._extract_judge_metrics(fundamental_data=fundamental_data, valuation_data=valuation_data)
        key_fund_summary = cls._build_key_fundamental_summary(fundamental_data=fundamental_data)
        obj = {
            "input_snapshot": raw_payload,
            "experts": {
                "fundamental_bear": expert_map.get("FundamentalAgent", ""),
                "technical_bull": expert_map.get("TechnicalAgent", ""),
                "event_impact": expert_map.get("EventAgent", ""),
            },
            "key_fundamental_summary": key_fund_summary,
            "judge_reference_metrics": {
                "current_pe_percentile": judge_metrics.get("pe_percentile"),
                "dcf_safety_margin_pct": judge_metrics.get("dcf_safety_margin_pct"),
                "eight_dim_total_score": judge_metrics.get("eight_dim_total_score"),
            },
            "instruction": "请作为法官做出二选一裁决并给出做多/做空胜率，并参考关键基本面指标。",
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

    async def analyze(
        self,
        user_content: str,
        llm_call: LLMCall,
        fundamental_data: dict | None = None,
        valuation_data: dict | None = None,
        news_context: dict | None = None,
        stock_code: str = "",
        stock_name: str = "",
        price_at_analysis: Any = None,
        analysis_date: Any = None,
    ) -> Dict[str, Any]:
        t0 = time.time()
        enhanced_mode = any(
            isinstance(x, dict) and bool(x) for x in [fundamental_data, valuation_data, news_context]
        )
        resolved_code, resolved_name = self._extract_code_name_from_payload(
            user_content=user_content,
            stock_code=stock_code,
            stock_name=stock_name,
            fundamental_data=fundamental_data if enhanced_mode else None,
        )
        resolved_price = self._extract_price_from_payload(
            user_content=user_content,
            price_at_analysis=price_at_analysis,
        )
        self_correction_prompt = (
            self.memory.generate_self_correction_prompt(resolved_code) if str(resolved_code).strip() else ""
        )
        fund_input = (
            self._build_fundamental_input(user_content, fundamental_data=fundamental_data, valuation_data=valuation_data)
            if enhanced_mode
            else user_content
        )
        tech_input = user_content
        event_input = self._build_event_input(user_content, news_context=news_context) if enhanced_mode else user_content

        experts = await asyncio.gather(
            self.fund_agent.run(fund_input, llm_call),
            self.tech_agent.run(tech_input, llm_call),
            self.event_agent.run(event_input, llm_call),
        )
        judge_input = self._build_judge_input(
            user_content,
            experts,
            fundamental_data=fundamental_data if enhanced_mode else None,
            valuation_data=valuation_data if enhanced_mode else None,
        )
        judge_prompt = self.judge_agent.config.system_prompt
        if self_correction_prompt:
            judge_prompt = f"{judge_prompt}\n\n{self_correction_prompt}"
        judge = await self.judge_agent.run(judge_input, llm_call, system_prompt_override=judge_prompt)

        merged_items = [x for x in experts if isinstance(x, dict)] + [judge]
        usage = self._merge_usage(merged_items)
        total_cost = float(sum(float((x or {}).get("cost", 0.0) or 0.0) for x in merged_items))
        elapsed = time.time() - t0
        final_text = self._compose_final_markdown(experts, judge)
        meta = self._build_meta(experts, judge)
        result = {
            "final_text": final_text,
            "experts": experts,
            "judge": judge,
            "usage": usage,
            "total_cost": total_cost,
            "elapsed": elapsed,
            "meta": meta,
            "self_correction_prompt": self_correction_prompt,
        }
        if str(resolved_code).strip() and resolved_price is not None:
            try:
                self.memory.save_prediction(
                    code=resolved_code,
                    name=resolved_name or resolved_code,
                    analysis_result=result,
                    price_at_analysis=resolved_price,
                    date=analysis_date or datetime.now().date(),
                )
            except Exception:
                pass
        return result
