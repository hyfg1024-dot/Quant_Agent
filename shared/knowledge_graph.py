from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import yaml

from shared.db_manager import connect_db, normalize_position_symbol
from shared.valuation_percentile import ValuationPercentile

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUPPLY_CHAIN_PATH = PROJECT_ROOT / "config" / "supply_chain_map.yaml"


class IndustryKnowledgeGraph:
    """A/HK 产业链知识图谱（DuckDB 版）。

    功能
    ----------
    1) build_from_akshare: 构建并落库 company_graph
    2) get_related_companies: 多跳关联查询
    3) propagation_alert: 触发信号后的传导预警（筛选低估值关联公司）
    """

    # 基础映射（可被 YAML 覆盖/扩展）
    SUPPLY_CHAIN_MAP: Dict[str, Dict[str, List[str]]] = {
        "锂矿": {"downstream": ["正极材料", "电池", "新能源汽车"], "aliases": ["锂资源", "盐湖提锂"]},
        "正极材料": {"downstream": ["电池"], "upstream": ["锂矿", "化工原料"]},
        "负极材料": {"downstream": ["电池"], "upstream": ["石墨", "化工原料"]},
        "隔膜": {"downstream": ["电池"], "upstream": ["化工原料"]},
        "电解液": {"downstream": ["电池"], "upstream": ["化工原料"]},
        "电池": {"upstream": ["锂矿", "正极材料", "负极材料", "隔膜", "电解液"], "downstream": ["新能源汽车", "储能"]},
        "新能源汽车": {"upstream": ["电池", "汽车零部件", "芯片设计"], "downstream": ["充电桩", "汽车电子"]},
        "汽车零部件": {"downstream": ["新能源汽车", "传统汽车"]},
        "充电桩": {"upstream": ["新能源汽车", "电网设备"], "related": ["储能"]},
        "光伏硅料": {"downstream": ["光伏电池片", "光伏组件"]},
        "光伏电池片": {"upstream": ["光伏硅料"], "downstream": ["光伏组件"]},
        "光伏组件": {"upstream": ["光伏硅料", "光伏电池片"], "downstream": ["储能", "电站运营"]},
        "储能": {"upstream": ["电池", "光伏组件"], "downstream": ["电网设备"]},
        "风电整机": {"upstream": ["风电零部件"], "downstream": ["电网设备"]},
        "风电零部件": {"downstream": ["风电整机"]},
        "半导体设备": {"downstream": ["晶圆制造", "芯片设计"]},
        "EDA": {"downstream": ["芯片设计"], "aliases": ["电子设计自动化"]},
        "芯片设计": {"upstream": ["半导体设备", "EDA"], "downstream": ["消费电子", "汽车电子", "AI算力"]},
        "晶圆制造": {"upstream": ["半导体设备"], "downstream": ["封装测试", "芯片设计"]},
        "封装测试": {"upstream": ["晶圆制造"], "downstream": ["消费电子", "汽车电子"]},
        "消费电子": {"upstream": ["芯片设计", "封装测试"], "related": ["AI算力"]},
        "汽车电子": {"upstream": ["芯片设计", "封装测试"], "downstream": ["新能源汽车"]},
        "AI算力": {"upstream": ["芯片设计", "服务器"], "downstream": ["数据中心", "软件服务"]},
        "服务器": {"upstream": ["芯片设计"], "downstream": ["AI算力", "数据中心"]},
        "数据中心": {"upstream": ["服务器", "AI算力"], "downstream": ["软件服务"]},
        "软件服务": {"upstream": ["AI算力", "数据中心"], "related": ["工业互联网"]},
        "工业互联网": {"upstream": ["软件服务"], "downstream": ["机器人"]},
        "机器人": {"upstream": ["工业互联网", "芯片设计"]},
        "房地产": {"downstream": ["建材", "家电", "装修"], "related": ["银行"]},
        "建材": {"upstream": ["房地产"]},
        "家电": {"upstream": ["房地产", "芯片设计"]},
        "装修": {"upstream": ["房地产", "建材"]},
        "白酒": {"upstream": ["粮食", "包装"], "related": ["消费"]},
        "粮食": {"downstream": ["白酒", "食品饮料"]},
        "包装": {"downstream": ["白酒", "食品饮料"]},
        "银行": {"related": ["保险", "券商", "房地产"]},
        "保险": {"related": ["银行", "券商"]},
        "券商": {"related": ["银行", "保险"]},
    }

    HOLDER_BLACKLIST_KEYWORDS = {
        "香港中央结算",
        "中国证券金融",
        "中央汇金",
        "全国社保",
        "中国工商银行",
        "中国银行",
        "中国建设银行",
        "招商银行",
    }

    RELATION_WEIGHTS = {
        "same_industry": 0.58,
        "supply_chain": 0.82,
        "equity_link": 0.72,
    }

    def __init__(
        self,
        supply_chain_path: str | Path | None = None,
        request_pause_sec: float = 0.06,
        max_components_per_industry: int = 80,
        max_holder_companies: int = 260,
    ) -> None:
        self.supply_chain_path = Path(supply_chain_path) if supply_chain_path else DEFAULT_SUPPLY_CHAIN_PATH
        self.request_pause_sec = max(0.0, float(request_pause_sec))
        self.max_components_per_industry = max(10, int(max_components_per_industry))
        self.max_holder_companies = max(20, int(max_holder_companies))
        self._valuation = ValuationPercentile()
        self.supply_chain_map = self._load_supply_chain_map()
        self._ensure_tables()

    @staticmethod
    def _safe_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _find_col(columns: Iterable[Any], keywords: List[str]) -> str:
        cols = [str(c) for c in columns]
        for kw in keywords:
            for col in cols:
                if kw.lower() in col.lower():
                    return col
        return ""

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
            if pd.isna(out):
                return None
            return out
        except Exception:
            return None

    @staticmethod
    def _dedup_keep_order(items: Iterable[Any]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for x in items:
            t = str(x or "").strip()
            if (not t) or t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    @staticmethod
    def _infer_market_by_code(code: str) -> str:
        digits = "".join(ch for ch in str(code or "") if ch.isdigit())
        return "HK" if len(digits) <= 5 else "A"

    def _normalize_code(self, code: Any, market_hint: Optional[str] = None) -> str:
        raw = self._safe_text(code).upper()
        if not raw:
            return ""
        try:
            mk, db_code, _ = normalize_position_symbol(raw, market=market_hint)
            if mk == "HK":
                return str(db_code).zfill(5)
            return str(db_code).zfill(6)
        except Exception:
            digits = "".join(ch for ch in raw if ch.isdigit())
            if not digits:
                return ""
            if market_hint == "HK" or len(digits) <= 5:
                return digits.zfill(5)
            return digits.zfill(6)

    def _load_supply_chain_map(self) -> Dict[str, Dict[str, List[str]]]:
        merged = deepcopy(self.SUPPLY_CHAIN_MAP)
        if not self.supply_chain_path.exists():
            return self._normalize_supply_chain_map(merged)

        try:
            ext = yaml.safe_load(self.supply_chain_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("读取 supply_chain_map 失败，回退内置映射: %s", exc)
            return self._normalize_supply_chain_map(merged)

        if not isinstance(ext, dict):
            return self._normalize_supply_chain_map(merged)

        for industry, conf in ext.items():
            key = self._safe_text(industry)
            if not key:
                continue
            base = merged.get(key, {}) if isinstance(merged.get(key), dict) else {}
            merged[key] = self._merge_supply_chain_entry(base, conf)
        return self._normalize_supply_chain_map(merged)

    def _merge_supply_chain_entry(self, base: Dict[str, Any], ext: Any) -> Dict[str, List[str]]:
        out = {
            "upstream": self._dedup_keep_order(base.get("upstream", [])),
            "downstream": self._dedup_keep_order(base.get("downstream", [])),
            "related": self._dedup_keep_order(base.get("related", [])),
            "aliases": self._dedup_keep_order(base.get("aliases", [])),
        }
        if isinstance(ext, dict):
            for k in ["upstream", "downstream", "related", "aliases"]:
                out[k] = self._dedup_keep_order([*out[k], *list(ext.get(k, []) or [])])
        elif isinstance(ext, list):
            out["related"] = self._dedup_keep_order([*out["related"], *list(ext)])
        return out

    def _normalize_supply_chain_map(self, raw_map: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
        out: Dict[str, Dict[str, List[str]]] = {}
        for key, conf in raw_map.items():
            node = self._safe_text(key)
            if not node:
                continue
            if isinstance(conf, dict):
                out[node] = {
                    "upstream": self._dedup_keep_order(conf.get("upstream", [])),
                    "downstream": self._dedup_keep_order(conf.get("downstream", [])),
                    "related": self._dedup_keep_order(conf.get("related", [])),
                    "aliases": self._dedup_keep_order(conf.get("aliases", [])),
                }
            elif isinstance(conf, list):
                out[node] = {
                    "upstream": [],
                    "downstream": [],
                    "related": self._dedup_keep_order(conf),
                    "aliases": [],
                }
            else:
                out[node] = {"upstream": [], "downstream": [], "related": [], "aliases": []}
        return out

    def _ensure_tables(self) -> None:
        with connect_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS company_graph (
                    source_code TEXT,
                    target_code TEXT,
                    relation_type TEXT,
                    weight DOUBLE,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (source_code, target_code, relation_type)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_company_graph_source ON company_graph(source_code)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_company_graph_target ON company_graph(target_code)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS company_graph_profile (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT,
                    industries TEXT,
                    updated_at TIMESTAMP
                )
                """
            )

    def _ak(self):
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"akshare 不可用: {exc}") from exc
        return ak

    def _match_chain_nodes(self, industry_name: str) -> List[str]:
        text = self._safe_text(industry_name)
        if not text:
            return []
        out: List[str] = []
        for node, conf in self.supply_chain_map.items():
            tokens = [node, *(conf.get("aliases", []) or [])]
            if any(tok and tok in text for tok in tokens):
                out.append(node)
        return out

    @staticmethod
    def _add_edge(edge_map: Dict[Tuple[str, str, str], float], source: str, target: str, rel: str, weight: float) -> None:
        if (not source) or (not target) or source == target:
            return
        key = (source, target, rel)
        w = max(0.01, min(1.0, float(weight)))
        old = edge_map.get(key)
        if old is None:
            edge_map[key] = w
            return
        edge_map[key] = min(1.0, max(float(old), w) + 0.03)

    def _fetch_industry_components(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], Dict[str, Set[str]]]:
        ak = self._ak()
        board_df = ak.stock_board_industry_name_em()
        if not isinstance(board_df, pd.DataFrame) or board_df.empty:
            raise RuntimeError("stock_board_industry_name_em 未返回有效数据")

        board_col = self._find_col(board_df.columns, ["板块名称", "板块", "名称", "行业"]) or str(board_df.columns[0])

        industry_members: Dict[str, List[str]] = {}
        profiles: Dict[str, Dict[str, Any]] = {}
        code_industries: Dict[str, Set[str]] = defaultdict(set)

        board_names = [self._safe_text(x) for x in board_df[board_col].tolist()]
        board_names = [x for x in board_names if x]
        total_boards = len(board_names)
        logger.info("[KG] 行业板块数量: %d", total_boards)

        for idx, board in enumerate(board_names, 1):
            if idx == 1 or idx % 10 == 0 or idx == total_boards:
                logger.info("[KG] 行业成分抓取进度: %d/%d (%s)", idx, total_boards, board)
            try:
                comp_df = ak.stock_board_industry_cons_em(symbol=board)
            except Exception as exc:
                logger.debug("[KG] 行业成分抓取失败 %s: %s", board, exc)
                continue
            if not isinstance(comp_df, pd.DataFrame) or comp_df.empty:
                continue

            code_col = self._find_col(comp_df.columns, ["代码", "证券代码", "symbol", "code"])
            name_col = self._find_col(comp_df.columns, ["名称", "证券简称", "name"])
            mv_col = self._find_col(comp_df.columns, ["总市值", "市值"])
            if not code_col:
                continue

            work_df = comp_df.copy()
            if mv_col:
                work_df["__mv"] = pd.to_numeric(work_df[mv_col], errors="coerce")
                work_df = work_df.sort_values("__mv", ascending=False)
            work_df = work_df.head(self.max_components_per_industry)

            codes: List[str] = []
            for _, row in work_df.iterrows():
                code = self._normalize_code(row.get(code_col), market_hint="A")
                if not code:
                    continue
                name = self._safe_text(row.get(name_col)) if name_col else code
                codes.append(code)
                code_industries[code].add(board)

                p = profiles.get(code, {"code": code, "name": name, "market": "A", "industries": set()})
                if name:
                    p["name"] = name
                p["market"] = "A"
                inds = p.get("industries", set())
                if not isinstance(inds, set):
                    inds = set(inds)
                inds.add(board)
                p["industries"] = inds
                profiles[code] = p

            if codes:
                industry_members[board] = sorted(set(codes))

            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)

        return industry_members, profiles, code_industries

    def _fetch_top_holders(self, code: str) -> List[str]:
        ak = self._ak()
        calls = [
            lambda: ak.stock_main_stock_holder_em(symbol=code),
            lambda: ak.stock_main_stock_holder_em(code),
        ]
        holder_df = pd.DataFrame()
        for fn in calls:
            try:
                out = fn()
                if isinstance(out, pd.DataFrame) and (not out.empty):
                    holder_df = out
                    break
            except Exception:
                continue
        if holder_df.empty:
            return []

        holder_col = self._find_col(holder_df.columns, ["股东名称", "股东", "名称", "holder"])
        rank_col = self._find_col(holder_df.columns, ["序号", "排名", "rank"])
        date_col = self._find_col(holder_df.columns, ["公告日期", "截止日期", "报告期", "日期", "date"])

        if not holder_col:
            return []

        df = holder_df.copy()
        if date_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            if dt.notna().any():
                latest_dt = dt.max()
                df = df[dt == latest_dt]

        if rank_col:
            rank_num = pd.to_numeric(df[rank_col], errors="coerce")
            if rank_num.notna().any():
                df = df.assign(__rank=rank_num).sort_values("__rank", ascending=True)
        df = df.head(10)

        holders = [self._safe_text(x) for x in df[holder_col].tolist()]
        holders = [x for x in holders if x]
        return self._dedup_keep_order(holders)

    def _is_informative_holder(self, holder: str) -> bool:
        text = self._safe_text(holder)
        if len(text) < 4:
            return False
        return not any(k in text for k in self.HOLDER_BLACKLIST_KEYWORDS)

    def _build_holder_links(self, codes: List[str]) -> Dict[Tuple[str, str], int]:
        holder_to_codes: Dict[str, Set[str]] = defaultdict(set)
        n = min(len(codes), self.max_holder_companies)
        scan_codes = codes[:n]
        logger.info("[KG] 股东关联抓取样本: %d/%d", len(scan_codes), len(codes))

        for idx, code in enumerate(scan_codes, 1):
            if idx == 1 or idx % 25 == 0 or idx == len(scan_codes):
                logger.info("[KG] 股东抓取进度: %d/%d (%s)", idx, len(scan_codes), code)
            try:
                holders = self._fetch_top_holders(code)
            except Exception as exc:
                logger.debug("[KG] 股东抓取失败 %s: %s", code, exc)
                holders = []
            for h in holders:
                if self._is_informative_holder(h):
                    holder_to_codes[h].add(code)
            if self.request_pause_sec > 0:
                time.sleep(self.request_pause_sec)

        pair_counter: Dict[Tuple[str, str], int] = defaultdict(int)
        for _, comp_set in holder_to_codes.items():
            members = sorted(comp_set)
            if len(members) < 2:
                continue
            # 避免单一超大股东产生 O(n^2) 过度连边
            if len(members) > 36:
                members = members[:36]
            for a, b in itertools.combinations(members, 2):
                pair_counter[(a, b)] += 1
        return pair_counter

    def _write_profiles(self, profiles: Dict[str, Dict[str, Any]]) -> None:
        rows: List[List[Any]] = []
        now = datetime.now()
        for code, profile in profiles.items():
            industries = profile.get("industries", set())
            if isinstance(industries, set):
                inds_text = "|".join(sorted(industries))
            elif isinstance(industries, list):
                inds_text = "|".join(self._dedup_keep_order(industries))
            else:
                inds_text = self._safe_text(industries)
            rows.append([code, self._safe_text(profile.get("name")), self._safe_text(profile.get("market")), inds_text, now])

        with connect_db() as conn:
            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO company_graph_profile(code, name, market, industries, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def build_from_akshare(self) -> Dict[str, Any]:
        """从 AKShare 构建图谱并落地 DuckDB。"""
        started = time.time()
        self._ensure_tables()
        self.supply_chain_map = self._load_supply_chain_map()

        logger.info("[KG] 开始构建行业知识图谱")
        industry_members, profiles, code_industries = self._fetch_industry_components()
        logger.info("[KG] 行业抓取完成: %d 个行业, %d 只股票", len(industry_members), len(profiles))

        edge_map: Dict[Tuple[str, str, str], float] = {}

        # 1) 同行业关系
        for _, members in industry_members.items():
            codes = sorted(set(members))
            if len(codes) < 2:
                continue
            for a, b in itertools.combinations(codes, 2):
                self._add_edge(edge_map, a, b, "same_industry", self.RELATION_WEIGHTS["same_industry"])
                self._add_edge(edge_map, b, a, "same_industry", self.RELATION_WEIGHTS["same_industry"])

        # 2) 供应链关系
        chain_to_codes: Dict[str, Set[str]] = defaultdict(set)
        for code, inds in code_industries.items():
            for ind in inds:
                for chain_node in self._match_chain_nodes(ind):
                    chain_to_codes[chain_node].add(code)

        for node, conf in self.supply_chain_map.items():
            src_codes = sorted(chain_to_codes.get(node, set()))
            if not src_codes:
                continue

            for down in conf.get("downstream", []):
                tgt_codes = sorted(chain_to_codes.get(down, set()))
                for s in src_codes:
                    for t in tgt_codes:
                        self._add_edge(edge_map, s, t, "supply_chain", self.RELATION_WEIGHTS["supply_chain"])

            for up in conf.get("upstream", []):
                up_codes = sorted(chain_to_codes.get(up, set()))
                for u in up_codes:
                    for t in src_codes:
                        self._add_edge(edge_map, u, t, "supply_chain", self.RELATION_WEIGHTS["supply_chain"])

            for rel in conf.get("related", []):
                rel_codes = sorted(chain_to_codes.get(rel, set()))
                for s in src_codes:
                    for t in rel_codes:
                        self._add_edge(edge_map, s, t, "supply_chain", self.RELATION_WEIGHTS["supply_chain"] * 0.88)
                        self._add_edge(edge_map, t, s, "supply_chain", self.RELATION_WEIGHTS["supply_chain"] * 0.88)

        # 3) 股权关联关系（共同大股东）
        all_codes = sorted(profiles.keys())
        pair_counter = self._build_holder_links(all_codes)
        for (a, b), common_cnt in pair_counter.items():
            w = min(0.96, self.RELATION_WEIGHTS["equity_link"] + 0.05 * max(0, common_cnt - 1))
            self._add_edge(edge_map, a, b, "equity_link", w)
            self._add_edge(edge_map, b, a, "equity_link", w)

        now = datetime.now()
        edge_rows = [[s, t, rel, float(w), now] for (s, t, rel), w in edge_map.items()]

        with connect_db() as conn:
            conn.execute("DELETE FROM company_graph")
            if edge_rows:
                conn.executemany(
                    """
                    INSERT INTO company_graph(source_code, target_code, relation_type, weight, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    edge_rows,
                )

        self._write_profiles(profiles)
        elapsed = time.time() - started
        logger.info("[KG] 图谱构建完成: edges=%d, nodes=%d, 用时=%.2fs", len(edge_rows), len(profiles), elapsed)
        return {
            "ok": True,
            "nodes": len(profiles),
            "edges": len(edge_rows),
            "industries": len(industry_members),
            "holder_pairs": len(pair_counter),
            "elapsed": round(elapsed, 3),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _lookup_names(self, codes: List[str]) -> Dict[str, str]:
        if not codes:
            return {}
        code_list = sorted(set([self._safe_text(c) for c in codes if self._safe_text(c)]))
        if not code_list:
            return {}
        placeholders = ",".join(["?"] * len(code_list))

        out: Dict[str, str] = {}
        with connect_db(read_only=True) as conn:
            try:
                df_profile = conn.execute(
                    f"SELECT code, name FROM company_graph_profile WHERE code IN ({placeholders})",
                    code_list,
                ).df()
                if not df_profile.empty:
                    for _, row in df_profile.iterrows():
                        code = self._safe_text(row.get("code"))
                        name = self._safe_text(row.get("name"))
                        if code and name:
                            out[code] = name
            except Exception:
                pass

            # 回退到 stock_basic
            missing = [c for c in code_list if c not in out]
            if missing:
                p2 = ",".join(["?"] * len(missing))
                try:
                    df_basic = conn.execute(
                        f"SELECT code, name FROM stock_basic WHERE code IN ({p2})",
                        missing,
                    ).df()
                    if not df_basic.empty:
                        for _, row in df_basic.iterrows():
                            code = self._safe_text(row.get("code"))
                            name = self._safe_text(row.get("name"))
                            if code and name and code not in out:
                                out[code] = name
                except Exception:
                    pass
        return out

    def get_related_companies(self, code: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """多跳查询关联公司（按强度排序）。"""
        norm = self._normalize_code(code, market_hint=None)
        if not norm:
            return []

        depth = max(1, min(int(max_depth), 4))
        sql = """
        WITH RECURSIVE walk AS (
            SELECT
                source_code,
                target_code,
                relation_type,
                weight,
                1 AS depth,
                CAST(weight AS DOUBLE) AS strength,
                ('|' || source_code || '|' || target_code || '|') AS path
            FROM company_graph
            WHERE source_code = ?

            UNION ALL

            SELECT
                w.source_code,
                g.target_code,
                g.relation_type,
                g.weight,
                w.depth + 1 AS depth,
                CAST(w.strength * g.weight * 0.90 AS DOUBLE) AS strength,
                w.path || g.target_code || '|'
            FROM walk w
            JOIN company_graph g
              ON g.source_code = w.target_code
            WHERE w.depth < ?
              AND INSTR(w.path, '|' || g.target_code || '|') = 0
        )
        SELECT
            target_code AS code,
            MIN(depth) AS depth,
            MAX(strength) AS strength,
            STRING_AGG(DISTINCT relation_type, '|') AS relation
        FROM walk
        WHERE target_code <> ?
        GROUP BY target_code
        ORDER BY strength DESC, depth ASC
        LIMIT 300
        """

        with connect_db(read_only=True) as conn:
            try:
                df = conn.execute(sql, [norm, depth, norm]).df()
            except Exception:
                return []

        if df.empty:
            return []

        codes = [self._safe_text(x) for x in df["code"].tolist()]
        name_map = self._lookup_names(codes)
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            c = self._safe_text(row.get("code"))
            if not c:
                continue
            out.append(
                {
                    "code": c,
                    "name": name_map.get(c, c),
                    "relation": self._safe_text(row.get("relation")) or "unknown",
                    "depth": int(row.get("depth") or 1),
                    "strength": round(float(row.get("strength") or 0.0), 6),
                }
            )
        return out

    def propagation_alert(
        self,
        trigger_code: str,
        trigger_signal: str,
        max_depth: int = 2,
        pe_percentile_threshold: float = 35.0,
    ) -> List[Dict[str, Any]]:
        """触发信号后的传导预警：筛选估值偏低关联公司。"""
        norm = self._normalize_code(trigger_code, market_hint=None)
        if not norm:
            return []

        related = self.get_related_companies(norm, max_depth=max_depth)
        if not related:
            return []

        alerts: List[Dict[str, Any]] = []
        for item in related:
            code = self._safe_text(item.get("code"))
            if not code:
                continue
            market = "HK" if len(code) <= 5 else "A"
            try:
                rep = self._valuation.compute_multi_percentile(market=market, code=code, lookback_years=5)
            except Exception:
                continue
            pe_pct = self._to_float(((rep.get("pe_ttm") or {}).get("percentile")))
            assess = self._safe_text(((rep.get("pe_ttm") or {}).get("assessment")))
            if pe_pct is None:
                continue
            if pe_pct > float(pe_percentile_threshold):
                continue

            rel_text = self._safe_text(item.get("relation")).replace("|", "+")
            msg = (
                f"{item.get('name', code)}({code})"
                f"({rel_text}, PE分位{pe_pct:.1f}%, {assess or '值得关注'})"
            )
            alerts.append(
                {
                    "trigger_code": norm,
                    "trigger_signal": self._safe_text(trigger_signal),
                    "code": code,
                    "name": item.get("name", code),
                    "relation": item.get("relation", ""),
                    "depth": int(item.get("depth") or 1),
                    "strength": float(item.get("strength") or 0.0),
                    "pe_percentile": round(float(pe_pct), 3),
                    "pe_assessment": assess,
                    "message": msg,
                }
            )

        alerts.sort(key=lambda x: (x["pe_percentile"], -x["strength"], x["depth"]))
        return alerts

    def format_propagation_tip(self, trigger_code: str, trigger_signal: str, top_n: int = 5) -> str:
        """将传导预警转成日报段落文本。"""
        alerts = self.propagation_alert(trigger_code=trigger_code, trigger_signal=trigger_signal, max_depth=2)
        if not alerts:
            return f"{trigger_code}触发{trigger_signal}，当前未发现估值偏低的强关联公司。"

        lines = [f"{trigger_code}触发{trigger_signal} -> 关联公司传导提示："]
        for one in alerts[: max(1, int(top_n))]:
            rel = self._safe_text(one.get("relation")).replace("|", "+")
            lines.append(
                f"- {one.get('name')}({one.get('code')})"
                f"(关联: {rel}, PE分位 {float(one.get('pe_percentile') or 0.0):.1f}%, {one.get('pe_assessment') or '值得关注'})"
            )
        return "\n".join(lines)

    def attach_propagation_to_hits(
        self,
        hits: List[Dict[str, Any]],
        signal_field: str = "summary",
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """集成到拐点日报：为每个命中股票追加产业链传导提示。"""
        out: List[Dict[str, Any]] = []
        for item in hits or []:
            one = dict(item or {})
            code = self._safe_text(one.get("code"))
            if not code:
                out.append(one)
                continue
            trigger_signal = self._safe_text(one.get(signal_field)) or "拐点信号"
            alerts = self.propagation_alert(trigger_code=code, trigger_signal=trigger_signal, max_depth=2)
            one["propagation_alerts"] = alerts[: max(1, int(top_n))]
            one["propagation_tip"] = self.format_propagation_tip(code, trigger_signal, top_n=top_n)
            out.append(one)
        return out


__all__ = ["IndustryKnowledgeGraph"]
