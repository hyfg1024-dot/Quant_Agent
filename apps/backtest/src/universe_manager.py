"""Universe CRUD helpers used by CLI."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config_loader import (
    ConfigError,
    UniverseConfig,
    UniverseGroup,
    UniverseSector,
    UniverseStock,
    load_universe,
    save_universe,
)


@dataclass
class ValidationResult:
    """Data availability validation output for one code."""

    code: str
    status: str
    rows: int
    start: str
    end: str
    message: str


class UniverseManager:
    """Manage universe yaml with stable APIs for CLI."""

    def __init__(self, universe_path: Path) -> None:
        self.universe_path = universe_path
        self.universe: UniverseConfig = load_universe(universe_path)

    def reload(self) -> None:
        """Reload universe from disk."""
        self.universe = load_universe(self.universe_path)

    def persist(self) -> None:
        """Persist in-memory universe to disk."""
        save_universe(self.universe, self.universe_path)

    def list_as_lines(self, sector: Optional[str] = None, group: Optional[str] = None) -> List[str]:
        """Render universe as readable lines for terminal output."""
        lines: List[str] = []
        sec_filter = (sector or "").strip()
        grp_filter = (group or "").strip()

        lines.append("Benchmarks:")
        for b in self.universe.benchmarks:
            lines.append(f"  - {b.get('code', '')}: {b.get('name', '')}")
        lines.append("")

        for sec in self.universe.sectors:
            if sec_filter and sec.key != sec_filter:
                continue
            lines.append(f"[{sec.key}] {sec.name} | benchmark={sec.sector_benchmark or '-'}")
            lines.append(f"  {sec.description}")
            for grp in sec.groups:
                if grp_filter and grp.key != grp_filter:
                    continue
                lines.append(f"  ({grp.key}) {grp.name} | {len(grp.stocks)} stocks")
                for st in grp.stocks:
                    tags = ",".join(st.tags) if st.tags else "-"
                    lines.append(f"    - {st.code:<8} {st.name} [{tags}]")
            lines.append("")
        return lines

    def add_sector(self, key: str, name: str, description: str, benchmark: str) -> None:
        """Add sector."""
        k = key.strip()
        if not k:
            raise ConfigError("sector key 不能为空")
        if any(s.key == k for s in self.universe.sectors):
            raise ConfigError(f"sector 已存在: {k}")
        self.universe.sectors.append(
            UniverseSector(
                key=k,
                name=name.strip() or k,
                description=description.strip(),
                sector_benchmark=benchmark.strip() or None,
                groups=[],
            )
        )

    def add_group(self, sector_key: str, group_key: str, name: str) -> None:
        """Add group under sector."""
        sec = self._find_sector(sector_key)
        gk = group_key.strip()
        if not gk:
            raise ConfigError("group key 不能为空")
        if any(g.key == gk for g in sec.groups):
            raise ConfigError(f"group 已存在: {gk}")
        sec.groups.append(UniverseGroup(key=gk, name=name.strip() or gk, stocks=[]))

    def add_stock(self, sector_key: str, group_key: str, code: str, name: str, tags: Optional[List[str]] = None) -> None:
        """Add one stock into a sector/group."""
        c = code.strip().upper()
        if not c:
            raise ConfigError("stock code 不能为空")
        if not name.strip():
            raise ConfigError("stock name 不能为空")
        if self._find_stock(c) is not None:
            raise ConfigError(f"股票代码已存在: {c}")

        grp = self._find_group(sector_key, group_key)
        grp.stocks.append(UniverseStock(code=c, name=name.strip(), tags=list(tags or [])))

    def import_stocks(self, csv_path: Path, sector_key: str, group_key: str) -> int:
        """Import stocks from csv with columns: code,name,tags(optional)."""
        if not csv_path.exists():
            raise ConfigError(f"CSV 不存在: {csv_path}")
        grp = self._find_group(sector_key, group_key)
        imported = 0

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"code", "name"}
            if not reader.fieldnames or not expected.issubset(set(reader.fieldnames)):
                raise ConfigError("CSV 必须至少包含列: code,name")
            for row in reader:
                code = str(row.get("code", "")).strip().upper()
                name = str(row.get("name", "")).strip()
                tags_text = str(row.get("tags", "")).strip()
                if not code or not name:
                    continue
                if self._find_stock(code) is not None:
                    continue
                tags = [x.strip() for x in tags_text.split(",") if x.strip()]
                grp.stocks.append(UniverseStock(code=code, name=name, tags=tags))
                imported += 1
        return imported

    def remove_stock(self, code: str) -> bool:
        """Remove stock by code globally."""
        c = code.strip().upper()
        for sec in self.universe.sectors:
            for grp in sec.groups:
                before = len(grp.stocks)
                grp.stocks = [s for s in grp.stocks if s.code != c]
                if len(grp.stocks) < before:
                    return True
        return False

    def remove_sector(self, sector_key: str) -> bool:
        """Remove whole sector."""
        before = len(self.universe.sectors)
        self.universe.sectors = [s for s in self.universe.sectors if s.key != sector_key]
        return len(self.universe.sectors) < before

    def validate_codes(
        self,
        validator,
        start: str,
        end: str,
        sector: Optional[str] = None,
    ) -> List[ValidationResult]:
        """Validate symbol data availability via external validator callback."""
        codes: List[str] = []
        for sec in self.universe.sectors:
            if sector and sec.key != sector:
                continue
            for grp in sec.groups:
                for st in grp.stocks:
                    codes.append(st.code)

        report = validator(codes, start=start, end=end)
        out: List[ValidationResult] = []
        for code in codes:
            item = report.get(code, {}) if isinstance(report, dict) else {}
            out.append(
                ValidationResult(
                    code=code,
                    status=str(item.get("status", "unknown")),
                    rows=int(item.get("rows", 0) or 0),
                    start=str(item.get("start", "")),
                    end=str(item.get("end", "")),
                    message=str(item.get("message", "")),
                )
            )
        return out

    def _find_sector(self, sector_key: str) -> UniverseSector:
        for sec in self.universe.sectors:
            if sec.key == sector_key:
                return sec
        raise ConfigError(f"sector 不存在: {sector_key}")

    def _find_group(self, sector_key: str, group_key: str) -> UniverseGroup:
        sec = self._find_sector(sector_key)
        for grp in sec.groups:
            if grp.key == group_key:
                return grp
        raise ConfigError(f"group 不存在: {sector_key}/{group_key}")

    def _find_stock(self, code: str) -> Optional[UniverseStock]:
        c = code.strip().upper()
        for sec in self.universe.sectors:
            for grp in sec.groups:
                for st in grp.stocks:
                    if st.code == c:
                        return st
        return None
