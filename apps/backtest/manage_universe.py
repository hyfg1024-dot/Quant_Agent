"""CLI for universe CRUD and data validation."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.config_loader import ConfigError
from src.data_manager import DataManager
from src.universe_manager import UniverseManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="管理标的池 universe.yaml")
    parser.add_argument("--universe", default="config/universe.yaml", help="universe 文件路径")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出板块和标的")
    p_list.add_argument("--sector", default="", help="仅显示某个 sector")
    p_list.add_argument("--group", default="", help="仅显示某个 group")

    p_add_sector = sub.add_parser("add-sector", help="添加板块")
    p_add_sector.add_argument("sector_key")
    p_add_sector.add_argument("--name", required=True)
    p_add_sector.add_argument("--description", default="")
    p_add_sector.add_argument("--benchmark", default="")

    p_add_group = sub.add_parser("add-group", help="添加分组")
    p_add_group.add_argument("sector_key")
    p_add_group.add_argument("group_key")
    p_add_group.add_argument("--name", required=True)

    p_add_stock = sub.add_parser("add-stock", help="添加标的")
    p_add_stock.add_argument("sector_key")
    p_add_stock.add_argument("group_key")
    p_add_stock.add_argument("code")
    p_add_stock.add_argument("--name", required=True)
    p_add_stock.add_argument("--tags", default="")

    p_import = sub.add_parser("import-stocks", help="从 CSV 批量导入")
    p_import.add_argument("--file", required=True)
    p_import.add_argument("--sector", required=True)
    p_import.add_argument("--group", required=True)

    p_remove_stock = sub.add_parser("remove-stock", help="删除标的")
    p_remove_stock.add_argument("code")

    p_remove_sector = sub.add_parser("remove-sector", help="删除板块")
    p_remove_sector.add_argument("sector_key")

    p_validate = sub.add_parser("validate", help="校验标的数据可用性")
    p_validate.add_argument("--sector", default="")
    p_validate.add_argument("--start", default="2021-01-01")
    p_validate.add_argument("--end", default="today")

    return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def resolve_end_date(text: str) -> str:
    t = str(text).strip().lower()
    return str(date.today()) if t == "today" else str(pd.Timestamp(t).date())


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        universe_path = resolve_path(base_dir, args.universe)
        mgr = UniverseManager(universe_path)

        if args.cmd == "list":
            lines = mgr.list_as_lines(sector=args.sector, group=args.group)
            print("\n".join(lines))
            return 0

        if args.cmd == "add-sector":
            mgr.add_sector(
                key=args.sector_key,
                name=args.name,
                description=args.description,
                benchmark=args.benchmark,
            )
            mgr.persist()
            print(f"[OK] 已添加 sector: {args.sector_key}")
            return 0

        if args.cmd == "add-group":
            mgr.add_group(sector_key=args.sector_key, group_key=args.group_key, name=args.name)
            mgr.persist()
            print(f"[OK] 已添加 group: {args.sector_key}/{args.group_key}")
            return 0

        if args.cmd == "add-stock":
            tags = [x.strip() for x in args.tags.split(",") if x.strip()]
            mgr.add_stock(
                sector_key=args.sector_key,
                group_key=args.group_key,
                code=args.code,
                name=args.name,
                tags=tags,
            )
            mgr.persist()
            print(f"[OK] 已添加 stock: {args.code.upper()}")
            return 0

        if args.cmd == "import-stocks":
            csv_path = resolve_path(base_dir, args.file)
            n = mgr.import_stocks(csv_path=csv_path, sector_key=args.sector, group_key=args.group)
            mgr.persist()
            print(f"[OK] 导入完成: {n} 支")
            return 0

        if args.cmd == "remove-stock":
            ok = mgr.remove_stock(args.code)
            if ok:
                mgr.persist()
                print(f"[OK] 已删除 stock: {args.code.upper()}")
            else:
                print(f"[WARN] 未找到 stock: {args.code.upper()}")
            return 0

        if args.cmd == "remove-sector":
            ok = mgr.remove_sector(args.sector_key)
            if ok:
                mgr.persist()
                print(f"[OK] 已删除 sector: {args.sector_key}")
            else:
                print(f"[WARN] 未找到 sector: {args.sector_key}")
            return 0

        if args.cmd == "validate":
            start = str(pd.Timestamp(args.start).date())
            end = resolve_end_date(args.end)
            dm = DataManager(data_dir=base_dir / "data", logger=print)
            rows = mgr.validate_codes(
                validator=lambda codes, start, end: dm.validate_universe(codes, start=start, end=end),
                start=start,
                end=end,
                sector=args.sector or None,
            )

            ok = 0
            for r in rows:
                print(f"{r.code:<10} | {r.status:<6} | rows={r.rows:<5} | {r.start} -> {r.end} | {r.message}")
                if r.status == "ok":
                    ok += 1
            print(f"[CHECK] 完成: ok={ok}, total={len(rows)}")
            return 0 if ok > 0 else 2

        print("[ERROR] 未知命令")
        return 1

    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        return 2
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
