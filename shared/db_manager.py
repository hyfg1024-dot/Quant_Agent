from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "quant_system.duckdb"

OUTPUT_COLUMNS = [
    "market",
    "code",
    "name",
    "industry",
    "pe_ttm",
    "pb",
    "dividend_yield",
    "roe",
    "asset_liability_ratio",
    "turnover_ratio",
    "volume_ratio",
    "total_mv",
    "revenue_cagr_5y",
    "profit_cagr_5y",
    "roe_avg_5y",
    "ocf_positive_years_5y",
    "debt_ratio_change_5y",
    "gross_margin_change_5y",
    "data_quality",
    "exclude_reasons",
    "missing_fields",
]


def _quote_sql(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect_db(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    _ensure_dir()
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def init_duckdb() -> None:
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_basic (
                market TEXT,
                code TEXT,
                name TEXT,
                industry TEXT,
                is_st BOOLEAN,
                sunset_industry_flag BOOLEAN,
                updated_at TIMESTAMP,
                PRIMARY KEY (market, code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_kline (
                trade_date DATE,
                market TEXT,
                code TEXT,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                turnover_ratio DOUBLE,
                volume_ratio DOUBLE,
                price_change_pct DOUBLE,
                updated_at TIMESTAMP,
                PRIMARY KEY (trade_date, market, code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_fundamental (
                trade_date DATE,
                market TEXT,
                code TEXT,
                pe_dynamic DOUBLE,
                pe_static DOUBLE,
                pe_ttm DOUBLE,
                pb DOUBLE,
                dividend_yield DOUBLE,
                total_mv DOUBLE,
                float_mv DOUBLE,
                roe DOUBLE,
                gross_margin DOUBLE,
                net_margin DOUBLE,
                asset_liability_ratio DOUBLE,
                current_ratio DOUBLE,
                operating_cashflow_3y DOUBLE,
                receivable_revenue_ratio DOUBLE,
                goodwill_equity_ratio DOUBLE,
                interest_debt_asset_ratio DOUBLE,
                ev_ebitda DOUBLE,
                revenue_growth DOUBLE,
                profit_growth DOUBLE,
                revenue_cagr_5y DOUBLE,
                profit_cagr_5y DOUBLE,
                roe_avg_5y DOUBLE,
                debt_ratio_avg_5y DOUBLE,
                gross_margin_avg_5y DOUBLE,
                debt_ratio_change_5y DOUBLE,
                gross_margin_change_5y DOUBLE,
                ocf_positive_years_5y INTEGER,
                investigation_flag BOOLEAN,
                penalty_flag BOOLEAN,
                fund_occupation_flag BOOLEAN,
                illegal_reduce_flag BOOLEAN,
                pledge_ratio DOUBLE,
                no_dividend_5y_flag BOOLEAN,
                audit_change_count INTEGER,
                audit_opinion TEXT,
                data_quality TEXT,
                coverage_ratio DOUBLE,
                enriched_at TIMESTAMP,
                source_note TEXT,
                updated_at TIMESTAMP,
                PRIMARY KEY (trade_date, market, code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                market TEXT NOT NULL,
                code TEXT NOT NULL,
                avg_cost DOUBLE NOT NULL,
                quantity DOUBLE NOT NULL,
                stop_loss_price DOUBLE,
                take_profit_price DOUBLE,
                open_date DATE,
                updated_at TIMESTAMP,
                PRIMARY KEY (market, code)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_position_flows (
                flow_id BIGINT,
                event_time TIMESTAMP,
                market TEXT NOT NULL,
                code TEXT NOT NULL,
                action TEXT NOT NULL,
                price DOUBLE,
                quantity_delta DOUBLE NOT NULL,
                quantity_after DOUBLE NOT NULL,
                note TEXT,
                PRIMARY KEY (flow_id)
            )
            """
        )


def normalize_position_symbol(code: str, market: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Normalize symbol into (market, db_code, display_symbol).
    - HK: db_code 5-digit (e.g. 00700), display 00700.HK
    - A : db_code 6-digit (e.g. 600036), display 600036
    """
    raw = str(code or "").strip().upper()
    if not raw:
        raise ValueError("代码不能为空")

    mk = str(market or "").strip().upper()
    db_code = raw
    if raw.endswith(".HK"):
        mk = "HK"
        db_code = raw.split(".")[0]
    elif raw.endswith(".SH") or raw.endswith(".SZ"):
        mk = "A"
        db_code = raw.split(".")[0]
    elif not mk:
        mk = "HK" if len("".join(ch for ch in raw if ch.isdigit())) <= 5 else "A"

    digits = "".join(ch for ch in db_code if ch.isdigit())
    if not digits:
        raise ValueError(f"非法代码: {code}")
    if mk == "HK":
        db_code = digits.zfill(5)
        display = f"{db_code}.HK"
    else:
        db_code = digits.zfill(6)
        display = db_code
    return mk, db_code, display


def _next_flow_id(conn: duckdb.DuckDBPyConnection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(flow_id), 0) + 1 FROM portfolio_position_flows").fetchone()
    return int(row[0]) if row else 1


def _insert_position_flow(
    conn: duckdb.DuckDBPyConnection,
    *,
    market: str,
    code: str,
    action: str,
    quantity_delta: float,
    quantity_after: float,
    price: Optional[float] = None,
    note: str = "",
) -> None:
    flow_id = _next_flow_id(conn)
    conn.execute(
        """
        INSERT INTO portfolio_position_flows
        (flow_id, event_time, market, code, action, price, quantity_delta, quantity_after, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            flow_id,
            datetime.now(),
            market,
            code,
            str(action).upper(),
            None if price is None else float(price),
            float(quantity_delta),
            float(quantity_after),
            str(note or ""),
        ],
    )


def upsert_position(
    *,
    code: str,
    avg_cost: float,
    quantity: float,
    stop_loss_price: Optional[float] = None,
    take_profit_price: Optional[float] = None,
    open_date: Optional[date] = None,
    market: Optional[str] = None,
    note: str = "manual update",
) -> Dict[str, Any]:
    """Create/update one position and append flow when quantity changed."""
    init_duckdb()
    mk, db_code, display = normalize_position_symbol(code, market=market)
    qty = float(quantity)
    if qty < 0:
        raise ValueError("持仓数量不能为负")
    cost = float(avg_cost)
    if cost <= 0:
        raise ValueError("平均成本必须大于0")
    sl = None if stop_loss_price in (None, "") else float(stop_loss_price)
    tp = None if take_profit_price in (None, "") else float(take_profit_price)
    od = open_date or datetime.now().date()

    with connect_db() as conn:
        old = conn.execute(
            "SELECT quantity FROM portfolio_positions WHERE market = ? AND code = ?",
            [mk, db_code],
        ).fetchone()
        old_qty = float(old[0]) if old else 0.0

        if qty <= 0:
            if old is not None:
                conn.execute(
                    "DELETE FROM portfolio_positions WHERE market = ? AND code = ?",
                    [mk, db_code],
                )
                _insert_position_flow(
                    conn,
                    market=mk,
                    code=db_code,
                    action="CLOSE",
                    quantity_delta=-old_qty,
                    quantity_after=0.0,
                    price=cost,
                    note=note or "position close",
                )
            return {"market": mk, "code": db_code, "symbol": display, "quantity": 0.0, "action": "closed"}

        conn.execute(
            """
            MERGE INTO portfolio_positions AS t
            USING (SELECT ? AS market, ? AS code) AS s
            ON t.market = s.market AND t.code = s.code
            WHEN MATCHED THEN UPDATE SET
                avg_cost = ?,
                quantity = ?,
                stop_loss_price = ?,
                take_profit_price = ?,
                open_date = ?,
                updated_at = ?
            WHEN NOT MATCHED THEN INSERT
                (market, code, avg_cost, quantity, stop_loss_price, take_profit_price, open_date, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                mk,
                db_code,
                cost,
                qty,
                sl,
                tp,
                od,
                datetime.now(),
                mk,
                db_code,
                cost,
                qty,
                sl,
                tp,
                od,
                datetime.now(),
            ],
        )

        delta = qty - old_qty
        if abs(delta) > 1e-9:
            if old_qty <= 0 and qty > 0:
                action = "OPEN"
            elif delta > 0:
                action = "ADD"
            elif qty > 0:
                action = "REDUCE"
            else:
                action = "CLOSE"
            _insert_position_flow(
                conn,
                market=mk,
                code=db_code,
                action=action,
                quantity_delta=delta,
                quantity_after=qty,
                price=cost,
                note=note,
            )

    return {"market": mk, "code": db_code, "symbol": display, "quantity": qty, "action": "upserted"}


def delete_position(code: str, market: Optional[str] = None, note: str = "manual delete") -> bool:
    """Delete one position and append close flow."""
    init_duckdb()
    mk, db_code, _ = normalize_position_symbol(code, market=market)
    with connect_db() as conn:
        old = conn.execute(
            "SELECT quantity, avg_cost FROM portfolio_positions WHERE market = ? AND code = ?",
            [mk, db_code],
        ).fetchone()
        if not old:
            return False
        old_qty = float(old[0])
        old_cost = float(old[1]) if old[1] is not None else None
        conn.execute("DELETE FROM portfolio_positions WHERE market = ? AND code = ?", [mk, db_code])
        _insert_position_flow(
            conn,
            market=mk,
            code=db_code,
            action="CLOSE",
            quantity_delta=-old_qty,
            quantity_after=0.0,
            price=old_cost,
            note=note,
        )
    return True


def get_positions() -> pd.DataFrame:
    """Return current positions with display symbol and basic info."""
    init_duckdb()
    with connect_db(read_only=True) as conn:
        df = conn.execute(
            """
            SELECT
                p.market,
                p.code,
                CASE WHEN p.market = 'HK' THEN p.code || '.HK' ELSE p.code END AS symbol,
                COALESCE(b.name, '') AS name,
                p.avg_cost,
                p.quantity,
                p.stop_loss_price,
                p.take_profit_price,
                p.open_date,
                p.updated_at
            FROM portfolio_positions p
            LEFT JOIN stock_basic b
              ON p.market = b.market AND p.code = b.code
            ORDER BY p.market, p.code
            """
        ).df()
    return df


def get_position_flows(limit: int = 200) -> pd.DataFrame:
    """Return latest position flow logs."""
    init_duckdb()
    with connect_db(read_only=True) as conn:
        df = conn.execute(
            """
            SELECT
                flow_id,
                event_time,
                market,
                code,
                CASE WHEN market = 'HK' THEN code || '.HK' ELSE code END AS symbol,
                action,
                price,
                quantity_delta,
                quantity_after,
                note
            FROM portfolio_position_flows
            ORDER BY flow_id DESC
            LIMIT ?
            """,
            [int(max(1, limit))],
        ).df()
    return df


def get_latest_prices_for_positions(positions_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch latest close price for given positions from daily_kline."""
    if positions_df is None or positions_df.empty:
        return pd.DataFrame(columns=["market", "code", "close", "trade_date"])
    codes = positions_df[["market", "code"]].dropna().drop_duplicates().copy()
    if codes.empty:
        return pd.DataFrame(columns=["market", "code", "close", "trade_date"])

    init_duckdb()
    with connect_db(read_only=True) as conn:
        conn.register("_pos_codes", codes)
        df = conn.execute(
            """
            WITH latest AS (
                SELECT k.market, k.code, MAX(k.trade_date) AS trade_date
                FROM daily_kline k
                INNER JOIN _pos_codes p
                  ON k.market = p.market AND k.code = p.code
                GROUP BY k.market, k.code
            )
            SELECT k.market, k.code, k.close, k.trade_date
            FROM latest l
            INNER JOIN daily_kline k
              ON l.market = k.market AND l.code = k.code AND l.trade_date = k.trade_date
            """
        ).df()
        conn.unregister("_pos_codes")
    return df


def get_position_pnl(total_capital: float) -> pd.DataFrame:
    """Build position PnL table with exposure ratio."""
    pos = get_positions()
    if pos.empty:
        return pos
    px = get_latest_prices_for_positions(pos)
    out = pos.merge(px, on=["market", "code"], how="left")
    out["close"] = pd.to_numeric(out.get("close"), errors="coerce")
    out["avg_cost"] = pd.to_numeric(out.get("avg_cost"), errors="coerce")
    out["quantity"] = pd.to_numeric(out.get("quantity"), errors="coerce")
    out["cost_value"] = out["avg_cost"] * out["quantity"]
    out["market_value"] = out["close"] * out["quantity"]
    out["unrealized_pnl"] = out["market_value"] - out["cost_value"]
    out["unrealized_pnl_pct"] = np.where(
        out["avg_cost"] > 0,
        out["close"] / out["avg_cost"] - 1.0,
        np.nan,
    )
    cap = float(total_capital) if float(total_capital) > 0 else np.nan
    out["weight_pct"] = np.where(np.isfinite(cap), out["market_value"] / cap, np.nan)
    return out


def get_atr20(market: str, code: str, lookback: int = 80, window: int = 20) -> Dict[str, Any]:
    """Calculate ATR(window) based on daily_kline for one symbol."""
    mk, db_code, symbol = normalize_position_symbol(code, market=market)
    init_duckdb()
    with connect_db(read_only=True) as conn:
        df = conn.execute(
            """
            SELECT trade_date, high, low, close
            FROM daily_kline
            WHERE market = ? AND code = ?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            [mk, db_code, int(max(40, lookback))],
        ).df()
    if df.empty:
        return {"market": mk, "code": db_code, "symbol": symbol, "atr": None, "close": None, "trade_date": None}

    df = df.sort_values("trade_date").copy()
    for col in ("high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        return {"market": mk, "code": db_code, "symbol": symbol, "atr": None, "close": None, "trade_date": None}

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(int(max(2, window))).mean().iloc[-1]
    last_close = float(df["close"].iloc[-1]) if len(df) else None
    last_date = pd.to_datetime(df["trade_date"].iloc[-1]).date() if len(df) else None
    return {
        "market": mk,
        "code": db_code,
        "symbol": symbol,
        "atr": None if pd.isna(atr) else float(atr),
        "close": last_close,
        "trade_date": last_date,
    }


def _normalize_trade_date(trade_date: Optional[str]) -> str:
    if trade_date:
        text = str(trade_date).strip()
        for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return datetime.now().strftime("%Y-%m-%d")


def _to_bool_col(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    true_set = {"1", "true", "y", "yes", "是", "True", "TRUE"}
    return series.map(lambda v: None if pd.isna(v) else str(v).strip() in true_set)


def _upsert_stock_basic(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    basic = df[["market", "code", "name", "industry", "is_st", "sunset_industry_flag"]].copy()
    basic["market"] = basic["market"].astype(str).str.strip()
    basic["code"] = basic["code"].astype(str).str.strip()
    basic["name"] = basic["name"].astype(str).str.strip()
    basic["industry"] = basic["industry"].astype(str).str.strip()
    basic["is_st"] = _to_bool_col(basic["is_st"])
    basic["sunset_industry_flag"] = _to_bool_col(basic["sunset_industry_flag"])
    basic["updated_at"] = datetime.now()

    conn.register("_stock_basic_src", basic)
    conn.execute(
        """
        MERGE INTO stock_basic AS t
        USING _stock_basic_src AS s
        ON t.market = s.market AND t.code = s.code
        WHEN MATCHED THEN UPDATE SET
            name = s.name,
            industry = s.industry,
            is_st = s.is_st,
            sunset_industry_flag = s.sunset_industry_flag,
            updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT VALUES
            (s.market, s.code, s.name, s.industry, s.is_st, s.sunset_industry_flag, s.updated_at)
        """
    )
    conn.unregister("_stock_basic_src")


def _upsert_daily_kline(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, trade_date: str) -> None:
    if df.empty:
        return
    kline = pd.DataFrame(
        {
            "trade_date": trade_date,
            "market": df["market"].astype(str).str.strip(),
            "code": df["code"].astype(str).str.strip(),
            "open": pd.to_numeric(df.get("close_price"), errors="coerce"),
            "high": pd.to_numeric(df.get("close_price"), errors="coerce"),
            "low": pd.to_numeric(df.get("close_price"), errors="coerce"),
            "close": pd.to_numeric(df.get("close_price"), errors="coerce"),
            "volume": pd.NA,
            "amount": pd.to_numeric(df.get("amount"), errors="coerce"),
            "turnover_ratio": pd.to_numeric(df.get("turnover_ratio"), errors="coerce"),
            "volume_ratio": pd.to_numeric(df.get("volume_ratio"), errors="coerce"),
            "price_change_pct": pd.to_numeric(df.get("price_change_pct"), errors="coerce"),
            "updated_at": datetime.now(),
        }
    )
    conn.register("_daily_kline_src", kline)
    conn.execute(
        """
        MERGE INTO daily_kline AS t
        USING _daily_kline_src AS s
        ON t.trade_date = s.trade_date AND t.market = s.market AND t.code = s.code
        WHEN MATCHED THEN UPDATE SET
            open = s.open,
            high = s.high,
            low = s.low,
            close = s.close,
            volume = s.volume,
            amount = s.amount,
            turnover_ratio = s.turnover_ratio,
            volume_ratio = s.volume_ratio,
            price_change_pct = s.price_change_pct,
            updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT VALUES
            (s.trade_date, s.market, s.code, s.open, s.high, s.low, s.close, s.volume, s.amount, s.turnover_ratio, s.volume_ratio, s.price_change_pct, s.updated_at)
        """
    )
    conn.unregister("_daily_kline_src")


def _upsert_daily_fundamental(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, trade_date: str) -> None:
    if df.empty:
        return
    f = pd.DataFrame(
        {
            "trade_date": trade_date,
            "market": df["market"].astype(str).str.strip(),
            "code": df["code"].astype(str).str.strip(),
            "pe_dynamic": pd.to_numeric(df.get("pe_dynamic"), errors="coerce"),
            "pe_static": pd.to_numeric(df.get("pe_static"), errors="coerce"),
            "pe_ttm": pd.to_numeric(df.get("pe_ttm"), errors="coerce"),
            "pb": pd.to_numeric(df.get("pb"), errors="coerce"),
            "dividend_yield": pd.to_numeric(df.get("dividend_yield"), errors="coerce"),
            "total_mv": pd.to_numeric(df.get("total_mv"), errors="coerce"),
            "float_mv": pd.to_numeric(df.get("float_mv"), errors="coerce"),
            "roe": pd.to_numeric(df.get("roe"), errors="coerce"),
            "gross_margin": pd.to_numeric(df.get("gross_margin"), errors="coerce"),
            "net_margin": pd.to_numeric(df.get("net_margin"), errors="coerce"),
            "asset_liability_ratio": pd.to_numeric(df.get("asset_liability_ratio"), errors="coerce"),
            "current_ratio": pd.to_numeric(df.get("current_ratio"), errors="coerce"),
            "operating_cashflow_3y": pd.to_numeric(df.get("operating_cashflow_3y"), errors="coerce"),
            "receivable_revenue_ratio": pd.to_numeric(df.get("receivable_revenue_ratio"), errors="coerce"),
            "goodwill_equity_ratio": pd.to_numeric(df.get("goodwill_equity_ratio"), errors="coerce"),
            "interest_debt_asset_ratio": pd.to_numeric(df.get("interest_debt_asset_ratio"), errors="coerce"),
            "ev_ebitda": pd.to_numeric(df.get("ev_ebitda"), errors="coerce"),
            "revenue_growth": pd.to_numeric(df.get("revenue_growth"), errors="coerce"),
            "profit_growth": pd.to_numeric(df.get("profit_growth"), errors="coerce"),
            "revenue_cagr_5y": pd.to_numeric(df.get("revenue_cagr_5y"), errors="coerce"),
            "profit_cagr_5y": pd.to_numeric(df.get("profit_cagr_5y"), errors="coerce"),
            "roe_avg_5y": pd.to_numeric(df.get("roe_avg_5y"), errors="coerce"),
            "debt_ratio_avg_5y": pd.to_numeric(df.get("debt_ratio_avg_5y"), errors="coerce"),
            "gross_margin_avg_5y": pd.to_numeric(df.get("gross_margin_avg_5y"), errors="coerce"),
            "debt_ratio_change_5y": pd.to_numeric(df.get("debt_ratio_change_5y"), errors="coerce"),
            "gross_margin_change_5y": pd.to_numeric(df.get("gross_margin_change_5y"), errors="coerce"),
            "ocf_positive_years_5y": pd.to_numeric(df.get("ocf_positive_years_5y"), errors="coerce").astype("Int64"),
            "investigation_flag": _to_bool_col(df.get("investigation_flag", pd.Series(dtype=object))),
            "penalty_flag": _to_bool_col(df.get("penalty_flag", pd.Series(dtype=object))),
            "fund_occupation_flag": _to_bool_col(df.get("fund_occupation_flag", pd.Series(dtype=object))),
            "illegal_reduce_flag": _to_bool_col(df.get("illegal_reduce_flag", pd.Series(dtype=object))),
            "pledge_ratio": pd.to_numeric(df.get("pledge_ratio"), errors="coerce"),
            "no_dividend_5y_flag": _to_bool_col(df.get("no_dividend_5y_flag", pd.Series(dtype=object))),
            "audit_change_count": pd.to_numeric(df.get("audit_change_count"), errors="coerce").astype("Int64"),
            "audit_opinion": df.get("audit_opinion", pd.Series(dtype=object)).astype(str),
            "data_quality": df.get("data_quality", pd.Series(dtype=object)).astype(str),
            "coverage_ratio": pd.to_numeric(df.get("coverage_ratio"), errors="coerce"),
            "enriched_at": pd.to_datetime(df.get("enriched_at"), errors="coerce"),
            "source_note": df.get("source_note", pd.Series(dtype=object)).astype(str),
            "updated_at": datetime.now(),
        }
    )
    conn.register("_daily_fund_src", f)
    conn.execute(
        """
        MERGE INTO daily_fundamental AS t
        USING _daily_fund_src AS s
        ON t.trade_date = s.trade_date AND t.market = s.market AND t.code = s.code
        WHEN MATCHED THEN UPDATE SET
            pe_dynamic = s.pe_dynamic,
            pe_static = s.pe_static,
            pe_ttm = s.pe_ttm,
            pb = s.pb,
            dividend_yield = s.dividend_yield,
            total_mv = s.total_mv,
            float_mv = s.float_mv,
            roe = s.roe,
            gross_margin = s.gross_margin,
            net_margin = s.net_margin,
            asset_liability_ratio = s.asset_liability_ratio,
            current_ratio = s.current_ratio,
            operating_cashflow_3y = s.operating_cashflow_3y,
            receivable_revenue_ratio = s.receivable_revenue_ratio,
            goodwill_equity_ratio = s.goodwill_equity_ratio,
            interest_debt_asset_ratio = s.interest_debt_asset_ratio,
            ev_ebitda = s.ev_ebitda,
            revenue_growth = s.revenue_growth,
            profit_growth = s.profit_growth,
            revenue_cagr_5y = s.revenue_cagr_5y,
            profit_cagr_5y = s.profit_cagr_5y,
            roe_avg_5y = s.roe_avg_5y,
            debt_ratio_avg_5y = s.debt_ratio_avg_5y,
            gross_margin_avg_5y = s.gross_margin_avg_5y,
            debt_ratio_change_5y = s.debt_ratio_change_5y,
            gross_margin_change_5y = s.gross_margin_change_5y,
            ocf_positive_years_5y = s.ocf_positive_years_5y,
            investigation_flag = s.investigation_flag,
            penalty_flag = s.penalty_flag,
            fund_occupation_flag = s.fund_occupation_flag,
            illegal_reduce_flag = s.illegal_reduce_flag,
            pledge_ratio = s.pledge_ratio,
            no_dividend_5y_flag = s.no_dividend_5y_flag,
            audit_change_count = s.audit_change_count,
            audit_opinion = s.audit_opinion,
            data_quality = s.data_quality,
            coverage_ratio = s.coverage_ratio,
            enriched_at = s.enriched_at,
            source_note = s.source_note,
            updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT VALUES
            (s.trade_date, s.market, s.code, s.pe_dynamic, s.pe_static, s.pe_ttm, s.pb, s.dividend_yield, s.total_mv, s.float_mv, s.roe, s.gross_margin, s.net_margin, s.asset_liability_ratio, s.current_ratio, s.operating_cashflow_3y, s.receivable_revenue_ratio, s.goodwill_equity_ratio, s.interest_debt_asset_ratio, s.ev_ebitda, s.revenue_growth, s.profit_growth, s.revenue_cagr_5y, s.profit_cagr_5y, s.roe_avg_5y, s.debt_ratio_avg_5y, s.gross_margin_avg_5y, s.debt_ratio_change_5y, s.gross_margin_change_5y, s.ocf_positive_years_5y, s.investigation_flag, s.penalty_flag, s.fund_occupation_flag, s.illegal_reduce_flag, s.pledge_ratio, s.no_dividend_5y_flag, s.audit_change_count, s.audit_opinion, s.data_quality, s.coverage_ratio, s.enriched_at, s.source_note, s.updated_at)
        """
    )
    conn.unregister("_daily_fund_src")


def bulk_upsert_market_daily(snapshot_df: pd.DataFrame, trade_date: Optional[str] = None) -> Dict[str, Any]:
    if snapshot_df is None or snapshot_df.empty:
        return {"trade_date": _normalize_trade_date(trade_date), "rows": 0}
    init_duckdb()
    td = _normalize_trade_date(trade_date)
    with connect_db() as conn:
        _upsert_stock_basic(conn, snapshot_df)
        _upsert_daily_kline(conn, snapshot_df, td)
        _upsert_daily_fundamental(conn, snapshot_df, td)
    return {"trade_date": td, "rows": int(len(snapshot_df))}


def _enabled(cfg: Dict[str, Any], key: str) -> bool:
    return bool(cfg.get(key, False))


def _extract_keywords(text: str) -> List[str]:
    raw = str(text or "")
    kws = [k.strip().lower() for k in raw.replace("，", ",").split(",")]
    return [k for k in kws if k]


def _build_predicates(cfg: Dict[str, Any], include_rearview: bool = True) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    risk = cfg.get("risk", {}) or {}
    quality = cfg.get("quality", {}) or {}
    valuation = cfg.get("valuation", {}) or {}
    growth = cfg.get("growth_liquidity", {}) or {}
    rear = cfg.get("rearview_5y", {}) or {}

    preds: List[str] = []
    reason_rules: List[Tuple[str, str]] = []
    missing_rules: List[Tuple[str, str]] = []

    scope = str(risk.get("market_scope", "all")).upper()
    if scope in {"A", "HK"}:
        preds.append(f"x.market = {_quote_sql(scope)}")
        reason_rules.append(("市场范围", f"x.market = {_quote_sql(scope)}"))

    if _enabled(risk, "industry_include_enabled"):
        kws = _extract_keywords(risk.get("industry_include_keywords", ""))
        if kws:
            ors = " OR ".join([f"LOWER(COALESCE(x.industry, '')) LIKE '%{k.replace('%', '%%')}%'" for k in kws])
            preds.append(f"({ors})")
            reason_rules.append(("行业包含", f"({ors})"))

    bool_exclude_rules = [
        ("exclude_st", "排除ST", "COALESCE(x.is_st, FALSE) = FALSE"),
        ("exclude_investigation", "排除立案", "COALESCE(x.investigation_flag, FALSE) = FALSE"),
        ("exclude_penalty", "排除处罚", "COALESCE(x.penalty_flag, FALSE) = FALSE"),
        ("exclude_fund_occupation", "排除资金占用", "COALESCE(x.fund_occupation_flag, FALSE) = FALSE"),
        ("exclude_illegal_reduce", "排除违规减持", "COALESCE(x.illegal_reduce_flag, FALSE) = FALSE"),
        ("exclude_no_dividend_5y", "排除5年无分红", "COALESCE(x.no_dividend_5y_flag, FALSE) = FALSE"),
        ("exclude_sunset_industry", "排除夕阳行业", "COALESCE(x.sunset_industry_flag, FALSE) = FALSE"),
    ]
    for key, label, expr in bool_exclude_rules:
        if _enabled(risk, key):
            preds.append(expr)
            reason_rules.append((label, expr))

    if _enabled(risk, "require_standard_audit"):
        expr = "LOWER(COALESCE(x.audit_opinion, '')) LIKE '%标准%'"
        preds.append(expr)
        reason_rules.append(("审计意见", expr))
        missing_rules.append(("audit_opinion", "x.audit_opinion"))

    numeric_rules = [
        (risk, "pledge_ratio_max_enabled", "pledge_ratio_max", "x.pledge_ratio <= {v}", "质押比例上限", "x.pledge_ratio"),
        (risk, "audit_change_max_enabled", "audit_change_max", "x.audit_change_count <= {v}", "审计机构变更上限", "x.audit_change_count"),
        (quality, "ocf_3y_min_enabled", "ocf_3y_min", "x.operating_cashflow_3y >= {v}", "经营现金流3年下限", "x.operating_cashflow_3y"),
        (quality, "asset_liability_max_enabled", "asset_liability_max", "x.asset_liability_ratio <= {v}", "资产负债率上限", "x.asset_liability_ratio"),
        (quality, "interest_debt_asset_max_enabled", "interest_debt_asset_max", "x.interest_debt_asset_ratio <= {v}", "有息负债率上限", "x.interest_debt_asset_ratio"),
        (quality, "roe_min_enabled", "roe_min", "x.roe >= {v}", "ROE下限", "x.roe"),
        (quality, "gross_margin_min_enabled", "gross_margin_min", "x.gross_margin >= {v}", "毛利率下限", "x.gross_margin"),
        (quality, "net_margin_min_enabled", "net_margin_min", "x.net_margin >= {v}", "净利率下限", "x.net_margin"),
        (quality, "receivable_ratio_max_enabled", "receivable_ratio_max", "x.receivable_revenue_ratio <= {v}", "应收占比上限", "x.receivable_revenue_ratio"),
        (quality, "goodwill_ratio_max_enabled", "goodwill_ratio_max", "x.goodwill_equity_ratio <= {v}", "商誉占比上限", "x.goodwill_equity_ratio"),
        (valuation, "pe_ttm_min_enabled", "pe_ttm_min", "x.pe_ttm >= {v}", "PE-TTM下限", "x.pe_ttm"),
        (valuation, "pe_ttm_max_enabled", "pe_ttm_max", "x.pe_ttm <= {v}", "PE-TTM上限", "x.pe_ttm"),
        (valuation, "pb_max_enabled", "pb_max", "x.pb <= {v}", "PB上限", "x.pb"),
        (valuation, "ev_ebitda_max_enabled", "ev_ebitda_max", "x.ev_ebitda <= {v}", "EV/EBITDA上限", "x.ev_ebitda"),
        (valuation, "dividend_min_enabled", "dividend_min", "x.dividend_yield >= {v}", "股息率下限", "x.dividend_yield"),
        (valuation, "dividend_max_enabled", "dividend_max", "x.dividend_yield <= {v}", "股息率上限", "x.dividend_yield"),
        (growth, "revenue_growth_min_enabled", "revenue_growth_min", "x.revenue_growth >= {v}", "营收增速下限", "x.revenue_growth"),
        (growth, "profit_growth_min_enabled", "profit_growth_min", "x.profit_growth >= {v}", "利润增速下限", "x.profit_growth"),
        (growth, "market_cap_min_enabled", "market_cap_min", "x.total_mv >= {v}", "市值下限", "x.total_mv"),
        (growth, "market_cap_max_enabled", "market_cap_max", "x.total_mv <= {v}", "市值上限", "x.total_mv"),
        (growth, "turnover_min_enabled", "turnover_min", "x.turnover_ratio >= {v}", "换手率下限", "x.turnover_ratio"),
        (growth, "turnover_max_enabled", "turnover_max", "x.turnover_ratio <= {v}", "换手率上限", "x.turnover_ratio"),
        (growth, "volume_ratio_min_enabled", "volume_ratio_min", "x.volume_ratio >= {v}", "量比下限", "x.volume_ratio"),
        (growth, "volume_ratio_max_enabled", "volume_ratio_max", "x.volume_ratio <= {v}", "量比上限", "x.volume_ratio"),
        (growth, "amount_min_enabled", "amount_min", "x.amount >= {v}", "成交额下限", "x.amount"),
    ]
    if include_rearview:
        numeric_rules.extend(
            [
                (rear, "revenue_cagr_5y_min_enabled", "revenue_cagr_5y_min", "x.revenue_cagr_5y >= {v}", "5年营收CAGR下限", "x.revenue_cagr_5y"),
                (rear, "profit_cagr_5y_min_enabled", "profit_cagr_5y_min", "x.profit_cagr_5y >= {v}", "5年利润CAGR下限", "x.profit_cagr_5y"),
                (rear, "roe_avg_5y_min_enabled", "roe_avg_5y_min", "x.roe_avg_5y >= {v}", "5年ROE均值下限", "x.roe_avg_5y"),
                (rear, "ocf_positive_years_5y_min_enabled", "ocf_positive_years_5y_min", "x.ocf_positive_years_5y >= {v}", "5年经营现金流为正年数下限", "x.ocf_positive_years_5y"),
                (rear, "debt_ratio_change_5y_max_enabled", "debt_ratio_change_5y_max", "x.debt_ratio_change_5y <= {v}", "5年负债率变化上限", "x.debt_ratio_change_5y"),
                (rear, "gross_margin_change_5y_min_enabled", "gross_margin_change_5y_min", "x.gross_margin_change_5y >= {v}", "5年毛利率变化下限", "x.gross_margin_change_5y"),
            ]
        )

    for section, enabled_key, value_key, tmpl, label, col in numeric_rules:
        if _enabled(section, enabled_key):
            value = section.get(value_key)
            if value is None:
                continue
            expr = tmpl.format(v=float(value))
            preds.append(expr)
            reason_rules.append((label, expr))
            missing_rules.append((label, col))

    return preds, reason_rules, missing_rules


def _base_sql(source_codes: Optional[Sequence[str]] = None) -> str:
    code_filter = ""
    if source_codes:
        items = ",".join([_quote_sql(str(c).strip()) for c in source_codes if str(c).strip()])
        if items:
            code_filter = f" AND f.code IN ({items})"
    return f"""
    WITH max_d AS (
        SELECT MAX(trade_date) AS d FROM daily_fundamental
    ),
    x AS (
        SELECT
            f.market,
            f.code,
            b.name,
            b.industry,
            b.is_st,
            b.sunset_industry_flag,
            k.close,
            k.amount,
            k.turnover_ratio,
            k.volume_ratio,
            k.price_change_pct,
            f.pe_dynamic,
            f.pe_static,
            f.pe_ttm,
            f.pb,
            f.dividend_yield,
            f.total_mv,
            f.float_mv,
            f.roe,
            f.gross_margin,
            f.net_margin,
            f.asset_liability_ratio,
            f.current_ratio,
            f.operating_cashflow_3y,
            f.receivable_revenue_ratio,
            f.goodwill_equity_ratio,
            f.interest_debt_asset_ratio,
            f.ev_ebitda,
            f.revenue_growth,
            f.profit_growth,
            f.revenue_cagr_5y,
            f.profit_cagr_5y,
            f.roe_avg_5y,
            f.debt_ratio_avg_5y,
            f.gross_margin_avg_5y,
            f.debt_ratio_change_5y,
            f.gross_margin_change_5y,
            f.ocf_positive_years_5y,
            f.investigation_flag,
            f.penalty_flag,
            f.fund_occupation_flag,
            f.illegal_reduce_flag,
            f.pledge_ratio,
            f.no_dividend_5y_flag,
            f.audit_change_count,
            f.audit_opinion,
            f.data_quality,
            f.coverage_ratio,
            f.enriched_at,
            f.source_note
        FROM daily_fundamental f
        LEFT JOIN daily_kline k
            ON f.trade_date = k.trade_date AND f.market = k.market AND f.code = k.code
        LEFT JOIN stock_basic b
            ON f.market = b.market AND f.code = b.code
        WHERE f.trade_date = (SELECT d FROM max_d)
        {code_filter}
    )
    """


def _select_projection(reason_expr: str, missing_expr: str) -> str:
    return f"""
    SELECT
        x.market,
        x.code,
        COALESCE(x.name, '') AS name,
        COALESCE(x.industry, '') AS industry,
        x.pe_ttm,
        x.pb,
        x.dividend_yield,
        x.roe,
        x.asset_liability_ratio,
        x.turnover_ratio,
        x.volume_ratio,
        x.total_mv,
        x.revenue_cagr_5y,
        x.profit_cagr_5y,
        x.roe_avg_5y,
        x.ocf_positive_years_5y,
        x.debt_ratio_change_5y,
        x.gross_margin_change_5y,
        CASE
            WHEN x.pe_ttm IS NOT NULL AND x.pb IS NOT NULL AND x.roe IS NOT NULL THEN 'full'
            WHEN x.pe_ttm IS NULL AND x.pb IS NULL AND x.roe IS NULL THEN 'missing'
            ELSE 'partial'
        END AS data_quality,
        {reason_expr} AS exclude_reasons,
        {missing_expr} AS missing_fields
    FROM x
    """


def query_filter_results(
    config: Dict[str, Any],
    source_codes: Optional[Sequence[str]] = None,
    include_rearview: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    init_duckdb()
    missing_policy = str(config.get("missing_policy", "ignore")).strip().lower()
    preds, reason_rules, missing_rules = _build_predicates(config, include_rearview=include_rearview)
    pred_expr = " AND ".join(preds) if preds else "TRUE"

    missing_parts = [f"CASE WHEN {col} IS NULL THEN '{label},' ELSE '' END" for label, col in missing_rules]
    missing_concat = " || ".join(missing_parts) if missing_parts else "''"
    missing_expr = f"TRIM(BOTH ',' FROM ({missing_concat}))"

    reason_parts = [f"CASE WHEN NOT ({expr}) THEN '{label},' ELSE '' END" for label, expr in reason_rules]
    if missing_policy == "exclude":
        reason_parts.append(f"CASE WHEN {missing_expr} <> '' THEN '缺失字段,' ELSE '' END")
    reason_concat = " || ".join(reason_parts) if reason_parts else "''"
    reason_expr = f"TRIM(BOTH ',' FROM ({reason_concat}))"

    base_sql = _base_sql(source_codes)
    select_sql = _select_projection(reason_expr=reason_expr, missing_expr=missing_expr)

    passed_where = f"({pred_expr})"
    if missing_policy == "exclude":
        passed_where = f"({passed_where}) AND ({missing_expr} = '')"

    if missing_policy == "exclude":
        rejected_where = f"(NOT ({pred_expr})) AND ({missing_expr} = '')"
    else:
        rejected_where = f"NOT ({pred_expr})"

    missing_where = f"{missing_expr} <> ''"

    with connect_db(read_only=True) as conn:
        passed_df = conn.execute(f"{base_sql} {select_sql} WHERE {passed_where} ORDER BY x.total_mv DESC NULLS LAST, x.code").df()
        rejected_df = conn.execute(f"{base_sql} {select_sql} WHERE {rejected_where} ORDER BY x.total_mv DESC NULLS LAST, x.code").df()
        missing_df = conn.execute(f"{base_sql} {select_sql} WHERE {missing_where} ORDER BY x.total_mv DESC NULLS LAST, x.code").df()
        total_rows = conn.execute(f"{base_sql} SELECT COUNT(*) AS c FROM x").fetchone()[0]

    for df in (passed_df, rejected_df, missing_df):
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[OUTPUT_COLUMNS]

    stats = {
        "total": int(total_rows),
        "passed": int(len(passed_df)),
        "rejected": int(len(rejected_df)),
        "missing": int(len(missing_df)),
    }
    return (
        passed_df[OUTPUT_COLUMNS].copy(),
        rejected_df[OUTPUT_COLUMNS].copy(),
        missing_df[OUTPUT_COLUMNS].copy(),
        stats,
    )


def run_bulk_upsert_from_filter_snapshot(trade_date: Optional[str] = None) -> Dict[str, Any]:
    """
    批量 Upsert 脚本入口：
    从 filter 快照库读取全市场 A/H 数据，写入 DuckDB 三张核心表。
    """
    filter_db = PROJECT_ROOT / "apps" / "filter" / "data" / "filter_market.db"
    if not filter_db.exists():
        return {"trade_date": _normalize_trade_date(trade_date), "rows": 0, "error": f"snapshot db not found: {filter_db}"}
    import sqlite3

    with sqlite3.connect(str(filter_db)) as conn:
        try:
            snap_df = pd.read_sql_query("SELECT * FROM market_snapshot", conn)
        except Exception as exc:
            return {"trade_date": _normalize_trade_date(trade_date), "rows": 0, "error": f"read snapshot failed: {exc}"}
    return bulk_upsert_market_daily(snap_df, trade_date=trade_date)


if __name__ == "__main__":
    result = run_bulk_upsert_from_filter_snapshot()
    print(result)
