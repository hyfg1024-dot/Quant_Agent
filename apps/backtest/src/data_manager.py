"""Data fetch/cache/clean utilities for HK long-short backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency runtime failure
    yf = None

try:
    import akshare as ak
except Exception:  # pragma: no cover
    ak = None


@dataclass
class PreparedSeries:
    """Prepared series for simulation usage."""

    raw: pd.DataFrame
    aligned: pd.DataFrame


class DataManager:
    """Download and cache price data with yfinance primary and akshare fallback."""

    def __init__(self, data_dir: Path, logger=print) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._anomaly_warned: set[str] = set()

    def fetch_stock_data(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch one stock OHLCV dataframe in standard columns."""
        return self._fetch_data(code=code, start=start, end=end, is_index=False)

    def fetch_index_data(self, code: str, start: str, end: str) -> pd.DataFrame:
        """Fetch one benchmark index dataframe in standard columns."""
        return self._fetch_data(code=code, start=start, end=end, is_index=True)

    def get_cached_data(self, code: str) -> Optional[pd.DataFrame]:
        """Read cached CSV for one code."""
        path = self._cache_path(code)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if "date" not in df.columns:
                return None
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            return self._standardize_columns(df)
        except Exception as exc:
            self.logger(f"[WARN] 读取缓存失败 {code}: {exc}")
            return None

    def update_cache(self, code: str, df: pd.DataFrame) -> None:
        """Write dataframe to local CSV cache."""
        out = df.copy().sort_index()
        out = self._standardize_columns(out)
        out.insert(0, "date", out.index)
        out.to_csv(self._cache_path(code), index=False)

    def validate_universe(self, universe: List[str], start: str, end: str) -> Dict[str, Dict[str, Any]]:
        """Validate data availability for a list of symbols."""
        report: Dict[str, Dict[str, Any]] = {}
        for code in universe:
            try:
                df = self.fetch_stock_data(code, start=start, end=end)
                if df.empty:
                    report[code] = {
                        "status": "empty",
                        "rows": 0,
                        "start": "",
                        "end": "",
                        "message": "无数据",
                    }
                    continue
                report[code] = {
                    "status": "ok",
                    "rows": int(len(df)),
                    "start": str(df.index.min().date()),
                    "end": str(df.index.max().date()),
                    "message": "",
                }
            except Exception as exc:
                report[code] = {
                    "status": "error",
                    "rows": 0,
                    "start": "",
                    "end": "",
                    "message": str(exc),
                }
        return report

    def prepare_for_calendar(
        self,
        df: pd.DataFrame,
        calendar: pd.DatetimeIndex,
        max_suspend_days: int = 30,
    ) -> PreparedSeries:
        """Align one symbol dataframe to shared calendar and generate tradability flags."""
        raw = self._standardize_columns(df)
        aligned = raw.reindex(calendar)

        missing_close = aligned["adj_close"].isna()
        miss_count = np.zeros(len(aligned), dtype=int)
        streak = 0
        for i, miss in enumerate(missing_close.tolist()):
            if miss:
                streak += 1
            else:
                streak = 0
            miss_count[i] = streak

        aligned["adj_close"] = aligned["adj_close"].ffill()
        aligned["close"] = aligned["close"].ffill()
        aligned["volume"] = aligned["volume"].fillna(0.0)
        aligned["tradable"] = (~missing_close) & (aligned["volume"] > 0)
        aligned["suspended"] = miss_count > int(max_suspend_days)
        aligned["missing_streak"] = miss_count
        return PreparedSeries(raw=raw, aligned=aligned)

    def _fetch_data(self, code: str, start: str, end: str, is_index: bool) -> pd.DataFrame:
        c = code.strip().upper()
        if not c:
            raise ValueError("code 不能为空")

        start_dt = pd.Timestamp(start).date()
        end_dt = pd.Timestamp(end).date()
        if start_dt >= end_dt:
            raise ValueError("start 必须小于 end")

        cached = self.get_cached_data(c)
        merged: Optional[pd.DataFrame] = None

        if cached is not None and not cached.empty:
            merged = cached
            max_cached = merged.index.max().date()
            stale_threshold = date.today() - timedelta(days=1)
            needs_update = max_cached < min(end_dt, stale_threshold)
            if needs_update:
                fetch_start = max(start_dt, max_cached - timedelta(days=7))
                try:
                    fresh = self._download_with_fallback(c, fetch_start, end_dt, is_index=is_index)
                    merged = self._merge_frames(merged, fresh)
                    self.update_cache(c, merged)
                except Exception as exc:
                    # 网络波动/限流时回退到已有缓存，保证回测不中断。
                    self.logger(f"[WARN] {c} 增量更新失败，回退使用本地缓存: {exc}")
        else:
            merged = self._download_with_fallback(c, start_dt, end_dt, is_index=is_index)
            self.update_cache(c, merged)

        out = merged.loc[(merged.index.date >= start_dt) & (merged.index.date <= end_dt)].copy()
        out = self._standardize_columns(out)
        if not out.empty:
            max_out = out.index.max().date()
            if max_out < end_dt:
                self.logger(f"[WARN] {c} 仅使用缓存至 {max_out}，未覆盖到 {end_dt}")
        self._warn_anomaly(c, out)
        return out

    def _download_with_fallback(self, code: str, start: date, end: date, is_index: bool) -> pd.DataFrame:
        errors: List[str] = []

        if yf is not None:
            try:
                yf_df = self._download_yfinance(code, start, end)
                if not yf_df.empty:
                    self.logger(f"[DATA] {code} 使用 yfinance 成功，{len(yf_df)} 行")
                    return yf_df
            except Exception as exc:
                errors.append(f"yfinance: {exc}")

        if ak is not None:
            try:
                ak_df = self._download_akshare(code, start, end, is_index=is_index)
                if not ak_df.empty:
                    self.logger(f"[DATA] {code} 使用 akshare 成功，{len(ak_df)} 行")
                    return ak_df
            except Exception as exc:
                errors.append(f"akshare: {exc}")

        raise RuntimeError(f"{code} 数据下载失败: {' | '.join(errors) if errors else '无可用数据源'}")

    def _download_yfinance(self, code: str, start: date, end: date) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()
        df = yf.download(
            tickers=code,
            start=str(start),
            end=str(end + timedelta(days=1)),
            auto_adjust=False,
            actions=False,
            progress=False,
            interval="1d",
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(c[0]) for c in df.columns]

        out = pd.DataFrame(index=pd.to_datetime(df.index).tz_localize(None))
        out["open"] = pd.to_numeric(df.get("Open"), errors="coerce")
        out["high"] = pd.to_numeric(df.get("High"), errors="coerce")
        out["low"] = pd.to_numeric(df.get("Low"), errors="coerce")
        out["close"] = pd.to_numeric(df.get("Close"), errors="coerce")
        out["adj_close"] = pd.to_numeric(df.get("Adj Close", out["close"]), errors="coerce")
        out["volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")
        return out.dropna(how="all")

    def _download_akshare(self, code: str, start: date, end: date, is_index: bool) -> pd.DataFrame:
        if ak is None:
            return pd.DataFrame()

        if is_index:
            return self._download_akshare_index(code, start, end)
        return self._download_akshare_stock(code, start, end)

    def _download_akshare_stock(self, code: str, start: date, end: date) -> pd.DataFrame:
        if not code.endswith(".HK"):
            raise RuntimeError("akshare fallback 当前仅实现 HK 股票")

        symbol = code.split(".")[0].zfill(4)
        s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        df = pd.DataFrame()

        # Variant 1: stock_hk_hist
        if hasattr(ak, "stock_hk_hist"):
            try:
                df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=s, end_date=e, adjust="")
            except Exception:
                df = pd.DataFrame()

        # Variant 2: stock_hk_daily
        if df.empty and hasattr(ak, "stock_hk_daily"):
            try:
                df = ak.stock_hk_daily(symbol=symbol)
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            raise RuntimeError(f"akshare 无法拉取股票: {code}")

        # Column normalization
        lower_cols = {str(c).lower(): c for c in df.columns}
        date_col = lower_cols.get("date") or lower_cols.get("日期")
        open_col = lower_cols.get("open") or lower_cols.get("开盘")
        high_col = lower_cols.get("high") or lower_cols.get("最高")
        low_col = lower_cols.get("low") or lower_cols.get("最低")
        close_col = lower_cols.get("close") or lower_cols.get("收盘")
        vol_col = lower_cols.get("volume") or lower_cols.get("成交量")

        if not (date_col and close_col):
            raise RuntimeError(f"akshare 字段不足: {code}")

        out = pd.DataFrame()
        out.index = pd.to_datetime(df[date_col]).tz_localize(None)
        out["open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else np.nan
        out["high"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else np.nan
        out["low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else np.nan
        out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        out["adj_close"] = out["close"]
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
        out = out.sort_index()
        out = out.loc[(out.index.date >= start) & (out.index.date <= end)]
        return out.dropna(how="all")

    def _download_akshare_index(self, code: str, start: date, end: date) -> pd.DataFrame:
        if code != "^HSI":
            raise RuntimeError(f"akshare index fallback 当前仅实现 ^HSI: {code}")
        df = pd.DataFrame()

        if hasattr(ak, "stock_hk_index_daily_sina"):
            try:
                df = ak.stock_hk_index_daily_sina(symbol="HSI")
            except Exception:
                df = pd.DataFrame()

        if df.empty and hasattr(ak, "stock_hk_index_daily_em"):
            try:
                df = ak.stock_hk_index_daily_em(symbol="HSI")
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            raise RuntimeError("akshare 无法拉取 ^HSI")

        lower_cols = {str(c).lower(): c for c in df.columns}
        date_col = lower_cols.get("date") or lower_cols.get("日期")
        close_col = lower_cols.get("close") or lower_cols.get("收盘")
        open_col = lower_cols.get("open") or lower_cols.get("开盘")
        high_col = lower_cols.get("high") or lower_cols.get("最高")
        low_col = lower_cols.get("low") or lower_cols.get("最低")
        vol_col = lower_cols.get("volume") or lower_cols.get("成交量")

        if not (date_col and close_col):
            raise RuntimeError("akshare ^HSI 字段不足")

        out = pd.DataFrame()
        out.index = pd.to_datetime(df[date_col]).tz_localize(None)
        out["open"] = pd.to_numeric(df[open_col], errors="coerce") if open_col else np.nan
        out["high"] = pd.to_numeric(df[high_col], errors="coerce") if high_col else np.nan
        out["low"] = pd.to_numeric(df[low_col], errors="coerce") if low_col else np.nan
        out["close"] = pd.to_numeric(df[close_col], errors="coerce")
        out["adj_close"] = out["close"]
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
        out = out.sort_index()
        out = out.loc[(out.index.date >= start) & (out.index.date <= end)]
        return out.dropna(how="all")

    def _cache_path(self, code: str) -> Path:
        safe = code.replace("^", "IDX_").replace("/", "_").replace(".", "_")
        return self.data_dir / f"{safe}.csv"

    def _merge_frames(self, old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if old is None or old.empty:
            return self._standardize_columns(new)
        if new is None or new.empty:
            return self._standardize_columns(old)
        merged = pd.concat([old, new], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return self._standardize_columns(merged)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            if c not in out.columns:
                out[c] = np.nan
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out[["open", "high", "low", "close", "adj_close", "volume"]].sort_index()

    def _warn_anomaly(self, code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if code in self._anomaly_warned:
            return
        ret = df["adj_close"].pct_change(fill_method=None)
        mask = ret.abs() > 0.50
        if bool(mask.any()):
            cnt = int(mask.sum())
            self.logger(f"[WARN] {code} 存在 {cnt} 个绝对涨跌幅>50%的交易日，请人工核查数据")
            self._anomaly_warned.add(code)
