from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "regime_hmm.pkl"


class RegimeDetector:
    """基于 GaussianHMM 的市场状态识别器。"""

    REGIME_ORDER = ["低波动上涨", "高波动震荡", "低波动阴跌", "恐慌杀跌"]
    REGIME_POSITION_RANGE = {
        "低波动上涨": (80.0, 100.0),
        "高波动震荡": (50.0, 70.0),
        "低波动阴跌": (20.0, 40.0),
        "恐慌杀跌": (0.0, 20.0),
    }
    REGIME_SEVERITY = {
        "低波动上涨": 0,
        "高波动震荡": 1,
        "低波动阴跌": 2,
        "恐慌杀跌": 3,
    }

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model: Any = None
        self.scaler: Any = None
        self.state_name_map: Dict[int, str] = {}
        self.feature_columns = ["ret", "vol20", "vol_chg20"]
        self.model_meta: Dict[str, Any] = {}
        self.inflection_signal_strength: float = 0.0
        self._last_state_trace: List[Dict[str, Any]] = []

    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        try:
            out = float(v)
            if np.isnan(out) or np.isinf(out):
                return None
            return out
        except Exception:
            return None

    @staticmethod
    def _find_col(columns: List[str], keywords: List[str]) -> str:
        cols = [str(c) for c in columns]
        for kw in keywords:
            for col in cols:
                if kw.lower() in col.lower():
                    return col
        return ""

    def _validate_and_standardize_input(self, index_daily: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(index_daily, pd.DataFrame) or index_daily.empty:
            raise ValueError("index_daily 为空，无法识别市场状态")

        df = index_daily.copy()
        df.columns = [str(c).strip() for c in df.columns]

        date_col = self._find_col(df.columns.tolist(), ["date", "日期", "trade_date"])
        close_col = self._find_col(df.columns.tolist(), ["close", "收盘"])
        volume_col = self._find_col(df.columns.tolist(), ["volume", "vol", "成交量"])

        if not close_col or not volume_col:
            raise ValueError("index_daily 必须包含 close 和 volume 列（支持中英文列名）")

        if not date_col:
            # 无日期列时，用索引兜底
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
            date_col = "date"

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "close": pd.to_numeric(df[close_col], errors="coerce"),
                "volume": pd.to_numeric(df[volume_col], errors="coerce"),
            }
        )
        out = out.dropna(subset=["date", "close", "volume"]).sort_values("date").reset_index(drop=True)

        if len(out) < 120:
            raise ValueError("有效日线数据不足（<120），无法稳定识别状态")
        return out

    def _build_observation(self, index_daily: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_and_standardize_input(index_daily)

        df["ret"] = df["close"].pct_change()
        df["vol20"] = df["ret"].rolling(20).std()
        df["vol_chg20"] = df["volume"].pct_change(20)

        feat = df[["date", "ret", "vol20", "vol_chg20"]].dropna().copy()
        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        if len(feat) < 80:
            raise ValueError("特征窗口不足（dropna 后 <80），无法训练/预测")
        return feat.reset_index(drop=True)

    def _import_hmm(self):
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:
            raise RuntimeError("缺少 hmmlearn 依赖，请先执行: pip install hmmlearn") from exc
        return GaussianHMM

    def _require_sklearn(self) -> None:
        if StandardScaler is None:
            raise RuntimeError("缺少 scikit-learn 依赖，请先执行: pip install scikit-learn")

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        if std <= 1e-12:
            return np.zeros_like(arr)
        return (arr - mean) / std

    def _map_state_name(self, state_stats: pd.DataFrame) -> Dict[int, str]:
        if state_stats.empty:
            return {}

        stats = state_stats.copy()
        stats["ret_z"] = self._zscore(stats["ret_mean"].to_numpy())
        stats["vol_z"] = self._zscore(stats["vol20_mean"].to_numpy())

        name_map: Dict[int, str] = {}
        for _, row in stats.iterrows():
            s = int(row["state"])
            ret_z = float(row["ret_z"])
            vol_z = float(row["vol_z"])

            # 四类模板打分
            scores = {
                "低波动上涨": ret_z - vol_z,
                "高波动震荡": -abs(ret_z) + vol_z,
                "低波动阴跌": -ret_z - vol_z,
                "恐慌杀跌": -ret_z + vol_z,
            }
            best = max(scores.items(), key=lambda x: x[1])[0]
            name_map[s] = best

        return name_map

    def _state_stats(self, obs_df: pd.DataFrame, states: np.ndarray) -> pd.DataFrame:
        tmp = obs_df.copy()
        tmp["state"] = states.astype(int)
        g = (
            tmp.groupby("state", as_index=False)
            .agg(
                count=("state", "size"),
                ret_mean=("ret", "mean"),
                vol20_mean=("vol20", "mean"),
                vol_chg20_mean=("vol_chg20", "mean"),
            )
            .sort_values("state")
            .reset_index(drop=True)
        )
        return g

    @staticmethod
    def _clip_position(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    def set_inflection_signal_strength(self, strength: float) -> None:
        """外部可注入拐点信号强度（0-1），用于恐慌状态下仓位上调到30%。"""
        v = self._safe_float(strength)
        self.inflection_signal_strength = 0.0 if v is None else float(max(0.0, min(1.0, v)))

    def _save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "state_name_map": self.state_name_map,
            "feature_columns": self.feature_columns,
            "model_meta": self.model_meta,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(payload, f)

    def _load_model_if_needed(self) -> None:
        if self.model is not None and self.scaler is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)

        self.model = payload.get("model")
        self.scaler = payload.get("scaler")
        self.state_name_map = payload.get("state_name_map", {}) or {}
        self.feature_columns = payload.get("feature_columns", ["ret", "vol20", "vol_chg20"])
        self.model_meta = payload.get("model_meta", {}) or {}

        if self.model is None or self.scaler is None:
            raise RuntimeError("模型文件损坏：缺少 model/scaler")

    def fit(self, index_daily: pd.DataFrame, n_states: int = 4) -> Dict[str, Any]:
        """训练 HMM 并保存模型。"""
        n_states = int(n_states)
        if n_states < 2:
            raise ValueError("n_states 必须 >= 2")
        self._require_sklearn()

        obs_df = self._build_observation(index_daily)
        if len(obs_df) < 252 * 3:
            raise ValueError("建议至少 3 年日线数据（>=756 条有效观测）")

        X = obs_df[self.feature_columns].to_numpy(dtype=float)
        scaler = StandardScaler()  # type: ignore[operator]
        X_scaled = scaler.fit_transform(X)

        GaussianHMM = self._import_hmm()
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            random_state=42,
        )
        model.fit(X_scaled)
        states = model.predict(X_scaled)

        stats = self._state_stats(obs_df, states)
        state_name_map = self._map_state_name(stats)

        self.model = model
        self.scaler = scaler
        self.state_name_map = state_name_map
        self.model_meta = {
            "n_states": n_states,
            "train_start": obs_df["date"].iloc[0].strftime("%Y-%m-%d"),
            "train_end": obs_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "n_obs": int(len(obs_df)),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save_model()

        stats_out = stats.copy()
        stats_out["regime_name"] = stats_out["state"].map(lambda s: state_name_map.get(int(s), f"状态{int(s)}"))

        return {
            "ok": True,
            "model_path": str(self.model_path),
            "n_states": n_states,
            "n_obs": int(len(obs_df)),
            "state_stats": stats_out.to_dict(orient="records"),
            "meta": self.model_meta,
        }

    def _compute_transition_risk(self, current_state: int) -> float:
        if self.model is None:
            return 0.0

        trans = getattr(self.model, "transmat_", None)
        if trans is None:
            return 0.0

        row = np.asarray(trans[int(current_state)], dtype=float)
        cur_name = self.state_name_map.get(int(current_state), "高波动震荡")
        cur_sev = self.REGIME_SEVERITY.get(cur_name, 1)

        risk = 0.0
        for s2, p in enumerate(row.tolist()):
            n2 = self.state_name_map.get(int(s2), "高波动震荡")
            if self.REGIME_SEVERITY.get(n2, 1) > cur_sev:
                risk += float(p)
        return float(max(0.0, min(1.0, risk)))

    def _suggest_position(self, regime_name: str, probability: float, transition_risk: float) -> float:
        low, high = self.REGIME_POSITION_RANGE.get(regime_name, (40.0, 60.0))
        base = low + (high - low) * float(max(0.0, min(1.0, probability)))

        # 风险越高，仓位越保守
        adj = base * (1.0 - 0.45 * float(max(0.0, min(1.0, transition_risk))))
        suggested = self._clip_position(adj, low, high)

        # 恐慌杀跌 + 强拐点 -> 可上调到 30%
        if regime_name == "恐慌杀跌" and self.inflection_signal_strength >= 0.70:
            uplift = 20.0 + 10.0 * self.inflection_signal_strength
            suggested = self._clip_position(max(suggested, uplift), 0.0, 30.0)

        return float(round(suggested, 2))

    def predict_current_regime(self, index_daily: pd.DataFrame) -> Dict[str, Any]:
        """预测当前市场状态。"""
        self._load_model_if_needed()
        self._require_sklearn()
        assert self.model is not None and self.scaler is not None

        obs_df = self._build_observation(index_daily)
        X = obs_df[self.feature_columns].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)

        states = self.model.predict(X_scaled).astype(int)
        probs = self.model.predict_proba(X_scaled)

        cur_state = int(states[-1])
        cur_prob = float(probs[-1, cur_state])
        cur_name = self.state_name_map.get(cur_state, "高波动震荡")

        transition_risk = self._compute_transition_risk(cur_state)
        suggested_position = self._suggest_position(cur_name, cur_prob, transition_risk)

        hist_n = min(60, len(obs_df))
        hist_dates = obs_df["date"].tail(hist_n).dt.strftime("%Y-%m-%d").tolist()
        hist_states = states[-hist_n:].tolist()
        history = [{"date": d, "regime": int(s)} for d, s in zip(hist_dates, hist_states)]

        self._last_state_trace = [
            {
                "date": d,
                "regime_id": int(s),
                "regime_name": self.state_name_map.get(int(s), "高波动震荡"),
            }
            for d, s in zip(obs_df["date"].dt.strftime("%Y-%m-%d").tolist(), states.tolist())
        ]

        return {
            "regime_id": cur_state,
            "regime_name": cur_name,
            "probability": round(cur_prob, 4),
            "transition_risk": round(transition_risk, 4),
            "suggested_position_pct": suggested_position,
            "history": history,
        }

    def detect_regime_change(self, lookback: int = 5) -> Dict[str, Any]:
        """检测近 N 天是否发生状态切换。"""
        n = max(2, int(lookback))
        trace = self._last_state_trace
        if len(trace) < 2:
            return {"changed": False, "from": "", "to": "", "change_date": ""}

        window = trace[-n:]
        changed_idx = -1
        for i in range(1, len(window)):
            if int(window[i]["regime_id"]) != int(window[i - 1]["regime_id"]):
                changed_idx = i

        if changed_idx == -1:
            cur = window[-1]["regime_name"]
            return {"changed": False, "from": cur, "to": cur, "change_date": ""}

        from_name = str(window[changed_idx - 1]["regime_name"])
        to_name = str(window[changed_idx]["regime_name"])
        change_date = str(window[changed_idx]["date"])
        return {"changed": True, "from": from_name, "to": to_name, "change_date": change_date}

    @staticmethod
    def format_daily_header(prediction: Dict[str, Any]) -> str:
        """日报头部展示文案。"""
        regime_name = str(prediction.get("regime_name", "未知状态"))
        prob = float(prediction.get("probability", 0.0) or 0.0) * 100.0
        pos = float(prediction.get("suggested_position_pct", 0.0) or 0.0)
        risk = float(prediction.get("transition_risk", 0.0) or 0.0) * 100.0
        return f"🌡️ 市场状态：{regime_name}（概率 {prob:.0f}%）| 建议仓位 {pos:.0f}% | 转向风险 {risk:.0f}%"


__all__ = ["RegimeDetector"]
