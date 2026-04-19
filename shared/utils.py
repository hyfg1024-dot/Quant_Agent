from __future__ import annotations

import random
import re
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

_NUM_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def to_float(value: Any) -> Optional[float]:
    """Convert mixed numeric input into float (default unit: 亿 when chinese unit exists)."""
    if value is None:
        return None

    if isinstance(value, (int, float, np.number)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return float(value)

    text = str(value).strip()
    if text in {"", "-", "--", "None", "nan", "NaN", "False"}:
        return None

    text = text.replace(",", "").replace("%", "")
    text = text.replace("亿元", "亿").replace("万元", "万")

    unit_mul = 1.0
    if text.endswith("万亿"):
        unit_mul = 10_000.0
        text = text[:-2]
    elif text.endswith("亿"):
        unit_mul = 1.0
        text = text[:-1]
    elif text.endswith("万"):
        unit_mul = 0.0001
        text = text[:-1]

    m = _NUM_PATTERN.search(text)
    if not m:
        return None

    try:
        return float(m.group(0)) * unit_mul
    except Exception:
        return None


def to_int(value: Any) -> int:
    f = to_float(value)
    if f is None:
        return 0
    return int(round(f))


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_flow_to_yi(value: Any) -> Optional[float]:
    """Normalize fund flow to unit '亿'."""
    raw = to_float(value)
    if raw is None:
        return None

    abs_v = abs(raw)
    if abs_v >= 1_000_000:
        return raw / 100_000_000.0
    if 10_000 <= abs_v < 1_000_000:
        return raw / 10_000.0
    return raw


def coalesce_number(*values: Any, default: Optional[float] = None) -> Optional[float]:
    for value in values:
        parsed = to_float(value)
        if parsed is not None:
            return parsed
    return to_float(default) if default is not None else None


def format_num(value: Any, digits: int = 2, suffix: str = "") -> str:
    v = to_float(value)
    if v is None:
        return f"--{suffix}"
    return f"{v:.{int(digits)}f}{suffix}"


def format_pct(value: Any, digits: int = 2) -> str:
    v = to_float(value)
    if v is None:
        return "--"
    return f"{v:.{int(digits)}f}%"


def retry_call(func, max_retries: int = 3):
    retries = max(1, int(max_retries))
    for i in range(retries):
        try:
            return func()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep((2**i) + random.uniform(0.1, 0.6))
    return None
