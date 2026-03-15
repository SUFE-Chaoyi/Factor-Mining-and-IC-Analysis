import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Tuple
import re

DEFAULT_CALENDAR_PATH = "/Users/chaoyi/Documents/factor_lab/pre_work/A_share_trade_calendar_2024.csv"

@lru_cache(maxsize=3)
def _load_trade_calendar(calendar_csv: str, col: str = "trade_date") -> pd.DatetimeIndex:
    path = Path(calendar_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Calendar file not found: {path}")

    df = pd.read_csv(path, usecols=[col], parse_dates=[col])
    idx = pd.DatetimeIndex(df[col]).normalize().sort_values().unique()
    return idx

def _parse_date_exist_or_adjust(x: pd.Timestamp | str, *, direction: str) -> pd.Timestamp:
    """
    把输入解析为“存在的自然日”：
    - direction="forward": 不存在日期则向后顺延到存在日期
    - direction="backward": 不存在日期则向前回退到存在日期

    仅对形如 YYYY-MM-DD（可带时间部分，但只取日期）的字符串做“纠正”。
    其他格式解析失败会直接报错。
    """
    if isinstance(x, pd.Timestamp):
        return x

    s = str(x).strip()

    # 先尝试正常解析（存在日期会直接成功）
    try:
        return pd.to_datetime(s, errors="raise")
    except Exception:
        pass

    # 尝试解析 YYYY-MM-DD（允许后面带时间：YYYY-MM-DD HH:MM:SS）
    m = re.match(r"^\s*(\d{4})-(\d{2})-(\d{2})(?:\s+.*)?\s*$", s)
    if not m:
        raise ValueError(f"无法解析日期字符串: {s!r}，请使用 'YYYY-MM-DD' 格式")

    y, mo, d = map(int, m.groups())

    if not (1 <= mo <= 12):
        raise ValueError(f"月份不合法: {mo} (from {s!r})")

    first = pd.Timestamp(y, mo, 1)
    last = (first + pd.offsets.MonthEnd(0))  # 当月最后一天（存在）

    # day 在合法范围：直接返回
    if 1 <= d <= last.day:
        return pd.Timestamp(y, mo, d)

    # day 太大：不存在日期（如 2/31）
    if d > last.day:
        if direction == "backward":
            # 向前：钳到当月最后一天（如 2/31 -> 2/29）
            return last
        elif direction == "forward":
            # 向后：按溢出天数顺延（如 2/31, 2024-02 最后一天 2/29，溢出2天 -> 3/2）
            overflow = d - last.day
            return last + pd.Timedelta(days=overflow)
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

    # day 太小（例如 day=0），极少见：按方向处理
    if d < 1:
        if direction == "forward":
            return first  # 向后就到当月1号
        elif direction == "backward":
            return first + pd.Timedelta(days=d - 1)  # day=0 -> 前一天
        else:
            raise ValueError("direction must be 'forward' or 'backward'")

    raise RuntimeError("unreachable")

def valid_trading_date(
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    calendar_csv: str = DEFAULT_CALENDAR_PATH
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    日期对齐逻辑：
    0) 先把“不存在的自然日”修正为存在自然日：
       - start: 向后修正
       - end  : 向前修正
    1) Start 向后找交易日（>= start）
    2) End   向前找交易日（<= end）
    """
    cal = _load_trade_calendar(calendar_csv)
    if len(cal) == 0:
        raise ValueError("Trading calendar is empty.")

    # 0) 先处理“日期不存在”的情况
    s = _parse_date_exist_or_adjust(start_date, direction="forward")
    e = _parse_date_exist_or_adjust(end_date, direction="backward")

    # tz-aware -> tz-naive
    if getattr(s, "tzinfo", None) is not None:
        s = s.tz_convert(None)
    if getattr(e, "tzinfo", None) is not None:
        e = e.tz_convert(None)

    s = s.normalize()
    e = e.normalize()

    if s > e:
        s, e = e, s

    # 1) Start Date 对齐 (向后找 >= s)
    si = cal.searchsorted(s, side="left")
    if si >= len(cal):
        raise ValueError(f"Start date {s.date()} is after the last trading day.")
    valid_s = cal[si]

    # 2) End Date 对齐 (向前找 <= e)
    ei = cal.searchsorted(e, side="right") - 1
    if ei < 0:
        raise ValueError(f"End date {e.date()} is before the first trading day.")
    valid_e = cal[ei]

    # 3) 检查调整后的有效性
    if valid_s > valid_e:
        raise ValueError(f"No trading days exist between {s.date()} and {e.date()}.")

    return valid_s.date(), valid_e.date()

# 示例：2月不存在31日
# vs, ve = valid_trading_date("2020-01-06", "2022-02-31")
# print(vs, ve)