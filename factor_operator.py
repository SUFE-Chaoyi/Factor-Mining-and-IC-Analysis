# features/factor_operator.py

"""
这个文件是工具箱。所有通用的数学计算、滚动窗口逻辑、状态机都放在这里。
支持 Series(单标的) 和 DataFrame(多标的) 输入。
"""
import numpy as np
import pandas as pd
import math

class FactorOperator:
    """
    通用算子集合。
    """

    @staticmethod
    def _int(x) -> int:
        """安全地将输入转换为整数（支持浮点数输入四舍五入）"""
        return int(round(float(x)))

    @staticmethod
    def _check_positive_int(T, name="T"):
        """判断参数 T 是否为正整数,name 为参数名"""
        w = FactorOperator._int(T)
        if w < 1:
            raise ValueError(f"{name} 需要为正整数 (>=1)，当前: {T!r}")
        return w

    @staticmethod
    def ensure_ts_index(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
        """数据清洗：设置时间索引，确保排序和唯一性"""
        if date_col == df.index.name:
            return df
        elif date_col in df.columns:
            out = df.copy()
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
            out = out.dropna(subset=[date_col]).sort_values(date_col)
            out = out.drop_duplicates(subset=[date_col], keep="last").set_index(date_col)
            return out
        else:
            raise KeyError(f"{date_col} not in index or cols")

    @staticmethod
    def spearman_corr(a: pd.Series, b: pd.Series) -> float:
        """计算序列 a,b 的 Spearman 相关系数。计算排名相关性"""
        aligned = pd.concat([a, b], axis=1).dropna()
        if len(aligned) < 3:
            return np.nan
        return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method='spearman')

    # ================= 基础/收益率计算 =================
    @staticmethod
    def compute_simple_returns(close: pd.Series, T: float = 1) -> pd.Series:
        """计算简单收益率"""
        w = FactorOperator._check_positive_int(T, "T")
        close = pd.to_numeric(close, errors="coerce")
        return close.pct_change(periods=w, fill_method=None)

    @staticmethod
    def compute_log_returns(close: pd.Series, T: float = 1) -> pd.Series:
        """计算对数收益率"""
        w = FactorOperator._check_positive_int(T, "T")
        close = pd.to_numeric(close, errors="coerce")
        close = close.where(close > 0)
        return np.log(close).diff(w)

    @staticmethod
    def log_return(px: pd.Series, h: float = 1) -> pd.Series:
        """固定周期对数收益 (支持浮点周期)"""
        w = FactorOperator._int(h)
        return np.log(px / px.shift(w))

    @staticmethod
    def realized_vol(px: pd.Series, v: float) -> pd.Series:
        """rolling std of daily log returns"""
        w = FactorOperator._int(v)
        r = np.log(px / px.shift(1))
        return r.rolling(w, min_periods=w).std()

    @staticmethod
    def zscore_h(px: pd.Series, h: float, v: float) -> pd.Series:
        """Z_t(h,v) = log(P_t/P_{t-h}) / (std(r,v)*sqrt(h))"""
        R = FactorOperator.log_return(px, h)
        vol_d = FactorOperator.realized_vol(px, v)
        denom = vol_d * np.sqrt(float(h))
        z = R / denom.replace(0.0, np.nan)
        return z

    @staticmethod
    def position_state(entry: pd.Series, exit_: pd.Series, max_hold: int = 0) -> pd.Series:
        """
        单标的状态机：
          s_t = 1 if (s_{t-1}=0 and entry_t) or (s_{t-1}=1 and not exit_t and not time_stop)
          否则 s_t = 0
        max_hold <= 0 表示不启用时间止损。
        """
        df = pd.concat([entry, exit_], axis=1)
        df.columns = ["entry", "exit"]
        df = df.fillna(False)

        s = np.zeros(len(df), dtype=int)
        hold_len = 0

        for i in range(len(df)):
            prev = 0 if i == 0 else s[i - 1]

            if prev == 0:
                if bool(df.iloc[i, 0]):
                    s[i] = 1
                    hold_len = 1
                else:
                    s[i] = 0
                    hold_len = 0
            else:
                hold_len += 1
                time_stop = (max_hold > 0) and (hold_len >= max_hold)
                if bool(df.iloc[i, 1]) or time_stop:
                    s[i] = 0
                    hold_len = 0
                else:
                    s[i] = 1

        return pd.Series(s, index=df.index)

    # ================= 序列/数学操作算子 =================
    @staticmethod
    def ts_minus(s1, s2):
        return s1 - s2

    @staticmethod
    def ts_div(s1, s2):
        return s1 / s2.replace(0, np.nan)

    @staticmethod
    def safe_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
        """log(x) with safe floor"""
        return np.log(x.clip(lower=float(eps)))

    @staticmethod
    def signed_power(x: pd.Series, a, eps: float = 1e-12) -> pd.Series:
        """SignedPower(x, a) = sign(x) * |x|^a （带 eps 防止 0^负数）"""
        xa = x.abs().clip(lower=float(eps))
        return np.sign(x) * (xa ** a)

    @staticmethod
    def scale(x: pd.Series, a: float = 1) -> pd.Series:
        """对 x 进行缩放，使得 sum(abs(x)) = a（默认 a = 1）"""
        x_abs = (x * x) ** 0.5
        k = a / x_abs.sum()
        return x * k

    @staticmethod
    def truple_operator(a: pd.Series, b: pd.Series, c: pd.Series) -> pd.Series:
        """三元运算符，a成立，返回b；a不成立，返回c。a是否成立默认为a>0"""
        if not (len(a) == len(b) == len(c)):
            raise ValueError(f"序列长度不一致")
        if not pd.api.types.is_numeric_dtype(a.dtypes):
            raise TypeError(f"序列a不是数值型")
        res = pd.Series(np.where(a > 0, b, c), index=a.index)
        return res

    # ================= 量价特定算子 =================
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """用 amount/vol 构造 vwap"""
        if "vwap" in df.columns:
            return df["vwap"].astype(float)
        amt = df["amount"].astype(float)
        vol = df["vol"].astype(float).replace(0.0, np.nan)
        return (amt / vol).astype(float)

    @staticmethod
    def adv(vol: pd.Series, n: float) -> pd.Series:
        """Average Daily Volume 滚动平均成交量"""
        w = FactorOperator._int(n)
        return vol.rolling(w, min_periods=w).mean()

    # ================= 滚动时间窗口算子 =================
    @staticmethod
    def delay(x: pd.Series, d: float):
        """d 天前 x 的值"""
        return x.shift(FactorOperator._int(d))

    @staticmethod
    def delta(x: pd.Series, d: float) -> pd.Series:
        """今天 x 的值减去 d 天前 x 的值"""
        w = FactorOperator._int(d)
        return x - x.shift(w)

    @staticmethod
    def ts_sum(x: pd.Series, n: float) -> pd.Series:
        """滚动求和"""
        w = FactorOperator._int(n)
        return x.rolling(w, min_periods=w).sum()

    @staticmethod
    def ts_ma(data: pd.DataFrame, window: float, min_periods: int | None = None) -> pd.DataFrame:
        """滚动均线"""
        w = FactorOperator._int(window)
        if min_periods is None:
            min_periods = w
        return data.rolling(window=w, min_periods=int(min_periods)).mean()

    @staticmethod
    def ts_std(data: pd.DataFrame | pd.Series, window: float) -> pd.DataFrame | pd.Series:
        """计算滚动标准差"""
        w = FactorOperator._check_positive_int(window, "window")
        return data.rolling(window=w, min_periods=w).std()

    @staticmethod
    def ts_max(x: pd.Series, d: float) -> pd.Series:
        """过去 d 天的时间序列最大值"""
        w = FactorOperator._check_positive_int(d, "d")
        return x.rolling(window=w, min_periods=w).max()

    @staticmethod
    def ts_min(x: pd.Series, d: float) -> pd.Series:
        """过去 d 天的时间序列最小值"""
        w = FactorOperator._check_positive_int(d, "d")
        return x.rolling(window=w, min_periods=w).min()

    @staticmethod
    def ts_argmax(x: pd.Series, d: float):
        """过去d天的最大值发生在距今哪一天"""
        w = FactorOperator._check_positive_int(d, "d")
        def calc_argmax(window: np.ndarray) -> float:
            if np.all(np.isnan(window)):
                return np.nan
            max_val = np.nanmax(window)
            max_indices = np.where(window == max_val)[0]
            latest_max_idx = max(max_indices)
            return (len(window) - 1) - latest_max_idx
        return x.rolling(window=w, min_periods=w).apply(calc_argmax, raw=True)

    @staticmethod
    def ts_argmin(x: pd.Series, d: float):
        """过去d天的最小值发生在距今哪一天"""
        w = FactorOperator._check_positive_int(d, "d")
        def calc_argmin(window: np.ndarray) -> float:
            if np.all(np.isnan(window)):
                return np.nan
            min_val = np.nanmin(window)
            min_indices = np.where(window == min_val)[0]
            latest_min_idx = max(min_indices)
            return (len(window) - 1) - latest_min_idx
        return x.rolling(window=w, min_periods=w).apply(calc_argmin, raw=True)

    @staticmethod
    def ts_corr(a: pd.Series, b: pd.Series, n: float) -> pd.Series:
        """滚动相关系数"""
        w = FactorOperator._int(n)
        return a.rolling(w, min_periods=w).corr(b)

    @staticmethod
    def covariance(x: pd.Series, y: pd.Series, d: float):
        """过去 d 天 x 和 y 的时间序列协方差 (返回最后一日标量, 若需序列请用 rolling.cov)"""
        w = FactorOperator._int(d)
        x_d, y_d = x.iloc[-w:], y.iloc[-w:]
        x_bar, y_bar = x_d.mean(), y_d.mean()
        xy_sum = (x_d * y_d).sum()
        res = xy_sum - w * x_bar * y_bar
        return res * (1 / (w - 1)) if w > 1 else res

    @staticmethod
    def correlation(x: pd.Series, y: pd.Series, d: float):
        """过去 d 天 x 和 y 的相关性 (返回最后一日标量)"""
        w = FactorOperator._int(d)
        x_d, y_d = x.iloc[-w:], y.iloc[-w:]
        x_bar, y_bar = x_d.mean(), y_d.mean()
        s_x = (1 / (w - 1) * ((x_d - x_bar) * (x_d - x_bar)).sum())
        s_y = (1 / (w - 1) * ((y_d - y_bar) * (y_d - y_bar)).sum())
        fenmu = math.sqrt(s_x * s_y)
        res = FactorOperator.covariance(x, y, w) / fenmu
        return res

    @staticmethod
    def ts_slope(data: pd.DataFrame | pd.Series, window: float) -> pd.DataFrame | pd.Series:
        """计算滚动线性回归斜率 (Slope)"""
        w = FactorOperator._check_positive_int(window, "window")
        def _calc_slope_1d(y):
            if np.any(~np.isfinite(y)):
                return np.nan
            x = np.arange(len(y), dtype=float)
            return np.polyfit(x, y, 1)[0]
        return data.rolling(window=w, min_periods=w).apply(_calc_slope_1d, raw=True)

    @staticmethod
    def decay_linear(x: pd.Series, d: float) -> pd.Series:
        """线性衰减加权均值"""
        w = FactorOperator._int(d)
        if w < 1:
            raise ValueError(f"窗口长度必须大于等于1（当前w={w}）")
        weights = np.arange(1, w + 1, dtype=float)
        weights /= weights.sum()

        def _wma(arr: np.ndarray) -> float:
            if np.any(~np.isfinite(arr)):
                return np.nan
            return float(np.dot(arr, weights))

        return x.rolling(w, min_periods=w).apply(_wma, raw=True)

    @staticmethod
    def ts_rank(x: pd.Series, n: float) -> pd.Series:
        """Ts_Rank：窗口内最后一个值的分位排名（0~1）"""
        w = FactorOperator._int(n)
        def _rank_last(arr: np.ndarray) -> float:
            if np.any(~np.isfinite(arr)):
                return np.nan
            s = pd.Series(arr)
            return float(s.rank(pct=True).iloc[-1])
        return x.rolling(w, min_periods=w).apply(_rank_last, raw=True)

    @staticmethod
    def ts_product(x: pd.Series, n: float) -> pd.Series:
        """过去 n 天滚动连乘"""
        w = FactorOperator._int(n)
        def _prod(arr: np.ndarray) -> float:
            if np.any(~np.isfinite(arr)):
                return np.nan
            return float(np.prod(arr))
        return x.rolling(w, min_periods=w).apply(_prod, raw=True)

    @staticmethod
    def ts_ma_at(data: pd.DataFrame, t: pd.Timestamp, window: float = 5) -> pd.Series:
        """计算“截止到 t（包含 t）”过去 window 天的均值（返回横截面 Series）"""
        w = FactorOperator._check_positive_int(window, "window")
        sub = data.loc[:t].tail(w)
        return sub.mean(axis=0)