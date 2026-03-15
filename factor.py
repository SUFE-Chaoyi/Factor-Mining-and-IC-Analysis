# features/factor.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import sys
features_folder = os.path.dirname(os.path.abspath(__file__))
# 把features文件夹加入搜索路径
sys.path.append(features_folder)

from factor_operator import FactorOperator as op
from typing import Literal

# 定义价格字段类型 下面函数输入设计 price:PriceField 时，参数智能填这几个
PriceField = Literal["open", "close", "high", "low", "vol", "amount"]


# ===========================================
@dataclass
# 父类-具体因子模板
class FactorBase:
    """
    单标的时序因子基类
    """
    df: pd.DataFrame       # 输入的原始数据。至少包含: ["open", "close", "high", "low", "vol", "amount"]
    symbol: str            # 股票代码
    shift: int = 0         # delay shift
    date_col: str = "trade_date"

    def __post_init__(self) -> None:
        self.df = op.ensure_ts_index(self.df, self.date_col)

    def code(self) -> str:
        """返回股票代码"""
        return self.symbol
    
    @property
    def factor_name(self) -> str:
        """因子名称，子类可覆盖"""
        return self.__class__.__name__
    
    @property
    def my_type(self) -> str:
        """因子类型 ts择时因子 cs截面因子"""
        return self.factor_type
    
    # -----------------------------------------------------
    # 核心：公式 (由子类实现)
    # -----------------------------------------------------
    def formula(self) -> pd.Series:
        """
        子类必须实现此函数。
        应当直接使用 self.df 中的列进行计算。
        返回: pd.Series (Index为时间)
        """
        raise NotImplementedError("子类必须实现 formula 方法")

    # -----------------------------------------------------
    # 功能函数
    # -----------------------------------------------------
    def score(self) -> pd.Series:
        """计算因子得分序列。"""
        fac = self.formula()
        
        if self.shift != 0:
            fac = fac.shift(self.shift)
        
        fac.name = self.factor_name
        self.df[fac.name] = fac
        return fac

    def ts_signal(self, epsilon: float = 0.0) -> pd.Series:
        """
        生成择时信号
        Factor > epsilon => 1
        Factor <= epsilon => 0
        """
        fac = self.score().dropna() 
        sig = (fac > float(epsilon)).astype(int)
        sig.name = f"signal_{self.factor_name}"
        return sig

    def ts_IC(self, hold_day: int = 1, price_col: PriceField = "close") -> float:
        """计算时序 IC (Information Coefficient)"""
        if hold_day <= 0:
            raise ValueError("hold_day must be > 0")

        if price_col not in self.df.columns:
            raise ValueError(f"Price column '{price_col}' not in dataframe")
        
        px = self.df[price_col]
        fwd_ret = px.shift(-hold_day) / px - 1.0
        fwd_ret.name = "fwd_ret"

        sig = self.score()
        return op.spearman_corr(sig, fwd_ret)

# =================================================

@dataclass
class Factor1(FactorBase):
    t1, t2, t3 = 10,20,5
    
    @property
    def my_type(self) -> str:
        return f"cs"
    
    @property
    def factor_name(self) -> str:
        return f"Factor1_slope_diff_vol_{self.t1}_{self.t2}_{self.t3}"
    
    def formula(self) -> pd.Series:
        slope1, slope2 = op.ts_slope(self.df["close"],self.t1), op.ts_slope(self.df["close"],self.t2)
        fenzi = op.ts_minus(slope1, slope2)
        fenmu = op.ts_std(self.df["close"], self.t3)
        return (-1) * op.ts_div(fenzi, fenmu)
    
@dataclass
class Factor2(FactorBase):
    t1, t2, t3 = 10,20,15
    
    @property
    def my_type(self):
        return "cs"
    
    @property
    def factor_name(self) -> str:
        return f"Factor2_beta_diff_root_vol_{self.t1}_{self.t2}_{self.t3}"
    
    def formula(self) -> pd.Series:
        beta1,beta2 = op.ts_slope(self.df["close"],self.t1),op.ts_slope(self.df["close"],self.t2)
        vol_t3 = op.ts_std(self.df["close"],self.t3)
        return op.ts_div( op.ts_minus(beta1,beta2) , vol_t3)

@dataclass
class Factor3(FactorBase):
    t1 = 20
    t2 = 2
    t3 = 5

    @property
    def my_type(self):
        return "cs"
    
    @property
    def factor_name(self) -> str:
        return f"Factor3:returns_std_{self.t1}"
    
    def formula(self) -> pd.Series:
        simple_return_20 = op.compute_simple_returns(self.df["close"], self.t1)
        return_20_std = op.ts_std(simple_return_20, self.t1)
        daily_return = op.compute_simple_returns(self.df["close"])
        
        res = op.truple_operator(daily_return, return_20_std, self.df["close"])
        res = res ** self.t3
        score = op.ts_argmax(res, self.t3) - 0.5
        
        return score

# -------------------------
# Factor3_1
# -------------------------
@dataclass
class Factor3_1(FactorBase):
    """
    优化1：修正原始实现中幂次/argmax窗口混用的问题，保留原始逻辑。
    """
    std_n: int = 20
    power_n: int = 2
    argmax_n: int = 5
    rank_n: int = 20
    eps: float = 1e-6

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor3_1"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        ret = op.compute_simple_returns(close)
        ret_std = op.ts_std(ret, self.std_n)

        # ret<0 用波动率，否则用 close，和原始思路一致
        core = op.truple_operator(ret, ret_std, close)
        core = op.signed_power(core, self.power_n, eps=self.eps)
        return op.ts_argmax(core, self.argmax_n) - 0.5


# -------------------------
# Factor3_2
# -------------------------
@dataclass
class Factor3_2(FactorBase):
    """
    优化2：做时间序列中间映射，减少极端值对排序的干扰。
    """
    std_n: int = 20
    rank_n: int = 60

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor3_2"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        ret = op.compute_simple_returns(close)
        ret_std = op.ts_std(ret, self.std_n)

        core = op.truple_operator(ret, ret_std, close)
        r = op.ts_rank(core, self.rank_n)
        return -(r - 0.5).abs()
    
@dataclass
class Factor4(FactorBase):
    h: int = 20
    v: int = 20
    theta_in: float = 2.0
    theta_out: float = 0.0
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"
    
    @property
    def factor_name(self) -> str:
        return f"Factor4_mr_fixedh_sigma_h{self.h}_v{self.v}_in{self.theta_in}_out{self.theta_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        z = op.zscore_h(px, self.h, self.v)

        entry = z <= -float(self.theta_in)
        exit_ = z >= -float(self.theta_out)

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))


@dataclass
class Factor5(FactorBase):
    h: int = 20
    v: int = 20
    theta_in: float = 2.0
    theta_out: float = 1.0
    max_hold: int = 0
    
    @property
    def my_type(self):
        return "ts"
    
    @property
    def factor_name(self) -> str:
        return f"Factor5_u_fixedh_sigma_h{self.h}_v{self.v}_in{self.theta_in}_out{self.theta_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        z = op.zscore_h(px, self.h, self.v)

        entry = z.abs() >= float(self.theta_in)
        exit_ = z.abs() <= float(self.theta_out)

        pos = op.position_state(entry, exit_, max_hold=int(self.max_hold))
        return (-1) * pos


@dataclass
class Factor6(FactorBase):
    h: int = 20
    v: int = 20
    N: int = 252
    q_in: float = 0.1
    q_out: float = 0.3
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"
    
    @property
    def factor_name(self) -> str:
        return f"Factor6_u_fixedh_qtl_h{self.h}_v{self.v}_N{self.N}_qin{self.q_in}_qout{self.q_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        z = op.zscore_h(px, self.h, self.v)

        z_hist = z.shift(1)
        low_in = z_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(float(self.q_in))
        high_in = z_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(1.0 - float(self.q_in))

        low_out = z_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(float(self.q_out))
        high_out = z_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(1.0 - float(self.q_out))

        entry = (z <= low_in) | (z >= high_in)
        exit_ = (z > low_out) & (z < high_out)

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))


@dataclass
class Factor7(FactorBase):
    h_min: int = 5
    h_max: int = 60
    v: int = 20
    theta_in: float = 2.0
    theta_out: float = 0.0
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"

    @property
    def factor_name(self) -> str:
        return f"Factor7_mr_varlen_sigma_h{self.h_min}-{self.h_max}_v{self.v}_in{self.theta_in}_out{self.theta_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        vol_d = op.realized_vol(px, int(self.v))

        zs = []
        for l in range(int(self.h_min), int(self.h_max) + 1):
            R = op.log_return(px, l)
            denom = vol_d * np.sqrt(float(l))
            z_l = R / denom.replace(0.0, np.nan)
            zs.append(z_l.rename(f"z{l}"))

        z_mat = pd.concat(zs, axis=1)
        z_min = z_mat.min(axis=1)

        entry = z_min <= -float(self.theta_in)
        exit_ = z_min >= -float(self.theta_out)

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))


@dataclass
class Factor8(FactorBase):
    h_min: int = 5
    h_max: int = 60
    v: int = 20
    N: int = 252
    q_in: float = 0.1
    q_out: float = 0.3
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"

    @property
    def factor_name(self) -> str:
        return f"Factor8_mr_varlen_qtl_h{self.h_min}-{self.h_max}_v{self.v}_N{self.N}_qin{self.q_in}_qout{self.q_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        vol_d = op.realized_vol(px, int(self.v))

        zs = []
        for l in range(int(self.h_min), int(self.h_max) + 1):
            R = op.log_return(px, l)
            denom = vol_d * np.sqrt(float(l))
            z_l = R / denom.replace(0.0, np.nan)
            zs.append(z_l)

        z_min = pd.concat(zs, axis=1).min(axis=1)

        y_hist = z_min.shift(1)
        thr_in = y_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(float(self.q_in))
        thr_out = y_hist.rolling(int(self.N), min_periods=int(self.N)).quantile(float(self.q_out))

        entry = z_min <= thr_in
        exit_ = z_min >= thr_out

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))


@dataclass
class Factor9(FactorBase):
    h: int = 20
    T: int = 60
    v: int = 20
    theta_in: float = 2.0
    theta_out: float = 0.0
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"

    @property
    def factor_name(self) -> str:
        return f"Factor9_mr_varstart_sigma_h{self.h}_T{self.T}_v{self.v}_in{self.theta_in}_out{self.theta_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        z_now = op.zscore_h(px, int(self.h), int(self.v))

        R_h = op.log_return(px, int(self.h))
        R_worst = R_h.rolling(int(self.T), min_periods=int(self.T)).min()

        vol_d = op.realized_vol(px, int(self.v))
        z_worst = R_worst / (vol_d * np.sqrt(float(self.h))).replace(0.0, np.nan)

        entry = z_worst <= -float(self.theta_in)
        exit_ = z_now >= -float(self.theta_out)

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))


@dataclass
class Factor10(FactorBase):
    L: int = 120
    k: int = 5
    S: int = 20
    d_in: float = 1.0
    d_out: float = 0.0
    max_hold: int = 0

    @property
    def my_type(self):
        return "ts"
    
    @property
    def factor_name(self) -> str:
        return f"Factor10_momfilter_pullback_L{self.L}_k{self.k}_S{self.S}_din{self.d_in}_dout{self.d_out}_mh{self.max_hold}"

    def formula(self) -> pd.Series:
        px = self.df["close"].astype(float)
        logp = np.log(px)

        M = np.log(px.shift(int(self.k)) / px.shift(int(self.L) + int(self.k)))
        ma = logp.rolling(int(self.S), min_periods=int(self.S)).mean()
        sd = logp.rolling(int(self.S), min_periods=int(self.S)).std()
        D = (logp - ma) / sd.replace(0.0, np.nan)

        entry = (M > 0.0) & (D <= -float(self.d_in))
        exit_ = (D >= -float(self.d_out)) | (M <= 0.0)

        return op.position_state(entry, exit_, max_hold=int(self.max_hold))
    

@dataclass
class Factor11(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor11"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        low = self.df["low"].astype(float)
        high = self.df["high"].astype(float)
        vol = self.df["vol"].astype(float)

        denom = (high - low).replace(0.0, np.nan)
        clv = ((close - low) - (high - close)) / denom
        a = clv * vol
        b = op.ts_argmax(close, 10)

        return -(2.0 * a - b)


@dataclass
class Factor12(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor12"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)
        adv180 = op.adv(vol, 180.0)

        left = vwap - op.ts_min(vwap, 16.1219)
        right = op.ts_corr(vwap, adv180, 17.9282)

        return right - left


@dataclass
class Factor13(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor13"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        adv20 = op.adv(vol, 20.0)
        sum_adv = op.ts_sum(adv20, 22.4101)
        a = op.ts_corr(vwap, sum_adv, 9.91009)

        b_left = 2.0 * open_
        b_right = ((high + low) / 2.0) + high
        b = (b_left < b_right).astype(float)

        return -(b - a)


@dataclass
class Factor14(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor14"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        low = self.df["low"].astype(float)
        high = self.df["high"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        mix1 = (open_ * 0.178404) + (low * (1.0 - 0.178404))
        s1 = op.ts_sum(mix1, 12.7054)

        adv120 = op.adv(vol, 120.0)
        s2 = op.ts_sum(adv120, 12.7054)

        left = op.ts_corr(s1, s2, 16.6208)

        mix2 = (((high + low) / 2.0) * 0.178404) + (vwap * (1.0 - 0.178404))
        right = op.delta(mix2, 3.69741)

        return -(right - left)


@dataclass
class Factor15(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor15"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv60 = op.adv(vol, 60.0)
        s = op.ts_sum(adv60, 8.6911)

        mix = (open_ * 0.00817205) + (vwap * (1.0 - 0.00817205))
        left = op.ts_corr(mix, s, 6.40374)

        right = open_ - op.ts_min(open_, 13.635)

        return -(right - left)


@dataclass
class Factor16(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor16"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        low = self.df["low"].astype(float)
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)

        a = op.decay_linear(op.delta(vwap, 3.51013), 7.23052)

        denom = (open_ - (high + low) / 2.0).replace(0.0, np.nan)
        frac = (low - vwap) / denom
        b = op.ts_rank(op.decay_linear(frac, 11.4157), 6.72611)

        return -(a + b)


@dataclass
class Factor17(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor17"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        close = self.df["close"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        adv15 = op.adv(vol, 15.0)

        c = op.ts_corr(high, adv15, 8.91644)
        left = op.ts_rank(c, 13.9333)

        mix = (close * 0.518371) + (low * (1.0 - 0.518371))
        right = op.delta(mix, 1.06157)

        return -(right - left)


@dataclass
class Factor18(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor18"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        low = self.df["low"].astype(float)
        open_ = self.df["open"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv180 = op.adv(vol, 180.0)

        a1 = op.ts_rank(close, 3.43976)
        a2 = op.ts_rank(adv180, 12.0647)
        c = op.ts_corr(a1, a2, 18.0175)
        left = op.ts_rank(op.decay_linear(c, 4.20501), 15.6948)

        expr = (low + open_) - (2.0 * vwap)
        right_in = expr ** 2.0
        right = op.ts_rank(op.decay_linear(right_in, 16.4662), 4.4388)

        return pd.concat([left, right], axis=1).max(axis=1)


@dataclass
class Factor19(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor19"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv40 = op.adv(vol, 40.0)
        num = op.decay_linear(op.ts_corr((high + low) / 2.0, adv40, 8.93345), 10.1519)

        a = op.ts_rank(vwap, 3.72469)
        b = op.ts_rank(vol, 18.5188)
        den = op.decay_linear(op.ts_corr(a, b, 6.86671), 2.95011).replace(0.0, np.nan)

        return num / den


@dataclass
class Factor20(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor20"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        open_ = self.df["open"].astype(float)
        low = self.df["low"].astype(float)

        left = op.decay_linear(op.delta(vwap, 4.72775), 2.91864)

        mix = (open_ * 0.147155) + (low * (1.0 - 0.147155))
        frac = op.delta(mix, 2.03608) / mix.replace(0.0, np.nan)
        inner = frac * -1.0
        right = op.ts_rank(op.decay_linear(op.delta(inner, 3.33829), 16.7411), 16.7411)

        m = pd.concat([left, right], axis=1).max(axis=1)
        return -m


@dataclass
class Factor21(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor21"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        high = self.df["high"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv30 = op.adv(vol, 30.0)
        s = op.ts_sum(adv30, 37.4843)
        left = op.ts_corr(close, s, 15.1365)

        mix = (high * 0.0261661) + (vwap * (1.0 - 0.0261661))
        right = op.ts_corr(mix, vol, 11.4791)

        return -(right - left)


@dataclass
class Factor22(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor23"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)
        low = self.df["low"].astype(float)

        left = op.ts_corr(vwap, vol, 4.24304)

        adv50 = op.adv(vol, 50.0)
        right = op.ts_corr(low, adv50, 12.4413)

        return right - left
    

@dataclass
class Factor23(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor23_{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)
        vol = self.df["vol"].astype(float)

        adv10 = op.adv(vol, 10.0)

        mix = (open_ * 0.868128) + (high * (1.0 - 0.868128))
        d = op.delta(mix, 4.04545)

        base01 = (np.sign(d) + 1.0) / 2.0
        base = op.ts_rank(base01, float(self.rank_n)).clip(lower=self.eps)

        corr = op.ts_corr(high, adv10, 5.11456)
        expo = op.ts_rank(corr, 5.53756)

        return -(base ** expo)


@dataclass
class Factor24(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha81_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv10 = op.adv(vol, 10.0)
        s = op.ts_sum(adv10, 49.6054)
        c1 = op.ts_corr(vwap, s, 8.47743)

        r1 = op.ts_rank(c1, float(self.rank_n))
        r2 = op.ts_rank(r1 ** 4, float(self.rank_n))

        prod = op.ts_product(r2.clip(lower=self.eps), 14.9655)
        left = op.ts_rank(op.safe_log(prod, eps=self.eps), float(self.rank_n))

        rvwap = op.ts_rank(vwap, float(self.rank_n))
        rvol = op.ts_rank(vol, float(self.rank_n))
        c2 = op.ts_corr(rvwap, rvol, 5.07914)
        right = op.ts_rank(c2, float(self.rank_n))

        return left - right

# -------------------------
# Factor24_1
# -------------------------
@dataclass
class Factor24_1(FactorBase):
    """
    优化1：降低左侧幂次放大，增加 corr 截尾。
    """
    rank_n: int = 120
    eps: float = 1e-6
    power_n: int = 2

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor24_1"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv10 = op.adv(vol, 10)
        s = op.ts_sum(adv10, 50)
        c1 = op.ts_corr(vwap, s, 8).clip(-0.95, 0.95)

        r1 = op.ts_rank(c1, self.rank_n)
        r2 = op.ts_rank(r1 ** self.power_n, self.rank_n)

        prod = op.ts_product(r2.clip(lower=self.eps), 15)
        left = op.ts_rank(op.log(prod, eps=self.eps), self.rank_n)

        rvwap = op.ts_rank(vwap, self.rank_n)
        rvol = op.ts_rank(vol, self.rank_n)
        c2 = op.ts_corr(rvwap, rvol, 5).clip(-0.95, 0.95)
        right = op.ts_rank(c2, self.rank_n)

        return left - right


# -------------------------
# Factor24_2    !!!!!
# -------------------------
@dataclass
class Factor24_2(FactorBase):
    """
    优化2：对 left-right 做时间序列甜蜜区映射。
    问题：变换之后全0
    """
    rank_n: int = 120
    eps: float = 1e-6
    power_n: int = 2

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor24_2"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        adv10 = op.adv(vol, 10)
        s = op.ts_sum(adv10, 50)
        c1 = op.ts_corr(vwap, s, 8).clip(-0.95, 0.95)

        r1 = op.ts_rank(c1, self.rank_n)
        r2 = op.ts_rank(r1 ** self.power_n, self.rank_n)

        prod = op.ts_product(r2.clip(lower=self.eps), 15)
        left = op.ts_rank(op.log(prod, eps=self.eps), self.rank_n)

        rvwap = op.ts_rank(vwap, self.rank_n)
        rvol = op.ts_rank(vol, self.rank_n)
        c2 = op.ts_corr(rvwap, rvol, 5).clip(-0.95, 0.95)
        right = op.ts_rank(c2, self.rank_n)

        core = left - right
        r = op.ts_rank(core, self.rank_n)
        return -(r - 0.5).abs()
    
@dataclass
class Factor25(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor25{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        vol = self.df["vol"].astype(float)

        x1 = op.decay_linear(op.delta(open_, 1.46063), 14.8717)
        a = op.ts_rank(x1, float(self.rank_n))

        c = op.ts_corr(vol, open_, 17.4842)
        x2 = op.decay_linear(c, 6.92131)
        b = op.ts_rank(x2, 13.4283)

        return -np.minimum(a, b)


@dataclass
class Factor26(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha83_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5.0) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)

        a = op.ts_rank(range_norm.shift(2), float(self.rank_n))
        b1 = op.ts_rank(vol, float(self.rank_n))
        b = op.ts_rank(b1, float(self.rank_n))

        num = a * b

        spread = (vwap - close)
        return num * spread / range_norm.replace(0.0, np.nan)

# -------------------------
# Factor26_1
# -------------------------
@dataclass
class Factor26_1(FactorBase):
    """
    优化1：给 range_norm 加地板，并对 spread 做截尾。
    """
    rank_n: int = 120
    floor_k: float = 1e-3
    spread_clip: float = 0.1

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor26_1"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)
        floor = ma5.abs() * self.floor_k
        range_norm = range_norm.where(range_norm > floor, floor)

        a = op.ts_rank(range_norm.shift(2), self.rank_n)
        b = op.ts_rank(op.ts_rank(vol, self.rank_n), self.rank_n)

        spread = ((vwap - close) / ma5.replace(0.0, np.nan)).clip(-self.spread_clip, self.spread_clip)
        return a * b * spread / range_norm


# -------------------------
# Factor26_2
# -------------------------
@dataclass
class Factor26_2(FactorBase):
    """
    优化2：把原始乘法结构改为更稳的“排名融合 + 中间映射”。
    """
    rank_n: int = 120
    floor_k: float = 1e-3

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor26_2"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)
        floor = ma5.abs() * self.floor_k
        range_norm = range_norm.where(range_norm > floor, floor)

        a = op.ts_rank(range_norm.shift(2), self.rank_n)
        b = op.ts_rank(op.ts_rank(vol, self.rank_n), self.rank_n)
        spread_rank = op.ts_rank((vwap - close) / range_norm.replace(0.0, np.nan), self.rank_n)

        core = 0.6 * (a * b) + 0.4 * spread_rank
        r = op.ts_rank(core, self.rank_n)
        return -(r - 0.5).abs()
    
@dataclass
class Factor27(FactorBase):
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor27"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        close = self.df["close"].astype(float)

        x = vwap - op.ts_max(vwap, 15.3217)
        base = op.ts_rank(x, 20.7127).clip(lower=self.eps)

        expo = op.delta(close, 4.96796)
        return op.signed_power(base, expo, eps=self.eps)


@dataclass
class Factor29(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha86_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv20 = op.adv(vol, 20.0)
        s = op.ts_sum(adv20, 14.7444)
        c = op.ts_corr(close, s, 6.00049)
        left = op.ts_rank(c, 20.4195)

        right = op.ts_rank(close - vwap, float(self.rank_n))

        return left - right


@dataclass
class Factor30(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha87_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        mix = (close * 0.369701) + (vwap * (1.0 - 0.369701))
        x1 = op.decay_linear(op.delta(mix, 1.91233), 2.65461)
        a = op.ts_rank(x1, float(self.rank_n))

        adv81 = op.adv(vol, 81.0)
        c = op.ts_corr(adv81, close, 13.4132)
        x2 = op.decay_linear(c.abs(), 4.89768)
        b = op.ts_rank(x2, 14.4535)

        return -np.maximum(a, b)


@dataclass
class Factor31(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha88_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ro = op.ts_rank(open_, float(self.rank_n))
        rl = op.ts_rank(low, float(self.rank_n))
        rh = op.ts_rank(high, float(self.rank_n))
        rc = op.ts_rank(close, float(self.rank_n))

        candle = (ro + rl) - (rh + rc)
        x1 = op.decay_linear(candle, 8.06882)
        a = op.ts_rank(x1, float(self.rank_n))

        adv60 = op.adv(vol, 60.0)
        c = op.ts_corr(op.ts_rank(close, 8.44728), op.ts_rank(adv60, 20.6966), 8.01266)
        x2 = op.decay_linear(c, 6.65053)
        b = op.ts_rank(x2, 2.61957)

        return np.minimum(a, b)


@dataclass
class Factor32(FactorBase):
    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "WQ_Alpha89_tsproxy"

    def formula(self) -> pd.Series:
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv10 = op.adv(vol, 10.0)
        c1 = op.ts_corr(low, adv10, 6.94279)
        x1 = op.decay_linear(c1, 5.51607)
        t1 = op.ts_rank(x1, 3.79744)

        x2 = op.decay_linear(op.delta(vwap, 3.48158), 10.1466)
        t2 = op.ts_rank(x2, 15.3012)

        return t1 - t2


@dataclass
class Factor33(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor33_{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        base_raw = close - op.ts_max(close, 4.66719)
        base = op.ts_rank(base_raw, float(self.rank_n)).clip(lower=self.eps)

        adv40 = op.adv(vol, 40.0)
        c = op.ts_corr(adv40, low, 5.38375)
        expo = op.ts_rank(c, 3.21856)

        return -(base ** expo)


@dataclass
class Factor34(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha91_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        c1 = op.ts_corr(close, vol, 9.74928)
        x1 = op.decay_linear(op.decay_linear(c1, 16.398), 3.83219)
        left = op.ts_rank(x1, 4.8667)

        adv30 = op.adv(vol, 30.0)
        c2 = op.ts_corr(vwap, adv30, 4.01303)
        x2 = op.decay_linear(c2, 2.6809)
        right = op.ts_rank(x2, float(self.rank_n))

        return right - left


@dataclass
class Factor35(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"WQ_Alpha92_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        mid = (high + low) / 2.0
        cond = (((mid + close) < (low + open_))).astype(float)
        x1 = op.decay_linear(cond, 14.7221)
        a = op.ts_rank(x1, 18.8683)

        adv30 = op.adv(vol, 30.0)
        rlow = op.ts_rank(low, float(self.rank_n))
        radv = op.ts_rank(adv30, float(self.rank_n))
        c = op.ts_corr(rlow, radv, 7.58555)
        x2 = op.decay_linear(c, 6.94024)
        b = op.ts_rank(x2, 6.80584)

        return np.minimum(a, b)


@dataclass
class Factor36(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor36_{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv81 = op.adv(vol, 81.0)
        c = op.ts_corr(vwap, adv81, 17.4193)
        x1 = op.decay_linear(c, 19.848)
        num = op.ts_rank(x1, 7.54455)

        mix = (close * 0.524434) + (vwap * (1.0 - 0.524434))
        x2 = op.decay_linear(op.delta(mix, 2.77377), 16.2664)
        den = op.ts_rank(x2, float(self.rank_n)).clip(lower=self.eps)

        return num / den

# -------------------------
# Factor36_1
# -------------------------
@dataclass
class Factor36_1(FactorBase):
    """
    优化1：保留原始比值结构，但提高分母下界。
    """
    rank_n: int = 120
    den_floor: float = 0.05

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor36_1"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv81 = op.adv(vol, 81)
        c = op.ts_corr(vwap, adv81, 17.4193)
        x1 = op.decay_linear(c, 19.848)
        num = op.ts_rank(x1, 7.54455)

        mix = close * 0.524434 + vwap * (1.0 - 0.524434)
        x2 = op.decay_linear(op.ts_delta(mix, 2.77377), 16.2664)
        den = op.ts_rank(x2, self.rank_n).clip(lower=self.den_floor)

        return num / den


# -------------------------
# Factor36_2
# -------------------------
@dataclass
class Factor36_2(FactorBase):
    """
    优化2：改为差值结构，减少分母接近0时的爆炸。
    """
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor36_2"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv81 = op.adv(vol, 81)
        c = op.ts_corr(vwap, adv81, 17.4193)
        x1 = op.decay_linear(c, 19.848)
        num = op.ts_rank(x1, 7.54455)

        mix = close * 0.524434 + vwap * (1.0 - 0.524434)
        x2 = op.decay_linear(op.ts_delta(mix, 2.77377), 16.2664)
        den = op.ts_rank(x2, self.rank_n)

        return num - den
    
@dataclass
class Factor37(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor37_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        base_raw = vwap - op.ts_min(vwap, 11.5783)
        base = op.ts_rank(base_raw, float(self.rank_n)).clip(lower=self.eps)

        adv60 = op.adv(vol, 60.0)
        c = op.ts_corr(op.ts_rank(vwap, 19.6462), op.ts_rank(adv60, 4.02992), 18.0926)
        expo = op.ts_rank(c, 2.70756)

        return -(base ** expo)

# -------------------------
# Factor37_1
# -------------------------
@dataclass
class Factor37_1(FactorBase):
    """
    优化1：提高 eps，拉长 expo 的平滑窗口。
    """
    rank_n: int = 120
    eps: float = 1e-4
    expo_n: int = 8

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor37_1"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        base_raw = vwap - op.ts_min(vwap, 11.5783)
        base = op.ts_rank(base_raw, self.rank_n).clip(lower=self.eps)

        adv60 = op.adv(vol, 60)
        c = op.ts_corr(op.ts_rank(vwap, 19.6462), op.ts_rank(adv60, 4.02992), 18.0926)
        expo = op.ts_rank(c, self.expo_n)

        return -(base ** expo)


# -------------------------
# Factor37_2
# -------------------------
@dataclass
class Factor37_2(FactorBase):
    """
    优化2：加入成交相对活跃门控，增强“有量确认”的时段。
    """
    rank_n: int = 120
    eps: float = 1e-4
    vol_gate_n: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor37_2"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        base_raw = vwap - op.ts_min(vwap, 11.5783)
        base = op.ts_rank(base_raw, self.rank_n).clip(lower=self.eps)

        adv60 = op.adv(vol, 60)
        c = op.ts_corr(op.ts_rank(vwap, 19.6462), op.ts_rank(adv60, 4.02992), 18.0926)
        expo = op.ts_rank(c, 5)

        adv20 = op.adv(vol, self.vol_gate_n).replace(0.0, np.nan)
        vol_gate = (vol / adv20).clip(0, 5)
        vol_gate = op.ts_rank(vol_gate, self.rank_n)

        return -(base ** expo) * (0.5 + 0.5 * vol_gate)
    
@dataclass
class Factor38(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor38_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        left = op.ts_rank(open_ - op.ts_min(open_, 12.4105), float(self.rank_n))

        mid = (high + low) / 2.0
        adv40 = op.adv(vol, 40.0)
        s_mid = op.ts_sum(mid, 19.1351)
        s_adv = op.ts_sum(adv40, 19.1351)
        c = op.ts_corr(s_mid, s_adv, 12.8742)
        r = op.ts_rank(c, float(self.rank_n))
        right = op.ts_rank(r ** 5, 11.7584)

        return right - left


@dataclass
class Factor40(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor40_tsproxy_r{self.rank_n}"

    def formula(self) -> pd.Series:
        low = self.df["low"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        mix = (low * 0.721001) + (vwap * (1.0 - 0.721001))
        x1 = op.decay_linear(op.delta(mix, 3.3705), 20.4523)
        rank_part = op.ts_rank(x1, float(self.rank_n))

        adv60 = op.adv(vol, 60.0)
        c = op.ts_corr(op.ts_rank(low, 7.87871), op.ts_rank(adv60, 17.255), 4.97547)
        t = op.ts_rank(c, 18.5925)
        x2 = op.decay_linear(t, 15.7152)
        ts_part = op.ts_rank(x2, 6.71659)

        return ts_part - rank_part


@dataclass
class Factor41(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "ts"

    @property
    def factor_name(self) -> str:
        return f"Factor41_{self.rank_n}"

    def formula(self) -> pd.Series:
        open_ = self.df["open"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        adv5 = op.adv(vol, 5.0)
        s = op.ts_sum(adv5, 26.4719)
        c1 = op.ts_corr(vwap, s, 4.58418)
        x1 = op.decay_linear(c1, 7.18088)
        left = op.ts_rank(x1, float(self.rank_n))

        adv15 = op.adv(vol, 15.0)
        ro = op.ts_rank(open_, float(self.rank_n))
        ra = op.ts_rank(adv15, float(self.rank_n))
        c2 = op.ts_corr(ro, ra, 20.8187)
        argmin = op.ts_argmin(c2, 8.62571).astype(float)
        t = op.ts_rank(argmin, 6.95668)
        x2 = op.decay_linear(t, 8.07206)
        right = op.ts_rank(x2, float(self.rank_n))

        return left - right


@dataclass
class Factor42(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor42_{self.rank_n}"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        mid = (high + low) / 2.0
        s_mid = op.ts_sum(mid, 19.8975)

        adv60 = op.adv(vol, 60.0)
        s_adv = op.ts_sum(adv60, 19.8975)

        c_left = op.ts_corr(s_mid, s_adv, 8.8136)
        left = op.ts_rank(c_left, float(self.rank_n))

        c_right = op.ts_corr(low, vol, 6.28259)
        right = op.ts_rank(c_right, float(self.rank_n))

        return left - right


@dataclass
class Factor43(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor43_{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        dev = (vwap - close) / close.replace(0.0, np.nan)
        m = vol.rolling(int(self.N), min_periods=int(self.N)).mean()
        s = vol.rolling(int(self.N), min_periods=int(self.N)).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        return dev * vol_z


@dataclass
class Factor44(FactorBase):
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor44"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vwap = op.vwap(self.df)

        rng = (high - low).replace(0.0, np.nan)
        return (vwap - close) / rng

# -------------------------
# Factor44_1
# -------------------------
@dataclass
class Factor44_1(FactorBase):
    """
    优化1：给日内振幅加相对地板，并对 spread 截尾。
    """
    floor_k: float = 1e-3
    spread_clip: float = 10.0

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor44_1"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vwap = op.vwap(self.df)

        rng = high - low
        floor = close.abs() * self.floor_k
        rng = rng.where(rng > floor, floor)

        f = (vwap - close) / rng
        return f.clip(-self.spread_clip, self.spread_clip)


# -------------------------
# Factor44_2
# -------------------------
@dataclass
class Factor44_2(FactorBase):
    """
    优化2：做时间序列中位化/甜蜜区映射，提拔中间状态。
    """
    floor_k: float = 1e-3
    rank_n: int = 60

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor44_2"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vwap = op.vwap(self.df)

        rng = high - low
        floor = close.abs() * self.floor_k
        rng = rng.where(rng > floor, floor)

        core = (vwap - close) / rng
        r = op.ts_rank(core, self.rank_n)
        return -(r - 0.5).abs()

@dataclass
class Factor45(FactorBase):
    L: int = 5
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor45_MR_drawdown_x_volratio_L{self.L}_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        hh = op.ts_max(close, float(self.L))
        dd = close / hh.replace(0.0, np.nan) - 1.0

        adv = op.adv(vol, float(self.N))
        vr = vol / adv.replace(0.0, np.nan)

        return -dd * vr


@dataclass
class Factor46(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor46_MR_ret_x_volz_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change(fill_method=None)

        m = vol.rolling(int(self.N), min_periods=int(self.N)).mean()
        s = vol.rolling(int(self.N), min_periods=int(self.N)).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        return -ret * vol_z


@dataclass
class Factor47(FactorBase):
    w_short: int = 5
    w_long: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor47_PV_corr_short_long_{self.w_short}_{self.w_long}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change(fill_method=None)
        dvol = vol.pct_change(fill_method=None)

        c1 = ret.rolling(int(self.w_short), min_periods=int(self.w_short)).corr(dvol)
        c2 = ret.rolling(int(self.w_long), min_periods=int(self.w_long)).corr(dvol)

        return c1 - c2


@dataclass
class Factor48(FactorBase):
    N: int = 20
    trend: int = 60

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor48_MR_vwap_volz_trendfilter_N{self.N}_T{self.trend}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        dev = (vwap - close) / close.replace(0.0, np.nan)

        m = vol.rolling(int(self.N), min_periods=int(self.N)).mean()
        s = vol.rolling(int(self.N), min_periods=int(self.N)).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        ma = close.rolling(int(self.trend), min_periods=int(self.trend)).mean()
        flt = (close > ma).astype(float)

        return dev * vol_z * flt


@dataclass
class Factor49(FactorBase):
    L: int = 20
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor49_MR_vwap_rangepos_x_volz_L{self.L}_N{self.N}"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        vol = self.df["vol"].astype(float)

        lo = op.ts_min(vwap, float(self.L))
        hi = op.ts_max(vwap, float(self.L))
        rng = (hi - lo).replace(0.0, np.nan)
        pos = (vwap - lo) / rng

        m = vol.rolling(int(self.N), min_periods=int(self.N)).mean()
        s = vol.rolling(int(self.N), min_periods=int(self.N)).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        return (1.0 - pos) * vol_z


@dataclass
class Factor50(FactorBase):
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor50_Ensemble_26_42_r{self.rank_n}"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5.0) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)
        a = op.ts_rank(range_norm.shift(2), float(self.rank_n))
        b1 = op.ts_rank(vol, float(self.rank_n))
        b = op.ts_rank(b1, float(self.rank_n))
        f83 = (a * b) * (vwap - close) / range_norm.replace(0.0, np.nan)

        mid = (high + low) / 2.0
        s_mid = op.ts_sum(mid, 19.8975)
        adv60 = op.adv(vol, 60.0)
        s_adv = op.ts_sum(adv60, 19.8975)
        left = op.ts_rank(op.ts_corr(s_mid, s_adv, 8.8136), float(self.rank_n))
        right = op.ts_rank(op.ts_corr(low, vol, 6.28259), float(self.rank_n))
        f99 = left - right

        return op.ts_rank(f83, float(self.rank_n)) + op.ts_rank(f99, float(self.rank_n))

# -------------------------
# Factor50_1
# -------------------------
@dataclass
class Factor50_1(FactorBase):
    """
    优化1：对组合做加权，提升 Factor26，降低 Factor42 的 regime 风险。
    """
    rank_n: int = 120
    w26: float = 0.65
    w42: float = 0.35

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor50_1"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)
        floor = ma5.abs() * 1e-3
        range_norm = range_norm.where(range_norm > floor, floor)

        a = op.ts_rank(range_norm.shift(2), self.rank_n)
        b = op.ts_rank(op.ts_rank(vol, self.rank_n), self.rank_n)
        f26 = (a * b) * (vwap - close) / range_norm

        mid = (high + low) / 2.0
        s_mid = op.ts_sum(mid, 19.8975)
        adv60 = op.adv(vol, 60)
        s_adv = op.ts_sum(adv60, 19.8975)
        left = op.ts_rank(op.ts_corr(s_mid, s_adv, 8.8136), self.rank_n)
        right = op.ts_rank(op.ts_corr(low, vol, 6.28259), self.rank_n)
        f42 = left - right

        r26 = op.ts_rank(f26, self.rank_n)
        r42 = op.ts_rank(f42, self.rank_n)
        return self.w26 * r26 + self.w42 * r42


# -------------------------
# Factor50_2
# -------------------------
@dataclass
class Factor50_2(FactorBase):
    """
    优化2：组合后再做甜蜜区映射，减少极端桶毒性。
    """
    rank_n: int = 120

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor50_2"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)
        vwap = op.vwap(self.df)

        ma5 = op.ts_sum(close, 5) / 5.0
        range_norm = (high - low) / ma5.replace(0.0, np.nan)
        floor = ma5.abs() * 1e-3
        range_norm = range_norm.where(range_norm > floor, floor)

        a = op.ts_rank(range_norm.shift(2), self.rank_n)
        b = op.ts_rank(op.ts_rank(vol, self.rank_n), self.rank_n)
        f26 = (a * b) * (vwap - close) / range_norm

        mid = (high + low) / 2.0
        s_mid = op.ts_sum(mid, 19.8975)
        adv60 = op.adv(vol, 60)
        s_adv = op.ts_sum(adv60, 19.8975)
        left = op.ts_rank(op.ts_corr(s_mid, s_adv, 8.8136), self.rank_n)
        right = op.ts_rank(op.ts_corr(low, vol, 6.28259), self.rank_n)
        f42 = left - right

        combo = 0.6 * op.ts_rank(f26, self.rank_n) + 0.4 * op.ts_rank(f42, self.rank_n)
        r = op.ts_rank(combo, self.rank_n)
        return -(r - 0.5).abs()

@dataclass
class Factor51(FactorBase):
    N: int = 60

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor51_Amihud_illq_negZ_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        amount = self.df["amount"].astype(float)

        r = close.pct_change().abs()
        illiq = r / amount.replace(0.0, np.nan)

        w = int(self.N)
        ill_m = illiq.rolling(w, min_periods=w).mean()
        m = ill_m.rolling(w, min_periods=w).mean()
        s = ill_m.rolling(w, min_periods=w).std()
        return -(ill_m - m) / s.replace(0.0, np.nan)


@dataclass
class Factor52(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor52_Corr_ret_logVol_rank_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        lv = op.safe_log(vol.replace(0.0, np.nan))
        c = op.ts_corr(ret, lv, float(self.N))
        return op.ts_rank(c, float(self.N))


@dataclass
class Factor53(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor53_UpDownVolRatio_Z_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        up = vol.where(ret > 0.0, 0.0)
        dn = vol.where(ret < 0.0, 0.0)

        w = int(self.N)
        up_s = up.rolling(w, min_periods=w).sum()
        dn_s = dn.rolling(w, min_periods=w).sum().replace(0.0, np.nan)

        ratio = op.safe_log(up_s / dn_s)
        m = ratio.rolling(w, min_periods=w).mean()
        s = ratio.rolling(w, min_periods=w).std()
        return (ratio - m) / s.replace(0.0, np.nan)


@dataclass
class Factor54(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor54_VWR_Z_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        w = int(self.N)
        num = (ret * vol).rolling(w, min_periods=w).sum()
        den = vol.rolling(w, min_periods=w).sum().replace(0.0, np.nan)
        vwr = num / den

        m = vwr.rolling(w, min_periods=w).mean()
        s = vwr.rolling(w, min_periods=w).std()
        return (vwr - m) / s.replace(0.0, np.nan)


@dataclass
class Factor55(FactorBase):
    N: int = 60
    h: int = 5

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor55_Mom{self.h}_x_volZ_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        mom = op.log_return(close, int(self.h))

        w = int(self.N)
        m = vol.rolling(w, min_periods=w).mean()
        s = vol.rolling(w, min_periods=w).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        return mom * vol_z


@dataclass
class Factor56(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor56_BreakoutHigh_x_volRank_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        high = self.df["high"].astype(float)
        vol = self.df["vol"].astype(float)

        hh = op.ts_max(high, float(self.N))
        brk = (close - hh) / close.replace(0.0, np.nan)
        v_rank = op.ts_rank(vol, float(self.N))
        return brk * v_rank


@dataclass
class Factor57(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor57_DirRet_x_VolRel_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        adv = op.adv(vol, float(self.N)).replace(0.0, np.nan)
        vol_rel = vol / adv
        return ret * vol_rel


@dataclass
class Factor58(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor58_VWAPDevZ_x_amtRank_N{self.N}"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        amount = self.df["amount"].astype(float)

        w = int(self.N)
        ma = vwap.rolling(w, min_periods=w).mean()
        dev = (vwap / ma.replace(0.0, np.nan)) - 1.0

        m = dev.rolling(w, min_periods=w).mean()
        s = dev.rolling(w, min_periods=w).std()
        dev_z = (dev - m) / s.replace(0.0, np.nan)

        amt_rank = op.ts_rank(amount, float(self.N))
        return dev_z * amt_rank


@dataclass
class Factor59(FactorBase):
    N: int = 14

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor59_MFI_proxy_N{self.N}"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        tp = (high + low + close) / 3.0
        mf = tp * vol
        up = mf.where(tp > tp.shift(1), 0.0)
        dn = mf.where(tp < tp.shift(1), 0.0)

        w = int(self.N)
        pos = up.rolling(w, min_periods=w).sum()
        neg = dn.rolling(w, min_periods=w).sum().replace(0.0, np.nan)
        mfi = 100.0 - 100.0 / (1.0 + (pos / neg))
        return (mfi - 50.0) / 50.0


@dataclass
class Factor60(FactorBase):
    short: int = 3
    long: int = 10

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor60_ChaikinOsc_s{self.short}_l{self.long}"

    def formula(self) -> pd.Series:
        high = self.df["high"].astype(float)
        low = self.df["low"].astype(float)
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        hl = (high - low).replace(0.0, np.nan)
        mfm = ((close - low) - (high - close)) / hl
        mfv = mfm.fillna(0.0) * vol
        ad = mfv.cumsum()

        s = ad.rolling(int(self.short), min_periods=int(self.short)).mean()
        l = ad.rolling(int(self.long), min_periods=int(self.long)).mean()
        return s - l


@dataclass
class Factor61(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor61_RV_x_VolRel_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        rv = op.realized_vol(close, int(self.N))
        adv = op.adv(vol, float(self.N)).replace(0.0, np.nan)
        return rv * (vol / adv)


@dataclass
class Factor62(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor62_PriceRank_minus_VolRank_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        pr = op.ts_rank(close, float(self.N))
        vr = op.ts_rank(vol, float(self.N))
        return pr - vr


@dataclass
class Factor63(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor63_VolCV_N{self.N}"

    def formula(self) -> pd.Series:
        vol = self.df["vol"].astype(float)
        w = int(self.N)
        m = vol.rolling(w, min_periods=w).mean().replace(0.0, np.nan)
        s = vol.rolling(w, min_periods=w).std()
        return s / m


@dataclass
class Factor64(FactorBase):
    N: int = 14

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor64_VW_RSI_proxy_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        gain = ret.clip(lower=0.0) * vol
        loss = (-ret).clip(lower=0.0) * vol

        w = int(self.N)
        g = gain.rolling(w, min_periods=w).sum()
        l = loss.rolling(w, min_periods=w).sum().replace(0.0, np.nan)
        rs = g / l
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return (rsi - 50.0) / 50.0


@dataclass
class Factor65(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor65_Corr_absRet_vol_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ar = close.pct_change().abs()
        return op.ts_corr(ar, vol, float(self.N))


@dataclass
class Factor66(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor66_IntraStrength_x_volZ_N{self.N}"

    def formula(self) -> pd.Series:
        o = self.df["open"].astype(float)
        c = self.df["close"].astype(float)
        h = self.df["high"].astype(float)
        l = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        rng = (h - l).replace(0.0, np.nan)
        strength = (c - o) / rng

        w = int(self.N)
        m = vol.rolling(w, min_periods=w).mean()
        s = vol.rolling(w, min_periods=w).std()
        vol_z = (vol - m) / s.replace(0.0, np.nan)

        return strength * vol_z

# -------------------------
# Factor67_1
# -------------------------
@dataclass
class Factor67_1(FactorBase):
    """
    优化1：gap 截尾，用 vol/adv 代替 ts_rank(vol)。
    """
    N: int = 20
    gap_clip: float = 0.1

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor67_1"

    def formula(self) -> pd.Series:
        o = self.df["open"].astype(float)
        c = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        gap = ((o / c.shift(1).replace(0.0, np.nan)) - 1.0).clip(-self.gap_clip, self.gap_clip)
        adv = op.adv(vol, self.N).replace(0.0, np.nan)
        vol_ratio = (vol / adv).clip(0, 5)
        return gap * vol_ratio


# -------------------------
# Factor67_2
# -------------------------
@dataclass
class Factor67_2(FactorBase):
    """
    优化2：在 gap*放量 的基础上叠加日内强弱确认，过滤“高开低走”。
    """
    N: int = 20
    gap_clip: float = 0.1
    floor_k: float = 1e-3

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor67_2"

    def formula(self) -> pd.Series:
        o = self.df["open"].astype(float)
        c = self.df["close"].astype(float)
        h = self.df["high"].astype(float)
        l = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        gap = ((o / c.shift(1).replace(0.0, np.nan)) - 1.0).clip(-self.gap_clip, self.gap_clip)
        adv = op.adv(vol, self.N).replace(0.0, np.nan)
        vol_ratio = (vol / adv).clip(0, 5)

        rng = h - l
        floor = c.abs() * self.floor_k
        rng = rng.where(rng > floor, floor)
        intraday = ((c - o) / rng).clip(-1, 1)

        return gap * vol_ratio * (0.5 + 0.5 * intraday.clip(lower=0))

@dataclass
class Factor67(FactorBase):
    N: int = 20

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor67_Gap_x_volRank_N{self.N}"

    def formula(self) -> pd.Series:
        o = self.df["open"].astype(float)
        c = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        gap = (o / c.shift(1).replace(0.0, np.nan)) - 1.0
        v_rank = op.ts_rank(vol, float(self.N))
        return gap * v_rank


@dataclass
class Factor68(FactorBase):
    N: int = 30

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor68_AvgTradeZ_x_volRank_N{self.N}"

    def formula(self) -> pd.Series:
        amount = self.df["amount"].astype(float)
        vol = self.df["vol"].astype(float)

        avg_trade = amount / vol.replace(0.0, np.nan)

        w = int(self.N)
        m = avg_trade.rolling(w, min_periods=w).mean()
        s = avg_trade.rolling(w, min_periods=w).std()
        at_z = (avg_trade - m) / s.replace(0.0, np.nan)

        v_rank = op.ts_rank(vol, float(self.N))
        return at_z * v_rank


@dataclass
class Factor69(FactorBase):
    N: int = 60

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor69_WRetSkewZ_N{self.N}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        vol = self.df["vol"].astype(float)

        ret = close.pct_change()
        w = int(self.N)
        mu = ret.rolling(w, min_periods=w).mean()
        cen3 = (ret - mu) ** 3
        num = (cen3 * vol).rolling(w, min_periods=w).sum()
        den = vol.rolling(w, min_periods=w).sum().replace(0.0, np.nan)
        m3 = num / den

        m = m3.rolling(w, min_periods=w).mean()
        s = m3.rolling(w, min_periods=w).std()
        return (m3 - m) / s.replace(0.0, np.nan)


@dataclass
class Factor70(FactorBase):
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return "Factor70"

    def formula(self) -> pd.Series:
        vwap = op.vwap(self.df)
        close = self.df["close"].astype(float)

        x = vwap - op.ts_max(vwap, 15.3217)
        base = op.ts_rank(x, 20.7127).clip(lower=self.eps)

        expo_raw = op.delta(close, 4.96796)
        expo = np.sign(expo_raw) * op.ts_rank(expo_raw.abs(), 20.0)

        f = op.signed_power(base, expo, eps=self.eps)

        r = f.groupby(level="trade_date").rank(pct=True)     # (0,1]
        return -(r - 0.5).abs()
    

@dataclass
class Factor71(FactorBase):
    rank_n: int = 120
    eps: float = 1e-12

    @property
    def my_type(self):
        return "cs"

    @property
    def factor_name(self) -> str:
        return f"Factor71_{self.rank_n}"

    def formula(self) -> pd.Series:
        close = self.df["close"].astype(float)
        low = self.df["low"].astype(float)
        vol = self.df["vol"].astype(float)

        base_raw = close - op.ts_max(close, 4.66719)
        base = op.ts_rank(base_raw, float(self.rank_n)).clip(lower=self.eps)

        adv40 = op.adv(vol, 40.0)
        c = op.ts_corr(adv40, low, 5.38375)
        expo = op.ts_rank(c, 3.21856)

        f = -(base ** expo)
        m = f.groupby(level=0).transform("median")

        return (-1) * (f - m).abs()