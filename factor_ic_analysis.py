# features/factor_ic_analysis.py
"""
factor_ic_analysis.py
单因子（横截面）IC 分析。

包含收益率对齐(T+1)、截面去极值与标准化、ICIR/胜率评估、十分组收益测试、自相关性评估。
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 引入进度条
try:
    from tqdm import tqdm
except ImportError:
    print("提示: 未安装 tqdm，将无法显示进度条。可使用 'pip install tqdm' 安装。")
    tqdm = lambda x, **kwargs: x  # Fallback 机制

# 屏蔽底层数学警告（保持控制台清爽）
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*constant.*')
warnings.filterwarnings('ignore', category=RuntimeWarning)

from factor import FactorBase  # type: ignore
import factor as factor_module  # type: ignore

# 导入交易日历模块
from trade_calendar import valid_trading_date, _load_trade_calendar, DEFAULT_CALENDAR_PATH

# -----------------------------
# 日志配置 (Logging Setup)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("FactorIC")

# -----------------------------
# IO & Calendar helpers
# -----------------------------

DEFAULT_FILE_SUFFIX = "_daily.parquet"  

def get_stock_file_list(folder_path: str) -> List[str]:
    file_names: List[str] = []
    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.name == ".DS_Store" or not entry.is_file():
                    continue
                if not entry.name.endswith(".parquet"):
                    continue
                code = entry.name
                if code.endswith(DEFAULT_FILE_SUFFIX):
                    code = code[: -len(DEFAULT_FILE_SUFFIX)]
                else:
                    code = code[:9]
                file_names.append(code)
        file_names.sort()
        return file_names
    except FileNotFoundError:
        logger.error(f"找不到文件夹 '{folder_path}'")
        return []

def _resolve_stock_path(data_dir: str, code: str) -> str:
    p1 = os.path.join(data_dir, f"{code}{DEFAULT_FILE_SUFFIX}")
    if os.path.exists(p1): return p1
    for p in (os.path.join(data_dir, f"{code}_daily.parquet"), os.path.join(data_dir, f"{code}.parquet")):
        if os.path.exists(p): return p
    return p1

def read_stock_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
        df = df.drop_duplicates(subset=["trade_date"], keep="last").set_index("trade_date")
    else:
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce").dt.normalize()
            df.index.name = df.index.name or "trade_date"
            df = df.sort_index()
        except Exception:
            pass
    return df

def get_buffer_start_date(target_date: pd.Timestamp, buffer_days: int) -> pd.Timestamp:
    cal = _load_trade_calendar(DEFAULT_CALENDAR_PATH)
    idx = cal.searchsorted(target_date, side="left")
    buffer_idx = max(0, idx - buffer_days)
    return cal[buffer_idx]

# -----------------------------
# Factor & Math helpers
# -----------------------------

def get_factor_class(factor_name: str):
    if not hasattr(factor_module, factor_name):
        avail = [n for n in dir(factor_module) if re.match(r"^Factor\d+$", n)]
        raise ValueError(f"factor.py 中找不到 '{factor_name}'，可用：{', '.join(avail)}")
    cls = getattr(factor_module, factor_name)
    if not isinstance(cls, type) or not issubclass(cls, FactorBase):
        raise ValueError(f"'{factor_name}' 不是 FactorBase 子类")
    return cls

def parse_kv_args(s: str) -> Dict[str, Any]:
    if not s: return {}
    out: Dict[str, Any] = {}
    for part in [p.strip() for p in s.split(",") if p.strip()]:
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if re.fullmatch(r"-?\d+", v): out[k] = int(v)
        elif re.fullmatch(r"-?\d+\.\d*", v) or re.fullmatch(r"-?\d*\.\d+", v): out[k] = float(v)
        elif v.lower() in ("true", "false"): out[k] = v.lower() == "true"
        else: out[k] = v
    return out

def compute_forward_returns(close: pd.Series, max_h: int) -> Dict[int, pd.Series]:
    """
    计算未来收益率 (防未来函数升级版)
    T日盘后计算的因子，最早只能在 T+1 日开盘或收盘买入。
    此处保守设定为 T+1 日收盘买入，T+1+h 日收盘卖出。
    """
    close = pd.to_numeric(close, errors="coerce")
    out: Dict[int, pd.Series] = {}
    
    buy_price = close.shift(-1) # 模拟 T+1 进场
    for h in range(1, max_h + 1):
        sell_price = close.shift(-(h + 1)) # 模拟 T+1+h 出场
        r = sell_price / buy_price - 1.0
        r.name = f"fwd_ret_{h}"
        out[h] = r
    return out

def _spearman_cs(x: pd.Series, y: pd.Series) -> float:
    aligned = pd.concat([x, y], axis=1).dropna()
    if len(aligned) < 3: return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")

# -----------------------------
# NEW: Cross-Sectional & Metrics Helpers
# -----------------------------

def preprocess_cross_section(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """截面预处理：MAD去极值 + Z-score标准化"""
    logger.info("执行截面数据预处理 (MAD去极值 + Z-score标准化)...")
    
    def _process_daily(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if len(s) < 3: return s
        
        # 1. MAD 去极值
        median = s.median()
        mad = (s - median).abs().median()
        if mad > 0:
            upper = median + 3.1483 * mad
            lower = median - 3.1483 * mad
            s = s.clip(lower=lower, upper=upper)
        
        # 2. Z-score 标准化
        std = s.std()
        if std > 0:
            s = (s - s.mean()) / std
        return s

    # 对每一天（行）应用处理
    processed = factor_panel.apply(_process_daily, axis=1)
    return processed

def compute_factor_autocorr(factor_panel: pd.DataFrame, lag: int = 1) -> float:
    """计算因子秩自相关性 (评估换手率)"""
    rank_panel = factor_panel.rank(axis=1)
    autocorr_series = rank_panel.corrwith(rank_panel.shift(lag), axis=1, method='spearman')
    return float(autocorr_series.mean())

def summary_ic_metrics(ic_df: pd.DataFrame) -> pd.DataFrame:
    """扩充的 IC 统计表 (含 ICIR 和 胜率)"""
    metrics = {}
    for col in ic_df.columns:
        ic_s = ic_df[col].dropna()
        if ic_s.empty: 
            print(f"IC_metrics 为空")
            continue
        else:
            n = len(metrics)
            print(f"ic_metrics 长度 = {n}")
        mean_ic = ic_s.mean()
        std_ic = ic_s.std()
        icir = mean_ic / std_ic if std_ic != 0 else np.nan
        win_rate = (ic_s > 0).sum() / len(ic_s)
        
        metrics[col] = {
            "Mean IC": mean_ic,
            "Std IC": std_ic,
            "ICIR (Ann)": icir * np.sqrt(252),  # 年化 ICIR
            "ICIR (Mon)": icir * np.sqrt(21),   # 月化 ICIR
            "IC WinRate": win_rate
        }
    return pd.DataFrame(metrics).T

# -----------------------------
# Core pipeline
# -----------------------------
def _chunk_worker(
    data_dir: str, codes_chunk: List[str], factor_name: str, factor_kwargs: Dict[str, Any], max_h: int,
    actual_start: pd.Timestamp, actual_end: pd.Timestamp, buffer_start: pd.Timestamp
) -> Tuple[List[pd.Series], Dict[int, List[pd.Series]], Dict[int, int], List[str]]:
    """
    Worker：一次性处理多只股票
    """
    chunk_factors = []
    chunk_fwd_rets = {h: [] for h in range(1, max_h + 1)}
    chunk_tail_missing = {h: 0 for h in range(1, max_h + 1)}
    chunk_errors = []

    # 动态获取因子类
    factor_cls = get_factor_class(factor_name)

    for code in codes_chunk:
        try:
            path = _resolve_stock_path(data_dir, code)
            if not os.path.exists(path): 
                continue

            df = read_stock_parquet(path)
            df = df.loc[(df.index >= buffer_start) & (df.index <= actual_end)]

            if len(df) < 10: 
                continue

            if "close" not in df.columns: 
                continue

            # 1. 计算因子
            fac_obj = factor_cls(df=df, symbol=code, **factor_kwargs)
            fac = fac_obj.score().rename(code)

            # 2. 计算未来收益率
            fwd = compute_forward_returns(df["close"], max_h)
            
            # 3. 切掉 buffer
            fac = fac.loc[fac.index >= actual_start]
            if len(fac) > 0:
                chunk_factors.append(fac)

            for h in range(1, max_h + 1):
                s = fwd[h].rename(code)
                s = s.loc[s.index >= actual_start]
                if len(s) > 0:
                    chunk_fwd_rets[h].append(s)
                chunk_tail_missing[h] += int(s.tail(h).isna().sum())

        except Exception as e:
            chunk_errors.append(f"{code}: {type(e).__name__}: {e}")

    return chunk_factors, chunk_fwd_rets, chunk_tail_missing, chunk_errors


def build_panels(
    data_dir: str, codes: Iterable[str], factor_name: str, factor_kwargs: Dict[str, Any],
    max_h: int, workers: int, actual_start: pd.Timestamp, actual_end: pd.Timestamp, buffer_start: pd.Timestamp
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, int]]:
    
    factor_series: List[pd.Series] = []
    ret_series: Dict[int, List[pd.Series]] = {h: [] for h in range(1, max_h + 1)}
    missing_tail_total: Dict[int, int] = {h: 0 for h in range(1, max_h + 1)}
    errors: List[str] = []
    codes_list = list(codes)

    # 把 5000 只股票切分成多个 chunk，每个 chunk 包含 50 只股票
    CHUNK_SIZE = 50
    chunks = [codes_list[i:i + CHUNK_SIZE] for i in range(0, len(codes_list), CHUNK_SIZE)]

    logger.info(f"进程数: {workers}，批处理大小: {CHUNK_SIZE}")
    
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(
                _chunk_worker, data_dir, chunk, factor_name, factor_kwargs,
                max_h, actual_start, actual_end, buffer_start
            ): chunk for chunk in chunks
        }

        # 进度条：position=0 强制在原地刷新，dynamic_ncols 适应屏幕宽度
        pbar = tqdm(as_completed(futs), total=len(chunks), desc="并发计算中", position=0, leave=True, dynamic_ncols=True)
        
        for fut in pbar:
            try:
                # 解析子进程传回来的批处理结果
                c_facs, c_fwds, c_missing, c_errs = fut.result()
                
                factor_series.extend(c_facs)
                for h in range(1, max_h + 1):
                    ret_series[h].extend(c_fwds[h])
                    missing_tail_total[h] += c_missing[h]
                errors.extend(c_errs)
                
                # 更新进度条显示的股票数
                pbar.set_postfix({"已处理批次": f"每批{CHUNK_SIZE}只"})
                
            except Exception as e:
                errors.append(f"Chunk Error: {e}")

    if errors:
        logger.warning(f"部分标的处理失败，共 {len(errors)} 条错误:")
        for msg in errors: logger.warning(f"  - {msg}")

    if not factor_series: raise RuntimeError("没有成功计算任何股票的因子值")

    logger.info("合并全市场数据宽表 (Data Panel)" + "\n")
    factor_panel = pd.concat(factor_series, axis=1).sort_index()
    
    ret_panels: Dict[int, pd.DataFrame] = {}
    for h in range(1, max_h + 1):
        if ret_series[h]:
            ret_panels[h] = pd.concat(ret_series[h], axis=1).sort_index()
        else:
            ret_panels[h] = pd.DataFrame(index=factor_panel.index)
        ret_panels[h] = ret_panels[h].reindex(factor_panel.index)

    return factor_panel, ret_panels, missing_tail_total

# =================================================================================================

def compute_daily_ic(factor_panel: pd.DataFrame, ret_panels: Dict[int, pd.DataFrame], max_h: int) -> pd.DataFrame:
    logger.info("开始计算每日横截面 IC (Spearman Rank Correlation)...")
    ic_df = pd.DataFrame(index=factor_panel.index)
    
    for h in tqdm(range(1, max_h + 1), desc="计算 IC", unit="h", position=0, leave=True, dynamic_ncols=True):
        rets = ret_panels[h]
        vals: List[float] = []
        for dt in ic_df.index:
            vals.append(_spearman_cs(factor_panel.loc[dt], rets.loc[dt]))
        ic_df[f"IC_h{h}"] = vals
    return ic_df
# -----------------------------
# NEW: Plotting & Outputs (Quantiles)
# -----------------------------

def _ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_ic_curves(ic_df: pd.DataFrame, output_dir: str, title_prefix: str, topk: int = 5) -> None:
    outdir = _ensure_output_dir(output_dir)
    mean_abs = ic_df.mean(numeric_only=True).abs().sort_values(ascending=False)
    sel = list(mean_abs.index[: max(1, int(topk))])

    plt.figure(figsize=(10,5))
    ic_df[sel].cumsum().plot()
    plt.title(f"{title_prefix} - Cumulative IC (Top {len(sel)} horizons)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cum_ic_topk.png"), dpi=150)
    plt.close()

def plot_ic_decay(metrics_df: pd.DataFrame, output_dir: str, title_prefix: str) -> None:
    """
    绘制 IC 衰减图 (IC Decay Plot)
    横轴：预测期限 h (1 到 max_h)
    纵轴：Mean IC
    """
    outdir = _ensure_output_dir(output_dir)
    
    if "Mean IC" not in metrics_df.columns:
        logger.warning("metrics_df 中找不到 'Mean IC'，无法绘制 IC 衰减图。")
        return
        
    mean_ic = metrics_df["Mean IC"]
    
    # 提取横坐标 h 的数字 (从 "IC_h1", "IC_h2" 等 index 中提取)
    horizons = []
    for col in mean_ic.index:
        match = re.search(r'\d+', str(col))
        if match:
            horizons.append(int(match.group()))
        else:
            horizons.append(col)
            
    plt.figure(figsize=(10, 5))
    
    # 画柱状图，表示每个 h 的 IC
    bars = plt.bar(horizons, mean_ic.values, color='skyblue', edgecolor='black', alpha=0.7, label='Mean IC')
    
    # 画折线图，强调衰减趋势
    plt.plot(horizons, mean_ic.values, marker='o', color='red', linestyle='-', linewidth=2, label='Decay Trend')
    
    # 添加 0 轴基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # 图表装饰
    plt.title(f"{title_prefix} - IC Decay Over Time")
    plt.xlabel("Horizon (Days)")
    plt.ylabel("Mean Rank IC")
    plt.xticks(horizons)  # 强制显示所有的 h 刻度
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(outdir, "ic_decay.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def compute_and_plot_quantiles(factor_panel: pd.DataFrame, fwd_ret: pd.DataFrame, best_h: int, output_dir: str, title: str, groups: int = 10):
    """十分组分析"""
    logger.info(f"进行 {groups} 分组收益率测试 (基于最佳持有期 h={best_h})...")
    
    # 展开并对齐
    f_stacked = factor_panel.stack().rename("factor")
    r_stacked = fwd_ret.stack().rename("ret")
    df = pd.concat([f_stacked, r_stacked], axis=1).dropna()
    
    if df.empty:
        logger.warning("数据不足，无法计算分组收益！")
        return
        
    def _group_daily(df_day):
        df_day = df_day.copy()
        n = len(df_day)

        # 你可以调这个阈值：比如至少 30/50/100 才做 10分组
        min_n = max(5, groups * 10)   
        if n < min_n:
            df_day["group"] = np.nan
            return df_day

        # rank
        ranks = df_day["factor"].rank(method="first")

        # 等人数分箱（避免 qcut 的 bin edge 问题）
        # 保证 min rank -> 1组，max rank -> groups组
        if n == 1:
            grp = pd.Series([1], index=df_day.index, dtype="float")
        else:
            grp = np.floor((ranks - 1) / (n - 1) * (groups - 1)) + 1

        df_day["group"] = grp
        return df_day

    # 1. 每天打上分组标签
    df_with_group = df.groupby(level=0, group_keys=False).apply(_group_daily)
    df_with_group = df_with_group.dropna(subset=["group"])
    
    # 2.使用 pivot_table 安全地计算透视表
    # index="trade_date" (也就是 level=0 的日期), columns="group", values="ret"
    # aggfunc='mean' 计算每天每个组的平均收益
    daily_group_ret = pd.pivot_table(
        df_with_group.reset_index(), 
        values="ret", 
        index="trade_date", 
        columns="group", 
        aggfunc="mean"
    )
    
    # 检查是否生成了预期的 DataFrame
    if not isinstance(daily_group_ret, pd.DataFrame):
        logger.error("透视表计算异常，未能生成 DataFrame。")
        return
        
    # 填充可能缺失的分组收益（极少数情况当天某组全为空），用 0 填充
    daily_group_ret = daily_group_ret.fillna(0)
    
    # 画图
    cum_ret = (1 + daily_group_ret).cumprod() - 1
    
    # 1. 各组累计收益图
    plt.figure(figsize=(10, 6))
    for col in cum_ret.columns:
        plt.plot(cum_ret.index, cum_ret[col], label=f"Group {col}")
    plt.title(f"{title} - {groups} Quantiles Cumulative Returns (h={best_h})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "quantile_cum_returns.png"), dpi=150)
    plt.close()

    # 输出 group 每组个数
    print(f"df 行数：{len(df)}")
    
    # 2. 多空对冲图 (Group 10 - Group 1)
    # 确保 Group 1 和 Group 10 都存在
    
    if groups in daily_group_ret.columns and 1 in daily_group_ret.columns:
        ls_ret = daily_group_ret[groups] - daily_group_ret[1]
        ls_cum = ls_ret.cumsum()
        plt.figure(figsize=(10, 4))
        plt.plot(ls_cum.index, ls_cum.values, color='red', label=f"Long G{groups} - Short G1")
        plt.title(f"{title} - Long-Short Spread (h={best_h})")
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "long_short_spread.png"), dpi=150)
        plt.close()
    else:
        logger.warning(f"数据不足以绘制多空对冲图，找不到 Group 1 或 Group {groups}")

# -----------------------------
# Public API
# -----------------------------

def run_ic_analysis(
    data_dir: str,
    factor_name: str,
    factor_args: str = "",
    max_h: int = 20,
    workers: int = 8,
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    buffer_days: int = 60,
    output_dir: Optional[str] = None,
    topk_plot: int = 5,
) -> Dict[str, Any]:
    
    logger.info("="*60)
    logger.info(f"开始执行 {factor_name} 因子: ")
    logger.info("="*60)

    # === 1. 时间处理 ===
    s_date, e_date = valid_trading_date(start, end)
    actual_start = pd.to_datetime(s_date)
    actual_end = pd.to_datetime(e_date)
    buffer_start = get_buffer_start_date(actual_start, buffer_days=buffer_days)
    
    # === 2. 加载代码表 ===
    factor_kwargs = parse_kv_args(factor_args)
    codes = get_stock_file_list(data_dir)
    if not codes: 
        logger.error(f"没有找到 parquet 文件：{data_dir}")
        return {}

    if output_dir is None: output_dir = os.path.join(os.getcwd(), "outputs", factor_name)
    output_dir = _ensure_output_dir(output_dir)

    # === 3. 并发构建数据宽表 ===
    raw_factor_panel, ret_panels, tail_missing = build_panels(
        data_dir, codes, factor_name, factor_kwargs, max_h, workers,
        actual_start, actual_end, buffer_start
    )

    # === 4. 截面去极值与标准化 ===
    factor_panel = preprocess_cross_section(raw_factor_panel)

    # === 5. 计算换手率评估 (自相关性) ===
    autocorr = compute_factor_autocorr(factor_panel, lag=1)

    # === 6. 计算 IC 与 核心指标 ===
    ic_df = compute_daily_ic(factor_panel, ret_panels, max_h)
    
    # 生成详尽度量表
    metrics_df = summary_ic_metrics(ic_df)
    if metrics_df.empty or ("Mean IC" not in metrics_df.columns):
        logger.error("IC 全为 NaN，metrics_df 为空：请检查因子是否为常数/含inf/NaN，或截面样本不足。")
        return {}
    
    # 找到绝对值 Mean IC 最大的持有期 best_h
    best_col = metrics_df["Mean IC"].abs().idxmax()
    best_h = int(best_col.replace("IC_h", ""))
    best_metrics = metrics_df.loc[best_col]

    # === 7. 十分组分析测试 ===
    compute_and_plot_quantiles(
        factor_panel=factor_panel, 
        fwd_ret=ret_panels[best_h], 
        best_h=best_h, 
        output_dir=output_dir, 
        title=factor_name,
        groups=10
    )

    # === 8. 存储与作图 ===
    logger.info("保存详细计算结果与绘制 IC 图表...")
    # ic_df.to_csv(os.path.join(output_dir, "daily_ic.csv"))
    # metrics_df.to_csv(os.path.join(output_dir, "ic_metrics_summary.csv"))
    
    plot_ic_curves(ic_df, output_dir, title_prefix=factor_name, topk=topk_plot)
    # 绘制并保存 IC 衰减图
    plot_ic_decay(metrics_df, output_dir, title_prefix=factor_name)
        
    logger.info("\n" + "="*40)
    logger.info(f" {factor_name} 最终评测报告 (面试展示专用)")
    logger.info("="*40)
    logger.info(f"有效测试标的数 : {factor_panel.shape[1]}")
    logger.info(f"最佳预测持有期 : {best_h} 天")
    logger.info(f"IC decay 衰减图已添加")
    logger.info(f"自相关性(换手) : {autocorr:.4f} (越接近1换手越低)")
    logger.info("-" * 40)
    logger.info(f"Mean IC      : {best_metrics['Mean IC']:.4f}")
    logger.info(f"年化 ICIR     : {best_metrics['ICIR (Ann)']:.4f}")
    logger.info(f"月化 ICIR    : {best_metrics['ICIR (Mon)']:.4f}") 
    logger.info(f"IC 胜率        : {best_metrics['IC WinRate']*100:.2f}%")
    logger.info("="*40)
    logger.info(f"报告及图片(十分组图等)已输出至: {output_dir}\n")

    return {"daily_ic": ic_df, "metrics": metrics_df, "best_h": best_h}

if __name__ == "__main__":
    fac_ls = []
    # 优化因子列
    fac2 = [# "Factor3_1","Factor3_2",
    #         "Factor24_1","Factor24_2",
            # "Factor26_1","Factor26_2",
            # "Factor36_1",
            # "Factor36_2",
            "Factor37_1",
            "Factor37_2",
            "Factor44_1",
            "Factor44_2",
            "Factor50_1",
            "Factor50_2",
            "Factor67_1",
            "Factor67_2",
            ]
    # 因子编号
    for i in range(40,71,1):
        fac_ls.append("Factor" + str(i))
    # fac_ls -= ["Factor39","Factor70","Factor28"]
    for fac in fac2:
        print("\n" + "*"*60)
        run_ic_analysis(
            data_dir="/Users/chaoyi/Documents/factor_lab/stock_data/daily", 
            factor_name=fac, 
            max_h=20, 
            workers=16, 
            start="2016-04-01", 
            end="2023-12-30",
            buffer_days=60,  
            output_dir=f"./outputs/{fac}_IC_Result" 
        )