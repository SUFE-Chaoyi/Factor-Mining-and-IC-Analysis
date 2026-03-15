# IC analysis

## 1. 模块定位
`factor_ic_analysis.py` 用于执行**单因子横截面 IC 分析**，核心目标是评估某个因子在全市场股票横截面上的预测能力。

完整流程包括：

1. 读取单股票日线 parquet；
2. 计算单股票时序因子；
3. 构建横截面因子面板与未来收益面板；
4. 做日度截面去极值与标准化；
5. 计算多个持有期的日度 IC；
6. 汇总 Mean IC / ICIR / WinRate / T-stat 等指标；
7. 计算因子秩自相关（作为换手代理指标）；
8. 基于最佳持有期做十分组收益和多空分析；
9. 输出结果文件和调试诊断文件。

---

## 2. 数据入口

### 2.1 股票文件扫描
- `get_stock_file_list(data_dir)` 扫描目录下所有 parquet 文件；
- 支持：
  - `{code}_daily.parquet`
  - `{code}.parquet`
- 返回股票代码列表 `codes`。

### 2.2 单股票数据读取
- `read_stock_parquet(path)`：
  - 若存在 `trade_date` 列，则转为 `DatetimeIndex`；
  - 若没有 `trade_date` 列，则尝试把索引解析成日期；
  - 去重、排序，最终返回按交易日升序排列的 DataFrame。

### 2.3 时间对齐
- `valid_trading_date(start, end)`：把自然日映射到有效交易日；
- `get_buffer_start_date(actual_start, buffer_days)`：往前取 buffer 区间，供滚动窗口预热使用。

---

## 3. 单股票层面的计算

### 3.1 因子类解析
- `get_factor_class(factor_name)`：
  - 在 `factor.py` 中寻找 `factor_name`；
  - 若不存在，直接抛异常；
  - 若不是 `FactorBase` 子类，也直接抛异常。

### 3.2 参数解析
- `parse_kv_args(factor_args)`：
  - 支持 `t1=10,t2=20,use_vol=true` 这种字符串；
  - 自动转为 int / float / bool / str；
  - 若未显式传参，则使用 `factor.py` 中因子类默认参数。

### 3.3 因子计算
对单只股票：

1. 读取 parquet；
2. 截取 `[buffer_start, actual_end]`；
3. 实例化因子类 `fac_obj = factor_cls(df=df, symbol=code, **factor_kwargs)`；
4. 调用 `fac_obj.score()` 生成时序因子序列。

### 3.4 未来收益率计算
使用：
- 买入价：`close.shift(-1)`，即 **T+1 收盘买入**；
- 卖出价：`close.shift(-(h+1))`，即 **T+1+h 收盘卖出**；
- 收益率：`sell / buy - 1`。

因此 `h=1` 表示从 **T+1 收盘持有到 T+2 收盘** 的收益。

### 3.5 去掉预热区间
- 因子序列与未来收益序列最终都只保留 `actual_start` 及之后的样本。

---

## 4. 全市场宽表构建

### 4.1 并发处理
- `build_panels(...)` 使用 `ProcessPoolExecutor`；
- 默认按 50 只股票分一个 chunk；
- 每个子进程负责一批股票的：
  - 因子序列；
  - 多 horizon 未来收益序列；
  - 尾部缺失统计；
  - 错误信息与样本统计。

### 4.2 宽表合并
最终得到：

- `raw_factor_panel`：
  - index = 日期
  - columns = 股票代码
  - values = 原始因子值

- `ret_panels[h]`：
  - index = 日期
  - columns = 股票代码
  - values = horizon = h 的未来收益率

并保证每个 `ret_panels[h]` 和 `raw_factor_panel` 行索引对齐。

---

## 5. 截面预处理

### 5.1 MAD 去极值
对每日横截面：
- 用中位数作为中心；
- 用 `median ± 3.1483 * MAD` 裁剪极端值。

### 5.2 Z-score 标准化
对去极值后的每日横截面：
- 减均值；
- 除以标准差。

输出为 `factor_panel`。

说明：
- 这是**横截面标准化**，不是沿时间序列标准化；
- 该步骤用于减少极端值和量纲差异对 IC 的影响。

---

## 6. 因子秩自相关

- 先对每日横截面做 rank；
- 再将当日 rank 与 lag 日前 rank 做截面 Spearman 相关；
- 输出日度序列 `autocorr_series` 以及其均值。

解释：
- 越接近 1，说明因子排序更稳定；
- 越接近 0，说明横截面排序变化较大；
- 这是**组合换手难易程度的代理指标**，不是精确换手率。

---

## 7. 日度 IC 计算

对每个 horizon `h = 1..max_h`：

1. 取 `factor_panel.loc[dt]`；
2. 取 `ret_panels[h].loc[dt]`；
3. 删除缺失；
4. 若有效配对样本数 < 3，则该日 IC 记为 NaN；
5. 否则计算 Spearman 相关。

输出：

- `ic_df`：日度 IC 时间序列；
- `ic_valid_counts`：每个 horizon、每个交易日的有效配对样本数。

---

## 8. IC 汇总指标

对每个 `IC_h{h}` 计算：

- `N`：有效交易日数；
- `Mean IC`；
- `Std IC`；
- `ICIR (Ann)`：`Mean / Std * sqrt(252)`；
- `ICIR (Mon)`：`Mean / Std * sqrt(21)`；
- `IC WinRate`：IC > 0 的比例；
- `T-stat`：`Mean / (Std / sqrt(N))`；
- `Min IC` / `Max IC`。

最佳持有期定义为：

- `abs(Mean IC)` 最大的 horizon。

---

## 9. 分组收益分析

基于最佳持有期 `best_h`：

1. 将 `factor_panel` 和 `ret_panels[best_h]` 展开成 `(date, code)` 长表；
2. 每日按因子值排序；
3. 做等人数分组（默认 10 组）；
4. 计算每日每组平均收益；
5. 输出：
   - 各组日收益；
   - 各组样本数；
   - 多空组合（G10 - G1）日收益；
   - 多空 simple-sum 累计收益；
   - 多空复利累计收益。

说明：
- 为避免 `qcut` 在重复值或边界上的不稳定，采用基于 rank 的等人数分组；
- 若某日股票数量太少（默认不足 `groups * 3`），则该日不参与分组。

---

## 10. 输出文件

修改后的脚本会输出以下文件：

### 核心结果
- `daily_ic.csv`
- `ic_metrics_summary.csv`
- `daily_ic_valid_counts.csv`
- `ic_valid_count_summary.csv`
- `factor_autocorr_daily.csv`

### 分组结果
- `quantile_daily_group_returns.csv`
- `quantile_daily_group_counts.csv`
- `quantile_long_short.csv`

### 诊断信息
- `panel_summary.csv`
- `raw_factor_panel_head20.csv`
- `processed_factor_panel_head20.csv`
- `build_stats.csv`
- `build_errors.csv`

### 图像
- `cum_ic_topk.png`
- `rolling_ic_best_h.png`
- `quantile_cum_returns.png`
- `long_short_spread.png`

---

## 11. 控制台建议输出内容
为了便于排查问题，控制台至少应输出：

- 时间区间、buffer 区间；
- 因子名、因子参数；
- 股票文件总数与样例股票；
- 宽表 shape、日期范围、覆盖率；
- 每日有效股票数均值/分位数；
- 各持有期有效配对样本数；
- IC 指标表预览；
- 最佳持有期；
- 自相关统计；
- 分组收益预览；
- 多空收益统计；
- 报错股票数与示例报错。

---

## 12. 当前代码已实现 vs 后续可补充

### 已实现
- 多 horizon IC
- Mean IC / ICIR / WinRate / T-stat
- 因子秩自相关
- 十分组收益
- 多空收益
- 滚动 IC 图（最佳 horizon）
- 丰富的中间诊断输出
- IC decay 柱状图
### 建议后续补充
- 月度 / 年度 IC 热力图

- 按市场状态分段的 IC 分析（牛熊/高波动/低波动）
- 行业中性化 / 市值中性化                    


