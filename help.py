# 块0 ：导入库 + 绩效指标函数

#计算净值曲线
equity = (1 + strategy_ret).cumprod()
#1 + strategy_ret：把“收益率”变成“每天资金乘数”，cumprod()：累计连乘

#年化收益率（复利年化）
ann_return = equity.iloc[-1] ** (freq / len(strategy_ret)) - 1
# equity.iloc[-1]：回测结束时的净值
# 复利年化收益率公式

# 年化波动率
ann_vol = strategy_ret.std(ddof=1) * np.sqrt(freq)
# strategy_ret.std(ddof=1)：日收益的样本标准差（ddof=1 是样本标准差）
# 日收益率的方差 × 一年的交易日数 = 年化方差 = 年化波动率的平方

#夏普比率
sharpe = np.nan if ann_vol == 0 else (strategy_ret.mean() * freq) / ann_vol

#最大回撤
rolling_max = equity.cummax() # equity.cummax()：历史最高净值
drawdown = equity / rolling_max - 1
max_dd = drawdown.min()


# ===== 数据来源：DEMO（系统随机/模拟数据）=====
fut_ret, factors = load_data(source="demo")
dates = fut_ret.index
# ===== 数据来源：EXCEL =====
fut_ret, factors = load_data(
    source="excel",
    excel_path="your_data.xlsx",
    excel_price_sheet="price",
    excel_factor_sheet="factors",
    date_col="date",
    price_col="close",
)
dates = fut_ret.index
# ===== 数据来源：DATAFRAME（内存变量）=====
fut_ret, factors = load_data(
    source="dataframe",
    price_df=price_df,
    factor_df=factor_df,
    date_col="date",
    price_col="close",
)
dates = fut_ret.index
# ===== 数据来源：API =====
def my_api_fetcher():
    # TODO: 这里写你真实的取数逻辑（HTTP / DB 都行）
    # 必须返回两个 DataFrame：price_df, factor_df
    return price_df, factor_df

fut_ret, factors = load_data(
    source="api",
    api_fetcher=my_api_fetcher,
    date_col="date",
    price_col="close",
)
dates = fut_ret.index




#块1 ：模拟数据生成

#设定种子
np.random.seed(42)
n = 2000
dates = pd.bdate_range("2015-01-01", periods=n) # bdate_range 不知道中国法定节假日，只跳过周末。演示够用，实盘要用交易所日历。

#生成期货日收益
base_noise = np.random.normal(0, 0.004, size=n) # 生成 n 个正态随机数，均值 0，标准差 0.004
trend = pd.Series(np.random.normal(0, 0.0002, size=n)).rolling(50).mean().fillna(0).values
# np.random.normal(0, 0.0002, size=n)：生成一个更小的噪声（0.02%级别）
# pd.Series(...).rolling(50).mean()：做 50 日滚动均值
# .fillna(0)：前 49 天滚动均值没法算，会是 NaN，这里填 0
# .values：把 Series 转成 numpy 数组，后面好相加
fut_ret = base_noise + trend
fut_ret = pd.Series(fut_ret, index=dates, name="fut_ret") # 后面回测要用日期对齐、rolling，所以必须是Series，而不是纯 numpy 数组

# 生成 6 个因子
factors = pd.DataFrame(index=dates) # 创建一个空表，先把日期索引铺好，之后每加一列因子，都会自动按日期对齐
factors["liq_r007"] = pd.Series(np.random.normal(0, 1, n), index=dates).rolling(5).mean()
# factors 是一个 DataFrame
# 行是日期（index=dates）
# 列是不同因子（资金、利差、持仓、技术）
factors["tech_mom20"] = fut_ret.rolling(20).sum()  # 20日动量（技术类示例）


# 块2 ：防未来函数
factors = factors.shift(1)
# 只要是rolling，就必须做防未来函数，一般都是shift（1）就可以了，用2会损失数据


# 块 3：因子预处理（平滑 + 标准化）

# 指数加权平滑（EWMA）
def rolling_zscore(s: pd.Series, win: int = 252) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0) # 总体标准差，ddof=1是样本标准差
    return (s - mu) / sd

f_smooth = factors.ewm(span=10, adjust=False).mean()
# ewm：指数加权移动平均，给近期数据赋予更高权重，远期数据赋予呈指数衰减的较低权重
# span=10：用大概最近10 天为主的信息，平滑掉单日跳动
# adjust=False：保证用递推形式进行ewm
f_z = f_smooth.apply(lambda x: rolling_zscore(x, win=252))

# 滚动 Z-score 标准化


# 块4 ：三值信号生成（分位数 → -1/0/+1）
def factor_to_signal(z: pd.Series, q: float = 0.7, direction: int = 1) -> pd.Series:
    z2 = z * direction # direction=1: 大->多；direction=-1: 小->多
    hi = z2.rolling(252).quantile(q).shift(1) # 对每一天 t，取过去 252 天（约 1 年）的 z 值分布，计算它的 q 分位点（例如 0.7 分位）
    lo = z2.rolling(252).quantile(1 - q).shift(1) # 过去一年里，因子处于较低位置的分界线

    sig = pd.Series(0, index=z2.index, dtype=float) # 先把所有日期的信号设为 0，再根据条件改成 1 或 -1
    sig[z2 > hi] = 1
    sig[z2 < lo] = -1
    return sig
# 确定因子方向
direction_map = {
    "liq_r007": -1,          # 假设：资金利率越高越利空（越小越好）
    "tech_mom20": +1,        # 动量越大越好
    "spread_10y_1y": +1,     # 假设利差越大越好（仅示例）
}


signals = pd.DataFrame({c: factor_to_signal(f_z[c], q=0.7, direction = direction_map.get(c, 1)) for c in f_z.columns},
                       index=dates) # 生成所有因子的 signals 表

# 当 q=0.7：z > 70%分位 → 看多，z < 30%分位 → 看空，中间 40% 区间 → 中性，所以每个因子大概会产生：~30% 的多信号，~30% 的空信号
#   ~40% 的空仓/观望。如果你把 q 调大（比如 0.8）：信号更少、更“极端”，换手更低，可能更稳，但机会更少。如果 q 调小（比如 0.6）：
#   信号更频繁，换手更高，更容易被噪声影响，这也是实习讨论很爱聊的点：阈值越极端，越“少做但更确定”。


# 块 5 ：多因子等权合成 → 最终仓位信号
combined = signals.mean(axis=1)
# axis = 1 ：按行计算每一天各因子的平均值
pos = pd.Series(0, index=dates, dtype=float) # 建一个全 0 的仓位序列
pos[combined > 0] = 1 #说明大多数因子投票偏多
pos[combined < 0] = -1 #说明大多数因子投票偏空


# 块 6：回测执行
pos_exec = pos.shift(1).fillna(0) # fillna（0）：直接把NaN替换为0，表示回测第一天空仓

cost_per_turnover = 0.0002
turnover = pos_exec.diff().abs().fillna(0)
# pos_exec.diff()：今天仓位 − 昨天仓位， .abs()：只关心变动大小，不关心方向，.fillna(0)：第一天没变化
cost = turnover * cost_per_turnover

strategy_ret = pos_exec * fut_ret - cost
strategy_ret.name = "strategy_ret"


# 块 7：样本内/外切分 + 绩效输出
split_date = dates[int(len(dates) * 0.6)]
in_sample = strategy_ret.loc[:split_date] #.loc[:split_date]：取从最开始到 split_date（包含 split_date）的所有行
out_sample = strategy_ret.loc[split_date + pd.Timedelta(days=1):]
# pd.Timedelta(days=1)：时间增量 1 天，split_date + ...：把切分点往后推 1 天，.loc[...:]：从那天到最后
print("=== IN-SAMPLE ===") # 输出样本内指标
print(perf_stats(in_sample))

print("\n=== OUT-OF-SAMPLE ===") # 输出样本外指标
print(perf_stats(out_sample))

oos = out_sample.dropna()
# (oos > 0)：把每天是否赚钱变成 True/False
win_rate = (oos > 0).mean() if len(oos) else np.nan
# .mean()：在布尔数组里，True=1，False=0，所以平均值就是胜率，if len(oos) else np.nan：防止空样本时报错
# np.nan 表示“这个位置现在没有一个有效数值”。仍然是float类型的数据
print(f"\nOOS WinRate: {win_rate:.3f}")

equity_oos = (1 + out_sample.fillna(0)).cumprod()
print("\nOOS Equity (tail):")
print(equity_oos.tail()) # .tail()：只看最后 5 行


