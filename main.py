import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
# main.py
from config import (
    DATA_SOURCE,
    EXCEL_PATH,
    EXCEL_PRICE_SHEET,
    EXCEL_FACTOR_SHEET,
    DATE_COL,
    PRICE_COL,
    SPLIT_RATIO,
)

from data import load_data
# from data import my_api_fetcher  # 如果你用 API
# =========================
# 0) 工具函数：绩效指标
# =========================
def perf_stats(strategy_ret: pd.Series, freq: int = 252) -> dict:
    strategy_ret = strategy_ret.dropna()
    if len(strategy_ret) == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    equity = (1 + strategy_ret).cumprod()
    ann_return = equity.iloc[-1] ** (freq / len(strategy_ret)) - 1
    ann_vol = strategy_ret.std(ddof=1) * np.sqrt(freq)
    sharpe = np.nan if ann_vol == 0 else (strategy_ret.mean() * freq) / ann_vol

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_days": int(len(strategy_ret)),
    }

# =========================
# 1️⃣ 数据读取（只看 config）
# =========================

if DATA_SOURCE == "demo":
    fut_ret, factors = load_data(source="demo")

elif DATA_SOURCE == "excel":
    fut_ret, factors = load_data(
        source="excel",
        excel_path=EXCEL_PATH,
        excel_price_sheet=EXCEL_PRICE_SHEET,
        excel_factor_sheet=EXCEL_FACTOR_SHEET,
        date_col=DATE_COL,
        price_col=PRICE_COL,
    )

elif DATA_SOURCE == "api":
    fut_ret, factors = load_data(
        source="api",
        api_fetcher=my_api_fetcher,
        date_col=DATE_COL,
        price_col=PRICE_COL,
    )

else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

dates = fut_ret.index

# =========================
# 2️⃣ 后面接你原来的策略流程
# =========================
# factors = factors.shift(1)
# 信号 → pos → pos_exec → 回测

# =========================
# 2) 防未来函数：t日信号只能用t-1已知信息
# =========================
factors = factors.shift(1)

# =========================
# 3) 因子预处理：平滑 + 滚动Z分数标准化
# =========================
def rolling_zscore(s: pd.Series, win: int = 252) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std(ddof=0)
    return (s - mu) / sd

# 平滑（可选）：EWMA降噪
f_smooth = factors.ewm(span=10, adjust=False).mean()

# 标准化：用滚动z，避免用全样本均值方差造成信息泄露
f_z = f_smooth.apply(lambda x: rolling_zscore(x, win=252))

# =========================
# 4) 单因子 -> 三值信号（-1/0/1），分位数阈值
# =========================
def factor_to_signal(z: pd.Series, q: float = 0.7, direction: int = 1) -> pd.Series:
    # direction=1: 越大越看多；direction=-1: 越小越看多
    z2 = z * direction

    # 更严格：阈值滞后一天，避免用到当日信息
    hi = z2.rolling(252).quantile(q).shift(1)
    lo = z2.rolling(252).quantile(1 - q).shift(1)

    sig = pd.Series(0, index=z2.index, dtype=float)
    sig[z2 > hi] = 1
    sig[z2 < lo] = -1
    return sig

direction_map = {
    "liq_r007": -1,
    "tech_mom20": +1,
    "spread_10y_1y": +1,
}

signals = pd.DataFrame(
    {c: factor_to_signal(f_z[c], q=0.7, direction=direction_map.get(c, 1)) for c in f_z.columns},
    index=dates
)

# =========================
# 5) 多因子等权合成 -> 最终交易信号
# =========================
combined = signals.mean(axis=1)

# 合成信号映射到仓位：>0 多，<0 空，=0 空仓
pos = pd.Series(0, index=dates, dtype=float)
pos[combined > 0] = 1
pos[combined < 0] = -1

# =========================
# 6) 回测：用 t 日仓位赚 t+1 日收益（所以仓位再 shift(1)）
# =========================
pos_exec = pos.shift(1).fillna(0)

# 交易成本：每次换仓收 cost（非常简化）
cost_per_turnover = 0.0002  # 2bp级别示例，你可按国债期货实际改
turnover = pos_exec.diff().abs().fillna(0)  # 0->1 算1，1->-1算2
cost = turnover * cost_per_turnover

strategy_ret = pos_exec * fut_ret - cost
strategy_ret.name = "strategy_ret"

# =========================
# 7) 样本内/外切分 & 指标输出
# =========================
split_date = dates[int(len(dates) * 0.6)]
in_sample = strategy_ret.loc[:split_date]
out_sample = strategy_ret.loc[split_date + pd.Timedelta(days=1):]

print("=== IN-SAMPLE ===")
print(perf_stats(in_sample))

print("\n=== OUT-OF-SAMPLE ===")
print(perf_stats(out_sample))

# 额外：看下胜率（样本外）
oos = out_sample.dropna()
win_rate = (oos > 0).mean() if len(oos) else np.nan
print(f"\nOOS WinRate: {win_rate:.3f}")

# 如果你想快速看最后的累计净值（不画图也能看）
equity_oos = (1 + out_sample.fillna(0)).cumprod()
print("\nOOS Equity (tail):")
print(equity_oos.tail())

summary = pd.DataFrame.from_dict(
    {
        "InSample": perf_stats(in_sample),
        "OutSample": perf_stats(out_sample),
    },
    orient="index"
)

print("\n=== Strategy Summary ===")
print(summary.round(4))

#净值曲线
equity = (1 + strategy_ret.fillna(0)).cumprod()
equity_in = equity.loc[:split_date]
equity_out = equity.loc[split_date:]

plt.figure(figsize=(10, 5))
plt.plot(equity_in, label="In-Sample")
plt.plot(equity_out, label="Out-of-Sample")
plt.axvline(split_date, linestyle="--", alpha=0.7, label="Split")
plt.title("Strategy Equity Curve")
plt.legend()
plt.grid(True)
plt.show()

#回撤曲线
rolling_max = equity.cummax()
drawdown = equity / rolling_max - 1

plt.figure(figsize=(10, 4))
plt.plot(drawdown, color="red")
plt.title("Drawdown")
plt.grid(True)
plt.show()

#多空仓位可视化
plt.figure(figsize=(10, 3))
plt.plot(pos, label="Position")
plt.title("Position (1=Long, -1=Short)")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()

#Web 可视化：Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity"))

# 竖线（跨全图高度：yref="paper", 0~1）
x_split = pd.to_datetime(split_date).to_pydatetime()

fig.add_shape(
    type="line",
    x0=x_split, x1=x_split,
    y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(dash="dash", width=1)
)

# 标注（可选）
fig.add_annotation(
    x=x_split, y=1, yref="paper",
    text="Split", showarrow=False,
    xanchor="left", yanchor="top"
)

fig.update_layout(title="Strategy Equity Curve", template="plotly_white")

# 有些 PyCharm 环境下建议用浏览器渲染
fig.show(renderer="browser")