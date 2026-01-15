# data.py
import numpy as np
import pandas as pd

# =========================
# 统一数据入口
# =========================
def load_data(
    source: str = "demo",                 # "demo" | "excel" | "api" | "dataframe"
    excel_path: str | None = None,
    excel_price_sheet: str = "price",
    excel_factor_sheet: str = "factors",
    date_col: str = "date",              # 这里指定日期列的名字，后续转换时会用
    price_col: str = "close",            # 同理设定价格列的名字
    factor_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
    api_fetcher=None,                     # 传入一个函数：api_fetcher() -> (price_df, factor_df)
) -> tuple[pd.Series, pd.DataFrame]:
    """
    返回：
      fut_ret: pd.Series (index=DatetimeIndex)
      factors: pd.DataFrame (index=DatetimeIndex)
    """

    if source == "demo":
        # 仍然保留你原来的模拟数据逻辑（后面我给你放进去）
        return make_demo_data()

    if source == "excel":
        if excel_path is None:
            raise ValueError("source='excel' 时必须提供 excel_path")

        excel_path = str(Path(excel_path))
        p = pd.read_excel(excel_path, sheet_name=excel_price_sheet)
        f = pd.read_excel(excel_path, sheet_name=excel_factor_sheet)

        # 日期列转 datetime
        p[date_col] = pd.to_datetime(p[date_col])
        f[date_col] = pd.to_datetime(f[date_col])

        # 设置 index
        p = p.set_index(date_col).sort_index()
        f = f.set_index(date_col).sort_index()

        # 价格 -> 日收益率
        if price_col not in p.columns:
            raise ValueError(f"price sheet 缺少列: {price_col}")
        fut_ret = p[price_col].pct_change().rename("fut_ret")

        # factors：保留除日期列外所有列
        factors = f.copy()
        return fut_ret, factors

    if source == "dataframe":
        if price_df is None or factor_df is None:
            raise ValueError("source='dataframe' 时必须同时传入 price_df 和 factor_df")

        # 约定：price_df 有 date_col 和 price_col 或者 index 就是日期
        p = price_df.copy()
        f = factor_df.copy()

        if date_col in p.columns:
            p[date_col] = pd.to_datetime(p[date_col])
            p = p.set_index(date_col)
        p = p.sort_index()

        if date_col in f.columns:
            f[date_col] = pd.to_datetime(f[date_col])
            f = f.set_index(date_col)
        f = f.sort_index()

        fut_ret = p[price_col].pct_change().rename("fut_ret")
        factors = f
        return fut_ret, factors

    if source == "api":
        if api_fetcher is None:
            raise ValueError("source='api' 时必须提供 api_fetcher 函数")

        p, f = api_fetcher()  # 你自己实现：返回两个 DataFrame
        return load_data(source="dataframe", price_df=p, factor_df=f, date_col=date_col, price_col=price_col)

    raise ValueError(f"未知source: {source}")

# =========================
# 定义随机生成数据的函数
# =========================
def make_demo_data(
    n: int = 2000,
    start: str = "2015-01-01",
    seed: int = 42,
):
    """
    生成可跑通的模拟日频数据（用于 demo / 调试）

    返回：
        fut_ret : pd.Series
            模拟的期货日收益率（index=DatetimeIndex）
        factors : pd.DataFrame
            模拟的因子数据（index=DatetimeIndex）
    """

    # =========================
    # 基础设置
    # =========================
    np.random.seed(seed)
    dates = pd.bdate_range(start=start, periods=n)

    # =========================
    # 模拟期货日收益率（可理解为10年国债期货主力复权收益）
    # =========================
    base_noise = np.random.normal(loc=0, scale=0.004, size=n)

    trend = (
        pd.Series(
            np.random.normal(loc=0, scale=0.0002, size=n),
            index=dates
        )
        .rolling(50)
        .mean()
        .fillna(0)
        .values
    )

    fut_ret = base_noise + trend
    fut_ret = pd.Series(fut_ret, index=dates, name="fut_ret")

    # =========================
    # 模拟因子（6 个示例）
    # =========================
    factors = pd.DataFrame(index=dates)

    factors["liq_r007"] = (
        pd.Series(np.random.normal(0, 1, n), index=dates)
        .rolling(5)
        .mean()
    )

    factors["liq_dr007"] = (
        pd.Series(np.random.normal(0, 1, n), index=dates)
        .rolling(10)
        .mean()
    )

    factors["spread_10y_1y"] = (
        pd.Series(np.random.normal(0, 1, n), index=dates)
        .rolling(20)
        .mean()
    )

    factors["spread_gz_gk"] = (
        pd.Series(np.random.normal(0, 1, n), index=dates)
        .rolling(20)
        .mean()
    )

    factors["fut_oi_chg"] = (
        pd.Series(np.random.normal(0, 1, n), index=dates)
        .rolling(5)
        .mean()
    )

    # 技术因子示例：20 日动量
    factors["tech_mom20"] = fut_ret.rolling(20).sum()

    # =========================
    # 4️⃣ 给部分因子注入“预测性”（仅用于演示）
    #    ⚠️ 真实研究中不要这样做
    # =========================
    factors["spread_10y_1y"] = factors["spread_10y_1y"] + fut_ret.shift(-1) * 30
    factors["liq_r007"] = factors["liq_r007"] - fut_ret.shift(-1) * 20

    return fut_ret, factors
