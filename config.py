# config.py
# =========================
# 数据源配置
# =========================

DATA_SOURCE = "demo"
# 可选: "demo" | "excel" | "dataframe" | "api"

# =========================
# Excel 配置（仅当 DATA_SOURCE="excel" 时用）
# =========================

EXCEL_PATH = "your_data.xlsx"
EXCEL_PRICE_SHEET = "price"
EXCEL_FACTOR_SHEET = "factors"

DATE_COL = "date"
PRICE_COL = "close"

# =========================
# 回测参数（你以后也可以放这里）
# =========================

SPLIT_RATIO = 0.6
