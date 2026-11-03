from pathlib import Path

import numpy as np
import pandas as pd
import exchange_calendars as xcals


# フォルダと期間
project_dir = Path(__file__).resolve().parent.parent
raw_dir = project_dir / "data" / "raw"

RAW_START = "2016-10-31"
RAW_END = "2026-10-28"


# データの読み込み
sp500 = pd.read_csv(
    raw_dir / "SP500.csv", na_values=".",
    parse_dates=["observation_date"]
)
vix = pd.read_csv(
    raw_dir / "VIXCLS.csv", na_values=".",
    parse_dates=["observation_date"]
)

sp500 = sp500.rename(columns={
    "observation_date": "date",
    "SP500": "sp500_close"
})
vix = vix.rename(columns={
    "observation_date": "date",
    "VIXCLS": "vix_close"
})


# S&P 500の取引日にVIXを結合
market = sp500.dropna(subset=["sp500_close"]).merge(
    vix, on="date", how="left", validate="one_to_one"
)
market = market.sort_values("date").reset_index(drop=True)


# データの確認
if (market["sp500_close"] <= 0).any():
    raise ValueError("S&P 500の終値に異常値があります。")

if market["vix_close"].isna().any():
    raise ValueError("VIXに欠損値があります。")

if (market["sp500_close"] <= 0).any():
    raise ValueError("S&P 500の終値に0以下の値があります。")

if market["vix_close"].isna().any():
    raise ValueError("取引日のVIXに欠損があります。")


# 日次対数リターン（%）
market["return_pct"] = 100 * np.log(market["sp500_close"]).diff()

# 予測時点で分かる情報だけを使うため、説明変数を1取引日ずらす。
market["return_lag1_pct"] = market["return_pct"].shift(1)
market["vix_lag1"] = market["vix_close"].shift(1)
market["forecast_origin_date"] = market["date"].shift(1)


# 分析期間
analysis = market.loc[market["date"] >= "2017-01-01"].copy()

# 前取引日のVIXが30以上ならHigh-VIX
analysis["high_vix_30"] = analysis["vix_lag1"] >= 30


# データの分割
analysis["split"] = "training"
analysis.loc[analysis["date"] >= "2019-01-01", "split"] = "calibration"
analysis.loc[analysis["date"] >= "2020-01-01", "split"] = "test"


# 期間別の集計
summary = analysis.groupby("split").agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    n=("date", "size"),
    high_vix_days=("high_vix_30", "sum")
).reindex(["training", "calibration", "test"])

print(summary.to_string())


# 境界の確認
boundary = analysis.loc[
    analysis["date"].between("2019-12-27", "2020-01-07"),
    ["date", "forecast_origin_date", "return_pct",
     "return_lag1_pct", "vix_lag1", "split"]
]


# CSVの保存
processed_dir = project_dir / "data" / "processed"
report_dir = project_dir / "reports"

processed_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

analysis.to_csv(processed_dir / "analysis_dataset.csv", index=False)
summary.to_csv(report_dir / "split_summary.csv")
boundary.to_csv(report_dir / "year_boundary_check.csv", index=False)

print("前処理完了")
