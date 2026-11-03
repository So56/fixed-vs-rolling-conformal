
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exchange_calendars as xcals

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs" / "modeling"
figure_dir = output_dir / "figures"

# データの読み込み
data = pd.read_csv(
    project_dir / "data" / "processed" / "analysis_dataset.csv",
    parse_dates=["date", "forecast_origin_date"],
).sort_values("date").reset_index(drop=True)

# データの確認
required_columns = ["return_pct", "return_lag1_pct", "vix_lag1"]

if not np.isfinite(data[required_columns].to_numpy()).all():
    raise ValueError("分析に必要な値に欠損値または無限大が含まれています。")

if (data["date"].min() != pd.Timestamp("2017-01-03")
        or data["date"].max() != pd.Timestamp("2026-10-28")):
    raise ValueError("分析期間を確認してください。")

print(data.groupby("split").size())

# NYSEの取引日との照合
raw_prices = pd.read_csv(
    project_dir / "data" / "raw" / "SP500.csv",
    na_values=["."],
    parse_dates=["observation_date"],
)

actual_dates = pd.DatetimeIndex(
    raw_prices.loc[raw_prices["SP500"].notna(), "observation_date"]
)

calendar = xcals.get_calendar(
    "XNYS", start="2016-10-31", end="2026-10-28"
)
expected_dates = calendar.sessions_in_range(
    "2016-10-31", "2026-10-28"
).tz_localize(None)

missing_dates = expected_dates.difference(actual_dates)
unexpected_dates = actual_dates.difference(expected_dates)

print("欠けている取引日:", missing_dates.tolist())
print("取引日以外の観測:", unexpected_dates.tolist())

if not actual_dates.is_unique or not missing_dates.empty or not unexpected_dates.empty:
    raise ValueError("取引日との不一致または日付の重複があります。")

# 記述統計量
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

stats = data.groupby("split")["return_pct"].agg(
    ["count", "mean", "std", "min", "max"]
).reindex(["training", "calibration", "test"])

yearly = data.assign(year=data["date"].dt.year).groupby("year").agg(
    n=("date", "size"),
    high_vix_days=("high_vix_30", "sum"),
)

print(stats.to_string())
print(yearly.to_string())

stats.to_csv(output_dir / "descriptive_stats.csv")
yearly.to_csv(output_dir / "high_vix_by_year.csv")

# グラフ作成
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(data["date"], data["return_pct"], linewidth=0.6)
axes[0].axhline(0, color="gray", linewidth=0.8)
axes[0].set_ylabel("Return (%)")
axes[0].set_title("S&P 500 daily log returns")

axes[1].plot(data["date"], data["vix_lag1"], linewidth=0.8)
axes[1].axhline(
    30, color="red", linestyle="--", label="High-VIX threshold"
)
axes[1].set_ylabel("Previous-session VIX")
axes[1].legend()

axes[2].hist(data["return_pct"], bins=80, edgecolor="white")
axes[2].set_xlabel("Return (%)")
axes[2].set_ylabel("Number of days")

fig.tight_layout()
fig.savefig(figure_dir / "returns_vix_distribution.png", dpi=150)
plt.close(fig)

print("データ確認とグラフ作成が完了しました。")
