from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import STAT_COLUMNS, summarize


def show_differences(results, index_column):
    selected = results.loc[
        (results["group"] == "high_vix") & (results["method"] == "difference"),
        [index_column, "n", "coverage", "width", "interval_score"],
    ].copy()
    selected["coverage"] *= 100
    print(selected.round(4).to_string(index=False))


def sample_acf(values, max_lag):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) <= max_lag or not np.isfinite(values).all():
        raise ValueError("ACFの入力を確認してください。")

    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    if denominator == 0:
        return np.full(max_lag + 1, np.nan)

    # 各ラグの積和を、全期間の偏差平方和で割る。
    return np.array([
        np.dot(centered[lag:], centered[:len(values) - lag]) / denominator
        for lag in range(max_lag + 1)
    ])


project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs" / "reporting"
figure_dir = output_dir / "figures"

daily = pd.read_csv(
    project_dir / "outputs" / "evaluation" / "daily_evaluation.csv",
    parse_dates=["date"],
).sort_values("date").reset_index(drop=True)

if (
    len(daily) != 1715
    or daily["date"].duplicated().any()
    or not np.isfinite(daily[STAT_COLUMNS].to_numpy()).all()
):
    raise ValueError("日次評価データを確認してください。")

figure_dir.mkdir(parents=True, exist_ok=True)
daily["year"] = daily["date"].dt.year
high = daily["vix_lag1"] >= 30

# 年ごとの集計
annual_results = []
for year, part in daily.groupby("year"):
    result = summarize(part, threshold=30)
    result["year"] = year
    result["first_date"] = part["date"].min()
    result["last_date"] = part["date"].max()
    annual_results.append(result)

annual = pd.concat(annual_results, ignore_index=True)
annual.to_csv(output_dir / "annual_summary.csv", index=False, date_format="%Y-%m-%d")

counts = daily.groupby("year").agg(n=("date", "size"))
counts["high_vix_days"] = high.groupby(daily["year"]).sum()
counts.to_csv(output_dir / "annual_counts.csv")
print(counts.to_string())
show_differences(annual, "year")

# 高VIXが始まる行で番号を1つ増やす
starts = high & ~high.shift(1, fill_value=False)
daily["episode_id"] = starts.cumsum()

episode_results = []
for episode_id, part in daily.loc[high].groupby("episode_id"):
    result = summarize(part, threshold=30)
    result = result.loc[result["group"] == "high_vix"].copy()
    result["episode_id"] = episode_id
    result["first_date"] = part["date"].min()
    result["last_date"] = part["date"].max()
    episode_results.append(result)

episodes = pd.concat(episode_results, ignore_index=True)
episodes.to_csv(
    output_dir / "high_vix_episodes.csv", index=False, date_format="%Y-%m-%d"
)
print("高VIXの連続期間数:", daily.loc[high, "episode_id"].nunique())

# 各年を1年ずつ除き、特定の年に結果が偏っていないか確認する
exclusion_results = []
for year in sorted(daily["year"].unique()):
    result = summarize(daily.loc[daily["year"] != year], threshold=30)
    result["excluded_year"] = year
    exclusion_results.append(result)

exclusions = pd.concat(exclusion_results, ignore_index=True)
exclusions.to_csv(output_dir / "leave_one_year_out.csv", index=False)
show_differences(exclusions, "excluded_year")

# 日次系列の自己相関
daily["high_vix_indicator"] = high.astype(float)
diagnostic_columns = [
    "difference_coverage",
    "difference_interval_score",
    "high_vix_indicator",
    "rolling_halfwidth",
]
max_lag = 300
acf_table = pd.DataFrame({"lag": np.arange(max_lag + 1)})
for column in diagnostic_columns:
    acf_table[column] = sample_acf(daily[column], max_lag)
acf_table.to_csv(output_dir / "acf_diagnostics.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, column in zip(axes.flat, diagnostic_columns):
    ax.plot(acf_table["lag"].iloc[1:], acf_table[column].iloc[1:], linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(column)
    ax.set_xlabel("Lag (market days)")
    ax.set_ylabel("ACF")

fig.tight_layout()
fig.savefig(figure_dir / "acf_diagnostics.png", dpi=200)
fig.savefig(figure_dir / "acf_diagnostics.pdf")
plt.close(fig)
