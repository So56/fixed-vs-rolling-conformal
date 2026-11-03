from pathlib import Path

import pandas as pd

from evaluation import evaluate_intervals, summarize

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs" / "evaluation"

intervals = pd.read_csv(
    project_dir / "outputs" / "modeling" / "prediction_intervals.csv",
    parse_dates=["date"],
).sort_values("date").reset_index(drop=True)

if (
    len(intervals) != 1715
    or intervals["date"].duplicated().any()
    or intervals["date"].min() != pd.Timestamp("2020-01-02")
    or intervals["date"].max() != pd.Timestamp("2026-10-28")
):
    raise ValueError("Testの日付または件数を確認してください。")

daily = evaluate_intervals(intervals, alpha=0.10)
summary = summarize(daily, threshold=30)

output_dir.mkdir(parents=True, exist_ok=True)
daily.to_csv(output_dir / "daily_evaluation.csv", index=False, date_format="%Y-%m-%d")
summary.to_csv(output_dir / "group_summary.csv", index=False)

# 保存するCoverageは0〜1、画面表示はパーセント。
display = summary.copy()
display["coverage"] *= 100
print(display.round(4).to_string(index=False))
