import math
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import STAT_COLUMNS, bootstrap_ci, evaluate_intervals, summarize

ALPHA = 0.10
REPEATS = 5000
SEED = 202701

project_dir = Path(__file__).resolve().parent.parent
modeling_dir = project_dir / "outputs" / "modeling"
evaluation_dir = project_dir / "outputs" / "evaluation"
main_dir = evaluation_dir / "bootstrap"
output_dir = evaluation_dir / "sensitivity"


def rolling_halfwidths(calibration_data, test_data, window):
    if not 1 <= window <= len(calibration_data):
        raise ValueError("初期Calibrationの件数を確認してください。")

    scores = calibration_data["score_abs"].tolist()
    last_date = calibration_data["date"].iloc[-1]
    widths = []
    # 主分析と同じ有限標本補正を、各窓幅に適用する。
    k = math.ceil((window + 1) * (1 - ALPHA))

    for row in test_data.itertuples(index=False):
        if last_date >= row.date:
            raise ValueError("当日以降の情報が混ざっています。")

        past_scores = scores[-window:]
        if k > window:
            q = np.inf
        else:
            q = np.sort(past_scores)[k - 1]
        widths.append(q)

        # 当日の誤差は、その日の区間を作った後で追加する。
        scores.append(row.score_abs)
        last_date = row.date

    return np.asarray(widths)


base = pd.read_csv(modeling_dir / "prediction_intervals.csv", parse_dates=["date"])
base = base.sort_values("date").reset_index(drop=True)
predictions = pd.read_csv(modeling_dir / "point_predictions.csv", parse_dates=["date"])
predictions = predictions.sort_values("date").reset_index(drop=True)

calibration = predictions.loc[predictions["split"] == "calibration"]
test = predictions.loc[predictions["split"] == "test"].reset_index(drop=True)
if not test["date"].equals(base["date"]):
    raise ValueError("点予測と区間の日付が一致しません。")

daily_by_window = {}
for window in [63, 126, 252]:
    halfwidths = rolling_halfwidths(calibration, test, window)
    if window == 252:
        np.testing.assert_allclose(
            halfwidths, base["rolling_halfwidth"], rtol=0, atol=1e-12
        )

    # 252日用の履歴情報を、別のwindowの結果に引き継がない。
    frame = base[[
        "date", "return_pct", "vix_lag1",
        "prediction_pct", "fixed_lower", "fixed_upper",
    ]].copy()
    frame["rolling_window"] = window
    frame["rolling_lower"] = frame["prediction_pct"] - halfwidths
    frame["rolling_upper"] = frame["prediction_pct"] + halfwidths
    daily_by_window[window] = evaluate_intervals(frame, ALPHA)

# 主設定は再計算せず、06_bootstrap_confidence_intervals.pyの保存結果を使う。
main_settings = pd.read_csv(main_dir / "settings.csv").iloc[0]
if (
    main_settings["repeats"] != REPEATS
    or main_settings["seed"] != SEED
    or main_settings["block_length"] != 10
    or main_settings["threshold"] != 30
):
    raise ValueError("主設定の保存済みBootstrap結果を確認してください。")

main_daily = pd.read_csv(evaluation_dir / "daily_evaluation.csv")
np.testing.assert_allclose(
    main_daily[STAT_COLUMNS].to_numpy(),
    daily_by_window[252][STAT_COLUMNS].to_numpy(),
    rtol=0, atol=1e-12,
)
main_starts = pd.read_csv(main_dir / "block_starts.csv")
main_ci = pd.read_csv(main_dir / "confidence_intervals.csv").assign(
    case="main", window=252, threshold=30, block_length=10,
    repeats=REPEATS, seed=SEED,
)

all_results = [main_ci]
cases = [
    ("block_5", 252, 30, 5),
    ("block_20", 252, 30, 20),
    ("vix_25", 252, 25, 10),
    ("vix_35", 252, 35, 10),
    ("window_63", 63, 30, 10),
    ("window_126", 126, 30, 10),
]

for name, window, threshold, block_length in cases:
    print("実行中:", name)
    daily = daily_by_window[window]
    ci, replicates, starts = bootstrap_ci(
        daily, block_length=block_length, repeats=REPEATS,
        seed=SEED, threshold=threshold,
    )

    # 同じブロック長なら、同じ日付の組合せで比較する。
    if block_length == 10:
        pd.testing.assert_frame_equal(starts, main_starts)

    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(case_dir / "daily_evaluation.csv", index=False, date_format="%Y-%m-%d")
    summarize(daily, threshold).to_csv(case_dir / "group_summary.csv", index=False)
    ci = ci.assign(
        case=name, window=window, threshold=threshold,
        block_length=block_length, repeats=REPEATS, seed=SEED,
    )
    ci.to_csv(case_dir / "confidence_intervals.csv", index=False)
    replicates.to_csv(case_dir / "replicates.csv", index=False)
    starts.to_csv(case_dir / "block_starts.csv", index=False)
    all_results.append(ci)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv(output_dir / "all_confidence_intervals.csv", index=False)

display = combined.loc[
    (combined["group"] == "high_vix") & (combined["method"] == "difference")
].copy()
display.loc[
    display["metric"] == "coverage", ["estimate", "lower", "upper"]
] *= 100
print(display[[
    "case", "metric", "n", "estimate", "lower", "upper",
    "valid_reps", "invalid_reps",
]].round(4).to_string(index=False))
