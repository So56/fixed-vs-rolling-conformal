import math
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.10
WINDOW = 252

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs" / "modeling"


def conformal_halfwidth(scores, alpha):
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scoreは空でない1次元配列にしてください。")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("scoreに欠損や不正な値があります。")
    if not 0 < alpha < 1:
        raise ValueError("alphaは0と1の間にしてください。")

    # 有限標本補正を含む順位。標本数を超える場合は半幅を無限大にする。
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        return np.inf
    return np.sort(scores)[k - 1]


def build_rolling_widths(calibration, test):
    scores = calibration["score_abs"].tolist()
    dates = calibration["date"].tolist()
    records = []

    for row in test.itertuples(index=False):
        past_scores = scores[-WINDOW:]
        past_dates = dates[-WINDOW:]

        if len(past_scores) != WINDOW:
            raise ValueError("Rollingの履歴が不足しています。")
        if past_dates[-1] >= row.date:
            raise ValueError("当日以降の情報が履歴に混入しています。")

        # 当日のscoreを加える前に、過去252日で半幅を決める
        records.append({
            "date": row.date,
            "rolling_halfwidth": conformal_halfwidth(past_scores, ALPHA),
            "history_start": past_dates[0],
            "history_end": past_dates[-1],
            "history_n": len(past_scores),
        })

        scores.append(row.score_abs)
        dates.append(row.date)

    return pd.DataFrame(records)


# データの読み込み
predictions = pd.read_csv(
    output_dir / "point_predictions.csv",
    parse_dates=["date", "forecast_origin_date"],
).sort_values("date").reset_index(drop=True)

calibration = predictions[predictions["split"] == "calibration"].copy()
test = predictions[predictions["split"] == "test"].copy().reset_index(drop=True)

if len(calibration) != WINDOW or not calibration["date"].between(
    "2019-01-01", "2019-12-31"
).all():
    raise ValueError("Calibrationは2019年の252日分を使います。")

if test.empty or not test["date"].between("2020-01-01", "2026-10-28").all():
    raise ValueError("Testの期間を確認してください。")

if calibration["date"].max() >= test["date"].min():
    raise ValueError("CalibrationとTestの時間順を確認してください。")


# FixedとRollingの予測区間
q_fixed = conformal_halfwidth(calibration["score_abs"], ALPHA)
rolling = build_rolling_widths(calibration, test)

intervals = test.copy()
intervals["fixed_halfwidth"] = q_fixed
intervals["fixed_lower"] = intervals["prediction_pct"] - q_fixed
intervals["fixed_upper"] = intervals["prediction_pct"] + q_fixed

intervals = intervals.merge(rolling, on="date", how="left", validate="one_to_one")
intervals["rolling_lower"] = intervals["prediction_pct"] - intervals["rolling_halfwidth"]
intervals["rolling_upper"] = intervals["prediction_pct"] + intervals["rolling_halfwidth"]

columns = [
    "date", "fixed_halfwidth", "rolling_halfwidth",
    "history_start", "history_end", "history_n",
]
print("Fixedの半幅:", q_fixed)
print(intervals[columns].head(3).to_string(index=False))


# 半幅と時間順序の確認
np.testing.assert_allclose(
    intervals.loc[0, "rolling_halfwidth"], q_fixed, rtol=0, atol=1e-12
)

if not intervals["history_n"].eq(WINDOW).all():
    raise ValueError("Rollingの履歴件数が252ではありません。")

if not intervals["history_end"].eq(intervals["forecast_origin_date"]).all():
    raise ValueError("履歴の終了日が前市場日になっていません。")

# 11行目以降のscoreを変えても、11行目までの半幅は変わらない
cut = 10
changed_test = test.copy()
changed_test.loc[cut:, "score_abs"] += 100
changed_rolling = build_rolling_widths(calibration, changed_test)

np.testing.assert_allclose(
    rolling["rolling_halfwidth"].iloc[:cut + 1],
    changed_rolling["rolling_halfwidth"].iloc[:cut + 1],
    rtol=0, atol=0, equal_nan=False,
)
print("Rollingの時間順序の検証に合格しました。")


# CSV保存
intervals.to_csv(
    output_dir / "prediction_intervals.csv",
    index=False,
    date_format="%Y-%m-%d",
)

settings = pd.DataFrame([{
    "alpha": ALPHA,
    "rolling_window": WINDOW,
    "calibration_n": len(calibration),
    "test_n": len(test),
    "k": math.ceil((len(calibration) + 1) * (1 - ALPHA)),
    "fixed_halfwidth": q_fixed,
}])
settings.to_csv(output_dir / "interval_settings.csv", index=False)

print("予測区間の保存が完了しました。")
