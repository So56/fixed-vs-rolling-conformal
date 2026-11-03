
from pathlib import Path
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs" / "modeling"

data = pd.read_csv(
    project_dir / "data" / "processed" / "analysis_dataset.csv",
    parse_dates=["date", "forecast_origin_date"],
).sort_values("date")

# TrainingデータでOLSを推定
train = data[data["split"] == "training"]
x = train["return_lag1_pct"]
y = train["return_pct"]

X = np.column_stack([np.ones(len(train)), x])
beta0, beta1 = np.linalg.lstsq(X, y, rcond=None)[0]

print("切片:", beta0)
print("前市場日リターンの係数:", beta1)

# 係数は学習期間の値に固定し、CalibrationとTestでは再推定しない。
predictions = data[data["split"].isin(["calibration", "test"])].copy()

predictions["prediction_pct"] = beta0 + beta1 * predictions["return_lag1_pct"]
predictions["score_abs"] = (
    predictions["return_pct"] - predictions["prediction_pct"]
).abs()

# CSV保存
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame({
    "term": ["intercept", "return_lag1_pct"],
    "estimate": [beta0, beta1],
}).to_csv(output_dir / "ols_coefficients.csv", index=False)

predictions.to_csv(
    output_dir / "point_predictions.csv",
    index=False,
    date_format="%Y-%m-%d",
)

print(predictions.groupby("split").size())
print(predictions[predictions["split"] == "test"].head(3).to_string(index=False))
