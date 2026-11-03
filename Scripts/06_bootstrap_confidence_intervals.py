import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import bootstrap_ci

BLOCK_LENGTH = 10
REPEATS = 5000
SEED = 202701
THRESHOLD = 30

project_dir = Path(__file__).resolve().parent.parent
input_path = project_dir / "outputs" / "evaluation" / "daily_evaluation.csv"
output_dir = project_dir / "outputs" / "evaluation" / "bootstrap"

daily = pd.read_csv(input_path, parse_dates=["date"])
daily = daily.sort_values("date").reset_index(drop=True)

if len(daily) != 1715 or daily["date"].duplicated().any():
    raise ValueError("Testの件数または日付を確認してください。")

ci, replicates, block_starts = bootstrap_ci(
    daily,
    block_length=BLOCK_LENGTH,
    repeats=REPEATS,
    seed=SEED,
    threshold=THRESHOLD,
)

output_dir.mkdir(parents=True, exist_ok=True)
ci.to_csv(output_dir / "confidence_intervals.csv", index=False)
replicates.to_csv(output_dir / "replicates.csv", index=False)
block_starts.to_csv(output_dir / "block_starts.csv", index=False)

# 条件と入力データを記録し、後で同じ計算を再現できるようにする。
settings = pd.DataFrame([{
    "block_length": BLOCK_LENGTH,
    "repeats": REPEATS,
    "seed": SEED,
    "threshold": THRESHOLD,
    "n": len(daily),
    "alpha": 0.10,
    "rolling_window": 252,
    "rng": "PCG64",
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "executed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
}])
settings.to_csv(output_dir / "settings.csv", index=False)

# Rolling − Fixedを表示。Coverageの差だけパーセントポイントに直す。
display = ci.loc[ci["method"] == "difference"].copy()
display.loc[
    display["metric"] == "coverage", ["estimate", "lower", "upper"]
] *= 100
print(display.round(4).to_string(index=False))
