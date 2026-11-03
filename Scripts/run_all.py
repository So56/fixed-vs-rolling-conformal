
"""入力とコードのコピーで全工程を実行し、保存済みの結果と照合する。"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


project_dir = Path(__file__).resolve().parent.parent

# 実行結果を保存するフォルダ
run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f_utc")
run_dir = project_dir / "reproduction" / run_id
run_dir.mkdir(parents=True)

scripts_dir = run_dir / "Scripts"
raw_dir = run_dir / "data" / "raw"
logs_dir = run_dir / "logs"
work_dir = run_dir / "working_directory"

for folder in [scripts_dir, raw_dir, logs_dir, work_dir]:
    folder.mkdir(parents=True, exist_ok=True)

# スクリプトと元データをコピー
for path in (project_dir / "Scripts").glob("*.py"):
    shutil.copy2(path, scripts_dir / path.name)

raw_hashes = {}
for name in ["SP500.csv", "VIXCLS.csv"]:
    src = project_dir / "data" / "raw" / name
    dst = raw_dir / name

    raw_hashes[name] = sha256(src)
    shutil.copy2(src, dst)

    if sha256(dst) != raw_hashes[name]:
        raise ValueError(f"{name} のコピーを確認してください。")

for name in ["requirements.txt", "requirements-lock.txt"]:
    path = project_dir / name
    if path.exists():
        shutil.copy2(path, run_dir / name)

# 実行環境の情報を保存
code_hashes = {}
for path in sorted(scripts_dir.glob("*.py")):
    code_hashes[path.name] = sha256(path)

packages = {}
for name in ["numpy", "pandas", "matplotlib", "exchange_calendars"]:
    packages[name] = version(name)

metadata = {
    "python": sys.version,
    "python_executable": sys.executable,
    "base_prefix": sys.base_prefix,
    "raw_sha256": raw_hashes,
    "code_sha256": code_hashes,
    "packages": packages,
}

with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# 分析を最初から実行
scripts = [
    "01_prepare_data.py",
    "02_explore_data.py",
    "03_fit_baseline.py",
    "04_build_prediction_intervals.py",
    "05_evaluate_prediction_intervals.py",
    "06_bootstrap_confidence_intervals.py",
    "07_analyze_sensitivity.py",
    "08_diagnose_results.py",
    "09_make_tables_and_figures.py",
]

env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["MPLBACKEND"] = "Agg"

for name in scripts:
    log_path = logs_dir / f"{Path(name).stem}.log"
    print("実行:", name, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(scripts_dir / name)],
            cwd=work_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

# 元のCSVと再実行したCSVを比較
files = [
    Path("data/processed/analysis_dataset.csv"),
    Path("reports/split_summary.csv"),
    Path("reports/year_boundary_check.csv"),
]

for stage in ["modeling", "evaluation", "reporting"]:
    output_dir = project_dir / "outputs" / stage

    for path in sorted(output_dir.rglob("*.csv")):
        if path.name == "settings.csv":
            continue
        files.append(path.relative_to(project_dir))

comparisons = []

for path in files:
    print("比較:", path, flush=True)

    original = pd.read_csv(project_dir / path)
    new = pd.read_csv(run_dir / path)

    pd.testing.assert_frame_equal(
        original,
        new,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )
    comparisons.append({"file": str(path), "result": "PASS"})

# settings.csvには実行日時もあるので、必要な項目だけ比較
settings_path = Path("outputs/evaluation/bootstrap/settings.csv")

original_settings = pd.read_csv(project_dir / settings_path)
new_settings = pd.read_csv(run_dir / settings_path)

keys = [
    "block_length", "repeats", "seed", "threshold",
    "n", "alpha", "rolling_window", "rng",
]

pd.testing.assert_frame_equal(
    original_settings[keys],
    new_settings[keys],
    check_exact=True,
)

# rawデータが途中で変わっていないか確認
for name in raw_hashes:
    original_hash = sha256(project_dir / "data" / "raw" / name)
    copied_hash = sha256(raw_dir / name)

    if original_hash != raw_hashes[name] or copied_hash != raw_hashes[name]:
        raise ValueError(f"{name} の内容が変更されています。")

# 図は同じファイルが作られたか確認する
figure_count = 0

for stage in ["modeling", "reporting"]:
    for path in (project_dir / "outputs" / stage).rglob("*"):
        if path.suffix.lower() not in [".png", ".pdf"]:
            continue

        relative_path = path.relative_to(project_dir)
        if not (run_dir / relative_path).is_file():
            raise FileNotFoundError(relative_path)

        figure_count += 1

pd.DataFrame(comparisons).to_csv(
    run_dir / "comparison_results.csv", index=False
)

verification = {
    "status": "PASS",
    "csv_files_compared": len(comparisons),
    "figure_files_created": figure_count,
    "rtol": 1e-10,
    "atol": 1e-10,
    "scope": "Re-execution from saved raw in the current environment",
}

with (run_dir / "verification.json").open("w", encoding="utf-8") as f:
    json.dump(verification, f, indent=2)

print("通し実行とCSV比較に合格しました。")
print("保存先:", run_dir)
