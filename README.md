# Fixed vs. Rolling Conformal Prediction

This project compares two approaches to constructing prediction intervals for daily S&P 500 log returns: **Fixed**, which keeps the residual quantile constant, and **Rolling**, which updates it using recent residuals. Coverage, interval width, and interval score are evaluated across market regimes defined by the previous trading day's VIX.

## Analysis Setup

Point forecasts come from an ordinary least squares (OLS) model using the previous trading day's return as the predictor. The coefficients are estimated on the training period and held fixed throughout the analysis. Fixed uses the absolute prediction errors from 2019 throughout, while Rolling updates its intervals using errors from the 252 trading days preceding each forecast date.

| Parameter | Setting |
| --- | --- |
| Training period | 2017–2018 (502 trading days) |
| Calibration period | 2019 (252 trading days) |
| Evaluation period | 2020-01-02 to 2026-10-28 (1,715 trading days) |
| Prediction intervals | 90% nominal coverage (`alpha = 0.10`) |
| High-VIX regime | Previous trading day's VIX ≥ 30 |
| Confidence intervals | 95% intervals from a moving block bootstrap |
| Bootstrap settings | Block length: 10; replications: 5,000; random seed: 202701 |
| Sensitivity analysis | Block lengths: 5 and 20; VIX thresholds: 25 and 35; window sizes: 63 and 126 trading days |

This analysis measures empirical coverage over time. The nominal coverage level of 90% does not guarantee 90% coverage in every period or market regime.

## Project Structure

```text
Scripts/                   Analysis scripts and shared evaluation functions
data/
  raw/                     Input CSV files (SP500.csv, VIXCLS.csv)
  processed/               Preprocessed data for analysis
outputs/
  modeling/                Descriptive statistics, OLS coefficients, forecasts, intervals
  evaluation/              Daily and regime-level evaluations, bootstrap/, sensitivity/
  reporting/               Annual and high-VIX episode diagnostics, tables/, figures/
reports/                   Data split checks, cleanup report, and file rename mapping
reproduction/              Code, data, and logs for each reproduction run (excluded from Git)
requirements.txt           Pinned dependency versions
```

In the CSV files, returns and interval widths are expressed as percentages, while coverage is recorded on a scale from 0 to 1. The `difference` column represents **Rolling − Fixed**. In tables and figures prepared for publication, coverage is converted to percentages and coverage differences to percentage points.

## Running the Analysis

Use Python 3.13. The workflow has been tested on Windows with Python 3.13.14. Run the following commands in PowerShell from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Scripts/run_all.py
```

`run_all.py` copies the code and raw data to `reproduction/run_..._utc/` and runs the full workflow there. It compares the generated CSV files against the saved reference files using tolerances of `rtol = atol = 1e-10` and checks that the figures are generated. The original `data/`, `outputs/`, and `reports/` directories are left unchanged. On successful completion, `verification.json` and `comparison_results.csv` are saved in the run directory.

Because `bootstrap/settings.csv` includes run-specific metadata such as timestamps, the reproducibility check compares its key computational settings. For figures, the check confirms that the files are generated; it does not automatically compare image contents.

On macOS and Linux, use `.venv/bin/python` as the virtual environment's Python executable. The workflow has not been tested on these platforms.

### Running Individual Steps

Run `python Scripts/<filename>` in the order shown below. Running scripts individually overwrites the corresponding files in `data/processed/`, `outputs/`, and `reports/`.

| Step | Script | Description |
| --- | --- | --- |
| 01 | `01_prepare_data.py` | Merge closing prices, compute log returns and lagged variables, and split the data into periods |
| 02 | `02_explore_data.py` | Check trading-day alignment, compute descriptive statistics, and plot distributions |
| 03 | `03_fit_baseline.py` | Fit the OLS model, generate point forecasts, and save absolute prediction errors |
| 04 | `04_build_prediction_intervals.py` | Construct Fixed and Rolling intervals and verify chronological ordering |
| 05 | `05_evaluate_prediction_intervals.py` | Evaluate results by day and VIX regime |
| 06 | `06_bootstrap_confidence_intervals.py` | Compute bootstrap confidence intervals |
| 07 | `07_analyze_sensitivity.py` | Assess sensitivity to alternative settings |
| 08 | `08_diagnose_results.py` | Summarize annual results and examine consecutive high-VIX episodes, leave-one-year-out results, and autocorrelation |
| 09 | `09_make_tables_and_figures.py` | Generate tables and figures |

`evaluation.py` provides shared functions used by the analysis scripts, and `run_all.py` is the entry point for running steps 01–09 in sequence. These two files are unnumbered because neither represents an individual analysis step.

## Input Data and Publication Checklist

The input files are `data/raw/SP500.csv` (S&P 500 closing prices) and `data/raw/VIXCLS.csv` (VIX closing values). The workflow uses the bundled CSV files and does not download data. Its date-range and observation-count checks are tailored to these files, so replacing the data requires reviewing both the configuration and the validation criteria.

Before publishing, verify the data sources, retrieval dates, and redistribution terms. A license has not yet been assigned to the code. The `reproduction/` and `.venv/` directories are excluded from Git because they contain machine-specific paths.

See the [cleanup report](reports/project_cleanup.md) for details of the project cleanup and potentially redundant files, and [file_renames.csv](reports/file_renames.csv) for the complete file rename mapping.
