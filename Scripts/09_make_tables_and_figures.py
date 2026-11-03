import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def format_ci_table(source, index_columns):
    table = source.copy()
    table.loc[table["metric"] == "coverage", ["estimate", "lower", "upper"]] *= 100
    table["estimate_ci"] = [
        f"{row.estimate:.2f} [{row.lower:.2f}, {row.upper:.2f}]"
        for row in table.itertuples(index=False)
    ]
    wide = table.pivot(index=index_columns, columns="metric", values="estimate_ci")
    return wide.reindex(columns=metrics).reset_index()


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=300)
    fig.savefig(figure_dir / f"{name}.pdf")
    plt.close(fig)


project_dir = Path(__file__).resolve().parent.parent
evaluation_dir = project_dir / "outputs" / "evaluation"
table_dir = project_dir / "outputs" / "reporting" / "tables"
figure_dir = project_dir / "outputs" / "reporting" / "figures"
metrics = ["coverage", "width", "interval_score"]

daily_path = evaluation_dir / "daily_evaluation.csv"
settings = pd.read_csv(evaluation_dir / "bootstrap" / "settings.csv").iloc[0]
if hashlib.sha256(daily_path.read_bytes()).hexdigest() != settings["input_sha256"]:
    raise ValueError("日次評価がBootstrap実行時から変更されています。")

ci = pd.read_csv(evaluation_dir / "bootstrap" / "confidence_intervals.csv")
if not (ci["valid_reps"] + ci["invalid_reps"]).eq(5000).all():
    raise ValueError("正式な5000回の結果を確認してください。")

sensitivity = pd.read_csv(evaluation_dir / "sensitivity" / "all_confidence_intervals.csv")
table_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

# 表1：標本の期間と件数
sample = pd.read_csv(project_dir / "reports" / "split_summary.csv")
sample.to_csv(table_dir / "sample_summary.csv", index=False)

# 表2：各手法の性能とCI
levels = ci.loc[ci["method"].isin(["fixed", "rolling"])]
table2 = format_ci_table(levels, ["group", "method", "n"])
table2.to_csv(table_dir / "method_performance_ci.csv", index=False)

# 表3：Rolling − Fixedの差とCI
differences = ci.loc[ci["method"] == "difference"]
table3 = format_ci_table(differences, ["group", "n"])
table3.to_csv(table_dir / "method_differences_ci.csv", index=False)

# 表4：高VIXの感度分析
selected = sensitivity.loc[
    (sensitivity["group"] == "high_vix") & (sensitivity["method"] == "difference")
]
table4 = format_ci_table(
    selected, ["case", "window", "threshold", "block_length", "n"]
)
case_order = [
    "main", "block_5", "block_20", "vix_25", "vix_35", "window_63", "window_126",
]
table4 = table4.set_index("case").reindex(case_order).reset_index()
table4.to_csv(table_dir / "high_vix_sensitivity_ci.csv", index=False)
print("論文用の表を保存しました。")

# 図1：リターンと前取引日のVIX
analysis = pd.read_csv(
    project_dir / "data" / "processed" / "analysis_dataset.csv", parse_dates=["date"]
)
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(analysis["date"], analysis["return_pct"], linewidth=0.6)
axes[0].set_ylabel("Return (%)")
axes[1].plot(analysis["date"], analysis["vix_lag1"], linewidth=0.8)
axes[1].axhline(30, color="red", linestyle="--")
axes[1].set_ylabel("Previous-session VIX")

for ax in axes:
    ax.axvspan(
        pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"),
        color="gray", alpha=0.2,
    )
save_figure(fig, "returns_and_vix")

# 図2：予測区間の半幅
daily = pd.read_csv(daily_path, parse_dates=["date"])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily["date"], daily["fixed_halfwidth"], label="Fixed")
ax.plot(daily["date"], daily["rolling_halfwidth"], label="Rolling", linewidth=0.8)
ax.set_ylabel("Half-width (return pp)")
ax.legend()
save_figure(fig, "prediction_interval_halfwidths")

# 図3：Rolling − Fixedの差と95%信頼区間
groups = ["overall", "high_vix", "non_high_vix"]
group_labels = ["All Test", "High VIX", "Non-high VIX"]
labels = [
    "Coverage difference (pp)",
    "Width difference (return pp)",
    "IS difference (return pp)",
]
y_positions = range(len(groups))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, metric, label in zip(axes, metrics, labels):
    part = differences.loc[differences["metric"] == metric].set_index("group")
    part = part.reindex(groups)
    scale = 100 if metric == "coverage" else 1

    ax.hlines(y_positions, part["lower"] * scale, part["upper"] * scale)
    ax.scatter(part["estimate"] * scale, y_positions)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(y_positions, group_labels)
    ax.invert_yaxis()
    ax.set_xlabel(label)
save_figure(fig, "method_differences_ci")
