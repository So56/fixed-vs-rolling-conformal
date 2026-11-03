"""予測区間の評価と移動ブロック・ブートストラップの共通処理。"""

import math

import numpy as np
import pandas as pd

GROUPS = ["overall", "high_vix", "non_high_vix"]
METHODS = ["fixed", "rolling", "difference"]
METRICS = ["coverage", "width", "interval_score"]
STAT_COLUMNS = [f"{method}_{metric}" for method in METHODS for metric in METRICS]


def evaluate_intervals(intervals, alpha=0.10):
    if not 0 < alpha < 1:
        raise ValueError("alphaは0と1の間にしてください。")

    daily = intervals.copy()
    required = [
        "return_pct", "vix_lag1", "fixed_lower", "fixed_upper",
        "rolling_lower", "rolling_upper",
    ]
    if not np.isfinite(daily[required].to_numpy(dtype=float)).all():
        raise ValueError("評価に必要な値に欠損や不正な値があります。")

    y = daily["return_pct"].to_numpy()
    for method in ["fixed", "rolling"]:
        lower = daily[f"{method}_lower"].to_numpy()
        upper = daily[f"{method}_upper"].to_numpy()
        if (lower > upper).any():
            raise ValueError("区間の下限が上限を超えています。")

        width = upper - lower
        daily[f"{method}_coverage"] = ((lower <= y) & (y <= upper)).astype(float)
        daily[f"{method}_width"] = width
        # 区間幅に、区間を外れた距離のペナルティを加える。
        daily[f"{method}_interval_score"] = (
            width
            + (2 / alpha) * np.maximum(lower - y, 0)
            + (2 / alpha) * np.maximum(y - upper, 0)
        )

    # 差の向きは、すべてRolling − Fixedにそろえる。
    for metric in METRICS:
        daily[f"difference_{metric}"] = (
            daily[f"rolling_{metric}"] - daily[f"fixed_{metric}"]
        )
    return daily


def summarize(daily, threshold=30):
    high = daily["vix_lag1"].to_numpy() >= threshold
    masks = [np.ones(len(daily), dtype=bool), high, ~high]
    records = []

    for group, mask in zip(GROUPS, masks):
        part = daily.loc[mask]
        for method in METHODS:
            record = {"group": group, "method": method, "n": len(part)}
            for metric in METRICS:
                record[metric] = part[f"{method}_{metric}"].mean()
            records.append(record)
    return pd.DataFrame(records)


def mbb_draws(daily, block_length, repeats, seed, threshold):
    values = daily[STAT_COLUMNS].to_numpy(dtype=float)
    high = daily["vix_lag1"].to_numpy() >= threshold
    n = len(daily)
    if not 1 <= block_length <= n or repeats < 1:
        raise ValueError("ブロック長または反復回数を確認してください。")
    if not np.isfinite(values).all():
        raise ValueError("日次評価に欠損や不正な値があります。")

    rng = np.random.Generator(np.random.PCG64(seed))
    n_blocks = math.ceil(n / block_length)
    draws = np.full((repeats, len(GROUPS), len(STAT_COLUMNS)), np.nan)
    counts = np.zeros((repeats, len(GROUPS)), dtype=int)
    start_positions = np.empty((repeats, n_blocks), dtype=np.int64)

    for b in range(repeats):
        starts = rng.integers(
            0, n - block_length + 1, size=n_blocks, dtype=np.int64
        )
        start_positions[b] = starts
        # 連続した日付をブロックで抽出し、元と同じ件数に切りそろえる。
        indices = np.concatenate([
            np.arange(start, start + block_length) for start in starts
        ])[:n]

        # 両手法とVIX区分に同じ抽出位置を使い、日ごとの対応を保つ。
        sampled_values = values[indices]
        sampled_high = high[indices]
        masks = [np.ones(n, dtype=bool), sampled_high, ~sampled_high]
        for g, mask in enumerate(masks):
            counts[b, g] = mask.sum()
            # 該当日がない区分はNaNのまま残す。
            if counts[b, g] > 0:
                draws[b, g] = sampled_values[mask].mean(axis=0)

        if (b + 1) % 1000 == 0:
            print(f"Bootstrap: {b + 1}/{repeats}")
    return draws, counts, start_positions


def bootstrap_ci(daily, block_length=10, repeats=5000, seed=202701, threshold=30):
    draws, counts, starts = mbb_draws(
        daily, block_length, repeats, seed, threshold
    )
    values = daily[STAT_COLUMNS].to_numpy(dtype=float)
    high = daily["vix_lag1"].to_numpy() >= threshold
    masks = [np.ones(len(daily), dtype=bool), high, ~high]
    records = []

    for g, group in enumerate(GROUPS):
        point = np.full(len(STAT_COLUMNS), np.nan)
        if masks[g].any():
            point = values[masks[g]].mean(axis=0)

        for j, column in enumerate(STAT_COLUMNS):
            method, metric = column.split("_", 1)
            valid = draws[:, g, j]
            # 該当日がない反復を除き、パーセンタイル法で95%信頼区間を求める。
            valid = valid[np.isfinite(valid)]
            lower, upper = np.nan, np.nan
            if len(valid) > 0:
                lower, upper = np.quantile(valid, [0.025, 0.975], method="linear")

            records.append({
                "group": group,
                "method": method,
                "metric": metric,
                "n": int(masks[g].sum()),
                "estimate": point[j],
                "lower": lower,
                "upper": upper,
                "valid_reps": len(valid),
                "invalid_reps": repeats - len(valid),
            })

    columns = [f"{group}__{column}" for group in GROUPS for column in STAT_COLUMNS]
    replicates = pd.DataFrame(draws.reshape(repeats, -1), columns=columns)
    replicates.insert(0, "replicate", np.arange(1, repeats + 1))
    for g, group in enumerate(GROUPS):
        replicates[f"{group}__n"] = counts[:, g]

    block_starts = pd.DataFrame(
        starts, columns=[f"block_{i + 1}" for i in range(starts.shape[1])]
    )
    block_starts.insert(0, "replicate", np.arange(1, repeats + 1))
    return pd.DataFrame(records), replicates, block_starts
