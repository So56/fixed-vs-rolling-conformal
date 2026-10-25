"""全Testのpaired vectorへ非循環Moving Block Bootstrapを適用する。
実行入口はrun_analysis.pyのstage 15。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_block_bootstrap(workspace):
    """保存済み入力を独立workspaceで検証し、この段階の結果だけを保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'bootstrap'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'bootstrap'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'daily_evaluation': PROJECT_DIR / 'reports/evaluation/daily_test_evaluation.csv',
        'comparison': PROJECT_DIR / 'reports/evaluation/comparison_by_regime.csv',
        'paired_differences': PROJECT_DIR / 'reports/evaluation/paired_differences_by_regime.csv',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    GROUPS = ['overall', 'high_vix', 'non_high_vix']
    GROUP_LABELS = {'overall': 'Overall', 'high_vix': 'High-VIX', 'non_high_vix': 'Non-high-VIX'}
    EXPECTED_N = {'overall': 1658, 'high_vix': 149, 'non_high_vix': 1509}
    VECTOR_COLUMNS = [
        'high_vix_30', 'covered_fixed', 'covered_rolling',
        'width_fixed_return_pp', 'width_rolling_return_pp',
        'interval_score_fixed_return_pp', 'interval_score_rolling_return_pp',
    ]
    MEAN_COLUMNS = [
        'coverage_fixed', 'coverage_rolling', 'mean_width_fixed_return_pp', 'mean_width_rolling_return_pp',
        'mean_interval_score_fixed_return_pp', 'mean_interval_score_rolling_return_pp',
    ]
    DIFFERENCE_COLUMNS = [
        'coverage_difference_rolling_minus_fixed', 'mean_width_difference_rolling_minus_fixed_return_pp',
        'mean_interval_score_difference_rolling_minus_fixed_return_pp',
    ]
    STAT_COLUMNS = MEAN_COLUMNS + DIFFERENCE_COLUMNS
    RTOL = ATOL = 1e-12  # 集計値の数値照合だけに使用する。




    def summarize_sample(sample):
        """1つのpaired再標本から3区分の件数・両手法平均・同日差平均を計算する。"""
        high = sample[:, 0].astype(bool)
        masks = [np.ones(len(sample), dtype=bool), high, ~high]
        rows = []
        for group, mask in zip(GROUPS, masks):
            n = int(mask.sum())
            if n == 0:
                statistics = [np.nan] * len(STAT_COLUMNS)
            else:
                selected = sample[mask]
                means = selected[:, 1:].mean(axis=0)
                differences = (selected[:, [2, 4, 6]] - selected[:, [1, 3, 5]]).mean(axis=0)
                statistics = [float(value) for value in np.concatenate([means, differences])]
            rows.append({'group': group, 'n': n, **dict(zip(STAT_COLUMNS, statistics))})
        return rows


    def draw_moving_blocks(rng, n, block_length):
        """非循環の連続blockを復元抽出し、連結順を維持して長さnへ切る。"""
        # nは圧縮しない抽出元Testの長さ。末尾を越える開始位置は抽出しない。
        number_of_blocks = math.ceil(n / block_length)
        block_starts = rng.integers(0, n - block_length + 1, size=number_of_blocks, dtype=np.int64)
        blocks = block_starts[:, None] + np.arange(block_length, dtype=np.int64)
        assert ((0 <= block_starts) & (block_starts <= n - block_length)).all()
        assert (np.diff(blocks, axis=1) == 1).all()  # 切り詰め前の全blockは連続10行。
        assert blocks.min() >= 0 and blocks.max() < n
        sample_indices = blocks.reshape(-1)[:n]
        assert len(sample_indices) == n
        return block_starts, blocks, sample_indices


    def percentile_bounds(values):
        """有効値だけの2.5/97.5%点。Conformalのorder statisticとは別の線形補間。"""
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan, np.nan, 0, len(values)
        lower, upper = np.quantile(valid, [.025, .975], method='linear')
        return float(lower), float(upper), len(valid), len(values) - len(valid)


    # 1. 設計と入力の確認。前段階は実行せず、区間や指標を作り直さない。
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert design['primary_inference'] == 'Paired Moving Block Bootstrap on the full chronological common-date test sequence'
    assert design['bootstrap_repetitions'] == 5000 and design['block_length'] == 10 and design['seed'] == 202701
    assert design['confidence_interval'] == '95% percentile'
    assert design['primary_hypothesis'] == 'two-sided high-VIX coverage difference'
    assert design['nominal_coverage'] == .90 and design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    B, BLOCK_LENGTH, SEED = int(design['bootstrap_repetitions']), int(design['block_length']), int(design['seed'])
    NOMINAL_COVERAGE, CONFIDENCE_LEVEL = float(design['nominal_coverage']), .95
    boolean_columns = ['high_vix_30', 'covered_fixed', 'covered_rolling']
    daily = pd.read_csv(
        INPUT_PATHS['daily_evaluation'], usecols=['date', 'forecast_origin_date', 'split', 'vix_lag1'] + VECTOR_COLUMNS,
        parse_dates=['date', 'forecast_origin_date'], float_precision='round_trip',
        dtype={**{column: 'boolean' for column in boolean_columns}, 'split': 'string'},
    )
    N = len(daily)
    assert N == 1658 and daily['date'].is_unique and daily['date'].is_monotonic_increasing
    assert daily['date'].iloc[0] == pd.Timestamp('2020-01-02') and daily['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert daily['split'].eq('test').all() and daily['forecast_origin_date'].lt(daily['date']).all()
    assert not daily.isna().any().any()
    for column in boolean_columns:
        assert str(daily[column].dtype) == 'boolean' and daily[column].isin([True, False]).all()
    assert daily['high_vix_30'].equals(daily['vix_lag1'].ge(30).astype('boolean'))
    assert int(daily['high_vix_30'].sum()) == 149 and int((~daily['high_vix_30']).sum()) == 1509
    paired_values = daily[VECTOR_COLUMNS].to_numpy(dtype=float)
    assert paired_values.shape == (N, 7) and np.isfinite(paired_values).all()
    assert np.isfinite(daily['vix_lag1']).all()
    assert np.isin(paired_values[:, :3], [0., 1.]).all()
    comparison = pd.read_csv(INPUT_PATHS['comparison'], float_precision='round_trip').set_index(['group', 'method'])
    saved_differences = pd.read_csv(INPUT_PATHS['paired_differences'], float_precision='round_trip').set_index('group')
    assert len(comparison) == 6 and comparison.index.is_unique and len(saved_differences) == 3 and saved_differences.index.is_unique
    original = pd.DataFrame(summarize_sample(paired_values)).set_index('group')
    for group in GROUPS:
        assert original.loc[group, 'n'] == EXPECTED_N[group]
        for method in ['fixed', 'rolling']:
            saved = comparison.loc[(group, method)]
            assert saved['n'] == EXPECTED_N[group]
            np.testing.assert_allclose(
                original.loc[group, [f'coverage_{method}', f'mean_width_{method}_return_pp', f'mean_interval_score_{method}_return_pp']].to_numpy(dtype=float),
                saved[['empirical_coverage', 'mean_width_return_pp', 'mean_interval_score_return_pp']].to_numpy(dtype=float), rtol=RTOL, atol=ATOL,
            )
        np.testing.assert_allclose(original.loc[group, DIFFERENCE_COLUMNS].to_numpy(dtype=float),
                                   saved_differences.loc[group, DIFFERENCE_COLUMNS].to_numpy(dtype=float), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(100 * original.loc[group, DIFFERENCE_COLUMNS[0]],
                                   saved_differences.loc[group, 'coverage_difference_rolling_minus_fixed_pp'], rtol=RTOL, atol=ATOL)

    # 空区分の分岐は検証用コピーで確認し、B個の再標本には混ぜない。
    for high_value, empty_group in [(0., 'high_vix'), (1., 'non_high_vix')]:
        artificial = paired_values[:4].copy()
        artificial[:, 0] = high_value
        checked = pd.DataFrame(summarize_sample(artificial)).set_index('group')
        assert checked.loc[empty_group, 'n'] == 0
        assert checked.loc[empty_group, STAT_COLUMNS].isna().all()
    empty_bounds = percentile_bounds([np.nan, np.nan])
    assert np.isnan(empty_bounds[:2]).all() and empty_bounds[2:] == (0, 2)

    # 2. 5,000個を生成。1つのindexで全7列をまとめて再抽出する。
    number_of_blocks = math.ceil(N / BLOCK_LENGTH)
    assert number_of_blocks == 166 and number_of_blocks * BLOCK_LENGTH - N == 2
    rng = np.random.default_rng(SEED)
    replicate_rows, first_block_rows = [], []
    prefix_starts, prefix_indices = [], []
    for replicate in range(1, B + 1):
        block_starts, blocks, sample_indices = draw_moving_blocks(rng, N, BLOCK_LENGTH)
        sample = paired_values[sample_indices]  # 手法・指標・High-VIXに共通のindex。ソート・重複除去しない。
        assert sample.shape == (N, 7)
        statistics = summarize_sample(sample)
        assert statistics[1]['n'] + statistics[2]['n'] == N and statistics[0]['n'] == N
        for row in statistics:
            replicate_rows.append({'replicate': replicate, **row})
        if replicate <= 3:
            prefix_starts.append(block_starts.copy())
            prefix_indices.append(sample_indices.copy())
            for block_number, start in enumerate(block_starts):
                retained_length = min(BLOCK_LENGTH, N - block_number * BLOCK_LENGTH)
                first_block_rows.append({
                    'replicate': replicate, 'block_number': block_number + 1, 'start_index': int(start),
                    'end_index_before_truncation': int(start + BLOCK_LENGTH - 1), 'retained_length': retained_length,
                    'retained_end_index': int(start + retained_length - 1),
                    'start_date': daily['date'].iloc[start],
                    'end_date_before_truncation': daily['date'].iloc[start + BLOCK_LENGTH - 1],
                })

    replicates = pd.DataFrame(replicate_rows)
    assert len(replicates) == B * 3 and replicates['replicate'].nunique() == B
    assert replicates.groupby('replicate').size().eq(3).all()
    count_pivot = replicates.pivot(index='replicate', columns='group', values='n')
    assert count_pivot['overall'].eq(N).all()
    assert (count_pivot['high_vix'] + count_pivot['non_high_vix']).eq(N).all()
    valid_rows = replicates['n'].gt(0)
    assert np.isfinite(replicates.loc[valid_rows, STAT_COLUMNS].to_numpy()).all()
    assert replicates.loc[~valid_rows, STAT_COLUMNS].isna().all().all()
    for method in ['fixed', 'rolling']:
        assert replicates.loc[valid_rows, f'coverage_{method}'].between(0, 1).all()
        assert replicates.loc[valid_rows, f'mean_width_{method}_return_pp'].ge(0).all()
        assert replicates.loc[valid_rows, f'mean_interval_score_{method}_return_pp'].ge(
            replicates.loc[valid_rows, f'mean_width_{method}_return_pp']).all()
    for difference, fixed_column, rolling_column in zip(DIFFERENCE_COLUMNS, MEAN_COLUMNS[::2], MEAN_COLUMNS[1::2]):
        np.testing.assert_allclose(replicates[difference], replicates[rolling_column] - replicates[fixed_column],
                                   rtol=RTOL, atol=ATOL, equal_nan=True)
    assert replicates.loc[valid_rows, DIFFERENCE_COLUMNS[0]].between(-1, 1).all()
    fixed_width_errors = {}
    for group in GROUPS:
        values = replicates.loc[replicates['group'].eq(group) & valid_rows, 'mean_width_fixed_return_pp']
        np.testing.assert_allclose(values, original.loc[group, 'mean_width_fixed_return_pp'], rtol=RTOL, atol=ATOL)
        fixed_width_errors[group] = float((values - original.loc[group, 'mean_width_fixed_return_pp']).abs().max())

    # 3. 同じseedで最初の3回だけ再生成し、開始位置・index・統計量を照合する。
    check_rng = np.random.default_rng(SEED)
    reproducibility_rows = []
    for replicate in range(1, 4):
        block_starts, blocks, sample_indices = draw_moving_blocks(check_rng, N, BLOCK_LENGTH)
        np.testing.assert_array_equal(block_starts, prefix_starts[replicate - 1])
        np.testing.assert_array_equal(sample_indices, prefix_indices[replicate - 1])
        recalculated = pd.DataFrame(summarize_sample(paired_values[sample_indices])).set_index('group').loc[GROUPS]
        first_saved = replicates.loc[replicates['replicate'].eq(replicate)].set_index('group').loc[GROUPS]
        np.testing.assert_array_equal(recalculated['n'], first_saved['n'])
        np.testing.assert_allclose(recalculated[STAT_COLUMNS], first_saved[STAT_COLUMNS], rtol=0, atol=0, equal_nan=True)
        reproducibility_rows.append({
            'replicate': replicate, 'sample_length': N, 'number_of_blocks': number_of_blocks,
            'high_vix_n': int(recalculated.loc['high_vix', 'n']), 'non_high_vix_n': int(recalculated.loc['non_high_vix', 'n']),
            'block_starts_exact_match': True, 'sample_indices_exact_match': True, 'statistics_exact_match': True,
        })

    # 4. 差の95% percentile CI。High-VIX Coverage差だけが主分析。
    metric_specs = [
        ('coverage_difference', DIFFERENCE_COLUMNS[0], 'proportion', 'coverage percentage points', 100.),
        ('mean_width_difference', DIFFERENCE_COLUMNS[1], 'return percentage points', 'return percentage points', 1.),
        ('mean_interval_score_difference', DIFFERENCE_COLUMNS[2], 'return percentage points', 'return percentage points', 1.),
    ]
    difference_rows, coverage_rows, count_rows = [], [], []
    for group in GROUPS:
        group_replicates = replicates.loc[replicates['group'].eq(group)]
        assert len(group_replicates) == B
        for metric, column, unit, display_unit, scale in metric_specs:
            lower, upper, n_valid, n_invalid = percentile_bounds(group_replicates[column])
            role = 'primary_two_sided' if group == 'high_vix' and metric == 'coverage_difference' else (
                'secondary_high_vix' if group == 'high_vix' else 'auxiliary')
            point = float(original.loc[group, column])
            assert n_valid + n_invalid == B
            if n_valid:
                assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
                if metric == 'coverage_difference':
                    assert -1 <= lower <= upper <= 1
            difference_rows.append({
                'group': group, 'n_original': EXPECTED_N[group], 'metric': metric, 'analysis_role': role,
                'unit': unit, 'point_estimate': point, 'ci_lower': lower, 'ci_upper': upper,
                'confidence_level': CONFIDENCE_LEVEL, 'percentile_method': 'linear',
                'valid_replicates': n_valid, 'invalid_replicates': n_invalid,
                'ci_includes_zero': bool(lower <= 0 <= upper) if n_valid else pd.NA,
                'display_unit': display_unit, 'display_point_estimate': scale * point,
                'display_ci_lower': scale * lower, 'display_ci_upper': scale * upper,
            })
        for method in ['fixed', 'rolling']:
            lower, upper, n_valid, n_invalid = percentile_bounds(group_replicates[f'coverage_{method}'])
            point = float(original.loc[group, f'coverage_{method}'])
            if n_valid:
                assert 0 <= lower <= upper <= 1
                relation = 'below_nominal' if upper < NOMINAL_COVERAGE else (
                    'above_nominal' if lower > NOMINAL_COVERAGE else 'includes_nominal')
                contains_nominal = bool(lower <= NOMINAL_COVERAGE <= upper)
                gap_contains_zero = bool(lower - NOMINAL_COVERAGE <= 0 <= upper - NOMINAL_COVERAGE)
                assert contains_nominal == gap_contains_zero
            else:
                relation, contains_nominal, gap_contains_zero = 'undefined_no_valid', pd.NA, pd.NA
            coverage_rows.append({
                'group': group, 'method': method, 'n_original': EXPECTED_N[group],
                'analysis_role': 'diagnostic_high_vix' if group == 'high_vix' else 'auxiliary_diagnostic',
                'coverage_unit': 'proportion', 'coverage_point_estimate': point,
                'coverage_ci_lower': lower, 'coverage_ci_upper': upper,
                'nominal_interval_coverage': NOMINAL_COVERAGE,
                'signed_gap_unit': 'coverage percentage points',
                'signed_gap_point_estimate_pp': 100 * (point - NOMINAL_COVERAGE),
                'signed_gap_ci_lower_pp': 100 * (lower - NOMINAL_COVERAGE),
                'signed_gap_ci_upper_pp': 100 * (upper - NOMINAL_COVERAGE),
                'coverage_ci_contains_nominal': contains_nominal, 'gap_ci_contains_zero': gap_contains_zero,
                'ci_relation_to_nominal': relation, 'confidence_level': CONFIDENCE_LEVEL,
                'valid_replicates': n_valid, 'invalid_replicates': n_invalid,
            })
        counts = group_replicates['n'].to_numpy()
        count_quantiles = np.quantile(counts, [.025, .25, .5, .75, .975], method='linear')
        count_rows.append({
            'group': group, 'n_original': EXPECTED_N[group], 'generated_replicates': B,
            'valid_group_replicates': int((counts > 0).sum()), 'zero_group_replicates': int((counts == 0).sum()),
            'min_n': int(counts.min()), 'max_n': int(counts.max()), 'mean_n': float(counts.mean()),
            'std_n_ddof1': float(counts.std(ddof=1)), 'n_q025': count_quantiles[0], 'n_q25': count_quantiles[1],
            'n_median': count_quantiles[2], 'n_q75': count_quantiles[3], 'n_q975': count_quantiles[4],
        })
    difference_ci = pd.DataFrame(difference_rows)
    difference_ci['ci_includes_zero'] = difference_ci['ci_includes_zero'].astype('boolean')
    coverage_ci = pd.DataFrame(coverage_rows)
    for column in ['coverage_ci_contains_nominal', 'gap_ci_contains_zero']:
        coverage_ci[column] = coverage_ci[column].astype('boolean')
    count_summary = pd.DataFrame(count_rows)
    assert len(difference_ci) == 9 and len(coverage_ci) == 6
    assert len(difference_ci.loc[difference_ci['analysis_role'].eq('primary_two_sided')]) == 1
    # 元の点推定がCI内にあることは要求しない。CIの補正・対称化も行わない。

    # 5. CSV保存と全列round-trip照合。全replicateの全indexは保存しない。
    output_frames = {
        'replicate_statistics.csv': replicates,
        'original_point_statistics.csv': original.reset_index(),
        'difference_confidence_intervals.csv': difference_ci,
        'coverage_gap_confidence_intervals.csv': coverage_ci,
        'replicate_group_count_summary.csv': count_summary,
        'first_replicate_block_starts.csv': pd.DataFrame(first_block_rows),
        'prefix_reproducibility_checks.csv': pd.DataFrame(reproducibility_rows),
    }
    for filename, frame in output_frames.items():
        path = REPORT_DIR / filename
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
        dates_to_parse = ['start_date', 'end_date_before_truncation'] if filename == 'first_replicate_block_starts.csv' else None
        dtype = {'ci_includes_zero': 'boolean'} if filename == 'difference_confidence_intervals.csv' else (
            {'coverage_ci_contains_nominal': 'boolean', 'gap_ci_contains_zero': 'boolean'}
            if filename == 'coverage_gap_confidence_intervals.csv' else None)
        saved = pd.read_csv(path, parse_dates=dates_to_parse, dtype=dtype, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)

    # 6. 点とCIの両端を別描画。pointを中心にした対称error barへ置換しない。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, layout='constrained')
    panel_labels = [
        ('coverage_difference', 'Coverage difference', 'Coverage difference (percentage points)'),
        ('mean_width_difference', 'Mean width difference', 'Width difference (return percentage points)'),
        ('mean_interval_score_difference', 'Mean interval score difference', 'IS difference (return percentage points)'),
    ]
    for ax, (metric, title, xlabel) in zip(axes, panel_labels):
        panel = difference_ci.loc[difference_ci['metric'].eq(metric)].set_index('group')
        for y, group in enumerate(GROUPS):
            row = panel.loc[group]
            color = '#245A81' if group == 'high_vix' else '#737373'
            lower, upper, point = row['display_ci_lower'], row['display_ci_upper'], row['display_point_estimate']
            if row['valid_replicates'] > 0:
                ax.hlines(y, lower, upper, color=color, linewidth=2, label='95% percentile CI' if y == 0 else None)
                ax.vlines([lower, upper], y - .07, y + .07, color=color, linewidth=1.2)
            ax.scatter(point, y, color=color, s=33, zorder=3, label='Original estimate' if y == 0 else None)
            ax.text(.02, y + .25, f'{point:.2f} [{lower:.2f}, {upper:.2f}]',
                    transform=ax.get_yaxis_transform(), fontsize=9)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_yticks(np.arange(3), [f'{GROUP_LABELS[group]}\n(original n={EXPECTED_N[group]:,})' for group in GROUPS])
        ax.grid(axis='x', alpha=.18)
        ax.margins(x=.12)
    axes[0].set_ylim(2.6, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Paired moving block bootstrap: Rolling minus Fixed (95% percentile CI)', fontsize=13)
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_paired_differences_with_confidence_intervals.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    high_coverage = coverage_ci.loc[coverage_ci['group'].eq('high_vix')].set_index('method')
    for y, method in enumerate(['fixed', 'rolling']):
        row = high_coverage.loc[method]
        lower, upper, point = 100 * row['coverage_ci_lower'], 100 * row['coverage_ci_upper'], 100 * row['coverage_point_estimate']
        color = '#B84B25' if method == 'fixed' else '#245A81'
        if row['valid_replicates'] > 0:
            ax.hlines(y, lower, upper, color=color, linewidth=2)
            ax.vlines([lower, upper], y - .07, y + .07, color=color, linewidth=1.2)
        ax.scatter(point, y, color=color, s=38, zorder=3)
        ax.text(.02, y + .25, f'{point:.2f}% [{lower:.2f}%, {upper:.2f}%]', transform=ax.get_yaxis_transform(), fontsize=10)
    ax.axvline(100 * NOMINAL_COVERAGE, color='#555555', linestyle='--', linewidth=1, label='Nominal interval coverage = 90%')
    ax.set_yticks([0, 1], ['Fixed', 'Rolling-252'])
    ax.set_ylim(1.6, -.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Empirical coverage (%)')
    ax.set_title('High-VIX coverage: original estimates and 95% percentile CI\nPaired MBB, block length 10, B=5,000; original n=149', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.18)
    ax.legend(loc='upper right', fontsize=9)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'02_high_vix_coverage_with_confidence_intervals.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    # 7. 入力保全・乱数方式・CI規則・全検証を記録する。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    assert input_hashes == input_hashes_after
    executed_at = datetime.now(timezone.utc)
    validation = {
        'status': 'PASS', 'generated_replicates': B, 'replicate_statistics_rows': len(replicates),
        'resample_length_each_replicate': N, 'block_length': BLOCK_LENGTH, 'blocks_per_replicate': number_of_blocks,
        'start_range_inclusive': [0, N - BLOCK_LENGTH], 'last_block_retained_rows': N - (number_of_blocks - 1) * BLOCK_LENGTH,
        'non_circular_contiguous_blocks_with_replacement': True, 'same_indices_used_for_whole_paired_vector': True,
        'resampled_sequence_sorted_or_deduplicated': False, 'group_counts_sum_to_N_each_replicate': True,
        'point_estimates_match_saved_e1_tables': True, 'paired_differences_match_method_means_each_replicate': True,
        'coverage_and_ci_ranges_and_endpoint_order_valid': True,
        'empty_group_statistics_nan_no_redraw': True, 'empty_group_branch_checked_with_validation_only_copies': True,
        'total_replicates_redrawn_for_empty_group': 0,
        'fixed_width_max_abs_deviation_from_original_mean': fixed_width_errors,
        'prefix_regenerated_replicates': 3, 'prefix_starts_indices_and_statistics_exact_match': True,
        'csv_all_columns_exact_round_trip': True,
        'original_point_inside_percentile_ci_required': False, 'ci_adjusted_to_contain_original_point': False,
        'comparison_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'prior_steps_refit_or_regenerated': False, 'sign_fraction_reported_as_p_value': False,
        'additional_tests_or_sensitivity_analysis': False,
    }
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {
        'executed_at_utc': executed_at.isoformat(),
        'executed_at_jst': executed_at.astimezone(timezone(timedelta(hours=9))).isoformat(),
        'environment': {'python_executable': sys.executable, 'python_version': sys.version,
                        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'working_directory': str(Path.cwd()), 'code_sha256': file_hash(Path(__file__).resolve()),
        'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]}
                        for name, path in INPUT_PATHS.items()},
        'output_csv_sha256': {filename: file_hash(REPORT_DIR / filename) for filename in output_frames},
        'settings': {'B_generated': B, 'N': N, 'block_length_observations': BLOCK_LENGTH, 'blocks_drawn': number_of_blocks,
                     'seed': SEED, 'rng': 'numpy.random.default_rng', 'bit_generator': type(rng.bit_generator).__name__,
                     'rng_integers': 'low=0, high=N-l+1 (exclusive), dtype=int64, size=ceil(N/l)',
                     'block_scheme': 'ordinary non-circular overlapping MBB; starts sampled with replacement; concatenate then retain first N',
                     'nominal_conformal_coverage': NOMINAL_COVERAGE, 'confidence_level': CONFIDENCE_LEVEL,
                     'percentile_probabilities': [.025, .975], 'percentile_method': 'numpy.quantile(method=linear)',
                     'empty_group_policy': 'NaN; no redraw; CI uses finite values; report valid and invalid counts'},
        'paired_vector_columns_in_order': VECTOR_COLUMNS, 'statistic_columns': STAT_COLUMNS, 'figures': figure_files,
        'scope': 'Step E-2 primary block length 10 only; saved model, intervals and daily vector held fixed',
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    # 8. すべての主・副次・補助結果を同じ規則で日本語の要約へ保存する。
    primary = difference_ci.loc[difference_ci['analysis_role'].eq('primary_two_sided')].iloc[0]
    if primary['valid_replicates'] == 0:
        primary_interpretation = '有効replicateがなく、主CIは未定義である。'
    elif primary['ci_includes_zero']:
        primary_interpretation = '主CIは0を含み、この設定のMBBでは高VIX Coverage差の方向を明確に支持する結果とはいえない。'
    else:
        direction = 'RollingのCoverageが高い方向' if primary['ci_lower'] > 0 else 'RollingのCoverageが低い方向'
        primary_interpretation = f'主CIは0を除外し、本bootstrap方法・設定の下で{direction}の差を支持する結果である。'
    summary = f'''# Step E-2：Paired Moving Block Bootstrapによる不確実性評価

## 実行した設定と範囲

保存済みの全Test日次評価vectorから、通常の非循環・重複候補blockのMoving Block Bootstrapを実行した。
元系列N={N}（2020-01-02〜2026-08-07）、元High-VIX149日、元Non-high-VIX1,509日。
B={B:,}は生成した再標本数、block length={BLOCK_LENGTH}市場日観測、seed={SEED}。
乱数は`np.random.default_rng({SEED})`、bit generatorは{type(rng.bit_generator).__name__}。
主仮説は事前指定どおり高VIX Coverage差のtwo-sidedで、差0を基準にした。観測差の符号を見て片側へ変更していない。
Conformalの名目Coverage90%と、bootstrap CIの信頼水準95%は別の量である。
前段階の実行、OLS再推定、score履歴・閾値・区間・日次指標の再生成、感度分析は行っていない。

## なぜpaired blockを使うのか

同じ市場日のFixedとRollingをpairedに扱うことで、その日に共通の市場環境を維持して差を比較する。
日を独立に再抽出する代わりに連続blockを使い、block内の短期的な時系列依存をある程度保持する。
高VIXだけを先に抜き出すと元の時間間隔やstressの集中を変えてしまうため、全1,658日を抽出対象とする。
1回のindexで`(H, covered_fixed, covered_rolling, width_fixed, width_rolling, IS_fixed, IS_rolling)`の全7列を一緒に再抽出した。
手法ごと・指標ごとの別抽出、抽出後の日付ソート、重複削除は行っていない。

開始indexは0〜{N - BLOCK_LENGTH}の全整数から復元抽出し、各blockは元index上の連続10行。
166 blockを抽出し、連結した1,660行の先頭1,658行を保持した。最後のblockは8行を使う。
末尾から先頭へ循環させず、10暦日ではなく10市場日観測を使っている。
各replicate内の再抽出indicatorからHigh/Non-highの件数を数え、Coverage・Width・ISとその差の分母を毎回計算し直した。
High-VIXの分母を元の149へ固定していない。

## 点推定とCIの計算・コード対応

実行入口：`15_block_bootstrap_evaluation.py`。

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\15_block_bootstrap_evaluation.py
```

コメント1でboolean型・有限性・全日付を確認し、元の日次vectorからの集計値をE-1の保存CSVと完全精度で数値照合した。
`draw_moving_blocks`が抽出indexを作り、コメント2の`sample=paired_values[indices]`で全paired vectorを再抽出する。
`summarize_sample`が各区分の件数と両手法の6平均、同日Rolling−Fixedの3平均差を計算する。
空区分の全統計量はNaNとし、再抽出や0埋めを行わない。CIは有限な有効replicateだけから計算する。
空区分の分岐は、元データを変更しない検証用コピーでも確認した。

コメント3では同じseedから最初の3回だけを再生成し、開始位置・index・統計量の完全一致を確認した。
コメント4の`percentile_bounds`は`np.quantile(valid, [0.025,0.975], method='linear')`を使用する。
これはConformal閾値の228番目のorder statisticとは別であり、bootstrap統計量の経験分布の両側2.5%点をCIとする。
元の点推定がCIに入ることは検証条件にせず、CIを含むよう修正・対称化していない。

## Rolling−Fixed差の95% percentile CI

Coverage差の表は表示用に100倍したcoverage pp。元の比例尺度もCSVへ保存した。
Width差・IS差の単位はreturn ppで、すべてRolling−Fixed。

| 区分 | 指標 | 位置づけ | 点推定 | 95%CI下限 | 95%CI上限 | 0を含む | 有効/無効 |
|---|---|---|---:|---:|---:|---|---|
'''
    for row in difference_ci.itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {GROUP_LABELS[row.group]} | {row.metric} ({row.display_unit}) | {row.analysis_role} | {row.display_point_estimate:+.6f} | {row.display_ci_lower:+.6f} | {row.display_ci_upper:+.6f} | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += f'''
主な推論対象はHigh-VIXのCoverage差で、点推定{primary['display_point_estimate']:+.6f} coverage pp、
95%CI [{primary['display_ci_lower']:+.6f}, {primary['display_ci_upper']:+.6f}] coverage pp。
{primary_interpretation}
High-VIXのWidth差・IS差は副次的、Overall・Non-high-VIXは補助的な評価とし、都合のよいCIだけを主結論へ採用しない。
bootstrapの符号割合をp値とは呼ばず、追加の検定・多重な仮説探索は導入していない。

## 各手法のCoverageと90%からのsigned Gap

CoverageとCIは下表では%表示、CSVは比例尺度。Gapの単位はcoverage pp。
GapのCIはCoverage CIの両端から0.90を引き、100倍して計算した。

| 区分 | 手法 | Coverage (%) | 95%CI (%) | Gap (pp) | Gap 95%CI (pp) | CIと90%の関係 | 有効/無効 |
|---|---|---:|---|---:|---|---|---|
'''
    for row in coverage_ci.itertuples():
        summary += f'| {GROUP_LABELS[row.group]} | {row.method} | {100*row.coverage_point_estimate:.4f} | [{100*row.coverage_ci_lower:.4f}, {100*row.coverage_ci_upper:.4f}] | {row.signed_gap_point_estimate_pp:+.4f} | [{row.signed_gap_ci_lower_pp:+.4f}, {row.signed_gap_ci_upper_pp:+.4f}] | {row.ci_relation_to_nominal} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    for row in coverage_ci.loc[coverage_ci['group'].eq('high_vix')].itertuples():
        if row.ci_relation_to_nominal == 'below_nominal':
            interpretation = 'Coverage CI全体が90%未満、Gap CI全体が0未満であり、この方法の下で名目値からの不足を示す結果である'
        elif row.ci_relation_to_nominal == 'above_nominal':
            interpretation = 'Coverage CI全体が90%より上、Gap CI全体が0より上にある'
        elif row.ci_relation_to_nominal == 'includes_nominal':
            interpretation = 'Coverage CIは90%を、Gap CIは0を含むが、90%を達成した証拠とは解釈しない'
        else:
            interpretation = '有効replicateがないため診断CIは未定義である'
        summary += f'\nHigh-VIXの{row.method}は、{interpretation}。\n'
    summary += '''
Coverageの相対的な改善と、90%に十分近いかという診断を分ける。
Width差が正なら区間が広がり、IS差が負なら外れ距離と幅を合わせた平均scoreが小さいことを示す。
Coverageだけでなく、Width・ISのCIも併せてtrade-offを見る。
Overallが90%に近くても、高VIX部分集団の不足を覆い隠さない。

## 有効数と各再標本の区分件数

| 区分 | 元n | 有効/無効replicate | 最小n | 平均n | 中央値n | 最大n | 件数の2.5/97.5%点 |
|---|---:|---|---:|---:|---:|---:|---|
'''
    for row in count_summary.itertuples():
        summary += f'| {GROUP_LABELS[row.group]} | {row.n_original} | {row.valid_group_replicates}/{row.zero_group_replicates} | {row.min_n} | {row.mean_n:.4f} | {row.n_median:.2f} | {row.max_n} | [{row.n_q025:.2f}, {row.n_q975:.2f}] |\n'
    summary += f'''
件数のpercentileは再標本件数の分布の要約であり、Coverage CIとは別である。
B={B:,}は生成数として維持し、空区分のために再抽出しなかった。
各CIはその統計量の有効値のみから計算し、有効/無効数をCI表へ記録した。

## 検証・保存先

全5,000回で長さ1,658、開始index範囲、各blockの連続性、同じindexによるpairing、High+Non-high=1,658を確認した。
両手法の平均の差と同日paired差の平均が一致した。CoverageとCIの範囲・上下限順序、保存後の全列round-trip一致も確認した。
Fixedの平均Widthのbootstrap変動は浮動小数点誤差の範囲で、値を書き換えて一定化していない。
最初の3回だけを再生成して完全一致を確認した。全5,000回を検証目的で再実行していない。
入力5ファイルは実行前後のハッシュが一致し、前段階成果物を上書きしていない。

- `reports/bootstrap/replicate_statistics.csv`：5,000回×3区分=15,000行。両手法の平均・差・各区分n。
- `reports/bootstrap/difference_confidence_intervals.csv`：主・副次・補助の差CI、単位、0包含、有効/無効数。
- `reports/bootstrap/coverage_gap_confidence_intervals.csv`：各手法のCoverage・signed Gap CIと90%の診断。
- `reports/bootstrap/replicate_group_count_summary.csv`：区分件数の分布。
- `reports/bootstrap/original_point_statistics.csv`：E-1の保存表と照合した元標本の集計。
- `reports/bootstrap/first_replicate_block_starts.csv`：最初の3回の166block開始位置・期間・保持行数。
- `reports/bootstrap/prefix_reproducibility_checks.csv`：最初の3回の区分件数と再現性確認。
- `reports/bootstrap/validation_results.json`、`execution_metadata.json`、`execution.log`：設定・乱数方式・版・ハッシュ・実行・検証記録。
- `figures/bootstrap/`：差のCIの3パネル図、High-VIX CoverageのCI図。各PNG 300dpi・PDF。
- 図の目視確認は実行後に別途行い、`reports/bootstrap/figure_visual_qa.md`に記録する。

## 解釈の限界と次に確認すべき点

この結果は保存済みのモデル・区間・評価系列を固定した不確実性評価で、Trainingから再推定する全工程の不確実性までは評価していない。
MBBもstationarity・weak dependence等の条件とblock lengthの選択に依存し、非定常な金融時系列で完全に正確なCIが保証されるわけではない。
CIが0を除外しても、厳密な有限標本検定やConformalのCoverage保証を得たという意味ではない。
高VIX149日を独立と仮定せず、時間的に集中していることを踏まえ、全時系列と連続blockを保つ設計を維持した。
今回はblock length=10の事前指定した主設定のみを実行し、結果を理由に期間・VIX閾値・window・モデル・主設定を変更していない。
事前指定されたblock length 5/20等の感度分析は別段階で確認する。今回は行っていない。
計算・再標本化・保存・再現性に未解決の不整合は見つからなかった。
'''
    (PROJECT_DIR / 'reports/step_e2_summary.md').write_text(summary, encoding='utf-8')
    print('Rolling−FixedのCI:\n' + difference_ci.to_string(index=False))
    print('\nCoverageとGapのCI:\n' + coverage_ci.to_string(index=False))
    print('\n再標本内の区分件数:\n' + count_summary.to_string(index=False))
    print(f'B={B}, block length={BLOCK_LENGTH}, seed={SEED}; 入力{len(input_hashes)}件のハッシュ不変。')
    print('Step E-2: PASS。前段階再実行・感度分析・追加検定は未実施。図の目視確認は別途実施。')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 15.')
    run_block_bootstrap(ANALYSIS_WORKSPACE)
