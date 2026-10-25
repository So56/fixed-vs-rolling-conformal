"""VIX閾値25/35を同じ抽出indexで比較し、主30の保存値を保持する。
実行入口はrun_analysis.pyのstage 17。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_vix_threshold_sensitivity(workspace):
    """保存済み入力を独立workspaceで検証し、この段階の結果だけを保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports/vix_threshold_sensitivity'
    FIGURE_DIR = PROJECT_DIR / 'figures/vix_threshold_sensitivity'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'daily_evaluation': PROJECT_DIR / 'reports/evaluation/daily_test_evaluation.csv',
        'main_comparison': PROJECT_DIR / 'reports/evaluation/comparison_by_regime.csv',
        'main_paired': PROJECT_DIR / 'reports/evaluation/paired_differences_by_regime.csv',
        'main_replicates': PROJECT_DIR / 'reports/bootstrap/replicate_statistics.csv',
        'main_difference_ci': PROJECT_DIR / 'reports/bootstrap/difference_confidence_intervals.csv',
        'main_coverage_ci': PROJECT_DIR / 'reports/bootstrap/coverage_gap_confidence_intervals.csv',
        'main_original': PROJECT_DIR / 'reports/bootstrap/original_point_statistics.csv',
        'main_counts': PROJECT_DIR / 'reports/bootstrap/replicate_group_count_summary.csv',
        'main_metadata': PROJECT_DIR / 'reports/bootstrap/execution_metadata.json',
        'main_block_starts': PROJECT_DIR / 'reports/bootstrap/first_replicate_block_starts.csv',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    THRESHOLDS = [25, 30, 35]
    NEW_THRESHOLDS = [25, 35]
    GROUPS = ['high_vix', 'non_high_vix']  # Overallのbootstrap統計・CIは既存値を再利用する。
    ALL_GROUPS = ['overall'] + GROUPS
    METRIC_COLUMNS = [
        'covered_fixed', 'covered_rolling', 'width_fixed_return_pp', 'width_rolling_return_pp',
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
    METRIC_SPECS = [
        ('coverage_difference', DIFFERENCE_COLUMNS[0], 'proportion', 'coverage percentage points', 100.),
        ('mean_width_difference', DIFFERENCE_COLUMNS[1], 'return percentage points', 'return percentage points', 1.),
        ('mean_interval_score_difference', DIFFERENCE_COLUMNS[2], 'return percentage points', 'return percentage points', 1.),
    ]
    RTOL = ATOL = 1e-12  # 数値照合だけに使用。VIX分類には許容差を使わない。


    def summarize_sample(metrics, high):
        """共通の日次指標を区分別に集計。日次指標や予測区間を作り直さない。"""
        rows = []
        for group, mask in zip(GROUPS, [high, ~high]):
            n = int(mask.sum())
            if n == 0:
                statistics = [np.nan] * len(STAT_COLUMNS)
            else:
                selected = metrics[mask]
                means = selected.mean(axis=0)
                differences = (selected[:, [1, 3, 5]] - selected[:, [0, 2, 4]]).mean(axis=0)
                statistics = [float(value) for value in np.concatenate([means, differences])]
            rows.append({'group': group, 'n': n, **dict(zip(STAT_COLUMNS, statistics))})
        return rows


    def descriptive_statistics(frame, group, primary=False):
        """保存日次値からCoverage・Gap・Width・ISを集計する。"""
        n, rows = len(frame), []
        for method in ['fixed', 'rolling']:
            covered = int(frame[f'covered_{method}'].sum())
            lower, upper = int(frame[f'lower_miss_{method}'].sum()), int(frame[f'upper_miss_{method}'].sum())
            assert covered + lower + upper == n
            coverage = covered / n if n else np.nan
            gap = 100 * (coverage - .90)
            rows.append({
                'group': group, 'method': method, 'n': n, 'covered_n': covered, 'lower_miss_n': lower, 'upper_miss_n': upper,
                'empirical_coverage': coverage, 'coverage_gap_pp': gap, 'absolute_coverage_gap_pp': abs(gap),
                'mean_width_return_pp': frame[f'width_{method}_return_pp'].mean(),
                'median_width_return_pp': frame[f'width_{method}_return_pp'].median(),
                'mean_interval_score_return_pp': frame[f'interval_score_{method}_return_pp'].mean(),
            })
        difference = frame['coverage_difference_rolling_minus_fixed'].mean()
        paired = {'group': group, 'n': n, 'primary_coverage_comparison': primary,
                  'coverage_difference_rolling_minus_fixed': difference,
                  'coverage_difference_rolling_minus_fixed_pp': 100 * difference,
                  'mean_width_difference_rolling_minus_fixed_return_pp': frame['width_difference_rolling_minus_fixed_return_pp'].mean(),
                  'mean_interval_score_difference_rolling_minus_fixed_return_pp': frame['interval_score_difference_rolling_minus_fixed_return_pp'].mean()}
        return rows, paired


    def with_provenance(frame, threshold, path, reused=False):
        result = frame.copy()
        result.insert(0, 'vix_threshold', threshold)
        result['setting_role'] = 'primary_setting' if threshold == 30 else 'sensitivity_setting'
        result.loc[result['group'].eq('overall'), 'setting_role'] = 'threshold_invariant_overall'
        result['result_origin'] = 'reused_existing' if reused else 'newly_computed'
        result['source_file'] = path.relative_to(PROJECT_DIR).as_posix()
        result['source_sha256'] = file_hash(path)
        return result


    # 主設定を実行する15/16はimportしない。以下の短い共通処理は同じ規則で配置する。

    def draw_moving_blocks(rng, n, block_length):
        # nは圧縮しない抽出元Testの長さ。末尾を越える開始位置は抽出しない。
        number_of_blocks = math.ceil(n / block_length)
        block_starts = rng.integers(0, n - block_length + 1, size=number_of_blocks, dtype=np.int64)
        blocks = block_starts[:, None] + np.arange(block_length, dtype=np.int64)
        assert ((0 <= block_starts) & (block_starts <= n - block_length)).all()
        assert (np.diff(blocks, axis=1) == 1).all()
        assert blocks.min() >= 0 and blocks.max() < n
        sample_indices = blocks.reshape(-1)[:n]
        assert len(sample_indices) == n
        return block_starts, sample_indices

    def percentile_bounds(values):
        values = np.asarray(values, dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return np.nan, np.nan, 0, len(values)
        lower, upper = np.quantile(valid, [.025, .975], method='linear')
        return float(lower), float(upper), len(valid), len(values) - len(valid)

    def build_ci_tables(replicates, original, b, nominal):
        """差とCoverageのpercentile CI。各設定で同じ計算を行う。"""
        difference_rows, coverage_rows = [], []
        for group in GROUPS:
            group_replicates = replicates.loc[replicates['group'].eq(group)]
            assert len(group_replicates) == b
            for metric, column, unit, display_unit, scale in METRIC_SPECS:
                lower, upper, n_valid, n_invalid = percentile_bounds(group_replicates[column])
                role = 'sensitivity_high_vix' if group == 'high_vix' else 'auxiliary_sensitivity'
                point = float(original.loc[group, column])
                assert n_valid + n_invalid == b
                if n_valid:
                    assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
                    if metric == 'coverage_difference':
                        assert -1 <= lower <= upper <= 1
                difference_rows.append({
                    'group': group, 'n_original': int(original.loc[group, 'n']), 'metric': metric, 'analysis_role': role,
                    'unit': unit, 'point_estimate': point, 'ci_lower': lower, 'ci_upper': upper,
                    'confidence_level': .95, 'percentile_method': 'linear',
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
                    relation = 'below_nominal' if upper < nominal else (
                        'above_nominal' if lower > nominal else 'includes_nominal')
                    contains_nominal = bool(lower <= nominal <= upper)
                    gap_contains_zero = bool(lower - nominal <= 0 <= upper - nominal)
                    assert contains_nominal == gap_contains_zero
                else:
                    relation, contains_nominal, gap_contains_zero = 'undefined_no_valid', pd.NA, pd.NA
                coverage_rows.append({
                    'group': group, 'method': method, 'n_original': int(original.loc[group, 'n']),
                    'analysis_role': 'diagnostic_high_vix_sensitivity' if group == 'high_vix' else 'auxiliary_diagnostic_sensitivity',
                    'coverage_unit': 'proportion', 'coverage_point_estimate': point,
                    'coverage_ci_lower': lower, 'coverage_ci_upper': upper, 'nominal_interval_coverage': nominal,
                    'signed_gap_unit': 'coverage percentage points',
                    'signed_gap_point_estimate_pp': 100 * (point - nominal),
                    'signed_gap_ci_lower_pp': 100 * (lower - nominal), 'signed_gap_ci_upper_pp': 100 * (upper - nominal),
                    'coverage_ci_contains_nominal': contains_nominal, 'gap_ci_contains_zero': gap_contains_zero,
                    'ci_relation_to_nominal': relation, 'confidence_level': .95,
                    'valid_replicates': n_valid, 'invalid_replicates': n_invalid,
                })
        difference_ci, coverage_ci = pd.DataFrame(difference_rows), pd.DataFrame(coverage_rows)
        difference_ci['ci_includes_zero'] = difference_ci['ci_includes_zero'].astype('boolean')
        for column in ['coverage_ci_contains_nominal', 'gap_ci_contains_zero']:
            coverage_ci[column] = coverage_ci[column].astype('boolean')
        assert len(difference_ci) == 6 and len(coverage_ci) == 4
        return difference_ci, coverage_ci

    def summarize_counts(replicates, b, original):
        rows = []
        for group in GROUPS:
            counts = replicates.loc[replicates['group'].eq(group), 'n'].to_numpy()
            quantiles = np.quantile(counts, [.025, .25, .5, .75, .975], method='linear')
            rows.append({
                'group': group, 'n_original': int(original.loc[group, 'n']), 'generated_replicates': b,
                'valid_group_replicates': int((counts > 0).sum()), 'zero_group_replicates': int((counts == 0).sum()),
                'min_n': int(counts.min()), 'max_n': int(counts.max()), 'mean_n': float(counts.mean()),
                'std_n_ddof1': float(counts.std(ddof=1)), 'n_q025': quantiles[0], 'n_q25': quantiles[1],
                'n_median': quantiles[2], 'n_q75': quantiles[3], 'n_q975': quantiles[4],
            })
        return pd.DataFrame(rows)

    def save_frame(path, frame):
        """CSV保存後に全列を再読込し、型・値・件数を確認する。"""
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
        dates = [column for column in frame if pd.api.types.is_datetime64_any_dtype(frame[column])]
        booleans = {column: 'boolean' for column in frame if str(frame[column].dtype) == 'boolean'}
        saved = pd.read_csv(path, parse_dates=dates or None, dtype=booleans or None, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)


    # 1. 主設定との入力一致と、保存済み全Testの型・日時・有限性を確認する。
    started_at = datetime.now(timezone.utc)
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    protected_paths = []
    for directory in ['reports/bootstrap', 'figures/bootstrap', 'reports/evaluation', 'figures/evaluation']:
        protected_paths.extend(path for path in (PROJECT_DIR / directory).glob('*') if path.is_file())
    protected_hashes = {path.relative_to(PROJECT_DIR).as_posix(): file_hash(path) for path in protected_paths}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    main_metadata = json.loads(INPUT_PATHS['main_metadata'].read_text(encoding='utf-8'))
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert design['sensitivity_vix_thresholds'] == THRESHOLDS and design['rolling_window'] == 252
    assert design['block_length'] == 10 and design['bootstrap_repetitions'] == 5000 and design['seed'] == 202701
    assert design['nominal_coverage'] == .90 and design['primary_hypothesis'] == 'two-sided high-VIX coverage difference'
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    settings = main_metadata['settings']
    assert settings['N'] == 1658 and settings['B_generated'] == 5000 and settings['block_length_observations'] == 10
    assert settings['seed'] == 202701 and settings['rng'] == 'numpy.random.default_rng' and settings['bit_generator'] == 'PCG64'
    assert settings['confidence_level'] == .95 and settings['percentile_probabilities'] == [.025, .975]
    assert settings['percentile_method'] == 'numpy.quantile(method=linear)'
    assert settings['block_scheme'] == 'ordinary non-circular overlapping MBB; starts sampled with replacement; concatenate then retain first N'
    for key, main_key in [('daily_evaluation', 'daily_evaluation'), ('main_comparison', 'comparison'),
                          ('main_paired', 'paired_differences'), ('research_design_lock', 'research_design_lock'),
                          ('research_design_source', 'research_design_source')]:
        assert input_hashes[key] == main_metadata['input_files'][main_key]['sha256']
    for key in ['main_replicates', 'main_difference_ci', 'main_coverage_ci', 'main_original', 'main_counts', 'main_block_starts']:
        assert input_hashes[key] == main_metadata['output_csv_sha256'][INPUT_PATHS[key].name]
    B, BLOCK_LENGTH, SEED, N = 5000, 10, 202701, 1658
    boolean_columns = ['high_vix_30'] + [f'{kind}_{method}' for method in ['fixed', 'rolling'] for kind in ['covered', 'lower_miss', 'upper_miss']]
    daily = pd.read_csv(INPUT_PATHS['daily_evaluation'], parse_dates=['date', 'forecast_origin_date'],
                        dtype={**{column: 'boolean' for column in boolean_columns}, 'split': 'string'}, float_precision='round_trip')
    assert len(daily) == N and daily['date'].is_unique and daily['date'].is_monotonic_increasing
    assert daily['date'].iloc[0] == pd.Timestamp('2020-01-02') and daily['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert not daily.isna().any().any() and daily['split'].eq('test').all()
    assert daily['forecast_origin_date'].lt(daily['date']).all()
    assert np.isfinite(daily.select_dtypes(include='number')).all().all()
    for column in boolean_columns:
        assert str(daily[column].dtype) == 'boolean' and daily[column].isin([True, False]).all()
    metrics = daily[METRIC_COLUMNS].to_numpy(dtype=float)
    assert metrics.shape == (N, 6) and np.isfinite(metrics).all() and np.isin(metrics[:, :2], [0., 1.]).all()
    indicators = np.column_stack([daily['vix_lag1'].ge(threshold).to_numpy(dtype=bool) for threshold in THRESHOLDS])
    np.testing.assert_array_equal(indicators[:, 1], daily['high_vix_30'].to_numpy(dtype=bool))
    assert indicators[:, 1].sum() == 149 and (~indicators[:, 1]).sum() == 1509
    assert (indicators[:, 2] <= indicators[:, 1]).all() and (indicators[:, 1] <= indicators[:, 0]).all()
    main_comparison = pd.read_csv(INPUT_PATHS['main_comparison'], float_precision='round_trip')
    main_paired = pd.read_csv(INPUT_PATHS['main_paired'], float_precision='round_trip')
    main_original = pd.read_csv(INPUT_PATHS['main_original'], float_precision='round_trip').set_index('group')
    main_replicates = pd.read_csv(INPUT_PATHS['main_replicates'], float_precision='round_trip')
    main_difference = pd.read_csv(INPUT_PATHS['main_difference_ci'], dtype={'ci_includes_zero': 'boolean'}, float_precision='round_trip')
    main_coverage = pd.read_csv(INPUT_PATHS['main_coverage_ci'], dtype={
        'coverage_ci_contains_nominal': 'boolean', 'gap_ci_contains_zero': 'boolean'}, float_precision='round_trip')
    main_counts = pd.read_csv(INPUT_PATHS['main_counts'], float_precision='round_trip')
    main_starts = pd.read_csv(INPUT_PATHS['main_block_starts'], parse_dates=['start_date', 'end_date_before_truncation'])
    assert len(main_replicates) == 3 * B and not main_replicates.duplicated(['replicate', 'group']).any()
    assert len(main_starts) == 3 * 166
    main_indexed = main_replicates.set_index(['replicate', 'group'])
    main_high_counts = main_indexed.xs('high_vix', level='group').sort_index()['n'].to_numpy()
    main_overall = main_indexed.xs('overall', level='group').sort_index()
    np.testing.assert_array_equal(main_overall.index, np.arange(1, B + 1))
    assert np.isfinite(main_replicates[STAT_COLUMNS]).all().all()
    assert main_overall['n'].eq(N).all()


    # 2. 標本構成・年別件数・連続区間。全Testで変化点を付けてから高VIX区間を記録。
    indicator_frame = daily[['date', 'vix_lag1']].copy()
    composition_rows, year_rows, run_rows, threshold_rows = [], [], [], []
    comparisons, paired_tables, original_points = {30: main_comparison}, {30: main_paired}, {30: main_original}
    overall_rows, overall_paired_check = descriptive_statistics(daily, 'overall')
    pd.testing.assert_frame_equal(pd.DataFrame(overall_rows), main_comparison.loc[main_comparison['group'].eq('overall')].reset_index(drop=True),
                                  check_exact=False, rtol=RTOL, atol=ATOL)
    pd.testing.assert_frame_equal(pd.DataFrame([overall_paired_check]), main_paired.loc[main_paired['group'].eq('overall')].reset_index(drop=True),
                                  check_exact=False, rtol=RTOL, atol=ATOL)
    for threshold, high_array in zip(THRESHOLDS, indicators.T):
        high = pd.Series(high_array, index=daily.index)
        indicator_frame[f'H_{threshold}'] = high
        high_n = int(high.sum())
        assert high_n + int((~high).sum()) == N
        for group, mask in zip(GROUPS, [high, ~high]):
            composition_rows.append({'vix_threshold': threshold, 'group': group, 'n': int(mask.sum()),
                                     'share_of_test': float(mask.mean()), 'test_n': N})
        for year, year_data in daily.groupby(daily['date'].dt.year):
            n_year, n_high = len(year_data), int(high.loc[year_data.index].sum())
            year_rows.append({'vix_threshold': threshold, 'year': int(year), 'n': n_year, 'high_vix_n': n_high,
                              'high_vix_share_within_year': n_high / n_year,
                              'share_of_threshold_high_vix_days': n_high / high_n if high_n else np.nan,
                              'first_date': year_data['date'].min(), 'last_date': year_data['date'].max(),
                              'partial_year': bool(year == 2026)})
        full_sequence = pd.DataFrame({'date': daily['date'], 'high': high,
                                      'run_id': high.ne(high.shift(fill_value=False)).cumsum()})
        threshold_runs = []
        for _, segment in full_sequence.groupby('run_id', sort=False):
            if segment['high'].iloc[0]:
                threshold_runs.append({'vix_threshold': threshold, 'high_run_number': len(threshold_runs) + 1,
                                       'start_date': segment['date'].iloc[0], 'end_date': segment['date'].iloc[-1],
                                       'high_vix_days': len(segment)})
        assert sum(row['high_vix_days'] for row in threshold_runs) == high_n
        run_rows.extend(threshold_runs)
        threshold_rows.append({'vix_threshold': threshold, 'setting_role': 'primary_setting' if threshold == 30 else 'sensitivity_setting',
                               'test_n': N, 'high_vix_n': high_n, 'non_high_vix_n': N - high_n,
                               'high_vix_share': high_n / N, 'high_vix_runs': len(threshold_runs),
                               'longest_high_vix_run_days': max([row['high_vix_days'] for row in threshold_runs], default=0)})
        point_rows, pair_rows = [], []
        for group, mask in zip(GROUPS, [high, ~high]):
            rows, pair = descriptive_statistics(daily.loc[mask], group, primary=(threshold == 30 and group == 'high_vix'))
            point_rows.extend(rows)
            pair_rows.append(pair)
        point, paired = pd.DataFrame(point_rows), pd.DataFrame(pair_rows)
        original = pd.DataFrame(summarize_sample(metrics, high_array)).set_index('group')
        if threshold == 30:
            pd.testing.assert_frame_equal(point, main_comparison.loc[main_comparison['group'].ne('overall')].reset_index(drop=True),
                                          check_exact=False, rtol=RTOL, atol=ATOL)
            pd.testing.assert_frame_equal(paired, main_paired.loc[main_paired['group'].ne('overall')].reset_index(drop=True),
                                          check_exact=False, rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(original, main_original.loc[GROUPS], rtol=RTOL, atol=ATOL)
        else:
            comparisons[threshold], paired_tables[threshold], original_points[threshold] = point, paired, original
        point_indexed = point.set_index(['group', 'method'])
        for group in GROUPS:
            pair = paired.set_index('group').loc[group]
            fixed, rolling = point_indexed.loc[(group, 'fixed')], point_indexed.loc[(group, 'rolling')]
            assert fixed['n'] == rolling['n'] == pair['n']
            np.testing.assert_allclose(pair[DIFFERENCE_COLUMNS].to_numpy(dtype=float),
                rolling[['empirical_coverage', 'mean_width_return_pp', 'mean_interval_score_return_pp']].to_numpy(dtype=float)
                - fixed[['empirical_coverage', 'mean_width_return_pp', 'mean_interval_score_return_pp']].to_numpy(dtype=float),
                rtol=RTOL, atol=ATOL, equal_nan=True)
        for method in ['fixed', 'rolling']:
            rows = point.loc[point['method'].eq(method)]
            overall = main_comparison.loc[main_comparison['group'].eq('overall') & main_comparison['method'].eq(method)].iloc[0]
            for column in ['n', 'covered_n', 'lower_miss_n', 'upper_miss_n']:
                assert rows[column].sum() == overall[column]
            # 中央値・absolute Gapは加重平均照合の対象にしない。
            for column in ['empirical_coverage', 'coverage_gap_pp', 'mean_width_return_pp', 'mean_interval_score_return_pp']:
                weighted = (rows.loc[rows['n'].gt(0), column] * rows.loc[rows['n'].gt(0), 'n']).sum() / N
                np.testing.assert_allclose(weighted, overall[column], rtol=RTOL, atol=ATOL)
    composition, year_composition = pd.DataFrame(composition_rows), pd.DataFrame(year_rows)
    runs, threshold_summary = pd.DataFrame(run_rows), pd.DataFrame(threshold_rows)
    root_frames = {'market_regime_indicators.csv': indicator_frame, 'sample_composition.csv': composition,
                   'yearly_composition.csv': year_composition, 'high_vix_runs.csv': runs, 'threshold_summary.csv': threshold_summary}
    for filename, frame in root_frames.items():
        save_frame(REPORT_DIR / filename, frame)
    print('Point/statistical composition checks PASS; threshold 30 and Overall will be reused.', flush=True)
    print(threshold_summary.to_string(index=False), flush=True)


    # 3. 同じ5,000組のindexで25/35を同時に再抽出。30/Overall統計・CIを再計算しない。
    for high_value, empty_group in [(False, 'high_vix'), (True, 'non_high_vix')]:
        checked = pd.DataFrame(summarize_sample(metrics[:4], np.full(4, high_value))).set_index('group')
        assert checked.loc[empty_group, 'n'] == 0 and checked.loc[empty_group, STAT_COLUMNS].isna().all()
    empty_bounds = percentile_bounds([np.nan, np.nan])
    assert np.isnan(empty_bounds[:2]).all() and empty_bounds[2:] == (0, 2)
    rng = np.random.default_rng(SEED)
    assert type(rng.bit_generator).__name__ == 'PCG64'
    replicate_rows = {threshold: [] for threshold in NEW_THRESHOLDS}
    first_block_rows, prefix_starts, prefix_indices, nesting_rows = [], [], [], []
    for replicate in range(1, B + 1):
        block_starts, sample_indices = draw_moving_blocks(rng, N, BLOCK_LENGTH)
        sampled_metrics, sampled_indicators = metrics[sample_indices], indicators[sample_indices]
        assert sampled_metrics.shape == (N, 6) and sampled_indicators.shape == (N, 3)
        assert (sampled_indicators[:, 2] <= sampled_indicators[:, 1]).all() and (sampled_indicators[:, 1] <= sampled_indicators[:, 0]).all()
        high_counts = {}
        for threshold, column in [(25, 0), (35, 2)]:
            statistics = summarize_sample(sampled_metrics, sampled_indicators[:, column])
            assert statistics[0]['n'] + statistics[1]['n'] == N
            high_counts[threshold] = statistics[0]['n']
            for row in statistics:
                replicate_rows[threshold].append({'replicate': replicate, **row})
        saved_high_30 = int(main_high_counts[replicate - 1])  # 保存件数を使用。30の統計量は再計算しない。
        assert high_counts[25] >= saved_high_30 >= high_counts[35]
        nesting_rows.append({'replicate': replicate, 'high_vix_n_25': high_counts[25], 'high_vix_n_30_reused': saved_high_30,
                             'high_vix_n_35': high_counts[35], 'nested_counts': True})
        if replicate <= 3:
            prefix_starts.append(block_starts.copy())
            prefix_indices.append(sample_indices.copy())
            saved_starts = main_starts.loc[main_starts['replicate'].eq(replicate)].sort_values('block_number')
            np.testing.assert_array_equal(block_starts, saved_starts['start_index'])
            for block_number, start in enumerate(block_starts):
                retained = min(BLOCK_LENGTH, N - block_number * BLOCK_LENGTH)
                first_block_rows.append({'replicate': replicate, 'block_number': block_number + 1, 'start_index': int(start),
                                         'end_index_before_truncation': int(start + BLOCK_LENGTH - 1), 'retained_length': retained,
                                         'retained_end_index': int(start + retained - 1), 'start_date': daily['date'].iloc[start],
                                         'end_date_before_truncation': daily['date'].iloc[start + BLOCK_LENGTH - 1]})
    first_blocks = pd.DataFrame(first_block_rows)
    pd.testing.assert_frame_equal(first_blocks, main_starts, check_exact=True)
    save_frame(REPORT_DIR / 'first_replicate_block_starts.csv', first_blocks)
    save_frame(REPORT_DIR / 'replicate_count_nesting.csv', pd.DataFrame(nesting_rows))
    replicates_by_threshold = {threshold: pd.DataFrame(rows) for threshold, rows in replicate_rows.items()}
    prefix_rows = {threshold: [] for threshold in NEW_THRESHOLDS}
    check_rng = np.random.default_rng(SEED)
    for replicate in range(1, 4):
        block_starts, sample_indices = draw_moving_blocks(check_rng, N, BLOCK_LENGTH)
        np.testing.assert_array_equal(block_starts, prefix_starts[replicate - 1])
        np.testing.assert_array_equal(sample_indices, prefix_indices[replicate - 1])
        for threshold, column in [(25, 0), (35, 2)]:
            checked = pd.DataFrame(summarize_sample(metrics[sample_indices], indicators[sample_indices, column])).set_index('group')
            saved = replicates_by_threshold[threshold].loc[lambda frame: frame['replicate'].eq(replicate)].set_index('group')
            np.testing.assert_allclose(checked, saved[['n'] + STAT_COLUMNS], rtol=0, atol=0, equal_nan=True)
            prefix_rows[threshold].append({'replicate': replicate, 'sample_length': N, 'high_vix_n': int(checked.loc['high_vix', 'n']),
                                           'non_high_vix_n': int(checked.loc['non_high_vix', 'n']),
                                           'block_starts_match_main_30': True, 'regenerated_starts_indices_statistics_exact_match': True})

    difference_tables, coverage_tables, count_tables = {30: main_difference}, {30: main_coverage}, {30: main_counts}
    for threshold in NEW_THRESHOLDS:
        run_dir = REPORT_DIR / f'threshold_{threshold}'
        run_dir.mkdir(parents=True, exist_ok=True)
        replicates = replicates_by_threshold[threshold]
        assert len(replicates) == 2 * B and replicates['replicate'].nunique() == B
        assert replicates.groupby('replicate').size().eq(2).all()
        valid = replicates['n'].gt(0)
        assert np.isfinite(replicates.loc[valid, STAT_COLUMNS]).all().all()
        assert replicates.loc[~valid, STAT_COLUMNS].isna().all().all()
        for method in ['fixed', 'rolling']:
            assert replicates.loc[valid, f'coverage_{method}'].between(0, 1).all()
            assert replicates.loc[valid, f'mean_width_{method}_return_pp'].ge(0).all()
            assert replicates.loc[valid, f'mean_interval_score_{method}_return_pp'].ge(replicates.loc[valid, f'mean_width_{method}_return_pp']).all()
        for difference, fixed, rolling in zip(DIFFERENCE_COLUMNS, MEAN_COLUMNS[::2], MEAN_COLUMNS[1::2]):
            np.testing.assert_allclose(replicates[difference], replicates[rolling] - replicates[fixed], rtol=RTOL, atol=ATOL, equal_nan=True)
        count_pivot = replicates.pivot(index='replicate', columns='group', values='n').sort_index()
        assert (count_pivot['high_vix'] + count_pivot['non_high_vix']).eq(N).all()
        # 同じindexのHigh/Non-highを合計すると、保存済みOverallの各平均・paired差と一致する。
        weighted_statistics = np.zeros((B, len(STAT_COLUMNS)))
        for group in GROUPS:
            rows = replicates.loc[replicates['group'].eq(group)].sort_values('replicate')
            values = rows[STAT_COLUMNS].to_numpy()
            weighted_statistics += np.where(rows['n'].to_numpy()[:, None] > 0, values, 0) * rows['n'].to_numpy()[:, None] / N
        np.testing.assert_allclose(weighted_statistics, main_overall[STAT_COLUMNS], rtol=RTOL, atol=ATOL)
        for group in GROUPS:
            values = replicates.loc[replicates['group'].eq(group) & valid, 'mean_width_fixed_return_pp']
            np.testing.assert_allclose(values, main_original.loc['overall', 'mean_width_fixed_return_pp'], rtol=RTOL, atol=ATOL)
        difference, coverage = build_ci_tables(replicates, original_points[threshold], B, .90)
        counts = summarize_counts(replicates, B, original_points[threshold])
        frames = {'replicate_statistics.csv': replicates, 'comparison_by_regime.csv': comparisons[threshold],
                  'paired_differences_by_regime.csv': paired_tables[threshold], 'original_point_statistics.csv': original_points[threshold].reset_index(),
                  'difference_confidence_intervals.csv': difference, 'coverage_gap_confidence_intervals.csv': coverage,
                  'replicate_group_count_summary.csv': counts, 'prefix_reproducibility_checks.csv': pd.DataFrame(prefix_rows[threshold])}
        for filename, frame in frames.items():
            save_frame(run_dir / filename, frame)
        difference_tables[threshold], coverage_tables[threshold], count_tables[threshold] = difference, coverage, counts
        metadata = {
            'executed_at_utc': datetime.now(timezone.utc).isoformat(),
            'executed_at_jst': datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat(),
            'environment': {'python_executable': sys.executable, 'python_version': sys.version, 'numpy': np.__version__,
                            'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
            'code_sha256': file_hash(Path(__file__).resolve()), 'working_directory': str(Path.cwd()),
            'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]} for name, path in INPUT_PATHS.items()},
            'settings': {'vix_threshold': threshold, 'vix_variable': 'vix_lag1', 'comparison': '>= without rounding or tolerance',
                         'primary_vix_threshold_unchanged': 30, 'rolling_window_unchanged': 252, 'B': B, 'N': N, 'block_length': BLOCK_LENGTH,
                         'blocks_drawn': 166, 'last_block_retained_rows': 8, 'seed': SEED, 'rng': 'numpy.random.default_rng',
                         'bit_generator': 'PCG64', 'same_index_for_thresholds_25_and_35': True,
                         'block_scheme': settings['block_scheme'], 'percentile_probabilities': [.025, .975], 'percentile_method': 'linear',
                         'nominal_conformal_coverage': .90, 'confidence_level': .95, 'empty_group_policy': settings['empty_group_policy']},
            'output_csv_sha256': {filename: file_hash(run_dir / filename) for filename in frames},
            'scope': 'new subgroup statistics for 25/35 only; saved 30 and Overall reused; no pipeline refit',
        }
        (run_dir / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        validation = {'status': 'PASS', 'vix_threshold': threshold, 'replicate_samples': B, 'replicate_subgroup_rows': len(replicates),
                      'sample_length': N, 'non_circular_contiguous_blocks_and_truncation_valid': True,
                      'same_index_for_methods_metrics_and_threshold_indicators': True, 'first_3_block_starts_equal_main_30': True,
                      'first_3_seed_reconstruction_exact': True, 'denominators_recomputed_each_replicate': True,
                      'counts_nested_25_ge_saved_30_ge_35': True, 'high_plus_nonhigh_n': N,
                      'paired_differences_equal_method_mean_differences': True, 'weighted_means_equal_reused_overall_each_replicate': True,
                      'empty_groups_nan_no_redraw': True, 'empty_branch_checked_with_validation_copies': True,
                      'ci_ranges_and_endpoint_order_valid': True, 'csv_round_trip_all_columns_exact': True,
                      'primary_30_statistics_or_ci_recomputed': False, 'overall_bootstrap_statistics_or_ci_recomputed': False,
                      'prior_steps_executed': False, 'daily_metrics_or_intervals_regenerated': False}
        (run_dir / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'execution.log').write_text(f'vix_lag1>={threshold}; B={B}, l=10, seed={SEED}, PCG64\nCommon indices with the other new threshold; 30/Overall reused.\n'
            + counts.to_string(index=False) + '\n' + difference.to_string(index=False) + '\n' + coverage.to_string(index=False)
            + '\nAll checks PASS; first 3 only regenerated for reproducibility.\n', encoding='utf-8')
        print(f'Threshold {threshold}: PASS; {B} samples, {len(replicates)} subgroup rows saved.', flush=True)


    # 4. 3閾値の比較表。30は全行、Overallは各閾値で保存済みの同じ行を転載する。
    table_specs = [
        ('performance_comparison.csv', comparisons, main_comparison, 'main_comparison', 'comparison_by_regime.csv'),
        ('paired_difference_comparison.csv', paired_tables, main_paired, 'main_paired', 'paired_differences_by_regime.csv'),
        ('difference_ci_comparison.csv', difference_tables, main_difference, 'main_difference_ci', 'difference_confidence_intervals.csv'),
        ('coverage_gap_ci_comparison.csv', coverage_tables, main_coverage, 'main_coverage_ci', 'coverage_gap_confidence_intervals.csv'),
        ('group_count_comparison.csv', count_tables, main_counts, 'main_counts', 'replicate_group_count_summary.csv'),
    ]
    combined = {}
    for filename, tables, main_table, input_key, new_filename in table_specs:
        pieces = []
        for threshold in THRESHOLDS:
            if threshold == 30:
                pieces.append(with_provenance(main_table, threshold, INPUT_PATHS[input_key], reused=True))
            else:
                pieces.append(with_provenance(main_table.loc[main_table['group'].eq('overall')], threshold, INPUT_PATHS[input_key], reused=True))
                pieces.append(with_provenance(tables[threshold], threshold, REPORT_DIR / f'threshold_{threshold}' / new_filename))
        table = pd.concat(pieces, ignore_index=True)
        # 元の主30全列が完全一致し、Overall全列も閾値によらず同じことを確認する。
        pd.testing.assert_frame_equal(table.loc[table['vix_threshold'].eq(30), main_table.columns].reset_index(drop=True), main_table, check_exact=True)
        for threshold in THRESHOLDS:
            reused = table.loc[table['vix_threshold'].eq(threshold) & table['group'].eq('overall'), main_table.columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(reused, main_table.loc[main_table['group'].eq('overall')].reset_index(drop=True), check_exact=True)
        if filename == 'difference_ci_comparison.csv':
            table['method'] = 'rolling_minus_fixed'
            table['ci_width'] = table['ci_upper'] - table['ci_lower']
            table['display_ci_width'] = table['display_ci_upper'] - table['display_ci_lower']
        if filename == 'coverage_gap_ci_comparison.csv':
            table['coverage_ci_width'] = table['coverage_ci_upper'] - table['coverage_ci_lower']
            for old, new in [('coverage_point_estimate', 'coverage_point_estimate_pct'), ('coverage_ci_lower', 'coverage_ci_lower_pct'),
                             ('coverage_ci_upper', 'coverage_ci_upper_pct'), ('coverage_ci_width', 'coverage_ci_width_pp')]:
                table[new] = 100 * table[old]
            table['signed_gap_ci_width_pp'] = table['signed_gap_ci_upper_pp'] - table['signed_gap_ci_lower_pp']
        save_frame(REPORT_DIR / filename, table)
        combined[filename] = table
    assert len(combined['performance_comparison.csv']) == 18 and len(combined['paired_difference_comparison.csv']) == 9
    assert len(combined['difference_ci_comparison.csv']) == 27 and len(combined['coverage_gap_ci_comparison.csv']) == 18
    high_difference = combined['difference_ci_comparison.csv'].loc[lambda frame: frame['group'].eq('high_vix')]
    high_coverage = combined['coverage_gap_ci_comparison.csv'].loc[lambda frame: frame['group'].eq('high_vix')]


    # 5. 主閾値30を明示。点とCI上下限を独立して描き、対称化・切り詰めを行わない。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for y, threshold in enumerate(THRESHOLDS):
        for offset, method, color in [(-.13, 'fixed', '#B84B25'), (.13, 'rolling', '#245A81')]:
            row = high_coverage.loc[high_coverage['vix_threshold'].eq(threshold) & high_coverage['method'].eq(method)].iloc[0]
            lower, upper, point = row['coverage_ci_lower_pct'], row['coverage_ci_upper_pct'], row['coverage_point_estimate_pct']
            position = y + offset
            if row['valid_replicates']:
                ax.hlines(position, lower, upper, color=color, linewidth=2)
                ax.vlines([lower, upper], position - .045, position + .045, color=color, linewidth=1.2)
            ax.scatter(point, position, color=color, s=32, zorder=3, label=('Fixed' if method == 'fixed' else 'Rolling-252') if y == 0 else None)
            ax.text(.01, position + .10, f'{point:.2f}% [{lower:.2f}%, {upper:.2f}%]', transform=ax.get_yaxis_transform(), color=color, fontsize=8)
    ax.axvline(90, color='#555555', linestyle='--', linewidth=1, label='Nominal interval coverage = 90%')
    high_counts_original = threshold_summary.set_index('vix_threshold')['high_vix_n']
    ax.set_yticks([0, 1, 2], [f'VIX >= {threshold} ({"primary; reused" if threshold == 30 else "sensitivity"})\nn={high_counts_original[threshold]:,}' for threshold in THRESHOLDS])
    ax.set_ylim(2.6, -.65)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Empirical coverage (%)')
    ax.set_title('High-VIX coverage by lagged-VIX threshold\nOriginal estimates and 95% percentile CI; paired MBB l=10, B=5,000', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.18)
    ax.legend(loc='upper right', fontsize=9)
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_high_vix_coverage_by_threshold.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, layout='constrained')
    for ax, (metric, title, xlabel) in zip(axes, [
        ('coverage_difference', 'Coverage difference', 'Coverage difference (percentage points)'),
        ('mean_width_difference', 'Mean width difference', 'Width difference (return percentage points)'),
        ('mean_interval_score_difference', 'Mean interval score difference', 'IS difference (return percentage points)'),
    ]):
        panel = high_difference.loc[high_difference['metric'].eq(metric)].set_index('vix_threshold')
        for y, threshold in enumerate(THRESHOLDS):
            row = panel.loc[threshold]
            lower, upper, point = row['display_ci_lower'], row['display_ci_upper'], row['display_point_estimate']
            color = '#245A81' if threshold == 30 else '#737373'
            if row['valid_replicates']:
                ax.hlines(y, lower, upper, color=color, linewidth=2, label='95% percentile CI' if y == 0 else None)
                ax.vlines([lower, upper], y - .07, y + .07, color=color, linewidth=1.2)
            ax.scatter(point, y, color=color, s=32, zorder=3, label='Original estimate' if y == 0 else None)
            ax.text(.02, y + .25, f'{point:.2f} [{lower:.2f}, {upper:.2f}]', transform=ax.get_yaxis_transform(), fontsize=9)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_yticks([0, 1, 2], [f'VIX >= {threshold} ({"primary" if threshold == 30 else "sensitivity"})\nn={high_counts_original[threshold]:,}' for threshold in THRESHOLDS])
        ax.grid(axis='x', alpha=.18)
        ax.margins(x=.12)
    axes[0].set_ylim(2.6, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('High-VIX: Rolling minus Fixed by lagged-VIX threshold\n95% percentile CI; primary threshold 30 unchanged; rolling window=252', fontsize=12)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'02_high_vix_differences_by_threshold.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)


    # 6. 入力・主分析の保全と設定を記録。主30のCIを計算する分岐はない。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    protected_hashes_after = {name: file_hash(PROJECT_DIR / name) for name in protected_hashes}
    assert input_hashes == input_hashes_after and protected_hashes == protected_hashes_after
    validation = {'status': 'PASS', 'threshold_30_indicator_exact_match': True, 'main_30_counts': [149, 1509],
                  'nested_original_indicators': True, 'nested_resample_counts_all_5000': True,
                  'threshold_30_descriptive_all_columns_match': True, 'threshold_30_ci_all_original_columns_exact_reuse': True,
                  'overall_all_original_columns_exact_reuse': True, 'same_daily_predictions_intervals_metrics_for_all_thresholds': True,
                  'original_weighted_means_equal_overall': True, 'median_and_absolute_gap_weighted_equality_required': False,
                  'replicate_weighted_means_equal_saved_overall': True, 'first_3_block_records_exact_match_main': True,
                  'first_3_only_regenerated_for_reproducibility': True, 'new_replications': B, 'new_thresholds': NEW_THRESHOLDS,
                  'main_30_replications_or_ci_recomputed': False, 'overall_replications_or_ci_recomputed': False,
                  'csv_round_trip_all_columns_exact': True, 'primary_threshold_block_length_window_unchanged': [30, 10, 252],
                  'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
                  'protected_main_hashes_before': protected_hashes, 'protected_main_hashes_after': protected_hashes_after}
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'started_at_utc': started_at.isoformat(), 'ended_at_utc': datetime.now(timezone.utc).isoformat(),
                'code_sha256': file_hash(Path(__file__).resolve()), 'new_thresholds': NEW_THRESHOLDS, 'reused_primary_threshold': 30,
                'bootstrap_draws_generated_once_for_both_thresholds': B, 'new_statistics_rows_per_threshold': 2 * B,
                'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]} for name, path in INPUT_PATHS.items()},
                'root_csv_sha256': {path.name: file_hash(path) for path in REPORT_DIR.glob('*.csv')}, 'figures': figure_files,
                'per_threshold_metadata': ['threshold_25/execution_metadata.json', 'threshold_35/execution_metadata.json']}
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


    # 7. 全閾値と非高VIXも報告する。閾値間の有意差検定や最適閾値の選択はしない。
    summary = '''# 追加検証②：前営業日VIX閾値の感度分析

## 実行範囲・方法

閾値25・35を新規計算し、主閾値30の性能表・5,000回統計・CIは保存済みCSVから再利用した。
主設定はVIX30、Rolling calibration window252、bootstrap block length10のままである。
前段階の実行、OLS・score・Conformal閾値・予測区間・各日の日次指標の再生成は行っていない。
分類は保存済みの`vix_lag1 >= threshold`。当日VIX・丸め・許容差・欠損のFalse補完を使っていない。

全Test1,658日の時系列から、非循環Paired Moving Block Bootstrapを行った。
`np.random.default_rng(202701)`/PCG64、B=5,000、block length10、開始位置0〜1,648の復元抽出、
166blockを連結し先頭1,658行を保持（最後8行）する。抽出後のソート・重複削除は行わない。
1組のindexを両手法の全保存指標とH25・H35へ同時適用した。高VIXだけに圧縮せず、区分件数と分母を毎回計算し直す。
空区分はNaN・再抽出なし・0埋めなしで、有効値の2.5/97.5%点を`np.quantile(method="linear")`で求めた。
名目Coverage90%とCIの信頼水準95%は別である。30とOverallのbootstrap統計量・CIは再計算していない。

## 標本構成と時期の集中

| VIX閾値 | 高VIX n | 非高VIX n | 高VIX割合 | 高VIX連続区間数 | 最長区間（市場日観測数） |
|---:|---:|---:|---:|---:|---:|
'''
    for row in threshold_summary.itertuples():
        summary += f'| {row.vix_threshold} | {row.high_vix_n} | {row.non_high_vix_n} | {100 * row.high_vix_share:.4f}% | {row.high_vix_runs} | {row.longest_high_vix_run_days} |\n'
    summary += '\nH35はH30に、H30はH25に含まれる。連続性は全Test上の変化点から識別し、区間数を独立な経済ショック数とは解釈しない。\n\n年別構成（割合の分母はその年の全Test観測数）：\n\n| 閾値 | 年 | 全n | 高VIX n | 年内割合 | 期間上の注意 |\n|---:|---:|---:|---:|---:|---|\n'
    for row in year_composition.itertuples():
        summary += f'| {row.vix_threshold} | {row.year} | {row.n} | {row.high_vix_n} | {100 * row.high_vix_share_within_year:.4f}% | {"8月7日までの部分年" if row.partial_year else "全暦年のTest期間"} |\n'
    summary += '\n## 高VIXの性能と差CI\n\nすべてRolling−Fixed。Coverage差はcoverage pp、Width差・IS差はreturn pp。主たる推論は閾値30のCoverage差（two-sided）であり、25/35は感度分析である。\n\n| 閾値 | 指標 | 点推定 | 95%CI | CI幅 | 0包含 | 有効/無効 |\n|---:|---|---:|---|---:|---|---|\n'
    for row in high_difference.itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {row.vix_threshold} | {row.metric} ({row.display_unit}) | {row.display_point_estimate:+.6f} | [{row.display_ci_lower:+.6f}, {row.display_ci_upper:+.6f}] | {row.display_ci_width:.6f} | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += '\nCoverageは%、signed Gapはcoverage pp。Gap CIはCoverage CIから0.90を引いて100倍する。\n\n| 閾値 | 手法 | Coverage (%) | Coverage 95%CI (%) | Gap (pp) | Gap 95%CI (pp) | 90%との関係 |\n|---:|---|---:|---|---:|---|---|\n'
    for row in high_coverage.itertuples():
        summary += f'| {row.vix_threshold} | {row.method} | {row.coverage_point_estimate_pct:.4f} | [{row.coverage_ci_lower_pct:.4f}, {row.coverage_ci_upper_pct:.4f}] | {row.signed_gap_point_estimate_pp:+.4f} | [{row.signed_gap_ci_lower_pp:+.4f}, {row.signed_gap_ci_upper_pp:+.4f}] | {row.ci_relation_to_nominal} |\n'
    summary += '\n## 維持された／変化した解釈\n\n'
    for threshold in THRESHOLDS:
        selected = high_difference.loc[high_difference['vix_threshold'].eq(threshold)].set_index('metric')
        messages = []
        for metric, label, positive in [('coverage_difference', 'Coverage差の正方向', True), ('mean_width_difference', 'Width差の正方向', True),
                                         ('mean_interval_score_difference', 'IS差の負方向', False)]:
            row = selected.loc[metric]
            if row['valid_replicates'] == 0:
                result = '有効値なし'
            elif (row['ci_lower'] > 0 if positive else row['ci_upper'] < 0):
                result = 'CIが0を除外して支持'
            elif row['ci_includes_zero']:
                result = 'CIが0を含む'
            else:
                result = 'CIが反対方向にある'
            messages.append(label + '：' + result)
        methods = high_coverage.loc[high_coverage['vix_threshold'].eq(threshold)]
        diagnoses = ', '.join(f'{row.method}={row.ci_relation_to_nominal}' for row in methods.itertuples())
        summary += f'- 閾値{threshold}：' + '、'.join(messages) + f'。Coverage診断：{diagnoses}。\n'
    summary += '''
閾値を変えると評価対象の日と市場環境の構成自体が変わるので、点推定・差の大きさは同じになる必要がない。
対象日は入れ子で、同じbootstrap indexも共有するため、閾値別推定結果は独立ではない。
CIの重なりだけから閾値間の差の有意性を判断せず、閾値間差の新たな検定や最適閾値の選択は行っていない。
元観測数、年別集中、連続区間、再標本件数の分布とCIを併せて読む。変化した結果は高VIX定義への感度としてそのまま報告する。
CIが90%を含むことをCoverage達成の証明とは解釈しない。Overallの90%への近さで高VIXの不足を覆い隠さない。

## 全閾値の記述統計（Overallは共通の既存結果）

| 閾値 | 区分 | 手法 | n | covered | Coverage (%) | signed Gap (pp) | absolute Gap (pp) | Mean Width | Median Width | Mean IS |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in combined['performance_comparison.csv'].itertuples():
        summary += f'| {row.vix_threshold} | {row.group} | {row.method} | {row.n} | {row.covered_n} | {100 * row.empirical_coverage:.4f} | {row.coverage_gap_pp:+.4f} | {row.absolute_coverage_gap_pp:.4f} | {row.mean_width_return_pp:.6f} | {row.median_width_return_pp:.6f} | {row.mean_interval_score_return_pp:.6f} |\n'
    summary += '\nWidth・ISの単位はreturn pp。非高VIXの補助的な差も全件示す：\n\n| 閾値 | 指標 | 点推定 | 95%CI | 0包含 | 有効/無効 |\n|---:|---|---:|---|---|---|\n'
    for row in combined['difference_ci_comparison.csv'].loc[lambda frame: frame['group'].eq('non_high_vix')].itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {row.vix_threshold} | {row.metric} ({row.display_unit}) | {row.display_point_estimate:+.6f} | [{row.display_ci_lower:+.6f}, {row.display_ci_upper:+.6f}] | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += '\n非高VIXの各手法Coverage・Gap CIと、閾値非依存のOverall CIも、比較CSVへ全件保存している。\n\n## 再標本の区分件数\n\n| 閾値 | 区分 | 有効/無効 | 最小 | 平均 | 中央値 | 最大 | 件数2.5/97.5%点 |\n|---:|---|---|---:|---:|---:|---:|---|\n'
    for row in combined['group_count_comparison.csv'].itertuples():
        summary += f'| {row.vix_threshold} | {row.group} | {row.valid_group_replicates}/{row.zero_group_replicates} | {row.min_n} | {row.mean_n:.4f} | {row.n_median:.2f} | {row.max_n} | [{row.n_q025:.2f}, {row.n_q975:.2f}] |\n'
    summary += '''
## 検証・保存先・限界

主30のindicator・149/1,509日・記述統計の全列を既存結果と照合した。比較表では主30の保存値を出所path/SHA-256付きで転載し、CI全元列の完全一致を確認した。
Overallも同じ既存結果を再利用した。元の高/非高の件数加重平均はOverallと一致し、中央値・absolute Gapには加重平均一致を要求しなかった。
全5,000回で長さ・開始位置・連続block・切り詰め・pairing・件数合計・入れ子の件数・paired差を確認した。
高/非高の各回の加重平均は、同じindexによる保存済みOverall統計と一致した。
最初3回のblock開始位置だけでなく全監査列が主設定の保存記録と完全一致した。再現性は最初3回だけ再生成して確認した。
空区分/全NaN CIの分岐は検証用コピーで確認し、研究データを変更していない。保存CSVは全列round-trip一致を確認した。
入力と保護対象の主分析成果物は実行前後のハッシュが一致した。未解決の実装・入力上の不整合は見つからなかった。

- `17_vix_threshold_sensitivity.py`：単独実行コード。前段階をimportしない。
- `reports/vix_threshold_sensitivity/threshold_25/`、`threshold_35/`：各5,000回×高/非高2区分=10,000行の新規統計、点・CI・件数要約・再現性・設定・環境・入力ハッシュ・検証・実行ログ。
- 親フォルダ：`market_regime_indicators.csv`、`sample_composition.csv`、`threshold_summary.csv`、`yearly_composition.csv`、`high_vix_runs.csv`。
- 親フォルダ：`performance_comparison.csv`、`paired_difference_comparison.csv`、`difference_ci_comparison.csv`、`coverage_gap_ci_comparison.csv`、`group_count_comparison.csv`。全行に出所path/SHA-256。
- 親フォルダ：`first_replicate_block_starts.csv`、`replicate_count_nesting.csv`、`validation_results.json`、`execution_metadata.json`、`execution.log`。
- `figures/vix_threshold_sensitivity/`：高VIX Coverage CIと3差CIの図。各PNG 300dpi・PDF。目視確認は実行後、同reportsフォルダの`figure_visual_qa.md`に記録する。

実行：`C:\\Users\\sohas\\anaconda3\\python.exe 17_vix_threshold_sensitivity.py`。
本分析も保存済みモデル・区間・評価系列に条件づけた近似的評価で、学習から再推定する全工程の不確実性は含まない。
stationarity・weak dependence・block length等に依存し、時系列依存や非定常性の問題が解消した、90%Coverageが保証されたとはいえない。
今回調べたVIX定義での感度を示すもので、あらゆるstress定義・市場環境への頑健性は主張しない。
主閾値30を維持し、block lengthとの全組合せ、Rolling window63/126、他モデルの感度分析は実行していない。
'''
    (PROJECT_DIR / 'reports/vix_threshold_sensitivity_summary.md').write_text(summary, encoding='utf-8')
    print('\nHigh-VIX differences:\n' + high_difference.to_string(index=False))
    print('\nHigh-VIX coverage:\n' + high_coverage.to_string(index=False))
    print('VIX threshold sensitivity: PASS. New 25/35 only; 30 and Overall reused; visual QA to follow.')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 17.')
    run_vix_threshold_sensitivity(ANALYSIS_WORKSPACE)
