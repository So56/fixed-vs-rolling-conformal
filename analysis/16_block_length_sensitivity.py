"""MBB block length 5/20を検証し、保存済み主設定10と比較する。
実行入口はrun_analysis.pyのstage 16。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_block_length_sensitivity(workspace):
    """保存済み入力を独立workspaceで検証し、この段階の結果だけを保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports/bootstrap_sensitivity'
    FIGURE_DIR = PROJECT_DIR / 'figures/bootstrap_sensitivity'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'daily_evaluation': PROJECT_DIR / 'reports/evaluation/daily_test_evaluation.csv',
        'main_difference_ci': PROJECT_DIR / 'reports/bootstrap/difference_confidence_intervals.csv',
        'main_coverage_ci': PROJECT_DIR / 'reports/bootstrap/coverage_gap_confidence_intervals.csv',
        'main_original': PROJECT_DIR / 'reports/bootstrap/original_point_statistics.csv',
        'main_counts': PROJECT_DIR / 'reports/bootstrap/replicate_group_count_summary.csv',
        'main_metadata': PROJECT_DIR / 'reports/bootstrap/execution_metadata.json',
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
    METRIC_SPECS = [
        ('coverage_difference', DIFFERENCE_COLUMNS[0], 'proportion', 'coverage percentage points', 100.),
        ('mean_width_difference', DIFFERENCE_COLUMNS[1], 'return percentage points', 'return percentage points', 1.),
        ('mean_interval_score_difference', DIFFERENCE_COLUMNS[2], 'return percentage points', 'return percentage points', 1.),
    ]
    RTOL = ATOL = 1e-12  # 保存済み統計量との数値照合に限って使用。




    # 設定ごとの処理を読めるよう、小さい計算関数はこのスクリプト内に保持する。
    def summarize_sample(sample):
        high = sample[:, 0].astype(bool)
        rows = []
        for group, mask in zip(GROUPS, [np.ones(len(sample), dtype=bool), high, ~high]):
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
                role = 'primary_two_sided' if group == 'high_vix' and metric == 'coverage_difference' else (
                    'secondary_high_vix' if group == 'high_vix' else 'auxiliary')
                point = float(original.loc[group, column])
                assert n_valid + n_invalid == b
                if n_valid:
                    assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
                    if metric == 'coverage_difference':
                        assert -1 <= lower <= upper <= 1
                difference_rows.append({
                    'group': group, 'n_original': EXPECTED_N[group], 'metric': metric, 'analysis_role': role,
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
                    'group': group, 'method': method, 'n_original': EXPECTED_N[group],
                    'analysis_role': 'diagnostic_high_vix' if group == 'high_vix' else 'auxiliary_diagnostic',
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
        assert len(difference_ci) == 9 and len(coverage_ci) == 6
        return difference_ci, coverage_ci


    def summarize_counts(replicates, b):
        rows = []
        for group in GROUPS:
            counts = replicates.loc[replicates['group'].eq(group), 'n'].to_numpy()
            quantiles = np.quantile(counts, [.025, .25, .5, .75, .975], method='linear')
            rows.append({
                'group': group, 'n_original': EXPECTED_N[group], 'generated_replicates': b,
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


    def with_provenance(frame, block_length, path):
        result = frame.copy()
        result.insert(0, 'block_length', block_length)
        result['setting_role'] = 'primary_setting' if block_length == 10 else 'sensitivity_setting'
        result['result_origin'] = 'reused_existing' if block_length == 10 else 'newly_computed'
        result['source_file'] = path.relative_to(PROJECT_DIR).as_posix()
        result['source_sha256'] = file_hash(path)
        return result


    # 1. 入力確認。主設定をimportせず、保存済み点推定・CIの完全精度を使用する。
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    # 主設定の全成果物についても、上書きしていないことを小さなハッシュ記録で確認する。
    main_paths = sorted((PROJECT_DIR / 'reports/bootstrap').glob('*')) + sorted((PROJECT_DIR / 'figures/bootstrap').glob('*'))
    main_hashes = {path.relative_to(PROJECT_DIR).as_posix(): file_hash(path) for path in main_paths if path.is_file()}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    main_metadata = json.loads(INPUT_PATHS['main_metadata'].read_text(encoding='utf-8'))
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert design['bootstrap_repetitions'] == 5000 and design['block_length'] == 10 and design['seed'] == 202701
    assert design['sensitivity_block_lengths'] == [5, 10, 20] and design['rolling_window'] == 252
    assert design['primary_hypothesis'] == 'two-sided high-VIX coverage difference'
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30' and design['nominal_coverage'] == .90
    settings = main_metadata['settings']
    assert settings['B_generated'] == 5000 and settings['N'] == 1658 and settings['block_length_observations'] == 10
    assert settings['seed'] == 202701 and settings['rng'] == 'numpy.random.default_rng' and settings['bit_generator'] == 'PCG64'
    assert settings['percentile_probabilities'] == [.025, .975] and settings['percentile_method'] == 'numpy.quantile(method=linear)'
    assert settings['confidence_level'] == .95 and settings['nominal_conformal_coverage'] == .90
    assert settings['block_scheme'] == 'ordinary non-circular overlapping MBB; starts sampled with replacement; concatenate then retain first N'
    for key in ['daily_evaluation', 'research_design_lock', 'research_design_source']:
        assert input_hashes[key] == main_metadata['input_files'][key]['sha256']
    for key in ['main_difference_ci', 'main_coverage_ci', 'main_original', 'main_counts']:
        assert input_hashes[key] == main_metadata['output_csv_sha256'][INPUT_PATHS[key].name]
    B, SEED, NOMINAL_COVERAGE = int(settings['B_generated']), int(settings['seed']), float(settings['nominal_conformal_coverage'])
    boolean_columns = ['high_vix_30', 'covered_fixed', 'covered_rolling']
    daily = pd.read_csv(
        INPUT_PATHS['daily_evaluation'], usecols=['date', 'forecast_origin_date', 'split', 'vix_lag1'] + VECTOR_COLUMNS,
        parse_dates=['date', 'forecast_origin_date'], float_precision='round_trip',
        dtype={**{column: 'boolean' for column in boolean_columns}, 'split': 'string'},
    )
    N = len(daily)
    assert N == 1658 and daily['date'].is_unique and daily['date'].is_monotonic_increasing
    assert daily['date'].iloc[0] == pd.Timestamp('2020-01-02') and daily['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert not daily.isna().any().any() and daily['split'].eq('test').all()
    assert daily['forecast_origin_date'].lt(daily['date']).all()
    assert daily['high_vix_30'].equals(daily['vix_lag1'].ge(30).astype('boolean'))
    assert int(daily['high_vix_30'].sum()) == 149 and int((~daily['high_vix_30']).sum()) == 1509
    for column in boolean_columns:
        assert str(daily[column].dtype) == 'boolean' and daily[column].isin([True, False]).all()
    paired_values = daily[VECTOR_COLUMNS].to_numpy(dtype=float)
    assert paired_values.shape == (N, 7) and np.isfinite(paired_values).all() and np.isfinite(daily['vix_lag1']).all()
    assert np.isin(paired_values[:, :3], [0., 1.]).all()
    original = pd.read_csv(INPUT_PATHS['main_original'], float_precision='round_trip').set_index('group')
    calculated = pd.DataFrame(summarize_sample(paired_values)).set_index('group')
    assert list(original.index) == GROUPS and list(calculated.index) == GROUPS
    np.testing.assert_array_equal(original['n'], [EXPECTED_N[group] for group in GROUPS])
    np.testing.assert_allclose(calculated, original, rtol=RTOL, atol=ATOL)
    main_difference = pd.read_csv(INPUT_PATHS['main_difference_ci'], float_precision='round_trip', dtype={'ci_includes_zero': 'boolean'})
    main_coverage = pd.read_csv(INPUT_PATHS['main_coverage_ci'], float_precision='round_trip', dtype={
        'coverage_ci_contains_nominal': 'boolean', 'gap_ci_contains_zero': 'boolean'})
    main_counts = pd.read_csv(INPUT_PATHS['main_counts'], float_precision='round_trip')
    assert len(main_difference) == 9 and not main_difference.duplicated(['group', 'metric']).any()
    assert len(main_coverage) == 6 and not main_coverage.duplicated(['group', 'method']).any()
    for group in GROUPS:
        for metric, column, unit, display_unit, scale in METRIC_SPECS:
            row = main_difference.loc[main_difference['group'].eq(group) & main_difference['metric'].eq(metric)].iloc[0]
            assert row['point_estimate'] == original.loc[group, column]
            assert row['unit'] == unit and row['display_unit'] == display_unit
            assert row['display_point_estimate'] == scale * row['point_estimate']
        for method in ['fixed', 'rolling']:
            row = main_coverage.loc[main_coverage['group'].eq(group) & main_coverage['method'].eq(method)].iloc[0]
            assert row['coverage_point_estimate'] == original.loc[group, f'coverage_{method}']
    # NaN分岐は検証用コピーに限る。生成する5,000回には混ぜない。
    for high_value, empty_group in [(0., 'high_vix'), (1., 'non_high_vix')]:
        artificial = paired_values[:4].copy()
        artificial[:, 0] = high_value
        checked = pd.DataFrame(summarize_sample(artificial)).set_index('group')
        assert checked.loc[empty_group, 'n'] == 0 and checked.loc[empty_group, STAT_COLUMNS].isna().all()
    empty_bounds = percentile_bounds([np.nan, np.nan])
    assert np.isnan(empty_bounds[:2]).all() and empty_bounds[2:] == (0, 2)
    print('Input checks PASS. l=10 is read from existing CI tables; no prior scripts imported/executed.', flush=True)


    # 2. 新規の5/20だけ計算する。設定ごとに同じseedで独立したgeneratorを初期化。
    differences_by_length = {10: main_difference}
    coverage_by_length = {10: main_coverage}
    counts_by_length = {10: main_counts}
    for block_length in [5, 20]:
        started_at = datetime.now(timezone.utc)
        run_dir = REPORT_DIR / f'block_{block_length}'
        run_dir.mkdir(parents=True, exist_ok=True)
        number_of_blocks = math.ceil(N / block_length)
        last_retained = N - (number_of_blocks - 1) * block_length
        assert (number_of_blocks, last_retained) == {5: (332, 3), 20: (83, 18)}[block_length]
        rng = np.random.default_rng(SEED)
        assert type(rng.bit_generator).__name__ == 'PCG64'
        replicate_rows, first_block_rows, prefix_starts, prefix_indices = [], [], [], []
        print(f'l={block_length}: generating B={B}, blocks={number_of_blocks}, last retained={last_retained}', flush=True)
        for replicate in range(1, B + 1):
            block_starts, sample_indices = draw_moving_blocks(rng, N, block_length)
            sample = paired_values[sample_indices]  # 全7列に共通。ソート・重複削除はしない。
            assert sample.shape == (N, 7)
            statistics = summarize_sample(sample)
            assert statistics[0]['n'] == N and statistics[1]['n'] + statistics[2]['n'] == N
            for row in statistics:
                replicate_rows.append({'replicate': replicate, **row})
            if replicate <= 3:
                prefix_starts.append(block_starts.copy())
                prefix_indices.append(sample_indices.copy())
                for block_number, start in enumerate(block_starts):
                    retained = min(block_length, N - block_number * block_length)
                    first_block_rows.append({
                        'replicate': replicate, 'block_number': block_number + 1, 'start_index': int(start),
                        'end_index_before_truncation': int(start + block_length - 1), 'retained_length': retained,
                        'retained_end_index': int(start + retained - 1), 'start_date': daily['date'].iloc[start],
                        'end_date_before_truncation': daily['date'].iloc[start + block_length - 1],
                    })
        replicates = pd.DataFrame(replicate_rows)
        assert len(replicates) == 3 * B and replicates['replicate'].nunique() == B
        assert replicates.groupby('replicate').size().eq(3).all()
        count_pivot = replicates.pivot(index='replicate', columns='group', values='n')
        assert count_pivot['overall'].eq(N).all() and (count_pivot['high_vix'] + count_pivot['non_high_vix']).eq(N).all()
        valid_rows = replicates['n'].gt(0)
        assert np.isfinite(replicates.loc[valid_rows, STAT_COLUMNS]).all().all()
        assert replicates.loc[~valid_rows, STAT_COLUMNS].isna().all().all()
        for method in ['fixed', 'rolling']:
            assert replicates.loc[valid_rows, f'coverage_{method}'].between(0, 1).all()
            assert replicates.loc[valid_rows, f'mean_width_{method}_return_pp'].ge(0).all()
            assert replicates.loc[valid_rows, f'mean_interval_score_{method}_return_pp'].ge(
                replicates.loc[valid_rows, f'mean_width_{method}_return_pp']).all()
        assert replicates.loc[valid_rows, DIFFERENCE_COLUMNS[0]].between(-1, 1).all()
        for difference, fixed_column, rolling_column in zip(DIFFERENCE_COLUMNS, MEAN_COLUMNS[::2], MEAN_COLUMNS[1::2]):
            np.testing.assert_allclose(replicates[difference], replicates[rolling_column] - replicates[fixed_column],
                                       rtol=RTOL, atol=ATOL, equal_nan=True)
        fixed_width_errors = {}
        for group in GROUPS:
            values = replicates.loc[replicates['group'].eq(group) & valid_rows, 'mean_width_fixed_return_pp']
            np.testing.assert_allclose(values, original.loc[group, 'mean_width_fixed_return_pp'], rtol=RTOL, atol=ATOL)
            fixed_width_errors[group] = float((values - original.loc[group, 'mean_width_fixed_return_pp']).abs().max())

        # 最初の3回だけ再生成する。主l10や全5,000回の再実行は行わない。
        check_rng = np.random.default_rng(SEED)
        prefix_rows = []
        for replicate in range(1, 4):
            block_starts, sample_indices = draw_moving_blocks(check_rng, N, block_length)
            np.testing.assert_array_equal(block_starts, prefix_starts[replicate - 1])
            np.testing.assert_array_equal(sample_indices, prefix_indices[replicate - 1])
            checked = pd.DataFrame(summarize_sample(paired_values[sample_indices])).set_index('group').loc[GROUPS]
            saved = replicates.loc[replicates['replicate'].eq(replicate)].set_index('group').loc[GROUPS]
            np.testing.assert_array_equal(checked['n'], saved['n'])
            np.testing.assert_allclose(checked[STAT_COLUMNS], saved[STAT_COLUMNS], rtol=0, atol=0, equal_nan=True)
            prefix_rows.append({
                'replicate': replicate, 'sample_length': N, 'number_of_blocks': number_of_blocks,
                'high_vix_n': int(checked.loc['high_vix', 'n']), 'non_high_vix_n': int(checked.loc['non_high_vix', 'n']),
                'block_starts_exact_match': True, 'sample_indices_exact_match': True, 'statistics_exact_match': True,
            })
        difference_ci, coverage_ci = build_ci_tables(replicates, original, B, NOMINAL_COVERAGE)
        count_summary = summarize_counts(replicates, B)
        frames = {
            'replicate_statistics.csv': replicates, 'original_point_statistics.csv': original.reset_index(),
            'difference_confidence_intervals.csv': difference_ci, 'coverage_gap_confidence_intervals.csv': coverage_ci,
            'replicate_group_count_summary.csv': count_summary, 'first_replicate_block_starts.csv': pd.DataFrame(first_block_rows),
            'prefix_reproducibility_checks.csv': pd.DataFrame(prefix_rows),
        }
        for filename, frame in frames.items():
            save_frame(run_dir / filename, frame)
        hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
        assert hashes_after == input_hashes
        ended_at = datetime.now(timezone.utc)
        validation = {
            'status': 'PASS', 'block_length': block_length, 'generated_replicates': B, 'replicate_rows': len(replicates),
            'resample_length_each_replicate': N, 'blocks_per_replicate': number_of_blocks, 'last_block_retained_rows': last_retained,
            'start_range_inclusive': [0, N - block_length], 'contiguous_non_circular_blocks_with_replacement': True,
            'same_indices_for_whole_paired_vector': True, 'resampled_order_sorted_or_deduplicated': False,
            'group_counts_sum_to_N_each_replicate': True, 'group_denominators_recomputed_each_replicate': True,
            'paired_differences_match_method_means': True, 'coverage_and_ci_ranges_and_order_valid': True,
            'fixed_width_max_abs_deviation': fixed_width_errors, 'empty_group_statistics_nan_no_redraw': True,
            'empty_group_branch_checked_with_validation_copies': True, 'redrawn_replicates': 0,
            'prefix_regenerated_replicates': 3, 'prefix_starts_indices_statistics_exact_match': True,
            'csv_all_columns_exact_round_trip': True, 'point_estimates_reused_from_main_full_precision': True,
            'point_estimates_match_daily_vector': True, 'original_point_inside_ci_required': False,
            'ci_width_monotonicity_required': False, 'ci_adjusted_or_symmetrized': False,
            'main_block_10_recomputed': False, 'prior_steps_executed': False,
            'comparison_tolerance': {'rtol': RTOL, 'atol': ATOL}, 'input_hashes_before': input_hashes, 'input_hashes_after': hashes_after,
        }
        (run_dir / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
        metadata = {
            'started_at_utc': started_at.isoformat(), 'ended_at_utc': ended_at.isoformat(),
            'ended_at_jst': ended_at.astimezone(timezone(timedelta(hours=9))).isoformat(),
            'environment': {'python_executable': sys.executable, 'python_version': sys.version, 'numpy': np.__version__,
                            'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
            'code_sha256': file_hash(Path(__file__).resolve()), 'working_directory': str(Path.cwd()),
            'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]}
                            for name, path in INPUT_PATHS.items()},
            'settings': {'B_generated': B, 'N': N, 'block_length_observations': block_length,
                         'blocks_drawn': number_of_blocks, 'last_block_retained_rows': last_retained,
                         'seed': SEED, 'rng': 'numpy.random.default_rng', 'bit_generator': type(rng.bit_generator).__name__,
                         'new_generator_initialized_for_each_block_length': True,
                         'block_scheme': settings['block_scheme'],
                         'rng_integers': 'low=0, high=N-l+1 (exclusive), dtype=int64, size=ceil(N/l)',
                         'nominal_conformal_coverage': NOMINAL_COVERAGE, 'confidence_level': .95,
                         'percentile_probabilities': [.025, .975], 'percentile_method': 'numpy.quantile(method=linear)',
                         'empty_group_policy': settings['empty_group_policy'], 'rolling_calibration_window_unchanged': 252},
            'paired_vector_columns_in_order': VECTOR_COLUMNS, 'statistic_columns': STAT_COLUMNS,
            'output_csv_sha256': {filename: file_hash(run_dir / filename) for filename in frames},
            'scope': 'block length sensitivity only; main block 10 reused; upstream pipeline fixed',
        }
        (run_dir / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        run_log = (
            f'Start UTC: {started_at.isoformat()}\nEnd UTC: {ended_at.isoformat()}\nPython: {sys.executable}\n'
            f'l={block_length}, B={B}, N={N}, seed={SEED}, PCG64, non-circular paired MBB\n'
            f'Blocks={number_of_blocks}; start range=0..{N - block_length}; final retained={last_retained}\n'
            '95% percentile CI, method=linear. Prefix reproducibility checked for first 3 only.\n'
            'No prior scripts imported/executed; no redraw; main l=10 not recomputed.\n'
            + count_summary.to_string(index=False) + '\n' + difference_ci.to_string(index=False)
            + '\n' + coverage_ci.to_string(index=False) + '\nValidation/round-trip/input hashes: PASS\n'
        )
        (run_dir / 'execution.log').write_text(run_log, encoding='utf-8')
        differences_by_length[block_length], coverage_by_length[block_length], counts_by_length[block_length] = difference_ci, coverage_ci, count_summary
        print(f'l={block_length}: PASS; generated {B}, saved {len(replicates)} group rows; first 3 reproduced.', flush=True)


    # 3. 主10は元のCIを変更せず転載。各行に参照元とハッシュを付ける。
    difference_frames, coverage_frames, count_frames = [], [], []
    for block_length in [5, 10, 20]:
        source_dir = PROJECT_DIR / 'reports/bootstrap' if block_length == 10 else REPORT_DIR / f'block_{block_length}'
        difference = with_provenance(differences_by_length[block_length], block_length, source_dir / 'difference_confidence_intervals.csv')
        difference['method'] = 'rolling_minus_fixed'
        difference['ci_width'] = difference['ci_upper'] - difference['ci_lower']
        difference['display_ci_width'] = difference['display_ci_upper'] - difference['display_ci_lower']
        coverage = with_provenance(coverage_by_length[block_length], block_length, source_dir / 'coverage_gap_confidence_intervals.csv')
        coverage['percentile_method'] = 'linear'
        coverage['coverage_ci_width'] = coverage['coverage_ci_upper'] - coverage['coverage_ci_lower']
        for old, new in [('coverage_point_estimate', 'coverage_point_estimate_pct'),
                         ('coverage_ci_lower', 'coverage_ci_lower_pct'), ('coverage_ci_upper', 'coverage_ci_upper_pct'),
                         ('coverage_ci_width', 'coverage_ci_width_pp')]:
            coverage[new] = 100 * coverage[old]
        coverage['signed_gap_ci_width_pp'] = coverage['signed_gap_ci_upper_pp'] - coverage['signed_gap_ci_lower_pp']
        counts = with_provenance(counts_by_length[block_length], block_length, source_dir / 'replicate_group_count_summary.csv')
        difference_frames.append(difference)
        coverage_frames.append(coverage)
        count_frames.append(counts)
    difference_comparison = pd.concat(difference_frames, ignore_index=True)
    coverage_comparison = pd.concat(coverage_frames, ignore_index=True)
    count_comparison = pd.concat(count_frames, ignore_index=True)
    assert len(difference_comparison) == 27 and len(coverage_comparison) == 18 and len(count_comparison) == 9
    assert difference_comparison.groupby(['group', 'metric'])['point_estimate'].nunique().eq(1).all()
    assert coverage_comparison.groupby(['group', 'method'])['coverage_point_estimate'].nunique().eq(1).all()
    # 主10の既存全列がそのまま転載されたことを確認。quantileの再計算はしない。
    pd.testing.assert_frame_equal(difference_comparison.loc[difference_comparison['block_length'].eq(10), main_difference.columns].reset_index(drop=True), main_difference, check_exact=True)
    pd.testing.assert_frame_equal(coverage_comparison.loc[coverage_comparison['block_length'].eq(10), main_coverage.columns].reset_index(drop=True), main_coverage, check_exact=True)
    for table in [difference_comparison, coverage_comparison]:
        assert (table['valid_replicates'] + table['invalid_replicates']).eq(B).all()
    comparison_frames = {
        'difference_ci_comparison.csv': difference_comparison,
        'coverage_gap_ci_comparison.csv': coverage_comparison,
        'group_count_comparison.csv': count_comparison,
    }
    for filename, frame in comparison_frames.items():
        save_frame(REPORT_DIR / filename, frame)


    # 4. 点とCI上下限を別描画し、対称化しない。0/90%も表示範囲に含める。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    high_differences = difference_comparison.loc[difference_comparison['group'].eq('high_vix')]
    high_coverage = coverage_comparison.loc[coverage_comparison['group'].eq('high_vix')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, layout='constrained')
    panel_specs = [
        ('coverage_difference', 'Coverage difference', 'Coverage difference (percentage points)'),
        ('mean_width_difference', 'Mean width difference', 'Width difference (return percentage points)'),
        ('mean_interval_score_difference', 'Mean interval score difference', 'IS difference (return percentage points)'),
    ]
    for ax, (metric, title, xlabel) in zip(axes, panel_specs):
        panel = high_differences.loc[high_differences['metric'].eq(metric)].set_index('block_length')
        for y, block_length in enumerate([5, 10, 20]):
            row = panel.loc[block_length]
            lower, upper, point = row['display_ci_lower'], row['display_ci_upper'], row['display_point_estimate']
            color = '#245A81' if block_length == 10 else '#737373'
            if row['valid_replicates'] > 0:
                ax.hlines(y, lower, upper, color=color, linewidth=2, label='95% percentile CI' if y == 0 else None)
                ax.vlines([lower, upper], y - .07, y + .07, color=color, linewidth=1.2)
            ax.scatter(point, y, color=color, s=32, zorder=3, label='Original estimate' if y == 0 else None)
            ax.text(.02, y + .25, f'{point:.2f} [{lower:.2f}, {upper:.2f}]', transform=ax.get_yaxis_transform(), fontsize=9)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_yticks([0, 1, 2], ['l=5 (sensitivity)', 'l=10 (primary; reused)', 'l=20 (sensitivity)'])
        ax.grid(axis='x', alpha=.18)
        ax.margins(x=.12)
    axes[0].set_ylim(2.6, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('High-VIX: Rolling minus Fixed by MBB block length\n95% percentile CI; B=5,000 per setting; same original n=149', fontsize=12)
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_high_vix_difference_ci_by_block_length.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for y, block_length in enumerate([5, 10, 20]):
        for offset, method, color in [(-.13, 'fixed', '#B84B25'), (.13, 'rolling', '#245A81')]:
            row = high_coverage.loc[high_coverage['block_length'].eq(block_length) & high_coverage['method'].eq(method)].iloc[0]
            lower, upper, point = row['coverage_ci_lower_pct'], row['coverage_ci_upper_pct'], row['coverage_point_estimate_pct']
            position = y + offset
            if row['valid_replicates'] > 0:
                ax.hlines(position, lower, upper, color=color, linewidth=2)
                ax.vlines([lower, upper], position - .045, position + .045, color=color, linewidth=1.2)
            ax.scatter(point, position, color=color, s=32, zorder=3, label='Fixed' if y == 0 and method == 'fixed' else (
                'Rolling-252' if y == 0 and method == 'rolling' else None))
            ax.text(.01, position + .10, f'{point:.2f}% [{lower:.2f}%, {upper:.2f}%]', transform=ax.get_yaxis_transform(), color=color, fontsize=8)
    ax.axvline(90, color='#555555', linestyle='--', linewidth=1, label='Nominal interval coverage = 90%')
    ax.set_yticks([0, 1, 2], ['l=5 (sensitivity)', 'l=10 (primary; reused)', 'l=20 (sensitivity)'])
    ax.set_ylim(2.55, -.65)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Empirical coverage (%)')
    ax.set_title('High-VIX coverage by MBB block length\nOriginal estimates and 95% percentile CI; B=5,000 per setting; original n=149', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.18)
    ax.legend(loc='upper right', fontsize=9)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'02_high_vix_coverage_ci_by_block_length.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)


    # 5. 入力と主設定を保全した記録。l10の再標本化を行う経路はない。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    main_hashes_after = {name: file_hash(PROJECT_DIR / name) for name in main_hashes}
    assert input_hashes_after == input_hashes and main_hashes_after == main_hashes
    validation = {
        'status': 'PASS', 'new_block_lengths_computed': [5, 20], 'primary_block_length_reused': 10,
        'main_block_10_recomputed': False, 'main_rows_all_original_columns_exact_match': True,
        'original_point_estimates_same_for_all_block_lengths': True, 'comparison_csv_round_trip_exact': True,
        'ci_width_monotonicity_required': False, 'original_point_inside_ci_required': False,
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'protected_main_hashes_before': main_hashes, 'protected_main_hashes_after': main_hashes_after,
        'primary_setting_unchanged': True, 'prior_steps_executed': False,
        'vix_threshold_or_rolling_window_or_model_sensitivity_run': False,
    }
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': file_hash(Path(__file__).resolve()),
        'new_settings': [5, 20], 'reused_setting': 10, 'main_metadata_file': INPUT_PATHS['main_metadata'].relative_to(PROJECT_DIR).as_posix(),
        'main_metadata_sha256': input_hashes['main_metadata'], 'input_hashes': input_hashes,
        'comparison_csv_sha256': {filename: file_hash(REPORT_DIR / filename) for filename in comparison_frames},
        'figures': figure_files, 'per_setting_metadata': ['block_5/execution_metadata.json', 'block_20/execution_metadata.json'],
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


    # 6. 全設定・全区分を報告する。CI幅や0包含が設定依存でも主設定を選び直さない。
    summary = '''# 追加検証①：Moving Block Bootstrapのblock length感度分析

## 実行範囲と方法

事前指定されたblock length 5・10・20を比較した。新規計算は5と20の各5,000回だけで、
主設定10は`reports/bootstrap/`の既存CSVから完全精度の点推定・CIを読み込み、再計算していない。
主分析は10のままである。各比較行に出所・ファイル・SHA-256を保存した。

全Test 1,658日（2020-01-02〜2026-08-07）、高VIX149日、非高VIX1,509日。
各新規設定の開始時に独立した`np.random.default_rng(202701)`（PCG64）を初期化した。
同じseedは再現性のためであり、異なるblock length間で再標本が同一という意味ではない。
全時系列から非循環の連続block開始位置を復元抽出し、連結した先頭1,658行を保持する。

| l（市場日観測） | 開始index範囲 | 抽出block数 | 最後のblock保持行数 | 出所 |
|---:|---|---:|---:|---|
| 5 | 0〜1,653 | 332 | 3 | 新規 |
| 10 | 0〜1,648 | 166 | 8 | 既存の主設定 |
| 20 | 0〜1,638 | 83 | 18 | 新規 |

同じindexを`(H, covered_fixed, covered_rolling, width_fixed, width_rolling, IS_fixed, IS_rolling)`全体に適用する。
高VIXだけに圧縮せず、抽出後の日付ソートや重複削除も行わない。区分件数・分母は各回のindicatorから計算し直す。
空区分はNaN、再抽出・0埋めなし。CIは有効値の2.5/97.5%点を`np.quantile(method="linear")`で計算する。
Conformal名目Coverage90%と、CIの信頼水準95%は別である。

変更したのはbootstrap block lengthだけで、Rolling calibration windowは252のまま。
前段階の実行、OLS・score履歴・閾値・予測区間・日次評価の再生成は行っていない。

実行入口は`16_block_length_sensitivity.py`。`draw_moving_blocks`がindex、`summarize_sample`が区分別の6平均・3差を作り、
`build_ci_tables`が新規5/20のCIを計算する。比較表は元標本の点推定が3設定で同じことも確認する。

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\16_block_length_sensitivity.py
```

## High-VIXの主・副次結果

すべてRolling−Fixed。Coverage差は表示用のcoverage pp、Width差・IS差はreturn pp。
Coverage差の比例尺度もCSVへ保存した。主な推論対象はtwo-sidedのCoverage差で、Width・ISは副次である。

| l | 指標 | 点推定 | 95%CI | CI幅 | 0包含 | 有効/無効 |
|---:|---|---:|---|---:|---|---|
'''
    for row in high_differences.itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {row.block_length} | {row.metric} ({row.display_unit}) | {row.display_point_estimate:+.6f} | [{row.display_ci_lower:+.6f}, {row.display_ci_upper:+.6f}] | {row.display_ci_width:.6f} | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += '\n## 高VIXのCoverageと90%からのsigned Gap\n\nCoverageは%、Gapはcoverage pp。Gap CIはCoverage CIから0.90を引いて100倍する。\n\n'
    summary += '| l | 手法 | Coverage | 95%CI (%) | CI幅 (coverage pp) | Gap (pp) | Gap 95%CI (pp) | 90%との関係 |\n|---:|---|---:|---|---:|---:|---|---|\n'
    for row in high_coverage.itertuples():
        summary += f'| {row.block_length} | {row.method} | {row.coverage_point_estimate_pct:.4f}% | [{row.coverage_ci_lower_pct:.4f}, {row.coverage_ci_upper_pct:.4f}] | {row.coverage_ci_width_pp:.4f} | {row.signed_gap_point_estimate_pp:+.4f} | [{row.signed_gap_ci_lower_pp:+.4f}, {row.signed_gap_ci_upper_pp:+.4f}] | {row.ci_relation_to_nominal} |\n'
    summary += '\n## 維持された解釈と変化した解釈\n\n'
    for metric, direction, label in [('coverage_difference', 'positive', '高VIX Coverage差'),
                                     ('mean_width_difference', 'positive', '高VIX Mean Width差'),
                                     ('mean_interval_score_difference', 'negative', '高VIX Mean IS差')]:
        selected = high_differences.loc[high_differences['metric'].eq(metric)]
        holds = selected['valid_replicates'].gt(0).all() and (selected['ci_lower'].gt(0).all() if direction == 'positive' else selected['ci_upper'].lt(0).all())
        summary += f'- {label}：' + ('3設定すべてでCI全体が' + ('正' if direction == 'positive' else '負') + 'の領域にあり、この範囲で方向の解釈は維持された。' if holds else 'CIの方向・0包含が全設定で揃うとは限らない。上表の設定依存性をそのまま報告し、主設定は変更しない。') + '\n'
    below_all = high_coverage['ci_relation_to_nominal'].eq('below_nominal').all()
    summary += '- 高VIXの両手法のCoverage CI：' + ('3設定すべてで90%未満で、相対的改善と名目値からの不足という区別は維持された。' if below_all else '90%との関係に設定差がある。各手法・設定の表の関係を区別する。') + '\n'
    summary += '\nCI幅（主設定10に対する比率も表示）：\n\n| 指標 | l=5 | l=10 | l=20 | 幅5/10 | 幅20/10 |\n|---|---:|---:|---:|---:|---:|\n'
    for metric, _, _, display_unit, _ in METRIC_SPECS:
        values = high_differences.loc[high_differences['metric'].eq(metric)].set_index('block_length')['display_ci_width']
        summary += f'| {metric} ({display_unit}) | {values[5]:.6f} | {values[10]:.6f} | {values[20]:.6f} | {values[5] / values[10]:.4f} | {values[20] / values[10]:.4f} |\n'
    summary += '\n以下の補助結果は全件を示す。都合のよいCIを主結論へ選び直さない。\n\n'
    summary += '| 区分 | 指標 | l=5 95%CI | l=10 95%CI | l=20 95%CI | 0包含の設定差 |\n|---|---|---|---|---|---|\n'
    for group in ['overall', 'non_high_vix']:
        for metric, _, _, display_unit, _ in METRIC_SPECS:
            rows = difference_comparison.loc[difference_comparison['group'].eq(group) & difference_comparison['metric'].eq(metric)].set_index('block_length')
            intervals = [f"[{rows.loc[l, 'display_ci_lower']:+.6f}, {rows.loc[l, 'display_ci_upper']:+.6f}]" for l in [5, 10, 20]]
            flags = ['未定義' if rows.loc[l, 'valid_replicates'] == 0 else ('含む' if rows.loc[l, 'ci_includes_zero'] else '含まない') for l in [5, 10, 20]]
            summary += f"| {GROUP_LABELS[group]} | {metric} ({display_unit}) | {' | '.join(intervals)} | " + ('共通：' + flags[0] if len(set(flags)) == 1 else '設定依存：5/10/20=' + '/'.join(flags)) + ' |\n'
    summary += '\n補助的なCoverage・Gapの全結果（CIの幅、有効数、単位はCSVにも保存）：\n\n| 区分 | 手法 | l | Coverage (%) | Coverage 95%CI (%) | Gap 95%CI (pp) | 90%との関係 |\n|---|---|---:|---:|---|---|---|\n'
    for row in coverage_comparison.loc[coverage_comparison['group'].ne('high_vix')].itertuples():
        summary += f'| {GROUP_LABELS[row.group]} | {row.method} | {row.block_length} | {row.coverage_point_estimate_pct:.4f} | [{row.coverage_ci_lower_pct:.4f}, {row.coverage_ci_upper_pct:.4f}] | [{row.signed_gap_ci_lower_pp:+.4f}, {row.signed_gap_ci_upper_pp:+.4f}] | {row.ci_relation_to_nominal} |\n'
    summary += '\n## 有効数・区分件数の分布と検証\n\n| l | 区分 | 有効/無効 | 最小 | 平均 | 中央値 | 最大 | 件数2.5/97.5%点 |\n|---:|---|---|---:|---:|---:|---:|---|\n'
    for row in count_comparison.itertuples():
        summary += f'| {row.block_length} | {GROUP_LABELS[row.group]} | {row.valid_group_replicates}/{row.zero_group_replicates} | {row.min_n} | {row.mean_n:.4f} | {row.n_median:.2f} | {row.max_n} | [{row.n_q025:.2f}, {row.n_q975:.2f}] |\n'
    summary += '''
新規の各設定で生成数5,000、長さ1,658、開始位置・block内連続性・末尾切り詰め・共通index・区分件数合計・paired差と平均差の一致を確認した。
各回の区分件数で分母を計算し、空区分での再抽出はしていない。空区分と全NaN CIの分岐は検証用コピーで確認した。
新規の各設定で最初の3回だけを再生成し、開始位置・index・件数・統計量が完全一致した。
Fixedの平均Widthの変動は浮動小数点誤差の範囲だった。保存CSVの全列round-trip一致も確認した。
元標本の点推定は全設定で同じであり、主10の既存CI全列を変更せず比較表へ転載した。
元点推定がCI内にあること、CI幅がlとともに必ず増えることを検証条件にはしていない。
入力ファイル・主設定の成果物は実行前後のSHA-256が一致した。未解決の実装・入力上の不整合は見つからなかった。

## 保存先

- `16_block_length_sensitivity.py`：保存済みCSVだけから実行。15や前段階をimportしない。
- `reports/bootstrap_sensitivity/block_5/`、`block_20/`：各15,000行のreplicate統計・区分件数、差/coverage/gapのCI、元点推定、件数要約、最初3回のblock開始位置と再現性、環境・設定・入力ハッシュ・検証・実行ログ。
- `reports/bootstrap_sensitivity/difference_ci_comparison.csv`：全3設定・全区分の差27行。元尺度/表示尺度の点・CI・幅・0包含・位置づけ・出所・ハッシュ。
- `reports/bootstrap_sensitivity/coverage_gap_ci_comparison.csv`：3設定×3区分×2手法=18行。Coverage/Gap・CI幅・90%との関係・有効数・出所。
- `reports/bootstrap_sensitivity/group_count_comparison.csv`：全3設定の件数分布。
- `reports/bootstrap_sensitivity/validation_results.json`、`execution_metadata.json`、`execution.log`：比較の検証、主設定保全、実行記録。
- `figures/bootstrap_sensitivity/`：高VIXの差CI 3パネルと両手法のCoverage CI。各PNG 300dpi・PDF。
- 図の目視確認は実行後に行い、同reportsフォルダの`figure_visual_qa.md`へ記録する。

## 残る限界と次の段階

今回調べたblock lengthの範囲で解釈が維持されることと、あらゆる時系列依存・非定常性への頑健性は異なる。
この分析だけでstationarity・weak dependenceなどのMBBの条件や、長期依存を十分扱えたことを証明できない。
保存済みモデル・区間・日次評価系列に条件づけた不確実性評価で、学習から再推定する全工程の不確実性は評価していない。
CIが0を除外しても厳密な有限標本検定・Conformal Coverage保証を得たとは扱わない。
Coverage CIが90%を含む場合も、目標Coverage達成の証明とは解釈しない。
高VIXのCoverage上昇、Width増加、IS変化を併せて読み、Overallで高VIXの不足を覆い隠さない。
主設定10、VIX閾値30、Rolling calibration window252、分析期間・モデルは維持した。
今回はblock length 5・10・20の比較までで終了した。VIX閾値・Rolling window・モデルの感度分析は実行していない。
'''
    (PROJECT_DIR / 'reports/block_length_sensitivity_summary.md').write_text(summary, encoding='utf-8')
    print('\nHigh-VIX differences:\n' + high_differences.to_string(index=False))
    print('\nHigh-VIX coverage:\n' + high_coverage.to_string(index=False))
    print('\nGroup count distributions:\n' + count_comparison.to_string(index=False))
    print('Block length sensitivity: PASS. New 5/20 only; 10 reused; visual QA to follow.')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 16.')
    run_block_length_sensitivity(ANALYSIS_WORKSPACE)
