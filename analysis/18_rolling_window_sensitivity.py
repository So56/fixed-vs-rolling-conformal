"""Rolling63/126の区間とpaired MBBを検証し、主252と共通Fixedを保持する。
実行入口はrun_analysis.pyのstage 18。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_rolling_window_sensitivity(workspace):
    """保存済み入力を独立workspaceで検証し、この段階の結果だけを保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports/rolling_window_sensitivity'
    FIGURE_DIR = PROJECT_DIR / 'figures/rolling_window_sensitivity'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'calibration_scores': PROJECT_DIR / 'reports/fixed_conformal/calibration_scores_2019.csv',
        'test_scores': PROJECT_DIR / 'reports/rolling_conformal/test_scores.csv',
        'predictions': PROJECT_DIR / 'reports/ols/test_point_predictions.csv',
        'fixed_intervals': PROJECT_DIR / 'reports/fixed_conformal/test_fixed_intervals.csv',
        'main_intervals': PROJECT_DIR / 'reports/rolling_conformal/test_rolling_intervals.csv',
        'main_history': PROJECT_DIR / 'reports/rolling_conformal/history_audit.csv',
        'daily_evaluation': PROJECT_DIR / 'reports/evaluation/daily_test_evaluation.csv',
        'main_comparison': PROJECT_DIR / 'reports/evaluation/comparison_by_regime.csv',
        'main_paired': PROJECT_DIR / 'reports/evaluation/paired_differences_by_regime.csv',
        'main_replicates': PROJECT_DIR / 'reports/bootstrap/replicate_statistics.csv',
        'main_difference_ci': PROJECT_DIR / 'reports/bootstrap/difference_confidence_intervals.csv',
        'main_coverage_ci': PROJECT_DIR / 'reports/bootstrap/coverage_gap_confidence_intervals.csv',
        'main_counts': PROJECT_DIR / 'reports/bootstrap/replicate_group_count_summary.csv',
        'main_metadata': PROJECT_DIR / 'reports/bootstrap/execution_metadata.json',
        'main_starts': PROJECT_DIR / 'reports/bootstrap/first_replicate_block_starts.csv',
        'rolling_metadata': PROJECT_DIR / 'reports/rolling_conformal/execution_metadata.json',
        'fixed_coefficients': PROJECT_DIR / 'models/ols/fixed_coefficients.json',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    WINDOWS, NEW_WINDOWS = [63, 126, 252], [63, 126]
    GROUPS = ['overall', 'high_vix', 'non_high_vix']
    EXPECTED_N = {'overall': 1658, 'high_vix': 149, 'non_high_vix': 1509}
    MEAN_COLUMNS = ['coverage_fixed', 'coverage_rolling', 'mean_width_fixed_return_pp', 'mean_width_rolling_return_pp',
                    'mean_interval_score_fixed_return_pp', 'mean_interval_score_rolling_return_pp']
    DIFFERENCE_COLUMNS = ['coverage_difference_rolling_minus_fixed', 'mean_width_difference_rolling_minus_fixed_return_pp',
                          'mean_interval_score_difference_rolling_minus_fixed_return_pp']
    STAT_COLUMNS = MEAN_COLUMNS + DIFFERENCE_COLUMNS
    METRIC_SPECS = [
        ('coverage_difference', DIFFERENCE_COLUMNS[0], 'proportion', 'coverage percentage points', 100.),
        ('mean_width_difference', DIFFERENCE_COLUMNS[1], 'return percentage points', 'return percentage points', 1.),
        ('mean_interval_score_difference', DIFFERENCE_COLUMNS[2], 'return percentage points', 'return percentage points', 1.),
    ]
    RTOL = ATOL = 1e-12  # 構成式の数値照合だけ。covered判定には使わない。


    def save_frame(path, frame):
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
        dates = [column for column in frame if pd.api.types.is_datetime64_any_dtype(frame[column])]
        dtype = {column: str(frame[column].dtype) for column in frame if str(frame[column].dtype) in ['boolean', 'Int64', 'string']}
        saved = pd.read_csv(path, parse_dates=dates or None, dtype=dtype or None, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)


    def with_provenance(frame, window, path):
        """FixedにRolling windowの値を付けない。差表のwindowは比較するRollingを表す。"""
        result = frame.copy()
        result.insert(0, 'rolling_window', pd.Series([window] * len(frame), index=frame.index, dtype='Int64'))
        result['setting_role'] = 'common_fixed_benchmark' if window is None else ('primary_setting' if window == 252 else 'sensitivity_setting')
        result['result_origin'] = 'reused_existing' if window is None or window == 252 else 'newly_computed'
        result['source_file'] = path.relative_to(PROJECT_DIR).as_posix()
        result['source_sha256'] = file_hash(path)
        return result


    def rolling_description(frame):
        rows, paired_rows = [], []
        for group, mask in zip(GROUPS, [np.ones(len(frame), dtype=bool), frame['high_vix_30'].to_numpy(dtype=bool), ~frame['high_vix_30'].to_numpy(dtype=bool)]):
            sample, n = frame.loc[mask], int(mask.sum())
            assert n == EXPECTED_N[group]
            coverage = int(sample['covered_rolling'].sum()) / n
            gap = 100 * (coverage - .90)
            rows.append({'group': group, 'method': 'rolling', 'n': n, 'covered_n': int(sample['covered_rolling'].sum()),
                         'lower_miss_n': int(sample['lower_miss_rolling'].sum()), 'upper_miss_n': int(sample['upper_miss_rolling'].sum()),
                         'empirical_coverage': coverage, 'coverage_gap_pp': gap, 'absolute_coverage_gap_pp': abs(gap),
                         'mean_width_return_pp': sample['width_rolling_return_pp'].mean(),
                         'median_width_return_pp': sample['width_rolling_return_pp'].median(),
                         'mean_interval_score_return_pp': sample['interval_score_rolling_return_pp'].mean()})
            delta = sample['coverage_difference_rolling_minus_fixed'].mean()
            paired_rows.append({'group': group, 'n': n, 'primary_coverage_comparison': False,
                                'coverage_difference_rolling_minus_fixed': delta, 'coverage_difference_rolling_minus_fixed_pp': 100 * delta,
                                'mean_width_difference_rolling_minus_fixed_return_pp': sample['width_difference_rolling_minus_fixed_return_pp'].mean(),
                                'mean_interval_score_difference_rolling_minus_fixed_return_pp': sample['interval_score_difference_rolling_minus_fixed_return_pp'].mean()})
        return pd.DataFrame(rows), pd.DataFrame(paired_rows)


    # 前段階はimportしない。以下は同じ計算規則の短い関数定義だけを配置したもの。

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

    def interval_metrics(actual, lower, upper, alpha):
        """境界を広げず、両端inclusiveのCoverageとWidth・Interval Scoreを計算する。"""
        actual, lower, upper = [np.asarray(values, dtype=float) for values in [actual, lower, upper]]
        assert actual.shape == lower.shape == upper.shape
        assert np.isfinite(actual).all() and np.isfinite(lower).all() and np.isfinite(upper).all()
        assert (lower <= upper).all()
        covered = (lower <= actual) & (actual <= upper)
        lower_miss = actual < lower
        upper_miss = actual > upper
        width = upper - lower
        lower_penalty = np.where(lower_miss, (2 / alpha) * (lower - actual), 0.0)
        upper_penalty = np.where(upper_miss, (2 / alpha) * (actual - upper), 0.0)
        return pd.DataFrame({
            'covered': covered, 'lower_miss': lower_miss, 'upper_miss': upper_miss,
            'width_return_pp': width, 'lower_penalty_return_pp': lower_penalty,
            'upper_penalty_return_pp': upper_penalty,
            'interval_score_return_pp': width + lower_penalty + upper_penalty,
        })


    def rolling_ci_tables(replicates, original):
        differences, coverage_rows = [], []
        for group in GROUPS:
            sample = replicates.loc[replicates['group'].eq(group)]
            assert len(sample) == B
            for metric, column, unit, display_unit, scale in METRIC_SPECS:
                lower, upper, valid, invalid = percentile_bounds(sample[column])
                if valid:
                    assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
                    if metric == 'coverage_difference':
                        assert -1 <= lower <= upper <= 1
                point = float(original.loc[group, column])
                differences.append({'group': group, 'n_original': EXPECTED_N[group], 'metric': metric,
                    'analysis_role': 'high_vix_window_sensitivity' if group == 'high_vix' else 'auxiliary_sensitivity',
                    'unit': unit, 'point_estimate': point, 'ci_lower': lower, 'ci_upper': upper,
                    'confidence_level': .95, 'percentile_method': 'linear', 'valid_replicates': valid, 'invalid_replicates': invalid,
                    'ci_includes_zero': bool(lower <= 0 <= upper) if valid else pd.NA,
                    'display_unit': display_unit, 'display_point_estimate': scale * point,
                    'display_ci_lower': scale * lower, 'display_ci_upper': scale * upper})
            lower, upper, valid, invalid = percentile_bounds(sample['coverage_rolling'])
            if valid:
                assert 0 <= lower <= upper <= 1
                relation = 'below_nominal' if upper < .90 else ('above_nominal' if lower > .90 else 'includes_nominal')
            else:
                relation = 'undefined_no_valid'
            point = float(original.loc[group, 'coverage_rolling'])
            coverage_rows.append({'group': group, 'method': 'rolling', 'n_original': EXPECTED_N[group],
                'analysis_role': 'diagnostic_high_vix_window_sensitivity' if group == 'high_vix' else 'auxiliary_diagnostic_sensitivity',
                'coverage_unit': 'proportion', 'coverage_point_estimate': point, 'coverage_ci_lower': lower, 'coverage_ci_upper': upper,
                'nominal_interval_coverage': .90, 'signed_gap_unit': 'coverage percentage points',
                'signed_gap_point_estimate_pp': 100 * (point - .90), 'signed_gap_ci_lower_pp': 100 * (lower - .90),
                'signed_gap_ci_upper_pp': 100 * (upper - .90),
                'coverage_ci_contains_nominal': bool(lower <= .90 <= upper) if valid else pd.NA,
                'gap_ci_contains_zero': bool(lower - .90 <= 0 <= upper - .90) if valid else pd.NA,
                'ci_relation_to_nominal': relation, 'confidence_level': .95, 'valid_replicates': valid, 'invalid_replicates': invalid})
        difference, coverage = pd.DataFrame(differences), pd.DataFrame(coverage_rows)
        difference['ci_includes_zero'] = difference['ci_includes_zero'].astype('boolean')
        for column in ['coverage_ci_contains_nominal', 'gap_ci_contains_zero']:
            coverage[column] = coverage[column].astype('boolean')
        assert len(difference) == 9 and len(coverage) == 3
        assert (difference['valid_replicates'] + difference['invalid_replicates']).eq(B).all()
        assert (coverage['valid_replicates'] + coverage['invalid_replicates']).eq(B).all()
        return difference, coverage


    # 1. 入力・共通標本の確認。前段階を実行せず、保存済みscoreと区間を読み込む。
    started_at = datetime.now(timezone.utc)
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    protected_paths = []
    for directory in ['reports/bootstrap', 'figures/bootstrap', 'reports/evaluation', 'figures/evaluation',
                      'reports/rolling_conformal', 'figures/rolling_conformal', 'reports/fixed_conformal', 'figures/fixed_conformal']:
        protected_paths.extend(path for path in (PROJECT_DIR / directory).glob('*') if path.is_file())
    protected_hashes = {path.relative_to(PROJECT_DIR).as_posix(): file_hash(path) for path in protected_paths}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    main_metadata = json.loads(INPUT_PATHS['main_metadata'].read_text(encoding='utf-8'))
    rolling_metadata = json.loads(INPUT_PATHS['rolling_metadata'].read_text(encoding='utf-8'))
    model = json.loads(INPUT_PATHS['fixed_coefficients'].read_text(encoding='utf-8'))
    assert design['sensitivity_rolling_windows'] == WINDOWS and design['rolling_window'] == 252
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    assert design['nominal_coverage'] == .90 and design['interval_endpoints'] == 'inclusive'
    assert design['block_length'] == 10 and design['bootstrap_repetitions'] == 5000 and design['seed'] == 202701
    assert design['primary_hypothesis'] == 'two-sided high-VIX coverage difference'
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    settings = main_metadata['settings']
    assert settings['N'] == 1658 and settings['B_generated'] == 5000 and settings['block_length_observations'] == 10
    assert settings['seed'] == 202701 and settings['rng'] == 'numpy.random.default_rng' and settings['bit_generator'] == 'PCG64'
    assert settings['percentile_probabilities'] == [.025, .975] and settings['confidence_level'] == .95
    assert settings['percentile_method'] == 'numpy.quantile(method=linear)'
    assert settings['block_scheme'] == 'ordinary non-circular overlapping MBB; starts sampled with replacement; concatenate then retain first N'
    for key, original_key in [('daily_evaluation', 'daily_evaluation'), ('main_comparison', 'comparison'), ('main_paired', 'paired_differences'),
                               ('research_design_lock', 'research_design_lock'), ('research_design_source', 'research_design_source')]:
        assert input_hashes[key] == main_metadata['input_files'][original_key]['sha256']
    for key in ['main_replicates', 'main_difference_ci', 'main_coverage_ci', 'main_counts']:
        assert input_hashes[key] == main_metadata['output_csv_sha256'][INPUT_PATHS[key].name]
    assert input_hashes['main_starts'] == main_metadata['output_csv_sha256']['first_replicate_block_starts.csv']
    for key, original_key in [('calibration_scores', 'calibration_scores'), ('predictions', 'test_point_predictions'),
                               ('fixed_intervals', 'fixed_intervals'), ('fixed_coefficients', 'fixed_coefficients')]:
        assert input_hashes[key] == rolling_metadata['input_files'][original_key]['sha256']
    for key in ['main_intervals', 'test_scores', 'main_history']:
        assert input_hashes[key] == rolling_metadata['output_csv_sha256'][INPUT_PATHS[key].name]
    B, N, BLOCK_LENGTH, SEED, ALPHA = 5000, 1658, 10, 202701, .10
    boolean_columns = ['high_vix_30'] + [f'{kind}_{method}' for method in ['fixed', 'rolling'] for kind in ['covered', 'lower_miss', 'upper_miss']]
    daily = pd.read_csv(INPUT_PATHS['daily_evaluation'], parse_dates=['date', 'forecast_origin_date'], float_precision='round_trip',
                        dtype={**{column: 'boolean' for column in boolean_columns}, 'split': 'string'})
    assert len(daily) == N and daily['date'].is_unique and daily['date'].is_monotonic_increasing
    assert daily['date'].iloc[0] == pd.Timestamp('2020-01-02') and daily['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert not daily.isna().any().any() and np.isfinite(daily.select_dtypes(include='number')).all().all()
    assert daily['split'].eq('test').all() and daily['forecast_origin_date'].lt(daily['date']).all()
    assert daily['high_vix_30'].equals(daily['vix_lag1'].ge(30).astype('boolean'))
    assert daily['high_vix_30'].sum() == 149 and (~daily['high_vix_30']).sum() == 1509
    frames = {}
    for name in ['predictions', 'test_scores', 'fixed_intervals', 'main_intervals']:
        frame = pd.read_csv(INPUT_PATHS[name], parse_dates=['date', 'forecast_origin_date'], dtype={'split': 'string'}, float_precision='round_trip')
        assert len(frame) == N and frame['date'].is_unique and frame['date'].is_monotonic_increasing
        assert not frame.isna().any().any() and np.isfinite(frame.select_dtypes(include='number')).all().all()
        common_columns = [column for column in frame if column in daily]
        joined = daily[common_columns].merge(frame, on='date', how='outer', validate='one_to_one', indicator=True, suffixes=('_reference', '_saved'))
        assert len(joined) == N and joined['_merge'].eq('both').all()
        for column in common_columns:
            if column != 'date':
                left, right = joined[f'{column}_reference'], joined[f'{column}_saved']
                if pd.api.types.is_numeric_dtype(left):
                    np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL)
                else:
                    assert left.equals(right)
        pd.testing.assert_series_equal(frame['date'], daily['date'], check_names=False)
        frames[name] = frame
    predictions, test_scores, fixed, main_intervals = [frames[name] for name in ['predictions', 'test_scores', 'fixed_intervals', 'main_intervals']]
    beta0, beta1 = model['coefficients']['const'], model['coefficients']['return_lag1_pct']
    np.testing.assert_allclose(predictions['predicted_return_pct'], beta0 + beta1 * predictions['return_lag1_pct'], rtol=RTOL, atol=ATOL)
    assert model['training_n'] == 502 and model['predictors'] == ['return_lag1_pct']
    calibration = pd.read_csv(INPUT_PATHS['calibration_scores'], parse_dates=['date', 'forecast_origin_date'], dtype={'split': 'string'}, float_precision='round_trip')
    assert len(calibration) == 252 and calibration['date'].is_unique and calibration['date'].is_monotonic_increasing
    assert calibration['date'].dt.year.eq(2019).all() and calibration['split'].eq('calibration').all()
    assert calibration['forecast_origin_date'].lt(calibration['date']).all() and not calibration.isna().any().any()
    assert np.isfinite(calibration.select_dtypes(include='number')).all().all() and calibration['score_pct'].ge(0).all()
    # 既存scoreの式の整合性を照合するだけで、新しいscoreファイルを生成しない。
    for frame, error_column, score_column in [(calibration, 'calibration_error_pct', 'score_pct'), (test_scores, 'test_error_pct', 'test_score_pct')]:
        np.testing.assert_allclose(frame[error_column], frame['return_pct'] - frame['predicted_return_pct'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(frame[score_column], frame[error_column].abs(), rtol=RTOL, atol=ATOL)
        assert frame[score_column].ge(0).all()
    main_history = pd.read_csv(INPUT_PATHS['main_history'], parse_dates=['date', 'forecast_origin_date', 'oldest_score_date', 'latest_score_date'], float_precision='round_trip')
    pd.testing.assert_series_equal(main_history['date'], daily['date'], check_names=False)
    assert main_history['score_count'].eq(252).all() and main_history['k'].eq(228).all()
    assert main_intervals['q_rolling_pct'].iloc[0] == fixed['q_fixed_pct'].iloc[0]  # 既存252だけの初日一致。
    assert fixed['q_fixed_pct'].eq(fixed['q_fixed_pct'].iloc[0]).all()
    main_comparison = pd.read_csv(INPUT_PATHS['main_comparison'], float_precision='round_trip')
    main_paired = pd.read_csv(INPUT_PATHS['main_paired'], float_precision='round_trip')
    main_replicates = pd.read_csv(INPUT_PATHS['main_replicates'], float_precision='round_trip')
    main_difference = pd.read_csv(INPUT_PATHS['main_difference_ci'], dtype={'ci_includes_zero': 'boolean'}, float_precision='round_trip')
    main_coverage = pd.read_csv(INPUT_PATHS['main_coverage_ci'], dtype={'coverage_ci_contains_nominal': 'boolean', 'gap_ci_contains_zero': 'boolean'}, float_precision='round_trip')
    main_counts = pd.read_csv(INPUT_PATHS['main_counts'], float_precision='round_trip')
    main_starts = pd.read_csv(INPUT_PATHS['main_starts'], parse_dates=['start_date', 'end_date_before_truncation'])
    assert len(main_replicates) == 3 * B and not main_replicates.duplicated(['replicate', 'group']).any()
    main_counts_by_rep = main_replicates.pivot(index='replicate', columns='group', values='n').sort_index()
    np.testing.assert_array_equal(main_counts_by_rep.index, np.arange(1, B + 1))
    fixed_description = main_comparison.loc[main_comparison['method'].eq('fixed')].reset_index(drop=True)
    test_inputs = predictions[['date', 'forecast_origin_date', 'split', 'predicted_return_pct']].copy()
    score_by_date = test_scores.set_index('date')['test_score_pct']
    calibration_records = [(row.date, float(row.score_pct), 'calibration') for row in calibration.itertuples()]
    catalog = pd.concat([calibration[['date', 'score_pct']].assign(score_source='calibration'),
                         test_scores[['date', 'test_score_pct']].rename(columns={'test_score_pct': 'score_pct'}).assign(score_source='test')], ignore_index=True)
    assert catalog['date'].is_unique and catalog['date'].is_monotonic_increasing
    print('Input/date/score checks PASS. Model and saved scores retained; common 1,658 dates / 149 high-VIX.', flush=True)


    # 2. 新しい区間を作成・検証して保存。その後に日次評価へ進む。
    intervals_by_window, evaluations, descriptions, paired_tables = {252: main_intervals}, {252: daily}, {252: main_comparison.loc[main_comparison['method'].eq('rolling')].reset_index(drop=True)}, {252: main_paired}
    initial_rows, stage_validations = [], {}
    first_main = main_history.iloc[0]
    initial_rows.append({'rolling_window': 252, 'k': 228, 'zero_based_index': 227, 'initial_history_start': first_main['oldest_score_date'],
                         'initial_history_end': first_main['latest_score_date'], 'first_q_rolling_pct': first_main['q_rolling_pct'],
                         'result_origin': 'reused_existing', 'source_file': INPUT_PATHS['main_history'].relative_to(PROJECT_DIR).as_posix(), 'source_sha256': input_hashes['main_history']})
    for window in NEW_WINDOWS:
        run_dir = REPORT_DIR / f'window_{window}'
        run_dir.mkdir(parents=True, exist_ok=True)
        k = math.ceil((window + 1) * .90)
        assert k == {63: 58, 126: 115}[window]
        observed_history = calibration_records.copy()
        history = observed_history[-window:].copy()
        interval_rows, audit_rows = [], []
        for i, row in enumerate(test_inputs.itertuples(index=False)):
            # A: 履歴は直近w個。全Calibration252+過去Testを基準にするため、初期tail(w)を維持できる。
            dates = pd.DatetimeIndex([record[0] for record in history])
            values = np.asarray([record[1] for record in history], dtype=float)
            sources = [record[2] for record in history]
            assert len(history) == window and history == observed_history[-window:]
            assert np.isfinite(values).all() and (values >= 0).all() and dates.is_unique and dates.is_monotonic_increasing
            assert (dates < row.date).all() and dates[-1] == row.forecast_origin_date
            assert sources.count('calibration') == max(window - i, 0) and sources.count('test') == min(i, window)
            assert set(sources).issubset({'calibration', 'test'}) and sources.count('training') == 0
            # B: コピーをソートしたk番目。時系列履歴の順序は変えず、補間しない。
            q = float(np.sort(values, kind='stable')[k - 1])
            assert math.isfinite(q) and q >= 0
            # C: 区間と監査を確定。当日の実測値・scoreをこの処理に渡さない。
            interval_rows.append({'date': row.date, 'forecast_origin_date': row.forecast_origin_date, 'split': row.split,
                                  'predicted_return_pct': row.predicted_return_pct, 'q_rolling_pct': q,
                                  'lower_rolling_pct': row.predicted_return_pct - q, 'upper_rolling_pct': row.predicted_return_pct + q})
            audit_rows.append({'date': row.date, 'forecast_origin_date': row.forecast_origin_date, 'rolling_window': window,
                               'score_count': len(history), 'k': k, 'zero_based_index': k - 1,
                               'oldest_score_date': dates[0], 'latest_score_date': dates[-1],
                               'calibration_score_count': sources.count('calibration'), 'past_test_score_count': sources.count('test'),
                               'training_score_count': 0, 'q_rolling_pct': q, 'history_start_sequence_index': 252 + i - window,
                               'history_end_sequence_index': 252 + i - 1})
            # D/E: ここで初めて当日の保存scoreを取得し、翌日用に追加・最古除去する。
            observed_score = float(score_by_date.loc[row.date])
            assert math.isfinite(observed_score) and observed_score >= 0
            observed_history.append((row.date, observed_score, 'test'))
            history.append((row.date, observed_score, 'test'))
            removed = history.pop(0)
            assert removed[0] == dates[0] and history == observed_history[-window:]
            audit_rows[-1]['removed_after_observation_date'] = removed[0]
            audit_rows[-1]['appended_after_observation_date'] = row.date
        intervals, audit = pd.DataFrame(interval_rows), pd.DataFrame(audit_rows)
        intervals['split'] = intervals['split'].astype('string')
        assert len(intervals) == N and audit['score_count'].eq(window).all()
        pd.testing.assert_frame_equal(intervals[list(test_inputs.columns)], test_inputs, check_exact=True)
        assert audit['latest_score_date'].lt(audit['date']).all() and audit['training_score_count'].eq(0).all()
        assert np.isfinite(intervals.select_dtypes(include='number')).all().all()
        assert intervals['lower_rolling_pct'].le(intervals['upper_rolling_pct']).all()
        np.testing.assert_allclose((intervals['lower_rolling_pct'] + intervals['upper_rolling_pct']) / 2, test_inputs['predicted_return_pct'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(intervals['upper_rolling_pct'] - intervals['lower_rolling_pct'], 2 * intervals['q_rolling_pct'], rtol=RTOL, atol=ATOL)
        save_frame(run_dir / 'test_rolling_intervals.csv', intervals)
        save_frame(run_dir / 'history_audit.csv', audit)
        # 別経路で保存scoreをdate<tに絞りtail(w)し、pandasの安定ソートから順位を選ぶ。
        reconstruction_rows = []
        for i in [0, 1, 2, N // 2, N - 1]:
            past = catalog.loc[catalog['date'].lt(intervals['date'].iloc[i])].tail(window)
            assert len(past) == window
            reconstructed_q = float(past['score_pct'].sort_values(kind='stable').iloc[k - 1])
            assert reconstructed_q == intervals['q_rolling_pct'].iloc[i]
            assert past['date'].iloc[0] == audit['oldest_score_date'].iloc[i] and past['date'].iloc[-1] == audit['latest_score_date'].iloc[i]
            assert int(past['score_source'].eq('calibration').sum()) == audit['calibration_score_count'].iloc[i]
            assert int(past['score_source'].eq('test').sum()) == audit['past_test_score_count'].iloc[i]
            reconstruction_rows.append({'date': intervals['date'].iloc[i], 'score_count': len(past), 'k': k,
                                         'oldest_score_date': past['date'].iloc[0], 'latest_score_date': past['date'].iloc[-1],
                                         'calibration_score_count': int(past['score_source'].eq('calibration').sum()),
                                         'past_test_score_count': int(past['score_source'].eq('test').sum()),
                                         'saved_q_rolling_pct': intervals['q_rolling_pct'].iloc[i], 'reconstructed_q_rolling_pct': reconstructed_q})
        save_frame(run_dir / 'validation_reconstructed_histories.csv', pd.DataFrame(reconstruction_rows))
        interval_validation = {'status': 'PASS', 'window': window, 'k': k, 'interval_rows': N,
            'all_days_exactly_last_w_observed_scores': True, 'all_history_dates_before_target': True,
            'calibration_count_max_w_minus_i_and_test_count_min_i_w': True, 'training_score_count': 0,
            'current_saved_score_read_only_after_interval_recorded': True, 'same_dates_origin_split_point_predictions': True,
            'finite_ordered_bounds_center_and_2q_width_valid': True, 'representative_history_reconstructions': len(reconstruction_rows),
            'first_interval_equal_fixed_required': False, 'saved_csv_exact_round_trip': True}
        (run_dir / 'interval_validation.json').write_text(json.dumps(interval_validation, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'execution.log').write_text(f'window={window}, k={k}: interval/history stage PASS, N={N}; no score regeneration.\n', encoding='utf-8')
        first = audit.iloc[0]
        initial_rows.append({'rolling_window': window, 'k': k, 'zero_based_index': k - 1, 'initial_history_start': first['oldest_score_date'],
                             'initial_history_end': first['latest_score_date'], 'first_q_rolling_pct': first['q_rolling_pct'],
                             'result_origin': 'newly_computed', 'source_file': (run_dir / 'history_audit.csv').relative_to(PROJECT_DIR).as_posix(),
                             'source_sha256': file_hash(run_dir / 'history_audit.csv')})
        print(f'window={window}: interval/history validation PASS, then evaluation.', flush=True)

        # 3. 検証済み区間をStep E-1と同じinclusive/Width/ISで評価。Fixedの日次値は再利用。
        keep = ['date', 'forecast_origin_date', 'split', 'return_pct', 'predicted_return_pct', 'vix_lag1', 'high_vix_30', 'market_regime',
                'q_fixed_pct', 'lower_fixed_pct', 'upper_fixed_pct', 'covered_fixed', 'lower_miss_fixed', 'upper_miss_fixed',
                'width_fixed_return_pp', 'lower_penalty_fixed_return_pp', 'upper_penalty_fixed_return_pp', 'interval_score_fixed_return_pp']
        evaluation = daily[keep].copy()
        for column in ['q_rolling_pct', 'lower_rolling_pct', 'upper_rolling_pct']:
            evaluation[column] = intervals[column].to_numpy()
        calculated = interval_metrics(evaluation['return_pct'], evaluation['lower_rolling_pct'], evaluation['upper_rolling_pct'], ALPHA)
        for column in calculated:
            name = column.replace('_return_pp', '_rolling_return_pp') if column.endswith('_return_pp') else f'{column}_rolling'
            evaluation[name] = calculated[column]
        evaluation['coverage_difference_rolling_minus_fixed'] = evaluation['covered_rolling'].astype('int64') - evaluation['covered_fixed'].astype('int64')
        evaluation['width_difference_rolling_minus_fixed_return_pp'] = evaluation['width_rolling_return_pp'] - evaluation['width_fixed_return_pp']
        evaluation['interval_score_difference_rolling_minus_fixed_return_pp'] = evaluation['interval_score_rolling_return_pp'] - evaluation['interval_score_fixed_return_pp']
        assert len(evaluation) == N and not evaluation.isna().any().any()
        assert (evaluation['covered_rolling'].astype(int) + evaluation['lower_miss_rolling'].astype(int) + evaluation['upper_miss_rolling'].astype(int)).eq(1).all()
        assert evaluation['width_rolling_return_pp'].ge(0).all() and evaluation['interval_score_rolling_return_pp'].ge(evaluation['width_rolling_return_pp']).all()
        np.testing.assert_allclose(evaluation.loc[evaluation['covered_rolling'], 'interval_score_rolling_return_pp'],
                                   evaluation.loc[evaluation['covered_rolling'], 'width_rolling_return_pp'], rtol=0, atol=0)
        pd.testing.assert_frame_equal(evaluation[keep], daily[keep], check_exact=True)
        description, paired = rolling_description(evaluation)
        for group in GROUPS:
            rolling = description.set_index('group').loc[group]
            benchmark = fixed_description.set_index('group').loc[group]
            pair = paired.set_index('group').loc[group]
            np.testing.assert_allclose(pair[DIFFERENCE_COLUMNS].to_numpy(dtype=float),
                rolling[['empirical_coverage', 'mean_width_return_pp', 'mean_interval_score_return_pp']].to_numpy(dtype=float)
                - benchmark[['empirical_coverage', 'mean_width_return_pp', 'mean_interval_score_return_pp']].to_numpy(dtype=float), rtol=RTOL, atol=ATOL)
        indexed = description.set_index('group')
        for column in ['n', 'covered_n', 'lower_miss_n', 'upper_miss_n']:
            assert indexed.loc['overall', column] == indexed.loc[GROUPS[1:], column].sum()
        for column in ['empirical_coverage', 'coverage_gap_pp', 'mean_width_return_pp', 'mean_interval_score_return_pp']:
            weighted = (indexed.loc[GROUPS[1:], column] * indexed.loc[GROUPS[1:], 'n']).sum() / N
            np.testing.assert_allclose(weighted, indexed.loc['overall', column], rtol=RTOL, atol=ATOL)
        for filename, frame in [('daily_test_evaluation.csv', evaluation), ('rolling_performance.csv', description), ('paired_differences_by_regime.csv', paired)]:
            save_frame(run_dir / filename, frame)
        evaluation_validation = {'status': 'PASS', 'daily_rows': N, 'regime_counts': EXPECTED_N,
            'inclusive_without_rounding_or_tolerance': True, 'interval_score_alpha': ALPHA, 'penalty_coefficient': 20,
            'fixed_daily_metrics_reused_exactly': True, 'covered_and_misses_partition_each_day': True,
            'paired_difference_equal_method_mean_difference': True, 'weighted_group_means_equal_overall': True,
            'median_and_absolute_gap_weighted_equality_required': False, 'csv_exact_round_trip': True}
        (run_dir / 'evaluation_validation.json').write_text(json.dumps(evaluation_validation, ensure_ascii=False, indent=2), encoding='utf-8')
        with (run_dir / 'execution.log').open('a', encoding='utf-8') as output:
            output.write('Evaluation stage PASS; same 1,658 dates, High-VIX149, Non-high1,509.\n' + description.to_string(index=False) + '\n')
        intervals_by_window[window], evaluations[window], descriptions[window], paired_tables[window] = intervals, evaluation, description, paired
        stage_validations[window] = {'interval': interval_validation, 'evaluation': evaluation_validation}
        print(f'window={window}: evaluation validation PASS; ready for common-index bootstrap.', flush=True)
    initial_history = pd.DataFrame(initial_rows).sort_values('rolling_window').reset_index(drop=True)
    save_frame(REPORT_DIR / 'initial_history_comparison.csv', initial_history)


    # 4. H・Fixed・Rolling63・Rolling126を1つのvectorとして同じindexで再抽出する。
    joint_values = np.column_stack([
        daily['high_vix_30'].to_numpy(dtype=float), daily['covered_fixed'].to_numpy(dtype=float),
        daily['width_fixed_return_pp'], daily['interval_score_fixed_return_pp'],
        evaluations[63]['covered_rolling'].to_numpy(dtype=float), evaluations[63]['width_rolling_return_pp'], evaluations[63]['interval_score_rolling_return_pp'],
        evaluations[126]['covered_rolling'].to_numpy(dtype=float), evaluations[126]['width_rolling_return_pp'], evaluations[126]['interval_score_rolling_return_pp'],
    ])
    pair_columns = {63: [0, 1, 4, 2, 5, 3, 6], 126: [0, 1, 7, 2, 8, 3, 9]}
    assert joint_values.shape == (N, 10) and np.isfinite(joint_values).all()
    original_points = {window: pd.DataFrame(summarize_sample(joint_values[:, pair_columns[window]])).set_index('group') for window in NEW_WINDOWS}
    for window in NEW_WINDOWS:
        for group in GROUPS:
            rolling_row = descriptions[window].set_index('group').loc[group]
            fixed_row = fixed_description.set_index('group').loc[group]
            pair = paired_tables[window].set_index('group').loc[group]
            np.testing.assert_allclose(original_points[window].loc[group, MEAN_COLUMNS].to_numpy(dtype=float),
                [fixed_row['empirical_coverage'], rolling_row['empirical_coverage'], fixed_row['mean_width_return_pp'],
                 rolling_row['mean_width_return_pp'], fixed_row['mean_interval_score_return_pp'], rolling_row['mean_interval_score_return_pp']], rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(original_points[window].loc[group, DIFFERENCE_COLUMNS].to_numpy(dtype=float), pair[DIFFERENCE_COLUMNS].to_numpy(dtype=float), rtol=RTOL, atol=ATOL)
    for high_value, empty_group in [(0., 'high_vix'), (1., 'non_high_vix')]:
        artificial = joint_values[:4, pair_columns[63]].copy()
        artificial[:, 0] = high_value
        checked = pd.DataFrame(summarize_sample(artificial)).set_index('group')
        assert checked.loc[empty_group, 'n'] == 0 and checked.loc[empty_group, STAT_COLUMNS].isna().all()
    empty_bounds = percentile_bounds([np.nan, np.nan])
    assert np.isnan(empty_bounds[:2]).all() and empty_bounds[2:] == (0, 2)
    rng = np.random.default_rng(SEED)
    assert type(rng.bit_generator).__name__ == 'PCG64'
    replicate_rows, prefix_starts, prefix_indices, first_block_rows = {window: [] for window in NEW_WINDOWS}, [], [], []
    for replicate in range(1, B + 1):
        block_starts, blocks, sample_indices = draw_moving_blocks(rng, N, BLOCK_LENGTH)
        sample = joint_values[sample_indices]  # H・全手法・全指標に共通。日付再ソートや重複削除はしない。
        assert sample.shape == (N, 10)
        for window in NEW_WINDOWS:
            statistics = summarize_sample(sample[:, pair_columns[window]])
            assert statistics[0]['n'] == N and statistics[1]['n'] + statistics[2]['n'] == N
            for row in statistics:
                replicate_rows[window].append({'replicate': replicate, **row})
        if replicate <= 3:
            prefix_starts.append(block_starts.copy())
            prefix_indices.append(sample_indices.copy())
            np.testing.assert_array_equal(block_starts, main_starts.loc[main_starts['replicate'].eq(replicate), 'start_index'])
            for block_number, start in enumerate(block_starts):
                retained = min(BLOCK_LENGTH, N - block_number * BLOCK_LENGTH)
                first_block_rows.append({'replicate': replicate, 'block_number': block_number + 1, 'start_index': int(start),
                    'end_index_before_truncation': int(start + BLOCK_LENGTH - 1), 'retained_length': retained,
                    'retained_end_index': int(start + retained - 1), 'start_date': daily['date'].iloc[start],
                    'end_date_before_truncation': daily['date'].iloc[start + BLOCK_LENGTH - 1]})
    first_blocks = pd.DataFrame(first_block_rows)
    pd.testing.assert_frame_equal(first_blocks, main_starts, check_exact=True)
    save_frame(REPORT_DIR / 'first_replicate_block_starts.csv', first_blocks)
    replicates_by_window = {window: pd.DataFrame(rows) for window, rows in replicate_rows.items()}
    prefix_rows = {window: [] for window in NEW_WINDOWS}
    check_rng = np.random.default_rng(SEED)
    for replicate in range(1, 4):
        block_starts, blocks, sample_indices = draw_moving_blocks(check_rng, N, BLOCK_LENGTH)
        np.testing.assert_array_equal(block_starts, prefix_starts[replicate - 1])
        np.testing.assert_array_equal(sample_indices, prefix_indices[replicate - 1])
        for window in NEW_WINDOWS:
            checked = pd.DataFrame(summarize_sample(joint_values[sample_indices][:, pair_columns[window]])).set_index('group')
            saved = replicates_by_window[window].loc[lambda frame: frame['replicate'].eq(replicate)].set_index('group')
            np.testing.assert_allclose(checked, saved[['n'] + STAT_COLUMNS], rtol=0, atol=0, equal_nan=True)
            prefix_rows[window].append({'replicate': replicate, 'sample_length': N, 'high_vix_n': int(checked.loc['high_vix', 'n']),
                'non_high_vix_n': int(checked.loc['non_high_vix', 'n']), 'block_starts_match_main_252': True,
                'regenerated_starts_indices_statistics_exact_match': True})
    difference_tables, coverage_tables = {252: main_difference}, {252: main_coverage.loc[main_coverage['method'].eq('rolling')].reset_index(drop=True)}
    main_indexed = main_replicates.set_index(['replicate', 'group']).sort_index()
    for window in NEW_WINDOWS:
        run_dir = REPORT_DIR / f'window_{window}'
        replicates = replicates_by_window[window]
        assert len(replicates) == 3 * B and replicates['replicate'].nunique() == B
        assert replicates.groupby('replicate').size().eq(3).all()
        new_indexed = replicates.set_index(['replicate', 'group']).sort_index()
        pd.testing.assert_series_equal(new_indexed['n'], main_indexed['n'], check_exact=True)
        np.testing.assert_allclose(new_indexed[MEAN_COLUMNS[::2]], main_indexed[MEAN_COLUMNS[::2]], rtol=RTOL, atol=ATOL, equal_nan=True)
        valid = replicates['n'].gt(0)
        assert np.isfinite(replicates.loc[valid, STAT_COLUMNS]).all().all() and replicates.loc[~valid, STAT_COLUMNS].isna().all().all()
        for method in ['fixed', 'rolling']:
            assert replicates.loc[valid, f'coverage_{method}'].between(0, 1).all()
            assert replicates.loc[valid, f'mean_width_{method}_return_pp'].ge(0).all()
            assert replicates.loc[valid, f'mean_interval_score_{method}_return_pp'].ge(replicates.loc[valid, f'mean_width_{method}_return_pp']).all()
        for difference, fixed_column, rolling_column in zip(DIFFERENCE_COLUMNS, MEAN_COLUMNS[::2], MEAN_COLUMNS[1::2]):
            np.testing.assert_allclose(replicates[difference], replicates[rolling_column] - replicates[fixed_column], rtol=RTOL, atol=ATOL, equal_nan=True)
        difference, coverage = rolling_ci_tables(replicates, original_points[window])
        counts = summarize_counts(replicates, B)
        pd.testing.assert_frame_equal(counts, main_counts, check_exact=True)  # 全回の区分件数は主252と同じ。
        outputs = {'replicate_statistics.csv': replicates, 'original_point_statistics.csv': original_points[window].reset_index(),
                   'difference_confidence_intervals.csv': difference, 'rolling_coverage_gap_confidence_intervals.csv': coverage,
                   'replicate_group_count_summary.csv': counts, 'prefix_reproducibility_checks.csv': pd.DataFrame(prefix_rows[window])}
        for filename, frame in outputs.items():
            save_frame(run_dir / filename, frame)
        bootstrap_validation = {'status': 'PASS', 'B': B, 'sample_length': N, 'block_length': BLOCK_LENGTH,
            'blocks_per_replicate': 166, 'last_block_retained_rows': 8, 'non_circular_contiguous_blocks_with_replacement': True,
            'same_index_for_H_fixed_rolling63_rolling126': True, 'all_replicate_group_counts_equal_saved_primary': True,
            'all_replicate_fixed_means_equal_saved_primary': True, 'paired_differences_equal_method_mean_differences': True,
            'first_3_block_records_exactly_equal_main': True, 'first_3_only_regenerated_exactly': True,
            'empty_groups_nan_no_redraw': True, 'empty_branch_checked_with_validation_copies': True,
            'finite_valid_statistics_and_ci_ranges': True, 'saved_csv_all_columns_exact_round_trip': True,
            'main_252_statistics_and_ci_recomputed': False, 'fixed_ci_recomputed': False}
        stage_validations[window]['bootstrap'] = bootstrap_validation
        (run_dir / 'bootstrap_validation.json').write_text(json.dumps(bootstrap_validation, ensure_ascii=False, indent=2), encoding='utf-8')
        metadata = {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
            'executed_at_jst': datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat(),
            'code_sha256': file_hash(Path(__file__).resolve()), 'working_directory': str(Path.cwd()),
            'environment': {'python_executable': sys.executable, 'python_version': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
            'settings': {'rolling_window': window, 'k': math.ceil((window + 1) * .90), 'nominal_coverage': .90, 'alpha': ALPHA,
                         'primary_window_unchanged': 252, 'high_vix_threshold_unchanged': 30, 'bootstrap_block_length': 10,
                         'B': B, 'N': N, 'seed': SEED, 'rng': 'numpy.random.default_rng', 'bit_generator': 'PCG64',
                         'block_scheme': settings['block_scheme'], 'percentile_probabilities': [.025, .975], 'percentile_method': 'linear',
                         'confidence_level': .95, 'empty_group_policy': settings['empty_group_policy'],
                         'shared_bootstrap_indices_between_new_windows': True},
            'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]} for name, path in INPUT_PATHS.items()},
            'output_csv_sha256': {path.name: file_hash(path) for path in run_dir.glob('*.csv')},
            'scope': 'new window only; scores/model/Fixed shared; main252 outputs reused'}
        (run_dir / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        with (run_dir / 'execution.log').open('a', encoding='utf-8') as output:
            output.write('Bootstrap stage PASS; l10/B5000/seed202701/PCG64/95% linear percentile CI.\n'
                         + difference.to_string(index=False) + '\n' + coverage.to_string(index=False) + '\n')
        difference_tables[window], coverage_tables[window] = difference, coverage
        print(f'window={window}: bootstrap validation PASS; {len(replicates)} group rows saved.', flush=True)


    # 5. 比較表。Fixedは単一benchmark、252は既存表の全精度を保持して転載する。
    performance_pieces = [with_provenance(fixed_description, None, INPUT_PATHS['main_comparison'])]
    coverage_pieces = [with_provenance(main_coverage.loc[main_coverage['method'].eq('fixed')].reset_index(drop=True), None, INPUT_PATHS['main_coverage_ci'])]
    paired_pieces, difference_pieces = [], []
    for window in WINDOWS:
        directory = REPORT_DIR / f'window_{window}'
        performance_pieces.append(with_provenance(descriptions[window], window, INPUT_PATHS['main_comparison'] if window == 252 else directory / 'rolling_performance.csv'))
        coverage_pieces.append(with_provenance(coverage_tables[window], window, INPUT_PATHS['main_coverage_ci'] if window == 252 else directory / 'rolling_coverage_gap_confidence_intervals.csv'))
        paired_pieces.append(with_provenance(paired_tables[window], window, INPUT_PATHS['main_paired'] if window == 252 else directory / 'paired_differences_by_regime.csv'))
        difference_pieces.append(with_provenance(difference_tables[window], window, INPUT_PATHS['main_difference_ci'] if window == 252 else directory / 'difference_confidence_intervals.csv'))
    performance = pd.concat(performance_pieces, ignore_index=True)
    coverage_comparison = pd.concat(coverage_pieces, ignore_index=True)
    paired_comparison = pd.concat(paired_pieces, ignore_index=True)
    difference_comparison = pd.concat(difference_pieces, ignore_index=True)
    assert len(performance) == 12 and len(coverage_comparison) == 12 and len(paired_comparison) == 9 and len(difference_comparison) == 27
    assert performance.loc[performance['method'].eq('fixed'), 'rolling_window'].isna().all()
    assert coverage_comparison.loc[coverage_comparison['method'].eq('fixed'), 'rolling_window'].isna().all()
    for table, original in [(performance, main_comparison), (coverage_comparison, main_coverage)]:
        for method in ['fixed', 'rolling']:
            mask = table['method'].eq(method) & (table['rolling_window'].isna() if method == 'fixed' else table['rolling_window'].eq(252))
            pd.testing.assert_frame_equal(table.loc[mask, original.columns].reset_index(drop=True), original.loc[original['method'].eq(method)].reset_index(drop=True), check_exact=True)
    for table, original in [(paired_comparison, main_paired), (difference_comparison, main_difference)]:
        pd.testing.assert_frame_equal(table.loc[table['rolling_window'].eq(252), original.columns].reset_index(drop=True), original, check_exact=True)
    difference_comparison['ci_width'] = difference_comparison['ci_upper'] - difference_comparison['ci_lower']
    difference_comparison['display_ci_width'] = difference_comparison['display_ci_upper'] - difference_comparison['display_ci_lower']
    for old, new in [('coverage_point_estimate', 'coverage_point_estimate_pct'), ('coverage_ci_lower', 'coverage_ci_lower_pct'), ('coverage_ci_upper', 'coverage_ci_upper_pct')]:
        coverage_comparison[new] = 100 * coverage_comparison[old]
    coverage_comparison['coverage_ci_width_pp'] = 100 * (coverage_comparison['coverage_ci_upper'] - coverage_comparison['coverage_ci_lower'])
    coverage_comparison['signed_gap_ci_width_pp'] = coverage_comparison['signed_gap_ci_upper_pp'] - coverage_comparison['signed_gap_ci_lower_pp']
    comparison_frames = {'performance_comparison.csv': performance, 'paired_difference_comparison.csv': paired_comparison,
                         'difference_ci_comparison.csv': difference_comparison, 'coverage_gap_ci_comparison.csv': coverage_comparison}
    for filename, frame in comparison_frames.items():
        save_frame(REPORT_DIR / filename, frame)
    common_counts = main_counts.copy()
    common_counts['source_file'] = INPUT_PATHS['main_counts'].relative_to(PROJECT_DIR).as_posix()
    common_counts['source_sha256'] = input_hashes['main_counts']
    save_frame(REPORT_DIR / 'common_replicate_group_counts.csv', common_counts)


    # 6. 半幅、High-VIX Coverage、paired差の図。全範囲と非対称CIを実際の値で表示。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {63: '#238B72', 126: '#A77B16', 252: '#245A81'}
    figure_files = []
    fig, ax = plt.subplots(figsize=(12, 5.2), layout='constrained')
    for window in WINDOWS:
        frame = intervals_by_window[window]
        ax.plot(frame['date'], frame['q_rolling_pct'], color=colors[window], linewidth=1.5 if window == 252 else 1,
                label=f'Rolling-{window}' + (' (primary; reused)' if window == 252 else ' (sensitivity)'))
    ax.plot(fixed['date'], fixed['q_fixed_pct'], color='#B84B25', linestyle='--', linewidth=1.2, label='Fixed (2019 benchmark; reused)')
    ax.set_title('Test prediction-interval half-widths: Rolling63/126/252 and common Fixed', loc='left', fontsize=12)
    ax.set_xlabel('Outcome date (SP500 market day)')
    ax.set_ylabel('Interval half-width (return percentage points)')
    ax.set_ylim(bottom=0)
    ax.margins(x=.01, y=.10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper right', fontsize=9)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_halfwidth_timeseries_by_window.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    high_coverage = coverage_comparison.loc[coverage_comparison['group'].eq('high_vix')].reset_index(drop=True)
    high_difference = difference_comparison.loc[difference_comparison['group'].eq('high_vix')]
    fig, ax = plt.subplots(figsize=(10, 6.2), layout='constrained')
    for y, row in enumerate(high_coverage.itertuples()):
        point, lower, upper = row.coverage_point_estimate_pct, row.coverage_ci_lower_pct, row.coverage_ci_upper_pct
        color = '#B84B25' if row.method == 'fixed' else colors[int(row.rolling_window)]
        if row.valid_replicates:
            ax.hlines(y, lower, upper, color=color, linewidth=2, label='95% percentile CI' if y == 0 else None)
            ax.vlines([lower, upper], y - .07, y + .07, color=color, linewidth=1.2)
        ax.scatter(point, y, color=color, s=35, zorder=3, label='Original estimate' if y == 0 else None)
        ax.text(.01, y + .25, f'{point:.2f}% [{lower:.2f}%, {upper:.2f}%]', transform=ax.get_yaxis_transform(), color=color, fontsize=9)
    ax.axvline(90, color='#555555', linestyle='--', linewidth=1, label='Nominal interval coverage = 90%')
    ax.set_yticks([0, 1, 2, 3], ['Fixed (2019 benchmark)', 'Rolling-63 (sensitivity)', 'Rolling-126 (sensitivity)', 'Rolling-252 (primary; reused)'])
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Empirical coverage (%)')
    ax.set_title('High-VIX coverage: common n=149 for every method\n95% percentile CI; paired MBB block length=10, B=5,000', loc='left', fontsize=12)
    ax.grid(axis='x', alpha=.18)
    ax.legend(loc='upper right', fontsize=8)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'02_high_vix_coverage_by_window.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, layout='constrained')
    for ax, (metric, title, xlabel) in zip(axes, [
        ('coverage_difference', 'Coverage difference', 'Coverage difference (percentage points)'),
        ('mean_width_difference', 'Mean width difference', 'Width difference (return percentage points)'),
        ('mean_interval_score_difference', 'Mean interval score difference', 'IS difference (return percentage points)'),
    ]):
        panel = high_difference.loc[high_difference['metric'].eq(metric)].set_index('rolling_window')
        for y, window in enumerate(WINDOWS):
            row = panel.loc[window]
            point, lower, upper = row['display_point_estimate'], row['display_ci_lower'], row['display_ci_upper']
            if row['valid_replicates']:
                ax.hlines(y, lower, upper, color=colors[window], linewidth=2, label='95% percentile CI' if y == 0 else None)
                ax.vlines([lower, upper], y - .07, y + .07, color=colors[window], linewidth=1.2)
            ax.scatter(point, y, color=colors[window], s=32, zorder=3, label='Original estimate' if y == 0 else None)
            ax.text(.02, y + .25, f'{point:.2f} [{lower:.2f}, {upper:.2f}]', transform=ax.get_yaxis_transform(), fontsize=9)
        ax.axvline(0, color='#555555', linestyle='--', linewidth=.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_yticks([0, 1, 2], ['Rolling-63 (sensitivity)', 'Rolling-126 (sensitivity)', 'Rolling-252 (primary)'])
        ax.grid(axis='x', alpha=.18)
        ax.margins(x=.12)
    axes[0].set_ylim(2.6, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('High-VIX: each Rolling window minus common Fixed (2019)\n95% percentile CI; same 149 outcome days; primary window 252 unchanged', fontsize=12)
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'03_high_vix_differences_by_window.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)


    # 7. 入力・主成果物を保全した記録と、段階別検証の完了を保存する。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    protected_hashes_after = {name: file_hash(PROJECT_DIR / name) for name in protected_hashes}
    assert input_hashes_after == input_hashes and protected_hashes_after == protected_hashes
    validation = {'status': 'PASS', 'new_windows': NEW_WINDOWS, 'primary_window_reused': 252, 'common_test_n': N, 'regime_counts': EXPECTED_N,
        'stage_validations': stage_validations, 'existing_252_and_fixed_original_columns_exact_reuse': True,
        'fixed_benchmark_has_no_rolling_window_value': True, 'all_bootstrap_group_counts_equal_saved_primary': True,
        'all_bootstrap_fixed_means_match_saved_primary': True, 'first_3_block_audit_exact_match_main': True,
        'shared_bootstrap_draws_generated_once': B, 'first_3_only_regenerated_for_verification': True,
        'saved_scores_reused_no_regeneration': True, 'ols_refit': False, 'primary_252_recomputed': False,
        'fixed_intervals_or_ci_regenerated': False, 'additional_setting_combinations_or_models_run': False,
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'protected_main_hashes_before': protected_hashes, 'protected_main_hashes_after': protected_hashes_after}
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'started_at_utc': started_at.isoformat(), 'ended_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_sha256': file_hash(Path(__file__).resolve()), 'new_windows': NEW_WINDOWS, 'reused_primary_window': 252,
        'fixed_is_common_2019_benchmark': True, 'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]} for name, path in INPUT_PATHS.items()},
        'root_csv_sha256': {path.name: file_hash(path) for path in REPORT_DIR.glob('*.csv')}, 'figures': figure_files,
        'per_window_metadata': ['window_63/execution_metadata.json', 'window_126/execution_metadata.json']}
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


    # 8. Methods/Results/限界を次段階で統合できるよう、日本語の説明と全補助結果を残す。
    summary = '''# 追加検証③：Rolling calibration windowの感度分析

## 範囲・固定条件・コード対応

新規window63・126だけで区間→履歴検証→日次評価→評価検証→bootstrapの順に実行した。
主window252の区間・日次評価・点推定・bootstrap統計量・CIは既存CSVから完全精度で再利用し、再計算していない。
OLSは再推定せず、固定係数JSON・点予測・実測リターン・既存Calibration/Test絶対誤差scoreは共通である。
Fixedは2019年全252 scoreに基づく単一の既存benchmarkで、短縮・再生成していない。
比較表のFixed行はrolling_windowを空欄にし、Fixedに63/126/252の設定があるようには表示していない。

市場区分は前営業日vix_lag1>=30（無丸め・無許容差）。全手法で2020-01-02〜2026-08-07の同じ1,658日を使い、High-VIX149日・Non-high1,509日を維持した。
Test開始日や評価期間をwindowごとに変更していない。2026年は8月7日までの部分年。
bootstrapはblock length10、B=5,000、seed202701、default_rng/PCG64、非循環Paired MBB。
開始位置0〜1,648から166個の10市場日blockを復元抽出し、連結後の先頭1,658行（最後8行）を保持した。
1つのindexをH・Fixed・R63・R126の全日次指標へ適用し、ソート・重複削除を行わず、分母は各回で数え直した。
空区分はNaNで再抽出・0埋めなし。有効値の2.5/97.5%点をlinear percentileでCIにした。
Conformal名目Coverage90%とbootstrap CIの信頼水準95%は別である。

実行コードは`18_rolling_window_sensitivity.py`。コメント1で入力とone-to-one日付照合、2で過去scoreの順位による区間と履歴監査、
3の`interval_metrics`でinclusive/Width/IS、4の`summarize_sample`/`rolling_ci_tables`でpaired bootstrapとCI、5で出所付き比較表、6で図を作る。
前段階をimportして実行する構成ではない。

## 初期履歴と最初の半幅

| Rolling window | 初期score最古日 | 最新日 | 順位k | Python添字 | Test初日のq（return pp） | 出所 |
|---:|---|---|---:|---:|---:|---|
'''
    for row in initial_history.itertuples():
        summary += f'| {row.rolling_window} | {row.initial_history_start:%Y-%m-%d} | {row.initial_history_end:%Y-%m-%d} | {row.k} | {row.zero_based_index} | {row.first_q_rolling_pct:.15g} | {row.result_origin} |\n'
    summary += '''
初期履歴は2019年末尾w個。各日の履歴は観測済み全scoreの末尾w個で、Training残差や高VIX限定のscore履歴を使っていない。
`k=ceil((w+1)*0.90)`とし、コピーをソートしたk番目を補間せず使用した。時系列履歴自体は並べ替えない。
区間を記録した後で当日の保存Test scoreを取得し、翌日用履歴へ追加・最古を除く。
Testの0始まりindex iで、Calibration由来max(w-i,0)、過去Test由来min(i,w)、Training由来0を全日確認した。
63/126の初日Fixed一致は要求しない。既存252だけは同じ全2019 scoreを用いるためFixedと初日一致する。

## High-VIXの性能（n=149、全手法共通）

Coverageは%、Gapはcoverage pp、Width/ISはreturn pp。Gapは100*(Coverage-0.90)。

| 手法 | covered | Coverage | signed Gap | absolute Gap | Mean Width | Median Width | Mean IS |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in performance.loc[performance['group'].eq('high_vix')].itertuples():
        label = 'Fixed (2019 benchmark)' if row.method == 'fixed' else f'Rolling-{int(row.rolling_window)}'
        summary += f'| {label} | {row.covered_n} | {100 * row.empirical_coverage:.4f}% | {row.coverage_gap_pp:+.4f} | {row.absolute_coverage_gap_pp:.4f} | {row.mean_width_return_pp:.6f} | {row.median_width_return_pp:.6f} | {row.mean_interval_score_return_pp:.6f} |\n'
    summary += '\nHigh-VIXの各Rolling−共通Fixedの差。主推論は252のCoverage差（two-sided）のまま、新規63/126は感度分析。\n\n| window | 指標 | 点推定 | 95%CI | CI幅 | 0包含 | 有効/無効 |\n|---:|---|---:|---|---:|---|---|\n'
    for row in high_difference.itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {row.rolling_window} | {row.metric} ({row.display_unit}) | {row.display_point_estimate:+.6f} | [{row.display_ci_lower:+.6f}, {row.display_ci_upper:+.6f}] | {row.display_ci_width:.6f} | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += '\nCoverage/GapのCI。Gap CIはCoverage CIから0.90を引いて100倍する。\n\n| 手法 | Coverage (%) | Coverage 95%CI (%) | Gap (pp) | Gap 95%CI (pp) | 90%との関係 |\n|---|---:|---|---:|---|---|\n'
    for row in high_coverage.itertuples():
        label = 'Fixed (2019 benchmark)' if row.method == 'fixed' else f'Rolling-{int(row.rolling_window)}'
        summary += f'| {label} | {row.coverage_point_estimate_pct:.4f} | [{row.coverage_ci_lower_pct:.4f}, {row.coverage_ci_upper_pct:.4f}] | {row.signed_gap_point_estimate_pp:+.4f} | [{row.signed_gap_ci_lower_pp:+.4f}, {row.signed_gap_ci_upper_pp:+.4f}] | {row.ci_relation_to_nominal} |\n'
    summary += '\n## 維持された／変化した解釈\n\n'
    for window in WINDOWS:
        selected = high_difference.loc[high_difference['rolling_window'].eq(window)].set_index('metric')
        messages = []
        for metric, label, positive in [('coverage_difference', 'Coverage差の正方向', True), ('mean_width_difference', 'Width差の正方向', True),
                                         ('mean_interval_score_difference', 'IS差の負方向', False)]:
            row = selected.loc[metric]
            result = '有効値なし' if row['valid_replicates'] == 0 else (
                'CIが0を除外して支持' if (row['ci_lower'] > 0 if positive else row['ci_upper'] < 0) else ('CIが0を含む' if row['ci_includes_zero'] else 'CIは反対方向'))
            messages.append(label + '：' + result)
        relation = high_coverage.loc[high_coverage['method'].eq('rolling') & high_coverage['rolling_window'].eq(window), 'ci_relation_to_nominal'].iloc[0]
        summary += f'- Rolling-{window}：' + '、'.join(messages) + f'。Coverage CIと90%の関係は{relation}。\n'
    summary += '''
windowが変わると区間が変わるため、点推定は変わり得る。評価日と市場区分の構成は変えていない。
同じ日のFixedと各Rollingのpaired差の不確実性と、異なるwindow同士の差の検定は別である。
window間の新しい仮説検定やCIの重なりだけによる有意差判断、最適windowの事後選択は行っていない。
短いwindowが常にCoverageを改善する、狭い区間になる、優れるとは仮定していない。
window変更は最近の誤差への追随だけでなく分位点推定の変動と有限標本の順位にも影響し、結果だけから適応速度が原因とは断定できない。
Coverage CIが90%を含んでも達成の証明とは扱わず、Overallの近さで高VIXの不足を覆い隠さない。

## Overall・非高VIXの補助結果

| 区分 | 手法 | n | Coverage (%) | signed Gap (pp) | Mean Width | Median Width | Mean IS |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    for row in performance.loc[performance['group'].ne('high_vix')].itertuples():
        label = 'Fixed (2019 benchmark)' if row.method == 'fixed' else f'Rolling-{int(row.rolling_window)}'
        summary += f'| {row.group} | {label} | {row.n} | {100 * row.empirical_coverage:.4f} | {row.coverage_gap_pp:+.4f} | {row.mean_width_return_pp:.6f} | {row.median_width_return_pp:.6f} | {row.mean_interval_score_return_pp:.6f} |\n'
    summary += '\n| 区分 | Rolling window | 指標 | 点推定 | 95%CI | 0包含 | 有効/無効 |\n|---|---:|---|---:|---|---|---|\n'
    for row in difference_comparison.loc[difference_comparison['group'].ne('high_vix')].itertuples():
        contains = '未定義' if row.valid_replicates == 0 else ('含む' if row.ci_includes_zero else '含まない')
        summary += f'| {row.group} | {row.rolling_window} | {row.metric} ({row.display_unit}) | {row.display_point_estimate:+.6f} | [{row.display_ci_lower:+.6f}, {row.display_ci_upper:+.6f}] | {contains} | {row.valid_replicates}/{row.invalid_replicates} |\n'
    summary += '\n各手法の全区分Coverage/Gap CIとabsolute Gapも比較CSVへ保存した。\n\n## 再標本件数と検証\n\n全windowで同じindexなので、各回の件数も共通。以下は主設定の保存記録と完全一致した件数分布。\n\n| 区分 | 有効/無効 | 最小 | 平均 | 中央値 | 最大 | 件数2.5/97.5%点 |\n|---|---|---:|---:|---:|---:|---|\n'
    for row in main_counts.itertuples():
        summary += f'| {row.group} | {row.valid_group_replicates}/{row.zero_group_replicates} | {row.min_n} | {row.mean_n:.4f} | {row.n_median:.2f} | {row.max_n} | [{row.n_q025:.2f}, {row.n_q975:.2f}] |\n'
    summary += '''
入力はdateのouter one-to-one結合で全日存在と共通のorigin・split・点予測・実測値・区間を確認した。不一致をdropしていない。
保存scoreの有限・非負とsigned error/絶対値の式を照合したが、新たなscoreは生成していない。
各区間の中心・上下限・2q、全日履歴の末尾w一致・source内訳を確認し、最初3日・中盤・最終日の5日は別のdate<t→tail(w)→sortで再構成した。
これらの検証後に日次評価へ進み、inclusive・IS>=Width・missの分割・paired平均差・区分加重平均と保存後の全列一致を確認した。
全5,000回で長さ・開始位置・block内連続性・共通index・区分件数を検証した。各回のFixedの平均も主保存統計と一致した。
最初3回のblock監査全列が主設定と完全一致した。再現性のための再生成は最初3回だけで、全5,000回を繰り返していない。
空区分・全NaN CIの分岐も検証用コピーで確認し、再抽出しない規則を維持した。
主252とFixedの比較表・CIは全元列の完全一致で転載し、FixedにRolling windowを付けていない。
必要な入力・主分析成果物のSHA-256は実行前後で不変だった。未解決の入力・実装上の不整合は見つからなかった。

## 保存先と次段階の準備

- `18_rolling_window_sensitivity.py`：保存済み入力から単独実行。
- `reports/rolling_window_sensitivity/window_63/`、`window_126/`：区間・履歴監査・代表日再構成、日次評価、Rolling性能・paired差、replicate統計15,000行、差/rolling Coverage/Gap CI、件数要約・再現性・段階別検証・設定・環境・入力ハッシュ・実行ログ。
- 親reportsフォルダ：`initial_history_comparison.csv`、`performance_comparison.csv`、`paired_difference_comparison.csv`、`difference_ci_comparison.csv`、`coverage_gap_ci_comparison.csv`、`common_replicate_group_counts.csv`、`first_replicate_block_starts.csv`、検証・メタデータ・実行ログ。
- `figures/rolling_window_sensitivity/`：半幅時系列、高VIX Coverage CI、3差CI。各PNG 300dpi・PDF。目視確認は実行後に同reportsフォルダの`figure_visual_qa.md`へ記録する。

実行：`C:\\Users\\sohas\\anaconda3\\python.exe 18_rolling_window_sensitivity.py`。
主分析と3種類の事前指定感度分析の結果・出所・検証が揃い、次に卒論用図表、Methods、Results、限界を統合できる状態である。
今回は統合原稿の作成や追加モデル・全組合せ分析には進んでいない。

## 残る限界

保存済みモデル・点予測・score・生成済み区間/評価系列に条件づけた近似的不確実性評価で、学習から全工程を再推定する不確実性は含まない。
MBBはstationarity・weak dependence・block lengthなどの条件の影響を受け、金融時系列や高VIX部分集団の90%Coverageを保証しない。
3つの事前指定windowの感度を確認したことは、あらゆる非定常性・時系列依存・stress定義への頑健性を示すものではない。
主分析は252のまま維持し、VIX25/35やblock length5/20との全組合せ、他モデルへ広げていない。
'''
    (PROJECT_DIR / 'reports/rolling_window_sensitivity_summary.md').write_text(summary, encoding='utf-8')
    print('\nInitial history:\n' + initial_history.to_string(index=False))
    print('\nHigh-VIX performance:\n' + performance.loc[performance['group'].eq('high_vix')].to_string(index=False))
    print('\nHigh-VIX differences:\n' + high_difference.to_string(index=False))
    print('\nHigh-VIX Coverage and Gap CI:\n' + high_coverage.to_string(index=False))
    print('Rolling window sensitivity: PASS. New 63/126 only; main252 and Fixed reused; visual QA to follow.')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 18.')
    run_rolling_window_sensitivity(ANALYSIS_WORKSPACE)
