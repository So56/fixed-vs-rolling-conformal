"""直前252市場日のscoreで区間を確定し、当日観測後に履歴を更新する。
実行入口はrun_analysis.pyのstage 13。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rolling_conformal(workspace):
    """保存済みの固定OLS予測を読み、独立workspaceへ区間と検証記録を保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'rolling_conformal'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'rolling_conformal'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'analysis_dataset': PROJECT_DIR / 'data/processed/analysis_dataset.csv',
        'fixed_coefficients': PROJECT_DIR / 'models/ols/fixed_coefficients.json',
        'test_point_predictions': PROJECT_DIR / 'reports/ols/test_point_predictions.csv',
        'calibration_scores': PROJECT_DIR / 'reports/fixed_conformal/calibration_scores_2019.csv',
        'fixed_threshold': PROJECT_DIR / 'models/fixed_conformal/fixed_threshold.json',
        'fixed_intervals': PROJECT_DIR / 'reports/fixed_conformal/test_fixed_intervals.csv',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    RTOL = ATOL = 1e-12  # 数値照合だけに使用。後続のcovered判定に足さない。




    def conformal_quantile(scores, alpha):
        """履歴を変更せず、コピーをソートしてk番目を選ぶ。補間・乱数は使わない。"""
        n = len(scores)
        k = math.ceil((n + 1) * (1 - alpha))
        sorted_scores = np.sort(np.asarray(scores, dtype=float), kind='stable')
        q = float(sorted_scores[k - 1]) if k <= n else math.inf
        return q, k


    def observe_and_update(past_history, target_date, point_prediction, actual_return):
        """区間確定後に呼ぶ。誤差・scoreを作り、コピーに追加して最古を除く。"""
        error = float(actual_return - point_prediction)
        score = abs(error)
        assert math.isfinite(error) and math.isfinite(score) and score >= 0
        updated = past_history.copy()
        updated.append((target_date, score, 'test'))
        removed = updated.pop(0)
        assert len(updated) == len(past_history)
        return error, score, updated, removed[0]


    # 1. 保存済み入力だけを読む。前段階の実行・モデル推定は行わない。
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    model = json.loads(INPUT_PATHS['fixed_coefficients'].read_text(encoding='utf-8'))
    fixed_threshold = json.loads(INPUT_PATHS['fixed_threshold'].read_text(encoding='utf-8'))
    assert design['point_model'] == 'OLS AR(1), fitted once on Training only'
    assert design['calibration'] == ['2019-01-01', '2019-12-31']
    assert design['test'] == ['2020-01-01', '2026-08-07']
    assert design['rolling_window'] == 252 and design['nominal_coverage'] == .90
    assert design['rolling_update'] == 'only observed prior scores; append current score after prediction'
    assert design['score'] == 'absolute forecast error' and design['interval_endpoints'] == 'inclusive'
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    assert design['quantile_rule'] == 'ceil((n+1)*0.90)-th sorted score; no interpolation; infinity if k>n'
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert input_hashes['analysis_dataset'] == model['input_csv_sha256']
    assert input_hashes['research_design_lock'] == model['research_design_lock_sha256']
    for name in ['analysis_dataset', 'fixed_coefficients', 'test_point_predictions', 'research_design_lock']:
        assert input_hashes[name] == fixed_threshold['input_files'][name]['sha256']
    assert input_hashes['calibration_scores'] == fixed_threshold['calibration_score_csv_sha256']
    WINDOW = int(design['rolling_window'])
    alpha = float(fixed_threshold['alpha'])
    q_fixed = float(fixed_threshold['q_fixed_pct'])
    assert alpha == .10 and fixed_threshold['nominal_coverage'] == design['nominal_coverage']
    assert fixed_threshold['n'] == WINDOW and fixed_threshold['k'] == 228
    assert math.isfinite(q_fixed) and q_fixed >= 0
    intercept, ar1_coefficient = model['coefficients']['const'], model['coefficients']['return_lag1_pct']
    assert model['training_n'] == 502 and model['predictors'] == ['return_lag1_pct']

    date_columns = ['date', 'forecast_origin_date']
    data = pd.read_csv(
        INPUT_PATHS['analysis_dataset'],
        usecols=['date', 'forecast_origin_date', 'split', 'return_lag1_pct', 'return_pct'],
        parse_dates=date_columns, dtype={'split': 'string', 'return_lag1_pct': 'float64', 'return_pct': 'float64'},
        float_precision='round_trip',
    )
    assert data['date'].is_unique and data['date'].is_monotonic_increasing
    reference = data.loc[data['split'].eq('test')].copy().reset_index(drop=True)
    predictions = pd.read_csv(
        INPUT_PATHS['test_point_predictions'], parse_dates=date_columns, dtype={'split': 'string'},
        float_precision='round_trip',
    )
    assert list(predictions.columns) == ['date', 'forecast_origin_date', 'split', 'return_lag1_pct', 'predicted_return_pct']
    for frame in [reference, predictions]:
        assert len(frame) == 1658 and frame['date'].is_unique and frame['date'].is_monotonic_increasing
        assert frame['date'].iloc[0] == pd.Timestamp('2020-01-02')
        assert frame['date'].iloc[-1] == pd.Timestamp('2026-08-07')
        assert frame['split'].eq('test').all() and frame['forecast_origin_date'].lt(frame['date']).all()
        assert not frame.isna().any().any()
    assert np.isfinite(reference[['return_pct', 'return_lag1_pct']].to_numpy()).all()
    assert np.isfinite(predictions[['predicted_return_pct', 'return_lag1_pct']].to_numpy()).all()
    joined = reference.merge(predictions, on='date', how='outer', validate='one_to_one', indicator=True,
                             suffixes=('_source', '_prediction'))
    assert len(joined) == 1658 and joined['_merge'].eq('both').all()
    assert joined['forecast_origin_date_source'].equals(joined['forecast_origin_date_prediction'])
    assert joined['split_source'].equals(joined['split_prediction'])
    np.testing.assert_allclose(joined['return_lag1_pct_source'], joined['return_lag1_pct_prediction'], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(joined['predicted_return_pct'], intercept + ar1_coefficient * joined['return_lag1_pct_source'], rtol=RTOL, atol=ATOL)
    # 予測入力と実測値を分ける。ループ内では区間記録後に実測値を参照する。
    test_inputs = pd.DataFrame({
        'date': joined['date'], 'forecast_origin_date': joined['forecast_origin_date_source'],
        'split': joined['split_source'], 'predicted_return_pct': joined['predicted_return_pct'],
    })
    actual_by_date = joined.set_index('date')['return_pct']
    pd.testing.assert_series_equal(test_inputs['date'], predictions['date'], check_names=False)

    fixed = pd.read_csv(INPUT_PATHS['fixed_intervals'], parse_dates=date_columns,
                        dtype={'split': 'string'}, float_precision='round_trip')
    assert len(fixed) == 1658 and fixed['date'].is_unique and fixed['date'].is_monotonic_increasing
    pd.testing.assert_frame_equal(fixed[list(test_inputs.columns)], test_inputs, check_exact=True)
    assert fixed['q_fixed_pct'].eq(q_fixed).all()

    # 2. 初期履歴は2019年のscoreだけ。各レコードは(date, score, source)。
    calibration = pd.read_csv(INPUT_PATHS['calibration_scores'], usecols=['date', 'split', 'score_pct'],
                              parse_dates=['date'], dtype={'split': 'string'}, float_precision='round_trip')
    assert len(calibration) == WINDOW and calibration['split'].eq('calibration').all()
    assert calibration['date'].is_unique and calibration['date'].is_monotonic_increasing
    assert calibration['date'].dt.year.eq(2019).all() and not calibration.isna().any().any()
    assert np.isfinite(calibration['score_pct']).all() and calibration['score_pct'].ge(0).all()
    calibration_dates = data.loc[data['split'].eq('calibration'), 'date'].reset_index(drop=True)
    pd.testing.assert_series_equal(calibration['date'], calibration_dates)
    history = [(row.date, float(row.score_pct), 'calibration') for row in calibration.itertuples(index=False)]
    observed_history = history.copy()  # 後から値・出所も照合するための観測済みscore全履歴。
    all_score_dates = calibration['date'].tolist() + test_inputs['date'].tolist()
    interval_rows, score_rows, audit_rows = [], [], []

    # 3. 各Test日をA→B→C→D→Eの順で処理する。
    for i, row in enumerate(test_inputs.itertuples(index=False)):
        # A: 当日を含まない直近252個を確認する。履歴自体をscore順に並べ替えない。
        dates = pd.DatetimeIndex([item[0] for item in history])
        values = np.asarray([item[1] for item in history], dtype=float)
        sources = [item[2] for item in history]
        assert len(history) == WINDOW and np.isfinite(values).all() and (values >= 0).all()
        assert dates.is_unique and dates.is_monotonic_increasing and (dates < row.date).all()
        assert dates[-1] == row.forecast_origin_date
        assert dates.tolist() == all_score_dates[i:i + WINDOW]
        assert history == observed_history[-WINDOW:]
        assert set(sources).issubset({'calibration', 'test'})
        calibration_n, past_test_n = sources.count('calibration'), sources.count('test')
        assert calibration_n == max(WINDOW - i, 0) and past_test_n == min(i, WINDOW)
        assert calibration_n + past_test_n == WINDOW and sources.count('training') == 0

        # B: np.sortはコピーを返す。228番目（添字227）を直接選ぶ。
        q_rolling, k = conformal_quantile(values, alpha)
        assert k == 228 and math.isfinite(q_rolling) and q_rolling >= 0
        # C: 点予測から区間を確定し、使用履歴と一緒に記録する。
        interval_rows.append({
            'date': row.date, 'forecast_origin_date': row.forecast_origin_date, 'split': row.split,
            'predicted_return_pct': row.predicted_return_pct, 'q_rolling_pct': q_rolling,
            'lower_rolling_pct': row.predicted_return_pct - q_rolling,
            'upper_rolling_pct': row.predicted_return_pct + q_rolling,
        })
        audit = {
            'date': row.date, 'forecast_origin_date': row.forecast_origin_date,
            'score_count': len(history), 'oldest_score_date': dates[0], 'latest_score_date': dates[-1],
            'calibration_score_count': calibration_n, 'past_test_score_count': past_test_n,
            'training_score_count': sources.count('training'), 'k': k, 'q_rolling_pct': q_rolling,
            'history_start_sequence_index': i, 'history_end_sequence_index': i + WINDOW - 1,
        }
        audit_rows.append(audit)

        # D: ここで初めて当日の実測値を参照し、signed errorとscoreを計算する。
        actual = float(actual_by_date.loc[row.date])
        # E: 当日のscoreを追加して最古を除き、翌日用の履歴へ更新する。
        error, score, updated_history, removed_date = observe_and_update(history, row.date, row.predicted_return_pct, actual)
        assert removed_date == dates[0]
        score_rows.append({
            'date': row.date, 'forecast_origin_date': row.forecast_origin_date, 'split': row.split,
            'return_pct': actual, 'predicted_return_pct': row.predicted_return_pct,
            'test_error_pct': error, 'test_score_pct': score,
        })
        audit['removed_after_observation_date'] = removed_date
        audit['appended_after_observation_date'] = row.date
        observed_history.append((row.date, score, 'test'))
        history = updated_history
        assert history == observed_history[-WINDOW:] and len(history) == WINDOW

    intervals, test_scores, audit = pd.DataFrame(interval_rows), pd.DataFrame(score_rows), pd.DataFrame(audit_rows)
    intervals['split'] = intervals['split'].astype('string')
    test_scores['split'] = test_scores['split'].astype('string')
    assert len(intervals) == len(test_scores) == len(audit) == 1658
    pd.testing.assert_frame_equal(intervals[list(test_inputs.columns)], test_inputs, check_exact=True)
    assert np.isfinite(intervals[['q_rolling_pct', 'lower_rolling_pct', 'upper_rolling_pct']].to_numpy()).all()
    assert intervals['lower_rolling_pct'].le(intervals['upper_rolling_pct']).all()
    assert test_scores['test_score_pct'].ge(0).all() and np.isfinite(test_scores['test_score_pct']).all()
    np.testing.assert_array_equal(test_scores['test_error_pct'], test_scores['return_pct'] - test_scores['predicted_return_pct'])
    np.testing.assert_array_equal(test_scores['test_score_pct'], np.abs(test_scores['test_error_pct']))
    np.testing.assert_allclose((intervals['lower_rolling_pct'] + intervals['upper_rolling_pct']) / 2,
                               intervals['predicted_return_pct'], rtol=RTOL, atol=ATOL)
    # 上下限の構成式の照合のみ。平均Widthの計算は行わない。
    np.testing.assert_allclose(intervals['upper_rolling_pct'] - intervals['lower_rolling_pct'],
                               2 * intervals['q_rolling_pct'], rtol=RTOL, atol=ATOL)
    assert audit['score_count'].eq(WINDOW).all() and audit['training_score_count'].eq(0).all()
    assert audit['latest_score_date'].lt(audit['date']).all()
    assert audit['q_rolling_pct'].equals(intervals['q_rolling_pct'])
    assert intervals['q_rolling_pct'].iloc[0] == q_fixed
    np.testing.assert_array_equal(intervals.loc[0, ['lower_rolling_pct', 'upper_rolling_pct']].to_numpy(dtype=float),
                                   fixed.loc[0, ['lower_fixed_pct', 'upper_fixed_pct']].to_numpy(dtype=float))
    assert audit['latest_score_date'].iloc[0] == pd.Timestamp('2019-12-31')
    np.testing.assert_array_equal(audit['calibration_score_count'].iloc[:3], [252, 251, 250])
    np.testing.assert_array_equal(audit['past_test_score_count'].iloc[:3], [0, 1, 2])

    # 4. CSV保存後、全列をround-tripで読み直す。
    output_frames = {
        'test_rolling_intervals.csv': intervals,
        'test_scores.csv': test_scores,
        'history_audit.csv': audit,
    }
    audit_date_columns = date_columns + ['oldest_score_date', 'latest_score_date',
                                         'removed_after_observation_date', 'appended_after_observation_date']
    saved_frames = {}
    for filename, frame in output_frames.items():
        path = REPORT_DIR / filename
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
        dates_to_parse = audit_date_columns if filename == 'history_audit.csv' else date_columns
        dtype = {'split': 'string'} if 'split' in frame.columns else None
        saved = pd.read_csv(path, parse_dates=dates_to_parse, dtype=dtype, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)
        saved_frames[filename] = saved

    # 5. 保存したscoreから代表日の履歴を別方法で再構成する（全252個×全日の展開はしない）。
    saved_calibration = pd.read_csv(INPUT_PATHS['calibration_scores'], usecols=['date', 'score_pct'],
                                   parse_dates=['date'], float_precision='round_trip')
    saved_calibration['score_source'] = 'calibration'
    saved_test_scores = saved_frames['test_scores.csv']
    catalog = saved_test_scores[['date', 'test_score_pct']].rename(columns={'test_score_pct': 'score_pct'}).copy()
    catalog['score_source'] = 'test'
    catalog = pd.concat([saved_calibration, catalog], ignore_index=True)
    assert len(catalog) == WINDOW + len(test_inputs) and catalog['date'].is_unique and catalog['date'].is_monotonic_increasing
    representative_indices = [0, 1, 2, len(test_inputs) // 2, len(test_inputs) - 1]
    reconstruction_rows, reconstructed_histories = [], {}
    for i in representative_indices:
        target_date = intervals['date'].iloc[i]
        past = catalog.loc[catalog['date'].lt(target_date)].tail(WINDOW).copy()
        assert len(past) == WINDOW and past['date'].lt(target_date).all()
        reconstructed = [(r.date, float(r.score_pct), r.score_source) for r in past.itertuples(index=False)]
        reconstructed_histories[i] = reconstructed
        # pandasのsort_valuesでも補間なしの同じ順位を直接選ぶ。
        q_reconstructed = float(past['score_pct'].sort_values(kind='stable').iloc[227])
        assert q_reconstructed == intervals['q_rolling_pct'].iloc[i]
        assert past['date'].iloc[0] == audit['oldest_score_date'].iloc[i]
        assert past['date'].iloc[-1] == audit['latest_score_date'].iloc[i]
        assert int(past['score_source'].eq('calibration').sum()) == audit['calibration_score_count'].iloc[i]
        assert int(past['score_source'].eq('test').sum()) == audit['past_test_score_count'].iloc[i]
        reconstruction_rows.append({
            'date': target_date, 'score_count': len(past), 'oldest_score_date': past['date'].iloc[0],
            'latest_score_date': past['date'].iloc[-1], 'calibration_score_count': int(past['score_source'].eq('calibration').sum()),
            'past_test_score_count': int(past['score_source'].eq('test').sum()), 'k': 228,
            'saved_q_rolling_pct': intervals['q_rolling_pct'].iloc[i], 'reconstructed_q_rolling_pct': q_reconstructed,
        })
    pd.DataFrame(reconstruction_rows).to_csv(REPORT_DIR / 'validation_reconstructed_histories.csv', index=False, date_format='%Y-%m-%d')

    # 6. 当日実測値だけをメモリ内で変更する検証。元CSV・点予測・本来の履歴は変更しない。
    perturbation_rows = []
    for i in [representative_indices[0], representative_indices[3], representative_indices[4]]:
        target_date = intervals['date'].iloc[i]
        past = reconstructed_histories[i]
        untouched_past = past.copy()
        point_prediction = float(intervals['predicted_return_pct'].iloc[i])
        original_actual = float(saved_test_scores['return_pct'].iloc[i])
        original_score = float(saved_test_scores['test_score_pct'].iloc[i])
        altered_actual = point_prediction + original_score + 100.0  # 検証専用の仮の実測値。
        checked_intervals, next_histories, checked_scores = [], [], []
        for actual in [original_actual, altered_actual]:
            q, checked_k = conformal_quantile([record[1] for record in past], alpha)
            checked_intervals.append((q, point_prediction - q, point_prediction + q))
            # 上の区間確定後に同じ更新関数を呼ぶ。翌日のqの変化は要求しない。
            error, score, next_history, removed_date = observe_and_update(past, target_date, point_prediction, actual)
            assert removed_date == past[0][0]
            next_histories.append(next_history)
            checked_scores.append(score)
        assert past == untouched_past
        assert checked_intervals[0] == checked_intervals[1]
        np.testing.assert_array_equal(checked_intervals[0], intervals.loc[i, ['q_rolling_pct', 'lower_rolling_pct', 'upper_rolling_pct']].to_numpy(dtype=float))
        assert checked_scores[0] == original_score and checked_scores[1] != original_score
        assert next_histories[0][:-1] == next_histories[1][:-1] == past[1:]
        assert next_histories[0][-1] == (target_date, original_score, 'test')
        assert next_histories[1][-1] == (target_date, checked_scores[1], 'test')
        expected_next = catalog.loc[catalog['date'].le(target_date)].tail(WINDOW)
        assert next_histories[0] == [(r.date, float(r.score_pct), r.score_source) for r in expected_next.itertuples(index=False)]
        perturbation_rows.append({
            'date': target_date, 'original_actual_return_pct': original_actual,
            'synthetic_actual_return_pct_validation_only': altered_actual,
            'original_score_pct': original_score, 'synthetic_score_pct_validation_only': checked_scores[1],
            'q_and_current_interval_unchanged': True, 'past_251_records_unchanged': True,
            'appended_score_changed': True, 'next_threshold_change_required': False,
        })
    pd.DataFrame(perturbation_rows).to_csv(REPORT_DIR / 'validation_current_actual_change.csv', index=False, date_format='%Y-%m-%d')

    # 7. TestのRolling半幅の推移。Fixed線との比較は閾値の図示で、性能評価ではない。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11.5, 5.2), layout='constrained')
    ax.plot(intervals['date'], intervals['q_rolling_pct'], color='#245A81', linewidth=1.1,
            label='Rolling: 228th of previous 252 observed scores')
    ax.axhline(q_fixed, color='#B84B25', linestyle='--', linewidth=1.2,
               label=f'Fixed (2019): {q_fixed:.6f} pp')
    ax.set_title('Test: Rolling-252 and Fixed calibration half-widths', loc='left')
    ax.set_xlabel('Outcome date (SP500 market day)')
    ax.set_ylabel('Prediction-interval half-width (percentage points)')
    ax.set_ylim(bottom=0)
    ax.margins(x=.01, y=.10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper right')
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_rolling_and_fixed_halfwidth_timeseries.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    # 8. 必要な入力の保全・実行記録。Coverage等の評価は計算しない。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    assert input_hashes == input_hashes_after
    executed_at = datetime.now(timezone.utc)
    validation = {
        'status': 'PASS', 'test_interval_rows': len(intervals), 'test_score_rows': len(test_scores),
        'history_audit_rows': len(audit), 'window': WINDOW, 'nominal_coverage': design['nominal_coverage'],
        'alpha': alpha, 'k': 228, 'zero_based_index': 227, 'random_tie_breaking': False,
        'initial_calibration_n': len(calibration), 'first_day_q_and_endpoints_exactly_equal_fixed': True,
        'all_days_history_matches_last_252_observed_records': True,
        'all_days_history_dates_match_concatenated_score_sequence_slice': True,
        'all_days_latest_score_equals_forecast_origin_and_precedes_target': True,
        'training_scores_in_history': 0, 'fixed_common_dates_origins_splits_predictions_exact_match': True,
        'interval_centers_finite_ordered_endpoints_and_2q_identity': True,
        'test_error_and_absolute_score_formula_match': True, 'saved_csv_all_columns_exact_round_trip': True,
        'history_reconstruction_dates': [intervals['date'].iloc[i].strftime('%Y-%m-%d') for i in representative_indices],
        'current_actual_change_dates': [row['date'].strftime('%Y-%m-%d') for row in perturbation_rows],
        'current_actual_change_leaves_interval_unchanged_and_changes_appended_score': True,
        'synthetic_checks_affect_saved_data_or_predictions': False, 'next_threshold_change_required': False,
        'numeric_comparison_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'future_covered_rule': 'lower <= actual <= upper; inclusive; no numeric-check tolerance added',
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'prior_steps_reexecuted': False, 'ols_refit': False,
        'coverage_mean_width_interval_score_bootstrap_sensitivity_computed': False,
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
        'window_observations': WINDOW, 'k': 228, 'zero_based_index': 227, 'figures': figure_files,
        'scope': 'Step D-2 only: Rolling intervals, post-outcome Test scores, history audit and validation',
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = f'''# Step D-2：Rolling-252予測区間とTest score

## 実行範囲

保存済みの加工CSV・固定係数・Test点予測・2019年score・Fixed閾値と区間・確定設計を使用した。
前段階のスクリプトを実行せず、OLSも再推定していない。JSONの完全精度の係数と閾値を読み込んだ。
Test 2020-01-02〜2026-08-07の1,658日すべてでRolling区間と、その区間確定後のsigned error・scoreを作成した。
Coverage、平均Width、Interval Score、bootstrap、感度分析は実施していない。

## FixedとRollingで共通のもの・異なるもの

目的変数・対象日・Trainingで固定したOLS AR(1)・点予測・名目90%・絶対誤差score・順位規則は共通である。
Fixedの半幅は2019年の全scoreから選んだ値をTest中固定する。
Rollingの半幅は、各予測時点までに観測済みの直近252個のscoreから更新する。
252暦日ではなく252個の有効なscoreで、週末・休場日をscoreとして追加しない。
VIXや高VIXの条件で履歴を絞り込まず、Training残差は履歴に使用しない。

## 処理順序とコード

実行入口は`13_build_rolling_conformal.py`で、前段のメモリ変数は不要。

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\13_build_rolling_conformal.py
```

番号付きコメント1〜2で入力結合と2019年252 scoreによる履歴初期化を行い、コメント3のループで次を繰り返す。

1. A：履歴数252、日付順・一意性・値の有限性・非負性、全score日付が当日より前であることを確認する。
2. B：`conformal_quantile`でscoreのコピーをソートし、`ceil((252+1)*0.90)=228`番目、Python添字227を直接選ぶ。
3. C：保存点予測に対して上下にq_rollingを足し引きし、区間と使用履歴の監査情報を記録する。
4. D：その後で`actual_by_date`から当日実測値を参照し、`observe_and_update`でsigned errorと絶対誤差scoreを計算する。
5. E：当日scoreと日付を追加して最古を除き、次回用に252個を維持する。

`np.sort`はコピーを返し、日付順の履歴自体はscoreの大きさ順に変更しない。補間・random tie-breakingは使わない。
当日のscoreを計算できるのは当日の実測リターンを観測した後なので、当日区間には使わず翌日以降にだけ使う。
Rollingはショックを観測した後に更新する仕組みであり、ショックの事前情報を持つモデルではない。

## 最初の3日

| 対象日 | 使用score数 | Calibration由来 | 過去Test由来 | 最古score日 | 最新score日 | q_rolling（percentage points） | 観測後に除く最古日 |
|---|---:|---:|---:|---|---|---:|---|
'''
    for row in audit.iloc[:3].itertuples():
        summary += f'| {row.date:%Y-%m-%d} | {row.score_count} | {row.calibration_score_count} | {row.past_test_score_count} | {row.oldest_score_date:%Y-%m-%d} | {row.latest_score_date:%Y-%m-%d} | {row.q_rolling_pct!r} | {row.removed_after_observation_date:%Y-%m-%d} |\n'
    summary += f'''
初日2020-01-02は2019年の252 scoreだけを使い、最新score日は2019-12-31、過去Test scoreは0個である。
したがってq_rollingと保存q_fixed={q_fixed!r}が完全一致し、共通の点予測を中心とする上下限も完全一致した。
2日目以降は観測済みTest scoreを1個ずつ加え、最古のCalibration scoreを1個ずつ除く。
qはorder statisticなので、履歴を1個更新しても同じ値にとどまる場合がある。

## 履歴監査と再構成

全1,658日について履歴が観測済み全履歴の末尾252個（値・日付・出所）と一致することを確認した。
同時に、CalibrationとTestの日付を連結した列の`[i:i+252]`と履歴日付が一致することも確認した。
履歴最新日は毎回forecast_origin_dateと一致し、対象日より前だった。Training由来scoreは常に0個。
Calibration由来はmax(252-i,0)、過去Test由来はmin(i,252)で、合計252を維持した。

保存したCalibration scoreとTest scoreを時系列順に連結すれば、任意の対象日より前の行の末尾252個で履歴を再構成できる。
`history_audit.csv`には、この連結列の0始まりの開始・終了添字も保存した。
全252個×全日の巨大な履歴CSVは作成していない。

代表日の再構成日：{', '.join(validation['history_reconstruction_dates'])}。
保存scoreを読み直し、日付による抽出とpandasのsort_valuesで228番目を選ぶ別方法でも、閾値・履歴期間・出所内訳が一致した。

当日実測値だけのメモリ内変更の検証日：{', '.join(validation['current_actual_change_dates'])}。
点予測と過去履歴を固定し、当日の実測値を変更しても当日qと上下限は不変で、当日scoreと次回履歴の追加scoreだけが変わることを確認した。
次回履歴の既存251個と日付は変わらず、最古の除去も一致した。翌日の228番目が必ず変わるという条件は課していない。
仮の変更は検証用コピー内だけで行い、元データ・保存済み点予測・本来のRolling区間とTest scoreには反映していない。

## 保存先と検証結果

- `reports/rolling_conformal/test_rolling_intervals.csv`：1,658日のRolling区間。
- `reports/rolling_conformal/test_scores.csv`：当日の観測後に利用可能になるsigned errorと絶対誤差score。自身の日の区間には不使用。
- `reports/rolling_conformal/history_audit.csv`：各日の252個の履歴期間・出所内訳・順位・閾値・更新日。
- `reports/rolling_conformal/validation_reconstructed_histories.csv`：代表5日の履歴再構成結果。
- `reports/rolling_conformal/validation_current_actual_change.csv`：仮の実測値変更3日の検証結果（主分析用scoreではない）。
- `reports/rolling_conformal/validation_results.json`、`execution_metadata.json`、`execution.log`：設定・入力ハッシュ・検証・実行記録。
- `figures/rolling_conformal/01_rolling_and_fixed_halfwidth_timeseries.png`とPDF：半幅の推移。目視確認は別途`reports/rolling_conformal/figure_visual_qa.md`に記録する。

Test実測値と点予測はouter one-to-oneで結合し、欠落・重複・予測元日・split・ラグ値を照合した。
点予測は保存係数の式と一致した。FixedとRollingは全1,658日の日付・予測元日・split・点予測が完全一致した。
区間中心、有限な上下限、lower<=upper、上限−下限=2*q_rollingを照合した。この構成式照合は平均Widthの評価ではない。
主要3 CSVは保存後のround-trip読込で全列・値が完全一致した。
構成式等の数値照合はrtol=atol=1e-12で、後続のcovered判定にこの許容差を加えない。
将来のcovered判定は両端inclusiveの`lower <= actual <= upper`をそのまま使用する。
必要な入力8ファイルの実行前後ハッシュが一致し、既存データ・係数・点予測・2019年score・Fixed閾値と区間・設計は変更していない。

## 解釈の限界とStep Eへの状態

半幅の更新は履歴内の誤差分布を反映するが、その変化だけからCoverage改善やRollingの優位を結論づけない。
標準的Conformalの有限標本保証は交換可能性に基づく。金融時系列の依存・非定常性や高VIX部分集団について、Rollingにしただけで90%Coverageが保証されるとは主張しない。
区間とscoreの生成・履歴・共通日付に未解決の不整合は見つからなかった。
Step Eで性能比較を行う準備が整っている。次回、Fixed/Rolling共通日付と元データをone-to-oneで結合して評価する。
今回はCoverage、平均Width、Interval Score、bootstrap、感度分析は計算せず、Rolling区間・Test score・履歴の生成と検証で終了した。
'''
    (PROJECT_DIR / 'reports/step_d2_summary.md').write_text(summary, encoding='utf-8')
    print('最初の3日の履歴:\n' + audit.iloc[:3].to_string(index=False, float_format=lambda value: f'{value:.17g}'))
    print('代表再構成日: ' + ', '.join(validation['history_reconstruction_dates']))
    print('当日実測値変更検証日: ' + ', '.join(validation['current_actual_change_dates']))
    print(f'初日のFixed一致: PASS; 全Test区間・score・履歴監査: 各{len(intervals)}行; 入力{len(input_hashes)}件不変')
    print('Step D-2: PASS。性能評価は未実施。図の目視確認は生成後に別途実施。')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 13.')
    build_rolling_conformal(ANALYSIS_WORKSPACE)
