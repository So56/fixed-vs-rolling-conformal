"""保存済み推定値・CIと既存下書きを出所付き予備要約へ統合する。
実行入口はrun_analysis.pyのstage 19。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_preliminary_results(workspace):
    """独立workspaceの保存入力を使う。上流モデル・区間・Bootstrapは再計算しない。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    workspace_dir = Path(workspace).resolve()
    report_dir = workspace_dir / 'reports/preliminary_summary'
    figure_dir = workspace_dir / 'figures/preliminary_summary'
    for directory in [report_dir, figure_dir / 'main', figure_dir / 'appendix']:
        directory.mkdir(parents=True, exist_ok=True)
    CSV_PATHS = {
        'split': 'reports/split_summary.csv', 'variables': 'docs/variable_definitions.csv',
        'coefficients': 'reports/ols/coefficient_table.csv', 'eda_split': 'reports/eda/split_regime_counts.csv',
        'main_performance': 'reports/evaluation/comparison_by_regime.csv', 'main_paired': 'reports/evaluation/paired_differences_by_regime.csv',
        'main_original': 'reports/bootstrap/original_point_statistics.csv',
        'main_difference': 'reports/bootstrap/difference_confidence_intervals.csv', 'main_coverage': 'reports/bootstrap/coverage_gap_confidence_intervals.csv',
        'block_difference': 'reports/bootstrap_sensitivity/difference_ci_comparison.csv', 'block_coverage': 'reports/bootstrap_sensitivity/coverage_gap_ci_comparison.csv',
        'vix_difference': 'reports/vix_threshold_sensitivity/difference_ci_comparison.csv', 'vix_coverage': 'reports/vix_threshold_sensitivity/coverage_gap_ci_comparison.csv',
        'vix_performance': 'reports/vix_threshold_sensitivity/performance_comparison.csv', 'vix_composition': 'reports/vix_threshold_sensitivity/threshold_summary.csv',
        'vix_yearly': 'reports/vix_threshold_sensitivity/yearly_composition.csv',
        'window_difference': 'reports/rolling_window_sensitivity/difference_ci_comparison.csv', 'window_coverage': 'reports/rolling_window_sensitivity/coverage_gap_ci_comparison.csv',
        'window_performance': 'reports/rolling_window_sensitivity/performance_comparison.csv', 'window_initial': 'reports/rolling_window_sensitivity/initial_history_comparison.csv',
        'bootstrap_counts': 'reports/bootstrap/replicate_group_count_summary.csv',
    }
    JSON_PATHS = {
        'design': 'docs/research_design_lock.json', 'model': 'models/ols/fixed_coefficients.json',
        'ols_fit': 'reports/ols/model_fit_statistics.json', 'fixed_threshold': 'models/fixed_conformal/fixed_threshold.json',
        'bootstrap_metadata': 'reports/bootstrap/execution_metadata.json', 'download_manifest': 'data/raw/download_manifest.json',
    }
    KEYS = {
        'split': ['split'], 'variables': ['variable'], 'coefficients': ['coefficient'], 'eda_split': ['split'],
        'main_performance': ['group', 'method'], 'main_paired': ['group'], 'main_original': ['group'],
        'main_difference': ['group', 'metric'], 'main_coverage': ['group', 'method'],
        'block_difference': ['block_length', 'group', 'metric'], 'block_coverage': ['block_length', 'group', 'method'],
        'vix_difference': ['vix_threshold', 'group', 'metric'], 'vix_coverage': ['vix_threshold', 'group', 'method'],
        'vix_performance': ['vix_threshold', 'group', 'method'], 'vix_composition': ['vix_threshold'], 'vix_yearly': ['vix_threshold', 'year'],
        'window_difference': ['rolling_window', 'group', 'metric'], 'window_coverage': ['rolling_window', 'group', 'method'],
        'window_performance': ['rolling_window', 'group', 'method'], 'window_initial': ['rolling_window'], 'bootstrap_counts': ['group'],
    }
    RECORD_DIRS = ['reports/eda', 'reports/ols', 'reports/fixed_conformal', 'reports/rolling_conformal', 'reports/evaluation',
                   'reports/bootstrap', 'reports/bootstrap_sensitivity', 'reports/vix_threshold_sensitivity', 'reports/rolling_window_sensitivity']
    CORE_VALIDATIONS = ['reports/eda/artifact_validation.json', 'reports/ols/validation_results.json', 'reports/fixed_conformal/validation_results.json',
                        'reports/rolling_conformal/validation_results.json', 'reports/evaluation/validation_results.json', 'reports/bootstrap/validation_results.json',
                        'reports/bootstrap_sensitivity/validation_results.json', 'reports/vix_threshold_sensitivity/validation_results.json', 'reports/rolling_window_sensitivity/validation_results.json']
    RTOL = ATOL = 1e-12  # 演算経路が異なる平均値の照合だけに適用。再掲値はexactで照合する。




    def native(value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        return None if pd.isna(value) else value


    def save_csv(name, frame):
        frame = frame.reset_index(drop=True)
        path = report_dir / name
        frame.to_csv(path, index=False)
        dtype = {column: str(frame[column].dtype) for column in frame if str(frame[column].dtype) in ['boolean', 'Int64', 'string']}
        saved = pd.read_csv(path, float_precision='round_trip', dtype=dtype or None)
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)
        tables[name] = frame


    def row_key(frame, index, keys):
        return json.dumps({key: native(frame.loc[index, key]) for key in keys}, ensure_ascii=False)


    def source_map(table, output_row, output_columns, source, source_index=None, fields=None, note='identity; full precision retained'):
        """1行の列群を、元CSV行またはJSON pathへ対応させる。"""
        source_columns = fields or output_columns
        is_csv = source in CSV_PATHS
        path = workspace_dir / (CSV_PATHS[source] if is_csv else JSON_PATHS[source])
        provenance.append({'output_table': table, 'output_row_1_based': output_row + 1,
            'output_row_key': row_key(tables[table], output_row, OUTPUT_KEYS[table]),
            'output_columns': json.dumps(output_columns, ensure_ascii=False), 'source_file': path.relative_to(workspace_dir).as_posix(), 'source_sha256': sha(path),
            'source_row_key': row_key(data[source], source_index, KEYS[source]) if is_csv else 'JSON paths',
            'source_csv_line_1_based': source_index + 2 if is_csv else np.nan,
            'source_fields_or_json_paths': json.dumps(source_columns, ensure_ascii=False), 'transformation': note})


    def copy_table(name, source, mask=None):
        selected = data[source] if mask is None else data[source].loc[mask]
        OUTPUT_KEYS[name] = KEYS[source]
        save_csv(name, selected.copy())
        for output_row, (source_index, _) in enumerate(selected.iterrows()):
            source_map(name, output_row, list(selected.columns), source, source_index)


    def check(name, condition, detail):
        checks.append({'check': name, 'status': 'PASS' if condition else 'FAIL', 'detail': detail})
        assert condition, f'{name}: {detail}'


    def one_index(source, **keys):
        mask = pd.Series(True, index=data[source].index)
        for key, value in keys.items():
            mask &= data[source][key].isna() if value is None else data[source][key].eq(value)
        indices = data[source].index[mask].tolist()
        assert len(indices) == 1, (source, keys, len(indices))
        return indices[0]


    def md_table(frame, columns, labels=None, formats=None):
        labels, formats = labels or {}, formats or {}
        text = '| ' + ' | '.join(labels.get(column, column) for column in columns) + ' |\n'
        text += '|' + '|'.join(['---'] * len(columns)) + '|\n'
        for _, row in frame.iterrows():
            values = []
            for column in columns:
                value = row[column]
                rendered = '—' if pd.isna(value) else (formats[column](value) if column in formats else str(value))
                values.append(rendered.replace('|', '\\|').replace('\n', ' '))
            text += '| ' + ' | '.join(values) + ' |\n'
        return text


    # 1. 出所と既存の検証記録を読む。旧スクリプトのimportや分析の実行はしない。
    input_paths = {name: workspace_dir / path for name, path in {**CSV_PATHS, **JSON_PATHS}.items()}
    input_paths.update({'methods_draft': report_dir / 'methods_limitations_ja.md', 'results_draft': report_dir / 'results_discussion_ja.md',
                        'design_source': workspace_dir / 'docs/final_research_design_v1_source.md', 'requirements': workspace_dir / 'requirements.txt'})
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    protected_paths = set(input_paths.values())
    for directory in RECORD_DIRS + ['data/raw', 'data/processed', 'models/ols', 'models/fixed_conformal',
            'figures/eda', 'figures/ols', 'figures/fixed_conformal', 'figures/rolling_conformal', 'figures/evaluation',
            'figures/bootstrap', 'figures/bootstrap_sensitivity', 'figures/vix_threshold_sensitivity', 'figures/rolling_window_sensitivity']:
        protected_paths.update(path for path in (workspace_dir / directory).rglob('*') if path.is_file())
    protected_hashes = {path.relative_to(workspace_dir).as_posix(): sha(path) for path in sorted(protected_paths)}
    tables, provenance, OUTPUT_KEYS, checks = {}, [], {}, []
    data = {}
    for name, path in CSV_PATHS.items():
        dtype = {'rolling_window': 'Int64'} if name.startswith('window_') else None
        data[name] = pd.read_csv(workspace_dir / path, float_precision='round_trip', dtype=dtype)
        check(f'unique keys: {name}', not data[name].duplicated(KEYS[name]).any(), ', '.join(KEYS[name]))
    documents = {name: json.loads((workspace_dir / path).read_text(encoding='utf-8')) for name, path in JSON_PATHS.items()}
    design, model = documents['design'], documents['model']
    assert sha(input_paths['design_source']) == design['source_extracted_text_sha256']
    assert design['rolling_window'] == 252 and design['block_length'] == 10 and design['seed'] == 202701
    assert design['bootstrap_repetitions'] == 5000 and design['nominal_coverage'] == .90
    assert design['primary_hypothesis'] == 'two-sided high-VIX coverage difference'
    assert design['sensitivity_block_lengths'] == [5, 10, 20] and design['sensitivity_vix_thresholds'] == [25, 30, 35] and design['sensitivity_rolling_windows'] == [63, 126, 252]

    # 2. 同じ主設定の再掲をexact照合。異なる演算経路の平均は既存の数値許容差で照合する。
    for split, n, high in [('training', 502, 5), ('calibration', 252, 0), ('test', 1658, 149)]:
        row = data['split'].loc[one_index('split', split=split)]
        other = data['eda_split'].loc[one_index('eda_split', split=split)]
        check(f'sample {split}', row['n'] == other['n'] == n and row['high_vix_n'] == other['high_vix_n'] == high, 'split_summary and saved EDA counts')
    check('OLS fit JSON agreement', model['fit_statistics'] == documents['ols_fit'], 'fixed_coefficients.fit_statistics == model_fit_statistics.json')
    for coefficient in ['const', 'return_lag1_pct']:
        value = data['coefficients'].loc[one_index('coefficients', coefficient=coefficient), 'estimate']
        check(f'OLS coefficient {coefficient}', value == model['coefficients'][coefficient], 'CSV estimate and full-precision JSON coefficient')
    for source, key, value in [('block_difference', 'block_length', 10), ('vix_difference', 'vix_threshold', 30), ('window_difference', 'rolling_window', 252)]:
        selected = data[source].loc[data[source][key].eq(value), data['main_difference'].columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, data['main_difference'], check_exact=True)
        check(f'main difference reused: {source}', True, f'{key}={value}; all original columns exactly equal')
    for source, key, value in [('block_coverage', 'block_length', 10), ('vix_coverage', 'vix_threshold', 30)]:
        selected = data[source].loc[data[source][key].eq(value), data['main_coverage'].columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, data['main_coverage'], check_exact=True)
        check(f'main coverage reused: {source}', True, 'all original columns exactly equal')
    for method in ['fixed', 'rolling']:
        mask = data['window_coverage']['method'].eq(method) & (data['window_coverage']['rolling_window'].isna() if method == 'fixed' else data['window_coverage']['rolling_window'].eq(252))
        expected = data['main_coverage'].loc[data['main_coverage']['method'].eq(method)].reset_index(drop=True)
        pd.testing.assert_frame_equal(data['window_coverage'].loc[mask, expected.columns].reset_index(drop=True), expected, check_exact=True)
        check(f'window coverage primary/common {method}', True, 'Fixed is one common benchmark with no rolling_window value')
    for source, key, value in [('vix_performance', 'vix_threshold', 30), ('window_performance', 'rolling_window', 252)]:
        for method in ['fixed', 'rolling']:
            mask = data[source]['method'].eq(method)
            mask &= data[source][key].isna() if source == 'window_performance' and method == 'fixed' else data[source][key].eq(value)
            expected = data['main_performance'].loc[data['main_performance']['method'].eq(method)].reset_index(drop=True)
            pd.testing.assert_frame_equal(data[source].loc[mask, expected.columns].reset_index(drop=True), expected, check_exact=True)
        check(f'main performance reused: {source}', True, 'all original columns exactly equal; Fixed common in window table')
    for source in ['main_difference', 'block_difference', 'vix_difference', 'window_difference']:
        frame = data[source]
        coverage = frame.loc[frame['metric'].eq('coverage_difference')]
        np.testing.assert_allclose(coverage['display_point_estimate'], 100 * coverage['point_estimate'], rtol=RTOL, atol=ATOL)
        check(f'coverage pp units: {source}', coverage['unit'].eq('proportion').all() and coverage['display_unit'].eq('coverage percentage points').all(), 'raw proportion -> display *100 only')
        high = frame.loc[frame['group'].eq('high_vix')]
        check(f'high-VIX comparison directions: {source}', high.loc[high['metric'].isin(['coverage_difference', 'mean_width_difference']), 'ci_lower'].gt(0).all()
              and high.loc[high['metric'].eq('mean_interval_score_difference'), 'ci_upper'].lt(0).all(), 'saved CI directions; no new statistical test')
    for source in ['main_coverage', 'block_coverage', 'vix_coverage', 'window_coverage']:
        frame = data[source]
        np.testing.assert_allclose(frame['signed_gap_ci_lower_pp'], 100 * (frame['coverage_ci_lower'] - .90), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(frame['signed_gap_ci_upper_pp'], 100 * (frame['coverage_ci_upper'] - .90), rtol=RTOL, atol=ATOL)
        check(f'Gap units and high-VIX shortfall: {source}', frame.loc[frame['group'].eq('high_vix'), 'coverage_ci_upper'].lt(.90).all(), 'Gap is coverage pp; all high-VIX CI upper endpoints below .90')
    for frame in data.values():
        if {'source_file', 'source_sha256'}.issubset(frame.columns):
            for row in frame[['source_file', 'source_sha256']].drop_duplicates().itertuples(index=False):
                check(f'upstream hash {row.source_file}', sha(workspace_dir / row.source_file) == row.source_sha256, 'saved source reference equals current file SHA-256')
    for group in ['overall', 'high_vix', 'non_high_vix']:
        pair = data['main_paired'].loc[one_index('main_paired', group=group)]
        for metric, column in [('coverage_difference', 'coverage_difference_rolling_minus_fixed'),
                               ('mean_width_difference', 'mean_width_difference_rolling_minus_fixed_return_pp'),
                               ('mean_interval_score_difference', 'mean_interval_score_difference_rolling_minus_fixed_return_pp')]:
            row = data['main_difference'].loc[one_index('main_difference', group=group, metric=metric)]
            np.testing.assert_allclose(pair[column], row['point_estimate'], rtol=RTOL, atol=ATOL)
    check('Overall IS sensitivity l20', bool(data['block_difference'].loc[one_index('block_difference', block_length=20, group='overall', metric='mean_interval_score_difference'), 'ci_includes_zero']), 'CI includes zero; sensitivity retained in narrative')
    check('VIX35 concentration', data['vix_composition'].loc[one_index('vix_composition', vix_threshold=35), 'high_vix_n'] == 59
          and data['vix_yearly'].loc[one_index('vix_yearly', vix_threshold=35, year=2020), 'high_vix_n'] == 50, '50 of 59 observations in 2020')
    for threshold, expected in [(25, False), (30, True), (35, True)]:
        flag = data['vix_difference'].loc[one_index('vix_difference', vix_threshold=threshold, group='non_high_vix', metric='mean_interval_score_difference'), 'ci_includes_zero']
        check(f'non-high-VIX IS sensitivity {threshold}', bool(flag) == expected, '25 positive CI; 30/35 include zero')


    # 3. 本文表A〜Eと付録表。計算用CSVの数値を丸めない。
    name = 'table_a1_sample_design.csv'
    a1 = data['split'][['split', 'first_date', 'last_date', 'n', 'high_vix_n', 'vix_below_30_n']].copy()
    a1['role'] = ['Training-only OLS estimation', '2019 out-of-sample calibration scores', 'common-date Fixed/Rolling evaluation']
    OUTPUT_KEYS[name] = ['split']
    save_csv(name, a1)
    for i in a1.index:
        source_map(name, i, list(a1.columns[:-1]), 'split', i)
    copy_table('table_a2_variable_definitions.csv', 'variables', data['variables']['variable'].isin(['date', 'forecast_origin_date', 'sp500_close', 'return_pct', 'return_lag1_pct', 'vix_lag1', 'high_vix_30']))
    name = 'table_a3_fixed_settings.csv'
    settings_keys = ['training', 'calibration', 'test', 'point_model', 'nominal_coverage', 'score', 'fixed_calibration', 'rolling_window',
                     'rolling_update', 'quantile_rule', 'interval_endpoints', 'high_vix_definition', 'bootstrap_repetitions', 'block_length', 'seed',
                     'confidence_interval', 'primary_hypothesis', 'sensitivity_block_lengths', 'sensitivity_vix_thresholds', 'sensitivity_rolling_windows']
    OUTPUT_KEYS[name] = ['setting']
    save_csv(name, pd.DataFrame([{'setting': key, 'value_json': json.dumps(design[key], ensure_ascii=False)} for key in settings_keys]))
    for i, key in enumerate(settings_keys):
        source_map(name, i, ['value_json'], 'design', fields=[key], note='JSON value serialized without changing its definition')
    name = 'table_b1_ols_coefficients.csv'
    b1 = data['coefficients'][['coefficient', 'estimate']].copy()
    b1['unit'] = [model['units']['const'], model['units']['return_lag1_pct_coefficient']]
    OUTPUT_KEYS[name] = ['coefficient']
    save_csv(name, b1)
    for i, row in b1.iterrows():
        source_map(name, i, ['estimate'], 'coefficients', i)
        source_map(name, i, ['unit'], 'model', fields=['units.const' if row['coefficient'] == 'const' else 'units.return_lag1_pct_coefficient'])
    name = 'table_b2_ols_fit.csv'
    OUTPUT_KEYS[name] = ['nobs']
    save_csv(name, pd.DataFrame([documents['ols_fit']]))
    source_map(name, 0, list(documents['ols_fit']), 'ols_fit', fields=list(documents['ols_fit']))
    name = 'table_c_main_performance.csv'
    point_columns = ['group', 'method', 'n', 'covered_n', 'empirical_coverage', 'coverage_gap_pp', 'absolute_coverage_gap_pp', 'mean_width_return_pp', 'mean_interval_score_return_pp']
    ci_columns = ['coverage_ci_lower', 'coverage_ci_upper', 'signed_gap_ci_lower_pp', 'signed_gap_ci_upper_pp', 'confidence_level', 'valid_replicates', 'invalid_replicates']
    c = data['main_performance'][point_columns].merge(data['main_coverage'][['group', 'method', 'n_original', 'coverage_point_estimate'] + ci_columns],
                                                    on=['group', 'method'], how='outer', validate='one_to_one', indicator=True)
    check('main performance/CI one-to-one', len(c) == 6 and c['_merge'].eq('both').all() and c['n'].eq(c['n_original']).all(), 'same group, method, dates and denominator')
    np.testing.assert_array_equal(c['empirical_coverage'], c['coverage_point_estimate'])
    c = c.drop(columns=['_merge', 'n_original', 'coverage_point_estimate'])
    OUTPUT_KEYS[name] = ['group', 'method']
    save_csv(name, c)
    for i, row in c.iterrows():
        source_map(name, i, point_columns, 'main_performance', one_index('main_performance', group=row['group'], method=row['method']))
        source_map(name, i, ci_columns, 'main_coverage', one_index('main_coverage', group=row['group'], method=row['method']))
    copy_table('table_d_main_paired_differences.csv', 'main_difference')
    for prefix, source in [('e1_block', 'block'), ('e2_vix', 'vix'), ('e3_window', 'window')]:
        copy_table(f'table_{prefix}_high_differences.csv', f'{source}_difference', data[f'{source}_difference']['group'].eq('high_vix'))
        copy_table(f'table_{prefix}_high_coverage_gap.csv', f'{source}_coverage', data[f'{source}_coverage']['group'].eq('high_vix'))
        copy_table(f'appendix_{prefix}_all_differences.csv', f'{source}_difference')
        copy_table(f'appendix_{prefix}_all_coverage_gap.csv', f'{source}_coverage')
    copy_table('appendix_vix_all_performance.csv', 'vix_performance')
    copy_table('appendix_window_all_performance.csv', 'window_performance')
    copy_table('appendix_vix_composition.csv', 'vix_composition')
    copy_table('appendix_vix_yearly_composition.csv', 'vix_yearly')
    copy_table('appendix_window_initial_histories.csv', 'window_initial')
    copy_table('appendix_bootstrap_group_counts.csv', 'bootstrap_counts')

    # E-1 pandas平均とE-2 NumPy平均の差は丸め順序による微小差。どちらも元値を保持する。
    mean_delta = 0.
    for group in ['overall', 'high_vix', 'non_high_vix']:
        original = data['main_original'].loc[one_index('main_original', group=group)]
        for method in ['fixed', 'rolling']:
            point = data['main_performance'].loc[one_index('main_performance', group=group, method=method)]
            for base, saved_column in [('mean_width', 'mean_width_return_pp'), ('mean_interval_score', 'mean_interval_score_return_pp')]:
                delta = abs(float(original[f'{base}_{method}_return_pp']) - float(point[saved_column]))
                mean_delta = max(mean_delta, delta)
                np.testing.assert_allclose(original[f'{base}_{method}_return_pp'], point[saved_column], rtol=RTOL, atol=ATOL)
    (report_dir / 'numeric_notes.md').write_text(f'''# 数値・単位の照合メモ

主性能の点推定はE-1の保存比較表、Coverage/Gap CIとpaired差CIはE-2の保存表を根拠とする。
同一主設定の感度表への再掲は全元列exact一致。演算経路が異なる平均値の照合だけはrtol=atol=1e-12を用いる。
E-1 pandas平均とE-2 NumPy平均の最大絶対差は{mean_delta:.17g}で、浮動小数点の集計順序の範囲だった。
これを都合のよい値へ置換せず、各表で採用した元値・出所を保持した。未解決の数値不一致は見つからなかった。
CoverageはCSVでproportion、本文で100倍した%。Gap/coverage差はcoverage pp、Width/ISとその差はreturn pp。
手法別Width/ISのCIは既存CI表にないため新たに計算・掲載していない。paired差のCIとは区別する。
掲載CIは個別の95% percentile CIで、多重比較調整済み・同時信頼区間ではない。
設計lockのcompleted_scopeは固定時点の履歴であり、現在の進捗へ書き換えていない。
''', encoding='utf-8')

    # 4. 既存の検証・実行記録を棚卸しする。記録を読むことと再検証の実行は区別する。
    record_rows = []
    for directory in RECORD_DIRS:
        for path in sorted((workspace_dir / directory).rglob('*')):
            if path.is_file() and (path.name in ['validation_results.json', 'artifact_validation.json', 'execution_metadata.json', 'figure_visual_qa.md']):
                content = path.read_text(encoding='utf-8')
                status = json.loads(content).get('status', 'metadata record') if path.suffix == '.json' else 'existing visual QA record read'
                record_rows.append({'file': path.relative_to(workspace_dir).as_posix(), 'sha256': sha(path), 'record_status': status,
                                    'review_scope': 'read saved record; upstream stage not rerun'})
    for relative in CORE_VALIDATIONS:
        check(f'existing validation {relative}', json.loads((workspace_dir / relative).read_text(encoding='utf-8'))['status'] == 'PASS', 'saved status only; not a rerun')

    # 5. 図は既存値の体裁調整または無変更コピー。CI端点を独立に描き、対称CIとは仮定しない。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    figure_rows = []


    def register_figure(identifier, stem, title, caption, origin, sources):
        for extension in ['png', 'pdf']:
            path = figure_dir / f'{stem}.{extension}'
            check(f'figure exists {identifier} {extension}', path.is_file() and path.stat().st_size > 0, 'separate final copy; original figure unchanged')
        figure_rows.append({'figure_id': identifier, 'placement': stem.split('/')[0], 'title': title,
            'png': (figure_dir / f'{stem}.png').relative_to(workspace_dir).as_posix(), 'pdf': (figure_dir / f'{stem}.pdf').relative_to(workspace_dir).as_posix(),
            'png_sha256': sha(figure_dir / f'{stem}.png'), 'pdf_sha256': sha(figure_dir / f'{stem}.pdf'),
            'origin': origin, 'source_files_and_sha256': json.dumps({p: sha(workspace_dir / p) for p in sources}, ensure_ascii=False), 'caption_ja': caption})


    def copy_figure(identifier, source_stem, target_stem, title, caption, data_sources):
        for extension in ['png', 'pdf']:
            source, destination = workspace_dir / f'{source_stem}.{extension}', figure_dir / f'{target_stem}.{extension}'
            shutil.copy2(source, destination)
            check(f'figure copy exact {identifier} {extension}', sha(source) == sha(destination), 'byte-for-byte copy')
        register_figure(identifier, target_stem, title, caption, f'unchanged copy: {source_stem}',
                        [f'{source_stem}.png', f'{source_stem}.pdf'] + data_sources)


    def save_figure(fig, stem):
        fig.savefig(figure_dir / f'{stem}.png', dpi=300, bbox_inches='tight')
        fig.savefig(figure_dir / f'{stem}.pdf', bbox_inches='tight')
        plt.close(fig)


    copy_figure('F01', 'figures/eda/01_returns_and_lagged_vix_timeseries', 'main/F01_returns_and_lagged_vix',
        'Returns, lagged VIX and chronological splits',
        '図F01。S&P 500百分率対数リターン（上、return pp）と前S&P 500市場日のVIX（下、VIX index points）。'
        'Training502日、Calibration252日、Test1,658日を日付軸と境界線で示す。VIX30線とHigh-VIX表示を含む。'
        '主区分はvix_lag1>=30で、Test149日。記述的時系列図のためCIはない。2026年は8月7日まで。',
        ['data/processed/analysis_dataset.csv', CSV_PATHS['split']])

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    groups = [('overall', 'Overall', 1658), ('high_vix', 'High-VIX', 149), ('non_high_vix', 'Non-high-VIX', 1509)]
    for method, offset, color, label in [('fixed', -.12, '#c05b38', 'Fixed'), ('rolling', .12, '#2475a5', 'Rolling-252')]:
        selected = data['main_coverage'].set_index(['group', 'method']).loc[[(g, method) for g, _, _ in groups]]
        x = np.arange(3) + offset
        lower, upper = selected['coverage_ci_lower'].to_numpy() * 100, selected['coverage_ci_upper'].to_numpy() * 100
        point = selected['coverage_point_estimate'].to_numpy() * 100
        ax.vlines(x, lower, upper, color=color, linewidth=2)
        ax.hlines(lower, x-.035, x+.035, color=color)
        ax.hlines(upper, x-.035, x+.035, color=color)
        ax.scatter(x, point, color=color, label=label, s=42, zorder=3)
        for xpos, value, bottom in zip(x, point, lower):
            ax.annotate(f'{value:.2f}%', (xpos, bottom), xytext=(0, -15 if method == 'fixed' else -27), textcoords='offset points', ha='center', color=color, fontsize=9)
    ax.axhline(90, color='#555555', linestyle='--', linewidth=1.2, label='Nominal coverage: 90%')
    ax.set_xticks(range(3), [f'{label}\n(n={n:,})' for _, label, n in groups])
    ax.set_ylim(0, 103)
    ax.set_ylabel('Empirical coverage (%)')
    ax.set_title('Main setting: coverage with 95% paired-MBB percentile CIs')
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='lower left', frameon=False, ncol=3)
    fig.tight_layout()
    save_figure(fig, 'main/F02_coverage_by_regime')
    register_figure('F02', 'main/F02_coverage_by_regime', 'Main coverage by market regime',
        '図F02。主設定（固定OLS、window252、前市場日VIX30、block10、B=5,000、seed202701）。'
        '点は手法別Coverage（%）、縦線は保存済み非循環Paired MBBの95% percentile CI（linear）。'
        'Overall n=1,658、High-VIX n=149、Non-high-VIX n=1,509。破線は90%。'
        '保存済みモデル・区間・評価系列に条件づけた近似CIであり、将来のCoverage保証ではない。',
        'reformatted saved estimates and CI endpoints only', [CSV_PATHS['main_coverage']])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))
    metric_specs = [('coverage_difference', 'Coverage difference\n(primary; two-sided)', 'Coverage percentage points'),
        ('mean_width_difference', 'Mean Width difference\n(secondary)', 'Return percentage points'),
        ('mean_interval_score_difference', 'Mean IS difference\n(secondary)', 'Return percentage points')]
    for ax, (metric, title, unit) in zip(axes, metric_specs):
        row = data['main_difference'].loc[one_index('main_difference', group='high_vix', metric=metric)]
        lower, upper, point = [float(row[column]) for column in ['display_ci_lower', 'display_ci_upper', 'display_point_estimate']]
        ax.vlines(0, lower, upper, color='#2475a5', linewidth=2)
        ax.hlines([lower, upper], -.06, .06, color='#2475a5')
        ax.scatter([0], [point], color='#2475a5', s=48, zorder=3)
        ax.axhline(0, color='#666666', linestyle='--', linewidth=1)
        low, high = min(lower, 0), max(upper, 0)
        margin = (high-low) * .18
        ax.set_ylim(low-margin, high+margin)
        ax.set_xlim(-.5, .5)
        ax.set_xticks([0], ['Rolling-252 minus Fixed'])
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.text(.5, .95, f'{point:+.2f} [{lower:+.2f}, {upper:+.2f}]', transform=ax.transAxes, ha='center', va='top', fontsize=9)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Main High-VIX comparison (n=149): estimates and 95% CIs', y=1.03)
    fig.tight_layout()
    save_figure(fig, 'main/F03_high_vix_paired_differences')
    register_figure('F03', 'main/F03_high_vix_paired_differences', 'Main High-VIX paired differences',
        '図F03。主High-VIX149日のRolling-252−Fixed。左からCoverage差（coverage pp、主・両側仮説）、'
        '平均Width差、平均IS差（いずれもreturn pp、副次）。点は保存推定値、縦線は95% percentile CI（非循環Paired MBB、'
        'block10、B=5,000、seed202701、linear）。横破線は0。固定OLS・前市場日VIX30。'
        '手法別Width/ISのCIではなく、同日paired差のCIである。',
        'reformatted saved estimates and CI endpoints only', [CSV_PATHS['main_difference']])

    copy_figure('A01', 'figures/bootstrap_sensitivity/01_high_vix_difference_ci_by_block_length', 'appendix/A01_block_length_sensitivity',
        'Block length sensitivity',
        '図A01。High-VIX149日のRolling-252−Fixed。block length5/10/20だけを変更、10が主設定。'
        'Coverage差はcoverage pp、平均Width/IS差はreturn pp。点と95% percentile CI（B=5,000、seed202701、linear）、0線を示す。'
        '固定OLS、VIX30、window252共通。主10は同じ保存結果の再掲。', [CSV_PATHS['block_difference']])
    copy_figure('A02', 'figures/vix_threshold_sensitivity/02_high_vix_differences_by_threshold', 'appendix/A02_vix_threshold_sensitivity',
        'Lagged-VIX threshold sensitivity',
        '図A02。前市場日VIX閾値25/30/35のHigh-VIX日（n=348/149/59）でのRolling-252−Fixed。30が主。'
        'Coverage差はcoverage pp、平均Width/IS差はreturn pp。点と95% percentile CI（block10、B=5,000、seed202701、linear）、0線。'
        '固定OLS・window252共通。閾値ごとに評価日が変わり、対象は入れ子で独立ではない。', [CSV_PATHS['vix_difference'], CSV_PATHS['vix_composition']])
    copy_figure('A03', 'figures/rolling_window_sensitivity/03_high_vix_differences_by_window', 'appendix/A03_rolling_window_sensitivity',
        'Rolling window sensitivity',
        '図A03。共通High-VIX149日のRolling63/126/252−共通Fixed。252が主。Coverage差はcoverage pp、'
        '平均Width/IS差はreturn pp。点と95% percentile CI（block10、B=5,000、seed202701、linear）、0線。'
        '固定OLS・前市場日VIX30共通。window間を直接検定した図や最適windowを選ぶ図ではない。', [CSV_PATHS['window_difference']])
    copy_figure('A04', 'figures/rolling_window_sensitivity/01_halfwidth_timeseries_by_window', 'appendix/A04_halfwidth_timeseries',
        'Saved half-width trajectories',
        '図A04。Test1,658日のFixedとRolling63/126/252の保存済み半幅q（return pp）。全幅は2q。'
        '固定OLS・名目90%、252が主、High-VIX定義は前市場日VIX30。記述的時系列のためCIはない。'
        '履歴長以外の主設定は固定。2026年は8月7日まで。',
        ['reports/rolling_window_sensitivity/window_63/test_rolling_intervals.csv',
         'reports/rolling_window_sensitivity/window_126/test_rolling_intervals.csv', 'reports/rolling_conformal/test_rolling_intervals.csv'])

    # 6. 表の全数値セルを元行またはJSON pathへたどれることを確認する。
    for name, frame in tables.items():
        for i, row in frame.iterrows():
            mapped = set()
            for entry in provenance:
                if entry['output_table'] == name and entry['output_row_1_based'] == i + 1:
                    mapped.update(json.loads(entry['output_columns']))
            numeric = [column for column in frame.select_dtypes(include='number').columns if pd.notna(row[column])]
            assert set(numeric).issubset(mapped), (name, i, set(numeric)-mapped)
    check('numeric-cell provenance coverage', True, 'all nonmissing numeric research-table cells map to source keys/JSON paths')
    save_csv('source_cell_mapping.csv', pd.DataFrame(provenance))
    save_csv('source_catalog.csv', pd.DataFrame([{'source_name': name, 'source_file': path.relative_to(workspace_dir).as_posix(),
        'sha256': input_hashes[name], 'csv_row_keys': json.dumps(KEYS.get(name, []), ensure_ascii=False)} for name, path in input_paths.items()]))
    save_csv('existing_record_review.csv', pd.DataFrame(record_rows))
    save_csv('figure_catalog.csv', pd.DataFrame(figure_rows))
    captions = '# 採用図とcaption（確認用）\n\n本文3図、付録4図。新しい推定・CIは計算していない。\n\n'
    for figure in figure_rows:
        captions += f"## {figure['figure_id']}：{figure['title']}\n\n{figure['caption_ja']}\n\n"
        captions += f"![{figure['figure_id']}](../../{figure['png']})\n\n[PDF](../../{figure['pdf']})\n\n"
        captions += f"出所・SHA-256：`figure_catalog.csv`の`figure_id={figure['figure_id']}`。\n\n"
    (report_dir / 'figure_captions_ja.md').write_text(captions, encoding='utf-8')

    # 7. 表の表示だけ丸める。日本語原稿は独立に読んだ2つの下書きを節単位で結合する。
    group_labels = {'overall': 'Overall（補助）', 'high_vix': 'High-VIX（主対象）', 'non_high_vix': 'Non-high-VIX（補助）'}
    method_labels = {'fixed': 'Fixed', 'rolling': 'Rolling-252'}
    methods_text = (report_dir / 'methods_limitations_ja.md').read_text(encoding='utf-8')
    results_text = (report_dir / 'results_discussion_ja.md').read_text(encoding='utf-8')
    methods = methods_text.split('## Methods（方法）\n', 1)[1].split('## Limitations（限界）\n', 1)[0]
    limitations = methods_text.split('## Limitations（限界）\n', 1)[1].split('## 本人・指導教員の確認事項\n', 1)[0]
    results = results_text.split('## Results\n', 1)[1].split('## Discussion\n', 1)[0]
    discussion = results_text.split('## Discussion\n', 1)[1].split('## 数値の根拠と行キー\n', 1)[0]
    report = '''# 卒論予備分析：主分析と感度分析の統合（日本語確認用下書き）

本人・指導教員が確認するための下書きであり、提出用完成稿ではない。
保存済みCSV・JSONだけを数値の根拠とした。Step Fでは表の統合・図の整理・説明文と出所の確認を行い、
OLS・score・予測区間・日次評価・bootstrap・追加検定を実行していない。主分析条件は変更していない。

暫定的な答え：この歴史的標本の前市場日VIX≥30の日には、Rolling-252はFixedより高いCoverageを示した。
改善には幅の増加を伴い、平均ISは低下した。ただしHigh-VIXのCoverageはなお90%未満である。
単位はCoverageが%、Gap/coverage差がcoverage pp、Width/ISとその差がreturn pp。CIは個別の95% percentile CIである。
公開事前登録は未確認であり、事前に固定した設計と区別する。

'''
    report += '## Methods（方法）\n\n' + methods
    report += '## Results（結果）\n\n' + results
    report += '## Discussion（考察）\n\n' + discussion
    report += '## Limitations（限界）\n\n' + limitations
    report += '## 統合表A：データ・変数・設計\n\n'
    report += md_table(tables['table_a1_sample_design.csv'], ['split', 'first_date', 'last_date', 'n', 'high_vix_n', 'vix_below_30_n', 'role'])
    report += '\n変数・単位は[表A2](table_a2_variable_definitions.csv)、固定設計の全設定は[表A3](table_a3_fixed_settings.csv)。\n\n'
    report += '## 統合表B：固定OLS baseline\n\n'
    report += md_table(b1, ['coefficient', 'estimate', 'unit'], formats={'estimate': lambda x: f'{x:.8f}'})
    report += '\n' + md_table(tables['table_b2_ols_fit.csv'], ['nobs', 'r_squared', 'adjusted_r_squared', 'residual_standard_error'],
        formats={'nobs': lambda x: f'{int(x)}', 'r_squared': lambda x: f'{x:.8f}', 'adjusted_r_squared': lambda x: f'{x:.8f}', 'residual_standard_error': lambda x: f'{x:.8f}'})
    report += '\nTrainingの二乗残差和を最小化した係数を固定し、FixedとRollingに共通の中心を用意する。R²はTraining内の説明率である。\n\n'
    report += '## 統合表C：主設定の性能\n\n'
    display = c.copy()
    display['group'] = display['group'].map(group_labels)
    display['method'] = display['method'].map(method_labels)
    display['Coverage % [95%CI]'] = c.apply(lambda r: f"{100*r.empirical_coverage:.2f} [{100*r.coverage_ci_lower:.2f}, {100*r.coverage_ci_upper:.2f}]", axis=1)
    report += md_table(display, ['group', 'method', 'n', 'Coverage % [95%CI]', 'coverage_gap_pp', 'mean_width_return_pp', 'mean_interval_score_return_pp'],
        labels={'coverage_gap_pp': 'signed Gap (coverage pp)', 'mean_width_return_pp': 'Mean Width (return pp)', 'mean_interval_score_return_pp': 'Mean IS (return pp)'},
        formats={'coverage_gap_pp': lambda x: f'{x:+.2f}', 'mean_width_return_pp': lambda x: f'{x:.3f}', 'mean_interval_score_return_pp': lambda x: f'{x:.3f}'})
    report += '\nCoverage CIはE-2、点推定はE-1。個別手法のWidth/IS CIは既存表にないため掲載しない。\n\n'
    report += '## 統合表D：主設定のRolling−Fixed\n\n'


    def difference_display(frame, setting=None):
        view = frame.copy()
        view['estimate [95%CI]'] = frame.apply(lambda r: f"{r.display_point_estimate:+.3f} [{r.display_ci_lower:+.3f}, {r.display_ci_upper:+.3f}]", axis=1)
        columns = ([setting] if setting else []) + ['group', 'metric', 'display_unit', 'estimate [95%CI]', 'analysis_role']
        return md_table(view, columns)


    report += difference_display(data['main_difference'])
    report += '\n## 統合表E：一要因感度分析のHigh-VIX結果\n\n主設定はblock10／VIX30／window252であり、同じ主設定の行は独立した実験ではない。\n\n'
    for prefix, source, setting, label in [('e1_block', 'block', 'block_length', 'E1：block length'), ('e2_vix', 'vix', 'vix_threshold', 'E2：前市場日VIX閾値'), ('e3_window', 'window', 'rolling_window', 'E3：Rolling window')]:
        report += f'### {label}\n\n' + difference_display(tables[f'table_{prefix}_high_differences.csv'], setting)
        report += f'\n手法別Coverage/Gap CI：[対応CSV](table_{prefix}_high_coverage_gap.csv)。全区分：[paired差](appendix_{prefix}_all_differences.csv)、[Coverage/Gap](appendix_{prefix}_all_coverage_gap.csv)。\n\n'
    report += '## 採用した本文図\n\n'
    for figure in figure_rows[:3]:
        report += f"![{figure['figure_id']}](../../{figure['png']})\n\n{figure['caption_ja']}\n\n"
    report += '''## 付録・出所・確認事項

- [採用7図と全caption](figure_captions_ja.md)、[図の出所・ハッシュ](figure_catalog.csv)。付録は3感度分析と半幅時系列を選び、本文との重複を抑えた。
- [セル群単位の出所・行キー・SHA-256](source_cell_mapping.csv)、[入力一覧](source_catalog.csv)。計算用CSVは丸めず、表の表示だけ丸める。
- [数値と単位の照合メモ](numeric_notes.md)、[既存検証・実行・図確認記録の確認一覧](existing_record_review.csv)、[今回の整合性検証](validation_results.json)。
- [再出力・再現手順](reproduction_guide_ja.md)、[未解決文献と本人・指導教員の確認事項](review_and_reference_items_ja.md)、[今回の実図確認](figure_visual_qa.md)。

技術的な数値不一致が解消していることと、文献・研究上の限界が解消していることは別である。
要文献確認箇所とcalendar/vintage等の未実施事項を保持し、結果に合わせた追加分析は行わない。

'''
    report += '## 文章中の数値の出所\n\n' + results_text.split('## 数値の根拠と行キー\n', 1)[1]
    report += '\n## Methodsの出所・実装対応\n\n' + methods_text.split('## 確認した出所と数値位置\n', 1)[1]
    (report_dir / 'preliminary_report_ja.md').write_text(report, encoding='utf-8')

    # 8. 入力保全・実行メタデータ。目視確認は実際に開いた後、別の記録へ記入する。
    changed = [relative for relative, digest in protected_hashes.items() if sha(workspace_dir / relative) != digest]
    check('protected source files unchanged', not changed, f'{len(protected_hashes)} files; differences={changed}')
    save_csv('protected_input_hashes.csv', pd.DataFrame([{'file': path, 'sha256_before': digest, 'sha256_after': sha(workspace_dir / path), 'unchanged': True} for path, digest in protected_hashes.items()]))
    validation = {'status': 'PASS', 'scope': 'saved-output integration only; no upstream analysis rerun', 'checks': checks,
        'all_saved_tables_roundtrip_exact': True, 'source_mapping_rows': len(provenance), 'protected_file_count': len(protected_hashes),
        'unresolved_numeric_discrepancies': [], 'floating_point_mean_max_abs_difference': mean_delta,
        'visual_qa': 'see figure_visual_qa.md after the final images have actually been opened'}
    (report_dir / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'python_executable': sys.executable, 'python_version': sys.version,
        'packages': {'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'script_sha256': sha(Path(__file__)), 'scope': 'compile saved CSV/JSON, copy/reformat figures, integrate Japanese review drafts',
        'not_executed': ['Steps A-E and sensitivity scripts', 'OLS fitting', 'score or interval generation', 'bootstrap', 'new statistical tests'],
        'input_files': {name: {'path': path.relative_to(workspace_dir).as_posix(), 'sha256': input_hashes[name]} for name, path in input_paths.items()},
        'output_csv_sha256': {name: sha(report_dir / name) for name in tables},
        'output_markdown_sha256': {path.name: sha(path) for path in report_dir.glob('*.md') if path.name != 'figure_visual_qa.md'},
        'figure_count': len(figure_rows), 'main_settings_unchanged': True}
    (report_dir / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Step F PASS: {len(tables)} CSVs, {len(figure_rows)} PNG/PDF figure pairs, {len(provenance)} source mappings.')
    print(f'Protected files unchanged: {len(protected_hashes)}; prior stages not executed.')
    print('Visual QA must be recorded after opening all final PNGs; this script does not claim human visual inspection.')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 19.')
    compile_preliminary_results(ANALYSIS_WORKSPACE)
