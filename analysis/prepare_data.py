"""保存済みFRED snapshotから加工データ・監査表を作る。実行はrun_analysis.pyから行う。
旧01〜09の順序と検証を保持する。期間を切る前に市場日リターンとラグを作る。
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import urlopen
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd


def read_fred(raw_dir, series_id, value_name):
    raw = pd.read_csv(raw_dir / f'{series_id}_fred.csv', dtype=str, keep_default_na=False)
    assert list(raw.columns) == ['observation_date', series_id]
    values = raw[series_id].str.strip()
    numeric_values = pd.to_numeric(values, errors='coerce')
    unexpected = numeric_values.isna() & ~values.isin(['', '.', 'NA', 'NaN', 'nan'])
    assert not unexpected.any(), f'{series_id}: 未知の値がある'
    result = pd.DataFrame({
        'date': pd.to_datetime(raw['observation_date'], format='%Y-%m-%d', errors='raise'),
        value_name: numeric_values,
    })
    assert result['date'].duplicated().sum() == 0
    assert result['date'].notna().all()
    result = result.sort_values('date').reset_index(drop=True)
    valid = result[value_name].dropna()
    assert np.isfinite(valid).all() and valid.gt(0).all()
    assert result['date'].dt.dayofweek.lt(5).all()
    return result


def prepare_data(workspace):
    """指定した独立workspaceのコピーを前処理し、同じworkspaceへ保存する。"""
    # 1. 設定と保存先
    # 役割: 共通import、保存場所、固定条件を用意する（実行順1）。
    # 前処理の実行: run_analysis.py run --workspace <copy> --stage prepare

    # 通常のPythonでも表を表示できるようにする。IPythonは不要。
    display = print

    # CLIで指定した独立workspaceを使う。原本や実行時の作業場所には出力しない。
    PROJECT_DIR = Path(workspace).resolve()
    assert (PROJECT_DIR / 'docs' / 'final_research_design_v1_source.md').exists()

    RAW_DIR = PROJECT_DIR / 'data' / 'raw'
    PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
    REPORT_DIR = PROJECT_DIR / 'reports'
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ANALYSIS_START = '2017-01-01'
    ANALYSIS_END = '2026-08-07'
    DOWNLOAD_START = '2016-12-01'
    ROLLING_WINDOW = 252
    VIX_THRESHOLD = 30
    REFRESH_RAW = False

    design = json.loads((PROJECT_DIR / 'docs' / 'research_design_lock.json').read_text(encoding='utf-8'))
    assert design['training'] == [ANALYSIS_START, '2018-12-31']
    assert design['calibration'] == ['2019-01-01', '2019-12-31']
    assert design['test'] == ['2020-01-01', ANALYSIS_END]
    assert design['download_warmup_start'] == DOWNLOAD_START
    assert design['rolling_window'] == ROLLING_WINDOW
    assert design['high_vix_definition'] == f'previous_SP500_market_day_VIXCLS >= {VIX_THRESHOLD}'

    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 140)

    # 2. 保存済みsnapshotの照合
    # 役割: 保存済み生CSVと取得記録のハッシュを照合する（実行順2）。
    # 入力: 01のRAW_DIR、期間、REFRESH_RAW、共通import。出力: manifest。
    manifest_path = RAW_DIR / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []

    for series_id in ['SP500', 'VIXCLS']:
        raw_path = RAW_DIR / f'{series_id}_fred.csv'
        if not REFRESH_RAW and not raw_path.exists():
            raise FileNotFoundError(f'保存済み生CSVがありません: {raw_path}。研究用snapshotを確認してください。')
        # 通常の再実行ではネット接続しない。明示的にTrueにした場合だけ取得する。
        if REFRESH_RAW:
            url = (f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}'
                   f'&cosd={DOWNLOAD_START}&coed={ANALYSIS_END}')
            with urlopen(url, timeout=30) as response:
                raw_bytes = response.read()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            if raw_path.exists():
                assert digest == hashlib.sha256(raw_path.read_bytes()).hexdigest(), (
                    '取得値がsnapshotと異なる。上書きせず、改訂内容を確認する。'
                )
            else:
                raw_path.write_bytes(raw_bytes)
                manifest.append({
                    'series_id': series_id, 'filename': raw_path.name, 'url': url,
                    'series_page': f'https://fred.stlouisfed.org/series/{series_id}',
                    'requested_start': DOWNLOAD_START, 'requested_end': ANALYSIS_END,
                    'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                    'bytes': len(raw_bytes), 'sha256': digest,
                    'acquisition': 'Python urllib.request',
                })
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    for record in manifest:
        raw_bytes = (RAW_DIR / record['filename']).read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == record['sha256']
        assert len(raw_bytes) == record['bytes']

    assert {record['series_id'] for record in manifest} == {'SP500', 'VIXCLS'}
    assert len(manifest) == 2
    for record in manifest:
        assert record['filename'] == f"{record['series_id']}_fred.csv"
        assert record['requested_start'] == DOWNLOAD_START
        assert record['requested_end'] == ANALYSIS_END
    display(pd.DataFrame(manifest)[['series_id', 'requested_start', 'requested_end',
                                    'retrieved_at_utc', 'bytes', 'sha256']])

    # 3. CSVの読込と品質検査
    # 役割: CSVを読み、型・欠損・重複・日付範囲を検査する（実行順3）。
    # 入力: 01の共通設定、02で検証済みのCSV。出力: sp500、vix、raw_audit。

    sp500 = read_fred(RAW_DIR, 'SP500', 'sp500_close')
    vix = read_fred(RAW_DIR, 'VIXCLS', 'vix_close')

    raw_audit_rows = []
    for series_id, frame, column in [('SP500', sp500, 'sp500_close'), ('VIXCLS', vix, 'vix_close')]:
        expected_weekdays = pd.bdate_range(DOWNLOAD_START, ANALYSIS_END)
        absent_weekdays = expected_weekdays.difference(frame['date'])
        raw_audit_rows.append({
            'series_id': series_id, 'rows': len(frame),
            'first_date': frame['date'].min(), 'last_date': frame['date'].max(),
            'valid_values': int(frame[column].notna().sum()),
            'missing_values': int(frame[column].isna().sum()),
            'duplicate_dates': int(frame['date'].duplicated().sum()),
            'absent_weekday_rows': len(absent_weekdays),
        })
        assert len(absent_weekdays) == 0, f'{series_id}: 日付行の欠落を確認する'

    raw_audit = pd.DataFrame(raw_audit_rows)
    display(raw_audit)
    display(pd.DataFrame({'series_id': ['SP500', 'VIXCLS'],
                          'date_dtype': [str(sp500['date'].dtype), str(vix['date'].dtype)],
                          'value_dtype': [str(sp500['sp500_close'].dtype), str(vix['vix_close'].dtype)]}))

    # 4. 市場日と欠損の対応
    # 役割: 2系列の日付・欠損の対応を監査する（実行順4）。
    # 入力: 03のsp500、vix。出力: calendar_audit、missing_dates、vix_only_dates。
    calendar_audit = sp500.merge(vix, on='date', how='outer', validate='one_to_one', indicator=True)
    calendar_audit = calendar_audit.sort_values('date').reset_index(drop=True)
    calendar_audit['sp500_missing'] = calendar_audit['sp500_close'].isna()
    calendar_audit['vix_missing'] = calendar_audit['vix_close'].isna()
    missing_dates = calendar_audit.loc[
        calendar_audit['sp500_missing'] | calendar_audit['vix_missing']
    ].copy()
    vix_only_dates = calendar_audit.loc[
        calendar_audit['sp500_missing'] & ~calendar_audit['vix_missing']
    ].copy()
    display(missing_dates.head(12))
    print('SP500欠損・VIX有効の日数:', len(vix_only_dates))
    print('SP500有効・VIX欠損の日数:', int((~calendar_audit['sp500_missing'] & calendar_audit['vix_missing']).sum()))

    # 5. リターン・市場日ラグ・期間分割
    # 役割: SP500市場日上でリターン・ラグを作り、期間分割する（実行順5）。
    # 入力: 01の固定条件、03のsp500、vix。出力: market、analysis。
    market = sp500.loc[sp500['sp500_close'].notna()].copy()
    market = market.merge(vix, on='date', how='left', validate='one_to_one')
    market = market.sort_values('date').reset_index(drop=True)

    market['return_pct'] = 100 * np.log(market['sp500_close']).diff()
    market['return_lag1_pct'] = market['return_pct'].shift(1)
    market['forecast_origin_date'] = market['date'].shift(1)
    market['vix_lag1'] = market['vix_close'].shift(1)
    market['vix_lag1_source_date'] = market['forecast_origin_date'].where(market['vix_lag1'].notna())
    market['calendar_days_since_previous_close'] = (market['date'] - market['forecast_origin_date']).dt.days

    for threshold in [25, 30, 35]:
        flag = market['vix_lag1'].ge(threshold).astype('boolean')
        market[f'high_vix_{threshold}'] = flag.mask(market['vix_lag1'].isna())

    market['market_regime'] = 'unknown_vix'
    market.loc[market['high_vix_30'].eq(False).fillna(False), 'market_regime'] = 'vix_below_30'
    market.loc[market['high_vix_30'].eq(True).fillna(False), 'market_regime'] = 'high_vix_30'

    analysis = market.loc[market['date'].between(ANALYSIS_START, ANALYSIS_END)].copy()
    analysis['split'] = np.select(
        [analysis['date'].le('2018-12-31'), analysis['date'].le('2019-12-31')],
        ['training', 'calibration'], default='test',
    )
    analysis['model_ready'] = analysis[['return_pct', 'return_lag1_pct']].notna().all(axis=1)
    analysis['regime_ready'] = analysis['vix_lag1'].notna()

    # score自体はまだ計算しない。将来scoreを作れる観測の数だけを確認する。
    eligible_score = analysis['model_ready'] & analysis['split'].isin(['calibration', 'test'])
    analysis['past_eligible_score_count'] = eligible_score.cumsum().shift(1, fill_value=0)
    analysis['rolling_history_ready'] = analysis['past_eligible_score_count'].ge(ROLLING_WINDOW)
    analysis['comparison_candidate'] = (
        analysis['split'].eq('test') & analysis['model_ready']
        & analysis['regime_ready'] & analysis['rolling_history_ready']
    )
    analysis = analysis.reset_index(drop=True)

    # 6. 時系列と変数の検証
    # 役割: リターン・ラグの参照先と期間境界を検証する（実行順6）。
    # 入力: 05のmarket、analysis。出力: 検証結果、check_columns。
    assert analysis['date'].is_monotonic_increasing and analysis['date'].is_unique
    assert analysis['date'].between(ANALYSIS_START, ANALYSIS_END).all()
    assert analysis['forecast_origin_date'].lt(analysis['date']).all()
    assert analysis['model_ready'].all(), '目的変数・説明変数の欠損を確認する'
    assert np.isfinite(analysis[['return_pct', 'return_lag1_pct']].to_numpy()).all()

    previous_values = market.set_index('date').reindex(analysis['forecast_origin_date'])
    np.testing.assert_allclose(analysis['return_lag1_pct'], previous_values['return_pct'], equal_nan=True)
    np.testing.assert_allclose(analysis['vix_lag1'], previous_values['vix_close'], equal_nan=True)
    np.testing.assert_allclose(
        analysis['return_pct'],
        100 * np.log(analysis['sp500_close'].to_numpy() / previous_values['sp500_close'].to_numpy()),
        atol=1e-10,
    )
    assert analysis['high_vix_30'].isna().equals(analysis['vix_lag1'].isna())
    assert analysis['vix_lag1_source_date'].dropna().lt(
        analysis.loc[analysis['vix_lag1_source_date'].notna(), 'date']
    ).all()

    split_dates = analysis.groupby('split')['date'].agg(['min', 'max'])
    assert split_dates.loc['training', 'max'] < split_dates.loc['calibration', 'min']
    assert split_dates.loc['calibration', 'max'] < split_dates.loc['test', 'min']

    check_columns = ['date', 'forecast_origin_date', 'sp500_close', 'return_pct',
                     'return_lag1_pct', 'vix_lag1', 'high_vix_30', 'split']
    display(analysis[check_columns].head(3))
    display(analysis.loc[analysis['date'].between('2019-12-30', '2020-01-06'), check_columns])
    print('日付、対数リターン、ラグ、欠損flag、期間境界の検証: PASS')

    # 7. 分割・High-VIXの集計
    # 役割: 期間別件数、高VIX日数、Rolling準備状況を集計する（実行順7）。
    # 入力: 05〜06のanalysis。出力: split_summary、test_data、test_high、
    # high_episode_count、calibration_n、candidate_dates。
    split_summary = analysis.groupby('split', sort=False).agg(
        n=('date', 'size'), first_date=('date', 'min'), last_date=('date', 'max'),
        model_ready_n=('model_ready', 'sum'), regime_ready_n=('regime_ready', 'sum'),
        high_vix_n=('high_vix_30', 'sum'), comparison_candidate_n=('comparison_candidate', 'sum'),
    )
    split_summary['vix_below_30_n'] = analysis.groupby('split', sort=False)['high_vix_30'].agg(lambda s: s.eq(False).sum())
    split_summary['unknown_vix_n'] = analysis.groupby('split', sort=False)['vix_lag1'].agg(lambda s: s.isna().sum())
    split_summary = split_summary.reset_index()
    display(split_summary)

    test_data = analysis.loc[analysis['split'].eq('test')].copy()
    test_high = test_data['high_vix_30'].fillna(False).astype(bool)
    high_episode_count = int((test_high & ~test_high.shift(1, fill_value=False)).sum())
    calibration_n = int((analysis['split'].eq('calibration') & analysis['model_ready']).sum())
    candidate_dates = analysis.loc[analysis['comparison_candidate'], 'date']
    print('2019年の将来score作成可能な観測数:', calibration_n)
    print('直接比較候補の最初の日:', candidate_dates.min())
    print('Test高VIXの連続区間数:', high_episode_count)
    print('高VIX日数は独立なショックの数ではない。')

    # 8. 加工データと監査表の保存
    # 役割: 加工データ・変数定義・監査表を保存し、再読込して照合する（実行順8）。
    # 入力: 01〜07の設定・データ・監査表。出力: data/processed、監査CSV。
    definitions = [
        ('date', '予測対象となるSP500市場日t', 'date', 'outcome_date'),
        ('sp500_close', 'SP500の日次終値。配当を含まない', 'index', 'outcome_day_audit'),
        ('vix_close', '当日VIXCLS終値。予測・当日のregime分類には使わない', 'index', 'outcome_day_audit'),
        ('return_pct', '100 * log(P_t / P_previous)', 'percent_log_return', 'target'),
        ('return_lag1_pct', '直前SP500市場日のreturn_pct', 'percent_log_return', 'predictor'),
        ('forecast_origin_date', '直前SP500市場日。必要な公式終値確定後を予測時点とする', 'date', 'known_before_target_close'),
        ('vix_lag1', '直前SP500市場日のVIX終値。欠損を補完しない', 'index', 'regime_input'),
        ('vix_lag1_source_date', 'vix_lag1が由来する日。値が欠ける場合も欠損', 'date', 'timing_audit'),
        ('calendar_days_since_previous_close', '前回SP500終値からの暦日数', 'days', 'calendar_audit'),
        ('high_vix_25', 'vix_lag1 >= 25。欠損なら未分類', 'nullable_boolean', 'prespecified_sensitivity'),
        ('high_vix_30', 'vix_lag1 >= 30。主stress定義、欠損なら未分類', 'nullable_boolean', 'primary_regime'),
        ('high_vix_35', 'vix_lag1 >= 35。欠損なら未分類', 'nullable_boolean', 'prespecified_sensitivity'),
        ('market_regime', 'high_vix_30 / vix_below_30 / unknown_vix', 'category', 'primary_regime'),
        ('split', 'training / calibration / test。日付で事前固定', 'category', 'sample_split'),
        ('model_ready', 'return_pctとreturn_lag1_pctが両方存在', 'boolean', 'data_eligibility'),
        ('regime_ready', 'vix_lag1が存在', 'boolean', 'subgroup_eligibility'),
        ('past_eligible_score_count', 'calibration開始後、当日より前にscoreを作れる行数。実際のscoreではない', 'count', 'history_audit'),
        ('rolling_history_ready', '過去のscore候補が252行以上', 'boolean', 'history_audit'),
        ('comparison_candidate', 'Testでmodel/regime/historyが揃う。区間評価の最終標本とは後続で照合', 'boolean', 'comparison_eligibility'),
    ]
    variable_definitions = pd.DataFrame(definitions, columns=['variable', 'definition_ja', 'unit', 'role'])
    assert set(variable_definitions['variable']) == set(analysis.columns)

    analysis.to_csv(PROCESSED_DIR / 'analysis_dataset.csv', index=False, date_format='%Y-%m-%d')
    analysis.loc[analysis['model_ready']].to_csv(PROCESSED_DIR / 'model_dataset.csv', index=False, date_format='%Y-%m-%d')
    analysis.loc[analysis['comparison_candidate']].to_csv(PROCESSED_DIR / 'comparison_candidates.csv', index=False, date_format='%Y-%m-%d')
    analysis[check_columns].head(5).to_csv(REPORT_DIR / 'data_preview.csv', index=False, date_format='%Y-%m-%d')
    variable_definitions.to_csv(PROJECT_DIR / 'docs' / 'variable_definitions.csv', index=False, encoding='utf-8-sig')
    raw_audit.to_csv(REPORT_DIR / 'raw_data_audit.csv', index=False, date_format='%Y-%m-%d')
    split_summary.to_csv(REPORT_DIR / 'split_summary.csv', index=False, date_format='%Y-%m-%d')
    calendar_audit.to_csv(REPORT_DIR / 'calendar_audit.csv', index=False, date_format='%Y-%m-%d')
    missing_dates.to_csv(REPORT_DIR / 'missing_dates.csv', index=False, date_format='%Y-%m-%d')
    analysis.isna().sum().rename('missing_n').to_csv(REPORT_DIR / 'processed_missing_counts.csv')

    date_columns = ['date', 'forecast_origin_date', 'vix_lag1_source_date']
    boolean_columns = {f'high_vix_{threshold}': 'boolean' for threshold in [25, 30, 35]}
    for filename, expected in [
        ('analysis_dataset.csv', analysis),
        ('model_dataset.csv', analysis.loc[analysis['model_ready']]),
        ('comparison_candidates.csv', analysis.loc[analysis['comparison_candidate']]),
    ]:
        saved = pd.read_csv(PROCESSED_DIR / filename, parse_dates=date_columns, dtype=boolean_columns)
        pd.testing.assert_frame_equal(
            saved, expected.reset_index(drop=True), check_dtype=False, rtol=1e-12, atol=1e-12,
        )
    print('加工CSVの保存後照合: PASS')

    # 9. 環境と検証結果の記録
    # 役割: 実行環境とStep Aの監査結果を記録する（実行順9）。
    # 入力: 01〜08の設定・データ・集計値。出力: environment.json、audit_summary.json。
    import matplotlib
    import scipy
    import statsmodels

    versions = {'python': sys.version, 'python_executable': sys.executable,
                'platform': platform.platform(), 'working_directory': str(Path.cwd()),
                'project_directory': str(PROJECT_DIR),
                'executed_at_utc': datetime.now(timezone.utc).isoformat(),
                'pandas': pd.__version__, 'numpy': np.__version__,
                'matplotlib': matplotlib.__version__, 'scipy': scipy.__version__,
                'statsmodels': statsmodels.__version__}
    (REPORT_DIR / 'environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    # requirements.txtは再現用の固定ファイルとし、実行時には上書きしない。

    audit = {
        'status': 'Step A completed; no model or interval estimated',
        'analysis_rows': len(analysis), 'test_rows': len(test_data),
        'calibration_eligible_rows': calibration_n,
        'test_high_vix_rows': int(test_high.sum()),
        'test_high_vix_episodes': high_episode_count,
        'comparison_candidate_rows': len(candidate_dates),
        'comparison_first_date': candidate_dates.min().strftime('%Y-%m-%d'),
        'analysis_first_date': analysis['date'].min().strftime('%Y-%m-%d'),
        'analysis_last_date': analysis['date'].max().strftime('%Y-%m-%d'),
        'duplicate_dates': int(analysis['date'].duplicated().sum()),
        'missing_lagged_vix': int(analysis['vix_lag1'].isna().sum()),
        'missing_model_rows': int((~analysis['model_ready']).sum()),
        'vix_only_rows_in_download_period': len(vix_only_dates),
        'validation_checks': 'PASS',
    }
    (REPORT_DIR / 'audit_summary.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    display(pd.Series(audit, name='value').to_frame())


if __name__ == '__main__':
    raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage prepare.')
