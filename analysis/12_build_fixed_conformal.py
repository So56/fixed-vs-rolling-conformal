"""2019年252 scoreの有限標本補正QuantileでFixed区間を作る。
実行入口はrun_analysis.pyのstage 12。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixed_conformal(workspace):
    """保存済みの固定OLS予測を読み、独立workspaceへ区間と検証記録を保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'fixed_conformal'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'fixed_conformal'
    MODEL_DIR = PROJECT_DIR / 'models' / 'fixed_conformal'
    for directory in [REPORT_DIR, FIGURE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH = MODEL_DIR / 'fixed_threshold.json'
    INPUT_PATHS = {
        'analysis_dataset': PROJECT_DIR / 'data/processed/analysis_dataset.csv',
        'fixed_coefficients': PROJECT_DIR / 'models/ols/fixed_coefficients.json',
        'calibration_point_predictions': PROJECT_DIR / 'reports/ols/calibration_point_predictions.csv',
        'test_point_predictions': PROJECT_DIR / 'reports/ols/test_point_predictions.csv',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    RTOL = ATOL = 1e-12




    # 1. 既存入力のハッシュ・設計を照合。Step A〜Cは実行しない。
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    fixed_model = json.loads(INPUT_PATHS['fixed_coefficients'].read_text(encoding='utf-8'))
    assert design['calibration'] == ['2019-01-01', '2019-12-31']
    assert design['test'] == ['2020-01-01', '2026-08-07']
    assert design['nominal_coverage'] == .90 and design['score'] == 'absolute forecast error'
    assert design['fixed_calibration'] == 'all valid 2019 scores; frozen in Test'
    assert design['quantile_rule'] == 'ceil((n+1)*0.90)-th sorted score; no interpolation; infinity if k>n'
    assert design['interval_endpoints'] == 'inclusive'
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert input_hashes['analysis_dataset'] == fixed_model['input_csv_sha256']
    assert input_hashes['research_design_lock'] == fixed_model['research_design_lock_sha256']
    assert fixed_model['training_n'] == 502 and fixed_model['predictors'] == ['return_lag1_pct']
    # JSONの完全精度の値を直接読む。報告書の丸め表示を転記しない。
    intercept = fixed_model['coefficients']['const']
    ar1_coefficient = fixed_model['coefficients']['return_lag1_pct']
    assert np.isfinite([intercept, ar1_coefficient]).all()

    date_columns = ['date', 'forecast_origin_date']
    reference_columns = ['date', 'forecast_origin_date', 'split', 'return_lag1_pct']
    data = pd.read_csv(
        INPUT_PATHS['analysis_dataset'], usecols=reference_columns + ['return_pct'],
        parse_dates=date_columns, float_precision='round_trip',
        dtype={'split': 'string', 'return_pct': 'float64', 'return_lag1_pct': 'float64'},
    )
    assert len(data) == 2412 and data['date'].is_unique and data['date'].is_monotonic_increasing
    assert not data.isna().any().any()
    expected = {
        'calibration': (252, '2019-01-02', '2019-12-31'),
        'test': (1658, '2020-01-02', '2026-08-07'),
    }
    aligned = {}
    join_checks = []

    # 2. dateでone-to-oneに結合。outerにより片側だけの日付を黙って落とさない。
    for split, (expected_n, first_date, last_date) in expected.items():
        reference = data.loc[data['split'].eq(split)].copy().reset_index(drop=True)
        predictions = pd.read_csv(
            INPUT_PATHS[f'{split}_point_predictions'], parse_dates=date_columns, float_precision='round_trip',
            dtype={'split': 'string', 'return_lag1_pct': 'float64', 'predicted_return_pct': 'float64'},
        )
        assert list(predictions.columns) == reference_columns + ['predicted_return_pct']
        assert len(reference) == len(predictions) == expected_n
        for frame in [reference, predictions]:
            assert frame['date'].is_unique and frame['date'].is_monotonic_increasing
            assert frame['date'].min() == pd.Timestamp(first_date) and frame['date'].max() == pd.Timestamp(last_date)
            assert frame['date'].between(*design[split]).all() and frame['split'].eq(split).all()
            assert frame['forecast_origin_date'].lt(frame['date']).all()
            assert not frame.isna().any().any()
        assert np.isfinite(reference[['return_pct', 'return_lag1_pct']].to_numpy()).all()
        assert np.isfinite(predictions[['predicted_return_pct', 'return_lag1_pct']].to_numpy()).all()
        # Testの実測値は有限性の確認だけで終え、結合・区間生成に渡す列から除く。
        columns_to_join = reference_columns + (['return_pct'] if split == 'calibration' else [])
        joined = reference[columns_to_join].merge(
            predictions, on='date', how='outer', validate='one_to_one', indicator=True,
            suffixes=('_source', '_prediction'),
        )
        assert len(joined) == expected_n and joined['_merge'].eq('both').all()
        assert joined['forecast_origin_date_source'].equals(joined['forecast_origin_date_prediction'])
        assert joined['split_source'].equals(joined['split_prediction'])
        np.testing.assert_allclose(joined['return_lag1_pct_source'], joined['return_lag1_pct_prediction'], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            joined['predicted_return_pct'], intercept + ar1_coefficient * joined['return_lag1_pct_source'], rtol=RTOL, atol=ATOL,
        )
        frame = pd.DataFrame({
            'date': joined['date'], 'forecast_origin_date': joined['forecast_origin_date_source'],
            'split': joined['split_source'], 'return_lag1_pct': joined['return_lag1_pct_source'],
            'predicted_return_pct': joined['predicted_return_pct'],
        })
        if split == 'calibration':
            frame['return_pct'] = joined['return_pct']
        assert frame['date'].is_monotonic_increasing and frame['date'].is_unique
        pd.testing.assert_series_equal(frame['date'], predictions['date'], check_names=False)
        aligned[split] = frame
        join_checks.append({'split': split, 'n': len(frame), 'outer_merge_all_both': True,
                            'origin_split_predictor_match': True, 'fixed_coefficient_formula_match': True})

    # 3. 2019年だけのout-of-sample誤差と絶対誤差。VIXによる絞り込みはしない。
    calibration = aligned['calibration'].copy()
    calibration['calibration_error_pct'] = calibration['return_pct'] - calibration['predicted_return_pct']
    calibration['score_pct'] = calibration['calibration_error_pct'].abs()
    calibration = calibration[[
        'date', 'forecast_origin_date', 'split', 'return_lag1_pct', 'return_pct',
        'predicted_return_pct', 'calibration_error_pct', 'score_pct',
    ]]
    assert len(calibration) == 252 and calibration['date'].dt.year.eq(2019).all()
    assert calibration['split'].eq('calibration').all()
    assert np.isfinite(calibration['score_pct']).all() and calibration['score_pct'].ge(0).all()
    assert calibration['date'].max() <= aligned['test']['forecast_origin_date'].min()
    assert calibration['date'].max() < aligned['test']['date'].min()
    np.testing.assert_array_equal(
        calibration['score_pct'], np.abs(calibration['return_pct'] - calibration['predicted_return_pct']),
    )
    score_path = REPORT_DIR / 'calibration_scores_2019.csv'
    calibration.to_csv(score_path, index=False, date_format='%Y-%m-%d')

    # 4. 補間せず、並べたscoreのk番目を直接選ぶ。同値scoreにも乱数は使わない。
    nominal_coverage = float(design['nominal_coverage'])
    alpha = .10
    assert math.isclose(1 - alpha, nominal_coverage)
    n = len(calibration)
    k = math.ceil((n + 1) * (1 - alpha))
    sorted_scores = np.sort(calibration['score_pct'].to_numpy(), kind='stable')
    q_fixed = float(sorted_scores[k - 1]) if k <= n else math.inf
    assert n == 252 and k == 228 and k - 1 == 227
    assert math.isfinite(q_fixed) and q_fixed >= 0
    assert q_fixed == sorted_scores[227]
    executed_at = datetime.now(timezone.utc)
    threshold_record = {
        'method': 'Fixed split conformal; 2019 Calibration scores only; frozen throughout Test',
        'q_fixed_pct': q_fixed, 'unit': 'percentage points', 'n': n,
        'alpha': alpha, 'nominal_coverage': nominal_coverage, 'k': k, 'zero_based_index': k - 1,
        'calibration_period_specified': design['calibration'],
        'calibration_first_date': calibration['date'].min().strftime('%Y-%m-%d'),
        'calibration_last_date': calibration['date'].max().strftime('%Y-%m-%d'),
        'score_definition': 'abs(return_pct - predicted_return_pct); out-of-sample 2019 errors only',
        'calibration_filter': 'all 252 valid 2019 observations; no VIX filtering',
        'quantile_rule': 'k=ceil((n+1)*(1-alpha)); q=sorted_scores[k-1]; infinity if k>n',
        'interpolation': 'none', 'tie_rule': 'stable sort; no random tie-breaking',
        'endpoints': 'inclusive', 'future_covered_rule': 'lower_fixed_pct <= actual_return_pct <= upper_fixed_pct',
        'threshold_uses_test_outcomes': False, 'test_predictions_hash_is_provenance_only': True,
        'input_files': {name: {'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256': input_hashes[name]}
                        for name, path in INPUT_PATHS.items()},
        'calibration_score_csv_sha256': file_hash(score_path),
        'executed_at_utc': executed_at.isoformat(),
        'executed_at_jst': executed_at.astimezone(timezone(timedelta(hours=9))).isoformat(),
        'environment': {'python_executable': sys.executable, 'python_version': sys.version,
                        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'code_sha256': file_hash(Path(__file__).resolve()),
    }
    THRESHOLD_PATH.write_text(json.dumps(threshold_record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    # 5. 保存閾値を読み直し、点予測とqだけからTestの区間を作る。実測リターンは渡さない。
    saved_threshold = json.loads(THRESHOLD_PATH.read_text(encoding='utf-8'))
    q_saved = saved_threshold['q_fixed_pct']
    assert q_saved == q_fixed
    test_predictions = aligned['test'][['date', 'forecast_origin_date', 'split', 'predicted_return_pct']].copy()
    assert 'return_pct' not in test_predictions.columns
    intervals = test_predictions.copy()
    intervals['q_fixed_pct'] = q_saved
    intervals['lower_fixed_pct'] = intervals['predicted_return_pct'] - q_saved
    intervals['upper_fixed_pct'] = intervals['predicted_return_pct'] + q_saved
    assert len(intervals) == 1658 and intervals['q_fixed_pct'].eq(q_saved).all()
    assert np.isfinite(intervals[['predicted_return_pct', 'q_fixed_pct', 'lower_fixed_pct', 'upper_fixed_pct']].to_numpy()).all()
    assert intervals['lower_fixed_pct'].le(intervals['upper_fixed_pct']).all()
    np.testing.assert_allclose((intervals['lower_fixed_pct'] + intervals['upper_fixed_pct']) / 2,
                               intervals['predicted_return_pct'], rtol=RTOL, atol=ATOL)
    # これは構成式の照合であり、平均Width等の性能評価ではない。
    np.testing.assert_allclose(intervals['upper_fixed_pct'] - intervals['lower_fixed_pct'], 2 * q_saved, rtol=RTOL, atol=ATOL)
    interval_path = REPORT_DIR / 'test_fixed_intervals.csv'
    intervals.to_csv(interval_path, index=False, date_format='%Y-%m-%d')

    # 6. 保存CSVをround-tripで再読込し、全列・値を照合する。
    for path, expected_frame in [(score_path, calibration), (interval_path, intervals)]:
        saved_frame = pd.read_csv(path, parse_dates=date_columns, dtype={'split': 'string'}, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved_frame, expected_frame, check_exact=True)
    saved_scores = pd.read_csv(score_path, float_precision='round_trip')['score_pct'].to_numpy()
    assert float(np.sort(saved_scores, kind='stable')[saved_threshold['k'] - 1]) == q_saved

    # 7. Calibration scoreのヒストグラム。Testの観測値やCoverageは表示しない。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 5.4), layout='constrained')
    ax.hist(calibration['score_pct'], bins=np.linspace(0, calibration['score_pct'].max() * 1.001, 26),
            color='#245A81', edgecolor='white', alpha=.8)
    ax.axvline(q_saved, color='#B84B25', linestyle='--', linewidth=1.8,
               label=f'Fixed half-width = {q_saved:.6f} pp (rank {k} of {n})')
    ax.set_title('2019 Calibration: absolute forecast-error scores (n = 252)', loc='left')
    ax.set_xlabel('Absolute forecast-error score (percentage points)')
    ax.set_ylabel('Calibration market days (count)')
    ax.set_xlim(0, calibration['score_pct'].max() * 1.06)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_calibration_scores_and_fixed_threshold.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    # 8. 入力6ファイルだけを実行前後で照合。前段階の全監査は繰り返さない。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    assert input_hashes == input_hashes_after
    validation = {
        'status': 'PASS', 'joins': join_checks, 'calibration_scores_n': n,
        'scores_2019_only_finite_nonnegative': True, 'signed_error_and_absolute_score_match': True,
        'n': n, 'k': k, 'zero_based_index': k - 1, 'q_fixed_pct': q_saved,
        'saved_score_order_statistic_matches_threshold': True, 'random_tie_breaking': False,
        'test_interval_rows': len(intervals), 'test_dates_match_saved_point_predictions': True,
        'centers_match_point_predictions': True, 'finite_ordered_endpoints': True,
        'same_q_all_rows_and_endpoint_difference_matches_2q': True,
        'saved_csv_all_columns_exact_round_trip': True, 'comparison_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'test_interval_generation_columns': list(test_predictions.columns),
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'ols_refit': False, 'step_a_to_c_reexecuted': False,
        'rolling_test_scores_coverage_mean_width_interval_score_bootstrap_computed': False,
    }
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {
        'executed_at_utc': threshold_record['executed_at_utc'], 'executed_at_jst': threshold_record['executed_at_jst'],
        'environment': threshold_record['environment'], 'working_directory': str(Path.cwd()),
        'input_hashes': input_hashes, 'code_sha256': threshold_record['code_sha256'],
        'fixed_threshold_json_sha256': file_hash(THRESHOLD_PATH), 'score_csv_sha256': file_hash(score_path),
        'fixed_interval_csv_sha256': file_hash(interval_path), 'figures': figure_files,
        'scope': 'Step D-1 only: 2019 scores, fixed threshold, Test fixed intervals; no performance evaluation',
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = f'''# Step D-1：2019年Calibration scoreとFixed予測区間

## 実行範囲と結果

保存済み加工CSV、固定係数JSON、Calibration/Test点予測CSV、確定設計から単独実行した。
Step A〜Cを再実行せず、OLSも再推定していない。入力6ファイルの実行前後のSHA-256は一致した。
固定係数はJSONの完全精度の値を読み、丸めた報告値を転記していない。

| 項目 | 結果 |
|---|---|
| Calibration期間 | 2019-01-02〜2019-12-31 |
| 有効score数 n | {n} |
| Nominal coverage / alpha | {nominal_coverage:.2f} / {alpha:.2f} |
| 順位 k（1始まり） | {k} |
| Python添字（0始まり） | {k - 1} |
| q_fixed（半幅、percentage points） | {q_saved!r} |
| 全幅（構成式2*q_fixed、percentage points） | {2 * q_saved!r} |
| Test期間 | 2020-01-02〜2026-08-07 |
| Test区間行数 | {len(intervals):,} |

全幅は同じ半幅を左右に足した構成式を示したもので、平均Widthを性能指標として集計していない。
q_fixedは全Test日で固定し、Testの結果による調整は行っていない。

## scoreと閾値をどう作ったか

OLSの係数推定にはTrainingを使用済みであるため、その当てはめ残差は学習内の誤差である。
Calibrationには、Trainingで固定したモデルが未使用の2019年を予測したout-of-sample誤差を用いる。
Training残差、残差標準誤差、係数の信頼区間、Testの誤差を閾値計算へ流用していない。
VIXによるCalibrationの絞り込みも行わず、2019年の全252日を使用した。

signed errorは`return_pct - predicted_return_pct`、scoreはその絶対値。
signed errorは予測からの上振れ・下振れの向き、scoreは方向を除いた予測誤差の大きさを測る。
どちらも単位はpercentage pointsで、CSVでは別列に保存した。

有限標本のsplit-conformal rank ruleに合わせ、`k=ceil((n+1)*(1-alpha))`を使う。
将来の1観測を加えたn+1個の順位を考える補正で、今回`ceil(253*0.90)=ceil(227.7)=228`となる。
交換可能性の下で目標marginal coverageに対応する順位を切り上げて選ぶもので、通常の補間90パーセンタイルとは異なる。
`np.sort(..., kind='stable')[k-1]`で228番目、Python添字227を直接選び、線形補間やrandom tie-breakingは行わない。
一般規則はk>nなら無限大だが、今回はk<=nで有限の閾値になった。

## コードの対応

- 1：設計・入力ハッシュ・固定係数JSONの確認。
- 2：dateのouter one-to-one結合。両側全日付、予測元日、split、説明変数、固定係数の予測式を照合。
- 3：2019年だけの`calibration_error_pct`と`score_pct`を作成。
- 4：n、ceilによる順位k、ソートしたscoreの閾値を計算し、JSONへ保存。
- 5：閾値JSONを再読込し、Testの保存済み点予測に`lower=prediction-q`、`upper=prediction+q`を適用。
- 6：保存CSVの全列round-trip照合と、再読込scoreの228番目の照合。
- 7：Calibration scoreのヒストグラムとq_fixedの位置をPNG 300dpi・PDFへ保存。
- 8：必要な入力の不変確認と検証結果の保存。

Test実測値は有限性の確認だけに使い、区間生成用データには渡していない。
Test区間CSVにはdate、forecast_origin_date、split、predicted_return_pct、q_fixed_pct、lower_fixed_pct、upper_fixed_pctの7列を保存した。
実測値・Test誤差・score・covered indicatorは保存していない。

## 保存先と実行方法

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\12_build_fixed_conformal.py
```

- `reports/fixed_conformal/calibration_scores_2019.csv`：2019年の実測値・点予測・signed error・score。
- `models/fixed_conformal/fixed_threshold.json`：完全精度の閾値、順位、計算条件、入力ハッシュ、実行日時。
- `reports/fixed_conformal/test_fixed_intervals.csv`：Test全1,658日のFixed区間。
- `figures/fixed_conformal/01_calibration_scores_and_fixed_threshold.png`とPDF：score分布と固定閾値。
- `reports/fixed_conformal/validation_results.json`、`execution_metadata.json`、`execution.log`：検証と実行記録。
- 図の目視確認記録：`reports/fixed_conformal/figure_visual_qa.md`（実行後に別途確認）。

scoreは2019年の252個だけで、すべて有限・非負。結合で落ちた日付や不一致はなかった。
保存CSVはround-trip読込により全列・値が完全一致し、閾値JSONも元の浮動小数点値を保持した。
全区間の中心と点予測、上下限の順序・有限性、同じq_fixed、上限−下限=2*q_fixedを確認した。
中心と構成式の数値照合の許容差はrtol=atol=1e-12。

## 解釈・限界・次の段階

この区間は翌SP500市場日の百分率対数リターンの予測区間であり、平均や回帰係数の信頼区間ではない。
q_fixedは半幅で、予測値から下にq_fixed、上にq_fixedを取るため全幅は2*q_fixedになる。
将来の評価では`lower_fixed_pct <= actual_return_pct <= upper_fixed_pct`をcoveredとする（両端inclusive）。
今回はそのindicatorやTestのCoverageを計算していない。

標準的なConformalの有限標本marginal coverage保証には交換可能性が必要である。
金融時系列の依存・市場環境の変化や、高VIXという部分集団で90%Coverageが自動的に保証されるとは主張しない。
Calibrationに高VIX日が0日であることだけで、FixedのCoverage低下を実証したとも扱わない。

計算・日付・入力の整合性に未解決の問題は見つからなかった。
次の段階では、今回保存した時系列順の2019年score252個をRolling-252の初期履歴として使用できる。
今回はRolling、Test score、Coverage・平均Width・Interval Score、bootstrapへ進まず、Fixed区間の生成で終了した。
'''
    (PROJECT_DIR / 'reports/step_d1_summary.md').write_text(summary, encoding='utf-8')
    print('結合確認:\n' + pd.DataFrame(join_checks).to_string(index=False))
    print(f'n={n}, k={k}, zero-based index={k-1}, q_fixed={q_saved!r} percentage points')
    print(f'Test Fixed intervals: {len(intervals)} rows; input hashes unchanged: {len(input_hashes)} files')
    print('Step D-1: PASS。図の目視確認は生成後に別途実施。Rolling・Test score・性能評価は未実施。')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 12.')
    build_fixed_conformal(ANALYSIS_WORKSPACE)
