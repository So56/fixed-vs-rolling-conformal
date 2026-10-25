"""保存済みFixed/Rolling区間のCoverage・Gap・Width・Interval Scoreを評価する。
実行入口はrun_analysis.pyのstage 14。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_prediction_intervals(workspace):
    """独立workspaceの保存区間を同一日で結合し、日次・区分別の評価を保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'evaluation'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'evaluation'
    for directory in [REPORT_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATHS = {
        'analysis_dataset': PROJECT_DIR / 'data/processed/analysis_dataset.csv',
        'fixed_intervals': PROJECT_DIR / 'reports/fixed_conformal/test_fixed_intervals.csv',
        'rolling_intervals': PROJECT_DIR / 'reports/rolling_conformal/test_rolling_intervals.csv',
        'test_scores': PROJECT_DIR / 'reports/rolling_conformal/test_scores.csv',
        'fixed_threshold': PROJECT_DIR / 'models/fixed_conformal/fixed_threshold.json',
        'research_design_lock': PROJECT_DIR / 'docs/research_design_lock.json',
        'research_design_source': PROJECT_DIR / 'docs/final_research_design_v1_source.md',
    }
    METHODS = ['fixed', 'rolling']
    GROUPS = ['overall', 'high_vix', 'non_high_vix']
    GROUP_LABELS = {'overall': 'Overall', 'high_vix': 'High-VIX', 'non_high_vix': 'Non-high-VIX'}
    EXPECTED_N = {'overall': 1658, 'high_vix': 149, 'non_high_vix': 1509}
    RTOL = ATOL = 1e-12  # 数値照合用。Coverageとmiss判定には絶対に使わない。




    def interval_metrics(actual_returns, lower_bounds, upper_bounds, alpha):
        """境界を広げず、両端inclusiveのCoverageとWidth・Interval Scoreを計算する。"""
        actual_returns, lower_bounds, upper_bounds = [np.asarray(values, dtype=float) for values in [actual_returns, lower_bounds, upper_bounds]]
        assert actual_returns.shape == lower_bounds.shape == upper_bounds.shape
        assert np.isfinite(actual_returns).all() and np.isfinite(lower_bounds).all() and np.isfinite(upper_bounds).all()
        assert (lower_bounds <= upper_bounds).all()
        covered = (lower_bounds <= actual_returns) & (actual_returns <= upper_bounds)
        lower_miss = actual_returns < lower_bounds
        upper_miss = actual_returns > upper_bounds
        width = upper_bounds - lower_bounds
        lower_penalty = np.where(lower_miss, (2 / alpha) * (lower_bounds - actual_returns), 0.0)
        upper_penalty = np.where(upper_miss, (2 / alpha) * (actual_returns - upper_bounds), 0.0)
        return pd.DataFrame({
            'covered': covered, 'lower_miss': lower_miss, 'upper_miss': upper_miss,
            'width_return_pp': width, 'lower_penalty_return_pp': lower_penalty,
            'upper_penalty_return_pp': upper_penalty,
            'interval_score_return_pp': width + lower_penalty + upper_penalty,
        })


    # 1. 固定設計と必要な入力を確認する。前段階は実行しない。
    input_hashes = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    design = json.loads(INPUT_PATHS['research_design_lock'].read_text(encoding='utf-8'))
    threshold = json.loads(INPUT_PATHS['fixed_threshold'].read_text(encoding='utf-8'))
    assert design['test'] == ['2020-01-01', '2026-08-07']
    assert design['nominal_coverage'] == .90 and design['interval_endpoints'] == 'inclusive'
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    assert design['rolling_window'] == 252 and design['point_model'] == 'OLS AR(1), fitted once on Training only'
    assert input_hashes['research_design_source'] == design['source_extracted_text_sha256']
    assert input_hashes['analysis_dataset'] == threshold['input_files']['analysis_dataset']['sha256']
    assert input_hashes['research_design_lock'] == threshold['input_files']['research_design_lock']['sha256']
    nominal_coverage, alpha = float(design['nominal_coverage']), float(threshold['alpha'])
    q_fixed = float(threshold['q_fixed_pct'])
    assert alpha == .10 and math.isclose(1 - alpha, nominal_coverage) and 2 / alpha == 20
    date_columns = ['date', 'forecast_origin_date']
    data = pd.read_csv(
        INPUT_PATHS['analysis_dataset'],
        usecols=date_columns + ['split', 'return_pct', 'vix_lag1', 'high_vix_30', 'market_regime'],
        parse_dates=date_columns, float_precision='round_trip',
        dtype={'split': 'string', 'market_regime': 'string', 'high_vix_30': 'boolean'},
    )
    assert len(data) == 2412 and data['date'].is_unique
    daily = data.loc[data['split'].eq('test')].copy().reset_index(drop=True)
    assert len(daily) == 1658 and daily['date'].is_monotonic_increasing
    assert daily['date'].iloc[0] == pd.Timestamp('2020-01-02') and daily['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert not daily.isna().any().any(), 'VIXの未知区分をFalseに置き換えず、入力を調査してください。'
    assert np.isfinite(daily[['return_pct', 'vix_lag1']].to_numpy()).all()
    assert daily['forecast_origin_date'].lt(daily['date']).all()
    assert daily['high_vix_30'].equals(daily['vix_lag1'].ge(30).astype('boolean'))
    assert daily.loc[daily['high_vix_30'], 'market_regime'].eq('high_vix_30').all()
    assert daily.loc[~daily['high_vix_30'], 'market_regime'].eq('vix_below_30').all()

    # 2. dateによるouter one-to-one結合。重複・片側だけの日付・識別列不一致を検出する。
    join_records = []
    for label in ['fixed', 'rolling', 'score']:
        path_key = f'{label}_intervals' if label != 'score' else 'test_scores'
        usecols = None if label != 'score' else date_columns + ['split', 'return_pct', 'predicted_return_pct']
        incoming = pd.read_csv(INPUT_PATHS[path_key], usecols=usecols, parse_dates=date_columns,
                               dtype={'split': 'string'}, float_precision='round_trip')
        assert len(incoming) == 1658 and incoming['date'].is_unique and incoming['date'].is_monotonic_increasing
        assert not incoming.isna().any().any()
        assert incoming['split'].eq('test').all() and incoming['forecast_origin_date'].lt(incoming['date']).all()
        assert np.isfinite(incoming.select_dtypes(include='number').to_numpy()).all()
        overlap_columns = [column for column in incoming.columns if column != 'date' and column in daily.columns]
        combined = daily.merge(incoming, on='date', how='outer', validate='one_to_one', indicator=True,
                               suffixes=('', f'_{label}_input'))
        assert len(combined) == 1658 and combined['_merge'].eq('both').all()
        for column in overlap_columns:
            copied = f'{column}_{label}_input'
            if pd.api.types.is_numeric_dtype(combined[column]):
                np.testing.assert_array_equal(combined[column], combined[copied])
            else:
                assert combined[column].equals(combined[copied]), column
        # 照合済みの重複列だけを除く。観測行は一切除外しない。
        daily = combined.drop(columns=[f'{column}_{label}_input' for column in overlap_columns] + ['_merge'])
        join_records.append({'input': label, 'rows': len(incoming), 'one_to_one_all_dates_match': True,
                             'overlapping_identity_and_value_columns_match': True})
    assert daily['date'].is_unique and daily['date'].is_monotonic_increasing
    assert daily['q_fixed_pct'].eq(q_fixed).all()
    assert not daily.isna().any().any()

    # 3. 小さな人工例で境界・符号を先に確認する。研究データには追加しない。
    artificial_actual = np.array([0., -1., 1., -2., 2., np.nextafter(-1., -np.inf), np.nextafter(1., np.inf)])
    artificial = interval_metrics(artificial_actual, np.full(7, -1.), np.full(7, 1.), alpha)
    np.testing.assert_array_equal(artificial['covered'], [True, True, True, False, False, False, False])
    np.testing.assert_array_equal(artificial['width_return_pp'], np.full(7, 2.))
    np.testing.assert_array_equal(artificial['interval_score_return_pp'].iloc[:5], [2., 2., 2., 22., 22.])
    assert artificial.loc[:2, ['lower_penalty_return_pp', 'upper_penalty_return_pp']].eq(0).all().all()
    assert artificial.loc[5, 'lower_penalty_return_pp'] > 0 and artificial.loc[6, 'upper_penalty_return_pp'] > 0
    artificial.insert(0, 'actual_return_pct', artificial_actual)
    artificial.insert(0, 'case', ['inside', 'at_lower', 'at_upper', 'below', 'above', 'one_float_below_lower', 'one_float_above_upper'])
    artificial['lower_pct'], artificial['upper_pct'] = -1., 1.
    artificial.to_csv(REPORT_DIR / 'validation_artificial_boundary_examples.csv', index=False)

    # 4. 両手法の実データ指標。同じ関数を使い、判定前の丸めや許容差は加えない。
    for method in METHODS:
        metrics = interval_metrics(daily['return_pct'], daily[f'lower_{method}_pct'], daily[f'upper_{method}_pct'], alpha)
        for column in metrics.columns:
            if column in ['covered', 'lower_miss', 'upper_miss']:
                output_name = f'{column}_{method}'
            else:
                output_name = column.replace('_return_pp', f'_{method}_return_pp')
            daily[output_name] = metrics[column]
        covered = daily[f'covered_{method}']
        assert (covered.astype('int64') + daily[f'lower_miss_{method}'].astype('int64')
                + daily[f'upper_miss_{method}'].astype('int64')).eq(1).all()
        assert np.isfinite(daily[[f'width_{method}_return_pp', f'interval_score_{method}_return_pp']].to_numpy()).all()
        assert daily[f'width_{method}_return_pp'].ge(0).all()
        assert daily[f'interval_score_{method}_return_pp'].ge(daily[f'width_{method}_return_pp']).all()
        assert daily.loc[covered, [f'lower_penalty_{method}_return_pp', f'upper_penalty_{method}_return_pp']].eq(0).all().all()
        np.testing.assert_allclose((daily[f'lower_{method}_pct'] + daily[f'upper_{method}_pct']) / 2,
                                   daily['predicted_return_pct'], rtol=RTOL, atol=ATOL)

    # boolのまま引かず、0/1へ変換する。日次差の方向はすべてRolling−Fixed。
    daily['coverage_difference_rolling_minus_fixed'] = daily['covered_rolling'].astype('int64') - daily['covered_fixed'].astype('int64')
    daily['width_difference_rolling_minus_fixed_return_pp'] = daily['width_rolling_return_pp'] - daily['width_fixed_return_pp']
    daily['interval_score_difference_rolling_minus_fixed_return_pp'] = daily['interval_score_rolling_return_pp'] - daily['interval_score_fixed_return_pp']
    for metric in ['covered', 'lower_miss', 'upper_miss', 'width', 'interval_score', 'lower_penalty', 'upper_penalty']:
        suffix = '' if metric in ['covered', 'lower_miss', 'upper_miss'] else '_return_pp'
        assert daily[f'{metric}_fixed{suffix}'].iloc[0] == daily[f'{metric}_rolling{suffix}'].iloc[0]
    for endpoint in ['lower', 'upper']:
        assert daily[f'{endpoint}_fixed_pct'].iloc[0] == daily[f'{endpoint}_rolling_pct'].iloc[0]
    np.testing.assert_allclose(daily['width_fixed_return_pp'], 2 * q_fixed, rtol=RTOL, atol=ATOL)

    # 5. 3区分×2手法の集計。全日CSVは高VIXだけに圧縮しない。
    group_masks = {
        'overall': pd.Series(True, index=daily.index),
        'high_vix': daily['high_vix_30'],
        'non_high_vix': ~daily['high_vix_30'],
    }
    summary_rows, paired_rows = [], []
    for group in GROUPS:
        sample = daily.loc[group_masks[group]]
        sample_size = len(sample)
        assert sample_size == EXPECTED_N[group]
        for method in METHODS:
            covered_count = int(sample[f'covered_{method}'].sum())
            lower_miss_count, upper_miss_count = int(sample[f'lower_miss_{method}'].sum()), int(sample[f'upper_miss_{method}'].sum())
            coverage = covered_count / sample_size
            assert covered_count + lower_miss_count + upper_miss_count == sample_size and 0 <= coverage <= 1
            coverage_gap_pp = 100 * (coverage - nominal_coverage)
            summary_rows.append({
                'group': group, 'method': method, 'n': sample_size, 'covered_n': covered_count,
                'lower_miss_n': lower_miss_count, 'upper_miss_n': upper_miss_count, 'empirical_coverage': coverage,
                'coverage_gap_pp': coverage_gap_pp, 'absolute_coverage_gap_pp': abs(coverage_gap_pp),
                'mean_width_return_pp': sample[f'width_{method}_return_pp'].mean(),
                'median_width_return_pp': sample[f'width_{method}_return_pp'].median(),
                'mean_interval_score_return_pp': sample[f'interval_score_{method}_return_pp'].mean(),
            })
        # 日次paired差を平均する経路を保持し、手法別平均の差に置き換えない。
        coverage_difference = sample['coverage_difference_rolling_minus_fixed'].mean()
        paired_rows.append({
            'group': group, 'n': sample_size, 'primary_coverage_comparison': group == 'high_vix',
            'coverage_difference_rolling_minus_fixed': coverage_difference,
            'coverage_difference_rolling_minus_fixed_pp': 100 * coverage_difference,
            'mean_width_difference_rolling_minus_fixed_return_pp': sample['width_difference_rolling_minus_fixed_return_pp'].mean(),
            'mean_interval_score_difference_rolling_minus_fixed_return_pp': sample['interval_score_difference_rolling_minus_fixed_return_pp'].mean(),
        })
    comparison = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    comparison_indexed = comparison.set_index(['group', 'method'])
    paired_indexed = paired.set_index('group')

    # 6. 集計の代数的な整合性を確認する。検定や信頼区間は計算しない。
    for group in GROUPS:
        fixed_row, rolling_row = comparison_indexed.loc[(group, 'fixed')], comparison_indexed.loc[(group, 'rolling')]
        paired_row = paired_indexed.loc[group]
        np.testing.assert_allclose(
            [paired_row['coverage_difference_rolling_minus_fixed'],
             paired_row['mean_width_difference_rolling_minus_fixed_return_pp'],
             paired_row['mean_interval_score_difference_rolling_minus_fixed_return_pp']],
            [rolling_row['empirical_coverage'] - fixed_row['empirical_coverage'],
             rolling_row['mean_width_return_pp'] - fixed_row['mean_width_return_pp'],
             rolling_row['mean_interval_score_return_pp'] - fixed_row['mean_interval_score_return_pp']], rtol=RTOL, atol=ATOL,
        )
    for method in METHODS:
        overall = comparison_indexed.loc[('overall', method)]
        high, non_high = comparison_indexed.loc[('high_vix', method)], comparison_indexed.loc[('non_high_vix', method)]
        for column in ['n', 'covered_n', 'lower_miss_n', 'upper_miss_n']:
            assert overall[column] == high[column] + non_high[column]
        # 加重平均は平均・Coverage・signed gapに適用。中央値やabsolute gapには適用しない。
        for column in ['empirical_coverage', 'coverage_gap_pp', 'mean_width_return_pp', 'mean_interval_score_return_pp']:
            weighted = (high[column] * high['n'] + non_high[column] * non_high['n']) / overall['n']
            np.testing.assert_allclose(overall[column], weighted, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(comparison.loc[comparison['method'].eq('fixed'), 'mean_width_return_pp'],
                               2 * q_fixed, rtol=RTOL, atol=ATOL)

    output_frames = {
        'daily_test_evaluation.csv': daily,
        'comparison_by_regime.csv': comparison,
        'paired_differences_by_regime.csv': paired,
    }
    for filename, frame in output_frames.items():
        path = REPORT_DIR / filename
        frame.to_csv(path, index=False, date_format='%Y-%m-%d')
        if filename == 'daily_test_evaluation.csv':
            saved = pd.read_csv(path, parse_dates=date_columns, float_precision='round_trip',
                                 dtype={'split': 'string', 'market_regime': 'string', 'high_vix_30': 'boolean'})
        else:
            saved = pd.read_csv(path, float_precision='round_trip')
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)

    # 7. Coverage・Width・ISを異なる単位の3パネルに分け、CIを付けず表示する。
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout='constrained')
    colors = {'fixed': '#B84B25', 'rolling': '#245A81'}
    labels = {'fixed': 'Fixed', 'rolling': 'Rolling-252'}
    x = np.arange(len(GROUPS))
    panels = [
        ('empirical_coverage', 100, 'Empirical coverage', 'Coverage (%)'),
        ('mean_width_return_pp', 1, 'Mean interval width', 'Return interval width (percentage points)'),
        ('mean_interval_score_return_pp', 1, 'Mean interval score', 'Interval score (return percentage points)'),
    ]
    for ax, (column, scale, title, ylabel) in zip(axes, panels):
        largest = float(comparison[column].max() * scale)
        for j, method in enumerate(METHODS):
            values = [float(comparison_indexed.loc[(group, method), column]) * scale for group in GROUPS]
            positions = x + (j - .5) * .34
            bars = ax.bar(positions, values, width=.32, color=colors[method], label=labels[method])
            for bar, value in zip(bars, values):
                label = f'{value:.2f}%' if column == 'empirical_coverage' else f'{value:.2f}'
                ax.text(bar.get_x() + bar.get_width() / 2, value + largest * .018, label,
                        ha='center', va='bottom', fontsize=9)
        if column == 'empirical_coverage':
            ax.axhline(100 * nominal_coverage, color='#555555', linestyle='--', linewidth=1)
            ax.text(-.43, 100 * nominal_coverage + 1, '90% target', fontsize=8, color='#555555')
            ax.set_ylim(0, 115)
        else:
            ax.set_ylim(0, largest * 1.30)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [f'{GROUP_LABELS[group]}\n(n={EXPECTED_N[group]:,})' for group in GROUPS])
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=8, framealpha=.9)
    fig.suptitle('Fixed vs Rolling-252: descriptive Test performance (no confidence intervals)', fontsize=13)
    figure_files = []
    for extension in ['png', 'pdf']:
        path = FIGURE_DIR / f'01_fixed_vs_rolling_performance.{extension}'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
    plt.close(fig)

    # 8. 入力保全・設定・次のMBB用主入力を記録する。MBB自体は実行しない。
    input_hashes_after = {name: file_hash(path) for name, path in INPUT_PATHS.items()}
    assert input_hashes == input_hashes_after
    executed_at = datetime.now(timezone.utc)
    validation = {
        'status': 'PASS', 'daily_test_rows': len(daily), 'group_counts': EXPECTED_N,
        'joins': join_records, 'regime_uses_vix_lag1_only': True, 'unknown_vix_coerced_to_false': False,
        'coverage_inclusive_without_rounding_or_numeric_tolerance': True,
        'covered_lower_miss_upper_miss_partition_n': True, 'coverage_in_0_1': True,
        'finite_nonnegative_width_and_interval_score_at_least_width': True,
        'covered_and_boundary_penalties_zero': True,
        'artificial_examples_pass': True, 'adjacent_float_outside_boundaries_not_covered': True,
        'first_day_intervals_and_daily_metrics_equal': True,
        'fixed_daily_width_and_all_group_mean_width_match_2q': True,
        'paired_mean_differences_equal_differences_of_method_means': True,
        'overall_counts_and_weighted_means_match_regime_partitions': True,
        'median_and_absolute_gap_not_checked_as_weighted_means': True,
        'saved_csv_all_columns_exact_round_trip': True,
        'numeric_validation_tolerance_only': {'rtol': RTOL, 'atol': ATOL},
        'input_hashes_before': input_hashes, 'input_hashes_after': input_hashes_after,
        'prior_steps_refit_score_or_threshold_regeneration': False,
        'bootstrap_confidence_intervals_hypothesis_tests_sensitivity_computed': False,
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
        'alpha': alpha, 'nominal_coverage': nominal_coverage, 'figures': figure_files,
        'units': {'empirical_coverage_and_raw_coverage_difference': 'proportion',
                  'coverage_gap_pp_absolute_gap_and_coverage_difference_pp': 'coverage percentage points',
                  'width_interval_score_penalty_and_their_differences': 'return percentage points'},
        'next_step_bootstrap': {
            'executed': False, 'primary_input': 'reports/evaluation/daily_test_evaluation.csv',
            'input_rows': 1658, 'resampling_sequence': 'full chronological paired vector; subgroup statistics computed within resample',
            'method': design['primary_inference'], 'B': design['bootstrap_repetitions'],
            'block_length': design['block_length'], 'seed': design['seed'], 'CI': design['confidence_interval'],
        },
        'scope': 'Step E-1 descriptive evaluation only; no inferential tests or bootstrap',
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    high_difference = paired_indexed.loc['high_vix']
    summary = f'''# Step E-1：FixedとRollingの記述的性能比較

## 実行範囲と評価標本

保存済み加工データ・Fixed/Rolling区間・Test score・Fixed閾値・確定設計を読み込んで単独実行した。
前段階のスクリプト、OLS再推定、score履歴や閾値の再生成・調整は行っていない。
Test 2020-01-02〜2026-08-07の1,658日をdateでouter one-to-one結合した。
全日付が両手法に存在し、予測元日・split・点予測が一致した。保存済みTest scoreの実測値・点予測も一致した。
欠損・無限大・重複・結合で落ちた観測はなく、VIX未知区分をFalseへ置き換えていない。
市場区分は当日VIXではなくvix_lag1>=30。High-VIX149日、Non-high-VIX1,509日で、同一区分の両手法の対象日と分母は同じ。
全1,658日の時系列を維持した日次CSVを保存し、高VIXだけに圧縮していない。

## 指標の式とコードの対応

実行入口は`14_evaluate_prediction_intervals.py`。

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\14_evaluate_prediction_intervals.py
```

コメント1〜2で入力・区分・日付結合を検証し、3で人工例、4で同じ`interval_metrics`関数を両手法に適用する。
5で集計と日次Rolling−Fixed差、6で集計整合性、7で比較図、8で入力保全と実行記録を保存する。

- Coverage indicator：`(lower <= actual) & (actual <= upper)`。両端inclusiveで、保存値を丸めず、rtol/atolで境界を広げない。
- Width：`upper - lower`。リターンのpercentage points（return pp）。
- Interval Score：`width + 20*(lower-actual)*I(actual<lower) + 20*(actual-upper)*I(actual>upper)`。
  alpha=0.10で、区間内・境界上ではpenalty=0かつIS=Width。外側では距離の正のpenaltyを追加する。小さいほどこの指標上は良い。
- Coverage Gap：`100*(coverage-0.90)`。単位はCoverageのpercentage points（coverage pp）。absolute gapはその絶対値。
- Paired difference：coveredを0/1整数に変換した差、およびWidth・ISの日次差を作り、同じ日のRolling−Fixedを区分内平均する。

リターン区間のreturn ppと、Coverage差のcoverage ppを混同しない。
CSVのCoverageは0〜1の比例尺度で保存し、下表では表示用に%へ変換した。数値照合用の許容差1e-12はCoverage判定に使っていない。

## 3区分×2手法の比較

| 区分 | 手法 | n | covered | 下側miss | 上側miss | Coverage (%) | signed Gap (coverage pp) | absolute Gap (coverage pp) | Mean Width (return pp) | Median Width (return pp) | Mean IS (return pp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in comparison.itertuples():
        summary += f'| {GROUP_LABELS[row.group]} | {row.method} | {row.n:,} | {row.covered_n} | {row.lower_miss_n} | {row.upper_miss_n} | {100*row.empirical_coverage:.4f} | {row.coverage_gap_pp:+.4f} | {row.absolute_coverage_gap_pp:.4f} | {row.mean_width_return_pp:.6f} | {row.median_width_return_pp:.6f} | {row.mean_interval_score_return_pp:.6f} |\n'
    summary += '''
## Paired difference：Rolling−Fixed

| 区分 | n | Coverage差 (coverage pp) | Mean Width差 (return pp) | Mean IS差 (return pp) |
|---|---:|---:|---:|---:|
'''
    for row in paired.itertuples():
        summary += f'| {GROUP_LABELS[row.group]} | {row.n:,} | {row.coverage_difference_rolling_minus_fixed_pp:+.4f} | {row.mean_width_difference_rolling_minus_fixed_return_pp:+.6f} | {row.mean_interval_score_difference_rolling_minus_fixed_return_pp:+.6f} |\n'
    summary += f'''
主な比較対象はHigh-VIXのCoverage差（{high_difference['coverage_difference_rolling_minus_fixed_pp']:+.4f} coverage pp）である。
Overall・Non-high-VIXの比較やWidth・ISは、全体像・効率性・trade-offを理解するための補助的な記述とする。
Coverage差が正ならRollingのCoverageが高く、Width差が正ならRollingが広く、IS差が負ならRollingの平均ISが小さい。
'''
    for group in GROUPS:
        frow, rrow = comparison_indexed.loc[(group, 'fixed')], comparison_indexed.loc[(group, 'rolling')]
        if rrow['absolute_coverage_gap_pp'] < frow['absolute_coverage_gap_pp']:
            closeness = 'Rollingの方が90%に近い'
        elif rrow['absolute_coverage_gap_pp'] > frow['absolute_coverage_gap_pp']:
            closeness = 'Fixedの方が90%に近い'
        else:
            closeness = '両手法は90%から同じ距離にある'
        summary += f'\n{GROUP_LABELS[group]}の90%からのabsolute gapはFixed {frow["absolute_coverage_gap_pp"]:.4f} pp、Rolling {rrow["absolute_coverage_gap_pp"]:.4f} ppで、点推定では{closeness}。\n'
    summary += '''
Coverageの高さと90%への近さを区別し、signed/absolute Gapも見る。
区間を広げるとCoverageが高くなる可能性がある一方、Widthの増加を伴うため、ISの外れ距離penaltyも含めてtrade-offを読む。
点推定は同一標本での記述であり、差の統計的有意性や将来の優位を結論づけるものではない。

## 検証結果

人工例L=-1、U=1ではactual=0,−1,1がcoveredかつWidth=IS=2、actual=−2,2はnot coveredかつIS=22だった。
境界の直外の隣接浮動小数点値もnot coveredとなり、許容差が判定へ混入していない。
人工例は検証専用CSVへ保存し、研究データには追加していない。

各手法・各区分でcovered+下側miss+上側miss=n、Coverageの0〜1範囲、有限・非負のWidth、IS>=Width、covered時のpenalty=0を確認した。
初日の上下限と日次指標はFixed/Rollingで一致した。
Fixedの日次Widthと3区分の平均Widthは2*q_fixedと数値誤差の範囲で一致した。
Paired差の平均は各手法の平均の差と一致し、Overallの件数・Coverage・平均Width・平均ISは高/非高VIXの件数加重集計と一致した。
中央値・absolute gapは非線形のため、加重平均一致の対象にしていない。
主要3 CSVの全列・値・件数・日付は保存後のround-trip読込で完全一致した。
入力7ファイルの実行前後SHA-256も一致した。既存のデータ・予測・区間・係数・設計は変更していない。

## 保存先と次の主入力

- `reports/evaluation/daily_test_evaluation.csv`：全1,658日の時系列順paired vector。実測値・共通点予測・lagged VIX・高VIXindicator・両区間・日次指標・Rolling−Fixed差。
- `reports/evaluation/comparison_by_regime.csv`：3区分×2手法の比較表。
- `reports/evaluation/paired_differences_by_regime.csv`：区分別の同日差平均。
- `reports/evaluation/validation_artificial_boundary_examples.csv`：研究データに含めない人工例。
- `reports/evaluation/validation_results.json`、`execution_metadata.json`、`execution.log`：検証・入力ハッシュ・実行記録。
- `figures/evaluation/01_fixed_vs_rolling_performance.png`とPDF：Coverage・平均Width・平均ISを単位別の3パネルで表示。Coverageに90%線とn、推測したerror barは付けていない。
- 図の目視確認記録：`reports/evaluation/figure_visual_qa.md`（実行後に別途確認）。

## 解釈の限界とStep E-2への状態

高VIX149日は独立な標本とは限らず、2020年・2022年に集中している。
今回は記述的比較だけで、二項検定・通常の独立標本CI・その他の係数検定は実施していない。
点推定から「統計的に有意」「90%を達成したと確認できた」とは結論づけない。
Rollingにしただけで交換可能性や90%Coverage保証が得られるわけではなく、90%への近さを理論保証の実証とも扱わない。
結果を見て期間・VIX閾値・window・モデルを変更していない。

計算・日付・共通評価標本に未解決の不整合は見つからなかった。
Step E-2では、上記の日次CSV全1,658日のpaired vectorを時系列blockで再抽出し、その再標本内で高VIX統計量を計算する。
高VIX149日だけの圧縮時系列を主入力にしない。
確定設定はPaired Moving Block Bootstrap、B=5000、block length=10、seed=202701、95% percentile CIである。
今回はbootstrap、信頼区間、感度分析は実施せず、性能指標の計算・検証・記述で終了した。
'''
    (PROJECT_DIR / 'reports/step_e1_summary.md').write_text(summary, encoding='utf-8')
    print('3区分×2手法の比較:\n' + comparison.to_string(index=False))
    print('\nPaired difference（Rolling−Fixed）:\n' + paired.to_string(index=False))
    print(f'日次評価: {len(daily)}行を時系列順で保存。入力{len(input_hashes)}件のハッシュ不変。')
    print('Step E-1: PASS。bootstrap・信頼区間・感度分析は未実施。図の目視確認は別途実施。')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 14.')
    evaluate_prediction_intervals(ANALYSIS_WORKSPACE)
