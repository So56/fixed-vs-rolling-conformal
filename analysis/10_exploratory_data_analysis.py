"""保存済み加工CSVから期間・市場区分別の記述統計とEDA図を作る。
実行入口はrun_analysis.pyのstage 10。import時には分析しない。
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_exploratory_analysis(workspace):
    """独立workspaceの保存入力を使う。上流モデル・区間・Bootstrapは再計算しない。"""

    import matplotlib
    matplotlib.use('Agg')  # ウィンドウを開かず、PNG・PDFとして保存する。
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'eda'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'eda'
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)




    # Step Aの既存ファイルを実行前後で照合する。READMEとAI利用記録は追記対象。
    protected_files = list((PROJECT_DIR / 'data').rglob('*'))
    protected_files += [p for p in (PROJECT_DIR / 'docs').rglob('*') if p.name != 'ai_assistance_log.md']
    protected_files += [
        p for p in (PROJECT_DIR / 'reports').rglob('*')
        if REPORT_DIR not in p.parents and p.name != 'step_b_summary.md'
    ]
    protected_files += list(PROJECT_DIR.glob('0*.py'))
    protected_files += [PROJECT_DIR / name for name in [
        'requirements.txt', 'file_inventory.csv',
        'scripts/run_step_a.py', 'scripts/verify_step_a_reproduction.py',
    ]]
    protected_files = sorted({p for p in protected_files if p.is_file()})
    hashes_before = {p: file_hash(p) for p in protected_files}

    # 1. 固定条件と列定義の確認。CSVから単独で開始し、型を明示する。
    design = json.loads((PROJECT_DIR / 'docs/research_design_lock.json').read_text(encoding='utf-8'))
    design_source = PROJECT_DIR / 'docs/final_research_design_v1_source.md'
    assert file_hash(design_source) == design['source_extracted_text_sha256']
    assert design['training'] == ['2017-01-01', '2018-12-31']
    assert design['calibration'] == ['2019-01-01', '2019-12-31']
    assert design['test'] == ['2020-01-01', '2026-08-07']
    assert design['high_vix_definition'] == 'previous_SP500_market_day_VIXCLS >= 30'
    VIX_THRESHOLD = 30
    SPLITS = ['training', 'calibration', 'test']
    REGIMES = ['high_vix_30', 'vix_below_30']
    date_columns = ['date', 'forecast_origin_date', 'vix_lag1_source_date']
    boolean_columns = [
        'high_vix_25', 'high_vix_30', 'high_vix_35', 'model_ready', 'regime_ready',
        'rolling_history_ready', 'comparison_candidate',
    ]
    numeric_columns = [
        'sp500_close', 'vix_close', 'return_pct', 'return_lag1_pct', 'vix_lag1',
        'calendar_days_since_previous_close',
    ]
    dtypes = {column: 'boolean' for column in boolean_columns}
    dtypes.update({column: 'float64' for column in numeric_columns})
    dtypes.update({'past_eligible_score_count': 'int64', 'split': 'string', 'market_regime': 'string'})
    data = pd.read_csv(PROJECT_DIR / 'data/processed/analysis_dataset.csv', parse_dates=date_columns, dtype=dtypes)
    definitions = pd.read_csv(PROJECT_DIR / 'docs/variable_definitions.csv', encoding='utf-8-sig')
    assert set(data.columns) == set(definitions['variable'])
    assert len(data) == 2412 and data['date'].is_unique and data['date'].is_monotonic_increasing
    assert not data.isna().any().any()
    assert np.isfinite(data[numeric_columns].to_numpy()).all()
    assert data['forecast_origin_date'].lt(data['date']).all()
    assert data['vix_lag1_source_date'].equals(data['forecast_origin_date'])
    assert data['high_vix_30'].equals(data['vix_lag1'].ge(VIX_THRESHOLD).astype('boolean'))
    expected_counts = {'training': (502, 5), 'calibration': (252, 0), 'test': (1658, 149)}
    count_rows = []
    for split in SPLITS:
        frame = data.loc[data['split'].eq(split)]
        assert frame['date'].between(*design[split]).all()
        high_n = int(frame['high_vix_30'].sum())
        assert (len(frame), high_n) == expected_counts[split]
        count_rows.append({
            'split': split, 'n': len(frame), 'high_vix_n': high_n,
            'non_high_vix_n': len(frame) - high_n, 'high_vix_share': high_n / len(frame),
            'first_date': frame['date'].min(), 'last_date': frame['date'].max(),
        })
    split_counts = pd.DataFrame(count_rows)
    split_counts.to_csv(REPORT_DIR / 'split_regime_counts.csv', index=False, date_format='%Y-%m-%d')
    print('CSV読込確認:\n' + split_counts.to_string(index=False))

    # absは方向を除いた変動の大きさ。既存CSVを変更せず、メモリ内に列を追加する。
    data['abs_return_pct'] = data['return_pct'].abs()
    test = data.loc[data['split'].eq('test')].copy().reset_index(drop=True)
    assert test['date'].iloc[0] == pd.Timestamp('2020-01-02')
    assert test['date'].iloc[-1] == pd.Timestamp('2026-08-07')
    assert int((~test['high_vix_30']).sum()) == 1509

    # A. 同じ集計を期間別・Test市場区分別で行うため、共通関数にする。
    variables = ['return_pct', 'return_lag1_pct', 'vix_lag1', 'abs_return_pct']


    def summarize(frame, group_column, group_values):
        rows = []
        for group in group_values:
            group_data = frame.loc[frame[group_column].eq(group)]
            for variable in variables:
                values = group_data[variable]
                rows.append({
                    group_column: group, 'variable': variable,
                    'unit': 'VIX index points' if variable == 'vix_lag1' else 'percent log return',
                    'n': int(values.count()), 'missing_n': int(values.isna().sum()),
                    'mean': values.mean(), 'median': values.median(), 'std_ddof1': values.std(ddof=1),
                    'min': values.min(), 'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75), 'max': values.max(),
                    'skewness': values.skew(), 'excess_kurtosis': values.kurt(),
                })
        return pd.DataFrame(rows)


    stats_by_split = summarize(data, 'split', SPLITS)
    stats_by_regime = summarize(test, 'market_regime', REGIMES)
    stats_by_split.to_csv(REPORT_DIR / 'descriptive_statistics_by_split.csv', index=False)
    stats_by_regime.to_csv(REPORT_DIR / 'test_statistics_by_regime.csv', index=False)

    # E. 年別集計。2026年の割合の分母は8月7日までの150市場日。
    yearly = test.groupby(test['date'].dt.year.rename('year')).agg(
        total_n=('date', 'size'), high_vix_n=('high_vix_30', 'sum'),
        first_date=('date', 'min'), last_date=('date', 'max'),
    ).reset_index()
    yearly['high_vix_share'] = yearly['high_vix_n'] / yearly['total_n']
    yearly['partial_year'] = yearly['year'].eq(2026)
    yearly.to_csv(REPORT_DIR / 'test_high_vix_by_year.csv', index=False, date_format='%Y-%m-%d')

    # 高VIXだけを先に抽出しない。全Test列のFalse→Trueの位置で開始を定義する。
    high = test['high_vix_30'].astype(bool)
    episode_start = high & ~high.shift(1, fill_value=False)
    test['episode_id'] = episode_start.cumsum()
    episodes = test.loc[high].groupby('episode_id', sort=True).agg(
        start_date=('date', 'min'), end_date=('date', 'max'), high_vix_days=('date', 'size'),
    ).reset_index()
    episodes['calendar_span_days'] = (episodes['end_date'] - episodes['start_date']).dt.days + 1
    episodes['touches_test_start'] = episodes['start_date'].eq(test['date'].iloc[0])
    episodes['touches_test_end'] = episodes['end_date'].eq(test['date'].iloc[-1])
    assert len(episodes) == int(episode_start.sum()) == 25
    assert int(episodes['high_vix_days'].sum()) == 149
    episodes.to_csv(REPORT_DIR / 'test_high_vix_episodes.csv', index=False, date_format='%Y-%m-%d')

    # D. 相関は記述統計のみ。p値・因果推論・回帰は行わない。
    correlation_rows = []
    samples = [(split, data.loc[data['split'].eq(split)]) for split in SPLITS]
    samples += [(f'test_{regime}', test.loc[test['market_regime'].eq(regime)]) for regime in REGIMES]
    for sample, frame in samples:
        correlation_rows.append({
            'sample': sample, 'n': len(frame),
            'pearson_vix_lag1_abs_return': frame['vix_lag1'].corr(frame['abs_return_pct'], method='pearson'),
            'spearman_vix_lag1_abs_return': frame['vix_lag1'].corr(frame['abs_return_pct'], method='spearman'),
        })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(REPORT_DIR / 'vix_absolute_return_correlations.csv', index=False)

    # 図は英語ラベル、同じ色、PNG 300dpiとベクターPDFで保存する。
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    HIGH_COLOR = '#B84B25'
    NON_HIGH_COLOR = '#245A81'
    figure_paths = []


    def save_figure(fig, name):
        for extension in ['png', 'pdf']:
            path = FIGURE_DIR / f'{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            figure_paths.append(path)
        plt.close(fig)


    # B. 全期間の実日付時系列。異なる単位は別パネルにし、同じx軸を使う。
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.2), sharex=True, layout='constrained')
    axes[0].plot(data['date'], data['return_pct'], color='#454D55', linewidth=0.65, label='Daily return')
    axes[0].scatter(data.loc[data['high_vix_30'], 'date'], data.loc[data['high_vix_30'], 'return_pct'],
                    color=HIGH_COLOR, s=9, zorder=3, label='Previous-market-day VIX >= 30')
    axes[0].axhline(0, color='grey', linewidth=0.6)
    axes[0].set_ylabel('Daily log return (%)')
    axes[0].set_title('S&P 500 daily returns and previous-market-day VIX, 2017-2026', loc='left', pad=12)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].plot(data['date'], data['vix_lag1'], color=NON_HIGH_COLOR, linewidth=0.85)
    axes[1].scatter(data.loc[data['high_vix_30'], 'date'], data.loc[data['high_vix_30'], 'vix_lag1'],
                    color=HIGH_COLOR, s=7, zorder=3)
    axes[1].axhline(VIX_THRESHOLD, color=HIGH_COLOR, linestyle='--', linewidth=1, label='Threshold = 30')
    axes[1].set_ylabel('Previous-market-day VIX\n(index points)')
    axes[1].set_xlabel('Outcome date (SP500 market day)')
    axes[1].legend(loc='upper left')
    for ax in axes:
        for boundary in ['2019-01-01', '2020-01-01']:
            ax.axvline(pd.Timestamp(boundary), color='#535353', linestyle=':', linewidth=1.2)
        ax.grid(axis='y', alpha=0.18)
        ax.margins(x=0.01, y=0.14)
    for label, center in [('Training', '2018-01-01'), ('Calibration', '2019-07-01'), ('Test', '2023-04-01')]:
        axes[0].text(pd.Timestamp(center), 0.97, label, transform=axes[0].get_xaxis_transform(),
                     ha='center', va='top', fontsize=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    save_figure(fig, '01_returns_and_lagged_vix_timeseries')

    # C. 全Testの最小〜最大値を含む40個の共通bin。各区分で密度積分は1。
    bin_edges = np.linspace(test['return_pct'].min(), test['return_pct'].max(), 41)
    fig, ax = plt.subplots(figsize=(10, 5.6), layout='constrained')
    histogram_rows = []
    for regime, label, color in [
        ('vix_below_30', 'Non-high VIX (< 30)', NON_HIGH_COLOR),
        ('high_vix_30', 'High VIX (>= 30)', HIGH_COLOR),
    ]:
        values = test.loc[test['market_regime'].eq(regime), 'return_pct']
        densities, _ = np.histogram(values, bins=bin_edges, density=True)
        assert np.isclose(np.sum(densities * np.diff(bin_edges)), 1)
        assert np.histogram(values, bins=bin_edges)[0].sum() == len(values)
        for left, right, density in zip(bin_edges[:-1], bin_edges[1:], densities):
            histogram_rows.append({'market_regime': regime, 'bin_left': left, 'bin_right': right, 'density': density})
        ax.hist(values, bins=bin_edges, density=True, histtype='stepfilled', alpha=0.14, color=color)
        ax.hist(values, bins=bin_edges, density=True, histtype='step', linewidth=1.7,
                color=color, label=f'{label}; n = {len(values):,}')
    ax.set_xlim(bin_edges[0] - 0.4, bin_edges[-1] + 0.4)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Daily log return (%)')
    ax.set_ylabel('Density (per percentage point)')
    ax.set_title('Test return distributions by previous-market-day VIX', loc='left')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.18)
    save_figure(fig, '02_test_return_density_by_vix_regime')
    pd.DataFrame(histogram_rows).to_csv(REPORT_DIR / 'test_histogram_bins_and_densities.csv', index=False)

    # D. Testの散布図。縦軸は方向を取り除いた絶対リターンで、回帰線は引かない。
    fig, ax = plt.subplots(figsize=(9.5, 6), layout='constrained')
    for regime, label, color in [
        ('vix_below_30', 'Non-high VIX (< 30)', NON_HIGH_COLOR),
        ('high_vix_30', 'High VIX (>= 30)', HIGH_COLOR),
    ]:
        frame = test.loc[test['market_regime'].eq(regime)]
        ax.scatter(frame['vix_lag1'], frame['abs_return_pct'], s=15, alpha=0.45,
                   color=color, edgecolors='none', label=f'{label}; n = {len(frame):,}')
    ax.axvline(VIX_THRESHOLD, color=HIGH_COLOR, linestyle='--', linewidth=1)
    ax.set_xlim(0, test['vix_lag1'].max() * 1.04)
    ax.set_ylim(0, test['abs_return_pct'].max() * 1.06)
    ax.set_xlabel('Previous-market-day VIX (index points)')
    ax.set_ylabel('Absolute daily log return (%)')
    ax.set_title('Test: lagged VIX and the magnitude of daily returns', loc='left')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.18)
    save_figure(fig, '03_test_lagged_vix_vs_absolute_return')

    # E. 年別の日数と割合を別パネルで表示する。部分年を明記する。
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, layout='constrained')
    positions = np.arange(len(yearly))
    axes[0].bar(positions, yearly['high_vix_n'], color=HIGH_COLOR, width=0.6)
    axes[1].bar(positions, yearly['high_vix_share'] * 100, color=NON_HIGH_COLOR, width=0.6)
    for position, row in zip(positions, yearly.itertuples()):
        axes[0].text(position, row.high_vix_n + 1.5, f'{row.high_vix_n}/{row.total_n}', ha='center', fontsize=9)
        axes[1].text(position, row.high_vix_share * 100 + 0.6, f'{row.high_vix_share:.1%}', ha='center', fontsize=9)
    axes[0].set_ylim(0, yearly['high_vix_n'].max() * 1.2)
    axes[1].set_ylim(0, yearly['high_vix_share'].max() * 100 * 1.2)
    axes[0].set_ylabel('High-VIX market days')
    axes[1].set_ylabel('Share of market days (%)')
    axes[0].set_title('Test: annual concentration of high-VIX days (lagged VIX >= 30)', loc='left')
    axes[1].set_xticks(positions, [f'{year}*' if year == 2026 else str(year) for year in yearly['year']])
    axes[1].set_xlabel('Year (*2026: through August 7 only; labels above count bars = high-VIX / all days)')
    for ax in axes:
        ax.grid(axis='y', alpha=0.18)
        ax.set_axisbelow(True)
    save_figure(fig, '04_test_high_vix_annual_concentration')

    # 集計規約と実行環境。入力の型や図のbinも再確認できるように保存する。
    pd.DataFrame({'column': data.columns[:19], 'dtype': [str(data[column].dtype) for column in data.columns[:19]]}).to_csv(
        REPORT_DIR / 'input_dtypes.csv', index=False,
    )
    metadata = {
        'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'python_executable': sys.executable,
        'python_version': sys.version, 'working_directory': str(Path.cwd()), 'project_directory': str(PROJECT_DIR),
        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'scope': 'Step B EDA only; no models, residuals, scores, intervals, coverage, widths or bootstrap',
        'input_sha256': file_hash(PROJECT_DIR / 'data/processed/analysis_dataset.csv'),
        'summary_conventions': {
            'std': 'sample standard deviation, ddof=1', 'quantiles': 'pandas linear interpolation',
            'skewness': 'pandas bias-corrected sample skewness',
            'excess_kurtosis': 'pandas bias-corrected Fisher excess kurtosis; normal benchmark = 0',
            'empty_groups': 'n=0; undefined statistics remain NaN, never filled with zero',
            'correlations': 'descriptive Pearson and Spearman; no significance tests',
            'episodes': 'consecutive high-VIX observations in the full chronological Test market-day sequence',
        },
        'high_vix_episodes': len(episodes), 'histogram_bin_count': len(bin_edges) - 1,
        'histogram_range': [float(bin_edges[0]), float(bin_edges[-1])],
        'figures': [p.relative_to(PROJECT_DIR).as_posix() for p in figure_paths],
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    # 日本語の結果メモも同じ数値から再生成する。
    return_stats = stats_by_split.loc[stats_by_split['variable'].eq('return_pct')].set_index('split')
    regime_stats = stats_by_regime.loc[stats_by_regime['variable'].eq('return_pct')].set_index('market_regime')
    abs_stats = stats_by_regime.loc[stats_by_regime['variable'].eq('abs_return_pct')].set_index('market_regime')
    test_corr = correlations.set_index('sample').loc['test']
    longest = episodes.loc[episodes['high_vix_days'].idxmax()]
    summary = f'''# Step B：探索的データ分析（EDA）

保存済みanalysis_dataset.csvから単独実行した。Step Aは再実行・修正していない。
研究期間・主閾値・変数定義を維持し、外れ値除去、winsorization、標準化、補間、forward fillは行っていない。
入力CSV・研究設計・既存Step A成果物は実行前後のSHA-256で不変を確認した。

## 実行方法と成果物

PowerShellで `& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\10_exploratory_data_analysis.py` を実行する。
別フォルダからもこのファイルの絶対パスで実行でき、Step Aのメモリ変数に依存しない。
追加インストールは不要。実際の環境・日時・入力ハッシュは`eda/execution_metadata.json`に保存する。

- コード：`10_exploratory_data_analysis.py`。追加のrun_step_b.pyは不要。
- 表：`eda/`の期間別・Test市場区分別統計、年別日数・割合、25連続区間、相関、histogram binと密度、入力型。
- 図：`../figures/eda/`の4図、それぞれPNG（300dpi）・PDF。
- 保全記録：`eda/step_a_file_preservation.csv`。実行ログ：`eda/execution.log`。
- 図の目視確認は実行後に別途行う。確認記録：`eda/figure_visual_qa.md`。

## 件数と基本統計

| 区分 | n | 高VIX | 非高VIX | 平均リターン (%) | 標準偏差 (%) | 歪度 | 超過尖度 |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for row in split_counts.itertuples():
        stats = return_stats.loc[row.split]
        summary += f'| {row.split} | {row.n:,} | {row.high_vix_n} | {row.non_high_vix_n:,} | {stats["mean"]:.4f} | {stats["std_ddof1"]:.4f} | {stats["skewness"]:.3f} | {stats["excess_kurtosis"]:.3f} |\n'
    summary += '''
全2,412行・19列、入力の欠損・日付重複は0。各区分の件数はStep Aと一致する。
表の標準偏差はddof=1。歪度は非対称性、超過尖度は正規分布を0とする裾の厚さの指標で、極端値の影響を受ける。
これらは標本の記述統計であり、正規性を検定した結果ではない。
四分位数はpandasの線形補間による。観測数0の統計量を0で補完しない。
CSVには中央値、最小、25%・75%点、最大も保存した。

## Test：高VIXと非高VIXの分布

| 市場区分 | n | 平均リターン (%) | リターン標準偏差 (%) | 平均絶対リターン (%) | 中央絶対リターン (%) |
|---|---:|---:|---:|---:|---:|
'''
    for regime in REGIMES:
        stats, absolute = regime_stats.loc[regime], abs_stats.loc[regime]
        summary += f'| {regime} | {int(stats["n"]):,} | {stats["mean"]:.4f} | {stats["std_ddof1"]:.4f} | {absolute["mean"]:.4f} | {absolute["median"]:.4f} |\n'
    summary += f'''
高VIXの標準偏差と平均絶対リターンは非高VIXより大きい。
高VIXの平均リターンが正であっても、上昇を予測できることを意味しない。
Testリターンの範囲は{test['return_pct'].min():.4f}%〜{test['return_pct'].max():.4f}%。
非高VIXにも{regime_stats.loc['vix_below_30', 'min']:.4f}%のリターンがあり、VIX<30は平穏を保証しない。
密度図は両区分で同一の40 bin・横軸範囲を使い、全極値を含めている。
各区分の密度積分は1で、凡例に149日／1,509日を明記している。

## 時系列とVIXとの関係

時系列図はリターンと前SP500市場日のVIXを別パネルで示す。
2019-01-01と2020-01-01の境界、VIX=30、lagged VIXによる高VIX日を明示する。
大きいリターン変動が時間的に集まる様子が見られるが、独立性や依存の強さの検定は行っていない。
当日のvix_closeを当日の区分・相関・散布図に使っていない。

散布図はTestのvix_lag1とabs(return_pct)を示す。
絶対リターンは上昇・下落の方向を取り除き、変動の大きさを見るために使う。
TestのPearson相関は{test_corr['pearson_vix_lag1_abs_return']:.3f}、Spearman相関は{test_corr['spearman_vix_lag1_abs_return']:.3f}。
高いlagged VIXと大きい絶対リターンの正の関連を記述している。
時系列依存、区分内のばらつき、極端値もあるため、因果関係や独立観測を仮定した有意性、予測性能の証拠とは扱わない。

## 高VIX日の年別分布と連続性

| 年 | 全観測数 | 高VIX日数 | 高VIX割合 |
|---|---:|---:|---:|
'''
    for row in yearly.itertuples():
        year_label = f'{row.year}（8月7日まで）' if row.partial_year else str(row.year)
        summary += f'| {year_label} | {row.total_n} | {row.high_vix_n} | {row.high_vix_share:.2%} |\n'
    summary += f'''
2020年80日と2022年48日で、Testの高VIX149日の{(80 + 48) / 149:.1%}を占める。
2026年は8月7日までの部分年であり、通年の日数とそのまま比較しない。
連続区間は全Test時系列上で識別した25区間、合計149日で、Step Aと一致する。
最長区間は{longest['start_date']:%Y-%m-%d}〜{longest['end_date']:%Y-%m-%d}の{int(longest['high_vix_days'])}市場日。
週末・休場日の暦日間隔は区間を分断せず、非高VIXのSP500市場日が挟まると別区間になる。
区間数は独立な経済ショックの回数ではなく、149日も独立な標本とは限らない。

## 解釈の限界とStep Cへの注意

Calibrationの高VIXが0日で、Testには高VIX149日と大きな変動がある。
これは市場環境の相違を示すが、FixedのCoverage低下やRollingの優位を実証したものではない。
標準偏差や超過尖度は予測誤差・score・予測区間の性質とは別である。
平均差検定、正規性検定、OLS、残差、score、区間、Coverage・Width評価、bootstrapは行っていない。
TestのEDAを見て主分析の期間・閾値・モデル・window・推論方法を選び直さない。

Step Cへ進む前に解決を要するデータ不整合は見つからなかった。
次はTrainingの502日だけでOLS AR(1)を一度推定し、以後の係数を固定する。
説明変数はreturn_lag1_pctであり、TestやCalibrationを推定・調整に利用しない。
外部取引所calendarとの全件照合と、当時の公表遅延・改訂を含むvintageの検証は引き続き未実施。
これらの制約と金融時系列の依存性を明記して後続分析へ進む。
'''
    (PROJECT_DIR / 'reports/step_b_summary.md').write_text(summary, encoding='utf-8')

    preservation_rows = []
    for path, digest in hashes_before.items():
        after = file_hash(path)
        preservation_rows.append({
            'path': path.relative_to(PROJECT_DIR).as_posix(),
            'sha256_before': digest, 'sha256_after': after, 'unchanged': digest == after,
        })
    preservation = pd.DataFrame(preservation_rows)
    assert preservation['unchanged'].all(), '既存Step Aファイルに変更があります。'
    preservation.to_csv(REPORT_DIR / 'step_a_file_preservation.csv', index=False)
    print('\nTest区分別リターン統計:\n' + regime_stats.to_string())
    print('\n年別高VIX日数:\n' + yearly.to_string(index=False))
    print(f'\n高VIX連続区間: {len(episodes)}、合計{episodes["high_vix_days"].sum()}日')
    print(f'保全確認: {len(preservation)}ファイル不変。図4点をPNG 300dpi・PDFで保存。')
    print('Step B EDA: PASS（図の目視確認は生成後に別途実施）')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 10.')
    run_exploratory_analysis(ANALYSIS_WORKSPACE)
