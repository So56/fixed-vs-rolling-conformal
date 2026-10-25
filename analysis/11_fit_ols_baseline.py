"""TrainingだけでOLS AR(1)を1回推定し、保存した固定係数で点予測する。
実行はrun_analysis.pyのstage 11。研究上の旧Step Cという出力名はそのまま保持する。
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import platform
import sys


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ols_baseline(workspace):
    """独立workspaceのTrainingで推定し、同じworkspaceへ結果を保存する。"""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import acf


    PROJECT_DIR = Path(workspace).resolve()
    REPORT_DIR = PROJECT_DIR / 'reports' / 'ols'
    FIGURE_DIR = PROJECT_DIR / 'figures' / 'ols'
    MODEL_DIR = PROJECT_DIR / 'models' / 'ols'
    for directory in [REPORT_DIR, FIGURE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    INPUT_PATH = PROJECT_DIR / 'data' / 'processed' / 'analysis_dataset.csv'
    LOCK_PATH = PROJECT_DIR / 'docs' / 'research_design_lock.json'
    SOURCE_PATH = PROJECT_DIR / 'docs' / 'final_research_design_v1_source.md'
    COEFFICIENT_PATH = MODEL_DIR / 'fixed_coefficients.json'
    FEATURE = 'return_lag1_pct'
    TARGET = 'return_pct'
    RTOL = ATOL = 1e-12




    # 既存Step A/Bのファイルは読取のみ。AI利用記録だけは実行後の追記対象。
    new_directories = [REPORT_DIR, FIGURE_DIR, MODEL_DIR]
    new_files = {Path(__file__).resolve(), PROJECT_DIR / 'reports' / 'step_c_summary.md'}
    protected_files = sorted(
        p for p in PROJECT_DIR.rglob('*')
        if p.is_file() and p not in new_files and '__pycache__' not in p.parts
        and p != PROJECT_DIR / 'docs' / 'ai_assistance_log.md'
        and not any(directory in p.parents for directory in new_directories)
    )
    hashes_before = {p: file_hash(p) for p in protected_files}

    # 1. 固定設計・Step Bの図確認記録・保存済みCSVを確認する。
    design = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    assert file_hash(SOURCE_PATH) == design['source_extracted_text_sha256']
    assert design['point_model'] == 'OLS AR(1), fitted once on Training only'
    assert design['training'] == ['2017-01-01', '2018-12-31']
    assert design['calibration'] == ['2019-01-01', '2019-12-31']
    assert design['test'] == ['2020-01-01', '2026-08-07']
    assert design['target'] == '100 * (log(P_t) - log(P_previous_SP500_market_day))'
    qa_path = PROJECT_DIR / 'reports' / 'eda' / 'figure_visual_qa.md'
    assert qa_path.exists(), 'Step Bの図の目視確認記録を確認してください。'
    assert '4つのPNGを実際に開いて確認した' in qa_path.read_text(encoding='utf-8')
    eda_metadata = json.loads((PROJECT_DIR / 'reports/eda/execution_metadata.json').read_text(encoding='utf-8'))
    assert file_hash(INPUT_PATH) == eda_metadata['input_sha256'], 'Step B以降の入力変更を確認してください。'
    assert len(pd.read_csv(INPUT_PATH, nrows=0).columns) == 19
    input_columns = ['date', 'forecast_origin_date', 'split', TARGET, FEATURE]
    data = pd.read_csv(
        INPUT_PATH, usecols=input_columns, parse_dates=['date', 'forecast_origin_date'],
        dtype={TARGET: 'float64', FEATURE: 'float64', 'split': 'string'},
    )
    assert len(data) == 2412 and data['date'].is_unique and data['date'].is_monotonic_increasing
    assert not data.isna().any().any()
    assert np.isfinite(data[[TARGET, FEATURE]].to_numpy()).all()
    assert set(data['split']) == {'training', 'calibration', 'test'}
    assert data['forecast_origin_date'].lt(data['date']).all()
    pd.testing.assert_series_equal(
        data['forecast_origin_date'].iloc[1:].reset_index(drop=True),
        data['date'].iloc[:-1].reset_index(drop=True), check_names=False,
    )
    np.testing.assert_allclose(data[FEATURE].iloc[1:], data[TARGET].iloc[:-1], rtol=RTOL, atol=ATOL)
    expected = {
        'training': (502, '2017-01-03', '2018-12-31'),
        'calibration': (252, '2019-01-02', '2019-12-31'),
        'test': (1658, '2020-01-02', '2026-08-07'),
    }
    split_rows = []
    for split, (n, first, last) in expected.items():
        frame = data.loc[data['split'].eq(split)]
        assert len(frame) == n and frame['date'].between(*design[split]).all()
        assert frame['date'].min() == pd.Timestamp(first) and frame['date'].max() == pd.Timestamp(last)
        split_rows.append({'split': split, 'n': n, 'first_date': first, 'last_date': last})

    # 最初の説明変数だけは分析CSV内に元日がないため、保存済み生CSV末尾の2終値で照合。
    # 2016年の目的変数をTrainingに追加する処理ではない。
    first_origin = data['forecast_origin_date'].iloc[0]
    assert first_origin == pd.Timestamp('2016-12-30')
    warmup = pd.read_csv(PROJECT_DIR / 'data/raw/SP500_fred.csv', parse_dates=['observation_date'])
    warmup['SP500'] = pd.to_numeric(warmup['SP500'], errors='coerce')
    warmup = warmup.loc[warmup['observation_date'].le(first_origin) & warmup['SP500'].notna()].tail(2)
    assert len(warmup) == 2 and warmup['observation_date'].iloc[-1] == first_origin
    warmup_return = 100 * (np.log(warmup['SP500'].iloc[-1]) - np.log(warmup['SP500'].iloc[-2]))
    np.testing.assert_allclose(data[FEATURE].iloc[0], warmup_return, rtol=RTOL, atol=ATOL)

    # 2. 推定するのはこの1回だけ。説明変数を明示し、VIXも他期間も渡さない。
    training = data.loc[data['split'].eq('training')].copy().set_index('date', drop=False)
    training_design = sm.add_constant(training[[FEATURE]], has_constant='add')
    training_returns = training[TARGET]
    assert list(training_design.columns) == ['const', FEATURE] and len(training_design) == 502
    assert training_design.index.year.isin([2017, 2018]).all()
    assert not training_design.index.isin(data.loc[~data['split'].eq('training'), 'date']).any()
    assert np.linalg.matrix_rank(training_design.to_numpy()) == 2
    model = sm.OLS(training_returns, training_design, missing='raise').fit()
    assert model.nobs == 502 and model.df_resid == 500 and len(model.params) == 2
    assert model.model.data.row_labels.equals(training.index)
    assert model.model.exog_names == ['const', FEATURE]

    coefficient_table = pd.DataFrame({
        'coefficient': model.params.index,
        'estimate': model.params.to_numpy(), 'standard_error_ols': model.bse.to_numpy(),
        't_value': model.tvalues.to_numpy(), 'p_value_ols': model.pvalues.to_numpy(),
        'parameter_ci95_lower': model.conf_int().iloc[:, 0].to_numpy(),
        'parameter_ci95_upper': model.conf_int().iloc[:, 1].to_numpy(),
    })
    coefficient_table.to_csv(REPORT_DIR / 'coefficient_table.csv', index=False)
    # 標準要約の診断欄はstatsmodelsによる自動掲載。追加の診断検定は実行しない。
    (REPORT_DIR / 'statsmodels_summary.txt').write_text(model.summary().as_text(), encoding='utf-8')

    # SSEにはstatsmodelsの当てはめ残差を使う。後の直接計算の残差とは演算経路を分ける。
    ols_residuals = model.resid.to_numpy()
    sse = float(np.dot(ols_residuals, ols_residuals))
    residual_standard_error = float(np.sqrt(sse / (len(training) - 2)))
    np.testing.assert_allclose(residual_standard_error, np.sqrt(model.mse_resid), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(training_design.to_numpy().T @ ols_residuals, [0, 0], rtol=0, atol=1e-10)
    fit_statistics = {
        'nobs': int(model.nobs), 'n_parameters_including_intercept': 2,
        'df_resid': int(model.df_resid), 'r_squared': float(model.rsquared),
        'adjusted_r_squared': float(model.rsquared_adj), 'training_sse': sse,
        'residual_standard_error': residual_standard_error,
        'residual_standard_error_formula': 'sqrt(SSE / (n - 2)); n=502, residual df=500',
        'covariance_type': model.cov_type,
    }
    (REPORT_DIR / 'model_fit_statistics.json').write_text(json.dumps(fit_statistics, indent=2), encoding='utf-8')

    # 3. 係数をJSONへ固定する。後続分析はこのJSONを読み、OLSを再推定しない。
    executed_at = datetime.now(timezone.utc)
    environment = {
        'python_executable': sys.executable, 'python_version': sys.version, 'platform': platform.platform(),
        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__,
    }
    coefficient_record = {
        'model': 'OLS AR(1) with intercept; fitted once on Training only',
        'formula': 'return_pct_t = const + return_lag1_pct_coefficient * return_lag1_pct_t + error_t',
        'target': TARGET, 'predictors': [FEATURE], 'includes_intercept': True,
        'coefficients': {name: float(value) for name, value in model.params.items()},
        'units': {TARGET: 'percent log return', FEATURE: 'percent log return',
                  'predicted_return_pct': 'percent log return', 'const': 'percent log return',
                  'return_lag1_pct_coefficient': 'dimensionless; percentage-point response per percentage-point predictor'},
        'training_period_specified': design['training'],
        'training_first_outcome_date': training['date'].min().strftime('%Y-%m-%d'),
        'training_last_outcome_date': training['date'].max().strftime('%Y-%m-%d'),
        'training_n': len(training), 'fit_statistics': fit_statistics,
        'input_csv': INPUT_PATH.relative_to(PROJECT_DIR).as_posix(), 'input_csv_sha256': file_hash(INPUT_PATH),
        'research_design_lock': LOCK_PATH.relative_to(PROJECT_DIR).as_posix(),
        'research_design_lock_sha256': file_hash(LOCK_PATH), 'research_design_source_sha256': file_hash(SOURCE_PATH),
        'code_sha256': file_hash(Path(__file__).resolve()),
        'executed_at_utc': executed_at.isoformat(),
        'executed_at_jst': executed_at.astimezone(timezone(timedelta(hours=9))).isoformat(),
        'environment': environment, 'coefficient_policy': 'frozen for all Calibration and Test predictions; no refitting',
    }
    COEFFICIENT_PATH.write_text(json.dumps(coefficient_record, ensure_ascii=False, indent=2), encoding='utf-8')
    loaded = json.loads(COEFFICIENT_PATH.read_text(encoding='utf-8'))
    intercept = loaded['coefficients']['const']
    ar1_coefficient = loaded['coefficients'][FEATURE]
    assert loaded['training_n'] == 502 and loaded['input_csv_sha256'] == file_hash(INPUT_PATH)

    # 4. Training内の当てはめ誤差。Calibration scoreとして使ってはいけない。
    training_output = training[['date', 'forecast_origin_date', 'split', FEATURE, TARGET]].copy()
    training_output['fitted_return_pct'] = intercept + ar1_coefficient * training_output[FEATURE]
    training_output['training_residual_pct'] = training_output[TARGET] - training_output['fitted_return_pct']
    np.testing.assert_allclose(training_output['fitted_return_pct'], model.fittedvalues, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(training_output['training_residual_pct'], model.resid, rtol=RTOL, atol=ATOL)
    training_output.to_csv(REPORT_DIR / 'training_fitted_residuals.csv', index=False, date_format='%Y-%m-%d')
    # 保存係数の直接計算から得る残差。上のOLS残差へ置き換えない。
    training_prediction_residuals = training_output['training_residual_pct']
    residual_summary = pd.DataFrame([{
        'n': training_prediction_residuals.count(), 'mean': training_prediction_residuals.mean(), 'median': training_prediction_residuals.median(),
        'std_ddof1': training_prediction_residuals.std(ddof=1), 'min': training_prediction_residuals.min(), 'q25': training_prediction_residuals.quantile(.25),
        'q75': training_prediction_residuals.quantile(.75), 'max': training_prediction_residuals.max(), 'skewness': training_prediction_residuals.skew(),
        'excess_kurtosis': training_prediction_residuals.kurt(), 'sse': sse, 'residual_standard_error_df_n_minus_2': residual_standard_error,
    }])
    residual_summary.to_csv(REPORT_DIR / 'training_residual_statistics.csv', index=False)

    # 5. 点予測の入力はラグだけ。Calibration/Testの目的変数・VIX・誤差を出力しない。
    prediction_checks = []
    prediction_columns = ['date', 'forecast_origin_date', 'split', FEATURE]
    for split in ['calibration', 'test']:
        predictions = data.loc[data['split'].eq(split), prediction_columns].copy().reset_index(drop=True)
        predictions['predicted_return_pct'] = intercept + ar1_coefficient * predictions[FEATURE]
        from_statsmodels = model.predict(sm.add_constant(predictions[[FEATURE]], has_constant='add'))
        np.testing.assert_allclose(predictions['predicted_return_pct'], from_statsmodels, rtol=RTOL, atol=ATOL)
        assert len(predictions) == expected[split][0]
        assert predictions['date'].is_unique and predictions['forecast_origin_date'].lt(predictions['date']).all()
        assert np.isfinite(predictions['predicted_return_pct']).all()
        output_path = REPORT_DIR / f'{split}_point_predictions.csv'
        predictions.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        saved = pd.read_csv(output_path, parse_dates=['date', 'forecast_origin_date'], dtype={'split': 'string'})
        pd.testing.assert_frame_equal(saved, predictions, check_dtype=False, rtol=RTOL, atol=ATOL)
        # 保存したCSVとJSONだけで再現。ここではstatsmodelsのfitもpredictも不要。
        reloaded_coefficients = json.loads(COEFFICIENT_PATH.read_text(encoding='utf-8'))['coefficients']
        reproduced = reloaded_coefficients['const'] + reloaded_coefficients[FEATURE] * saved[FEATURE]
        np.testing.assert_allclose(saved['predicted_return_pct'], reproduced, rtol=RTOL, atol=ATOL)
        prediction_checks.append({
            'split': split, 'n': len(saved), 'first_date': saved['date'].min().strftime('%Y-%m-%d'),
            'last_date': saved['date'].max().strftime('%Y-%m-%d'),
            'max_abs_difference_statsmodels_vs_fixed': float(np.max(np.abs(from_statsmodels - predictions['predicted_return_pct']))),
            'max_abs_difference_json_reload_vs_saved_csv': float(np.max(np.abs(saved['predicted_return_pct'] - reproduced))),
        })

    # 6. Training残差と残差二乗の記述的なACF。信頼帯や追加の検定は付けない。
    residual_acf = acf(training_prediction_residuals.to_numpy(), nlags=20, fft=False, adjusted=False, missing='raise')
    squared_residual_acf = acf(training_prediction_residuals.to_numpy() ** 2, nlags=20, fft=False, adjusted=False, missing='raise')
    acf_table = pd.DataFrame({
        'lag': np.arange(21), 'training_residual_acf': residual_acf,
        'training_squared_residual_acf': squared_residual_acf,
    })
    acf_table.to_csv(REPORT_DIR / 'training_residual_acf.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    figure_files = []


    def save_figure(fig, name):
        for extension in ['png', 'pdf']:
            path = FIGURE_DIR / f'{name}.{extension}'
            fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
            figure_files.append(path.relative_to(PROJECT_DIR).as_posix())
        plt.close(fig)


    fig, ax = plt.subplots(figsize=(11.5, 4.5), layout='constrained')
    ax.plot(training_output['date'], training_prediction_residuals, color='#245A81', linewidth=.8)
    ax.axhline(0, color='grey', linewidth=.8)
    ax.set_title('Training only: OLS AR(1) residuals, 2017-2018', loc='left')
    ax.set_xlabel('Outcome date (SP500 market day)')
    ax.set_ylabel('Training residual (percentage points)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='y', alpha=.18)
    ax.margins(x=.01, y=.10)
    save_figure(fig, '01_training_residual_timeseries')

    fig, ax = plt.subplots(figsize=(8.8, 5), layout='constrained')
    ax.hist(training_prediction_residuals, bins=np.linspace(training_prediction_residuals.min(), training_prediction_residuals.max(), 31), density=True,
            color='#245A81', alpha=.7, edgecolor='white', label=f'Training residuals; n = {len(training_prediction_residuals)}')
    ax.axvline(0, color='#B84B25', linestyle='--', linewidth=1)
    ax.set_title('Training only: residual distribution (all observations)', loc='left')
    ax.set_xlabel('Training residual (percentage points)')
    ax.set_ylabel('Density (per percentage point)')
    ax.set_xlim(training_prediction_residuals.min() - .2, training_prediction_residuals.max() + .2)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=.18)
    save_figure(fig, '02_training_residual_histogram')

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, layout='constrained')
    for ax, values, title, color in [
        (axes[0], residual_acf, 'Residuals', '#245A81'),
        (axes[1], squared_residual_acf, 'Squared residuals', '#B84B25'),
    ]:
        ax.bar(np.arange(1, 21), values[1:], width=.65, color=color)
        ax.axhline(0, color='grey', linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel('Lag (SP500 market-day observations)')
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Autocorrelation (unitless)')
    fig.suptitle('Training only: descriptive ACF, lags 1-20 (no significance bands)', fontsize=12)
    save_figure(fig, '03_training_residual_and_squared_residual_acf')

    # 7. 日本語の説明。係数解釈と、当てはめ／将来の点予測を明確に分ける。
    largest_squared_lag = int(np.argmax(np.abs(squared_residual_acf[1:])) + 1)
    coefficient_sign = '正' if ar1_coefficient >= 0 else '負'
    summary = f'''# Step C：固定OLS AR(1)による基本点予測

## 実行範囲とモデルの目的

Training（2017-01-03〜2018-12-31）の502行だけで、定数項付きOLS AR(1)を1回推定した。
Step A/Bは再実行・変更せず、保存済み加工CSVから単独で開始した。
Step Bのfigure_visual_qa.mdには4図を開いて確認済みとの記録があり、EDAの再実行は不要だった。
VIXや追加特徴量は推定に使っていない。モデル選択、外れ値除去、変換、再学習も行っていない。

この単純なモデルを使う目的は株価予測精度の最大化ではなく、FixedとRollingの共通の区間中心を用意することである。
点予測モデルを固定し、後続で変わるcalibration historyの違いを比較できるようにする。
OLSはTraining内の二乗誤差和
`SSE = sum((return_pct_t - beta0 - beta1 * return_lag1_pct_t)**2)`
を最小にするbeta0とbeta1を選ぶ。

## コードの対応と実行

実行入口は`11_fit_ols_baseline.py`。PowerShellで以下を使用する。

```powershell
& "C:\\Users\\sohas\\anaconda3\\python.exe" .\\11_fit_ols_baseline.py
```

コード内の番号付きコメントで処理を区別している。

- 1：入力・期間・ラグ時点の確認。最初の説明変数は2016-12-30の既知リターンであり、2016年の目的変数は学習に追加しない。
- 2：Trainingの`training[[FEATURE]]`に定数項を追加し、`sm.OLS(...).fit()`を1回だけ実行する。
- 3：係数を`models/ols/fixed_coefficients.json`に保存し、読み直す。
- 4：Trainingの当てはめ値と残差を保存する。
- 5：読み直した同じ係数でCalibration・Testの点予測を計算する。`model.predict`は数値照合にだけ使う。
- 6：Training残差だけを図示する。3図をPNG 300dpi・PDFで保存する。

使用環境・入力CSV・設計lock・ソース・コードのハッシュと実行日時は固定係数JSONおよび`ols/execution_metadata.json`に記録した。
追加インストールは行っていない。今回の再実行はOLSを再度推定するため、後続のStep Dではこのスクリプトを実行せず、固定係数JSONと保存済み点予測を読み込む。

## 推定結果と解釈（Training内）

`fitted_return_pct = {intercept:.12f} + ({ar1_coefficient:.12f}) * return_lag1_pct`

| 項目 | 値 |
|---|---:|
| 定数項 beta0 | {intercept:.12f} |
| ラグ係数 beta1 | {ar1_coefficient:.12f} |
| 観測数 | {int(model.nobs)} |
| パラメータ数（定数項を含む） | 2 |
| 残差自由度 | {int(model.df_resid)} |
| R-squared | {model.rsquared:.8f} |
| Adjusted R-squared | {model.rsquared_adj:.8f} |
| 残差標準誤差（percentage points） | {residual_standard_error:.8f} |

定数項は前市場日のリターンが0%のときの条件付き平均の推定値で、単位は百分率対数リターン。
ラグ係数は{coefficient_sign}で、前市場日リターンが1パーセントポイント高い場合の予測値の変化を示す。
これはTraining内で推定した線形な関係であり、因果効果や将来の売買収益を示さない。
R-squaredはTraining内のリターン変動の約{model.rsquared * 100:.3f}%を、この定数項＋ラグの当てはめが説明するという記述である。
Adjusted R-squaredはパラメータ数を考慮した指標であり、どちらもCalibration/Testの予測精度ではない。

残差標準誤差は`sqrt(SSE / (n - 2))`で、n=502・分母500。
残差の標本標準偏差（分母n-1）とは区別する。statsmodelsの`mse_resid`の平方根と一致することを確認した。
係数表の通常OLS標準誤差・t値・p値・95%係数信頼区間は、誤差の無相関・等分散などの仮定に依存する。
通常の有限標本t推論では正規性等も前提になるため、金融時系列で数値を無条件に信頼せず、有意性を本研究の主要結論にしない。
係数の信頼区間はConformalの予測区間ではない。HACや追加の係数検定は導入していない。
statsmodels標準要約にはOmnibus/JB等の診断欄が自動掲載されるが、追加の検定課題や研究上の判定には使っていない。

## Training残差

Training残差は`return_pct - fitted_return_pct`、すなわち推定に使ったデータ上の当てはめ誤差である。
2019年のout-of-sample Calibration scoreではなく、Fixed/Rollingのscore履歴へ流用しない。
残差の平均は{training_prediction_residuals.mean():.3e}、標本標準偏差は{training_prediction_residuals.std(ddof=1):.6f}、
最小{training_prediction_residuals.min():.6f}、最大{training_prediction_residuals.max():.6f}、歪度{training_prediction_residuals.skew():.3f}、超過尖度{training_prediction_residuals.kurt():.3f}。
平均がほぼ0なのは定数項を含むOLSの性質であり、将来の不偏性の保証ではない。
超過尖度は正規分布を0とする指標で、極端値の影響を受ける。正規性検定の結果ではない。

時系列・ヒストグラムは極端な残差も含めて表示した。
残差と残差二乗のACFは20ラグまで計算し、図ではlag 1〜20を示す。lag 0の1はCSVにのみ含めた。
残差のlag 1 ACFは{residual_acf[1]:.4f}、残差二乗のlag 1 ACFは{squared_residual_acf[1]:.4f}。
残差二乗のlag 1〜20で絶対値が最大なのはlag {largest_squared_lag}（{squared_residual_acf[largest_squared_lag]:.4f}）。
残差二乗は符号を取り除いた誤差の大きさの時間的な集中を見るために用いる。
ACFは平均を引いた系列の記述的な自己相関（adjusted=False）であり、信頼帯・p値は付けていない。
図から独立性・正規性・等分散性を証明したとは解釈しない。
残差の大きさや依存が問題になっても、主設計を結果に合わせて変更せず限界として記録する。
図の実際の目視確認は別途`ols/figure_visual_qa.md`に記録する。

## 固定係数による点予測（out-of-sample）

Calibration 252行・Test 1,658行を、Trainingで固定した同じ係数で予測した。
予測式の入力は前SP500市場日リターンのみで、当日の実測リターンやVIXを使っていない。
Calibration/Testでのfit呼出しはなく、誤差・絶対誤差・score・予測精度指標も生成していない。
予測値は百分率対数リターンであり、日付ごとの価格水準の予測ではない。

予測CSVには対象日date、予測元日forecast_origin_date、split、return_lag1_pct、predicted_return_pctの5列を保存した。
Training内の当てはめ値とout-of-sampleの点予測を別ファイル・別の列名で区別している。
後続では元の加工CSVと日付をone-to-oneで結合し、予測元日・splitも一致することを確認する。

## 成果物と検証

- `models/ols/fixed_coefficients.json`：再推定せず読み込む固定係数と推定条件。
- `reports/ols/coefficient_table.csv`、`model_fit_statistics.json`、`statsmodels_summary.txt`：回帰結果。
- `reports/ols/training_fitted_residuals.csv`、`training_residual_statistics.csv`、`training_residual_acf.csv`：Training内の結果。
- `reports/ols/calibration_point_predictions.csv`、`test_point_predictions.csv`：out-of-sample点予測。
- `figures/ols/`：Training残差の3図、各PNG・PDF。
- `reports/ols/validation_results.json`、`existing_file_preservation.csv`、`execution.log`：検証・保全・実行記録。

推定日付がTrainingの502日のみであること、ラグが前SP500市場日のリターンであること、各予測元日が対象日より前であることを確認した。
statsmodels予測と固定係数の直接計算、保存JSON再読込による計算、保存CSV再読込が許容差rtol=atol=1e-12で一致した。
入力CSV・設計・既存Step A/B成果物のSHA-256を実行前後で照合し、変更していないことを確認した。

## 残る限界とStep Dへの状態

この段階では予測区間の信頼性やFixed/Rollingの優劣を判断できない。
Calibrationの高VIXが0日でも、それだけでFixedのCoverage低下を実証したことにはならない。
外部取引所calendarとの全件照合と、当時の公表遅延・改訂を含むvintage再現は未実施の制約として残る。
データ・保存係数・点予測の整合性に未解決の問題がなければ、Step Dで2019年のout-of-sample誤差からscoreを作成できる状態である。
今回はConformal score、quantile、Fixed/Rolling区間、Coverage・Width・Interval Score、bootstrapへ進んでいない。
'''
    (PROJECT_DIR / 'reports/step_c_summary.md').write_text(summary, encoding='utf-8')

    preservation_rows = []
    for path, before in hashes_before.items():
        after = file_hash(path)
        preservation_rows.append({
            'path': path.relative_to(PROJECT_DIR).as_posix(), 'sha256_before': before,
            'sha256_after': after, 'unchanged': before == after,
        })
    preservation = pd.DataFrame(preservation_rows)
    assert preservation['unchanged'].all(), '既存Step A/Bファイルに変更があります。'
    preservation.to_csv(REPORT_DIR / 'existing_file_preservation.csv', index=False)
    validation = {
        'status': 'PASS', 'ols_fit_calls': 1, 'training_n': 502,
        'training_only_dates': [training['date'].min().strftime('%Y-%m-%d'), training['date'].max().strftime('%Y-%m-%d')],
        'exog_names': model.model.exog_names, 'prior_market_day_lags_checked': True,
        'first_training_predictor_source_date': first_origin.strftime('%Y-%m-%d'),
        'first_training_predictor_matches_saved_raw_warmup_return': True,
        'rse_matches_sqrt_sse_over_n_minus_2': True, 'predictions': prediction_checks,
        'prediction_csv_columns': prediction_columns + ['predicted_return_pct'],
        'prediction_errors_or_scores_generated': False, 'coefficient_json_reload_prediction_check': 'PASS',
        'comparison_tolerance': {'rtol': RTOL, 'atol': ATOL},
        'existing_files_unchanged': len(preservation), 'step_a_b_reexecuted': False,
    }
    (REPORT_DIR / 'validation_results.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata = {
        'executed_at_utc': coefficient_record['executed_at_utc'], 'executed_at_jst': coefficient_record['executed_at_jst'],
        'environment': environment, 'working_directory': str(Path.cwd()), 'project_directory': str(PROJECT_DIR),
        'input_csv_sha256': file_hash(INPUT_PATH), 'research_design_lock_sha256': file_hash(LOCK_PATH),
        'research_design_source_sha256': file_hash(SOURCE_PATH), 'code_sha256': file_hash(Path(__file__).resolve()),
        'fixed_coefficient_json_sha256': file_hash(COEFFICIENT_PATH), 'figures': figure_files,
        'scope': 'OLS fit once on Training; Training residuals; frozen point predictions for Calibration and Test only',
    }
    (REPORT_DIR / 'execution_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print('事前確認:\n' + pd.DataFrame(split_rows).to_string(index=False))
    print('\n係数表（通常OLS標準誤差は誤差の仮定に依存）:\n' + coefficient_table.to_string(index=False))
    print('\nTraining推定統計:\n' + json.dumps(fit_statistics, ensure_ascii=False, indent=2))
    print('\n点予測の検証:\n' + json.dumps(prediction_checks, ensure_ascii=False, indent=2))
    print(f'既存ファイル保全: {len(preservation)}件不変。Step A/Bは再実行していません。')
    print('Step C: PASS。score・予測区間・区間評価は未実施。図の目視確認は別途実施。')


if __name__ == '__main__':
    if 'ANALYSIS_WORKSPACE' not in globals():
        raise SystemExit('Use run_analysis.py run --workspace <prepared copy> --stage 11.')
    fit_ols_baseline(ANALYSIS_WORKSPACE)
