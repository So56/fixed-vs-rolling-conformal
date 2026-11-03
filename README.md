# Fixed vs. Rolling Conformal Prediction

S&P 500の日次対数リターンに対して、残差の分位点を固定する **Fixed** と、直近の残差で更新する **Rolling** の予測区間を比較するプロジェクトです。前取引日のVIXによる市場区分ごとに、被覆率・区間幅・interval scoreを評価します。

## 分析の設定

点予測には前取引日のリターンを説明変数とするOLSを使い、係数は学習期間の推定値に固定します。Fixedは2019年の絶対予測誤差を使い続け、Rollingは予測日より前の252取引日の誤差で区間を更新します。

| 項目 | 設定 |
| --- | --- |
| 学習期間 | 2017–2018年（502取引日） |
| 校正期間 | 2019年（252取引日） |
| 評価期間 | 2020-01-02〜2026-10-28（1,715取引日） |
| 予測区間 | 名目被覆率90%（`alpha = 0.10`） |
| 高VIXの定義 | 前取引日のVIXが30以上 |
| 信頼区間 | 移動ブロック・ブートストラップによる95%区間 |
| ブートストラップ | ブロック長10、反復5,000回、乱数シード202701 |
| 感度分析 | ブロック長5・20、VIX閾値25・35、窓幅63・126 |

時系列上の被覆率を実測する分析です。名目被覆率90%は、各期間・市場区分での実際の被覆率を保証するものではありません。

## フォルダ構成

```text
Scripts/                   分析スクリプトと共通の評価関数
data/
  raw/                     入力CSV（SP500.csv、VIXCLS.csv）
  processed/               前処理済みの分析データ
outputs/
  modeling/                記述統計、OLS係数、点予測、予測区間
  evaluation/              日次・区分別評価、bootstrap/、sensitivity/
  reporting/               年別・連続高VIX期間の診断、tables/、figures/
reports/                   データ分割の確認、整理内容と改名一覧
reproduction/              再現実行ごとのコード・データ・ログ（Git対象外）
requirements.txt           依存パッケージの固定バージョン
```

CSVの値は、リターンと区間幅が%単位、被覆率が0〜1です。`difference` は **Rolling − Fixed** を表します。公開用の表・図では、被覆率を%（差はパーセントポイント）に換算します。

## 実行方法

Python 3.13を使用します。動作確認はWindows・Python 3.13.14で行っています。以下はプロジェクト直下のPowerShellで実行する例です。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Scripts/run_all.py
```

`run_all.py` は、コードと元データを `reproduction/run_..._utc/` にコピーして全工程を実行します。保存済みのCSVと許容誤差 `rtol = atol = 1e-10` で比較し、図の生成も確認します。元の `data/`・`outputs/`・`reports/` は上書きしません。正常終了すると実行先に `verification.json` と `comparison_results.csv` が保存されます。

`bootstrap/settings.csv` は実行日時などを含むため、再現確認では主要な計算設定を比較します。図はファイルの生成を確認し、画像内容の一致までは自動判定しません。

macOS・Linuxでは、仮想環境のPythonを `.venv/bin/python` に読み替えてください。これらの環境での動作は未確認です。

### 個別に実行する場合

次の順序で `python Scripts/<ファイル名>` を実行します。個別実行では、対応する `data/processed/`・`outputs/`・`reports/` の成果物を上書きします。

| 順序 | スクリプト | 処理 |
| --- | --- | --- |
| 01 | `01_prepare_data.py` | 終値の結合、対数リターン・ラグ変数の作成、期間分割 |
| 02 | `02_explore_data.py` | 取引日の照合、記述統計、分布図 |
| 03 | `03_fit_baseline.py` | OLSの推定、点予測、絶対予測誤差の保存 |
| 04 | `04_build_prediction_intervals.py` | Fixed・Rollingの区間作成と時間順序の検証 |
| 05 | `05_evaluate_prediction_intervals.py` | 日次・VIX区分別の評価 |
| 06 | `06_bootstrap_confidence_intervals.py` | ブートストラップによる信頼区間の計算 |
| 07 | `07_analyze_sensitivity.py` | 設定を変えた感度分析 |
| 08 | `08_diagnose_results.py` | 年別集計、連続高VIX期間、年除外、自己相関の診断 |
| 09 | `09_make_tables_and_figures.py` | 表と図の作成 |

`evaluation.py` は各スクリプトから読み込む共通関数、`run_all.py` は01〜09を順に実行する入口です。個別工程に当たらないため、この2ファイルには番号を付けていません。

## 入力データと公開前の確認

入力CSVは `data/raw/SP500.csv`（S&P 500終値）と `data/raw/VIXCLS.csv`（VIX終値）です。ネットワークからの取得処理はなく、同梱CSVから実行します。期間・件数の検証はこのデータに合わせてあるため、別のデータに差し替える際は設定と検証条件の見直しが必要です。

公開前に、元データの取得元・取得日・再配布条件を確認してください。コードのライセンスも未設定です。`reproduction/` と `.venv/` には端末固有のパスが含まれるため、Gitから除外しています。

今回の変更と不要候補は [整理報告](reports/project_cleanup.md)、全ファイルの改名対応は [file_renames.csv](reports/file_renames.csv) に記載しています。
# fixed-vs-rolling-conformal
