# Fixed and Rolling Conformal Calibration

研究テーマ：How Do Fixed and Rolling Conformal Calibration Differ in Prediction Interval Coverage and Width During High-VIX Market Conditions?

保存済みFRED snapshotから前処理、Training-only OLS AR(1)、Fixed/Rolling区間、評価、paired Moving Block Bootstrap、感度分析、予備結果統合を順に実行するPythonコード。研究原本とは別の公開用コピーである。入力snapshotは付属していない。このフォルダだけでデータ取得や全分析を開始することはできない。

## 構成

```text
github_release/
  README.md
  requirements.txt
  run_analysis.py                 # 独立workspaceの準備と実行入口
  analysis/
    prepare_data.py               # 旧01〜09の前処理・監査を実行順に統合
    10_exploratory_data_analysis.py
    11_fit_ols_baseline.py
    12_build_fixed_conformal.py
    13_build_rolling_conformal.py
    14_evaluate_prediction_intervals.py
    15_block_bootstrap_evaluation.py
    16_block_length_sensitivity.py
    17_vix_threshold_sensitivity.py
    18_rolling_window_sensitivity.py
    19_compile_preliminary_results.py
  tests/
    README.md
    test_entry_points.py           # データ不要の入口・保護テスト
    test_regression.py             # 保存snapshot・監査traceを使う詳細テスト
```

各スクリプトはworkspaceを受け取る明示的な実行関数を持つ。importだけでは分析やファイル出力を行わない。短い統計関数は処理を説明しやすい位置に残した。クラス、共通エンジン、common.pyは追加していない。

## 環境と入力

Pythonと`requirements.txt`の固定バージョンを使う。必要な場合は別の仮想環境へインストールする。

```powershell
python -m pip install -r requirements.txt
python -X utf8 -B tests/test_entry_points.py
```

検証で使用したPython・ライブラリの実バージョンは、研究原本側の`refactor_validation/step_h/environment.json`に記録する。異なる環境で完全精度の一致が得られるとは仮定しない。

入力は既存研究の保存snapshot。最低限の生データだけでは実行できず、次の関連ファイルを含む必要がある。

- `data/raw/SP500_fred.csv`、`VIXCLS_fred.csv`、`download_manifest.json`。
- `docs/research_design_lock.json`、そのSHAと対応する`final_research_design_v1_source.md`、変数定義・既存確認記録。
- 選択したstageが読む`data/processed/`、`models/`、`reports/`、`figures/`の保存済み上流成果物。
- 旧スクリプト・`scripts/`・`requirements.txt`・`file_inventory.csv`等の既存保全対象。19は`reports/preliminary_summary/methods_limitations_ja.md`と`results_discussion_ja.md`も読む。

各stageは必要なファイルと設計lockをassertで検査する。入力を省略して検査を外す運用はしない。`prepare_data.py`の`REFRESH_RAW=False`を固定し、CLIから変更できない。保存snapshotだけを使い、外部から再取得しない。

## 実行方法

公開フォルダを作業場所として、入力の外側にある**存在しない**出力先を指定する。

```powershell
python -X utf8 -B run_analysis.py prepare --input-dir "C:/path/to/saved_snapshot" --output-dir "C:/path/to/new_workspace"
```

この`prepare`コマンドは複写だけを行う。次の`--stage prepare`は前処理を実行する別の操作である。入力先との入れ子、公開コードフォルダへの出力、既存出力先、外部へのリンクを拒否する。`.git`、ブラウザprofile、vendor、bytecodeは複写しない。分析は新workspaceのコピーを更新する。

| 順序 | analysis内のコード／実行関数 | 主な入力 | workspace内の主な出力 |
|---|---|---|---|
| prepare | prepare_data.py / prepare_data | 保存FRED、設計lock | data/processed、reportsのデータ監査、docs変数定義 |
| 10 | 10_exploratory_data_analysis.py / run_exploratory_analysis | 加工CSV、変数定義 | reports/eda、figures/eda、reports/step_b_summary.md |
| 11 | 11_fit_ols_baseline.py / fit_ols_baseline | 加工CSV、EDA確認記録 | models/ols、reports/ols、figures/ols |
| 12 | 12_build_fixed_conformal.py / build_fixed_conformal | 保存OLS、Calibration予測 | models/fixed_conformal、reports/fixed_conformal |
| 13 | 13_build_rolling_conformal.py / build_rolling_conformal | 保存OLS、2019 score、Fixed設定 | reports/rolling_conformal、figures/rolling_conformal |
| 14 | 14_evaluate_prediction_intervals.py / evaluate_prediction_intervals | Fixed/Rolling区間、共通Test日 | reports/evaluation、figures/evaluation |
| 15 | 15_block_bootstrap_evaluation.py / run_block_bootstrap | 日次評価の全Test系列 | reports/bootstrap、figures/bootstrap |
| 16 | 16_block_length_sensitivity.py / run_block_length_sensitivity | 日次評価、主MBB | reports/bootstrap_sensitivity、figures/bootstrap_sensitivity |
| 17 | 17_vix_threshold_sensitivity.py / run_vix_threshold_sensitivity | 日次評価、主MBB | reports/vix_threshold_sensitivity、figures/vix_threshold_sensitivity |
| 18 | 18_rolling_window_sensitivity.py / run_rolling_window_sensitivity | 保存OLS予測・score、主MBB | reports/rolling_window_sensitivity、figures/rolling_window_sensitivity |
| 19 | 19_compile_preliminary_results.py / compile_preliminary_results | 保存集計・CI・図・既存下書き | reports/preliminary_summary、figures/preliminary_summary |

全分析を再現する場合は`prepare`stage→10→…→19の順。まとめて実行する明示的コマンドは次の通り。**OLS・区間・Bootstrapも再計算する。**

```powershell
python -X utf8 -B run_analysis.py run --workspace "C:/path/to/new_workspace" --stage all
```

保存済み結果を整理するだけなら、上流ファイルが揃った別workspaceで19のみを実行する。19はOLS・区間履歴・Bootstrap・新規検定を再計算しない。

```powershell
python -X utf8 -B run_analysis.py run --workspace "C:/path/to/new_workspace" --stage 19
```

EDAだけなら`--stage 10`。他の単独stageも同じ形式。stageの一部だけを再実行すると、後続に複写された旧成果物は新実行の結果にはならない。例えば11の再実行は係数が一致しても時刻・コードSHAを含むJSONのhashを更新する。対応する後続stageを実行せず旧入力hash記録を現行の証拠として扱わない。

`_workspace_manifest.json`は複写時点のファイルSHA。`_execution_log.json`は実際に使った公開コードのパス・SHA・順序。workspaceに残る旧番号付きコードは入力snapshotの記録であり、CLIの実行対象ではない。キャッシュ・一時ファイルはworkspaceの`.runtime/`へ置く。各stageの描画設定を独立させる。

## 保持した統計仕様

- 有効市場日上で`100 * log(close).diff()`を作り、その後に期間を切る。リターンとVIXのラグは前SP500有効市場日。欠損補間や外れ値削除は行わない。
- Training502日（2017–2018）の定数項付きOLS AR(1)を1回推定し、係数を固定する。Calibration252日（2019）とTest1,658日（2020-01-02–2026-08-07）で再推定しない。
- Scoreは絶対予測誤差。名目90%、`k = ceil((n+1)*0.90)`番目の順序統計量を補間せず選ぶ。Fixedは2019年252 scoreから`k=228`。Testを使ってqを変更しない。
- Rollingは直前252個のscoreを日付順に保持する。区間を確定してから当日のscoreを追加し、最古を除く。初日のqはFixedと一致する。252暦日ではない。
- High-VIXは前SP500市場日のVIX≥30。全TestのHigh149日、Non-high1,509日。連続High25 episodeは市場日系列上の連続区分であり、独立な25ショックとは解釈しない。
- Coverageは両端を含む区間内指標の平均。Gapは`100*(coverage−0.90)`（coverage percentage points）。Widthは上限−下限。ISはWidthに区間外距離の`2/alpha=20`倍を加える。Width/ISはreturn percentage points。paired差は同日のRolling−Fixedを平均し、別々の平均の差へ計算経路を置き換えない。
- 主MBBは全Test1,658日から非循環の重複する連続blockを復元抽出する。L10、B5,000、seed202701、`numpy.random.default_rng`/PCG64。開始位置は`integers(0,N−L+1,size=ceil(N/L),dtype=int64)`、連結して先頭N行を使う。同じindexで両手法とHigh分類を抽出し、日付再ソート・High-only圧縮はしない。再標本のHigh件数は可変。
- CIはfinite反復値の2.5/97.5 percentile、`np.quantile(method='linear')`。空群はNaNで再抽出しない。各CIは個別95%で、同時CI・多重比較調整済みではない。
- 感度分析はL5/10/20、VIX25/30/35、Rolling63/126/252。主設定を変更しない。16はL5/20それぞれ同じseedへ初期化。17は25/35、18は63/126で同じindexを共有する。主設定の保存値を再利用する。各本反復5,000回に加え、別生成器によるprefix3回の元検証を残す。

## 出力・文書の扱い

CSVは完全精度を保持し、Markdown表の表示だけを丸める。19の`source_cell_mapping.csv`は数値セル群を元CSV行キー／JSON pathとSHAへ対応させ、`source_catalog.csv`と`figure_catalog.csv`は入力・図の出所を記録する。FigureのCoverage差とWidth/IS差は単位を分ける。

19が生成するものは**既存の予備要約**であり、論文最終版ではない。現在の研究で承認された本文4表・3図／Appendix構成を自動的に確定する機能はない。`reports/thesis_final_results_v1/`は作成しない。既存2下書きの結合・表示を維持し、未承認本文や参考文献を追加しない。

一部の保存要約には元の生成コードに存在しない追記・修正がある。作成者・日時・操作の来歴は未確認。再実行した文書がその追記まで再現するとは扱わない。10/19の文書差もStep H監査で別記する。コード中・生成文中の旧「Step A〜F」は旧研究工程名であり、このリファクタリングA〜Hとは異なる。生成文の歴史的な直接実行コマンドより、本READMEのCLIを実行手順として使う。

## 検証と解釈の範囲

詳細は`tests/README.md`。元プロジェクトの`refactor_validation/step_a`〜`step_h`に、隔離した旧版／新版／CLI、差分、全数値照合、図の検証、原本保護SHA、失敗の診断を保存する。公開用コピーには大容量snapshot・監査traceを無条件に含めていない。

**実装の再現性**：完全精度のCSV・係数・予測・区間、Rolling履歴と当日score排除、Bootstrapの全開始位置・index・乱数state・反復統計量・CIを各対応stageで検証した。Step Hは10/19のみ再実行し、上流は保存値照合と過去実行証拠を区別する。PNGピクセル、PDF本文・描画も比較する。全stageを整理後の1プロセスで通す`--stage all`の再実行はStep Hでは行わない。

**統計的推論の妥当性**：コードの一致は定常性・弱依存やMBBの有効性の証明ではない。High-VIXのCoverage改善と幅の増加・IS低下、両手法ともHighで90%未達という既存の解釈を保持する。Overall IS差はL20でCIが0を含むため、block長に頑健とは説明しない。2020年除外で改善幅が縮小する点と時期集中を追加監査の限界として残す。追加監査のMBB40/60/120、Stationary Bootstrap、HACは本実行列の主分析とは別で、19に自動統合していない。

**将来市場への一般化可能性**：歴史的標本の実証CoverageとConformal Predictionの理論保証は区別する。Bootstrap CIは将来の予測区間のCoverage保証ではない。calendar・実際の公表時刻・FRED vintage等の時間整合性は追加監査を参照し、本リファクタリングで新たに確認済みとしない。

過去報告のOverall Width2.91/4.18・IS3.39/3.02と、現行保存値Width約2.470/3.742・IS約6.397/6.027との差は、原因不明という監査上の留保を維持する。リファクタリングの数値一致でこの不一致が解明されたとは扱わない。

## GitHub公開前

公開物のライセンス、FRED等のデータ再配布条件、匿名化・個人パス、文献と承認本文、再配布可能なsnapshot/テストfixtureの提供方法を確認する。詳細回帰テストには別配布の監査データが必要であり、コードだけで全再現可能と表示しない。研究原本のGit履歴はこのコピーへ移していない。commit・push・公開は実施していない。
