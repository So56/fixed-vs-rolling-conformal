# テストの使い方

## 公開フォルダだけで動くテスト

公開フォルダを作業場所として実行する。

```powershell
python -X utf8 -B tests/test_entry_points.py
```

研究データを使わず、全12モジュールのimport時に分析・出力・通信を行わないこと、既存／入れ子出力先の拒否、prepare markerのない原本実行の拒否を確認する。小さな仮入力と出力はOSの一時フォルダ内に限定する。数値的な研究再現を証明するテストではない。

## 保存結果・実行traceを使う詳細テスト

`test_regression.py`はStep Aで確立した12テストを変更せず保存したもの。必要な保存snapshotとtraceは公開コードに含まれていない。`unittest discover`だけで全研究結果を再現できる構成ではない。

原プロジェクト側で利用する場合は、対応stageの`refactor_validation/step_*/run_tests.py`を使う。今回の最終確認は次のコマンド。

```powershell
python -X utf8 -B refactor_validation/step_h/run_tests.py
```

この監査コマンドは既存出力を上書きせず、結果ファイルが既に存在すると停止する。独立した再監査は新しい監査フォルダへ作成する。Step H報告とSHA manifestを参照する。

詳細テストは、日付・分類・件数・整数・indexを完全一致、浮動小数をround_trip読込と許容差0で照合する。元からあるpandas/NumPyの平均等の演算経路差を、旧新版の差と区別する。検証の追加drawを本分析の生成器へ行わない。

Step Hは現在の10/19を旧版・新関数・CLIで再実行した証拠、OLSはStep D、主Rolling252はStep E、Bootstrap/感度分析はStep G、追加MBB/SB/HACはStep Aの既存証拠を区別する。上流保存値はStep Hのコピーでも照合するが、Step Hで上流を再計算したとは扱わない。メモリ上の誤り注入は想定したAssertionErrorの理由まで確認し、監査原本へ書き込まない。

実装の再現性、MBBの推論条件、将来Coverage保証は別である。過去Width/IS報告の原因不明という留保は維持する。
