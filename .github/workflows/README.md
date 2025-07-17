# GitHub Actions ワークフロー

このディレクトリには、パチスロ予測システムの自動化ワークフローが含まれています。

## ワークフロー一覧

### 1. daily_prediction.yml
**毎日自動実行される標準予測**
- 実行時刻: 毎朝6:00（日本時間）
- 機能:
  - 翌日のパチスロ出率を予測
  - 特別日（1の付く日）の場合は自動で通知
  - 結果をGitHub Pagesで公開
- URL: https://halc8312.github.io/slot_obu/

### 2. predict_specific_date.yml
**任意の日付を予測（手動実行）**
- 実行方法: Actions タブから手動で実行
- 予測タイプ:
  - `single`: 特定の1日を予測
  - `week`: 今後1週間を予測
  - `month`: 指定月の1の付く日を全て予測
  - `special`: 次の1の付く日を予測
- 入力パラメータ:
  - `target_date`: 予測対象日（YYYY-MM-DD形式）
  - `prediction_type`: 予測タイプ
  - `month`: 月指定（YYYY-MM形式、monthタイプの場合のみ）

## 使用方法

### 特定の日付を予測する場合
1. GitHubリポジトリの「Actions」タブを開く
2. 「Predict Specific Date」を選択
3. 「Run workflow」をクリック
4. パラメータを入力:
   - prediction_type: `single`
   - target_date: `2025-08-01`（例）
5. 「Run workflow」ボタンをクリック

### 次の「1の付く日」を予測する場合
1. 同様にActionsタブから実行
2. パラメータ:
   - prediction_type: `special`
3. 実行後、結果がArtifactsに保存される

## 技術詳細

### モデルファイル
- GitHub Releasesから自動ダウンロード
- quantile_loss_model.pth (1.4MB)
- quantile_feature_scaler.joblib
- machine_master.csv（2025年7月版）

### 予測精度
- MAE: 15.7957%
- 特別日効果: 平均+11.4%（学習済み）
- 13ヶ月分の実データで学習

### 出力ファイル
- predictions_YYYYMMDD.csv: 予測結果
- prediction_analysis_YYYYMMDD.png: 分析グラフ
- report_YYYYMMDD.md: レポート

## 注意事項
- 無料のGitHub Actionsを使用（月2000分まで）
- 予測には約2-3分かかります
- 結果は30日間保存されます