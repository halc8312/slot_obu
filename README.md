# スロット予測システム

パチスロの出率を予測する自動化システムです。

## 🎰 機能

- 毎日自動でデータ収集・予測実行
- 特別日（1の付く日）の高度な予測（単純な補正ではなく、学習済みパターンを活用）
- 任意の日付を指定して予測可能 ⭐ NEW!
- Web上で予測結果を確認可能
- 完全無料で運用可能
- 実際の機種名表示対応（2025年7月版）

## 📊 予測結果

最新の予測結果はこちら：
https://halc8312.github.io/slot_obu/

## 🎯 日付指定予測の使い方

### 基本的な使い方

```bash
# 明日の予測
python predict_any_date.py

# 特定の日付を予測
python predict_any_date.py 2025-08-01

# 次の「1の付く日」を予測
python predict_any_date.py --next-special

# 今後1週間を予測
python predict_any_date.py --week

# 特定月の1の付く日を全て予測
python predict_any_date.py --month 2025-08
```

### オプション

- `--top N`: 表示する上位台数を指定（デフォルト: 30）
- `--visualize`: 予測結果のグラフを作成
- `--simple`: シンプルな出力形式

### 特別日（1の付く日）について

本システムは13ヶ月分の実データで学習済みのため、「1の付く日」の高出率パターンを自然に予測します。
単純な一律補正ではなく、機種ごとの特性や曜日との組み合わせ効果も考慮した高度な予測を行います。

## 🔧 技術スタック

- Python (PyTorch, Pandas)
- Transformer-LSTM ハイブリッドモデル
- Quantile Loss（MAE: 15.7957%）
- GitHub Actions（自動実行）
- GitHub Pages（結果公開）

## 📈 予測精度

- 平均絶対誤差（MAE）: 15.7957%
- 特別日効果: 平均+11.4%（学習済み）
- 機種別・曜日別パターン: 自動学習

## 📝 ライセンス

MIT License