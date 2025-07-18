# 実データファイルのアップロード手順

## 重要：妥協なしの予測システムについて

このシステムは13ヶ月分の実データで学習されたモデルを使用しており、シミュレーションデータでは動作しません。
実データがない場合は予測を実行せず、エラーメッセージを表示します。

## 必要なファイル

予測を実行するには以下のファイルが必要です：

1. `final_integrated_13months_data.csv` (約23MB)
   - 13ヶ月分の全640台の実データ
   - モデルの学習に使用したものと同じ形式

## GitHub Releaseへのアップロード手順

### 方法1：Web UIを使用

1. https://github.com/halc8312/slot_obu/releases/tag/quantile-model-v1 にアクセス
2. 「Edit」ボタンをクリック
3. 「Attach binaries by dropping them here or selecting them」のエリアに`final_integrated_13months_data.csv`をドラッグ＆ドロップ
4. 「Update release」をクリック

### 方法2：GitHub CLIを使用

```bash
gh release upload quantile-model-v1 final_integrated_13months_data.csv --repo halc8312/slot_obu
```

## アップロード後の確認

1. Release ページで`final_integrated_13months_data.csv`がアセットに表示されることを確認
2. GitHub Actionsで「Predict Specific Date」を実行
3. エラーなく実行されることを確認

## 注意事項

- ファイルサイズが大きい（約23MB）ため、アップロードに時間がかかる場合があります
- GitHub Releaseの容量制限は2GBまでなので問題ありません
- 実データには個人情報が含まれていないことを確認してください

## なぜ実データが必要か

1. **モデルの特性**: Transformer-LSTMモデルは時系列パターンを学習しており、過去30日分のシーケンスが必要
2. **特別日効果**: 「1の付く日」の効果は実データのパターンから自然に学習されている
3. **機種別特性**: 各機種の特有の挙動は実データからのみ取得可能
4. **精度保証**: MAE 15.7957%の精度は実データでのみ保証される

シミュレーションデータでの動作は「妥協」であり、本システムの設計思想に反します。