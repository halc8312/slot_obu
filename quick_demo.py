"""
日付指定予測のクイックデモ
"""
import sys
from datetime import datetime, timedelta

print("=== スロット予測システム - 日付指定予測デモ ===")
print()

# 明日の日付
tomorrow = datetime.now() + timedelta(days=1)
print(f"明日 ({tomorrow.strftime('%Y年%m月%d日')}) の予測例：")
print()

# 特別日チェック
if tomorrow.day in [1, 11, 21, 31]:
    print("★★★ 明日は特別日（1の付く日）です！ ★★★")
    print("高出率が期待できます！")
else:
    print("明日は通常日です")

print()
print("予測結果の例：")
print("-" * 70)
print("順位  台番号  機種名                     予測出率    評価")
print("-" * 70)

# サンプルデータ
sample_predictions = [
    (1, 256, "Lスマスロ北斗の拳", 112.3, "★★★"),
    (2, 412, "アイムジャグラーEX-TP", 108.9, "★★"),
    (3, 123, "Lゴジラ対エヴァンゲリオン", 107.5, "★★"),
    (4, 89, "押忍!番長ZERO", 106.8, "★★"),
    (5, 334, "バジリスク絆2天膳", 105.9, "★★"),
]

for rank, machine, name, payout, eval in sample_predictions:
    print(f"{rank:<6}{machine:<8}{name:<28}{payout:>7.1f}%  {eval}")

print()
print("=" * 70)
print()
print("実際の予測を実行するには：")
print("  python predict_any_date.py")
print()
print("次の「1の付く日」を予測：")
print("  python predict_any_date.py --next-special")
print()
print("特定の日付を予測：")
print("  python predict_any_date.py 2025-08-01")
print()
print("詳細は README.md をご覧ください。")