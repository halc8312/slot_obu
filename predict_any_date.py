#!/usr/bin/env python
"""
任意の日付を予測する簡単なインターフェース
使用例:
    python predict_any_date.py                    # 明日を予測
    python predict_any_date.py 2025-08-01        # 2025年8月1日を予測
    python predict_any_date.py --next-special    # 次の「1の付く日」を予測
"""

import sys
import os
from datetime import datetime, timedelta
import argparse

# スクリプトディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def find_next_special_day(from_date: datetime = None) -> datetime:
    """次の「1の付く日」を見つける"""
    if from_date is None:
        from_date = datetime.now()
    
    current = from_date
    while True:
        if current.day in [1, 11, 21, 31]:
            if current > from_date:
                return current
        current += timedelta(days=1)
        
        # 月末の処理
        if current.day == 1 and current > from_date:
            return current

def main():
    parser = argparse.ArgumentParser(
        description='パチスロ出率予測 - 任意の日付を指定可能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s                     # 明日を予測
  %(prog)s 2025-08-01         # 特定の日付を予測
  %(prog)s --next-special     # 次の1の付く日を予測
  %(prog)s --week             # 今後1週間を予測
  %(prog)s --month 2025-08    # 特定の月の1の付く日を全て予測
"""
    )
    
    parser.add_argument('date', nargs='?', help='予測日付 (YYYY-MM-DD形式)')
    parser.add_argument('--next-special', action='store_true', 
                       help='次の「1の付く日」を予測')
    parser.add_argument('--week', action='store_true', 
                       help='今後1週間の予測')
    parser.add_argument('--month', type=str, 
                       help='指定月の1の付く日を全て予測 (YYYY-MM形式)')
    parser.add_argument('--top', type=int, default=30, 
                       help='表示する上位台数 (デフォルト: 30)')
    parser.add_argument('--visualize', action='store_true', 
                       help='グラフを作成')
    parser.add_argument('--simple', action='store_true', 
                       help='シンプルな出力')
    
    args = parser.parse_args()
    
    # 予測対象日の決定
    target_dates = []
    
    if args.next_special:
        # 次の特別日
        next_special = find_next_special_day()
        target_dates.append(next_special)
        print(f"\n次の「1の付く日」: {next_special.strftime('%Y年%m月%d日 (%a)')}")
        
    elif args.week:
        # 今後1週間
        today = datetime.now().date()
        for i in range(7):
            target_dates.append(datetime.combine(today + timedelta(days=i+1), datetime.min.time()))
        print("\n今後1週間の予測を実行します")
        
    elif args.month:
        # 指定月の1の付く日
        try:
            year, month = map(int, args.month.split('-'))
            for day in [1, 11, 21, 31]:
                try:
                    date = datetime(year, month, day)
                    target_dates.append(date)
                except ValueError:
                    # 31日が存在しない月
                    continue
            print(f"\n{year}年{month}月の「1の付く日」を予測します")
        except:
            print("エラー: 月の指定は YYYY-MM 形式で入力してください")
            return
            
    elif args.date:
        # 指定日付
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
            target_dates.append(target_date)
        except ValueError:
            print("エラー: 日付は YYYY-MM-DD 形式で入力してください")
            return
    else:
        # デフォルトは明日
        tomorrow = datetime.now() + timedelta(days=1)
        target_dates.append(tomorrow)
    
    # 各日付について予測を実行
    for target_date in target_dates:
        print(f"\n{'='*70}")
        print(f"予測日: {target_date.strftime('%Y年%m月%d日 (%a)')}")
        
        # 特別日チェック
        if target_date.day in [1, 11, 21, 31]:
            print("★★★ 特別日（1の付く日）です！高出率が期待できます ★★★")
        
        print(f"{'='*70}")
        
        # 予測実行
        if args.simple or len(target_dates) > 1:
            # シンプルモード（複数日付の場合は自動的にシンプル）
            from predict_specific_date import predict_for_date
            results = predict_for_date(target_date, top_n=args.top, save_csv=(len(target_dates)==1))
            
            if len(target_dates) > 1 and results is not None:
                # 複数日付の場合は要約のみ
                high_payout = len(results[results['predicted_payout'] >= 105])
                print(f"  → 105%以上: {high_payout}台, "
                      f"平均: {results[results['status']=='稼働中']['predicted_payout'].mean():.1f}%")
        else:
            # 詳細モード
            from predict_with_real_data import SpecificDatePredictor
            
            predictor = SpecificDatePredictor()
            predictor.load_model()
            
            # 過去データ読み込み
            historical_df = predictor.load_historical_data(target_date, days_back=60)
            
            # 予測
            results = predictor.predict_date(target_date, historical_df)
            
            # 結果表示
            print(f"\n【TOP {args.top} 予測結果】")
            print(f"{'順位':<4} {'台番':<6} {'機種名':<25} {'予測出率':<10} {'評価'}")
            print("-" * 60)
            
            for _, row in results.head(args.top).iterrows():
                machine_type_short = row['machine_type'][:23] + '..' if len(row['machine_type']) > 25 else row['machine_type']
                print(f"{row['rank']:<4} {row['machine_number']:<6} {machine_type_short:<25} "
                      f"{row['predicted_payout']:>7.1f}% {row['evaluation']}")
            
            # 統計情報
            print(f"\n【統計情報】")
            print(f"予測台数: {len(results)}台")
            print(f"平均予測出率: {results['predicted_payout'].mean():.1f}%")
            print(f"110%以上: {len(results[results['predicted_payout'] >= 110])}台")
            print(f"105%以上: {len(results[results['predicted_payout'] >= 105])}台")
            print(f"100%以上: {len(results[results['predicted_payout'] >= 100])}台")
            
            # 可視化
            if args.visualize:
                predictor.visualize_predictions(results, target_date)
                print(f"\nグラフを保存しました: prediction_analysis_{target_date.strftime('%Y%m%d')}.png")
            
            # CSV保存
            if len(target_dates) == 1:
                filename = f"predictions_{target_date.strftime('%Y%m%d')}.csv"
                results.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"\n予測結果を保存しました: {filename}")
    
    # 特別日の場合の追加アドバイス
    special_dates = [d for d in target_dates if d.day in [1, 11, 21, 31]]
    if special_dates:
        print(f"\n{'='*70}")
        print("【特別日（1の付く日）攻略アドバイス】")
        print("1. 朝一からの立ち回りが重要")
        print("2. 上位予測台は争奪戦になる可能性大")
        print("3. データカウンターで実績を確認しながら台選び")
        print("4. 高設定示唆が出たら粘り強く")
        print("5. 投資上限を決めて計画的に")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()