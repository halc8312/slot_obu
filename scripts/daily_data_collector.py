"""
GitHub Actions用 - 日次データ収集スクリプト
軽量・高速・エラー耐性を重視
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import json

def collect_pachislot_data(date):
    """
    指定日のパチスロデータを収集（ダミー実装）
    実際の実装では適切なデータソースから取得
    """
    print(f"Collecting data for {date}...")
    
    # ダミーデータ生成（実際はWebスクレイピング）
    machines = []
    for machine_num in range(1, 641):
        machines.append({
            'date': date,
            'machine_number': machine_num,
            'total_games': 3000 + (machine_num * 10) % 2000,
            'payout_rate': 92 + (machine_num % 20) * 0.5,
            'max_payout': 5000 + (machine_num * 100) % 3000,
            # 他の特徴量...
        })
    
    return pd.DataFrame(machines)

def validate_data(df):
    """データの妥当性チェック"""
    # 欠損値チェック
    if df.isnull().sum().sum() > 0:
        print(f"Warning: Found {df.isnull().sum().sum()} missing values")
        df = df.fillna(method='ffill').fillna(0)
    
    # 異常値チェック
    df = df[df['payout_rate'] > 0]
    df = df[df['payout_rate'] < 200]
    
    return df

def main():
    # 出力ディレクトリ作成
    os.makedirs('data', exist_ok=True)
    
    # 昨日のデータを収集（ホールの集計が完了している）
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # データ収集
        df = collect_pachislot_data(yesterday)
        
        # データ検証
        df = validate_data(df)
        
        # 保存
        output_path = f'data/daily_{yesterday}.csv'
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        # 統計情報を記録
        stats = {
            'date': yesterday,
            'total_machines': len(df),
            'avg_payout_rate': df['payout_rate'].mean(),
            'collection_time': datetime.now().isoformat()
        }
        
        with open('data/collection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
    except Exception as e:
        print(f"Error collecting data: {e}")
        # エラーでもワークフローは継続
        exit(0)

if __name__ == "__main__":
    main()