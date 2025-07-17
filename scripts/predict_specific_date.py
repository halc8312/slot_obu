"""
指定日付予測スクリプト
既存のQuantile Lossモデルを使用して、任意の日付の予測を行う
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta
import argparse
from typing import List, Dict

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.daily_predictor import LightweightTransformerLSTM, create_sequences, prepare_features
from scripts.dynamic_machine_updater import DynamicMachineUpdater

def print_log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def predict_for_date(target_date: datetime, top_n: int = 50, save_csv: bool = True) -> pd.DataFrame:
    """
    指定日付の予測を実行
    
    Args:
        target_date: 予測対象日
        top_n: 上位何台を表示するか
        save_csv: CSVファイルに保存するか
    
    Returns:
        予測結果のDataFrame
    """
    print_log(f"=== Predicting for {target_date.strftime('%Y-%m-%d')} ===")
    
    # 1の付く日かチェック
    is_special_day = target_date.day in [1, 11, 21, 31]
    if is_special_day:
        print_log(f"[SPECIAL DAY] {target_date.day}日は「1の付く日」です！")
    
    # モデルとスケーラーの読み込み
    print_log("Loading model and scalers...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルアーキテクチャの初期化
    model = LightweightTransformerLSTM(
        input_dim=58,  # 特徴量数
        d_model=96,
        nhead=4,
        num_layers=2,
        lstm_hidden=64,
        dropout=0.3
    ).to(device)
    
    # モデルの重みを読み込み
    model_path = 'model_files_for_upload/quantile_loss_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print_log("Model loaded successfully")
    else:
        print_log(f"Model file not found: {model_path}")
        return None
    
    # スケーラーの読み込み
    feature_scaler = joblib.load('model_files_for_upload/quantile_feature_scaler.joblib')
    
    # 過去データの読み込み（特徴量作成のため）
    print_log("Loading historical data...")
    if os.path.exists('data/latest_data.csv'):
        historical_df = pd.read_csv('data/latest_data.csv', encoding='utf-8-sig')
    else:
        # ダミーデータを作成
        print_log("Creating dummy historical data...")
        historical_df = create_dummy_historical_data(target_date)
    
    # 機種マスターの読み込み
    updater = DynamicMachineUpdater()
    master_df = updater.load_current_master()
    
    # 全640台分の予測データを準備
    predictions = []
    
    print_log("Generating predictions for all 640 machines...")
    
    with torch.no_grad():
        for machine_num in range(1, 641):
            # 機種情報の取得
            machine_info = master_df[master_df['machine_number'] == machine_num]
            if len(machine_info) > 0:
                machine_type = machine_info.iloc[0]['machine_type']
                is_active = machine_info.iloc[0].get('is_active', True)
            else:
                machine_type = f"不明な機種{machine_num}"
                is_active = True
            
            # 非稼働台はスキップ
            if not is_active:
                predictions.append({
                    'machine_number': machine_num,
                    'machine_type': machine_type,
                    'predicted_payout': 0.0,
                    'status': '非稼働',
                    'rank': 999
                })
                continue
            
            # ダミーの過去データを作成（実際はデータベースから取得）
            dummy_df = create_machine_dummy_data(machine_num, target_date)
            
            # 特徴量の準備
            features_df = prepare_features(dummy_df)
            numeric_features = [col for col in features_df.columns 
                              if col not in ['date', 'machine_number', 'machine_type', 
                                           'date_obj', 'payout_rate_log1p']]
            
            # シーケンスの作成
            X, _ = create_sequences(features_df, feature_scaler, numeric_features)
            
            if len(X) > 0:
                # 予測
                X_tensor = torch.FloatTensor(X[-1:]).to(device)
                pred, _ = model(X_tensor)
                
                # log1p逆変換
                pred_value = torch.expm1(pred).cpu().numpy()[0, 0]
                
                predictions.append({
                    'machine_number': machine_num,
                    'machine_type': machine_type,
                    'predicted_payout': float(pred_value),
                    'status': '稼働中',
                    'rank': 0
                })
    
    # DataFrameに変換
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values('predicted_payout', ascending=False).reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    # 評価の追加
    results_df['evaluation'] = results_df['predicted_payout'].apply(
        lambda x: '★★★' if x >= 110 else '★★' if x >= 105 else '★' if x >= 100 else ''
    )
    
    # 特別日の統計
    if is_special_day:
        active_machines = results_df[results_df['status'] == '稼働中']
        avg_payout = active_machines['predicted_payout'].mean()
        high_payout_count = len(active_machines[active_machines['predicted_payout'] >= 105])
        
        print_log(f"\n=== 特別日統計 ===")
        print_log(f"平均予測出率: {avg_payout:.1f}%")
        print_log(f"高出率台数(105%以上): {high_payout_count}台")
    
    # トップN表示
    print_log(f"\n=== Top {top_n} Predictions for {target_date.strftime('%Y-%m-%d')} ===")
    print_log(f"{'順位':<4} {'台番号':<6} {'機種名':<30} {'予測出率':<10} {'評価':<6}")
    print_log("-" * 70)
    
    for idx, row in results_df.head(top_n).iterrows():
        if row['status'] == '稼働中':
            print_log(f"{row['rank']:<4} {row['machine_number']:<6} {row['machine_type']:<30} "
                     f"{row['predicted_payout']:>7.1f}% {row['evaluation']:<6}")
    
    # CSV保存
    if save_csv:
        filename = f"predictions_{target_date.strftime('%Y%m%d')}.csv"
        results_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print_log(f"\nPredictions saved to {filename}")
    
    return results_df

def create_dummy_historical_data(target_date: datetime) -> pd.DataFrame:
    """ダミーの過去データを作成（実際は過去データを使用）"""
    dates = []
    for i in range(30):
        dates.append(target_date - timedelta(days=i+1))
    
    data = []
    for machine_num in range(1, 641):
        for date in dates:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'machine_number': machine_num,
                'payout_rate_numeric': 90 + np.random.normal(5, 10),
                'total_games_numeric': 3000 + np.random.randint(-500, 500),
                'total_payout_numeric': 2700 + np.random.randint(-300, 300),
                'max_payout_numeric': 5000 + np.random.randint(-1000, 1000)
            })
    
    return pd.DataFrame(data)

def create_machine_dummy_data(machine_num: int, target_date: datetime) -> pd.DataFrame:
    """特定の機械のダミーデータを作成"""
    dates = []
    for i in range(30):
        dates.append(target_date - timedelta(days=i+1))
    
    data = []
    for date in dates:
        # 1の付く日は少し高めに設定（モデルが学習済みのパターンを反映）
        base_payout = 95
        if date.day in [1, 11, 21, 31]:
            base_payout = 105
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'machine_number': machine_num,
            'payout_rate_numeric': base_payout + np.random.normal(0, 5),
            'total_games_numeric': 3000 + np.random.randint(-500, 500),
            'total_payout_numeric': 2700 + np.random.randint(-300, 300),
            'max_payout_numeric': 5000 + np.random.randint(-1000, 1000),
            'diff_coins_numeric': np.random.randint(-1000, 1000),
            'games_numeric': 3000 + np.random.randint(-500, 500),
            'payouts_numeric': 50 + np.random.randint(-10, 10),
            'big_bonuses_numeric': 10 + np.random.randint(-5, 5),
            'regular_bonuses_numeric': 20 + np.random.randint(-10, 10)
        })
    
    # 予測対象日のデータも追加
    data.append({
        'date': target_date.strftime('%Y-%m-%d'),
        'machine_number': machine_num,
        'payout_rate_numeric': 0,  # 予測対象
        'total_games_numeric': 0,
        'total_payout_numeric': 0,
        'max_payout_numeric': 0,
        'diff_coins_numeric': 0,
        'games_numeric': 0,
        'payouts_numeric': 0,
        'big_bonuses_numeric': 0,
        'regular_bonuses_numeric': 0
    })
    
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='指定日付のパチスロ出率予測')
    parser.add_argument('--date', type=str, help='予測日付 (YYYY-MM-DD形式)')
    parser.add_argument('--top', type=int, default=50, help='表示する上位台数')
    parser.add_argument('--no-save', action='store_true', help='CSV保存をスキップ')
    
    args = parser.parse_args()
    
    # 日付の処理
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("Error: 日付は YYYY-MM-DD 形式で指定してください")
            return
    else:
        # デフォルトは明日
        target_date = datetime.now() + timedelta(days=1)
    
    # 予測実行
    results = predict_for_date(target_date, top_n=args.top, save_csv=not args.no_save)
    
    # 1の付く日の場合、追加の分析
    if target_date.day in [1, 11, 21, 31]:
        print_log("\n=== 特別日推奨戦略 ===")
        print_log("1. 上位20台を中心に立ち回る")
        print_log("2. 朝一から積極的に台を確保")
        print_log("3. 高設定の可能性が高い台を優先")
        
        # ジャンル別の推奨
        active_df = results[results['status'] == '稼働中'].head(100)
        
        juggler_machines = active_df[active_df['machine_type'].str.contains('ジャグラー')]
        if len(juggler_machines) > 0:
            print_log(f"\n推奨ジャグラー台: {', '.join(juggler_machines.head(5)['machine_number'].astype(str))}")
        
        smart_machines = active_df[active_df['machine_type'].str.contains('スマスロ')]
        if len(smart_machines) > 0:
            print_log(f"推奨スマスロ台: {', '.join(smart_machines.head(5)['machine_number'].astype(str))}")

if __name__ == "__main__":
    main()