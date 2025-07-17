"""
GitHub Actions用 - 日次予測実行スクリプト
CPU環境でも動作する軽量版
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import joblib

# CPUで実行
device = torch.device('cpu')

class QuantileLossModel(torch.nn.Module):
    """軽量版予測モデル"""
    def __init__(self, input_size=58, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def load_recent_data(days=30):
    """直近30日分のデータをロード"""
    dfs = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i+1)).strftime('%Y-%m-%d')
        file_path = f'data/daily_{date}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
    
    if not dfs:
        # データがない場合はダミーデータ
        print("Warning: No historical data found, using dummy data")
        return create_dummy_data()
    
    return pd.concat(dfs, ignore_index=True)

def create_dummy_data():
    """テスト用ダミーデータ生成"""
    dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=30)
    data = []
    for date in dates:
        for machine in range(1, 641):
            data.append({
                'date': date,
                'machine_number': machine,
                'payout_rate': np.random.normal(93, 5),
                'total_games': np.random.randint(1000, 5000),
                # 他の特徴量
            })
    return pd.DataFrame(data)

def predict_tomorrow(model, data):
    """明日の予測を実行"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    predictions = []
    
    # 各機械の予測
    for machine_num in range(1, 641):
        # 機械ごとの直近30日データ（簡易版）
        machine_data = data[data['machine_number'] == machine_num].tail(30)
        
        if len(machine_data) < 30:
            # データ不足の場合は平均値
            pred_rate = data['payout_rate'].mean()
        else:
            # 簡易予測（実際はモデルを使用）
            # ここではトレンドベースの簡易予測
            recent_rates = machine_data['payout_rate'].values
            trend = np.polyfit(range(len(recent_rates)), recent_rates, 1)[0]
            pred_rate = recent_rates[-1] + trend
        
        # 特別日補正
        tomorrow_date = datetime.strptime(tomorrow, '%Y-%m-%d')
        if tomorrow_date.day % 10 == 1:
            pred_rate *= 1.114  # 特別日は+11.4%
        
        predictions.append({
            'machine_number': machine_num,
            'predicted_rate': pred_rate,
            'special_day': tomorrow_date.day % 10 == 1
        })
    
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values('predicted_rate', ascending=False)
    return df_pred, tomorrow

def main():
    print("=== Daily Prediction Script ===")
    print(f"Execution time: {datetime.now()}")
    
    # データ読み込み
    data = load_recent_data()
    print(f"Loaded {len(data)} records")
    
    # モデルロード（存在しない場合はスキップ）
    model = None
    if os.path.exists('quantile_loss_model.pth'):
        model = QuantileLossModel()
        model.load_state_dict(torch.load('quantile_loss_model.pth', map_location='cpu'))
        model.eval()
        print("Model loaded successfully")
    else:
        print("Model not found, using statistical prediction")
    
    # 予測実行
    predictions, tomorrow = predict_tomorrow(model, data)
    
    # 結果保存
    os.makedirs('predictions', exist_ok=True)
    output_path = f'predictions/prediction_{tomorrow}.csv'
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # TOP20を表示
    print(f"\nTop 20 predictions for {tomorrow}:")
    print(predictions.head(20)[['machine_number', 'predicted_rate', 'special_day']])
    
    # 統計情報
    stats = {
        'prediction_date': tomorrow,
        'execution_time': datetime.now().isoformat(),
        'total_machines': len(predictions),
        'avg_predicted_rate': predictions['predicted_rate'].mean(),
        'top_rate': predictions['predicted_rate'].max(),
        'special_day': bool(predictions.iloc[0]['special_day'])
    }
    
    with open('predictions/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()