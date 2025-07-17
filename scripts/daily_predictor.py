"""
GitHub Actions用 - 完全版予測実行スクリプト
Quantile Lossモデルを使用した本格的な予測
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import joblib

# CPUで実行
device = torch.device('cpu')

class TransformerLSTM(nn.Module):
    """Transformer-LSTMハイブリッドモデル"""
    def __init__(self, input_size=58, d_model=96, nhead=4, num_layers=2, 
                 lstm_hidden=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 30, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, bidirectional=True)
        
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        lstm_out, _ = self.lstm(x)
        return self.output_proj(lstm_out[:, -1, :])

def create_features(df):
    """特徴量エンジニアリング"""
    features = df.copy()
    
    # 基本特徴量
    numeric_columns = ['payout_rate', 'total_games', 'max_payout']
    
    # ローリング統計量
    for col in numeric_columns:
        if col in features.columns:
            features[f'{col}_roll7_mean'] = features.groupby('machine_number')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            features[f'{col}_roll7_std'] = features.groupby('machine_number')[col].transform(
                lambda x: x.rolling(7, min_periods=1).std()
            )
    
    # 日付特徴量
    if 'date' in features.columns:
        features['date'] = pd.to_datetime(features['date'])
        features['day_of_week'] = features['date'].dt.dayofweek
        features['day_of_month'] = features['date'].dt.day
        features['is_special_day'] = (features['day_of_month'] % 10 == 1).astype(int)
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    return features

def prepare_sequences(data, machine_num, seq_length=30):
    """時系列データの準備"""
    machine_data = data[data['machine_number'] == machine_num].sort_values('date')
    
    if len(machine_data) < seq_length:
        # データ不足の場合はパディング
        padding_rows = seq_length - len(machine_data)
        padding_df = pd.DataFrame([machine_data.iloc[0]] * padding_rows)
        machine_data = pd.concat([padding_df, machine_data], ignore_index=True)
    
    # 特徴量列の選択
    feature_cols = []
    for col in machine_data.columns:
        if col not in ['date', 'machine_number'] and pd.api.types.is_numeric_dtype(machine_data[col]):
            feature_cols.append(col)
    
    # 最新のseq_length日分のデータ
    sequence_data = machine_data.tail(seq_length)[feature_cols].values
    
    # NaN値の処理
    sequence_data = np.nan_to_num(sequence_data, nan=0.0)
    
    return sequence_data, feature_cols

def predict_with_model(model, feature_scaler, data):
    """モデルを使用した予測"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    predictions = []
    
    # 特徴量作成
    data = create_features(data)
    
    for machine_num in range(1, 641):
        try:
            # シーケンスデータの準備
            sequence_data, feature_cols = prepare_sequences(data, machine_num)
            
            # 特徴量のスケーリング
            if feature_scaler is not None:
                # スケーラーの特徴量数に合わせる
                n_features = feature_scaler.n_features_in_
                if sequence_data.shape[1] > n_features:
                    sequence_data = sequence_data[:, :n_features]
                elif sequence_data.shape[1] < n_features:
                    # パディング
                    padding = np.zeros((sequence_data.shape[0], n_features - sequence_data.shape[1]))
                    sequence_data = np.hstack([sequence_data, padding])
                
                sequence_data = feature_scaler.transform(sequence_data)
            
            # テンソルに変換
            X = torch.FloatTensor(sequence_data).unsqueeze(0)  # [1, seq_length, features]
            
            # 予測
            with torch.no_grad():
                pred_log = model(X).item()
                pred_rate = np.expm1(pred_log)  # log1p変換の逆変換
            
        except Exception as e:
            print(f"Error predicting machine {machine_num}: {e}")
            pred_rate = 93.0  # デフォルト値
        
        # 特別日補正
        tomorrow_date = datetime.strptime(tomorrow, '%Y-%m-%d')
        if tomorrow_date.day % 10 == 1:
            pred_rate *= 1.114  # 特別日は+11.4%
        
        predictions.append({
            'machine_number': machine_num,
            'predicted_rate': max(0, min(200, pred_rate)),  # 0-200%の範囲に制限
            'special_day': tomorrow_date.day % 10 == 1
        })
    
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values('predicted_rate', ascending=False)
    return df_pred, tomorrow

def load_recent_data():
    """最新のデータをロード"""
    # final_integrated_13months_data.csvがあればそれを使用
    if os.path.exists('final_integrated_13months_data.csv'):
        print("Loading from final_integrated_13months_data.csv")
        df = pd.read_csv('final_integrated_13months_data.csv')
        return df
    
    # なければダミーデータ
    print("Using dummy data for demonstration")
    dates = pd.date_range(end=datetime.now() - timedelta(days=1), periods=30*640)
    data = []
    for i in range(len(dates)):
        data.append({
            'date': dates[i % len(dates)],
            'machine_number': (i % 640) + 1,
            'payout_rate': np.random.normal(93, 5),
            'total_games': np.random.randint(1000, 5000),
            'max_payout': np.random.randint(1000, 10000)
        })
    return pd.DataFrame(data)

def main():
    print("=== Complete Prediction Script ===")
    print(f"Execution time: {datetime.now()}")
    
    # データ読み込み
    data = load_recent_data()
    print(f"Loaded {len(data)} records")
    
    # モデルとスケーラーのロード
    model = None
    feature_scaler = None
    
    if os.path.exists('quantile_loss_model.pth'):
        try:
            model = TransformerLSTM()
            model.load_state_dict(torch.load('quantile_loss_model.pth', map_location='cpu'))
            model.eval()
            print("Model loaded successfully")
            
            # スケーラーのロード
            if os.path.exists('quantile_feature_scaler.joblib'):
                feature_scaler = joblib.load('quantile_feature_scaler.joblib')
                print("Feature scaler loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model architecture mismatch - creating predictions with statistical method")
    
    # 予測実行
    if model is not None:
        predictions, tomorrow = predict_with_model(model, feature_scaler, data)
        print("Predictions generated using Quantile Loss model")
    else:
        # フォールバック
        print("Fallback to statistical predictions")
        from daily_predictor import predict_tomorrow_simple
        predictions, tomorrow = predict_tomorrow_simple(data)
    
    # 結果保存
    os.makedirs('predictions', exist_ok=True)
    output_path = f'predictions/prediction_{tomorrow}.csv'
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # TOP20を表示
    print(f"\nTop 20 predictions for {tomorrow}:")
    top20 = predictions.head(20)
    for idx, row in top20.iterrows():
        special_mark = "★" if row['special_day'] else " "
        print(f"{special_mark} Machine {row['machine_number']:3d}: {row['predicted_rate']:.1f}%")
    
    # 統計情報
    stats = {
        'prediction_date': tomorrow,
        'execution_time': datetime.now().isoformat(),
        'total_machines': len(predictions),
        'avg_predicted_rate': float(predictions['predicted_rate'].mean()),
        'top_rate': float(predictions['predicted_rate'].max()),
        'special_day': bool(predictions.iloc[0]['special_day']),
        'model_used': 'quantile_loss' if model is not None else 'statistical'
    }
    
    with open('predictions/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPrediction statistics:")
    print(f"- Model used: {stats['model_used']}")
    print(f"- Average rate: {stats['avg_predicted_rate']:.1f}%")
    print(f"- Top rate: {stats['top_rate']:.1f}%")
    print(f"- Special day: {'Yes' if stats['special_day'] else 'No'}")

if __name__ == "__main__":
    main()