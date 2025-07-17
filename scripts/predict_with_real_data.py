"""
実データを使用した指定日付予測スクリプト
過去の実データを活用して、より精度の高い予測を行う
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
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUIなし環境用）
import matplotlib.pyplot as plt
import seaborn as sns

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 日本語フォント設定（環境によって異なる）
try:
    plt.rcParams['font.sans-serif'] = ['MS Gothic']
except:
    # Linux環境用
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def print_log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

class SpecificDatePredictor:
    """指定日付予測クラス"""
    
    def __init__(self, model_path: str = 'model_files_for_upload/quantile_loss_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.sequence_length = 30
        
        # 必要なインポート
        try:
            from scripts.daily_predictor import LightweightTransformerLSTM
        except ImportError:
            from daily_predictor import LightweightTransformerLSTM
        self.LightweightTransformerLSTM = LightweightTransformerLSTM
        
    def load_model(self):
        """モデルとスケーラーを読み込む"""
        print_log("Loading model and scalers...")
        
        # モデルの初期化と読み込み
        self.model = self.LightweightTransformerLSTM(
            input_dim=58,
            d_model=96,
            nhead=4,
            num_layers=2,
            lstm_hidden=64,
            dropout=0.3
        ).to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print_log("Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # スケーラーの読み込み
        scaler_path = self.model_path.replace('quantile_loss_model.pth', 'quantile_feature_scaler.joblib')
        self.feature_scaler = joblib.load(scaler_path)
        print_log("Feature scaler loaded")
    
    def load_historical_data(self, end_date: datetime, days_back: int = 60) -> pd.DataFrame:
        """過去データを読み込む"""
        print_log(f"Loading historical data (last {days_back} days)...")
        
        # まず統合データから読み込みを試みる
        if os.path.exists('../final_integrated_13months_data.csv'):
            df = pd.read_csv('../final_integrated_13months_data.csv', encoding='utf-8-sig')
            df['date'] = pd.to_datetime(df['date'])
            
            # 期間でフィルタ
            start_date = end_date - timedelta(days=days_back)
            df = df[(df['date'] >= start_date) & (df['date'] < end_date)]
            
            print_log(f"Loaded {len(df)} records from integrated data")
            return df
        
        # 個別の日次ファイルから読み込み
        all_data = []
        for i in range(days_back):
            date = end_date - timedelta(days=i+1)
            date_str = date.strftime('%Y-%m-%d')
            
            # 複数のパスを試す
            paths = [
                f'data/daily_{date_str}.csv',
                f'../data/daily_{date_str}.csv',
                f'data/{date_str}_data.csv'
            ]
            
            for path in paths:
                if os.path.exists(path):
                    df = pd.read_csv(path, encoding='utf-8-sig')
                    df['date'] = date_str
                    all_data.append(df)
                    break
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            print_log(f"Loaded {len(combined_df)} records from daily files")
            return combined_df
        
        print_log("Warning: No historical data found, using simulated data")
        return self.create_simulated_data(end_date, days_back)
    
    def create_simulated_data(self, end_date: datetime, days_back: int) -> pd.DataFrame:
        """シミュレーションデータを作成（実データがない場合）"""
        print_log("Creating simulated historical data...")
        
        data = []
        for machine_num in range(1, 641):
            for i in range(days_back):
                date = end_date - timedelta(days=i+1)
                
                # 基本出率（機種タイプによって変化）
                if machine_num <= 100:  # ジャグラー系
                    base_payout = 98
                elif machine_num <= 200:  # ART/AT機
                    base_payout = 95
                elif machine_num <= 300:  # スマスロ
                    base_payout = 97
                else:
                    base_payout = 94
                
                # 1の付く日の効果（既存データから学習されたパターン）
                if date.day in [1, 11, 21, 31]:
                    special_boost = np.random.normal(11.4, 3)  # 平均+11.4%
                else:
                    special_boost = 0
                
                # 曜日効果
                weekday_effect = [2, 0, -1, -1, 1, 5, 4][date.weekday()]
                
                payout = base_payout + special_boost + weekday_effect + np.random.normal(0, 8)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'machine_number': machine_num,
                    'machine_type': f"Machine_{machine_num}",
                    'payout_rate_numeric': max(50, min(150, payout)),
                    'total_games_numeric': 3000 + np.random.randint(-1000, 1000),
                    'total_payout_numeric': 2700 + np.random.randint(-500, 500),
                    'max_payout_numeric': 5000 + np.random.randint(-2000, 2000),
                    'diff_coins_numeric': np.random.randint(-2000, 2000),
                    'games_numeric': 3000 + np.random.randint(-1000, 1000),
                    'payouts_numeric': 50 + np.random.randint(-20, 20),
                    'big_bonuses_numeric': 8 + np.random.randint(-5, 10),
                    'regular_bonuses_numeric': 20 + np.random.randint(-10, 20)
                })
        
        return pd.DataFrame(data)
    
    def prepare_features_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を準備"""
        print_log("Preparing advanced features...")
        
        # advanced_hybrid_modelのFeatureEngineeringを使用
        try:
            from advanced_hybrid_model import AdvancedFeatureEngineering
            feature_eng = AdvancedFeatureEngineering()
            df_features = feature_eng.create_features(df)
        except ImportError:
            print_log("Using basic feature engineering")
            df_features = self.prepare_basic_features(df)
        
        return df_features
    
    def prepare_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な特徴量を準備"""
        df = df.copy()
        df['date_obj'] = pd.to_datetime(df['date'])
        
        # 時間的特徴量
        df['day'] = df['date_obj'].dt.day
        df['month'] = df['date_obj'].dt.month
        df['weekday'] = df['date_obj'].dt.weekday
        df['day_of_year'] = df['date_obj'].dt.dayofyear
        
        # 周期的特徴量（モデルが学習済み）
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # 特別日フラグ（モデルが暗黙的に学習）
        df['is_special_day'] = df['day'].isin([1, 11, 21, 31]).astype(int)
        
        # 機械ごとの統計量
        df = df.sort_values(['machine_number', 'date_obj'])
        for window in [3, 7, 14]:
            df[f'payout_ma_{window}'] = df.groupby('machine_number')['payout_rate_numeric'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'payout_std_{window}'] = df.groupby('machine_number')['payout_rate_numeric'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0)
        
        # log1p変換（モデルが期待する形式）
        df['payout_rate_clipped'] = df['payout_rate_numeric'].clip(20, 150)
        df['payout_rate_log1p'] = np.log1p(df['payout_rate_clipped'])
        
        return df
    
    def predict_date(self, target_date: datetime, historical_df: pd.DataFrame) -> pd.DataFrame:
        """指定日付の予測を実行"""
        print_log(f"\nPredicting for {target_date.strftime('%Y-%m-%d')}...")
        
        # 特別日チェック
        is_special = target_date.day in [1, 11, 21, 31]
        if is_special:
            print_log(f"★ {target_date.day}日 - 特別日（1の付く日）です！")
        
        # 機種マスターの読み込み
        try:
            from scripts.dynamic_machine_updater import DynamicMachineUpdater
        except ImportError:
            from dynamic_machine_updater import DynamicMachineUpdater
        updater = DynamicMachineUpdater()
        master_df = updater.load_current_master()
        
        predictions = []
        
        # 全台予測
        with torch.no_grad():
            for machine_num in range(1, 641):
                # 機種情報
                machine_info = master_df[master_df['machine_number'] == machine_num]
                if len(machine_info) > 0:
                    machine_type = machine_info.iloc[0]['machine_type']
                    is_active = machine_info.iloc[0].get('is_active', True)
                else:
                    machine_type = f"Machine_{machine_num}"
                    is_active = True
                
                if not is_active:
                    continue
                
                # 該当機械の過去データ
                machine_hist = historical_df[historical_df['machine_number'] == machine_num].copy()
                
                if len(machine_hist) < 7:  # 最低7日分のデータが必要
                    continue
                
                # 予測日のダミーレコードを追加
                dummy_record = machine_hist.iloc[-1:].copy()
                dummy_record['date'] = target_date.strftime('%Y-%m-%d')
                dummy_record['date_obj'] = target_date
                dummy_record['payout_rate_numeric'] = 0
                
                machine_data = pd.concat([machine_hist, dummy_record], ignore_index=True)
                
                # 特徴量準備
                features_df = self.prepare_basic_features(machine_data)
                
                # 数値特徴量のみ選択
                numeric_cols = [col for col in features_df.columns 
                              if features_df[col].dtype in ['float64', 'int64'] 
                              and col not in ['machine_number', 'payout_rate_log1p']]
                
                # シーケンス作成
                if len(features_df) >= self.sequence_length:
                    sequence = features_df[numeric_cols].iloc[-self.sequence_length:].values
                    sequence_scaled = self.feature_scaler.transform(sequence)
                    
                    # 予測
                    X = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
                    pred, _ = self.model(X)
                    
                    # 逆変換
                    pred_value = torch.expm1(pred).cpu().numpy()[0, 0]
                    
                    predictions.append({
                        'machine_number': machine_num,
                        'machine_type': machine_type,
                        'predicted_payout': float(pred_value),
                        'is_special_day': is_special
                    })
        
        # 結果をDataFrameに
        results_df = pd.DataFrame(predictions)
        results_df = results_df.sort_values('predicted_payout', ascending=False).reset_index(drop=True)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # 評価追加
        results_df['evaluation'] = results_df.apply(
            lambda x: '★★★' if x['predicted_payout'] >= 110 
                    else '★★' if x['predicted_payout'] >= 105 
                    else '★' if x['predicted_payout'] >= 100 
                    else '', axis=1
        )
        
        return results_df
    
    def visualize_predictions(self, results_df: pd.DataFrame, target_date: datetime):
        """予測結果を可視化"""
        print_log("Creating visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{target_date.strftime("%Y年%m月%d日")} 予測結果分析', fontsize=16)
        
        # 1. 出率分布
        ax1 = axes[0, 0]
        ax1.hist(results_df['predicted_payout'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results_df['predicted_payout'].mean(), color='red', linestyle='--', 
                   label=f'平均: {results_df["predicted_payout"].mean():.1f}%')
        ax1.set_xlabel('予測出率 (%)')
        ax1.set_ylabel('台数')
        ax1.set_title('予測出率分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top 20台
        ax2 = axes[0, 1]
        top20 = results_df.head(20)
        bars = ax2.barh(range(len(top20)), top20['predicted_payout'])
        ax2.set_yticks(range(len(top20)))
        ax2.set_yticklabels([f"{row['machine_number']} - {row['machine_type'][:15]}" 
                            for _, row in top20.iterrows()])
        ax2.set_xlabel('予測出率 (%)')
        ax2.set_title('Top 20 高出率予測台')
        ax2.grid(True, axis='x', alpha=0.3)
        
        # 色分け
        for i, (_, row) in enumerate(top20.iterrows()):
            if row['predicted_payout'] >= 110:
                bars[i].set_color('red')
            elif row['predicted_payout'] >= 105:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('green')
        
        # 3. 機種タイプ別平均
        ax3 = axes[1, 0]
        # 機種名から大まかなタイプを推定
        results_df['type'] = results_df['machine_type'].apply(
            lambda x: 'ジャグラー' if 'ジャグラー' in x 
                    else 'スマスロ' if 'スマスロ' in x or 'L' in x[:2]
                    else 'AT/ART' if any(word in x for word in ['番長', '北斗', 'バジリスク'])
                    else 'その他'
        )
        
        type_avg = results_df.groupby('type')['predicted_payout'].agg(['mean', 'count'])
        type_avg = type_avg[type_avg['count'] >= 5]  # 5台以上のタイプのみ
        
        ax3.bar(type_avg.index, type_avg['mean'])
        ax3.set_xlabel('機種タイプ')
        ax3.set_ylabel('平均予測出率 (%)')
        ax3.set_title('機種タイプ別平均予測')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 台数を表示
        for i, (idx, row) in enumerate(type_avg.iterrows()):
            ax3.text(i, row['mean'] + 0.5, f"n={row['count']}", ha='center')
        
        # 4. 統計サマリー
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        予測日: {target_date.strftime('%Y年%m月%d日')}
        {'【特別日（1の付く日）】' if target_date.day in [1, 11, 21, 31] else '【通常日】'}
        
        総台数: {len(results_df)}台
        平均予測出率: {results_df['predicted_payout'].mean():.1f}%
        
        110%以上: {len(results_df[results_df['predicted_payout'] >= 110])}台
        105%以上: {len(results_df[results_df['predicted_payout'] >= 105])}台
        100%以上: {len(results_df[results_df['predicted_payout'] >= 100])}台
        
        最高予測: {results_df['predicted_payout'].max():.1f}%
        最低予測: {results_df['predicted_payout'].min():.1f}%
        標準偏差: {results_df['predicted_payout'].std():.1f}%
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存
        filename = f"prediction_analysis_{target_date.strftime('%Y%m%d')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print_log(f"Visualization saved to {filename}")
    
    def generate_report(self, results_df: pd.DataFrame, target_date: datetime) -> str:
        """予測レポートを生成"""
        is_special = target_date.day in [1, 11, 21, 31]
        
        report = f"""
# パチスロ予測レポート - {target_date.strftime('%Y年%m月%d日')}

## 概要
- 予測対象日: {target_date.strftime('%Y年%m月%d日 (%a)')}
- 日付タイプ: {'【特別日】1の付く日' if is_special else '通常日'}
- 予測台数: {len(results_df)}台
- 平均予測出率: {results_df['predicted_payout'].mean():.1f}%

## 高出率予測 TOP 20

| 順位 | 台番号 | 機種名 | 予測出率 | 評価 |
|------|--------|--------|----------|------|
"""
        
        for _, row in results_df.head(20).iterrows():
            report += f"| {row['rank']} | {row['machine_number']} | {row['machine_type']} | {row['predicted_payout']:.1f}% | {row['evaluation']} |\n"
        
        if is_special:
            report += f"""
## 特別日攻略のポイント

1. **高予測台の優先確保**
   - 上位20台は朝一から狙い目
   - 特に110%以上の予測台は最優先

2. **ジャンル別推奨台**
"""
            # ジャグラー系
            juggler_top = results_df[results_df['machine_type'].str.contains('ジャグラー')].head(5)
            if len(juggler_top) > 0:
                report += f"   - ジャグラー系: {', '.join(juggler_top['machine_number'].astype(str))}\n"
            
            # スマスロ系
            smart_top = results_df[results_df['machine_type'].str.contains('スマスロ|^L')].head(5)
            if len(smart_top) > 0:
                report += f"   - スマスロ系: {', '.join(smart_top['machine_number'].astype(str))}\n"
            
            report += """
3. **立ち回り戦略**
   - 高設定の可能性が高い台を中心に
   - データカウンターで実際の出率を確認
   - 早めの見切りも重要
"""
        
        return report

def main():
    parser = argparse.ArgumentParser(description='実データを使用した指定日付予測')
    parser.add_argument('--date', type=str, help='予測日付 (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=60, help='使用する過去データの日数')
    parser.add_argument('--top', type=int, default=50, help='表示する上位台数')
    parser.add_argument('--visualize', action='store_true', help='可視化を作成')
    parser.add_argument('--report', action='store_true', help='レポートを生成')
    
    args = parser.parse_args()
    
    # 日付処理
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now() + timedelta(days=1)
    
    # 予測器の初期化
    predictor = SpecificDatePredictor()
    predictor.load_model()
    
    # 過去データの読み込み
    historical_df = predictor.load_historical_data(target_date, args.days_back)
    
    # 予測実行
    results = predictor.predict_date(target_date, historical_df)
    
    # 結果表示
    print_log(f"\n=== Top {args.top} Predictions ===")
    print_log(f"{'Rank':<6} {'Machine':<8} {'Type':<30} {'Payout':<10} {'Eval':<6}")
    print_log("-" * 70)
    
    for _, row in results.head(args.top).iterrows():
        print_log(f"{row['rank']:<6} #{row['machine_number']:<7} {row['machine_type'][:28]:<30} "
                 f"{row['predicted_payout']:>7.1f}% {row['evaluation']:<6}")
    
    # CSV保存
    filename = f"predictions_{target_date.strftime('%Y%m%d')}_detailed.csv"
    results.to_csv(filename, index=False, encoding='utf-8-sig')
    print_log(f"\nResults saved to {filename}")
    
    # 可視化
    if args.visualize:
        predictor.visualize_predictions(results, target_date)
    
    # レポート生成
    if args.report:
        report = predictor.generate_report(results, target_date)
        report_file = f"report_{target_date.strftime('%Y%m%d')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print_log(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()