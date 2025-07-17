"""
高度な特別日効果分析システム
「1の付く日」の機種別反応パターンを詳細に分析
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

class SpecialDayPatternAnalyzer:
    """特別日パターンの高度な分析クラス"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.special_day_patterns = {}
        self.machine_reactions = {}
        
    def load_data(self) -> pd.DataFrame:
        """13ヶ月分のデータを読み込む"""
        print("Loading 13 months data...")
        df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def extract_special_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特別日に関する詳細な特徴量を抽出"""
        df = df.copy()
        
        # 基本的な日付特徴
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['weekday'] = df['date'].dt.weekday
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # 1の付く日フラグ（1, 11, 21, 31日）
        df['is_special_day'] = df['day'].isin([1, 11, 21, 31]).astype(int)
        
        # 詳細な特別日タイプ
        df['special_day_type'] = 'normal'
        df.loc[df['day'] == 1, 'special_day_type'] = 'month_start'
        df.loc[df['day'] == 11, 'special_day_type'] = 'day_11'
        df.loc[df['day'] == 21, 'special_day_type'] = 'day_21'
        df.loc[df['day'] == 31, 'special_day_type'] = 'month_end_31'
        
        # ゾロ目の日（11, 22日）
        df['is_zorome'] = df['day'].isin([11, 22]).astype(int)
        
        # 月初・月末フラグ
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        
        # 曜日との組み合わせ効果
        df['special_weekday_combo'] = df['is_special_day'] * (df['weekday'] + 1)
        
        # 連続特別日（前日も特別日だったか）
        df['prev_special'] = df.groupby('machine_number')['is_special_day'].shift(1).fillna(0)
        df['consecutive_special'] = (df['is_special_day'] * df['prev_special']).astype(int)
        
        # 月内の特別日順序（何番目の特別日か）
        df['special_day_order'] = df.groupby(['machine_number', df['date'].dt.to_period('M')])['is_special_day'].cumsum()
        df.loc[df['is_special_day'] == 0, 'special_day_order'] = 0
        
        return df
    
    def analyze_machine_specific_patterns(self, df: pd.DataFrame) -> Dict:
        """機種別の特別日反応パターンを分析"""
        results = {}
        
        # 機種ごとに分析
        for machine_type in df['machine_type'].unique():
            machine_data = df[df['machine_type'] == machine_type].copy()
            
            # 特別日と通常日の比較
            special_days = machine_data[machine_data['is_special_day'] == 1]
            normal_days = machine_data[machine_data['is_special_day'] == 0]
            
            if len(special_days) > 0 and len(normal_days) > 0:
                # 基本統計
                special_mean = special_days['payout_rate_numeric'].mean()
                normal_mean = normal_days['payout_rate_numeric'].mean()
                boost_rate = (special_mean - normal_mean) / normal_mean * 100
                
                # 特別日タイプ別の分析
                type_analysis = {}
                for stype in ['month_start', 'day_11', 'day_21', 'month_end_31']:
                    type_data = machine_data[machine_data['special_day_type'] == stype]
                    if len(type_data) > 0:
                        type_mean = type_data['payout_rate_numeric'].mean()
                        type_boost = (type_mean - normal_mean) / normal_mean * 100
                        type_analysis[stype] = {
                            'mean_payout': type_mean,
                            'boost_rate': type_boost,
                            'sample_size': len(type_data)
                        }
                
                # 曜日別の特別日効果
                weekday_effects = {}
                for wd in range(7):
                    wd_special = special_days[special_days['weekday'] == wd]
                    if len(wd_special) > 0:
                        wd_mean = wd_special['payout_rate_numeric'].mean()
                        wd_boost = (wd_mean - normal_mean) / normal_mean * 100
                        weekday_effects[wd] = wd_boost
                
                # 変動性の分析
                special_std = special_days['payout_rate_numeric'].std()
                normal_std = normal_days['payout_rate_numeric'].std()
                volatility_ratio = special_std / normal_std if normal_std > 0 else 1.0
                
                results[machine_type] = {
                    'base_stats': {
                        'special_mean': special_mean,
                        'normal_mean': normal_mean,
                        'boost_rate': boost_rate,
                        'special_std': special_std,
                        'normal_std': normal_std,
                        'volatility_ratio': volatility_ratio
                    },
                    'type_analysis': type_analysis,
                    'weekday_effects': weekday_effects,
                    'sample_sizes': {
                        'special_days': len(special_days),
                        'normal_days': len(normal_days)
                    }
                }
        
        return results
    
    def create_special_day_features(self, date: datetime, machine_type: str) -> np.ndarray:
        """指定日付と機種に対する特別日特徴量を生成"""
        features = []
        
        # 基本的な日付情報
        day = date.day
        month = date.month
        weekday = date.weekday()
        day_of_year = date.timetuple().tm_yday
        
        # 1の付く日判定
        is_special = 1 if day in [1, 11, 21, 31] else 0
        
        # 特別日タイプのワンホットエンコーディング
        special_type = [0, 0, 0, 0]  # month_start, day_11, day_21, month_end_31
        if day == 1:
            special_type[0] = 1
        elif day == 11:
            special_type[1] = 1
        elif day == 21:
            special_type[2] = 1
        elif day == 31:
            special_type[3] = 1
        
        # 特別日と曜日の組み合わせ
        special_weekday_combo = is_special * (weekday + 1)
        
        # 月初・月末
        is_month_start = 1 if day <= 3 else 0
        is_month_end = 1 if day >= 28 else 0
        
        # ゾロ目
        is_zorome = 1 if day in [11, 22] else 0
        
        # 正弦波による周期性の表現
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        weekday_sin = np.sin(2 * np.pi * weekday / 7)
        weekday_cos = np.cos(2 * np.pi * weekday / 7)
        
        # 特徴量を結合
        features = [
            is_special,
            *special_type,
            special_weekday_combo,
            is_month_start,
            is_month_end,
            is_zorome,
            day_sin, day_cos,
            month_sin, month_cos,
            weekday_sin, weekday_cos,
            day / 31,  # 正規化された日
            month / 12,  # 正規化された月
            weekday / 6,  # 正規化された曜日
            day_of_year / 365  # 正規化された年内日数
        ]
        
        return np.array(features)
    
    def visualize_patterns(self, analysis_results: Dict):
        """分析結果を可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('機種別特別日効果の詳細分析', fontsize=16)
        
        # 1. トップ20機種のブースト率
        boost_data = [(mt, res['base_stats']['boost_rate']) 
                      for mt, res in analysis_results.items()
                      if res['sample_sizes']['special_days'] >= 10]
        boost_data.sort(key=lambda x: x[1], reverse=True)
        top_20 = boost_data[:20]
        
        ax1 = axes[0, 0]
        machines, boosts = zip(*top_20)
        bars = ax1.bar(range(len(machines)), boosts)
        ax1.set_xlabel('機種')
        ax1.set_ylabel('ブースト率 (%)')
        ax1.set_title('特別日ブースト率 TOP20')
        ax1.set_xticks(range(len(machines)))
        ax1.set_xticklabels(machines, rotation=45, ha='right')
        
        # 色分け
        for i, bar in enumerate(bars):
            if boosts[i] > 15:
                bar.set_color('red')
            elif boosts[i] > 10:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 2. 特別日タイプ別の平均効果
        ax2 = axes[0, 1]
        type_effects = {'month_start': [], 'day_11': [], 'day_21': [], 'month_end_31': []}
        
        for mt, res in analysis_results.items():
            for stype, effect in res['type_analysis'].items():
                if effect['sample_size'] >= 5:
                    type_effects[stype].append(effect['boost_rate'])
        
        type_means = {k: np.mean(v) if v else 0 for k, v in type_effects.items()}
        type_labels = ['月初(1日)', '11日', '21日', '月末(31日)']
        
        ax2.bar(type_labels, list(type_means.values()))
        ax2.set_ylabel('平均ブースト率 (%)')
        ax2.set_title('特別日タイプ別効果')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. 曜日との組み合わせ効果
        ax3 = axes[1, 0]
        weekday_matrix = np.zeros((7, 4))  # 7曜日 x 4特別日タイプ
        weekday_counts = np.zeros((7, 4))
        
        for mt, res in analysis_results.items():
            for wd, boost in res['weekday_effects'].items():
                weekday_matrix[wd, :] += boost
                weekday_counts[wd, :] += 1
        
        # 平均を計算
        weekday_matrix = np.divide(weekday_matrix, weekday_counts, 
                                  where=weekday_counts!=0, out=np.zeros_like(weekday_matrix))
        
        weekday_labels = ['月', '火', '水', '木', '金', '土', '日']
        im = ax3.imshow(weekday_matrix.T, aspect='auto', cmap='RdYlGn')
        ax3.set_xticks(range(7))
        ax3.set_xticklabels(weekday_labels)
        ax3.set_yticks(range(4))
        ax3.set_yticklabels(type_labels)
        ax3.set_title('曜日×特別日タイプの効果')
        plt.colorbar(im, ax=ax3)
        
        # 4. ボラティリティ分析
        ax4 = axes[1, 1]
        volatility_data = [(mt, res['base_stats']['volatility_ratio']) 
                          for mt, res in analysis_results.items()
                          if res['sample_sizes']['special_days'] >= 10]
        volatility_data.sort(key=lambda x: x[1], reverse=True)
        top_volatility = volatility_data[:20]
        
        machines_v, ratios = zip(*top_volatility)
        ax4.bar(range(len(machines_v)), ratios)
        ax4.set_xlabel('機種')
        ax4.set_ylabel('ボラティリティ比率')
        ax4.set_title('特別日のボラティリティ増加率 TOP20')
        ax4.set_xticks(range(len(machines_v)))
        ax4.set_xticklabels(machines_v, rotation=45, ha='right')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('advanced_special_day_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Advanced analysis visualization saved!")


class SpecialDayNeuralPredictor(nn.Module):
    """特別日効果を学習する高度なニューラルネットワーク"""
    
    def __init__(self, base_input_dim: int, special_dim: int = 20, hidden_dim: int = 128):
        super().__init__()
        
        # 特別日専用のエンコーダー
        self.special_encoder = nn.Sequential(
            nn.Linear(special_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # 基本特徴量のエンコーダー
        self.base_encoder = nn.Sequential(
            nn.Linear(base_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 特別日と基本特徴量の相互作用をモデル化
        self.interaction_layer = nn.Sequential(
            nn.Linear(32 + hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # アテンション機構（特別日の重要度を動的に調整）
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 最終予測層
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, base_features, special_features):
        # エンコード
        special_encoded = self.special_encoder(special_features)
        base_encoded = self.base_encoder(base_features)
        
        # 結合
        combined = torch.cat([special_encoded, base_encoded], dim=1)
        interaction = self.interaction_layer(combined)
        
        # アテンション適用
        attention_weights = self.attention(interaction)
        attended_features = interaction * attention_weights
        
        # 予測
        output = self.output_layer(attended_features)
        
        return output, attention_weights


def analyze_special_days():
    """特別日効果の高度な分析を実行"""
    print("=== Advanced Special Day Effect Analysis ===")
    
    # データパスの確認
    data_path = 'final_integrated_13months_data.csv'
    if not pd.io.common.file_exists(data_path):
        print(f"Data file not found: {data_path}")
        return None
    
    # 分析器の初期化
    analyzer = SpecialDayPatternAnalyzer(data_path)
    
    # データ読み込みと特徴量抽出
    df = analyzer.load_data()
    df = analyzer.extract_special_day_features(df)
    
    # 機種別パターン分析
    print("\nAnalyzing machine-specific patterns...")
    analysis_results = analyzer.analyze_machine_specific_patterns(df)
    
    # 結果の保存
    results_df = []
    for machine_type, results in analysis_results.items():
        row = {
            'machine_type': machine_type,
            'boost_rate': results['base_stats']['boost_rate'],
            'special_mean': results['base_stats']['special_mean'],
            'normal_mean': results['base_stats']['normal_mean'],
            'volatility_ratio': results['base_stats']['volatility_ratio'],
            'special_days_count': results['sample_sizes']['special_days']
        }
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_df.sort_values('boost_rate', ascending=False, inplace=True)
    results_df.to_csv('temp_slot_obu/advanced_special_day_analysis.csv', index=False, encoding='utf-8-sig')
    
    # 可視化
    print("\nCreating visualizations...")
    analyzer.visualize_patterns(analysis_results)
    
    # トップ10の結果を表示
    print("\n=== Top 10 Special Day Boost Machines ===")
    print(results_df.head(10).to_string(index=False))
    
    return analyzer, analysis_results


if __name__ == "__main__":
    analyzer, results = analyze_special_days()
    print("\nAnalysis complete! Results saved to advanced_special_day_analysis.csv")