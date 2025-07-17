"""
強化版データ収集スクリプト - 動的機種更新対応版
2025年7月対応 - 5号機撤去済み、6号機・スマスロ主流
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dynamic_machine_updater import DynamicMachineUpdater

def print_log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 機種名マッピング（実際のデータに基づいて更新が必要）
MACHINE_TYPE_MAPPING = {
    # 人気機種の例
    1: "バジリスク絆2",
    2: "北斗の拳 宿命",
    3: "ゴッドイーター",
    4: "番長3",
    5: "まどマギ叛逆",
    6: "モンキーターンV",
    7: "ヴァルヴレイヴ",
    8: "ハナハナ鳳凰",
    9: "ジャグラーEX",
    10: "マイジャグラーV",
    # 実際の運用では全640台分のマッピングが必要
}

def get_machine_type(machine_number):
    """機種名を取得（動的更新対応版）"""
    # DynamicMachineUpdaterを使用
    updater = DynamicMachineUpdater()
    machine_type = updater.get_machine_type_smart(machine_number)
    
    # 稼働状況の確認
    master = updater.load_current_master()
    if len(master) > 0:
        machine_info = master[master['machine_number'] == machine_number]
        if len(machine_info) > 0:
            if not machine_info.iloc[0].get('is_active', True):
                print_log(f"Machine {machine_number} is inactive (removed or reserved)")
                return f"予備台{machine_number}"
    
    return machine_type

def collect_pachislot_data(date):
    """
    指定日のパチスロデータを収集（動的機種更新対応）
    """
    print_log(f"Collecting data for {date}")
    
    # 実際のホールデータ収集（ここではダミーデータ）
    # 本番環境では外部APIやWebスクレイピングで取得
    machines = []
    updater = DynamicMachineUpdater()
    
    for machine_num in range(1, 641):
        machine_type = get_machine_type(machine_num)
        
        # 実際のデータ収集ロジック（ダミー）
        machine_data = {
            'date': date,
            'machine_number': machine_num,
            'machine_type': machine_type,
            'payout_rate': 85 + (machine_num % 30),
            'total_games': 3000 + (machine_num * 10) % 2000,
            'payout_rate_numeric': 85 + (machine_num % 30),
            'total_games_numeric': 3000 + (machine_num * 10) % 2000,
            'total_payout_numeric': 2700 + (machine_num * 9) % 1800,
            'max_payout_numeric': 5000 + (machine_num * 50) % 3000,
            # 他の特徴量...
        }
        
        # 稼働中の台のみデータを収集
        master = updater.load_current_master()
        if len(master) > 0:
            machine_info = master[master['machine_number'] == machine_num]
            if len(machine_info) > 0 and not machine_info.iloc[0].get('is_active', True):
                # 非稼働台は出率を0に
                machine_data['payout_rate'] = 0
                machine_data['payout_rate_numeric'] = 0
                machine_data['total_games'] = 0
                machine_data['total_games_numeric'] = 0
        
        machines.append(machine_data)
    
    df_machines = pd.DataFrame(machines)
    
    # 機種変更の検出（本番環境では実装）
    # changes = updater.check_machine_changes(df_machines)
    # if changes:
    #     print_log(f"Detected {len(changes)} machine changes")
    
    return df_machines

def save_machine_master():
    """機種マスターデータを保存（実際の機種名を使用）"""
    machine_master = []
    
    # 実際の機種名マッピング（640台分の一部）
    actual_machine_names = {
        # ジャグラーシリーズ
        **{i: "アイムジャグラーEX" for i in range(1, 11)},
        **{i: "マイジャグラーV" for i in range(11, 21)},
        **{i: "ファンキージャグラー2" for i in range(21, 31)},
        **{i: "ハッピージャグラーVIII" for i in range(31, 41)},
        **{i: "ゴーゴージャグラー3" for i in range(41, 51)},
        **{i: "ジャグラーガールズSS" for i in range(51, 61)},
        **{i: "ミラクルジャグラー" for i in range(61, 71)},
        
        # 北斗シリーズ
        **{i: "北斗の拳 宿命" for i in range(71, 81)},
        **{i: "北斗の拳 天昇" for i in range(81, 91)},
        **{i: "北斗の拳 羅刹" for i in range(91, 101)},
        **{i: "北斗の拳 修羅の国篇" for i in range(101, 111)},
        
        # バジリスクシリーズ
        **{i: "バジリスク絆2" for i in range(111, 121)},
        **{i: "バジリスク甲賀忍法帖III" for i in range(121, 131)},
        **{i: "バジリスク桜花忍法帖" for i in range(131, 141)},
        
        # 番長シリーズ
        **{i: "押忍!番長3" for i in range(141, 151)},
        **{i: "押忍!番長ZERO" for i in range(151, 161)},
        **{i: "押忍!サラリーマン番長2" for i in range(161, 171)},
        
        # その他人気機種
        **{i: "まどかマギカ叛逆" for i in range(171, 181)},
        **{i: "新鬼武者2" for i in range(181, 191)},
        **{i: "モンキーターンV" for i in range(191, 201)},
        **{i: "ヴァルヴレイヴ" for i in range(201, 211)},
        **{i: "ゴッドイーター" for i in range(211, 221)},
        **{i: "戦国乙女4" for i in range(221, 231)},
        **{i: "リゼロ2" for i in range(231, 241)},
        **{i: "エウレカセブン4" for i in range(241, 251)},
        **{i: "からくりサーカス" for i in range(251, 261)},
        **{i: "ダンまち2" for i in range(261, 271)},
        **{i: "戦姫絶唱シンフォギア" for i in range(271, 281)},
        **{i: "真天下布武" for i in range(281, 291)},
        **{i: "ディスクアップ2" for i in range(291, 301)},
        **{i: "HEY!エリートサラリーマン鏡" for i in range(301, 311)},
        **{i: "幼女戦記" for i in range(311, 321)},
        **{i: "ゴブリンスレイヤー" for i in range(321, 331)},
        **{i: "コードギアスR2" for i in range(331, 341)},
        **{i: "甘いバジリスク絆2" for i in range(341, 351)},
        **{i: "甘い番長3" for i in range(351, 361)},
        **{i: "アクエリオン ALL STARS" for i in range(361, 371)},
        **{i: "マクロスΔ" for i in range(371, 381)},
        **{i: "麻雀物語4" for i in range(381, 391)},
        **{i: "サンダーVライトニング" for i in range(391, 401)},
        **{i: "アナザーゴッドハーデス" for i in range(401, 411)},
        **{i: "凱旋" for i in range(411, 421)},
        **{i: "スマスロ北斗の拳" for i in range(421, 431)},
        **{i: "スマスロバジリスク" for i in range(431, 441)},
        **{i: "スマスロ鏡" for i in range(441, 451)},
        **{i: "スマスロ花の慶次" for i in range(451, 461)},
        **{i: "スマスロヴァルヴレイヴ" for i in range(461, 471)},
        **{i: "スマスロ炎炎ノ消防隊" for i in range(471, 481)},
        **{i: "スマスロストライクザブラッド" for i in range(481, 491)},
        **{i: "スマスロゴジラ対エヴァ" for i in range(491, 501)},
        **{i: "花火絶景" for i in range(501, 511)},
        **{i: "ニューパルサーSP" for i in range(511, 521)},
        **{i: "バーサスリヴァイズ" for i in range(521, 531)},
        **{i: "クランキーセレブレーション" for i in range(531, 541)},
        **{i: "ワードオブライツII" for i in range(541, 551)},
        **{i: "ハナハナ鳳凰" for i in range(551, 561)},
        **{i: "ドリームハナハナ" for i in range(561, 571)},
        **{i: "ニューキングハナハナ" for i in range(571, 581)},
        **{i: "沖ドキ!GOLD" for i in range(581, 591)},
        **{i: "沖ドキ!DUO" for i in range(591, 601)},
        **{i: "花の慶次〜裂一刀両断〜" for i in range(601, 611)},
        **{i: "吉宗RISING" for i in range(611, 621)},
        **{i: "政宗 戦極" for i in range(621, 631)},
        **{i: "犬夜叉" for i in range(631, 641)},
    }
    
    for machine_num in range(1, 641):
        machine_type = actual_machine_names.get(machine_num, f"スロット{machine_num}")
        
        # カテゴリーの判定
        if "ジャグラー" in machine_type:
            category = "A-Type"
        elif "番長" in machine_type or "北斗" in machine_type or "バジリスク" in machine_type:
            category = "ART/AT"
        elif "甘い" in machine_type:
            category = "甘デジ"
        elif "スマスロ" in machine_type:
            category = "スマスロ"
        elif "ハナハナ" in machine_type or "沖ドキ" in machine_type:
            category = "A-Type"
        else:
            category = "その他"
        
        # メーカーの判定
        if "ジャグラー" in machine_type:
            manufacturer = "北電子"
        elif "番長" in machine_type:
            manufacturer = "大都技研"
        elif "北斗" in machine_type or "バジリスク" in machine_type:
            manufacturer = "サミー"
        elif "ハナハナ" in machine_type:
            manufacturer = "パイオニア"
        else:
            manufacturer = "その他メーカー"
        
        machine_master.append({
            'machine_number': machine_num,
            'machine_type': machine_type,
            'category': category,
            'manufacturer': manufacturer,
            'introduction_date': '2024-01-01',
            'popular_rank': ((machine_num - 1) // 10) + 1
        })
    
    df_master = pd.DataFrame(machine_master)
    os.makedirs('data', exist_ok=True)
    df_master.to_csv('data/machine_master.csv', index=False, encoding='utf-8-sig')
    print_log(f"Machine master saved with {len(df_master)} records")
    
    return df_master

def main():
    print_log("=== Enhanced Data Collection (2025 Edition) ===")
    
    # 機種マスターデータの確認
    if not os.path.exists('data/machine_master.csv'):
        print_log("Machine master data not found - it should be downloaded from GitHub Release")
        # 2025年版マスターファイルが必要
        print_log("ERROR: Please ensure machine_master.csv is downloaded from GitHub Release")
        return
    else:
        print_log("Machine master data found, checking version...")
        # マスターファイルのバージョン確認
        master_df = pd.read_csv('data/machine_master.csv', encoding='utf-8-sig')
        if 'is_active' in master_df.columns:
            active_count = master_df['is_active'].sum()
            print_log(f"Active machines: {active_count} / {len(master_df)}")
        else:
            print_log("WARNING: Old format machine master detected")
    
    # 昨日のデータを収集
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    try:
        # データ収集
        df = collect_pachislot_data(yesterday)
        
        # 保存
        os.makedirs('data', exist_ok=True)
        
        # 日次ファイル
        output_path = f'data/daily_{yesterday}.csv'
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print_log(f"Data saved to {output_path}")
        
        # 最新データとしても保存
        df.to_csv('data/latest_data.csv', index=False, encoding='utf-8-sig')
        
        # 統計情報
        stats = {
            'date': yesterday,
            'total_machines': len(df),
            'unique_machine_types': df['machine_type'].nunique(),
            'avg_payout_rate': df['payout_rate'].mean(),
            'collection_time': datetime.now().isoformat()
        }
        
        with open('data/collection_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
    except Exception as e:
        print_log(f"Error collecting data: {e}")
        exit(0)

if __name__ == "__main__":
    main()