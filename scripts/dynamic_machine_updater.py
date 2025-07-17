"""
動的機種マスター更新システム
実際のホールデータから機種情報を取得・更新
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import json

class DynamicMachineUpdater:
    def __init__(self):
        self.master_path = 'data/machine_master.csv'
        self.history_path = 'data/machine_history.csv'
        
    def load_current_master(self):
        """現在の機種マスターを読み込み"""
        if os.path.exists(self.master_path):
            return pd.read_csv(self.master_path, encoding='utf-8-sig')
        else:
            print("機種マスターが存在しません。新規作成します。")
            return pd.DataFrame()
    
    def check_machine_changes(self, hall_data):
        """ホールデータから機種の変更を検出"""
        current_master = self.load_current_master()
        changes = []
        
        # ホールデータの機種と現在のマスターを比較
        for machine_num in hall_data['machine_number'].unique():
            hall_machine = hall_data[hall_data['machine_number'] == machine_num].iloc[0]
            
            if len(current_master) > 0:
                master_machine = current_master[current_master['machine_number'] == machine_num]
                
                if len(master_machine) > 0:
                    # 機種が変更されているかチェック
                    if master_machine.iloc[0]['machine_type'] != hall_machine.get('machine_type', 'Unknown'):
                        changes.append({
                            'machine_number': machine_num,
                            'old_type': master_machine.iloc[0]['machine_type'],
                            'new_type': hall_machine.get('machine_type', 'Unknown'),
                            'change_date': datetime.now().strftime('%Y-%m-%d')
                        })
        
        return changes
    
    def update_machine_master(self, machine_num, new_machine_info):
        """特定の台番号の機種情報を更新"""
        master = self.load_current_master()
        
        # 更新履歴を記録
        if len(master[master['machine_number'] == machine_num]) > 0:
            old_info = master[master['machine_number'] == machine_num].iloc[0]
            self.add_history_record(
                machine_num=machine_num,
                old_type=old_info['machine_type'],
                new_type=new_machine_info['machine_type'],
                action='replaced',
                reason='新台入れ替え'
            )
        
        # マスターを更新
        master.loc[master['machine_number'] == machine_num, 'machine_type'] = new_machine_info['machine_type']
        master.loc[master['machine_number'] == machine_num, 'category'] = new_machine_info.get('category', 'その他')
        master.loc[master['machine_number'] == machine_num, 'manufacturer'] = new_machine_info.get('manufacturer', 'その他')
        master.loc[master['machine_number'] == machine_num, 'last_updated'] = datetime.now().strftime('%Y-%m-%d')
        master.loc[master['machine_number'] == machine_num, 'is_active'] = True
        
        # 保存
        os.makedirs('data', exist_ok=True)
        master.to_csv(self.master_path, index=False, encoding='utf-8-sig')
        
    def add_history_record(self, machine_num, old_type, new_type, action, reason):
        """機種変更履歴を追加"""
        history = pd.DataFrame()
        if os.path.exists(self.history_path):
            history = pd.read_csv(self.history_path, encoding='utf-8-sig')
        
        new_record = pd.DataFrame([{
            'machine_number': machine_num,
            'old_type': old_type,
            'new_type': new_type,
            'action': action,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'reason': reason
        }])
        
        history = pd.concat([history, new_record], ignore_index=True)
        history.to_csv(self.history_path, index=False, encoding='utf-8-sig')
    
    def detect_removed_machines(self, days_inactive=30):
        """長期間稼働していない機種を検出"""
        master = self.load_current_master()
        removed = []
        
        # ここでは実際のデータ取得ロジックが必要
        # 仮のロジック：last_updatedが古い機種を検出
        if 'last_updated' in master.columns:
            threshold_date = (datetime.now() - timedelta(days=days_inactive)).strftime('%Y-%m-%d')
            inactive = master[master['last_updated'] < threshold_date]
            
            for _, machine in inactive.iterrows():
                if machine['is_active']:
                    removed.append({
                        'machine_number': machine['machine_number'],
                        'machine_type': machine['machine_type'],
                        'last_seen': machine['last_updated']
                    })
        
        return removed
    
    def get_machine_type_smart(self, machine_number, hall_name=None):
        """
        賢い機種名取得
        1. 最新のホールデータから取得
        2. マスターデータから取得
        3. 機種カテゴリーから推定
        """
        master = self.load_current_master()
        
        if len(master) > 0:
            machine_info = master[master['machine_number'] == machine_number]
            if len(machine_info) > 0 and machine_info.iloc[0]['is_active']:
                return machine_info.iloc[0]['machine_type']
        
        # フォールバック：カテゴリーベースの推定
        if machine_number <= 100:
            return f"ジャグラー系{machine_number}"
        elif machine_number <= 200:
            return f"スマスロ{machine_number - 100}"
        elif machine_number <= 300:
            return f"6号機AT{machine_number - 200}"
        else:
            return f"その他{machine_number}"

def update_from_hall_data(hall_data_file):
    """ホールデータから機種マスターを更新"""
    updater = DynamicMachineUpdater()
    
    # ホールデータを読み込む（実際の実装では外部APIやスクレイピング）
    if os.path.exists(hall_data_file):
        hall_data = pd.read_csv(hall_data_file, encoding='utf-8-sig')
        
        # 変更を検出
        changes = updater.check_machine_changes(hall_data)
        
        if changes:
            print(f"検出された機種変更: {len(changes)}件")
            for change in changes:
                print(f"  台番号{change['machine_number']}: {change['old_type']} → {change['new_type']}")
        
        # 撤去された機種を検出
        removed = updater.detect_removed_machines()
        if removed:
            print(f"\n撤去の可能性がある機種: {len(removed)}台")
            for machine in removed:
                print(f"  台番号{machine['machine_number']}: {machine['machine_type']} (最終確認: {machine['last_seen']})")

def main():
    print("=== 動的機種マスター更新システム ===")
    
    # テスト用：仮のホールデータで更新をシミュレート
    test_hall_data = pd.DataFrame([
        {'machine_number': 100, 'machine_type': 'Lスマスロ北斗の拳', 'category': 'スマスロ'},
        {'machine_number': 200, 'machine_type': 'L戦国乙女4', 'category': 'スマスロ'},
        {'machine_number': 419, 'machine_type': 'Lからくりサーカス', 'category': 'スマスロ'},  # 凱旋→からくり
    ])
    
    # 更新処理
    updater = DynamicMachineUpdater()
    changes = updater.check_machine_changes(test_hall_data)
    
    print(f"変更検出結果: {len(changes)}件")

if __name__ == "__main__":
    main()