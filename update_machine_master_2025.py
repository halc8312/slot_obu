"""
2025年7月時点の最新機種マスター生成スクリプト
5号機は全て撤去済み、6号機・6.5号機（スマスロ）が主流
"""
import pandas as pd
import os
from datetime import datetime

def create_current_machine_master():
    """2025年7月時点の実際の機種構成"""
    machine_master = []
    machine_num = 1
    
    # 機種構成（2025年7月時点の一般的なホール）
    machine_config = [
        # ジャグラーシリーズ（A-Type）- 継続的に人気
        ("アイムジャグラーEX-TP", "A-Type", "北電子", 20),
        ("マイジャグラーV", "A-Type", "北電子", 15),
        ("ファンキージャグラー2", "A-Type", "北電子", 10),
        ("ハッピージャグラーVIII", "A-Type", "北電子", 10),
        ("ゴーゴージャグラー3", "A-Type", "北電子", 10),
        ("ジャグラーガールズSS", "A-Type", "北電子", 5),
        
        # スマスロ（6.5号機）- 2025年の主力
        ("Lスマスロ北斗の拳", "スマスロ", "サミー", 20),
        ("Lからくりサーカス", "スマスロ", "SANKYO", 15),
        ("L戦国乙女4 戦乱に閃く炯眼の軍師", "スマスロ", "オリンピア", 15),
        ("Lゴジラ対エヴァンゲリオン", "スマスロ", "ビスティ", 10),
        ("L炎炎ノ消防隊", "スマスロ", "大都技研", 10),
        ("Lバジリスク〜甲賀忍法帖〜絆2 天膳 BLACK EDITION", "スマスロ", "ユニバーサル", 15),
        ("L主役は銭形4", "スマスロ", "オリンピア", 10),
        ("Lゴールデンカムイ", "スマスロ", "サミー", 10),
        ("Lモンキーターン5", "スマスロ", "山佐", 10),
        ("L革命機ヴァルヴレイヴ", "スマスロ", "SANKYO", 15),
        ("L ToLOVEるダークネス", "スマスロ", "藤商事", 10),
        ("Lエウレカセブン4 HI-EVOLUTION", "スマスロ", "サミー", 10),
        ("Lコードギアス 復活のルルーシュ", "スマスロ", "サミー", 10),
        ("L吉宗RISING", "スマスロ", "大都技研", 10),
        ("Lストライク・ザ・ブラッド", "スマスロ", "藤商事", 10),
        
        # 6号機（AT/ART）
        ("新鬼武者2", "6号機AT", "エンターライズ", 10),
        ("ゴッドイーター リザレクション", "6号機AT", "山佐", 10),
        ("ダンまち2", "6号機AT", "北電子", 10),
        ("戦姫絶唱シンフォギア 勇気の歌", "6号機AT", "SANKYO", 10),
        ("ディスクアップ2", "6号機AT", "Sammy", 5),
        ("押忍!番長4", "6号機AT", "大都技研", 15),
        ("甘い誘惑", "6号機AT", "平和", 10),
        
        # パチスロ系（A-Type）
        ("ハナハナ鳳凰-30", "A-Type", "パイオニア", 10),
        ("ドリームハナハナ-30", "A-Type", "パイオニア", 10),
        ("ニューキングハナハナ-30", "A-Type", "パイオニア", 10),
        ("沖ドキ!GOLD-30", "A-Type", "ユニバーサル", 15),
        ("沖ドキ!DUO-30", "A-Type", "ユニバーサル", 10),
        ("沖ドキ!BLACK", "A-Type", "ユニバーサル", 10),
        
        # その他6号機
        ("バーサスリヴァイズ", "6号機", "大都技研", 5),
        ("クランキーコレクション", "6号機", "アクロス", 5),
        ("ニューパルサーSPIII", "6号機", "山佐", 10),
        ("ニューパルサーDX3", "6号機", "山佐", 10),
        ("サンダーVリベンジ", "6号機", "ユニバーサル", 10),
        
        # スマスロ新台（2025年導入）
        ("Lパチスロ ガールズ&パンツァー 最終章", "スマスロ", "平和", 10),
        ("L聖闘士星矢 海皇覚醒 CUSTOM EDITION", "スマスロ", "三洋", 10),
        ("Lマクロスフロンティア4", "スマスロ", "SANKYO", 10),
        ("Lゴブリンスレイヤー", "スマスロ", "藤商事", 10),
        ("L真・一騎当千", "スマスロ", "大都技研", 10),
        
        # 甘デジ系
        ("甘い誘惑 ボーナス", "甘デジ", "平和", 10),
        ("マイフラワー30", "甘デジ", "ベルコ", 10),
        ("アクエリオン ALL STARS 甘い", "甘デジ", "SANKYO", 10),
        
        # パチンコ連動スロット
        ("e Re:ゼロから始める異世界生活 season2", "6号機AT", "大都技研", 10),
        ("麻雀格闘倶楽部 覚醒", "6号機", "KPE", 5),
        ("パチスロ頭文字D", "6号機AT", "サミー", 10)
    ]
    
    # 機種を配置
    for machine_type, category, manufacturer, count in machine_config:
        for i in range(count):
            machine_master.append({
                'machine_number': machine_num,
                'machine_type': machine_type,
                'category': category,
                'manufacturer': manufacturer,
                'introduction_date': '2024-01-01' if 'L' in machine_type else '2023-01-01',
                'popular_rank': machine_num // 10 + 1,
                'is_active': True,  # 稼働中フラグ
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            })
            machine_num += 1
    
    # 残りの台番号は予備枠として設定
    while machine_num <= 640:
        machine_master.append({
            'machine_number': machine_num,
            'machine_type': f"予備台{machine_num}",
            'category': "予備",
            'manufacturer': "未定",
            'introduction_date': '2025-01-01',
            'popular_rank': 99,
            'is_active': False,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        })
        machine_num += 1
    
    return pd.DataFrame(machine_master)

def create_machine_history():
    """機種入れ替え履歴を記録"""
    history = []
    
    # 5号機撤去履歴の例
    removed_machines = [
        ("凱旋", "5号機", "2022-01-31", "規制により撤去"),
        ("アナザーゴッドハーデス", "5号機", "2022-01-31", "規制により撤去"),
        ("バジリスク絆", "5号機", "2022-01-31", "規制により撤去"),
        ("番長3", "5号機", "2022-11-30", "規制により撤去"),
        ("まどかマギカ叛逆", "5号機", "2022-11-30", "規制により撤去"),
    ]
    
    for machine, category, removal_date, reason in removed_machines:
        history.append({
            'machine_type': machine,
            'category': category,
            'action': 'removed',
            'date': removal_date,
            'reason': reason
        })
    
    return pd.DataFrame(history)

def main():
    print("=== 2025年7月版 機種マスター生成 ===")
    
    # 現在の機種マスター作成
    df_master = create_current_machine_master()
    
    # 保存
    os.makedirs('data', exist_ok=True)
    df_master.to_csv('data/machine_master_2025.csv', index=False, encoding='utf-8-sig')
    print(f"機種マスター保存完了: {len(df_master)} 台")
    
    # 統計情報
    print("\n=== 機種構成統計 ===")
    print(f"総台数: {len(df_master)}")
    print(f"稼働中: {df_master['is_active'].sum()} 台")
    print(f"予備: {(~df_master['is_active']).sum()} 台")
    
    print("\nカテゴリー別台数:")
    category_counts = df_master[df_master['is_active']].groupby('category').size()
    for category, count in category_counts.items():
        print(f"  {category}: {count} 台")
    
    print("\nメーカー別台数:")
    manufacturer_counts = df_master[df_master['is_active']].groupby('manufacturer').size().head(10)
    for manufacturer, count in manufacturer_counts.items():
        print(f"  {manufacturer}: {count} 台")
    
    # 機種履歴も保存
    df_history = create_machine_history()
    df_history.to_csv('data/machine_history.csv', index=False, encoding='utf-8-sig')
    print(f"\n機種履歴保存完了: {len(df_history)} 件")
    
    # アップロード用にもコピー
    df_master.to_csv('model_files_for_upload/machine_master.csv', index=False, encoding='utf-8-sig')
    print("\nアップロード用ファイルも更新しました。")

if __name__ == "__main__":
    main()