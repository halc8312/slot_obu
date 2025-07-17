"""
機種マスターを実際の機種名で更新するスクリプト
"""
import pandas as pd
import os

# 実際の機種名マッピング（640台分）
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

machine_master = []

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
print(f"Machine master saved with {len(df_master)} records")

# 確認のため最初と最後を表示
print("\n最初の5台:")
print(df_master.head())
print("\n最後の5台:")
print(df_master.tail())
print(f"\n機械番号419: {df_master[df_master['machine_number'] == 419]['machine_type'].values[0]}")