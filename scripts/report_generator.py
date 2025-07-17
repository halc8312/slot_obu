"""
GitHub Actions用 - レポート生成スクリプト
GitHub Pagesで公開可能なHTML形式
"""

import pandas as pd
import json
from datetime import datetime, timedelta
import os

def generate_html_report(predictions_df, date, stats):
    """HTMLレポートを生成"""
    
    # 特別日かチェック
    pred_date = datetime.strptime(date, '%Y-%m-%d')
    is_special = pred_date.day % 10 == 1
    
    # TOP20を抽出
    top20 = predictions_df.head(20)
    
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>パチスロ予測 - {date}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .special-day {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: bold;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #6c757d;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #007bff;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .rank-1 {{ font-weight: bold; color: #d4af37; }}
        .rank-2 {{ font-weight: bold; color: #c0c0c0; }}
        .rank-3 {{ font-weight: bold; color: #cd7f32; }}
        .high-rate {{ color: #28a745; font-weight: bold; }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
        @media (max-width: 600px) {{
            .container {{ padding: 15px; }}
            table {{ font-size: 0.9em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎰 パチスロ予測レポート</h1>
        <h2>{date} {'(特別日)' if is_special else ''}</h2>
        
        {'<div class="special-day">⭐ 本日は特別日です！出率が通常より高い傾向があります。</div>' if is_special else ''}
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{top20.iloc[0]["predicted_rate"]:.1f}%</div>
                <div class="stat-label">最高予測出率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["avg_predicted_rate"]:.1f}%</div>
                <div class="stat-label">平均予測出率</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(top20[top20["predicted_rate"] > 100])}</div>
                <div class="stat-label">100%超え予測</div>
            </div>
        </div>
        
        <h3>📊 推奨機械 TOP 20</h3>
        <table>
            <thead>
                <tr>
                    <th>順位</th>
                    <th>機械番号</th>
                    <th>機種名</th>
                    <th>予測出率</th>
                    <th>評価</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for idx, row in top20.iterrows():
        rank = idx + 1
        rank_class = f"rank-{rank}" if rank <= 3 else ""
        rate_class = "high-rate" if row['predicted_rate'] > 100 else ""
        
        evaluation = "★★★" if row['predicted_rate'] > 105 else "★★" if row['predicted_rate'] > 100 else "★"
        
        machine_type = row.get('machine_type', 'Unknown')
        
        html += f"""
                <tr>
                    <td class="{rank_class}">{rank}</td>
                    <td>No.{row['machine_number']}</td>
                    <td>{machine_type}</td>
                    <td class="{rate_class}">{row['predicted_rate']:.1f}%</td>
                    <td>{evaluation}</td>
                </tr>
"""
    
    html += f"""
            </tbody>
        </table>
        
        <div class="footer">
            <p>生成時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC)</p>
            <p>※ この予測は過去データに基づく統計的な推定です。投資は自己責任でお願いします。</p>
        </div>
    </div>
</body>
</html>
"""
    
    return html

def generate_index_page():
    """インデックスページを生成（過去の予測一覧）"""
    predictions_dir = 'predictions'
    reports = []
    
    # 過去の予測ファイルを検索
    if os.path.exists(predictions_dir):
        for file in os.listdir(predictions_dir):
            if file.startswith('prediction_') and file.endswith('.csv'):
                date = file.replace('prediction_', '').replace('.csv', '')
                reports.append(date)
    
    reports.sort(reverse=True)  # 新しい順
    
    html = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>パチスロ予測システム</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .report-list {
            list-style: none;
            padding: 0;
        }
        .report-item {
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 8px;
            transition: all 0.3s;
        }
        .report-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }
        .report-item a {
            text-decoration: none;
            color: #007bff;
            font-weight: 500;
            display: block;
        }
        .special {
            background: #fff3cd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎰 パチスロ予測システム</h1>
        <h2>予測レポート一覧</h2>
        <ul class="report-list">
"""
    
    for date in reports[:30]:  # 最新30件まで
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        is_special = date_obj.day % 10 == 1
        special_class = "special" if is_special else ""
        
        html += f"""
            <li class="report-item {special_class}">
                <a href="report_{date}.html">
                    {date} {'⭐ 特別日' if is_special else ''}
                </a>
            </li>
"""
    
    html += """
        </ul>
    </div>
</body>
</html>
"""
    
    return html

def main():
    # 最新の予測を読み込み
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    prediction_file = f'predictions/prediction_{tomorrow}.csv'
    
    if not os.path.exists(prediction_file):
        print(f"Prediction file not found: {prediction_file}")
        return
    
    # データ読み込み
    predictions = pd.read_csv(prediction_file)
    
    # 統計情報読み込み
    with open('predictions/stats.json', 'r') as f:
        stats = json.load(f)
    
    # レポートディレクトリ作成
    os.makedirs('reports', exist_ok=True)
    
    # HTMLレポート生成
    html_report = generate_html_report(predictions, tomorrow, stats)
    report_path = f'reports/report_{tomorrow}.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"Report generated: {report_path}")
    
    # インデックスページ生成
    index_html = generate_index_page()
    with open('reports/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    print("Index page generated: reports/index.html")
    
    # 最新レポートへのリダイレクト
    latest_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=report_{tomorrow}.html">
</head>
<body>
    <p>最新のレポートへリダイレクトしています...</p>
</body>
</html>
"""
    with open('reports/latest.html', 'w', encoding='utf-8') as f:
        f.write(latest_html)

if __name__ == "__main__":
    main()