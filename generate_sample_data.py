import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data():
    """
    レストランカテゴリ別のサンプルデータを生成
    """
    # カテゴリの定義
    categories = ['イタリアン', '中華', '和食', 'フレンチ', 'カフェ']
    
    # 期間設定（2年間のデータ）
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    all_data = []
    
    for category in categories:
        # カテゴリごとの基本パラメータ
        base_params = {
            'イタリアン': {'base_sales': 800000, 'seasonality': 1.2, 'trend': 1.05},
            '中華': {'base_sales': 600000, 'seasonality': 1.1, 'trend': 1.03},
            '和食': {'base_sales': 1000000, 'seasonality': 1.3, 'trend': 1.08},
            'フレンチ': {'base_sales': 1200000, 'seasonality': 1.4, 'trend': 1.06},
            'カフェ': {'base_sales': 400000, 'seasonality': 0.9, 'trend': 1.02}
        }
        
        params = base_params[category]
        
        current_date = start_date
        while current_date <= end_date:
            # 月ごとのデータ生成
            year = current_date.year
            month = current_date.month
            
            # 季節性の計算
            seasonal_factor = params['seasonality'] * (1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12))
            
            # トレンドの計算
            months_passed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
            trend_factor = params['trend'] ** (months_passed / 12)
            
            # 基本売上にノイズを加える
            base_sales = params['base_sales']
            noise = np.random.normal(0, 0.1)  # 10%のノイズ
            
            # 月間売上計算
            target_amount = base_sales * seasonal_factor * trend_factor * (1 + noise)
            
            # その他の特徴量の生成
            avg_monthly_population = np.random.normal(50000, 10000)  # 人流
            rating_score = np.random.normal(3.5, 0.5)  # 食べログ評価
            rating_cnt = np.random.poisson(50)  # 評価数
            num_seats = np.random.choice([20, 30, 40, 50, 60])  # 席数
            
            # 最寄り駅情報
            stations = ['渋谷駅', '新宿駅', '池袋駅', '東京駅', '品川駅']
            nearest_station = np.random.choice(stations)
            
            # データ追加
            all_data.append({
                'YEAR': year,
                'MONTH': month,
                'CATEGORY': category,
                'AVG_MONTHLY_POPULATION': avg_monthly_population,
                'RATING_SCORE': max(1.0, min(5.0, rating_score)),
                'RATING_CNT': max(1, rating_cnt),
                'NUM_SEATS': num_seats,
                'NEAREST_STATION_INFO': nearest_station,
                'target_amount': max(0, target_amount)
            })
            
            # 次の月へ
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
    
    # DataFrameに変換
    df = pd.DataFrame(all_data)
    
    # 日付列を追加（Prophet用）
    df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(DAY=1))
    
    return df

if __name__ == "__main__":
    # サンプルデータ生成
    sample_data = generate_sample_data()
    
    # CSVファイルとして保存
    sample_data.to_csv('restaurant_sales_data.csv', index=False)
    
    print("サンプルデータが生成されました: restaurant_sales_data.csv")
    print(f"データ形状: {sample_data.shape}")
    print("\nカテゴリ別データ数:")
    print(sample_data['CATEGORY'].value_counts())
    print("\n最初の5行:")
    print(sample_data.head()) 