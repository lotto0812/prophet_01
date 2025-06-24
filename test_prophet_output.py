#!/usr/bin/env python3
"""
Prophetモデルの出力形式をテストするスクリプト
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from font_setup import setup_japanese_font

# 日本語フォント設定
setup_japanese_font()

def test_prophet_output():
    """
    Prophetモデルの出力形式をテスト
    """
    # サンプルデータを読み込み
    df = pd.read_csv('restaurant_sales_data.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # イタリアンカテゴリのデータを使用
    italian_data = df[df['CATEGORY'] == 'イタリアン'].copy()
    
    # Prophet用データに変換
    prophet_data = italian_data[['DATE', 'target_amount']].copy()
    prophet_data.columns = ['ds', 'y']
    
    print("=== 入力データ ===")
    print(prophet_data.head())
    print(f"データ型: {prophet_data.dtypes}")
    
    # Prophetモデルを作成
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # モデル学習
    model.fit(prophet_data)
    
    # 未来データフレーム作成
    future = model.make_future_dataframe(periods=6, freq='M')
    
    # 予測実行
    forecast = model.predict(future)
    
    print("\n=== 予測結果 ===")
    print("予測結果の列名:")
    print(forecast.columns.tolist())
    
    print("\n予測結果の最初の5行:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    
    print("\n予測結果の最後の5行（未来予測）:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    print("\n=== データ型確認 ===")
    print(f"ds列の型: {forecast['ds'].dtype}")
    print(f"yhat列の型: {forecast['yhat'].dtype}")
    print(f"yhat_lower列の型: {forecast['yhat_lower'].dtype}")
    print(f"yhat_upper列の型: {forecast['yhat_upper'].dtype}")
    
    # 未来予測のみを抽出して表示
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
    future_forecast.columns = ['日付', '予測売上', '予測下限', '予測上限']
    
    print("\n=== 未来予測（日本語列名） ===")
    print(future_forecast)
    
    print("\n=== 数値形式での表示 ===")
    print(future_forecast.round(0))

if __name__ == "__main__":
    test_prophet_output() 