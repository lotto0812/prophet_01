#!/usr/bin/env python3
"""
レストラン売上予測モデルの統合実行スクリプト
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """
    コマンドを実行し、結果を表示
    """
    print(f"\n{'='*50}")
    print(f"実行中: {description}")
    print(f"コマンド: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功")
        if result.stdout:
            print("出力:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ エラーが発生しました")
        print(f"エラーコード: {e.returncode}")
        if e.stdout:
            print("標準出力:")
            print(e.stdout)
        if e.stderr:
            print("エラー出力:")
            print(e.stderr)
        return False

def check_file_exists(filename):
    """
    ファイルの存在確認
    """
    return os.path.exists(filename)

def main():
    """
    メイン実行関数
    """
    print("🍽️ レストランカテゴリ別売上予測モデル")
    print("Prophetを使用した時系列予測システム")
    print("\n" + "="*60)
    
    # ステップ1: サンプルデータ生成
    if not check_file_exists('restaurant_sales_data.csv'):
        print("\n📊 ステップ1: サンプルデータの生成")
        if not run_command('python generate_sample_data.py', 'サンプルデータ生成'):
            print("データ生成に失敗しました。処理を中止します。")
            return
    else:
        print("\n📊 ステップ1: サンプルデータは既に存在します")
    
    # ステップ2: データ分析
    print("\n📈 ステップ2: データ分析と可視化")
    if not run_command('python data_analysis.py', 'データ分析'):
        print("データ分析に失敗しました。処理を中止します。")
        return
    
    # ステップ3: Prophet予測モデル
    print("\n🔮 ステップ3: Prophet予測モデルの実行")
    if not run_command('python prophet_sales_forecast.py', 'Prophet予測モデル'):
        print("予測モデルの実行に失敗しました。")
        return
    
    # 完了メッセージ
    print("\n" + "="*60)
    print("🎉 全ての処理が完了しました！")
    print("\n生成されたファイル:")
    
    # 生成されたファイルの確認
    output_files = [
        'restaurant_sales_data.csv',
        'sales_trends_analysis.png',
        'feature_analysis.png',
        'correlation_matrix.png',
        'seasonal_analysis.png',
        'model_performance.csv'
    ]
    
    for file in output_files:
        if check_file_exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (見つかりません)")
    
    # 予測ファイルの確認
    categories = ['イタリアン', '中華', '和食', 'フレンチ', 'カフェ']
    for category in categories:
        forecast_file = f'forecast_{category}.csv'
        if check_file_exists(forecast_file):
            print(f"✅ {forecast_file}")
    
    print("\n📋 次のステップ:")
    print("1. 生成されたグラフを確認してデータの特徴を理解")
    print("2. model_performance.csvでモデル性能を確認")
    print("3. forecast_[カテゴリ].csvで予測結果を確認")
    print("4. 必要に応じてパラメータを調整して再実行")

if __name__ == "__main__":
    main() 