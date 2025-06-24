import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
from font_setup import setup_japanese_font
setup_japanese_font()

def load_and_explore_data(file_path):
    """
    データを読み込み、基本的な探索的分析を実行
    """
    # データ読み込み
    df = pd.read_csv(file_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    print("=== データ概要 ===")
    print(f"データ形状: {df.shape}")
    print(f"期間: {df['DATE'].min()} 〜 {df['DATE'].max()}")
    print(f"カテゴリ数: {df['CATEGORY'].nunique()}")
    print(f"カテゴリ: {list(df['CATEGORY'].unique())}")
    
    print("\n=== 基本統計 ===")
    print(df.describe())
    
    print("\n=== カテゴリ別データ数 ===")
    print(df['CATEGORY'].value_counts())
    
    return df

def plot_sales_trends(df):
    """
    売上トレンドの可視化
    """
    plt.figure(figsize=(15, 10))
    
    # カテゴリ別売上トレンド
    plt.subplot(2, 2, 1)
    for category in df['CATEGORY'].unique():
        category_data = df[df['CATEGORY'] == category]
        plt.plot(category_data['DATE'], category_data['target_amount'], 
                label=category, marker='o', markersize=3)
    
    plt.title('カテゴリ別月間売上トレンド')
    plt.xlabel('日付')
    plt.ylabel('月間売上（円）')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # カテゴリ別平均売上
    plt.subplot(2, 2, 2)
    avg_sales = df.groupby('CATEGORY')['target_amount'].mean().sort_values(ascending=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(avg_sales)))
    bars = plt.bar(avg_sales.index, avg_sales.values, color=colors)
    plt.title('カテゴリ別平均月間売上')
    plt.xlabel('カテゴリ')
    plt.ylabel('平均月間売上（円）')
    plt.xticks(rotation=45)
    
    # 売上値をバーの上に表示
    for bar, value in zip(bars, avg_sales.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f'{value:,.0f}', ha='center', va='bottom')
    
    # 月別売上パターン
    plt.subplot(2, 2, 3)
    monthly_avg = df.groupby('MONTH')['target_amount'].mean()
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=6)
    plt.title('月別平均売上パターン')
    plt.xlabel('月')
    plt.ylabel('平均月間売上（円）')
    plt.grid(True, alpha=0.3)
    
    # 年別売上比較
    plt.subplot(2, 2, 4)
    yearly_avg = df.groupby('YEAR')['target_amount'].mean()
    plt.bar(yearly_avg.index, yearly_avg.values, color=['skyblue', 'lightcoral'])
    plt.title('年別平均月間売上')
    plt.xlabel('年')
    plt.ylabel('平均月間売上（円）')
    
    # 売上値をバーの上に表示
    for i, value in enumerate(yearly_avg.values):
        plt.text(yearly_avg.index[i], value + 10000, f'{value:,.0f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sales_trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_analysis(df):
    """
    特徴量の分析と可視化
    """
    plt.figure(figsize=(15, 12))
    
    # 人流と売上の関係
    plt.subplot(2, 3, 1)
    for category in df['CATEGORY'].unique():
        category_data = df[df['CATEGORY'] == category]
        plt.scatter(category_data['AVG_MONTHLY_POPULATION'], 
                   category_data['target_amount'], 
                   label=category, alpha=0.6)
    
    plt.title('人流と売上の関係')
    plt.xlabel('月間平均人流')
    plt.ylabel('月間売上（円）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 評価スコアと売上の関係
    plt.subplot(2, 3, 2)
    for category in df['CATEGORY'].unique():
        category_data = df[df['CATEGORY'] == category]
        plt.scatter(category_data['RATING_SCORE'], 
                   category_data['target_amount'], 
                   label=category, alpha=0.6)
    
    plt.title('食べログ評価と売上の関係')
    plt.xlabel('食べログ評価スコア')
    plt.ylabel('月間売上（円）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 席数と売上の関係
    plt.subplot(2, 3, 3)
    for category in df['CATEGORY'].unique():
        category_data = df[df['CATEGORY'] == category]
        plt.scatter(category_data['NUM_SEATS'], 
                   category_data['target_amount'], 
                   label=category, alpha=0.6)
    
    plt.title('席数と売上の関係')
    plt.xlabel('席数')
    plt.ylabel('月間売上（円）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # カテゴリ別特徴量の分布
    plt.subplot(2, 3, 4)
    df.boxplot(column='AVG_MONTHLY_POPULATION', by='CATEGORY', ax=plt.gca())
    plt.title('カテゴリ別人流分布')
    plt.suptitle('')  # デフォルトタイトルを削除
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 5)
    df.boxplot(column='RATING_SCORE', by='CATEGORY', ax=plt.gca())
    plt.title('カテゴリ別評価スコア分布')
    plt.suptitle('')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 6)
    df.boxplot(column='NUM_SEATS', by='CATEGORY', ax=plt.gca())
    plt.title('カテゴリ別席数分布')
    plt.suptitle('')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df):
    """
    相関分析
    """
    # 数値列のみを選択
    numeric_cols = ['target_amount', 'AVG_MONTHLY_POPULATION', 'RATING_SCORE', 
                   'RATING_CNT', 'NUM_SEATS']
    
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('特徴量間の相関係数')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== 相関分析結果 ===")
    print("売上との相関係数:")
    sales_corr = correlation_matrix['target_amount'].sort_values(ascending=False)
    for feature, corr in sales_corr.items():
        if feature != 'target_amount':
            print(f"{feature}: {corr:.3f}")

def seasonal_analysis(df):
    """
    季節性分析
    """
    plt.figure(figsize=(15, 10))
    
    # カテゴリ別月別平均売上
    monthly_category_avg = df.pivot_table(
        values='target_amount', 
        index='MONTH', 
        columns='CATEGORY', 
        aggfunc='mean'
    )
    
    plt.subplot(2, 2, 1)
    for category in monthly_category_avg.columns:
        plt.plot(monthly_category_avg.index, monthly_category_avg[category], 
                marker='o', label=category, linewidth=2)
    
    plt.title('カテゴリ別月別平均売上パターン')
    plt.xlabel('月')
    plt.ylabel('平均月間売上（円）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 季節性の強さを計算
    plt.subplot(2, 2, 2)
    seasonal_strength = {}
    for category in df['CATEGORY'].unique():
        category_data = df[df['CATEGORY'] == category]
        monthly_std = category_data.groupby('MONTH')['target_amount'].std()
        monthly_mean = category_data.groupby('MONTH')['target_amount'].mean()
        seasonal_strength[category] = (monthly_std / monthly_mean).mean()
    
    strength_df = pd.Series(seasonal_strength).sort_values(ascending=False)
    plt.bar(strength_df.index, strength_df.values, color=plt.cm.Set3(np.linspace(0, 1, len(strength_df))))
    plt.title('カテゴリ別季節性の強さ')
    plt.xlabel('カテゴリ')
    plt.ylabel('季節性指標（変動係数）')
    plt.xticks(rotation=45)
    
    # 最寄り駅別売上
    plt.subplot(2, 2, 3)
    station_sales = df.groupby('NEAREST_STATION_INFO')['target_amount'].mean().sort_values(ascending=False)
    plt.bar(station_sales.index, station_sales.values, color=plt.cm.Set3(np.linspace(0, 1, len(station_sales))))
    plt.title('最寄り駅別平均売上')
    plt.xlabel('最寄り駅')
    plt.ylabel('平均月間売上（円）')
    plt.xticks(rotation=45)
    
    # 評価数と売上の関係
    plt.subplot(2, 2, 4)
    plt.scatter(df['RATING_CNT'], df['target_amount'], alpha=0.5)
    plt.title('評価数と売上の関係')
    plt.xlabel('食べログ評価数')
    plt.ylabel('月間売上（円）')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    メイン実行関数
    """
    try:
        # データ読み込みと基本分析
        df = load_and_explore_data('restaurant_sales_data.csv')
        
        # 売上トレンド分析
        print("\n売上トレンド分析中...")
        plot_sales_trends(df)
        
        # 特徴量分析
        print("\n特徴量分析中...")
        plot_feature_analysis(df)
        
        # 相関分析
        print("\n相関分析中...")
        correlation_analysis(df)
        
        # 季節性分析
        print("\n季節性分析中...")
        seasonal_analysis(df)
        
        print("\n分析完了！以下のファイルが生成されました:")
        print("- sales_trends_analysis.png")
        print("- feature_analysis.png")
        print("- correlation_matrix.png")
        print("- seasonal_analysis.png")
        
    except FileNotFoundError:
        print("データファイルが見つかりません。まずサンプルデータを生成してください。")
        print("python generate_sample_data.py を実行してください。")

if __name__ == "__main__":
    main() 