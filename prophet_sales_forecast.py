import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
from font_setup import setup_japanese_font
setup_japanese_font()

class RestaurantSalesForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
    def load_data(self, file_path):
        """
        データを読み込み、Prophet用にフォーマット
        """
        self.data = pd.read_csv(file_path)
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        print(f"データ読み込み完了: {self.data.shape}")
        print(f"カテゴリ: {self.data['CATEGORY'].unique()}")
        
    def prepare_prophet_data(self, category_data):
        """
        Prophet用のデータフォーマットに変換
        """
        prophet_data = category_data[['DATE', 'target_amount']].copy()
        prophet_data.columns = ['ds', 'y']  # Prophetの標準列名
        return prophet_data
        
    def create_prophet_model(self, category, regressors=None):
        """
        カテゴリ別のProphetモデルを作成
        """
        # カテゴリデータを抽出
        category_data = self.data[self.data['CATEGORY'] == category].copy()
        
        # Prophet用データに変換
        prophet_data = self.prepare_prophet_data(category_data)
        
        # Prophetモデルの初期化
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # 追加の回帰変数がある場合
        if regressors:
            for regressor in regressors:
                if regressor in category_data.columns:
                    # 回帰変数をProphetデータに追加
                    prophet_data[regressor] = category_data[regressor].values
                    # モデルに回帰変数を追加
                    model.add_regressor(regressor)
        
        # モデルの学習
        model.fit(prophet_data)
        
        return model, prophet_data
        
    def train_models(self, regressors=None):
        """
        全カテゴリのモデルを学習
        """
        # 回帰変数をクラス内に保存
        self.regressors = regressors
        
        categories = self.data['CATEGORY'].unique()
        
        for category in categories:
            print(f"\n{category}カテゴリのモデルを学習中...")
            
            # モデル作成
            model, prophet_data = self.create_prophet_model(category, regressors)
            
            # モデル保存
            self.models[category] = model
            
            # 予測用の未来データフレーム作成
            future = model.make_future_dataframe(periods=12, freq='M')  # 12ヶ月先まで予測
            
            # 回帰変数がある場合、未来データにも追加
            if regressors:
                for regressor in regressors:
                    if regressor in self.data.columns:
                        # 未来の回帰変数値（ここでは平均値を使用）
                        future[regressor] = self.data[regressor].mean()
            
            # 予測実行
            forecast = model.predict(future)
            self.forecasts[category] = forecast
            
            print(f"{category}カテゴリの学習完了")
            
    def evaluate_models(self):
        """
        モデルの性能評価
        """
        for category in self.models.keys():
            # 実際のデータ
            actual_data = self.data[self.data['CATEGORY'] == category]
            actual_sales = actual_data['target_amount'].values
            
            # 予測データ（学習期間のみ）
            forecast_data = self.forecasts[category]
            predicted_sales = forecast_data['yhat'].values[:len(actual_sales)]
            
            # 性能指標計算
            mae = mean_absolute_error(actual_sales, predicted_sales)
            rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))
            r2 = r2_score(actual_sales, predicted_sales)
            
            self.performance_metrics[category] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': np.mean(np.abs((actual_sales - predicted_sales) / actual_sales)) * 100
            }
            
            print(f"\n{category}カテゴリの性能指標:")
            print(f"MAE: {mae:,.0f}円")
            print(f"RMSE: {rmse:,.0f}円")
            print(f"R2: {r2:.3f}")
            print(f"MAPE: {self.performance_metrics[category]['MAPE']:.1f}%")
            
    def plot_forecasts(self, save_plots=True):
        """
        予測結果の可視化
        """
        for category in self.models.keys():
            # Prophetの標準プロット
            fig = self.models[category].plot(self.forecasts[category])
            plt.title(f'{category}カテゴリ - 売上予測')
            plt.xlabel('日付')
            plt.ylabel('月間売上（円）')
            
            if save_plots:
                plt.savefig(f'forecast_{category}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 成分分解プロット
            fig2 = self.models[category].plot_components(self.forecasts[category])
            plt.suptitle(f'{category}カテゴリ - 予測成分分解')
            
            if save_plots:
                plt.savefig(f'components_{category}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def get_future_forecast(self, category, periods=12):
        """
        特定カテゴリの未来予測を取得
        """
        if category not in self.models:
            raise ValueError(f"カテゴリ '{category}' のモデルが見つかりません")
            
        future = self.models[category].make_future_dataframe(periods=periods, freq='M')
        
        # 回帰変数がある場合、未来データにも追加
        if hasattr(self, 'regressors') and self.regressors:
            for regressor in self.regressors:
                if regressor in self.data.columns:
                    # 未来の回帰変数値（ここでは平均値を使用）
                    future[regressor] = self.data[regressor].mean()
        
        forecast = self.models[category].predict(future)
        
        # 未来の予測のみを抽出
        future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        future_forecast.columns = ['日付', '予測売上', '予測下限', '予測上限']
        
        return future_forecast
        
    def save_results(self):
        """
        結果をCSVファイルとして保存
        """
        # 性能指標をDataFrameに変換
        performance_df = pd.DataFrame(self.performance_metrics).T
        performance_df.to_csv('model_performance.csv')
        
        # 各カテゴリの未来予測を保存
        for category in self.models.keys():
            future_forecast = self.get_future_forecast(category)
            
            # 数値フォーマットを改善
            formatted_forecast = future_forecast.copy()
            formatted_forecast['予測売上'] = formatted_forecast['予測売上'].round(0).astype(int)
            formatted_forecast['予測下限'] = formatted_forecast['予測下限'].round(0).astype(int)
            formatted_forecast['予測上限'] = formatted_forecast['予測上限'].round(0).astype(int)
            
            formatted_forecast.to_csv(f'forecast_{category}.csv', index=False)
            
        print("結果を保存しました:")
        print("- model_performance.csv: モデル性能指標")
        print("- forecast_[カテゴリ].csv: 各カテゴリの予測結果")

def main():
    # フォアキャスターの初期化
    forecaster = RestaurantSalesForecaster()
    
    # データ読み込み
    try:
        forecaster.load_data('restaurant_sales_data.csv')
    except FileNotFoundError:
        print("データファイルが見つかりません。まずサンプルデータを生成してください。")
        print("python generate_sample_data.py を実行してください。")
        return
    
    # 回帰変数の設定（オプション）
    regressors = ['AVG_MONTHLY_POPULATION', 'RATING_SCORE', 'RATING_CNT', 'NUM_SEATS']
    
    # モデル学習
    print("Prophetモデルの学習を開始します...")
    forecaster.train_models(regressors)
    
    # モデル評価
    print("\nモデル性能評価中...")
    forecaster.evaluate_models()
    
    # 予測結果の可視化
    print("\n予測結果を可視化中...")
    forecaster.plot_forecasts()
    
    # 結果保存
    forecaster.save_results()
    
    # 未来予測の表示
    print("\n=== 未来12ヶ月の予測 ===")
    for category in forecaster.models.keys():
        print(f"\n{category}カテゴリ:")
        future_forecast = forecaster.get_future_forecast(category)
        
        # 数値フォーマットを改善
        formatted_forecast = future_forecast.copy()
        formatted_forecast['予測売上'] = formatted_forecast['予測売上'].round(0).astype(int)
        formatted_forecast['予測下限'] = formatted_forecast['予測下限'].round(0).astype(int)
        formatted_forecast['予測上限'] = formatted_forecast['予測上限'].round(0).astype(int)
        
        print(formatted_forecast)
        
        # 統計情報も表示
        print(f"平均予測売上: {formatted_forecast['予測売上'].mean():,.0f}円")
        print(f"予測売上の範囲: {formatted_forecast['予測売上'].min():,.0f}円 〜 {formatted_forecast['予測売上'].max():,.0f}円")

if __name__ == "__main__":
    main() 