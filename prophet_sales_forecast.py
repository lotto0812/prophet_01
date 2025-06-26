import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIエラーを回避するためにAggバックエンドを使用
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
import matplotlib.font_manager as fm
try:
    plt.rcParams['font.family'] = 'Yu Gothic'
    print("日本語フォントを設定しました: Yu Gothic")
except:
    try:
        plt.rcParams['font.family'] = 'MS Gothic'
        print("日本語フォントを設定しました: MS Gothic")
    except:
        print("日本語フォントの設定に失敗しました")

# === 前処理関数 ===
def preprocess_features(df):
    """
    指定カラムのデータフレームをProphet用に前処理する
    - 日付整形
    - 創業年数
    - bool/カテゴリ変換
    - OneHotエンコーディング
    """
    proc = df.copy()
    # 日付整形
    if 'YEAR' in proc.columns and 'MONTH' in proc.columns:
        proc['DATE'] = pd.to_datetime(proc['YEAR'].astype(str) + '-' + proc['MONTH'].astype(str).str.zfill(2) + '-01')
    else:
        proc['DATE'] = pd.to_datetime(proc['DATE'])
    # 創業年数
    proc['OPENING_DATE'] = pd.to_datetime(proc['OPENING_DATE'], errors='coerce')
    proc['創業年数'] = proc['DATE'].dt.year - proc['OPENING_DATE'].dt.year
    proc['創業年数'] = proc['創業年数'].fillna(proc['創業年数'].median())
    # bool変換
    proc['DINNER_INFO'] = proc['DINNER_INFO'].apply(lambda x: 1 if str(x).startswith('営業') else 0)
    proc['LUNCH_INFO'] = proc['LUNCH_INFO'].apply(lambda x: 1 if str(x).startswith('営業') else 0)
    proc['HOME_PAGE_URL'] = proc['HOME_PAGE_URL'].apply(lambda x: 0 if pd.isna(x) or x=='' else 1)
    proc['PHONE_NUM'] = proc['PHONE_NUM'].apply(lambda x: 1 if pd.notna(x) and '-' in str(x) else 0)
    proc['RESERVATION_POSSIBILITY_INFO'] = proc['RESERVATION_POSSIBILITY_INFO'].apply(lambda x: 1 if '可' in str(x) else 0)
    # 最大予約人数の空欄補完
    proc['MAX_NUM_PEOPLE_FOR_RESERVATION'] = proc['MAX_NUM_PEOPLE_FOR_RESERVATION'].fillna(proc['NUM_SEATS'])
    # カテゴリ変数のOneHotエンコーディング（CITY, CUISINE_CAT_1）
    cat_cols = ['CITY', 'CUISINE_CAT_1']
    proc = pd.get_dummies(proc, columns=cat_cols, drop_first=True)
    return proc

def create_lag_features(df, target_col='target_amount', max_lags=6):
    """
    ラグ特徴量を作成する（過去1〜6ヶ月の売上データ）
    """
    df_lagged = df.copy()
    
    # 日付でソート
    df_lagged = df_lagged.sort_values('DATE')
    
    # ラグ特徴量を作成
    for lag in range(1, max_lags + 1):
        df_lagged[f'{target_col}_lag_{lag}'] = df_lagged[target_col].shift(lag)
    
    return df_lagged

class RestaurantSalesForecaster:
    """
    レストラン売上予測クラス
    ProphetとRandomForestを使用して時系列予測を行います
    """
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        self.data = None
        self.proc_data = None
        self.regressors = None

    def load_data(self, file_path):
        """
        指定カラムのデータを読み込み、前処理を実行
        """
        # 指定カラムのみ読み込み
        usecols = [
            'YEAR','MONTH','sakaya_cd','sakaya_dealer_cd','sakaya_dealer_nm','SAKAYA_MATCH_FLAG','RN','RST_ADDRESS','PREFECTURE','CITY','DISTRICT','NEAREST_STATION_INFO','LATITUDE','LONGITUDE','MESH50MID','AVG_MONTHLY_POPULATION','RST_CAT','CUISINE_CAT_1','CUISINE_CAT_2','CUISINE_CAT_3','RATING_CNT','RATING_SCORE','DINNER_INFO','LUNCH_INFO','HOME_PAGE_URL','RST_URL','PHONE_NUM','BUSINESS_HOURS_INFO','NUM_SEATS','MAX_NUM_PEOPLE_FOR_RESERVATION','RESERVATION_POSSIBILITY_INFO','HOLIDAY_INFO','OPENING_DATE','RESERVATION_NUM','CATEGORY','target_amount'
        ]
        self.data = pd.read_csv(file_path, usecols=usecols)
        print(f"データ読み込み完了: {self.data.shape}")
        # 前処理
        self.proc_data = preprocess_features(self.data)
        print(f"前処理後データ: {self.proc_data.shape}")

    def train_and_evaluate_per_shop(self, regressors=None, save_prefix="shop_"):
        """
        店舗ごとにProphetモデルを学習・評価・保存
        進捗（何件中何件、何%）を表示
        """
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        results = []
        groups = self.proc_data['sakaya_dealer_cd'].unique()  # 全店舗のIDリスト
        total = len(groups)  # 店舗数
        for idx, group in enumerate(groups, 1):
            percent = int(idx / total * 100)
            # 進捗を表示
            print(f"[{idx}/{total}件目 {percent}%] 店舗 {group} のモデルを学習中...")
            # この店舗のデータだけ抽出
            group_df = self.proc_data[self.proc_data['sakaya_dealer_cd'] == group].copy()
            # Prophet用データ（ds:日付, y:売上）を作成
            prophet_data = group_df[['DATE', 'target_amount']].copy()
            prophet_data.columns = ['ds', 'y']
            # 回帰変数（特徴量）を追加
            if regressors:
                for reg in regressors:
                    if reg in group_df.columns:
                        prophet_data[reg] = group_df[reg].values
            # Prophetモデルを初期化
            # 年次季節性のみ有効、乗法的季節性、変化点・季節性の強さも指定
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            # 回帰変数（特徴量）をモデルに追加
            if regressors:
                for reg in regressors:
                    if reg in group_df.columns:
                        model.add_regressor(reg)
            # モデル学習（fit）
            model.fit(prophet_data)
            self.models[group] = model
            # 未来12ヶ月分の予測用データフレームを作成
            future = model.make_future_dataframe(periods=12, freq='M')
            # 未来データにも回帰変数（特徴量）の平均値をセット
            if regressors:
                for reg in regressors:
                    if reg in group_df.columns:
                        future[reg] = group_df[reg].mean()
            # 予測実行
            forecast = model.predict(future)
            self.forecasts[group] = forecast
            # モデルの性能評価（学習期間のみ）
            actual = group_df['target_amount'].values  # 実際の売上
            pred = forecast['yhat'].values[:len(actual)]  # 予測値
            # 各種指標を計算
            mae = mean_absolute_error(actual, pred)  # 平均絶対誤差
            rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
            r2 = r2_score(actual, pred)  # 決定係数
            mape = np.mean(np.abs((actual - pred) / actual)) * 100  # 平均絶対パーセント誤差
            self.performance_metrics[group] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"MAE: {mae:,.0f}円, RMSE: {rmse:,.0f}円, R2: {r2:.3f}, MAPE: {mape:.1f}%")
            
            # 未来12ヶ月の予測結果を取得
            future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
            future_forecast_values = future_forecast['yhat'].round(0).astype(int).tolist()
            
            # 結果をリストに追加（1店舗1行）
            result_row = {
                'sakaya_dealer_cd': group,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            # 未来12ヶ月の予測値を列として追加
            for i, value in enumerate(future_forecast_values, 1):
                result_row[f'forecast_month_{i}'] = value
            
            results.append(result_row)
        
        # 全店舗分の性能指標と予測結果をまとめてCSV保存
        pd.DataFrame(results).to_csv(f'{save_prefix}model_performance.csv', index=False)
        print(f"\n{save_prefix}model_performance.csv を保存しました（1店舗1行形式）")

    def analyze_feature_importance_simple(self, category, regressors, save_prefix="cat_"):
        """
        特徴量重要度を簡易分析する（相関ベース）
        より軽量で高速な分析
        """
        print(f"\n{category} の特徴量重要度を簡易分析中...")
        
        # このカテゴリのデータだけ抽出（元のデータから）
        group_df = self.data[self.data['CUISINE_CAT_1'] == category].copy()
        
        # 数値型の特徴量のみを対象とする
        numeric_features = []
        feature_correlations = {}
        
        for reg in regressors:
            if reg in group_df.columns:
                # 数値型かチェック
                if pd.api.types.is_numeric_dtype(group_df[reg]):
                    numeric_features.append(reg)
                    # 売上との相関係数を計算
                    correlation = group_df[reg].corr(group_df['target_amount'])
                    feature_correlations[reg] = abs(correlation)  # 絶対値で重要度を評価
        
        if not feature_correlations:
            print(f"{category}: 分析可能な数値特徴量が見つかりませんでした")
            return None
        
        # 相関係数の絶対値でソート
        sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
        
        # 結果をDataFrameに変換
        importance_df = pd.DataFrame(sorted_features, columns=['特徴量', '重要度（相関係数絶対値）'])
        
        # グラフ作成
        plt.figure(figsize=(10, 6))
        features = [x[0] for x in sorted_features]
        importance_values = [x[1] for x in sorted_features]
        
        bars = plt.barh(range(len(features)), importance_values, color='skyblue', alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('重要度（相関係数絶対値）')
        plt.title(f'{category} - 特徴量重要度分析（簡易版）')
        plt.grid(True, alpha=0.3)
        
        # 値ラベルを追加
        for i, (bar, value) in enumerate(zip(bars, importance_values)):
            plt.text(value + 0.01, i, f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(f'{save_prefix}feature_importance_{category}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # CSV保存
        importance_df.to_csv(f'{save_prefix}feature_importance_{category}.csv', index=False)
        
        print(f"特徴量重要度分析完了: {save_prefix}feature_importance_{category}.png, {save_prefix}feature_importance_{category}.csv")
        
        return importance_df

    def train_and_evaluate_per_category(self, regressors=None, save_prefix="cat_", test_mode=False, analyze_features=False):
        """
        カテゴリごとにProphetモデルを学習・評価・保存
        グラフも出力（カテゴリ数が少ないため）
        """
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        results = []
        # 元のデータからカテゴリを取得（OneHotエンコーディング前）
        groups = self.data['CUISINE_CAT_1'].unique()  # 全カテゴリのリスト
        if test_mode:
            groups = groups[:3]  # テスト用：最初の3つのカテゴリのみ
            print(f"テストモード: 最初の3つのカテゴリのみ処理します")
        total = len(groups)  # カテゴリ数
        for idx, group in enumerate(groups, 1):
            percent = int(idx / total * 100)
            # 進捗を表示
            print(f"[{idx}/{total}件目 {percent}%] カテゴリ {group} のモデルを学習中...")
            # このカテゴリのデータだけ抽出（元のデータから）
            group_df = self.data[self.data['CUISINE_CAT_1'] == group].copy()
            # 日付整形
            if 'YEAR' in group_df.columns and 'MONTH' in group_df.columns:
                group_df['DATE'] = pd.to_datetime(group_df['YEAR'].astype(str) + '-' + group_df['MONTH'].astype(str).str.zfill(2) + '-01')
            # カテゴリごとに月次で集計
            monthly_data = group_df.groupby('DATE')['target_amount'].sum().reset_index()
            # Prophet用データ（ds:日付, y:売上）を作成
            prophet_data = monthly_data[['DATE', 'target_amount']].copy()
            prophet_data.columns = ['ds', 'y']
            # Prophetモデルを初期化（回帰変数なしでシンプルに）
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            # モデル学習（fit）
            model.fit(prophet_data)
            self.models[group] = model
            # 未来12ヶ月分の予測用データフレームを作成
            future = model.make_future_dataframe(periods=12, freq='M')
            # 予測実行
            forecast = model.predict(future)
            self.forecasts[group] = forecast
            # モデルの性能評価（学習期間のみ）
            actual = prophet_data['y'].values  # 実際の売上
            pred = forecast['yhat'].values[:len(actual)]  # 予測値
            # 各種指標を計算
            mae = mean_absolute_error(actual, pred)  # 平均絶対誤差
            rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
            r2 = r2_score(actual, pred)  # 決定係数
            mape = np.mean(np.abs((actual - pred) / actual)) * 100  # 平均絶対パーセント誤差
            self.performance_metrics[group] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"MAE: {mae:,.0f}円, RMSE: {rmse:,.0f}円, R2: {r2:.3f}, MAPE: {mape:.1f}%")
            
            # 未来12ヶ月の予測結果を取得
            future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
            future_forecast_values = future_forecast['yhat'].round(0).astype(int).tolist()
            
            # 結果をリストに追加（1カテゴリ1行）
            result_row = {
                'CUISINE_CAT_1': group,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }
            # 未来12ヶ月の予測値を列として追加
            for i, value in enumerate(future_forecast_values, 1):
                result_row[f'forecast_month_{i}'] = value
            
            results.append(result_row)
            
            # 予測結果のグラフを保存
            fig = model.plot(forecast)
            plt.title(f'{group} - 売上予測')
            plt.xlabel('日付')
            plt.ylabel('月間売上（円）')
            plt.savefig(f'{save_prefix}forecast_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 成分分解（トレンド・季節性など）グラフも保存
            fig2 = model.plot_components(forecast)
            plt.suptitle(f'{group} - 予測成分分解（トレンド・季節性）')
            plt.savefig(f'{save_prefix}components_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # トレンドのみのグラフ
            fig3, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast['ds'], forecast['trend'], 'b-', linewidth=2, label='トレンド')
            ax.fill_between(forecast['ds'], forecast['trend_lower'], forecast['trend_upper'], 
                           alpha=0.3, color='blue', label='トレンド信頼区間')
            ax.set_title(f'{group} - トレンド分析')
            ax.set_xlabel('日付')
            ax.set_ylabel('売上（円）')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}trend_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 季節性のみのグラフ
            fig4, ax = plt.subplots(figsize=(12, 6))
            ax.plot(forecast['ds'], forecast['yearly'], 'g-', linewidth=2, label='年次季節性')
            ax.set_title(f'{group} - 年次季節性分析')
            ax.set_xlabel('日付')
            ax.set_ylabel('季節性効果')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}seasonality_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 特徴量重要度分析（オプション）
            if analyze_features and regressors:
                self.analyze_feature_importance_simple(group, regressors, save_prefix)
            
            # 未来12ヶ月の予測結果をCSV保存
            future_forecast_csv = future_forecast.copy()
            future_forecast_csv.columns = ['日付', '予測売上', '予測下限', '予測上限']
            future_forecast_csv['予測売上'] = future_forecast_csv['予測売上'].round(0).astype(int)
            future_forecast_csv['予測下限'] = future_forecast_csv['予測下限'].round(0).astype(int)
            future_forecast_csv['予測上限'] = future_forecast_csv['予測上限'].round(0).astype(int)
            future_forecast_csv.to_csv(f'{save_prefix}forecast_{group}.csv', index=False)
        
        # 全カテゴリ分の性能指標と予測結果をまとめてCSV保存
        pd.DataFrame(results).to_csv(f'{save_prefix}model_performance.csv', index=False)
        output_files = f"{save_prefix}model_performance.csv, {save_prefix}forecast_[category].csv, {save_prefix}forecast_[category].png, {save_prefix}components_[category].png, {save_prefix}trend_[category].png, {save_prefix}seasonality_[category].png"
        if analyze_features:
            output_files += f", {save_prefix}feature_importance_[category].png"
        print(f"\n{output_files} などを保存しました")

    def train_and_evaluate_randomforest_per_category(self, regressors=None, save_prefix="rf_cat_", test_mode=False, max_lags=6):
        """
        カテゴリごとにRandomForestモデルを学習・評価・保存
        """
        self.rf_models = {}
        self.rf_forecasts = {}
        self.rf_performance_metrics = {}
        self.rf_scalers = {}  # スケーラーを保存
        results = []
        
        # 元のデータからカテゴリを取得（OneHotエンコーディング前）
        groups = self.data['CUISINE_CAT_1'].unique()  # 全カテゴリのリスト
        if test_mode:
            groups = groups[:3]  # テスト用：最初の3つのカテゴリのみ
            print(f"テストモード: 最初の3つのカテゴリのみ処理します")
        
        total = len(groups)  # カテゴリ数
        
        for idx, group in enumerate(groups, 1):
            percent = int(idx / total * 100)
            # 進捗を表示
            print(f"[{idx}/{total}件目 {percent}%] カテゴリ {group} のRandomForestモデルを学習中...")
            
            # このカテゴリのデータだけ抽出（元のデータから）
            group_df = self.data[self.data['CUISINE_CAT_1'] == group].copy()
            
            # 日付整形
            if 'YEAR' in group_df.columns and 'MONTH' in group_df.columns:
                group_df['DATE'] = pd.to_datetime(group_df['YEAR'].astype(str) + '-' + group_df['MONTH'].astype(str).str.zfill(2) + '-01')
            
            # カテゴリごとに月次で集計
            monthly_data = group_df.groupby('DATE')['target_amount'].sum().reset_index()
            
            # ラグ特徴量を作成
            monthly_data_lagged = create_lag_features(monthly_data, 'target_amount', max_lags)
            
            # 特徴量を追加
            feature_data = monthly_data_lagged.copy()
            
            # 基本特徴量を追加（カテゴリごとの平均値を使用）
            if regressors:
                for reg in regressors:
                    if reg in group_df.columns:
                        # 数値型の特徴量のみを使用
                        if pd.api.types.is_numeric_dtype(group_df[reg]):
                            feature_data[reg] = group_df.groupby('DATE')[reg].mean().values
            
            # 欠損値を処理（ラグ特徴量の欠損値は削除）
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:  # データが少なすぎる場合はスキップ
                print(f"{group}: データが不足しています（{len(feature_data)}件）")
                continue
            
            # 特徴量とターゲットを分離
            feature_columns = [col for col in feature_data.columns if col not in ['DATE', 'target_amount']]
            X = feature_data[feature_columns]
            y = feature_data['target_amount']
            
            # データを分割（時系列なので時間順）
            split_idx = int(len(feature_data) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            # 特徴量のスケーリング
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.rf_scalers[group] = scaler
            
            # RandomForestハイパーパラメータチューニング
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # GridSearchCVで最適パラメータを探索
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # 最適パラメータでモデルを再訓練
            rf_model = RandomForestRegressor(
                **grid_search.best_params_,
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            self.rf_models[group] = rf_model
            
            # 予測
            y_pred_train = rf_model.predict(X_train_scaled)
            y_pred_test = rf_model.predict(X_test_scaled)
            
            # 性能評価
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            self.rf_performance_metrics[group] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"MAE: {mae:,.0f}円, RMSE: {rmse:,.0f}円, R2: {r2:.3f}, MAPE: {mape:.1f}%")
            print(f"最適パラメータ: {grid_search.best_params_}")
            
            # 未来12ヶ月の予測
            future_predictions = []
            last_data = feature_data.iloc[-1:].copy()
            
            for month in range(1, 13):
                # 次の月の特徴量を準備
                next_month_data = last_data.copy()
                
                # ラグ特徴量を更新
                for lag in range(max_lags, 0, -1):
                    if lag == 1:
                        next_month_data[f'target_amount_lag_{lag}'] = last_data['target_amount'].iloc[0]
                    else:
                        next_month_data[f'target_amount_lag_{lag}'] = last_data[f'target_amount_lag_{lag-1}'].iloc[0]
                
                # 予測実行（スケーリングを適用）
                next_features = next_month_data[feature_columns]
                next_features_scaled = scaler.transform(next_features)
                prediction = rf_model.predict(next_features_scaled)[0]
                future_predictions.append(prediction)
                
                # 次の予測のためにデータを更新
                last_data = next_month_data.copy()
                last_data['target_amount'] = prediction
            
            # 結果をリストに追加
            result_row = {
                'CUISINE_CAT_1': group,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Best_Params': str(grid_search.best_params_)
            }
            # 未来12ヶ月の予測値を列として追加
            for i, value in enumerate(future_predictions, 1):
                result_row[f'forecast_month_{i}'] = int(value)
            
            results.append(result_row)
            
            # グラフ作成
            # 1. 予測結果グラフ
            plt.figure(figsize=(12, 6))
            plt.plot(feature_data['DATE'], feature_data['target_amount'], 'b-', label='実際の売上', linewidth=2)
            
            # テストデータの日付を取得
            test_dates = feature_data['DATE'].iloc[split_idx:]
            plt.plot(test_dates, y_pred_test, 'r-', label='予測値（テスト）', linewidth=2)
            
            # 未来12ヶ月の予測を追加
            last_date = feature_data['DATE'].iloc[-1]
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
            plt.plot(future_dates, future_predictions, 'g--', label='未来12ヶ月予測', linewidth=2)
            
            plt.title(f'{group} - RandomForest売上予測')
            plt.xlabel('日付')
            plt.ylabel('月間売上（円）')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}forecast_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 特徴量重要度グラフ
            feature_importance = rf_model.feature_importances_
            feature_names = feature_columns
            
            # 重要度でソート
            importance_df = pd.DataFrame({
                '特徴量': feature_names,
                '重要度': feature_importance
            }).sort_values('重要度', ascending=False)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['重要度'], color='skyblue', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['特徴量'])
            plt.xlabel('重要度')
            plt.title(f'{group} - RandomForest特徴量重要度')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}feature_importance_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 予測vs実際の散布図
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('実際の売上')
            plt.ylabel('予測売上')
            plt.title(f'{group} - 予測vs実際（RandomForest）')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}scatter_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # CSV保存
            future_forecast_df = pd.DataFrame({
                '日付': future_dates,
                '予測売上': future_predictions
            })
            future_forecast_df['予測売上'] = future_forecast_df['予測売上'].round(0).astype(int)
            future_forecast_df.to_csv(f'{save_prefix}forecast_{group}.csv', index=False)
            
            # 特徴量重要度をCSV保存
            importance_df.to_csv(f'{save_prefix}feature_importance_{group}.csv', index=False)
        
        # 全カテゴリ分の性能指標と予測結果をまとめてCSV保存
        pd.DataFrame(results).to_csv(f'{save_prefix}model_performance.csv', index=False)
        print(f"\n{save_prefix}model_performance.csv, {save_prefix}forecast_[category].csv, {save_prefix}forecast_[category].png, {save_prefix}feature_importance_[category].png, {save_prefix}scatter_[category].png などを保存しました")

    def train_and_evaluate_randomforest_per_shop(self, regressors=None, save_prefix="rf_shop_", test_mode=False, max_lags=6):
        """
        店舗ごとにRandomForestモデルを学習・評価・保存
        """
        self.rf_models = {}
        self.rf_forecasts = {}
        self.rf_performance_metrics = {}
        self.rf_scalers = {}  # スケーラーを保存
        results = []
        
        # 元のデータから店舗を取得（OneHotエンコーディング前）
        groups = self.data['sakaya_dealer_cd'].unique()  # 全店舗のリスト
        if test_mode:
            groups = groups[:3]  # テスト用：最初の3つの店舗のみ
            print(f"テストモード: 最初の3つの店舗のみ処理します")
        
        total = len(groups)  # 店舗数
        
        for idx, group in enumerate(groups, 1):
            percent = int(idx / total * 100)
            # 進捗を表示
            print(f"[{idx}/{total}件目 {percent}%] 店舗 {group} のRandomForestモデルを学習中...")
            
            # この店舗のデータだけ抽出（元のデータから）
            group_df = self.data[self.data['sakaya_dealer_cd'] == group].copy()
            
            # 日付整形
            if 'YEAR' in group_df.columns and 'MONTH' in group_df.columns:
                group_df['DATE'] = pd.to_datetime(group_df['YEAR'].astype(str) + '-' + group_df['MONTH'].astype(str).str.zfill(2) + '-01')
            
            # カテゴリごとに月次で集計
            monthly_data = group_df.groupby('DATE')['target_amount'].sum().reset_index()
            
            # ラグ特徴量を作成
            monthly_data_lagged = create_lag_features(monthly_data, 'target_amount', max_lags)
            
            # 特徴量を追加
            feature_data = monthly_data_lagged.copy()
            
            # 基本特徴量を追加（店舗ごとの平均値を使用）
            if regressors:
                for reg in regressors:
                    if reg in group_df.columns:
                        # 数値型の特徴量のみを使用
                        if pd.api.types.is_numeric_dtype(group_df[reg]):
                            feature_data[reg] = group_df.groupby('DATE')[reg].mean().values
            
            # 欠損値を処理（ラグ特徴量の欠損値は削除）
            feature_data = feature_data.dropna()
            
            if len(feature_data) < 10:  # データが少なすぎる場合はスキップ
                print(f"{group}: データが不足しています（{len(feature_data)}件）")
                continue
            
            # 特徴量とターゲットを分離
            feature_columns = [col for col in feature_data.columns if col not in ['DATE', 'target_amount']]
            X = feature_data[feature_columns]
            y = feature_data['target_amount']
            
            # データを分割（時系列なので時間順）
            split_idx = int(len(feature_data) * 0.8)
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            # 特徴量のスケーリング
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.rf_scalers[group] = scaler
            
            # RandomForestハイパーパラメータチューニング
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # GridSearchCVで最適パラメータを探索
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            # 最適パラメータでモデルを再訓練
            rf_model = RandomForestRegressor(
                **grid_search.best_params_,
                random_state=42
            )
            rf_model.fit(X_train_scaled, y_train)
            self.rf_models[group] = rf_model
            
            # 予測
            y_pred_train = rf_model.predict(X_train_scaled)
            y_pred_test = rf_model.predict(X_test_scaled)
            
            # 性能評価
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
            
            self.rf_performance_metrics[group] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            print(f"MAE: {mae:,.0f}円, RMSE: {rmse:,.0f}円, R2: {r2:.3f}, MAPE: {mape:.1f}%")
            print(f"最適パラメータ: {grid_search.best_params_}")
            
            # 未来12ヶ月の予測
            future_predictions = []
            last_data = feature_data.iloc[-1:].copy()
            
            for month in range(1, 13):
                # 次の月の特徴量を準備
                next_month_data = last_data.copy()
                
                # ラグ特徴量を更新
                for lag in range(max_lags, 0, -1):
                    if lag == 1:
                        next_month_data[f'target_amount_lag_{lag}'] = last_data['target_amount'].iloc[0]
                    else:
                        next_month_data[f'target_amount_lag_{lag}'] = last_data[f'target_amount_lag_{lag-1}'].iloc[0]
                
                # 予測実行（スケーリングを適用）
                next_features = next_month_data[feature_columns]
                next_features_scaled = scaler.transform(next_features)
                prediction = rf_model.predict(next_features_scaled)[0]
                future_predictions.append(prediction)
                
                # 次の予測のためにデータを更新
                last_data = next_month_data.copy()
                last_data['target_amount'] = prediction
            
            # 結果をリストに追加
            result_row = {
                'sakaya_dealer_cd': group,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Best_Params': str(grid_search.best_params_)
            }
            # 未来12ヶ月の予測値を列として追加
            for i, value in enumerate(future_predictions, 1):
                result_row[f'forecast_month_{i}'] = int(value)
            
            results.append(result_row)
            
            # グラフ作成
            # 1. 予測結果グラフ
            plt.figure(figsize=(12, 6))
            plt.plot(feature_data['DATE'], feature_data['target_amount'], 'b-', label='実際の売上', linewidth=2)
            
            # テストデータの日付を取得
            test_dates = feature_data['DATE'].iloc[split_idx:]
            plt.plot(test_dates, y_pred_test, 'r-', label='予測値（テスト）', linewidth=2)
            
            # 未来12ヶ月の予測を追加
            last_date = feature_data['DATE'].iloc[-1]
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
            plt.plot(future_dates, future_predictions, 'g--', label='未来12ヶ月予測', linewidth=2)
            
            plt.title(f'{group} - RandomForest売上予測')
            plt.xlabel('日付')
            plt.ylabel('月間売上（円）')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}forecast_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 特徴量重要度グラフ
            feature_importance = rf_model.feature_importances_
            feature_names = feature_columns
            
            # 重要度でソート
            importance_df = pd.DataFrame({
                '特徴量': feature_names,
                '重要度': feature_importance
            }).sort_values('重要度', ascending=False)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['重要度'], color='skyblue', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['特徴量'])
            plt.xlabel('重要度')
            plt.title(f'{group} - RandomForest特徴量重要度')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}feature_importance_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 予測vs実際の散布図
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('実際の売上')
            plt.ylabel('予測売上')
            plt.title(f'{group} - 予測vs実際（RandomForest）')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_prefix}scatter_{group}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # CSV保存
            future_forecast_df = pd.DataFrame({
                '日付': future_dates,
                '予測売上': future_predictions
            })
            future_forecast_df['予測売上'] = future_forecast_df['予測売上'].round(0).astype(int)
            future_forecast_df.to_csv(f'{save_prefix}forecast_{group}.csv', index=False)
            
            # 特徴量重要度をCSV保存
            importance_df.to_csv(f'{save_prefix}feature_importance_{group}.csv', index=False)
        
        # 全店舗分の性能指標と予測結果をまとめてCSV保存
        pd.DataFrame(results).to_csv(f'{save_prefix}model_performance.csv', index=False)
        print(f"\n{save_prefix}model_performance.csv, {save_prefix}forecast_[shop].csv, {save_prefix}forecast_[shop].png, {save_prefix}feature_importance_[shop].png, {save_prefix}scatter_[shop].png などを保存しました")

def main():
    """
    メイン実行関数
    4つの予測タイプを順番に実行
    """
    # フォアキャスターの初期化
    forecaster = RestaurantSalesForecaster()
    
    # データファイルの読み込み
    try:
        forecaster.load_data('restaurant_sales_data.csv')
    except FileNotFoundError:
        print("データファイルが見つかりません。まずサンプルデータを生成してください。")
        print("python generate_sample_data.py を実行してください。"); return
    
    # 前処理後のカラムから回帰変数を自動抽出
    regressors = [
        'AVG_MONTHLY_POPULATION', 'RATING_CNT', 'RATING_SCORE', 'DINNER_INFO', 'LUNCH_INFO', 'HOME_PAGE_URL', 'PHONE_NUM', 'NUM_SEATS', 'MAX_NUM_PEOPLE_FOR_RESERVATION', 'RESERVATION_POSSIBILITY_INFO', '創業年数'
    ] + [col for col in forecaster.proc_data.columns if col.startswith('CITY_') or col.startswith('CUISINE_CAT_1_')]
    
    # [c1] RandomForestでカテゴリ別 3種
    print("\n=== [c1] RandomForestでカテゴリ別 3種 ===")
    print("各カテゴリの売上データを使ってRandomForestで予測モデルを作成します")
    forecaster.train_and_evaluate_randomforest_per_category(regressors, test_mode=True, max_lags=6)
    
    # [c2] RandomForestで店舗別 3件
    print("\n=== [c2] RandomForestで店舗別 3件 ===")
    print("各店舗の売上データを使ってRandomForestで予測モデルを作成します")
    forecaster.train_and_evaluate_randomforest_per_shop(regressors, test_mode=True, max_lags=6)
    
    # [c3] Prophetでカテゴリ別 3種
    print("\n=== [c3] Prophetでカテゴリ別 3種 ===")
    print("各カテゴリの売上データを使ってProphetで予測モデルを作成します")
    forecaster.train_and_evaluate_per_category(regressors, test_mode=True, analyze_features=False)
    
    # [c4] Prophetで店舗別 3件
    print("\n=== [c4] Prophetで店舗別 3件 ===")
    print("各店舗の売上データを使ってProphetで予測モデルを作成します")
    forecaster.train_and_evaluate_per_shop(regressors)

if __name__ == "__main__":
    main() 