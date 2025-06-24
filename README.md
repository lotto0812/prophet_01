# レストランカテゴリ別売上予測モデル（Prophet）

このプロジェクトは、Facebook Prophetを使用してレストランのカテゴリ別売上予測モデルを構築するものです。

## 概要

- **目的**: レストランのカテゴリ別月間売上を予測
- **手法**: Facebook Prophet（時系列予測）
- **特徴**: 季節性、トレンド、外部要因を考慮した予測

## データ構造

### 計測データ
- `YEAR`: 年
- `MONTH`: 月
- `AVG_MONTHLY_POPULATION`: 月ごとの人流の平均
- `RATING_SCORE`: 食べログのレーティング
- `RATING_CNT`: 食べログのレーティング数
- `NUM_SEATS`: 席数
- `NEAREST_STATION_INFO`: 最寄り駅

### 教師データ
- `target_amount`: 月ごとの売上

## ファイル構成

```
timeSeriesAnalysis_prophet_simple/
├── requirements.txt              # 必要なライブラリ
├── generate_sample_data.py       # サンプルデータ生成
├── data_analysis.py             # データ分析・可視化
├── prophet_sales_forecast.py    # メイン予測モデル
├── run_all.py                   # 統合実行スクリプト
├── README.md                    # このファイル
└── restaurant_sales_data.csv    # 生成されるデータファイル
```

## セットアップ

### 1. 環境構築

```bash
# 必要なライブラリをインストール
pip install -r requirements.txt
```

### 2. サンプルデータの生成

```bash
python generate_sample_data.py
```

これにより、`restaurant_sales_data.csv`が生成されます。

## 使用方法

### 1. 全処理の一括実行

```bash
python run_all.py
```

### 2. 個別実行

#### データ分析

```bash
python data_analysis.py
```

以下の分析結果が生成されます：
- 売上トレンド分析
- 特徴量分析
- 相関分析
- 季節性分析

#### 予測モデルの実行

```bash
python prophet_sales_forecast.py
```

以下の結果が生成されます：
- 各カテゴリのProphetモデル
- 予測結果の可視化
- モデル性能指標
- 未来12ヶ月の予測

## 出力ファイル

### 分析結果
- `sales_trends_analysis.png`: 売上トレンド分析
- `feature_analysis.png`: 特徴量分析
- `correlation_matrix.png`: 相関分析
- `seasonal_analysis.png`: 季節性分析

### 予測結果
- `forecast_[カテゴリ].png`: 各カテゴリの予測グラフ
- `components_[カテゴリ].png`: 予測成分分解
- `forecast_[カテゴリ].csv`: 各カテゴリの予測データ
- `model_performance.csv`: モデル性能指標

## カテゴリ

現在のモデルでは以下の5つのカテゴリを対象としています：
- イタリアン
- 中華
- 和食
- フレンチ
- カフェ

## モデルの特徴

### Prophet設定
- **年次季節性**: 有効
- **週次季節性**: 無効
- **日次季節性**: 無効
- **季節性モード**: 乗法的
- **回帰変数**: 人流、評価スコア、評価数、席数

### 性能指標
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (決定係数)
- MAPE (Mean Absolute Percentage Error)

## カスタマイズ

### 新しいカテゴリの追加
`generate_sample_data.py`の`categories`リストに新しいカテゴリを追加し、`base_params`にパラメータを設定してください。

### 回帰変数の変更
`prophet_sales_forecast.py`の`regressors`リストを編集して、使用する特徴量を変更できます。

### 予測期間の変更
`train_models`メソッド内の`periods`パラメータを変更して、予測期間を調整できます。

## 注意事項

- Prophetのインストールには時間がかかる場合があります
- 大量のデータを扱う場合は、メモリ使用量に注意してください
- 予測精度はデータの質と量に大きく依存します

## トラブルシューティング

### Prophetのインストールエラー
```bash
# Windowsの場合
conda install -c conda-forge prophet

# または
pip install prophet --no-cache-dir
```

### メモリ不足エラー
- データサイズを小さくする
- カテゴリ数を減らす
- 予測期間を短くする

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要望は、GitHubのIssueでお知らせください。 