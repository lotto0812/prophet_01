# レストランカテゴリ別売上予測モデル（Prophet）

このプロジェクトは、Facebook Prophetを使用してレストランのカテゴリ別売上予測モデルを構築するものです。

## 📊 概要

- **目的**: レストランのカテゴリ別月間売上を予測
- **手法**: Facebook Prophet（時系列予測）
- **特徴**: 季節性、トレンド、外部要因を考慮した予測
- **対象カテゴリ**: イタリアン、中華、和食、フレンチ、カフェ

## 🗂️ ファイル構成

```
timeSeriesAnalysis_prophet_simple/
├── requirements.txt              # 必要なライブラリ
├── generate_sample_data.py       # サンプルデータ生成
├── data_analysis.py             # データ分析・可視化
├── prophet_sales_forecast.py    # メイン予測モデル
├── font_setup.py                # 日本語フォント設定
├── run_all.py                   # 統合実行スクリプト
├── README.md                    # このファイル
└── restaurant_sales_data.csv    # 生成されるデータファイル
```

## 📈 データ構造

### 入力特徴量
- `YEAR`: 年
- `MONTH`: 月
- `AVG_MONTHLY_POPULATION`: 月ごとの人流の平均
- `RATING_SCORE`: 食べログのレーティングスコア
- `RATING_CNT`: 食べログのレーティング数
- `NUM_SEATS`: 席数
- `NEAREST_STATION_INFO`: 最寄り駅情報

### 予測対象
- `target_amount`: 月ごとの売上（万円）

## 🚀 クイックスタート

### 1. 環境構築

```bash
# 必要なライブラリをインストール
pip install -r requirements.txt
```

### 2. 全処理の一括実行

```bash
python run_all.py
```

これにより以下が実行されます：
- サンプルデータの生成
- データ分析と可視化
- 各カテゴリのProphetモデル構築
- 予測結果の出力

## 📋 詳細な使用方法

### 個別実行

#### サンプルデータ生成
```bash
python generate_sample_data.py
```

#### データ分析
```bash
python data_analysis.py
```

#### 予測モデル実行
```bash
python prophet_sales_forecast.py
```

## 📊 出力ファイル

### 分析結果
- `sales_trends_analysis.png`: 売上トレンド分析
- `feature_analysis.png`: 特徴量分析
- `correlation_matrix.png`: 相関分析
- `seasonal_analysis.png`: 季節性分析

### 予測結果
- `forecast_[カテゴリ].png`: 各カテゴリの予測グラフ
- `components_[カテゴリ].png`: 予測成分分解（トレンド・季節性）
- `forecast_[カテゴリ].csv`: 各カテゴリの予測データ
- `model_performance.csv`: モデル性能指標

## ⚙️ モデル設定

### Prophet設定
- **年次季節性**: 有効（12ヶ月周期）
- **週次季節性**: 無効
- **日次季節性**: 無効
- **季節性モード**: 乗法的
- **回帰変数**: 人流、評価スコア、評価数、席数

### 性能指標
- **MAE**: 平均絶対誤差
- **RMSE**: 平均二乗誤差の平方根
- **R²**: 決定係数
- **MAPE**: 平均絶対パーセンテージ誤差

## 🎯 カスタマイズ

### 新しいカテゴリの追加
`generate_sample_data.py`の`categories`リストに新しいカテゴリを追加：

```python
categories = ['イタリアン', '中華', '和食', 'フレンチ', 'カフェ', '新カテゴリ']
```

### 回帰変数の変更
`prophet_sales_forecast.py`の`regressors`リストを編集：

```python
regressors = ['AVG_MONTHLY_POPULATION', 'RATING_SCORE', 'RATING_CNT', 'NUM_SEATS']
```

### 予測期間の変更
`train_models`メソッド内の`periods`パラメータを変更：

```python
future = model.make_future_dataframe(periods=12)  # 12ヶ月先まで予測
```

## 🔧 トラブルシューティング

### Prophetのインストールエラー
```bash
# Windowsの場合
conda install -c conda-forge prophet

# または
pip install prophet --no-cache-dir
```

### 日本語文字化け
- `font_setup.py`が自動的に日本語フォントを設定します
- 問題が続く場合は、システムに日本語フォントがインストールされているか確認してください

### メモリ不足エラー
- データサイズを小さくする
- カテゴリ数を減らす
- 予測期間を短くする

## 📝 注意事項

- Prophetのインストールには時間がかかる場合があります
- 大量のデータを扱う場合は、メモリ使用量に注意してください
- 予測精度はデータの質と量に大きく依存します
- 日本語フォントの設定により、グラフの文字化けを防いでいます

## 🤝 貢献

バグ報告や機能要望は、GitHubのIssueでお知らせください。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**開発者**: [Your Name]  
**最終更新**: 2024年12月
