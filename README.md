# 飲食店売上予測プロジェクト（Prophet & RandomForest）

## 概要
本プロジェクトは、飲食店の月次売上データをもとに、ProphetおよびRandomForestを用いて将来12ヶ月分の売上を予測するPythonスクリプト集です。カテゴリ別・店舗別の両方で予測が可能で、特徴量の自動スケーリングやハイパーパラメータチューニングも実装されています。

## 特徴
- **Prophet/RandomForestによる売上予測**（カテゴリ別・店舗別）
- **特徴量スケーリング・ハイパーパラメータ自動探索**（RandomForest）
- **ラグ特徴量自動生成**（RandomForest）
- **予測結果・特徴量重要度・散布図の自動保存**
- **柔軟なカスタマイズ**

## 必要環境・依存パッケージ
- Python 3.8 以上推奨
- 必要パッケージは `requirements.txt` で管理しています。

インストール例：
```bash
pip install -r requirements.txt
```

## データ構造と前提
- 入力データ: `restaurant_sales_data.csv`
    - 店舗ID（sakaya_dealer_cd）、カテゴリ（CUISINE_CAT_1）、年月（YEAR, MONTH）、売上（target_amount）などのカラムを含む
    - サンプルデータ生成スクリプトも同梱（`generate_sample_data.py`）

## 使い方
1. 必要なパッケージをインストール
2. データファイル（`restaurant_sales_data.csv`）を用意
    - サンプルデータを使う場合：
      ```bash
      python generate_sample_data.py
      ```
3. メインスクリプトを実行
    ```bash
    python prophet_sales_forecast.py
    ```

## 予測モード（4パターン）
スクリプト実行時、以下4つの予測が自動で順番に実行されます：

1. **[c1] RandomForestでカテゴリ別（3種）**
2. **[c2] RandomForestで店舗別（3件）**
3. **[c3] Prophetでカテゴリ別（3種）**
4. **[c4] Prophetで店舗別（3件）**

※テストモード時は各3カテゴリ・3店舗のみ。全件実行したい場合は`prophet_sales_forecast.py`内の`test_mode`引数を調整してください。

## 出力ファイル
- `rf_cat_model_performance.csv`：カテゴリ別RandomForestの性能指標
- `rf_cat_forecast_*.csv`：カテゴリ別RandomForestの未来12ヶ月予測
- `rf_cat_forecast_*.png`：カテゴリ別RandomForestの予測グラフ
- `rf_cat_feature_importance_*.png`：カテゴリ別RandomForestの特徴量重要度
- `rf_cat_scatter_*.png`：カテゴリ別RandomForestの予測vs実測散布図
- `rf_shop_model_performance.csv`：店舗別RandomForestの性能指標
- `rf_shop_forecast_*.csv`：店舗別RandomForestの未来12ヶ月予測
- `rf_shop_forecast_*.png`：店舗別RandomForestの予測グラフ
- `rf_shop_feature_importance_*.png`：店舗別RandomForestの特徴量重要度
- `rf_shop_scatter_*.png`：店舗別RandomForestの予測vs実測散布図
- `cat_model_performance.csv`：カテゴリ別Prophetの性能指標
- `cat_forecast_*.csv`：カテゴリ別Prophetの未来12ヶ月予測
- `cat_forecast_*.png`：カテゴリ別Prophetの予測グラフ
- `cat_components_*.png`：カテゴリ別Prophetの分解図
- `cat_trend_*.png`：カテゴリ別Prophetのトレンド
- `cat_seasonality_*.png`：カテゴリ別Prophetの季節性
- `shop_model_performance.csv`：店舗別Prophetの性能指標
- `shop_forecast_*.csv`：店舗別Prophetの未来12ヶ月予測

## カスタマイズ方法
- 予測対象カテゴリや店舗数を増やしたい場合は、`prophet_sales_forecast.py`の`test_mode`引数を`False`に変更してください。
- 特徴量の追加や前処理のカスタマイズも`prophet_sales_forecast.py`内で柔軟に編集可能です。
- 予測期間やラグ数も引数で調整できます。

## トラブルシューティング
- **matplotlibのエラーが出る場合**：バックエンドは`Agg`に設定済みですが、エラーが出る場合は`matplotlib.use('Agg')`の位置を確認してください。
- **データが足りない場合**：十分な月数のデータがないとモデルが学習できません。
- **Prophetのインストールエラー**：`cmdstanpy`や`pystan`の依存関係に注意してください。

## ライセンス
- 本プロジェクトはMITライセンスです。

---

ご質問・ご要望はIssueまたはPull Requestでお気軽にどうぞ！
