# RFM Customer Insights

這個專案用直播電商銷售資料做客戶分群、客群標籤與優惠策略建議。  

## 這次改善了什麼

1. `Monetary` 改成使用 `Quantity * Price`
   原本只加總 `Price`，會嚴重低估大量購買客戶的價值。

2. 將退貨 / 調整單與正常交易分開
   資料中有負數數量與負數金額。現在會在清理階段先標記，RFM 只用正向消費行為計算，但評估時仍保留退貨風險指標。

3. `Frequency` 改成「購買日數」
   由於資料沒有明確訂單編號，直接計算資料列數會把同一天的多個商品行拆成很多次購買。現在改用 `CustomerID + Date(日期)` 作為訂單代理。

4. 補上 `R/F/M Score` 與 `Segment`
   除了數值特徵外，現在也會輸出像 `Champions`、`Loyal Customers`、`Hibernating` 這種可以直接解讀的客群標籤。

5. 分群流程加入 fallback
   若環境有 `scikit-learn`，會用 KMeans 做聚類；若沒有，會自動退回到 RFM 規則式分群，避免程式無法執行。

6. 評估改成真實描述性指標
   拿掉隨機模擬的優惠券使用率，改成輸出客群規模、營收占比、平均客單價、回購率、近 90 天活躍率與退貨風險。

## 專案結構

- `scr/preprocess.py`
  清理欄位型別、標記退貨與計算 `LineAmount`
- `scr/compute_rfm.py`
  計算 RFM、補充衍生欄位、產生 RFM 分數與客群標籤
- `scr/clustering.py`
  執行 KMeans 或 fallback 分群，並輸出分群摘要與圖表
- `scr/coupon_recommendation.py`
  根據 `Segment` 產生優惠策略建議
- `scr/evaluation.py`
  產出客群層級的營運評估摘要與推薦商品結果

## 主要輸出檔案

- `data/cleaned_data.csv`
- `data/customer_rfm.csv`
- `data/customer_rfm_clusters.csv`
- `data/customer_rfm_recommendations.csv`
- `data/cluster_summary.csv`
- `data/evaluation_summary.csv`
- `data/segment_top_products.csv`
- `images/cluster_distribution.png`
- `images/rfm_distribution.png`
- `images/performance_metrics.png`

## 建議執行順序

```bash
python scr/preprocess.py
python scr/compute_rfm.py
python scr/clustering.py
python scr/coupon_recommendation.py
python scr/evaluation.py
```

## 目前分析上最值得再往前走的方向

1. 加入時間切分驗證
   例如以前 9 個月建立分群，後 3 個月驗證回購或營收差異，讓分析更接近預測場景。

2. 把商品偏好納入客群洞察
   目前已輸出各客群 Top Products，下一步可以做商品類型偏好或關聯推薦。

3. 做真正的優惠成效評估
   如果未來有優惠券發送與使用紀錄，可以改做 A/B test、uplift analysis 或 propensity model。

4. 把國家 / 地區與退貨風險加入分群特徵
   可以幫助營運更細緻地設計投放策略。
