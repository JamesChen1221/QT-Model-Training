# QT 當沖潛力預測系統

使用機器學習預測股票的當沖潛力。

---

## 🚀 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 訓練模型
```bash
python train_qt_xgboost.py
```

### 3. 預測新股票
```bash
# 方法 1: 使用批次檔（最快）
雙擊 快速預測.bat

# 方法 2: 使用命令列
python predict_new_stocks.py --input "single stock.xlsx"
```

---

## 📁 專案結構

```
QT Model Training/
├── data/
│   └── QT Training Data.xlsx       # 訓練資料
│
├── models/
│   ├── qt_xgboost_model.pkl        # 訓練好的模型
│   └── training_results.png        # 訓練結果圖
│
├── train_qt_xgboost.py             # 訓練腳本
├── predict_new_stocks.py           # 預測腳本
├── 快速預測.bat                     # 快速預測工具
├── single stock.xlsx               # 預測範例
│
├── requirements.txt                # Python 依賴
├── config.py                       # 配置參數
│
├── README.md                       # 本文件
├── ✅ 預測成功！下次這樣做.md       # 快速操作指南
└── 訓練概念說明.md                  # 機器學習概念說明
```

---

## 📊 資料格式

### 訓練資料（QT Training Data.xlsx）

**必要欄位（22 個）：**
- 開盤日期、公司代碼、產業
- 價格距離指標（5日、1個月、6個月）
- 昨日收盤價、盤前漲幅
- 觸發類型、消息情緒分數
- EPS/Revenue Surprise、展望
- RSI/ADX 序列（5天、1個月、6個月）

**目標欄位（4 個，帶 `#`）：**
- `#開盤 (%)` - 開盤漲幅
- `#10分鐘低價 (%)` - 10分鐘內最低價
- `#1.5小時高價 (%)` - 1.5小時內最高價
- `#最高價前的最低價 (%)` - 最高價前的最低價

### 預測資料

只需要特徵欄位，**不需要**目標欄位。

**序列格式：**
```
[60.9, 63.3, 54.6, 47.2, 48.3]
```

---

## 🎯 使用方式

### 訓練模型

```bash
python train_qt_xgboost.py
```

**輸出：**
- `models/qt_xgboost_model.pkl` - 訓練好的模型
- `models/training_results.png` - 訓練結果圖表

### 預測新股票

**方法 1: 批次檔（推薦）**
```
1. 在 Excel 更新 single stock.xlsx
2. 雙擊 快速預測.bat
3. 自動開啟結果檔案
```

**方法 2: 命令列**
```bash
python predict_new_stocks.py --input "single stock.xlsx"
```

**輸出：**
- 控制台顯示預測結果
- `data/new_predictions.csv` - 詳細結果

### 批量預測

在 Excel 中新增多行資料，執行相同命令即可一次預測多支股票。

---

## 📈 解讀結果

### 預測值意義

```
預測_#開盤 (%) = 9.33

意思：
→ 模型預測開盤會上漲 9.33%
→ 相對於昨日收盤價
→ 數值越高，當沖潛力越大
```

### 建議

| 預測值 | 意義 | 建議 |
|--------|------|------|
| +10% 以上 | 非常有潛力 | 重點關注 |
| +5% ~ +10% | 有潛力 | 可以考慮 |
| +2% ~ +5% | 一般 | 謹慎評估 |
| 0% ~ +2% | 潛力較小 | 觀望 |
| 負數 | 預期下跌 | 避免做多 |

---

## 🎓 訓練概念

系統使用 XGBoost 機器學習模型，主要概念：

- **訓練集/測試集**：80/20 分割
- **特徵數量**：25 個（包含技術指標）
- **學習率**：0.1
- **樹的數量**：100 棵
- **評估指標**：RMSE、MAE、R² Score

詳細說明請參考 `訓練概念說明.md`

---

## ⚠️ 重要提醒

### 模型限制

```
✓ 目前只有 6 筆訓練資料
✓ 預測準確度有限
✓ 僅供參考，不是投資建議
✓ 需要結合其他分析方法
```

### 如何提升準確度

1. **收集更多資料**（目標：100+ 筆）
2. **重新訓練模型**：`python train_qt_xgboost.py`
3. **持續驗證**：記錄預測結果，與實際比較

---

## 🛠️ 常見問題

### Q: 找不到模型檔案？
**A:** 先執行 `python train_qt_xgboost.py` 訓練模型

### Q: 序列格式錯誤？
**A:** 確保格式為 `[值1, 值2, 值3]`，用逗號分隔

### Q: 預測結果不準？
**A:** 正常，因為訓練資料太少。建議收集 100+ 筆資料後重新訓練

### Q: 如何預測多支股票？
**A:** 在 Excel 中新增多行，執行相同命令即可

---

## 📝 快速參考

### 最常用命令

```bash
# 訓練
python train_qt_xgboost.py

# 預測（批次檔）
雙擊 快速預測.bat

# 預測（命令列）
python predict_new_stocks.py --input "single stock.xlsx"

# 查看結果
start data\new_predictions.csv
```

### 檔案位置

```
輸入: single stock.xlsx
輸出: data\new_predictions.csv
模型: models\qt_xgboost_model.pkl
```

---

## 📚 相關文件

- **✅ 預測成功！下次這樣做.md** - 快速操作指南
- **訓練概念說明.md** - 機器學習概念詳解

---

## 🎯 授權

MIT License

---

**祝你交易順利！** 🚀
