# QT Model Training

This is a small side project focused on predicting intraday trading potential for individual stocks, using data from the QT data collecting project. Given a set of pre-market features, the system predicts four key price targets within the first 1.5 hours of trading.

---

## Purpose

The goal is to build a lightweight, practical prediction system that can assist in identifying high-potential stocks for intraday trading before market open. The system predicts:

- **Opening change** `#開盤 (%)` — how much the stock gaps up/down at open
- **10-minute low** `#10分鐘低價 (%)` — the lowest point in the first 10 minutes
- **0.5-hour high** `#0.5小時最高價 (%)` — the highest point in the first 30 minutes
- **1.5-hour high** `#1.5小時高價 (%)` — the highest point in the first 90 minutes

---

## Development Approach

### Machine Learning

- **Algorithm**: XGBoost (gradient boosted trees) — chosen for its strong performance on small tabular datasets and compatibility with Python 3.14
- **Architecture**: 4 independent regression models, one per prediction target
- **Training strategy**: 80/20 train/test split for evaluation, then retrain on full dataset for the final saved model
- **Confidence estimation**: Bootstrap resampling (10 iterations) to estimate prediction uncertainty

### Feature Engineering

Features are extracted from the Excel training data and include:

- Price distance indicators (5-day, 1-month, 6-month highs/lows)
- Pre-market change percentage
- Earnings surprise metrics (EPS, Revenue, Guidance)
- Industry encoding (One-Hot)
- 15 trend slope features extracted from a 120-day closing price sequence (5-day, 20-day, 120-day windows, 5 segments each)
- Average daily trading volume (past 20 days) as a liquidity proxy

### AI-Assisted Development

This project was developed with the assistance of **Kiro** (an AI-powered development environment). AI was used throughout the development process, including:

- Designing the feature engineering pipeline
- Debugging data preprocessing issues
- Implementing the Bootstrap confidence scoring system
- Identifying and resolving edge cases (e.g., NaN filtering, Pandas Copy-on-Write warnings, detached HEAD git state)
- Writing documentation and analysis

---

## Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies**: Python 3.8+, XGBoost, pandas, numpy, scikit-learn, matplotlib, openpyxl

---

## Usage

### Step 1 — Prepare Data

The Excel data files are not included in this repository (excluded via `.gitignore`). You need to provide your own:

- `data/QT Training Data.xlsx` — historical training data
- `data/Stock TBP.xlsx` — stocks to predict

See [Data Format](#data-format) below for the required structure.

### Step 2 — Train Models

```bash
# Using batch file (Windows)
快速訓練.bat

# Or via command line
python train_qt_xgboost.py
```

This trains 4 models and saves them to `models/`. Training results and feature importance charts are saved as `models/training_results_*.png`.

### Step 3 — Predict

```bash
# Basic prediction (fast, a few seconds)
快速預測.bat

# Or via command line
python predict_new_stocks.py --input "data/Stock TBP.xlsx"
```

```bash
# Prediction with confidence scores (slower, ~2–5 min depending on data size)
快速預測_含可信度.bat

# Or via command line
python predict_new_stocks.py --input "data/Stock TBP.xlsx" --confidence
```

Results are saved to `data/new_predictions.csv`.

---

## Data Format

### Column Naming Convention

Columns in the Excel file use prefixes to indicate their role:

| Prefix | Role | Example |
|--------|------|---------|
| *(none)* | Feature used for training | `5日高價距離 (%)` |
| `#` | Prediction target (label) | `#開盤 (%)` |
| `*` | Ignored column | `*備註` |

### Training Data (`QT Training Data.xlsx`)

**Required feature columns:**

| Column | Description |
|--------|-------------|
| `開盤日期` | Trading date |
| `公司代碼` | Stock ticker |
| `產業` | Industry code (integer) |
| `5日高價距離 (%)` | Distance from 5-day high |
| `5日低價距離 (%)` | Distance from 5-day low |
| `1個月高價距離 (%)` | Distance from 1-month high |
| `1個月低價距離 (%)` | Distance from 1-month low |
| `6個月高價距離 (%)` | Distance from 6-month high |
| `6個月低價距離 (%)` | Distance from 6-month low |
| `盤前 (%)` | Pre-market change % |
| `EPS Surprise (%)` | Earnings per share surprise |
| `Revenue Surprise (%)` | Revenue surprise |
| `展望 (Guidance)` | Forward guidance (1=raised, 0=neutral, -1=lowered) |
| `過去 20 天平均交易金額` | Average daily trading value, past 20 days |
| `120天收盤價序列` | 120-day closing price sequence, format: `[100.5, 102.3, ...]` |

**Target label columns (prefix `#`):**

| Column | Description |
|--------|-------------|
| `#開盤 (%)` | Opening change % |
| `#10分鐘低價 (%)` | 10-minute low % |
| `#0.5小時最高價 (%)` | 30-minute high % |
| `#1.5小時高價 (%)` | 90-minute high % |

### Prediction Data (`Stock TBP.xlsx`)

Same structure as training data, but **without** the `#` target columns.

---

## Output

### Basic Prediction

```
開盤日期    公司代碼  預測_#開盤 (%)  預測_#10分鐘低價 (%)  預測_#0.5小時最高價 (%)  預測_#1.5小時高價 (%)
2026-03-05  NVDA       4.58            2.35                  8.11                    8.97
2026-03-05  AAPL       1.48           -1.27                  1.41                    0.62
```

### With Confidence Scores

Additional columns are appended:

| Column | Description |
|--------|-------------|
| `可信度_#開盤 (%)` | Confidence score (0–1) |

**Confidence levels:**

| Level | Score | Recommendation |
|-------|-------|----------------|
| High (高) | > 0.7 | Reliable — consider acting |
| Medium (中) | 0.4–0.7 | Use with caution, combine with other analysis |
| Low (低) | < 0.4 | High uncertainty — stand aside |

Confidence is calculated from three components:
- **Prediction interval width** (Bootstrap 95% CI)
- **Data similarity** (nearest-neighbor distance to training set)
- **Model consistency** (standard deviation across Bootstrap iterations)

---

## Project Structure

```
QT Model Training/
├── data/
│   ├── QT Training Data.xlsx       # Training data (not in repo)
│   ├── Stock TBP.xlsx              # Stocks to predict (not in repo)
│   └── new_predictions.csv         # Prediction output (auto-generated)
│
├── models/
│   ├── qt_model_開盤_pct.pkl
│   ├── qt_model_10分鐘低價_pct.pkl
│   ├── qt_model_0.5小時最高價_pct.pkl
│   ├── qt_model_1.5小時高價_pct.pkl
│   └── training_results_*.png
│
├── documents/                      # Analysis and design documents
│
├── train_qt_xgboost.py             # Training script
├── predict_new_stocks.py           # Prediction script
├── 可信度實作範例.py                # Confidence scoring example
│
├── 快速訓練.bat                     # One-click train (Windows)
├── 快速預測.bat                     # One-click predict (Windows)
├── 快速預測_含可信度.bat            # One-click predict with confidence (Windows)
│
├── requirements.txt
└── .gitignore
```

---

## Notes and Limitations

- **Data not included**: Excel files are excluded from the repository. You must supply your own data collected from the QT data collecting project.
- **Model accuracy**: Performance depends heavily on training data size and quality. With ~200–300 samples, expect test MAE of roughly 3–5% and R² around 0.70–0.75.
- **Not financial advice**: Predictions are for research and reference only. Always combine with your own analysis before making trading decisions.
- **Confidence scores with small datasets**: Confidence scores tend to be lower when training data is under 100 samples, as Bootstrap estimates become less stable.
- **Sequence length filtering**: Stocks with fewer than 120 days of closing price history are automatically excluded from training.

---

## Related Documents

The `documents/` folder contains detailed analysis notes (in Traditional Chinese):

| File | Description |
|------|-------------|
| `訓練流程說明.md` | Training pipeline explanation |
| `市場反應速度特徵分析.md` | Analysis of missing features affecting reaction speed |
| `交易金額時間窗口分析.md` | Why 20-day average trading volume was chosen |
| `特徵工程實戰指南.md` | Feature engineering practical guide |
| `特徵數量與資料量關係分析.md` | Feature count vs. data size analysis |
| `訓練概念說明.md` | ML concepts overview |
| `預測可信度分析.md` | Confidence scoring methodology |

---

## License

MIT License
