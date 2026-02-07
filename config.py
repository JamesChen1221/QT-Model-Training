"""
配置文件 - 定義模型訓練和預測的參數
"""

# 資料路徑
DATA_DIR = "data"
MODEL_DIR = "models"
RAW_DATA_PATH = f"{DATA_DIR}/raw_stock_data.csv"
PROCESSED_DATA_PATH = f"{DATA_DIR}/processed_data.csv"
FEATURES_PATH = f"{DATA_DIR}/features.csv"

# 特徵工程參數
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2
}

# 當沖潛力計算參數
DAYTRADING_PARAMS = {
    'min_volatility': 0.02,  # 最小波動率 2%
    'min_volume_ratio': 1.5,  # 最小成交量比率
    'profit_threshold': 0.03  # 獲利門檻 3%
}

# 模型訓練參數
MODEL_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5
}

# XGBoost 參數
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# LSTM 參數
LSTM_PARAMS = {
    'sequence_length': 20,
    'lstm_units': 64,
    'dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}
