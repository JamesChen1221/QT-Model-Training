"""
使用訓練好的模型預測新股票的當沖潛力
只需要輸入特徵資料，不需要目標標籤
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb


def calculate_trend_slope(prices):
    """
    計算趨勢線斜率（線性回歸）
    
    參數:
    prices: 價格序列（已標準化）
    
    返回:
    slope: 趨勢線斜率
    """
    if len(prices) < 2:
        return 0
    x = np.arange(len(prices))
    y = prices
    slope, _ = np.polyfit(x, y, 1)
    return slope


def extract_overlapping_slopes(prices, num_segments=5):
    """
    用重疊窗口提取趨勢斜率
    
    參數:
    prices: 價格序列（已標準化）
    num_segments: 要切成幾段
    
    返回:
    list: 每段的趨勢斜率
    """
    prices = np.array(prices)
    n = len(prices)
    
    if n < num_segments + 1:
        return [0] * num_segments
    
    segment_span = (n - 1) / num_segments
    slopes = []
    
    for i in range(num_segments):
        start_idx = int(i * segment_span)
        end_idx = int((i + 1) * segment_span)
        segment = prices[start_idx:end_idx + 1]
        slope = calculate_trend_slope(segment)
        slopes.append(round(slope, 6))
    
    return slopes


def extract_trend_features_from_120d(price_sequence_120d):
    """
    從120天收盤價序列提取15個趨勢斜率特徵
    
    參數:
    price_sequence_120d: 120天收盤價序列
    
    返回:
    dict: 15個斜率特徵 (單位為 %/天)
    """
    if pd.isna(price_sequence_120d):
        return {f'slope_{i}': 0 for i in range(1, 16)}
    
    if isinstance(price_sequence_120d, str):
        s = price_sequence_120d.strip('[]').strip()
        prices = [float(v.strip()) for v in s.split(',') if v.strip()]
    else:
        prices = list(price_sequence_120d)
    
    if len(prices) < 120:
        return {f'slope_{i}': 0 for i in range(1, 16)}
    
    prices_array = np.array(prices)
    
    def normalize_and_extract(price_segment):
        """標準化並提取斜率"""
        if len(price_segment) < 2:
            return [0] * 5
        
        if price_segment[0] != 0:
            normalized = price_segment / price_segment[0]
        else:
            normalized = price_segment
        
        slopes = extract_overlapping_slopes(normalized, num_segments=5)
        return [s * 100 for s in slopes]  # 乘以100
    
    # 1. 120天切5段
    slopes_120d = normalize_and_extract(prices_array)
    
    # 2. 20天切5段
    if len(prices) >= 21:
        slopes_20d = normalize_and_extract(prices_array[-21:])
    else:
        slopes_20d = [0] * 5
    
    # 3. 5天切5段
    if len(prices) >= 6:
        slopes_5d = normalize_and_extract(prices_array[-6:])
    else:
        slopes_5d = [0] * 5
    
    return {
        '120d_seg1_slope': slopes_120d[0],
        '120d_seg2_slope': slopes_120d[1],
        '120d_seg3_slope': slopes_120d[2],
        '120d_seg4_slope': slopes_120d[3],
        '120d_seg5_slope': slopes_120d[4],
        '20d_seg1_slope': slopes_20d[0],
        '20d_seg2_slope': slopes_20d[1],
        '20d_seg3_slope': slopes_20d[2],
        '20d_seg4_slope': slopes_20d[3],
        '20d_seg5_slope': slopes_20d[4],
        '5d_seg1_slope': slopes_5d[0],
        '5d_seg2_slope': slopes_5d[1],
        '5d_seg3_slope': slopes_5d[2],
        '5d_seg4_slope': slopes_5d[3],
        '5d_seg5_slope': slopes_5d[4]
    }


def preprocess_new_data(data, feature_columns):
    """
    預處理新資料（與訓練時相同的處理）
    
    參數:
    - data: 新資料 DataFrame
    - feature_columns: 訓練時使用的特徵欄位
    
    注意: 此函數的邏輯必須與 train_qt_xgboost.py 中的 preprocess_data 完全一致
    """
    df = data.copy()
    
    print("\n資料預處理...")
    
    # 0. 移除無效欄位（空白或數字欄位名）
    invalid_cols = [col for col in df.columns if isinstance(col, (int, float)) or str(col).strip() == '']
    if invalid_cols:
        print(f"✓ 移除無效欄位: {invalid_cols}")
        df = df.drop(columns=invalid_cols)
    
    # === 從 120天收盤價序列提取15個斜率特徵 ===
    if '120天收盤價序列' in df.columns:
        print(f"✓ 從 120天收盤價序列提取趨勢斜率特徵...")
        
        slope_features_list = []
        for idx, row in df.iterrows():
            slope_features = extract_trend_features_from_120d(row['120天收盤價序列'])
            slope_features_list.append(slope_features)
        
        slope_df = pd.DataFrame(slope_features_list)
        df = pd.concat([df, slope_df], axis=1)
        print(f"✓ 成功提取 {len(slope_df.columns)} 個趨勢斜率特徵")
    else:
        print(f"⚠ 警告: 找不到 '120天收盤價序列' 欄位")
        print(f"  趨勢斜率特徵將全部設為 0")
        print(f"  請在 Excel 中加入此欄位以獲得更準確的預測")
        
        # 如果缺少序列，需要手動創建斜率特徵（全部為0）
        slope_feature_names = [
            '120d_seg1_slope', '120d_seg2_slope', '120d_seg3_slope', '120d_seg4_slope', '120d_seg5_slope',
            '20d_seg1_slope', '20d_seg2_slope', '20d_seg3_slope', '20d_seg4_slope', '20d_seg5_slope',
            '5d_seg1_slope', '5d_seg2_slope', '5d_seg3_slope', '5d_seg4_slope', '5d_seg5_slope'
        ]
        for feat in slope_feature_names:
            df[feat] = 0
    
    # 1. 處理產業欄位（One-Hot Encoding）
    if '產業' in df.columns:
        industry_dummies = pd.get_dummies(df['產業'], prefix='產業')
        df = pd.concat([df, industry_dummies], axis=1)
        print(f"✓ 產業欄位已轉換為 One-Hot Encoding ({len(industry_dummies.columns)} 個類別)")
    
    # 2. 確保所有訓練時的特徵都存在
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        print(f"⚠ 缺少特徵: {missing_features}")
        print(f"  將用 0 填補")
        for feat in missing_features:
            df[feat] = 0
    
    # 3. 填補缺失值
    # 特殊處理: 觸發類型 2 (消息面) 的財報欄位應該填 0，不是中位數
    financial_cols = ['EPS Surprise (%)', 'Revenue Surprise (%)', '展望 (Guidance)']
    
    if '觸發類型' in df.columns:
        for col in financial_cols:
            if col in feature_columns and col in df.columns:
                # 觸發類型 2 的財報欄位填 0
                mask_type2 = df['觸發類型'] == 2
                if mask_type2.any():
                    df.loc[mask_type2, col] = df.loc[mask_type2, col].fillna(0)
                    print(f"✓ 觸發類型 2 的 {col} 填 0")
                
                # 其他類型用中位數填補
                mask_other = df['觸發類型'] != 2
                if mask_other.any():
                    median_val = df.loc[mask_other, col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    df.loc[mask_other, col] = df.loc[mask_other, col].fillna(median_val)
    
    # 處理其他欄位的缺失值（用中位數）
    for col in feature_columns:
        if col not in financial_cols and col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col].fillna(median_val, inplace=True)
    
    # 4. 只保留訓練時使用的特徵，並按照相同順序
    # 注意: feature_columns 已經排除了 * 和 # 開頭的欄位
    df = df[feature_columns]
    
    print(f"✓ 預處理完成，特徵數: {len(feature_columns)}")
    
    return df


def calculate_confidence_score(prediction, interval_width, similarity_score, model_std):
    """
    計算綜合可信度分數（0-1）
    
    參數:
    - prediction: 預測值
    - interval_width: 預測區間寬度
    - similarity_score: 與訓練資料的相似度（0-1）
    - model_std: 模型預測的標準差
    
    返回:
    - confidence: 可信度分數（0-1）
    """
    # 1. 預測區間分數（區間越窄越好）
    # 假設區間寬度 < 10% 為高可信度，> 20% 為低可信度
    interval_score = max(0, 1 - interval_width / 20)
    
    # 2. 相似度分數（已經是 0-1）
    # similarity_score 已經標準化
    
    # 3. 模型一致性分數（標準差越小越好）
    # 假設標準差 < 3% 為高可信度，> 6% 為低可信度
    consistency_score = max(0, 1 - model_std / 6)
    
    # 綜合分數（加權平均）
    confidence = (
        0.4 * interval_score +      # 40% 權重：預測區間
        0.3 * similarity_score +     # 30% 權重：資料相似度
        0.3 * consistency_score      # 30% 權重：模型一致性
    )
    
    return confidence


def predict_with_confidence(model, X_new, X_train, y_train, scaler, n_bootstrap=30):
    """
    預測並計算可信度
    
    參數:
    - model: 訓練好的模型
    - X_new: 新資料（已標準化）
    - X_train: 訓練資料特徵（未標準化）
    - y_train: 訓練資料目標
    - scaler: 標準化器
    - n_bootstrap: Bootstrap 迭代次數（預設 30）
    
    返回:
    - dict: 包含預測值、可信度、預測區間等資訊
    """
    # 1. 基本預測
    prediction = model.predict(X_new)[0]
    
    # 2. Bootstrap 估計不確定性
    predictions = []
    
    for i in range(n_bootstrap):
        # 重採樣訓練資料
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # 標準化
        X_boot_scaled = scaler.transform(X_boot)
        
        # 訓練模型
        boot_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=i,
            verbosity=0
        )
        boot_model.fit(X_boot_scaled, y_boot)
        
        # 預測
        pred = boot_model.predict(X_new)[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 3. 計算統計量
    mean_pred = predictions.mean()
    std_pred = predictions.std()
    lower_95 = np.percentile(predictions, 2.5)
    upper_95 = np.percentile(predictions, 97.5)
    interval_width = upper_95 - lower_95
    
    # 4. 計算相似度（與訓練資料的相似程度）
    # 使用標準化後的資料計算距離
    X_train_scaled = scaler.transform(X_train)
    nn = NearestNeighbors(n_neighbors=min(5, len(X_train)))
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_new)
    avg_distance = distances.mean()
    
    # 標準化距離（0-1）
    max_distance = np.linalg.norm(X_train_scaled.max(axis=0) - X_train_scaled.min(axis=0))
    if max_distance > 0:
        similarity_score = max(0, 1 - (avg_distance / max_distance))
    else:
        similarity_score = 1.0
    
    # 5. 計算可信度分數（0-1）
    confidence = calculate_confidence_score(
        prediction=mean_pred,
        interval_width=interval_width,
        similarity_score=similarity_score,
        model_std=std_pred
    )
    
    # 6. 可信度等級
    if confidence > 0.7:
        confidence_level = "高"
    elif confidence > 0.4:
        confidence_level = "中"
    else:
        confidence_level = "低"
    
    return {
        'prediction': prediction,
        'mean_prediction': mean_pred,
        'confidence_score': confidence,
        'confidence_level': confidence_level,
        'interval_95_lower': lower_95,
        'interval_95_upper': upper_95,
        'interval_width': interval_width,
        'std': std_pred,
        'similarity': similarity_score
    }


def predict_stocks(input_file, model_dir='models/', 
                   output_file='data/new_predictions.csv',
                   calculate_confidence=False,
                   training_data_file='data/QT Training Data.xlsx'):
    """
    使用所有訓練好的模型預測新股票的當沖潛力
    
    參數:
    - input_file: 輸入檔案路徑（Excel 或 CSV）
    - model_dir: 模型資料夾路徑
    - output_file: 輸出檔案路徑
    - calculate_confidence: 是否計算可信度（預設 False，因為較慢）
    - training_data_file: 訓練資料檔案路徑（計算可信度時需要）
    """
    print("=" * 60)
    print("QT 當沖潛力預測系統 - 多目標預測")
    print("=" * 60)
    
    # 載入訓練資料（如果需要計算可信度）
    train_data = None
    if calculate_confidence:
        print(f"\n⚠ 可信度計算模式已啟用（需要較長時間）")
        print(f"載入訓練資料: {training_data_file}")
        
        if not Path(training_data_file).exists():
            print(f"❌ 錯誤: 找不到訓練資料檔案 {training_data_file}")
            print(f"  可信度計算需要訓練資料，將關閉可信度計算")
            calculate_confidence = False
        else:
            train_data = pd.read_excel(training_data_file, sheet_name='工作表1')
            print(f"✓ 訓練資料載入完成: {len(train_data)} 筆")
    
    # 1. 尋找所有模型檔案
    print(f"\n步驟 1: 載入模型")
    model_files = list(Path(model_dir).glob('qt_model_*.pkl'))
    
    if not model_files:
        print(f"❌ 錯誤: 在 {model_dir} 中找不到模型檔案")
        print(f"\n請先訓練模型:")
        print(f"  python train_qt_xgboost.py")
        return None
    
    print(f"✓ 找到 {len(model_files)} 個模型:")
    
    models_data = []
    for model_file in sorted(model_files):
        model_data = joblib.load(model_file)
        target_col = model_data['target_column']
        models_data.append({
            'path': model_file,
            'data': model_data,
            'target': target_col
        })
        print(f"  • {target_col:30s} ← {model_file.name}")
    
    # 2. 載入新資料
    print(f"\n步驟 2: 載入新資料")
    print(f"輸入檔案: {input_file}")
    
    if not Path(input_file).exists():
        print(f"❌ 錯誤: 找不到輸入檔案 {input_file}")
        return None
    
    # 根據檔案類型載入
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        data = pd.read_excel(input_file, sheet_name='工作表1')
    elif input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
    else:
        print(f"❌ 錯誤: 不支援的檔案格式，請使用 .xlsx 或 .csv")
        return None
    
    print(f"✓ 載入資料: {len(data)} 筆記錄")
    print(f"✓ 欄位數: {len(data.columns)}")
    
    # 3. 對每個模型進行預測
    result = data.copy()
    
    for i, model_info in enumerate(models_data, 1):
        print(f"\n步驟 {i+2}: 預測 {model_info['target']}")
        print("-" * 60)
        
        model_data = model_info['data']
        model = model_data['model']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        target_column = model_info['target']  # 從 model_info 取得，不是從 model_data
        
        # 預處理資料
        processed_data = preprocess_new_data(data, feature_columns)
        
        # 標準化
        X = processed_data.values
        X_scaled = scaler.transform(X)
        
        # 預測
        predictions = model.predict(X_scaled)
        
        # 將預測結果加入
        result[f'預測_{target_column}'] = predictions
        
        print(f"✓ 預測完成")
        
        # 計算可信度（如果啟用）
        if calculate_confidence and train_data is not None:
            print(f"  計算可信度...")
            
            # 預處理訓練資料
            train_processed = preprocess_new_data(train_data, feature_columns)
            X_train = train_processed.values
            
            # 確保訓練資料有目標欄位
            if target_column in train_data.columns:
                y_train = train_data[target_column].values
                
                # 為每筆新資料計算可信度
                confidence_scores = []
                confidence_levels = []
                interval_lowers = []
                interval_uppers = []
                
                for idx in range(len(X_scaled)):
                    X_new_single = X_scaled[idx:idx+1]
                    
                    conf_result = predict_with_confidence(
                        model=model,
                        X_new=X_new_single,
                        X_train=X_train,
                        y_train=y_train,
                        scaler=scaler,
                        n_bootstrap=30
                    )
                    
                    confidence_scores.append(conf_result['confidence_score'])
                    confidence_levels.append(conf_result['confidence_level'])
                    interval_lowers.append(conf_result['interval_95_lower'])
                    interval_uppers.append(conf_result['interval_95_upper'])
                
                # 加入可信度欄位
                result[f'可信度_{target_column}'] = confidence_scores
                result[f'可信度等級_{target_column}'] = confidence_levels
                result[f'預測下界_{target_column}'] = interval_lowers
                result[f'預測上界_{target_column}'] = interval_uppers
                
                print(f"✓ 可信度計算完成")
            else:
                print(f"⚠ 警告: 訓練資料中找不到目標欄位 {target_column}，跳過可信度計算")
        
        # 如果原始資料有目標欄位，計算誤差
        if target_column in result.columns:
            result[f'誤差_{target_column}'] = np.abs(result[target_column] - predictions)
    
    # 4. 整理並顯示結果
    print(f"\n" + "=" * 80)
    print("預測結果總覽")
    print("=" * 80)
    
    # 定義目標順序：開盤、10分鐘低價、最高價前最低價、1.5小時高價
    target_order = [
        '#開盤 (%)',
        '#10分鐘低價 (%)',
        '#最高價前的最低價 (%)',
        '#1.5小時高價 (%)'
    ]
    
    # 選擇要顯示的欄位：日期、公司、4個預測目標（按順序）
    display_cols = ['開盤日期', '公司代碼']
    
    # 按照指定順序加入預測欄位
    for target_col in target_order:
        pred_col = f'預測_{target_col}'
        if pred_col in result.columns:
            display_cols.append(pred_col)
    
    # 格式化顯示（靠左對齊）
    display_df = result[display_cols].copy()
    
    # 格式化數值欄位（保留2位小數）
    for col in display_df.columns:
        if col.startswith('預測_'):
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    print("\n" + display_df.to_string(index=False, justify='left'))
    
    # 5. 儲存結果
    print(f"\n步驟 {len(models_data)+3}: 儲存結果")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 定義目標順序：開盤、10分鐘低價、最高價前最低價、1.5小時高價
    target_order = [
        '#開盤 (%)',
        '#10分鐘低價 (%)',
        '#最高價前的最低價 (%)',
        '#1.5小時高價 (%)'
    ]
    
    # 準備輸出欄位：只保留必要欄位
    output_cols = ['開盤日期', '公司代碼']
    
    # 按照指定順序加入預測欄位和可信度
    for target_col in target_order:
        pred_col = f'預測_{target_col}'
        if pred_col in result.columns:
            output_cols.append(pred_col)
            
            # 如果有可信度，只加入可信度分數（不加入等級、上下界）
            if calculate_confidence and f'可信度_{target_col}' in result.columns:
                output_cols.append(f'可信度_{target_col}')
    
    # 只輸出選定的欄位
    result[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 結果已儲存至: {output_file}")
    print(f"✓ 輸出欄位: {len(output_cols)} 個")
    
    # 6. 統計摘要
    print(f"\n" + "=" * 60)
    print("預測統計摘要")
    print("=" * 60)
    print(f"預測筆數: {len(result)}")
    
    # 按照指定順序顯示統計
    target_order = [
        '#開盤 (%)',
        '#10分鐘低價 (%)',
        '#最高價前的最低價 (%)',
        '#1.5小時高價 (%)'
    ]
    
    for target_col in target_order:
        pred_col = f'預測_{target_col}'
        if pred_col not in result.columns:
            continue
            
        predictions = result[pred_col]
        
        print(f"\n{target_col}:")
        print(f"  平均值: {predictions.mean():>8.2f}%")
        print(f"  標準差: {predictions.std():>8.2f}%")
        print(f"  最大值: {predictions.max():>8.2f}%")
        print(f"  最小值: {predictions.min():>8.2f}%")
        
        # 顯示可信度統計
        if calculate_confidence and f'可信度_{target_col}' in result.columns:
            conf_scores = result[f'可信度_{target_col}']
            conf_levels = result[f'可信度等級_{target_col}']
            
            print(f"  平均可信度: {conf_scores.mean():>6.2f}")
            print(f"  可信度分布:")
            print(f"    高: {(conf_levels == '高').sum()} 筆")
            print(f"    中: {(conf_levels == '中').sum()} 筆")
            print(f"    低: {(conf_levels == '低').sum()} 筆")
        
        if target_col in result.columns:
            errors = result[f'誤差_{target_col}']
            print(f"  平均誤差: {errors.mean():>6.2f}%")
    
    print(f"\n" + "=" * 60)
    print("✓ 預測完成！")
    print("=" * 60)
    
    return result


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用訓練好的模型預測股票當沖潛力（多目標預測）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 預測新資料（自動使用所有模型）:
   python predict_new_stocks.py --input data/Stock TBP.xlsx

2. 指定輸出檔案:
   python predict_new_stocks.py --input data/Stock TBP.xlsx --output results/predictions.csv

3. 指定模型資料夾:
   python predict_new_stocks.py --input data/Stock TBP.xlsx --model-dir models/

4. 計算預測可信度（需要較長時間）:
   python predict_new_stocks.py --input data/Stock TBP.xlsx --confidence

5. 指定訓練資料位置（計算可信度時）:
   python predict_new_stocks.py --input data/Stock TBP.xlsx --confidence --training-data data/QT Training Data.xlsx

輸入資料格式:
- 必須包含所有訓練時使用的特徵欄位
- 不需要包含目標標籤（帶 # 的欄位）
- 支援 Excel (.xlsx) 和 CSV (.csv) 格式

輸出結果:
- 會包含所有 4 個目標的預測值
- #開盤 (%)
- #10分鐘低價 (%)
- #1.5小時高價 (%)
- #最高價前的最低價 (%)
- 如果啟用 --confidence，會額外包含：
  * 可信度分數（0-1）
  * 可信度等級（高/中/低）
  * 95% 預測區間（上界、下界）

可信度說明:
- 高可信度（> 0.7）: 可以信賴預測結果
- 中等可信度（0.4-0.7）: 謹慎評估
- 低可信度（< 0.4）: 建議觀望
- 可信度計算使用 Bootstrap 方法（30次迭代），需要較長時間
        """
    )
    
    parser.add_argument('--input', '-i', type=str, 
                       default='data/Stock TBP.xlsx',
                       help='輸入資料檔案路徑')
    parser.add_argument('--model-dir', '-m', type=str,
                       default='models/',
                       help='模型資料夾路徑')
    parser.add_argument('--output', '-o', type=str,
                       default='data/new_predictions.csv',
                       help='輸出結果檔案路徑')
    parser.add_argument('--confidence', '-c', action='store_true',
                       help='計算預測可信度（需要較長時間，使用 Bootstrap 30次）')
    parser.add_argument('--training-data', '-t', type=str,
                       default='data/QT Training Data.xlsx',
                       help='訓練資料檔案路徑（計算可信度時需要）')
    
    args = parser.parse_args()
    
    # 執行預測
    result = predict_stocks(
        input_file=args.input,
        model_dir=args.model_dir,
        output_file=args.output,
        calculate_confidence=args.confidence,
        training_data_file=args.training_data
    )
    
    if result is not None:
        print(f"\n提示:")
        print(f"- 查看完整結果: {args.output}")
        print(f"- 預測值越高，當沖潛力越大")
        print(f"- 建議關注各項預測值較高的股票")
        
        if args.confidence:
            print(f"\n可信度使用建議:")
            print(f"- 高可信度（> 0.7）: 可以採取行動")
            print(f"- 中等可信度（0.4-0.7）: 謹慎評估，結合其他分析")
            print(f"- 低可信度（< 0.4）: 建議觀望")
            print(f"- 注意: 資料量少時（< 100筆），所有預測的可信度都會偏低")


if __name__ == "__main__":
    main()
