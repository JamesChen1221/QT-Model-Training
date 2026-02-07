"""
使用訓練好的模型預測新股票的當沖潛力
只需要輸入特徵資料，不需要目標標籤
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path


def parse_sequence(x):
    """解析序列字串"""
    if pd.isna(x):
        return []
    s = str(x).strip('[]').strip()
    return [float(v.strip()) for v in s.split(',') if v.strip()]


def preprocess_new_data(data, feature_columns, use_advanced=False):
    """
    預處理新資料（與訓練時相同的處理）
    
    參數:
    - data: 新資料 DataFrame
    - feature_columns: 訓練時使用的特徵欄位
    - use_advanced: 是否使用進階特徵提取
    """
    df = data.copy()
    
    print("\n資料預處理...")
    
    # 0. 移除無效欄位（數字欄位名稱或空欄位）
    invalid_cols = [col for col in df.columns if isinstance(col, (int, float)) or str(col).strip() == '']
    if invalid_cols:
        print(f"✓ 移除無效欄位: {invalid_cols}")
        df = df.drop(columns=invalid_cols)
    
    # 1. 處理產業欄位
    if '產業' in df.columns:
        industry_dummies = pd.get_dummies(df['產業'], prefix='產業')
        df = pd.concat([df, industry_dummies], axis=1)
        print(f"✓ 產業欄位已轉換")
    
    # 2. 處理序列資料
    sequence_cols = [col for col in df.columns if '序列' in str(col)]
    
    if use_advanced:
        # 進階版本：提取多個特徵
        print(f"✓ 使用進階特徵提取（每個序列 6 個特徵）")
        for col in sequence_cols:
            sequences = df[col].apply(parse_sequence)
            
            df[f"{col}_last"] = sequences.apply(lambda x: x[-1] if x else np.nan)
            df[f"{col}_mean"] = sequences.apply(lambda x: np.mean(x) if x else np.nan)
            df[f"{col}_std"] = sequences.apply(lambda x: np.std(x) if x else np.nan)
            df[f"{col}_trend"] = sequences.apply(lambda x: x[-1] - x[0] if len(x) > 1 else np.nan)
            df[f"{col}_max"] = sequences.apply(lambda x: np.max(x) if x else np.nan)
            df[f"{col}_min"] = sequences.apply(lambda x: np.min(x) if x else np.nan)
            
            df = df.drop(columns=[col])
    else:
        # 基礎版本：只用最後一個值
        print(f"✓ 使用基礎特徵提取（只用最新值）")
        for col in sequence_cols:
            df[col] = df[col].apply(lambda x: parse_sequence(x)[-1] if parse_sequence(x) else np.nan)
    
    # 3. 確保所有訓練時的特徵都存在
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        print(f"⚠ 缺少特徵: {missing_features}")
        print(f"  將用 0 填補")
        for feat in missing_features:
            df[feat] = 0
    
    # 4. 填補缺失值
    for col in feature_columns:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median() if df[col].median() == df[col].median() else 0, inplace=True)
    
    # 5. 只保留訓練時使用的特徵，並按照相同順序
    df = df[feature_columns]
    
    print(f"✓ 預處理完成，特徵數: {len(feature_columns)}")
    
    return df


def predict_stocks(input_file, model_dir='models/', 
                   output_file='data/new_predictions.csv', use_advanced=False):
    """
    使用所有訓練好的模型預測新股票的當沖潛力
    
    參數:
    - input_file: 輸入檔案路徑（Excel 或 CSV）
    - model_dir: 模型資料夾路徑
    - output_file: 輸出檔案路徑
    - use_advanced: 是否使用進階模型
    """
    print("=" * 60)
    print("QT 當沖潛力預測系統 - 多目標預測")
    print("=" * 60)
    
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
        processed_data = preprocess_new_data(data, feature_columns, use_advanced)
        
        # 標準化
        X = processed_data.values
        X_scaled = scaler.transform(X)
        
        # 預測
        predictions = model.predict(X_scaled)
        
        # 將預測結果加入
        result[f'預測_{target_column}'] = predictions
        
        print(f"✓ 預測完成")
        
        # 如果原始資料有目標欄位，計算誤差
        if target_column in result.columns:
            result[f'誤差_{target_column}'] = np.abs(result[target_column] - predictions)
    
    # 4. 整理並顯示結果
    print(f"\n" + "=" * 60)
    print("預測結果總覽")
    print("=" * 60)
    
    # 選擇要顯示的欄位
    display_cols = ['開盤日期', '公司代碼']
    if '產業' in result.columns:
        display_cols.append('產業')
    
    # 加入所有預測欄位
    for model_info in models_data:
        target_col = model_info['target']
        display_cols.append(f'預測_{target_col}')
    
    # 確保欄位存在
    display_cols = [col for col in display_cols if col in result.columns]
    
    print("\n" + result[display_cols].to_string(index=False))
    
    # 5. 儲存結果
    print(f"\n步驟 {len(models_data)+3}: 儲存結果")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 結果已儲存至: {output_file}")
    
    # 6. 統計摘要
    print(f"\n" + "=" * 60)
    print("預測統計摘要")
    print("=" * 60)
    print(f"預測筆數: {len(result)}")
    
    for model_info in models_data:
        target_col = model_info['target']
        pred_col = f'預測_{target_col}'
        predictions = result[pred_col]
        
        print(f"\n{target_col}:")
        print(f"  平均值: {predictions.mean():>8.2f}%")
        print(f"  標準差: {predictions.std():>8.2f}%")
        print(f"  最大值: {predictions.max():>8.2f}%")
        print(f"  最小值: {predictions.min():>8.2f}%")
        
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
    parser.add_argument('--advanced', '-a', action='store_true',
                       help='使用進階模型（需要先訓練 train_qt_advanced.py）')
    
    args = parser.parse_args()
    
    # 執行預測
    result = predict_stocks(
        input_file=args.input,
        model_dir=args.model_dir,
        output_file=args.output,
        use_advanced=args.advanced
    )
    
    if result is not None:
        print(f"\n提示:")
        print(f"- 查看完整結果: {args.output}")
        print(f"- 預測值越高，當沖潛力越大")
        print(f"- 建議關注各項預測值較高的股票")


if __name__ == "__main__":
    main()
