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


def predict_stocks(input_file, model_path='models/qt_xgboost_model.pkl', 
                   output_file='data/new_predictions.csv', use_advanced=False):
    """
    預測新股票的當沖潛力
    
    參數:
    - input_file: 輸入檔案路徑（Excel 或 CSV）
    - model_path: 模型檔案路徑
    - output_file: 輸出檔案路徑
    - use_advanced: 是否使用進階模型
    """
    print("=" * 60)
    print("QT 當沖潛力預測系統")
    print("=" * 60)
    
    # 1. 載入模型
    print(f"\n步驟 1: 載入模型")
    print(f"模型路徑: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ 錯誤: 找不到模型檔案 {model_path}")
        print(f"\n請先訓練模型:")
        if use_advanced:
            print(f"  python train_qt_advanced.py")
        else:
            print(f"  python train_qt_xgboost.py")
        return None
    
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    target_column = model_data['target_column']
    
    print(f"✓ 模型載入成功")
    print(f"✓ 預測目標: {target_column}")
    print(f"✓ 需要的特徵數: {len(feature_columns)}")
    
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
    
    # 顯示欄位
    print(f"\n資料欄位:")
    for i, col in enumerate(data.columns, 1):
        print(f"  {i}. {col}")
    
    # 3. 預處理資料
    print(f"\n步驟 3: 預處理資料")
    processed_data = preprocess_new_data(data, feature_columns, use_advanced)
    
    # 4. 標準化
    print(f"\n步驟 4: 標準化特徵")
    X = processed_data.values
    X_scaled = scaler.transform(X)
    print(f"✓ 標準化完成")
    
    # 5. 預測
    print(f"\n步驟 5: 進行預測")
    predictions = model.predict(X_scaled)
    print(f"✓ 預測完成")
    
    # 6. 整理結果
    print(f"\n步驟 6: 整理結果")
    
    # 將預測結果加入原始資料
    result = data.copy()
    result[f'預測_{target_column}'] = predictions
    
    # 如果原始資料有目標欄位，計算誤差
    if target_column in result.columns:
        result['預測誤差'] = np.abs(result[target_column] - predictions)
        print(f"✓ 原始資料包含目標欄位，已計算預測誤差")
    
    # 按預測值排序
    result = result.sort_values(f'預測_{target_column}', ascending=False)
    
    # 7. 顯示結果
    print(f"\n" + "=" * 60)
    print("預測結果（按預測值排序）")
    print("=" * 60)
    
    # 選擇要顯示的欄位
    display_cols = ['開盤日期', '公司代碼']
    if '產業' in result.columns:
        display_cols.append('產業')
    display_cols.append(f'預測_{target_column}')
    if target_column in result.columns:
        display_cols.extend([target_column, '預測誤差'])
    
    # 確保欄位存在
    display_cols = [col for col in display_cols if col in result.columns]
    
    print("\n" + result[display_cols].to_string(index=False))
    
    # 8. 儲存結果
    print(f"\n步驟 7: 儲存結果")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ 結果已儲存至: {output_file}")
    
    # 9. 統計摘要
    print(f"\n" + "=" * 60)
    print("預測統計摘要")
    print("=" * 60)
    print(f"預測筆數: {len(predictions)}")
    print(f"預測平均值: {predictions.mean():.2f}%")
    print(f"預測標準差: {predictions.std():.2f}%")
    print(f"預測最大值: {predictions.max():.2f}%")
    print(f"預測最小值: {predictions.min():.2f}%")
    
    if target_column in result.columns:
        errors = result['預測誤差']
        print(f"\n預測誤差統計:")
        print(f"平均絕對誤差: {errors.mean():.2f}%")
        print(f"誤差標準差: {errors.std():.2f}%")
        print(f"最大誤差: {errors.max():.2f}%")
        print(f"最小誤差: {errors.min():.2f}%")
    
    print(f"\n" + "=" * 60)
    print("✓ 預測完成！")
    print("=" * 60)
    
    return result


def main():
    """主程式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用訓練好的模型預測股票當沖潛力',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:

1. 預測新資料（使用基礎模型）:
   python predict_new_stocks.py --input data/new_stocks.xlsx

2. 預測新資料（使用進階模型）:
   python predict_new_stocks.py --input data/new_stocks.xlsx --advanced

3. 指定輸出檔案:
   python predict_new_stocks.py --input data/new_stocks.xlsx --output results/predictions.csv

4. 使用自訓練的模型:
   python predict_new_stocks.py --input data/new_stocks.xlsx --model models/my_model.pkl

輸入資料格式:
- 必須包含所有訓練時使用的特徵欄位
- 不需要包含目標標籤（帶 # 的欄位）
- 支援 Excel (.xlsx) 和 CSV (.csv) 格式
        """
    )
    
    parser.add_argument('--input', '-i', type=str, 
                       default='data/Stock TBP.xlsx',
                       help='輸入資料檔案路徑')
    parser.add_argument('--model', '-m', type=str,
                       default='models/qt_xgboost_model.pkl',
                       help='模型檔案路徑')
    parser.add_argument('--output', '-o', type=str,
                       default='data/new_predictions.csv',
                       help='輸出結果檔案路徑')
    parser.add_argument('--advanced', '-a', action='store_true',
                       help='使用進階模型（需要先訓練 train_qt_advanced.py）')
    
    args = parser.parse_args()
    
    # 如果使用進階模型，自動調整模型路徑
    if args.advanced and args.model == 'models/qt_xgboost_model.pkl':
        args.model = 'models/qt_advanced_model.pkl'
    
    # 執行預測
    result = predict_stocks(
        input_file=args.input,
        model_path=args.model,
        output_file=args.output,
        use_advanced=args.advanced
    )
    
    if result is not None:
        print(f"\n提示:")
        print(f"- 查看完整結果: {args.output}")
        print(f"- 預測值越高，當沖潛力越大")
        print(f"- 建議關注預測值前 10 名的股票")


if __name__ == "__main__":
    main()
