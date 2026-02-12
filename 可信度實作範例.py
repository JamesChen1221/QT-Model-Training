"""
預測可信度實作範例
簡化版：適合快速整合到現有專案
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb


def predict_with_confidence_simple(model, X_new, X_train, y_train, n_bootstrap=30):
    """
    簡化版：使用 Bootstrap 估計預測可信度
    
    參數:
    - model: 訓練好的模型
    - X_new: 新資料（要預測的）
    - X_train: 訓練資料特徵
    - y_train: 訓練資料目標
    - n_bootstrap: Bootstrap 迭代次數（預設 30）
    
    返回:
    - dict: 包含預測值、可信度、預測區間等資訊
    """
    
    # 1. 基本預測
    prediction = model.predict(X_new)[0]
    
    # 2. Bootstrap 估計不確定性
    print(f"正在計算可信度（Bootstrap {n_bootstrap} 次）...")
    predictions = []
    
    for i in range(n_bootstrap):
        # 重採樣訓練資料
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # 訓練模型
        boot_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=i,
            verbosity=0
        )
        boot_model.fit(X_boot, y_boot)
        
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
    nn = NearestNeighbors(n_neighbors=min(5, len(X_train)))
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_new)
    avg_distance = distances.mean()
    
    # 標準化距離（0-1）
    max_distance = np.linalg.norm(X_train.max(axis=0) - X_train.min(axis=0))
    similarity_score = max(0, 1 - (avg_distance / max_distance))
    
    # 5. 計算可信度分數（0-1）
    # 區間寬度分數（區間越窄越好，假設 < 10% 為高可信度）
    interval_score = max(0, 1 - interval_width / 20)
    
    # 模型一致性分數（標準差越小越好，假設 < 3% 為高可信度）
    consistency_score = max(0, 1 - std_pred / 6)
    
    # 綜合可信度（加權平均）
    confidence = (
        0.4 * interval_score +      # 40% 權重：預測區間
        0.3 * similarity_score +     # 30% 權重：資料相似度
        0.3 * consistency_score      # 30% 權重：模型一致性
    )
    
    # 6. 可信度等級
    if confidence > 0.7:
        confidence_level = "高"
        recommendation = "可以採取行動"
    elif confidence > 0.4:
        confidence_level = "中"
        recommendation = "謹慎評估"
    else:
        confidence_level = "低"
        recommendation = "建議觀望"
    
    return {
        'prediction': prediction,
        'mean_prediction': mean_pred,
        'confidence_score': confidence,
        'confidence_level': confidence_level,
        'recommendation': recommendation,
        'interval_95': (lower_95, upper_95),
        'interval_width': interval_width,
        'std': std_pred,
        'similarity': similarity_score,
        'details': {
            'interval_score': interval_score,
            'consistency_score': consistency_score,
            'similarity_score': similarity_score
        }
    }


def print_confidence_result(result, stock_name=""):
    """美化輸出可信度結果"""
    
    print("\n" + "="*60)
    if stock_name:
        print(f"股票: {stock_name}")
    print("="*60)
    
    print(f"\n📊 預測結果:")
    print(f"  預測值: {result['prediction']:>8.2f}%")
    print(f"  平均值: {result['mean_prediction']:>8.2f}%")
    
    print(f"\n🎯 可信度分析:")
    print(f"  可信度分數: {result['confidence_score']:.2f} ({result['confidence_level']})")
    print(f"  建議: {result['recommendation']}")
    
    print(f"\n📈 預測區間 (95%):")
    print(f"  下界: {result['interval_95'][0]:>8.2f}%")
    print(f"  上界: {result['interval_95'][1]:>8.2f}%")
    print(f"  寬度: {result['interval_width']:>8.2f}%")
    
    print(f"\n🔍 詳細指標:")
    print(f"  預測標準差: {result['std']:>8.2f}%")
    print(f"  資料相似度: {result['similarity']:>8.2f}")
    print(f"  區間分數:   {result['details']['interval_score']:>8.2f}")
    print(f"  一致性分數: {result['details']['consistency_score']:>8.2f}")
    
    print("\n" + "="*60)


# ============================================================================
# 使用範例
# ============================================================================

if __name__ == "__main__":
    """
    使用範例：整合到現有的預測流程
    """
    
    import joblib
    
    # 1. 載入模型和資料
    print("載入模型...")
    model_data = joblib.load('models/qt_model_開盤_pct.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # 2. 載入訓練資料（用於計算相似度）
    print("載入訓練資料...")
    train_data = pd.read_excel('data/QT Training Data.xlsx', sheet_name='工作表1')
    
    # 預處理訓練資料（與訓練時相同的處理）
    # ... 這裡省略預處理步驟，實際使用時需要完整處理 ...
    
    # 假設已經處理好
    X_train = train_data[feature_columns].values
    y_train = train_data['#開盤 (%)'].values
    X_train_scaled = scaler.transform(X_train)
    
    # 3. 載入新資料
    print("載入預測資料...")
    new_data = pd.read_excel('data/Stock TBP.xlsx', sheet_name='工作表1')
    
    # 預處理新資料
    # ... 這裡省略預處理步驟 ...
    
    # 假設已經處理好
    X_new = new_data[feature_columns].values
    X_new_scaled = scaler.transform(X_new)
    
    # 4. 預測並計算可信度
    print("\n開始預測...")
    result = predict_with_confidence_simple(
        model=model,
        X_new=X_new_scaled[:1],  # 預測第一筆
        X_train=X_train_scaled,
        y_train=y_train,
        n_bootstrap=30  # 可以調整，越多越準確但越慢
    )
    
    # 5. 顯示結果
    stock_name = new_data.iloc[0]['公司代碼'] if '公司代碼' in new_data.columns else ""
    print_confidence_result(result, stock_name)
    
    # 6. 根據可信度決策
    print("\n💡 決策建議:")
    if result['confidence_score'] > 0.7:
        print("  ✅ 高可信度預測")
        print("  → 可以根據預測值採取行動")
        print(f"  → 預期開盤漲幅: {result['prediction']:.2f}%")
    elif result['confidence_score'] > 0.4:
        print("  ⚠️ 中等可信度預測")
        print("  → 建議結合其他分析方法")
        print(f"  → 預測範圍: [{result['interval_95'][0]:.2f}%, {result['interval_95'][1]:.2f}%]")
    else:
        print("  ❌ 低可信度預測")
        print("  → 建議觀望，不要輕易行動")
        print("  → 可能原因：")
        if result['similarity'] < 0.5:
            print("     • 新資料與訓練資料差異較大")
        if result['interval_width'] > 15:
            print("     • 預測區間過寬，不確定性高")
        if result['std'] > 5:
            print("     • 模型預測不一致")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
