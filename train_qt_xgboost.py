"""
QT 當沖潛力預測模型訓練腳本（使用 XGBoost）
適合小資料集，不需要 TensorFlow

新版本: 從 120天收盤價序列提取15個趨勢斜率特徵
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


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
    y = prices  # 直接使用價格（已標準化）
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
    
    # 計算每段的跨度
    segment_span = (n - 1) / num_segments
    
    slopes = []
    
    for i in range(num_segments):
        start_idx = int(i * segment_span)
        end_idx = int((i + 1) * segment_span)
        
        segment = prices[start_idx:end_idx + 1]
        slope = calculate_trend_slope(segment)
        slopes.append(round(slope, 6))  # 保留更多小數位
    
    return slopes


def extract_trend_features_from_120d(price_sequence_120d):
    """
    從120天收盤價序列提取15個趨勢斜率特徵
    
    方法: 標準化價格後計算趨勢線斜率
    - 將價格除以第一天價格進行標準化
    - 斜率直接反映百分比變化
    - 不同價格的股票可以公平比較
    
    參數:
    price_sequence_120d: 120天收盤價序列 (絕對價格，例如 [100, 102, 101, ...])
    
    返回:
    dict: 15個斜率特徵 (已乘以100，單位為 %/天)
    """
    # 解析序列
    if pd.isna(price_sequence_120d):
        return {f'slope_{i}': 0 for i in range(1, 16)}
    
    if isinstance(price_sequence_120d, str):
        s = price_sequence_120d.strip('[]').strip()
        prices = [float(v.strip()) for v in s.split(',') if v.strip()]
    else:
        prices = list(price_sequence_120d)
    
    if len(prices) < 120:
        print(f"  ⚠ 警告: 序列長度不足 ({len(prices)} < 120)")
        return {f'slope_{i}': 0 for i in range(1, 16)}
    
    prices_array = np.array(prices)
    
    # === 方案 B: 標準化價格 ===
    # 將每段的價格除以該段第一天的價格
    # 這樣斜率就代表「相對於起始價格的每日變化率」
    
    def normalize_and_extract(price_segment):
        """標準化並提取斜率"""
        if len(price_segment) < 2:
            return [0] * 5
        
        # 標準化: 除以第一個價格
        if price_segment[0] != 0:
            normalized = price_segment / price_segment[0]
        else:
            normalized = price_segment
        
        # 提取斜率並乘以100，轉換為 %/天
        slopes = extract_overlapping_slopes(normalized, num_segments=5)
        return [s * 100 for s in slopes]  # 乘以100
    
    # 1. 120天切5段 (使用全部120天)
    slopes_120d = normalize_and_extract(prices_array)
    
    # 2. 20天切5段 (使用最近21天)
    if len(prices) >= 21:
        slopes_20d = normalize_and_extract(prices_array[-21:])
    else:
        slopes_20d = [0] * 5
    
    # 3. 5天切5段 (使用最近6天)
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


class QTModelTrainer:
    """QT 當沖潛力預測模型訓練器（XGBoost 版本）"""
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.feature_columns = None
        self.target_columns = None
        self.feature_importance = None
        
    def load_data(self, filepath='data/QT Training Data.xlsx'):
        """載入 Excel 資料"""
        print("\n" + "╔" + "=" * 58 + "╗")
        print("║" + " " * 10 + "QT 當沖潛力預測模型訓練系統" + " " * 18 + "║")
        print("╚" + "=" * 58 + "╝")
        
        print("\n" + "=" * 60)
        print("步驟 1: 載入資料")
        print("=" * 60)
        
        self.data = pd.read_excel(filepath, sheet_name='工作表1')
        print(f"✓ 載入資料: {len(self.data)} 筆記錄")
        print(f"✓ 欄位數量: {len(self.data.columns)} 個")
        print(f"\n資料形狀: {self.data.shape}")
        
        return self
    
    def preprocess_data(self):
        """資料預處理"""
        print("\n" + "=" * 60)
        print("步驟 2: 資料預處理")
        print("=" * 60)
        
        # 分離特徵和目標變數
        self.target_columns = [col for col in self.data.columns if col.startswith('#')]
        exclude_cols = ['開盤日期', '公司代碼'] + self.target_columns
        
        # 先不包含任何序列欄位和無效欄位
        self.feature_columns = [col for col in self.data.columns 
                               if col not in exclude_cols 
                               and '序列' not in col
                               and not col.startswith('Unnamed')
                               and col != '備註']  # 排除備註欄位
        
        print(f"✓ 原始特徵欄位數: {len(self.feature_columns)}")
        print(f"✓ 目標欄位數: {len(self.target_columns)}")
        print(f"\n目標變數: {', '.join(self.target_columns)}")
        
        # === 新增: 從 120天收盤價序列提取15個斜率特徵 ===
        if '120天收盤價序列' in self.data.columns:
            print(f"\n✓ 發現 120天收盤價序列，開始提取趨勢斜率特徵...")
            
            # 顯示提取結果
            print("\n" + "-" * 60)
            print("趨勢斜率特徵提取結果:")
            print("-" * 60)
            
            slope_features_list = []
            for idx, row in self.data.iterrows():
                print(f"\n記錄 {idx + 1} ({row['公司代碼']}):")
                
                # 提取15個斜率特徵
                slope_features = extract_trend_features_from_120d(row['120天收盤價序列'])
                slope_features_list.append(slope_features)
                
                # 顯示提取的特徵
                print("  120天 5段斜率:")
                for i in range(1, 6):
                    print(f"    段{i}: {slope_features[f'120d_seg{i}_slope']:>8.4f} %/天")
                
                print("  20天 5段斜率:")
                for i in range(1, 6):
                    print(f"    段{i}: {slope_features[f'20d_seg{i}_slope']:>8.4f} %/天")
                
                print("  5天 5段斜率:")
                for i in range(1, 6):
                    print(f"    段{i}: {slope_features[f'5d_seg{i}_slope']:>8.4f} %/天")
            
            # 將斜率特徵加入資料
            slope_df = pd.DataFrame(slope_features_list)
            self.data = pd.concat([self.data, slope_df], axis=1)
            
            # 更新特徵欄位列表
            self.feature_columns.extend(slope_df.columns.tolist())
            
            print("\n" + "-" * 60)
            print(f"✓ 成功提取 {len(slope_df.columns)} 個趨勢斜率特徵")
            print("-" * 60)
        else:
            print("\n⚠ 警告: 找不到 '120天收盤價序列' 欄位")
        
        # 處理產業欄位（One-Hot Encoding）
        if '產業' in self.data.columns:
            industry_dummies = pd.get_dummies(self.data['產業'], prefix='產業')
            self.data = pd.concat([self.data, industry_dummies], axis=1)
            self.feature_columns.remove('產業')
            self.feature_columns.extend(industry_dummies.columns.tolist())
            print(f"\n✓ 產業欄位已轉換為 One-Hot Encoding ({len(industry_dummies.columns)} 個類別)")
        
        # 處理缺失值
        missing = self.data[self.feature_columns].isnull().sum().sum()
        if missing > 0:
            print(f"✓ 發現 {missing} 個缺失值，使用中位數填補")
            for col in self.feature_columns:
                if self.data[col].isnull().any():
                    median_val = self.data[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    self.data[col].fillna(median_val, inplace=True)
        else:
            print("✓ 無缺失值")
        
        print(f"\n最終特徵數量: {len(self.feature_columns)}")
        print(f"特徵列表: {self.feature_columns}")
        
        return self
    
    def split_data(self, target_column, test_size=0.2, random_state=42):
        """分割訓練集和測試集"""
        print("\n" + "=" * 60)
        print(f"步驟 3: 分割訓練集和測試集 - {target_column}")
        print("=" * 60)
        
        X = self.data[self.feature_columns].values
        y = self.data[target_column].values
        
        # 分割資料
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✓ 訓練集樣本數: {len(self.X_train)} 筆 ({(1-test_size)*100:.0f}%)")
        print(f"✓ 測試集樣本數: {len(self.X_test)} 筆 ({test_size*100:.0f}%)")
        print(f"✓ 特徵維度: {self.X_train.shape[1]}")
        
        # 標準化
        print(f"\n✓ 標準化特徵")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def build_and_train_model(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        """
        建立並訓練 XGBoost 模型
        
        參數說明:
        - n_estimators: 樹的數量（類似訓練週期）
        - max_depth: 樹的最大深度（控制模型複雜度）
        - learning_rate: 學習率（每棵樹的貢獻權重）
        """
        print("\n" + "=" * 60)
        print("步驟 4: 建立並訓練 XGBoost 模型")
        print("=" * 60)
        
        print(f"✓ 模型參數:")
        print(f"  - 樹的數量 (n_estimators): {n_estimators}")
        print(f"  - 最大深度 (max_depth): {max_depth}")
        print(f"  - 學習率 (learning_rate): {learning_rate}")
        
        # 建立模型
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        
        print(f"\n開始訓練...")
        print("-" * 60)
        
        # 訓練模型（帶驗證集）
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
            verbose=False
        )
        
        print("-" * 60)
        print("✓ 訓練完成！")
        
        # 取得特徵重要性
        self.feature_importance = pd.DataFrame({
            '特徵': self.feature_columns,
            '重要性': self.model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        return self
    
    def evaluate_model(self):
        """評估模型"""
        print("\n" + "=" * 60)
        print("步驟 5: 評估模型")
        print("=" * 60)
        
        # 訓練集預測
        y_train_pred = self.model.predict(self.X_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # 測試集預測
        y_test_pred = self.model.predict(self.X_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print(f"✓ 訓練集評估:")
        print(f"  - RMSE: {train_rmse:.4f}")
        print(f"  - MAE: {train_mae:.4f}")
        print(f"  - R² Score: {train_r2:.4f}")
        
        print(f"\n✓ 測試集評估:")
        print(f"  - RMSE: {test_rmse:.4f}")
        print(f"  - MAE: {test_mae:.4f}")
        print(f"  - R² Score: {test_r2:.4f}")
        
        # 預測範例
        print(f"\n✓ 預測範例:")
        comparison = pd.DataFrame({
            '實際值': self.y_test,
            '預測值': y_test_pred,
            '誤差': np.abs(self.y_test - y_test_pred)
        })
        print(comparison.to_string(index=False))
        
        return self
    
    def plot_results(self, save_path='models/training_results.png'):
        """視覺化結果"""
        print("\n" + "=" * 60)
        print("步驟 6: 視覺化結果")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 特徵重要性（前10名）
        top_features = self.feature_importance.head(10)
        axes[0, 0].barh(range(len(top_features)), top_features['重要性'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['特徵'])
        axes[0, 0].set_xlabel('重要性分數')
        axes[0, 0].set_title('特徵重要性排名（前10名）', fontweight='bold')
        axes[0, 0].invert_yaxis()
        
        # 2. 預測 vs 實際（測試集）
        y_pred = self.model.predict(self.X_test)
        axes[0, 1].scatter(self.y_test, y_pred, alpha=0.6)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 
                       'r--', lw=2)
        axes[0, 1].set_xlabel('實際值')
        axes[0, 1].set_ylabel('預測值')
        axes[0, 1].set_title('預測值 vs 實際值', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 誤差分布
        errors = np.abs(self.y_test - y_pred)
        axes[1, 0].hist(errors, bins=10, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('絕對誤差')
        axes[1, 0].set_ylabel('頻率')
        axes[1, 0].set_title('預測誤差分布', fontweight='bold')
        axes[1, 0].axvline(errors.mean(), color='r', linestyle='--', 
                          label=f'平均誤差: {errors.mean():.2f}')
        axes[1, 0].legend()
        
        # 4. 訓練進度（使用 evals_result）
        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        axes[1, 1].plot(range(epochs), results['validation_0']['rmse'], 
                       label='訓練集 RMSE', linewidth=2)
        axes[1, 1].plot(range(epochs), results['validation_1']['rmse'], 
                       label='測試集 RMSE', linewidth=2)
        axes[1, 1].set_xlabel('迭代次數')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('訓練過程（損失曲線）', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 儲存圖片
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 結果圖表已儲存至: {save_path}")
        
        plt.show()
        
        return self
    
    def save_model(self, target_column, model_path='models/qt_xgboost_model.pkl'):
        """儲存模型"""
        print("\n" + "=" * 60)
        print(f"步驟 7: 儲存模型 - {target_column}")
        print("=" * 60)
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 儲存模型和相關資訊
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': target_column,
            'feature_importance': self.feature_importance
        }, model_path)
        
        print(f"✓ 模型已儲存至: {model_path}")
        
        return self


def train_single_target(trainer, target_column, model_index):
    """訓練單一目標的模型"""
    print("\n" + "╔" + "=" * 58 + "╗")
    print(f"║  模型 {model_index}/4: {target_column}" + " " * (58 - len(f"  模型 {model_index}/4: {target_column}")) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 生成模型檔案名稱
    safe_name = target_column.replace('#', '').replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    model_path = f'models/qt_model_{safe_name}.pkl'
    plot_path = f'models/training_results_{safe_name}.png'
    
    # 執行訓練流程
    trainer.split_data(target_column=target_column, test_size=0.2, random_state=42)\
           .build_and_train_model(n_estimators=100, max_depth=5, learning_rate=0.1)\
           .evaluate_model()\
           .plot_results(save_path=plot_path)\
           .save_model(target_column=target_column, model_path=model_path)
    
    return model_path


def main():
    """主程式 - 訓練所有目標的模型"""
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "QT 當沖潛力預測 - 多目標訓練系統" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 初始化訓練器
    trainer = QTModelTrainer()
    
    # 載入資料並預處理（只需要做一次）
    trainer.load_data('data/QT Training Data.xlsx')\
           .preprocess_data()
    
    # 訓練每個目標
    trained_models = []
    for i, target_col in enumerate(trainer.target_columns, 1):
        model_path = train_single_target(trainer, target_col, i)
        trained_models.append((target_col, model_path))
    
    # 總結
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 20 + "訓練完成總結" + " " * 22 + "║")
    print("╚" + "=" * 58 + "╝")
    print(f"\n✓ 成功訓練 {len(trained_models)} 個模型:")
    for target_col, model_path in trained_models:
        print(f"  • {target_col:30s} → {model_path}")
    
    print("\n下一步:")
    print("1. 查看結果圖表: models/training_results_*.png")
    print("2. 使用模型預測: python predict_new_stocks.py")
    print("3. 收集更多資料以提升準確度")
    print()


if __name__ == "__main__":
    main()
