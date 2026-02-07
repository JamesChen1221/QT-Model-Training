"""
QT 當沖潛力預測模型訓練腳本（使用 XGBoost）
適合小資料集，不需要 TensorFlow
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
        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"✓ 特徵欄位數: {len(self.feature_columns)}")
        print(f"✓ 目標欄位數: {len(self.target_columns)}")
        print(f"\n目標變數: {self.target_columns[0]}")
        
        # 處理產業欄位（One-Hot Encoding）
        if '產業' in self.data.columns:
            industry_dummies = pd.get_dummies(self.data['產業'], prefix='產業')
            self.data = pd.concat([self.data, industry_dummies], axis=1)
            self.feature_columns.remove('產業')
            self.feature_columns.extend(industry_dummies.columns.tolist())
            print(f"✓ 產業欄位已轉換為 One-Hot Encoding ({len(industry_dummies.columns)} 個類別)")
        
        # 處理序列資料（取最後一個值）
        sequence_cols = [col for col in self.feature_columns if '序列' in col]
        if sequence_cols:
            print(f"✓ 處理序列資料: {len(sequence_cols)} 個欄位")
            for col in sequence_cols:
                def parse_sequence(x):
                    if pd.isna(x):
                        return np.nan
                    # 處理列表格式 '[41.9, 39.7, 25.0]' 或逗號分隔 '41.9, 39.7, 25.0'
                    s = str(x).strip('[]').strip()
                    values = [float(v.strip()) for v in s.split(',') if v.strip()]
                    return values[-1] if values else np.nan
                
                self.data[col] = self.data[col].apply(parse_sequence)
                # 確保轉換為數值型
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # 處理缺失值
        missing = self.data[self.feature_columns].isnull().sum().sum()
        if missing > 0:
            print(f"✓ 發現 {missing} 個缺失值，使用中位數填補")
            for col in self.feature_columns:
                if self.data[col].isnull().any():
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        else:
            print("✓ 無缺失值")
        
        print(f"\n最終特徵數量: {len(self.feature_columns)}")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """分割訓練集和測試集"""
        print("\n" + "=" * 60)
        print("步驟 3: 分割訓練集和測試集")
        print("=" * 60)
        
        X = self.data[self.feature_columns].values
        y = self.data[self.target_columns[0]].values
        
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
    
    def save_model(self, model_path='models/qt_xgboost_model.pkl'):
        """儲存模型"""
        print("\n" + "=" * 60)
        print("步驟 7: 儲存模型")
        print("=" * 60)
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 儲存模型和相關資訊
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_columns[0],
            'feature_importance': self.feature_importance
        }, model_path)
        
        print(f"✓ 模型已儲存至: {model_path}")
        
        return self


def main():
    """主程式"""
    trainer = QTModelTrainer()
    
    # 執行完整訓練流程
    trainer.load_data('data/QT Training Data.xlsx')\
           .preprocess_data()\
           .split_data(test_size=0.2, random_state=42)\
           .build_and_train_model(n_estimators=100, max_depth=5, learning_rate=0.1)\
           .evaluate_model()\
           .plot_results()\
           .save_model()
    
    print("\n" + "=" * 60)
    print("✓ 所有步驟完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 查看結果圖表: models/training_results.png")
    print("2. 使用模型預測: python predict_qt_xgboost.py")
    print("3. 收集更多資料以提升準確度")
    print()


if __name__ == "__main__":
    main()
