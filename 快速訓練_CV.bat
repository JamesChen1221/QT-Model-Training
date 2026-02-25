@echo off
chcp 65001 >nul
echo ========================================
echo QT 當沖潛力預測 - Cross-Validation 訓練
echo ========================================
echo.
echo 使用 5-Fold Cross-Validation
echo - 更可靠的評估結果
echo - 充分利用所有資料
echo - 了解模型穩定性
echo.
echo 開始訓練...
echo.

python train_qt_xgboost.py --cv

echo.
echo ========================================
echo 訓練完成！
echo ========================================
echo.
echo 模型已儲存至 models/ 資料夾
echo 訓練結果圖表:
echo   • models\training_results_開盤_pct.png
echo   • models\training_results_10分鐘低價_pct.png
echo   • models\training_results_0.5小時最高價_pct.png
echo   • models\training_results_1.5小時高價_pct.png
echo.
echo 按任意鍵關閉...
pause >nul
