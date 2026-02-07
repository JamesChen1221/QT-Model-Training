@echo off
chcp 65001 > nul
echo ========================================
echo    QT 當沖潛力預測 - 訓練所有模型
echo ========================================
echo.
echo 開始訓練 4 個模型...
echo.

python train_qt_xgboost.py

echo.
echo ========================================
echo 訓練完成！
echo ========================================
echo.
echo 已生成 4 個模型檔案:
echo   • models\qt_model_開盤_pct.pkl
echo   • models\qt_model_10分鐘低價_pct.pkl
echo   • models\qt_model_1.5小時高價_pct.pkl
echo   • models\qt_model_最高價前的最低價_pct.pkl
echo.
echo 按任意鍵關閉...
pause > nul
