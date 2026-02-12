@echo off
chcp 65001 >nul
echo ========================================
echo QT 當沖潛力預測（含可信度計算）
echo ========================================
echo.
echo 注意：可信度計算需要較長時間（約 2-5 分鐘）
echo.

python predict_new_stocks.py --input "data/Stock TBP.xlsx" --confidence

echo.
echo ========================================
echo 預測完成！
echo ========================================
echo.
echo 結果已儲存至: data/new_predictions.csv
echo.
pause
