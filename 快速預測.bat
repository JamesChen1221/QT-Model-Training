@echo off
chcp 65001 > nul
echo ========================================
echo    QT 當沖潛力預測系統
echo ========================================
echo.
echo 開始預測 data\Stock TBP.xlsx...
echo.

python predict_new_stocks.py --input "data\Stock TBP.xlsx"

echo.
echo ========================================
echo 預測完成！
echo ========================================
echo.
echo 結果已儲存至: data\new_predictions.csv
echo.
echo 按任意鍵查看結果...
pause > nul

start data\new_predictions.csv
