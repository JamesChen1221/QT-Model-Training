# 上傳到 GitHub 指南

## ✅ 已完成的準備工作

1. ✅ 更新 `.gitignore` 排除 Excel 檔案
2. ✅ 從 Git 追蹤中移除 Excel 檔案
3. ✅ 新增 `data/README.md` 說明資料格式
4. ✅ 更新主 README 提醒使用者準備資料

## 📋 上傳步驟

### 1. 在 GitHub 上創建新倉庫

1. 前往 https://github.com/new
2. 填寫倉庫資訊：
   - **Repository name**: `qt-daytrading-prediction`（或你喜歡的名稱）
   - **Description**: `QT 當沖潛力預測系統 - 使用 XGBoost 預測股票當沖潛力`
   - **Public/Private**: 選擇 Public 或 Private
   - **不要**勾選 "Initialize this repository with a README"（我們已經有了）
3. 點擊 "Create repository"

### 2. 連接本地倉庫到 GitHub

複製 GitHub 給你的倉庫 URL（例如：`https://github.com/你的用戶名/qt-daytrading-prediction.git`）

然後執行：

```bash
# 添加遠端倉庫
git remote add origin https://github.com/你的用戶名/qt-daytrading-prediction.git

# 檢查遠端倉庫
git remote -v
```

### 3. 推送到 GitHub

由於目前處於 detached HEAD 狀態，需要先創建分支：

```bash
# 創建並切換到 main 分支
git checkout -b main

# 推送到 GitHub
git push -u origin main
```

或者如果你想保留原有的提交歷史：

```bash
# 回到原來的分支
git checkout master  # 或 git checkout main

# 合併變更
git merge 9166aee

# 推送到 GitHub
git push -u origin master  # 或 main
```

### 4. 驗證上傳

前往你的 GitHub 倉庫頁面，確認：
- ✅ 所有程式碼檔案都已上傳
- ✅ Excel 檔案（`*.xlsx`）沒有被上傳
- ✅ README.md 正確顯示
- ✅ 資料目錄有 README.md 說明

## 🔒 確認 Excel 檔案已被排除

執行以下命令確認：

```bash
# 檢查 Git 狀態
git status

# 檢查哪些檔案會被追蹤
git ls-files | grep -i xlsx

# 應該沒有任何輸出（表示沒有 xlsx 檔案被追蹤）
```

## 📝 .gitignore 內容

已設定排除以下檔案：

```
# 資料檔案
*.csv
*.xlsx
*.xls
data/new_predictions.csv
data/*.xlsx
data/*.xls
```

## ⚠️ 重要提醒

1. **Excel 檔案不會被上傳**
   - 訓練資料（QT Training Data.xlsx）
   - 預測資料（Stock TBP.xlsx）
   - 這些檔案只存在於你的本地電腦

2. **其他人使用此專案時**
   - 需要自行準備訓練資料
   - 參考 `data/README.md` 了解資料格式

3. **模型檔案也不會被上傳**
   - `models/*.pkl` 已在 .gitignore 中
   - 其他人需要自行訓練模型

## 🎯 後續維護

### 更新程式碼

```bash
# 修改程式碼後
git add .
git commit -m "描述你的變更"
git push
```

### 拉取最新程式碼

```bash
git pull
```

## 📚 相關文件

- `README.md` - 專案主要說明
- `data/README.md` - 資料格式說明
- `.gitignore` - Git 忽略規則

---

**準備好了嗎？** 按照上述步驟操作即可將專案上傳到 GitHub！
