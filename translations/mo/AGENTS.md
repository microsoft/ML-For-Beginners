<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:00:25+00:00",
  "source_file": "AGENTS.md",
  "language_code": "mo"
}
-->
# AGENTS.md

## 專案概述

這是 **Machine Learning for Beginners**，一個全面的12週、26課程的學習計劃，涵蓋使用 Python（主要使用 Scikit-learn）和 R 的經典機器學習概念。本倉庫設計為自學資源，包含實作專案、測驗和作業。每節課通過來自世界各地不同文化和地區的真實數據探索機器學習概念。

主要內容：
- **教育內容**：26節課程，涵蓋機器學習入門、回歸、分類、聚類、自然語言處理（NLP）、時間序列和強化學習
- **測驗應用**：基於 Vue.js 的測驗應用，提供課前和課後評估
- **多語言支持**：通過 GitHub Actions 自動翻譯至40多種語言
- **雙語支持**：課程提供 Python（Jupyter notebooks）和 R（R Markdown 文件）版本
- **專案式學習**：每個主題都包含實作專案和作業

## 倉庫結構

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

每個課程文件夾通常包含：
- `README.md` - 主要課程內容
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - 解答代碼（Python 和 R 版本）
- `assignment.md` - 練習題
- `images/` - 視覺資源

## 設置指令

### Python 課程

大多數課程使用 Jupyter notebooks。安裝所需的依賴項：

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### R 課程

R 課程位於 `solution/R/` 文件夾中，格式為 `.rmd` 或 `.ipynb` 文件：

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### 測驗應用

測驗應用是一個 Vue.js 應用，位於 `quiz-app/` 目錄中：

```bash
cd quiz-app
npm install
```

### 文件站點

本地運行文件站點：

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## 開發工作流程

### 使用課程筆記本

1. 進入課程目錄（例如 `2-Regression/1-Tools/`）
2. 打開 Jupyter notebook：
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. 完成課程內容和練習
4. 如有需要，可查看 `solution/` 文件夾中的解答

### Python 開發

- 課程使用標準的 Python 數據科學庫
- 使用 Jupyter notebooks 進行互動式學習
- 每節課的 `solution/` 文件夾中提供解答代碼

### R 開發

- R 課程以 `.rmd` 格式（R Markdown）提供
- 解答位於 `solution/R/` 子目錄中
- 使用 RStudio 或帶有 R kernel 的 Jupyter 運行 R notebooks

### 測驗應用開發

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## 測試說明

### 測驗應用測試

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**注意**：這主要是一個教育課程倉庫，課程內容沒有自動化測試。驗證方式包括：
- 完成課程練習
- 成功運行 notebook 的所有單元格
- 將輸出與解答中的預期結果進行比對

## 代碼風格指南

### Python 代碼
- 遵循 PEP 8 風格指南
- 使用清晰、描述性的變量名稱
- 為複雜操作添加註解
- Jupyter notebooks 應包含解釋概念的 markdown 單元格

### JavaScript/Vue.js（測驗應用）
- 遵循 Vue.js 風格指南
- ESLint 配置位於 `quiz-app/package.json`
- 運行 `npm run lint` 檢查並自動修復問題

### 文件
- Markdown 文件應清晰且結構良好
- 在圍欄代碼塊中包含代碼示例
- 使用相對鏈接進行內部引用
- 遵循現有的格式約定

## 構建與部署

### 測驗應用部署

測驗應用可部署至 Azure Static Web Apps：

1. **先決條件**：
   - Azure 帳戶
   - GitHub 倉庫（已經 fork）

2. **部署至 Azure**：
   - 創建 Azure Static Web App 資源
   - 連接至 GitHub 倉庫
   - 設置應用位置：`/quiz-app`
   - 設置輸出位置：`dist`
   - Azure 自動創建 GitHub Actions 工作流程

3. **GitHub Actions 工作流程**：
   - 工作流程文件創建於 `.github/workflows/azure-static-web-apps-*.yml`
   - 推送至主分支時自動構建和部署

### 文件 PDF

從文件生成 PDF：

```bash
npm install
npm run convert
```

## 翻譯工作流程

**重要**：翻譯通過 GitHub Actions 使用 Co-op Translator 自動完成。

- 當更改推送至 `main` 分支時，翻譯會自動生成
- **請勿手動翻譯內容** - 系統會處理
- 工作流程定義於 `.github/workflows/co-op-translator.yml`
- 使用 Azure AI/OpenAI 服務進行翻譯
- 支持40多種語言

## 貢獻指南

### 對內容貢獻者

1. **Fork 倉庫**並創建功能分支
2. **修改課程內容**以添加或更新課程
3. **不要修改翻譯文件** - 它們是自動生成的
4. **測試代碼** - 確保所有 notebook 單元格成功運行
5. **驗證鏈接和圖片**是否正常工作
6. **提交 pull request**並提供清晰的描述

### Pull Request 指南

- **標題格式**：`[Section] 簡要描述更改`
  - 示例：`[Regression] 修正第5課中的拼寫錯誤`
  - 示例：`[Quiz-App] 更新依賴項`
- **提交前**：
  - 確保所有 notebook 單元格無錯誤執行
  - 如果修改了 quiz-app，運行 `npm run lint`
  - 驗證 markdown 格式
  - 測試任何新的代碼示例
- **PR 必須包含**：
  - 更改描述
  - 更改原因
  - 如果有 UI 更改，提供截圖
- **行為準則**：遵循 [Microsoft 開源行為準則](CODE_OF_CONDUCT.md)
- **CLA**：您需要簽署貢獻者許可協議

## 課程結構

每節課程遵循一致的模式：

1. **課前測驗** - 測試基礎知識
2. **課程內容** - 書面指導和解釋
3. **代碼演示** - notebook 中的實作示例
4. **知識檢查** - 驗證學習理解
5. **挑戰** - 獨立應用概念
6. **作業** - 延伸練習
7. **課後測驗** - 評估學習成果

## 常用指令參考

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## 附加資源

- **Microsoft Learn 集合**：[ML for Beginners 模組](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **測驗應用**：[線上測驗](https://ff-quizzes.netlify.app/en/ml/)
- **討論板**：[GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **視頻教程**：[YouTube 播放列表](https://aka.ms/ml-beginners-videos)

## 核心技術

- **Python**：機器學習課程的主要語言（Scikit-learn、Pandas、NumPy、Matplotlib）
- **R**：使用 tidyverse、tidymodels、caret 的替代實現
- **Jupyter**：Python 課程的互動式筆記本
- **R Markdown**：R 課程的文檔格式
- **Vue.js 3**：測驗應用框架
- **Flask**：機器學習模型部署的 Web 應用框架
- **Docsify**：文件站點生成器
- **GitHub Actions**：CI/CD 和自動翻譯

## 安全考量

- **代碼中不包含秘密信息**：切勿提交 API 密鑰或憑證
- **依賴項**：保持 npm 和 pip 包更新
- **用戶輸入**：Flask Web 應用示例包含基本輸入驗證
- **敏感數據**：示例數據集是公開且無敏感信息的

## 疑難排解

### Jupyter Notebooks

- **Kernel 問題**：如果單元格掛起，重啟 Kernel：Kernel → Restart
- **導入錯誤**：確保使用 pip 安裝了所有所需的包
- **路徑問題**：從 notebook 所在的目錄運行

### 測驗應用

- **npm install 失敗**：清除 npm 緩存：`npm cache clean --force`
- **端口衝突**：更改端口：`npm run serve -- --port 8081`
- **構建錯誤**：刪除 `node_modules` 並重新安裝：`rm -rf node_modules && npm install`

### R 課程

- **找不到包**：使用以下指令安裝：`install.packages("package-name")`
- **RMarkdown 渲染**：確保已安裝 rmarkdown 包
- **Kernel 問題**：可能需要為 Jupyter 安裝 IRkernel

## 專案特定注意事項

- 這主要是一個 **學習課程**，而非生產代碼
- 重點在於通過實作練習 **理解機器學習概念**
- 代碼示例以 **清晰性優先於優化** 為原則
- 大多數課程是 **自包含** 的，可獨立完成
- **提供解答**，但學習者應先嘗試完成練習
- 倉庫使用 **Docsify** 生成 Web 文件，無需構建步驟
- **Sketchnotes** 提供概念的視覺摘要
- **多語言支持** 使內容全球可訪問

---

**免責聲明**：  
本文件已使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於關鍵資訊，建議使用專業人工翻譯。我們對因使用此翻譯而產生的任何誤解或錯誤解釋不承擔責任。