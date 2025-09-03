<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T17:58:57+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "hk"
}
-->
# 測驗

這些測驗是機器學習課程（https://aka.ms/ml-beginners）的課前和課後測驗。

## 專案設置

```
npm install
```

### 編譯並熱加載以進行開發

```
npm run serve
```

### 編譯並壓縮以進行生產環境使用

```
npm run build
```

### 檢查並修復文件

```
npm run lint
```

### 自定義配置

請參閱[配置參考](https://cli.vuejs.org/config/)。

致謝：感謝這個測驗應用程式的原始版本：https://github.com/arpan45/simple-quiz-vue

## 部署到 Azure

以下是幫助你開始的逐步指南：

1. Fork 一個 GitHub 儲存庫  
確保你的靜態網站應用程式代碼在你的 GitHub 儲存庫中。Fork 此儲存庫。

2. 建立 Azure 靜態網站應用程式  
- 建立並[註冊 Azure 帳戶](http://azure.microsoft.com)  
- 前往 [Azure 入口網站](https://portal.azure.com)  
- 點擊「建立資源」，然後搜尋「Static Web App」。  
- 點擊「建立」。  

3. 配置靜態網站應用程式  
- 基本設定：  
  - 訂閱：選擇你的 Azure 訂閱。  
  - 資源群組：建立一個新的資源群組或使用現有的資源群組。  
  - 名稱：為你的靜態網站應用程式提供一個名稱。  
  - 區域：選擇最接近你的使用者的區域。  

- #### 部署詳情：  
  - 原始碼來源：選擇「GitHub」。  
  - GitHub 帳戶：授權 Azure 訪問你的 GitHub 帳戶。  
  - 組織：選擇你的 GitHub 組織。  
  - 儲存庫：選擇包含靜態網站應用程式的儲存庫。  
  - 分支：選擇你想要部署的分支。  

- #### 構建詳情：  
  - 構建預設值：選擇你的應用程式所使用的框架（例如 React、Angular、Vue 等）。  
  - 應用程式位置：指定包含應用程式代碼的文件夾（例如，如果在根目錄，則為 /）。  
  - API 位置：如果有 API，請指定其位置（可選）。  
  - 輸出位置：指定構建輸出生成的文件夾（例如 build 或 dist）。  

4. 檢查並建立  
檢查你的設置，然後點擊「建立」。Azure 會設置必要的資源，並在你的儲存庫中建立一個 GitHub Actions 工作流程。

5. GitHub Actions 工作流程  
Azure 會自動在你的儲存庫中建立一個 GitHub Actions 工作流程文件（.github/workflows/azure-static-web-apps-<name>.yml）。這個工作流程將處理構建和部署過程。

6. 監控部署  
前往你的 GitHub 儲存庫中的「Actions」標籤。  
你應該會看到一個工作流程正在運行。這個工作流程會構建並部署你的靜態網站應用程式到 Azure。  
當工作流程完成後，你的應用程式將在提供的 Azure URL 上線。

### 範例工作流程文件

以下是一個 GitHub Actions 工作流程文件的範例：  
name: Azure Static Web Apps CI/CD  
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### 其他資源
- [Azure 靜態網站應用程式文件](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions 文件](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**免責聲明**：  
本文件已使用人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要信息，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋概不負責。