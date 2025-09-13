<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T23:47:49+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "ja"
}
-->
# クイズ

これらのクイズは、https://aka.ms/ml-beginners にあるMLカリキュラムの講義前後のクイズです。

## プロジェクトのセットアップ

```
npm install
```

### 開発用のコンパイルとホットリロード

```
npm run serve
```

### 本番環境用のコンパイルと最小化

```
npm run build
```

### ファイルのリントと修正

```
npm run lint
```

### 設定のカスタマイズ

[Configuration Reference](https://cli.vuejs.org/config/) を参照してください。

クレジット: このクイズアプリのオリジナルバージョンに感謝します: https://github.com/arpan45/simple-quiz-vue

## Azureへのデプロイ

以下は、始めるためのステップバイステップガイドです:

1. GitHubリポジトリをフォークする  
静的ウェブアプリのコードをGitHubリポジトリに保存してください。このリポジトリをフォークします。

2. Azure Static Web Appを作成する  
- [Azureアカウント](http://azure.microsoft.com) を作成します。  
- [Azureポータル](https://portal.azure.com) にアクセスします。  
- 「リソースの作成」をクリックし、「Static Web App」を検索します。  
- 「作成」をクリックします。

3. Static Web Appを設定する  
- 基本設定:  
  - サブスクリプション: Azureのサブスクリプションを選択します。  
  - リソースグループ: 新しいリソースグループを作成するか、既存のものを使用します。  
  - 名前: 静的ウェブアプリの名前を入力します。  
  - リージョン: ユーザーに最も近いリージョンを選択します。

- #### デプロイメントの詳細:  
  - ソース: 「GitHub」を選択します。  
  - GitHubアカウント: AzureにGitHubアカウントへのアクセスを許可します。  
  - 組織: GitHubの組織を選択します。  
  - リポジトリ: 静的ウェブアプリを含むリポジトリを選択します。  
  - ブランチ: デプロイするブランチを選択します。

- #### ビルドの詳細:  
  - ビルドプリセット: アプリが構築されているフレームワークを選択します（例: React、Angular、Vueなど）。  
  - アプリの場所: アプリコードが含まれるフォルダを指定します（例: ルートにある場合は /）。  
  - APIの場所: APIがある場合はその場所を指定します（オプション）。  
  - 出力の場所: ビルド出力が生成されるフォルダを指定します（例: build または dist）。

4. 設定の確認と作成  
設定を確認し、「作成」をクリックします。Azureが必要なリソースを設定し、GitHub Actionsのワークフローをリポジトリに作成します。

5. GitHub Actionsワークフロー  
Azureは自動的にリポジトリ内にGitHub Actionsワークフローファイル (.github/workflows/azure-static-web-apps-<name>.yml) を作成します。このワークフローがビルドとデプロイプロセスを処理します。

6. デプロイの監視  
GitHubリポジトリの「Actions」タブに移動します。  
ワークフローが実行されているのが確認できます。このワークフローが静的ウェブアプリをAzureにビルドしてデプロイします。  
ワークフローが完了すると、提供されたAzure URLでアプリが公開されます。

### ワークフローファイルの例

以下はGitHub Actionsワークフローの例です:
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

### 追加リソース
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**免責事項**:  
この文書はAI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された文書を正式な情報源としてご参照ください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤認について、当方は一切の責任を負いません。