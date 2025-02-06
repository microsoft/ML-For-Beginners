# クイズ

これらのクイズは、https://aka.ms/ml-beginners にあるMLカリキュラムの前後のレクチャークイズです。

## プロジェクトセットアップ

```
npm install
```

### 開発のためのコンパイルとホットリロード

```
npm run serve
```

### 本番用のコンパイルと最小化

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

以下は、始めるためのステップバイステップガイドです：

1. GitHubリポジトリをフォークする
静的WebアプリのコードをGitHubリポジトリに入れてください。このリポジトリをフォークします。

2. Azure静的Webアプリを作成する
- [Azureアカウント](http://azure.microsoft.com) を作成する
- [Azureポータル](https://portal.azure.com) にアクセスする
- 「リソースの作成」をクリックして、「Static Web App」を検索する。
- 「作成」をクリックする。

3. 静的Webアプリを設定する
- 基本: サブスクリプション: Azureサブスクリプションを選択します。
- リソースグループ: 新しいリソースグループを作成するか、既存のものを使用します。
- 名前: 静的Webアプリの名前を入力します。
- リージョン: ユーザーに最も近いリージョンを選択します。

- #### デプロイメントの詳細:
- ソース: 「GitHub」を選択します。
- GitHubアカウント: AzureがGitHubアカウントにアクセスすることを許可します。
- 組織: GitHubの組織を選択します。
- リポジトリ: 静的Webアプリを含むリポジトリを選択します。
- ブランチ: デプロイするブランチを選択します。

- #### ビルドの詳細:
- ビルドプリセット: アプリが構築されているフレームワークを選択します（例：React, Angular, Vueなど）。
- アプリの場所: アプリコードを含むフォルダを指定します（例：ルートにある場合は/）。
- APIの場所: APIがある場合、その場所を指定します（オプション）。
- 出力場所: ビルド出力が生成されるフォルダを指定します（例：buildまたはdist）。

4. レビューと作成
設定を確認し、「作成」をクリックします。Azureは必要なリソースを設定し、リポジトリにGitHub Actionsワークフローを作成します。

5. GitHub Actionsワークフロー
Azureは自動的にリポジトリにGitHub Actionsワークフローファイル (.github/workflows/azure-static-web-apps-<name>.yml) を作成します。このワークフローがビルドとデプロイのプロセスを処理します。

6. デプロイの監視
GitHubリポジトリの「Actions」タブに移動します。
ワークフローが実行中であることが確認できます。このワークフローは静的WebアプリをAzureにビルドしてデプロイします。
ワークフローが完了すると、アプリは提供されたAzure URLで公開されます。

### ワークフローファイルの例

以下は、GitHub Actionsワークフローファイルの例です：
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

**免責事項**：
この文書は、機械翻訳AIサービスを使用して翻訳されています。正確さを期していますが、自動翻訳には誤りや不正確さが含まれる場合があります。元の言語の文書が権威ある情報源と見なされるべきです。重要な情報については、専門の人間による翻訳をお勧めします。この翻訳の使用に起因する誤解や誤訳について、当社は一切の責任を負いません。