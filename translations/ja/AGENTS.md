<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:01:56+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ja"
}
-->
# AGENTS.md

## プロジェクト概要

これは **Machine Learning for Beginners** という、Python（主にScikit-learn）とRを使用して古典的な機械学習の概念を網羅した12週間、26レッスンの包括的なカリキュラムです。このリポジトリは、自己ペースで学習できるリソースとして設計されており、実践的なプロジェクト、クイズ、課題が含まれています。各レッスンでは、世界中のさまざまな文化や地域の実際のデータを通じて機械学習の概念を探求します。

主な構成要素:
- **教育コンテンツ**: 機械学習の導入、回帰、分類、クラスタリング、NLP、時系列、強化学習を含む26のレッスン
- **クイズアプリケーション**: Vue.jsベースのクイズアプリで、レッスン前後の評価を実施
- **多言語対応**: GitHub Actionsを使用して40以上の言語に自動翻訳
- **二言語対応**: Python（Jupyterノートブック）とR（R Markdownファイル）の両方で利用可能
- **プロジェクトベース学習**: 各トピックに実践的なプロジェクトと課題を含む

## リポジトリ構造

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

各レッスンフォルダには通常以下が含まれます:
- `README.md` - レッスンの主要コンテンツ
- `notebook.ipynb` - PythonのJupyterノートブック
- `solution/` - 解答コード（PythonおよびRバージョン）
- `assignment.md` - 練習問題
- `images/` - ビジュアルリソース

## セットアップコマンド

### Pythonレッスンの場合

ほとんどのレッスンはJupyterノートブックを使用します。必要な依存関係をインストールしてください:

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

### Rレッスンの場合

Rレッスンは`solution/R/`フォルダ内に`.rmd`または`.ipynb`ファイルとしてあります:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### クイズアプリケーションの場合

クイズアプリは`quiz-app/`ディレクトリにあるVue.jsアプリケーションです:

```bash
cd quiz-app
npm install
```

### ドキュメントサイトの場合

ローカルでドキュメントを実行するには:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## 開発ワークフロー

### レッスンノートブックの操作

1. レッスンディレクトリに移動（例: `2-Regression/1-Tools/`）
2. Jupyterノートブックを開く:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. レッスン内容と練習問題を進める
4. 必要に応じて`solution/`フォルダ内の解答を確認する

### Python開発

- レッスンは標準的なPythonデータサイエンスライブラリを使用
- インタラクティブ学習のためのJupyterノートブック
- 各レッスンの`solution/`フォルダに解答コードが含まれる

### R開発

- Rレッスンは`.rmd`形式（R Markdown）
- 解答は`solution/R/`サブディレクトリに配置
- RStudioまたはRカーネル付きJupyterを使用してRノートブックを実行

### クイズアプリケーション開発

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

## テスト手順

### クイズアプリケーションのテスト

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**注意**: これは主に教育用カリキュラムリポジトリです。レッスン内容に対する自動テストはありません。検証は以下を通じて行います:
- レッスンの練習問題を完了する
- ノートブックのセルを正常に実行する
- 解答と期待される結果を照合する

## コードスタイルガイドライン

### Pythonコード
- PEP 8スタイルガイドラインに従う
- 明確で説明的な変数名を使用
- 複雑な操作にはコメントを追加
- Jupyterノートブックには概念を説明するMarkdownセルを含める

### JavaScript/Vue.js（クイズアプリ）
- Vue.jsスタイルガイドに従う
- ESLint設定は`quiz-app/package.json`に記載
- `npm run lint`を実行して問題をチェックおよび自動修正

### ドキュメント
- Markdownファイルは明確で構造化されていること
- フェンスコードブロック内にコード例を含める
- 内部参照には相対リンクを使用
- 既存のフォーマット規則に従う

## ビルドとデプロイ

### クイズアプリケーションのデプロイ

クイズアプリはAzure Static Web Appsにデプロイ可能です:

1. **前提条件**:
   - Azureアカウント
   - GitHubリポジトリ（すでにフォーク済み）

2. **Azureへのデプロイ**:
   - Azure Static Web Appリソースを作成
   - GitHubリポジトリに接続
   - アプリの場所を設定: `/quiz-app`
   - 出力場所を設定: `dist`
   - Azureが自動的にGitHub Actionsワークフローを作成

3. **GitHub Actionsワークフロー**:
   - ワークフローファイルは`.github/workflows/azure-static-web-apps-*.yml`に作成
   - メインブランチへのプッシュ時に自動的にビルドとデプロイ

### ドキュメントPDF

ドキュメントからPDFを生成:

```bash
npm install
npm run convert
```

## 翻訳ワークフロー

**重要**: 翻訳はGitHub Actionsを使用してCo-op Translatorで自動化されています。

- 翻訳は`main`ブランチに変更がプッシュされると自動生成されます
- **コンテンツを手動で翻訳しないでください** - システムが処理します
- ワークフローは`.github/workflows/co-op-translator.yml`で定義
- Azure AI/OpenAIサービスを使用して翻訳
- 40以上の言語をサポート

## 貢献ガイドライン

### コンテンツ貢献者向け

1. **リポジトリをフォーク**し、フィーチャーブランチを作成
2. **レッスン内容を変更**（追加/更新する場合）
3. **翻訳されたファイルを変更しない** - 自動生成されます
4. **コードをテスト** - すべてのノートブックセルが正常に実行されることを確認
5. **リンクと画像が正しく動作することを確認**
6. **プルリクエストを提出** - 明確な説明を添えて

### プルリクエストガイドライン

- **タイトル形式**: `[セクション] 変更内容の簡単な説明`
  - 例: `[Regression] レッスン5のタイポ修正`
  - 例: `[Quiz-App] 依存関係の更新`
- **提出前**:
  - すべてのノートブックセルがエラーなく実行されることを確認
  - クイズアプリを変更した場合は`npm run lint`を実行
  - Markdownのフォーマットを確認
  - 新しいコード例をテスト
- **PRに含めるべき内容**:
  - 変更内容の説明
  - 変更理由
  - UI変更の場合はスクリーンショット
- **行動規範**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)に従う
- **CLA**: 貢献者ライセンス契約に署名が必要

## レッスン構造

各レッスンは一貫したパターンに従います:

1. **講義前クイズ** - 基本知識をテスト
2. **レッスン内容** - 書かれた指示と説明
3. **コードデモンストレーション** - ノートブックでの実践例
4. **知識チェック** - 理解度を確認
5. **チャレンジ** - 概念を独自に適用
6. **課題** - 拡張練習
7. **講義後クイズ** - 学習成果を評価

## 共通コマンドリファレンス

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

## 追加リソース

- **Microsoft Learn Collection**: [ML for Beginnersモジュール](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **クイズアプリ**: [オンラインクイズ](https://ff-quizzes.netlify.app/en/ml/)
- **ディスカッションボード**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **ビデオウォークスルー**: [YouTubeプレイリスト](https://aka.ms/ml-beginners-videos)

## 主な技術

- **Python**: 機械学習レッスンの主要言語（Scikit-learn、Pandas、NumPy、Matplotlib）
- **R**: tidyverse、tidymodels、caretを使用した代替実装
- **Jupyter**: Pythonレッスン用のインタラクティブノートブック
- **R Markdown**: Rレッスン用のドキュメント
- **Vue.js 3**: クイズアプリケーションフレームワーク
- **Flask**: 機械学習モデルデプロイ用のWebアプリケーションフレームワーク
- **Docsify**: ドキュメントサイトジェネレーター
- **GitHub Actions**: CI/CDおよび自動翻訳

## セキュリティに関する考慮事項

- **コードに秘密情報を含めない**: APIキーや認証情報をコミットしない
- **依存関係**: npmおよびpipパッケージを最新に保つ
- **ユーザー入力**: FlaskのWebアプリ例には基本的な入力検証を含む
- **機密データ**: 使用するデータセットは公開されており、機密性はない

## トラブルシューティング

### Jupyterノートブック

- **カーネルの問題**: セルが停止した場合はカーネルを再起動: Kernel → Restart
- **インポートエラー**: 必要なパッケージがpipでインストールされていることを確認
- **パスの問題**: ノートブックをその含まれるディレクトリから実行

### クイズアプリケーション

- **npm installが失敗**: npmキャッシュをクリア: `npm cache clean --force`
- **ポート競合**: ポートを変更: `npm run serve -- --port 8081`
- **ビルドエラー**: `node_modules`を削除して再インストール: `rm -rf node_modules && npm install`

### Rレッスン

- **パッケージが見つからない**: 以下でインストール: `install.packages("package-name")`
- **RMarkdownのレンダリング**: rmarkdownパッケージがインストールされていることを確認
- **カーネルの問題**: JupyterでIRkernelをインストールする必要がある場合あり

## プロジェクト固有の注意事項

- これは主に**学習カリキュラム**であり、プロダクションコードではありません
- 実践を通じて**機械学習の概念を理解すること**に重点を置いています
- コード例は**最適化よりも明確さを優先**
- ほとんどのレッスンは**自己完結型**で、独立して完了可能
- **解答が提供されます**が、まず練習問題に取り組むべきです
- リポジトリは**Docsify**を使用してビルドステップなしでWebドキュメントを提供
- **スケッチノート**が概念のビジュアル要約を提供
- **多言語対応**によりコンテンツが世界中で利用可能

---

**免責事項**:  
この文書は、AI翻訳サービス [Co-op Translator](https://github.com/Azure/co-op-translator) を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された文書を正式な情報源としてお考えください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解について、当方は責任を負いません。