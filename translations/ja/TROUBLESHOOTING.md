<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:40:05+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ja"
}
-->
# トラブルシューティングガイド

このガイドは、Machine Learning for Beginners カリキュラムでよくある問題を解決するための手助けをします。ここで解決策が見つからない場合は、[Discord Discussions](https://aka.ms/foundry/discord) を確認するか、[問題を報告](https://github.com/microsoft/ML-For-Beginners/issues)してください。

## 目次

- [インストールの問題](../..)
- [Jupyter Notebook の問題](../..)
- [Python パッケージの問題](../..)
- [R 環境の問題](../..)
- [クイズアプリケーションの問題](../..)
- [データとファイルパスの問題](../..)
- [よくあるエラーメッセージ](../..)
- [パフォーマンスの問題](../..)
- [環境と設定](../..)

---

## インストールの問題

### Python のインストール

**問題**: `python: command not found`

**解決策**:
1. [python.org](https://www.python.org/downloads/) から Python 3.8 以上をインストールしてください。
2. インストールを確認: `python --version` または `python3 --version`
3. macOS/Linux では、`python` の代わりに `python3` を使用する必要がある場合があります。

**問題**: 複数の Python バージョンが競合している

**解決策**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter のインストール

**問題**: `jupyter: command not found`

**解決策**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**問題**: Jupyter がブラウザで起動しない

**解決策**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R のインストール

**問題**: R パッケージがインストールできない

**解決策**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**問題**: IRkernel が Jupyter で利用できない

**解決策**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook の問題

### カーネルの問題

**問題**: カーネルが頻繁に停止または再起動する

**解決策**:
1. カーネルを再起動: `Kernel → Restart`
2. 出力をクリアして再起動: `Kernel → Restart & Clear Output`
3. メモリの問題を確認 (詳細は [パフォーマンスの問題](../..) を参照)
4. 問題のあるコードを特定するためにセルを個別に実行

**問題**: 間違った Python カーネルが選択されている

**解決策**:
1. 現在のカーネルを確認: `Kernel → Change Kernel`
2. 正しい Python バージョンを選択
3. カーネルが見つからない場合は作成:
```bash
python -m ipykernel install --user --name=ml-env
```

**問題**: カーネルが起動しない

**解決策**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### ノートブックセルの問題

**問題**: セルが実行されているが出力が表示されない

**解決策**:
1. セルがまだ実行中か確認 (`[*]` インジケータを探す)
2. カーネルを再起動してすべてのセルを実行: `Kernel → Restart & Run All`
3. ブラウザコンソールで JavaScript エラーを確認 (F12)

**問題**: セルが実行できない - "Run" をクリックしても反応がない

**解決策**:
1. Jupyter サーバーがターミナルでまだ動作しているか確認
2. ブラウザページをリフレッシュ
3. ノートブックを閉じて再度開く
4. Jupyter サーバーを再起動

---

## Python パッケージの問題

### インポートエラー

**問題**: `ModuleNotFoundError: No module named 'sklearn'`

**解決策**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**問題**: `ImportError: cannot import name 'X' from 'sklearn'`

**解決策**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### バージョンの競合

**問題**: パッケージのバージョンが互換性エラーを引き起こす

**解決策**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**問題**: `pip install` が権限エラーで失敗する

**解決策**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### データ読み込みの問題

**問題**: CSV ファイルを読み込む際に `FileNotFoundError` が発生

**解決策**:
```python
import os
# Check current working directory
print(os.getcwd())

# Use relative paths from notebook location
df = pd.read_csv('../../data/filename.csv')

# Or use absolute paths
df = pd.read_csv('/full/path/to/data/filename.csv')
```

---

## R 環境の問題

### パッケージのインストール

**問題**: パッケージのインストールがコンパイルエラーで失敗する

**解決策**:
```r
# Install binary version (Windows/macOS)
install.packages("package-name", type = "binary")

# Update R to latest version if packages require it
# Check R version
R.version.string

# Install system dependencies (Linux)
# For Ubuntu/Debian, in terminal:
# sudo apt-get install r-base-dev
```

**問題**: `tidyverse` がインストールできない

**解決策**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown の問題

**問題**: RMarkdown がレンダリングされない

**解決策**:
```r
# Install/update rmarkdown
install.packages("rmarkdown")

# Install pandoc if needed
install.packages("pandoc")

# For PDF output, install tinytex
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## クイズアプリケーションの問題

### ビルドとインストール

**問題**: `npm install` が失敗する

**解決策**:
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install

# If still fails, try with legacy peer deps
npm install --legacy-peer-deps
```

**問題**: ポート 8080 がすでに使用されている

**解決策**:
```bash
# Use different port
npm run serve -- --port 8081

# Or find and kill process using port 8080
# On Linux/macOS:
lsof -ti:8080 | xargs kill -9

# On Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### ビルドエラー

**問題**: `npm run build` が失敗する

**解決策**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**問題**: リンティングエラーがビルドを妨げる

**解決策**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## データとファイルパスの問題

### パスの問題

**問題**: ノートブックを実行する際にデータファイルが見つからない

**解決策**:
1. **ノートブックをそのディレクトリ内から実行する**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **コード内の相対パスを確認する**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **必要に応じて絶対パスを使用する**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### データファイルが見つからない

**問題**: データセットファイルが欠落している

**解決策**:
1. データがリポジトリ内にあるべきか確認 - ほとんどのデータセットは含まれています。
2. 一部のレッスンではデータのダウンロードが必要な場合があります - レッスンの README を確認してください。
3. 最新の変更をプルしていることを確認:
   ```bash
   git pull origin main
   ```

---

## よくあるエラーメッセージ

### メモリエラー

**エラー**: `MemoryError` またはデータ処理中にカーネルが停止

**解決策**:
```python
# Load data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Or read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# Free memory when done
del large_dataframe
import gc
gc.collect()
```

### 収束警告

**警告**: `ConvergenceWarning: Maximum number of iterations reached`

**解決策**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### プロットの問題

**問題**: Jupyter でプロットが表示されない

**解決策**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**問題**: Seaborn のプロットが異なって見える、またはエラーが発生する

**解決策**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/エンコーディングエラー

**問題**: ファイルを読み込む際に `UnicodeDecodeError` が発生

**解決策**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## パフォーマンスの問題

### ノートブックの実行が遅い

**問題**: ノートブックの実行が非常に遅い

**解決策**:
1. **メモリを解放するためにカーネルを再起動**: `Kernel → Restart`
2. **使用していないノートブックを閉じる**: リソースを解放
3. **テスト用に小さいデータサンプルを使用**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **コードをプロファイル**してボトルネックを特定:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### 高メモリ使用量

**問題**: システムがメモリ不足になる

**解決策**:
```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize data types
df['column'] = df['column'].astype('int32')  # Instead of int64

# Drop unnecessary columns
df = df[['col1', 'col2']]  # Keep only needed columns

# Process in batches
for batch in np.array_split(df, 10):
    process(batch)
```

---

## 環境と設定

### 仮想環境の問題

**問題**: 仮想環境が有効化されない

**解決策**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Check if activated (should show venv name in prompt)
which python  # Should point to venv python
```

**問題**: パッケージがインストールされているがノートブックで見つからない

**解決策**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git の問題

**問題**: 最新の変更をプルできない - マージ競合

**解決策**:
```bash
# Stash your changes
git stash

# Pull latest
git pull origin main

# Reapply your changes
git stash pop

# If conflicts, resolve manually or:
git checkout --theirs path/to/file  # Take remote version
git checkout --ours path/to/file    # Keep your version
```

### VS Code の統合

**問題**: Jupyter ノートブックが VS Code で開かない

**解決策**:
1. VS Code に Python 拡張機能をインストール
2. VS Code に Jupyter 拡張機能をインストール
3. 正しい Python インタープリターを選択: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code を再起動

---

## 追加リソース

- **Discord Discussions**: [#ml-for-beginners チャンネルで質問や解決策を共有](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners モジュール](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **ビデオチュートリアル**: [YouTube プレイリスト](https://aka.ms/ml-beginners-videos)
- **問題トラッカー**: [バグを報告](https://github.com/microsoft/ML-For-Beginners/issues)

---

## まだ問題が解決しない場合

上記の解決策を試しても問題が解決しない場合:

1. **既存の問題を検索**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord のディスカッションを確認**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **新しい問題を報告**: 以下を含めてください:
   - 使用しているオペレーティングシステムとバージョン
   - Python/R のバージョン
   - エラーメッセージ (完全なトレースバック)
   - 問題を再現する手順
   - すでに試したこと

私たちはお手伝いします！ 🚀

---

**免責事項**:  
この文書は、AI翻訳サービス[Co-op Translator](https://github.com/Azure/co-op-translator)を使用して翻訳されています。正確性を追求しておりますが、自動翻訳には誤りや不正確な部分が含まれる可能性があります。元の言語で記載された文書を正式な情報源としてお考えください。重要な情報については、専門の人間による翻訳を推奨します。この翻訳の使用に起因する誤解や誤解釈について、当方は一切の責任を負いません。