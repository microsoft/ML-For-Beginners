<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:39:12+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "hk"
}
-->
# 疑難排解指南

本指南旨在幫助您解決使用《機器學習初學者》課程時常見的問題。如果您在此未找到解決方案，請查看我們的 [Discord 討論區](https://aka.ms/foundry/discord) 或 [提交問題](https://github.com/microsoft/ML-For-Beginners/issues)。

## 目錄

- [安裝問題](../..)
- [Jupyter Notebook 問題](../..)
- [Python 套件問題](../..)
- [R 環境問題](../..)
- [測驗應用程式問題](../..)
- [數據和文件路徑問題](../..)
- [常見錯誤訊息](../..)
- [性能問題](../..)
- [環境與配置](../..)

---

## 安裝問題

### Python 安裝

**問題**：`python: command not found`

**解決方案**：
1. 從 [python.org](https://www.python.org/downloads/) 安裝 Python 3.8 或更高版本
2. 驗證安裝：`python --version` 或 `python3 --version`
3. 在 macOS/Linux 上，可能需要使用 `python3` 而非 `python`

**問題**：多個 Python 版本導致衝突

**解決方案**：
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter 安裝

**問題**：`jupyter: command not found`

**解決方案**：
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**問題**：Jupyter 無法在瀏覽器中啟動

**解決方案**：
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R 安裝

**問題**：R 套件無法安裝

**解決方案**：
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**問題**：IRkernel 無法在 Jupyter 中使用

**解決方案**：
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook 問題

### 核心問題

**問題**：核心不斷崩潰或重啟

**解決方案**：
1. 重啟核心：`Kernel → Restart`
2. 清除輸出並重啟：`Kernel → Restart & Clear Output`
3. 檢查記憶體問題（請參閱 [性能問題](../..)）
4. 嘗試逐個執行單元以識別有問題的代碼

**問題**：選擇了錯誤的 Python 核心

**解決方案**：
1. 檢查當前核心：`Kernel → Change Kernel`
2. 選擇正確的 Python 版本
3. 如果核心缺失，請創建它：
```bash
python -m ipykernel install --user --name=ml-env
```

**問題**：核心無法啟動

**解決方案**：
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook 單元問題

**問題**：單元正在執行但未顯示輸出

**解決方案**：
1. 檢查單元是否仍在執行（查看 `[*]` 指示器）
2. 重啟核心並執行所有單元：`Kernel → Restart & Run All`
3. 檢查瀏覽器控制台是否有 JavaScript 錯誤（按 F12）

**問題**：無法執行單元 - 點擊「執行」無反應

**解決方案**：
1. 檢查 Jupyter 伺服器是否仍在終端中運行
2. 刷新瀏覽器頁面
3. 關閉並重新打開 Notebook
4. 重啟 Jupyter 伺服器

---

## Python 套件問題

### 匯入錯誤

**問題**：`ModuleNotFoundError: No module named 'sklearn'`

**解決方案**：
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**問題**：`ImportError: cannot import name 'X' from 'sklearn'`

**解決方案**：
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### 版本衝突

**問題**：套件版本不兼容錯誤

**解決方案**：
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**問題**：`pip install` 因權限錯誤而失敗

**解決方案**：
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### 數據加載問題

**問題**：加載 CSV 文件時出現 `FileNotFoundError`

**解決方案**：
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

## R 環境問題

### 套件安裝

**問題**：套件安裝因編譯錯誤而失敗

**解決方案**：
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

**問題**：`tidyverse` 無法安裝

**解決方案**：
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown 問題

**問題**：RMarkdown 無法渲染

**解決方案**：
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

## 測驗應用程式問題

### 構建與安裝

**問題**：`npm install` 失敗

**解決方案**：
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

**問題**：8080 端口已被佔用

**解決方案**：
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

### 構建錯誤

**問題**：`npm run build` 失敗

**解決方案**：
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**問題**：Linting 錯誤阻止構建

**解決方案**：
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## 數據和文件路徑問題

### 路徑問題

**問題**：運行 Notebook 時找不到數據文件

**解決方案**：
1. **始終從 Notebook 所在目錄運行**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **檢查代碼中的相對路徑**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **必要時使用絕對路徑**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### 缺少數據文件

**問題**：數據集文件丟失

**解決方案**：
1. 檢查數據是否應包含在倉庫中 - 大多數數據集已包含
2. 某些課程可能需要下載數據 - 查看課程 README
3. 確保已拉取最新更改：
   ```bash
   git pull origin main
   ```

---

## 常見錯誤訊息

### 記憶體錯誤

**錯誤**：`MemoryError` 或核心在處理數據時崩潰

**解決方案**：
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

### 收斂警告

**警告**：`ConvergenceWarning: Maximum number of iterations reached`

**解決方案**：
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 繪圖問題

**問題**：Jupyter 中未顯示繪圖

**解決方案**：
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**問題**：Seaborn 繪圖顯示異常或報錯

**解決方案**：
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/編碼錯誤

**問題**：讀取文件時出現 `UnicodeDecodeError`

**解決方案**：
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## 性能問題

### Notebook 執行緩慢

**問題**：Notebook 運行速度非常慢

**解決方案**：
1. **重啟核心以釋放記憶體**：`Kernel → Restart`
2. **關閉未使用的 Notebook** 以釋放資源
3. **使用較小的數據樣本進行測試**：
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **分析代碼性能** 以找到瓶頸：
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### 高記憶體使用率

**問題**：系統記憶體不足

**解決方案**：
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

## 環境與配置

### 虛擬環境問題

**問題**：虛擬環境無法激活

**解決方案**：
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

**問題**：套件已安裝但在 Notebook 中找不到

**解決方案**：
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git 問題

**問題**：無法拉取最新更改 - 合併衝突

**解決方案**：
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

### VS Code 集成

**問題**：Jupyter Notebook 無法在 VS Code 中打開

**解決方案**：
1. 在 VS Code 中安裝 Python 擴展
2. 安裝 Jupyter 擴展
3. 選擇正確的 Python 解釋器：`Ctrl+Shift+P` → "Python: Select Interpreter"
4. 重啟 VS Code

---

## 其他資源

- **Discord 討論區**：[在 #ml-for-beginners 頻道提問並分享解決方案](https://aka.ms/foundry/discord)
- **Microsoft Learn**：[機器學習初學者模組](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **視頻教程**：[YouTube 播放列表](https://aka.ms/ml-beginners-videos)
- **問題追蹤器**：[報告錯誤](https://github.com/microsoft/ML-For-Beginners/issues)

---

## 仍有問題？

如果您已嘗試上述解決方案但仍遇到問題：

1. **搜索現有問題**：[GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **查看 Discord 討論**：[Discord Discussions](https://aka.ms/foundry/discord)
3. **提交新問題**：請包括以下內容：
   - 您的操作系統及版本
   - Python/R 版本
   - 錯誤訊息（完整回溯）
   - 重現問題的步驟
   - 您已嘗試的解決方法

我們隨時為您提供幫助！🚀

---

**免責聲明**：  
本文件已使用人工智能翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。儘管我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要信息，建議使用專業人工翻譯。我們對因使用此翻譯而引起的任何誤解或錯誤解釋概不負責。