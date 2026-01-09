<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:43:02+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "pa"
}
-->
# ਸਮੱਸਿਆ ਹੱਲ ਕਰਨ ਦੀ ਗਾਈਡ

ਇਹ ਗਾਈਡ ਤੁਹਾਨੂੰ Machine Learning for Beginners ਪਾਠਕ੍ਰਮ ਨਾਲ ਕੰਮ ਕਰਦੇ ਸਮੇਂ ਆਮ ਸਮੱਸਿਆਵਾਂ ਨੂੰ ਹੱਲ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰਦੀ ਹੈ। ਜੇ ਤੁਸੀਂ ਇੱਥੇ ਹੱਲ ਨਹੀਂ ਲੱਭਦੇ, ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਸਾਡੇ [Discord Discussions](https://aka.ms/foundry/discord) ਜਾਂ [ਇੱਕ ਮੁੱਦਾ ਖੋਲ੍ਹੋ](https://github.com/microsoft/ML-For-Beginners/issues) ਨੂੰ ਚੈੱਕ ਕਰੋ।

## ਸਮੱਗਰੀ ਦੀ ਸੂਚੀ

- [ਇੰਸਟਾਲੇਸ਼ਨ ਸਮੱਸਿਆਵਾਂ](../..)
- [Jupyter Notebook ਸਮੱਸਿਆਵਾਂ](../..)
- [Python ਪੈਕੇਜ ਸਮੱਸਿਆਵਾਂ](../..)
- [R Environment ਸਮੱਸਿਆਵਾਂ](../..)
- [ਕੁਇਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਸਮੱਸਿਆਵਾਂ](../..)
- [ਡਾਟਾ ਅਤੇ ਫਾਈਲ ਪਾਥ ਸਮੱਸਿਆਵਾਂ](../..)
- [ਆਮ ਗਲਤੀ ਸੁਨੇਹੇ](../..)
- [ਕਾਰਗੁਜ਼ਾਰੀ ਸਮੱਸਿਆਵਾਂ](../..)
- [Environment ਅਤੇ Configuration](../..)

---

## ਇੰਸਟਾਲੇਸ਼ਨ ਸਮੱਸਿਆਵਾਂ

### Python ਇੰਸਟਾਲੇਸ਼ਨ

**ਸਮੱਸਿਆ**: `python: command not found`

**ਹੱਲ**:
1. [python.org](https://www.python.org/downloads/) ਤੋਂ Python 3.8 ਜਾਂ ਇਸ ਤੋਂ ਉੱਚਾ ਵਰਜਨ ਇੰਸਟਾਲ ਕਰੋ।
2. ਇੰਸਟਾਲੇਸ਼ਨ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ: `python --version` ਜਾਂ `python3 --version`
3. macOS/Linux 'ਤੇ, ਤੁਹਾਨੂੰ `python3` ਦੀ ਵਰਤੋਂ ਕਰਨੀ ਪੈ ਸਕਦੀ ਹੈ `python` ਦੀ ਬਜਾਏ।

**ਸਮੱਸਿਆ**: ਕਈ Python ਵਰਜਨ ਟਕਰਾਅ ਪੈਦਾ ਕਰ ਰਹੇ ਹਨ

**ਹੱਲ**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter ਇੰਸਟਾਲੇਸ਼ਨ

**ਸਮੱਸਿਆ**: `jupyter: command not found`

**ਹੱਲ**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**ਸਮੱਸਿਆ**: Jupyter ਬ੍ਰਾਊਜ਼ਰ ਵਿੱਚ ਸ਼ੁਰੂ ਨਹੀਂ ਹੋ ਰਿਹਾ

**ਹੱਲ**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R ਇੰਸਟਾਲੇਸ਼ਨ

**ਸਮੱਸਿਆ**: R ਪੈਕੇਜ ਇੰਸਟਾਲ ਨਹੀਂ ਹੋ ਰਹੇ

**ਹੱਲ**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**ਸਮੱਸਿਆ**: IRkernel Jupyter ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ

**ਹੱਲ**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook ਸਮੱਸਿਆਵਾਂ

### Kernel ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: Kernel ਮੁੜ-ਮੁੜ ਮਰਦਾ ਜਾਂ ਰੀਸਟਾਰਟ ਹੁੰਦਾ ਹੈ

**ਹੱਲ**:
1. Kernel ਨੂੰ ਰੀਸਟਾਰਟ ਕਰੋ: `Kernel → Restart`
2. ਆਉਟਪੁੱਟ ਸਾਫ਼ ਕਰੋ ਅਤੇ ਰੀਸਟਾਰਟ ਕਰੋ: `Kernel → Restart & Clear Output`
3. ਮੈਮਰੀ ਸਮੱਸਿਆਵਾਂ ਦੀ ਜਾਂਚ ਕਰੋ (ਵੇਖੋ [ਕਾਰਗੁਜ਼ਾਰੀ ਸਮੱਸਿਆਵਾਂ](../..))
4. ਸਮੱਸਿਆਵਾਲੀ ਕੋਡ ਦੀ ਪਹਿਚਾਣ ਕਰਨ ਲਈ ਸੈੱਲ ਨੂੰ ਵੱਖ-ਵੱਖ ਚਲਾਓ

**ਸਮੱਸਿਆ**: ਗਲਤ Python kernel ਚੁਣਿਆ ਗਿਆ

**ਹੱਲ**:
1. ਮੌਜੂਦਾ kernel ਦੀ ਜਾਂਚ ਕਰੋ: `Kernel → Change Kernel`
2. ਸਹੀ Python ਵਰਜਨ ਚੁਣੋ
3. ਜੇ kernel ਗਾਇਬ ਹੈ, ਤਾਂ ਇਸਨੂੰ ਬਣਾਓ:
```bash
python -m ipykernel install --user --name=ml-env
```

**ਸਮੱਸਿਆ**: Kernel ਸ਼ੁਰੂ ਨਹੀਂ ਹੋ ਰਿਹਾ

**ਹੱਲ**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Cell ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: ਸੈੱਲ ਚੱਲ ਰਹੇ ਹਨ ਪਰ ਆਉਟਪੁੱਟ ਨਹੀਂ ਦਿਖਾ ਰਹੇ

**ਹੱਲ**:
1. ਜਾਂਚੋ ਕਿ ਸੈੱਲ ਅਜੇ ਵੀ ਚੱਲ ਰਿਹਾ ਹੈ (ਇਸ ਲਈ `[*]` ਇੰਡੀਕੇਟਰ ਵੇਖੋ)
2. Kernel ਨੂੰ ਰੀਸਟਾਰਟ ਕਰੋ ਅਤੇ ਸਾਰੇ ਸੈੱਲ ਚਲਾਓ: `Kernel → Restart & Run All`
3. ਬ੍ਰਾਊਜ਼ਰ ਕਨਸੋਲ ਵਿੱਚ JavaScript ਗਲਤੀਆਂ ਦੀ ਜਾਂਚ ਕਰੋ (F12)

**ਸਮੱਸਿਆ**: ਸੈੱਲ ਨਹੀਂ ਚੱਲ ਰਹੇ - "Run" 'ਤੇ ਕਲਿਕ ਕਰਨ 'ਤੇ ਕੋਈ ਜਵਾਬ ਨਹੀਂ

**ਹੱਲ**:
1. ਜਾਂਚੋ ਕਿ Jupyter ਸਰਵਰ ਅਜੇ ਵੀ ਟਰਮੀਨਲ ਵਿੱਚ ਚੱਲ ਰਿਹਾ ਹੈ
2. ਬ੍ਰਾਊਜ਼ਰ ਪੇਜ ਨੂੰ ਰੀਫ੍ਰੈਸ਼ ਕਰੋ
3. Notebook ਨੂੰ ਬੰਦ ਕਰੋ ਅਤੇ ਮੁੜ ਖੋਲ੍ਹੋ
4. Jupyter ਸਰਵਰ ਨੂੰ ਰੀਸਟਾਰਟ ਕਰੋ

---

## Python ਪੈਕੇਜ ਸਮੱਸਿਆਵਾਂ

### Import Errors

**ਸਮੱਸਿਆ**: `ModuleNotFoundError: No module named 'sklearn'`

**ਹੱਲ**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**ਸਮੱਸਿਆ**: `ImportError: cannot import name 'X' from 'sklearn'`

**ਹੱਲ**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### ਵਰਜਨ ਟਕਰਾਅ

**ਸਮੱਸਿਆ**: ਪੈਕੇਜ ਵਰਜਨ ਅਸੰਗਤਤਾ ਗਲਤੀਆਂ

**ਹੱਲ**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**ਸਮੱਸਿਆ**: `pip install` ਅਧਿਕਾਰ ਗਲਤੀਆਂ ਨਾਲ ਫੇਲ੍ਹ ਹੋ ਜਾਂਦਾ ਹੈ

**ਹੱਲ**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### ਡਾਟਾ ਲੋਡ ਕਰਨ ਦੀ ਸਮੱਸਿਆ

**ਸਮੱਸਿਆ**: CSV ਫਾਈਲਾਂ ਲੋਡ ਕਰਦੇ ਸਮੇਂ `FileNotFoundError`

**ਹੱਲ**:
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

## R Environment ਸਮੱਸਿਆਵਾਂ

### ਪੈਕੇਜ ਇੰਸਟਾਲੇਸ਼ਨ

**ਸਮੱਸਿਆ**: ਪੈਕੇਜ ਇੰਸਟਾਲੇਸ਼ਨ ਕੰਪਾਇਲੇਸ਼ਨ ਗਲਤੀਆਂ ਨਾਲ ਫੇਲ੍ਹ ਹੋ ਜਾਂਦਾ ਹੈ

**ਹੱਲ**:
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

**ਸਮੱਸਿਆ**: `tidyverse` ਇੰਸਟਾਲ ਨਹੀਂ ਹੋ ਰਿਹਾ

**ਹੱਲ**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: RMarkdown ਰੈਂਡਰ ਨਹੀਂ ਹੋ ਰਿਹਾ

**ਹੱਲ**:
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

## ਕੁਇਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਸਮੱਸਿਆਵਾਂ

### ਬਿਲਡ ਅਤੇ ਇੰਸਟਾਲੇਸ਼ਨ

**ਸਮੱਸਿਆ**: `npm install` ਫੇਲ੍ਹ ਹੋ ਜਾਂਦਾ ਹੈ

**ਹੱਲ**:
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

**ਸਮੱਸਿਆ**: Port 8080 ਪਹਿਲਾਂ ਹੀ ਵਰਤ ਵਿੱਚ ਹੈ

**ਹੱਲ**:
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

### ਬਿਲਡ ਗਲਤੀਆਂ

**ਸਮੱਸਿਆ**: `npm run build` ਫੇਲ੍ਹ ਹੋ ਜਾਂਦਾ ਹੈ

**ਹੱਲ**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**ਸਮੱਸਿਆ**: Linting ਗਲਤੀਆਂ ਬਿਲਡ ਨੂੰ ਰੋਕ ਰਹੀਆਂ ਹਨ

**ਹੱਲ**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## ਡਾਟਾ ਅਤੇ ਫਾਈਲ ਪਾਥ ਸਮੱਸਿਆਵਾਂ

### ਪਾਥ ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: Notebook ਚਲਾਉਣ ਸਮੇਂ ਡਾਟਾ ਫਾਈਲਾਂ ਨਹੀਂ ਮਿਲ ਰਹੀਆਂ

**ਹੱਲ**:
1. **ਹਮੇਸ਼ਾ Notebook ਨੂੰ ਇਸਦੇ ਸਮੇਟਣ ਵਾਲੇ ਡਾਇਰੈਕਟਰੀ ਤੋਂ ਚਲਾਓ**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **ਕੋਡ ਵਿੱਚ ਰਿਸ਼ਤੇਦਾਰ ਪਾਥ ਦੀ ਜਾਂਚ ਕਰੋ**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **ਜਰੂਰਤ ਪੈਣ 'ਤੇ ਅਬਸੋਲਿਊਟ ਪਾਥ ਦੀ ਵਰਤੋਂ ਕਰੋ**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### ਗੁੰਮ ਹੋਈ ਡਾਟਾ ਫਾਈਲਾਂ

**ਸਮੱਸਿਆ**: ਡਾਟਾਸੈਟ ਫਾਈਲਾਂ ਗੁੰਮ ਹਨ

**ਹੱਲ**:
1. ਜਾਂਚੋ ਕਿ ਡਾਟਾ ਰਿਪੋਜ਼ਟਰੀ ਵਿੱਚ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ - ਜ਼ਿਆਦਾਤਰ ਡਾਟਾਸੈਟ ਸ਼ਾਮਲ ਹਨ।
2. ਕੁਝ ਪਾਠਾਂ ਨੂੰ ਡਾਟਾ ਡਾਊਨਲੋਡ ਕਰਨ ਦੀ ਲੋੜ ਹੋ ਸਕਦੀ ਹੈ - ਪਾਠ README ਦੀ ਜਾਂਚ ਕਰੋ।
3. ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਤੁਸੀਂ ਤਾਜ਼ਾ ਬਦਲਾਅ ਖਿੱਚੇ ਹਨ:
   ```bash
   git pull origin main
   ```

---

## ਆਮ ਗਲਤੀ ਸੁਨੇਹੇ

### ਮੈਮਰੀ ਗਲਤੀਆਂ

**ਗਲਤੀ**: `MemoryError` ਜਾਂ kernel ਡਾਟਾ ਪ੍ਰੋਸੈਸ ਕਰਦੇ ਸਮੇਂ ਮਰ ਜਾਂਦਾ ਹੈ

**ਹੱਲ**:
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

### Convergence ਚੇਤਾਵਨੀਆਂ

**ਚੇਤਾਵਨੀ**: `ConvergenceWarning: Maximum number of iterations reached`

**ਹੱਲ**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### ਪਲਾਟਿੰਗ ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: Jupyter ਵਿੱਚ ਪਲਾਟ ਨਹੀਂ ਦਿਖ ਰਹੇ

**ਹੱਲ**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**ਸਮੱਸਿਆ**: Seaborn ਪਲਾਟ ਵੱਖਰੇ ਦਿਖ ਰਹੇ ਹਨ ਜਾਂ ਗਲਤੀਆਂ ਦਿਖਾ ਰਹੇ ਹਨ

**ਹੱਲ**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding ਗਲਤੀਆਂ

**ਸਮੱਸਿਆ**: ਫਾਈਲਾਂ ਪੜ੍ਹਦੇ ਸਮੇਂ `UnicodeDecodeError`

**ਹੱਲ**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## ਕਾਰਗੁਜ਼ਾਰੀ ਸਮੱਸਿਆਵਾਂ

### Notebook ਚਲਾਉਣ ਵਿੱਚ ਧੀਮਾਪਨ

**ਸਮੱਸਿਆ**: Notebook ਚਲਾਉਣ ਵਿੱਚ ਬਹੁਤ ਧੀਮੇ ਹਨ

**ਹੱਲ**:
1. **ਮੈਮਰੀ ਖਾਲੀ ਕਰਨ ਲਈ Kernel ਨੂੰ ਰੀਸਟਾਰਟ ਕਰੋ**: `Kernel → Restart`
2. **ਵਰਤੋਂ ਵਿੱਚ ਨਾ ਆ ਰਹੇ Notebook ਬੰਦ ਕਰੋ** ਰਿਸੋਰਸ ਖਾਲੀ ਕਰਨ ਲਈ।
3. **ਟੈਸਟਿੰਗ ਲਈ ਛੋਟੇ ਡਾਟਾ ਸੈਂਪਲ ਦੀ ਵਰਤੋਂ ਕਰੋ**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **ਆਪਣੇ ਕੋਡ ਨੂੰ ਪ੍ਰੋਫਾਈਲ ਕਰੋ** ਬੋਤਲਨੈਕਸ ਲੱਭਣ ਲਈ:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### ਉੱਚ ਮੈਮਰੀ ਵਰਤੋਂ

**ਸਮੱਸਿਆ**: ਸਿਸਟਮ ਮੈਮਰੀ ਖਤਮ ਹੋ ਰਹੀ ਹੈ

**ਹੱਲ**:
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

## Environment ਅਤੇ Configuration

### Virtual Environment ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: Virtual environment ਐਕਟੀਵੇਟ ਨਹੀਂ ਹੋ ਰਿਹਾ

**ਹੱਲ**:
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

**ਸਮੱਸਿਆ**: ਪੈਕੇਜ ਇੰਸਟਾਲ ਹੋਏ ਹਨ ਪਰ Notebook ਵਿੱਚ ਨਹੀਂ ਮਿਲ ਰਹੇ

**ਹੱਲ**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git ਸਮੱਸਿਆਵਾਂ

**ਸਮੱਸਿਆ**: ਤਾਜ਼ਾ ਬਦਲਾਅ ਖਿੱਚ ਨਹੀਂ ਸਕਦੇ - ਮਰਜ ਟਕਰਾਅ

**ਹੱਲ**:
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

### VS Code ਇੰਟੀਗ੍ਰੇਸ਼ਨ

**ਸਮੱਸਿਆ**: Jupyter Notebook VS Code ਵਿੱਚ ਨਹੀਂ ਖੁਲ ਰਹੇ

**ਹੱਲ**:
1. VS Code ਵਿੱਚ Python ਐਕਸਟੈਂਸ਼ਨ ਇੰਸਟਾਲ ਕਰੋ।
2. VS Code ਵਿੱਚ Jupyter ਐਕਸਟੈਂਸ਼ਨ ਇੰਸਟਾਲ ਕਰੋ।
3. ਸਹੀ Python interpreter ਚੁਣੋ: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code ਨੂੰ ਰੀਸਟਾਰਟ ਕਰੋ।

---

## ਵਾਧੂ ਸਰੋਤ

- **Discord Discussions**: [#ml-for-beginners ਚੈਨਲ ਵਿੱਚ ਪ੍ਰਸ਼ਨ ਪੁੱਛੋ ਅਤੇ ਹੱਲ ਸਾਂਝੇ ਕਰੋ](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners ਮੋਡਿਊਲ](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **ਵੀਡੀਓ ਟਿਊਟੋਰਿਅਲ**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [ਬੱਗ ਰਿਪੋਰਟ ਕਰੋ](https://github.com/microsoft/ML-For-Beginners/issues)

---

## ਅਜੇ ਵੀ ਸਮੱਸਿਆਵਾਂ ਹਨ?

ਜੇ ਤੁਸੀਂ ਉਪਰੋਕਤ ਹੱਲਾਂ ਦੀ ਕੋਸ਼ਿਸ਼ ਕੀਤੀ ਹੈ ਅਤੇ ਅਜੇ ਵੀ ਸਮੱਸਿਆਵਾਂ ਦਾ ਸਾਹਮਣਾ ਕਰ ਰਹੇ ਹੋ:

1. **ਮੌਜੂਦਾ ਮੁੱਦਿਆਂ ਦੀ ਖੋਜ ਕਰੋ**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord ਵਿੱਚ ਚਰਚਾ ਦੀ ਜਾਂਚ ਕਰੋ**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **ਨਵਾਂ ਮੁੱਦਾ ਖੋਲ੍ਹੋ**: ਸ਼ਾਮਲ ਕਰੋ:
   - ਤੁਹਾਡਾ ਓਪਰੇਟਿੰਗ ਸਿਸਟਮ ਅਤੇ ਵਰਜਨ
   - Python/R ਵਰਜਨ
   - ਗਲਤੀ ਸੁਨੇਹਾ (ਪੂਰਾ ਟ੍ਰੇਸਬੈਕ)
   - ਸਮੱਸਿਆ ਨੂੰ ਦੁਹਰਾਉਣ ਦੇ ਕਦਮ
   - ਜੋ ਤੁਸੀਂ ਪਹਿਲਾਂ ਹੀ ਕੋਸ਼ਿਸ਼ ਕੀਤੀ ਹੈ

ਅਸੀਂ ਮਦਦ ਕਰਨ ਲਈ ਇੱਥੇ ਹਾਂ! 🚀

---

**ਅਸਵੀਕਰਤਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦਾ ਯਤਨ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁੱਤੀਆਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਮੌਜੂਦ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਪ੍ਰਮਾਣਿਕ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੇ ਪ੍ਰਯੋਗ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।