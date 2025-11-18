<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-11-18T18:11:28+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "pcm"
}
-->
# Troubleshooting Guide

Dis guide go help you solve common wahala wey fit happen wen you dey work wit di Machine Learning for Beginners curriculum. If you no see solution here, abeg check our [Discord Discussions](https://aka.ms/foundry/discord) or [open an issue](https://github.com/microsoft/ML-For-Beginners/issues).

## Table of Contents

- [Installation Wahala](../..)
- [Jupyter Notebook Wahala](../..)
- [Python Package Wahala](../..)
- [R Environment Wahala](../..)
- [Quiz Application Wahala](../..)
- [Data and File Path Wahala](../..)
- [Common Error Messages](../..)
- [Performance Wahala](../..)
- [Environment and Configuration](../..)

---

## Installation Wahala

### Python Installation

**Wahala**: `python: command not found`

**Solution**:
1. Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. Check say Python don install: `python --version` or `python3 --version`
3. For macOS/Linux, you fit need use `python3` instead of `python`

**Wahala**: Multiple Python versions dey cause wahala

**Solution**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter Installation

**Wahala**: `jupyter: command not found`

**Solution**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Wahala**: Jupyter no wan open for browser

**Solution**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R Installation

**Wahala**: R packages no wan install

**Solution**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Wahala**: IRkernel no dey available for Jupyter

**Solution**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook Wahala

### Kernel Wahala

**Wahala**: Kernel dey die or restart anyhow

**Solution**:
1. Restart di kernel: `Kernel → Restart`
2. Clear output and restart: `Kernel → Restart & Clear Output`
3. Check memory wahala (see [Performance Wahala](../..))
4. Try run di cells one by one to see di code wey dey cause wahala

**Wahala**: Wrong Python kernel dey selected

**Solution**:
1. Check di current kernel: `Kernel → Change Kernel`
2. Select di correct Python version
3. If kernel no dey, create am:
```bash
python -m ipykernel install --user --name=ml-env
```

**Wahala**: Kernel no wan start

**Solution**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Cell Wahala

**Wahala**: Cells dey run but output no dey show

**Solution**:
1. Check if cell still dey run (look for `[*]` indicator)
2. Restart kernel and run all cells: `Kernel → Restart & Run All`
3. Check browser console for JavaScript errors (F12)

**Wahala**: Cells no dey run - no response wen you click "Run"

**Solution**:
1. Check if Jupyter server still dey run for terminal
2. Refresh di browser page
3. Close and reopen di notebook
4. Restart Jupyter server

---

## Python Package Wahala

### Import Errors

**Wahala**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Wahala**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solution**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Version Conflicts

**Wahala**: Package version dey incompatible

**Solution**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Wahala**: `pip install` dey fail wit permission wahala

**Solution**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Data Loading Wahala

**Wahala**: `FileNotFoundError` wen you dey load CSV files

**Solution**:
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

## R Environment Wahala

### Package Installation

**Wahala**: Package installation dey fail wit compilation wahala

**Solution**:
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

**Wahala**: `tidyverse` no wan install

**Solution**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown Wahala

**Wahala**: RMarkdown no wan render

**Solution**:
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

## Quiz Application Wahala

### Build and Installation

**Wahala**: `npm install` dey fail

**Solution**:
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

**Wahala**: Port 8080 don already dey use

**Solution**:
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

### Build Errors

**Wahala**: `npm run build` dey fail

**Solution**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Wahala**: Linting errors dey stop build

**Solution**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data and File Path Wahala

### Path Wahala

**Wahala**: Data files no dey wen you dey run notebooks

**Solution**:
1. **Always run notebooks from di directory wey dey contain dem**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Check di relative paths for code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Use absolute paths if e necessary**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Missing Data Files

**Wahala**: Dataset files no dey

**Solution**:
1. Check if data suppose dey di repository - most datasets dey included
2. Some lessons fit need make you download data - check lesson README
3. Make sure say you don pull di latest changes:
   ```bash
   git pull origin main
   ```

---

## Common Error Messages

### Memory Wahala

**Error**: `MemoryError` or kernel dey die wen e dey process data

**Solution**:
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

### Convergence Warnings

**Warning**: `ConvergenceWarning: Maximum number of iterations reached`

**Solution**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Plotting Wahala

**Wahala**: Plots no dey show for Jupyter

**Solution**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Wahala**: Seaborn plots dey look different or dey throw errors

**Solution**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding Wahala

**Wahala**: `UnicodeDecodeError` wen you dey read files

**Solution**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Performance Wahala

### Slow Notebook Execution

**Wahala**: Notebooks dey run very slow

**Solution**:
1. **Restart kernel to free memory**: `Kernel → Restart`
2. **Close notebooks wey you no dey use** to free resources
3. **Use smaller data samples for testing**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profile your code** to find di bottlenecks:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### High Memory Usage

**Wahala**: System dey run out of memory

**Solution**:
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

## Environment and Configuration

### Virtual Environment Wahala

**Wahala**: Virtual environment no dey activate

**Solution**:
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

**Wahala**: Packages don install but notebook no dey see dem

**Solution**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git Wahala

**Wahala**: No fit pull latest changes - merge conflicts dey

**Solution**:
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

### VS Code Integration

**Wahala**: Jupyter notebooks no wan open for VS Code

**Solution**:
1. Install Python extension for VS Code
2. Install Jupyter extension for VS Code
3. Select correct Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Restart VS Code

---

## Additional Resources

- **Discord Discussions**: [Ask questions and share solutions for di #ml-for-beginners channel](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video Tutorials**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [Report bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Still Get Wahala?

If you don try di solutions wey dey above and wahala still dey:

1. **Search existing issues**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Check discussions for Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Open new issue**: Include:
   - Your operating system and version
   - Python/R version
   - Error message (full traceback)
   - Steps wey you take to reproduce di wahala
   - Wetin you don already try

We dey here to help! 🚀

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis docu don dey translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am accurate, abeg sabi say automatic translation fit get mistake or no correct well. Di original docu for im native language na di main correct source. For important information, e go beta make professional human translator check am. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->