<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:34:37+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "en"
}
-->
# Troubleshooting Guide

This guide helps you resolve common issues when working with the Machine Learning for Beginners curriculum. If you don't find a solution here, please visit our [Discord Discussions](https://aka.ms/foundry/discord) or [open an issue](https://github.com/microsoft/ML-For-Beginners/issues).

## Table of Contents

- [Installation Issues](../..)
- [Jupyter Notebook Issues](../..)
- [Python Package Issues](../..)
- [R Environment Issues](../..)
- [Quiz Application Issues](../..)
- [Data and File Path Issues](../..)
- [Common Error Messages](../..)
- [Performance Issues](../..)
- [Environment and Configuration](../..)

---

## Installation Issues

### Python Installation

**Problem**: `python: command not found`

**Solution**:
1. Install Python 3.8 or higher from [python.org](https://www.python.org/downloads/)
2. Verify installation: `python --version` or `python3 --version`
3. On macOS/Linux, you may need to use `python3` instead of `python`

**Problem**: Multiple Python versions causing conflicts

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

**Problem**: `jupyter: command not found`

**Solution**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter won't launch in browser

**Solution**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R Installation

**Problem**: R packages won't install

**Solution**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel not available in Jupyter

**Solution**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook Issues

### Kernel Issues

**Problem**: Kernel keeps dying or restarting

**Solution**:
1. Restart the kernel: `Kernel → Restart`
2. Clear output and restart: `Kernel → Restart & Clear Output`
3. Check for memory issues (see [Performance Issues](../..))
4. Try running cells individually to identify problematic code

**Problem**: Wrong Python kernel selected

**Solution**:
1. Check current kernel: `Kernel → Change Kernel`
2. Select the correct Python version
3. If kernel is missing, create it:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Kernel won't start

**Solution**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Cell Issues

**Problem**: Cells are running but not showing output

**Solution**:
1. Check if cell is still running (look for `[*]` indicator)
2. Restart kernel and run all cells: `Kernel → Restart & Run All`
3. Check browser console for JavaScript errors (F12)

**Problem**: Can't run cells - no response when clicking "Run"

**Solution**:
1. Check if Jupyter server is still running in terminal
2. Refresh the browser page
3. Close and reopen the notebook
4. Restart Jupyter server

---

## Python Package Issues

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solution**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Version Conflicts

**Problem**: Package version incompatibility errors

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

**Problem**: `pip install` fails with permission errors

**Solution**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Data Loading Issues

**Problem**: `FileNotFoundError` when loading CSV files

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

## R Environment Issues

### Package Installation

**Problem**: Package installation fails with compilation errors

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

**Problem**: `tidyverse` won't install

**Solution**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown Issues

**Problem**: RMarkdown won't render

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

## Quiz Application Issues

### Build and Installation

**Problem**: `npm install` fails

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

**Problem**: Port 8080 already in use

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

**Problem**: `npm run build` fails

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

**Problem**: Linting errors preventing build

**Solution**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data and File Path Issues

### Path Problems

**Problem**: Data files not found when running notebooks

**Solution**:
1. **Always run notebooks from their containing directory**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Check relative paths in code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Use absolute paths if needed**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Missing Data Files

**Problem**: Dataset files are missing

**Solution**:
1. Check if data should be in the repository - most datasets are included
2. Some lessons may require downloading data - check lesson README
3. Ensure you've pulled the latest changes:
   ```bash
   git pull origin main
   ```

---

## Common Error Messages

### Memory Errors

**Error**: `MemoryError` or kernel dies when processing data

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

### Plotting Issues

**Problem**: Plots not showing in Jupyter

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

**Problem**: Seaborn plots look different or throw errors

**Solution**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding Errors

**Problem**: `UnicodeDecodeError` when reading files

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

## Performance Issues

### Slow Notebook Execution

**Problem**: Notebooks are very slow to run

**Solution**:
1. **Restart kernel to free memory**: `Kernel → Restart`
2. **Close unused notebooks** to free resources
3. **Use smaller data samples for testing**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profile your code** to find bottlenecks:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### High Memory Usage

**Problem**: System running out of memory

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

### Virtual Environment Issues

**Problem**: Virtual environment not activating

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

**Problem**: Packages installed but not found in notebook

**Solution**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git Issues

**Problem**: Can't pull latest changes - merge conflicts

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

**Problem**: Jupyter notebooks won't open in VS Code

**Solution**:
1. Install Python extension in VS Code
2. Install Jupyter extension in VS Code
3. Select correct Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Restart VS Code

---

## Additional Resources

- **Discord Discussions**: [Ask questions and share solutions in the #ml-for-beginners channel](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video Tutorials**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [Report bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Still Having Issues?

If you've tried the solutions above and still experiencing problems:

1. **Search existing issues**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Check discussions in Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Open a new issue**: Include:
   - Your operating system and version
   - Python/R version
   - Error message (full traceback)
   - Steps to reproduce the problem
   - What you've already tried

We're here to help! 🚀

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.