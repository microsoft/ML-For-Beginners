<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:56:43+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "my"
}
-->
# ပြဿနာဖြေရှင်းလမ်းညွှန်

ဤလမ်းညွှန်သည် Machine Learning for Beginners သင်ခန်းစာများနှင့်အလုပ်လုပ်နေစဉ်တွင် ကြုံတွေ့ရသော ပုံမှန်ပြဿနာများကို ဖြေရှင်းရန် ကူညီပေးပါသည်။ ဤနေရာတွင် ဖြေရှင်းချက်မတွေ့ပါက [Discord Discussions](https://aka.ms/foundry/discord) သို့မဟုတ် [open an issue](https://github.com/microsoft/ML-For-Beginners/issues) တွင် စစ်ဆေးပါ။

## အကြောင်းအရာများဇယား

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

**ပြဿနာ**: `python: command not found`

**ဖြေရှင်းချက်**:
1. [python.org](https://www.python.org/downloads/) မှ Python 3.8 သို့မဟုတ် အထက်ကို ထည့်သွင်းပါ။
2. ထည့်သွင်းမှုကို စစ်ဆေးပါ: `python --version` သို့မဟုတ် `python3 --version`
3. macOS/Linux တွင် `python` အစား `python3` ကို အသုံးပြုရန် လိုအပ်နိုင်သည်။

**ပြဿနာ**: Python ဗားရှင်းများစွာကြောင့် အကျိုးသက်ရောက်မှုရှိခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: `jupyter: command not found`

**ဖြေရှင်းချက်**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**ပြဿနာ**: Jupyter ကို browser တွင် မဖွင့်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R Installation

**ပြဿနာ**: R packages မထည့်သွင်းနိုင်ခြင်း

**ဖြေရှင်းချက်**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**ပြဿနာ**: IRkernel ကို Jupyter တွင် မရရှိနိုင်ခြင်း

**ဖြေရှင်းချက်**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook Issues

### Kernel Issues

**ပြဿနာ**: Kernel သေသွားခြင်း သို့မဟုတ် ပြန်စတင်ခြင်း

**ဖြေရှင်းချက်**:
1. Kernel ကို ပြန်စတင်ပါ: `Kernel → Restart`
2. Output ကို ရှင်းပြီး ပြန်စတင်ပါ: `Kernel → Restart & Clear Output`
3. memory ပြဿနာများကို စစ်ဆေးပါ ([Performance Issues](../..) ကို ကြည့်ပါ)
4. ပြဿနာရှိ code ကို ရှာဖွေရန် cell များကို တစ်ခုချင်းစီ run လုပ်ပါ

**ပြဿနာ**: မှားသော Python kernel ကို ရွေးချယ်ထားခြင်း

**ဖြေရှင်းချက်**:
1. လက်ရှိ kernel ကို စစ်ဆေးပါ: `Kernel → Change Kernel`
2. မှန်ကန်သော Python ဗားရှင်းကို ရွေးချယ်ပါ
3. Kernel မရှိပါက ထည့်သွင်းပါ:
```bash
python -m ipykernel install --user --name=ml-env
```

**ပြဿနာ**: Kernel မစတင်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Cell Issues

**ပြဿနာ**: Cell များ run လုပ်နေသော်လည်း output မပေါ်ခြင်း

**ဖြေရှင်းချက်**:
1. Cell သည် run လုပ်နေဆဲဖြစ်ကြောင်း စစ်ဆေးပါ (`[*]` indicator ကို ကြည့်ပါ)
2. Kernel ကို ပြန်စတင်ပြီး Cell အားလုံးကို run လုပ်ပါ: `Kernel → Restart & Run All`
3. Browser console တွင် JavaScript error များကို စစ်ဆေးပါ (F12)

**ပြဿနာ**: "Run" ကို click လုပ်သောအခါ Cell မ run လုပ်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
1. Jupyter server သည် terminal တွင် run လုပ်နေဆဲဖြစ်ကြောင်း စစ်ဆေးပါ
2. Browser page ကို refresh လုပ်ပါ
3. Notebook ကို ပိတ်ပြီး ပြန်ဖွင့်ပါ
4. Jupyter server ကို ပြန်စတင်ပါ

---

## Python Package Issues

### Import Errors

**ပြဿနာ**: `ModuleNotFoundError: No module named 'sklearn'`

**ဖြေရှင်းချက်**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**ပြဿနာ**: `ImportError: cannot import name 'X' from 'sklearn'`

**ဖြေရှင်းချက်**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Version Conflicts

**ပြဿနာ**: Package version မတူညီမှု error များ

**ဖြေရှင်းချက်**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**ပြဿနာ**: `pip install` သည် permission error များဖြင့် မအောင်မြင်ခြင်း

**ဖြေရှင်းချက်**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Data Loading Issues

**ပြဿနာ**: CSV ဖိုင်များကို load လုပ်သောအခါ `FileNotFoundError`

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Package installation သည် compilation error များဖြင့် မအောင်မြင်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: `tidyverse` မထည့်သွင်းနိုင်ခြင်း

**ဖြေရှင်းချက်**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown Issues

**ပြဿနာ**: RMarkdown မ render လုပ်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: `npm install` မအောင်မြင်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Port 8080 သည် အသုံးပြုနေပြီးဖြစ်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: `npm run build` မအောင်မြင်ခြင်း

**ဖြေရှင်းချက်**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**ပြဿနာ**: Linting error များကြောင့် build မလုပ်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data and File Path Issues

### Path Problems

**ပြဿနာ**: Notebook များ run လုပ်သောအခါ data ဖိုင်များ မတွေ့ရှိခြင်း

**ဖြေရှင်းချက်**:
1. **Notebook များကို ၎င်းတို့ရှိသော directory မှ run လုပ်ပါ**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Code တွင် relative path များကို စစ်ဆေးပါ**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **လိုအပ်ပါက absolute path များကို အသုံးပြုပါ**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Missing Data Files

**ပြဿနာ**: Dataset ဖိုင်များ မရှိခြင်း

**ဖြေရှင်းချက်**:
1. Data သည် repository တွင် ရှိသင့်ကြောင်း စစ်ဆေးပါ - dataset များအများစုကို ထည့်သွင်းထားသည်။
2. အချို့သော သင်ခန်းစာများတွင် data ကို download လုပ်ရန် လိုအပ်နိုင်သည် - lesson README ကို စစ်ဆေးပါ။
3. နောက်ဆုံး update များကို pull လုပ်ထားကြောင်း သေချာပါ:
   ```bash
   git pull origin main
   ```

---

## Common Error Messages

### Memory Errors

**Error**: `MemoryError` သို့မဟုတ် kernel သေသွားခြင်း

**ဖြေရှင်းချက်**:
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

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Jupyter တွင် plot မပေါ်ခြင်း

**ဖြေရှင်းချက်**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**ပြဿနာ**: Seaborn plot များကွဲပြားခြင်း သို့မဟုတ် error ပေးခြင်း

**ဖြေရှင်းချက်**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding Errors

**ပြဿနာ**: ဖိုင်များကို ဖတ်သောအခါ `UnicodeDecodeError`

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Notebook များ run လုပ်ရန် အလွန်နှေးကွေးခြင်း

**ဖြေရှင်းချက်**:
1. **Memory ကို လွတ်လပ်စေရန် Kernel ကို ပြန်စတင်ပါ**: `Kernel → Restart`
2. **အသုံးမပြုသော notebook များကို ပိတ်ပါ** memory ကို လွတ်လပ်စေရန်
3. **စမ်းသပ်ရန် data sample များကို သေးငယ်သောအတိုင်း အသုံးပြုပါ**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Code ကို profile လုပ်ပါ** bottleneck များကို ရှာဖွေရန်:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### High Memory Usage

**ပြဿနာ**: System memory မလုံလောက်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Virtual environment မ activate လုပ်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Package များကို ထည့်သွင်းထားသော်လည်း notebook တွင် မတွေ့ရှိခြင်း

**ဖြေရှင်းချက်**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git Issues

**ပြဿနာ**: နောက်ဆုံး update များကို pull လုပ်၍ မရနိုင်ခြင်း - merge conflict

**ဖြေရှင်းချက်**:
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

**ပြဿနာ**: Jupyter notebook များကို VS Code တွင် မဖွင့်နိုင်ခြင်း

**ဖြေရှင်းချက်**:
1. VS Code တွင် Python extension ကို ထည့်သွင်းပါ
2. VS Code တွင် Jupyter extension ကို ထည့်သွင်းပါ
3. မှန်ကန်သော Python interpreter ကို ရွေးချယ်ပါ: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code ကို ပြန်စတင်ပါ

---

## အပိုဆောင်းအရင်းအမြစ်များ

- **Discord Discussions**: [#ml-for-beginners channel တွင် မေးခွန်းများမေးပြီး ဖြေရှင်းချက်များမျှဝေပါ](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video Tutorials**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [Report bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## မသေချာသေးပါက?

အထက်ပါ ဖြေရှင်းချက်များကို စမ်းသပ်ပြီး ပြဿနာများ ဆက်လက်ဖြစ်ပေါ်နေပါက-

1. **ရှိပြီးသား issue များကို ရှာဖွေပါ**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord Discussions တွင် ဆွေးနွေးချက်များကို စစ်ဆေးပါ**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Issue အသစ်တစ်ခုကို ဖွင့်ပါ**: အပါအဝင်-
   - သင့် operating system နှင့် version
   - Python/R version
   - Error message (full traceback)
   - ပြဿနာကို ပြန်လည်ဖြစ်ပေါ်စေသော လုပ်ဆောင်မှုများ
   - သင်စမ်းသပ်ပြီးဖြစ်သောအရာများ

ကျွန်ုပ်တို့ကူညီပေးပါမည်! 🚀

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေသော်လည်း အလိုအလျောက် ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါရှိနိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတရ အရင်းအမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူက ဘာသာပြန်မှုကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွတ်များ သို့မဟုတ် အနားယူမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။