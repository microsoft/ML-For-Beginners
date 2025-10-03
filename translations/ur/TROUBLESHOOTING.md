<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:37:51+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ur"
}
-->
# خرابیوں کا سراغ لگانے کی گائیڈ

یہ گائیڈ آپ کو مشین لرننگ فار بیگنرز نصاب کے ساتھ کام کرتے وقت عام مسائل حل کرنے میں مدد دیتی ہے۔ اگر آپ کو یہاں حل نہ ملے تو براہ کرم ہمارے [Discord Discussions](https://aka.ms/foundry/discord) کو چیک کریں یا [ایک مسئلہ درج کریں](https://github.com/microsoft/ML-For-Beginners/issues)۔

## فہرست

- [انسٹالیشن کے مسائل](../..)
- [جوپیٹر نوٹ بک کے مسائل](../..)
- [پائتھون پیکج کے مسائل](../..)
- [آر ماحول کے مسائل](../..)
- [کوئز ایپلیکیشن کے مسائل](../..)
- [ڈیٹا اور فائل پاتھ کے مسائل](../..)
- [عام ایرر میسجز](../..)
- [کارکردگی کے مسائل](../..)
- [ماحول اور کنفیگریشن](../..)

---

## انسٹالیشن کے مسائل

### پائتھون انسٹالیشن

**مسئلہ**: `python: command not found`

**حل**:
1. [python.org](https://www.python.org/downloads/) سے Python 3.8 یا اس سے اوپر کا ورژن انسٹال کریں۔
2. انسٹالیشن کی تصدیق کریں: `python --version` یا `python3 --version`
3. macOS/Linux پر، آپ کو `python` کی بجائے `python3` استعمال کرنا پڑ سکتا ہے۔

**مسئلہ**: متعدد پائتھون ورژنز کے تنازعات

**حل**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### جوپیٹر انسٹالیشن

**مسئلہ**: `jupyter: command not found`

**حل**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**مسئلہ**: جوپیٹر براؤزر میں لانچ نہیں ہو رہا

**حل**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### آر انسٹالیشن

**مسئلہ**: آر پیکجز انسٹال نہیں ہو رہے

**حل**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**مسئلہ**: IRkernel جوپیٹر میں دستیاب نہیں

**حل**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## جوپیٹر نوٹ بک کے مسائل

### کرنل کے مسائل

**مسئلہ**: کرنل بار بار بند یا ری اسٹارٹ ہو رہا ہے

**حل**:
1. کرنل کو ری اسٹارٹ کریں: `Kernel → Restart`
2. آؤٹ پٹ صاف کریں اور ری اسٹارٹ کریں: `Kernel → Restart & Clear Output`
3. میموری کے مسائل چیک کریں (دیکھیں [کارکردگی کے مسائل](../..))
4. مسئلہ پیدا کرنے والے کوڈ کو شناخت کرنے کے لیے سیلز کو الگ الگ چلائیں۔

**مسئلہ**: غلط پائتھون کرنل منتخب کیا گیا

**حل**:
1. موجودہ کرنل چیک کریں: `Kernel → Change Kernel`
2. درست پائتھون ورژن منتخب کریں۔
3. اگر کرنل غائب ہے، تو اسے بنائیں:
```bash
python -m ipykernel install --user --name=ml-env
```

**مسئلہ**: کرنل شروع نہیں ہو رہا

**حل**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### نوٹ بک سیل کے مسائل

**مسئلہ**: سیلز چل رہے ہیں لیکن آؤٹ پٹ نہیں دکھا رہے

**حل**:
1. چیک کریں کہ سیل ابھی بھی چل رہا ہے (دیکھیں `[*]` انڈیکیٹر)۔
2. کرنل کو ری اسٹارٹ کریں اور تمام سیلز چلائیں: `Kernel → Restart & Run All`
3. براؤزر کنسول میں جاوا اسکرپٹ ایررز چیک کریں (F12)۔

**مسئلہ**: سیلز نہیں چل رہے - "Run" پر کلک کرنے پر کوئی جواب نہیں

**حل**:
1. چیک کریں کہ جوپیٹر سرور ٹرمینل میں چل رہا ہے۔
2. براؤزر پیج کو ریفریش کریں۔
3. نوٹ بک کو بند کریں اور دوبارہ کھولیں۔
4. جوپیٹر سرور کو ری اسٹارٹ کریں۔

---

## پائتھون پیکج کے مسائل

### امپورٹ ایررز

**مسئلہ**: `ModuleNotFoundError: No module named 'sklearn'`

**حل**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**مسئلہ**: `ImportError: cannot import name 'X' from 'sklearn'`

**حل**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### ورژن تنازعات

**مسئلہ**: پیکج ورژن کی عدم مطابقت کے ایررز

**حل**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**مسئلہ**: `pip install` اجازت کے ایررز کے ساتھ ناکام ہو رہا ہے

**حل**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### ڈیٹا لوڈنگ کے مسائل

**مسئلہ**: CSV فائلز لوڈ کرتے وقت `FileNotFoundError`

**حل**:
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

## آر ماحول کے مسائل

### پیکج انسٹالیشن

**مسئلہ**: پیکج انسٹالیشن کمپائلیشن ایررز کے ساتھ ناکام ہو رہی ہے

**حل**:
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

**مسئلہ**: `tidyverse` انسٹال نہیں ہو رہا

**حل**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### آر مارک ڈاؤن کے مسائل

**مسئلہ**: آر مارک ڈاؤن رینڈر نہیں ہو رہا

**حل**:
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

## کوئز ایپلیکیشن کے مسائل

### بلڈ اور انسٹالیشن

**مسئلہ**: `npm install` ناکام ہو رہا ہے

**حل**:
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

**مسئلہ**: پورٹ 8080 پہلے سے استعمال میں ہے

**حل**:
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

### بلڈ ایررز

**مسئلہ**: `npm run build` ناکام ہو رہا ہے

**حل**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**مسئلہ**: لنٹنگ ایررز بلڈ کو روک رہے ہیں

**حل**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## ڈیٹا اور فائل پاتھ کے مسائل

### پاتھ کے مسائل

**مسئلہ**: نوٹ بکس چلاتے وقت ڈیٹا فائلز نہیں مل رہیں

**حل**:
1. **ہمیشہ نوٹ بکس کو ان کی موجودہ ڈائریکٹری سے چلائیں**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **کوڈ میں رشتہ دار پاتھز چیک کریں**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **ضرورت پڑنے پر مطلق پاتھز استعمال کریں**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### گمشدہ ڈیٹا فائلز

**مسئلہ**: ڈیٹاسیٹ فائلز غائب ہیں

**حل**:
1. چیک کریں کہ آیا ڈیٹا ریپوزٹری میں ہونا چاہیے - زیادہ تر ڈیٹاسیٹس شامل ہیں۔
2. کچھ اسباق میں ڈیٹا ڈاؤن لوڈ کرنے کی ضرورت ہو سکتی ہے - سبق کے README کو چیک کریں۔
3. یقینی بنائیں کہ آپ نے تازہ ترین تبدیلیاں کھینچ لی ہیں:
   ```bash
   git pull origin main
   ```

---

## عام ایرر میسجز

### میموری ایررز

**ایرر**: `MemoryError` یا ڈیٹا پروسیسنگ کے دوران کرنل بند ہو جاتا ہے

**حل**:
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

### کنورجنس وارننگز

**وارننگ**: `ConvergenceWarning: Maximum number of iterations reached`

**حل**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### پلاٹنگ کے مسائل

**مسئلہ**: جوپیٹر میں پلاٹس نہیں دکھا رہے

**حل**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**مسئلہ**: سیبورن پلاٹس مختلف نظر آ رہے ہیں یا ایررز دے رہے ہیں

**حل**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### یونیکوڈ/انکوڈنگ ایررز

**مسئلہ**: فائلز پڑھتے وقت `UnicodeDecodeError`

**حل**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## کارکردگی کے مسائل

### نوٹ بک کا سست عمل

**مسئلہ**: نوٹ بکس چلانے میں بہت سست ہیں

**حل**:
1. **میموری خالی کرنے کے لیے کرنل کو ری اسٹارٹ کریں**: `Kernel → Restart`
2. **غیر استعمال شدہ نوٹ بکس بند کریں** تاکہ وسائل خالی ہوں۔
3. **ٹیسٹنگ کے لیے چھوٹے ڈیٹا سیمپلز استعمال کریں**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **اپنے کوڈ کی پروفائلنگ کریں** تاکہ کمزور پوائنٹس کا پتہ چلے:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### زیادہ میموری کا استعمال

**مسئلہ**: سسٹم میموری ختم ہو رہی ہے

**حل**:
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

## ماحول اور کنفیگریشن

### ورچوئل ماحول کے مسائل

**مسئلہ**: ورچوئل ماحول فعال نہیں ہو رہا

**حل**:
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

**مسئلہ**: پیکجز انسٹال ہیں لیکن نوٹ بک میں نہیں مل رہے

**حل**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### گٹ کے مسائل

**مسئلہ**: تازہ ترین تبدیلیاں کھینچ نہیں پا رہے - مرج تنازعات

**حل**:
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

### وی ایس کوڈ انٹیگریشن

**مسئلہ**: جوپیٹر نوٹ بکس وی ایس کوڈ میں نہیں کھل رہیں

**حل**:
1. وی ایس کوڈ میں پائتھون ایکسٹینشن انسٹال کریں۔
2. وی ایس کوڈ میں جوپیٹر ایکسٹینشن انسٹال کریں۔
3. درست پائتھون انٹرپریٹر منتخب کریں: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. وی ایس کوڈ کو ری اسٹارٹ کریں۔

---

## اضافی وسائل

- **ڈسکارڈ ڈسکشنز**: [#ml-for-beginners چینل میں سوالات پوچھیں اور حل شیئر کریں](https://aka.ms/foundry/discord)
- **مائیکروسافٹ لرن**: [ML for Beginners ماڈیولز](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **ویڈیو ٹیوٹوریلز**: [یوٹیوب پلے لسٹ](https://aka.ms/ml-beginners-videos)
- **ایشو ٹریکر**: [بگز رپورٹ کریں](https://github.com/microsoft/ML-For-Beginners/issues)

---

## اب بھی مسائل کا سامنا ہے؟

اگر آپ نے اوپر دیے گئے حل آزما لیے ہیں اور اب بھی مسائل کا سامنا کر رہے ہیں:

1. **موجودہ مسائل تلاش کریں**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **ڈسکارڈ میں ڈسکشنز چیک کریں**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **نیا مسئلہ درج کریں**: شامل کریں:
   - آپریٹنگ سسٹم اور ورژن
   - پائتھون/آر ورژن
   - ایرر میسج (مکمل ٹریس بیک)
   - مسئلہ دوبارہ پیدا کرنے کے اقدامات
   - آپ نے پہلے سے کیا آزمایا ہے

ہم مدد کے لیے حاضر ہیں! 🚀

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کی بھرپور کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔