<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:37:28+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "fa"
}
-->
# راهنمای رفع مشکلات

این راهنما به شما کمک می‌کند مشکلات رایج هنگام کار با دوره آموزشی یادگیری ماشین برای مبتدیان را حل کنید. اگر راه‌حلی در اینجا پیدا نکردید، لطفاً به [بحث‌های دیسکورد](https://aka.ms/foundry/discord) مراجعه کنید یا [یک مشکل جدید باز کنید](https://github.com/microsoft/ML-For-Beginners/issues).

## فهرست مطالب

- [مشکلات نصب](../..)
- [مشکلات Jupyter Notebook](../..)
- [مشکلات بسته‌های Python](../..)
- [مشکلات محیط R](../..)
- [مشکلات اپلیکیشن آزمون](../..)
- [مشکلات داده و مسیر فایل](../..)
- [پیام‌های خطای رایج](../..)
- [مشکلات عملکرد](../..)
- [محیط و تنظیمات](../..)

---

## مشکلات نصب

### نصب Python

**مشکل**: `python: command not found`

**راه‌حل**:
1. Python نسخه 3.8 یا بالاتر را از [python.org](https://www.python.org/downloads/) نصب کنید.
2. نصب را تأیید کنید: `python --version` یا `python3 --version`
3. در macOS/Linux ممکن است نیاز باشد از `python3` به جای `python` استفاده کنید.

**مشکل**: نسخه‌های متعدد Python باعث ایجاد تداخل می‌شوند.

**راه‌حل**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### نصب Jupyter

**مشکل**: `jupyter: command not found`

**راه‌حل**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**مشکل**: Jupyter در مرورگر باز نمی‌شود.

**راه‌حل**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### نصب R

**مشکل**: بسته‌های R نصب نمی‌شوند.

**راه‌حل**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**مشکل**: IRkernel در Jupyter موجود نیست.

**راه‌حل**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## مشکلات Jupyter Notebook

### مشکلات Kernel

**مشکل**: Kernel مدام از کار می‌افتد یا دوباره راه‌اندازی می‌شود.

**راه‌حل**:
1. Kernel را مجدداً راه‌اندازی کنید: `Kernel → Restart`
2. خروجی را پاک کرده و Kernel را مجدداً راه‌اندازی کنید: `Kernel → Restart & Clear Output`
3. مشکلات حافظه را بررسی کنید (به [مشکلات عملکرد](../..) مراجعه کنید).
4. سلول‌ها را به صورت جداگانه اجرا کنید تا کد مشکل‌دار را شناسایی کنید.

**مشکل**: Kernel اشتباه انتخاب شده است.

**راه‌حل**:
1. Kernel فعلی را بررسی کنید: `Kernel → Change Kernel`
2. نسخه صحیح Python را انتخاب کنید.
3. اگر Kernel موجود نیست، آن را ایجاد کنید:
```bash
python -m ipykernel install --user --name=ml-env
```

**مشکل**: Kernel شروع نمی‌شود.

**راه‌حل**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### مشکلات سلول‌های Notebook

**مشکل**: سلول‌ها اجرا می‌شوند اما خروجی نشان داده نمی‌شود.

**راه‌حل**:
1. بررسی کنید آیا سلول هنوز در حال اجرا است (به دنبال نشانگر `[*]` باشید).
2. Kernel را مجدداً راه‌اندازی کرده و همه سلول‌ها را اجرا کنید: `Kernel → Restart & Run All`
3. کنسول مرورگر را برای خطاهای JavaScript بررسی کنید (F12).

**مشکل**: نمی‌توان سلول‌ها را اجرا کرد - هنگام کلیک روی "Run" هیچ پاسخی دریافت نمی‌شود.

**راه‌حل**:
1. بررسی کنید آیا سرور Jupyter هنوز در ترمینال در حال اجرا است.
2. صفحه مرورگر را تازه کنید.
3. Notebook را ببندید و دوباره باز کنید.
4. سرور Jupyter را مجدداً راه‌اندازی کنید.

---

## مشکلات بسته‌های Python

### خطاهای Import

**مشکل**: `ModuleNotFoundError: No module named 'sklearn'`

**راه‌حل**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**مشکل**: `ImportError: cannot import name 'X' from 'sklearn'`

**راه‌حل**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### تداخل نسخه‌ها

**مشکل**: خطاهای ناسازگاری نسخه بسته‌ها

**راه‌حل**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**مشکل**: `pip install` با خطاهای دسترسی شکست می‌خورد.

**راه‌حل**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### مشکلات بارگذاری داده‌ها

**مشکل**: `FileNotFoundError` هنگام بارگذاری فایل‌های CSV

**راه‌حل**:
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

## مشکلات محیط R

### نصب بسته‌ها

**مشکل**: نصب بسته‌ها با خطاهای کامپایل شکست می‌خورد.

**راه‌حل**:
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

**مشکل**: `tidyverse` نصب نمی‌شود.

**راه‌حل**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### مشکلات RMarkdown

**مشکل**: RMarkdown رندر نمی‌شود.

**راه‌حل**:
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

## مشکلات اپلیکیشن آزمون

### ساخت و نصب

**مشکل**: `npm install` شکست می‌خورد.

**راه‌حل**:
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

**مشکل**: پورت 8080 قبلاً استفاده شده است.

**راه‌حل**:
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

### خطاهای ساخت

**مشکل**: `npm run build` شکست می‌خورد.

**راه‌حل**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**مشکل**: خطاهای linting مانع ساخت می‌شوند.

**راه‌حل**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## مشکلات داده و مسیر فایل

### مشکلات مسیر

**مشکل**: فایل‌های داده هنگام اجرای Notebook‌ها پیدا نمی‌شوند.

**راه‌حل**:
1. **همیشه Notebook‌ها را از دایرکتوری حاوی آن‌ها اجرا کنید.**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **مسیرهای نسبی در کد را بررسی کنید.**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **در صورت نیاز از مسیرهای مطلق استفاده کنید.**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### فایل‌های داده گم شده

**مشکل**: فایل‌های مجموعه داده گم شده‌اند.

**راه‌حل**:
1. بررسی کنید آیا داده‌ها باید در مخزن باشند - بیشتر مجموعه داده‌ها شامل هستند.
2. برخی درس‌ها ممکن است نیاز به دانلود داده داشته باشند - README درس را بررسی کنید.
3. مطمئن شوید آخرین تغییرات را کشیده‌اید:
   ```bash
   git pull origin main
   ```

---

## پیام‌های خطای رایج

### خطاهای حافظه

**خطا**: `MemoryError` یا Kernel هنگام پردازش داده‌ها از کار می‌افتد.

**راه‌حل**:
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

### هشدارهای همگرایی

**هشدار**: `ConvergenceWarning: Maximum number of iterations reached`

**راه‌حل**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### مشکلات رسم نمودار

**مشکل**: نمودارها در Jupyter نمایش داده نمی‌شوند.

**راه‌حل**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**مشکل**: نمودارهای Seaborn متفاوت به نظر می‌رسند یا خطا می‌دهند.

**راه‌حل**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### خطاهای Unicode/Encoding

**مشکل**: `UnicodeDecodeError` هنگام خواندن فایل‌ها

**راه‌حل**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## مشکلات عملکرد

### اجرای کند Notebook‌ها

**مشکل**: Notebook‌ها بسیار کند اجرا می‌شوند.

**راه‌حل**:
1. **Kernel را برای آزاد کردن حافظه مجدداً راه‌اندازی کنید**: `Kernel → Restart`
2. **Notebook‌های استفاده نشده را ببندید** تا منابع آزاد شوند.
3. **از نمونه‌های داده کوچک‌تر برای آزمایش استفاده کنید**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **کد خود را پروفایل کنید** تا نقاط ضعف را پیدا کنید:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### استفاده زیاد از حافظه

**مشکل**: سیستم حافظه کم می‌آورد.

**راه‌حل**:
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

## محیط و تنظیمات

### مشکلات محیط مجازی

**مشکل**: محیط مجازی فعال نمی‌شود.

**راه‌حل**:
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

**مشکل**: بسته‌ها نصب شده‌اند اما در Notebook پیدا نمی‌شوند.

**راه‌حل**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### مشکلات Git

**مشکل**: نمی‌توان آخرین تغییرات را کشید - تداخل‌های ادغام

**راه‌حل**:
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

### یکپارچگی VS Code

**مشکل**: Notebook‌های Jupyter در VS Code باز نمی‌شوند.

**راه‌حل**:
1. افزونه Python را در VS Code نصب کنید.
2. افزونه Jupyter را در VS Code نصب کنید.
3. مفسر صحیح Python را انتخاب کنید: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code را مجدداً راه‌اندازی کنید.

---

## منابع اضافی

- **بحث‌های دیسکورد**: [سوالات بپرسید و راه‌حل‌ها را در کانال #ml-for-beginners به اشتراک بگذارید](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ماژول‌های یادگیری ماشین برای مبتدیان](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **آموزش‌های ویدیویی**: [لیست پخش یوتیوب](https://aka.ms/ml-beginners-videos)
- **ردیاب مشکلات**: [گزارش خطاها](https://github.com/microsoft/ML-For-Beginners/issues)

---

## هنوز مشکل دارید؟

اگر راه‌حل‌های بالا را امتحان کرده‌اید و هنوز مشکل دارید:

1. **مشکلات موجود را جستجو کنید**: [مشکلات GitHub](https://github.com/microsoft/ML-For-Beginners/issues)
2. **بحث‌ها را در دیسکورد بررسی کنید**: [بحث‌های دیسکورد](https://aka.ms/foundry/discord)
3. **یک مشکل جدید باز کنید**: شامل موارد زیر باشد:
   - سیستم عامل و نسخه شما
   - نسخه Python/R
   - پیام خطا (traceback کامل)
   - مراحل بازتولید مشکل
   - آنچه قبلاً امتحان کرده‌اید

ما اینجا هستیم تا کمک کنیم! 🚀

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما تلاش می‌کنیم دقت را حفظ کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، توصیه می‌شود از ترجمه انسانی حرفه‌ای استفاده کنید. ما مسئولیتی در قبال سوء تفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.