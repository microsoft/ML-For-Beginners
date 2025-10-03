<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:41:23+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "bn"
}
-->
# সমস্যার সমাধানের গাইড

এই গাইডটি আপনাকে Machine Learning for Beginners কারিকুলামের সাথে কাজ করার সময় সাধারণ সমস্যাগুলি সমাধান করতে সাহায্য করবে। যদি এখানে সমাধান না পান, তাহলে আমাদের [Discord Discussions](https://aka.ms/foundry/discord) বা [একটি ইস্যু খুলুন](https://github.com/microsoft/ML-For-Beginners/issues) দেখুন।

## সূচিপত্র

- [ইনস্টলেশন সমস্যা](../..)
- [Jupyter Notebook সমস্যা](../..)
- [Python প্যাকেজ সমস্যা](../..)
- [R পরিবেশ সমস্যা](../..)
- [কুইজ অ্যাপ্লিকেশন সমস্যা](../..)
- [ডেটা এবং ফাইল পাথ সমস্যা](../..)
- [সাধারণ ত্রুটি বার্তা](../..)
- [পারফরম্যান্স সমস্যা](../..)
- [পরিবেশ এবং কনফিগারেশন](../..)

---

## ইনস্টলেশন সমস্যা

### Python ইনস্টলেশন

**সমস্যা**: `python: command not found`

**সমাধান**:
1. [python.org](https://www.python.org/downloads/) থেকে Python 3.8 বা তার বেশি সংস্করণ ইনস্টল করুন।
2. ইনস্টলেশন যাচাই করুন: `python --version` বা `python3 --version`
3. macOS/Linux-এ, আপনাকে `python` এর পরিবর্তে `python3` ব্যবহার করতে হতে পারে।

**সমস্যা**: একাধিক Python সংস্করণে দ্বন্দ্ব

**সমাধান**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter ইনস্টলেশন

**সমস্যা**: `jupyter: command not found`

**সমাধান**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**সমস্যা**: Jupyter ব্রাউজারে চালু হচ্ছে না

**সমাধান**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R ইনস্টলেশন

**সমস্যা**: R প্যাকেজ ইনস্টল হচ্ছে না

**সমাধান**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**সমস্যা**: IRkernel Jupyter-এ উপলব্ধ নয়

**সমাধান**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook সমস্যা

### Kernel সমস্যা

**সমস্যা**: Kernel বারবার বন্ধ হয়ে যাচ্ছে বা পুনরায় চালু হচ্ছে

**সমাধান**:
1. Kernel পুনরায় চালু করুন: `Kernel → Restart`
2. আউটপুট মুছে পুনরায় চালু করুন: `Kernel → Restart & Clear Output`
3. মেমোরি সমস্যা পরীক্ষা করুন (দেখুন [পারফরম্যান্স সমস্যা](../..))
4. কোডের সমস্যাযুক্ত অংশ চিহ্নিত করতে পৃথক সেল চালান।

**সমস্যা**: ভুল Python kernel নির্বাচন করা হয়েছে

**সমাধান**:
1. বর্তমান kernel পরীক্ষা করুন: `Kernel → Change Kernel`
2. সঠিক Python সংস্করণ নির্বাচন করুন।
3. যদি kernel অনুপস্থিত থাকে, এটি তৈরি করুন:
```bash
python -m ipykernel install --user --name=ml-env
```

**সমস্যা**: Kernel চালু হচ্ছে না

**সমাধান**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Cell সমস্যা

**সমস্যা**: সেল চালানো হচ্ছে কিন্তু আউটপুট দেখাচ্ছে না

**সমাধান**:
1. সেল এখনও চালু আছে কিনা পরীক্ষা করুন (`[*]` নির্দেশক দেখুন)।
2. Kernel পুনরায় চালু করুন এবং সব সেল চালান: `Kernel → Restart & Run All`
3. ব্রাউজার কনসোলে JavaScript ত্রুটি পরীক্ষা করুন (F12)।

**সমস্যা**: সেল চালানো যাচ্ছে না - "Run" ক্লিক করলে কোনো প্রতিক্রিয়া নেই

**সমাধান**:
1. Jupyter সার্ভার টার্মিনালে চালু আছে কিনা পরীক্ষা করুন।
2. ব্রাউজার পেজ রিফ্রেশ করুন।
3. নোটবুক বন্ধ করে পুনরায় খুলুন।
4. Jupyter সার্ভার পুনরায় চালু করুন।

---

## Python প্যাকেজ সমস্যা

### Import Errors

**সমস্যা**: `ModuleNotFoundError: No module named 'sklearn'`

**সমাধান**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**সমস্যা**: `ImportError: cannot import name 'X' from 'sklearn'`

**সমাধান**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### সংস্করণ দ্বন্দ্ব

**সমস্যা**: প্যাকেজ সংস্করণ অসঙ্গতি ত্রুটি

**সমাধান**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**সমস্যা**: `pip install` অনুমতি ত্রুটির কারণে ব্যর্থ হচ্ছে

**সমাধান**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### ডেটা লোডিং সমস্যা

**সমস্যা**: CSV ফাইল লোড করার সময় `FileNotFoundError`

**সমাধান**:
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

## R পরিবেশ সমস্যা

### প্যাকেজ ইনস্টলেশন

**সমস্যা**: প্যাকেজ ইনস্টলেশন কম্পাইলেশন ত্রুটির কারণে ব্যর্থ হচ্ছে

**সমাধান**:
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

**সমস্যা**: `tidyverse` ইনস্টল হচ্ছে না

**সমাধান**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown সমস্যা

**সমস্যা**: RMarkdown রেন্ডার হচ্ছে না

**সমাধান**:
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

## কুইজ অ্যাপ্লিকেশন সমস্যা

### বিল্ড এবং ইনস্টলেশন

**সমস্যা**: `npm install` ব্যর্থ হচ্ছে

**সমাধান**:
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

**সমস্যা**: Port 8080 ইতিমধ্যে ব্যবহৃত হচ্ছে

**সমাধান**:
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

### বিল্ড ত্রুটি

**সমস্যা**: `npm run build` ব্যর্থ হচ্ছে

**সমাধান**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**সমস্যা**: লিন্টিং ত্রুটি বিল্ডে বাধা দিচ্ছে

**সমাধান**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## ডেটা এবং ফাইল পাথ সমস্যা

### পাথ সমস্যা

**সমস্যা**: নোটবুক চালানোর সময় ডেটা ফাইল পাওয়া যাচ্ছে না

**সমাধান**:
1. **সবসময় নোটবুক তার সংরক্ষিত ডিরেক্টরি থেকে চালান**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **কোডে আপেক্ষিক পাথ পরীক্ষা করুন**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **প্রয়োজনে সম্পূর্ণ পাথ ব্যবহার করুন**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### অনুপস্থিত ডেটা ফাইল

**সমস্যা**: ডেটাসেট ফাইল অনুপস্থিত

**সমাধান**:
1. ডেটা রিপোজিটরিতে থাকা উচিত কিনা পরীক্ষা করুন - বেশিরভাগ ডেটাসেট অন্তর্ভুক্ত।
2. কিছু পাঠে ডেটা ডাউনলোড করতে হতে পারে - পাঠের README দেখুন।
3. সর্বশেষ পরিবর্তনগুলি টেনে আনুন:
   ```bash
   git pull origin main
   ```

---

## সাধারণ ত্রুটি বার্তা

### মেমোরি ত্রুটি

**ত্রুটি**: `MemoryError` বা ডেটা প্রক্রিয়াকরণের সময় kernel বন্ধ হয়ে যাচ্ছে

**সমাধান**:
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

### কনভারজেন্স সতর্কতা

**সতর্কতা**: `ConvergenceWarning: Maximum number of iterations reached`

**সমাধান**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### প্লটিং সমস্যা

**সমস্যা**: Jupyter-এ প্লট দেখাচ্ছে না

**সমাধান**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**সমস্যা**: Seaborn প্লট ভিন্ন দেখাচ্ছে বা ত্রুটি দিচ্ছে

**সমাধান**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### ইউনিকোড/এনকোডিং ত্রুটি

**সমস্যা**: ফাইল পড়ার সময় `UnicodeDecodeError`

**সমাধান**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## পারফরম্যান্স সমস্যা

### নোটবুক চালানোর গতি কম

**সমস্যা**: নোটবুক চালাতে অনেক সময় লাগছে

**সমাধান**:
1. **মেমোরি মুক্ত করতে Kernel পুনরায় চালু করুন**: `Kernel → Restart`
2. **অপ্রয়োজনীয় নোটবুক বন্ধ করুন**: রিসোর্স মুক্ত করতে।
3. **পরীক্ষার জন্য ছোট ডেটা নমুনা ব্যবহার করুন**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **কোড প্রোফাইল করুন**: সমস্যা চিহ্নিত করতে।
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### উচ্চ মেমোরি ব্যবহার

**সমস্যা**: সিস্টেম মেমোরি শেষ হয়ে যাচ্ছে

**সমাধান**:
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

## পরিবেশ এবং কনফিগারেশন

### ভার্চুয়াল পরিবেশ সমস্যা

**সমস্যা**: ভার্চুয়াল পরিবেশ সক্রিয় হচ্ছে না

**সমাধান**:
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

**সমস্যা**: প্যাকেজ ইনস্টল হয়েছে কিন্তু নোটবুকে পাওয়া যাচ্ছে না

**সমাধান**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git সমস্যা

**সমস্যা**: সর্বশেষ পরিবর্তন টানতে পারছেন না - মর্জ কনফ্লিক্ট

**সমাধান**:
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

### VS Code ইন্টিগ্রেশন

**সমস্যা**: Jupyter নোটবুক VS Code-এ খুলছে না

**সমাধান**:
1. VS Code-এ Python এক্সটেনশন ইনস্টল করুন।
2. VS Code-এ Jupyter এক্সটেনশন ইনস্টল করুন।
3. সঠিক Python ইন্টারপ্রেটার নির্বাচন করুন: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code পুনরায় চালু করুন।

---

## অতিরিক্ত রিসোর্স

- **Discord Discussions**: [#ml-for-beginners চ্যানেলে প্রশ্ন করুন এবং সমাধান শেয়ার করুন](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners মডিউল](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **ভিডিও টিউটোরিয়াল**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **ইস্যু ট্র্যাকার**: [বাগ রিপোর্ট করুন](https://github.com/microsoft/ML-For-Beginners/issues)

---

## এখনও সমস্যা হচ্ছে?

যদি উপরের সমাধানগুলি চেষ্টা করার পরেও সমস্যা থেকে যায়:

1. **বিদ্যমান ইস্যু অনুসন্ধান করুন**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord-এ আলোচনা দেখুন**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **একটি নতুন ইস্যু খুলুন**: অন্তর্ভুক্ত করুন:
   - আপনার অপারেটিং সিস্টেম এবং সংস্করণ
   - Python/R সংস্করণ
   - ত্রুটি বার্তা (সম্পূর্ণ traceback)
   - সমস্যাটি পুনরায় তৈরি করার ধাপ
   - আপনি ইতিমধ্যে যা চেষ্টা করেছেন

আমরা সাহায্য করতে প্রস্তুত! 🚀

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসাধ্য সঠিকতার জন্য চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।