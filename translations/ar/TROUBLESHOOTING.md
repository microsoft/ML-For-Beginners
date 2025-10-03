<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:37:02+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ar"
}
-->
# دليل استكشاف الأخطاء وإصلاحها

يساعدك هذا الدليل في حل المشكلات الشائعة أثناء العمل مع منهج تعلم الآلة للمبتدئين. إذا لم تجد حلاً هنا، يرجى التحقق من [مناقشات Discord](https://aka.ms/foundry/discord) أو [فتح قضية](https://github.com/microsoft/ML-For-Beginners/issues).

## جدول المحتويات

- [مشاكل التثبيت](../..)
- [مشاكل Jupyter Notebook](../..)
- [مشاكل حزم Python](../..)
- [مشاكل بيئة R](../..)
- [مشاكل تطبيق الاختبارات](../..)
- [مشاكل البيانات ومسارات الملفات](../..)
- [رسائل الخطأ الشائعة](../..)
- [مشاكل الأداء](../..)
- [البيئة والتكوين](../..)

---

## مشاكل التثبيت

### تثبيت Python

**المشكلة**: `python: command not found`

**الحل**:
1. قم بتثبيت Python 3.8 أو أعلى من [python.org](https://www.python.org/downloads/)
2. تحقق من التثبيت: `python --version` أو `python3 --version`
3. على macOS/Linux، قد تحتاج إلى استخدام `python3` بدلاً من `python`

**المشكلة**: وجود نسخ متعددة من Python تسبب تعارضات

**الحل**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### تثبيت Jupyter

**المشكلة**: `jupyter: command not found`

**الحل**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**المشكلة**: Jupyter لا يفتح في المتصفح

**الحل**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### تثبيت R

**المشكلة**: حزم R لا يتم تثبيتها

**الحل**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**المشكلة**: IRkernel غير متوفر في Jupyter

**الحل**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## مشاكل Jupyter Notebook

### مشاكل النواة

**المشكلة**: النواة تستمر في التوقف أو إعادة التشغيل

**الحل**:
1. أعد تشغيل النواة: `Kernel → Restart`
2. امسح المخرجات وأعد التشغيل: `Kernel → Restart & Clear Output`
3. تحقق من مشاكل الذاكرة (راجع [مشاكل الأداء](../..))
4. حاول تشغيل الخلايا بشكل فردي لتحديد الكود المسبب للمشكلة

**المشكلة**: اختيار نواة Python خاطئة

**الحل**:
1. تحقق من النواة الحالية: `Kernel → Change Kernel`
2. اختر إصدار Python الصحيح
3. إذا كانت النواة مفقودة، قم بإنشائها:
```bash
python -m ipykernel install --user --name=ml-env
```

**المشكلة**: النواة لا تبدأ

**الحل**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### مشاكل خلايا الدفتر

**المشكلة**: الخلايا تعمل ولكن لا تظهر المخرجات

**الحل**:
1. تحقق مما إذا كانت الخلية لا تزال تعمل (ابحث عن المؤشر `[*]`)
2. أعد تشغيل النواة وشغل جميع الخلايا: `Kernel → Restart & Run All`
3. تحقق من وحدة التحكم في المتصفح بحثًا عن أخطاء JavaScript (F12)

**المشكلة**: لا يمكن تشغيل الخلايا - لا يوجد استجابة عند النقر على "Run"

**الحل**:
1. تحقق مما إذا كان خادم Jupyter لا يزال يعمل في الطرفية
2. قم بتحديث صفحة المتصفح
3. أغلق وأعد فتح الدفتر
4. أعد تشغيل خادم Jupyter

---

## مشاكل حزم Python

### أخطاء الاستيراد

**المشكلة**: `ModuleNotFoundError: No module named 'sklearn'`

**الحل**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**المشكلة**: `ImportError: cannot import name 'X' from 'sklearn'`

**الحل**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### تعارضات الإصدارات

**المشكلة**: أخطاء عدم توافق إصدار الحزمة

**الحل**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**المشكلة**: `pip install` يفشل بسبب أخطاء الأذونات

**الحل**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### مشاكل تحميل البيانات

**المشكلة**: `FileNotFoundError` عند تحميل ملفات CSV

**الحل**:
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

## مشاكل بيئة R

### تثبيت الحزم

**المشكلة**: فشل تثبيت الحزمة بسبب أخطاء الترجمة

**الحل**:
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

**المشكلة**: `tidyverse` لا يتم تثبيته

**الحل**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### مشاكل RMarkdown

**المشكلة**: RMarkdown لا يتم عرضه

**الحل**:
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

## مشاكل تطبيق الاختبارات

### البناء والتثبيت

**المشكلة**: `npm install` يفشل

**الحل**:
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

**المشكلة**: المنفذ 8080 قيد الاستخدام بالفعل

**الحل**:
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

### أخطاء البناء

**المشكلة**: `npm run build` يفشل

**الحل**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**المشكلة**: أخطاء الفحص تمنع البناء

**الحل**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## مشاكل البيانات ومسارات الملفات

### مشاكل المسارات

**المشكلة**: ملفات البيانات غير موجودة عند تشغيل الدفاتر

**الحل**:
1. **قم دائمًا بتشغيل الدفاتر من الدليل الذي يحتوي عليها**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **تحقق من المسارات النسبية في الكود**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **استخدم المسارات المطلقة إذا لزم الأمر**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### ملفات البيانات المفقودة

**المشكلة**: ملفات مجموعة البيانات مفقودة

**الحل**:
1. تحقق مما إذا كانت البيانات يجب أن تكون في المستودع - معظم مجموعات البيانات مضمنة
2. قد تتطلب بعض الدروس تنزيل البيانات - تحقق من README الخاص بالدرس
3. تأكد من أنك قمت بسحب أحدث التغييرات:
   ```bash
   git pull origin main
   ```

---

## رسائل الخطأ الشائعة

### أخطاء الذاكرة

**الخطأ**: `MemoryError` أو توقف النواة عند معالجة البيانات

**الحل**:
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

### تحذيرات التقارب

**التحذير**: `ConvergenceWarning: Maximum number of iterations reached`

**الحل**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### مشاكل الرسم

**المشكلة**: الرسومات لا تظهر في Jupyter

**الحل**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**المشكلة**: رسومات Seaborn تبدو مختلفة أو تظهر أخطاء

**الحل**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### أخطاء الترميز/Unicode

**المشكلة**: `UnicodeDecodeError` عند قراءة الملفات

**الحل**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## مشاكل الأداء

### بطء تنفيذ الدفاتر

**المشكلة**: الدفاتر بطيئة جدًا في التشغيل

**الحل**:
1. **أعد تشغيل النواة لتحرير الذاكرة**: `Kernel → Restart`
2. **أغلق الدفاتر غير المستخدمة** لتحرير الموارد
3. **استخدم عينات بيانات أصغر للاختبار**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **قم بتحليل الكود الخاص بك** لتحديد نقاط الاختناق:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### استخدام عالي للذاكرة

**المشكلة**: النظام ينفد من الذاكرة

**الحل**:
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

## البيئة والتكوين

### مشاكل البيئة الافتراضية

**المشكلة**: البيئة الافتراضية لا تعمل

**الحل**:
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

**المشكلة**: الحزم مثبتة ولكن غير موجودة في الدفتر

**الحل**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### مشاكل Git

**المشكلة**: لا يمكن سحب أحدث التغييرات - تعارضات الدمج

**الحل**:
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

### تكامل VS Code

**المشكلة**: دفاتر Jupyter لا تفتح في VS Code

**الحل**:
1. قم بتثبيت إضافة Python في VS Code
2. قم بتثبيت إضافة Jupyter في VS Code
3. اختر مترجم Python الصحيح: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. أعد تشغيل VS Code

---

## موارد إضافية

- **مناقشات Discord**: [اطرح الأسئلة وشارك الحلول في قناة #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [وحدات تعلم الآلة للمبتدئين](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **دروس الفيديو**: [قائمة تشغيل YouTube](https://aka.ms/ml-beginners-videos)
- **متتبع القضايا**: [الإبلاغ عن الأخطاء](https://github.com/microsoft/ML-For-Beginners/issues)

---

## لا تزال تواجه مشاكل؟

إذا جربت الحلول أعلاه وما زلت تواجه مشاكل:

1. **ابحث عن القضايا الموجودة**: [قضايا GitHub](https://github.com/microsoft/ML-For-Beginners/issues)
2. **تحقق من المناقشات في Discord**: [مناقشات Discord](https://aka.ms/foundry/discord)
3. **افتح قضية جديدة**: قم بتضمين:
   - نظام التشغيل والإصدار الخاص بك
   - إصدار Python/R
   - رسالة الخطأ (التتبع الكامل)
   - خطوات إعادة إنتاج المشكلة
   - ما الذي جربته بالفعل

نحن هنا للمساعدة! 🚀

---

**إخلاء المسؤولية**:  
تم ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو عدم دقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة ناتجة عن استخدام هذه الترجمة.