<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:49:46+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "he"
}
-->
# מדריך לפתרון תקלות

מדריך זה יעזור לכם לפתור בעיות נפוצות בעת עבודה עם תכנית הלימודים של "למידת מכונה למתחילים". אם לא מצאתם פתרון כאן, אנא בדקו את [דיוני הדיסקורד](https://aka.ms/foundry/discord) או [פתחו בעיה](https://github.com/microsoft/ML-For-Beginners/issues).

## תוכן עניינים

- [בעיות התקנה](../..)
- [בעיות במחברת Jupyter](../..)
- [בעיות בחבילות Python](../..)
- [בעיות בסביבת R](../..)
- [בעיות באפליקציית השאלונים](../..)
- [בעיות נתונים ונתיבי קבצים](../..)
- [הודעות שגיאה נפוצות](../..)
- [בעיות ביצועים](../..)
- [סביבה וקונפיגורציה](../..)

---

## בעיות התקנה

### התקנת Python

**בעיה**: `python: command not found`

**פתרון**:
1. התקינו Python 3.8 או גרסה גבוהה יותר מ-[python.org](https://www.python.org/downloads/)
2. בדקו את ההתקנה: `python --version` או `python3 --version`
3. ב-macOS/Linux ייתכן שתצטרכו להשתמש ב-`python3` במקום `python`

**בעיה**: גרסאות מרובות של Python גורמות להתנגשויות

**פתרון**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### התקנת Jupyter

**בעיה**: `jupyter: command not found`

**פתרון**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**בעיה**: Jupyter לא נפתח בדפדפן

**פתרון**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### התקנת R

**בעיה**: חבילות R לא מותקנות

**פתרון**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**בעיה**: IRkernel לא זמין ב-Jupyter

**פתרון**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## בעיות במחברת Jupyter

### בעיות בקרנל

**בעיה**: הקרנל מת או מופעל מחדש באופן קבוע

**פתרון**:
1. הפעל מחדש את הקרנל: `Kernel → Restart`
2. נקה פלט והפעל מחדש: `Kernel → Restart & Clear Output`
3. בדקו בעיות זיכרון (ראו [בעיות ביצועים](../..))
4. נסו להריץ תאים בנפרד כדי לזהות קוד בעייתי

**בעיה**: נבחר קרנל Python שגוי

**פתרון**:
1. בדקו את הקרנל הנוכחי: `Kernel → Change Kernel`
2. בחרו את גרסת Python הנכונה
3. אם הקרנל חסר, צרו אותו:
```bash
python -m ipykernel install --user --name=ml-env
```

**בעיה**: הקרנל לא מופעל

**פתרון**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### בעיות בתאי המחברת

**בעיה**: תאים רצים אך לא מציגים פלט

**פתרון**:
1. בדקו אם התא עדיין רץ (חפשו את הסימון `[*]`)
2. הפעל מחדש את הקרנל והריצו את כל התאים: `Kernel → Restart & Run All`
3. בדקו את קונסולת הדפדפן לשגיאות JavaScript (F12)

**בעיה**: לא ניתן להריץ תאים - אין תגובה בלחיצה על "Run"

**פתרון**:
1. בדקו אם שרת Jupyter עדיין פועל בטרמינל
2. רעננו את דף הדפדפן
3. סגרו ופתחו מחדש את המחברת
4. הפעל מחדש את שרת Jupyter

---

## בעיות בחבילות Python

### שגיאות ייבוא

**בעיה**: `ModuleNotFoundError: No module named 'sklearn'`

**פתרון**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**בעיה**: `ImportError: cannot import name 'X' from 'sklearn'`

**פתרון**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### התנגשויות גרסאות

**בעיה**: שגיאות אי-התאמה בין גרסאות חבילות

**פתרון**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**בעיה**: `pip install` נכשל עם שגיאות הרשאה

**פתרון**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### בעיות טעינת נתונים

**בעיה**: `FileNotFoundError` בעת טעינת קבצי CSV

**פתרון**:
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

## בעיות בסביבת R

### התקנת חבילות

**בעיה**: התקנת חבילה נכשלת עם שגיאות קומפילציה

**פתרון**:
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

**בעיה**: `tidyverse` לא מותקן

**פתרון**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### בעיות RMarkdown

**בעיה**: RMarkdown לא מוצג

**פתרון**:
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

## בעיות באפליקציית השאלונים

### בנייה והתקנה

**בעיה**: `npm install` נכשל

**פתרון**:
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

**בעיה**: פורט 8080 כבר בשימוש

**פתרון**:
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

### שגיאות בנייה

**בעיה**: `npm run build` נכשל

**פתרון**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**בעיה**: שגיאות Linting מונעות בנייה

**פתרון**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## בעיות נתונים ונתיבי קבצים

### בעיות נתיב

**בעיה**: קבצי נתונים לא נמצאים בעת הרצת מחברות

**פתרון**:
1. **הריצו תמיד מחברות מתוך התיקייה שלהן**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **בדקו נתיבים יחסיים בקוד**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **השתמשו בנתיבים מוחלטים אם צריך**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### קבצי נתונים חסרים

**בעיה**: קבצי מערכי נתונים חסרים

**פתרון**:
1. בדקו אם הנתונים אמורים להיות במאגר - רוב מערכי הנתונים כלולים
2. חלק מהשיעורים עשויים לדרוש הורדת נתונים - בדקו את README של השיעור
3. ודאו שמשכתם את השינויים האחרונים:
   ```bash
   git pull origin main
   ```

---

## הודעות שגיאה נפוצות

### שגיאות זיכרון

**שגיאה**: `MemoryError` או קרנל מת בעת עיבוד נתונים

**פתרון**:
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

### אזהרות התכנסות

**אזהרה**: `ConvergenceWarning: Maximum number of iterations reached`

**פתרון**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### בעיות גרפים

**בעיה**: גרפים לא מוצגים ב-Jupyter

**פתרון**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**בעיה**: גרפים של Seaborn נראים שונה או גורמים לשגיאות

**פתרון**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### שגיאות Unicode/קידוד

**בעיה**: `UnicodeDecodeError` בעת קריאת קבצים

**פתרון**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## בעיות ביצועים

### הרצת מחברות איטית

**בעיה**: מחברות רצות באיטיות רבה

**פתרון**:
1. **הפעילו מחדש את הקרנל כדי לפנות זיכרון**: `Kernel → Restart`
2. **סגרו מחברות לא בשימוש** כדי לפנות משאבים
3. **השתמשו בדגימות נתונים קטנות יותר לבדיקות**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **בצעו פרופיל לקוד שלכם** כדי לזהות צווארי בקבוק:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### שימוש גבוה בזיכרון

**בעיה**: המערכת נגמרת מזיכרון

**פתרון**:
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

## סביבה וקונפיגורציה

### בעיות בסביבה וירטואלית

**בעיה**: הסביבה הווירטואלית לא מופעלת

**פתרון**:
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

**בעיה**: חבילות מותקנות אך לא נמצאות במחברת

**פתרון**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### בעיות Git

**בעיה**: לא ניתן למשוך שינויים אחרונים - התנגשויות מיזוג

**פתרון**:
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

### אינטגרציה עם VS Code

**בעיה**: מחברות Jupyter לא נפתחות ב-VS Code

**פתרון**:
1. התקינו את תוסף Python ב-VS Code
2. התקינו את תוסף Jupyter ב-VS Code
3. בחרו את מפרש Python הנכון: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. הפעילו מחדש את VS Code

---

## משאבים נוספים

- **דיוני דיסקורד**: [שאלו שאלות ושתפו פתרונות בערוץ #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [מודולים של ML למתחילים](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **מדריכי וידאו**: [רשימת השמעה ביוטיוב](https://aka.ms/ml-beginners-videos)
- **מעקב בעיות**: [דווחו על באגים](https://github.com/microsoft/ML-For-Beginners/issues)

---

## עדיין יש בעיות?

אם ניסיתם את הפתרונות לעיל ועדיין חווים בעיות:

1. **חפשו בעיות קיימות**: [בעיות ב-GitHub](https://github.com/microsoft/ML-For-Beginners/issues)
2. **בדקו דיונים בדיסקורד**: [דיוני דיסקורד](https://aka.ms/foundry/discord)
3. **פתחו בעיה חדשה**: כללו:
   - מערכת ההפעלה והגרסה שלכם
   - גרסת Python/R
   - הודעת השגיאה (כולל כל ה-traceback)
   - שלבים לשחזור הבעיה
   - מה כבר ניסיתם

אנחנו כאן כדי לעזור! 🚀

---

**הצהרת אחריות**:  
מסמך זה תורגם באמצעות שירות תרגום מבוסס AI [Co-op Translator](https://github.com/Azure/co-op-translator). למרות שאנו שואפים לדיוק, יש לקחת בחשבון שתרגומים אוטומטיים עשויים להכיל שגיאות או אי דיוקים. המסמך המקורי בשפתו המקורית צריך להיחשב כמקור הסמכותי. עבור מידע קריטי, מומלץ להשתמש בתרגום מקצועי על ידי אדם. אנו לא נושאים באחריות לאי הבנות או לפרשנויות שגויות הנובעות משימוש בתרגום זה.