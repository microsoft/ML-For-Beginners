<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:57:16+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "uk"
}
-->
# Посібник з усунення несправностей

Цей посібник допоможе вам вирішити поширені проблеми під час роботи з навчальним курсом "Машинне навчання для початківців". Якщо ви не знайдете рішення тут, перевірте наші [обговорення в Discord](https://aka.ms/foundry/discord) або [створіть нову проблему](https://github.com/microsoft/ML-For-Beginners/issues).

## Зміст

- [Проблеми з установкою](../..)
- [Проблеми з Jupyter Notebook](../..)
- [Проблеми з Python-пакетами](../..)
- [Проблеми з середовищем R](../..)
- [Проблеми з додатком для тестів](../..)
- [Проблеми з даними та шляхами до файлів](../..)
- [Поширені повідомлення про помилки](../..)
- [Проблеми з продуктивністю](../..)
- [Середовище та конфігурація](../..)

---

## Проблеми з установкою

### Установка Python

**Проблема**: `python: command not found`

**Рішення**:
1. Встановіть Python 3.8 або новішу версію з [python.org](https://www.python.org/downloads/)
2. Перевірте установку: `python --version` або `python3 --version`
3. На macOS/Linux можливо, доведеться використовувати `python3` замість `python`

**Проблема**: Конфлікти через кілька версій Python

**Рішення**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Установка Jupyter

**Проблема**: `jupyter: command not found`

**Рішення**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Проблема**: Jupyter не відкривається в браузері

**Рішення**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Установка R

**Проблема**: R-пакети не встановлюються

**Рішення**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Проблема**: IRkernel недоступний у Jupyter

**Рішення**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Проблеми з Jupyter Notebook

### Проблеми з ядром

**Проблема**: Ядро постійно завершує роботу або перезапускається

**Рішення**:
1. Перезапустіть ядро: `Kernel → Restart`
2. Очистіть вивід і перезапустіть: `Kernel → Restart & Clear Output`
3. Перевірте проблеми з пам'яттю (див. [Проблеми з продуктивністю](../..))
4. Запускайте клітинки окремо, щоб знайти проблемний код

**Проблема**: Вибрано неправильне ядро Python

**Рішення**:
1. Перевірте поточне ядро: `Kernel → Change Kernel`
2. Виберіть правильну версію Python
3. Якщо ядро відсутнє, створіть його:
```bash
python -m ipykernel install --user --name=ml-env
```

**Проблема**: Ядро не запускається

**Рішення**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Проблеми з клітинками в ноутбуці

**Проблема**: Клітинки виконуються, але не показують результат

**Рішення**:
1. Перевірте, чи клітинка все ще виконується (шукайте індикатор `[*]`)
2. Перезапустіть ядро і виконайте всі клітинки: `Kernel → Restart & Run All`
3. Перевірте консоль браузера на наявність помилок JavaScript (F12)

**Проблема**: Клітинки не виконуються - немає реакції на натискання "Run"

**Рішення**:
1. Перевірте, чи сервер Jupyter все ще працює в терміналі
2. Оновіть сторінку браузера
3. Закрийте та знову відкрийте ноутбук
4. Перезапустіть сервер Jupyter

---

## Проблеми з Python-пакетами

### Помилки імпорту

**Проблема**: `ModuleNotFoundError: No module named 'sklearn'`

**Рішення**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Проблема**: `ImportError: cannot import name 'X' from 'sklearn'`

**Рішення**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Конфлікти версій

**Проблема**: Помилки несумісності версій пакетів

**Рішення**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Проблема**: `pip install` не працює через помилки дозволів

**Рішення**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Проблеми з завантаженням даних

**Проблема**: `FileNotFoundError` при завантаженні CSV-файлів

**Рішення**:
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

## Проблеми з середовищем R

### Установка пакетів

**Проблема**: Установка пакета завершується помилками компіляції

**Рішення**:
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

**Проблема**: `tidyverse` не встановлюється

**Рішення**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Проблеми з RMarkdown

**Проблема**: RMarkdown не рендериться

**Рішення**:
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

## Проблеми з додатком для тестів

### Збірка та установка

**Проблема**: `npm install` не працює

**Рішення**:
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

**Проблема**: Порт 8080 вже зайнятий

**Рішення**:
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

### Помилки збірки

**Проблема**: `npm run build` не працює

**Рішення**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Проблема**: Помилки лінтингу заважають збірці

**Рішення**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Проблеми з даними та шляхами до файлів

### Проблеми з шляхами

**Проблема**: Дані не знайдено при виконанні ноутбуків

**Рішення**:
1. **Завжди запускайте ноутбуки з їхньої директорії**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Перевірте відносні шляхи в коді**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Використовуйте абсолютні шляхи, якщо потрібно**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Відсутні файли даних

**Проблема**: Файли з даними відсутні

**Рішення**:
1. Перевірте, чи дані повинні бути в репозиторії - більшість наборів даних включені
2. Деякі уроки можуть вимагати завантаження даних - перевірте README уроку
3. Переконайтеся, що ви завантажили останні зміни:
   ```bash
   git pull origin main
   ```

---

## Поширені повідомлення про помилки

### Помилки пам'яті

**Помилка**: `MemoryError` або ядро завершує роботу при обробці даних

**Рішення**:
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

### Попередження про збіжність

**Попередження**: `ConvergenceWarning: Maximum number of iterations reached`

**Рішення**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Проблеми з побудовою графіків

**Проблема**: Графіки не відображаються в Jupyter

**Рішення**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Проблема**: Графіки Seaborn виглядають інакше або викликають помилки

**Рішення**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Помилки кодування/Unicode

**Проблема**: `UnicodeDecodeError` при читанні файлів

**Рішення**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Проблеми з продуктивністю

### Повільне виконання ноутбуків

**Проблема**: Ноутбуки виконуються дуже повільно

**Рішення**:
1. **Перезапустіть ядро, щоб звільнити пам'ять**: `Kernel → Restart`
2. **Закрийте невикористовувані ноутбуки**, щоб звільнити ресурси
3. **Використовуйте менші вибірки даних для тестування**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Профілюйте ваш код**, щоб знайти вузькі місця:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Високе використання пам'яті

**Проблема**: Система вичерпує пам'ять

**Рішення**:
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

## Середовище та конфігурація

### Проблеми з віртуальним середовищем

**Проблема**: Віртуальне середовище не активується

**Рішення**:
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

**Проблема**: Пакети встановлені, але не знайдені в ноутбуці

**Рішення**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Проблеми з Git

**Проблема**: Неможливо завантажити останні зміни - конфлікти злиття

**Рішення**:
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

### Інтеграція з VS Code

**Проблема**: Jupyter ноутбуки не відкриваються в VS Code

**Рішення**:
1. Встановіть розширення Python у VS Code
2. Встановіть розширення Jupyter у VS Code
3. Виберіть правильний інтерпретатор Python: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Перезапустіть VS Code

---

## Додаткові ресурси

- **Обговорення в Discord**: [Ставте запитання та діліться рішеннями в каналі #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Модулі "Машинне навчання для початківців"](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Відеоуроки**: [Плейлист на YouTube](https://aka.ms/ml-beginners-videos)
- **Трекер проблем**: [Повідомляйте про помилки](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Все ще виникають проблеми?

Якщо ви спробували наведені вище рішення і все ще стикаєтеся з проблемами:

1. **Шукайте існуючі проблеми**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Перевірте обговорення в Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Створіть нову проблему**: Включіть:
   - Вашу операційну систему та її версію
   - Версію Python/R
   - Повідомлення про помилку (повний стек трасування)
   - Кроки для відтворення проблеми
   - Що ви вже спробували

Ми тут, щоб допомогти! 🚀

---

**Відмова від відповідальності**:  
Цей документ був перекладений за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, звертаємо вашу увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ на його рідній мові слід вважати авторитетним джерелом. Для критичної інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.