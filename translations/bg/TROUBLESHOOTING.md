<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:54:36+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "bg"
}
-->
# Ръководство за отстраняване на проблеми

Това ръководство ще ви помогне да решите често срещани проблеми при работа с учебната програма "Машинно обучение за начинаещи". Ако не намерите решение тук, моля, проверете нашите [дискусии в Discord](https://aka.ms/foundry/discord) или [отворете нов проблем](https://github.com/microsoft/ML-For-Beginners/issues).

## Съдържание

- [Проблеми с инсталацията](../..)
- [Проблеми с Jupyter Notebook](../..)
- [Проблеми с Python пакети](../..)
- [Проблеми с R средата](../..)
- [Проблеми с приложението за тестове](../..)
- [Проблеми с данни и пътища към файлове](../..)
- [Често срещани съобщения за грешки](../..)
- [Проблеми с производителността](../..)
- [Среда и конфигурация](../..)

---

## Проблеми с инсталацията

### Инсталация на Python

**Проблем**: `python: command not found`

**Решение**:
1. Инсталирайте Python 3.8 или по-нова версия от [python.org](https://www.python.org/downloads/)
2. Проверете инсталацията: `python --version` или `python3 --version`
3. На macOS/Linux може да се наложи да използвате `python3` вместо `python`

**Проблем**: Конфликти между различни версии на Python

**Решение**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Инсталация на Jupyter

**Проблем**: `jupyter: command not found`

**Решение**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Проблем**: Jupyter не се отваря в браузъра

**Решение**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Инсталация на R

**Проблем**: R пакетите не се инсталират

**Решение**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Проблем**: IRkernel не е наличен в Jupyter

**Решение**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Проблеми с Jupyter Notebook

### Проблеми с ядрото (Kernel)

**Проблем**: Ядрото постоянно се рестартира или спира

**Решение**:
1. Рестартирайте ядрото: `Kernel → Restart`
2. Изчистете изхода и рестартирайте: `Kernel → Restart & Clear Output`
3. Проверете за проблеми с паметта (вижте [Проблеми с производителността](../..))
4. Стартирайте клетките една по една, за да идентифицирате проблемния код

**Проблем**: Избрано е грешно Python ядро

**Решение**:
1. Проверете текущото ядро: `Kernel → Change Kernel`
2. Изберете правилната версия на Python
3. Ако ядрото липсва, създайте го:
```bash
python -m ipykernel install --user --name=ml-env
```

**Проблем**: Ядрото не стартира

**Решение**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Проблеми с клетките в Notebook

**Проблем**: Клетките се изпълняват, но не показват резултати

**Решение**:
1. Проверете дали клетката все още се изпълнява (потърсете индикатора `[*]`)
2. Рестартирайте ядрото и изпълнете всички клетки: `Kernel → Restart & Run All`
3. Проверете конзолата на браузъра за грешки в JavaScript (F12)

**Проблем**: Клетките не се изпълняват - няма реакция при натискане на "Run"

**Решение**:
1. Проверете дали Jupyter сървърът все още работи в терминала
2. Презаредете страницата на браузъра
3. Затворете и отворете отново Notebook-а
4. Рестартирайте Jupyter сървъра

---

## Проблеми с Python пакети

### Грешки при импортиране

**Проблем**: `ModuleNotFoundError: No module named 'sklearn'`

**Решение**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Проблем**: `ImportError: cannot import name 'X' from 'sklearn'`

**Решение**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Конфликти с версии

**Проблем**: Грешки за несъвместимост на версии на пакети

**Решение**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Проблем**: `pip install` не успява поради грешки с права

**Решение**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Проблеми с зареждане на данни

**Проблем**: `FileNotFoundError` при зареждане на CSV файлове

**Решение**:
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

## Проблеми с R средата

### Инсталация на пакети

**Проблем**: Инсталацията на пакети се проваля с грешки при компилация

**Решение**:
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

**Проблем**: `tidyverse` не се инсталира

**Решение**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Проблеми с RMarkdown

**Проблем**: RMarkdown не се рендерира

**Решение**:
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

## Проблеми с приложението за тестове

### Създаване и инсталация

**Проблем**: `npm install` не успява

**Решение**:
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

**Проблем**: Порт 8080 вече се използва

**Решение**:
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

### Грешки при създаване

**Проблем**: `npm run build` не успява

**Решение**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Проблем**: Грешки при проверка на кода (Linting), които пречат на създаването

**Решение**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Проблеми с данни и пътища към файлове

### Проблеми с пътища

**Проблем**: Данните не се намират при изпълнение на Notebook-и

**Решение**:
1. **Винаги изпълнявайте Notebook-ите от директорията, в която се намират**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Проверете относителните пътища в кода**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Използвайте абсолютни пътища, ако е необходимо**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Липсващи файлове с данни

**Проблем**: Липсват файлове с данни

**Решение**:
1. Проверете дали данните трябва да са в репозитория - повечето набори от данни са включени
2. Някои уроци може да изискват изтегляне на данни - проверете README на урока
3. Уверете се, че сте изтеглили последните промени:
   ```bash
   git pull origin main
   ```

---

## Често срещани съобщения за грешки

### Грешки с паметта

**Грешка**: `MemoryError` или ядрото спира при обработка на данни

**Решение**:
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

### Предупреждения за конвергенция

**Предупреждение**: `ConvergenceWarning: Maximum number of iterations reached`

**Решение**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Проблеми с визуализацията

**Проблем**: Графиките не се показват в Jupyter

**Решение**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Проблем**: Графиките на Seaborn изглеждат различно или хвърлят грешки

**Решение**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Грешки с Unicode/кодиране

**Проблем**: `UnicodeDecodeError` при четене на файлове

**Решение**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Проблеми с производителността

### Бавно изпълнение на Notebook-и

**Проблем**: Notebook-ите се изпълняват много бавно

**Решение**:
1. **Рестартирайте ядрото, за да освободите памет**: `Kernel → Restart`
2. **Затворете неизползвани Notebook-и**, за да освободите ресурси
3. **Използвайте по-малки проби от данни за тестване**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Профилирайте кода си**, за да намерите тесните места:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Висока употреба на памет

**Проблем**: Системата изчерпва паметта

**Решение**:
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

## Среда и конфигурация

### Проблеми с виртуална среда

**Проблем**: Виртуалната среда не се активира

**Решение**:
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

**Проблем**: Пакетите са инсталирани, но не се намират в Notebook-а

**Решение**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Проблеми с Git

**Проблем**: Не може да се изтеглят последните промени - конфликти при сливане

**Решение**:
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

### Интеграция с VS Code

**Проблем**: Jupyter Notebook-ите не се отварят в VS Code

**Решение**:
1. Инсталирайте разширението Python в VS Code
2. Инсталирайте разширението Jupyter в VS Code
3. Изберете правилния Python интерпретатор: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Рестартирайте VS Code

---

## Допълнителни ресурси

- **Дискусии в Discord**: [Задавайте въпроси и споделяйте решения в канала #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Модули за ML за начинаещи](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Видео уроци**: [Плейлист в YouTube](https://aka.ms/ml-beginners-videos)
- **Тракер за проблеми**: [Докладвайте грешки](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Все още имате проблеми?

Ако сте опитали горните решения и все още срещате трудности:

1. **Потърсете съществуващи проблеми**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Проверете дискусиите в Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Отворете нов проблем**: Включете:
   - Вашата операционна система и версия
   - Версия на Python/R
   - Съобщение за грешка (пълен traceback)
   - Стъпки за възпроизвеждане на проблема
   - Какво вече сте опитали

Тук сме, за да помогнем! 🚀

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.