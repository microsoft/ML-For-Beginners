<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:55:10+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "sr"
}
-->
# Водич за решавање проблема

Овај водич вам помаже да решите уобичајене проблеме приликом рада са наставним планом и програмом „Машинско учење за почетнике“. Ако овде не пронађете решење, проверите наше [Discord дискусије](https://aka.ms/foundry/discord) или [отворите проблем](https://github.com/microsoft/ML-For-Beginners/issues).

## Садржај

- [Проблеми са инсталацијом](../..)
- [Проблеми са Jupyter Notebook-ом](../..)
- [Проблеми са Python пакетима](../..)
- [Проблеми са R окружењем](../..)
- [Проблеми са апликацијом за квиз](../..)
- [Проблеми са подацима и путањама до датотека](../..)
- [Уобичајене поруке о грешкама](../..)
- [Проблеми са перформансама](../..)
- [Окружење и конфигурација](../..)

---

## Проблеми са инсталацијом

### Инсталација Python-а

**Проблем**: `python: command not found`

**Решење**:
1. Инсталирајте Python 3.8 или новији са [python.org](https://www.python.org/downloads/)
2. Проверите инсталацију: `python --version` или `python3 --version`
3. На macOS/Linux, можда ћете морати да користите `python3` уместо `python`

**Проблем**: Конфликти због више верзија Python-а

**Решење**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Инсталација Jupyter-а

**Проблем**: `jupyter: command not found`

**Решење**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Проблем**: Jupyter се не покреће у прегледачу

**Решење**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Инсталација R-а

**Проблем**: R пакети се не инсталирају

**Решење**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Проблем**: IRkernel није доступан у Jupyter-у

**Решење**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Проблеми са Jupyter Notebook-ом

### Проблеми са кернелом

**Проблем**: Кернел се стално гаси или рестартује

**Решење**:
1. Рестартујте кернел: `Kernel → Restart`
2. Очистите излаз и рестартујте: `Kernel → Restart & Clear Output`
3. Проверите проблеме са меморијом (погледајте [Проблеми са перформансама](../..))
4. Покрените ћелије појединачно да бисте идентификовали проблематичан код

**Проблем**: Изабран је погрешан Python кернел

**Решење**:
1. Проверите тренутни кернел: `Kernel → Change Kernel`
2. Изаберите исправну верзију Python-а
3. Ако кернел недостаје, креирајте га:
```bash
python -m ipykernel install --user --name=ml-env
```

**Проблем**: Кернел се не покреће

**Решење**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Проблеми са ћелијама у Notebook-у

**Проблем**: Ћелије се покрећу, али не приказују излаз

**Решење**:
1. Проверите да ли је ћелија још увек у процесу извршавања (потражите индикатор `[*]`)
2. Рестартујте кернел и покрените све ћелије: `Kernel → Restart & Run All`
3. Проверите конзолу прегледача за JavaScript грешке (F12)

**Проблем**: Ћелије се не могу покренути - нема одговора када кликнете на „Run“

**Решење**:
1. Проверите да ли Jupyter сервер још увек ради у терминалу
2. Освежите страницу у прегледачу
3. Затворите и поново отворите Notebook
4. Рестартујте Jupyter сервер

---

## Проблеми са Python пакетима

### Грешке при увозу

**Проблем**: `ModuleNotFoundError: No module named 'sklearn'`

**Решење**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Проблем**: `ImportError: cannot import name 'X' from 'sklearn'`

**Решење**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Конфликти верзија

**Проблем**: Грешке због некомпатибилности верзија пакета

**Решење**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Проблем**: `pip install` не успева због проблема са дозволама

**Решење**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Проблеми са учитавањем података

**Проблем**: `FileNotFoundError` приликом учитавања CSV датотека

**Решење**:
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

## Проблеми са R окружењем

### Инсталација пакета

**Проблем**: Инсталација пакета не успева због грешака у компилацији

**Решење**:
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

**Проблем**: `tidyverse` се не инсталира

**Решење**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Проблеми са RMarkdown-ом

**Проблем**: RMarkdown се не рендерује

**Решење**:
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

## Проблеми са апликацијом за квиз

### Изградња и инсталација

**Проблем**: `npm install` не успева

**Решење**:
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

**Проблем**: Порт 8080 је већ у употреби

**Решење**:
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

### Грешке при изградњи

**Проблем**: `npm run build` не успева

**Решење**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Проблем**: Грешке у линтовању спречавају изградњу

**Решење**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Проблеми са подацима и путањама до датотека

### Проблеми са путањама

**Проблем**: Датотеке са подацима нису пронађене приликом покретања Notebook-а

**Решење**:
1. **Увек покрећите Notebook из његовог директоријума**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Проверите релативне путање у коду**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Користите апсолутне путање ако је потребно**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Недостају датотеке са подацима

**Проблем**: Недостају датотеке са сетовима података

**Решење**:
1. Проверите да ли би подаци требало да буду у репозиторијуму - већина сетова података је укључена
2. Неке лекције могу захтевати преузимање података - проверите README лекције
3. Уверите се да сте повукли најновије измене:
   ```bash
   git pull origin main
   ```

---

## Уобичајене поруке о грешкама

### Грешке са меморијом

**Грешка**: `MemoryError` или кернел се гаси приликом обраде података

**Решење**:
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

### Упозорења о конвергенцији

**Упозорење**: `ConvergenceWarning: Maximum number of iterations reached`

**Решење**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Проблеми са графицима

**Проблем**: Графици се не приказују у Jupyter-у

**Решење**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Проблем**: Seaborn графици изгледају другачије или изазивају грешке

**Решење**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding грешке

**Проблем**: `UnicodeDecodeError` приликом читања датотека

**Решење**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Проблеми са перформансама

### Споро извршавање Notebook-а

**Проблем**: Notebook-и се веома споро извршавају

**Решење**:
1. **Рестартујте кернел да ослободите меморију**: `Kernel → Restart`
2. **Затворите непотребне Notebook-е** да ослободите ресурсе
3. **Користите мање узорке података за тестирање**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Профилишите свој код** да бисте пронашли уска грла:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Висока употреба меморије

**Проблем**: Систем остаје без меморије

**Решење**:
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

## Окружење и конфигурација

### Проблеми са виртуелним окружењем

**Проблем**: Виртуелно окружење се не активира

**Решење**:
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

**Проблем**: Пакети су инсталирани, али нису пронађени у Notebook-у

**Решење**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Проблеми са Git-ом

**Проблем**: Не могу да повучем најновије измене - конфликти приликом спајања

**Решење**:
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

### Интеграција са VS Code-ом

**Проблем**: Jupyter Notebook-и се не отварају у VS Code-у

**Решење**:
1. Инсталирајте Python екстензију у VS Code-у
2. Инсталирајте Jupyter екстензију у VS Code-у
3. Изаберите исправан Python интерпретер: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Рестартујте VS Code

---

## Додатни ресурси

- **Discord дискусије**: [Поставите питања и поделите решења у каналу #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Модули „Машинско учење за почетнике“](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Видео туторијали**: [YouTube плејлиста](https://aka.ms/ml-beginners-videos)
- **Трагач за проблемима**: [Пријавите грешке](https://github.com/microsoft/ML-For-Beginners/issues)

---

## И даље имате проблеме?

Ако сте пробали горе наведена решења и још увек имате проблеме:

1. **Претражите постојеће проблеме**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Проверите дискусије на Discord-у**: [Discord дискусије](https://aka.ms/foundry/discord)
3. **Отворите нови проблем**: Укључите:
   - Ваш оперативни систем и верзију
   - Python/R верзију
   - Поруку о грешци (цели traceback)
   - Кораке за репродукцију проблема
   - Шта сте већ пробали

Ту смо да помогнемо! 🚀

---

**Одрицање од одговорности**:  
Овај документ је преведен помоћу услуге за превођење уз помоћ вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати меродавним извором. За критичне информације препоручује се професионални превод од стране људског преводиоца. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.