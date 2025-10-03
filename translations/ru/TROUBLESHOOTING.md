<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:36:32+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ru"
}
-->
# Руководство по устранению неполадок

Это руководство поможет вам решить распространенные проблемы при работе с учебной программой "Машинное обучение для начинающих". Если вы не нашли решение здесь, загляните в наши [обсуждения на Discord](https://aka.ms/foundry/discord) или [откройте новый запрос](https://github.com/microsoft/ML-For-Beginners/issues).

## Содержание

- [Проблемы с установкой](../..)
- [Проблемы с Jupyter Notebook](../..)
- [Проблемы с Python-пакетами](../..)
- [Проблемы с окружением R](../..)
- [Проблемы с приложением для викторин](../..)
- [Проблемы с данными и путями к файлам](../..)
- [Распространенные сообщения об ошибках](../..)
- [Проблемы с производительностью](../..)
- [Окружение и конфигурация](../..)

---

## Проблемы с установкой

### Установка Python

**Проблема**: `python: command not found`

**Решение**:
1. Установите Python версии 3.8 или выше с [python.org](https://www.python.org/downloads/)
2. Проверьте установку: `python --version` или `python3 --version`
3. На macOS/Linux возможно потребуется использовать `python3` вместо `python`

**Проблема**: Конфликты из-за нескольких версий Python

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

### Установка Jupyter

**Проблема**: `jupyter: command not found`

**Решение**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Проблема**: Jupyter не открывается в браузере

**Решение**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Установка R

**Проблема**: R-пакеты не устанавливаются

**Решение**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Проблема**: IRkernel недоступен в Jupyter

**Решение**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Проблемы с Jupyter Notebook

### Проблемы с ядром

**Проблема**: Ядро постоянно завершает работу или перезапускается

**Решение**:
1. Перезапустите ядро: `Kernel → Restart`
2. Очистите вывод и перезапустите: `Kernel → Restart & Clear Output`
3. Проверьте проблемы с памятью (см. [Проблемы с производительностью](../..))
4. Запустите ячейки по одной, чтобы найти проблемный код

**Проблема**: Выбран неправильный Python-ядро

**Решение**:
1. Проверьте текущее ядро: `Kernel → Change Kernel`
2. Выберите правильную версию Python
3. Если ядро отсутствует, создайте его:
```bash
python -m ipykernel install --user --name=ml-env
```

**Проблема**: Ядро не запускается

**Решение**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Проблемы с ячейками в ноутбуке

**Проблема**: Ячейки выполняются, но не показывают вывод

**Решение**:
1. Проверьте, выполняется ли ячейка (ищите индикатор `[*]`)
2. Перезапустите ядро и выполните все ячейки: `Kernel → Restart & Run All`
3. Проверьте консоль браузера на наличие ошибок JavaScript (F12)

**Проблема**: Ячейки не запускаются - нет реакции при нажатии "Run"

**Решение**:
1. Проверьте, работает ли сервер Jupyter в терминале
2. Обновите страницу браузера
3. Закройте и снова откройте ноутбук
4. Перезапустите сервер Jupyter

---

## Проблемы с Python-пакетами

### Ошибки импорта

**Проблема**: `ModuleNotFoundError: No module named 'sklearn'`

**Решение**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Проблема**: `ImportError: cannot import name 'X' from 'sklearn'`

**Решение**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Конфликты версий

**Проблема**: Ошибки несовместимости версий пакетов

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

**Проблема**: `pip install` завершается с ошибками прав доступа

**Решение**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Проблемы с загрузкой данных

**Проблема**: `FileNotFoundError` при загрузке CSV-файлов

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

## Проблемы с окружением R

### Установка пакетов

**Проблема**: Установка пакетов завершается с ошибками компиляции

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

**Проблема**: `tidyverse` не устанавливается

**Решение**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Проблемы с RMarkdown

**Проблема**: RMarkdown не рендерится

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

## Проблемы с приложением для викторин

### Сборка и установка

**Проблема**: `npm install` завершается с ошибкой

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

**Проблема**: Порт 8080 уже используется

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

### Ошибки сборки

**Проблема**: `npm run build` завершается с ошибкой

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

**Проблема**: Ошибки линтинга мешают сборке

**Решение**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Проблемы с данными и путями к файлам

### Проблемы с путями

**Проблема**: Данные не найдены при запуске ноутбуков

**Решение**:
1. **Всегда запускайте ноутбуки из их директории**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Проверьте относительные пути в коде**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Используйте абсолютные пути, если необходимо**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Отсутствие файлов данных

**Проблема**: Файлы с данными отсутствуют

**Решение**:
1. Проверьте, должны ли данные быть в репозитории - большинство наборов данных включены
2. Некоторые уроки могут требовать загрузки данных - проверьте README урока
3. Убедитесь, что вы загрузили последние изменения:
   ```bash
   git pull origin main
   ```

---

## Распространенные сообщения об ошибках

### Ошибки памяти

**Ошибка**: `MemoryError` или ядро завершается при обработке данных

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

### Предупреждения о сходимости

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

### Проблемы с построением графиков

**Проблема**: Графики не отображаются в Jupyter

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

**Проблема**: Графики Seaborn выглядят иначе или вызывают ошибки

**Решение**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Ошибки кодировки/Unicode

**Проблема**: `UnicodeDecodeError` при чтении файлов

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

## Проблемы с производительностью

### Медленное выполнение ноутбуков

**Проблема**: Ноутбуки выполняются очень медленно

**Решение**:
1. **Перезапустите ядро, чтобы освободить память**: `Kernel → Restart`
2. **Закройте неиспользуемые ноутбуки**, чтобы освободить ресурсы
3. **Используйте меньшие выборки данных для тестирования**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Профилируйте ваш код**, чтобы найти узкие места:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Высокое использование памяти

**Проблема**: Система исчерпывает память

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

## Окружение и конфигурация

### Проблемы с виртуальным окружением

**Проблема**: Виртуальное окружение не активируется

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

**Проблема**: Пакеты установлены, но не обнаруживаются в ноутбуке

**Решение**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Проблемы с Git

**Проблема**: Не удается загрузить последние изменения - конфликты слияния

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

**Проблема**: Jupyter ноутбуки не открываются в VS Code

**Решение**:
1. Установите расширение Python в VS Code
2. Установите расширение Jupyter в VS Code
3. Выберите правильный интерпретатор Python: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Перезапустите VS Code

---

## Дополнительные ресурсы

- **Обсуждения на Discord**: [Задавайте вопросы и делитесь решениями в канале #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Модули "Машинное обучение для начинающих"](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Видеоуроки**: [Плейлист на YouTube](https://aka.ms/ml-beginners-videos)
- **Трекер проблем**: [Сообщить об ошибках](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Все еще есть проблемы?

Если вы попробовали предложенные решения и все еще сталкиваетесь с проблемами:

1. **Ищите существующие запросы**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Проверьте обсуждения в Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Откройте новый запрос**, указав:
   - Вашу операционную систему и версию
   - Версию Python/R
   - Сообщение об ошибке (полный стек вызовов)
   - Шаги для воспроизведения проблемы
   - Что вы уже попробовали

Мы готовы помочь! 🚀

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Несмотря на наши усилия обеспечить точность, автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.