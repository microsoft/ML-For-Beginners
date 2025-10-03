<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:44:59+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "pl"
}
-->
# Przewodnik rozwiązywania problemów

Ten przewodnik pomoże Ci rozwiązać najczęstsze problemy związane z korzystaniem z programu nauczania "Machine Learning for Beginners". Jeśli nie znajdziesz tutaj rozwiązania, sprawdź nasze [dyskusje na Discordzie](https://aka.ms/foundry/discord) lub [zgłoś problem](https://github.com/microsoft/ML-For-Beginners/issues).

## Spis treści

- [Problemy z instalacją](../..)
- [Problemy z Jupyter Notebook](../..)
- [Problemy z pakietami Pythona](../..)
- [Problemy ze środowiskiem R](../..)
- [Problemy z aplikacją quizową](../..)
- [Problemy z danymi i ścieżkami plików](../..)
- [Typowe komunikaty o błędach](../..)
- [Problemy z wydajnością](../..)
- [Środowisko i konfiguracja](../..)

---

## Problemy z instalacją

### Instalacja Pythona

**Problem**: `python: command not found`

**Rozwiązanie**:
1. Zainstaluj Python 3.8 lub nowszy z [python.org](https://www.python.org/downloads/)
2. Zweryfikuj instalację: `python --version` lub `python3 --version`
3. Na macOS/Linux możesz potrzebować użyć `python3` zamiast `python`

**Problem**: Konflikty spowodowane wieloma wersjami Pythona

**Rozwiązanie**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalacja Jupyter

**Problem**: `jupyter: command not found`

**Rozwiązanie**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter nie otwiera się w przeglądarce

**Rozwiązanie**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalacja R

**Problem**: Nie można zainstalować pakietów R

**Rozwiązanie**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel nie jest dostępny w Jupyter

**Rozwiązanie**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemy z Jupyter Notebook

### Problemy z jądrem (Kernel)

**Problem**: Jądro ciągle się restartuje lub przestaje działać

**Rozwiązanie**:
1. Zrestartuj jądro: `Kernel → Restart`
2. Wyczyść wyniki i zrestartuj: `Kernel → Restart & Clear Output`
3. Sprawdź problemy z pamięcią (zobacz [Problemy z wydajnością](../..))
4. Uruchamiaj komórki pojedynczo, aby zidentyfikować problematyczny kod

**Problem**: Wybrano niewłaściwe jądro Pythona

**Rozwiązanie**:
1. Sprawdź aktualne jądro: `Kernel → Change Kernel`
2. Wybierz odpowiednią wersję Pythona
3. Jeśli brakuje jądra, utwórz je:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Jądro nie chce się uruchomić

**Rozwiązanie**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemy z komórkami w notatniku

**Problem**: Komórki działają, ale nie pokazują wyników

**Rozwiązanie**:
1. Sprawdź, czy komórka nadal działa (szukaj wskaźnika `[*]`)
2. Zrestartuj jądro i uruchom wszystkie komórki: `Kernel → Restart & Run All`
3. Sprawdź konsolę przeglądarki pod kątem błędów JavaScript (F12)

**Problem**: Nie można uruchomić komórek - brak reakcji po kliknięciu "Run"

**Rozwiązanie**:
1. Sprawdź, czy serwer Jupyter nadal działa w terminalu
2. Odśwież stronę przeglądarki
3. Zamknij i ponownie otwórz notatnik
4. Zrestartuj serwer Jupyter

---

## Problemy z pakietami Pythona

### Błędy importu

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Rozwiązanie**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Rozwiązanie**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflikty wersji

**Problem**: Błędy niezgodności wersji pakietów

**Rozwiązanie**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problem**: `pip install` kończy się błędami uprawnień

**Rozwiązanie**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemy z ładowaniem danych

**Problem**: `FileNotFoundError` podczas ładowania plików CSV

**Rozwiązanie**:
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

## Problemy ze środowiskiem R

### Instalacja pakietów

**Problem**: Instalacja pakietów kończy się błędami kompilacji

**Rozwiązanie**:
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

**Problem**: Nie można zainstalować `tidyverse`

**Rozwiązanie**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemy z RMarkdown

**Problem**: RMarkdown nie renderuje się

**Rozwiązanie**:
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

## Problemy z aplikacją quizową

### Budowanie i instalacja

**Problem**: `npm install` kończy się błędem

**Rozwiązanie**:
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

**Problem**: Port 8080 jest już zajęty

**Rozwiązanie**:
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

### Błędy budowania

**Problem**: `npm run build` kończy się błędem

**Rozwiązanie**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Błędy lintingu uniemożliwiają budowanie

**Rozwiązanie**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemy z danymi i ścieżkami plików

### Problemy ze ścieżkami

**Problem**: Pliki danych nie są znajdowane podczas uruchamiania notatników

**Rozwiązanie**:
1. **Zawsze uruchamiaj notatniki z ich katalogu macierzystego**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Sprawdź ścieżki względne w kodzie**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Użyj ścieżek bezwzględnych, jeśli to konieczne**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Brakujące pliki danych

**Problem**: Brak plików z danymi

**Rozwiązanie**:
1. Sprawdź, czy dane powinny znajdować się w repozytorium - większość zestawów danych jest dołączona
2. Niektóre lekcje mogą wymagać pobrania danych - sprawdź README lekcji
3. Upewnij się, że masz najnowsze zmiany:
   ```bash
   git pull origin main
   ```

---

## Typowe komunikaty o błędach

### Błędy pamięci

**Błąd**: `MemoryError` lub jądro przestaje działać podczas przetwarzania danych

**Rozwiązanie**:
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

### Ostrzeżenia o zbieżności

**Ostrzeżenie**: `ConvergenceWarning: Maximum number of iterations reached`

**Rozwiązanie**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemy z wykresami

**Problem**: Wykresy nie wyświetlają się w Jupyter

**Rozwiązanie**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problem**: Wykresy Seaborn wyglądają inaczej lub generują błędy

**Rozwiązanie**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Błędy Unicode/kodowania

**Problem**: `UnicodeDecodeError` podczas odczytu plików

**Rozwiązanie**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problemy z wydajnością

### Wolne działanie notatników

**Problem**: Notatniki działają bardzo wolno

**Rozwiązanie**:
1. **Zrestartuj jądro, aby zwolnić pamięć**: `Kernel → Restart`
2. **Zamknij nieużywane notatniki**, aby zwolnić zasoby
3. **Używaj mniejszych próbek danych do testów**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profiluj swój kod**, aby znaleźć wąskie gardła:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Wysokie zużycie pamięci

**Problem**: Systemowi kończy się pamięć

**Rozwiązanie**:
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

## Środowisko i konfiguracja

### Problemy z wirtualnym środowiskiem

**Problem**: Wirtualne środowisko nie aktywuje się

**Rozwiązanie**:
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

**Problem**: Pakiety są zainstalowane, ale nie są znajdowane w notatniku

**Rozwiązanie**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemy z Git

**Problem**: Nie można pobrać najnowszych zmian - konflikty scalania

**Rozwiązanie**:
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

### Integracja z VS Code

**Problem**: Notatniki Jupyter nie otwierają się w VS Code

**Rozwiązanie**:
1. Zainstaluj rozszerzenie Python w VS Code
2. Zainstaluj rozszerzenie Jupyter w VS Code
3. Wybierz odpowiedni interpreter Pythona: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Zrestartuj VS Code

---

## Dodatkowe zasoby

- **Dyskusje na Discordzie**: [Zadawaj pytania i dziel się rozwiązaniami w kanale #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduły ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Samouczki wideo**: [Lista odtwarzania na YouTube](https://aka.ms/ml-beginners-videos)
- **Śledzenie problemów**: [Zgłaszaj błędy](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Nadal masz problemy?

Jeśli wypróbowałeś powyższe rozwiązania i nadal masz trudności:

1. **Przeszukaj istniejące zgłoszenia**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Sprawdź dyskusje na Discordzie**: [Dyskusje na Discordzie](https://aka.ms/foundry/discord)
3. **Otwórz nowe zgłoszenie**: Dołącz:
   - Twój system operacyjny i jego wersję
   - Wersję Pythona/R
   - Komunikat o błędzie (pełny traceback)
   - Kroki do odtworzenia problemu
   - Co już próbowałeś

Jesteśmy tu, aby pomóc! 🚀

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż staramy się zapewnić dokładność, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za źródło autorytatywne. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.