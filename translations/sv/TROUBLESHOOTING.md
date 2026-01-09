<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:47:22+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "sv"
}
-->
# Felsökningsguide

Den här guiden hjälper dig att lösa vanliga problem när du arbetar med kursplanen för Machine Learning for Beginners. Om du inte hittar en lösning här, kolla gärna våra [Discord-diskussioner](https://aka.ms/foundry/discord) eller [öppna ett ärende](https://github.com/microsoft/ML-For-Beginners/issues).

## Innehållsförteckning

- [Installationsproblem](../..)
- [Problem med Jupyter Notebook](../..)
- [Problem med Python-paket](../..)
- [Problem med R-miljö](../..)
- [Problem med quiz-applikation](../..)
- [Problem med data och filvägar](../..)
- [Vanliga felmeddelanden](../..)
- [Prestandaproblem](../..)
- [Miljö och konfiguration](../..)

---

## Installationsproblem

### Python-installation

**Problem**: `python: command not found`

**Lösning**:
1. Installera Python 3.8 eller senare från [python.org](https://www.python.org/downloads/)
2. Verifiera installationen: `python --version` eller `python3 --version`
3. På macOS/Linux kan du behöva använda `python3` istället för `python`

**Problem**: Konflikter mellan flera Python-versioner

**Lösning**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter-installation

**Problem**: `jupyter: command not found`

**Lösning**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter öppnas inte i webbläsaren

**Lösning**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R-installation

**Problem**: R-paket installeras inte

**Lösning**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel är inte tillgängligt i Jupyter

**Lösning**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problem med Jupyter Notebook

### Kernel-problem

**Problem**: Kernel dör eller startar om hela tiden

**Lösning**:
1. Starta om kernel: `Kernel → Restart`
2. Rensa output och starta om: `Kernel → Restart & Clear Output`
3. Kontrollera minnesproblem (se [Prestandaproblem](../..))
4. Kör celler individuellt för att identifiera problematisk kod

**Problem**: Fel Python-kernel vald

**Lösning**:
1. Kontrollera aktuell kernel: `Kernel → Change Kernel`
2. Välj rätt Python-version
3. Om kernel saknas, skapa den:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Kernel startar inte

**Lösning**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problem med notebook-celler

**Problem**: Celler körs men visar ingen output

**Lösning**:
1. Kontrollera om cellen fortfarande körs (titta efter `[*]`-indikatorn)
2. Starta om kernel och kör alla celler: `Kernel → Restart & Run All`
3. Kontrollera webbläsarens konsol för JavaScript-fel (F12)

**Problem**: Kan inte köra celler - ingen respons vid klick på "Run"

**Lösning**:
1. Kontrollera om Jupyter-servern fortfarande körs i terminalen
2. Uppdatera webbläsarsidan
3. Stäng och öppna notebooken igen
4. Starta om Jupyter-servern

---

## Problem med Python-paket

### Importfel

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Lösning**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Lösning**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versionskonflikter

**Problem**: Felmeddelanden om paketversionskompatibilitet

**Lösning**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problem**: `pip install` misslyckas med behörighetsfel

**Lösning**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problem med dataladdning

**Problem**: `FileNotFoundError` vid laddning av CSV-filer

**Lösning**:
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

## Problem med R-miljö

### Paketinstallation

**Problem**: Paketinstallation misslyckas med kompileringsfel

**Lösning**:
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

**Problem**: `tidyverse` installeras inte

**Lösning**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problem med RMarkdown

**Problem**: RMarkdown renderas inte

**Lösning**:
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

## Problem med quiz-applikation

### Bygg och installation

**Problem**: `npm install` misslyckas

**Lösning**:
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

**Problem**: Port 8080 används redan

**Lösning**:
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

### Byggfel

**Problem**: `npm run build` misslyckas

**Lösning**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Lint-fel hindrar byggprocessen

**Lösning**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problem med data och filvägar

### Problem med filvägar

**Problem**: Datafiler hittas inte vid körning av notebooks

**Lösning**:
1. **Kör alltid notebooks från deras innehållande katalog**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Kontrollera relativa filvägar i koden**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Använd absoluta filvägar vid behov**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Saknade datafiler

**Problem**: Datasetfiler saknas

**Lösning**:
1. Kontrollera om data ska finnas i repositoryt - de flesta dataset är inkluderade
2. Vissa lektioner kan kräva nedladdning av data - kontrollera lektionens README
3. Se till att du har hämtat de senaste ändringarna:
   ```bash
   git pull origin main
   ```

---

## Vanliga felmeddelanden

### Minnesfel

**Fel**: `MemoryError` eller kernel dör vid databehandling

**Lösning**:
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

### Konvergensvarningar

**Varning**: `ConvergenceWarning: Maximum number of iterations reached`

**Lösning**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problem med diagram

**Problem**: Diagram visas inte i Jupyter

**Lösning**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problem**: Seaborn-diagram ser annorlunda ut eller ger fel

**Lösning**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode-/kodningsfel

**Problem**: `UnicodeDecodeError` vid läsning av filer

**Lösning**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Prestandaproblem

### Långsam körning av notebooks

**Problem**: Notebooks är mycket långsamma att köra

**Lösning**:
1. **Starta om kernel för att frigöra minne**: `Kernel → Restart`
2. **Stäng oanvända notebooks** för att frigöra resurser
3. **Använd mindre dataprover för testning**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilera din kod** för att hitta flaskhalsar:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Hög minnesanvändning

**Problem**: Systemet får slut på minne

**Lösning**:
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

## Miljö och konfiguration

### Problem med virtuella miljöer

**Problem**: Virtuell miljö aktiveras inte

**Lösning**:
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

**Problem**: Paket är installerade men hittas inte i notebook

**Lösning**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git-problem

**Problem**: Kan inte hämta senaste ändringar - konflikt vid sammanslagning

**Lösning**:
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

### Integration med VS Code

**Problem**: Jupyter-notebooks öppnas inte i VS Code

**Lösning**:
1. Installera Python-tillägget i VS Code
2. Installera Jupyter-tillägget i VS Code
3. Välj rätt Python-interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Starta om VS Code

---

## Ytterligare resurser

- **Discord-diskussioner**: [Ställ frågor och dela lösningar i kanalen #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners-moduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutorials**: [YouTube-spellista](https://aka.ms/ml-beginners-videos)
- **Ärendehantering**: [Rapportera buggar](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Har du fortfarande problem?

Om du har försökt lösningarna ovan och fortfarande har problem:

1. **Sök efter befintliga ärenden**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Kolla diskussioner på Discord**: [Discord-diskussioner](https://aka.ms/foundry/discord)
3. **Öppna ett nytt ärende**: Inkludera:
   - Ditt operativsystem och version
   - Python/R-version
   - Felmeddelande (full traceback)
   - Steg för att återskapa problemet
   - Vad du redan har försökt

Vi finns här för att hjälpa dig! 🚀

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.