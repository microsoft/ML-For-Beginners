<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:48:18+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "no"
}
-->
# Feilsøkingsguide

Denne guiden hjelper deg med å løse vanlige problemer når du jobber med Machine Learning for Beginners-kurset. Hvis du ikke finner en løsning her, kan du sjekke våre [Discord-diskusjoner](https://aka.ms/foundry/discord) eller [åpne en sak](https://github.com/microsoft/ML-For-Beginners/issues).

## Innholdsfortegnelse

- [Installasjonsproblemer](../..)
- [Problemer med Jupyter Notebook](../..)
- [Problemer med Python-pakker](../..)
- [Problemer med R-miljøet](../..)
- [Problemer med quiz-applikasjonen](../..)
- [Data- og filbaneproblemer](../..)
- [Vanlige feilmeldinger](../..)
- [Ytelsesproblemer](../..)
- [Miljø og konfigurasjon](../..)

---

## Installasjonsproblemer

### Python-installasjon

**Problem**: `python: command not found`

**Løsning**:
1. Installer Python 3.8 eller nyere fra [python.org](https://www.python.org/downloads/)
2. Verifiser installasjonen: `python --version` eller `python3 --version`
3. På macOS/Linux kan det hende du må bruke `python3` i stedet for `python`

**Problem**: Konflikter mellom flere Python-versjoner

**Løsning**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter-installasjon

**Problem**: `jupyter: command not found`

**Løsning**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter åpnes ikke i nettleseren

**Løsning**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R-installasjon

**Problem**: R-pakker installeres ikke

**Løsning**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel er ikke tilgjengelig i Jupyter

**Løsning**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemer med Jupyter Notebook

### Kernel-problemer

**Problem**: Kernel fortsetter å krasje eller starte på nytt

**Løsning**:
1. Start kernel på nytt: `Kernel → Restart`
2. Tøm utdata og start på nytt: `Kernel → Restart & Clear Output`
3. Sjekk for minneproblemer (se [Ytelsesproblemer](../..))
4. Prøv å kjøre celler individuelt for å identifisere problematisk kode

**Problem**: Feil Python-kernel valgt

**Løsning**:
1. Sjekk gjeldende kernel: `Kernel → Change Kernel`
2. Velg riktig Python-versjon
3. Hvis kernel mangler, opprett den:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Kernel starter ikke

**Løsning**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemer med notebook-celler

**Problem**: Celler kjører, men viser ingen utdata

**Løsning**:
1. Sjekk om cellen fortsatt kjører (se etter `[*]`-indikator)
2. Start kernel på nytt og kjør alle celler: `Kernel → Restart & Run All`
3. Sjekk nettleserkonsollen for JavaScript-feil (F12)

**Problem**: Kan ikke kjøre celler - ingen respons når du klikker "Run"

**Løsning**:
1. Sjekk om Jupyter-serveren fortsatt kjører i terminalen
2. Oppdater nettlesersiden
3. Lukk og åpne notebooken på nytt
4. Start Jupyter-serveren på nytt

---

## Problemer med Python-pakker

### Importfeil

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Løsning**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Løsning**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versjonskonflikter

**Problem**: Feil relatert til inkompatible pakkeversjoner

**Løsning**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problem**: `pip install` feiler med tillatelsesfeil

**Løsning**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemer med datalasting

**Problem**: `FileNotFoundError` ved lasting av CSV-filer

**Løsning**:
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

## Problemer med R-miljøet

### Pakkeinstallasjon

**Problem**: Pakkeinstallasjon feiler med kompilasjonsfeil

**Løsning**:
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

**Problem**: `tidyverse` installeres ikke

**Løsning**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemer med RMarkdown

**Problem**: RMarkdown rendrer ikke

**Løsning**:
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

## Problemer med quiz-applikasjonen

### Bygging og installasjon

**Problem**: `npm install` feiler

**Løsning**:
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

**Problem**: Port 8080 er allerede i bruk

**Løsning**:
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

### Byggefeil

**Problem**: `npm run build` feiler

**Løsning**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Linting-feil hindrer bygging

**Løsning**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data- og filbaneproblemer

### Banerelaterte problemer

**Problem**: Datafiler ikke funnet ved kjøring av notebooks

**Løsning**:
1. **Kjør alltid notebooks fra deres inneholdende mappe**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Sjekk relative baner i koden**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Bruk absolutte baner hvis nødvendig**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Manglende datafiler

**Problem**: Datasettfiler mangler

**Løsning**:
1. Sjekk om data skal være i repository - de fleste datasett er inkludert
2. Noen leksjoner kan kreve nedlasting av data - sjekk README for leksjonen
3. Sørg for at du har hentet de nyeste endringene:
   ```bash
   git pull origin main
   ```

---

## Vanlige feilmeldinger

### Minnefeil

**Feil**: `MemoryError` eller kernel krasjer ved databehandling

**Løsning**:
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

### Konvergeringsadvarsler

**Advarsel**: `ConvergenceWarning: Maximum number of iterations reached`

**Løsning**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemer med plotting

**Problem**: Plot vises ikke i Jupyter

**Løsning**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problem**: Seaborn-plots ser annerledes ut eller gir feil

**Løsning**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode-/kodingsfeil

**Problem**: `UnicodeDecodeError` ved lesing av filer

**Løsning**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Ytelsesproblemer

### Langsom kjøring av notebooks

**Problem**: Notebooks er veldig trege å kjøre

**Løsning**:
1. **Start kernel på nytt for å frigjøre minne**: `Kernel → Restart`
2. **Lukk ubrukte notebooks** for å frigjøre ressurser
3. **Bruk mindre datasett for testing**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profiler koden din** for å finne flaskehalser:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Høyt minneforbruk

**Problem**: Systemet går tom for minne

**Løsning**:
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

## Miljø og konfigurasjon

### Problemer med virtuelle miljøer

**Problem**: Virtuelt miljø aktiveres ikke

**Løsning**:
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

**Problem**: Pakker er installert, men ikke funnet i notebook

**Løsning**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git-problemer

**Problem**: Kan ikke hente de nyeste endringene - konflikt ved sammenslåing

**Løsning**:
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

### Integrasjon med VS Code

**Problem**: Jupyter-notebooks åpnes ikke i VS Code

**Løsning**:
1. Installer Python-utvidelsen i VS Code
2. Installer Jupyter-utvidelsen i VS Code
3. Velg riktig Python-tolk: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Start VS Code på nytt

---

## Tilleggsressurser

- **Discord-diskusjoner**: [Still spørsmål og del løsninger i #ml-for-beginners-kanalen](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners-moduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutorials**: [YouTube-spilleliste](https://aka.ms/ml-beginners-videos)
- **Feilsporing**: [Rapporter feil](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Har du fortsatt problemer?

Hvis du har prøvd løsningene ovenfor og fortsatt opplever problemer:

1. **Søk etter eksisterende saker**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Sjekk diskusjoner på Discord**: [Discord-diskusjoner](https://aka.ms/foundry/discord)
3. **Åpne en ny sak**: Inkluder:
   - Operativsystem og versjon
   - Python-/R-versjon
   - Feilmelding (full traceback)
   - Steg for å gjenskape problemet
   - Hva du allerede har prøvd

Vi er her for å hjelpe! 🚀

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.