<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:47:49+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "da"
}
-->
# Fejlfindingsguide

Denne guide hjælper dig med at løse almindelige problemer, når du arbejder med Machine Learning for Beginners-kurset. Hvis du ikke finder en løsning her, kan du tjekke vores [Discord-diskussioner](https://aka.ms/foundry/discord) eller [oprette en sag](https://github.com/microsoft/ML-For-Beginners/issues).

## Indholdsfortegnelse

- [Installationsproblemer](../..)
- [Problemer med Jupyter Notebook](../..)
- [Problemer med Python-pakker](../..)
- [Problemer med R-miljøet](../..)
- [Problemer med quiz-applikationen](../..)
- [Problemer med data og filstier](../..)
- [Almindelige fejlmeddelelser](../..)
- [Ydelsesproblemer](../..)
- [Miljø og konfiguration](../..)

---

## Installationsproblemer

### Installation af Python

**Problem**: `python: command not found`

**Løsning**:
1. Installer Python 3.8 eller nyere fra [python.org](https://www.python.org/downloads/)
2. Bekræft installationen: `python --version` eller `python3 --version`
3. På macOS/Linux skal du muligvis bruge `python3` i stedet for `python`

**Problem**: Flere Python-versioner skaber konflikter

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

### Installation af Jupyter

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

**Problem**: Jupyter åbner ikke i browseren

**Løsning**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Installation af R

**Problem**: R-pakker installeres ikke

**Løsning**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel er ikke tilgængelig i Jupyter

**Løsning**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemer med Jupyter Notebook

### Kernelproblemer

**Problem**: Kernel bliver ved med at dø eller genstarte

**Løsning**:
1. Genstart kernelen: `Kernel → Restart`
2. Ryd output og genstart: `Kernel → Restart & Clear Output`
3. Tjek for hukommelsesproblemer (se [Ydelsesproblemer](../..))
4. Prøv at køre celler enkeltvis for at finde problematisk kode

**Problem**: Forkert Python-kernel valgt

**Løsning**:
1. Tjek den aktuelle kernel: `Kernel → Change Kernel`
2. Vælg den korrekte Python-version
3. Hvis kernel mangler, opret den:
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

**Problem**: Celler kører, men viser ikke output

**Løsning**:
1. Tjek om cellen stadig kører (se efter `[*]`-indikatoren)
2. Genstart kernel og kør alle celler: `Kernel → Restart & Run All`
3. Tjek browserens konsol for JavaScript-fejl (F12)

**Problem**: Kan ikke køre celler - ingen respons, når "Run" klikkes

**Løsning**:
1. Tjek om Jupyter-serveren stadig kører i terminalen
2. Opdater browserens side
3. Luk og genåbn notebooken
4. Genstart Jupyter-serveren

---

## Problemer med Python-pakker

### Importfejl

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

### Versionskonflikter

**Problem**: Fejl ved inkompatible pakkeversioner

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

**Problem**: `pip install` fejler med tilladelsesfejl

**Løsning**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemer med dataindlæsning

**Problem**: `FileNotFoundError` ved indlæsning af CSV-filer

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

### Installation af pakker

**Problem**: Installation af pakker fejler med kompileringsfejl

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

**Problem**: RMarkdown gengives ikke

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

## Problemer med quiz-applikationen

### Bygning og installation

**Problem**: `npm install` fejler

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

**Problem**: Port 8080 er allerede i brug

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

### Bygningsfejl

**Problem**: `npm run build` fejler

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

**Problem**: Lint-fejl forhindrer bygning

**Løsning**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemer med data og filstier

### Stiproblemer

**Problem**: Datafiler findes ikke, når notebooks køres

**Løsning**:
1. **Kør altid notebooks fra deres indeholdende mappe**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Tjek relative stier i koden**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Brug absolutte stier, hvis nødvendigt**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Manglende datafiler

**Problem**: Datasæt-filer mangler

**Løsning**:
1. Tjek om data skal være i repository - de fleste datasæt er inkluderet
2. Nogle lektioner kræver muligvis download af data - tjek lektionens README
3. Sørg for, at du har hentet de nyeste ændringer:
   ```bash
   git pull origin main
   ```

---

## Almindelige fejlmeddelelser

### Hukommelsesfejl

**Fejl**: `MemoryError` eller kernel dør under databehandling

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

### Konvergensadvarsler

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

### Problemer med plot

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

**Problem**: Seaborn-plots ser anderledes ud eller giver fejl

**Løsning**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode-/kodningsfejl

**Problem**: `UnicodeDecodeError` ved læsning af filer

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

## Ydelsesproblemer

### Langsom udførelse af notebooks

**Problem**: Notebooks kører meget langsomt

**Løsning**:
1. **Genstart kernel for at frigøre hukommelse**: `Kernel → Restart`
2. **Luk ubrugte notebooks** for at frigøre ressourcer
3. **Brug mindre datasæt til test**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilér din kode** for at finde flaskehalse:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Høj hukommelsesbrug

**Problem**: Systemet løber tør for hukommelse

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

## Miljø og konfiguration

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

**Problem**: Pakker er installeret, men findes ikke i notebook

**Løsning**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemer med Git

**Problem**: Kan ikke hente nyeste ændringer - sammenfletningskonflikter

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

### Integration med VS Code

**Problem**: Jupyter-notebooks åbner ikke i VS Code

**Løsning**:
1. Installer Python-udvidelsen i VS Code
2. Installer Jupyter-udvidelsen i VS Code
3. Vælg den korrekte Python-fortolker: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Genstart VS Code

---

## Yderligere ressourcer

- **Discord-diskussioner**: [Stil spørgsmål og del løsninger i #ml-for-beginners-kanalen](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners-moduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutorials**: [YouTube-playliste](https://aka.ms/ml-beginners-videos)
- **Fejlsporing**: [Rapportér fejl](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Stadig problemer?

Hvis du har prøvet løsningerne ovenfor og stadig oplever problemer:

1. **Søg efter eksisterende sager**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Tjek diskussioner på Discord**: [Discord-diskussioner](https://aka.ms/foundry/discord)
3. **Opret en ny sag**: Inkluder:
   - Dit operativsystem og version
   - Python/R-version
   - Fejlmeddelelse (fuld traceback)
   - Trin til at genskabe problemet
   - Hvad du allerede har prøvet

Vi er her for at hjælpe! 🚀

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.