<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:49:16+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "nl"
}
-->
# Probleemoplossingsgids

Deze gids helpt je bij het oplossen van veelvoorkomende problemen bij het werken met het curriculum Machine Learning voor Beginners. Als je hier geen oplossing vindt, kijk dan op onze [Discord Discussies](https://aka.ms/foundry/discord) of [open een issue](https://github.com/microsoft/ML-For-Beginners/issues).

## Inhoudsopgave

- [Installatieproblemen](../..)
- [Jupyter Notebook Problemen](../..)
- [Python Pakket Problemen](../..)
- [R Omgevingsproblemen](../..)
- [Quiz Applicatie Problemen](../..)
- [Data- en Bestandspadproblemen](../..)
- [Veelvoorkomende Foutmeldingen](../..)
- [Prestatieproblemen](../..)
- [Omgeving en Configuratie](../..)

---

## Installatieproblemen

### Python Installatie

**Probleem**: `python: command not found`

**Oplossing**:
1. Installeer Python 3.8 of hoger via [python.org](https://www.python.org/downloads/)
2. Controleer de installatie: `python --version` of `python3 --version`
3. Op macOS/Linux moet je mogelijk `python3` gebruiken in plaats van `python`

**Probleem**: Meerdere Python-versies veroorzaken conflicten

**Oplossing**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter Installatie

**Probleem**: `jupyter: command not found`

**Oplossing**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Probleem**: Jupyter opent niet in de browser

**Oplossing**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R Installatie

**Probleem**: R-pakketten worden niet geïnstalleerd

**Oplossing**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Probleem**: IRkernel niet beschikbaar in Jupyter

**Oplossing**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook Problemen

### Kernel Problemen

**Probleem**: Kernel blijft crashen of opnieuw starten

**Oplossing**:
1. Herstart de kernel: `Kernel → Restart`
2. Wis output en herstart: `Kernel → Restart & Clear Output`
3. Controleer op geheugenproblemen (zie [Prestatieproblemen](../..))
4. Voer cellen afzonderlijk uit om problematische code te identificeren

**Probleem**: Verkeerde Python-kernel geselecteerd

**Oplossing**:
1. Controleer de huidige kernel: `Kernel → Change Kernel`
2. Selecteer de juiste Python-versie
3. Als de kernel ontbreekt, maak deze aan:
```bash
python -m ipykernel install --user --name=ml-env
```

**Probleem**: Kernel start niet

**Oplossing**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Celproblemen

**Probleem**: Cellen worden uitgevoerd maar tonen geen output

**Oplossing**:
1. Controleer of de cel nog wordt uitgevoerd (zoek naar de `[*]` indicator)
2. Herstart de kernel en voer alle cellen uit: `Kernel → Restart & Run All`
3. Controleer de browserconsole op JavaScript-fouten (F12)

**Probleem**: Kan cellen niet uitvoeren - geen reactie bij klikken op "Run"

**Oplossing**:
1. Controleer of de Jupyter-server nog actief is in de terminal
2. Vernieuw de browserpagina
3. Sluit en heropen het notebook
4. Herstart de Jupyter-server

---

## Python Pakket Problemen

### Importfouten

**Probleem**: `ModuleNotFoundError: No module named 'sklearn'`

**Oplossing**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Probleem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Oplossing**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versieconflicten

**Probleem**: Pakketversie incompatibiliteitsfouten

**Oplossing**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Probleem**: `pip install` mislukt met permissiefouten

**Oplossing**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Data Laadproblemen

**Probleem**: `FileNotFoundError` bij het laden van CSV-bestanden

**Oplossing**:
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

## R Omgevingsproblemen

### Pakketinstallatie

**Probleem**: Pakketinstallatie mislukt met compilatiefouten

**Oplossing**:
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

**Probleem**: `tidyverse` wordt niet geïnstalleerd

**Oplossing**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown Problemen

**Probleem**: RMarkdown wordt niet gerenderd

**Oplossing**:
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

## Quiz Applicatie Problemen

### Build en Installatie

**Probleem**: `npm install` mislukt

**Oplossing**:
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

**Probleem**: Poort 8080 is al in gebruik

**Oplossing**:
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

### Build Fouten

**Probleem**: `npm run build` mislukt

**Oplossing**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Probleem**: Linting fouten verhinderen build

**Oplossing**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data- en Bestandspadproblemen

### Padproblemen

**Probleem**: Data-bestanden niet gevonden bij het uitvoeren van notebooks

**Oplossing**:
1. **Voer notebooks altijd uit vanuit hun eigen map**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Controleer relatieve paden in de code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Gebruik absolute paden indien nodig**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Ontbrekende Data-bestanden

**Probleem**: Dataset-bestanden ontbreken

**Oplossing**:
1. Controleer of de data in de repository moet staan - de meeste datasets zijn inbegrepen
2. Sommige lessen vereisen het downloaden van data - controleer de README van de les
3. Zorg ervoor dat je de laatste wijzigingen hebt opgehaald:
   ```bash
   git pull origin main
   ```

---

## Veelvoorkomende Foutmeldingen

### Geheugenfouten

**Fout**: `MemoryError` of kernel crasht bij het verwerken van data

**Oplossing**:
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

### Convergentie Waarschuwingen

**Waarschuwing**: `ConvergenceWarning: Maximum number of iterations reached`

**Oplossing**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Plotproblemen

**Probleem**: Plots worden niet weergegeven in Jupyter

**Oplossing**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Probleem**: Seaborn-plots zien er anders uit of geven fouten

**Oplossing**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding Fouten

**Probleem**: `UnicodeDecodeError` bij het lezen van bestanden

**Oplossing**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Prestatieproblemen

### Langzame Notebook Uitvoering

**Probleem**: Notebooks zijn erg traag om uit te voeren

**Oplossing**:
1. **Herstart kernel om geheugen vrij te maken**: `Kernel → Restart`
2. **Sluit ongebruikte notebooks** om bronnen vrij te maken
3. **Gebruik kleinere datasamples voor testen**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profiel je code** om knelpunten te vinden:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Hoog Geheugenverbruik

**Probleem**: Systeem raakt geheugen kwijt

**Oplossing**:
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

## Omgeving en Configuratie

### Virtuele Omgevingsproblemen

**Probleem**: Virtuele omgeving wordt niet geactiveerd

**Oplossing**:
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

**Probleem**: Pakketten geïnstalleerd maar niet gevonden in notebook

**Oplossing**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git Problemen

**Probleem**: Kan laatste wijzigingen niet ophalen - merge conflicten

**Oplossing**:
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

### VS Code Integratie

**Probleem**: Jupyter notebooks openen niet in VS Code

**Oplossing**:
1. Installeer de Python-extensie in VS Code
2. Installeer de Jupyter-extensie in VS Code
3. Selecteer de juiste Python-interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Herstart VS Code

---

## Aanvullende Bronnen

- **Discord Discussies**: [Stel vragen en deel oplossingen in het #ml-for-beginners kanaal](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML voor Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutorials**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [Meld bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Nog steeds problemen?

Als je de bovenstaande oplossingen hebt geprobeerd en nog steeds problemen ondervindt:

1. **Zoek bestaande issues**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Controleer discussies op Discord**: [Discord Discussies](https://aka.ms/foundry/discord)
3. **Open een nieuw issue**: Voeg het volgende toe:
   - Je besturingssysteem en versie
   - Python/R versie
   - Foutmelding (volledige traceback)
   - Stappen om het probleem te reproduceren
   - Wat je al hebt geprobeerd

We staan klaar om te helpen! 🚀

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.