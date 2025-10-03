<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:35:52+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "de"
}
-->
# Fehlerbehebungsanleitung

Diese Anleitung hilft Ihnen, häufige Probleme beim Arbeiten mit dem Machine Learning for Beginners-Lehrplan zu lösen. Sollten Sie hier keine Lösung finden, schauen Sie in unseren [Discord-Diskussionen](https://aka.ms/foundry/discord) vorbei oder [eröffnen Sie ein Issue](https://github.com/microsoft/ML-For-Beginners/issues).

## Inhaltsverzeichnis

- [Installationsprobleme](../..)
- [Probleme mit Jupyter Notebook](../..)
- [Probleme mit Python-Paketen](../..)
- [Probleme mit der R-Umgebung](../..)
- [Probleme mit der Quiz-Anwendung](../..)
- [Daten- und Dateipfadprobleme](../..)
- [Häufige Fehlermeldungen](../..)
- [Leistungsprobleme](../..)
- [Umgebung und Konfiguration](../..)

---

## Installationsprobleme

### Python-Installation

**Problem**: `python: command not found`

**Lösung**:
1. Installieren Sie Python 3.8 oder höher von [python.org](https://www.python.org/downloads/)
2. Überprüfen Sie die Installation: `python --version` oder `python3 --version`
3. Unter macOS/Linux müssen Sie möglicherweise `python3` anstelle von `python` verwenden

**Problem**: Mehrere Python-Versionen verursachen Konflikte

**Lösung**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter-Installation

**Problem**: `jupyter: command not found`

**Lösung**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter öffnet sich nicht im Browser

**Lösung**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R-Installation

**Problem**: R-Pakete lassen sich nicht installieren

**Lösung**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel ist in Jupyter nicht verfügbar

**Lösung**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Probleme mit Jupyter Notebook

### Kernel-Probleme

**Problem**: Kernel stürzt ab oder startet ständig neu

**Lösung**:
1. Kernel neu starten: `Kernel → Restart`
2. Ausgabe löschen und neu starten: `Kernel → Restart & Clear Output`
3. Überprüfen Sie Speicherprobleme (siehe [Leistungsprobleme](../..))
4. Führen Sie die Zellen einzeln aus, um problematischen Code zu identifizieren

**Problem**: Falscher Python-Kernel ausgewählt

**Lösung**:
1. Aktuellen Kernel überprüfen: `Kernel → Change Kernel`
2. Wählen Sie die richtige Python-Version aus
3. Falls der Kernel fehlt, erstellen Sie ihn:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Kernel startet nicht

**Lösung**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Probleme mit Notebook-Zellen

**Problem**: Zellen laufen, zeigen aber keine Ausgabe

**Lösung**:
1. Überprüfen Sie, ob die Zelle noch läuft (achten Sie auf den `[*]`-Indikator)
2. Kernel neu starten und alle Zellen ausführen: `Kernel → Restart & Run All`
3. Überprüfen Sie die Browser-Konsole auf JavaScript-Fehler (F12)

**Problem**: Zellen lassen sich nicht ausführen - keine Reaktion beim Klicken auf "Run"

**Lösung**:
1. Überprüfen Sie, ob der Jupyter-Server noch im Terminal läuft
2. Aktualisieren Sie die Browserseite
3. Schließen und öffnen Sie das Notebook erneut
4. Starten Sie den Jupyter-Server neu

---

## Probleme mit Python-Paketen

### Importfehler

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Lösung**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Lösung**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versionskonflikte

**Problem**: Fehler aufgrund von Paketversionsinkompatibilitäten

**Lösung**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problem**: `pip install` schlägt mit Berechtigungsfehlern fehl

**Lösung**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Probleme beim Laden von Daten

**Problem**: `FileNotFoundError` beim Laden von CSV-Dateien

**Lösung**:
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

## Probleme mit der R-Umgebung

### Paketinstallation

**Problem**: Paketinstallation schlägt mit Kompilierungsfehlern fehl

**Lösung**:
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

**Problem**: `tidyverse` lässt sich nicht installieren

**Lösung**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Probleme mit RMarkdown

**Problem**: RMarkdown lässt sich nicht rendern

**Lösung**:
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

## Probleme mit der Quiz-Anwendung

### Build und Installation

**Problem**: `npm install` schlägt fehl

**Lösung**:
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

**Problem**: Port 8080 ist bereits belegt

**Lösung**:
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

### Build-Fehler

**Problem**: `npm run build` schlägt fehl

**Lösung**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Linting-Fehler verhindern den Build

**Lösung**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Daten- und Dateipfadprobleme

### Pfadprobleme

**Problem**: Daten werden beim Ausführen von Notebooks nicht gefunden

**Lösung**:
1. **Führen Sie Notebooks immer aus ihrem Verzeichnis aus**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Überprüfen Sie relative Pfade im Code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Verwenden Sie bei Bedarf absolute Pfade**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Fehlende Datendateien

**Problem**: Datensätze fehlen

**Lösung**:
1. Überprüfen Sie, ob die Daten im Repository enthalten sein sollten – die meisten Datensätze sind enthalten
2. Einige Lektionen erfordern das Herunterladen von Daten – überprüfen Sie die README der Lektion
3. Stellen Sie sicher, dass Sie die neuesten Änderungen gezogen haben:
   ```bash
   git pull origin main
   ```

---

## Häufige Fehlermeldungen

### Speicherfehler

**Fehler**: `MemoryError` oder Kernel stürzt beim Verarbeiten von Daten ab

**Lösung**:
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

### Konvergenzwarnungen

**Warnung**: `ConvergenceWarning: Maximum number of iterations reached`

**Lösung**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Probleme mit Diagrammen

**Problem**: Diagramme werden in Jupyter nicht angezeigt

**Lösung**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problem**: Seaborn-Diagramme sehen anders aus oder werfen Fehler

**Lösung**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode-/Codierungsfehler

**Problem**: `UnicodeDecodeError` beim Lesen von Dateien

**Lösung**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Leistungsprobleme

### Langsame Ausführung von Notebooks

**Problem**: Notebooks laufen sehr langsam

**Lösung**:
1. **Kernel neu starten, um Speicher freizugeben**: `Kernel → Restart`
2. **Schließen Sie ungenutzte Notebooks**, um Ressourcen freizugeben
3. **Verwenden Sie kleinere Datensamples zum Testen**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilieren Sie Ihren Code**, um Engpässe zu finden:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Hoher Speicherverbrauch

**Problem**: System läuft aufgrund von Speichermangel langsam

**Lösung**:
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

## Umgebung und Konfiguration

### Probleme mit virtuellen Umgebungen

**Problem**: Virtuelle Umgebung wird nicht aktiviert

**Lösung**:
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

**Problem**: Pakete installiert, aber im Notebook nicht gefunden

**Lösung**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git-Probleme

**Problem**: Kann neueste Änderungen nicht ziehen – Merge-Konflikte

**Lösung**:
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

### VS Code-Integration

**Problem**: Jupyter-Notebooks lassen sich in VS Code nicht öffnen

**Lösung**:
1. Installieren Sie die Python-Erweiterung in VS Code
2. Installieren Sie die Jupyter-Erweiterung in VS Code
3. Wählen Sie den richtigen Python-Interpreter aus: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Starten Sie VS Code neu

---

## Zusätzliche Ressourcen

- **Discord-Diskussionen**: [Stellen Sie Fragen und teilen Sie Lösungen im #ml-for-beginners-Kanal](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners-Module](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video-Tutorials**: [YouTube-Playlist](https://aka.ms/ml-beginners-videos)
- **Issue-Tracker**: [Fehler melden](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Haben Sie immer noch Probleme?

Wenn Sie die oben genannten Lösungen ausprobiert haben und weiterhin Probleme auftreten:

1. **Suchen Sie nach bestehenden Issues**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Überprüfen Sie Diskussionen auf Discord**: [Discord-Diskussionen](https://aka.ms/foundry/discord)
3. **Eröffnen Sie ein neues Issue**: Geben Sie Folgendes an:
   - Ihr Betriebssystem und dessen Version
   - Python-/R-Version
   - Fehlermeldung (vollständiger Traceback)
   - Schritte zur Reproduktion des Problems
   - Was Sie bereits ausprobiert haben

Wir helfen Ihnen gerne weiter! 🚀

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.