<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:53:08+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "cs"
}
-->
# Průvodce řešením problémů

Tento průvodce vám pomůže vyřešit běžné problémy při práci s kurikulem Machine Learning for Beginners. Pokud zde nenajdete řešení, podívejte se na naše [Diskuze na Discordu](https://aka.ms/foundry/discord) nebo [otevřete nový problém](https://github.com/microsoft/ML-For-Beginners/issues).

## Obsah

- [Problémy s instalací](../..)
- [Problémy s Jupyter Notebookem](../..)
- [Problémy s Python balíčky](../..)
- [Problémy s prostředím R](../..)
- [Problémy s aplikací kvízů](../..)
- [Problémy s daty a cestami k souborům](../..)
- [Běžné chybové zprávy](../..)
- [Problémy s výkonem](../..)
- [Prostředí a konfigurace](../..)

---

## Problémy s instalací

### Instalace Pythonu

**Problém**: `python: command not found`

**Řešení**:
1. Nainstalujte Python 3.8 nebo novější z [python.org](https://www.python.org/downloads/)
2. Ověřte instalaci: `python --version` nebo `python3 --version`
3. Na macOS/Linuxu možná budete muset použít `python3` místo `python`

**Problém**: Konflikty způsobené více verzemi Pythonu

**Řešení**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalace Jupyteru

**Problém**: `jupyter: command not found`

**Řešení**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problém**: Jupyter se nespustí v prohlížeči

**Řešení**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalace R

**Problém**: R balíčky se nenainstalují

**Řešení**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problém**: IRkernel není dostupný v Jupyteru

**Řešení**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problémy s Jupyter Notebookem

### Problémy s jádrem (Kernel)

**Problém**: Jádro se neustále ukončuje nebo restartuje

**Řešení**:
1. Restartujte jádro: `Kernel → Restart`
2. Vymažte výstup a restartujte: `Kernel → Restart & Clear Output`
3. Zkontrolujte problémy s pamětí (viz [Problémy s výkonem](../..))
4. Zkuste spouštět buňky jednotlivě, abyste identifikovali problematický kód

**Problém**: Vybráno špatné jádro Pythonu

**Řešení**:
1. Zkontrolujte aktuální jádro: `Kernel → Change Kernel`
2. Vyberte správnou verzi Pythonu
3. Pokud jádro chybí, vytvořte ho:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problém**: Jádro se nespustí

**Řešení**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problémy s buňkami v notebooku

**Problém**: Buňky se spouštějí, ale nezobrazují výstup

**Řešení**:
1. Zkontrolujte, zda buňka stále běží (hledáte indikátor `[*]`)
2. Restartujte jádro a spusťte všechny buňky: `Kernel → Restart & Run All`
3. Zkontrolujte konzoli prohlížeče na chyby JavaScriptu (F12)

**Problém**: Buňky nelze spustit - žádná odezva při kliknutí na "Run"

**Řešení**:
1. Zkontrolujte, zda Jupyter server stále běží v terminálu
2. Obnovte stránku v prohlížeči
3. Zavřete a znovu otevřete notebook
4. Restartujte Jupyter server

---

## Problémy s Python balíčky

### Chyby při importu

**Problém**: `ModuleNotFoundError: No module named 'sklearn'`

**Řešení**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problém**: `ImportError: cannot import name 'X' from 'sklearn'`

**Řešení**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflikty verzí

**Problém**: Chyby způsobené nekompatibilitou verzí balíčků

**Řešení**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problém**: `pip install` selže kvůli problémům s oprávněním

**Řešení**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problémy s načítáním dat

**Problém**: `FileNotFoundError` při načítání CSV souborů

**Řešení**:
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

## Problémy s prostředím R

### Instalace balíčků

**Problém**: Instalace balíčků selže kvůli chybám při kompilaci

**Řešení**:
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

**Problém**: `tidyverse` se nenainstaluje

**Řešení**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problémy s RMarkdown

**Problém**: RMarkdown se nevygeneruje

**Řešení**:
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

## Problémy s aplikací kvízů

### Sestavení a instalace

**Problém**: `npm install` selže

**Řešení**:
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

**Problém**: Port 8080 je již používán

**Řešení**:
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

### Chyby při sestavení

**Problém**: `npm run build` selže

**Řešení**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problém**: Chyby při lintování brání sestavení

**Řešení**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problémy s daty a cestami k souborům

### Problémy s cestami

**Problém**: Datové soubory nejsou nalezeny při spuštění notebooků

**Řešení**:
1. **Vždy spouštějte notebooky z jejich obsahujícího adresáře**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Zkontrolujte relativní cesty v kódu**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Použijte absolutní cesty, pokud je to nutné**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Chybějící datové soubory

**Problém**: Dataset soubory chybí

**Řešení**:
1. Zkontrolujte, zda data mají být v repozitáři - většina datasetů je zahrnuta
2. Některé lekce mohou vyžadovat stažení dat - zkontrolujte README lekce
3. Ujistěte se, že jste stáhli nejnovější změny:
   ```bash
   git pull origin main
   ```

---

## Běžné chybové zprávy

### Chyby paměti

**Chyba**: `MemoryError` nebo jádro se ukončí při zpracování dat

**Řešení**:
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

### Varování o konvergenci

**Varování**: `ConvergenceWarning: Maximum number of iterations reached`

**Řešení**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problémy s vykreslováním grafů

**Problém**: Grafy se nezobrazují v Jupyteru

**Řešení**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problém**: Grafy Seaborn vypadají jinak nebo vyvolávají chyby

**Řešení**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Chyby Unicode/kódování

**Problém**: `UnicodeDecodeError` při čtení souborů

**Řešení**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problémy s výkonem

### Pomalé spouštění notebooků

**Problém**: Notebooky běží velmi pomalu

**Řešení**:
1. **Restartujte jádro, abyste uvolnili paměť**: `Kernel → Restart`
2. **Zavřete nepoužívané notebooky**, abyste uvolnili zdroje
3. **Používejte menší vzorky dat pro testování**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilujte svůj kód**, abyste našli úzká místa:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Vysoké využití paměti

**Problém**: Systém dochází paměť

**Řešení**:
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

## Prostředí a konfigurace

### Problémy s virtuálním prostředím

**Problém**: Virtuální prostředí se neaktivuje

**Řešení**:
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

**Problém**: Balíčky jsou nainstalovány, ale nejsou nalezeny v notebooku

**Řešení**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problémy s Git

**Problém**: Nelze stáhnout nejnovější změny - konflikty při slučování

**Řešení**:
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

### Integrace s VS Code

**Problém**: Jupyter notebooky se neotevírají ve VS Code

**Řešení**:
1. Nainstalujte rozšíření Python ve VS Code
2. Nainstalujte rozšíření Jupyter ve VS Code
3. Vyberte správný interpret Pythonu: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Restartujte VS Code

---

## Další zdroje

- **Diskuze na Discordu**: [Pokládejte otázky a sdílejte řešení v kanálu #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduly ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video tutoriály**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Sledování problémů**: [Hlášení chyb](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Stále máte problémy?

Pokud jste vyzkoušeli výše uvedená řešení a stále máte problémy:

1. **Vyhledejte existující problémy**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Zkontrolujte diskuze na Discordu**: [Diskuze na Discordu](https://aka.ms/foundry/discord)
3. **Otevřete nový problém**: Uveďte:
   - Váš operační systém a jeho verzi
   - Verzi Pythonu/R
   - Chybovou zprávu (celý traceback)
   - Kroky k reprodukci problému
   - Co jste již vyzkoušeli

Jsme tu, abychom vám pomohli! 🚀

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby AI pro překlady [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.