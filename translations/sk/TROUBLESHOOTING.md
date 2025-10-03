<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:53:38+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "sk"
}
-->
# Príručka na riešenie problémov

Táto príručka vám pomôže vyriešiť bežné problémy pri práci s učebnými osnovami Machine Learning for Beginners. Ak tu nenájdete riešenie, pozrite si naše [Diskusie na Discorde](https://aka.ms/foundry/discord) alebo [otvorte problém](https://github.com/microsoft/ML-For-Beginners/issues).

## Obsah

- [Problémy s inštaláciou](../..)
- [Problémy s Jupyter Notebookom](../..)
- [Problémy s balíčkami Pythonu](../..)
- [Problémy s prostredím R](../..)
- [Problémy s aplikáciou kvízu](../..)
- [Problémy s dátami a cestami k súborom](../..)
- [Bežné chybové hlásenia](../..)
- [Problémy s výkonom](../..)
- [Prostredie a konfigurácia](../..)

---

## Problémy s inštaláciou

### Inštalácia Pythonu

**Problém**: `python: command not found`

**Riešenie**:
1. Nainštalujte Python 3.8 alebo vyšší z [python.org](https://www.python.org/downloads/)
2. Overte inštaláciu: `python --version` alebo `python3 --version`
3. Na macOS/Linux môže byť potrebné použiť `python3` namiesto `python`

**Problém**: Konflikty spôsobené viacerými verziami Pythonu

**Riešenie**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Inštalácia Jupyteru

**Problém**: `jupyter: command not found`

**Riešenie**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problém**: Jupyter sa neotvorí v prehliadači

**Riešenie**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Inštalácia R

**Problém**: Balíčky R sa nedajú nainštalovať

**Riešenie**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problém**: IRkernel nie je dostupný v Jupyteri

**Riešenie**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problémy s Jupyter Notebookom

### Problémy s kernelom

**Problém**: Kernel sa neustále vypína alebo reštartuje

**Riešenie**:
1. Reštartujte kernel: `Kernel → Restart`
2. Vymažte výstup a reštartujte: `Kernel → Restart & Clear Output`
3. Skontrolujte problémy s pamäťou (pozrite [Problémy s výkonom](../..))
4. Skúste spustiť bunky jednotlivo, aby ste identifikovali problematický kód

**Problém**: Vybraný nesprávny kernel Pythonu

**Riešenie**:
1. Skontrolujte aktuálny kernel: `Kernel → Change Kernel`
2. Vyberte správnu verziu Pythonu
3. Ak kernel chýba, vytvorte ho:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problém**: Kernel sa nespustí

**Riešenie**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problémy s bunkami notebooku

**Problém**: Bunky sa spúšťajú, ale nezobrazujú výstup

**Riešenie**:
1. Skontrolujte, či bunka stále beží (hľadajte indikátor `[*]`)
2. Reštartujte kernel a spustite všetky bunky: `Kernel → Restart & Run All`
3. Skontrolujte konzolu prehliadača na chyby JavaScriptu (F12)

**Problém**: Bunky sa nedajú spustiť - žiadna odozva pri kliknutí na "Run"

**Riešenie**:
1. Skontrolujte, či server Jupyter stále beží v termináli
2. Obnovte stránku v prehliadači
3. Zatvorte a znova otvorte notebook
4. Reštartujte server Jupyter

---

## Problémy s balíčkami Pythonu

### Chyby pri importe

**Problém**: `ModuleNotFoundError: No module named 'sklearn'`

**Riešenie**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problém**: `ImportError: cannot import name 'X' from 'sklearn'`

**Riešenie**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflikty verzií

**Problém**: Chyby nekompatibility verzií balíčkov

**Riešenie**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problém**: `pip install` zlyháva s chybami povolení

**Riešenie**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problémy s načítaním dát

**Problém**: `FileNotFoundError` pri načítaní CSV súborov

**Riešenie**:
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

## Problémy s prostredím R

### Inštalácia balíčkov

**Problém**: Inštalácia balíčkov zlyháva s chybami kompilácie

**Riešenie**:
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

**Problém**: `tidyverse` sa nedá nainštalovať

**Riešenie**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problémy s RMarkdown

**Problém**: RMarkdown sa nedá vykresliť

**Riešenie**:
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

## Problémy s aplikáciou kvízu

### Build a inštalácia

**Problém**: `npm install` zlyháva

**Riešenie**:
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

**Problém**: Port 8080 je už obsadený

**Riešenie**:
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

### Chyby pri buildovaní

**Problém**: `npm run build` zlyháva

**Riešenie**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problém**: Chyby lintovania bránia buildovaniu

**Riešenie**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problémy s dátami a cestami k súborom

### Problémy s cestami

**Problém**: Dáta sa nenašli pri spúšťaní notebookov

**Riešenie**:
1. **Vždy spúšťajte notebooky z ich obsahujúceho adresára**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Skontrolujte relatívne cesty v kóde**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Použite absolútne cesty, ak je to potrebné**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Chýbajúce dátové súbory

**Problém**: Dataset súbory chýbajú

**Riešenie**:
1. Skontrolujte, či dáta majú byť v repozitári - väčšina datasetov je zahrnutá
2. Niektoré lekcie môžu vyžadovať stiahnutie dát - pozrite README lekcie
3. Uistite sa, že ste stiahli najnovšie zmeny:
   ```bash
   git pull origin main
   ```

---

## Bežné chybové hlásenia

### Chyby pamäte

**Chyba**: `MemoryError` alebo kernel sa vypne pri spracovaní dát

**Riešenie**:
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

### Varovania o konvergencii

**Varovanie**: `ConvergenceWarning: Maximum number of iterations reached`

**Riešenie**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problémy s vykresľovaním grafov

**Problém**: Grafy sa nezobrazujú v Jupyteri

**Riešenie**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problém**: Grafy Seaborn vyzerajú inak alebo spôsobujú chyby

**Riešenie**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Chyby Unicode/kódovania

**Problém**: `UnicodeDecodeError` pri čítaní súborov

**Riešenie**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problémy s výkonom

### Pomalé spúšťanie notebookov

**Problém**: Notebooky sa spúšťajú veľmi pomaly

**Riešenie**:
1. **Reštartujte kernel na uvoľnenie pamäte**: `Kernel → Restart`
2. **Zatvorte nepoužívané notebooky**, aby ste uvoľnili zdroje
3. **Použite menšie vzorky dát na testovanie**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilujte svoj kód**, aby ste našli úzke miesta:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Vysoké využitie pamäte

**Problém**: Systém má nedostatok pamäte

**Riešenie**:
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

## Prostredie a konfigurácia

### Problémy s virtuálnym prostredím

**Problém**: Virtuálne prostredie sa neaktivuje

**Riešenie**:
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

**Problém**: Balíčky sú nainštalované, ale nie sú dostupné v notebooku

**Riešenie**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problémy s Gitom

**Problém**: Nedajú sa stiahnuť najnovšie zmeny - konflikty pri zlúčení

**Riešenie**:
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

### Integrácia s VS Code

**Problém**: Jupyter notebooky sa neotvárajú vo VS Code

**Riešenie**:
1. Nainštalujte rozšírenie Python vo VS Code
2. Nainštalujte rozšírenie Jupyter vo VS Code
3. Vyberte správny interpret Pythonu: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Reštartujte VS Code

---

## Ďalšie zdroje

- **Diskusie na Discorde**: [Pýtajte sa otázky a zdieľajte riešenia v kanáli #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduly ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video tutoriály**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Sledovanie problémov**: [Nahláste chyby](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Stále máte problémy?

Ak ste vyskúšali vyššie uvedené riešenia a stále máte problémy:

1. **Vyhľadajte existujúce problémy**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Skontrolujte diskusie na Discorde**: [Diskusie na Discorde](https://aka.ms/foundry/discord)
3. **Otvorte nový problém**: Uveďte:
   - Váš operačný systém a jeho verziu
   - Verziu Pythonu/R
   - Chybové hlásenie (celý traceback)
   - Kroky na reprodukciu problému
   - Čo ste už vyskúšali

Sme tu, aby sme vám pomohli! 🚀

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.