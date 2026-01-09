<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:56:09+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "sl"
}
-->
# Vodnik za odpravljanje težav

Ta vodnik vam pomaga rešiti pogoste težave pri delu s kurikulumom Strojnega učenja za začetnike. Če rešitve tukaj ne najdete, preverite naše [Discord razprave](https://aka.ms/foundry/discord) ali [odprite težavo](https://github.com/microsoft/ML-For-Beginners/issues).

## Kazalo

- [Težave z namestitvijo](../..)
- [Težave z Jupyter Notebook](../..)
- [Težave s Python paketi](../..)
- [Težave z okoljem R](../..)
- [Težave z aplikacijo za kvize](../..)
- [Težave s podatki in potmi do datotek](../..)
- [Pogosta sporočila o napakah](../..)
- [Težave z zmogljivostjo](../..)
- [Okolje in konfiguracija](../..)

---

## Težave z namestitvijo

### Namestitev Pythona

**Težava**: `python: command not found`

**Rešitev**:
1. Namestite Python 3.8 ali novejšo različico s [python.org](https://www.python.org/downloads/)
2. Preverite namestitev: `python --version` ali `python3 --version`
3. Na macOS/Linux morda potrebujete ukaz `python3` namesto `python`

**Težava**: Konflikti zaradi več nameščenih različic Pythona

**Rešitev**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Namestitev Jupyterja

**Težava**: `jupyter: command not found`

**Rešitev**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Težava**: Jupyter se ne odpre v brskalniku

**Rešitev**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Namestitev R

**Težava**: R paketov ni mogoče namestiti

**Rešitev**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Težava**: IRkernel ni na voljo v Jupyterju

**Rešitev**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Težave z Jupyter Notebook

### Težave s kernelom

**Težava**: Kernel se nenehno ustavlja ali znova zažene

**Rešitev**:
1. Znova zaženite kernel: `Kernel → Restart`
2. Počistite izhod in znova zaženite: `Kernel → Restart & Clear Output`
3. Preverite težave s pomnilnikom (glejte [Težave z zmogljivostjo](../..))
4. Poskusite zagnati celice posamezno, da ugotovite problematično kodo

**Težava**: Izbran je napačen Python kernel

**Rešitev**:
1. Preverite trenutni kernel: `Kernel → Change Kernel`
2. Izberite pravilno različico Pythona
3. Če kernela ni, ga ustvarite:
```bash
python -m ipykernel install --user --name=ml-env
```

**Težava**: Kernel se ne zažene

**Rešitev**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Težave s celicami v beležnici

**Težava**: Celice se izvajajo, vendar ne prikazujejo izhoda

**Rešitev**:
1. Preverite, ali se celica še vedno izvaja (poiščite indikator `[*]`)
2. Znova zaženite kernel in zaženite vse celice: `Kernel → Restart & Run All`
3. Preverite konzolo brskalnika za napake v JavaScriptu (F12)

**Težava**: Celic ni mogoče zagnati - brez odziva ob kliku na "Run"

**Rešitev**:
1. Preverite, ali Jupyter strežnik še vedno deluje v terminalu
2. Osvežite stran v brskalniku
3. Zaprite in znova odprite beležnico
4. Znova zaženite Jupyter strežnik

---

## Težave s Python paketi

### Napake pri uvozu

**Težava**: `ModuleNotFoundError: No module named 'sklearn'`

**Rešitev**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Težava**: `ImportError: cannot import name 'X' from 'sklearn'`

**Rešitev**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflikti različic

**Težava**: Napake zaradi nezdružljivosti različic paketov

**Rešitev**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Težava**: `pip install` ne uspe zaradi težav s pravicami

**Rešitev**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Težave z nalaganjem podatkov

**Težava**: `FileNotFoundError` pri nalaganju CSV datotek

**Rešitev**:
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

## Težave z okoljem R

### Namestitev paketov

**Težava**: Namestitev paketa ne uspe zaradi napak pri prevajanju

**Rešitev**:
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

**Težava**: `tidyverse` ni mogoče namestiti

**Rešitev**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Težave z RMarkdown

**Težava**: RMarkdown se ne upodobi

**Rešitev**:
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

## Težave z aplikacijo za kvize

### Gradnja in namestitev

**Težava**: `npm install` ne uspe

**Rešitev**:
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

**Težava**: Vrata 8080 so že v uporabi

**Rešitev**:
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

### Napake pri gradnji

**Težava**: `npm run build` ne uspe

**Rešitev**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Težava**: Napake pri preverjanju kode preprečujejo gradnjo

**Rešitev**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Težave s podatki in potmi do datotek

### Težave s potmi

**Težava**: Podatkovne datoteke niso najdene pri zagonu beležnic

**Rešitev**:
1. **Vedno zaženite beležnice iz njihove vsebujoče mape**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Preverite relativne poti v kodi**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Po potrebi uporabite absolutne poti**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Manjkajoče podatkovne datoteke

**Težava**: Manjkajo datoteke z nabori podatkov

**Rešitev**:
1. Preverite, ali bi morali biti podatki v repozitoriju - večina naborov podatkov je vključena
2. Nekatere lekcije morda zahtevajo prenos podatkov - preverite README lekcije
3. Prepričajte se, da ste povlekli najnovejše spremembe:
   ```bash
   git pull origin main
   ```

---

## Pogosta sporočila o napakah

### Napake s pomnilnikom

**Napaka**: `MemoryError` ali kernel preneha delovati pri obdelavi podatkov

**Rešitev**:
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

### Opozorila o konvergenci

**Opozorilo**: `ConvergenceWarning: Maximum number of iterations reached`

**Rešitev**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Težave s prikazovanjem grafov

**Težava**: Grafi se ne prikazujejo v Jupyterju

**Rešitev**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Težava**: Grafi Seaborn izgledajo drugače ali povzročajo napake

**Rešitev**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Napake Unicode/kodiranja

**Težava**: `UnicodeDecodeError` pri branju datotek

**Rešitev**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Težave z zmogljivostjo

### Počasno izvajanje beležnic

**Težava**: Beležnice se izvajajo zelo počasi

**Rešitev**:
1. **Znova zaženite kernel za sprostitev pomnilnika**: `Kernel → Restart`
2. **Zaprite neuporabljene beležnice**, da sprostite vire
3. **Za testiranje uporabite manjše vzorce podatkov**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilirajte svojo kodo**, da najdete ozka grla:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Visoka poraba pomnilnika

**Težava**: Sistem ostaja brez pomnilnika

**Rešitev**:
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

## Okolje in konfiguracija

### Težave z virtualnim okoljem

**Težava**: Virtualno okolje se ne aktivira

**Rešitev**:
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

**Težava**: Paketi so nameščeni, vendar jih beležnica ne najde

**Rešitev**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Težave z Gitom

**Težava**: Ni mogoče povleči najnovejših sprememb - konflikti združevanja

**Rešitev**:
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

### Integracija z VS Code

**Težava**: Jupyter beležnic ni mogoče odpreti v VS Code

**Rešitev**:
1. Namestite razširitev Python v VS Code
2. Namestite razširitev Jupyter v VS Code
3. Izberite pravilen Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Znova zaženite VS Code

---

## Dodatni viri

- **Discord razprave**: [Postavite vprašanja in delite rešitve v kanalu #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduli ML za začetnike](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video vadnice**: [YouTube seznam predvajanja](https://aka.ms/ml-beginners-videos)
- **Sledilnik težav**: [Prijavite napake](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Še vedno imate težave?

Če ste preizkusili zgornje rešitve in še vedno naletite na težave:

1. **Poiščite obstoječe težave**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Preverite razprave na Discordu**: [Discord razprave](https://aka.ms/foundry/discord)
3. **Odprite novo težavo**: Vključite:
   - Vaš operacijski sistem in različico
   - Različico Pythona/R
   - Sporočilo o napaki (celoten izpis napake)
   - Korake za ponovitev težave
   - Kaj ste že poskusili

Tukaj smo, da vam pomagamo! 🚀

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki bi nastale zaradi uporabe tega prevoda.