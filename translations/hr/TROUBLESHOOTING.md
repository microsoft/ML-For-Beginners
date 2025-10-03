<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:55:42+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "hr"
}
-->
# Vodič za rješavanje problema

Ovaj vodič pomaže u rješavanju uobičajenih problema pri radu s kurikulumom "Strojno učenje za početnike". Ako ovdje ne pronađete rješenje, provjerite naše [Discord rasprave](https://aka.ms/foundry/discord) ili [otvorite problem](https://github.com/microsoft/ML-For-Beginners/issues).

## Sadržaj

- [Problemi s instalacijom](../..)
- [Problemi s Jupyter Notebookom](../..)
- [Problemi s Python paketima](../..)
- [Problemi s R okruženjem](../..)
- [Problemi s aplikacijom za kviz](../..)
- [Problemi s podacima i putanjama datoteka](../..)
- [Uobičajene poruke o greškama](../..)
- [Problemi s performansama](../..)
- [Okruženje i konfiguracija](../..)

---

## Problemi s instalacijom

### Instalacija Pythona

**Problem**: `python: command not found`

**Rješenje**:
1. Instalirajte Python 3.8 ili noviji s [python.org](https://www.python.org/downloads/)
2. Provjerite instalaciju: `python --version` ili `python3 --version`
3. Na macOS/Linuxu možda ćete morati koristiti `python3` umjesto `python`

**Problem**: Konflikti zbog više verzija Pythona

**Rješenje**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalacija Jupytera

**Problem**: `jupyter: command not found`

**Rješenje**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problem**: Jupyter se ne otvara u pregledniku

**Rješenje**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalacija R-a

**Problem**: R paketi se ne instaliraju

**Rješenje**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problem**: IRkernel nije dostupan u Jupyteru

**Rješenje**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemi s Jupyter Notebookom

### Problemi s kernelom

**Problem**: Kernel se stalno gasi ili ponovno pokreće

**Rješenje**:
1. Ponovno pokrenite kernel: `Kernel → Restart`
2. Očistite izlaz i ponovno pokrenite: `Kernel → Restart & Clear Output`
3. Provjerite probleme s memorijom (pogledajte [Problemi s performansama](../..))
4. Pokušajte pokrenuti ćelije pojedinačno kako biste identificirali problematični kod

**Problem**: Odabran je pogrešan Python kernel

**Rješenje**:
1. Provjerite trenutni kernel: `Kernel → Change Kernel`
2. Odaberite ispravnu verziju Pythona
3. Ako kernel nedostaje, kreirajte ga:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problem**: Kernel se ne pokreće

**Rješenje**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemi s ćelijama u notebooku

**Problem**: Ćelije se izvršavaju, ali ne prikazuju izlaz

**Rješenje**:
1. Provjerite je li ćelija još uvijek u procesu izvršavanja (potražite indikator `[*]`)
2. Ponovno pokrenite kernel i izvršite sve ćelije: `Kernel → Restart & Run All`
3. Provjerite konzolu preglednika za JavaScript greške (F12)

**Problem**: Ćelije se ne mogu pokrenuti - nema odgovora na klik "Run"

**Rješenje**:
1. Provjerite radi li Jupyter server još uvijek u terminalu
2. Osvježite stranicu u pregledniku
3. Zatvorite i ponovno otvorite notebook
4. Ponovno pokrenite Jupyter server

---

## Problemi s Python paketima

### Greške pri uvozu

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Rješenje**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Rješenje**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflikti verzija

**Problem**: Greške zbog nekompatibilnosti verzija paketa

**Rješenje**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problem**: `pip install` ne uspijeva zbog grešaka s dozvolama

**Rješenje**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemi s učitavanjem podataka

**Problem**: `FileNotFoundError` pri učitavanju CSV datoteka

**Rješenje**:
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

## Problemi s R okruženjem

### Instalacija paketa

**Problem**: Instalacija paketa ne uspijeva zbog grešaka pri kompilaciji

**Rješenje**:
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

**Problem**: `tidyverse` se ne instalira

**Rješenje**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemi s RMarkdownom

**Problem**: RMarkdown se ne renderira

**Rješenje**:
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

## Problemi s aplikacijom za kviz

### Izgradnja i instalacija

**Problem**: `npm install` ne uspijeva

**Rješenje**:
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

**Problem**: Port 8080 je već zauzet

**Rješenje**:
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

### Greške pri izgradnji

**Problem**: `npm run build` ne uspijeva

**Rješenje**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problem**: Greške pri provjeri sintakse sprječavaju izgradnju

**Rješenje**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemi s podacima i putanjama datoteka

### Problemi s putanjama

**Problem**: Datoteke s podacima nisu pronađene pri pokretanju notebooka

**Rješenje**:
1. **Uvijek pokrećite notebooke iz direktorija u kojem se nalaze**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Provjerite relativne putanje u kodu**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Koristite apsolutne putanje ako je potrebno**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Nedostaju datoteke s podacima

**Problem**: Datoteke s datasetima nedostaju

**Rješenje**:
1. Provjerite trebaju li podaci biti u repozitoriju - većina datasetova je uključena
2. Neke lekcije mogu zahtijevati preuzimanje podataka - provjerite README lekcije
3. Provjerite jeste li povukli najnovije promjene:
   ```bash
   git pull origin main
   ```

---

## Uobičajene poruke o greškama

### Greške s memorijom

**Greška**: `MemoryError` ili kernel se gasi pri obradi podataka

**Rješenje**:
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

### Upozorenja o konvergenciji

**Upozorenje**: `ConvergenceWarning: Maximum number of iterations reached`

**Rješenje**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemi s grafovima

**Problem**: Grafovi se ne prikazuju u Jupyteru

**Rješenje**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problem**: Grafovi iz Seaborna izgledaju drugačije ili uzrokuju greške

**Rješenje**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Encoding greške

**Problem**: `UnicodeDecodeError` pri čitanju datoteka

**Rješenje**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problemi s performansama

### Sporo izvršavanje notebooka

**Problem**: Notebooki se vrlo sporo izvršavaju

**Rješenje**:
1. **Ponovno pokrenite kernel kako biste oslobodili memoriju**: `Kernel → Restart`
2. **Zatvorite nepotrebne notebooke** kako biste oslobodili resurse
3. **Koristite manje uzorke podataka za testiranje**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilirajte svoj kod** kako biste pronašli uska grla:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Visoka potrošnja memorije

**Problem**: Sustav ostaje bez memorije

**Rješenje**:
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

## Okruženje i konfiguracija

### Problemi s virtualnim okruženjem

**Problem**: Virtualno okruženje se ne aktivira

**Rješenje**:
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

**Problem**: Paketi su instalirani, ali nisu dostupni u notebooku

**Rješenje**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemi s Gitom

**Problem**: Ne mogu povući najnovije promjene - konflikti pri spajanju

**Rješenje**:
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

### Integracija s VS Codeom

**Problem**: Jupyter notebooki se ne otvaraju u VS Codeu

**Rješenje**:
1. Instalirajte Python ekstenziju u VS Codeu
2. Instalirajte Jupyter ekstenziju u VS Codeu
3. Odaberite ispravan Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Ponovno pokrenite VS Code

---

## Dodatni resursi

- **Discord rasprave**: [Postavljajte pitanja i dijelite rješenja u kanalu #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduli za ML za početnike](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video tutorijali**: [YouTube playlist](https://aka.ms/ml-beginners-videos)
- **Praćenje problema**: [Prijavite greške](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Još uvijek imate problema?

Ako ste isprobali gore navedena rješenja i još uvijek imate poteškoća:

1. **Pretražite postojeće probleme**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Provjerite rasprave na Discordu**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Otvorite novi problem**: Uključite:
   - Vaš operativni sustav i verziju
   - Python/R verziju
   - Poruku o grešci (cijeli traceback)
   - Korake za reprodukciju problema
   - Što ste već pokušali

Tu smo da pomognemo! 🚀

---

**Izjava o odricanju odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za nesporazume ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.