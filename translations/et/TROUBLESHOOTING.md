<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-11T11:13:28+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "et"
}
-->
# Tõrkeotsingu juhend

See juhend aitab lahendada levinumaid probleeme, mis võivad tekkida algajatele mõeldud masinõppe õppekava kasutamisel. Kui siit lahendust ei leia, vaadake meie [Discordi arutelusid](https://aka.ms/foundry/discord) või [avage probleem](https://github.com/microsoft/ML-For-Beginners/issues).

## Sisukord

- [Paigaldusprobleemid](../..)
- [Jupyter Notebooki probleemid](../..)
- [Pythoni pakettide probleemid](../..)
- [R-keskkonna probleemid](../..)
- [Viktoriinirakenduse probleemid](../..)
- [Andmete ja failiteede probleemid](../..)
- [Levinud veateated](../..)
- [Jõudlusprobleemid](../..)
- [Keskkond ja konfiguratsioon](../..)

---

## Paigaldusprobleemid

### Pythoni paigaldamine

**Probleem**: `python: command not found`

**Lahendus**:
1. Paigalda Python 3.8 või uuem versioon [python.org](https://www.python.org/downloads/) lehelt.
2. Kontrolli paigaldust: `python --version` või `python3 --version`.
3. macOS/Linuxi puhul võib olla vajalik kasutada `python3` asemel `python`.

**Probleem**: Mitme Pythoni versiooni konfliktid

**Lahendus**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyteri paigaldamine

**Probleem**: `jupyter: command not found`

**Lahendus**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Probleem**: Jupyter ei avane brauseris

**Lahendus**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R-i paigaldamine

**Probleem**: R-pakette ei saa paigaldada

**Lahendus**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Probleem**: IRkernel pole Jupyteris saadaval

**Lahendus**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebooki probleemid

### Tuuma probleemid

**Probleem**: Tuum jookseb pidevalt kokku või taaskäivitub

**Lahendus**:
1. Taaskäivita tuum: `Kernel → Restart`.
2. Tühjenda väljund ja taaskäivita: `Kernel → Restart & Clear Output`.
3. Kontrolli mäluprobleeme (vt [Jõudlusprobleemid](../..)).
4. Proovi käivitada lahtrid ükshaaval, et tuvastada probleemne kood.

**Probleem**: Vale Pythoni tuum on valitud

**Lahendus**:
1. Kontrolli praegust tuuma: `Kernel → Change Kernel`.
2. Vali õige Pythoni versioon.
3. Kui tuum puudub, loo see:
```bash
python -m ipykernel install --user --name=ml-env
```

**Probleem**: Tuum ei käivitu

**Lahendus**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebooki lahtrite probleemid

**Probleem**: Lahtrid töötavad, kuid ei näita väljundit

**Lahendus**:
1. Kontrolli, kas lahter on endiselt töös (otsige `[*]` indikaatorit).
2. Taaskäivita tuum ja käivita kõik lahtrid: `Kernel → Restart & Run All`.
3. Kontrolli brauseri konsooli JavaScripti vigade osas (F12).

**Probleem**: Lahtrid ei tööta - "Run" klõpsamisel ei reageeri

**Lahendus**:
1. Kontrolli, kas Jupyteri server töötab endiselt terminalis.
2. Värskenda brauseri lehte.
3. Sulge ja ava notebook uuesti.
4. Taaskäivita Jupyteri server.

---

## Pythoni pakettide probleemid

### Importimise vead

**Probleem**: `ModuleNotFoundError: No module named 'sklearn'`

**Lahendus**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Probleem**: `ImportError: cannot import name 'X' from 'sklearn'`

**Lahendus**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versioonikonfliktid

**Probleem**: Paketi versiooni ühilduvuse vead

**Lahendus**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Probleem**: `pip install` ebaõnnestub õiguste vigade tõttu

**Lahendus**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Andmete laadimise probleemid

**Probleem**: `FileNotFoundError` CSV-failide laadimisel

**Lahendus**:
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

## R-keskkonna probleemid

### Pakettide paigaldamine

**Probleem**: Paketi paigaldamine ebaõnnestub kompileerimisvigade tõttu

**Lahendus**:
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

**Probleem**: `tidyverse` ei paigaldu

**Lahendus**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdowni probleemid

**Probleem**: RMarkdown ei renderdu

**Lahendus**:
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

## Viktoriinirakenduse probleemid

### Ehitamine ja paigaldamine

**Probleem**: `npm install` ebaõnnestub

**Lahendus**:
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

**Probleem**: Port 8080 on juba kasutusel

**Lahendus**:
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

### Ehitamisvead

**Probleem**: `npm run build` ebaõnnestub

**Lahendus**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Probleem**: Lintimise vead takistavad ehitamist

**Lahendus**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Andmete ja failiteede probleemid

### Teede probleemid

**Probleem**: Andmefaile ei leita, kui käivitatakse notebooke

**Lahendus**:
1. **Käivita notebooke alati nende kaustast**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Kontrolli koodis suhtelisi teid**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Kasuta vajadusel absoluutseid teid**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Puuduvad andmefailid

**Probleem**: Andmefailid puuduvad

**Lahendus**:
1. Kontrolli, kas andmed peaksid olema repositooriumis - enamik andmestikke on kaasas.
2. Mõned õppetunnid võivad nõuda andmete allalaadimist - vaata õppetunni README-d.
3. Veendu, et oled tõmmanud viimased muudatused:
   ```bash
   git pull origin main
   ```

---

## Levinud veateated

### Mäluga seotud vead

**Viga**: `MemoryError` või tuum jookseb andmete töötlemisel kokku

**Lahendus**:
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

### Konvergentsi hoiatused

**Hoiatus**: `ConvergenceWarning: Maximum number of iterations reached`

**Lahendus**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Graafikute probleemid

**Probleem**: Graafikud ei ilmu Jupyteris

**Lahendus**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Probleem**: Seaborn graafikud näevad erinevad välja või annavad vigu

**Lahendus**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode'i/kodeeringu vead

**Probleem**: `UnicodeDecodeError` failide lugemisel

**Lahendus**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Jõudlusprobleemid

### Aeglane notebookide täitmine

**Probleem**: Notebookid töötavad väga aeglaselt

**Lahendus**:
1. **Taaskäivita tuum, et vabastada mälu**: `Kernel → Restart`.
2. **Sulge kasutamata notebookid**, et vabastada ressursse.
3. **Kasuta testimiseks väiksemaid andmeproove**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profiili oma koodi**, et leida kitsaskohad:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Kõrge mälukasutus

**Probleem**: Süsteemil saab mälu otsa

**Lahendus**:
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

## Keskkond ja konfiguratsioon

### Virtuaalse keskkonna probleemid

**Probleem**: Virtuaalset keskkonda ei saa aktiveerida

**Lahendus**:
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

**Probleem**: Paketid on paigaldatud, kuid notebookis neid ei leita

**Lahendus**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git-i probleemid

**Probleem**: Ei saa viimaseid muudatusi tõmmata - liitmisvead

**Lahendus**:
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

### VS Code integratsioon

**Probleem**: Jupyter notebooke ei saa avada VS Code'is

**Lahendus**:
1. Paigalda VS Code'i Python laiendus.
2. Paigalda VS Code'i Jupyteri laiendus.
3. Vali õige Pythoni tõlk: `Ctrl+Shift+P` → "Python: Select Interpreter".
4. Taaskäivita VS Code.

---

## Lisamaterjalid

- **Discordi arutelud**: [Esita küsimusi ja jaga lahendusi #ml-for-beginners kanalil](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML algajatele moodulid](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutvustused**: [YouTube'i esitusloend](https://aka.ms/ml-beginners-videos)
- **Probleemide jälgija**: [Teata vigadest](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Kas probleemid püsivad?

Kui olete proovinud ülaltoodud lahendusi ja probleemid püsivad:

1. **Otsi olemasolevaid probleeme**: [GitHubi probleemid](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Kontrolli Discordi arutelusid**: [Discordi arutelud](https://aka.ms/foundry/discord)
3. **Ava uus probleem**: Lisa:
   - Teie operatsioonisüsteem ja versioon
   - Python/R versioon
   - Veateade (täielik jälg)
   - Sammud probleemi taastamiseks
   - Mida olete juba proovinud

Oleme siin, et aidata! 🚀

---

**Lahtiütlus**:  
See dokument on tõlgitud, kasutades AI tõlketeenust [Co-op Translator](https://github.com/Azure/co-op-translator). Kuigi püüame tagada täpsust, palun arvestage, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks lugeda autoriteetseks allikaks. Olulise teabe puhul on soovitatav kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valede tõlgenduste eest.