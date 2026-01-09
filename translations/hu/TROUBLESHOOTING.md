<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:52:39+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "hu"
}
-->
# Hibaelhárítási útmutató

Ez az útmutató segít megoldani a gyakori problémákat a Machine Learning for Beginners tananyag használata során. Ha itt nem talál megoldást, nézze meg a [Discord Beszélgetéseket](https://aka.ms/foundry/discord) vagy [nyisson egy hibajegyet](https://github.com/microsoft/ML-For-Beginners/issues).

## Tartalomjegyzék

- [Telepítési problémák](../..)
- [Jupyter Notebook problémák](../..)
- [Python csomag problémák](../..)
- [R környezet problémák](../..)
- [Kvíz alkalmazás problémák](../..)
- [Adat- és fájlútvonal problémák](../..)
- [Gyakori hibaüzenetek](../..)
- [Teljesítmény problémák](../..)
- [Környezet és konfiguráció](../..)

---

## Telepítési problémák

### Python telepítés

**Probléma**: `python: command not found`

**Megoldás**:
1. Telepítse a Python 3.8 vagy újabb verzióját innen: [python.org](https://www.python.org/downloads/)
2. Ellenőrizze a telepítést: `python --version` vagy `python3 --version`
3. macOS/Linux rendszeren lehet, hogy `python3`-at kell használnia `python` helyett

**Probléma**: Több Python verzió konfliktust okoz

**Megoldás**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter telepítés

**Probléma**: `jupyter: command not found`

**Megoldás**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Probléma**: A Jupyter nem indul el a böngészőben

**Megoldás**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R telepítés

**Probléma**: R csomagok nem települnek

**Megoldás**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Probléma**: Az IRkernel nem érhető el a Jupyterben

**Megoldás**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook problémák

### Kernel problémák

**Probléma**: A kernel folyamatosan leáll vagy újraindul

**Megoldás**:
1. Indítsa újra a kernelt: `Kernel → Restart`
2. Törölje a kimenetet és indítsa újra: `Kernel → Restart & Clear Output`
3. Ellenőrizze a memória problémákat (lásd [Teljesítmény problémák](../..))
4. Próbálja meg egyenként futtatni a cellákat, hogy azonosítsa a problémás kódot

**Probléma**: Rossz Python kernel van kiválasztva

**Megoldás**:
1. Ellenőrizze az aktuális kernelt: `Kernel → Change Kernel`
2. Válassza ki a megfelelő Python verziót
3. Ha a kernel hiányzik, hozza létre:
```bash
python -m ipykernel install --user --name=ml-env
```

**Probléma**: A kernel nem indul el

**Megoldás**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook cella problémák

**Probléma**: A cellák futnak, de nem jelenítenek meg kimenetet

**Megoldás**:
1. Ellenőrizze, hogy a cella még fut-e (keresse a `[*]` jelzést)
2. Indítsa újra a kernelt és futtassa az összes cellát: `Kernel → Restart & Run All`
3. Ellenőrizze a böngésző konzolt JavaScript hibákért (F12)

**Probléma**: Nem lehet futtatni a cellákat - nincs válasz a "Run" gombra kattintva

**Megoldás**:
1. Ellenőrizze, hogy a Jupyter szerver még fut-e a terminálban
2. Frissítse a böngésző oldalt
3. Zárja be és nyissa meg újra a notebookot
4. Indítsa újra a Jupyter szervert

---

## Python csomag problémák

### Import hibák

**Probléma**: `ModuleNotFoundError: No module named 'sklearn'`

**Megoldás**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Probléma**: `ImportError: cannot import name 'X' from 'sklearn'`

**Megoldás**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Verzió konfliktusok

**Probléma**: Csomag verzió összeférhetetlenségi hibák

**Megoldás**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Probléma**: `pip install` engedélyezési hibákkal meghiúsul

**Megoldás**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Adatbetöltési problémák

**Probléma**: `FileNotFoundError` CSV fájlok betöltésekor

**Megoldás**:
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

## R környezet problémák

### Csomag telepítés

**Probléma**: A csomag telepítése fordítási hibákkal meghiúsul

**Megoldás**:
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

**Probléma**: A `tidyverse` nem települ

**Megoldás**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown problémák

**Probléma**: Az RMarkdown nem jelenik meg

**Megoldás**:
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

## Kvíz alkalmazás problémák

### Építés és telepítés

**Probléma**: `npm install` meghiúsul

**Megoldás**:
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

**Probléma**: A 8080-as port már használatban van

**Megoldás**:
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

### Építési hibák

**Probléma**: `npm run build` meghiúsul

**Megoldás**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Probléma**: Linting hibák akadályozzák az építést

**Megoldás**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Adat- és fájlútvonal problémák

### Útvonal problémák

**Probléma**: Az adatfájlok nem találhatók a notebookok futtatásakor

**Megoldás**:
1. **Mindig a notebookok tartalmazó könyvtárából futtassa őket**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Ellenőrizze a relatív útvonalakat a kódban**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Használjon abszolút útvonalakat, ha szükséges**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Hiányzó adatfájlok

**Probléma**: Hiányoznak az adatfájlok

**Megoldás**:
1. Ellenőrizze, hogy az adatoknak a repóban kell-e lenniük - a legtöbb dataset benne van
2. Néhány lecke megkövetelheti az adatok letöltését - nézze meg a lecke README fájlját
3. Győződjön meg róla, hogy a legfrissebb változásokat lehúzta:
   ```bash
   git pull origin main
   ```

---

## Gyakori hibaüzenetek

### Memória hibák

**Hiba**: `MemoryError` vagy a kernel leáll adatfeldolgozás közben

**Megoldás**:
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

### Konvergencia figyelmeztetések

**Figyelmeztetés**: `ConvergenceWarning: Maximum number of iterations reached`

**Megoldás**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Ábrázolási problémák

**Probléma**: Az ábrák nem jelennek meg a Jupyterben

**Megoldás**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Probléma**: A Seaborn ábrák eltérően néznek ki vagy hibát dobnak

**Megoldás**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/kódolási hibák

**Probléma**: `UnicodeDecodeError` fájlok olvasásakor

**Megoldás**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Teljesítmény problémák

### Lassú notebook futtatás

**Probléma**: A notebookok nagyon lassan futnak

**Megoldás**:
1. **Indítsa újra a kernelt a memória felszabadításához**: `Kernel → Restart`
2. **Zárja be a nem használt notebookokat** a források felszabadításához
3. **Használjon kisebb adatmintákat teszteléshez**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilozza a kódját**, hogy megtalálja a szűk keresztmetszeteket:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Magas memóriahasználat

**Probléma**: A rendszer kifogy a memóriából

**Megoldás**:
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

## Környezet és konfiguráció

### Virtuális környezet problémák

**Probléma**: A virtuális környezet nem aktiválódik

**Megoldás**:
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

**Probléma**: A csomagok telepítve vannak, de nem találhatók a notebookban

**Megoldás**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git problémák

**Probléma**: Nem lehet lehúzni a legfrissebb változásokat - összeolvasztási konfliktusok

**Megoldás**:
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

### VS Code integráció

**Probléma**: A Jupyter notebookok nem nyílnak meg a VS Code-ban

**Megoldás**:
1. Telepítse a Python kiterjesztést a VS Code-ban
2. Telepítse a Jupyter kiterjesztést a VS Code-ban
3. Válassza ki a megfelelő Python értelmezőt: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Indítsa újra a VS Code-ot

---

## További források

- **Discord Beszélgetések**: [Tegyen fel kérdéseket és ossza meg megoldásait a #ml-for-beginners csatornán](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners modulok](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videó oktatóanyagok**: [YouTube lejátszási lista](https://aka.ms/ml-beginners-videos)
- **Hibajegy követő**: [Jelentsen hibákat](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Még mindig problémái vannak?

Ha kipróbálta a fenti megoldásokat, és még mindig problémákat tapasztal:

1. **Keressen meglévő hibajegyeket**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Nézze meg a Discord beszélgetéseket**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Nyisson egy új hibajegyet**: Tartalmazza:
   - Az operációs rendszerét és verzióját
   - Python/R verzióját
   - Hibaüzenetet (teljes traceback)
   - A probléma reprodukálásának lépéseit
   - Amit már kipróbált

Segítünk! 🚀

---

**Felelősség kizárása**:  
Ez a dokumentum az [Co-op Translator](https://github.com/Azure/co-op-translator) AI fordítási szolgáltatás segítségével került lefordításra. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.