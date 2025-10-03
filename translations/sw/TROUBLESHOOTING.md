<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:52:10+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "sw"
}
-->
# Mwongozo wa Kutatua Shida

Mwongozo huu utakusaidia kutatua matatizo ya kawaida unayoweza kukutana nayo unapofanya kazi na mtaala wa Kujifunza Mashine kwa Kompyuta za Kuanza. Ikiwa huwezi kupata suluhisho hapa, tafadhali angalia [Majadiliano ya Discord](https://aka.ms/foundry/discord) au [fungua suala](https://github.com/microsoft/ML-For-Beginners/issues).

## Jedwali la Yaliyomo

- [Masuala ya Usakinishaji](../..)
- [Masuala ya Jupyter Notebook](../..)
- [Masuala ya Vifurushi vya Python](../..)
- [Masuala ya Mazingira ya R](../..)
- [Masuala ya Programu ya Maswali](../..)
- [Masuala ya Data na Njia za Faili](../..)
- [Ujumbe wa Makosa ya Kawaida](../..)
- [Masuala ya Utendaji](../..)
- [Mazingira na Usanidi](../..)

---

## Masuala ya Usakinishaji

### Usakinishaji wa Python

**Tatizo**: `python: command not found`

**Suluhisho**:
1. Sakinisha Python 3.8 au toleo la juu zaidi kutoka [python.org](https://www.python.org/downloads/)
2. Thibitisha usakinishaji: `python --version` au `python3 --version`
3. Kwenye macOS/Linux, huenda ukahitaji kutumia `python3` badala ya `python`

**Tatizo**: Matoleo mengi ya Python yanayosababisha migongano

**Suluhisho**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Usakinishaji wa Jupyter

**Tatizo**: `jupyter: command not found`

**Suluhisho**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Tatizo**: Jupyter haifunguki kwenye kivinjari

**Suluhisho**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Usakinishaji wa R

**Tatizo**: Vifurushi vya R havisakinishwi

**Suluhisho**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Tatizo**: IRkernel haipatikani kwenye Jupyter

**Suluhisho**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Masuala ya Jupyter Notebook

### Masuala ya Kernel

**Tatizo**: Kernel inazima au kuanzisha upya mara kwa mara

**Suluhisho**:
1. Anzisha upya kernel: `Kernel → Restart`
2. Futa matokeo na uanzishe upya: `Kernel → Restart & Clear Output`
3. Angalia masuala ya kumbukumbu (tazama [Masuala ya Utendaji](../..))
4. Jaribu kuendesha seli moja moja ili kubaini msimbo wenye tatizo

**Tatizo**: Kernel isiyo sahihi imechaguliwa

**Suluhisho**:
1. Angalia kernel ya sasa: `Kernel → Change Kernel`
2. Chagua toleo sahihi la Python
3. Ikiwa kernel haipo, tengeneza:
```bash
python -m ipykernel install --user --name=ml-env
```

**Tatizo**: Kernel haianzi

**Suluhisho**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Masuala ya Seli za Notebook

**Tatizo**: Seli zinaendesha lakini hazionyeshi matokeo

**Suluhisho**:
1. Angalia ikiwa seli bado inaendesha (angalia kiashiria cha `[*]`)
2. Anzisha upya kernel na uendeshe seli zote: `Kernel → Restart & Run All`
3. Angalia console ya kivinjari kwa makosa ya JavaScript (F12)

**Tatizo**: Seli haziwezi kuendeshwa - hakuna majibu unapobofya "Run"

**Suluhisho**:
1. Angalia ikiwa seva ya Jupyter bado inaendesha kwenye terminal
2. Pakia upya ukurasa wa kivinjari
3. Funga na ufungue tena notebook
4. Anzisha upya seva ya Jupyter

---

## Masuala ya Vifurushi vya Python

### Makosa ya Uingizaji

**Tatizo**: `ModuleNotFoundError: No module named 'sklearn'`

**Suluhisho**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Tatizo**: `ImportError: cannot import name 'X' from 'sklearn'`

**Suluhisho**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Migongano ya Matoleo

**Tatizo**: Makosa ya kutokubaliana kwa matoleo ya vifurushi

**Suluhisho**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Tatizo**: `pip install` inashindwa kwa makosa ya ruhusa

**Suluhisho**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Masuala ya Kupakia Data

**Tatizo**: `FileNotFoundError` wakati wa kupakia faili za CSV

**Suluhisho**:
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

## Masuala ya Mazingira ya R

### Usakinishaji wa Vifurushi

**Tatizo**: Usakinishaji wa kifurushi unashindwa kwa makosa ya uundaji

**Suluhisho**:
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

**Tatizo**: `tidyverse` haisakinishwi

**Suluhisho**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Masuala ya RMarkdown

**Tatizo**: RMarkdown haionyeshi

**Suluhisho**:
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

## Masuala ya Programu ya Maswali

### Ujenzi na Usakinishaji

**Tatizo**: `npm install` inashindwa

**Suluhisho**:
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

**Tatizo**: Bandari 8080 tayari inatumika

**Suluhisho**:
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

### Makosa ya Ujenzi

**Tatizo**: `npm run build` inashindwa

**Suluhisho**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Tatizo**: Makosa ya linting yanazuia ujenzi

**Suluhisho**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Masuala ya Data na Njia za Faili

### Masuala ya Njia

**Tatizo**: Faili za data hazipatikani wakati wa kuendesha notebooks

**Suluhisho**:
1. **Daima endesha notebooks kutoka kwenye folda yake**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Angalia njia za jamaa kwenye msimbo**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Tumia njia kamili ikiwa inahitajika**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Faili za Data Zinazokosekana

**Tatizo**: Faili za dataset zinakosekana

**Suluhisho**:
1. Angalia ikiwa data inapaswa kuwa kwenye hifadhi - datasets nyingi zimejumuishwa
2. Masomo mengine yanaweza kuhitaji kupakua data - angalia README ya somo
3. Hakikisha umepakua mabadiliko ya hivi karibuni:
   ```bash
   git pull origin main
   ```

---

## Ujumbe wa Makosa ya Kawaida

### Makosa ya Kumbukumbu

**Kosa**: `MemoryError` au kernel inakufa wakati wa kuchakata data

**Suluhisho**:
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

### Onyo la Mkutano

**Onyo**: `ConvergenceWarning: Maximum number of iterations reached`

**Suluhisho**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Masuala ya Kuchora Michoro

**Tatizo**: Michoro haionyeshwi kwenye Jupyter

**Suluhisho**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Tatizo**: Michoro ya Seaborn inaonekana tofauti au inatoa makosa

**Suluhisho**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Makosa ya Unicode/Kodifikesheni

**Tatizo**: `UnicodeDecodeError` wakati wa kusoma faili

**Suluhisho**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Masuala ya Utendaji

### Utekelezaji Polepole wa Notebook

**Tatizo**: Notebooks zinaendesha polepole sana

**Suluhisho**:
1. **Anzisha upya kernel ili kuachilia kumbukumbu**: `Kernel → Restart`
2. **Funga notebooks zisizotumika** ili kuachilia rasilimali
3. **Tumia sampuli ndogo za data kwa majaribio**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Pima msimbo wako** ili kubaini sehemu zenye matatizo:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Matumizi Makubwa ya Kumbukumbu

**Tatizo**: Mfumo unakosa kumbukumbu

**Suluhisho**:
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

## Mazingira na Usanidi

### Masuala ya Mazingira ya Kawaida

**Tatizo**: Mazingira ya kawaida hayafanyi kazi

**Suluhisho**:
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

**Tatizo**: Vifurushi vimesakinishwa lakini havipatikani kwenye notebook

**Suluhisho**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Masuala ya Git

**Tatizo**: Huwezi kupakua mabadiliko ya hivi karibuni - migogoro ya kuunganisha

**Suluhisho**:
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

### Ushirikiano wa VS Code

**Tatizo**: Jupyter notebooks hazifunguki kwenye VS Code

**Suluhisho**:
1. Sakinisha kiendelezi cha Python kwenye VS Code
2. Sakinisha kiendelezi cha Jupyter kwenye VS Code
3. Chagua tafsiri sahihi ya Python: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Anzisha upya VS Code

---

## Rasilimali za Ziada

- **Majadiliano ya Discord**: [Uliza maswali na shiriki suluhisho kwenye kituo cha #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduli za ML kwa Kompyuta](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Mafunzo ya Video**: [Orodha ya YouTube](https://aka.ms/ml-beginners-videos)
- **Kifuatiliaji cha Masuala**: [Ripoti hitilafu](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Bado Unakutana na Shida?

Ikiwa umejaribu suluhisho zilizo hapo juu na bado unakutana na matatizo:

1. **Tafuta masuala yaliyopo**: [Masuala ya GitHub](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Angalia majadiliano kwenye Discord**: [Majadiliano ya Discord](https://aka.ms/foundry/discord)
3. **Fungua suala jipya**: Jumuisha:
   - Mfumo wako wa uendeshaji na toleo lake
   - Toleo la Python/R
   - Ujumbe wa kosa (traceback kamili)
   - Hatua za kuzalisha tatizo
   - Kile ambacho tayari umejaribu

Tuko hapa kusaidia! 🚀

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.