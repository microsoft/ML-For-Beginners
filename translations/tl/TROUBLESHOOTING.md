<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:51:39+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "tl"
}
-->
# Gabay sa Pag-aayos ng Problema

Ang gabay na ito ay makakatulong sa iyo na lutasin ang mga karaniwang problema kapag ginagamit ang kurikulum ng Machine Learning for Beginners. Kung hindi mo makita ang solusyon dito, maaari mong bisitahin ang aming [Discord Discussions](https://aka.ms/foundry/discord) o [magbukas ng isyu](https://github.com/microsoft/ML-For-Beginners/issues).

## Talaan ng Nilalaman

- [Mga Isyu sa Pag-install](../..)
- [Mga Isyu sa Jupyter Notebook](../..)
- [Mga Isyu sa Python Package](../..)
- [Mga Isyu sa R Environment](../..)
- [Mga Isyu sa Quiz Application](../..)
- [Mga Isyu sa Data at File Path](../..)
- [Karaniwang Mensahe ng Error](../..)
- [Mga Isyu sa Performance](../..)
- [Kapaligiran at Konfigurasyon](../..)

---

## Mga Isyu sa Pag-install

### Pag-install ng Python

**Problema**: `python: command not found`

**Solusyon**:
1. I-install ang Python 3.8 o mas mataas mula sa [python.org](https://www.python.org/downloads/)
2. I-verify ang pag-install: `python --version` o `python3 --version`
3. Sa macOS/Linux, maaaring kailanganin mong gamitin ang `python3` sa halip na `python`

**Problema**: Maraming bersyon ng Python na nagdudulot ng salungatan

**Solusyon**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Pag-install ng Jupyter

**Problema**: `jupyter: command not found`

**Solusyon**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problema**: Hindi magbukas ang Jupyter sa browser

**Solusyon**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Pag-install ng R

**Problema**: Hindi ma-install ang mga R package

**Solusyon**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problema**: Hindi available ang IRkernel sa Jupyter

**Solusyon**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Mga Isyu sa Jupyter Notebook

### Mga Isyu sa Kernel

**Problema**: Patuloy na namamatay o nagre-restart ang kernel

**Solusyon**:
1. I-restart ang kernel: `Kernel → Restart`
2. I-clear ang output at i-restart: `Kernel → Restart & Clear Output`
3. Suriin ang mga isyu sa memorya (tingnan ang [Mga Isyu sa Performance](../..))
4. Subukang patakbuhin ang mga cell nang paisa-isa upang matukoy ang problemang code

**Problema**: Mali ang napiling Python kernel

**Solusyon**:
1. Suriin ang kasalukuyang kernel: `Kernel → Change Kernel`
2. Piliin ang tamang bersyon ng Python
3. Kung nawawala ang kernel, gumawa nito:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problema**: Hindi magbukas ang kernel

**Solusyon**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Mga Isyu sa Notebook Cell

**Problema**: Tumatakbo ang mga cell ngunit walang ipinapakitang output

**Solusyon**:
1. Suriin kung tumatakbo pa ang cell (hanapin ang `[*]` na indicator)
2. I-restart ang kernel at patakbuhin ang lahat ng cell: `Kernel → Restart & Run All`
3. Suriin ang console ng browser para sa mga error sa JavaScript (F12)

**Problema**: Hindi maipatakbo ang mga cell - walang tugon kapag kiniklik ang "Run"

**Solusyon**:
1. Suriin kung tumatakbo pa ang Jupyter server sa terminal
2. I-refresh ang pahina ng browser
3. Isara at muling buksan ang notebook
4. I-restart ang Jupyter server

---

## Mga Isyu sa Python Package

### Mga Error sa Import

**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Solusyon**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problema**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solusyon**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Mga Salungatan sa Bersyon

**Problema**: Mga error sa hindi pagkakatugma ng bersyon ng package

**Solusyon**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problema**: Nabigo ang `pip install` dahil sa mga error sa permiso

**Solusyon**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Mga Isyu sa Pag-load ng Data

**Problema**: `FileNotFoundError` kapag naglo-load ng mga CSV file

**Solusyon**:
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

## Mga Isyu sa R Environment

### Pag-install ng Package

**Problema**: Nabigo ang pag-install ng package dahil sa mga error sa compilation

**Solusyon**:
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

**Problema**: Hindi ma-install ang `tidyverse`

**Solusyon**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Mga Isyu sa RMarkdown

**Problema**: Hindi mag-render ang RMarkdown

**Solusyon**:
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

## Mga Isyu sa Quiz Application

### Pagbuo at Pag-install

**Problema**: Nabigo ang `npm install`

**Solusyon**:
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

**Problema**: Ang Port 8080 ay ginagamit na

**Solusyon**:
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

### Mga Error sa Pagbuo

**Problema**: Nabigo ang `npm run build`

**Solusyon**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problema**: Mga error sa linting na pumipigil sa pagbuo

**Solusyon**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Mga Isyu sa Data at File Path

### Mga Problema sa Path

**Problema**: Hindi mahanap ang mga data file kapag pinapatakbo ang mga notebook

**Solusyon**:
1. **Laging patakbuhin ang mga notebook mula sa kanilang direktoryo**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Suriin ang mga relative path sa code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Gumamit ng absolute path kung kinakailangan**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Nawawalang Mga Data File

**Problema**: Nawawala ang mga dataset file

**Solusyon**:
1. Suriin kung ang data ay dapat nasa repository - karamihan sa mga dataset ay kasama
2. Ang ilang mga aralin ay maaaring mangailangan ng pag-download ng data - tingnan ang README ng aralin
3. Siguraduhing na-pull mo ang pinakabagong mga pagbabago:
   ```bash
   git pull origin main
   ```

---

## Karaniwang Mensahe ng Error

### Mga Error sa Memorya

**Error**: `MemoryError` o namamatay ang kernel kapag nagpoproseso ng data

**Solusyon**:
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

### Mga Babala sa Convergence

**Warning**: `ConvergenceWarning: Maximum number of iterations reached`

**Solusyon**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Mga Isyu sa Pag-plot

**Problema**: Hindi nagpapakita ang mga plot sa Jupyter

**Solusyon**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problema**: Iba ang hitsura ng mga plot ng Seaborn o nagkakaroon ng mga error

**Solusyon**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Mga Error sa Unicode/Encoding

**Problema**: `UnicodeDecodeError` kapag nagbabasa ng mga file

**Solusyon**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Mga Isyu sa Performance

### Mabagal na Pagpapatakbo ng Notebook

**Problema**: Napakabagal tumakbo ang mga notebook

**Solusyon**:
1. **I-restart ang kernel upang magpalaya ng memorya**: `Kernel → Restart`
2. **Isara ang mga hindi ginagamit na notebook** upang magpalaya ng resources
3. **Gumamit ng mas maliit na sample ng data para sa testing**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **I-profile ang iyong code** upang matukoy ang mga bottleneck:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Mataas na Paggamit ng Memorya

**Problema**: Nauubos ang memorya ng sistema

**Solusyon**:
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

## Kapaligiran at Konfigurasyon

### Mga Isyu sa Virtual Environment

**Problema**: Hindi ma-activate ang virtual environment

**Solusyon**:
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

**Problema**: Na-install ang mga package ngunit hindi makita sa notebook

**Solusyon**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Mga Isyu sa Git

**Problema**: Hindi ma-pull ang pinakabagong mga pagbabago - mga salungatan sa merge

**Solusyon**:
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

### Integrasyon sa VS Code

**Problema**: Hindi magbukas ang mga Jupyter notebook sa VS Code

**Solusyon**:
1. I-install ang Python extension sa VS Code
2. I-install ang Jupyter extension sa VS Code
3. Piliin ang tamang Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. I-restart ang VS Code

---

## Karagdagang Mga Mapagkukunan

- **Discord Discussions**: [Magtanong at magbahagi ng mga solusyon sa #ml-for-beginners channel](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Mga module ng ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Mga Video Tutorial**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **Issue Tracker**: [Mag-ulat ng mga bug](https://github.com/microsoft/ML-For-Beginners/issues)

---

## May Problema Pa Rin?

Kung sinubukan mo na ang mga solusyon sa itaas ngunit patuloy pa rin ang problema:

1. **Hanapin ang mga umiiral na isyu**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Suriin ang mga talakayan sa Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Magbukas ng bagong isyu**: Isama ang:
   - Ang iyong operating system at bersyon
   - Bersyon ng Python/R
   - Mensahe ng error (buong traceback)
   - Mga hakbang upang ulitin ang problema
   - Ang mga solusyon na sinubukan mo na

Handa kaming tumulong! 🚀

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, mangyaring tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.