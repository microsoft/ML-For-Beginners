<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:57:45+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "lt"
}
-->
# Trikčių šalinimo vadovas

Šis vadovas padės išspręsti dažniausiai pasitaikančias problemas, susijusias su „Machine Learning for Beginners“ mokymo programa. Jei čia nerandate sprendimo, apsilankykite mūsų [Discord diskusijose](https://aka.ms/foundry/discord) arba [atidarykite problemą](https://github.com/microsoft/ML-For-Beginners/issues).

## Turinys

- [Diegimo problemos](../..)
- [Jupyter Notebook problemos](../..)
- [Python paketų problemos](../..)
- [R aplinkos problemos](../..)
- [Klausimynų programos problemos](../..)
- [Duomenų ir failų kelių problemos](../..)
- [Dažnos klaidų žinutės](../..)
- [Našumo problemos](../..)
- [Aplinka ir konfigūracija](../..)

---

## Diegimo problemos

### Python diegimas

**Problema**: `python: command not found`

**Sprendimas**:
1. Įdiekite Python 3.8 ar naujesnę versiją iš [python.org](https://www.python.org/downloads/)
2. Patikrinkite diegimą: `python --version` arba `python3 --version`
3. macOS/Linux sistemose gali reikėti naudoti `python3` vietoj `python`

**Problema**: Konfliktai dėl kelių Python versijų

**Sprendimas**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter diegimas

**Problema**: `jupyter: command not found`

**Sprendimas**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problema**: Jupyter neatidaro naršyklėje

**Sprendimas**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R diegimas

**Problema**: R paketai neįdiegiami

**Sprendimas**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problema**: IRkernel nėra Jupyter aplinkoje

**Sprendimas**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook problemos

### Branduolio problemos

**Problema**: Branduolys nuolat miršta arba persikrauna

**Sprendimas**:
1. Perkraukite branduolį: `Kernel → Restart`
2. Išvalykite išvestį ir perkraukite: `Kernel → Restart & Clear Output`
3. Patikrinkite atminties problemas (žr. [Našumo problemos](../..))
4. Bandykite vykdyti langelius po vieną, kad nustatytumėte probleminį kodą

**Problema**: Pasirinktas neteisingas Python branduolys

**Sprendimas**:
1. Patikrinkite esamą branduolį: `Kernel → Change Kernel`
2. Pasirinkite tinkamą Python versiją
3. Jei branduolio nėra, sukurkite jį:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problema**: Branduolys neprasideda

**Sprendimas**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook langelių problemos

**Problema**: Langeliai vykdomi, bet nerodo išvesties

**Sprendimas**:
1. Patikrinkite, ar langelis vis dar vykdomas (ieškokite `[*]` indikatoriaus)
2. Perkraukite branduolį ir vykdykite visus langelius: `Kernel → Restart & Run All`
3. Patikrinkite naršyklės konsolę dėl JavaScript klaidų (F12)

**Problema**: Negalima vykdyti langelių – nėra reakcijos paspaudus „Run“

**Sprendimas**:
1. Patikrinkite, ar Jupyter serveris vis dar veikia terminale
2. Atnaujinkite naršyklės puslapį
3. Uždarykite ir vėl atidarykite užrašų knygelę
4. Perkraukite Jupyter serverį

---

## Python paketų problemos

### Importavimo klaidos

**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Sprendimas**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problema**: `ImportError: cannot import name 'X' from 'sklearn'`

**Sprendimas**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versijų konfliktai

**Problema**: Paketų versijų nesuderinamumo klaidos

**Sprendimas**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problema**: `pip install` nepavyksta dėl leidimų klaidų

**Sprendimas**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Duomenų įkėlimo problemos

**Problema**: `FileNotFoundError` įkeliant CSV failus

**Sprendimas**:
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

## R aplinkos problemos

### Paketų diegimas

**Problema**: Paketų diegimas nepavyksta dėl kompiliavimo klaidų

**Sprendimas**:
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

**Problema**: `tidyverse` neįdiegiama

**Sprendimas**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown problemos

**Problema**: RMarkdown nerenderuoja

**Sprendimas**:
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

## Klausimynų programos problemos

### Kūrimas ir diegimas

**Problema**: `npm install` nepavyksta

**Sprendimas**:
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

**Problema**: Uostas 8080 jau naudojamas

**Sprendimas**:
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

### Kūrimo klaidos

**Problema**: `npm run build` nepavyksta

**Sprendimas**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problema**: Linting klaidos trukdo kurti

**Sprendimas**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Duomenų ir failų kelių problemos

### Kelių problemos

**Problema**: Duomenų failai nerandami vykdant užrašų knygeles

**Sprendimas**:
1. **Visada vykdykite užrašų knygeles iš jų aplanko**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Patikrinkite santykinius kelius kode**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Naudokite absoliučius kelius, jei reikia**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Trūkstami duomenų failai

**Problema**: Trūksta duomenų rinkinių failų

**Sprendimas**:
1. Patikrinkite, ar duomenys turėtų būti saugykloje – dauguma duomenų rinkinių yra įtraukti
2. Kai kurios pamokos gali reikalauti atsisiųsti duomenis – patikrinkite pamokos README
3. Įsitikinkite, kad atsisiuntėte naujausius pakeitimus:
   ```bash
   git pull origin main
   ```

---

## Dažnos klaidų žinutės

### Atminties klaidos

**Klaida**: `MemoryError` arba branduolys miršta apdorojant duomenis

**Sprendimas**:
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

### Konvergencijos įspėjimai

**Įspėjimas**: `ConvergenceWarning: Maximum number of iterations reached`

**Sprendimas**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Vaizdavimo problemos

**Problema**: Grafikai nerodomi Jupyter aplinkoje

**Sprendimas**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problema**: Seaborn grafikai atrodo kitaip arba meta klaidas

**Sprendimas**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/kodavimo klaidos

**Problema**: `UnicodeDecodeError` skaitant failus

**Sprendimas**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Našumo problemos

### Lėtas užrašų knygelių vykdymas

**Problema**: Užrašų knygelės vykdomos labai lėtai

**Sprendimas**:
1. **Perkraukite branduolį, kad atlaisvintumėte atmintį**: `Kernel → Restart`
2. **Uždarykite nenaudojamas užrašų knygeles**, kad atlaisvintumėte resursus
3. **Naudokite mažesnius duomenų pavyzdžius testavimui**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilinkite savo kodą**, kad rastumėte našumo problemas:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Didelis atminties naudojimas

**Problema**: Sistema išnaudoja visą atmintį

**Sprendimas**:
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

## Aplinka ir konfigūracija

### Virtualios aplinkos problemos

**Problema**: Virtuali aplinka neaktyvuojama

**Sprendimas**:
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

**Problema**: Paketai įdiegti, bet nerandami užrašų knygelėje

**Sprendimas**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git problemos

**Problema**: Nepavyksta atsisiųsti naujausių pakeitimų – susijungimo konfliktai

**Sprendimas**:
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

### VS Code integracija

**Problema**: Jupyter užrašų knygelės neatidaromos VS Code

**Sprendimas**:
1. Įdiekite Python plėtinį VS Code
2. Įdiekite Jupyter plėtinį VS Code
3. Pasirinkite tinkamą Python interpretatorių: `Ctrl+Shift+P` → „Python: Select Interpreter“
4. Perkraukite VS Code

---

## Papildomi ištekliai

- **Discord diskusijos**: [Užduokite klausimus ir dalinkitės sprendimais #ml-for-beginners kanale](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners moduliai](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Vaizdo pamokos**: [YouTube grojaraštis](https://aka.ms/ml-beginners-videos)
- **Problemų sekimo įrankis**: [Praneškite apie klaidas](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Vis dar kyla problemų?

Jei išbandėte aukščiau pateiktus sprendimus ir vis dar susiduriate su problemomis:

1. **Ieškokite esamų problemų**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Patikrinkite diskusijas Discord**: [Discord diskusijos](https://aka.ms/foundry/discord)
3. **Atidarykite naują problemą**: Įtraukite:
   - Savo operacinę sistemą ir versiją
   - Python/R versiją
   - Klaidos pranešimą (pilną atsekimo informaciją)
   - Žingsnius, kaip atkurti problemą
   - Ką jau bandėte

Mes pasiruošę padėti! 🚀

---

**Atsakomybės atsisakymas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors stengiamės užtikrinti tikslumą, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.