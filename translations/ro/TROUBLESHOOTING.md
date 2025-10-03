<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:54:09+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ro"
}
-->
# Ghid de depanare

Acest ghid te ajută să rezolvi problemele comune întâlnite în lucrul cu curriculumul Machine Learning for Beginners. Dacă nu găsești o soluție aici, verifică [Discuțiile pe Discord](https://aka.ms/foundry/discord) sau [deschide o problemă](https://github.com/microsoft/ML-For-Beginners/issues).

## Cuprins

- [Probleme de instalare](../..)
- [Probleme cu Jupyter Notebook](../..)
- [Probleme cu pachetele Python](../..)
- [Probleme cu mediul R](../..)
- [Probleme cu aplicația de quiz](../..)
- [Probleme cu datele și căile fișierelor](../..)
- [Mesaje de eroare comune](../..)
- [Probleme de performanță](../..)
- [Mediu și configurare](../..)

---

## Probleme de instalare

### Instalarea Python

**Problemă**: `python: command not found`

**Soluție**:
1. Instalează Python 3.8 sau o versiune mai nouă de pe [python.org](https://www.python.org/downloads/)
2. Verifică instalarea: `python --version` sau `python3 --version`
3. Pe macOS/Linux, poate fi necesar să folosești `python3` în loc de `python`

**Problemă**: Versiuni multiple de Python cauzează conflicte

**Soluție**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalarea Jupyter

**Problemă**: `jupyter: command not found`

**Soluție**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problemă**: Jupyter nu se deschide în browser

**Soluție**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalarea R

**Problemă**: Pachetele R nu se instalează

**Soluție**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problemă**: IRkernel nu este disponibil în Jupyter

**Soluție**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Probleme cu Jupyter Notebook

### Probleme cu kernel-ul

**Problemă**: Kernel-ul continuă să se oprească sau să se restarteze

**Soluție**:
1. Restartează kernel-ul: `Kernel → Restart`
2. Șterge output-ul și restartează: `Kernel → Restart & Clear Output`
3. Verifică problemele de memorie (vezi [Probleme de performanță](../..))
4. Rulează celulele individual pentru a identifica codul problematic

**Problemă**: Kernel-ul Python greșit selectat

**Soluție**:
1. Verifică kernel-ul curent: `Kernel → Change Kernel`
2. Selectează versiunea corectă de Python
3. Dacă kernel-ul lipsește, creează-l:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problemă**: Kernel-ul nu pornește

**Soluție**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Probleme cu celulele notebook-ului

**Problemă**: Celulele rulează, dar nu afișează output

**Soluție**:
1. Verifică dacă celula încă rulează (caută indicatorul `[*]`)
2. Restartează kernel-ul și rulează toate celulele: `Kernel → Restart & Run All`
3. Verifică consola browserului pentru erori JavaScript (F12)

**Problemă**: Nu pot rula celulele - nu există răspuns la clic pe "Run"

**Soluție**:
1. Verifică dacă serverul Jupyter încă rulează în terminal
2. Reîmprospătează pagina browserului
3. Închide și redeschide notebook-ul
4. Restartează serverul Jupyter

---

## Probleme cu pachetele Python

### Erori de import

**Problemă**: `ModuleNotFoundError: No module named 'sklearn'`

**Soluție**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problemă**: `ImportError: cannot import name 'X' from 'sklearn'`

**Soluție**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Conflicte de versiuni

**Problemă**: Erori de incompatibilitate a versiunilor pachetelor

**Soluție**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problemă**: `pip install` eșuează cu erori de permisiune

**Soluție**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Probleme la încărcarea datelor

**Problemă**: `FileNotFoundError` la încărcarea fișierelor CSV

**Soluție**:
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

## Probleme cu mediul R

### Instalarea pachetelor

**Problemă**: Instalarea pachetelor eșuează cu erori de compilare

**Soluție**:
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

**Problemă**: `tidyverse` nu se instalează

**Soluție**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Probleme cu RMarkdown

**Problemă**: RMarkdown nu se renderizează

**Soluție**:
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

## Probleme cu aplicația de quiz

### Construire și instalare

**Problemă**: `npm install` eșuează

**Soluție**:
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

**Problemă**: Portul 8080 este deja utilizat

**Soluție**:
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

### Erori de construire

**Problemă**: `npm run build` eșuează

**Soluție**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problemă**: Erori de linting împiedică construirea

**Soluție**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Probleme cu datele și căile fișierelor

### Probleme cu căile

**Problemă**: Fișierele de date nu sunt găsite la rularea notebook-urilor

**Soluție**:
1. **Rulează întotdeauna notebook-urile din directorul lor conținător**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Verifică căile relative în cod**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Folosește căi absolute dacă este necesar**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Fișiere de date lipsă

**Problemă**: Fișierele dataset lipsesc

**Soluție**:
1. Verifică dacă datele ar trebui să fie în repository - majoritatea dataset-urilor sunt incluse
2. Unele lecții pot necesita descărcarea datelor - verifică README-ul lecției
3. Asigură-te că ai tras ultimele modificări:
   ```bash
   git pull origin main
   ```

---

## Mesaje de eroare comune

### Erori de memorie

**Eroare**: `MemoryError` sau kernel-ul se oprește la procesarea datelor

**Soluție**:
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

### Avertismente de convergență

**Avertisment**: `ConvergenceWarning: Maximum number of iterations reached`

**Soluție**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Probleme de afișare grafică

**Problemă**: Graficele nu se afișează în Jupyter

**Soluție**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problemă**: Graficele Seaborn arată diferit sau generează erori

**Soluție**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Erori de Unicode/codificare

**Problemă**: `UnicodeDecodeError` la citirea fișierelor

**Soluție**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Probleme de performanță

### Execuție lentă a notebook-urilor

**Problemă**: Notebook-urile rulează foarte lent

**Soluție**:
1. **Restartează kernel-ul pentru a elibera memoria**: `Kernel → Restart`
2. **Închide notebook-urile neutilizate** pentru a elibera resurse
3. **Folosește mostre de date mai mici pentru testare**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilează codul** pentru a identifica punctele slabe:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Utilizare mare de memorie

**Problemă**: Sistemul rămâne fără memorie

**Soluție**:
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

## Mediu și configurare

### Probleme cu mediul virtual

**Problemă**: Mediul virtual nu se activează

**Soluție**:
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

**Problemă**: Pachetele sunt instalate, dar nu sunt găsite în notebook

**Soluție**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Probleme cu Git

**Problemă**: Nu pot trage ultimele modificări - conflicte de îmbinare

**Soluție**:
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

### Integrarea cu VS Code

**Problemă**: Notebook-urile Jupyter nu se deschid în VS Code

**Soluție**:
1. Instalează extensia Python în VS Code
2. Instalează extensia Jupyter în VS Code
3. Selectează interpretul Python corect: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Restartează VS Code

---

## Resurse suplimentare

- **Discuții pe Discord**: [Pune întrebări și împărtășește soluții în canalul #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Module ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutoriale video**: [Playlist YouTube](https://aka.ms/ml-beginners-videos)
- **Tracker de probleme**: [Raportează erori](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Încă ai probleme?

Dacă ai încercat soluțiile de mai sus și încă întâmpini probleme:

1. **Caută probleme existente**: [Probleme pe GitHub](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Verifică discuțiile pe Discord**: [Discuții pe Discord](https://aka.ms/foundry/discord)
3. **Deschide o problemă nouă**: Include:
   - Sistemul de operare și versiunea
   - Versiunea Python/R
   - Mesajul de eroare (traceback complet)
   - Pașii pentru reproducerea problemei
   - Ce ai încercat deja

Suntem aici să te ajutăm! 🚀

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.