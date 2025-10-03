<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:44:31+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "it"
}
-->
# Guida alla Risoluzione dei Problemi

Questa guida ti aiuterà a risolvere i problemi più comuni durante l'utilizzo del curriculum Machine Learning for Beginners. Se non trovi una soluzione qui, consulta le nostre [Discussioni su Discord](https://aka.ms/foundry/discord) o [apri un problema](https://github.com/microsoft/ML-For-Beginners/issues).

## Indice

- [Problemi di Installazione](../..)
- [Problemi con Jupyter Notebook](../..)
- [Problemi con i Pacchetti Python](../..)
- [Problemi con l'Ambiente R](../..)
- [Problemi con l'Applicazione Quiz](../..)
- [Problemi con i Dati e i Percorsi dei File](../..)
- [Messaggi di Errore Comuni](../..)
- [Problemi di Prestazioni](../..)
- [Ambiente e Configurazione](../..)

---

## Problemi di Installazione

### Installazione di Python

**Problema**: `python: command not found`

**Soluzione**:
1. Installa Python 3.8 o superiore da [python.org](https://www.python.org/downloads/)
2. Verifica l'installazione: `python --version` o `python3 --version`
3. Su macOS/Linux, potresti dover usare `python3` invece di `python`

**Problema**: Conflitti causati da più versioni di Python

**Soluzione**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Installazione di Jupyter

**Problema**: `jupyter: command not found`

**Soluzione**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problema**: Jupyter non si apre nel browser

**Soluzione**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Installazione di R

**Problema**: I pacchetti R non si installano

**Soluzione**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problema**: IRkernel non disponibile in Jupyter

**Soluzione**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problemi con Jupyter Notebook

### Problemi con il Kernel

**Problema**: Il kernel continua a morire o riavviarsi

**Soluzione**:
1. Riavvia il kernel: `Kernel → Restart`
2. Cancella l'output e riavvia: `Kernel → Restart & Clear Output`
3. Controlla eventuali problemi di memoria (vedi [Problemi di Prestazioni](../..))
4. Prova a eseguire le celle singolarmente per identificare il codice problematico

**Problema**: Kernel Python sbagliato selezionato

**Soluzione**:
1. Controlla il kernel corrente: `Kernel → Change Kernel`
2. Seleziona la versione corretta di Python
3. Se il kernel manca, crealo:
```bash
python -m ipykernel install --user --name=ml-env
```

**Problema**: Il kernel non si avvia

**Soluzione**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problemi con le Celle del Notebook

**Problema**: Le celle vengono eseguite ma non mostrano output

**Soluzione**:
1. Controlla se la cella è ancora in esecuzione (cerca l'indicatore `[*]`)
2. Riavvia il kernel ed esegui tutte le celle: `Kernel → Restart & Run All`
3. Controlla la console del browser per errori JavaScript (F12)

**Problema**: Non è possibile eseguire le celle - nessuna risposta quando si clicca su "Run"

**Soluzione**:
1. Controlla se il server Jupyter è ancora in esecuzione nel terminale
2. Aggiorna la pagina del browser
3. Chiudi e riapri il notebook
4. Riavvia il server Jupyter

---

## Problemi con i Pacchetti Python

### Errori di Importazione

**Problema**: `ModuleNotFoundError: No module named 'sklearn'`

**Soluzione**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problema**: `ImportError: cannot import name 'X' from 'sklearn'`

**Soluzione**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Conflitti di Versione

**Problema**: Errori di incompatibilità delle versioni dei pacchetti

**Soluzione**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problema**: `pip install` fallisce con errori di permessi

**Soluzione**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problemi di Caricamento Dati

**Problema**: `FileNotFoundError` durante il caricamento di file CSV

**Soluzione**:
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

## Problemi con l'Ambiente R

### Installazione dei Pacchetti

**Problema**: L'installazione dei pacchetti fallisce con errori di compilazione

**Soluzione**:
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

**Problema**: `tidyverse` non si installa

**Soluzione**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problemi con RMarkdown

**Problema**: RMarkdown non viene renderizzato

**Soluzione**:
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

## Problemi con l'Applicazione Quiz

### Build e Installazione

**Problema**: `npm install` fallisce

**Soluzione**:
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

**Problema**: Porta 8080 già in uso

**Soluzione**:
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

### Errori di Build

**Problema**: `npm run build` fallisce

**Soluzione**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problema**: Errori di linting che impediscono la build

**Soluzione**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problemi con i Dati e i Percorsi dei File

### Problemi di Percorso

**Problema**: File di dati non trovati durante l'esecuzione dei notebook

**Soluzione**:
1. **Esegui sempre i notebook dalla loro directory contenente**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Controlla i percorsi relativi nel codice**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Usa percorsi assoluti se necessario**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### File di Dati Mancanti

**Problema**: I file del dataset sono mancanti

**Soluzione**:
1. Controlla se i dati dovrebbero essere nel repository - la maggior parte dei dataset è inclusa
2. Alcune lezioni potrebbero richiedere il download dei dati - controlla il README della lezione
3. Assicurati di aver scaricato gli ultimi aggiornamenti:
   ```bash
   git pull origin main
   ```

---

## Messaggi di Errore Comuni

### Errori di Memoria

**Errore**: `MemoryError` o il kernel si arresta durante l'elaborazione dei dati

**Soluzione**:
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

### Avvisi di Convergenza

**Avviso**: `ConvergenceWarning: Maximum number of iterations reached`

**Soluzione**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problemi di Visualizzazione

**Problema**: I grafici non vengono mostrati in Jupyter

**Soluzione**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problema**: I grafici di Seaborn appaiono diversi o generano errori

**Soluzione**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Errori di Codifica/Unicode

**Problema**: `UnicodeDecodeError` durante la lettura dei file

**Soluzione**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problemi di Prestazioni

### Esecuzione Lenta dei Notebook

**Problema**: I notebook sono molto lenti da eseguire

**Soluzione**:
1. **Riavvia il kernel per liberare memoria**: `Kernel → Restart`
2. **Chiudi i notebook non utilizzati** per liberare risorse
3. **Usa campioni di dati più piccoli per i test**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilare il codice** per individuare i colli di bottiglia:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Elevato Utilizzo di Memoria

**Problema**: Il sistema esaurisce la memoria

**Soluzione**:
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

## Ambiente e Configurazione

### Problemi con l'Ambiente Virtuale

**Problema**: L'ambiente virtuale non si attiva

**Soluzione**:
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

**Problema**: I pacchetti installati non vengono trovati nel notebook

**Soluzione**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problemi con Git

**Problema**: Impossibile scaricare gli ultimi aggiornamenti - conflitti di merge

**Soluzione**:
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

### Integrazione con VS Code

**Problema**: I notebook Jupyter non si aprono in VS Code

**Soluzione**:
1. Installa l'estensione Python in VS Code
2. Installa l'estensione Jupyter in VS Code
3. Seleziona l'interprete Python corretto: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Riavvia VS Code

---

## Risorse Aggiuntive

- **Discussioni su Discord**: [Fai domande e condividi soluzioni nel canale #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Moduli ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutorial Video**: [Playlist su YouTube](https://aka.ms/ml-beginners-videos)
- **Tracker dei Problemi**: [Segnala bug](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Ancora Problemi?

Se hai provato le soluzioni sopra e stai ancora riscontrando problemi:

1. **Cerca problemi esistenti**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Controlla le discussioni su Discord**: [Discussioni su Discord](https://aka.ms/foundry/discord)
3. **Apri un nuovo problema**: Includi:
   - Il tuo sistema operativo e la versione
   - Versione di Python/R
   - Messaggio di errore (traccia completa)
   - Passaggi per riprodurre il problema
   - Cosa hai già provato

Siamo qui per aiutarti! 🚀

---

**Disclaimer (Avviso di responsabilità)**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire la precisione, si prega di notare che le traduzioni automatiche potrebbero contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.