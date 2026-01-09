<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:46:14+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "el"
}
-->
# Οδηγός Αντιμετώπισης Προβλημάτων

Αυτός ο οδηγός σας βοηθά να λύσετε κοινά προβλήματα κατά την εργασία με το πρόγραμμα σπουδών Machine Learning for Beginners. Αν δεν βρείτε λύση εδώ, παρακαλούμε ελέγξτε τις [Συζητήσεις στο Discord](https://aka.ms/foundry/discord) ή [ανοίξτε ένα ζήτημα](https://github.com/microsoft/ML-For-Beginners/issues).

## Πίνακας Περιεχομένων

- [Προβλήματα Εγκατάστασης](../..)
- [Προβλήματα με το Jupyter Notebook](../..)
- [Προβλήματα με Πακέτα Python](../..)
- [Προβλήματα με το Περιβάλλον R](../..)
- [Προβλήματα με την Εφαρμογή Κουίζ](../..)
- [Προβλήματα Δεδομένων και Διαδρομών Αρχείων](../..)
- [Συνηθισμένα Μηνύματα Σφαλμάτων](../..)
- [Προβλήματα Απόδοσης](../..)
- [Περιβάλλον και Ρυθμίσεις](../..)

---

## Προβλήματα Εγκατάστασης

### Εγκατάσταση Python

**Πρόβλημα**: `python: command not found`

**Λύση**:
1. Εγκαταστήστε Python 3.8 ή νεότερη έκδοση από [python.org](https://www.python.org/downloads/)
2. Επαληθεύστε την εγκατάσταση: `python --version` ή `python3 --version`
3. Σε macOS/Linux, ίσως χρειαστεί να χρησιμοποιήσετε `python3` αντί για `python`

**Πρόβλημα**: Πολλαπλές εκδόσεις Python προκαλούν συγκρούσεις

**Λύση**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Εγκατάσταση Jupyter

**Πρόβλημα**: `jupyter: command not found`

**Λύση**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Πρόβλημα**: Το Jupyter δεν ανοίγει στον περιηγητή

**Λύση**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Εγκατάσταση R

**Πρόβλημα**: Τα πακέτα R δεν εγκαθίστανται

**Λύση**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Πρόβλημα**: Το IRkernel δεν είναι διαθέσιμο στο Jupyter

**Λύση**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Προβλήματα με το Jupyter Notebook

### Προβλήματα με τον Kernel

**Πρόβλημα**: Ο Kernel συνεχώς τερματίζεται ή επανεκκινείται

**Λύση**:
1. Επανεκκινήστε τον kernel: `Kernel → Restart`
2. Καθαρίστε την έξοδο και επανεκκινήστε: `Kernel → Restart & Clear Output`
3. Ελέγξτε για προβλήματα μνήμης (δείτε [Προβλήματα Απόδοσης](../..))
4. Δοκιμάστε να εκτελέσετε τα κελιά ξεχωριστά για να εντοπίσετε προβληματικό κώδικα

**Πρόβλημα**: Επιλέχθηκε λάθος kernel Python

**Λύση**:
1. Ελέγξτε τον τρέχοντα kernel: `Kernel → Change Kernel`
2. Επιλέξτε τη σωστή έκδοση Python
3. Αν λείπει ο kernel, δημιουργήστε τον:
```bash
python -m ipykernel install --user --name=ml-env
```

**Πρόβλημα**: Ο Kernel δεν ξεκινά

**Λύση**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Προβλήματα με τα Κελιά του Notebook

**Πρόβλημα**: Τα κελιά εκτελούνται αλλά δεν εμφανίζουν έξοδο

**Λύση**:
1. Ελέγξτε αν το κελί εξακολουθεί να εκτελείται (αναζητήστε τον δείκτη `[*]`)
2. Επανεκκινήστε τον kernel και εκτελέστε όλα τα κελιά: `Kernel → Restart & Run All`
3. Ελέγξτε την κονσόλα του περιηγητή για σφάλματα JavaScript (F12)

**Πρόβλημα**: Δεν μπορείτε να εκτελέσετε κελιά - καμία αντίδραση όταν κάνετε κλικ στο "Run"

**Λύση**:
1. Ελέγξτε αν ο server του Jupyter εξακολουθεί να εκτελείται στο τερματικό
2. Ανανεώστε τη σελίδα του περιηγητή
3. Κλείστε και ανοίξτε ξανά το notebook
4. Επανεκκινήστε τον server του Jupyter

---

## Προβλήματα με Πακέτα Python

### Σφάλματα Εισαγωγής

**Πρόβλημα**: `ModuleNotFoundError: No module named 'sklearn'`

**Λύση**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Πρόβλημα**: `ImportError: cannot import name 'X' from 'sklearn'`

**Λύση**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Συγκρούσεις Εκδόσεων

**Πρόβλημα**: Σφάλματα ασυμβατότητας εκδόσεων πακέτων

**Λύση**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Πρόβλημα**: Η εντολή `pip install` αποτυγχάνει λόγω σφαλμάτων δικαιωμάτων

**Λύση**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Προβλήματα Φόρτωσης Δεδομένων

**Πρόβλημα**: `FileNotFoundError` κατά τη φόρτωση αρχείων CSV

**Λύση**:
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

## Προβλήματα με το Περιβάλλον R

### Εγκατάσταση Πακέτων

**Πρόβλημα**: Η εγκατάσταση πακέτων αποτυγχάνει με σφάλματα μεταγλώττισης

**Λύση**:
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

**Πρόβλημα**: Το `tidyverse` δεν εγκαθίσταται

**Λύση**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Προβλήματα με το RMarkdown

**Πρόβλημα**: Το RMarkdown δεν αποδίδει

**Λύση**:
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

## Προβλήματα με την Εφαρμογή Κουίζ

### Δημιουργία και Εγκατάσταση

**Πρόβλημα**: Η εντολή `npm install` αποτυγχάνει

**Λύση**:
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

**Πρόβλημα**: Η θύρα 8080 είναι ήδη σε χρήση

**Λύση**:
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

### Σφάλματα Δημιουργίας

**Πρόβλημα**: Η εντολή `npm run build` αποτυγχάνει

**Λύση**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Πρόβλημα**: Σφάλματα linting εμποδίζουν τη δημιουργία

**Λύση**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Προβλήματα Δεδομένων και Διαδρομών Αρχείων

### Προβλήματα Διαδρομών

**Πρόβλημα**: Τα αρχεία δεδομένων δεν βρίσκονται κατά την εκτέλεση notebooks

**Λύση**:
1. **Πάντα να εκτελείτε τα notebooks από τον φάκελο που τα περιέχει**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Ελέγξτε τις σχετικές διαδρομές στον κώδικα**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Χρησιμοποιήστε απόλυτες διαδρομές αν χρειάζεται**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Ελλείποντα Αρχεία Δεδομένων

**Πρόβλημα**: Λείπουν αρχεία dataset

**Λύση**:
1. Ελέγξτε αν τα δεδομένα πρέπει να βρίσκονται στο αποθετήριο - τα περισσότερα datasets περιλαμβάνονται
2. Ορισμένα μαθήματα μπορεί να απαιτούν λήψη δεδομένων - ελέγξτε το README του μαθήματος
3. Βεβαιωθείτε ότι έχετε τραβήξει τις τελευταίες αλλαγές:
   ```bash
   git pull origin main
   ```

---

## Συνηθισμένα Μηνύματα Σφαλμάτων

### Σφάλματα Μνήμης

**Σφάλμα**: `MemoryError` ή ο kernel τερματίζεται κατά την επεξεργασία δεδομένων

**Λύση**:
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

### Προειδοποιήσεις Σύγκλισης

**Προειδοποίηση**: `ConvergenceWarning: Maximum number of iterations reached`

**Λύση**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Προβλήματα Σχεδίασης

**Πρόβλημα**: Τα γραφήματα δεν εμφανίζονται στο Jupyter

**Λύση**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Πρόβλημα**: Τα γραφήματα Seaborn φαίνονται διαφορετικά ή εμφανίζουν σφάλματα

**Λύση**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Σφάλματα Unicode/Κωδικοποίησης

**Πρόβλημα**: `UnicodeDecodeError` κατά την ανάγνωση αρχείων

**Λύση**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Προβλήματα Απόδοσης

### Αργή Εκτέλεση Notebook

**Πρόβλημα**: Τα notebooks εκτελούνται πολύ αργά

**Λύση**:
1. **Επανεκκινήστε τον kernel για να ελευθερώσετε μνήμη**: `Kernel → Restart`
2. **Κλείστε αχρησιμοποίητα notebooks** για να ελευθερώσετε πόρους
3. **Χρησιμοποιήστε μικρότερα δείγματα δεδομένων για δοκιμές**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Προφίλ του κώδικα σας** για να βρείτε σημεία συμφόρησης:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Υψηλή Χρήση Μνήμης

**Πρόβλημα**: Το σύστημα εξαντλεί τη μνήμη

**Λύση**:
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

## Περιβάλλον και Ρυθμίσεις

### Προβλήματα Εικονικού Περιβάλλοντος

**Πρόβλημα**: Το εικονικό περιβάλλον δεν ενεργοποιείται

**Λύση**:
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

**Πρόβλημα**: Τα πακέτα είναι εγκατεστημένα αλλά δεν βρίσκονται στο notebook

**Λύση**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Προβλήματα Git

**Πρόβλημα**: Δεν μπορείτε να τραβήξετε τις τελευταίες αλλαγές - συγκρούσεις συγχώνευσης

**Λύση**:
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

### Ενσωμάτωση με το VS Code

**Πρόβλημα**: Τα Jupyter notebooks δεν ανοίγουν στο VS Code

**Λύση**:
1. Εγκαταστήστε την επέκταση Python στο VS Code
2. Εγκαταστήστε την επέκταση Jupyter στο VS Code
3. Επιλέξτε τη σωστή Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Επανεκκινήστε το VS Code

---

## Πρόσθετοι Πόροι

- **Συζητήσεις στο Discord**: [Κάντε ερωτήσεις και μοιραστείτε λύσεις στο κανάλι #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Ενότητες ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Βίντεο Tutorials**: [Λίστα αναπαραγωγής στο YouTube](https://aka.ms/ml-beginners-videos)
- **Παρακολούθηση Ζητημάτων**: [Αναφορά σφαλμάτων](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Εξακολουθείτε να Έχετε Προβλήματα;

Αν έχετε δοκιμάσει τις παραπάνω λύσεις και εξακολουθείτε να αντιμετωπίζετε προβλήματα:

1. **Αναζητήστε υπάρχοντα ζητήματα**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Ελέγξτε τις συζητήσεις στο Discord**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **Ανοίξτε ένα νέο ζήτημα**: Συμπεριλάβετε:
   - Το λειτουργικό σας σύστημα και την έκδοση
   - Την έκδοση Python/R
   - Το μήνυμα σφάλματος (πλήρες traceback)
   - Τα βήματα για την αναπαραγωγή του προβλήματος
   - Τι έχετε ήδη δοκιμάσει

Είμαστε εδώ για να βοηθήσουμε! 🚀

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτόματες μεταφράσεις ενδέχεται να περιέχουν λάθη ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.