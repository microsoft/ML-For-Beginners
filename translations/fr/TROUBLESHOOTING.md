<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:34:59+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "fr"
}
-->
# Guide de dépannage

Ce guide vous aide à résoudre les problèmes courants rencontrés lors de l'utilisation du programme Machine Learning pour les débutants. Si vous ne trouvez pas de solution ici, consultez nos [discussions sur Discord](https://aka.ms/foundry/discord) ou [ouvrez un ticket](https://github.com/microsoft/ML-For-Beginners/issues).

## Table des matières

- [Problèmes d'installation](../..)
- [Problèmes avec Jupyter Notebook](../..)
- [Problèmes avec les packages Python](../..)
- [Problèmes avec l'environnement R](../..)
- [Problèmes avec l'application de quiz](../..)
- [Problèmes de données et de chemins de fichiers](../..)
- [Messages d'erreur courants](../..)
- [Problèmes de performance](../..)
- [Environnement et configuration](../..)

---

## Problèmes d'installation

### Installation de Python

**Problème** : `python: command not found`

**Solution** :
1. Installez Python 3.8 ou une version supérieure depuis [python.org](https://www.python.org/downloads/)
2. Vérifiez l'installation : `python --version` ou `python3 --version`
3. Sur macOS/Linux, vous devrez peut-être utiliser `python3` au lieu de `python`

**Problème** : Conflits entre plusieurs versions de Python

**Solution** :
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Installation de Jupyter

**Problème** : `jupyter: command not found`

**Solution** :
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Problème** : Jupyter ne s'ouvre pas dans le navigateur

**Solution** :
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Installation de R

**Problème** : Les packages R ne s'installent pas

**Solution** :
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Problème** : IRkernel n'est pas disponible dans Jupyter

**Solution** :
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Problèmes avec Jupyter Notebook

### Problèmes de kernel

**Problème** : Le kernel cesse de fonctionner ou redémarre constamment

**Solution** :
1. Redémarrez le kernel : `Kernel → Restart`
2. Effacez la sortie et redémarrez : `Kernel → Restart & Clear Output`
3. Vérifiez les problèmes de mémoire (voir [Problèmes de performance](../..))
4. Exécutez les cellules individuellement pour identifier le code problématique

**Problème** : Mauvais kernel Python sélectionné

**Solution** :
1. Vérifiez le kernel actuel : `Kernel → Change Kernel`
2. Sélectionnez la version correcte de Python
3. Si le kernel est manquant, créez-le :
```bash
python -m ipykernel install --user --name=ml-env
```

**Problème** : Le kernel ne démarre pas

**Solution** :
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Problèmes avec les cellules du notebook

**Problème** : Les cellules s'exécutent mais n'affichent pas de sortie

**Solution** :
1. Vérifiez si la cellule est toujours en cours d'exécution (recherchez l'indicateur `[*]`)
2. Redémarrez le kernel et exécutez toutes les cellules : `Kernel → Restart & Run All`
3. Vérifiez la console du navigateur pour des erreurs JavaScript (F12)

**Problème** : Impossible d'exécuter les cellules - aucune réponse lorsque vous cliquez sur "Run"

**Solution** :
1. Vérifiez si le serveur Jupyter fonctionne toujours dans le terminal
2. Actualisez la page du navigateur
3. Fermez et rouvrez le notebook
4. Redémarrez le serveur Jupyter

---

## Problèmes avec les packages Python

### Erreurs d'importation

**Problème** : `ModuleNotFoundError: No module named 'sklearn'`

**Solution** :
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Problème** : `ImportError: cannot import name 'X' from 'sklearn'`

**Solution** :
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Conflits de version

**Problème** : Erreurs d'incompatibilité de version des packages

**Solution** :
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Problème** : `pip install` échoue avec des erreurs de permission

**Solution** :
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Problèmes de chargement des données

**Problème** : `FileNotFoundError` lors du chargement de fichiers CSV

**Solution** :
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

## Problèmes avec l'environnement R

### Installation de packages

**Problème** : L'installation de packages échoue avec des erreurs de compilation

**Solution** :
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

**Problème** : `tidyverse` ne s'installe pas

**Solution** :
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Problèmes avec RMarkdown

**Problème** : RMarkdown ne se rend pas

**Solution** :
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

## Problèmes avec l'application de quiz

### Construction et installation

**Problème** : `npm install` échoue

**Solution** :
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

**Problème** : Le port 8080 est déjà utilisé

**Solution** :
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

### Erreurs de construction

**Problème** : `npm run build` échoue

**Solution** :
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Problème** : Erreurs de lint empêchant la construction

**Solution** :
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Problèmes de données et de chemins de fichiers

### Problèmes de chemin

**Problème** : Les fichiers de données ne sont pas trouvés lors de l'exécution des notebooks

**Solution** :
1. **Exécutez toujours les notebooks depuis leur répertoire contenant**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Vérifiez les chemins relatifs dans le code**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Utilisez des chemins absolus si nécessaire**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Fichiers de données manquants

**Problème** : Les fichiers de dataset sont manquants

**Solution** :
1. Vérifiez si les données doivent être dans le dépôt - la plupart des datasets sont inclus
2. Certains cours peuvent nécessiter le téléchargement de données - consultez le README du cours
3. Assurez-vous d'avoir récupéré les dernières modifications :
   ```bash
   git pull origin main
   ```

---

## Messages d'erreur courants

### Erreurs de mémoire

**Erreur** : `MemoryError` ou le kernel cesse de fonctionner lors du traitement des données

**Solution** :
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

### Avertissements de convergence

**Avertissement** : `ConvergenceWarning: Maximum number of iterations reached`

**Solution** :
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Problèmes de visualisation

**Problème** : Les graphiques ne s'affichent pas dans Jupyter

**Solution** :
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Problème** : Les graphiques Seaborn semblent différents ou génèrent des erreurs

**Solution** :
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Erreurs Unicode/Encodage

**Problème** : `UnicodeDecodeError` lors de la lecture de fichiers

**Solution** :
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Problèmes de performance

### Exécution lente des notebooks

**Problème** : Les notebooks sont très lents à exécuter

**Solution** :
1. **Redémarrez le kernel pour libérer de la mémoire** : `Kernel → Restart`
2. **Fermez les notebooks inutilisés** pour libérer des ressources
3. **Utilisez des échantillons de données plus petits pour les tests** :
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Analysez votre code** pour identifier les goulots d'étranglement :
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Utilisation élevée de la mémoire

**Problème** : Le système manque de mémoire

**Solution** :
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

## Environnement et configuration

### Problèmes avec les environnements virtuels

**Problème** : L'environnement virtuel ne s'active pas

**Solution** :
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

**Problème** : Les packages sont installés mais non trouvés dans le notebook

**Solution** :
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Problèmes avec Git

**Problème** : Impossible de récupérer les dernières modifications - conflits de fusion

**Solution** :
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

### Intégration avec VS Code

**Problème** : Les notebooks Jupyter ne s'ouvrent pas dans VS Code

**Solution** :
1. Installez l'extension Python dans VS Code
2. Installez l'extension Jupyter dans VS Code
3. Sélectionnez l'interpréteur Python correct : `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Redémarrez VS Code

---

## Ressources supplémentaires

- **Discussions sur Discord** : [Posez vos questions et partagez vos solutions dans le canal #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn** : [Modules ML pour les débutants](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutoriels vidéo** : [Playlist YouTube](https://aka.ms/ml-beginners-videos)
- **Suivi des problèmes** : [Signalez des bugs](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Vous rencontrez toujours des problèmes ?

Si vous avez essayé les solutions ci-dessus et que vous rencontrez toujours des problèmes :

1. **Recherchez les problèmes existants** : [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Consultez les discussions sur Discord** : [Discussions sur Discord](https://aka.ms/foundry/discord)
3. **Ouvrez un nouveau ticket** : Incluez :
   - Votre système d'exploitation et sa version
   - Version de Python/R
   - Message d'erreur (trace complète)
   - Étapes pour reproduire le problème
   - Ce que vous avez déjà essayé

Nous sommes là pour vous aider ! 🚀

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.