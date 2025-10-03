<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:48:46+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "fi"
}
-->
# Vianmääritysopas

Tämä opas auttaa ratkaisemaan yleisiä ongelmia Machine Learning for Beginners -opetussuunnitelman parissa. Jos et löydä ratkaisua täältä, tarkista [Discord-keskustelut](https://aka.ms/foundry/discord) tai [avaa ongelma](https://github.com/microsoft/ML-For-Beginners/issues).

## Sisällysluettelo

- [Asennusongelmat](../..)
- [Jupyter Notebook -ongelmat](../..)
- [Python-paketti-ongelmat](../..)
- [R-ympäristöongelmat](../..)
- [Tietovisa-sovelluksen ongelmat](../..)
- [Data- ja tiedostopolkuongelmat](../..)
- [Yleiset virheilmoitukset](../..)
- [Suorituskykyongelmat](../..)
- [Ympäristö ja konfiguraatio](../..)

---

## Asennusongelmat

### Python-asennus

**Ongelma**: `python: command not found`

**Ratkaisu**:
1. Asenna Python 3.8 tai uudempi [python.org](https://www.python.org/downloads/) -sivustolta
2. Varmista asennus: `python --version` tai `python3 --version`
3. macOS/Linuxissa saatat joutua käyttämään `python3` sijasta `python`

**Ongelma**: Useat Python-versiot aiheuttavat ristiriitoja

**Ratkaisu**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter-asennus

**Ongelma**: `jupyter: command not found`

**Ratkaisu**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Ongelma**: Jupyter ei avaudu selaimessa

**Ratkaisu**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R-asennus

**Ongelma**: R-paketit eivät asennu

**Ratkaisu**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Ongelma**: IRkernel ei ole saatavilla Jupyterissa

**Ratkaisu**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook -ongelmat

### Kernel-ongelmat

**Ongelma**: Kernel kaatuu tai käynnistyy uudelleen jatkuvasti

**Ratkaisu**:
1. Käynnistä kernel uudelleen: `Kernel → Restart`
2. Tyhjennä tulosteet ja käynnistä uudelleen: `Kernel → Restart & Clear Output`
3. Tarkista muistiongelmat (katso [Suorituskykyongelmat](../..))
4. Suorita solut yksitellen ongelmallisen koodin tunnistamiseksi

**Ongelma**: Väärä Python-kernel valittu

**Ratkaisu**:
1. Tarkista nykyinen kernel: `Kernel → Change Kernel`
2. Valitse oikea Python-versio
3. Jos kernel puuttuu, luo se:
```bash
python -m ipykernel install --user --name=ml-env
```

**Ongelma**: Kernel ei käynnisty

**Ratkaisu**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook-solujen ongelmat

**Ongelma**: Solut suoritetaan, mutta tuloste ei näy

**Ratkaisu**:
1. Tarkista, onko solu edelleen käynnissä (etsi `[*]`-merkintä)
2. Käynnistä kernel uudelleen ja suorita kaikki solut: `Kernel → Restart & Run All`
3. Tarkista selaimen konsoli JavaScript-virheiden varalta (F12)

**Ongelma**: Soluja ei voi suorittaa - "Run"-painike ei reagoi

**Ratkaisu**:
1. Tarkista, onko Jupyter-palvelin edelleen käynnissä terminaalissa
2. Päivitä selaimen sivu
3. Sulje ja avaa notebook uudelleen
4. Käynnistä Jupyter-palvelin uudelleen

---

## Python-paketti-ongelmat

### Tuontivirheet

**Ongelma**: `ModuleNotFoundError: No module named 'sklearn'`

**Ratkaisu**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Ongelma**: `ImportError: cannot import name 'X' from 'sklearn'`

**Ratkaisu**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Versioristiriidat

**Ongelma**: Pakettiversioiden yhteensopimattomuusvirheet

**Ratkaisu**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Ongelma**: `pip install` epäonnistuu käyttöoikeusvirheiden vuoksi

**Ratkaisu**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Datan latausongelmat

**Ongelma**: `FileNotFoundError` CSV-tiedostoja ladattaessa

**Ratkaisu**:
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

## R-ympäristöongelmat

### Pakettien asennus

**Ongelma**: Pakettien asennus epäonnistuu käännösvirheiden vuoksi

**Ratkaisu**:
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

**Ongelma**: `tidyverse` ei asennu

**Ratkaisu**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown-ongelmat

**Ongelma**: RMarkdown ei renderöidy

**Ratkaisu**:
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

## Tietovisa-sovelluksen ongelmat

### Rakennus ja asennus

**Ongelma**: `npm install` epäonnistuu

**Ratkaisu**:
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

**Ongelma**: Portti 8080 on jo käytössä

**Ratkaisu**:
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

### Rakennusvirheet

**Ongelma**: `npm run build` epäonnistuu

**Ratkaisu**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Ongelma**: Lint-virheet estävät rakennuksen

**Ratkaisu**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Data- ja tiedostopolkuongelmat

### Polkuongelmat

**Ongelma**: Datatiedostoja ei löydy notebookeja suoritettaessa

**Ratkaisu**:
1. **Suorita notebookit aina niiden sisältävästä hakemistosta**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Tarkista koodin suhteelliset polut**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Käytä tarvittaessa absoluuttisia polkuja**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Puuttuvat datatiedostot

**Ongelma**: Dataset-tiedostot puuttuvat

**Ratkaisu**:
1. Tarkista, kuuluuko data olla repositoriossa - useimmat datasetit sisältyvät
2. Jotkut oppitunnit saattavat vaatia datan lataamista - tarkista oppitunnin README
3. Varmista, että olet hakenut uusimmat muutokset:
   ```bash
   git pull origin main
   ```

---

## Yleiset virheilmoitukset

### Muistivirheet

**Virhe**: `MemoryError` tai kernel kaatuu datan käsittelyn aikana

**Ratkaisu**:
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

### Konvergenssivaroitukset

**Varoitus**: `ConvergenceWarning: Maximum number of iterations reached`

**Ratkaisu**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Piirto-ongelmat

**Ongelma**: Piirrokset eivät näy Jupyterissa

**Ratkaisu**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Ongelma**: Seaborn-piirrokset näyttävät erilaisilta tai aiheuttavat virheitä

**Ratkaisu**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode-/koodausvirheet

**Ongelma**: `UnicodeDecodeError` tiedostoja luettaessa

**Ratkaisu**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Suorituskykyongelmat

### Hidas notebookin suoritus

**Ongelma**: Notebookien suoritus on erittäin hidasta

**Ratkaisu**:
1. **Käynnistä kernel uudelleen vapauttaaksesi muistia**: `Kernel → Restart`
2. **Sulje käyttämättömät notebookit** vapauttaaksesi resursseja
3. **Käytä pienempiä datanäytteitä testaukseen**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profiloi koodisi** pullonkaulojen löytämiseksi:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Korkea muistin käyttö

**Ongelma**: Järjestelmästä loppuu muisti

**Ratkaisu**:
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

## Ympäristö ja konfiguraatio

### Virtuaaliympäristöongelmat

**Ongelma**: Virtuaaliympäristö ei aktivoidu

**Ratkaisu**:
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

**Ongelma**: Paketit asennettu, mutta eivät löydy notebookista

**Ratkaisu**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git-ongelmat

**Ongelma**: Uusimpia muutoksia ei voi hakea - yhdistämisristiriidat

**Ratkaisu**:
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

### VS Code -integraatio

**Ongelma**: Jupyter-notebookit eivät avaudu VS Codessa

**Ratkaisu**:
1. Asenna Python-laajennus VS Codeen
2. Asenna Jupyter-laajennus VS Codeen
3. Valitse oikea Python-tulkki: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Käynnistä VS Code uudelleen

---

## Lisäresurssit

- **Discord-keskustelut**: [Esitä kysymyksiä ja jaa ratkaisuja #ml-for-beginners-kanavalla](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners -moduulit](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Videotutoriaalit**: [YouTube-soittolista](https://aka.ms/ml-beginners-videos)
- **Ongelmanseuranta**: [Ilmoita virheistä](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Onko ongelmia edelleen?

Jos olet kokeillut yllä olevia ratkaisuja ja ongelmat jatkuvat:

1. **Etsi olemassa olevia ongelmia**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Tarkista keskustelut Discordissa**: [Discord-keskustelut](https://aka.ms/foundry/discord)
3. **Avaa uusi ongelma**: Sisällytä:
   - Käyttöjärjestelmäsi ja sen versio
   - Python/R-versio
   - Virheilmoitus (koko traceback)
   - Vaiheet ongelman toistamiseksi
   - Mitä olet jo kokeillut

Olemme täällä auttamassa! 🚀

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.