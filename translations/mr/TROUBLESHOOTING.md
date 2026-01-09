<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:42:02+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "mr"
}
-->
# समस्या निराकरण मार्गदर्शक

ही मार्गदर्शिका मशीन लर्निंग फॉर बिगिनर्स अभ्यासक्रमाशी संबंधित सामान्य समस्या सोडविण्यास मदत करते. जर तुम्हाला येथे उपाय सापडत नसेल, तर कृपया आमच्या [Discord Discussions](https://aka.ms/foundry/discord) किंवा [open an issue](https://github.com/microsoft/ML-For-Beginners/issues) येथे तपासा.

## विषय सूची

- [इंस्टॉलेशन समस्या](../..)
- [ज्युपिटर नोटबुक समस्या](../..)
- [पायथन पॅकेज समस्या](../..)
- [R पर्यावरण समस्या](../..)
- [क्विझ अॅप्लिकेशन समस्या](../..)
- [डेटा आणि फाइल पथ समस्या](../..)
- [सामान्य त्रुटी संदेश](../..)
- [कामगिरी समस्या](../..)
- [पर्यावरण आणि कॉन्फिगरेशन](../..)

---

## इंस्टॉलेशन समस्या

### पायथन इंस्टॉलेशन

**समस्या**: `python: command not found`

**उपाय**:
1. [python.org](https://www.python.org/downloads/) वरून Python 3.8 किंवा त्याहून उच्च आवृत्ती इंस्टॉल करा.
2. इंस्टॉलेशन सत्यापित करा: `python --version` किंवा `python3 --version`
3. macOS/Linux वर, तुम्हाला `python3` ऐवजी `python` वापरावे लागेल.

**समस्या**: अनेक पायथन आवृत्त्यांमुळे संघर्ष

**उपाय**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### ज्युपिटर इंस्टॉलेशन

**समस्या**: `jupyter: command not found`

**उपाय**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**समस्या**: ज्युपिटर ब्राउझरमध्ये सुरू होत नाही

**उपाय**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R इंस्टॉलेशन

**समस्या**: R पॅकेजेस इंस्टॉल होत नाहीत

**उपाय**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**समस्या**: IRkernel ज्युपिटरमध्ये उपलब्ध नाही

**उपाय**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## ज्युपिटर नोटबुक समस्या

### कर्नल समस्या

**समस्या**: कर्नल सतत बंद होतो किंवा पुन्हा सुरू होतो

**उपाय**:
1. कर्नल रीस्टार्ट करा: `Kernel → Restart`
2. आउटपुट साफ करा आणि रीस्टार्ट करा: `Kernel → Restart & Clear Output`
3. मेमरी समस्या तपासा (पहा [कामगिरी समस्या](../..))
4. त्रासदायक कोड ओळखण्यासाठी सेल्स स्वतंत्रपणे चालवा.

**समस्या**: चुकीचा पायथन कर्नल निवडला आहे

**उपाय**:
1. सध्याचा कर्नल तपासा: `Kernel → Change Kernel`
2. योग्य पायथन आवृत्ती निवडा.
3. जर कर्नल गायब असेल, तर तो तयार करा:
```bash
python -m ipykernel install --user --name=ml-env
```

**समस्या**: कर्नल सुरू होत नाही

**उपाय**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### नोटबुक सेल समस्या

**समस्या**: सेल्स चालत आहेत पण आउटपुट दिसत नाही

**उपाय**:
1. तपासा की सेल अजून चालू आहे (`[*]` निर्देशक शोधा).
2. कर्नल रीस्टार्ट करा आणि सर्व सेल्स चालवा: `Kernel → Restart & Run All`
3. ब्राउझर कन्सोलमध्ये JavaScript त्रुटी तपासा (F12).

**समस्या**: सेल्स चालवता येत नाहीत - "Run" क्लिक केल्यावर प्रतिसाद नाही

**उपाय**:
1. तपासा की ज्युपिटर सर्व्हर अजून टर्मिनलमध्ये चालू आहे.
2. ब्राउझर पृष्ठ रीफ्रेश करा.
3. नोटबुक बंद करा आणि पुन्हा उघडा.
4. ज्युपिटर सर्व्हर रीस्टार्ट करा.

---

## पायथन पॅकेज समस्या

### आयात त्रुटी

**समस्या**: `ModuleNotFoundError: No module named 'sklearn'`

**उपाय**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**समस्या**: `ImportError: cannot import name 'X' from 'sklearn'`

**उपाय**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### आवृत्ती संघर्ष

**समस्या**: पॅकेज आवृत्ती विसंगती त्रुटी

**उपाय**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**समस्या**: `pip install` परवानगी त्रुटींसह अयशस्वी होते

**उपाय**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### डेटा लोडिंग समस्या

**समस्या**: CSV फाइल्स लोड करताना `FileNotFoundError`

**उपाय**:
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

## R पर्यावरण समस्या

### पॅकेज इंस्टॉलेशन

**समस्या**: पॅकेज इंस्टॉलेशन संकलन त्रुटींसह अयशस्वी होते

**उपाय**:
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

**समस्या**: `tidyverse` इंस्टॉल होत नाही

**उपाय**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown समस्या

**समस्या**: RMarkdown रेंडर होत नाही

**उपाय**:
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

## क्विझ अॅप्लिकेशन समस्या

### बिल्ड आणि इंस्टॉलेशन

**समस्या**: `npm install` अयशस्वी होते

**उपाय**:
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

**समस्या**: पोर्ट 8080 आधीच वापरात आहे

**उपाय**:
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

### बिल्ड त्रुटी

**समस्या**: `npm run build` अयशस्वी होते

**उपाय**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**समस्या**: लिंटिंग त्रुटी बिल्ड रोखत आहेत

**उपाय**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## डेटा आणि फाइल पथ समस्या

### पथ समस्या

**समस्या**: नोटबुक चालवताना डेटा फाइल्स सापडत नाहीत

**उपाय**:
1. **नेहमी नोटबुक्स त्यांच्या असलेल्या डिरेक्टरीमधून चालवा**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **कोडमधील सापेक्ष पथ तपासा**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **गरज असल्यास पूर्ण पथ वापरा**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### हरवलेल्या डेटा फाइल्स

**समस्या**: डेटासेट फाइल्स हरवल्या आहेत

**उपाय**:
1. तपासा की डेटा रेपॉझिटरीमध्ये असायला हवा - बहुतेक डेटासेट्स समाविष्ट आहेत.
2. काही धडे डेटा डाउनलोड करण्याची आवश्यकता असू शकतात - धडा README तपासा.
3. सुनिश्चित करा की तुम्ही नवीनतम बदल पुल केले आहेत:
   ```bash
   git pull origin main
   ```

---

## सामान्य त्रुटी संदेश

### मेमरी त्रुटी

**त्रुटी**: `MemoryError` किंवा डेटा प्रक्रिया करताना कर्नल बंद होतो

**उपाय**:
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

### अभिसरण चेतावणी

**चेतावणी**: `ConvergenceWarning: Maximum number of iterations reached`

**उपाय**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### प्लॉटिंग समस्या

**समस्या**: ज्युपिटरमध्ये प्लॉट्स दिसत नाहीत

**उपाय**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**समस्या**: Seaborn प्लॉट्स वेगळे दिसतात किंवा त्रुटी देतात

**उपाय**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### युनिकोड/एन्कोडिंग त्रुटी

**समस्या**: फाइल्स वाचताना `UnicodeDecodeError`

**उपाय**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## कामगिरी समस्या

### नोटबुक कार्यक्षमता मंद

**समस्या**: नोटबुक चालवायला खूप वेळ लागतो

**उपाय**:
1. **मेमरी मुक्त करण्यासाठी कर्नल रीस्टार्ट करा**: `Kernel → Restart`
2. **न वापरलेले नोटबुक्स बंद करा** संसाधने मुक्त करण्यासाठी.
3. **चाचणीसाठी लहान डेटा नमुने वापरा**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **तुमचा कोड प्रोफाइल करा** बॉटलनेक्स शोधण्यासाठी:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### उच्च मेमरी वापर

**समस्या**: सिस्टम मेमरी संपत आहे

**उपाय**:
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

## पर्यावरण आणि कॉन्फिगरेशन

### वर्चुअल पर्यावरण समस्या

**समस्या**: वर्चुअल पर्यावरण सक्रिय होत नाही

**उपाय**:
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

**समस्या**: पॅकेजेस इंस्टॉल केले पण नोटबुकमध्ये सापडत नाहीत

**उपाय**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git समस्या

**समस्या**: नवीनतम बदल पुल करता येत नाहीत - मर्ज संघर्ष

**उपाय**:
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

### VS कोड एकत्रीकरण

**समस्या**: VS कोडमध्ये ज्युपिटर नोटबुक्स उघडत नाहीत

**उपाय**:
1. VS कोडमध्ये पायथन एक्सटेंशन इंस्टॉल करा.
2. VS कोडमध्ये ज्युपिटर एक्सटेंशन इंस्टॉल करा.
3. योग्य पायथन इंटरप्रिटर निवडा: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS कोड रीस्टार्ट करा.

---

## अतिरिक्त संसाधने

- **Discord Discussions**: [#ml-for-beginners चॅनेलमध्ये प्रश्न विचारा आणि उपाय शेअर करा](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners मॉड्यूल्स](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **व्हिडिओ ट्यूटोरियल्स**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **इश्यू ट्रॅकर**: [बग्स रिपोर्ट करा](https://github.com/microsoft/ML-For-Beginners/issues)

---

## अजूनही समस्या आहेत?

जर तुम्ही वरील उपाय प्रयत्न केले असतील आणि अजूनही समस्या येत असतील:

1. **अस्तित्वातील समस्या शोधा**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord मध्ये चर्चा तपासा**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **नवीन समस्या उघडा**: समाविष्ट करा:
   - तुमची ऑपरेटिंग सिस्टम आणि आवृत्ती
   - पायथन/R आवृत्ती
   - त्रुटी संदेश (पूर्ण ट्रेसबॅक)
   - समस्या पुनरुत्पादित करण्याचे चरण
   - तुम्ही आधीच काय प्रयत्न केले आहे

आम्ही मदतीसाठी येथे आहोत! 🚀

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित केला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये चुका किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी, व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून उद्भवलेल्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.