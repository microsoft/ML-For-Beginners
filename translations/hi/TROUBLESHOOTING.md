<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:40:53+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "hi"
}
-->
# समस्या निवारण गाइड

यह गाइड आपको Machine Learning for Beginners पाठ्यक्रम के साथ काम करते समय आम समस्याओं को हल करने में मदद करता है। यदि आपको यहां समाधान नहीं मिलता है, तो कृपया हमारे [Discord Discussions](https://aka.ms/foundry/discord) देखें या [एक समस्या दर्ज करें](https://github.com/microsoft/ML-For-Beginners/issues)।

## सामग्री सूची

- [इंस्टॉलेशन समस्याएं](../..)
- [Jupyter Notebook समस्याएं](../..)
- [Python पैकेज समस्याएं](../..)
- [R एनवायरनमेंट समस्याएं](../..)
- [क्विज़ एप्लिकेशन समस्याएं](../..)
- [डेटा और फाइल पथ समस्याएं](../..)
- [आम त्रुटि संदेश](../..)
- [प्रदर्शन समस्याएं](../..)
- [एनवायरनमेंट और कॉन्फ़िगरेशन](../..)

---

## इंस्टॉलेशन समस्याएं

### Python इंस्टॉलेशन

**समस्या**: `python: command not found`

**समाधान**:
1. [python.org](https://www.python.org/downloads/) से Python 3.8 या उच्च संस्करण इंस्टॉल करें।
2. इंस्टॉलेशन सत्यापित करें: `python --version` या `python3 --version`
3. macOS/Linux पर, आपको `python` के बजाय `python3` का उपयोग करना पड़ सकता है।

**समस्या**: कई Python संस्करणों के कारण संघर्ष

**समाधान**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter इंस्टॉलेशन

**समस्या**: `jupyter: command not found`

**समाधान**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**समस्या**: Jupyter ब्राउज़र में लॉन्च नहीं हो रहा है

**समाधान**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R इंस्टॉलेशन

**समस्या**: R पैकेज इंस्टॉल नहीं हो रहे हैं

**समाधान**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**समस्या**: IRkernel Jupyter में उपलब्ध नहीं है

**समाधान**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook समस्याएं

### Kernel समस्याएं

**समस्या**: Kernel बार-बार बंद हो रहा है या रीस्टार्ट हो रहा है

**समाधान**:
1. Kernel को रीस्टार्ट करें: `Kernel → Restart`
2. आउटपुट साफ करें और रीस्टार्ट करें: `Kernel → Restart & Clear Output`
3. मेमोरी समस्याओं की जांच करें (देखें [प्रदर्शन समस्याएं](../..))
4. समस्या वाले कोड की पहचान करने के लिए कोशिकाओं को अलग-अलग चलाएं।

**समस्या**: गलत Python Kernel चुना गया है

**समाधान**:
1. वर्तमान Kernel की जांच करें: `Kernel → Change Kernel`
2. सही Python संस्करण चुनें।
3. यदि Kernel गायब है, तो इसे बनाएं:
```bash
python -m ipykernel install --user --name=ml-env
```

**समस्या**: Kernel शुरू नहीं हो रहा है

**समाधान**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook कोशिका समस्याएं

**समस्या**: कोशिकाएं चल रही हैं लेकिन आउटपुट नहीं दिखा रही हैं

**समाधान**:
1. जांचें कि कोशिका अभी भी चल रही है (देखें `[*]` संकेतक)।
2. Kernel को रीस्टार्ट करें और सभी कोशिकाएं चलाएं: `Kernel → Restart & Run All`
3. ब्राउज़र कंसोल में JavaScript त्रुटियों की जांच करें (F12)।

**समस्या**: कोशिकाएं नहीं चल रही हैं - "Run" पर क्लिक करने पर कोई प्रतिक्रिया नहीं

**समाधान**:
1. जांचें कि Jupyter सर्वर अभी भी टर्मिनल में चल रहा है।
2. ब्राउज़र पेज को रिफ्रेश करें।
3. Notebook को बंद करें और फिर से खोलें।
4. Jupyter सर्वर को रीस्टार्ट करें।

---

## Python पैकेज समस्याएं

### आयात त्रुटियां

**समस्या**: `ModuleNotFoundError: No module named 'sklearn'`

**समाधान**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**समस्या**: `ImportError: cannot import name 'X' from 'sklearn'`

**समाधान**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### संस्करण संघर्ष

**समस्या**: पैकेज संस्करण असंगतता त्रुटियां

**समाधान**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**समस्या**: `pip install` अनुमति त्रुटियों के साथ विफल हो रहा है

**समाधान**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### डेटा लोडिंग समस्याएं

**समस्या**: CSV फाइल लोड करते समय `FileNotFoundError`

**समाधान**:
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

## R एनवायरनमेंट समस्याएं

### पैकेज इंस्टॉलेशन

**समस्या**: पैकेज इंस्टॉलेशन संकलन त्रुटियों के साथ विफल हो रहा है

**समाधान**:
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

**समस्या**: `tidyverse` इंस्टॉल नहीं हो रहा है

**समाधान**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown समस्याएं

**समस्या**: RMarkdown रेंडर नहीं हो रहा है

**समाधान**:
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

## क्विज़ एप्लिकेशन समस्याएं

### बिल्ड और इंस्टॉलेशन

**समस्या**: `npm install` विफल हो रहा है

**समाधान**:
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

**समस्या**: पोर्ट 8080 पहले से उपयोग में है

**समाधान**:
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

### बिल्ड त्रुटियां

**समस्या**: `npm run build` विफल हो रहा है

**समाधान**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**समस्या**: लिंटिंग त्रुटियां बिल्ड को रोक रही हैं

**समाधान**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## डेटा और फाइल पथ समस्याएं

### पथ समस्याएं

**समस्या**: Notebook चलाते समय डेटा फाइलें नहीं मिल रही हैं

**समाधान**:
1. **हमेशा Notebook को उसकी कंटेनिंग डायरेक्टरी से चलाएं**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **कोड में रिलेटिव पथ की जांच करें**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **आवश्यक होने पर पूर्ण पथ का उपयोग करें**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### गायब डेटा फाइलें

**समस्या**: Dataset फाइलें गायब हैं

**समाधान**:
1. जांचें कि डेटा रिपॉजिटरी में होना चाहिए - अधिकांश डेटा सेट शामिल हैं।
2. कुछ पाठों में डेटा डाउनलोड करने की आवश्यकता हो सकती है - पाठ README देखें।
3. सुनिश्चित करें कि आपने नवीनतम परिवर्तन खींचे हैं:
   ```bash
   git pull origin main
   ```

---

## आम त्रुटि संदेश

### मेमोरी त्रुटियां

**त्रुटि**: `MemoryError` या डेटा प्रोसेसिंग करते समय Kernel बंद हो जाता है

**समाधान**:
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

### कन्वर्जेंस चेतावनियां

**चेतावनी**: `ConvergenceWarning: Maximum number of iterations reached`

**समाधान**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### प्लॉटिंग समस्याएं

**समस्या**: Jupyter में प्लॉट्स नहीं दिख रहे हैं

**समाधान**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**समस्या**: Seaborn प्लॉट्स अलग दिख रहे हैं या त्रुटियां दे रहे हैं

**समाधान**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### यूनिकोड/एन्कोडिंग त्रुटियां

**समस्या**: फाइल पढ़ते समय `UnicodeDecodeError`

**समाधान**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## प्रदर्शन समस्याएं

### धीमी Notebook निष्पादन

**समस्या**: Notebook चलाने में बहुत धीमा है

**समाधान**:
1. **मेमोरी खाली करने के लिए Kernel को रीस्टार्ट करें**: `Kernel → Restart`
2. **अप्रयुक्त Notebook बंद करें** ताकि संसाधन मुक्त हों।
3. **परीक्षण के लिए छोटे डेटा नमूनों का उपयोग करें**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **अपने कोड का प्रोफाइल करें** ताकि बाधाओं की पहचान हो सके:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### उच्च मेमोरी उपयोग

**समस्या**: सिस्टम मेमोरी खत्म हो रही है

**समाधान**:
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

## एनवायरनमेंट और कॉन्फ़िगरेशन

### वर्चुअल एनवायरनमेंट समस्याएं

**समस्या**: वर्चुअल एनवायरनमेंट सक्रिय नहीं हो रहा है

**समाधान**:
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

**समस्या**: पैकेज इंस्टॉल हैं लेकिन Notebook में नहीं मिल रहे हैं

**समाधान**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git समस्याएं

**समस्या**: नवीनतम परिवर्तन खींचने में असमर्थ - मर्ज संघर्ष

**समाधान**:
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

### VS Code इंटीग्रेशन

**समस्या**: Jupyter Notebook VS Code में नहीं खुल रहे हैं

**समाधान**:
1. VS Code में Python एक्सटेंशन इंस्टॉल करें।
2. VS Code में Jupyter एक्सटेंशन इंस्टॉल करें।
3. सही Python इंटरप्रेटर चुनें: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code को रीस्टार्ट करें।

---

## अतिरिक्त संसाधन

- **Discord Discussions**: [#ml-for-beginners चैनल में प्रश्न पूछें और समाधान साझा करें](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners मॉड्यूल](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **वीडियो ट्यूटोरियल्स**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)
- **इश्यू ट्रैकर**: [बग रिपोर्ट करें](https://github.com/microsoft/ML-For-Beginners/issues)

---

## अभी भी समस्याएं हो रही हैं?

यदि आपने ऊपर दिए गए समाधान आजमाए हैं और अभी भी समस्याओं का सामना कर रहे हैं:

1. **मौजूदा समस्याओं की खोज करें**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord में चर्चाओं की जांच करें**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **एक नई समस्या दर्ज करें**: इसमें शामिल करें:
   - आपका ऑपरेटिंग सिस्टम और संस्करण
   - Python/R संस्करण
   - त्रुटि संदेश (पूर्ण ट्रेसबैक)
   - समस्या को पुन: उत्पन्न करने के चरण
   - आपने पहले से क्या प्रयास किया है

हम आपकी मदद करने के लिए यहां हैं! 🚀

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में दस्तावेज़ को प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।