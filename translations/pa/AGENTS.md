<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:05:19+00:00",
  "source_file": "AGENTS.md",
  "language_code": "pa"
}
-->
# AGENTS.md

## ਪ੍ਰੋਜੈਕਟ ਝਲਕ

ਇਹ **ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਫਾਰ ਬਿਗਿਨਰਜ਼** ਹੈ, ਜੋ ਕਿ 12 ਹਫ਼ਤਿਆਂ, 26 ਪਾਠਾਂ ਦੀ ਵਿਸਤ੍ਰਿਤ ਪਾਠਕ੍ਰਮ ਹੈ। ਇਸ ਵਿੱਚ Python (ਮੁੱਖ ਤੌਰ 'ਤੇ Scikit-learn) ਅਤੇ R ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਕਲਾਸਿਕ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਸੰਕਲਪਾਂ ਨੂੰ ਕਵਰ ਕੀਤਾ ਗਿਆ ਹੈ। ਇਹ ਰਿਪੋਜ਼ਟਰੀ ਸਵੈ-ਗਤੀਵਾਨ ਸਿੱਖਣ ਦੇ ਸਰੋਤ ਵਜੋਂ ਡਿਜ਼ਾਈਨ ਕੀਤਾ ਗਿਆ ਹੈ, ਜਿਸ ਵਿੱਚ ਹੱਥ-ਅਨੁਭਵ ਪ੍ਰੋਜੈਕਟ, ਕਵਿਜ਼ ਅਤੇ ਅਸਾਈਨਮੈਂਟ ਸ਼ਾਮਲ ਹਨ। ਹਰ ਪਾਠ ਵਿਸ਼ਵ ਦੇ ਵੱਖ-ਵੱਖ ਸੱਭਿਆਚਾਰਾਂ ਅਤੇ ਖੇਤਰਾਂ ਦੇ ਅਸਲ ਡਾਟਾ ਰਾਹੀਂ ML ਸੰਕਲਪਾਂ ਦੀ ਪੜਚੋਲ ਕਰਦਾ ਹੈ।

ਮੁੱਖ ਹਿੱਸੇ:
- **ਸਿੱਖਣ ਸਮੱਗਰੀ**: 26 ਪਾਠਾਂ ਵਿੱਚ ML ਦਾ ਪਰਿਚਯ, ਰਿਗ੍ਰੈਸ਼ਨ, ਕਲਾਸੀਫਿਕੇਸ਼ਨ, ਕਲਸਟਰਿੰਗ, NLP, ਟਾਈਮ ਸੀਰੀਜ਼, ਅਤੇ ਰੀਇਨਫੋਰਸਮੈਂਟ ਲਰਨਿੰਗ ਸ਼ਾਮਲ ਹਨ।
- **ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ**: Vue.js-ਅਧਾਰਿਤ ਕਵਿਜ਼ ਐਪ, ਪਾਠ ਤੋਂ ਪਹਿਲਾਂ ਅਤੇ ਬਾਅਦ ਦੇ ਮੁਲਾਂਕਣਾਂ ਨਾਲ।
- **ਬਹੁ-ਭਾਸ਼ਾ ਸਹਾਇਤਾ**: GitHub Actions ਰਾਹੀਂ 40+ ਭਾਸ਼ਾਵਾਂ ਵਿੱਚ ਆਟੋਮੈਟਿਕ ਅਨੁਵਾਦ।
- **ਦੁਅਲ ਭਾਸ਼ਾ ਸਹਾਇਤਾ**: ਪਾਠ Python (Jupyter notebooks) ਅਤੇ R (R Markdown files) ਵਿੱਚ ਉਪਲਬਧ।
- **ਪ੍ਰੋਜੈਕਟ-ਅਧਾਰਿਤ ਸਿੱਖਣ**: ਹਰ ਵਿਸ਼ੇ ਵਿੱਚ ਪ੍ਰੈਕਟਿਕਲ ਪ੍ਰੋਜੈਕਟ ਅਤੇ ਅਸਾਈਨਮੈਂਟ ਸ਼ਾਮਲ ਹਨ।

## ਰਿਪੋਜ਼ਟਰੀ ਸਟ੍ਰਕਚਰ

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

ਹਰ ਪਾਠ ਫੋਲਡਰ ਵਿੱਚ ਆਮ ਤੌਰ 'ਤੇ ਇਹ ਸ਼ਾਮਲ ਹੁੰਦਾ ਹੈ:
- `README.md` - ਮੁੱਖ ਪਾਠ ਸਮੱਗਰੀ
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - ਹੱਲ ਕੋਡ (Python ਅਤੇ R ਵਰਜਨ)
- `assignment.md` - ਅਭਿਆਸ ਅਸਾਈਨਮੈਂਟ
- `images/` - ਵਿਜ਼ੁਅਲ ਸਰੋਤ

## ਸੈਟਅਪ ਕਮਾਂਡ

### Python ਪਾਠਾਂ ਲਈ

ਅਧਿਕਤਰ ਪਾਠ Jupyter notebooks ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹਨ। ਲੋੜੀਂਦੇ ਡਿਪੈਂਡੈਂਸੀਜ਼ ਇੰਸਟਾਲ ਕਰੋ:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### R ਪਾਠਾਂ ਲਈ

R ਪਾਠ `solution/R/` ਫੋਲਡਰ ਵਿੱਚ `.rmd` ਜਾਂ `.ipynb` ਫਾਈਲਾਂ ਦੇ ਰੂਪ ਵਿੱਚ ਹਨ:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਲਈ

ਕਵਿਜ਼ ਐਪ Vue.js ਐਪਲੀਕੇਸ਼ਨ ਹੈ ਜੋ `quiz-app/` ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ ਸਥਿਤ ਹੈ:

```bash
cd quiz-app
npm install
```

### ਡੌਕੂਮੈਂਟੇਸ਼ਨ ਸਾਈਟ ਲਈ

ਡੌਕੂਮੈਂਟੇਸ਼ਨ ਨੂੰ ਲੋਕਲ ਰਨ ਕਰਨ ਲਈ:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## ਵਿਕਾਸ ਵਰਕਫਲੋ

### ਪਾਠ ਨੋਟਬੁੱਕ ਨਾਲ ਕੰਮ ਕਰਨਾ

1. ਪਾਠ ਡਾਇਰੈਕਟਰੀ ਵਿੱਚ ਜਾਓ (ਜਿਵੇਂ `2-Regression/1-Tools/`)
2. Jupyter notebook ਖੋਲ੍ਹੋ:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. ਪਾਠ ਸਮੱਗਰੀ ਅਤੇ ਅਭਿਆਸਾਂ 'ਤੇ ਕੰਮ ਕਰੋ
4. ਜਰੂਰਤ ਪੈਣ 'ਤੇ `solution/` ਫੋਲਡਰ ਵਿੱਚ ਹੱਲ ਚੈੱਕ ਕਰੋ

### Python ਵਿਕਾਸ

- ਪਾਠ ਸਧਾਰਨ Python ਡਾਟਾ ਸਾਇੰਸ ਲਾਇਬ੍ਰੇਰੀਆਂ ਦੀ ਵਰਤੋਂ ਕਰਦੇ ਹਨ
- ਇੰਟਰਐਕਟਿਵ ਸਿੱਖਣ ਲਈ Jupyter notebooks
- ਹੱਲ ਕੋਡ ਹਰ ਪਾਠ ਦੇ `solution/` ਫੋਲਡਰ ਵਿੱਚ ਉਪਲਬਧ ਹੈ

### R ਵਿਕਾਸ

- R ਪਾਠ `.rmd` ਫਾਰਮੈਟ (R Markdown) ਵਿੱਚ ਹਨ
- ਹੱਲ `solution/R/` ਸਬਡਾਇਰੈਕਟਰੀ ਵਿੱਚ ਸਥਿਤ ਹਨ
- RStudio ਜਾਂ Jupyter ਨਾਲ R kernel ਦੀ ਵਰਤੋਂ ਕਰਕੇ R notebooks ਚਲਾਓ

### ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਵਿਕਾਸ

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## ਟੈਸਟਿੰਗ ਨਿਰਦੇਸ਼

### ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਟੈਸਟਿੰਗ

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**ਨੋਟ**: ਇਹ ਮੁੱਖ ਤੌਰ 'ਤੇ ਇੱਕ ਸਿੱਖਣ ਪਾਠਕ੍ਰਮ ਰਿਪੋਜ਼ਟਰੀ ਹੈ। ਪਾਠ ਸਮੱਗਰੀ ਲਈ ਕੋਈ ਆਟੋਮੈਟਿਕ ਟੈਸਟ ਨਹੀਂ ਹਨ। ਵੈਰੀਫਿਕੇਸ਼ਨ ਇਸ ਤਰ੍ਹਾਂ ਕੀਤੀ ਜਾਂਦੀ ਹੈ:
- ਪਾਠ ਅਭਿਆਸ ਪੂਰੇ ਕਰਨਾ
- ਨੋਟਬੁੱਕ ਸੈੱਲਾਂ ਨੂੰ ਸਫਲਤਾਪੂਰਵਕ ਚਲਾਉਣਾ
- ਹੱਲਾਂ ਵਿੱਚ ਉਮੀਦ ਕੀਤੇ ਨਤੀਜਿਆਂ ਦੇ ਖਿਲਾਫ ਆਉਟਪੁੱਟ ਦੀ ਜਾਂਚ

## ਕੋਡ ਸਟਾਈਲ ਗਾਈਡਲਾਈਨ

### Python ਕੋਡ
- PEP 8 ਸਟਾਈਲ ਗਾਈਡਲਾਈਨ ਦੀ ਪਾਲਣਾ ਕਰੋ
- ਸਪਸ਼ਟ, ਵਰਣਨਾਤਮਕ ਵੈਰੀਏਬਲ ਨਾਮ ਵਰਤੋ
- ਜਟਿਲ ਕਾਰਵਾਈਆਂ ਲਈ ਟਿੱਪਣੀਆਂ ਸ਼ਾਮਲ ਕਰੋ
- Jupyter notebooks ਵਿੱਚ ਸੰਕਲਪਾਂ ਨੂੰ ਸਮਝਾਉਣ ਵਾਲੇ ਮਾਰਕਡਾਊਨ ਸੈੱਲ ਹੋਣੇ ਚਾਹੀਦੇ ਹਨ

### JavaScript/Vue.js (ਕਵਿਜ਼ ਐਪ)
- Vue.js ਸਟਾਈਲ ਗਾਈਡ ਦੀ ਪਾਲਣਾ ਕਰਦਾ ਹੈ
- ESLint ਕਨਫਿਗਰੇਸ਼ਨ `quiz-app/package.json` ਵਿੱਚ
- ਮੁੱਦੇ ਚੈੱਕ ਕਰਨ ਅਤੇ ਆਟੋ-ਫਿਕਸ ਕਰਨ ਲਈ `npm run lint` ਚਲਾਓ

### ਡੌਕੂਮੈਂਟੇਸ਼ਨ
- ਮਾਰਕਡਾਊਨ ਫਾਈਲਾਂ ਸਪਸ਼ਟ ਅਤੇ ਚੰਗੀ ਤਰ੍ਹਾਂ ਸਟ੍ਰਕਚਰਡ ਹੋਣੀ ਚਾਹੀਦੀ ਹੈ
- ਫੈਂਸਡ ਕੋਡ ਬਲਾਕਾਂ ਵਿੱਚ ਕੋਡ ਉਦਾਹਰਨ ਸ਼ਾਮਲ ਕਰੋ
- ਅੰਦਰੂਨੀ ਰਿਫਰੰਸਾਂ ਲਈ ਰਿਲੇਟਿਵ ਲਿੰਕ ਵਰਤੋ
- ਮੌਜੂਦਾ ਫਾਰਮੈਟਿੰਗ ਰਿਵਾਜਾਂ ਦੀ ਪਾਲਣਾ ਕਰੋ

## ਬਿਲਡ ਅਤੇ ਡਿਪਲੌਇਮੈਂਟ

### ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਡਿਪਲੌਇਮੈਂਟ

ਕਵਿਜ਼ ਐਪ ਨੂੰ Azure Static Web Apps 'ਤੇ ਡਿਪਲੌਇ ਕੀਤਾ ਜਾ ਸਕਦਾ ਹੈ:

1. **ਪੂਰਵ ਸ਼ਰਤਾਂ**:
   - Azure ਖਾਤਾ
   - GitHub ਰਿਪੋਜ਼ਟਰੀ (ਪਹਿਲਾਂ ਹੀ ਫੋਰਕ ਕੀਤਾ ਹੋਇਆ)

2. **Azure 'ਤੇ ਡਿਪਲੌਇ ਕਰੋ**:
   - Azure Static Web App ਸਰੋਤ ਬਣਾਓ
   - GitHub ਰਿਪੋਜ਼ਟਰੀ ਨਾਲ ਕਨੈਕਟ ਕਰੋ
   - ਐਪ ਸਥਾਨ ਸੈਟ ਕਰੋ: `/quiz-app`
   - ਆਉਟਪੁੱਟ ਸਥਾਨ ਸੈਟ ਕਰੋ: `dist`
   - Azure ਆਟੋਮੈਟਿਕ ਤੌਰ 'ਤੇ GitHub Actions ਵਰਕਫਲੋ ਬਣਾਉਂਦਾ ਹੈ

3. **GitHub Actions ਵਰਕਫਲੋ**:
   - ਵਰਕਫਲੋ ਫਾਈਲ `.github/workflows/azure-static-web-apps-*.yml` ਵਿੱਚ ਬਣਾਈ ਜਾਂਦੀ ਹੈ
   - ਮੁੱਖ ਸ਼ਾਖਾ 'ਤੇ ਪੁਸ਼ ਕਰਨ 'ਤੇ ਆਟੋਮੈਟਿਕ ਤੌਰ 'ਤੇ ਬਿਲਡ ਅਤੇ ਡਿਪਲੌਇ ਕਰਦਾ ਹੈ

### ਡੌਕੂਮੈਂਟੇਸ਼ਨ PDF

ਡੌਕੂਮੈਂਟੇਸ਼ਨ ਤੋਂ PDF ਬਣਾਓ:

```bash
npm install
npm run convert
```

## ਅਨੁਵਾਦ ਵਰਕਫਲੋ

**ਮਹੱਤਵਪੂਰਨ**: ਅਨੁਵਾਦ GitHub Actions ਰਾਹੀਂ Co-op Translator ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਆਟੋਮੈਟਿਕ ਹਨ।

- ਜਦੋਂ `main` ਸ਼ਾਖਾ 'ਤੇ ਬਦਲਾਅ ਪੇਸ਼ ਕੀਤੇ ਜਾਂਦੇ ਹਨ, ਅਨੁਵਾਦ ਆਟੋ-ਜਨਰੇਟ ਹੁੰਦੇ ਹਨ
- **ਸਮੱਗਰੀ ਨੂੰ ਹੱਥੋਂ ਅਨੁਵਾਦ ਨਾ ਕਰੋ** - ਸਿਸਟਮ ਇਸ ਨੂੰ ਸੰਭਾਲਦਾ ਹੈ
- ਵਰਕਫਲੋ `.github/workflows/co-op-translator.yml` ਵਿੱਚ ਪਰਿਭਾਸ਼ਿਤ ਹੈ
- ਅਨੁਵਾਦ ਲਈ Azure AI/OpenAI ਸੇਵਾਵਾਂ ਦੀ ਵਰਤੋਂ ਕਰਦਾ ਹੈ
- 40+ ਭਾਸ਼ਾਵਾਂ ਦਾ ਸਮਰਥਨ ਕਰਦਾ ਹੈ

## ਯੋਗਦਾਨ ਦੇ ਨਿਰਦੇਸ਼

### ਸਮੱਗਰੀ ਯੋਗਦਾਨਕਰਤਿਆਂ ਲਈ

1. **ਰਿਪੋਜ਼ਟਰੀ ਫੋਰਕ ਕਰੋ** ਅਤੇ ਇੱਕ ਫੀਚਰ ਸ਼ਾਖਾ ਬਣਾਓ
2. **ਪਾਠ ਸਮੱਗਰੀ ਵਿੱਚ ਬਦਲਾਅ ਕਰੋ** ਜੇਕਰ ਪਾਠ ਸ਼ਾਮਲ/ਅਪਡੇਟ ਕਰ ਰਹੇ ਹੋ
3. **ਅਨੁਵਾਦ ਕੀਤੀਆਂ ਫਾਈਲਾਂ ਨੂੰ ਸੋਧੋ ਨਾ** - ਇਹ ਆਟੋ-ਜਨਰੇਟ ਕੀਤੀਆਂ ਜਾਂਦੀਆਂ ਹਨ
4. **ਆਪਣੇ ਕੋਡ ਦੀ ਜਾਂਚ ਕਰੋ** - ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਸਾਰੇ ਨੋਟਬੁੱਕ ਸੈੱਲ ਸਫਲਤਾਪੂਰਵਕ ਚਲਦੇ ਹਨ
5. **ਲਿੰਕ ਅਤੇ ਚਿੱਤਰਾਂ ਦੀ ਜਾਂਚ ਕਰੋ** ਕਿ ਇਹ ਸਹੀ ਕੰਮ ਕਰ ਰਹੇ ਹਨ
6. **ਸਪਸ਼ਟ ਵਰਣਨ ਦੇ ਨਾਲ ਪੁਲ ਰਿਕਵੇਸਟ ਪੇਸ਼ ਕਰੋ**

### ਪੁਲ ਰਿਕਵੇਸਟ ਨਿਰਦੇਸ਼

- **ਟਾਈਟਲ ਫਾਰਮੈਟ**: `[Section] ਬਦਲਾਅ ਦਾ ਸੰਖੇਪ ਵਰਣਨ`
  - ਉਦਾਹਰਨ: `[Regression] ਪਾਠ 5 ਵਿੱਚ ਟਾਈਪੋ ਠੀਕ ਕਰੋ`
  - ਉਦਾਹਰਨ: `[Quiz-App] ਡਿਪੈਂਡੈਂਸੀਜ਼ ਅਪਡੇਟ ਕਰੋ`
- **ਪੇਸ਼ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ**:
  - ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਸਾਰੇ ਨੋਟਬੁੱਕ ਸੈੱਲ ਬਿਨਾਂ ਗਲਤੀ ਦੇ ਚਲਦੇ ਹਨ
  - ਜੇਕਰ ਕਵਿਜ਼-ਐਪ ਨੂੰ ਸੋਧ ਰਹੇ ਹੋ ਤਾਂ `npm run lint` ਚਲਾਓ
  - ਮਾਰਕਡਾਊਨ ਫਾਰਮੈਟਿੰਗ ਦੀ ਜਾਂਚ ਕਰੋ
  - ਕੋਈ ਨਵਾਂ ਕੋਡ ਉਦਾਹਰਨ ਟੈਸਟ ਕਰੋ
- **PR ਵਿੱਚ ਸ਼ਾਮਲ ਹੋਣਾ ਚਾਹੀਦਾ ਹੈ**:
  - ਬਦਲਾਅ ਦਾ ਵਰਣਨ
  - ਬਦਲਾਅ ਦਾ ਕਾਰਨ
  - ਜੇ UI ਬਦਲਾਅ ਹਨ ਤਾਂ ਸਕ੍ਰੀਨਸ਼ਾਟ
- **ਕੋਡ ਆਫ ਕੰਡਕਟ**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) ਦੀ ਪਾਲਣਾ ਕਰੋ
- **CLA**: ਤੁਹਾਨੂੰ Contributor License Agreement 'ਤੇ ਦਸਤਖਤ ਕਰਨੇ ਪੈਣਗੇ

## ਪਾਠ ਸਟ੍ਰਕਚਰ

ਹਰ ਪਾਠ ਇੱਕ ਸਥਿਰ ਪੈਟਰਨ ਦੀ ਪਾਲਣਾ ਕਰਦਾ ਹੈ:

1. **ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼** - ਬੇਸਲਾਈਨ ਗਿਆਨ ਦੀ ਜਾਂਚ ਕਰੋ
2. **ਪਾਠ ਸਮੱਗਰੀ** - ਲਿਖਤ ਨਿਰਦੇਸ਼ ਅਤੇ ਵਿਆਖਿਆ
3. **ਕੋਡ ਡੈਮੋਨਸਟਰੇਸ਼ਨ** - ਨੋਟਬੁੱਕ ਵਿੱਚ ਹੱਥ-ਅਨੁਭਵ ਉਦਾਹਰਨ
4. **ਗਿਆਨ ਦੀ ਜਾਂਚ** - ਸਮਝ ਦੀ ਪੁਸ਼ਟੀ ਕਰੋ
5. **ਚੈਲੈਂਜ** - ਸੰਕਲਪਾਂ ਨੂੰ ਸਵੈ-ਨਿਰਭਰਤਾ ਨਾਲ ਲਾਗੂ ਕਰੋ
6. **ਅਸਾਈਨਮੈਂਟ** - ਵਿਆਪਕ ਅਭਿਆਸ
7. **ਪੋਸਟ-ਲੈਕਚਰ ਕਵਿਜ਼** - ਸਿੱਖਣ ਦੇ ਨਤੀਜੇ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ

## ਆਮ ਕਮਾਂਡ ਰਿਫਰੈਂਸ

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## ਵਾਧੂ ਸਰੋਤ

- **Microsoft Learn Collection**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **ਕਵਿਜ਼ ਐਪ**: [ਆਨਲਾਈਨ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)
- **ਚਰਚਾ ਬੋਰਡ**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **ਵੀਡੀਓ ਵਾਕਥਰੂ**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## ਮੁੱਖ ਤਕਨਾਲੋਜੀਆਂ

- **Python**: ML ਪਾਠਾਂ ਲਈ ਮੁੱਖ ਭਾਸ਼ਾ (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਵਿਕਲਪਕ ਇੰਪਲੀਮੈਂਟੇਸ਼ਨ
- **Jupyter**: Python ਪਾਠਾਂ ਲਈ ਇੰਟਰਐਕਟਿਵ ਨੋਟਬੁੱਕ
- **R Markdown**: R ਪਾਠਾਂ ਲਈ ਦਸਤਾਵੇਜ਼
- **Vue.js 3**: ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ ਫਰੇਮਵਰਕ
- **Flask**: ML ਮਾਡਲ ਡਿਪਲੌਇਮੈਂਟ ਲਈ ਵੈਬ ਐਪਲੀਕੇਸ਼ਨ ਫਰੇਮਵਰਕ
- **Docsify**: ਡੌਕੂਮੈਂਟੇਸ਼ਨ ਸਾਈਟ ਜਨਰੇਟਰ
- **GitHub Actions**: CI/CD ਅਤੇ ਆਟੋਮੈਟਿਕ ਅਨੁਵਾਦ

## ਸੁਰੱਖਿਆ ਵਿਚਾਰ

- **ਕੋਡ ਵਿੱਚ ਕੋਈ ਗੁਪਤ ਜਾਣਕਾਰੀ ਨਹੀਂ**: API keys ਜਾਂ credentials ਕਦੇ ਵੀ commit ਨਾ ਕਰੋ
- **ਡਿਪੈਂਡੈਂਸੀਜ਼**: npm ਅਤੇ pip ਪੈਕੇਜ ਨੂੰ ਅਪਡੇਟ ਰੱਖੋ
- **ਯੂਜ਼ਰ ਇਨਪੁਟ**: Flask ਵੈਬ ਐਪ ਉਦਾਹਰਨਾਂ ਵਿੱਚ ਬੁਨਿਆਦੀ ਇਨਪੁਟ ਵੈਰੀਫਿਕੇਸ਼ਨ ਸ਼ਾਮਲ ਹੈ
- **ਸੰਵੇਦਨਸ਼ੀਲ ਡਾਟਾ**: ਉਦਾਹਰਨ ਡਾਟਾਸੈਟ ਜਨਤਕ ਅਤੇ ਗੈਰ-ਸੰਵੇਦਨਸ਼ੀਲ ਹਨ

## ਟਰਬਲਸ਼ੂਟਿੰਗ

### Jupyter ਨੋਟਬੁੱਕ

- **Kernel ਸਮੱਸਿਆਵਾਂ**: ਜੇ ਸੈੱਲ ਰੁਕ ਜਾਂਦੇ ਹਨ ਤਾਂ Kernel → Restart ਕਰੋ
- **Import errors**: ਯਕੀਨੀ ਬਣਾਓ ਕਿ ਸਾਰੇ ਲੋੜੀਂਦੇ ਪੈਕੇਜ pip ਨਾਲ ਇੰਸਟਾਲ ਕੀਤੇ ਗਏ ਹਨ
- **Path issues**: ਨੋਟਬੁੱਕ ਨੂੰ ਉਸ ਡਾਇਰੈਕਟਰੀ ਤੋਂ ਚਲਾਓ ਜਿਸ ਵਿੱਚ ਇਹ ਸਥਿਤ ਹੈ

### ਕਵਿਜ਼ ਐਪਲੀਕੇਸ਼ਨ

- **npm install ਫੇਲ੍ਹ**: npm cache ਸਾਫ ਕਰੋ: `npm cache clean --force`
- **Port conflicts**: ਪੋਰਟ ਬਦਲੋ: `npm run serve -- --port 8081`
- **Build errors**: `node_modules` ਡਿਲੀਟ ਕਰੋ ਅਤੇ ਮੁੜ ਇੰਸਟਾਲ ਕਰੋ: `rm -rf node_modules && npm install`

### R ਪਾਠ

- **ਪੈਕੇਜ ਨਹੀਂ ਮਿਲਿਆ**: `install.packages("package-name")` ਨਾਲ ਇੰਸਟਾਲ ਕਰੋ
- **RMarkdown rendering**: ਯਕੀਨੀ ਬਣਾਓ ਕਿ rmarkdown ਪੈਕੇਜ ਇੰਸਟਾਲ ਹੈ
- **Kernel issues**: Jupyter ਲਈ IRkernel ਇੰਸਟਾਲ ਕਰਨ ਦੀ ਲੋੜ ਹੋ ਸਕਦੀ ਹੈ

## ਪ੍ਰੋਜੈਕਟ-ਵਿਸ਼ੇਸ਼ ਨੋਟਸ

- ਇਹ ਮੁੱਖ ਤੌਰ 'ਤੇ **ਸਿੱਖਣ ਪਾਠਕ੍ਰਮ** ਹੈ, ਪ੍ਰੋਡਕਸ਼ਨ ਕੋਡ ਨਹੀਂ
- ਹੱਥ-ਅਨੁਭਵ ਰਾਹੀਂ **ML ਸੰਕਲਪਾਂ ਨੂੰ ਸਮਝਣ** 'ਤੇ ਧਿਆਨ ਹੈ
- ਕੋਡ ਉਦਾਹਰਨ **ਸਪਸ਼ਟਤਾ ਨੂੰ ਤਰਜੀਹ ਦਿੰਦੇ ਹਨ** ਨਾ ਕਿ ਅਪਟਿਮਾਈਜ਼ੇਸ਼ਨ ਨੂੰ
- ਜ਼ਿਆਦਾਤਰ ਪਾਠ **ਸਵੈ-ਨਿਰਭਰ** ਹਨ ਅਤੇ ਵੱਖ-ਵੱਖ ਪੂਰੇ ਕੀਤੇ ਜਾ ਸਕਦੇ ਹਨ
- **ਹੱਲ ਉਪਲਬਧ ਹਨ** ਪਰ ਸਿੱਖਣ ਵਾਲਿਆਂ ਨੂੰ ਪਹਿਲਾਂ ਅਭਿਆਸ ਕਰਨ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ
- ਰਿਪੋਜ਼ਟਰੀ **Docsify** ਦੀ ਵਰਤੋਂ ਕਰਦਾ ਹੈ ਵੈਬ ਡੌਕੂਮੈਂਟੇਸ਼ਨ ਲਈ ਬਿਨਾਂ ਬਿਲਡ ਸਟੈਪ ਦੇ
- **Sketchnotes** ਸੰਕਲਪਾਂ ਦੇ ਵਿਜ਼ੁਅਲ ਸਾਰਾਂ ਪ੍ਰਦਾਨ ਕਰਦੇ ਹਨ
- **ਬਹੁ-ਭਾਸ਼ਾ ਸਹਾਇਤਾ** ਸਮੱਗਰੀ ਨੂੰ ਵਿਸ਼ਵ ਪੱਧਰ 'ਤੇ ਪਹੁੰਚਯੋਗ ਬਣਾਉਂਦੀ ਹੈ

---

**ਅਸਵੀਕਰਤਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਣੀਕਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਮੌਜੂਦ ਅਸਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤ ਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।