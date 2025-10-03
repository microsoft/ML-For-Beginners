<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:59:25+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ur"
}
-->
# AGENTS.md

## پروجیکٹ کا جائزہ

یہ **مشین لرننگ فار بیگنرز** ہے، ایک جامع 12 ہفتوں کا، 26 اسباق پر مشتمل نصاب جو Python (زیادہ تر Scikit-learn کے ساتھ) اور R کا استعمال کرتے ہوئے کلاسک مشین لرننگ کے تصورات کا احاطہ کرتا ہے۔ یہ ریپوزٹری خود سے سیکھنے کے وسائل کے طور پر ڈیزائن کی گئی ہے، جس میں عملی پروجیکٹس، کوئزز، اور اسائنمنٹس شامل ہیں۔ ہر سبق حقیقی دنیا کے ڈیٹا کے ذریعے مختلف ثقافتوں اور علاقوں سے ML کے تصورات کو دریافت کرتا ہے۔

اہم اجزاء:
- **تعلیمی مواد**: 26 اسباق جن میں ML کا تعارف، ریگریشن، کلاسیفیکیشن، کلسٹرنگ، NLP، ٹائم سیریز، اور ریئنفورسمنٹ لرننگ شامل ہیں
- **کوئز ایپلیکیشن**: Vue.js پر مبنی کوئز ایپ، سبق سے پہلے اور بعد کے جائزے کے ساتھ
- **کثیر زبان کی حمایت**: GitHub Actions کے ذریعے 40+ زبانوں میں خودکار ترجمے
- **دوہری زبان کی حمایت**: اسباق Python (Jupyter نوٹ بکس) اور R (R Markdown فائلز) دونوں میں دستیاب ہیں
- **پروجیکٹ پر مبنی سیکھنا**: ہر موضوع میں عملی پروجیکٹس اور اسائنمنٹس شامل ہیں

## ریپوزٹری کا ڈھانچہ

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

ہر سبق کے فولڈر میں عام طور پر شامل ہوتا ہے:
- `README.md` - مرکزی سبق کا مواد
- `notebook.ipynb` - Python Jupyter نوٹ بک
- `solution/` - حل کا کوڈ (Python اور R ورژنز)
- `assignment.md` - مشق کے لیے اسائنمنٹس
- `images/` - بصری وسائل

## سیٹ اپ کمانڈز

### Python اسباق کے لیے

زیادہ تر اسباق Jupyter نوٹ بکس استعمال کرتے ہیں۔ مطلوبہ ڈپینڈنسیز انسٹال کریں:

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

### R اسباق کے لیے

R اسباق `solution/R/` فولڈرز میں `.rmd` یا `.ipynb` فائلز کے طور پر موجود ہیں:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### کوئز ایپلیکیشن کے لیے

کوئز ایپ Vue.js ایپلیکیشن ہے جو `quiz-app/` ڈائریکٹری میں موجود ہے:

```bash
cd quiz-app
npm install
```

### دستاویزات سائٹ کے لیے

دستاویزات کو مقامی طور پر چلانے کے لیے:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## ترقیاتی ورک فلو

### سبق کی نوٹ بکس کے ساتھ کام کرنا

1. سبق کی ڈائریکٹری پر جائیں (مثلاً، `2-Regression/1-Tools/`)
2. Jupyter نوٹ بک کھولیں:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. سبق کے مواد اور مشقوں پر کام کریں
4. اگر ضرورت ہو تو `solution/` فولڈر میں حل چیک کریں

### Python ترقی

- اسباق معیاری Python ڈیٹا سائنس لائبریریز استعمال کرتے ہیں
- انٹرایکٹو سیکھنے کے لیے Jupyter نوٹ بکس
- ہر سبق کے `solution/` فولڈر میں حل کا کوڈ دستیاب ہے

### R ترقی

- R اسباق `.rmd` فارمیٹ (R Markdown) میں ہیں
- حل `solution/R/` سب ڈائریکٹریز میں موجود ہیں
- RStudio یا Jupyter کے ساتھ R کرنل استعمال کریں R نوٹ بکس چلانے کے لیے

### کوئز ایپلیکیشن ترقی

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

## ٹیسٹنگ ہدایات

### کوئز ایپلیکیشن ٹیسٹنگ

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**نوٹ**: یہ بنیادی طور پر ایک تعلیمی نصاب کی ریپوزٹری ہے۔ سبق کے مواد کے لیے کوئی خودکار ٹیسٹ نہیں ہیں۔ توثیق درج ذیل طریقوں سے کی جاتی ہے:
- سبق کی مشقیں مکمل کرنا
- نوٹ بک سیلز کو کامیابی سے چلانا
- حل میں متوقع نتائج کے خلاف آؤٹ پٹ چیک کرنا

## کوڈ اسٹائل گائیڈ لائنز

### Python کوڈ
- PEP 8 اسٹائل گائیڈ لائنز پر عمل کریں
- واضح اور وضاحتی ویریبل نام استعمال کریں
- پیچیدہ آپریشنز کے لیے تبصرے شامل کریں
- Jupyter نوٹ بکس میں تصورات کی وضاحت کرنے والے مارک ڈاؤن سیلز ہونے چاہئیں

### JavaScript/Vue.js (کوئز ایپ)
- Vue.js اسٹائل گائیڈ پر عمل کریں
- ESLint کنفیگریشن `quiz-app/package.json` میں موجود ہے
- `npm run lint` چلائیں مسائل چیک کرنے اور خودکار طور پر ٹھیک کرنے کے لیے

### دستاویزات
- مارک ڈاؤن فائلز واضح اور اچھی طرح سے ساختہ ہونی چاہئیں
- fenced کوڈ بلاکس میں کوڈ کی مثالیں شامل کریں
- اندرونی حوالوں کے لیے نسبتی لنکس استعمال کریں
- موجودہ فارمیٹنگ کنونشنز پر عمل کریں

## بلڈ اور ڈیپلائمنٹ

### کوئز ایپلیکیشن ڈیپلائمنٹ

کوئز ایپ Azure Static Web Apps پر ڈیپلائی کی جا سکتی ہے:

1. **ضروریات**:
   - Azure اکاؤنٹ
   - GitHub ریپوزٹری (پہلے سے فورک شدہ)

2. **Azure پر ڈیپلائی کریں**:
   - Azure Static Web App ریسورس بنائیں
   - GitHub ریپوزٹری سے کنیکٹ کریں
   - ایپ لوکیشن سیٹ کریں: `/quiz-app`
   - آؤٹ پٹ لوکیشن سیٹ کریں: `dist`
   - Azure خودکار طور پر GitHub Actions ورک فلو بناتا ہے

3. **GitHub Actions ورک فلو**:
   - ورک فلو فائل `.github/workflows/azure-static-web-apps-*.yml` میں بنائی جاتی ہے
   - مین برانچ پر پش کرنے پر خودکار طور پر بلڈ اور ڈیپلائی کرتا ہے

### دستاویزات PDF

دستاویزات سے PDF بنائیں:

```bash
npm install
npm run convert
```

## ترجمہ ورک فلو

**اہم**: ترجمے GitHub Actions کے ذریعے Co-op Translator کا استعمال کرتے ہوئے خودکار ہیں۔

- جب تبدیلیاں `main` برانچ پر پش کی جاتی ہیں تو ترجمے خودکار طور پر بنائے جاتے ہیں
- **مواد کو دستی طور پر ترجمہ نہ کریں** - سسٹم اس کو سنبھالتا ہے
- ورک فلو `.github/workflows/co-op-translator.yml` میں بیان کیا گیا ہے
- ترجمے کے لیے Azure AI/OpenAI سروسز استعمال کرتا ہے
- 40+ زبانوں کی حمایت کرتا ہے

## تعاون کی گائیڈ لائنز

### مواد کے تعاون کنندگان کے لیے

1. **ریپوزٹری کو فورک کریں** اور ایک فیچر برانچ بنائیں
2. **سبق کے مواد میں تبدیلی کریں** اگر اسباق شامل/اپ ڈیٹ کر رہے ہیں
3. **ترجمہ شدہ فائلز میں ترمیم نہ کریں** - وہ خودکار طور پر بنائی جاتی ہیں
4. **اپنے کوڈ کی جانچ کریں** - یقینی بنائیں کہ تمام نوٹ بک سیلز کامیابی سے چلتے ہیں
5. **لنکس اور تصاویر کی تصدیق کریں** کہ وہ صحیح کام کر رہے ہیں
6. **ایک واضح وضاحت کے ساتھ پل ریکویسٹ جمع کریں**

### پل ریکویسٹ گائیڈ لائنز

- **عنوان کا فارمیٹ**: `[سیکشن] تبدیلیوں کی مختصر وضاحت`
  - مثال: `[Regression] سبق 5 میں ٹائپو درست کریں`
  - مثال: `[Quiz-App] ڈپینڈنسیز اپ ڈیٹ کریں`
- **جمع کرنے سے پہلے**:
  - یقینی بنائیں کہ تمام نوٹ بک سیلز بغیر کسی غلطی کے چلتے ہیں
  - اگر کوئز ایپ میں ترمیم کر رہے ہیں تو `npm run lint` چلائیں
  - مارک ڈاؤن فارمیٹنگ کی تصدیق کریں
  - کسی بھی نئے کوڈ کی مثالوں کی جانچ کریں
- **PR میں شامل ہونا چاہیے**:
  - تبدیلیوں کی وضاحت
  - تبدیلیوں کی وجہ
  - UI تبدیلیوں کے لیے اسکرین شاٹس
- **کوڈ آف کنڈکٹ**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) پر عمل کریں
- **CLA**: آپ کو Contributor License Agreement پر دستخط کرنے کی ضرورت ہوگی

## سبق کا ڈھانچہ

ہر سبق ایک مستقل پیٹرن پر عمل کرتا ہے:

1. **سبق سے پہلے کا کوئز** - بنیادی معلومات کی جانچ کریں
2. **سبق کا مواد** - تحریری ہدایات اور وضاحتیں
3. **کوڈ مظاہرے** - نوٹ بکس میں عملی مثالیں
4. **علم کی جانچ** - پورے سبق میں سمجھ کی تصدیق کریں
5. **چیلنج** - تصورات کو خود سے لاگو کریں
6. **اسائنمنٹ** - توسیعی مشق
7. **سبق کے بعد کا کوئز** - سیکھنے کے نتائج کا جائزہ لیں

## عام کمانڈز کا حوالہ

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

## اضافی وسائل

- **Microsoft Learn Collection**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **کوئز ایپ**: [آن لائن کوئزز](https://ff-quizzes.netlify.app/en/ml/)
- **ڈسکشن بورڈ**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **ویڈیو واک تھرو**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## کلیدی ٹیکنالوجیز

- **Python**: ML اسباق کے لیے بنیادی زبان (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret کا استعمال کرتے ہوئے متبادل عمل درآمد
- **Jupyter**: Python اسباق کے لیے انٹرایکٹو نوٹ بکس
- **R Markdown**: R اسباق کے لیے دستاویزات
- **Vue.js 3**: کوئز ایپلیکیشن فریم ورک
- **Flask**: ML ماڈل ڈیپلائمنٹ کے لیے ویب ایپلیکیشن فریم ورک
- **Docsify**: دستاویزات سائٹ جنریٹر
- **GitHub Actions**: CI/CD اور خودکار ترجمے

## سیکیورٹی کے تحفظات

- **کوڈ میں کوئی راز نہیں**: API کیز یا اسناد کو کبھی بھی کمیٹ نہ کریں
- **ڈپینڈنسیز**: npm اور pip پیکجز کو اپ ڈیٹ رکھیں
- **یوزر ان پٹ**: Flask ویب ایپ کی مثالوں میں بنیادی ان پٹ ویلیڈیشن شامل ہے
- **حساس ڈیٹا**: مثال کے ڈیٹا سیٹس عوامی اور غیر حساس ہیں

## مسائل کا حل

### Jupyter نوٹ بکس

- **کرنل کے مسائل**: اگر سیلز رک جائیں تو کرنل کو ری اسٹارٹ کریں: Kernel → Restart
- **امپورٹ کی غلطیاں**: یقینی بنائیں کہ تمام مطلوبہ پیکجز pip کے ساتھ انسٹال ہیں
- **پاتھ کے مسائل**: نوٹ بکس کو ان کی موجودہ ڈائریکٹری سے چلائیں

### کوئز ایپلیکیشن

- **npm install ناکام ہو جاتا ہے**: npm کیش صاف کریں: `npm cache clean --force`
- **پورٹ کے تنازعات**: پورٹ تبدیل کریں: `npm run serve -- --port 8081`
- **بلڈ کی غلطیاں**: `node_modules` کو حذف کریں اور دوبارہ انسٹال کریں: `rm -rf node_modules && npm install`

### R اسباق

- **پیکج نہیں ملا**: انسٹال کریں: `install.packages("package-name")`
- **RMarkdown رینڈرنگ**: یقینی بنائیں کہ rmarkdown پیکج انسٹال ہے
- **کرنل کے مسائل**: Jupyter کے لیے IRkernel انسٹال کرنے کی ضرورت ہو سکتی ہے

## پروجیکٹ کے مخصوص نوٹس

- یہ بنیادی طور پر ایک **تعلیمی نصاب** ہے، پروڈکشن کوڈ نہیں
- توجہ **مشین لرننگ کے تصورات کو سمجھنے** پر ہے عملی مشق کے ذریعے
- کوڈ کی مثالیں **وضاحت کو ترجیح دیتی ہیں** اصلاح کے بجائے
- زیادہ تر اسباق **خود مختار** ہیں اور آزادانہ طور پر مکمل کیے جا سکتے ہیں
- **حل فراہم کیے گئے ہیں** لیکن سیکھنے والوں کو پہلے مشقیں کرنے کی کوشش کرنی چاہیے
- ریپوزٹری **Docsify** استعمال کرتی ہے ویب دستاویزات کے لیے بغیر بلڈ مرحلے کے
- **Sketchnotes** تصورات کے بصری خلاصے فراہم کرتے ہیں
- **کثیر زبان کی حمایت** مواد کو عالمی سطح پر قابل رسائی بناتی ہے

---

**ڈس کلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا خامیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے لیے ہم ذمہ دار نہیں ہیں۔