<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:02:59+00:00",
  "source_file": "AGENTS.md",
  "language_code": "hi"
}
-->
# AGENTS.md

## परियोजना का अवलोकन

यह **Machine Learning for Beginners** है, एक व्यापक 12-सप्ताह, 26-पाठ का पाठ्यक्रम जो Python (मुख्य रूप से Scikit-learn) और R का उपयोग करके क्लासिक मशीन लर्निंग अवधारणाओं को कवर करता है। यह रिपॉजिटरी स्व-गति से सीखने के संसाधन के रूप में डिज़ाइन की गई है, जिसमें प्रोजेक्ट, क्विज़ और असाइनमेंट शामिल हैं। प्रत्येक पाठ विभिन्न संस्कृतियों और क्षेत्रों के वास्तविक डेटा के माध्यम से ML अवधारणाओं की खोज करता है।

मुख्य घटक:
- **शैक्षिक सामग्री**: 26 पाठ जो ML का परिचय, रिग्रेशन, क्लासिफिकेशन, क्लस्टरिंग, NLP, टाइम सीरीज़, और रिइंफोर्समेंट लर्निंग को कवर करते हैं
- **क्विज़ एप्लिकेशन**: Vue.js आधारित क्विज़ ऐप जिसमें पाठ से पहले और बाद के आकलन होते हैं
- **बहु-भाषा समर्थन**: GitHub Actions के माध्यम से 40+ भाषाओं में स्वचालित अनुवाद
- **दोहरी भाषा समर्थन**: पाठ Python (Jupyter नोटबुक) और R (R Markdown फाइलें) दोनों में उपलब्ध हैं
- **प्रोजेक्ट-आधारित सीखना**: प्रत्येक विषय में व्यावहारिक प्रोजेक्ट और असाइनमेंट शामिल हैं

## रिपॉजिटरी संरचना

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

प्रत्येक पाठ फ़ोल्डर में आमतौर पर निम्नलिखित होते हैं:
- `README.md` - मुख्य पाठ सामग्री
- `notebook.ipynb` - Python Jupyter नोटबुक
- `solution/` - समाधान कोड (Python और R संस्करण)
- `assignment.md` - अभ्यास अभ्यास
- `images/` - दृश्य संसाधन

## सेटअप कमांड्स

### Python पाठों के लिए

अधिकांश पाठ Jupyter नोटबुक का उपयोग करते हैं। आवश्यक डिपेंडेंसी इंस्टॉल करें:

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

### R पाठों के लिए

R पाठ `solution/R/` फ़ोल्डरों में `.rmd` या `.ipynb` फाइलों के रूप में हैं:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### क्विज़ एप्लिकेशन के लिए

क्विज़ ऐप `quiz-app/` डायरेक्टरी में स्थित एक Vue.js एप्लिकेशन है:

```bash
cd quiz-app
npm install
```

### डॉक्यूमेंटेशन साइट के लिए

डॉक्यूमेंटेशन को लोकल चलाने के लिए:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## विकास वर्कफ़्लो

### पाठ नोटबुक के साथ काम करना

1. पाठ डायरेक्टरी पर जाएं (जैसे, `2-Regression/1-Tools/`)
2. Jupyter नोटबुक खोलें:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. पाठ सामग्री और अभ्यासों के माध्यम से काम करें
4. यदि आवश्यक हो तो `solution/` फ़ोल्डर में समाधान जांचें

### Python विकास

- पाठ मानक Python डेटा साइंस लाइब्रेरी का उपयोग करते हैं
- इंटरैक्टिव सीखने के लिए Jupyter नोटबुक
- प्रत्येक पाठ के `solution/` फ़ोल्डर में समाधान कोड उपलब्ध है

### R विकास

- R पाठ `.rmd` प्रारूप (R Markdown) में हैं
- समाधान `solution/R/` सबडायरेक्टरी में स्थित हैं
- RStudio या Jupyter के साथ R कर्नेल का उपयोग करके R नोटबुक चलाएं

### क्विज़ एप्लिकेशन विकास

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

## परीक्षण निर्देश

### क्विज़ एप्लिकेशन परीक्षण

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**नोट**: यह मुख्य रूप से एक शैक्षिक पाठ्यक्रम रिपॉजिटरी है। पाठ सामग्री के लिए कोई स्वचालित परीक्षण नहीं हैं। सत्यापन निम्नलिखित के माध्यम से किया जाता है:
- पाठ अभ्यास पूरा करना
- नोटबुक सेल्स को सफलतापूर्वक चलाना
- समाधान में अपेक्षित परिणामों के खिलाफ आउटपुट की जांच करना

## कोड शैली दिशानिर्देश

### Python कोड
- PEP 8 शैली दिशानिर्देशों का पालन करें
- स्पष्ट, वर्णनात्मक वेरिएबल नामों का उपयोग करें
- जटिल ऑपरेशनों के लिए टिप्पणियां शामिल करें
- Jupyter नोटबुक में अवधारणाओं को समझाने वाले मार्कडाउन सेल्स होने चाहिए

### JavaScript/Vue.js (क्विज़ ऐप)
- Vue.js शैली गाइड का पालन करता है
- `quiz-app/package.json` में ESLint कॉन्फ़िगरेशन
- मुद्दों की जांच और ऑटो-फिक्स के लिए `npm run lint` चलाएं

### डॉक्यूमेंटेशन
- Markdown फाइलें स्पष्ट और अच्छी तरह से संरचित होनी चाहिए
- fenced कोड ब्लॉक्स में कोड उदाहरण शामिल करें
- आंतरिक संदर्भों के लिए सापेक्ष लिंक का उपयोग करें
- मौजूदा स्वरूपण परंपराओं का पालन करें

## निर्माण और परिनियोजन

### क्विज़ एप्लिकेशन परिनियोजन

क्विज़ ऐप को Azure Static Web Apps पर परिनियोजित किया जा सकता है:

1. **पूर्वापेक्षाएँ**:
   - Azure खाता
   - GitHub रिपॉजिटरी (पहले से फोर्क किया हुआ)

2. **Azure पर परिनियोजित करें**:
   - Azure Static Web App संसाधन बनाएं
   - GitHub रिपॉजिटरी से कनेक्ट करें
   - ऐप स्थान सेट करें: `/quiz-app`
   - आउटपुट स्थान सेट करें: `dist`
   - Azure स्वचालित रूप से GitHub Actions वर्कफ़्लो बनाता है

3. **GitHub Actions वर्कफ़्लो**:
   - वर्कफ़्लो फाइल `.github/workflows/azure-static-web-apps-*.yml` में बनाई जाती है
   - मुख्य शाखा पर पुश करने पर स्वचालित रूप से निर्माण और परिनियोजन करता है

### डॉक्यूमेंटेशन PDF

डॉक्यूमेंटेशन से PDF उत्पन्न करें:

```bash
npm install
npm run convert
```

## अनुवाद वर्कफ़्लो

**महत्वपूर्ण**: अनुवाद GitHub Actions के माध्यम से Co-op Translator का उपयोग करके स्वचालित हैं।

- जब परिवर्तन `main` शाखा पर पुश किए जाते हैं तो अनुवाद स्वचालित रूप से उत्पन्न होते हैं
- **सामग्री को मैन्युअल रूप से अनुवादित न करें** - सिस्टम इसे संभालता है
- वर्कफ़्लो `.github/workflows/co-op-translator.yml` में परिभाषित है
- अनुवाद के लिए Azure AI/OpenAI सेवाओं का उपयोग करता है
- 40+ भाषाओं का समर्थन करता है

## योगदान दिशानिर्देश

### सामग्री योगदानकर्ताओं के लिए

1. **रिपॉजिटरी फोर्क करें** और एक फीचर शाखा बनाएं
2. **पाठ सामग्री में परिवर्तन करें** यदि पाठ जोड़ रहे हैं/अपडेट कर रहे हैं
3. **अनुवादित फाइलों को संशोधित न करें** - वे स्वचालित रूप से उत्पन्न होती हैं
4. **अपने कोड का परीक्षण करें** - सुनिश्चित करें कि सभी नोटबुक सेल्स सफलतापूर्वक चलते हैं
5. **लिंक और छवियों की जांच करें** कि वे सही ढंग से काम कर रहे हैं
6. **स्पष्ट विवरण के साथ एक पुल अनुरोध सबमिट करें**

### पुल अनुरोध दिशानिर्देश

- **शीर्षक प्रारूप**: `[Section] परिवर्तनों का संक्षिप्त विवरण`
  - उदाहरण: `[Regression] पाठ 5 में टाइपो ठीक करें`
  - उदाहरण: `[Quiz-App] डिपेंडेंसी अपडेट करें`
- **सबमिट करने से पहले**:
  - सुनिश्चित करें कि सभी नोटबुक सेल्स त्रुटियों के बिना निष्पादित होते हैं
  - यदि क्विज़-ऐप संशोधित कर रहे हैं तो `npm run lint` चलाएं
  - Markdown स्वरूपण सत्यापित करें
  - किसी भी नए कोड उदाहरण का परीक्षण करें
- **PR में शामिल होना चाहिए**:
  - परिवर्तनों का विवरण
  - परिवर्तनों का कारण
  - UI परिवर्तनों के लिए स्क्रीनशॉट
- **आचार संहिता**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) का पालन करें
- **CLA**: आपको Contributor License Agreement पर हस्ताक्षर करना होगा

## पाठ संरचना

प्रत्येक पाठ एक सुसंगत पैटर्न का अनुसरण करता है:

1. **प्री-लेक्चर क्विज़** - प्रारंभिक ज्ञान का परीक्षण करें
2. **पाठ सामग्री** - लिखित निर्देश और व्याख्याएँ
3. **कोड प्रदर्शन** - नोटबुक में व्यावहारिक उदाहरण
4. **ज्ञान जांच** - समझ को सत्यापित करें
5. **चुनौती** - अवधारणाओं को स्वतंत्र रूप से लागू करें
6. **असाइनमेंट** - विस्तारित अभ्यास
7. **पोस्ट-लेक्चर क्विज़** - सीखने के परिणामों का आकलन करें

## सामान्य कमांड संदर्भ

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

## अतिरिक्त संसाधन

- **Microsoft Learn संग्रह**: [ML for Beginners मॉड्यूल](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **क्विज़ ऐप**: [ऑनलाइन क्विज़](https://ff-quizzes.netlify.app/en/ml/)
- **चर्चा बोर्ड**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **वीडियो वॉकथ्रू**: [YouTube प्लेलिस्ट](https://aka.ms/ml-beginners-videos)

## प्रमुख तकनीकें

- **Python**: ML पाठों के लिए प्राथमिक भाषा (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret का उपयोग करके वैकल्पिक कार्यान्वयन
- **Jupyter**: Python पाठों के लिए इंटरैक्टिव नोटबुक
- **R Markdown**: R पाठों के लिए दस्तावेज़
- **Vue.js 3**: क्विज़ एप्लिकेशन फ्रेमवर्क
- **Flask**: ML मॉडल परिनियोजन के लिए वेब एप्लिकेशन फ्रेमवर्क
- **Docsify**: डॉक्यूमेंटेशन साइट जनरेटर
- **GitHub Actions**: CI/CD और स्वचालित अनुवाद

## सुरक्षा विचार

- **कोड में कोई गुप्त जानकारी नहीं**: API कुंजी या क्रेडेंशियल्स को कभी भी कमिट न करें
- **डिपेंडेंसी**: npm और pip पैकेज को अपडेट रखें
- **उपयोगकर्ता इनपुट**: Flask वेब ऐप उदाहरणों में बुनियादी इनपुट सत्यापन शामिल है
- **संवेदनशील डेटा**: उदाहरण डेटा सेट सार्वजनिक और गैर-संवेदनशील हैं

## समस्या निवारण

### Jupyter नोटबुक

- **कर्नेल समस्याएँ**: यदि सेल्स अटकते हैं तो कर्नेल को पुनः प्रारंभ करें: Kernel → Restart
- **इंपोर्ट त्रुटियाँ**: सुनिश्चित करें कि सभी आवश्यक पैकेज pip के साथ इंस्टॉल किए गए हैं
- **पथ समस्याएँ**: नोटबुक को उनकी समाहित डायरेक्टरी से चलाएं

### क्विज़ एप्लिकेशन

- **npm install विफल**: npm कैश साफ करें: `npm cache clean --force`
- **पोर्ट संघर्ष**: पोर्ट बदलें: `npm run serve -- --port 8081`
- **बिल्ड त्रुटियाँ**: `node_modules` हटाएं और पुनः इंस्टॉल करें: `rm -rf node_modules && npm install`

### R पाठ

- **पैकेज नहीं मिला**: इंस्टॉल करें: `install.packages("package-name")`
- **RMarkdown रेंडरिंग**: सुनिश्चित करें कि rmarkdown पैकेज इंस्टॉल है
- **कर्नेल समस्याएँ**: Jupyter के लिए IRkernel इंस्टॉल करना पड़ सकता है

## परियोजना-विशिष्ट नोट्स

- यह मुख्य रूप से एक **शिक्षण पाठ्यक्रम** है, उत्पादन कोड नहीं
- हाथों-हाथ अभ्यास के माध्यम से **ML अवधारणाओं को समझने** पर ध्यान केंद्रित है
- कोड उदाहरण **स्पष्टता को अनुकूलन पर प्राथमिकता देते हैं**
- अधिकांश पाठ **स्वतंत्र** हैं और स्वतंत्र रूप से पूरे किए जा सकते हैं
- **समाधान प्रदान किए गए हैं**, लेकिन शिक्षार्थियों को पहले अभ्यास करने का प्रयास करना चाहिए
- रिपॉजिटरी **Docsify** का उपयोग करता है वेब डॉक्यूमेंटेशन के लिए बिना बिल्ड स्टेप के
- **Sketchnotes** अवधारणाओं का दृश्य सारांश प्रदान करते हैं
- **बहु-भाषा समर्थन** सामग्री को वैश्विक रूप से सुलभ बनाता है

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।