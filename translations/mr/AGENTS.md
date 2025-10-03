<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:04:07+00:00",
  "source_file": "AGENTS.md",
  "language_code": "mr"
}
-->
# AGENTS.md

## प्रकल्पाचा आढावा

**Machine Learning for Beginners** हा 12 आठवड्यांचा, 26 धड्यांचा अभ्यासक्रम आहे जो Python (मुख्यतः Scikit-learn) आणि R वापरून पारंपरिक मशीन लर्निंग संकल्पना समजावून सांगतो. हे रिपॉझिटरी स्व-गतीने शिकण्यासाठी संसाधन म्हणून डिझाइन केले आहे ज्यामध्ये प्रकल्प, क्विझ आणि असाइनमेंट्स समाविष्ट आहेत. प्रत्येक धडा विविध संस्कृती आणि प्रदेशांमधील वास्तविक डेटा वापरून ML संकल्पना स्पष्ट करतो.

महत्त्वाचे घटक:
- **शैक्षणिक सामग्री**: ML ची ओळख, रिग्रेशन, क्लासिफिकेशन, क्लस्टरिंग, NLP, टाइम सिरीज आणि रिइन्फोर्समेंट लर्निंग यावर आधारित 26 धडे
- **क्विझ अॅप्लिकेशन**: Vue.js आधारित क्विझ अॅप, प्री- आणि पोस्ट-लेसन मूल्यांकनांसह
- **बहुभाषिक समर्थन**: GitHub Actions द्वारे 40+ भाषांमध्ये स्वयंचलित भाषांतर
- **दुहेरी भाषा समर्थन**: Python (Jupyter नोटबुक्स) आणि R (R Markdown फाइल्स) मध्ये धडे उपलब्ध
- **प्रकल्प आधारित शिक्षण**: प्रत्येक विषयात व्यावहारिक प्रकल्प आणि असाइनमेंट्स समाविष्ट आहेत

## रिपॉझिटरी संरचना

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

प्रत्येक धड्याच्या फोल्डरमध्ये सामान्यतः खालील गोष्टी असतात:
- `README.md` - मुख्य धड्याची सामग्री
- `notebook.ipynb` - Python Jupyter नोटबुक
- `solution/` - सोल्यूशन कोड (Python आणि R आवृत्त्या)
- `assignment.md` - सरावासाठी व्यायाम
- `images/` - दृश्य संसाधने

## सेटअप कमांड्स

### Python धड्यांसाठी

बहुतेक धडे Jupyter नोटबुक्स वापरतात. आवश्यक dependencies इंस्टॉल करा:

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

### R धड्यांसाठी

R धडे `solution/R/` फोल्डर्समध्ये `.rmd` किंवा `.ipynb` फाइल्स म्हणून आहेत:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### क्विझ अॅप्लिकेशनसाठी

क्विझ अॅप `quiz-app/` डिरेक्टरीमध्ये Vue.js अॅप्लिकेशन आहे:

```bash
cd quiz-app
npm install
```

### डॉक्युमेंटेशन साइटसाठी

डॉक्युमेंटेशन स्थानिक पातळीवर चालवण्यासाठी:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## विकास कार्यप्रवाह

### धड्याच्या नोटबुक्ससह काम करणे

1. धड्याच्या डिरेक्टरीमध्ये जा (उदा., `2-Regression/1-Tools/`)
2. Jupyter नोटबुक उघडा:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. धड्याची सामग्री आणि व्यायाम पूर्ण करा
4. आवश्यक असल्यास `solution/` फोल्डरमधील सोल्यूशन्स तपासा

### Python विकास

- धडे मानक Python डेटा सायन्स लायब्ररी वापरतात
- परस्पर शिकण्यासाठी Jupyter नोटबुक्स
- प्रत्येक धड्याच्या `solution/` फोल्डरमध्ये सोल्यूशन कोड उपलब्ध

### R विकास

- R धडे `.rmd` स्वरूपात (R Markdown) आहेत
- सोल्यूशन्स `solution/R/` उपडिरेक्टरीमध्ये आहेत
- RStudio किंवा Jupyter सह R कर्नल वापरून R नोटबुक्स चालवा

### क्विझ अॅप्लिकेशन विकास

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

## चाचणी सूचना

### क्विझ अॅप्लिकेशन चाचणी

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**टीप**: हे मुख्यतः शैक्षणिक अभ्यासक्रम रिपॉझिटरी आहे. धड्याच्या सामग्रीसाठी स्वयंचलित चाचण्या नाहीत. सत्यापन खालीलप्रमाणे केले जाते:
- धड्याचे व्यायाम पूर्ण करणे
- नोटबुक सेल्स यशस्वीरित्या चालवणे
- सोल्यूशन्समधील अपेक्षित परिणामांशी आउटपुट तपासणे

## कोड शैली मार्गदर्शक तत्त्वे

### Python कोड
- PEP 8 शैली मार्गदर्शकांचे पालन करा
- स्पष्ट, वर्णनात्मक व्हेरिएबल नावे वापरा
- जटिल ऑपरेशन्ससाठी टिप्पण्या समाविष्ट करा
- Jupyter नोटबुक्समध्ये संकल्पना स्पष्ट करणारे Markdown सेल्स असावेत

### JavaScript/Vue.js (क्विझ अॅप)
- Vue.js शैली मार्गदर्शकांचे पालन करते
- `quiz-app/package.json` मध्ये ESLint कॉन्फिगरेशन
- समस्या तपासण्यासाठी आणि स्वयंचलितपणे दुरुस्त करण्यासाठी `npm run lint` चालवा

### डॉक्युमेंटेशन
- Markdown फाइल्स स्पष्ट आणि चांगल्या प्रकारे संरचित असाव्यात
- fenced code blocks मध्ये कोड उदाहरणे समाविष्ट करा
- अंतर्गत संदर्भांसाठी सापेक्ष दुवे वापरा
- विद्यमान स्वरूपन परंपरांचे पालन करा

## बिल्ड आणि डिप्लॉयमेंट

### क्विझ अॅप्लिकेशन डिप्लॉयमेंट

क्विझ अॅप Azure Static Web Apps वर डिप्लॉय केला जाऊ शकतो:

1. **पूर्वतयारी**:
   - Azure खाते
   - GitHub रिपॉझिटरी (आधीच fork केलेले)

2. **Azure वर डिप्लॉय करा**:
   - Azure Static Web App संसाधन तयार करा
   - GitHub रिपॉझिटरीशी कनेक्ट करा
   - अॅप स्थान सेट करा: `/quiz-app`
   - आउटपुट स्थान सेट करा: `dist`
   - Azure स्वयंचलितपणे GitHub Actions कार्यप्रवाह तयार करते

3. **GitHub Actions कार्यप्रवाह**:
   - `.github/workflows/azure-static-web-apps-*.yml` येथे कार्यप्रवाह फाइल तयार केली जाते
   - मुख्य शाखेवर पुश केल्यावर स्वयंचलितपणे बिल्ड आणि डिप्लॉय करते

### डॉक्युमेंटेशन PDF

डॉक्युमेंटेशनमधून PDF तयार करा:

```bash
npm install
npm run convert
```

## भाषांतर कार्यप्रवाह

**महत्त्वाचे**: GitHub Actions वापरून Co-op Translator द्वारे भाषांतर स्वयंचलित आहे.

- मुख्य शाखेत बदल पुश केल्यावर भाषांतर स्वयंचलितपणे तयार होते
- **सामग्री मॅन्युअली भाषांतर करू नका** - प्रणाली हे हाताळते
- `.github/workflows/co-op-translator.yml` मध्ये कार्यप्रवाह परिभाषित
- भाषांतरासाठी Azure AI/OpenAI सेवा वापरते
- 40+ भाषांचे समर्थन

## योगदान मार्गदर्शक तत्त्वे

### सामग्री योगदानकर्त्यांसाठी

1. **रिपॉझिटरी fork करा** आणि फीचर शाखा तयार करा
2. **धड्याच्या सामग्रीत बदल करा** जर धडे जोडत/अपडेट करत असाल
3. **भाषांतरित फाइल्स बदलू नका** - त्या स्वयंचलितपणे तयार होतात
4. **तुमचा कोड चाचणी करा** - सर्व नोटबुक सेल्स यशस्वीरित्या चालवले जातील याची खात्री करा
5. **दुवे आणि प्रतिमा** योग्य प्रकारे कार्य करत असल्याची खात्री करा
6. **स्पष्ट वर्णनासह pull request सबमिट करा**

### Pull Request मार्गदर्शक तत्त्वे

- **शीर्षक स्वरूप**: `[Section] बदलांचे संक्षिप्त वर्णन`
  - उदाहरण: `[Regression] धडा 5 मधील टायपो दुरुस्त करा`
  - उदाहरण: `[Quiz-App] dependencies अपडेट करा`
- **सबमिट करण्यापूर्वी**:
  - सर्व नोटबुक सेल्स त्रुटीशिवाय चालवले जातील याची खात्री करा
  - quiz-app मध्ये बदल करत असल्यास `npm run lint` चालवा
  - Markdown स्वरूपन सत्यापित करा
  - नवीन कोड उदाहरणे चाचणी करा
- **PR मध्ये समाविष्ट असले पाहिजे**:
  - बदलांचे वर्णन
  - बदलांचे कारण
  - UI बदल असल्यास स्क्रीनशॉट्स
- **आचारसंहिता**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) चे पालन करा
- **CLA**: तुम्हाला Contributor License Agreement वर स्वाक्षरी करावी लागेल

## धड्याची संरचना

प्रत्येक धडा सुसंगत पद्धतीचे अनुसरण करतो:

1. **प्री-लेक्चर क्विझ** - प्राथमिक ज्ञान तपासा
2. **धड्याची सामग्री** - लेखी सूचना आणि स्पष्टीकरण
3. **कोड डेमोन्स्ट्रेशन** - नोटबुक्समधील व्यावहारिक उदाहरणे
4. **ज्ञान तपासणी** - समज सत्यापित करा
5. **चॅलेंज** - स्वतंत्रपणे संकल्पना लागू करा
6. **असाइनमेंट** - विस्तारित सराव
7. **पोस्ट-लेक्चर क्विझ** - शिकण्याचे परिणाम मोजा

## सामान्य कमांड्स संदर्भ

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

## अतिरिक्त संसाधने

- **Microsoft Learn संग्रह**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **क्विझ अॅप**: [ऑनलाइन क्विझेस](https://ff-quizzes.netlify.app/en/ml/)
- **चर्चा बोर्ड**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **व्हिडिओ वॉकथ्रू**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## प्रमुख तंत्रज्ञान

- **Python**: ML धड्यांसाठी प्राथमिक भाषा (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret वापरून पर्यायी अंमलबजावणी
- **Jupyter**: Python धड्यांसाठी परस्पर नोटबुक्स
- **R Markdown**: R धड्यांसाठी दस्तऐवज
- **Vue.js 3**: क्विझ अॅप्लिकेशन फ्रेमवर्क
- **Flask**: ML मॉडेल डिप्लॉयमेंटसाठी वेब अॅप्लिकेशन फ्रेमवर्क
- **Docsify**: डॉक्युमेंटेशन साइट जनरेटर
- **GitHub Actions**: CI/CD आणि स्वयंचलित भाषांतर

## सुरक्षा विचार

- **कोडमध्ये गुपिते नाहीत**: API कीज किंवा क्रेडेन्शियल्स कधीही commit करू नका
- **Dependencies**: npm आणि pip पॅकेजेस अद्ययावत ठेवा
- **वापरकर्ता इनपुट**: Flask वेब अॅप उदाहरणांमध्ये मूलभूत इनपुट व्हॅलिडेशन समाविष्ट आहे
- **संवेदनशील डेटा**: उदाहरण डेटासेट्स सार्वजनिक आणि असंवेदनशील आहेत

## समस्या निराकरण

### Jupyter नोटबुक्स

- **कर्नल समस्या**: सेल्स अडकले असल्यास कर्नल रीस्टार्ट करा: Kernel → Restart
- **इम्पोर्ट त्रुटी**: pip सह सर्व आवश्यक पॅकेजेस इंस्टॉल असल्याची खात्री करा
- **पथ समस्या**: नोटबुक्स त्यांच्या समाविष्ट डिरेक्टरीमधून चालवा

### क्विझ अॅप्लिकेशन

- **npm install अयशस्वी**: npm कॅश साफ करा: `npm cache clean --force`
- **पोर्ट संघर्ष**: पोर्ट बदला: `npm run serve -- --port 8081`
- **बिल्ड त्रुटी**: `node_modules` हटवा आणि पुन्हा इंस्टॉल करा: `rm -rf node_modules && npm install`

### R धडे

- **पॅकेज सापडले नाही**: `install.packages("package-name")` सह इंस्टॉल करा
- **RMarkdown रेंडरिंग**: rmarkdown पॅकेज इंस्टॉल असल्याची खात्री करा
- **कर्नल समस्या**: Jupyter साठी IRkernel इंस्टॉल करणे आवश्यक असू शकते

## प्रकल्प-विशिष्ट टीपा

- हे मुख्यतः **शिकण्याचा अभ्यासक्रम** आहे, उत्पादन कोड नाही
- **हाताळण्याच्या सरावाद्वारे ML संकल्पना समजणे** यावर लक्ष केंद्रित
- कोड उदाहरणे **स्पष्टतेला प्राधान्य देतात, ऑप्टिमायझेशनला नाही**
- बहुतेक धडे **स्वतंत्र आहेत** आणि स्वतंत्रपणे पूर्ण केले जाऊ शकतात
- **सोल्यूशन्स उपलब्ध** आहेत परंतु शिकणाऱ्यांनी प्रथम व्यायाम करण्याचा प्रयत्न करावा
- रिपॉझिटरी **Docsify** वापरते वेब डॉक्युमेंटेशनसाठी बिल्ड स्टेपशिवाय
- **Sketchnotes** संकल्पनांचे दृश्य सारांश प्रदान करतात
- **बहुभाषिक समर्थन** सामग्री जागतिक स्तरावर प्रवेशयोग्य बनवते

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी कृपया लक्षात ठेवा की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून उद्भवलेल्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.