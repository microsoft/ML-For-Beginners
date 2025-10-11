<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-11T11:09:42+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ta"
}
-->
# AGENTS.md

## திட்டத்தின் மேற்பார்வை

இது **துவக்கத்திற்கான இயந்திரக் கற்றல்**, Python (முக்கியமாக Scikit-learn) மற்றும் R பயன்படுத்தி இயந்திரக் கற்றல் கருத்துக்களை உள்ளடக்கிய 12 வாரங்கள், 26 பாடங்கள் கொண்ட விரிவான பாடத்திட்டம். இந்த களஞ்சியம் சுயமாக கற்றல் செய்யும் வளமாக வடிவமைக்கப்பட்டுள்ளது, இதில் செயல்முறை திட்டங்கள், வினாடி வினா மற்றும் பணிகள் உள்ளன. ஒவ்வொரு பாடமும் உலகம் முழுவதிலுமுள்ள பல்வேறு கலாச்சாரங்கள் மற்றும் பிராந்தியங்களின் உண்மையான தரவுகளைப் பயன்படுத்தி ML கருத்துக்களை ஆராய்கிறது.

முக்கிய கூறுகள்:
- **கல்வி உள்ளடக்கம்**: ML அறிமுகம், பின்வாங்கல், வகைப்படுத்தல், குழுமம், NLP, நேர வரிசை மற்றும் வலியுறுத்தல் கற்றல் ஆகியவற்றை உள்ளடக்கிய 26 பாடங்கள்
- **வினாடி வினா பயன்பாடு**: Vue.js அடிப்படையிலான வினாடி வினா பயன்பாடு, பாடத்திற்கு முன் மற்றும் பின் மதிப்பீடுகளுடன்
- **பல மொழி ஆதரவு**: GitHub Actions மூலம் 40+ மொழிகளுக்கு தானியங்கி மொழிபெயர்ப்பு
- **இரட்டை மொழி ஆதரவு**: Python (Jupyter notebooks) மற்றும் R (R Markdown கோப்புகள்) இரண்டிலும் பாடங்கள் கிடைக்கின்றன
- **திட்ட அடிப்படையிலான கற்றல்**: ஒவ்வொரு தலைப்பும் நடைமுறை திட்டங்கள் மற்றும் பணிகளை உள்ளடக்கியது

## களஞ்சிய அமைப்பு

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

ஒவ்வொரு பாடக் கோப்புறையும் பொதுவாக கீழ்கண்டவற்றை உள்ளடக்கியது:
- `README.md` - முக்கிய பாட உள்ளடக்கம்
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - தீர்வு குறியீடு (Python மற்றும் R பதிப்புகள்)
- `assignment.md` - பயிற்சி பயிற்சிகள்
- `images/` - காட்சித் வளங்கள்

## அமைப்பு கட்டளைகள்

### Python பாடங்களுக்கு

பெரும்பாலான பாடங்கள் Jupyter notebooks பயன்படுத்துகின்றன. தேவையான சார்புகளை நிறுவவும்:

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

### R பாடங்களுக்கு

R பாடங்கள் `solution/R/` கோப்புறைகளில் `.rmd` அல்லது `.ipynb` கோப்புகளாக உள்ளன:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### வினாடி வினா பயன்பாட்டிற்கு

வினாடி வினா பயன்பாடு `quiz-app/` கோப்புறையில் அமைந்துள்ள Vue.js பயன்பாடாகும்:

```bash
cd quiz-app
npm install
```

### ஆவணத்தளத்திற்கு

உள்ளூர் ஆவணத்தை இயக்க:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## மேம்பாட்டு பணியியல்

### பாடக் குறிப்புகளுடன் வேலை செய்வது

1. பாடக் கோப்புறைக்கு செல்லவும் (எ.கா., `2-Regression/1-Tools/`)
2. Jupyter notebook-ஐ திறக்கவும்:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. பாட உள்ளடக்கம் மற்றும் பயிற்சிகளைச் செய்யவும்
4. தேவைப்பட்டால் `solution/` கோப்புறையில் தீர்வுகளைச் சரிபார்க்கவும்

### Python மேம்பாடு

- பாடங்கள் நிலையான Python தரவியல் அறிவியல் நூலகங்களைப் பயன்படுத்துகின்றன
- Python பாடங்களுக்கு இடையூறு கற்றலுக்கான Jupyter notebooks
- ஒவ்வொரு பாடத்தின் `solution/` கோப்புறையில் தீர்வு குறியீடு கிடைக்கிறது

### R மேம்பாடு

- R பாடங்கள் `.rmd` வடிவத்தில் (R Markdown)
- தீர்வுகள் `solution/R/` துணைக் கோப்புறைகளில் உள்ளன
- RStudio அல்லது Jupyter உடன் R kernel பயன்படுத்தி R notebooks இயக்கவும்

### வினாடி வினா பயன்பாட்டு மேம்பாடு

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

## சோதனை வழிமுறைகள்

### வினாடி வினா பயன்பாட்டு சோதனை

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**குறிப்பு**: இது முதன்மையாக கல்வி பாடத்திட்ட களஞ்சியம். பாட உள்ளடக்கத்திற்கு தானியங்கி சோதனைகள் இல்லை. சரிபார்ப்பு கீழ்கண்டவைகளின் மூலம் செய்யப்படுகிறது:
- பாட பயிற்சிகளை முடிக்கவும்
- notebook cells வெற்றிகரமாக இயக்கவும்
- தீர்வுகளில் எதிர்பார்க்கப்பட்ட முடிவுகளுடன் வெளியீட்டைச் சரிபார்க்கவும்

## குறியீட்டு பாணி வழிகாட்டுதல்கள்

### Python குறியீடு
- PEP 8 பாணி வழிகாட்டுதல்களைப் பின்பற்றவும்
- தெளிவான, விளக்கமான மாறி பெயர்களைப் பயன்படுத்தவும்
- சிக்கலான செயல்பாடுகளுக்கு கருத்துகளைச் சேர்க்கவும்
- Jupyter notebooks-ல் கருத்துக்களை விளக்கும் markdown cells இருக்க வேண்டும்

### JavaScript/Vue.js (வினாடி வினா பயன்பாடு)
- Vue.js பாணி வழிகாட்டுதல்களைப் பின்பற்றவும்
- `quiz-app/package.json` இல் ESLint கட்டமைப்பு
- `npm run lint` இயக்கி சிக்கல்களைச் சரி செய்யவும்

### ஆவணங்கள்
- Markdown கோப்புகள் தெளிவாகவும் நன்றாக அமைக்கப்பட்டவையாகவும் இருக்க வேண்டும்
- fenced code blocks-ல் குறியீட்டு உதாரணங்களைச் சேர்க்கவும்
- உள் குறிப்புகளுக்கு சார்பு இணைப்புகளைப் பயன்படுத்தவும்
- உள்ளமைப்பு ஒழுங்குகளைப் பின்பற்றவும்

## கட்டமைப்பு மற்றும் வெளியீடு

### வினாடி வினா பயன்பாட்டின் வெளியீடு

வினாடி வினா பயன்பாட்டை Azure Static Web Apps-க்கு வெளியிடலாம்:

1. **முன் தேவைகள்**:
   - Azure கணக்கு
   - GitHub களஞ்சியம் (ஏற்கனவே fork செய்யப்பட்டது)

2. **Azure-க்கு வெளியிடவும்**:
   - Azure Static Web App வளத்தை உருவாக்கவும்
   - GitHub களஞ்சியத்தை இணைக்கவும்
   - பயன்பாட்டு இடம் அமைக்கவும்: `/quiz-app`
   - வெளியீட்டு இடம் அமைக்கவும்: `dist`
   - Azure தானாக GitHub Actions பணியியல் உருவாக்குகிறது

3. **GitHub Actions பணியியல்**:
   - `.github/workflows/azure-static-web-apps-*.yml` இல் பணியியல் கோப்பு உருவாக்கப்பட்டது
   - முக்கிய கிளைக்கு push செய்யும் போது தானாக கட்டமைக்கவும் மற்றும் வெளியிடவும்

### ஆவண PDF

ஆவணத்திலிருந்து PDF உருவாக்கவும்:

```bash
npm install
npm run convert
```

## மொழிபெயர்ப்பு பணியியல்

**முக்கியம்**: மொழிபெயர்ப்புகள் GitHub Actions மூலம் Co-op Translator பயன்படுத்தி தானியங்கி செய்யப்படுகின்றன.

- `main` கிளைக்கு மாற்றங்கள் push செய்யும் போது மொழிபெயர்ப்புகள் தானாக உருவாக்கப்படும்
- **உள்ளடக்கத்தை கையேடு மொழிபெயர்க்க வேண்டாம்** - அமைப்பு இதைச் செய்கிறது
- `.github/workflows/co-op-translator.yml` இல் பணியியல் வரையறுக்கப்பட்டுள்ளது
- மொழிபெயர்ப்புக்கு Azure AI/OpenAI சேவைகளைப் பயன்படுத்துகிறது
- 40+ மொழிகளை ஆதரிக்கிறது

## பங்களிப்பு வழிகாட்டுதல்கள்

### உள்ளடக்க பங்களிப்பாளர்களுக்கு

1. **களஞ்சியத்தை fork செய்யவும்** மற்றும் ஒரு அம்ச கிளையை உருவாக்கவும்
2. **பாட உள்ளடக்கத்தில் மாற்றங்களைச் செய்யவும்** (பாடங்களைச் சேர்க்க/புதுப்பிக்க)
3. **மொழிபெயர்க்கப்பட்ட கோப்புகளை மாற்ற வேண்டாம்** - அவை தானியங்கி உருவாக்கப்பட்டவை
4. **உங்கள் குறியீட்டைச் சோதிக்கவும்** - அனைத்து notebook cells வெற்றிகரமாக இயங்க வேண்டும்
5. **இணைப்புகள் மற்றும் படங்கள் சரியாக வேலை செய்கின்றனவா என்பதைச் சரிபார்க்கவும்**
6. **தெளிவான விளக்கத்துடன் pull request சமர்ப்பிக்கவும்**

### Pull Request வழிகாட்டுதல்கள்

- **தலைப்பு வடிவம்**: `[பகுதி] மாற்றங்களின் சுருக்கமான விளக்கம்`
  - உதாரணம்: `[Regression] பாடம் 5 இல் தப்பை சரி செய்யவும்`
  - உதாரணம்: `[Quiz-App] சார்புகளைப் புதுப்பிக்கவும்`
- **சமர்ப்பிக்கும் முன்**:
  - அனைத்து notebook cells பிழையின்றி செயல்படுகின்றனவா என்பதை உறுதிப்படுத்தவும்
  - quiz-app-ஐ மாற்றினால் `npm run lint` இயக்கவும்
  - Markdown வடிவமைப்பைச் சரிபார்க்கவும்
  - புதிய குறியீட்டு உதாரணங்களைச் சோதிக்கவும்
- **PR-ல் உள்ளடக்க வேண்டும்**:
  - மாற்றங்களின் விளக்கம்
  - மாற்றங்களின் காரணம்
  - UI மாற்றங்கள் இருந்தால் திரைக்காட்சிகள்
- **நடத்தை விதிமுறைகள்**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) பின்பற்றவும்
- **CLA**: Contributor License Agreement கையொப்பமிட வேண்டும்

## பாட அமைப்பு

ஒவ்வொரு பாடமும் ஒரே மாதிரியான முறைபாட்டைப் பின்பற்றுகிறது:

1. **பாடத்திற்கு முன் வினாடி வினா** - அடிப்படை அறிவைச் சோதிக்கவும்
2. **பாட உள்ளடக்கம்** - எழுதப்பட்ட வழிமுறைகள் மற்றும் விளக்கங்கள்
3. **குறியீட்டு விளக்கங்கள்** - notebooks-ல் நடைமுறை உதாரணங்கள்
4. **அறிவு சரிபார்ப்புகள்** - புரிதலை உறுதிப்படுத்தவும்
5. **சவால்** - கருத்துக்களை சுயமாகப் பயன்படுத்தவும்
6. **பணி** - விரிவான பயிற்சி
7. **பாடத்திற்கு பின் வினாடி வினா** - கற்றல் முடிவுகளை மதிப்பீடு செய்யவும்

## பொதுவான கட்டளைகள் குறிப்பு

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

## கூடுதல் வளங்கள்

- **Microsoft Learn தொகுப்பு**: [துவக்கத்திற்கான ML modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **வினாடி வினா பயன்பாடு**: [ஆன்லைன் வினாடி வினாக்கள்](https://ff-quizzes.netlify.app/en/ml/)
- **விவாத வாரியம்**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **வீடியோ வழிகாட்டுதல்கள்**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## முக்கிய தொழில்நுட்பங்கள்

- **Python**: ML பாடங்களுக்கு முதன்மை மொழி (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret பயன்படுத்தி மாற்று செயல்பாடு
- **Jupyter**: Python பாடங்களுக்கு இடையூறு notebooks
- **R Markdown**: R பாடங்களுக்கான ஆவணங்கள்
- **Vue.js 3**: வினாடி வினா பயன்பாட்டு கட்டமைப்பு
- **Flask**: ML மாதிரி வெளியீட்டிற்கான வலை பயன்பாட்டு கட்டமைப்பு
- **Docsify**: ஆவணத்தள உருவாக்கி
- **GitHub Actions**: CI/CD மற்றும் தானியங்கி மொழிபெயர்ப்புகள்

## பாதுகாப்பு கருத்துக்கள்

- **குறியீட்டில் ரகசியங்கள் இல்லை**: API விசைகள் அல்லது அங்கீகாரங்களை commit செய்ய வேண்டாம்
- **சார்புகள்**: npm மற்றும் pip தொகுப்புகளை புதுப்பித்து வைத்திருக்கவும்
- **பயனர் உள்ளீடு**: Flask வலை பயன்பாட்டு உதாரணங்களில் அடிப்படை உள்ளீடு சரிபார்ப்பு உள்ளது
- **முக்கிய தரவுகள்**: உதாரண தரவுத்தொகுப்புகள் பொது மற்றும் முக்கியமற்றவை

## சிக்கல் தீர்வு

### Jupyter notebooks

- **Kernel சிக்கல்கள்**: cells தடைபட்டால் kernel-ஐ மீண்டும் தொடங்கவும்: Kernel → Restart
- **Import பிழைகள்**: தேவையான அனைத்து தொகுப்புகளும் pip மூலம் நிறுவப்பட்டுள்ளனவா என்பதை உறுதிப்படுத்தவும்
- **பாதை சிக்கல்கள்**: notebooks-ஐ அவற்றின் உள்ளடக்க கோப்புறையிலிருந்து இயக்கவும்

### வினாடி வினா பயன்பாடு

- **npm install தோல்வி**: npm cache-ஐ அழிக்கவும்: `npm cache clean --force`
- **Port conflicts**: port-ஐ மாற்றவும்: `npm run serve -- --port 8081`
- **Build பிழைகள்**: `node_modules`-ஐ அழித்து மீண்டும் நிறுவவும்: `rm -rf node_modules && npm install`

### R பாடங்கள்

- **தொகுப்பு கிடைக்கவில்லை**: `install.packages("package-name")` மூலம் நிறுவவும்
- **RMarkdown உருவாக்கம்**: rmarkdown தொகுப்பு நிறுவப்பட்டுள்ளதா என்பதை உறுதிப்படுத்தவும்
- **Kernel சிக்கல்கள்**: Jupyter-க்கு IRkernel நிறுவ வேண்டியிருக்கலாம்

## திட்டத்திற்கு குறிப்பிட்ட குறிப்புகள்

- இது முதன்மையாக **கல்வி பாடத்திட்டம்**, உற்பத்தி குறியீடு அல்ல
- **கைமுறையாகப் பயிற்சி செய்வதன் மூலம் ML கருத்துக்களைப் புரிந்து கொள்ள** கவனம் செலுத்துகிறது
- குறியீட்டு உதாரணங்கள் **தெளிவுக்கு முன்னுரிமை** அளிக்கின்றன, ஆப்டிமைசேஷன் அல்ல
- பெரும்பாலான பாடங்கள் **சுயமாக முடிக்க** முடியும்
- **தீர்வுகள் வழங்கப்பட்டுள்ளன**, ஆனால் கற்றல் பயிற்சிகளை முதலில் முயற்சிக்க வேண்டும்
- களஞ்சியம் **Docsify** பயன்படுத்துகிறது, build படி இல்லாமல் வலை ஆவணத்திற்காக
- **Sketchnotes** கருத்துக்களின் காட்சித் சுருக்கங்களை வழங்குகிறது
- **பல மொழி ஆதரவு** உள்ளடக்கத்தை உலகளாவிய அளவில் அணுகக்கூடியதாக மாற்றுகிறது

---

**குறிப்பு**:  
இந்த ஆவணம் [Co-op Translator](https://github.com/Azure/co-op-translator) என்ற AI மொழிபெயர்ப்பு சேவையைப் பயன்படுத்தி மொழிபெயர்க்கப்பட்டுள்ளது. நாங்கள் துல்லியத்திற்காக முயற்சிக்கிறோம், ஆனால் தானியக்க மொழிபெயர்ப்புகளில் பிழைகள் அல்லது தவறான தகவல்கள் இருக்கக்கூடும் என்பதை தயவுசெய்து கவனத்தில் கொள்ளுங்கள். அதன் தாய்மொழியில் உள்ள மூல ஆவணம் அதிகாரப்பூர்வ ஆதாரமாக கருதப்பட வேண்டும். முக்கியமான தகவல்களுக்கு, தொழில்முறை மனித மொழிபெயர்ப்பு பரிந்துரைக்கப்படுகிறது. இந்த மொழிபெயர்ப்பைப் பயன்படுத்துவதால் ஏற்படும் எந்த தவறான புரிதல்கள் அல்லது தவறான விளக்கங்களுக்கு நாங்கள் பொறுப்பல்ல.