<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-12-19T12:38:53+00:00",
  "source_file": "AGENTS.md",
  "language_code": "te"
}
-->
# AGENTS.md

## Project Overview

ఇది **Machine Learning for Beginners**, పాఠ్యాంశాల 26తో కూడిన 12 వారాల సమగ్ర పాఠ్యక్రమం, ఇది Python (ప్రధానంగా Scikit-learn తో) మరియు R ఉపయోగించి క్లాసిక్ మెషీన్ లెర్నింగ్ కాన్సెప్ట్‌లను కవర్ చేస్తుంది. ఈ రిపోజిటరీ స్వీయ-గతిలో నేర్చుకునే వనరుగా రూపొందించబడింది, ఇందులో ప్రాక్టికల్ ప్రాజెక్టులు, క్విజ్‌లు మరియు అసైన్‌మెంట్‌లు ఉన్నాయి. ప్రతి పాఠం ప్రపంచవ్యాప్తంగా వివిధ సంస్కృతులు మరియు ప్రాంతాల నుండి వాస్తవ డేటా ద్వారా ML కాన్సెప్ట్‌లను అన్వేషిస్తుంది.

ప్రధాన భాగాలు:
- **విద్యా విషయాలు**: ML పరిచయం, రిగ్రెషన్, క్లాసిఫికేషన్, క్లస్టరింగ్, NLP, టైమ్ సిరీస్, మరియు రీఇన్ఫోర్స్‌మెంట్ లెర్నింగ్‌ను కవర్ చేసే 26 పాఠాలు
- **క్విజ్ అప్లికేషన్**: Vue.js ఆధారిత క్విజ్ యాప్, పాఠం ముందు మరియు తర్వాత అంచనాలతో
- **బహుభాషా మద్దతు**: GitHub Actions ద్వారా 40+ భాషలకు ఆటోమేటెడ్ అనువాదాలు
- **రెండు భాషల మద్దతు**: పాఠాలు Python (Jupyter నోట్‌బుక్స్) మరియు R (R Markdown ఫైళ్లలో) అందుబాటులో ఉన్నాయి
- **ప్రాజెక్ట్ ఆధారిత నేర్చుకోవడం**: ప్రతి అంశం ప్రాక్టికల్ ప్రాజెక్టులు మరియు అసైన్‌మెంట్‌లను కలిగి ఉంటుంది

## Repository Structure

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

ప్రతి పాఠం ఫోల్డర్ సాధారణంగా కలిగి ఉంటుంది:
- `README.md` - ప్రధాన పాఠ్యాంశం
- `notebook.ipynb` - Python Jupyter నోట్‌బుక్
- `solution/` - సొల్యూషన్ కోడ్ (Python మరియు R వెర్షన్లు)
- `assignment.md` - ప్రాక్టీస్ వ్యాయామాలు
- `images/` - విజువల్ వనరులు

## Setup Commands

### For Python Lessons

అధిక భాగం పాఠాలు Jupyter నోట్‌బుక్స్ ఉపయోగిస్తాయి. అవసరమైన డిపెండెన్సీలను ఇన్‌స్టాల్ చేయండి:

```bash
# ఇప్పటికే ఇన్‌స్టాల్ చేయకపోతే Python 3.8+ ఇన్‌స్టాల్ చేయండి
python --version

# Jupyter ఇన్‌స్టాల్ చేయండి
pip install jupyter

# సాధారణ ML లైబ్రరీలను ఇన్‌స్టాల్ చేయండి
pip install scikit-learn pandas numpy matplotlib seaborn

# నిర్దిష్ట పాఠాల కోసం, పాఠం-స్పెసిఫిక్ అవసరాలను తనిఖీ చేయండి
# ఉదాహరణ: వెబ్ యాప్ పాఠం
pip install flask
```

### For R Lessons

R పాఠాలు `solution/R/` ఫోల్డర్లలో `.rmd` లేదా `.ipynb` ఫైళ్లుగా ఉంటాయి:

```bash
# R మరియు అవసరమైన ప్యాకేజీలను ఇన్‌స్టాల్ చేయండి
# R కన్సోల్‌లో:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### For Quiz Application

క్విజ్ యాప్ `quiz-app/` డైరెక్టరీలో ఉన్న Vue.js అప్లికేషన్:

```bash
cd quiz-app
npm install
```

### For Documentation Site

డాక్యుమెంటేషన్ స్థానికంగా నడపడానికి:

```bash
# డాక్సిఫైని ఇన్‌స్టాల్ చేయండి
npm install -g docsify-cli

# రిపోజిటరీ రూట్ నుండి సర్వ్ చేయండి
docsify serve

# http://localhost:3000 వద్ద యాక్సెస్ చేయండి
```

## Development Workflow

### Working with Lesson Notebooks

1. పాఠం డైరెక్టరీకి వెళ్లండి (ఉదా: `2-Regression/1-Tools/`)
2. Jupyter నోట్‌బుక్ తెరవండి:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. పాఠ్యాంశం మరియు వ్యాయామాలపై పని చేయండి
4. అవసరమైతే `solution/` ఫోల్డర్‌లో సొల్యూషన్లను తనిఖీ చేయండి

### Python Development

- పాఠాలు ప్రామాణిక Python డేటా సైన్స్ లైబ్రరీలను ఉపయోగిస్తాయి
- ఇంటరాక్టివ్ నేర్చుకోవడానికి Jupyter నోట్‌బుక్స్
- ప్రతి పాఠం `solution/` ఫోల్డర్‌లో సొల్యూషన్ కోడ్ అందుబాటులో ఉంటుంది

### R Development

- R పాఠాలు `.rmd` ఫార్మాట్ (R Markdown)లో ఉంటాయి
- సొల్యూషన్లు `solution/R/` ఉపడైరెక్టరీలలో ఉంటాయి
- R నోట్‌బుక్స్ నడపడానికి RStudio లేదా R కర్నెల్‌తో Jupyter ఉపయోగించండి

### Quiz Application Development

```bash
cd quiz-app

# అభివృద్ధి సర్వర్ ప్రారంభించండి
npm run serve
# http://localhost:8080 వద్ద యాక్సెస్ చేయండి

# ఉత్పత్తి కోసం నిర్మించండి
npm run build

# ఫైళ్లను లింట్ చేసి సరిచేయండి
npm run lint
```

## Testing Instructions

### Quiz Application Testing

```bash
cd quiz-app

# కోడ్‌ను లింట్ చేయండి
npm run lint

# ఎటువంటి లోపాలు లేవని నిర్ధారించడానికి నిర్మించండి
npm run build
```

**గమనిక**: ఇది ప్రధానంగా విద్యా పాఠ్యక్రమం రిపోజిటరీ. పాఠ్యాంశం కోసం ఆటోమేటెడ్ టెస్టులు లేవు. ధృవీకరణ ఈ విధంగా జరుగుతుంది:
- పాఠం వ్యాయామాలు పూర్తి చేయడం
- నోట్‌బుక్ సెల్స్ విజయవంతంగా నడపడం
- సొల్యూషన్లలో అంచనా ఫలితాలతో అవుట్‌పుట్ తనిఖీ చేయడం

## Code Style Guidelines

### Python Code
- PEP 8 స్టైల్ మార్గదర్శకాలను అనుసరించండి
- స్పష్టమైన, వివరణాత్మక వేరియబుల్ పేర్లను ఉపయోగించండి
- క్లిష్టమైన ఆపరేషన్లకు వ్యాఖ్యలు చేర్చండి
- Jupyter నోట్‌బుక్స్‌లో కాన్సెప్ట్‌లను వివరించే మార్క్డౌన్ సెల్స్ ఉండాలి

### JavaScript/Vue.js (Quiz App)
- Vue.js స్టైల్ గైడ్‌ను అనుసరిస్తుంది
- `quiz-app/package.json`లో ESLint కాన్ఫిగరేషన్
- సమస్యలను తనిఖీ చేయడానికి మరియు ఆటో-ఫిక్స్ చేయడానికి `npm run lint` నడపండి

### Documentation
- మార్క్డౌన్ ఫైళ్లు స్పష్టంగా మరియు బాగా నిర్మించబడాలి
- కోడ్ ఉదాహరణలను fenced కోడ్ బ్లాక్స్‌లో చేర్చండి
- అంతర్గత సూచనలకు సంబంధిత లింకులను ఉపయోగించండి
- ఉన్న ఫార్మాటింగ్ సంప్రదాయాలను అనుసరించండి

## Build and Deployment

### Quiz Application Deployment

క్విజ్ యాప్‌ను Azure Static Web Apps కు డిప్లాయ్ చేయవచ్చు:

1. **అవసరాలు**:
   - Azure ఖాతా
   - GitHub రిపోజిటరీ (ఇప్పటికే ఫోర్క్ చేయబడింది)

2. **Azure కు డిప్లాయ్ చేయండి**:
   - Azure Static Web App వనరును సృష్టించండి
   - GitHub రిపోజిటరీకి కనెక్ట్ చేయండి
   - యాప్ లొకేషన్: `/quiz-app` గా సెట్ చేయండి
   - అవుట్‌పుట్ లొకేషన్: `dist` గా సెట్ చేయండి
   - Azure ఆటోమేటిక్‌గా GitHub Actions వర్క్‌ఫ్లో సృష్టిస్తుంది

3. **GitHub Actions Workflow**:
   - `.github/workflows/azure-static-web-apps-*.yml` వద్ద వర్క్‌ఫ్లో ఫైల్ సృష్టించబడుతుంది
   - ప్రధాన బ్రాంచ్‌కు పుష్ చేసినప్పుడు ఆటోమేటిక్‌గా బిల్డ్ చేసి డిప్లాయ్ చేస్తుంది

### Documentation PDF

డాక్యుమెంటేషన్ నుండి PDF రూపొందించండి:

```bash
npm install
npm run convert
```

## Translation Workflow

**ముఖ్యమైనది**: అనువాదాలు GitHub Actions ద్వారా Co-op Translator ఉపయోగించి ఆటోమేటెడ్‌గా జరుగుతాయి.

- మార్పులు `main` బ్రాంచ్‌కు పుష్ చేసినప్పుడు అనువాదాలు ఆటోమేటిక్‌గా ఉత్పత్తి అవుతాయి
- **కంటెంట్‌ను మానవీయంగా అనువదించవద్దు** - సిస్టమ్ దీనిని నిర్వహిస్తుంది
- వర్క్‌ఫ్లో `.github/workflows/co-op-translator.yml`లో నిర్వచించబడింది
- అనువాదానికి Azure AI/OpenAI సేవలను ఉపయోగిస్తుంది
- 40+ భాషలకు మద్దతు ఇస్తుంది

## Contributing Guidelines

### For Content Contributors

1. **రిపోజిటరీని ఫోర్క్ చేసి** ఫీచర్ బ్రాంచ్ సృష్టించండి
2. **పాఠ్యాంశం మార్చండి** లేదా కొత్త పాఠాలు జోడించండి
3. **అనువదించిన ఫైళ్లను మార్చవద్దు** - అవి ఆటోమేటెడ్‌గా ఉత్పత్తి అవుతాయి
4. **మీ కోడ్‌ను పరీక్షించండి** - అన్ని నోట్‌బుక్ సెల్స్ విజయవంతంగా నడవాలి
5. **లింకులు మరియు చిత్రాలు సరిచూసుకోండి**
6. **స్పష్టమైన వివరణతో పుల్ రిక్వెస్ట్ సమర్పించండి**

### Pull Request Guidelines

- **శీర్షిక ఫార్మాట్**: `[Section] మార్పుల సంక్షిప్త వివరణ`
  - ఉదా: `[Regression] పాఠం 5లో టైపో సరిచేయండి`
  - ఉదా: `[Quiz-App] డిపెండెన్సీలను నవీకరించండి`
- **సమర్పించే ముందు**:
  - అన్ని నోట్‌బుక్ సెల్స్ ఎర్రర్ల లేకుండా నడవాలి
  - quiz-app మార్చినట్లయితే `npm run lint` నడపండి
  - మార్క్డౌన్ ఫార్మాటింగ్ తనిఖీ చేయండి
  - కొత్త కోడ్ ఉదాహరణలను పరీక్షించండి
- **PRలో ఉండవలసినవి**:
  - మార్పుల వివరణ
  - మార్పుల కారణం
  - UI మార్పులుంటే స్క్రీన్‌షాట్లు
- **Code of Conduct**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) అనుసరించండి
- **CLA**: Contributor License Agreement సంతకం చేయాలి

## Lesson Structure

ప్రతి పాఠం ఒక సుస్పష్టమైన నమూనాను అనుసరిస్తుంది:

1. **పాఠం ముందు క్విజ్** - ప్రాథమిక జ్ఞానాన్ని పరీక్షించండి
2. **పాఠ్యాంశం** - వ్రాత సూచనలు మరియు వివరణలు
3. **కోడ్ డెమోస్** - నోట్‌బుక్స్‌లో ప్రాక్టికల్ ఉదాహరణలు
4. **జ్ఞాన తనిఖీలు** - అర్థం చేసుకున్నదాన్ని నిర్ధారించండి
5. **చాలెంజ్** - స్వతంత్రంగా కాన్సెప్ట్‌లను వర్తించండి
6. **అసైన్‌మెంట్** - విస్తృత ప్రాక్టీస్
7. **పాఠం తర్వాత క్విజ్** - నేర్చుకున్న ఫలితాలను అంచనా వేయండి

## Common Commands Reference

```bash
# Python/Jupyter
jupyter notebook                    # Jupyter సర్వర్ ప్రారంభించండి
jupyter notebook notebook.ipynb     # నిర్దిష్ట నోట్‌బుక్ తెరవండి
pip install -r requirements.txt     # ఆధారాలు ఇన్‌స్టాల్ చేయండి (అక్కడ అందుబాటులో ఉంటే)

# క్విజ్ యాప్
cd quiz-app
npm install                        # ఆధారాలు ఇన్‌స్టాల్ చేయండి
npm run serve                      # అభివృద్ధి సర్వర్
npm run build                      # ఉత్పత్తి బిల్డ్
npm run lint                       # లింట్ చేసి సరిచేయండి

# డాక్యుమెంటేషన్
docsify serve                      # డాక్యుమెంటేషన్‌ను స్థానికంగా సర్వ్ చేయండి
npm run convert                    # PDF రూపొందించండి

# Git వర్క్‌ఫ్లో
git checkout -b feature/my-change  # ఫీచర్ బ్రాంచ్ సృష్టించండి
git add .                         # మార్పులను స్టేజ్ చేయండి
git commit -m "Description"       # మార్పులను కమిట్ చేయండి
git push origin feature/my-change # రిమోట్‌కు పుష్ చేయండి
```

## Additional Resources

- **Microsoft Learn Collection**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz App**: [Online quizzes](https://ff-quizzes.netlify.app/en/ml/)
- **Discussion Board**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video Walkthroughs**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Key Technologies

- **Python**: ML పాఠాల కోసం ప్రాథమిక భాష (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret ఉపయోగించి ప్రత్యామ్నాయ అమలు
- **Jupyter**: Python పాఠాల కోసం ఇంటరాక్టివ్ నోట్‌బుక్స్
- **R Markdown**: R పాఠాల డాక్యుమెంట్లు
- **Vue.js 3**: క్విజ్ అప్లికేషన్ ఫ్రేమ్‌వర్క్
- **Flask**: ML మోడల్ డిప్లాయ్‌మెంట్ కోసం వెబ్ అప్లికేషన్ ఫ్రేమ్‌వర్క్
- **Docsify**: డాక్యుమెంటేషన్ సైట్ జనరేటర్
- **GitHub Actions**: CI/CD మరియు ఆటోమేటెడ్ అనువాదాలు

## Security Considerations

- **కోడ్‌లో రహస్యాలు లేవు**: API కీలు లేదా క్రెడెన్షియల్స్ ఎప్పుడూ కమిట్ చేయవద్దు
- **డిపెండెన్సీలు**: npm మరియు pip ప్యాకేజీలను నవీకరించండి
- **వినియోగదారు ఇన్‌పుట్**: Flask వెబ్ యాప్ ఉదాహరణలు ప్రాథమిక ఇన్‌పుట్ ధృవీకరణను కలిగి ఉంటాయి
- **సున్నితమైన డేటా**: ఉదాహరణ డేటాసెట్‌లు పబ్లిక్ మరియు సున్నితమైనవి కావు

## Troubleshooting

### Jupyter Notebooks

- **కర్నెల్ సమస్యలు**: సెల్స్ హ్యాంగ్ అయితే కర్నెల్ రీస్టార్ట్ చేయండి: Kernel → Restart
- **ఇంపోర్ట్ లోపాలు**: అవసరమైన అన్ని ప్యాకేజీలు pip తో ఇన్‌స్టాల్ చేయండి
- **పాత్ సమస్యలు**: నోట్‌బుక్స్‌ను వాటి ఉన్న డైరెక్టరీ నుండి నడపండి

### Quiz Application

- **npm install విఫలమైతే**: npm క్యాష్ క్లియర్ చేయండి: `npm cache clean --force`
- **పోర్ట్ సంకర్షణలు**: పోర్ట్ మార్చండి: `npm run serve -- --port 8081`
- **బిల్డ్ లోపాలు**: `node_modules` తొలగించి మళ్లీ ఇన్‌స్టాల్ చేయండి: `rm -rf node_modules && npm install`

### R Lessons

- **ప్యాకేజీ కనుగొనబడకపోతే**: ఇన్‌స్టాల్ చేయండి: `install.packages("package-name")`
- **RMarkdown రెండరింగ్**: rmarkdown ప్యాకేజీ ఇన్‌స్టాల్ ఉందని నిర్ధారించండి
- **కర్నెల్ సమస్యలు**: Jupyter కోసం IRkernel ఇన్‌స్టాల్ చేయవలసి ఉండవచ్చు

## Project-Specific Notes

- ఇది ప్రధానంగా **నెర్చుకునే పాఠ్యక్రమం**, ప్రొడక్షన్ కోడ్ కాదు
- ప్రాధాన్యం **ML కాన్సెప్ట్‌లను అర్థం చేసుకోవడంలో** ఉంది, ప్రాక్టికల్ ద్వారా
- కోడ్ ఉదాహరణలు **స్పష్టతపై ఎక్కువ దృష్టి** పెట్టాయి, ఆప్టిమైజేషన్ కంటే
- ఎక్కువ భాగం పాఠాలు **స్వతంత్రంగా పూర్తి చేయగలవు**
- **సొల్యూషన్లు అందుబాటులో ఉన్నాయి**, కానీ నేర్చుకునేవారు ముందుగా వ్యాయామాలు ప్రయత్నించాలి
- రిపోజిటరీ Docsify ఉపయోగించి వెబ్ డాక్యుమెంటేషన్ అందిస్తుంది, బిల్డ్ స్టెప్ అవసరం లేదు
- **స్కెచ్‌నోట్లు** కాన్సెప్ట్‌ల విజువల్ సారాంశాలను అందిస్తాయి
- **బహుభాషా మద్దతు** కంటెంట్‌ను ప్రపంచవ్యాప్తంగా అందుబాటులో ఉంచుతుంది

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**అస్పష్టత**:  
ఈ పత్రాన్ని AI అనువాద సేవ [Co-op Translator](https://github.com/Azure/co-op-translator) ఉపయోగించి అనువదించబడింది. మేము ఖచ్చితత్వానికి ప్రయత్నించినప్పటికీ, ఆటోమేటెడ్ అనువాదాల్లో పొరపాట్లు లేదా తప్పిదాలు ఉండవచ్చు. అసలు పత్రం దాని స్వదేశీ భాషలో ఉన్నది అధికారిక మూలంగా పరిగణించాలి. ముఖ్యమైన సమాచారానికి, ప్రొఫెషనల్ మానవ అనువాదం సిఫార్సు చేయబడుతుంది. ఈ అనువాదం వాడకంలో ఏర్పడిన ఏవైనా అపార్థాలు లేదా తప్పుదారుల కోసం మేము బాధ్యత వహించము.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->