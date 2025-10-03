<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:14:50+00:00",
  "source_file": "AGENTS.md",
  "language_code": "sw"
}
-->
# AGENTS.md

## Muhtasari wa Mradi

Hii ni **Kujifunza Mashine kwa Anayeanza**, mtaala wa wiki 12 wenye masomo 26 unaofundisha dhana za kimsingi za kujifunza mashine kwa kutumia Python (hasa na Scikit-learn) na R. Hifadhi hii imetengenezwa kama rasilimali ya kujifunza kwa kasi yako mwenyewe ikiwa na miradi ya vitendo, maswali ya majaribio, na mazoezi. Kila somo linachunguza dhana za ML kupitia data halisi kutoka tamaduni na maeneo mbalimbali duniani.

Vipengele muhimu:
- **Maudhui ya Elimu**: Masomo 26 yanayojumuisha utangulizi wa ML, urejeleaji, uainishaji, ugawanyaji, NLP, mfululizo wa muda, na kujifunza kwa kuimarisha
- **Programu ya Maswali**: Programu ya maswali inayotumia Vue.js na tathmini kabla na baada ya somo
- **Msaada wa Lugha Nyingi**: Tafsiri za kiotomatiki kwa lugha zaidi ya 40 kupitia GitHub Actions
- **Msaada wa Lugha Mbili**: Masomo yanapatikana kwa Python (Jupyter notebooks) na R (R Markdown files)
- **Kujifunza kwa Miradi**: Kila mada inajumuisha miradi ya vitendo na mazoezi

## Muundo wa Hifadhi

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

Kila folda ya somo kwa kawaida ina:
- `README.md` - Maudhui makuu ya somo
- `notebook.ipynb` - Jupyter notebook ya Python
- `solution/` - Msimbo wa suluhisho (toleo la Python na R)
- `assignment.md` - Mazoezi ya vitendo
- `images/` - Rasilimali za kuona

## Amri za Kuweka

### Kwa Masomo ya Python

Masomo mengi yanatumia Jupyter notebooks. Weka mahitaji yanayohitajika:

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

### Kwa Masomo ya R

Masomo ya R yapo kwenye folda za `solution/R/` kama faili za `.rmd` au `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Kwa Programu ya Maswali

Programu ya maswali ni programu ya Vue.js iliyoko kwenye folda ya `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Kwa Tovuti ya Nyaraka

Kuendesha nyaraka kwa ndani:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Mtiririko wa Maendeleo

### Kufanya Kazi na Notebooks za Masomo

1. Nenda kwenye folda ya somo (mfano, `2-Regression/1-Tools/`)
2. Fungua Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Fanya kazi kupitia maudhui ya somo na mazoezi
4. Angalia suluhisho kwenye folda ya `solution/` ikiwa inahitajika

### Maendeleo ya Python

- Masomo yanatumia maktaba za kawaida za data za Python
- Jupyter notebooks kwa kujifunza kwa maingiliano
- Msimbo wa suluhisho unapatikana kwenye folda ya `solution/` ya kila somo

### Maendeleo ya R

- Masomo ya R yapo katika muundo wa `.rmd` (R Markdown)
- Suluhisho ziko kwenye folda ndogo za `solution/R/`
- Tumia RStudio au Jupyter na kernel ya R kuendesha notebooks za R

### Maendeleo ya Programu ya Maswali

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

## Maelekezo ya Kupima

### Kupima Programu ya Maswali

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Kumbuka**: Hii ni hifadhi ya mtaala wa elimu. Hakuna majaribio ya kiotomatiki kwa maudhui ya masomo. Uthibitishaji unafanywa kupitia:
- Kukamilisha mazoezi ya somo
- Kuendesha seli za notebook kwa mafanikio
- Kuangalia matokeo dhidi ya suluhisho zinazotarajiwa

## Miongozo ya Mtindo wa Msimbo

### Msimbo wa Python
- Fuata miongozo ya mtindo wa PEP 8
- Tumia majina ya kutofautisha na kueleweka
- Ongeza maelezo kwa operesheni ngumu
- Jupyter notebooks zinapaswa kuwa na seli za markdown zinazofafanua dhana

### JavaScript/Vue.js (Programu ya Maswali)
- Fuata mwongozo wa mtindo wa Vue.js
- Usanidi wa ESLint katika `quiz-app/package.json`
- Endesha `npm run lint` kuangalia na kurekebisha masuala kiotomatiki

### Nyaraka
- Faili za markdown zinapaswa kuwa wazi na zenye muundo mzuri
- Jumuisha mifano ya msimbo katika vizuizi vya msimbo vilivyofungwa
- Tumia viungo vya ndani vya jamaa
- Fuata muundo uliopo wa uundaji

## Ujenzi na Uwekaji

### Uwekaji wa Programu ya Maswali

Programu ya maswali inaweza kuwekwa kwenye Azure Static Web Apps:

1. **Mahitaji ya Awali**:
   - Akaunti ya Azure
   - Hifadhi ya GitHub (tayari imeforkiwa)

2. **Weka kwenye Azure**:
   - Unda rasilimali ya Azure Static Web App
   - Unganisha na hifadhi ya GitHub
   - Weka eneo la programu: `/quiz-app`
   - Weka eneo la matokeo: `dist`
   - Azure huunda kiotomatiki mtiririko wa kazi wa GitHub Actions

3. **Mtiririko wa Kazi wa GitHub Actions**:
   - Faili ya mtiririko wa kazi imeundwa katika `.github/workflows/azure-static-web-apps-*.yml`
   - Huunda na kuweka kiotomatiki unapofanya push kwenye tawi kuu

### PDF ya Nyaraka

Tengeneza PDF kutoka kwa nyaraka:

```bash
npm install
npm run convert
```

## Mtiririko wa Tafsiri

**Muhimu**: Tafsiri zinafanywa kiotomatiki kupitia GitHub Actions kwa kutumia Co-op Translator.

- Tafsiri zinatengenezwa kiotomatiki unapofanya mabadiliko kwenye tawi la `main`
- **USIFANYE tafsiri za mikono** - mfumo unashughulikia hili
- Mtiririko wa kazi umefafanuliwa katika `.github/workflows/co-op-translator.yml`
- Inatumia huduma za Azure AI/OpenAI kwa tafsiri
- Inasaidia lugha zaidi ya 40

## Miongozo ya Kuchangia

### Kwa Wachangiaji wa Maudhui

1. **Fork hifadhi** na unda tawi la kipengele
2. **Fanya mabadiliko kwenye maudhui ya somo** ikiwa unaongeza/kusasisha masomo
3. **Usibadilishe faili zilizotafsiriwa** - zinatengenezwa kiotomatiki
4. **Jaribu msimbo wako** - hakikisha seli zote za notebook zinafanya kazi bila makosa
5. **Hakikisha viungo na picha** vinafanya kazi vizuri
6. **Tuma ombi la kuvuta** na maelezo wazi

### Miongozo ya Ombi la Kuvuta

- **Muundo wa kichwa**: `[Sehemu] Maelezo mafupi ya mabadiliko`
  - Mfano: `[Regression] Rekebisha makosa ya tahajia katika somo la 5`
  - Mfano: `[Quiz-App] Sasisha utegemezi`
- **Kabla ya kutuma**:
  - Hakikisha seli zote za notebook zinafanya kazi bila makosa
  - Endesha `npm run lint` ikiwa unarekebisha quiz-app
  - Thibitisha muundo wa markdown
  - Jaribu mifano yoyote mpya ya msimbo
- **PR lazima ijumuishwe**:
  - Maelezo ya mabadiliko
  - Sababu za mabadiliko
  - Picha za skrini ikiwa kuna mabadiliko ya UI
- **Kanuni za Maadili**: Fuata [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Utahitaji kusaini Mkataba wa Leseni ya Mchangiaji

## Muundo wa Somo

Kila somo linafuata muundo thabiti:

1. **Maswali ya kabla ya somo** - Jaribu maarifa ya msingi
2. **Maudhui ya somo** - Maelekezo yaliyoandikwa na maelezo
3. **Maonyesho ya msimbo** - Mifano ya vitendo katika notebooks
4. **Ukaguzi wa maarifa** - Thibitisha uelewa wakati wa somo
5. **Changamoto** - Tumia dhana kwa kujitegemea
6. **Mazoezi** - Mazoezi ya muda mrefu
7. **Maswali ya baada ya somo** - Pima matokeo ya kujifunza

## Marejeleo ya Amri za Kawaida

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

## Rasilimali za Ziada

- **Mkusanyiko wa Microsoft Learn**: [Moduli za ML kwa Anayeanza](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Programu ya Maswali**: [Maswali mtandaoni](https://ff-quizzes.netlify.app/en/ml/)
- **Bodi ya Majadiliano**: [Majadiliano ya GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Maonyesho ya Video**: [Orodha ya YouTube](https://aka.ms/ml-beginners-videos)

## Teknolojia Muhimu

- **Python**: Lugha kuu kwa masomo ya ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Utekelezaji mbadala kwa kutumia tidyverse, tidymodels, caret
- **Jupyter**: Notebooks za maingiliano kwa masomo ya Python
- **R Markdown**: Nyaraka kwa masomo ya R
- **Vue.js 3**: Mfumo wa programu ya maswali
- **Flask**: Mfumo wa programu ya wavuti kwa uwekaji wa modeli za ML
- **Docsify**: Jenereta ya tovuti ya nyaraka
- **GitHub Actions**: CI/CD na tafsiri za kiotomatiki

## Masuala ya Usalama

- **Hakuna siri katika msimbo**: Kamwe usiweke funguo za API au hati za siri
- **Utegemezi**: Weka vifurushi vya npm na pip vikiwa vimesasishwa
- **Ingizo la mtumiaji**: Mifano ya programu ya wavuti ya Flask inajumuisha uthibitishaji wa msingi wa ingizo
- **Data nyeti**: Seti za data za mfano ni za umma na zisizo nyeti

## Utatuzi wa Matatizo

### Jupyter Notebooks

- **Masuala ya kernel**: Anzisha upya kernel ikiwa seli zinakwama: Kernel → Restart
- **Makosa ya uingizaji**: Hakikisha vifurushi vyote vinavyohitajika vimewekwa kwa pip
- **Masuala ya njia**: Endesha notebooks kutoka kwenye folda inayozihifadhi

### Programu ya Maswali

- **npm install inashindwa**: Futa cache ya npm: `npm cache clean --force`
- **Migongano ya bandari**: Badilisha bandari kwa: `npm run serve -- --port 8081`
- **Makosa ya ujenzi**: Futa `node_modules` na weka tena: `rm -rf node_modules && npm install`

### Masomo ya R

- **Kifurushi hakipatikani**: Weka kwa: `install.packages("package-name")`
- **Utoaji wa RMarkdown**: Hakikisha kifurushi cha rmarkdown kimewekwa
- **Masuala ya kernel**: Huenda ukahitaji kusakinisha IRkernel kwa Jupyter

## Vidokezo Maalum vya Mradi

- Hii ni hasa **mtaala wa kujifunza**, si msimbo wa uzalishaji
- Lengo ni **kuelewa dhana za ML** kupitia mazoezi ya vitendo
- Mifano ya msimbo inazingatia **uwazi badala ya ufanisi**
- Masomo mengi ni **ya kujitegemea** na yanaweza kukamilishwa bila msaada
- **Suluhisho zinapatikana** lakini wanafunzi wanapaswa kujaribu mazoezi kwanza
- Hifadhi inatumia **Docsify** kwa nyaraka za wavuti bila hatua ya ujenzi
- **Sketchnotes** zinatoa muhtasari wa dhana kwa njia ya kuona
- **Msaada wa lugha nyingi** hufanya maudhui kufikika kimataifa

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya tafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuchukuliwa kama chanzo cha mamlaka. Kwa taarifa muhimu, inashauriwa kutumia huduma ya tafsiri ya kibinadamu ya kitaalamu. Hatutawajibika kwa maelewano mabaya au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.