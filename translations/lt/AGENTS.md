<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:20:51+00:00",
  "source_file": "AGENTS.md",
  "language_code": "lt"
}
-->
# AGENTS.md

## Projekto apžvalga

Tai yra **Mašininis mokymasis pradedantiesiems**, išsami 12 savaičių, 26 pamokų mokymo programa, apimanti klasikines mašininio mokymosi sąvokas naudojant Python (daugiausia su Scikit-learn) ir R. Šis saugyklos projektas skirtas savarankiškam mokymuisi su praktiniais projektais, testais ir užduotimis. Kiekviena pamoka nagrinėja ML sąvokas naudojant realius duomenis iš įvairių kultūrų ir regionų visame pasaulyje.

Pagrindiniai komponentai:
- **Mokomoji medžiaga**: 26 pamokos, apimančios ML įvadą, regresiją, klasifikaciją, klasterizavimą, NLP, laiko eiles ir pastiprinamąjį mokymąsi
- **Testų programa**: Vue.js pagrindu sukurta testų programa su prieš ir po pamokų vertinimais
- **Daugiakalbė parama**: Automatiniai vertimai į daugiau nei 40 kalbų naudojant GitHub Actions
- **Dviguba kalbų parama**: Pamokos pateikiamos tiek Python (Jupyter užrašinėse), tiek R (R Markdown failuose)
- **Projektinis mokymasis**: Kiekviena tema apima praktinius projektus ir užduotis

## Saugyklos struktūra

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

Kiekviename pamokos aplanke paprastai yra:
- `README.md` - Pagrindinė pamokos medžiaga
- `notebook.ipynb` - Python Jupyter užrašinė
- `solution/` - Sprendimų kodas (Python ir R versijos)
- `assignment.md` - Praktinės užduotys
- `images/` - Vizualiniai ištekliai

## Nustatymo komandos

### Python pamokoms

Dauguma pamokų naudoja Jupyter užrašines. Įdiekite reikalingas priklausomybes:

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

### R pamokoms

R pamokos yra `solution/R/` aplankuose kaip `.rmd` arba `.ipynb` failai:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Testų programai

Testų programa yra Vue.js programa, esanti `quiz-app/` kataloge:

```bash
cd quiz-app
npm install
```

### Dokumentacijos svetainei

Norėdami paleisti dokumentaciją vietoje:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Kūrimo darbo eiga

### Darbas su pamokų užrašinėmis

1. Eikite į pamokos katalogą (pvz., `2-Regression/1-Tools/`)
2. Atidarykite Jupyter užrašinę:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Peržiūrėkite pamokos turinį ir užduotis
4. Jei reikia, patikrinkite sprendimus `solution/` aplanke

### Python kūrimas

- Pamokose naudojamos standartinės Python duomenų mokslo bibliotekos
- Jupyter užrašinės interaktyviam mokymuisi
- Sprendimų kodas pateikiamas kiekvienos pamokos `solution/` aplanke

### R kūrimas

- R pamokos pateikiamos `.rmd` formatu (R Markdown)
- Sprendimai yra `solution/R/` poaplankiuose
- Naudokite RStudio arba Jupyter su R branduoliu, kad paleistumėte R užrašines

### Testų programos kūrimas

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

## Testavimo instrukcijos

### Testų programos testavimas

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Pastaba**: Tai pirmiausia yra mokomoji mokymo programa. Pamokų turiniui nėra automatinių testų. Tikrinimas atliekamas:
- Atlikus pamokų užduotis
- Sėkmingai paleidus užrašinių langelius
- Patikrinus rezultatų atitikimą sprendimams

## Kodo stiliaus gairės

### Python kodas
- Laikykitės PEP 8 stiliaus gairių
- Naudokite aiškius, aprašomuosius kintamųjų pavadinimus
- Komentuokite sudėtingas operacijas
- Jupyter užrašinės turėtų turėti Markdown langelius, paaiškinančius sąvokas

### JavaScript/Vue.js (Testų programa)
- Laikykitės Vue.js stiliaus gairių
- ESLint konfigūracija `quiz-app/package.json`
- Paleiskite `npm run lint`, kad patikrintumėte ir automatiškai ištaisytumėte problemas

### Dokumentacija
- Markdown failai turėtų būti aiškūs ir gerai struktūrizuoti
- Įtraukite kodo pavyzdžius į aptvertus kodo blokus
- Naudokite santykines nuorodas vidiniams šaltiniams
- Laikykitės esamų formatavimo konvencijų

## Kūrimas ir diegimas

### Testų programos diegimas

Testų programa gali būti diegiama į Azure Static Web Apps:

1. **Reikalavimai**:
   - Azure paskyra
   - GitHub saugykla (jau nukopijuota)

2. **Diegimas į Azure**:
   - Sukurkite Azure Static Web App išteklių
   - Prijunkite prie GitHub saugyklos
   - Nustatykite programos vietą: `/quiz-app`
   - Nustatykite išvesties vietą: `dist`
   - Azure automatiškai sukuria GitHub Actions darbo eigą

3. **GitHub Actions darbo eiga**:
   - Darbo eigos failas sukuriamas `.github/workflows/azure-static-web-apps-*.yml`
   - Automatiškai kuria ir diegia, kai stumiama į pagrindinę šaką

### Dokumentacijos PDF

Generuokite PDF iš dokumentacijos:

```bash
npm install
npm run convert
```

## Vertimo darbo eiga

**Svarbu**: Vertimai atliekami automatiškai per GitHub Actions naudojant Co-op Translator.

- Vertimai generuojami automatiškai, kai pakeitimai įkeliami į `main` šaką
- **NEVERTINKITE turinio rankiniu būdu** - sistema tai atlieka
- Darbo eiga apibrėžta `.github/workflows/co-op-translator.yml`
- Naudojamos Azure AI/OpenAI paslaugos vertimui
- Palaikoma daugiau nei 40 kalbų

## Prisidėjimo gairės

### Turinį prisidedantiems

1. **Nukopijuokite saugyklą** ir sukurkite funkcijų šaką
2. **Atlikite pakeitimus pamokų turinyje**, jei pridedate/atnaujinate pamokas
3. **Nemodifikuokite išverstų failų** - jie generuojami automatiškai
4. **Išbandykite savo kodą** - įsitikinkite, kad visi užrašinių langeliai veikia be klaidų
5. **Patikrinkite nuorodas ir vaizdus** - įsitikinkite, kad jie veikia teisingai
6. **Pateikite traukimo užklausą** su aiškiu aprašymu

### Traukimo užklausų gairės

- **Pavadinimo formatas**: `[Skyrius] Trumpas pakeitimų aprašymas`
  - Pavyzdys: `[Regresija] Ištaisyta klaida 5 pamokoje`
  - Pavyzdys: `[Testų programa] Atnaujintos priklausomybės`
- **Prieš pateikdami**:
  - Įsitikinkite, kad visi užrašinių langeliai vykdomi be klaidų
  - Paleiskite `npm run lint`, jei modifikuojate testų programą
  - Patikrinkite Markdown formatavimą
  - Išbandykite naujus kodo pavyzdžius
- **PR turi apimti**:
  - Pakeitimų aprašymą
  - Pakeitimų priežastį
  - Ekrano nuotraukas, jei yra UI pakeitimų
- **Elgesio kodeksas**: Laikykitės [Microsoft atvirojo kodo elgesio kodekso](CODE_OF_CONDUCT.md)
- **CLA**: Turėsite pasirašyti Contributor License Agreement

## Pamokų struktūra

Kiekviena pamoka laikosi nuoseklios struktūros:

1. **Prieš paskaitą testas** - Patikrinkite pradinį žinių lygį
2. **Pamokos turinys** - Rašytinės instrukcijos ir paaiškinimai
3. **Kodo demonstracijos** - Praktiniai pavyzdžiai užrašinėse
4. **Žinių patikrinimai** - Patikrinkite supratimą pamokos metu
5. **Iššūkis** - Savarankiškai pritaikykite sąvokas
6. **Užduotis** - Išplėstinė praktika
7. **Po paskaitos testas** - Įvertinkite mokymosi rezultatus

## Dažniausiai naudojamų komandų nuoroda

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

## Papildomi ištekliai

- **Microsoft Learn kolekcija**: [ML pradedantiesiems moduliai](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Testų programa**: [Testai internetu](https://ff-quizzes.netlify.app/en/ml/)
- **Diskusijų lenta**: [GitHub diskusijos](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Vaizdo įrašų apžvalgos**: [YouTube grojaraštis](https://aka.ms/ml-beginners-videos)

## Pagrindinės technologijos

- **Python**: Pagrindinė kalba ML pamokoms (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatyvus įgyvendinimas naudojant tidyverse, tidymodels, caret
- **Jupyter**: Interaktyvios užrašinės Python pamokoms
- **R Markdown**: Dokumentai R pamokoms
- **Vue.js 3**: Testų programos karkasas
- **Flask**: Tinklo programų karkasas ML modelių diegimui
- **Docsify**: Dokumentacijos svetainės generatorius
- **GitHub Actions**: CI/CD ir automatiniai vertimai

## Saugumo aspektai

- **Jokių slaptažodžių kode**: Niekada neįkelkite API raktų ar prisijungimo duomenų
- **Priklausomybės**: Nuolat atnaujinkite npm ir pip paketus
- **Vartotojo įvestis**: Flask tinklo programų pavyzdžiai apima pagrindinį įvesties tikrinimą
- **Jautrūs duomenys**: Pavyzdiniai duomenų rinkiniai yra vieši ir nejautrūs

## Trikčių šalinimas

### Jupyter užrašinės

- **Branduolio problemos**: Perkraukite branduolį, jei langeliai užstringa: Kernel → Restart
- **Importavimo klaidos**: Įsitikinkite, kad visos reikalingos bibliotekos įdiegtos su pip
- **Kelio problemos**: Paleiskite užrašines iš jų turinio katalogo

### Testų programa

- **npm install nepavyksta**: Išvalykite npm talpyklą: `npm cache clean --force`
- **Porto konfliktai**: Pakeiskite portą su: `npm run serve -- --port 8081`
- **Kūrimo klaidos**: Ištrinkite `node_modules` ir iš naujo įdiekite: `rm -rf node_modules && npm install`

### R pamokos

- **Paketas nerastas**: Įdiekite su: `install.packages("package-name")`
- **RMarkdown atvaizdavimas**: Įsitikinkite, kad įdiegtas rmarkdown paketas
- **Branduolio problemos**: Gali reikėti įdiegti IRkernel Jupyter programai

## Projekto specifinės pastabos

- Tai pirmiausia yra **mokymo programa**, o ne gamybinis kodas
- Dėmesys skiriamas **ML sąvokų supratimui** per praktinę veiklą
- Kodo pavyzdžiai prioritetą teikia **aiškumui, o ne optimizavimui**
- Dauguma pamokų yra **savarankiškos** ir gali būti baigtos atskirai
- **Pateikiami sprendimai**, tačiau mokiniai turėtų pirmiausia bandyti užduotis
- Saugykla naudoja **Docsify** internetinei dokumentacijai be kūrimo žingsnio
- **Sketchnotes** pateikia vizualias sąvokų santraukas
- **Daugiakalbė parama** daro turinį prieinamą visame pasaulyje

---

**Atsakomybės atsisakymas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors stengiamės užtikrinti tikslumą, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar neteisingą interpretaciją, atsiradusią dėl šio vertimo naudojimo.