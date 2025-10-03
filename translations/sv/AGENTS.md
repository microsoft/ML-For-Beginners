<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:09:52+00:00",
  "source_file": "AGENTS.md",
  "language_code": "sv"
}
-->
# AGENTS.md

## Projektöversikt

Detta är **Maskininlärning för nybörjare**, en omfattande 12-veckors, 26-lektions kursplan som täcker klassiska maskininlärningskoncept med Python (främst med Scikit-learn) och R. Repositoriet är utformat som en självstudieressurs med praktiska projekt, quiz och uppgifter. Varje lektion utforskar ML-koncept med verkliga data från olika kulturer och regioner världen över.

Huvudkomponenter:
- **Utbildningsinnehåll**: 26 lektioner som täcker introduktion till ML, regression, klassificering, klustring, NLP, tidsserier och förstärkningsinlärning
- **Quiz-applikation**: Quiz-app baserad på Vue.js med tester före och efter lektionerna
- **Flerspråkigt stöd**: Automatiserade översättningar till 40+ språk via GitHub Actions
- **Dubbelspråkigt stöd**: Lektioner tillgängliga både i Python (Jupyter-notebooks) och R (R Markdown-filer)
- **Projektbaserat lärande**: Varje ämne inkluderar praktiska projekt och uppgifter

## Repositoriets struktur

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

Varje lektionsmapp innehåller vanligtvis:
- `README.md` - Huvudinnehåll för lektionen
- `notebook.ipynb` - Python Jupyter-notebook
- `solution/` - Lösningskod (Python- och R-versioner)
- `assignment.md` - Övningsuppgifter
- `images/` - Visuella resurser

## Installationskommandon

### För Python-lektioner

De flesta lektioner använder Jupyter-notebooks. Installera nödvändiga beroenden:

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

### För R-lektioner

R-lektioner finns i `solution/R/`-mappar som `.rmd` eller `.ipynb`-filer:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### För Quiz-applikationen

Quiz-appen är en Vue.js-applikation som finns i katalogen `quiz-app/`:

```bash
cd quiz-app
npm install
```

### För dokumentationssidan

För att köra dokumentationen lokalt:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Utvecklingsarbetsflöde

### Arbeta med lektionsnotebooks

1. Navigera till lektionskatalogen (t.ex. `2-Regression/1-Tools/`)
2. Öppna Jupyter-notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Arbeta igenom lektionsinnehållet och övningarna
4. Kontrollera lösningar i mappen `solution/` vid behov

### Python-utveckling

- Lektioner använder standardbibliotek för datavetenskap i Python
- Jupyter-notebooks för interaktivt lärande
- Lösningskod finns i varje lektions `solution/`-mapp

### R-utveckling

- R-lektioner är i `.rmd`-format (R Markdown)
- Lösningar finns i `solution/R/`-undermappar
- Använd RStudio eller Jupyter med R-kärna för att köra R-notebooks

### Quiz-applikationsutveckling

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

## Testinstruktioner

### Testning av Quiz-applikationen

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Obs**: Detta är främst ett utbildningsrepo. Det finns inga automatiserade tester för lektionsinnehåll. Validering görs genom:
- Att slutföra lektionsövningar
- Att köra notebook-celler framgångsrikt
- Att kontrollera utdata mot förväntade resultat i lösningarna

## Kodstilsguider

### Python-kod
- Följ PEP 8-stilguiden
- Använd tydliga, beskrivande variabelnamn
- Inkludera kommentarer för komplexa operationer
- Jupyter-notebooks bör ha markdown-celler som förklarar koncept

### JavaScript/Vue.js (Quiz-app)
- Följer Vue.js-stilguiden
- ESLint-konfiguration i `quiz-app/package.json`
- Kör `npm run lint` för att kontrollera och automatiskt fixa problem

### Dokumentation
- Markdown-filer bör vara tydliga och välstrukturerade
- Inkludera kodexempel i avgränsade kodblock
- Använd relativa länkar för interna referenser
- Följ befintliga formateringskonventioner

## Bygg och distribution

### Distribution av Quiz-applikationen

Quiz-appen kan distribueras till Azure Static Web Apps:

1. **Förutsättningar**:
   - Azure-konto
   - GitHub-repo (redan forkad)

2. **Distribuera till Azure**:
   - Skapa en Azure Static Web App-resurs
   - Anslut till GitHub-repo
   - Ange appens plats: `/quiz-app`
   - Ange utdata-plats: `dist`
   - Azure skapar automatiskt GitHub Actions-arbetsflöde

3. **GitHub Actions-arbetsflöde**:
   - Arbetsflödesfil skapas i `.github/workflows/azure-static-web-apps-*.yml`
   - Byggs och distribueras automatiskt vid push till huvudgrenen

### Dokumentation PDF

Generera PDF från dokumentationen:

```bash
npm install
npm run convert
```

## Översättningsarbetsflöde

**Viktigt**: Översättningar är automatiserade via GitHub Actions med Co-op Translator.

- Översättningar genereras automatiskt när ändringar pushas till `main`-grenen
- **ÖVERSÄTT INTE innehåll manuellt** - systemet hanterar detta
- Arbetsflöde definieras i `.github/workflows/co-op-translator.yml`
- Använder Azure AI/OpenAI-tjänster för översättning
- Stödjer 40+ språk

## Riktlinjer för bidrag

### För innehållsbidragare

1. **Forka repositoriet** och skapa en feature-gren
2. **Gör ändringar i lektionsinnehållet** om du lägger till/uppdaterar lektioner
3. **Ändra inte översatta filer** - de genereras automatiskt
4. **Testa din kod** - säkerställ att alla notebook-celler körs framgångsrikt
5. **Verifiera länkar och bilder** fungerar korrekt
6. **Skicka en pull request** med tydlig beskrivning

### Riktlinjer för pull requests

- **Titelformat**: `[Sektion] Kort beskrivning av ändringar`
  - Exempel: `[Regression] Fixade stavfel i lektion 5`
  - Exempel: `[Quiz-App] Uppdaterade beroenden`
- **Innan du skickar**:
  - Säkerställ att alla notebook-celler körs utan fel
  - Kör `npm run lint` om du ändrar quiz-appen
  - Verifiera markdown-formatering
  - Testa eventuella nya kodexempel
- **PR måste inkludera**:
  - Beskrivning av ändringar
  - Orsak till ändringar
  - Skärmdumpar vid UI-ändringar
- **Uppförandekod**: Följ [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Du måste signera Contributor License Agreement

## Lektionsstruktur

Varje lektion följer ett konsekvent mönster:

1. **Quiz före lektionen** - Testa grundläggande kunskaper
2. **Lektionsinnehåll** - Skriftliga instruktioner och förklaringar
3. **Koddemonstrationer** - Praktiska exempel i notebooks
4. **Kunskapskontroller** - Verifiera förståelse under lektionen
5. **Utmaning** - Tillämpa koncept självständigt
6. **Uppgift** - Fördjupad övning
7. **Quiz efter lektionen** - Utvärdera läranderesultat

## Referens för vanliga kommandon

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

## Ytterligare resurser

- **Microsoft Learn Collection**: [ML för nybörjarmoduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz-app**: [Online-quiz](https://ff-quizzes.netlify.app/en/ml/)
- **Diskussionsforum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videogenomgångar**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Viktiga teknologier

- **Python**: Huvudspråk för ML-lektioner (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativ implementering med tidyverse, tidymodels, caret
- **Jupyter**: Interaktiva notebooks för Python-lektioner
- **R Markdown**: Dokument för R-lektioner
- **Vue.js 3**: Ramverk för quiz-applikationen
- **Flask**: Ramverk för webapplikationer för ML-modelldistribution
- **Docsify**: Generator för dokumentationssidor
- **GitHub Actions**: CI/CD och automatiserade översättningar

## Säkerhetsöverväganden

- **Inga hemligheter i koden**: Lämna aldrig API-nycklar eller autentiseringsuppgifter
- **Beroenden**: Håll npm- och pip-paket uppdaterade
- **Användarinmatning**: Flask-webapp-exempel inkluderar grundläggande validering av inmatning
- **Känsliga data**: Exempeldataset är offentliga och icke-känsliga

## Felsökning

### Jupyter-notebooks

- **Kernelproblem**: Starta om kernel om celler fastnar: Kernel → Restart
- **Importfel**: Säkerställ att alla nödvändiga paket är installerade med pip
- **Sökvägsproblem**: Kör notebooks från deras innehållande katalog

### Quiz-applikation

- **npm install misslyckas**: Rensa npm-cache: `npm cache clean --force`
- **Portkonflikter**: Ändra port med: `npm run serve -- --port 8081`
- **Byggfel**: Ta bort `node_modules` och installera om: `rm -rf node_modules && npm install`

### R-lektioner

- **Paket saknas**: Installera med: `install.packages("package-name")`
- **RMarkdown-rendering**: Säkerställ att rmarkdown-paketet är installerat
- **Kernelproblem**: Kan behöva installera IRkernel för Jupyter

## Projekt-specifika anteckningar

- Detta är främst en **utbildningskursplan**, inte produktionskod
- Fokus ligger på **att förstå ML-koncept** genom praktisk övning
- Kodexempel prioriterar **tydlighet över optimering**
- De flesta lektioner är **självständiga** och kan slutföras oberoende
- **Lösningar tillhandahålls**, men deltagare bör försöka övningarna först
- Repositoriet använder **Docsify** för webbdokumentation utan byggsteg
- **Sketchnotes** ger visuella sammanfattningar av koncept
- **Flerspråkigt stöd** gör innehållet globalt tillgängligt

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiserade översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.