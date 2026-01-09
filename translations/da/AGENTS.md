<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:10:20+00:00",
  "source_file": "AGENTS.md",
  "language_code": "da"
}
-->
# AGENTS.md

## Projektoversigt

Dette er **Machine Learning for Beginners**, et omfattende 12-ugers, 26-lektioners kursus, der dækker klassiske machine learning-koncepter ved hjælp af Python (primært med Scikit-learn) og R. Repositoriet er designet som en selvstudie-læringsressource med praktiske projekter, quizzer og opgaver. Hver lektion udforsker ML-koncepter gennem virkelige data fra forskellige kulturer og regioner verden over.

Nøglekomponenter:
- **Uddannelsesindhold**: 26 lektioner, der dækker introduktion til ML, regression, klassifikation, clustering, NLP, tidsserier og forstærkningslæring
- **Quiz-applikation**: Vue.js-baseret quiz-app med før- og efter-lektionsvurderinger
- **Flersproget support**: Automatiske oversættelser til 40+ sprog via GitHub Actions
- **Dobbelt sprogsupport**: Lektioner tilgængelige i både Python (Jupyter-notebooks) og R (R Markdown-filer)
- **Projektbaseret læring**: Hvert emne inkluderer praktiske projekter og opgaver

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

Hver lektionsmappe indeholder typisk:
- `README.md` - Hovedindholdet for lektionen
- `notebook.ipynb` - Python Jupyter-notebook
- `solution/` - Løsningskode (Python- og R-versioner)
- `assignment.md` - Øvelsesopgaver
- `images/` - Visuelle ressourcer

## Opsætningskommandoer

### For Python-lektioner

De fleste lektioner bruger Jupyter-notebooks. Installer nødvendige afhængigheder:

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

### For R-lektioner

R-lektioner findes i `solution/R/` mapper som `.rmd` eller `.ipynb` filer:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### For Quiz-applikationen

Quiz-appen er en Vue.js-applikation placeret i `quiz-app/` mappen:

```bash
cd quiz-app
npm install
```

### For dokumentationssiden

For at køre dokumentationen lokalt:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Udviklingsarbejdsgang

### Arbejde med lektionsnotebooks

1. Naviger til lektionsmappen (f.eks. `2-Regression/1-Tools/`)
2. Åbn Jupyter-notebooken:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Gennemgå lektionsindholdet og øvelserne
4. Tjek løsninger i `solution/` mappen, hvis nødvendigt

### Python-udvikling

- Lektioner bruger standard Python data science-biblioteker
- Jupyter-notebooks til interaktiv læring
- Løsningskode tilgængelig i hver lektions `solution/` mappe

### R-udvikling

- R-lektioner er i `.rmd` format (R Markdown)
- Løsninger findes i `solution/R/` undermapper
- Brug RStudio eller Jupyter med R-kernel til at køre R-notebooks

### Quiz-applikationsudvikling

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

### Test af quiz-applikationen

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Bemærk**: Dette er primært et uddannelsesrepositorium. Der er ingen automatiserede tests for lektionsindhold. Validering sker gennem:
- Fuldførelse af lektionsøvelser
- Succesfuld kørsel af notebook-celler
- Sammenligning af output med forventede resultater i løsninger

## Kodestilretningslinjer

### Python-kode
- Følg PEP 8 stilretningslinjer
- Brug klare, beskrivende variabelnavne
- Inkluder kommentarer til komplekse operationer
- Jupyter-notebooks bør have markdown-celler, der forklarer koncepter

### JavaScript/Vue.js (Quiz-app)
- Følger Vue.js stilguide
- ESLint-konfiguration i `quiz-app/package.json`
- Kør `npm run lint` for at tjekke og automatisk rette problemer

### Dokumentation
- Markdown-filer skal være klare og velstrukturerede
- Inkluder kodeeksempler i indhegnede kodeblokke
- Brug relative links til interne referencer
- Følg eksisterende formateringskonventioner

## Bygning og udrulning

### Udrulning af quiz-applikationen

Quiz-appen kan udrulles til Azure Static Web Apps:

1. **Forudsætninger**:
   - Azure-konto
   - GitHub-repositorium (allerede forked)

2. **Udrul til Azure**:
   - Opret Azure Static Web App-ressource
   - Forbind til GitHub-repositorium
   - Angiv app-placering: `/quiz-app`
   - Angiv output-placering: `dist`
   - Azure opretter automatisk GitHub Actions workflow

3. **GitHub Actions workflow**:
   - Workflow-fil oprettet i `.github/workflows/azure-static-web-apps-*.yml`
   - Bygger og udruller automatisk ved push til main branch

### Dokumentations-PDF

Generer PDF fra dokumentationen:

```bash
npm install
npm run convert
```

## Oversættelsesarbejdsgang

**Vigtigt**: Oversættelser er automatiserede via GitHub Actions ved hjælp af Co-op Translator.

- Oversættelser genereres automatisk, når ændringer pushes til `main` branch
- **MÅ IKKE manuelt oversætte indhold** - systemet håndterer dette
- Workflow defineret i `.github/workflows/co-op-translator.yml`
- Bruger Azure AI/OpenAI-tjenester til oversættelse
- Understøtter 40+ sprog

## Retningslinjer for bidrag

### For indholdsbidragydere

1. **Fork repositoriet** og opret en feature branch
2. **Foretag ændringer i lektionsindhold** hvis du tilføjer/opdaterer lektioner
3. **Undlad at ændre oversatte filer** - de genereres automatisk
4. **Test din kode** - sørg for, at alle notebook-celler kører succesfuldt
5. **Verificer links og billeder** fungerer korrekt
6. **Indsend en pull request** med en klar beskrivelse

### Retningslinjer for pull requests

- **Titelformat**: `[Sektion] Kort beskrivelse af ændringer`
  - Eksempel: `[Regression] Ret stavefejl i lektion 5`
  - Eksempel: `[Quiz-App] Opdater afhængigheder`
- **Før indsendelse**:
  - Sørg for, at alle notebook-celler kører uden fejl
  - Kør `npm run lint`, hvis du ændrer quiz-app
  - Verificer markdown-formatering
  - Test eventuelle nye kodeeksempler
- **PR skal inkludere**:
  - Beskrivelse af ændringer
  - Årsag til ændringer
  - Skærmbilleder, hvis der er UI-ændringer
- **Adfærdskodeks**: Følg [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Du skal underskrive Contributor License Agreement

## Lektionstruktur

Hver lektion følger et konsistent mønster:

1. **Quiz før lektionen** - Test grundlæggende viden
2. **Lektionsindhold** - Skriftlige instruktioner og forklaringer
3. **Kodeeksempler** - Praktiske eksempler i notebooks
4. **Videnskontrol** - Bekræft forståelse undervejs
5. **Udfordring** - Anvend koncepter selvstændigt
6. **Opgave** - Udvidet praksis
7. **Quiz efter lektionen** - Vurder læringsresultater

## Reference for almindelige kommandoer

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

## Yderligere ressourcer

- **Microsoft Learn Collection**: [ML for Beginners-moduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz-app**: [Online quizzer](https://ff-quizzes.netlify.app/en/ml/)
- **Diskussionsforum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videovejledninger**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Nøgleteknologier

- **Python**: Primært sprog til ML-lektioner (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativ implementering ved hjælp af tidyverse, tidymodels, caret
- **Jupyter**: Interaktive notebooks til Python-lektioner
- **R Markdown**: Dokumenter til R-lektioner
- **Vue.js 3**: Quiz-applikationsramme
- **Flask**: Webapplikationsramme til ML-modeludrulning
- **Docsify**: Generator til dokumentationssider
- **GitHub Actions**: CI/CD og automatiserede oversættelser

## Sikkerhedsovervejelser

- **Ingen hemmeligheder i kode**: Aldrig commit API-nøgler eller legitimationsoplysninger
- **Afhængigheder**: Hold npm- og pip-pakker opdaterede
- **Brugerinput**: Flask-webapp-eksempler inkluderer grundlæggende inputvalidering
- **Følsomme data**: Eksempeldatasæt er offentlige og ikke-følsomme

## Fejlfinding

### Jupyter-notebooks

- **Kernelproblemer**: Genstart kernel, hvis celler hænger: Kernel → Restart
- **Importfejl**: Sørg for, at alle nødvendige pakker er installeret med pip
- **Stiproblemer**: Kør notebooks fra deres indeholdende mappe

### Quiz-applikation

- **npm install fejler**: Rens npm-cache: `npm cache clean --force`
- **Portkonflikter**: Skift port med: `npm run serve -- --port 8081`
- **Bygningsfejl**: Slet `node_modules` og geninstaller: `rm -rf node_modules && npm install`

### R-lektioner

- **Pakke ikke fundet**: Installer med: `install.packages("pakke-navn")`
- **RMarkdown rendering**: Sørg for, at rmarkdown-pakken er installeret
- **Kernelproblemer**: Kan kræve installation af IRkernel til Jupyter

## Projektspecifikke noter

- Dette er primært et **læringskursus**, ikke produktionskode
- Fokus er på **forståelse af ML-koncepter** gennem praktisk øvelse
- Kodeeksempler prioriterer **klarhed frem for optimering**
- De fleste lektioner er **selvstændige** og kan gennemføres uafhængigt
- **Løsninger er tilgængelige**, men lærende bør forsøge øvelser først
- Repositoriet bruger **Docsify** til webdokumentation uden build-trin
- **Sketchnotes** giver visuelle opsummeringer af koncepter
- **Flersproget support** gør indhold globalt tilgængeligt

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.