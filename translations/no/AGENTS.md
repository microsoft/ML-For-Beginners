<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:10:50+00:00",
  "source_file": "AGENTS.md",
  "language_code": "no"
}
-->
# AGENTS.md

## Prosjektoversikt

Dette er **Maskinlæring for nybegynnere**, et omfattende 12-ukers, 26-leksjons pensum som dekker klassiske maskinlæringskonsepter ved bruk av Python (primært med Scikit-learn) og R. Repositoriet er designet som en selvstyrt læringsressurs med praktiske prosjekter, quizzer og oppgaver. Hver leksjon utforsker ML-konsepter gjennom virkelige data fra ulike kulturer og regioner verden over.

Hovedkomponenter:
- **Pedagogisk innhold**: 26 leksjoner som dekker introduksjon til ML, regresjon, klassifisering, klynging, NLP, tidsserier og forsterkende læring
- **Quiz-applikasjon**: Quiz-app basert på Vue.js med vurderinger før og etter leksjoner
- **Flerspråklig støtte**: Automatiserte oversettelser til 40+ språk via GitHub Actions
- **To-språklig støtte**: Leksjoner tilgjengelig både i Python (Jupyter-notebooks) og R (R Markdown-filer)
- **Prosjektbasert læring**: Hvert tema inkluderer praktiske prosjekter og oppgaver

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

Hver leksjonsmappe inneholder vanligvis:
- `README.md` - Hovedinnholdet i leksjonen
- `notebook.ipynb` - Python Jupyter-notebook
- `solution/` - Løsningskode (Python- og R-versjoner)
- `assignment.md` - Øvingsoppgaver
- `images/` - Visuelle ressurser

## Oppsettskommandoer

### For Python-leksjoner

De fleste leksjoner bruker Jupyter-notebooks. Installer nødvendige avhengigheter:

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

### For R-leksjoner

R-leksjoner finnes i `solution/R/`-mapper som `.rmd` eller `.ipynb`-filer:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### For Quiz-applikasjonen

Quiz-appen er en Vue.js-applikasjon som ligger i `quiz-app/`-katalogen:

```bash
cd quiz-app
npm install
```

### For dokumentasjonsnettstedet

For å kjøre dokumentasjonen lokalt:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Utviklingsarbeidsflyt

### Arbeide med leksjonsnotebooks

1. Naviger til leksjonsmappen (f.eks. `2-Regression/1-Tools/`)
2. Åpne Jupyter-notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Jobb deg gjennom leksjonsinnholdet og oppgavene
4. Sjekk løsninger i `solution/`-mappen hvis nødvendig

### Python-utvikling

- Leksjoner bruker standard Python-biblioteker for datavitenskap
- Jupyter-notebooks for interaktiv læring
- Løsningskode tilgjengelig i hver leksjons `solution/`-mappe

### R-utvikling

- R-leksjoner er i `.rmd`-format (R Markdown)
- Løsninger finnes i `solution/R/`-undermapper
- Bruk RStudio eller Jupyter med R-kjerne for å kjøre R-notebooks

### Quiz-applikasjonsutvikling

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

## Testinstruksjoner

### Testing av quiz-applikasjonen

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Merk**: Dette er primært et pedagogisk pensum-repositorium. Det finnes ingen automatiserte tester for leksjonsinnhold. Validering gjøres gjennom:
- Fullføring av leksjonsoppgaver
- Kjøre notebook-celler vellykket
- Sjekke output mot forventede resultater i løsninger

## Retningslinjer for kodestil

### Python-kode
- Følg PEP 8-stilretningslinjer
- Bruk klare, beskrivende variabelnavn
- Inkluder kommentarer for komplekse operasjoner
- Jupyter-notebooks bør ha markdown-celler som forklarer konsepter

### JavaScript/Vue.js (Quiz-app)
- Følger Vue.js-stilguide
- ESLint-konfigurasjon i `quiz-app/package.json`
- Kjør `npm run lint` for å sjekke og automatisk fikse problemer

### Dokumentasjon
- Markdown-filer bør være klare og godt strukturerte
- Inkluder kodeeksempler i avgrensede kodeblokker
- Bruk relative lenker for interne referanser
- Følg eksisterende formateringskonvensjoner

## Bygging og distribusjon

### Distribusjon av quiz-applikasjonen

Quiz-appen kan distribueres til Azure Static Web Apps:

1. **Forutsetninger**:
   - Azure-konto
   - GitHub-repositorium (allerede forked)

2. **Distribuer til Azure**:
   - Opprett Azure Static Web App-ressurs
   - Koble til GitHub-repositorium
   - Sett app-plassering: `/quiz-app`
   - Sett output-plassering: `dist`
   - Azure oppretter automatisk GitHub Actions-arbeidsflyt

3. **GitHub Actions-arbeidsflyt**:
   - Arbeidsflytfil opprettet i `.github/workflows/azure-static-web-apps-*.yml`
   - Bygger og distribuerer automatisk ved push til hovedgrenen

### Dokumentasjon PDF

Generer PDF fra dokumentasjonen:

```bash
npm install
npm run convert
```

## Oversettelsesarbeidsflyt

**Viktig**: Oversettelser er automatisert via GitHub Actions ved bruk av Co-op Translator.

- Oversettelser genereres automatisk når endringer pushes til `main`-grenen
- **IKKE oversett innhold manuelt** - systemet håndterer dette
- Arbeidsflyt definert i `.github/workflows/co-op-translator.yml`
- Bruker Azure AI/OpenAI-tjenester for oversettelse
- Støtter 40+ språk

## Retningslinjer for bidrag

### For innholdsbidragsytere

1. **Fork repositoriet** og opprett en feature branch
2. **Gjør endringer i leksjonsinnholdet** hvis du legger til/oppdaterer leksjoner
3. **Ikke endre oversatte filer** - de genereres automatisk
4. **Test koden din** - sørg for at alle notebook-celler kjører vellykket
5. **Verifiser lenker og bilder** fungerer korrekt
6. **Send inn en pull request** med klar beskrivelse

### Retningslinjer for pull requests

- **Tittelformat**: `[Seksjon] Kort beskrivelse av endringer`
  - Eksempel: `[Regression] Fiks skrivefeil i leksjon 5`
  - Eksempel: `[Quiz-App] Oppdater avhengigheter`
- **Før innsending**:
  - Sørg for at alle notebook-celler kjører uten feil
  - Kjør `npm run lint` hvis du endrer quiz-app
  - Verifiser markdown-formatering
  - Test eventuelle nye kodeeksempler
- **PR må inkludere**:
  - Beskrivelse av endringer
  - Årsak til endringer
  - Skjermbilder hvis UI-endringer
- **Code of Conduct**: Følg [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Du må signere Contributor License Agreement

## Leksjonsstruktur

Hver leksjon følger et konsistent mønster:

1. **Quiz før leksjon** - Test grunnleggende kunnskap
2. **Leksjonsinnhold** - Skriftlige instruksjoner og forklaringer
3. **Kodeeksempler** - Praktiske eksempler i notebooks
4. **Kunnskapssjekker** - Verifiser forståelse underveis
5. **Utfordring** - Bruk konsepter selvstendig
6. **Oppgave** - Utvidet praksis
7. **Quiz etter leksjon** - Vurder læringsutbytte

## Referanse for vanlige kommandoer

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

## Ekstra ressurser

- **Microsoft Learn Collection**: [ML for Beginners-moduler](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz-app**: [Online quizzer](https://ff-quizzes.netlify.app/en/ml/)
- **Diskusjonsforum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videogjennomganger**: [YouTube-spilleliste](https://aka.ms/ml-beginners-videos)

## Viktige teknologier

- **Python**: Hovedspråk for ML-leksjoner (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativ implementering med tidyverse, tidymodels, caret
- **Jupyter**: Interaktive notebooks for Python-leksjoner
- **R Markdown**: Dokumenter for R-leksjoner
- **Vue.js 3**: Rammeverk for quiz-applikasjonen
- **Flask**: Webapplikasjonsrammeverk for distribusjon av ML-modeller
- **Docsify**: Generator for dokumentasjonsnettsted
- **GitHub Actions**: CI/CD og automatiserte oversettelser

## Sikkerhetsvurderinger

- **Ingen hemmeligheter i kode**: Aldri legg inn API-nøkler eller legitimasjon
- **Avhengigheter**: Hold npm- og pip-pakker oppdatert
- **Brukerinndata**: Flask-webapp-eksempler inkluderer grunnleggende validering av input
- **Sensitive data**: Eksempeldatasett er offentlige og ikke-sensitive

## Feilsøking

### Jupyter-notebooks

- **Kjerneproblemer**: Start kjerne på nytt hvis celler henger: Kernel → Restart
- **Importfeil**: Sørg for at alle nødvendige pakker er installert med pip
- **Stiproblemer**: Kjør notebooks fra deres inneholdende katalog

### Quiz-applikasjon

- **npm install feiler**: Tøm npm-cache: `npm cache clean --force`
- **Portkonflikter**: Endre port med: `npm run serve -- --port 8081`
- **Byggefeil**: Slett `node_modules` og installer på nytt: `rm -rf node_modules && npm install`

### R-leksjoner

- **Pakke ikke funnet**: Installer med: `install.packages("package-name")`
- **RMarkdown-rendering**: Sørg for at rmarkdown-pakken er installert
- **Kjerneproblemer**: Kan trenge å installere IRkernel for Jupyter

## Prosjektspesifikke notater

- Dette er primært et **læringspensum**, ikke produksjonskode
- Fokus er på **forståelse av ML-konsepter** gjennom praktisk øvelse
- Kodeeksempler prioriterer **klarhet fremfor optimalisering**
- De fleste leksjoner er **selvstendige** og kan fullføres uavhengig
- **Løsninger er tilgjengelige**, men lærende bør forsøke oppgavene først
- Repositoriet bruker **Docsify** for webdokumentasjon uten byggeprosess
- **Sketchnotes** gir visuelle oppsummeringer av konsepter
- **Flerspråklig støtte** gjør innholdet globalt tilgjengelig

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.