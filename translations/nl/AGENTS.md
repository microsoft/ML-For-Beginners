<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:11:53+00:00",
  "source_file": "AGENTS.md",
  "language_code": "nl"
}
-->
# AGENTS.md

## Projectoverzicht

Dit is **Machine Learning voor Beginners**, een uitgebreide 12-weekse, 26-lessen curriculum dat klassieke machine learning-concepten behandelt met behulp van Python (voornamelijk met Scikit-learn) en R. De repository is ontworpen als een zelfstudiebron met praktische projecten, quizzen en opdrachten. Elke les verkent ML-concepten aan de hand van real-world data uit verschillende culturen en regio's wereldwijd.

Belangrijke onderdelen:
- **Educatieve inhoud**: 26 lessen over introductie tot ML, regressie, classificatie, clustering, NLP, tijdreeksen en reinforcement learning
- **Quizapplicatie**: Vue.js-gebaseerde quizapp met pre- en post-les beoordelingen
- **Meertalige ondersteuning**: Automatische vertalingen naar meer dan 40 talen via GitHub Actions
- **Dubbele taalondersteuning**: Lessen beschikbaar in zowel Python (Jupyter-notebooks) als R (R Markdown-bestanden)
- **Projectgebaseerd leren**: Elk onderwerp bevat praktische projecten en opdrachten

## Repositorystructuur

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

Elke lesmap bevat doorgaans:
- `README.md` - Hoofdinhoud van de les
- `notebook.ipynb` - Python Jupyter-notebook
- `solution/` - Oplossingscode (Python- en R-versies)
- `assignment.md` - Oefenopdrachten
- `images/` - Visuele bronnen

## Setupcommando's

### Voor Python-lessen

De meeste lessen gebruiken Jupyter-notebooks. Installeer de vereiste afhankelijkheden:

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

### Voor R-lessen

R-lessen bevinden zich in `solution/R/` mappen als `.rmd` of `.ipynb` bestanden:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Voor Quizapplicatie

De quizapp bevindt zich in de `quiz-app/` map:

```bash
cd quiz-app
npm install
```

### Voor documentatiesite

Om de documentatie lokaal te draaien:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Ontwikkelworkflow

### Werken met lesnotebooks

1. Navigeer naar de lesmap (bijv. `2-Regression/1-Tools/`)
2. Open de Jupyter-notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Werk door de lesinhoud en oefeningen
4. Controleer oplossingen in de `solution/` map indien nodig

### Python-ontwikkeling

- Lessen gebruiken standaard Python data science-bibliotheken
- Jupyter-notebooks voor interactieve leerervaring
- Oplossingscode beschikbaar in de `solution/` map van elke les

### R-ontwikkeling

- R-lessen zijn in `.rmd` formaat (R Markdown)
- Oplossingen bevinden zich in `solution/R/` submappen
- Gebruik RStudio of Jupyter met R-kernel om R-notebooks uit te voeren

### Ontwikkeling van quizapplicatie

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

## Testinstructies

### Testen van quizapplicatie

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Let op**: Dit is voornamelijk een educatieve curriculumrepository. Er zijn geen geautomatiseerde tests voor lesinhoud. Validatie gebeurt door:
- Het voltooien van lesoefeningen
- Het succesvol uitvoeren van notebookcellen
- Het vergelijken van output met verwachte resultaten in oplossingen

## Richtlijnen voor codestijl

### Python-code
- Volg PEP 8-stijlrichtlijnen
- Gebruik duidelijke, beschrijvende variabelenamen
- Voeg opmerkingen toe voor complexe operaties
- Jupyter-notebooks moeten markdowncellen bevatten die concepten uitleggen

### JavaScript/Vue.js (Quizapp)
- Volgt Vue.js-stijlgids
- ESLint-configuratie in `quiz-app/package.json`
- Voer `npm run lint` uit om problemen te controleren en automatisch op te lossen

### Documentatie
- Markdown-bestanden moeten duidelijk en goed gestructureerd zijn
- Voeg codevoorbeelden toe in afgebakende codeblokken
- Gebruik relatieve links voor interne verwijzingen
- Volg bestaande opmaakconventies

## Build en deployment

### Deployment van quizapplicatie

De quizapp kan worden gedeployed naar Azure Static Web Apps:

1. **Vereisten**:
   - Azure-account
   - GitHub-repository (al geforkt)

2. **Deploy naar Azure**:
   - Maak een Azure Static Web App-resource
   - Verbind met GitHub-repository
   - Stel app-locatie in: `/quiz-app`
   - Stel outputlocatie in: `dist`
   - Azure maakt automatisch een GitHub Actions-workflow aan

3. **GitHub Actions-workflow**:
   - Workflowbestand aangemaakt in `.github/workflows/azure-static-web-apps-*.yml`
   - Bouwt en deployt automatisch bij push naar de hoofdbranch

### Documentatie PDF

Genereer PDF vanuit documentatie:

```bash
npm install
npm run convert
```

## Vertaalworkflow

**Belangrijk**: Vertalingen worden automatisch uitgevoerd via GitHub Actions met behulp van Co-op Translator.

- Vertalingen worden automatisch gegenereerd wanneer wijzigingen naar de `main` branch worden gepusht
- **NIET handmatig vertalen** - het systeem regelt dit
- Workflow gedefinieerd in `.github/workflows/co-op-translator.yml`
- Gebruikt Azure AI/OpenAI-services voor vertaling
- Ondersteunt meer dan 40 talen

## Richtlijnen voor bijdragen

### Voor inhoudelijke bijdragers

1. **Fork de repository** en maak een featurebranch
2. **Breng wijzigingen aan in lesinhoud** als je lessen toevoegt of bijwerkt
3. **Wijzig geen vertaalde bestanden** - deze worden automatisch gegenereerd
4. **Test je code** - zorg ervoor dat alle notebookcellen succesvol worden uitgevoerd
5. **Controleer links en afbeeldingen** op correcte werking
6. **Dien een pull request in** met een duidelijke beschrijving

### Richtlijnen voor pull requests

- **Titelindeling**: `[Sectie] Korte beschrijving van wijzigingen`
  - Voorbeeld: `[Regressie] Typfout in les 5 corrigeren`
  - Voorbeeld: `[Quiz-App] Dependencies bijwerken`
- **Voor het indienen**:
  - Zorg ervoor dat alle notebookcellen zonder fouten worden uitgevoerd
  - Voer `npm run lint` uit als je de quiz-app wijzigt
  - Controleer markdown-opmaak
  - Test eventuele nieuwe codevoorbeelden
- **PR moet bevatten**:
  - Beschrijving van wijzigingen
  - Reden voor wijzigingen
  - Screenshots bij UI-wijzigingen
- **Gedragscode**: Volg de [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Je moet de Contributor License Agreement ondertekenen

## Lesstructuur

Elke les volgt een consistent patroon:

1. **Pre-lecture quiz** - Test basiskennis
2. **Lesinhoud** - Geschreven instructies en uitleg
3. **Codevoorbeelden** - Praktische voorbeelden in notebooks
4. **Kennischecks** - Controleer begrip gedurende de les
5. **Uitdaging** - Pas concepten zelfstandig toe
6. **Opdracht** - Uitgebreide oefening
7. **Post-lecture quiz** - Beoordeel leerresultaten

## Referentie voor veelgebruikte commando's

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

## Aanvullende bronnen

- **Microsoft Learn Collection**: [ML voor Beginners-modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quizapp**: [Online quizzen](https://ff-quizzes.netlify.app/en/ml/)
- **Discussieforum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videowalkthroughs**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Belangrijke technologieën

- **Python**: Primaire taal voor ML-lessen (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatieve implementatie met tidyverse, tidymodels, caret
- **Jupyter**: Interactieve notebooks voor Python-lessen
- **R Markdown**: Documenten voor R-lessen
- **Vue.js 3**: Framework voor quizapplicatie
- **Flask**: Webapplicatieframework voor ML-modeldeployment
- **Docsify**: Generator voor documentatiesites
- **GitHub Actions**: CI/CD en automatische vertalingen

## Overwegingen voor beveiliging

- **Geen geheimen in code**: Commit nooit API-sleutels of inloggegevens
- **Afhankelijkheden**: Houd npm- en pip-pakketten up-to-date
- **Gebruikersinvoer**: Flask-webappvoorbeelden bevatten basisvalidatie van invoer
- **Gevoelige data**: Voorbeelddatasets zijn openbaar en niet gevoelig

## Probleemoplossing

### Jupyter-notebooks

- **Kernelproblemen**: Herstart kernel als cellen vastlopen: Kernel → Restart
- **Importfouten**: Zorg ervoor dat alle vereiste pakketten zijn geïnstalleerd met pip
- **Padproblemen**: Voer notebooks uit vanuit hun map

### Quizapplicatie

- **npm install mislukt**: Wis npm-cache: `npm cache clean --force`
- **Poortconflicten**: Wijzig poort met: `npm run serve -- --port 8081`
- **Buildfouten**: Verwijder `node_modules` en installeer opnieuw: `rm -rf node_modules && npm install`

### R-lessen

- **Pakket niet gevonden**: Installeer met: `install.packages("package-name")`
- **RMarkdown rendering**: Zorg ervoor dat het rmarkdown-pakket is geïnstalleerd
- **Kernelproblemen**: Mogelijk moet IRkernel worden geïnstalleerd voor Jupyter

## Projectspecifieke opmerkingen

- Dit is voornamelijk een **leerplan**, geen productiecode
- Focus ligt op **begrijpen van ML-concepten** door praktische oefeningen
- Codevoorbeelden geven prioriteit aan **duidelijkheid boven optimalisatie**
- De meeste lessen zijn **zelfstandig** en kunnen onafhankelijk worden voltooid
- **Oplossingen beschikbaar**, maar deelnemers moeten eerst oefeningen proberen
- Repository gebruikt **Docsify** voor webdocumentatie zonder buildstap
- **Sketchnotes** bieden visuele samenvattingen van concepten
- **Meertalige ondersteuning** maakt inhoud wereldwijd toegankelijk

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.