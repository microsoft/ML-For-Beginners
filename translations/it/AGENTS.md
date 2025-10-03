<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:07:03+00:00",
  "source_file": "AGENTS.md",
  "language_code": "it"
}
-->
# AGENTS.md

## Panoramica del Progetto

Questo è **Machine Learning per Principianti**, un curriculum completo di 12 settimane e 26 lezioni che copre i concetti classici del machine learning utilizzando Python (principalmente con Scikit-learn) e R. Il repository è progettato come una risorsa di apprendimento autonomo con progetti pratici, quiz e compiti. Ogni lezione esplora i concetti di ML attraverso dati reali provenienti da diverse culture e regioni del mondo.

Componenti principali:
- **Contenuti Educativi**: 26 lezioni che coprono introduzione al ML, regressione, classificazione, clustering, NLP, serie temporali e apprendimento per rinforzo
- **Applicazione Quiz**: App per quiz basata su Vue.js con valutazioni pre e post lezione
- **Supporto Multilingue**: Traduzioni automatiche in oltre 40 lingue tramite GitHub Actions
- **Supporto Duale**: Lezioni disponibili sia in Python (notebook Jupyter) che in R (file R Markdown)
- **Apprendimento Basato su Progetti**: Ogni argomento include progetti pratici e compiti

## Struttura del Repository

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

Ogni cartella delle lezioni contiene tipicamente:
- `README.md` - Contenuto principale della lezione
- `notebook.ipynb` - Notebook Jupyter in Python
- `solution/` - Codice soluzione (versioni Python e R)
- `assignment.md` - Esercizi pratici
- `images/` - Risorse visive

## Comandi di Configurazione

### Per Lezioni in Python

La maggior parte delle lezioni utilizza notebook Jupyter. Installa le dipendenze necessarie:

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

### Per Lezioni in R

Le lezioni in R si trovano nelle cartelle `solution/R/` come file `.rmd` o `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Per Applicazione Quiz

L'app per quiz è un'applicazione Vue.js situata nella directory `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Per Sito di Documentazione

Per eseguire la documentazione localmente:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Flusso di Lavoro per lo Sviluppo

### Lavorare con i Notebook delle Lezioni

1. Vai alla directory della lezione (es. `2-Regression/1-Tools/`)
2. Apri il notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Segui il contenuto della lezione e gli esercizi
4. Controlla le soluzioni nella cartella `solution/` se necessario

### Sviluppo in Python

- Le lezioni utilizzano librerie standard di data science in Python
- Notebook Jupyter per apprendimento interattivo
- Codice soluzione disponibile nella cartella `solution/` di ogni lezione

### Sviluppo in R

- Le lezioni in R sono in formato `.rmd` (R Markdown)
- Soluzioni situate nelle sottodirectory `solution/R/`
- Usa RStudio o Jupyter con kernel R per eseguire i notebook R

### Sviluppo Applicazione Quiz

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

## Istruzioni per il Testing

### Testing Applicazione Quiz

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Nota**: Questo è principalmente un repository di curriculum educativo. Non ci sono test automatizzati per il contenuto delle lezioni. La validazione viene effettuata tramite:
- Completamento degli esercizi delle lezioni
- Esecuzione delle celle del notebook con successo
- Controllo dei risultati rispetto alle soluzioni previste

## Linee Guida per lo Stile del Codice

### Codice Python
- Segui le linee guida dello stile PEP 8
- Usa nomi di variabili chiari e descrittivi
- Includi commenti per operazioni complesse
- I notebook Jupyter devono avere celle markdown che spiegano i concetti

### JavaScript/Vue.js (App Quiz)
- Segue la guida di stile Vue.js
- Configurazione ESLint in `quiz-app/package.json`
- Esegui `npm run lint` per controllare e correggere automaticamente i problemi

### Documentazione
- I file markdown devono essere chiari e ben strutturati
- Includi esempi di codice in blocchi di codice delimitati
- Usa link relativi per riferimenti interni
- Segui le convenzioni di formattazione esistenti

## Build e Deployment

### Deployment Applicazione Quiz

L'app per quiz può essere distribuita su Azure Static Web Apps:

1. **Prerequisiti**:
   - Account Azure
   - Repository GitHub (già forkato)

2. **Distribuisci su Azure**:
   - Crea una risorsa Azure Static Web App
   - Connetti al repository GitHub
   - Imposta la posizione dell'app: `/quiz-app`
   - Imposta la posizione di output: `dist`
   - Azure crea automaticamente il workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - File workflow creato in `.github/workflows/azure-static-web-apps-*.yml`
   - Costruisce e distribuisce automaticamente al push sul branch principale

### PDF Documentazione

Genera PDF dalla documentazione:

```bash
npm install
npm run convert
```

## Flusso di Lavoro per le Traduzioni

**Importante**: Le traduzioni sono automatizzate tramite GitHub Actions utilizzando Co-op Translator.

- Le traduzioni vengono generate automaticamente quando vengono apportate modifiche al branch `main`
- **NON tradurre manualmente il contenuto** - il sistema gestisce questo
- Workflow definito in `.github/workflows/co-op-translator.yml`
- Utilizza servizi Azure AI/OpenAI per la traduzione
- Supporta oltre 40 lingue

## Linee Guida per i Contributi

### Per i Contributori di Contenuti

1. **Forka il repository** e crea un branch per la feature
2. **Apporta modifiche al contenuto delle lezioni** se aggiungi/aggiorni lezioni
3. **Non modificare i file tradotti** - sono generati automaticamente
4. **Testa il tuo codice** - assicurati che tutte le celle del notebook vengano eseguite con successo
5. **Verifica che i link e le immagini** funzionino correttamente
6. **Invia una pull request** con una descrizione chiara

### Linee Guida per le Pull Request

- **Formato del titolo**: `[Sezione] Breve descrizione delle modifiche`
  - Esempio: `[Regression] Correggi errore di battitura nella lezione 5`
  - Esempio: `[Quiz-App] Aggiorna dipendenze`
- **Prima di inviare**:
  - Assicurati che tutte le celle del notebook vengano eseguite senza errori
  - Esegui `npm run lint` se modifichi quiz-app
  - Verifica la formattazione markdown
  - Testa eventuali nuovi esempi di codice
- **La PR deve includere**:
  - Descrizione delle modifiche
  - Motivo delle modifiche
  - Screenshot se ci sono modifiche all'interfaccia utente
- **Codice di Condotta**: Segui il [Codice di Condotta Open Source di Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Sarà necessario firmare il Contributor License Agreement

## Struttura delle Lezioni

Ogni lezione segue un modello coerente:

1. **Quiz pre-lezione** - Testa le conoscenze di base
2. **Contenuto della lezione** - Istruzioni scritte e spiegazioni
3. **Dimostrazioni di codice** - Esempi pratici nei notebook
4. **Verifiche di conoscenza** - Controlla la comprensione durante la lezione
5. **Sfida** - Applica i concetti in modo indipendente
6. **Compito** - Pratica estesa
7. **Quiz post-lezione** - Valuta i risultati dell'apprendimento

## Riferimenti ai Comandi Comuni

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

## Risorse Aggiuntive

- **Collezione Microsoft Learn**: [Moduli ML per Principianti](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **App Quiz**: [Quiz online](https://ff-quizzes.netlify.app/en/ml/)
- **Forum di Discussione**: [Discussioni su GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video Tutorial**: [Playlist YouTube](https://aka.ms/ml-beginners-videos)

## Tecnologie Chiave

- **Python**: Linguaggio principale per le lezioni di ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Implementazione alternativa utilizzando tidyverse, tidymodels, caret
- **Jupyter**: Notebook interattivi per lezioni in Python
- **R Markdown**: Documenti per lezioni in R
- **Vue.js 3**: Framework per l'applicazione quiz
- **Flask**: Framework per applicazioni web per il deployment di modelli ML
- **Docsify**: Generatore di siti di documentazione
- **GitHub Actions**: CI/CD e traduzioni automatizzate

## Considerazioni sulla Sicurezza

- **Nessun segreto nel codice**: Non commettere mai chiavi API o credenziali
- **Dipendenze**: Mantieni aggiornati i pacchetti npm e pip
- **Input utente**: Gli esempi di app web Flask includono una validazione di input di base
- **Dati sensibili**: I dataset di esempio sono pubblici e non sensibili

## Risoluzione dei Problemi

### Notebook Jupyter

- **Problemi con il kernel**: Riavvia il kernel se le celle si bloccano: Kernel → Restart
- **Errori di importazione**: Assicurati che tutti i pacchetti richiesti siano installati con pip
- **Problemi di percorso**: Esegui i notebook dalla loro directory contenente

### Applicazione Quiz

- **npm install fallisce**: Pulisci la cache npm: `npm cache clean --force`
- **Conflitti di porta**: Cambia porta con: `npm run serve -- --port 8081`
- **Errori di build**: Elimina `node_modules` e reinstalla: `rm -rf node_modules && npm install`

### Lezioni in R

- **Pacchetto non trovato**: Installa con: `install.packages("nome-pacchetto")`
- **Rendering RMarkdown**: Assicurati che il pacchetto rmarkdown sia installato
- **Problemi con il kernel**: Potrebbe essere necessario installare IRkernel per Jupyter

## Note Specifiche del Progetto

- Questo è principalmente un **curriculum di apprendimento**, non codice di produzione
- L'obiettivo è **comprendere i concetti di ML** attraverso la pratica
- Gli esempi di codice privilegiano **chiarezza rispetto all'ottimizzazione**
- La maggior parte delle lezioni è **autonoma** e può essere completata indipendentemente
- **Soluzioni fornite**, ma i partecipanti dovrebbero tentare gli esercizi prima
- Il repository utilizza **Docsify** per la documentazione web senza passaggi di build
- **Sketchnotes** forniscono riassunti visivi dei concetti
- Il supporto **multilingue** rende il contenuto accessibile globalmente

---

**Avvertenza**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa deve essere considerato la fonte autorevole. Per informazioni critiche, si consiglia una traduzione professionale eseguita da un traduttore umano. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.