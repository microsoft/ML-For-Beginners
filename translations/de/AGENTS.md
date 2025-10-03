<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:57:35+00:00",
  "source_file": "AGENTS.md",
  "language_code": "de"
}
-->
# AGENTS.md

## Projektübersicht

Dies ist **Maschinelles Lernen für Anfänger**, ein umfassender 12-wöchiger Lehrplan mit 26 Lektionen, der klassische Konzepte des maschinellen Lernens mit Python (hauptsächlich mit Scikit-learn) und R behandelt. Das Repository ist als selbstgesteuertes Lernressource mit praktischen Projekten, Quizfragen und Aufgaben konzipiert. Jede Lektion untersucht ML-Konzepte anhand von realen Daten aus verschiedenen Kulturen und Regionen weltweit.

Wichtige Bestandteile:
- **Bildungsinhalte**: 26 Lektionen zu Einführung in ML, Regression, Klassifikation, Clustering, NLP, Zeitreihen und Verstärkungslernen
- **Quiz-Anwendung**: Quiz-App auf Basis von Vue.js mit Vor- und Nach-Lektionsbewertungen
- **Mehrsprachige Unterstützung**: Automatische Übersetzungen in über 40 Sprachen via GitHub Actions
- **Duale Sprachunterstützung**: Lektionen verfügbar in Python (Jupyter-Notebooks) und R (R Markdown-Dateien)
- **Projektbasiertes Lernen**: Jedes Thema enthält praktische Projekte und Aufgaben

## Repository-Struktur

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

Jeder Lektionen-Ordner enthält typischerweise:
- `README.md` - Hauptinhalt der Lektion
- `notebook.ipynb` - Python Jupyter-Notebook
- `solution/` - Lösungscode (Python- und R-Versionen)
- `assignment.md` - Übungsaufgaben
- `images/` - Visuelle Ressourcen

## Setup-Befehle

### Für Python-Lektionen

Die meisten Lektionen verwenden Jupyter-Notebooks. Installieren Sie die erforderlichen Abhängigkeiten:

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

### Für R-Lektionen

R-Lektionen befinden sich in den `solution/R/`-Ordnern als `.rmd`- oder `.ipynb`-Dateien:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Für die Quiz-Anwendung

Die Quiz-App ist eine Vue.js-Anwendung im Verzeichnis `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Für die Dokumentationsseite

Um die Dokumentation lokal auszuführen:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Entwicklungsworkflow

### Arbeiten mit Lektionen-Notebooks

1. Navigieren Sie zum Lektionen-Verzeichnis (z. B. `2-Regression/1-Tools/`)
2. Öffnen Sie das Jupyter-Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Arbeiten Sie die Lektioneninhalte und Übungen durch
4. Überprüfen Sie die Lösungen im `solution/`-Ordner bei Bedarf

### Python-Entwicklung

- Lektionen verwenden Standard-Bibliotheken für Datenwissenschaft in Python
- Jupyter-Notebooks für interaktives Lernen
- Lösungscode ist in jedem Lektionen-Ordner im `solution/`-Ordner verfügbar

### R-Entwicklung

- R-Lektionen sind im `.rmd`-Format (R Markdown)
- Lösungen befinden sich in den `solution/R/`-Unterverzeichnissen
- Verwenden Sie RStudio oder Jupyter mit R-Kernel, um R-Notebooks auszuführen

### Entwicklung der Quiz-Anwendung

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

## Testanweisungen

### Testen der Quiz-Anwendung

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Hinweis**: Dies ist hauptsächlich ein Bildungs-Repository. Es gibt keine automatisierten Tests für die Lektioneninhalte. Die Validierung erfolgt durch:
- Abschluss der Lektionenübungen
- Erfolgreiches Ausführen der Notebook-Zellen
- Überprüfung der Ausgabe mit den erwarteten Ergebnissen in den Lösungen

## Richtlinien für Code-Stil

### Python-Code
- Befolgen Sie die PEP 8-Stilrichtlinien
- Verwenden Sie klare, beschreibende Variablennamen
- Fügen Sie Kommentare für komplexe Operationen hinzu
- Jupyter-Notebooks sollten Markdown-Zellen enthalten, die Konzepte erklären

### JavaScript/Vue.js (Quiz-App)
- Befolgt die Vue.js-Stilrichtlinien
- ESLint-Konfiguration in `quiz-app/package.json`
- Führen Sie `npm run lint` aus, um Probleme zu überprüfen und automatisch zu beheben

### Dokumentation
- Markdown-Dateien sollten klar und gut strukturiert sein
- Codebeispiele in umschlossenen Codeblöcken einfügen
- Relative Links für interne Verweise verwenden
- Bestehende Formatierungskonventionen befolgen

## Build und Deployment

### Deployment der Quiz-Anwendung

Die Quiz-App kann auf Azure Static Web Apps bereitgestellt werden:

1. **Voraussetzungen**:
   - Azure-Konto
   - GitHub-Repository (bereits geforkt)

2. **Bereitstellung auf Azure**:
   - Erstellen Sie eine Azure Static Web App-Ressource
   - Verbinden Sie das GitHub-Repository
   - Legen Sie den App-Standort fest: `/quiz-app`
   - Legen Sie den Ausgabeort fest: `dist`
   - Azure erstellt automatisch einen GitHub Actions-Workflow

3. **GitHub Actions-Workflow**:
   - Workflow-Datei wird unter `.github/workflows/azure-static-web-apps-*.yml` erstellt
   - Automatischer Build und Deployment bei Push auf den Hauptbranch

### Dokumentation als PDF

Generieren Sie ein PDF aus der Dokumentation:

```bash
npm install
npm run convert
```

## Übersetzungsworkflow

**Wichtig**: Übersetzungen werden automatisiert über GitHub Actions mit Co-op Translator durchgeführt.

- Übersetzungen werden automatisch generiert, wenn Änderungen in den `main`-Branch gepusht werden
- **NICHT manuell Inhalte übersetzen** - das System übernimmt dies
- Workflow definiert in `.github/workflows/co-op-translator.yml`
- Verwendet Azure AI/OpenAI-Dienste für Übersetzungen
- Unterstützt über 40 Sprachen

## Richtlinien für Beiträge

### Für Inhaltsbeiträge

1. **Forken Sie das Repository** und erstellen Sie einen Feature-Branch
2. **Ändern Sie die Lektioneninhalte**, wenn Sie Lektionen hinzufügen/aktualisieren
3. **Ändern Sie keine übersetzten Dateien** - diese werden automatisch generiert
4. **Testen Sie Ihren Code** - stellen Sie sicher, dass alle Notebook-Zellen erfolgreich ausgeführt werden
5. **Überprüfen Sie Links und Bilder**, ob sie korrekt funktionieren
6. **Reichen Sie eine Pull-Anfrage ein** mit einer klaren Beschreibung

### Richtlinien für Pull-Anfragen

- **Titel-Format**: `[Abschnitt] Kurze Beschreibung der Änderungen`
  - Beispiel: `[Regression] Tippfehler in Lektion 5 korrigiert`
  - Beispiel: `[Quiz-App] Abhängigkeiten aktualisiert`
- **Vor dem Einreichen**:
  - Stellen Sie sicher, dass alle Notebook-Zellen fehlerfrei ausgeführt werden
  - Führen Sie `npm run lint` aus, wenn Sie die Quiz-App ändern
  - Überprüfen Sie die Markdown-Formatierung
  - Testen Sie alle neuen Codebeispiele
- **PR muss enthalten**:
  - Beschreibung der Änderungen
  - Grund für die Änderungen
  - Screenshots bei UI-Änderungen
- **Verhaltenskodex**: Befolgen Sie den [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Sie müssen die Contributor License Agreement unterzeichnen

## Lektionenstruktur

Jede Lektion folgt einem konsistenten Muster:

1. **Quiz vor der Vorlesung** - Testen Sie das Basiswissen
2. **Lektioneninhalt** - Schriftliche Anweisungen und Erklärungen
3. **Code-Demonstrationen** - Praktische Beispiele in Notebooks
4. **Wissensüberprüfungen** - Überprüfen Sie das Verständnis während der Lektion
5. **Herausforderung** - Wenden Sie Konzepte eigenständig an
6. **Aufgabe** - Erweiterte Übung
7. **Quiz nach der Vorlesung** - Bewerten Sie die Lernergebnisse

## Referenz für häufige Befehle

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

## Zusätzliche Ressourcen

- **Microsoft Learn Collection**: [ML für Anfänger-Module](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz-App**: [Online-Quiz](https://ff-quizzes.netlify.app/en/ml/)
- **Diskussionsforum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videoanleitungen**: [YouTube-Playlist](https://aka.ms/ml-beginners-videos)

## Schlüsseltechnologien

- **Python**: Hauptsprache für ML-Lektionen (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternative Implementierung mit tidyverse, tidymodels, caret
- **Jupyter**: Interaktive Notebooks für Python-Lektionen
- **R Markdown**: Dokumente für R-Lektionen
- **Vue.js 3**: Framework für die Quiz-Anwendung
- **Flask**: Webanwendungs-Framework für ML-Modellbereitstellung
- **Docsify**: Generator für Dokumentationsseiten
- **GitHub Actions**: CI/CD und automatisierte Übersetzungen

## Sicherheitsüberlegungen

- **Keine Geheimnisse im Code**: Niemals API-Schlüssel oder Zugangsdaten einfügen
- **Abhängigkeiten**: Halten Sie npm- und pip-Pakete aktuell
- **Benutzereingaben**: Flask-Web-App-Beispiele enthalten grundlegende Eingabevalidierung
- **Sensible Daten**: Beispieldatensätze sind öffentlich und nicht sensibel

## Fehlerbehebung

### Jupyter-Notebooks

- **Kernel-Probleme**: Starten Sie den Kernel neu, wenn Zellen hängen: Kernel → Neustart
- **Importfehler**: Stellen Sie sicher, dass alle erforderlichen Pakete mit pip installiert sind
- **Pfadprobleme**: Führen Sie Notebooks aus ihrem enthaltenen Verzeichnis aus

### Quiz-Anwendung

- **npm install schlägt fehl**: Löschen Sie den npm-Cache: `npm cache clean --force`
- **Portkonflikte**: Ändern Sie den Port mit: `npm run serve -- --port 8081`
- **Build-Fehler**: Löschen Sie `node_modules` und installieren Sie neu: `rm -rf node_modules && npm install`

### R-Lektionen

- **Paket nicht gefunden**: Installieren Sie es mit: `install.packages("package-name")`
- **RMarkdown-Rendering**: Stellen Sie sicher, dass das rmarkdown-Paket installiert ist
- **Kernel-Probleme**: Möglicherweise müssen Sie IRkernel für Jupyter installieren

## Projektspezifische Hinweise

- Dies ist hauptsächlich ein **Lernlehrplan**, kein Produktionscode
- Der Fokus liegt auf dem **Verständnis von ML-Konzepten** durch praktische Übungen
- Codebeispiele priorisieren **Klarheit vor Optimierung**
- Die meisten Lektionen sind **eigenständig** und können unabhängig abgeschlossen werden
- **Lösungen sind verfügbar**, aber Lernende sollten zuerst die Übungen versuchen
- Das Repository verwendet **Docsify** für die Web-Dokumentation ohne Build-Schritt
- **Sketchnotes** bieten visuelle Zusammenfassungen von Konzepten
- **Mehrsprachige Unterstützung** macht Inhalte weltweit zugänglich

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.