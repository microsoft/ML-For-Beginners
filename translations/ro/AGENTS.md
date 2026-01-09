<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:16:59+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ro"
}
-->
# AGENTS.md

## Prezentare Generală a Proiectului

Acesta este **Machine Learning pentru Începători**, un curriculum cuprinzător de 12 săptămâni și 26 de lecții care acoperă concepte clasice de machine learning folosind Python (în principal cu Scikit-learn) și R. Repozitoriul este conceput ca o resursă de învățare în ritm propriu, cu proiecte practice, teste și teme. Fiecare lecție explorează concepte de ML prin date reale din diverse culturi și regiuni ale lumii.

Componente cheie:
- **Conținut Educațional**: 26 de lecții care acoperă introducerea în ML, regresia, clasificarea, clustering-ul, NLP, seriile temporale și învățarea prin întărire
- **Aplicație de Teste**: Aplicație de teste bazată pe Vue.js cu evaluări înainte și după lecții
- **Suport Multilingv**: Traduceri automate în peste 40 de limbi prin GitHub Actions
- **Suport Dual de Limbaj**: Lecții disponibile atât în Python (notebook-uri Jupyter), cât și în R (fișiere R Markdown)
- **Învățare Bazată pe Proiecte**: Fiecare subiect include proiecte practice și teme

## Structura Repozitoriului

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

Fiecare folder de lecție conține, de obicei:
- `README.md` - Conținutul principal al lecției
- `notebook.ipynb` - Notebook Jupyter pentru Python
- `solution/` - Codul soluției (versiuni Python și R)
- `assignment.md` - Exerciții practice
- `images/` - Resurse vizuale

## Comenzi de Configurare

### Pentru Lecțiile Python

Majoritatea lecțiilor folosesc notebook-uri Jupyter. Instalați dependențele necesare:

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

### Pentru Lecțiile R

Lecțiile R se află în folderele `solution/R/` ca fișiere `.rmd` sau `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Pentru Aplicația de Teste

Aplicația de teste este o aplicație Vue.js situată în directorul `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Pentru Site-ul de Documentație

Pentru a rula documentația local:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Flux de Lucru pentru Dezvoltare

### Lucrul cu Notebook-urile Lecțiilor

1. Navigați la directorul lecției (de exemplu, `2-Regression/1-Tools/`)
2. Deschideți notebook-ul Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Parcurgeți conținutul lecției și exercițiile
4. Verificați soluțiile în folderul `solution/` dacă este necesar

### Dezvoltare Python

- Lecțiile folosesc biblioteci standard de știința datelor în Python
- Notebook-uri Jupyter pentru învățare interactivă
- Codul soluției este disponibil în folderul `solution/` al fiecărei lecții

### Dezvoltare R

- Lecțiile R sunt în format `.rmd` (R Markdown)
- Soluțiile sunt situate în subdirectoarele `solution/R/`
- Folosiți RStudio sau Jupyter cu kernel-ul R pentru a rula notebook-uri R

### Dezvoltare Aplicație de Teste

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

## Instrucțiuni de Testare

### Testarea Aplicației de Teste

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Notă**: Acesta este în principal un repo de curriculum educațional. Nu există teste automate pentru conținutul lecțiilor. Validarea se face prin:
- Completarea exercițiilor lecțiilor
- Rularea celulelor notebook-urilor cu succes
- Verificarea rezultatelor față de cele așteptate în soluții

## Ghiduri de Stil pentru Cod

### Cod Python
- Respectați ghidurile de stil PEP 8
- Folosiți nume de variabile clare și descriptive
- Includeți comentarii pentru operațiuni complexe
- Notebook-urile Jupyter ar trebui să conțină celule markdown care explică conceptele

### JavaScript/Vue.js (Aplicația de Teste)
- Respectă ghidul de stil Vue.js
- Configurația ESLint în `quiz-app/package.json`
- Rulați `npm run lint` pentru a verifica și corecta automat problemele

### Documentație
- Fișierele markdown ar trebui să fie clare și bine structurate
- Includeți exemple de cod în blocuri de cod delimitate
- Folosiți linkuri relative pentru referințe interne
- Respectați convențiile existente de formatare

## Construire și Implementare

### Implementarea Aplicației de Teste

Aplicația de teste poate fi implementată pe Azure Static Web Apps:

1. **Prerechizite**:
   - Cont Azure
   - Repozitoriu GitHub (deja bifurcat)

2. **Implementare pe Azure**:
   - Creați o resursă Azure Static Web App
   - Conectați-vă la repozitoriul GitHub
   - Setați locația aplicației: `/quiz-app`
   - Setați locația output-ului: `dist`
   - Azure creează automat workflow-ul GitHub Actions

3. **Workflow GitHub Actions**:
   - Fișierul workflow este creat în `.github/workflows/azure-static-web-apps-*.yml`
   - Construiește și implementează automat la push pe branch-ul principal

### Documentație PDF

Generați PDF din documentație:

```bash
npm install
npm run convert
```

## Flux de Traducere

**Important**: Traducerile sunt automate prin GitHub Actions folosind Co-op Translator.

- Traducerile sunt generate automat când se fac modificări pe branch-ul `main`
- **NU traduceți manual conținutul** - sistemul se ocupă de acest lucru
- Workflow-ul este definit în `.github/workflows/co-op-translator.yml`
- Folosește servicii Azure AI/OpenAI pentru traducere
- Suportă peste 40 de limbi

## Ghiduri de Contribuire

### Pentru Contribuitorii de Conținut

1. **Bifurcați repozitoriul** și creați un branch de funcționalitate
2. **Faceți modificări la conținutul lecției** dacă adăugați/actualizați lecții
3. **Nu modificați fișierele traduse** - acestea sunt generate automat
4. **Testați codul** - asigurați-vă că toate celulele notebook-urilor rulează cu succes
5. **Verificați linkurile și imaginile** să funcționeze corect
6. **Trimiteți un pull request** cu o descriere clară

### Ghiduri pentru Pull Request

- **Format titlu**: `[Secțiune] Descriere scurtă a modificărilor`
  - Exemplu: `[Regression] Fix typo în lecția 5`
  - Exemplu: `[Quiz-App] Actualizare dependențe`
- **Înainte de trimitere**:
  - Asigurați-vă că toate celulele notebook-urilor se execută fără erori
  - Rulați `npm run lint` dacă modificați quiz-app
  - Verificați formatarea markdown
  - Testați orice exemple noi de cod
- **PR trebuie să includă**:
  - Descrierea modificărilor
  - Motivul modificărilor
  - Capturi de ecran dacă sunt modificări UI
- **Cod de Conduită**: Respectați [Codul de Conduită Open Source Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Va trebui să semnați Acordul de Licență pentru Contribuitori

## Structura Lecțiilor

Fiecare lecție urmează un model consistent:

1. **Test preliminar** - Testați cunoștințele de bază
2. **Conținutul lecției** - Instrucțiuni și explicații scrise
3. **Demonstrații de cod** - Exemple practice în notebook-uri
4. **Verificări de cunoștințe** - Verificați înțelegerea pe parcurs
5. **Provocare** - Aplicați conceptele independent
6. **Temă** - Practică extinsă
7. **Test final** - Evaluați rezultatele învățării

## Referință Comenzi Comune

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

## Resurse Suplimentare

- **Colecția Microsoft Learn**: [Module ML pentru Începători](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplicația de Teste**: [Teste online](https://ff-quizzes.netlify.app/en/ml/)
- **Forum de Discuții**: [Discuții GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Tutoriale Video**: [Playlist YouTube](https://aka.ms/ml-beginners-videos)

## Tehnologii Cheie

- **Python**: Limbaj principal pentru lecțiile ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Implementare alternativă folosind tidyverse, tidymodels, caret
- **Jupyter**: Notebook-uri interactive pentru lecțiile Python
- **R Markdown**: Documente pentru lecțiile R
- **Vue.js 3**: Framework pentru aplicația de teste
- **Flask**: Framework pentru aplicații web pentru implementarea modelelor ML
- **Docsify**: Generator de site-uri de documentație
- **GitHub Actions**: CI/CD și traduceri automate

## Considerații de Securitate

- **Fără secrete în cod**: Nu comiteți niciodată chei API sau credențiale
- **Dependențe**: Mențineți pachetele npm și pip actualizate
- **Input utilizator**: Exemplele de aplicații web Flask includ validare de bază a input-ului
- **Date sensibile**: Seturile de date exemplu sunt publice și non-sensibile

## Depanare

### Notebook-uri Jupyter

- **Probleme kernel**: Reporniți kernel-ul dacă celulele se blochează: Kernel → Restart
- **Erori de import**: Asigurați-vă că toate pachetele necesare sunt instalate cu pip
- **Probleme de cale**: Rulați notebook-urile din directorul lor conținător

### Aplicația de Teste

- **npm install eșuează**: Goliți cache-ul npm: `npm cache clean --force`
- **Conflicte de port**: Schimbați portul cu: `npm run serve -- --port 8081`
- **Erori de build**: Ștergeți `node_modules` și reinstalați: `rm -rf node_modules && npm install`

### Lecții R

- **Pachetul nu este găsit**: Instalați cu: `install.packages("nume-pachet")`
- **Redarea RMarkdown**: Asigurați-vă că pachetul rmarkdown este instalat
- **Probleme kernel**: Poate fi necesar să instalați IRkernel pentru Jupyter

## Note Specifice Proiectului

- Acesta este în principal un **curriculum de învățare**, nu cod de producție
- Accentul este pe **înțelegerea conceptelor ML** prin practică
- Exemplele de cod prioritizează **claritatea în detrimentul optimizării**
- Majoritatea lecțiilor sunt **autonome** și pot fi completate independent
- **Soluțiile sunt furnizate**, dar cursanții ar trebui să încerce mai întâi exercițiile
- Repozitoriul folosește **Docsify** pentru documentație web fără pas de construire
- **Sketchnotes** oferă rezumate vizuale ale conceptelor
- **Suportul multilingv** face conținutul accesibil la nivel global

---

**Declinarea responsabilității**:  
Acest document a fost tradus utilizând serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși depunem eforturi pentru a asigura acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa nativă trebuie considerat sursa autoritară. Pentru informații critice, se recomandă traducerea realizată de un profesionist uman. Nu ne asumăm răspunderea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.