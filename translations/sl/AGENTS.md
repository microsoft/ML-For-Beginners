<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:18:59+00:00",
  "source_file": "AGENTS.md",
  "language_code": "sl"
}
-->
# AGENTS.md

## Pregled projekta

To je **Strojno učenje za začetnike**, obsežen 12-tedenski, 26-lekcijski učni načrt, ki pokriva klasične koncepte strojnega učenja z uporabo Pythona (predvsem s Scikit-learn) in R. Repozitorij je zasnovan kot vir za samostojno učenje s praktičnimi projekti, kvizi in nalogami. Vsaka lekcija raziskuje koncepte strojnega učenja skozi podatke iz resničnega sveta, ki izvirajo iz različnih kultur in regij po svetu.

Ključne komponente:
- **Izobraževalna vsebina**: 26 lekcij, ki pokrivajo uvod v strojno učenje, regresijo, klasifikacijo, gručenje, obdelavo naravnega jezika (NLP), časovne vrste in okrepljeno učenje
- **Aplikacija za kvize**: Aplikacija, zasnovana v Vue.js, z ocenjevanji pred in po lekcijah
- **Podpora za več jezikov**: Samodejni prevodi v več kot 40 jezikov prek GitHub Actions
- **Dvojna jezikovna podpora**: Lekcije so na voljo tako v Pythonu (Jupyter beležnice) kot v R (R Markdown datoteke)
- **Učenje na osnovi projektov**: Vsaka tema vključuje praktične projekte in naloge

## Struktura repozitorija

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

Vsaka mapa lekcije običajno vsebuje:
- `README.md` - Glavna vsebina lekcije
- `notebook.ipynb` - Python Jupyter beležnica
- `solution/` - Rešitve kode (Python in R različice)
- `assignment.md` - Vaje za prakso
- `images/` - Vizualni viri

## Ukazi za nastavitev

### Za Python lekcije

Večina lekcij uporablja Jupyter beležnice. Namestite potrebne odvisnosti:

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

### Za R lekcije

R lekcije se nahajajo v mapah `solution/R/` kot `.rmd` ali `.ipynb` datoteke:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Za aplikacijo za kvize

Aplikacija za kvize je Vue.js aplikacija, ki se nahaja v mapi `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Za spletno mesto dokumentacije

Za zagon dokumentacije lokalno:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Potek razvoja

### Delo z beležnicami lekcij

1. Pojdite v mapo lekcije (npr. `2-Regression/1-Tools/`)
2. Odprite Jupyter beležnico:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Preučite vsebino lekcije in vaje
4. Po potrebi preverite rešitve v mapi `solution/`

### Razvoj v Pythonu

- Lekcije uporabljajo standardne knjižnice za podatkovno znanost v Pythonu
- Jupyter beležnice za interaktivno učenje
- Rešitve kode so na voljo v mapi `solution/` vsake lekcije

### Razvoj v R

- R lekcije so v formatu `.rmd` (R Markdown)
- Rešitve se nahajajo v podmapah `solution/R/`
- Za zagon R beležnic uporabite RStudio ali Jupyter z R jedrom

### Razvoj aplikacije za kvize

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

## Navodila za testiranje

### Testiranje aplikacije za kvize

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Opomba**: To je predvsem izobraževalni repozitorij. Za vsebino lekcij ni avtomatiziranih testov. Preverjanje poteka prek:
- Reševanja vaj v lekcijah
- Uspešnega zagona celic v beležnicah
- Preverjanja rezultatov glede na pričakovane rešitve

## Smernice za slog kode

### Python koda
- Upoštevajte smernice sloga PEP 8
- Uporabljajte jasna in opisna imena spremenljivk
- Dodajte komentarje za kompleksne operacije
- Jupyter beležnice naj vsebujejo markdown celice za razlago konceptov

### JavaScript/Vue.js (aplikacija za kvize)
- Upoštevajte smernice sloga Vue.js
- ESLint konfiguracija v `quiz-app/package.json`
- Za preverjanje in samodejno odpravljanje težav zaženite `npm run lint`

### Dokumentacija
- Markdown datoteke naj bodo jasne in dobro strukturirane
- Vključite primere kode v označenih blokih kode
- Za notranje reference uporabite relativne povezave
- Upoštevajte obstoječe konvencije oblikovanja

## Gradnja in uvajanje

### Uvajanje aplikacije za kvize

Aplikacijo za kvize lahko uvedete na Azure Static Web Apps:

1. **Predpogoji**:
   - Azure račun
   - GitHub repozitorij (že forkano)

2. **Uvedba na Azure**:
   - Ustvarite vir Azure Static Web App
   - Povežite se z GitHub repozitorijem
   - Nastavite lokacijo aplikacije: `/quiz-app`
   - Nastavite lokacijo izhoda: `dist`
   - Azure samodejno ustvari GitHub Actions potek dela

3. **GitHub Actions potek dela**:
   - Datoteka poteka dela ustvarjena v `.github/workflows/azure-static-web-apps-*.yml`
   - Samodejno gradi in uvaja ob potisku v glavno vejo

### PDF dokumentacija

Ustvarite PDF iz dokumentacije:

```bash
npm install
npm run convert
```

## Potek prevajanja

**Pomembno**: Prevajanja so avtomatizirana prek GitHub Actions z uporabo Co-op Translator.

- Prevajanja se samodejno ustvarijo, ko so spremembe potisnjene v vejo `main`
- **NE prevajajte vsebine ročno** - sistem to opravi samodejno
- Potek dela je definiran v `.github/workflows/co-op-translator.yml`
- Uporablja Azure AI/OpenAI storitve za prevajanje
- Podpira več kot 40 jezikov

## Smernice za prispevanje

### Za vsebinske prispevke

1. **Forkajte repozitorij** in ustvarite vejo za funkcijo
2. **Spremenite vsebino lekcije**, če dodajate/posodabljate lekcije
3. **Ne spreminjajte prevedenih datotek** - te so samodejno ustvarjene
4. **Preizkusite svojo kodo** - zagotovite, da vse celice v beležnicah uspešno tečejo
5. **Preverite povezave in slike**, da delujejo pravilno
6. **Oddajte pull request** z jasnim opisom

### Smernice za pull request

- **Format naslova**: `[Odsek] Kratek opis sprememb`
  - Primer: `[Regresija] Popravek tipkarske napake v lekciji 5`
  - Primer: `[Quiz-App] Posodobitev odvisnosti`
- **Pred oddajo**:
  - Zagotovite, da vse celice v beležnicah tečejo brez napak
  - Zaženite `npm run lint`, če spreminjate aplikacijo za kvize
  - Preverite oblikovanje markdowna
  - Preizkusite nove primere kode
- **PR mora vključevati**:
  - Opis sprememb
  - Razlog za spremembe
  - Posnetke zaslona, če gre za spremembe vmesnika
- **Kodeks ravnanja**: Upoštevajte [Microsoftov kodeks ravnanja za odprtokodno programsko opremo](CODE_OF_CONDUCT.md)
- **CLA**: Podpisati boste morali pogodbo o licenciranju prispevkov

## Struktura lekcije

Vsaka lekcija sledi doslednemu vzorcu:

1. **Kvizi pred predavanjem** - Preverjanje osnovnega znanja
2. **Vsebina lekcije** - Pisna navodila in razlage
3. **Demonstracije kode** - Praktični primeri v beležnicah
4. **Preverjanje znanja** - Preverjanje razumevanja skozi lekcijo
5. **Izziv** - Samostojna uporaba konceptov
6. **Naloga** - Razširjena praksa
7. **Kvizi po predavanju** - Ocena učnih rezultatov

## Referenca pogostih ukazov

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

## Dodatni viri

- **Microsoft Learn zbirka**: [Moduli za začetnike v strojnem učenju](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikacija za kvize**: [Spletni kvizi](https://ff-quizzes.netlify.app/en/ml/)
- **Plošča za razprave**: [GitHub razprave](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video vodiči**: [YouTube seznam predvajanja](https://aka.ms/ml-beginners-videos)

## Ključne tehnologije

- **Python**: Glavni jezik za lekcije strojnega učenja (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativna izvedba z uporabo tidyverse, tidymodels, caret
- **Jupyter**: Interaktivne beležnice za Python lekcije
- **R Markdown**: Dokumenti za R lekcije
- **Vue.js 3**: Okvir za aplikacijo za kvize
- **Flask**: Okvir za spletne aplikacije za uvajanje ML modelov
- **Docsify**: Generator spletne dokumentacije
- **GitHub Actions**: CI/CD in avtomatizirani prevodi

## Varnostni vidiki

- **Brez skrivnosti v kodi**: Nikoli ne vključujte API ključev ali poverilnic
- **Odvisnosti**: Posodabljajte npm in pip pakete
- **Uporabniški vnosi**: Primeri spletnih aplikacij Flask vključujejo osnovno preverjanje vnosa
- **Občutljivi podatki**: Primeri podatkovnih nizov so javni in neobčutljivi

## Odpravljanje težav

### Jupyter beležnice

- **Težave z jedrom**: Znova zaženite jedro, če se celice zataknejo: Kernel → Restart
- **Napake pri uvozu**: Prepričajte se, da so vse potrebne knjižnice nameščene s pip
- **Težave s potjo**: Beležnice zaženite iz njihove vsebovane mape

### Aplikacija za kvize

- **npm install ne uspe**: Počistite npm predpomnilnik: `npm cache clean --force`
- **Konflikti vrat**: Spremenite vrata z: `npm run serve -- --port 8081`
- **Napake pri gradnji**: Izbrišite `node_modules` in znova namestite: `rm -rf node_modules && npm install`

### R lekcije

- **Paket ni najden**: Namestite z: `install.packages("ime-paketa")`
- **Upodabljanje RMarkdown**: Prepričajte se, da je nameščen paket rmarkdown
- **Težave z jedrom**: Morda boste morali namestiti IRkernel za Jupyter

## Posebne opombe o projektu

- To je predvsem **učni načrt**, ne produkcijska koda
- Poudarek je na **razumevanju konceptov strojnega učenja** skozi praktično delo
- Primeri kode dajejo prednost **jasnosti pred optimizacijo**
- Večina lekcij je **samostojnih** in jih je mogoče dokončati neodvisno
- **Rešitve so na voljo**, vendar naj se učenci najprej poskusijo sami
- Repozitorij uporablja **Docsify** za spletno dokumentacijo brez koraka gradnje
- **Sketchnotes** zagotavljajo vizualne povzetke konceptov
- **Podpora za več jezikov** omogoča globalno dostopnost vsebine

---

**Izjava o omejitvi odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitne nesporazume ali napačne razlage, ki izhajajo iz uporabe tega prevoda.