<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:15:53+00:00",
  "source_file": "AGENTS.md",
  "language_code": "cs"
}
-->
# AGENTS.md

## Přehled projektu

Toto je **Strojové učení pro začátečníky**, komplexní 12týdenní, 26lekční kurikulum pokrývající klasické koncepty strojového učení pomocí Pythonu (primárně s knihovnou Scikit-learn) a R. Repozitář je navržen jako zdroj pro samostatné studium s praktickými projekty, kvízy a úkoly. Každá lekce zkoumá koncepty strojového učení prostřednictvím reálných dat z různých kultur a regionů po celém světě.

Klíčové komponenty:
- **Vzdělávací obsah**: 26 lekcí pokrývajících úvod do strojového učení, regresi, klasifikaci, shlukování, NLP, časové řady a posilované učení
- **Aplikace kvízů**: Kvízová aplikace založená na Vue.js s hodnocením před a po lekci
- **Podpora více jazyků**: Automatické překlady do více než 40 jazyků prostřednictvím GitHub Actions
- **Podpora dvou jazyků**: Lekce dostupné v Pythonu (Jupyter notebooky) i R (R Markdown soubory)
- **Učení založené na projektech**: Každé téma zahrnuje praktické projekty a úkoly

## Struktura repozitáře

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

Každá složka lekce obvykle obsahuje:
- `README.md` - Hlavní obsah lekce
- `notebook.ipynb` - Jupyter notebook pro Python
- `solution/` - Řešení kódu (verze pro Python a R)
- `assignment.md` - Cvičné úkoly
- `images/` - Vizuální zdroje

## Příkazy pro nastavení

### Pro lekce v Pythonu

Většina lekcí používá Jupyter notebooky. Nainstalujte potřebné závislosti:

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

### Pro lekce v R

Lekce v R se nacházejí ve složkách `solution/R/` jako `.rmd` nebo `.ipynb` soubory:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Pro aplikaci kvízů

Aplikace kvízů je Vue.js aplikace umístěná ve složce `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Pro dokumentační web

Pro spuštění dokumentace lokálně:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Pracovní postup vývoje

### Práce s notebooky lekcí

1. Přejděte do složky lekce (např. `2-Regression/1-Tools/`)
2. Otevřete Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Projděte obsah lekce a cvičení
4. Pokud je to nutné, zkontrolujte řešení ve složce `solution/`

### Vývoj v Pythonu

- Lekce používají standardní knihovny pro datovou vědu v Pythonu
- Jupyter notebooky pro interaktivní učení
- Řešení kódu dostupné ve složce `solution/` každé lekce

### Vývoj v R

- Lekce v R jsou ve formátu `.rmd` (R Markdown)
- Řešení se nachází v podadresářích `solution/R/`
- Použijte RStudio nebo Jupyter s R jádrem pro spuštění notebooků v R

### Vývoj aplikace kvízů

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

## Pokyny k testování

### Testování aplikace kvízů

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Poznámka**: Toto je primárně repozitář vzdělávacího kurikula. Neexistují žádné automatizované testy pro obsah lekcí. Validace se provádí prostřednictvím:
- Dokončení cvičení lekcí
- Úspěšného spuštění buněk notebooku
- Kontroly výstupu oproti očekávaným výsledkům v řešeních

## Pokyny ke stylu kódu

### Kód v Pythonu
- Dodržujte stylové pokyny PEP 8
- Používejte jasné, popisné názvy proměnných
- Přidávejte komentáře ke složitým operacím
- Jupyter notebooky by měly obsahovat markdown buňky vysvětlující koncepty

### JavaScript/Vue.js (aplikace kvízů)
- Dodržujte stylový průvodce Vue.js
- Konfigurace ESLint v `quiz-app/package.json`
- Spusťte `npm run lint` pro kontrolu a automatické opravy problémů

### Dokumentace
- Markdown soubory by měly být jasné a dobře strukturované
- Zahrnujte příklady kódu v ohraničených blocích kódu
- Používejte relativní odkazy pro interní reference
- Dodržujte existující formátovací konvence

## Sestavení a nasazení

### Nasazení aplikace kvízů

Aplikace kvízů může být nasazena na Azure Static Web Apps:

1. **Předpoklady**:
   - Účet Azure
   - GitHub repozitář (již forkovaný)

2. **Nasazení na Azure**:
   - Vytvořte zdroj Azure Static Web App
   - Připojte se k GitHub repozitáři
   - Nastavte umístění aplikace: `/quiz-app`
   - Nastavte umístění výstupu: `dist`
   - Azure automaticky vytvoří workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - Workflow soubor vytvořen v `.github/workflows/azure-static-web-apps-*.yml`
   - Automaticky sestavuje a nasazuje při push na hlavní větev

### Dokumentace PDF

Generování PDF z dokumentace:

```bash
npm install
npm run convert
```

## Pracovní postup překladu

**Důležité**: Překlady jsou automatizované prostřednictvím GitHub Actions pomocí Co-op Translator.

- Překlady jsou automaticky generovány při změnách pushnutých do větve `main`
- **NEPŘEKLÁDEJTE obsah ručně** - systém to zajišťuje
- Workflow definováno v `.github/workflows/co-op-translator.yml`
- Používá služby Azure AI/OpenAI pro překlad
- Podporuje více než 40 jazyků

## Pokyny pro přispěvatele

### Pro přispěvatele obsahu

1. **Forkněte repozitář** a vytvořte větev pro funkci
2. **Proveďte změny v obsahu lekce**, pokud přidáváte/aktualizujete lekce
3. **Neměňte přeložené soubory** - jsou generovány automaticky
4. **Otestujte svůj kód** - ujistěte se, že všechny buňky notebooku běží úspěšně
5. **Ověřte odkazy a obrázky**, zda fungují správně
6. **Odešlete pull request** s jasným popisem

### Pokyny pro pull requesty

- **Formát názvu**: `[Sekce] Stručný popis změn`
  - Příklad: `[Regrese] Oprava překlepu v lekci 5`
  - Příklad: `[Quiz-App] Aktualizace závislostí`
- **Před odesláním**:
  - Ujistěte se, že všechny buňky notebooku se spustí bez chyb
  - Spusťte `npm run lint`, pokud upravujete quiz-app
  - Ověřte formátování markdownu
  - Otestujte nové příklady kódu
- **PR musí obsahovat**:
  - Popis změn
  - Důvod změn
  - Screenshoty, pokud se jedná o změny UI
- **Kodex chování**: Dodržujte [Kodex chování Microsoft Open Source](CODE_OF_CONDUCT.md)
- **CLA**: Budete muset podepsat Smlouvu o licenci přispěvatele

## Struktura lekce

Každá lekce dodržuje konzistentní vzor:

1. **Kvíz před lekcí** - Test základních znalostí
2. **Obsah lekce** - Písemné instrukce a vysvětlení
3. **Ukázky kódu** - Praktické příklady v notebooku
4. **Kontroly znalostí** - Ověření porozumění během lekce
5. **Výzva** - Samostatné použití konceptů
6. **Úkol** - Rozšířená praxe
7. **Kvíz po lekci** - Hodnocení výsledků učení

## Referenční příkazy

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

## Další zdroje

- **Microsoft Learn Collection**: [Moduly ML pro začátečníky](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikace kvízů**: [Online kvízy](https://ff-quizzes.netlify.app/en/ml/)
- **Diskusní fórum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video průvodce**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Klíčové technologie

- **Python**: Primární jazyk pro lekce strojového učení (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativní implementace pomocí tidyverse, tidymodels, caret
- **Jupyter**: Interaktivní notebooky pro lekce v Pythonu
- **R Markdown**: Dokumenty pro lekce v R
- **Vue.js 3**: Framework pro aplikaci kvízů
- **Flask**: Framework pro webové aplikace pro nasazení modelů strojového učení
- **Docsify**: Generátor dokumentačního webu
- **GitHub Actions**: CI/CD a automatizované překlady

## Bezpečnostní úvahy

- **Žádná tajemství v kódu**: Nikdy neukládejte API klíče nebo přihlašovací údaje
- **Závislosti**: Udržujte balíčky npm a pip aktuální
- **Vstup uživatele**: Příklady webových aplikací Flask zahrnují základní validaci vstupu
- **Citlivá data**: Příkladové datové sady jsou veřejné a neobsahují citlivé informace

## Řešení problémů

### Jupyter notebooky

- **Problémy s jádrem**: Restartujte jádro, pokud buňky zamrznou: Kernel → Restart
- **Chyby při importu**: Ujistěte se, že všechny požadované balíčky jsou nainstalovány pomocí pip
- **Problémy s cestou**: Spouštějte notebooky z jejich obsahujícího adresáře

### Aplikace kvízů

- **npm install selže**: Vyčistěte cache npm: `npm cache clean --force`
- **Konflikty portů**: Změňte port pomocí: `npm run serve -- --port 8081`
- **Chyby při sestavení**: Smažte `node_modules` a znovu nainstalujte: `rm -rf node_modules && npm install`

### Lekce v R

- **Balíček nenalezen**: Nainstalujte pomocí: `install.packages("package-name")`
- **Renderování RMarkdown**: Ujistěte se, že balíček rmarkdown je nainstalován
- **Problémy s jádrem**: Možná bude nutné nainstalovat IRkernel pro Jupyter

## Poznámky k projektu

- Toto je primárně **vzdělávací kurikulum**, nikoli produkční kód
- Důraz je kladen na **pochopení konceptů strojového učení** prostřednictvím praktického cvičení
- Příklady kódu upřednostňují **srozumitelnost před optimalizací**
- Většina lekcí je **samostatná** a lze je dokončit nezávisle
- **Řešení jsou poskytována**, ale studenti by měli nejprve zkusit cvičení
- Repozitář používá **Docsify** pro webovou dokumentaci bez kroku sestavení
- **Sketchnotes** poskytují vizuální shrnutí konceptů
- **Podpora více jazyků** zpřístupňuje obsah globálně

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby AI pro překlady [Co-op Translator](https://github.com/Azure/co-op-translator). I když se snažíme o přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.