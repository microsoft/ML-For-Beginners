<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:16:25+00:00",
  "source_file": "AGENTS.md",
  "language_code": "sk"
}
-->
# AGENTS.md

## Prehľad projektu

Toto je **Strojové učenie pre začiatočníkov**, komplexný 12-týždňový, 26-lekciový kurz pokrývajúci klasické koncepty strojového učenia pomocou Pythonu (primárne so Scikit-learn) a R. Repozitár je navrhnutý ako zdroj na samoštúdium s praktickými projektmi, kvízmi a úlohami. Každá lekcia skúma koncepty strojového učenia prostredníctvom reálnych dát z rôznych kultúr a regiónov sveta.

Kľúčové komponenty:
- **Vzdelávací obsah**: 26 lekcií pokrývajúcich úvod do strojového učenia, regresiu, klasifikáciu, zhlukovanie, NLP, časové rady a posilňovacie učenie
- **Aplikácia na kvízy**: Kvízová aplikácia založená na Vue.js s hodnoteniami pred a po lekciách
- **Podpora viacerých jazykov**: Automatizované preklady do viac ako 40 jazykov prostredníctvom GitHub Actions
- **Podpora dvoch jazykov**: Lekcie dostupné v Python (Jupyter notebooky) aj R (R Markdown súbory)
- **Učenie založené na projektoch**: Každá téma obsahuje praktické projekty a úlohy

## Štruktúra repozitára

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

Každý priečinok lekcie zvyčajne obsahuje:
- `README.md` - Hlavný obsah lekcie
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - Riešenie kódu (verzie pre Python a R)
- `assignment.md` - Cvičenia na precvičenie
- `images/` - Vizualizačné zdroje

## Príkazy na nastavenie

### Pre lekcie v Pythone

Väčšina lekcií používa Jupyter notebooky. Nainštalujte potrebné závislosti:

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

### Pre lekcie v R

Lekcie v R sa nachádzajú v priečinkoch `solution/R/` ako `.rmd` alebo `.ipynb` súbory:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Pre aplikáciu na kvízy

Kvízová aplikácia je Vue.js aplikácia umiestnená v priečinku `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Pre dokumentačný web

Na spustenie dokumentácie lokálne:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Pracovný postup vývoja

### Práca s notebookmi lekcií

1. Prejdite do priečinka lekcie (napr. `2-Regression/1-Tools/`)
2. Otvorte Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Prejdite obsah lekcie a cvičenia
4. Skontrolujte riešenia v priečinku `solution/`, ak je to potrebné

### Vývoj v Pythone

- Lekcie používajú štandardné knižnice pre dátovú vedu v Pythone
- Jupyter notebooky na interaktívne učenie
- Riešenia kódu sú dostupné v priečinku `solution/` každej lekcie

### Vývoj v R

- Lekcie v R sú vo formáte `.rmd` (R Markdown)
- Riešenia sa nachádzajú v podpriečinkoch `solution/R/`
- Použite RStudio alebo Jupyter s R kernelom na spustenie notebookov v R

### Vývoj aplikácie na kvízy

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

## Pokyny na testovanie

### Testovanie aplikácie na kvízy

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Poznámka**: Toto je primárne vzdelávací repozitár. Neexistujú automatizované testy pre obsah lekcií. Validácia sa vykonáva prostredníctvom:
- Dokončenia cvičení lekcií
- Úspešného spustenia buniek notebookov
- Porovnania výstupu s očakávanými výsledkami v riešeniach

## Pokyny pre štýl kódu

### Python kód
- Dodržiavajte štýlové pokyny PEP 8
- Používajte jasné, popisné názvy premenných
- Pridávajte komentáre k zložitým operáciám
- Jupyter notebooky by mali obsahovať markdown bunky vysvetľujúce koncepty

### JavaScript/Vue.js (aplikácia na kvízy)
- Dodržiavajte štýlové pokyny Vue.js
- Konfigurácia ESLint v `quiz-app/package.json`
- Spustite `npm run lint` na kontrolu a automatické opravy problémov

### Dokumentácia
- Markdown súbory by mali byť jasné a dobre štruktúrované
- Zahrňte príklady kódu v ohradených blokoch kódu
- Používajte relatívne odkazy na interné referencie
- Dodržiavajte existujúce formátovacie konvencie

## Build a nasadenie

### Nasadenie aplikácie na kvízy

Aplikácia na kvízy môže byť nasadená na Azure Static Web Apps:

1. **Predpoklady**:
   - Azure účet
   - GitHub repozitár (už forkovaný)

2. **Nasadenie na Azure**:
   - Vytvorte zdroj Azure Static Web App
   - Pripojte sa k GitHub repozitáru
   - Nastavte umiestnenie aplikácie: `/quiz-app`
   - Nastavte umiestnenie výstupu: `dist`
   - Azure automaticky vytvorí GitHub Actions workflow

3. **GitHub Actions workflow**:
   - Workflow súbor vytvorený v `.github/workflows/azure-static-web-apps-*.yml`
   - Automaticky sa builduje a nasadzuje pri pushnutí do hlavnej vetvy

### Dokumentácia PDF

Generujte PDF z dokumentácie:

```bash
npm install
npm run convert
```

## Pracovný postup prekladu

**Dôležité**: Preklady sú automatizované prostredníctvom GitHub Actions pomocou Co-op Translator.

- Preklady sa generujú automaticky pri zmenách pushnutých do vetvy `main`
- **NEPREKLADAJTE obsah manuálne** - systém to spracuje
- Workflow definovaný v `.github/workflows/co-op-translator.yml`
- Používa služby Azure AI/OpenAI na preklad
- Podporuje viac ako 40 jazykov

## Pokyny pre prispievateľov

### Pre prispievateľov obsahu

1. **Forknite repozitár** a vytvorte vetvu pre funkciu
2. **Upravte obsah lekcie**, ak pridávate alebo aktualizujete lekcie
3. **Nemeňte preložené súbory** - sú generované automaticky
4. **Otestujte svoj kód** - uistite sa, že všetky bunky notebookov sa úspešne spustia
5. **Overte odkazy a obrázky**, či fungujú správne
6. **Odošlite pull request** s jasným popisom

### Pokyny pre pull requesty

- **Formát názvu**: `[Sekcia] Stručný popis zmien`
  - Príklad: `[Regresia] Oprava preklepu v lekcii 5`
  - Príklad: `[Quiz-App] Aktualizácia závislostí`
- **Pred odoslaním**:
  - Uistite sa, že všetky bunky notebookov sa spustia bez chýb
  - Spustite `npm run lint`, ak upravujete quiz-app
  - Overte formátovanie markdownu
  - Otestujte akékoľvek nové príklady kódu
- **PR musí obsahovať**:
  - Popis zmien
  - Dôvod zmien
  - Screenshoty, ak ide o zmeny v UI
- **Kódex správania**: Dodržiavajte [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Budete musieť podpísať Contributor License Agreement

## Štruktúra lekcie

Každá lekcia nasleduje konzistentný vzor:

1. **Kvíz pred prednáškou** - Test základných znalostí
2. **Obsah lekcie** - Písomné pokyny a vysvetlenia
3. **Ukážky kódu** - Praktické príklady v notebookoch
4. **Kontroly znalostí** - Overenie porozumenia počas lekcie
5. **Výzva** - Samostatné aplikovanie konceptov
6. **Úloha** - Rozšírené precvičenie
7. **Kvíz po prednáške** - Hodnotenie výsledkov učenia

## Referencia bežných príkazov

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

## Dodatočné zdroje

- **Microsoft Learn Collection**: [Moduly ML pre začiatočníkov](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikácia na kvízy**: [Online kvízy](https://ff-quizzes.netlify.app/en/ml/)
- **Diskusné fórum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video prehliadky**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Kľúčové technológie

- **Python**: Primárny jazyk pre lekcie strojového učenia (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatívna implementácia pomocou tidyverse, tidymodels, caret
- **Jupyter**: Interaktívne notebooky pre lekcie v Pythone
- **R Markdown**: Dokumenty pre lekcie v R
- **Vue.js 3**: Framework pre aplikáciu na kvízy
- **Flask**: Framework pre webové aplikácie na nasadenie modelov strojového učenia
- **Docsify**: Generátor dokumentačného webu
- **GitHub Actions**: CI/CD a automatizované preklady

## Bezpečnostné úvahy

- **Žiadne tajné údaje v kóde**: Nikdy nekomitujte API kľúče alebo prihlasovacie údaje
- **Závislosti**: Udržujte npm a pip balíčky aktualizované
- **Vstupy používateľov**: Príklady webových aplikácií Flask obsahujú základnú validáciu vstupov
- **Citlivé údaje**: Príkladové datasety sú verejné a neobsahujú citlivé údaje

## Riešenie problémov

### Jupyter notebooky

- **Problémy s kernelom**: Reštartujte kernel, ak bunky zamrznú: Kernel → Restart
- **Import chyby**: Uistite sa, že všetky potrebné balíčky sú nainštalované pomocou pip
- **Problémy s cestami**: Spúšťajte notebooky z ich obsahujúceho priečinka

### Aplikácia na kvízy

- **npm install zlyhá**: Vyčistite npm cache: `npm cache clean --force`
- **Konflikty portov**: Zmeňte port pomocou: `npm run serve -- --port 8081`
- **Chyby buildu**: Odstráňte `node_modules` a znovu nainštalujte: `rm -rf node_modules && npm install`

### Lekcie v R

- **Balíček nenájdený**: Nainštalujte pomocou: `install.packages("package-name")`
- **Renderovanie RMarkdown**: Uistite sa, že balíček rmarkdown je nainštalovaný
- **Problémy s kernelom**: Možno bude potrebné nainštalovať IRkernel pre Jupyter

## Poznámky k projektu

- Toto je primárne **vzdelávací kurz**, nie produkčný kód
- Zameranie je na **pochopenie konceptov strojového učenia** prostredníctvom praktického precvičenia
- Príklady kódu uprednostňujú **jasnosť pred optimalizáciou**
- Väčšina lekcií je **samostatná** a môže byť dokončená nezávisle
- **Riešenia sú poskytované**, ale študenti by mali najskôr skúsiť cvičenia
- Repozitár používa **Docsify** na webovú dokumentáciu bez kroku buildovania
- **Sketchnotes** poskytujú vizuálne zhrnutia konceptov
- **Podpora viacerých jazykov** robí obsah globálne prístupným

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, upozorňujeme, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.