<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:15:18+00:00",
  "source_file": "AGENTS.md",
  "language_code": "hu"
}
-->
# AGENTS.md

## Projektáttekintés

Ez a **Gépitanulás kezdőknek** egy átfogó, 12 hetes, 26 leckéből álló tananyag, amely a klasszikus gépitanulási fogalmakat mutatja be Python (elsősorban Scikit-learn) és R használatával. A repozitórium önálló tanulási forrásként készült, gyakorlati projektekkel, kvízekkel és feladatokkal. Minden lecke valós adatokat használ különböző kultúrákból és régiókból világszerte, hogy bemutassa az ML fogalmakat.

Főbb elemek:
- **Oktatási tartalom**: 26 lecke, amelyek az ML bevezetését, regressziót, osztályozást, klaszterezést, NLP-t, idősorokat és megerősítéses tanulást fedik le
- **Kvíz alkalmazás**: Vue.js-alapú kvíz alkalmazás előzetes és utólagos leckeértékelésekkel
- **Többnyelvű támogatás**: Automatikus fordítás 40+ nyelvre GitHub Actions segítségével
- **Két nyelv támogatása**: Leckék elérhetők Python (Jupyter notebookok) és R (R Markdown fájlok) formájában
- **Projektalapú tanulás**: Minden témához gyakorlati projektek és feladatok tartoznak

## Repozitórium felépítése

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

Minden lecke mappa általában tartalmazza:
- `README.md` - Fő lecke tartalom
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - Megoldási kód (Python és R verziók)
- `assignment.md` - Gyakorló feladatok
- `images/` - Vizualizációs források

## Telepítési parancsok

### Python leckékhez

A legtöbb lecke Jupyter notebookokat használ. Telepítse a szükséges függőségeket:

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

### R leckékhez

Az R leckék a `solution/R/` mappákban találhatók `.rmd` vagy `.ipynb` fájlokként:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Kvíz alkalmazáshoz

A kvíz alkalmazás egy Vue.js alkalmazás, amely a `quiz-app/` könyvtárban található:

```bash
cd quiz-app
npm install
```

### Dokumentációs oldalhoz

A dokumentáció helyi futtatásához:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Fejlesztési munkafolyamat

### Lecke notebookokkal való munka

1. Navigáljon a lecke könyvtárába (pl. `2-Regression/1-Tools/`)
2. Nyissa meg a Jupyter notebookot:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Dolgozza át a lecke tartalmát és feladatait
4. Szükség esetén ellenőrizze a megoldásokat a `solution/` mappában

### Python fejlesztés

- A leckék standard Python adatfeldolgozó könyvtárakat használnak
- Jupyter notebookok interaktív tanuláshoz
- Megoldási kód elérhető minden lecke `solution/` mappájában

### R fejlesztés

- Az R leckék `.rmd` formátumban vannak (R Markdown)
- Megoldások a `solution/R/` almappákban találhatók
- Használja az RStudio-t vagy a Jupyter-t R kernellel az R notebookok futtatásához

### Kvíz alkalmazás fejlesztése

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

## Tesztelési utasítások

### Kvíz alkalmazás tesztelése

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Megjegyzés**: Ez elsősorban egy oktatási tananyag repozitórium. A lecke tartalmához nincs automatikus tesztelés. Az ellenőrzés az alábbiak révén történik:
- A lecke feladatainak elvégzése
- Notebook cellák sikeres futtatása
- Az eredmények összehasonlítása a megoldásokban várható eredményekkel

## Kódstílus irányelvek

### Python kód
- Kövesse a PEP 8 stílusirányelveket
- Használjon egyértelmű, leíró változóneveket
- Komplex műveletekhez adjon hozzá megjegyzéseket
- A Jupyter notebookok tartalmazzanak markdown cellákat a fogalmak magyarázatához

### JavaScript/Vue.js (Kvíz alkalmazás)
- Kövesse a Vue.js stílusirányelveket
- ESLint konfiguráció a `quiz-app/package.json` fájlban
- Futtassa az `npm run lint` parancsot a problémák ellenőrzéséhez és automatikus javításához

### Dokumentáció
- A markdown fájlok legyenek világosak és jól strukturáltak
- Kódpéldák bekeretezett kódblokkokban
- Belső hivatkozásokhoz relatív linkeket használjon
- Kövesse a meglévő formázási konvenciókat

## Build és telepítés

### Kvíz alkalmazás telepítése

A kvíz alkalmazás telepíthető Azure Static Web Apps-ra:

1. **Előfeltételek**:
   - Azure fiók
   - GitHub repozitórium (már fork-olt)

2. **Telepítés Azure-ra**:
   - Hozzon létre Azure Static Web App erőforrást
   - Csatlakoztassa a GitHub repozitóriumhoz
   - Állítsa be az alkalmazás helyét: `/quiz-app`
   - Állítsa be a kimeneti helyet: `dist`
   - Az Azure automatikusan létrehozza a GitHub Actions munkafolyamatot

3. **GitHub Actions munkafolyamat**:
   - A munkafolyamat fájl létrejön a `.github/workflows/azure-static-web-apps-*.yml` helyen
   - Automatikusan buildeli és telepíti a fő ágra történő push esetén

### Dokumentáció PDF

PDF generálása a dokumentációból:

```bash
npm install
npm run convert
```

## Fordítási munkafolyamat

**Fontos**: A fordítások automatikusan történnek GitHub Actions segítségével a Co-op Translator használatával.

- A fordítások automatikusan generálódnak, amikor változások kerülnek a `main` ágra
- **NE fordítsa manuálisan a tartalmat** - a rendszer kezeli ezt
- A munkafolyamat a `.github/workflows/co-op-translator.yml` fájlban van definiálva
- Azure AI/OpenAI szolgáltatásokat használ a fordításhoz
- 40+ nyelvet támogat

## Hozzájárulási irányelvek

### Tartalmi hozzájárulók számára

1. **Forkolja a repozitóriumot**, és hozzon létre egy feature branch-et
2. **Módosítsa a lecke tartalmát**, ha új leckéket ad hozzá vagy frissít
3. **Ne módosítsa a fordított fájlokat** - ezek automatikusan generálódnak
4. **Tesztelje a kódját** - győződjön meg róla, hogy minden notebook cella sikeresen fut
5. **Ellenőrizze a linkek és képek működését**
6. **Nyújtson be egy pull requestet** egyértelmű leírással

### Pull Request irányelvek

- **Cím formátuma**: `[Szekció] Változtatások rövid leírása`
  - Példa: `[Regression] Fix typo in lesson 5`
  - Példa: `[Quiz-App] Update dependencies`
- **Beküldés előtt**:
  - Győződjön meg róla, hogy minden notebook cella hibamentesen fut
  - Futtassa az `npm run lint` parancsot, ha a kvíz alkalmazást módosítja
  - Ellenőrizze a markdown formázást
  - Tesztelje az új kódpéldákat
- **PR-nek tartalmaznia kell**:
  - A változtatások leírását
  - A változtatások okát
  - Képernyőképeket, ha UI változások történtek
- **Magatartási kódex**: Kövesse a [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) irányelveit
- **CLA**: Alá kell írnia a Contributor License Agreement-et

## Lecke felépítése

Minden lecke következetes mintát követ:

1. **Előadás előtti kvíz** - Alapvető tudás tesztelése
2. **Lecke tartalom** - Írott utasítások és magyarázatok
3. **Kód bemutatók** - Gyakorlati példák notebookokban
4. **Tudásellenőrzések** - Megértés ellenőrzése közben
5. **Kihívás** - Fogalmak önálló alkalmazása
6. **Feladat** - Kiterjesztett gyakorlás
7. **Előadás utáni kvíz** - Tanulási eredmények értékelése

## Gyakori parancsok referenciája

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

## További források

- **Microsoft Learn gyűjtemény**: [ML kezdőknek modulok](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Kvíz alkalmazás**: [Online kvízek](https://ff-quizzes.netlify.app/en/ml/)
- **Vita fórum**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Videós bemutatók**: [YouTube lejátszási lista](https://aka.ms/ml-beginners-videos)

## Kulcstechnológiák

- **Python**: Elsődleges nyelv az ML leckékhez (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatív megvalósítás tidyverse, tidymodels, caret használatával
- **Jupyter**: Interaktív notebookok Python leckékhez
- **R Markdown**: Dokumentumok R leckékhez
- **Vue.js 3**: Kvíz alkalmazás keretrendszer
- **Flask**: Webalkalmazás keretrendszer ML modellek telepítéséhez
- **Docsify**: Dokumentációs oldal generátor
- **GitHub Actions**: CI/CD és automatikus fordítások

## Biztonsági megfontolások

- **Nincs titkos adat a kódban**: Soha ne kövessen el API kulcsokat vagy hitelesítő adatokat
- **Függőségek**: Tartsa naprakészen az npm és pip csomagokat
- **Felhasználói bemenet**: Flask webalkalmazás példák alapvető bemenetellenőrzést tartalmaznak
- **Érzékeny adatok**: A példákban használt adatkészletek nyilvánosak és nem érzékenyek

## Hibaelhárítás

### Jupyter notebookok

- **Kernel problémák**: Indítsa újra a kernelt, ha a cellák lefagynak: Kernel → Restart
- **Import hibák**: Győződjön meg róla, hogy minden szükséges csomag telepítve van pip segítségével
- **Útvonal problémák**: Futtassa a notebookokat a tartalmazó könyvtárból

### Kvíz alkalmazás

- **npm install sikertelen**: Törölje az npm cache-t: `npm cache clean --force`
- **Port ütközések**: Módosítsa a portot: `npm run serve -- --port 8081`
- **Build hibák**: Törölje a `node_modules` mappát, és telepítse újra: `rm -rf node_modules && npm install`

### R leckék

- **Csomag nem található**: Telepítse: `install.packages("package-name")`
- **RMarkdown renderelés**: Győződjön meg róla, hogy az rmarkdown csomag telepítve van
- **Kernel problémák**: Lehet, hogy telepíteni kell az IRkernel-t a Jupyterhez

## Projekt-specifikus megjegyzések

- Ez elsősorban egy **tanulási tananyag**, nem pedig produkciós kód
- A fókusz az **ML fogalmak megértésén** van gyakorlati tapasztalatok révén
- A kódpéldák prioritása a **tisztaság az optimalizációval szemben**
- A legtöbb lecke **önálló**, és függetlenül elvégezhető
- **Megoldások biztosítottak**, de a tanulóknak először meg kell próbálniuk a feladatokat
- A repozitórium **Docsify**-t használ webes dokumentációhoz build lépés nélkül
- **Sketchnotes** vizuális összefoglalókat nyújtanak a fogalmakról
- **Többnyelvű támogatás** globálisan elérhetővé teszi a tartalmat

---

**Felelősség kizárása**:  
Ez a dokumentum az [Co-op Translator](https://github.com/Azure/co-op-translator) AI fordítási szolgáltatás segítségével került lefordításra. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Fontos információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.