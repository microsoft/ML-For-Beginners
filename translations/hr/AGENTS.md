<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:18:38+00:00",
  "source_file": "AGENTS.md",
  "language_code": "hr"
}
-->
# AGENTS.md

## Pregled projekta

Ovo je **Strojno učenje za početnike**, sveobuhvatan 12-tjedni kurikulum s 26 lekcija koji pokriva klasične koncepte strojnog učenja koristeći Python (prvenstveno sa Scikit-learn) i R. Repozitorij je osmišljen kao resurs za samostalno učenje s praktičnim projektima, kvizovima i zadacima. Svaka lekcija istražuje koncepte strojnog učenja kroz stvarne podatke iz različitih kultura i regija širom svijeta.

Ključne komponente:
- **Edukativni sadržaj**: 26 lekcija koje pokrivaju uvod u strojno učenje, regresiju, klasifikaciju, grupiranje, NLP, vremenske serije i učenje pojačanjem
- **Aplikacija za kvizove**: Vue.js aplikacija za kvizove s procjenama prije i nakon lekcija
- **Podrška za više jezika**: Automatski prijevodi na više od 40 jezika putem GitHub Actions
- **Podrška za dva jezika**: Lekcije dostupne na Pythonu (Jupyter bilježnice) i R-u (R Markdown datoteke)
- **Učenje kroz projekte**: Svaka tema uključuje praktične projekte i zadatke

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

Svaka mapa lekcije obično sadrži:
- `README.md` - Glavni sadržaj lekcije
- `notebook.ipynb` - Python Jupyter bilježnica
- `solution/` - Kod rješenja (verzije za Python i R)
- `assignment.md` - Vježbe za praksu
- `images/` - Vizualni resursi

## Komande za postavljanje

### Za lekcije na Pythonu

Većina lekcija koristi Jupyter bilježnice. Instalirajte potrebne ovisnosti:

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

### Za lekcije na R-u

Lekcije na R-u nalaze se u mapama `solution/R/` kao `.rmd` ili `.ipynb` datoteke:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Za aplikaciju za kvizove

Aplikacija za kvizove je Vue.js aplikacija smještena u direktoriju `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Za dokumentacijsku stranicu

Za pokretanje dokumentacije lokalno:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Radni tijek razvoja

### Rad s bilježnicama lekcija

1. Idite u direktorij lekcije (npr. `2-Regression/1-Tools/`)
2. Otvorite Jupyter bilježnicu:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Prođite kroz sadržaj lekcije i vježbe
4. Provjerite rješenja u mapi `solution/` ako je potrebno

### Razvoj na Pythonu

- Lekcije koriste standardne Python biblioteke za podatkovnu znanost
- Jupyter bilježnice za interaktivno učenje
- Kod rješenja dostupan je u mapi `solution/` svake lekcije

### Razvoj na R-u

- Lekcije na R-u su u `.rmd` formatu (R Markdown)
- Rješenja se nalaze u poddirektorijima `solution/R/`
- Koristite RStudio ili Jupyter s R kernelom za pokretanje R bilježnica

### Razvoj aplikacije za kvizove

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

## Upute za testiranje

### Testiranje aplikacije za kvizove

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Napomena**: Ovo je prvenstveno repozitorij edukativnog kurikuluma. Ne postoje automatizirani testovi za sadržaj lekcija. Validacija se provodi kroz:
- Rješavanje vježbi iz lekcija
- Uspješno pokretanje ćelija u bilježnicama
- Provjeru rezultata u odnosu na očekivane rezultate u rješenjima

## Smjernice za stil kodiranja

### Python kod
- Slijedite PEP 8 smjernice za stil
- Koristite jasna, opisna imena varijabli
- Dodajte komentare za složene operacije
- Jupyter bilježnice trebaju sadržavati markdown ćelije koje objašnjavaju koncepte

### JavaScript/Vue.js (aplikacija za kvizove)
- Slijedi Vue.js smjernice za stil
- ESLint konfiguracija u `quiz-app/package.json`
- Pokrenite `npm run lint` za provjeru i automatsko ispravljanje problema

### Dokumentacija
- Markdown datoteke trebaju biti jasne i dobro strukturirane
- Uključite primjere koda u blokove s ograničenjem
- Koristite relativne poveznice za interne reference
- Slijedite postojeće konvencije formatiranja

## Izrada i implementacija

### Implementacija aplikacije za kvizove

Aplikacija za kvizove može se implementirati na Azure Static Web Apps:

1. **Preduvjeti**:
   - Azure račun
   - GitHub repozitorij (već forkiran)

2. **Implementacija na Azure**:
   - Kreirajte resurs Azure Static Web App
   - Povežite se s GitHub repozitorijem
   - Postavite lokaciju aplikacije: `/quiz-app`
   - Postavite lokaciju izlaza: `dist`
   - Azure automatski kreira GitHub Actions tijek rada

3. **GitHub Actions tijek rada**:
   - Datoteka tijeka rada kreirana u `.github/workflows/azure-static-web-apps-*.yml`
   - Automatski se gradi i implementira pri svakom pushu na glavnu granu

### PDF dokumentacija

Generirajte PDF iz dokumentacije:

```bash
npm install
npm run convert
```

## Tijek prijevoda

**Važno**: Prijevodi se automatski obavljaju putem GitHub Actions koristeći Co-op Translator.

- Prijevodi se automatski generiraju kada se promjene pošalju na granu `main`
- **NE prevodite sadržaj ručno** - sustav to obavlja
- Tijek rada definiran je u `.github/workflows/co-op-translator.yml`
- Koristi Azure AI/OpenAI usluge za prijevod
- Podržava više od 40 jezika

## Smjernice za doprinos

### Za suradnike na sadržaju

1. **Forkajte repozitorij** i kreirajte granu za značajke
2. **Izvršite promjene u sadržaju lekcija** ako dodajete/ažurirate lekcije
3. **Ne mijenjajte prevedene datoteke** - one se automatski generiraju
4. **Testirajte svoj kod** - osigurajte da se sve ćelije bilježnice uspješno pokreću
5. **Provjerite poveznice i slike** da ispravno rade
6. **Pošaljite pull request** s jasnim opisom

### Smjernice za pull request

- **Format naslova**: `[Sekcija] Kratak opis promjena`
  - Primjer: `[Regresija] Ispravak tipfelera u lekciji 5`
  - Primjer: `[Quiz-App] Ažuriranje ovisnosti`
- **Prije slanja**:
  - Osigurajte da se sve ćelije bilježnice izvršavaju bez grešaka
  - Pokrenite `npm run lint` ako mijenjate quiz-app
  - Provjerite formatiranje markdowna
  - Testirajte sve nove primjere koda
- **PR mora sadržavati**:
  - Opis promjena
  - Razlog promjena
  - Snimke zaslona ako postoje promjene u korisničkom sučelju
- **Kodeks ponašanja**: Slijedite [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Morat ćete potpisati Ugovor o licenciranju suradnika

## Struktura lekcije

Svaka lekcija slijedi dosljedan obrazac:

1. **Kviz prije predavanja** - Testiranje osnovnog znanja
2. **Sadržaj lekcije** - Pisane upute i objašnjenja
3. **Demonstracije koda** - Praktični primjeri u bilježnicama
4. **Provjere znanja** - Provjera razumijevanja tijekom lekcije
5. **Izazov** - Primjena koncepata samostalno
6. **Zadatak** - Proširena praksa
7. **Kviz nakon predavanja** - Procjena rezultata učenja

## Referenca za uobičajene komande

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

## Dodatni resursi

- **Microsoft Learn kolekcija**: [Moduli za početnike u ML-u](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikacija za kvizove**: [Online kvizovi](https://ff-quizzes.netlify.app/en/ml/)
- **Forum za raspravu**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video vodiči**: [YouTube playlist](https://aka.ms/ml-beginners-videos)

## Ključne tehnologije

- **Python**: Glavni jezik za lekcije o strojnome učenju (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternativna implementacija koristeći tidyverse, tidymodels, caret
- **Jupyter**: Interaktivne bilježnice za lekcije na Pythonu
- **R Markdown**: Dokumenti za lekcije na R-u
- **Vue.js 3**: Okvir za aplikaciju za kvizove
- **Flask**: Okvir za web aplikacije za implementaciju ML modela
- **Docsify**: Generator dokumentacijskih stranica
- **GitHub Actions**: CI/CD i automatizirani prijevodi

## Sigurnosni aspekti

- **Bez tajni u kodu**: Nikada ne komitirajte API ključeve ili vjerodajnice
- **Ovisnosti**: Održavajte npm i pip pakete ažuriranima
- **Unos korisnika**: Primjeri web aplikacija u Flasku uključuju osnovnu validaciju unosa
- **Osjetljivi podaci**: Primjeri skupova podataka su javni i neosjetljivi

## Rješavanje problema

### Jupyter bilježnice

- **Problemi s kernelom**: Ponovno pokrenite kernel ako ćelije zastanu: Kernel → Restart
- **Import greške**: Osigurajte da su sve potrebne biblioteke instalirane s pip
- **Problemi s putanjom**: Pokrenite bilježnice iz njihovog direktorija

### Aplikacija za kvizove

- **npm install ne uspijeva**: Očistite npm cache: `npm cache clean --force`
- **Sukobi portova**: Promijenite port s: `npm run serve -- --port 8081`
- **Greške pri izradi**: Obrišite `node_modules` i ponovno instalirajte: `rm -rf node_modules && npm install`

### Lekcije na R-u

- **Paket nije pronađen**: Instalirajte s: `install.packages("ime-paketa")`
- **RMarkdown renderiranje**: Osigurajte da je rmarkdown paket instaliran
- **Problemi s kernelom**: Možda ćete morati instalirati IRkernel za Jupyter

## Napomene specifične za projekt

- Ovo je prvenstveno **kurikulum za učenje**, a ne produkcijski kod
- Fokus je na **razumijevanju koncepata strojnog učenja** kroz praktičnu primjenu
- Primjeri koda daju prednost **jasnoći nad optimizacijom**
- Većina lekcija je **samostalna** i može se završiti neovisno
- **Rješenja su dostupna**, ali učenici bi trebali prvo pokušati riješiti vježbe
- Repozitorij koristi **Docsify** za web dokumentaciju bez koraka izrade
- **Sketchnotes** pružaju vizualne sažetke koncepata
- **Podrška za više jezika** čini sadržaj globalno dostupnim

---

**Izjava o odricanju odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za nesporazume ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.