<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-11T11:10:12+00:00",
  "source_file": "AGENTS.md",
  "language_code": "et"
}
-->
# AGENTS.md

## Projekti Ülevaade

See on **Masinõpe Algajatele**, põhjalik 12-nädalane, 26-õppetunnist koosnev õppekava, mis käsitleb klassikalisi masinõppe kontseptsioone Pythonis (peamiselt Scikit-learniga) ja R-is. Repositoorium on loodud iseseisvaks õppimiseks praktiliste projektide, viktoriinide ja ülesannetega. Iga õppetund uurib masinõppe kontseptsioone, kasutades pärismaailma andmeid erinevatest kultuuridest ja piirkondadest üle maailma.

Peamised komponendid:
- **Õppematerjalid**: 26 õppetundi, mis hõlmavad sissejuhatust masinõppesse, regressiooni, klassifikatsiooni, klasterdamist, NLP-d, ajaseeriaid ja tugevdusõpet
- **Viktoriinirakendus**: Vue.js-põhine viktoriinirakendus eel- ja järelõppetundide hindamiseks
- **Mitmekeelne tugi**: Automaatne tõlge enam kui 40 keelde GitHub Actionsi kaudu
- **Kahe keele tugi**: Õppetunnid saadaval nii Pythonis (Jupyteri märkmikud) kui ka R-is (R Markdowni failid)
- **Projektipõhine õpe**: Iga teema sisaldab praktilisi projekte ja ülesandeid

## Repositooriumi Struktuur

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

Iga õppetunni kaust sisaldab tavaliselt:
- `README.md` - Peamine õppetunni sisu
- `notebook.ipynb` - Pythoni Jupyteri märkmik
- `solution/` - Lahenduskood (Pythoni ja R-i versioonid)
- `assignment.md` - Harjutusülesanded
- `images/` - Visuaalsed ressursid

## Seadistamise Käsud

### Pythoni Õppetundide Jaoks

Enamik õppetunde kasutab Jupyteri märkmikke. Paigalda vajalikud sõltuvused:

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

### R-i Õppetundide Jaoks

R-i õppetunnid asuvad `solution/R/` kaustades `.rmd` või `.ipynb` failidena:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Viktoriinirakenduse Jaoks

Viktoriinirakendus asub `quiz-app/` kataloogis:

```bash
cd quiz-app
npm install
```

### Dokumentatsiooni Veebilehe Jaoks

Dokumentatsiooni kohalikuks käivitamiseks:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Arenduse Töövoog

### Õppetundide Märkmikega Töötamine

1. Liigu õppetunni kataloogi (nt `2-Regression/1-Tools/`)
2. Ava Jupyteri märkmik:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Töötle läbi õppetunni sisu ja harjutused
4. Vajadusel vaata lahendusi `solution/` kaustas

### Pythoni Arendus

- Õppetunnid kasutavad standardseid Pythoni andmeteaduse teeke
- Jupyteri märkmikud interaktiivseks õppimiseks
- Lahenduskood saadaval iga õppetunni `solution/` kaustas

### R-i Arendus

- R-i õppetunnid on `.rmd` formaadis (R Markdown)
- Lahendused asuvad `solution/R/` alamkataloogides
- Kasuta RStudio või Jupyterit R-i kerneliga R-i märkmike käivitamiseks

### Viktoriinirakenduse Arendus

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

## Testimise Juhised

### Viktoriinirakenduse Testimine

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Märkus**: See on peamiselt hariduslik õppekava repositoorium. Õppetundide sisule automaatseid teste ei ole. Valideerimine toimub:
- Õppetundide harjutuste läbimisega
- Märkmiku lahtrite edukal käivitamisel
- Väljundi kontrollimisel lahenduste oodatud tulemustega

## Koodistiili Juhised

### Pythoni Kood
- Järgi PEP 8 stiilijuhiseid
- Kasuta selgeid ja kirjeldavaid muutujanimesid
- Lisa kommentaare keerukate operatsioonide jaoks
- Jupyteri märkmikud peaksid sisaldama markdown-lahtrites kontseptsioonide selgitusi

### JavaScript/Vue.js (Viktoriinirakendus)
- Järgib Vue.js stiilijuhendit
- ESLinti konfiguratsioon `quiz-app/package.json` failis
- Käivita `npm run lint`, et kontrollida ja automaatselt parandada probleeme

### Dokumentatsioon
- Markdown-failid peaksid olema selged ja hästi struktureeritud
- Lisa koodinäited piiratud koodiplokkidesse
- Kasuta sisemiste viidete jaoks suhtelisi linke
- Järgi olemasolevaid vorminduskonventsioone

## Ehitamine ja Juurutamine

### Viktoriinirakenduse Juurutamine

Viktoriinirakendust saab juurutada Azure Static Web Appsis:

1. **Eeltingimused**:
   - Azure'i konto
   - GitHubi repositoorium (juba kahvli tehtud)

2. **Juuruta Azure'i**:
   - Loo Azure Static Web App ressurss
   - Ühenda GitHubi repositooriumiga
   - Määra rakenduse asukoht: `/quiz-app`
   - Määra väljundi asukoht: `dist`
   - Azure loob automaatselt GitHub Actionsi töövoo

3. **GitHub Actionsi Töövoog**:
   - Töövoo fail luuakse `.github/workflows/azure-static-web-apps-*.yml` kataloogi
   - Automaatne ehitamine ja juurutamine `main` harule pushimisel

### Dokumentatsiooni PDF

Loo PDF dokumentatsioonist:

```bash
npm install
npm run convert
```

## Tõlke Töövoog

**Oluline**: Tõlked tehakse automaatselt GitHub Actionsi kaudu, kasutades Co-op Translatorit.

- Tõlked genereeritakse automaatselt, kui muudatused lükatakse `main` harule
- **ÄRA tõlgi sisu käsitsi** - süsteem teeb seda automaatselt
- Töövoog määratletud `.github/workflows/co-op-translator.yml` failis
- Kasutab Azure AI/OpenAI teenuseid tõlkimiseks
- Toetab enam kui 40 keelt

## Kaastöö Juhised

### Sisu Kaastöölistele

1. **Kahvli repositoorium** ja loo funktsiooni haru
2. **Tee muudatusi õppetunni sisus**, kui lisad/värskendad õppetunde
3. **Ära muuda tõlgitud faile** - need genereeritakse automaatselt
4. **Testi oma koodi** - veendu, et kõik märkmiku lahtrid töötavad edukalt
5. **Kontrolli linkide ja piltide** korrektset toimimist
6. **Esita tõmbepäring** selge kirjeldusega

### Tõmbepäringu Juhised

- **Pealkirja formaat**: `[Sektsioon] Lühike muudatuste kirjeldus`
  - Näide: `[Regressioon] Paranda viga õppetunnis 5`
  - Näide: `[Viktoriinirakendus] Uuenda sõltuvusi`
- **Enne esitamist**:
  - Veendu, et kõik märkmiku lahtrid töötavad veatult
  - Käivita `npm run lint`, kui muudad viktoriinirakendust
  - Kontrolli markdowni vormindust
  - Testi uusi koodinäiteid
- **Tõmbepäring peab sisaldama**:
  - Muudatuste kirjeldust
  - Muudatuste põhjust
  - Ekraanipilte, kui UI-s on muudatusi
- **Käitumisjuhend**: Järgi [Microsofti avatud lähtekoodi käitumisjuhendit](CODE_OF_CONDUCT.md)
- **CLA**: Pead allkirjastama kaastöölise litsentsilepingu

## Õppetunni Struktuur

Iga õppetund järgib ühtset mustrit:

1. **Eelloengu viktoriin** - Testi algteadmisi
2. **Õppetunni sisu** - Kirjalikud juhised ja selgitused
3. **Koodi demonstratsioonid** - Praktilised näited märkmikes
4. **Teadmiste kontrollid** - Kontrolli arusaamist õppetunni jooksul
5. **Väljakutse** - Rakenda kontseptsioone iseseisvalt
6. **Ülesanne** - Põhjalikum harjutamine
7. **Järelloengu viktoriin** - Hinda õpitulemusi

## Üldised Käskude Viited

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

## Täiendavad Ressursid

- **Microsoft Learn Kogumik**: [Masinõpe Algajatele moodulid](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Viktoriinirakendus**: [Veebiviktoriinid](https://ff-quizzes.netlify.app/en/ml/)
- **Arutelu Foorum**: [GitHubi Arutelud](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video Juhendid**: [YouTube'i Esitusloend](https://aka.ms/ml-beginners-videos)

## Peamised Tehnoloogiad

- **Python**: Peamine keel masinõppe õppetundide jaoks (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatiivne teostus, kasutades tidyverse'i, tidymodels'i, caret'i
- **Jupyter**: Interaktiivsed märkmikud Pythoni õppetundide jaoks
- **R Markdown**: Dokumendid R-i õppetundide jaoks
- **Vue.js 3**: Viktoriinirakenduse raamistik
- **Flask**: Veebirakenduse raamistik masinõppe mudelite juurutamiseks
- **Docsify**: Dokumentatsiooni veebilehe generaator
- **GitHub Actions**: CI/CD ja automaatsed tõlked

## Turvalisuse Kaalutlused

- **Ära lisa koodi saladusi**: Ära kunagi commit'i API võtmeid või mandaate
- **Sõltuvused**: Hoia npm ja pip paketid ajakohased
- **Kasutaja sisend**: Flaski veebirakenduse näited sisaldavad põhilist sisendi valideerimist
- **Tundlikud andmed**: Näitedatasetid on avalikud ja mittetundlikud

## Tõrkeotsing

### Jupyteri Märkmikud

- **Kerneliprobleemid**: Taaskäivita kernel, kui lahtrid jäävad kinni: Kernel → Restart
- **Importimisvead**: Veendu, et kõik vajalikud paketid on paigaldatud pip'iga
- **Raja probleemid**: Käivita märkmikud nende sisaldavast kataloogist

### Viktoriinirakendus

- **npm install ebaõnnestub**: Tühjenda npm vahemälu: `npm cache clean --force`
- **Porti konfliktid**: Muuda porti käsuga: `npm run serve -- --port 8081`
- **Ehitusvead**: Kustuta `node_modules` ja paigalda uuesti: `rm -rf node_modules && npm install`

### R-i Õppetunnid

- **Paketti ei leitud**: Paigalda käsuga: `install.packages("package-name")`
- **RMarkdowni renderdamine**: Veendu, et rmarkdown pakett on paigaldatud
- **Kerneliprobleemid**: Võib olla vaja paigaldada IRkernel Jupyteri jaoks

## Projekti Spetsiifilised Märkused

- See on peamiselt **õppeõppekava**, mitte tootmiskood
- Keskendutakse **masinõppe kontseptsioonide mõistmisele** praktilise harjutamise kaudu
- Koodinäited eelistavad **selgust optimeerimise asemel**
- Enamik õppetunde on **iseseisvad** ja neid saab lõpetada eraldi
- **Lahendused on saadaval**, kuid õppijad peaksid esmalt harjutusi ise proovima
- Repositoorium kasutab **Docsify't** veebidokumentatsiooni jaoks ilma ehitusetapita
- **Sketchnotes** pakuvad visuaalseid kokkuvõtteid kontseptsioonidest
- **Mitmekeelne tugi** teeb sisu globaalselt kättesaadavaks

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.