<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:11:19+00:00",
  "source_file": "AGENTS.md",
  "language_code": "fi"
}
-->
# AGENTS.md

## Projektin yleiskuvaus

Tämä on **Machine Learning for Beginners**, kattava 12 viikon ja 26 oppitunnin kurssi, joka käsittelee klassisia koneoppimisen konsepteja Pythonilla (pääasiassa Scikit-learnilla) ja R:llä. Tämä repositorio on suunniteltu itseopiskeluun sisältäen käytännön projekteja, visailuja ja tehtäviä. Jokainen oppitunti tutkii koneoppimisen käsitteitä käyttäen todellista dataa eri kulttuureista ja alueista ympäri maailmaa.

Keskeiset osat:
- **Opetussisältö**: 26 oppituntia, jotka käsittelevät koneoppimisen perusteita, regressiota, luokittelua, klusterointia, NLP:tä, aikasarjoja ja vahvistusoppimista
- **Visailusovellus**: Vue.js-pohjainen visailusovellus, jossa on ennakko- ja jälkituntiarvioinnit
- **Monikielinen tuki**: Automaattiset käännökset yli 40 kielelle GitHub Actionsin avulla
- **Kaksikielinen tuki**: Oppitunnit saatavilla sekä Pythonilla (Jupyter-notebookit) että R:llä (R Markdown -tiedostot)
- **Projektipohjainen oppiminen**: Jokainen aihe sisältää käytännön projekteja ja tehtäviä

## Repositorion rakenne

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

Jokainen oppituntikansio sisältää yleensä:
- `README.md` - Oppitunnin pääsisältö
- `notebook.ipynb` - Python Jupyter-notebook
- `solution/` - Ratkaisukoodi (Python- ja R-versiot)
- `assignment.md` - Harjoitustehtävät
- `images/` - Visuaaliset resurssit

## Asennuskomennot

### Python-oppitunteja varten

Useimmat oppitunnit käyttävät Jupyter-notebookeja. Asenna tarvittavat riippuvuudet:

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

### R-oppitunteja varten

R-oppitunnit löytyvät `solution/R/`-kansioista `.rmd`- tai `.ipynb`-tiedostoina:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Visailusovellusta varten

Visailusovellus on Vue.js-sovellus, joka sijaitsee `quiz-app/`-hakemistossa:

```bash
cd quiz-app
npm install
```

### Dokumentaatiosivustoa varten

Aja dokumentaatio paikallisesti:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Kehitystyön kulku

### Oppituntien notebookien kanssa työskentely

1. Siirry oppituntikansioon (esim. `2-Regression/1-Tools/`)
2. Avaa Jupyter-notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Käy läpi oppitunnin sisältö ja harjoitukset
4. Tarkista ratkaisut `solution/`-kansiosta tarvittaessa

### Python-kehitys

- Oppitunnit käyttävät standardeja Pythonin data-analytiikkakirjastoja
- Jupyter-notebookit interaktiiviseen oppimiseen
- Ratkaisukoodi saatavilla jokaisen oppitunnin `solution/`-kansiossa

### R-kehitys

- R-oppitunnit ovat `.rmd`-muodossa (R Markdown)
- Ratkaisut löytyvät `solution/R/`-alikansioista
- Käytä RStudioa tai Jupyteria R-ytimen kanssa R-notebookien suorittamiseen

### Visailusovelluksen kehitys

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

## Testausohjeet

### Visailusovelluksen testaus

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Huom**: Tämä on ensisijaisesti opetussuunnitelmarepositorio. Oppituntisisällölle ei ole automatisoituja testejä. Validointi tapahtuu:
- Suorittamalla oppituntien harjoitukset
- Ajamalla notebook-solut onnistuneesti
- Tarkistamalla tulokset ratkaisujen odotettujen tulosten kanssa

## Koodityyliohjeet

### Python-koodi
- Noudata PEP 8 -tyyliohjeita
- Käytä selkeitä ja kuvaavia muuttujanimiä
- Sisällytä kommentteja monimutkaisiin operaatioihin
- Jupyter-notebookeissa tulisi olla markdown-solut, jotka selittävät käsitteitä

### JavaScript/Vue.js (Visailusovellus)
- Noudata Vue.js-tyyliohjeita
- ESLint-konfiguraatio `quiz-app/package.json`-tiedostossa
- Aja `npm run lint` tarkistaaksesi ja korjataksesi ongelmat automaattisesti

### Dokumentaatio
- Markdown-tiedostojen tulee olla selkeitä ja hyvin jäsenneltyjä
- Sisällytä koodiesimerkkejä aidatuissa koodilohkoissa
- Käytä suhteellisia linkkejä sisäisiin viittauksiin
- Noudata olemassa olevia muotoilukäytäntöjä

## Rakennus ja käyttöönotto

### Visailusovelluksen käyttöönotto

Visailusovellus voidaan ottaa käyttöön Azure Static Web Appsissa:

1. **Edellytykset**:
   - Azure-tili
   - GitHub-repositorio (jo haarukoitu)

2. **Käyttöönotto Azureen**:
   - Luo Azure Static Web App -resurssi
   - Yhdistä GitHub-repositorioon
   - Aseta sovelluksen sijainti: `/quiz-app`
   - Aseta tulosteen sijainti: `dist`
   - Azure luo automaattisesti GitHub Actions -työnkulun

3. **GitHub Actions -työnkulku**:
   - Työnkulun tiedosto luodaan `.github/workflows/azure-static-web-apps-*.yml`
   - Rakentaa ja ottaa käyttöön automaattisesti, kun päähaaraan tehdään push

### Dokumentaation PDF

Luo PDF dokumentaatiosta:

```bash
npm install
npm run convert
```

## Käännöstyönkulku

**Tärkeää**: Käännökset tehdään automaattisesti GitHub Actionsin avulla Co-op Translatorilla.

- Käännökset luodaan automaattisesti, kun muutokset pushataan `main`-haaraan
- **ÄLÄ käännä sisältöä manuaalisesti** - järjestelmä hoitaa tämän
- Työnkulku määritelty tiedostossa `.github/workflows/co-op-translator.yml`
- Käyttää Azure AI/OpenAI-palveluita käännöksiin
- Tukee yli 40 kieltä

## Ohjeet osallistumiseen

### Sisällön kontribuuttoreille

1. **Haarukoi repositorio** ja luo ominaisuushaara
2. **Tee muutoksia oppituntisisältöön**, jos lisäät/päivität oppitunteja
3. **Älä muokkaa käännettyjä tiedostoja** - ne luodaan automaattisesti
4. **Testaa koodisi** - varmista, että kaikki notebook-solut toimivat onnistuneesti
5. **Varmista linkkien ja kuvien toimivuus**
6. **Lähetä pull request** selkeällä kuvauksella

### Pull request -ohjeet

- **Otsikon muoto**: `[Osio] Lyhyt kuvaus muutoksista`
  - Esimerkki: `[Regression] Korjaa kirjoitusvirhe oppitunnissa 5`
  - Esimerkki: `[Quiz-App] Päivitä riippuvuudet`
- **Ennen lähettämistä**:
  - Varmista, että kaikki notebook-solut suoritetaan ilman virheitä
  - Aja `npm run lint`, jos muokkaat visailusovellusta
  - Tarkista markdown-muotoilu
  - Testaa uudet koodiesimerkit
- **PR:n tulee sisältää**:
  - Muutosten kuvaus
  - Muutosten syy
  - Kuvakaappaukset, jos UI:ta on muutettu
- **Toimintaohjeet**: Noudata [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Sinun tulee allekirjoittaa Contributor License Agreement

## Oppituntien rakenne

Jokainen oppitunti noudattaa yhtenäistä kaavaa:

1. **Ennakkoarviointi** - Testaa lähtötiedot
2. **Oppituntisisältö** - Kirjalliset ohjeet ja selitykset
3. **Koodiesimerkit** - Käytännön esimerkit notebookeissa
4. **Tietotarkistukset** - Varmista ymmärrys oppitunnin aikana
5. **Haaste** - Sovella käsitteitä itsenäisesti
6. **Tehtävä** - Laajennettu harjoittelu
7. **Jälkiarviointi** - Arvioi oppimistulokset

## Yleiset komennot

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

## Lisäresurssit

- **Microsoft Learn Collection**: [ML for Beginners -moduulit](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Visailusovellus**: [Online-visailut](https://ff-quizzes.netlify.app/en/ml/)
- **Keskustelupalsta**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video-opastukset**: [YouTube-soittolista](https://aka.ms/ml-beginners-videos)

## Keskeiset teknologiat

- **Python**: Pääasiallinen kieli koneoppimisoppitunneille (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Vaihtoehtoinen toteutus tidyverse-, tidymodels- ja caret-kirjastoilla
- **Jupyter**: Interaktiiviset notebookit Python-oppitunneille
- **R Markdown**: Dokumentit R-oppitunneille
- **Vue.js 3**: Visailusovelluksen kehys
- **Flask**: Web-sovelluskehys koneoppimismallien käyttöönottoon
- **Docsify**: Dokumentaatiosivuston generaattori
- **GitHub Actions**: CI/CD ja automaattiset käännökset

## Turvallisuushuomiot

- **Ei salaisuuksia koodissa**: Älä koskaan lisää API-avaimia tai tunnuksia
- **Riippuvuudet**: Pidä npm- ja pip-paketit ajan tasalla
- **Käyttäjän syötteet**: Flask-web-sovellusesimerkit sisältävät perustason syötteen validoinnin
- **Arkaluontoinen data**: Esimerkkidatasetit ovat julkisia ja ei-arkaluontoisia

## Vianmääritys

### Jupyter-notebookit

- **Ytimen ongelmat**: Käynnistä ydin uudelleen, jos solut jumittuvat: Kernel → Restart
- **Tuontivirheet**: Varmista, että kaikki tarvittavat paketit on asennettu pipillä
- **Polkuongelmat**: Aja notebookit niiden sisältävästä hakemistosta

### Visailusovellus

- **npm install epäonnistuu**: Tyhjennä npm-välimuisti: `npm cache clean --force`
- **Porttikonfliktit**: Vaihda porttia komennolla: `npm run serve -- --port 8081`
- **Rakennusvirheet**: Poista `node_modules` ja asenna uudelleen: `rm -rf node_modules && npm install`

### R-oppitunnit

- **Pakettia ei löydy**: Asenna komennolla: `install.packages("package-name")`
- **RMarkdown-renderointi**: Varmista, että rmarkdown-paketti on asennettu
- **Ytimen ongelmat**: Saatat joutua asentamaan IRkernelin Jupyteria varten

## Projektikohtaiset huomiot

- Tämä on ensisijaisesti **oppimiskurssi**, ei tuotantokoodi
- Keskittyy **koneoppimisen käsitteiden ymmärtämiseen** käytännön harjoitusten avulla
- Koodiesimerkit painottavat **selkeyttä optimoinnin sijaan**
- Useimmat oppitunnit ovat **itsenäisiä** ja ne voi suorittaa erikseen
- **Ratkaisut saatavilla**, mutta oppijoiden tulisi yrittää tehtäviä ensin
- Repositorio käyttää **Docsifyä** verkkodokumentaatioon ilman rakennusvaihetta
- **Sketchnotes** tarjoavat visuaalisia yhteenvetoja käsitteistä
- **Monikielinen tuki** tekee sisällöstä globaalisti saavutettavaa

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.