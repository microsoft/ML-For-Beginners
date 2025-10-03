<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:14:17+00:00",
  "source_file": "AGENTS.md",
  "language_code": "tl"
}
-->
# AGENTS.md

## Pangkalahatang-ideya ng Proyekto

Ito ay **Machine Learning para sa mga Baguhan**, isang komprehensibong kurikulum na tumatagal ng 12 linggo at may 26 na aralin na sumasaklaw sa mga klasikong konsepto ng machine learning gamit ang Python (pangunahing gamit ang Scikit-learn) at R. Ang repositoryo ay idinisenyo bilang isang self-paced na mapagkukunan ng pag-aaral na may mga hands-on na proyekto, pagsusulit, at mga takdang-aralin. Ang bawat aralin ay nag-eeksplora ng mga konsepto ng ML gamit ang mga totoong datos mula sa iba't ibang kultura at rehiyon sa buong mundo.

Mga pangunahing bahagi:
- **Nilalaman Pang-edukasyon**: 26 na aralin na sumasaklaw sa pagpapakilala sa ML, regression, classification, clustering, NLP, time series, at reinforcement learning
- **Quiz Application**: Vue.js-based na quiz app na may pre- at post-lesson assessments
- **Suporta sa Multi-language**: Awtomatikong pagsasalin sa mahigit 40 wika gamit ang GitHub Actions
- **Suporta sa Dalawang Wika**: Mga aralin na magagamit sa Python (Jupyter notebooks) at R (R Markdown files)
- **Pag-aaral Batay sa Proyekto**: Ang bawat paksa ay may kasamang praktikal na proyekto at mga takdang-aralin

## Istruktura ng Repositoryo

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

Ang bawat folder ng aralin ay karaniwang naglalaman ng:
- `README.md` - Pangunahing nilalaman ng aralin
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - Solution code (Python at R na bersyon)
- `assignment.md` - Mga pagsasanay na gawain
- `images/` - Mga visual na mapagkukunan

## Mga Setup na Utos

### Para sa Mga Aralin sa Python

Karamihan sa mga aralin ay gumagamit ng Jupyter notebooks. I-install ang mga kinakailangang dependencies:

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

### Para sa Mga Aralin sa R

Ang mga aralin sa R ay nasa `solution/R/` na mga folder bilang `.rmd` o `.ipynb` na mga file:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Para sa Quiz Application

Ang quiz app ay isang Vue.js application na matatagpuan sa `quiz-app/` na direktoryo:

```bash
cd quiz-app
npm install
```

### Para sa Documentation Site

Upang patakbuhin ang dokumentasyon nang lokal:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Workflow ng Pag-develop

### Paggawa gamit ang Lesson Notebooks

1. Pumunta sa direktoryo ng aralin (hal., `2-Regression/1-Tools/`)
2. Buksan ang Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Dumaan sa nilalaman ng aralin at mga pagsasanay
4. Tingnan ang mga solusyon sa `solution/` na folder kung kinakailangan

### Pag-develop gamit ang Python

- Ang mga aralin ay gumagamit ng karaniwang Python data science libraries
- Jupyter notebooks para sa interactive na pag-aaral
- Ang solution code ay magagamit sa `solution/` na folder ng bawat aralin

### Pag-develop gamit ang R

- Ang mga aralin sa R ay nasa `.rmd` na format (R Markdown)
- Ang mga solusyon ay matatagpuan sa `solution/R/` na subdirectories
- Gumamit ng RStudio o Jupyter na may R kernel upang patakbuhin ang R notebooks

### Pag-develop ng Quiz Application

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

## Mga Tagubilin sa Pagsubok

### Pagsubok ng Quiz Application

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Tandaan**: Pangunahing pang-edukasyon na kurikulum ang repositoryo na ito. Walang awtomatikong pagsubok para sa nilalaman ng aralin. Ang validation ay ginagawa sa pamamagitan ng:
- Pagtatapos ng mga pagsasanay sa aralin
- Matagumpay na pagpapatakbo ng mga notebook cells
- Pagsusuri ng output laban sa inaasahang resulta sa mga solusyon

## Mga Alituntunin sa Estilo ng Code

### Python Code
- Sundin ang PEP 8 na mga alituntunin sa estilo
- Gumamit ng malinaw at deskriptibong mga pangalan ng variable
- Maglagay ng mga komento para sa mga kumplikadong operasyon
- Ang Jupyter notebooks ay dapat may markdown cells na nagpapaliwanag ng mga konsepto

### JavaScript/Vue.js (Quiz App)
- Sundin ang Vue.js na gabay sa estilo
- ESLint configuration sa `quiz-app/package.json`
- Patakbuhin ang `npm run lint` upang suriin at awtomatikong ayusin ang mga isyu

### Dokumentasyon
- Ang mga markdown files ay dapat malinaw at maayos ang istruktura
- Maglagay ng mga halimbawa ng code sa fenced code blocks
- Gumamit ng mga relative links para sa internal na mga reference
- Sundin ang umiiral na mga convention sa formatting

## Build at Deployment

### Deployment ng Quiz Application

Ang quiz app ay maaaring i-deploy sa Azure Static Web Apps:

1. **Mga Kinakailangan**:
   - Azure account
   - GitHub repository (na na-fork na)

2. **I-deploy sa Azure**:
   - Gumawa ng Azure Static Web App resource
   - Ikonekta sa GitHub repository
   - Itakda ang lokasyon ng app: `/quiz-app`
   - Itakda ang lokasyon ng output: `dist`
   - Awtomatikong gagawa ang Azure ng GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Ang workflow file ay gagawin sa `.github/workflows/azure-static-web-apps-*.yml`
   - Awtomatikong magbi-build at magde-deploy kapag may push sa main branch

### PDF ng Dokumentasyon

Gumawa ng PDF mula sa dokumentasyon:

```bash
npm install
npm run convert
```

## Workflow ng Pagsasalin

**Mahalaga**: Ang mga pagsasalin ay awtomatikong ginagawa gamit ang GitHub Actions gamit ang Co-op Translator.

- Ang mga pagsasalin ay awtomatikong ginagawa kapag may mga pagbabago sa `main` branch
- **HUWAG mano-manong isalin ang nilalaman** - ang sistema ang bahala dito
- Ang workflow ay tinukoy sa `.github/workflows/co-op-translator.yml`
- Gumagamit ng Azure AI/OpenAI services para sa pagsasalin
- Sinusuportahan ang mahigit 40 wika

## Mga Alituntunin sa Pag-aambag

### Para sa Mga Contributor ng Nilalaman

1. **I-fork ang repositoryo** at gumawa ng feature branch
2. **Gumawa ng mga pagbabago sa nilalaman ng aralin** kung magdadagdag o mag-a-update ng mga aralin
3. **Huwag baguhin ang mga isinaling file** - awtomatikong ginagawa ang mga ito
4. **Subukan ang iyong code** - tiyaking lahat ng notebook cells ay tumatakbo nang matagumpay
5. **Suriin ang mga link at imahe** kung gumagana nang tama
6. **Mag-submit ng pull request** na may malinaw na deskripsyon

### Mga Alituntunin sa Pull Request

- **Format ng Pamagat**: `[Section] Maikling deskripsyon ng mga pagbabago`
  - Halimbawa: `[Regression] Ayusin ang typo sa lesson 5`
  - Halimbawa: `[Quiz-App] I-update ang dependencies`
- **Bago mag-submit**:
  - Tiyaking lahat ng notebook cells ay tumatakbo nang walang error
  - Patakbuhin ang `npm run lint` kung magbabago sa quiz-app
  - Suriin ang formatting ng markdown
  - Subukan ang anumang bagong halimbawa ng code
- **Dapat kasama sa PR**:
  - Deskripsyon ng mga pagbabago
  - Dahilan ng mga pagbabago
  - Mga screenshot kung may pagbabago sa UI
- **Code of Conduct**: Sundin ang [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: Kailangan mong pumirma sa Contributor License Agreement

## Istruktura ng Aralin

Ang bawat aralin ay sumusunod sa pare-parehong pattern:

1. **Pre-lecture quiz** - Subukan ang baseline na kaalaman
2. **Nilalaman ng aralin** - Nakasaad na mga tagubilin at paliwanag
3. **Mga demonstrasyon ng code** - Mga hands-on na halimbawa sa notebooks
4. **Mga pagsusuri ng kaalaman** - Suriin ang pag-unawa sa buong aralin
5. **Hamunin** - I-apply ang mga konsepto nang mag-isa
6. **Takdang-aralin** - Pinalawak na pagsasanay
7. **Post-lecture quiz** - Suriin ang mga natutunan

## Karaniwang Mga Utos na Sanggunian

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

## Karagdagang Mga Mapagkukunan

- **Microsoft Learn Collection**: [ML para sa mga Baguhan na modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz App**: [Online quizzes](https://ff-quizzes.netlify.app/en/ml/)
- **Discussion Board**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Mga Video Walkthroughs**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Mga Pangunahing Teknolohiya

- **Python**: Pangunahing wika para sa mga aralin sa ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternatibong implementasyon gamit ang tidyverse, tidymodels, caret
- **Jupyter**: Interactive notebooks para sa mga aralin sa Python
- **R Markdown**: Mga dokumento para sa mga aralin sa R
- **Vue.js 3**: Framework para sa quiz application
- **Flask**: Framework para sa web application ng ML model deployment
- **Docsify**: Generator para sa documentation site
- **GitHub Actions**: CI/CD at awtomatikong pagsasalin

## Mga Pagsasaalang-alang sa Seguridad

- **Walang mga lihim sa code**: Huwag kailanman mag-commit ng API keys o mga kredensyal
- **Mga Dependencies**: Panatilihing updated ang npm at pip packages
- **Input ng User**: Ang mga halimbawa ng Flask web app ay may kasamang basic input validation
- **Sensitibong datos**: Ang mga dataset na ginamit ay pampubliko at hindi sensitibo

## Pag-troubleshoot

### Jupyter Notebooks

- **Mga isyu sa Kernel**: I-restart ang kernel kung mag-hang ang mga cells: Kernel → Restart
- **Mga error sa Import**: Tiyaking lahat ng kinakailangang packages ay naka-install gamit ang pip
- **Mga isyu sa Path**: Patakbuhin ang notebooks mula sa kanilang containing directory

### Quiz Application

- **npm install fails**: I-clear ang npm cache: `npm cache clean --force`
- **Mga conflict sa Port**: Palitan ang port gamit ang: `npm run serve -- --port 8081`
- **Mga error sa Build**: Tanggalin ang `node_modules` at i-reinstall: `rm -rf node_modules && npm install`

### Mga Aralin sa R

- **Package not found**: I-install gamit ang: `install.packages("package-name")`
- **RMarkdown rendering**: Tiyaking naka-install ang rmarkdown package
- **Mga isyu sa Kernel**: Maaaring kailangang i-install ang IRkernel para sa Jupyter

## Mga Tala Tungkol sa Proyekto

- Pangunahing **kurikulum sa pag-aaral** ito, hindi production code
- Ang pokus ay sa **pag-unawa sa mga konsepto ng ML** sa pamamagitan ng hands-on na pagsasanay
- Ang mga halimbawa ng code ay inuuna ang **kalinawan kaysa sa optimization**
- Karamihan sa mga aralin ay **self-contained** at maaaring tapusin nang mag-isa
- **May mga solusyon** ngunit dapat subukan muna ng mga mag-aaral ang mga pagsasanay
- Ang repositoryo ay gumagamit ng **Docsify** para sa web documentation nang walang build step
- Ang **Sketchnotes** ay nagbibigay ng visual na buod ng mga konsepto
- Ang **Suporta sa Multi-language** ay ginagawang globally accessible ang nilalaman

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, mangyaring tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.