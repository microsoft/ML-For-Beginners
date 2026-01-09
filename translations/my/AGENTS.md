<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:19:32+00:00",
  "source_file": "AGENTS.md",
  "language_code": "my"
}
-->
# AGENTS.md

## ပရောဂျက်အကျဉ်းချုပ်

ဒီဟာက **Machine Learning for Beginners** ဖြစ်ပြီး Python (အဓိကအားဖြင့် Scikit-learn) နဲ့ R ကို အသုံးပြုပြီး စဉ်ဆက်မပြတ် ၁၂ ပတ်၊ ၂၆ သင်ခန်းစာများပါဝင်တဲ့ စုံလင်တဲ့ သင်ရိုးညွှန်းတစ်ခုပါ။ ဒီ repository ကို ကိုယ်တိုင်လေ့လာနိုင်တဲ့ အရင်းအမြစ်အဖြစ် ဒီဇိုင်းထုတ်ထားပြီး လက်တွေ့လုပ်ငန်းများ၊ မေးခွန်းများနဲ့ လေ့ကျင့်ခန်းများပါဝင်ပါတယ်။ သင်ခန်းစာတစ်ခုစီမှာ ကမ္ဘာ့အခြားဒေသများမှ အချက်အလက်များကို အသုံးပြုပြီး ML အကြောင်းအရာများကို လေ့လာနိုင်ပါတယ်။

အဓိကအပိုင်းများ:
- **ပညာရေးအကြောင်းအရာ**: ML အကျဉ်းချုပ်၊ regression, classification, clustering, NLP, time series, reinforcement learning စသည်တို့ပါဝင်တဲ့ သင်ခန်းစာ ၂၆ ခု
- **မေးခွန်းအက်ပလီကေးရှင်း**: Vue.js အခြေခံထားတဲ့ မေးခွန်းအက်ပလီကေးရှင်း၊ သင်ခန်းစာမတိုင်မီနဲ့ပြီးနောက် အကဲဖြတ်မှုများ
- **ဘာသာစကားများအထောက်အပံ့**: GitHub Actions မှတဆင့် ၄၀+ ဘာသာစကားများကို အလိုအလျောက် ဘာသာပြန်
- **နှစ်မျိုးဘာသာစကားအထောက်အပံ့**: Python (Jupyter notebooks) နဲ့ R (R Markdown files) နှစ်မျိုးလုံးအတွက် သင်ခန်းစာများ
- **ပရောဂျက်အခြေခံသင်ယူမှု**: အကြောင်းအရာတစ်ခုစီမှာ လက်တွေ့လုပ်ငန်းများနဲ့ လေ့ကျင့်ခန်းများပါဝင်ပါတယ်

## Repository ဖွဲ့စည်းပုံ

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

သင်ခန်းစာ folder တစ်ခုစီမှာ အများအားဖြင့် အောက်ပါအရာများပါဝင်ပါတယ်:
- `README.md` - အဓိကသင်ခန်းစာအကြောင်းအရာ
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - ဖြေရှင်းချက်ကုဒ် (Python နဲ့ R ဗားရှင်းများ)
- `assignment.md` - လေ့ကျင့်ခန်း
- `images/` - ရုပ်ပုံအရင်းအမြစ်များ

## Setup Commands

### Python သင်ခန်းစာများအတွက်

အများဆုံးသင်ခန်းစာများမှာ Jupyter notebooks ကို အသုံးပြုပါတယ်။ လိုအပ်တဲ့ dependencies ကို install လုပ်ပါ:

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

### R သင်ခန်းစာများအတွက်

R သင်ခန်းစာများကို `solution/R/` folder တွေမှာ `.rmd` သို့မဟုတ် `.ipynb` ဖိုင်အဖြစ်တွေ့နိုင်ပါတယ်:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### မေးခွန်းအက်ပလီကေးရှင်းအတွက်

မေးခွန်းအက်ပလီကေးရှင်းကို `quiz-app/` directory မှာတွေ့နိုင်ပါတယ်:

```bash
cd quiz-app
npm install
```

### Documentation Site အတွက်

Documentation ကို local မှာ run လုပ်ရန်:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Development Workflow

### သင်ခန်းစာ Notebooks တွေကို အလုပ်လုပ်ခြင်း

1. သင်ခန်းစာ directory (ဥပမာ `2-Regression/1-Tools/`) ကို သွားပါ
2. Jupyter notebook ကို ဖွင့်ပါ:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. သင်ခန်းစာအကြောင်းအရာနဲ့ လေ့ကျင့်ခန်းများကို လုပ်ဆောင်ပါ
4. လိုအပ်ပါက `solution/` folder မှာ ဖြေရှင်းချက်များကို စစ်ဆေးပါ

### Python Development

- သင်ခန်းစာများမှာ စံပြ Python data science libraries ကို အသုံးပြုပါတယ်
- Jupyter notebooks ကို interactive learning အတွက် အသုံးပြုပါ
- ဖြေရှင်းချက်ကုဒ်ကို သင်ခန်းစာတစ်ခုစီရဲ့ `solution/` folder မှာ ရနိုင်ပါတယ်

### R Development

- R သင်ခန်းစာများကို `.rmd` format (R Markdown) အဖြစ် ရနိုင်ပါတယ်
- ဖြေရှင်းချက်များကို `solution/R/` subdirectories တွေမှာ ရနိုင်ပါတယ်
- RStudio သို့မဟုတ် Jupyter နဲ့ R kernel ကို အသုံးပြုပြီး R notebooks ကို run လုပ်ပါ

### မေးခွန်းအက်ပလီကေးရှင်း Development

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

## Testing Instructions

### မေးခွန်းအက်ပလီကေးရှင်း Testing

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**မှတ်ချက်**: ဒီဟာက ပညာရေးသင်ရိုးညွှန်း repository ဖြစ်ပါတယ်။ သင်ခန်းစာအကြောင်းအရာအတွက် automated tests မပါဝင်ပါဘူး။ Validation ကို အောက်ပါအတိုင်းလုပ်ဆောင်ပါ:
- သင်ခန်းစာလေ့ကျင့်ခန်းများကို ပြီးမြောက်စေခြင်း
- notebook cells တွေကို အောင်မြင်စွာ run လုပ်ခြင်း
- ဖြေရှင်းချက်များမှာ ရလဒ်ကို စစ်ဆေးခြင်း

## Code Style Guidelines

### Python Code
- PEP 8 style guidelines ကို လိုက်နာပါ
- ရှင်းလင်းပြီး ဖော်ပြချက်ပေးတဲ့ variable names ကို အသုံးပြုပါ
- ရှုပ်ထွေးတဲ့ လုပ်ဆောင်မှုများအတွက် comments ထည့်ပါ
- Jupyter notebooks တွေမှာ concept တွေကို ရှင်းပြတဲ့ markdown cells ပါဝင်သင့်ပါတယ်

### JavaScript/Vue.js (မေးခွန်းအက်ပလီကေးရှင်း)
- Vue.js style guide ကို လိုက်နာပါ
- ESLint configuration ကို `quiz-app/package.json` မှာ တွေ့နိုင်ပါတယ်
- `npm run lint` ကို run လုပ်ပြီး issue တွေကို auto-fix လုပ်ပါ

### Documentation
- Markdown ဖိုင်တွေကို ရှင်းလင်းပြီး ဖွဲ့စည်းမှုကောင်းစွာရှိသင့်ပါတယ်
- fenced code blocks တွေမှာ code examples ထည့်ပါ
- internal references အတွက် relative links ကို အသုံးပြုပါ
- ရှိပြီးသား formatting conventions ကို လိုက်နာပါ

## Build and Deployment

### မေးခွန်းအက်ပလီကေးရှင်း Deployment

မေးခွန်းအက်ပလီကေးရှင်းကို Azure Static Web Apps မှာ deploy လုပ်နိုင်ပါတယ်:

1. **လိုအပ်ချက်များ**:
   - Azure account
   - GitHub repository (fork လုပ်ပြီးသား)

2. **Azure မှာ Deploy လုပ်ခြင်း**:
   - Azure Static Web App resource ကို ဖန်တီးပါ
   - GitHub repository ကို ချိတ်ဆက်ပါ
   - app location ကို `/quiz-app` သတ်မှတ်ပါ
   - output location ကို `dist` သတ်မှတ်ပါ
   - Azure က GitHub Actions workflow ကို အလိုအလျောက် ဖန်တီးပေးပါမယ်

3. **GitHub Actions Workflow**:
   - Workflow ဖိုင်ကို `.github/workflows/azure-static-web-apps-*.yml` မှာ ဖန်တီးထားပါတယ်
   - main branch ကို push လုပ်တိုင်း အလိုအလျောက် build နဲ့ deploy လုပ်ပေးပါတယ်

### Documentation PDF

Documentation မှ PDF ကို ဖန်တီးရန်:

```bash
npm install
npm run convert
```

## Translation Workflow

**အရေးကြီး**: ဘာသာပြန်မှုများကို GitHub Actions မှတဆင့် Co-op Translator အသုံးပြုပြီး အလိုအလျောက်လုပ်ဆောင်ပါတယ်။

- ဘာသာပြန်မှုများကို main branch ကို ပြောင်းလဲမှုများ push လုပ်တိုင်း အလိုအလျောက်လုပ်ဆောင်ပါတယ်
- **ဘာသာပြန်မှုများကို ကိုယ်တိုင် လုပ်ဆောင်မထားပါနဲ့** - စနစ်က အလိုအလျောက်လုပ်ဆောင်ပါတယ်
- Workflow ကို `.github/workflows/co-op-translator.yml` မှာ သတ်မှတ်ထားပါတယ်
- Azure AI/OpenAI services ကို အသုံးပြုပြီး ဘာသာပြန်မှုများကို လုပ်ဆောင်ပါတယ်
- ၄၀+ ဘာသာစကားများကို ထောက်ပံ့ပါတယ်

## Contributing Guidelines

### Content Contributors အတွက်

1. **Repository ကို fork လုပ်ပြီး feature branch ဖန်တီးပါ**
2. **သင်ခန်းစာအကြောင်းအရာကို ပြောင်းလဲမှုများလုပ်ပါ** သင်ခန်းစာအသစ်ထည့်ခြင်း/အပ်ဒိတ်လုပ်ခြင်း
3. **ဘာသာပြန်ဖိုင်များကို မပြောင်းလဲပါနဲ့** - အလိုအလျောက်ဖန်တီးထားပါတယ်
4. **သင့်ကုဒ်ကို စမ်းသပ်ပါ** - notebook cells အားလုံးကို အောင်မြင်စွာ run လုပ်နိုင်ရမယ်
5. **လင့်ခ်များနဲ့ ရုပ်ပုံများကို စစ်ဆေးပါ** အလုပ်လုပ်ရမယ်
6. **ပြောင်းလဲမှုအကြောင်းအရာကို ရှင်းလင်းဖော်ပြထားတဲ့ pull request တင်ပါ**

### Pull Request Guidelines

- **Title format**: `[Section] Brief description of changes`
  - ဥပမာ: `[Regression] Fix typo in lesson 5`
  - ဥပမာ: `[Quiz-App] Update dependencies`
- **Submit လုပ်မတိုင်မီ**:
  - notebook cells အားလုံးကို error မရှိအောင် run လုပ်ပါ
  - quiz-app ကို ပြောင်းလဲမှုများရှိရင် `npm run lint` ကို run လုပ်ပါ
  - markdown formatting ကို စစ်ဆေးပါ
  - code examples အသစ်များကို စမ်းသပ်ပါ
- **PR မှာ ပါဝင်ရမယ်**:
  - ပြောင်းလဲမှုအကြောင်းအရာ
  - ပြောင်းလဲမှုအကြောင်းရင်း
  - UI ပြောင်းလဲမှုများရှိရင် screenshots
- **Code of Conduct**: [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md) ကို လိုက်နာပါ
- **CLA**: Contributor License Agreement ကို လက်မှတ်ထိုးရပါမယ်

## သင်ခန်းစာဖွဲ့စည်းပုံ

သင်ခန်းစာတစ်ခုစီမှာ အောက်ပါပုံစံကို လိုက်နာပါတယ်:

1. **Pre-lecture quiz** - အခြေခံအသိပညာကို စမ်းသပ်ခြင်း
2. **Lesson content** - ရေးသားထားတဲ့ လမ်းညွှန်ချက်များနဲ့ ရှင်းလင်းချက်များ
3. **Code demonstrations** - notebooks တွေမှာ လက်တွေ့နမူနာများ
4. **Knowledge checks** - နားလည်မှုကို စစ်ဆေးခြင်း
5. **Challenge** - ကိုယ်တိုင် concepts တွေကို အသုံးချခြင်း
6. **Assignment** - တိုးတက်တဲ့ လေ့ကျင့်ခန်း
7. **Post-lecture quiz** - သင်ယူမှုရလဒ်ကို အကဲဖြတ်ခြင်း

## Common Commands Reference

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

## အပိုဆောင်းအရင်းအမြစ်များ

- **Microsoft Learn Collection**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **မေးခွန်းအက်ပလီကေးရှင်း**: [Online quizzes](https://ff-quizzes.netlify.app/en/ml/)
- **ဆွေးနွေးမှုဘုတ်**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **ဗီဒီယိုလမ်းညွှန်များ**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## အဓိကနည်းပညာများ

- **Python**: ML သင်ခန်းစာများအတွက် အဓိကဘာသာစကား (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret ကို အသုံးပြုတဲ့ အခြား implementation
- **Jupyter**: Python သင်ခန်းစာများအတွက် interactive notebooks
- **R Markdown**: R သင်ခန်းစာများအတွက် စာရွက်များ
- **Vue.js 3**: မေးခွန်းအက်ပလီကေးရှင်း framework
- **Flask**: ML model deployment အတွက် web application framework
- **Docsify**: Documentation site generator
- **GitHub Actions**: CI/CD နဲ့ automated translations

## လုံခြုံရေးစဉ်းစားချက်များ

- **ကုဒ်ထဲမှာ လျှို့ဝှက်ချက်မပါဝင်ပါနဲ့**: API keys သို့မဟုတ် credentials ကို commit မလုပ်ပါနဲ့
- **Dependencies**: npm နဲ့ pip packages တွေကို အပ်ဒိတ်ထားပါ
- **User input**: Flask web app နမူနာတွေမှာ အခြေခံ input validation ပါဝင်ပါတယ်
- **Sensitive data**: နမူနာ datasets တွေက public နဲ့ sensitive မဟုတ်တဲ့အချက်အလက်တွေပါ

## Troubleshooting

### Jupyter Notebooks

- **Kernel ပြဿနာများ**: Kernel ကို restart လုပ်ပါ: Kernel → Restart
- **Import errors**: pip နဲ့လိုအပ်တဲ့ packages အားလုံးကို install လုပ်ပါ
- **Path ပြဿနာများ**: notebooks တွေကို သူ့ folder ထဲကနေ run လုပ်ပါ

### မေးခွန်းအက်ပလီကေးရှင်း

- **npm install မအောင်မြင်ပါ**: npm cache ကို ရှင်းပါ: `npm cache clean --force`
- **Port conflicts**: Port ကို ပြောင်းပါ: `npm run serve -- --port 8081`
- **Build errors**: `node_modules` ကို ဖျက်ပြီး ပြန် install လုပ်ပါ: `rm -rf node_modules && npm install`

### R သင်ခန်းစာများ

- **Package မတွေ့ပါ**: `install.packages("package-name")` နဲ့ install လုပ်ပါ
- **RMarkdown rendering**: rmarkdown package ကို install လုပ်ထားပါ
- **Kernel ပြဿနာများ**: Jupyter အတွက် IRkernel ကို install လုပ်ရနိုင်ပါတယ်

## ပရောဂျက်အထူးမှတ်ချက်များ

- ဒီဟာက **သင်ရိုးညွှန်း** ဖြစ်ပြီး production code မဟုတ်ပါ
- **ML concepts** ကို လက်တွေ့လေ့လာခြင်းအတွက် အဓိကထားပါတယ်
- နမူနာကုဒ်တွေက **ရှင်းလင်းမှု** ကို ဦးစားပေးထားပြီး optimization မဟုတ်ပါ
- သင်ခန်းစာအများစုက **ကိုယ်တိုင်လုပ်ဆောင်နိုင်တဲ့** အကြောင်းအရာများပါ
- **ဖြေရှင်းချက်များ** ရနိုင်ပေမယ့် သင်ခန်းစာလေ့ကျင့်ခန်းများကို အရင်လုပ်ဆောင်သင့်ပါတယ်
- Repository က **Docsify** ကို အသုံးပြုပြီး build step မလိုအပ်ပါ
- **Sketchnotes** တွေက concept တွေကို visual summary ပေးပါတယ်
- **Multi-language support** က အကြောင်းအရာကို ကမ္ဘာတစ်ဝှမ်း အသုံးပြုနိုင်အောင် လုပ်ဆောင်ပါတယ်

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာတရ အရင်းအမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များကို အသုံးပြု၍ ဘာသာပြန်ခြင်းကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအလွတ်များ သို့မဟုတ် အနားယူမှားမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။