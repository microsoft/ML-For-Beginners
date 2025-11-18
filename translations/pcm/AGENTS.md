<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-11-18T18:09:06+00:00",
  "source_file": "AGENTS.md",
  "language_code": "pcm"
}
-->
# AGENTS.md

## Project Overview

Dis na **Machine Learning for Beginners**, na 12-week, 26-lesson curriculum wey cover classic machine learning concepts using Python (mainly with Scikit-learn) and R. Dis repository na self-paced learning resource wey get hands-on projects, quizzes, and assignments. Each lesson dey explore ML concepts with real-world data from different cultures and regions for di world.

Key components:
- **Educational Content**: 26 lessons wey cover intro to ML, regression, classification, clustering, NLP, time series, and reinforcement learning
- **Quiz Application**: Vue.js-based quiz app wey get pre- and post-lesson assessments
- **Multi-language Support**: Automated translations to 40+ languages via GitHub Actions
- **Dual Language Support**: Lessons dey available for both Python (Jupyter notebooks) and R (R Markdown files)
- **Project-Based Learning**: Each topic get practical projects and assignments

## Repository Structure

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

Each lesson folder dey usually contain:
- `README.md` - Main lesson content
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - Solution code (Python and R versions)
- `assignment.md` - Practice exercises
- `images/` - Visual resources

## Setup Commands

### For Python Lessons

Most lessons dey use Jupyter notebooks. Install di required dependencies:

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

### For R Lessons

R lessons dey inside `solution/R/` folders as `.rmd` or `.ipynb` files:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### For Quiz Application

Di quiz app na Vue.js application wey dey inside di `quiz-app/` directory:

```bash
cd quiz-app
npm install
```

### For Documentation Site

To run di documentation locally:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Development Workflow

### Working with Lesson Notebooks

1. Go to di lesson directory (e.g., `2-Regression/1-Tools/`)
2. Open di Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Work through di lesson content and exercises
4. Check solutions for di `solution/` folder if you need am

### Python Development

- Lessons dey use standard Python data science libraries
- Jupyter notebooks dey for interactive learning
- Solution code dey available for each lesson `solution/` folder

### R Development

- R lessons dey for `.rmd` format (R Markdown)
- Solutions dey for `solution/R/` subdirectories
- Use RStudio or Jupyter with R kernel to run R notebooks

### Quiz Application Development

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

### Quiz Application Testing

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Note**: Dis na mainly educational curriculum repository. E no get automated tests for lesson content. Validation dey happen through:
- Completing lesson exercises
- Running notebook cells successfully
- Checking output against expected results for solutions

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use clear, descriptive variable names
- Add comments for complex operations
- Jupyter notebooks suppose get markdown cells wey explain concepts

### JavaScript/Vue.js (Quiz App)
- Follow Vue.js style guide
- ESLint configuration dey for `quiz-app/package.json`
- Run `npm run lint` to check and auto-fix issues

### Documentation
- Markdown files suppose dey clear and well-structured
- Add code examples for fenced code blocks
- Use relative links for internal references
- Follow di existing formatting conventions

## Build and Deployment

### Quiz Application Deployment

Di quiz app fit deploy to Azure Static Web Apps:

1. **Prerequisites**:
   - Azure account
   - GitHub repository (wey you don fork already)

2. **Deploy to Azure**:
   - Create Azure Static Web App resource
   - Connect to GitHub repository
   - Set app location: `/quiz-app`
   - Set output location: `dist`
   - Azure go automatically create GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Workflow file dey created for `.github/workflows/azure-static-web-apps-*.yml`
   - E dey automatically build and deploy when you push to main branch

### Documentation PDF

Generate PDF from documentation:

```bash
npm install
npm run convert
```

## Translation Workflow

**Important**: Translations dey automated via GitHub Actions using Co-op Translator.

- Translations dey auto-generated when changes dey pushed to `main` branch
- **NO manually translate content** - di system go handle am
- Workflow dey defined for `.github/workflows/co-op-translator.yml`
- E dey use Azure AI/OpenAI services for translation
- E support 40+ languages

## Contributing Guidelines

### For Content Contributors

1. **Fork di repository** and create feature branch
2. **Make changes to lesson content** if you dey add/update lessons
3. **No touch translated files** - dem dey auto-generated
4. **Test your code** - make sure all notebook cells dey run successfully
5. **Verify links and images** dey work well
6. **Submit pull request** with clear description

### Pull Request Guidelines

- **Title format**: `[Section] Brief description of changes`
  - Example: `[Regression] Fix typo for lesson 5`
  - Example: `[Quiz-App] Update dependencies`
- **Before you submit**:
  - Make sure all notebook cells dey execute without errors
  - Run `npm run lint` if you dey modify quiz-app
  - Verify markdown formatting
  - Test any new code examples
- **PR suppose include**:
  - Description of changes
  - Reason for changes
  - Screenshots if UI changes dey
- **Code of Conduct**: Follow di [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: You go need sign di Contributor License Agreement

## Lesson Structure

Each lesson dey follow consistent pattern:

1. **Pre-lecture quiz** - Test baseline knowledge
2. **Lesson content** - Written instructions and explanations
3. **Code demonstrations** - Hands-on examples for notebooks
4. **Knowledge checks** - Verify understanding throughout
5. **Challenge** - Apply concepts by yourself
6. **Assignment** - Extended practice
7. **Post-lecture quiz** - Assess learning outcomes

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

## Additional Resources

- **Microsoft Learn Collection**: [ML for Beginners modules](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Quiz App**: [Online quizzes](https://ff-quizzes.netlify.app/en/ml/)
- **Discussion Board**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video Walkthroughs**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## Key Technologies

- **Python**: Main language for ML lessons (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternative implementation using tidyverse, tidymodels, caret
- **Jupyter**: Interactive notebooks for Python lessons
- **R Markdown**: Documents for R lessons
- **Vue.js 3**: Quiz application framework
- **Flask**: Web application framework for ML model deployment
- **Docsify**: Documentation site generator
- **GitHub Actions**: CI/CD and automated translations

## Security Considerations

- **No secrets for code**: No ever commit API keys or credentials
- **Dependencies**: Keep npm and pip packages updated
- **User input**: Flask web app examples get basic input validation
- **Sensitive data**: Example datasets dey public and no sensitive

## Troubleshooting

### Jupyter Notebooks

- **Kernel issues**: Restart kernel if cells dey hang: Kernel → Restart
- **Import errors**: Make sure all required packages dey installed with pip
- **Path issues**: Run notebooks from di directory wey dem dey

### Quiz Application

- **npm install fails**: Clear npm cache: `npm cache clean --force`
- **Port conflicts**: Change port with: `npm run serve -- --port 8081`
- **Build errors**: Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

### R Lessons

- **Package no dey found**: Install with: `install.packages("package-name")`
- **RMarkdown rendering**: Make sure rmarkdown package dey installed
- **Kernel issues**: You fit need install IRkernel for Jupyter

## Project-Specific Notes

- Dis na mainly **learning curriculum**, no be production code
- Focus dey on **understanding ML concepts** through hands-on practice
- Code examples dey prioritize **clarity over optimization**
- Most lessons dey **self-contained** and you fit complete dem independently
- **Solutions dey provided** but learners suppose try di exercises first
- Repository dey use **Docsify** for web documentation without build step
- **Sketchnotes** dey provide visual summaries of concepts
- **Multi-language support** dey make content globally accessible

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator) do di translation. Even as we dey try make am accurate, abeg sabi say automated translations fit get mistake or no dey correct well. Di original dokyument for di native language na di main source wey you go trust. For important mata, na beta make you use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->