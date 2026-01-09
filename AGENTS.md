# AGENTS.md

## Project Overview

This is **Machine Learning for Beginners**, a comprehensive 12-week, 26-lesson curriculum covering classic machine learning concepts using Python (primarily with Scikit-learn) and R. The repository is designed as a self-paced learning resource with hands-on projects, quizzes, and assignments. Each lesson explores ML concepts through real-world data from various cultures and regions worldwide.

Key components:
- **Educational Content**: 26 lessons covering introduction to ML, regression, classification, clustering, NLP, time series, and reinforcement learning
- **Quiz Application**: Vue.js-based quiz app with pre- and post-lesson assessments
- **Multi-language Support**: Automated translations to 40+ languages via GitHub Actions
- **Dual Language Support**: Lessons available in both Python (Jupyter notebooks) and R (R Markdown files)
- **Project-Based Learning**: Each topic includes practical projects and assignments

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

Each lesson folder typically contains:
- `README.md` - Main lesson content
- `notebook.ipynb` - Python Jupyter notebook
- `solution/` - Solution code (Python and R versions)
- `assignment.md` - Practice exercises
- `images/` - Visual resources

## Setup Commands

### For Python Lessons

Most lessons use Jupyter notebooks. Install required dependencies:

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

R lessons are in `solution/R/` folders as `.rmd` or `.ipynb` files:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### For Quiz Application

The quiz app is a Vue.js application located in the `quiz-app/` directory:

```bash
cd quiz-app
npm install
```

### For Documentation Site

To run the documentation locally:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Development Workflow

### Working with Lesson Notebooks

1. Navigate to the lesson directory (e.g., `2-Regression/1-Tools/`)
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Work through the lesson content and exercises
4. Check solutions in the `solution/` folder if needed

### Python Development

- Lessons use standard Python data science libraries
- Jupyter notebooks for interactive learning
- Solution code available in each lesson's `solution/` folder

### R Development

- R lessons are in `.rmd` format (R Markdown)
- Solutions located in `solution/R/` subdirectories
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

**Note**: This is primarily an educational curriculum repository. There are no automated tests for lesson content. Validation is done through:
- Completing lesson exercises
- Running notebook cells successfully
- Checking output against expected results in solutions

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guidelines
- Use clear, descriptive variable names
- Include comments for complex operations
- Jupyter notebooks should have markdown cells explaining concepts

### JavaScript/Vue.js (Quiz App)
- Follows Vue.js style guide
- ESLint configuration in `quiz-app/package.json`
- Run `npm run lint` to check and auto-fix issues

### Documentation
- Markdown files should be clear and well-structured
- Include code examples in fenced code blocks
- Use relative links for internal references
- Follow existing formatting conventions

## Build and Deployment

### Quiz Application Deployment

The quiz app can be deployed to Azure Static Web Apps:

1. **Prerequisites**:
   - Azure account
   - GitHub repository (already forked)

2. **Deploy to Azure**:
   - Create Azure Static Web App resource
   - Connect to GitHub repository
   - Set app location: `/quiz-app`
   - Set output location: `dist`
   - Azure automatically creates GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Workflow file created at `.github/workflows/azure-static-web-apps-*.yml`
   - Automatically builds and deploys on push to main branch

### Documentation PDF

Generate PDF from documentation:

```bash
npm install
npm run convert
```

## Translation Workflow

**Important**: Translations are automated via GitHub Actions using Co-op Translator.

- Translations are auto-generated when changes are pushed to `main` branch
- **DO NOT manually translate content** - the system handles this
- Workflow defined in `.github/workflows/co-op-translator.yml`
- Uses Azure AI/OpenAI services for translation
- Supports 40+ languages

## Contributing Guidelines

### For Content Contributors

1. **Fork the repository** and create a feature branch
2. **Make changes to lesson content** if adding/updating lessons
3. **Do not modify translated files** - they are auto-generated
4. **Test your code** - ensure all notebook cells run successfully
5. **Verify links and images** work correctly
6. **Submit a pull request** with clear description

### Pull Request Guidelines

- **Title format**: `[Section] Brief description of changes`
  - Example: `[Regression] Fix typo in lesson 5`
  - Example: `[Quiz-App] Update dependencies`
- **Before submitting**:
  - Ensure all notebook cells execute without errors
  - Run `npm run lint` if modifying quiz-app
  - Verify markdown formatting
  - Test any new code examples
- **PR must include**:
  - Description of changes
  - Reason for changes
  - Screenshots if UI changes
- **Code of Conduct**: Follow the [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: You will need to sign the Contributor License Agreement

## Lesson Structure

Each lesson follows a consistent pattern:

1. **Pre-lecture quiz** - Test baseline knowledge
2. **Lesson content** - Written instructions and explanations
3. **Code demonstrations** - Hands-on examples in notebooks
4. **Knowledge checks** - Verify understanding throughout
5. **Challenge** - Apply concepts independently
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

- **Python**: Primary language for ML lessons (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Alternative implementation using tidyverse, tidymodels, caret
- **Jupyter**: Interactive notebooks for Python lessons
- **R Markdown**: Documents for R lessons
- **Vue.js 3**: Quiz application framework
- **Flask**: Web application framework for ML model deployment
- **Docsify**: Documentation site generator
- **GitHub Actions**: CI/CD and automated translations

## Security Considerations

- **No secrets in code**: Never commit API keys or credentials
- **Dependencies**: Keep npm and pip packages updated
- **User input**: Flask web app examples include basic input validation
- **Sensitive data**: Example datasets are public and non-sensitive

## Troubleshooting

### Jupyter Notebooks

- **Kernel issues**: Restart kernel if cells hang: Kernel → Restart
- **Import errors**: Ensure all required packages are installed with pip
- **Path issues**: Run notebooks from their containing directory

### Quiz Application

- **npm install fails**: Clear npm cache: `npm cache clean --force`
- **Port conflicts**: Change port with: `npm run serve -- --port 8081`
- **Build errors**: Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

### R Lessons

- **Package not found**: Install with: `install.packages("package-name")`
- **RMarkdown rendering**: Ensure rmarkdown package is installed
- **Kernel issues**: May need to install IRkernel for Jupyter

## Project-Specific Notes

- This is primarily a **learning curriculum**, not production code
- Focus is on **understanding ML concepts** through hands-on practice
- Code examples prioritize **clarity over optimization**
- Most lessons are **self-contained** and can be completed independently
- **Solutions provided** but learners should attempt exercises first
- Repository uses **Docsify** for web documentation without build step
- **Sketchnotes** provide visual summaries of concepts
- **Multi-language support** makes content globally accessible
