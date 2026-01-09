<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:56:28+00:00",
  "source_file": "AGENTS.md",
  "language_code": "fr"
}
-->
# AGENTS.md

## Aperçu du projet

Voici **Machine Learning pour les débutants**, un programme complet de 12 semaines et 26 leçons couvrant les concepts classiques de l'apprentissage automatique en utilisant Python (principalement avec Scikit-learn) et R. Ce dépôt est conçu comme une ressource d'apprentissage autonome avec des projets pratiques, des quiz et des devoirs. Chaque leçon explore les concepts de l'IA à travers des données réelles provenant de diverses cultures et régions du monde.

Principaux éléments :
- **Contenu éducatif** : 26 leçons couvrant l'introduction à l'IA, la régression, la classification, le clustering, le NLP, les séries temporelles et l'apprentissage par renforcement
- **Application de quiz** : Application de quiz basée sur Vue.js avec des évaluations avant et après les leçons
- **Support multilingue** : Traductions automatiques dans plus de 40 langues via GitHub Actions
- **Support bilingue** : Leçons disponibles en Python (notebooks Jupyter) et en R (fichiers R Markdown)
- **Apprentissage basé sur des projets** : Chaque sujet inclut des projets pratiques et des devoirs

## Structure du dépôt

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

Chaque dossier de leçon contient généralement :
- `README.md` - Contenu principal de la leçon
- `notebook.ipynb` - Notebook Jupyter en Python
- `solution/` - Code solution (versions Python et R)
- `assignment.md` - Exercices pratiques
- `images/` - Ressources visuelles

## Commandes d'installation

### Pour les leçons en Python

La plupart des leçons utilisent des notebooks Jupyter. Installez les dépendances nécessaires :

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

### Pour les leçons en R

Les leçons en R se trouvent dans les dossiers `solution/R/` sous forme de fichiers `.rmd` ou `.ipynb` :

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Pour l'application de quiz

L'application de quiz est une application Vue.js située dans le répertoire `quiz-app/` :

```bash
cd quiz-app
npm install
```

### Pour le site de documentation

Pour exécuter la documentation localement :

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Flux de travail de développement

### Travailler avec les notebooks de leçon

1. Accédez au répertoire de la leçon (par exemple, `2-Regression/1-Tools/`)
2. Ouvrez le notebook Jupyter :
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Parcourez le contenu de la leçon et les exercices
4. Consultez les solutions dans le dossier `solution/` si nécessaire

### Développement en Python

- Les leçons utilisent les bibliothèques standards de science des données en Python
- Notebooks Jupyter pour un apprentissage interactif
- Code solution disponible dans le dossier `solution/` de chaque leçon

### Développement en R

- Les leçons en R sont au format `.rmd` (R Markdown)
- Solutions situées dans les sous-dossiers `solution/R/`
- Utilisez RStudio ou Jupyter avec le noyau R pour exécuter les notebooks R

### Développement de l'application de quiz

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

## Instructions de test

### Test de l'application de quiz

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Remarque** : Il s'agit principalement d'un dépôt de programme éducatif. Il n'y a pas de tests automatisés pour le contenu des leçons. La validation se fait par :
- La réalisation des exercices des leçons
- L'exécution réussie des cellules des notebooks
- La vérification des résultats par rapport aux solutions attendues

## Directives de style de code

### Code Python
- Suivez les directives de style PEP 8
- Utilisez des noms de variables clairs et descriptifs
- Ajoutez des commentaires pour les opérations complexes
- Les notebooks Jupyter doivent inclure des cellules markdown expliquant les concepts

### JavaScript/Vue.js (Application de quiz)
- Suivez le guide de style Vue.js
- Configuration ESLint dans `quiz-app/package.json`
- Exécutez `npm run lint` pour vérifier et corriger automatiquement les problèmes

### Documentation
- Les fichiers Markdown doivent être clairs et bien structurés
- Incluez des exemples de code dans des blocs de code délimités
- Utilisez des liens relatifs pour les références internes
- Suivez les conventions de formatage existantes

## Construction et déploiement

### Déploiement de l'application de quiz

L'application de quiz peut être déployée sur Azure Static Web Apps :

1. **Prérequis** :
   - Compte Azure
   - Dépôt GitHub (déjà forké)

2. **Déployer sur Azure** :
   - Créez une ressource Azure Static Web App
   - Connectez-vous au dépôt GitHub
   - Définissez l'emplacement de l'application : `/quiz-app`
   - Définissez l'emplacement de sortie : `dist`
   - Azure crée automatiquement un workflow GitHub Actions

3. **Workflow GitHub Actions** :
   - Fichier de workflow créé dans `.github/workflows/azure-static-web-apps-*.yml`
   - Construction et déploiement automatiques lors d'un push sur la branche principale

### Documentation PDF

Générez un PDF à partir de la documentation :

```bash
npm install
npm run convert
```

## Flux de travail de traduction

**Important** : Les traductions sont automatisées via GitHub Actions en utilisant Co-op Translator.

- Les traductions sont générées automatiquement lorsque des modifications sont poussées sur la branche `main`
- **NE PAS traduire manuellement le contenu** - le système s'en charge
- Workflow défini dans `.github/workflows/co-op-translator.yml`
- Utilise les services Azure AI/OpenAI pour la traduction
- Prend en charge plus de 40 langues

## Directives de contribution

### Pour les contributeurs de contenu

1. **Forkez le dépôt** et créez une branche de fonctionnalité
2. **Apportez des modifications au contenu des leçons** si vous ajoutez ou mettez à jour des leçons
3. **Ne modifiez pas les fichiers traduits** - ils sont générés automatiquement
4. **Testez votre code** - assurez-vous que toutes les cellules des notebooks s'exécutent correctement
5. **Vérifiez les liens et les images** pour s'assurer qu'ils fonctionnent
6. **Soumettez une pull request** avec une description claire

### Directives pour les pull requests

- **Format du titre** : `[Section] Brève description des modifications`
  - Exemple : `[Regression] Correction d'une faute de frappe dans la leçon 5`
  - Exemple : `[Quiz-App] Mise à jour des dépendances`
- **Avant de soumettre** :
  - Assurez-vous que toutes les cellules des notebooks s'exécutent sans erreurs
  - Exécutez `npm run lint` si vous modifiez l'application de quiz
  - Vérifiez le formatage Markdown
  - Testez tout nouvel exemple de code
- **La PR doit inclure** :
  - Description des modifications
  - Raison des modifications
  - Captures d'écran si des modifications de l'interface utilisateur sont apportées
- **Code de conduite** : Suivez le [Code de conduite Open Source de Microsoft](CODE_OF_CONDUCT.md)
- **CLA** : Vous devrez signer l'accord de licence de contributeur

## Structure des leçons

Chaque leçon suit un modèle cohérent :

1. **Quiz pré-lecture** - Testez les connaissances de base
2. **Contenu de la leçon** - Instructions et explications écrites
3. **Démonstrations de code** - Exemples pratiques dans les notebooks
4. **Vérifications des connaissances** - Vérifiez la compréhension tout au long
5. **Défi** - Appliquez les concepts de manière autonome
6. **Devoir** - Pratique approfondie
7. **Quiz post-lecture** - Évaluez les résultats d'apprentissage

## Référence des commandes courantes

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

## Ressources supplémentaires

- **Collection Microsoft Learn** : [Modules ML pour les débutants](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Application de quiz** : [Quiz en ligne](https://ff-quizzes.netlify.app/en/ml/)
- **Forum de discussion** : [Discussions GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Tutoriels vidéo** : [Playlist YouTube](https://aka.ms/ml-beginners-videos)

## Technologies clés

- **Python** : Langage principal pour les leçons d'IA (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R** : Implémentation alternative utilisant tidyverse, tidymodels, caret
- **Jupyter** : Notebooks interactifs pour les leçons en Python
- **R Markdown** : Documents pour les leçons en R
- **Vue.js 3** : Framework de l'application de quiz
- **Flask** : Framework d'application web pour le déploiement de modèles IA
- **Docsify** : Générateur de site de documentation
- **GitHub Actions** : CI/CD et traductions automatisées

## Considérations de sécurité

- **Pas de secrets dans le code** : Ne jamais inclure de clés API ou de identifiants
- **Dépendances** : Maintenez les packages npm et pip à jour
- **Entrées utilisateur** : Les exemples d'applications web Flask incluent une validation de base des entrées
- **Données sensibles** : Les ensembles de données d'exemple sont publics et non sensibles

## Résolution de problèmes

### Notebooks Jupyter

- **Problèmes de noyau** : Redémarrez le noyau si les cellules se bloquent : Kernel → Restart
- **Erreurs d'importation** : Assurez-vous que tous les packages requis sont installés avec pip
- **Problèmes de chemin** : Exécutez les notebooks depuis leur répertoire contenant

### Application de quiz

- **Échec de npm install** : Nettoyez le cache npm : `npm cache clean --force`
- **Conflits de port** : Changez le port avec : `npm run serve -- --port 8081`
- **Erreurs de construction** : Supprimez `node_modules` et réinstallez : `rm -rf node_modules && npm install`

### Leçons en R

- **Package introuvable** : Installez avec : `install.packages("nom-du-package")`
- **RMarkdown non rendu** : Assurez-vous que le package rmarkdown est installé
- **Problèmes de noyau** : Vous devrez peut-être installer IRkernel pour Jupyter

## Notes spécifiques au projet

- Il s'agit principalement d'un **programme éducatif**, et non de code de production
- L'objectif est de **comprendre les concepts de l'IA** grâce à une pratique concrète
- Les exemples de code privilégient **la clarté plutôt que l'optimisation**
- La plupart des leçons sont **autonomes** et peuvent être réalisées indépendamment
- **Solutions fournies**, mais les apprenants doivent d'abord tenter les exercices
- Le dépôt utilise **Docsify** pour la documentation web sans étape de construction
- Les **Sketchnotes** offrent des résumés visuels des concepts
- Le **support multilingue** rend le contenu accessible à l'échelle mondiale

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.