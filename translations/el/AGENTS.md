<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:08:41+00:00",
  "source_file": "AGENTS.md",
  "language_code": "el"
}
-->
# AGENTS.md

## Επισκόπηση Έργου

Αυτό είναι το **Machine Learning για Αρχάριους**, ένα ολοκληρωμένο πρόγραμμα σπουδών 12 εβδομάδων με 26 μαθήματα που καλύπτουν κλασικές έννοιες μηχανικής μάθησης χρησιμοποιώντας Python (κυρίως με Scikit-learn) και R. Το αποθετήριο έχει σχεδιαστεί ως πόρος αυτοδιδασκαλίας με πρακτικά έργα, κουίζ και ασκήσεις. Κάθε μάθημα εξερευνά έννοιες ML μέσω δεδομένων από πραγματικό κόσμο που προέρχονται από διάφορους πολιτισμούς και περιοχές παγκοσμίως.

Κύρια στοιχεία:
- **Εκπαιδευτικό Περιεχόμενο**: 26 μαθήματα που καλύπτουν εισαγωγή στη ML, παλινδρόμηση, ταξινόμηση, ομαδοποίηση, NLP, χρονοσειρές και ενισχυτική μάθηση
- **Εφαρμογή Κουίζ**: Εφαρμογή κουίζ βασισμένη στο Vue.js με αξιολογήσεις πριν και μετά το μάθημα
- **Υποστήριξη Πολλών Γλωσσών**: Αυτόματες μεταφράσεις σε 40+ γλώσσες μέσω GitHub Actions
- **Διπλή Υποστήριξη Γλωσσών**: Μαθήματα διαθέσιμα τόσο σε Python (Jupyter notebooks) όσο και σε R (αρχεία R Markdown)
- **Μάθηση μέσω Έργων**: Κάθε θέμα περιλαμβάνει πρακτικά έργα και ασκήσεις

## Δομή Αποθετηρίου

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

Κάθε φάκελος μαθήματος περιέχει συνήθως:
- `README.md` - Κύριο περιεχόμενο μαθήματος
- `notebook.ipynb` - Jupyter notebook για Python
- `solution/` - Κώδικας λύσεων (εκδόσεις Python και R)
- `assignment.md` - Ασκήσεις πρακτικής
- `images/` - Οπτικοί πόροι

## Εντολές Ρύθμισης

### Για Μαθήματα Python

Τα περισσότερα μαθήματα χρησιμοποιούν Jupyter notebooks. Εγκαταστήστε τις απαιτούμενες εξαρτήσεις:

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

### Για Μαθήματα R

Τα μαθήματα R βρίσκονται στους φακέλους `solution/R/` ως `.rmd` ή `.ipynb` αρχεία:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Για Εφαρμογή Κουίζ

Η εφαρμογή κουίζ είναι μια εφαρμογή Vue.js που βρίσκεται στον φάκελο `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Για Ιστότοπο Τεκμηρίωσης

Για να εκτελέσετε την τεκμηρίωση τοπικά:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Ροή Εργασίας Ανάπτυξης

### Εργασία με Notebooks Μαθημάτων

1. Μεταβείτε στον φάκελο μαθήματος (π.χ., `2-Regression/1-Tools/`)
2. Ανοίξτε το Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Εργαστείτε με το περιεχόμενο και τις ασκήσεις του μαθήματος
4. Ελέγξτε τις λύσεις στον φάκελο `solution/` αν χρειαστεί

### Ανάπτυξη Python

- Τα μαθήματα χρησιμοποιούν τυπικές βιβλιοθήκες επιστήμης δεδομένων Python
- Jupyter notebooks για διαδραστική μάθηση
- Κώδικας λύσεων διαθέσιμος στον φάκελο `solution/` κάθε μαθήματος

### Ανάπτυξη R

- Τα μαθήματα R είναι σε μορφή `.rmd` (R Markdown)
- Λύσεις βρίσκονται στους υποφακέλους `solution/R/`
- Χρησιμοποιήστε RStudio ή Jupyter με πυρήνα R για να εκτελέσετε notebooks R

### Ανάπτυξη Εφαρμογής Κουίζ

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

## Οδηγίες Δοκιμών

### Δοκιμή Εφαρμογής Κουίζ

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Σημείωση**: Αυτό είναι κυρίως ένα αποθετήριο εκπαιδευτικού προγράμματος. Δεν υπάρχουν αυτοματοποιημένες δοκιμές για το περιεχόμενο των μαθημάτων. Η επικύρωση γίνεται μέσω:
- Ολοκλήρωσης ασκήσεων μαθήματος
- Επιτυχούς εκτέλεσης κελιών notebook
- Ελέγχου αποτελεσμάτων σε σχέση με τις αναμενόμενες λύσεις

## Οδηγίες Στυλ Κώδικα

### Κώδικας Python
- Ακολουθήστε τις οδηγίες στυλ PEP 8
- Χρησιμοποιήστε σαφή, περιγραφικά ονόματα μεταβλητών
- Συμπεριλάβετε σχόλια για σύνθετες λειτουργίες
- Τα Jupyter notebooks πρέπει να έχουν markdown κελιά που εξηγούν έννοιες

### JavaScript/Vue.js (Εφαρμογή Κουίζ)
- Ακολουθεί τον οδηγό στυλ Vue.js
- Ρύθμιση ESLint στο `quiz-app/package.json`
- Εκτελέστε `npm run lint` για έλεγχο και αυτόματη διόρθωση προβλημάτων

### Τεκμηρίωση
- Τα αρχεία Markdown πρέπει να είναι σαφή και καλά δομημένα
- Συμπεριλάβετε παραδείγματα κώδικα σε περιφραγμένα μπλοκ κώδικα
- Χρησιμοποιήστε σχετικούς συνδέσμους για εσωτερικές αναφορές
- Ακολουθήστε τις υπάρχουσες συμβάσεις μορφοποίησης

## Δημιουργία και Ανάπτυξη

### Ανάπτυξη Εφαρμογής Κουίζ

Η εφαρμογή κουίζ μπορεί να αναπτυχθεί σε Azure Static Web Apps:

1. **Προαπαιτούμενα**:
   - Λογαριασμός Azure
   - Αποθετήριο GitHub (ήδη forked)

2. **Ανάπτυξη στο Azure**:
   - Δημιουργήστε πόρο Azure Static Web App
   - Συνδέστε το αποθετήριο GitHub
   - Ορίστε τοποθεσία εφαρμογής: `/quiz-app`
   - Ορίστε τοποθεσία εξόδου: `dist`
   - Το Azure δημιουργεί αυτόματα GitHub Actions workflow

3. **GitHub Actions Workflow**:
   - Το αρχείο workflow δημιουργείται στο `.github/workflows/azure-static-web-apps-*.yml`
   - Αυτόματα δημιουργεί και αναπτύσσει με push στον κύριο κλάδο

### PDF Τεκμηρίωσης

Δημιουργήστε PDF από την τεκμηρίωση:

```bash
npm install
npm run convert
```

## Ροή Εργασίας Μετάφρασης

**Σημαντικό**: Οι μεταφράσεις γίνονται αυτόματα μέσω GitHub Actions χρησιμοποιώντας το Co-op Translator.

- Οι μεταφράσεις δημιουργούνται αυτόματα όταν γίνονται αλλαγές στον κλάδο `main`
- **ΜΗΝ μεταφράζετε χειροκίνητα το περιεχόμενο** - το σύστημα το διαχειρίζεται
- Το workflow ορίζεται στο `.github/workflows/co-op-translator.yml`
- Χρησιμοποιεί υπηρεσίες Azure AI/OpenAI για μετάφραση
- Υποστηρίζει 40+ γλώσσες

## Οδηγίες Συνεισφοράς

### Για Συνεισφέροντες Περιεχομένου

1. **Κάντε fork το αποθετήριο** και δημιουργήστε έναν κλάδο χαρακτηριστικών
2. **Κάντε αλλαγές στο περιεχόμενο μαθημάτων** αν προσθέτετε/ενημερώνετε μαθήματα
3. **Μην τροποποιείτε αρχεία μεταφρασμένα** - αυτά δημιουργούνται αυτόματα
4. **Δοκιμάστε τον κώδικα σας** - βεβαιωθείτε ότι όλα τα κελιά notebook εκτελούνται επιτυχώς
5. **Επαληθεύστε συνδέσμους και εικόνες** ότι λειτουργούν σωστά
6. **Υποβάλετε ένα pull request** με σαφή περιγραφή

### Οδηγίες Pull Request

- **Μορφή τίτλου**: `[Ενότητα] Σύντομη περιγραφή αλλαγών`
  - Παράδειγμα: `[Regression] Διόρθωση τυπογραφικού λάθους στο μάθημα 5`
  - Παράδειγμα: `[Quiz-App] Ενημέρωση εξαρτήσεων`
- **Πριν την υποβολή**:
  - Βεβαιωθείτε ότι όλα τα κελιά notebook εκτελούνται χωρίς σφάλματα
  - Εκτελέστε `npm run lint` αν τροποποιείτε το quiz-app
  - Επαληθεύστε τη μορφοποίηση markdown
  - Δοκιμάστε οποιαδήποτε νέα παραδείγματα κώδικα
- **Το PR πρέπει να περιλαμβάνει**:
  - Περιγραφή αλλαγών
  - Λόγο αλλαγών
  - Στιγμιότυπα οθόνης αν υπάρχουν αλλαγές UI
- **Κώδικας Συμπεριφοράς**: Ακολουθήστε τον [Κώδικα Συμπεριφοράς Ανοιχτού Κώδικα της Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Θα χρειαστεί να υπογράψετε τη Συμφωνία Άδειας Χρήσης Συνεισφέροντος

## Δομή Μαθήματος

Κάθε μάθημα ακολουθεί ένα συνεπές μοτίβο:

1. **Κουίζ πριν το μάθημα** - Δοκιμή βασικών γνώσεων
2. **Περιεχόμενο μαθήματος** - Γραπτές οδηγίες και εξηγήσεις
3. **Επιδείξεις κώδικα** - Πρακτικά παραδείγματα σε notebooks
4. **Έλεγχοι γνώσεων** - Επαλήθευση κατανόησης καθ' όλη τη διάρκεια
5. **Πρόκληση** - Εφαρμογή εννοιών ανεξάρτητα
6. **Άσκηση** - Εκτεταμένη πρακτική
7. **Κουίζ μετά το μάθημα** - Αξιολόγηση μαθησιακών αποτελεσμάτων

## Αναφορά Κοινών Εντολών

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

## Πρόσθετοι Πόροι

- **Συλλογή Microsoft Learn**: [Μαθήματα ML για Αρχάριους](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Εφαρμογή Κουίζ**: [Κουίζ online](https://ff-quizzes.netlify.app/en/ml/)
- **Πίνακας Συζητήσεων**: [Συζητήσεις GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Βίντεο Οδηγίες**: [Λίστα αναπαραγωγής YouTube](https://aka.ms/ml-beginners-videos)

## Κύριες Τεχνολογίες

- **Python**: Κύρια γλώσσα για μαθήματα ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Εναλλακτική υλοποίηση χρησιμοποιώντας tidyverse, tidymodels, caret
- **Jupyter**: Διαδραστικά notebooks για μαθήματα Python
- **R Markdown**: Έγγραφα για μαθήματα R
- **Vue.js 3**: Πλαίσιο εφαρμογής κουίζ
- **Flask**: Πλαίσιο εφαρμογής ιστού για ανάπτυξη μοντέλων ML
- **Docsify**: Γεννήτρια ιστότοπου τεκμηρίωσης
- **GitHub Actions**: CI/CD και αυτοματοποιημένες μεταφράσεις

## Σκέψεις Ασφαλείας

- **Χωρίς μυστικά στον κώδικα**: Μην δεσμεύετε ποτέ API keys ή διαπιστευτήρια
- **Εξαρτήσεις**: Διατηρήστε ενημερωμένα τα πακέτα npm και pip
- **Εισαγωγή χρήστη**: Τα παραδείγματα εφαρμογών Flask περιλαμβάνουν βασική επικύρωση εισόδου
- **Ευαίσθητα δεδομένα**: Τα παραδείγματα συνόλων δεδομένων είναι δημόσια και μη ευαίσθητα

## Αντιμετώπιση Προβλημάτων

### Jupyter Notebooks

- **Προβλήματα πυρήνα**: Επανεκκινήστε τον πυρήνα αν τα κελιά κολλήσουν: Kernel → Restart
- **Σφάλματα εισαγωγής**: Βεβαιωθείτε ότι όλες οι απαιτούμενες βιβλιοθήκες είναι εγκατεστημένες με pip
- **Προβλήματα διαδρομής**: Εκτελέστε τα notebooks από τον φάκελο που τα περιέχει

### Εφαρμογή Κουίζ

- **npm install αποτυγχάνει**: Καθαρίστε την cache npm: `npm cache clean --force`
- **Συγκρούσεις θύρας**: Αλλάξτε θύρα με: `npm run serve -- --port 8081`
- **Σφάλματα δημιουργίας**: Διαγράψτε το `node_modules` και επανεγκαταστήστε: `rm -rf node_modules && npm install`

### Μαθήματα R

- **Πακέτο δεν βρέθηκε**: Εγκαταστήστε με: `install.packages("package-name")`
- **Απόδοση RMarkdown**: Βεβαιωθείτε ότι το πακέτο rmarkdown είναι εγκατεστημένο
- **Προβλήματα πυρήνα**: Ίσως χρειαστεί να εγκαταστήσετε το IRkernel για το Jupyter

## Σημειώσεις Σχετικές με το Έργο

- Αυτό είναι κυρίως ένα **εκπαιδευτικό πρόγραμμα σπουδών**, όχι κώδικας παραγωγής
- Η εστίαση είναι στην **κατανόηση εννοιών ML** μέσω πρακτικής εξάσκησης
- Τα παραδείγματα κώδικα δίνουν προτεραιότητα στη **σαφήνεια αντί της βελτιστοποίησης**
- Τα περισσότερα μαθήματα είναι **αυτοτελή** και μπορούν να ολοκληρωθούν ανεξάρτητα
- **Παρέχονται λύσεις**, αλλά οι μαθητές πρέπει να προσπαθήσουν πρώτα τις ασκήσεις
- Το αποθετήριο χρησιμοποιεί **Docsify** για τεκμηρίωση ιστού χωρίς βήμα δημιουργίας
- **Sketchnotes** παρέχουν οπτικές περιλήψεις εννοιών
- **Υποστήριξη πολλών γλωσσών** καθιστά το περιεχόμενο προσβάσιμο παγκοσμίως

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτόματες μεταφράσεις ενδέχεται να περιέχουν σφάλματα ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.