<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T00:40:43+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "el"
}
-->
# Κουίζ

Αυτά τα κουίζ είναι τα κουίζ πριν και μετά τη διάλεξη για το πρόγραμμα σπουδών ML στο https://aka.ms/ml-beginners

## Ρύθμιση έργου

```
npm install
```

### Συμπίεση και άμεση επαναφόρτωση για ανάπτυξη

```
npm run serve
```

### Συμπίεση και ελαχιστοποίηση για παραγωγή

```
npm run build
```

### Έλεγχος και διόρθωση αρχείων

```
npm run lint
```

### Προσαρμογή ρυθμίσεων

Δείτε [Αναφορά Ρυθμίσεων](https://cli.vuejs.org/config/).

Ευχαριστίες: Ευχαριστούμε την αρχική έκδοση αυτής της εφαρμογής κουίζ: https://github.com/arpan45/simple-quiz-vue

## Ανάπτυξη στο Azure

Ακολουθεί ένας οδηγός βήμα προς βήμα για να ξεκινήσετε:

1. Κλωνοποίηση ενός GitHub Repository  
Βεβαιωθείτε ότι ο κώδικας της στατικής web εφαρμογής σας βρίσκεται στο GitHub repository σας. Κλωνοποιήστε αυτό το repository.

2. Δημιουργία μιας Στατικής Web Εφαρμογής στο Azure  
- Δημιουργήστε έναν [λογαριασμό Azure](http://azure.microsoft.com)  
- Μεταβείτε στο [Azure portal](https://portal.azure.com)  
- Κάντε κλικ στο “Create a resource” και αναζητήστε “Static Web App”.  
- Κάντε κλικ στο “Create”.

3. Ρύθμιση της Στατικής Web Εφαρμογής  
- Βασικά:  
  - Συνδρομή: Επιλέξτε τη συνδρομή σας στο Azure.  
  - Ομάδα Πόρων: Δημιουργήστε μια νέα ομάδα πόρων ή χρησιμοποιήστε μια υπάρχουσα.  
  - Όνομα: Δώστε ένα όνομα για τη στατική web εφαρμογή σας.  
  - Περιοχή: Επιλέξτε την περιοχή που είναι πιο κοντά στους χρήστες σας.

- #### Λεπτομέρειες Ανάπτυξης:  
  - Πηγή: Επιλέξτε “GitHub”.  
  - Λογαριασμός GitHub: Εξουσιοδοτήστε το Azure να έχει πρόσβαση στον λογαριασμό σας στο GitHub.  
  - Οργάνωση: Επιλέξτε την οργάνωση σας στο GitHub.  
  - Repository: Επιλέξτε το repository που περιέχει τη στατική web εφαρμογή σας.  
  - Κλάδος: Επιλέξτε τον κλάδο από τον οποίο θέλετε να γίνει η ανάπτυξη.

- #### Λεπτομέρειες Κατασκευής:  
  - Προεπιλογές Κατασκευής: Επιλέξτε το framework με το οποίο έχει κατασκευαστεί η εφαρμογή σας (π.χ., React, Angular, Vue, κλπ.).  
  - Τοποθεσία Εφαρμογής: Καθορίστε τον φάκελο που περιέχει τον κώδικα της εφαρμογής σας (π.χ., / αν βρίσκεται στη ρίζα).  
  - Τοποθεσία API: Αν έχετε API, καθορίστε την τοποθεσία του (προαιρετικό).  
  - Τοποθεσία Εξόδου: Καθορίστε τον φάκελο όπου δημιουργείται η έξοδος της κατασκευής (π.χ., build ή dist).

4. Ανασκόπηση και Δημιουργία  
Ανασκοπήστε τις ρυθμίσεις σας και κάντε κλικ στο “Create”. Το Azure θα δημιουργήσει τους απαραίτητους πόρους και θα δημιουργήσει ένα GitHub Actions workflow στο repository σας.

5. GitHub Actions Workflow  
Το Azure θα δημιουργήσει αυτόματα ένα αρχείο GitHub Actions workflow στο repository σας (.github/workflows/azure-static-web-apps-<name>.yml). Αυτό το workflow θα χειριστεί τη διαδικασία κατασκευής και ανάπτυξης.

6. Παρακολούθηση της Ανάπτυξης  
Μεταβείτε στην καρτέλα “Actions” στο repository σας στο GitHub.  
Θα πρέπει να δείτε ένα workflow να εκτελείται. Αυτό το workflow θα κατασκευάσει και θα αναπτύξει τη στατική web εφαρμογή σας στο Azure.  
Μόλις ολοκληρωθεί το workflow, η εφαρμογή σας θα είναι διαθέσιμη στη διεύθυνση URL που παρέχεται από το Azure.

### Παράδειγμα Αρχείου Workflow

Ακολουθεί ένα παράδειγμα του αρχείου GitHub Actions workflow:  
name: Azure Static Web Apps CI/CD  
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### Πρόσθετοι Πόροι  
- [Τεκμηρίωση Στατικών Web Εφαρμογών στο Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Τεκμηρίωση GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτοματοποιημένες μεταφράσεις ενδέχεται να περιέχουν λάθη ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.