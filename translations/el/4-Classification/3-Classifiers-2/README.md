<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T00:50:19+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "el"
}
-->
# Ταξινομητές Κουζίνας 2

Σε αυτό το δεύτερο μάθημα ταξινόμησης, θα εξερευνήσετε περισσότερους τρόπους ταξινόμησης αριθμητικών δεδομένων. Θα μάθετε επίσης για τις συνέπειες της επιλογής ενός ταξινομητή έναντι ενός άλλου.

## [Προ-μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/)

### Προαπαιτούμενα

Υποθέτουμε ότι έχετε ολοκληρώσει τα προηγούμενα μαθήματα και έχετε ένα καθαρισμένο σύνολο δεδομένων στον φάκελο `data` που ονομάζεται _cleaned_cuisines.csv_ στη ρίζα αυτού του φακέλου με τα 4 μαθήματα.

### Προετοιμασία

Έχουμε φορτώσει το αρχείο σας _notebook.ipynb_ με το καθαρισμένο σύνολο δεδομένων και το έχουμε χωρίσει σε X και y dataframes, έτοιμα για τη διαδικασία δημιουργίας μοντέλου.

## Χάρτης ταξινόμησης

Προηγουμένως, μάθατε για τις διάφορες επιλογές που έχετε όταν ταξινομείτε δεδομένα χρησιμοποιώντας το cheat sheet της Microsoft. Το Scikit-learn προσφέρει ένα παρόμοιο, αλλά πιο λεπτομερές cheat sheet που μπορεί να βοηθήσει περαιτέρω να περιορίσετε τους εκτιμητές σας (ένας άλλος όρος για τους ταξινομητές):

![ML Map from Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Συμβουλή: [επισκεφθείτε αυτόν τον χάρτη online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) και κάντε κλικ κατά μήκος της διαδρομής για να διαβάσετε την τεκμηρίωση.

### Το σχέδιο

Αυτός ο χάρτης είναι πολύ χρήσιμος μόλις έχετε μια σαφή κατανόηση των δεδομένων σας, καθώς μπορείτε να "περπατήσετε" κατά μήκος των διαδρομών του για να πάρετε μια απόφαση:

- Έχουμε >50 δείγματα
- Θέλουμε να προβλέψουμε μια κατηγορία
- Έχουμε δεδομένα με ετικέτες
- Έχουμε λιγότερα από 100K δείγματα
- ✨ Μπορούμε να επιλέξουμε Linear SVC
- Αν αυτό δεν λειτουργήσει, δεδομένου ότι έχουμε αριθμητικά δεδομένα
    - Μπορούμε να δοκιμάσουμε ✨ KNeighbors Classifier 
      - Αν αυτό δεν λειτουργήσει, δοκιμάστε ✨ SVC και ✨ Ensemble Classifiers

Αυτή είναι μια πολύ χρήσιμη διαδρομή για να ακολουθήσετε.

## Άσκηση - χωρίστε τα δεδομένα

Ακολουθώντας αυτή τη διαδρομή, θα πρέπει να ξεκινήσουμε εισάγοντας κάποιες βιβλιοθήκες για χρήση.

1. Εισάγετε τις απαραίτητες βιβλιοθήκες:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Χωρίστε τα δεδομένα εκπαίδευσης και δοκιμής:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Ταξινομητής Linear SVC

Η ταξινόμηση Support-Vector (SVC) είναι μέλος της οικογένειας τεχνικών ML Support-Vector Machines (μάθετε περισσότερα για αυτές παρακάτω). Σε αυτή τη μέθοδο, μπορείτε να επιλέξετε έναν 'πυρήνα' για να αποφασίσετε πώς να ομαδοποιήσετε τις ετικέτες. Η παράμετρος 'C' αναφέρεται στην 'κανονικοποίηση', η οποία ρυθμίζει την επιρροή των παραμέτρων. Ο πυρήνας μπορεί να είναι ένας από [διάφορους](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); εδώ τον ορίζουμε ως 'linear' για να διασφαλίσουμε ότι χρησιμοποιούμε Linear SVC. Η πιθανότητα έχει ως προεπιλογή την τιμή 'false'; εδώ την ορίζουμε ως 'true' για να συλλέξουμε εκτιμήσεις πιθανότητας. Ορίζουμε την τυχαία κατάσταση ως '0' για να ανακατέψουμε τα δεδομένα ώστε να πάρουμε πιθανότητες.

### Άσκηση - εφαρμόστε Linear SVC

Ξεκινήστε δημιουργώντας έναν πίνακα ταξινομητών. Θα προσθέσετε προοδευτικά σε αυτόν τον πίνακα καθώς δοκιμάζουμε.

1. Ξεκινήστε με Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Εκπαιδεύστε το μοντέλο σας χρησιμοποιώντας το Linear SVC και εκτυπώστε μια αναφορά:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Το αποτέλεσμα είναι αρκετά καλό:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Ταξινομητής K-Neighbors

Ο K-Neighbors είναι μέρος της οικογένειας "neighbors" των μεθόδων ML, οι οποίες μπορούν να χρησιμοποιηθούν τόσο για εποπτευόμενη όσο και για μη εποπτευόμενη μάθηση. Σε αυτή τη μέθοδο, δημιουργείται ένας προκαθορισμένος αριθμός σημείων και τα δεδομένα συγκεντρώνονται γύρω από αυτά τα σημεία, ώστε να μπορούν να προβλεφθούν γενικευμένες ετικέτες για τα δεδομένα.

### Άσκηση - εφαρμόστε τον ταξινομητή K-Neighbors

Ο προηγούμενος ταξινομητής ήταν καλός και λειτούργησε καλά με τα δεδομένα, αλλά ίσως μπορούμε να πετύχουμε καλύτερη ακρίβεια. Δοκιμάστε έναν ταξινομητή K-Neighbors.

1. Προσθέστε μια γραμμή στον πίνακα ταξινομητών σας (προσθέστε ένα κόμμα μετά το στοιχείο Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Το αποτέλεσμα είναι λίγο χειρότερο:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Μάθετε για [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Ταξινομητής Support Vector

Οι ταξινομητές Support-Vector είναι μέρος της οικογένειας [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) των μεθόδων ML που χρησιμοποιούνται για εργασίες ταξινόμησης και παλινδρόμησης. Οι SVMs "χαρτογραφούν παραδείγματα εκπαίδευσης σε σημεία στο χώρο" για να μεγιστοποιήσουν την απόσταση μεταξύ δύο κατηγοριών. Τα επόμενα δεδομένα χαρτογραφούνται σε αυτόν τον χώρο ώστε να μπορεί να προβλεφθεί η κατηγορία τους.

### Άσκηση - εφαρμόστε έναν ταξινομητή Support Vector

Ας προσπαθήσουμε για λίγο καλύτερη ακρίβεια με έναν ταξινομητή Support Vector.

1. Προσθέστε ένα κόμμα μετά το στοιχείο K-Neighbors και στη συνέχεια προσθέστε αυτή τη γραμμή:

    ```python
    'SVC': SVC(),
    ```

    Το αποτέλεσμα είναι αρκετά καλό!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Μάθετε για [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ταξινομητές Ensemble

Ας ακολουθήσουμε τη διαδρομή μέχρι το τέλος, παρόλο που η προηγούμενη δοκιμή ήταν αρκετά καλή. Ας δοκιμάσουμε κάποιους ταξινομητές Ensemble, συγκεκριμένα Random Forest και AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Το αποτέλεσμα είναι πολύ καλό, ειδικά για το Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Μάθετε για [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Αυτή η μέθοδος Μηχανικής Μάθησης "συνδυάζει τις προβλέψεις αρκετών βασικών εκτιμητών" για να βελτιώσει την ποιότητα του μοντέλου. Στο παράδειγμά μας, χρησιμοποιήσαμε Random Trees και AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), μια μέθοδος μέσου όρου, δημιουργεί ένα 'δάσος' από 'δέντρα αποφάσεων' με τυχαιότητα για να αποφύγει την υπερπροσαρμογή. Η παράμετρος n_estimators ορίζεται στον αριθμό των δέντρων.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) προσαρμόζει έναν ταξινομητή σε ένα σύνολο δεδομένων και στη συνέχεια προσαρμόζει αντίγραφα αυτού του ταξινομητή στο ίδιο σύνολο δεδομένων. Εστιάζει στα βάρη των αντικειμένων που ταξινομήθηκαν λανθασμένα και προσαρμόζει την προσαρμογή για τον επόμενο ταξινομητή ώστε να διορθώσει.

---

## 🚀Πρόκληση

Καθένα από αυτά τα τεχνικά έχει μεγάλο αριθμό παραμέτρων που μπορείτε να προσαρμόσετε. Ερευνήστε τις προεπιλεγμένες παραμέτρους του καθενός και σκεφτείτε τι θα σήμαινε η προσαρμογή αυτών των παραμέτρων για την ποιότητα του μοντέλου.

## [Μετά το μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/)

## Ανασκόπηση & Αυτομελέτη

Υπάρχει πολλή ορολογία σε αυτά τα μαθήματα, οπότε αφιερώστε λίγο χρόνο για να αναθεωρήσετε [αυτήν τη λίστα](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) χρήσιμης ορολογίας!

## Εργασία 

[Παίξτε με τις παραμέτρους](assignment.md)

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτοματοποιημένες μεταφράσεις ενδέχεται να περιέχουν σφάλματα ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.