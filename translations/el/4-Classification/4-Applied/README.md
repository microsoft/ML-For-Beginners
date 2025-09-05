<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T00:47:47+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "el"
}
-->
# Δημιουργία Εφαρμογής Web για Προτάσεις Κουζίνας

Σε αυτό το μάθημα, θα δημιουργήσετε ένα μοντέλο ταξινόμησης χρησιμοποιώντας κάποιες από τις τεχνικές που μάθατε σε προηγούμενα μαθήματα και με το νόστιμο dataset κουζίνας που χρησιμοποιήθηκε σε αυτή τη σειρά. Επιπλέον, θα δημιουργήσετε μια μικρή εφαρμογή web για να χρησιμοποιήσετε ένα αποθηκευμένο μοντέλο, αξιοποιώντας το web runtime του Onnx.

Μία από τις πιο χρήσιμες πρακτικές εφαρμογές της μηχανικής μάθησης είναι η δημιουργία συστημάτων προτάσεων, και μπορείτε να κάνετε το πρώτο βήμα προς αυτή την κατεύθυνση σήμερα!

[![Παρουσίαση αυτής της εφαρμογής web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Κάντε κλικ στην εικόνα παραπάνω για ένα βίντεο: Η Jen Looper δημιουργεί μια εφαρμογή web χρησιμοποιώντας δεδομένα ταξινομημένης κουζίνας

## [Προ-μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/)

Σε αυτό το μάθημα θα μάθετε:

- Πώς να δημιουργήσετε ένα μοντέλο και να το αποθηκεύσετε ως μοντέλο Onnx
- Πώς να χρησιμοποιήσετε το Netron για να επιθεωρήσετε το μοντέλο
- Πώς να χρησιμοποιήσετε το μοντέλο σας σε μια εφαρμογή web για πρόβλεψη

## Δημιουργία του μοντέλου σας

Η δημιουργία εφαρμοσμένων συστημάτων μηχανικής μάθησης είναι ένα σημαντικό μέρος της αξιοποίησης αυτών των τεχνολογιών για τα επιχειρηματικά σας συστήματα. Μπορείτε να χρησιμοποιήσετε μοντέλα μέσα στις εφαρμογές web σας (και έτσι να τα χρησιμοποιήσετε σε offline περιβάλλον αν χρειαστεί) χρησιμοποιώντας το Onnx.

Σε ένα [προηγούμενο μάθημα](../../3-Web-App/1-Web-App/README.md), δημιουργήσατε ένα μοντέλο Regression σχετικά με θεάσεις UFO, το "pickled" και το χρησιμοποιήσατε σε μια εφαρμογή Flask. Παρόλο που αυτή η αρχιτεκτονική είναι πολύ χρήσιμη, είναι μια πλήρης εφαρμογή Python, και οι απαιτήσεις σας μπορεί να περιλαμβάνουν τη χρήση μιας εφαρμογής JavaScript.

Σε αυτό το μάθημα, μπορείτε να δημιουργήσετε ένα βασικό σύστημα βασισμένο σε JavaScript για πρόβλεψη. Πρώτα, όμως, πρέπει να εκπαιδεύσετε ένα μοντέλο και να το μετατρέψετε για χρήση με το Onnx.

## Άσκηση - εκπαίδευση μοντέλου ταξινόμησης

Πρώτα, εκπαιδεύστε ένα μοντέλο ταξινόμησης χρησιμοποιώντας το καθαρισμένο dataset κουζινών που χρησιμοποιήσαμε.

1. Ξεκινήστε εισάγοντας χρήσιμες βιβλιοθήκες:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Χρειάζεστε το '[skl2onnx](https://onnx.ai/sklearn-onnx/)' για να βοηθήσετε στη μετατροπή του μοντέλου Scikit-learn σε μορφή Onnx.

1. Στη συνέχεια, δουλέψτε με τα δεδομένα σας με τον ίδιο τρόπο που κάνατε σε προηγούμενα μαθήματα, διαβάζοντας ένα αρχείο CSV χρησιμοποιώντας τη `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Αφαιρέστε τις δύο πρώτες περιττές στήλες και αποθηκεύστε τα υπόλοιπα δεδομένα ως 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Αποθηκεύστε τις ετικέτες ως 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Ξεκινήστε τη διαδικασία εκπαίδευσης

Θα χρησιμοποιήσουμε τη βιβλιοθήκη 'SVC', η οποία έχει καλή ακρίβεια.

1. Εισάγετε τις κατάλληλες βιβλιοθήκες από το Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Χωρίστε τα δεδομένα σε σύνολα εκπαίδευσης και δοκιμής:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Δημιουργήστε ένα μοντέλο ταξινόμησης SVC όπως κάνατε στο προηγούμενο μάθημα:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Τώρα, δοκιμάστε το μοντέλο σας, καλώντας τη `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Εκτυπώστε μια αναφορά ταξινόμησης για να ελέγξετε την ποιότητα του μοντέλου:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Όπως είδαμε πριν, η ακρίβεια είναι καλή:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Μετατροπή του μοντέλου σας σε Onnx

Βεβαιωθείτε ότι κάνετε τη μετατροπή με τον σωστό αριθμό Tensor. Αυτό το dataset έχει 380 συστατικά, οπότε πρέπει να σημειώσετε αυτόν τον αριθμό στο `FloatTensorType`:

1. Μετατρέψτε χρησιμοποιώντας αριθμό tensor 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Δημιουργήστε το αρχείο onx και αποθηκεύστε το ως **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Σημείωση, μπορείτε να περάσετε [επιλογές](https://onnx.ai/sklearn-onnx/parameterized.html) στο σενάριο μετατροπής σας. Σε αυτή την περίπτωση, περάσαμε το 'nocl' να είναι True και το 'zipmap' να είναι False. Επειδή αυτό είναι ένα μοντέλο ταξινόμησης, έχετε την επιλογή να αφαιρέσετε το ZipMap που παράγει μια λίστα λεξικών (δεν είναι απαραίτητο). Το `nocl` αναφέρεται στις πληροφορίες κατηγορίας που περιλαμβάνονται στο μοντέλο. Μειώστε το μέγεθος του μοντέλου σας ορίζοντας το `nocl` σε 'True'.

Εκτελώντας ολόκληρο το notebook, θα δημιουργηθεί ένα μοντέλο Onnx και θα αποθηκευτεί σε αυτόν τον φάκελο.

## Δείτε το μοντέλο σας

Τα μοντέλα Onnx δεν είναι πολύ ορατά στο Visual Studio Code, αλλά υπάρχει ένα πολύ καλό δωρεάν λογισμικό που χρησιμοποιούν πολλοί ερευνητές για να οπτικοποιήσουν το μοντέλο και να βεβαιωθούν ότι έχει κατασκευαστεί σωστά. Κατεβάστε το [Netron](https://github.com/lutzroeder/Netron) και ανοίξτε το αρχείο model.onnx. Μπορείτε να δείτε το απλό μοντέλο σας οπτικοποιημένο, με τα 380 inputs και τον ταξινομητή να εμφανίζονται:

![Οπτικοποίηση Netron](../../../../4-Classification/4-Applied/images/netron.png)

Το Netron είναι ένα χρήσιμο εργαλείο για την προβολή των μοντέλων σας.

Τώρα είστε έτοιμοι να χρησιμοποιήσετε αυτό το ωραίο μοντέλο σε μια εφαρμογή web. Ας δημιουργήσουμε μια εφαρμογή που θα είναι χρήσιμη όταν κοιτάτε στο ψυγείο σας και προσπαθείτε να καταλάβετε ποιος συνδυασμός των υπολειμμάτων συστατικών σας μπορεί να χρησιμοποιηθεί για να μαγειρέψετε μια συγκεκριμένη κουζίνα, όπως καθορίζεται από το μοντέλο σας.

## Δημιουργία εφαρμογής web προτάσεων

Μπορείτε να χρησιμοποιήσετε το μοντέλο σας απευθείας σε μια εφαρμογή web. Αυτή η αρχιτεκτονική σας επιτρέπει επίσης να το εκτελέσετε τοπικά και ακόμη και offline αν χρειαστεί. Ξεκινήστε δημιουργώντας ένα αρχείο `index.html` στον ίδιο φάκελο όπου αποθηκεύσατε το αρχείο `model.onnx`.

1. Στο αρχείο _index.html_, προσθέστε την παρακάτω σήμανση:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Τώρα, δουλεύοντας μέσα στις ετικέτες `body`, προσθέστε λίγη σήμανση για να εμφανίσετε μια λίστα με checkboxes που αντιπροσωπεύουν κάποια συστατικά:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Παρατηρήστε ότι κάθε checkbox έχει δοθεί μια τιμή. Αυτό αντανακλά τον δείκτη όπου βρίσκεται το συστατικό σύμφωνα με το dataset. Το μήλο, για παράδειγμα, σε αυτήν την αλφαβητική λίστα, καταλαμβάνει την πέμπτη στήλη, οπότε η τιμή του είναι '4' καθώς ξεκινάμε την αρίθμηση από το 0. Μπορείτε να συμβουλευτείτε το [spreadsheet συστατικών](../../../../4-Classification/data/ingredient_indexes.csv) για να ανακαλύψετε τον δείκτη ενός δεδομένου συστατικού.

    Συνεχίζοντας τη δουλειά σας στο αρχείο index.html, προσθέστε ένα μπλοκ script όπου καλείται το μοντέλο μετά το τελικό κλείσιμο `</div>`.

1. Πρώτα, εισάγετε το [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Το Onnx Runtime χρησιμοποιείται για να επιτρέψει την εκτέλεση των μοντέλων Onnx σας σε μια ευρεία γκάμα πλατφορμών υλικού, συμπεριλαμβανομένων βελτιστοποιήσεων και ενός API για χρήση.

1. Μόλις το Runtime είναι στη θέση του, μπορείτε να το καλέσετε:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

Σε αυτόν τον κώδικα, συμβαίνουν αρκετά πράγματα:

1. Δημιουργήσατε έναν πίνακα με 380 πιθανές τιμές (1 ή 0) που θα οριστούν και θα σταλούν στο μοντέλο για πρόβλεψη, ανάλογα με το αν έχει επιλεγεί ένα checkbox συστατικού.
2. Δημιουργήσατε έναν πίνακα με checkboxes και έναν τρόπο να προσδιορίσετε αν έχουν επιλεγεί σε μια συνάρτηση `init` που καλείται όταν ξεκινά η εφαρμογή. Όταν επιλέγεται ένα checkbox, ο πίνακας `ingredients` τροποποιείται για να αντικατοπτρίζει το επιλεγμένο συστατικό.
3. Δημιουργήσατε μια συνάρτηση `testCheckboxes` που ελέγχει αν έχει επιλεγεί κάποιο checkbox.
4. Χρησιμοποιείτε τη συνάρτηση `startInference` όταν πατηθεί το κουμπί και, αν έχει επιλεγεί κάποιο checkbox, ξεκινάτε την πρόβλεψη.
5. Η διαδικασία πρόβλεψης περιλαμβάνει:
   1. Ρύθμιση ασύγχρονης φόρτωσης του μοντέλου
   2. Δημιουργία δομής Tensor για αποστολή στο μοντέλο
   3. Δημιουργία 'feeds' που αντικατοπτρίζει την είσοδο `float_input` που δημιουργήσατε κατά την εκπαίδευση του μοντέλου σας (μπορείτε να χρησιμοποιήσετε το Netron για να επαληθεύσετε αυτό το όνομα)
   4. Αποστολή αυτών των 'feeds' στο μοντέλο και αναμονή για απάντηση

## Δοκιμάστε την εφαρμογή σας

Ανοίξτε μια συνεδρία τερματικού στο Visual Studio Code στον φάκελο όπου βρίσκεται το αρχείο index.html. Βεβαιωθείτε ότι έχετε εγκαταστήσει [http-server](https://www.npmjs.com/package/http-server) παγκοσμίως και πληκτρολογήστε `http-server` στη γραμμή εντολών. Ένας localhost θα ανοίξει και μπορείτε να δείτε την εφαρμογή web σας. Ελέγξτε ποια κουζίνα προτείνεται με βάση διάφορα συστατικά:

![Εφαρμογή web συστατικών](../../../../4-Classification/4-Applied/images/web-app.png)

Συγχαρητήρια, δημιουργήσατε μια εφαρμογή web 'προτάσεων' με λίγα πεδία. Αφιερώστε λίγο χρόνο για να αναπτύξετε αυτό το σύστημα!

## 🚀Πρόκληση

Η εφαρμογή web σας είναι πολύ βασική, οπότε συνεχίστε να την αναπτύσσετε χρησιμοποιώντας συστατικά και τους δείκτες τους από τα δεδομένα [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Ποιοι συνδυασμοί γεύσεων λειτουργούν για τη δημιουργία ενός εθνικού πιάτου;

## [Μετά το μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/)

## Ανασκόπηση & Αυτομελέτη

Ενώ αυτό το μάθημα απλώς άγγιξε τη χρησιμότητα της δημιουργίας ενός συστήματος προτάσεων για συστατικά τροφίμων, αυτή η περιοχή εφαρμογών μηχανικής μάθησης είναι πολύ πλούσια σε παραδείγματα. Διαβάστε περισσότερα για το πώς κατασκευάζονται αυτά τα συστήματα:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Εργασία 

[Δημιουργήστε ένα νέο σύστημα προτάσεων](assignment.md)

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτοματοποιημένες μεταφράσεις ενδέχεται να περιέχουν λάθη ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.