<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "482bccabe1df958496ea71a3667995cd",
  "translation_date": "2025-09-04T23:55:18+00:00",
  "source_file": "7-TimeSeries/3-SVR/README.md",
  "language_code": "el"
}
-->
# Πρόβλεψη Χρονοσειρών με Support Vector Regressor

Στο προηγούμενο μάθημα, μάθατε πώς να χρησιμοποιείτε το μοντέλο ARIMA για να κάνετε προβλέψεις χρονοσειρών. Τώρα θα εξετάσετε το μοντέλο Support Vector Regressor, το οποίο είναι ένα μοντέλο παλινδρόμησης που χρησιμοποιείται για την πρόβλεψη συνεχών δεδομένων.

## [Προ-μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/) 

## Εισαγωγή

Σε αυτό το μάθημα, θα ανακαλύψετε έναν συγκεκριμένο τρόπο για να δημιουργήσετε μοντέλα με [**SVM**: **S**upport **V**ector **M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) για παλινδρόμηση, ή **SVR: Support Vector Regressor**. 

### SVR στο πλαίσιο των χρονοσειρών [^1]

Πριν κατανοήσετε τη σημασία του SVR στην πρόβλεψη χρονοσειρών, εδώ είναι μερικές σημαντικές έννοιες που πρέπει να γνωρίζετε:

- **Παλινδρόμηση:** Τεχνική εποπτευόμενης μάθησης για την πρόβλεψη συνεχών τιμών από ένα δεδομένο σύνολο εισόδων. Η ιδέα είναι να προσαρμοστεί μια καμπύλη (ή γραμμή) στον χώρο χαρακτηριστικών που έχει τον μέγιστο αριθμό σημείων δεδομένων. [Κάντε κλικ εδώ](https://en.wikipedia.org/wiki/Regression_analysis) για περισσότερες πληροφορίες.
- **Support Vector Machine (SVM):** Ένας τύπος εποπτευόμενου μοντέλου μηχανικής μάθησης που χρησιμοποιείται για ταξινόμηση, παλινδρόμηση και ανίχνευση ανωμαλιών. Το μοντέλο είναι ένα υπερεπίπεδο στον χώρο χαρακτηριστικών, το οποίο στην περίπτωση της ταξινόμησης λειτουργεί ως όριο, και στην περίπτωση της παλινδρόμησης λειτουργεί ως η καλύτερη γραμμή προσαρμογής. Στο SVM, μια συνάρτηση Kernel χρησιμοποιείται γενικά για τη μετατροπή του συνόλου δεδομένων σε έναν χώρο με μεγαλύτερο αριθμό διαστάσεων, ώστε να μπορούν να διαχωριστούν εύκολα. [Κάντε κλικ εδώ](https://en.wikipedia.org/wiki/Support-vector_machine) για περισσότερες πληροφορίες σχετικά με τα SVM.
- **Support Vector Regressor (SVR):** Ένας τύπος SVM, που βρίσκει την καλύτερη γραμμή προσαρμογής (η οποία στην περίπτωση του SVM είναι ένα υπερεπίπεδο) που έχει τον μέγιστο αριθμό σημείων δεδομένων.

### Γιατί SVR; [^1]

Στο προηγούμενο μάθημα μάθατε για το ARIMA, το οποίο είναι μια πολύ επιτυχημένη στατιστική γραμμική μέθοδος για την πρόβλεψη δεδομένων χρονοσειρών. Ωστόσο, σε πολλές περιπτώσεις, τα δεδομένα χρονοσειρών έχουν *μη γραμμικότητα*, η οποία δεν μπορεί να χαρτογραφηθεί από γραμμικά μοντέλα. Σε τέτοιες περιπτώσεις, η ικανότητα του SVM να λαμβάνει υπόψη τη μη γραμμικότητα των δεδομένων για εργασίες παλινδρόμησης καθιστά το SVR επιτυχημένο στην πρόβλεψη χρονοσειρών.

## Άσκηση - δημιουργία μοντέλου SVR

Τα πρώτα βήματα για την προετοιμασία δεδομένων είναι ίδια με αυτά του προηγούμενου μαθήματος για το [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA). 

Ανοίξτε τον φάκελο [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/3-SVR/working) σε αυτό το μάθημα και βρείτε το αρχείο [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/3-SVR/working/notebook.ipynb).[^2]

1. Εκτελέστε το notebook και εισάγετε τις απαραίτητες βιβλιοθήκες:  [^2]

   ```python
   import sys
   sys.path.append('../../')
   ```

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```

2. Φορτώστε τα δεδομένα από το αρχείο `/data/energy.csv` σε ένα Pandas dataframe και δείτε τα:  [^2]

   ```python
   energy = load_data('../../data')[['load']]
   ```

3. Σχεδιάστε όλα τα διαθέσιμα δεδομένα ενέργειας από τον Ιανουάριο 2012 έως τον Δεκέμβριο 2014: [^2]

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![πλήρη δεδομένα](../../../../7-TimeSeries/3-SVR/images/full-data.png)

   Τώρα, ας δημιουργήσουμε το μοντέλο SVR.

### Δημιουργία συνόλων εκπαίδευσης και δοκιμής

Τώρα που τα δεδομένα σας έχουν φορτωθεί, μπορείτε να τα διαχωρίσετε σε σύνολα εκπαίδευσης και δοκιμής. Στη συνέχεια, θα αναδιαμορφώσετε τα δεδομένα για να δημιουργήσετε ένα σύνολο δεδομένων βασισμένο σε χρονικά βήματα, το οποίο θα χρειαστεί για το SVR. Θα εκπαιδεύσετε το μοντέλο σας στο σύνολο εκπαίδευσης. Αφού ολοκληρωθεί η εκπαίδευση του μοντέλου, θα αξιολογήσετε την ακρίβειά του στο σύνολο εκπαίδευσης, στο σύνολο δοκιμής και στη συνέχεια στο πλήρες σύνολο δεδομένων για να δείτε τη συνολική απόδοση. Πρέπει να βεβαιωθείτε ότι το σύνολο δοκιμής καλύπτει μια μεταγενέστερη χρονική περίοδο από το σύνολο εκπαίδευσης, ώστε να διασφαλίσετε ότι το μοντέλο δεν αποκτά πληροφορίες από μελλοντικές χρονικές περιόδους [^2] (μια κατάσταση γνωστή ως *Υπερπροσαρμογή*).

1. Κατανομή μιας δίμηνης περιόδου από την 1η Σεπτεμβρίου έως την 31η Οκτωβρίου 2014 στο σύνολο εκπαίδευσης. Το σύνολο δοκιμής θα περιλαμβάνει τη δίμηνη περίοδο από την 1η Νοεμβρίου έως την 31η Δεκεμβρίου 2014: [^2]

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

2. Οπτικοποίηση των διαφορών: [^2]

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![δεδομένα εκπαίδευσης και δοκιμής](../../../../7-TimeSeries/3-SVR/images/train-test.png)



### Προετοιμασία των δεδομένων για εκπαίδευση

Τώρα, πρέπει να προετοιμάσετε τα δεδομένα για εκπαίδευση πραγματοποιώντας φιλτράρισμα και κλιμάκωση των δεδομένων σας. Φιλτράρετε το σύνολο δεδομένων σας ώστε να περιλαμβάνει μόνο τις χρονικές περιόδους και τις στήλες που χρειάζεστε, και κλιμάκωση για να διασφαλίσετε ότι τα δεδομένα προβάλλονται στο διάστημα 0,1.

1. Φιλτράρετε το αρχικό σύνολο δεδομένων ώστε να περιλαμβάνει μόνο τις προαναφερθείσες χρονικές περιόδους ανά σύνολο και μόνο τη στήλη 'load' και την ημερομηνία: [^2]

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```
   
2. Κλιμακώστε τα δεδομένα εκπαίδευσης ώστε να βρίσκονται στο εύρος (0, 1): [^2]

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   ```
   
4. Τώρα, κλιμακώστε τα δεδομένα δοκιμής: [^2]

   ```python
   test['load'] = scaler.transform(test)
   ```

### Δημιουργία δεδομένων με χρονικά βήματα [^1]

Για το SVR, μετατρέπετε τα δεδομένα εισόδου ώστε να έχουν τη μορφή `[batch, timesteps]`. Έτσι, αναδιαμορφώνετε τα υπάρχοντα `train_data` και `test_data` ώστε να υπάρχει μια νέα διάσταση που αναφέρεται στα χρονικά βήματα. 

```python
# Converting to numpy arrays
train_data = train.values
test_data = test.values
```

Για αυτό το παράδειγμα, παίρνουμε `timesteps = 5`. Έτσι, οι είσοδοι στο μοντέλο είναι τα δεδομένα για τα πρώτα 4 χρονικά βήματα, και η έξοδος θα είναι τα δεδομένα για το 5ο χρονικό βήμα.

```python
timesteps=5
```

Μετατροπή δεδομένων εκπαίδευσης σε 2D tensor χρησιμοποιώντας εμφωλευμένη λίστα κατανόησης:

```python
train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```output
(1412, 5)
```

Μετατροπή δεδομένων δοκιμής σε 2D tensor:

```python
test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```output
(44, 5)
```

Επιλογή εισόδων και εξόδων από δεδομένα εκπαίδευσης και δοκιμής:

```python
x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```

```output
(1412, 4) (1412, 1)
(44, 4) (44, 1)
```

### Υλοποίηση SVR [^1]

Τώρα, είναι ώρα να υλοποιήσετε το SVR. Για να διαβάσετε περισσότερα σχετικά με αυτήν την υλοποίηση, μπορείτε να ανατρέξετε [σε αυτήν την τεκμηρίωση](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html). Για την υλοποίησή μας, ακολουθούμε αυτά τα βήματα:

  1. Ορίστε το μοντέλο καλώντας το `SVR()` και περνώντας τις υπερπαραμέτρους του μοντέλου: kernel, gamma, c και epsilon
  2. Προετοιμάστε το μοντέλο για τα δεδομένα εκπαίδευσης καλώντας τη συνάρτηση `fit()`
  3. Κάντε προβλέψεις καλώντας τη συνάρτηση `predict()`

Τώρα δημιουργούμε ένα μοντέλο SVR. Εδώ χρησιμοποιούμε το [RBF kernel](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel), και ορίζουμε τις υπερπαραμέτρους gamma, C και epsilon ως 0.5, 10 και 0.05 αντίστοιχα.

```python
model = SVR(kernel='rbf',gamma=0.5, C=10, epsilon = 0.05)
```

#### Εκπαίδευση του μοντέλου στα δεδομένα εκπαίδευσης [^1]

```python
model.fit(x_train, y_train[:,0])
```

```output
SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5,
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
```

#### Δημιουργία προβλέψεων από το μοντέλο [^1]

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```

```output
(1412, 1) (44, 1)
```

Έχετε δημιουργήσει το SVR σας! Τώρα πρέπει να το αξιολογήσουμε.

### Αξιολόγηση του μοντέλου σας [^1]

Για την αξιολόγηση, πρώτα θα κλιμακώσουμε πίσω τα δεδομένα στην αρχική μας κλίμακα. Στη συνέχεια, για να ελέγξουμε την απόδοση, θα σχεδιάσουμε το αρχικό και το προβλεπόμενο γράφημα χρονοσειρών και θα εκτυπώσουμε επίσης το αποτέλεσμα MAPE.

Κλιμάκωση της προβλεπόμενης και αρχικής εξόδου:

```python
# Scaling the predictions
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

print(len(y_train_pred), len(y_test_pred))
```

```python
# Scaling the original values
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

print(len(y_train), len(y_test))
```

#### Έλεγχος απόδοσης του μοντέλου στα δεδομένα εκπαίδευσης και δοκιμής [^1]

Εξάγουμε τις χρονικές σημάνσεις από το σύνολο δεδομένων για να τις δείξουμε στον άξονα x του γραφήματός μας. Σημειώστε ότι χρησιμοποιούμε τις πρώτες ```timesteps-1``` τιμές ως είσοδο για την πρώτη έξοδο, οπότε οι χρονικές σημάνσεις για την έξοδο θα ξεκινήσουν μετά από αυτό.

```python
train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
test_timestamps = energy[test_start_dt:].index[timesteps-1:]

print(len(train_timestamps), len(test_timestamps))
```

```output
1412 44
```

Σχεδιάστε τις προβλέψεις για τα δεδομένα εκπαίδευσης:

```python
plt.figure(figsize=(25,6))
plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.title("Training data prediction")
plt.show()
```

![πρόβλεψη δεδομένων εκπαίδευσης](../../../../7-TimeSeries/3-SVR/images/train-data-predict.png)

Εκτύπωση MAPE για τα δεδομένα εκπαίδευσης

```python
print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
```

```output
MAPE for training data: 1.7195710200875551 %
```

Σχεδιάστε τις προβλέψεις για τα δεδομένα δοκιμής

```python
plt.figure(figsize=(10,3))
plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![πρόβλεψη δεδομένων δοκιμής](../../../../7-TimeSeries/3-SVR/images/test-data-predict.png)

Εκτύπωση MAPE για τα δεδομένα δοκιμής

```python
print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
```

```output
MAPE for testing data:  1.2623790187854018 %
```

🏆 Έχετε ένα πολύ καλό αποτέλεσμα στο σύνολο δεδομένων δοκιμής!

### Έλεγχος απόδοσης του μοντέλου στο πλήρες σύνολο δεδομένων [^1]

```python
# Extracting load values as numpy array
data = energy.copy().values

# Scaling
data = scaler.transform(data)

# Transforming to 2D tensor as per model input requirement
data_timesteps=np.array([[j for j in data[i:i+timesteps]] for i in range(0,len(data)-timesteps+1)])[:,:,0]
print("Tensor shape: ", data_timesteps.shape)

# Selecting inputs and outputs from data
X, Y = data_timesteps[:,:timesteps-1],data_timesteps[:,[timesteps-1]]
print("X shape: ", X.shape,"\nY shape: ", Y.shape)
```

```output
Tensor shape:  (26300, 5)
X shape:  (26300, 4) 
Y shape:  (26300, 1)
```

```python
# Make model predictions
Y_pred = model.predict(X).reshape(-1,1)

# Inverse scale and reshape
Y_pred = scaler.inverse_transform(Y_pred)
Y = scaler.inverse_transform(Y)
```

```python
plt.figure(figsize=(30,8))
plt.plot(Y, color = 'red', linewidth=2.0, alpha = 0.6)
plt.plot(Y_pred, color = 'blue', linewidth=0.8)
plt.legend(['Actual','Predicted'])
plt.xlabel('Timestamp')
plt.show()
```

![πρόβλεψη πλήρων δεδομένων](../../../../7-TimeSeries/3-SVR/images/full-data-predict.png)

```python
print('MAPE: ', mape(Y_pred, Y)*100, '%')
```

```output
MAPE:  2.0572089029888656 %
```



🏆 Πολύ ωραία γραφήματα, που δείχνουν ένα μοντέλο με καλή ακρίβεια. Μπράβο!

---

## 🚀Πρόκληση

- Προσπαθήστε να τροποποιήσετε τις υπερπαραμέτρους (gamma, C, epsilon) κατά τη δημιουργία του μοντέλου και να αξιολογήσετε τα δεδομένα για να δείτε ποιο σύνολο υπερπαραμέτρων δίνει τα καλύτερα αποτελέσματα στα δεδομένα δοκιμής. Για να μάθετε περισσότερα σχετικά με αυτές τις υπερπαραμέτρους, μπορείτε να ανατρέξετε στο έγγραφο [εδώ](https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel). 
- Προσπαθήστε να χρησιμοποιήσετε διαφορετικές συναρτήσεις kernel για το μοντέλο και να αναλύσετε τις επιδόσεις τους στο σύνολο δεδομένων. Ένα χρήσιμο έγγραφο μπορείτε να βρείτε [εδώ](https://scikit-learn.org/stable/modules/svm.html#kernel-functions).
- Προσπαθήστε να χρησιμοποιήσετε διαφορετικές τιμές για `timesteps` για το μοντέλο ώστε να κοιτάξει πίσω για να κάνει πρόβλεψη.

## [Μετά το μάθημα κουίζ](https://ff-quizzes.netlify.app/en/ml/)

## Ανασκόπηση & Αυτομελέτη

Αυτό το μάθημα είχε σκοπό να εισαγάγει την εφαρμογή του SVR για την πρόβλεψη χρονοσειρών. Για να διαβάσετε περισσότερα σχετικά με το SVR, μπορείτε να ανατρέξετε [σε αυτό το blog](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/). Αυτή η [τεκμηρίωση στο scikit-learn](https://scikit-learn.org/stable/modules/svm.html) παρέχει μια πιο ολοκληρωμένη εξήγηση σχετικά με τα SVM γενικά, [SVRs](https://scikit-learn.org/stable/modules/svm.html#regression) και επίσης άλλες λεπτομέρειες υλοποίησης όπως οι διαφορετικές [συναρτήσεις kernel](https://scikit-learn.org/stable/modules/svm.html#kernel-functions) που μπορούν να χρησιμοποιηθούν και οι παράμετροί τους.

## Εργασία

[Ένα νέο μοντέλο SVR](assignment.md)



## Πιστώσεις


[^1]: Το κείμενο, ο κώδικας και η έξοδος σε αυτήν την ενότητα συνεισφέρθηκαν από τον [@AnirbanMukherjeeXD](https://github.com/AnirbanMukherjeeXD)
[^2]: Το κείμενο, ο κώδικας και η έξοδος σε αυτήν την ενότητα ελήφθησαν από το [ARIMA](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA)

---

**Αποποίηση ευθύνης**:  
Αυτό το έγγραφο έχει μεταφραστεί χρησιμοποιώντας την υπηρεσία αυτόματης μετάφρασης [Co-op Translator](https://github.com/Azure/co-op-translator). Παρόλο που καταβάλλουμε προσπάθειες για ακρίβεια, παρακαλούμε να έχετε υπόψη ότι οι αυτοματοποιημένες μεταφράσεις ενδέχεται να περιέχουν λάθη ή ανακρίβειες. Το πρωτότυπο έγγραφο στη μητρική του γλώσσα θα πρέπει να θεωρείται η αυθεντική πηγή. Για κρίσιμες πληροφορίες, συνιστάται επαγγελματική ανθρώπινη μετάφραση. Δεν φέρουμε ευθύνη για τυχόν παρεξηγήσεις ή εσφαλμένες ερμηνείες που προκύπτουν από τη χρήση αυτής της μετάφρασης.