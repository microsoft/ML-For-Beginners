# Classificatori di cucina 2

In questa seconda lezione sulla classificazione, si esploreranno più modi per classificare i dati numerici. Si Impareranno anche le ramificazioni per la scelta di un classificatore rispetto all'altro.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23/?loc=it)

### Prerequisito

Si parte dal presupposto che siano state completate le lezioni precedenti e si disponga di un insieme di dati pulito nella cartella `data` chiamato _clean_cuisine.csv_ nella radice di questa cartella di 4 lezioni.

### Preparazione

Il file _notebook.ipynb_ è stato caricato con l'insieme di dati pulito ed è stato diviso in dataframe di dati X e y, pronti per il processo di creazione del modello.

## Una mappa di classificazione

In precedenza, si sono apprese le varie opzioni a disposizione durante la classificazione dei dati utilizzando il cheat sheet di Microsoft. Scikit-learn offre un cheat sheet simile, ma più granulare che può aiutare ulteriormente a restringere i propri stimatori (un altro termine per i classificatori):

![Mappa ML da Scikit-learn](../images/map.png)
> Suggerimento: [visitare questa mappa online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) e fare clic lungo il percorso per leggere la documentazione.

### Il piano

Questa mappa è molto utile una volta che si ha una chiara comprensione dei propri dati, poiché si può "camminare" lungo i suoi percorsi verso una decisione:

- Ci sono >50 campioni
- Si vuole pronosticare una categoria
- I dati sono etichettati
- Ci sono meno di 100K campioni
- ✨ Si può scegliere un SVC lineare
- Se non funziona, visto che ci sono dati numerici
   - Si può provare un ✨ KNeighbors Classifier
      - Se non funziona, si prova ✨ SVC e ✨ Classificatori di ensemble

Questo è un percorso molto utile da seguire.

## Esercizio: dividere i dati

Seguendo questo percorso, si dovrebbe iniziare importando alcune librerie da utilizzare.

1. Importare le librerie necessarie:

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
   import numpy as np
   ```

1. Dividere i dati per allenamento e test:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
   ```

## Classificatore lineare SVC

Il clustering Support-Vector (SVC) è figlio della famiglia di tecniche ML Support-Vector (ulteriori informazioni su queste di seguito). In questo metodo, si può scegliere un "kernel" per decidere come raggruppare le etichette. Il parametro 'C' si riferisce alla 'regolarizzazione' che regola l'influenza dei parametri. Il kernel può essere uno dei [tanti](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); qui si imposta su 'lineare' per assicurarsi di sfruttare l'SVC lineare. Il valore predefinito di probabilità è 'false'; qui si imposta su 'true' per raccogliere stime di probabilità. Si imposta lo stato casuale su "0" per mescolare i dati per ottenere le probabilità.

### Esercizio: applicare una SVC lineare

Iniziare creando un array di classificatori. Si aggiungerà progressivamente a questo array durante il test.

1. Iniziare con un SVC lineare:

   ```python
   C = 10
   # Create different classifiers.
   classifiers = {
       'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
   }
   ```

2. Addestrare il modello utilizzando Linear SVC e stampare un rapporto:

   ```python
   n_classifiers = len(classifiers)

   for index, (name, classifier) in enumerate(classifiers.items()):
       classifier.fit(X_train, np.ravel(y_train))

       y_pred = classifier.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
       print(classification_report(y_test,y_pred))
   ```

   Il risultato è abbastanza buono:

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

## Classificatore K-Neighbors

K-Neighbors fa parte della famiglia dei metodi ML "neighbors" (vicini), che possono essere utilizzati sia per l'apprendimento supervisionato che non supervisionato. In questo metodo, viene creato un numero predefinito di punti e i dati vengono raccolti attorno a questi punti in modo tale da poter prevedere etichette generalizzate per i dati.

### Esercizio: applicare il classificatore K-Neighbors

Il classificatore precedente era buono e funzionava bene con i dati, ma forse si può ottenere una maggiore precisione. Provare un classificatore K-Neighbors.

1. Aggiungere una riga all'array classificatore (aggiungere una virgola dopo l'elemento Linear SVC):

   ```python
   'KNN classifier': KNeighborsClassifier(C),
   ```

   Il risultato è un po' peggio:

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

   ✅ Scoprire [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Classificatore Support Vector

I classificatori Support-Vector fanno parte della famiglia di metodi ML [Support-Vector Machine](https://it.wikipedia.org/wiki/Macchine_a_vettori_di_supporto) utilizzati per le attività di classificazione e regressione. Le SVM "mappano esempi di addestramento in punti nello spazio" per massimizzare la distanza tra due categorie. I dati successivi vengono mappati in questo spazio in modo da poter prevedere la loro categoria.

### Esercizio: applicare un classificatore di vettori di supporto

Si prova a ottenere una precisione leggermente migliore con un classificatore di vettori di supporto.

1. Aggiungere una virgola dopo l'elemento K-Neighbors, quindi aggiungere questa riga:

   ```python
   'SVC': SVC(),
   ```

   Il risultato è abbastanza buono!

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

   ✅ Scoprire i vettori di [supporto](https://scikit-learn.org/stable/modules/svm.html#svm)

## Classificatori ensamble

Si segue il percorso fino alla fine, anche se il test precedente è stato abbastanza buono. Si provano un po' di classificatori di ensemble, nello specifico Random Forest e AdaBoost:

```python
'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Il risultato è molto buono, soprattutto per Random Forest:

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

✅ Ulteriori informazioni sui [classificatori di ensemble](https://scikit-learn.org/stable/modules/ensemble.html)

Questo metodo di Machine Learning "combina le previsioni di diversi stimatori di base" per migliorare la qualità del modello. In questo  esempio, si è utilizzato Random Trees e AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), un metodo di calcolo della media, costruisce una "foresta" di "alberi decisionali" infusi di casualità per evitare il sovradattamento. Il parametro n_estimators è impostato sul numero di alberi.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) adatta un classificatore a un insieme di dati e quindi adatta le copie di quel classificatore allo stesso insieme di dati. Si concentra sui pesi degli elementi classificati in modo errato e regola l'adattamento per il successivo classificatore da correggere.

---

## 🚀 Sfida

Ognuna di queste tecniche ha un gran numero di parametri che si possono modificare. Ricercare i parametri predefiniti di ciascuno e pensare a cosa significherebbe modificare questi parametri per la qualità del modello.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24/?loc=it)

## Revisione e Auto Apprendimento

C'è molto gergo in queste lezioni, quindi si prenda un minuto per rivedere [questo elenco](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) di terminologia utile!

## Compito

[Giocore coi parametri](assignment.it.md)
