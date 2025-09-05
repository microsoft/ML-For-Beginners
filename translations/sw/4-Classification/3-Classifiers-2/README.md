<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T16:23:09+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "sw"
}
-->
# Wainishi wa vyakula 2

Katika somo hili la pili la uainishaji, utachunguza njia zaidi za kuainisha data ya nambari. Pia utajifunza kuhusu athari za kuchagua mainishi moja badala ya jingine.

## [Jaribio la awali la somo](https://ff-quizzes.netlify.app/en/ml/)

### Mahitaji ya awali

Tunadhani kuwa umekamilisha masomo ya awali na una dataset iliyosafishwa katika folda yako ya `data` inayoitwa _cleaned_cuisines.csv_ katika mzizi wa folda hii ya masomo 4.

### Maandalizi

Tumeweka faili yako ya _notebook.ipynb_ na dataset iliyosafishwa na tumeigawanya katika fremu za data za X na y, tayari kwa mchakato wa kujenga modeli.

## Ramani ya uainishaji

Hapo awali, ulijifunza kuhusu chaguo mbalimbali unazoweza kutumia kuainisha data kwa kutumia karatasi ya udanganyifu ya Microsoft. Scikit-learn inatoa karatasi ya udanganyifu inayofanana, lakini ya kina zaidi, ambayo inaweza kusaidia zaidi kupunguza chaguo zako za makadirio (neno lingine kwa wainishi):

![Ramani ya ML kutoka Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Kidokezo: [tembelea ramani hii mtandaoni](https://scikit-learn.org/stable/tutorial/machine_learning_map/) na bonyeza njia zake kusoma nyaraka.

### Mpango

Ramani hii ni muhimu sana mara tu unapokuwa na uelewa wa wazi wa data yako, kwani unaweza 'kutembea' kwenye njia zake hadi kufikia uamuzi:

- Tuna sampuli >50
- Tunataka kutabiri kategoria
- Tuna data yenye lebo
- Tuna sampuli chini ya 100K
- ✨ Tunaweza kuchagua Linear SVC
- Ikiwa hiyo haifanyi kazi, kwa kuwa tuna data ya nambari
    - Tunaweza kujaribu ✨ KNeighbors Classifier 
      - Ikiwa hiyo haifanyi kazi, jaribu ✨ SVC na ✨ Ensemble Classifiers

Hii ni njia muhimu sana ya kufuata.

## Zoezi - gawanya data

Kwa kufuata njia hii, tunapaswa kuanza kwa kuingiza baadhi ya maktaba za kutumia.

1. Ingiza maktaba zinazohitajika:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Gawanya data yako ya mafunzo na majaribio:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Wainishi wa Linear SVC

Support-Vector clustering (SVC) ni sehemu ya familia ya Support-Vector machines ya mbinu za ML (jifunze zaidi kuhusu hizi hapa chini). Katika mbinu hii, unaweza kuchagua 'kernel' kuamua jinsi ya kuainisha lebo. Kipengele cha 'C' kinahusu 'regularization' ambacho kinadhibiti ushawishi wa vigezo. Kernel inaweza kuwa moja ya [kadhaa](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); hapa tunaiweka kuwa 'linear' ili kuhakikisha tunatumia Linear SVC. Uwezekano unakuwa 'false' kwa default; hapa tunaiweka kuwa 'true' ili kupata makadirio ya uwezekano. Tunaiweka random state kuwa '0' ili kuchanganya data kupata uwezekano.

### Zoezi - tumia Linear SVC

Anza kwa kuunda safu ya wainishi. Utaongeza hatua kwa hatua kwenye safu hii tunapojaribu. 

1. Anza na Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Funza modeli yako kwa kutumia Linear SVC na chapisha ripoti:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Matokeo ni mazuri:

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

## Wainishi wa K-Neighbors

K-Neighbors ni sehemu ya familia ya "neighbors" ya mbinu za ML, ambazo zinaweza kutumika kwa kujifunza kwa usimamizi na bila usimamizi. Katika mbinu hii, idadi ya alama zilizowekwa awali huundwa na data hukusanywa karibu na alama hizi ili lebo za jumla ziweze kutabiriwa kwa data.

### Zoezi - tumia wainishi wa K-Neighbors

Wainishi wa awali ulikuwa mzuri, na ulifanya kazi vizuri na data, lakini labda tunaweza kupata usahihi bora. Jaribu wainishi wa K-Neighbors.

1. Ongeza mstari kwenye safu yako ya wainishi (ongeza koma baada ya kipengele cha Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Matokeo ni kidogo mabaya:

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

    ✅ Jifunze kuhusu [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Wainishi wa Support Vector

Wainishi wa Support-Vector ni sehemu ya familia ya [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) ya mbinu za ML zinazotumika kwa kazi za uainishaji na urejeleaji. SVMs "huweka mifano ya mafunzo kwenye alama katika nafasi" ili kuongeza umbali kati ya kategoria mbili. Data inayofuata huwekwa kwenye nafasi hii ili kategoria yake iweze kutabiriwa.

### Zoezi - tumia wainishi wa Support Vector

Hebu jaribu kupata usahihi bora kidogo kwa wainishi wa Support Vector.

1. Ongeza koma baada ya kipengele cha K-Neighbors, kisha ongeza mstari huu:

    ```python
    'SVC': SVC(),
    ```

    Matokeo ni mazuri sana!

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

    ✅ Jifunze kuhusu [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Wainishi wa Ensemble

Hebu fuata njia hadi mwisho kabisa, ingawa jaribio la awali lilikuwa zuri sana. Hebu jaribu baadhi ya 'Ensemble Classifiers', hasa Random Forest na AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Matokeo ni mazuri sana, hasa kwa Random Forest:

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

✅ Jifunze kuhusu [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Mbinu hii ya Kujifunza kwa Mashine "inaunganisha makadirio ya wainishi kadhaa wa msingi" ili kuboresha ubora wa modeli. Katika mfano wetu, tulitumia Random Trees na AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), mbinu ya wastani, huunda 'msitu' wa 'miti ya maamuzi' yenye nasibu ili kuepuka overfitting. Kipengele cha n_estimators kimewekwa kwa idadi ya miti.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) huweka wainishi kwenye dataset na kisha huweka nakala za wainishi huo kwenye dataset hiyo hiyo. Inazingatia uzito wa vitu vilivyoainishwa vibaya na kurekebisha fit kwa wainishi unaofuata ili kusahihisha.

---

## 🚀Changamoto

Kila moja ya mbinu hizi ina idadi kubwa ya vigezo ambavyo unaweza kurekebisha. Tafiti vigezo vya default vya kila moja na fikiria kuhusu maana ya kurekebisha vigezo hivi kwa ubora wa modeli.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujisomea

Kuna msamiati mwingi katika masomo haya, kwa hivyo chukua muda kupitia [orodha hii](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ya istilahi muhimu!

## Kazi 

[Cheza na vigezo](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.