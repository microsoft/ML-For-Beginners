<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "808a71076f76ae8f5458862a8edd9215",
  "translation_date": "2025-08-29T13:58:48+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "tl"
}
-->
# Mga Classifier ng Lutuin 2

Sa ikalawang aralin ng klasipikasyon na ito, mas marami kang matutuklasang paraan upang iklasipika ang numeric na datos. Malalaman mo rin ang mga epekto ng pagpili ng isang classifier kumpara sa iba.

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/23/)

### Kinakailangan

Inaakala namin na natapos mo na ang mga nakaraang aralin at mayroon kang nalinis na dataset sa iyong `data` folder na tinatawag na _cleaned_cuisines.csv_ sa root ng folder na ito na may 4 na aralin.

### Paghahanda

Na-load na namin ang iyong _notebook.ipynb_ file gamit ang nalinis na dataset at hinati ito sa X at y dataframes, handa na para sa proseso ng pagbuo ng modelo.

## Isang mapa ng klasipikasyon

Noong nakaraan, natutunan mo ang iba't ibang opsyon na mayroon ka kapag nagkaklasipika ng datos gamit ang cheat sheet ng Microsoft. Ang Scikit-learn ay nag-aalok ng katulad ngunit mas detalyadong cheat sheet na makakatulong upang mas paliitin ang iyong mga estimator (isa pang termino para sa classifiers):

![ML Map mula sa Scikit-learn](../../../../translated_images/map.e963a6a51349425ab107b38f6c7307eb4c0d0c7ccdd2e81a5e1919292bab9ac7.tl.png)
> Tip: [bisitahin ang mapa online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) at mag-click sa mga landas upang basahin ang dokumentasyon.

### Ang plano

Ang mapa na ito ay napaka-kapaki-pakinabang kapag malinaw na ang iyong pag-unawa sa datos, dahil maaari kang 'maglakad' sa mga landas nito patungo sa isang desisyon:

- Mayroon tayong >50 na sample
- Gusto nating hulaan ang isang kategorya
- Mayroon tayong labeled na datos
- Mayroon tayong mas kaunti sa 100K na sample
- ✨ Maaari tayong pumili ng Linear SVC
- Kung hindi ito gumana, dahil mayroon tayong numeric na datos
    - Maaari nating subukan ang ✨ KNeighbors Classifier 
      - Kung hindi ito gumana, subukan ang ✨ SVC at ✨ Ensemble Classifiers

Napaka-kapaki-pakinabang na landas na sundan.

## Ehersisyo - hatiin ang datos

Kasunod ng landas na ito, dapat tayong magsimula sa pag-import ng ilang mga library na gagamitin.

1. I-import ang mga kinakailangang library:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Hatiin ang iyong training at test data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Ang Support-Vector clustering (SVC) ay bahagi ng pamilya ng Support-Vector machines ng mga teknik sa ML (matuto pa tungkol dito sa ibaba). Sa pamamaraang ito, maaari kang pumili ng 'kernel' upang magpasya kung paano iklasipika ang mga label. Ang 'C' na parameter ay tumutukoy sa 'regularization' na nagre-regulate sa impluwensya ng mga parameter. Ang kernel ay maaaring isa sa [marami](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); dito natin ito itinatakda sa 'linear' upang matiyak na ginagamit natin ang linear SVC. Ang probability ay default na 'false'; dito natin ito itinatakda sa 'true' upang makakuha ng mga probability estimate. Itinatakda natin ang random state sa '0' upang i-shuffle ang datos para makakuha ng probabilities.

### Ehersisyo - gamitin ang linear SVC

Magsimula sa paglikha ng array ng mga classifier. Magdadagdag ka nang paunti-unti sa array na ito habang sinusubukan natin.

1. Magsimula sa Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. I-train ang iyong modelo gamit ang Linear SVC at i-print ang ulat:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Maganda ang resulta:

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

## K-Neighbors classifier

Ang K-Neighbors ay bahagi ng "neighbors" na pamilya ng mga pamamaraang ML, na maaaring gamitin para sa parehong supervised at unsupervised na pag-aaral. Sa pamamaraang ito, isang paunang natukoy na bilang ng mga punto ang nilikha at ang datos ay kinokolekta sa paligid ng mga puntong ito upang ang mga generalized na label ay mahulaan para sa datos.

### Ehersisyo - gamitin ang K-Neighbors classifier

Maganda ang naunang classifier at gumana nang maayos sa datos, ngunit baka mas mapabuti pa natin ang accuracy. Subukan ang K-Neighbors classifier.

1. Magdagdag ng linya sa iyong classifier array (magdagdag ng comma pagkatapos ng Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Ang resulta ay medyo mas mababa:

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

    ✅ Matuto tungkol sa [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Ang Support-Vector classifiers ay bahagi ng [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) na pamilya ng mga pamamaraang ML na ginagamit para sa mga gawain ng klasipikasyon at regression. Ang SVMs ay "nagmamapa ng mga training example sa mga punto sa espasyo" upang ma-maximize ang distansya sa pagitan ng dalawang kategorya. Ang mga susunod na datos ay minamapa sa espasyong ito upang mahulaan ang kanilang kategorya.

### Ehersisyo - gamitin ang Support Vector Classifier

Subukan natin ang mas magandang accuracy gamit ang Support Vector Classifier.

1. Magdagdag ng comma pagkatapos ng K-Neighbors item, at pagkatapos ay idagdag ang linyang ito:

    ```python
    'SVC': SVC(),
    ```

    Maganda ang resulta!

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

    ✅ Matuto tungkol sa [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Sundan natin ang landas hanggang sa dulo, kahit na maganda na ang naunang pagsusulit. Subukan natin ang ilang 'Ensemble Classifiers', partikular ang Random Forest at AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Maganda ang resulta, lalo na para sa Random Forest:

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

✅ Matuto tungkol sa [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Ang pamamaraang ito ng Machine Learning ay "pinagsasama ang mga prediksyon ng ilang base estimators" upang mapabuti ang kalidad ng modelo. Sa ating halimbawa, ginamit natin ang Random Trees at AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), isang averaging method, ay bumubuo ng 'forest' ng 'decision trees' na may kasamang randomness upang maiwasan ang overfitting. Ang n_estimators parameter ay itinakda sa bilang ng mga puno.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ay nagfi-fit ng classifier sa dataset at pagkatapos ay nagfi-fit ng mga kopya ng classifier na iyon sa parehong dataset. Nakatuon ito sa mga weights ng mga maling na-klasipikang item at ina-adjust ang fit para sa susunod na classifier upang maitama.

---

## 🚀Hamunin

Ang bawat isa sa mga pamamaraang ito ay may malaking bilang ng mga parameter na maaari mong i-tweak. Mag-research sa default na mga parameter ng bawat isa at pag-isipan kung ano ang magiging epekto ng pag-tweak ng mga parameter na ito sa kalidad ng modelo.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/24/)

## Review & Pag-aaral sa Sarili

Maraming jargon sa mga araling ito, kaya maglaan ng oras upang suriin ang [listahang ito](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ng mga kapaki-pakinabang na termino!

## Takdang Aralin 

[Parameter play](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.