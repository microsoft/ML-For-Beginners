<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T00:50:41+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "fi"
}
-->
# Ruokakulttuuriluokittelijat 2

Tässä toisessa luokittelutunnissa tutustut tarkemmin tapoihin luokitella numeerista dataa. Opit myös, mitä seurauksia on sillä, että valitset yhden luokittelijan toisen sijaan.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

### Esitiedot

Oletamme, että olet suorittanut aiemmat oppitunnit ja sinulla on puhdistettu datasetti `data`-kansiossasi nimeltä _cleaned_cuisines.csv_, joka sijaitsee tämän neljän oppitunnin kansion juurihakemistossa.

### Valmistelut

Olemme ladanneet _notebook.ipynb_-tiedostosi puhdistetulla datasetillä ja jakaneet sen X- ja y-datafreimeihin, jotka ovat valmiita mallin rakennusprosessia varten.

## Luokittelukartta

Aiemmin opit eri vaihtoehdoista datan luokitteluun Microsoftin huijauslistan avulla. Scikit-learn tarjoaa vastaavan, mutta tarkemman huijauslistan, joka voi auttaa kaventamaan valintaa luokittelijoiden (toinen termi estimointimenetelmille) välillä:

![ML-kartta Scikit-learnista](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Vinkki: [vieraile kartassa verkossa](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ja klikkaa polkuja lukeaksesi dokumentaatiota.

### Suunnitelma

Tämä kartta on erittäin hyödyllinen, kun ymmärrät datasi hyvin, sillä voit "kulkea" sen polkuja pitkin päätökseen:

- Meillä on >50 näytettä
- Haluamme ennustaa kategorian
- Meillä on merkitty data
- Meillä on alle 100K näytettä
- ✨ Voimme valita Linear SVC:n
- Jos se ei toimi, koska meillä on numeerista dataa
    - Voimme kokeilla ✨ KNeighbors-luokittelijaa 
      - Jos se ei toimi, kokeile ✨ SVC:tä ja ✨ Ensemble-luokittelijoita

Tämä on erittäin hyödyllinen polku seurattavaksi.

## Harjoitus - jaa data

Seuraamalla tätä polkua meidän tulisi aloittaa tarvittavien kirjastojen tuonnilla.

1. Tuo tarvittavat kirjastot:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Jaa koulutus- ja testidatasi:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC -luokittelija

Support-Vector Clustering (SVC) kuuluu Support-Vector Machines -perheeseen ML-tekniikoissa (lisätietoja alla). Tässä menetelmässä voit valita "kernelin" päättääksesi, miten etiketit ryhmitellään. 'C'-parametri viittaa 'regularisointiin', joka säätelee parametrien vaikutusta. Kernel voi olla yksi [useista](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); tässä asetamme sen 'lineaariseksi' varmistaaksemme, että hyödynnämme lineaarista SVC:tä. Todennäköisyys oletuksena on 'false'; tässä asetamme sen 'true' saadaksemme todennäköisyysarvioita. Asetamme satunnaistilan '0':ksi sekoittaaksemme datan todennäköisyyksien saamiseksi.

### Harjoitus - käytä lineaarista SVC:tä

Aloita luomalla luokittelijoiden taulukko. Lisäät tähän taulukkoon asteittain, kun testaamme.

1. Aloita lineaarisella SVC:llä:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Kouluta mallisi käyttäen lineaarista SVC:tä ja tulosta raportti:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Tulokset ovat melko hyviä:

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

## K-Neighbors -luokittelija

K-Neighbors kuuluu ML-menetelmien "naapurit"-perheeseen, jota voidaan käyttää sekä valvottuun että valvomattomaan oppimiseen. Tässä menetelmässä määritellään ennalta määrätty määrä pisteitä, ja data kerätään näiden pisteiden ympärille siten, että yleistetyt etiketit voidaan ennustaa datalle.

### Harjoitus - käytä K-Neighbors -luokittelijaa

Edellinen luokittelija oli hyvä ja toimi hyvin datan kanssa, mutta ehkä voimme saada paremman tarkkuuden. Kokeile K-Neighbors -luokittelijaa.

1. Lisää rivi luokittelijataulukkoon (lisää pilkku Linear SVC -kohdan jälkeen):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Tulokset ovat hieman huonommat:

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

    ✅ Lue lisää [K-Neighborsista](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector -luokittelija

Support-Vector -luokittelijat kuuluvat [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) -perheeseen ML-menetelmissä, joita käytetään luokittelu- ja regressiotehtäviin. SVM:t "karttavat koulutusesimerkit pisteiksi avaruudessa" maksimoidakseen etäisyyden kahden kategorian välillä. Seuraava data kartataan tähän avaruuteen, jotta sen kategoria voidaan ennustaa.

### Harjoitus - käytä Support Vector -luokittelijaa

Kokeillaan hieman parempaa tarkkuutta Support Vector -luokittelijalla.

1. Lisää pilkku K-Neighbors -kohdan jälkeen ja lisää tämä rivi:

    ```python
    'SVC': SVC(),
    ```

    Tulokset ovat erittäin hyviä!

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

    ✅ Lue lisää [Support-Vectorsista](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble-luokittelijat

Seurataan polkua aivan loppuun asti, vaikka edellinen testi oli erittäin hyvä. Kokeillaan joitakin 'Ensemble-luokittelijoita', erityisesti Random Forestia ja AdaBoostia:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Tulokset ovat erittäin hyviä, erityisesti Random Forestin osalta:

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

✅ Lue lisää [Ensemble-luokittelijoista](https://scikit-learn.org/stable/modules/ensemble.html)

Tämä koneoppimismenetelmä "yhdistää useiden perusestimointimenetelmien ennusteet" parantaakseen mallin laatua. Esimerkissämme käytimme Random Trees -menetelmää ja AdaBoostia. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), keskiarvomenetelmä, rakentaa "metsän" "päätöspuista", joihin lisätään satunnaisuutta ylisovituksen välttämiseksi. n_estimators-parametri asetetaan puiden määräksi.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) sovittaa luokittelijan datasettiin ja sovittaa kopioita tästä luokittelijasta samaan datasettiin. Se keskittyy väärin luokiteltujen kohteiden painoihin ja säätää seuraavan luokittelijan sovitusta korjatakseen.

---

## 🚀Haaste

Jokaisella näistä tekniikoista on suuri määrä parametreja, joita voit säätää. Tutki kunkin oletusparametreja ja mieti, mitä näiden parametrien säätäminen tarkoittaisi mallin laadulle.

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Näissä oppitunneissa on paljon ammattikieltä, joten ota hetki aikaa tarkastellaksesi [tätä listaa](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) hyödyllisistä termeistä!

## Tehtävä 

[Parametrien säätö](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.