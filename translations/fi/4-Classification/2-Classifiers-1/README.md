<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T00:44:45+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "fi"
}
-->
# Ruokakulttuuriluokittelijat 1

Tässä oppitunnissa käytät edellisessä oppitunnissa tallentamaasi tasapainoista ja siistiä datasettiä, joka käsittelee ruokakulttuureja.

Käytät tätä datasettiä eri luokittelijoiden kanssa _ennustaaksesi tietyn kansallisen ruokakulttuurin ainesosaryhmän perusteella_. Samalla opit lisää tavoista, joilla algoritmeja voidaan hyödyntää luokittelutehtävissä.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)
# Valmistelu

Jos olet suorittanut [Oppitunnin 1](../1-Introduction/README.md), varmista, että _cleaned_cuisines.csv_-tiedosto on olemassa juurihakemistossa `/data` näitä neljää oppituntia varten.

## Harjoitus - ennusta kansallinen ruokakulttuuri

1. Työskentele tämän oppitunnin _notebook.ipynb_-kansiossa ja tuo tiedosto sekä Pandas-kirjasto:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data näyttää tältä:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Tuo nyt lisää kirjastoja:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Jaa X- ja y-koordinaatit kahteen datafreimiin koulutusta varten. `cuisine` voi olla labelien datafreimi:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Se näyttää tältä:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Poista `Unnamed: 0`-sarake ja `cuisine`-sarake käyttämällä `drop()`. Tallenna loput tiedot koulutettaviksi ominaisuuksiksi:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ominaisuudet näyttävät tältä:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nyt olet valmis kouluttamaan mallisi!

## Luokittelijan valinta

Kun datasi on puhdas ja valmis koulutukseen, sinun täytyy päättää, mitä algoritmia käytät tehtävään.

Scikit-learn ryhmittelee luokittelun ohjatun oppimisen alle, ja tässä kategoriassa on monia tapoja luokitella. [Vaihtoehtojen määrä](https://scikit-learn.org/stable/supervised_learning.html) voi aluksi tuntua hämmentävältä. Seuraavat menetelmät sisältävät luokittelutekniikoita:

- Lineaariset mallit
- Tukivektorikoneet
- Stokastinen gradienttilaskenta
- Lähimmät naapurit
- Gaussin prosessit
- Päätöspuut
- Yhdistelmämallit (äänestysluokittelija)
- Moniluokka- ja monitulostusalgoritmit (moniluokka- ja monilabel-luokittelu, moniluokka-monitulostusluokittelu)

> Voit myös käyttää [neuroverkkoja datan luokitteluun](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), mutta se ei kuulu tämän oppitunnin aihepiiriin.

### Minkä luokittelijan valita?

Minkä luokittelijan siis valitset? Usein useiden kokeileminen ja hyvän tuloksen etsiminen on tapa testata. Scikit-learn tarjoaa [vertailun rinnakkain](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) luodulla datasetillä, jossa verrataan KNeighbors, SVC kahdella tavalla, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ja QuadraticDiscriminationAnalysis, ja tulokset visualisoidaan:

![luokittelijoiden vertailu](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Kuva Scikit-learnin dokumentaatiosta

> AutoML ratkaisee tämän ongelman kätevästi suorittamalla nämä vertailut pilvessä, jolloin voit valita parhaan algoritmin datallesi. Kokeile [täällä](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Parempi lähestymistapa

Parempi tapa kuin arvaaminen on seurata ladattavaa [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tässä huomataan, että moniluokkaongelmaamme varten meillä on joitakin vaihtoehtoja:

![moniluokkaongelmien huijauslista](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Osa Microsoftin algoritmien huijauslistasta, joka käsittelee moniluokkaluokitteluvaihtoehtoja

✅ Lataa tämä huijauslista, tulosta se ja ripusta seinällesi!

### Perustelu

Katsotaan, voimmeko perustella eri lähestymistapoja annettujen rajoitusten perusteella:

- **Neuroverkot ovat liian raskaita**. Puhdas mutta minimaalinen datasetti ja se, että koulutus tapahtuu paikallisesti notebookien kautta, tekevät neuroverkoista liian raskaita tähän tehtävään.
- **Ei kaksiluokkaista luokittelijaa**. Emme käytä kaksiluokkaista luokittelijaa, joten se sulkee pois one-vs-all-menetelmän.
- **Päätöspuu tai logistinen regressio voisi toimia**. Päätöspuu voisi toimia, tai logistinen regressio moniluokkaiselle datalle.
- **Moniluokkaiset Boosted Decision Trees ratkaisevat eri ongelman**. Moniluokkainen Boosted Decision Tree sopii parhaiten ei-parametrisiin tehtäviin, kuten tehtäviin, jotka on suunniteltu luomaan sijoituksia, joten se ei ole hyödyllinen meille.

### Scikit-learnin käyttö

Käytämme Scikit-learnia datan analysointiin. On kuitenkin monia tapoja käyttää logistista regressiota Scikit-learnissa. Katso [parametrit, jotka voit asettaa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Käytännössä on kaksi tärkeää parametria - `multi_class` ja `solver` - jotka meidän täytyy määrittää, kun pyydämme Scikit-learnia suorittamaan logistisen regression. `multi_class`-arvo soveltaa tiettyä käyttäytymistä. Solverin arvo määrittää, mitä algoritmia käytetään. Kaikkia solvereita ei voi yhdistää kaikkiin `multi_class`-arvoihin.

Dokumentaation mukaan moniluokkaisessa tapauksessa koulutusalgoritmi:

- **Käyttää one-vs-rest (OvR) -menetelmää**, jos `multi_class`-vaihtoehto on asetettu `ovr`
- **Käyttää ristientropiahäviötä**, jos `multi_class`-vaihtoehto on asetettu `multinomial`. (Tällä hetkellä `multinomial`-vaihtoehto on tuettu vain ‘lbfgs’, ‘sag’, ‘saga’ ja ‘newton-cg’ solvereilla.)

> 🎓 'Menetelmä' voi olla joko 'ovr' (one-vs-rest) tai 'multinomial'. Koska logistinen regressio on suunniteltu tukemaan binääriluokittelua, nämä menetelmät auttavat sitä käsittelemään paremmin moniluokkaluokittelutehtäviä. [lähde](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' määritellään "algoritmiksi, jota käytetään optimointiongelmassa". [lähde](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn tarjoaa tämän taulukon selittämään, miten solverit käsittelevät eri haasteita, joita eri datarakenteet esittävät:

![solverit](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Harjoitus - jaa data

Voimme keskittyä logistiseen regressioon ensimmäisessä koulutuskokeilussamme, koska opit siitä äskettäin edellisessä oppitunnissa.
Jaa datasi koulutus- ja testiryhmiin kutsumalla `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Harjoitus - sovella logistista regressiota

Koska käytät moniluokkaista tapausta, sinun täytyy valita, mitä _menetelmää_ käytät ja mitä _solveria_ asetat. Käytä LogisticRegressionia moniluokka-asetuksella ja **liblinear**-solveria koulutukseen.

1. Luo logistinen regressio, jossa multi_class on asetettu `ovr` ja solver asetettu `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Kokeile eri solveria, kuten `lbfgs`, joka on usein asetettu oletusarvoksi
> Huomaa, käytä Pandasin [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) -funktiota litistääksesi datasi tarvittaessa.
Tarkkuus on hyvä, yli **80%**!

1. Voit nähdä tämän mallin toiminnassa testaamalla yhtä riviä (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Tulos tulostetaan:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Kokeile eri rivinumeroa ja tarkista tulokset

1. Syvemmälle mentäessä voit tarkistaa tämän ennusteen tarkkuuden:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Tulos tulostetaan - Intialainen keittiö on paras arvaus, hyvällä todennäköisyydellä:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Voitko selittää, miksi malli on melko varma, että kyseessä on intialainen keittiö?

1. Saat lisää yksityiskohtia tulostamalla luokitteluraportin, kuten teit regressio-opetuksissa:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | tarkkuus | recall | f1-score | tuki    |
    | ------------ | -------- | ------ | -------- | ------- |
    | chinese      | 0.73     | 0.71   | 0.72     | 229     |
    | indian       | 0.91     | 0.93   | 0.92     | 254     |
    | japanese     | 0.70     | 0.75   | 0.72     | 220     |
    | korean       | 0.86     | 0.76   | 0.81     | 242     |
    | thai         | 0.79     | 0.85   | 0.82     | 254     |
    | tarkkuus     | 0.80     | 1199   |          |         |
    | keskiarvo    | 0.80     | 0.80   | 0.80     | 1199    |
    | painotettu   | 0.80     | 0.80   | 0.80     | 1199    |

## 🚀Haaste

Tässä oppitunnissa käytit siivottua dataasi rakentaaksesi koneoppimismallin, joka voi ennustaa kansallisen keittiön ainesosien perusteella. Käytä aikaa tutkiaksesi Scikit-learnin tarjoamia monia vaihtoehtoja datan luokitteluun. Syvenny tarkemmin 'solver'-käsitteeseen ymmärtääksesi, mitä kulissien takana tapahtuu.

## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus & Itseopiskelu

Tutustu hieman tarkemmin logistisen regression matematiikkaan [tässä oppitunnissa](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Tehtävä 

[Tutki solvereita](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.