<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T13:04:51+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "hr"
}
-->
# Klasifikatori kuhinja 1

U ovoj lekciji koristit ćete skup podataka koji ste spremili iz prethodne lekcije, pun uravnoteženih i očišćenih podataka o kuhinjama.

Koristit ćete ovaj skup podataka s raznim klasifikatorima kako biste _predvidjeli određenu nacionalnu kuhinju na temelju grupe sastojaka_. Dok to radite, naučit ćete više o načinima na koje se algoritmi mogu koristiti za zadatke klasifikacije.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)
# Priprema

Pod pretpostavkom da ste završili [Lekciju 1](../1-Introduction/README.md), provjerite postoji li datoteka _cleaned_cuisines.csv_ u korijenskoj mapi `/data` za ove četiri lekcije.

## Vježba - predviđanje nacionalne kuhinje

1. Radite u mapi _notebook.ipynb_ ove lekcije i uvezite tu datoteku zajedno s Pandas bibliotekom:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Podaci izgledaju ovako:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sada uvezite još nekoliko biblioteka:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Podijelite X i y koordinate u dva dataframea za treniranje. `cuisine` može biti dataframe s oznakama:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Izgledat će ovako:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Izbacite stupce `Unnamed: 0` i `cuisine` koristeći `drop()`. Ostatak podataka spremite kao značajke za treniranje:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše značajke izgledaju ovako:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sada ste spremni za treniranje modela!

## Odabir klasifikatora

Sada kada su vaši podaci očišćeni i spremni za treniranje, morate odlučiti koji algoritam koristiti za zadatak.

Scikit-learn grupira klasifikaciju pod Nadzirano učenje, a u toj kategoriji postoji mnogo načina za klasifikaciju. [Raznolikost](https://scikit-learn.org/stable/supervised_learning.html) može na prvi pogled djelovati zbunjujuće. Sljedeće metode uključuju tehnike klasifikacije:

- Linearni modeli
- Strojevi za potporne vektore (SVM)
- Stohastički gradijentni spust
- Najbliži susjedi
- Gaussovi procesi
- Stabla odluke
- Metode ansambla (glasajući klasifikator)
- Višeklasni i višerezultatski algoritmi (višeklasna i višeznačna klasifikacija, višeklasna-višerezultatska klasifikacija)

> Također možete koristiti [neuronske mreže za klasifikaciju podataka](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ali to je izvan opsega ove lekcije.

### Koji klasifikator odabrati?

Dakle, koji klasifikator odabrati? Često je korisno isprobati nekoliko njih i tražiti dobar rezultat. Scikit-learn nudi [usporedbu](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na stvorenom skupu podataka, uspoređujući KNeighbors, SVC na dva načina, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB i QuadraticDiscriminationAnalysis, prikazujući rezultate vizualno:

![usporedba klasifikatora](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafovi generirani u dokumentaciji Scikit-learn-a

> AutoML rješava ovaj problem jednostavno pokretanjem ovih usporedbi u oblaku, omogućujući vam odabir najboljeg algoritma za vaše podatke. Isprobajte [ovdje](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Bolji pristup

Bolji način od nasumičnog pogađanja je slijediti ideje iz ovog preuzimljivog [ML Cheat Sheeta](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Ovdje otkrivamo da za naš višeklasni problem imamo nekoliko izbora:

![cheatsheet za višeklasne probleme](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Dio Microsoftovog Algorithm Cheat Sheeta, koji detaljno opisuje opcije za višeklasnu klasifikaciju

✅ Preuzmite ovaj cheat sheet, isprintajte ga i objesite na zid!

### Razmišljanje

Pokušajmo razmotriti različite pristupe s obzirom na ograničenja koja imamo:

- **Neuronske mreže su preteške**. S obzirom na naš očišćeni, ali minimalni skup podataka i činjenicu da treniranje provodimo lokalno putem bilježnica, neuronske mreže su preteške za ovaj zadatak.
- **Nema klasifikatora za dvije klase**. Ne koristimo klasifikator za dvije klase, što isključuje one-vs-all.
- **Stablo odluke ili logistička regresija mogli bi raditi**. Stablo odluke moglo bi raditi, kao i logistička regresija za višeklasne podatke.
- **Višeklasna Boosted Decision Trees rješava drugi problem**. Višeklasno pojačano stablo odluke najprikladnije je za neparametarske zadatke, npr. zadatke dizajnirane za izradu rangiranja, pa nam nije korisno.

### Korištenje Scikit-learn-a

Koristit ćemo Scikit-learn za analizu naših podataka. Međutim, postoji mnogo načina za korištenje logističke regresije u Scikit-learn-u. Pogledajte [parametre za prosljeđivanje](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

U osnovi, postoje dva važna parametra - `multi_class` i `solver` - koje trebamo specificirati kada tražimo od Scikit-learn-a da izvede logističku regresiju. Vrijednost `multi_class` primjenjuje određeno ponašanje. Vrijednost solvera određuje koji algoritam koristiti. Nisu svi solveri kompatibilni sa svim vrijednostima `multi_class`.

Prema dokumentaciji, u slučaju višeklasne klasifikacije, algoritam treniranja:

- **Koristi shemu one-vs-rest (OvR)**, ako je opcija `multi_class` postavljena na `ovr`
- **Koristi gubitak unakrsne entropije**, ako je opcija `multi_class` postavljena na `multinomial`. (Trenutno opciju `multinomial` podržavaju samo solveri ‘lbfgs’, ‘sag’, ‘saga’ i ‘newton-cg’.)

> 🎓 'Shema' ovdje može biti 'ovr' (one-vs-rest) ili 'multinomial'. Budući da je logistička regresija zapravo dizajnirana za podršku binarnoj klasifikaciji, ove sheme omogućuju joj bolje rukovanje zadacima višeklasne klasifikacije. [izvor](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' je definiran kao "algoritam koji se koristi u problemu optimizacije". [izvor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn nudi ovu tablicu kako bi objasnio kako solveri rješavaju različite izazove koje predstavljaju različite vrste struktura podataka:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Vježba - podjela podataka

Možemo se usredotočiti na logističku regresiju za naš prvi pokušaj treniranja jer ste nedavno naučili o njoj u prethodnoj lekciji.
Podijelite svoje podatke u grupe za treniranje i testiranje pozivom `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Vježba - primjena logističke regresije

Budući da koristite slučaj višeklasne klasifikacije, trebate odabrati koju _shemu_ koristiti i koji _solver_ postaviti. Koristite LogisticRegression s višeklasnim postavkama i **liblinear** solverom za treniranje.

1. Kreirajte logističku regresiju s multi_class postavljenim na `ovr` i solverom postavljenim na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Isprobajte drugi solver poput `lbfgs`, koji je često postavljen kao zadani
> Napomena, koristite Pandasovu funkciju [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) za izravnavanje podataka kada je to potrebno.
Točnost je dobra, preko **80%**!

1. Možete vidjeti ovaj model u akciji testiranjem jednog retka podataka (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultat se ispisuje:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Isprobajte drugi broj retka i provjerite rezultate.

1. Ako želite dublje istražiti, možete provjeriti točnost ove predikcije:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultat se ispisuje - Indijska kuhinja je najbolja pretpostavka, s dobrom vjerojatnošću:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Možete li objasniti zašto je model prilično siguran da se radi o indijskoj kuhinji?

1. Dobijte više detalja ispisivanjem izvještaja o klasifikaciji, kao što ste radili u lekcijama o regresiji:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | preciznost | odziv | f1-rezultat | podrška |
    | ------------ | ---------- | ----- | ----------- | ------- |
    | chinese      | 0.73       | 0.71  | 0.72        | 229     |
    | indian       | 0.91       | 0.93  | 0.92        | 254     |
    | japanese     | 0.70       | 0.75  | 0.72        | 220     |
    | korean       | 0.86       | 0.76  | 0.81        | 242     |
    | thai         | 0.79       | 0.85  | 0.82        | 254     |
    | točnost      | 0.80       | 1199  |             |         |
    | prosjek makro| 0.80       | 0.80  | 0.80        | 1199    |
    | prosjek tež. | 0.80       | 0.80  | 0.80        | 1199    |

## 🚀Izazov

U ovoj lekciji koristili ste očišćene podatke za izradu modela strojnog učenja koji može predvidjeti nacionalnu kuhinju na temelju niza sastojaka. Odvojite malo vremena da proučite mnoge opcije koje Scikit-learn nudi za klasifikaciju podataka. Dublje istražite koncept 'solver' kako biste razumjeli što se događa iza kulisa.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Dublje istražite matematiku iza logističke regresije u [ovoj lekciji](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Zadatak 

[Proučite solvere](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.