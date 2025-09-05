<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T13:05:40+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "sl"
}
-->
# Razvrščevalniki kuhinj 1

V tej lekciji boste uporabili podatkovni niz, ki ste ga shranili v prejšnji lekciji, poln uravnoteženih in čistih podatkov o kuhinjah.

Ta podatkovni niz boste uporabili z različnimi razvrščevalniki, da _napoveste določeno nacionalno kuhinjo na podlagi skupine sestavin_. Med tem boste spoznali več o tem, kako lahko algoritme uporabimo za naloge razvrščanja.

## [Predlekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)
# Priprava

Če ste zaključili [Lekcijo 1](../1-Introduction/README.md), preverite, ali datoteka _cleaned_cuisines.csv_ obstaja v korenskem imeniku `/data` za te štiri lekcije.

## Vaja - napoved nacionalne kuhinje

1. V mapi _notebook.ipynb_ te lekcije uvozite to datoteko skupaj s knjižnico Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Podatki izgledajo takole:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Zdaj uvozite še nekaj knjižnic:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Razdelite koordinate X in y v dva podatkovna okvira za učenje. `cuisine` lahko uporabite kot podatkovni okvir z oznakami:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Videti bo takole:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstranite stolpca `Unnamed: 0` in `cuisine` z uporabo funkcije `drop()`. Preostale podatke shranite kot značilnosti za učenje:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše značilnosti izgledajo takole:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Zdaj ste pripravljeni na učenje modela!

## Izbira razvrščevalnika

Zdaj, ko so vaši podatki čisti in pripravljeni za učenje, morate izbrati algoritem za nalogo.

Scikit-learn razvršča razvrščanje pod Nadzorovano učenje, v tej kategoriji pa najdete veliko načinov za razvrščanje. [Raznolikost](https://scikit-learn.org/stable/supervised_learning.html) je na prvi pogled precej osupljiva. Naslednje metode vključujejo tehnike razvrščanja:

- Linearni modeli
- Podporni vektorski stroji
- Stohastični gradientni spust
- Najbližji sosedje
- Gaussovi procesi
- Odločitvena drevesa
- Metode ansambla (glasovalni razvrščevalnik)
- Večrazredni in večizhodni algoritmi (večrazredna in večoznačna razvrstitev, večrazredna-večizhodna razvrstitev)

> Za razvrščanje podatkov lahko uporabite tudi [nevronske mreže](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), vendar to presega obseg te lekcije.

### Kateri razvrščevalnik izbrati?

Torej, kateri razvrščevalnik izbrati? Pogosto je smiselno preizkusiti več razvrščevalnikov in iskati najboljši rezultat. Scikit-learn ponuja [primerjavo](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na ustvarjenem podatkovnem nizu, kjer primerja KNeighbors, SVC na dva načina, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB in QuadraticDiscriminationAnalysis, pri čemer so rezultati vizualizirani:

![primerjava razvrščevalnikov](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafi, ustvarjeni v dokumentaciji Scikit-learn

> AutoML to težavo elegantno reši tako, da te primerjave izvaja v oblaku, kar vam omogoča izbiro najboljšega algoritma za vaše podatke. Preizkusite ga [tukaj](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Boljši pristop

Boljši način kot naključno ugibanje je, da sledite idejam na tem prenosljivem [ML plonk listu](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tukaj ugotovimo, da imamo za našo večrazredno težavo nekaj možnosti:

![plonk list za večrazredne težave](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Del Microsoftovega plonk lista algoritmov, ki podrobno opisuje možnosti za večrazredno razvrščanje

✅ Prenesite ta plonk list, natisnite ga in obesite na steno!

### Razmišljanje

Poglejmo, ali lahko z razmišljanjem izberemo različne pristope glede na omejitve, ki jih imamo:

- **Nevronske mreže so pretežke**. Glede na naš čist, a minimalen podatkovni niz in dejstvo, da izvajamo učenje lokalno prek beležk, so nevronske mreže pretežke za to nalogo.
- **Ni razvrščevalnika za dva razreda**. Ne uporabljamo razvrščevalnika za dva razreda, zato izključimo one-vs-all.
- **Odločitveno drevo ali logistična regresija bi lahko delovala**. Odločitveno drevo bi lahko delovalo, prav tako logistična regresija za večrazredne podatke.
- **Večrazredna izboljšana odločitvena drevesa rešujejo drugačen problem**. Večrazredna izboljšana odločitvena drevesa so najbolj primerna za neparametrične naloge, npr. naloge za ustvarjanje razvrstitev, zato za nas niso uporabna.

### Uporaba Scikit-learn 

Za analizo podatkov bomo uporabili Scikit-learn. Vendar pa obstaja veliko načinov za uporabo logistične regresije v Scikit-learn. Oglejte si [parametre za nastavitev](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

V bistvu sta dva pomembna parametra - `multi_class` in `solver` - ki ju moramo določiti, ko Scikit-learn prosimo za izvedbo logistične regresije. Vrednost `multi_class` določa določeno vedenje. Vrednost `solver` določa, kateri algoritem uporabiti. Vsi reševalci ne morejo biti združeni z vsemi vrednostmi `multi_class`.

Po dokumentaciji, v primeru večrazredne naloge, algoritem za učenje:

- **Uporablja shemo one-vs-rest (OvR)**, če je možnost `multi_class` nastavljena na `ovr`
- **Uporablja izgubo navzkrižne entropije**, če je možnost `multi_class` nastavljena na `multinomial`. (Trenutno možnost `multinomial` podpirajo samo reševalci ‘lbfgs’, ‘sag’, ‘saga’ in ‘newton-cg’.)

> 🎓 'Shema' tukaj je lahko 'ovr' (one-vs-rest) ali 'multinomial'. Ker je logistična regresija zasnovana za podporo binarni razvrstitvi, ji te sheme omogočajo boljše obravnavanje večrazrednih nalog razvrščanja. [vir](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' je definiran kot "algoritem za uporabo pri optimizacijskem problemu". [vir](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ponuja to tabelo za razlago, kako reševalci obravnavajo različne izzive, ki jih predstavljajo različne vrste podatkovnih struktur:

![reševalci](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Vaja - razdelitev podatkov

Osredotočimo se na logistično regresijo za naš prvi poskus učenja, saj ste se o njej nedavno učili v prejšnji lekciji.
Razdelite svoje podatke v skupine za učenje in testiranje z uporabo funkcije `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Vaja - uporaba logistične regresije

Ker uporabljate večrazredno nalogo, morate izbrati, katero _shemo_ uporabiti in kateri _reševalec_ nastaviti. Uporabite LogisticRegression z večrazredno nastavitvijo in reševalcem **liblinear** za učenje.

1. Ustvarite logistično regresijo z `multi_class` nastavljeno na `ovr` in reševalcem nastavljenim na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Preizkusite drugega reševalca, kot je `lbfgs`, ki je pogosto nastavljen kot privzet.
> Upoštevajte, uporabite funkcijo Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) za sploščitev vaših podatkov, kadar je to potrebno.
Natančnost je dobra pri več kot **80%**!

1. Ta model lahko preizkusite z eno vrstico podatkov (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Rezultat se izpiše:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Poskusite z drugo številko vrstice in preverite rezultate.

1. Če želite podrobneje raziskati, lahko preverite natančnost te napovedi:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Rezultat se izpiše - indijska kuhinja je najboljša ugotovitev z dobro verjetnostjo:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Ali lahko razložite, zakaj je model precej prepričan, da gre za indijsko kuhinjo?

1. Pridobite več podrobnosti z izpisom poročila o klasifikaciji, kot ste to storili pri lekcijah o regresiji:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | chinese      | 0.73      | 0.71   | 0.72     | 229     |
    | indian       | 0.91      | 0.93   | 0.92     | 254     |
    | japanese     | 0.70      | 0.75   | 0.72     | 220     |
    | korean       | 0.86      | 0.76   | 0.81     | 242     |
    | thai         | 0.79      | 0.85   | 0.82     | 254     |
    | accuracy     | 0.80      | 1199   |          |         |
    | macro avg    | 0.80      | 0.80   | 0.80     | 1199    |
    | weighted avg | 0.80      | 0.80   | 0.80     | 1199    |

## 🚀Izziv

V tej lekciji ste uporabili očiščene podatke za izdelavo modela strojnega učenja, ki lahko napove nacionalno kuhinjo na podlagi serije sestavin. Vzemite si čas in preberite številne možnosti, ki jih Scikit-learn ponuja za klasifikacijo podatkov. Podrobneje raziščite koncept 'solverja', da boste razumeli, kaj se dogaja v ozadju.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled & Samostojno učenje

Podrobneje raziščite matematiko za logistično regresijo v [tej lekciji](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Naloga 

[Preučite solverje](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.