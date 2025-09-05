<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T16:18:25+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "sk"
}
-->
# Klasifikátory kuchýň 1

V tejto lekcii použijete dataset, ktorý ste si uložili z predchádzajúcej lekcie, plný vyvážených a čistých údajov o kuchyniach.

Tento dataset použijete s rôznymi klasifikátormi na _predpovedanie národnej kuchyne na základe skupiny ingrediencií_. Pri tom sa dozviete viac o tom, ako môžu byť algoritmy využívané na klasifikačné úlohy.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)
# Príprava

Za predpokladu, že ste dokončili [Lekciu 1](../1-Introduction/README.md), uistite sa, že súbor _cleaned_cuisines.csv_ existuje v koreňovom priečinku `/data` pre tieto štyri lekcie.

## Cvičenie - predpovedanie národnej kuchyne

1. Pracujte v priečinku _notebook.ipynb_ tejto lekcie, importujte tento súbor spolu s knižnicou Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Údaje vyzerajú takto:

|     | Unnamed: 0 | kuchyňa | mandľa | angelika | aníz | anízové semeno | jablko | jablkový brandy | marhuľa | armagnac | ... | whiskey | biely chlieb | biele víno | celozrnná pšeničná múka | víno | drevo | yam | droždie | jogurt | cuketa |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indická | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indická | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indická | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indická | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indická | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Teraz importujte niekoľko ďalších knižníc:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Rozdeľte X a y súradnice do dvoch dataframeov na trénovanie. `cuisine` môže byť dataframe s označeniami:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Bude to vyzerať takto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstráňte stĺpec `Unnamed: 0` a stĺpec `cuisine` pomocou `drop()`. Zvyšok údajov uložte ako trénovateľné vlastnosti:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše vlastnosti vyzerajú takto:

|      | mandľa | angelika | aníz | anízové semeno | jablko | jablkový brandy | marhuľa | armagnac | artemisia | artičok |  ... | whiskey | biely chlieb | biele víno | celozrnná pšeničná múka | víno | drevo | yam | droždie | jogurt | cuketa |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Teraz ste pripravení trénovať váš model!

## Výber klasifikátora

Keď sú vaše údaje čisté a pripravené na trénovanie, musíte sa rozhodnúť, ktorý algoritmus použiť na túto úlohu.

Scikit-learn zaraďuje klasifikáciu pod Supervised Learning, a v tejto kategórii nájdete mnoho spôsobov klasifikácie. [Rozmanitosť](https://scikit-learn.org/stable/supervised_learning.html) je na prvý pohľad dosť zarážajúca. Nasledujúce metódy zahŕňajú techniky klasifikácie:

- Lineárne modely
- Support Vector Machines
- Stochastický gradientný zostup
- Najbližší susedia
- Gaussovské procesy
- Rozhodovacie stromy
- Ensemble metódy (hlasovací klasifikátor)
- Multiclass a multioutput algoritmy (multiclass a multilabel klasifikácia, multiclass-multioutput klasifikácia)

> Na klasifikáciu údajov môžete použiť aj [neurónové siete](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ale to je mimo rozsah tejto lekcie.

### Aký klasifikátor zvoliť?

Takže, ktorý klasifikátor by ste si mali vybrať? Často je dobré vyskúšať niekoľko a hľadať dobrý výsledok. Scikit-learn ponúka [porovnanie vedľa seba](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na vytvorenom datasete, kde porovnáva KNeighbors, SVC dvoma spôsobmi, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB a QuadraticDiscriminationAnalysis, pričom výsledky vizualizuje:

![porovnanie klasifikátorov](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafy generované na dokumentácii Scikit-learn

> AutoML tento problém elegantne rieši tým, že vykonáva tieto porovnania v cloude, čo vám umožňuje vybrať najlepší algoritmus pre vaše údaje. Vyskúšajte to [tu](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Lepší prístup

Lepší spôsob ako náhodne hádať je však nasledovať nápady z tejto stiahnuteľnej [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Tu zistíme, že pre náš multiclass problém máme niekoľko možností:

![cheatsheet pre multiclass problémy](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Časť Microsoftovej Algorithm Cheat Sheet, ktorá podrobne opisuje možnosti multiclass klasifikácie

✅ Stiahnite si tento cheat sheet, vytlačte ho a zaveste na stenu!

### Úvahy

Pozrime sa, či dokážeme rozumne zhodnotiť rôzne prístupy vzhľadom na obmedzenia, ktoré máme:

- **Neurónové siete sú príliš náročné**. Vzhľadom na náš čistý, ale minimálny dataset a fakt, že trénovanie prebieha lokálne cez notebooky, sú neurónové siete príliš náročné na túto úlohu.
- **Žiadny dvojtriedny klasifikátor**. Nepoužívame dvojtriedny klasifikátor, takže to vylučuje one-vs-all.
- **Rozhodovací strom alebo logistická regresia by mohli fungovať**. Rozhodovací strom by mohol fungovať, alebo logistická regresia pre multiclass údaje.
- **Multiclass Boosted Decision Trees riešia iný problém**. Multiclass Boosted Decision Tree je najvhodnejší pre neparametrické úlohy, napr. úlohy určené na vytváranie rebríčkov, takže pre nás nie je užitočný.

### Použitie Scikit-learn 

Budeme používať Scikit-learn na analýzu našich údajov. Existuje však mnoho spôsobov, ako použiť logistickú regresiu v Scikit-learn. Pozrite sa na [parametre na nastavenie](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

V podstate existujú dva dôležité parametre - `multi_class` a `solver` - ktoré musíme špecifikovať, keď požiadame Scikit-learn o vykonanie logistickej regresie. Hodnota `multi_class` aplikuje určitú logiku. Hodnota solveru určuje, aký algoritmus sa použije. Nie všetky solvery môžu byť spárované so všetkými hodnotami `multi_class`.

Podľa dokumentácie, v prípade multiclass, tréningový algoritmus:

- **Používa schému one-vs-rest (OvR)**, ak je možnosť `multi_class` nastavená na `ovr`
- **Používa cross-entropy loss**, ak je možnosť `multi_class` nastavená na `multinomial`. (Momentálne je možnosť `multinomial` podporovaná iba solvermi ‘lbfgs’, ‘sag’, ‘saga’ a ‘newton-cg’.)

> 🎓 Schéma tu môže byť buď 'ovr' (one-vs-rest) alebo 'multinomial'. Keďže logistická regresia je skutočne navrhnutá na podporu binárnej klasifikácie, tieto schémy jej umožňujú lepšie zvládnuť úlohy multiclass klasifikácie. [zdroj](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Solver je definovaný ako "algoritmus, ktorý sa použije na optimalizačný problém". [zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ponúka túto tabuľku na vysvetlenie, ako solvery zvládajú rôzne výzvy, ktoré predstavujú rôzne typy dátových štruktúr:

![solvery](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Cvičenie - rozdelenie údajov

Môžeme sa zamerať na logistickú regresiu pre náš prvý tréningový pokus, keďže ste sa o nej nedávno učili v predchádzajúcej lekcii.
Rozdeľte svoje údaje na trénovacie a testovacie skupiny pomocou `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Cvičenie - aplikácia logistickej regresie

Keďže používate prípad multiclass, musíte si vybrať, akú _schému_ použiť a aký _solver_ nastaviť. Použite LogisticRegression s nastavením multiclass a solverom **liblinear** na trénovanie.

1. Vytvorte logistickú regresiu s multi_class nastavenou na `ovr` a solverom nastaveným na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Vyskúšajte iný solver, napríklad `lbfgs`, ktorý je často nastavený ako predvolený.
> Poznámka: Použite funkciu Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) na zjednodušenie vašich údajov, keď je to potrebné.
Presnosť je dobrá na viac ako **80%**!

1. Tento model si môžete vyskúšať na jednom riadku dát (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Výsledok sa zobrazí:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Skúste iné číslo riadku a skontrolujte výsledky.

1. Ak chcete ísť hlbšie, môžete skontrolovať presnosť tejto predikcie:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Výsledok sa zobrazí - indická kuchyňa je najpravdepodobnejší odhad s dobrou pravdepodobnosťou:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Dokážete vysvetliť, prečo si model myslí, že ide o indickú kuchyňu?

1. Získajte viac detailov vytlačením klasifikačnej správy, ako ste to robili v lekciách o regresii:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | presnosť | recall | f1-skóre | podpora |
    | ------------ | -------- | ------ | -------- | ------- |
    | chinese      | 0.73     | 0.71   | 0.72     | 229     |
    | indian       | 0.91     | 0.93   | 0.92     | 254     |
    | japanese     | 0.70     | 0.75   | 0.72     | 220     |
    | korean       | 0.86     | 0.76   | 0.81     | 242     |
    | thai         | 0.79     | 0.85   | 0.82     | 254     |
    | presnosť     | 0.80     | 1199   |          |         |
    | priemer makro| 0.80     | 0.80   | 0.80     | 1199    |
    | vážený priemer| 0.80     | 0.80   | 0.80     | 1199    |

## 🚀Výzva

V tejto lekcii ste použili vyčistené dáta na vytvorenie modelu strojového učenia, ktorý dokáže predpovedať národnú kuchyňu na základe série ingrediencií. Venujte čas preskúmaniu mnohých možností, ktoré Scikit-learn ponúka na klasifikáciu dát. Ponorte sa hlbšie do konceptu 'solver', aby ste pochopili, čo sa deje v zákulisí.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad & Samoštúdium

Preskúmajte matematiku za logistickou regresiou v [tejto lekcii](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Zadanie 

[Preskúmajte solvery](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.