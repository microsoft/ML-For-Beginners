<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T00:42:34+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "cs"
}
-->
# Klasifikátory kuchyní 1

V této lekci použijete dataset, který jste si uložili z minulé lekce, plný vyvážených a čistých dat o kuchyních.

Tento dataset použijete s různými klasifikátory k _predikci národní kuchyně na základě skupiny ingrediencí_. Přitom se dozvíte více o některých způsobech, jak lze algoritmy využít pro klasifikační úlohy.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)
# Příprava

Za předpokladu, že jste dokončili [Lekci 1](../1-Introduction/README.md), ujistěte se, že soubor _cleaned_cuisines.csv_ existuje v kořenové složce `/data` pro tyto čtyři lekce.

## Cvičení - predikce národní kuchyně

1. Pracujte ve složce _notebook.ipynb_ této lekce, importujte tento soubor spolu s knihovnou Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data vypadají takto:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Nyní importujte několik dalších knihoven:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Rozdělte souřadnice X a y do dvou datových rámců pro trénování. `cuisine` může být datový rámec s označeními:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Bude vypadat takto:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Odstraňte sloupec `Unnamed: 0` a sloupec `cuisine` pomocí `drop()`. Zbytek dat uložte jako trénovací vlastnosti:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vaše vlastnosti vypadají takto:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Nyní jste připraveni trénovat svůj model!

## Výběr klasifikátoru

Nyní, když jsou vaše data čistá a připravená k trénování, musíte se rozhodnout, jaký algoritmus použít.

Scikit-learn zařazuje klasifikaci pod Supervised Learning, a v této kategorii najdete mnoho způsobů klasifikace. [Rozmanitost](https://scikit-learn.org/stable/supervised_learning.html) je na první pohled docela ohromující. Následující metody zahrnují techniky klasifikace:

- Lineární modely
- Support Vector Machines
- Stochastic Gradient Descent
- Nejbližší sousedé
- Gaussovské procesy
- Rozhodovací stromy
- Ensemble metody (hlasovací klasifikátor)
- Multiclass a multioutput algoritmy (multiclass a multilabel klasifikace, multiclass-multioutput klasifikace)

> Můžete také použít [neuronové sítě k klasifikaci dat](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ale to je mimo rozsah této lekce.

### Jaký klasifikátor zvolit?

Takže, jaký klasifikátor byste měli zvolit? Často je dobré vyzkoušet několik a hledat dobrý výsledek. Scikit-learn nabízí [srovnání vedle sebe](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) na vytvořeném datasetu, kde porovnává KNeighbors, SVC dvěma způsoby, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB a QuadraticDiscriminationAnalysis, a vizualizuje výsledky:

![srovnání klasifikátorů](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Grafy generované na dokumentaci Scikit-learn

> AutoML tento problém elegantně řeší tím, že provádí tato srovnání v cloudu, což vám umožňuje vybrat nejlepší algoritmus pro vaše data. Vyzkoušejte to [zde](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Lepší přístup

Lepší způsob než náhodné hádání je však řídit se nápady z této stahovatelné [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Zde zjistíme, že pro náš problém s více třídami máme několik možností:

![cheatsheet pro problémy s více třídami](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Část Microsoft Algorithm Cheat Sheet, popisující možnosti klasifikace s více třídami

✅ Stáhněte si tento cheat sheet, vytiskněte ho a pověste na zeď!

### Úvahy

Podívejme se, jestli dokážeme logicky projít různé přístupy vzhledem k omezením, která máme:

- **Neuronové sítě jsou příliš náročné**. Vzhledem k našemu čistému, ale minimálnímu datasetu a skutečnosti, že trénování provádíme lokálně pomocí notebooků, jsou neuronové sítě pro tuto úlohu příliš náročné.
- **Žádný klasifikátor pro dvě třídy**. Nepoužíváme klasifikátor pro dvě třídy, takže to vylučuje one-vs-all.
- **Rozhodovací strom nebo logistická regrese by mohly fungovat**. Rozhodovací strom by mohl fungovat, nebo logistická regrese pro data s více třídami.
- **Multiclass Boosted Decision Trees řeší jiný problém**. Multiclass Boosted Decision Tree je nejvhodnější pro neparametrické úlohy, např. úlohy určené k vytváření pořadí, takže pro nás není užitečný.

### Použití Scikit-learn 

Budeme používat Scikit-learn k analýze našich dat. Existuje však mnoho způsobů, jak použít logistickou regresi v Scikit-learn. Podívejte se na [parametry k předání](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

V podstatě existují dva důležité parametry - `multi_class` a `solver` - které musíme specifikovat, když požádáme Scikit-learn o provedení logistické regrese. Hodnota `multi_class` aplikuje určité chování. Hodnota solveru určuje, jaký algoritmus použít. Ne všechny solvery lze kombinovat se všemi hodnotami `multi_class`.

Podle dokumentace v případě klasifikace s více třídami trénovací algoritmus:

- **Používá schéma one-vs-rest (OvR)**, pokud je možnost `multi_class` nastavena na `ovr`
- **Používá ztrátu křížové entropie**, pokud je možnost `multi_class` nastavena na `multinomial`. (V současné době je možnost `multinomial` podporována pouze solvery ‘lbfgs’, ‘sag’, ‘saga’ a ‘newton-cg’.)

> 🎓 'Schéma' zde může být buď 'ovr' (one-vs-rest) nebo 'multinomial'. Vzhledem k tomu, že logistická regrese je skutečně navržena pro podporu binární klasifikace, tato schémata jí umožňují lépe zvládat úlohy klasifikace s více třídami. [zdroj](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' je definován jako "algoritmus použitý v optimalizačním problému". [zdroj](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn nabízí tuto tabulku, která vysvětluje, jak solvery zvládají různé výzvy, které představují různé typy datových struktur:

![solvery](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Cvičení - rozdělení dat

Můžeme se zaměřit na logistickou regresi pro náš první pokus o trénování, protože jste se o ní nedávno učili v předchozí lekci.
Rozdělte svá data na trénovací a testovací skupiny pomocí `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Cvičení - aplikace logistické regrese

Vzhledem k tomu, že používáte případ s více třídami, musíte si vybrat, jaké _schéma_ použít a jaký _solver_ nastavit. Použijte LogisticRegression s nastavením multi_class na `ovr` a solverem na **liblinear** pro trénování.

1. Vytvořte logistickou regresi s multi_class nastaveným na `ovr` a solverem nastaveným na `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Vyzkoušejte jiný solver, například `lbfgs`, který je často nastaven jako výchozí
> Poznámka: Použijte funkci Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) k vyrovnání vašich dat, když je to potřeba.
Přesnost je dobrá na více než **80 %**!

1. Můžete vidět tento model v akci testováním jednoho řádku dat (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Výsledek je vytištěn:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Vyzkoušejte jiný číslo řádku a zkontrolujte výsledky.

1. Pokud se ponoříte hlouběji, můžete ověřit přesnost této predikce:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Výsledek je vytištěn - indická kuchyně je nejlepší odhad s dobrou pravděpodobností:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Dokážete vysvětlit, proč si model je docela jistý, že se jedná o indickou kuchyni?

1. Získejte více detailů vytištěním klasifikační zprávy, stejně jako jste to udělali v lekcích o regresi:

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

## 🚀Výzva

V této lekci jste použili svá vyčištěná data k vytvoření modelu strojového učení, který dokáže předpovědět národní kuchyni na základě série ingrediencí. Věnujte nějaký čas prozkoumání mnoha možností, které Scikit-learn nabízí pro klasifikaci dat. Ponořte se hlouběji do konceptu 'solver', abyste pochopili, co se děje v zákulisí.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Prozkoumejte trochu více matematiku za logistickou regresí v [této lekci](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Úkol 

[Prostudujte solvery](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.