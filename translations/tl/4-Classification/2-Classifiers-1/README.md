<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T18:20:32+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "tl"
}
-->
# Mga Classifier ng Lutuin 1

Sa araling ito, gagamitin mo ang dataset na na-save mo mula sa nakaraang aralin na puno ng balanseng, malinis na datos tungkol sa mga lutuin.

Gagamitin mo ang dataset na ito gamit ang iba't ibang classifier upang _hulaan ang isang pambansang lutuin batay sa isang grupo ng mga sangkap_. Habang ginagawa ito, matututo ka pa tungkol sa ilang paraan kung paano magagamit ang mga algorithm para sa mga gawain ng klasipikasyon.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Paghahanda

Kung natapos mo na ang [Aralin 1](../1-Introduction/README.md), tiyakin na mayroong _cleaned_cuisines.csv_ file sa root `/data` folder para sa apat na araling ito.

## Ehersisyo - hulaan ang pambansang lutuin

1. Sa folder ng _notebook.ipynb_ ng araling ito, i-import ang file na iyon kasama ang Pandas library:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Ganito ang hitsura ng data:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Ngayon, i-import ang ilang karagdagang library:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Hatiin ang X at y coordinates sa dalawang dataframe para sa training. Ang `cuisine` ay maaaring maging labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ganito ang magiging hitsura nito:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. I-drop ang `Unnamed: 0` column at ang `cuisine` column gamit ang `drop()`. I-save ang natitirang data bilang mga trainable features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ganito ang hitsura ng iyong mga features:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Ngayon handa ka nang i-train ang iyong modelo!

## Pagpili ng classifier

Ngayon na malinis na ang iyong data at handa na para sa training, kailangan mong magdesisyon kung anong algorithm ang gagamitin para sa trabaho.

Ang Scikit-learn ay nagkakategorya ng klasipikasyon sa ilalim ng Supervised Learning, at sa kategoryang iyon makakakita ka ng maraming paraan upang mag-classify. [Ang iba't ibang pamamaraan](https://scikit-learn.org/stable/supervised_learning.html) ay maaaring nakakalito sa unang tingin. Ang mga sumusunod na pamamaraan ay lahat may kasamang mga teknik sa klasipikasyon:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass at multioutput algorithms (multiclass at multilabel classification, multiclass-multioutput classification)

> Maaari ka ring gumamit ng [neural networks upang mag-classify ng data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ngunit ito ay labas sa saklaw ng araling ito.

### Anong classifier ang gagamitin?

Kaya, aling classifier ang dapat mong piliin? Madalas, ang pagsubok sa ilang classifier at paghahanap ng magandang resulta ay isang paraan upang mag-test. Ang Scikit-learn ay nag-aalok ng [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sa isang nilikhang dataset, na ikinukumpara ang KNeighbors, SVC dalawang paraan, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB at QuadraticDiscrinationAnalysis, na ipinapakita ang mga resulta sa visual na paraan:

![comparison of classifiers](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Mga plot na nilikha mula sa dokumentasyon ng Scikit-learn

> Ang AutoML ay maayos na nilulutas ang problemang ito sa pamamagitan ng pagtakbo ng mga paghahambing sa cloud, na nagbibigay-daan sa iyo upang piliin ang pinakamahusay na algorithm para sa iyong data. Subukan ito [dito](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Isang mas mahusay na paraan

Isang mas mahusay na paraan kaysa sa random na paghula ay ang sundin ang mga ideya sa downloadable na [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Dito, matutuklasan natin na, para sa ating multiclass na problema, mayroon tayong ilang mga pagpipilian:

![cheatsheet for multiclass problems](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Isang bahagi ng Algorithm Cheat Sheet ng Microsoft, na nagdedetalye ng mga opsyon para sa multiclass classification

✅ I-download ang cheat sheet na ito, i-print ito, at isabit sa iyong dingding!

### Pangangatwiran

Tingnan natin kung kaya nating mag-isip ng iba't ibang paraan batay sa mga limitasyon na mayroon tayo:

- **Masyadong mabigat ang neural networks**. Dahil sa malinis ngunit minimal na dataset, at ang katotohanan na tayo ay nagta-training nang lokal gamit ang notebooks, masyadong mabigat ang neural networks para sa gawaing ito.
- **Walang two-class classifier**. Hindi tayo gagamit ng two-class classifier, kaya hindi natin gagamitin ang one-vs-all.
- **Maaaring gumana ang decision tree o logistic regression**. Maaaring gumana ang decision tree, o logistic regression para sa multiclass na data.
- **Ang Multiclass Boosted Decision Trees ay para sa ibang problema**. Ang multiclass boosted decision tree ay pinakaangkop para sa mga nonparametric na gawain, halimbawa mga gawain na idinisenyo upang bumuo ng rankings, kaya hindi ito kapaki-pakinabang para sa atin.

### Paggamit ng Scikit-learn 

Gagamitin natin ang Scikit-learn upang suriin ang ating data. Gayunpaman, maraming paraan upang gamitin ang logistic regression sa Scikit-learn. Tingnan ang [mga parameter na maaaring ipasa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

May dalawang mahalagang parameter - `multi_class` at `solver` - na kailangan nating tukuyin kapag hinihiling natin sa Scikit-learn na magsagawa ng logistic regression. Ang `multi_class` value ay nag-aaplay ng tiyak na behavior. Ang value ng solver ay ang algorithm na gagamitin. Hindi lahat ng solver ay maaaring ipares sa lahat ng `multi_class` values.

Ayon sa dokumentasyon, sa multiclass na kaso, ang training algorithm:

- **Gumagamit ng one-vs-rest (OvR) scheme**, kung ang `multi_class` option ay nakatakda sa `ovr`
- **Gumagamit ng cross-entropy loss**, kung ang `multi_class` option ay nakatakda sa `multinomial`. (Sa kasalukuyan ang `multinomial` option ay sinusuportahan lamang ng ‘lbfgs’, ‘sag’, ‘saga’ at ‘newton-cg’ solvers.)"

> 🎓 Ang 'scheme' dito ay maaaring 'ovr' (one-vs-rest) o 'multinomial'. Dahil ang logistic regression ay talagang idinisenyo upang suportahan ang binary classification, ang mga scheme na ito ay nagbibigay-daan dito upang mas mahusay na hawakan ang multiclass classification tasks. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Ang 'solver' ay tinukoy bilang "ang algorithm na gagamitin sa optimization problem". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Nag-aalok ang Scikit-learn ng talahanayan na ito upang ipaliwanag kung paano hinahawakan ng mga solver ang iba't ibang hamon na ipinapakita ng iba't ibang uri ng data structures:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Ehersisyo - hatiin ang data

Maaari tayong mag-focus sa logistic regression para sa ating unang training trial dahil kamakailan mo itong natutunan sa nakaraang aralin.
Hatiin ang iyong data sa training at testing groups sa pamamagitan ng pagtawag sa `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Ehersisyo - gamitin ang logistic regression

Dahil gumagamit ka ng multiclass case, kailangan mong pumili kung anong _scheme_ ang gagamitin at kung anong _solver_ ang itatakda. Gumamit ng LogisticRegression na may multiclass setting at ang **liblinear** solver para sa training.

1. Gumawa ng logistic regression na may multi_class na nakatakda sa `ovr` at ang solver na nakatakda sa `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Subukan ang ibang solver tulad ng `lbfgs`, na madalas na nakatakda bilang default
> Tandaan, gamitin ang Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) na function upang i-flatten ang iyong data kapag kinakailangan.
Ang katumpakan ay maganda sa higit **80%**!

1. Makikita mo ang modelong ito sa aksyon sa pamamagitan ng pagsubok sa isang hilera ng data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Ang resulta ay naka-print:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Subukan ang ibang numero ng hilera at suriin ang mga resulta

1. Masusing pagsusuri, maaari mong suriin ang katumpakan ng prediksyon na ito:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Ang resulta ay naka-print - Indian cuisine ang pinakamagandang hula nito, na may magandang posibilidad:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Maaari mo bang ipaliwanag kung bakit sigurado ang modelo na ito ay Indian cuisine?

1. Makakuha ng mas detalyado sa pamamagitan ng pag-print ng isang classification report, tulad ng ginawa mo sa regression lessons:

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

## 🚀Hamunin

Sa araling ito, ginamit mo ang nalinis na data upang bumuo ng isang machine learning model na maaaring magpredikta ng pambansang cuisine batay sa isang serye ng mga sangkap. Maglaan ng oras upang basahin ang maraming opsyon na inaalok ng Scikit-learn para sa pag-classify ng data. Masusing pag-aralan ang konsepto ng 'solver' upang maunawaan ang nangyayari sa likod ng eksena.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Pag-aaral sa Sarili

Masusing pag-aralan ang matematika sa likod ng logistic regression sa [araling ito](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Takdang Aralin 

[Pag-aralan ang mga solvers](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.