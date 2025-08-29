<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9579f42e3ff5114c58379cc9e186a828",
  "translation_date": "2025-08-29T13:54:12+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "tl"
}
-->
# Mga Classifier ng Lutuin 1

Sa araling ito, gagamitin mo ang dataset na na-save mo mula sa nakaraang aralin na puno ng balanseng, malinis na datos tungkol sa mga lutuin.

Gagamitin mo ang dataset na ito gamit ang iba't ibang classifier upang _hulaan ang isang partikular na pambansang lutuin batay sa isang grupo ng mga sangkap_. Habang ginagawa ito, matututo ka pa tungkol sa iba't ibang paraan kung paano magagamit ang mga algorithm para sa mga gawain ng klasipikasyon.

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/21/)
# Paghahanda

Kung natapos mo na ang [Aralin 1](../1-Introduction/README.md), siguraduhing mayroong _cleaned_cuisines.csv_ na file sa root `/data` folder para sa apat na araling ito.

## Ehersisyo - hulaan ang isang pambansang lutuin

1. Sa folder na _notebook.ipynb_ ng araling ito, i-import ang file na iyon kasama ang Pandas library:

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
  

1. Ngayon, mag-import ng ilang karagdagang library:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Hatiin ang X at y coordinates sa dalawang dataframe para sa pagsasanay. Ang `cuisine` ay maaaring gawing labels dataframe:

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

1. I-drop ang `Unnamed: 0` column at ang `cuisine` column gamit ang `drop()`. I-save ang natitirang data bilang mga trainable feature:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ganito ang hitsura ng iyong mga feature:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Ngayon handa ka nang sanayin ang iyong modelo!

## Pagpili ng iyong classifier

Ngayon na ang iyong data ay malinis at handa na para sa pagsasanay, kailangan mong magdesisyon kung aling algorithm ang gagamitin para sa gawain.

Ang Scikit-learn ay nagkakategorya ng klasipikasyon sa ilalim ng Supervised Learning, at sa kategoryang iyon makakakita ka ng maraming paraan upang mag-classify. [Ang iba't ibang pamamaraan](https://scikit-learn.org/stable/supervised_learning.html) ay maaaring nakakalito sa unang tingin. Ang mga sumusunod na pamamaraan ay lahat may kasamang mga teknik sa klasipikasyon:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass at multioutput algorithms (multiclass at multilabel classification, multiclass-multioutput classification)

> Maaari mo ring gamitin ang [neural networks upang mag-classify ng data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), ngunit ito ay labas sa saklaw ng araling ito.

### Aling classifier ang gagamitin?

Kaya, aling classifier ang dapat mong piliin? Madalas, ang pagsubok sa ilan at paghahanap ng magandang resulta ay isang paraan upang mag-eksperimento. Ang Scikit-learn ay nag-aalok ng isang [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) sa isang nilikhang dataset, na inihahambing ang KNeighbors, SVC sa dalawang paraan, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB at QuadraticDiscriminationAnalysis, na ipinapakita ang mga resulta sa biswal na paraan:

![paghahambing ng mga classifier](../../../../translated_images/comparison.edfab56193a85e7fdecbeaa1b1f8c99e94adbf7178bed0de902090cf93d6734f.tl.png)
> Mga plot na nilikha mula sa dokumentasyon ng Scikit-learn

> Ang AutoML ay mahusay na solusyon sa problemang ito sa pamamagitan ng pagsasagawa ng mga paghahambing na ito sa cloud, na nagbibigay-daan sa iyong pumili ng pinakamahusay na algorithm para sa iyong data. Subukan ito [dito](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Isang mas mahusay na paraan

Isang mas mahusay na paraan kaysa sa basta-basta paghula ay ang sundin ang mga ideya sa downloadable na [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Dito, matutuklasan natin na, para sa ating multiclass na problema, mayroon tayong ilang mga pagpipilian:

![cheatsheet para sa mga multiclass na problema](../../../../translated_images/cheatsheet.07a475ea444d22234cb8907a3826df5bdd1953efec94bd18e4496f36ff60624a.tl.png)
> Isang bahagi ng Algorithm Cheat Sheet ng Microsoft, na nagdedetalye ng mga opsyon para sa multiclass classification

✅ I-download ang cheat sheet na ito, i-print ito, at idikit ito sa iyong dingding!

### Pangangatwiran

Tingnan natin kung kaya nating pangatwiranan ang iba't ibang mga pamamaraan batay sa mga limitasyon na mayroon tayo:

- **Masyadong mabigat ang neural networks**. Dahil sa ating malinis ngunit minimal na dataset, at ang katotohanang isinasagawa natin ang pagsasanay nang lokal gamit ang mga notebook, masyadong mabigat ang neural networks para sa gawaing ito.
- **Hindi angkop ang two-class classifier**. Hindi tayo gagamit ng two-class classifier, kaya hindi natin gagamitin ang one-vs-all.
- **Maaaring gumana ang decision tree o logistic regression**. Maaaring gumana ang decision tree, o logistic regression para sa multiclass na data.
- **Ang Multiclass Boosted Decision Trees ay para sa ibang problema**. Ang multiclass boosted decision tree ay pinakaangkop para sa mga nonparametric na gawain, tulad ng mga gawain na idinisenyo upang bumuo ng mga ranggo, kaya hindi ito kapaki-pakinabang para sa atin.

### Paggamit ng Scikit-learn 

Gagamitin natin ang Scikit-learn upang suriin ang ating data. Gayunpaman, maraming paraan upang gamitin ang logistic regression sa Scikit-learn. Tingnan ang [mga parameter na maaaring ipasa](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Sa esensya, may dalawang mahalagang parameter - `multi_class` at `solver` - na kailangan nating tukuyin kapag hinihiling natin sa Scikit-learn na magsagawa ng logistic regression. Ang halaga ng `multi_class` ay nag-aaplay ng partikular na pag-uugali. Ang halaga ng solver ay tumutukoy kung anong algorithm ang gagamitin. Hindi lahat ng solver ay maaaring ipares sa lahat ng `multi_class` na halaga.

Ayon sa dokumentasyon, sa kaso ng multiclass, ang training algorithm:

- **Gumagamit ng one-vs-rest (OvR) scheme**, kung ang `multi_class` na opsyon ay nakatakda sa `ovr`
- **Gumagamit ng cross-entropy loss**, kung ang `multi_class` na opsyon ay nakatakda sa `multinomial`. (Sa kasalukuyan, ang `multinomial` na opsyon ay sinusuportahan lamang ng ‘lbfgs’, ‘sag’, ‘saga’ at ‘newton-cg’ na mga solver.)

> 🎓 Ang 'scheme' dito ay maaaring 'ovr' (one-vs-rest) o 'multinomial'. Dahil ang logistic regression ay talagang idinisenyo upang suportahan ang binary classification, ang mga scheme na ito ay nagpapahintulot dito na mas mahusay na hawakan ang mga gawain ng multiclass classification. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Ang 'solver' ay tinukoy bilang "ang algorithm na gagamitin sa optimization problem". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Nag-aalok ang Scikit-learn ng talahanayan na ito upang ipaliwanag kung paano hinaharap ng mga solver ang iba't ibang hamon na ipinapakita ng iba't ibang uri ng data structure:

![solvers](../../../../translated_images/solvers.5fc648618529e627dfac29b917b3ccabda4b45ee8ed41b0acb1ce1441e8d1ef1.tl.png)

## Ehersisyo - hatiin ang data

Maaari tayong mag-focus sa logistic regression para sa ating unang pagsubok sa pagsasanay dahil kamakailan mo lang natutunan ang tungkol dito sa isang nakaraang aralin.
Hatiin ang iyong data sa mga grupo ng pagsasanay at pagsubok sa pamamagitan ng pagtawag sa `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Ehersisyo - gamitin ang logistic regression

Dahil gumagamit ka ng multiclass na kaso, kailangan mong pumili kung anong _scheme_ ang gagamitin at kung anong _solver_ ang itatakda. Gumamit ng LogisticRegression na may multiclass setting at ang **liblinear** solver para sa pagsasanay.

1. Gumawa ng logistic regression na may multi_class na nakatakda sa `ovr` at ang solver na nakatakda sa `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Subukan ang ibang solver tulad ng `lbfgs`, na madalas na nakatakda bilang default
> Tandaan, gamitin ang Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) na function upang gawing patag ang iyong data kapag kinakailangan.
Ang katumpakan ay maganda sa higit **80%**!

1. Makikita mo ang modelong ito sa aksyon sa pamamagitan ng pagsubok ng isang hilera ng data (#50):

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

1. Kung nais mong mas maintindihan, maaari mong suriin ang katumpakan ng prediksyon na ito:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Ang resulta ay naka-print - Indian cuisine ang pinakamalapit na hula, na may mataas na posibilidad:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Kaya mo bang ipaliwanag kung bakit sigurado ang modelo na ito ay Indian cuisine?

1. Makakuha ng mas detalyadong impormasyon sa pamamagitan ng pag-print ng classification report, tulad ng ginawa mo sa regression lessons:

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

Sa araling ito, ginamit mo ang nalinis mong data upang bumuo ng isang machine learning model na kayang hulaan ang isang national cuisine base sa mga sangkap. Maglaan ng oras upang basahin ang maraming opsyon na inaalok ng Scikit-learn para sa pag-classify ng data. Mas pag-aralan ang konsepto ng 'solver' upang maintindihan kung ano ang nangyayari sa likod ng proseso.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/22/)

## Review at Pag-aaral sa Sarili

Mas pag-aralan ang matematika sa likod ng logistic regression sa [araling ito](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Takdang Aralin 

[Pag-aralan ang mga solvers](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.