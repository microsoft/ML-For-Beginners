<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T16:16:51+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "sw"
}
-->
# Wainishaji wa vyakula 1

Katika somo hili, utatumia seti ya data uliyohifadhi kutoka somo la mwisho iliyojaa data safi na yenye usawa kuhusu vyakula.

Utatumia seti hii ya data na aina mbalimbali za wainishaji ili _kutabiri aina ya chakula cha kitaifa kulingana na kikundi cha viungo_. Wakati wa kufanya hivyo, utajifunza zaidi kuhusu baadhi ya njia ambazo algorithimu zinaweza kutumika kwa kazi za uainishaji.

## [Jaribio la kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)
# Maandalizi

Ukikamilisha [Somo la 1](../1-Introduction/README.md), hakikisha kuwa faili _cleaned_cuisines.csv_ ipo katika folda ya mizizi `/data` kwa masomo haya manne.

## Zoezi - tabiri aina ya chakula cha kitaifa

1. Ukifanya kazi katika folda ya _notebook.ipynb_ ya somo hili, leta faili hilo pamoja na maktaba ya Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Data inaonekana kama hii:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Sasa, leta maktaba zaidi:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Gawanya viwianishi vya X na y katika fremu mbili za data kwa mafunzo. `cuisine` inaweza kuwa fremu ya lebo:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Itaonekana kama hii:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Ondoa safu ya `Unnamed: 0` na safu ya `cuisine`, ukitumia `drop()`. Hifadhi data iliyobaki kama vipengele vya mafunzo:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Vipengele vyako vinaonekana kama hii:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Sasa uko tayari kufundisha modeli yako!

## Kuchagua wainishaji wako

Sasa data yako ni safi na tayari kwa mafunzo, unapaswa kuamua ni algorithimu gani ya kutumia kwa kazi hiyo.

Scikit-learn inaweka uainishaji chini ya Kujifunza kwa Usimamizi, na katika kategoria hiyo utapata njia nyingi za kuainisha. [Aina mbalimbali](https://scikit-learn.org/stable/supervised_learning.html) zinaweza kuwa za kushangaza mwanzoni. Njia zifuatazo zote zinajumuisha mbinu za uainishaji:

- Miundo ya Linear
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Mbinu za Ensemble (Voting Classifier)
- Algorithimu za Multiclass na multioutput (uainishaji wa multilabel, uainishaji wa multiclass-multioutput)

> Unaweza pia kutumia [mitandao ya neva kuainisha data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), lakini hiyo iko nje ya wigo wa somo hili.

### Ni wainishaji gani wa kuchagua?

Kwa hivyo, ni wainishaji gani unapaswa kuchagua? Mara nyingi, kujaribu kadhaa na kutafuta matokeo mazuri ni njia ya kupima. Scikit-learn inatoa [linganisho la kando kwa kando](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) kwenye seti ya data iliyoundwa, ikilinganishwa KNeighbors, SVC kwa njia mbili, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB na QuadraticDiscrinationAnalysis, ikionyesha matokeo kwa njia ya picha:

![linganisho la wainishaji](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Mchoro uliotolewa kwenye nyaraka za Scikit-learn

> AutoML inatatua tatizo hili kwa urahisi kwa kuendesha kulinganisha hizi kwenye wingu, ikikuruhusu kuchagua algorithimu bora kwa data yako. Jaribu [hapa](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Njia bora zaidi

Njia bora zaidi kuliko kubahatisha bila mpangilio, hata hivyo, ni kufuata mawazo kwenye [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) inayoweza kupakuliwa. Hapa, tunagundua kuwa, kwa tatizo letu la multiclass, tuna chaguo kadhaa:

![cheatsheet kwa matatizo ya multiclass](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Sehemu ya Algorithm Cheat Sheet ya Microsoft, ikielezea chaguo za uainishaji wa multiclass

✅ Pakua cheatsheet hii, ichapishe, na uitundike ukutani kwako!

### Mantiki

Hebu tuone kama tunaweza kufikiria njia tofauti kulingana na vikwazo tulivyo navyo:

- **Mitandao ya neva ni nzito sana**. Kwa kuzingatia seti yetu ya data safi lakini ndogo, na ukweli kwamba tunafanya mafunzo kwa ndani kupitia notebooks, mitandao ya neva ni nzito sana kwa kazi hii.
- **Hakuna wainishaji wa darasa mbili**. Hatutumii wainishaji wa darasa mbili, kwa hivyo hiyo inatupilia mbali one-vs-all.
- **Decision tree au logistic regression inaweza kufanya kazi**. Decision tree inaweza kufanya kazi, au logistic regression kwa data ya multiclass.
- **Multiclass Boosted Decision Trees hutatua tatizo tofauti**. Multiclass boosted decision tree inafaa zaidi kwa kazi zisizo za parametric, mfano kazi zilizoundwa kujenga viwango, kwa hivyo haitufai.

### Kutumia Scikit-learn 

Tutatumia Scikit-learn kuchambua data yetu. Hata hivyo, kuna njia nyingi za kutumia logistic regression katika Scikit-learn. Angalia [vigezo vya kupitisha](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Kimsingi kuna vigezo viwili muhimu - `multi_class` na `solver` - ambavyo tunahitaji kubainisha, tunapoiomba Scikit-learn kufanya logistic regression. Thamani ya `multi_class` inatumia tabia fulani. Thamani ya solver ni algorithimu ya kutumia. Si solvers zote zinaweza kuunganishwa na thamani zote za `multi_class`.

Kulingana na nyaraka, katika kesi ya multiclass, algorithimu ya mafunzo:

- **Inatumia mpango wa one-vs-rest (OvR)**, ikiwa chaguo la `multi_class` limewekwa kuwa `ovr`
- **Inatumia hasara ya cross-entropy**, ikiwa chaguo la `multi_class` limewekwa kuwa `multinomial`. (Kwa sasa chaguo la `multinomial` linasaidiwa tu na solvers ‘lbfgs’, ‘sag’, ‘saga’ na ‘newton-cg’.)"

> 🎓 'Mpango' hapa unaweza kuwa 'ovr' (one-vs-rest) au 'multinomial'. Kwa kuwa logistic regression imeundwa hasa kusaidia uainishaji wa binary, mipango hii inaiwezesha kushughulikia kazi za uainishaji wa multiclass vizuri zaidi. [chanzo](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' inafafanuliwa kama "algorithimu ya kutumia katika tatizo la uboreshaji". [chanzo](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn inatoa jedwali hili kuelezea jinsi solvers zinavyoshughulikia changamoto tofauti zinazotolewa na miundo tofauti ya data:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Zoezi - gawanya data

Tunaweza kuzingatia logistic regression kwa jaribio letu la kwanza la mafunzo kwa kuwa ulijifunza hivi karibuni kuhusu hili katika somo la awali.
Gawanya data yako katika vikundi vya mafunzo na majaribio kwa kutumia `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Zoezi - tumia logistic regression

Kwa kuwa unatumia kesi ya multiclass, unahitaji kuchagua ni _mpango_ gani wa kutumia na ni _solver_ gani wa kuweka. Tumia LogisticRegression na mpangilio wa multiclass na solver **liblinear** kwa mafunzo.

1. Unda logistic regression na multi_class iliyowekwa kuwa `ovr` na solver iliyowekwa kuwa `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Jaribu solver tofauti kama `lbfgs`, ambayo mara nyingi huwekwa kama chaguo-msingi
> Kumbuka, tumia Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) kazi ili kulainisha data yako inapohitajika.
Usahihi ni mzuri kwa zaidi ya **80%!**

1. Unaweza kuona mfano huu ukifanya kazi kwa kujaribu safu moja ya data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Matokeo yanachapishwa:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Jaribu namba tofauti ya safu na angalia matokeo

1. Kuchunguza zaidi, unaweza kuangalia usahihi wa utabiri huu:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Matokeo yanachapishwa - vyakula vya Kihindi ni dhana bora zaidi, kwa uwezekano mzuri:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Je, unaweza kueleza kwa nini mfano una uhakika kuwa hiki ni chakula cha Kihindi?

1. Pata maelezo zaidi kwa kuchapisha ripoti ya uainishaji, kama ulivyofanya katika masomo ya regression:

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

## 🚀Changamoto

Katika somo hili, ulitumia data yako iliyosafishwa kujenga mfano wa kujifunza mashine ambao unaweza kutabiri aina ya chakula cha kitaifa kulingana na mfululizo wa viungo. Chukua muda kusoma chaguo nyingi ambazo Scikit-learn inatoa ili kuainisha data. Chunguza zaidi dhana ya 'solver' ili kuelewa kinachotokea nyuma ya pazia.

## [Maswali baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujisomea

Chunguza zaidi hesabu nyuma ya logistic regression katika [somo hili](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Kazi 

[Chunguza solvers](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.