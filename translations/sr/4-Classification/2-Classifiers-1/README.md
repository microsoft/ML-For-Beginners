<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T13:03:54+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "sr"
}
-->
# Класификатори кухиња 1

У овом часу, користићете скуп података који сте сачували из претходног часа, пун уравнотежених и чистих података о кухињама.

Користићете овај скуп података са различитим класификаторима да _предвидите одређену националну кухињу на основу групе састојака_. Док то радите, научићете више о неким начинима на које алгоритми могу бити коришћени за задатке класификације.

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)
# Припрема

Под претпоставком да сте завршили [Час 1](../1-Introduction/README.md), уверите се да датотека _cleaned_cuisines.csv_ постоји у коренском `/data` фолдеру за ова четири часа.

## Вежба - предвидите националну кухињу

1. Радите у фолдеру _notebook.ipynb_ овог часа, увезите ту датотеку заједно са библиотеком Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Подаци изгледају овако:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Сада увезите још неколико библиотека:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Поделите X и y координате у два датафрејма за тренинг. `cuisine` може бити датафрејм са ознакама:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Изгледаће овако:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Избаците колону `Unnamed: 0` и колону `cuisine`, користећи `drop()`. Сачувајте остатак података као карактеристике за тренинг:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ваше карактеристике изгледају овако:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Сада сте спремни да обучите свој модел!

## Избор класификатора

Сада када су ваши подаци чисти и спремни за тренинг, морате одлучити који алгоритам да користите за задатак.

Scikit-learn групише класификацију под Надгледано учење, и у тој категорији ћете пронаћи много начина за класификацију. [Разноврсност](https://scikit-learn.org/stable/supervised_learning.html) може изгледати збуњујуће на први поглед. Следеће методе укључују технике класификације:

- Линеарни модели
- Машине за подршку векторима
- Стохастички градијентни спуст
- Најближи суседи
- Гаусови процеси
- Дрвеће одлука
- Методе ансамбла (гласајући класификатор)
- Алгоритми за више класа и више излаза (класификација више класа и више ознака, класификација више класа-више излаза)

> Такође можете користити [неуронске мреже за класификацију података](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), али то је ван оквира овог часа.

### Који класификатор одабрати?

Па, који класификатор треба да изаберете? Често је тестирање неколико њих и тражење доброг резултата начин да се испроба. Scikit-learn нуди [упоредни приказ](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на креираном скупу података, упоређујући KNeighbors, SVC на два начина, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB и QuadraticDiscrinationAnalysis, приказујући резултате визуализоване:

![упоредни приказ класификатора](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Графикони генерисани у документацији Scikit-learn-а

> AutoML решава овај проблем елегантно тако што врши ова поређења у облаку, омогућавајући вам да изаберете најбољи алгоритам за ваше податке. Пробајте [овде](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Бољи приступ

Бољи начин од насумичног погађања је да следите идеје из овог преузимљивог [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Овде откривамо да, за наш проблем са више класа, имамо неке опције:

![чит листа за проблеме са више класа](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Део Microsoft-ове чит листе алгоритама, који детаљно описује опције класификације више класа

✅ Преузмите ову чит листу, одштампајте је и окачите на зид!

### Размишљање

Хајде да видимо да ли можемо да размишљамо о различитим приступима с обзиром на ограничења која имамо:

- **Неуронске мреже су превише захтевне**. С обзиром на наш чист, али минималан скуп података, и чињеницу да тренинг изводимо локално преко нотебука, неуронске мреже су превише захтевне за овај задатак.
- **Нема класификатора за две класе**. Не користимо класификатор за две класе, тако да то искључује one-vs-all.
- **Дрво одлука или логистичка регресија би могли да раде**. Дрво одлука би могло да ради, или логистичка регресија за податке са више класа.
- **Побољшана дрва одлука за више класа решавају другачији проблем**. Побољшано дрво одлука за више класа је најпогодније за непараметарске задатке, нпр. задатке дизајниране за креирање рангирања, тако да нам није корисно.

### Коришћење Scikit-learn-а

Користићемо Scikit-learn за анализу наших података. Међутим, постоји много начина за коришћење логистичке регресије у Scikit-learn-у. Погледајте [параметре за прослеђивање](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

У суштини, постоје два важна параметра - `multi_class` и `solver` - које треба да наведемо када тражимо од Scikit-learn-а да изврши логистичку регресију. Вредност `multi_class` примењује одређено понашање. Вредност solver-а је алгоритам који се користи. Не могу се сви solver-и упарити са свим вредностима `multi_class`.

Према документацији, у случају више класа, алгоритам за тренинг:

- **Користи шему one-vs-rest (OvR)**, ако је опција `multi_class` постављена на `ovr`
- **Користи губитак унакрсне ентропије**, ако је опција `multi_class` постављена на `multinomial`. (Тренутно је опција `multinomial` подржана само од стране solver-а ‘lbfgs’, ‘sag’, ‘saga’ и ‘newton-cg’.)"

> 🎓 'Шема' овде може бити 'ovr' (one-vs-rest) или 'multinomial'. Пошто је логистичка регресија заправо дизајнирана да подржи бинарну класификацију, ове шеме јој омогућавају да боље обради задатке класификације више класа. [извор](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'Solver' је дефинисан као "алгоритам који се користи у проблему оптимизације". [извор](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn нуди ову табелу да објасни како solver-и решавају различите изазове које представљају различите врсте структура података:

![solver-и](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Вежба - поделите податке

Можемо се фокусирати на логистичку регресију за наш први тренинг, пошто сте недавно научили о њој у претходном часу.
Поделите своје податке у групе за тренинг и тестирање позивањем `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Вежба - примените логистичку регресију

Пошто користите случај више класа, потребно је да изаберете коју _шему_ да користите и који _solver_ да поставите. Користите LogisticRegression са подешавањем више класа и **liblinear** solver-ом за тренинг.

1. Креирајте логистичку регресију са multi_class постављеним на `ovr` и solver-ом постављеним на `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Пробајте другачији solver као што је `lbfgs`, који је често постављен као подразумеван
> Напомена, користите Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) функцију за изравнавање ваших података када је то потребно.
Тачност је добра, преко **80%**!

1. Можете видети овај модел у акцији тестирањем једног реда података (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Резултат се исписује:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Пробајте други број реда и проверите резултате.

1. Ако желите да истражите дубље, можете проверити тачност ове предикције:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Резултат се исписује - индијска кухиња је најбоља претпоставка, са добром вероватноћом:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Можете ли објаснити зашто је модел прилично сигуран да је ово индијска кухиња?

1. Добијте више детаља исписивањем извештаја о класификацији, као што сте радили у лекцијама о регресији:

    ```python
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    ```

    |              | прецизност | одзив | f1-резултат | подршка |
    | ------------ | ---------- | ----- | ----------- | ------- |
    | chinese      | 0.73       | 0.71  | 0.72        | 229     |
    | indian       | 0.91       | 0.93  | 0.92        | 254     |
    | japanese     | 0.70       | 0.75  | 0.72        | 220     |
    | korean       | 0.86       | 0.76  | 0.81        | 242     |
    | thai         | 0.79       | 0.85  | 0.82        | 254     |
    | тачност      | 0.80       | 1199  |             |         |
    | макро просек | 0.80       | 0.80  | 0.80        | 1199    |
    | пондерисан просек | 0.80   | 0.80  | 0.80        | 1199    |

## 🚀Изазов

У овој лекцији, користили сте очишћене податке за изградњу модела машинског учења који може предвидети националну кухињу на основу серије састојака. Одвојите време да прочитате многе опције које Scikit-learn пружа за класификацију података. Истражите дубље концепт 'solver'-а да бисте разумели шта се дешава иза сцене.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Истражите мало више математику иза логистичке регресије у [овој лекцији](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Задатак 

[Проучите решаваче](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако настојимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на изворном језику треба сматрати меродавним извором. За критичне информације препоручује се професионални превод од стране људи. Не сносимо одговорност за било каква погрешна тумачења или неспоразуме који могу произаћи из коришћења овог превода.