<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T00:41:49+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "bg"
}
-->
# Класификатори за кухни 1

В този урок ще използвате набора от данни, който запазихте от предишния урок, пълен с балансирани и почистени данни за различни кухни.

Ще използвате този набор от данни с разнообразие от класификатори, за да _предвидите дадена национална кухня въз основа на група съставки_. Докато правите това, ще научите повече за някои от начините, по които алгоритмите могат да бъдат използвани за задачи по класификация.

## [Тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/)
# Подготовка

При условие че сте завършили [Урок 1](../1-Introduction/README.md), уверете се, че файлът _cleaned_cuisines.csv_ съществува в основната папка `/data` за тези четири урока.

## Упражнение - предвиждане на национална кухня

1. Работейки в папката _notebook.ipynb_ на този урок, импортирайте този файл заедно с библиотеката Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Данните изглеждат така:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Сега импортирайте още няколко библиотеки:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Разделете X и y координатите в два датафрейма за обучение. `cuisine` може да бъде датафреймът с етикети:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Ще изглежда така:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Премахнете колоната `Unnamed: 0` и колоната `cuisine`, използвайки `drop()`. Запазете останалите данни като обучаеми характеристики:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Вашите характеристики изглеждат така:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Сега сте готови да обучите модела си!

## Избор на класификатор

Сега, когато данните ви са почистени и готови за обучение, трябва да решите кой алгоритъм да използвате за задачата.

Scikit-learn групира класификацията под Надзорно Обучение, и в тази категория ще намерите много начини за класифициране. [Разнообразието](https://scikit-learn.org/stable/supervised_learning.html) може да изглежда объркващо на пръв поглед. Следните методи включват техники за класификация:

- Линейни модели
- Машини за опорни вектори
- Стохастичен градиентен спуск
- Най-близки съседи
- Гаусови процеси
- Дървета за решения
- Методи на ансамбъл (гласуващ класификатор)
- Мултикласови и мултиизходни алгоритми (мултикласова и мултиетикетна класификация, мултикласова-мултиизходна класификация)

> Можете също да използвате [невронни мрежи за класифициране на данни](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), но това е извън обхвата на този урок.

### Какъв класификатор да изберете?

И така, кой класификатор трябва да изберете? Често, преминаването през няколко и търсенето на добър резултат е начин за тестване. Scikit-learn предлага [сравнение рамо до рамо](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на създаден набор от данни, сравнявайки KNeighbors, SVC по два начина, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB и QuadraticDiscriminationAnalysis, показвайки резултатите визуализирани:

![сравнение на класификатори](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Графики, генерирани от документацията на Scikit-learn

> AutoML решава този проблем лесно, като изпълнява тези сравнения в облака, позволявайки ви да изберете най-добрия алгоритъм за вашите данни. Опитайте [тук](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### По-добър подход

По-добър начин от случайното предположение е да следвате идеите от този изтегляем [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Тук откриваме, че за нашия мултикласов проблем имаме някои опции:

![cheatsheet за мултикласови проблеми](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Част от Cheat Sheet на Microsoft за алгоритми, описваща опции за мултикласова класификация

✅ Изтеглете този cheat sheet, разпечатайте го и го закачете на стената си!

### Разсъждения

Нека видим дали можем да разсъждаваме върху различни подходи, като се вземат предвид ограниченията, които имаме:

- **Невронните мрежи са твърде тежки**. Като се има предвид нашият почистен, но минимален набор от данни и фактът, че изпълняваме обучението локално чрез ноутбуци, невронните мрежи са твърде тежки за тази задача.
- **Не използваме двукласов класификатор**. Не използваме двукласов класификатор, така че това изключва one-vs-all.
- **Дърво за решения или логистична регресия може да работи**. Дърво за решения може да работи, или логистична регресия за мултикласови данни.
- **Мултикласовите Boosted Decision Trees решават различен проблем**. Мултикласовото Boosted Decision Tree е най-подходящо за непараметрични задачи, например задачи, предназначени за изграждане на класации, така че не е полезно за нас.

### Използване на Scikit-learn 

Ще използваме Scikit-learn за анализ на нашите данни. Въпреки това, има много начини за използване на логистична регресия в Scikit-learn. Вижте [параметрите за предаване](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

По същество има два важни параметъра - `multi_class` и `solver` - които трябва да зададем, когато поискаме от Scikit-learn да изпълни логистична регресия. Стойността на `multi_class` прилага определено поведение. Стойността на solver определя кой алгоритъм да се използва. Не всички solver могат да се комбинират с всички стойности на `multi_class`.

Според документацията, в случая на мултиклас, алгоритъмът за обучение:

- **Използва схемата one-vs-rest (OvR)**, ако опцията `multi_class` е зададена на `ovr`
- **Използва загубата на кръстосана ентропия**, ако опцията `multi_class` е зададена на `multinomial`. (В момента опцията `multinomial` се поддържа само от solver-ите ‘lbfgs’, ‘sag’, ‘saga’ и ‘newton-cg’.)"

> 🎓 "Схемата" тук може да бъде 'ovr' (one-vs-rest) или 'multinomial'. Тъй като логистичната регресия е наистина предназначена да поддържа бинарна класификация, тези схеми й позволяват да се справя по-добре с задачи за мултикласова класификация. [източник](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "Solver" се определя като "алгоритъмът, който да се използва в проблема за оптимизация". [източник](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn предлага тази таблица, за да обясни как solver-ите се справят с различни предизвикателства, представени от различни видове структури на данни:

![solver-и](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Упражнение - разделяне на данните

Можем да се фокусираме върху логистичната регресия за първия ни опит за обучение, тъй като наскоро научихте за нея в предишен урок.
Разделете данните си на групи за обучение и тестване, като извикате `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Упражнение - прилагане на логистична регресия

Тъй като използвате случая на мултиклас, трябва да изберете каква _схема_ да използвате и какъв _solver_ да зададете. Използвайте LogisticRegression с настройка за мултиклас и solver **liblinear** за обучение.

1. Създайте логистична регресия с multi_class, зададено на `ovr`, и solver, зададен на `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Опитайте различен solver като `lbfgs`, който често е зададен като стандартен
> Забележка: Използвайте функцията Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html), за да изравните данните си, когато е необходимо.
Точността е добра при над **80%**!

1. Можете да видите този модел в действие, като тествате един ред данни (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Резултатът се отпечатва:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Опитайте с различен номер на ред и проверете резултатите.

1. Ако искате да се задълбочите, можете да проверите точността на тази прогноза:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Резултатът се отпечатва - индийската кухня е най-доброто предположение, с добра вероятност:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Можете ли да обясните защо моделът е доста сигурен, че това е индийска кухня?

1. Получете повече подробности, като отпечатате отчет за класификация, както направихте в уроците за регресия:

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

## 🚀Предизвикателство

В този урок използвахте почистените си данни, за да изградите модел за машинно обучение, който може да предскаже национална кухня въз основа на серия от съставки. Отделете време да разгледате многото опции, които Scikit-learn предоставя за класифициране на данни. Задълбочете се в концепцията за 'solver', за да разберете какво се случва зад кулисите.

## [Тест след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостоятелно обучение

Разгледайте малко повече математиката зад логистичната регресия в [този урок](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Задача 

[Проучете решаващите алгоритми](assignment.md)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.