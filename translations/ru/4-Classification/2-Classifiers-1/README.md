<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T08:34:16+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "ru"
}
-->
# Классификаторы кухонь 1

В этом уроке вы будете использовать набор данных, который вы сохранили в предыдущем уроке, содержащий сбалансированные и очищенные данные о кухнях.

Вы будете использовать этот набор данных с различными классификаторами, чтобы _предсказать национальную кухню на основе группы ингредиентов_. В процессе вы узнаете больше о том, как алгоритмы могут быть использованы для задач классификации.

## [Тест перед лекцией](https://ff-quizzes.netlify.app/en/ml/)
# Подготовка

Если вы завершили [Урок 1](../1-Introduction/README.md), убедитесь, что файл _cleaned_cuisines.csv_ находится в корневой папке `/data` для этих четырех уроков.

## Упражнение - предсказание национальной кухни

1. Работая в папке _notebook.ipynb_ этого урока, импортируйте файл вместе с библиотекой Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Данные выглядят следующим образом:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

1. Теперь импортируйте несколько дополнительных библиотек:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Разделите координаты X и y на два датафрейма для обучения. `cuisine` может быть датафреймом с метками:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Это будет выглядеть так:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Удалите столбец `Unnamed: 0` и столбец `cuisine`, используя `drop()`. Сохраните оставшиеся данные как обучающие признаки:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ваши признаки выглядят следующим образом:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Теперь вы готовы обучать вашу модель!

## Выбор классификатора

Теперь, когда ваши данные очищены и готовы к обучению, нужно решить, какой алгоритм использовать для задачи.

Scikit-learn группирует классификацию в категории Обучения с учителем, и в этой категории вы найдете множество способов классификации. [Разнообразие](https://scikit-learn.org/stable/supervised_learning.html) может показаться ошеломляющим на первый взгляд. Следующие методы включают техники классификации:

- Линейные модели
- Машины опорных векторов
- Стохастический градиентный спуск
- Ближайшие соседи
- Гауссовские процессы
- Деревья решений
- Ансамблевые методы (голосующий классификатор)
- Алгоритмы для многоклассовой и многоцелевой классификации (многоклассовая и многометочная классификация, многоклассовая-многоцелевая классификация)

> Вы также можете использовать [нейронные сети для классификации данных](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), но это выходит за рамки данного урока.

### Какой классификатор выбрать?

Итак, какой классификатор выбрать? Часто можно протестировать несколько и выбрать лучший результат. Scikit-learn предлагает [сравнение бок о бок](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на созданном наборе данных, сравнивая KNeighbors, SVC двумя способами, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB и QuadraticDiscrinationAnalysis, визуализируя результаты:

![сравнение классификаторов](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Графики, созданные на основе документации Scikit-learn

> AutoML решает эту проблему, выполняя сравнения в облаке, позволяя вам выбрать лучший алгоритм для ваших данных. Попробуйте [здесь](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Более разумный подход

Более разумный подход, чем просто угадывать, — это следовать идеям из этого загружаемого [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Здесь мы обнаруживаем, что для нашей многоклассовой задачи у нас есть несколько вариантов:

![шпаргалка для многоклассовых задач](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Раздел шпаргалки Microsoft Algorithm Cheat Sheet, описывающий варианты многоклассовой классификации

✅ Скачайте эту шпаргалку, распечатайте и повесьте на стену!

### Рассуждения

Давайте попробуем рассуждать о различных подходах, учитывая наши ограничения:

- **Нейронные сети слишком тяжелы**. Учитывая наш очищенный, но минимальный набор данных и тот факт, что мы проводим обучение локально через ноутбуки, нейронные сети слишком громоздки для этой задачи.
- **Двухклассовый классификатор не подходит**. Мы не используем двухклассовый классификатор, поэтому исключаем one-vs-all.
- **Дерево решений или логистическая регрессия могут подойти**. Дерево решений может подойти, или логистическая регрессия для многоклассовых данных.
- **Многоклассовые усиленные деревья решений решают другую задачу**. Многоклассовое усиленное дерево решений наиболее подходит для непараметрических задач, например, задач, предназначенных для построения ранжирования, поэтому оно нам не подходит.

### Использование Scikit-learn

Мы будем использовать Scikit-learn для анализа наших данных. Однако существует множество способов использования логистической регрессии в Scikit-learn. Ознакомьтесь с [параметрами для передачи](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Существует два важных параметра - `multi_class` и `solver` - которые необходимо указать, когда мы просим Scikit-learn выполнить логистическую регрессию. Значение `multi_class` определяет определенное поведение. Значение solver указывает, какой алгоритм использовать. Не все solvers могут быть использованы с любыми значениями `multi_class`.

Согласно документации, в многоклассовом случае алгоритм обучения:

- **Использует схему one-vs-rest (OvR)**, если параметр `multi_class` установлен в `ovr`
- **Использует функцию потерь перекрестной энтропии**, если параметр `multi_class` установлен в `multinomial`. (В настоящее время опция `multinomial` поддерживается только solvers ‘lbfgs’, ‘sag’, ‘saga’ и ‘newton-cg’.)"

> 🎓 "Схема" здесь может быть либо 'ovr' (one-vs-rest), либо 'multinomial'. Поскольку логистическая регрессия изначально предназначена для поддержки бинарной классификации, эти схемы позволяют ей лучше справляться с задачами многоклассовой классификации. [источник](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "Solver" определяется как "алгоритм, используемый в задаче оптимизации". [источник](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn предлагает эту таблицу, чтобы объяснить, как solvers справляются с различными задачами, представленными различными структурами данных:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Упражнение - разделение данных

Мы можем сосредоточиться на логистической регрессии для нашего первого обучения, так как вы недавно изучали ее в предыдущем уроке.
Разделите ваши данные на группы для обучения и тестирования, вызвав `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Упражнение - применение логистической регрессии

Поскольку вы используете многоклассовый случай, вам нужно выбрать, какую _схему_ использовать и какой _solver_ установить. Используйте LogisticRegression с многоклассовой настройкой и **liblinear** solver для обучения.

1. Создайте логистическую регрессию с multi_class, установленным в `ovr`, и solver, установленным в `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Попробуйте другой solver, например `lbfgs`, который часто устанавливается по умолчанию.
> Обратите внимание, используйте функцию Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) для преобразования данных в плоский вид, когда это необходимо.
Точность составляет более **80%**!

1. Вы можете увидеть, как работает эта модель, протестировав одну строку данных (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Результат выводится:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Попробуйте другой номер строки и проверьте результаты.

1. Углубляясь дальше, вы можете проверить точность этого предсказания:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Результат выводится - индийская кухня является лучшим предположением с высокой вероятностью:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Можете ли вы объяснить, почему модель уверена, что это индийская кухня?

1. Получите больше деталей, напечатав отчет о классификации, как вы делали в уроках по регрессии:

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

## 🚀Задание

В этом уроке вы использовали очищенные данные для создания модели машинного обучения, которая может предсказывать национальную кухню на основе набора ингредиентов. Потратьте время, чтобы изучить множество возможностей, которые Scikit-learn предоставляет для классификации данных. Углубитесь в концепцию 'solver', чтобы понять, что происходит за кулисами.

## [Тест после лекции](https://ff-quizzes.netlify.app/en/ml/)

## Обзор и самостоятельное изучение

Углубитесь немного больше в математику логистической регрессии в [этом уроке](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf).
## Задание 

[Изучите решатели](assignment.md)

---

**Отказ от ответственности**:  
Этот документ был переведен с помощью сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Несмотря на наши усилия обеспечить точность, автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его родном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.