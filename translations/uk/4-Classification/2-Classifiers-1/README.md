<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T13:06:26+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "uk"
}
-->
# Класифікатори кухонь 1

У цьому уроці ви будете використовувати набір даних, який ви зберегли з попереднього уроку, наповнений збалансованими та очищеними даними про кухні.

Ви будете використовувати цей набір даних із різними класифікаторами, щоб _передбачити національну кухню на основі групи інгредієнтів_. Під час цього ви дізнаєтеся більше про способи використання алгоритмів для задач класифікації.

## [Тест перед лекцією](https://ff-quizzes.netlify.app/en/ml/)
# Підготовка

Якщо ви завершили [Урок 1](../1-Introduction/README.md), переконайтеся, що файл _cleaned_cuisines.csv_ знаходиться в кореневій папці `/data` для цих чотирьох уроків.

## Вправа - передбачення національної кухні

1. Працюючи в папці _notebook.ipynb_ цього уроку, імпортуйте цей файл разом із бібліотекою Pandas:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Дані виглядають так:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Тепер імпортуйте ще кілька бібліотек:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Розділіть координати X та y на два датафрейми для тренування. `cuisine` може бути датафреймом міток:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    Це виглядатиме так:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Видаліть стовпці `Unnamed: 0` та `cuisine`, використовуючи `drop()`. Збережіть решту даних як тренувальні ознаки:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Ваші ознаки виглядають так:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Тепер ви готові тренувати вашу модель!

## Вибір класифікатора

Тепер, коли ваші дані очищені та готові до тренування, вам потрібно вирішити, який алгоритм використовувати для задачі.

Scikit-learn групує класифікацію під Навчання з учителем, і в цій категорії ви знайдете багато способів класифікації. [Різноманіття](https://scikit-learn.org/stable/supervised_learning.html) може здатися приголомшливим на перший погляд. Наступні методи включають техніки класифікації:

- Лінійні моделі
- Машини опорних векторів
- Стохастичний градієнтний спуск
- Найближчі сусіди
- Гауссові процеси
- Дерева рішень
- Ансамблеві методи (голосуючий класифікатор)
- Алгоритми для багатокласових і багатовихідних задач (багатокласова і багатоміткова класифікація, багатокласова-багатовихідна класифікація)

> Ви також можете використовувати [нейронні мережі для класифікації даних](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), але це виходить за рамки цього уроку.

### Який класифікатор обрати?

Отже, який класифікатор слід обрати? Часто тестування кількох класифікаторів і пошук найкращого результату є способом перевірки. Scikit-learn пропонує [порівняння](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) на створеному наборі даних, порівнюючи KNeighbors, SVC двома способами, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB і QuadraticDiscriminationAnalysis, показуючи результати візуалізованими:

![порівняння класифікаторів](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Графіки, створені на документації Scikit-learn

> AutoML вирішує цю проблему, виконуючи ці порівняння в хмарі, дозволяючи вам обрати найкращий алгоритм для ваших даних. Спробуйте [тут](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Кращий підхід

Кращий спосіб, ніж просто здогадуватися, — це слідувати ідеям із цього завантажуваного [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Тут ми дізнаємося, що для нашої багатокласової задачі у нас є кілька варіантів:

![шпаргалка для багатокласових задач](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Частина шпаргалки Microsoft Algorithm Cheat Sheet, що описує варіанти багатокласової класифікації

✅ Завантажте цю шпаргалку, роздрукуйте її та повісьте на стіну!

### Міркування

Давайте спробуємо розібратися в різних підходах, враховуючи наші обмеження:

- **Нейронні мережі занадто важкі**. З огляду на наш очищений, але мінімальний набір даних, і той факт, що ми запускаємо тренування локально через ноутбуки, нейронні мережі занадто важкі для цього завдання.
- **Не використовуємо класифікатор для двох класів**. Ми не використовуємо класифікатор для двох класів, тому це виключає one-vs-all.
- **Дерева рішень або логістична регресія можуть підійти**. Дерева рішень можуть підійти, або логістична регресія для багатокласових даних.
- **Багатокласові Boosted Decision Trees вирішують іншу задачу**. Багатокласове Boosted Decision Tree найбільш підходить для непараметричних задач, наприклад, задач, спрямованих на створення рейтингів, тому це нам не підходить.

### Використання Scikit-learn 

Ми будемо використовувати Scikit-learn для аналізу наших даних. Однак існує багато способів використання логістичної регресії в Scikit-learn. Ознайомтеся з [параметрами для передачі](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Фактично, є два важливі параметри - `multi_class` і `solver` - які нам потрібно вказати, коли ми просимо Scikit-learn виконати логістичну регресію. Значення `multi_class` визначає певну поведінку. Значення solver визначає, який алгоритм використовувати. Не всі solvers можна поєднувати з усіма значеннями `multi_class`.

Згідно з документацією, у випадку багатокласової задачі, алгоритм навчання:

- **Використовує схему one-vs-rest (OvR)**, якщо параметр `multi_class` встановлено як `ovr`
- **Використовує функцію втрат крос-ентропії**, якщо параметр `multi_class` встановлено як `multinomial`. (На даний момент опція `multinomial` підтримується лише solvers: ‘lbfgs’, ‘sag’, ‘saga’ і ‘newton-cg’.)

> 🎓 "Схема" тут може бути або 'ovr' (one-vs-rest), або 'multinomial'. Оскільки логістична регресія насправді призначена для підтримки бінарної класифікації, ці схеми дозволяють їй краще справлятися з багатокласовими задачами класифікації. [джерело](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 "Solver" визначається як "алгоритм, який використовується для вирішення задачі оптимізації". [джерело](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn пропонує цю таблицю, щоб пояснити, як solvers справляються з різними викликами, представленими різними типами структур даних:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Вправа - розділіть дані

Ми можемо зосередитися на логістичній регресії для нашого першого тренувального випробування, оскільки ви нещодавно вивчали її на попередньому уроці.
Розділіть ваші дані на тренувальні та тестові групи, викликавши `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Вправа - застосуйте логістичну регресію

Оскільки ви використовуєте багатокласовий випадок, вам потрібно вибрати, яку _схему_ використовувати і який _solver_ встановити. Використовуйте LogisticRegression із багатокласовим налаштуванням і **liblinear** solver для тренування.

1. Створіть логістичну регресію з multi_class, встановленим як `ovr`, і solver, встановленим як `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Спробуйте інший solver, наприклад `lbfgs`, який часто встановлюється за замовчуванням.
> Зверніть увагу, використовуйте функцію Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) для згладжування ваших даних, коли це необхідно.
Точність становить понад **80%**!

1. Ви можете побачити, як ця модель працює, протестувавши один рядок даних (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Результат виводиться:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Спробуйте інший номер рядка та перевірте результати.

1. Заглиблюючись, ви можете перевірити точність цього прогнозу:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Результат виводиться - індійська кухня є найкращим припущенням моделі з високою ймовірністю:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Чи можете ви пояснити, чому модель досить впевнена, що це індійська кухня?

1. Отримайте більше деталей, вивівши звіт про класифікацію, як ви робили в уроках регресії:

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

## 🚀Виклик

У цьому уроці ви використали очищені дані для створення моделі машинного навчання, яка може передбачати національну кухню на основі серії інгредієнтів. Приділіть час, щоб ознайомитися з багатьма варіантами, які Scikit-learn пропонує для класифікації даних. Заглибтеся в концепцію 'solver', щоб зрозуміти, що відбувається за лаштунками.

## [Тест після лекції](https://ff-quizzes.netlify.app/en/ml/)

## Огляд і самостійне навчання

Заглибтеся трохи більше в математику логістичної регресії в [цьому уроці](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Завдання 

[Вивчіть solvers](assignment.md)

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, зверніть увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для критично важливої інформації рекомендується професійний людський переклад. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.