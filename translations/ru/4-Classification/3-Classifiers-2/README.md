<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T08:35:18+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ru"
}
-->
# Классификаторы кухонь 2

Во втором уроке по классификации вы изучите дополнительные методы классификации числовых данных. Вы также узнаете о последствиях выбора одного классификатора вместо другого.

## [Тест перед лекцией](https://ff-quizzes.netlify.app/en/ml/)

### Предварительные знания

Мы предполагаем, что вы завершили предыдущие уроки и у вас есть очищенный набор данных в папке `data` под названием _cleaned_cuisines.csv_ в корневой папке этого курса из 4 уроков.

### Подготовка

Мы загрузили ваш файл _notebook.ipynb_ с очищенным набором данных и разделили его на датафреймы X и y, готовые для процесса построения модели.

## Карта классификации

Ранее вы узнали о различных вариантах классификации данных, используя шпаргалку Microsoft. Scikit-learn предлагает похожую, но более детализированную шпаргалку, которая поможет сузить выбор оценщиков (другое название классификаторов):

![Карта ML от Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Совет: [посетите эту карту онлайн](https://scikit-learn.org/stable/tutorial/machine_learning_map/) и следуйте по пути, чтобы изучить документацию.

### План

Эта карта очень полезна, если вы хорошо понимаете свои данные, так как вы можете "пройти" по её путям к решению:

- У нас есть >50 образцов
- Мы хотим предсказать категорию
- У нас есть размеченные данные
- У нас менее 100K образцов
- ✨ Мы можем выбрать Linear SVC
- Если это не сработает, так как у нас числовые данные:
    - Мы можем попробовать ✨ KNeighbors Classifier 
      - Если это не сработает, попробуйте ✨ SVC и ✨ Ensemble Classifiers

Это очень полезный путь для следования.

## Упражнение - разделите данные

Следуя этому пути, начнем с импорта необходимых библиотек.

1. Импортируйте нужные библиотеки:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Разделите данные на тренировочные и тестовые:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Классификатор Linear SVC

Кластеризация с использованием метода опорных векторов (SVC) является частью семейства методов машинного обучения Support-Vector Machines (SVM). В этом методе вы можете выбрать "ядро" для определения способа кластеризации меток. Параметр 'C' относится к "регуляризации", которая регулирует влияние параметров. Ядро может быть одним из [нескольких](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); здесь мы устанавливаем его как 'linear', чтобы использовать линейный SVC. По умолчанию вероятность установлена как 'false'; здесь мы устанавливаем её как 'true', чтобы получить оценки вероятности. Мы устанавливаем random state как '0', чтобы перемешать данные для получения вероятностей.

### Упражнение - примените Linear SVC

Начните с создания массива классификаторов. Вы будете постепенно добавлять в этот массив по мере тестирования.

1. Начните с Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Обучите вашу модель, используя Linear SVC, и выведите отчет:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Результат довольно хороший:

    ```output
    Accuracy (train) for Linear SVC: 78.6% 
                  precision    recall  f1-score   support
    
         chinese       0.71      0.67      0.69       242
          indian       0.88      0.86      0.87       234
        japanese       0.79      0.74      0.76       254
          korean       0.85      0.81      0.83       242
            thai       0.71      0.86      0.78       227
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

## Классификатор K-Neighbors

K-Neighbors является частью семейства методов машинного обучения "соседи", которые могут использоваться как для контролируемого, так и для неконтролируемого обучения. В этом методе создается заранее определенное количество точек, и данные собираются вокруг этих точек таким образом, чтобы можно было предсказать обобщенные метки для данных.

### Упражнение - примените классификатор K-Neighbors

Предыдущий классификатор был хорош и хорошо работал с данными, но, возможно, мы можем добиться лучшей точности. Попробуйте классификатор K-Neighbors.

1. Добавьте строку в массив классификаторов (добавьте запятую после элемента Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Результат немного хуже:

    ```output
    Accuracy (train) for KNN classifier: 73.8% 
                  precision    recall  f1-score   support
    
         chinese       0.64      0.67      0.66       242
          indian       0.86      0.78      0.82       234
        japanese       0.66      0.83      0.74       254
          korean       0.94      0.58      0.72       242
            thai       0.71      0.82      0.76       227
    
        accuracy                           0.74      1199
       macro avg       0.76      0.74      0.74      1199
    weighted avg       0.76      0.74      0.74      1199
    ```

    ✅ Узнайте больше о [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Классификатор Support Vector

Классификаторы Support Vector являются частью семейства методов машинного обучения [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), которые используются для задач классификации и регрессии. SVM "отображают обучающие примеры в точки в пространстве", чтобы максимизировать расстояние между двумя категориями. Последующие данные отображаются в этом пространстве, чтобы предсказать их категорию.

### Упражнение - примените классификатор Support Vector

Попробуем добиться немного лучшей точности с помощью классификатора Support Vector.

1. Добавьте запятую после элемента K-Neighbors, а затем добавьте эту строку:

    ```python
    'SVC': SVC(),
    ```

    Результат довольно хороший!

    ```output
    Accuracy (train) for SVC: 83.2% 
                  precision    recall  f1-score   support
    
         chinese       0.79      0.74      0.76       242
          indian       0.88      0.90      0.89       234
        japanese       0.87      0.81      0.84       254
          korean       0.91      0.82      0.86       242
            thai       0.74      0.90      0.81       227
    
        accuracy                           0.83      1199
       macro avg       0.84      0.83      0.83      1199
    weighted avg       0.84      0.83      0.83      1199
    ```

    ✅ Узнайте больше о [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ансамблевые классификаторы

Давайте пройдем путь до самого конца, даже если предыдущий тест был довольно хорош. Попробуем некоторые ансамблевые классификаторы, в частности Random Forest и AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Результат очень хороший, особенно для Random Forest:

```output
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

✅ Узнайте больше о [Ансамблевых классификаторах](https://scikit-learn.org/stable/modules/ensemble.html)

Этот метод машинного обучения "объединяет предсказания нескольких базовых оценщиков", чтобы улучшить качество модели. В нашем примере мы использовали Random Trees и AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), метод усреднения, строит "лес" из "деревьев решений", насыщенных случайностью, чтобы избежать переобучения. Параметр n_estimators устанавливается как количество деревьев.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) обучает классификатор на наборе данных, а затем обучает копии этого классификатора на том же наборе данных. Он фокусируется на весах неправильно классифицированных элементов и корректирует подгонку для следующего классификатора, чтобы исправить ошибки.

---

## 🚀Задача

У каждого из этих методов есть множество параметров, которые можно настроить. Изучите параметры по умолчанию для каждого метода и подумайте, что изменение этих параметров может означать для качества модели.

## [Тест после лекции](https://ff-quizzes.netlify.app/en/ml/)

## Обзор и самостоятельное изучение

В этих уроках много терминологии, поэтому уделите минуту, чтобы изучить [этот список](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) полезных терминов!

## Задание 

[Игра с параметрами](assignment.md)

---

**Отказ от ответственности**:  
Этот документ был переведен с использованием сервиса автоматического перевода [Co-op Translator](https://github.com/Azure/co-op-translator). Хотя мы стремимся к точности, пожалуйста, имейте в виду, что автоматические переводы могут содержать ошибки или неточности. Оригинальный документ на его исходном языке следует считать авторитетным источником. Для получения критически важной информации рекомендуется профессиональный перевод человеком. Мы не несем ответственности за любые недоразумения или неправильные интерпретации, возникшие в результате использования данного перевода.