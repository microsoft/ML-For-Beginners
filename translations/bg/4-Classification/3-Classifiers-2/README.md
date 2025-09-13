<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T00:49:19+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "bg"
}
-->
# Класификатори за кухня 2

В този втори урок за класификация ще разгледате повече начини за класифициране на числови данни. Ще научите също за последиците от избора на един класификатор пред друг.

## [Тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/)

### Предпоставки

Предполагаме, че сте завършили предишните уроци и имате почистен набор от данни в папката `data`, наречен _cleaned_cuisines.csv_, в основната директория на тази папка с 4 урока.

### Подготовка

Заредили сме вашия файл _notebook.ipynb_ с почистения набор от данни и сме го разделили на X и y датафреймове, готови за процеса на изграждане на модела.

## Карта за класификация

По-рано научихте за различните опции, които имате при класифициране на данни, използвайки помощния лист на Microsoft. Scikit-learn предлага подобен, но по-подробен помощен лист, който може допълнително да помогне за стесняване на избора на оценители (друг термин за класификатори):

![ML Карта от Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Съвет: [посетете тази карта онлайн](https://scikit-learn.org/stable/tutorial/machine_learning_map/) и кликнете по пътя, за да прочетете документацията.

### Планът

Тази карта е много полезна, когато имате ясна представа за вашите данни, тъй като можете да „вървите“ по нейните пътеки към решение:

- Имаме >50 проби
- Искаме да предвидим категория
- Имаме етикетирани данни
- Имаме по-малко от 100K проби
- ✨ Можем да изберем Linear SVC
- Ако това не работи, тъй като имаме числови данни
    - Можем да опитаме ✨ KNeighbors Classifier 
      - Ако това не работи, опитайте ✨ SVC и ✨ Ensemble Classifiers

Това е много полезен път за следване.

## Упражнение - разделете данните

Следвайки този път, трябва да започнем с импортиране на някои библиотеки за използване.

1. Импортирайте необходимите библиотеки:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Разделете вашите тренировъчни и тестови данни:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Класификатор Linear SVC

Support-Vector clustering (SVC) е част от семейството на Support-Vector machines техники за машинно обучение (научете повече за тях по-долу). В този метод можете да изберете „ядро“, за да решите как да класифицирате етикетите. Параметърът 'C' се отнася до 'регуларизация', която регулира влиянието на параметрите. Ядрото може да бъде едно от [няколко](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); тук го задаваме на 'linear', за да гарантираме, че използваме Linear SVC. Вероятността по подразбиране е 'false'; тук я задаваме на 'true', за да съберем оценки за вероятност. Задаваме случайното състояние на '0', за да разбъркаме данните и да получим вероятности.

### Упражнение - приложете Linear SVC

Започнете, като създадете масив от класификатори. Ще добавяте постепенно към този масив, докато тестваме.

1. Започнете с Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Обучете вашия модел, използвайки Linear SVC, и отпечатайте отчет:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Резултатът е доста добър:

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

## Класификатор K-Neighbors

K-Neighbors е част от семейството "neighbors" методи за машинно обучение, които могат да се използват както за контролирано, така и за неконтролирано обучение. В този метод се създава предварително определен брой точки и данните се събират около тези точки, така че да могат да се предвидят обобщени етикети за данните.

### Упражнение - приложете класификатора K-Neighbors

Предишният класификатор беше добър и работеше добре с данните, но може би можем да постигнем по-добра точност. Опитайте класификатор K-Neighbors.

1. Добавете ред към масива с класификатори (добавете запетая след елемента Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Резултатът е малко по-лош:

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

    ✅ Научете повече за [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Класификатор Support Vector

Класификаторите Support-Vector са част от семейството [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) методи за машинно обучение, които се използват за задачи по класификация и регресия. SVMs „картират тренировъчните примери към точки в пространството“, за да максимизират разстоянието между две категории. Последващите данни се картографират в това пространство, за да се предвиди тяхната категория.

### Упражнение - приложете класификатор Support Vector

Нека опитаме за малко по-добра точност с класификатор Support Vector.

1. Добавете запетая след елемента K-Neighbors и след това добавете този ред:

    ```python
    'SVC': SVC(),
    ```

    Резултатът е доста добър!

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

    ✅ Научете повече за [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Класификатори Ensemble

Нека следваме пътя до самия край, въпреки че предишният тест беше доста добър. Нека опитаме някои 'Ensemble Classifiers', конкретно Random Forest и AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Резултатът е много добър, особено за Random Forest:

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

✅ Научете повече за [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Този метод на машинно обучение „комбинира прогнозите на няколко базови оценители“, за да подобри качеството на модела. В нашия пример използвахме Random Trees и AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), метод за осредняване, изгражда „гора“ от „решаващи дървета“, изпълнени със случайност, за да се избегне прекомерно напасване. Параметърът n_estimators е зададен на броя на дърветата.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) напасва класификатор към набор от данни и след това напасва копия на този класификатор към същия набор от данни. Той се фокусира върху теглата на неправилно класифицираните елементи и коригира напасването за следващия класификатор, за да ги поправи.

---

## 🚀Предизвикателство

Всеки от тези техники има голям брой параметри, които можете да настроите. Проучете стандартните параметри на всеки и помислете какво би означавало настройването на тези параметри за качеството на модела.

## [Тест след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостоятелно обучение

Има много жаргон в тези уроци, така че отделете минута, за да прегледате [този списък](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) с полезна терминология!

## Задание 

[Игра с параметри](assignment.md)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за каквито и да е недоразумения или погрешни интерпретации, произтичащи от използването на този превод.