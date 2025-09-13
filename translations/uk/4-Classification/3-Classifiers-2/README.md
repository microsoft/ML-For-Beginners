<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T13:15:00+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "uk"
}
-->
# Класифікатори кухонь 2

У цьому другому уроці класифікації ви дослідите більше способів класифікації числових даних. Ви також дізнаєтеся про наслідки вибору одного класифікатора над іншим.

## [Тест перед лекцією](https://ff-quizzes.netlify.app/en/ml/)

### Попередні знання

Ми припускаємо, що ви завершили попередні уроки та маєте очищений набір даних у папці `data`, який називається _cleaned_cuisines.csv_ у кореневій папці цього 4-урочного блоку.

### Підготовка

Ми завантажили ваш файл _notebook.ipynb_ з очищеним набором даних і розділили його на X та y датафрейми, готові до процесу побудови моделі.

## Карта класифікації

Раніше ви дізналися про різні варіанти класифікації даних, використовуючи шпаргалку Microsoft. Scikit-learn пропонує схожу, але більш детальну шпаргалку, яка може допомогти ще більше звузити вибір оцінювачів (інша назва класифікаторів):

![Карта ML від Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Порада: [відвідайте цю карту онлайн](https://scikit-learn.org/stable/tutorial/machine_learning_map/) і натискайте на шляхи, щоб прочитати документацію.

### План

Ця карта дуже корисна, коли ви добре розумієте свої дані, оскільки ви можете "пройти" її шляхами до рішення:

- У нас є >50 зразків
- Ми хочемо передбачити категорію
- У нас є мічені дані
- У нас менше ніж 100К зразків
- ✨ Ми можемо вибрати Linear SVC
- Якщо це не спрацює, оскільки у нас числові дані
    - Ми можемо спробувати ✨ KNeighbors Classifier 
      - Якщо це не спрацює, спробуйте ✨ SVC та ✨ Ensemble Classifiers

Це дуже корисний шлях для слідування.

## Вправа - розділіть дані

Слідуючи цьому шляху, ми повинні почати з імпорту деяких бібліотек для використання.

1. Імпортуйте необхідні бібліотеки:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Розділіть ваші дані на тренувальні та тестові:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Класифікатор Linear SVC

Support-Vector clustering (SVC) є частиною сімейства методів машинного навчання Support-Vector machines (докладніше про них нижче). У цьому методі ви можете вибрати "ядро", щоб вирішити, як кластеризувати мітки. Параметр 'C' стосується "регуляризації", яка регулює вплив параметрів. Ядро може бути одним із [кількох](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); тут ми встановлюємо його як 'linear', щоб використовувати Linear SVC. За замовчуванням ймовірність встановлена як 'false'; тут ми встановлюємо її як 'true', щоб отримати оцінки ймовірності. Ми встановлюємо random state як '0', щоб перемішати дані для отримання ймовірностей.

### Вправа - застосуйте Linear SVC

Почніть із створення масиву класифікаторів. Ви будете поступово додавати до цього масиву, тестуючи.

1. Почніть із Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Навчіть вашу модель, використовуючи Linear SVC, і виведіть звіт:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Результат досить хороший:

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

## Класифікатор K-Neighbors

K-Neighbors є частиною сімейства методів ML "neighbors", які можуть використовуватися як для контрольованого, так і для неконтрольованого навчання. У цьому методі створюється задана кількість точок, і дані збираються навколо цих точок таким чином, щоб можна було передбачити узагальнені мітки для даних.

### Вправа - застосуйте класифікатор K-Neighbors

Попередній класифікатор був хорошим і добре працював із даними, але, можливо, ми можемо отримати кращу точність. Спробуйте класифікатор K-Neighbors.

1. Додайте рядок до вашого масиву класифікаторів (додайте кому після елемента Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Результат трохи гірший:

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

    ✅ Дізнайтеся більше про [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Класифікатор Support Vector

Класифікатори Support-Vector є частиною сімейства методів ML [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine), які використовуються для задач класифікації та регресії. SVM "відображають навчальні приклади в точки в просторі", щоб максимізувати відстань між двома категоріями. Наступні дані відображаються в цьому просторі, щоб можна було передбачити їхню категорію.

### Вправа - застосуйте класифікатор Support Vector

Спробуємо отримати трохи кращу точність за допомогою класифікатора Support Vector.

1. Додайте кому після елемента K-Neighbors, а потім додайте цей рядок:

    ```python
    'SVC': SVC(),
    ```

    Результат досить хороший!

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

    ✅ Дізнайтеся більше про [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Давайте пройдемо шлях до самого кінця, навіть якщо попередній тест був досить хорошим. Спробуємо деякі класифікатори 'Ensemble', зокрема Random Forest та AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Результат дуже хороший, особливо для Random Forest:

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

✅ Дізнайтеся більше про [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

Цей метод машинного навчання "об'єднує прогнози кількох базових оцінювачів", щоб покращити якість моделі. У нашому прикладі ми використовували Random Trees та AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), метод усереднення, будує "ліс" із "дерев рішень", наповнених випадковістю, щоб уникнути перенавчання. Параметр n_estimators встановлюється як кількість дерев.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) підбирає класифікатор до набору даних, а потім підбирає копії цього класифікатора до того ж набору даних. Він зосереджується на вагах неправильно класифікованих елементів і коригує підбір для наступного класифікатора, щоб виправити помилки.

---

## 🚀Виклик

Кожен із цих методів має велику кількість параметрів, які ви можете налаштувати. Дослідіть параметри за замовчуванням кожного методу та подумайте, що означатиме їхнє налаштування для якості моделі.

## [Тест після лекції](https://ff-quizzes.netlify.app/en/ml/)

## Огляд та самостійне навчання

У цих уроках багато термінології, тому приділіть хвилинку, щоб переглянути [цей список](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) корисних термінів!

## Завдання 

[Гра з параметрами](assignment.md)

---

**Відмова від відповідальності**:  
Цей документ було перекладено за допомогою сервісу автоматичного перекладу [Co-op Translator](https://github.com/Azure/co-op-translator). Хоча ми прагнемо до точності, звертаємо вашу увагу, що автоматичні переклади можуть містити помилки або неточності. Оригінальний документ мовою оригіналу слід вважати авторитетним джерелом. Для критично важливої інформації рекомендується професійний переклад людиною. Ми не несемо відповідальності за будь-які непорозуміння або неправильні тлумачення, що виникли внаслідок використання цього перекладу.