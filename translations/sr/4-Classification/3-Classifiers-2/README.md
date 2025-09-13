<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T13:13:50+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "sr"
}
-->
# Класификатори кухиња 2

У овој другој лекцији о класификацији, истражићете више начина за класификацију нумеричких података. Такође ћете научити о последицама избора једног класификатора у односу на други.

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

### Предуслови

Претпостављамо да сте завршили претходне лекције и да имате очишћен скуп података у вашем фолдеру `data` под називом _cleaned_cuisines.csv_ у корену овог фолдера са 4 лекције.

### Припрема

Учитали смо ваш фајл _notebook.ipynb_ са очишћеним скупом података и поделили га на X и y оквире података, спремне за процес изградње модела.

## Мапа класификације

Претходно сте научили о различитим опцијама које имате приликом класификације података користећи Microsoft-ов „cheat sheet“. Scikit-learn нуди сличан, али детаљнији „cheat sheet“ који вам може додатно помоћи да сузите избор класификатора (други термин за класификаторе):

![МЛ мапа из Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)  
> Савет: [погледајте ову мапу онлајн](https://scikit-learn.org/stable/tutorial/machine_learning_map/) и кликните дуж путање да бисте прочитали документацију.

### План

Ова мапа је веома корисна када имате јасну представу о вашим подацима, јер можете „ходати“ дуж њених путања до одлуке:

- Имамо >50 узорака  
- Желимо да предвидимо категорију  
- Имамо означене податке  
- Имамо мање од 100.000 узорака  
- ✨ Можемо изабрати Linear SVC  
- Ако то не ради, пошто имамо нумеричке податке  
    - Можемо пробати ✨ KNeighbors Classifier  
      - Ако то не ради, пробајте ✨ SVC и ✨ Ensemble Classifiers  

Ово је веома корисна путања коју треба пратити.

## Вежба - поделите податке

Пратећи ову путању, требало би да почнемо увозом неких библиотека за коришћење.

1. Увезите потребне библиотеке:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

2. Поделите ваше податке на тренинг и тест:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC класификатор

Support-Vector Clustering (SVC) је део породице техника машинског учења Support-Vector Machines (СВМ) (више о томе у наставку). У овом методу, можете изабрати „kernel“ да одлучите како да групишете ознаке. Параметар 'C' се односи на 'регуларизацију', која регулише утицај параметара. Kernel може бити један од [неколико](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); овде га постављамо на 'linear' како бисмо искористили Linear SVC. Вероватноћа је подразумевано 'false'; овде је постављамо на 'true' како бисмо добили процене вероватноће. Постављамо random state на '0' како бисмо измешали податке за добијање вероватноћа.

### Вежба - примените Linear SVC

Почните са креирањем низа класификатора. Додаваћете постепено у овај низ како будемо тестирали.

1. Почните са Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Тренирајте ваш модел користећи Linear SVC и испишите извештај:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    Резултат је прилично добар:

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

## K-Neighbors класификатор

K-Neighbors је део породице метода машинског учења „neighbors“, које се могу користити за надгледано и ненадгледано учење. У овом методу, унапред дефинисан број тачака се креира, а подаци се групишу око тих тачака тако да се могу предвидети генерализоване ознаке за податке.

### Вежба - примените K-Neighbors класификатор

Претходни класификатор је био добар и добро је радио са подацима, али можда можемо добити бољу тачност. Пробајте K-Neighbors класификатор.

1. Додајте линију у ваш низ класификатора (додајте зарез након Linear SVC ставке):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    Резултат је мало лошији:

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

    ✅ Сазнајте више о [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector класификатори су део породице метода машинског учења [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) које се користе за задатке класификације и регресије. СВМ-ови „мапирају примере тренинга у тачке у простору“ како би максимизовали удаљеност између две категорије. Накнадни подаци се мапирају у овај простор како би се предвидела њихова категорија.

### Вежба - примените Support Vector Classifier

Покушајмо да добијемо мало бољу тачност са Support Vector Classifier.

1. Додајте зарез након K-Neighbors ставке, а затим додајте ову линију:

    ```python
    'SVC': SVC(),
    ```

    Резултат је прилично добар!

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

    ✅ Сазнајте више о [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble класификатори

Хајде да пратимо путању до самог краја, иако је претходни тест био прилично добар. Пробајмо неке 'Ensemble класификаторе', конкретно Random Forest и AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Резултат је веома добар, посебно за Random Forest:

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

✅ Сазнајте више о [Ensemble класификаторима](https://scikit-learn.org/stable/modules/ensemble.html)

Овај метод машинског учења „комбинује предвиђања неколико основних процењивача“ како би побољшао квалитет модела. У нашем примеру, користили смо Random Trees и AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), метод просечавања, гради „шуму“ од „одлука дрвећа“ са додатком случајности како би се избегло пренапасавање. Параметар n_estimators је подешен на број стабала.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) прилагођава класификатор скупу података, а затим прилагођава копије тог класификатора истом скупу података. Фокусира се на тежине погрешно класификованих ставки и прилагођава следећи класификатор како би исправио грешке.

---

## 🚀Изазов

Свака од ових техника има велики број параметара које можете подесити. Истражите подразумеване параметре сваке од њих и размислите шта би значило подешавање тих параметара за квалитет модела.

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

У овим лекцијама има доста жаргона, па одвојите минут да прегледате [ову листу](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) корисне терминологије!

## Задатак

[Играње са параметрима](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако настојимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на изворном језику треба сматрати меродавним извором. За критичне информације препоручује се професионални превод од стране људи. Не сносимо одговорност за било каква погрешна тумачења или неспоразуме који могу произаћи из коришћења овог превода.