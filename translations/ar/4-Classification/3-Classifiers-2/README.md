<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-04T20:50:07+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ar"
}
-->
# مصنفات المأكولات 2

في درس التصنيف الثاني هذا، ستستكشف طرقًا إضافية لتصنيف البيانات الرقمية. كما ستتعرف على العواقب المترتبة على اختيار مصنف معين بدلاً من آخر.

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

### المتطلبات الأساسية

نفترض أنك قد أكملت الدروس السابقة ولديك مجموعة بيانات نظيفة في مجلد `data` باسم _cleaned_cuisines.csv_ في جذر هذا المجلد المكون من 4 دروس.

### التحضير

قمنا بتحميل ملف _notebook.ipynb_ الخاص بك مع مجموعة البيانات النظيفة وقمنا بتقسيمها إلى إطاري بيانات X و y، وهي جاهزة لعملية بناء النموذج.

## خريطة التصنيف

في الدرس السابق، تعلمت عن الخيارات المختلفة المتاحة لتصنيف البيانات باستخدام ورقة الغش الخاصة بمايكروسوفت. تقدم مكتبة Scikit-learn ورقة غش مشابهة ولكن أكثر تفصيلًا يمكن أن تساعدك بشكل أكبر في تضييق نطاق المصنفات (وهو مصطلح آخر للمصنفات):

![خريطة تعلم الآلة من Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> نصيحة: [قم بزيارة هذه الخريطة عبر الإنترنت](https://scikit-learn.org/stable/tutorial/machine_learning_map/) وانقر على المسارات لقراءة الوثائق.

### الخطة

تعد هذه الخريطة مفيدة جدًا بمجرد أن تكون لديك فكرة واضحة عن بياناتك، حيث يمكنك "السير" على طول مساراتها لاتخاذ قرار:

- لدينا >50 عينة
- نريد التنبؤ بفئة
- لدينا بيانات معنونة
- لدينا أقل من 100 ألف عينة
- ✨ يمكننا اختيار Linear SVC
- إذا لم ينجح ذلك، بما أن لدينا بيانات رقمية
    - يمكننا تجربة ✨ KNeighbors Classifier 
      - إذا لم ينجح ذلك، جرب ✨ SVC و ✨ Ensemble Classifiers

هذا مسار مفيد جدًا للمتابعة.

## تمرين - تقسيم البيانات

باتباع هذا المسار، يجب أن نبدأ باستيراد بعض المكتبات اللازمة.

1. استيراد المكتبات المطلوبة:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. قسّم بيانات التدريب والاختبار:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## مصنف Linear SVC

يعد Support-Vector Clustering (SVC) أحد أفراد عائلة تقنيات تعلم الآلة Support-Vector Machines (تعرف على المزيد عنها أدناه). في هذه الطريقة، يمكنك اختيار "نواة" لتحديد كيفية تجميع العلامات. يشير المعامل 'C' إلى "التنظيم" الذي ينظم تأثير المعاملات. يمكن أن تكون النواة واحدة من [عدة خيارات](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)؛ هنا نضبطها على 'linear' لضمان استخدام Linear SVC. يتم ضبط الاحتمالية افتراضيًا على 'false'؛ هنا نضبطها على 'true' للحصول على تقديرات الاحتمالية. نضبط الحالة العشوائية على '0' لخلط البيانات للحصول على الاحتمالات.

### تمرين - تطبيق Linear SVC

ابدأ بإنشاء مصفوفة من المصنفات. ستضيف تدريجيًا إلى هذه المصفوفة أثناء الاختبار.

1. ابدأ باستخدام Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. قم بتدريب النموذج باستخدام Linear SVC واطبع تقريرًا:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    النتيجة جيدة جدًا:

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

## مصنف K-Neighbors

K-Neighbors هو جزء من عائلة "الجيران" لطرق تعلم الآلة، والتي يمكن استخدامها للتعلم الموجه وغير الموجه. في هذه الطريقة، يتم إنشاء عدد محدد مسبقًا من النقاط ويتم جمع البيانات حول هذه النقاط بحيث يمكن التنبؤ بالعلامات العامة للبيانات.

### تمرين - تطبيق مصنف K-Neighbors

كان المصنف السابق جيدًا وعمل بشكل جيد مع البيانات، ولكن ربما يمكننا تحقيق دقة أفضل. جرب مصنف K-Neighbors.

1. أضف سطرًا إلى مصفوفة المصنفات (أضف فاصلة بعد عنصر Linear SVC):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    النتيجة أسوأ قليلاً:

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

    ✅ تعرف على المزيد حول [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## مصنف Support Vector

مصنفات Support-Vector هي جزء من عائلة [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) لطرق تعلم الآلة التي تُستخدم لمهام التصنيف والانحدار. تقوم SVMs "برسم أمثلة التدريب كنقاط في الفضاء" لزيادة المسافة بين فئتين. يتم بعد ذلك رسم البيانات اللاحقة في هذا الفضاء بحيث يمكن التنبؤ بفئتها.

### تمرين - تطبيق مصنف Support Vector

دعنا نحاول تحقيق دقة أفضل قليلاً باستخدام مصنف Support Vector.

1. أضف فاصلة بعد عنصر K-Neighbors، ثم أضف هذا السطر:

    ```python
    'SVC': SVC(),
    ```

    النتيجة جيدة جدًا!

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

    ✅ تعرف على المزيد حول [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## المصنفات المجمعة (Ensemble Classifiers)

دعنا نتبع المسار حتى النهاية، على الرغم من أن الاختبار السابق كان جيدًا جدًا. دعنا نجرب بعض "المصنفات المجمعة"، وتحديدًا Random Forest و AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

النتيجة جيدة جدًا، خاصة بالنسبة لـ Random Forest:

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

✅ تعرف على المزيد حول [المصنفات المجمعة](https://scikit-learn.org/stable/modules/ensemble.html)

تجمع هذه الطريقة في تعلم الآلة "تنبؤات عدة مصنفات أساسية" لتحسين جودة النموذج. في مثالنا، استخدمنا Random Trees و AdaBoost.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)، وهي طريقة تعتمد على المتوسط، تبني "غابة" من "أشجار القرار" مع إضافة العشوائية لتجنب الإفراط في التخصيص. يتم ضبط معامل n_estimators على عدد الأشجار.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) يقوم بتدريب مصنف على مجموعة بيانات ثم يقوم بتدريب نسخ من هذا المصنف على نفس مجموعة البيانات. يركز على أوزان العناصر التي تم تصنيفها بشكل خاطئ ويعدل التخصيص للمصنف التالي لتصحيحها.

---

## 🚀تحدي

كل من هذه التقنيات لديها عدد كبير من المعاملات التي يمكنك تعديلها. قم بالبحث عن المعاملات الافتراضية لكل تقنية وفكر في ما يعنيه تعديل هذه المعاملات لجودة النموذج.

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

هناك الكثير من المصطلحات في هذه الدروس، لذا خذ دقيقة لمراجعة [هذه القائمة](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) من المصطلحات المفيدة!

## الواجب

[لعب المعاملات](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.