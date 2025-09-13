<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T08:55:57+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ur"
}
-->
# کھانوں کی درجہ بندی 2

اس دوسرے درجہ بندی کے سبق میں، آپ عددی ڈیٹا کو درجہ بندی کرنے کے مزید طریقے دریافت کریں گے۔ آپ یہ بھی سیکھیں گے کہ ایک درجہ بندی کنندہ کو دوسرے پر منتخب کرنے کے کیا اثرات ہو سکتے ہیں۔

## [لیکچر سے پہلے کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

### پیشگی شرط

ہم فرض کرتے ہیں کہ آپ نے پچھلے اسباق مکمل کر لیے ہیں اور آپ کے پاس `data` فولڈر میں ایک صاف شدہ ڈیٹاسیٹ موجود ہے جس کا نام _cleaned_cuisines.csv_ ہے، جو اس 4-سبق والے فولڈر کے روٹ میں موجود ہے۔

### تیاری

ہم نے آپ کے _notebook.ipynb_ فائل کو صاف شدہ ڈیٹاسیٹ کے ساتھ لوڈ کیا ہے اور اسے X اور y ڈیٹافریمز میں تقسیم کیا ہے، جو ماڈل بنانے کے عمل کے لیے تیار ہیں۔

## ایک درجہ بندی کا نقشہ

پہلے، آپ نے مائیکروسافٹ کے چیٹ شیٹ کا استعمال کرتے ہوئے ڈیٹا کو درجہ بندی کرنے کے مختلف اختیارات کے بارے میں سیکھا۔ Scikit-learn ایک مشابہ لیکن زیادہ تفصیلی چیٹ شیٹ پیش کرتا ہے جو آپ کے تخمینے (درجہ بندی کنندگان کے لیے ایک اور اصطلاح) کو مزید محدود کرنے میں مدد دے سکتا ہے:

![Scikit-learn کا ML نقشہ](../../../../4-Classification/3-Classifiers-2/images/map.png)  
> ٹپ: [اس نقشے کو آن لائن دیکھیں](https://scikit-learn.org/stable/tutorial/machine_learning_map/) اور راستے پر کلک کر کے دستاویزات پڑھیں۔

### منصوبہ

یہ نقشہ اس وقت بہت مددگار ہوتا ہے جب آپ کو اپنے ڈیٹا کی واضح سمجھ ہو، کیونکہ آپ اس کے راستوں پر چل کر فیصلہ کر سکتے ہیں:

- ہمارے پاس >50 نمونے ہیں  
- ہم ایک زمرہ کی پیش گوئی کرنا چاہتے ہیں  
- ہمارے پاس لیبل شدہ ڈیٹا ہے  
- ہمارے پاس 100K سے کم نمونے ہیں  
- ✨ ہم ایک Linear SVC منتخب کر سکتے ہیں  
- اگر یہ کام نہ کرے، چونکہ ہمارے پاس عددی ڈیٹا ہے  
    - ہم ✨ KNeighbors Classifier آزما سکتے ہیں  
      - اگر یہ کام نہ کرے، تو ✨ SVC اور ✨ Ensemble Classifiers آزمائیں  

یہ ایک بہت مددگار راستہ ہے جس پر عمل کیا جا سکتا ہے۔

## مشق - ڈیٹا کو تقسیم کریں

اس راستے پر عمل کرتے ہوئے، ہمیں کچھ لائبریریاں درآمد کرنی چاہئیں۔

1. مطلوبہ لائبریریاں درآمد کریں:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. اپنے تربیتی اور ٹیسٹ ڈیٹا کو تقسیم کریں:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC درجہ بندی کنندہ

Support-Vector Clustering (SVC) مشین لرننگ تکنیکوں کے Support-Vector مشینز خاندان کا حصہ ہے (نیچے ان کے بارے میں مزید جانیں)۔ اس طریقے میں، آپ لیبلز کو کلسٹر کرنے کے لیے ایک 'kernel' منتخب کر سکتے ہیں۔ 'C' پیرامیٹر 'regularization' کو ظاہر کرتا ہے جو پیرامیٹرز کے اثر کو منظم کرتا ہے۔ kernel [کئی](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) میں سے ایک ہو سکتا ہے؛ یہاں ہم اسے 'linear' پر سیٹ کرتے ہیں تاکہ Linear SVC کا فائدہ اٹھایا جا سکے۔ Probability ڈیفالٹ میں 'false' ہوتی ہے؛ یہاں ہم اسے 'true' پر سیٹ کرتے ہیں تاکہ probability estimates حاصل کر سکیں۔ ہم random state کو '0' پر سیٹ کرتے ہیں تاکہ ڈیٹا کو شفل کر کے probabilities حاصل کی جا سکیں۔

### مشق - Linear SVC کا اطلاق کریں

ایک درجہ بندی کنندگان کی صف بنائیں۔ آپ اس صف میں بتدریج اضافہ کریں گے جب ہم ٹیسٹ کریں گے۔

1. Linear SVC سے شروع کریں:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC کا استعمال کرتے ہوئے اپنے ماڈل کو تربیت دیں اور رپورٹ پرنٹ کریں:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    نتیجہ کافی اچھا ہے:

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

## K-Neighbors درجہ بندی کنندہ

K-Neighbors مشین لرننگ کے "پڑوسیوں" خاندان کا حصہ ہے، جو سپروائزڈ اور ان سپروائزڈ لرننگ دونوں کے لیے استعمال کیا جا سکتا ہے۔ اس طریقے میں، ایک پہلے سے طے شدہ تعداد میں پوائنٹس بنائے جاتے ہیں اور ڈیٹا ان پوائنٹس کے ارد گرد جمع کیا جاتا ہے تاکہ ڈیٹا کے لیے عمومی لیبلز کی پیش گوئی کی جا سکے۔

### مشق - K-Neighbors درجہ بندی کنندہ کا اطلاق کریں

پچھلا درجہ بندی کنندہ اچھا تھا اور ڈیٹا کے ساتھ اچھا کام کیا، لیکن شاید ہم بہتر درستگی حاصل کر سکیں۔ K-Neighbors درجہ بندی کنندہ آزمائیں۔

1. اپنے درجہ بندی کنندگان کی صف میں ایک لائن شامل کریں (Linear SVC آئٹم کے بعد ایک کاما شامل کریں):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    نتیجہ تھوڑا خراب ہے:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) کے بارے میں جانیں

## Support Vector درجہ بندی کنندہ

Support-Vector درجہ بندی کنندگان مشین لرننگ کے [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) خاندان کا حصہ ہیں جو درجہ بندی اور ریگریشن کے کاموں کے لیے استعمال ہوتے ہیں۔ SVMs "تربیتی مثالوں کو خلا میں پوائنٹس پر نقشہ بناتے ہیں" تاکہ دو زمروں کے درمیان فاصلہ زیادہ سے زیادہ ہو۔ بعد میں ڈیٹا کو اس خلا میں نقشہ بنایا جاتا ہے تاکہ ان کے زمرے کی پیش گوئی کی جا سکے۔

### مشق - Support Vector درجہ بندی کنندہ کا اطلاق کریں

Support Vector درجہ بندی کنندہ کے ساتھ تھوڑی بہتر درستگی حاصل کرنے کی کوشش کریں۔

1. K-Neighbors آئٹم کے بعد ایک کاما شامل کریں، اور پھر یہ لائن شامل کریں:

    ```python
    'SVC': SVC(),
    ```

    نتیجہ کافی اچھا ہے!

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

    ✅ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) کے بارے میں جانیں

## Ensemble درجہ بندی کنندگان

چاہے پچھلا ٹیسٹ کافی اچھا تھا، آئیے راستے کے آخر تک چلتے ہیں۔ آئیے کچھ 'Ensemble Classifiers' آزمائیں، خاص طور پر Random Forest اور AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

نتیجہ بہت اچھا ہے، خاص طور پر Random Forest کے لیے:

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

✅ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html) کے بارے میں جانیں

مشین لرننگ کا یہ طریقہ "کئی بنیادی تخمینے کے پیش گوئیوں کو یکجا کرتا ہے" تاکہ ماڈل کے معیار کو بہتر بنایا جا سکے۔ ہمارے مثال میں، ہم نے Random Trees اور AdaBoost کا استعمال کیا۔

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)، ایک اوسطی طریقہ، 'فیصلہ درختوں' کا ایک 'جنگل' بناتا ہے جس میں بے ترتیب پن شامل ہوتا ہے تاکہ اوورفٹنگ سے بچا جا سکے۔ n_estimators پیرامیٹر درختوں کی تعداد پر سیٹ کیا جاتا ہے۔

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ایک ڈیٹاسیٹ پر درجہ بندی کنندہ کو فٹ کرتا ہے اور پھر اسی ڈیٹاسیٹ پر اس درجہ بندی کنندہ کی کاپیاں فٹ کرتا ہے۔ یہ غلط طور پر درجہ بند اشیاء کے وزن پر توجہ مرکوز کرتا ہے اور اگلے درجہ بندی کنندہ کے لیے فٹ کو ایڈجسٹ کرتا ہے تاکہ درستگی ہو۔

---

## 🚀چیلنج

ان میں سے ہر تکنیک کے پاس بہت سے پیرامیٹرز ہیں جنہیں آپ ایڈجسٹ کر سکتے ہیں۔ ہر ایک کے ڈیفالٹ پیرامیٹرز پر تحقیق کریں اور سوچیں کہ ان پیرامیٹرز کو ایڈجسٹ کرنے کا ماڈل کے معیار پر کیا اثر پڑے گا۔

## [لیکچر کے بعد کا کوئز](https://ff-quizzes.netlify.app/en/ml/)

## جائزہ اور خود مطالعہ

ان اسباق میں بہت زیادہ اصطلاحات ہیں، اس لیے ایک لمحہ نکال کر [اس فہرست](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) کا جائزہ لیں جو مفید اصطلاحات پر مشتمل ہے!

## اسائنمنٹ

[پیرامیٹر پلے](assignment.md)

---

**ڈسکلیمر**:  
یہ دستاویز AI ترجمہ سروس [Co-op Translator](https://github.com/Azure/co-op-translator) کا استعمال کرتے ہوئے ترجمہ کی گئی ہے۔ ہم درستگی کے لیے کوشش کرتے ہیں، لیکن براہ کرم آگاہ رہیں کہ خودکار ترجمے میں غلطیاں یا غیر درستیاں ہو سکتی ہیں۔ اصل دستاویز کو اس کی اصل زبان میں مستند ذریعہ سمجھا جانا چاہیے۔ اہم معلومات کے لیے، پیشہ ور انسانی ترجمہ کی سفارش کی جاتی ہے۔ ہم اس ترجمے کے استعمال سے پیدا ہونے والی کسی بھی غلط فہمی یا غلط تشریح کے ذمہ دار نہیں ہیں۔