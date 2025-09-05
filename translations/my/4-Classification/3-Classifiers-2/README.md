<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T13:15:21+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "my"
}
-->
# Cuisine classifiers 2

ဒီ classification သင်ခန်းစာအပိုင်း (၂) မှာ သင် numeric data ကို classify လုပ်နိုင်တဲ့ နည်းလမ်းတွေကို ပိုမိုလေ့လာနိုင်မှာဖြစ်ပါတယ်။ အပြင်မှာ classifier တစ်ခုကို ရွေးချယ်တဲ့အခါမှာ ဖြစ်နိုင်တဲ့ အကျိုးဆက်တွေကိုလည်း သင်လေ့လာနိုင်ပါမယ်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### ကြိုတင်လိုအပ်ချက်

သင်ဟာ အရင် lessons တွေကိုပြီးစီးထားပြီး `data` folder ထဲမှာ _cleaned_cuisines.csv_ ဆိုတဲ့ cleaned dataset ကိုရှိထားတယ်လို့ ယူဆပါတယ်။ ဒီ dataset ဟာ lessons (၄) ခုပါဝင်တဲ့ folder ရဲ့ root မှာရှိပါတယ်။

### ပြင်ဆင်မှု

သင့် _notebook.ipynb_ ဖိုင်ကို cleaned dataset နဲ့ loaded လုပ်ပြီး model တည်ဆောက်မှုအတွက် X နဲ့ y dataframes အဖြစ် ခွဲထားပါတယ်။

## Classification map

အရင် lessons မှာ Microsoft ရဲ့ cheat sheet ကို အသုံးပြုပြီး data ကို classify လုပ်နိုင်တဲ့ နည်းလမ်းတွေကို သင်လေ့လာခဲ့ပါတယ်။ Scikit-learn မှာ cheat sheet တစ်ခုလည်းရှိပြီး၊ အဲဒီ cheat sheet က သင့် estimators (classifier ကိုခေါ်တဲ့အခြားအမည်) ကို ပိုမိုကျဉ်းကျဉ်းစစ်နိုင်အောင် ကူညီပေးနိုင်ပါတယ်။

![ML Map from Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [visit this map online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) နှင့် path ကို click လုပ်ပြီး documentation ကိုဖတ်ပါ။

### အစီအစဉ်

ဒီ map ဟာ သင့် data ကို ရှင်းလင်းစွာနားလည်ပြီးတဲ့အခါမှာ အလွန်အသုံးဝင်ပါတယ်။ သင့် data ကို 'လမ်း' တစ်ခုလိုဖြတ်သန်းပြီး ဆုံးဖြတ်ချက်ကို ရယူနိုင်ပါတယ်။

- ကျွန်တော်တို့မှာ >50 samples ရှိပါတယ်
- ကျွန်တော်တို့ category တစ်ခုကို predict လုပ်ချင်ပါတယ်
- ကျွန်တော်တို့မှာ labeled data ရှိပါတယ်
- ကျွန်တော်တို့မှာ 100K samples ထက်နည်းပါတယ်
- ✨ Linear SVC ကို ရွေးချယ်နိုင်ပါတယ်
- အဲဒါမအောင်မြင်ရင်၊ ကျွန်တော်တို့မှာ numeric data ရှိတဲ့အတွက်
    - ✨ KNeighbors Classifier ကို စမ်းကြည့်နိုင်ပါတယ်
      - အဲဒါမအောင်မြင်ရင် ✨ SVC နဲ့ ✨ Ensemble Classifiers ကို စမ်းကြည့်ပါ

ဒီလမ်းကြောင်းဟာ အလွန်အသုံးဝင်ပါတယ်။

## လေ့ကျင့်မှု - data ကို ခွဲခြားပါ

ဒီလမ်းကြောင်းကိုလိုက်ပြီး အသုံးပြုရန် libraries တချို့ကို import လုပ်ရပါမယ်။

1. လိုအပ်တဲ့ libraries တွေကို Import လုပ်ပါ:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. သင့် training data နဲ့ test data ကို ခွဲခြားပါ:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector clustering (SVC) ဟာ Support-Vector machines ML techniques မျိုးရိုးရဲ့ အစိတ်အပိုင်းတစ်ခုဖြစ်ပါတယ် (အောက်မှာပိုမိုလေ့လာနိုင်ပါတယ်)။ ဒီနည်းလမ်းမှာ label တွေကို cluster လုပ်ဖို့ 'kernel' ကိုရွေးချယ်နိုင်ပါတယ်။ 'C' parameter ဟာ 'regularization' ကိုဆိုလိုပြီး parameters တွေ၏ အကျိုးသက်ရောက်မှုကို ထိန်းညှိပေးပါတယ်။ Kernel ဟာ [အမျိုးမျိုး](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) ရှိနိုင်ပြီး၊ ဒီမှာ 'linear' ကို set လုပ်ထားပါတယ်။ Probability ဟာ 'false' အဖြစ် default ဖြစ်ပြီး၊ ဒီမှာ 'true' ကို set လုပ်ထားပါတယ်။ Random state ကို '0' အဖြစ် set လုပ်ထားပြီး data ကို shuffle လုပ်ကာ probabilities ရယူထားပါတယ်။

### လေ့ကျင့်မှု - Linear SVC ကို အသုံးပြုပါ

Classifiers array တစ်ခုကို စတင်ဖန်တီးပါ။ စမ်းသပ်မှုအတိုင်း ဒီ array ကို တိုးချဲ့သွားပါမယ်။

1. Linear SVC ကို စတင်ပါ:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC ကို အသုံးပြုပြီး model ကို train လုပ်ပြီး report ကို print ထုတ်ပါ:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    ရလဒ်က အတော်လည်းကောင်းပါတယ်:

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

## K-Neighbors classifier

K-Neighbors ဟာ ML methods "neighbors" မျိုးရိုးရဲ့ အစိတ်အပိုင်းတစ်ခုဖြစ်ပြီး supervised နဲ့ unsupervised learning နှစ်မျိုးလုံးမှာ အသုံးပြုနိုင်ပါတယ်။ ဒီနည်းလမ်းမှာ point အရေအတွက်ကို predefined လုပ်ပြီး၊ generalized labels တွေကို predict လုပ်နိုင်ဖို့ data တွေကို အဲဒီ point တွေကို စုစည်းထားပါတယ်။

### လေ့ကျင့်မှု - K-Neighbors classifier ကို အသုံးပြုပါ

အရင် classifier က data နဲ့ အတော်လည်းကောင်းပါတယ်၊ ဒါပေမယ့် accuracy ပိုမိုကောင်းနိုင်မလား စမ်းကြည့်ပါ။ K-Neighbors classifier ကို စမ်းသပ်ပါ။

1. Linear SVC item အပြီးမှာ comma ထည့်ပြီး classifier array ကို တိုးချဲ့ပါ:

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    ရလဒ်က နည်းနည်းပိုမိုဆိုးပါတယ်:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) ကိုလေ့လာပါ

## Support Vector Classifier

Support-Vector classifiers ဟာ [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) မျိုးရိုးရဲ့ အစိတ်အပိုင်းတစ်ခုဖြစ်ပြီး classification နဲ့ regression tasks တွေမှာ အသုံးပြုနိုင်ပါတယ်။ SVMs ဟာ "training examples တွေကို အာကာသထဲမှာ point အဖြစ် map လုပ်ပြီး" category နှစ်ခုကြားအကွာအဝေးကို maximize လုပ်ပါတယ်။ နောက်ထပ် data တွေကို အဲဒီအာကာသထဲ map လုပ်ပြီး category ကို predict လုပ်နိုင်ပါတယ်။

### လေ့ကျင့်မှု - Support Vector Classifier ကို အသုံးပြုပါ

Support Vector Classifier ကို အသုံးပြုပြီး accuracy ပိုမိုကောင်းနိုင်မလား စမ်းကြည့်ပါ။

1. K-Neighbors item အပြီးမှာ comma ထည့်ပြီး ဒီ line ကို ထည့်ပါ:

    ```python
    'SVC': SVC(),
    ```

    ရလဒ်က အတော်လည်းကောင်းပါတယ်!

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

    ✅ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) ကိုလေ့လာပါ

## Ensemble Classifiers

အရင် test က အတော်လည်းကောင်းပါတယ်၊ ဒါပေမယ့် လမ်းကြောင်းကို အဆုံးထိလိုက်ကြည့်ပါ။ 'Ensemble Classifiers' ကို စမ်းသပ်ကြည့်ပါ၊ အထူးသဖြင့် Random Forest နဲ့ AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

ရလဒ်က အတော်လည်းကောင်းပါတယ်၊ အထူးသဖြင့် Random Forest:

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

✅ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html) ကိုလေ့လာပါ

Machine Learning ရဲ့ ဒီနည်းလမ်းဟာ "base estimators အများအပြားရဲ့ prediction တွေကို ပေါင်းစပ်ပြီး" model quality ကို တိုးတက်စေပါတယ်။ ကျွန်တော်တို့ရဲ့ ဥပမာမှာ Random Trees နဲ့ AdaBoost ကို အသုံးပြုထားပါတယ်။

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) ဟာ averaging method ဖြစ်ပြီး၊ 'decision trees' တွေကို 'forest' တစ်ခုအဖြစ် တည်ဆောက်ကာ randomness ကို ထည့်သွင်းပြီး overfitting ကိုရှောင်ရှားပါတယ်။ n_estimators parameter ကို trees အရေအတွက်အဖြစ် set လုပ်ထားပါတယ်။

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ဟာ dataset ကို classifier တစ်ခုနဲ့ fit လုပ်ပြီး၊ အဲဒီ classifier ကို dataset အပေါ်မှာ ထပ်မံ fit လုပ်ပါတယ်။ အမှားဖြစ်တဲ့ items တွေ၏ weight ကို အာရုံစိုက်ပြီး၊ နောက်ထပ် classifier အတွက် fit ကို ပြင်ဆင်ပေးပါတယ်။

---

## 🚀Challenge

ဒီ techniques တစ်ခုချင်းစီမှာ parameters အများအပြားရှိပြီး၊ default parameters တွေကို tweak လုပ်နိုင်ပါတယ်။ တစ်ခုချင်းစီရဲ့ default parameters တွေကို လေ့လာပြီး၊ parameters တွေကို tweak လုပ်ရင် model quality အပေါ် ဘယ်လိုသက်ရောက်မှုရှိမလဲ စဉ်းစားပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

ဒီ lessons တွေမှာ jargon အများကြီးပါဝင်ပါတယ်၊ [ဒီစာရင်း](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ကိုကြည့်ပြီး terminology အသုံးဝင်တဲ့အရာတွေကို ပြန်လည်သုံးသပ်ပါ။

## Assignment 

[Parameter play](assignment.md)

---

**အကြောင်းကြားချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက် ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မတိကျမှုများ ပါရှိနိုင်သည်ကို သတိပြုပါ။ မူရင်းဘာသာစကားဖြင့် ရေးသားထားသော စာရွက်စာတမ်းကို အာဏာရှိသော ရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူ့ဘာသာပြန်ပညာရှင်များမှ ပရော်ဖက်ရှင်နယ် ဘာသာပြန်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော အလွဲအမှားများ သို့မဟုတ် အနားလွဲမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။