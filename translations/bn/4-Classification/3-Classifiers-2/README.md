<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-04T21:11:41+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "bn"
}
-->
# কুইজিন শ্রেণীবিভাজক ২

এই দ্বিতীয় শ্রেণীবিভাজন পাঠে, আপনি সংখ্যাসূচক ডেটা শ্রেণীবিভাজনের আরও পদ্ধতি অন্বেষণ করবেন। এছাড়াও, আপনি একটি শ্রেণীবিভাজক নির্বাচন করার প্রভাব সম্পর্কে শিখবেন।

## [পূর্ব-পাঠ কুইজ](https://ff-quizzes.netlify.app/en/ml/)

### পূর্বশর্ত

আমরা ধরে নিচ্ছি যে আপনি পূর্ববর্তী পাঠগুলি সম্পন্ন করেছেন এবং আপনার `data` ফোল্ডারে একটি পরিষ্কার ডেটাসেট রয়েছে যার নাম _cleaned_cuisines.csv_, যা এই ৪-পাঠের ফোল্ডারের মূল অংশে রয়েছে।

### প্রস্তুতি

আমরা আপনার _notebook.ipynb_ ফাইলটি পরিষ্কার ডেটাসেট দিয়ে লোড করেছি এবং এটি X এবং y ডেটাফ্রেমে ভাগ করেছি, যা মডেল তৈরির প্রক্রিয়ার জন্য প্রস্তুত।

## একটি শ্রেণীবিভাজন মানচিত্র

পূর্বে, আপনি মাইক্রোসফটের চিট শিট ব্যবহার করে ডেটা শ্রেণীবিভাজনের বিভিন্ন বিকল্প সম্পর্কে শিখেছেন। Scikit-learn একটি অনুরূপ, কিন্তু আরও বিস্তারিত চিট শিট অফার করে যা আপনার শ্রেণীবিভাজক নির্বাচনকে আরও সংকুচিত করতে সাহায্য করতে পারে:

![Scikit-learn থেকে ML মানচিত্র](../../../../4-Classification/3-Classifiers-2/images/map.png)
> টিপ: [এই মানচিত্রটি অনলাইনে দেখুন](https://scikit-learn.org/stable/tutorial/machine_learning_map/) এবং পথ ধরে ক্লিক করে ডকুমেন্টেশন পড়ুন।

### পরিকল্পনা

এই মানচিত্রটি আপনার ডেটা সম্পর্কে পরিষ্কার ধারণা থাকলে খুবই সহায়ক, কারণ আপনি এর পথ ধরে একটি সিদ্ধান্তে পৌঁছাতে পারেন:

- আমাদের কাছে >৫০ নমুনা রয়েছে
- আমরা একটি বিভাগ পূর্বাভাস দিতে চাই
- আমাদের লেবেলযুক্ত ডেটা রয়েছে
- আমাদের কাছে ১০০K-এর কম নমুনা রয়েছে
- ✨ আমরা একটি Linear SVC বেছে নিতে পারি
- যদি এটি কাজ না করে, যেহেতু আমাদের সংখ্যাসূচক ডেটা রয়েছে
    - আমরা ✨ KNeighbors Classifier চেষ্টা করতে পারি 
      - যদি এটি কাজ না করে, ✨ SVC এবং ✨ Ensemble Classifiers চেষ্টা করুন

এটি অনুসরণ করার জন্য একটি খুব সহায়ক পথ।

## অনুশীলন - ডেটা ভাগ করুন

এই পথ অনুসরণ করে, আমাদের প্রয়োজনীয় লাইব্রেরি আমদানি করে শুরু করা উচিত।

1. প্রয়োজনীয় লাইব্রেরি আমদানি করুন:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. আপনার প্রশিক্ষণ এবং পরীক্ষার ডেটা ভাগ করুন:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## লিনিয়ার SVC শ্রেণীবিভাজক

সাপোর্ট-ভেক্টর ক্লাস্টারিং (SVC) হল সাপোর্ট-ভেক্টর মেশিন পরিবারের একটি অংশ, যা মেশিন লার্নিংয়ের একটি কৌশল (নিচে আরও জানুন)। এই পদ্ধতিতে, আপনি একটি 'কর্নেল' নির্বাচন করতে পারেন যা লেবেলগুলিকে কীভাবে ক্লাস্টার করা হবে তা নির্ধারণ করে। 'C' প্যারামিটারটি 'নিয়ন্ত্রণ' নির্দেশ করে, যা প্যারামিটারগুলির প্রভাব নিয়ন্ত্রণ করে। কর্নেল [বিভিন্ন](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) হতে পারে; এখানে আমরা এটি 'লিনিয়ার' সেট করেছি যাতে আমরা লিনিয়ার SVC ব্যবহার করতে পারি। প্রোবাবিলিটি ডিফল্টভাবে 'ফলস'; এখানে আমরা এটি 'ট্রু' সেট করেছি যাতে সম্ভাবনার অনুমান সংগ্রহ করা যায়। আমরা র‍্যান্ডম স্টেট '0' সেট করেছি যাতে ডেটা শাফল করা যায় এবং সম্ভাবনা পাওয়া যায়।

### অনুশীলন - একটি লিনিয়ার SVC প্রয়োগ করুন

একটি শ্রেণীবিভাজকের অ্যারে তৈরি করে শুরু করুন। আমরা পরীক্ষা করার সময় এই অ্যারেতে ক্রমান্বয়ে যোগ করব।

1. একটি লিনিয়ার SVC দিয়ে শুরু করুন:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. লিনিয়ার SVC ব্যবহার করে আপনার মডেল প্রশিক্ষণ দিন এবং একটি রিপোর্ট প্রিন্ট করুন:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    ফলাফল বেশ ভালো:

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

## K-Neighbors শ্রেণীবিভাজক

K-Neighbors হল "পড়শি" পরিবারের অংশ, যা মেশিন লার্নিংয়ের পদ্ধতি এবং এটি সুপারভাইজড এবং আনসুপারভাইজড লার্নিং উভয়ের জন্য ব্যবহার করা যেতে পারে। এই পদ্ধতিতে, একটি পূর্বনির্ধারিত সংখ্যক পয়েন্ট তৈরি করা হয় এবং ডেটা এই পয়েন্টগুলির চারপাশে সংগ্রহ করা হয় যাতে ডেটার জন্য সাধারণ লেবেল পূর্বাভাস দেওয়া যায়।

### অনুশীলন - K-Neighbors শ্রেণীবিভাজক প্রয়োগ করুন

পূর্ববর্তী শ্রেণীবিভাজকটি ভালো ছিল এবং ডেটার সাথে ভালো কাজ করেছে, তবে হয়তো আমরা আরও ভালো নির্ভুলতা পেতে পারি। একটি K-Neighbors শ্রেণীবিভাজক চেষ্টা করুন।

1. আপনার শ্রেণীবিভাজক অ্যারেতে একটি লাইন যোগ করুন (লিনিয়ার SVC আইটেমের পরে একটি কমা যোগ করুন):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    ফলাফল একটু খারাপ:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) সম্পর্কে জানুন

## সাপোর্ট ভেক্টর শ্রেণীবিভাজক

সাপোর্ট-ভেক্টর শ্রেণীবিভাজক [সাপোর্ট-ভেক্টর মেশিন](https://wikipedia.org/wiki/Support-vector_machine) পরিবারের অংশ, যা শ্রেণীবিভাজন এবং রিগ্রেশন কাজের জন্য ব্যবহৃত হয়। SVMs "প্রশিক্ষণ উদাহরণগুলিকে স্থানগুলিতে পয়েন্টে ম্যাপ করে" যাতে দুটি বিভাগের মধ্যে দূরত্ব সর্বাধিক করা যায়। পরবর্তী ডেটা এই স্থানে ম্যাপ করা হয় যাতে তাদের বিভাগ পূর্বাভাস দেওয়া যায়।

### অনুশীলন - সাপোর্ট ভেক্টর শ্রেণীবিভাজক প্রয়োগ করুন

আরও ভালো নির্ভুলতার জন্য একটি সাপোর্ট ভেক্টর শ্রেণীবিভাজক চেষ্টা করুন।

1. K-Neighbors আইটেমের পরে একটি কমা যোগ করুন এবং তারপর এই লাইনটি যোগ করুন:

    ```python
    'SVC': SVC(),
    ```

    ফলাফল বেশ ভালো!

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

    ✅ [সাপোর্ট-ভেক্টর](https://scikit-learn.org/stable/modules/svm.html#svm) সম্পর্কে জানুন

## Ensemble শ্রেণীবিভাজক

পথের একেবারে শেষ পর্যন্ত অনুসরণ করি, যদিও পূর্ববর্তী পরীক্ষা বেশ ভালো ছিল। আসুন কিছু 'Ensemble শ্রেণীবিভাজক' চেষ্টা করি, বিশেষ করে Random Forest এবং AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

ফলাফল খুবই ভালো, বিশেষ করে Random Forest-এর জন্য:

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

✅ [Ensemble শ্রেণীবিভাজক](https://scikit-learn.org/stable/modules/ensemble.html) সম্পর্কে জানুন

মেশিন লার্নিংয়ের এই পদ্ধতি "কয়েকটি বেস এস্টিমেটরের পূর্বাভাসকে একত্রিত করে" মডেলের গুণমান উন্নত করে। আমাদের উদাহরণে, আমরা Random Trees এবং AdaBoost ব্যবহার করেছি।

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), একটি গড় পদ্ধতি, 'ডিসিশন ট্রি' এর একটি 'ফরেস্ট' তৈরি করে যা অতিরিক্ত ফিটিং এড়াতে র‍্যান্ডমনেস দিয়ে সংযোজিত হয়। n_estimators প্যারামিটারটি ট্রির সংখ্যায় সেট করা হয়।

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) একটি ডেটাসেটে একটি শ্রেণীবিভাজক ফিট করে এবং তারপর সেই শ্রেণীবিভাজকের কপি একই ডেটাসেটে ফিট করে। এটি ভুলভাবে শ্রেণীবদ্ধ আইটেমগুলির ওজনের উপর ফোকাস করে এবং পরবর্তী শ্রেণীবিভাজকের ফিট সামঞ্জস্য করে সেগুলি সংশোধন করে।

---

## 🚀চ্যালেঞ্জ

এই পদ্ধতিগুলির প্রতিটিতে অনেক সংখ্যক প্যারামিটার রয়েছে যা আপনি পরিবর্তন করতে পারেন। প্রতিটির ডিফল্ট প্যারামিটারগুলি গবেষণা করুন এবং ভাবুন এই প্যারামিটারগুলি পরিবর্তন করলে মডেলের গুণমানের জন্য কী অর্থ হতে পারে।

## [পাঠ-পরবর্তী কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা ও স্ব-অধ্যয়ন

এই পাঠগুলিতে অনেক জটিল শব্দ রয়েছে, তাই [এই তালিকা](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) পর্যালোচনা করতে এক মিনিট সময় নিন, যেখানে দরকারী পরিভাষা রয়েছে!

## অ্যাসাইনমেন্ট 

[প্যারামিটার নিয়ে খেলা](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসাধ্য সঠিকতা নিশ্চিত করার চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।