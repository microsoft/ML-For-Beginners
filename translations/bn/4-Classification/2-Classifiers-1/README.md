<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-04T21:10:04+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "bn"
}
-->
# কুইজিন শ্রেণীবিন্যাসকারী ১

এই পাঠে, আপনি আগের পাঠে সংরক্ষিত ডেটাসেট ব্যবহার করবেন, যা বিভিন্ন কুইজিন সম্পর্কিত ভারসাম্যপূর্ণ এবং পরিষ্কার ডেটা নিয়ে গঠিত।

আপনি এই ডেটাসেটটি বিভিন্ন শ্রেণীবিন্যাসকারী ব্যবহার করে _উপাদানগুলোর একটি গ্রুপের উপর ভিত্তি করে একটি নির্দিষ্ট জাতীয় কুইজিন পূর্বানুমান করতে_ ব্যবহার করবেন। এটি করতে গিয়ে, আপনি শ্রেণীবিন্যাস কাজের জন্য অ্যালগরিদম কীভাবে ব্যবহার করা যায় সে সম্পর্কে আরও জানতে পারবেন।

## [পাঠ-পূর্ব কুইজ](https://ff-quizzes.netlify.app/en/ml/)
# প্রস্তুতি

ধরে নেওয়া হচ্ছে আপনি [Lesson 1](../1-Introduction/README.md) সম্পন্ন করেছেন, নিশ্চিত করুন যে একটি _cleaned_cuisines.csv_ ফাইল `/data` ফোল্ডারের মূল অংশে এই চারটি পাঠের জন্য বিদ্যমান।

## অনুশীলন - একটি জাতীয় কুইজিন পূর্বানুমান করুন

1. এই পাঠের _notebook.ipynb_ ফোল্ডারে কাজ করে, সেই ফাইলটি এবং Pandas লাইব্রেরি আমদানি করুন:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ডেটা দেখতে এরকম:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. এখন আরও কিছু লাইব্রেরি আমদানি করুন:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X এবং y কোঅর্ডিনেটগুলোকে দুটি ডেটাফ্রেমে ভাগ করুন। `cuisine` লেবেল ডেটাফ্রেম হতে পারে:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    এটি দেখতে এরকম হবে:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` কলাম এবং `cuisine` কলামটি `drop()` কল করে বাদ দিন। বাকি ডেটা প্রশিক্ষণযোগ্য বৈশিষ্ট্য হিসেবে সংরক্ষণ করুন:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    আপনার বৈশিষ্ট্যগুলো দেখতে এরকম:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

এখন আপনি আপনার মডেল প্রশিক্ষণের জন্য প্রস্তুত!

## আপনার শ্রেণীবিন্যাসকারী নির্বাচন করা

আপনার ডেটা পরিষ্কার এবং প্রশিক্ষণের জন্য প্রস্তুত হওয়ার পরে, আপনাকে কাজের জন্য কোন অ্যালগরিদম ব্যবহার করবেন তা সিদ্ধান্ত নিতে হবে।

Scikit-learn শ্রেণীবিন্যাসকে Supervised Learning এর অধীনে গ্রুপ করে এবং সেই বিভাগে আপনি শ্রেণীবিন্যাস করার অনেক উপায় খুঁজে পাবেন। [বিভিন্ন পদ্ধতি](https://scikit-learn.org/stable/supervised_learning.html) প্রথম দেখায় বেশ বিভ্রান্তিকর মনে হতে পারে। নিম্নলিখিত পদ্ধতিগুলোতে শ্রেণীবিন্যাস কৌশল অন্তর্ভুক্ত রয়েছে:

- লিনিয়ার মডেল
- সাপোর্ট ভেক্টর মেশিন
- স্টোকাস্টিক গ্রেডিয়েন্ট ডিসেন্ট
- নিকটতম প্রতিবেশী
- গাউসিয়ান প্রসেস
- সিদ্ধান্ত গাছ
- এনসেম্বল পদ্ধতি (ভোটিং শ্রেণীবিন্যাসকারী)
- মাল্টিক্লাস এবং মাল্টি-আউটপুট অ্যালগরিদম (মাল্টিক্লাস এবং মাল্টিলেবেল শ্রেণীবিন্যাস, মাল্টিক্লাস-মাল্টি-আউটপুট শ্রেণীবিন্যাস)

> আপনি [নিউরাল নেটওয়ার্ক ব্যবহার করে ডেটা শ্রেণীবিন্যাস করতে পারেন](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), তবে এটি এই পাঠের পরিধির বাইরে।

### কোন শ্রেণীবিন্যাসকারী ব্যবহার করবেন?

তাহলে, কোন শ্রেণীবিন্যাসকারী বেছে নেবেন? প্রায়শই, কয়েকটি চেষ্টা করে এবং একটি ভালো ফলাফল খুঁজে বের করা একটি পরীক্ষার উপায়। Scikit-learn একটি [পাশাপাশি তুলনা](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) অফার করে একটি তৈরি করা ডেটাসেটে, KNeighbors, SVC দুটি উপায়ে, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB এবং QuadraticDiscrinationAnalysis এর তুলনা দেখায়, ফলাফলগুলো চিত্রিত করে:

![শ্রেণীবিন্যাসকারীদের তুলনা](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Plots Scikit-learn এর ডকুমেন্টেশনে তৈরি করা হয়েছে

> AutoML এই সমস্যাটি সুন্দরভাবে সমাধান করে ক্লাউডে এই তুলনাগুলো চালিয়ে, আপনাকে আপনার ডেটার জন্য সেরা অ্যালগরিদম বেছে নিতে দেয়। এটি চেষ্টা করুন [এখানে](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### একটি ভালো পদ্ধতি

অন্ধভাবে অনুমান করার চেয়ে একটি ভালো উপায় হলো এই ডাউনলোডযোগ্য [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) অনুসরণ করা। এখানে, আমরা আবিষ্কার করি যে আমাদের মাল্টিক্লাস সমস্যার জন্য আমাদের কিছু বিকল্প রয়েছে:

![মাল্টিক্লাস সমস্যার জন্য চিটশিট](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Microsoft's Algorithm Cheat Sheet এর একটি অংশ, মাল্টিক্লাস শ্রেণীবিন্যাস বিকল্পগুলো বিস্তারিতভাবে দেখানো হয়েছে

✅ এই চিটশিটটি ডাউনলোড করুন, প্রিন্ট করুন এবং আপনার দেয়ালে ঝুলিয়ে রাখুন!

### যুক্তি

চলুন দেখি আমরা বিভিন্ন পদ্ধতির মধ্য দিয়ে যুক্তি দিয়ে এগিয়ে যেতে পারি কিনা, আমাদের সীমাবদ্ধতাগুলো বিবেচনা করে:

- **নিউরাল নেটওয়ার্ক খুব ভারী**। আমাদের পরিষ্কার কিন্তু সীমিত ডেটাসেট এবং আমরা স্থানীয়ভাবে নোটবুকের মাধ্যমে প্রশিক্ষণ চালাচ্ছি, নিউরাল নেটওয়ার্ক এই কাজের জন্য খুব ভারী।
- **দুই-শ্রেণীর শ্রেণীবিন্যাসকারী নয়**। আমরা দুই-শ্রেণীর শ্রেণীবিন্যাসকারী ব্যবহার করি না, তাই এটি one-vs-all কে বাদ দেয়।
- **সিদ্ধান্ত গাছ বা লজিস্টিক রিগ্রেশন কাজ করতে পারে**। একটি সিদ্ধান্ত গাছ কাজ করতে পারে, অথবা মাল্টিক্লাস ডেটার জন্য লজিস্টিক রিগ্রেশন।
- **মাল্টিক্লাস Boosted Decision Trees ভিন্ন সমস্যা সমাধান করে**। মাল্টিক্লাস Boosted Decision Tree অ-পরামিতিক কাজের জন্য সবচেয়ে উপযুক্ত, যেমন র‌্যাঙ্কিং তৈরি করার জন্য ডিজাইন করা কাজ, তাই এটি আমাদের জন্য উপযোগী নয়।

### Scikit-learn ব্যবহার করা

আমরা Scikit-learn ব্যবহার করে আমাদের ডেটা বিশ্লেষণ করব। তবে, Scikit-learn এ লজিস্টিক রিগ্রেশন ব্যবহার করার অনেক উপায় রয়েছে। পাস করার [প্যারামিটারগুলো দেখুন](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression)।  

মূলত দুটি গুরুত্বপূর্ণ প্যারামিটার রয়েছে - `multi_class` এবং `solver` - যা আমাদের নির্দিষ্ট করতে হবে, যখন আমরা Scikit-learn কে লজিস্টিক রিগ্রেশন করতে বলি। `multi_class` মান একটি নির্দিষ্ট আচরণ প্রয়োগ করে। solver এর মান হলো কোন অ্যালগরিদম ব্যবহার করা হবে। সব solver সব `multi_class` মানের সাথে যুক্ত করা যায় না।

ডক অনুযায়ী, মাল্টিক্লাস ক্ষেত্রে, প্রশিক্ষণ অ্যালগরিদম:

- **one-vs-rest (OvR) স্কিম ব্যবহার করে**, যদি `multi_class` অপশনটি `ovr` এ সেট করা হয়
- **ক্রস-এন্ট্রপি লস ব্যবহার করে**, যদি `multi_class` অপশনটি `multinomial` এ সেট করা হয়। (বর্তমানে `multinomial` অপশন শুধুমাত্র ‘lbfgs’, ‘sag’, ‘saga’ এবং ‘newton-cg’ solver দ্বারা সমর্থিত।)

> 🎓 এখানে 'scheme' হয় 'ovr' (one-vs-rest) অথবা 'multinomial' হতে পারে। যেহেতু লজিস্টিক রিগ্রেশন মূলত বাইনারি শ্রেণীবিন্যাসকে সমর্থন করার জন্য ডিজাইন করা হয়েছে, এই স্কিমগুলো এটিকে মাল্টিক্লাস শ্রেণীবিন্যাস কাজগুলো আরও ভালোভাবে পরিচালনা করতে সাহায্য করে। [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' সংজ্ঞায়িত করা হয়েছে "অপ্টিমাইজেশন সমস্যায় ব্যবহৃত অ্যালগরিদম" হিসেবে। [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn এই টেবিলটি অফার করে, যা ব্যাখ্যা করে কীভাবে solver বিভিন্ন চ্যালেঞ্জ মোকাবিলা করে, যা বিভিন্ন ধরনের ডেটা কাঠামো দ্বারা উপস্থাপিত হয়:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## অনুশীলন - ডেটা ভাগ করুন

আমরা আমাদের প্রথম প্রশিক্ষণ পরীক্ষার জন্য লজিস্টিক রিগ্রেশনকে কেন্দ্র করে কাজ করতে পারি, যেহেতু আপনি আগের পাঠে এটি সম্পর্কে শিখেছেন।
আপনার ডেটাকে প্রশিক্ষণ এবং পরীক্ষার গ্রুপে ভাগ করুন `train_test_split()` কল করে:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## অনুশীলন - লজিস্টিক রিগ্রেশন প্রয়োগ করুন

যেহেতু আপনি মাল্টিক্লাস ক্ষেত্রে কাজ করছেন, আপনাকে কোন _scheme_ ব্যবহার করবেন এবং কোন _solver_ সেট করবেন তা বেছে নিতে হবে। মাল্টিক্লাস সেটিং এবং **liblinear** solver দিয়ে LogisticRegression ব্যবহার করুন।

1. একটি লজিস্টিক রিগ্রেশন তৈরি করুন যেখানে multi_class `ovr` এ সেট করা এবং solver `liblinear` এ সেট করা:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ একটি ভিন্ন solver চেষ্টা করুন, যেমন `lbfgs`, যা প্রায়শই ডিফল্ট হিসেবে সেট করা হয়
পান্ডাসের [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ফাংশন ব্যবহার করুন আপনার ডেটা সমতল করতে যখন প্রয়োজন।
সঠিকতা **৮০%** এর বেশি ভালো!

1. আপনি একটি ডেটার সারি (#৫০) পরীক্ষা করে এই মডেলটি কার্যকর দেখতে পারেন:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ফলাফলটি প্রিন্ট করা হয়:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ একটি ভিন্ন সারি নম্বর চেষ্টা করুন এবং ফলাফল পরীক্ষা করুন

1. আরও গভীরে গিয়ে, আপনি এই পূর্বাভাসের সঠিকতা পরীক্ষা করতে পারেন:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ফলাফলটি প্রিন্ট করা হয় - ভারতীয় খাবার এটি সেরা অনুমান, ভালো সম্ভাবনার সাথে:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ আপনি কি ব্যাখ্যা করতে পারেন কেন মডেলটি নিশ্চিত যে এটি একটি ভারতীয় খাবার?

1. আরও বিস্তারিত জানার জন্য একটি শ্রেণীবিন্যাস রিপোর্ট প্রিন্ট করুন, যেমনটি আপনি রিগ্রেশন পাঠে করেছিলেন:

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

## 🚀চ্যালেঞ্জ

এই পাঠে, আপনি আপনার পরিষ্কার করা ডেটা ব্যবহার করে একটি মেশিন লার্নিং মডেল তৈরি করেছেন যা উপাদানগুলির একটি সিরিজের উপর ভিত্তি করে একটি জাতীয় খাবার পূর্বাভাস দিতে পারে। Scikit-learn এর বিভিন্ন ডেটা শ্রেণীবিন্যাসের বিকল্পগুলি সম্পর্কে আরও জানার জন্য কিছু সময় নিন। 'solver' ধারণাটি আরও গভীরভাবে বুঝুন যাতে এর পেছনের কার্যপ্রক্রিয়া বোঝা যায়।

## [পাঠ-পরবর্তী কুইজ](https://ff-quizzes.netlify.app/en/ml/)

## পর্যালোচনা ও স্ব-অধ্যয়ন

লজিস্টিক রিগ্রেশনের পেছনের গণিত সম্পর্কে আরও জানুন [এই পাঠে](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## অ্যাসাইনমেন্ট 

[solver সম্পর্কে অধ্যয়ন করুন](assignment.md)

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা দায়বদ্ধ থাকব না।