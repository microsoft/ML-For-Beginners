<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-05T13:07:34+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "my"
}
-->
# Cuisine classifiers 1

ဒီသင်ခန်းစာမှာ မိမိအရင်သင်ခန်းစာမှာ သိမ်းဆည်းထားတဲ့ အစားအစာအမျိုးအစားများနှင့်ပတ်သက်သော အချက်အလက်များကို အသုံးပြုပါမည်။

ဒီအချက်အလက်များကို အသုံးပြုပြီး _ပစ္စည်းများအုပ်စုအပေါ်မူတည်၍ အမျိုးသားအစားအစာကို ခန့်မှန်းနိုင်ရန်_ အမျိုးမျိုးသော classifiers များကို အသုံးပြုပါမည်။ ဒီလုပ်ငန်းစဉ်တွင် classification tasks များအတွက် algorithm များကို ဘယ်လိုအသုံးချနိုင်သည်ဆိုတာကို ပိုမိုလေ့လာနိုင်ပါမည်။

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# ပြင်ဆင်မှု

[Lesson 1](../1-Introduction/README.md) ကို ပြီးမြောက်ထားသည်ဟု ခန့်မှန်းပါက, _cleaned_cuisines.csv_ ဖိုင်ကို `/data` folder ရဲ့ root မှာ ရှိနေဖို့ သေချာပါစေ။

## လေ့ကျင့်ခန်း - အမျိုးသားအစားအစာကို ခန့်မှန်းပါ

1. ဒီသင်ခန်းစာရဲ့ _notebook.ipynb_ folder မှာ Pandas library နဲ့အတူ ဖိုင်ကို import လုပ်ပါ:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    အချက်အလက်တွေက ဒီလိုပုံစံနဲ့ ရှိပါမယ်:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. အခြား libraries များကို import လုပ်ပါ:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X နဲ့ y coordinates ကို training အတွက် dataframes နှစ်ခုအဖြစ် ခွဲပါ။ `cuisine` ကို labels dataframe အဖြစ် သတ်မှတ်ပါ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ဒါဟာ ဒီလိုပုံစံနဲ့ ရှိပါမယ်:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` column နဲ့ `cuisine` column ကို `drop()` ကိုခေါ်ပြီး ဖယ်ရှားပါ။ ကျန်တဲ့ data ကို trainable features အဖြစ် သိမ်းဆည်းပါ:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Features တွေက ဒီလိုပုံစံနဲ့ ရှိပါမယ်:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

အခုတော့ မိမိရဲ့ model ကို training လုပ်ဖို့ အဆင်သင့်ဖြစ်ပါပြီ!

## Classifier ရွေးချယ်ခြင်း

အချက်အလက်တွေကို training လုပ်ဖို့ အဆင်သင့်ဖြစ်ပြီးနောက်, မိမိလုပ်ငန်းစဉ်အတွက် algorithm ကို ရွေးချယ်ရပါမည်။

Scikit-learn က classification ကို Supervised Learning အဖြစ် grouping လုပ်ပြီး, အဲဒီ category မှာ classification လုပ်နိုင်တဲ့ နည်းလမ်းများစွာကို တွေ့နိုင်ပါမည်။ [အမျိုးမျိုးသောနည်းလမ်းများ](https://scikit-learn.org/stable/supervised_learning.html) ကို ပထမဆုံးကြည့်တဲ့အခါ အတော်လေးရှုပ်ထွေးစေပါသည်။ အောက်ပါနည်းလမ်းများသည် classification techniques များပါဝင်သည်:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

> [neural networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) ကို data classification အတွက် အသုံးပြုနိုင်သော်လည်း, ဒီသင်ခန်းစာရဲ့ အကျိုးအာနိသင်အတွင်းမှာ မပါဝင်ပါ။

### ဘယ် classifier ကို သုံးမလဲ?

ဒါဆို, ဘယ် classifier ကို သုံးသင့်သလဲ? အများအားဖြင့်, အမျိုးမျိုးကို run လုပ်ပြီး ရလဒ်ကောင်းတစ်ခုကို ရှာဖွေခြင်းသည် စမ်းသပ်မှုတစ်ခုဖြစ်သည်။ Scikit-learn က [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) ကို dataset တစ်ခုမှာ ပြုလုပ်ပြီး, KNeighbors, SVC နှစ်မျိုး, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB နဲ့ QuadraticDiscrinationAnalysis တို့ကို visualized ရလဒ်များဖြင့် နှိုင်းယှဉ်ပြထားသည်:

![comparison of classifiers](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Scikit-learn ရဲ့ documentation မှ Plots

> AutoML က ဒီပြဿနာကို cloud မှာ comparison များ run လုပ်ခြင်းဖြင့် အလွယ်တကူ ဖြေရှင်းပေးပြီး, မိမိ data အတွက် algorithm အကောင်းဆုံးကို ရွေးချယ်နိုင်စေသည်။ [ဒီမှာ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) စမ်းသပ်ကြည့်ပါ။

### ပိုမိုကောင်းမွန်တဲ့နည်းလမ်း

အကြံပေးမှုများကို လိုက်နာခြင်းသည် wild guessing လုပ်ခြင်းထက် ပိုမိုကောင်းမွန်ပါသည်။ [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ကို download လုပ်ပြီး, multiclass problem အတွက် ရွေးချယ်စရာများကို ရှာဖွေကြည့်ပါ:

![cheatsheet for multiclass problems](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Microsoft's Algorithm Cheat Sheet ရဲ့ multiclass classification options ကို ဖော်ပြထားသော အပိုင်း

✅ Cheat sheet ကို download လုပ်ပြီး, print ထုတ်ထားပြီး မိမိရဲ့ နံရံမှာ တင်ထားပါ!

### Reasoning

မိမိရဲ့ constraints တွေကို အခြေခံပြီး အမျိုးမျိုးသောနည်းလမ်းများကို reasoning လုပ်ကြည့်ပါ:

- **Neural networks မလိုအပ်ပါ**။ မိမိရဲ့ clean ဖြစ်တဲ့ dataset နဲ့ local notebooks မှာ training လုပ်နေတဲ့အခြေအနေကြောင့်, neural networks က ဒီ task အတွက် မလိုအပ်ပါ။
- **Two-class classifier မလိုအပ်ပါ**။ Two-class classifier ကို မသုံးသင့်ပါ, ဒါကြောင့် one-vs-all ကို မသုံးပါ။
- **Decision tree သို့မဟုတ် logistic regression သုံးနိုင်ပါ**။ Decision tree သို့မဟုတ် logistic regression ကို multiclass data အတွက် အသုံးပြုနိုင်ပါသည်။
- **Multiclass Boosted Decision Trees က အခြားပြဿနာကို ဖြေရှင်းသည်**။ Multiclass boosted decision tree က nonparametric tasks များအတွက် အကောင်းဆုံးဖြစ်ပြီး, ranking များကို ဖန်တီးရန်အတွက် အသုံးပြုသည်, ဒါကြောင့် ဒီ task အတွက် မသင့်တော်ပါ။

### Scikit-learn ကို အသုံးပြုခြင်း

Scikit-learn ကို အသုံးပြုပြီး data ကို analysis လုပ်ပါမည်။ သို့သော်, Scikit-learn မှာ logistic regression ကို အသုံးပြုနိုင်တဲ့ နည်းလမ်းများစွာ ရှိပါသည်။ [parameters to pass](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) ကို ကြည့်ပါ။

အဓိကအားဖြင့် `multi_class` နဲ့ `solver` ဆိုတဲ့ parameters နှစ်ခုကို သတ်မှတ်ရန်လိုအပ်ပါသည်။ Scikit-learn ကို logistic regression လုပ်ရန် တောင်းဆိုတဲ့အခါ, `multi_class` value က behavior တစ်ခုကို သတ်မှတ်ပေးပါသည်။ Solver ရဲ့ value က algorithm ကို သတ်မှတ်ပေးပါသည်။ Solvers အားလုံးကို `multi_class` values အားလုံးနဲ့ pair လုပ်လို့ မရပါ။

Docs အရ, multiclass case မှာ training algorithm:

- **one-vs-rest (OvR) scheme ကို အသုံးပြုသည်**, အကယ်၍ `multi_class` option ကို `ovr` အဖြစ် သတ်မှတ်ထားပါက
- **cross-entropy loss ကို အသုံးပြုသည်**, အကယ်၍ `multi_class` option ကို `multinomial` အဖြစ် သတ်မှတ်ထားပါက။ (လက်ရှိမှာ `multinomial` option ကို ‘lbfgs’, ‘sag’, ‘saga’ နဲ့ ‘newton-cg’ solvers တွေကသာ support လုပ်ပါသည်။)

> 🎓 'scheme' ဆိုတာက 'ovr' (one-vs-rest) သို့မဟုတ် 'multinomial' ဖြစ်နိုင်ပါသည်။ Logistic regression ကို binary classification အတွက် design လုပ်ထားသော်လည်း, ဒီ schemes တွေက multiclass classification tasks များကို ပိုမိုကောင်းမွန်စွာ handle လုပ်နိုင်စေပါသည်။ [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' ကို "optimization problem အတွက် အသုံးပြုမည့် algorithm" အဖြစ် သတ်မှတ်ထားသည်။ [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn က solvers တွေက data structure မျိုးစုံရဲ့ ပြဿနာများကို handle လုပ်ပုံကို ရှင်းပြထားသော table ကို ပေးထားပါသည်:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## လေ့ကျင့်ခန်း - data ကို ခွဲခြားပါ

မိမိအရင်သင်ခန်းစာမှာ logistic regression ကို လေ့လာထားပြီးဖြစ်သောကြောင့်, training trial အတွက် logistic regression ကို အဓိကထားပါမည်။
`train_test_split()` ကို ခေါ်ပြီး data ကို training group နဲ့ testing group အဖြစ် ခွဲပါ:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## လေ့ကျင့်ခန်း - logistic regression ကို အသုံးပြုပါ

Multiclass case ကို အသုံးပြုနေသောကြောင့်, မိမိသုံးမည့် _scheme_ နဲ့ _solver_ ကို ရွေးချယ်ရန်လိုအပ်ပါသည်။ LogisticRegression ကို multi_class setting နဲ့ **liblinear** solver ကို သတ်မှတ်ပြီး training လုပ်ပါ။

1. multi_class ကို `ovr` အဖြစ် သတ်မှတ်ပြီး solver ကို `liblinear` အဖြစ် သတ်မှတ်ထားသော logistic regression ကို ဖန်တီးပါ:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs` က default အဖြစ် သတ်မှတ်ထားသော solver ဖြစ်ပြီး, အဲဒီ solver ကို စမ်းသပ်ကြည့်ပါ
> မှတ်ချက်၊ သင့်ဒေတာကိုလိုအပ်သောအခါချောမောစေရန် Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function ကိုအသုံးပြုပါ။
အတိအကျမှုသည် **80%** အထက်မှာကောင်းမွန်ပါသည်။

1. ဒီမော်ဒယ်ကို စမ်းသပ်ရန်အတွက် ဒေတာတစ်တန်း (#50) ကို စမ်းကြည့်နိုင်သည်-

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ရလဒ်ကို ပုံနှိပ်ထားသည်-

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ အခြားတန်းနံပါတ်ကို စမ်းကြည့်ပြီး ရလဒ်များကို စစ်ဆေးပါ။

1. နက်နက်ရှိုင်းရှိုင်း စစ်ဆေးလိုပါက ဒီခန့်မှန်းမှု၏ အတိအကျမှုကို စစ်ဆေးနိုင်သည်-

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ရလဒ်ကို ပုံနှိပ်ထားသည် - အိန္ဒိယအစားအစာဟာ အကောင်းဆုံးခန့်မှန်းချက်ဖြစ်ပြီး အလားအလာကောင်းများပါသည်-

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ မော်ဒယ်က အိန္ဒိယအစားအစာဖြစ်တယ်လို့ အတော်လေးသေချာနေတာကို ရှင်းပြနိုင်ပါသလား?

1. Regression သင်ခန်းစာများတွင် လုပ်ခဲ့သလို Classification Report ကို ပုံနှိပ်ပြီး အသေးစိတ်ကို ရယူပါ-

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

## 🚀စိန်ခေါ်မှု

ဒီသင်ခန်းစာမှာ သင်ရဲ့ သန့်စင်ထားသော ဒေတာကို အသုံးပြုပြီး အစားအစာအမျိုးအစားကို ခန့်မှန်းနိုင်တဲ့ Machine Learning မော်ဒယ်တစ်ခုကို တည်ဆောက်ခဲ့ပါတယ်။ Scikit-learn မှာ ဒေတာကို ခွဲခြားဖို့ ရရှိနိုင်တဲ့ အမျိုးမျိုးသော ရွေးချယ်မှုများကို ဖတ်ရှုရန် အချိန်ယူပါ။ 'solver' ဆိုတဲ့ အယူအဆကို နက်နက်ရှိုင်းရှိုင်း လေ့လာပြီး မော်ဒယ်အတွင်းမှာ ဘာတွေဖြစ်ပျက်နေလဲဆိုတာကို နားလည်ပါ။

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## ပြန်လည်သုံးသပ်ခြင်းနှင့် ကိုယ်တိုင်လေ့လာခြင်း

Logistic Regression ရဲ့ သင်္ချာဆိုင်ရာ အခြေခံကို နက်နက်ရှိုင်းရှိုင်း လေ့လာရန် [ဒီသင်ခန်းစာ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) ကို ဖတ်ရှုပါ။
## အလုပ်ပေးစာ

[Solvers ကို လေ့လာပါ](assignment.md)

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေပါသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်မှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို ကျေးဇူးပြု၍ သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်ရန် လိုအပ်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူကောင်းမွန်သော ပရော်ဖက်ရှင်နယ်ဘာသာပြန်ဝန်ဆောင်မှုကို အကြံပြုပါသည်။ ဤဘာသာပြန်မှုကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားများ သို့မဟုတ် အဓိပ္ပာယ်မှားများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။