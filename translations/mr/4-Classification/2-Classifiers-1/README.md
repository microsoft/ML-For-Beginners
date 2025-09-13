<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T06:15:57+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "mr"
}
-->
# पाककृती वर्गीकरण 1

या धड्यात, तुम्ही मागील धड्यातून जतन केलेल्या डेटासेटचा वापर कराल, ज्यामध्ये विविध प्रकारच्या स्वयंपाकशैलींविषयी संतुलित आणि स्वच्छ डेटा आहे.

तुम्ही या डेटासेटचा वापर विविध वर्गीकरण करणाऱ्या अल्गोरिदमसह कराल, जे _साहित्यांच्या गटावर आधारित राष्ट्रीय स्वयंपाकशैलीचा अंदाज_ लावतील. असे करताना, तुम्ही वर्गीकरण कार्यांसाठी अल्गोरिदम कसे वापरले जाऊ शकतात याबद्दल अधिक जाणून घ्याल.

## [पूर्व-व्याख्यान प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/)
# तयारी

समजा तुम्ही [धडा 1](../1-Introduction/README.md) पूर्ण केला आहे, तर खात्री करा की _cleaned_cuisines.csv_ फाइल या चार धड्यांसाठी मूळ `/data` फोल्डरमध्ये आहे.

## व्यायाम - राष्ट्रीय स्वयंपाकशैलीचा अंदाज लावा

1. या धड्याच्या _notebook.ipynb_ फोल्डरमध्ये काम करताना, त्या फाइलसह Pandas लायब्ररी आयात करा:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डेटा असा दिसतो:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. आता आणखी काही लायब्ररी आयात करा:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X आणि y निर्देशांक दोन डेटाफ्रेममध्ये प्रशिक्षणासाठी विभाजित करा. `cuisine` हे लेबल्स डेटाफ्रेम असू शकते:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ते असे दिसेल:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` आणि `cuisine` कॉलम `drop()` वापरून काढा. उर्वरित डेटा प्रशिक्षणासाठी वैशिष्ट्ये म्हणून जतन करा:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    तुमची वैशिष्ट्ये अशी दिसतील:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

आता तुमचा मॉडेल प्रशिक्षणासाठी तयार आहे!

## तुमचा वर्गीकरणकर्ता निवडणे

आता तुमचा डेटा स्वच्छ आणि प्रशिक्षणासाठी तयार आहे, तुम्हाला कोणता अल्गोरिदम वापरायचा आहे ते ठरवावे लागेल.

Scikit-learn वर्गीकरणाला सुपरवाइज्ड लर्निंग अंतर्गत गटबद्ध करते, आणि त्या श्रेणीत तुम्हाला वर्गीकरणासाठी अनेक पद्धती सापडतील. [विविधता](https://scikit-learn.org/stable/supervised_learning.html) प्रथमदर्शनी थोडी गोंधळात टाकणारी वाटू शकते. खालील पद्धती वर्गीकरण तंत्रांचा समावेश करतात:

- रेषीय मॉडेल्स
- सपोर्ट व्हेक्टर मशीन
- स्टोकेस्टिक ग्रेडियंट डिसेंट
- जवळचे शेजारी
- गॉसियन प्रक्रिया
- निर्णय वृक्ष
- एन्सेम्बल पद्धती (मतदार वर्गीकरणकर्ता)
- मल्टीक्लास आणि मल्टीआउटपुट अल्गोरिदम (मल्टीक्लास आणि मल्टीलेबल वर्गीकरण, मल्टीक्लास-मल्टीआउटपुट वर्गीकरण)

> तुम्ही [डेटा वर्गीकृत करण्यासाठी न्यूरल नेटवर्क्सचा वापर](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) देखील करू शकता, परंतु ते या धड्याच्या कक्षेबाहेर आहे.

### कोणता वर्गीकरणकर्ता निवडायचा?

तर, तुम्ही कोणता वर्गीकरणकर्ता निवडावा? अनेक वेळा, काही वर्गीकरणकर्त्यांचा वापर करून चांगले परिणाम मिळतात का ते तपासणे हा एक मार्ग असतो. Scikit-learn [साइड-बाय-साइड तुलना](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) ऑफर करते, ज्यामध्ये KNeighbors, SVC दोन प्रकारे, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB आणि QuadraticDiscriminationAnalysis यांची तुलना केली जाते, आणि परिणाम व्हिज्युअल स्वरूपात दाखवले जातात:

![वर्गीकरणकर्त्यांची तुलना](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Scikit-learn च्या दस्तऐवजांवरून तयार केलेले प्लॉट्स

> AutoML हे काम सहज सोडवते, क्लाउडमध्ये या तुलना चालवून तुम्हाला तुमच्या डेटासाठी सर्वोत्तम अल्गोरिदम निवडण्याची परवानगी देते. [येथे प्रयत्न करा](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### एक चांगला दृष्टिकोन

अंधाधुंद अंदाज लावण्यापेक्षा, डाउनलोड करण्यायोग्य [ML चीट शीट](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) वर आधारित विचार करणे चांगले आहे. येथे, आम्हाला कळते की आमच्या मल्टीक्लास समस्येसाठी आमच्याकडे काही पर्याय आहेत:

![मल्टीक्लास समस्यांसाठी चीटशीट](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Microsoft's Algorithm Cheat Sheet चा एक भाग, ज्यामध्ये मल्टीक्लास वर्गीकरण पर्याय तपशीलवार दिले आहेत

✅ ही चीट शीट डाउनलोड करा, प्रिंट करा, आणि तुमच्या भिंतीवर लावा!

### विचारमंथन

आमच्याकडे असलेल्या अडचणींवर आधारित विविध दृष्टिकोनांचा विचार करूया:

- **न्यूरल नेटवर्क्स खूप जड आहेत**. आमच्या स्वच्छ, परंतु मर्यादित डेटासेटसाठी, आणि आम्ही स्थानिकपणे नोटबुकद्वारे प्रशिक्षण चालवत असल्यामुळे, न्यूरल नेटवर्क्स या कार्यासाठी खूप जड आहेत.
- **दोन-वर्ग वर्गीकरणकर्ता नाही**. आम्ही दोन-वर्ग वर्गीकरणकर्ता वापरत नाही, त्यामुळे one-vs-all वगळले जाते.
- **निर्णय वृक्ष किंवा लॉजिस्टिक रिग्रेशन चालू शकते**. निर्णय वृक्ष चालू शकतो, किंवा मल्टीक्लास डेटासाठी लॉजिस्टिक रिग्रेशन चालू शकते.
- **मल्टीक्लास बूस्टेड निर्णय वृक्ष वेगळ्या समस्येचे निराकरण करतात**. मल्टीक्लास बूस्टेड निर्णय वृक्ष मुख्यतः नॉनपॅरामेट्रिक कार्यांसाठी योग्य आहेत, जसे की रँकिंग तयार करणे, त्यामुळे ते आमच्यासाठी उपयुक्त नाहीत.

### Scikit-learn वापरणे

आम्ही आमचा डेटा विश्लेषण करण्यासाठी Scikit-learn वापरणार आहोत. तथापि, Scikit-learn मध्ये लॉजिस्टिक रिग्रेशन वापरण्याचे अनेक मार्ग आहेत. [पास करण्यासाठी पॅरामीटर्स](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) पहा.

मूलतः दोन महत्त्वाचे पॅरामीटर्स आहेत - `multi_class` आणि `solver` - जे आम्हाला Scikit-learn ला लॉजिस्टिक रिग्रेशन करण्यास सांगताना निर्दिष्ट करणे आवश्यक आहे. `multi_class` मूल्य विशिष्ट वर्तन लागू करते. solver चे मूल्य कोणता अल्गोरिदम वापरायचा ते ठरवते. सर्व solvers सर्व `multi_class` मूल्यांसह जोडले जाऊ शकत नाहीत.

दस्तऐवजांनुसार, मल्टीक्लास प्रकरणात, प्रशिक्षण अल्गोरिदम:

- **one-vs-rest (OvR) योजना वापरते**, जर `multi_class` पर्याय `ovr` वर सेट केला असेल
- **क्रॉस-एंट्रॉपी लॉस वापरते**, जर `multi_class` पर्याय `multinomial` वर सेट केला असेल. (सध्या `multinomial` पर्याय फक्त ‘lbfgs’, ‘sag’, ‘saga’ आणि ‘newton-cg’ solvers द्वारे समर्थित आहे.)"

> 🎓 येथे 'योजना' 'ovr' (one-vs-rest) किंवा 'multinomial' असू शकते. लॉजिस्टिक रिग्रेशन मुख्यतः बायनरी वर्गीकरणासाठी डिझाइन केले असल्याने, या योजनांमुळे ते मल्टीक्लास वर्गीकरण कार्ये अधिक चांगल्या प्रकारे हाताळू शकते. [स्रोत](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' म्हणजे "ऑप्टिमायझेशन समस्येसाठी वापरायचा अल्गोरिदम". [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn या टेबलद्वारे स्पष्ट करते की solvers वेगवेगळ्या प्रकारच्या डेटा संरचनांमुळे निर्माण होणाऱ्या आव्हानांना कसे हाताळतात:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## व्यायाम - डेटा विभाजित करा

आम्ही आमच्या पहिल्या प्रशिक्षण चाचणीसाठी लॉजिस्टिक रिग्रेशनवर लक्ष केंद्रित करू शकतो कारण तुम्ही मागील धड्यात याबद्दल शिकले होते.
तुमचा डेटा `train_test_split()` कॉल करून प्रशिक्षण आणि चाचणी गटांमध्ये विभाजित करा:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## व्यायाम - लॉजिस्टिक रिग्रेशन लागू करा

तुम्ही मल्टीक्लास प्रकरण वापरत असल्याने, तुम्हाला कोणती _योजना_ वापरायची आणि कोणता _solver_ सेट करायचा ते निवडावे लागेल. मल्टीक्लास सेटिंगसह आणि **liblinear** solver सह LogisticRegression वापरा.

1. `multi_class` `ovr` वर सेट करा आणि solver `liblinear` वर सेट करून लॉजिस्टिक रिग्रेशन तयार करा:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs` सारखा वेगळा solver वापरून पहा, जो अनेकदा डिफॉल्ट म्हणून सेट केला जातो
> लक्षात घ्या, तुमच्या डेटाला सपाट करण्यासाठी आवश्यक असल्यास Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) फंक्शन वापरा.
अचूकता **80%** पेक्षा जास्त चांगली आहे!

1. तुम्ही एका ओळीत डेटा (#50) तपासून या मॉडेलचे काम पाहू शकता:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    परिणाम प्रिंट केला जातो:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ वेगळ्या ओळ क्रमांकाचा प्रयत्न करा आणि परिणाम तपासा

1. अधिक खोलवर जाऊन, तुम्ही या अंदाजाची अचूकता तपासू शकता:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    परिणाम प्रिंट केला जातो - भारतीय खाद्यपदार्थ हे त्याचे सर्वोत्तम अनुमान आहे, चांगल्या संभाव्यतेसह:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ मॉडेलला भारतीय खाद्यपदार्थ असल्याचा विश्वास का आहे हे तुम्ही स्पष्ट करू शकता का?

1. वर्गीकरण अहवाल प्रिंट करून अधिक तपशील मिळवा, जसे तुम्ही रिग्रेशन धड्यांमध्ये केले होते:

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

## 🚀चॅलेंज

या धड्यात, तुम्ही स्वच्छ केलेल्या डेटाचा वापर करून एक मशीन लर्निंग मॉडेल तयार केले जे घटकांच्या मालिकेवर आधारित राष्ट्रीय खाद्यपदार्थाचा अंदाज लावू शकते. Scikit-learn डेटा वर्गीकृत करण्यासाठी प्रदान केलेल्या अनेक पर्यायांबद्दल वाचा. 'solver' या संकल्पनेत खोलवर जा आणि पडद्यामागे काय चालते ते समजून घ्या.

## [पाठानंतरचा क्विझ](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

लॉजिस्टिक रिग्रेशनमागील गणिताबद्दल अधिक जाणून घ्या [या धड्यात](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## असाइनमेंट 

[solversचा अभ्यास करा](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर केल्यामुळे उद्भवणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.