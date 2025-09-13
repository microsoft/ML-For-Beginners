<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T06:36:29+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "ne"
}
-->
# खाना वर्गीकरणकर्ता १

यस पाठमा, तपाईंले अघिल्लो पाठमा बचत गरिएको डेटा सेट प्रयोग गर्नुहुनेछ, जुन सन्तुलित, सफा डेटा हो र विभिन्न प्रकारका खानाका बारेमा जानकारी समेटिएको छ।

तपाईंले यो डेटा सेट विभिन्न वर्गीकरणकर्ताहरूसँग प्रयोग गर्नुहुनेछ ताकि _सामग्रीहरूको समूहको आधारमा कुनै राष्ट्रिय खानाको भविष्यवाणी गर्न सकियोस्_। यस क्रममा, तपाईंले वर्गीकरण कार्यहरूको लागि एल्गोरिदमहरू कसरी प्रयोग गर्न सकिन्छ भन्ने बारे थप जान्नुहुनेछ।

## [पाठ अघि क्विज](https://ff-quizzes.netlify.app/en/ml/)
# तयारी

यदि तपाईंले [पाठ १](../1-Introduction/README.md) पूरा गर्नुभएको छ भने, सुनिश्चित गर्नुहोस् कि _cleaned_cuisines.csv_ फाइल `/data` फोल्डरको मूलमा यी चार पाठहरूको लागि उपलब्ध छ।

## अभ्यास - राष्ट्रिय खाना भविष्यवाणी गर्नुहोस्

1. यस पाठको _notebook.ipynb_ फोल्डरमा काम गर्दै, उक्त फाइल र Pandas लाइब्रेरी आयात गर्नुहोस्:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डेटा यस प्रकार देखिन्छ:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. अब, केही थप लाइब्रेरीहरू आयात गर्नुहोस्:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X र y समन्वयलाई दुई डेटा फ्रेममा विभाजन गर्नुहोस्। `cuisine` लेबलहरूको डेटा फ्रेम हुन सक्छ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    यो यस प्रकार देखिन्छ:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` स्तम्भ र `cuisine` स्तम्भलाई `drop()` प्रयोग गरेर हटाउनुहोस्। बाँकी डेटा प्रशिक्षण योग्य सुविधाहरूको रूपमा बचत गर्नुहोस्:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    तपाईंको सुविधाहरू यस प्रकार देखिन्छ:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

अब तपाईं आफ्नो मोडेल प्रशिक्षण गर्न तयार हुनुहुन्छ!

## वर्गीकरणकर्ता चयन गर्दै

अब तपाईंको डेटा सफा र प्रशिक्षणको लागि तयार छ, तपाईंले कुन एल्गोरिदम प्रयोग गर्ने निर्णय गर्नुपर्छ।

Scikit-learn ले वर्गीकरणलाई Supervised Learning अन्तर्गत समेट्छ, र यस श्रेणीमा तपाईंले वर्गीकरणका लागि धेरै विधिहरू पाउनुहुनेछ। [विविधता](https://scikit-learn.org/stable/supervised_learning.html) पहिलो नजरमा अलमलमा पार्न सक्छ। निम्न विधिहरूले वर्गीकरण प्रविधिहरू समावेश गर्छन्:

- रेखीय मोडेलहरू
- सपोर्ट भेक्टर मेसिनहरू
- स्टोकास्टिक ग्रेडियन्ट डिसेन्ट
- नजिकका छिमेकीहरू
- गाउसीयन प्रक्रियाहरू
- निर्णय वृक्षहरू
- Ensemble विधिहरू (मतदान वर्गीकरणकर्ता)
- बहु-वर्ग र बहु-आउटपुट एल्गोरिदमहरू (बहु-वर्ग र बहु-लेबल वर्गीकरण, बहु-वर्ग-बहु-आउटपुट वर्गीकरण)

> तपाईं [न्यूरल नेटवर्कहरू प्रयोग गरेर डेटा वर्गीकृत गर्न](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) पनि सक्नुहुन्छ, तर यो पाठको दायरा बाहिर छ।

### कुन वर्गीकरणकर्ता चयन गर्ने?

त्यसो भए, कुन वर्गीकरणकर्ता चयन गर्ने? प्रायः, धेरै विधिहरू चलाएर राम्रो नतिजा खोज्नु परीक्षण गर्ने तरिका हो। Scikit-learn ले [साइड-बाई-साइड तुलना](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) प्रदान गर्दछ, जहाँ KNeighbors, SVC दुई तरिकाले, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB र QuadraticDiscriminationAnalysis को तुलना गरिएको छ, र नतिजाहरू दृश्यात्मक रूपमा देखाइएको छ:

![वर्गीकरणकर्ताहरूको तुलना](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Scikit-learn को दस्तावेजमा उत्पन्न प्लटहरू

> AutoML ले यो समस्या सजिलै समाधान गर्छ, यी तुलना क्लाउडमा चलाएर तपाईंलाई तपाईंको डेटा लागि सबैभन्दा उपयुक्त एल्गोरिदम चयन गर्न अनुमति दिन्छ। यसलाई [यहाँ प्रयास गर्नुहोस्](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### राम्रो दृष्टिकोण

अनुमान लगाउने भन्दा राम्रो तरिका भनेको यो डाउनलोड गर्न मिल्ने [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) को विचारहरू अनुसरण गर्नु हो। यहाँ, हामी पत्ता लगाउँछौं कि हाम्रो बहु-वर्ग समस्याको लागि, हामीसँग केही विकल्पहरू छन्:

![बहु-वर्ग समस्याहरूको लागि चिट शीट](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> माइक्रोसफ्टको एल्गोरिदम चिट शीटको एक भाग, बहु-वर्ग वर्गीकरण विकल्पहरू विवरण गर्दै

✅ यो चिट शीट डाउनलोड गर्नुहोस्, प्रिन्ट गर्नुहोस्, र तपाईंको भित्तामा टाँस्नुहोस्!

### तर्क

हामीसँग भएका सीमाहरूलाई ध्यानमा राख्दै विभिन्न दृष्टिकोणहरूको तर्क गर्न प्रयास गरौं:

- **न्यूरल नेटवर्कहरू धेरै भारी छन्**। हाम्रो सफा, तर न्यूनतम डेटा सेटलाई ध्यानमा राख्दै, र तथ्य यो हो कि हामी स्थानीय रूपमा नोटबुकहरू मार्फत प्रशिक्षण चलाउँदैछौं, न्यूरल नेटवर्कहरू यस कार्यको लागि धेरै भारी छन्।
- **दुई-वर्ग वर्गीकरणकर्ता छैन**। हामी दुई-वर्ग वर्गीकरणकर्ता प्रयोग गर्दैनौं, त्यसैले यसले one-vs-all लाई अस्वीकार गर्छ।
- **निर्णय वृक्ष वा Logistic Regression काम गर्न सक्छ**। निर्णय वृक्ष काम गर्न सक्छ, वा बहु-वर्ग डेटा लागि Logistic Regression।
- **बहु-वर्ग Boosted Decision Trees फरक समस्या समाधान गर्छ**। बहु-वर्ग Boosted Decision Tree गैर-प्यारामेट्रिक कार्यहरूको लागि सबैभन्दा उपयुक्त छ, जस्तै रैंकिङ निर्माण गर्न डिजाइन गरिएका कार्यहरू, त्यसैले यो हाम्रो लागि उपयोगी छैन।

### Scikit-learn प्रयोग गर्दै

हामी Scikit-learn प्रयोग गरेर हाम्रो डेटा विश्लेषण गर्नेछौं। तर, Scikit-learn मा Logistic Regression प्रयोग गर्ने धेरै तरिकाहरू छन्। पास गर्नुपर्ने [प्यारामिटरहरू](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) हेर्नुहोस्।  

मूलतः दुई महत्त्वपूर्ण प्यारामिटरहरू छन् - `multi_class` र `solver` - जुन हामीले निर्दिष्ट गर्नुपर्छ, जब हामी Scikit-learn लाई Logistic Regression प्रदर्शन गर्न सोध्छौं। `multi_class` मानले निश्चित व्यवहार लागू गर्छ। solver को मान कुन एल्गोरिदम प्रयोग गर्ने हो। सबै solvers लाई सबै `multi_class` मानहरूसँग जोड्न सकिँदैन।

दस्तावेज अनुसार, बहु-वर्ग केसमा, प्रशिक्षण एल्गोरिदम:

- **one-vs-rest (OvR) योजना प्रयोग गर्छ**, यदि `multi_class` विकल्प `ovr` मा सेट गरिएको छ भने।
- **cross-entropy loss प्रयोग गर्छ**, यदि `multi_class` विकल्प `multinomial` मा सेट गरिएको छ भने। (हाल `multinomial` विकल्प केवल ‘lbfgs’, ‘sag’, ‘saga’ र ‘newton-cg’ solvers द्वारा समर्थित छ।)

> 🎓 यहाँ 'scheme' या त 'ovr' (one-vs-rest) वा 'multinomial' हुन सक्छ। Logistic Regression वास्तवमा द्वि-वर्ग वर्गीकरणलाई समर्थन गर्न डिजाइन गरिएको हो, यी योजनाहरूले यसलाई बहु-वर्ग वर्गीकरण कार्यहरू राम्रोसँग ह्यान्डल गर्न अनुमति दिन्छ। [स्रोत](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' लाई "अनुकूलन समस्यामा प्रयोग गर्नुपर्ने एल्गोरिदम" भनेर परिभाषित गरिएको छ। [स्रोत](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ले विभिन्न प्रकारका डेटा संरचनाहरूले प्रस्तुत गर्ने चुनौतीहरू कसरी solvers ले ह्यान्डल गर्छन् भन्ने व्याख्या गर्न यो तालिका प्रदान गर्दछ:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## अभ्यास - डेटा विभाजन गर्नुहोस्

तपाईंले अघिल्लो पाठमा Logistic Regression को बारेमा सिक्नुभएकोले, हामी पहिलो प्रशिक्षण प्रयासको लागि यसमा ध्यान केन्द्रित गर्न सक्छौं।
तपाईंको डेटा `train_test_split()` कल गरेर प्रशिक्षण र परीक्षण समूहहरूमा विभाजन गर्नुहोस्:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## अभ्यास - Logistic Regression लागू गर्नुहोस्

तपाईं बहु-वर्ग केस प्रयोग गर्दै हुनुहुन्छ, त्यसैले तपाईंले कुन _scheme_ प्रयोग गर्ने र कुन _solver_ सेट गर्ने निर्णय गर्नुपर्छ। बहु-वर्ग सेटिङ र **liblinear** solver प्रयोग गरेर Logistic Regression लागू गर्नुहोस्।

1. `multi_class` लाई `ovr` मा सेट गरेर र solver लाई `liblinear` मा सेट गरेर Logistic Regression सिर्जना गर्नुहोस्:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ `lbfgs` जस्तो फरक solver प्रयास गर्नुहोस्, जुन प्रायः डिफल्टको रूपमा सेट गरिएको हुन्छ।
> नोट, आवश्यक परेको बेला आफ्नो डेटा समतल बनाउन Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) फङ्सन प्रयोग गर्नुहोस्।
यो मोडेलको सटीकता **८०% भन्दा बढी** राम्रो छ!

1. तपाईंले यो मोडेललाई एउटा पङ्क्ति (#५०) परीक्षण गरेर प्रयोगमा देख्न सक्नुहुन्छ:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    नतिजा प्रिन्ट हुन्छ:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ फरक पङ्क्ति नम्बर प्रयास गर्नुहोस् र नतिजा जाँच गर्नुहोस्।

1. अझ गहिराइमा जानुहोस्, तपाईं यस भविष्यवाणीको सटीकता जाँच गर्न सक्नुहुन्छ:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    नतिजा प्रिन्ट हुन्छ - भारतीय खाना यसको सबैभन्दा राम्रो अनुमान हो, राम्रो सम्भावनासहित:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ तपाईं यस मोडेललाई किन भारतीय खाना भनेर निश्चित लागेको छ भनेर व्याख्या गर्न सक्नुहुन्छ?

1. थप विवरण प्राप्त गर्न, वर्गीकरण रिपोर्ट प्रिन्ट गर्नुहोस्, जस्तै तपाईंले regression पाठहरूमा गर्नुभएको थियो:

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

## 🚀चुनौती

यस पाठमा, तपाईंले आफ्नो सफा गरिएको डाटालाई प्रयोग गरेर एउटा मेसिन लर्निङ मोडेल निर्माण गर्नुभयो, जसले सामग्रीहरूको आधारमा राष्ट्रिय खानाको भविष्यवाणी गर्न सक्छ। Scikit-learn ले डाटा वर्गीकरण गर्न प्रदान गर्ने धेरै विकल्पहरू पढ्न समय निकाल्नुहोस्। 'solver' को अवधारणामा गहिराइमा जानुहोस् र पर्दा पछाडि के हुन्छ बुझ्नुहोस्।

## [पाठपछिको प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म-अध्ययन

[यस पाठ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) मा logistic regression को गणितीय पक्षमा अझ गहिराइमा जानुहोस्।
## असाइनमेन्ट 

[solvers अध्ययन गर्नुहोस्](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादहरूमा त्रुटि वा अशुद्धता हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।