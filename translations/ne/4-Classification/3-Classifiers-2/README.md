<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T06:37:39+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "ne"
}
-->
# खाना वर्गीकरणकर्ता २

यस दोस्रो वर्गीकरण पाठमा, तपाईंले संख्यात्मक डाटालाई वर्गीकरण गर्ने थप तरिकाहरू अन्वेषण गर्नुहुनेछ। साथै, तपाईंले एक वर्गीकरणकर्ता चयन गर्दा हुने प्रभावहरूको बारेमा सिक्नुहुनेछ।

## [पाठ अघि क्विज](https://ff-quizzes.netlify.app/en/ml/)

### पूर्वशर्त

हामी मान्दछौं कि तपाईंले अघिल्लो पाठहरू पूरा गर्नुभएको छ र तपाईंको `data` फोल्डरमा _cleaned_cuisines.csv_ नामक सफा गरिएको डेटासेट छ, जुन यो ४-पाठको फोल्डरको मूलमा छ।

### तयारी

हामीले तपाईंको _notebook.ipynb_ फाइललाई सफा गरिएको डेटासेटसँग लोड गरेका छौं र यसलाई X र y डाटाफ्रेमहरूमा विभाजन गरेका छौं, मोडेल निर्माण प्रक्रियाको लागि तयार।

## वर्गीकरण नक्सा

पहिले, तपाईंले माइक्रोसफ्टको चिट शीट प्रयोग गरेर डाटा वर्गीकरण गर्दा विभिन्न विकल्पहरूको बारेमा सिक्नुभएको थियो। Scikit-learn ले यस्तै तर अझ विस्तृत चिट शीट प्रदान गर्दछ, जसले तपाईंलाई वर्गीकरणकर्ता चयन गर्न अझ सटीक रूपमा मद्दत गर्न सक्छ:

![Scikit-learn बाट ML नक्सा](../../../../4-Classification/3-Classifiers-2/images/map.png)
> टिप: [यो नक्सा अनलाइन हेर्नुहोस्](https://scikit-learn.org/stable/tutorial/machine_learning_map/) र मार्गमा क्लिक गरेर दस्तावेज पढ्नुहोस्।

### योजना

यो नक्सा तपाईंको डाटाको स्पष्ट समझ भएपछि धेरै उपयोगी हुन्छ, किनकि तपाईं यसका मार्गहरू 'हिँडेर' निर्णयमा पुग्न सक्नुहुन्छ:

- हामीसँग >५० नमूनाहरू छन्
- हामीले एक श्रेणीको भविष्यवाणी गर्नुपर्छ
- हामीसँग लेबल गरिएको डाटा छ
- हामीसँग १००K भन्दा कम नमूनाहरू छन्
- ✨ हामीले Linear SVC चयन गर्न सक्छौं
- यदि यो काम गरेन भने, किनकि हामीसँग संख्यात्मक डाटा छ
    - हामी ✨ KNeighbors Classifier प्रयास गर्न सक्छौं 
      - यदि यो काम गरेन भने, ✨ SVC र ✨ Ensemble Classifiers प्रयास गर्नुहोस्

यो पछ्याउन धेरै उपयोगी मार्ग हो।

## अभ्यास - डाटा विभाजन गर्नुहोस्

यस मार्गलाई पछ्याउँदै, हामीले प्रयोग गर्न केही पुस्तकालयहरू आयात गर्नुपर्छ।

1. आवश्यक पुस्तकालयहरू आयात गर्नुहोस्:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. तपाईंको प्रशिक्षण र परीक्षण डाटा विभाजन गर्नुहोस्:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC वर्गीकरणकर्ता

Support-Vector Clustering (SVC) Support-Vector Machines परिवारको ML प्रविधिको एक हिस्सा हो (तल यसबारे थप जान्नुहोस्)। यस विधिमा, तपाईंले 'kernel' चयन गर्न सक्नुहुन्छ जसले लेबलहरू कसरी समूहबद्ध गर्ने निर्णय गर्दछ। 'C' प्यारामिटर 'regularization' लाई जनाउँछ, जसले प्यारामिटरहरूको प्रभावलाई नियमन गर्दछ। Kernel [कयौं](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) मध्ये एक हुन सक्छ; यहाँ हामीले यसलाई 'linear' मा सेट गरेका छौं ताकि Linear SVC प्रयोग गर्न सकियोस्। Probability डिफल्टमा 'false' हुन्छ; यहाँ हामीले यसलाई 'true' मा सेट गरेका छौं ताकि probability estimates प्राप्त गर्न सकियोस्। हामीले random state लाई '0' मा सेट गरेका छौं ताकि डाटा शफल गरेर probabilities प्राप्त गर्न सकियोस्।

### अभ्यास - Linear SVC लागू गर्नुहोस्

क्लासिफायरहरूको एक array सिर्जना गरेर सुरु गर्नुहोस्। हामीले परीक्षण गर्दा यस array मा क्रमिक रूपमा थप्नेछौं।

1. Linear SVC बाट सुरु गर्नुहोस्:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC प्रयोग गरेर तपाईंको मोडेल प्रशिक्षण गर्नुहोस् र रिपोर्ट प्रिन्ट गर्नुहोस्:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    नतिजा धेरै राम्रो छ:

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

## K-Neighbors वर्गीकरणकर्ता

K-Neighbors "neighbors" परिवारको ML विधिको हिस्सा हो, जसले supervised र unsupervised दुवै सिकाइका लागि प्रयोग गर्न सकिन्छ। यस विधिमा, पूर्वनिर्धारित बिन्दुहरूको संख्या सिर्जना गरिन्छ र डाटा ती बिन्दुहरूको वरिपरि संकलन गरिन्छ ताकि सामान्यीकृत लेबलहरू डाटाको लागि भविष्यवाणी गर्न सकियोस्।

### अभ्यास - K-Neighbors वर्गीकरणकर्ता लागू गर्नुहोस्

अघिल्लो वर्गीकरणकर्ता राम्रो थियो, र डाटासँग राम्रोसँग काम गर्यो, तर सायद हामी अझ राम्रो accuracy प्राप्त गर्न सक्छौं। K-Neighbors वर्गीकरणकर्ता प्रयास गर्नुहोस्।

1. तपाईंको क्लासिफायर array मा एक लाइन थप्नुहोस् (Linear SVC आइटम पछि comma थप्नुहोस्):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    नतिजा अलि खराब छ:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) को बारेमा जान्नुहोस्

## Support Vector Classifier

Support-Vector Classifiers [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) परिवारको ML विधिको हिस्सा हुन्, जसले वर्गीकरण र regression कार्यहरूको लागि प्रयोग गरिन्छ। SVMs "प्रशिक्षण उदाहरणहरूलाई ठाउँमा बिन्दुहरूमा म्याप" गर्छन् ताकि दुई श्रेणीहरू बीचको दूरी अधिकतम गर्न सकियोस्। त्यसपछि डाटालाई यस ठाउँमा म्याप गरिन्छ ताकि तिनीहरूको श्रेणी भविष्यवाणी गर्न सकियोस्।

### अभ्यास - Support Vector Classifier लागू गर्नुहोस्

Support Vector Classifier प्रयोग गरेर अलि राम्रो accuracy प्राप्त गर्ने प्रयास गरौं।

1. K-Neighbors आइटम पछि comma थप्नुहोस्, र यो लाइन थप्नुहोस्:

    ```python
    'SVC': SVC(),
    ```

    नतिजा धेरै राम्रो छ!

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

    ✅ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) को बारेमा जान्नुहोस्

## Ensemble Classifiers

पथको अन्त्यसम्म पुगौं, यद्यपि अघिल्लो परीक्षण धेरै राम्रो थियो। 'Ensemble Classifiers' प्रयास गरौं, विशेष गरी Random Forest र AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

नतिजा धेरै राम्रो छ, विशेष गरी Random Forest को लागि:

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

✅ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html) को बारेमा जान्नुहोस्

Machine Learning को यो विधिले "कई आधार अनुमानकर्ताहरूको भविष्यवाणीलाई संयोजन" गरेर मोडेलको गुणस्तर सुधार गर्दछ। हाम्रो उदाहरणमा, हामीले Random Trees र AdaBoost प्रयोग गरेका छौं।

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), एक औसत विधि, 'decision trees' को 'forest' निर्माण गर्दछ जसमा randomness समावेश गरिएको हुन्छ ताकि overfitting रोक्न सकियोस्। n_estimators प्यारामिटरलाई रूखहरूको संख्या सेट गरिएको छ।

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) एक वर्गीकरणकर्तालाई डेटासेटमा फिट गर्छ र त्यस वर्गीकरणकर्ताको प्रतिलिपिहरूलाई सोही डेटासेटमा फिट गर्छ। यसले गलत वर्गीकृत वस्तुहरूको वजनमा ध्यान केन्द्रित गर्छ र अर्को वर्गीकरणकर्ताको फिटलाई सुधार गर्न समायोजन गर्छ।

---

## 🚀 चुनौती

यी प्रत्येक प्रविधिहरूमा धेरै प्यारामिटरहरू छन् जसलाई तपाईं समायोजन गर्न सक्नुहुन्छ। प्रत्येकको डिफल्ट प्यारामिटरहरूको अनुसन्धान गर्नुहोस् र यी प्यारामिटरहरू समायोजन गर्दा मोडेलको गुणस्तरमा के प्रभाव पर्छ भनेर सोच्नुहोस्।

## [पाठ पछि क्विज](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा र आत्म अध्ययन

यी पाठहरूमा धेरै जटिल शब्दावली छ, त्यसैले [यो सूची](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) को उपयोगी शब्दावली समीक्षा गर्न एक मिनेट लिनुहोस्!

## असाइनमेन्ट 

[प्यारामिटर खेल](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।