<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-05T10:27:55+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "hi"
}
-->
# व्यंजन वर्गीकरणकर्ता 2

इस दूसरे वर्गीकरण पाठ में, आप संख्यात्मक डेटा को वर्गीकृत करने के और तरीके जानेंगे। आप यह भी समझेंगे कि एक वर्गीकरणकर्ता को दूसरे पर चुनने के क्या परिणाम हो सकते हैं।

## [पाठ-पूर्व प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)

### पूर्वापेक्षा

हम मानते हैं कि आपने पिछले पाठ पूरे कर लिए हैं और आपके पास `data` फ़ोल्डर में _cleaned_cuisines.csv_ नामक एक साफ़ किया हुआ डेटासेट है, जो इस 4-पाठ फ़ोल्डर की रूट में है।

### तैयारी

हमने आपके _notebook.ipynb_ फ़ाइल में साफ़ किया हुआ डेटासेट लोड कर दिया है और इसे X और y डेटा फ्रेम में विभाजित कर दिया है, जो मॉडल निर्माण प्रक्रिया के लिए तैयार हैं।

## एक वर्गीकरण मानचित्र

पिछले पाठ में, आपने डेटा को वर्गीकृत करने के विभिन्न विकल्पों के बारे में सीखा था, जिसमें Microsoft का चीट शीट शामिल था। Scikit-learn एक समान, लेकिन अधिक विस्तृत चीट शीट प्रदान करता है, जो आपके वर्गीकरणकर्ताओं (जिसे 'एस्टिमेटर्स' भी कहा जाता है) को और अधिक संकीर्ण करने में मदद कर सकता है:

![Scikit-learn से ML मानचित्र](../../../../4-Classification/3-Classifiers-2/images/map.png)
> टिप: [इस मानचित्र को ऑनलाइन देखें](https://scikit-learn.org/stable/tutorial/machine_learning_map/) और मार्ग पर क्लिक करके प्रलेखन पढ़ें।

### योजना

यह मानचित्र तब बहुत सहायक होता है जब आपको अपने डेटा की स्पष्ट समझ हो, क्योंकि आप इसके मार्गों पर चलते हुए निर्णय ले सकते हैं:

- हमारे पास >50 नमूने हैं
- हम एक श्रेणी की भविष्यवाणी करना चाहते हैं
- हमारे पास लेबल किया हुआ डेटा है
- हमारे पास 100K से कम नमूने हैं
- ✨ हम एक Linear SVC चुन सकते हैं
- यदि यह काम नहीं करता है, क्योंकि हमारे पास संख्यात्मक डेटा है
    - हम ✨ KNeighbors Classifier आज़मा सकते हैं 
      - यदि यह काम नहीं करता है, तो ✨ SVC और ✨ Ensemble Classifiers आज़माएं

यह अनुसरण करने के लिए एक बहुत ही सहायक मार्ग है।

## अभ्यास - डेटा विभाजित करें

इस मार्ग का अनुसरण करते हुए, हमें उपयोग के लिए कुछ लाइब्रेरी आयात करके शुरू करना चाहिए।

1. आवश्यक लाइब्रेरी आयात करें:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. अपने प्रशिक्षण और परीक्षण डेटा को विभाजित करें:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC वर्गीकरणकर्ता

सपोर्ट-वेक्टर क्लस्टरिंग (SVC) मशीन लर्निंग तकनीकों के सपोर्ट-वेक्टर मशीन परिवार का हिस्सा है (नीचे इनके बारे में और जानें)। इस विधि में, आप लेबल को क्लस्टर करने के लिए 'कर्नेल' चुन सकते हैं। 'C' पैरामीटर 'रेग्युलराइज़ेशन' को संदर्भित करता है, जो पैरामीटरों के प्रभाव को नियंत्रित करता है। कर्नेल [कई](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) में से एक हो सकता है; यहां हम इसे 'linear' पर सेट करते हैं ताकि हम Linear SVC का लाभ उठा सकें। Probability डिफ़ॉल्ट रूप से 'false' होती है; यहां हम इसे 'true' पर सेट करते हैं ताकि संभावना अनुमान प्राप्त कर सकें। हम डेटा को शफल करने के लिए रैंडम स्टेट को '0' पर सेट करते हैं।

### अभ्यास - Linear SVC लागू करें

क्लासिफायर का एक ऐरे बनाकर शुरू करें। जैसे-जैसे हम परीक्षण करेंगे, आप इस ऐरे में प्रगतिशील रूप से जोड़ेंगे।

1. Linear SVC से शुरू करें:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC का उपयोग करके अपने मॉडल को प्रशिक्षित करें और एक रिपोर्ट प्रिंट करें:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    परिणाम काफी अच्छा है:

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

K-Neighbors "पड़ोसियों" परिवार का हिस्सा है, जिसे सुपरवाइज़्ड और अनसुपरवाइज़्ड लर्निंग दोनों के लिए उपयोग किया जा सकता है। इस विधि में, एक पूर्वनिर्धारित संख्या में बिंदु बनाए जाते हैं और डेटा को इन बिंदुओं के चारों ओर इकट्ठा किया जाता है ताकि डेटा के लिए सामान्यीकृत लेबल की भविष्यवाणी की जा सके।

### अभ्यास - K-Neighbors वर्गीकरणकर्ता लागू करें

पिछला वर्गीकरणकर्ता अच्छा था और डेटा के साथ अच्छी तरह से काम किया, लेकिन शायद हम बेहतर सटीकता प्राप्त कर सकते हैं। K-Neighbors वर्गीकरणकर्ता आज़माएं।

1. अपने क्लासिफायर ऐरे में एक पंक्ति जोड़ें (Linear SVC आइटम के बाद एक कॉमा जोड़ें):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    परिणाम थोड़ा खराब है:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) के बारे में जानें

## सपोर्ट वेक्टर क्लासिफायर

सपोर्ट-वेक्टर क्लासिफायर [सपोर्ट-वेक्टर मशीन](https://wikipedia.org/wiki/Support-vector_machine) परिवार का हिस्सा हैं, जो वर्गीकरण और प्रतिगमन कार्यों के लिए उपयोग किए जाते हैं। SVM "प्रशिक्षण उदाहरणों को स्थान में बिंदुओं पर मैप करते हैं" ताकि दो श्रेणियों के बीच की दूरी को अधिकतम किया जा सके। इसके बाद डेटा को इस स्थान में मैप किया जाता है ताकि उनकी श्रेणी की भविष्यवाणी की जा सके।

### अभ्यास - सपोर्ट वेक्टर क्लासिफायर लागू करें

आइए सपोर्ट वेक्टर क्लासिफायर के साथ थोड़ी बेहतर सटीकता प्राप्त करने का प्रयास करें।

1. K-Neighbors आइटम के बाद एक कॉमा जोड़ें, और फिर यह पंक्ति जोड़ें:

    ```python
    'SVC': SVC(),
    ```

    परिणाम काफी अच्छा है!

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

    ✅ [सपोर्ट-वेक्टर](https://scikit-learn.org/stable/modules/svm.html#svm) के बारे में जानें

## एन्सेम्बल क्लासिफायर

आइए इस मार्ग के अंत तक चलते हैं, भले ही पिछला परीक्षण काफी अच्छा था। आइए कुछ 'एन्सेम्बल क्लासिफायर' आज़माएं, विशेष रूप से Random Forest और AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

परिणाम बहुत अच्छा है, विशेष रूप से Random Forest के लिए:

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

✅ [एन्सेम्बल क्लासिफायर](https://scikit-learn.org/stable/modules/ensemble.html) के बारे में जानें

यह मशीन लर्निंग विधि "कई बेस एस्टिमेटर्स की भविष्यवाणियों को जोड़ती है" ताकि मॉडल की गुणवत्ता में सुधार हो सके। हमारे उदाहरण में, हमने Random Trees और AdaBoost का उपयोग किया।

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), एक एवरेजिंग विधि, 'डिसीजन ट्री' का एक 'फॉरेस्ट' बनाता है, जिसमें यादृच्छिकता को शामिल किया जाता है ताकि ओवरफिटिंग से बचा जा सके। n_estimators पैरामीटर पेड़ों की संख्या पर सेट होता है।

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) एक डेटासेट पर एक क्लासिफायर फिट करता है और फिर उसी डेटासेट पर उस क्लासिफायर की प्रतियां फिट करता है। यह गलत वर्गीकृत वस्तुओं के वज़न पर ध्यान केंद्रित करता है और अगले क्लासिफायर के लिए फिट को सही करने के लिए समायोजित करता है।

---

## 🚀चुनौती

इनमें से प्रत्येक तकनीक में कई पैरामीटर होते हैं जिन्हें आप समायोजित कर सकते हैं। प्रत्येक के डिफ़ॉल्ट पैरामीटर का अध्ययन करें और सोचें कि इन पैरामीटरों को समायोजित करने से मॉडल की गुणवत्ता पर क्या प्रभाव पड़ेगा।

## [पाठ-उत्तर प्रश्नोत्तरी](https://ff-quizzes.netlify.app/en/ml/)

## समीक्षा और स्व-अध्ययन

इन पाठों में बहुत सारा तकनीकी शब्दावली है, इसलिए [इस सूची](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) को पढ़ने के लिए कुछ समय निकालें, जिसमें उपयोगी शब्दावली दी गई है!

## असाइनमेंट 

[पैरामीटर प्ले](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयासरत हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  