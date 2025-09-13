<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T06:17:10+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "mr"
}
-->
# क्युझीन वर्गीकरण २

या दुसऱ्या वर्गीकरणाच्या धड्यात, तुम्ही संख्यात्मक डेटाचे वर्गीकरण करण्याचे अधिक मार्ग शोधाल. तसेच, एका वर्गीकरण पद्धतीच्या निवडीचे परिणाम काय असू शकतात हे देखील शिकाल.

## [पूर्व-व्याख्यान प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/)

### पूर्वअट

आम्ही गृहीत धरतो की तुम्ही मागील धडे पूर्ण केले आहेत आणि तुमच्या `data` फोल्डरमध्ये _cleaned_cuisines.csv_ नावाचा स्वच्छ डेटा संच आहे, जो या ४-धड्यांच्या फोल्डरच्या मूळ ठिकाणी आहे.

### तयारी

आम्ही तुमच्या _notebook.ipynb_ फाइलमध्ये स्वच्छ डेटा लोड केला आहे आणि तो X आणि y डेटा फ्रेम्समध्ये विभागला आहे, जो मॉडेल तयार करण्याच्या प्रक्रियेसाठी तयार आहे.

## वर्गीकरणाचा नकाशा

यापूर्वी, तुम्ही मायक्रोसॉफ्टच्या चीट शीटचा वापर करून डेटा वर्गीकृत करण्याचे विविध पर्याय शिकला होता. Scikit-learn देखील एक समान, परंतु अधिक तपशीलवार चीट शीट प्रदान करते, जी तुमच्या वर्गीकरणासाठी योग्य पर्याय निवडण्यात मदत करू शकते:

![Scikit-learn कडून ML नकाशा](../../../../4-Classification/3-Classifiers-2/images/map.png)
> टीप: [हा नकाशा ऑनलाइन पहा](https://scikit-learn.org/stable/tutorial/machine_learning_map/) आणि मार्गावर क्लिक करून दस्तऐवज वाचा.

### योजना

हा नकाशा तुमच्या डेटाचा स्पष्ट अंदाज आल्यावर खूप उपयुक्त ठरतो, कारण तुम्ही त्याच्या मार्गांवरून निर्णय घेऊ शकता:

- आमच्याकडे >50 नमुने आहेत
- आम्हाला श्रेणीचा अंदाज लावायचा आहे
- आमच्याकडे लेबल केलेला डेटा आहे
- आमच्याकडे 100K पेक्षा कमी नमुने आहेत
- ✨ आम्ही Linear SVC निवडू शकतो
- जर ते काम केले नाही, कारण आमच्याकडे संख्यात्मक डेटा आहे
    - आम्ही ✨ KNeighbors Classifier वापरून पाहू शकतो
      - जर तेही काम केले नाही, तर ✨ SVC आणि ✨ Ensemble Classifiers वापरून पाहू शकतो

हा मार्ग अनुसरण्यासाठी खूप उपयुक्त आहे.

## व्यायाम - डेटा विभाजित करा

या मार्गाचे अनुसरण करताना, आपल्याला वापरण्यासाठी काही लायब्ररी आयात करणे आवश्यक आहे.

1. आवश्यक लायब्ररी आयात करा:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. तुमचा प्रशिक्षण आणि चाचणी डेटा विभाजित करा:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC वर्गीकरण

Support-Vector Clustering (SVC) ही Support-Vector Machines या ML तंत्रज्ञानाच्या कुटुंबातील एक पद्धत आहे (खाली याबद्दल अधिक जाणून घ्या). या पद्धतीत, तुम्ही लेबल्स कसे गटबद्ध करायचे हे ठरवण्यासाठी 'kernel' निवडू शकता. 'C' पॅरामीटर 'regularization' दर्शवतो, जो पॅरामीटर्सच्या प्रभावाचे नियमन करतो. Kernel [काही पर्यायांपैकी](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) एक असू शकतो; येथे आम्ही Linear SVC वापरण्यासाठी 'linear' सेट करतो. Probability डीफॉल्टने 'false' असते; येथे आम्ही Probability Estimates गोळा करण्यासाठी 'true' सेट करतो. Random State '0' वर सेट करतो, जेणेकरून डेटा शफल होईल आणि Probability मिळेल.

### व्यायाम - Linear SVC लागू करा

वर्गीकरणांची एक array तयार करून सुरुवात करा. आम्ही चाचणी करताना या array मध्ये हळूहळू भर घालू.

1. Linear SVC ने सुरुवात करा:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Linear SVC वापरून तुमचे मॉडेल प्रशिक्षण द्या आणि रिपोर्ट प्रिंट करा:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    परिणाम खूप चांगला आहे:

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

## K-Neighbors वर्गीकरण

K-Neighbors ही ML पद्धतींच्या "neighbors" कुटुंबाचा भाग आहे, जी पर्यवेक्षित आणि अप्रत्यक्ष शिक्षणासाठी वापरली जाऊ शकते. या पद्धतीत, पूर्वनिर्धारित बिंदू तयार केले जातात आणि डेटा या बिंदूंच्या आसपास गोळा केला जातो, ज्यामुळे डेटासाठी सामान्यीकृत लेबल्सचा अंदाज लावता येतो.

### व्यायाम - K-Neighbors वर्गीकरण लागू करा

मागील वर्गीकरण चांगले होते आणि डेटासह चांगले काम केले, परंतु कदाचित आम्हाला अधिक चांगली अचूकता मिळू शकेल. K-Neighbors वर्गीकरण वापरून पाहा.

1. तुमच्या वर्गीकरण array मध्ये एक ओळ जोडा (Linear SVC आयटमनंतर अल्पविराम जोडा):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    परिणाम थोडा वाईट आहे:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) बद्दल जाणून घ्या

## Support Vector Classifier

Support-Vector Classifiers हे [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) कुटुंबातील भाग आहेत, जे वर्गीकरण आणि पुनर्रचना कार्यांसाठी वापरले जातात. SVMs "प्रशिक्षण उदाहरणांना जागेतील बिंदूंमध्ये नकाशित करतात" जेणेकरून दोन श्रेणींमधील अंतर जास्तीत जास्त होईल. त्यानंतरचा डेटा या जागेत नकाशित केला जातो, त्यामुळे त्यांची श्रेणी अंदाजित केली जाऊ शकते.

### व्यायाम - Support Vector Classifier लागू करा

थोडी अधिक चांगली अचूकता मिळवण्यासाठी Support Vector Classifier वापरून पाहूया.

1. K-Neighbors आयटमनंतर अल्पविराम जोडा आणि ही ओळ जोडा:

    ```python
    'SVC': SVC(),
    ```

    परिणाम खूप चांगला आहे!

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

    ✅ [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm) बद्दल जाणून घ्या

## Ensemble Classifiers

मागील चाचणी खूप चांगली होती, तरीही आपण शेवटपर्यंतचा मार्ग अनुसरूया. 'Ensemble Classifiers' वापरून पाहूया, विशेषतः Random Forest आणि AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

Random Forest साठी परिणाम खूप चांगला आहे:

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

✅ [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html) बद्दल जाणून घ्या

ही Machine Learning पद्धत "काही बेस estimators च्या अंदाजांना एकत्र करते" जेणेकरून मॉडेलची गुणवत्ता सुधारली जाईल. आपल्या उदाहरणात, आम्ही Random Trees आणि AdaBoost वापरले.

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), एक सरासरी पद्धत, 'decision trees' चा 'forest' तयार करते, ज्यामध्ये randomness समाविष्ट असते, ज्यामुळे overfitting टाळले जाते. n_estimators पॅरामीटर झाडांची संख्या सेट करतो.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) एका डेटासेटवर classifier बसवतो आणि नंतर त्याच डेटासेटवर त्या classifier च्या प्रती बसवतो. चुकीने वर्गीकृत केलेल्या आयटम्सच्या वजनांवर लक्ष केंद्रित करतो आणि पुढील classifier साठी फिट समायोजित करतो.

---

## 🚀 आव्हान

या प्रत्येक तंत्रज्ञानामध्ये तुम्ही बदलू शकता असे बरेच पॅरामीटर्स असतात. प्रत्येकाच्या डीफॉल्ट पॅरामीटर्सचा अभ्यास करा आणि हे पॅरामीटर्स बदलल्याने मॉडेलच्या गुणवत्तेवर काय परिणाम होईल याचा विचार करा.

## [व्याख्यानानंतरची प्रश्नमंजुषा](https://ff-quizzes.netlify.app/en/ml/)

## पुनरावलोकन आणि स्व-अभ्यास

या धड्यांमध्ये बरीच तांत्रिक शब्दावली आहे, त्यामुळे [या यादीचा](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) आढावा घ्या, जी उपयुक्त संज्ञांची आहे!

## असाइनमेंट 

[पॅरामीटर प्ले](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) चा वापर करून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी, कृपया लक्षात घ्या की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील मूळ दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर केल्यामुळे उद्भवणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.