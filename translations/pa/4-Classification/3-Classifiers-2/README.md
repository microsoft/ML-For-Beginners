<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T07:09:41+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "pa"
}
-->
# ਖਾਣੇ ਦੇ ਵਰਗੀਕਰਨ 2

ਇਸ ਦੂਜੇ ਵਰਗੀਕਰਨ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਗਿਣਤੀ ਡਾਟਾ ਨੂੰ ਵਰਗੀਕਰਣ ਦੇ ਹੋਰ ਤਰੀਕੇ ਖੋਜੋਗੇ। ਤੁਸੀਂ ਇਹ ਵੀ ਸਿੱਖੋਗੇ ਕਿ ਇੱਕ ਵਰਗੀਕਰਣਕਰਤਾ ਦੀ ਚੋਣ ਕਰਨ ਦੇ ਨਤੀਜੇ ਕੀ ਹੋ ਸਕਦੇ ਹਨ।

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

### ਪੂਰਵ ਸ਼ਰਤ

ਅਸੀਂ ਮੰਨਦੇ ਹਾਂ ਕਿ ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠ ਪੂਰੇ ਕਰ ਲਏ ਹਨ ਅਤੇ ਤੁਹਾਡੇ `data` ਫੋਲਡਰ ਵਿੱਚ ਇੱਕ ਸਾਫ ਕੀਤਾ ਹੋਇਆ ਡਾਟਾਸੇਟ ਹੈ ਜਿਸਦਾ ਨਾਮ _cleaned_cuisines.csv_ ਹੈ, ਜੋ ਕਿ ਇਸ 4-ਪਾਠਾਂ ਵਾਲੇ ਫੋਲਡਰ ਦੇ ਰੂਟ ਵਿੱਚ ਹੈ।

### ਤਿਆਰੀ

ਅਸੀਂ ਤੁਹਾਡੀ _notebook.ipynb_ ਫਾਈਲ ਨੂੰ ਸਾਫ ਕੀਤੇ ਡਾਟਾਸੇਟ ਨਾਲ ਲੋਡ ਕੀਤਾ ਹੈ ਅਤੇ ਇਸਨੂੰ X ਅਤੇ y ਡਾਟਾਫਰੇਮ ਵਿੱਚ ਵੰਡਿਆ ਹੈ, ਮਾਡਲ ਬਣਾਉਣ ਦੀ ਪ੍ਰਕਿਰਿਆ ਲਈ ਤਿਆਰ।

## ਵਰਗੀਕਰਨ ਦਾ ਨਕਸ਼ਾ

ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਸਿੱਖਿਆ ਕਿ ਡਾਟਾ ਨੂੰ ਵਰਗੀਕਰਣ ਲਈ ਤੁਹਾਡੇ ਕੋਲ ਕਿਹੜੇ ਵਿਕਲਪ ਹਨ, ਜਿਸ ਵਿੱਚ ਮਾਈਕਰੋਸਾਫਟ ਦੀ ਚੀਟ ਸ਼ੀਟ ਦੀ ਵਰਤੋਂ ਕੀਤੀ ਗਈ। Scikit-learn ਇੱਕ ਹੋਰ ਵਿਸਥਾਰਕ ਚੀਟ ਸ਼ੀਟ ਪੇਸ਼ ਕਰਦਾ ਹੈ ਜੋ ਤੁਹਾਡੇ ਵਰਗੀਕਰਣਕਰਤਾਵਾਂ (ਜਿਨ੍ਹਾਂ ਨੂੰ ਅਨੁਮਾਨਕਰਤਾ ਵੀ ਕਿਹਾ ਜਾਂਦਾ ਹੈ) ਨੂੰ ਚੁਣਨ ਵਿੱਚ ਹੋਰ ਮਦਦ ਕਰ ਸਕਦਾ ਹੈ:

![Scikit-learn ਤੋਂ ML ਨਕਸ਼ਾ](../../../../4-Classification/3-Classifiers-2/images/map.png)
> ਸੁਝਾਅ: [ਇਸ ਨਕਸ਼ੇ ਨੂੰ ਆਨਲਾਈਨ ਵੇਖੋ](https://scikit-learn.org/stable/tutorial/machine_learning_map/) ਅਤੇ ਦਸਤਾਵੇਜ਼ਾਂ ਨੂੰ ਪੜ੍ਹਨ ਲਈ ਰਾਹ 'ਤੇ ਕਲਿਕ ਕਰੋ।

### ਯੋਜਨਾ

ਇਹ ਨਕਸ਼ਾ ਤੁਹਾਡੇ ਡਾਟਾ ਦੀ ਸਪਸ਼ਟ ਸਮਝ ਹੋਣ 'ਤੇ ਬਹੁਤ ਮਦਦਗਾਰ ਹੈ, ਕਿਉਂਕਿ ਤੁਸੀਂ ਇਸਦੇ ਰਾਹਾਂ 'ਤੇ 'ਚੱਲ' ਸਕਦੇ ਹੋ:

- ਸਾਡੇ ਕੋਲ >50 ਨਮੂਨੇ ਹਨ
- ਅਸੀਂ ਇੱਕ ਸ਼੍ਰੇਣੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹਾਂ
- ਸਾਡੇ ਕੋਲ ਲੇਬਲਡ ਡਾਟਾ ਹੈ
- ਸਾਡੇ ਕੋਲ 100K ਤੋਂ ਘੱਟ ਨਮੂਨੇ ਹਨ
- ✨ ਅਸੀਂ ਇੱਕ Linear SVC ਚੁਣ ਸਕਦੇ ਹਾਂ
- ਜੇ ਇਹ ਕੰਮ ਨਹੀਂ ਕਰਦਾ, ਕਿਉਂਕਿ ਸਾਡੇ ਕੋਲ ਗਿਣਤੀ ਡਾਟਾ ਹੈ
    - ਅਸੀਂ ✨ KNeighbors Classifier ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰ ਸਕਦੇ ਹਾਂ 
      - ਜੇ ਇਹ ਕੰਮ ਨਹੀਂ ਕਰਦਾ, ✨ SVC ਅਤੇ ✨ Ensemble Classifiers ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ

ਇਹ ਪਾਲਣ ਕਰਨ ਲਈ ਬਹੁਤ ਹੀ ਮਦਦਗਾਰ ਰਾਹ ਹੈ।

## ਅਭਿਆਸ - ਡਾਟਾ ਵੰਡੋ

ਇਸ ਰਾਹ ਨੂੰ ਪਾਲਣ ਕਰਦੇ ਹੋਏ, ਸਾਨੂੰ ਕੁਝ ਲਾਇਬ੍ਰੇਰੀਆਂ ਨੂੰ ਇੰਪੋਰਟ ਕਰਨਾ ਸ਼ੁਰੂ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ।

1. ਲੋੜੀਂਦੀਆਂ ਲਾਇਬ੍ਰੇਰੀਆਂ ਇੰਪੋਰਟ ਕਰੋ:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. ਆਪਣੇ ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟ ਡਾਟਾ ਨੂੰ ਵੰਡੋ:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## ਲੀਨਿਅਰ SVC ਵਰਗੀਕਰਣਕਰਤਾ

ਸਪੋਰਟ-ਵੈਕਟਰ ਕਲਸਟਰਿੰਗ (SVC) ML ਤਕਨੀਕਾਂ ਦੇ ਸਪੋਰਟ-ਵੈਕਟਰ ਮਸ਼ੀਨ ਪਰਿਵਾਰ ਦਾ ਹਿੱਸਾ ਹੈ (ਹੇਠਾਂ ਇਸ ਬਾਰੇ ਹੋਰ ਜਾਣੋ)। ਇਸ ਵਿਧੀ ਵਿੱਚ, ਤੁਸੀਂ 'ਕਰਨਲ' ਚੁਣ ਸਕਦੇ ਹੋ ਕਿ ਲੇਬਲਾਂ ਨੂੰ ਕਿਵੇਂ ਕਲਸਟਰ ਕੀਤਾ ਜਾਵੇ। 'C' ਪੈਰਾਮੀਟਰ 'ਰੇਗੂਲਰਾਈਜ਼ੇਸ਼ਨ' ਨੂੰ ਦਰਸਾਉਂਦਾ ਹੈ ਜੋ ਪੈਰਾਮੀਟਰਾਂ ਦੇ ਪ੍ਰਭਾਵ ਨੂੰ ਨਿਯਮਿਤ ਕਰਦਾ ਹੈ। ਕਰਨਲ [ਕਈ](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) ਹੋ ਸਕਦੇ ਹਨ; ਇੱਥੇ ਅਸੀਂ ਇਸਨੂੰ 'ਲੀਨਿਅਰ' 'ਤੇ ਸੈਟ ਕਰਦੇ ਹਾਂ ਤਾਂ ਜੋ ਅਸੀਂ ਲੀਨਿਅਰ SVC ਦਾ ਲਾਭ ਲੈ ਸਕੀਏ। Probability ਡਿਫਾਲਟ 'false' ਹੈ; ਇੱਥੇ ਅਸੀਂ ਇਸਨੂੰ 'true' 'ਤੇ ਸੈਟ ਕਰਦੇ ਹਾਂ ਤਾਂ ਜੋ probability ਅੰਦਾਜ਼ੇ ਪ੍ਰਾਪਤ ਕੀਤੇ ਜਾ ਸਕਣ। ਅਸੀਂ random state ਨੂੰ '0' 'ਤੇ ਸੈਟ ਕਰਦੇ ਹਾਂ ਤਾਂ ਜੋ ਡਾਟਾ ਨੂੰ ਸ਼ਫਲ ਕਰਕੇ probabilities ਪ੍ਰਾਪਤ ਕੀਤੇ ਜਾ ਸਕਣ।

### ਅਭਿਆਸ - ਲੀਨਿਅਰ SVC ਲਾਗੂ ਕਰੋ

ਵਰਗੀਕਰਣਕਰਤਾਵਾਂ ਦੀ ਇੱਕ ਐਰੇ ਬਣਾਉਣ ਨਾਲ ਸ਼ੁਰੂ ਕਰੋ। ਜਦੋਂ ਅਸੀਂ ਟੈਸਟ ਕਰਦੇ ਹਾਂ, ਤੁਸੀਂ ਇਸ ਐਰੇ ਵਿੱਚ ਲਗਾਤਾਰ ਸ਼ਾਮਲ ਕਰਦੇ ਜਾਵੋਗੇ।

1. ਲੀਨਿਅਰ SVC ਨਾਲ ਸ਼ੁਰੂ ਕਰੋ:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਲੀਨਿਅਰ SVC ਨਾਲ ਟ੍ਰੇਨ ਕਰੋ ਅਤੇ ਇੱਕ ਰਿਪੋਰਟ ਪ੍ਰਿੰਟ ਕਰੋ:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    ਨਤੀਜਾ ਕਾਫੀ ਚੰਗਾ ਹੈ:

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

## K-Neighbors ਵਰਗੀਕਰਣਕਰਤਾ

K-Neighbors ML ਵਿਧੀਆਂ ਦੇ "neighbors" ਪਰਿਵਾਰ ਦਾ ਹਿੱਸਾ ਹੈ, ਜੋ ਕਿ ਸਪਰਵਾਈਜ਼ਡ ਅਤੇ ਅਨਸਪਰਵਾਈਜ਼ਡ ਲਰਨਿੰਗ ਦੋਵਾਂ ਲਈ ਵਰਤਿਆ ਜਾ ਸਕਦਾ ਹੈ। ਇਸ ਵਿਧੀ ਵਿੱਚ, ਇੱਕ ਪੂਰਵ-ਨਿਰਧਾਰਤ ਗਿਣਤੀ ਦੇ ਬਿੰਦੂ ਬਣਾਏ ਜਾਂਦੇ ਹਨ ਅਤੇ ਡਾਟਾ ਨੂੰ ਇਨ੍ਹਾਂ ਬਿੰਦੂਆਂ ਦੇ ਆਲੇ-ਦੁਆਲੇ ਇਕੱਠਾ ਕੀਤਾ ਜਾਂਦਾ ਹੈ ਤਾਂ ਜੋ ਡਾਟਾ ਲਈ ਜਨਰਲ ਲੇਬਲਾਂ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕੀਤੀ ਜਾ ਸਕੇ।

### ਅਭਿਆਸ - K-Neighbors ਵਰਗੀਕਰਣਕਰਤਾ ਲਾਗੂ ਕਰੋ

ਪਿਛਲਾ ਵਰਗੀਕਰਣਕਰਤਾ ਚੰਗਾ ਸੀ ਅਤੇ ਡਾਟਾ ਨਾਲ ਚੰਗਾ ਕੰਮ ਕੀਤਾ, ਪਰ ਸ਼ਾਇਦ ਅਸੀਂ ਹੋਰ ਵਧੀਆ ਸਹੀਤਾ ਪ੍ਰਾਪਤ ਕਰ ਸਕੀਏ। K-Neighbors ਵਰਗੀਕਰਣਕਰਤਾ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।

1. ਆਪਣੇ ਵਰਗੀਕਰਣਕਰਤਾ ਐਰੇ ਵਿੱਚ ਇੱਕ ਲਾਈਨ ਸ਼ਾਮਲ ਕਰੋ (Linear SVC ਆਈਟਮ ਦੇ ਬਾਅਦ ਇੱਕ ਕਾਮਾ ਸ਼ਾਮਲ ਕਰੋ):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    ਨਤੀਜਾ ਥੋੜਾ ਬੁਰਾ ਹੈ:

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

    ✅ [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors) ਬਾਰੇ ਸਿੱਖੋ

## ਸਪੋਰਟ ਵੈਕਟਰ ਵਰਗੀਕਰਣਕਰਤਾ

ਸਪੋਰਟ-ਵੈਕਟਰ ਵਰਗੀਕਰਣਕਰਤਾ [ਸਪੋਰਟ-ਵੈਕਟਰ ਮਸ਼ੀਨ](https://wikipedia.org/wiki/Support-vector_machine) ਪਰਿਵਾਰ ਦਾ ਹਿੱਸਾ ਹੈ ਜੋ ਵਰਗੀਕਰਣ ਅਤੇ ਰਿਗਰੈਸ਼ਨ ਕਾਰਜਾਂ ਲਈ ਵਰਤਿਆ ਜਾਂਦਾ ਹੈ। SVMs "ਟ੍ਰੇਨਿੰਗ ਉਦਾਹਰਣਾਂ ਨੂੰ ਸਪੇਸ ਵਿੱਚ ਬਿੰਦੂਆਂ 'ਤੇ ਮੈਪ ਕਰਦੇ ਹਨ" ਤਾਂ ਜੋ ਦੋ ਸ਼੍ਰੇਣੀਆਂ ਦੇ ਵਿਚਕਾਰ ਦੂਰੀ ਵਧਾਈ ਜਾ ਸਕੇ। ਬਾਅਦ ਦਾ ਡਾਟਾ ਇਸ ਸਪੇਸ ਵਿੱਚ ਮੈਪ ਕੀਤਾ ਜਾਂਦਾ ਹੈ ਤਾਂ ਜੋ ਇਸਦੀ ਸ਼੍ਰੇਣੀ ਦੀ ਭਵਿੱਖਵਾਣੀ ਕੀਤੀ ਜਾ ਸਕੇ।

### ਅਭਿਆਸ - ਸਪੋਰਟ ਵੈਕਟਰ ਵਰਗੀਕਰਣਕਰਤਾ ਲਾਗੂ ਕਰੋ

ਚੰਗੀ ਸਹੀਤਾ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਸਪੋਰਟ ਵੈਕਟਰ ਵਰਗੀਕਰਣਕਰਤਾ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ।

1. K-Neighbors ਆਈਟਮ ਦੇ ਬਾਅਦ ਇੱਕ ਕਾਮਾ ਸ਼ਾਮਲ ਕਰੋ, ਅਤੇ ਫਿਰ ਇਹ ਲਾਈਨ ਸ਼ਾਮਲ ਕਰੋ:

    ```python
    'SVC': SVC(),
    ```

    ਨਤੀਜਾ ਕਾਫੀ ਚੰਗਾ ਹੈ!

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

    ✅ [ਸਪੋਰਟ-ਵੈਕਟਰ](https://scikit-learn.org/stable/modules/svm.html#svm) ਬਾਰੇ ਸਿੱਖੋ

## ਐਨਸੈਂਬਲ ਵਰਗੀਕਰਣਕਰਤਾ

ਆਓ ਰਾਹ ਦੇ ਬਿਲਕੁਲ ਅੰਤ ਤੱਕ ਪਹੁੰਚੀਏ, ਹਾਲਾਂਕਿ ਪਿਛਲਾ ਟੈਸਟ ਕਾਫੀ ਚੰਗਾ ਸੀ। ਆਓ ਕੁਝ 'ਐਨਸੈਂਬਲ ਵਰਗੀਕਰਣਕਰਤਾ' ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੀਏ, ਖਾਸ ਤੌਰ 'ਤੇ Random Forest ਅਤੇ AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

ਨਤੀਜਾ ਬਹੁਤ ਚੰਗਾ ਹੈ, ਖਾਸ ਤੌਰ 'ਤੇ Random Forest ਲਈ:

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

✅ [ਐਨਸੈਂਬਲ ਵਰਗੀਕਰਣਕਰਤਾ](https://scikit-learn.org/stable/modules/ensemble.html) ਬਾਰੇ ਸਿੱਖੋ

ਇਹ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਦੀ ਵਿਧੀ "ਕਈ ਬੇਸ ਅਨੁਮਾਨਕਰਤਾਵਾਂ ਦੀ ਭਵਿੱਖਵਾਣੀ ਨੂੰ ਜੋੜਦੀ ਹੈ" ਤਾਂ ਜੋ ਮਾਡਲ ਦੀ ਗੁਣਵੱਤਾ ਵਿੱਚ ਸੁਧਾਰ ਕੀਤਾ ਜਾ ਸਕੇ। ਸਾਡੇ ਉਦਾਹਰਣ ਵਿੱਚ, ਅਸੀਂ Random Trees ਅਤੇ AdaBoost ਦੀ ਵਰਤੋਂ ਕੀਤੀ।

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), ਇੱਕ ਐਵਰੇਜਿੰਗ ਵਿਧੀ, 'decision trees' ਦਾ ਇੱਕ 'forest' ਬਣਾਉਂਦਾ ਹੈ ਜਿਸ ਵਿੱਚ randomness ਸ਼ਾਮਲ ਹੁੰਦੀ ਹੈ ਤਾਂ ਜੋ overfitting ਤੋਂ ਬਚਿਆ ਜਾ ਸਕੇ। n_estimators ਪੈਰਾਮੀਟਰ ਨੂੰ ਰੁੱਖਾਂ ਦੀ ਗਿਣਤੀ 'ਤੇ ਸੈਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ।

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) ਇੱਕ ਵਰਗੀਕਰਣਕਰਤਾ ਨੂੰ ਡਾਟਾਸੇਟ 'ਤੇ ਫਿੱਟ ਕਰਦਾ ਹੈ ਅਤੇ ਫਿਰ ਉਸ ਵਰਗੀਕਰਣਕਰਤਾ ਦੀਆਂ ਕਾਪੀਆਂ ਨੂੰ ਉਸੇ ਡਾਟਾਸੇਟ 'ਤੇ ਫਿੱਟ ਕਰਦਾ ਹੈ। ਇਹ ਗਲਤ ਵਰਗੀਕਰਿਤ ਆਈਟਮਾਂ ਦੇ ਵਜ਼ਨਾਂ 'ਤੇ ਧਿਆਨ ਕੇਂਦਰਿਤ ਕਰਦਾ ਹੈ ਅਤੇ ਅਗਲੇ ਵਰਗੀਕਰਣਕਰਤਾ ਲਈ ਫਿੱਟ ਨੂੰ ਠੀਕ ਕਰਨ ਲਈ ਸਮਰੂਪ ਕਰਦਾ ਹੈ।

---

## 🚀ਚੁਣੌਤੀ

ਇਨ੍ਹਾਂ ਤਕਨੀਕਾਂ ਵਿੱਚ ਬਹੁਤ ਸਾਰੇ ਪੈਰਾਮੀਟਰ ਹਨ ਜਿਨ੍ਹਾਂ ਨੂੰ ਤੁਸੀਂ ਬਦਲ ਸਕਦੇ ਹੋ। ਹਰ ਇੱਕ ਦੇ ਡਿਫਾਲਟ ਪੈਰਾਮੀਟਰਾਂ ਬਾਰੇ ਖੋਜ ਕਰੋ ਅਤੇ ਸੋਚੋ ਕਿ ਇਹ ਪੈਰਾਮੀਟਰਾਂ ਨੂੰ ਬਦਲਣ ਦਾ ਮਾਡਲ ਦੀ ਗੁਣਵੱਤਾ 'ਤੇ ਕੀ ਅਸਰ ਹੋਵੇਗਾ।

## [ਪੋਸਟ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

ਇਨ੍ਹਾਂ ਪਾਠਾਂ ਵਿੱਚ ਬਹੁਤ ਸਾਰਾ ਜਾਰਗਨ ਹੈ, ਇਸ ਲਈ [ਇਸ ਸੂਚੀ](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) ਨੂੰ ਸਮੀਖਿਆ ਕਰਨ ਲਈ ਇੱਕ ਮਿੰਟ ਲਓ ਜਿਸ ਵਿੱਚ ਉਪਯੋਗ ਸ਼ਬਦਾਵਲੀ ਹੈ!

## ਅਸਾਈਨਮੈਂਟ 

[ਪੈਰਾਮੀਟਰ ਖੇਡ](assignment.md)

---

**ਅਸਵੀਕਾਰਨਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਹਾਲਾਂਕਿ ਅਸੀਂ ਸਹੀਅਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਨਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।