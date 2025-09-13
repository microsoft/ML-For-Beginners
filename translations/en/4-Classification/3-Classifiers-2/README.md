<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "49047911108adc49d605cddfb455749c",
  "translation_date": "2025-09-06T10:57:01+00:00",
  "source_file": "4-Classification/3-Classifiers-2/README.md",
  "language_code": "en"
}
-->
# Cuisine classifiers 2

In this second classification lesson, you will explore additional methods for classifying numeric data. You will also learn about the implications of choosing one classifier over another.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Prerequisite

We assume that you have completed the previous lessons and have a cleaned dataset in your `data` folder named _cleaned_cuisines.csv_ in the root of this 4-lesson folder.

### Preparation

We have preloaded your _notebook.ipynb_ file with the cleaned dataset and divided it into X and y dataframes, ready for the model-building process.

## A classification map

Previously, you learned about the various options available for classifying data using Microsoft's cheat sheet. Scikit-learn provides a similar but more detailed cheat sheet that can help you further narrow down your choice of estimators (another term for classifiers):

![ML Map from Scikit-learn](../../../../4-Classification/3-Classifiers-2/images/map.png)
> Tip: [visit this map online](https://scikit-learn.org/stable/tutorial/machine_learning_map/) and click along the path to read documentation.

### The plan

This map is very useful once you have a clear understanding of your data, as you can follow its paths to make a decision:

- We have >50 samples
- We want to predict a category
- We have labeled data
- We have fewer than 100K samples
- ✨ We can choose a Linear SVC
- If that doesn't work, since we have numeric data:
    - We can try a ✨ KNeighbors Classifier 
      - If that doesn't work, try ✨ SVC and ✨ Ensemble Classifiers

This is a very helpful guide to follow.

## Exercise - split the data

Following this path, we should start by importing some libraries for use.

1. Import the necessary libraries:

    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    import numpy as np
    ```

1. Split your training and test data:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
    ```

## Linear SVC classifier

Support-Vector Clustering (SVC) is part of the Support-Vector Machines family of ML techniques (learn more about these below). In this method, you can choose a 'kernel' to determine how to cluster the labels. The 'C' parameter refers to 'regularization,' which controls the influence of parameters. The kernel can be one of [several](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC); here, we set it to 'linear' to ensure we use linear SVC. Probability defaults to 'false'; here, we set it to 'true' to gather probability estimates. We set the random state to '0' to shuffle the data and obtain probabilities.

### Exercise - apply a linear SVC

Start by creating an array of classifiers. You will add to this array progressively as we test.

1. Start with a Linear SVC:

    ```python
    C = 10
    # Create different classifiers.
    classifiers = {
        'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
    }
    ```

2. Train your model using the Linear SVC and print out a report:

    ```python
    n_classifiers = len(classifiers)
    
    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X_train, np.ravel(y_train))
    
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        print(classification_report(y_test,y_pred))
    ```

    The result is quite good:

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

K-Neighbors belongs to the "neighbors" family of ML methods, which can be used for both supervised and unsupervised learning. In this method, a predefined number of points is created, and data is grouped around these points so that generalized labels can be predicted for the data.

### Exercise - apply the K-Neighbors classifier

The previous classifier performed well with the data, but perhaps we can achieve better accuracy. Try a K-Neighbors classifier.

1. Add a line to your classifier array (add a comma after the Linear SVC item):

    ```python
    'KNN classifier': KNeighborsClassifier(C),
    ```

    The result is slightly worse:

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

    ✅ Learn about [K-Neighbors](https://scikit-learn.org/stable/modules/neighbors.html#neighbors)

## Support Vector Classifier

Support-Vector Classifiers are part of the [Support-Vector Machine](https://wikipedia.org/wiki/Support-vector_machine) family of ML methods used for classification and regression tasks. SVMs "map training examples to points in space" to maximize the distance between two categories. Subsequent data is mapped into this space so their category can be predicted.

### Exercise - apply a Support Vector Classifier

Let's aim for slightly better accuracy with a Support Vector Classifier.

1. Add a comma after the K-Neighbors item, and then add this line:

    ```python
    'SVC': SVC(),
    ```

    The result is quite good!

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

    ✅ Learn about [Support-Vectors](https://scikit-learn.org/stable/modules/svm.html#svm)

## Ensemble Classifiers

Let's follow the path to the very end, even though the previous test performed well. Let's try some 'Ensemble Classifiers,' specifically Random Forest and AdaBoost:

```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```

The result is excellent, especially for Random Forest:

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

✅ Learn about [Ensemble Classifiers](https://scikit-learn.org/stable/modules/ensemble.html)

This Machine Learning method "combines the predictions of several base estimators" to improve the model's quality. In our example, we used Random Trees and AdaBoost. 

- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest), an averaging method, builds a 'forest' of 'decision trees' infused with randomness to avoid overfitting. The n_estimators parameter specifies the number of trees.

- [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) fits a classifier to a dataset and then fits copies of that classifier to the same dataset. It focuses on the weights of incorrectly classified items and adjusts the fit for the next classifier to correct.

---

## 🚀Challenge

Each of these techniques has numerous parameters that you can adjust. Research the default parameters for each one and consider how tweaking these parameters might affect the model's quality.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

There is a lot of terminology in these lessons, so take a moment to review [this list](https://docs.microsoft.com/dotnet/machine-learning/resources/glossary?WT.mc_id=academic-77952-leestott) of useful terms!

## Assignment 

[Parameter play](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.