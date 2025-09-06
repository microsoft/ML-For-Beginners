<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T10:55:54+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "en"
}
-->
# Cuisine classifiers 1

In this lesson, you will use the dataset you saved from the previous lesson, which contains balanced and clean data about cuisines.

You will use this dataset with various classifiers to _predict the national cuisine based on a set of ingredients_. Along the way, you'll learn more about how algorithms can be applied to classification tasks.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Preparation

If you completed [Lesson 1](../1-Introduction/README.md), ensure that a _cleaned_cuisines.csv_ file exists in the root `/data` folder for these four lessons.

## Exercise - predict a national cuisine

1. In this lesson's _notebook.ipynb_ folder, import the file along with the Pandas library:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    The data looks like this:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Next, import several additional libraries:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Separate the X and y coordinates into two dataframes for training. Use `cuisine` as the labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    It will look like this:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Drop the `Unnamed: 0` column and the `cuisine` column using `drop()`. Save the remaining data as trainable features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Your features will look like this:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Now you're ready to train your model!

## Choosing your classifier

With your data clean and ready for training, it's time to decide which algorithm to use for the task.

Scikit-learn categorizes classification under Supervised Learning, offering a wide range of classification methods. [The options](https://scikit-learn.org/stable/supervised_learning.html) can seem overwhelming at first glance. These methods include:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

> You can also use [neural networks for classification](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), but that is beyond the scope of this lesson.

### Which classifier should you choose?

So, how do you decide on a classifier? Often, testing several options and comparing results is a good approach. Scikit-learn provides a [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) on a sample dataset, showcasing KNeighbors, SVC (two variations), GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB, and QuadraticDiscriminationAnalysis, with visualized results:

![comparison of classifiers](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Plots generated from Scikit-learn's documentation

> AutoML simplifies this process by running these comparisons in the cloud, helping you select the best algorithm for your data. Try it [here](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### A more informed approach

Instead of guessing, you can refer to this downloadable [ML Cheat Sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). For our multiclass problem, it suggests several options:

![cheatsheet for multiclass problems](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> A section of Microsoft's Algorithm Cheat Sheet, detailing multiclass classification options

✅ Download this cheat sheet, print it out, and keep it handy!

### Reasoning

Let's evaluate different approaches based on our constraints:

- **Neural networks are too resource-intensive**. Given our clean but small dataset and the fact that we're training locally in notebooks, neural networks are not ideal for this task.
- **Avoid two-class classifiers**. Since this is not a binary classification problem, two-class classifiers like one-vs-all are not suitable.
- **Decision tree or logistic regression could work**. Both decision trees and logistic regression are viable options for multiclass data.
- **Multiclass Boosted Decision Trees are not suitable**. These are better for nonparametric tasks like ranking, which is not relevant here.

### Using Scikit-learn 

We'll use Scikit-learn to analyze our data. Logistic regression in Scikit-learn offers several options. Check out the [parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) you can configure.

Two key parameters to set are `multi_class` and `solver`. These determine the behavior and algorithm used for logistic regression. Not all solvers are compatible with all `multi_class` values.

According to the documentation, for multiclass classification:

- **The one-vs-rest (OvR) scheme** is used if `multi_class` is set to `ovr`.
- **Cross-entropy loss** is used if `multi_class` is set to `multinomial`. (The `multinomial` option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’, and ‘newton-cg’ solvers.)

> 🎓 The 'scheme' refers to how logistic regression handles multiclass classification. It can be 'ovr' (one-vs-rest) or 'multinomial'. These schemes adapt logistic regression, which is primarily designed for binary classification, to handle multiclass tasks. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 The 'solver' is the algorithm used to optimize the problem. [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn provides this table to explain how solvers handle different challenges based on data structures:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## Exercise - split the data

Let's start with logistic regression for our first training attempt, as you recently learned about it in a previous lesson.
Split your data into training and testing sets using `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercise - apply logistic regression

Since this is a multiclass problem, you need to choose a _scheme_ and a _solver_. Use LogisticRegression with a multiclass setting and the **liblinear** solver for training.

1. Create a logistic regression model with `multi_class` set to `ovr` and the solver set to `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Try using a different solver like `lbfgs`, which is often the default option.
> Note, use Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function to flatten your data when needed.
The accuracy is good at over **80%**!

1. You can see this model in action by testing one row of data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    The result is printed:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ Try a different row number and check the results.

1. Digging deeper, you can check the accuracy of this prediction:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    The result is printed - Indian cuisine is its best guess, with good probability:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ Can you explain why the model is quite confident this is an Indian cuisine?

1. Get more detail by printing a classification report, as you did in the regression lessons:

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

## 🚀Challenge

In this lesson, you used your cleaned data to build a machine learning model that can predict a national cuisine based on a series of ingredients. Take some time to read through the many options Scikit-learn provides to classify data. Dig deeper into the concept of 'solver' to understand what goes on behind the scenes.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Dig a little more into the math behind logistic regression in [this lesson](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)

## Assignment 

[Study the solvers](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.