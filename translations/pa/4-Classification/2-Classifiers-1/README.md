<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-09-06T07:08:07+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "pa"
}
-->
# ਖਾਣੇ ਦੇ ਵਰਗ 1

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਸੇਵ ਕੀਤੇ ਡਾਟਾਸੈਟ ਨੂੰ ਵਰਤੋਂਗੇ, ਜੋ ਖਾਣੇ ਬਾਰੇ ਸੰਤੁਲਿਤ ਅਤੇ ਸਾਫ ਡਾਟਾ ਨਾਲ ਭਰਿਆ ਹੋਇਆ ਹੈ।

ਤੁਸੀਂ ਇਸ ਡਾਟਾਸੈਟ ਨੂੰ ਵੱਖ-ਵੱਖ ਵਰਗਬੱਧ ਕਰਨ ਵਾਲੇ ਤਰੀਕਿਆਂ ਨਾਲ ਵਰਤੋਂਗੇ ਤਾਂ ਜੋ _ਸਮੂਹ ਦੇ ਸਮੱਗਰੀ ਦੇ ਆਧਾਰ 'ਤੇ ਦਿੱਤੇ ਗਏ ਰਾਸ਼ਟਰੀ ਖਾਣੇ ਦੀ ਪੇਸ਼ਗੂਈ ਕੀਤੀ ਜਾ ਸਕੇ_। ਇਸ ਦੌਰਾਨ, ਤੁਸੀਂ ਇਹ ਵੀ ਸਿੱਖੋਗੇ ਕਿ ਵਰਗਬੱਧ ਕਰਨ ਵਾਲੇ ਕੰਮਾਂ ਲਈ ਅਲਗੋਰਿਥਮ ਨੂੰ ਕਿਵੇਂ ਵਰਤਿਆ ਜਾ ਸਕਦਾ ਹੈ।

## [ਪ੍ਰੀ-ਪਾਠ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)
# ਤਿਆਰੀ

ਜੇਕਰ ਤੁਸੀਂ [ਪਾਠ 1](../1-Introduction/README.md) ਪੂਰਾ ਕੀਤਾ ਹੈ, ਤਾਂ ਯਕੀਨੀ ਬਣਾਓ ਕਿ _cleaned_cuisines.csv_ ਫਾਈਲ `/data` ਫੋਲਡਰ ਦੇ ਰੂਟ ਵਿੱਚ ਮੌਜੂਦ ਹੈ, ਜੋ ਕਿ ਇਹਨਾਂ ਚਾਰ ਪਾਠਾਂ ਲਈ ਹੈ।

## ਅਭਿਆਸ - ਰਾਸ਼ਟਰੀ ਖਾਣੇ ਦੀ ਪੇਸ਼ਗੂਈ ਕਰੋ

1. ਇਸ ਪਾਠ ਦੇ _notebook.ipynb_ ਫੋਲਡਰ ਵਿੱਚ ਕੰਮ ਕਰਦੇ ਹੋਏ, ਉਸ ਫਾਈਲ ਨੂੰ Pandas ਲਾਇਬ੍ਰੇਰੀ ਦੇ ਨਾਲ ਇੰਪੋਰਟ ਕਰੋ:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    ਡਾਟਾ ਇਸ ਤਰ੍ਹਾਂ ਦਿਖਾਈ ਦਿੰਦਾ ਹੈ:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. ਹੁਣ, ਹੋਰ ਕਈ ਲਾਇਬ੍ਰੇਰੀਆਂ ਇੰਪੋਰਟ ਕਰੋ:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. X ਅਤੇ y ਕੋਆਰਡੀਨੇਟ ਨੂੰ ਦੋ ਡਾਟਾਫਰੇਮ ਵਿੱਚ ਵੰਡੋ। `cuisine` ਲੇਬਲ ਡਾਟਾਫਰੇਮ ਹੋ ਸਕਦਾ ਹੈ:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    ਇਹ ਇਸ ਤਰ੍ਹਾਂ ਦਿਖਾਈ ਦੇਵੇਗਾ:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. `Unnamed: 0` ਕਾਲਮ ਅਤੇ `cuisine` ਕਾਲਮ ਨੂੰ ਹਟਾਓ, `drop()` ਕਾਲ ਕਰਕੇ। ਬਾਕੀ ਡਾਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਫੀਚਰ ਵਜੋਂ ਸੇਵ ਕਰੋ:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    ਤੁਹਾਡੇ ਫੀਚਰ ਇਸ ਤਰ੍ਹਾਂ ਦਿਖਾਈ ਦਿੰਦੇ ਹਨ:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

ਹੁਣ ਤੁਸੀਂ ਆਪਣੇ ਮਾਡਲ ਨੂੰ ਟ੍ਰੇਨ ਕਰਨ ਲਈ ਤਿਆਰ ਹੋ!

## ਆਪਣੇ ਵਰਗਬੱਧ ਕਰਨ ਵਾਲੇ ਤਰੀਕੇ ਦੀ ਚੋਣ

ਹੁਣ ਜਦੋਂ ਤੁਹਾਡਾ ਡਾਟਾ ਸਾਫ ਅਤੇ ਟ੍ਰੇਨਿੰਗ ਲਈ ਤਿਆਰ ਹੈ, ਤੁਹਾਨੂੰ ਇਹ ਫੈਸਲਾ ਕਰਨਾ ਪਵੇਗਾ ਕਿ ਇਸ ਕੰਮ ਲਈ ਕਿਹੜਾ ਅਲਗੋਰਿਥਮ ਵਰਤਣਾ ਹੈ।

Scikit-learn ਵਰਗਬੱਧ ਕਰਨ ਨੂੰ Supervised Learning ਦੇ ਅਧੀਨ ਰੱਖਦਾ ਹੈ, ਅਤੇ ਇਸ ਸ਼੍ਰੇਣੀ ਵਿੱਚ ਤੁਹਾਨੂੰ ਵਰਗਬੱਧ ਕਰਨ ਦੇ ਕਈ ਤਰੀਕੇ ਮਿਲਣਗੇ। [ਵੱਖ-ਵੱਖ ਤਰੀਕੇ](https://scikit-learn.org/stable/supervised_learning.html) ਪਹਿਲਾਂ ਦ੍ਰਿਸ਼ਟੀ ਵਿੱਚ ਬਹੁਤ ਹੀ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਲੱਗਦੇ ਹਨ। ਹੇਠਾਂ ਦਿੱਤੇ ਤਰੀਕੇ ਸਾਰੇ ਵਰਗਬੱਧ ਕਰਨ ਦੇ ਤਕਨੀਕੀ ਤਰੀਕੇ ਸ਼ਾਮਲ ਕਰਦੇ ਹਨ:

- ਲੀਨੀਅਰ ਮਾਡਲ
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

> ਤੁਸੀਂ [neural networks ਨੂੰ ਵਰਗਬੱਧ ਕਰਨ ਲਈ](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification) ਵੀ ਵਰਤ ਸਕਦੇ ਹੋ, ਪਰ ਇਹ ਪਾਠ ਦੇ ਦਾਇਰੇ ਤੋਂ ਬਾਹਰ ਹੈ।

### ਕਿਹੜਾ ਵਰਗਬੱਧ ਕਰਨ ਵਾਲਾ ਤਰੀਕਾ ਚੁਣਨਾ ਹੈ?

ਤਾਂ, ਤੁਸੀਂ ਕਿਹੜਾ ਤਰੀਕਾ ਚੁਣਨਾ ਚਾਹੁੰਦੇ ਹੋ? ਅਕਸਰ, ਕਈ ਤਰੀਕਿਆਂ ਨੂੰ ਚਲਾਉਣਾ ਅਤੇ ਚੰਗੇ ਨਤੀਜੇ ਦੀ ਭਾਲ ਕਰਨਾ ਇੱਕ ਤਰੀਕਾ ਹੁੰਦਾ ਹੈ। Scikit-learn ਇੱਕ [ਸਾਈਡ-ਬਾਈ-ਸਾਈਡ ਤੁਲਨਾ](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) ਪੇਸ਼ ਕਰਦਾ ਹੈ, ਜੋ ਕਿ KNeighbors, SVC ਦੋ ਤਰੀਕੇ, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB ਅਤੇ QuadraticDiscrinationAnalysis ਦੀ ਤੁਲਨਾ ਦਿਖਾਉਂਦਾ ਹੈ:

![ਵਰਗਬੱਧ ਕਰਨ ਵਾਲਿਆਂ ਦੀ ਤੁਲਨਾ](../../../../4-Classification/2-Classifiers-1/images/comparison.png)
> Plots Scikit-learn ਦੇ ਦਸਤਾਵੇਜ਼ਾਂ 'ਤੇ ਬਣਾਏ ਗਏ

> AutoML ਇਸ ਸਮੱਸਿਆ ਨੂੰ ਬਹੁਤ ਹੀ ਸੌਖੇ ਤਰੀਕੇ ਨਾਲ ਹੱਲ ਕਰਦਾ ਹੈ, ਕਲਾਉਡ ਵਿੱਚ ਇਹ ਤੁਲਨਾਵਾਂ ਚਲਾਉਂਦਾ ਹੈ, ਅਤੇ ਤੁਹਾਨੂੰ ਤੁਹਾਡੇ ਡਾਟਾ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ ਅਲਗੋਰਿਥਮ ਚੁਣਨ ਦੀ ਆਗਿਆ ਦਿੰਦਾ ਹੈ। ਇਸਨੂੰ [ਇੱਥੇ](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott) ਅਜ਼ਮਾਓ।

### ਇੱਕ ਵਧੀਆ ਤਰੀਕਾ

ਅੰਧੇਵਾਹ ਚੋਣ ਕਰਨ ਦੇ ਬਦਲੇ, ਇੱਕ ਵਧੀਆ ਤਰੀਕਾ ਇਹ ਹੈ ਕਿ ਇਸ ਡਾਊਨਲੋਡ ਕਰਨ ਯੋਗ [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott) ਦੇ ਵਿਚਾਰਾਂ ਦੀ ਪਾਲਣਾ ਕਰੋ। ਇੱਥੇ, ਅਸੀਂ ਪਤਾ ਲਗਾਉਂਦੇ ਹਾਂ ਕਿ ਸਾਡੇ multiclass ਸਮੱਸਿਆ ਲਈ ਸਾਡੇ ਕੋਲ ਕੁਝ ਚੋਣਾਂ ਹਨ:

![multiclass ਸਮੱਸਿਆਵਾਂ ਲਈ cheatsheet](../../../../4-Classification/2-Classifiers-1/images/cheatsheet.png)
> Microsoft's Algorithm Cheat Sheet ਦਾ ਇੱਕ ਹਿੱਸਾ, ਜੋ multiclass ਵਰਗਬੱਧ ਕਰਨ ਦੇ ਵਿਕਲਪਾਂ ਨੂੰ ਵੇਖਾਉਂਦਾ ਹੈ

✅ ਇਸ cheatsheet ਨੂੰ ਡਾਊਨਲੋਡ ਕਰੋ, ਪ੍ਰਿੰਟ ਕਰੋ, ਅਤੇ ਇਸਨੂੰ ਆਪਣੀ ਕੰਮ ਵਾਲੀ ਜਗ੍ਹਾ 'ਤੇ ਲਟਕਾਓ!

### ਤਰਕ

ਆਓ ਵੇਖੀਏ ਕਿ ਕੀ ਅਸੀਂ ਆਪਣੇ ਪਾਬੰਦੀਆਂ ਦੇ ਆਧਾਰ 'ਤੇ ਵੱਖ-ਵੱਖ ਤਰੀਕਿਆਂ ਨੂੰ ਸਮਝ ਸਕਦੇ ਹਾਂ:

- **Neural networks ਬਹੁਤ ਭਾਰੀ ਹਨ**। ਸਾਡੇ ਸਾਫ, ਪਰ ਘੱਟ ਡਾਟਾਸੈਟ ਨੂੰ ਦੇਖਦੇ ਹੋਏ, ਅਤੇ ਇਹ ਹਕੀਕਤ ਕਿ ਅਸੀਂ ਟ੍ਰੇਨਿੰਗ ਨੂੰ ਸਥਾਨਕ ਤੌਰ 'ਤੇ notebooks ਰਾਹੀਂ ਚਲਾ ਰਹੇ ਹਾਂ, neural networks ਇਸ ਕੰਮ ਲਈ ਬਹੁਤ ਭਾਰੀ ਹਨ।
- **ਕੋਈ ਦੋ-ਵਰਗਬੱਧ ਕਰਨ ਵਾਲਾ ਤਰੀਕਾ ਨਹੀਂ**। ਅਸੀਂ ਦੋ-ਵਰਗਬੱਧ ਕਰਨ ਵਾਲਾ ਤਰੀਕਾ ਨਹੀਂ ਵਰਤਦੇ, ਇਸ ਲਈ ਇਹ one-vs-all ਨੂੰ ਰੱਦ ਕਰਦਾ ਹੈ।
- **Decision tree ਜਾਂ logistic regression ਕੰਮ ਕਰ ਸਕਦੇ ਹਨ**। ਇੱਕ decision tree ਕੰਮ ਕਰ ਸਕਦਾ ਹੈ, ਜਾਂ multiclass ਡਾਟਾ ਲਈ logistic regression।
- **Multiclass Boosted Decision Trees ਵੱਖਰੀ ਸਮੱਸਿਆ ਹੱਲ ਕਰਦੇ ਹਨ**। Multiclass Boosted Decision Trees ਜ਼ਿਆਦਾਤਰ nonparametric ਕੰਮਾਂ ਲਈ ਉਚਿਤ ਹਨ, ਜਿਵੇਂ ਕਿ ਰੈਂਕਿੰਗ ਬਣਾਉਣ ਲਈ ਕੰਮ, ਇਸ ਲਈ ਇਹ ਸਾਡੇ ਲਈ ਲਾਭਕਾਰੀ ਨਹੀਂ ਹੈ।

### Scikit-learn ਦੀ ਵਰਤੋਂ

ਅਸੀਂ Scikit-learn ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਆਪਣੇ ਡਾਟਾ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰਾਂਗੇ। ਹਾਲਾਂਕਿ, Scikit-learn ਵਿੱਚ logistic regression ਦੀ ਵਰਤੋਂ ਕਰਨ ਦੇ ਕਈ ਤਰੀਕੇ ਹਨ। [ਪਾਸ ਕਰਨ ਲਈ ਪੈਰਾਮੀਟਰਾਂ](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression) 'ਤੇ ਇੱਕ ਨਜ਼ਰ ਮਾਰੋ।

ਅਸਲ ਵਿੱਚ, ਦੋ ਮਹੱਤਵਪੂਰਨ ਪੈਰਾਮੀਟਰ ਹਨ - `multi_class` ਅਤੇ `solver` - ਜਿਨ੍ਹਾਂ ਨੂੰ ਸਪਸ਼ਟ ਕਰਨ ਦੀ ਲੋੜ ਹੈ, ਜਦੋਂ ਅਸੀਂ Scikit-learn ਨੂੰ logistic regression ਕਰਨ ਲਈ ਕਹਿੰਦੇ ਹਾਂ। `multi_class` ਦਾ ਮੁੱਲ ਇੱਕ ਨਿਰਧਾਰਿਤ ਵਿਹਾਰ ਲਾਗੂ ਕਰਦਾ ਹੈ। solver ਦਾ ਮੁੱਲ ਇਹ ਹੈ ਕਿ ਕਿਹੜਾ ਅਲਗੋਰਿਥਮ ਵਰਤਣਾ ਹੈ। ਸਾਰੇ solvers ਨੂੰ ਸਾਰੇ `multi_class` ਮੁੱਲਾਂ ਨਾਲ ਜੋੜਿਆ ਨਹੀਂ ਜਾ ਸਕਦਾ।

ਦਸਤਾਵੇਜ਼ਾਂ ਦੇ ਅਨੁਸਾਰ, multiclass ਕੇਸ ਵਿੱਚ, ਟ੍ਰੇਨਿੰਗ ਅਲਗੋਰਿਥਮ:

- **Uses the one-vs-rest (OvR) scheme**, ਜੇ `multi_class` ਵਿਕਲਪ `ovr` 'ਤੇ ਸੈਟ ਕੀਤਾ ਗਿਆ ਹੈ।
- **Uses the cross-entropy loss**, ਜੇ `multi_class` ਵਿਕਲਪ `multinomial` 'ਤੇ ਸੈਟ ਕੀਤਾ ਗਿਆ ਹੈ। (ਵਰਤਮਾਨ ਵਿੱਚ `multinomial` ਵਿਕਲਪ ਸਿਰਫ ‘lbfgs’, ‘sag’, ‘saga’ ਅਤੇ ‘newton-cg’ solvers ਦੁਆਰਾ ਸਮਰਥਿਤ ਹੈ।)

> 🎓 'scheme' ਇੱਥੇ 'ovr' (one-vs-rest) ਜਾਂ 'multinomial' ਹੋ ਸਕਦਾ ਹੈ। ਕਿਉਂਕਿ logistic regression ਅਸਲ ਵਿੱਚ binary classification ਨੂੰ ਸਮਰਥਨ ਕਰਨ ਲਈ ਡਿਜ਼ਾਈਨ ਕੀਤਾ ਗਿਆ ਹੈ, ਇਹ schemes ਇਸਨੂੰ multiclass classification ਕੰਮਾਂ ਨੂੰ ਬਿਹਤਰ ਢੰਗ ਨਾਲ ਸੰਭਾਲਣ ਵਿੱਚ ਮਦਦ ਕਰਦੇ ਹਨ। [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 'solver' ਨੂੰ "optimization ਸਮੱਸਿਆ ਵਿੱਚ ਵਰਤਣ ਲਈ algorithm" ਵਜੋਂ ਪਰਿਭਾਸ਼ਿਤ ਕੀਤਾ ਗਿਆ ਹੈ। [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn ਇਹ ਟੇਬਲ ਪੇਸ਼ ਕਰਦਾ ਹੈ ਜੋ ਦਿਖਾਉਂਦਾ ਹੈ ਕਿ solvers ਵੱਖ-ਵੱਖ ਡਾਟਾ ਸਟ੍ਰਕਚਰਾਂ ਦੁਆਰਾ ਪੇਸ਼ ਕੀਤੀਆਂ ਚੁਣੌਤੀਆਂ ਨੂੰ ਕਿਵੇਂ ਸੰਭਾਲਦੇ ਹਨ:

![solvers](../../../../4-Classification/2-Classifiers-1/images/solvers.png)

## ਅਭਿਆਸ - ਡਾਟਾ ਨੂੰ ਵੰਡੋ

ਅਸੀਂ ਆਪਣੇ ਪਹਿਲੇ ਟ੍ਰੇਨਿੰਗ ਟ੍ਰਾਇਲ ਲਈ logistic regression 'ਤੇ ਧਿਆਨ ਦੇ ਸਕਦੇ ਹਾਂ ਕਿਉਂਕਿ ਤੁਸੀਂ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ ਇਸ ਬਾਰੇ ਸਿੱਖਿਆ ਹੈ।
ਆਪਣੇ ਡਾਟਾ ਨੂੰ ਟ੍ਰੇਨਿੰਗ ਅਤੇ ਟੈਸਟਿੰਗ ਸਮੂਹਾਂ ਵਿੱਚ ਵੰਡੋ, `train_test_split()` ਕਾਲ ਕਰਕੇ:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## ਅਭਿਆਸ - logistic regression ਲਾਗੂ ਕਰੋ

ਕਿਉਂਕਿ ਤੁਸੀਂ multiclass ਕੇਸ ਦੀ ਵਰਤੋਂ ਕਰ ਰਹੇ ਹੋ, ਤੁਹਾਨੂੰ ਇਹ ਚੁਣਨਾ ਪਵੇਗਾ ਕਿ ਕਿਹੜਾ _scheme_ ਵਰਤਣਾ ਹੈ ਅਤੇ ਕਿਹੜਾ _solver_ ਸੈਟ ਕਰਨਾ ਹੈ। LogisticRegression ਦੀ ਵਰਤੋਂ ਕਰੋ ਜਿਸ ਵਿੱਚ multi_class ਸੈਟ `ovr` 'ਤੇ ਹੈ ਅਤੇ solver **liblinear** 'ਤੇ ਸੈਟ ਹੈ।

1. ਇੱਕ logistic regression ਬਣਾਓ ਜਿਸ ਵਿੱਚ multi_class `ovr` 'ਤੇ ਸੈਟ ਹੈ ਅਤੇ solver `liblinear` 'ਤੇ ਸੈਟ ਹੈ:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ ਇੱਕ ਵੱਖਰਾ solver ਜਿਵੇਂ `lbfgs` ਅਜ਼ਮਾਓ, ਜੋ ਅਕਸਰ default ਵਜੋਂ ਸੈਟ ਹੁੰਦਾ ਹੈ
ਨੋਟ ਕਰੋ, ਜਦੋਂ ਲੋੜ ਹੋਵੇ ਤਾਂ ਆਪਣੇ ਡਾਟਾ ਨੂੰ ਸਮਤਲ ਕਰਨ ਲਈ Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) ਫੰਕਸ਼ਨ ਦੀ ਵਰਤੋਂ ਕਰੋ।
ਸਹੀਤਾ **80%** ਤੋਂ ਵੱਧ ਹੈ!

1. ਤੁਸੀਂ ਇਸ ਮਾਡਲ ਨੂੰ ਇੱਕ ਡਾਟਾ ਦੀ ਲਾਈਨ (#50) ਦੀ ਜਾਂਚ ਕਰਕੇ ਕੰਮ ਕਰਦੇ ਵੇਖ ਸਕਦੇ ਹੋ:

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    ਨਤੀਜਾ ਪ੍ਰਿੰਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

   ✅ ਇੱਕ ਵੱਖਰੀ ਲਾਈਨ ਨੰਬਰ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰੋ ਅਤੇ ਨਤੀਜੇ ਚੈੱਕ ਕਰੋ

1. ਹੋਰ ਗਹਿਰਾਈ ਵਿੱਚ ਜਾਣ ਲਈ, ਤੁਸੀਂ ਇਸ ਪੇਸ਼ਗੂਈ ਦੀ ਸਹੀਤਾ ਦੀ ਜਾਂਚ ਕਰ ਸਕਦੇ ਹੋ:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    ਨਤੀਜਾ ਪ੍ਰਿੰਟ ਕੀਤਾ ਜਾਂਦਾ ਹੈ - ਭਾਰਤੀ ਖਾਣਾ ਇਸਦਾ ਸਭ ਤੋਂ ਵਧੀਆ ਅਨੁਮਾਨ ਹੈ, ਚੰਗੀ ਸੰਭਾਵਨਾ ਨਾਲ:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ ਕੀ ਤੁਸੀਂ ਸਮਝਾ ਸਕਦੇ ਹੋ ਕਿ ਮਾਡਲ ਕਿਉਂ ਇਹ ਭਾਰਤੀ ਖਾਣਾ ਹੋਣ 'ਤੇ ਕਾਫ਼ੀ ਯਕੀਨ ਹੈ?

1. ਹੋਰ ਵਿਸਥਾਰ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ, ਜਿਵੇਂ ਤੁਸੀਂ ਰਿਗ੍ਰੈਸ਼ਨ ਪਾਠਾਂ ਵਿੱਚ ਕੀਤਾ ਸੀ, ਇੱਕ ਕਲਾਸੀਫਿਕੇਸ਼ਨ ਰਿਪੋਰਟ ਪ੍ਰਿੰਟ ਕਰੋ:

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

## 🚀ਚੁਣੌਤੀ

ਇਸ ਪਾਠ ਵਿੱਚ, ਤੁਸੀਂ ਆਪਣੇ ਸਾਫ ਕੀਤੇ ਡਾਟਾ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਇੱਕ ਮਸ਼ੀਨ ਲਰਨਿੰਗ ਮਾਡਲ ਬਣਾਇਆ ਜੋ ਸਾਮਗਰੀਆਂ ਦੇ ਇੱਕ ਸੈਟ ਦੇ ਆਧਾਰ 'ਤੇ ਰਾਸ਼ਟਰੀ ਖਾਣੇ ਦੀ ਪੇਸ਼ਗੂਈ ਕਰ ਸਕਦਾ ਹੈ। Scikit-learn ਦੁਆਰਾ ਡਾਟਾ ਨੂੰ ਕਲਾਸੀਫਾਈ ਕਰਨ ਲਈ ਦਿੱਤੇ ਗਏ ਕਈ ਵਿਕਲਪਾਂ ਨੂੰ ਪੜ੍ਹਨ ਲਈ ਕੁਝ ਸਮਾਂ ਲਓ। 'solver' ਦੇ ਸੰਕਲਪ ਵਿੱਚ ਹੋਰ ਗਹਿਰਾਈ ਵਿੱਚ ਜਾਓ ਤਾਂ ਜੋ ਪਿੱਛੇ ਚੱਲ ਰਹੇ ਪ੍ਰਕਿਰਿਆ ਨੂੰ ਸਮਝ ਸਕੋ।

## [ਪਾਠ-ਬਾਅਦ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

ਲੌਜਿਸਟਿਕ ਰਿਗ੍ਰੈਸ਼ਨ ਦੇ ਗਣਿਤ ਵਿੱਚ ਹੋਰ ਗਹਿਰਾਈ ਵਿੱਚ ਜਾਓ [ਇਸ ਪਾਠ](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf) ਵਿੱਚ
## ਅਸਾਈਨਮੈਂਟ 

[ਸੋਲਵਰਾਂ ਦਾ ਅਧਿਐਨ ਕਰੋ](assignment.md)

---

**ਅਸਵੀਕਰਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀਤਾ ਲਈ ਯਤਨਸ਼ੀਲ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਚਤਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।