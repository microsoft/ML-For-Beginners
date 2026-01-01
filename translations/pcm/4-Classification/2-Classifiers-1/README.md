<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1a6e9e46b34a2e559fbbfc1f95397c7b",
  "translation_date": "2025-11-18T18:50:19+00:00",
  "source_file": "4-Classification/2-Classifiers-1/README.md",
  "language_code": "pcm"
}
-->
# Cuisine classifiers 1

For dis lesson, you go use di dataset wey you save from di last lesson wey get balanced, clean data about cuisines.

You go use dis dataset with different classifiers to _predict one national cuisine based on di group of ingredients_. As you dey do am, you go learn more about di ways wey algorithms fit help for classification tasks.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
# Preparation

If you don finish [Lesson 1](../1-Introduction/README.md), make sure say _cleaned_cuisines.csv_ file dey inside di root `/data` folder for dis four lessons.

## Exercise - predict one national cuisine

1. For dis lesson _notebook.ipynb_ folder, import di file plus di Pandas library:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    Di data go look like dis:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. Now, import more libraries:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. Divide di X and y coordinates into two dataframes for training. `cuisine` fit be di labels dataframe:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    E go look like dis:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. Drop di `Unnamed: 0` column plus di `cuisine` column, use `drop()`. Save di rest of di data as trainable features:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    Your features go look like dis:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    3 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    4 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      1 |        0 | 0 |

Now you fit train your model!

## Choosing your classifier

Now wey your data don clean and e dey ready for training, you go need decide which algorithm you go use for di work.

Scikit-learn dey group classification under Supervised Learning, and for dat category you go find plenty ways to classify. [Di variety](https://scikit-learn.org/stable/supervised_learning.html) fit dey confusing for first sight. Di methods wey dey include classification techniques na:

- Linear Models
- Support Vector Machines
- Stochastic Gradient Descent
- Nearest Neighbors
- Gaussian Processes
- Decision Trees
- Ensemble methods (voting Classifier)
- Multiclass and multioutput algorithms (multiclass and multilabel classification, multiclass-multioutput classification)

> You fit also use [neural networks to classify data](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification), but dat one no dey inside dis lesson.

### Which classifier you go choose?

So, which classifier you go use? Sometimes, to try different ones and check di result na one way to test. Scikit-learn dey offer [side-by-side comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) for one created dataset, wey compare KNeighbors, SVC two ways, GaussianProcessClassifier, DecisionTreeClassifier, RandomForestClassifier, MLPClassifier, AdaBoostClassifier, GaussianNB and QuadraticDiscrinationAnalysis, wey show di results visualized:

![comparison of classifiers](../../../../translated_images/comparison.edfab56193a85e7f.pcm.png)
> Plots wey dem generate for Scikit-learn documentation

> AutoML dey solve dis problem well by running dis comparisons for di cloud, e go allow you choose di best algorithm for your data. Try am [here](https://docs.microsoft.com/learn/modules/automate-model-selection-with-azure-automl/?WT.mc_id=academic-77952-leestott)

### Better way

Better way wey pass to dey guess anyhow na to follow di ideas wey dey dis downloadable [ML Cheat sheet](https://docs.microsoft.com/azure/machine-learning/algorithm-cheat-sheet?WT.mc_id=academic-77952-leestott). Here, we go see say for our multiclass problem, we get some options:

![cheatsheet for multiclass problems](../../../../translated_images/cheatsheet.07a475ea444d2223.pcm.png)
> Part of Microsoft's Algorithm Cheat Sheet, wey show multiclass classification options

✅ Download dis cheat sheet, print am, and hang am for your wall!

### Reasoning

Make we try reason di different approaches wey we fit use based on di constraints wey we get:

- **Neural networks dey too heavy**. Based on our clean but small dataset, and di fact say we dey run training locally for notebooks, neural networks dey too much for dis task.
- **No two-class classifier**. We no go use two-class classifier, so dat one rule out one-vs-all.
- **Decision tree or logistic regression fit work**. Decision tree fit work, or logistic regression for multiclass data.
- **Multiclass Boosted Decision Trees dey solve different problem**. Di multiclass boosted decision tree dey best for nonparametric tasks, e.g. tasks wey dey build rankings, so e no go help us.

### Using Scikit-learn 

We go use Scikit-learn to analyze our data. But, plenty ways dey to use logistic regression for Scikit-learn. Check di [parameters to pass](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).  

Di two important parameters na `multi_class` and `solver` wey we need to set when we dey ask Scikit-learn to do logistic regression. Di `multi_class` value dey apply certain behavior. Di value of di solver na di algorithm wey e go use. No be all solvers fit pair with all `multi_class` values.

According to di docs, for di multiclass case, di training algorithm:

- **Dey use di one-vs-rest (OvR) scheme**, if di `multi_class` option dey set to `ovr`
- **Dey use di cross-entropy loss**, if di `multi_class` option dey set to `multinomial`. (Currently di `multinomial` option dey supported only by di ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)"

> 🎓 Di 'scheme' here fit be 'ovr' (one-vs-rest) or 'multinomial'. Since logistic regression dey really designed to support binary classification, dis schemes dey help am handle multiclass classification tasks better. [source](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

> 🎓 Di 'solver' dey defined as "di algorithm wey e go use for di optimization problem". [source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic%20regressio#sklearn.linear_model.LogisticRegression).

Scikit-learn dey offer dis table to explain how solvers dey handle different challenges wey different kinds of data structures dey bring:

![solvers](../../../../translated_images/solvers.5fc648618529e627.pcm.png)

## Exercise - split di data

Make we focus on logistic regression for our first training trial since you don learn about am for di previous lesson.
Split your data into training and testing groups by calling `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## Exercise - apply logistic regression

Since you dey use di multiclass case, you need choose wetin _scheme_ to use and wetin _solver_ to set. Use LogisticRegression with multiclass setting and di **liblinear** solver to train.

1. Create logistic regression with multi_class set to `ovr` and di solver set to `liblinear`:

    ```python
    lr = LogisticRegression(multi_class='ovr',solver='liblinear')
    model = lr.fit(X_train, np.ravel(y_train))
    
    accuracy = model.score(X_test, y_test)
    print ("Accuracy is {}".format(accuracy))
    ```

    ✅ Try different solver like `lbfgs`, wey dem dey often set as default

    > Note, use Pandas [`ravel`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ravel.html) function to flatten your data when e dey needed.

    Di accuracy dey good at over **80%**!

1. You fit see dis model for action by testing one row of data (#50):

    ```python
    print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
    print(f'cuisine: {y_test.iloc[50]}')
    ```

    Di result go show:

   ```output
   ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
   cuisine: indian
   ```

✅ Try use different row number and check wetin e go show

1. If you wan sabi more, you fit check how correct dis prediction be:

    ```python
    test= X_test.iloc[50].values.reshape(-1, 1).T
    proba = model.predict_proba(test)
    classes = model.classes_
    resultdf = pd.DataFrame(data=proba, columns=classes)
    
    topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
    topPrediction.head()
    ```

    Di result wey e print - Indian food na di best guess, and e get beta chance:

    |          |        0 |
    | -------: | -------: |
    |   indian | 0.715851 |
    |  chinese | 0.229475 |
    | japanese | 0.029763 |
    |   korean | 0.017277 |
    |     thai | 0.007634 |

    ✅ You fit explain why di model sure say na Indian food?

1. Get more info by printing classification report, like you do for regression lessons:

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

For dis lesson, you use di clean data take build machine learning model wey fit predict national food based on di ingredients wey dem use. Take time read di plenty options wey Scikit-learn get to classify data. Try sabi di concept of 'solver' well to understand wetin dey happen for di background.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Try sabi di mathematics wey dey behind logistic regression for [dis lesson](https://people.eecs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194%20Fall%202011%20Lecture%2006.pdf)
## Assignment 

[Study di solvers](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI transleto service [Co-op Translator](https://github.com/Azure/co-op-translator) do di translation. Even as we dey try make am correct, abeg sabi say machine translation fit get mistake or no dey accurate well. Di original dokyument for im native language na di main correct source. For important mata, e good make una use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because una use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->