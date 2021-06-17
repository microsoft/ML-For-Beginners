# Introduction to Classification

In these four lessons, you will discover the 'meat and potatoes' of classic machine learning - Classification. No pun intended - we will walk through using various classification algorithms with a dataset all about the brilliant cuisines of Asia and India. Hope you're hungry!

Classification is a form of [supervised learning](https://wikipedia.org/wiki/Supervised_learning) that bears a lot in common with Regression techniques. If machine learning is all about assigning names to things via datasets, then classification generally falls into two groups: binary classification and multiclass classfication.

[![Introduction to Classification](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introduction to Classification")

> 🎥 Click the image above for a video: MIT's John Guttag introduces Classification

Remember, Linear Regression helped you predict relationships between variables and make accurate predictions on where a new datapoint would fall in relationship to that line. So, you could predict what price a pumpkin would be in September vs. December, for example. Logistic Regression helped you discover binary categories: at this price point, is this pumpkin orange or not-orange?

Classification uses various algorithms to determine other ways of determining a data point's label or class. Let's work with this cuisine data to see whether, by observing a group of ingredients, we can determine its cuisine of origin.
## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/19/)

### Introduction

Classification is one of the fundamental activities of the machine learning researcher and data scientist. From basic classification of a binary value ("is this email spam or not?") to complex image classification and segmentation using computer vision, it's always useful to be able to sort data into classes and ask questions of it. Or, to state the process in a more scientific way, your classification method creates a predictive model that enables you to map the relationship between input variables to output variables. 

Before starting the process of cleaning our data, visualizing it, and prepping it for our ML tasks, let's learn a bit about the various ways machine learning can be leveraged to classify data.

Derived from [statistics](https://wikipedia.org/wiki/Statistical_classification), classification using classic machine learning uses features, such as 'smoker','weight', and 'age' to determine 'likelihood of developing X disease'. As a supervised learning technique similar to the Regression exercises you performed earlier, your data is labeled and the ML algorithms use those labels to classify and predict classes (or 'features') of a dataset and assign them to a group or outcome.

✅ Take a moment to imagine a dataset about cuisines. What would a multiclass model be able to answer? What would a binary model be able to answer? What if you wanted to determine whether a given cuisine was likely to use fenugreek? What if you wanted to see if, given a present of a grocery bag full of star anise, artichokes, cauliflower, and horseradish, you could create a typical Indian dish?

## Hello 'classifier'

The question we want to ask of this cuisine dataset is actually a **multiclass question**, as we have several potential national cuisines to work with. Given a batch of ingredients, which of these many classes will the data fit?

Scikit-Learn offers several different algorithms to use to classify data, depending on the kind of problem you want to solve. In the next two lessons, you'll learn about several of these algorithms.

## Clean and Balance Your Data

The first task at hand before starting this project is to clean and **balance** your data to get better results. Start with the blank `notebook.ipynb` file ini the root of this folder.

The first think to install is [imblearn](https://imbalanced-learn.org/stable/). This is a Scikit-Learn package that will allow you to better balance the data (you will learn more about this task in a minute).

```python
pip install imblearn
```

Then, import the packages you need to import your data and visualize it. Import SMOTE from imblearn.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```
The next task will be to import the data:

```python
df  = pd.read_csv('../data/cuisines.csv')
```

Check the data's shape:

```python
df.head()
```

The first five rows look like this:


|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |

Get info about this data:

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2448 entries, 0 to 2447
Columns: 385 entries, Unnamed: 0 to zucchini
dtypes: int64(384), object(1)
memory usage: 7.2+ MB
```
## Learning about cuisines

Now the work starts to become more interesting. Let's discover the distribution of data, per cuisine:

```python
df.cuisine.value_counts().plot.barh()
```

![cuisine data distribution](images/cuisine-dist.png)

There are a finite number of cuisines, but the distribution of data is uneven. You can fix that! Before doing so, explore a little more. How much data exactly is available per cuisine?

```python
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```
thai df: (289, 385)
japanese df: (320, 385)
chinese df: (442, 385)
indian df: (598, 385)
korean df: (799, 385)

## Discovering ingredients

Now you can dig deeper into the data and learn what are the typical ingredients per cuisine. You should clean out recurrent data that creates confusion between cuisines, so let's learn about this problem.

Create a function in Python to create an ingredient dataframe. This function will start by dropping an unhelpful column and sort through ingredients by their count:

```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False
    inplace=False)
    return ingredient_df
```
Now you can use that function to get an idea of top ten most popular ingredients by cuisine:

```python
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```
![thai](images/thai.png)

```python
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```
![japanese](images/japanese.png)

```python
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```
![chinese](images/chinese.png)

```python
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```
![indian](images/indian.png)

```python
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```
![korean](images/korean.png)

Now, drop the most common ingredients that create confusion between distinct cuisines. Everyone loves rice, garlic and ginger!

```python
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()
```
## Balance the dataset

Now that you have cleaned the data, use [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Synthetic Minority Over-sampling Technique" - to balance it. This strategy generates new samples by interpolation.

```python
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```
By balancing your data, you'll have better results when classifying it. Think about a binary classification. If most of your data is one class, a ML model is going to predict that class more frequently, just because there is more data for it. Balancing the data takes any skewed data and helps remove this imbalance. 

Now you can check the numbers of labels per ingredient:

```python
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```

```
new label count: korean      799
chinese     799
indian      799
japanese    799
thai        799
Name: cuisine, dtype: int64
old label count: korean      799
indian      598
chinese     442
japanese    320
thai        289
Name: cuisine, dtype: int64
```

The data is nice and clean, balanced, and very delicious! You can take one more look at the data using `transformed_df.head()` and `transformed_df.info()`. Save a copy of this data for use in future lessons:

```python
transformed_df.to_csv("../../data/cleaned_cuisine.csv")
```
This fresh CSV can now be found in the root data folder.
## 🚀Challenge

This curriculum contains several interesting datasets. Dig through the `data` folders and see if any contain datasets that would be appropriate for binary or multi-class classification? What questions would you ask of this dataset?

## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/20/)

## Review & Self Study

Explore SMOTE's API. What use cases is it best used for? What problems does it solve?

## Assignment 

[Explore classification methods](assignment.md)
