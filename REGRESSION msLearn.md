
# Regression - Experimenting with additional models

In the previous notebook, we used simple regression models to look at the relationship between features of a bike rentals dataset. In this notebook, we'll experiment with more complex models to improve our regression performance.

Let's start by loading the bicycle sharing data as a **Pandas** DataFrame and viewing the first few rows. We'll also split our data into training and test datasets.

```python

# Import modules we'll need for this notebook

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

  

# load the training dataset

!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv

bike_data = pd.read_csv('daily-bike-share.csv')

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

numeric_features = ['temp', 'atemp', 'hum', 'windspeed']

categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

bike_data[numeric_features + ['rentals']].describe()

print(bike_data.head())

  
  

# Separate features and labels

# After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.

X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values

  

# Split data 70%-30% into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

  

print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))
```





```Now we have the following four datasets:

- **X_train**: The feature values we'll use to train the model
- **y_train**: The corresponding labels we'll use to train the model
- **X_test**: The feature values we'll use to validate the model
- **y_test**: The corresponding labels we'll use to validate the model

Now we're ready to train a model by fitting a suitable regression algorithm to the training data.

## Experiment with Algorithms

The linear-regression algorithm we used last time to train the model has some predictive capability, but there are many kinds of regression algorithm we could try, including:

- **Linear algorithms**: Not just the Linear Regression algorithm we used above (which is technically an _Ordinary Least Squares_ algorithm), but other variants such as _Lasso_ and _Ridge_.
- **Tree-based algorithms**: Algorithms that build a decision tree to reach a prediction.
- **Ensemble algorithms**: Algorithms that combine the outputs of multiple base algorithms to improve generalizability.

> **Note**: For a full list of Scikit-Learn estimators that encapsulate algorithms for supervised machine learning, see the [Scikit-Learn documentation](https://scikit-learn.org/stable/supervised_learning.html). There are many algorithms from which to choose, but for most real-world scenarios, the [Scikit-Learn estimator cheat sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) can help you find a suitable starting point.

### Try Another Linear Algorithm

Let's try training our regression model by using a **Lasso** algorithm. We can do this by just changing the estimator in the training code.
```


```python
from sklearn.linear_model import Lasso

  

# Fit a lasso model on the training set

model = Lasso().fit(X_train, y_train)

print (model, "\n")

  

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

  

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()
```


### Try a Decision Tree Algorithm

  

As an alternative to a linear model, there's a category of algorithms for machine learning that uses a tree-based approach in which the features in the dataset are examined in a series of evaluations, each of which results in a *branch* in a *decision tree* based on the feature value. At the end of each series of branches are leaf-nodes with the predicted label value based on the feature values.

  

It's easiest to see how this works with an example. Let's train a Decision Tree regression model using the bike rental data. After training the model, the following code will print the model definition and a text representation of the tree it uses to predict label values.


```python
from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import export_text

  

# Train the model

model = DecisionTreeRegressor().fit(X_train, y_train)

print (model, "\n")

  

# Visualize the model tree

tree = export_text(model)

print(tree)
```


So now we have a tree-based model, but is it any good? Let's evaluate it with the test data.


```python
# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

  

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()
```


The tree-based model doesn't seem to have improved over the linear model, so what else could we try?

### Try an Ensemble Algorithm

Ensemble algorithms work by combining multiple base estimators to produce an optimal model, either by applying an aggregate function to a collection of base models (sometimes referred to a _bagging_) or by building a sequence of models that build on one another to improve predictive performance (referred to as _boosting_).

For example, let's try a Random Forest model, which applies an averaging function to multiple Decision Tree models for a better overall model.



```python
from sklearn.ensemble import RandomForestRegressor

  

# Train the model

model = RandomForestRegressor().fit(X_train, y_train)

print (model, "\n")

  

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

  

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()
```


For good measure, let's also try a *boosting* ensemble algorithm. We'll use a Gradient Boosting estimator, which like a Random Forest algorithm builds multiple trees; but instead of building them all independently and taking the average result, each tree is built on the outputs of the previous one in an attempt to incrementally reduce the *loss* (error) in the model.

```python
# Train the model

from sklearn.ensemble import GradientBoostingRegressor

  

# Fit a lasso model on the training set

model = GradientBoostingRegressor().fit(X_train, y_train)

print (model, "\n")

  

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

  

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()
```



## Summary

  

Here, we've tried a number of new regression algorithms to improve performance. In our next notebook, we'll look at *tuning* these algorithms to improve performance.

  

## Further Reading

To learn more about Scikit-Learn, see the [Scikit-Learn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).


### Lasso model

In statistics and machine learning,
- Lasso (Least Absolute Shrinkage and Selection Operator; also Lasso or LASSO)
- is a regression analysis method that performs both
- variable selection and 
- regularization in order to enhance the prediction accuracy and
- interpretability of the resulting statistical model.

Lasso was originally formulated for linear regression models. This simple case reveals a substantial amount about the estimator. [These include its relationship to ridge regression and best subset selection and the connections between lasso coefficient estimates and so-called soft thresholding](https://en.wikipedia.org/wiki/Lasso_%28statistics%29)[1](https://en.wikipedia.org/wiki/Lasso_%28statistics%29).

The optimization objective for Lasso is:​

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mfrac><mn>1</mn><mrow><mn>2</mn><msub><mi>n</mi><mrow><mi>s</mi><mi>a</mi><mi>m</mi><mi>p</mi><mi>l</mi><mi>e</mi><mi>s</mi></mrow></msub></mrow></mfrac><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∣</mi><mi>y</mi><mo>−</mo><mi>X</mi><mi>w</mi><mi mathvariant="normal">∣</mi><msubsup><mi mathvariant="normal">∣</mi><mn>2</mn><mn>2</mn></msubsup><mo>+</mo><mi>α</mi><mi mathvariant="normal">∣</mi><mi mathvariant="normal">∣</mi><mi>w</mi><mi mathvariant="normal">∣</mi><msub><mi mathvariant="normal">∣</mi><mn>1</mn></msub></mrow><annotation encoding="application/x-tex">\frac{1}{2n_{samples}}||y-Xw||^2_2+\alpha||w||_1
</annotation></semantics></math>


where `n_samples` is the number of samples, `y` is the target variable, `X` is the input data, `w` is the weight vector, and `alpha` is a constant that multiplies the L1 term.

Lasso’s ability to perform subset selection relies on the form of the constraint and has a variety of interpretations including in terms of geometry, Bayesian statistics, and convex analysis. The LASSO method regularizes model parameters by shrinking the regression coefficients, reducing some of them to zero. The feature selection phase occurs after the shrinkage, where every non-zero value is selected to be used in the model

This method is significant in the minimization of prediction errors that are common in statistical models. It’s used over regression methods for a more accurate prediction. This model uses shrinkage where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple, sparse models (i.e., models with fewer parameters)


### explanation
Sure, let’s imagine you’re playing a game of soccer with your friends. Now, each of your friends has different skills. Some are good at scoring goals, some are good at defending, and some are good at passing the ball.

Now, you’re the team captain and you want to pick the best team. But you can only pick a few friends to be on your team. How do you decide?

You could just pick randomly, but that might not give you the best team. Instead, you want a way to pick the best players based on their skills.

This is where Lasso comes in. In our analogy, Lasso is like a smart team captain. It looks at all your friends’ skills (these are like the ‘features’ in a dataset), and picks the ones that are most important for winning the game (these are the ‘variables’ in our model).

Just like a good captain won’t pick a friend who can’t run fast or kick well, Lasso also ‘shrinks’ the importance of less useful features down to zero - effectively leaving them out of the model.

So, Lasso helps us make better decisions by focusing on what’s really important and ignoring what’s not. And just like picking the right team can help you win your soccer game, using Lasso can help make better predictions with data!



After formulating a hypothesis in a machine learning study, the typical steps are as follows:

1. **Data Collection**: Gather the data that you’ll use to train and test your machine learning models. This could involve scraping websites, conducting surveys, performing experiments, or a number of other data-gathering activities.
    
2. **Data Preprocessing**: Clean and format your data so it can be input into machine learning models. This often involves handling missing values, dealing with outliers, normalizing numerical data, and encoding categorical data.
    
3. **Feature Selection/Engineering**: Identify which features (variables) in your dataset you will use to train your model. You might also create new features that can better represent the patterns in your data.
    
4. **Model Selection**: Choose the type of model or models you’ll use, such as linear regression, decision trees, neural networks etc. This decision is often based on the nature of your data and the problem you’re trying to solve.
    
5. **Training**: Train your model on a subset of your data.
    
6. **Validation**: Validate your model’s performance using a different subset of your data (the validation set). This helps ensure that your model not only fits the training data well but can also generalize to new data.
    
7. **Evaluation**: Evaluate your model using various metrics like accuracy, precision, recall, F1 score etc., depending on the problem at hand.
    
8. **Hyperparameter Tuning**: Adjust the settings (hyperparameters) of your model to see if you can improve performance.
    
9. **Testing**: Test your model on a test set that hasn’t been used during training or validation phases.
    
10. **Interpretation**: Interpret the results and understand how well the model is performing and why it’s making the predictions it’s making.
    
11. **Reporting**: Document all these steps, findings, and any conclusions you can draw in a clear and reproducible way.
    

For example, let’s say we’re working on a binary classification problem where we want to predict whether an email is spam or not (our hypothesis). We would:

- Collect a dataset of emails that have been labeled as “spam” or “not spam”.
- Preprocess this data by cleaning up the email text (removing punctuation, making everything lowercase) and encoding our labels as 0 (not spam) and 1 (spam).
- Choose features like the frequency of certain words or characters.
- Split our dataset into training, validation, and test sets.
- Choose a model like logistic regression.
- Train our logistic regression model on our training set.
- Validate our model on our validation set.
- Evaluate its performance using metrics like accuracy.
- Tune any hyperparameters of our logistic regression model to try to improve performance.
- Test our model on our test set to get a final measure of its performance.
- Interpret our results and report them in a clear and understandable way.

Remember that these steps might vary slightly depending on the specific problem or field of study. 😊