def create_ingredient_df(df):

    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')

    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]

    ingredient_df = ingredient_df.sort_values(by='value', ascending=False,

                                               inplace=False)

    return ingredient_df

  

country_dfs = {

    'thai': thai_df,

    'japanese': japanese_df,

    'chinese': chinese_df,

    'indian': indian_df,

    'korean': korean_df

}

  

# Create a dictionary to store the ingredient dataframes for each country

country_ingredient_dfs = {}

for country, df in country_dfs.items():

    country_ingredient_dfs[country] = create_ingredient_df(df)

  

# Plot a bar chart of the top 10 ingredients for each country

for country, df in country_ingredient_dfs.items():

    df.head(10).plot.barh(title=f'{country.title()} Ingredient Popularity')





### MS LEARN ...
[Skip to main content](https://learn.microsoft.com/en-us/training/modules/train-evaluate-regression-models/5-exercise-powerful-models#main)

[

](https://www.microsoft.com/)

- [Learn](https://learn.microsoft.com/en-us/)
- [Documentation](https://learn.microsoft.com/en-us/docs/)
- [Training](https://learn.microsoft.com/en-us/training/)
- [Credentials](https://learn.microsoft.com/en-us/credentials/)
- [Q&A](https://learn.microsoft.com/en-us/answers/)
- [Code Samples](https://learn.microsoft.com/en-us/samples/browse/)
- [Assessments](https://learn.microsoft.com/en-us/assessments/)
- [Shows](https://learn.microsoft.com/en-us/shows/)

Search

![](data:image/svg+xml, %3Csvg xmlns='http://www.w3.org/2000/svg' height='64' class='font-weight-bold' style='font: 600 30.11764705882353px "SegoeUI", Arial' width='64'%3E%3Ccircle fill='hsl(163.2, 63%, 25%)' cx='32' cy='32' r='32' /%3E%3Ctext x='50%25' y='55%25' dominant-baseline='middle' text-anchor='middle' fill='%23FFF' %3EJN%3C/text%3E%3C/svg%3E)

[Training](https://learn.microsoft.com/en-us/training/)

- Products
- Career Paths
- [Learning Paths](https://learn.microsoft.com/en-us/training/browse/)
- [Courses](https://learn.microsoft.com/en-us/training/courses/browse/)
- Educator Center
- Student Hub
- [FAQ & Help](https://learn.microsoft.com/en-us/training/support/)

1. [Learn](https://learn.microsoft.com/en-us/) 
 3. [Training](https://learn.microsoft.com/en-us/training/) 
 5. [Browse](https://learn.microsoft.com/en-us/training/browse/) 
 7. [Train and evaluate regression models](https://learn.microsoft.com/en-us/training/modules/train-evaluate-regression-models/) 

Add

[Previous](https://learn.microsoft.com/en-us/training/modules/train-evaluate-regression-models/4-discover-new-regression-models/)

- Unit 5 of 9

[Next](https://learn.microsoft.com/en-us/training/modules/train-evaluate-regression-models/6-improve-models/)

# Exercise - Experiment with more powerful regression models

Completed100 XP

- 10 minutes

Sandbox activated! Time remaining: 

1 hr 57 min

You have used 1 of 10 sandboxes for today. More sandboxes will be available tomorrow.

Execution succeeded with no output for cell at position 2. Kernel is now idle

__

RuntimeFileEditView

__Run all__

__azureml_py38__

__

__

____

[1]
```
# Import modules we'll need for this notebook

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

# load the training dataset

!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv

bike_data = pd.read_csv('daily-bike-share.csv')

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

numeric_features = ['temp', 'atemp', 'hum', 'windspeed']

categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

bike_data[numeric_features + ['rentals']].describe()

print(bike_data.head())

# Separate features and labels

# After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.

X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values

# Split data 70%-30% into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

```





`Matplotlib is building the font cache using fc-list. This may take a moment.` `--2023-10-14 18:18:28-- [https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv](https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv) Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ... Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected. HTTP request sent, awaiting response... 200 OK Length: 48800 (48K) [text/plain] Saving to: ‘daily-bike-share.csv’ daily-bike-share.cs 100%[===================>] 47.66K --.-KB/s in 0.008s 2023-10-14 18:18:28 (5.76 MB/s) - ‘daily-bike-share.csv’ saved [48800/48800] instant dteday season yr mnth holiday weekday workingday \ 0 1 1/1/2011 1 0 1 0 6 0 1 2 1/2/2011 1 0 1 0 0 0 2 3 1/3/2011 1 0 1 0 1 1 3 4 1/4/2011 1 0 1 0 2 1 4 5 1/5/2011 1 0 1 0 3 1 weathersit temp atemp hum windspeed rentals day 0 2 0.344167 0.363625 0.805833 0.160446 331 1 1 2 0.363478 0.353739 0.696087 0.248539 131 2 2 1 0.196364 0.189405 0.437273 0.248309 120 3 3 1 0.200000 0.212122 0.590435 0.160296 108 4 4 1 0.226957 0.229270 0.436957 0.186900 82 5 Training Set: 511 rows Test Set: 220 rows`

```PYTHON

```
from sklearn.linear_model import Lasso

# Fit a lasso model on the training set

model = Lasso().fit(X_train, y_train)

print (model, "\n")

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import export_text

# Train the model

model = DecisionTreeRegressor().fit(X_train, y_train)

print (model, "\n")

# Visualize the model tree

tree = export_text(model)

print(tree)

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()

from sklearn.ensemble import RandomForestRegressor

# Train the model

model = RandomForestRegressor().fit(X_train, y_train)

print (model, "\n")

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()

# Train the model

from sklearn.ensemble import GradientBoostingRegressor

# Fit a lasso model on the training set

model = GradientBoostingRegressor().fit(X_train, y_train)

print (model, "\n")

# Evaluate the model using the test data

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("MSE:", mse)

rmse = np.sqrt(mse)

print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)

print("R2:", r2)

# Plot predicted vs actual

plt.scatter(y_test, predictions)

plt.xlabel('Actual Labels')

plt.ylabel('Predicted Labels')

plt.title('Daily Bike Share Predictions')

# overlay the regression line

z = np.polyfit(y_test, predictions, 1)

p = np.poly1d(z)

plt.plot(y_test,p(y_test), color='magenta')

plt.show()

__learn-notebooks-bc2fd8aa-5b1f-44ce-a745-cac114299f27

Compute connected

__Viewing

Kernel idle

azureml_py38

---

## Next unit: Improve models with hyperparameters

[Continue](https://learn.microsoft.com/en-us/training/modules/train-evaluate-regression-models/6-improve-models/)

Need help? See our [troubleshooting guide](https://learn.microsoft.com/en-us/training/support/troubleshooting?uid=learn.wwl.train-evaluate-regression-models.exercise-powerful-models&documentId=d189d9e6-568a-fe91-e351-b0a8234848c3&versionIndependentDocumentId=3c48d34b-e2b6-e3c8-c86c-a6284baa2b0d&contentPath=%2FMicrosoftDocs%2Flearn-pr%2Fblob%2Flive%2Flearn-pr%2Fmachine-learning%2Ftrain-evaluate-regression-models%2F5-exercise-powerful-models.yml&url=https%3A%2F%2Flearn.microsoft.com%2Fen-us%2Ftraining%2Fmodules%2Ftrain-evaluate-regression-models%2F5-exercise-powerful-models&author=jasdeb) or provide specific feedback by [reporting an issue](https://learn.microsoft.com/en-us/training/support/troubleshooting?uid=learn.wwl.train-evaluate-regression-models.exercise-powerful-models&documentId=d189d9e6-568a-fe91-e351-b0a8234848c3&versionIndependentDocumentId=3c48d34b-e2b6-e3c8-c86c-a6284baa2b0d&contentPath=%2FMicrosoftDocs%2Flearn-pr%2Fblob%2Flive%2Flearn-pr%2Fmachine-learning%2Ftrain-evaluate-regression-models%2F5-exercise-powerful-models.yml&url=https%3A%2F%2Flearn.microsoft.com%2Fen-us%2Ftraining%2Fmodules%2Ftrain-evaluate-regression-models%2F5-exercise-powerful-models&author=jasdeb#report-feedback).

How are we doing?

TerriblePoorFairGoodGreat

[English (United States)](https://learn.microsoft.com/en-us/locale?target=https%3A%2F%2Flearn.microsoft.com%2Fen-us%2Ftraining%2Fmodules%2Ftrain-evaluate-regression-models%2F5-exercise-powerful-models)

Theme

- [Previous Versions](https://learn.microsoft.com/en-us/previous-versions/)
- [Blog](https://techcommunity.microsoft.com/t5/microsoft-learn-blog/bg-p/MicrosoftLearnBlog)
- [Contribute](https://learn.microsoft.com/en-us/contribute/)
- [Privacy](https://go.microsoft.com/fwlink/?LinkId=521839)
- [Terms of Use](https://learn.microsoft.com/en-us/legal/termsofuse)
- [Trademarks](https://www.microsoft.com/legal/intellectualproperty/Trademarks/)
- © Microsoft 2023