# Time series forecasting with Support Vector Regressor

In the previous lesson, you learned how to use ARIMA model to make time series predictions. Now you'll be looking at Support Vector Regressor model which is a regressor model used to predict continuous data.




## Introduction

In this lesson, you will discover a specific way to build models with [**SVM**: **S**upport **V**ector**M**achine](https://en.wikipedia.org/wiki/Support-vector_machine) for regression, or **SVR: Support Vector Regressor**. 

## SVR in the context of time series

Let's unpack the parts of ARIMA to better understand how it helps us model time series and help us make predictions against it.

- **Regression:** Supervised learning technique to predict continuous values from a given set of inputs. The idea is to fit a curve (or line) in the feature space that has the maximum number of data points.
- **Support Vector Machine (SVM):** A type of supervised machine learning model used for classification, regression and outliers detection. The model is a hyperplane in the feature space, which in case of classification acts as a boundary, and in case of regression acts as the best-fit line. In SVM, a kernel function is generally used to transform the dataset so that a non-linear decision surface is able to transform to a linear equation in a higher number of dimension spaces
- **Support Vector Regressor (SVR):** A type of SVM, to find the best fit line (which in the case of SVM is a hyperplane) that has the maximum number of data points.

## Exercise - build an SVR model

The first few steps for data preparation are the same as that of the previous lesson. Open the _/working_ folder in this lesson and find the _notebook.ipynb_ file.

1. Run the notebook to load the `statsmodels` Python library; you will need this for ARIMA models.

2. Load necessary libraries

3. Now, load up several more libraries useful for plotting data:

   ```python
   import os
   import warnings
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import datetime as dt
   import math
   
   from sklearn.svm import SVR
   from sklearn.preprocessing import MinMaxScaler
   from common.utils import load_data, mape
   ```
   
4. Load the data from the `/data/energy.csv` file into a Pandas dataframe and take a look:

   ```python
   energy = load_data('./data')[['load']]
   energy.head(10)
   ```
   
5. Plot all the available energy data from January 2012 to December 2014. There should be no surprises as we saw this data in the last lesson:

   ```python
   energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   Now, let's build a model!

### Create training and testing datasets

Now your data is loaded, so you can separate it into train and test sets. You'll train your model on the train set. As usual, after the model has finished training, you'll evaluate its accuracy using the test set. You need to ensure that the test set covers a later period in time from the training set to ensure that the model does not gain information from future time periods.

1. Allocate a two-month period from September 1 to October 31, 2014 to the training set. The test set will include the two-month period of November 1 to December 31, 2014:

   ```python
   train_start_dt = '2014-11-01 00:00:00'
   test_start_dt = '2014-12-30 00:00:00'
   ```

   Since this data reflects the daily consumption of energy, there is a strong seasonal pattern, but the consumption is most similar to the consumption in more recent days.

2. Visualize the differences:

   ```python
   energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
       .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
       .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
   plt.xlabel('timestamp', fontsize=12)
   plt.ylabel('load', fontsize=12)
   plt.show()
   ```

   ![training and testing data](images/train-test.png)

   Therefore, using a relatively small window of time for training the data should be sufficient.


### Prepare the data for training

Now, you need to prepare the data for training by performing filtering and scaling of your data. Filter your dataset to only include the time periods and columns you need, and scaling to ensure the data is projected in the interval 0,1.

1. Filter the original dataset to include only the aforementioned time periods per set and only including the needed column 'load' plus the date:

   ```python
   train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
   test = energy.copy()[energy.index >= test_start_dt][['load']]
   
   print('Training data shape: ', train.shape)
   print('Test data shape: ', test.shape)
   ```

   You can see the shape of the data:

   ```output
   Training data shape:  (1416, 1)
   Test data shape:  (48, 1)
   ```

2. Scale the data to be in the range (0, 1).

   ```python
   scaler = MinMaxScaler()
   train['load'] = scaler.fit_transform(train)
   train.head(10)
   ```

4. Now that you have calibrated the scaled data, you can scale the test data:

   ```python
   test['load'] = scaler.transform(test)
   test.head()
   ```

### Create data with time-steps

For the SVR, you transform the input data to be of the form `[batch, timesteps]`. So, you reshape the existing `train_data` and `test_data` such that there is a new dimension which refers to the timesteps. 

```python
# Converting to numpy arrays

train_data = train.values
test_data = test.values
```

For this example, we take `timesteps = 5`. So, the inputs to the model are the data for the first 4 timesteps, and the output will be the data for the 5th timestep.

```python
# Selecting the timesteps
timesteps=5
```

```python
# Converting training data to 3D tensor using nested list comprehension

train_data_timesteps=np.array([[j for j in train_data[i:i+timesteps]] for i in range(0,len(train_data)-timesteps+1)])[:,:,0]
train_data_timesteps.shape
```

```python
# Converting testing data to 3D tensor

test_data_timesteps=np.array([[j for j in test_data[i:i+timesteps]] for i in range(0,len(test_data)-timesteps+1)])[:,:,0]
test_data_timesteps.shape
```

```python
# Selecting inputs and outputs from training and testing data

x_train, y_train = train_data_timesteps[:,:timesteps-1],train_data_timesteps[:,[timesteps-1]]
x_test, y_test = test_data_timesteps[:,:timesteps-1],test_data_timesteps[:,[timesteps-1]]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
```


### Implement SVR

It's time to implement SVR, which you'll do from the `SVR` library that you installed earlier.

Now you need to follow several steps

      1. Define the model by calling `SVR()` and passing in the model hyperparameters: kernel, gamma and c
      2. Prepare the model for the training data by calling the fit() function.
      3. Make predictions calling the `predict()` function



**1. Create an SVR model**

```python
model = SVR(kernel='rbf',gamma=0.5, C=10)
```

**2. Fit the model on training data**

```python
model.fit(x_train, y_train[:,0])
```

**3. Make model predictions**

```python
y_train_pred = model.predict(x_train).reshape(-1,1)
y_test_pred = model.predict(x_test).reshape(-1,1)

print(y_train_pred.shape, y_test_pred.shape)
```



You've built your SVR! Now we need to find a way to evaluate it.

### Evaluate your model

This process provides a more robust estimation of how the model will perform in practice. However, it comes at the computation cost of creating so many models. This is acceptable if the data is small or if the model is simple, but could be an issue at scale.

2. Scale the predicted and original output

    ```python
    # Scaling the predictions
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    
    print(len(y_train_pred), len(y_test_pred))
    ```
    
    ```python
    # Scaling the original values
    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)
    
    print(len(y_train), len(y_test))
    ```
    
    ### Check model performance
    
    **Extract the timesteps for x-axis**
    
    ```python
    # Extract the timesteps for x-axis
    
    train_timestamps = energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)].index[timesteps-1:]
    test_timestamps = energy[test_start_dt:].index[timesteps-1:]
    
    print(len(train_timestamps), len(test_timestamps))
    ```
    
    **Plot the predictions**
    
    ```python
    plt.figure(figsize=(25,6))
    plt.plot(train_timestamps, y_train, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(train_timestamps, y_train_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.title("Training data prediction")
    plt.show()
    ```
    
    ![training and testing data](images/train-data-predict.png)
    
    **Print MAPE for training data**
    
    ```python
    print('MAPE for training data: ', mape(y_train_pred, y_train)*100, '%')
    ```
    
    ​	MAPE for training data:  2.6702350467033176 %
    
    **Plot the predictions**
    
    ```python
    plt.figure(figsize=(10,3))
    plt.plot(test_timestamps, y_test, color = 'red', linewidth=2.0, alpha = 0.6)
    plt.plot(test_timestamps, y_test_pred, color = 'blue', linewidth=0.8)
    plt.legend(['Actual','Predicted'])
    plt.xlabel('Timestamp')
    plt.show()
    ```
    
    ![training and testing data](images/test-data-predict.png)
    
    **Print MAPE for testing data**
    
    ```python
    print('MAPE for testing data: ', mape(y_test_pred, y_test)*100, '%')
    ```
    
    ​	MAPE for testing data:  1.4628890659719878 %