## Dataset 
Dataset is of retail company that collected data from their customers wheather or not they purchased some product.
Each row belong to different customers and includes customer name, age, salary and wheather or not they purchased some product (YES or NO).

x --> freatures  (first three colums) </br>
y --> dependent variable vector (last column)

- - - - -

## Importing dataset
> What is the difference between the independent variables and the dependent variable?<br>

The independent variables are the input data that you have, with each you want to predict something. That
something is the dependent variable.

<br>

> In Python, why do we create X and y separately?

Because we want to work with Numpy arrays, instead of Pandas dataframes. Numpy arrays are the most
convenient format to work with when doing data preprocessing and building Machine Learning models. So
we create two separate arrays, one that contains our independent variables (also called the input features),
and another one that contains our dependent variable (what we want to predict).

<br>

> In Python, what does ’iloc’ exactly do?

It locates the column by its index. In other words, using ’iloc’ allows us to take columns by just taking their
index.

<br>

> In Python, what does ’.values’ exactly do?

It returns the values of the columns you are taking (by their index) inside a Numpy array. That is basically
how X and y become Numpy arrays

<br>

- - - - -

## Taking care of missing data
> In Python, what is the difference between fit and transform?

The fit part is used to extract some info of the data on which the object is applied (here, Imputer will
spot the missing values and get the mean of the column). Then, the transform part is used to apply some
transformation (here, Imputer will replace the missing value by the mean).


<br>

- - - - -

## Encoding categorical data
> In Python, what do the two ’fit_transform’ methods do?

When the ’fit_transform()’ method is called from the LabelEncoder() class, it transforms the categories
strings into integers. For example, it transforms France, Spain and Germany into 0, 1 and 2. Then, when
the ’fit_transform()’ method is called from the OneHotEncoder() class, it creates separate columns for each
different labels with binary values 0 and 1. Those separate columns are the dummy variables.

<br>

- - - - -

## Splitting the dataset into the Training set and Test set
> What is the difference between the training set and the test set?


The training set is a subset of your data on which your model will learn how to predict the dependent
variable with the independent variables. The test set is the complimentary subset from the training set, on
which you will evaluate your model to see if it manages to predict correctly the dependent variable with the
independent variables.

<br>

> Why do we split on the dependent variable?


Because we want to have well distributed values of the dependent variable in the training and test set. For
example if we only had the same value of the dependent variable in the training set, our model wouldn’t be
able to learn any correlation between the independent and dependent variables.

<br>

- - - - -

## Feature scaling

> Do we really have to apply Feature Scaling on the dummy variables?


Yes, if you want to optimize the accuracy of your model predictions.
No, if you want to keep the most interpretation as possible in your model.

<br>

> When should we use Standardization and Normalization?

Generally you should normalize (normalization) when the data is normally distributed, and scale (standardization) 
when the data is not normally distributed. In doubt, you should go for standardization. Howeverwhat is commonly 
done is that the two scaling methods are tested.
