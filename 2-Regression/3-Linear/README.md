# Build a Regression Model using Scikit-Learn: Regression Two Ways
## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/9/)
### Introduction

So far you have explored what regression is with sample data gathered from the pumpkin pricing dataset that we will use throughout this unit. You have also visualized it using Matplotlib. Now you are ready to dive deeper into regression for ML. In this lesson, you will learn more about two types of regression: simple regression and polynomial regression, along with some of the math underlying these techniques. 

> Throughout this curriculum, we assume minimal knowledge of math, and seek to make it very accessible for students coming from other fields, so watch for notes, callouts, diagrams, and other learning tools to aid in comprehension.
### Prerequisite

You should be familiar by now with the structure of the pumpkin data that we are examining. You can find it preloaded and pre-cleaned in this lesson's notebook.ipynb files, with the pumpkin price displayed per bushel in a new dataframe.  Make sure you can run these notebooks in kernels in VS Code.
### Preparation

As a reminder, you are loading this data so as to ask questions of it. When is the best time to buy pumpkins? What price can I expect of a miniature pumpkin? Should I buy them in half-bushel baskets or by the 1 1/9 bushel box? Let's keep digging into this data.

In the previous lesson, you created a Pandas dataframe and populated it with part of the original dataset, standardizing the pricing by the bushel. By doing that, however, you were only able to gather about 400 datapoints and only for the fall months. Take a look at the data that we preloaded in this lesson's accompanying notebook. Maybe we can get a little more detail about the nature of the data by cleaning it more.
## A Linear Regression Line

As you learned in Lesson 1, the goal of a linear regression exercise is to be able to plot a line to show the relationship between variables and make accurate predictions on where a new datapoint would fall in relationship to that line. 

> **🧮 Show me the math** 
> 
> This line has an equation: `Y = a + bX`. It is typical of **Least-Squares Regression** to draw this type of line. 
>
> `X` is the 'explanatory variable'. `Y` is the 'dependent variable'. The slope of the line is `b` and `a` is the intercept, which refers to the value of `Y` when `X = 0`. 
>
> In other words, and referring to our pumpkin data's original question: "predict the price of a pumpkin per bushel by month", `X` would refer to the price and `Y` would refer to the month of sale. The math that calculates the line must demonstrate the slope of the line, which is also dependent on the intercept, or where `Y` is situated when `X = 0`.
>
> You can observe the method of calculation for these values on the [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) web site.
>
> A common method of regression is **Least-Squares Regression** which means that all the datapoints surounding the regression line are squared and then added up. Ideally, that final sum is as small as possible, because we want a low number of errors, or `least-squares`.
>
> One more term to understand is the **Correlation Coefficient** between given X and Y variables. For a scatterplot, you can quickly visualize this coefficient. A plot with datapoints scattered in a neat line have high correlation, but a plot with datapoints scattered everywhere between X and Y have a low correlation.
>
> A good regression model will be one that has a low (nearly zero) Correlation Coefficient using the Least-Squares Regression method with a line of regression.

✅ Run the notebook accompanying this lesson. Does the data associating City to Price for pumpkin sales seem to have high or low correlation, according to your visual interpretation of the scatterplot?
## Create a Linear Regression Model correlating Pumpkin Datapoints

Now that you have an understanding of the math behind this exercise, create a Regression model to see if you can predict which type of pumpkins will have the best pumpkin prices. Someone buying pumpkins for a holiday pumpkin patch might want this information to be able to pre-order the best-priced pumpkins for the patch (normally there is a mix of miniature and large pumpkins in a patch).

Since you'll use Scikit-Learn, there's no reason to do this by hand (although you could!). In the main data-processing block of your lesson notebook, add a library from Scikit-Learn to automatically convert all string data to numbers:

```python
from sklearn.preprocessing import LabelEncoder
...
new_pumpkins.iloc[:, 0:-1] = new_pumpkins.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
new_pumpkins.iloc[:, 0:-1] = new_pumpkins.iloc[:, 0:-1].apply(LabelEncoder().fit_transform)
```

If you look at the new_pumpkins dataframe now, you see that all the strings are now numeric. This makes it harder to read but much more intelligible to Scikit-Learn!

Now, you can make more educated decisions (not just based on eyeballing a scatterplot) about the data that is best suited to regression. `

Try to find a good correlation between two points of your data. As it turns out, there's only weak correlation between the City and Price:

```python
print(new_pumpkins['City'].corr(new_pumpkins['Price']))
0.3236397181608923
```
However there's a better correlation between the Variety and its Price (makes sense, right? Think about miniature pumpkin prices vs. the big pumpkins you might buy for Halloween. The little ones are more expensive, volume-wise, than the big ones)

```python
print(new_pumpkins['Variety'].corr(new_pumpkins['Price']))
-0.8634790400214403
```
This is a negative correlation, meaning the slope heads downhill, but it's still useful. So, a question to ask of this data will be: 'What price can I expect of a given type of pumpkin?'

Let's build this regression model
## Building the model

Before building your model, do one more tidy-up of your data. Drop any null data and check once more what the data looks like.

```python
new_pumpkins.dropna(inplace=True)
new_pumpkins.info()
```

Then, create a new dataframe from this minimal set:

```python
new_columns = ['Variety', 'Price']
ml_pumpkins = new_pumpkins.drop([c for c in new_pumpkins.columns if c not in new_columns], axis='columns')

ml_pumpkins

```

Now you can assign your X and y coordinate data:

```python
X = ml_pumpkins.values[:, :1]
y = ml_pumpkins.values[:, 1:2]
```
> What's going on here? You're using [Python slice notation](https://stackoverflow.com/questions/509211/understanding-slice-notation/509295#509295) to create arrays to populate `X` and `y`.

Next, start the regression model-building routines:

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

pred = lin_reg.predict(X_test)

accuracy_score = lin_reg.score(X_train,y_train)
print('Model Accuracy: ', accuracy_score)

# The coefficients
print('Coefficients: ', lin_reg.coef_)
# The mean squared error
print('Mean squared error: ',
      mean_squared_error(y_test, pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: ',
      r2_score(y_test, pred)) 
```

Because there's a reasonably high correlation between the two variables, there accuracy of this model isn't too bad!

```
Model Accuracy:  0.7327987875929955
Coefficients:  [[-8.54296764]]
Mean squared error:  23.443815358076087
Coefficient of determination:  0.7802537224707632
```

You can visualize the line that's drawn in the process:

```python
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

Congratulations, you just created a model that can help predict the price of a few varieties of pumpkins. Your holiday pumpkin patch will be beautiful.
## Polynomial Regression

Another type of Linear Regression is Polynomial Regression. While sometimes there's a linear relationship between variables - the bigger the pumpkin in volume, the higher the price - sometimes these relationships can't be plotted as a plane or straight line. Take another look at the relationship between City to Price in the new_pumpkins data.

✅ Here are [some more examples](https://online.stat.psu.edu/stat501/lesson/9/9.8) of data that could use Polynomial Regression

```python
import matplotlib.pyplot as plt
plt.scatter('City','Price',data=new_pumpkins)
```
Does the resultant scatterplot seem like it could be analyzed by a straight line? Perhaps not. In this case, you should try Polynomial Regression.

✅ Polynomials are mathematical expressions that might consist of one or more variables and coefficients

🚀 Challenge: Test several different variables in this notebook to see how correlation corresponds to model accuracy.

## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/10/)

## Review & Self Study

In this lesson we learned about linear regression 

**Assignment**: [Assignment Name](assignment.md)
