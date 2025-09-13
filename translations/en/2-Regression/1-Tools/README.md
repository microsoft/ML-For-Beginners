<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-06T10:46:54+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "en"
}
-->
# Get started with Python and Scikit-learn for regression models

![Summary of regressions in a sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

> ### [This lesson is available in R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduction

In these four lessons, you will learn how to build regression models. We'll discuss their purpose shortly. But before diving in, ensure you have the necessary tools set up to begin!

In this lesson, you will:

- Set up your computer for local machine learning tasks.
- Work with Jupyter notebooks.
- Install and use Scikit-learn.
- Explore linear regression through a hands-on exercise.

## Installations and configurations

[![ML for beginners - Setup your tools ready to build Machine Learning models](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML for beginners -Setup your tools ready to build Machine Learning models")

> 🎥 Click the image above for a short video on configuring your computer for ML.

1. **Install Python**. Ensure that [Python](https://www.python.org/downloads/) is installed on your computer. Python is essential for many data science and machine learning tasks. Most systems already have Python installed. You can also use [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) to simplify the setup process.

   Some Python tasks require specific versions of the software. To manage this, it's helpful to work within a [virtual environment](https://docs.python.org/3/library/venv.html).

2. **Install Visual Studio Code**. Make sure Visual Studio Code is installed on your computer. Follow these instructions to [install Visual Studio Code](https://code.visualstudio.com/) for the basic setup. Since you'll use Python in Visual Studio Code during this course, you might want to review how to [configure Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) for Python development.

   > Familiarize yourself with Python by exploring this collection of [Learn modules](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Setup Python with Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Setup Python with Visual Studio Code")
   >
   > 🎥 Click the image above for a video on using Python within VS Code.

3. **Install Scikit-learn** by following [these instructions](https://scikit-learn.org/stable/install.html). Since Python 3 is required, using a virtual environment is recommended. If you're installing this library on an M1 Mac, refer to the special instructions on the linked page.

4. **Install Jupyter Notebook**. You'll need to [install the Jupyter package](https://pypi.org/project/jupyter/).

## Your ML authoring environment

You'll use **notebooks** to develop Python code and create machine learning models. Notebooks are a popular tool for data scientists, identifiable by their `.ipynb` extension.

Notebooks provide an interactive environment where developers can code, add notes, and write documentation alongside their code—ideal for experimental or research-oriented projects.

[![ML for beginners - Set up Jupyter Notebooks to start building regression models](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML for beginners - Set up Jupyter Notebooks to start building regression models")

> 🎥 Click the image above for a short video on setting up Jupyter Notebooks.

### Exercise - work with a notebook

In this folder, you'll find the file _notebook.ipynb_.

1. Open _notebook.ipynb_ in Visual Studio Code.

   A Jupyter server will start with Python 3+. You'll find areas of the notebook containing `run` sections of code. You can execute a code block by clicking the play button icon.

2. Select the `md` icon and add some markdown with the text **# Welcome to your notebook**.

   Next, add some Python code.

3. Type **print('hello notebook')** in the code block.
4. Click the arrow to run the code.

   You should see the printed output:

    ```output
    hello notebook
    ```

![VS Code with a notebook open](../../../../2-Regression/1-Tools/images/notebook.jpg)

You can intersperse your code with comments to document the notebook.

✅ Reflect on how a web developer's working environment differs from that of a data scientist.

## Up and running with Scikit-learn

Now that Python is set up locally and you're comfortable with Jupyter notebooks, let's get familiar with Scikit-learn (pronounced `sci` as in `science`). Scikit-learn offers an [extensive API](https://scikit-learn.org/stable/modules/classes.html#api-ref) for performing ML tasks.

According to its [website](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities."

In this course, you'll use Scikit-learn and other tools to build machine learning models for 'traditional machine learning' tasks. Neural networks and deep learning are excluded, as they are covered in our upcoming 'AI for Beginners' curriculum.

Scikit-learn simplifies model building and evaluation. It primarily focuses on numeric data and includes several ready-made datasets for learning. It also provides pre-built models for experimentation. Let's explore how to load prepackaged data and use a built-in estimator to create your first ML model with Scikit-learn.

## Exercise - your first Scikit-learn notebook

> This tutorial was inspired by the [linear regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) on Scikit-learn's website.

[![ML for beginners - Your First Linear Regression Project in Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML for beginners - Your First Linear Regression Project in Python")

> 🎥 Click the image above for a short video on this exercise.

In the _notebook.ipynb_ file associated with this lesson, clear all cells by clicking the 'trash can' icon.

In this section, you'll work with a small diabetes dataset built into Scikit-learn for learning purposes. Imagine testing a treatment for diabetic patients. Machine learning models can help identify which patients might respond better to the treatment based on variable combinations. Even a basic regression model, when visualized, can reveal insights about variables that could guide clinical trials.

✅ There are various regression methods, and your choice depends on the question you're trying to answer. For example, if you want to predict someone's probable height based on their age, you'd use linear regression to find a **numeric value**. If you're determining whether a cuisine is vegan, you'd use logistic regression for a **category assignment**. Think about questions you could ask of data and which method would be most suitable.

Let's begin.

### Import libraries

For this task, we'll import the following libraries:

- **matplotlib**: A useful [graphing tool](https://matplotlib.org/) for creating visualizations.
- **numpy**: [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) is a library for handling numeric data in Python.
- **sklearn**: The [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) library.

Import the libraries you'll need for this task.

1. Add imports by typing the following code:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Here, you're importing `matplotlib`, `numpy`, and `datasets`, `linear_model`, and `model_selection` from `sklearn`. `model_selection` is used for splitting data into training and test sets.

### The diabetes dataset

The built-in [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) contains 442 samples of diabetes-related data with 10 feature variables, including:

- age: age in years
- bmi: body mass index
- bp: average blood pressure
- s1 tc: T-Cells (a type of white blood cells)

✅ This dataset includes 'sex' as a feature variable, which is significant in diabetes research. Many medical datasets use binary classifications like this. Consider how such categorizations might exclude certain populations from treatments.

Now, load the X and y data.

> 🎓 Remember, this is supervised learning, so we need a labeled 'y' target.

In a new code cell, load the diabetes dataset by calling `load_diabetes()`. The input `return_X_y=True` ensures `X` is a data matrix and `y` is the regression target.

1. Add print commands to display the shape of the data matrix and its first element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    The response is a tuple. You're assigning the first two values of the tuple to `X` and `y`. Learn more [about tuples](https://wikipedia.org/wiki/Tuple).

    You'll see the data has 442 items shaped into arrays of 10 elements:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Reflect on the relationship between the data and the regression target. Linear regression predicts relationships between feature X and target variable y. Can you find the [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) for the diabetes dataset in the documentation? What does this dataset demonstrate, given the target?

2. Select a portion of the dataset to plot by choosing the 3rd column. Use the `:` operator to select all rows, and then select the 3rd column using the index (2). Reshape the data into a 2D array for plotting using `reshape(n_rows, n_columns)`. If one parameter is -1, the corresponding dimension is calculated automatically.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Print the data at any time to check its shape.

3. With the data ready for plotting, use a machine to determine a logical split between the numbers in the dataset. Split both the data (X) and the target (y) into test and training sets. Scikit-learn provides a simple way to do this; you can split your test data at a specific point.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Train your model! Load the linear regression model and train it with your X and y training sets using `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` is a function you'll encounter in many ML libraries like TensorFlow.

5. Create a prediction using test data with the `predict()` function. This will help draw the line between data groups.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Finally, visualize the data using Matplotlib. Create a scatterplot of all the X and y test data, and use the prediction to draw a line that best separates the data groups.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![a scatterplot showing datapoints around diabetes](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Think about what's happening here. A straight line is passing through many small data points, but what is its purpose? Can you see how this line can help predict where a new, unseen data point might align with the plot's y-axis? Try to describe the practical application of this model in your own words.

Congratulations, you've built your first linear regression model, made a prediction with it, and visualized it in a plot!

---
## 🚀Challenge

Plot a different variable from this dataset. Hint: modify this line: `X = X[:,2]`. Considering the target of this dataset, what insights can you gain about the progression of diabetes as a disease?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

In this tutorial, you worked with simple linear regression, rather than univariate or multiple linear regression. Take some time to read about the differences between these methods, or watch [this video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Learn more about the concept of regression and reflect on the types of questions this technique can help answer. Consider taking [this tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) to deepen your understanding.

## Assignment

[A different dataset](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.