# Get started with Python and Scikit-learn for regression models

![Ringkisan regresi dalam sebuah catatan sketsa](../../sketchnotes/ml-regression.png)

> Catatan sketsa oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuis Pra-ceramah](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/9/)
## Pembukaan

Dalam keempat pelajaran ini, kamu akan belajar bagaimana membangun model regresi. Kita akan berdiskusi apa fungsi model tersebut dalam sejenak. Tetapi sebelum kamu melakukan apapun, pastikan bahwa kamu sudah mempunyai alat-alat yang diperlukan untuk memulai!

Dalam pelajaran ini, kamu akan belajar bagaimana untuk:

- Konfigurasi komputermu untuk tugas pembelajaran.
- Bekerja dengan Jupyter notebooks.
- Menggunakan Scikit-learn, termasuk instalasi.
- Menjelajahi regresi linear dengan latihan *hands-on*.


## Instalasi dan konfigurasi

[![Menggunakan Python dalam Visual Studio Code](https://img.youtube.com/vi/7EXd4_ttIuw/0.jpg)](https://youtu.be/7EXd4_ttIuw "Menggunakan Python dalam Visual Studio Code")

> 🎥 Klik foto di atas untuk sebuah video: menggunakan Python dalam VS Code

1. **Pasang Python**. Pastikan bahwa [Python](https://www.python.org/downloads/) telah dipasang di komputermu. Kamu akan menggunakan Python untuk banyak tugas *data science* dan *machine learning*. Kebanyakan sistem komputer sudah diinstal dengan Python. Adapula *[Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-15963-cxa)* yang berguna untuk membantu proses pemasangan untuk beberapa pengguna.

   Beberapa penggunaan Python memerlukan satu versi perangkat lunak tersebut, sedangkan beberapa penggunaan lainnya mungkin memerlukan versi Python yang beda lagi. Oleh sebab itulah akan sangat berguna untuk bekerja dalam sebuah *[virtual environment](https://docs.python.org/3/library/venv.html)* (lingkungan virtual).

2. **Pasang Visual Studio Code**. Pastikan kamu sudah memasangkan Visual Studio Code di komputermu. Ikuti instruksi-instruksi ini untuk [memasangkan Visual Studio Code](https://code.visualstudio.com/) untuk instalasi dasar. Kamu akan menggunakan Python dalam Visual Studio Code dalam kursus ini, jadi kamu mungkin akan ingin mencari tahu cara [mengkonfigurasi Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-15963-cxa) untuk menggunakan Python.

   > Nyamankan diri dengan Python dengan mengerjakan [koleksi modul pembelajaran ini](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-15963-cxa)

3. **Pasang Scikit-learn**, dengan mengikuti [instruksi di sini](https://scikit-learn.org/stable/install.html). Karena harus dipastikan bahwa kamu sedang menggunakan Python 3, kami anjurkan kamu menggunakan sebuah *virtual environment*. Ingatlah juga bahwa jika kamu ingin memasangkan ini di sebuah M1 Mac, ada instruksi khusus dalam laman yang ditautkan di atas.

4. **Pasang Jupyter Notebook**. Kamu akan harus [memasang paket Jupyter](https://pypi.org/project/jupyter/). 

## Lingkungan penulisan ML-mu

Kamu akan menggunakan ***notebooks*** untuk bekerja dengan kode Python-mu dan membuat model *machine learning*-mu. Jenis file ini adalah alat yang sering digunakan *data scientists* dan dapat diidentifikasikan dengan akhiran/ekstensi `.ipynb`. 

Notebooks are an interactive environment that allow the developer to both code and add notes and write documentation around the code which is quite helpful for experimental or research-oriented projects.

### Exercise - work with a notebook

In this folder, you will find the file _notebook.ipynb_. 

1. Open _notebook.ipynb_ in Visual Studio Code.

   A Jupyter server will start with Python 3+ started. You will find areas of the notebook that can be `run`, pieces of code. You can run a code block, by selecting the icon that looks like a play button.

1. Select the `md` icon and add a bit of markdown, and the following text **# Welcome to your notebook**.

   Next, add some Python code. 

1. Type **print('hello notebook')** in the code block.
1. Select the arrow to run the code.

   You should see the printed statement:

    ```output
    hello notebook
    ```

![VS Code with a notebook open](images/notebook.png)

You can interleaf your code with comments to self-document the notebook.

✅ Think for a minute how different a web developer's working environment is versus that of a data scientist.

## Up and running with Scikit-learn

Now that Python is set up in your local environment, and you are comfortable with Jupyter notebooks, let's get equally comfortable with Scikit-learn (pronounce it `sci` as in `science`). Scikit-learn provides an [extensive API](https://scikit-learn.org/stable/modules/classes.html#api-ref) to help you perform ML tasks.

According to their [website](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities."

In this course, you will use Scikit-learn and other tools to build machine learning models to perform what we call 'traditional machine learning' tasks. We have deliberately avoided neural networks and deep learning, as they are better covered in our forthcoming 'AI for Beginners' curriculum. 

Scikit-learn makes it straightforward to build models and evaluate them for use. It is primarily focused on using numeric data and contains several ready-made datasets for use as learning tools. It also includes pre-built models for students to try. Let's explore the process of loading prepackaged data and using a built in estimator  first ML model with Scikit-learn with some basic data.

## Exercise - your first Scikit-learn notebook

> This tutorial was inspired by the [linear regression example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) on Scikit-learn's web site.

In the _notebook.ipynb_ file associated to this lesson, clear out all the cells by pressing the 'trash can' icon.

In this section, you will work with a small dataset about diabetes that is built into Scikit-learn for learning purposes. Imagine that you wanted to test a treatment for diabetic patients. Machine Learning models might help you determine which patients would respond better to the treatment, based on combinations of variables. Even a very basic regression model, when visualized, might show information about variables that would help you organize your theoretical clinical trials.

✅ There are many types of regression methods, and which one you pick depends on the answer you're looking for. If you want to predict the probable height for a person of a given age, you'd use linear regression, as you're seeking a **numeric value**. If you're interested in discovering whether a type of cuisine should be considered vegan or not, you're looking for a **category assignment** so you would use logistic regression. You'll learn more about logistic regression later. Think a bit about some questions you can ask of data, and which of these methods would be more appropriate.

Let's get started on this task.

### Import libraries

For this task we will import some libraries:

- **matplotlib**. It's a useful [graphing tool](https://matplotlib.org/) and we will use it to create a line plot.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) is a useful library for handling numeric data in Python.
- **sklearn**. This is the Scikit-learn library.

Import some libraries to help with your tasks.

1. Add imports by typing the following code:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Above you are importing `matplottlib`, `numpy` and you are importing `datasets`, `linear_model` and `model_selection` from `sklearn`. `model_selection` is used for splitting data into training and test sets.

### The diabetes dataset

The built-in [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) includes 442 samples of data around diabetes, with 10 feature variables, some of which include:

age: age in years
bmi: body mass index
bp: average blood pressure
s1 tc: T-Cells (a type of white blood cells)

✅ This dataset includes the concept of 'sex' as a feature variable important to research around diabetes. Many medical datasets include this type of binary classification. Think a bit about how categorizations such as this might exclude certain parts of a population from treatments.

Now, load up the X and y data.

> 🎓 Remember, this is supervised learning, and we need a named 'y' target.

In a new code cell, load the diabetes dataset by calling `load_diabetes()`. The input `return_X_y=True` signals that `X` will be a data matrix, and `y` will be the regression target. 

1. Add some print commands to show the shape of the data matrix and its first element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    What you are getting back as a response, is a tuple. What you are doing is to assign the two first values of the tuple to `X` and `y` respectively. Learn more [about tuples](https://wikipedia.org/wiki/Tuple).

    You can see that this data has 442 items shaped in arrays of 10 elements:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Think a bit about the relationship between the data and the regression target. Linear regression predicts relationships between feature X and target variable y. Can you find the [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) for the diabetes dataset in the documentation? What is this dataset demonstrating, given that target?

2. Next, select a portion of this dataset to plot by arranging it into a new array using numpy's `newaxis` function. We are going to use linear regression to generate a line between values in this data, according to a pattern it determines.

   ```python
   X = X[:, np.newaxis, 2]
   ```

   ✅ At any time, print out the data to check its shape.

3. Now that you have data ready to be plotted, you can see if a machine can help determine a logical split between the numbers in this dataset. To do this, you need to split both the data (X) and the target (y) into test and training sets. Scikit-learn has a straightforward way to do this; you can split your test data at a given point.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Now you are ready to train your model! Load up the linear regression model and train it with your X and y training sets using `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` is a function you'll see in many ML libraries such as TensorFlow

5. Then, create a prediction using test data, using the function `predict()`. This will be used to draw the line between data groups

    ```python
    y_pred = model.predict(X_test)
    ```

6. Now it's time to show the data in a plot. Matplotlib is a very useful tool for this task. Create a scatterplot of all the X and y test data, and use the prediction to draw a line in the most appropriate place, between the model's data groupings.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()
    ```

   ![a scatterplot showing datapoints around diabetes](./images/scatterplot.png)

   ✅ Think a bit about what's going on here. A straight line is running through many small dots of data, but what is it doing exactly? Can you see how you should be able to use this line to predict where a new, unseen data point should fit in relationship to the plot's y axis? Try to put into words the practical use of this model.

Congratulations, you built your first linear regression model, created a prediction with it, and displayed it in a plot!

---
## 🚀Challenge

Plot a different variable from this dataset. Hint: edit this line: `X = X[:, np.newaxis, 2]`. Given this dataset's target, what are you able to discover about the progression of diabetes as a disease?
## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/10/)

## Review & Self Study

In this tutorial, you worked with simple linear regression, rather than univariate or multiple linear regression. Read a little about the differences between these methods, or take a look at [this video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Read more about the concept of regression and think about what kinds of questions can be answered by this technique. Take this [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-15963-cxa) to deepen your understanding.

## Assignment 

[A different dataset](assignment.md)
