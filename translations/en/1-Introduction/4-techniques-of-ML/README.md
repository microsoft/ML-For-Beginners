<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-06T10:53:40+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "en"
}
-->
# Techniques of Machine Learning

The process of creating, using, and maintaining machine learning models and the data they rely on is quite different from many other development workflows. In this lesson, we will break down the process and outline the key techniques you need to understand. You will:

- Gain a high-level understanding of the processes behind machine learning.
- Explore foundational concepts such as 'models,' 'predictions,' and 'training data.'

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Click the image above for a short video walkthrough of this lesson.

## Introduction

At a high level, the process of creating machine learning (ML) systems involves several steps:

1. **Define the question**. Most ML processes begin with a question that cannot be answered using a simple conditional program or rules-based system. These questions often focus on making predictions based on a dataset.
2. **Collect and prepare data**. To answer your question, you need data. The quality and sometimes the quantity of your data will determine how well you can address your initial question. Visualizing data is an important part of this phase. This phase also includes splitting the data into training and testing sets to build a model.
3. **Select a training method**. Depending on your question and the nature of your data, you need to choose how to train a model to best represent your data and make accurate predictions. This step often requires specific expertise and a significant amount of experimentation.
4. **Train the model**. Using your training data, you'll apply various algorithms to train a model to recognize patterns in the data. The model may use internal weights that can be adjusted to prioritize certain parts of the data over others to improve its performance.
5. **Evaluate the model**. You use unseen data (your testing data) from your dataset to assess how well the model performs.
6. **Tune parameters**. Based on the model's performance, you can repeat the process using different parameters or variables that control the behavior of the algorithms used to train the model.
7. **Make predictions**. Use new inputs to test the model's accuracy.

## What question to ask

Computers excel at uncovering hidden patterns in data. This capability is particularly useful for researchers who have questions about a specific domain that cannot be easily answered by creating a rules-based system. For example, in an actuarial task, a data scientist might create handcrafted rules to analyze the mortality rates of smokers versus non-smokers.

However, when many other variables are introduced, an ML model might be more effective at predicting future mortality rates based on past health data. A more optimistic example could involve predicting the weather for April in a specific location using data such as latitude, longitude, climate change, proximity to the ocean, jet stream patterns, and more.

✅ This [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) on weather models provides a historical perspective on using ML for weather analysis.

## Pre-building tasks

Before you start building your model, there are several tasks you need to complete. To test your question and form a hypothesis based on the model's predictions, you need to identify and configure several elements.

### Data

To answer your question with confidence, you need a sufficient amount of data of the right type. At this stage, you need to:

- **Collect data**. Referencing the previous lesson on fairness in data analysis, collect your data carefully. Be mindful of its sources, any inherent biases, and document its origin.
- **Prepare data**. Data preparation involves several steps. You may need to combine and normalize data from different sources. You can enhance the quality and quantity of your data through methods like converting strings to numbers (as seen in [Clustering](../../5-Clustering/1-Visualize/README.md)). You might also generate new data based on the original (as seen in [Classification](../../4-Classification/1-Introduction/README.md)). You can clean and edit the data (as we will do before the [Web App](../../3-Web-App/README.md) lesson). Additionally, you may need to randomize and shuffle the data depending on your training techniques.

✅ After collecting and processing your data, take a moment to evaluate whether its structure will allow you to address your intended question. Sometimes, the data may not perform well for your specific task, as we discover in our [Clustering](../../5-Clustering/1-Visualize/README.md) lessons!

### Features and Target

A [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) is a measurable property of your data. In many datasets, it is represented as a column heading like 'date,' 'size,' or 'color.' Feature variables, often represented as `X` in code, are the input variables used to train the model.

A target is what you are trying to predict. Targets, usually represented as `y` in code, are the answers to the questions you are asking of your data: In December, what **color** pumpkins will be cheapest? In San Francisco, which neighborhoods will have the best real estate **prices**? Sometimes, the target is also referred to as the label attribute.

### Selecting your feature variable

🎓 **Feature Selection and Feature Extraction** How do you decide which variables to use when building a model? You will likely go through a process of feature selection or feature extraction to identify the best variables for the most effective model. These processes differ: "Feature extraction creates new features from functions of the original features, whereas feature selection returns a subset of the features." ([source](https://wikipedia.org/wiki/Feature_selection))

### Visualize your data

Visualization is a powerful tool in a data scientist's toolkit. Libraries like Seaborn or MatPlotLib allow you to represent your data visually, which can help uncover hidden correlations that you can leverage. Visualizations can also reveal bias or imbalances in your data (as seen in [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Split your dataset

Before training, you need to divide your dataset into two or more parts of unequal size that still represent the data well.

- **Training**. This portion of the dataset is used to train your model. It typically constitutes the majority of the original dataset.
- **Testing**. A test dataset is an independent subset of the original data used to validate the model's performance.
- **Validating**. A validation set is a smaller independent subset used to fine-tune the model's hyperparameters or architecture to improve its performance. Depending on the size of your data and the question you are addressing, you may not need to create this third set (as noted in [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Building a model

Using your training data, your goal is to build a model—a statistical representation of your data—using various algorithms to **train** it. Training a model exposes it to data, enabling it to identify patterns, validate them, and accept or reject them.

### Decide on a training method

Depending on your question and the nature of your data, you will select a method to train the model. By exploring [Scikit-learn's documentation](https://scikit-learn.org/stable/user_guide.html)—which we use in this course—you can examine various ways to train a model. Depending on your experience, you may need to try multiple methods to build the best model. Data scientists often evaluate a model's performance by testing it with unseen data, checking for accuracy, bias, and other issues, and selecting the most suitable training method for the task.

### Train a model

With your training data, you are ready to 'fit' it to create a model. In many ML libraries, you will encounter the code 'model.fit'—this is where you input your feature variable as an array of values (usually 'X') and a target variable (usually 'y').

### Evaluate the model

Once the training process is complete (it may require many iterations, or 'epochs,' to train a large model), you can evaluate the model's quality using test data to measure its performance. This test data is a subset of the original data that the model has not previously analyzed. You can generate a table of metrics to assess the model's quality.

🎓 **Model fitting**

In machine learning, model fitting refers to how accurately the model's underlying function analyzes data it has not encountered before.

🎓 **Underfitting** and **overfitting** are common issues that reduce a model's quality. An underfit model fails to analyze both its training data and unseen data accurately. An overfit model performs too well on training data because it has learned the data's details and noise excessively. Both scenarios lead to poor predictions.

![overfitting model](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infographic by [Jen Looper](https://twitter.com/jenlooper)

## Parameter tuning

After initial training, evaluate the model's quality and consider improving it by adjusting its 'hyperparameters.' Learn more about this process [in the documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediction

This is the stage where you use entirely new data to test your model's accuracy. In an applied ML setting, such as building web applications for production, this process might involve gathering user input (e.g., a button press) to set a variable and send it to the model for inference or evaluation.

In these lessons, you will learn how to prepare, build, test, evaluate, and predict—covering all the steps of a data scientist and more as you progress toward becoming a 'full stack' ML engineer.

---

## 🚀Challenge

Create a flow chart illustrating the steps of an ML practitioner. Where do you see yourself in this process right now? Where do you anticipate challenges? What seems straightforward to you?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Search online for interviews with data scientists discussing their daily work. Here is [one](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Assignment

[Interview a data scientist](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.