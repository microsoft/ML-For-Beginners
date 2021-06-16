In this lesson, you will learn:

- the process of doing machine learning at a high level.

# Introduction

On a high-level the craft of doing machine learning, ML goes through a number of steps. Here's what it looks like:

1. **Decide on the question**. You have a question you want answered, here's where you decide what that question should be.
1. **Collect and prepare data**. To be able to answer your question, you need data, lots of it.
1. **Choose a model**. A model is the same thing as an algorithm and you need to train in order for it to recognize what you need it to recognize.
1. **Train the model**. You take part of your collected data and make sure the model changes basd on its input. The model has internal weights that gets adjusted based on what you feed it.
1. **Evaluate the model**. You use never before seen data from your collected data to see how the model is performing.
1. **Parameter tuning**. 
1. **Predict**, make predictions with your model based on new input.

## What question to ask

Ok, so why are you doing machine learning? Well, you have a question you want to ask, like if there's a correlation between your living habits and diabetes, maybe age is a factor. You think about it, you have your question, you want to know what causes diabetes. 

### Identifying factors

You have a hypotheses on what might cause diabetes like age, living habits, maybe a gene is involved. Great you are off to a great start. But to be able to get further, you need data, lots of data, the more the better.

## Data

To be able to answer your question with any kind of certainty, you need a lot of data, and the right type. There are two things you need to do at this point:

- **Collect data**. Any which way you can collect data, do it. For things like diabetes there are actually free datasets you can use that are event built-in two libraries. Once you've either used datasets out there or data a ton of measurements, you have data. This data is also referred to as _training data_.
- **Prepare data**. First you need to fuse together your data if it comes from many different sources. You might need to improve the data a little at this point, like cleaning and editing it. Finally you might also need to randomize it, this is to ensure that there is no actual correlation depending on how you later feed the data into the model for training.

NOTE: After all this data collection and data preparation, can I address my intended question. You need a resounding yes at this point, or there's no point in continuing.  

### 🎓 Feature Variable

A [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) is a measurable property of your data. In many datasets it is expressed as a column heading like 'date' 'size' or 'color'.

### Visualizing your data

You most likely need to visualize your data at this point. There might be interesting correlations that you an make use of. One thing to be aware of is if your collected dataset represents what you are likely to find in the wild. Example, let's say you collected a lot of images on Dogs and Cats and you want a model to recognize animals in general, you need to be aware of this bias towards Dogs and Cats. Cause the consequences might be that your model labels everything as a Dog or a Cat when it's in fact a Squirrel.

### Split your dataset

You need to split your dataset at this point into some major parts.

- **Training**, this part of the dataset goes into your model to train it. The size of this chunk constitutes the majority of the original dataset.
- **Evaluation**. A validation set is a smaller independent group of examples that you use to tune the model's hyperparameters, or architecture, to improve the model.
- **Test dataset**. A test dataset is another independent group of data, often gathered from the original data, that you use to confirm the performance of the built model.

## The model

A model is another word for algorithm or actually many smaller algorithms working together. Because we are doing Machine Learning, lets refer to it henceforth as the _model_.

🎓 **Feature Selection and Feature Extraction** How do you know which variable to choose when building a model? You'll probably go through a process of feature selection or feature extraction to choose the right variables for the most performant model. They're not the same thing, however: "Feature extraction creates new features from functions of the original features, whereas feature selection returns a subset of the features." [source](https://wikipedia.org/wiki/Feature_selection)

### Decide on a model

What we need to know at this point is that we need to select a model type. There are lot of existing models out there specialized on different things like images, text and so on. You are likely to go through a process whereby data scientists evaluate the performance of a model or any other relevant metric of a model by feeding it unseen data, selecting the most appropriate model for the task at hand.

## Training

At this point you are ready to go. You have your training data, that subset of your original dataset and you need to feed it into the model.

## Evaluate the model

At this point, you want to check if your model is any good, is it able to answer the question you set out for it? The way to test is by using your evaluation data and see how it performs. It's important it's data the model hasn't seen before so it simulates how it would perform in the real world.

### 🎓 Model Fitting

In the context of machine learning, Model fitting refers to the accuracy of the model's underlying function as it attempts to analyze data with which it is not familiar. 

**Underfitting** and **overfitting** are common problems that degrade the quality of the model as the model fits either not well enough or too well. This causes the model to make predictions either too closely aligned or too loosely aligned with its training data. An overfit model predicts training data too well because it has learned the data's details and noise too well. An underfit model is not accurate as it can neither accurately analyze its training data nor data it has not yet 'seen'.

The lessons in this section cover types of Regression in the context of machine learning. Regression models can help determine the _relationship_ between variables. This type of model can predict values such as length, temperature, or age, thus uncovering relationships between variables as it analyzes data points.

In this series of lessons, you'll discover the difference between Linear vs. Logistic Regression, and when you should use one or the other.

## Parameter tuning

Ok, you've made some initial assumptions before starting out. Now it's time to look at something called hyperparameters. What we are looking to do is to control the learning process, see if we can make it better. Hyperparameters affect the speed and quality of this process and don't affect the performance of the model.

## Prediction

You've made it to your goal hopefully. The whole point of this process was to combine an algorithm, i.e model and training data so you can make a prediction of data you haven't seen yet. Will take stock increase or decrease, is it sunny tomorrow and so on?


 