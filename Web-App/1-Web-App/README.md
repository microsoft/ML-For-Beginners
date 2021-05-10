# Build a Web App to use a ML Model

In this lesson, you will train a Linear Regression model and a Classification model on a dataset that's out of this world: UFO Sightings over the past century, sourced from [NUFORC's database](https://www.nuforc.org). We will continue our use of notebooks to clean data and train our model, but you can take the process one step further by exploring using a model 'in the wild', so to speak: in a web app. To do this, you need to build a web app using Flask.

There are several ways to build web apps to consume machine learning models. Your web architecture may influence the way your model is trained. Imagine that you are working in a business where the data science group has trained a model that they want you to use in an app. There are many questions you need to ask: Is it a web app, or a mobile app? Where will the model reside, in the cloud or locally? Does the app have to work offline? And what technology was used to train the model, because that may influence the tooling you need to use? 

If you are training a model using TensorFlow, for example, that ecosystem provides the ability to convert a TensorFlow model for use in a web app by using [TensorFlow.js](https://www.tensorflow.org/js/). If you are building a mobile app or need to use the model in an IoT context, you could use [TensorFlow Lite](https://www.tensorflow.org/lite/) and use the model in an Android or iOS app. 

If you are building a model using [PyTorch](https://pytorch.org/), you have the option to export it in [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format for use in JavaScript web apps that can use [onnx.js](https://github.com/Microsoft/onnxjs). This option will be explored in a future lesson.

If you are using an ML SaaS (Software as a Service) system such as [Lobe.ai](https://lobe.ai/) or [Azure Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/) to train a model, this type of software provides ways to export the model for many platforms, including building a bespoke API to be queried in the cloud by your online application.

You also have the opportunity to build an entire Flask web app that would be able to train the model itself in a web browser. This can also be done using TensorFlow.js in a JavaScript context. For our purposes, since we have been working with notebooks, let's explore the steps you need to take to export a trained model to a format readable by a Python-built web app. 

## Tools 

For this task, you need two tools: Flask and Pickle, both of which run on Python.
## [Pre-lecture quiz](link-to-quiz-app)

✅ Knowledge Check - use this moment to stretch students' knowledge with open questions

🚀 Challenge: Add a challenge for students to work on collaboratively in class to enhance the project

## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

**Assignment**: [Assignment Name](assignment.md)


