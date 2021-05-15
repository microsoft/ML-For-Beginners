# Build a Web App to use a ML Model

In this lesson, you will train a Linear Regression model and a Classification model on a dataset that's out of this world: UFO Sightings over the past century, sourced from [NUFORC's database](https://www.nuforc.org). We will continue our use of notebooks to clean data and train our model, but you can take the process one step further by exploring using a model 'in the wild', so to speak: in a web app. To do this, you need to build a web app using Flask.
## [Pre-lecture quiz](link-to-quiz-app)

There are several ways to build web apps to consume machine learning models. Your web architecture may influence the way your model is trained. Imagine that you are working in a business where the data science group has trained a model that they want you to use in an app. There are many questions you need to ask: Is it a web app, or a mobile app? Where will the model reside, in the cloud or locally? Does the app have to work offline? And what technology was used to train the model, because that may influence the tooling you need to use? 

If you are training a model using TensorFlow, for example, that ecosystem provides the ability to convert a TensorFlow model for use in a web app by using [TensorFlow.js](https://www.tensorflow.org/js/). If you are building a mobile app or need to use the model in an IoT context, you could use [TensorFlow Lite](https://www.tensorflow.org/lite/) and use the model in an Android or iOS app. 

If you are building a model using [PyTorch](https://pytorch.org/), you have the option to export it in [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format for use in JavaScript web apps that can use [onnx.js](https://github.com/Microsoft/onnxjs). This option will be explored in a future lesson.

If you are using an ML SaaS (Software as a Service) system such as [Lobe.ai](https://lobe.ai/) or [Azure Custom Vision](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/) to train a model, this type of software provides ways to export the model for many platforms, including building a bespoke API to be queried in the cloud by your online application.

You also have the opportunity to build an entire Flask web app that would be able to train the model itself in a web browser. This can also be done using TensorFlow.js in a JavaScript context. For our purposes, since we have been working with notebooks, let's explore the steps you need to take to export a trained model to a format readable by a Python-built web app. 

## Tools 

For this task, you need two tools: Flask and Pickle, both of which run on Python.

✅ What's [Flask](https://palletsprojects.com/p/flask/)? Defined as a 'micro-framework' by its creators, Flask provides the basic features of web frameworks using Python and a templating engine to build web pages.  Take a look at [this Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-15963-cxa) to practice building with Flask.

✅ What's [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 is a Python module that serializes and de-serializes a Python object structure. When you 'pickle' a model, you serialize or flatten its structure for use on the web. Be careful: pickle is not intrinsically secure, so be careful if prompted to 'un-pickle' a file. A pickled file has the suffix `.pkl`.

## Clean your data

In this lesson you'll use data from 80,000 UFO sightings, gathered by [NUFORC](https://nuforc.org) (The National UFO Reporting Center). This data has some interesting descriptions of UFO sightings, for example "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot" or simply "the lights chased us". The [ufos.csv](./data/ufos.csv) spreadsheet includes columns about the city, state and country where the sighting occurred, the object's shape and its latitude and longitude. 

In the blank [notebook](notebook.ipynb) included in this lesson, import pandas, matplotlib, and numpy as you did in previous lessons and import the ufos spreadsheet. You can take a look at a sample data set:

```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('../data/ufos.csv')
ufos.head()
```
Convert the ufos data to a small dataframe with fresh titles. Check the unique values in the Country field.

```python
from sklearn.preprocessing import LabelEncoder

ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```

Now, you can reduce the amount of data we need to deal with by dropping any null values and only importing sightings between 1-60 seconds:

```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

Next, import Scikit-Learn's LabelEncoder library to convert the text values for countries to a number. 

✅ LabelEncoder encodes data alphabetically

```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```

Your data should look like this:

```
	Seconds	Country	Latitude	Longitude
2	20.0	3	    53.200000	-2.916667
3	20.0	4	    28.978333	-96.645833
14	30.0	4	    35.823889	-80.253611
23	60.0	4	    45.582778	-122.352222
24	3.0	    3	    51.783333	-0.783333
```
## Build your model

Now you can get ready to train a model by diving the data into the training and testing group. Select the three features you want to train on as your X vector, and the y vector will be the Country. You want to be able to input seconds, latitude and longitude and get a country id to return.

```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Finally, train your model using Logistic Regression:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

The accuracy isn't bad (around 95%), unsurprisingly, as country and latitude/longitude have a good correlation. The model you created isn't very revolutionary, but it's a good exercise to try to train from raw data that you cleaned, export, and then use this model in a web app.
## Pickle your model

Now, it's time to pickle your model! You can do that in just a few lines of code. Load your pickled model and test it against a sample data array containing values for seconds, latitude and longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```
The model returns '3', which is the country code for the UK. Wild! 👽

## Build a Flask app

Now you can build a Flask app to call your model and return similar results, but in a more visually pleasing way.

Start by creating a folder called web-app next to the notebook.ipynb file where your ufo-model.pkl file resides. In that folder create three more folders: `static`, with a folder `css` inside it, and `templates`. 

> Refer to the solution folder for a view of the finished app

The first file to create in `web-app` is a `requirements.txt` file. Like `package.json` in a JavaScript app, this file lists dependencies required by the app. In `requirements.txt` add the lines: 

```text
scikit-learn
pandas
numpy
flask
```
Now, run this file by navigating to `web-app` (`cd web-app`) in your terminal and typing `pip install -r requirements.txt`.

Now, you're ready to create three more files to finish the app:

1. Create `app.py` in the root
2. Create `index.html` in `templates`
3. Create `styles.css` in `static/css`

Build out the styles.css file with a few styles:

```css
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```
Next, build out the `index.html` file:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>🛸 UFO Appearance Prediction! 👽</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}"> 
</head>

<body>
 <div class="grid">

  <div class="box">

  <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

    <form action="{{ url_for('predict')}}"method="post">
    	<input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
      <input type="text" name="latitude" placeholder="Latitude" required="required" />
		  <input type="text" name="longitude" placeholder="Longitude" required="required" />
      <button type="submit" class="btn">Predict country where the UFO is seen</button>
    </form>

  
   <p>{{ prediction_text }}</p>

 </div>
</div>

</body>
</html>
```
Take a look at the templating in this file. Notice the 'mustache' syntax around variables that will be provided by the app, like the prediction text: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

In `app.py` add:

```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("../ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```
If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`.

First, dependencies are loaded and the app starts. Then, the model is imported. Then, index.html is rendered on the home route. On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned. 
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction. In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

## 🚀 Challenge: 

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. What are the pros and cons of pursuing this method?

## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

There are many ways to build a web app to consume ML models. Make a list of the ways you could use JavaScript or Python to build a web app to leverage machine learning. Consider architecture: should the model stay in the app or live in the cloud? If the latter, how would you access it? Draw out an architectural model for an applied ML web solution.

**Assignment**: [Try a different model](assignment.md)


