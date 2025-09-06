<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T10:55:08+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "en"
}
-->
# Build a Web App to use a ML Model

In this lesson, you will train a machine learning model using a fascinating dataset: _UFO sightings over the past century_, sourced from NUFORC's database.

You will learn:

- How to save a trained model using 'pickle'
- How to integrate that model into a Flask web application

We'll continue using notebooks to clean data and train our model, but we'll take it a step further by exploring how to use the model in a real-world scenario: a web app.

To achieve this, you'll need to build a web app using Flask.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Building an app

There are multiple ways to create web apps that utilize machine learning models. The architecture of your web app may influence how the model is trained. Imagine you're working in a company where the data science team has trained a model, and they want you to integrate it into an app.

### Considerations

Here are some important questions to consider:

- **Is it a web app or a mobile app?** If you're building a mobile app or need to use the model in an IoT context, you could use [TensorFlow Lite](https://www.tensorflow.org/lite/) to integrate the model into an Android or iOS app.
- **Where will the model be hosted?** Will it reside in the cloud or locally?
- **Offline support.** Does the app need to function offline?
- **What technology was used to train the model?** The technology used may dictate the tools required for integration.
    - **Using TensorFlow.** If the model was trained using TensorFlow, you can convert it for use in a web app with [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Using PyTorch.** If the model was built using [PyTorch](https://pytorch.org/), you can export it in [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format for use in JavaScript web apps with [Onnx Runtime](https://www.onnxruntime.ai/). This approach will be covered in a future lesson for a Scikit-learn-trained model.
    - **Using Lobe.ai or Azure Custom Vision.** If you used an ML SaaS platform like [Lobe.ai](https://lobe.ai/) or [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott), these tools provide options to export the model for various platforms, including creating a custom API for cloud-based queries.

You could also build a complete Flask web app capable of training the model directly in a web browser. This can be achieved using TensorFlow.js in a JavaScript environment.

For this lesson, since we've been working with Python-based notebooks, we'll focus on exporting a trained model from a notebook into a format that can be used in a Python-built web app.

## Tool

To complete this task, you'll need two tools: Flask and Pickle, both of which run on Python.

✅ What is [Flask](https://palletsprojects.com/p/flask/)? Flask is a lightweight web framework for Python that provides essential features for building web applications, including a templating engine for creating web pages. Check out [this Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) to practice working with Flask.

✅ What is [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 is a Python module used to serialize and deserialize Python object structures. When you 'pickle' a model, you flatten its structure for use in a web app. Be cautious: Pickle is not inherently secure, so exercise caution when prompted to 'un-pickle' a file. Pickled files typically have the `.pkl` extension.

## Exercise - clean your data

In this lesson, you'll work with data from 80,000 UFO sightings collected by [NUFORC](https://nuforc.org) (The National UFO Reporting Center). The dataset includes intriguing descriptions of UFO sightings, such as:

- **Long example description.** "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot."
- **Short example description.** "The lights chased us."

The [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) spreadsheet contains columns for the `city`, `state`, and `country` where the sighting occurred, the object's `shape`, and its `latitude` and `longitude`.

In the blank [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) provided in this lesson:

1. Import `pandas`, `matplotlib`, and `numpy` as you did in previous lessons, and load the UFO dataset. Here's a sample of the data:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convert the UFO data into a smaller dataframe with updated column names. Check the unique values in the `Country` field.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Reduce the dataset by removing rows with null values and filtering sightings that lasted between 1-60 seconds:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Use Scikit-learn's `LabelEncoder` library to convert text values in the `Country` column into numeric values:

    ✅ LabelEncoder encodes data alphabetically.

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Your cleaned data should look like this:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercise - build your model

Now, divide the data into training and testing sets to prepare for model training.

1. Select three features for your X vector, and use the `Country` column as your y vector. The goal is to input `Seconds`, `Latitude`, and `Longitude` to predict a country ID.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Train your model using logistic regression:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

The accuracy is quite good **(around 95%)**, which is expected since `Country` correlates strongly with `Latitude` and `Longitude`.

While the model isn't groundbreaking—predicting a `Country` from its `Latitude` and `Longitude` is straightforward—it serves as a valuable exercise in cleaning data, training a model, exporting it, and using it in a web app.

## Exercise - 'pickle' your model

Next, save your trained model using Pickle. After pickling the model, load it and test it with a sample data array containing values for seconds, latitude, and longitude:

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

The model predicts **'3'**, which corresponds to the UK. Fascinating! 👽

## Exercise - build a Flask app

Now, create a Flask app to call your model and display results in a user-friendly format.

1. Start by creating a folder named **web-app** next to the _notebook.ipynb_ file where your _ufo-model.pkl_ file is located.

1. Inside the **web-app** folder, create three subfolders: **static** (with a **css** folder inside) and **templates**. Your directory structure should look like this:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Refer to the solution folder for a completed app example.

1. Create a **requirements.txt** file in the **web-app** folder. This file lists the app's dependencies, similar to _package.json_ in JavaScript apps. Add the following lines to **requirements.txt**:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Navigate to the **web-app** folder and run the following command:

    ```bash
    cd web-app
    ```

1. Install the libraries listed in **requirements.txt** by typing `pip install` in your terminal:

    ```bash
    pip install -r requirements.txt
    ```

1. Create three additional files to complete the app:

    1. **app.py** in the root directory.
    2. **index.html** in the **templates** folder.
    3. **styles.css** in the **static/css** folder.

1. Add some basic styles to the **styles.css** file:

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

1. Build the **index.html** file:

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
    
            <form action="{{ url_for('predict')}}" method="post">
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

    Notice the templating syntax in this file. Variables provided by the app, such as the prediction text, are enclosed in `{{}}`. The form posts data to the `/predict` route.

1. Finally, create the Python file that handles the model and displays predictions:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
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

    > 💡 Tip: Adding [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the Flask app allows you to see changes immediately without restarting the server. However, avoid enabling this mode in production.

Run `python app.py` or `python3 app.py` to start your local web server. You can then fill out the form to discover where UFOs have been sighted!

Before testing the app, review the structure of `app.py`:

1. Dependencies are loaded, and the app is initialized.
1. The model is imported.
1. The home route renders the `index.html` file.

On the `/predict` route, the following occurs when the form is submitted:

1. Form variables are collected and converted into a numpy array. The array is sent to the model, which returns a prediction.
2. Predicted country codes are converted into readable text and sent back to `index.html` for display.

Using a model with Flask and Pickle is relatively simple. The key challenge is understanding the data format required by the model for predictions, which depends on how the model was trained. In this case, three data points are needed for predictions.

In a professional setting, clear communication between the team training the model and the team integrating it into an app is crucial. In this lesson, you're both teams!

---

## 🚀 Challenge

Instead of training the model in a notebook and importing it into the Flask app, try training the model directly within the Flask app on a route called `train`. What are the advantages and disadvantages of this approach?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

There are various ways to build a web app that utilizes machine learning models. Make a list of methods for using JavaScript or Python to create such an app. Consider the architecture: should the model reside in the app or in the cloud? If hosted in the cloud, how would you access it? Sketch an architectural diagram for an applied ML web solution.

## Assignment

[Try a different model](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.