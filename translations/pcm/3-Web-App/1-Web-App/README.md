<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-11-18T19:03:49+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "pcm"
}
-->
# Build Web App wey go use ML Model

For dis lesson, you go train ML model for one data set wey dey special: _UFO sightings for di past century_, wey dem collect from NUFORC database.

You go learn:

- How to 'pickle' trained model
- How to use di model for Flask app

We go still dey use notebooks to clean data and train di model, but you fit carry di process go one step further by trying to use di model 'for di wild', like for web app.

To do dis one, you go need to build web app wey dey use Flask.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## How to build app

Plenty ways dey to build web apps wey go use machine learning models. Di way wey your web architecture be fit affect how you go train di model. Imagine say you dey work for one business wey di data science team don train model wey dem wan make you use for app.

### Things to think about

Plenty questions dey wey you go need ask:

- **Na web app or mobile app?** If na mobile app you dey build or you need use di model for IoT context, you fit use [TensorFlow Lite](https://www.tensorflow.org/lite/) to use di model for Android or iOS app.
- **Where di model go dey?** Na for cloud or for local machine?
- **Offline support.** Di app go need work offline?
- **Which technology dem use train di model?** Di technology wey dem choose fit affect di tools wey you go need use.
    - **If na TensorFlow.** If na TensorFlow you dey use train di model, di ecosystem dey allow you convert TensorFlow model to use for web app with [TensorFlow.js](https://www.tensorflow.org/js/).
    - **If na PyTorch.** If na library like [PyTorch](https://pytorch.org/) you dey use build di model, you fit export am for [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format to use for JavaScript web apps wey fit use [Onnx Runtime](https://www.onnxruntime.ai/). We go talk about dis option for future lesson for Scikit-learn-trained model.
    - **If na Lobe.ai or Azure Custom Vision.** If you dey use ML SaaS (Software as a Service) system like [Lobe.ai](https://lobe.ai/) or [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) to train model, dis software dey provide ways to export di model for plenty platforms, including building custom API wey go dey query for cloud by your online app.

You fit also build full Flask web app wey fit train di model by itself for web browser. Dis one fit also happen with TensorFlow.js for JavaScript context.

For our case, since we don dey use Python-based notebooks, make we look di steps wey you go need take to export trained model from di notebook to format wey Python-built web app fit read.

## Tool

For dis task, you go need two tools: Flask and Pickle, both dey run for Python.

✅ Wetin be [Flask](https://palletsprojects.com/p/flask/)? Di creators call am 'micro-framework', Flask dey provide di basic features of web frameworks wey dey use Python and templating engine to build web pages. Check [dis Learn module](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) to practice how to build with Flask.

✅ Wetin be [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 na Python module wey dey serialize and de-serialize Python object structure. When you 'pickle' model, you dey serialize or flatten di structure to use for web. Be careful: pickle no dey secure by itself, so make sure say you dey careful if dem ask you to 'un-pickle' file. Pickled file dey end with `.pkl`.

## Exercise - clean your data

For dis lesson, you go use data from 80,000 UFO sightings wey [NUFORC](https://nuforc.org) (Di National UFO Reporting Center) gather. Dis data get some interesting descriptions of UFO sightings, like:

- **Long example description.** "One man come out from beam of light wey dey shine for grassy field for night, e run go Texas Instruments parking lot".
- **Short example description.** "di lights dey chase us".

Di [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) spreadsheet get columns about di `city`, `state` and `country` wey di sighting happen, di object's `shape` and di `latitude` and `longitude`.

For di blank [notebook](notebook.ipynb) wey dey dis lesson:

1. import `pandas`, `matplotlib`, and `numpy` like you don do for di previous lessons and import di ufos spreadsheet. You fit check sample data set:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convert di ufos data to small dataframe with fresh titles. Check di unique values for di `Country` field.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Now, you fit reduce di amount of data wey we go deal with by dropping any null values and only importing sightings wey dey between 1-60 seconds:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Import Scikit-learn `LabelEncoder` library to convert di text values for countries to number:

    ✅ LabelEncoder dey encode data alphabetically

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Your data go look like dis:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercise - build your model

Now you fit prepare to train model by dividing di data into training and testing group.

1. Select di three features wey you wan train on as your X vector, and di y vector go be di `Country`. You wan fit input `Seconds`, `Latitude` and `Longitude` and get country id wey go return.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Train di model using logistic regression:

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

Di accuracy no bad **(around 95%)**, e no surprise, as `Country` and `Latitude/Longitude` dey correlate.

Di model wey you create no too special as you suppose fit infer `Country` from di `Latitude` and `Longitude`, but e good exercise to try train from raw data wey you don clean, export, and then use di model for web app.

## Exercise - 'pickle' your model

Now, na time to _pickle_ your model! You fit do am with few lines of code. Once e don _pickled_, load di pickled model and test am against sample data array wey get values for seconds, latitude and longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Di model go return **'3'**, wey be di country code for UK. Wild! 👽

## Exercise - build Flask app

Now you fit build Flask app wey go call di model and return similar results, but e go dey more fine for eye.

1. Start by creating folder wey you go call **web-app** next to di _notebook.ipynb_ file wey your _ufo-model.pkl_ file dey.

1. For di folder, create three more folders: **static**, with folder **css** inside am, and **templates**. You suppose get di following files and directories:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Check di solution folder to see how di finished app go look

1. Di first file wey you go create for _web-app_ folder na **requirements.txt** file. Like _package.json_ for JavaScript app, dis file dey list di dependencies wey di app need. For **requirements.txt** add di lines:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Now, run dis file by navigating to _web-app_:

    ```bash
    cd web-app
    ```

1. For your terminal type `pip install`, to install di libraries wey dey _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Now, you don ready to create three more files to finish di app:

    1. Create **app.py** for di root.
    2. Create **index.html** for _templates_ directory.
    3. Create **styles.css** for _static/css_ directory.

1. Build di _styles.css_ file with small styles:

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

1. Next, build di _index.html_ file:

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

    Check di templating for dis file. Notice di 'mustache' syntax around variables wey di app go provide, like di prediction text: `{{}}`. E get form wey dey post prediction to `/predict` route.

    Finally, you don ready to build di python file wey go drive di model consumption and di display of predictions:

1. For `app.py` add:

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

    > 💡 Tip: when you add [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while you dey run di web app with Flask, any changes wey you make to your application go show immediately without restarting di server. But be careful! No enable dis mode for production app.

If you run `python app.py` or `python3 app.py` - your web server go start locally, and you fit fill short form to get answer to your question about where UFOs don dey sight!

Before you do am, check di parts of `app.py`:

1. First, dependencies go load and di app go start.
1. Then, di model go import.
1. Then, index.html go render for di home route.

For di `/predict` route, plenty things go happen when di form post:

1. Di form variables go gather and convert to numpy array. Dem go send am to di model and prediction go return.
2. Di Countries wey we wan show go re-render as readable text from di predicted country code, and dat value go send back to index.html to render for di template.

To use model like dis, with Flask and pickled model, e dey straightforward. Di hardest part na to understand di shape of di data wey you go send to di model to get prediction. Dat one dey depend on how di model take train. Dis one need three data points to input to get prediction.

For professional setting, you fit see how good communication dey important between di people wey train di model and di people wey dey use am for web or mobile app. For our case, na only one person, you!

---

## 🚀 Challenge

Instead of working for notebook and importing di model to Flask app, you fit train di model inside di Flask app! Try convert your Python code for di notebook, maybe after you don clean di data, to train di model from inside di app for route wey you go call `train`. Wetin be di pros and cons of dis method?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Plenty ways dey to build web app wey go use ML models. Make list of di ways wey you fit use JavaScript or Python to build web app wey go use machine learning. Think about architecture: di model go dey inside di app or e go dey for cloud? If na di second one, how you go access am? Draw architectural model for applied ML web solution.

## Assignment

[Try different model](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis dokyument don use AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator) do di translation. Even as we dey try make am accurate, abeg make you sabi say automated translations fit get mistake or no dey correct well. Di original dokyument for im native language na di main source wey you go trust. For important information, e good make professional human translation dey use. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->