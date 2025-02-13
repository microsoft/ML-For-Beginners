# Construisez une application Web pour utiliser un modèle ML

Dans cette leçon, vous allez entraîner un modèle ML sur un ensemble de données qui sort de l'ordinaire : _les observations d'OVNIs au cours du siècle dernier_, provenant de la base de données de NUFORC.

Vous apprendrez :

- Comment "pickler" un modèle entraîné
- Comment utiliser ce modèle dans une application Flask

Nous continuerons à utiliser des notebooks pour nettoyer les données et entraîner notre modèle, mais vous pouvez pousser le processus un peu plus loin en explorant l'utilisation d'un modèle "dans la nature", pour ainsi dire : dans une application web.

Pour ce faire, vous devez construire une application web en utilisant Flask.

## [Quiz pré-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construction d'une application

Il existe plusieurs façons de construire des applications web pour consommer des modèles d'apprentissage automatique. Votre architecture web peut influencer la façon dont votre modèle est entraîné. Imaginez que vous travaillez dans une entreprise où le groupe de science des données a entraîné un modèle qu'il souhaite que vous utilisiez dans une application.

### Considérations

Il y a de nombreuses questions à poser :

- **Est-ce une application web ou une application mobile ?** Si vous construisez une application mobile ou si vous devez utiliser le modèle dans un contexte IoT, vous pourriez utiliser [TensorFlow Lite](https://www.tensorflow.org/lite/) et utiliser le modèle dans une application Android ou iOS.
- **Où le modèle sera-t-il hébergé ?** Dans le cloud ou localement ?
- **Support hors ligne.** L'application doit-elle fonctionner hors ligne ?
- **Quelle technologie a été utilisée pour entraîner le modèle ?** La technologie choisie peut influencer les outils que vous devez utiliser.
    - **Utilisation de TensorFlow.** Si vous entraînez un modèle en utilisant TensorFlow, par exemple, cet écosystème offre la possibilité de convertir un modèle TensorFlow pour une utilisation dans une application web en utilisant [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Utilisation de PyTorch.** Si vous construisez un modèle en utilisant une bibliothèque telle que [PyTorch](https://pytorch.org/), vous avez la possibilité de l'exporter au format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) pour une utilisation dans des applications web JavaScript qui peuvent utiliser [Onnx Runtime](https://www.onnxruntime.ai/). Cette option sera explorée dans une leçon future pour un modèle entraîné avec Scikit-learn.
    - **Utilisation de Lobe.ai ou Azure Custom Vision.** Si vous utilisez un système ML SaaS (Software as a Service) tel que [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) pour entraîner un modèle, ce type de logiciel propose des moyens d'exporter le modèle pour de nombreuses plateformes, y compris la création d'une API sur mesure à interroger dans le cloud par votre application en ligne.

Vous avez également l'opportunité de construire une application web Flask entière qui serait capable d'entraîner le modèle lui-même dans un navigateur web. Cela peut également être fait en utilisant TensorFlow.js dans un contexte JavaScript.

Pour nos besoins, puisque nous avons travaillé avec des notebooks basés sur Python, explorons les étapes que vous devez suivre pour exporter un modèle entraîné depuis un tel notebook vers un format lisible par une application web construite en Python.

## Outil

Pour cette tâche, vous avez besoin de deux outils : Flask et Pickle, tous deux fonctionnant sur Python.

✅ Qu'est-ce que [Flask](https://palletsprojects.com/p/flask/) ? Défini comme un 'micro-framework' par ses créateurs, Flask fournit les fonctionnalités de base des frameworks web utilisant Python et un moteur de templating pour construire des pages web. Jetez un œil à [ce module d'apprentissage](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) pour vous entraîner à construire avec Flask.

✅ Qu'est-ce que [Pickle](https://docs.python.org/3/library/pickle.html) ? Pickle 🥒 est un module Python qui sérialise et désérialise une structure d'objet Python. Lorsque vous "picklez" un modèle, vous sérialisez ou aplatissez sa structure pour une utilisation sur le web. Faites attention : pickle n'est pas intrinsèquement sécurisé, donc soyez prudent si vous êtes invité à "dé-pickler" un fichier. Un fichier picklé a le suffixe `.pkl`.

## Exercice - nettoyez vos données

Dans cette leçon, vous utiliserez des données provenant de 80 000 observations d'OVNIs, recueillies par [NUFORC](https://nuforc.org) (Le Centre National de Rapport d'OVNIs). Ces données contiennent des descriptions intéressantes d'observations d'OVNIs, par exemple :

- **Longue description d'exemple.** "Un homme émerge d'un faisceau de lumière qui brille sur un champ herbeux la nuit et il court vers le parking de Texas Instruments".
- **Courte description d'exemple.** "les lumières nous ont poursuivis".

Le tableau [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) comprend des colonnes sur le `city`, `state` et `country` où l'observation a eu lieu, le `shape` de l'objet et ses `latitude` et `longitude`.

Dans le [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) vierge inclus dans cette leçon :

1. importez `pandas`, `matplotlib`, et `numpy` comme vous l'avez fait dans les leçons précédentes et importez le tableau ufos. Vous pouvez jeter un œil à un échantillon de données :

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convertissez les données ufos en un petit dataframe avec des titres nouveaux. Vérifiez les valeurs uniques dans le champ `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Maintenant, vous pouvez réduire la quantité de données avec lesquelles nous devons traiter en supprimant les valeurs nulles et en n'important que les observations entre 1 et 60 secondes :

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importez la bibliothèque `LabelEncoder` de Scikit-learn pour convertir les valeurs textuelles des pays en nombres :

    ✅ LabelEncoder encode les données par ordre alphabétique

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Vos données devraient ressembler à ceci :

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercice - construisez votre modèle

Maintenant, vous pouvez vous préparer à entraîner un modèle en divisant les données en groupe d'entraînement et de test.

1. Sélectionnez les trois caractéristiques sur lesquelles vous souhaitez vous entraîner en tant que vecteur X, et le vecteur y sera `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` et obtenez un identifiant de pays à retourner.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Entraînez votre modèle en utilisant la régression logistique :

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

La précision n'est pas mauvaise **(environ 95%)**, sans surprise, car `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, mais c'est un bon exercice d'essayer d'entraîner à partir de données brutes que vous avez nettoyées, exportées, puis d'utiliser ce modèle dans une application web.

## Exercice - 'picklez' votre modèle

Maintenant, il est temps de _pickler_ votre modèle ! Vous pouvez le faire en quelques lignes de code. Une fois qu'il est _picklé_, chargez votre modèle picklé et testez-le contre un tableau de données d'échantillon contenant des valeurs pour les secondes, la latitude et la longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Le modèle retourne **'3'**, qui est le code pays pour le Royaume-Uni. Étonnant ! 👽

## Exercice - construisez une application Flask

Maintenant, vous pouvez construire une application Flask pour appeler votre modèle et retourner des résultats similaires, mais d'une manière plus visuellement agréable.

1. Commencez par créer un dossier appelé **web-app** à côté du fichier _notebook.ipynb_ où se trouve votre fichier _ufo-model.pkl_.

1. Dans ce dossier, créez trois autres dossiers : **static**, avec un dossier **css** à l'intérieur, et **templates**. Vous devriez maintenant avoir les fichiers et répertoires suivants :

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consultez le dossier de solution pour voir l'application terminée

1. Le premier fichier à créer dans le dossier _web-app_ est le fichier **requirements.txt**. Comme _package.json_ dans une application JavaScript, ce fichier liste les dépendances requises par l'application. Dans **requirements.txt**, ajoutez les lignes :

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Maintenant, exécutez ce fichier en naviguant vers _web-app_ :

    ```bash
    cd web-app
    ```

1. Dans votre terminal, tapez `pip install`, pour installer les bibliothèques listées dans _requirements.txt_ :

    ```bash
    pip install -r requirements.txt
    ```

1. Maintenant, vous êtes prêt à créer trois autres fichiers pour terminer l'application :

    1. Créez **app.py** à la racine.
    2. Créez **index.html** dans le répertoire _templates_.
    3. Créez **styles.css** dans le répertoire _static/css_.

1. Développez le fichier _styles.css_ avec quelques styles :

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

1. Ensuite, développez le fichier _index.html_ :

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

    Jetez un œil au templating dans ce fichier. Remarquez la syntaxe 'mustache' autour des variables qui seront fournies par l'application, comme le texte de prédiction : `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` ajoutez :

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

    > 💡 Astuce : lorsque vous ajoutez [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Quels sont les avantages et les inconvénients de cette méthode ?

## [Quiz post-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Révision & Auto-apprentissage

Il existe de nombreuses façons de construire une application web pour consommer des modèles ML. Faites une liste des façons dont vous pourriez utiliser JavaScript ou Python pour construire une application web afin de tirer parti de l'apprentissage automatique. Considérez l'architecture : le modèle doit-il rester dans l'application ou vivre dans le cloud ? Si c'est le cas, comment y accéderiez-vous ? Dessinez un modèle architectural pour une solution web ML appliquée.

## Devoir

[Essayez un modèle différent](assignment.md)

I'm sorry, but I can't provide a translation into "mo" as it is not a recognized language or dialect. If you meant a specific language, please specify, and I'll be happy to assist you!