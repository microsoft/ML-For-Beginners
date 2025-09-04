<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2680c691fbdb6367f350761a275e2508",
  "translation_date": "2025-09-03T23:43:46+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "fr"
}
-->
# Construire une application web pour utiliser un modèle de machine learning

Dans cette leçon, vous allez entraîner un modèle de machine learning sur un ensemble de données hors du commun : _les observations d'OVNIs au cours du siècle dernier_, issues de la base de données de NUFORC.

Vous apprendrez :

- Comment "pickler" un modèle entraîné
- Comment utiliser ce modèle dans une application Flask

Nous continuerons à utiliser des notebooks pour nettoyer les données et entraîner notre modèle, mais vous pouvez aller plus loin en explorant l'utilisation d'un modèle "dans la nature", pour ainsi dire : dans une application web.

Pour ce faire, vous devez construire une application web en utilisant Flask.

## [Quiz avant la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construire une application

Il existe plusieurs façons de construire des applications web pour consommer des modèles de machine learning. Votre architecture web peut influencer la manière dont votre modèle est entraîné. Imaginez que vous travaillez dans une entreprise où l'équipe de data science a entraîné un modèle qu'elle souhaite que vous utilisiez dans une application.

### Points à considérer

De nombreuses questions doivent être posées :

- **S'agit-il d'une application web ou mobile ?** Si vous construisez une application mobile ou devez utiliser le modèle dans un contexte IoT, vous pourriez utiliser [TensorFlow Lite](https://www.tensorflow.org/lite/) et intégrer le modèle dans une application Android ou iOS.
- **Où résidera le modèle ?** Dans le cloud ou localement ?
- **Support hors ligne.** L'application doit-elle fonctionner hors ligne ?
- **Quelle technologie a été utilisée pour entraîner le modèle ?** La technologie choisie peut influencer les outils nécessaires.
    - **Utilisation de TensorFlow.** Si vous entraînez un modèle avec TensorFlow, par exemple, cet écosystème permet de convertir un modèle TensorFlow pour une utilisation dans une application web via [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Utilisation de PyTorch.** Si vous construisez un modèle avec une bibliothèque comme [PyTorch](https://pytorch.org/), vous avez la possibilité de l'exporter au format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) pour une utilisation dans des applications web JavaScript via [Onnx Runtime](https://www.onnxruntime.ai/). Cette option sera explorée dans une future leçon pour un modèle entraîné avec Scikit-learn.
    - **Utilisation de Lobe.ai ou Azure Custom Vision.** Si vous utilisez un système SaaS de machine learning comme [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott), ce type de logiciel offre des moyens d'exporter le modèle pour de nombreuses plateformes, y compris la création d'une API sur mesure interrogeable dans le cloud par votre application en ligne.

Vous avez également la possibilité de construire une application web Flask complète capable d'entraîner le modèle directement dans un navigateur web. Cela peut également être fait avec TensorFlow.js dans un contexte JavaScript.

Pour nos besoins, puisque nous avons travaillé avec des notebooks basés sur Python, explorons les étapes nécessaires pour exporter un modèle entraîné depuis un notebook vers un format lisible par une application web construite en Python.

## Outil

Pour cette tâche, vous avez besoin de deux outils : Flask et Pickle, tous deux fonctionnant avec Python.

✅ Qu'est-ce que [Flask](https://palletsprojects.com/p/flask/) ? Défini comme un "micro-framework" par ses créateurs, Flask fournit les fonctionnalités de base des frameworks web en utilisant Python et un moteur de templates pour construire des pages web. Consultez [ce module d'apprentissage](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) pour vous entraîner à construire avec Flask.

✅ Qu'est-ce que [Pickle](https://docs.python.org/3/library/pickle.html) ? Pickle 🥒 est un module Python qui sérialise et désérialise une structure d'objet Python. Lorsque vous "picklez" un modèle, vous sérialisez ou aplatissez sa structure pour une utilisation sur le web. Attention : Pickle n'est pas intrinsèquement sécurisé, soyez donc prudent si vous êtes invité à "dé-pickler" un fichier. Un fichier picklé a l'extension `.pkl`.

## Exercice - Nettoyer vos données

Dans cette leçon, vous utiliserez des données provenant de 80 000 observations d'OVNIs, collectées par le [NUFORC](https://nuforc.org) (National UFO Reporting Center). Ces données contiennent des descriptions intéressantes d'observations d'OVNIs, par exemple :

- **Description longue.** "Un homme émerge d'un faisceau de lumière qui éclaire un champ herbeux la nuit et court vers le parking de Texas Instruments."
- **Description courte.** "Les lumières nous ont poursuivis."

Le fichier [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) contient des colonnes sur la `ville`, l'`état` et le `pays` où l'observation a eu lieu, la `forme` de l'objet, ainsi que sa `latitude` et sa `longitude`.

Dans le [notebook](notebook.ipynb) vierge inclus dans cette leçon :

1. Importez `pandas`, `matplotlib` et `numpy` comme dans les leçons précédentes, et importez le fichier ufos. Vous pouvez examiner un échantillon de l'ensemble de données :

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convertissez les données ufos en un petit dataframe avec des titres actualisés. Vérifiez les valeurs uniques dans le champ `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Réduisez ensuite la quantité de données à traiter en supprimant les valeurs nulles et en ne conservant que les observations entre 1 et 60 secondes :

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

## Exercice - Construire votre modèle

Vous pouvez maintenant vous préparer à entraîner un modèle en divisant les données en groupes d'entraînement et de test.

1. Sélectionnez les trois caractéristiques sur lesquelles vous souhaitez entraîner votre modèle comme vecteur X, et le vecteur y sera le `Country`. Vous voulez pouvoir entrer `Seconds`, `Latitude` et `Longitude` et obtenir un identifiant de pays en retour.

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

La précision n'est pas mauvaise **(environ 95%)**, ce qui n'est pas surprenant, car `Country` et `Latitude/Longitude` sont corrélés.

Le modèle que vous avez créé n'est pas très révolutionnaire, car vous devriez pouvoir déduire un `Country` à partir de sa `Latitude` et de sa `Longitude`, mais c'est un bon exercice pour s'entraîner à partir de données brutes que vous avez nettoyées, exportées, puis utilisées dans une application web.

## Exercice - "Pickler" votre modèle

Il est maintenant temps de _pickler_ votre modèle ! Vous pouvez le faire en quelques lignes de code. Une fois qu'il est _picklé_, chargez votre modèle picklé et testez-le avec un tableau de données exemple contenant des valeurs pour les secondes, la latitude et la longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Le modèle renvoie **'3'**, qui est le code pays pour le Royaume-Uni. Incroyable ! 👽

## Exercice - Construire une application Flask

Vous pouvez maintenant construire une application Flask pour appeler votre modèle et renvoyer des résultats similaires, mais de manière plus visuellement attrayante.

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

    ✅ Consultez le dossier solution pour voir l'application terminée

1. Le premier fichier à créer dans le dossier _web-app_ est le fichier **requirements.txt**. Comme _package.json_ dans une application JavaScript, ce fichier liste les dépendances nécessaires à l'application. Dans **requirements.txt**, ajoutez les lignes :

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ensuite, exécutez ce fichier en naviguant vers _web-app_ :

    ```bash
    cd web-app
    ```

1. Dans votre terminal, tapez `pip install` pour installer les bibliothèques listées dans _requirements.txt_ :

    ```bash
    pip install -r requirements.txt
    ```

1. Vous êtes maintenant prêt à créer trois autres fichiers pour terminer l'application :

    1. Créez **app.py** à la racine.
    2. Créez **index.html** dans le répertoire _templates_.
    3. Créez **styles.css** dans le répertoire _static/css_.

1. Construisez le fichier _styles.css_ avec quelques styles :

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

1. Ensuite, construisez le fichier _index.html_ :

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

    Examinez le templating dans ce fichier. Remarquez la syntaxe "mustache" autour des variables fournies par l'application, comme le texte de prédiction : `{{}}`. Il y a également un formulaire qui envoie une prédiction à la route `/predict`.

    Enfin, vous êtes prêt à construire le fichier Python qui pilote la consommation du modèle et l'affichage des prédictions :

1. Dans `app.py`, ajoutez :

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

    > 💡 Astuce : lorsque vous ajoutez [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) lors de l'exécution de l'application web avec Flask, toute modification apportée à votre application sera immédiatement reflétée sans avoir besoin de redémarrer le serveur. Attention ! Ne pas activer ce mode dans une application en production.

Si vous exécutez `python app.py` ou `python3 app.py`, votre serveur web démarre localement, et vous pouvez remplir un court formulaire pour obtenir une réponse à votre question brûlante sur les lieux où les OVNIs ont été observés !

Avant cela, examinez les parties de `app.py` :

1. Tout d'abord, les dépendances sont chargées et l'application démarre.
1. Ensuite, le modèle est importé.
1. Enfin, index.html est rendu sur la route principale.

Sur la route `/predict`, plusieurs choses se produisent lorsque le formulaire est soumis :

1. Les variables du formulaire sont collectées et converties en un tableau numpy. Elles sont ensuite envoyées au modèle, et une prédiction est renvoyée.
2. Les pays que nous voulons afficher sont retransformés en texte lisible à partir de leur code pays prédit, et cette valeur est renvoyée à index.html pour être rendue dans le template.

Utiliser un modèle de cette manière, avec Flask et un modèle picklé, est relativement simple. Le plus difficile est de comprendre la forme des données qui doivent être envoyées au modèle pour obtenir une prédiction. Tout dépend de la manière dont le modèle a été entraîné. Celui-ci nécessite trois points de données en entrée pour produire une prédiction.

Dans un cadre professionnel, vous pouvez voir à quel point une bonne communication est nécessaire entre les personnes qui entraînent le modèle et celles qui le consomment dans une application web ou mobile. Dans notre cas, c'est une seule personne : vous !

---

## 🚀 Défi

Au lieu de travailler dans un notebook et d'importer le modèle dans l'application Flask, vous pourriez entraîner le modèle directement dans l'application Flask ! Essayez de convertir votre code Python dans le notebook pour entraîner le modèle directement dans l'application, sur une route appelée `train`. Quels sont les avantages et les inconvénients de cette méthode ?

## [Quiz après la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Révision et auto-apprentissage

Il existe de nombreuses façons de construire une application web pour consommer des modèles de machine learning. Faites une liste des façons dont vous pourriez utiliser JavaScript ou Python pour construire une application web exploitant le machine learning. Réfléchissez à l'architecture : le modèle doit-il rester dans l'application ou vivre dans le cloud ? Si c'est le cas, comment y accéderiez-vous ? Dessinez un modèle architectural pour une solution web appliquée au machine learning.

## Devoir

[Essayez un modèle différent](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.