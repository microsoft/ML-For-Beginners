# Costruisci un'app Web per utilizzare un modello ML

In questa lezione, addestrerai un modello ML su un set di dati fuori dal comune: _avvistamenti UFO nell'ultimo secolo_, provenienti dal database di NUFORC.

Imparerai:

- Come 'pickle' un modello addestrato
- Come utilizzare quel modello in un'app Flask

Continueremo a usare i notebook per pulire i dati e addestrare il nostro modello, ma puoi fare un ulteriore passo avanti esplorando l'uso di un modello "nel mondo reale", per così dire: in un'app web.

Per fare questo, è necessario costruire un'app web utilizzando Flask.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Costruzione di un'app

Ci sono diversi modi per costruire app web che consumano modelli di machine learning. La tua architettura web potrebbe influenzare il modo in cui il tuo modello viene addestrato. Immagina di lavorare in un'azienda in cui il gruppo di data science ha addestrato un modello che vogliono tu usi in un'app.

### Considerazioni

Ci sono molte domande che devi farti:

- **È un'app web o un'app mobile?** Se stai costruendo un'app mobile o hai bisogno di utilizzare il modello in un contesto IoT, potresti usare [TensorFlow Lite](https://www.tensorflow.org/lite/) e utilizzare il modello in un'app Android o iOS.
- **Dove risiederà il modello?** Nel cloud o localmente?
- **Supporto offline.** L'app deve funzionare offline?
- **Quale tecnologia è stata utilizzata per addestrare il modello?** La tecnologia scelta potrebbe influenzare gli strumenti che devi usare.
    - **Usando TensorFlow.** Se stai addestrando un modello usando TensorFlow, ad esempio, quell'ecosistema offre la possibilità di convertire un modello TensorFlow per l'uso in un'app web utilizzando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Se stai costruendo un modello usando una libreria come [PyTorch](https://pytorch.org/), hai l'opzione di esportarlo in formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) per l'uso in app web JavaScript che possono utilizzare [Onnx Runtime](https://www.onnxruntime.ai/). Questa opzione verrà esplorata in una lezione futura per un modello addestrato con Scikit-learn.
    - **Usando Lobe.ai o Azure Custom Vision.** Se stai usando un sistema ML SaaS (Software as a Service) come [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) per addestrare un modello, questo tipo di software fornisce modi per esportare il modello per molte piattaforme, incluso la costruzione di un'API su misura da interrogare nel cloud dalla tua applicazione online.

Hai anche l'opportunità di costruire un'intera app web Flask che sarebbe in grado di addestrare il modello stesso in un browser web. Questo può essere fatto anche utilizzando TensorFlow.js in un contesto JavaScript.

Per i nostri scopi, poiché abbiamo lavorato con notebook basati su Python, esploriamo i passaggi necessari per esportare un modello addestrato da un tale notebook in un formato leggibile da un'app web costruita in Python.

## Strumenti

Per questo compito, hai bisogno di due strumenti: Flask e Pickle, entrambi funzionanti su Python.

✅ Cos'è [Flask](https://palletsprojects.com/p/flask/)? Definito come un 'micro-framework' dai suoi creatori, Flask fornisce le caratteristiche di base dei framework web utilizzando Python e un motore di template per costruire pagine web. Dai un'occhiata a [questo modulo di apprendimento](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) per esercitarti a costruire con Flask.

✅ Cos'è [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 è un modulo Python che serializza e deserializza una struttura di oggetti Python. Quando 'pickle' un modello, ne serializzi o appiattisci la struttura per l'uso sul web. Attenzione: pickle non è intrinsecamente sicuro, quindi fai attenzione se ti viene chiesto di 'un-pickle' un file. Un file pickled ha il suffisso `.pkl`.

## Esercizio - pulisci i tuoi dati

In questa lezione userai dati provenienti da 80.000 avvistamenti UFO, raccolti da [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Questi dati contengono alcune descrizioni interessanti degli avvistamenti UFO, ad esempio:

- **Descrizione lunga.** "Un uomo emerge da un raggio di luce che brilla su un campo erboso di notte e corre verso il parcheggio della Texas Instruments".
- **Descrizione breve.** "le luci ci hanno inseguito".

Il foglio di calcolo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) include colonne riguardanti `city`, `state` e `country` dove è avvenuto l'avvistamento, l'`shape` dell'oggetto e il suo `latitude` e `longitude`.

Nel [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) incluso in questa lezione:

1. importa `pandas`, `matplotlib`, e `numpy` come hai fatto nelle lezioni precedenti e importa il foglio di calcolo ufos. Puoi dare un'occhiata a un set di dati di esempio:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converti i dati ufos in un piccolo dataframe con nuovi titoli. Controlla i valori unici nel campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ora, puoi ridurre la quantità di dati con cui dobbiamo lavorare eliminando eventuali valori nulli e importando solo gli avvistamenti tra 1-60 secondi:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa la libreria `LabelEncoder` di Scikit-learn per convertire i valori di testo per i paesi in un numero:

    ✅ LabelEncoder codifica i dati in ordine alfabetico

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    I tuoi dati dovrebbero apparire così:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Esercizio - costruisci il tuo modello

Ora puoi prepararti ad addestrare un modello dividendo i dati nel gruppo di addestramento e nel gruppo di test.

1. Seleziona le tre caratteristiche su cui vuoi addestrarti come il tuo vettore X, e il vettore y sarà `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` e ottieni un id paese da restituire.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Addestra il tuo modello utilizzando la regressione logistica:

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

La precisione non è male **(circa il 95%)**, non sorprende, poiché `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, ma è un buon esercizio provare ad addestrare partendo da dati grezzi che hai pulito, esportato e poi usare questo modello in un'app web.

## Esercizio - 'pickle' il tuo modello

Ora, è il momento di _pickle_ il tuo modello! Puoi farlo in poche righe di codice. Una volta che è _pickled_, carica il tuo modello pickled e testalo contro un array di dati di esempio contenente valori per secondi, latitudine e longitudine,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Il modello restituisce **'3'**, che è il codice paese per il Regno Unito. Incredibile! 👽

## Esercizio - costruisci un'app Flask

Ora puoi costruire un'app Flask per chiamare il tuo modello e restituire risultati simili, ma in un modo più visivamente piacevole.

1. Inizia creando una cartella chiamata **web-app** accanto al file _notebook.ipynb_ dove risiede il tuo file _ufo-model.pkl_.

1. In quella cartella crea altre tre cartelle: **static**, con una cartella **css** all'interno, e **templates**. Ora dovresti avere i seguenti file e directory:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta la cartella della soluzione per una visuale dell'app finita

1. Il primo file da creare nella cartella _web-app_ è il file **requirements.txt**. Come _package.json_ in un'app JavaScript, questo file elenca le dipendenze richieste dall'app. In **requirements.txt** aggiungi le righe:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ora, esegui questo file navigando fino a _web-app_:

    ```bash
    cd web-app
    ```

1. Nel tuo terminale digita `pip install`, per installare le librerie elencate in _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ora, sei pronto per creare altri tre file per completare l'app:

    1. Crea **app.py** nella root.
    2. Crea **index.html** nella directory _templates_.
    3. Crea **styles.css** nella directory _static/css_.

1. Costruisci il file _styles.css_ con alcuni stili:

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

1. Successivamente, costruisci il file _index.html_:

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

    Dai un'occhiata al templating in questo file. Nota la sintassi 'mustache' attorno alle variabili che verranno fornite dall'app, come il testo della previsione: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` aggiungi:

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

    > 💡 Suggerimento: quando aggiungi [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

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

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Quali sono i pro e i contro di perseguire questo metodo?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Revisione & Studio autonomo

Ci sono molti modi per costruire un'app web che consumi modelli ML. Fai un elenco dei modi in cui potresti usare JavaScript o Python per costruire un'app web che sfrutti il machine learning. Considera l'architettura: il modello dovrebbe rimanere nell'app o vivere nel cloud? Se la seconda opzione, come lo accederesti? Disegna un modello architettonico per una soluzione ML applicata.

## Compito

[Prova un modello diverso](assignment.md)

**Avvertenza**:
Questo documento è stato tradotto utilizzando servizi di traduzione automatica basati su intelligenza artificiale. Sebbene ci impegniamo per l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o inesattezze. Il documento originale nella sua lingua madre deve essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda la traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.