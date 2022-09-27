# Creare un'app web per utilizzare un modello ML

In questa lezione, si addestrerà un modello ML su un insieme di dati fuori dal mondo: _avvistamenti di UFO nel secolo scorso_, provenienti dal [database di NUFORC](https://www.nuforc.org).

Si imparerà:

- Come serializzare/deserializzare un modello addestrato
- Come usare quel modello in un'app Flask

Si continuerà a utilizzare il notebook per pulire i dati e addestrare il modello, ma si può fare un ulteriore passo avanti nel processo esplorando l'utilizzo del modello direttamente in un'app web.

Per fare ciò, è necessario creare un'app Web utilizzando Flask.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/?loc=it)

## Costruire un'app

Esistono diversi modi per creare app Web per utilizzare modelli di machine learning. L'architettura web può influenzare il modo in cui il modello viene addestrato. Si immagini di lavorare in un'azienda nella quale il gruppo di data science ha addestrato un modello che va utilizzato in un'app.

### Considerazioni

Ci sono molte domande da porsi:

- **È un'app web o un'app su dispositivo mobile?** Se si sta creando un'app su dispositivo mobile o si deve usare il modello in un contesto IoT, ci si può avvalere [di TensorFlow Lite](https://www.tensorflow.org/lite/) e usare il modello in un'app Android o iOS.
- **Dove risiederà il modello**? E' utilizzato in cloud o in locale?
- **Supporto offline**. L'app deve funzionare offline?
- **Quale tecnologia è stata utilizzata per addestrare il modello?** La tecnologia scelta può influenzare gli strumenti che è necessario utilizzare.
   - **Utilizzare** TensorFlow. Se si sta addestrando un modello utilizzando TensorFlow, ad esempio, tale ecosistema offre la possibilità di convertire un modello TensorFlow per l'utilizzo in un'app Web utilizzando [TensorFlow.js](https://www.tensorflow.org/js/).
   - **Utilizzare PyTorch**. Se si sta costruendo un modello utilizzando una libreria come PyTorch[,](https://pytorch.org/) si ha la possibilità di esportarlo in formato [ONNX](https://onnx.ai/) ( Open Neural Network Exchange) per l'utilizzo in app Web JavaScript che possono utilizzare il [motore di esecuzione Onnx](https://www.onnxruntime.ai/). Questa opzione verrà esplorata in una lezione futura per un modello addestrato da Scikit-learn
   - **Utilizzo di Lobe.ai o Azure Custom vision**. Se si sta usando un sistema ML SaaS (Software as a Service) come [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) per addestrare un modello, questo tipo di software fornisce modi per esportare il modello per molte piattaforme, inclusa la creazione di un'API su misura da interrogare nel cloud dalla propria applicazione online.

Si ha anche l'opportunità di creare un'intera app Web Flask in grado di addestrare il modello stesso in un browser Web. Questo può essere fatto anche usando TensorFlow.js in un contesto JavaScript.

Per questo scopo, poiché si è lavorato con i notebook basati su Python, verranno esplorati i passaggi necessari per esportare un modello addestrato da tale notebook in un formato leggibile da un'app Web creata in Python.

## Strumenti

Per questa attività sono necessari due strumenti: Flask e Pickle, entrambi eseguiti su Python.

✅ Cos'è [Flask](https://palletsprojects.com/p/flask/)? Definito come un "micro-framework" dai suoi creatori, Flask fornisce le funzionalità di base dei framework web utilizzando Python e un motore di template per creare pagine web. Si dia un'occhiata a [questo modulo di apprendimento](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) per esercitarsi a sviluppare con Flask.

✅ Cos'è [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 è un modulo Python che serializza e de-serializza la struttura di un oggetto Python. Quando si utilizza pickle in un modello, si serializza o si appiattisce la sua struttura per l'uso sul web. Cautela: pickle non è intrinsecamente sicuro, quindi si faccia  attenzione se viene chiesto di de-serializzare un file. Un file creato con pickle ha il suffisso `.pkl`.

## Esercizio: pulire i dati

In questa lezione verranno utilizzati i dati di 80.000 avvistamenti UFO, raccolti dal Centro Nazionale per gli Avvistamenti di UFO [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Questi dati hanno alcune descrizioni interessanti di avvistamenti UFO, ad esempio:

- **Descrizione di esempio lunga**. "Un uomo emerge da un raggio di luce che di notte brilla su un campo erboso e corre verso il parcheggio della Texas Instruments".
- **Descrizione di esempio breve**. "le luci ci hanno inseguito".

Il foglio di calcolo [ufo.csv](../data/ufos.csv) include colonne su città (`city`), stato (`state`) e nazione (`country`) in cui è avvenuto l'avvistamento, la forma (`shape`) dell'oggetto e la sua latitudine (`latitude`) e longitudine (`longitude`).

Nel [notebook](../notebook.ipynb) vuoto incluso in questa lezione:

1. importare `pandas`, `matplotlib` e `numpy` come fatto nelle lezioni precedenti e importare il foglio di calcolo ufo.csv. Si può dare un'occhiata a un insieme di dati campione:

   ```python
   import pandas as pd
   import numpy as np

   ufos = pd.read_csv('../data/ufos.csv')
   ufos.head()
   ```

1. Convertire i dati ufos in un piccolo dataframe con nuove intestazioni Controllare i valori univoci nel campo `Country` .

   ```python
   ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

   ufos.Country.unique()
   ```

1. Ora si può ridurre la quantità di dati da gestire eliminando qualsiasi valore nullo e importando solo avvistamenti tra 1-60 secondi:

   ```python
   ufos.dropna(inplace=True)

   ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

   ufos.info()
   ```

1. Importare la libreria `LabelEncoder` di Scikit-learn per convertire i valori di testo per le nazioni in un numero:

   ✅ LabelEncoder codifica i dati in ordine alfabetico

   ```python
   from sklearn.preprocessing import LabelEncoder

   ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

   ufos.head()
   ```

   I dati dovrebbero assomigliare a questo:

   ```output
   	Seconds	Country	Latitude	Longitude
   2	20.0	3	    53.200000	-2.916667
   3	20.0	4	    28.978333	-96.645833
   14	30.0	4	    35.823889	-80.253611
   23	60.0	4	    45.582778	-122.352222
   24	3.0	    3	    51.783333	-0.783333
   ```

## Esercizio: costruire il proprio modello

Ora ci si può preparare per addestrare un modello portando i dati nei gruppi di addestramento e test.

1. Selezionare le tre caratteristiche su cui lo si vuole allenare come vettore X mentre il vettore y sarà `Country` Si deve essere in grado di inserire secondi (`Seconds`), latitudine (`Latitude`) e longitudine (`Longitude`) e ottenere un ID nazione da restituire.

   ```python
   from sklearn.model_selection import train_test_split

   Selected_features = ['Seconds','Latitude','Longitude']

   X = ufos[Selected_features]
   y = ufos['Country']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   ```

1. Addestrare il modello usando la regressione logistica:

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

La precisione non è male **(circa il 95%)**, non sorprende che `Country` e `Latitude/Longitude` siano correlati.

Il modello creato non è molto rivoluzionario in quanto si dovrebbe essere in grado di dedurre una nazione (`Country`) dalla sua latitudine e longitudine (`Latitude` e `Longitude`), ma è un buon esercizio provare ad allenare dai dati grezzi che sono stati puliti ed esportati, e quindi utilizzare questo modello in una app web.

## Esercizio: usare pickle con il modello

Ora è il momento di utilizzare _pickle_ con il modello! Lo si può fare in poche righe di codice. Una volta che è stato _serializzato con pickle_, caricare il modello e testarlo rispetto a un array di dati di esempio contenente valori per secondi, latitudine e longitudine,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Il modello restituisce **"3"**, che è il codice nazione per il Regno Unito. Fantastico! 👽

## Esercizio: creare un'app Flask

Ora si può creare un'app Flask per chiamare il modello e restituire risultati simili, ma in un modo visivamente più gradevole.

1. Iniziare creando una cartella chiamata **web-app** a livello del file _notebook.ipynb_ dove risiede il file _ufo-model.pkl_.

1. In quella cartella creare altre tre cartelle: **static**, con una cartella **css** al suo interno e **templates**. Ora si dovrebbero avere i seguenti file e directory:

   ```output
   web-app/
     static/
       css/
     templates/
   notebook.ipynb
   ufo-model.pkl
   ```

   ✅ Fare riferimento alla cartella della soluzione per una visualizzazione dell'app finita.

1. Il primo file da creare nella cartella _web-app_ è il file **requirements.txt**. Come _package.json_ in un'app JavaScript, questo file elenca le dipendenze richieste dall'app. In **requirements.txt** aggiungere le righe:

   ```text
   scikit-learn
   pandas
   numpy
   flask
   ```

1. Ora, eseguire questo file portandosi su _web-app_:

   ```bash
   cd web-app
   ```

1. Aprire una finestra di terminale dove risiede requirements.txt e digitare `pip install`, per installare le librerie elencate in _reuirements.txt_:

   ```bash
   pip install -r requirements.txt
   ```

1. Ora si è pronti per creare altri tre file per completare l'app:

   1. Creare **app.py** nella directory radice.
   2. Creare **index.html** nella directory _templates_.
   3. Creare **sytles.css** nella directory _static/css_.

1. Inserire nel file _styles.css_ alcuni stili:

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

1. Quindi, creare il file _index.html_ :

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

   Dare un'occhiata al template di questo file. Notare la sintassi con le parentesi graffe attorno alle variabili che verranno fornite dall'app, come il testo di previsione: `{{}}`. C'è anche un modulo che invia una previsione alla rotta `/predict`.

   Infine, si è pronti per creare il file python che guida il consumo del modello e la visualizzazione delle previsioni:

1. In `app.py` aggiungere:

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

   > 💡 Suggerimento: quando si aggiunge [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) durante l'esecuzione dell'app web utilizzando Flask, qualsiasi modifica apportata all'applicazione verrà recepita immediatamente senza la necessità di riavviare il server. Attenzione! Non abilitare questa modalità in un'app di produzione.

Se si esegue `python app.py` o `python3 app.py` , il server web si avvia, localmente, e si può compilare un breve modulo per ottenere una risposta alla domanda scottante su dove sono stati avvistati gli UFO!

Prima di farlo, dare un'occhiata alle parti di `app.py`:

1. Innanzitutto, le dipendenze vengono caricate e l'app si avvia.
1. Poi il modello viene importato.
1. Infine index.html viene visualizzato sulla rotta home.

Sulla rotta `/predict` , accadono diverse cose quando il modulo viene inviato:

1. Le variabili del modulo vengono raccolte e convertite in un array numpy. Vengono quindi inviate al modello e viene restituita una previsione.
2. Le nazioni che si vogliono visualizzare vengono nuovamente esposte come testo leggibile ricavato dal loro codice paese previsto e tale valore viene inviato a index.html per essere visualizzato nel template della pagina web.

Usare un modello in questo modo, con Flask e un modello serializzato è relativamente semplice. La cosa più difficile è capire che forma hanno i dati che devono essere inviati al modello per ottenere una previsione. Tutto dipende da come è stato addestrato il modello. Questo ha tre punti dati da inserire per ottenere una previsione.

In un ambiente professionale, si può vedere quanto sia necessaria una buona comunicazione tra le persone che addestrano il modello e coloro che lo consumano in un'app web o su dispositivo mobile. In questo caso, si ricoprono entrambi i ruoli!

---

## 🚀 Sfida

Invece di lavorare su un notebook e importare il modello nell'app Flask, si può addestrare il modello direttamente nell'app Flask! Provare a convertire il codice Python nel notebook, magari dopo che i dati sono stati puliti, per addestrare il modello dall'interno dell'app su un percorso chiamato `/train`. Quali sono i pro e i contro nel seguire questo metodo?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/?loc=it)

## Revisione e Auto Apprendimento

Esistono molti modi per creare un'app web per utilizzare i modelli ML. Elencare dei modi in cui si potrebbe utilizzare JavaScript o Python per creare un'app web per sfruttare machine learning. Considerare l'architettura: il modello dovrebbe rimanere nell'app o risiedere nel cloud? In quest'ultimo casi, come accedervi? Disegnare un modello architettonico per una soluzione web ML applicata.

## Compito

[Provare un modello diverso](assignment.it.md)


