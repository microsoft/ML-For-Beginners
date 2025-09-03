<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2680c691fbdb6367f350761a275e2508",
  "translation_date": "2025-08-29T21:37:23+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "it"
}
-->
# Costruire un'app web per utilizzare un modello ML

In questa lezione, allenerai un modello ML su un set di dati davvero fuori dal comune: _Avvistamenti UFO nell'ultimo secolo_, provenienti dal database di NUFORC.

Imparerai:

- Come 'pickle' un modello allenato
- Come utilizzare quel modello in un'app Flask

Continueremo a utilizzare i notebook per pulire i dati e allenare il nostro modello, ma puoi fare un passo ulteriore esplorando l'uso di un modello "nel mondo reale", per così dire: in un'app web.

Per fare ciò, devi costruire un'app web utilizzando Flask.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Costruire un'app

Ci sono diversi modi per costruire app web che consumano modelli di machine learning. La tua architettura web potrebbe influenzare il modo in cui il tuo modello viene allenato. Immagina di lavorare in un'azienda dove il gruppo di data science ha allenato un modello che vogliono tu utilizzi in un'app.

### Considerazioni

Ci sono molte domande che devi porti:

- **È un'app web o un'app mobile?** Se stai costruendo un'app mobile o hai bisogno di utilizzare il modello in un contesto IoT, potresti usare [TensorFlow Lite](https://www.tensorflow.org/lite/) e utilizzare il modello in un'app Android o iOS.
- **Dove risiederà il modello?** Nel cloud o localmente?
- **Supporto offline.** L'app deve funzionare offline?
- **Quale tecnologia è stata utilizzata per allenare il modello?** La tecnologia scelta potrebbe influenzare gli strumenti che devi utilizzare.
    - **Utilizzando TensorFlow.** Se stai allenando un modello con TensorFlow, ad esempio, quell'ecosistema offre la possibilità di convertire un modello TensorFlow per l'uso in un'app web utilizzando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Utilizzando PyTorch.** Se stai costruendo un modello utilizzando una libreria come [PyTorch](https://pytorch.org/), hai la possibilità di esportarlo in formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) per l'uso in app web JavaScript che possono utilizzare [Onnx Runtime](https://www.onnxruntime.ai/). Questa opzione sarà esplorata in una lezione futura per un modello allenato con Scikit-learn.
    - **Utilizzando Lobe.ai o Azure Custom Vision.** Se stai utilizzando un sistema ML SaaS (Software as a Service) come [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) per allenare un modello, questo tipo di software offre modi per esportare il modello per molte piattaforme, incluso costruire un'API personalizzata da interrogare nel cloud dalla tua applicazione online.

Hai anche l'opportunità di costruire un'intera app web Flask che sarebbe in grado di allenare il modello direttamente in un browser web. Questo può essere fatto anche utilizzando TensorFlow.js in un contesto JavaScript.

Per i nostri scopi, poiché abbiamo lavorato con notebook basati su Python, esploriamo i passaggi necessari per esportare un modello allenato da un notebook in un formato leggibile da un'app web costruita in Python.

## Strumenti

Per questo compito, hai bisogno di due strumenti: Flask e Pickle, entrambi funzionanti su Python.

✅ Cos'è [Flask](https://palletsprojects.com/p/flask/)? Definito come un 'micro-framework' dai suoi creatori, Flask fornisce le funzionalità di base dei framework web utilizzando Python e un motore di template per costruire pagine web. Dai un'occhiata a [questo modulo di apprendimento](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) per esercitarti a costruire con Flask.

✅ Cos'è [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 è un modulo Python che serializza e deserializza una struttura di oggetti Python. Quando 'pickle' un modello, ne serializzi o appiattisci la struttura per l'uso sul web. Attenzione: pickle non è intrinsecamente sicuro, quindi fai attenzione se ti viene chiesto di 'un-pickle' un file. Un file pickled ha il suffisso `.pkl`.

## Esercizio - pulire i dati

In questa lezione utilizzerai dati provenienti da 80.000 avvistamenti UFO, raccolti da [NUFORC](https://nuforc.org) (National UFO Reporting Center). Questi dati contengono descrizioni interessanti degli avvistamenti UFO, ad esempio:

- **Descrizione lunga.** "Un uomo emerge da un raggio di luce che illumina un campo erboso di notte e corre verso il parcheggio di Texas Instruments".
- **Descrizione breve.** "le luci ci inseguivano".

Il foglio di calcolo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) include colonne riguardanti la `città`, lo `stato` e il `paese` dove è avvenuto l'avvistamento, la `forma` dell'oggetto e la sua `latitudine` e `longitudine`.

Nel [notebook](notebook.ipynb) vuoto incluso in questa lezione:

1. importa `pandas`, `matplotlib` e `numpy` come hai fatto nelle lezioni precedenti e importa il foglio di calcolo ufos. Puoi dare un'occhiata a un set di dati di esempio:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converti i dati ufos in un piccolo dataframe con titoli aggiornati. Controlla i valori unici nel campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ora puoi ridurre la quantità di dati con cui lavorare eliminando i valori nulli e importando solo avvistamenti tra 1-60 secondi:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa la libreria `LabelEncoder` di Scikit-learn per convertire i valori di testo dei paesi in numeri:

    ✅ LabelEncoder codifica i dati alfabeticamente

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

## Esercizio - costruire il modello

Ora puoi prepararti ad allenare un modello dividendo i dati in gruppi di allenamento e test.

1. Seleziona le tre caratteristiche su cui vuoi allenarti come vettore X, e il vettore y sarà il `Country`. Vuoi essere in grado di inserire `Seconds`, `Latitude` e `Longitude` e ottenere un id paese da restituire.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Allena il tuo modello utilizzando la regressione logistica:

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

L'accuratezza non è male **(circa 95%)**, non sorprende, poiché `Country` e `Latitude/Longitude` sono correlati.

Il modello che hai creato non è molto rivoluzionario, poiché dovresti essere in grado di dedurre un `Country` dalla sua `Latitude` e `Longitude`, ma è un buon esercizio per provare ad allenarti partendo da dati grezzi che hai pulito, esportato e poi utilizzato in un'app web.

## Esercizio - 'pickle' il modello

Ora è il momento di _pickle_ il tuo modello! Puoi farlo in poche righe di codice. Una volta _pickled_, carica il tuo modello pickled e testalo con un array di dati di esempio contenente valori per seconds, latitude e longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Il modello restituisce **'3'**, che è il codice paese per il Regno Unito. Incredibile! 👽

## Esercizio - costruire un'app Flask

Ora puoi costruire un'app Flask per chiamare il tuo modello e restituire risultati simili, ma in un modo più visivamente piacevole.

1. Inizia creando una cartella chiamata **web-app** accanto al file _notebook.ipynb_ dove risiede il tuo file _ufo-model.pkl_.

1. In quella cartella crea altre tre cartelle: **static**, con una cartella **css** al suo interno, e **templates**. Ora dovresti avere i seguenti file e directory:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta la cartella soluzione per una vista dell'app completata

1. Il primo file da creare nella cartella _web-app_ è il file **requirements.txt**. Come _package.json_ in un'app JavaScript, questo file elenca le dipendenze richieste dall'app. In **requirements.txt** aggiungi le righe:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ora, esegui questo file navigando in _web-app_:

    ```bash
    cd web-app
    ```

1. Nel tuo terminale digita `pip install`, per installare le librerie elencate in _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ora sei pronto per creare altri tre file per completare l'app:

    1. Crea **app.py** nella radice.
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

    Dai un'occhiata al templating in questo file. Nota la sintassi 'mustache' attorno alle variabili che saranno fornite dall'app, come il testo della previsione: `{{}}`. C'è anche un modulo che invia una previsione al percorso `/predict`.

    Infine, sei pronto per costruire il file Python che guida il consumo del modello e la visualizzazione delle previsioni:

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

    > 💡 Suggerimento: quando aggiungi [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) mentre esegui l'app web utilizzando Flask, qualsiasi modifica apportata alla tua applicazione sarà immediatamente riflessa senza la necessità di riavviare il server. Attenzione! Non abilitare questa modalità in un'app di produzione.

Se esegui `python app.py` o `python3 app.py` - il tuo server web si avvia localmente e puoi compilare un breve modulo per ottenere una risposta alla tua domanda urgente su dove sono stati avvistati gli UFO!

Prima di farlo, dai un'occhiata alle parti di `app.py`:

1. Prima vengono caricate le dipendenze e l'app si avvia.
1. Poi viene importato il modello.
1. Successivamente, index.html viene renderizzato sul percorso home.

Sul percorso `/predict`, accadono diverse cose quando il modulo viene inviato:

1. Le variabili del modulo vengono raccolte e convertite in un array numpy. Vengono quindi inviate al modello e viene restituita una previsione.
2. I Paesi che vogliamo visualizzare vengono ri-renderizzati come testo leggibile dal loro codice paese previsto, e quel valore viene inviato nuovamente a index.html per essere renderizzato nel template.

Utilizzare un modello in questo modo, con Flask e un modello pickled, è relativamente semplice. La cosa più difficile è capire quale forma devono avere i dati che devono essere inviati al modello per ottenere una previsione. Tutto dipende da come il modello è stato allenato. Questo ha tre punti dati da inserire per ottenere una previsione.

In un contesto professionale, puoi vedere quanto sia necessaria una buona comunicazione tra le persone che allenano il modello e quelle che lo consumano in un'app web o mobile. Nel nostro caso, è solo una persona: tu!

---

## 🚀 Sfida

Invece di lavorare in un notebook e importare il modello nell'app Flask, potresti allenare il modello direttamente nell'app Flask! Prova a convertire il tuo codice Python nel notebook, magari dopo che i tuoi dati sono stati puliti, per allenare il modello direttamente nell'app su un percorso chiamato `train`. Quali sono i pro e i contro di perseguire questo metodo?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Revisione e studio autonomo

Ci sono molti modi per costruire un'app web che consumi modelli ML. Fai un elenco dei modi in cui potresti utilizzare JavaScript o Python per costruire un'app web che sfrutti il machine learning. Considera l'architettura: il modello dovrebbe rimanere nell'app o vivere nel cloud? Se quest'ultimo, come lo accederesti? Disegna un modello architettonico per una soluzione ML web applicata.

## Compito

[Prova un modello diverso](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.