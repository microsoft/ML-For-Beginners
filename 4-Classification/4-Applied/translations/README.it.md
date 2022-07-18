# Costruire un'App Web per Consigliare una Cucina

In questa lezione si creerà un modello di classificazione utilizzando alcune delle tecniche apprese nelle lezioni precedenti e con il delizioso insieme di dati sulla cucina utilizzato in questa serie. Inoltre, si creerà una piccola app web per utilizzare un modello salvato, sfruttando il runtime web di Onnx.

Uno degli usi pratici più utili dell'apprendimento automatico è la creazione di sistemi di raccomandazione e oggi si può fare il primo passo in quella direzione!

[![Introduzione ai Sistemi di Raccomandazione](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Fare clic sull'immagine sopra per un video

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/?loc=it)

In questa lezione, si imparerà:

- Come costruire un modello e salvarlo come modello Onnx
- Come usare Netron per ispezionare il modello
- Come utilizzare il modello in un'app web per l'inferenza

## Costruire il modello

La creazione di sistemi ML applicati è una parte importante dell'utilizzo di queste tecnologie per i sistemi aziendali. Si possono utilizzare i modelli all'interno delle proprie applicazioni web (e quindi utilizzarli in un contesto offline se necessario) utilizzando Onnx.

In una [lezione precedente](../../../3-Web-App/1-Web-App/translations/README.it.md) si è costruito un modello di regressione sugli avvistamenti di UFO, è stato serializzato e lo si è utilizzato in un'app Flask. Sebbene questa architettura sia molto utile da conoscere, è un'app Python completa e i requisiti potrebbero includere l'uso di un'applicazione JavaScript.

In questa lezione si può creare un sistema di inferenza di base utilizzando JavaScript. Prima, tuttavia, è necessario addestrare un modello e convertirlo per l'utilizzo con Onnx.

## Esercizio - modello di classificazione di addestramento

Innanzitutto, addestrare un modello di classificazione utilizzando l'insieme di dati pulito delle cucine precedentemente usato.

1. Iniziare importando librerie utili:

   ```python
   !pip install skl2onnx
   import pandas as pd
   ```

   Serve '[skl2onnx](https://onnx.ai/sklearn-onnx/)' per poter convertire il modello di Scikit-learn in formato Onnx.

1. Quindi si  lavora con i dati nello stesso modo delle lezioni precedenti, leggendo un file CSV usando `read_csv()`:

   ```python
   data = pd.read_csv('../data/cleaned_cuisine.csv')
   data.head()
   ```

1. Rimuovere le prime due colonne non necessarie e salvare i dati rimanenti come "X":

   ```python
   X = data.iloc[:,2:]
   X.head()
   ```

1. Salvare le etichette come "y":

   ```python
   y = data[['cuisine']]
   y.head()

   ```

### Iniziare la routine di addestramento

Verrà usata la libreria 'SVC' che ha una buona precisione.

1. Importare le librerie appropriate da Scikit-learn:

   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.svm import SVC
   from sklearn.model_selection import cross_val_score
   from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
   ```

1. Separare gli insiemi di allenamento e test:

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
   ```

1. Costruire un modello di classificazione SVC come fatto nella lezione precedente:

   ```python
   model = SVC(kernel='linear', C=10, probability=True,random_state=0)
   model.fit(X_train,y_train.values.ravel())
   ```

1. Ora provare il modello, chiamando `predict()`:

   ```python
   y_pred = model.predict(X_test)
   ```

1. Stampare un rapporto di classificazione per verificare la qualità del modello:

   ```python
   print(classification_report(y_test,y_pred))
   ```

   Come visto prima, la precisione è buona:

   ```output
                   precision    recall  f1-score   support

        chinese       0.72      0.69      0.70       257
         indian       0.91      0.87      0.89       243
       japanese       0.79      0.77      0.78       239
         korean       0.83      0.79      0.81       236
           thai       0.72      0.84      0.78       224

       accuracy                           0.79      1199
      macro avg       0.79      0.79      0.79      1199
   weighted avg       0.79      0.79      0.79      1199
   ```

### Convertire il modello in Onnx

Assicurarsi di eseguire la conversione con il numero tensore corretto. Questo insieme di dati ha 380 ingredienti elencati, quindi è necessario annotare quel numero in `FloatTensorType`:

1. Convertire usando un numero tensore di 380.

   ```python
   from skl2onnx import convert_sklearn
   from skl2onnx.common.data_types import FloatTensorType

   initial_type = [('float_input', FloatTensorType([None, 380]))]
   options = {id(model): {'nocl': True, 'zipmap': False}}
   ```

1. Creare l'onx e salvarlo come file **model.onnx**:

   ```python
   onx = convert_sklearn(model, initial_types=initial_type, options=options)
   with open("./model.onnx", "wb") as f:
       f.write(onx.SerializeToString())
   ```

   > Nota, si possono passare le[opzioni](https://onnx.ai/sklearn-onnx/parameterized.html) nello  script di conversione. In questo caso, si è passato 'nocl' come True e 'zipmap' come False. Poiché questo è un modello di classificazione, si ha la possibilità di rimuovere ZipMap che produce un elenco di dizionari (non necessario). `nocl` si riferisce alle informazioni sulla classe incluse nel modello. Ridurre le dimensioni del modello impostando `nocl` su 'True'.

L'esecuzione dell'intero notebook ora creerà un modello Onnx e lo salverà in questa cartella.

## Visualizzare il modello

I modelli Onnx non sono molto visualizzabili in Visual Studio code, ma c'è un ottimo software gratuito che molti ricercatori usano per visualizzare il modello per assicurarsi che sia costruito correttamente. Scaricare [Netron](https://github.com/lutzroeder/Netron) e aprire il file model.onnx. Si può vedere il modello semplice visualizzato, con i suoi 380 input e classificatore elencati:

![Vista Netron ](../images/netron.png)

Netron è uno strumento utile per visualizzare i modelli.

Ora si è pronti per utilizzare questo modello accurato in un'app web. Si costruisce un'app che tornerà utile quando si guarda nel frigorifero e si prova a capire quale combinazione di ingredienti avanzati si può usare per cucinare una determinata tipologia di cucina, come determinato dal  modello.

## Creare un'applicazione web di raccomandazione

Si può utilizzare il modello direttamente in un'app web. Questa architettura consente anche di eseguirlo localmente e anche offline se necessario. Iniziare creando un file `index.html` nella stessa cartella in cui si è salvato il file `model.onnx`.

1. In questo file _index.html_, aggiungere il seguente codice markup:

   ```html
   <!DOCTYPE html>
   <html>
       <header>
           <title>Cuisine Matcher</title>
       </header>
       <body>
           ...
       </body>
   </html>
   ```

1. Ora, lavorando all'interno del tag `body` , aggiungere un piccolo markup per mostrare un elenco di caselle di controllo che riflettono alcuni ingredienti:

   ```html
   <h1>Check your refrigerator. What can you create?</h1>
           <div id="wrapper">
               <div class="boxCont">
                   <input type="checkbox" value="4" class="checkbox">
                   <label>apple</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="247" class="checkbox">
                   <label>pear</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="77" class="checkbox">
                   <label>cherry</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="126" class="checkbox">
                   <label>fenugreek</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="302" class="checkbox">
                   <label>sake</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="327" class="checkbox">
                   <label>soy sauce</label>
               </div>

               <div class="boxCont">
                   <input type="checkbox" value="112" class="checkbox">
                   <label>cumin</label>
               </div>
           </div>
           <div style="padding-top:10px">
               <button onClick="startInference()">What kind of cuisine can you make?</button>
           </div>
   ```

   Notare che a ogni casella di controllo viene assegnato un valore. Questo riflette l'indice in cui si trova l'ingrediente in base all'insieme di dati. Apple, ad esempio, in questo elenco alfabetico, occupa la quinta colonna, quindi il suo valore è "4" poiché si inizia a contare da 0. Si può consultare il [foglio di calcolo degli ingredienti](../../data/ingredient_indexes.csv) per scoprire l'indice di un determinato ingrediente.

   Continuando il lavoro nel file index.html, aggiungere un blocco di script in cui viene chiamato il modello dopo la chiusura del tag `</div>` finale.

1. Innanzitutto, importare il [runtime Onnx](https://www.onnxruntime.ai/):

   ```html
   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.8.0-dev.20210608.0/dist/ort.min.js"></script>
   ```

   > Onnx Runtime viene utilizzato per consentire l'esecuzione dei modelli Onnx su un'ampia gamma di piattaforme hardware, comprese le ottimizzazioni e un'API da utilizzare.

1. Una volta che il Runtime è a posto, lo si può chiamare:

   ```javascript
   <script>
               const ingredients = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

               const checks = [].slice.call(document.querySelectorAll('.checkbox'));

               // use an async context to call onnxruntime functions.
               function init() {

                   checks.forEach(function (checkbox, index) {
                       checkbox.onchange = function () {
                           if (this.checked) {
                               var index = checkbox.value;

                               if (index !== -1) {
                                   ingredients[index] = 1;
                               }
                               console.log(ingredients)
                           }
                           else {
                               var index = checkbox.value;

                               if (index !== -1) {
                                   ingredients[index] = 0;
                               }
                               console.log(ingredients)
                           }
                       }
                   })
               }

               function testCheckboxes() {
                       for (var i = 0; i < checks.length; i++)
                           if (checks[i].type == "checkbox")
                               if (checks[i].checked)
                                   return true;
                       return false;
               }

               async function startInference() {

                   let checked = testCheckboxes()

                   if (checked) {

                   try {
                       // create a new session and load the model.

                       const session = await ort.InferenceSession.create('./model.onnx');

                       const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                       const feeds = { float_input: input };

                       // feed inputs and run

                       const results = await session.run(feeds);

                       // read from results
                       alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

                   } catch (e) {
                       console.log(`failed to inference ONNX model: ${e}.`);
                   }
               }
               else alert("Please check an ingredient")

               }
       init();

           </script>
   ```

In questo codice, accadono diverse cose:

1. Si è creato un array di 380 possibili valori (1 o 0) da impostare e inviare al modello per l'inferenza, a seconda che una casella di controllo dell'ingrediente sia selezionata.
2. Si è creata una serie di caselle di controllo e un modo per determinare se sono state selezionate in una funzione `init` chiamata all'avvio dell'applicazione. Quando una casella di controllo è selezionata, l 'array `ingredients` viene modificato per riflettere l'ingrediente scelto.
3. Si è creata una funzione `testCheckboxes` che controlla se una casella di controllo è stata selezionata.
4. Si utilizza quella funzione quando si preme il pulsante e, se una casella di controllo è selezionata, si avvia l'inferenza.
5. La routine di inferenza include:
   1. Impostazione di un caricamento asincrono del modello
   2. Creazione di una struttura tensoriale da inviare al modello
   3. Creazione di "feed" che riflettano l'input `float_input` creato durante l'addestramento del modello (si può usare Netron per verificare quel nome)
   4. Invio di questi "feed" al modello e attesa di una risposta

## Verificare l'applicazione

Aprire una sessione terminale in Visual Studio Code nella cartella in cui risiede il file index.html. Assicurarsi di avere [http-server](https://www.npmjs.com/package/http-server) installato globalmente e digitare `http-server` al prompt. Dovrebbe aprirsi nel browser un localhost e si può visualizzare l'app web. Controllare quale cucina è consigliata in base ai vari ingredienti:

![app web degli ingredienti](../images/web-app.png)

Congratulazioni, si è creato un'app web di "raccomandazione" con pochi campi. Si prenda del tempo per costruire questo sistema!
## 🚀 Sfida

L'app web è molto minimale, quindi continuare a costruirla usando gli ingredienti e i loro indici dai dati [ingredient_indexes](../../data/ingredient_indexes.csv) . Quali combinazioni di sapori funzionano per creare un determinato piatto nazionale?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/?loc=it)

## Revisione e Auto Apprendimento

Sebbene questa lezione abbia appena toccato l'utilità di creare un sistema di raccomandazione per gli ingredienti alimentari, quest'area delle applicazioni ML è molto ricca di esempi. Leggere di più su come sono costruiti questi sistemi:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Compito

[Creare un nuovo sistema di raccomandazione](assignment.it.md)
