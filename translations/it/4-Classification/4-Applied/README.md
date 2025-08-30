<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-08-29T21:47:13+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "it"
}
-->
# Crea un'app web per raccomandare cucine

In questa lezione, costruirai un modello di classificazione utilizzando alcune delle tecniche apprese nelle lezioni precedenti e il delizioso dataset di cucine utilizzato in questa serie. Inoltre, creerai una piccola app web per utilizzare un modello salvato, sfruttando il runtime web di Onnx.

Uno degli utilizzi pratici più utili del machine learning è la creazione di sistemi di raccomandazione, e oggi puoi fare il primo passo in questa direzione!

[![Presentazione di questa app web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Clicca sull'immagine sopra per un video: Jen Looper crea un'app web utilizzando dati di cucina classificati

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

In questa lezione imparerai:

- Come costruire un modello e salvarlo come modello Onnx
- Come utilizzare Netron per ispezionare il modello
- Come utilizzare il tuo modello in un'app web per fare inferenze

## Crea il tuo modello

Costruire sistemi di machine learning applicati è una parte importante per sfruttare queste tecnologie nei sistemi aziendali. Puoi utilizzare i modelli all'interno delle tue applicazioni web (e quindi usarli in un contesto offline, se necessario) utilizzando Onnx.

In una [lezione precedente](../../3-Web-App/1-Web-App/README.md), hai costruito un modello di regressione sui avvistamenti UFO, lo hai "pickled" e lo hai utilizzato in un'app Flask. Sebbene questa architettura sia molto utile da conoscere, è un'app Python full-stack, e i tuoi requisiti potrebbero includere l'uso di un'applicazione JavaScript.

In questa lezione, puoi costruire un sistema di base basato su JavaScript per fare inferenze. Prima, però, devi allenare un modello e convertirlo per l'uso con Onnx.

## Esercizio - allenare un modello di classificazione

Per prima cosa, allena un modello di classificazione utilizzando il dataset di cucine pulito che abbiamo usato.

1. Inizia importando le librerie utili:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Hai bisogno di '[skl2onnx](https://onnx.ai/sklearn-onnx/)' per aiutarti a convertire il tuo modello Scikit-learn in formato Onnx.

1. Poi, lavora con i tuoi dati nello stesso modo in cui hai fatto nelle lezioni precedenti, leggendo un file CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Rimuovi le prime due colonne inutili e salva i dati rimanenti come 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Salva le etichette come 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Inizia la routine di allenamento

Utilizzeremo la libreria 'SVC', che ha una buona accuratezza.

1. Importa le librerie appropriate da Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separa i set di allenamento e test:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Costruisci un modello di classificazione SVC come hai fatto nella lezione precedente:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Ora, testa il tuo modello chiamando `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Stampa un report di classificazione per verificare la qualità del modello:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Come abbiamo visto prima, l'accuratezza è buona:

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

### Converti il tuo modello in Onnx

Assicurati di fare la conversione con il numero corretto di Tensor. Questo dataset ha 380 ingredienti elencati, quindi devi annotare quel numero in `FloatTensorType`:

1. Converti utilizzando un numero di tensor pari a 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crea il file onx e salvalo come **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, puoi passare [opzioni](https://onnx.ai/sklearn-onnx/parameterized.html) nel tuo script di conversione. In questo caso, abbiamo passato 'nocl' come True e 'zipmap' come False. Poiché questo è un modello di classificazione, hai l'opzione di rimuovere ZipMap che produce una lista di dizionari (non necessaria). `nocl` si riferisce alle informazioni di classe incluse nel modello. Riduci la dimensione del tuo modello impostando `nocl` su 'True'.

Eseguendo l'intero notebook ora costruirai un modello Onnx e lo salverai in questa cartella.

## Visualizza il tuo modello

I modelli Onnx non sono molto visibili in Visual Studio Code, ma c'è un ottimo software gratuito che molti ricercatori usano per visualizzare il modello e assicurarsi che sia costruito correttamente. Scarica [Netron](https://github.com/lutzroeder/Netron) e apri il file model.onnx. Puoi vedere il tuo semplice modello visualizzato, con i suoi 380 input e il classificatore elencato:

![Visualizzazione Netron](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.it.png)

Netron è uno strumento utile per visualizzare i tuoi modelli.

Ora sei pronto per utilizzare questo modello in un'app web. Creiamo un'app che sarà utile quando guardi nel tuo frigorifero e cerchi di capire quale combinazione di ingredienti avanzati puoi usare per cucinare un determinato piatto, come determinato dal tuo modello.

## Crea un'applicazione web di raccomandazione

Puoi utilizzare il tuo modello direttamente in un'app web. Questa architettura ti consente anche di eseguirla localmente e persino offline, se necessario. Inizia creando un file `index.html` nella stessa cartella in cui hai salvato il file `model.onnx`.

1. In questo file _index.html_, aggiungi il seguente markup:

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

1. Ora, lavorando all'interno dei tag `body`, aggiungi un po' di markup per mostrare un elenco di checkbox che riflettono alcuni ingredienti:

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

    Nota che a ogni checkbox viene assegnato un valore. Questo riflette l'indice in cui l'ingrediente si trova secondo il dataset. La mela, ad esempio, in questa lista alfabetica, occupa la quinta colonna, quindi il suo valore è '4' poiché iniziamo a contare da 0. Puoi consultare il [foglio di calcolo degli ingredienti](../../../../4-Classification/data/ingredient_indexes.csv) per scoprire l'indice di un determinato ingrediente.

    Continuando il tuo lavoro nel file index.html, aggiungi un blocco di script dove il modello viene chiamato dopo il tag di chiusura finale `</div>`.

1. Per prima cosa, importa il [Runtime Onnx](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime viene utilizzato per eseguire i modelli Onnx su una vasta gamma di piattaforme hardware, includendo ottimizzazioni e un'API da utilizzare.

1. Una volta che il Runtime è in posizione, puoi chiamarlo:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
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
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

In questo codice, ci sono diverse cose che accadono:

1. Hai creato un array di 380 possibili valori (1 o 0) da impostare e inviare al modello per l'inferenza, a seconda che una checkbox di ingrediente sia selezionata.
2. Hai creato un array di checkbox e un modo per determinare se sono selezionate in una funzione `init` che viene chiamata quando l'applicazione inizia. Quando una checkbox è selezionata, l'array `ingredients` viene modificato per riflettere l'ingrediente scelto.
3. Hai creato una funzione `testCheckboxes` che verifica se è stata selezionata almeno una checkbox.
4. Usi la funzione `startInference` quando il pulsante viene premuto e, se è selezionata almeno una checkbox, inizi l'inferenza.
5. La routine di inferenza include:
   1. Configurare un caricamento asincrono del modello
   2. Creare una struttura Tensor da inviare al modello
   3. Creare 'feeds' che riflettono l'input `float_input` che hai creato durante l'allenamento del modello (puoi usare Netron per verificare quel nome)
   4. Inviare questi 'feeds' al modello e attendere una risposta

## Testa la tua applicazione

Apri una sessione terminale in Visual Studio Code nella cartella in cui risiede il tuo file index.html. Assicurati di avere [http-server](https://www.npmjs.com/package/http-server) installato globalmente e digita `http-server` al prompt. Si aprirà un localhost e potrai visualizzare la tua app web. Controlla quale cucina viene raccomandata in base ai vari ingredienti:

![App web degli ingredienti](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.it.png)

Congratulazioni, hai creato un'app web di 'raccomandazione' con pochi campi. Prenditi del tempo per sviluppare ulteriormente questo sistema!

## 🚀Sfida

La tua app web è molto minimale, quindi continua a svilupparla utilizzando gli ingredienti e i loro indici dai dati [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Quali combinazioni di sapori funzionano per creare un determinato piatto nazionale?

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## Revisione & Studio Autonomo

Sebbene questa lezione abbia solo accennato all'utilità di creare un sistema di raccomandazione per gli ingredienti alimentari, quest'area di applicazioni ML è molto ricca di esempi. Leggi di più su come vengono costruiti questi sistemi:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Compito 

[Costruisci un nuovo sistema di raccomandazione](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali incomprensioni o interpretazioni errate derivanti dall'uso di questa traduzione.