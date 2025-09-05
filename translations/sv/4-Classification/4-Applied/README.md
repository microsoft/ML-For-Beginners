<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T21:52:12+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "sv"
}
-->
# Bygg en webapp för matrekommendationer

I den här lektionen kommer du att bygga en klassificeringsmodell med hjälp av några av de tekniker du har lärt dig i tidigare lektioner och med det läckra datasetet om mat som används genom hela denna serie. Dessutom kommer du att bygga en liten webapp för att använda en sparad modell, med hjälp av Onnx:s web runtime.

En av de mest praktiska användningarna av maskininlärning är att bygga rekommendationssystem, och idag kan du ta det första steget i den riktningen!

[![Presentera denna webapp](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klicka på bilden ovan för en video: Jen Looper bygger en webapp med klassificerad matdata

## [Quiz före lektionen](https://ff-quizzes.netlify.app/en/ml/)

I den här lektionen kommer du att lära dig:

- Hur man bygger en modell och sparar den som en Onnx-modell
- Hur man använder Netron för att inspektera modellen
- Hur man använder din modell i en webapp för inferens

## Bygg din modell

Att bygga tillämpade ML-system är en viktig del av att utnyttja dessa teknologier för dina affärssystem. Du kan använda modeller inom dina webbapplikationer (och därmed använda dem offline om det behövs) genom att använda Onnx.

I en [tidigare lektion](../../3-Web-App/1-Web-App/README.md) byggde du en regressionsmodell om UFO-observationer, "picklade" den och använde den i en Flask-app. Även om denna arkitektur är mycket användbar att känna till, är det en fullstack Python-app, och dina krav kan inkludera användning av en JavaScript-applikation.

I den här lektionen kan du bygga ett grundläggande JavaScript-baserat system för inferens. Först måste du dock träna en modell och konvertera den för användning med Onnx.

## Övning - träna klassificeringsmodell

Först, träna en klassificeringsmodell med det rensade datasetet om mat som vi använde.

1. Börja med att importera användbara bibliotek:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Du behöver '[skl2onnx](https://onnx.ai/sklearn-onnx/)' för att hjälpa till att konvertera din Scikit-learn-modell till Onnx-format.

1. Arbeta sedan med din data på samma sätt som du gjorde i tidigare lektioner, genom att läsa en CSV-fil med `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Ta bort de två första onödiga kolumnerna och spara den återstående datan som 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Spara etiketterna som 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Starta träningsrutinen

Vi kommer att använda biblioteket 'SVC' som har bra noggrannhet.

1. Importera lämpliga bibliotek från Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separera tränings- och testuppsättningar:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Bygg en SVC-klassificeringsmodell som du gjorde i den tidigare lektionen:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Testa nu din modell genom att kalla på `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Skriv ut en klassificeringsrapport för att kontrollera modellens kvalitet:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Som vi såg tidigare är noggrannheten bra:

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

### Konvertera din modell till Onnx

Se till att göra konverteringen med rätt Tensor-nummer. Detta dataset har 380 ingredienser listade, så du behöver ange det numret i `FloatTensorType`:

1. Konvertera med ett tensornummer på 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Skapa onx och spara som en fil **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Observera att du kan skicka in [alternativ](https://onnx.ai/sklearn-onnx/parameterized.html) i ditt konverteringsskript. I detta fall skickade vi in 'nocl' som True och 'zipmap' som False. Eftersom detta är en klassificeringsmodell har du möjlighet att ta bort ZipMap som producerar en lista med ordböcker (inte nödvändigt). `nocl` hänvisar till att klassinformation inkluderas i modellen. Minska modellens storlek genom att ställa in `nocl` till 'True'.

Om du kör hela notebooken kommer du nu att bygga en Onnx-modell och spara den i den här mappen.

## Visa din modell

Onnx-modeller är inte särskilt synliga i Visual Studio Code, men det finns en mycket bra gratis programvara som många forskare använder för att visualisera modellen och säkerställa att den är korrekt byggd. Ladda ner [Netron](https://github.com/lutzroeder/Netron) och öppna din model.onnx-fil. Du kan se din enkla modell visualiserad, med dess 380 ingångar och klassificerare listade:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron är ett användbart verktyg för att visa dina modeller.

Nu är du redo att använda denna smarta modell i en webapp. Låt oss bygga en app som kommer till nytta när du tittar i ditt kylskåp och försöker lista ut vilken kombination av dina kvarvarande ingredienser du kan använda för att laga en viss maträtt, som bestäms av din modell.

## Bygg en webapplikation för rekommendationer

Du kan använda din modell direkt i en webapp. Denna arkitektur gör det också möjligt att köra den lokalt och till och med offline om det behövs. Börja med att skapa en `index.html`-fil i samma mapp där du sparade din `model.onnx`-fil.

1. I denna fil _index.html_, lägg till följande markup:

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

1. Nu, inom `body`-taggarna, lägg till lite markup för att visa en lista med kryssrutor som representerar några ingredienser:

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

    Observera att varje kryssruta har ett värde. Detta representerar indexet där ingrediensen finns enligt datasetet. Äpple, till exempel, i denna alfabetiska lista, upptar den femte kolumnen, så dess värde är '4' eftersom vi börjar räkna från 0. Du kan konsultera [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) för att upptäcka en viss ingrediens index.

    Fortsätt ditt arbete i index.html-filen och lägg till ett skriptblock där modellen anropas efter den sista stängande `</div>`.

1. Först, importera [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime används för att möjliggöra körning av dina Onnx-modeller över ett brett spektrum av hårdvaruplattformar, inklusive optimeringar och ett API att använda.

1. När Runtime är på plats kan du kalla på den:

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

I denna kod händer flera saker:

1. Du skapade en array med 380 möjliga värden (1 eller 0) som ska ställas in och skickas till modellen för inferens, beroende på om en ingrediens kryssruta är markerad.
2. Du skapade en array med kryssrutor och ett sätt att avgöra om de var markerade i en `init`-funktion som anropas när applikationen startar. När en kryssruta är markerad ändras `ingredients`-arrayen för att återspegla den valda ingrediensen.
3. Du skapade en `testCheckboxes`-funktion som kontrollerar om någon kryssruta var markerad.
4. Du använder `startInference`-funktionen när knappen trycks och, om någon kryssruta är markerad, startar du inferens.
5. Inferensrutinen inkluderar:
   1. Att sätta upp en asynkron laddning av modellen
   2. Skapa en Tensor-struktur att skicka till modellen
   3. Skapa 'feeds' som återspeglar `float_input`-ingången som du skapade när du tränade din modell (du kan använda Netron för att verifiera det namnet)
   4. Skicka dessa 'feeds' till modellen och vänta på ett svar

## Testa din applikation

Öppna en terminalsession i Visual Studio Code i mappen där din index.html-fil finns. Se till att du har [http-server](https://www.npmjs.com/package/http-server) installerat globalt och skriv `http-server` vid prompten. En localhost bör öppnas och du kan visa din webapp. Kontrollera vilken maträtt som rekommenderas baserat på olika ingredienser:

![ingredient web app](../../../../4-Classification/4-Applied/images/web-app.png)

Grattis, du har skapat en webapp för 'rekommendationer' med några fält. Ta lite tid att bygga ut detta system!

## 🚀Utmaning

Din webapp är väldigt minimal, så fortsätt att bygga ut den med ingredienser och deras index från [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv)-datan. Vilka smakkombinationer fungerar för att skapa en viss nationalrätt?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Även om denna lektion bara berörde nyttan av att skapa ett rekommendationssystem för matingredienser, är detta område av ML-applikationer mycket rikt på exempel. Läs mer om hur dessa system byggs:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Uppgift 

[Bygg en ny rekommendationsmotor](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör du vara medveten om att automatiserade översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.