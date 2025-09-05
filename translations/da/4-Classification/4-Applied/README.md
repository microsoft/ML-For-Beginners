<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T00:47:17+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "da"
}
-->
# Byg en webapp til anbefaling af køkkener

I denne lektion vil du bygge en klassifikationsmodel ved hjælp af nogle af de teknikker, du har lært i tidligere lektioner, og med det lækre køkkendatasæt, der er blevet brugt gennem hele denne serie. Derudover vil du bygge en lille webapp til at bruge en gemt model, der udnytter Onnx's web-runtime.

En af de mest praktiske anvendelser af maskinlæring er at bygge anbefalingssystemer, og du kan tage det første skridt i den retning i dag!

[![Præsentation af denne webapp](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klik på billedet ovenfor for en video: Jen Looper bygger en webapp ved hjælp af klassificerede køkkendata

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

I denne lektion vil du lære:

- Hvordan man bygger en model og gemmer den som en Onnx-model
- Hvordan man bruger Netron til at inspicere modellen
- Hvordan man bruger din model i en webapp til inferens

## Byg din model

At bygge anvendte ML-systemer er en vigtig del af at udnytte disse teknologier til dine forretningssystemer. Du kan bruge modeller i dine webapplikationer (og dermed bruge dem offline, hvis nødvendigt) ved hjælp af Onnx.

I en [tidligere lektion](../../3-Web-App/1-Web-App/README.md) byggede du en regressionsmodel om UFO-observationer, "picklede" den og brugte den i en Flask-app. Selvom denne arkitektur er meget nyttig at kende, er det en fuld-stack Python-app, og dine krav kan inkludere brugen af en JavaScript-applikation.

I denne lektion kan du bygge et grundlæggende JavaScript-baseret system til inferens. Først skal du dog træne en model og konvertere den til brug med Onnx.

## Øvelse - træning af klassifikationsmodel

Start med at træne en klassifikationsmodel ved hjælp af det rensede køkkendatasæt, vi har brugt.

1. Start med at importere nyttige biblioteker:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Du skal bruge '[skl2onnx](https://onnx.ai/sklearn-onnx/)' til at hjælpe med at konvertere din Scikit-learn-model til Onnx-format.

1. Arbejd derefter med dine data på samme måde som i tidligere lektioner ved at læse en CSV-fil med `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Fjern de første to unødvendige kolonner og gem de resterende data som 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Gem etiketterne som 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Start træningsrutinen

Vi vil bruge 'SVC'-biblioteket, som har god nøjagtighed.

1. Importer de relevante biblioteker fra Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Opdel trænings- og testdatasæt:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Byg en SVC-klassifikationsmodel, som du gjorde i den tidligere lektion:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Test nu din model ved at kalde `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Udskriv en klassifikationsrapport for at kontrollere modellens kvalitet:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Som vi så før, er nøjagtigheden god:

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

### Konverter din model til Onnx

Sørg for at udføre konverteringen med det korrekte antal tensorer. Dette datasæt har 380 ingredienser opført, så du skal angive det antal i `FloatTensorType`:

1. Konverter med et tensorantal på 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Opret onx-filen og gem den som **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Bemærk, at du kan angive [indstillinger](https://onnx.ai/sklearn-onnx/parameterized.html) i dit konverteringsscript. I dette tilfælde angav vi 'nocl' til at være True og 'zipmap' til at være False. Da dette er en klassifikationsmodel, har du mulighed for at fjerne ZipMap, som producerer en liste over ordbøger (ikke nødvendigt). `nocl` refererer til klasseinformation, der inkluderes i modellen. Reducer modellens størrelse ved at sætte `nocl` til 'True'.

Ved at køre hele notebooken vil du nu bygge en Onnx-model og gemme den i denne mappe.

## Se din model

Onnx-modeller er ikke særlig synlige i Visual Studio Code, men der er en meget god gratis software, som mange forskere bruger til at visualisere modellen for at sikre, at den er korrekt bygget. Download [Netron](https://github.com/lutzroeder/Netron) og åbn din model.onnx-fil. Du kan se din simple model visualiseret med dens 380 inputs og klassifikator angivet:

![Netron visualisering](../../../../4-Classification/4-Applied/images/netron.png)

Netron er et nyttigt værktøj til at se dine modeller.

Nu er du klar til at bruge denne smarte model i en webapp. Lad os bygge en app, der vil være nyttig, når du kigger i dit køleskab og prøver at finde ud af, hvilken kombination af dine resterende ingredienser du kan bruge til at lave en given ret, som bestemt af din model.

## Byg en anbefalingswebapplikation

Du kan bruge din model direkte i en webapp. Denne arkitektur giver dig også mulighed for at køre den lokalt og endda offline, hvis nødvendigt. Start med at oprette en `index.html`-fil i den samme mappe, hvor du gemte din `model.onnx`-fil.

1. I denne fil _index.html_, tilføj følgende markup:

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

1. Arbejd nu inden for `body`-tags og tilføj lidt markup for at vise en liste med afkrydsningsfelter, der afspejler nogle ingredienser:

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

    Bemærk, at hvert afkrydsningsfelt har en værdi. Dette afspejler den indeks, hvor ingrediensen findes i henhold til datasættet. Æble, for eksempel, i denne alfabetiske liste, optager den femte kolonne, så dens værdi er '4', da vi starter med at tælle fra 0. Du kan konsultere [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) for at finde en given ingrediens' indeks.

    Fortsæt dit arbejde i index.html-filen, og tilføj et script-blok, hvor modellen kaldes efter den sidste lukkende `</div>`.

1. Først, importer [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime bruges til at muliggøre kørsel af dine Onnx-modeller på tværs af en bred vifte af hardwareplatforme, inklusive optimeringer og en API til brug.

1. Når Runtime er på plads, kan du kalde det:

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

I denne kode sker der flere ting:

1. Du oprettede et array med 380 mulige værdier (1 eller 0), der skal indstilles og sendes til modellen til inferens, afhængigt af om et afkrydsningsfelt er markeret.
2. Du oprettede et array med afkrydsningsfelter og en måde at afgøre, om de blev markeret i en `init`-funktion, der kaldes, når applikationen starter. Når et afkrydsningsfelt er markeret, ændres `ingredients`-arrayet for at afspejle den valgte ingrediens.
3. Du oprettede en `testCheckboxes`-funktion, der kontrollerer, om nogen afkrydsningsfelter blev markeret.
4. Du bruger `startInference`-funktionen, når knappen trykkes, og hvis nogen afkrydsningsfelter er markeret, starter du inferens.
5. Inferensrutinen inkluderer:
   1. Opsætning af en asynkron indlæsning af modellen
   2. Oprettelse af en Tensor-struktur til at sende til modellen
   3. Oprettelse af 'feeds', der afspejler `float_input`-input, som du oprettede, da du trænede din model (du kan bruge Netron til at verificere det navn)
   4. Afsendelse af disse 'feeds' til modellen og venten på et svar

## Test din applikation

Åbn en terminalsession i Visual Studio Code i den mappe, hvor din index.html-fil ligger. Sørg for, at du har [http-server](https://www.npmjs.com/package/http-server) installeret globalt, og skriv `http-server` ved prompten. En localhost bør åbne, og du kan se din webapp. Tjek, hvilket køkken der anbefales baseret på forskellige ingredienser:

![webapp med ingredienser](../../../../4-Classification/4-Applied/images/web-app.png)

Tillykke, du har oprettet en 'anbefalings'-webapp med nogle få felter. Tag dig tid til at udbygge dette system!

## 🚀Udfordring

Din webapp er meget minimal, så fortsæt med at udbygge den ved hjælp af ingredienser og deres indekser fra [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv)-data. Hvilke smagskombinationer fungerer for at skabe en given national ret?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Selvom denne lektion kun berørte nytten af at skabe et anbefalingssystem for madingredienser, er dette område af ML-applikationer meget rigt på eksempler. Læs mere om, hvordan disse systemer bygges:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Opgave 

[Byg en ny anbefaler](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os intet ansvar for misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.