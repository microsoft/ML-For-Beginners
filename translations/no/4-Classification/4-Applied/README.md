<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T21:52:38+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "no"
}
-->
# Bygg en webapp for matanbefalinger

I denne leksjonen skal du bygge en klassifiseringsmodell ved hjelp av noen av teknikkene du har lært i tidligere leksjoner, samt det deilige matdatasettet som har blitt brukt gjennom denne serien. I tillegg skal du lage en liten webapp for å bruke en lagret modell, ved hjelp av Onnx sin web-runtime.

En av de mest nyttige praktiske bruksområdene for maskinlæring er å lage anbefalingssystemer, og du kan ta det første steget i den retningen i dag!

[![Presentere denne webappen](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Klikk på bildet over for en video: Jen Looper lager en webapp ved hjelp av klassifisert matdata

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

I denne leksjonen vil du lære:

- Hvordan bygge en modell og lagre den som en Onnx-modell
- Hvordan bruke Netron for å inspisere modellen
- Hvordan bruke modellen din i en webapp for inferens

## Bygg modellen din

Å bygge anvendte ML-systemer er en viktig del av å utnytte disse teknologiene for dine forretningssystemer. Du kan bruke modeller i webapplikasjonene dine (og dermed bruke dem i en offline-kontekst om nødvendig) ved hjelp av Onnx.

I en [tidligere leksjon](../../3-Web-App/1-Web-App/README.md) bygde du en regresjonsmodell om UFO-observasjoner, "picklet" den, og brukte den i en Flask-app. Selv om denne arkitekturen er veldig nyttig å kjenne til, er det en full-stack Python-app, og kravene dine kan inkludere bruk av en JavaScript-applikasjon.

I denne leksjonen kan du bygge et grunnleggende JavaScript-basert system for inferens. Først må du imidlertid trene en modell og konvertere den for bruk med Onnx.

## Øvelse - tren klassifiseringsmodell

Først, tren en klassifiseringsmodell ved hjelp av det rensede matdatasettet vi brukte.

1. Start med å importere nyttige biblioteker:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Du trenger '[skl2onnx](https://onnx.ai/sklearn-onnx/)' for å hjelpe med å konvertere Scikit-learn-modellen din til Onnx-format.

1. Deretter, arbeid med dataene dine på samme måte som du gjorde i tidligere leksjoner, ved å lese en CSV-fil med `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Fjern de to første unødvendige kolonnene og lagre de gjenværende dataene som 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Lagre etikettene som 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Start treningsrutinen

Vi skal bruke 'SVC'-biblioteket som har god nøyaktighet.

1. Importer de relevante bibliotekene fra Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Del opp trenings- og testsett:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Bygg en SVC-klassifiseringsmodell slik du gjorde i den forrige leksjonen:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Test modellen din ved å kalle `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Skriv ut en klassifiseringsrapport for å sjekke modellens kvalitet:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Som vi så tidligere, er nøyaktigheten god:

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

### Konverter modellen din til Onnx

Sørg for å gjøre konverteringen med riktig Tensor-nummer. Dette datasettet har 380 ingredienser oppført, så du må notere det nummeret i `FloatTensorType`:

1. Konverter med et tensor-nummer på 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Lag onx-filen og lagre den som **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Merk, du kan sende inn [alternativer](https://onnx.ai/sklearn-onnx/parameterized.html) i konverteringsskriptet ditt. I dette tilfellet sendte vi inn 'nocl' som True og 'zipmap' som False. Siden dette er en klassifiseringsmodell, har du muligheten til å fjerne ZipMap som produserer en liste med ordbøker (ikke nødvendig). `nocl` refererer til klasseinformasjon som inkluderes i modellen. Reduser modellens størrelse ved å sette `nocl` til 'True'.

Hvis du kjører hele notebooken, vil du nå bygge en Onnx-modell og lagre den i denne mappen.

## Vis modellen din

Onnx-modeller er ikke veldig synlige i Visual Studio Code, men det finnes en veldig god gratis programvare som mange forskere bruker for å visualisere modellen og sikre at den er riktig bygget. Last ned [Netron](https://github.com/lutzroeder/Netron) og åpne model.onnx-filen din. Du kan se den enkle modellen visualisert, med sine 380 input og klassifiserer listet opp:

![Netron visual](../../../../4-Classification/4-Applied/images/netron.png)

Netron er et nyttig verktøy for å se modellene dine.

Nå er du klar til å bruke denne flotte modellen i en webapp. La oss bygge en app som vil være nyttig når du ser i kjøleskapet ditt og prøver å finne ut hvilken kombinasjon av ingredienser du kan bruke for å lage en bestemt matrett, som bestemt av modellen din.

## Bygg en anbefalingswebapplikasjon

Du kan bruke modellen din direkte i en webapp. Denne arkitekturen lar deg også kjøre den lokalt og til og med offline om nødvendig. Start med å lage en `index.html`-fil i samme mappe der du lagret `model.onnx`-filen.

1. I denne filen _index.html_, legg til følgende markup:

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

1. Nå, arbeid innenfor `body`-taggene, og legg til litt markup for å vise en liste med avkrysningsbokser som reflekterer noen ingredienser:

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

    Legg merke til at hver avkrysningsboks har en verdi. Dette reflekterer indeksen der ingrediensen finnes i henhold til datasettet. Eple, for eksempel, i denne alfabetiske listen, opptar den femte kolonnen, så verdien er '4' siden vi begynner å telle fra 0. Du kan konsultere [ingredients spreadsheet](../../../../4-Classification/data/ingredient_indexes.csv) for å finne en gitt ingrediens sin indeks.

    Fortsett arbeidet ditt i index.html-filen, og legg til en script-blokk der modellen kalles etter den siste lukkende `</div>`.

1. Først, importer [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime brukes for å muliggjøre kjøring av Onnx-modeller på tvers av et bredt spekter av maskinvareplattformer, inkludert optimaliseringer og en API for bruk.

1. Når Runtime er på plass, kan du kalle den:

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

I denne koden skjer det flere ting:

1. Du opprettet et array med 380 mulige verdier (1 eller 0) som skal settes og sendes til modellen for inferens, avhengig av om en ingrediens avkrysningsboks er krysset av.
2. Du opprettet et array med avkrysningsbokser og en måte å avgjøre om de var krysset av i en `init`-funksjon som kalles når applikasjonen starter. Når en avkrysningsboks er krysset av, endres `ingredients`-arrayet for å reflektere den valgte ingrediensen.
3. Du opprettet en `testCheckboxes`-funksjon som sjekker om noen avkrysningsboks er krysset av.
4. Du bruker `startInference`-funksjonen når knappen trykkes, og hvis noen avkrysningsboks er krysset av, starter du inferens.
5. Inferensrutinen inkluderer:
   1. Oppsett av en asynkron last av modellen
   2. Opprettelse av en Tensor-struktur som skal sendes til modellen
   3. Opprettelse av 'feeds' som reflekterer `float_input`-input som du opprettet da du trente modellen din (du kan bruke Netron for å verifisere det navnet)
   4. Sending av disse 'feeds' til modellen og venter på et svar

## Test applikasjonen din

Åpne en terminalsesjon i Visual Studio Code i mappen der index.html-filen din ligger. Sørg for at du har [http-server](https://www.npmjs.com/package/http-server) installert globalt, og skriv `http-server` ved ledeteksten. En localhost skal åpne seg, og du kan se webappen din. Sjekk hvilken matrett som anbefales basert på ulike ingredienser:

![ingredient web app](../../../../4-Classification/4-Applied/images/web-app.png)

Gratulerer, du har laget en 'anbefalings'-webapp med noen få felt. Ta deg tid til å bygge ut dette systemet!

## 🚀Utfordring

Webappen din er veldig enkel, så fortsett å bygge den ut ved hjelp av ingredienser og deres indekser fra [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv)-data. Hvilke smakskombinasjoner fungerer for å lage en gitt nasjonalrett?

## [Quiz etter leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Selv om denne leksjonen bare berørte nytten av å lage et anbefalingssystem for matingredienser, er dette området innen ML-applikasjoner veldig rikt på eksempler. Les mer om hvordan disse systemene bygges:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Oppgave

[Bygg en ny anbefaler](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.