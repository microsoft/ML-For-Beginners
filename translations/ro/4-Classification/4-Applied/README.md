<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T16:22:03+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "ro"
}
-->
# Construiește o aplicație web pentru recomandarea bucătăriilor

În această lecție, vei construi un model de clasificare folosind unele dintre tehnicile pe care le-ai învățat în lecțiile anterioare și dataset-ul delicios de bucătării utilizat pe parcursul acestei serii. În plus, vei construi o mică aplicație web pentru a utiliza un model salvat, folosind runtime-ul web al Onnx.

Una dintre cele mai utile aplicații practice ale învățării automate este construirea sistemelor de recomandare, iar astăzi poți face primul pas în această direcție!

[![Prezentarea acestei aplicații web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "ML aplicat")

> 🎥 Fă clic pe imaginea de mai sus pentru un videoclip: Jen Looper construiește o aplicație web folosind date clasificate despre bucătării

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

În această lecție vei învăța:

- Cum să construiești un model și să-l salvezi ca model Onnx
- Cum să folosești Netron pentru a inspecta modelul
- Cum să utilizezi modelul într-o aplicație web pentru inferență

## Construiește modelul tău

Construirea sistemelor ML aplicate este o parte importantă a valorificării acestor tehnologii pentru sistemele tale de afaceri. Poți utiliza modelele în aplicațiile tale web (și astfel să le folosești într-un context offline, dacă este necesar) folosind Onnx.

Într-o [lecție anterioară](../../3-Web-App/1-Web-App/README.md), ai construit un model de regresie despre observațiile OZN, l-ai "pickled" și l-ai utilizat într-o aplicație Flask. Deși această arhitectură este foarte utilă de cunoscut, este o aplicație Python full-stack, iar cerințele tale pot include utilizarea unei aplicații JavaScript.

În această lecție, poți construi un sistem de bază bazat pe JavaScript pentru inferență. Mai întâi, însă, trebuie să antrenezi un model și să-l convertești pentru utilizare cu Onnx.

## Exercițiu - antrenează modelul de clasificare

Mai întâi, antrenează un model de clasificare folosind dataset-ul curățat de bucătării pe care l-am utilizat.

1. Începe prin importarea bibliotecilor utile:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Ai nevoie de '[skl2onnx](https://onnx.ai/sklearn-onnx/)' pentru a ajuta la conversia modelului Scikit-learn în format Onnx.

1. Apoi, lucrează cu datele în același mod ca în lecțiile anterioare, citind un fișier CSV folosind `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Elimină primele două coloane inutile și salvează datele rămase ca 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Salvează etichetele ca 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Începe rutina de antrenare

Vom folosi biblioteca 'SVC', care are o acuratețe bună.

1. Importă bibliotecile corespunzătoare din Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separă seturile de antrenare și testare:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Construiește un model de clasificare SVC, așa cum ai făcut în lecția anterioară:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Acum, testează modelul, apelând `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Afișează un raport de clasificare pentru a verifica calitatea modelului:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Așa cum am văzut anterior, acuratețea este bună:

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

### Convertește modelul tău în Onnx

Asigură-te că faci conversia cu numărul corect de tensori. Acest dataset are 380 de ingrediente listate, așa că trebuie să notezi acest număr în `FloatTensorType`:

1. Convertește folosind un număr de tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Creează fișierul onx și salvează-l ca **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Notă: poți trece [opțiuni](https://onnx.ai/sklearn-onnx/parameterized.html) în scriptul de conversie. În acest caz, am setat 'nocl' să fie True și 'zipmap' să fie False. Deoarece acesta este un model de clasificare, ai opțiunea de a elimina ZipMap, care produce o listă de dicționare (nu este necesar). `nocl` se referă la includerea informațiilor despre clasă în model. Redu dimensiunea modelului setând `nocl` la 'True'.

Rularea întregului notebook va construi acum un model Onnx și îl va salva în acest folder.

## Vizualizează modelul tău

Modelele Onnx nu sunt foarte vizibile în Visual Studio Code, dar există un software gratuit foarte bun pe care mulți cercetători îl folosesc pentru a vizualiza modelul și a se asigura că este construit corect. Descarcă [Netron](https://github.com/lutzroeder/Netron) și deschide fișierul model.onnx. Poți vedea modelul tău simplu vizualizat, cu cele 380 de intrări și clasificatorul listat:

![Vizualizare Netron](../../../../4-Classification/4-Applied/images/netron.png)

Netron este un instrument util pentru vizualizarea modelelor tale.

Acum ești pregătit să folosești acest model interesant într-o aplicație web. Hai să construim o aplicație care va fi utilă atunci când te uiți în frigider și încerci să-ți dai seama ce combinație de ingrediente rămase poți folosi pentru a găti o anumită bucătărie, determinată de modelul tău.

## Construiește o aplicație web de recomandare

Poți utiliza modelul tău direct într-o aplicație web. Această arhitectură îți permite, de asemenea, să o rulezi local și chiar offline, dacă este necesar. Începe prin crearea unui fișier `index.html` în același folder unde ai salvat fișierul `model.onnx`.

1. În acest fișier _index.html_, adaugă următorul markup:

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

1. Acum, lucrând în interiorul tag-urilor `body`, adaugă puțin markup pentru a afișa o listă de checkbox-uri care reflectă unele ingrediente:

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

    Observă că fiecare checkbox are o valoare. Aceasta reflectă indexul unde se află ingredientul conform dataset-ului. De exemplu, mărul, în această listă alfabetică, ocupă a cincea coloană, deci valoarea sa este '4', deoarece începem să numărăm de la 0. Poți consulta [fișierul de ingrediente](../../../../4-Classification/data/ingredient_indexes.csv) pentru a descoperi indexul unui ingredient dat.

    Continuând lucrul în fișierul index.html, adaugă un bloc de script unde modelul este apelat după închiderea finală a `</div>`.

1. Mai întâi, importă [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime este utilizat pentru a permite rularea modelelor Onnx pe o gamă largă de platforme hardware, incluzând optimizări și un API pentru utilizare.

1. Odată ce Runtime-ul este în loc, îl poți apela:

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

În acest cod, se întâmplă mai multe lucruri:

1. Ai creat un array de 380 de valori posibile (1 sau 0) care să fie setate și trimise modelului pentru inferență, în funcție de faptul că un checkbox de ingredient este bifat.
2. Ai creat un array de checkbox-uri și o modalitate de a determina dacă acestea au fost bifate într-o funcție `init` care este apelată atunci când aplicația pornește. Când un checkbox este bifat, array-ul `ingredients` este modificat pentru a reflecta ingredientul ales.
3. Ai creat o funcție `testCheckboxes` care verifică dacă vreun checkbox a fost bifat.
4. Folosești funcția `startInference` atunci când butonul este apăsat și, dacă vreun checkbox este bifat, începi inferența.
5. Rutina de inferență include:
   1. Configurarea unei încărcări asincrone a modelului
   2. Crearea unei structuri Tensor pentru a fi trimisă modelului
   3. Crearea 'feeds' care reflectă intrarea `float_input` pe care ai creat-o când ai antrenat modelul (poți folosi Netron pentru a verifica acest nume)
   4. Trimiterea acestor 'feeds' către model și așteptarea unui răspuns

## Testează aplicația ta

Deschide o sesiune de terminal în Visual Studio Code în folderul unde se află fișierul index.html. Asigură-te că ai [http-server](https://www.npmjs.com/package/http-server) instalat global și tastează `http-server` la prompt. Ar trebui să se deschidă un localhost și poți vizualiza aplicația ta web. Verifică ce bucătărie este recomandată pe baza diferitelor ingrediente:

![Aplicație web ingrediente](../../../../4-Classification/4-Applied/images/web-app.png)

Felicitări, ai creat o aplicație web de 'recomandare' cu câmpuri minime. Ia-ți timp să dezvolți acest sistem!

## 🚀Provocare

Aplicația ta web este foarte minimală, așa că continuă să o dezvolți folosind ingrediente și indexurile lor din datele [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Ce combinații de arome funcționează pentru a crea un fel de mâncare național?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Deși această lecție doar a atins utilitatea creării unui sistem de recomandare pentru ingrediente alimentare, acest domeniu al aplicațiilor ML este foarte bogat în exemple. Citește mai multe despre cum sunt construite aceste sisteme:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Temă

[Construiește un nou sistem de recomandare](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.