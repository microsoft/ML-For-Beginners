<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T13:11:19+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "sl"
}
-->
# Zgradite spletno aplikacijo za priporočanje kulinarike

V tej lekciji boste zgradili model za klasifikacijo z uporabo nekaterih tehnik, ki ste se jih naučili v prejšnjih lekcijah, ter z okusnim kulinaričnim naborom podatkov, ki smo ga uporabljali skozi celotno serijo. Poleg tega boste zgradili majhno spletno aplikacijo, ki bo uporabljala shranjen model, pri čemer boste izkoristili Onnx-ov spletni runtime.

Ena najbolj uporabnih praktičnih aplikacij strojnega učenja je gradnja sistemov za priporočanje, in danes lahko naredite prvi korak v tej smeri!

[![Predstavitev te spletne aplikacije](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Kliknite zgornjo sliko za video: Jen Looper gradi spletno aplikacijo z uporabo klasificiranih kulinaričnih podatkov

## [Predlekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

V tej lekciji se boste naučili:

- Kako zgraditi model in ga shraniti kot Onnx model
- Kako uporabiti Netron za pregled modela
- Kako uporabiti vaš model v spletni aplikaciji za sklepanje

## Zgradite svoj model

Gradnja aplikativnih sistemov strojnega učenja je pomemben del uporabe teh tehnologij v poslovnih sistemih. Modele lahko uporabite znotraj svojih spletnih aplikacij (in jih tako po potrebi uporabite v offline kontekstu) z uporabo Onnx-a.

V [prejšnji lekciji](../../3-Web-App/1-Web-App/README.md) ste zgradili regresijski model o opažanjih NLP-jev, ga "pickle-ali" in uporabili v Flask aplikaciji. Čeprav je ta arhitektura zelo uporabna, gre za polno Python aplikacijo, vaše zahteve pa lahko vključujejo uporabo JavaScript aplikacije.

V tej lekciji lahko zgradite osnovni sistem za sklepanje, ki temelji na JavaScript-u. Najprej pa morate trenirati model in ga pretvoriti za uporabo z Onnx-om.

## Naloga - trenirajte klasifikacijski model

Najprej trenirajte klasifikacijski model z uporabo očiščenega nabora podatkov o kulinariki, ki smo ga uporabljali.

1. Začnite z uvozom uporabnih knjižnic:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Potrebujete '[skl2onnx](https://onnx.ai/sklearn-onnx/)', da pomagate pretvoriti vaš Scikit-learn model v Onnx format.

1. Nato obdelajte podatke na enak način kot v prejšnjih lekcijah, tako da preberete CSV datoteko z uporabo `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Odstranite prvi dve nepotrebni stolpci in shranite preostale podatke kot 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Shranite oznake kot 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Začnite rutino treniranja

Uporabili bomo knjižnico 'SVC', ki ima dobro natančnost.

1. Uvozite ustrezne knjižnice iz Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Ločite podatke na učne in testne sklope:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Zgradite klasifikacijski model SVC, kot ste to storili v prejšnji lekciji:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Zdaj preizkusite svoj model z uporabo `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Izpišite poročilo o klasifikaciji, da preverite kakovost modela:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Kot smo videli prej, je natančnost dobra:

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

### Pretvorite svoj model v Onnx

Poskrbite, da bo pretvorba izvedena z ustreznim številom tenzorjev. Ta nabor podatkov ima 380 sestavin, zato morate to število označiti v `FloatTensorType`:

1. Pretvorite z uporabo števila tenzorjev 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Ustvarite onx in shranite kot datoteko **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Opomba: V vašem skriptu za pretvorbo lahko podate [možnosti](https://onnx.ai/sklearn-onnx/parameterized.html). V tem primeru smo nastavili 'nocl' na True in 'zipmap' na False. Ker gre za klasifikacijski model, imate možnost odstraniti ZipMap, ki ustvari seznam slovarjev (ni potrebno). `nocl` se nanaša na vključitev informacij o razredih v model. Zmanjšajte velikost modela tako, da nastavite `nocl` na 'True'.

Zagon celotnega zvezka bo zdaj zgradil Onnx model in ga shranil v to mapo.

## Preglejte svoj model

Onnx modeli niso zelo vidni v Visual Studio Code, vendar obstaja zelo dobra brezplačna programska oprema, ki jo mnogi raziskovalci uporabljajo za vizualizacijo modela, da se prepričajo, da je pravilno zgrajen. Prenesite [Netron](https://github.com/lutzroeder/Netron) in odprite datoteko model.onnx. Videli boste vizualizacijo vašega preprostega modela, z njegovimi 380 vhodi in klasifikatorjem:

![Netron vizualizacija](../../../../4-Classification/4-Applied/images/netron.png)

Netron je uporabno orodje za pregled vaših modelov.

Zdaj ste pripravljeni uporabiti ta zanimiv model v spletni aplikaciji. Zgradimo aplikacijo, ki bo uporabna, ko boste pogledali v svoj hladilnik in poskušali ugotoviti, katero kombinacijo preostalih sestavin lahko uporabite za pripravo določene kulinarike, kot jo določi vaš model.

## Zgradite spletno aplikacijo za priporočanje

Svoj model lahko uporabite neposredno v spletni aplikaciji. Ta arhitektura omogoča tudi lokalno delovanje in celo offline uporabo, če je potrebno. Začnite z ustvarjanjem datoteke `index.html` v isti mapi, kjer ste shranili svojo datoteko `model.onnx`.

1. V tej datoteki _index.html_ dodajte naslednjo oznako:

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

1. Zdaj, znotraj oznak `body`, dodajte nekaj oznak za prikaz seznama potrditvenih polj, ki odražajo nekatere sestavine:

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

    Opazite, da je vsakemu potrditvenemu polju dodeljena vrednost. To odraža indeks, kjer je sestavina najdena glede na nabor podatkov. Jabolko, na primer, v tem abecednem seznamu zaseda peti stolpec, zato je njegova vrednost '4', saj začnemo šteti pri 0. Posvetujte se s [preglednico sestavin](../../../../4-Classification/data/ingredient_indexes.csv), da odkrijete indeks določene sestavine.

    Nadaljujte delo v datoteki index.html in dodajte blok skripta, kjer je model poklican po zadnjem zapiralnem `</div>`.

1. Najprej uvozite [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime omogoča izvajanje vaših Onnx modelov na širokem spektru strojne opreme, vključno z optimizacijami in API-jem za uporabo.

1. Ko je Runtime na mestu, ga lahko pokličete:

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

V tej kodi se dogaja več stvari:

1. Ustvarili ste polje 380 možnih vrednosti (1 ali 0), ki se nastavijo in pošljejo modelu za sklepanje, odvisno od tega, ali je potrditveno polje označeno.
2. Ustvarili ste polje potrditvenih polj in način za določanje, ali so označena, v funkciji `init`, ki se pokliče ob zagonu aplikacije. Ko je potrditveno polje označeno, se polje `ingredients` spremeni, da odraža izbrano sestavino.
3. Ustvarili ste funkcijo `testCheckboxes`, ki preverja, ali je bilo katero potrditveno polje označeno.
4. Uporabite funkcijo `startInference`, ko je gumb pritisnjen, in če je katero potrditveno polje označeno, začnete sklepanje.
5. Rutina sklepanja vključuje:
   1. Nastavitev asinhronega nalaganja modela
   2. Ustvarjanje strukture Tensor za pošiljanje modelu
   3. Ustvarjanje 'feeds', ki odražajo vhod `float_input`, ki ste ga ustvarili med treniranjem modela (lahko uporabite Netron za preverjanje tega imena)
   4. Pošiljanje teh 'feeds' modelu in čakanje na odgovor

## Preizkusite svojo aplikacijo

Odprite terminal v Visual Studio Code v mapi, kjer se nahaja vaša datoteka index.html. Poskrbite, da imate [http-server](https://www.npmjs.com/package/http-server) globalno nameščen, in vnesite `http-server` v ukazni vrstici. Odpre se localhost, kjer lahko vidite svojo spletno aplikacijo. Preverite, katera kulinarika je priporočena glede na različne sestavine:

![spletna aplikacija za sestavine](../../../../4-Classification/4-Applied/images/web-app.png)

Čestitamo, ustvarili ste spletno aplikacijo za 'priporočanje' z nekaj polji. Vzemite si čas za nadgradnjo tega sistema!

## 🚀Izziv

Vaša spletna aplikacija je zelo osnovna, zato jo nadaljujte z nadgradnjo z uporabo sestavin in njihovih indeksov iz podatkov [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Katere kombinacije okusov delujejo za pripravo določene nacionalne jedi?

## [Po-lekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Čeprav je ta lekcija le na kratko obravnavala uporabnost ustvarjanja sistema za priporočanje sestavin, je to področje aplikacij strojnega učenja zelo bogato z zgledi. Preberite več o tem, kako so ti sistemi zgrajeni:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Naloga 

[Zgradite nov sistem za priporočanje](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.