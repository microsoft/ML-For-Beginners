<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T13:10:42+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "hr"
}
-->
# Izgradnja web aplikacije za preporuku kuhinja

U ovoj lekciji izgradit ćete model klasifikacije koristeći neke od tehnika koje ste naučili u prethodnim lekcijama, uz ukusni dataset kuhinja koji se koristi kroz cijeli ovaj serijal. Osim toga, izradit ćete malu web aplikaciju za korištenje spremljenog modela, koristeći Onnx web runtime.

Jedna od najkorisnijih praktičnih primjena strojnog učenja je izrada sustava za preporuke, a danas možete napraviti prvi korak u tom smjeru!

[![Predstavljanje ove web aplikacije](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Primijenjeno ML")

> 🎥 Kliknite na sliku iznad za video: Jen Looper izrađuje web aplikaciju koristeći klasificirane podatke o kuhinjama

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

U ovoj lekciji naučit ćete:

- Kako izraditi model i spremiti ga kao Onnx model
- Kako koristiti Netron za pregled modela
- Kako koristiti svoj model u web aplikaciji za inferenciju

## Izradite svoj model

Izrada primijenjenih ML sustava važan je dio korištenja ovih tehnologija za poslovne sustave. Možete koristiti modele unutar svojih web aplikacija (i tako ih koristiti u offline kontekstu ako je potrebno) koristeći Onnx.

U [prethodnoj lekciji](../../3-Web-App/1-Web-App/README.md) izradili ste regresijski model o viđenjima NLO-a, "pickle-ali" ga i koristili u Flask aplikaciji. Iako je ova arhitektura vrlo korisna za znati, radi se o full-stack Python aplikaciji, a vaši zahtjevi mogu uključivati korištenje JavaScript aplikacije.

U ovoj lekciji možete izraditi osnovni sustav temeljen na JavaScriptu za inferenciju. No, prvo morate trenirati model i konvertirati ga za korištenje s Onnx-om.

## Vježba - treniranje modela klasifikacije

Prvo, trenirajte model klasifikacije koristeći očišćeni dataset kuhinja koji smo koristili.

1. Započnite uvozom korisnih biblioteka:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Trebat će vam '[skl2onnx](https://onnx.ai/sklearn-onnx/)' za pomoć pri konverziji vašeg Scikit-learn modela u Onnx format.

1. Zatim radite s podacima na isti način kao u prethodnim lekcijama, čitajući CSV datoteku koristeći `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Uklonite prva dva nepotrebna stupca i spremite preostale podatke kao 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Spremite oznake kao 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Započnite rutinu treniranja

Koristit ćemo biblioteku 'SVC' koja ima dobru točnost.

1. Uvezite odgovarajuće biblioteke iz Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Odvojite skupove za treniranje i testiranje:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Izradite SVC model klasifikacije kao što ste to učinili u prethodnoj lekciji:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Sada testirajte svoj model pozivajući `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Ispišite izvještaj o klasifikaciji kako biste provjerili kvalitetu modela:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Kao što smo vidjeli prije, točnost je dobra:

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

### Konvertirajte svoj model u Onnx

Pobrinite se da konverziju obavite s odgovarajućim brojem tenzora. Ovaj dataset ima 380 navedenih sastojaka, pa morate zabilježiti taj broj u `FloatTensorType`:

1. Konvertirajte koristeći broj tenzora 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Kreirajte onx i spremite kao datoteku **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Napomena: možete proslijediti [opcije](https://onnx.ai/sklearn-onnx/parameterized.html) u svom skriptu za konverziju. U ovom slučaju, postavili smo 'nocl' na True i 'zipmap' na False. Budući da je ovo model klasifikacije, imate opciju ukloniti ZipMap koji proizvodi popis rječnika (nije potrebno). `nocl` se odnosi na uključivanje informacija o klasama u model. Smanjite veličinu svog modela postavljanjem `nocl` na 'True'.

Pokretanjem cijele bilježnice sada ćete izraditi Onnx model i spremiti ga u ovu mapu.

## Pregledajte svoj model

Onnx modeli nisu baš vidljivi u Visual Studio Code-u, ali postoji vrlo dobar besplatan softver koji mnogi istraživači koriste za vizualizaciju modela kako bi se osigurali da je pravilno izrađen. Preuzmite [Netron](https://github.com/lutzroeder/Netron) i otvorite svoju model.onnx datoteku. Možete vidjeti svoj jednostavni model vizualiziran, s njegovih 380 ulaza i klasifikatorom:

![Netron vizualizacija](../../../../4-Classification/4-Applied/images/netron.png)

Netron je koristan alat za pregled vaših modela.

Sada ste spremni koristiti ovaj zgodni model u web aplikaciji. Izradimo aplikaciju koja će biti korisna kada pogledate u svoj hladnjak i pokušate odrediti koju kombinaciju svojih preostalih sastojaka možete koristiti za pripremu određene kuhinje, prema vašem modelu.

## Izradite web aplikaciju za preporuke

Možete koristiti svoj model izravno u web aplikaciji. Ova arhitektura također omogućuje lokalno pokretanje, pa čak i offline ako je potrebno. Započnite stvaranjem datoteke `index.html` u istoj mapi gdje ste spremili svoju datoteku `model.onnx`.

1. U ovoj datoteci _index.html_, dodajte sljedeći markup:

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

1. Sada, radeći unutar oznaka `body`, dodajte malo markupa za prikaz popisa checkboxova koji odražavaju neke sastojke:

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

    Primijetite da je svakom checkboxu dodijeljena vrijednost. To odražava indeks gdje se sastojak nalazi prema datasetu. Jabuka, na primjer, u ovom abecednom popisu zauzima peti stupac, pa je njezina vrijednost '4' jer počinjemo brojati od 0. Možete konzultirati [tablicu sastojaka](../../../../4-Classification/data/ingredient_indexes.csv) kako biste otkrili indeks određenog sastojka.

    Nastavljajući rad u datoteci index.html, dodajte blok skripte gdje se model poziva nakon završnog zatvaranja `</div>`.

1. Prvo, uvezite [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime se koristi za omogućavanje pokretanja vaših Onnx modela na širokom rasponu hardverskih platformi, uključujući optimizacije i API za korištenje.

1. Kada je Runtime na mjestu, možete ga pozvati:

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

U ovom kodu događa se nekoliko stvari:

1. Kreirali ste niz od 380 mogućih vrijednosti (1 ili 0) koje se postavljaju i šalju modelu za inferenciju, ovisno o tome je li checkbox označen.
2. Kreirali ste niz checkboxova i način za određivanje jesu li označeni u funkciji `init` koja se poziva kada aplikacija započne. Kada je checkbox označen, niz `ingredients` se mijenja kako bi odražavao odabrani sastojak.
3. Kreirali ste funkciju `testCheckboxes` koja provjerava je li neki checkbox označen.
4. Koristite funkciju `startInference` kada se pritisne gumb i, ako je neki checkbox označen, započinjete inferenciju.
5. Rutina inferencije uključuje:
   1. Postavljanje asinkronog učitavanja modela
   2. Kreiranje Tensor strukture za slanje modelu
   3. Kreiranje 'feeds' koji odražava `float_input` ulaz koji ste kreirali prilikom treniranja modela (možete koristiti Netron za provjeru tog naziva)
   4. Slanje ovih 'feeds' modelu i čekanje odgovora

## Testirajte svoju aplikaciju

Otvorite terminal sesiju u Visual Studio Code-u u mapi gdje se nalazi vaša datoteka index.html. Osigurajte da imate [http-server](https://www.npmjs.com/package/http-server) instaliran globalno i upišite `http-server` na promptu. Trebao bi se otvoriti localhost i možete pregledati svoju web aplikaciju. Provjerite koja kuhinja se preporučuje na temelju različitih sastojaka:

![web aplikacija sa sastojcima](../../../../4-Classification/4-Applied/images/web-app.png)

Čestitamo, izradili ste web aplikaciju za 'preporuke' s nekoliko polja. Odvojite malo vremena za proširenje ovog sustava!

## 🚀Izazov

Vaša web aplikacija je vrlo minimalna, pa je nastavite proširivati koristeći sastojke i njihove indekse iz podataka [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Koje kombinacije okusa funkcioniraju za stvaranje određenog nacionalnog jela?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Iako je ova lekcija samo dotaknula korisnost stvaranja sustava za preporuke za sastojke hrane, ovo područje primjena strojnog učenja vrlo je bogato primjerima. Pročitajte više o tome kako se ovi sustavi izrađuju:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Zadatak 

[Izradite novi sustav za preporuke](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prijevod [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.