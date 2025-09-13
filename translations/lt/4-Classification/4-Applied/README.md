<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T07:59:28+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "lt"
}
-->
# Sukurkite virtuvės rekomendacijų žiniatinklio programą

Šioje pamokoje sukursite klasifikavimo modelį, naudodami kai kurias technikas, kurias išmokote ankstesnėse pamokose, ir skanų virtuvės duomenų rinkinį, naudotą visoje šioje serijoje. Be to, sukursite nedidelę žiniatinklio programą, kuri naudos išsaugotą modelį, pasitelkdama Onnx žiniatinklio vykdymo aplinką.

Vienas iš naudingiausių praktinių mašininio mokymosi pritaikymų yra rekomendacijų sistemų kūrimas, ir šiandien galite žengti pirmąjį žingsnį šia kryptimi!

[![Šios žiniatinklio programos pristatymas](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Taikomas ML")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad peržiūrėtumėte vaizdo įrašą: Jen Looper kuria žiniatinklio programą, naudodama klasifikuotus virtuvės duomenis

## [Prieš pamoką - testas](https://ff-quizzes.netlify.app/en/ml/)

Šioje pamokoje išmoksite:

- Kaip sukurti modelį ir išsaugoti jį kaip Onnx modelį
- Kaip naudoti Netron modelio peržiūrai
- Kaip naudoti savo modelį žiniatinklio programoje prognozėms

## Sukurkite savo modelį

Taikomosios ML sistemos kūrimas yra svarbi šių technologijų pritaikymo jūsų verslo sistemoms dalis. Modelius galite naudoti savo žiniatinklio programose (taigi, jei reikia, juos galima naudoti ir neprisijungus) pasitelkdami Onnx.

[Ankstesnėje pamokoje](../../3-Web-App/1-Web-App/README.md) sukūrėte regresijos modelį apie NSO stebėjimus, „marinuotą“ jį ir panaudojote Flask programoje. Nors ši architektūra yra labai naudinga, tai yra pilnos apimties Python programa, o jūsų reikalavimai gali apimti JavaScript programos naudojimą.

Šioje pamokoje galite sukurti pagrindinę JavaScript pagrįstą sistemą prognozėms. Tačiau pirmiausia turite išmokyti modelį ir konvertuoti jį naudoti su Onnx.

## Užduotis - išmokykite klasifikavimo modelį

Pirmiausia išmokykite klasifikavimo modelį, naudodami išvalytą virtuvės duomenų rinkinį, kurį naudojome.

1. Pradėkite importuodami naudingas bibliotekas:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Jums reikės '[skl2onnx](https://onnx.ai/sklearn-onnx/)', kad padėtų konvertuoti jūsų Scikit-learn modelį į Onnx formatą.

1. Tada dirbkite su savo duomenimis taip, kaip darėte ankstesnėse pamokose, skaitydami CSV failą naudodami `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Pašalinkite pirmas dvi nereikalingas stulpelius ir išsaugokite likusius duomenis kaip 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Išsaugokite etiketes kaip 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Pradėkite mokymo procesą

Naudosime 'SVC' biblioteką, kuri pasižymi geru tikslumu.

1. Importuokite tinkamas bibliotekas iš Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Atskirkite mokymo ir testavimo rinkinius:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Sukurkite SVC klasifikavimo modelį, kaip darėte ankstesnėje pamokoje:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Dabar išbandykite savo modelį, iškviesdami `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Išveskite klasifikavimo ataskaitą, kad patikrintumėte modelio kokybę:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Kaip matėme anksčiau, tikslumas yra geras:

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

### Konvertuokite savo modelį į Onnx

Įsitikinkite, kad konvertavimas atliekamas su tinkamu tensorių skaičiumi. Šiame duomenų rinkinyje yra 380 ingredientų, todėl turite nurodyti šį skaičių `FloatTensorType`:

1. Konvertuokite, naudodami tensorių skaičių 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Sukurkite onx ir išsaugokite kaip failą **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Pastaba: galite perduoti [parinktis](https://onnx.ai/sklearn-onnx/parameterized.html) savo konvertavimo skripte. Šiuo atveju mes nustatėme 'nocl' kaip True ir 'zipmap' kaip False. Kadangi tai yra klasifikavimo modelis, turite galimybę pašalinti ZipMap, kuris sukuria žodynų sąrašą (nereikalinga). `nocl` reiškia, kad klasės informacija įtraukta į modelį. Sumažinkite savo modelio dydį, nustatydami `nocl` kaip 'True'.

Vykdydami visą užrašų knygelę dabar sukursite Onnx modelį ir išsaugosite jį šiame aplanke.

## Peržiūrėkite savo modelį

Onnx modeliai nėra labai matomi Visual Studio Code, tačiau yra labai gera nemokama programinė įranga, kurią daugelis tyrėjų naudoja modelio vizualizavimui, kad įsitikintų, jog jis tinkamai sukurtas. Atsisiųskite [Netron](https://github.com/lutzroeder/Netron) ir atidarykite savo model.onnx failą. Galite pamatyti savo paprastą modelį, vizualizuotą su 380 įvestimis ir klasifikatoriumi:

![Netron vizualizacija](../../../../4-Classification/4-Applied/images/netron.png)

Netron yra naudingas įrankis modelių peržiūrai.

Dabar esate pasiruošę naudoti šį puikų modelį žiniatinklio programoje. Sukurkime programą, kuri bus naudinga, kai žiūrėsite į savo šaldytuvą ir bandysite suprasti, kokią kombinaciją likusių ingredientų galite naudoti, kad pagamintumėte tam tikrą virtuvės patiekalą, kaip nustatyta jūsų modeliu.

## Sukurkite rekomendacijų žiniatinklio programą

Galite tiesiogiai naudoti savo modelį žiniatinklio programoje. Ši architektūra taip pat leidžia ją paleisti vietoje ir net neprisijungus, jei reikia. Pradėkite kurdami `index.html` failą tame pačiame aplanke, kuriame išsaugojote savo `model.onnx` failą.

1. Šiame faile _index.html_ pridėkite šį žymėjimą:

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

1. Dabar, dirbdami tarp `body` žymių, pridėkite šiek tiek žymėjimo, kad parodytumėte sąrašą žymimųjų langelių, atspindinčių kai kuriuos ingredientus:

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

    Atkreipkite dėmesį, kad kiekvienam žymimajam langeliui priskiriama vertė. Tai atspindi indeksą, kuriame ingredientas yra pagal duomenų rinkinį. Pavyzdžiui, obuolys šiame abėcėliniame sąraše užima penktą stulpelį, todėl jo vertė yra '4', nes pradedame skaičiuoti nuo 0. Galite pasikonsultuoti su [ingredientų skaičiuokle](../../../../4-Classification/data/ingredient_indexes.csv), kad sužinotumėte tam tikro ingrediento indeksą.

    Tęsdami darbą index.html faile, pridėkite scenarijaus bloką, kuriame modelis bus iškviestas po paskutinės uždaromos `</div>` žymės.

1. Pirmiausia importuokite [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime naudojamas, kad būtų galima paleisti jūsų Onnx modelius įvairiose aparatinės įrangos platformose, įskaitant optimizacijas ir API naudojimą.

1. Kai Runtime yra vietoje, galite jį iškviesti:

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

Šiame kode vyksta keli dalykai:

1. Sukūrėte 380 galimų verčių masyvą (1 arba 0), kuris bus nustatytas ir išsiųstas modeliui prognozėms, priklausomai nuo to, ar žymimasis langelis pažymėtas.
2. Sukūrėte žymimųjų langelių masyvą ir būdą nustatyti, ar jie buvo pažymėti, funkcijoje `init`, kuri iškviečiama, kai programa paleidžiama. Kai žymimasis langelis pažymėtas, masyvas `ingredients` pakeičiamas, kad atspindėtų pasirinktą ingredientą.
3. Sukūrėte funkciją `testCheckboxes`, kuri tikrina, ar pažymėtas bent vienas žymimasis langelis.
4. Naudojate funkciją `startInference`, kai paspaudžiamas mygtukas, ir, jei pažymėtas bent vienas žymimasis langelis, pradedate prognozavimą.
5. Prognozavimo rutina apima:
   1. Asinchroninį modelio įkėlimą
   2. Tensor struktūros sukūrimą, kuri bus išsiųsta modeliui
   3. „Feeds“ sukūrimą, kuris atspindi `float_input` įvestį, kurią sukūrėte mokydami savo modelį (galite naudoti Netron, kad patikrintumėte šį pavadinimą)
   4. Šių „feeds“ siuntimą modeliui ir atsakymo laukimą

## Išbandykite savo programą

Atidarykite terminalo sesiją Visual Studio Code aplanke, kuriame yra jūsų index.html failas. Įsitikinkite, kad turite [http-server](https://www.npmjs.com/package/http-server) įdiegtą globaliai, ir įveskite `http-server` komandoje. Turėtų atsidaryti localhost, ir galėsite peržiūrėti savo žiniatinklio programą. Patikrinkite, kokia virtuvė rekomenduojama pagal įvairius ingredientus:

![ingredientų žiniatinklio programa](../../../../4-Classification/4-Applied/images/web-app.png)

Sveikiname, sukūrėte „rekomendacijų“ žiniatinklio programą su keliais laukais. Skirkite šiek tiek laiko šios sistemos plėtojimui!

## 🚀Iššūkis

Jūsų žiniatinklio programa yra labai minimali, todėl toliau ją plėtokite, naudodami ingredientus ir jų indeksus iš [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) duomenų. Kokios skonio kombinacijos tinka tam tikram nacionaliniam patiekalui sukurti?

## [Po pamokos - testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Nors ši pamoka tik trumpai palietė maisto ingredientų rekomendacijų sistemos kūrimo naudingumą, ši ML pritaikymo sritis yra labai turtinga pavyzdžiais. Skaitykite daugiau apie tai, kaip šios sistemos kuriamos:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Užduotis 

[Sukurkite naują rekomendacijų sistemą](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors stengiamės užtikrinti tikslumą, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.