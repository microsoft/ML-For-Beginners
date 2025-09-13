<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T16:21:03+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "hu"
}
-->
# Építs egy konyhai ajánló webalkalmazást

Ebben a leckében egy osztályozási modellt fogsz építeni, felhasználva az előző leckékben tanult technikákat, valamint a sorozat során használt ízletes konyhai adatbázist. Ezen kívül egy kis webalkalmazást is készítesz, amely egy mentett modellt használ, az Onnx webes futtatókörnyezetét kihasználva.

A gépi tanulás egyik legpraktikusabb alkalmazása az ajánlórendszerek építése, és ma te is megteheted az első lépést ebbe az irányba!

[![Webalkalmazás bemutatása](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Alkalmazott ML")

> 🎥 Kattints a fenti képre a videóért: Jen Looper egy webalkalmazást épít osztályozott konyhai adatokkal

## [Előzetes kvíz](https://ff-quizzes.netlify.app/en/ml/)

Ebben a leckében megtanulod:

- Hogyan építs modellt és mentsd el Onnx formátumban
- Hogyan használd a Netron-t a modell vizsgálatához
- Hogyan használd a modelledet egy webalkalmazásban következtetéshez

## Építsd meg a modelledet

Az alkalmazott gépi tanulási rendszerek építése fontos része annak, hogy ezeket a technológiákat üzleti rendszereidben hasznosítsd. A modelleket webalkalmazásokban is használhatod (így offline környezetben is, ha szükséges), az Onnx segítségével.

Egy [korábbi leckében](../../3-Web-App/1-Web-App/README.md) egy regressziós modellt építettél UFO észlelésekről, "pickle"-be mentetted, és egy Flask alkalmazásban használtad. Bár ez az architektúra nagyon hasznos, egy teljes Python alapú alkalmazás, és az igényeid között szerepelhet egy JavaScript alkalmazás használata.

Ebben a leckében egy alapvető JavaScript-alapú rendszert építhetsz következtetéshez. Először azonban egy modellt kell betanítanod, és átalakítanod Onnx formátumra.

## Gyakorlat - osztályozási modell betanítása

Először is, taníts be egy osztályozási modellt a korábban használt tisztított konyhai adatbázis segítségével.

1. Kezdd azzal, hogy importálod a szükséges könyvtárakat:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Szükséged lesz a '[skl2onnx](https://onnx.ai/sklearn-onnx/)' könyvtárra, hogy a Scikit-learn modelledet Onnx formátumra konvertáld.

1. Ezután dolgozz az adataiddal ugyanúgy, ahogy az előző leckékben, olvasd be a CSV fájlt a `read_csv()` segítségével:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Távolítsd el az első két felesleges oszlopot, és mentsd el a fennmaradó adatokat 'X' néven:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Mentsd el a címkéket 'y' néven:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Kezdd el a betanítási folyamatot

Az 'SVC' könyvtárat fogjuk használni, amely jó pontosságot biztosít.

1. Importáld a megfelelő könyvtárakat a Scikit-learn-ből:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Válaszd szét a betanítási és tesztkészleteket:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Építs egy SVC osztályozási modellt, ahogy az előző leckében:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Most teszteld a modelledet a `predict()` hívásával:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Nyomtass ki egy osztályozási jelentést, hogy ellenőrizd a modell minőségét:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Ahogy korábban láttuk, a pontosság jó:

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

### Konvertáld a modelledet Onnx formátumra

Győződj meg róla, hogy a konverziót a megfelelő Tensor számmal végzed. Ez az adatbázis 380 összetevőt tartalmaz, így ezt a számot meg kell adnod a `FloatTensorType`-ban:

1. Konvertáld 380-as tensor számmal.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Hozd létre az onx fájlt, és mentsd el **model.onnx** néven:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Fontos, hogy [opciókat](https://onnx.ai/sklearn-onnx/parameterized.html) adhatsz meg a konverziós szkriptben. Ebben az esetben a 'nocl' értéket True-ra, a 'zipmap' értéket False-ra állítottuk. Mivel ez egy osztályozási modell, lehetőséged van eltávolítani a ZipMap-et, amely egy listát készít szótárakból (nem szükséges). A `nocl` arra utal, hogy az osztályinformációk szerepelnek-e a modellben. Csökkentsd a modell méretét a `nocl` True-ra állításával.

A teljes notebook futtatása most létrehozza az Onnx modellt, és elmenti ebbe a mappába.

## Tekintsd meg a modelledet

Az Onnx modellek nem láthatók jól a Visual Studio Code-ban, de van egy nagyon jó ingyenes szoftver, amelyet sok kutató használ a modellek vizualizálására, hogy megbizonyosodjon arról, hogy megfelelően épültek. Töltsd le a [Netron](https://github.com/lutzroeder/Netron) programot, és nyisd meg a model.onnx fájlt. Láthatod az egyszerű modelledet vizualizálva, a 380 bemenettel és az osztályozóval:

![Netron vizualizáció](../../../../4-Classification/4-Applied/images/netron.png)

A Netron egy hasznos eszköz a modellek megtekintéséhez.

Most készen állsz arra, hogy ezt az ügyes modellt egy webalkalmazásban használd. Építsünk egy alkalmazást, amely hasznos lehet, amikor a hűtőszekrényedbe nézel, és megpróbálod kitalálni, hogy a maradék összetevők kombinációjával milyen konyhát készíthetsz, ahogy azt a modelled meghatározza.

## Építs egy ajánló webalkalmazást

A modelledet közvetlenül egy webalkalmazásban használhatod. Ez az architektúra lehetővé teszi, hogy helyben és akár offline is futtasd. Kezdd azzal, hogy létrehozol egy `index.html` fájlt abban a mappában, ahol a `model.onnx` fájl található.

1. Ebben a fájlban _index.html_, add hozzá a következő jelölést:

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

1. Most, a `body` címkék között adj hozzá egy kis jelölést, amely néhány összetevőt tükröző jelölőnégyzeteket mutat:

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

    Figyeld meg, hogy minden jelölőnégyzethez érték van rendelve. Ez tükrözi az összetevő helyét az adatbázis szerint. Az alma például ebben az ábécé szerinti listában az ötödik oszlopot foglalja el, így az értéke '4', mivel 0-tól kezdünk számolni. Az összetevők indexét a [összetevők táblázatában](../../../../4-Classification/data/ingredient_indexes.csv) találhatod meg.

    Folytatva a munkát az index.html fájlban, adj hozzá egy szkript blokkot, ahol a modell hívása történik a végső záró `</div>` után.

1. Először importáld az [Onnx Runtime](https://www.onnxruntime.ai/) könyvtárat:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Az Onnx Runtime lehetővé teszi, hogy az Onnx modelleket különböző hardverplatformokon futtasd, optimalizálásokkal és egy API-val.

1. Miután a Runtime helyén van, hívd meg:

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

Ebben a kódban több dolog történik:

1. Létrehoztál egy 380 lehetséges értéket (1 vagy 0) tartalmazó tömböt, amelyet beállítasz és elküldesz a modellnek következtetéshez, attól függően, hogy egy összetevő jelölőnégyzet be van-e jelölve.
2. Létrehoztál egy jelölőnégyzetek tömbjét és egy módot annak meghatározására, hogy be vannak-e jelölve egy `init` függvényben, amelyet az alkalmazás indításakor hívsz meg. Amikor egy jelölőnégyzet be van jelölve, az `ingredients` tömb módosul, hogy tükrözze a kiválasztott összetevőt.
3. Létrehoztál egy `testCheckboxes` függvényt, amely ellenőrzi, hogy van-e bejelölt jelölőnégyzet.
4. A `startInference` függvényt használod, amikor a gombot megnyomják, és ha van bejelölt jelölőnégyzet, elindítod a következtetést.
5. A következtetési rutin tartalmazza:
   1. A modell aszinkron betöltésének beállítását
   2. Egy Tensor struktúra létrehozását, amelyet elküldesz a modellnek
   3. 'Feeds' létrehozását, amely tükrözi a `float_input` bemenetet, amelyet a modelled betanításakor hoztál létre (a Netron segítségével ellenőrizheted ezt a nevet)
   4. Ezeknek a 'feeds'-eknek a modellhez való elküldését és a válasz megvárását

## Teszteld az alkalmazásodat

Nyiss egy terminálablakot a Visual Studio Code-ban abban a mappában, ahol az index.html fájl található. Győződj meg róla, hogy a [http-server](https://www.npmjs.com/package/http-server) globálisan telepítve van, és írd be a `http-server` parancsot. Egy localhost megnyílik, és megtekintheted a webalkalmazásodat. Ellenőrizd, hogy milyen konyhát ajánl a különböző összetevők alapján:

![összetevő webalkalmazás](../../../../4-Classification/4-Applied/images/web-app.png)

Gratulálok, létrehoztál egy 'ajánló' webalkalmazást néhány mezővel. Szánj időt arra, hogy továbbfejleszd ezt a rendszert!

## 🚀Kihívás

A webalkalmazásod nagyon minimális, így folytasd a fejlesztését az összetevők és azok indexei alapján a [összetevők indexei](../../../../4-Classification/data/ingredient_indexes.csv) adatból. Milyen ízkombinációk működnek egy adott nemzeti étel elkészítéséhez?

## [Utólagos kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Bár ez a lecke csak érintette az ételösszetevők ajánlórendszerének létrehozásának hasznosságát, ez a gépi tanulási alkalmazások területe nagyon gazdag példákban. Olvass többet arról, hogyan épülnek ezek a rendszerek:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Feladat 

[Építs egy új ajánlót](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.