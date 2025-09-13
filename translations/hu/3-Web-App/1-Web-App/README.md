<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T16:12:58+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "hu"
}
-->
# Építsünk egy webalkalmazást gépi tanulási modell használatához

Ebben a leckében egy gépi tanulási modellt fogsz betanítani egy igazán különleges adathalmazon: _UFO-észlelések az elmúlt évszázadból_, amelyet a NUFORC adatbázisából származtatunk.

A következőket fogod megtanulni:

- Hogyan lehet egy betanított modellt "pickle"-elni
- Hogyan lehet ezt a modellt egy Flask alkalmazásban használni

Továbbra is notebookokat fogunk használni az adatok tisztítására és a modell betanítására, de egy lépéssel tovább mehetsz, ha felfedezed, hogyan lehet egy modellt "a vadonban" használni, azaz egy webalkalmazásban.

Ehhez egy Flask alapú webalkalmazást kell építened.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Alkalmazás építése

Számos módja van annak, hogy webalkalmazásokat építsünk gépi tanulási modellek használatához. A webes architektúra befolyásolhatja, hogyan kell a modellt betanítani. Képzeld el, hogy egy olyan vállalatnál dolgozol, ahol az adatkutatási csoport betanított egy modellt, amelyet neked kellene egy alkalmazásban használnod.

### Szempontok

Számos kérdést kell feltenned:

- **Webalkalmazásról vagy mobilalkalmazásról van szó?** Ha mobilalkalmazást építesz, vagy az IoT kontextusában kell használnod a modellt, használhatod a [TensorFlow Lite](https://www.tensorflow.org/lite/) megoldást, és a modellt Android vagy iOS alkalmazásban használhatod.
- **Hol fog a modell elhelyezkedni?** A felhőben vagy helyileg?
- **Offline támogatás.** Az alkalmazásnak offline is működnie kell?
- **Milyen technológiával lett a modell betanítva?** A választott technológia befolyásolhatja a szükséges eszközöket.
    - **TensorFlow használata.** Ha például TensorFlow-val tanítasz modellt, az ökoszisztéma lehetőséget biztosít arra, hogy a modellt webalkalmazásban használhatóvá alakítsd a [TensorFlow.js](https://www.tensorflow.org/js/) segítségével.
    - **PyTorch használata.** Ha például [PyTorch](https://pytorch.org/) könyvtárral építesz modellt, lehetőséged van azt [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formátumba exportálni, amely JavaScript webalkalmazásokban használható az [Onnx Runtime](https://www.onnxruntime.ai/) segítségével. Ezt az opciót egy későbbi leckében fogjuk megvizsgálni egy Scikit-learn-nel betanított modell esetében.
    - **Lobe.ai vagy Azure Custom Vision használata.** Ha egy ML SaaS (Software as a Service) rendszert, például a [Lobe.ai](https://lobe.ai/) vagy az [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) szolgáltatást használod modell betanítására, az ilyen típusú szoftverek lehetőséget biztosítanak a modell különböző platformokra történő exportálására, beleértve egy egyedi API létrehozását, amelyet a felhőben lehet lekérdezni az online alkalmazásod által.

Lehetőséged van arra is, hogy egy teljes Flask webalkalmazást építs, amely képes lenne a modellt közvetlenül a böngészőben betanítani. Ez szintén megvalósítható a TensorFlow.js segítségével JavaScript környezetben.

A mi esetünkben, mivel Python-alapú notebookokkal dolgozunk, nézzük meg, milyen lépéseket kell megtenned ahhoz, hogy egy betanított modellt exportálj egy Python-alapú webalkalmazás által olvasható formátumba.

## Eszközök

Ehhez a feladathoz két eszközre lesz szükséged: Flask és Pickle, amelyek mind Pythonon futnak.

✅ Mi az a [Flask](https://palletsprojects.com/p/flask/)? A készítők "mikrokeretrendszerként" definiálják, a Flask biztosítja a webes keretrendszerek alapvető funkcióit Pythonban, valamint egy sablonmotor segítségével weboldalak építésére. Nézd meg [ezt a tanulási modult](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), hogy gyakorold a Flask használatát.

✅ Mi az a [Pickle](https://docs.python.org/3/library/pickle.html)? A Pickle 🥒 egy Python modul, amely egy Python objektumstruktúrát sorosít és visszafejt. Amikor egy modellt "pickle"-elsz, akkor annak struktúráját sorosítod vagy lapítod el, hogy a weben használható legyen. Vigyázz: a pickle önmagában nem biztonságos, ezért légy óvatos, ha egy fájl "un-pickle"-elésére kérnek. Egy pickle fájl kiterjesztése `.pkl`.

## Gyakorlat - tisztítsd meg az adataidat

Ebben a leckében 80 000 UFO-észlelés adatait fogod használni, amelyeket a [NUFORC](https://nuforc.org) (National UFO Reporting Center) gyűjtött össze. Ezek az adatok érdekes leírásokat tartalmaznak az UFO-észlelésekről, például:

- **Hosszú példa leírás.** "Egy férfi egy fénysugárból lép elő, amely egy füves mezőre világít éjszaka, majd a Texas Instruments parkolójába fut."
- **Rövid példa leírás.** "A fények üldöztek minket."

A [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) táblázat oszlopokat tartalmaz az észlelés helyéről (`city`, `state`, `country`), az objektum `alakjáról`, valamint annak `szélességi` és `hosszúsági` koordinátáiról.

A leckéhez tartozó üres [notebookban](../../../../3-Web-App/1-Web-App/notebook.ipynb):

1. importáld a `pandas`, `matplotlib` és `numpy` könyvtárakat, ahogy az előző leckékben tetted, és importáld az ufos táblázatot. Megnézheted az adathalmaz egy mintáját:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Alakítsd az ufos adatokat egy kisebb adatkeretté friss címekkel. Ellenőrizd az egyedi értékeket a `Country` mezőben.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Most csökkentheted a feldolgozandó adatok mennyiségét azáltal, hogy eldobod a hiányzó értékeket, és csak az 1-60 másodperc közötti észleléseket importálod:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importáld a Scikit-learn `LabelEncoder` könyvtárát, hogy a szöveges országértékeket számokká alakítsd:

    ✅ A LabelEncoder ábécé sorrendben kódolja az adatokat

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Az adataidnak így kell kinézniük:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Gyakorlat - építsd fel a modelledet

Most felkészülhetsz a modell betanítására az adatok tanulási és tesztelési csoportra osztásával.

1. Válaszd ki azt a három jellemzőt, amelyeken a modellt betanítod, mint X vektor, és az y vektor a `Country` lesz. A cél az, hogy a `Seconds`, `Latitude` és `Longitude` értékekből egy országazonosítót kapj vissza.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Tanítsd be a modelledet logisztikus regresszióval:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

Az eredmény pontossága nem rossz **(kb. 95%)**, ami nem meglepő, mivel a `Country` és a `Latitude/Longitude` összefügg.

A létrehozott modell nem túl forradalmi, mivel a `Latitude` és `Longitude` alapján könnyen kikövetkeztethető az `Country`, de jó gyakorlat arra, hogy nyers adatokból indulva tisztítsd, exportáld, majd egy webalkalmazásban használd a modellt.

## Gyakorlat - "pickle"-eld a modelledet

Most itt az ideje, hogy _pickle_-eld a modelledet! Ezt néhány sor kóddal megteheted. Miután _pickle_-elted, töltsd be a pickle fájlt, és teszteld egy mintaadat tömbbel, amely tartalmazza a másodpercek, szélességi és hosszúsági értékeket.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

A modell az **'3'** értéket adja vissza, amely az Egyesült Királyság országkódja. Hihetetlen! 👽

## Gyakorlat - építs egy Flask alkalmazást

Most építhetsz egy Flask alkalmazást, amely meghívja a modelledet, és hasonló eredményeket ad vissza, de vizuálisan vonzóbb módon.

1. Kezdd azzal, hogy létrehozol egy **web-app** nevű mappát a _notebook.ipynb_ fájl mellé, ahol az _ufo-model.pkl_ fájl található.

1. Ebben a mappában hozz létre három további mappát: **static**, benne egy **css** mappával, és **templates**. Most a következő fájlokkal és könyvtárakkal kell rendelkezned:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Nézd meg a megoldás mappát a kész alkalmazás nézetéhez

1. Az első fájl, amit létre kell hoznod a _web-app_ mappában, a **requirements.txt** fájl. Ez a fájl, hasonlóan a _package.json_-hoz egy JavaScript alkalmazásban, felsorolja az alkalmazás által igényelt függőségeket. A **requirements.txt** fájlba írd be a következő sorokat:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Most futtasd ezt a fájlt a _web-app_ mappába navigálva:

    ```bash
    cd web-app
    ```

1. A terminálban írd be a `pip install` parancsot, hogy telepítsd a _requirements.txt_-ben felsorolt könyvtárakat:

    ```bash
    pip install -r requirements.txt
    ```

1. Most készen állsz arra, hogy három további fájlt hozz létre az alkalmazás befejezéséhez:

    1. Hozd létre az **app.py** fájlt a gyökérben.
    2. Hozd létre az **index.html** fájlt a _templates_ könyvtárban.
    3. Hozd létre a **styles.css** fájlt a _static/css_ könyvtárban.

1. Töltsd ki a _styles.css_ fájlt néhány stílussal:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Ezután töltsd ki az _index.html_ fájlt:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Nézd meg a sablonozást ebben a fájlban. Figyeld meg a változók körüli 'mustache' szintaxist, amelyeket az alkalmazás biztosít, például az előrejelzési szöveget: `{{}}`. Van egy űrlap is, amely a `/predict` útvonalra küld egy előrejelzést.

    Végül készen állsz arra, hogy megírd azt a Python fájlt, amely a modell fogyasztását és az előrejelzések megjelenítését vezérli:

1. Az `app.py` fájlba írd be:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Tipp: Ha a Flask alkalmazás futtatása közben hozzáadod a [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) opciót, az alkalmazásban végrehajtott változtatások azonnal tükröződnek, anélkül hogy újra kellene indítanod a szervert. Vigyázz! Ne engedélyezd ezt az üzemmódot egy éles alkalmazásban.

Ha futtatod a `python app.py` vagy `python3 app.py` parancsot, a webkiszolgáló helyileg elindul, és kitölthetsz egy rövid űrlapot, hogy választ kapj az UFO-észlelésekkel kapcsolatos égető kérdésedre!

Mielőtt ezt megtennéd, nézd meg az `app.py` részeit:

1. Először a függőségek betöltődnek, és az alkalmazás elindul.
1. Ezután a modell importálódik.
1. Ezután az index.html renderelődik a kezdő útvonalon.

A `/predict` útvonalon több dolog történik, amikor az űrlapot elküldik:

1. Az űrlap változói összegyűjtésre kerülnek, és numpy tömbbé konvertálódnak. Ezeket elküldik a modellnek, amely visszaad egy előrejelzést.
2. Az országok, amelyeket meg akarunk jeleníteni, olvasható szöveggé alakulnak a megjósolt országkódból, és ez az érték visszaküldésre kerül az index.html-nek, hogy a sablonban megjelenjen.

Egy modellt ilyen módon használni Flask és pickle segítségével viszonylag egyszerű. A legnehezebb dolog megérteni, hogy milyen formátumú adatokat kell a modellnek küldeni az előrejelzéshez. Ez teljes mértékben attól függ, hogyan lett a modell betanítva. Ennél a modellnél három adatpontot kell megadni az előrejelzéshez.

Egy professzionális környezetben láthatod, hogy mennyire fontos a jó kommunikáció azok között, akik a modellt betanítják, és azok között, akik azt egy web- vagy mobilalkalmazásban használják. A mi esetünkben ez csak egy személy, te vagy!

---

## 🚀 Kihívás

Ahelyett, hogy notebookban dolgoznál, és onnan importálnád a modellt a Flask alkalmazásba, próbáld meg a modellt közvetlenül a Flask alkalmazásban betanítani! Próbáld meg átalakítani a notebookban lévő Python kódot úgy, hogy az alkalmazásban, például egy `train` nevű útvonalon belül történjen a modell betanítása. Mik az előnyei és hátrányai ennek a módszernek?

## [Előadás utáni kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Számos módja van annak, hogy webalkalmazást építsünk gépi tanulási modellek használatához. Készíts egy listát azokról a módokról, ahogyan JavaScript vagy Python segítségével webalkalmazást építhetsz gépi tanulás alkalmazására. Gondolj az architektúrára: a modell az alkalmazásban maradjon, vagy a felhőben legyen? Ha az utóbbi, hogyan érnéd el? Rajzolj egy architekturális modellt egy alkalmazott ML webes megoldáshoz.

## Feladat

[Próbálj ki egy másik modellt](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.