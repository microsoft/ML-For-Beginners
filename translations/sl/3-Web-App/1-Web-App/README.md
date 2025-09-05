<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T12:58:49+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "sl"
}
-->
# Zgradite spletno aplikacijo za uporabo modela strojnega učenja

V tej lekciji boste trenirali model strojnega učenja na podatkovnem naboru, ki je resnično izjemen: _opazovanja NLP-jev v zadnjem stoletju_, pridobljenih iz baze podatkov NUFORC.

Naučili se boste:

- Kako 'shraniti' treniran model
- Kako uporabiti ta model v aplikaciji Flask

Še naprej bomo uporabljali beležke za čiščenje podatkov in treniranje modela, vendar lahko proces nadgradite tako, da raziščete uporabo modela 'v divjini', če lahko tako rečemo: v spletni aplikaciji.

Za to morate zgraditi spletno aplikacijo z uporabo Flask.

## [Predlekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Gradnja aplikacije

Obstaja več načinov za gradnjo spletnih aplikacij, ki uporabljajo modele strojnega učenja. Vaša spletna arhitektura lahko vpliva na način, kako je model treniran. Predstavljajte si, da delate v podjetju, kjer je skupina za podatkovno znanost trenirala model, ki ga želite uporabiti v aplikaciji.

### Premisleki

Obstaja veliko vprašanj, ki jih morate zastaviti:

- **Ali gre za spletno ali mobilno aplikacijo?** Če gradite mobilno aplikacijo ali morate model uporabiti v kontekstu IoT, lahko uporabite [TensorFlow Lite](https://www.tensorflow.org/lite/) in model uporabite v aplikaciji za Android ali iOS.
- **Kje bo model shranjen?** V oblaku ali lokalno?
- **Podpora brez povezave.** Ali mora aplikacija delovati brez povezave?
- **Katera tehnologija je bila uporabljena za treniranje modela?** Izbrana tehnologija lahko vpliva na orodja, ki jih morate uporabiti.
    - **Uporaba TensorFlow.** Če trenirate model z uporabo TensorFlow, na primer, ta ekosistem omogoča pretvorbo modela TensorFlow za uporabo v spletni aplikaciji z uporabo [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Uporaba PyTorch.** Če gradite model z uporabo knjižnice, kot je [PyTorch](https://pytorch.org/), imate možnost, da ga izvozite v format [ONNX](https://onnx.ai/) (Open Neural Network Exchange) za uporabo v JavaScript spletnih aplikacijah, ki lahko uporabljajo [Onnx Runtime](https://www.onnxruntime.ai/). To možnost bomo raziskali v prihodnji lekciji za model, treniran s Scikit-learn.
    - **Uporaba Lobe.ai ali Azure Custom Vision.** Če uporabljate sistem ML SaaS (Software as a Service), kot sta [Lobe.ai](https://lobe.ai/) ali [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) za treniranje modela, ta vrsta programske opreme omogoča načine za izvoz modela za številne platforme, vključno z gradnjo prilagojenega API-ja, ki ga vaša spletna aplikacija lahko poizveduje v oblaku.

Imate tudi možnost zgraditi celotno spletno aplikacijo Flask, ki bi lahko trenirala model kar v spletnem brskalniku. To je mogoče storiti tudi z uporabo TensorFlow.js v kontekstu JavaScript.

Za naše namene, ker smo delali z beležkami, ki temeljijo na Pythonu, raziščimo korake, ki jih morate narediti, da izvozite treniran model iz takšne beležke v format, ki ga lahko bere spletna aplikacija, zgrajena v Pythonu.

## Orodje

Za to nalogo potrebujete dve orodji: Flask in Pickle, oba delujeta na Pythonu.

✅ Kaj je [Flask](https://palletsprojects.com/p/flask/)? Flask, opisan kot 'mikro-okvir' s strani njegovih ustvarjalcev, ponuja osnovne funkcije spletnih okvirov z uporabo Pythona in predloge za gradnjo spletnih strani. Oglejte si [ta modul za učenje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), da vadite gradnjo s Flask.

✅ Kaj je [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 je Python modul, ki serializira in de-serializira strukturo Python objektov. Ko 'shranite' model, serializirate ali sploščite njegovo strukturo za uporabo na spletu. Bodite previdni: pickle ni sam po sebi varen, zato bodite previdni, če vas pozove k 'odshranjevanju' datoteke. Shranjena datoteka ima pripono `.pkl`.

## Naloga - očistite svoje podatke

V tej lekciji boste uporabili podatke iz 80.000 opazovanj NLP-jev, zbranih s strani [NUFORC](https://nuforc.org) (Nacionalni center za poročanje o NLP-jih). Ti podatki vsebujejo zanimive opise opazovanj NLP-jev, na primer:

- **Dolgi opis primera.** "Moški se pojavi iz svetlobnega žarka, ki sveti na travnato polje ponoči, in teče proti parkirišču Texas Instruments."
- **Kratki opis primera.** "luči so nas lovile."

Preglednica [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) vključuje stolpce o `mestu`, `državi` in `državi`, kjer se je opazovanje zgodilo, obliki objekta (`shape`) ter njegovi `zemljepisni širini` in `zemljepisni dolžini`.

V prazni [beležki](../../../../3-Web-App/1-Web-App/notebook.ipynb), ki je vključena v to lekcijo:

1. uvozite `pandas`, `matplotlib` in `numpy`, kot ste to storili v prejšnjih lekcijah, ter uvozite preglednico NLP-jev. Lahko si ogledate vzorčni podatkovni nabor:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Pretvorite podatke NLP-jev v majhen podatkovni okvir z novimi naslovi. Preverite unikatne vrednosti v polju `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Zdaj lahko zmanjšate količino podatkov, s katerimi se morate ukvarjati, tako da odstranite vse vrednosti null in uvozite samo opazovanja med 1-60 sekundami:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Uvozite knjižnico Scikit-learn `LabelEncoder`, da pretvorite besedilne vrednosti za države v številke:

    ✅ LabelEncoder kodira podatke po abecedi

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Vaši podatki bi morali izgledati takole:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Naloga - zgradite svoj model

Zdaj se lahko pripravite na treniranje modela tako, da podatke razdelite v skupino za treniranje in testiranje.

1. Izberite tri značilnosti, na katerih želite trenirati, kot vaš X vektor, medtem ko bo y vektor `Country`. Želite vnesti `Seconds`, `Latitude` in `Longitude` ter dobiti ID države kot rezultat.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trenirajte svoj model z uporabo logistične regresije:

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

Natančnost ni slaba **(približno 95%)**, kar ni presenetljivo, saj se `Country` in `Latitude/Longitude` korelirata.

Model, ki ste ga ustvarili, ni zelo revolucionaren, saj bi morali biti sposobni sklepati `Country` iz njegovih `Latitude` in `Longitude`, vendar je to dobra vaja za treniranje iz surovih podatkov, ki ste jih očistili, izvozili in nato uporabili ta model v spletni aplikaciji.

## Naloga - 'shranite' svoj model

Zdaj je čas, da _shranite_ svoj model! To lahko storite v nekaj vrsticah kode. Ko je _shranjen_, naložite shranjeni model in ga preizkusite na vzorčnem podatkovnem nizu, ki vsebuje vrednosti za sekunde, zemljepisno širino in dolžino.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model vrne **'3'**, kar je koda države za Združeno kraljestvo. Neverjetno! 👽

## Naloga - zgradite aplikacijo Flask

Zdaj lahko zgradite aplikacijo Flask, ki bo klicala vaš model in vračala podobne rezultate, vendar na bolj vizualno privlačen način.

1. Začnite z ustvarjanjem mape **web-app** poleg datoteke _notebook.ipynb_, kjer se nahaja vaša datoteka _ufo-model.pkl_.

1. V tej mapi ustvarite še tri mape: **static**, z mapo **css** znotraj nje, in **templates**. Zdaj bi morali imeti naslednje datoteke in direktorije:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Oglejte si mapo z rešitvami za pogled na končano aplikacijo

1. Prva datoteka, ki jo ustvarite v mapi _web-app_, je datoteka **requirements.txt**. Kot _package.json_ v aplikaciji JavaScript, ta datoteka navaja odvisnosti, ki jih aplikacija potrebuje. V **requirements.txt** dodajte vrstice:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Zdaj zaženite to datoteko tako, da se premaknete v _web-app_:

    ```bash
    cd web-app
    ```

1. V terminalu vnesite `pip install`, da namestite knjižnice, navedene v _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Zdaj ste pripravljeni ustvariti še tri datoteke za dokončanje aplikacije:

    1. Ustvarite **app.py** v korenu.
    2. Ustvarite **index.html** v mapi _templates_.
    3. Ustvarite **styles.css** v mapi _static/css_.

1. Izdelajte datoteko _styles.css_ z nekaj slogi:

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

1. Nato izdelajte datoteko _index.html_:

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

    Oglejte si predloge v tej datoteki. Opazite sintakso 'mustache' okoli spremenljivk, ki jih bo zagotovila aplikacija, kot je besedilo napovedi: `{{}}`. Obstaja tudi obrazec, ki pošlje napoved na pot `/predict`.

    Končno ste pripravljeni zgraditi Python datoteko, ki upravlja porabo modela in prikaz napovedi:

1. V `app.py` dodajte:

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

    > 💡 Nasvet: ko dodate [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) med zagonom spletne aplikacije z uporabo Flask, bodo vse spremembe, ki jih naredite v svoji aplikaciji, takoj vidne brez potrebe po ponovnem zagonu strežnika. Pazite! Ne omogočite tega načina v produkcijski aplikaciji.

Če zaženete `python app.py` ali `python3 app.py` - vaš spletni strežnik se zažene lokalno in lahko izpolnite kratek obrazec, da dobite odgovor na svoje pereče vprašanje o tem, kje so bili NLP-ji opaženi!

Preden to storite, si oglejte dele `app.py`:

1. Najprej se naložijo odvisnosti in aplikacija se zažene.
1. Nato se model uvozi.
1. Nato se na domači poti prikaže index.html.

Na poti `/predict` se zgodi več stvari, ko je obrazec poslan:

1. Spremenljivke obrazca se zberejo in pretvorijo v numpy array. Nato se pošljejo modelu in vrne se napoved.
2. Države, ki jih želimo prikazati, se ponovno prikažejo kot berljivo besedilo iz njihove predvidene kode države, in ta vrednost se pošlje nazaj v index.html, da se prikaže v predlogi.

Uporaba modela na ta način, z Flask in shranjenim modelom, je razmeroma preprosta. Najtežje je razumeti, kakšne oblike morajo biti podatki, ki jih je treba poslati modelu, da dobite napoved. To je odvisno od tega, kako je bil model treniran. Ta model zahteva tri podatkovne točke za vnos, da vrne napoved.

V profesionalnem okolju lahko vidite, kako pomembna je dobra komunikacija med ljudmi, ki trenirajo model, in tistimi, ki ga uporabljajo v spletni ali mobilni aplikaciji. V našem primeru ste to le vi!

---

## 🚀 Izziv

Namesto da delate v beležki in uvozite model v aplikacijo Flask, bi lahko model trenirali kar znotraj aplikacije Flask! Poskusite pretvoriti svojo Python kodo v beležki, morda po tem, ko so vaši podatki očiščeni, da trenirate model znotraj aplikacije na poti, imenovani `train`. Kakšne so prednosti in slabosti tega pristopa?

## [Po-lekcijski kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Obstaja veliko načinov za gradnjo spletne aplikacije, ki uporablja modele strojnega učenja. Naredite seznam načinov, kako bi lahko uporabili JavaScript ali Python za gradnjo spletne aplikacije, ki izkorišča strojno učenje. Razmislite o arhitekturi: naj model ostane v aplikaciji ali naj živi v oblaku? Če slednje, kako bi dostopali do njega? Narišite arhitekturni model za rešitev spletne aplikacije z uporabo strojnega učenja.

## Naloga

[Preizkusite drugačen model](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.