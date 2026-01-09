<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-10-11T12:04:11+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "et"
}
-->
# Ehita veebirakendus ML-mudeli kasutamiseks

Selles tunnis treenid ML-mudelit andmekogumiga, mis on täiesti teistsugune: _UFO-vaatlused viimase sajandi jooksul_, pärinedes NUFORC-i andmebaasist.

Sa õpid:

- Kuidas 'marineerida' treenitud mudelit
- Kuidas kasutada seda mudelit Flaski rakenduses

Jätkame sülearvutite kasutamist andmete puhastamiseks ja mudeli treenimiseks, kuid võid protsessi viia sammu võrra kaugemale, uurides mudeli kasutamist "metsikus looduses", nii öelda: veebirakenduses.

Selleks pead ehitama veebirakenduse, kasutades Flaski.

## [Loengu-eelne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Rakenduse loomine

Masinõppe mudelite tarbimiseks veebirakenduste loomiseks on mitmeid viise. Sinu veebiarhitektuur võib mõjutada mudeli treenimise viisi. Kujuta ette, et töötad ettevõttes, kus andmeteaduse meeskond on treeninud mudeli, mida nad soovivad rakenduses kasutada.

### Mõtteainet

On mitmeid küsimusi, mida pead endalt küsima:

- **Kas see on veebirakendus või mobiilirakendus?** Kui ehitad mobiilirakendust või vajad mudelit IoT kontekstis, võid kasutada [TensorFlow Lite](https://www.tensorflow.org/lite/) ja kasutada mudelit Androidi või iOS-i rakenduses.
- **Kus mudel asub?** Pilves või kohapeal?
- **Võimalus töötada võrguühenduseta.** Kas rakendus peab töötama võrguühenduseta?
- **Millist tehnoloogiat kasutati mudeli treenimiseks?** Valitud tehnoloogia võib mõjutada vajalikke tööriistu.
    - **TensorFlow kasutamine.** Kui treenid mudelit TensorFlow abil, pakub see ökosüsteem võimalust konverteerida TensorFlow mudel veebirakenduses kasutamiseks, kasutades [TensorFlow.js](https://www.tensorflow.org/js/).
    - **PyTorch kasutamine.** Kui ehitad mudelit, kasutades sellist teeki nagu [PyTorch](https://pytorch.org/), on sul võimalus eksportida see [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formaadis JavaScripti veebirakenduste jaoks, mis saavad kasutada [Onnx Runtime](https://www.onnxruntime.ai/). Seda võimalust uuritakse tulevases tunnis Scikit-learniga treenitud mudeli jaoks.
    - **Lobe.ai või Azure Custom Vision kasutamine.** Kui kasutad ML SaaS (tarkvara teenusena) süsteemi, nagu [Lobe.ai](https://lobe.ai/) või [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) mudeli treenimiseks, pakub selline tarkvara viise mudeli eksportimiseks mitmetele platvormidele, sealhulgas kohandatud API loomist, mida saab pilves päringuteks kasutada.

Sul on ka võimalus ehitada terve Flaski veebirakendus, mis suudab ise mudelit veebibrauseris treenida. Seda saab teha ka JavaScripti kontekstis, kasutades TensorFlow.js-i.

Meie eesmärkide jaoks, kuna oleme töötanud Pythonil põhinevate sülearvutitega, uurime samme, mida pead tegema, et eksportida treenitud mudel sellisest sülearvutist Pythonil ehitatud veebirakenduse jaoks loetavasse formaati.

## Tööriist

Selle ülesande jaoks vajad kahte tööriista: Flaski ja Pickle'it, mis mõlemad töötavad Pythonis.

✅ Mis on [Flask](https://palletsprojects.com/p/flask/)? Selle loojate poolt defineeritud kui 'mikro-raamistik', pakub Flask veebiraamistike põhifunktsioone, kasutades Pythonit ja mallimootorit veebilehtede loomiseks. Vaata [seda õppe moodulit](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), et harjutada Flaskiga ehitamist.

✅ Mis on [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 on Pythoni moodul, mis serialiseerib ja deserialiseerib Pythoni objektistruktuuri. Kui 'marineerid' mudelit, serialiseerid või lamedad selle struktuuri veebis kasutamiseks. Ole ettevaatlik: Pickle ei ole olemuselt turvaline, seega ole ettevaatlik, kui sind kutsutakse faili 'lahti marineerima'. Marineeritud failil on järelliide `.pkl`.

## Harjutus - puhasta oma andmed

Selles tunnis kasutad andmeid 80 000 UFO-vaatlusest, mis on kogutud [NUFORC](https://nuforc.org) (Riiklik UFO-raportite keskus) poolt. Need andmed sisaldavad huvitavaid kirjeldusi UFO-vaatlustest, näiteks:

- **Pikk näite kirjeldus.** "Mees ilmub valguskiirest, mis paistab öösel rohtunud väljal, ja jookseb Texas Instrumentsi parklat suunas."
- **Lühike näite kirjeldus.** "tuled jälitasid meid."

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) tabel sisaldab veerge `city`, `state` ja `country`, kus vaatlus toimus, objekti `shape` ning selle `latitude` ja `longitude`.

Kaasaolevas tühjas [sülearvutis](notebook.ipynb):

1. impordi `pandas`, `matplotlib` ja `numpy`, nagu tegid eelnevates tundides, ning impordi ufode tabel. Võid vaadata näidisandmekogumit:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konverteeri ufode andmed väiksemaks andmeraamiks värskete pealkirjadega. Kontrolli unikaalseid väärtusi `Country` väljal.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nüüd saad vähendada andmete hulka, millega pead tegelema, eemaldades kõik nullväärtused ja importides ainult vaatlused, mis kestavad 1-60 sekundit:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Impordi Scikit-learn'i `LabelEncoder` teek, et konverteerida riikide tekstiväärtused numbriteks:

    ✅ LabelEncoder kodeerib andmeid tähestikulises järjekorras

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Sinu andmed peaksid välja nägema sellised:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Harjutus - ehita oma mudel

Nüüd saad valmistuda mudeli treenimiseks, jagades andmed treening- ja testimisgruppi.

1. Vali kolm funktsiooni, mille põhjal soovid mudelit treenida, kui oma X vektor, ja y vektoriks saab `Country`. Soovid sisestada `Seconds`, `Latitude` ja `Longitude` ning saada riigi ID-d tagastamiseks.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Treeni oma mudel logistilise regressiooni abil:

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

Täpsus pole halb **(umbes 95%)**, mis pole üllatav, kuna `Country` ja `Latitude/Longitude` korreleeruvad.

Loodud mudel pole väga revolutsiooniline, kuna peaksid suutma `Country` tuletada selle `Latitude` ja `Longitude` põhjal, kuid see on hea harjutus, et proovida treenida toorandmetest, mida oled puhastanud, eksportinud ja seejärel kasutada seda mudelit veebirakenduses.

## Harjutus - 'marineeri' oma mudel

Nüüd on aeg oma mudel _marineerida_! Seda saad teha mõne koodirea abil. Kui see on _marineeritud_, laadi oma marineeritud mudel ja testi seda näidisandmete massiivi vastu, mis sisaldab väärtusi sekundite, laius- ja pikkuskraadi kohta.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Mudel tagastab **'3'**, mis on Ühendkuningriigi riigikood. Hämmastav! 👽

## Harjutus - ehita Flaski rakendus

Nüüd saad ehitada Flaski rakenduse, et kutsuda oma mudelit ja tagastada sarnaseid tulemusi, kuid visuaalselt meeldivamal viisil.

1. Alusta kausta **web-app** loomisega _notebook.ipynb_ faili kõrvale, kus asub sinu _ufo-model.pkl_ fail.

1. Loo sellesse kausta veel kolm kausta: **static**, mille sees on kaust **css**, ja **templates**. Sul peaks nüüd olema järgmised failid ja kataloogid:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Vaata lahenduse kausta, et näha valmis rakenduse vaadet

1. Esimene fail, mida _web-app_ kaustas luua, on **requirements.txt** fail. Nagu _package.json_ JavaScripti rakenduses, loetleb see fail rakenduse jaoks vajalikud sõltuvused. Lisa **requirements.txt** faili read:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Nüüd käivita see fail, liikudes _web-app_ kausta:

    ```bash
    cd web-app
    ```

1. Oma terminalis kirjuta `pip install`, et installida _requirements.txt_ failis loetletud teegid:

    ```bash
    pip install -r requirements.txt
    ```

1. Nüüd oled valmis looma veel kolm faili, et rakendus lõpetada:

    1. Loo **app.py** juurkausta.
    2. Loo **index.html** _templates_ kataloogi.
    3. Loo **styles.css** _static/css_ kataloogi.

1. Täienda _styles.css_ faili mõne stiiliga:

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

1. Järgmisena täienda _index.html_ faili:

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

    Vaata selle faili mallindust. Pane tähele 'vuntside' süntaksit ümber muutujate, mille rakendus esitab, nagu ennustuse tekst: `{{}}`. Seal on ka vorm, mis postitab ennustuse `/predict` marsruudile.

    Lõpuks oled valmis ehitama Python-faili, mis juhib mudeli tarbimist ja ennustuste kuvamist:

1. Lisa `app.py` faili:

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

    > 💡 Näpunäide: kui lisad [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) Flaski veebirakenduse käivitamisel, kajastuvad kõik muudatused, mida teed oma rakenduses, kohe, ilma et peaksid serverit taaskäivitama. Ole ettevaatlik! Ära luba seda režiimi tootmisrakenduses.

Kui käivitad `python app.py` või `python3 app.py` - sinu veebiserver käivitub kohapeal ja saad täita lühikese vormi, et saada vastus oma põletavale küsimusele UFO-vaatluste kohta!

Enne seda vaata `app.py` osi:

1. Kõigepealt laaditakse sõltuvused ja rakendus käivitub.
1. Seejärel imporditakse mudel.
1. Seejärel renderdatakse index.html kodumarsruudil.

`/predict` marsruudil juhtub mitu asja, kui vorm postitatakse:

1. Vormimuutujad kogutakse ja konverteeritakse numpy massiiviks. Seejärel saadetakse need mudelile ja tagastatakse ennustus.
2. Riigid, mida soovime kuvada, renderdatakse uuesti loetava tekstina nende ennustatud riigikoodist, ja see väärtus saadetakse tagasi index.html-le, et see mallis renderdada.

Mudeli kasutamine sel viisil, Flaski ja marineeritud mudeliga, on suhteliselt lihtne. Kõige raskem on mõista, millises vormis peavad andmed olema, et neid mudelile saata ja ennustust saada. See kõik sõltub sellest, kuidas mudel treeniti. Sellel mudelil on kolm andmepunkti, mida tuleb sisestada, et saada ennustus.

Professionaalses keskkonnas näed, kui oluline on hea kommunikatsioon nende inimeste vahel, kes mudelit treenivad, ja nende vahel, kes seda veebis või mobiilirakenduses tarbivad. Meie puhul on see ainult üks inimene, sina!

---

## 🚀 Väljakutse

Selle asemel, et töötada sülearvutis ja importida mudel Flaski rakendusse, võiksid treenida mudeli otse Flaski rakenduses! Proovi konverteerida oma Python-koodi sülearvutis, võib-olla pärast andmete puhastamist, et treenida mudelit otse rakenduses marsruudil `train`. Millised on selle meetodi plussid ja miinused?

## [Loengu-järgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õppimine

Veebirakenduse loomiseks ML-mudelite tarbimiseks on mitmeid viise. Koosta nimekiri viisidest, kuidas saaksid kasutada JavaScripti või Pythonit veebirakenduse loomiseks, et kasutada masinõpet. Mõtle arhitektuurile: kas mudel peaks jääma rakendusse või elama pilves? Kui viimane, siis kuidas sellele ligi pääseda? Joonista arhitektuurimudel rakendatud ML-veebilahenduse jaoks.

## Ülesanne

[Proovi teistsugust mudelit](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.