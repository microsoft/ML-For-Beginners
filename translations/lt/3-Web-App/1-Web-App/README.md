<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T07:57:52+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "lt"
}
-->
# Sukurkite internetinę programą, naudojančią ML modelį

Šioje pamokoje treniruosite ML modelį su duomenų rinkiniu, kuris yra tiesiog neįtikėtinas: _NSO stebėjimai per pastarąjį šimtmetį_, gauti iš NUFORC duomenų bazės.

Jūs išmoksite:

- Kaip „užkonservuoti“ (pickle) treniruotą modelį
- Kaip naudoti tą modelį Flask programoje

Mes ir toliau naudosime užrašų knygeles duomenims valyti ir modeliams treniruoti, tačiau galite žengti dar vieną žingsnį toliau, tyrinėdami, kaip naudoti modelį „laukinėje gamtoje“, kitaip tariant, internetinėje programoje.

Norėdami tai padaryti, turite sukurti internetinę programą naudodami Flask.

## [Prieš paskaitą – testas](https://ff-quizzes.netlify.app/en/ml/)

## Programos kūrimas

Yra keletas būdų, kaip sukurti internetines programas, kurios naudoja mašininio mokymosi modelius. Jūsų interneto architektūra gali turėti įtakos tam, kaip modelis yra treniruojamas. Įsivaizduokite, kad dirbate įmonėje, kur duomenų mokslininkų grupė sukūrė modelį, kurį jie nori, kad jūs panaudotumėte programoje.

### Svarstymai

Yra daug klausimų, kuriuos reikia užduoti:

- **Ar tai internetinė programa ar mobilioji programa?** Jei kuriate mobiliąją programą arba norite naudoti modelį IoT kontekste, galite naudoti [TensorFlow Lite](https://www.tensorflow.org/lite/) ir modelį integruoti į Android arba iOS programą.
- **Kur bus laikomas modelis?** Debesyje ar vietoje?
- **Darbas neprisijungus.** Ar programa turi veikti neprisijungus?
- **Kokia technologija buvo naudojama modelio treniravimui?** Pasirinkta technologija gali turėti įtakos įrankiams, kuriuos reikia naudoti.
    - **Naudojant TensorFlow.** Jei treniruojate modelį naudodami TensorFlow, pavyzdžiui, ši ekosistema suteikia galimybę konvertuoti TensorFlow modelį naudoti internetinėje programoje naudojant [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Naudojant PyTorch.** Jei kuriate modelį naudodami biblioteką, tokią kaip [PyTorch](https://pytorch.org/), turite galimybę eksportuoti jį [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formatu, kad galėtumėte naudoti JavaScript internetinėse programose, kurios naudoja [Onnx Runtime](https://www.onnxruntime.ai/). Ši galimybė bus nagrinėjama būsimoje pamokoje, skirtoje Scikit-learn treniruotam modeliui.
    - **Naudojant Lobe.ai arba Azure Custom Vision.** Jei naudojate ML SaaS (programinė įranga kaip paslauga) sistemą, tokią kaip [Lobe.ai](https://lobe.ai/) arba [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott), šio tipo programinė įranga suteikia galimybes eksportuoti modelį įvairioms platformoms, įskaitant API kūrimą, kurį galima užklausyti debesyje jūsų internetinėje programoje.

Taip pat turite galimybę sukurti visą Flask internetinę programą, kuri galėtų pati treniruoti modelį interneto naršyklėje. Tai taip pat galima padaryti naudojant TensorFlow.js JavaScript kontekste.

Mūsų tikslams, kadangi dirbome su Python pagrindu sukurtomis užrašų knygelėmis, panagrinėkime žingsnius, kuriuos reikia atlikti norint eksportuoti treniruotą modelį iš tokios užrašų knygelės į formatą, kurį gali perskaityti Python sukurta internetinė programa.

## Įrankiai

Šiai užduočiai jums reikės dviejų įrankių: Flask ir Pickle, abu veikia su Python.

✅ Kas yra [Flask](https://palletsprojects.com/p/flask/)? Flask kūrėjai apibūdina kaip „mikro-framework“, kuris suteikia pagrindines interneto sistemų funkcijas naudojant Python ir šablonų variklį interneto puslapiams kurti. Pažvelkite į [šį mokymosi modulį](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), kad galėtumėte praktikuotis kurdami su Flask.

✅ Kas yra [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 yra Python modulis, kuris serializuoja ir de-serializuoja Python objektų struktūrą. Kai „užkonservuojate“ modelį, jūs serializuojate arba „išlyginate“ jo struktūrą, kad galėtumėte naudoti internete. Būkite atsargūs: Pickle nėra iš esmės saugus, todėl būkite atsargūs, jei jums siūloma „iškonservuoti“ failą. „Užkonservuotas“ failas turi `.pkl` plėtinį.

## Užduotis – išvalykite savo duomenis

Šioje pamokoje naudosite duomenis apie 80,000 NSO stebėjimų, surinktų [NUFORC](https://nuforc.org) (Nacionalinis NSO pranešimų centras). Šiuose duomenyse yra įdomių NSO stebėjimų aprašymų, pavyzdžiui:

- **Ilgas aprašymo pavyzdys.** „Vyras išlenda iš šviesos spindulio, kuris šviečia ant žolėtos pievos naktį, ir bėga link Texas Instruments automobilių stovėjimo aikštelės“.
- **Trumpas aprašymo pavyzdys.** „Šviesos mus vijosi“.

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) skaičiuoklėje yra stulpeliai apie `miestą`, `valstiją` ir `šalį`, kurioje įvyko stebėjimas, objekto `formą` bei jo `platumą` ir `ilgumą`.

Tuščioje [užrašų knygelėje](../../../../3-Web-App/1-Web-App/notebook.ipynb), pridėtoje prie šios pamokos:

1. importuokite `pandas`, `matplotlib` ir `numpy`, kaip tai darėte ankstesnėse pamokose, ir importuokite NSO skaičiuoklę. Galite peržiūrėti pavyzdinį duomenų rinkinį:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konvertuokite NSO duomenis į mažą duomenų rėmelį su naujais pavadinimais. Patikrinkite unikalius `Šalis` lauko reikšmes.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Dabar galite sumažinti duomenų kiekį, su kuriuo reikia dirbti, pašalindami bet kokias tuščias reikšmes ir importuodami tik stebėjimus, kurie truko nuo 1 iki 60 sekundžių:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importuokite Scikit-learn bibliotekos `LabelEncoder`, kad konvertuotumėte tekstines šalių reikšmes į skaičius:

    ✅ LabelEncoder koduoja duomenis abėcėlės tvarka

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Jūsų duomenys turėtų atrodyti taip:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Užduotis – sukurkite savo modelį

Dabar galite pasiruošti treniruoti modelį, padalindami duomenis į treniravimo ir testavimo grupes.

1. Pasirinkite tris funkcijas, kurias norite treniruoti kaip savo X vektorių, o y vektorius bus `Šalis`. Jūs norite įvesti `Sekundės`, `Platuma` ir `Ilguma` ir gauti šalies ID, kurį grąžinsite.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Treniruokite savo modelį naudodami logistinę regresiją:

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

Tikslumas nėra blogas **(apie 95%)**, nenuostabu, nes `Šalis` ir `Platuma/Ilguma` koreliuoja.

Sukurtas modelis nėra labai revoliucinis, nes turėtumėte sugebėti nustatyti `Šalį` pagal jos `Platuma` ir `Ilguma`, tačiau tai yra gera praktika treniruoti modelį iš neapdorotų duomenų, kuriuos išvalėte, eksportavote ir tada naudojote šį modelį internetinėje programoje.

## Užduotis – „užkonservuokite“ savo modelį

Dabar laikas _užkonservuoti_ savo modelį! Tai galite padaryti keliose kodo eilutėse. Kai modelis bus _užkonservuotas_, įkelkite jį ir išbandykite su pavyzdiniu duomenų masyvu, kuriame yra reikšmės sekundėms, platumai ir ilgumai.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modelis grąžina **'3'**, kuris yra JK šalies kodas. Neįtikėtina! 👽

## Užduotis – sukurkite Flask programą

Dabar galite sukurti Flask programą, kuri iškviečia jūsų modelį ir grąžina panašius rezultatus, tačiau vizualiai patrauklesniu būdu.

1. Pradėkite sukurdami aplanką **web-app** šalia _notebook.ipynb_ failo, kur yra jūsų _ufo-model.pkl_ failas.

1. Tame aplanke sukurkite dar tris aplankus: **static**, su aplanku **css** viduje, ir **templates**. Dabar turėtumėte turėti šiuos failus ir katalogus:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Žiūrėkite sprendimų aplanką, kad pamatytumėte baigtos programos vaizdą

1. Pirmasis failas, kurį reikia sukurti _web-app_ aplanke, yra **requirements.txt** failas. Kaip _package.json_ JavaScript programoje, šis failas nurodo priklausomybes, kurių reikia programai. Į **requirements.txt** pridėkite eilutes:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Dabar paleiskite šį failą, naršydami į _web-app_:

    ```bash
    cd web-app
    ```

1. Savo terminale įveskite `pip install`, kad įdiegtumėte bibliotėkas, nurodytas _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Dabar esate pasiruošę sukurti dar tris failus, kad užbaigtumėte programą:

    1. Sukurkite **app.py** šakniniame kataloge.
    2. Sukurkite **index.html** _templates_ kataloge.
    3. Sukurkite **styles.css** _static/css_ kataloge.

1. Sukurkite _styles.css_ failą su keliais stiliais:

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

1. Toliau sukurkite _index.html_ failą:

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

    Pažvelkite į šablonų sintaksę šiame faile. Atkreipkite dėmesį į „ūsų“ sintaksę aplink kintamuosius, kuriuos pateiks programa, pvz., prognozės tekstą: `{{}}`. Taip pat yra forma, kuri siunčia prognozę į `/predict` maršrutą.

    Galiausiai esate pasiruošę sukurti Python failą, kuris valdo modelio naudojimą ir prognozių rodymą:

1. Į `app.py` pridėkite:

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

    > 💡 Patarimas: kai pridedate [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) paleisdami internetinę programą naudodami Flask, bet kokie pakeitimai, kuriuos atliksite savo programoje, bus iškart matomi be poreikio iš naujo paleisti serverį. Atsargiai! Neįjunkite šio režimo gamybinėje programoje.

Jei paleisite `python app.py` arba `python3 app.py`, jūsų interneto serveris paleidžiamas vietoje, ir galite užpildyti trumpą formą, kad gautumėte atsakymą į savo degantį klausimą apie tai, kur buvo pastebėti NSO!

Prieš tai padarydami, pažvelkite į `app.py` dalis:

1. Pirmiausia įkeliamos priklausomybės ir paleidžiama programa.
1. Tada importuojamas modelis.
1. Tada pagrindiniame maršrute pateikiamas index.html.

Maršrute `/predict`, kai forma pateikiama, vyksta keli dalykai:

1. Formos kintamieji surenkami ir konvertuojami į numpy masyvą. Jie siunčiami į modelį, ir grąžinama prognozė.
2. Šalys, kurias norime rodyti, paverčiamos į skaitomą tekstą iš jų prognozuoto šalies kodo, ir ta reikšmė grąžinama į index.html, kad būtų pateikta šablone.

Naudoti modelį tokiu būdu, su Flask ir užkonservuotu modeliu, yra gana paprasta. Sunkiausia yra suprasti, kokios formos duomenys turi būti siunčiami į modelį, kad gautumėte prognozę. Tai visiškai priklauso nuo to, kaip modelis buvo treniruotas. Šis modelis turi tris duomenų taškus, kuriuos reikia įvesti, kad gautumėte prognozę.

Profesinėje aplinkoje galite matyti, kaip svarbu gerai komunikuoti tarp žmonių, kurie treniruoja modelį, ir tų, kurie jį naudoja internetinėje ar mobiliojoje programoje. Mūsų atveju tai tik vienas žmogus – jūs!

---

## 🚀 Iššūkis

Užuot dirbę užrašų knygelėje ir importavę modelį į Flask programą, galite treniruoti modelį tiesiogiai Flask programoje! Pabandykite konvertuoti savo Python kodą užrašų knygelėje, galbūt po to, kai jūsų duomenys yra išvalyti, kad treniruotumėte modelį tiesiogiai programoje maršrute, vadinamame `train`. Kokie yra šio metodo privalumai ir trūkumai?

## [Po paskaitos – testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Yra daug būdų, kaip sukurti internetinę programą, kuri naudoja ML modelius. Sudarykite sąrašą būdų, kaip galite naudoti JavaScript arba Python, kad sukurtumėte internetinę programą, kuri naudoja mašininį mokymąsi. Apsvarstykite architektūrą: ar modelis turėtų likti programoje, ar gyventi debesyje? Jei pastarasis, kaip jį pasiektumėte? Nubraižykite architektūrinį modelį taikomos ML internetinės sprendimo.

## Užduotis

[Išbandykite kitą modelį](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.