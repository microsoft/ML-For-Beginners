<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T12:58:07+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "hr"
}
-->
# Izgradnja web aplikacije za korištenje ML modela

U ovoj lekciji, trenirat ćete ML model na skupu podataka koji je doslovno van ovog svijeta: _viđenja NLO-a tijekom prošlog stoljeća_, preuzetih iz NUFORC baze podataka.

Naučit ćete:

- Kako 'pickle-ati' trenirani model
- Kako koristiti taj model u Flask aplikaciji

Nastavit ćemo koristiti bilježnice za čišćenje podataka i treniranje modela, ali možete napraviti korak dalje istražujući kako koristiti model "u divljini", da tako kažemo: u web aplikaciji.

Da biste to učinili, trebate izgraditi web aplikaciju koristeći Flask.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Izgradnja aplikacije

Postoji nekoliko načina za izgradnju web aplikacija koje koriste modele strojnog učenja. Vaša web arhitektura može utjecati na način na koji je model treniran. Zamislite da radite u tvrtki gdje je tim za podatkovnu znanost trenirao model koji žele da koristite u aplikaciji.

### Razmatranja

Postoji mnogo pitanja koja trebate postaviti:

- **Je li to web aplikacija ili mobilna aplikacija?** Ako gradite mobilnu aplikaciju ili trebate koristiti model u IoT kontekstu, možete koristiti [TensorFlow Lite](https://www.tensorflow.org/lite/) i koristiti model u Android ili iOS aplikaciji.
- **Gdje će model biti smješten?** U oblaku ili lokalno?
- **Podrška za offline rad.** Mora li aplikacija raditi offline?
- **Koja tehnologija je korištena za treniranje modela?** Odabrana tehnologija može utjecati na alate koje trebate koristiti.
    - **Korištenje TensorFlow-a.** Ako trenirate model koristeći TensorFlow, na primjer, taj ekosustav omogućuje konverziju TensorFlow modela za korištenje u web aplikaciji pomoću [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Korištenje PyTorch-a.** Ako gradite model koristeći biblioteku poput [PyTorch](https://pytorch.org/), imate opciju izvesti ga u [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formatu za korištenje u JavaScript web aplikacijama koje mogu koristiti [Onnx Runtime](https://www.onnxruntime.ai/). Ova opcija će biti istražena u budućoj lekciji za model treniran pomoću Scikit-learn.
    - **Korištenje Lobe.ai ili Azure Custom Vision.** Ako koristite ML SaaS (Software as a Service) sustav poput [Lobe.ai](https://lobe.ai/) ili [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) za treniranje modela, ovaj tip softvera pruža načine za izvoz modela za mnoge platforme, uključujući izgradnju prilagođenog API-ja koji se može upitati u oblaku putem vaše online aplikacije.

Također imate priliku izgraditi cijelu Flask web aplikaciju koja bi mogla trenirati model unutar web preglednika. To se također može učiniti koristeći TensorFlow.js u JavaScript kontekstu.

Za naše potrebe, budući da radimo s bilježnicama temeljenim na Pythonu, istražimo korake koje trebate poduzeti kako biste izvezli trenirani model iz takve bilježnice u format čitljiv Python-izgrađenoj web aplikaciji.

## Alat

Za ovaj zadatak trebate dva alata: Flask i Pickle, oba se pokreću na Pythonu.

✅ Što je [Flask](https://palletsprojects.com/p/flask/)? Definiran kao 'mikro-okvir' od strane svojih kreatora, Flask pruža osnovne značajke web okvira koristeći Python i motor za predloške za izgradnju web stranica. Pogledajte [ovaj modul za učenje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) kako biste vježbali izgradnju s Flaskom.

✅ Što je [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 je Python modul koji serijalizira i de-serijalizira strukturu Python objekta. Kada 'pickle-ate' model, serijalizirate ili "spljoštavate" njegovu strukturu za korištenje na webu. Budite oprezni: pickle nije inherentno siguran, pa budite oprezni ako vas se potiče da 'un-pickle-ate' datoteku. Pickle-ana datoteka ima sufiks `.pkl`.

## Vježba - očistite svoje podatke

U ovoj lekciji koristit ćete podatke iz 80.000 viđenja NLO-a, prikupljenih od strane [NUFORC](https://nuforc.org) (Nacionalni centar za prijavu NLO-a). Ovi podaci imaju zanimljive opise viđenja NLO-a, na primjer:

- **Dugi opis primjera.** "Čovjek izlazi iz zrake svjetlosti koja obasjava travnato polje noću i trči prema parkiralištu Texas Instrumentsa".
- **Kratki opis primjera.** "svjetla su nas progonila".

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) proračunska tablica uključuje stupce o `grad`, `država` i `zemlja` gdje se viđenje dogodilo, oblik objekta (`shape`) te njegovu `širinu` i `dužinu`.

U praznoj [bilježnici](../../../../3-Web-App/1-Web-App/notebook.ipynb) uključene u ovu lekciju:

1. importirajte `pandas`, `matplotlib` i `numpy` kao što ste to učinili u prethodnim lekcijama i importirajte proračunsku tablicu ufos. Možete pogledati uzorak skupa podataka:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Pretvorite podatke ufos u mali dataframe sa svježim naslovima. Provjerite jedinstvene vrijednosti u polju `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Sada možete smanjiti količinu podataka s kojima trebamo raditi tako da odbacite sve null vrijednosti i uvezete samo viđenja između 1-60 sekundi:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importirajte Scikit-learnovu biblioteku `LabelEncoder` za pretvaranje tekstualnih vrijednosti za zemlje u broj:

    ✅ LabelEncoder kodira podatke abecedno

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Vaši podaci trebali bi izgledati ovako:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Vježba - izgradite svoj model

Sada se možete pripremiti za treniranje modela dijeljenjem podataka u skup za treniranje i testiranje.

1. Odaberite tri značajke na kojima želite trenirati kao svoj X vektor, a y vektor će biti `Country`. Želite moći unijeti `Seconds`, `Latitude` i `Longitude` i dobiti ID zemlje kao rezultat.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Trenirajte svoj model koristeći logističku regresiju:

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

Točnost nije loša **(oko 95%)**, što nije iznenađujuće, jer `Country` i `Latitude/Longitude` koreliraju.

Model koji ste kreirali nije baš revolucionaran jer biste trebali moći zaključiti `Country` iz njegovih `Latitude` i `Longitude`, ali ovo je dobra vježba za pokušaj treniranja od sirovih podataka koje ste očistili, izvezli, a zatim koristili ovaj model u web aplikaciji.

## Vježba - 'pickle-ajte' svoj model

Sada je vrijeme da _pickle-ate_ svoj model! To možete učiniti u nekoliko linija koda. Jednom kada je _pickle-an_, učitajte svoj pickle-ani model i testirajte ga na uzorku podataka koji sadrži vrijednosti za sekunde, širinu i dužinu.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model vraća **'3'**, što je kod zemlje za UK. Nevjerojatno! 👽

## Vježba - izgradite Flask aplikaciju

Sada možete izgraditi Flask aplikaciju koja poziva vaš model i vraća slične rezultate, ali na vizualno privlačniji način.

1. Započnite stvaranjem mape nazvane **web-app** pored datoteke _notebook.ipynb_ gdje se nalazi vaša datoteka _ufo-model.pkl_.

1. U toj mapi stvorite još tri mape: **static**, s mapom **css** unutar nje, i **templates**. Sada biste trebali imati sljedeće datoteke i direktorije:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Pogledajte mapu rješenja za pregled gotove aplikacije

1. Prva datoteka koju treba stvoriti u mapi _web-app_ je datoteka **requirements.txt**. Kao _package.json_ u JavaScript aplikaciji, ova datoteka navodi ovisnosti potrebne za aplikaciju. U **requirements.txt** dodajte linije:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Sada pokrenite ovu datoteku navigacijom do _web-app_:

    ```bash
    cd web-app
    ```

1. U svom terminalu upišite `pip install`, kako biste instalirali biblioteke navedene u _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Sada ste spremni stvoriti još tri datoteke za dovršetak aplikacije:

    1. Stvorite **app.py** u korijenu.
    2. Stvorite **index.html** u direktoriju _templates_.
    3. Stvorite **styles.css** u direktoriju _static/css_.

1. Izgradite datoteku _styles.css_ s nekoliko stilova:

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

1. Zatim izgradite datoteku _index.html_:

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

    Pogledajte predloške u ovoj datoteci. Primijetite 'mustache' sintaksu oko varijabli koje će biti pružene od strane aplikacije, poput teksta predikcije: `{{}}`. Tu je i obrazac koji šalje predikciju na rutu `/predict`.

    Konačno, spremni ste izgraditi Python datoteku koja pokreće korištenje modela i prikaz predikcija:

1. U `app.py` dodajte:

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

    > 💡 Savjet: kada dodate [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) dok pokrećete web aplikaciju koristeći Flask, sve promjene koje napravite u svojoj aplikaciji bit će odmah vidljive bez potrebe za ponovnim pokretanjem servera. Pazite! Nemojte omogućiti ovaj način rada u produkcijskoj aplikaciji.

Ako pokrenete `python app.py` ili `python3 app.py` - vaš web server se pokreće lokalno, i možete ispuniti kratki obrazac kako biste dobili odgovor na svoje goruće pitanje o tome gdje su NLO-i viđeni!

Prije nego što to učinite, pogledajte dijelove `app.py`:

1. Prvo se učitavaju ovisnosti i aplikacija se pokreće.
1. Zatim se model importira.
1. Zatim se index.html prikazuje na početnoj ruti.

Na ruti `/predict`, nekoliko stvari se događa kada se obrazac pošalje:

1. Varijable obrasca se prikupljaju i pretvaraju u numpy niz. Zatim se šalju modelu i vraća se predikcija.
2. Zemlje koje želimo prikazati ponovno se prikazuju kao čitljiv tekst iz njihovog predviđenog koda zemlje, i ta vrijednost se vraća na index.html kako bi se prikazala u predlošku.

Korištenje modela na ovaj način, s Flaskom i pickle-anim modelom, relativno je jednostavno. Najteže je razumjeti kakvog oblika moraju biti podaci koji se šalju modelu kako bi se dobila predikcija. Sve ovisi o tome kako je model treniran. Ovaj ima tri podatkovne točke koje treba unijeti kako bi se dobila predikcija.

U profesionalnom okruženju, možete vidjeti koliko je dobra komunikacija važna između ljudi koji treniraju model i onih koji ga koriste u web ili mobilnoj aplikaciji. U našem slučaju, to je samo jedna osoba, vi!

---

## 🚀 Izazov

Umjesto rada u bilježnici i uvoza modela u Flask aplikaciju, mogli biste trenirati model direktno unutar Flask aplikacije! Pokušajte pretvoriti svoj Python kod iz bilježnice, možda nakon što su vaši podaci očišćeni, kako biste trenirali model unutar aplikacije na ruti nazvanoj `train`. Koji su prednosti i nedostaci ovog pristupa?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Postoji mnogo načina za izgradnju web aplikacije koja koristi ML modele. Napravite popis načina na koje biste mogli koristiti JavaScript ili Python za izgradnju web aplikacije koja koristi strojno učenje. Razmotrite arhitekturu: treba li model ostati u aplikaciji ili živjeti u oblaku? Ako je ovo drugo, kako biste mu pristupili? Nacrtajte arhitektonski model za primijenjeno ML web rješenje.

## Zadatak

[Isprobajte drugačiji model](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane čovjeka. Ne preuzimamo odgovornost za nesporazume ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.