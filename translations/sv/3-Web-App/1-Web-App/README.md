<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T21:47:01+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "sv"
}
-->
# Bygg en webbapp för att använda en ML-modell

I denna lektion kommer du att träna en ML-modell på en dataset som är utomjordisk: _UFO-observationer under det senaste århundradet_, hämtad från NUFORC:s databas.

Du kommer att lära dig:

- Hur man 'picklar' en tränad modell
- Hur man använder den modellen i en Flask-app

Vi kommer att fortsätta använda notebooks för att rensa data och träna vår modell, men du kan ta processen ett steg längre genom att utforska hur man använder en modell "i det vilda", så att säga: i en webbapp.

För att göra detta behöver du bygga en webbapp med Flask.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Bygga en app

Det finns flera sätt att bygga webbappar för att använda maskininlärningsmodeller. Din webbarkitektur kan påverka hur din modell tränas. Föreställ dig att du arbetar i ett företag där dataanalysgruppen har tränat en modell som de vill att du ska använda i en app.

### Överväganden

Det finns många frågor du behöver ställa:

- **Är det en webbapp eller en mobilapp?** Om du bygger en mobilapp eller behöver använda modellen i ett IoT-sammanhang kan du använda [TensorFlow Lite](https://www.tensorflow.org/lite/) och använda modellen i en Android- eller iOS-app.
- **Var kommer modellen att finnas?** I molnet eller lokalt?
- **Offline-stöd.** Måste appen fungera offline?
- **Vilken teknik användes för att träna modellen?** Den valda tekniken kan påverka vilka verktyg du behöver använda.
    - **Använda TensorFlow.** Om du tränar en modell med TensorFlow, till exempel, erbjuder det ekosystemet möjligheten att konvertera en TensorFlow-modell för användning i en webbapp med [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Använda PyTorch.** Om du bygger en modell med ett bibliotek som [PyTorch](https://pytorch.org/) har du möjlighet att exportera den i [ONNX](https://onnx.ai/) (Open Neural Network Exchange)-format för användning i JavaScript-webbappar som kan använda [Onnx Runtime](https://www.onnxruntime.ai/). Detta alternativ kommer att utforskas i en framtida lektion för en Scikit-learn-tränad modell.
    - **Använda Lobe.ai eller Azure Custom Vision.** Om du använder ett ML SaaS-system (Software as a Service) som [Lobe.ai](https://lobe.ai/) eller [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) för att träna en modell, erbjuder denna typ av mjukvara sätt att exportera modellen för många plattformar, inklusive att bygga ett skräddarsytt API som kan frågas i molnet av din onlineapplikation.

Du har också möjlighet att bygga en hel Flask-webbapp som kan träna modellen direkt i en webbläsare. Detta kan också göras med TensorFlow.js i en JavaScript-kontext.

För våra ändamål, eftersom vi har arbetat med Python-baserade notebooks, låt oss utforska stegen du behöver ta för att exportera en tränad modell från en sådan notebook till ett format som kan läsas av en Python-byggd webbapp.

## Verktyg

För denna uppgift behöver du två verktyg: Flask och Pickle, båda körs på Python.

✅ Vad är [Flask](https://palletsprojects.com/p/flask/)? Flask definieras som ett 'mikroramverk' av sina skapare och erbjuder de grundläggande funktionerna för webbramverk med Python och en mallmotor för att bygga webbsidor. Ta en titt på [denna Learn-modul](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) för att öva på att bygga med Flask.

✅ Vad är [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 är en Python-modul som serialiserar och deserialiserar en Python-objektstruktur. När du 'picklar' en modell serialiserar eller plattar du ut dess struktur för användning på webben. Var försiktig: pickle är inte intrinsiskt säker, så var försiktig om du blir ombedd att 'un-pickla' en fil. En picklad fil har suffixet `.pkl`.

## Övning - rensa din data

I denna lektion kommer du att använda data från 80 000 UFO-observationer, insamlade av [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Denna data har några intressanta beskrivningar av UFO-observationer, till exempel:

- **Lång exempelbeskrivning.** "En man kommer ut från en ljusstråle som lyser på en gräsbevuxen fält på natten och han springer mot Texas Instruments parkeringsplats".
- **Kort exempelbeskrivning.** "ljusen jagade oss".

Kalkylbladet [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) innehåller kolumner om `city`, `state` och `country` där observationen inträffade, objektets `shape` samt dess `latitude` och `longitude`.

I den tomma [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) som ingår i denna lektion:

1. importera `pandas`, `matplotlib` och `numpy` som du gjorde i tidigare lektioner och importera UFO-kalkylbladet. Du kan ta en titt på ett exempeldata:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Konvertera UFO-datan till en liten dataframe med nya titlar. Kontrollera de unika värdena i fältet `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Nu kan du minska mängden data vi behöver hantera genom att ta bort eventuella null-värden och endast importera observationer mellan 1-60 sekunder:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importera Scikit-learns `LabelEncoder`-bibliotek för att konvertera textvärden för länder till ett nummer:

    ✅ LabelEncoder kodar data alfabetiskt

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Din data bör se ut så här:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Övning - bygg din modell

Nu kan du förbereda dig för att träna en modell genom att dela upp datan i tränings- och testgrupper.

1. Välj de tre funktioner du vill träna på som din X-vektor, och y-vektorn kommer att vara `Country`. Du vill kunna mata in `Seconds`, `Latitude` och `Longitude` och få ett land-id som returneras.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Träna din modell med logistisk regression:

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

Noggrannheten är inte dålig **(runt 95%)**, vilket inte är förvånande, eftersom `Country` och `Latitude/Longitude` korrelerar.

Modellen du skapade är inte särskilt revolutionerande eftersom du borde kunna dra slutsatsen ett `Country` från dess `Latitude` och `Longitude`, men det är en bra övning att försöka träna från rådata som du rensade, exporterade och sedan använda denna modell i en webbapp.

## Övning - 'pickla' din modell

Nu är det dags att _pickla_ din modell! Du kan göra det med några rader kod. När den är _picklad_, ladda din picklade modell och testa den mot en exempeldata-array som innehåller värden för sekunder, latitud och longitud,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modellen returnerar **'3'**, vilket är landskoden för Storbritannien. Galet! 👽

## Övning - bygg en Flask-app

Nu kan du bygga en Flask-app för att kalla din modell och returnera liknande resultat, men på ett mer visuellt tilltalande sätt.

1. Börja med att skapa en mapp som heter **web-app** bredvid filen _notebook.ipynb_ där din _ufo-model.pkl_-fil finns.

1. I den mappen skapar du tre ytterligare mappar: **static**, med en mapp **css** inuti, och **templates**. Du bör nu ha följande filer och kataloger:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Se lösningsmappen för en vy av den färdiga appen

1. Den första filen att skapa i _web-app_-mappen är **requirements.txt**-filen. Precis som _package.json_ i en JavaScript-app listar denna fil beroenden som krävs av appen. I **requirements.txt** lägger du till raderna:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Kör nu denna fil genom att navigera till _web-app_:

    ```bash
    cd web-app
    ```

1. I din terminal skriver du `pip install` för att installera biblioteken som listas i _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Nu är du redo att skapa tre ytterligare filer för att slutföra appen:

    1. Skapa **app.py** i roten.
    2. Skapa **index.html** i _templates_-katalogen.
    3. Skapa **styles.css** i _static/css_-katalogen.

1. Bygg ut _styles.css_-filen med några stilar:

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

1. Nästa steg är att bygga ut _index.html_-filen:

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

    Titta på mallningen i denna fil. Notera 'mustache'-syntaxen runt variabler som kommer att tillhandahållas av appen, som prediktionstexten: `{{}}`. Det finns också ett formulär som skickar en prediktion till `/predict`-routen.

    Slutligen är du redo att bygga Python-filen som driver konsumtionen av modellen och visningen av prediktioner:

1. I `app.py` lägger du till:

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

    > 💡 Tips: när du lägger till [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) medan du kör webbappen med Flask, kommer alla ändringar du gör i din applikation att återspeglas omedelbart utan att du behöver starta om servern. Var försiktig! Aktivera inte detta läge i en produktionsapp.

Om du kör `python app.py` eller `python3 app.py` - startar din webbserver lokalt, och du kan fylla i ett kort formulär för att få svar på din brinnande fråga om var UFO:n har observerats!

Innan du gör det, ta en titt på delarna av `app.py`:

1. Först laddas beroenden och appen startar.
1. Sedan importeras modellen.
1. Därefter renderas index.html på hemrouten.

På `/predict`-routen händer flera saker när formuläret skickas:

1. Formulärvariablerna samlas in och konverteras till en numpy-array. De skickas sedan till modellen och en prediktion returneras.
2. Länderna som vi vill visa renderas om som läsbar text från deras förutsagda landskod, och det värdet skickas tillbaka till index.html för att renderas i mallen.

Att använda en modell på detta sätt, med Flask och en picklad modell, är relativt enkelt. Det svåraste är att förstå vilken form datan måste ha som skickas till modellen för att få en prediktion. Det beror helt på hur modellen tränades. Denna har tre datapunkter som måste matas in för att få en prediktion.

I en professionell miljö kan du se hur viktig kommunikation är mellan de som tränar modellen och de som använder den i en webb- eller mobilapp. I vårt fall är det bara en person, du!

---

## 🚀 Utmaning

Istället för att arbeta i en notebook och importera modellen till Flask-appen, kan du träna modellen direkt i Flask-appen! Försök att konvertera din Python-kod i notebooken, kanske efter att din data har rensats, för att träna modellen direkt i appen på en route som heter `train`. Vilka är för- och nackdelarna med att använda denna metod?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Det finns många sätt att bygga en webbapp för att använda ML-modeller. Gör en lista över sätt du kan använda JavaScript eller Python för att bygga en webbapp som utnyttjar maskininlärning. Tänk på arkitektur: bör modellen stanna i appen eller finnas i molnet? Om det senare, hur skulle du komma åt den? Rita en arkitekturmodell för en tillämpad ML-webblösning.

## Uppgift

[Prova en annan modell](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.