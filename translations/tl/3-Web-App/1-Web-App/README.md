<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T18:19:59+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "tl"
}
-->
# Gumawa ng Web App para Gamitin ang ML Model

Sa araling ito, magtetrain ka ng ML model gamit ang isang data set na kakaiba: _mga sightings ng UFO sa nakaraang siglo_, na galing sa database ng NUFORC.

Matutunan mo:

- Paano mag-'pickle' ng isang trained model
- Paano gamitin ang model na iyon sa isang Flask app

Ipagpapatuloy natin ang paggamit ng notebooks para linisin ang data at i-train ang model, ngunit maaari mong dalhin ang proseso sa susunod na antas sa pamamagitan ng paggamit ng model sa isang web app.

Para magawa ito, kailangan mong gumawa ng web app gamit ang Flask.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Pagbuo ng App

Maraming paraan para gumawa ng web apps na gumagamit ng machine learning models. Ang arkitektura ng iyong web app ay maaaring makaapekto sa paraan ng pag-train ng iyong model. Isipin na nagtatrabaho ka sa isang negosyo kung saan ang grupo ng data science ay nagtetrain ng model na nais nilang gamitin mo sa isang app.

### Mga Dapat Isaalang-alang

Maraming tanong ang kailangang sagutin:

- **Web app ba o mobile app?** Kung gumagawa ka ng mobile app o kailangang gamitin ang model sa IoT context, maaari mong gamitin ang [TensorFlow Lite](https://www.tensorflow.org/lite/) para magamit ang model sa Android o iOS app.
- **Saan ilalagay ang model?** Sa cloud o lokal?
- **Offline support.** Kailangan bang gumana ang app offline?
- **Anong teknolohiya ang ginamit para i-train ang model?** Ang napiling teknolohiya ay maaaring makaapekto sa mga tool na kailangang gamitin.
    - **Gamit ang TensorFlow.** Kung nagtrain ka ng model gamit ang TensorFlow, halimbawa, ang ecosystem nito ay nagbibigay ng kakayahang i-convert ang TensorFlow model para magamit sa web app gamit ang [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Gamit ang PyTorch.** Kung gumagawa ka ng model gamit ang library tulad ng [PyTorch](https://pytorch.org/), may opsyon kang i-export ito sa [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format para magamit sa JavaScript web apps na maaaring gumamit ng [Onnx Runtime](https://www.onnxruntime.ai/). Ang opsyon na ito ay tatalakayin sa susunod na aralin para sa isang Scikit-learn-trained model.
    - **Gamit ang Lobe.ai o Azure Custom Vision.** Kung gumagamit ka ng ML SaaS (Software as a Service) system tulad ng [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para magtrain ng model, ang ganitong uri ng software ay nagbibigay ng paraan para i-export ang model para sa iba't ibang platform, kabilang ang paggawa ng bespoke API na maaaring i-query sa cloud ng iyong online application.

Mayroon ka ring pagkakataon na gumawa ng buong Flask web app na maaaring magtrain ng model mismo sa web browser. Maaari rin itong gawin gamit ang TensorFlow.js sa JavaScript context.

Para sa ating layunin, dahil gumagamit tayo ng Python-based notebooks, tuklasin natin ang mga hakbang na kailangang gawin para i-export ang isang trained model mula sa notebook patungo sa format na mababasa ng Python-built web app.

## Tool

Para sa gawaing ito, kailangan mo ng dalawang tool: Flask at Pickle, na parehong tumatakbo sa Python.

✅ Ano ang [Flask](https://palletsprojects.com/p/flask/)? Tinukoy bilang isang 'micro-framework' ng mga tagalikha nito, ang Flask ay nagbibigay ng mga pangunahing tampok ng web frameworks gamit ang Python at isang templating engine para gumawa ng web pages. Tingnan ang [Learn module na ito](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para magsanay sa paggawa gamit ang Flask.

✅ Ano ang [Pickle](https://docs.python.org/3/library/pickle.html)? Ang Pickle 🥒 ay isang Python module na nagseserialize at nagde-deserialize ng Python object structure. Kapag nag-'pickle' ka ng model, sineserialize o pinaplat ang istruktura nito para magamit sa web. Mag-ingat: ang pickle ay hindi intrinsically secure, kaya mag-ingat kung hinihikayat kang mag-'un-pickle' ng file. Ang pickled file ay may suffix na `.pkl`.

## Ehersisyo - linisin ang iyong data

Sa araling ito gagamit ka ng data mula sa 80,000 sightings ng UFO, na nakalap ng [NUFORC](https://nuforc.org) (The National UFO Reporting Center). Ang data na ito ay may mga kawili-wiling deskripsyon ng sightings ng UFO, halimbawa:

- **Mahabang halimbawa ng deskripsyon.** "Isang lalaki ang lumabas mula sa sinag ng liwanag na tumama sa damuhan sa gabi at tumakbo patungo sa parking lot ng Texas Instruments".
- **Maikling halimbawa ng deskripsyon.** "hinabol kami ng mga ilaw".

Ang [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) spreadsheet ay may mga column tungkol sa `city`, `state` at `country` kung saan nangyari ang sighting, ang `shape` ng object, at ang `latitude` at `longitude` nito.

Sa blangkong [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) na kasama sa araling ito:

1. I-import ang `pandas`, `matplotlib`, at `numpy` tulad ng ginawa mo sa mga nakaraang aralin at i-import ang ufos spreadsheet. Maaari mong tingnan ang sample data set:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. I-convert ang ufos data sa isang maliit na dataframe na may bagong mga pamagat. Tingnan ang mga unique na halaga sa field na `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ngayon, maaari mong bawasan ang dami ng data na kailangang i-proseso sa pamamagitan ng pag-drop ng anumang null values at pag-import lamang ng sightings na tumagal ng 1-60 segundo:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. I-import ang Scikit-learn's `LabelEncoder` library para i-convert ang text values ng mga bansa sa numero:

    ✅ Ang LabelEncoder ay nag-eencode ng data alphabetically

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Ang iyong data ay dapat magmukhang ganito:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Ehersisyo - gumawa ng iyong model

Ngayon maaari kang maghanda para magtrain ng model sa pamamagitan ng paghahati ng data sa training at testing group.

1. Piliin ang tatlong features na gusto mong i-train bilang iyong X vector, at ang y vector ay ang `Country`. Gusto mong makapag-input ng `Seconds`, `Latitude` at `Longitude` at makakuha ng country id bilang output.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. I-train ang iyong model gamit ang logistic regression:

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

Ang accuracy ay hindi masama **(mga 95%)**, na hindi nakakagulat, dahil ang `Country` at `Latitude/Longitude` ay may kaugnayan.

Ang model na ginawa mo ay hindi masyadong rebolusyonaryo dahil dapat mong ma-infer ang `Country` mula sa `Latitude` at `Longitude`, ngunit ito ay isang magandang ehersisyo para subukang magtrain mula sa raw data na nilinis, in-export, at pagkatapos ay gamitin ang model na ito sa isang web app.

## Ehersisyo - 'pickle' ang iyong model

Ngayon, oras na para _i-pickle_ ang iyong model! Magagawa mo ito sa ilang linya ng code. Kapag ito ay _na-pickle_, i-load ang iyong pickled model at subukan ito laban sa isang sample data array na naglalaman ng mga halaga para sa seconds, latitude, at longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Ang model ay nagbabalik ng **'3'**, na siyang country code para sa UK. Astig! 👽

## Ehersisyo - gumawa ng Flask app

Ngayon maaari kang gumawa ng Flask app para tawagin ang iyong model at magbalik ng katulad na resulta, ngunit sa mas kaaya-ayang paraan.

1. Magsimula sa pamamagitan ng paglikha ng folder na **web-app** sa tabi ng _notebook.ipynb_ file kung saan naroroon ang iyong _ufo-model.pkl_ file.

1. Sa folder na iyon, gumawa ng tatlong karagdagang folder: **static**, na may folder na **css** sa loob nito, at **templates**. Dapat mayroon ka na ngayong mga sumusunod na file at direktoryo:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Tingnan ang solution folder para sa view ng tapos na app

1. Ang unang file na gagawin sa _web-app_ folder ay ang **requirements.txt** file. Tulad ng _package.json_ sa isang JavaScript app, ang file na ito ay naglilista ng mga dependencies na kinakailangan ng app. Sa **requirements.txt** idagdag ang mga linya:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ngayon, patakbuhin ang file na ito sa pamamagitan ng pag-navigate sa _web-app_:

    ```bash
    cd web-app
    ```

1. Sa iyong terminal, i-type ang `pip install`, para i-install ang mga libraries na nakalista sa _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ngayon, handa ka nang gumawa ng tatlong karagdagang file para tapusin ang app:

    1. Gumawa ng **app.py** sa root.
    2. Gumawa ng **index.html** sa _templates_ directory.
    3. Gumawa ng **styles.css** sa _static/css_ directory.

1. I-build ang _styles.css_ file gamit ang ilang styles:

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

1. Susunod, i-build ang _index.html_ file:

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

    Tingnan ang templating sa file na ito. Pansinin ang 'mustache' syntax sa paligid ng mga variables na ibibigay ng app, tulad ng prediction text: `{{}}`. Mayroon ding form na nagpo-post ng prediction sa `/predict` route.

    Sa wakas, handa ka nang i-build ang python file na nagda-drive ng consumption ng model at ang display ng predictions:

1. Sa `app.py` idagdag:

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

    > 💡 Tip: kapag nagdagdag ka ng [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) habang pinapatakbo ang web app gamit ang Flask, anumang pagbabago na ginawa mo sa iyong application ay agad na makikita nang hindi na kailangang i-restart ang server. Mag-ingat! Huwag i-enable ang mode na ito sa production app.

Kung patakbuhin mo ang `python app.py` o `python3 app.py` - magsisimula ang iyong web server, lokal, at maaari kang mag-fill out ng maikling form para makakuha ng sagot sa iyong tanong tungkol sa kung saan nakita ang mga UFO!

Bago gawin iyon, tingnan ang mga bahagi ng `app.py`:

1. Una, ang mga dependencies ay na-load at nagsimula ang app.
1. Pagkatapos, ang model ay na-import.
1. Pagkatapos, ang index.html ay na-render sa home route.

Sa `/predict` route, ilang bagay ang nangyayari kapag na-post ang form:

1. Ang mga variables ng form ay kinokolekta at kino-convert sa numpy array. Ang mga ito ay ipinapadala sa model at ang prediction ay ibinabalik.
2. Ang mga bansa na gusto nating ipakita ay nire-render bilang nababasang text mula sa kanilang predicted country code, at ang value na iyon ay ibinabalik sa index.html para ma-render sa template.

Ang paggamit ng model sa ganitong paraan, gamit ang Flask at pickled model, ay medyo simple. Ang pinakamahirap na bahagi ay ang pag-unawa kung anong hugis ng data ang kailangang ipadala sa model para makakuha ng prediction. Ang lahat ng ito ay nakadepende sa kung paano na-train ang model. Ang model na ito ay may tatlong data points na kailangang i-input para makakuha ng prediction.

Sa isang propesyonal na setting, makikita mo kung gaano kahalaga ang magandang komunikasyon sa pagitan ng mga taong nagtrain ng model at ng mga gumagamit nito sa web o mobile app. Sa ating kaso, ikaw lang ang gumagawa nito!

---

## 🚀 Hamon

Sa halip na magtrabaho sa notebook at i-import ang model sa Flask app, maaari mong i-train ang model mismo sa loob ng Flask app! Subukang i-convert ang iyong Python code sa notebook, marahil pagkatapos malinis ang iyong data, para i-train ang model mula sa loob ng app sa isang route na tinatawag na `train`. Ano ang mga pros at cons ng paggamit ng pamamaraang ito?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pag-aaral sa Sarili

Maraming paraan para gumawa ng web app na gumagamit ng ML models. Gumawa ng listahan ng mga paraan kung paano mo magagamit ang JavaScript o Python para gumawa ng web app na gumagamit ng machine learning. Isaalang-alang ang arkitektura: dapat bang manatili ang model sa app o nasa cloud? Kung nasa cloud, paano mo ito maa-access? Gumuhit ng arkitektural na modelo para sa isang applied ML web solution.

## Takdang Aralin

[Subukan ang ibang model](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.