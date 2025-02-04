# Jenga Tovuti Kutumia Mfano wa ML

Katika somo hili, utafundisha mfano wa ML kwenye seti ya data ambayo ni ya kipekee: _matukio ya UFO katika karne iliyopita_, iliyotolewa kutoka kwenye hifadhidata ya NUFORC.

Utajifunza:

- Jinsi ya 'pickle' mfano uliyo fundishwa
- Jinsi ya kutumia mfano huo katika programu ya Flask

Tutaendelea kutumia daftari za maelezo kusafisha data na kufundisha mfano wetu, lakini unaweza kuchukua hatua moja zaidi kwa kuchunguza kutumia mfano 'katika mazingira halisi', kwa maneno mengine: katika programu ya wavuti.

Ili kufanya hivi, unahitaji kujenga programu ya wavuti kwa kutumia Flask.

## [Jaribio la kabla ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Kujenga programu

Kuna njia kadhaa za kujenga programu za wavuti ili kutumia mifano ya kujifunza mashine. Muundo wako wa wavuti unaweza kuathiri jinsi mfano wako unavyofundishwa. Fikiria kuwa unafanya kazi katika biashara ambapo kikundi cha sayansi ya data kimefundisha mfano ambao wanataka utumie katika programu.

### Mambo ya Kuzingatia

Kuna maswali mengi unayohitaji kuuliza:

- **Je, ni programu ya wavuti au programu ya simu?** Ikiwa unajenga programu ya simu au unahitaji kutumia mfano katika muktadha wa IoT, unaweza kutumia [TensorFlow Lite](https://www.tensorflow.org/lite/) na kutumia mfano katika programu ya Android au iOS.
- **Mfano utakuwa wapi?** Katika wingu au ndani ya nchi?
- **Msaada wa nje ya mtandao.** Je, programu inahitaji kufanya kazi nje ya mtandao?
- **Teknolojia gani ilitumika kufundisha mfano?** Teknolojia iliyochaguliwa inaweza kuathiri zana unazohitaji kutumia.
    - **Kutumia TensorFlow.** Ikiwa unafundisha mfano kwa kutumia TensorFlow, kwa mfano, mfumo huo unatoa uwezo wa kubadilisha mfano wa TensorFlow kwa matumizi katika programu ya wavuti kwa kutumia [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Kutumia PyTorch.** Ikiwa unajenga mfano kwa kutumia maktaba kama [PyTorch](https://pytorch.org/), una chaguo la kuuza nje katika muundo wa [ONNX](https://onnx.ai/) (Open Neural Network Exchange) kwa matumizi katika programu za wavuti za JavaScript zinazoweza kutumia [Onnx Runtime](https://www.onnxruntime.ai/). Chaguo hili litachunguzwa katika somo la baadaye kwa mfano uliofundishwa na Scikit-learn.
    - **Kutumia Lobe.ai au Azure Custom Vision.** Ikiwa unatumia mfumo wa ML SaaS (Software as a Service) kama [Lobe.ai](https://lobe.ai/) au [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) kufundisha mfano, aina hii ya programu inatoa njia za kuuza nje mfano kwa majukwaa mengi, ikiwa ni pamoja na kujenga API maalum ya kuulizwa katika wingu na programu yako ya mtandaoni.

Pia una nafasi ya kujenga programu kamili ya wavuti ya Flask ambayo ingeweza kufundisha mfano yenyewe katika kivinjari cha wavuti. Hii inaweza pia kufanywa kwa kutumia TensorFlow.js katika muktadha wa JavaScript.

Kwa madhumuni yetu, kwa kuwa tumekuwa tukifanya kazi na daftari za maelezo za msingi wa Python, hebu tuchunguze hatua unazohitaji kuchukua ili kuuza nje mfano uliofundishwa kutoka daftari kama hilo kwa muundo unaosomeka na programu ya wavuti iliyojengwa kwa Python.

## Zana

Kwa kazi hii, unahitaji zana mbili: Flask na Pickle, zote zinaendesha kwenye Python.

✅ Flask ni nini? [Flask](https://palletsprojects.com/p/flask/) ni mfumo wa 'micro-framework' kama ulivyoelezwa na waumbaji wake, Flask hutoa vipengele vya msingi vya mifumo ya wavuti kwa kutumia Python na injini ya templating kujenga kurasa za wavuti. Angalia [moduli hii ya kujifunza](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) ili kufanya mazoezi ya kujenga na Flask.

✅ Pickle ni nini? [Pickle](https://docs.python.org/3/library/pickle.html) 🥒 ni moduli ya Python inayosarifu na kufungua muundo wa kitu cha Python. Unapofanya 'pickle' mfano, unasarifu au kupanua muundo wake kwa matumizi kwenye wavuti. Kuwa mwangalifu: pickle sio salama kiasili, kwa hivyo kuwa mwangalifu ikiwa umeombwa kufungua faili iliyofunguliwa kwa pickle. Faili iliyofunguliwa kwa pickle ina kiambishi cha `.pkl`.

## Zoezi - safisha data yako

Katika somo hili utatumia data kutoka kwa matukio 80,000 ya UFO, yaliyokusanywa na [NUFORC](https://nuforc.org) (Kituo cha Kitaifa cha Kuripoti UFO). Data hii ina maelezo ya kuvutia ya matukio ya UFO, kwa mfano:

- **Maelezo marefu ya mfano.** "Mtu anatoka kwenye mwanga unaong'aa kwenye uwanja wa nyasi usiku na anakimbia kuelekea kwenye maegesho ya Texas Instruments".
- **Maelezo mafupi ya mfano.** "taa zilituandama".

Faili ya [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inajumuisha safu kuhusu `city`, `state` na `country` ambapo tukio lilitokea, `shape` ya kitu na `latitude` na `longitude`.

Katika [daftari](../../../../3-Web-App/1-Web-App/notebook.ipynb) lililojumuishwa katika somo hili:

1. ingiza `pandas`, `matplotlib`, na `numpy` kama ulivyofanya katika masomo yaliyopita na ingiza faili ya ufos. Unaweza kuangalia seti ya data ya sampuli:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Badilisha data ya ufos kuwa dataframe ndogo na vichwa vipya. Angalia maadili ya kipekee katika uwanja wa `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Sasa, unaweza kupunguza kiasi cha data tunachohitaji kushughulika nacho kwa kuondoa maadili yoyote ya null na kuingiza tu matukio kati ya sekunde 1-60:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Ingiza maktaba ya `LabelEncoder` ya Scikit-learn ili kubadilisha maadili ya maandishi kwa nchi kuwa namba:

    ✅ LabelEncoder inasimbua data kwa alfabeti

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Data yako inapaswa kuonekana kama hii:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Zoezi - jenga mfano wako

Sasa unaweza kujiandaa kufundisha mfano kwa kugawanya data katika kikundi cha mafunzo na majaribio.

1. Chagua vipengele vitatu unavyotaka kufundisha kama vector yako ya X, na vector ya y itakuwa `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` na pata kitambulisho cha nchi kurudisha.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Fundisha mfano wako kwa kutumia regression ya logistic:

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

Usahihi sio mbaya **(karibu 95%)**, bila kushangaza, kama `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, lakini ni zoezi nzuri kujaribu kufundisha kutoka kwa data mbichi uliyoisafisha, kuiuza nje, na kisha kutumia mfano huu katika programu ya wavuti.

## Zoezi - 'pickle' mfano wako

Sasa, ni wakati wa _pickle_ mfano wako! Unaweza kufanya hivyo kwa mistari michache ya msimbo. Mara tu unapofanya _pickle_, pakia mfano wako uliopickle na ujaribu dhidi ya safu ya data ya sampuli iliyo na maadili ya sekunde, latitudo na longitudo,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Mfano unarudisha **'3'**, ambayo ni nambari ya nchi kwa Uingereza. Ajabu! 👽

## Zoezi - jenga programu ya Flask

Sasa unaweza kujenga programu ya Flask ili kuita mfano wako na kurudisha matokeo sawa, lakini kwa njia ya kuvutia zaidi.

1. Anza kwa kuunda folda inayoitwa **web-app** karibu na faili ya _notebook.ipynb_ ambapo faili yako ya _ufo-model.pkl_ ipo.

1. Katika folda hiyo unda folda nyingine tatu: **static**, yenye folda **css** ndani yake, na **templates**. Sasa unapaswa kuwa na faili na saraka zifuatazo:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Rejelea folda ya suluhisho kwa mtazamo wa programu iliyokamilika

1. Faili ya kwanza kuunda katika folda ya _web-app_ ni faili ya **requirements.txt**. Kama _package.json_ katika programu ya JavaScript, faili hii inataja utegemezi unaohitajika na programu. Katika **requirements.txt** ongeza mistari:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Sasa, endesha faili hii kwa kuvinjari kwenye _web-app_:

    ```bash
    cd web-app
    ```

1. Katika terminal yako andika `pip install`, ili kusakinisha maktaba zilizoorodheshwa katika _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Sasa, uko tayari kuunda faili tatu zaidi kumaliza programu:

    1. Unda **app.py** katika mzizi.
    2. Unda **index.html** katika saraka ya _templates_.
    3. Unda **styles.css** katika saraka ya _static/css_.

1. Jenga faili ya _styles.css_ na mitindo michache:

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

1. Kisha, jenga faili ya _index.html_:

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

    Angalia templating katika faili hii. Kumbuka sintaksia ya 'mustache' kuzunguka mabadiliko ambayo yatatolewa na programu, kama maandishi ya utabiri: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` ongeza:

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

    > 💡 Kidokezo: unapoongeza [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Je, ni faida na hasara gani za kufuata njia hii?

## [Jaribio la baada ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Mapitio & Kujisomea

Kuna njia nyingi za kujenga programu ya wavuti ili kutumia mifano ya ML. Fanya orodha ya njia ambazo unaweza kutumia JavaScript au Python kujenga programu ya wavuti ili kutumia kujifunza mashine. Fikiria muundo: je, mfano unabaki katika programu au unaishi katika wingu? Ikiwa ni la pili, utaupataje? Chora muundo wa kimuundo kwa suluhisho la wavuti linalotumia ML.

## Kazi

[Jaribu mfano tofauti](assignment.md)

**Kanusho**:
Hati hii imetafsiriwa kwa kutumia huduma za tafsiri za AI za mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati asilia katika lugha yake ya asili inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, inashauriwa kutumia tafsiri ya kibinadamu ya kitaalamu. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.