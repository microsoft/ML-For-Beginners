<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T16:12:26+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "sw"
}
-->
# Jenga Programu ya Wavuti Kutumia Modeli ya ML

Katika somo hili, utapokea mafunzo ya modeli ya ML kwa kutumia seti ya data isiyo ya kawaida: _Matukio ya UFO katika karne iliyopita_, yaliyokusanywa kutoka hifadhidata ya NUFORC.

Utajifunza:

- Jinsi ya 'pickle' modeli iliyofunzwa
- Jinsi ya kutumia modeli hiyo katika programu ya Flask

Tutaendelea kutumia daftari za maelezo (notebooks) kusafisha data na kufunza modeli yetu, lakini unaweza kuchukua hatua moja zaidi kwa kuchunguza jinsi ya kutumia modeli 'katika mazingira halisi', yaani: katika programu ya wavuti.

Ili kufanya hivyo, unahitaji kujenga programu ya wavuti kwa kutumia Flask.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Kujenga programu

Kuna njia kadhaa za kujenga programu za wavuti zinazotumia modeli za kujifunza mashine. Muundo wa wavuti yako unaweza kuathiri jinsi modeli yako inavyofunzwa. Fikiria kuwa unafanya kazi katika biashara ambapo kikundi cha sayansi ya data kimefunza modeli wanayotaka uitumie katika programu.

### Mambo ya Kuzingatia

Kuna maswali mengi unayohitaji kuuliza:

- **Je, ni programu ya wavuti au ya simu?** Ikiwa unajenga programu ya simu au unahitaji kutumia modeli katika muktadha wa IoT, unaweza kutumia [TensorFlow Lite](https://www.tensorflow.org/lite/) na kutumia modeli hiyo katika programu ya Android au iOS.
- **Modeli itakuwa wapi?** Katika wingu au ndani ya kifaa?
- **Msaada wa nje ya mtandao.** Je, programu inapaswa kufanya kazi bila mtandao?
- **Teknolojia gani ilitumika kufunza modeli?** Teknolojia iliyochaguliwa inaweza kuathiri zana unazohitaji kutumia.
    - **Kutumia TensorFlow.** Ikiwa unafunza modeli kwa kutumia TensorFlow, kwa mfano, mfumo huo unatoa uwezo wa kubadilisha modeli ya TensorFlow kwa matumizi katika programu ya wavuti kwa kutumia [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Kutumia PyTorch.** Ikiwa unajenga modeli kwa kutumia maktaba kama [PyTorch](https://pytorch.org/), una chaguo la kuisafirisha katika muundo wa [ONNX](https://onnx.ai/) (Open Neural Network Exchange) kwa matumizi katika programu za wavuti za JavaScript zinazoweza kutumia [Onnx Runtime](https://www.onnxruntime.ai/). Chaguo hili litachunguzwa katika somo la baadaye kwa modeli iliyofunzwa na Scikit-learn.
    - **Kutumia Lobe.ai au Azure Custom Vision.** Ikiwa unatumia mfumo wa ML SaaS (Software as a Service) kama [Lobe.ai](https://lobe.ai/) au [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) kufunza modeli, programu ya aina hii inatoa njia za kusafirisha modeli kwa majukwaa mengi, ikiwa ni pamoja na kujenga API maalum inayoweza kuulizwa katika wingu na programu yako ya mtandaoni.

Pia una nafasi ya kujenga programu nzima ya wavuti ya Flask ambayo inaweza kufunza modeli yenyewe katika kivinjari cha wavuti. Hii inaweza pia kufanywa kwa kutumia TensorFlow.js katika muktadha wa JavaScript.

Kwa madhumuni yetu, kwa kuwa tumekuwa tukifanya kazi na daftari za maelezo za msingi wa Python, hebu tuchunguze hatua unazohitaji kuchukua kusafirisha modeli iliyofunzwa kutoka daftari kama hiyo hadi muundo unaoweza kusomwa na programu ya wavuti iliyojengwa kwa Python.

## Zana

Kwa kazi hii, unahitaji zana mbili: Flask na Pickle, zote zinazoendeshwa na Python.

✅ [Flask](https://palletsprojects.com/p/flask/) ni nini? Imeelezwa kama 'micro-framework' na waumbaji wake, Flask hutoa vipengele vya msingi vya mifumo ya wavuti kwa kutumia Python na injini ya kutengeneza kurasa za wavuti. Angalia [moduli hii ya kujifunza](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) ili kufanya mazoezi ya kujenga kwa kutumia Flask.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) ni nini? Pickle 🥒 ni moduli ya Python inayosawazisha na kuondoa usawazishaji wa muundo wa kitu cha Python. Unapofanya 'pickle' modeli, unasawazisha au kuisawazisha kwa matumizi kwenye wavuti. Kuwa makini: pickle si salama kiasili, kwa hivyo kuwa mwangalifu ikiwa umeombwa 'kuondoa pickle' faili. Faili iliyofanyiwa pickle ina kiambishi `.pkl`.

## Zoezi - safisha data yako

Katika somo hili utatumia data kutoka kwa matukio 80,000 ya UFO, yaliyokusanywa na [NUFORC](https://nuforc.org) (Kituo cha Kitaifa cha Kuripoti Matukio ya UFO). Data hii ina maelezo ya kuvutia kuhusu matukio ya UFO, kwa mfano:

- **Maelezo marefu ya mfano.** "Mtu anatokea kutoka kwenye mwanga unaong'aa kwenye uwanja wa nyasi usiku na anakimbilia kwenye maegesho ya Texas Instruments".
- **Maelezo mafupi ya mfano.** "taa zilituandama".

Faili ya [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inajumuisha safu kuhusu `mji`, `jimbo` na `nchi` ambapo tukio lilitokea, `umbo` la kitu na `latitudo` na `longitudo` yake.

Katika [daftari](../../../../3-Web-App/1-Web-App/notebook.ipynb) tupu lililojumuishwa katika somo hili:

1. Ingiza `pandas`, `matplotlib`, na `numpy` kama ulivyofanya katika masomo ya awali na ingiza faili ya ufos. Unaweza kuangalia sampuli ya seti ya data:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Badilisha data ya ufos kuwa fremu ndogo ya data yenye vichwa vipya. Angalia maadili ya kipekee katika safu ya `Nchi`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Sasa, unaweza kupunguza kiasi cha data tunachohitaji kushughulikia kwa kuondoa maadili tupu na kuingiza tu matukio ya muda wa sekunde 1-60:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Ingiza maktaba ya `LabelEncoder` ya Scikit-learn ili kubadilisha maadili ya maandishi ya nchi kuwa nambari:

    ✅ LabelEncoder inasimba data kwa mpangilio wa alfabeti

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

## Zoezi - jenga modeli yako

Sasa unaweza kujiandaa kufunza modeli kwa kugawanya data katika kikundi cha mafunzo na majaribio.

1. Chagua vipengele vitatu unavyotaka kufunza kama vector yako ya X, na vector ya y itakuwa `Nchi`. Unataka kuweza kuingiza `Sekunde`, `Latitudo` na `Longitudo` na kupata kitambulisho cha nchi cha kurudisha.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Funza modeli yako kwa kutumia regression ya logistic:

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

Usahihi si mbaya **(karibu 95%)**, bila kushangaza, kwa kuwa `Nchi` na `Latitudo/Longitudo` vina uhusiano.

Modeli uliyounda si ya mapinduzi sana kwa kuwa unapaswa kuweza kubaini `Nchi` kutoka kwa `Latitudo` na `Longitudo`, lakini ni zoezi zuri kujaribu kufunza kutoka kwa data ghafi uliyosafisha, ukasafirisha, na kisha kutumia modeli hii katika programu ya wavuti.

## Zoezi - 'pickle' modeli yako

Sasa, ni wakati wa _pickle_ modeli yako! Unaweza kufanya hivyo kwa mistari michache ya msimbo. Mara tu imefanyiwa _pickle_, pakia faili ya modeli hiyo na ujaribu dhidi ya safu ya data ya sampuli inayojumuisha maadili ya sekunde, latitudo na longitudo,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Modeli inarudisha **'3'**, ambayo ni nambari ya nchi ya Uingereza. Ajabu! 👽

## Zoezi - jenga programu ya Flask

Sasa unaweza kujenga programu ya Flask ili kuita modeli yako na kurudisha matokeo yanayofanana, lakini kwa njia ya kuvutia zaidi.

1. Anza kwa kuunda folda inayoitwa **web-app** karibu na faili _notebook.ipynb_ ambapo faili yako _ufo-model.pkl_ iko.

1. Katika folda hiyo unda folda tatu zaidi: **static**, yenye folda **css** ndani yake, na **templates**. Sasa unapaswa kuwa na faili na folda zifuatazo:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Rejelea folda ya suluhisho ili kuona programu iliyokamilika

1. Faili ya kwanza ya kuunda katika folda ya _web-app_ ni faili ya **requirements.txt**. Kama _package.json_ katika programu ya JavaScript, faili hii inaorodhesha utegemezi unaohitajika na programu. Katika **requirements.txt** ongeza mistari:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Sasa, endesha faili hii kwa kuvinjari hadi _web-app_:

    ```bash
    cd web-app
    ```

1. Katika terminal yako andika `pip install`, ili kusakinisha maktaba zilizoorodheshwa katika _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Sasa, uko tayari kuunda faili tatu zaidi ili kukamilisha programu:

    1. Unda **app.py** katika mzizi.
    2. Unda **index.html** katika folda ya _templates_.
    3. Unda **styles.css** katika folda ya _static/css_.

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

    Angalia templating katika faili hii. Tambua sintaksia ya 'mustache' karibu na vigezo ambavyo vitatolewa na programu, kama maandishi ya utabiri: `{{}}`. Pia kuna fomu inayotuma utabiri kwa njia ya `/predict`.

    Hatimaye, uko tayari kujenga faili ya Python inayosukuma matumizi ya modeli na kuonyesha utabiri:

1. Katika `app.py` ongeza:

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

    > 💡 Kidokezo: unapoongeza [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) wakati wa kuendesha programu ya wavuti kwa kutumia Flask, mabadiliko yoyote unayofanya kwenye programu yako yataonyeshwa mara moja bila hitaji la kuanzisha upya seva. Tahadhari! Usifungue hali hii katika programu ya uzalishaji.

Ukikimbia `python app.py` au `python3 app.py` - seva yako ya wavuti inaanza, ndani ya kifaa chako, na unaweza kujaza fomu fupi ili kupata jibu la swali lako kuhusu mahali UFO zimeonekana!

Kabla ya kufanya hivyo, angalia sehemu za `app.py`:

1. Kwanza, utegemezi unapakuliwa na programu inaanza.
1. Kisha, modeli inasafirishwa.
1. Kisha, index.html inatolewa kwenye njia ya nyumbani.

Katika njia ya `/predict`, mambo kadhaa hufanyika wakati fomu inatumwa:

1. Vigezo vya fomu vinakusanywa na kubadilishwa kuwa safu ya numpy. Kisha vinatumwa kwa modeli na utabiri unarudishwa.
2. Nchi tunazotaka kuonyesha zinabadilishwa kuwa maandishi yanayosomeka kutoka kwa nambari ya nchi iliyotabiriwa, na thamani hiyo inarudishwa kwa index.html ili kutolewa katika template.

Kutumia modeli kwa njia hii, kwa Flask na modeli iliyofanyiwa pickle, ni rahisi kiasi. Jambo gumu zaidi ni kuelewa umbo la data ambalo lazima litumwe kwa modeli ili kupata utabiri. Hilo linategemea jinsi modeli ilivyofunzwa. Hii ina alama tatu za data za kuingiza ili kupata utabiri.

Katika mazingira ya kitaalamu, unaweza kuona jinsi mawasiliano mazuri yanavyohitajika kati ya wale wanaofunza modeli na wale wanaoitumia katika programu ya wavuti au ya simu. Katika kesi yetu, ni mtu mmoja tu, wewe!

---

## 🚀 Changamoto

Badala ya kufanya kazi katika daftari na kuingiza modeli kwenye programu ya Flask, unaweza kufunza modeli moja kwa moja ndani ya programu ya Flask! Jaribu kubadilisha msimbo wako wa Python katika daftari, labda baada ya data yako kusafishwa, ili kufunza modeli kutoka ndani ya programu kwenye njia inayoitwa `train`. Je, ni faida na hasara gani za kufuata njia hii?

## [Maswali ya baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio na Kujifunza Binafsi

Kuna njia nyingi za kujenga programu ya wavuti inayotumia modeli za ML. Tengeneza orodha ya njia unazoweza kutumia JavaScript au Python kujenga programu ya wavuti inayotumia kujifunza mashine. Fikiria muundo: je, modeli inapaswa kubaki katika programu au kuishi katika wingu? Ikiwa ni ya pili, utaipataje? Chora muundo wa usanifu wa suluhisho la ML linalotumika kwenye wavuti.

## Kazi

[Jaribu modeli tofauti](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.