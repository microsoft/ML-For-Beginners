<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T12:57:25+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "sr"
}
-->
# Изградња веб апликације за коришћење ML модела

У овој лекцији, обучићете ML модел на скупу података који је ван овог света: _УФО виђења у последњем веку_, преузетих из базе података NUFORC-а.

Научићете:

- Како да „пиклујете“ обучени модел
- Како да користите тај модел у Flask апликацији

Наставићемо да користимо бележнице за чишћење података и обуку модела, али можете отићи корак даље и истражити како да користите модел „у природи“, односно у веб апликацији.

Да бисте то урадили, потребно је да изградите веб апликацију користећи Flask.

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

## Изградња апликације

Постоји неколико начина за изградњу веб апликација које користе моделе машинског учења. Ваша веб архитектура може утицати на начин на који је модел обучен. Замислите да радите у компанији где је тим за науку о подацима обучио модел који желе да користите у апликацији.

### Разматрања

Постоји много питања која треба поставити:

- **Да ли је то веб апликација или мобилна апликација?** Ако градите мобилну апликацију или треба да користите модел у IoT контексту, можете користити [TensorFlow Lite](https://www.tensorflow.org/lite/) и применити модел у Android или iOS апликацији.
- **Где ће модел бити смештен?** У облаку или локално?
- **Подршка за рад ван мреже.** Да ли апликација мора да ради ван мреже?
- **Која технологија је коришћена за обуку модела?** Одабрана технологија може утицати на алате које треба користити.
    - **Коришћење TensorFlow-а.** Ако обучавате модел користећи TensorFlow, на пример, тај екосистем омогућава конвертовање TensorFlow модела за употребу у веб апликацији помоћу [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Коришћење PyTorch-а.** Ако градите модел користећи библиотеку као што је [PyTorch](https://pytorch.org/), имате могућност да га извезете у [ONNX](https://onnx.ai/) (Open Neural Network Exchange) формат за употребу у JavaScript веб апликацијама које могу користити [Onnx Runtime](https://www.onnxruntime.ai/). Ова опција ће бити истражена у будућој лекцији за модел обучен помоћу Scikit-learn-а.
    - **Коришћење Lobe.ai или Azure Custom Vision.** Ако користите ML SaaS (Software as a Service) систем као што су [Lobe.ai](https://lobe.ai/) или [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) за обуку модела, ова врста софтвера пружа начине за извоз модела за многе платформе, укључујући изградњу прилагођеног API-ја који се може упитати у облаку од стране ваше онлајн апликације.

Такође имате могућност да изградите целу Flask веб апликацију која би могла да обучава модел директно у веб прегледачу. Ово се такође може урадити користећи TensorFlow.js у JavaScript контексту.

За наше потребе, пошто смо радили са бележницама заснованим на Python-у, хајде да истражимо кораке које треба предузети да бисмо извезли обучени модел из такве бележнице у формат који је читљив за Python веб апликацију.

## Алат

За овај задатак потребна су вам два алата: Flask и Pickle, оба која раде на Python-у.

✅ Шта је [Flask](https://palletsprojects.com/p/flask/)? Flask је дефинисан као „микро-оквир“ од стране својих креатора и пружа основне функције веб оквира користећи Python и механизам за шаблонирање за изградњу веб страница. Погледајте [овај модул за учење](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) да бисте вежбали изградњу са Flask-ом.

✅ Шта је [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 је Python модул који серијализује и десеријализује структуру Python објеката. Када „пиклујете“ модел, серијализујете или спљоштите његову структуру за употребу на вебу. Будите опрезни: Pickle није инхерентно безбедан, па будите пажљиви ако вам се предложи да „распакујете“ датотеку. Pickle датотека има наставак `.pkl`.

## Вежба - очистите своје податке

У овој лекцији користићете податке из 80.000 УФО виђења, прикупљених од стране [NUFORC](https://nuforc.org) (Национални центар за пријаву УФО-а). Ови подаци садрже занимљиве описе УФО виђења, на пример:

- **Дуг опис примера.** „Човек излази из зрака светлости који обасјава травнато поље ноћу и трчи према паркингу Texas Instruments-а“.
- **Кратак опис примера.** „Светла су нас јурила“.

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) табела укључује колоне о `граду`, `држави` и `земљи` где се виђење догодило, облику објекта (`shape`) и његовој `географској ширини` и `дужини`.

У празној [бележници](../../../../3-Web-App/1-Web-App/notebook.ipynb) која је укључена у ову лекцију:

1. Увезите `pandas`, `matplotlib` и `numpy` као што сте радили у претходним лекцијама и увезите табелу са УФО подацима. Можете погледати пример скупа података:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Претворите УФО податке у мањи dataframe са новим насловима. Проверите јединствене вредности у пољу `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Сада можете смањити количину података са којима треба да радимо тако што ћете избацити све вредности које недостају и увезти само виђења у трајању од 1-60 секунди:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Увезите библиотеку `LabelEncoder` из Scikit-learn-а да бисте текстуалне вредности за земље претворили у бројеве:

    ✅ LabelEncoder кодира податке азбучним редом

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Ваши подаци би требало да изгледају овако:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Вежба - изградите свој модел

Сада можете припремити податке за обуку модела тако што ћете их поделити у групе за обуку и тестирање.

1. Изаберите три карактеристике које желите да користите за обуку као свој X вектор, док ће y вектор бити `Country`. Желите да унесете `Seconds`, `Latitude` и `Longitude` и добијете ID земље као резултат.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Обучите свој модел користећи логистичку регресију:

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

Тачност није лоша **(око 95%)**, што није изненађујуће, јер `Country` и `Latitude/Longitude` корелирају.

Модел који сте креирали није баш револуционаран, јер би требало да можете да закључите `Country` из његове `Latitude` и `Longitude`, али је добра вежба покушати обучити модел од сирових података које сте очистили, извезли и затим користили у веб апликацији.

## Вежба - „пиклујте“ свој модел

Сада је време да _пиклујете_ свој модел! То можете урадити у неколико линија кода. Када је модел _пиклован_, учитајте га и тестирајте на пример подацима који садрже вредности за секунде, географску ширину и дужину.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Модел враћа **'3'**, што је код земље за Велику Британију. Невероватно! 👽

## Вежба - изградите Flask апликацију

Сада можете изградити Flask апликацију која ће позивати ваш модел и враћати сличне резултате, али на визуелно привлачнији начин.

1. Почните тако што ћете креирати фасциклу под називом **web-app** поред датотеке _notebook.ipynb_ где се налази ваша _ufo-model.pkl_ датотека.

1. У тој фасцикли креирајте још три фасцикле: **static**, са фасциклом **css** унутар ње, и **templates**. Сада би требало да имате следеће датотеке и директоријуме:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Погледајте решење у фасцикли за преглед готове апликације

1. Прва датотека коју треба креирати у фасцикли _web-app_ је **requirements.txt**. Као _package.json_ у JavaScript апликацији, ова датотека наводи зависности које су потребне апликацији. У **requirements.txt** додајте линије:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Сада покрените ову датотеку тако што ћете отићи у фасциклу _web-app_:

    ```bash
    cd web-app
    ```

1. У вашем терминалу укуцајте `pip install`, да бисте инсталирали библиотеке наведене у _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Сада сте спремни да креирате још три датотеке за завршетак апликације:

    1. Креирајте **app.py** у корену.
    2. Креирајте **index.html** у фасцикли _templates_.
    3. Креирајте **styles.css** у фасцикли _static/css_.

1. Направите _styles.css_ датотеку са неколико стилова:

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

1. Затим направите _index.html_ датотеку:

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

    Погледајте шаблонирање у овој датотеци. Приметите „mustache“ синтаксу око променљивих које ће бити обезбеђене од стране апликације, као што је текст предвиђања: `{{}}`. Такође постоји форма која шаље предвиђање на руту `/predict`.

    Коначно, спремни сте да изградите Python датотеку која управља коришћењем модела и приказом предвиђања:

1. У `app.py` додајте:

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

    > 💡 Савет: када додате [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) док покрећете веб апликацију користећи Flask, све промене које направите у вашој апликацији биће одмах видљиве без потребе за поновним покретањем сервера. Пазите! Немојте омогућити овај режим у продукцијској апликацији.

Ако покренете `python app.py` или `python3 app.py` - ваш веб сервер ће се покренути локално, и можете попунити кратку форму да добијете одговор на ваше горуће питање о томе где су УФО-и виђени!

Пре него што то урадите, погледајте делове `app.py`:

1. Прво се учитавају зависности и апликација се покреће.
1. Затим се модел увози.
1. Затим се `index.html` рендерује на почетној рути.

На рути `/predict`, дешава се неколико ствари када се форма пошаље:

1. Променљиве из форме се прикупљају и претварају у numpy низ. Оне се затим шаљу моделу и враћа се предвиђање.
2. Земље које желимо да прикажемо се поново рендерују као читљив текст из њиховог предвиђеног кода земље, и та вредност се шаље назад у `index.html` да би се рендеровала у шаблону.

Коришћење модела на овај начин, са Flask-ом и пиклованим моделом, је релативно једноставно. Најтежа ствар је разумети у ком облику подаци морају бити послати моделу да би се добило предвиђање. То све зависи од тога како је модел обучен. Овај модел захтева три тачке података за унос како би се добило предвиђање.

У професионалном окружењу, можете видети колико је добра комуникација неопходна између људи који обучавају модел и оних који га користе у веб или мобилној апликацији. У нашем случају, то је само једна особа - ви!

---

## 🚀 Изазов

Уместо рада у бележници и увоза модела у Flask апликацију, могли бисте обучити модел директно у Flask апликацији! Покушајте да конвертујете свој Python код из бележнице, можда након што очистите податке, како бисте обучили модел директно у апликацији на рути под називом `train`. Које су предности и мане овог приступа?

## [Квиз након предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Постоји много начина за изградњу веб апликације која користи ML моделе. Направите списак начина на које бисте могли користити JavaScript или Python за изградњу веб апликације која користи машинско учење. Размотрите архитектуру: да ли модел треба да остане у апликацији или да буде у облаку? Ако је у облаку, како бисте му приступили? Нацртајте архитектонски модел за примену ML решења у веб апликацији.

## Задатак

[Испробајте другачији модел](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати меродавним извором. За критичне информације препоручује се професионални превод од стране људи. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.