<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T00:36:52+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "bg"
}
-->
# Създаване на уеб приложение за използване на ML модел

В този урок ще обучите ML модел върху набор от данни, който е извън този свят: _забелязвания на НЛО през последния век_, взети от базата данни на NUFORC.

Ще научите:

- Как да „pickle“-нете обучен модел
- Как да използвате този модел в Flask приложение

Ще продължим да използваме notebooks за почистване на данни и обучение на модела, но можете да направите процеса още една стъпка напред, като изследвате използването на модел „в дивата природа“, така да се каже: в уеб приложение.

За да направите това, трябва да изградите уеб приложение с Flask.

## [Тест преди лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Създаване на приложение

Има няколко начина за създаване на уеб приложения, които да използват машинно обучени модели. Вашата уеб архитектура може да повлияе на начина, по който моделът ви е обучен. Представете си, че работите в бизнес, където групата за анализ на данни е обучила модел, който искат да използвате в приложение.

### Съображения

Има много въпроси, които трябва да зададете:

- **Уеб приложение или мобилно приложение?** Ако създавате мобилно приложение или трябва да използвате модела в IoT контекст, можете да използвате [TensorFlow Lite](https://www.tensorflow.org/lite/) и да използвате модела в Android или iOS приложение.
- **Къде ще се намира моделът?** В облака или локално?
- **Поддръжка офлайн.** Трябва ли приложението да работи офлайн?
- **Каква технология е използвана за обучение на модела?** Избраната технология може да повлияе на инструментите, които трябва да използвате.
    - **Използване на TensorFlow.** Ако обучавате модел с TensorFlow, например, тази екосистема предоставя възможност за конвертиране на TensorFlow модел за използване в уеб приложение чрез [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Използване на PyTorch.** Ако изграждате модел с библиотека като [PyTorch](https://pytorch.org/), имате опция да го експортирате във формат [ONNX](https://onnx.ai/) (Open Neural Network Exchange) за използване в JavaScript уеб приложения, които могат да използват [Onnx Runtime](https://www.onnxruntime.ai/). Тази опция ще бъде разгледана в бъдещ урок за модел, обучен със Scikit-learn.
    - **Използване на Lobe.ai или Azure Custom Vision.** Ако използвате ML SaaS (Software as a Service) система като [Lobe.ai](https://lobe.ai/) или [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) за обучение на модел, този тип софтуер предоставя начини за експортиране на модела за много платформи, включително изграждане на персонализиран API, който да бъде запитван в облака от вашето онлайн приложение.

Имате също така възможност да изградите цялостно Flask уеб приложение, което би могло да обучава модела директно в уеб браузър. Това може да се направи и с TensorFlow.js в JavaScript контекст.

За нашите цели, тъй като работим с notebooks, базирани на Python, нека разгледаме стъпките, които трябва да предприемете, за да експортирате обучен модел от такъв notebook във формат, четим от уеб приложение, изградено с Python.

## Инструмент

За тази задача ви трябват два инструмента: Flask и Pickle, и двата работещи на Python.

✅ Какво е [Flask](https://palletsprojects.com/p/flask/)? Определен като „микро-фреймуърк“ от своите създатели, Flask предоставя основните функции на уеб фреймуъркове, използвайки Python и шаблонен двигател за изграждане на уеб страници. Разгледайте [този модул за обучение](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott), за да практикувате изграждане с Flask.

✅ Какво е [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 е Python модул, който сериализира и десериализира структурата на Python обект. Когато „pickle“-нете модел, вие сериализирате или „сплесквате“ неговата структура за използване в уеб. Бъдете внимателни: pickle не е по същество сигурен, така че бъдете внимателни, ако бъдете подканени да „un-pickle“-нете файл. Pickle файл има суфикс `.pkl`.

## Упражнение - почистете данните си

В този урок ще използвате данни от 80,000 забелязвания на НЛО, събрани от [NUFORC](https://nuforc.org) (Националния център за докладване на НЛО). Тези данни съдържат интересни описания на забелязвания на НЛО, например:

- **Дълго описание.** "Мъж излиза от лъч светлина, който осветява тревисто поле през нощта, и той тича към паркинга на Texas Instruments".
- **Кратко описание.** "светлините ни преследваха".

Електронната таблица [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) включва колони за `city`, `state` и `country`, където е настъпило забелязването, формата на обекта `shape` и неговите `latitude` и `longitude`.

В празния [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb), включен в този урок:

1. Импортирайте `pandas`, `matplotlib` и `numpy`, както направихте в предишните уроци, и импортирайте електронната таблица ufos. Можете да разгледате примерен набор от данни:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Конвертирайте данните за ufos в малък dataframe с нови заглавия. Проверете уникалните стойности в полето `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Сега можете да намалите количеството данни, с които трябва да работите, като премахнете всички null стойности и импортирате само забелязвания между 1-60 секунди:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Импортирайте библиотеката `LabelEncoder` на Scikit-learn, за да конвертирате текстовите стойности за страните в числа:

    ✅ LabelEncoder кодира данни по азбучен ред

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Вашите данни трябва да изглеждат така:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Упражнение - изградете модела си

Сега можете да се подготвите за обучение на модел, като разделите данните на групи за обучение и тестване.

1. Изберете трите характеристики, върху които искате да обучите модела, като ваш X вектор, а y векторът ще бъде `Country`. Искате да можете да въведете `Seconds`, `Latitude` и `Longitude` и да получите идентификатор на страна за връщане.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Обучете модела си, използвайки логистична регресия:

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

Точността не е лоша **(около 95%)**, което не е изненадващо, тъй като `Country` и `Latitude/Longitude` са свързани.

Моделът, който създадохте, не е много революционен, тъй като би трябвало да можете да заключите `Country` от неговите `Latitude` и `Longitude`, но това е добра практика за обучение от сурови данни, които сте почистили, експортирали и след това използвали този модел в уеб приложение.

## Упражнение - „pickle“-нете модела си

Сега е време да _pickle_-нете модела си! Можете да направите това с няколко реда код. След като е _pickled_, заредете вашия pickled модел и го тествайте срещу примерен масив от данни, съдържащ стойности за секунди, ширина и дължина.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Моделът връща **'3'**, което е кодът на страната за Великобритания. Невероятно! 👽

## Упражнение - изградете Flask приложение

Сега можете да изградите Flask приложение, което да извиква вашия модел и да връща подобни резултати, но по-визуално приятен начин.

1. Започнете, като създадете папка, наречена **web-app**, до файла _notebook.ipynb_, където се намира вашият _ufo-model.pkl_ файл.

1. В тази папка създайте още три папки: **static**, с папка **css** вътре, и **templates**. Сега трябва да имате следните файлове и директории:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Вижте папката с решението за изглед на завършеното приложение

1. Първият файл, който трябва да създадете в папката _web-app_, е файлът **requirements.txt**. Подобно на _package.json_ в JavaScript приложение, този файл изброява зависимостите, необходими за приложението. В **requirements.txt** добавете редовете:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Сега изпълнете този файл, като навигирате до _web-app_:

    ```bash
    cd web-app
    ```

1. В терминала си напишете `pip install`, за да инсталирате библиотеките, изброени в _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Сега сте готови да създадете още три файла, за да завършите приложението:

    1. Създайте **app.py** в root директорията.
    2. Създайте **index.html** в директорията _templates_.
    3. Създайте **styles.css** в директорията _static/css_.

1. Изградете файла _styles.css_ с няколко стила:

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

1. След това изградете файла _index.html_:

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

    Разгледайте шаблонирането в този файл. Забележете синтаксиса „мустачки“ около променливите, които ще бъдат предоставени от приложението, като текста на предсказанието: `{{}}`. Има и форма, която изпраща предсказание към маршрута `/predict`.

    Накрая сте готови да изградите Python файла, който управлява използването на модела и показването на предсказанията:

1. В `app.py` добавете:

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

    > 💡 Съвет: когато добавите [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode), докато изпълнявате уеб приложението с Flask, всяка промяна, която направите в приложението, ще се отрази незабавно, без да е необходимо да рестартирате сървъра. Внимавайте! Не активирайте този режим в продукционно приложение.

Ако изпълните `python app.py` или `python3 app.py` - вашият уеб сървър стартира локално и можете да попълните кратка форма, за да получите отговор на вашия горещ въпрос за това къде са били забелязани НЛО!

Преди да направите това, разгледайте частите на `app.py`:

1. Първо, зависимостите се зареждат и приложението стартира.
1. След това моделът се импортира.
1. След това `index.html` се рендерира на началния маршрут.

На маршрута `/predict` се случват няколко неща, когато формата се изпрати:

1. Променливите от формата се събират и конвертират в numpy масив. След това те се изпращат към модела и се връща предсказание.
2. Страните, които искаме да се показват, се рендерират отново като четим текст от техния предсказан код на страна, и тази стойност се връща към `index.html`, за да бъде рендерирана в шаблона.

Използването на модел по този начин, с Flask и pickled модел, е сравнително лесно. Най-трудното е да разберете каква форма трябва да има данните, които трябва да бъдат изпратени към модела, за да получите предсказание. Това зависи изцяло от начина, по който моделът е обучен. Този има три точки данни, които трябва да бъдат въведени, за да се получи предсказание.

В професионална среда можете да видите колко е важно добрата комуникация между хората, които обучават модела, и тези, които го използват в уеб или мобилно приложение. В нашия случай, това е само един човек - вие!

---

## 🚀 Предизвикателство

Вместо да работите в notebook и да импортирате модела в Flask приложението, можете да обучите модела директно в Flask приложението! Опитайте да конвертирате вашия Python код в notebook, може би след като данните ви са почистени, за да обучите модела директно в приложението на маршрут, наречен `train`. Какви са плюсовете и минусите на този метод?

## [Тест след лекцията](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостоятелно обучение

Има много начини за създаване на уеб приложение, което да използва ML модели. Направете списък с начините, по които можете да използвате JavaScript или Python за изграждане на уеб приложение, което използва машинно обучение. Помислете за архитектурата: трябва ли моделът да остане в приложението или да живее в облака? Ако е второто, как бихте го достъпили? Начертайте архитектурен модел за приложено ML уеб решение.

## Задание

[Опитайте различен модел](assignment.md)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.