<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T00:46:18+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "bg"
}
-->
# Създаване на уеб приложение за препоръки на кухни

В този урок ще създадете модел за класификация, използвайки някои от техниките, които научихте в предишните уроци, и с помощта на вкусния набор от данни за кухни, използван в тази серия. Освен това ще изградите малко уеб приложение, което използва запазен модел, като се възползвате от уеб средата на Onnx.

Едно от най-полезните практически приложения на машинното обучение е създаването на системи за препоръки, и днес можете да направите първата стъпка в тази посока!

[![Представяне на това уеб приложение](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Кликнете върху изображението по-горе за видео: Джен Лупър създава уеб приложение, използвайки класифицирани данни за кухни

## [Тест преди урока](https://ff-quizzes.netlify.app/en/ml/)

В този урок ще научите:

- Как да създадете модел и да го запазите като Onnx модел
- Как да използвате Netron за инспекция на модела
- Как да използвате модела си в уеб приложение за извършване на предсказания

## Създайте своя модел

Създаването на приложни ML системи е важна част от използването на тези технологии за вашите бизнес системи. Можете да използвате модели в уеб приложенията си (и следователно да ги използвате в офлайн контекст, ако е необходимо) чрез Onnx.

В [предишен урок](../../3-Web-App/1-Web-App/README.md) създадохте регресионен модел за наблюдения на НЛО, "пиклирахте" го и го използвахте във Flask приложение. Въпреки че тази архитектура е много полезна, тя представлява пълноценна Python апликация, а вашите изисквания може да включват използването на JavaScript приложение.

В този урок можете да създадете базова JavaScript-базирана система за предсказания. Но първо трябва да обучите модел и да го конвертирате за използване с Onnx.

## Упражнение - обучение на модел за класификация

Първо, обучете модел за класификация, използвайки почистения набор от данни за кухни, който използвахме.

1. Започнете с импортиране на полезни библиотеки:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Ще ви трябва '[skl2onnx](https://onnx.ai/sklearn-onnx/)', за да помогне при конвертирането на вашия Scikit-learn модел в Onnx формат.

1. След това работете с данните си по същия начин, както в предишните уроци, като прочетете CSV файл с `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Премахнете първите две ненужни колони и запазете останалите данни като 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Запазете етикетите като 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Започнете рутината за обучение

Ще използваме библиотеката 'SVC', която има добра точност.

1. Импортирайте подходящите библиотеки от Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Разделете данните на тренировъчен и тестов набор:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Създайте SVC модел за класификация, както направихте в предишния урок:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Сега тествайте модела си, като извикате `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Отпечатайте отчет за класификация, за да проверите качеството на модела:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Както видяхме преди, точността е добра:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Конвертирайте модела си в Onnx

Уверете се, че правите конверсията с правилния брой тензори. Този набор от данни има 380 изброени съставки, така че трябва да отбележите този брой в `FloatTensorType`:

1. Конвертирайте, използвайки тензор с брой 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Създайте onx файл и го запазете като **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Забележка: Можете да предадете [опции](https://onnx.ai/sklearn-onnx/parameterized.html) във вашия скрипт за конверсия. В този случай зададохме 'nocl' да бъде True и 'zipmap' да бъде False. Тъй като това е модел за класификация, имате възможност да премахнете ZipMap, който произвежда списък от речници (не е необходимо). `nocl` се отнася до включването на информация за класовете в модела. Намалете размера на модела си, като зададете `nocl` на 'True'.

Изпълнението на целия ноутбук сега ще създаде Onnx модел и ще го запази в тази папка.

## Прегледайте модела си

Onnx моделите не са много видими във Visual Studio Code, но има много добър безплатен софтуер, който много изследователи използват, за да визуализират модела и да се уверят, че е правилно създаден. Изтеглете [Netron](https://github.com/lutzroeder/Netron) и отворете вашия model.onnx файл. Можете да видите вашия прост модел визуализиран, с неговите 380 входа и класификатор:

![Netron визуализация](../../../../4-Classification/4-Applied/images/netron.png)

Netron е полезен инструмент за преглед на вашите модели.

Сега сте готови да използвате този интересен модел в уеб приложение. Нека създадем приложение, което ще бъде полезно, когато погледнете в хладилника си и се опитате да разберете коя комбинация от останалите ви съставки можете да използвате, за да приготвите дадена кухня, определена от вашия модел.

## Създайте уеб приложение за препоръки

Можете да използвате модела си директно в уеб приложение. Тази архитектура също така ви позволява да го стартирате локално и дори офлайн, ако е необходимо. Започнете, като създадете файл `index.html` в същата папка, където сте запазили вашия `model.onnx` файл.

1. В този файл _index.html_ добавете следния маркъп:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Сега, работейки в таговете `body`, добавете малко маркъп, за да покажете списък с чекбоксове, отразяващи някои съставки:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Забележете, че на всеки чекбокс е зададена стойност. Това отразява индекса, където съставката се намира според набора от данни. Например, ябълката в този азбучен списък заема петата колона, така че нейната стойност е '4', тъй като започваме броенето от 0. Можете да се консултирате с [електронната таблица със съставки](../../../../4-Classification/data/ingredient_indexes.csv), за да откриете индекса на дадена съставка.

    Продължавайки работата си във файла index.html, добавете блок със скрипт, където моделът се извиква след последния затварящ `</div>`.

1. Първо, импортирайте [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime се използва, за да позволи изпълнението на вашите Onnx модели на широк спектър от хардуерни платформи, включително оптимизации и API за използване.

1. След като Runtime е на място, можете да го извикате:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

В този код се случват няколко неща:

1. Създадохте масив от 380 възможни стойности (1 или 0), които да бъдат зададени и изпратени към модела за предсказание, в зависимост от това дали чекбоксът за съставка е маркиран.
2. Създадохте масив от чекбоксове и начин за определяне дали са маркирани във функция `init`, която се извиква при стартиране на приложението. Когато чекбокс е маркиран, масивът `ingredients` се променя, за да отрази избраната съставка.
3. Създадохте функция `testCheckboxes`, която проверява дали някой чекбокс е маркиран.
4. Използвате функцията `startInference`, когато бутонът е натиснат, и ако някой чекбокс е маркиран, започвате предсказанието.
5. Рутината за предсказание включва:
   1. Настройка на асинхронно зареждане на модела
   2. Създаване на структура Tensor, която да се изпрати към модела
   3. Създаване на 'feeds', които отразяват входа `float_input`, който създадохте при обучението на модела (можете да използвате Netron, за да проверите това име)
   4. Изпращане на тези 'feeds' към модела и изчакване на отговор

## Тествайте приложението си

Отворете терминална сесия във Visual Studio Code в папката, където се намира вашият файл index.html. Уверете се, че имате инсталиран глобално [http-server](https://www.npmjs.com/package/http-server), и напишете `http-server` в командния ред. Трябва да се отвори localhost, където можете да видите вашето уеб приложение. Проверете коя кухня се препоръчва въз основа на различни съставки:

![уеб приложение за съставки](../../../../4-Classification/4-Applied/images/web-app.png)

Поздравления, създадохте уеб приложение за препоръки с няколко полета. Отделете време, за да разширите тази система!

## 🚀Предизвикателство

Вашето уеб приложение е много минимално, така че продължете да го разширявате, използвайки съставките и техните индекси от данните [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). Какви комбинации от вкусове работят за създаване на дадено национално ястие?

## [Тест след урока](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостоятелно обучение

Докато този урок само докосна полезността на създаването на система за препоръки за хранителни съставки, тази област на ML приложения е много богата на примери. Прочетете повече за това как се изграждат тези системи:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Задача

[Създайте нова система за препоръки](assignment.md)

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.