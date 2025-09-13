<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-05T13:09:58+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "sr"
}
-->
# Направите веб апликацију за препоруку кухиње

У овом часу, направићете модел класификације користећи неке од техника које сте научили у претходним лекцијама, уз помоћ укусног скупа података о кухињама који се користио током ове серије. Поред тога, направићете малу веб апликацију која користи сачувани модел, користећи Onnx веб рунтајм.

Једна од најкориснијих практичних примена машинског учења је изградња система за препоруке, и данас можете направити први корак у том правцу!

[![Презентација ове веб апликације](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 Кликните на слику изнад за видео: Џен Лупер прави веб апликацију користећи класификоване податке о кухињама

## [Квиз пре предавања](https://ff-quizzes.netlify.app/en/ml/)

У овом часу ћете научити:

- Како направити модел и сачувати га као Onnx модел
- Како користити Netron за преглед модела
- Како користити ваш модел у веб апликацији за инференцију

## Направите свој модел

Изградња примењених ML система је важан део коришћења ових технологија за ваше пословне системе. Моделе можете користити унутар ваших веб апликација (и тако их користити у офлајн контексту ако је потребно) користећи Onnx.

У [претходном часу](../../3-Web-App/1-Web-App/README.md), направили сте регресиони модел о виђењима НЛО-а, "пикловали" га и користили у Flask апликацији. Иако је ова архитектура веома корисна за познавање, то је апликација са пуним стеком у Python-у, а ваши захтеви могу укључивати употребу JavaScript апликације.

У овом часу, можете направити основни систем заснован на JavaScript-у за инференцију. Прво, међутим, потребно је да обучите модел и конвертујете га за употребу са Onnx-ом.

## Вежба - обучите модел класификације

Прво, обучите модел класификације користећи очишћени скуп података о кухињама који смо користили.

1. Почните увозом корисних библиотека:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Потребан вам је '[skl2onnx](https://onnx.ai/sklearn-onnx/)' да бисте помогли у конверзији вашег Scikit-learn модела у Onnx формат.

1. Затим, радите са вашим подацима на исти начин као у претходним часовима, читајући CSV фајл користећи `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Уклоните прве две непотребне колоне и сачувајте преостале податке као 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Сачувајте ознаке као 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Започните рутину тренинга

Користићемо библиотеку 'SVC' која има добру тачност.

1. Увезите одговарајуће библиотеке из Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Одвојите сетове за тренинг и тестирање:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Направите SVC модел класификације као што сте урадили у претходном часу:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Сада тестирајте ваш модел, позивајући `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Испишите извештај о класификацији да проверите квалитет модела:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Као што смо видели раније, тачност је добра:

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

### Конвертујте ваш модел у Onnx

Уверите се да сте извршили конверзију са одговарајућим бројем тензора. Овај скуп података има 380 састојака, па је потребно да наведете тај број у `FloatTensorType`:

1. Конвертујте користећи број тензора 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Направите onx и сачувајте као фајл **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Напомена, можете проследити [опције](https://onnx.ai/sklearn-onnx/parameterized.html) у вашем скрипту за конверзију. У овом случају, поставили смо 'nocl' да буде True и 'zipmap' да буде False. Пошто је ово модел класификације, имате опцију да уклоните ZipMap који производи листу речника (није неопходно). `nocl` се односи на информације о класи које су укључене у модел. Смањите величину вашег модела постављањем `nocl` на 'True'.

Извршавање целог нотебука сада ће направити Onnx модел и сачувати га у овај фолдер.

## Прегледајте ваш модел

Onnx модели нису баш видљиви у Visual Studio Code-у, али постоји веома добар бесплатан софтвер који многи истраживачи користе за визуализацију модела како би се уверили да је правилно направљен. Преузмите [Netron](https://github.com/lutzroeder/Netron) и отворите ваш model.onnx фајл. Можете видети ваш једноставан модел визуализован, са његових 380 улаза и класификатором:

![Netron визуализација](../../../../4-Classification/4-Applied/images/netron.png)

Netron је користан алат за преглед ваших модела.

Сада сте спремни да користите овај занимљив модел у веб апликацији. Направимо апликацију која ће бити корисна када погледате у ваш фрижидер и покушате да схватите коју комбинацију ваших преосталих састојака можете користити за припрему одређене кухиње, како је одредио ваш модел.

## Направите веб апликацију за препоруке

Можете користити ваш модел директно у веб апликацији. Ова архитектура такође омогућава да га покренете локално и чак офлајн ако је потребно. Почните креирањем фајла `index.html` у истом фолдеру где сте сачували ваш `model.onnx` фајл.

1. У овом фајлу _index.html_, додајте следећи маркуп:

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

1. Сада, радећи унутар `body` тагова, додајте мало маркупа да прикажете листу чекбокса који одражавају неке састојке:

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

    Приметите да је сваком чекбоксу додељена вредност. Ово одражава индекс где се састојак налази према скупу података. На пример, јабука у овој азбучној листи заузима пету колону, па је њена вредност '4' јер почињемо бројати од 0. Можете консултовати [табелу састојака](../../../../4-Classification/data/ingredient_indexes.csv) да откријете индекс датог састојка.

    Настављајући рад у index.html фајлу, додајте блок скрипте где се модел позива након завршног затварајућег `</div>`.

1. Прво, увезите [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime се користи за омогућавање покретања ваших Onnx модела на широком спектру хардверских платформи, укључујући оптимизације и API за употребу.

1. Када је Runtime на месту, можете га позвати:

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

У овом коду, дешава се неколико ствари:

1. Направили сте низ од 380 могућих вредности (1 или 0) који се постављају и шаљу моделу за инференцију, у зависности од тога да ли је чекбокс означен.
2. Направили сте низ чекбокса и начин да утврдите да ли су означени у функцији `init` која се позива када апликација почне. Када је чекбокс означен, низ `ingredients` се мења да одражава изабрани састојак.
3. Направили сте функцију `testCheckboxes` која проверава да ли је било који чекбокс означен.
4. Користите функцију `startInference` када се притисне дугме и, ако је било који чекбокс означен, започињете инференцију.
5. Рутина инференције укључује:
   1. Постављање асинхроног учитавања модела
   2. Креирање Tensor структуре за слање моделу
   3. Креирање 'feeds' који одражавају `float_input` улаз који сте креирали приликом тренинга вашег модела (можете користити Netron да проверите то име)
   4. Слање ових 'feeds' моделу и чекање одговора

## Тестирајте вашу апликацију

Отворите терминал у Visual Studio Code-у у фолдеру где се налази ваш index.html фајл. Уверите се да имате [http-server](https://www.npmjs.com/package/http-server) инсталиран глобално, и укуцајте `http-server` на промпту. Требало би да се отвори localhost и можете видети вашу веб апликацију. Проверите која кухиња се препоручује на основу различитих састојака:

![веб апликација за састојке](../../../../4-Classification/4-Applied/images/web-app.png)

Честитамо, направили сте веб апликацију за препоруке са неколико поља. Одвојите мало времена да развијете овај систем!

## 🚀Изазов

Ваша веб апликација је веома минимална, па наставите да је развијате користећи састојке и њихове индексе из [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) података. Које комбинације укуса раде за креирање одређеног националног јела?

## [Квиз после предавања](https://ff-quizzes.netlify.app/en/ml/)

## Преглед и самостално учење

Иако је ова лекција само дотакла корисност креирања система за препоруке за састојке хране, ова област примена машинског учења је веома богата примерима. Прочитајте више о томе како се ови системи граде:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Задатак

[Направите нови систем за препоруке](assignment.md)

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако тежимо тачности, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не сносимо одговорност за било каква неспоразумевања или погрешна тумачења која могу произаћи из коришћења овог превода.