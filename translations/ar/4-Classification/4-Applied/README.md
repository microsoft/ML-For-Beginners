<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-04T20:49:44+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "ar"
}
-->
# بناء تطبيق ويب لتوصية المأكولات

في هذه الدرس، ستقوم ببناء نموذج تصنيف باستخدام بعض التقنيات التي تعلمتها في الدروس السابقة ومع مجموعة بيانات المأكولات الشهية التي تم استخدامها طوال هذه السلسلة. بالإضافة إلى ذلك، ستقوم ببناء تطبيق ويب صغير لاستخدام النموذج المحفوظ، مستفيدًا من تشغيل الويب الخاص بـ Onnx.

واحدة من أكثر الاستخدامات العملية المفيدة لتعلم الآلة هي بناء أنظمة التوصية، ويمكنك اتخاذ الخطوة الأولى في هذا الاتجاه اليوم!

[![عرض هذا التطبيق](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> 🎥 انقر على الصورة أعلاه لمشاهدة الفيديو: جين لوبر تبني تطبيق ويب باستخدام بيانات المأكولات المصنفة

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

في هذا الدرس ستتعلم:

- كيفية بناء نموذج وحفظه كـ Onnx model
- كيفية استخدام Netron لفحص النموذج
- كيفية استخدام النموذج الخاص بك في تطبيق ويب للاستنتاج

## بناء النموذج الخاص بك

بناء أنظمة تعلم الآلة التطبيقية هو جزء مهم من الاستفادة من هذه التقنيات في أنظمة الأعمال الخاصة بك. يمكنك استخدام النماذج داخل تطبيقات الويب الخاصة بك (وبالتالي استخدامها في سياق غير متصل إذا لزم الأمر) باستخدام Onnx.

في [درس سابق](../../3-Web-App/1-Web-App/README.md)، قمت ببناء نموذج انحدار حول مشاهدات UFO، وقمت بـ "تخزينه"، واستخدمته في تطبيق Flask. بينما هذه البنية مفيدة جدًا للمعرفة، فهي تطبيق Python كامل، وقد تتطلب احتياجاتك استخدام تطبيق JavaScript.

في هذا الدرس، يمكنك بناء نظام أساسي يعتمد على JavaScript للاستنتاج. ولكن أولاً، تحتاج إلى تدريب نموذج وتحويله للاستخدام مع Onnx.

## تمرين - تدريب نموذج التصنيف

أولاً، قم بتدريب نموذج تصنيف باستخدام مجموعة بيانات المأكولات المنظفة التي استخدمناها.

1. ابدأ باستيراد المكتبات المفيدة:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    تحتاج إلى '[skl2onnx](https://onnx.ai/sklearn-onnx/)' للمساعدة في تحويل نموذج Scikit-learn الخاص بك إلى تنسيق Onnx.

1. ثم، قم بمعالجة بياناتك بنفس الطريقة التي قمت بها في الدروس السابقة، عن طريق قراءة ملف CSV باستخدام `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. قم بإزالة أول عمودين غير ضروريين واحفظ البيانات المتبقية كـ 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. احفظ التصنيفات كـ 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### بدء روتين التدريب

سنستخدم مكتبة 'SVC' التي تتمتع بدقة جيدة.

1. قم باستيراد المكتبات المناسبة من Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. قم بفصل مجموعات التدريب والاختبار:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. قم ببناء نموذج تصنيف SVC كما فعلت في الدرس السابق:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. الآن، اختبر النموذج الخاص بك باستخدام `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. اطبع تقرير التصنيف للتحقق من جودة النموذج:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    كما رأينا سابقًا، الدقة جيدة:

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

### تحويل النموذج الخاص بك إلى Onnx

تأكد من إجراء التحويل باستخدام الرقم الصحيح للتنسور. تحتوي هذه المجموعة من البيانات على 380 مكونًا مدرجًا، لذا تحتاج إلى تدوين هذا الرقم في `FloatTensorType`:

1. قم بالتحويل باستخدام رقم تنسور 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. قم بإنشاء ملف onx واحفظه كملف **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > ملاحظة، يمكنك تمرير [خيارات](https://onnx.ai/sklearn-onnx/parameterized.html) في نص التحويل الخاص بك. في هذه الحالة، قمنا بتمرير 'nocl' ليكون True و 'zipmap' ليكون False. نظرًا لأن هذا نموذج تصنيف، لديك خيار إزالة ZipMap الذي ينتج قائمة من القواميس (غير ضروري). `nocl` يشير إلى تضمين معلومات التصنيف في النموذج. قم بتقليل حجم النموذج الخاص بك عن طريق تعيين `nocl` إلى 'True'.

تشغيل دفتر الملاحظات بالكامل الآن سيبني نموذج Onnx ويحفظه في هذا المجلد.

## عرض النموذج الخاص بك

نماذج Onnx ليست مرئية جدًا في Visual Studio Code، ولكن هناك برنامج مجاني جيد يستخدمه العديد من الباحثين لتصور النموذج للتأكد من أنه تم بناؤه بشكل صحيح. قم بتنزيل [Netron](https://github.com/lutzroeder/Netron) وافتح ملف model.onnx الخاص بك. يمكنك رؤية النموذج البسيط الخاص بك مصورًا، مع مدخلاته الـ 380 والمصنف المدرج:

![عرض Netron](../../../../4-Classification/4-Applied/images/netron.png)

Netron هو أداة مفيدة لعرض النماذج الخاصة بك.

الآن أنت جاهز لاستخدام هذا النموذج الرائع في تطبيق ويب. دعنا نبني تطبيقًا سيكون مفيدًا عندما تنظر في ثلاجتك وتحاول معرفة أي مجموعة من المكونات المتبقية يمكنك استخدامها لطهي طبق معين، كما يحدده النموذج الخاص بك.

## بناء تطبيق ويب للتوصية

يمكنك استخدام النموذج الخاص بك مباشرة في تطبيق ويب. هذه البنية تتيح لك أيضًا تشغيله محليًا وحتى في وضع غير متصل إذا لزم الأمر. ابدأ بإنشاء ملف `index.html` في نفس المجلد حيث قمت بتخزين ملف `model.onnx`.

1. في هذا الملف _index.html_، أضف العلامات التالية:

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

1. الآن، أثناء العمل داخل علامات `body`، أضف بعض العلامات لعرض قائمة من مربعات الاختيار التي تعكس بعض المكونات:

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

    لاحظ أن كل مربع اختيار تم إعطاؤه قيمة. هذا يعكس الفهرس حيث يتم العثور على المكون وفقًا لمجموعة البيانات. التفاح، على سبيل المثال، في هذه القائمة الأبجدية، يحتل العمود الخامس، لذا قيمته هي '4' لأننا نبدأ العد من 0. يمكنك الرجوع إلى [جدول بيانات المكونات](../../../../4-Classification/data/ingredient_indexes.csv) لاكتشاف فهرس مكون معين.

    أثناء استمرار العمل في ملف index.html، أضف كتلة نصية حيث يتم استدعاء النموذج بعد الإغلاق النهائي لـ `</div>`.

1. أولاً، قم باستيراد [Onnx Runtime](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > يتم استخدام Onnx Runtime لتمكين تشغيل نماذج Onnx الخاصة بك عبر مجموعة واسعة من منصات الأجهزة، بما في ذلك التحسينات وواجهة برمجة التطبيقات للاستخدام.

1. بمجرد أن يكون Runtime في مكانه، يمكنك استدعاؤه:

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

في هذا الكود، هناك عدة أمور تحدث:

1. قمت بإنشاء مصفوفة من 380 قيمة ممكنة (1 أو 0) ليتم إعدادها وإرسالها إلى النموذج للاستنتاج، بناءً على ما إذا كان مربع اختيار المكون محددًا.
2. قمت بإنشاء مصفوفة من مربعات الاختيار وطريقة لتحديد ما إذا كانت محددة في وظيفة `init` التي يتم استدعاؤها عند بدء التطبيق. عندما يتم تحديد مربع اختيار، يتم تعديل مصفوفة `ingredients` لتعكس المكون المختار.
3. قمت بإنشاء وظيفة `testCheckboxes` التي تتحقق مما إذا كان أي مربع اختيار قد تم تحديده.
4. تستخدم وظيفة `startInference` عندما يتم الضغط على الزر، وإذا تم تحديد أي مربع اختيار، تبدأ الاستنتاج.
5. يتضمن روتين الاستنتاج:
   1. إعداد تحميل غير متزامن للنموذج
   2. إنشاء بنية Tensor لإرسالها إلى النموذج
   3. إنشاء 'feeds' التي تعكس الإدخال `float_input` الذي قمت بإنشائه عند تدريب النموذج الخاص بك (يمكنك استخدام Netron للتحقق من الاسم)
   4. إرسال هذه 'feeds' إلى النموذج وانتظار الرد

## اختبار التطبيق الخاص بك

افتح جلسة طرفية في Visual Studio Code في المجلد حيث يوجد ملف index.html الخاص بك. تأكد من أن لديك [http-server](https://www.npmjs.com/package/http-server) مثبتًا عالميًا، واكتب `http-server` في الموجه. يجب أن يفتح localhost ويمكنك عرض تطبيق الويب الخاص بك. تحقق من الطبق الموصى به بناءً على المكونات المختلفة:

![تطبيق ويب المكونات](../../../../4-Classification/4-Applied/images/web-app.png)

تهانينا، لقد قمت بإنشاء تطبيق ويب للتوصية مع بعض الحقول. خذ بعض الوقت لتطوير هذا النظام!

## 🚀التحدي

تطبيق الويب الخاص بك بسيط جدًا، لذا استمر في تطويره باستخدام المكونات وفهارسها من بيانات [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). ما هي تركيبات النكهات التي تعمل على إنشاء طبق وطني معين؟

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

بينما تناول هذا الدرس فقط فائدة إنشاء نظام توصية لمكونات الطعام، فإن هذا المجال من تطبيقات تعلم الآلة غني بالأمثلة. اقرأ المزيد حول كيفية بناء هذه الأنظمة:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## الواجب 

[قم ببناء نظام توصية جديد](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.