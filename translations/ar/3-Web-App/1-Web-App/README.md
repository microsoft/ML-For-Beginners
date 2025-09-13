<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-04T20:48:36+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "ar"
}
-->
# بناء تطبيق ويب لاستخدام نموذج تعلم الآلة

في هذه الدرس، ستقوم بتدريب نموذج تعلم الآلة على مجموعة بيانات غير مألوفة: _مشاهدات الأجسام الطائرة المجهولة خلال القرن الماضي_، مأخوذة من قاعدة بيانات NUFORC.

ستتعلم:

- كيفية "تخزين" نموذج مدرب باستخدام Pickle
- كيفية استخدام هذا النموذج في تطبيق Flask

سنواصل استخدام دفاتر الملاحظات لتنظيف البيانات وتدريب النموذج، ولكن يمكنك أخذ العملية خطوة إضافية من خلال استكشاف استخدام النموذج "في العالم الحقيقي"، أي في تطبيق ويب.

للقيام بذلك، تحتاج إلى بناء تطبيق ويب باستخدام Flask.

## [اختبار ما قبل المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## بناء التطبيق

هناك عدة طرق لبناء تطبيقات ويب لاستهلاك نماذج تعلم الآلة. قد تؤثر بنية الويب الخاصة بك على الطريقة التي يتم بها تدريب النموذج. تخيل أنك تعمل في شركة حيث قامت مجموعة علوم البيانات بتدريب نموذج يريدون منك استخدامه في تطبيق.

### اعتبارات

هناك العديد من الأسئلة التي تحتاج إلى طرحها:

- **هل هو تطبيق ويب أم تطبيق جوال؟** إذا كنت تبني تطبيقًا جوالًا أو تحتاج إلى استخدام النموذج في سياق إنترنت الأشياء، يمكنك استخدام [TensorFlow Lite](https://www.tensorflow.org/lite/) واستخدام النموذج في تطبيق Android أو iOS.
- **أين سيقيم النموذج؟** في السحابة أم محليًا؟
- **الدعم دون اتصال.** هل يجب أن يعمل التطبيق دون اتصال؟
- **ما هي التقنية المستخدمة لتدريب النموذج؟** قد تؤثر التقنية المختارة على الأدوات التي تحتاج إلى استخدامها.
    - **استخدام TensorFlow.** إذا كنت تدرب نموذجًا باستخدام TensorFlow، على سبيل المثال، يوفر هذا النظام البيئي القدرة على تحويل نموذج TensorFlow للاستخدام في تطبيق ويب باستخدام [TensorFlow.js](https://www.tensorflow.org/js/).
    - **استخدام PyTorch.** إذا كنت تبني نموذجًا باستخدام مكتبة مثل [PyTorch](https://pytorch.org/)، لديك خيار تصديره بتنسيق [ONNX](https://onnx.ai/) (تبادل الشبكة العصبية المفتوحة) للاستخدام في تطبيقات ويب JavaScript التي يمكنها استخدام [Onnx Runtime](https://www.onnxruntime.ai/). سيتم استكشاف هذا الخيار في درس مستقبلي لنموذج مدرب باستخدام Scikit-learn.
    - **استخدام Lobe.ai أو Azure Custom Vision.** إذا كنت تستخدم نظام SaaS (البرمجيات كخدمة) مثل [Lobe.ai](https://lobe.ai/) أو [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) لتدريب نموذج، يوفر هذا النوع من البرمجيات طرقًا لتصدير النموذج للعديد من المنصات، بما في ذلك بناء API مخصص للاستعلام عنه في السحابة بواسطة تطبيقك عبر الإنترنت.

لديك أيضًا فرصة لبناء تطبيق ويب كامل باستخدام Flask يمكنه تدريب النموذج نفسه في متصفح الويب. يمكن القيام بذلك أيضًا باستخدام TensorFlow.js في سياق JavaScript.

بالنسبة لأغراضنا، نظرًا لأننا عملنا مع دفاتر ملاحظات تعتمد على Python، دعنا نستكشف الخطوات التي تحتاج إلى اتخاذها لتصدير نموذج مدرب من دفتر ملاحظات إلى تنسيق يمكن قراءته بواسطة تطبيق ويب مبني باستخدام Python.

## الأدوات

لهذه المهمة، تحتاج إلى أداتين: Flask وPickle، وكلاهما يعمل على Python.

✅ ما هو [Flask](https://palletsprojects.com/p/flask/)؟ يُعرف بأنه "إطار عمل صغير" من قبل منشئيه، يوفر Flask الميزات الأساسية لإطارات عمل الويب باستخدام Python ومحرك قوالب لبناء صفحات الويب. ألقِ نظرة على [وحدة التعلم هذه](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) لممارسة البناء باستخدام Flask.

✅ ما هو [Pickle](https://docs.python.org/3/library/pickle.html)؟ Pickle 🥒 هو وحدة Python تقوم بتسلسل وإلغاء تسلسل هيكل كائن Python. عندما تقوم "بتخزين" نموذج، فإنك تقوم بتسلسل أو تسطيح هيكله للاستخدام على الويب. كن حذرًا: Pickle ليس آمنًا بطبيعته، لذا كن حذرًا إذا طُلب منك "إلغاء تخزين" ملف. يحتوي الملف المخزن على اللاحقة `.pkl`.

## تمرين - تنظيف البيانات

في هذا الدرس ستستخدم بيانات من 80,000 مشاهدة للأجسام الطائرة المجهولة، تم جمعها بواسطة [NUFORC](https://nuforc.org) (المركز الوطني للإبلاغ عن الأجسام الطائرة المجهولة). تحتوي هذه البيانات على أوصاف مثيرة للاهتمام لمشاهدات الأجسام الطائرة المجهولة، على سبيل المثال:

- **وصف طويل كمثال.** "رجل يخرج من شعاع ضوء يضيء على حقل عشبي في الليل ويركض نحو موقف سيارات Texas Instruments".
- **وصف قصير كمثال.** "الأضواء طاردتنا".

تشمل جدول البيانات [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) أعمدة حول `المدينة`، `الولاية` و`الدولة` حيث حدثت المشاهدة، شكل الجسم، و`خط العرض` و`خط الطول`.

في [دفتر الملاحظات](../../../../3-Web-App/1-Web-App/notebook.ipynb) الفارغ المرفق في هذا الدرس:

1. قم باستيراد `pandas`، `matplotlib`، و`numpy` كما فعلت في الدروس السابقة واستيراد جدول بيانات الأجسام الطائرة المجهولة. يمكنك إلقاء نظرة على مجموعة بيانات نموذجية:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. قم بتحويل بيانات الأجسام الطائرة المجهولة إلى إطار بيانات صغير مع عناوين جديدة. تحقق من القيم الفريدة في حقل `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. الآن، يمكنك تقليل كمية البيانات التي نحتاج إلى التعامل معها عن طريق حذف أي قيم فارغة واستيراد المشاهدات فقط بين 1-60 ثانية:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. قم باستيراد مكتبة `LabelEncoder` الخاصة بـ Scikit-learn لتحويل القيم النصية للدول إلى أرقام:

    ✅ يقوم LabelEncoder بترميز البيانات أبجديًا

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    يجب أن تبدو بياناتك كما يلي:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## تمرين - بناء النموذج

الآن يمكنك الاستعداد لتدريب نموذج عن طريق تقسيم البيانات إلى مجموعة التدريب والاختبار.

1. اختر ثلاث ميزات تريد التدريب عليها كمتجه X، وسيكون المتجه y هو `Country`. تريد أن تكون قادرًا على إدخال `Seconds`، `Latitude` و`Longitude` والحصول على معرف الدولة كإجابة.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. قم بتدريب النموذج باستخدام الانحدار اللوجستي:

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

الدقة ليست سيئة **(حوالي 95%)**، وليس من المستغرب، حيث أن `Country` و`Latitude/Longitude` مترابطان.

النموذج الذي أنشأته ليس ثوريًا جدًا حيث يجب أن تكون قادرًا على استنتاج `Country` من `Latitude` و`Longitude`، ولكنه تمرين جيد لمحاولة التدريب من بيانات خام قمت بتنظيفها وتصديرها، ثم استخدام هذا النموذج في تطبيق ويب.

## تمرين - تخزين النموذج

الآن، حان الوقت لتخزين النموذج! يمكنك القيام بذلك في بضعة أسطر من التعليمات البرمجية. بمجرد تخزينه، قم بتحميل النموذج المخزن واختبره مقابل مجموعة بيانات نموذجية تحتوي على قيم للثواني، خط العرض وخط الطول.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

النموذج يعيد **'3'**، وهو رمز الدولة للمملكة المتحدة. مذهل! 👽

## تمرين - بناء تطبيق Flask

الآن يمكنك بناء تطبيق Flask لاستدعاء النموذج وإرجاع نتائج مشابهة، ولكن بطريقة أكثر جاذبية بصريًا.

1. ابدأ بإنشاء مجلد يسمى **web-app** بجانب ملف _notebook.ipynb_ حيث يوجد ملف _ufo-model.pkl_.

1. في هذا المجلد، قم بإنشاء ثلاثة مجلدات أخرى: **static**، مع مجلد **css** بداخله، و**templates**. يجب أن يكون لديك الآن الملفات والمجلدات التالية:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ راجع مجلد الحل للحصول على عرض للتطبيق النهائي

1. أول ملف يتم إنشاؤه في مجلد _web-app_ هو ملف **requirements.txt**. مثل _package.json_ في تطبيق JavaScript، يسرد هذا الملف التبعيات المطلوبة للتطبيق. في **requirements.txt** أضف الأسطر:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. الآن، قم بتشغيل هذا الملف عن طريق التنقل إلى _web-app_:

    ```bash
    cd web-app
    ```

1. في الطرفية الخاصة بك، اكتب `pip install`، لتثبيت المكتبات المدرجة في _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. الآن، أنت جاهز لإنشاء ثلاثة ملفات أخرى لإنهاء التطبيق:

    1. قم بإنشاء **app.py** في الجذر.
    2. قم بإنشاء **index.html** في مجلد _templates_.
    3. قم بإنشاء **styles.css** في مجلد _static/css_.

1. قم ببناء ملف _styles.css_ ببعض الأنماط:

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

1. بعد ذلك، قم ببناء ملف _index.html_:

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

    ألقِ نظرة على القوالب في هذا الملف. لاحظ صياغة "mustache" حول المتغيرات التي سيتم توفيرها بواسطة التطبيق، مثل نص التنبؤ: `{{}}`. هناك أيضًا نموذج ينشر تنبؤًا إلى مسار `/predict`.

    أخيرًا، أنت جاهز لبناء ملف Python الذي يدير استهلاك النموذج وعرض التنبؤات:

1. في `app.py` أضف:

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

    > 💡 نصيحة: عند إضافة [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) أثناء تشغيل تطبيق الويب باستخدام Flask، ستنعكس أي تغييرات تجريها على التطبيق فورًا دون الحاجة إلى إعادة تشغيل الخادم. احذر! لا تقم بتمكين هذا الوضع في تطبيق الإنتاج.

إذا قمت بتشغيل `python app.py` أو `python3 app.py` - يبدأ خادم الويب الخاص بك محليًا، ويمكنك ملء نموذج قصير للحصول على إجابة لسؤالك الملح حول مكان مشاهدة الأجسام الطائرة المجهولة!

قبل القيام بذلك، ألقِ نظرة على أجزاء `app.py`:

1. أولاً، يتم تحميل التبعيات ويبدأ التطبيق.
1. ثم يتم استيراد النموذج.
1. ثم يتم عرض index.html على المسار الرئيسي.

على مسار `/predict`، تحدث عدة أمور عند نشر النموذج:

1. يتم جمع متغيرات النموذج وتحويلها إلى مصفوفة numpy. ثم يتم إرسالها إلى النموذج ويتم إرجاع تنبؤ.
2. يتم إعادة عرض الدول التي نريد عرضها كنص قابل للقراءة من رمز الدولة المتوقع، ويتم إرسال تلك القيمة مرة أخرى إلى index.html ليتم عرضها في القالب.

استخدام النموذج بهذه الطريقة، مع Flask ونموذج مخزن، هو أمر بسيط نسبيًا. أصعب شيء هو فهم شكل البيانات التي يجب إرسالها إلى النموذج للحصول على تنبؤ. يعتمد ذلك كله على كيفية تدريب النموذج. يحتوي هذا النموذج على ثلاث نقاط بيانات يجب إدخالها للحصول على تنبؤ.

في بيئة احترافية، يمكنك أن ترى كيف أن التواصل الجيد ضروري بين الأشخاص الذين يدربون النموذج وأولئك الذين يستهلكونه في تطبيق ويب أو جوال. في حالتنا، هو شخص واحد فقط، أنت!

---

## 🚀 تحدي

بدلاً من العمل في دفتر ملاحظات واستيراد النموذج إلى تطبيق Flask، يمكنك تدريب النموذج مباشرة داخل تطبيق Flask! حاول تحويل كود Python في دفتر الملاحظات، ربما بعد تنظيف البيانات، لتدريب النموذج من داخل التطبيق على مسار يسمى `train`. ما هي الإيجابيات والسلبيات لمتابعة هذه الطريقة؟

## [اختبار ما بعد المحاضرة](https://ff-quizzes.netlify.app/en/ml/)

## المراجعة والدراسة الذاتية

هناك العديد من الطرق لبناء تطبيق ويب لاستهلاك نماذج تعلم الآلة. قم بعمل قائمة بالطرق التي يمكنك من خلالها استخدام JavaScript أو Python لبناء تطبيق ويب للاستفادة من تعلم الآلة. فكر في البنية: هل يجب أن يبقى النموذج في التطبيق أم يعيش في السحابة؟ إذا كان الخيار الأخير، كيف ستصل إليه؟ ارسم نموذجًا معماريًا لحل تعلم الآلة في تطبيق ويب.

## الواجب

[جرب نموذجًا مختلفًا](assignment.md)

---

**إخلاء المسؤولية**:  
تمت ترجمة هذا المستند باستخدام خدمة الترجمة الآلية [Co-op Translator](https://github.com/Azure/co-op-translator). بينما نسعى لتحقيق الدقة، يرجى العلم أن الترجمات الآلية قد تحتوي على أخطاء أو معلومات غير دقيقة. يجب اعتبار المستند الأصلي بلغته الأصلية هو المصدر الموثوق. للحصول على معلومات حساسة أو هامة، يُوصى بالاستعانة بترجمة بشرية احترافية. نحن غير مسؤولين عن أي سوء فهم أو تفسيرات خاطئة تنشأ عن استخدام هذه الترجمة.