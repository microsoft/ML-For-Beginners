# ML modeli istifadə etmək üçün veb tətbiq yaradaq

Bu dərsdə ML modelini dünyamızdan tamamilə kənar data toplusu ilə öyrədəcəyik: _Ötən əsrdə UFO müşahidələri_, nümunəsi NUFORC verilənlər bazasından mənbə götürmüşdür.

Nələr öyrənəcəksiniz:

- Öyrədilmiş modeli necə pikl etmək ("turşuya qoymaq") (uzunmüddətli istifadə üçün) olar.
- Bu modeli Flask proqramında necə istifadə etmək olar

Məlumatları təmizləmək və modelimizi öyrətmək üçün noutbuklardan istifadəni davam etdirəcəyik. Prosesi bir addım irəli aparıb modeli “yabanı mühitdə” istifadə etməklə təcrübə apara bilərsiniz: Veb tətbiq daxilində bunu etmək üçün Flask istifadə edərək veb proqram qurmalısınız.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/?loc=az)

## Proqram yaratmaq

Maşın öyrənmə modellərini emal etmək üçün veb tətbiq yaratmağın bir neçə üsulu var. Veb arxitekturası modelin öyrədilməsinə birbaşa təsir göstərə bilər. Fərz edin ki, data ilə bağlı elmi qrupda proqram daxilində istifadə etməyinizi istədikləri modeli öyrədiblər.

### Mülahizələr

Soruşmalı olduğunuz çoxlu suallar var:

- **Bu veb yoxsa mobil tətbiqdir?** Əgər mobil tətbiq yaradırsınızsa və tətbiqi IoT konteksində istifadə edəcəksinizsə, modeli Android və ya iOS tətbiqdə istifadə etmək üçün [TensorFlow Lite](https://www.tensorflow.org/lite/) istifadə edə bilərsiniz.
- **Model harada yerləşəcək?** Bulud yoxsa lokal yaddaşda?
- **Oflayn dəstək.** Tətbiq oflayn da işləməlidirmi?
- **Modeli öyrətmək üçün hansı texnologiya istifadə edilmişdir?** Seçilmiş texnologiya istifadə etməli olduğunuz alətlərə təsir göstərə bilər.
    - **TensorFlow.** Məsələn, modeli TensorFlow istifadə edərək öyrədirsinizsə, ekosistem TensorFlow modelini veb proqramında istifadə etmək üçün çevirmək/dəyişdirmək imkanı verir. [TensorFlow.js](https://www.tensorflow.org/js/).
    - **PyTorch.** Əgər modeli [PyTorch](https://pytorch.org/) ilə öyrədirsinizsə, onu [ONNX](https://onnx.ai/) istifadə edərək JavaScript veb tətbiqində [Onnx Runtime](https://www.onnxruntime.ai/) işlədə bilən xüsusi (Open Neural Network Exchange) formata eksport edə bilərsiniz. Bu seçim növbəti dərsdə Scikit tərəfindən öyrənilən model üçün araşdırılacaq.
    - **Lobe.ai və ya Azure Custom Vision** Əgər modeli ML SaaS (Xidmət kimi Proqram) sistemi kimi [Lobe.ai](https://lobe.ai/) və ya [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) istifadə edərək öyrədirsinizsə, bu tip proqram təminatı bir çox platformalar üçün onu eksport etmək imkanı, həmçinin onlayn tətbiq tərəfindən buludda sorğulanan növbəli API yaradılmasını təmin edir.

Bütöv bir Flask veb tətbiqi yaradaraq veb brauzerdə modeli öyrətmək imkanınız var. Bunu həm də JavaScript konteksində TensorFlow.js istifadə edərək də edə bilərsiniz.

Məqsədə uyğun olaraq, Python əsaslı noutbuklarla işlədiyimizə görə öyrədilmiş modeli Python-da qurulmuş veb tətbiqi ilə oxuna bilən formatda daxil etmək üçün atmalı olduğunuz addımları araşdıraq.

## Alət

Bu tapşırığı yerinə yetirmək üçün iki alətə ehtiyacınız var: Python-da işləyən Flask və Pickle.

✅ [Flask](https://palletsprojects.com/p/flask/) nədir? Yaradıcıları tərəfindən 'mikro-çərçivə' kimi adlandırılan Flask, Python istifadə edərək veb çərçivələrin əsas xüsusiyyətlərini və veb səhifələr yaratmaq üçün şablon mexanizmini təmin edir. Flask yaratmaq üçün [bu öyrənmə moduluna](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) nəzər yetirin.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) nədir? Pickle 🥒, Python obyekt strukturunu seriallaşdıran və deseriallaşdıran bir Python moduludur. Modeli 'pickle' etdiyiniz zaman, onun strukturunu veb üçün seriallaşdırır və dəqiqləşdirirsiniz. Diqqət edin ki pickle mahiyyətcə təhlükəsiz deyil. Bu səbəbdən faylı 'un-pickle' etməyə çağrıldıqda diqqətli olmaq lazımdır. 'pickle' edilmiş fayl `.pkl` uzantısına sahib olur.

## Məşğələ - datanı təmizlə

Bu dərsdə [NUFORC](https://nuforc.org) (Milli UFO Melumatlar Mərkəzi) tərəfindən toplanmış 80,000 UFO görülməsi məlumatlarından istifadə edəcəksiniz. Bu məlumatlar UFO görmələrin maraqlı təsvirlərini əhatə edir, məsələn:

- **Əhatəli nümunə.** "Bir adam gecə otlu bir tarlada parlayan işıq şüasından çıxır və Texas Instruments dayanacağına tərəf qaçır".
- **Qısa nümunə.** "İşıq bizi təqib edir".

[ufos.csv](../data/ufos.csv) cədvəli görülmənin baş verdiyi `şəhər`, `ştat` and `ölkə`, obyektin `forması`, `coğrafi enlik` və `çoğrafi uzunluq` sütunlarına bölün.

Bu dərsə daxil olan boş [qeyd notbuk](../notebook.ipynb) verilmişdir.

1. `pandas`, `matplotlib`, və `numpy` kitabxanalarını və ufolar cədvəlini keçən dərs olduğu kimi daxil edin. Sadə verilənlər toplusuna nəzər yetirin:

    ```python
    import pandas as pd
    import numpy as np

    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. UFO məlumatlarını yeni başlıqlarla kiçik dataframə çevirin. `Ölkə` sütununda unikal dəyərləri yoxlayın.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

    ufos.Country.unique()
    ```

1. İndi isə, prosesi işlənməli olan datada lazımsız null dəyərləri silməklə və ancaq 1-60 saniyə arası görülmələri daxil etməklə qısalda bilərik:

    ```python
    ufos.dropna(inplace=True)

    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

    ufos.info()
    ```

1. Scikit-learn `LabelEncoder` kitabxanası ölkələr üçün növbəli dəyərləri rəqəmlərə keçirtmək üçündür:

    ✅ LabelEncoder datanı əlifba sırası ilə kodlaşdırır.

    ```python
    from sklearn.preprocessing import LabelEncoder

    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

    ufos.head()
    ```

    Datanız belə görünməlidir:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Məşğələ - modelini yarat

Artıq modeli öyrətmək üçün məlumatları tədris və sınaq qruplarına bölməyə başlaya bilərsiniz.

1. X vektorunda öyrətmək istədiyiniz 3 xüsusiyyəti seçin, və Y vektoru `Ölkə` olacaq. `Saniyələr`, `Coğrafi enlik`, `Coğrafi uzunluq` daxil edə bilməli və ölkənin `id` dəyərini geri qaytara bilməlisiniz.

    ```python
    from sklearn.model_selection import train_test_split

    Selected_features = ['Seconds','Latitude','Longitude']

    X = ufos[Selected_features]
    y = ufos['Country']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Məntiqi reqressiya testi istifadə etməklə modelinizi öyrədin:

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

Dəqiqlik o qədər də pis deyil **(təxminən 95%)**. Təəcüblü deyil ki `Ölkə` və `Coğrafi enlik, Coğrafi uzunluq` məlumatları əlaqəlidir.

Yaratdığımız model o qədər də inqilabi deyil, çünki `Ölkə`ni `Coğrafi enlik` və `Coğrafi uzunluq`dan çıxara bilməliyik. Bununla belə, təmizlədiyiniz, daxil etdiyiniz, daha sonra ilkin məlumatlardan öyrətməyə çalışdığımız məlumat datanı məşq etdirmək üçün faydalıdır. Bu modeli veb proqramında istifadə edək.

## Məşğələ - modelini 'pikl' edək

İndi, modelinizi _pikl_ etməyin vaxtıdır! Bunu bir neçə kod sətiri ilə edə bilərsiniz. Pikl(turşu) halına gələndən sonra, pikl modelinizi yükləyin və onu saniyə, coğrafi enlik və coğrafi uzunluq dəyərlərini ehtiva edən məlumat massivi ilə sınaqdan keçirin.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model **'3'** qaytarır, hansıki Birləşmiş Krallığın kodudur. İnanılmaz! 👽

## Məşğələ - Flask tətbiq yarat

İndi isə modeli çağırmaq və oxşar nəticələri qaytarmaq üçün Flask proqramı yarada bilərsiniz, lakin vizual olaraq daha xoşagəlimli olmalıdır.

1. **web-app** adlanan bir qovluq yaratmaqla başlayın. Qovluq _notebook.ipynb_ faylının yanında _ufo-model.pkl_ faylınızın yerləşdiyi yerdə olmalıdır.

1. Həmin qovluqda daha üç qovluq yaradın: daxilində **css** olan **static** qovluğu və **templates**. İndi aşağıdakı formada fayl və istiqamətlər olmalıdır:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Yekunlaşmış tətbiq üçün mənbə faylına nəzər yetirin.

1. _web-app_ qovluğunda yaratmalı olduğunuz ilk fayl **requirements.txt** faylıdır. JavaScript  tətbiqində _package.json_ faylı kimi, tətbiq tərəfindən tələb olunan faylları sıralayır. **requirements.txt** faylında aşağıdakı sətirləri əlavə edin:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. İndi isə, _web-app_ faylına keçməklə tətbiqi işə salın:

    ```bash
    cd web-app
    ```

1. Terminalınızda _requirements.txt_ faylında qeyd olunmuş kitabxanaları yükləmək üçün `pip install` yazın:

    ```bash
    pip install -r requirements.txt
    ```

1. Artıq daha üç fayl yaradaraq tətbiqi hazırlamağı yekunlaşdıra bilərik:

    1. **app.py** faylını əsas qovluqda yaradın.
    2. **index.html** faylını _templates_ kataloqunda yaradın.
    3. **styles.css** faylını _static/css_ kataloqunda yaradın.

1. _styles.css_ faylını aşağıdakı bir neçə dizayn skriptləri ilə yaradın:

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

1. Daha sonra, _index.html_ faylını yaradın:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO görülməsini öncədən müəyyən etmək! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>

      <body>
        <div class="grid">

          <div class="box">

            <p>Saniyə dəyəri, coğrafi enlik və corafi uzunluğa görə, hansı ölkə UFO gördüyünü bildirmişdir?</p>

            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">UFO görünən ölkəni təxmin et</button>
            </form>

            <p>{{ prediction_text }}</p>

          </div>

        </div>

      </body>
    </html>
    ```

    Bu faylda şablonlaşdırmaya diqqət yetirin. Tətbiq tərəfindən dəyişənlər ətrafında 'saqqal' sintaksisini görəcəksiniz, məsələn təxmin mətni kimi: `{{}}`. Həmçinin təxminin `/predict` istiqamətinə göndərən xüsusi form da var.

   Yekunda modelin emalı və proqnozların göstərilməsini təmin edən python faylını yaratmağa hazırsınız:

1. `app.py` faylında aşağıdakıları əlavə edin:

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

    > 💡 Ipucu: Flask istifadə edərək veb tətbiqi işə salanda [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) əlavə edərkən proqramınıza etdiyiniz istənilən dəyişiklik ani olaraq tətbiq olunacaq. Serveri yenidən başlatmağa ehtiyac olmayacaq. Ehtiyyatlı olun! Bu formatı real mühitdə aktiv etmək doğru olmaz.

Əgər `python app.py` və ya `python3 app.py` işlədirsinizsə - veb-serveriniz daxili serverdə (lokalda) işə başlayır və bu zaman UFO-ların harada göründüyü barədə sualınıza cavab almaq üçün qısa formanı doldura bilərsiniz!

Bunu etmədən öncə, `app.py` faylına nəzər yetirək:

1. Əvvəlcə lazımi resurslar yüklənir və proqram başlayır.
1. Sonra model daxil edilir.
1. Daha sonra index.html lokal mühitdə göstərilir.

`/predict` istiqamətində form yerləşdirilərkən birdən çox proses baş verir:

1. Forma dəyişənləri toplanır və numpy massivinə çevrilir. Daha sonra modelə göndərilir və proqnoz geri qaytarılır.
2. Göstərilməsini istədiyimiz ölkələr onların proqnozlaşdırılan ölkə kodundan istifadə etməklə oxunaqlı mətn kimi yenidən göstərilir və həmin dəyər şablonda göstərilmək üçün index.html-ə geri göndərilir.

Flask və pikl model ilə bu şəkildə bir model istifadə etmək nisbətən sadədir. Ən çətini proqnoz almaq üçün modelə göndərilməli olan məlumatların hansı formada olduğunu başa düşməkdir. Bütün bunlar modelin necə öyrədildiyindən asılıdır. Proqnoz əldə etmək üçün daxil edilməli üç məlumat nöqtəsi var.

Peşəkar şəraitdə modeli öyrədən insanlarla onu veb və ya mobil proqramda istifadə edənlər arasında yaxşı ünsiyyətin nə qədər zəruri olduğunu görmək olar. Bizim şərtlər daxilində bu yalnız bir nəfərdir, siz!

---

## 🚀 Rubrika

Noutbukda işləyərək modeli Flask proqramına daxil etmək yerinə onu birbaşa Flask proqramında öyrədə bilərsiniz! Modeli `train` adlı istiqamət üzrə proqram daxilində öyrətmək üçün, məsələn, məlumatlarınız təmizləndikdən sonra Python kodunuzu notbukda çevirməyə cəhd edin. Bu metodu tətbiq etməyin müsbət və mənfi tərəfləri nələrdir?

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/?loc=az)

## Təkrarla və özün öyrən

ML modellərini dərk etmək üçün veb tətbiq yaratmağın bir çox üsulu var. Maşın öyrənməsində istifadə etmək üçün veb tətbiqi yaratmağa JavaScript və ya Python-dan istifadə edə biləcəyiniz üsulların siyahısını hazırlayın. Arxitekturaya nəzər salın: model proqramda qalmalıdır yoxsa buludda? Ən son versiyaya necə daxil olardınız? Tətbiq olunan ML veb həlli üçün arxitektura modelini çəkin.

## Tapşırıq

[Başqa bir model yoxlayın](assignment.az.md)
