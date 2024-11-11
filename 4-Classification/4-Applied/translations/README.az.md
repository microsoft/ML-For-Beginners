# Mətbəx tövsiyyə edən Veb tətbiq yaradaq

Bu dərsdə əvvəlki dərslərdə öyrəndiyimiz bəzi üsullar və bu bölmədə istifadə olunan ləzzətli mətbəx data seti ilə qruplaşdırıcı modeli quracağıq. Bundan əlavə, Onnx-in veb runtime versiyasından istifadə edərək yadda saxlanmış modeli tətbiq etmək üçün kiçik veb tətbiqi yaratmalıyıq.

Maşın öyrənməsinin ən faydalı praktik istifadələrindən biri tövsiyə sistemlərinin qurulmasıdır və biz bu istiqamətdə ilk addımı atacağıq!

[![Veb tətbiqin təqdimatı](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Tətbiqi MÖ")

> 🎥 Video üçün yuxardakı şəklə klikləyin: Jen Looper qruplaşdırılmış mətbəx datası istifadə etməklə veb tətbiq yaradır.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/?loc=az)

Bu dərsdə öyrənəcəksiniz:

- Veb modeli necə yaratmalı və onu Onnx model kimi necə yadda saxlamalı
- Modeli nəzərdən keçirməklə Netron modeli necə istifadə etməli
- Veb tətbiqdə nəticə çıxarmaq üçün modeli necə istifadə etməli

## Modelinizi yaradın

Tətbiq olunan ML sistemlərinin yaradılması biznes sistemləri üçün bu texnologiyalardan istifadə etməyin vacib hissəsidir. Onnx-dən istifadə etməklə veb proqramlar daxilində modellərdən istifadə etmək olar (ehtiyac olduqda onları oflayn mühitdə istifadə etmək lazımdır).

[Əvvəlki dərsdə](../../../3-Web-App/1-Web-App/translations/README.az.md) UFO ilə bağlı reqresiya modelini yaradıb, "pickle" edib Flask tətbiqi daxilində istifadə etmişdik. Bu arxitekturanı bilmək çox faydalı olsa da, tamamilə Python proqramıdır. Lakin, sizdən Javascript tətbiqi də yazmaq tələb oluna bilər.

Bu dərsdə nəticə almaq üçün Javascript əsasında veb tətbiq yarada bilərik. Əvvəlcə modeli öyrətmək və Onnx ilə işləməsi üçün hazırlamaq lazımdır.

## Tapşırıq - qruplaşdırıcı modelini öyrədin

İlk öncə qruplaşdırıcı modeli təmizlənmiş mətbəxlər data dəstini istifadə etməklə öyrədək.

1. Lazımi kitabxanaları əlavə etməklə başlayaq:

    ```python
    !pip install skl2onnx
    import pandas as pd
    ```

    Scikit-learn modelini Onnx formatına keçirmək üçün '[skl2onnx](https://onnx.ai/sklearn-onnx/)' köməyinə ehtiyacımız var.

1. CSV faylını oxumaq üçün `read_csv()` istifadə etməklə əvvəlki dərsdə etdiyimiz kimi dataset ilə işləməliyik:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. İlk iki lazımsız sütunu ləğv edərək yerdə qalan datanı 'X' kimi yadda saxlayaq:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Etiketləri isə 'y' kimi qeyd edək:

    ```python
    y = data[['cuisine']]
    y.head()
    ```

### Təlim rutininə başlayaq

Biz yüksək dəqiqliyə malik "SVC" kitabxanasından istifadə edəcəyik.

1. Scikit-learn vasitəsilə lazımlı kitabxanaları əlavə edək:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Təlim və test setlərini ayıraq:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Əvvəlki dərsdə etdiyiniz kimi SVC təsnifat modelini yaradaq:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. `predict()` testini çağırmaqla modeli test edək:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Modelin keyfiyyətini yoxlamaq üçün qruplaşdırıcı hesabatını çap edək:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Daha əvvəl də gördüyümüz kimi dəqiqlik dərəcəsi çox yaxşıdır:

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

### Modeli Onnx-ə çevirək

Düzgün Tensor nömrəsi ilə çevirmə etdiyimizə əmin olmalıyıq. Bu data dəstində 380 inqrediyent sadalanıb, ona görə də bu rəqəmi `FloatTensorType`-da qeyd etmək lazımdır:

1. 380 tensor nömrəsindən istifadə edərək çevirməni həyata keçirək:

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. onx yaradıb faylı **model.onnx** kimi yadda saxlayaq:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Konvertasiya skriptində [options](https://onnx.ai/sklearn-onnx/parameterized.html) ötürə bilərik. Bu halda, 'nocl' True və 'zipmap' False olmasını təmin etdik.
    Bu qruplaşdırıcı model olduğundan lüğətlərin siyahısını yaradan ZipMap-ı silmək seçimimiz var (vacib deyil). `nocl` modelə daxil edilmiş sinif məlumatlarına istinad edir. `nocl`-i 'True' olaraq təyin edərək modelinizin ölçüsünü azaldaq.

Bütün notbuku işə salmaqla Onnx modelini yaradaraq onu bu qovluqda yadda saxlaya biləcəyik.

## Modelə nəzər yetirək

Onnx modelləri Visual Studio kodunda visual olaraq görünmür. Bir çox tədqiqatçı modelin düzgün qurulmasını təmin etmək üçün onu vizuallaşdırmağa istifadə olunan pulsuz proqramlardan istifadə edir. [Netron](https://github.com/lutzroeder/Netron) yükləyək və model.onnx faylını açaq. Sadə modelin 380 girişi və siyahıda göstərilən təsnifatı ilə vizuallaşdırıldığını görə bilərsiniz:

![Netron vizualı](../images/netron.png)

Netron modelləri vizual olaraq görmək üçün bir vasitədir.

İndi bu səliqəli modeli veb proqramda istifadə etməyə hazırıq. Gəlin soyuducuya baxan zaman işimizə yarayacaq bir proqram yaradaq və modelə görə müəyyən edilmiş mətbəx növünü bişirmək üçün qalıq inqrediyentlərin hansı kombinasiyasından istifadə edə biləcəyinizi anlamağa çalışaq.

## Tövsiyə verən bir veb tətbiq yaradaq

Modeli birbaşa veb proqramında istifadə edə bilərik. Bu arxitektura həm də lazım olduqda onu lokal və ya oflayn rejimdə işə salmağa imkan verir. `model.onnx` faylını saxladığımız eyni qovluqda `index.html` faylı yaratmaqla başlayaq.

1. Bu faylda _index.html_, aşağıdakı kodu əlavə edək:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Mətbəx Uyğunlaşdırıcısı</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. İndi isə `body` teqləri ilə işləyərək ingridientləri əks etdirən bölmələri əlavə edək:

    ```html
    <h1>Soyuducuya bax. Nə bişirə bilərsən?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>alma</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>armud</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>vişnə</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>çəmən otu</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>düyü şərabı</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soya sousu</label>
                </div>

                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cirə</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">Hansı növ mətbəxə müraciət edə bilərsən?</button>
            </div>
    ```

    Qeyd edək ki, hər bir xanaya dəyər verilir. Bu, verilənlər bazasına uyğun olaraq tərkib hissəsinin tapıldığı indeksi əks etdirir. Məsələn, 'Alma' əlifba siyahısında beşinci sütundadır, ona görə də biz 0-dan saymağa başladığımız zaman verilmiş tərkib indeksinə görə onun dəyəri '4'dür. [inqrediyentlər cədvəli](../../data/ingredient_indexes.csv)ni nəzərdən keçirməklə verilmiş tərkib hissələrini görə bilərsiniz.

    index.html faylında işi davam etdirərək, `</div>` yekunlaşdırıcı bağlandıqdan sonra modelin çağrıldığı skript blokunu əlavə edin.

1. İlk öncə, [Onnx Runtime](https://www.onnxruntime.ai/) əlavə edək:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script>
    ```

    > Onnx Runtime, Onnx modelləri optimallaşdırmaq və istifadə edə bilmək üçün API daxil olmaqla geniş çeşidli platformalarda işləməyə imkan vermək üçün istifadə olunur.

1. Runtime yerini aldıqdan sonra işə sala bilərik:

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

Bu kodda eyni anda bir neçə prosses baş verir:

1. Mümkün 380 dəyərdən (1 və ya 0) ibarət massiv yaratdıq və inqrediyentin işarələnib göstərilməsindən asılı olaraq nəticə çıxarmaq üçün modelə göndəriləcək.
2. Qeyd qutuları massivinin proqram başlayanda çağırılan `init` funksiyasında yoxlanılıb-yoxlanılmadığını müəyyən etmək üçün bir üsul yaratdıq. Yoxlama qutusu işarələndikdə, `ingredients` massivi seçilmiş inqrediyenti əks etdirmək üçün yenilənir.
3. `testCheckboxes` funksiyası yaradaraq hansısa qeyd qutusunun işarələnib-işarələnmədiyini yoxlanılır.
4. `startInference` funksiyasını istifadə edərək hansısa düymə basılanda və ya hər hansı işarə qutusu seçiləndə hərəkətə keçilir.
5. Nəticə çıxarma ardıcıllığı aşağıdakıları əhatə edir:
    1. Modelin asinxron yükünün qurulması
    2. Modelə göndərmək üçün Tensor strukturunun yaradılması
    3. Modeli öyrədərkən yaratdığımız `float_input` daxiletməsini əks etdirən `feeds` datasının yaradılması (bu adı yoxlamaq üçün Netron-dan istifadə edə bilərsiniz)
    4. `feeds` məlumatını modelə göndərmək və ondan cavab gözləmək

## Proqramı test edək

index.html faylının yerləşdiyi qovluqda Visual Studio-da yeni terminal açaq və qlobalda quraşdırılmış [http-server](https://www.npmjs.com/package/http-server) mövcud olduğundan əmin olaq. Sorğuda `http-server` yazdıqdan sonra Localhost açılacaq. Beləliklə Veb tətbiqə nəzər yetirə bilərik. Müxtəlif inqrediyentlər əsasında hansı mətbəxin tövsiyə olunduğunu yoxlayaq:

![ingrediyent veb tətbiq](../images/web-app.png)

Təbriklər, 'tövsiyyə edən' veb tətbiq yaratdıq. İndi isə sistemi qurmaq üçün daha çox vaxt ayıraq!

## 🚀 Məşğələ

Hazırki veb tətbiq minimal səviyyədədir. Bu səbəbdən inqrediyentlər və onların indekslərinə əsasən [ingredient_indexes](../../data/ingredient_indexes.csv) datasından istifadə etməklə davam etmək lazımdır. Verilmiş milli yeməyi hazırlamaq üçün hansı dadlardan istifadə edilməlidir, müəyyən edək.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/?loc=az)

## Təkrarlayın və özünüz öyrənin

Biz bu dərsdə qida maddələri üçün tövsiyə sisteminin yaradılmasına səthi toxunsaq da, ML tətbiqlərinin bu sahədə töhvələri çox zəngindir. Bu sistemlərin necə qurulduğu haqqında daha ətraflı oxuyun:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tapşırıq

[Yeni tövsiyyə sistemi yaradın](assignment.az.md)
