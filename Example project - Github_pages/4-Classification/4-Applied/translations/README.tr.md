# Mutfak Önerici Bir Web Uygulaması Oluşturun

Bu derste, önceki derslerde öğrendiğiniz bazı yöntemleri kullanarak, bu seri boyunca kullanılan leziz mutfak veri setiyle bir sınıflandırma modeli oluşturacaksınız. Ayrıca, kaydettiğiniz modeli kullanmak üzere, Onnx'un web çalışma zamanından yararlanan küçük bir web uygulaması oluşturacaksınız.

Makine öğreniminin en faydalı pratik kullanımlarından biri, önerici/tavsiyeci sistemler oluşturmaktır ve bu yöndeki ilk adımınızı bugün atabilirsiniz!

[![Önerici Sistemler Tanıtımı](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Applied ML")

> :movie_camera: Video için yukarıdaki fotoğrafa tıklayın

## [Ders öncesi kısa sınavı](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/?loc=tr)

Bu derste şunları öğreneceksiniz:

- Bir model nasıl oluşturulur ve Onnx modeli olarak kaydedilir
- Modeli denetlemek için Netron nasıl kullanılır
- Modeliniz çıkarım için bir web uygulamasında nasıl kullanılabilir

## Modelinizi oluşturun

Uygulamalı Makine Öğrenimi sistemleri oluşturmak, bu teknolojilerden kendi iş sistemleriniz için yararlanmanızın önemli bir parçasıdır. Onnx kullanarak modelleri kendi web uygulamalarınız içerisinde kullanabilirsiniz (Böylece gerektiğinde çevrim dışı bir içerikte kullanabilirsiniz.).

[Önceki bir derste](../../../3-Web-App/1-Web-App/README.md) UFO gözlemleriyle ilgili bir Regresyon modeli oluşturmuş, "pickle" kullanmış ve bir Flask uygulamasında kullanmıştınız. Bu mimariyi bilmek çok faydalıdır, ancak bu tam yığın Python uygulamasıdır ve bir JavaScript uygulaması kullanımı gerekebilir.

Bu derste, çıkarım için temel JavaScript tabanlı bir sistem oluşturabilirsiniz. Ancak öncelikle, bir model eğitmeniz ve Onnx ile kullanım için dönüştürmeniz gerekmektedir.

## Alıştırma - sınıflandırma modelini eğitin

Öncelikle, kullandığımız temiz mutfak veri setini kullanarak bir sınıflandırma modeli eğitin.

1. Faydalı kütüphaneler almakla başlayın:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Scikit-learn modelinizi Onnx biçimine dönüştürmeyi sağlamak için '[skl2onnx](https://onnx.ai/sklearn-onnx/)'a ihtiyacınız var.

1. Sonra, önceki derslerde yaptığınız şekilde, `read_csv()` kullanarak bir CSV dosyasını okuyarak veriniz üzerinde çalışın:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. İlk iki gereksiz sütunu kaldırın ve geriye kalan veriyi 'X' olarak kaydedin:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Etiketleri 'y' olarak kaydedin:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Eğitme rutinine başlayın

İyi doğruluğu olan 'SVC' kütüphanesini kullanacağız.

1. Scikit-learn'den uygun kütüphaneleri alın:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Eğitme ve sınama kümelerini ayırın:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Önceki derste yaptığınız gibi bir SVC Sınıflandırma modeli oluşturun:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Şimdi, `predict()` fonksiyonunu çağırarak modelinizi sınayın:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Modelin kalitesini kontrol etmek için bir sınıflandırma raporu bastırın:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Daha önce de gördüğümüz gibi, doğruluk iyi:

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

### Modelinizi Onnx'a dönüştürün

Dönüştürmeyi uygun Tensor sayısıyla yaptığınıza emin olun. Bu veri seti listelenmiş 380 malzeme içeriyor, dolayısıyla bu sayıyı `FloatTensorType` içinde belirtmeniz gerekiyor:

1. 380 tensor sayısını kullanarak dönüştürün.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. onx'u oluşturun ve **model.onnx** diye bir dosya olarak kaydedin:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Not olarak, dönüştürme senaryonuzda [seçenekler](https://onnx.ai/sklearn-onnx/parameterized.html) geçirebilirsiniz. Biz bu durumda, 'nocl' parametresini True ve 'zipmap' parametresini 'False' olarak geçirdik. Bu bir sınıflandırma modeli olduğundan, bir sözlük listesi üreten (gerekli değil) ZipMap'i kaldırma seçeneğiniz var. `nocl`, modelde sınıf bilgisinin barındırılmasını ifade eder. `nocl` parametresini 'True' olarak ayarlayarak modelinizin boyutunu küçültün.

Tüm not defterini çalıştırmak şimdi bir Onnx modeli oluşturacak ve bu klasöre kaydedecek.

## Modelinizi inceleyin

Onnx modelleri Visual Studio code'da pek görünür değiller ama birçok araştırmacının modelin doğru oluştuğundan emin olmak üzere modeli görselleştirmek için kullandığı çok iyi bir yazılım var. [Netron](https://github.com/lutzroeder/Netron)'u indirin ve model.onnx dosyanızı açın. 380 girdisi ve sınıflandırıcısıyla basit modelinizin görselleştirildiğini görebilirsiniz:

![Netron görseli](../images/netron.png)

Netron, modellerinizi incelemek için faydalı bir araçtır.

Şimdi, bu düzenli modeli web uygulamanızda kullanmak için hazırsınız. Buzdolabınıza baktığınızda ve verilen bir mutfak için artık malzemelerin hangi birleşimini kullanabileceğinizi bulmayı denediğinizde kullanışlı olacak bir uygulama oluşturalım. Bu birleşim modeliniz tarafından belirlenecek.

## Önerici bir web uygulaması oluşturun

Modelinizi doğrudan bir web uygulamasında kullanabilirsiniz. Bu mimari, modelinizi yerelde ve hatta gerektiğinde çevrim dışı çalıştırabilmenizi de sağlar. `model.onnx` dosyanızı kaydettiğiniz klasörde `index.html` dosyasını oluşturarak başlayın.

1. Bu _index.html_ dosyasında aşağıdaki işaretlemeyi ekleyin:

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

1. Şimdi, `body` etiketleri içinde çalışarak, bazı malzemeleri ifade eden bir onay kutusu listesi göstermek için küçük bir işaretleme ekleyin:

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

    Her bir onay kutusuna bir değer verildiğine dikkat edin. Bu, veri setine göre malzemenin bulunduğu indexi ifade eder. Örneğin bu alfabetik listede elma beşinci sütundadır, dolayısıyla onun değeri '4'tür çünkü saymaya 0'dan başlıyoruz. Verilen malzemenin indexini görmek için [malzemeler tablosuna](../../data/ingredient_indexes.csv) başvurabilirsiniz.

    index.html dosyasındaki işinize devam ederek, son `</div>` kapamasından sonra modelinizin çağrılacağı bir script bloğu ekleyin.

1. Öncelikle, [Onnx Runtime](https://www.onnxruntime.ai/) alın:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.8.0-dev.20210608.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime, Onnx modelinizin, eniyileştirmeler ve kullanmak için bir API da dahil olmak üzere, geniş bir donanım platform yelpazesinde çalışmasını sağlamak için kullanılır.

1. Runtime uygun hale geldiğinde, onu çağırabilirsiniz:

    ```javascript
    <script>
                const ingredients = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
                const checks = [].slice.call(document.querySelectorAll('.checkbox'));
    
                // use an async context to call onnxruntime functions.
                function init() {
                    
                    checks.forEach(function (checkbox, index) {
                        checkbox.onchange = function () {
                            if (this.checked) {
                                var index = checkbox.value;
    
                                if (index !== -1) {
                                    ingredients[index] = 1;
                                }
                                console.log(ingredients)
                            }
                            else {
                                var index = checkbox.value;
    
                                if (index !== -1) {
                                    ingredients[index] = 0;
                                }
                                console.log(ingredients)
                            }
                        }
                    })
                }
    
                function testCheckboxes() {
                        for (var i = 0; i < checks.length; i++)
                            if (checks[i].type == "checkbox")
                                if (checks[i].checked)
                                    return true;
                        return false;
                }
    
                async function startInference() {
    
                    let checked = testCheckboxes()
    
                    if (checked) {
    
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
                        console.log(`failed to inference ONNX model: ${e}.`);
                    }
                }
                else alert("Please check an ingredient")
                    
                }
        init();
               
            </script>
    ```

Bu kodda birçok şey gerçekleşiyor:

1. Ayarlanması ve çıkarım için modele gönderilmesi için, bir malzeme onay kutusunun işaretli olup olmadığına bağlı 380 muhtemel değerden (ya 1 ya da 0) oluşan bir dizi oluşturdunuz.
2. Onay kutularından oluşan bir dizi ve uygulama başladığında çağrılan bir `init` fonksiyonunda işaretli olup olmadıklarını belirleme yolu oluşturdunuz. Eğer onay kutusu işaretliyse, `ingredients` dizisi, seçilen malzemeyi ifade etmek üzere değiştirilir.
3. Herhangi bir onay kutusunun işaretli olup olmadığını kontrol eden bir `testCheckboxes` fonksiyonu oluşturdunuz.
4. Düğmeye basıldığında o fonksiyonu kullanıyor ve eğer herhangi bir onay kutusu işaretlenmişse çıkarıma başlıyorsunuz.
5. Çıkarım rutini şunları içerir:
   1. Makinenin eşzamansız bir yüklemesini ayarlama
   2. Modele göndermek için bir Tensor yapısı oluşturma
   3. Modelinizi eğitirken oluşturduğunuz `float_input` (Bu adı doğrulamak için Netron kullanabilirsiniz.) girdisini ifade eden 'feeds' oluşturma
   4. Bu 'feeds'i modele gönderme ve yanıt için bekleme

## Uygulamanızı test edin

index.html dosyanızın olduğu klasördeyken Visual Studio Code'da bir terminal açın. Global kapsamda [http-server](https://www.npmjs.com/package/http-server) indirilmiş olduğundan emin olun ve istemde `http-server` yazın. Bir yerel ana makine açılmalı ve web uygulamanızı görebilirsiniz. Çeşitli malzemeleri baz alarak hangi mutfağın önerildiğine bakın:

![malzeme web uygulaması](../images/web-app.png)

Tebrikler, birkaç değişkenle bir 'önerici' web uygulaması oluşturdunuz! Bu sistemi oluşturmak için biraz zaman ayırın!
## :rocket: Meydan okuma

Web uygulamanız çok minimal, bu yüzden [ingredient_indexes](../../data/ingredient_indexes.csv) verisinden malzemeleri ve indexlerini kullanarak web uygulamanızı oluşturmaya devam edin. Verilen bir ulusal yemeği yapmak için hangi tat birleşimleri işe yarıyor?

## [Ders sonrası kısa sınavı](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/?loc=tr)

## Gözden Geçirme & Kendi Kendine Çalışma

Bu dersin sadece yemek malzemeleri için bir öneri sistemi oluşturmanın olanaklarına değinmesiyle beraber, makine öğrenimi uygulamalarının bu alanı örnekler açısından çok zengin. Bu sistemlerin nasıl oluşturulduğu hakkında biraz daha okuyun:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Ödev 

[Yeni bir önerici oluşturun](assignment.tr.md)
