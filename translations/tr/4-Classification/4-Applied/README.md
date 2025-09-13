<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61bdec27ed2da8b098cd9065405d9bb0",
  "translation_date": "2025-09-06T07:59:41+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "tr"
}
-->
# Bir Mutfak Tavsiye Web Uygulaması Oluşturun

Bu derste, önceki derslerde öğrendiğiniz bazı teknikleri kullanarak ve bu seride kullanılan lezzetli mutfak veri setiyle bir sınıflandırma modeli oluşturacaksınız. Ayrıca, kaydedilmiş bir modeli kullanmak için Onnx'in web çalışma zamanını kullanarak küçük bir web uygulaması geliştireceksiniz.

Makine öğreniminin en faydalı pratik kullanımlarından biri öneri sistemleri oluşturmaktır ve bugün bu yönde ilk adımı atabilirsiniz!

[![Bu web uygulamasını sunuyoruz](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "Uygulamalı ML")

> 🎥 Yukarıdaki görüntüye tıklayın: Jen Looper sınıflandırılmış mutfak verilerini kullanarak bir web uygulaması oluşturuyor

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

Bu derste öğreneceksiniz:

- Bir model nasıl oluşturulur ve Onnx modeli olarak nasıl kaydedilir
- Netron'u kullanarak modeli nasıl inceleyeceğiniz
- Modelinizi bir web uygulamasında çıkarım için nasıl kullanacağınız

## Modelinizi Oluşturun

Uygulamalı ML sistemleri oluşturmak, bu teknolojileri iş sistemlerinizde kullanmanın önemli bir parçasıdır. Onnx kullanarak modelleri web uygulamalarınızda (ve gerekirse çevrimdışı bir bağlamda) kullanabilirsiniz.

[Önceki bir derste](../../3-Web-App/1-Web-App/README.md), UFO gözlemleri hakkında bir Regresyon modeli oluşturmuş, "pickle" yapmış ve bunu bir Flask uygulamasında kullanmıştınız. Bu mimariyi bilmek çok faydalı olsa da, bu tam yığın bir Python uygulamasıdır ve gereksinimleriniz bir JavaScript uygulamasını kullanmayı içerebilir.

Bu derste, çıkarım için temel bir JavaScript tabanlı sistem oluşturabilirsiniz. Ancak önce bir model eğitmeniz ve Onnx ile kullanmak üzere dönüştürmeniz gerekiyor.

## Egzersiz - Sınıflandırma Modeli Eğitin

Öncelikle, kullandığımız temizlenmiş mutfak veri setini kullanarak bir sınıflandırma modeli eğitin.

1. Faydalı kütüphaneleri içe aktararak başlayın:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Scikit-learn modelinizi Onnx formatına dönüştürmeye yardımcı olmak için '[skl2onnx](https://onnx.ai/sklearn-onnx/)' gereklidir.

1. Ardından, önceki derslerde yaptığınız gibi bir CSV dosyasını `read_csv()` kullanarak işleyin:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. İlk iki gereksiz sütunu kaldırın ve kalan verileri 'X' olarak kaydedin:

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Etiketleri 'y' olarak kaydedin:

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Eğitim Rutini Başlatın

'SVC' kütüphanesini kullanacağız çünkü iyi bir doğruluğa sahiptir.

1. Scikit-learn'den uygun kütüphaneleri içe aktarın:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Eğitim ve test setlerini ayırın:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Önceki derste yaptığınız gibi bir SVC Sınıflandırma modeli oluşturun:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Şimdi modelinizi test edin, `predict()` çağırarak:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Modelin kalitesini kontrol etmek için bir sınıflandırma raporu yazdırın:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Daha önce gördüğümüz gibi, doğruluk iyidir:

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

### Modelinizi Onnx'e Dönüştürün

Dönüşümü doğru Tensor numarasıyla yaptığınızdan emin olun. Bu veri setinde 380 malzeme listelenmiştir, bu nedenle `FloatTensorType` içinde bu numarayı belirtmeniz gerekir:

1. 380 tensor numarası kullanarak dönüştürün.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Onx oluşturun ve **model.onnx** dosyası olarak kaydedin:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Not: Dönüşüm betiğinizde [seçenekler](https://onnx.ai/sklearn-onnx/parameterized.html) geçirebilirsiniz. Bu durumda, 'nocl' True ve 'zipmap' False olarak ayarlandı. Bu bir sınıflandırma modeli olduğundan, bir liste sözlükleri üreten ZipMap'i kaldırma seçeneğiniz vardır (gerekli değil). `nocl`, sınıf bilgilerinin modele dahil edilmesini ifade eder. Modelinizin boyutunu azaltmak için `nocl`'yi 'True' olarak ayarlayın.

Tüm not defterini çalıştırmak artık bir Onnx modeli oluşturacak ve bu klasöre kaydedecektir.

## Modelinizi Görüntüleyin

Onnx modelleri Visual Studio Code'da çok görünür değildir, ancak birçok araştırmacının modeli doğru bir şekilde oluşturulduğundan emin olmak için kullandığı çok iyi bir ücretsiz yazılım vardır. [Netron](https://github.com/lutzroeder/Netron)'u indirin ve model.onnx dosyanızı açın. Basit modelinizi, 380 girdisi ve sınıflandırıcısı ile görselleştirilmiş olarak görebilirsiniz:

![Netron görseli](../../../../4-Classification/4-Applied/images/netron.png)

Netron, modellerinizi görüntülemek için faydalı bir araçtır.

Artık bu harika modeli bir web uygulamasında kullanmaya hazırsınız. Buzdolabınıza baktığınızda ve kalan malzemelerinizin hangi kombinasyonunu kullanarak modeliniz tarafından belirlenen bir mutfağı pişirebileceğinizi anlamaya çalıştığınızda işe yarayacak bir uygulama oluşturalım.

## Bir Tavsiye Web Uygulaması Oluşturun

Modelinizi doğrudan bir web uygulamasında kullanabilirsiniz. Bu mimari, gerekirse yerel olarak ve hatta çevrimdışı çalıştırmanıza da olanak tanır. `index.html` dosyasını, `model.onnx` dosyanızı kaydettiğiniz aynı klasörde oluşturarak başlayın.

1. Bu dosyada _index.html_, aşağıdaki işaretlemeyi ekleyin:

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

1. Şimdi, `body` etiketleri içinde, bazı malzemeleri yansıtan bir dizi onay kutusu göstermek için biraz işaretleme ekleyin:

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

    Her onay kutusuna bir değer verildiğini fark edin. Bu, veri setine göre malzemenin bulunduğu indeksi yansıtır. Örneğin, elma bu alfabetik listede beşinci sütunu işgal eder, bu nedenle değeri '4' olur çünkü 0'dan saymaya başlıyoruz. Belirli bir malzemenin indeksini keşfetmek için [malzeme elektronik tablosunu](../../../../4-Classification/data/ingredient_indexes.csv) inceleyebilirsiniz.

    Çalışmanızı index.html dosyasında sürdürerek, son kapanış `</div>` etiketinden sonra bir script bloğu ekleyin.

1. İlk olarak, [Onnx Runtime](https://www.onnxruntime.ai/) içe aktarın:

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > Onnx Runtime, Onnx modellerinizi geniş bir donanım platformu yelpazesinde çalıştırmayı sağlamak için optimizasyonlar ve bir API sunar.

1. Runtime yerleştirildikten sonra, onu çağırabilirsiniz:

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

Bu kodda birkaç şey oluyor:

1. Seçilen malzeme onay kutusuna bağlı olarak çıkarım için modele gönderilecek 380 olası değer (1 veya 0) içeren bir dizi oluşturdunuz.
2. Uygulama başladığında çağrılan bir `init` fonksiyonunda bir dizi onay kutusu ve bunların işaretlenip işaretlenmediğini belirleme yöntemi oluşturdunuz. Bir onay kutusu işaretlendiğinde, `ingredients` dizisi seçilen malzemeyi yansıtacak şekilde değişir.
3. Herhangi bir onay kutusunun işaretlenip işaretlenmediğini kontrol eden bir `testCheckboxes` fonksiyonu oluşturdunuz.
4. Bir düğmeye basıldığında `startInference` fonksiyonunu kullanıyorsunuz ve herhangi bir onay kutusu işaretlenmişse çıkarımı başlatıyorsunuz.
5. Çıkarım rutini şunları içerir:
   1. Modelin asenkron yüklenmesini ayarlama
   2. Modele gönderilecek bir Tensor yapısı oluşturma
   3. Eğitim sırasında oluşturduğunuz `float_input` girdisini yansıtan 'feeds' oluşturma (adı doğrulamak için Netron'u kullanabilirsiniz)
   4. Bu 'feeds'leri modele gönderme ve bir yanıt bekleme

## Uygulamanızı Test Edin

Visual Studio Code'da index.html dosyanızın bulunduğu klasörde bir terminal oturumu açın. [http-server](https://www.npmjs.com/package/http-server)'ın global olarak yüklü olduğundan emin olun ve istemde `http-server` yazın. Bir localhost açılmalı ve web uygulamanızı görüntüleyebilirsiniz. Çeşitli malzemelere göre hangi mutfağın önerildiğini kontrol edin:

![malzeme web uygulaması](../../../../4-Classification/4-Applied/images/web-app.png)

Tebrikler, birkaç alan içeren bir 'tavsiye' web uygulaması oluşturdunuz. Bu sistemi geliştirmek için biraz zaman ayırın!

## 🚀Meydan Okuma

Web uygulamanız oldukça minimal, bu yüzden [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv) verilerinden malzemeleri ve indekslerini kullanarak geliştirmeye devam edin. Hangi lezzet kombinasyonları belirli bir ulusal yemeği oluşturmak için işe yarıyor?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu ders, yemek malzemeleri için bir öneri sistemi oluşturmanın faydasını sadece yüzeysel olarak ele aldı, ancak bu ML uygulamaları alanı örneklerle oldukça zengindir. Bu sistemlerin nasıl oluşturulduğu hakkında daha fazla bilgi edinin:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Ödev 

[Yeni bir öneri sistemi oluşturun](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.