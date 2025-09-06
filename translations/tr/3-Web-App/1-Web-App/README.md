<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-06T07:57:46+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "tr"
}
-->
# Bir ML Modelini Kullanmak için Bir Web Uygulaması Oluşturun

Bu derste, NUFORC'un veritabanından alınan _son yüzyıldaki UFO gözlemleri_ verileri üzerinde bir ML modeli eğiteceksiniz.

Öğrenecekleriniz:

- Eğitilmiş bir modeli 'pickle'lama
- Bu modeli bir Flask uygulamasında kullanma

Verileri temizlemek ve modelimizi eğitmek için defterleri kullanmaya devam edeceğiz, ancak süreci bir adım öteye taşıyarak modeli bir web uygulamasında kullanmayı keşfedebilirsiniz.

Bunu yapmak için Flask kullanarak bir web uygulaması oluşturmanız gerekecek.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## Bir Uygulama Oluşturma

Makine öğrenimi modellerini tüketmek için web uygulamaları oluşturmanın birkaç yolu vardır. Web mimariniz, modelinizin nasıl eğitildiğini etkileyebilir. Bir işletmede çalıştığınızı ve veri bilimi ekibinin bir model eğittiğini ve bu modeli bir uygulamada kullanmanızı istediğini hayal edin.

### Dikkate Alınması Gerekenler

Sormanız gereken birçok soru var:

- **Bu bir web uygulaması mı yoksa mobil uygulama mı?** Eğer bir mobil uygulama geliştiriyorsanız veya modeli bir IoT bağlamında kullanmanız gerekiyorsa, [TensorFlow Lite](https://www.tensorflow.org/lite/) kullanabilir ve modeli bir Android veya iOS uygulamasında kullanabilirsiniz.
- **Model nerede barındırılacak?** Bulutta mı yoksa yerel olarak mı?
- **Çevrimdışı destek.** Uygulama çevrimdışı çalışmak zorunda mı?
- **Modeli eğitmek için hangi teknoloji kullanıldı?** Seçilen teknoloji, kullanmanız gereken araçları etkileyebilir.
    - **TensorFlow kullanımı.** Örneğin, TensorFlow kullanarak bir model eğitiyorsanız, bu ekosistem, bir web uygulamasında kullanılmak üzere bir TensorFlow modelini [TensorFlow.js](https://www.tensorflow.org/js/) kullanarak dönüştürme yeteneği sağlar.
    - **PyTorch kullanımı.** Eğer [PyTorch](https://pytorch.org/) gibi bir kütüphane kullanarak bir model oluşturuyorsanız, modeli [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formatında dışa aktarma seçeneğiniz vardır. Bu format, [Onnx Runtime](https://www.onnxruntime.ai/) kullanabilen JavaScript web uygulamalarında kullanılabilir. Bu seçenek, Scikit-learn ile eğitilmiş bir model için ilerideki bir derste keşfedilecektir.
    - **Lobe.ai veya Azure Custom Vision kullanımı.** Eğer bir ML SaaS (Hizmet Olarak Yazılım) sistemi olan [Lobe.ai](https://lobe.ai/) veya [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) kullanarak bir model eğitiyorsanız, bu tür yazılımlar, modeli birçok platform için dışa aktarma yolları sağlar. Buna, çevrimiçi uygulamanız tarafından bulutta sorgulanabilecek özel bir API oluşturma da dahildir.

Ayrıca, bir web tarayıcısında modeli kendisi eğitebilecek bir Flask web uygulaması oluşturma fırsatınız da var. Bu, JavaScript bağlamında TensorFlow.js kullanılarak da yapılabilir.

Bizim amacımız için, Python tabanlı defterlerle çalıştığımızdan, eğitilmiş bir modeli böyle bir defterden Python ile oluşturulmuş bir web uygulaması tarafından okunabilir bir formata dışa aktarmak için gereken adımları inceleyelim.

## Araçlar

Bu görev için iki araca ihtiyacınız var: Flask ve Pickle, her ikisi de Python üzerinde çalışır.

✅ [Flask](https://palletsprojects.com/p/flask/) nedir? Yaratıcıları tarafından bir 'mikro-çerçeve' olarak tanımlanan Flask, Python kullanarak web çerçevelerinin temel özelliklerini ve web sayfaları oluşturmak için bir şablon motoru sağlar. Flask ile uygulama geliştirmeyi pratik etmek için [bu Öğrenme modülüne](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) göz atın.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) nedir? Pickle 🥒, bir Python nesne yapısını serileştiren ve serileştirmeyi kaldıran bir Python modülüdür. Bir modeli 'pickle'ladığınızda, yapısını webde kullanım için serileştirir veya düzleştirirsiniz. Dikkatli olun: pickle doğası gereği güvenli değildir, bu yüzden bir dosyayı 'un-pickle'lamanız istendiğinde dikkatli olun. Picklelanmış bir dosya `.pkl` uzantısına sahiptir.

## Alıştırma - Verilerinizi Temizleyin

Bu derste, [NUFORC](https://nuforc.org) (Ulusal UFO Raporlama Merkezi) tarafından toplanan 80.000 UFO gözlemi verilerini kullanacaksınız. Bu verilerde UFO gözlemlerine dair ilginç açıklamalar bulunuyor, örneğin:

- **Uzun örnek açıklama.** "Gece bir çimenlik alana ışık huzmesiyle inen bir adam, Texas Instruments otoparkına doğru koşuyor."
- **Kısa örnek açıklama.** "Işıklar bizi kovaladı."

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) elektronik tablosu, gözlemin gerçekleştiği `şehir`, `eyalet` ve `ülke`, nesnenin `şekli` ve `enlem` ile `boylam` bilgilerini içeren sütunlar içerir.

Bu derste yer alan boş [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) dosyasında:

1. Daha önceki derslerde olduğu gibi `pandas`, `matplotlib` ve `numpy` modüllerini içe aktarın ve UFO elektronik tablosunu yükleyin. Örnek bir veri setine göz atabilirsiniz:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. UFO verilerini yeni başlıklarla küçük bir veri çerçevesine dönüştürün. `Country` alanındaki benzersiz değerleri kontrol edin.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Şimdi, ele almamız gereken veri miktarını azaltmak için boş değerleri düşürerek ve yalnızca 1-60 saniye arasındaki gözlemleri içe aktararak verileri azaltabilirsiniz:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Ülkeler için metin değerlerini bir sayıya dönüştürmek için Scikit-learn'ün `LabelEncoder` kütüphanesini içe aktarın:

    ✅ LabelEncoder verileri alfabetik olarak kodlar

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Verileriniz şu şekilde görünmelidir:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Alıştırma - Modelinizi Oluşturun

Şimdi, verileri eğitim ve test gruplarına ayırarak bir model eğitmeye hazır olabilirsiniz.

1. X vektörünüz olarak eğitmek istediğiniz üç özelliği seçin ve y vektörü `Country` olacaktır. `Seconds`, `Latitude` ve `Longitude` girdilerini alıp bir ülke kimliği döndürmek istiyorsunuz.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Modelinizi lojistik regresyon kullanarak eğitin:

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

Doğruluk oranı fena değil **(%95 civarında)**, şaşırtıcı değil, çünkü `Country` ve `Latitude/Longitude` arasında bir ilişki var.

Oluşturduğunuz model çok devrimsel değil, çünkü bir ülkeyi `Latitude` ve `Longitude` değerlerinden çıkarabilmelisiniz, ancak bu, temizlediğiniz ham verilerden bir model eğitme, dışa aktarma ve ardından bu modeli bir web uygulamasında kullanma alıştırması yapmak için iyi bir egzersizdir.

## Alıştırma - Modelinizi 'Pickle'layın

Şimdi, modelinizi _pickle_lama zamanı! Bunu birkaç satır kodla yapabilirsiniz. Model _pickle_landıktan sonra, picklelanmış modeli yükleyin ve saniye, enlem ve boylam değerlerini içeren bir örnek veri dizisine karşı test edin.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model **'3'** döndürüyor, bu da Birleşik Krallık'ın ülke kodu. Harika! 👽

## Alıştırma - Bir Flask Uygulaması Oluşturun

Şimdi, modelinizi çağırıp benzer sonuçları daha görsel olarak hoş bir şekilde döndüren bir Flask uygulaması oluşturabilirsiniz.

1. _notebook.ipynb_ dosyasının yanına **web-app** adlı bir klasör oluşturun ve _ufo-model.pkl_ dosyanız burada bulunsun.

1. Bu klasörde üç klasör daha oluşturun: **static** (içinde bir **css** klasörü ile) ve **templates**. Şimdi aşağıdaki dosya ve dizinlere sahip olmalısınız:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Bitmiş uygulamanın görünümünü görmek için çözüm klasörüne bakın

1. _web-app_ klasöründe oluşturulacak ilk dosya **requirements.txt** dosyasıdır. Bir JavaScript uygulamasındaki _package.json_ gibi, bu dosya uygulama tarafından gereken bağımlılıkları listeler. **requirements.txt** dosyasına şu satırları ekleyin:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Şimdi, bu dosyayı çalıştırmak için _web-app_ dizinine gidin:

    ```bash
    cd web-app
    ```

1. Terminalinizde `pip install` yazarak _requirements.txt_ dosyasında listelenen kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

1. Şimdi, uygulamayı tamamlamak için üç dosya daha oluşturabilirsiniz:

    1. Kök dizinde **app.py** oluşturun.
    2. _templates_ dizininde **index.html** oluşturun.
    3. _static/css_ dizininde **styles.css** oluşturun.

1. _styles.css_ dosyasını birkaç stil ile oluşturun:

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

1. Ardından, _index.html_ dosyasını oluşturun:

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

    Bu dosyadaki şablonlamaya bir göz atın. Uygulama tarafından sağlanacak değişkenlerin etrafındaki 'bıyık' sözdizimine dikkat edin, örneğin tahmin metni: `{{}}`. Ayrıca, `/predict` rotasına bir tahmin gönderen bir form da var.

    Son olarak, modeli tüketen ve tahminlerin görüntülenmesini sağlayan Python dosyasını oluşturabilirsiniz:

1. `app.py` dosyasına şunları ekleyin:

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

    > 💡 İpucu: Flask kullanarak web uygulamasını çalıştırırken [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) eklediğinizde, uygulamanızda yaptığınız değişiklikler sunucuyu yeniden başlatmaya gerek kalmadan hemen yansıtılır. Dikkat! Bu modu bir üretim uygulamasında etkinleştirmeyin.

`python app.py` veya `python3 app.py` çalıştırırsanız - yerel olarak web sunucunuz başlar ve UFO'ların nerede görüldüğüne dair merak ettiğiniz soruya cevap almak için kısa bir form doldurabilirsiniz!

Bunu yapmadan önce, `app.py` dosyasının bölümlerine bir göz atın:

1. İlk olarak, bağımlılıklar yüklenir ve uygulama başlar.
1. Daha sonra, model içe aktarılır.
1. Ardından, ana rotada index.html işlenir.

`/predict` rotasında, form gönderildiğinde birkaç şey olur:

1. Form değişkenleri toplanır ve bir numpy dizisine dönüştürülür. Daha sonra modele gönderilir ve bir tahmin döndürülür.
2. Görüntülenmesini istediğimiz ülkeler, tahmin edilen ülke kodundan okunabilir metin olarak yeniden işlenir ve bu değer index.html'e şablonda işlenmek üzere geri gönderilir.

Bir modeli bu şekilde, Flask ve picklelanmış bir model ile kullanmak oldukça basittir. En zor şey, modele bir tahmin almak için gönderilmesi gereken verilerin şeklini anlamaktır. Bu tamamen modelin nasıl eğitildiğine bağlıdır. Bu model, bir tahmin almak için üç veri noktası girişi gerektirir.

Profesyonel bir ortamda, modeli eğiten kişiler ile bunu bir web veya mobil uygulamada tüketen kişiler arasında iyi iletişimin ne kadar önemli olduğunu görebilirsiniz. Bizim durumumuzda, bu sadece bir kişi, yani sizsiniz!

---

## 🚀 Zorluk

Bir defterde çalışmak ve modeli Flask uygulamasına aktarmak yerine, modeli doğrudan Flask uygulamasında eğitebilirsiniz! Python kodunuzu defterdeki veriler temizlendikten sonra uygulama içinde bir `train` rotasında modeli eğitmek için dönüştürmeyi deneyin. Bu yöntemi takip etmenin artıları ve eksileri nelerdir?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## İnceleme ve Kendi Kendine Çalışma

ML modellerini tüketmek için bir web uygulaması oluşturmanın birçok yolu vardır. Makine öğrenimini kullanmak için JavaScript veya Python ile bir web uygulaması oluşturmanın yollarını listeleyin. Mimarileri düşünün: Model uygulamada mı kalmalı yoksa bulutta mı barındırılmalı? Eğer bulutta barındırılacaksa, ona nasıl erişirsiniz? Uygulamalı bir ML web çözümü için bir mimari model çizin.

## Ödev

[Farklı bir model deneyin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.