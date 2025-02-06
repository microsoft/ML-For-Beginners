# Bir ML Modelini Kullanmak için Web Uygulaması Oluşturun

Bu derste, _son yüzyıldaki UFO gözlemleri_ gibi dünyadışı bir veri seti üzerinde bir ML modeli eğiteceksiniz. Bu veriler NUFORC'un veritabanından alınmıştır.

Öğreneceğiniz konular:

- Eğitilmiş bir modeli nasıl 'pickle' yapacağınız
- Bu modeli bir Flask uygulamasında nasıl kullanacağınız

Verileri temizlemek ve modelimizi eğitmek için defterleri kullanmaya devam edeceğiz, ancak süreci bir adım öteye taşıyarak, modelinizi bir web uygulamasında kullanmayı keşfedebilirsiniz.

Bunu yapmak için Flask kullanarak bir web uygulaması oluşturmanız gerekecek.

## [Ders Öncesi Testi](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Bir Uygulama Oluşturmak

Makine öğrenimi modellerini tüketen web uygulamaları oluşturmanın birkaç yolu vardır. Web mimariniz, modelinizin nasıl eğitildiğini etkileyebilir. Bir işletmede çalıştığınızı ve veri bilimi grubunun bir model eğittiğini ve bu modeli bir uygulamada kullanmanızı istediğini hayal edin.

### Dikkat Edilmesi Gerekenler

Sormanız gereken birçok soru var:

- **Bu bir web uygulaması mı yoksa mobil uygulama mı?** Bir mobil uygulama oluşturuyorsanız veya modeli bir IoT bağlamında kullanmanız gerekiyorsa, [TensorFlow Lite](https://www.tensorflow.org/lite/) kullanarak modeli bir Android veya iOS uygulamasında kullanabilirsiniz.
- **Model nerede bulunacak?** Bulutta mı yoksa yerel olarak mı?
- **Çevrimdışı destek.** Uygulamanın çevrimdışı çalışması gerekiyor mu?
- **Modeli eğitmek için hangi teknoloji kullanıldı?** Seçilen teknoloji, kullanmanız gereken araçları etkileyebilir.
    - **TensorFlow Kullanmak.** Örneğin, TensorFlow kullanarak bir model eğitiyorsanız, bu ekosistem, [TensorFlow.js](https://www.tensorflow.org/js/) kullanarak bir web uygulamasında kullanmak üzere bir TensorFlow modelini dönüştürme yeteneği sağlar.
    - **PyTorch Kullanmak.** [PyTorch](https://pytorch.org/) gibi bir kütüphane kullanarak bir model oluşturuyorsanız, modeli JavaScript web uygulamalarında kullanmak üzere [Onnx Runtime](https://www.onnxruntime.ai/) kullanarak [ONNX](https://onnx.ai/) (Open Neural Network Exchange) formatında dışa aktarma seçeneğiniz vardır. Bu seçenek, gelecekteki bir derste Scikit-learn ile eğitilmiş bir model için incelenecektir.
    - **Lobe.ai veya Azure Custom Vision Kullanmak.** [Lobe.ai](https://lobe.ai/) veya [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) gibi bir ML SaaS (Hizmet Olarak Yazılım) sistemi kullanarak bir model eğitiyorsanız, bu tür yazılımlar, modeli birçok platform için dışa aktarma yolları sağlar, bu da çevrimiçi uygulamanız tarafından bulutta sorgulanacak özel bir API oluşturmayı içerir.

Ayrıca, modelin kendisini bir web tarayıcısında eğitebilecek bir Flask web uygulaması oluşturma fırsatınız da var. Bu, bir JavaScript bağlamında TensorFlow.js kullanılarak da yapılabilir.

Bizim amacımız için, Python tabanlı defterlerle çalıştığımızdan, eğitilmiş bir modeli bu tür bir defterden Python ile oluşturulmuş bir web uygulaması tarafından okunabilir bir formata nasıl dışa aktaracağınızı inceleyelim.

## Araç

Bu görev için iki araca ihtiyacınız var: Flask ve Pickle, her ikisi de Python üzerinde çalışır.

✅ [Flask](https://palletsprojects.com/p/flask/) nedir? Yaratıcıları tarafından bir 'mikro-çerçeve' olarak tanımlanan Flask, Python kullanarak web çerçevelerinin temel özelliklerini ve web sayfaları oluşturmak için bir şablon motoru sağlar. Flask ile inşa etmeyi pratik yapmak için [bu Öğrenme modülüne](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) göz atın.

✅ [Pickle](https://docs.python.org/3/library/pickle.html) nedir? Pickle 🥒, bir Python nesne yapısını serileştiren ve serileştiren bir Python modülüdür. Bir modeli 'pickle' yaptığınızda, yapısını webde kullanmak üzere serileştirir veya düzleştirirsiniz. Dikkatli olun: pickle doğası gereği güvenli değildir, bu yüzden bir dosyayı 'un-pickle' yapmanız istendiğinde dikkatli olun. Bir pickled dosyası `.pkl` uzantısına sahiptir.

## Alıştırma - verilerinizi temizleyin

Bu derste, [NUFORC](https://nuforc.org) (Ulusal UFO Raporlama Merkezi) tarafından toplanan 80.000 UFO gözleminden veri kullanacaksınız. Bu veriler, UFO gözlemlerine dair ilginç açıklamalar içerir, örneğin:

- **Uzun örnek açıklama.** "Bir adam geceleyin çimenli bir alana parlayan bir ışık huzmesinden çıkar ve Texas Instruments otoparkına doğru koşar".
- **Kısa örnek açıklama.** "ışıklar bizi kovaladı".

[ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) elektronik tablosu, gözlemin `city`, `state` ve `country` nerede gerçekleştiği, nesnenin `shape` ve `latitude` ve `longitude` ile ilgili sütunları içerir.

Bu derste yer alan boş [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) dosyasında:

1. Önceki derslerde yaptığınız gibi `pandas`, `matplotlib` ve `numpy` içe aktarın ve ufos elektronik tablosunu içe aktarın. Örnek bir veri setine göz atabilirsiniz:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Ufolar verilerini yeni başlıklarla küçük bir dataframe'e dönüştürün. `Country` alanındaki benzersiz değerleri kontrol edin.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Şimdi, ele almamız gereken veri miktarını azaltmak için herhangi bir boş değeri atabilir ve sadece 1-60 saniye arasındaki gözlemleri içe aktarabilirsiniz:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Metin değerlerini ülkelere dönüştürmek için Scikit-learn'ün `LabelEncoder` kütüphanesini içe aktarın:

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

## Alıştırma - modelinizi oluşturun

Şimdi verileri eğitim ve test gruplarına ayırarak bir model eğitmeye hazır olabilirsiniz.

1. Eğitmek istediğiniz üç özelliği X vektörü olarak seçin ve y vektörü `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` olacak ve bir ülke kimliği döndürecek.

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

Doğruluk fena değil **(yaklaşık %95)**, şaşırtıcı değil, çünkü `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, ancak ham verilerden temizlediğiniz, dışa aktardığınız ve ardından bu modeli bir web uygulamasında kullandığınız bir modeli eğitmeye çalışmak iyi bir egzersizdir.

## Alıştırma - modelinizi 'pickle' yapın

Şimdi, modelinizi _pickle_ yapma zamanı! Bunu birkaç satır kodla yapabilirsiniz. Bir kez _pickled_ olduktan sonra, pickled modelinizi yükleyin ve saniye, enlem ve boylam değerlerini içeren bir örnek veri dizisine karşı test edin,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

Model **'3'** değerini döndürüyor, bu da Birleşik Krallık için ülke kodu. Harika! 👽

## Alıştırma - bir Flask uygulaması oluşturun

Şimdi modelinizi çağıracak ve benzer sonuçlar döndürecek, ancak daha görsel olarak hoş bir şekilde, bir Flask uygulaması oluşturabilirsiniz.

1. _notebook.ipynb_ dosyasının yanında **web-app** adlı bir klasör oluşturun ve _ufo-model.pkl_ dosyanızın bulunduğu yer.

1. Bu klasörde üç klasör daha oluşturun: **static**, içinde bir **css** klasörü bulunan ve **templates**. Şimdi aşağıdaki dosya ve dizinlere sahip olmalısınız:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Bitmiş uygulamanın bir görünümünü görmek için çözüm klasörüne başvurun

1. _web-app_ klasöründe oluşturulacak ilk dosya **requirements.txt** dosyasıdır. Bir JavaScript uygulamasındaki _package.json_ gibi, bu dosya uygulama tarafından gerekli bağımlılıkları listeler. **requirements.txt** dosyasına şu satırları ekleyin:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Şimdi, _web-app_ klasörüne giderek bu dosyayı çalıştırın:

    ```bash
    cd web-app
    ```

1. Terminalinizde `pip install` yazarak _requirements.txt_ dosyasında listelenen kütüphaneleri yükleyin:

    ```bash
    pip install -r requirements.txt
    ```

1. Şimdi, uygulamayı bitirmek için üç dosya daha oluşturmaya hazırsınız:

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

    Bu dosyadaki şablonlamaya bir göz atın. Uygulama tarafından sağlanacak değişkenler etrafındaki 'bıyık' sözdizimine dikkat edin, örneğin tahmin metni: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` ekleyin:

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

    > 💡 İpucu: [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Bu yöntemi takip etmenin artıları ve eksileri nelerdir?

## [Ders Sonrası Testi](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Gözden Geçirme ve Kendi Kendine Çalışma

ML modellerini tüketen bir web uygulaması oluşturmanın birçok yolu vardır. Makine öğrenimini kullanmak için JavaScript veya Python kullanarak bir web uygulaması oluşturmanın yollarını listeleyin. Mimariyi göz önünde bulundurun: model uygulamada mı kalmalı yoksa bulutta mı yaşamalı? Eğer ikinci seçenekse, ona nasıl erişirsiniz? Uygulamalı bir ML web çözümü için bir mimari model çizin.

## Ödev

[Farklı bir model deneyin](assignment.md)

**Feragatname**:
Bu belge, makine tabanlı AI çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belgenin kendi dilindeki hali yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi tavsiye edilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.