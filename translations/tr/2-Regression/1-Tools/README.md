<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-06T07:46:52+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "tr"
}
-->
# Python ve Scikit-learn ile Regresyon Modellerine Başlangıç

![Bir sketchnote'ta regresyonların özeti](../../../../sketchnotes/ml-regression.png)

> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bu ders R dilinde de mevcut!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Giriş

Bu dört derste, regresyon modelleri oluşturmayı öğreneceksiniz. Bunların ne işe yaradığını birazdan tartışacağız. Ancak başlamadan önce, sürece başlamak için doğru araçlara sahip olduğunuzdan emin olun!

Bu derste şunları öğreneceksiniz:

- Bilgisayarınızı yerel makine öğrenimi görevleri için yapılandırmayı.
- Jupyter defterleriyle çalışmayı.
- Scikit-learn'ü kullanmayı ve kurulumunu.
- Uygulamalı bir egzersizle doğrusal regresyonu keşfetmeyi.

## Kurulumlar ve Yapılandırmalar

[![Başlangıç için Araçlarınızı Hazırlayın - Makine Öğrenimi Modelleri Oluşturma](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "Başlangıç için Araçlarınızı Hazırlayın - Makine Öğrenimi Modelleri Oluşturma")

> 🎥 Yukarıdaki görsele tıklayarak bilgisayarınızı ML için yapılandırma sürecini anlatan kısa bir videoyu izleyebilirsiniz.

1. **Python'u Kurun**. Bilgisayarınızda [Python](https://www.python.org/downloads/) yüklü olduğundan emin olun. Python, birçok veri bilimi ve makine öğrenimi görevi için kullanılacaktır. Çoğu bilgisayar sisteminde zaten bir Python kurulumu bulunur. Bazı kullanıcılar için kurulumu kolaylaştıran [Python Kodlama Paketleri](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) de mevcuttur.

   Ancak, Python'un bazı kullanımları bir yazılım sürümünü gerektirirken, diğerleri farklı bir sürüm gerektirir. Bu nedenle, bir [sanallaştırılmış ortam](https://docs.python.org/3/library/venv.html) içinde çalışmak faydalı olabilir.

2. **Visual Studio Code'u Kurun**. Bilgisayarınızda Visual Studio Code'un yüklü olduğundan emin olun. [Visual Studio Code'u kurma](https://code.visualstudio.com/) talimatlarını izleyerek temel kurulumu gerçekleştirin. Bu kursta Python'u Visual Studio Code'da kullanacağınız için, Python geliştirme için [Visual Studio Code'u yapılandırma](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) konusunda bilgi edinmek isteyebilirsiniz.

   > Python ile rahat çalışmak için bu [öğrenme modülleri](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott) koleksiyonunu inceleyin.
   >
   > [![Visual Studio Code ile Python Kurulumu](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Visual Studio Code ile Python Kurulumu")
   >
   > 🎥 Yukarıdaki görsele tıklayarak Visual Studio Code'da Python kullanımıyla ilgili bir video izleyebilirsiniz.

3. **Scikit-learn'ü Kurun**, [bu talimatları](https://scikit-learn.org/stable/install.html) izleyerek. Python 3 kullanmanız gerektiğinden, bir sanallaştırılmış ortam kullanmanız önerilir. Eğer bu kütüphaneyi bir M1 Mac'te kuruyorsanız, yukarıdaki bağlantıda özel talimatlar bulunmaktadır.

4. **Jupyter Notebook'u Kurun**. [Jupyter paketini kurmanız](https://pypi.org/project/jupyter/) gerekecek.

## Makine Öğrenimi Geliştirme Ortamınız

Python kodunuzu geliştirmek ve makine öğrenimi modelleri oluşturmak için **defterler** kullanacaksınız. Bu tür dosyalar veri bilimciler için yaygın bir araçtır ve `.ipynb` uzantısıyla tanınabilirler.

Defterler, geliştiricinin hem kod yazmasına hem de kodun etrafında notlar ve belgeler oluşturmasına olanak tanıyan etkileşimli bir ortamdır. Bu, deneysel veya araştırma odaklı projeler için oldukça faydalıdır.

[![Başlangıç için Jupyter Defterlerini Ayarlayın - Regresyon Modelleri Oluşturma](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "Başlangıç için Jupyter Defterlerini Ayarlayın - Regresyon Modelleri Oluşturma")

> 🎥 Yukarıdaki görsele tıklayarak bu egzersizi anlatan kısa bir video izleyebilirsiniz.

### Egzersiz - Bir Defterle Çalışma

Bu klasörde _notebook.ipynb_ dosyasını bulacaksınız.

1. _notebook.ipynb_ dosyasını Visual Studio Code'da açın.

   Bir Jupyter sunucusu Python 3+ ile başlatılacaktır. Defterde `çalıştırılabilir` kod parçaları bulacaksınız. Bir kod bloğunu çalıştırmak için, oynat düğmesine benzeyen simgeyi seçebilirsiniz.

2. `md` simgesini seçin ve biraz markdown ekleyin, ardından şu metni yazın: **# Defterinize Hoş Geldiniz**.

   Şimdi biraz Python kodu ekleyin.

3. Kod bloğuna **print('hello notebook')** yazın.
4. Kodu çalıştırmak için oku seçin.

   Yazdırılan ifadeyi görmelisiniz:

    ```output
    hello notebook
    ```

![Bir defter açıkken VS Code](../../../../2-Regression/1-Tools/images/notebook.jpg)

Kodunuzu yorumlarla birleştirerek defteri kendi kendine belgeleyebilirsiniz.

✅ Bir web geliştiricisinin çalışma ortamı ile bir veri bilimcinin çalışma ortamı arasındaki farkları bir dakika düşünün.

## Scikit-learn ile Çalışmaya Başlama

Artık Python yerel ortamınızda kurulu olduğuna ve Jupyter defterleriyle rahat olduğunuza göre, Scikit-learn ile de aynı rahatlığa ulaşalım (bunu `sci` olarak, `science` kelimesindeki gibi telaffuz edin). Scikit-learn, ML görevlerini gerçekleştirmenize yardımcı olacak [kapsamlı bir API](https://scikit-learn.org/stable/modules/classes.html#api-ref) sağlar.

Web sitelerine göre ([kaynak](https://scikit-learn.org/stable/getting_started.html)), "Scikit-learn, denetimli ve denetimsiz öğrenmeyi destekleyen açık kaynaklı bir makine öğrenimi kütüphanesidir. Ayrıca model uyumu, veri ön işleme, model seçimi ve değerlendirme gibi çeşitli araçlar sunar."

Bu kursta, Scikit-learn ve diğer araçları kullanarak 'geleneksel makine öğrenimi' görevlerini gerçekleştirecek modeller oluşturacaksınız. Sinir ağları ve derin öğrenmeyi özellikle dahil etmedik, çünkü bunlar yakında çıkacak olan 'Başlangıç için Yapay Zeka' müfredatımızda daha iyi ele alınacaktır.

Scikit-learn, modeller oluşturmayı ve bunları değerlendirmeyi kolaylaştırır. Öncelikli olarak sayısal verilerle çalışır ve öğrenme araçları olarak kullanılabilecek birkaç hazır veri seti içerir. Ayrıca, öğrencilerin denemesi için önceden oluşturulmuş modeller de içerir. Şimdi, önceden paketlenmiş verileri yükleme ve yerleşik bir tahminci kullanarak ilk ML modelimizi oluşturma sürecini keşfedelim.

## Egzersiz - İlk Scikit-learn Defteriniz

> Bu eğitim, Scikit-learn'ün web sitesindeki [doğrusal regresyon örneğinden](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) ilham alınarak hazırlanmıştır.

[![Başlangıç için Python'da İlk Doğrusal Regresyon Projeniz](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "Başlangıç için Python'da İlk Doğrusal Regresyon Projeniz")

> 🎥 Yukarıdaki görsele tıklayarak bu egzersizi anlatan kısa bir video izleyebilirsiniz.

Bu derse ait _notebook.ipynb_ dosyasında, tüm hücreleri 'çöp kutusu' simgesine basarak temizleyin.

Bu bölümde, öğrenme amaçlı Scikit-learn'e dahil edilmiş diyabetle ilgili küçük bir veri setiyle çalışacaksınız. Diyabet hastaları için bir tedaviyi test etmek istediğinizi hayal edin. Makine öğrenimi modelleri, değişkenlerin kombinasyonlarına bağlı olarak hangi hastaların tedaviye daha iyi yanıt vereceğini belirlemenize yardımcı olabilir. Görselleştirildiğinde, çok temel bir regresyon modeli bile teorik klinik denemelerinizi organize etmenize yardımcı olacak değişkenler hakkında bilgi gösterebilir.

✅ Birçok regresyon yöntemi vardır ve hangisini seçeceğiniz, aradığınız cevaba bağlıdır. Örneğin, belirli bir yaşta bir kişinin muhtemel boyunu tahmin etmek istiyorsanız, **sayısal bir değer** aradığınız için doğrusal regresyon kullanırsınız. Eğer bir mutfağın vegan olup olmadığını belirlemek istiyorsanız, bir **kategori ataması** arıyorsunuz demektir ve lojistik regresyon kullanırsınız. Lojistik regresyonu daha sonra öğreneceksiniz. Verilerden sorabileceğiniz bazı soruları ve bu yöntemlerden hangisinin daha uygun olacağını düşünün.

Hadi bu göreve başlayalım.

### Kütüphaneleri İçe Aktarma

Bu görev için bazı kütüphaneleri içe aktaracağız:

- **matplotlib**. [Grafik oluşturma aracı](https://matplotlib.org/) ve bir çizgi grafiği oluşturmak için kullanacağız.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html), Python'da sayısal verilerle çalışmak için kullanışlı bir kütüphanedir.
- **sklearn**. Bu, [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) kütüphanesidir.

Görevlerinize yardımcı olması için bazı kütüphaneleri içe aktarın.

1. Aşağıdaki kodu yazarak içe aktarmaları ekleyin:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Yukarıda `matplotlib`, `numpy` ve `sklearn`'den `datasets`, `linear_model` ve `model_selection` içe aktarılıyor. `model_selection`, verileri eğitim ve test setlerine ayırmak için kullanılır.

### Diyabet Veri Seti

Yerleşik [diyabet veri seti](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), diyabetle ilgili 442 veri örneği içerir ve 10 özellik değişkeni içerir. Bunlardan bazıları şunlardır:

- age: yaş (yıl olarak)
- bmi: vücut kitle indeksi
- bp: ortalama kan basıncı
- s1 tc: T-Hücreleri (bir tür beyaz kan hücresi)

✅ Bu veri seti, diyabetle ilgili araştırmalarda önemli bir özellik değişkeni olarak 'cinsiyet' kavramını içerir. Birçok tıbbi veri seti, bu tür ikili sınıflandırmaları içerir. Bu tür sınıflandırmaların, bir nüfusun belirli bölümlerini tedavilerden nasıl dışlayabileceğini biraz düşünün.

Şimdi, X ve y verilerini yükleyin.

> 🎓 Unutmayın, bu denetimli bir öğrenmedir ve adlandırılmış bir 'y' hedefi gereklidir.

Yeni bir kod hücresinde, `load_diabetes()` fonksiyonunu çağırarak diyabet veri setini yükleyin. `return_X_y=True` girdisi, `X`'in bir veri matrisi ve `y`'nin regresyon hedefi olacağını belirtir.

1. Veri matrisinin şeklini ve ilk elemanını göstermek için bazı print komutları ekleyin:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Aldığınız yanıt bir tuple'dır. Tuple'ın ilk iki değerini sırasıyla `X` ve `y`'ye atıyorsunuz. Daha fazla bilgi için [tuple'lar hakkında bilgi edinin](https://wikipedia.org/wiki/Tuple).

    Bu verilerin, 10 elemanlı diziler halinde şekillendirilmiş 442 öğeye sahip olduğunu görebilirsiniz:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Veriler ile regresyon hedefi arasındaki ilişkiyi biraz düşünün. Doğrusal regresyon, özellik X ile hedef değişken y arasındaki ilişkileri tahmin eder. Diyabet veri seti için [hedefi](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) belgelerde bulabilir misiniz? Bu veri seti neyi gösteriyor?

2. Daha sonra, bu veri setinin bir kısmını seçerek çizim yapın. Veri setinin 3. sütununu seçmek için `:` operatörünü kullanarak tüm satırları seçin ve ardından 3. sütunu (2. indeks) seçin. Verileri, çizim için gerekli olan 2D bir diziye dönüştürmek için `reshape(n_rows, n_columns)` kullanabilirsiniz. Parametrelerden biri -1 ise, ilgili boyut otomatik olarak hesaplanır.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ İstediğiniz zaman, verilerin şeklini kontrol etmek için yazdırabilirsiniz.

3. Artık çizim için hazır olan verilere sahipsiniz. Şimdi, bu veri setindeki sayılar arasında mantıklı bir ayrım yapıp yapamayacağını görmek için bir makine kullanabilirsiniz. Bunu yapmak için, hem verileri (X) hem de hedefi (y) test ve eğitim setlerine ayırmanız gerekir. Scikit-learn, bunu yapmak için basit bir yol sunar; test verilerinizi belirli bir noktada bölebilirsiniz.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Şimdi modelinizi eğitmeye hazırsınız! Doğrusal regresyon modelini yükleyin ve `model.fit()` kullanarak X ve y eğitim setlerinizle eğitin:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` TensorFlow gibi birçok ML kütüphanesinde göreceğiniz bir fonksiyondur.

5. Daha sonra, test verilerini kullanarak bir tahmin oluşturun. Bu, veri grupları arasındaki çizgiyi çizmek için kullanılacaktır.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Şimdi verileri bir grafikte gösterme zamanı. Matplotlib bu görev için çok kullanışlı bir araçtır. Tüm X ve y test verilerinin bir dağılım grafiğini oluşturun ve modelin veri grupları arasındaki en uygun yere bir çizgi çizmek için tahmini kullanın.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![diyabetle ilgili veri noktalarını gösteren bir dağılım grafiği](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Burada neler olduğunu biraz düşünün. Bir düz çizgi, birçok küçük veri noktasının arasından geçiyor, ancak tam olarak ne yapıyor? Bu çizgiyi, yeni ve daha önce görülmemiş bir veri noktasının, grafiğin y ekseniyle olan ilişkisini tahmin etmek için nasıl kullanabileceğinizi görebiliyor musunuz? Bu modelin pratik kullanımını kelimelere dökmeye çalışın.

Tebrikler, ilk doğrusal regresyon modelinizi oluşturdunuz, bununla bir tahmin yaptınız ve bunu bir grafikte gösterdiniz!

---
## 🚀Meydan Okuma

Bu veri kümesinden farklı bir değişkeni görselleştirin. İpucu: Şu satırı düzenleyin: `X = X[:,2]`. Bu veri kümesinin hedefi göz önüne alındığında, diyabetin bir hastalık olarak ilerleyişi hakkında neler keşfedebilirsiniz?
## [Ders sonrası sınav](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu derste, basit doğrusal regresyon ile çalıştınız, tek değişkenli veya çok değişkenli doğrusal regresyon ile değil. Bu yöntemler arasındaki farklar hakkında biraz okuyun veya [bu videoya](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef) göz atın.

Regresyon kavramı hakkında daha fazla bilgi edinin ve bu teknikle hangi tür soruların yanıtlanabileceğini düşünün. Anlayışınızı derinleştirmek için [bu eğitimi](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) alın.

## Ödev

[Farklı bir veri kümesi](assignment.md)

---

**Feragatname**:  
Bu belge, [Co-op Translator](https://github.com/Azure/co-op-translator) adlı yapay zeka çeviri hizmeti kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlama veya yanlış yorumlamalardan sorumlu değiliz.