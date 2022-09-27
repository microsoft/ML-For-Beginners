# Regresyon modelleri için Python ve Scikit-learn'e giriş

![Summary of regressions in a sketchnote](../../../sketchnotes/ml-regression.png)

> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Ders öncesi quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [R dili ile bu dersin içeriği!](././solution/R/lesson_1-R.ipynb)

## Giriş

Bu dört derste, regresyon modellerinin nasıl oluşturulacağını keşfedeceksiniz.Bunların ne için olduğunu birazdan tartışacağız. Ancak herhangi bir şey yapmadan önce, süreci başlatmak için doğru araçlara sahip olduğunuzdan emin olun!

Bu derste, şunları öğreneceğiz:

- Bilgisayarınızı yerel makine öğrenimi görevleri için yapılandırma.
- Jupyter notebooks ile çalışmayı.
- Scikit-learn kullanmayı, kurulum da dahil.
- Uygulamalı alıştırma ile doğrusal(lineer) regresyonu keşfedin.

## Kurulum ve Konfigürasyonlar

[![Visual Studio Code ile Python kurulumu](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Setup Python with Visual Studio Code")

> 🎥 Video için yukarıdaki resme tıklayınız: Python'u VS Code içinde kullanma.

1. **Python Kurulumu**. [Python](https://www.python.org/downloads/) kurulumunun bilgisayarınızda yüklü olduğundan emin olun.Python'u birçok veri bilimi ve makine öğrenimi görevi için kullanacaksınız. Çoğu bilgisayar sistemi zaten bir Python kurulumu içerir. Şurada [Python Kodlama Paketleri](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) mevcut, bazı kullanıcılar için kurulumu daha kolay.

   Ancak Python'un bazı kullanımları, yazılımın spesifik bir sürümünü gerektirir, diğerleri ise farklı bir sürüm gerektirir. Bu yüzden, [virtual environment](https://docs.python.org/3/library/venv.html) (sanal ortamlar) ile çalışmak daha kullanışlıdır.

2. **Visual Studio Code kurulumu**. Visual Studio Code'un bilgisayarınıza kurulduğundan emin olun. [Visual Studio Code kurulumu](https://code.visualstudio.com/) bu adımları takip ederek basitçe bir kurulum yapabilirsiniz. Bu kursta Python'ı Visual Studio Code'un içinde kullanacaksınız, bu yüzden nasıl yapılacağını görmek isteyebilirsiniz. Python ile geliştirme için [Visual Studio Code konfigürasyonu](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott).

   > Bu koleksiyon üzerinde çalışarak Python ile rahatlayın. [Modülleri öğren](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)

3. **Scikit-learn kurulumu**, [bu talimatları](https://scikit-learn.org/stable/install.html) takip ediniz. Python 3 kullandığınızdan emin olmanız gerektiğinden, sanal ortam kullanmanız önerilir. Not, bu kütüphaneyi bir M1 Mac'e kuruyorsanız, yukarıda bağlantısı verilen sayfada özel talimatlar var onları takip ediniz.

1. **Jupyter Notebook kurulumu**. [Jupyter package'ı](https://pypi.org/project/jupyter/) kurmanız gerekmektedir.

## Makine öğrenimi geliştirme ortamınız

Python kodunuzu geliştirmek ve makine öğrenimi modelleri oluşturmak için **notebook** kullanacaksınız. Bu dosya türü, veri bilimcileri için yaygın bir araçtır ve bunlar, ".ipynb" son eki veya uzantısıyla tanımlanabilir.

Notebook'lar, geliştiricinin hem kod yazmasına hem de notlar eklemesine ve kodun etrafına deneysel veya araştırma odaklı projeler için oldukça yararlı olan dökümantasyonlar yazmasına izin veren etkileşimli bir ortamdır.
### Alıştırma - notebook'larla çalışmak

Bu klasörde, _notebook.ipynb_ adlı dosyası bulacaksınız.

1.  _notebook.ipynb_ dosyasını Visual Studio Code ile açınız.

   Bir Jupyter serveri Python 3+ ile beraber başlayacaktır. Notebook içinde kod parçalarını çalıştıracak `run` alanını göreceksiniz. Play butonuna benzeyen buton ile kod bloklarını çalıştırabileceksiniz.

1. `md` ikonunu seçip bir markdown ekleyelim ve **# Welcome to your notebook** yazalım.

   Sonra, biraz Python kodu ekleyelim.

1. Kod bloğuna **print('hello notebook')** yazalım.
1. Ok işaretini seçip kodu çalıştıralım.

   Bu ifadeyi çıktı olarak göreceksiniz:

    ```output
    hello notebook
    ```

![VS Code ile notebook açma](../images/notebook.jpg)

Notebook'kunuzu dökümante etmek için kodunuza yorumlar ekleyebilirsiniz.

✅ Bir web geliştiricisinin çalışma ortamının bir veri bilimcisinden ne kadar farklı olduğunu bir an için düşünün.

## Scikit-learn çalışır durumda

Artık Python yerel ortamınızda kurulduğuna göre ve Jupyter notebook ile rahatsanız, hadi Scikit-learn ile de eşit derecede rahat edelim.(`sci` `science`'ın kısaltması yani bilim anlamı taşır). Scikit-learn sağladığı [yaygın API](https://scikit-learn.org/stable/modules/classes.html#api-ref) ile ML görevlerinde sizlere yardım eder.

[websitelerine](https://scikit-learn.org/stable/getting_started.html) göre, "Scikit-learn, denetimli ve denetimsiz öğrenmeyi destekleyen açık kaynaklı bir makine öğrenimi kütüphanesidir. Ayrıca model uydurma, veri ön işleme, model seçimi ve değerlendirmesi gibi diğer birçok şey için yardımcı olacak çeşitli araçlar sağlar."

Bu kursta, 'geleneksel makine öğrenimi' olarak adlandırdığımız görevleri gerçekleştirmek üzere ve makine öğrenimi modelleri oluşturmak için Scikit-learn ve diğer araçları kullanacaksınız. Yakında çıkacak olan 'Yeni Başlayanlar için Yapay Zeka' müfredatımızda daha iyi ele alındığı için sinir ağlarından ve derin öğrenme konularından bilinçli olarak kaçındık.

Scikit-learn, modeller oluşturmayı ve bunları kullanım için  modeli değerlendirmeyi kolaylaştırır. Öncelikle sayısal verileri kullanmaya odaklanır ve öğrenme araçları olarak kullanılmak üzere birkaç hazır veri seti içerir. Ayrıca öğrencilerin denemesi için önceden oluşturulmuş modelleri de içerir. Önceden paketlenmiş verileri yükleme ve bazı temel verilerle birlikte Scikit-learn'de ilk ML modelini kullanma sürecini keşfedelim.

## Alıştırma - ilk Scikit-learn notebook'unuz

> Bu eğitim  Scikit-learn web sitesindeki [lineer regresyon örneğinden](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) ilham alınmıştır.

_notebook.ipynb_ dosyasıda bu dersle ilgili olan, tüm hücreleri 'çöp kutusu' simgesine basarak temizleyin.

Bu bölümde, öğrenme amacıyla Scikit-learn'de yerleşik olarak bulunan diyabetle ilgili küçük bir veri seti ile çalışacaksınız. Diyabet hastaları için bir tedaviyi test etmek istediğinizi hayal edin. Makine Öğrenimi modelleri, değişken kombinasyonlarına göre hangi hastaların tedaviye daha iyi yanıt vereceğini belirlemenize yardımcı olabilir. Çok basit bir regresyon modeli bile görselleştirildiğinde, teorik klinik denemelerinizi düzenlemenize yardımcı olacak değişkenler hakkında bilgi verebilir.

✅ Pek çok regresyon yöntemi vardır ve hangisini seçeceğiniz, aradığınız cevaba bağlıdır. Belirli bir yaştaki bir kişinin olası boyunu tahmin etmek istiyorsanız, **sayısal bir değer** aradığınız için doğrusal regresyon kullanırsınız. Bir yemeğin vegan olarak kabul edilip edilmeyeceğini keşfetmekle ilgileniyorsanız, **kategorik görev** olduğu için lojistik regresyon kullanmalısınız. Daha sonra lojistik regresyon hakkında daha fazla bilgi edineceksiniz. Verilere sorabileceğiniz bazı sorular ve bu yöntemlerden hangisinin daha uygun olacağı hakkında biraz düşünün.

Hadi bu görev ile başlayalım.

### Kütüphaneleri Import etmek

Bu görev için bazı kütüphaneleri import edeceğiz:

- **matplotlib**. Kullanışlı bir [grafiksel bir araç](https://matplotlib.org/) ve bir çizgi grafiği oluşturmak için kullanacağız.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) Python'da nümerik verileri ele almak için kullanışlı bir kütüphane.
- **sklearn**. Bu da [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) kütüphanesi.

Bu görevimizde yardımcı olacak bazı kütüphaneleri import edelim.

1. Aşağıdaki kodu yazarak import edelim:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

    `matplotlib`, `numpy` import ettik ve `datasets`, `linear_model` , `model_selection` 'ı `sklearn` den import ettik. `model_selection` veri setimizi eğitim ve test kümeleri şeklinde bölmemize yardımcı olacak.

### Diyabet veri seti

[Diyabet veri seti](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) 442 tane diyabet ile ilgili örnek içeririr, 10 öznitelik değişkeni,bazıları şunları içerir:

- age: yaşı
- bmi: vücut kitle indeksi
- bp: ortalama kan basıncı
- s1 tc: T-Cells (bir tür beyaz kan hücresi)

✅ Bu veri seti, diyabet hakkında araştırma yapmak için önemli bir özellik değişkeni olarak 'cinsiyet' kavramını içerir. Birçok tıbbi veri kümesi bu tür ikili sınıflandırmayı içerir. Bunun gibi sınıflandırmaların bir popülasyonun belirli bölümlerini tedavilerden nasıl dışlayabileceğini biraz düşünün.

Şimdi, X ve y verilerini yükleyelim.

> 🎓 Unutmayın, bu denetimli öğrenmedir ve bir 'y' hedefine ihtiyaç vardır.

Yeni bir hücrede, load_diabetes()'i çağırarak diyabet veri setini yükleyin. 'return_X_y=True' girişi, X'in bir veri matrisi olacağını ve y'nin regresyon hedefi olacağını bildirir.

1. Verinin, matriksinin şekline ve ilk elementine göz atmak için print kullanalım.

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Yanıt olarak geri aldığınız şey, bir tuple. Yaptığınız şey, tuple'ın ilk iki değerini sırasıyla 'X' ve 'y'ye atamak. [Tuple hakkında  daha fazla bilgi](https://wikipedia.org/wiki/Tuple).

  Bu verinin 10 elemanlı dizilerde(array) şekillendirilmiş 442 öğeye sahip olduğunu görebilirsiniz:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Veriler ve regresyon hedefi arasındaki ilişki hakkında biraz düşünün. Doğrusal regresyon, özellik X ile hedef değişken y arasındaki ilişkileri tahmin eder. Diyabet veri setindeki dökümantasyonda [hedefi](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) bulabildiniz mi? Bu hedef göz önüne alındığında, bu veri kümesi neyi gösteriyor?

2. Ardından, numpy'nin 'newaxis' fonksiyonunu kullanarak yeni bir array düzenleyerek bu veri kümesinin bir bölümünü seçin. Belirlediği bir kalıba göre bu verilerdeki değerler arasında bir çizgi oluşturmak için doğrusal regresyon kullanacağız.
   ```python
   X = X[:, np.newaxis, 2]
   ```

   ✅ İstediğiniz zaman, verinin şeklini kontrol etmek için yazdırın.

3. Artık çizilmeye hazır verileriniz olduğuna göre, bir makinenin bu veri kümesindeki sayılar arasında mantıksal bir ayrım belirlemeye yardımcı olup olamayacağını görebilirsiniz. Bunun için, hem verileri (X) hem de hedefi (y), test ve eğitim kümelerine ayırmanız gerekir. Scikit-learn'in bunu yapmanın basit bir yolu var; test verilerinizi belirli bir noktada bölebilirsiniz.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Artık modelinizi eğitmeye hazırsınız! Doğrusal regresyon modelini yükleyin ve onu "model.fit()" kullanarak X ve y eğitim kümelerinde eğitin:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` TensorFlow gibi birçok ML kütüphanesinde görebileceğiniz bir fonksiyondur.

5. Ardından, "predict()" fonksiyonunu kullanarak test verilerini üzerinde bir tahmin oluşturun. Bu, veri grupları arasındaki çizgiyi çizmek için kullanılacaktır.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Şimdi verileri bir plotta gösterme zamanı. Matplotlib bu görev için çok kullanışlı bir araçtır. Tüm X ve y test verilerinin bir dağılım grafiğini(scatterplot) oluşturun ve modelin veri grupları arasında en uygun yere bir çizgi çizmek için tahminleri kullanın.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![a scatterplot showing datapoints around diabetes](././images/scatterplot.png)

   ✅ Burada neler olduğunu biraz düşünün. Birçok küçük veri noktalarından düz bir çizgi geçiyor ama tam olarak ne yapıyor? Yeni, görünmeyen bir veri noktasının grafiğin y eksenine göre nerede olması gerektiğini tahmin etmek için bu çizgiyi nasıl kullanabilmeniz gerektiğini görebiliyor musunuz? Bu modelin pratik kullanımını kelimelere dökmeye çalışın.

Tebrikler, ilk doğrusal regresyon modelinizi oluşturdunuz, onunla bir tahmin oluşturdunuz ve bunu bir grafikte görüntülediniz!

---
## 🚀Challenge

Bu veri kümesinden farklı bir değişken çizin. İpucu: bu satırı düzenleyin: `X = X[:, np.newaxis, 2]`. Bu veri setinin hedefi göz önüne alındığında, diyabetin bir hastalık olarak ilerlemesi hakkında neler keşfedebilirsiniz?
## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## İnceleme ve Bireysel Çalışma

Bu eğitimde, tek değişkenli veya çoklu doğrusal regresyon yerine basit doğrusal regresyonla çalıştınızBu yöntemler arasındaki farklar hakkında biraz bilgi edinin veya şuna bir göz atın: [bu videoya](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Regresyon kavramı hakkında daha fazla bilgi edinin ve bu teknikle ne tür soruların yanıtlanabileceğini düşünün. Anlayışınızı derinleştirmek için bu [eğitime](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) göz atabilirsiniz.

## Assignment

[A different dataset](assignment.md)
