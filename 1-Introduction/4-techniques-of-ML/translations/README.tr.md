# Machine Learning Teknikleri

Makine öğrenimi (ML) modellerinin oluşturulması, kullanılması ve bakımı süreci, birçok diğer geliştirme iş akışından oldukça farklıdır. Bu derste, süreci açıklığa kavuşturacak ve bilmeniz gereken temel teknikleri açıklayacağız. Şunları yapacaksınız:

- Makine öğreniminin temelini oluşturan süreçleri yüksek düzeyde anlayın.
- 'Modeller', 'tahminler' ve 'eğitim verileri' gibi temel kavramları keşfedin.

## [Konferans Öncesi Anket](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7?loc=es)

## Giriş

Yüksek bir düzeyde, makine öğrenimi (ML) süreçlerini oluşturma sanatı birkaç adımdan oluşur:

1. **Soruyu Belirlemek**. Çoğu ML süreci, basit bir koşullu program veya kurallara dayalı bir motorla yanıtlanamayan bir soruyu sormakla başlar. Bu sorular genellikle veri toplamaya dayalı tahminlerle ilgilidir.
2. **Veri Toplayın ve Hazırlayın**. Sorunuza yanıt verebilmek için verilere ihtiyacınız vardır. Verilerinizin kalitesi ve bazen miktarı, ilk sorunuza ne kadar iyi yanıt verebileceğinizi belirleyecektir. Veri görselleştirme, bu aşamanın önemli bir yönüdür. Bu aşama ayrıca verilerinizi bir eğitim ve test grubuna bölerek bir model oluşturmayı da içerir.
3. **Bir Eğitim Yöntemi Seçin**. Sorunuza ve verilerinizin doğasına bağlı olarak, bir modeli verilerinizi en iyi şekilde yansıtacak ve onlara karşı doğru tahminler yapacak şekilde nasıl eğitmek istediğinizi seçmelisiniz. Bu, ML sürecinizin özel bilgi ve genellikle önemli ölçüde deneme-yanılma gerektiren kısmıdır.
4. **Modeli Eğitin**. Eğitim verilerinizi kullanarak, bir modelin verilerdeki desenleri tanımasını sağlamak için çeşitli algoritmalar kullanırsınız. Model, daha iyi bir model oluşturmak için verilerin bazı bölümlerini diğerlerine göre önceliklendirmek amacıyla ayarlanabilir içsel ağırlıklardan yararlanabilir.
5. **Modeli Değerlendirin**. Toplanan verilerinizden daha önce hiç görülmemiş verileri (test verileriniz) kullanarak modelin performansını görürsünüz.
6. **Parametre Ayarı**. Modelinizin performansına bağlı olarak, modeli eğitmek için kullanılan algoritmaların davranışını kontrol eden farklı parametreler veya değişkenler kullanarak süreci yeniden yapabilirsiniz.
7. **Tahmin Yapın**. Modelinizin doğruluğunu test etmek için yeni girdiler kullanın.

## Hangi Soruları Sormalı

Bilgisayarlar, verilerdeki gizli desenleri keşfetme konusunda oldukça yeteneklidir. Bu yetenek, bir veri bilimcisinin yalnızca koşullu kurallara dayalı bir motor oluşturarak kolayca cevaplayamayacağı bir alandaki soruları yanıtlamak için faydalıdır.

Örneğin, aktüeryal bir görevde, bir veri bilimcisi sigara içenlerin ve içmeyenlerin ölüm oranları hakkında elle oluşturulan kurallar geliştirebilir. Ancak, denkleme birçok başka değişken dahil edildiğinde, bir ML modeli sağlık geçmişine dayalı olarak gelecekteki ölüm oranlarını tahmin etmede daha verimli olabilir. Daha neşeli bir örnek, belirli bir yerde Nisan ayı için hava tahminleri yapmak olabilir.

✅ Bu [sunum](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf), ML'in meteorolojik analizlerde kullanımına dair tarihsel bir perspektif sunmaktadır.

## Model Oluşturmadan Önceki Görevler

Modelinizi oluşturmaya başlamadan önce tamamlamanız gereken birkaç görev vardır. Sorunuzu incelemek ve modelinizin tahminlerine dayalı bir hipotez oluşturmak için birkaç öğeyi tanımlamanız ve ayarlamanız gerekir.

### Veriler

Sorunuza bir miktar kesinlikle yanıt verebilmek için doğru türden yeterli miktarda veriye ihtiyacınız vardır.

Bu noktada yapmanız gereken iki şey vardır:

- **Veri Toplamak**. Verilerinizi dikkatlice toplayın. Bu verilerin kaynağını, sahip olabileceği herhangi bir örtük önyargıyı göz önünde bulundurun ve kaynağını belgelerle belirtin.
- **Verileri Hazırlamak**. Verileri hazırlama sürecinde birkaç adım vardır. Verilerinizi toplamanız ve çeşitli kaynaklardan geliyorsa normalleştirmeniz gerekebilir. Verilerin kalitesini artırmak ve miktarını genişletmek için bir dizi yöntem kullanılabilir.

✅ Verilerinizi topladıktan ve işledikten sonra, biçimlerinin sorunuza yanıt verip veremeyeceğini kontrol edin. Bazen verileriniz görev için uygun olmayabilir!

### Özellikler ve Hedef

Bir özellik, verilerin ölçülebilir bir özelliğidir. Bir hedef ise tahmin etmeye çalıştığınız şeydir. Bu genellikle `y` olarak ifade edilir ve sorunuzun cevabını temsil eder.

### Özellik Seçimi

🎓 **Özellik Seçimi ve Çıkarımı** Doğru değişkenleri nasıl seçeceğinizi bilmek, model performansı için önemlidir. Çıkarım ve seçim farklıdır: "Özellik çıkarımı, orijinal özelliklerden yeni özellikler oluşturur, seçim ise bir alt küme döndürür."

### Verilerinizi Görselleştirin

Veri görselleştirme, ilişkileri keşfetmenize yardımcı olur ve dengesizlikleri tespit etmenize olanak tanır. Matplotlib veya Seaborn gibi araçlar bu konuda faydalıdır.

### Veri Setini Bölmek

Model eğitimi öncesinde, veri setinizi iki veya daha fazla parçaya bölün:

- **Eğitim Verileri**. Modeli eğitmek için kullanılır.
- **Test Verileri**. Modelin performansını doğrulamak için kullanılır.
- **Doğrulama Verileri**. Model hiperparametrelerini ayarlamak için kullanılır.

## Model Oluşturma

Eğitim verilerinizle, desenleri öğrenen ve bunları doğrulayan bir model oluşturun.

### Bir Eğitim Yöntemi Seçin

Sorunuza ve verilerinize bağlı olarak uygun bir yöntem seçin. Scikit-learn belgeleri bu konuda harika bir kaynaktır.

### Modeli Eğitin

`model.fit` gibi yöntemlerle modeli verilerinize uygularsınız.

### Modeli Değerlendirin

Modeli test verileriyle değerlendirerek doğruluğunu ve yanılgılarını inceleyin.

🎓 **Aşırı Uyum ve Yetersiz Uyum**

Aşırı uyum (overfitting) ve yetersiz uyum (underfitting) model kalitesini düşüren yaygın sorunlardır.

![Model Aşırı Uyum Görseli](images/overfitting.png)
> [Jen Looper](https://twitter.com/jenlooper) tarafından hazırlanmıştır.

## Parametre Ayarı

Hiperparametreleri ayarlayarak modelin performansını geliştirin.

## Tahmin

Modeli, yeni verilerle test ederek doğruluğunu ölçün. Bu, ML süreçlerinizin nihai adımıdır ve modelin uygulanabilirliğini belirler.

---

## 🚀 Zorluk

Makine öğrenimi süreçlerini yansıtan bir akış diyagramı çizin. Kendinizi bu süreçte nerede görüyorsunuz? Hangi adımda zorluk yaşayacağınızı düşünüyorsunuz?

## [Konferans Sonrası Anket](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8?loc=es)

## İnceleme ve Öz-Çalışma

Veri bilimcilerin günlük işlerini analiz ettiği çevrimiçi röportajları araştırın. İşte [bir tanesi](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Görev

[Bir veri bilimciyle röportaj yapın](assignment.tr.md)
