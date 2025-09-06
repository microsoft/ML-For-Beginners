<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-06T07:55:34+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "tr"
}
-->
# Makine Öğrenimi Teknikleri

Makine öğrenimi modellerini oluşturma, kullanma ve bu modellerin kullandığı verileri yönetme süreci, birçok diğer geliştirme iş akışından oldukça farklıdır. Bu derste, süreci açıklığa kavuşturacak ve bilmeniz gereken temel teknikleri özetleyeceğiz. Şunları yapacaksınız:

- Makine öğreniminin temel süreçlerini yüksek seviyede anlayacaksınız.
- 'Modeller', 'tahminler' ve 'eğitim verisi' gibi temel kavramları keşfedeceksiniz.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

[![Yeni Başlayanlar için ML - Makine Öğrenimi Teknikleri](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "Yeni Başlayanlar için ML - Makine Öğrenimi Teknikleri")

> 🎥 Yukarıdaki görsele tıklayarak bu dersle ilgili kısa bir videoya ulaşabilirsiniz.

## Giriş

Genel olarak, makine öğrenimi (ML) süreçlerini oluşturma sanatı birkaç adımdan oluşur:

1. **Soruyu belirleyin**. Çoğu ML süreci, basit bir koşullu program veya kurallara dayalı bir motorla cevaplanamayan bir soruyu sormakla başlar. Bu sorular genellikle bir veri koleksiyonuna dayalı tahminlerle ilgilidir.
2. **Veri toplayın ve hazırlayın**. Sorunuzu cevaplayabilmek için veriye ihtiyacınız var. Verinizin kalitesi ve bazen miktarı, başlangıçtaki sorunuza ne kadar iyi cevap verebileceğinizi belirler. Veriyi görselleştirmek bu aşamanın önemli bir parçasıdır. Bu aşama ayrıca veriyi bir eğitim ve test grubuna ayırmayı içerir.
3. **Eğitim yöntemini seçin**. Sorunuza ve verinizin doğasına bağlı olarak, verinizi en iyi şekilde yansıtacak ve doğru tahminler yapacak bir model eğitme yöntemini seçmeniz gerekir. Bu, ML sürecinizin özel uzmanlık gerektiren ve genellikle önemli miktarda deneme gerektiren kısmıdır.
4. **Modeli eğitin**. Eğitim verinizi kullanarak, çeşitli algoritmalarla bir model eğiterek verideki desenleri tanımasını sağlarsınız. Model, verinin belirli bölümlerini diğerlerine göre önceliklendirmek için ayarlanabilir içsel ağırlıklar kullanabilir.
5. **Modeli değerlendirin**. Topladığınız veri setinden daha önce hiç görülmemiş verileri (test verinizi) kullanarak modelin performansını değerlendirirsiniz.
6. **Parametre ayarı**. Modelinizin performansına bağlı olarak, modeli eğitmek için kullanılan algoritmaların davranışını kontrol eden farklı parametreler veya değişkenler kullanarak süreci yeniden yapabilirsiniz.
7. **Tahmin yapın**. Modelinizin doğruluğunu test etmek için yeni girdiler kullanın.

## Hangi Soruyu Sormalı?

Bilgisayarlar, verilerdeki gizli desenleri keşfetme konusunda oldukça yeteneklidir. Bu yetenek, belirli bir alanda basit bir kurallara dayalı motor oluşturarak kolayca cevaplanamayan soruları olan araştırmacılar için çok faydalıdır. Örneğin, bir aktüeryal görevde, bir veri bilimci sigara içenler ile içmeyenlerin ölüm oranları hakkında el yapımı kurallar oluşturabilir.

Ancak, birçok başka değişken denkleme dahil edildiğinde, bir ML modeli geçmiş sağlık geçmişine dayanarak gelecekteki ölüm oranlarını tahmin etmekte daha verimli olabilir. Daha neşeli bir örnek olarak, bir konumda Nisan ayı için hava durumu tahminleri yapmak, enlem, boylam, iklim değişikliği, okyanusa yakınlık, jet akımı desenleri ve daha fazlasını içeren verilere dayanabilir.

✅ Bu [sunum dosyası](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf), hava analizi için ML kullanımı hakkında tarihsel bir perspektif sunmaktadır.

## Model Oluşturmadan Önceki Görevler

Modelinizi oluşturmaya başlamadan önce tamamlamanız gereken birkaç görev vardır. Sorunuzu test etmek ve bir modelin tahminlerine dayalı bir hipotez oluşturmak için birkaç unsuru tanımlamanız ve yapılandırmanız gerekir.

### Veri

Sorunuzu herhangi bir kesinlikle cevaplayabilmek için doğru türde yeterli miktarda veriye ihtiyacınız var. Bu noktada yapmanız gereken iki şey var:

- **Veri toplayın**. Veri analizi hakkındaki önceki derste adalet konusunu göz önünde bulundurarak, verinizi dikkatlice toplayın. Bu verinin kaynaklarının, sahip olabileceği herhangi bir içsel önyargının farkında olun ve kökenini belgeleyin.
- **Veriyi hazırlayın**. Veri hazırlama sürecinde birkaç adım vardır. Veriler farklı kaynaklardan geliyorsa, verileri birleştirmeniz ve normalleştirmeniz gerekebilir. Verinin kalitesini ve miktarını, dizeleri sayılara dönüştürmek gibi çeşitli yöntemlerle artırabilirsiniz (örneğin [Kümeleme](../../5-Clustering/1-Visualize/README.md) dersinde yaptığımız gibi). Ayrıca, orijinal veriye dayanarak yeni veri üretebilirsiniz (örneğin [Sınıflandırma](../../4-Classification/1-Introduction/README.md) dersinde yaptığımız gibi). Veriyi temizleyebilir ve düzenleyebilirsiniz (örneğin [Web Uygulaması](../../3-Web-App/README.md) dersinden önce yapacağımız gibi). Son olarak, eğitim tekniklerinize bağlı olarak veriyi rastgeleleştirmeniz ve karıştırmanız gerekebilir.

✅ Verinizi topladıktan ve işledikten sonra, şeklinin hedeflediğiniz soruyu ele almanıza izin verip vermeyeceğini kontrol etmek için bir an durun. Verinin, belirli bir görevde iyi performans göstermeyeceğini [Kümeleme](../../5-Clustering/1-Visualize/README.md) derslerinde keşfettiğimiz gibi fark edebilirsiniz!

### Özellikler ve Hedef

Bir [özellik](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection), verinizin ölçülebilir bir özelliğidir. Çoğu veri setinde, 'tarih', 'boyut' veya 'renk' gibi sütun başlıkları olarak ifade edilir. Kodda genellikle `X` olarak temsil edilen özellik değişkeniniz, modeli eğitmek için kullanılacak giriş değişkenini temsil eder.

Bir hedef, tahmin etmeye çalıştığınız şeydir. Kodda genellikle `y` olarak temsil edilen hedef, verinizden sormaya çalıştığınız sorunun cevabını temsil eder: Aralık ayında hangi **renkteki** kabaklar en ucuz olacak? San Francisco'da hangi mahalleler en iyi gayrimenkul **fiyatına** sahip olacak? Hedef bazen etiket özelliği olarak da adlandırılır.

### Özellik Değişkeninizi Seçmek

🎓 **Özellik Seçimi ve Özellik Çıkarımı** Model oluştururken hangi değişkeni seçeceğinizi nasıl bileceksiniz? Muhtemelen en iyi performans gösteren model için doğru değişkenleri seçmek üzere bir özellik seçimi veya özellik çıkarımı sürecinden geçeceksiniz. Ancak, bunlar aynı şey değildir: "Özellik çıkarımı, orijinal özelliklerin fonksiyonlarından yeni özellikler oluştururken, özellik seçimi özelliklerin bir alt kümesini döndürür." ([kaynak](https://wikipedia.org/wiki/Feature_selection))

### Verinizi Görselleştirin

Bir veri bilimcinin araç setinin önemli bir yönü, Seaborn veya MatPlotLib gibi birkaç mükemmel kütüphaneyi kullanarak veriyi görselleştirme gücüdür. Verinizi görsel olarak temsil etmek, yararlanabileceğiniz gizli korelasyonları ortaya çıkarmanıza olanak sağlayabilir. Görselleştirmeleriniz ayrıca önyargı veya dengesiz veriyi ortaya çıkarmanıza yardımcı olabilir (örneğin [Sınıflandırma](../../4-Classification/2-Classifiers-1/README.md) dersinde keşfettiğimiz gibi).

### Veri Setinizi Bölün

Eğitimden önce, veri setinizi eşit olmayan boyutlarda iki veya daha fazla parçaya ayırmanız gerekir, ancak bu parçalar yine de veriyi iyi temsil etmelidir.

- **Eğitim**. Veri setinin bu kısmı, modelinizi eğitmek için modele uyarlanır. Bu set, orijinal veri setinin çoğunluğunu oluşturur.
- **Test**. Test veri seti, genellikle orijinal veriden toplanan bağımsız bir veri grubudur ve oluşturulan modelin performansını doğrulamak için kullanılır.
- **Doğrulama**. Doğrulama seti, modelin hiperparametrelerini veya mimarisini iyileştirmek için kullandığınız daha küçük bağımsız bir örnek grubudur. Verinizin boyutuna ve sorduğunuz soruya bağlı olarak, bu üçüncü seti oluşturmanız gerekmeyebilir (örneğin [Zaman Serisi Tahmini](../../7-TimeSeries/1-Introduction/README.md) dersinde belirttiğimiz gibi).

## Model Oluşturma

Eğitim verinizi kullanarak, amacınız çeşitli algoritmalar kullanarak verinizin istatistiksel bir temsilini oluşturmak, yani bir model oluşturmaktır. Modeli eğitmek, veriyi analiz etmesine, algıladığı desenler hakkında varsayımlar yapmasına, doğrulamasına ve kabul veya reddetmesine olanak tanır.

### Eğitim Yöntemini Belirleyin

Sorunuza ve verinizin doğasına bağlı olarak, onu eğitmek için bir yöntem seçersiniz. Bu kursta kullandığımız [Scikit-learn belgelerini](https://scikit-learn.org/stable/user_guide.html) inceleyerek bir modeli eğitmek için birçok yöntemi keşfedebilirsiniz. Deneyiminize bağlı olarak, en iyi modeli oluşturmak için birkaç farklı yöntemi denemeniz gerekebilir. Veri bilimcilerin bir modeli performansını değerlendirmek için daha önce görülmemiş verilerle beslediği, doğruluk, önyargı ve diğer kaliteyi düşüren sorunları kontrol ettiği ve mevcut görev için en uygun eğitim yöntemini seçtiği bir süreçten geçmeniz muhtemeldir.

### Modeli Eğitin

Eğitim verinizle donanmış olarak, bir model oluşturmak için onu 'uydurmaya' hazırsınız. Birçok ML kütüphanesinde 'model.fit' kodunu göreceksiniz - bu, özellik değişkeninizi (genellikle 'X') ve hedef değişkeninizi (genellikle 'y') bir değer dizisi olarak gönderdiğiniz zamandır.

### Modeli Değerlendirin

Eğitim süreci tamamlandıktan sonra (büyük bir modeli eğitmek için birçok yineleme veya 'epoch' gerekebilir), modelin kalitesini test verilerini kullanarak performansını ölçerek değerlendirebilirsiniz. Bu veri, modelin daha önce analiz etmediği orijinal verinin bir alt kümesidir. Modelinizin kalitesi hakkında bir metrik tablosu yazdırabilirsiniz.

🎓 **Model Uydurma**

Makine öğrenimi bağlamında, model uydurma, modelin tanımadığı veriyi analiz etmeye çalışırken temel fonksiyonunun doğruluğunu ifade eder.

🎓 **Eksik Uydurma** ve **Aşırı Uydurma**, modelin kalitesini düşüren yaygın sorunlardır. Model ya yeterince iyi uydurulmaz ya da çok iyi uydurulur. Bu, modelin tahminlerini ya eğitim verisine çok sıkı ya da çok gevşek bir şekilde hizalamasına neden olur. Aşırı uydurulmuş bir model, verinin ayrıntılarını ve gürültüsünü çok iyi öğrendiği için eğitim verisini çok iyi tahmin eder. Eksik uydurulmuş bir model ise ne eğitim verisini ne de daha önce 'görmediği' veriyi doğru bir şekilde analiz edebilir.

![aşırı uydurma modeli](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> [Jen Looper](https://twitter.com/jenlooper) tarafından hazırlanan infografik

## Parametre Ayarı

İlk eğitiminiz tamamlandıktan sonra, modelin kalitesini gözlemleyin ve 'hiperparametrelerini' ayarlayarak iyileştirmeyi düşünün. Süreç hakkında daha fazla bilgi için [belgelere](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott) göz atın.

## Tahmin

Bu, tamamen yeni verileri kullanarak modelinizin doğruluğunu test edebileceğiniz andır. 'Uygulamalı' bir ML ortamında, modeli üretimde kullanmak için web varlıkları oluşturduğunuzda, bu süreç bir değişkeni ayarlamak ve değerlendirme veya çıkarım için modele göndermek üzere kullanıcı girdisi (örneğin bir düğme basışı) toplamayı içerebilir.

Bu derslerde, bir veri bilimcinin tüm hareketlerini ve daha fazlasını keşfederek, bir 'tam yığın' ML mühendisi olma yolculuğunuzda ilerlerken bu adımları hazırlama, oluşturma, test etme, değerlendirme ve tahmin yapma süreçlerini öğreneceksiniz.

---

## 🚀Meydan Okuma

Bir ML uygulayıcısının adımlarını yansıtan bir akış şeması çizin. Sürecin şu an neresinde olduğunuzu düşünüyorsunuz? Nerede zorluk yaşayacağınızı tahmin ediyorsunuz? Size kolay gelen nedir?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Günlük işlerini tartışan veri bilimcilerle yapılan röportajları çevrimiçi arayın. İşte [bir tanesi](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Ödev

[Bir veri bilimciyle röportaj yapın](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.