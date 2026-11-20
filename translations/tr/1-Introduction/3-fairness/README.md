# Sorumlu AI ile Makine Öğrenimi çözümleri geliştirmek
 
![Makine Öğreniminde sorumlu AI'nın bir sketchnote özeti](../../../../translated_images/tr/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote [Tomomi Imura](https://www.twitter.com/girlie_mac) tarafından

## [Ön-ders sınavı](https://ff-quizzes.netlify.app/en/ml/)
 
## Giriş

Bu müfredatta, makine öğrenmesinin nasıl ve günlük yaşamımızı nasıl etkilediğini keşfetmeye başlayacaksınız. Hatta şu anda bile, sistemler ve modeller sağlık teşhisleri, kredi onayları veya dolandırıcılık tespiti gibi günlük karar verme görevlerinde yer almaktadır. Bu nedenle, bu modellerin güvenilir sonuçlar sağlamak için iyi çalışması önemlidir. Herhangi bir yazılım uygulaması gibi, AI sistemleri de beklentileri karşılamayabilir veya istenmeyen bir sonuç verebilir. Bu yüzden bir AI modelinin davranışını anlayabilmek ve açıklayabilmek çok önemlidir.

Bu modelleri oluşturmak için kullandığınız veriler belirli demografik özelliklerden, örneğin ırk, cinsiyet, politik görüş, din gibi unsurlardan yoksunsa ya da bu demografikleri orantısız bir şekilde temsil ediyorsa ne olabilir? Modelin çıktısı bazı demografik grupları kayıracak şekilde yorumlanırsa ne olur? Uygulama için sonucu nedir? Ayrıca, model olumsuz bir sonuç verdiğinde ve insanlara zarar verdiğinde ne olur? AI sisteminin davranışından kim sorumludur? İşte bu müfredatta keşfedeceğimiz bazı sorular bunlardır.

Bu derste siz:

- Makine öğreniminde adaletin ve adaletle ilgili zararın önemine dair farkındalığınızı artıracaksınız.
- Güvenilirlik ve güvenlik sağlamak için aykırı değerleri ve alışılmadık senaryoları keşfetme uygulaması ile tanışacaksınız.
- Kapsayıcı sistemler tasarlayarak herkesi güçlendirmenin gerekliliğini anlayacaksınız.
- Verilerin ve insanların gizliliğini ve güvenliğini korumanın ne kadar hayati olduğunu keşfedeceksiniz.
- AI modellerinin davranışını açıklamak için cam kutu yaklaşımının önemini göreceksiniz.
- AI sistemlerinde güven yaratmak için hesap verebilirliğin ne kadar gerekli olduğunu göz önünde bulunduracaksınız.

## Önkoşul

Ön koşul olarak, lütfen "Sorumlu AI İlkeleri" Öğrenme Yolunu tamamlayın ve aşağıdaki videoyu izleyin:

Sorumlu AI hakkında daha fazla öğrenmek için bu [Öğrenme Yolunu](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) takip edin

[![Microsoft'un Sorumlu AI Yaklaşımı](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft'un Sorumlu AI Yaklaşımı")

> 🎥 Yukarıdaki görsele tıklayarak video izleyin: Microsoft'un Sorumlu AI Yaklaşımı

## Adalet

AI sistemleri herkese adil davranmalı ve benzer insan gruplarını farklı şekillerde etkilemekten kaçınmalıdır. Örneğin, AI sistemleri tıbbi tedavi, kredi başvuruları veya işe alım konusunda rehberlik sağladığında, benzer belirtilere, finansal durumlara veya mesleki niteliklere sahip herkese aynı önerileri vermelidir. Hepimiz insan olarak kararlarımızı ve eylemlerimizi etkileyen kalıtsal önyargılar taşıyoruz. Bu önyargılar AI sistemlerini eğitmek için kullandığımız verilere de yansıyabilir. Bazen bu tür manipülasyonlar farkında olmadan gerçekleşebilir. Veriye kasten bilinçli şekilde önyargı katıldığını anlamak genellikle zordur.

**"Adaletsizlik"**, ırk, cinsiyet, yaş veya engellilik durumu gibi tanımlanan bir insan grubuna yönelik olumsuz etkileri veya "zararları" kapsar. Ana adaletle ilgili zararlar şu şekilde sınıflandırılabilir:

- **Tahsis**, örneğin bir cinsiyetin veya etnik grubun diğerine kıyasla tercih edilmesi.
- **Hizmet kalitesi**. Verileri belirli bir senaryo için eğitirseniz ancak gerçeklik çok daha karmaşıksa, zayıf performans gösteren bir hizmet ortaya çıkar. Örneğin, koyu tenli insanları algılayamayan bir el sabunu dağıtıcısı. [Referans](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Aşağılama**. Bir şeyi veya birini haksızca eleştirmek ve etiketlemek. Örneğin, görüntü etiketleme teknolojisi koyu tenli kişilerin resimlerini maymun olarak yanlış etiketlemesiyle kötü üne kavuştu.
- **Aşırı veya yetersiz temsil**. Bazı grupların belirli mesleklerde görülmemesi durumu ve bunu teşvik eden herhangi bir hizmet veya fonksiyonun zarara katkıda bulunması.
- **Stereotipleme**. Bir grubu önceden atanan özelliklerle ilişkilendirmek. Örneğin, İngilizce ve Türkçe arasında dil çeviri sistemlerinde, cinsiyetle ilgili stereotipik çağrışımlar içeren kelimeler nedeniyle doğruluk sorunları olabilir.

![Türkçeye çeviri](../../../../translated_images/tr/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> Türkçeye çeviri

![İngilizceye geri çeviri](../../../../translated_images/tr/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> İngilizceye geri çeviri

AI sistemlerini tasarlarken ve test ederken, AI'nın adil olduğundan ve programlanmış olarak önyargılı ya da ayrımcı kararlar vermediğinden emin olmamız gerekir; çünkü insanlara da bu tür kararlar vermek yasaktır. AI ve makine öğreniminde adaleti garanti etmek karmaşık bir sosyo-teknik sorundur.

### Güvenilirlik ve güvenlik

Güven inşa etmek için AI sistemleri normal ve beklenmedik koşullar altında güvenilir, güvenli ve tutarlı olmalıdır. AI sistemlerinin çeşitli durumlarda, özellikle aykırı durumlarda nasıl davranacağını bilmek çok önemlidir. AI çözümleri geliştirilirken, AI'nın karşılaşacağı çok çeşitli koşulların nasıl ele alınacağına büyük önem verilmelidir. Örneğin, otonom bir araba insanların güvenliğini en üst düzeyde tutmalıdır. Buna bağlı olarak, arabayı çalıştıran AI, gece, fırtına veya kar fırtınası, sokaktan koşan çocuklar, evcil hayvanlar, yol çalışmaları gibi karşılaşabileceği tüm senaryoları göz önünde bulundurmalıdır. Bir AI sisteminin çok çeşitli koşulları güvenilir ve güvenli bir şekilde yönetebilme yeteneği, veri bilimcisi veya AI geliştiricisinin tasarım veya test aşamasında ne kadar öngörüde bulunduğunu yansıtır.

> [🎥 AI'da Güvenilirlik ve Güvenlik videosu için buraya tıklayın](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Kapsayıcılık

AI sistemleri herkesin katılımını sağlamalı ve herkesi güçlendirecek şekilde tasarlanmalıdır. AI sistemlerini tasarlayan ve uygulayan veri bilimciler ve AI geliştiriciler, insanların kazara dışlanmasına yol açabilecek potansiyel engelleri belirler ve çözüm üretir. Örneğin, dünya genelinde 1 milyar engelli kişi var. AI ilerledikçe, günlük hayatlarında çok çeşitli bilgi ve fırsatlara daha kolay erişebilirler. Engeller kaldırıldığında, herkese fayda sağlayan daha iyi deneyimler sunan AI ürünleri geliştirme ve yenilik yapma fırsatı ortaya çıkar.

> [🎥 AI'da Kapsayıcılık videosu için buraya tıklayın](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Güvenlik ve gizlilik

AI sistemleri güvenli olmalı ve insanların gizliliğine saygı göstermelidir. İnsanlar, gizliliklerini, bilgilerini veya yaşamlarını riske atan sistemlere daha az güvenirler. Makine öğrenimi modellerini eğitirken, en iyi sonuçları almak için veriye dayanırız. Bu nedenle verinin kaynağı ve bütünlüğü göz önünde bulundurulmalıdır. Örneğin, veri kullanıcı tarafından mı gönderildi yoksa kamuya açık mı? Daha sonra, verilerle çalışırken gizli bilgileri koruyabilen ve saldırılara karşı dayanıklı AI sistemleri geliştirmek kritik önemdedir. AI yaygınlaştıkça, gizliliği korumak ve önemli kişisel ve ticari bilgileri güvence altına almak giderek daha karmaşık ve önemli hale geliyor. Gizlilik ve veri güvenliği sorunları, veri erişiminin AI sistemlerinin insanlar hakkında doğru ve bilinçli tahminler ve kararlar verebilmeleri için gerekli olması nedeniyle AI için özellikle dikkat gerektirir.

> [🎥 AI'da Güvenlik videosu için buraya tıklayın](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sektör olarak GDPR (Genel Veri Koruma Yönetmeliği) gibi düzenlemelerle önemli ilerlemeler kaydettik.
- Ancak AI sistemlerinde, sistemleri daha kişisel ve etkili yapmak için daha fazla kişisel veriye duyulan ihtiyaç ile gizlilik arasındaki gerilimi kabul etmeliyiz.
- İnternetle bağlantılı bilgisayarların doğuşunda olduğu gibi, AI ile ilgili güvenlik sorunlarının sayısında da büyük artışlar görüyoruz.
- Aynı zamanda, AI'nın güvenliği artırmak için kullanıldığını da gördük. Örneğin, çoğu modern antivirüs tarayıcısı bugün AI sezgisel kurallarıyla çalışmaktadır.
- Veri Bilimi süreçlerimizin en son gizlilik ve güvenlik uygulamalarıyla uyum içinde olduğundan emin olmamız gerekiyor.


### Şeffaflık
AI sistemleri anlaşılır olmalıdır. Şeffaflığın kritik bir parçası, AI sistemlerinin ve bileşenlerinin davranışlarını açıklamaktır. AI sistemlerinin anlaşılmasını geliştirmek, paydaşların bu sistemlerin nasıl ve neden çalıştığını anlamalarını gerektirir; böylece potansiyel performans sorunlarını, güvenlik ve gizlilik endişelerini, önyargıları, hariç tutucu uygulamaları veya istenmeyen sonuçları tanımlayabilirler. Ayrıca AI sistemlerini kullananların, ne zaman, neden ve nasıl kullandıklarını ve sistemlerin sınırlamalarını dürüstçe bildirmeleri gerektiğine inanıyoruz. Örneğin, bir banka tüketici kredi kararlarını desteklemek için AI sistemini kullanıyorsa, sonuçları incelemek ve sistemin önerilerini hangi verilerin etkilediğini anlamak önemlidir. Hükümetler AI'yı endüstrilerde düzenlemeye başladı, bu nedenle veri bilimciler ve kuruluşlar, özellikle istenmeyen bir sonuç olduğunda, AI sisteminin düzenleyici gereksinimleri karşılayıp karşılamadığını açıklamalıdır.

> [🎥 AI'da Şeffaflık videosu için buraya tıklayın](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- AI sistemleri çok karmaşık olduğu için nasıl çalıştıklarını anlamak ve sonuçları yorumlamak zordur.
- Bu anlayış eksikliği, bu sistemlerin nasıl yönetildiğini, işletildiğini ve belgelendiğini etkiler.
- Daha da önemlisi, bu anlayış eksikliği, bu sistemlerin ürettiği sonuçlarla verilen kararları etkiler.

### Hesap verebilirlik
 
AI sistemlerini tasarlayan ve kullanan kişiler sistemlerin nasıl çalıştığından sorumlu olmalıdır. Hesap verebilirlik özellikle yüz tanıma gibi hassas teknolojilerde kritik önemdedir. Son zamanlarda, özellikle kayıp çocukları bulma gibi kullanımlarda bu teknolojiye olan talep artmıştır. Ancak bu teknolojiler, örneğin belirli bireylerin sürekli gözetimini mümkün kılarak, bir hükümetin vatandaşlarının temel özgürlüklerini riske atmak için kullanabileceği potansiyel tehlikeler barındırır. Bu nedenle veri bilimciler ve kuruluşlar AI sistemlerinin bireyler veya toplum üzerindeki etkilerinden sorumlu olmalıdır.

[![Yüz Tanıma ile Kapsamlı Gözetim Konusunda Önde Gelen AI Araştırmacısından Uyarı](../../../../translated_images/tr/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft'un Sorumlu AI Yaklaşımı")

> 🎥 Yukarıdaki görsele tıklayarak video izleyin: Yüz Tanıma ile Kapsamlı Gözetim Uyarıları

Sonuç olarak, toplumla AI'yı tanıştıran ilk nesil olarak en büyük sorulardan biri, bilgisayarların insanlara karşı hesap verebilir kalmasını nasıl sağlayacağımız ve bilgisayarları tasarlayan insanların diğer herkes karşısında nasıl hesap verebilir olacağını sağlamaktır.

## Etki değerlendirmesi

Bir makine öğrenimi modeli eğitmeden önce, AI sisteminin amacını; kullanım şeklinin ne olduğunu; nerede uygulama yapılacağını; ve sistemle kimlerin etkileşimde bulunacağını anlamak için bir etki değerlendirmesi yapmak önemlidir. Bu, sistemi değerlendiren gözden geçirenler veya test edenler için potansiyel riskleri ve beklenen sonuçları belirlerken dikkate almaları gereken faktörleri bilmek açısından faydalıdır.

Etki değerlendirmesi yaparken odaklanılan alanlar şunlardır:

* **Bireylere olumsuz etkiler**. Herhangi bir kısıtlama veya gereksinim, desteklenmeyen kullanım ya da bilinen sınırlamalar sistemin performansını engelliyorsa bunun farkında olmak ve sistemin bireylere zarar verecek şekilde kullanılmamasını sağlamak kritiktir.
* **Veri gereksinimleri**. Sistemin veriyi nasıl ve nerede kullanacağını anlamak, gözden geçirenlerin dikkat etmesi gereken veri gereksinimlerini (örneğin GDPR veya HIPAA düzenlemeleri) anlamasını sağlar. Ayrıca, verinin kaynağı veya miktarının eğitim için yeterli olup olmadığını incelemek gerekir.
* **Etki özeti**. Sistemin kullanımıyla ortaya çıkabilecek potansiyel zararları listeleyin. Makine öğrenimi yaşam döngüsü boyunca tespit edilen sorunların hafifletilip giderilip giderilmediğini gözden geçirin.
* Altı temel prensibin her biri için **geçerli hedefler**. Her prensibin hedeflerinin karşılanıp karşılanmadığını ve varsa eksiklikleri değerlendirin.


## Sorumlu AI ile hata ayıklama  

Bir yazılım uygulamasında hata ayıklamak gibi, bir AI sisteminde hata ayıklamak da sistemdeki sorunları tanımlamak ve çözmek için gerekli bir süreçtir. Bir modelin beklendiği gibi veya sorumlu şekilde performans göstermemesini etkileyen birçok faktör olabilir. Geleneksel model performans ölçütlerinin çoğu modelin performansının sayısal özetleridir ve bir modelin sorumlu AI prensiplerine nasıl aykırı davrandığını analiz etmeye yetmez. Ayrıca, makine öğrenimi modeli bir kara kutudur ve çıktısını nelerin etkilediğini anlamayı veya hata yaptığında açıklama yapmayı zorlaştırır. Bu kursta daha sonra, AI sistemlerini hata ayıklamada yardımcı olan Sorumlu AI panosunu nasıl kullanacağımızı öğreneceğiz. Pano, veri bilimciler ve AI geliştiriciler için bütünsel bir araç sağlar:

* **Hata analizi**. Sistemin adaletini veya güvenilirliğini etkileyebilecek hata dağılımını tanımlamak için.
* **Model genel görünümü**. Modelin farklı veri kümeleri üzerindeki performansındaki farklılıkları keşfetmek için.
* **Veri analizi**. Veri dağılımını anlamak ve adalet, kapsayıcılık ve güvenilirlik sorunlarına yol açabilecek önyargıları belirlemek için.
* **Model yorumlanabilirliği**. Modelin tahminlerini neyin etkilediğini anlamak için. Bu, şeffaflık ve hesap verebilirlik açısından önemlidir.


## 🚀 Zorluk
 
Zararlara sebebiyet verilmesini önlemek için:

- Sistemde çalışan kişiler arasında çeşitli geçmişler ve bakış açıları olmalı
- Toplumun çeşitliliğini yansıtan veri setlerine yatırım yapılmalı
- Makine öğrenimi yaşam döngüsü boyunca sorumsuz AI’yi tespit edip düzeltmek için daha iyi yöntemler geliştirilmelidir

Bir modelin güvensizliğinin model oluşturma ve kullanımda belirgin olduğu gerçek hayat senaryolarını düşünün. Başka neleri göz önünde bulundurmalıyız?

## [Sonrasında sınav](https://ff-quizzes.netlify.app/en/ml/)

## Tekrar ve Kendi Kendine Çalışma
 
Bu derste, makine öğreniminde adalet ve adaletsizlik kavramları hakkında bazı temel bilgileri öğrendiniz.  
 
Konuları daha derinlemesine incelemek için bu atölyeyi izleyin: 

- Sorumlu AI peşinde: İlkeleri uygulamaya dönüştürmek, Besmira Nushi, Mehrnoosh Sameki ve Amit Sharma tarafından

[![Sorumlu AI Araç Kutusu: Sorumlu AI oluşturmak için açık kaynaklı bir çerçeve](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Sorumlu AI oluşturmak için açık kaynaklı bir çerçeve")

> 🎥 Yukarıdaki resme tıklayın: Besmira Nushi, Mehrnoosh Sameki ve Amit Sharma tarafından hazırlanan Sorumlu AI oluşturmak için açık kaynaklı bir çerçeve: RAI Toolbox videosu

Ayrıca şunları okuyun: 

- Microsoft’un RAI kaynak merkezi: [Sorumlu AI Kaynakları – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft’un FATE araştırma grubu: [FATE: AI’da Adalet, Hesap Verebilirlik, Şeffaflık ve Etik - Microsoft Araştırma](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Sorumlu AI Araç Kutusu GitHub deposu](https://github.com/microsoft/responsible-ai-toolbox)

Adalet sağlamak için Azure Machine Learning araçları hakkında bilgi edinin:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Ödev

[RAI Toolbox'u Keşfet](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlamalardan veya yanlış yorumlamalardan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->