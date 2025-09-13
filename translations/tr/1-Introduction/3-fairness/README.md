<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-06T07:54:44+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "tr"
}
-->
# Sorumlu Yapay Zeka ile Makine Öğrenimi Çözümleri Oluşturma

![Makine Öğreniminde sorumlu yapay zekanın özetini içeren bir sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## Giriş

Bu müfredatta, makine öğreniminin günlük hayatımızı nasıl etkilediğini ve etkilemeye devam ettiğini keşfetmeye başlayacaksınız. Şu anda bile, sistemler ve modeller sağlık teşhisleri, kredi onayları veya dolandırıcılığı tespit etme gibi günlük karar verme görevlerinde yer alıyor. Bu nedenle, bu modellerin güvenilir sonuçlar sunmak için iyi çalışması önemlidir. Her yazılım uygulaması gibi, yapay zeka sistemleri de beklentileri karşılamayabilir veya istenmeyen sonuçlar doğurabilir. Bu yüzden bir yapay zeka modelinin davranışını anlamak ve açıklamak çok önemlidir.

Bu modelleri oluşturmak için kullandığınız veriler belirli demografik grupları (ırk, cinsiyet, siyasi görüş, din gibi) içermediğinde veya bu demografik grupları orantısız bir şekilde temsil ettiğinde neler olabileceğini hayal edin. Peki ya modelin çıktısı bazı demografik grupları kayıracak şekilde yorumlandığında ne olur? Uygulama için sonuçları nelerdir? Ayrıca, modelin olumsuz bir sonucu olduğunda ve insanlara zarar verdiğinde ne olur? Yapay zeka sistemlerinin davranışından kim sorumludur? Bu müfredatta bu tür soruları inceleyeceğiz.

Bu derste şunları öğreneceksiniz:

- Makine öğreniminde adaletin önemi ve adaletle ilgili zararlar hakkında farkındalık oluşturmak.
- Güvenilirlik ve güvenliği sağlamak için aykırı değerleri ve olağandışı senaryoları keşfetme pratiğini öğrenmek.
- Herkesin kapsayıcı sistemler tasarlayarak güçlendirilmesi gerektiğini anlamak.
- Verilerin ve insanların gizliliğini ve güvenliğini korumanın ne kadar önemli olduğunu keşfetmek.
- Yapay zeka modellerinin davranışını açıklamak için şeffaf bir yaklaşımın önemini görmek.
- Yapay zeka sistemlerinde güven oluşturmak için hesap verebilirliğin ne kadar önemli olduğunu anlamak.

## Ön Koşul

Ön koşul olarak, "Sorumlu Yapay Zeka İlkeleri" öğrenme yolunu tamamlayın ve aşağıdaki videoyu izleyin:

Sorumlu Yapay Zeka hakkında daha fazla bilgi edinmek için bu [Öğrenme Yolunu](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) takip edin.

[![Microsoft'un Sorumlu Yapay Zeka Yaklaşımı](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft'un Sorumlu Yapay Zeka Yaklaşımı")

> 🎥 Yukarıdaki görüntüye tıklayarak videoyu izleyin: Microsoft'un Sorumlu Yapay Zeka Yaklaşımı

## Adalet

Yapay zeka sistemleri herkese adil davranmalı ve benzer grupları farklı şekillerde etkilemekten kaçınmalıdır. Örneğin, yapay zeka sistemleri tıbbi tedavi, kredi başvuruları veya istihdam konusunda rehberlik sağladığında, benzer semptomlara, finansal koşullara veya mesleki niteliklere sahip herkese aynı önerileri sunmalıdır. Hepimiz, kararlarımızı ve eylemlerimizi etkileyen kalıtsal önyargılar taşırız. Bu önyargılar, yapay zeka sistemlerini eğitmek için kullandığımız verilere yansıyabilir. Bu tür manipülasyonlar bazen istemeden gerçekleşebilir. Verilere önyargı eklediğinizi bilinçli olarak fark etmek genellikle zordur.

**“Adaletsizlik”**, bir grup insan için (örneğin ırk, cinsiyet, yaş veya engellilik durumu açısından tanımlanan) olumsuz etkileri veya “zararları” kapsar. Adaletle ilgili başlıca zararlar şu şekilde sınıflandırılabilir:

- **Tahsis**: Örneğin, bir cinsiyet veya etnik kökenin diğerine tercih edilmesi.
- **Hizmet kalitesi**: Verileri yalnızca belirli bir senaryo için eğitmek, ancak gerçekliğin çok daha karmaşık olması, kötü performans gösteren bir hizmete yol açar. Örneğin, koyu tenli insanları algılayamayan bir el sabunu dağıtıcısı. [Referans](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Aşağılama**: Bir şeyi veya birini haksız yere eleştirmek ve etiketlemek. Örneğin, bir görüntü etiketleme teknolojisi koyu tenli insanları goril olarak yanlış etiketlemiştir.
- **Aşırı veya yetersiz temsil**: Belirli bir grubun belirli bir meslekte görülmemesi fikri ve bu durumu sürekli olarak teşvik eden herhangi bir hizmet veya işlev zarara katkıda bulunur.
- **Stereotipleme**: Belirli bir grubu önceden atanmış özelliklerle ilişkilendirme. Örneğin, İngilizce ve Türkçe arasında çeviri yapan bir dil sistemi, cinsiyetle ilgili stereotipik ilişkilere dayalı hatalar içerebilir.

![Türkçeye çeviri](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> Türkçeye çeviri

![İngilizceye geri çeviri](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> İngilizceye geri çeviri

Yapay zeka sistemlerini tasarlarken ve test ederken, yapay zekanın adil olmasını ve önyargılı veya ayrımcı kararlar vermek üzere programlanmamasını sağlamamız gerekir. Yapay zekada ve makine öğreniminde adaleti garanti altına almak, karmaşık bir sosyo-teknik zorluk olmaya devam ediyor.

### Güvenilirlik ve Güvenlik

Güven oluşturmak için yapay zeka sistemlerinin normal ve beklenmedik koşullar altında güvenilir, güvenli ve tutarlı olması gerekir. Yapay zeka sistemlerinin çeşitli durumlarda nasıl davranacağını bilmek önemlidir, özellikle de aykırı değerler söz konusu olduğunda. Yapay zeka çözümleri oluştururken, yapay zeka çözümlerinin karşılaşabileceği çeşitli koşulları ele alma konusunda önemli bir odaklanma gereklidir. Örneğin, bir otonom araç insanların güvenliğini en öncelikli olarak ele almalıdır. Sonuç olarak, aracı çalıştıran yapay zeka, gece, fırtına, kar fırtınası, yola koşan çocuklar, evcil hayvanlar, yol çalışmaları gibi aracın karşılaşabileceği tüm olası senaryoları dikkate almalıdır. Bir yapay zeka sisteminin geniş bir koşul yelpazesini ne kadar güvenilir ve güvenli bir şekilde ele alabildiği, veri bilimci veya yapay zeka geliştiricisinin tasarım veya test sırasında ne kadar öngörüde bulunduğunu yansıtır.

> [🎥 Video için buraya tıklayın: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Kapsayıcılık

Yapay zeka sistemleri herkesin katılımını sağlamalı ve güçlendirmelidir. Yapay zeka sistemlerini tasarlarken ve uygularken, veri bilimciler ve yapay zeka geliştiriciler, sistemi istemeden insanları dışlayabilecek potansiyel engelleri belirler ve ele alır. Örneğin, dünya genelinde 1 milyar engelli insan bulunmaktadır. Yapay zekanın ilerlemesiyle, günlük yaşamlarında bilgiye ve fırsatlara daha kolay erişebilirler. Engelleri ele alarak, herkes için daha iyi deneyimler sunan yapay zeka ürünlerini yenilikçi bir şekilde geliştirme fırsatları yaratılır.

> [🎥 Video için buraya tıklayın: yapay zekada kapsayıcılık](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Güvenlik ve Gizlilik

Yapay zeka sistemleri güvenli olmalı ve insanların gizliliğine saygı göstermelidir. Gizliliği, bilgileri veya hayatları riske atan sistemlere insanlar daha az güvenir. Makine öğrenimi modellerini eğitirken, en iyi sonuçları elde etmek için verilere güveniriz. Bunu yaparken, verilerin kaynağı ve bütünlüğü dikkate alınmalıdır. Örneğin, veriler kullanıcı tarafından mı gönderildi yoksa kamuya açık mıydı? Ardından, verilerle çalışırken, gizli bilgileri koruyabilen ve saldırılara karşı dirençli yapay zeka sistemleri geliştirmek önemlidir. Yapay zeka daha yaygın hale geldikçe, gizliliği korumak ve önemli kişisel ve iş bilgilerini güvence altına almak giderek daha kritik ve karmaşık hale geliyor. Gizlilik ve veri güvenliği sorunları, yapay zeka için özellikle dikkat gerektirir çünkü verilere erişim, yapay zeka sistemlerinin insanlar hakkında doğru ve bilgilendirilmiş tahminler ve kararlar vermesi için gereklidir.

> [🎥 Video için buraya tıklayın: yapay zekada güvenlik](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Endüstri olarak, GDPR (Genel Veri Koruma Yönetmeliği) gibi düzenlemelerle önemli ilerlemeler kaydettik.
- Ancak yapay zeka sistemleriyle, sistemleri daha kişisel ve etkili hale getirmek için daha fazla kişisel veri ihtiyacı ile gizlilik arasındaki gerilimi kabul etmeliyiz.
- İnternetle bağlantılı bilgisayarların doğuşunda olduğu gibi, yapay zeka ile ilgili güvenlik sorunlarında da büyük bir artış görüyoruz.
- Aynı zamanda, yapay zekanın güvenliği artırmak için kullanıldığını da görüyoruz. Örneğin, modern antivirüs tarayıcılarının çoğu bugün yapay zeka sezgileriyle çalışıyor.
- Veri Bilimi süreçlerimizin en son gizlilik ve güvenlik uygulamalarıyla uyum içinde olmasını sağlamalıyız.

### Şeffaflık

Yapay zeka sistemleri anlaşılabilir olmalıdır. Şeffaflığın önemli bir kısmı, yapay zeka sistemlerinin ve bileşenlerinin davranışını açıklamaktır. Yapay zeka sistemlerinin anlaşılmasını geliştirmek, paydaşların nasıl ve neden çalıştıklarını anlamalarını gerektirir, böylece potansiyel performans sorunlarını, güvenlik ve gizlilik endişelerini, önyargıları, dışlayıcı uygulamaları veya istenmeyen sonuçları belirleyebilirler. Ayrıca, yapay zeka sistemlerini kullananların, bunları ne zaman, neden ve nasıl kullanmayı seçtikleri konusunda dürüst ve açık olmaları gerektiğine inanıyoruz. Kullandıkları sistemlerin sınırlamaları hakkında da bilgi vermelidirler. Örneğin, bir banka tüketici kredi kararlarını desteklemek için bir yapay zeka sistemi kullanıyorsa, sonuçları incelemek ve sistemin önerilerini hangi verilerin etkilediğini anlamak önemlidir. Hükümetler, endüstrilerde yapay zekayı düzenlemeye başladığından, veri bilimciler ve kuruluşlar, bir yapay zeka sisteminin düzenleyici gereklilikleri karşılayıp karşılamadığını, özellikle istenmeyen bir sonuç olduğunda açıklamalıdır.

> [🎥 Video için buraya tıklayın: yapay zekada şeffaflık](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Yapay zeka sistemleri çok karmaşık olduğu için nasıl çalıştıklarını anlamak ve sonuçları yorumlamak zordur.
- Bu anlayış eksikliği, bu sistemlerin nasıl yönetildiğini, işletildiğini ve belgelenmesini etkiler.
- Daha da önemlisi, bu anlayış eksikliği, bu sistemlerin ürettiği sonuçlara dayanarak alınan kararları etkiler.

### Hesap Verebilirlik

Yapay zeka sistemlerini tasarlayan ve uygulayan kişiler, sistemlerinin nasıl çalıştığı konusunda hesap verebilir olmalıdır. Hesap verebilirlik ihtiyacı, özellikle yüz tanıma gibi hassas teknolojilerde çok önemlidir. Son zamanlarda, özellikle kayıp çocukları bulmak gibi kullanımlarda teknolojinin potansiyelini gören kolluk kuvvetleri tarafından yüz tanıma teknolojisine olan talep artmıştır. Ancak, bu teknolojiler bir hükümet tarafından vatandaşlarının temel özgürlüklerini riske atmak için kullanılabilir, örneğin belirli bireylerin sürekli gözetimini sağlamak için. Bu nedenle, veri bilimciler ve kuruluşlar, yapay zeka sistemlerinin bireyler veya toplum üzerindeki etkisinden sorumlu olmalıdır.

[![Önde Gelen Yapay Zeka Araştırmacısı Yüz Tanıma Yoluyla Toplu Gözetim Uyarısı](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft'un Sorumlu Yapay Zeka Yaklaşımı")

> 🎥 Yukarıdaki görüntüye tıklayarak videoyu izleyin: Yüz Tanıma Yoluyla Toplu Gözetim Uyarısı

Sonuç olarak, yapay zekayı topluma getiren ilk nesil olarak, bilgisayarların insanlara karşı hesap verebilir olmasını nasıl sağlayacağımız ve bilgisayarları tasarlayan insanların diğer herkese karşı hesap verebilir olmasını nasıl sağlayacağımız, neslimizin en büyük sorularından biridir.

## Etki Değerlendirmesi

Bir makine öğrenimi modelini eğitmeden önce, yapay zeka sisteminin amacını, planlanan kullanımını, nerede uygulanacağını ve sistemle kimlerin etkileşimde bulunacağını anlamak için bir etki değerlendirmesi yapmak önemlidir. Bunlar, sistemi değerlendiren inceleyiciler veya test edenler için potansiyel riskleri ve beklenen sonuçları belirlerken dikkate alınması gereken faktörleri anlamalarına yardımcı olur.

Etki değerlendirmesi yaparken odaklanılması gereken alanlar şunlardır:

* **Bireyler üzerindeki olumsuz etkiler**. Sistemin performansını engelleyen herhangi bir kısıtlama veya gereklilik, desteklenmeyen kullanım veya bilinen sınırlamaların farkında olmak, sistemin bireylere zarar verebilecek şekilde kullanılmamasını sağlamak için önemlidir.
* **Veri gereksinimleri**. Sistemin verileri nasıl ve nerede kullanacağını anlamak, inceleyicilerin dikkate alması gereken veri gereksinimlerini (örneğin, GDPR veya HIPPA veri düzenlemeleri) keşfetmelerini sağlar. Ayrıca, verilerin kaynağı veya miktarının eğitim için yeterli olup olmadığını inceleyin.
* **Etki özeti**. Sistemin kullanımından kaynaklanabilecek potansiyel zararların bir listesini toplayın. Makine öğrenimi yaşam döngüsü boyunca, belirlenen sorunların azaltılıp azaltılmadığını veya ele alınıp alınmadığını gözden geçirin.
* **Altı temel ilke için uygulanabilir hedefler**. Her bir ilkenin hedeflerinin karşılanıp karşılanmadığını ve herhangi bir boşluk olup olmadığını değerlendirin.

## Sorumlu Yapay Zeka ile Hata Ayıklama

Bir yazılım uygulamasını hata ayıklamak gibi, bir yapay zeka sistemini hata ayıklamak, sistemdeki sorunları belirleme ve çözme sürecidir. Bir modelin beklenildiği gibi veya sorumlu bir şekilde performans göstermemesine neden olan birçok faktör vardır. Çoğu geleneksel model performans metriği, bir modelin performansının nicel toplamlarıdır ve bir modelin sorumlu yapay zeka ilkelerini nasıl ihlal ettiğini analiz etmek için yeterli değildir. Ayrıca, bir makine öğrenimi modeli, sonuçlarını neyin yönlendirdiğini anlamayı veya hata yaptığında açıklama sağlamayı zorlaştıran bir kara kutudur. Bu kursta daha sonra, yapay zeka sistemlerini hata ayıklamak için Sorumlu Yapay Zeka panosunu nasıl kullanacağımızı öğreneceğiz. Pano, veri bilimciler ve yapay zeka geliştiriciler için şu işlemleri gerçekleştirmek üzere kapsamlı bir araç sağlar:

* **Hata analizi**. Modelin adalet veya güvenilirliği etkileyebilecek hata dağılımını belirlemek.
* **Model genel görünümü**. Modelin performansında veri grupları arasında nerede farklılıklar olduğunu keşfetmek.
* **Veri analizi**. Veri dağılımını anlamak ve adalet, kapsayıcılık ve güvenilirlik sorunlarına yol açabilecek olası önyargıları belirlemek.
* **Model yorumlanabilirliği**. Modelin tahminlerini neyin etkilediğini veya yönlendirdiğini anlamak. Bu, modelin davranışını açıklamak için önemlidir ve şeffaflık ve hesap verebilirlik açısından kritiktir.

## 🚀 Zorluk

Zararların baştan önlenmesi için şunları yapmalıyız:

- sistemler üzerinde çalışan insanlar arasında farklı geçmişlere ve bakış açılarına sahip olmak
- toplumumuzun çeşitliliğini yansıtan veri setlerine yatırım yapmak
- makine öğrenimi yaşam döngüsü boyunca sorumlu yapay zekayı tespit etmek ve düzeltmek için daha iyi yöntemler geliştirmek

Model oluşturma ve kullanımı sırasında bir modelin güvenilmezliğinin açıkça görüldüğü gerçek yaşam senaryolarını düşünün. Başka neleri dikkate almalıyız?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu derste, makine öğreniminde adalet ve adaletsizlik kavramlarının temellerini öğrendiniz.
Bu atölyeyi izleyerek konulara daha derinlemesine dalın: 

- Sorumlu yapay zeka arayışı: İlkeleri uygulamaya dökmek - Besmira Nushi, Mehrnoosh Sameki ve Amit Sharma tarafından

[![Sorumlu AI Araç Kutusu: Sorumlu yapay zeka oluşturmak için açık kaynaklı bir çerçeve](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Sorumlu yapay zeka oluşturmak için açık kaynaklı bir çerçeve")


> 🎥 Yukarıdaki görsele tıklayarak videoyu izleyin: RAI Toolbox: Sorumlu yapay zeka oluşturmak için açık kaynaklı bir çerçeve - Besmira Nushi, Mehrnoosh Sameki ve Amit Sharma tarafından

Ayrıca okuyun: 

- Microsoft’un Sorumlu AI kaynak merkezi: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Microsoft’un FATE araştırma grubu: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Araç Kutusu: 

- [Responsible AI Toolbox GitHub deposu](https://github.com/microsoft/responsible-ai-toolbox)

Azure Machine Learning'in adalet sağlama araçları hakkında okuyun:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Ödev

[RAI Toolbox’u Keşfedin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.