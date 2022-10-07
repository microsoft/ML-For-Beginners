# Makine Öğrenimine Giriş

[![ML, AI, Derin öğrenme - Farkları nelerdir?](https://img.youtube.com/vi/lTd9RSxS9ZE/0.jpg)](https://youtu.be/lTd9RSxS9ZE "ML, AI, Derin öğrenme - Farkları nelerdir?")

> 🎥  Makine öğrenimi, yapay zeka ve derin öğrenme arasındaki farkı tartışan bir video için yukarıdaki resme tıklayın.

## [Ders öncesi sınav](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=tr)

### Introduction

Yeni başlayanlar için klasik makine öğrenimi üzerine olan bu kursa hoş geldiniz! İster bu konuda tamamen yeni olun, ister belli bir alandaki bilgilerini tazelemek isteyen deneyimli bir makine öğrenimi uygulayıcısı olun, aramıza katılmanızdan mutluluk duyarız! Makine öğrenimi çalışmanız için samimi bir başlangıç ​​noktası oluşturmak istiyoruz ve [geri bildiriminizi](https://github.com/microsoft/ML-For-Beginners/discussions) değerlendirmekten, yanıtlamaktan ve hayata geçirmekten memnuniyet duyarız.

[![Makine Öğrenimine Giriş](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Makine Öğrenimine Giriş")

> 🎥 Video için yukarıdaki resme tıklayın: MIT'den John Guttag, makine öğrenimini tanıtıyor
### Makine Öğrenimine Başlamak

Bu müfredata başlamadan önce, bilgisayarınızın yerel olarak (Jupyter) not defterlerini çalıştırmak için hazır olması gerekir.

- **Makinenizi bu videolar rehberliğinde yapılandırın**. Bu [video setinde](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6) makinenizi nasıl kuracağınız hakkında daha fazla bilgi edinin.
- **Python öğrenin**. Ayrıca, veri bilimciler için faydalı bir programlama dili olan ve bu derslerde kullandığımız [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) programlama dili hakkında temel bilgilere sahip olmanız da önerilir.
- **Node.js ve JavaScript'i öğrenin**. Web uygulamaları oluştururken de bu kursta JavaScript'i birkaç kez kullanıyoruz, bu nedenle [node](https://nodejs.org), [npm](https://www.npmjs.com/) ve ayrıca hem Python hem de JavaScript geliştirme için kullanılabilen [Visual Studio Code](https://code.visualstudio.com/) yüklü olmalıdır.
- **GitHub hesabı oluşturun**. Bizi burada [GitHub](https://github.com) üzerinde bulduğunuza göre, zaten bir hesabınız olabilir, ancak mevcut değilse, bir tane hesap oluşturun ve ardından bu müfredatı kendi başınıza kullanmak için çatallayın (fork). (Bize de yıldız vermekten çekinmeyin 😊)
- **Scikit-learn'ü keşfedin**. Bu derslerde referans verdiğimiz, bir dizi ML kütüphanesinden oluşan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) hakkında bilgi edinin.

### Makine öğrenimi nedir?

'Makine öğrenimi' terimi, günümüzün en popüler ve sık kullanılan terimlerinden biridir. Hangi alanda çalışırsanız çalışın, teknolojiyle ilgili bir tür aşinalığınız varsa, bu terimi en az bir kez duymuş olma ihtimaliniz yüksektir. Bununla birlikte, makine öğreniminin mekanikleri, yani çalışma prensipleri, çoğu insan için bir gizemdir. Makine öğrenimine yeni başlayan biri için konu bazen bunaltıcı gelebilir. Bu nedenle, makine öğreniminin gerçekte ne olduğunu anlamak ve pratik örnekler üzerinden adım adım öğrenmek önemlidir.

![ML heyecan eğrisi](../images/hype.png)

> Google Trendler, 'makine öğrenimi' teriminin son 'heyecan eğrisini' gösteriyor

Büyüleyici gizemlerle dolu bir evrende yaşıyoruz. Stephen Hawking, Albert Einstein ve daha pek çoğu gibi büyük bilim adamları, hayatlarını çevremizdeki dünyanın gizemlerini ortaya çıkaran anlamlı bilgiler aramaya adadılar. Öğrenmenin insani yönü de budur: insan evladı yeni şeyler öğrenir ve yetişkinliğe doğru büyüdükçe her yıl kendi dünyasının yapısını ortaya çıkarır.

Bir çocuğun beyni ve duyuları, çevrelerindeki gerçekleri algılar ve çocuğun, öğrenilen kalıpları tanımlamak için mantıksal kurallar oluşturmasına yardımcı olan gizli yaşam kalıplarını yavaş yavaş öğrenir. İnsan beyninin öğrenme süreci, insanı bu dünyanın en gelişmiş canlısı yapar. Gizli kalıpları keşfederek sürekli öğrenmek ve sonra bu kalıplar üzerinde yenilik yapmak, yaşamımız boyunca kendimizi giderek daha iyi hale getirmemizi sağlar. Bu öğrenme kapasitesi ve gelişen kabiliyet, [beyin plastisitesi](https://www.simplypsychology.org/brain-plasticity.html) adı verilen bir kavramla ilgilidir. Yüzeysel olarak, insan beyninin öğrenme süreci ile makine öğrenimi kavramları arasında bazı motivasyonel benzerlikler çizebiliriz.

[İnsan beyni](https://www.livescience.com/29365-human-brain.html) gerçek dünyadaki şeyleri algılar, algılanan bilgileri işler, mantıksal kararlar verir ve koşullara göre belirli eylemler gerçekleştirir. Akıllıca davranmak dediğimiz şey buydu işte. Bir makineye akıllı davranış sürecinin bir kopyasını programladığımızda buna yapay zeka (İngilizce haliyle artificial intelligence, kısaca **AI**) denir.

Terimler karıştırılabilse de, makine öğrenimi (İngilizce haliyle machine learning, kısaca **ML**), yapay zekanın önemli bir alt kümesidir. **ML, mantıklı karar verme sürecini desteklemek için anlamlı bilgileri ortaya çıkarmak ve algılanan verilerden gizli kalıpları bulmak için özel algoritmalar kullanmakla ilgilenir**.

![AI, ML, derin öğrenme, veri bilimi](../images/ai-ml-ds.png)

> Yapay zeka, makine öğrenimi, derin öğrenme ve veri bilimi arasındaki ilişkileri gösteren bir diyagram. Bu infografik, [şu grafikten](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-) ilham alan [Jen Looper](https://twitter.com/jenlooper) tarafından hazırlanmıştır.

> AI (Artificial Intelligence): Yapay zekâ  
> ML(Machine Learning): Makine öğrenimi  
> Deep Learning: Derin Öğrenme  
> Data Science: Veri bilimi  

## Bu kursta neler öğreneceksiniz

Bu müfredatta, yalnızca yeni başlayanların bilmesi gereken makine öğreniminin temel kavramlarını ele alacağız. 'Klasik makine öğrenimi' dediğimiz şeyi, öncelikle birçok öğrencinin temel bilgileri öğrenmek için kullandığı mükemmel bir kütüphane olan Scikit-learn'ü kullanarak ele alıyoruz. Daha geniş yapay zeka veya derin öğrenme kavramlarını anlamak için, güçlü bir temel makine öğrenimi bilgisi vazgeçilmezdir ve bu yüzden onu burada sunmak istiyoruz.

Bu kursta şunları öğreneceksiniz:

- makine öğreniminin temel kavramları
- ML'nin tarihi
- ML ve adillik
- regresyon ML teknikleri
- sınıflandırma ML teknikleri
- kümeleme ML teknikleri
- doğal dil işleme ML teknikleri
- zaman serisi tahmini ML teknikleri
- pekiştirmeli öğrenme
- ML için gerçek-dünya uygulamaları

## Neyi kapsamayacağız

- derin öğrenme
- sinir ağları
- yapay zeka
  
Daha iyi bir öğrenme deneyimi sağlamak için, farklı bir müfredatta tartışacağımız sinir ağları, 'derin öğrenme' (sinir ağlarını kullanarak çok katmanlı modeller oluşturma) ve yapay zekânın karmaşıklıklarından kaçınacağız. Ayrıca, bu daha geniş alanın bu yönüne odaklanmak için yakında çıkacak bir veri bilimi müfredatı sunacağız.

## Neden makine öğrenimi üzerinde çalışmalısınız?

Sistemler perspektifinden makine öğrenimi, akıllı kararlar almaya yardımcı olmak için verilerden gizli kalıpları öğrenebilen otomatik sistemlerin oluşturulması olarak tanımlanır.

Bu motivasyon, insan beyninin dış dünyadan algıladığı verilere dayanarak belirli şeyleri nasıl öğrendiğinden bir miktar esinlenmiştir.

✅ Bir işletmenin, sabit kurallara dayalı bir karar aracı oluşturmak yerine neden makine öğrenimi stratejilerini kullanmayı denemek isteyebileceklerini bir an için düşünün.

### Makine öğrenimi uygulamaları

Makine öğrenimi uygulamaları artık neredeyse her yerde ve akıllı telefonlarımız, internete bağlı cihazlarımız ve diğer sistemlerimiz tarafından üretilen, toplumlarımızda akan veriler kadar yaygın hale gelmiş durumda. Son teknoloji makine öğrenimi algoritmalarının muazzam potansiyelini göz önünde bulunduran araştırmacılar, bu algoritmaların çok boyutlu ve çok disiplinli gerçek hayat problemlerini çözme yeteneklerini araştırıyorlar ve oldukça olumlu sonuçlar alıyorlar.

**Makine öğrenimini birçok şekilde kullanabilirsiniz**:

- Bir hastanın tıbbi geçmişinden veya raporlarından hastalık olasılığını tahmin etmek
- Hava olaylarını tahmin etmek için hava durumu verilerini kullanmak
- Bir metnin duygu durumunu anlamak
- Propagandanın yayılmasını durdurmak için sahte haberleri tespit etmek

Finans, ekonomi, yer bilimi, uzay araştırmaları, biyomedikal mühendislik, bilişsel bilim ve hatta beşeri bilimlerdeki alanlar, kendi alanlarının zorlu ve ağır veri işleme sorunlarını çözmek için makine öğrenimini tekniklerini kullanmaya başladılar.

Makine öğrenimi, gerçek dünyadan veya oluşturulan verilerden anlamlı içgörüler bularak örüntü bulma sürecini otomatikleştirir. Diğerlerinin yanı sıra iş, sağlık ve finansal uygulamalarda son derece değerli olduğunu kanıtlamıştır.

Yakın gelecekte, yaygın olarak benimsenmesi nedeniyle makine öğreniminin temellerini anlamak, tüm alanlardan insanlar için bir zorunluluk olacak.

---
## 🚀 Meydan Okuma

Kağıt üzerinde veya [Excalidraw](https://excalidraw.com/) gibi çevrimiçi bir uygulama kullanarak AI, makine öğrenimi, derin öğrenme ve veri bilimi arasındaki farkları anladığınızdan emin olun. Bu tekniklerin her birinin çözmede iyi olduğu bazı problem fikirleri ekleyin.

## [Ders sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2?loc=tr)

## İnceleme ve Bireysel Çalışma

Bulutta makine öğrenimi algoritmalarıyla nasıl çalışabileceğiniz hakkında daha fazla bilgi edinmek için bu [Eğitim Patikasını](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott) izleyin.

## Ödev

[Haydi başlayalım!](assignment.tr.md)