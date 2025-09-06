<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-06T07:56:10+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "tr"
}
-->
# Makine Öğrenimine Giriş

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

---

[![Yeni Başlayanlar için Makine Öğrenimine Giriş](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "Yeni Başlayanlar için Makine Öğrenimine Giriş")

> 🎥 Yukarıdaki görsele tıklayarak bu dersi anlatan kısa bir videoyu izleyebilirsiniz.

Yeni başlayanlar için klasik makine öğrenimi üzerine hazırlanan bu kursa hoş geldiniz! Bu konuya tamamen yabancı olsanız da, belirli bir alanda bilgilerinizi tazelemek isteyen deneyimli bir ML uygulayıcısı olsanız da, bizimle olduğunuz için mutluyuz! Makine öğrenimi çalışmalarınıza dostane bir başlangıç noktası oluşturmayı hedefliyoruz ve [geri bildirimlerinizi](https://github.com/microsoft/ML-For-Beginners/discussions) değerlendirmek, yanıtlamak ve dahil etmekten memnuniyet duyarız.

[![Makine Öğrenimine Giriş](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Makine Öğrenimine Giriş")

> 🎥 Yukarıdaki görsele tıklayarak MIT'den John Guttag'ın makine öğrenimini tanıttığı videoyu izleyebilirsiniz.

---
## Makine Öğrenimine Başlarken

Bu müfredata başlamadan önce, bilgisayarınızı yerel olarak not defterlerini çalıştırmaya hazır hale getirmeniz gerekiyor.

- **Bilgisayarınızı bu videolarla yapılandırın**. Sisteminizde [Python'u nasıl kuracağınızı](https://youtu.be/CXZYvNRIAKM) ve geliştirme için bir [metin düzenleyiciyi nasıl ayarlayacağınızı](https://youtu.be/EU8eayHWoZg) öğrenmek için aşağıdaki bağlantıları kullanın.
- **Python öğrenin**. Veri bilimciler için faydalı bir programlama dili olan [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott) hakkında temel bir anlayışa sahip olmanız önerilir. Bu kursta Python kullanacağız.
- **Node.js ve JavaScript öğrenin**. Bu kursta web uygulamaları oluştururken birkaç kez JavaScript kullanacağız, bu nedenle [node](https://nodejs.org) ve [npm](https://www.npmjs.com/) kurulu olmalı ve hem Python hem de JavaScript geliştirme için [Visual Studio Code](https://code.visualstudio.com/) kullanılabilir olmalıdır.
- **GitHub hesabı oluşturun**. Bizi burada [GitHub](https://github.com) üzerinde bulduğunuza göre, muhtemelen bir hesabınız vardır, ancak yoksa bir hesap oluşturun ve ardından bu müfredatı kendi kullanımınız için çatallayın. (Bize bir yıldız vermekten çekinmeyin 😊)
- **Scikit-learn'ü keşfedin**. Bu derslerde referans verdiğimiz bir dizi ML kütüphanesi olan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) ile tanışın.

---
## Makine Öğrenimi Nedir?

'Makine öğrenimi' terimi, günümüzün en popüler ve sık kullanılan terimlerinden biridir. Teknolojiyle bir şekilde tanışıklığınız varsa, hangi alanda çalışıyor olursanız olun, bu terimi en az bir kez duymuş olma olasılığınız oldukça yüksektir. Ancak, makine öğreniminin mekanikleri çoğu insan için bir muammadır. Makine öğrenimine yeni başlayan biri için konu bazen bunaltıcı gelebilir. Bu nedenle, makine öğreniminin gerçekte ne olduğunu anlamak ve pratik örneklerle adım adım öğrenmek önemlidir.

---
## Hype Eğrisi

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends, 'makine öğrenimi' teriminin son zamanlardaki 'hype eğrisini' gösteriyor.

---
## Gizemli Bir Evren

Büyüleyici gizemlerle dolu bir evrende yaşıyoruz. Stephen Hawking, Albert Einstein ve daha birçok büyük bilim insanı, çevremizdeki dünyanın gizemlerini ortaya çıkaran anlamlı bilgileri aramaya hayatlarını adamışlardır. Bu, öğrenmenin insan doğasıdır: Bir insan çocuğu, büyüdükçe her yıl yeni şeyler öğrenir ve dünyasının yapısını keşfeder.

---
## Çocuğun Beyni

Bir çocuğun beyni ve duyuları, çevresindeki gerçekleri algılar ve yaşamın gizli kalıplarını öğrenerek çocuğun öğrendiği kalıpları tanımlamak için mantıksal kurallar oluşturmasına yardımcı olur. İnsan beyninin öğrenme süreci, insanları bu dünyanın en sofistike canlıları yapar. Gizli kalıpları keşfederek sürekli öğrenmek ve ardından bu kalıplar üzerinde yenilik yapmak, yaşamımız boyunca kendimizi daha iyi hale getirmemizi sağlar. Bu öğrenme kapasitesi ve evrimleşme yeteneği, [beyin plastisitesi](https://www.simplypsychology.org/brain-plasticity.html) adı verilen bir kavramla ilişkilidir. Yüzeysel olarak, insan beyninin öğrenme süreci ile makine öğrenimi kavramları arasında bazı motive edici benzerlikler çizebiliriz.

---
## İnsan Beyni

[İnsan beyni](https://www.livescience.com/29365-human-brain.html), gerçek dünyadan şeyleri algılar, algılanan bilgiyi işler, mantıklı kararlar alır ve koşullara bağlı olarak belirli eylemleri gerçekleştirir. Buna zeki davranış denir. Zeki davranış sürecinin bir benzerini bir makineye programladığımızda, buna yapay zeka (AI) denir.

---
## Bazı Terimler

Terimler karıştırılabilir olsa da, makine öğrenimi (ML), yapay zekanın önemli bir alt kümesidir. **ML, algılanan verilerden anlamlı bilgiler ortaya çıkarmak ve gizli kalıpları bulmak için özel algoritmalar kullanarak mantıklı karar verme sürecini desteklemekle ilgilidir**.

---
## AI, ML, Derin Öğrenme

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> AI, ML, derin öğrenme ve veri bilimi arasındaki ilişkileri gösteren bir diyagram. [Jen Looper](https://twitter.com/jenlooper) tarafından hazırlanmış, [bu grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining) ilham alınarak oluşturulmuştur.

---
## Kapsanacak Kavramlar

Bu müfredatta, bir başlangıç seviyesindeki kişinin bilmesi gereken yalnızca temel makine öğrenimi kavramlarını ele alacağız. 'Klasik makine öğrenimi' dediğimiz konuları, birçok öğrencinin temelleri öğrenmek için kullandığı mükemmel bir kütüphane olan Scikit-learn kullanarak ele alacağız. Yapay zeka veya derin öğrenmenin daha geniş kavramlarını anlamak için, makine öğreniminin güçlü bir temel bilgisine sahip olmak gereklidir ve bunu burada sunmak istiyoruz.

---
## Bu Kursta Öğrenecekleriniz:

- makine öğreniminin temel kavramları
- ML'nin tarihi
- ML ve adalet
- regresyon ML teknikleri
- sınıflandırma ML teknikleri
- kümeleme ML teknikleri
- doğal dil işleme ML teknikleri
- zaman serisi tahmini ML teknikleri
- pekiştirmeli öğrenme
- ML'nin gerçek dünya uygulamaları

---
## Kapsamayacaklarımız

- derin öğrenme
- sinir ağları
- yapay zeka

Daha iyi bir öğrenme deneyimi sağlamak için, sinir ağlarının karmaşıklıklarından, 'derin öğrenme' - sinir ağlarını kullanarak çok katmanlı model oluşturma - ve yapay zekadan kaçınacağız. Bunları farklı bir müfredatta ele alacağız. Ayrıca, bu daha geniş alanın bir yönüne odaklanmak için yakında bir veri bilimi müfredatı sunacağız.

---
## Neden Makine Öğrenimi Çalışmalıyız?

Sistemler perspektifinden bakıldığında, makine öğrenimi, verilerden gizli kalıpları öğrenebilen ve akıllı kararlar almaya yardımcı olan otomatik sistemlerin oluşturulması olarak tanımlanır.

Bu motivasyon, insan beyninin dış dünyadan algıladığı verilere dayanarak belirli şeyleri nasıl öğrendiğinden gevşek bir şekilde ilham almıştır.

✅ Bir işletmenin neden sabit kodlanmış kurallara dayalı bir motor oluşturmak yerine makine öğrenimi stratejilerini kullanmayı tercih edebileceğini bir dakika düşünün.

---
## Makine Öğreniminin Uygulamaları

Makine öğreniminin uygulamaları artık neredeyse her yerde ve toplumlarımızda dolaşan, akıllı telefonlarımız, bağlı cihazlarımız ve diğer sistemler tarafından üretilen veriler kadar yaygındır. Son teknoloji makine öğrenimi algoritmalarının muazzam potansiyelini göz önünde bulundurarak, araştırmacılar, çok boyutlu ve çok disiplinli gerçek yaşam problemlerini büyük olumlu sonuçlarla çözme yeteneklerini keşfetmektedir.

---
## Uygulamalı ML Örnekleri

**Makine öğrenimini birçok şekilde kullanabilirsiniz**:

- Bir hastanın tıbbi geçmişinden veya raporlarından hastalık olasılığını tahmin etmek için.
- Hava durumu verilerini kullanarak hava olaylarını tahmin etmek için.
- Bir metnin duygusunu anlamak için.
- Propagandanın yayılmasını durdurmak için sahte haberleri tespit etmek için.

Finans, ekonomi, yer bilimi, uzay keşfi, biyomedikal mühendislik, bilişsel bilim ve hatta beşeri bilimler gibi alanlar, kendi alanlarındaki zorlu, veri işleme ağırlıklı problemleri çözmek için makine öğrenimini benimsemiştir.

---
## Sonuç

Makine öğrenimi, gerçek dünya veya üretilmiş verilerden anlamlı içgörüler bularak kalıp keşfetme sürecini otomatikleştirir. İş, sağlık ve finansal uygulamalar gibi birçok alanda son derece değerli olduğunu kanıtlamıştır.

Yakın gelecekte, makine öğreniminin temellerini anlamak, yaygın benimsenmesi nedeniyle herhangi bir alandan insanlar için bir zorunluluk haline gelecektir.

---
# 🚀 Zorluk

AI, ML, derin öğrenme ve veri bilimi arasındaki farkları kağıt üzerinde veya [Excalidraw](https://excalidraw.com/) gibi çevrimiçi bir uygulama kullanarak çizin. Bu tekniklerin her birinin çözmekte iyi olduğu problemlerle ilgili bazı fikirler ekleyin.

# [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

---
# Gözden Geçirme ve Kendi Kendine Çalışma

ML algoritmalarıyla bulutta nasıl çalışabileceğinizi öğrenmek için bu [Öğrenme Yolu](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott) bağlantısını takip edin.

ML'nin temelleri hakkında bilgi edinmek için bir [Öğrenme Yolu](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) alın.

---
# Ödev

[Başlamak için buraya tıklayın](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çeviriler hata veya yanlışlıklar içerebilir. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlama durumunda sorumluluk kabul edilmez.