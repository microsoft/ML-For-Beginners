<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-06T08:02:46+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "tr"
}
-->
# Pekiştirmeli Öğrenmeye Giriş

Pekiştirmeli öğrenme, RL, denetimli öğrenme ve denetimsiz öğrenmenin yanında temel makine öğrenimi paradigmalarından biri olarak görülür. RL tamamen kararlarla ilgilidir: doğru kararlar vermek veya en azından onlardan öğrenmek.

Hayal edin ki borsa gibi simüle edilmiş bir ortamınız var. Belirli bir düzenleme uygularsanız ne olur? Bunun olumlu mu yoksa olumsuz bir etkisi mi olur? Eğer olumsuz bir şey olursa, bu _olumsuz pekiştirmeyi_ almanız, bundan öğrenmeniz ve rotanızı değiştirmeniz gerekir. Eğer olumlu bir sonuç olursa, bu _olumlu pekiştirme_ üzerine inşa etmeniz gerekir.

![peter ve kurt](../../../8-Reinforcement/images/peter.png)

> Peter ve arkadaşları aç kurtlardan kaçmak zorunda! Görsel: [Jen Looper](https://twitter.com/jenlooper)

## Bölgesel Konu: Peter ve Kurt (Rusya)

[Peter ve Kurt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf), Rus besteci [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) tarafından yazılmış bir müzikal masaldır. Bu hikaye, genç öncü Peter'ın cesurca evinden çıkıp kurtu kovalamak için orman açıklığına gitmesini anlatır. Bu bölümde, Peter'a yardımcı olacak makine öğrenimi algoritmalarını eğiteceğiz:

- **Keşfetmek**: Çevredeki alanı keşfetmek ve en uygun navigasyon haritasını oluşturmak.
- **Öğrenmek**: Daha hızlı hareket edebilmek için kaykay kullanmayı ve dengede durmayı öğrenmek.

[![Peter ve Kurt](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Prokofiev'in Peter ve Kurt eserini dinlemek için yukarıdaki görsele tıklayın.

## Pekiştirmeli Öğrenme

Önceki bölümlerde, iki tür makine öğrenimi problemini gördünüz:

- **Denetimli**, burada çözmek istediğimiz probleme örnek çözümler sunan veri setlerimiz var. [Sınıflandırma](../4-Classification/README.md) ve [regresyon](../2-Regression/README.md) denetimli öğrenme görevleridir.
- **Denetimsiz**, burada etiketlenmiş eğitim verilerimiz yok. Denetimsiz öğrenmenin ana örneği [Kümeleme](../5-Clustering/README.md)'dir.

Bu bölümde, etiketlenmiş eğitim verisi gerektirmeyen yeni bir öğrenme problem türüyle tanışacaksınız. Bu tür problemlerin birkaç çeşidi vardır:

- **[Yarı denetimli öğrenme](https://wikipedia.org/wiki/Semi-supervised_learning)**, burada modeli önceden eğitmek için kullanılabilecek çok miktarda etiketlenmemiş veri bulunur.
- **[Pekiştirmeli öğrenme](https://wikipedia.org/wiki/Reinforcement_learning)**, burada bir ajan, bazı simüle edilmiş ortamlarda deneyler yaparak nasıl davranması gerektiğini öğrenir.

### Örnek - Bilgisayar Oyunu

Diyelim ki bir bilgisayara bir oyun oynamayı öğretmek istiyorsunuz, örneğin satranç veya [Super Mario](https://wikipedia.org/wiki/Super_Mario). Bilgisayarın bir oyun oynaması için, her oyun durumunda hangi hamleyi yapacağını tahmin etmesi gerekir. Bu bir sınıflandırma problemi gibi görünebilir, ancak öyle değildir - çünkü elimizde durumlar ve karşılık gelen eylemlerle ilgili bir veri seti yoktur. Satranç maçları veya oyuncuların Super Mario oynarkenki kayıtları gibi bazı verilerimiz olabilir, ancak bu veriler muhtemelen yeterince geniş bir durum yelpazesini kapsamayacaktır.

Mevcut oyun verilerini aramak yerine, **Pekiştirmeli Öğrenme** (RL), bilgisayarı *birçok kez oyun oynatmak* ve sonucu gözlemlemek fikrine dayanır. Bu nedenle, Pekiştirmeli Öğrenmeyi uygulamak için iki şeye ihtiyacımız var:

- **Bir ortam** ve **bir simülatör**, bu simülatör bize oyunu birçok kez oynama imkanı sağlar. Bu simülatör, tüm oyun kurallarını, olası durumları ve eylemleri tanımlar.

- **Bir ödül fonksiyonu**, bu fonksiyon her hamle veya oyun sırasında ne kadar iyi performans gösterdiğimizi bize söyler.

Diğer makine öğrenimi türleri ile RL arasındaki temel fark, RL'de genellikle oyunu bitirene kadar kazanıp kazanmadığımızı bilmememizdir. Bu nedenle, belirli bir hamlenin tek başına iyi olup olmadığını söyleyemeyiz - ödülü yalnızca oyunun sonunda alırız. Amacımız, belirsiz koşullar altında bir modeli eğitmemizi sağlayacak algoritmalar tasarlamaktır. **Q-learning** adlı bir RL algoritmasını öğreneceğiz.

## Dersler

1. [Pekiştirmeli öğrenmeye ve Q-Learning'e giriş](1-QLearning/README.md)
2. [Bir gym simülasyon ortamı kullanma](2-Gym/README.md)

## Katkıda Bulunanlar

"Pekiştirmeli Öğrenmeye Giriş" [Dmitry Soshnikov](http://soshnikov.com) tarafından ♥️ ile yazılmıştır.

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.