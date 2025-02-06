# Pekiştirmeli Öğrenmeye Giriş

Pekiştirmeli öğrenme, RL, denetimli öğrenme ve denetimsiz öğrenmenin yanında temel makine öğrenme paradigmalarından biri olarak görülür. RL, kararlarla ilgilidir: doğru kararları vermek veya en azından onlardan öğrenmek.

Bir simüle edilmiş ortamınız olduğunu hayal edin, örneğin borsa. Belirli bir düzenleme getirirseniz ne olur? Olumlu veya olumsuz bir etkisi var mı? Olumsuz bir şey olursa, bu _olumsuz pekiştirmeyi_ almalı, ondan öğrenmeli ve rotanızı değiştirmelisiniz. Eğer olumlu bir sonuç olursa, bu _olumlu pekiştirmeyi_ geliştirmelisiniz.

![peter ve kurt](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.tr.png)

> Peter ve arkadaşlarının aç kurttan kaçması gerekiyor! Görsel [Jen Looper](https://twitter.com/jenlooper) tarafından

## Bölgesel Konu: Peter ve Kurt (Rusya)

[Peter ve Kurt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf), Rus besteci [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev) tarafından yazılmış bir müzikli peri masalıdır. Bu, genç öncü Peter'in cesurca evinden çıkıp ormanda kurtu kovalamaya gittiği bir hikayedir. Bu bölümde, Peter'e yardımcı olacak makine öğrenme algoritmalarını eğiteceğiz:

- Çevreyi **keşfetmek** ve optimal bir navigasyon haritası oluşturmak
- Daha hızlı hareket edebilmek için kaykay kullanmayı ve üzerinde denge kurmayı **öğrenmek**.

[![Peter ve Kurt](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Peter ve Kurt'u dinlemek için yukarıdaki görsele tıklayın

## Pekiştirmeli Öğrenme

Önceki bölümlerde, iki tür makine öğrenme problemi örneği gördünüz:

- **Denetimli**, çözmek istediğimiz probleme örnek çözümler öneren veri kümelerimiz olduğunda. [Sınıflandırma](../4-Classification/README.md) ve [regresyon](../2-Regression/README.md) denetimli öğrenme görevleridir.
- **Denetimsiz**, etiketlenmiş eğitim verilerimizin olmadığı durumlarda. Denetimsiz öğrenmenin ana örneği [Kümeleme](../5-Clustering/README.md)'dir.

Bu bölümde, etiketlenmiş eğitim verileri gerektirmeyen yeni bir öğrenme problem türüyle tanışacaksınız. Bu tür problemlerin birkaç türü vardır:

- **[Yarı denetimli öğrenme](https://wikipedia.org/wiki/Semi-supervised_learning)**, çok sayıda etiketlenmemiş verinin modeli önceden eğitmek için kullanılabileceği durumlar.
- **[Pekiştirmeli öğrenme](https://wikipedia.org/wiki/Reinforcement_learning)**, bir ajanının simüle edilmiş bir ortamda deneyler yaparak nasıl davranacağını öğrendiği durumlar.

### Örnek - Bilgisayar Oyunu

Bir bilgisayara bir oyun, örneğin satranç veya [Super Mario](https://wikipedia.org/wiki/Super_Mario) oynamayı öğretmek istediğinizi varsayalım. Bilgisayarın oyun oynaması için, her oyun durumunda hangi hamleyi yapacağını tahmin etmesi gerekir. Bu bir sınıflandırma problemi gibi görünse de, değildir - çünkü durumlar ve karşılık gelen eylemlerle ilgili bir veri kümesine sahip değiliz. Mevcut satranç maçları veya Super Mario oynayan oyuncuların kayıtları gibi bazı verilere sahip olsak da, bu verilerin yeterince geniş bir durumu kapsamayacağı muhtemeldir.

Mevcut oyun verilerini aramak yerine, **Pekiştirmeli Öğrenme** (RL), *bilgisayarı birçok kez oynamaya ve sonucu gözlemlemeye* dayalıdır. Bu nedenle, Pekiştirmeli Öğrenmeyi uygulamak için iki şeye ihtiyacımız var:

- **Bir ortam** ve **bir simülatör**, bu da oyunu birçok kez oynamamıza izin verir. Bu simülatör, tüm oyun kurallarını, olası durumları ve eylemleri tanımlar.

- **Bir ödül fonksiyonu**, bu da her hamle veya oyun sırasında ne kadar iyi olduğumuzu bize söyler.

Diğer makine öğrenme türleri ile RL arasındaki temel fark, RL'de genellikle oyunu bitirene kadar kazanıp kazanmadığımızı bilmememizdir. Bu nedenle, belirli bir hamlenin tek başına iyi olup olmadığını söyleyemeyiz - sadece oyunun sonunda bir ödül alırız. Amacımız, belirsiz koşullar altında bir modeli eğitmemizi sağlayacak algoritmalar tasarlamaktır. **Q-learning** adı verilen bir RL algoritmasını öğreneceğiz.

## Dersler

1. [Pekiştirmeli öğrenme ve Q-Learning'e giriş](1-QLearning/README.md)
2. [Gym simülasyon ortamını kullanma](2-Gym/README.md)

## Katkıda Bulunanlar

"Pekiştirmeli Öğrenmeye Giriş" [Dmitry Soshnikov](http://soshnikov.com) tarafından ♥️ ile yazılmıştır.

**Feragatname**: 
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için, profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.