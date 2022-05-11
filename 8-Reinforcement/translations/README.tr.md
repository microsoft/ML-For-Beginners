# Pekiştirmeli Öğrenmeye Giriş

Pekiştirmeli öğrenme (reinforcement learning), RL, denetimli öğrenme ve denetimsiz öğrenme gibi temel makine öğrenmesi paradigmalarından biri olarak görülüyor. RL tamamen kararlar ile ilgilidir: doğru kararları verebilmek veya en azından onlardan öğrenmektir.

Simüle edilmiş bir ortamınız olduğunu hayal edin, borsa gibi. Belirli bir düzenlemeyi(regülasyon) uygularsanız ne olur? Pozitif mi negatif mi etki eder? Eğer negatif etki ettiyse bunu _negative reinforcement_ olarak almalı, bundan birşeyler öğrenmeli ve rotanızı buna göre değiştirmelisiniz. Eğer pozitif bir sonuç elde ederseniz,  _positive reinforcement_ olarak bunun üzerine birşeyler inşa etmelisiniz.

![peter and the wolf](../images/peter.png)

> Peter ve arkadaşı aç kurttan kaçmalı! Image by [Jen Looper](https://twitter.com/jenlooper)
## Bölgesel Konu: Peter ve Kurt (Rusya)

[Peter ve Kurt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf), Rus bir besteci[Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev)  tarafından yazılmış bir müzikal peri masalıdır. Kurdu kovalamak için evinden cesurca ormana giden genç Peter hakkında bir hikaye. Bu bölümde Peter'a yardımcı olacak makine öğrenmesi algoritmaları eğiteceğiz:

- Çevredeki alanı **keşfedin** ve yol gösterici harita oluşturun.
- Daha hızlı hareket etmek için kaykay kullanmayı ve üzerinde dengede durmayı **öğrenin**.

[![Peter and the Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Prokofiev'in Peter ve Kurt şarkısını dinlemek için yukarıdaki resme tıklayın.
## Pekiştirmeli Öğrenme

Bir önceki bölümde, iki çeşit makine öğrenmesi problemi örneğini gördünüz:

- **Denetimli**, çözmek istediğimiz soruna örnek çözümler öneren veri kümelerimiz var. [Sınıflandırma(Classification)](../../4-Classification/README.md) ve [regresyon(regression)](../2-Regression/README.md) denetimli öğrenme görevlerindendir.
- **Denetimsiz**, etiketli eğitim verisine sahip değiliz. Denetimsiz öğrenmenin başlıca örneği [kümelemedir(Clustering)](../../5-Clustering/README.md).

Bu bölümde, etiketlenmiş eğitim verileri ihtiyaç duymayan yeni bir öğrenme problemi türünü size tanıtacağız. Bu tür problemlerin birkaç türü vardır:

- **[Yarı-denetimli öğrenme](https://wikipedia.org/wiki/Semi-supervised_learning)**, modeli önceden eğitmek için kullanılabilecek çok sayıda etiketlenmemiş veriye sahip olduğumuz yer.
- **[Pekiştirmeli öğrenme](https://wikipedia.org/wiki/Reinforcement_learning)**, bir ajanın(agent, öğrenme işini yapacak olan), simüle edilmiş bir ortamda denemeler yaparak nasıl davranacağını öğrendiği.

### Örnek - bilgisayar oyunu

Bir bilgisayara satranç gibi bir oyun oynamayı öğretmek istediğinizi varsayalım, veya [Super Mario](https://wikipedia.org/wiki/Super_Mario). 
Bir bilgisayarın bir oyunu oynaması için, oyun durumlarının her birinde hangi hamleyi yapacağını tahmin etmemiz gerekir. Bu bir sınıflandırma problemi gibi görünse de, değil - çünkü bu durumları ve karşılık gelen aksiyonları içeren bir veri kümemiz yok. Mevcut satranç maçları veya Super Mario oynayan oyuncuların kayıtları gibi bazı verilere sahip olabiliriz, bu verilerin yeterince büyük sayıda olmaması veya olası durumları yeterince kapsamaması muhtemeldir.

**Pekiştirmeli öğrenme** (RL) mevcut oyun verilerini aramak yerine, *bilgisayarın defalarca oynamasını sağlama* ve sonucu gözlemleme fikrine dayanır. Bu nedenle **pekiştirmeli öğrenmeyi** uygulamak için iki şeye ihtiyacımız var:

- **Bir ortam** ve **bir similatör** birçok kez oyun oynamamıza imkan verecektir. Bu simülatör, olası tüm durumlar ve tüm eylemlerin yanı sıra tüm oyun kurallarını tanımlayacaktır.

- **Bir ödül fonksiyonu**, bu bize her harekette veya oyunda ne kadar iyi ilerlediğimizi söyleyecektir.

Diğer makine öğrenimi türleri ile RL arasındaki temel fark, RL'de oyunu bitirene kadar kazanıp kazanmadığımızı genellikle bilemememizdir. Bu nedenle, belirli bir hareketin tek başına iyi olup olmadığını söyleyemeyiz - sadece oyunun sonunda bir ödül alırız. Ve hedefimiz ise belirsiz koşullar altında bir modeli eğitmemizi sağlayacak algoritmalar tasarlamak. **Q-learning** adında ki bir RL algoritmasını öğreneceğiz.

## Dersler

1. [Introduction to reinforcement learning and Q-Learning](../1-QLearning/README.md)
2. [Using a gym simulation environment](../2-Gym/README.md)

## Katkıda bulunanlar

  "Pekiştirmeli Öğrenmeye Giriş" ♥️ [Dmitry Soshnikov](http://soshnikov.com) tarafından yazıldı.
