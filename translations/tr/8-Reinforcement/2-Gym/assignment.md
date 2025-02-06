# Dağ Arabası Eğitimi

[OpenAI Gym](http://gym.openai.com), tüm ortamların aynı API'yi sağlaması için tasarlanmıştır - yani aynı `reset`, `step` ve `render` yöntemleri ve **eylem alanı** ve **gözlem alanı**nın aynı soyutlamaları. Bu nedenle, aynı pekiştirmeli öğrenme algoritmalarını, minimum kod değişiklikleri ile farklı ortamlara uyarlamak mümkün olmalıdır.

## Bir Dağ Arabası Ortamı

[Dağ Arabası ortamı](https://gym.openai.com/envs/MountainCar-v0/), bir vadide sıkışmış bir araba içerir:
Amacınız vadiden çıkmak ve bayrağı ele geçirmek, her adımda aşağıdaki eylemlerden birini yaparak:

| Değer | Anlam |
|---|---|
| 0 | Sola hızlan |
| 1 | Hızlanma |
| 2 | Sağa hızlan |

Bu problemin ana püf noktası, arabanın motorunun dağı tek bir geçişte ölçeklendirmek için yeterince güçlü olmamasıdır. Bu nedenle, başarılı olmanın tek yolu, momentum kazanmak için ileri geri gitmektir.

Gözlem alanı sadece iki değerden oluşur:

| Num | Gözlem  | Min | Max |
|-----|---------|-----|-----|
|  0  | Araba Pozisyonu | -1.2| 0.6 |
|  1  | Araba Hızı | -0.07 | 0.07 |

Dağ arabası için ödül sistemi oldukça zordur:

 * Dağın tepesindeki bayrağa (pozisyon = 0.5) ulaşan ajana 0 ödülü verilir.
 * Ajanın pozisyonu 0.5'ten azsa -1 ödülü verilir.

Eğer araba pozisyonu 0.5'ten fazla olursa veya bölüm uzunluğu 200'den fazla olursa bölüm sona erer.
## Talimatlar

Dağ arabası problemini çözmek için pekiştirmeli öğrenme algoritmamızı uyarlayın. Mevcut [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) koduyla başlayın, yeni ortamı ekleyin, durum ayrıştırma fonksiyonlarını değiştirin ve mevcut algoritmanın minimum kod değişiklikleri ile eğitilmesini sağlamaya çalışın. Sonucu hiperparametreleri ayarlayarak optimize edin.

> **Not**: Algoritmanın yakınsamasını sağlamak için hiperparametre ayarlamaları gerekebilir. 
## Değerlendirme Kriterleri

| Kriter | Örnek | Yeterli | Geliştirme Gerekiyor |
|--------|-------|---------|----------------------|
|        | Q-Öğrenme algoritması, minimum kod değişiklikleri ile CartPole örneğinden başarıyla uyarlanmış ve 200 adımın altında bayrağı ele geçirme problemini çözebilecek şekilde çalışıyor. | İnternetten yeni bir Q-Öğrenme algoritması uyarlanmış, ancak iyi belgelenmiş; veya mevcut algoritma uyarlanmış, ancak istenen sonuçlara ulaşamıyor | Öğrenci herhangi bir algoritmayı başarıyla uyarlayamamış, ancak çözüme doğru önemli adımlar atmış (durum ayrıştırma, Q-Tablo veri yapısı, vb. uygulanmış) |

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belgenin kendi dilindeki versiyonu yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.