<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-06T08:05:23+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "tr"
}
-->
# Dağ Arabası Eğitimi

[OpenAI Gym](http://gym.openai.com), tüm ortamların aynı API'yi sağladığı şekilde tasarlanmıştır - yani aynı `reset`, `step` ve `render` yöntemleri ve **eylem alanı** ve **gözlem alanı** için aynı soyutlamalar. Bu nedenle, aynı pekiştirmeli öğrenme algoritmalarını farklı ortamlara minimum kod değişikliği ile uyarlamak mümkün olmalıdır.

## Bir Dağ Arabası Ortamı

[Dağ Arabası ortamı](https://gym.openai.com/envs/MountainCar-v0/), bir vadide sıkışmış bir arabayı içerir:

Amaç, vadiden çıkmak ve bayrağı yakalamaktır. Bunun için her adımda aşağıdaki eylemlerden birini gerçekleştirebilirsiniz:

| Değer | Anlam |
|---|---|
| 0 | Sola hızlan |
| 1 | Hızlanma |
| 2 | Sağa hızlan |

Bu problemin asıl püf noktası, arabanın motorunun dağı tek bir seferde tırmanacak kadar güçlü olmamasıdır. Bu nedenle, başarılı olmanın tek yolu ileri geri giderek momentum kazanmaktır.

Gözlem alanı sadece iki değerden oluşur:

| Num | Gözlem       | Min   | Max   |
|-----|--------------|-------|-------|
|  0  | Araba Pozisyonu | -1.2  | 0.6   |
|  1  | Araba Hızı       | -0.07 | 0.07  |

Dağ arabası için ödül sistemi oldukça karmaşıktır:

 * Ajan, dağın tepesindeki bayrağa ulaştığında (pozisyon = 0.5), 0 ödül kazanır.
 * Ajanın pozisyonu 0.5'ten küçükse, -1 ödül kazanır.

Bölüm, araba pozisyonu 0.5'ten büyük olduğunda veya bölüm uzunluğu 200'ü aştığında sona erer.

## Talimatlar

Pekiştirmeli öğrenme algoritmamızı dağ arabası problemini çözmek için uyarlayın. Mevcut [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) koduyla başlayın, yeni ortamı yerine koyun, durum ayrıklaştırma fonksiyonlarını değiştirin ve mevcut algoritmayı minimum kod değişikliği ile eğitmeye çalışın. Sonuçları hiperparametreleri ayarlayarak optimize edin.

> **Not**: Algoritmanın yakınsamasını sağlamak için hiperparametre ayarlamaları gerekebilir.

## Değerlendirme Kriterleri

| Kriter | Örnek | Yeterli | Geliştirme Gerekli |
|--------|-------|---------|--------------------|
|        | Q-Öğrenme algoritması, minimum kod değişikliği ile CartPole örneğinden başarıyla uyarlanmış ve 200 adımın altında bayrağı yakalama problemini çözebilmiştir. | İnternetten yeni bir Q-Öğrenme algoritması alınmış, ancak iyi belgelenmiş; ya da mevcut algoritma uyarlanmış, ancak istenen sonuçlara ulaşamamış. | Öğrenci herhangi bir algoritmayı başarıyla uyarlayamamış, ancak çözüm yolunda önemli adımlar atmıştır (durum ayrıklaştırma, Q-Tablo veri yapısı vb. uygulanmış). |

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul edilmez.