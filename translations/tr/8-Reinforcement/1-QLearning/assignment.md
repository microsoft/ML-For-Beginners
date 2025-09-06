<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-06T08:04:28+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "tr"
}
-->
# Daha Gerçekçi Bir Dünya

Bizim senaryomuzda, Peter neredeyse hiç yorulmadan veya acıkmadan hareket edebiliyordu. Daha gerçekçi bir dünyada, Peter zaman zaman oturup dinlenmek ve kendini beslemek zorunda kalır. Dünyamızı daha gerçekçi hale getirelim ve şu kuralları uygulayalım:

1. Bir yerden başka bir yere hareket ettiğinde, Peter **enerji** kaybeder ve biraz **yorgunluk** kazanır.
2. Peter, elma yiyerek daha fazla enerji kazanabilir.
3. Peter, ağacın altında veya çimenlerin üzerinde dinlenerek yorgunluğunu giderebilir (yani, tahtada bir ağaç veya çimen bulunan bir konuma yürümek - yeşil alan).
4. Peter, kurdu bulup öldürmek zorundadır.
5. Kurdu öldürmek için Peter'ın belirli enerji ve yorgunluk seviyelerine sahip olması gerekir, aksi takdirde savaşı kaybeder.

## Talimatlar

Çözümünüz için başlangıç noktası olarak orijinal [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) defterini kullanın.

Yukarıdaki ödül fonksiyonunu oyunun kurallarına göre değiştirin, en iyi stratejiyi öğrenmek için pekiştirmeli öğrenme algoritmasını çalıştırın ve rastgele yürüyüş ile algoritmanızın sonuçlarını kazanan ve kaybedilen oyun sayısı açısından karşılaştırın.

> **Not**: Yeni dünyanızda, durum daha karmaşıktır ve insanın pozisyonuna ek olarak yorgunluk ve enerji seviyelerini de içerir. Durumu bir demet (Board,energy,fatigue) olarak temsil etmeyi seçebilir, bir sınıf tanımlayabilir (bunu `Board` sınıfından türetmek isteyebilirsiniz), veya [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py) içindeki orijinal `Board` sınıfını değiştirebilirsiniz.

Çözümünüzde, rastgele yürüyüş stratejisinden sorumlu olan kodu koruyun ve algoritmanızın sonuçlarını rastgele yürüyüş ile sonunda karşılaştırın.

> **Not**: Çalışması için hiperparametreleri ayarlamanız gerekebilir, özellikle epoch sayısını. Oyunun başarısı (kurdu yenmek) nadir bir olay olduğu için, çok daha uzun bir eğitim süresi bekleyebilirsiniz.

## Değerlendirme Kriterleri

| Kriter   | Örnek                                                                                                                                                                                                 | Yeterli                                                                                                                                                                                | Geliştirme Gerekiyor                                                                                                                       |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Yeni dünya kurallarının tanımı, Q-Learning algoritması ve bazı metinsel açıklamalar içeren bir defter sunulmuştur. Q-Learning, rastgele yürüyüşe kıyasla sonuçları önemli ölçüde iyileştirebilmektedir. | Bir defter sunulmuş, Q-Learning uygulanmış ve rastgele yürüyüşe kıyasla sonuçları iyileştirmiş ancak önemli ölçüde değil; veya defter zayıf bir şekilde belgelenmiş ve kod iyi yapılandırılmamış. | Dünyanın kurallarını yeniden tanımlamak için bazı girişimlerde bulunulmuş, ancak Q-Learning algoritması çalışmıyor veya ödül fonksiyonu tam olarak tanımlanmamış.                  |

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.