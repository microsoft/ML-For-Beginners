# Daha Gerçekçi Bir Dünya

Bizim durumumuzda, Peter neredeyse hiç yorulmadan veya acıkmadan dolaşabiliyordu. Daha gerçekçi bir dünyada, arada bir oturup dinlenmesi ve kendini beslemesi gerekecek. Dünyamızı daha gerçekçi hale getirelim ve aşağıdaki kuralları uygulayalım:

1. Bir yerden bir yere hareket ederek, Peter **enerji** kaybeder ve biraz **yorgunluk** kazanır.
2. Peter elma yiyerek daha fazla enerji kazanabilir.
3. Peter, ağacın altında veya çimenlerin üzerinde dinlenerek yorgunluğundan kurtulabilir (yani, tahtada bir ağaç veya çimen bulunan bir yere yürüyerek - yeşil alan)
4. Peter, kurdu bulup öldürmek zorunda.
5. Kurdu öldürmek için, Peter'ın belirli seviyelerde enerji ve yorgunluğa sahip olması gerekir, aksi takdirde savaşı kaybeder.
## Talimatlar

Çözümünüz için başlangıç noktası olarak orijinal [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) defterini kullanın.

Ödül fonksiyonunu oyunun kurallarına göre yukarıda belirtildiği şekilde değiştirin, pekiştirmeli öğrenme algoritmasını çalıştırarak oyunu kazanmak için en iyi stratejiyi öğrenin ve rastgele yürüyüş ile algoritmanızın sonuçlarını, kazanılan ve kaybedilen oyun sayısı açısından karşılaştırın.

> **Note**: Yeni dünyanızda, durum daha karmaşıktır ve insan pozisyonuna ek olarak yorgunluk ve enerji seviyelerini de içerir. Durumu bir demet (Tahta, enerji, yorgunluk) olarak temsil etmeyi seçebilir veya durum için bir sınıf tanımlayabilirsiniz (bunu `Board`'dan türetmek isteyebilirsiniz), ya da orijinal `Board` sınıfını [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py) içinde değiştirebilirsiniz.

Çözümünüzde, rastgele yürüyüş stratejisinden sorumlu olan kodu koruyun ve algoritmanızın sonuçlarını rastgele yürüyüş ile sonunda karşılaştırın.

> **Note**: Çalışması için hiperparametreleri ayarlamanız gerekebilir, özellikle epoch sayısını. Oyunun başarısı (kurtla savaşma) nadir bir olay olduğu için, çok daha uzun eğitim süresi bekleyebilirsiniz.
## Değerlendirme Kriterleri

| Kriterler | Örnek                                                                                                                                                                                                 | Yeterli                                                                                                                                                                                 | Geliştirmeye İhtiyaç Var                                                                                                                     |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Yeni dünya kurallarının tanımı, Q-Öğrenme algoritması ve bazı metinsel açıklamalar içeren bir defter sunulmuştur. Q-Öğrenme, rastgele yürüyüşle karşılaştırıldığında sonuçları önemli ölçüde iyileştirebilir. | Defter sunulmuş, Q-Öğrenme uygulanmış ve rastgele yürüyüşle karşılaştırıldığında sonuçları iyileştirmiş, ancak önemli ölçüde değil; ya da defter kötü belgelenmiş ve kod iyi yapılandırılmamış | Dünyanın kurallarını yeniden tanımlamak için bazı girişimlerde bulunulmuş, ancak Q-Öğrenme algoritması çalışmıyor veya ödül fonksiyonu tam olarak tanımlanmamış |

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından doğabilecek herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.