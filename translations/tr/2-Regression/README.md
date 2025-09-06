<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-06T07:43:53+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "tr"
}
-->
# Makine Öğrenimi için Regresyon Modelleri
## Bölgesel Konu: Kuzey Amerika'da Kabak Fiyatları için Regresyon Modelleri 🎃

Kuzey Amerika'da kabaklar genellikle Cadılar Bayramı için korkutucu yüzler şeklinde oyulur. Haydi, bu büyüleyici sebzeler hakkında daha fazla keşif yapalım!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> Fotoğraf: <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> tarafından <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> üzerinde
  
## Öğrenecekleriniz

[![Regresyona Giriş](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Regresyon Giriş Videosu - İzlemek için Tıklayın!")
> 🎥 Yukarıdaki görsele tıklayarak bu ders için kısa bir giriş videosu izleyebilirsiniz.

Bu bölümdeki dersler, makine öğrenimi bağlamında regresyon türlerini kapsar. Regresyon modelleri, değişkenler arasındaki _ilişkiyi_ belirlemeye yardımcı olabilir. Bu tür modeller, uzunluk, sıcaklık veya yaş gibi değerleri tahmin edebilir ve veri noktalarını analiz ederken değişkenler arasındaki ilişkileri ortaya çıkarabilir.

Bu ders serisinde, doğrusal ve lojistik regresyon arasındaki farkları ve hangisini ne zaman tercih etmeniz gerektiğini keşfedeceksiniz.

[![Yeni Başlayanlar için ML - Makine Öğrenimi için Regresyon Modellerine Giriş](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "Yeni Başlayanlar için ML - Makine Öğrenimi için Regresyon Modellerine Giriş")

> 🎥 Yukarıdaki görsele tıklayarak regresyon modellerine kısa bir giriş videosu izleyebilirsiniz.

Bu ders grubunda, makine öğrenimi görevlerine başlamak için gerekli ayarları yapacaksınız. Buna, veri bilimciler için yaygın bir ortam olan notebook'ları yönetmek için Visual Studio Code'u yapılandırmak da dahildir. Scikit-learn adlı bir makine öğrenimi kütüphanesini keşfedecek ve bu bölümde Regresyon modellerine odaklanarak ilk modellerinizi oluşturacaksınız.

> Regresyon modelleriyle çalışmayı öğrenmenize yardımcı olabilecek kullanışlı düşük kod araçları vardır. Bu görev için [Azure ML'yi deneyin](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### Dersler

1. [Araçlar ve Yöntemler](1-Tools/README.md)
2. [Veri Yönetimi](2-Data/README.md)
3. [Doğrusal ve Polinom Regresyon](3-Linear/README.md)
4. [Lojistik Regresyon](4-Logistic/README.md)

---
### Katkıda Bulunanlar

"Regresyon ile ML" ♥️ ile [Jen Looper](https://twitter.com/jenlooper) tarafından yazılmıştır.

♥️ Quiz katkıcıları arasında: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) ve [Ornella Altunyan](https://twitter.com/ornelladotcom) bulunmaktadır.

Kabak veri seti [Kaggle'daki bu proje](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) tarafından önerilmiştir ve veriler [Amerika Birleşik Devletleri Tarım Bakanlığı](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) tarafından dağıtılan "Specialty Crops Terminal Markets Standard Reports" kaynaklıdır. Dağılımı normalleştirmek için çeşitlere göre renk etrafında bazı noktalar ekledik. Bu veri kamu malıdır.

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.