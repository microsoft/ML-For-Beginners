<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-06T07:50:46+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "tr"
}
-->
# Makine öğrenimi için kümeleme modelleri

Kümeleme, birbirine benzeyen nesneleri bulup bunları "küme" adı verilen gruplara ayırmayı amaçlayan bir makine öğrenimi görevidir. Kümelemenin makine öğrenimindeki diğer yaklaşımlardan farkı, işlemlerin otomatik olarak gerçekleşmesidir; aslında, denetimli öğrenmenin tam tersidir demek doğru olur.

## Bölgesel konu: Nijeryalı bir kitlenin müzik zevkine yönelik kümeleme modelleri 🎧

Nijerya'nın çeşitli kitlesi, farklı müzik zevklerine sahiptir. [Bu makaleden](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) ilham alınarak Spotify'dan toplanan verileri kullanarak Nijerya'da popüler olan bazı müziklere bakalım. Bu veri seti, çeşitli şarkıların 'dans edilebilirlik' puanı, 'akustiklik', ses yüksekliği, 'konuşma oranı', popülerlik ve enerji gibi verilerini içerir. Bu verilerdeki desenleri keşfetmek oldukça ilginç olacak!

![Bir plak çalar](../../../5-Clustering/images/turntable.jpg)

> Fotoğraf: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a>, <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Bu ders serisinde, kümeleme tekniklerini kullanarak verileri analiz etmenin yeni yollarını keşfedeceksiniz. Kümeleme, veri setinizde etiketler olmadığında özellikle faydalıdır. Eğer etiketler varsa, önceki derslerde öğrendiğiniz sınıflandırma teknikleri daha kullanışlı olabilir. Ancak, etiketlenmemiş verileri gruplamaya çalıştığınız durumlarda, kümeleme desenleri keşfetmek için harika bir yöntemdir.

> Kümeleme modelleriyle çalışmayı öğrenmenize yardımcı olabilecek kullanışlı düşük kod araçları vardır. Bu görev için [Azure ML'yi deneyin](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Dersler

1. [Kümelemeye giriş](1-Visualize/README.md)
2. [K-Means kümeleme](2-K-Means/README.md)

## Katkıda Bulunanlar

Bu dersler, [Jen Looper](https://www.twitter.com/jenlooper) tarafından 🎶 ile yazılmış ve [Rishit Dagli](https://rishit_dagli) ile [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) tarafından faydalı incelemelerle desteklenmiştir.

[Nijeryalı Şarkılar](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) veri seti, Spotify'dan toplanarak Kaggle'dan alınmıştır.

Bu dersi oluştururken yardımcı olan faydalı K-Means örnekleri arasında [iris incelemesi](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [tanıtıcı bir not defteri](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ve [varsayımsal bir STK örneği](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) bulunmaktadır.

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.