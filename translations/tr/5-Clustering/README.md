# Makine öğrenimi için kümeleme modelleri

Kümeleme, nesnelerin birbirine benzeyenlerini bulup bunları kümeler adı verilen gruplara ayırmayı amaçlayan bir makine öğrenimi görevidir. Kümelemeyi makine öğrenimindeki diğer yaklaşımlardan ayıran özellik, her şeyin otomatik olarak gerçekleşmesidir; aslında, denetlenmiş öğrenmenin tam tersi olduğu söylenebilir.

## Bölgesel konu: Nijeryalı izleyicinin müzik zevklerine yönelik kümeleme modelleri 🎧

Nijerya’nın çeşitli izleyicileri farklı müzik zevklerine sahiptir. Spotify'dan toplanan verileri kullanarak (bu makaleden esinlenilmiştir [this article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), Nijerya’da popüler olan bazı müziklere bakalım. Bu veri seti çeşitli şarkıların 'danceability' (dans edilebilirlik) puanı, 'acousticness' (akustiklik), ses şiddeti, 'speechiness' (konuşma yoğunluğu), popülerlik ve enerji gibi verilerini içerir. Bu verideki desenleri keşfetmek ilginç olacak!

![Bir pikap](../../../translated_images/tr/turntable.f2b86b13c53302dc.webp)

> Fotoğrafçı <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a>, <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> üzerinden
  
Bu ders serisinde, kümeleme tekniklerini kullanarak verileri analiz etmenin yeni yollarını keşfedeceksiniz. Kümeleme, özellikle veri setinizde etiket olmadığında çok faydalıdır. Eğer etiketler varsa, önceki derslerde öğrendiğiniz sınıflandırma teknikleri daha faydalı olabilir. Ancak etiketlenmemiş verileri gruplamak istediğiniz durumlarda, kümeleme desenleri keşfetmek için harika bir yöntemdir.

> Kümeleme modelleriyle çalışmayı öğrenmenize yardımcı olabilecek kullanışlı düşük kodlu araçlar vardır. Bu görev için [Azure ML’yi deneyin](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Dersler

1. [Kümelemeye giriş](1-Visualize/README.md)
2. [K-Ortalamalar kümeleme](2-K-Means/README.md)

## Teşekkürler

Bu dersler, 🎶 ile [Jen Looper](https://www.twitter.com/jenlooper) tarafından yazılmıştır ve [Rishit Dagli](https://rishit_dagli/) ve [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) tarafından faydalı incelemeler yapılmıştır.

[Nijerya Şarkıları](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) veri seti Spotify’dan kazınarak Kaggle’dan sağlanmıştır.

Bu dersin hazırlanmasında faydalanılan bazı K-Ortalamalar örnekleri arasında bu [iris keşfi](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), bu [giriş defteri](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ve bu [varsayımsal STK örneği](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) bulunmaktadır.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Feragatname**:
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayınız. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımı sonucu ortaya çıkabilecek yanlış anlamalardan veya yanlış yorumlamalardan sorumlu değiliz.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->