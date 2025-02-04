# Makine öğrenimi için kümeleme modelleri

Kümeleme, benzer nesneleri bulmayı ve bunları kümeler olarak adlandırılan gruplar halinde gruplamayı amaçlayan bir makine öğrenimi görevidir. Kümelemeyi makine öğrenimindeki diğer yaklaşımlardan ayıran şey, her şeyin otomatik olarak gerçekleşmesidir. Aslında, denetimli öğrenmenin tam tersidir demek doğru olur.

## Bölgesel konu: Nijeryalı bir izleyici kitlesinin müzik zevkine yönelik kümeleme modelleri 🎧

Nijerya'nın çeşitli izleyici kitlesi, çeşitli müzik zevklerine sahiptir. Spotify'dan alınan verileri kullanarak (bu makaleden ilham alarak: [bu makale](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), Nijerya'da popüler olan bazı müziklere bakalım. Bu veri kümesi, çeşitli şarkıların 'dans edilebilirlik' puanı, 'akustiklik', ses yüksekliği, 'konuşkanlık', popülerlik ve enerji hakkında veriler içerir. Bu verilerdeki kalıpları keşfetmek ilginç olacak!

![Bir turntable](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.tr.jpg)

> Fotoğraf <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> tarafından <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> üzerinde
  
Bu ders serisinde, kümeleme tekniklerini kullanarak verileri analiz etmenin yeni yollarını keşfedeceksiniz. Kümeleme, veri kümenizde etiketler olmadığında özellikle yararlıdır. Eğer etiketler varsa, önceki derslerde öğrendiğiniz sınıflandırma teknikleri daha yararlı olabilir. Ancak etiketlenmemiş verileri gruplamayı amaçladığınız durumlarda, kümeleme kalıpları keşfetmenin harika bir yoludur.

> Kümeleme modelleri ile çalışmayı öğrenmenize yardımcı olabilecek kullanışlı düşük kod araçları vardır. Bu görev için [Azure ML'yi deneyin](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Dersler

1. [Kümelenmeye giriş](1-Visualize/README.md)
2. [K-Means kümeleme](2-K-Means/README.md)

## Katkıda Bulunanlar

Bu dersler 🎶 ile [Jen Looper](https://www.twitter.com/jenlooper) tarafından yazıldı ve [Rishit Dagli](https://rishit_dagli) ve [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) tarafından faydalı incelemelerle desteklendi.

[Nijeryalı Şarkılar](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) veri kümesi, Spotify'dan alınarak Kaggle'dan temin edilmiştir.

Bu dersi oluştururken yardımcı olan faydalı K-Means örnekleri arasında bu [iris keşfi](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), bu [giriş not defteri](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ve bu [varsayımsal STK örneği](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering) bulunmaktadır.

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belge, kendi dilinde yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi tavsiye edilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.