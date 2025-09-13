<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-06T07:47:42+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "tr"
}
-->
# Scikit-learn ile Bir Regresyon Modeli Oluşturma: Veriyi Hazırlama ve Görselleştirme

![Veri görselleştirme infografiği](../../../../2-Regression/2-Data/images/data-visualization.png)

İnfografik: [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bu ders R dilinde de mevcut!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Giriş

Scikit-learn ile makine öğrenimi modeli oluşturmak için gerekli araçları kurduğunuza göre, artık verilerinizle ilgili sorular sormaya hazırsınız. Verilerle çalışırken ve makine öğrenimi çözümleri uygularken, doğru soruları sormanın, veri setinizin potansiyelini doğru bir şekilde ortaya çıkarmak için çok önemli olduğunu anlamak önemlidir.

Bu derste şunları öğreneceksiniz:

- Verilerinizi model oluşturma için nasıl hazırlayacağınızı.
- Matplotlib kullanarak veri görselleştirmeyi.

## Verilerinizle Doğru Soruyu Sormak

Cevaplanmasını istediğiniz soru, hangi tür makine öğrenimi algoritmalarını kullanacağınızı belirleyecektir. Aldığınız cevabın kalitesi ise büyük ölçüde verilerinizin doğasına bağlı olacaktır.

Bu ders için sağlanan [verilere](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) bir göz atın. Bu .csv dosyasını VS Code'da açabilirsiniz. Hızlı bir inceleme, boşluklar, metin ve sayısal verilerin bir karışımı olduğunu hemen gösteriyor. Ayrıca, 'Package' adında, verilerin 'sacks', 'bins' ve diğer değerlerin bir karışımı olduğu garip bir sütun var. Aslında, bu veri biraz dağınık.

[![Başlangıç Seviyesi ML - Bir Veri Setini Analiz Etme ve Temizleme](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "Başlangıç Seviyesi ML - Bir Veri Setini Analiz Etme ve Temizleme")

> 🎥 Yukarıdaki görsele tıklayarak bu ders için veriyi hazırlama sürecini içeren kısa bir videoyu izleyebilirsiniz.

Aslında, kutudan çıktığı gibi bir makine öğrenimi modeli oluşturmak için tamamen hazır bir veri seti elde etmek pek yaygın değildir. Bu derste, standart Python kütüphanelerini kullanarak ham bir veri setini nasıl hazırlayacağınızı öğreneceksiniz. Ayrıca, veriyi görselleştirmek için çeşitli teknikler öğreneceksiniz.

## Vaka Çalışması: 'Balkabağı Pazarı'

Bu klasörde, kök `data` klasöründe [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) adlı bir .csv dosyası bulacaksınız. Bu dosya, şehir bazında gruplandırılmış, balkabağı pazarına dair 1757 satır veri içeriyor. Bu, Amerika Birleşik Devletleri Tarım Bakanlığı tarafından dağıtılan [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) raporlarından çıkarılmış ham bir veridir.

### Veriyi Hazırlama

Bu veri kamuya açık bir alandadır. USDA web sitesinden şehir başına birçok ayrı dosya olarak indirilebilir. Çok fazla ayrı dosyadan kaçınmak için, tüm şehir verilerini tek bir elektronik tabloya birleştirdik, bu nedenle veriyi biraz _hazırlamış_ olduk. Şimdi, veriye daha yakından bakalım.

### Balkabağı Verisi - İlk Sonuçlar

Bu veri hakkında ne fark ettiniz? Zaten metin, sayılar, boşluklar ve anlamlandırmanız gereken garip değerlerin bir karışımı olduğunu gördünüz.

Bu veriye bir Regresyon tekniği kullanarak hangi soruyu sorabilirsiniz? Örneğin, "Belirli bir ayda satışa sunulan bir balkabağının fiyatını tahmin et." Veriye tekrar bakıldığında, bu göreve uygun bir veri yapısı oluşturmak için bazı değişiklikler yapmanız gerektiği görülüyor.

## Egzersiz - Balkabağı Verisini Analiz Etme

Balkabağı verisini analiz etmek ve hazırlamak için, verileri şekillendirmede çok kullanışlı bir araç olan [Pandas](https://pandas.pydata.org/) kütüphanesini kullanacağız.

### İlk Olarak, Eksik Tarihleri Kontrol Edin

Öncelikle eksik tarihleri kontrol etmek için şu adımları izleyin:

1. Tarihleri ay formatına dönüştürün (bu tarihler ABD formatında, yani `AA/GG/YYYY`).
2. Ayı yeni bir sütuna çıkarın.

_VS Code_ içinde _notebook.ipynb_ dosyasını açın ve elektronik tabloyu yeni bir Pandas dataframe'ine aktarın.

1. İlk beş satırı görüntülemek için `head()` fonksiyonunu kullanın.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Son beş satırı görüntülemek için hangi fonksiyonu kullanırdınız?

1. Mevcut dataframe'de eksik veri olup olmadığını kontrol edin:

    ```python
    pumpkins.isnull().sum()
    ```

    Eksik veri var, ancak bu belki de ele alınan görev için önemli olmayabilir.

1. Dataframe'inizi daha kolay çalışılabilir hale getirmek için, yalnızca ihtiyacınız olan sütunları seçin. Bunun için `loc` fonksiyonunu kullanabilirsiniz. Bu fonksiyon, orijinal dataframe'den bir grup satır (ilk parametre) ve sütun (ikinci parametre) çıkarır. Aşağıdaki durumda `:` ifadesi "tüm satırlar" anlamına gelir.

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### İkinci Olarak, Balkabağının Ortalama Fiyatını Belirleyin

Belirli bir ayda bir balkabağının ortalama fiyatını belirlemek için hangi sütunları seçerdiniz? İpucu: 3 sütuna ihtiyacınız olacak.

Çözüm: `Low Price` ve `High Price` sütunlarının ortalamasını alarak yeni bir `Price` sütununu doldurun ve `Date` sütununu yalnızca ayı gösterecek şekilde dönüştürün. Neyse ki, yukarıdaki kontrol sonucuna göre, tarihler veya fiyatlar için eksik veri yok.

1. Ortalamayı hesaplamak için şu kodu ekleyin:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Kontrol etmek istediğiniz herhangi bir veriyi `print(month)` kullanarak yazdırabilirsiniz.

2. Şimdi, dönüştürdüğünüz veriyi yeni bir Pandas dataframe'ine kopyalayın:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Dataframe'inizi yazdırdığınızda, yeni regresyon modelinizi oluşturabileceğiniz temiz ve düzenli bir veri seti göreceksiniz.

### Ama Durun! Burada Garip Bir Şey Var

`Package` sütununa bakarsanız, balkabaklarının birçok farklı şekilde satıldığını görebilirsiniz. Bazıları '1 1/9 bushel' ölçülerinde, bazıları '1/2 bushel' ölçülerinde, bazıları balkabağı başına, bazıları pound başına ve bazıları da genişlikleri değişen büyük kutularda satılıyor.

> Balkabaklarını tutarlı bir şekilde tartmak oldukça zor görünüyor.

Orijinal veriye bakıldığında, `Unit of Sale` değeri 'EACH' veya 'PER BIN' olan her şeyin `Package` türü de inç başına, kutu başına veya 'her biri' olarak görünüyor. Balkabaklarını tutarlı bir şekilde tartmak oldukça zor görünüyor, bu yüzden `Package` sütununda 'bushel' kelimesini içeren balkabaklarını seçerek filtreleme yapalım.

1. Dosyanın başında, ilk .csv içe aktarma işleminin altına bir filtre ekleyin:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Şimdi veriyi yazdırırsanız, yalnızca 'bushel' içeren yaklaşık 415 satır veriyi aldığınızı görebilirsiniz.

### Ama Durun! Yapılacak Bir Şey Daha Var

Bushel miktarının satır başına değiştiğini fark ettiniz mi? Fiyatlandırmayı normalize etmeniz ve bushel başına fiyatı göstermeniz gerekiyor, bu yüzden standartlaştırmak için biraz matematik yapın.

1. Yeni dataframe'inizi oluşturduğunuz bloğun altına şu satırları ekleyin:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) sitesine göre, bir bushel'in ağırlığı ürün türüne bağlıdır, çünkü bu bir hacim ölçüsüdür. "Örneğin, bir bushel domatesin ağırlığı 56 pound olmalıdır... Yapraklar ve yeşillikler daha az ağırlıkla daha fazla yer kaplar, bu yüzden bir bushel ıspanak sadece 20 pound'dur." Bu oldukça karmaşık! Bushel'den pound'a dönüşüm yapmaya zahmet etmeyelim ve bunun yerine bushel başına fiyatlandırma yapalım. Ancak, tüm bu bushel çalışması, verilerinizin doğasını anlamanın ne kadar önemli olduğunu gösteriyor!

Şimdi, bushel ölçümüne dayalı birim başına fiyatlandırmayı analiz edebilirsiniz. Veriyi bir kez daha yazdırırsanız, nasıl standartlaştırıldığını görebilirsiniz.

✅ Yarım bushel ile satılan balkabaklarının çok pahalı olduğunu fark ettiniz mi? Bunun nedenini anlayabilir misiniz? İpucu: Küçük balkabakları büyük olanlardan çok daha pahalıdır, muhtemelen bir bushel başına çok daha fazla küçük balkabağı sığdığı için.

## Görselleştirme Stratejileri

Bir veri bilimcisinin rolü, üzerinde çalıştığı verilerin kalitesini ve doğasını göstermektir. Bunu yapmak için genellikle ilginç görselleştirmeler, yani grafikler, çizimler ve tablolar oluştururlar. Bu şekilde, görsel olarak ilişkileri ve aksi takdirde ortaya çıkması zor olan boşlukları gösterebilirler.

[![Başlangıç Seviyesi ML - Matplotlib ile Veri Görselleştirme](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "Başlangıç Seviyesi ML - Matplotlib ile Veri Görselleştirme")

> 🎥 Yukarıdaki görsele tıklayarak bu ders için veriyi görselleştirme sürecini içeren kısa bir videoyu izleyebilirsiniz.

Görselleştirmeler ayrıca, veri için en uygun makine öğrenimi tekniğini belirlemeye yardımcı olabilir. Örneğin, bir çizgiyi takip ediyor gibi görünen bir dağılım grafiği, verinin doğrusal regresyon çalışması için iyi bir aday olduğunu gösterebilir.

Jupyter defterlerinde iyi çalışan bir veri görselleştirme kütüphanesi [Matplotlib](https://matplotlib.org/) (önceki derste de gördüğünüz) kütüphanesidir.

> Veri görselleştirme konusunda daha fazla deneyim kazanmak için [bu eğitimlere](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott) göz atın.

## Egzersiz - Matplotlib ile Deney Yapın

Yeni oluşturduğunuz dataframe'i görüntülemek için bazı temel grafikler oluşturmaya çalışın. Basit bir çizgi grafiği ne gösterir?

1. Dosyanın başında, Pandas içe aktarma işleminin altına Matplotlib'i ekleyin:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Tüm defteri yeniden çalıştırarak yenileyin.
1. Defterin altına, veriyi bir kutu grafiği olarak çizmek için bir hücre ekleyin:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Fiyat ve ay ilişkisini gösteren bir dağılım grafiği](../../../../2-Regression/2-Data/images/scatterplot.png)

    Bu faydalı bir grafik mi? Sizi şaşırtan bir şey var mı?

    Bu grafik pek faydalı değil, çünkü yalnızca verilerinizi belirli bir ayda bir dizi nokta olarak gösteriyor.

### Daha Faydalı Hale Getirin

Grafiklerin faydalı veriler göstermesi için genellikle verileri bir şekilde gruplamanız gerekir. Y ekseninin ayları gösterdiği ve verilerin dağılımını gösterdiği bir grafik oluşturalım.

1. Gruplandırılmış bir çubuk grafik oluşturmak için bir hücre ekleyin:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Fiyat ve ay ilişkisini gösteren bir çubuk grafik](../../../../2-Regression/2-Data/images/barchart.png)

    Bu daha faydalı bir veri görselleştirme! Balkabaklarının en yüksek fiyatının Eylül ve Ekim aylarında olduğunu gösteriyor gibi görünüyor. Bu beklentinizi karşılıyor mu? Neden veya neden değil?

---

## 🚀Meydan Okuma

Matplotlib'in sunduğu farklı görselleştirme türlerini keşfedin. Hangi türler regresyon problemleri için daha uygundur?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Verileri görselleştirmenin birçok yoluna göz atın. Mevcut çeşitli kütüphanelerin bir listesini yapın ve hangilerinin belirli görev türleri için en iyi olduğunu not edin, örneğin 2D görselleştirmeler ve 3D görselleştirmeler. Ne keşfediyorsunuz?

## Ödev

[Veri görselleştirme keşfi](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul edilmez.