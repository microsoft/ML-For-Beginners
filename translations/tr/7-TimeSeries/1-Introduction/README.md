<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-06T07:49:25+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "tr"
}
-->
# Zaman Serisi Tahminine Giriş

![Zaman serilerinin bir özetini içeren sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Bu derste ve sonraki derste, zaman serisi tahmini hakkında biraz bilgi edineceksiniz. Bu, bir makine öğrenimi bilim insanının repertuarında ilginç ve değerli bir alan olup, diğer konular kadar yaygın bilinmeyebilir. Zaman serisi tahmini, bir tür 'kristal küre' gibidir: fiyat gibi bir değişkenin geçmiş performansına dayanarak, gelecekteki potansiyel değerini tahmin edebilirsiniz.

[![Zaman serisi tahminine giriş](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Zaman serisi tahminine giriş")

> 🎥 Zaman serisi tahmini hakkında bir video için yukarıdaki görsele tıklayın

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

Zaman serisi tahmini, fiyatlandırma, envanter ve tedarik zinciri sorunlarına doğrudan uygulanabilirliği nedeniyle iş dünyası için gerçek bir değer taşıyan faydalı ve ilginç bir alandır. Derin öğrenme teknikleri, gelecekteki performansı daha iyi tahmin etmek için daha fazla içgörü elde etmek amacıyla kullanılmaya başlanmış olsa da, zaman serisi tahmini hala klasik makine öğrenimi teknikleriyle büyük ölçüde şekillenmektedir.

> Penn State'in faydalı zaman serisi müfredatına [buradan](https://online.stat.psu.edu/stat510/lesson/1) ulaşabilirsiniz.

## Giriş

Diyelim ki, zaman içinde ne sıklıkla ve ne kadar süreyle kullanıldıklarına dair veri sağlayan bir dizi akıllı parkmetreyi yönetiyorsunuz.

> Parkmetrenin geçmiş performansına dayanarak, arz ve talep yasalarına göre gelecekteki değerini tahmin edebileceğinizi hayal edin.

Hedefinize ulaşmak için ne zaman harekete geçmeniz gerektiğini doğru bir şekilde tahmin etmek, zaman serisi tahmini ile ele alınabilecek bir zorluktur. Yoğun zamanlarda park yeri arayan insanlardan daha fazla ücret almak onları mutlu etmeyebilir, ancak sokakları temizlemek için gelir elde etmenin kesin bir yolu olurdu!

Şimdi bazı zaman serisi algoritmalarını inceleyelim ve veri temizleme ve hazırlama işlemlerine başlamak için bir notebook oluşturalım. Analiz edeceğiniz veri, GEFCom2014 tahmin yarışmasından alınmıştır. Bu veri, 2012 ile 2014 yılları arasında 3 yıl boyunca saatlik elektrik yükü ve sıcaklık değerlerini içermektedir. Elektrik yükü ve sıcaklık verilerinin geçmişteki kalıplarına dayanarak, elektrik yükünün gelecekteki değerlerini tahmin edebilirsiniz.

Bu örnekte, yalnızca geçmiş yük verilerini kullanarak bir zaman adımını ileriye tahmin etmeyi öğreneceksiniz. Ancak başlamadan önce, perde arkasında neler olduğunu anlamak faydalı olacaktır.

## Bazı Tanımlar

'Zaman serisi' terimiyle karşılaştığınızda, bunun birkaç farklı bağlamda nasıl kullanıldığını anlamanız gerekir.

🎓 **Zaman Serisi**

Matematikte, "zaman serisi, zaman sırasına göre dizilmiş (veya listelenmiş veya grafiğe dökülmüş) veri noktalarının bir serisidir. En yaygın olarak, zaman serisi, ardışık olarak eşit aralıklarla alınan bir dizidir." Zaman serisine bir örnek, [Dow Jones Sanayi Ortalaması](https://wikipedia.org/wiki/Time_series)'nın günlük kapanış değeridir. Zaman serisi grafikleri ve istatistiksel modelleme, sinyal işleme, hava tahmini, deprem tahmini ve olayların meydana geldiği ve veri noktalarının zaman içinde grafiğe dökülebileceği diğer alanlarda sıkça karşılaşılan bir yöntemdir.

🎓 **Zaman Serisi Analizi**

Zaman serisi analizi, yukarıda bahsedilen zaman serisi verilerinin analizidir. Zaman serisi verileri, bir kesintili zaman serisi gibi farklı biçimler alabilir; bu, bir kesinti olayından önce ve sonra bir zaman serisinin evrimindeki kalıpları tespit eder. Zaman serisi için gereken analiz türü, verinin doğasına bağlıdır. Zaman serisi verileri, sayı veya karakter dizileri biçiminde olabilir.

Yapılacak analiz, frekans alanı ve zaman alanı, doğrusal ve doğrusal olmayan gibi çeşitli yöntemler kullanır. Bu tür verileri analiz etmenin birçok yolu hakkında [buradan](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) daha fazla bilgi edinin.

🎓 **Zaman Serisi Tahmini**

Zaman serisi tahmini, geçmişte toplanan verilerin gösterdiği kalıplara dayanarak gelecekteki değerleri tahmin etmek için bir modelin kullanılmasıdır. Zaman serisi verilerini keşfetmek için regresyon modelleri kullanmak mümkün olsa da, zaman serisi verileri en iyi şekilde özel türde modeller kullanılarak analiz edilir.

Zaman serisi verileri, sıralı gözlemler listesidir ve doğrusal regresyonla analiz edilebilecek verilerden farklıdır. En yaygın olanı ARIMA'dır; bu, "Oto-Regresif Entegre Hareketli Ortalama" anlamına gelen bir kısaltmadır.

[ARIMA modelleri](https://online.stat.psu.edu/stat510/lesson/1/1.1), "bir serinin mevcut değerini geçmiş değerler ve geçmiş tahmin hatalarıyla ilişkilendirir." Bu modeller, verilerin zaman içinde sıralandığı zaman alanı verilerini analiz etmek için en uygundur.

> ARIMA modellerinin birkaç türü vardır. Bunlar hakkında [buradan](https://people.duke.edu/~rnau/411arim.htm) daha fazla bilgi edinebilir ve bir sonraki derste bu modellere değinebilirsiniz.

Bir sonraki derste, zaman içinde değerini değiştiren bir değişkene odaklanan [Tek Değişkenli Zaman Serisi](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) kullanarak bir ARIMA modeli oluşturacaksınız. Bu tür verilere bir örnek, Mauna Loa Gözlemevi'nde aylık CO2 konsantrasyonunu kaydeden [bu veri setidir](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm):

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Bu veri setinde zaman içinde değişen değişkeni belirleyin.

## Zaman Serisi Verilerinde Dikkate Alınması Gereken Özellikler

Zaman serisi verilerine baktığınızda, bu verilerin [belirli özelliklere](https://online.stat.psu.edu/stat510/lesson/1/1.1) sahip olduğunu fark edebilirsiniz. Bu özellikleri daha iyi anlamak ve kalıplarını analiz etmek için bazı istatistiksel teknikler kullanarak bu 'gürültüyü' azaltmanız gerekebilir.

Zaman serisi verilerini analiz etmek istediğiniz bir 'sinyal' olarak düşünürseniz, bu özellikler 'gürültü' olarak değerlendirilebilir. İşte zaman serisi ile çalışabilmek için bilmeniz gereken bazı kavramlar:

🎓 **Trendler**

Trendler, zaman içinde ölçülebilir artışlar ve azalmalar olarak tanımlanır. [Daha fazla bilgi edinin](https://machinelearningmastery.com/time-series-trends-in-python). Zaman serisi bağlamında, trendleri nasıl kullanacağınız ve gerekirse zaman serinizden nasıl çıkaracağınız hakkında bilgi edinebilirsiniz.

🎓 **[Mevsimsellik](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Mevsimsellik, örneğin tatil dönemlerinde satışları etkileyebilecek periyodik dalgalanmalar olarak tanımlanır. [Buradan](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) farklı türde grafiklerin verilerdeki mevsimselliği nasıl gösterdiğine göz atabilirsiniz.

🎓 **Aykırı Değerler**

Aykırı değerler, standart veri varyansından oldukça uzak olan değerlerdir.

🎓 **Uzun Dönem Döngü**

Mevsimsellikten bağımsız olarak, veriler bir yıldan uzun süren ekonomik durgunluk gibi uzun dönemli bir döngü gösterebilir.

🎓 **Sabit Varyans**

Zaman içinde bazı veriler, örneğin günlük ve gece enerji kullanımı gibi sabit dalgalanmalar gösterebilir.

🎓 **Ani Değişiklikler**

Veriler, daha fazla analize ihtiyaç duyabilecek ani bir değişiklik gösterebilir. Örneğin, COVID nedeniyle işletmelerin ani kapanması verilerde değişikliklere neden olmuştur.

✅ İşte birkaç yıl boyunca günlük oyun içi para harcamasını gösteren bir [örnek zaman serisi grafiği](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python). Bu verilerde yukarıda listelenen özelliklerden herhangi birini belirleyebilir misiniz?

![Oyun içi para harcaması](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Egzersiz - Elektrik Kullanımı Verileriyle Başlamak

Geçmiş kullanım verilerine dayanarak gelecekteki elektrik kullanımını tahmin etmek için bir zaman serisi modeli oluşturmaya başlayalım.

> Bu örnekteki veriler, GEFCom2014 tahmin yarışmasından alınmıştır. 2012 ile 2014 yılları arasında 3 yıl boyunca saatlik elektrik yükü ve sıcaklık değerlerini içermektedir.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli ve Rob J. Hyndman, "Olasılıksal enerji tahmini: Küresel Enerji Tahmin Yarışması 2014 ve sonrası", Uluslararası Tahmin Dergisi, cilt 32, sayı 3, s. 896-913, Temmuz-Eylül, 2016.

1. Bu dersin `working` klasöründe _notebook.ipynb_ dosyasını açın. Verileri yüklemenize ve görselleştirmenize yardımcı olacak kütüphaneleri ekleyerek başlayın:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Not: Ortamınızı ayarlayan ve verileri indirme işlemini gerçekleştiren `common` klasöründeki dosyaları kullanıyorsunuz.

2. Ardından, `load_data()` ve `head()` çağrısı yaparak veriyi bir dataframe olarak inceleyin:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    İki sütunun tarih ve yükü temsil ettiğini görebilirsiniz:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Şimdi, `plot()` çağrısı yaparak veriyi görselleştirin:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![enerji grafiği](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Şimdi, 2014 Temmuz ayının ilk haftasını `[başlangıç tarihi]:[bitiş tarihi]` deseniyle `energy` girdisi olarak sağlayarak görselleştirin:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![temmuz](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Harika bir grafik! Bu grafiklere bakın ve yukarıda listelenen özelliklerden herhangi birini belirleyebilir misiniz? Veriyi görselleştirerek ne çıkarımlar yapabiliriz?

Bir sonraki derste, bazı tahminler oluşturmak için bir ARIMA modeli oluşturacaksınız.

---

## 🚀Meydan Okuma

Zaman serisi tahmininden fayda sağlayabilecek tüm endüstriler ve araştırma alanlarının bir listesini yapın. Bu tekniklerin sanatlarda bir uygulamasını düşünebilir misiniz? Ekonometrik? Ekoloji? Perakende? Endüstri? Finans? Başka nerelerde?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Burada ele alınmayacak olsa da, sinir ağları bazen zaman serisi tahmininin klasik yöntemlerini geliştirmek için kullanılır. Bu konuda [bu makalede](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412) daha fazla bilgi edinin.

## Ödev

[Daha fazla zaman serisi görselleştirin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.