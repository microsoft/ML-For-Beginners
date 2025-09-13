<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-06T07:48:34+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "tr"
}
-->
# ARIMA ile Zaman Serisi Tahmini

Önceki derste, zaman serisi tahmini hakkında biraz bilgi edindiniz ve bir zaman dilimi boyunca elektrik yükündeki dalgalanmaları gösteren bir veri seti yüklediniz.

[![ARIMA'ya Giriş](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "ARIMA'ya Giriş")

> 🎥 Yukarıdaki görsele tıklayarak bir video izleyebilirsiniz: ARIMA modellerine kısa bir giriş. Örnek R dilinde yapılmıştır, ancak kavramlar evrenseldir.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## Giriş

Bu derste, [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average) ile model oluşturmanın özel bir yolunu keşfedeceksiniz. ARIMA modelleri, özellikle [durağan olmayan](https://wikipedia.org/wiki/Stationary_process) verileri modellemek için uygundur.

## Genel Kavramlar

ARIMA ile çalışabilmek için bilmeniz gereken bazı kavramlar vardır:

- 🎓 **Durağanlık**. İstatistiksel bağlamda durağanlık, zaman içinde kaydırıldığında dağılımı değişmeyen verilere atıfta bulunur. Durağan olmayan veriler ise analiz edilebilmesi için dönüştürülmesi gereken eğilimlerden kaynaklanan dalgalanmalar gösterir. Örneğin, mevsimsellik verilerde dalgalanmalara neden olabilir ve 'mevsimsel fark alma' işlemiyle ortadan kaldırılabilir.

- 🎓 **[Fark Alma](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. İstatistiksel bağlamda fark alma, durağan olmayan verileri durağan hale getirmek için sabit olmayan eğilimi ortadan kaldırma işlemine atıfta bulunur. "Fark alma, bir zaman serisinin seviyesindeki değişiklikleri ortadan kaldırır, eğilim ve mevsimselliği yok eder ve dolayısıyla zaman serisinin ortalamasını sabitler." [Shixiong ve diğerleri tarafından yazılan makale](https://arxiv.org/abs/1904.07632)

## Zaman Serisi Bağlamında ARIMA

ARIMA'nın bölümlerini inceleyerek zaman serisi verilerini modellemeye nasıl yardımcı olduğunu ve tahmin yapmamıza nasıl olanak sağladığını daha iyi anlayalım.

- **AR - Otoregresif (AutoRegressive)**. Otoregresif modeller, adından da anlaşılacağı gibi, verilerinizdeki önceki değerleri analiz etmek ve bunlar hakkında varsayımlarda bulunmak için 'geçmişe' bakar. Bu önceki değerlere 'gecikmeler' denir. Örneğin, aylık kalem satışlarını gösteren bir veri. Her ayın satış toplamı, veri setinde 'gelişen bir değişken' olarak kabul edilir. Bu model, "ilgili gelişen değişkenin kendi gecikmeli (yani, önceki) değerlerine göre regresyon yapılması" ile oluşturulur. [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - Entegre (Integrated)**. Benzer 'ARMA' modellerinden farklı olarak, ARIMA'daki 'I', *[entegre](https://wikipedia.org/wiki/Order_of_integration)* yönüne atıfta bulunur. Veriler, durağan olmayanlığı ortadan kaldırmak için fark alma adımları uygulandığında 'entegre' hale gelir.

- **MA - Hareketli Ortalama (Moving Average)**. Bu modelin [hareketli ortalama](https://wikipedia.org/wiki/Moving-average_model) yönü, gecikmelerin mevcut ve geçmiş değerlerini gözlemleyerek belirlenen çıktı değişkenine atıfta bulunur.

Sonuç: ARIMA, zaman serisi verilerinin özel formuna mümkün olduğunca yakın bir model oluşturmak için kullanılır.

## Alıştırma - ARIMA Modeli Oluşturma

Bu dersteki [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) klasörünü açın ve [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb) dosyasını bulun.

1. ARIMA modelleri için gerekli olan `statsmodels` Python kütüphanesini yüklemek için notebook'u çalıştırın.

1. Gerekli kütüphaneleri yükleyin.

1. Şimdi, verileri görselleştirmek için birkaç kütüphane daha yükleyin:

    ```python
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import math

    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.preprocessing import MinMaxScaler
    from common.utils import load_data, mape
    from IPython.display import Image

    %matplotlib inline
    pd.options.display.float_format = '{:,.2f}'.format
    np.set_printoptions(precision=2)
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    ```

1. `/data/energy.csv` dosyasındaki verileri bir Pandas veri çerçevesine yükleyin ve inceleyin:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Ocak 2012'den Aralık 2014'e kadar mevcut tüm enerji verilerini görselleştirin. Bu verileri önceki derste gördüğümüz için sürpriz olmamalı:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Şimdi bir model oluşturalım!

### Eğitim ve Test Veri Setleri Oluşturma

Verileriniz yüklendi, şimdi bunları eğitim ve test setlerine ayırabilirsiniz. Modelinizi eğitim seti üzerinde eğiteceksiniz. Her zamanki gibi, model eğitimi tamamlandıktan sonra doğruluğunu test seti kullanarak değerlendireceksiniz. Modelin gelecekteki zaman dilimlerinden bilgi edinmemesini sağlamak için test setinin eğitim setinden daha sonraki bir zaman dilimini kapsaması gerekir.

1. Eğitim seti için 1 Eylül - 31 Ekim 2014 arasındaki iki aylık bir dönem ayırın. Test seti ise 1 Kasım - 31 Aralık 2014 arasındaki iki aylık dönemi içerecek:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Bu veriler günlük enerji tüketimini yansıttığı için güçlü bir mevsimsel desen vardır, ancak tüketim en çok son günlerdeki tüketime benzer.

1. Farklılıkları görselleştirin:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![eğitim ve test verileri](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Bu nedenle, verileri eğitmek için nispeten küçük bir zaman aralığı kullanmak yeterli olmalıdır.

    > Not: ARIMA modelini uyarlamak için kullandığımız fonksiyon, uyarlama sırasında örnek içi doğrulama kullandığından, doğrulama verilerini atlayacağız.

### Verileri Eğitime Hazırlama

Şimdi, verileri filtreleme ve ölçeklendirme işlemleri yaparak eğitime hazırlamanız gerekiyor. Veri setinizi yalnızca ihtiyaç duyduğunuz zaman dilimlerini ve sütunları içerecek şekilde filtreleyin ve verilerin 0,1 aralığında projeksiyonunu sağlamak için ölçeklendirme yapın.

1. Orijinal veri setini yalnızca belirtilen zaman dilimlerini ve yalnızca gerekli 'load' sütunu ile tarihi içerecek şekilde filtreleyin:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Verilerin şekline bakabilirsiniz:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Verileri (0, 1) aralığında ölçeklendirin.

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Orijinal ve ölçeklendirilmiş verileri görselleştirin:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![orijinal](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Orijinal veri

    ![ölçeklendirilmiş](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Ölçeklendirilmiş veri

1. Ölçeklendirilmiş verileri kalibre ettikten sonra test verilerini ölçeklendirebilirsiniz:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### ARIMA'yı Uygulama

Artık ARIMA'yı uygulama zamanı! Daha önce yüklediğiniz `statsmodels` kütüphanesini kullanacaksınız.

Şimdi birkaç adımı takip etmeniz gerekiyor:

   1. Modeli tanımlamak için `SARIMAX()` fonksiyonunu çağırın ve model parametrelerini (p, d, q ve P, D, Q parametreleri) geçirin.
   2. Modeli eğitim verileri için hazırlamak üzere `fit()` fonksiyonunu çağırın.
   3. Tahmin yapmak için `forecast()` fonksiyonunu çağırın ve tahmin edilecek adım sayısını (`horizon`) belirtin.

> 🎓 Bu parametreler ne işe yarar? Bir ARIMA modelinde, bir zaman serisinin ana yönlerini modellemeye yardımcı olmak için kullanılan 3 parametre vardır: mevsimsellik, eğilim ve gürültü. Bu parametreler şunlardır:

`p`: Modelin otoregresif yönüyle ilişkili parametre, *geçmiş* değerleri içerir.
`d`: Modelin entegre yönüyle ilişkili parametre, bir zaman serisine uygulanacak *fark alma* miktarını etkiler (🎓 fark alma 👆 hatırlıyor musunuz?).
`q`: Modelin hareketli ortalama yönüyle ilişkili parametre.

> Not: Verilerinizin mevsimsel bir yönü varsa - bu veri setinde olduğu gibi - mevsimsel ARIMA modeli (SARIMA) kullanırız. Bu durumda, `p`, `d` ve `q` ile aynı ilişkileri tanımlayan ancak modelin mevsimsel bileşenlerine karşılık gelen başka bir parametre seti (`P`, `D`, ve `Q`) kullanmanız gerekir.

1. Tercih ettiğiniz horizon değerini ayarlayarak başlayın. 3 saat deneyelim:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Bir ARIMA modelinin parametreleri için en iyi değerleri seçmek zor olabilir çünkü bu biraz öznel ve zaman alıcıdır. [`pyramid` kütüphanesinden](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) bir `auto_arima()` fonksiyonu kullanmayı düşünebilirsiniz.

1. Şimdilik iyi bir model bulmak için bazı manuel seçimler yapmayı deneyin.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Bir sonuç tablosu yazdırılır.

İlk modelinizi oluşturdunuz! Şimdi bunu değerlendirmek için bir yol bulmamız gerekiyor.

### Modelinizi Değerlendirin

Modelinizi değerlendirmek için, `walk forward` doğrulama adı verilen bir yöntem uygulayabilirsiniz. Pratikte, zaman serisi modelleri her yeni veri geldiğinde yeniden eğitilir. Bu, modelin her zaman adımında en iyi tahmini yapmasını sağlar.

Bu teknikle zaman serisinin başından başlayarak, modeli eğitim veri seti üzerinde eğitin. Ardından bir sonraki zaman adımında tahmin yapın. Tahmin, bilinen değerle karşılaştırılır. Eğitim seti, bilinen değeri içerecek şekilde genişletilir ve işlem tekrarlanır.

> Not: Daha verimli bir eğitim için eğitim seti penceresini sabit tutmalısınız, böylece her yeni gözlemi eğitim setine eklediğinizde, setin başlangıcındaki gözlemi kaldırırsınız.

Bu işlem, modelin pratikte nasıl performans göstereceğine dair daha sağlam bir tahmin sağlar. Ancak, bu kadar çok model oluşturmanın hesaplama maliyeti vardır. Veri küçükse veya model basitse bu kabul edilebilir, ancak büyük ölçeklerde sorun olabilir.

Walk-forward doğrulama, zaman serisi model değerlendirmesinin altın standardıdır ve kendi projelerinizde önerilir.

1. Her HORIZON adımı için bir test veri noktası oluşturun.

    ```python
    test_shifted = test.copy()

    for t in range(1, HORIZON+1):
        test_shifted['load+'+str(t)] = test_shifted['load'].shift(-t, freq='H')

    test_shifted = test_shifted.dropna(how='any')
    test_shifted.head(5)
    ```

    |            |          | load | load+1 | load+2 |
    | ---------- | -------- | ---- | ------ | ------ |
    | 2014-12-30 | 00:00:00 | 0.33 | 0.29   | 0.27   |
    | 2014-12-30 | 01:00:00 | 0.29 | 0.27   | 0.27   |
    | 2014-12-30 | 02:00:00 | 0.27 | 0.27   | 0.30   |
    | 2014-12-30 | 03:00:00 | 0.27 | 0.30   | 0.41   |
    | 2014-12-30 | 04:00:00 | 0.30 | 0.41   | 0.57   |

    Veri, horizon noktasına göre yatay olarak kaydırılır.

1. Test verilerinizde bu kaydırma pencere yaklaşımını kullanarak bir döngü içinde tahminler yapın:

    ```python
    %%time
    training_window = 720 # dedicate 30 days (720 hours) for training

    train_ts = train['load']
    test_ts = test_shifted

    history = [x for x in train_ts]
    history = history[(-training_window):]

    predictions = list()

    order = (2, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    for t in range(test_ts.shape[0]):
        model = SARIMAX(endog=history, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        yhat = model_fit.forecast(steps = HORIZON)
        predictions.append(yhat)
        obs = list(test_ts.iloc[t])
        # move the training window
        history.append(obs[0])
        history.pop(0)
        print(test_ts.index[t])
        print(t+1, ': predicted =', yhat, 'expected =', obs)
    ```

    Eğitim işlemini izleyebilirsiniz:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Tahminleri gerçek yük ile karşılaştırın:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Çıktı
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Saatlik verilerin tahminini, gerçek yük ile karşılaştırın. Ne kadar doğru?

### Model Doğruluğunu Kontrol Etme

Modelinizin doğruluğunu, tüm tahminler üzerindeki ortalama mutlak yüzde hatası (MAPE) ile test ederek kontrol edin.
> **🧮 Matematiği Göster**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) yukarıdaki formülle tanımlanan bir oran olarak tahmin doğruluğunu göstermek için kullanılır. Gerçek ve tahmin edilen arasındaki fark, gerçeğe bölünür. "Bu hesaplamadaki mutlak değer, her tahmin edilen zaman noktası için toplanır ve uydurulan noktaların sayısına (n) bölünür." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Kodda denklemi ifade et:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Bir adımın MAPE'sini hesapla:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    Bir adım tahmin MAPE:  0.5570581332313952 %

1. Çok adımlı tahmin MAPE'sini yazdır:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Düşük bir sayı en iyisidir: Unutmayın, MAPE değeri 10 olan bir tahmin %10 oranında yanlıştır.

1. Ancak her zaman olduğu gibi, bu tür bir doğruluk ölçümünü görsel olarak görmek daha kolaydır, hadi bunu çizelim:

    ```python
     if(HORIZON == 1):
        ## Plotting single step forecast
        eval_df.plot(x='timestamp', y=['actual', 'prediction'], style=['r', 'b'], figsize=(15, 8))

    else:
        ## Plotting multi step forecast
        plot_df = eval_df[(eval_df.h=='t+1')][['timestamp', 'actual']]
        for t in range(1, HORIZON+1):
            plot_df['t+'+str(t)] = eval_df[(eval_df.h=='t+'+str(t))]['prediction'].values

        fig = plt.figure(figsize=(15, 8))
        ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
        ax = fig.add_subplot(111)
        for t in range(1, HORIZON+1):
            x = plot_df['timestamp'][(t-1):]
            y = plot_df['t+'+str(t)][0:len(x)]
            ax.plot(x, y, color='blue', linewidth=4*math.pow(.9,t), alpha=math.pow(0.8,t))

        ax.legend(loc='best')

    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![bir zaman serisi modeli](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Çok güzel bir grafik, iyi doğruluğa sahip bir modeli gösteriyor. Tebrikler!

---

## 🚀Meydan Okuma

Bir Zaman Serisi Modelinin doğruluğunu test etmenin yollarını araştırın. Bu derste MAPE'ye değindik, ancak kullanabileceğiniz başka yöntemler var mı? Bunları araştırın ve notlar alın. Faydalı bir belge [burada](https://otexts.com/fpp2/accuracy.html) bulunabilir.

## [Ders sonrası test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu ders, ARIMA ile Zaman Serisi Tahmininin yalnızca temel konularına değiniyor. Bilginizi derinleştirmek için [bu depo](https://microsoft.github.io/forecasting/) ve çeşitli model türlerini inceleyerek Zaman Serisi modelleri oluşturmanın diğer yollarını öğrenmek için zaman ayırın.

## Ödev

[Yeni bir ARIMA modeli](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul edilmez.