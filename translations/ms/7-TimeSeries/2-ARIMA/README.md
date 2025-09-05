<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T19:01:16+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "ms"
}
-->
# Ramalan siri masa dengan ARIMA

Dalam pelajaran sebelumnya, anda telah mempelajari sedikit tentang ramalan siri masa dan memuatkan dataset yang menunjukkan turun naik beban elektrik sepanjang tempoh masa.

[![Pengenalan kepada ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Pengenalan kepada ARIMA")

> 🎥 Klik imej di atas untuk video: Pengenalan ringkas kepada model ARIMA. Contoh dilakukan dalam R, tetapi konsepnya adalah universal.

## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Pengenalan

Dalam pelajaran ini, anda akan meneroka cara khusus untuk membina model dengan [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Model ARIMA sangat sesuai untuk data yang menunjukkan [ketidakstasioner](https://wikipedia.org/wiki/Stationary_process).

## Konsep umum

Untuk bekerja dengan ARIMA, terdapat beberapa konsep yang perlu anda ketahui:

- 🎓 **Stasioneriti**. Dalam konteks statistik, stasioneriti merujuk kepada data yang taburannya tidak berubah apabila dialihkan dalam masa. Data yang tidak stasioner menunjukkan turun naik akibat trend yang perlu diubah untuk dianalisis. Musim, sebagai contoh, boleh memperkenalkan turun naik dalam data dan boleh dihapuskan melalui proses 'seasonal-differencing'.

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differencing data, sekali lagi dalam konteks statistik, merujuk kepada proses mengubah data yang tidak stasioner untuk menjadikannya stasioner dengan menghapuskan trend yang tidak tetap. "Differencing menghapuskan perubahan dalam tahap siri masa, menghapuskan trend dan musim, dan seterusnya menstabilkan purata siri masa." [Kertas oleh Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA dalam konteks siri masa

Mari kita pecahkan bahagian ARIMA untuk memahami dengan lebih baik bagaimana ia membantu kita memodelkan siri masa dan membuat ramalan terhadapnya.

- **AR - untuk AutoRegressive**. Model autoregressive, seperti namanya, melihat 'ke belakang' dalam masa untuk menganalisis nilai sebelumnya dalam data anda dan membuat andaian mengenainya. Nilai sebelumnya ini dipanggil 'lags'. Contohnya ialah data yang menunjukkan jualan bulanan pensel. Jumlah jualan setiap bulan akan dianggap sebagai 'pembolehubah yang berkembang' dalam dataset. Model ini dibina sebagai "pembolehubah yang berkembang diregreskan pada nilai lagged (iaitu, sebelumnya) sendiri." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - untuk Integrated**. Berbeza dengan model 'ARMA' yang serupa, 'I' dalam ARIMA merujuk kepada aspek *[integrated](https://wikipedia.org/wiki/Order_of_integration)*. Data 'diintegrasikan' apabila langkah-langkah differencing digunakan untuk menghapuskan ketidakstasioner.

- **MA - untuk Moving Average**. Aspek [moving-average](https://wikipedia.org/wiki/Moving-average_model) model ini merujuk kepada pembolehubah output yang ditentukan dengan memerhatikan nilai lag semasa dan sebelumnya.

Kesimpulan: ARIMA digunakan untuk membuat model sesuai dengan bentuk khas data siri masa sebaik mungkin.

## Latihan - bina model ARIMA

Buka folder [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) dalam pelajaran ini dan cari fail [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Jalankan notebook untuk memuatkan pustaka Python `statsmodels`; anda akan memerlukan ini untuk model ARIMA.

1. Muatkan pustaka yang diperlukan.

1. Sekarang, muatkan beberapa pustaka lagi yang berguna untuk memplot data:

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

1. Muatkan data dari fail `/data/energy.csv` ke dalam dataframe Pandas dan lihat:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plot semua data tenaga yang tersedia dari Januari 2012 hingga Disember 2014. Tidak ada kejutan kerana kita telah melihat data ini dalam pelajaran terakhir:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Sekarang, mari kita bina model!

### Cipta dataset latihan dan ujian

Sekarang data anda telah dimuatkan, anda boleh memisahkannya kepada set latihan dan ujian. Anda akan melatih model anda pada set latihan. Seperti biasa, selepas model selesai dilatih, anda akan menilai ketepatannya menggunakan set ujian. Anda perlu memastikan bahawa set ujian meliputi tempoh masa yang lebih lewat daripada set latihan untuk memastikan model tidak mendapat maklumat daripada tempoh masa akan datang.

1. Peruntukkan tempoh dua bulan dari 1 September hingga 31 Oktober 2014 kepada set latihan. Set ujian akan merangkumi tempoh dua bulan dari 1 November hingga 31 Disember 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Oleh kerana data ini mencerminkan penggunaan tenaga harian, terdapat corak bermusim yang kuat, tetapi penggunaan paling serupa dengan penggunaan pada hari-hari yang lebih baru.

1. Visualisasikan perbezaan:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![data latihan dan ujian](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Oleh itu, menggunakan tetingkap masa yang agak kecil untuk melatih data sepatutnya mencukupi.

    > Nota: Oleh kerana fungsi yang kita gunakan untuk memuatkan model ARIMA menggunakan pengesahan dalam-sampel semasa pemuatan, kita akan mengabaikan data pengesahan.

### Sediakan data untuk latihan

Sekarang, anda perlu menyediakan data untuk latihan dengan melakukan penapisan dan penskalaan data anda. Tapis dataset anda untuk hanya memasukkan tempoh masa dan lajur yang diperlukan, dan skala untuk memastikan data diproyeksikan dalam selang 0,1.

1. Tapis dataset asal untuk hanya memasukkan tempoh masa yang disebutkan sebelum ini bagi setiap set dan hanya memasukkan lajur 'load' yang diperlukan serta tarikh:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Anda boleh melihat bentuk data:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skala data untuk berada dalam julat (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisasikan data asal vs. data yang telah diskala:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![asal](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Data asal

    ![diskala](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Data yang telah diskala

1. Sekarang setelah anda menentukur data yang telah diskala, anda boleh menskalakan data ujian:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Laksanakan ARIMA

Sudah tiba masanya untuk melaksanakan ARIMA! Anda kini akan menggunakan pustaka `statsmodels` yang telah anda pasang sebelum ini.

Sekarang anda perlu mengikuti beberapa langkah:

   1. Tentukan model dengan memanggil `SARIMAX()` dan memasukkan parameter model: parameter p, d, dan q, serta parameter P, D, dan Q.
   2. Sediakan model untuk data latihan dengan memanggil fungsi fit().
   3. Buat ramalan dengan memanggil fungsi `forecast()` dan menentukan bilangan langkah (horizon) untuk diramalkan.

> 🎓 Apakah semua parameter ini? Dalam model ARIMA terdapat 3 parameter yang digunakan untuk membantu memodelkan aspek utama siri masa: musim, trend, dan bunyi. Parameter ini adalah:

`p`: parameter yang berkaitan dengan aspek auto-regressive model, yang menggabungkan nilai *masa lalu*.
`d`: parameter yang berkaitan dengan bahagian integrated model, yang mempengaruhi jumlah *differencing* (🎓 ingat differencing 👆?) yang digunakan pada siri masa.
`q`: parameter yang berkaitan dengan bahagian moving-average model.

> Nota: Jika data anda mempunyai aspek bermusim - seperti data ini -, kita menggunakan model ARIMA bermusim (SARIMA). Dalam kes ini, anda perlu menggunakan satu set parameter lain: `P`, `D`, dan `Q` yang menerangkan hubungan yang sama seperti `p`, `d`, dan `q`, tetapi berkaitan dengan komponen bermusim model.

1. Mulakan dengan menetapkan nilai horizon pilihan anda. Mari cuba 3 jam:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Memilih nilai terbaik untuk parameter model ARIMA boleh menjadi cabaran kerana ia agak subjektif dan memakan masa. Anda mungkin mempertimbangkan untuk menggunakan fungsi `auto_arima()` daripada pustaka [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Buat masa ini cuba beberapa pilihan manual untuk mencari model yang baik.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Jadual hasil dicetak.

Anda telah membina model pertama anda! Sekarang kita perlu mencari cara untuk menilainya.

### Nilai model anda

Untuk menilai model anda, anda boleh melakukan pengesahan `walk forward`. Dalam praktiknya, model siri masa dilatih semula setiap kali data baru tersedia. Ini membolehkan model membuat ramalan terbaik pada setiap langkah masa.

Bermula dari awal siri masa menggunakan teknik ini, latih model pada set data latihan. Kemudian buat ramalan pada langkah masa seterusnya. Ramalan dinilai terhadap nilai yang diketahui. Set latihan kemudian diperluaskan untuk memasukkan nilai yang diketahui dan proses diulang.

> Nota: Anda harus mengekalkan tetingkap set latihan tetap untuk latihan yang lebih cekap supaya setiap kali anda menambah pemerhatian baru ke set latihan, anda menghapuskan pemerhatian dari permulaan set.

Proses ini memberikan anggaran yang lebih kukuh tentang bagaimana model akan berfungsi dalam praktik. Walau bagaimanapun, ia datang dengan kos pengiraan untuk mencipta begitu banyak model. Ini boleh diterima jika data kecil atau jika model mudah, tetapi boleh menjadi isu pada skala besar.

Pengesahan walk-forward adalah standard emas untuk penilaian model siri masa dan disyorkan untuk projek anda sendiri.

1. Pertama, cipta titik data ujian untuk setiap langkah HORIZON.

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

    Data dialihkan secara mendatar mengikut titik horizon.

1. Buat ramalan pada data ujian anda menggunakan pendekatan tetingkap gelongsor dalam gelung sepanjang panjang data ujian:

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

    Anda boleh melihat latihan berlaku:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Bandingkan ramalan dengan beban sebenar:

    ```python
    eval_df = pd.DataFrame(predictions, columns=['t+'+str(t) for t in range(1, HORIZON+1)])
    eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
    eval_df = pd.melt(eval_df, id_vars='timestamp', value_name='prediction', var_name='h')
    eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
    eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_df[['prediction', 'actual']])
    eval_df.head()
    ```

    Output
    |     |            | timestamp | h   | prediction | actual   |
    | --- | ---------- | --------- | --- | ---------- | -------- |
    | 0   | 2014-12-30 | 00:00:00  | t+1 | 3,008.74   | 3,023.00 |
    | 1   | 2014-12-30 | 01:00:00  | t+1 | 2,955.53   | 2,935.00 |
    | 2   | 2014-12-30 | 02:00:00  | t+1 | 2,900.17   | 2,899.00 |
    | 3   | 2014-12-30 | 03:00:00  | t+1 | 2,917.69   | 2,886.00 |
    | 4   | 2014-12-30 | 04:00:00  | t+1 | 2,946.99   | 2,963.00 |

    Perhatikan ramalan data setiap jam, dibandingkan dengan beban sebenar. Seberapa tepatkah ini?

### Periksa ketepatan model

Periksa ketepatan model anda dengan menguji ralat peratusan mutlak purata (MAPE) model tersebut ke atas semua ramalan.
> **🧮 Tunjukkan matematik**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) digunakan untuk menunjukkan ketepatan ramalan sebagai nisbah yang ditakrifkan oleh formula di atas. Perbezaan antara nilai sebenar 

dan nilai ramalan 

dibahagikan dengan nilai sebenar. "Nilai mutlak dalam pengiraan ini dijumlahkan untuk setiap titik ramalan dalam masa dan dibahagikan dengan bilangan titik yang sesuai n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Ungkapkan persamaan dalam kod:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Kira MAPE untuk satu langkah:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE ramalan satu langkah:  0.5570581332313952 %

1. Cetak MAPE ramalan berbilang langkah:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Nombor yang rendah adalah terbaik: pertimbangkan bahawa ramalan dengan MAPE 10 adalah tersasar sebanyak 10%.

1. Tetapi seperti biasa, lebih mudah untuk melihat ukuran ketepatan seperti ini secara visual, jadi mari kita plotkan:

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

    ![model siri masa](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Plot yang sangat baik, menunjukkan model dengan ketepatan yang bagus. Syabas!

---

## 🚀Cabaran

Selidiki cara-cara untuk menguji ketepatan Model Siri Masa. Kita menyentuh tentang MAPE dalam pelajaran ini, tetapi adakah kaedah lain yang boleh digunakan? Lakukan penyelidikan dan buat anotasi. Dokumen yang berguna boleh didapati [di sini](https://otexts.com/fpp2/accuracy.html)

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Pelajaran ini hanya menyentuh asas Peramalan Siri Masa dengan ARIMA. Luangkan masa untuk mendalami pengetahuan anda dengan menyelidiki [repositori ini](https://microsoft.github.io/forecasting/) dan pelbagai jenis modelnya untuk mempelajari cara lain membina model Siri Masa.

## Tugasan

[Model ARIMA baharu](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.