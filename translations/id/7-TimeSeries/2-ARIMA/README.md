<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "917dbf890db71a322f306050cb284749",
  "translation_date": "2025-09-05T19:00:37+00:00",
  "source_file": "7-TimeSeries/2-ARIMA/README.md",
  "language_code": "id"
}
-->
# Peramalan Deret Waktu dengan ARIMA

Pada pelajaran sebelumnya, Anda telah mempelajari sedikit tentang peramalan deret waktu dan memuat dataset yang menunjukkan fluktuasi beban listrik selama periode waktu tertentu.

[![Pengantar ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Pengantar ARIMA")

> 🎥 Klik gambar di atas untuk video: Pengantar singkat tentang model ARIMA. Contoh dilakukan dalam R, tetapi konsepnya bersifat universal.

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Pengantar

Dalam pelajaran ini, Anda akan mempelajari cara spesifik untuk membangun model dengan [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average). Model ARIMA sangat cocok untuk data yang menunjukkan [non-stasioneritas](https://wikipedia.org/wiki/Stationary_process).

## Konsep Umum

Untuk dapat bekerja dengan ARIMA, ada beberapa konsep yang perlu Anda ketahui:

- 🎓 **Stasioneritas**. Dalam konteks statistik, stasioneritas mengacu pada data yang distribusinya tidak berubah ketika digeser dalam waktu. Data non-stasioner, sebaliknya, menunjukkan fluktuasi akibat tren yang harus diubah agar dapat dianalisis. Musim, misalnya, dapat memperkenalkan fluktuasi dalam data dan dapat dihilangkan melalui proses 'seasonal-differencing'.

- 🎓 **[Differencing](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing)**. Differencing data, dalam konteks statistik, mengacu pada proses mengubah data non-stasioner menjadi stasioner dengan menghilangkan tren yang tidak konstan. "Differencing menghilangkan perubahan tingkat dalam deret waktu, menghilangkan tren dan musiman, sehingga menstabilkan rata-rata deret waktu." [Makalah oleh Shixiong et al](https://arxiv.org/abs/1904.07632)

## ARIMA dalam Konteks Deret Waktu

Mari kita uraikan bagian-bagian ARIMA untuk lebih memahami bagaimana model ini membantu kita memodelkan deret waktu dan membuat prediksi terhadapnya.

- **AR - untuk AutoRegressive**. Model autoregresif, seperti namanya, melihat 'ke belakang' dalam waktu untuk menganalisis nilai-nilai sebelumnya dalam data Anda dan membuat asumsi tentangnya. Nilai-nilai sebelumnya ini disebut 'lags'. Contohnya adalah data yang menunjukkan penjualan bulanan pensil. Total penjualan setiap bulan akan dianggap sebagai 'variabel yang berkembang' dalam dataset. Model ini dibangun sebagai "variabel yang berkembang diregresikan pada nilai-nilai lagged (yaitu, sebelumnya) miliknya." [wikipedia](https://wikipedia.org/wiki/Autoregressive_integrated_moving_average)

- **I - untuk Integrated**. Berbeda dengan model 'ARMA' yang serupa, 'I' dalam ARIMA mengacu pada aspek *[integrated](https://wikipedia.org/wiki/Order_of_integration)*. Data 'diintegrasikan' ketika langkah-langkah differencing diterapkan untuk menghilangkan non-stasioneritas.

- **MA - untuk Moving Average**. Aspek [moving-average](https://wikipedia.org/wiki/Moving-average_model) dari model ini mengacu pada variabel output yang ditentukan dengan mengamati nilai lag saat ini dan sebelumnya.

Intinya: ARIMA digunakan untuk membuat model sesuai dengan bentuk khusus data deret waktu sedekat mungkin.

## Latihan - Membangun Model ARIMA

Buka folder [_/working_](https://github.com/microsoft/ML-For-Beginners/tree/main/7-TimeSeries/2-ARIMA/working) dalam pelajaran ini dan temukan file [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/7-TimeSeries/2-ARIMA/working/notebook.ipynb).

1. Jalankan notebook untuk memuat pustaka Python `statsmodels`; Anda akan membutuhkan ini untuk model ARIMA.

1. Muat pustaka yang diperlukan.

1. Sekarang, muat beberapa pustaka tambahan yang berguna untuk memplot data:

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

1. Muat data dari file `/data/energy.csv` ke dalam dataframe Pandas dan lihat:

    ```python
    energy = load_data('./data')[['load']]
    energy.head(10)
    ```

1. Plot semua data energi yang tersedia dari Januari 2012 hingga Desember 2014. Tidak ada kejutan karena kita telah melihat data ini di pelajaran sebelumnya:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    Sekarang, mari kita bangun model!

### Membuat Dataset Pelatihan dan Pengujian

Sekarang data Anda telah dimuat, Anda dapat memisahkannya menjadi set pelatihan dan pengujian. Anda akan melatih model Anda pada set pelatihan. Seperti biasa, setelah model selesai dilatih, Anda akan mengevaluasi akurasinya menggunakan set pengujian. Anda perlu memastikan bahwa set pengujian mencakup periode waktu yang lebih baru dari set pelatihan untuk memastikan bahwa model tidak mendapatkan informasi dari periode waktu di masa depan.

1. Alokasikan periode dua bulan dari 1 September hingga 31 Oktober 2014 untuk set pelatihan. Set pengujian akan mencakup periode dua bulan dari 1 November hingga 31 Desember 2014:

    ```python
    train_start_dt = '2014-11-01 00:00:00'
    test_start_dt = '2014-12-30 00:00:00'
    ```

    Karena data ini mencerminkan konsumsi energi harian, ada pola musiman yang kuat, tetapi konsumsi paling mirip dengan konsumsi pada hari-hari yang lebih baru.

1. Visualisasikan perbedaannya:

    ```python
    energy[(energy.index < test_start_dt) & (energy.index >= train_start_dt)][['load']].rename(columns={'load':'train'}) \
        .join(energy[test_start_dt:][['load']].rename(columns={'load':'test'}), how='outer') \
        .plot(y=['train', 'test'], figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![data pelatihan dan pengujian](../../../../7-TimeSeries/2-ARIMA/images/train-test.png)

    Oleh karena itu, menggunakan jendela waktu yang relatif kecil untuk melatih data seharusnya cukup.

    > Catatan: Karena fungsi yang kita gunakan untuk menyesuaikan model ARIMA menggunakan validasi in-sample selama fitting, kita akan menghilangkan data validasi.

### Mempersiapkan Data untuk Pelatihan

Sekarang, Anda perlu mempersiapkan data untuk pelatihan dengan melakukan penyaringan dan penskalaan data Anda. Saring dataset Anda untuk hanya menyertakan periode waktu dan kolom yang Anda butuhkan, serta penskalaan untuk memastikan data diproyeksikan dalam interval 0,1.

1. Saring dataset asli untuk hanya menyertakan periode waktu yang disebutkan sebelumnya per set dan hanya menyertakan kolom 'load' yang diperlukan plus tanggal:

    ```python
    train = energy.copy()[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']]
    test = energy.copy()[energy.index >= test_start_dt][['load']]

    print('Training data shape: ', train.shape)
    print('Test data shape: ', test.shape)
    ```

    Anda dapat melihat bentuk data:

    ```output
    Training data shape:  (1416, 1)
    Test data shape:  (48, 1)
    ```

1. Skala data agar berada dalam rentang (0, 1).

    ```python
    scaler = MinMaxScaler()
    train['load'] = scaler.fit_transform(train)
    train.head(10)
    ```

1. Visualisasikan data asli vs. data yang telah diskalakan:

    ```python
    energy[(energy.index >= train_start_dt) & (energy.index < test_start_dt)][['load']].rename(columns={'load':'original load'}).plot.hist(bins=100, fontsize=12)
    train.rename(columns={'load':'scaled load'}).plot.hist(bins=100, fontsize=12)
    plt.show()
    ```

    ![asli](../../../../7-TimeSeries/2-ARIMA/images/original.png)

    > Data asli

    ![diskalakan](../../../../7-TimeSeries/2-ARIMA/images/scaled.png)

    > Data yang telah diskalakan

1. Sekarang setelah Anda mengkalibrasi data yang telah diskalakan, Anda dapat menskalakan data pengujian:

    ```python
    test['load'] = scaler.transform(test)
    test.head()
    ```

### Menerapkan ARIMA

Saatnya menerapkan ARIMA! Anda sekarang akan menggunakan pustaka `statsmodels` yang telah Anda instal sebelumnya.

Sekarang Anda perlu mengikuti beberapa langkah:

   1. Definisikan model dengan memanggil `SARIMAX()` dan memasukkan parameter model: parameter p, d, dan q, serta parameter P, D, dan Q.
   2. Persiapkan model untuk data pelatihan dengan memanggil fungsi fit().
   3. Buat prediksi dengan memanggil fungsi `forecast()` dan menentukan jumlah langkah (horizon) untuk diprediksi.

> 🎓 Apa fungsi semua parameter ini? Dalam model ARIMA, ada 3 parameter yang digunakan untuk membantu memodelkan aspek utama dari deret waktu: musiman, tren, dan noise. Parameter ini adalah:

`p`: parameter yang terkait dengan aspek autoregresif model, yang menggabungkan nilai *masa lalu*.
`d`: parameter yang terkait dengan bagian terintegrasi dari model, yang memengaruhi jumlah *differencing* (🎓 ingat differencing 👆?) yang diterapkan pada deret waktu.
`q`: parameter yang terkait dengan bagian moving-average dari model.

> Catatan: Jika data Anda memiliki aspek musiman - seperti data ini -, kita menggunakan model ARIMA musiman (SARIMA). Dalam kasus ini, Anda perlu menggunakan satu set parameter lain: `P`, `D`, dan `Q` yang menggambarkan asosiasi yang sama seperti `p`, `d`, dan `q`, tetapi sesuai dengan komponen musiman dari model.

1. Mulailah dengan menetapkan nilai horizon yang Anda inginkan. Mari coba 3 jam:

    ```python
    # Specify the number of steps to forecast ahead
    HORIZON = 3
    print('Forecasting horizon:', HORIZON, 'hours')
    ```

    Memilih nilai terbaik untuk parameter model ARIMA bisa menjadi tantangan karena sifatnya yang subjektif dan memakan waktu. Anda mungkin mempertimbangkan menggunakan fungsi `auto_arima()` dari pustaka [`pyramid`](https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html).

1. Untuk saat ini, coba beberapa pilihan manual untuk menemukan model yang baik.

    ```python
    order = (4, 1, 0)
    seasonal_order = (1, 1, 0, 24)

    model = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    print(results.summary())
    ```

    Sebuah tabel hasil dicetak.

Anda telah membangun model pertama Anda! Sekarang kita perlu menemukan cara untuk mengevaluasinya.

### Mengevaluasi Model Anda

Untuk mengevaluasi model Anda, Anda dapat melakukan validasi `walk forward`. Dalam praktiknya, model deret waktu dilatih ulang setiap kali data baru tersedia. Ini memungkinkan model membuat prediksi terbaik pada setiap langkah waktu.

Mulai dari awal deret waktu menggunakan teknik ini, latih model pada set data pelatihan. Kemudian buat prediksi pada langkah waktu berikutnya. Prediksi dievaluasi terhadap nilai yang diketahui. Set pelatihan kemudian diperluas untuk menyertakan nilai yang diketahui dan proses diulang.

> Catatan: Anda harus menjaga jendela set pelatihan tetap tetap untuk pelatihan yang lebih efisien sehingga setiap kali Anda menambahkan pengamatan baru ke set pelatihan, Anda menghapus pengamatan dari awal set.

Proses ini memberikan estimasi yang lebih kuat tentang bagaimana model akan bekerja dalam praktik. Namun, ini datang dengan biaya komputasi untuk membuat begitu banyak model. Ini dapat diterima jika datanya kecil atau jika modelnya sederhana, tetapi bisa menjadi masalah dalam skala besar.

Validasi walk-forward adalah standar emas evaluasi model deret waktu dan direkomendasikan untuk proyek Anda sendiri.

1. Pertama, buat titik data pengujian untuk setiap langkah HORIZON.

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

    Data digeser secara horizontal sesuai dengan titik horizon-nya.

1. Buat prediksi pada data pengujian Anda menggunakan pendekatan jendela geser dalam loop sebesar panjang data pengujian:

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

    Anda dapat melihat pelatihan yang terjadi:

    ```output
    2014-12-30 00:00:00
    1 : predicted = [0.32 0.29 0.28] expected = [0.32945389435989236, 0.2900626678603402, 0.2739480752014323]

    2014-12-30 01:00:00
    2 : predicted = [0.3  0.29 0.3 ] expected = [0.2900626678603402, 0.2739480752014323, 0.26812891674127126]

    2014-12-30 02:00:00
    3 : predicted = [0.27 0.28 0.32] expected = [0.2739480752014323, 0.26812891674127126, 0.3025962399283795]
    ```

1. Bandingkan prediksi dengan beban aktual:

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

    Perhatikan prediksi data per jam dibandingkan dengan beban aktual. Seberapa akurat ini?

### Periksa Akurasi Model

Periksa akurasi model Anda dengan menguji mean absolute percentage error (MAPE) dari semua prediksi.
> **🧮 Tunjukkan perhitungannya**
>
> ![MAPE](../../../../7-TimeSeries/2-ARIMA/images/mape.png)
>
> [MAPE](https://www.linkedin.com/pulse/what-mape-mad-msd-time-series-allameh-statistics/) digunakan untuk menunjukkan akurasi prediksi sebagai rasio yang didefinisikan oleh rumus di atas. Perbedaan antara nilai aktual dan nilai prediksi dibagi dengan nilai aktual. "Nilai absolut dalam perhitungan ini dijumlahkan untuk setiap titik waktu yang diprediksi dan dibagi dengan jumlah titik yang sesuai n." [wikipedia](https://wikipedia.org/wiki/Mean_absolute_percentage_error)
1. Ekspresikan persamaan dalam kode:

    ```python
    if(HORIZON > 1):
        eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']
        print(eval_df.groupby('h')['APE'].mean())
    ```

1. Hitung MAPE untuk satu langkah:

    ```python
    print('One step forecast MAPE: ', (mape(eval_df[eval_df['h'] == 't+1']['prediction'], eval_df[eval_df['h'] == 't+1']['actual']))*100, '%')
    ```

    MAPE prediksi satu langkah:  0.5570581332313952 %

1. Cetak MAPE prediksi multi-langkah:

    ```python
    print('Multi-step forecast MAPE: ', mape(eval_df['prediction'], eval_df['actual'])*100, '%')
    ```

    ```output
    Multi-step forecast MAPE:  1.1460048657704118 %
    ```

    Angka yang rendah adalah yang terbaik: pertimbangkan bahwa prediksi dengan MAPE sebesar 10 berarti meleset sebesar 10%.

1. Namun seperti biasa, lebih mudah melihat pengukuran akurasi semacam ini secara visual, jadi mari kita plot:

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

    ![model deret waktu](../../../../7-TimeSeries/2-ARIMA/images/accuracy.png)

🏆 Plot yang sangat bagus, menunjukkan model dengan akurasi yang baik. Kerja bagus!

---

## 🚀Tantangan

Telusuri cara-cara untuk menguji akurasi model deret waktu. Kita membahas MAPE dalam pelajaran ini, tetapi apakah ada metode lain yang bisa digunakan? Lakukan penelitian dan beri anotasi. Dokumen yang bermanfaat dapat ditemukan [di sini](https://otexts.com/fpp2/accuracy.html)

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Pelajaran ini hanya menyentuh dasar-dasar Peramalan Deret Waktu dengan ARIMA. Luangkan waktu untuk memperdalam pengetahuan Anda dengan menjelajahi [repositori ini](https://microsoft.github.io/forecasting/) dan berbagai jenis modelnya untuk mempelajari cara lain membangun model Deret Waktu.

## Tugas

[Model ARIMA baru](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan manusia profesional. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.