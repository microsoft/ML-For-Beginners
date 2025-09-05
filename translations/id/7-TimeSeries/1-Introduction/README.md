<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T19:04:52+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "id"
}
-->
# Pengantar Peramalan Deret Waktu

![Ringkasan deret waktu dalam sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dalam pelajaran ini dan pelajaran berikutnya, Anda akan mempelajari sedikit tentang peramalan deret waktu, bagian yang menarik dan berharga dari repertoar seorang ilmuwan ML yang mungkin kurang dikenal dibandingkan topik lainnya. Peramalan deret waktu adalah semacam 'bola kristal': berdasarkan kinerja masa lalu dari suatu variabel seperti harga, Anda dapat memprediksi nilai potensialnya di masa depan.

[![Pengantar peramalan deret waktu](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Pengantar peramalan deret waktu")

> 🎥 Klik gambar di atas untuk video tentang peramalan deret waktu

## [Kuis pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

Ini adalah bidang yang berguna dan menarik dengan nilai nyata bagi bisnis, mengingat penerapannya langsung pada masalah harga, inventaris, dan rantai pasokan. Meskipun teknik pembelajaran mendalam mulai digunakan untuk mendapatkan wawasan lebih baik dalam memprediksi kinerja masa depan, peramalan deret waktu tetap menjadi bidang yang sangat dipengaruhi oleh teknik ML klasik.

> Kurikulum deret waktu yang berguna dari Penn State dapat ditemukan [di sini](https://online.stat.psu.edu/stat510/lesson/1)

## Pengantar

Misalkan Anda mengelola serangkaian meteran parkir pintar yang menyediakan data tentang seberapa sering mereka digunakan dan berapa lama selama periode waktu tertentu.

> Bagaimana jika Anda dapat memprediksi, berdasarkan kinerja masa lalu meteran tersebut, nilai masa depannya sesuai dengan hukum penawaran dan permintaan?

Memprediksi dengan akurat kapan harus bertindak untuk mencapai tujuan Anda adalah tantangan yang dapat diatasi dengan peramalan deret waktu. Meskipun mungkin tidak menyenangkan bagi orang-orang untuk dikenakan biaya lebih tinggi pada waktu sibuk saat mereka mencari tempat parkir, itu akan menjadi cara yang pasti untuk menghasilkan pendapatan guna membersihkan jalanan!

Mari kita eksplorasi beberapa jenis algoritma deret waktu dan mulai membuat notebook untuk membersihkan dan mempersiapkan data. Data yang akan Anda analisis diambil dari kompetisi peramalan GEFCom2014. Data ini terdiri dari 3 tahun nilai beban listrik dan suhu per jam antara tahun 2012 dan 2014. Berdasarkan pola historis beban listrik dan suhu, Anda dapat memprediksi nilai beban listrik di masa depan.

Dalam contoh ini, Anda akan belajar bagaimana meramalkan satu langkah waktu ke depan, hanya menggunakan data beban historis. Namun sebelum memulai, ada baiknya memahami apa yang terjadi di balik layar.

## Beberapa definisi

Saat menemukan istilah 'deret waktu', Anda perlu memahami penggunaannya dalam beberapa konteks berbeda.

🎓 **Deret waktu**

Dalam matematika, "deret waktu adalah serangkaian titik data yang diindeks (atau dicantumkan atau digrafikkan) dalam urutan waktu. Paling umum, deret waktu adalah urutan yang diambil pada titik-titik waktu yang berjarak sama secara berturut-turut." Contoh deret waktu adalah nilai penutupan harian [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Penggunaan plot deret waktu dan pemodelan statistik sering ditemukan dalam pemrosesan sinyal, peramalan cuaca, prediksi gempa bumi, dan bidang lain di mana peristiwa terjadi dan titik data dapat digrafikkan dari waktu ke waktu.

🎓 **Analisis deret waktu**

Analisis deret waktu adalah analisis data deret waktu yang disebutkan di atas. Data deret waktu dapat berbentuk berbeda, termasuk 'deret waktu terputus' yang mendeteksi pola dalam evolusi deret waktu sebelum dan setelah peristiwa yang mengganggu. Jenis analisis yang diperlukan untuk deret waktu bergantung pada sifat data. Data deret waktu itu sendiri dapat berupa serangkaian angka atau karakter.

Analisis yang dilakukan menggunakan berbagai metode, termasuk domain frekuensi dan domain waktu, linear dan non-linear, dan lainnya. [Pelajari lebih lanjut](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) tentang berbagai cara menganalisis jenis data ini.

🎓 **Peramalan deret waktu**

Peramalan deret waktu adalah penggunaan model untuk memprediksi nilai masa depan berdasarkan pola yang ditampilkan oleh data yang dikumpulkan sebelumnya saat terjadi di masa lalu. Meskipun dimungkinkan untuk menggunakan model regresi untuk mengeksplorasi data deret waktu, dengan indeks waktu sebagai variabel x pada plot, data semacam itu paling baik dianalisis menggunakan jenis model khusus.

Data deret waktu adalah daftar pengamatan yang terurut, berbeda dengan data yang dapat dianalisis dengan regresi linear. Model yang paling umum adalah ARIMA, singkatan dari "Autoregressive Integrated Moving Average".

[Model ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "menghubungkan nilai saat ini dari suatu deret dengan nilai masa lalu dan kesalahan prediksi masa lalu." Model ini paling cocok untuk menganalisis data domain waktu, di mana data diurutkan berdasarkan waktu.

> Ada beberapa jenis model ARIMA, yang dapat Anda pelajari [di sini](https://people.duke.edu/~rnau/411arim.htm) dan yang akan Anda bahas dalam pelajaran berikutnya.

Dalam pelajaran berikutnya, Anda akan membangun model ARIMA menggunakan [Deret Waktu Univariat](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), yang berfokus pada satu variabel yang nilainya berubah dari waktu ke waktu. Contoh jenis data ini adalah [dataset ini](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) yang mencatat konsentrasi CO2 bulanan di Observatorium Mauna Loa:

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

✅ Identifikasi variabel yang berubah dari waktu ke waktu dalam dataset ini

## Karakteristik data deret waktu yang perlu dipertimbangkan

Saat melihat data deret waktu, Anda mungkin memperhatikan bahwa data tersebut memiliki [karakteristik tertentu](https://online.stat.psu.edu/stat510/lesson/1/1.1) yang perlu Anda perhatikan dan kurangi untuk lebih memahami polanya. Jika Anda menganggap data deret waktu sebagai potensi memberikan 'sinyal' yang ingin Anda analisis, karakteristik ini dapat dianggap sebagai 'gangguan'. Anda sering kali perlu mengurangi 'gangguan' ini dengan mengimbangi beberapa karakteristik ini menggunakan teknik statistik tertentu.

Berikut adalah beberapa konsep yang perlu Anda ketahui untuk dapat bekerja dengan deret waktu:

🎓 **Tren**

Tren didefinisikan sebagai peningkatan dan penurunan yang dapat diukur dari waktu ke waktu. [Baca lebih lanjut](https://machinelearningmastery.com/time-series-trends-in-python). Dalam konteks deret waktu, ini tentang bagaimana menggunakan dan, jika perlu, menghilangkan tren dari deret waktu Anda.

🎓 **[Musiman](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Musiman didefinisikan sebagai fluktuasi periodik, seperti lonjakan liburan yang mungkin memengaruhi penjualan, misalnya. [Lihat](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) bagaimana berbagai jenis plot menampilkan musiman dalam data.

🎓 **Pencilan**

Pencilan adalah data yang jauh dari variansi standar.

🎓 **Siklus jangka panjang**

Terlepas dari musiman, data mungkin menunjukkan siklus jangka panjang seperti penurunan ekonomi yang berlangsung lebih dari satu tahun.

🎓 **Variansi konstan**

Seiring waktu, beberapa data menunjukkan fluktuasi konstan, seperti penggunaan energi per hari dan malam.

🎓 **Perubahan mendadak**

Data mungkin menunjukkan perubahan mendadak yang mungkin memerlukan analisis lebih lanjut. Penutupan bisnis secara tiba-tiba akibat COVID, misalnya, menyebabkan perubahan dalam data.

✅ Berikut adalah [contoh plot deret waktu](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) yang menunjukkan pengeluaran mata uang dalam game harian selama beberapa tahun. Bisakah Anda mengidentifikasi salah satu karakteristik yang tercantum di atas dalam data ini?

![Pengeluaran mata uang dalam game](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Latihan - memulai dengan data penggunaan daya

Mari kita mulai membuat model deret waktu untuk memprediksi penggunaan daya di masa depan berdasarkan penggunaan masa lalu.

> Data dalam contoh ini diambil dari kompetisi peramalan GEFCom2014. Data ini terdiri dari 3 tahun nilai beban listrik dan suhu per jam antara tahun 2012 dan 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli dan Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, Juli-September, 2016.

1. Di folder `working` pelajaran ini, buka file _notebook.ipynb_. Mulailah dengan menambahkan pustaka yang akan membantu Anda memuat dan memvisualisasikan data

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Perhatikan, Anda menggunakan file dari folder `common` yang disertakan yang mengatur lingkungan Anda dan menangani pengunduhan data.

2. Selanjutnya, periksa data sebagai dataframe dengan memanggil `load_data()` dan `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Anda dapat melihat bahwa ada dua kolom yang mewakili tanggal dan beban:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Sekarang, plot data dengan memanggil `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![plot energi](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Sekarang, plot minggu pertama Juli 2014, dengan memberikannya sebagai input ke `energy` dalam pola `[dari tanggal]: [ke tanggal]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![juli](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Plot yang indah! Lihatlah plot ini dan lihat apakah Anda dapat menentukan salah satu karakteristik yang tercantum di atas. Apa yang dapat kita simpulkan dengan memvisualisasikan data?

Dalam pelajaran berikutnya, Anda akan membuat model ARIMA untuk membuat beberapa prediksi.

---

## 🚀Tantangan

Buat daftar semua industri dan bidang penyelidikan yang dapat Anda pikirkan yang akan mendapat manfaat dari peramalan deret waktu. Bisakah Anda memikirkan penerapan teknik ini dalam seni? Dalam Ekonometrika? Ekologi? Ritel? Industri? Keuangan? Di mana lagi?

## [Kuis pasca-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Meskipun kami tidak akan membahasnya di sini, jaringan saraf terkadang digunakan untuk meningkatkan metode klasik peramalan deret waktu. Baca lebih lanjut tentang mereka [dalam artikel ini](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tugas

[Visualisasikan lebih banyak deret waktu](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.