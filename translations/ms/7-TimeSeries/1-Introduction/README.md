<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T19:05:23+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada ramalan siri masa

![Ringkasan siri masa dalam sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dalam pelajaran ini dan pelajaran berikutnya, anda akan mempelajari sedikit tentang ramalan siri masa, satu bahagian menarik dan bernilai dalam repertoir seorang saintis ML yang kurang dikenali berbanding topik lain. Ramalan siri masa adalah seperti 'bola kristal': berdasarkan prestasi masa lalu sesuatu pemboleh ubah seperti harga, anda boleh meramalkan nilai potensinya di masa depan.

[![Pengenalan kepada ramalan siri masa](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Pengenalan kepada ramalan siri masa")

> 🎥 Klik imej di atas untuk video tentang ramalan siri masa

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

Ia adalah bidang yang berguna dan menarik dengan nilai sebenar kepada perniagaan, memandangkan aplikasinya secara langsung kepada masalah harga, inventori, dan isu rantaian bekalan. Walaupun teknik pembelajaran mendalam telah mula digunakan untuk mendapatkan lebih banyak wawasan bagi meramalkan prestasi masa depan, ramalan siri masa kekal sebagai bidang yang banyak dipengaruhi oleh teknik ML klasik.

> Kurikulum siri masa yang berguna dari Penn State boleh didapati [di sini](https://online.stat.psu.edu/stat510/lesson/1)

## Pengenalan

Bayangkan anda menguruskan rangkaian meter parkir pintar yang menyediakan data tentang kekerapan ia digunakan dan untuk berapa lama sepanjang masa.

> Bagaimana jika anda boleh meramalkan, berdasarkan prestasi masa lalu meter tersebut, nilai masa depannya mengikut undang-undang penawaran dan permintaan?

Meramalkan dengan tepat bila untuk bertindak bagi mencapai matlamat anda adalah cabaran yang boleh ditangani oleh ramalan siri masa. Ia mungkin tidak menggembirakan orang ramai apabila dikenakan bayaran lebih tinggi pada waktu sibuk ketika mereka mencari tempat parkir, tetapi ia pasti cara untuk menjana pendapatan bagi membersihkan jalan!

Mari kita terokai beberapa jenis algoritma siri masa dan mulakan buku nota untuk membersihkan dan menyediakan data. Data yang akan anda analisis diambil daripada pertandingan ramalan GEFCom2014. Ia terdiri daripada 3 tahun nilai beban elektrik dan suhu setiap jam antara tahun 2012 dan 2014. Berdasarkan corak sejarah beban elektrik dan suhu, anda boleh meramalkan nilai masa depan beban elektrik.

Dalam contoh ini, anda akan belajar bagaimana untuk meramalkan satu langkah masa ke hadapan, menggunakan data beban sejarah sahaja. Sebelum memulakan, bagaimanapun, adalah berguna untuk memahami apa yang berlaku di belakang tabir.

## Beberapa definisi

Apabila menemui istilah 'siri masa', anda perlu memahami penggunaannya dalam beberapa konteks yang berbeza.

🎓 **Siri masa**

Dalam matematik, "siri masa adalah satu siri titik data yang diindeks (atau disenaraikan atau diplotkan) mengikut susunan masa. Selalunya, siri masa adalah urutan yang diambil pada titik masa yang sama jaraknya." Contoh siri masa ialah nilai penutupan harian [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Penggunaan plot siri masa dan pemodelan statistik sering ditemui dalam pemprosesan isyarat, ramalan cuaca, ramalan gempa bumi, dan bidang lain di mana peristiwa berlaku dan titik data boleh diplotkan sepanjang masa.

🎓 **Analisis siri masa**

Analisis siri masa adalah analisis data siri masa yang disebutkan di atas. Data siri masa boleh mengambil bentuk yang berbeza, termasuk 'siri masa terganggu' yang mengesan corak dalam evolusi siri masa sebelum dan selepas peristiwa yang mengganggu. Jenis analisis yang diperlukan untuk siri masa bergantung pada sifat data. Data siri masa itu sendiri boleh berbentuk siri nombor atau aksara.

Analisis yang akan dilakukan menggunakan pelbagai kaedah, termasuk domain frekuensi dan domain masa, linear dan tidak linear, dan banyak lagi. [Ketahui lebih lanjut](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) tentang pelbagai cara untuk menganalisis jenis data ini.

🎓 **Ramalan siri masa**

Ramalan siri masa adalah penggunaan model untuk meramalkan nilai masa depan berdasarkan corak yang ditunjukkan oleh data yang dikumpulkan sebelumnya seperti yang berlaku pada masa lalu. Walaupun mungkin menggunakan model regresi untuk meneroka data siri masa, dengan indeks masa sebagai pemboleh ubah x pada plot, data sedemikian paling baik dianalisis menggunakan jenis model khas.

Data siri masa adalah senarai pemerhatian yang teratur, tidak seperti data yang boleh dianalisis oleh regresi linear. Yang paling biasa ialah ARIMA, akronim yang bermaksud "Autoregressive Integrated Moving Average".

[Model ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "mengaitkan nilai semasa siri dengan nilai masa lalu dan kesilapan ramalan masa lalu." Ia paling sesuai untuk menganalisis data domain masa, di mana data diatur mengikut masa.

> Terdapat beberapa jenis model ARIMA, yang boleh anda pelajari [di sini](https://people.duke.edu/~rnau/411arim.htm) dan yang akan anda sentuh dalam pelajaran seterusnya.

Dalam pelajaran seterusnya, anda akan membina model ARIMA menggunakan [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), yang memberi tumpuan kepada satu pemboleh ubah yang mengubah nilainya sepanjang masa. Contoh jenis data ini ialah [set data ini](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) yang merekodkan kepekatan C02 bulanan di Observatori Mauna Loa:

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

✅ Kenal pasti pemboleh ubah yang berubah sepanjang masa dalam set data ini

## Ciri-ciri data siri masa yang perlu dipertimbangkan

Apabila melihat data siri masa, anda mungkin perasan bahawa ia mempunyai [ciri-ciri tertentu](https://online.stat.psu.edu/stat510/lesson/1/1.1) yang perlu anda ambil kira dan kurangkan untuk memahami coraknya dengan lebih baik. Jika anda menganggap data siri masa sebagai berpotensi memberikan 'isyarat' yang ingin anda analisis, ciri-ciri ini boleh dianggap sebagai 'gangguan'. Anda sering perlu mengurangkan 'gangguan' ini dengan mengimbangi beberapa ciri ini menggunakan teknik statistik tertentu.

Berikut adalah beberapa konsep yang perlu anda ketahui untuk dapat bekerja dengan siri masa:

🎓 **Trend**

Trend ditakrifkan sebagai peningkatan dan penurunan yang boleh diukur sepanjang masa. [Baca lebih lanjut](https://machinelearningmastery.com/time-series-trends-in-python). Dalam konteks siri masa, ia berkaitan dengan cara menggunakan dan, jika perlu, menghapuskan trend daripada siri masa anda.

🎓 **[Musim](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Musim ditakrifkan sebagai turun naik berkala, seperti lonjakan jualan semasa musim perayaan, contohnya. [Lihat](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) bagaimana jenis plot yang berbeza memaparkan musim dalam data.

🎓 **Nilai luar**

Nilai luar adalah jauh daripada varians data standard.

🎓 **Kitaran jangka panjang**

Bebas daripada musim, data mungkin memaparkan kitaran jangka panjang seperti kemerosotan ekonomi yang berlangsung lebih lama daripada setahun.

🎓 **Varians tetap**

Sepanjang masa, sesetengah data memaparkan turun naik tetap, seperti penggunaan tenaga setiap hari dan malam.

🎓 **Perubahan mendadak**

Data mungkin memaparkan perubahan mendadak yang mungkin memerlukan analisis lanjut. Penutupan perniagaan secara mendadak akibat COVID, contohnya, menyebabkan perubahan dalam data.

✅ Berikut adalah [plot siri masa sampel](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) yang menunjukkan perbelanjaan mata wang dalam permainan setiap hari selama beberapa tahun. Bolehkah anda mengenal pasti mana-mana ciri yang disenaraikan di atas dalam data ini?

![Perbelanjaan mata wang dalam permainan](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Latihan - bermula dengan data penggunaan kuasa

Mari kita mulakan dengan mencipta model siri masa untuk meramalkan penggunaan kuasa masa depan berdasarkan penggunaan masa lalu.

> Data dalam contoh ini diambil daripada pertandingan ramalan GEFCom2014. Ia terdiri daripada 3 tahun nilai beban elektrik dan suhu setiap jam antara tahun 2012 dan 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli dan Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. Dalam folder `working` pelajaran ini, buka fail _notebook.ipynb_. Mulakan dengan menambah perpustakaan yang akan membantu anda memuatkan dan memvisualisasikan data

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Perhatikan, anda menggunakan fail dari folder `common` yang disertakan yang menyediakan persekitaran anda dan mengendalikan muat turun data.

2. Seterusnya, periksa data sebagai dataframe dengan memanggil `load_data()` dan `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Anda boleh melihat bahawa terdapat dua lajur yang mewakili tarikh dan beban:

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

    ![plot tenaga](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Sekarang, plot minggu pertama Julai 2014, dengan memberikannya sebagai input kepada `energy` dalam pola `[dari tarikh]: [ke tarikh]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![july](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Plot yang cantik! Lihat plot ini dan lihat jika anda boleh menentukan mana-mana ciri yang disenaraikan di atas. Apa yang boleh kita simpulkan dengan memvisualisasikan data?

Dalam pelajaran seterusnya, anda akan mencipta model ARIMA untuk menghasilkan beberapa ramalan.

---

## 🚀Cabaran

Buat senarai semua industri dan bidang penyelidikan yang anda boleh fikirkan yang akan mendapat manfaat daripada ramalan siri masa. Bolehkah anda memikirkan aplikasi teknik ini dalam seni? Dalam Ekonometrik? Ekologi? Runcit? Industri? Kewangan? Di mana lagi?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Walaupun kita tidak akan membincangkannya di sini, rangkaian neural kadangkala digunakan untuk meningkatkan kaedah klasik ramalan siri masa. Baca lebih lanjut mengenainya [dalam artikel ini](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Tugasan

[Visualisasikan lebih banyak siri masa](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.