<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T18:55:51+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "id"
}
-->
# Membangun Model Regresi Menggunakan Scikit-learn: Persiapkan dan Visualisasikan Data

![Infografis visualisasi data](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografis oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Pendahuluan

Setelah Anda memiliki alat yang diperlukan untuk mulai membangun model pembelajaran mesin dengan Scikit-learn, Anda siap untuk mulai mengajukan pertanyaan terhadap data Anda. Saat bekerja dengan data dan menerapkan solusi ML, sangat penting untuk memahami cara mengajukan pertanyaan yang tepat agar dapat memanfaatkan potensi dataset Anda dengan benar.

Dalam pelajaran ini, Anda akan belajar:

- Cara mempersiapkan data Anda untuk membangun model.
- Cara menggunakan Matplotlib untuk visualisasi data.

## Mengajukan Pertanyaan yang Tepat pada Data Anda

Pertanyaan yang ingin Anda jawab akan menentukan jenis algoritma ML yang akan Anda gunakan. Kualitas jawaban yang Anda dapatkan sangat bergantung pada sifat data Anda.

Lihatlah [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) yang disediakan untuk pelajaran ini. Anda dapat membuka file .csv ini di VS Code. Sekilas, Anda akan segera melihat bahwa ada data kosong dan campuran antara string dan data numerik. Ada juga kolom aneh bernama 'Package' di mana datanya adalah campuran antara 'sacks', 'bins', dan nilai lainnya. Data ini, sebenarnya, cukup berantakan.

[![ML untuk Pemula - Cara Menganalisis dan Membersihkan Dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML untuk Pemula - Cara Menganalisis dan Membersihkan Dataset")

> 🎥 Klik gambar di atas untuk video singkat tentang persiapan data untuk pelajaran ini.

Faktanya, sangat jarang mendapatkan dataset yang sepenuhnya siap digunakan untuk membuat model ML langsung. Dalam pelajaran ini, Anda akan belajar cara mempersiapkan dataset mentah menggunakan pustaka Python standar. Anda juga akan mempelajari berbagai teknik untuk memvisualisasikan data.

## Studi Kasus: 'Pasar Labu'

Dalam folder ini, Anda akan menemukan file .csv di folder root `data` bernama [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) yang mencakup 1757 baris data tentang pasar labu, yang dikelompokkan berdasarkan kota. Ini adalah data mentah yang diekstrak dari [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) yang didistribusikan oleh Departemen Pertanian Amerika Serikat.

### Mempersiapkan Data

Data ini berada di domain publik. Data ini dapat diunduh dalam banyak file terpisah, per kota, dari situs web USDA. Untuk menghindari terlalu banyak file terpisah, kami telah menggabungkan semua data kota ke dalam satu spreadsheet, sehingga kami telah _mempersiapkan_ data sedikit. Selanjutnya, mari kita lihat lebih dekat data tersebut.

### Data Labu - Kesimpulan Awal

Apa yang Anda perhatikan tentang data ini? Anda sudah melihat bahwa ada campuran string, angka, data kosong, dan nilai aneh yang perlu Anda pahami.

Pertanyaan apa yang dapat Anda ajukan pada data ini menggunakan teknik Regresi? Bagaimana dengan "Memprediksi harga labu yang dijual selama bulan tertentu". Melihat kembali data tersebut, ada beberapa perubahan yang perlu Anda lakukan untuk membuat struktur data yang diperlukan untuk tugas ini.

## Latihan - Analisis Data Labu

Mari gunakan [Pandas](https://pandas.pydata.org/), (nama ini berasal dari `Python Data Analysis`) alat yang sangat berguna untuk membentuk data, untuk menganalisis dan mempersiapkan data labu ini.

### Pertama, Periksa Tanggal yang Hilang

Anda pertama-tama perlu mengambil langkah-langkah untuk memeriksa tanggal yang hilang:

1. Konversikan tanggal ke format bulan (ini adalah tanggal AS, jadi formatnya adalah `MM/DD/YYYY`).
2. Ekstrak bulan ke kolom baru.

Buka file _notebook.ipynb_ di Visual Studio Code dan impor spreadsheet ke dalam dataframe Pandas baru.

1. Gunakan fungsi `head()` untuk melihat lima baris pertama.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Fungsi apa yang akan Anda gunakan untuk melihat lima baris terakhir?

1. Periksa apakah ada data yang hilang dalam dataframe saat ini:

    ```python
    pumpkins.isnull().sum()
    ```

    Ada data yang hilang, tetapi mungkin tidak akan menjadi masalah untuk tugas ini.

1. Untuk membuat dataframe Anda lebih mudah digunakan, pilih hanya kolom yang Anda butuhkan, menggunakan fungsi `loc` yang mengekstrak dari dataframe asli sekelompok baris (diberikan sebagai parameter pertama) dan kolom (diberikan sebagai parameter kedua). Ekspresi `:` dalam kasus ini berarti "semua baris".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Kedua, Tentukan Harga Rata-rata Labu

Pikirkan tentang cara menentukan harga rata-rata labu dalam bulan tertentu. Kolom apa yang akan Anda pilih untuk tugas ini? Petunjuk: Anda akan membutuhkan 3 kolom.

Solusi: ambil rata-rata dari kolom `Low Price` dan `High Price` untuk mengisi kolom Harga baru, dan konversikan kolom Tanggal untuk hanya menampilkan bulan. Untungnya, menurut pemeriksaan di atas, tidak ada data yang hilang untuk tanggal atau harga.

1. Untuk menghitung rata-rata, tambahkan kode berikut:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Silakan cetak data apa pun yang ingin Anda periksa menggunakan `print(month)`.

2. Sekarang, salin data yang telah dikonversi ke dataframe Pandas baru:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Mencetak dataframe Anda akan menunjukkan dataset yang bersih dan rapi yang dapat Anda gunakan untuk membangun model regresi baru Anda.

### Tapi Tunggu! Ada Sesuatu yang Aneh di Sini

Jika Anda melihat kolom `Package`, labu dijual dalam banyak konfigurasi yang berbeda. Beberapa dijual dalam ukuran '1 1/9 bushel', beberapa dalam ukuran '1/2 bushel', beberapa per labu, beberapa per pon, dan beberapa dalam kotak besar dengan lebar yang bervariasi.

> Labu tampaknya sangat sulit untuk ditimbang secara konsisten

Menggali data asli, menarik bahwa apa pun dengan `Unit of Sale` yang sama dengan 'EACH' atau 'PER BIN' juga memiliki tipe `Package` per inci, per bin, atau 'each'. Labu tampaknya sangat sulit untuk ditimbang secara konsisten, jadi mari kita memfilter mereka dengan memilih hanya labu dengan string 'bushel' di kolom `Package`.

1. Tambahkan filter di bagian atas file, di bawah impor .csv awal:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Jika Anda mencetak data sekarang, Anda dapat melihat bahwa Anda hanya mendapatkan sekitar 415 baris data yang berisi labu berdasarkan bushel.

### Tapi Tunggu! Ada Satu Hal Lagi yang Harus Dilakukan

Apakah Anda memperhatikan bahwa jumlah bushel bervariasi per baris? Anda perlu menormalkan harga sehingga Anda menunjukkan harga per bushel, jadi lakukan beberapa perhitungan untuk standarisasi.

1. Tambahkan baris ini setelah blok yang membuat dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Menurut [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), berat bushel tergantung pada jenis hasil panen, karena ini adalah pengukuran volume. "Satu bushel tomat, misalnya, seharusnya memiliki berat 56 pon... Daun dan sayuran mengambil lebih banyak ruang dengan berat lebih sedikit, sehingga satu bushel bayam hanya memiliki berat 20 pon." Semuanya cukup rumit! Mari kita tidak repot-repot membuat konversi bushel-ke-pon, dan sebagai gantinya harga berdasarkan bushel. Semua studi tentang bushel labu ini, bagaimanapun, menunjukkan betapa pentingnya memahami sifat data Anda!

Sekarang, Anda dapat menganalisis harga per unit berdasarkan pengukuran bushel mereka. Jika Anda mencetak data sekali lagi, Anda dapat melihat bagaimana data tersebut telah distandarisasi.

✅ Apakah Anda memperhatikan bahwa labu yang dijual berdasarkan setengah bushel sangat mahal? Bisakah Anda mencari tahu alasannya? Petunjuk: labu kecil jauh lebih mahal daripada yang besar, mungkin karena ada lebih banyak labu kecil per bushel, mengingat ruang kosong yang tidak terpakai yang diambil oleh satu labu besar untuk pai.

## Strategi Visualisasi

Bagian dari peran ilmuwan data adalah menunjukkan kualitas dan sifat data yang mereka kerjakan. Untuk melakukan ini, mereka sering membuat visualisasi yang menarik, seperti plot, grafik, dan diagram, yang menunjukkan berbagai aspek data. Dengan cara ini, mereka dapat secara visual menunjukkan hubungan dan celah yang sulit ditemukan.

[![ML untuk Pemula - Cara Memvisualisasikan Data dengan Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML untuk Pemula - Cara Memvisualisasikan Data dengan Matplotlib")

> 🎥 Klik gambar di atas untuk video singkat tentang memvisualisasikan data untuk pelajaran ini.

Visualisasi juga dapat membantu menentukan teknik pembelajaran mesin yang paling sesuai untuk data. Sebuah scatterplot yang tampaknya mengikuti garis, misalnya, menunjukkan bahwa data tersebut adalah kandidat yang baik untuk latihan regresi linier.

Salah satu pustaka visualisasi data yang bekerja dengan baik di Jupyter notebook adalah [Matplotlib](https://matplotlib.org/) (yang juga Anda lihat dalam pelajaran sebelumnya).

> Dapatkan lebih banyak pengalaman dengan visualisasi data dalam [tutorial ini](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Latihan - Bereksperimen dengan Matplotlib

Cobalah membuat beberapa plot dasar untuk menampilkan dataframe baru yang baru saja Anda buat. Apa yang akan ditampilkan oleh plot garis dasar?

1. Impor Matplotlib di bagian atas file, di bawah impor Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Jalankan ulang seluruh notebook untuk menyegarkan.
1. Di bagian bawah notebook, tambahkan sel untuk memplot data sebagai kotak:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Scatterplot yang menunjukkan hubungan harga dengan bulan](../../../../2-Regression/2-Data/images/scatterplot.png)

    Apakah ini plot yang berguna? Apakah ada sesuatu yang mengejutkan Anda?

    Ini tidak terlalu berguna karena hanya menampilkan data Anda sebagai sebaran titik dalam bulan tertentu.

### Buatlah Berguna

Untuk mendapatkan grafik yang menampilkan data yang berguna, Anda biasanya perlu mengelompokkan data dengan cara tertentu. Mari coba membuat plot di mana sumbu y menunjukkan bulan dan data menunjukkan distribusi data.

1. Tambahkan sel untuk membuat grafik batang yang dikelompokkan:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Grafik batang yang menunjukkan hubungan harga dengan bulan](../../../../2-Regression/2-Data/images/barchart.png)

    Ini adalah visualisasi data yang lebih berguna! Tampaknya menunjukkan bahwa harga tertinggi untuk labu terjadi pada bulan September dan Oktober. Apakah itu sesuai dengan ekspektasi Anda? Mengapa atau mengapa tidak?

---

## 🚀Tantangan

Jelajahi berbagai jenis visualisasi yang ditawarkan oleh Matplotlib. Jenis mana yang paling sesuai untuk masalah regresi?

## [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Lihatlah berbagai cara untuk memvisualisasikan data. Buat daftar berbagai pustaka yang tersedia dan catat mana yang terbaik untuk jenis tugas tertentu, misalnya visualisasi 2D vs. visualisasi 3D. Apa yang Anda temukan?

## Tugas

[Menjelajahi visualisasi](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.