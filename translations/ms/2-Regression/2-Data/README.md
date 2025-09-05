<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T18:56:22+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "ms"
}
-->
# Bina model regresi menggunakan Scikit-learn: sediakan dan visualisasikan data

![Infografik visualisasi data](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografik oleh [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Pengenalan

Sekarang anda telah bersedia dengan alat yang diperlukan untuk mula membina model pembelajaran mesin menggunakan Scikit-learn, anda sudah bersedia untuk mula bertanya soalan kepada data anda. Semasa anda bekerja dengan data dan menerapkan penyelesaian ML, adalah sangat penting untuk memahami cara bertanya soalan yang betul untuk membuka potensi dataset anda dengan tepat.

Dalam pelajaran ini, anda akan belajar:

- Cara menyediakan data anda untuk pembinaan model.
- Cara menggunakan Matplotlib untuk visualisasi data.

## Bertanya soalan yang betul kepada data anda

Soalan yang anda perlukan jawapannya akan menentukan jenis algoritma ML yang akan anda gunakan. Dan kualiti jawapan yang anda peroleh sangat bergantung pada sifat data anda.

Lihat [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) yang disediakan untuk pelajaran ini. Anda boleh membuka fail .csv ini dalam VS Code. Sekilas pandang menunjukkan terdapat kekosongan dan campuran data string dan numerik. Terdapat juga kolum yang pelik dipanggil 'Package' di mana datanya adalah campuran antara 'sacks', 'bins' dan nilai lain. Data ini, sebenarnya, agak bersepah.

[![ML untuk pemula - Cara Menganalisis dan Membersihkan Dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML untuk pemula - Cara Menganalisis dan Membersihkan Dataset")

> 🎥 Klik imej di atas untuk video pendek yang menunjukkan cara menyediakan data untuk pelajaran ini.

Sebenarnya, adalah tidak biasa untuk menerima dataset yang sepenuhnya siap digunakan untuk mencipta model ML secara langsung. Dalam pelajaran ini, anda akan belajar cara menyediakan dataset mentah menggunakan pustaka Python standard. Anda juga akan belajar pelbagai teknik untuk memvisualisasikan data.

## Kajian kes: 'pasaran labu'

Dalam folder ini, anda akan menemui fail .csv dalam folder root `data` yang dipanggil [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) yang mengandungi 1757 baris data tentang pasaran labu, disusun mengikut kumpulan berdasarkan bandar. Ini adalah data mentah yang diekstrak daripada [Laporan Standard Pasaran Terminal Tanaman Khas](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) yang diedarkan oleh Jabatan Pertanian Amerika Syarikat.

### Menyediakan data

Data ini adalah dalam domain awam. Ia boleh dimuat turun dalam banyak fail berasingan, mengikut bandar, dari laman web USDA. Untuk mengelakkan terlalu banyak fail berasingan, kami telah menggabungkan semua data bandar ke dalam satu spreadsheet, jadi kami telah _menyediakan_ data sedikit. Seterusnya, mari kita lihat lebih dekat data ini.

### Data labu - kesimpulan awal

Apa yang anda perhatikan tentang data ini? Anda sudah melihat bahawa terdapat campuran string, nombor, kekosongan dan nilai pelik yang perlu anda fahami.

Soalan apa yang boleh anda tanyakan kepada data ini, menggunakan teknik Regresi? Bagaimana dengan "Meramalkan harga labu yang dijual pada bulan tertentu". Melihat semula data, terdapat beberapa perubahan yang perlu anda lakukan untuk mencipta struktur data yang diperlukan untuk tugas ini.

## Latihan - analisis data labu

Mari gunakan [Pandas](https://pandas.pydata.org/), (nama ini bermaksud `Python Data Analysis`) alat yang sangat berguna untuk membentuk data, untuk menganalisis dan menyediakan data labu ini.

### Pertama, periksa tarikh yang hilang

Anda perlu mengambil langkah untuk memeriksa tarikh yang hilang:

1. Tukarkan tarikh kepada format bulan (ini adalah tarikh AS, jadi formatnya adalah `MM/DD/YYYY`).
2. Ekstrak bulan ke kolum baru.

Buka fail _notebook.ipynb_ dalam Visual Studio Code dan import spreadsheet ke dalam dataframe Pandas yang baru.

1. Gunakan fungsi `head()` untuk melihat lima baris pertama.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Fungsi apa yang akan anda gunakan untuk melihat lima baris terakhir?

1. Periksa jika terdapat data yang hilang dalam dataframe semasa:

    ```python
    pumpkins.isnull().sum()
    ```

    Terdapat data yang hilang, tetapi mungkin ia tidak akan menjadi masalah untuk tugas ini.

1. Untuk memudahkan dataframe anda bekerja, pilih hanya kolum yang anda perlukan, menggunakan fungsi `loc` yang mengekstrak dari dataframe asal sekumpulan baris (diberikan sebagai parameter pertama) dan kolum (diberikan sebagai parameter kedua). Ungkapan `:` dalam kes di bawah bermaksud "semua baris".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Kedua, tentukan harga purata labu

Fikirkan cara menentukan harga purata labu dalam bulan tertentu. Kolum apa yang akan anda pilih untuk tugas ini? Petunjuk: anda memerlukan 3 kolum.

Penyelesaian: ambil purata kolum `Low Price` dan `High Price` untuk mengisi kolum Harga baru, dan tukarkan kolum Tarikh untuk hanya menunjukkan bulan. Nasib baik, menurut pemeriksaan di atas, tiada data yang hilang untuk tarikh atau harga.

1. Untuk mengira purata, tambahkan kod berikut:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Anda bebas untuk mencetak sebarang data yang anda ingin periksa menggunakan `print(month)`.

2. Sekarang, salin data yang telah ditukar ke dalam dataframe Pandas yang baru:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Mencetak dataframe anda akan menunjukkan dataset yang bersih dan kemas di mana anda boleh membina model regresi baru anda.

### Tetapi tunggu! Ada sesuatu yang pelik di sini

Jika anda melihat kolum `Package`, labu dijual dalam pelbagai konfigurasi. Ada yang dijual dalam ukuran '1 1/9 bushel', ada yang dalam ukuran '1/2 bushel', ada yang per labu, ada yang per pound, dan ada yang dalam kotak besar dengan lebar yang berbeza.

> Labu nampaknya sangat sukar untuk ditimbang secara konsisten

Menggali data asal, menarik bahawa apa-apa dengan `Unit of Sale` yang sama dengan 'EACH' atau 'PER BIN' juga mempunyai jenis `Package` per inci, per bin, atau 'each'. Labu nampaknya sangat sukar untuk ditimbang secara konsisten, jadi mari kita tapis mereka dengan memilih hanya labu dengan string 'bushel' dalam kolum `Package`.

1. Tambahkan penapis di bahagian atas fail, di bawah import .csv awal:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Jika anda mencetak data sekarang, anda boleh melihat bahawa anda hanya mendapatkan sekitar 415 baris data yang mengandungi labu mengikut bushel.

### Tetapi tunggu! Ada satu lagi perkara yang perlu dilakukan

Adakah anda perasan bahawa jumlah bushel berbeza setiap baris? Anda perlu menormalkan harga supaya anda menunjukkan harga per bushel, jadi lakukan beberapa pengiraan untuk menyeragamkannya.

1. Tambahkan baris ini selepas blok yang mencipta dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Menurut [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), berat bushel bergantung pada jenis hasil, kerana ia adalah ukuran volum. "Bushel tomato, sebagai contoh, sepatutnya beratnya 56 paun... Daun dan sayur-sayuran mengambil lebih banyak ruang dengan berat yang lebih sedikit, jadi bushel bayam hanya 20 paun." Semuanya agak rumit! Mari kita tidak bersusah payah membuat penukaran bushel-ke-paun, dan sebaliknya harga mengikut bushel. Semua kajian tentang bushel labu ini, bagaimanapun, menunjukkan betapa pentingnya memahami sifat data anda!

Sekarang, anda boleh menganalisis harga per unit berdasarkan ukuran bushel mereka. Jika anda mencetak data sekali lagi, anda boleh melihat bagaimana ia telah diseragamkan.

✅ Adakah anda perasan bahawa labu yang dijual mengikut setengah bushel sangat mahal? Bolehkah anda mengetahui sebabnya? Petunjuk: labu kecil jauh lebih mahal daripada yang besar, mungkin kerana terdapat lebih banyak daripadanya per bushel, memandangkan ruang yang tidak digunakan yang diambil oleh satu labu pai besar yang berongga.

## Strategi Visualisasi

Sebahagian daripada peranan saintis data adalah untuk menunjukkan kualiti dan sifat data yang mereka kerjakan. Untuk melakukan ini, mereka sering mencipta visualisasi yang menarik, seperti plot, graf, dan carta, yang menunjukkan aspek data yang berbeza. Dengan cara ini, mereka dapat menunjukkan secara visual hubungan dan jurang yang sukar untuk ditemui.

[![ML untuk pemula - Cara Memvisualisasikan Data dengan Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML untuk pemula - Cara Memvisualisasikan Data dengan Matplotlib")

> 🎥 Klik imej di atas untuk video pendek yang menunjukkan cara memvisualisasikan data untuk pelajaran ini.

Visualisasi juga boleh membantu menentukan teknik pembelajaran mesin yang paling sesuai untuk data. Plot taburan yang kelihatan mengikuti garis, sebagai contoh, menunjukkan bahawa data adalah calon yang baik untuk latihan regresi linear.

Satu pustaka visualisasi data yang berfungsi dengan baik dalam Jupyter notebooks ialah [Matplotlib](https://matplotlib.org/) (yang juga anda lihat dalam pelajaran sebelumnya).

> Dapatkan lebih banyak pengalaman dengan visualisasi data dalam [tutorial ini](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Latihan - eksperimen dengan Matplotlib

Cuba buat beberapa plot asas untuk memaparkan dataframe baru yang baru anda cipta. Apa yang akan ditunjukkan oleh plot garis asas?

1. Import Matplotlib di bahagian atas fail, di bawah import Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Jalankan semula keseluruhan notebook untuk menyegarkan.
1. Di bahagian bawah notebook, tambahkan sel untuk plot data sebagai kotak:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Plot taburan menunjukkan hubungan harga dengan bulan](../../../../2-Regression/2-Data/images/scatterplot.png)

    Adakah ini plot yang berguna? Adakah sesuatu mengenainya mengejutkan anda?

    Ia tidak begitu berguna kerana ia hanya memaparkan data anda sebagai taburan titik dalam bulan tertentu.

### Jadikannya berguna

Untuk mendapatkan carta yang memaparkan data berguna, anda biasanya perlu mengelompokkan data dengan cara tertentu. Mari cuba buat plot di mana paksi y menunjukkan bulan dan data menunjukkan taburan data.

1. Tambahkan sel untuk mencipta carta bar yang dikelompokkan:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Carta bar menunjukkan hubungan harga dengan bulan](../../../../2-Regression/2-Data/images/barchart.png)

    Ini adalah visualisasi data yang lebih berguna! Ia nampaknya menunjukkan bahawa harga tertinggi untuk labu berlaku pada bulan September dan Oktober. Adakah itu memenuhi jangkaan anda? Mengapa atau mengapa tidak?

---

## 🚀Cabaran

Terokai pelbagai jenis visualisasi yang ditawarkan oleh Matplotlib. Jenis mana yang paling sesuai untuk masalah regresi?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Lihat pelbagai cara untuk memvisualisasikan data. Buat senarai pustaka yang tersedia dan catatkan mana yang terbaik untuk jenis tugas tertentu, contohnya visualisasi 2D vs. visualisasi 3D. Apa yang anda temui?

## Tugasan

[Menjelajahi visualisasi](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.