# Memulai dengan Python dan Scikit-learn untuk model regresi

![Ringkasan regresi dalam sketchnote](../../../../translated_images/id/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuis pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Pendahuluan

Dalam empat pelajaran ini, Anda akan menemukan cara membangun model regresi. Kita akan membahas untuk apa model ini sesaat lagi. Tapi sebelum Anda melakukan apa pun, pastikan Anda memiliki alat yang tepat untuk memulai proses!

Dalam pelajaran ini, Anda akan belajar cara untuk:

- Mengonfigurasi komputer Anda untuk tugas pembelajaran mesin lokal.
- Bekerja dengan Jupyter Notebooks.
- Menggunakan Scikit-learn, termasuk instalasi.
- Menjelajahi regresi linier dengan latihan langsung.

## Instalasi dan konfigurasi

[![ML untuk pemula - Siapkan alat Anda untuk membangun model Pembelajaran Mesin](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML untuk pemula - Siapkan alat Anda untuk membangun model Pembelajaran Mesin")

> 🎥 Klik gambar di atas untuk video singkat tentang konfigurasi komputer Anda untuk ML.

1. **Pasang Python**. Pastikan [Python](https://www.python.org/downloads/) sudah terpasang di komputer Anda. Anda akan menggunakan Python untuk banyak tugas ilmu data dan pembelajaran mesin. Sebagian besar sistem komputer sudah memiliki instalasi Python. Ada juga [Paket Kode Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) yang berguna untuk memudahkan pengaturan bagi beberapa pengguna.

   Namun, beberapa penggunaan Python memerlukan versi perangkat lunak yang berbeda. Oleh karena itu, berguna untuk bekerja dalam [environment virtual](https://docs.python.org/3/library/venv.html).

2. **Pasang Visual Studio Code**. Pastikan Anda sudah memasang Visual Studio Code di komputer Anda. Ikuti instruksi ini untuk [memasang Visual Studio Code](https://code.visualstudio.com/) untuk instalasi dasar. Anda akan menggunakan Python di Visual Studio Code dalam kursus ini, jadi Anda mungkin ingin mempelajari cara [mengonfigurasi Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) untuk pengembangan Python.

   > Kenali Python dengan menjalani rangkaian modul [Learn](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Siapkan Python dengan Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Siapkan Python dengan Visual Studio Code")
   >
   > 🎥 Klik gambar di atas untuk video: menggunakan Python dalam VS Code.

3. **Pasang Scikit-learn**, dengan mengikuti [instruksi ini](https://scikit-learn.org/stable/install.html). Karena Anda harus memastikan menggunakan Python 3, disarankan menggunakan environment virtual. Perhatikan, jika Anda memasang pustaka ini di M1 Mac, ada instruksi khusus di halaman yang terhubung di atas.

1. **Pasang Jupyter Notebook**. Anda perlu [memasang paket Jupyter](https://pypi.org/project/jupyter/).

## Lingkungan penulisan ML Anda

Anda akan menggunakan **notebook** untuk mengembangkan kode Python Anda dan membuat model pembelajaran mesin. Jenis file ini adalah alat umum bagi ilmuwan data, dan dapat dikenali dari akhiran atau ekstensi `.ipynb`.

Notebook adalah lingkungan interaktif yang memungkinkan pengembang untuk menulis kode serta menambahkan catatan dan dokumentasi di sekitar kode yang sangat membantu untuk proyek eksperimental atau penelitian.

[![ML untuk pemula - Siapkan Jupyter Notebooks untuk mulai membangun model regresi](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML untuk pemula - Siapkan Jupyter Notebooks untuk mulai membangun model regresi")

> 🎥 Klik gambar di atas untuk video singkat yang mengerjakan latihan ini.

### Latihan - bekerja dengan notebook

Dalam folder ini, Anda akan menemukan file _notebook.ipynb_.

1. Buka _notebook.ipynb_ di Visual Studio Code.

   Server Jupyter akan mulai dengan Python 3+ aktif. Anda akan menemukan area notebook yang dapat `dijalankan`, potongan kode. Anda dapat menjalankan blok kode dengan memilih ikon yang mirip tombol putar.

1. Pilih ikon `md` dan tambahkan sedikit markdown, dan teks berikut **# Selamat datang di notebook Anda**.

   Berikutnya, tambahkan beberapa kode Python.

1. Ketik **print('hello notebook')** di blok kode.
1. Pilih panah untuk menjalankan kode.

   Anda akan melihat pernyataan tercetak:

    ```output
    hello notebook
    ```

![VS Code dengan notebook terbuka](../../../../translated_images/id/notebook.4a3ee31f396b8832.webp)

Anda dapat menyela kode Anda dengan komentar untuk mendokumentasi notebook secara mandiri.

✅ Pikirkan sebentar betapa berbeda lingkungan kerja pengembang web dibandingkan ilmuwan data.

## Siap pakai dengan Scikit-learn

Sekarang Python sudah terpasang di lingkungan lokal Anda, dan Anda sudah nyaman dengan Jupyter Notebooks, mari sama-sama kenal dengan Scikit-learn (baca `sci` seperti dalam `science`). Scikit-learn menyediakan [API luas](https://scikit-learn.org/stable/modules/classes.html#api-ref) untuk membantu Anda melakukan tugas ML.

Menurut [situs mereka](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn adalah perpustakaan pembelajaran mesin sumber terbuka yang mendukung pembelajaran terawasi dan tidak terawasi. Ia juga menyediakan berbagai alat untuk penyetelan model, pra-pemrosesan data, seleksi model dan evaluasi, serta banyak utilitas lainnya."

Dalam kursus ini, Anda akan menggunakan Scikit-learn dan alat lain untuk membangun model pembelajaran mesin guna melakukan apa yang kami sebut tugas 'pembelajaran mesin tradisional'. Kami sengaja menghindari jaringan saraf dan pembelajaran mendalam, karena itu lebih baik dibahas dalam kurikulum 'AI untuk Pemula' yang akan datang.

Scikit-learn memudahkan membangun model dan mengevaluasinya untuk digunakan. Fokus utamanya pada data numerik dan berisi beberapa dataset siap pakai untuk proses pembelajaran. Ia juga memiliki model bawaan untuk dicoba oleh siswa. Mari kita eksplorasi proses memuat data siap pakai dan menggunakan estimator bawaan untuk membuat model ML pertama Anda dengan Scikit-learn dengan data dasar.

## Latihan - notebook Scikit-learn pertama Anda

> Tutorial ini terinspirasi oleh [contoh regresi linier](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) di situs web Scikit-learn.


[![ML untuk pemula - Proyek Regresi Linier Pertama Anda dalam Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML untuk pemula - Proyek Regresi Linier Pertama Anda dalam Python")

> 🎥 Klik gambar di atas untuk video singkat yang mengerjakan latihan ini.

Dalam file _notebook.ipynb_ yang terkait dengan pelajaran ini, kosongkan semua sel dengan menekan ikon 'tempat sampah'.

Pada bagian ini, Anda akan bekerja dengan dataset kecil tentang diabetes yang sudah ada dalam Scikit-learn untuk tujuan pembelajaran. Bayangkan Anda ingin menguji pengobatan untuk pasien diabetes. Model Pembelajaran Mesin dapat membantu menentukan pasien mana yang akan merespons lebih baik berdasarkan kombinasi variabel. Bahkan model regresi sederhana, saat divisualisasikan, bisa menunjukkan informasi tentang variabel yang membantu Anda mengatur uji klinis teoritis.

✅ Ada banyak metode regresi, dan pilihan metode bergantung pada pertanyaan yang Anda cari jawabannya. Jika Anda ingin memprediksi tinggi badan seorang berdasarkan umur, Anda akan menggunakan regresi linier karena Anda mencari **nilai numerik**. Jika Anda ingin mengetahui apakah jenis masakan tertentu adalah vegan atau bukan, Anda mencari **penetapan kategori**, sehingga akan menggunakan regresi logistik. Anda akan mempelajari lebih banyak tentang regresi logistik nanti. Pikirkan sebentar tentang beberapa pertanyaan yang bisa Anda ajukan pada data, dan metode mana yang lebih tepat.

Mari kita mulai tugas ini.

### Impor pustaka

Untuk tugas ini kita akan mengimpor beberapa pustaka:

- **matplotlib**. Ini adalah [alat penggambaran](https://matplotlib.org/) yang berguna dan akan kita gunakan untuk membuat plot garis.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) adalah pustaka berguna untuk menangani data numerik dalam Python.
- **sklearn**. Ini adalah pustaka [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Impor beberapa pustaka untuk membantu tugas Anda.

1. Tambahkan impor dengan mengetik kode berikut:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Di atas Anda sedang mengimpor `matplotlib`, `numpy` dan mengimpor `datasets`, `linear_model` dan `model_selection` dari `sklearn`. `model_selection` digunakan untuk membagi data menjadi set pelatihan dan uji.

### Dataset diabetes

Dataset bawaan [diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) ini memuat 442 sampel data tentang diabetes, dengan 10 variabel fitur, beberapa di antaranya meliputi:

- age: umur dalam tahun
- bmi: indeks massa tubuh
- bp: tekanan darah rata-rata
- s1 tc: Sel T (jenis sel darah putih)

✅ Dataset ini memasukkan konsep 'jenis kelamin' sebagai variabel fitur yang penting untuk penelitian tentang diabetes. Banyak dataset medis menggunakan klasifikasi biner seperti ini. Pikirkan sebentar bagaimana kategorisasi semacam ini mungkin mengecualikan sebagian populasi dari pengobatan.

Sekarang, muat data X dan y.

> 🎓 Ingat, ini adalah pembelajaran terawasi, dan kita perlu target bernama 'y'.

Di sel kode baru, muat dataset diabetes dengan memanggil `load_diabetes()`. Input `return_X_y=True` memberi sinyal bahwa `X` adalah matriks data, dan `y` adalah target regresi.

1. Tambahkan beberapa perintah print untuk menampilkan bentuk matriks data dan elemen pertama:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Apa yang Anda dapat sebagai jawaban adalah tuple. Anda sedang menetapkan dua nilai pertama tuple tersebut ke `X` dan `y` secara berurutan. Pelajari lebih lanjut [tentang tuple](https://wikipedia.org/wiki/Tuple).

    Anda dapat melihat bahwa data ini memiliki 442 item yang dibentuk dalam array berisi 10 elemen:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pikirkan hubungan antara data dan target regresi. Regresi linier memprediksi hubungan antara fitur X dan variabel target y. Dapatkah Anda menemukan [target](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) untuk dataset diabetes dalam dokumentasi? Apa yang ditunjukkan dataset ini, berdasarkan target tersebut?

2. Selanjutnya, pilih bagian dari dataset ini untuk diplot dengan memilih kolom ke-3 data. Anda dapat menggunakan operator `:` untuk memilih semua baris, kemudian memilih kolom ke-3 menggunakan indeks (2). Anda juga dapat mengubah bentuk data menjadi array 2D - sesuai kebutuhan plot - dengan menggunakan `reshape(n_baris, n_kolom)`. Jika salah satu parameter adalah -1, dimensi yang bersesuaian dihitung secara otomatis.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Kapan pun, cetak data untuk memeriksa bentuknya.

3. Setelah Anda memiliki data siap dipetakan, Anda dapat melihat apakah mesin dapat membantu menentukan pemisahan logis antara angka dalam dataset ini. Untuk melakukannya, Anda perlu membagi data (X) dan target (y) menjadi set uji dan pelatihan. Scikit-learn memiliki cara sederhana untuk ini; Anda dapat membagi data uji pada titik tertentu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sekarang Anda siap melatih model! Muat model regresi linier dan latih dengan set pelatihan X dan y Anda menggunakan `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` adalah fungsi yang sering ditemui di banyak pustaka ML seperti TensorFlow

5. Kemudian, buat prediksi menggunakan data uji, dengan fungsi `predict()`. Ini akan digunakan untuk menggambar garis di antara kelompok data

    ```python
    y_pred = model.predict(X_test)
    ```

6. Sekarang saatnya menampilkan data dalam plot. Matplotlib adalah alat yang sangat berguna untuk tugas ini. Buat scatterplot dari semua data uji X dan y, dan gunakan prediksi untuk menggambar garis di tempat paling tepat, antara pengelompokan data model.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot yang menunjukkan titik data diabetes](../../../../translated_images/id/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Pikirkan sebentar tentang apa yang sebenarnya terjadi di sini. Sebuah garis lurus melewati banyak titik data kecil, tapi apa sebenarnya yang dilakukannya? Apakah Anda bisa melihat bagaimana garis ini dapat digunakan untuk memprediksi di mana titik data baru yang belum terlihat harus ditempatkan terkait sumbu y plot? Coba jelaskan manfaat praktis model ini.

Selamat, Anda telah membangun model regresi linier pertama Anda, membuat prediksi dengannya, dan menampilkannya dalam plot!

---
## 🚀Tantangan

Plot variabel berbeda dari dataset ini. Petunjuk: edit baris ini: `X = X[:,2]`. Berdasarkan target dataset ini, apa yang dapat Anda temukan tentang perkembangan diabetes sebagai penyakit?
## [Kuis pasca-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Studi Mandiri

Dalam tutorial ini, Anda bekerja dengan regresi linier sederhana, bukan regresi linier univariat atau multivariat. Bacalah sedikit tentang perbedaan metode ini, atau lihat [video ini](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Baca lebih lanjut tentang konsep regresi dan pikirkan jenis pertanyaan apa yang dapat dijawab dengan teknik ini. Ikuti [tutorial ini](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) untuk memperdalam pemahaman Anda.

## Tugas

[Dataset yang berbeda](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->