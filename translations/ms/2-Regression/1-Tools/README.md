# Mula dengan Python dan Scikit-learn untuk model regresi

![Ringkasan regresi dalam sketchnote](../../../../translated_images/ms/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini juga tersedia dalam R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Pengenalan

Dalam empat pelajaran ini, anda akan mempelajari cara membina model regresi. Kami akan bincangkan apakah kegunaan model ini sebentar lagi. Tetapi sebelum anda mula, pastikan anda mempunyai alat yang betul untuk memulakan proses!

Dalam pelajaran ini, anda akan belajar bagaimana untuk:

- Menyediakan komputer anda untuk tugasan pembelajaran mesin setempat.
- Bekerja dengan Jupyter Notebooks.
- Menggunakan Scikit-learn, termasuk pemasangan.
- Meneroka regresi linear dengan latihan praktikal.

## Pemasangan dan konfigurasi

[![ML untuk pemula - Sediakan alat anda untuk membina model Pembelajaran Mesin](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML untuk pemula -Sediakan alat anda untuk membina model Pembelajaran Mesin")

> 🎥 Klik gambar di atas untuk video pendek yang menunjukkan cara menyediakan komputer anda untuk ML.

1. **Pasang Python**. Pastikan [Python](https://www.python.org/downloads/) telah dipasang pada komputer anda. Anda akan menggunakan Python untuk banyak tugasan sains data dan pembelajaran mesin. Kebanyakan sistem komputer sudah mempunyai pemasangan Python. Terdapat juga [Pakej Pengkodan Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) yang berguna, untuk memudahkan persediaan bagi beberapa pengguna.

   Walau bagaimanapun, beberapa penggunaan Python memerlukan versi perisian yang berbeza, oleh itu berguna untuk bekerja dalam [persekitaran maya](https://docs.python.org/3/library/venv.html).

2. **Pasang Visual Studio Code**. Pastikan anda memasang Visual Studio Code pada komputer anda. Ikuti arahan ini untuk [memasang Visual Studio Code](https://code.visualstudio.com/) untuk pemasangan asas. Anda akan menggunakan Python dalam Visual Studio Code dalam kursus ini, jadi anda mungkin ingin mengulang cara [menyediakan Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) untuk pembangunan Python.

   > Biasakan diri dengan Python dengan mengikuti koleksi [modul Pembelajaran](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sediakan Python dengan Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sediakan Python dengan Visual Studio Code")
   >
   > 🎥 Klik gambar di atas untuk video: menggunakan Python dalam VS Code.

3. **Pasang Scikit-learn**, dengan mengikuti [arahan ini](https://scikit-learn.org/stable/install.html). Oleh kerana anda perlu memastikan menggunakan Python 3, disarankan menggunakan persekitaran maya. Nota, jika anda memasang perpustakaan ini di Mac M1, terdapat arahan khas di halaman yang dilampirkan di atas.

1. **Pasang Jupyter Notebook**. Anda perlu [memasang pakej Jupyter](https://pypi.org/project/jupyter/).

## Persekitaran pengarang ML anda

Anda akan menggunakan **notebook** untuk membangunkan kod Python dan membuat model pembelajaran mesin. Jenis fail ini adalah alat biasa bagi saintis data, dan ia dikenal pasti dengan akhiran atau lanjutan `.ipynb`.

Notebook adalah persekitaran interaktif yang membolehkan pembangun menulis kod serta catatan dan dokumentasi di sekeliling kod yang sangat membantu untuk projek percubaan atau penyelidikan.

[![ML untuk pemula - Sediakan Jupyter Notebooks untuk mula membina model regresi](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML untuk pemula - Sediakan Jupyter Notebooks untuk mula membina model regresi")

> 🎥 Klik gambar di atas untuk video pendek yang menunjukkan latihan ini.

### Latihan - bekerja dengan notebook

Dalam folder ini, anda akan menemui fail _notebook.ipynb_.

1. Buka _notebook.ipynb_ dalam Visual Studio Code.

   Pelayan Jupyter akan dimulakan dengan Python 3+. Anda akan menemui bahagian dalam notebook yang boleh `run`, bahagian kod. Anda boleh menjalankan blok kod dengan memilih ikon seperti butang main.

1. Pilih ikon `md` dan tambahkan sedikit markdown, dan teks berikut **# Selamat datang ke notebook anda**.

   Seterusnya, tambahkan beberapa kod Python.

1. Taip **print('hello notebook')** dalam blok kod.
1. Pilih anak panah untuk menjalankan kod.

   Anda akan melihat pernyataan yang dicetak:

    ```output
    hello notebook
    ```

![VS Code dengan notebook dibuka](../../../../translated_images/ms/notebook.4a3ee31f396b8832.webp)

Anda boleh menyelangi kod anda dengan komen untuk mendokumentasikan notebook sendiri.

✅ Fikirkan sejenak bagaimana perbezaan persekitaran kerja pembangun web berbanding saintis data.

## Mulakan dengan Scikit-learn

Sekarang Python telah disediakan dalam persekitaran setempat anda, dan anda sudah biasa dengan Jupyter Notebooks, mari kita jadi sama selesa dengan Scikit-learn (sebut `sci` seperti dalam `science`). Scikit-learn menyediakan [API yang meluas](https://scikit-learn.org/stable/modules/classes.html#api-ref) untuk membantu anda menjalankan tugasan ML.

Menurut [laman web mereka](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn adalah perpustakaan pembelajaran mesin sumber terbuka yang menyokong pembelajaran berarah dan tidak berarah. Ia juga menyediakan pelbagai alat untuk pemadanan model, pemprosesan awal data, pemilihan model dan penilaian, serta banyak kegunaan lain."

Dalam kursus ini, anda akan menggunakan Scikit-learn dan alat lain untuk membina model pembelajaran mesin untuk melaksanakan apa yang kita panggil tugasan 'pembelajaran mesin tradisional'. Kami sengaja mengelakkan rangkaian neural dan pembelajaran mendalam, kerana ia lebih baik dibahas dalam kurikulum 'AI untuk Pemula' yang akan datang.

Scikit-learn memudahkan pembinaan model dan penilaiannya untuk kegunaan. Ia tertumpu terutamanya pada penggunaan data berangka dan mengandungi beberapa set data siap guna untuk digunakan sebagai alat pembelajaran. Ia juga termasuk model siap pakai untuk pelajar cuba. Mari kita terokai proses memuat data terbina dan menggunakan penaksir terbina untuk mencipta model ML pertama anda dengan Scikit-learn menggunakan data asas.

## Latihan - notebook Scikit-learn pertama anda

> Tutorial ini diinspirasikan oleh [contoh regresi linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) di laman web Scikit-learn.


[![ML untuk pemula - Projek Regresi Linear Pertama Anda dalam Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML untuk pemula - Projek Regresi Linear Pertama Anda dalam Python")

> 🎥 Klik gambar di atas untuk video pendek yang menunjukkan latihan ini.

Dalam fail _notebook.ipynb_ yang berkaitan dengan pelajaran ini, kosongkan semua sel dengan menekan ikon 'tong sampah'.

Dalam bahagian ini, anda akan bekerja dengan set data kecil tentang diabetes yang dibina ke dalam Scikit-learn untuk tujuan pembelajaran. Bayangkan anda ingin menguji rawatan untuk pesakit diabetes. Model Pembelajaran Mesin mungkin membantu menentukan pesakit yang memberi tindak balas lebih baik terhadap rawatan, berdasarkan gabungan pembolehubah. Model regresi yang sangat asas, apabila divisualisasikan, mungkin menunjukkan maklumat tentang pembolehubah yang membantu anda mengatur kajian klinikal teori anda.

✅ Terdapat banyak jenis kaedah regresi, dan yang anda pilih bergantung pada jawapan yang anda cari. Jika anda mahu meramalkan tinggi badan untuk seseorang pada umur tertentu, anda akan menggunakan regresi linear, kerana anda mencari **nilai berangka**. Jika anda berminat mengetahui sama ada jenis masakan adalah vegan atau tidak, anda mencari **penugasan kategori** jadi anda akan menggunakan regresi logistik. Anda akan belajar lebih lanjut tentang regresi logistik kemudian. Fikirkan sedikit tentang soalan yang boleh anda ajukan pada data, dan kaedah mana yang lebih sesuai.

Mari kita mulakan tugasan ini.

### Import perpustakaan

Untuk tugasan ini kita akan import beberapa perpustakaan:

- **matplotlib**. Ia adalah [alat graf yang berguna](https://matplotlib.org/) dan kita akan menggunakannya untuk membuat plot garis.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) adalah perpustakaan berguna untuk mengurus data berangka dalam Python.
- **sklearn**. Ini adalah perpustakaan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Import beberapa perpustakaan untuk membantu tugasan anda.

1. Tambah import dengan menaip kod berikut:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Di atas anda mengimport `matplotlib`, `numpy` dan mengimport `datasets`, `linear_model` dan `model_selection` dari `sklearn`. `model_selection` digunakan untuk membahagikan data kepada set latihan dan ujian.

### Set data diabetes

Set data [diabetes terbina](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) termasuk 442 sampel data tentang diabetes, dengan 10 pembolehubah ciri, beberapa antaranya termasuk:

- umur: umur dalam tahun
- bmi: indeks jisim badan
- bp: purata tekanan darah
- s1 tc: Sel T (jenis sel darah putih)

✅ Set data ini termasuk konsep 'jantina' sebagai pembolehubah ciri penting dalam penyelidikan diabetes. Banyak set data perubatan termasuk klasifikasi binari sebegini. Fikirkan sedikit bagaimana pengkategorian sebegini boleh mengecualikan sebahagian penduduk dari rawatan.

Sekarang, muatkan data X dan y.

> 🎓 Ingat, ini adalah pembelajaran berarah, dan kita perlukan sasaran bernama 'y'.

Dalam sel kod baru, muat set data diabetes dengan memanggil `load_diabetes()`. Input `return_X_y=True` menandakan bahawa `X` adalah matriks data, dan `y` adalah sasaran regresi.

1. Tambah beberapa arahan cetak untuk menunjukkan bentuk matriks data dan elemen pertamanya:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Apa yang anda terima sebagai maklum balas adalah satu tuple. Apa yang anda lakukan adalah menugaskan dua nilai pertama tuple tersebut ke `X` dan `y` masing-masing. Ketahui lebih lanjut [tentang tuple](https://wikipedia.org/wiki/Tuple).

    Anda boleh lihat data ini mempunyai 442 item yang dibentuk dalam array 10 elemen:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Fikirkan sedikit tentang hubungan antara data dan sasaran regresi. Regresi linear meramalkan hubungan antara ciri X dan pembolehubah sasaran y. Bolehkah anda temui [sasaran](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) untuk set data diabetes dalam dokumentasi? Apakah yang dataset ini demonstrasikan, berdasarkan sasaran tersebut?

2. Seterusnya, pilih sebahagian set data ini untuk dilakar dengan memilih lajur ke-3 set data. Anda boleh lakukan ini dengan menggunakan operator `:` untuk memilih semua baris, dan kemudian pilih lajur ke-3 menggunakan indeks (2). Anda juga boleh membentuk data menjadi array 2D - seperti yang diperlukan untuk pembuatan plot - dengan menggunakan `reshape(n_baris, n_lajur)`. Jika salah satu parameter adalah -1, dimensi yang sepadan dikira secara automatik.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Pada bila-bila masa, cetak data untuk periksa bentuknya.

3. Sekarang data sudah sedia untuk dilakar, anda boleh lihat jika mesin dapat membantu menentukan pecahan logik antara nombor dalam dataset ini. Untuk melakukan ini, anda perlu membahagi kedua-dua data (X) dan sasaran (y) kepada set ujian dan latihan. Scikit-learn mempunyai cara mudah untuk melakukan ini; anda boleh membahagi data ujian pada titik tertentu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Kini anda sudah sedia melatih model! Muat model regresi linear dan latih dengan set latihan X dan y menggunakan `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` adalah fungsi yang biasa anda lihat dalam banyak perpustakaan ML seperti TensorFlow

5. Kemudian, buat ramalan menggunakan data ujian, dengan fungsi `predict()`. Ini akan digunakan untuk melukis garis antara kumpulan data

    ```python
    y_pred = model.predict(X_test)
    ```

6. Kini tiba masa untuk tunjuk data dalam plot. Matplotlib adalah alat yang sangat berguna untuk tugasan ini. Buat scatterplot semua data ujian X dan y, dan gunakan ramalan tersebut untuk melukis garis di tempat yang paling sesuai, antara kumpulan data model.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot menunjukkan titik data berkaitan diabetes](../../../../translated_images/ms/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Fikirkan sedikit apa yang sedang berlaku di sini. Garis lurus berjalan melalui banyak titik kecil data, tetapi apa sebenarnya yang ia lakukan? Bolehkah anda lihat bagaimana anda sepatutnya menggunakan garis ini untuk meramalkan di mana titik data baru yang belum pernah dilihat harus sesuai berkaitan paksi y plot? Cuba nyatakan dalam kata-kata kegunaan praktikal model ini.

Tahniah, anda sudah membina model regresi linear pertama anda, buat ramalan dengannya, dan paparkannya dalam plot!

---
## 🚀Cabaran

Lakarkan pembolehubah berbeza dari dataset ini. Petunjuk: sunting baris ini: `X = X[:,2]`. Berdasarkan sasaran dataset ini, apa yang anda dapat temui tentang perkembangan diabetes sebagai penyakit?
## [Kuiz pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Belajar Sendiri

Dalam tutorial ini, anda bekerja dengan regresi linear mudah, bukan regresi linear univariat atau multivariat. Baca sedikit tentang perbezaan antara kaedah ini, atau lihat [video ini](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Baca lebih lanjut mengenai konsep regresi dan fikirkan jenis soalan apa yang boleh dijawab dengan teknik ini. Ikut [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) ini untuk memperdalam pemahaman anda.

## Tugasan

[Set data yang berbeza](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan oleh manusia profesional adalah disyorkan. Kami tidak bertanggungjawab terhadap sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->