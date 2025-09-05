<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "fa81d226c71d5af7a2cade31c1c92b88",
  "translation_date": "2025-09-05T18:52:44+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "ms"
}
-->
# Bermula dengan Python dan Scikit-learn untuk model regresi

![Ringkasan regresi dalam sketchnote](../../../../sketchnotes/ml-regression.png)

> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

> ### [Pelajaran ini tersedia dalam R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Pengenalan

Dalam empat pelajaran ini, anda akan belajar cara membina model regresi. Kita akan membincangkan kegunaannya sebentar lagi. Tetapi sebelum memulakan apa-apa, pastikan anda mempunyai alat yang betul untuk memulakan proses ini!

Dalam pelajaran ini, anda akan belajar cara:

- Mengkonfigurasi komputer anda untuk tugas pembelajaran mesin secara tempatan.
- Bekerja dengan Jupyter notebooks.
- Menggunakan Scikit-learn, termasuk pemasangan.
- Meneroka regresi linear melalui latihan praktikal.

## Pemasangan dan konfigurasi

[![ML untuk pemula - Sediakan alat anda untuk membina model Pembelajaran Mesin](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML untuk pemula - Sediakan alat anda untuk membina model Pembelajaran Mesin")

> 🎥 Klik imej di atas untuk video pendek tentang cara mengkonfigurasi komputer anda untuk ML.

1. **Pasang Python**. Pastikan [Python](https://www.python.org/downloads/) dipasang pada komputer anda. Anda akan menggunakan Python untuk banyak tugas sains data dan pembelajaran mesin. Kebanyakan sistem komputer sudah mempunyai Python yang dipasang. Terdapat juga [Pakej Pengkodan Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) yang berguna untuk memudahkan pemasangan bagi sesetengah pengguna.

   Walau bagaimanapun, beberapa penggunaan Python memerlukan satu versi perisian, manakala yang lain memerlukan versi yang berbeza. Oleh itu, adalah berguna untuk bekerja dalam [persekitaran maya](https://docs.python.org/3/library/venv.html).

2. **Pasang Visual Studio Code**. Pastikan anda mempunyai Visual Studio Code yang dipasang pada komputer anda. Ikuti arahan ini untuk [memasang Visual Studio Code](https://code.visualstudio.com/) untuk pemasangan asas. Anda akan menggunakan Python dalam Visual Studio Code dalam kursus ini, jadi anda mungkin ingin menyegarkan pengetahuan anda tentang cara [mengkonfigurasi Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) untuk pembangunan Python.

   > Biasakan diri dengan Python dengan melalui koleksi [modul pembelajaran ini](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Sediakan Python dengan Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Sediakan Python dengan Visual Studio Code")
   >
   > 🎥 Klik imej di atas untuk video: menggunakan Python dalam VS Code.

3. **Pasang Scikit-learn**, dengan mengikuti [arahan ini](https://scikit-learn.org/stable/install.html). Oleh kerana anda perlu memastikan bahawa anda menggunakan Python 3, disarankan agar anda menggunakan persekitaran maya. Perhatikan, jika anda memasang pustaka ini pada Mac M1, terdapat arahan khas pada halaman yang dipautkan di atas.

4. **Pasang Jupyter Notebook**. Anda perlu [memasang pakej Jupyter](https://pypi.org/project/jupyter/).

## Persekitaran pengarang ML anda

Anda akan menggunakan **notebooks** untuk membangunkan kod Python anda dan mencipta model pembelajaran mesin. Jenis fail ini adalah alat biasa untuk saintis data, dan ia boleh dikenalpasti melalui akhiran atau sambungan `.ipynb`.

Notebooks adalah persekitaran interaktif yang membolehkan pembangun untuk menulis kod serta menambah nota dan dokumentasi di sekitar kod, yang sangat berguna untuk projek eksperimen atau berorientasikan penyelidikan.

[![ML untuk pemula - Sediakan Jupyter Notebooks untuk mula membina model regresi](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML untuk pemula - Sediakan Jupyter Notebooks untuk mula membina model regresi")

> 🎥 Klik imej di atas untuk video pendek melalui latihan ini.

### Latihan - bekerja dengan notebook

Dalam folder ini, anda akan menemui fail _notebook.ipynb_.

1. Buka _notebook.ipynb_ dalam Visual Studio Code.

   Pelayan Jupyter akan bermula dengan Python 3+ dimulakan. Anda akan menemui kawasan dalam notebook yang boleh `dijalankan`, iaitu potongan kod. Anda boleh menjalankan blok kod dengan memilih ikon yang kelihatan seperti butang main.

2. Pilih ikon `md` dan tambahkan sedikit markdown, serta teks berikut **# Selamat datang ke notebook anda**.

   Seterusnya, tambahkan beberapa kod Python.

3. Taip **print('hello notebook')** dalam blok kod.
4. Pilih anak panah untuk menjalankan kod.

   Anda sepatutnya melihat kenyataan yang dicetak:

    ```output
    hello notebook
    ```

![VS Code dengan notebook dibuka](../../../../2-Regression/1-Tools/images/notebook.jpg)

Anda boleh menyelitkan kod anda dengan komen untuk mendokumentasikan notebook secara sendiri.

✅ Fikirkan sebentar tentang betapa berbezanya persekitaran kerja pembangun web berbanding dengan saintis data.

## Memulakan dengan Scikit-learn

Sekarang Python telah disediakan dalam persekitaran tempatan anda, dan anda sudah selesa dengan Jupyter notebooks, mari kita menjadi sama selesa dengan Scikit-learn (disebut `sci` seperti dalam `science`). Scikit-learn menyediakan [API yang luas](https://scikit-learn.org/stable/modules/classes.html#api-ref) untuk membantu anda melaksanakan tugas ML.

Menurut [laman web mereka](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn adalah pustaka pembelajaran mesin sumber terbuka yang menyokong pembelajaran terarah dan tidak terarah. Ia juga menyediakan pelbagai alat untuk pemasangan model, prapemprosesan data, pemilihan model dan penilaian, serta banyak utiliti lain."

Dalam kursus ini, anda akan menggunakan Scikit-learn dan alat lain untuk membina model pembelajaran mesin untuk melaksanakan apa yang kita panggil tugas 'pembelajaran mesin tradisional'. Kami sengaja mengelakkan rangkaian neural dan pembelajaran mendalam, kerana ia lebih sesuai diliputi dalam kurikulum 'AI untuk Pemula' kami yang akan datang.

Scikit-learn memudahkan untuk membina model dan menilai mereka untuk digunakan. Ia terutamanya memberi tumpuan kepada penggunaan data berangka dan mengandungi beberapa dataset siap sedia untuk digunakan sebagai alat pembelajaran. Ia juga termasuk model pra-bina untuk pelajar mencuba. Mari kita terokai proses memuatkan data yang telah dipakejkan dan menggunakan estimator terbina untuk model ML pertama dengan Scikit-learn menggunakan data asas.

## Latihan - notebook Scikit-learn pertama anda

> Tutorial ini diilhamkan oleh [contoh regresi linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) di laman web Scikit-learn.

[![ML untuk pemula - Projek Regresi Linear Pertama Anda dalam Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML untuk pemula - Projek Regresi Linear Pertama Anda dalam Python")

> 🎥 Klik imej di atas untuk video pendek melalui latihan ini.

Dalam fail _notebook.ipynb_ yang berkaitan dengan pelajaran ini, kosongkan semua sel dengan menekan ikon 'tong sampah'.

Dalam bahagian ini, anda akan bekerja dengan dataset kecil tentang diabetes yang dibina dalam Scikit-learn untuk tujuan pembelajaran. Bayangkan anda ingin menguji rawatan untuk pesakit diabetes. Model Pembelajaran Mesin mungkin membantu anda menentukan pesakit mana yang akan memberi tindak balas lebih baik terhadap rawatan, berdasarkan gabungan pembolehubah. Malah model regresi yang sangat asas, apabila divisualisasikan, mungkin menunjukkan maklumat tentang pembolehubah yang akan membantu anda mengatur ujian klinikal teori anda.

✅ Terdapat banyak jenis kaedah regresi, dan yang mana anda pilih bergantung pada jawapan yang anda cari. Jika anda ingin meramalkan ketinggian yang mungkin untuk seseorang berdasarkan umur tertentu, anda akan menggunakan regresi linear, kerana anda mencari **nilai berangka**. Jika anda berminat untuk mengetahui sama ada jenis masakan harus dianggap vegan atau tidak, anda mencari **penugasan kategori**, jadi anda akan menggunakan regresi logistik. Anda akan belajar lebih lanjut tentang regresi logistik kemudian. Fikirkan sedikit tentang beberapa soalan yang boleh anda tanyakan kepada data, dan kaedah mana yang lebih sesuai.

Mari kita mulakan tugas ini.

### Import pustaka

Untuk tugas ini, kita akan mengimport beberapa pustaka:

- **matplotlib**. Ia adalah alat [grafik yang berguna](https://matplotlib.org/) dan kita akan menggunakannya untuk mencipta plot garis.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) adalah pustaka berguna untuk mengendalikan data berangka dalam Python.
- **sklearn**. Ini adalah pustaka [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Import beberapa pustaka untuk membantu tugas anda.

1. Tambahkan import dengan menaip kod berikut:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Di atas, anda mengimport `matplotlib`, `numpy` dan anda mengimport `datasets`, `linear_model` dan `model_selection` dari `sklearn`. `model_selection` digunakan untuk membahagikan data kepada set latihan dan ujian.

### Dataset diabetes

Dataset [diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) yang dibina dalam Scikit-learn termasuk 442 sampel data tentang diabetes, dengan 10 pembolehubah ciri, beberapa daripadanya termasuk:

- age: umur dalam tahun
- bmi: indeks jisim badan
- bp: tekanan darah purata
- s1 tc: T-Cells (sejenis sel darah putih)

✅ Dataset ini termasuk konsep 'sex' sebagai pembolehubah ciri yang penting untuk penyelidikan tentang diabetes. Banyak dataset perubatan termasuk jenis klasifikasi binari ini. Fikirkan sedikit tentang bagaimana pengkategorian seperti ini mungkin mengecualikan bahagian tertentu populasi daripada rawatan.

Sekarang, muatkan data X dan y.

> 🎓 Ingat, ini adalah pembelajaran terarah, dan kita memerlukan sasaran 'y' yang bernama.

Dalam sel kod baru, muatkan dataset diabetes dengan memanggil `load_diabetes()`. Input `return_X_y=True` menandakan bahawa `X` akan menjadi matriks data, dan `y` akan menjadi sasaran regresi.

1. Tambahkan beberapa arahan cetak untuk menunjukkan bentuk matriks data dan elemen pertamanya:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Apa yang anda dapat sebagai respons adalah tuple. Apa yang anda lakukan adalah menetapkan dua nilai pertama tuple kepada `X` dan `y` masing-masing. Ketahui lebih lanjut [tentang tuple](https://wikipedia.org/wiki/Tuple).

    Anda boleh melihat bahawa data ini mempunyai 442 item yang dibentuk dalam array 10 elemen:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Fikirkan sedikit tentang hubungan antara data dan sasaran regresi. Regresi linear meramalkan hubungan antara ciri X dan pembolehubah sasaran y. Bolehkah anda mencari [sasaran](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) untuk dataset diabetes dalam dokumentasi? Apa yang dataset ini tunjukkan, memandangkan sasaran?

2. Seterusnya, pilih sebahagian dataset ini untuk diplot dengan memilih lajur ke-3 dataset. Anda boleh melakukannya dengan menggunakan operator `:` untuk memilih semua baris, dan kemudian memilih lajur ke-3 menggunakan indeks (2). Anda juga boleh membentuk semula data menjadi array 2D - seperti yang diperlukan untuk plot - dengan menggunakan `reshape(n_rows, n_columns)`. Jika salah satu parameter adalah -1, dimensi yang sepadan dikira secara automatik.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Pada bila-bila masa, cetak data untuk memeriksa bentuknya.

3. Sekarang setelah anda mempunyai data yang siap untuk diplot, anda boleh melihat sama ada mesin dapat membantu menentukan pemisahan logik antara nombor dalam dataset ini. Untuk melakukan ini, anda perlu membahagikan kedua-dua data (X) dan sasaran (y) kepada set ujian dan latihan. Scikit-learn mempunyai cara yang mudah untuk melakukan ini; anda boleh membahagikan data ujian anda pada titik tertentu.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Sekarang anda bersedia untuk melatih model anda! Muatkan model regresi linear dan latih dengan set latihan X dan y anda menggunakan `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` adalah fungsi yang akan anda lihat dalam banyak pustaka ML seperti TensorFlow.

5. Kemudian, buat ramalan menggunakan data ujian, dengan menggunakan fungsi `predict()`. Ini akan digunakan untuk melukis garis antara kumpulan data model.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Sekarang tiba masanya untuk menunjukkan data dalam plot. Matplotlib adalah alat yang sangat berguna untuk tugas ini. Buat scatterplot semua data ujian X dan y, dan gunakan ramalan untuk melukis garis di tempat yang paling sesuai, antara kumpulan data model.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![scatterplot menunjukkan titik data tentang diabetes](../../../../2-Regression/1-Tools/images/scatterplot.png)
✅ Fikirkan sedikit tentang apa yang sedang berlaku di sini. Garis lurus sedang melalui banyak titik kecil data, tetapi apa sebenarnya yang sedang dilakukan? Bolehkah anda melihat bagaimana anda sepatutnya dapat menggunakan garis ini untuk meramalkan di mana titik data baru yang belum dilihat sepatutnya sesuai dalam hubungan dengan paksi y plot? Cuba nyatakan dalam kata-kata kegunaan praktikal model ini.

Tahniah, anda telah membina model regresi linear pertama anda, mencipta ramalan dengannya, dan memaparkannya dalam plot!

---
## 🚀Cabaran

Plotkan pemboleh ubah yang berbeza daripada dataset ini. Petunjuk: edit baris ini: `X = X[:,2]`. Berdasarkan sasaran dataset ini, apakah yang anda dapat temui tentang perkembangan diabetes sebagai penyakit?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Dalam tutorial ini, anda bekerja dengan regresi linear mudah, bukannya regresi univariat atau regresi berganda. Baca sedikit tentang perbezaan antara kaedah-kaedah ini, atau lihat [video ini](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Baca lebih lanjut tentang konsep regresi dan fikirkan tentang jenis soalan yang boleh dijawab oleh teknik ini. Ambil [tutorial ini](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) untuk mendalami pemahaman anda.

## Tugasan

[Dataset yang berbeza](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.