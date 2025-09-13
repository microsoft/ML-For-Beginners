<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T19:36:18+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "id"
}
-->
# Teknik Pembelajaran Mesin

Proses membangun, menggunakan, dan memelihara model pembelajaran mesin serta data yang digunakan sangat berbeda dari banyak alur kerja pengembangan lainnya. Dalam pelajaran ini, kita akan mengungkap proses tersebut dan merangkum teknik utama yang perlu Anda ketahui. Anda akan:

- Memahami proses yang mendasari pembelajaran mesin pada tingkat tinggi.
- Mengeksplorasi konsep dasar seperti 'model', 'prediksi', dan 'data pelatihan'.

## [Kuis sebelum pelajaran](https://ff-quizzes.netlify.app/en/ml/)

[![ML untuk pemula - Teknik Pembelajaran Mesin](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML untuk pemula - Teknik Pembelajaran Mesin")

> 🎥 Klik gambar di atas untuk video singkat yang membahas pelajaran ini.

## Pendahuluan

Secara umum, seni menciptakan proses pembelajaran mesin (ML) terdiri dari beberapa langkah:

1. **Tentukan pertanyaan**. Sebagian besar proses ML dimulai dengan mengajukan pertanyaan yang tidak dapat dijawab oleh program bersyarat sederhana atau mesin berbasis aturan. Pertanyaan ini sering kali berkisar pada prediksi berdasarkan kumpulan data.
2. **Kumpulkan dan siapkan data**. Untuk dapat menjawab pertanyaan Anda, Anda memerlukan data. Kualitas dan, terkadang, kuantitas data Anda akan menentukan seberapa baik Anda dapat menjawab pertanyaan awal. Memvisualisasikan data adalah aspek penting dari fase ini. Fase ini juga mencakup pembagian data menjadi kelompok pelatihan dan pengujian untuk membangun model.
3. **Pilih metode pelatihan**. Bergantung pada pertanyaan Anda dan sifat data Anda, Anda perlu memilih cara melatih model agar dapat mencerminkan data Anda dengan baik dan membuat prediksi yang akurat. Bagian dari proses ML ini membutuhkan keahlian khusus dan, sering kali, sejumlah besar eksperimen.
4. **Latih model**. Dengan menggunakan data pelatihan Anda, Anda akan menggunakan berbagai algoritma untuk melatih model agar mengenali pola dalam data. Model mungkin memanfaatkan bobot internal yang dapat disesuaikan untuk memprioritaskan bagian tertentu dari data dibandingkan yang lain guna membangun model yang lebih baik.
5. **Evaluasi model**. Anda menggunakan data yang belum pernah dilihat sebelumnya (data pengujian Anda) dari kumpulan yang dikumpulkan untuk melihat bagaimana kinerja model.
6. **Penyetelan parameter**. Berdasarkan kinerja model Anda, Anda dapat mengulangi proses menggunakan parameter atau variabel yang berbeda yang mengontrol perilaku algoritma yang digunakan untuk melatih model.
7. **Prediksi**. Gunakan input baru untuk menguji akurasi model Anda.

## Pertanyaan yang harus diajukan

Komputer sangat mahir dalam menemukan pola tersembunyi dalam data. Kemampuan ini sangat membantu bagi peneliti yang memiliki pertanyaan tentang suatu domain tertentu yang tidak dapat dengan mudah dijawab dengan membuat mesin berbasis aturan bersyarat. Dalam tugas aktuaria, misalnya, seorang ilmuwan data mungkin dapat membangun aturan buatan seputar tingkat kematian perokok vs non-perokok.

Namun, ketika banyak variabel lain dimasukkan ke dalam persamaan, model ML mungkin lebih efisien untuk memprediksi tingkat kematian di masa depan berdasarkan riwayat kesehatan sebelumnya. Contoh yang lebih ceria mungkin adalah membuat prediksi cuaca untuk bulan April di lokasi tertentu berdasarkan data yang mencakup garis lintang, garis bujur, perubahan iklim, kedekatan dengan laut, pola aliran jet, dan lainnya.

✅ [Slide presentasi ini](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) tentang model cuaca menawarkan perspektif historis tentang penggunaan ML dalam analisis cuaca.  

## Tugas sebelum membangun

Sebelum mulai membangun model Anda, ada beberapa tugas yang perlu Anda selesaikan. Untuk menguji pertanyaan Anda dan membentuk hipotesis berdasarkan prediksi model, Anda perlu mengidentifikasi dan mengonfigurasi beberapa elemen.

### Data

Untuk dapat menjawab pertanyaan Anda dengan tingkat kepastian apa pun, Anda memerlukan sejumlah data yang cukup dan jenis data yang tepat. Ada dua hal yang perlu Anda lakukan pada tahap ini:

- **Kumpulkan data**. Dengan mengingat pelajaran sebelumnya tentang keadilan dalam analisis data, kumpulkan data Anda dengan hati-hati. Perhatikan sumber data ini, bias bawaan yang mungkin dimilikinya, dan dokumentasikan asalnya.
- **Siapkan data**. Ada beberapa langkah dalam proses persiapan data. Anda mungkin perlu menggabungkan data dan menormalkannya jika berasal dari berbagai sumber. Anda dapat meningkatkan kualitas dan kuantitas data melalui berbagai metode seperti mengonversi string menjadi angka (seperti yang kita lakukan dalam [Clustering](../../5-Clustering/1-Visualize/README.md)). Anda juga dapat menghasilkan data baru berdasarkan data asli (seperti yang kita lakukan dalam [Classification](../../4-Classification/1-Introduction/README.md)). Anda dapat membersihkan dan mengedit data (seperti yang akan kita lakukan sebelum pelajaran [Web App](../../3-Web-App/README.md)). Akhirnya, Anda mungkin juga perlu mengacak dan mengacaknya, tergantung pada teknik pelatihan Anda.

✅ Setelah mengumpulkan dan memproses data Anda, luangkan waktu untuk melihat apakah bentuknya memungkinkan Anda menjawab pertanyaan yang dimaksud. Mungkin saja data tidak akan bekerja dengan baik dalam tugas yang diberikan, seperti yang kita temukan dalam pelajaran [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Fitur dan Target

[Fitur](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) adalah properti yang dapat diukur dari data Anda. Dalam banyak kumpulan data, fitur diekspresikan sebagai judul kolom seperti 'tanggal', 'ukuran', atau 'warna'. Variabel fitur Anda, biasanya direpresentasikan sebagai `X` dalam kode, mewakili variabel input yang akan digunakan untuk melatih model.

Target adalah hal yang Anda coba prediksi. Target, biasanya direpresentasikan sebagai `y` dalam kode, mewakili jawaban atas pertanyaan yang Anda coba ajukan dari data Anda: pada bulan Desember, **warna** labu mana yang akan paling murah? Di San Francisco, lingkungan mana yang akan memiliki **harga** real estat terbaik? Kadang-kadang target juga disebut sebagai atribut label.

### Memilih variabel fitur Anda

🎓 **Pemilihan Fitur dan Ekstraksi Fitur** Bagaimana Anda tahu variabel mana yang harus dipilih saat membangun model? Anda mungkin akan melalui proses pemilihan fitur atau ekstraksi fitur untuk memilih variabel yang tepat untuk model yang paling berkinerja. Namun, keduanya tidak sama: "Ekstraksi fitur menciptakan fitur baru dari fungsi fitur asli, sedangkan pemilihan fitur mengembalikan subset dari fitur tersebut." ([sumber](https://wikipedia.org/wiki/Feature_selection))

### Visualisasikan data Anda

Aspek penting dari alat ilmuwan data adalah kemampuan untuk memvisualisasikan data menggunakan beberapa pustaka yang sangat baik seperti Seaborn atau MatPlotLib. Mewakili data Anda secara visual mungkin memungkinkan Anda menemukan korelasi tersembunyi yang dapat Anda manfaatkan. Visualisasi Anda juga dapat membantu Anda menemukan bias atau data yang tidak seimbang (seperti yang kita temukan dalam [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Pisahkan dataset Anda

Sebelum pelatihan, Anda perlu membagi dataset Anda menjadi dua atau lebih bagian dengan ukuran yang tidak sama yang tetap mewakili data dengan baik.

- **Pelatihan**. Bagian dataset ini digunakan untuk melatih model Anda. Set ini merupakan mayoritas dari dataset asli.
- **Pengujian**. Dataset pengujian adalah kelompok data independen, sering kali dikumpulkan dari data asli, yang Anda gunakan untuk mengonfirmasi kinerja model yang dibangun.
- **Validasi**. Set validasi adalah kelompok contoh independen yang lebih kecil yang Anda gunakan untuk menyetel hiperparameter atau arsitektur model untuk meningkatkan model. Bergantung pada ukuran data Anda dan pertanyaan yang Anda ajukan, Anda mungkin tidak perlu membangun set ketiga ini (seperti yang kita catat dalam [Time Series Forecasting](../../7-TimeSeries/1-Introduction/README.md)).

## Membangun model

Dengan menggunakan data pelatihan Anda, tujuan Anda adalah membangun model, atau representasi statistik dari data Anda, menggunakan berbagai algoritma untuk **melatih** model tersebut. Melatih model memaparkannya pada data dan memungkinkan model membuat asumsi tentang pola yang ditemukan, memvalidasi, dan menerima atau menolak.

### Tentukan metode pelatihan

Bergantung pada pertanyaan Anda dan sifat data Anda, Anda akan memilih metode untuk melatihnya. Dengan menjelajahi [dokumentasi Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - yang kita gunakan dalam kursus ini - Anda dapat mengeksplorasi banyak cara untuk melatih model. Bergantung pada pengalaman Anda, Anda mungkin harus mencoba beberapa metode berbeda untuk membangun model terbaik. Anda kemungkinan akan melalui proses di mana ilmuwan data mengevaluasi kinerja model dengan memberinya data yang belum pernah dilihat sebelumnya, memeriksa akurasi, bias, dan masalah kualitas lainnya, serta memilih metode pelatihan yang paling sesuai untuk tugas yang diberikan.

### Latih model

Dengan data pelatihan Anda, Anda siap untuk 'memasangkannya' untuk membuat model. Anda akan melihat bahwa dalam banyak pustaka ML, Anda akan menemukan kode 'model.fit' - pada saat inilah Anda mengirimkan variabel fitur Anda sebagai array nilai (biasanya 'X') dan variabel target (biasanya 'y').

### Evaluasi model

Setelah proses pelatihan selesai (dapat memakan waktu banyak iterasi, atau 'epoch', untuk melatih model besar), Anda akan dapat mengevaluasi kualitas model dengan menggunakan data pengujian untuk mengukur kinerjanya. Data ini adalah subset dari data asli yang belum pernah dianalisis oleh model sebelumnya. Anda dapat mencetak tabel metrik tentang kualitas model Anda.

🎓 **Pemasangan model**

Dalam konteks pembelajaran mesin, pemasangan model mengacu pada akurasi fungsi dasar model saat mencoba menganalisis data yang tidak dikenalnya.

🎓 **Underfitting** dan **overfitting** adalah masalah umum yang menurunkan kualitas model, karena model tidak cocok dengan baik atau terlalu cocok. Hal ini menyebabkan model membuat prediksi yang terlalu selaras atau terlalu longgar dengan data pelatihannya. Model yang terlalu cocok memprediksi data pelatihan terlalu baik karena telah mempelajari detail dan kebisingan data terlalu baik. Model yang kurang cocok tidak akurat karena tidak dapat menganalisis data pelatihannya maupun data yang belum pernah 'dilihat' dengan akurat.

![model overfitting](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

## Penyetelan parameter

Setelah pelatihan awal Anda selesai, amati kualitas model dan pertimbangkan untuk meningkatkannya dengan menyetel 'hiperparameter'-nya. Baca lebih lanjut tentang proses ini [dalam dokumentasi](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksi

Ini adalah momen di mana Anda dapat menggunakan data yang benar-benar baru untuk menguji akurasi model Anda. Dalam pengaturan ML 'terapan', di mana Anda membangun aset web untuk menggunakan model dalam produksi, proses ini mungkin melibatkan pengumpulan input pengguna (misalnya, menekan tombol) untuk menetapkan variabel dan mengirimkannya ke model untuk inferensi atau evaluasi.

Dalam pelajaran ini, Anda akan menemukan cara menggunakan langkah-langkah ini untuk mempersiapkan, membangun, menguji, mengevaluasi, dan memprediksi - semua gerakan seorang ilmuwan data dan lebih banyak lagi, saat Anda maju dalam perjalanan Anda untuk menjadi seorang insinyur ML 'full stack'.

---

## 🚀Tantangan

Buat diagram alur yang mencerminkan langkah-langkah seorang praktisi ML. Di mana Anda melihat diri Anda saat ini dalam proses tersebut? Di mana Anda memprediksi akan menemukan kesulitan? Apa yang tampaknya mudah bagi Anda?

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Cari wawancara online dengan ilmuwan data yang membahas pekerjaan harian mereka. Berikut adalah [salah satunya](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tugas

[Wawancarai seorang ilmuwan data](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.