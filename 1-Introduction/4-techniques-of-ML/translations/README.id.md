# Teknik-teknik Machine Learning

Proses membangun, menggunakan, dan memelihara model machine learning dan data yang digunakan adalah proses yang sangat berbeda dari banyak alur kerja pengembangan lainnya. Dalam pelajaran ini, kita akan mengungkap prosesnya dan menguraikan teknik utama yang perlu Kamu ketahui. Kamu akan: 

- Memahami gambaran dari proses yang mendasari machine learning.
- Menjelajahi konsep dasar seperti '*models*', '*predictions*', dan '*training data*'. 
  
## [Quiz Pra-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/)
## Pengantar

Gambaran membuat proses machine learning (ML) terdiri dari sejumlah langkah: 

1. **Menentukan pertanyaan**. Sebagian besar proses ML dimulai dengan mengajukan pertanyaan yang tidak dapat dijawab oleh program kondisional sederhana atau mesin berbasis aturan (*rules-based engine*). Pertanyaan-pertanyaan ini sering berkisar seputar prediksi berdasarkan kumpulan data.
2. **Mengumpulkan dan menyiapkan data**. Untuk dapat menjawab pertanyaanmu, Kamu memerlukan data. Bagaimana kualitas dan terkadang kuantitas data kamu akan menentukan seberapa baik kamu dapat menjawab pertanyaan awal kamu. Memvisualisasikan data merupakan aspek penting dari fase ini. Fase ini juga mencakup pemisahan data menjadi kelompok *training* dan *testing* untuk membangun model.
3. **Memilih metode training**. Tergantung dari pertanyaan dan sifat datamu, Kamu perlu memilih bagaimana kamu ingin men-training sebuah model untuk mencerminkan data kamu dengan baik dan membuat prediksi yang akurat terhadapnya. Ini adalah bagian dari proses ML yang membutuhkan keahlian khusus dan seringkali perlu banyak eksperimen.
4. **Melatih model**. Dengan menggunakan data *training*, kamu akan menggunakan berbagai algoritma untuk melatih model guna mengenali pola dalam data. Modelnya mungkin bisa memanfaatkan *internal weight* yang dapat disesuaikan untuk memberi hak istimewa pada bagian tertentu dari data dibandingkan bagian lainnya untuk membangun model yang lebih baik. 
5. **Mengevaluasi model**. Gunakan data yang belum pernah dilihat sebelumnya (data *testing*) untuk melihat bagaimana kinerja model. 
6. **Parameter tuning**. Berdasarkan kinerja modelmu, Kamu dapat mengulang prosesnya menggunakan parameter atau variabel yang berbeda, yang mengontrol perilaku algoritma yang digunakan untuk melatih model. 
7. **Prediksi**. Gunakan input baru untuk menguji keakuratan model kamu. 

## Pertanyaan apa yang harus ditanyakan?

Komputer sangat ahli dalam menemukan pola tersembunyi dalam data. Hal ini sangat membantu peneliti yang memiliki pertanyaan tentang domain tertentu yang tidak dapat dijawab dengan mudah dari hanya membuat mesin berbasis aturan kondisional (*conditionally-based rules engine*). Untuk tugas aktuaria misalnya, seorang data scientist mungkin dapat membuat aturan secara manual seputar mortalitas perokok vs non-perokok. 

Namun, ketika banyak variabel lain dimasukkan ke dalam persamaan, model ML mungkin terbukti lebih efisien untuk memprediksi tingkat mortalitas di masa depan berdasarkan riwayat kesehatan masa lalu. Contoh yang lebih menyenangkan mungkin membuat prediksi cuaca untuk bulan April di lokasi tertentu berdasarkan data yang mencakup garis lintang, garis bujur, perubahan iklim, kedekatan dengan laut, pola aliran udara (Jet Stream), dan banyak lagi. 

✅ [Slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) ini menawarkan perspektif historis pada model cuaca dengan menggunakan ML dalam analisis cuaca.

## Tugas Pra-Pembuatan

Sebelum mulai membangun model kamu, ada beberapa tugas yang harus kamu selesaikan. Untuk menguji pertanyaan kamu dan membentuk hipotesis berdasarkan prediksi model, Kamu perlu mengidentifikasi dan mengonfigurasi beberapa elemen. 

### Data

Untuk dapat menjawab pertanyaan kamu dengan kepastian, Kamu memerlukan sejumlah besar data dengan jenis yang tepat. Ada dua hal yang perlu kamu lakukan pada saat ini: 

- **Mengumpulkan data**. Ingat pelajaran sebelumnya tentang keadilan dalam analisis data, kumpulkan data kamu dengan hati-hati. Waspadai sumber datanya, bias bawaan apa pun yang mungkin dimiliki, dan dokumentasikan asalnya. 
- **Menyiapkan data**. Ada beberapa langkah dalam proses persiapan data. Kamu mungkin perlu menyusun data dan melakukan normalisasi jika berasal dari berbagai sumber. Kamu dapat meningkatkan kualitas dan kuantitas data melalui berbagai metode seperti mengonversi string menjadi angka (seperti yang kita lakukan di [Clustering](../../5-Clustering/1-Visualize/translations/README.id.md)). Kamu mungkin juga bisa membuat data baru berdasarkan data yang asli (seperti yang kita lakukan di [Classification](../../4-Classification/1-Introduction/translations/README.id.md)). Kamu bisa membersihkan dan mengubah data (seperti yang kita lakukan sebelum pelajaran [Web App](../3-Web-App/translations/README.id.md)). Terakhir, Kamu mungkin juga perlu mengacaknya dan mengubah urutannya, tergantung pada teknik *training* kamu. 

✅ Setelah mengumpulkan dan memproses data kamu, luangkan waktu sejenak untuk melihat apakah bentuknya memungkinkan kamu untuk menjawab pertanyaan yang kamu maksudkan. Mungkin data tidak akan berkinerja baik dalam tugas yang kamu berikan, seperti yang kita temukan dalam pelajaran [Clustering](../../5-Clustering/1-Visualize/translations/README.id.md).

### Fitur dan Target

Fitur adalah properti terukur dari data Anda. Dalam banyak set data, data tersebut dinyatakan sebagai judul kolom seperti 'date' 'size' atau 'color'. Variabel fitur Anda, biasanya direpresentasikan sebagai `X` dalam kode, mewakili variabel input yang akan digunakan untuk melatih model.

A target is a thing you are trying to predict. Target usually represented as `y` in code, represents the answer to the question you are trying to ask of your data: in December, what color pumpkins will be cheapest? in San Francisco, what neighborhoods will have the best real estate price? Sometimes target is also referred as label attribute.

### Memilih variabel fiturmu

🎓 **Feature Selection dan Feature Extraction** Bagaimana kamu tahu variabel mana yang harus dipilih saat membangun model? Kamu mungkin akan melalui proses pemilihan fitur (*Feature Selection*) atau ekstraksi fitur (*Feature Extraction*) untuk memilih variabel yang tepat untuk membuat model yang berkinerja paling baik. Namun, keduanya tidak sama: "Ekstraksi fitur membuat fitur baru dari fungsi fitur asli, sedangkan pemilihan fitur mengembalikan subset fitur." ([sumber](https://wikipedia.org/wiki/Feature_selection))
### Visualisasikan datamu

Aspek penting dari toolkit data scientist adalah kemampuan untuk memvisualisasikan data menggunakan beberapa *library* seperti Seaborn atau MatPlotLib. Merepresentasikan data kamu secara visual memungkinkan kamu mengungkap korelasi tersembunyi yang dapat kamu manfaatkan. Visualisasimu mungkin juga membantu kamu mengungkap data yang bias atau tidak seimbang (seperti yang kita temukan dalam [Classification](../../4-Classification/2-Classifiers-1/translations/README.id.md)).
### Membagi dataset

Sebelum memulai *training*, Kamu perlu membagi dataset menjadi dua atau lebih bagian dengan ukuran yang tidak sama tapi masih mewakili data dengan baik.

- **Training**. Bagian dataset ini digunakan untuk men-training model kamu. Bagian dataset ini merupakan mayoritas dari dataset asli. 
- **Testing**. Sebuah dataset tes adalah kelompok data independen, seringkali dikumpulkan dari data yang asli yang akan digunakan untuk mengkonfirmasi kinerja dari model yang dibuat.
- **Validating**. Dataset validasi adalah kumpulan contoh mandiri yang lebih kecil yang kamu gunakan untuk menyetel hyperparameter atau arsitektur model untuk meningkatkan model. Tergantung dari ukuran data dan pertanyaan yang kamu ajukan, Kamu mungkin tidak perlu membuat dataset ketiga ini (seperti yang kita catat dalam [Time Series Forecasting](../7-TimeSeries/1-Introduction/translations/README.id.md)).

## Membuat sebuah model

Dengan menggunakan data *training*, tujuan kamu adalah membuat model atau representasi statistik data kamu menggunakan berbagai algoritma untuk **melatihnya**. Melatih model berarti mengeksposnya dengan data dan mengizinkannya membuat asumsi tentang pola yang ditemukan, divalidasi, dan diterima atau ditolak. 

### Tentukan metode training

Tergantung dari pertanyaan dan sifat datamu, Kamu akan memilih metode untuk melatihnya. Buka dokumentasi [Scikit-learn](https://scikit-learn.org/stable/user_guide.html) yang kita gunakan dalam pelajaran ini, kamu bisa menjelajahi banyak cara untuk melatih sebuah model. Tergantung dari pengalamanmu, kamu mungkin perlu mencoba beberapa metode yang berbeda untuk membuat model yang terbaik. Kemungkinan kamu akan melalui proses di mana data scientist mengevaluasi kinerja model dengan memasukkan data yang belum pernah dilihat, memeriksa akurasi, bias, dan masalah penurunan kualitas lainnya, dan memilih metode training yang paling tepat untuk tugas yang ada.

### Melatih sebuah model

Berbekan dengan data pelatihan Anda, Anda siap untuk 'menyesuaikan' untuk membuat model. Anda akan melihat bahwa di banyak perpustakaan ML Anda akan menemukan kode 'model.fit' - saat inilah Anda mengirim variabel fitur Anda sebagai array nilai (biasanya `X`) dan variabel target (biasanya `y`).

### Mengevaluasi model

Setelah proses *training* selesai (ini mungkin membutuhkan banyak iterasi, atau 'epoch', untuk melatih model besar), Kamu akan dapat mengevaluasi kualitas model dengan menggunakan data tes untuk mengukur kinerjanya. Data ini merupakan subset dari data asli yang modelnya belum pernah dianalisis sebelumnya. Kamu dapat mencetak tabel metrik tentang kualitas model kamu. 

🎓 **Model fitting**

Dalam konteks machine learning, *model fitting* mengacu pada keakuratan dari fungsi yang mendasari model saat mencoba menganalisis data yang tidak familiar. 

🎓 **Underfitting** dan **overfitting** adalah masalah umum yang menurunkan kualitas model, karena model tidak cukup akurat atau terlalu akurat. Hal ini menyebabkan model membuat prediksi yang terlalu selaras atau tidak cukup selaras dengan data trainingnya. Model overfit memprediksi data *training* terlalu baik karena telah mempelajari detail dan noise data dengan terlalu baik. Model underfit tidak akurat karena tidak dapat menganalisis data *training* atau data yang belum pernah dilihat sebelumnya secara akurat. 

![overfitting model](../images/overfitting.png)
> Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

## Parameter tuning

Setelah *training* awal selesai, amati kualitas model dan pertimbangkan untuk meningkatkannya dengan mengubah 'hyperparameter' nya. Baca lebih lanjut tentang prosesnya [di dalam dokumentasi](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediksi

Ini adalah saat di mana Kamu dapat menggunakan data yang sama sekali baru untuk menguji akurasi model kamu. Dalam setelan ML 'terapan', di mana kamu membangun aset web untuk menggunakan modelnya dalam produksi, proses ini mungkin melibatkan pengumpulan input pengguna (misalnya menekan tombol) untuk menyetel variabel dan mengirimkannya ke model untuk inferensi, atau evaluasi. 

Dalam pelajaran ini, Kamu akan menemukan cara untuk menggunakan langkah-langkah ini untuk mempersiapkan, membangun, menguji, mengevaluasi, dan memprediksi - semua gestur data scientist dan banyak lagi, seiring kemajuanmu dalam perjalanan menjadi 'full stack' ML engineer. 

---

## 🚀Tantangan

Gambarlah sebuah flow chart yang mencerminkan langkah-langkah seorang praktisi ML. Di mana kamu melihat diri kamu saat ini dalam prosesnya? Di mana kamu memprediksi kamu akan menemukan kesulitan? Apa yang tampak mudah bagi kamu? 

## [Quiz Pra-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/)

## Ulasan & Belajar Mandiri

Cari di Internet mengenai wawancara dengan data scientist yang mendiskusikan pekerjaan sehari-hari mereka. Ini [salah satunya](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tugas

[Wawancara dengan data scientist](assignment.id.md)
