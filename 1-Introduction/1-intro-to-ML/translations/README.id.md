# Pengantar Machine Learning

[![ML, AI, deep learning - Apa perbedaannya?](https://img.youtube.com/vi/lTd9RSxS9ZE/0.jpg)](https://youtu.be/lTd9RSxS9ZE "ML, AI, deep learning - Apa perbedaannya?")

> 🎥 Klik gambar diatas untuk menonton video yang mendiskusikan perbedaan antara Machine Learning, AI, dan Deep Learning.

## [Quiz Pra-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

### Pengantar

Selamat datang di pelajaran Machine Learning klasik untuk pemula! Baik kamu yang masih benar-benar baru, atau seorang praktisi ML berpengalaman yang ingin meningkatkan kemampuan kamu, kami senang kamu ikut bersama kami! Kami ingin membuat sebuah titik mulai yang ramah untuk pembelajaran ML kamu dan akan sangat senang untuk mengevaluasi, merespon, dan memasukkan [umpan balik](https://github.com/microsoft/ML-For-Beginners/discussions) kamu.

[![Pengantar Machine Learning](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Pengantar Machine Learning")

> 🎥 Klik gambar diatas untuk menonton video: John Guttag dari MIT yang memberikan pengantar Machine Learning.
### Memulai Machine Learning

Sebelum memulai kurikulum ini, kamu perlu memastikan komputer kamu sudah dipersiapkan untuk menjalankan *notebook* secara lokal.

- **Konfigurasi komputer kamu dengan video ini**. Pelajari bagaimana menyiapkan komputer kamu dalam [video-video](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6) ini.
- **Belajar Python**. Disarankan juga untuk memiliki pemahaman dasar dari [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), sebuah bahasa pemrograman yang digunakan oleh data scientist yang juga akan kita gunakan dalam pelajaran ini. 
- **Belajar Node.js dan JavaScript**. Kita juga menggunakan JavaScript beberapa kali dalam pelajaran ini ketika membangun aplikasi web, jadi kamu perlu menginstal [node](https://nodejs.org) dan [npm](https://www.npmjs.com/), serta [Visual Studio Code](https://code.visualstudio.com/) yang tersedia untuk pengembangan Python dan JavaScript.
- **Buat akun GitHub**. Karena kamu menemukan kami di [GitHub](https://github.com), kamu mungkin sudah punya akun, tapi jika belum, silakan buat akun baru kemudian *fork* kurikulum ini untuk kamu pergunakan sendiri. (Jangan ragu untuk memberikan kami bintang juga 😊)
- **Jelajahi Scikit-learn**. Buat diri kamu familiar dengan [Scikit-learn]([https://scikit-learn.org/stable/user_guide.html), seperangkat *library* ML yang kita acu dalam pelajaran-pelajaran ini.

### Apa itu Machine Learning?

Istilah 'Machine Learning' merupakan salah satu istilah yang paling populer dan paling sering digunakan saat ini. Ada kemungkinan kamu pernah mendengar istilah ini paling tidak sekali jika kamu familiar dengan teknologi. Tetapi untuk mekanisme Machine Learning sendiri, merupakan sebuah misteri bagi sebagian besar orang. Karena itu, penting untuk memahami sebenarnya apa itu Machine Learning, dan mempelajarinya langkah demi langkah melalui contoh praktis.

![kurva tren ml](../images/hype.png)

> Google Trends memperlihatkan 'kurva tren' dari istilah 'Machine Learning' belakangan ini.

Kita hidup di sebuah alam semesta yang penuh dengan misteri yang menarik. Ilmuwan-ilmuwan besar seperti Stephen Hawking, Albert Einstein, dan banyak lagi telah mengabdikan hidup mereka untuk mencari informasi yang berarti yang mengungkap misteri dari dunia disekitar kita. Ini adalah kondisi belajar manusia: seorang anak manusia belajar hal-hal baru dan mengungkap struktur dari dunianya tahun demi tahun saat mereka tumbuh dewasa. 

Otak dan indera seorang anak memahami fakta-fakta di sekitarnya dan secara bertahap mempelajari pola-pola kehidupan yang tersembunyi yang membantu anak untuk menyusun aturan-aturan logis untuk mengidentifikasi pola-pola yang dipelajari. Proses pembelajaran otak manusia ini menjadikan manusia sebagai makhluk hidup paling canggih di dunia ini. Belajar terus menerus dengan menemukan pola-pola tersembunyi dan kemudian berinovasi pada pola-pola itu memungkinkan kita untuk terus menjadikan diri kita lebih baik sepanjang hidup. Kapasitas belajar dan kemampuan berkembang ini terkait dengan konsep yang disebut dengan *[brain plasticity](https://www.simplypsychology.org/brain-plasticity.html)*. Secara sempit, kita dapat menarik beberapa kesamaan motivasi antara proses pembelajaran otak manusia dan konsep Machine Learning.

[Otak manusia](https://www.livescience.com/29365-human-brain.html) menerima banyak hal dari dunia nyata, memproses informasi yang diterima, membuat keputusan rasional, dan melakukan aksi-aksi tertentu berdasarkan keadaan. Inilah yang kita sebut dengan berperilaku cerdas. Ketika kita memprogram sebuah salinan dari proses perilaku cerdas ke sebuah mesin, ini dinamakan kecerdasan buatan atau Artificial Intelligence (AI).

Meskipun istilah-stilahnya bisa membingungkan, Machine Learning (ML) adalah bagian penting dari Artificial Intelligence. **ML berkaitan dengan menggunakan algoritma-algoritma terspesialisasi untuk mengungkap informasi yang berarti dan mencari pola-pola tersembunyi dari data yang diterima untuk mendukung proses pembuatan keputusan rasional**.

![AI, ML, deep learning, data science](../images/ai-ml-ds.png)

> Sebuah diagram yang memperlihatkan hubungan antara AI, ML, Deep Learning, dan Data Science. Infografis oleh [Jen Looper](https://twitter.com/jenlooper) terinspirasi dari [infografis ini](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

## Apa yang akan kamu pelajari

Dalam kurikulum ini, kita hanya akan membahas konsep inti dari Machine Learning yang harus diketahui oleh seorang pemula. Kita membahas apa yang kami sebut sebagai 'Machine Learning klasik' utamanya menggunakan Scikit-learn, sebuah *library* luar biasa yang banyak digunakan para siswa untuk belajar dasarnya. Untuk memahami konsep Artificial Intelligence atau Deep Learning yang lebih luas, pengetahuan dasar yang kuat tentang Machine Learning sangat diperlukan, itulah yang ingin kami tawarkan di sini. 

Kamu akan belajar:

- Konsep inti ML
- Sejarah dari ML
- Keadilan dan ML
- Teknik regresi ML
- Teknik klasifikasi ML
- Teknik *clustering* ML
- Teknik *natural language processing* ML
- Teknik *time series forecasting* ML
- *Reinforcement learning*
- Penerapan nyata dari ML
## Yang tidak akan kita bahas

- *deep learning*
- *neural networks*
- AI

Untuk membuat pengalaman belajar yang lebih baik, kita akan menghindari kerumitan dari *neural network*, *deep learning* - membangun *many-layered model* menggunakan *neural network* - dan AI, yang mana akan kita bahas dalam kurikulum yang berbeda. Kami juga akan menawarkan kurikulum *data science* yang berfokus pada aspek bidang tersebut. 
## Kenapa belajar Machine Learning?

Machine Learning, dari perspektif sistem, didefinisikan sebagai pembuatan sistem otomatis yang dapat mempelajari pola-pola tersembunyi dari data untuk membantu membuat keputusan cerdas. 

Motivasi ini secara bebas terinspirasi dari bagaimana otak manusia mempelajari hal-hal tertentu berdasarkan data yang diterimanya dari dunia luar. 

✅ Pikirkan sejenak mengapa sebuah bisnis ingin mencoba menggunakan strategi Machine Learning dibandingkan membuat sebuah mesin berbasis aturan yang tertanam (*hard-coded*). 

### Penerapan Machine Learning

Penerapan Machine Learning saat ini hampir ada di mana-mana, seperti data yang mengalir di sekitar kita, yang dihasilkan oleh ponsel pintar, perangkat yang terhubung, dan sistem lainnya. Mempertimbangkan potensi besar dari algoritma Machine Learning terkini, para peneliti telah mengeksplorasi kemampuan Machine Learning untuk memecahkan masalah kehidupan nyata multi-dimensi dan multi-disiplin dengan hasil positif yang luar biasa. 

**Kamu bisa menggunakan Machine Learning dalam banyak hal**:

- Untuk memprediksi kemungkinan penyakit berdasarkan riwayat atau laporan medis pasien.
- Untuk memanfaatkan data cuaca untuk memprediksi peristiwa cuaca.
- Untuk memahami sentimen sebuah teks.
- Untuk mendeteksi berita palsu untuk menghentikan penyebaran propaganda.

Keuangan, ekonomi, geosains, eksplorasi ruang angkasa, teknik biomedis, ilmu kognitif, dan bahkan bidang humaniora telah mengadaptasi Machine Learning untuk memecahkan masalah sulit pemrosesan data di bidang mereka. 

Machine Learning mengotomatiskan proses penemuan pola dengan menemukan wawasan yang berarti dari dunia nyata atau dari data yang dihasilkan. Machine Learning terbukti sangat berharga dalam penerapannya di berbagai bidang, diantaranya adalah bidang bisnis, kesehatan, dan keuangan.

Dalam waktu dekat, memahami dasar-dasar Machine Learning akan menjadi suatu keharusan bagi orang-orang dari bidang apa pun karena adopsinya yang luas. 

---
## 🚀 Tantangan

Buat sketsa di atas kertas atau menggunakan aplikasi seperti [Excalidraw](https://excalidraw.com/), mengenai pemahaman kamu tentang perbedaan antara AI, ML, Deep Learning, dan Data Science. Tambahkan beberapa ide masalah yang cocok diselesaikan masing-masing teknik.

## [Quiz Pasca-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

## Ulasan & Belajar Mandiri

Untuk mempelajari lebih lanjut tentang bagaimana kamu dapat menggunakan algoritma ML di cloud, ikuti [Jalur Belajar](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott) ini. 

## Tugas

[Persiapan](assignment.id.md)
