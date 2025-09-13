<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T19:39:30+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "id"
}
-->
# Pengantar Pembelajaran Mesin

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML untuk Pemula - Pengantar Pembelajaran Mesin untuk Pemula](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML untuk Pemula - Pengantar Pembelajaran Mesin untuk Pemula")

> 🎥 Klik gambar di atas untuk video singkat yang membahas pelajaran ini.

Selamat datang di kursus pembelajaran mesin klasik untuk pemula! Baik Anda benar-benar baru dalam topik ini, atau seorang praktisi ML berpengalaman yang ingin menyegarkan pengetahuan di area tertentu, kami senang Anda bergabung dengan kami! Kami ingin menciptakan tempat awal yang ramah untuk studi ML Anda dan akan senang mengevaluasi, merespons, dan mengintegrasikan [masukan Anda](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Pengantar ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Pengantar ML")

> 🎥 Klik gambar di atas untuk video: John Guttag dari MIT memperkenalkan pembelajaran mesin

---
## Memulai dengan Pembelajaran Mesin

Sebelum memulai kurikulum ini, Anda perlu memastikan komputer Anda siap untuk menjalankan notebook secara lokal.

- **Konfigurasikan komputer Anda dengan video ini**. Gunakan tautan berikut untuk mempelajari [cara menginstal Python](https://youtu.be/CXZYvNRIAKM) di sistem Anda dan [menyiapkan editor teks](https://youtu.be/EU8eayHWoZg) untuk pengembangan.
- **Pelajari Python**. Disarankan juga untuk memiliki pemahaman dasar tentang [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), bahasa pemrograman yang berguna bagi ilmuwan data dan yang akan kita gunakan dalam kursus ini.
- **Pelajari Node.js dan JavaScript**. Kami juga menggunakan JavaScript beberapa kali dalam kursus ini saat membangun aplikasi web, jadi Anda perlu menginstal [node](https://nodejs.org) dan [npm](https://www.npmjs.com/), serta memiliki [Visual Studio Code](https://code.visualstudio.com/) untuk pengembangan Python dan JavaScript.
- **Buat akun GitHub**. Karena Anda menemukan kami di [GitHub](https://github.com), Anda mungkin sudah memiliki akun, tetapi jika belum, buatlah satu akun dan kemudian fork kurikulum ini untuk digunakan sendiri. (Jangan ragu untuk memberi kami bintang juga 😊)
- **Jelajahi Scikit-learn**. Kenali [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), kumpulan pustaka ML yang akan kita referensikan dalam pelajaran ini.

---
## Apa itu Pembelajaran Mesin?

Istilah 'pembelajaran mesin' adalah salah satu istilah yang paling populer dan sering digunakan saat ini. Ada kemungkinan besar Anda pernah mendengar istilah ini setidaknya sekali jika Anda memiliki sedikit keterkaitan dengan teknologi, tidak peduli di bidang apa Anda bekerja. Namun, mekanisme pembelajaran mesin adalah misteri bagi kebanyakan orang. Bagi pemula pembelajaran mesin, subjek ini kadang-kadang bisa terasa membingungkan. Oleh karena itu, penting untuk memahami apa sebenarnya pembelajaran mesin itu, dan mempelajarinya langkah demi langkah melalui contoh praktis.

---
## Kurva Hype

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends menunjukkan 'kurva hype' terbaru dari istilah 'pembelajaran mesin'

---
## Alam Semesta yang Misterius

Kita hidup di alam semesta yang penuh dengan misteri yang menakjubkan. Ilmuwan hebat seperti Stephen Hawking, Albert Einstein, dan banyak lainnya telah mendedikasikan hidup mereka untuk mencari informasi bermakna yang mengungkap misteri dunia di sekitar kita. Ini adalah kondisi manusia untuk belajar: seorang anak manusia belajar hal-hal baru dan mengungkap struktur dunia mereka tahun demi tahun saat mereka tumbuh dewasa.

---
## Otak Anak

Otak dan indra seorang anak merasakan fakta-fakta di sekitarnya dan secara bertahap mempelajari pola-pola tersembunyi dalam kehidupan yang membantu anak tersebut menyusun aturan logis untuk mengenali pola-pola yang telah dipelajari. Proses pembelajaran otak manusia membuat manusia menjadi makhluk hidup paling canggih di dunia ini. Belajar secara terus-menerus dengan menemukan pola-pola tersembunyi dan kemudian berinovasi berdasarkan pola-pola tersebut memungkinkan kita untuk menjadi lebih baik sepanjang hidup kita. Kapasitas belajar dan kemampuan berkembang ini terkait dengan konsep yang disebut [plastisitas otak](https://www.simplypsychology.org/brain-plasticity.html). Secara dangkal, kita dapat menarik beberapa kesamaan motivasi antara proses pembelajaran otak manusia dan konsep pembelajaran mesin.

---
## Otak Manusia

[Otak manusia](https://www.livescience.com/29365-human-brain.html) merasakan hal-hal dari dunia nyata, memproses informasi yang dirasakan, membuat keputusan rasional, dan melakukan tindakan tertentu berdasarkan keadaan. Inilah yang kita sebut berperilaku secara cerdas. Ketika kita memprogram tiruan dari proses perilaku cerdas ke sebuah mesin, itu disebut kecerdasan buatan (AI).

---
## Beberapa Terminologi

Meskipun istilah-istilah ini dapat membingungkan, pembelajaran mesin (ML) adalah subset penting dari kecerdasan buatan. **ML berkaitan dengan penggunaan algoritma khusus untuk menemukan informasi bermakna dan menemukan pola tersembunyi dari data yang dirasakan untuk mendukung proses pengambilan keputusan rasional**.

---
## AI, ML, Pembelajaran Mendalam

![AI, ML, pembelajaran mendalam, ilmu data](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram yang menunjukkan hubungan antara AI, ML, pembelajaran mendalam, dan ilmu data. Infografik oleh [Jen Looper](https://twitter.com/jenlooper) terinspirasi oleh [grafik ini](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Konsep yang Akan Dibahas

Dalam kurikulum ini, kita akan membahas hanya konsep inti pembelajaran mesin yang harus diketahui oleh pemula. Kita akan membahas apa yang kita sebut 'pembelajaran mesin klasik' terutama menggunakan Scikit-learn, pustaka yang sangat baik yang banyak digunakan oleh siswa untuk mempelajari dasar-dasarnya. Untuk memahami konsep yang lebih luas tentang kecerdasan buatan atau pembelajaran mendalam, pengetahuan dasar yang kuat tentang pembelajaran mesin sangat penting, dan kami ingin menyediakannya di sini.

---
## Dalam Kursus Ini Anda Akan Belajar:

- konsep inti pembelajaran mesin
- sejarah ML
- ML dan keadilan
- teknik regresi ML
- teknik klasifikasi ML
- teknik pengelompokan ML
- teknik pemrosesan bahasa alami ML
- teknik peramalan deret waktu ML
- pembelajaran penguatan
- aplikasi dunia nyata untuk ML

---
## Apa yang Tidak Akan Dibahas

- pembelajaran mendalam
- jaringan saraf
- AI

Untuk pengalaman belajar yang lebih baik, kami akan menghindari kompleksitas jaringan saraf, 'pembelajaran mendalam' - pembangunan model berlapis-lapis menggunakan jaringan saraf - dan AI, yang akan kita bahas dalam kurikulum yang berbeda. Kami juga akan menawarkan kurikulum ilmu data yang akan datang untuk fokus pada aspek tersebut dari bidang yang lebih besar ini.

---
## Mengapa Mempelajari Pembelajaran Mesin?

Pembelajaran mesin, dari perspektif sistem, didefinisikan sebagai pembuatan sistem otomatis yang dapat mempelajari pola tersembunyi dari data untuk membantu dalam membuat keputusan yang cerdas.

Motivasi ini secara longgar terinspirasi oleh bagaimana otak manusia mempelajari hal-hal tertentu berdasarkan data yang dirasakannya dari dunia luar.

✅ Pikirkan sejenak mengapa sebuah bisnis ingin mencoba menggunakan strategi pembelajaran mesin dibandingkan dengan membuat mesin berbasis aturan yang dikodekan secara manual.

---
## Aplikasi Pembelajaran Mesin

Aplikasi pembelajaran mesin sekarang hampir ada di mana-mana, dan sama melimpahnya dengan data yang mengalir di sekitar masyarakat kita, yang dihasilkan oleh ponsel pintar, perangkat yang terhubung, dan sistem lainnya. Mengingat potensi besar algoritma pembelajaran mesin mutakhir, para peneliti telah mengeksplorasi kemampuannya untuk menyelesaikan masalah kehidupan nyata yang multi-dimensi dan multi-disiplin dengan hasil yang sangat positif.

---
## Contoh Penerapan ML

**Anda dapat menggunakan pembelajaran mesin dalam berbagai cara**:

- Untuk memprediksi kemungkinan penyakit dari riwayat medis atau laporan pasien.
- Untuk memanfaatkan data cuaca guna memprediksi peristiwa cuaca.
- Untuk memahami sentimen dari sebuah teks.
- Untuk mendeteksi berita palsu guna menghentikan penyebaran propaganda.

Keuangan, ekonomi, ilmu bumi, eksplorasi luar angkasa, teknik biomedis, ilmu kognitif, dan bahkan bidang humaniora telah mengadaptasi pembelajaran mesin untuk menyelesaikan masalah berat yang melibatkan pemrosesan data di domain mereka.

---
## Kesimpulan

Pembelajaran mesin mengotomatisasi proses penemuan pola dengan menemukan wawasan bermakna dari data dunia nyata atau data yang dihasilkan. Ini telah terbukti sangat berharga dalam aplikasi bisnis, kesehatan, dan keuangan, di antara lainnya.

Di masa depan, memahami dasar-dasar pembelajaran mesin akan menjadi keharusan bagi orang-orang dari berbagai bidang karena adopsinya yang luas.

---
# 🚀 Tantangan

Gambarkan, di atas kertas atau menggunakan aplikasi online seperti [Excalidraw](https://excalidraw.com/), pemahaman Anda tentang perbedaan antara AI, ML, pembelajaran mendalam, dan ilmu data. Tambahkan beberapa ide tentang masalah yang baik untuk diselesaikan oleh masing-masing teknik ini.

# [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

---
# Tinjauan & Studi Mandiri

Untuk mempelajari lebih lanjut tentang bagaimana Anda dapat bekerja dengan algoritma ML di cloud, ikuti [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott) ini.

Ikuti [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) tentang dasar-dasar ML.

---
# Tugas

[Mulai dan jalankan](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang timbul dari penggunaan terjemahan ini.