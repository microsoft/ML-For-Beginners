<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T19:36:47+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "ms"
}
-->
# Teknik Pembelajaran Mesin

Proses membina, menggunakan, dan mengekalkan model pembelajaran mesin serta data yang digunakan adalah sangat berbeza daripada banyak aliran kerja pembangunan lain. Dalam pelajaran ini, kita akan menjelaskan proses tersebut dan menggariskan teknik utama yang perlu anda ketahui. Anda akan:

- Memahami proses yang mendasari pembelajaran mesin pada tahap tinggi.
- Meneroka konsep asas seperti 'model', 'ramalan', dan 'data latihan'.

## [Kuiz pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

[![ML untuk pemula - Teknik Pembelajaran Mesin](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML untuk pemula - Teknik Pembelajaran Mesin")

> 🎥 Klik imej di atas untuk video pendek yang menerangkan pelajaran ini.

## Pengenalan

Secara umum, seni mencipta proses pembelajaran mesin (ML) terdiri daripada beberapa langkah:

1. **Tentukan soalan**. Kebanyakan proses ML bermula dengan menanyakan soalan yang tidak dapat dijawab oleh program bersyarat atau enjin berasaskan peraturan yang mudah. Soalan-soalan ini sering berkisar pada ramalan berdasarkan koleksi data.
2. **Kumpul dan sediakan data**. Untuk menjawab soalan anda, anda memerlukan data. Kualiti dan, kadangkala, kuantiti data anda akan menentukan sejauh mana anda dapat menjawab soalan awal anda. Memvisualkan data adalah aspek penting dalam fasa ini. Fasa ini juga termasuk membahagikan data kepada kumpulan latihan dan ujian untuk membina model.
3. **Pilih kaedah latihan**. Bergantung pada soalan anda dan sifat data anda, anda perlu memilih cara untuk melatih model agar mencerminkan data anda dengan baik dan membuat ramalan yang tepat. Bahagian proses ML ini memerlukan kepakaran khusus dan, sering kali, sejumlah besar eksperimen.
4. **Latih model**. Menggunakan data latihan anda, anda akan menggunakan pelbagai algoritma untuk melatih model agar mengenali pola dalam data. Model mungkin menggunakan berat dalaman yang boleh disesuaikan untuk memberi keutamaan kepada bahagian tertentu data berbanding yang lain untuk membina model yang lebih baik.
5. **Nilai model**. Anda menggunakan data yang belum pernah dilihat sebelumnya (data ujian anda) daripada set yang dikumpulkan untuk melihat bagaimana prestasi model.
6. **Penalaan parameter**. Berdasarkan prestasi model anda, anda boleh mengulangi proses menggunakan parameter atau pembolehubah yang berbeza yang mengawal tingkah laku algoritma yang digunakan untuk melatih model.
7. **Ramalkan**. Gunakan input baru untuk menguji ketepatan model anda.

## Soalan yang perlu ditanya

Komputer sangat mahir dalam menemui pola tersembunyi dalam data. Kegunaan ini sangat membantu penyelidik yang mempunyai soalan tentang domain tertentu yang tidak dapat dijawab dengan mudah dengan mencipta enjin peraturan bersyarat. Sebagai contoh, dalam tugas aktuari, seorang saintis data mungkin dapat membina peraturan buatan tangan tentang kadar kematian perokok berbanding bukan perokok.

Namun, apabila banyak pembolehubah lain dimasukkan ke dalam persamaan, model ML mungkin lebih efisien untuk meramalkan kadar kematian masa depan berdasarkan sejarah kesihatan masa lalu. Contoh yang lebih ceria mungkin membuat ramalan cuaca untuk bulan April di lokasi tertentu berdasarkan data yang merangkumi latitud, longitud, perubahan iklim, jarak ke laut, pola aliran jet, dan banyak lagi.

✅ [Slide ini](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) tentang model cuaca menawarkan perspektif sejarah untuk menggunakan ML dalam analisis cuaca.  

## Tugas sebelum membina

Sebelum memulakan pembinaan model anda, terdapat beberapa tugas yang perlu anda selesaikan. Untuk menguji soalan anda dan membentuk hipotesis berdasarkan ramalan model, anda perlu mengenal pasti dan mengkonfigurasi beberapa elemen.

### Data

Untuk menjawab soalan anda dengan sebarang kepastian, anda memerlukan sejumlah data yang mencukupi dan jenis yang betul. Terdapat dua perkara yang perlu dilakukan pada tahap ini:

- **Kumpul data**. Mengambil kira pelajaran sebelumnya tentang keadilan dalam analisis data, kumpulkan data anda dengan berhati-hati. Sedar akan sumber data ini, sebarang bias yang mungkin ada, dan dokumentasikan asal usulnya.
- **Sediakan data**. Terdapat beberapa langkah dalam proses penyediaan data. Anda mungkin perlu menggabungkan data dan menormalkannya jika ia berasal dari sumber yang pelbagai. Anda boleh meningkatkan kualiti dan kuantiti data melalui pelbagai kaedah seperti menukar string kepada nombor (seperti yang kita lakukan dalam [Pengelompokan](../../5-Clustering/1-Visualize/README.md)). Anda juga boleh menghasilkan data baru berdasarkan data asal (seperti yang kita lakukan dalam [Klasifikasi](../../4-Classification/1-Introduction/README.md)). Anda boleh membersihkan dan mengedit data (seperti yang akan kita lakukan sebelum pelajaran [Aplikasi Web](../../3-Web-App/README.md)). Akhirnya, anda mungkin juga perlu mengacak dan mencampurkannya, bergantung pada teknik latihan anda.

✅ Selepas mengumpul dan memproses data anda, luangkan masa untuk melihat sama ada bentuknya akan membolehkan anda menangani soalan yang dimaksudkan. Mungkin data tersebut tidak akan berfungsi dengan baik dalam tugas yang diberikan, seperti yang kita temui dalam pelajaran [Pengelompokan](../../5-Clustering/1-Visualize/README.md)!

### Ciri dan Sasaran

[Ciri](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) adalah sifat yang boleh diukur daripada data anda. Dalam banyak set data, ia dinyatakan sebagai tajuk lajur seperti 'tarikh', 'saiz', atau 'warna'. Pembolehubah ciri anda, biasanya diwakili sebagai `X` dalam kod, mewakili pembolehubah input yang akan digunakan untuk melatih model.

Sasaran adalah perkara yang anda cuba ramalkan. Sasaran, biasanya diwakili sebagai `y` dalam kod, mewakili jawapan kepada soalan yang anda cuba tanyakan kepada data anda: pada bulan Disember, **warna** labu mana yang akan paling murah? Di San Francisco, kawasan kejiranan mana yang akan mempunyai **harga** hartanah terbaik? Kadangkala sasaran juga dirujuk sebagai atribut label.

### Memilih pembolehubah ciri anda

🎓 **Pemilihan Ciri dan Ekstraksi Ciri** Bagaimana anda tahu pembolehubah mana yang perlu dipilih semasa membina model? Anda mungkin akan melalui proses pemilihan ciri atau ekstraksi ciri untuk memilih pembolehubah yang betul untuk model yang paling berprestasi. Walau bagaimanapun, mereka tidak sama: "Ekstraksi ciri mencipta ciri baru daripada fungsi ciri asal, manakala pemilihan ciri mengembalikan subset ciri." ([sumber](https://wikipedia.org/wiki/Feature_selection))

### Visualkan data anda

Aspek penting dalam alat saintis data adalah kuasa untuk memvisualkan data menggunakan beberapa pustaka yang sangat baik seperti Seaborn atau MatPlotLib. Mewakili data anda secara visual mungkin membolehkan anda menemui korelasi tersembunyi yang boleh anda manfaatkan. Visualisasi anda juga mungkin membantu anda menemui bias atau data yang tidak seimbang (seperti yang kita temui dalam [Klasifikasi](../../4-Classification/2-Classifiers-1/README.md)).

### Bahagikan set data anda

Sebelum latihan, anda perlu membahagikan set data anda kepada dua atau lebih bahagian yang tidak sama saiz tetapi masih mewakili data dengan baik.

- **Latihan**. Bahagian set data ini digunakan untuk melatih model anda. Set ini membentuk sebahagian besar daripada set data asal.
- **Ujian**. Set ujian adalah kumpulan data bebas, sering kali diambil daripada data asal, yang anda gunakan untuk mengesahkan prestasi model yang dibina.
- **Pengesahan**. Set pengesahan adalah kumpulan contoh bebas yang lebih kecil yang anda gunakan untuk menala parameter hiper model, atau seni bina, untuk meningkatkan model. Bergantung pada saiz data anda dan soalan yang anda tanyakan, anda mungkin tidak perlu membina set ketiga ini (seperti yang kita perhatikan dalam [Ramalan Siri Masa](../../7-TimeSeries/1-Introduction/README.md)).

## Membina model

Menggunakan data latihan anda, matlamat anda adalah untuk membina model, atau representasi statistik data anda, menggunakan pelbagai algoritma untuk **melatih**nya. Melatih model mendedahkannya kepada data dan membolehkan ia membuat andaian tentang pola yang ditemui, disahkan, dan diterima atau ditolak.

### Tentukan kaedah latihan

Bergantung pada soalan anda dan sifat data anda, anda akan memilih kaedah untuk melatihnya. Melalui [dokumentasi Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - yang kita gunakan dalam kursus ini - anda boleh meneroka banyak cara untuk melatih model. Bergantung pada pengalaman anda, anda mungkin perlu mencuba beberapa kaedah yang berbeza untuk membina model terbaik. Anda mungkin akan melalui proses di mana saintis data menilai prestasi model dengan memberinya data yang belum dilihat, memeriksa ketepatan, bias, dan isu lain yang merosakkan kualiti, serta memilih kaedah latihan yang paling sesuai untuk tugas yang diberikan.

### Latih model

Dengan data latihan anda, anda bersedia untuk 'memasangkannya' untuk mencipta model. Anda akan perasan bahawa dalam banyak pustaka ML, anda akan menemui kod 'model.fit' - pada masa ini anda menghantar pembolehubah ciri anda sebagai array nilai (biasanya 'X') dan pembolehubah sasaran (biasanya 'y').

### Nilai model

Setelah proses latihan selesai (ia boleh mengambil banyak iterasi, atau 'epoch', untuk melatih model besar), anda akan dapat menilai kualiti model dengan menggunakan data ujian untuk mengukur prestasinya. Data ini adalah subset daripada data asal yang belum dianalisis oleh model. Anda boleh mencetak jadual metrik tentang kualiti model anda.

🎓 **Pemasangan model**

Dalam konteks pembelajaran mesin, pemasangan model merujuk kepada ketepatan fungsi asas model semasa ia cuba menganalisis data yang tidak dikenali.

🎓 **Underfitting** dan **overfitting** adalah masalah biasa yang merosakkan kualiti model, kerana model sama ada tidak cukup baik atau terlalu baik. Ini menyebabkan model membuat ramalan yang terlalu selaras atau terlalu longgar dengan data latihannya. Model yang terlalu sesuai meramalkan data latihan terlalu baik kerana ia telah mempelajari butiran dan bunyi data terlalu baik. Model yang kurang sesuai tidak tepat kerana ia tidak dapat menganalisis data latihannya atau data yang belum 'dilihat' dengan tepat.

![model overfitting](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

## Penalaan parameter

Setelah latihan awal anda selesai, perhatikan kualiti model dan pertimbangkan untuk meningkatkannya dengan menyesuaikan 'parameter hiper'nya. Baca lebih lanjut tentang proses ini [dalam dokumentasi](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Ramalan

Ini adalah saat di mana anda boleh menggunakan data yang benar-benar baru untuk menguji ketepatan model anda. Dalam tetapan ML 'terapan', di mana anda membina aset web untuk menggunakan model dalam pengeluaran, proses ini mungkin melibatkan pengumpulan input pengguna (tekanan butang, sebagai contoh) untuk menetapkan pembolehubah dan menghantarnya kepada model untuk inferens, atau penilaian.

Dalam pelajaran ini, anda akan menemui cara menggunakan langkah-langkah ini untuk menyediakan, membina, menguji, menilai, dan meramalkan - semua gerakan seorang saintis data dan banyak lagi, semasa anda maju dalam perjalanan anda untuk menjadi jurutera ML 'full stack'.

---

## 🚀Cabaran

Lukiskan carta alir yang mencerminkan langkah-langkah seorang pengamal ML. Di mana anda melihat diri anda sekarang dalam proses ini? Di mana anda meramalkan anda akan menghadapi kesukaran? Apa yang kelihatan mudah bagi anda?

## [Kuiz selepas pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Cari dalam talian untuk temu bual dengan saintis data yang membincangkan kerja harian mereka. Berikut adalah [satu](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tugasan

[Temu bual seorang saintis data](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.