<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T19:39:56+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada pembelajaran mesin

## [Kuiz sebelum kuliah](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML untuk pemula - Pengenalan kepada Pembelajaran Mesin untuk Pemula](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML untuk pemula - Pengenalan kepada Pembelajaran Mesin untuk Pemula")

> 🎥 Klik imej di atas untuk video pendek yang menerangkan pelajaran ini.

Selamat datang ke kursus ini tentang pembelajaran mesin klasik untuk pemula! Sama ada anda benar-benar baru dalam topik ini, atau seorang pengamal ML berpengalaman yang ingin menyegarkan pengetahuan dalam bidang tertentu, kami gembira anda menyertai kami! Kami ingin mencipta tempat permulaan yang mesra untuk kajian ML anda dan akan gembira untuk menilai, memberi maklum balas, dan menggabungkan [maklum balas](https://github.com/microsoft/ML-For-Beginners/discussions) anda.

[![Pengenalan kepada ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Pengenalan kepada ML")

> 🎥 Klik imej di atas untuk video: John Guttag dari MIT memperkenalkan pembelajaran mesin

---
## Memulakan pembelajaran mesin

Sebelum memulakan kurikulum ini, anda perlu menyediakan komputer anda dan bersedia untuk menjalankan notebook secara tempatan.

- **Konfigurasikan mesin anda dengan video ini**. Gunakan pautan berikut untuk belajar [cara memasang Python](https://youtu.be/CXZYvNRIAKM) dalam sistem anda dan [menyediakan editor teks](https://youtu.be/EU8eayHWoZg) untuk pembangunan.
- **Belajar Python**. Adalah disyorkan untuk mempunyai pemahaman asas tentang [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), bahasa pengaturcaraan yang berguna untuk saintis data yang kami gunakan dalam kursus ini.
- **Belajar Node.js dan JavaScript**. Kami juga menggunakan JavaScript beberapa kali dalam kursus ini semasa membina aplikasi web, jadi anda perlu mempunyai [node](https://nodejs.org) dan [npm](https://www.npmjs.com/) dipasang, serta [Visual Studio Code](https://code.visualstudio.com/) tersedia untuk pembangunan Python dan JavaScript.
- **Buat akaun GitHub**. Oleh kerana anda menemui kami di [GitHub](https://github.com), anda mungkin sudah mempunyai akaun, tetapi jika tidak, buat satu dan kemudian fork kurikulum ini untuk digunakan sendiri. (Jangan lupa beri kami bintang juga 😊)
- **Terokai Scikit-learn**. Biasakan diri dengan [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), satu set perpustakaan ML yang kami rujuk dalam pelajaran ini.

---
## Apa itu pembelajaran mesin?

Istilah 'pembelajaran mesin' adalah salah satu istilah yang paling popular dan sering digunakan hari ini. Terdapat kemungkinan besar anda pernah mendengar istilah ini sekurang-kurangnya sekali jika anda mempunyai sedikit pengetahuan tentang teknologi, tidak kira bidang kerja anda. Walau bagaimanapun, mekanik pembelajaran mesin adalah misteri bagi kebanyakan orang. Bagi pemula pembelajaran mesin, subjek ini kadangkala boleh terasa mengelirukan. Oleh itu, adalah penting untuk memahami apa sebenarnya pembelajaran mesin, dan mempelajarinya langkah demi langkah, melalui contoh praktikal.

---
## Kurva hype

![kurva hype ml](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends menunjukkan 'kurva hype' terkini istilah 'pembelajaran mesin'

---
## Alam semesta yang misteri

Kita hidup dalam alam semesta yang penuh dengan misteri yang menarik. Saintis hebat seperti Stephen Hawking, Albert Einstein, dan ramai lagi telah mendedikasikan hidup mereka untuk mencari maklumat bermakna yang membongkar misteri dunia di sekeliling kita. Ini adalah keadaan manusia untuk belajar: seorang kanak-kanak manusia belajar perkara baru dan membongkar struktur dunia mereka tahun demi tahun semasa mereka membesar menjadi dewasa.

---
## Otak kanak-kanak

Otak dan deria seorang kanak-kanak memerhatikan fakta-fakta di sekeliling mereka dan secara beransur-ansur mempelajari corak tersembunyi kehidupan yang membantu kanak-kanak itu mencipta peraturan logik untuk mengenal pasti corak yang dipelajari. Proses pembelajaran otak manusia menjadikan manusia makhluk hidup yang paling canggih di dunia ini. Belajar secara berterusan dengan menemui corak tersembunyi dan kemudian berinovasi berdasarkan corak tersebut membolehkan kita menjadi lebih baik sepanjang hayat kita. Keupayaan pembelajaran dan kemampuan berkembang ini berkaitan dengan konsep yang dipanggil [keplastikan otak](https://www.simplypsychology.org/brain-plasticity.html). Secara luaran, kita boleh menarik beberapa persamaan motivasi antara proses pembelajaran otak manusia dan konsep pembelajaran mesin.

---
## Otak manusia

[Otak manusia](https://www.livescience.com/29365-human-brain.html) memerhatikan perkara dari dunia nyata, memproses maklumat yang diperhatikan, membuat keputusan rasional, dan melakukan tindakan tertentu berdasarkan keadaan. Inilah yang kita panggil berkelakuan secara pintar. Apabila kita memprogramkan tiruan proses tingkah laku pintar kepada mesin, ia dipanggil kecerdasan buatan (AI).

---
## Beberapa istilah

Walaupun istilah-istilah ini boleh mengelirukan, pembelajaran mesin (ML) adalah subset penting dalam kecerdasan buatan. **ML berkaitan dengan penggunaan algoritma khusus untuk mencari maklumat bermakna dan menemui corak tersembunyi daripada data yang diperhatikan untuk menyokong proses membuat keputusan secara rasional**.

---
## AI, ML, Pembelajaran Mendalam

![AI, ML, pembelajaran mendalam, sains data](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram menunjukkan hubungan antara AI, ML, pembelajaran mendalam, dan sains data. Infografik oleh [Jen Looper](https://twitter.com/jenlooper) yang diilhamkan oleh [grafik ini](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Konsep yang akan dibincangkan

Dalam kurikulum ini, kita akan membincangkan hanya konsep asas pembelajaran mesin yang mesti diketahui oleh pemula. Kami membincangkan apa yang kami panggil 'pembelajaran mesin klasik' terutamanya menggunakan Scikit-learn, perpustakaan yang sangat baik yang digunakan oleh ramai pelajar untuk mempelajari asas-asas. Untuk memahami konsep yang lebih luas tentang kecerdasan buatan atau pembelajaran mendalam, pengetahuan asas yang kukuh tentang pembelajaran mesin adalah sangat penting, dan kami ingin menawarkannya di sini.

---
## Dalam kursus ini anda akan belajar:

- konsep asas pembelajaran mesin
- sejarah ML
- ML dan keadilan
- teknik regresi ML
- teknik klasifikasi ML
- teknik pengelompokan ML
- teknik pemprosesan bahasa semula jadi ML
- teknik ramalan siri masa ML
- pembelajaran pengukuhan
- aplikasi dunia nyata untuk ML

---
## Apa yang tidak akan dibincangkan

- pembelajaran mendalam
- rangkaian neural
- AI

Untuk pengalaman pembelajaran yang lebih baik, kami akan mengelakkan kerumitan rangkaian neural, 'pembelajaran mendalam' - pembinaan model berlapis-lapis menggunakan rangkaian neural - dan AI, yang akan kami bincangkan dalam kurikulum yang berbeza. Kami juga akan menawarkan kurikulum sains data yang akan datang untuk memberi fokus kepada aspek tersebut dalam bidang yang lebih besar ini.

---
## Mengapa belajar pembelajaran mesin?

Pembelajaran mesin, dari perspektif sistem, didefinisikan sebagai penciptaan sistem automatik yang dapat mempelajari corak tersembunyi dari data untuk membantu membuat keputusan pintar.

Motivasi ini secara longgar diilhamkan oleh bagaimana otak manusia mempelajari perkara tertentu berdasarkan data yang diperhatikan dari dunia luar.

✅ Fikirkan sejenak mengapa sesebuah perniagaan ingin menggunakan strategi pembelajaran mesin berbanding mencipta enjin berasaskan peraturan yang dikodkan secara keras.

---
## Aplikasi pembelajaran mesin

Aplikasi pembelajaran mesin kini hampir di mana-mana, dan sama meluasnya dengan data yang mengalir di sekitar masyarakat kita, yang dihasilkan oleh telefon pintar, peranti yang disambungkan, dan sistem lain. Memandangkan potensi besar algoritma pembelajaran mesin terkini, penyelidik telah meneroka keupayaannya untuk menyelesaikan masalah kehidupan nyata yang pelbagai dimensi dan pelbagai disiplin dengan hasil yang sangat positif.

---
## Contoh ML yang diterapkan

**Anda boleh menggunakan pembelajaran mesin dalam pelbagai cara**:

- Untuk meramalkan kemungkinan penyakit daripada sejarah perubatan atau laporan pesakit.
- Untuk memanfaatkan data cuaca bagi meramalkan kejadian cuaca.
- Untuk memahami sentimen dalam teks.
- Untuk mengesan berita palsu bagi menghentikan penyebaran propaganda.

Kewangan, ekonomi, sains bumi, penerokaan angkasa, kejuruteraan bioperubatan, sains kognitif, dan bahkan bidang dalam kemanusiaan telah menyesuaikan pembelajaran mesin untuk menyelesaikan masalah berat pemprosesan data dalam domain mereka.

---
## Kesimpulan

Pembelajaran mesin mengautomasi proses penemuan corak dengan mencari wawasan bermakna daripada data dunia nyata atau data yang dihasilkan. Ia telah membuktikan dirinya sangat berharga dalam aplikasi perniagaan, kesihatan, dan kewangan, antara lain.

Dalam masa terdekat, memahami asas pembelajaran mesin akan menjadi keperluan bagi orang dari mana-mana bidang kerana penerimaannya yang meluas.

---
# 🚀 Cabaran

Lukiskan, di atas kertas atau menggunakan aplikasi dalam talian seperti [Excalidraw](https://excalidraw.com/), pemahaman anda tentang perbezaan antara AI, ML, pembelajaran mendalam, dan sains data. Tambahkan beberapa idea tentang masalah yang sesuai untuk diselesaikan oleh setiap teknik ini.

# [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

---
# Ulasan & Kajian Kendiri

Untuk mengetahui lebih lanjut tentang cara anda boleh bekerja dengan algoritma ML di awan, ikuti [Laluan Pembelajaran](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ikuti [Laluan Pembelajaran](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) tentang asas-asas ML.

---
# Tugasan

[Mulakan](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.