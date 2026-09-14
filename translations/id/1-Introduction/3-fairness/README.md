# Membangun solusi Machine Learning dengan AI yang bertanggung jawab
 
![Ringkasan AI yang bertanggung jawab dalam Machine Learning dalam bentuk sketchnote](../../../../translated_images/id/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuis pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)
 
## Pendahuluan

Dalam kurikulum ini, Anda akan mulai menemukan bagaimana machine learning dapat dan sedang mempengaruhi kehidupan sehari-hari kita. Bahkan sekarang, sistem dan model terlibat dalam tugas pengambilan keputusan harian, seperti diagnosis kesehatan, persetujuan pinjaman, atau pendeteksian penipuan. Jadi, penting bahwa model-model ini bekerja dengan baik untuk memberikan hasil yang dapat dipercaya. Sama seperti aplikasi perangkat lunak apa pun, sistem AI akan terkadang gagal memenuhi harapan atau menghasilkan hasil yang tidak diinginkan. Itulah sebabnya penting untuk bisa memahami dan menjelaskan perilaku model AI.

Bayangkan apa yang bisa terjadi ketika data yang Anda gunakan untuk membangun model ini kurang mewakili demografi tertentu, seperti ras, gender, pandangan politik, agama, atau tidak proporsional mewakili demografi tersebut. Bagaimana jika keluaran model diinterpretasikan untuk memihak beberapa demografi? Apa konsekuensinya untuk aplikasi tersebut? Selain itu, apa yang terjadi ketika model menghasilkan hasil yang merugikan dan membahayakan orang? Siapa yang bertanggung jawab atas perilaku sistem AI? Ini adalah beberapa pertanyaan yang akan kita jelajahi dalam kurikulum ini.

Dalam pelajaran ini, Anda akan:

- Meningkatkan kesadaran Anda tentang pentingnya keadilan dalam machine learning dan kerugian terkait ketidakadilan.
- Mengenal praktik menjelajahi outlier dan skenario tidak biasa untuk memastikan keandalan dan keselamatan
- Memahami kebutuhan untuk memberdayakan semua orang dengan merancang sistem inklusif
- Menjelajahi betapa pentingnya melindungi privasi dan keamanan data serta orang
- Melihat pentingnya pendekatan kotak kaca untuk menjelaskan perilaku model AI
- Memperhatikan bagaimana akuntabilitas sangat penting untuk membangun kepercayaan dalam sistem AI

## Prasyarat

Sebagai prasyarat, silakan ambil Jalur Pembelajaran "Prinsip AI yang Bertanggung Jawab" dan tonton video di bawah ini tentang topik tersebut:

Pelajari lebih lanjut tentang AI yang Bertanggung Jawab dengan mengikuti [Jalur Pembelajaran](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Pendekatan Microsoft terhadap AI yang Bertanggung Jawab](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Pendekatan Microsoft terhadap AI yang Bertanggung Jawab")

> 🎥 Klik gambar di atas untuk video: Pendekatan Microsoft terhadap AI yang Bertanggung Jawab

## Keadilan

Sistem AI harus memperlakukan semua orang secara adil dan menghindari memengaruhi kelompok orang yang serupa dengan cara yang berbeda. Misalnya, ketika sistem AI memberikan panduan tentang perawatan medis, aplikasi pinjaman, atau pekerjaan, mereka harus memberikan rekomendasi yang sama kepada semua orang dengan gejala, kondisi keuangan, atau kualifikasi profesional yang serupa. Setiap dari kita sebagai manusia membawa bias yang diwariskan yang memengaruhi keputusan dan tindakan kita. Bias ini dapat terlihat dalam data yang kita gunakan untuk melatih sistem AI. Manipulasi seperti ini terkadang terjadi tanpa sengaja. Seringkali sulit untuk secara sadar mengetahui kapan Anda memperkenalkan bias dalam data.

**"Ketidakadilan"** mencakup dampak negatif, atau "kerugian", bagi sekelompok orang, seperti yang didefinisikan berdasarkan ras, gender, usia, atau status disabilitas. Kerugian utama terkait keadilan dapat diklasifikasikan sebagai:

- **Alokasi**, misalnya jika gender atau etnis favorit dibanding yang lain.
- **Kualitas layanan**. Jika Anda melatih data untuk satu skenario tertentu tetapi kenyataannya jauh lebih kompleks, ini menghasilkan layanan dengan performa rendah. Sebagai contoh, dispenser sabun tangan yang sepertinya tidak bisa mendeteksi orang dengan kulit gelap. [Referensi](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Penghinaan**. Mengkritik dan melabeli sesuatu atau seseorang secara tidak adil. Misalnya, teknologi pelabelan gambar yang terkenal salah melabeli gambar orang dengan kulit gelap sebagai gorila.
- **Representasi berlebihan atau kurang**. Ide dasarnya adalah suatu kelompok tertentu tidak terlihat dalam profesi tertentu, dan layanan atau fungsi apa pun yang terus mempromosikan hal itu berkontribusi pada kerugian.
- **Stereotip**. Mengaitkan suatu kelompok dengan atribut yang telah ditentukan. Contohnya, sistem terjemahan bahasa antara Inggris dan Turki mungkin memiliki ketidakakuratan karena kata dengan asosiasi stereotip terkait gender.

![terjemahan ke bahasa Turki](../../../../translated_images/id/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> terjemahan ke bahasa Turki

![terjemahan kembali ke bahasa Inggris](../../../../translated_images/id/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> terjemahan kembali ke bahasa Inggris

Saat merancang dan menguji sistem AI, kita perlu memastikan bahwa AI adil dan tidak diprogram untuk membuat keputusan yang bias atau diskriminatif, yang juga dilarang untuk dilakukan oleh manusia. Menjamin keadilan dalam AI dan machine learning tetap menjadi tantangan sosioteknis yang kompleks.

### Keandalan dan keselamatan

Untuk membangun kepercayaan, sistem AI harus dapat diandalkan, aman, dan konsisten dalam kondisi normal dan tak terduga. Penting untuk mengetahui bagaimana sistem AI akan berperilaku dalam berbagai situasi, terutama saat berada pada kondisi outlier. Saat membangun solusi AI, perlu ada fokus besar pada cara menangani berbagai keadaan yang mungkin ditemui solusi AI. Misalnya, mobil swakemudi harus memprioritaskan keselamatan orang. Akibatnya, AI yang menggerakkan mobil tersebut harus mempertimbangkan semua skenario yang mungkin ditemui mobil, seperti malam hari, badai petir atau badai salju, anak-anak yang berlari melintasi jalan, hewan peliharaan, konstruksi jalan, dll. Seberapa baik sistem AI dapat menangani berbagai kondisi secara andal dan aman mencerminkan tingkat antisipasi yang dipertimbangkan oleh ilmuwan data atau pengembang AI selama perancangan atau pengujian sistem.

> [🎥 Klik di sini untuk video: Keandalan dan keselamatan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusi

Sistem AI harus dirancang untuk melibatkan dan memberdayakan semua orang. Saat merancang dan mengimplementasikan sistem AI, ilmuwan data dan pengembang AI mengidentifikasi dan mengatasi hambatan potensial dalam sistem yang bisa secara tidak sengaja mengecualikan orang. Misalnya, ada 1 miliar orang dengan disabilitas di seluruh dunia. Dengan kemajuan AI, mereka dapat mengakses berbagai informasi dan peluang dengan lebih mudah dalam kehidupan sehari-hari. Dengan mengatasi hambatan ini, tercipta peluang untuk berinovasi dan mengembangkan produk AI dengan pengalaman yang lebih baik yang menguntungkan semua orang.

> [🎥 Klik di sini untuk video: Inklusi dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Keamanan dan privasi

Sistem AI harus aman dan menghormati privasi orang. Orang cenderung kurang percaya pada sistem yang mempertaruhkan privasi, informasi, atau nyawa mereka. Saat melatih model machine learning, kita mengandalkan data untuk menghasilkan hasil terbaik. Dalam proses itu, asal usul data dan integritasnya harus dipertimbangkan. Misalnya, apakah data tersebut dikirim oleh pengguna atau tersedia secara publik? Selanjutnya, saat bekerja dengan data, penting untuk mengembangkan sistem AI yang dapat melindungi informasi rahasia dan tahan terhadap serangan. Seiring AI semakin meluas, melindungi privasi dan mengamankan informasi penting pribadi dan bisnis menjadi semakin penting dan kompleks. Masalah privasi dan keamanan data perlu mendapat perhatian khusus untuk AI karena akses ke data sangat penting bagi sistem AI untuk membuat prediksi dan keputusan yang akurat dan berdasar tentang orang.

> [🎥 Klik di sini untuk video: Keamanan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sebagai industri, kita telah membuat kemajuan signifikan dalam Privasi & keamanan, yang didorong secara signifikan oleh regulasi seperti GDPR (General Data Protection Regulation).
- Namun dengan sistem AI kita harus mengakui adanya ketegangan antara kebutuhan akan data pribadi lebih banyak untuk membuat sistem lebih personal dan efektif – serta privasi.
- Sama seperti pada masa lahirnya komputer terhubung dengan internet, kita juga melihat peningkatan besar dalam jumlah masalah keamanan terkait AI.
- Pada saat yang sama, kita telah melihat AI digunakan untuk meningkatkan keamanan. Sebagai contoh, sebagian besar pemindai anti-virus modern hari ini didukung oleh heuristik AI.
- Kita harus memastikan bahwa proses Data Science kita berpadu harmonis dengan praktik privasi dan keamanan terbaru.


### Transparansi
Sistem AI harus dapat dimengerti. Bagian penting dari transparansi adalah menjelaskan perilaku sistem AI dan komponennya. Meningkatkan pemahaman tentang sistem AI mengharuskan para pemangku kepentingan memahami bagaimana dan mengapa sistem tersebut berfungsi agar mereka dapat mengidentifikasi potensi masalah performa, kekhawatiran keselamatan dan privasi, bias, praktik eksklusi, atau hasil yang tidak diinginkan. Kami juga meyakini bahwa mereka yang menggunakan sistem AI harus jujur dan terbuka tentang kapan, mengapa, dan bagaimana mereka memilih untuk menggunakannya. Serta keterbatasan sistem yang mereka gunakan. Misalnya, jika sebuah bank menggunakan sistem AI untuk mendukung keputusan pinjaman konsumennya, penting untuk memeriksa hasilnya dan memahami data mana yang memengaruhi rekomendasi sistem tersebut. Pemerintah mulai mengatur AI di berbagai industri, jadi ilmuwan data dan organisasi harus menjelaskan jika sistem AI memenuhi persyaratan regulasi, terutama ketika ada hasil yang tidak diinginkan.

> [🎥 Klik di sini untuk video: Transparansi dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Karena sistem AI sangat kompleks, sulit untuk memahami bagaimana mereka bekerja dan menginterpretasikan hasilnya.
- Kurangnya pemahaman ini memengaruhi cara sistem ini dikelola, dioperasionalkan, dan didokumentasikan.
- Kurangnya pemahaman ini, yang lebih penting, memengaruhi keputusan yang dibuat menggunakan hasil yang dihasilkan oleh sistem ini.

### Akuntabilitas
 
Orang-orang yang merancang dan menerapkan sistem AI harus bertanggung jawab atas cara sistem mereka beroperasi. Kebutuhan akan akuntabilitas sangat penting terutama pada teknologi yang sensitif seperti pengenalan wajah. Baru-baru ini, ada permintaan yang meningkat untuk teknologi pengenalan wajah, terutama dari organisasi penegak hukum yang melihat potensi teknologi dalam penggunaan seperti menemukan anak hilang. Namun, teknologi ini berpotensi digunakan oleh pemerintah untuk membahayakan kebebasan dasar warga negara mereka dengan, misalnya, memungkinkan pengawasan terus-menerus terhadap individu tertentu. Oleh karena itu, ilmuwan data dan organisasi harus bertanggung jawab atas bagaimana sistem AI mereka berdampak pada individu atau masyarakat.

[![Peneliti AI Terdepan Memberi Peringatan tentang Pengawasan Massal Melalui Pengenalan Wajah](../../../../translated_images/id/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Pendekatan Microsoft terhadap AI yang Bertanggung Jawab")

> 🎥 Klik gambar di atas untuk video: Peringatan tentang Pengawasan Massal Melalui Pengenalan Wajah

Pada akhirnya, salah satu pertanyaan terbesar bagi generasi kita, sebagai generasi pertama yang membawa AI ke masyarakat, adalah bagaimana memastikan bahwa komputer tetap bertanggung jawab kepada manusia dan bagaimana memastikan bahwa orang yang merancang komputer tetap bertanggung jawab kepada semua orang.

## Penilaian dampak

Sebelum melatih model machine learning, penting untuk melakukan penilaian dampak untuk memahami tujuan sistem AI; apa penggunaan yang dimaksudkan; dimana sistem akan diterapkan; dan siapa yang akan berinteraksi dengan sistem tersebut. Ini berguna bagi peninjau atau penguji yang mengevaluasi sistem untuk mengetahui faktor apa yang harus dipertimbangkan saat mengidentifikasi potensi risiko dan konsekuensi yang diharapkan.

Berikut adalah bidang fokus saat melakukan penilaian dampak:

* **Dampak negatif pada individu**. Menyadari setiap pembatasan atau persyaratan, penggunaan yang tidak didukung atau setiap keterbatasan yang diketahui yang menghambat performa sistem sangat penting untuk memastikan sistem tidak digunakan dengan cara yang dapat membahayakan individu.
* **Persyaratan data**. Memahami bagaimana dan di mana sistem akan menggunakan data memungkinkan peninjau mengeksplorasi persyaratan data yang harus diperhatikan (misalnya, regulasi data GDPR atau HIPAA). Selain itu, memeriksa apakah sumber atau jumlah data cukup untuk pelatihan.
* **Ringkasan dampak**. Kumpulkan daftar potensi kerugian yang mungkin timbul dari penggunaan sistem. Sepanjang siklus hidup ML, tinjau apakah masalah yang diidentifikasi telah diminimalkan atau ditangani.
* **Tujuan yang berlaku** untuk masing-masing dari enam prinsip inti. Evaluasi apakah tujuan dari masing-masing prinsip tercapai dan apakah ada celah.


## Debugging dengan AI yang bertanggung jawab

Serupa dengan debugging aplikasi perangkat lunak, debugging sistem AI adalah proses penting untuk mengidentifikasi dan menyelesaikan masalah dalam sistem. Ada banyak faktor yang dapat mempengaruhi model tidak berperforma sesuai harapan atau secara bertanggung jawab. Sebagian besar metrik performa model tradisional adalah agregat kuantitatif dari performa model, yang tidak cukup untuk menganalisis bagaimana model melanggar prinsip AI yang bertanggung jawab. Selain itu, model machine learning adalah kotak hitam yang menyulitkan untuk memahami apa yang mendorong hasilnya atau memberi penjelasan saat terjadi kesalahan. Nanti dalam kursus ini, kita akan belajar bagaimana menggunakan dashboard Responsible AI untuk membantu debug sistem AI. Dashboard ini menyediakan alat holistik bagi ilmuwan data dan pengembang AI untuk melakukan:

* **Analisis kesalahan**. Untuk mengidentifikasi distribusi kesalahan model yang dapat memengaruhi keadilan atau keandalan sistem.
* **Tinjauan model**. Untuk menemukan di mana terdapat disparitas performa model di antara kohort data.
* **Analisis data**. Untuk memahami distribusi data dan mengidentifikasi potensi bias dalam data yang dapat menyebabkan masalah keadilan, inklusivitas, dan keandalan.
* **Interpretabilitas model**. Untuk memahami apa yang mempengaruhi atau memengaruhi prediksi model. Ini membantu dalam menjelaskan perilaku model, yang penting untuk transparansi dan akuntabilitas.


## 🚀 Tantangan
 
Untuk mencegah kerugian yang timbul sejak awal, kita harus:

- memiliki keberagaman latar belakang dan perspektif di antara orang-orang yang mengerjakan sistem
- berinvestasi dalam kumpulan data yang mencerminkan keberagaman masyarakat kita
- mengembangkan metode yang lebih baik sepanjang siklus hidup machine learning untuk mendeteksi dan memperbaiki AI yang tidak bertanggung jawab ketika terjadi

Pikirkan tentang skenario kehidupan nyata di mana ketidakpercayaan model terlihat dalam pembangunan dan penggunaan model. Apa lagi yang harus kita pertimbangkan?

## [Kuis pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Review & Studi Mandiri
 
Dalam pelajaran ini, Anda telah mempelajari beberapa dasar konsep keadilan dan ketidakadilan dalam machine learning.
 
Tonton lokakarya ini untuk menggali lebih dalam topik-topik tersebut:

- Dalam upaya AI yang bertanggung jawab: Menerapkan prinsip ke praktik oleh Besmira Nushi, Mehrnoosh Sameki dan Amit Sharma

[![Kotak Peralatan AI Bertanggung Jawab: kerangka kerja sumber terbuka untuk membangun AI yang bertanggung jawab](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Kotak Peralatan RAI: kerangka kerja sumber terbuka untuk membangun AI yang bertanggung jawab")

> 🎥 Klik gambar di atas untuk video: Kotak Peralatan RAI: kerangka kerja sumber terbuka untuk membangun AI yang bertanggung jawab oleh Besmira Nushi, Mehrnoosh Sameki, dan Amit Sharma

Juga, baca: 

- Pusat sumber daya RAI Microsoft: [Sumber Daya AI Bertanggung Jawab – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Kelompok riset FATE Microsoft: [FATE: Keberimbangan, Akuntabilitas, Transparansi, dan Etika dalam AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Kotak Peralatan RAI: 

- [Repositori GitHub Kotak Peralatan AI Bertanggung Jawab](https://github.com/microsoft/responsible-ai-toolbox)

Baca tentang alat Azure Machine Learning untuk memastikan keadilan:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Tugas

[Jelajahi Kotak Peralatan RAI](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->