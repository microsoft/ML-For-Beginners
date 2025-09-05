<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T19:32:45+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "id"
}
-->
# Membangun Solusi Machine Learning dengan AI yang Bertanggung Jawab

![Ringkasan AI yang bertanggung jawab dalam Machine Learning dalam bentuk sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Pendahuluan

Dalam kurikulum ini, Anda akan mulai memahami bagaimana machine learning dapat dan sedang memengaruhi kehidupan kita sehari-hari. Bahkan saat ini, sistem dan model terlibat dalam tugas pengambilan keputusan harian, seperti diagnosis kesehatan, persetujuan pinjaman, atau mendeteksi penipuan. Oleh karena itu, penting bahwa model-model ini bekerja dengan baik untuk memberikan hasil yang dapat dipercaya. Sama seperti aplikasi perangkat lunak lainnya, sistem AI dapat meleset dari ekspektasi atau menghasilkan hasil yang tidak diinginkan. Itulah mengapa penting untuk memahami dan menjelaskan perilaku model AI.

Bayangkan apa yang bisa terjadi ketika data yang Anda gunakan untuk membangun model ini tidak mencakup demografi tertentu, seperti ras, gender, pandangan politik, agama, atau secara tidak proporsional mewakili demografi tersebut. Bagaimana jika output model diinterpretasikan untuk menguntungkan beberapa demografi? Apa konsekuensinya bagi aplikasi tersebut? Selain itu, apa yang terjadi ketika model menghasilkan hasil yang merugikan dan membahayakan orang? Siapa yang bertanggung jawab atas perilaku sistem AI? Ini adalah beberapa pertanyaan yang akan kita eksplorasi dalam kurikulum ini.

Dalam pelajaran ini, Anda akan:

- Meningkatkan kesadaran tentang pentingnya keadilan dalam machine learning dan dampak terkait keadilan.
- Mengenal praktik eksplorasi outlier dan skenario tidak biasa untuk memastikan keandalan dan keamanan.
- Memahami kebutuhan untuk memberdayakan semua orang dengan merancang sistem yang inklusif.
- Mengeksplorasi betapa pentingnya melindungi privasi dan keamanan data serta individu.
- Melihat pentingnya pendekatan transparan untuk menjelaskan perilaku model AI.
- Menyadari bahwa akuntabilitas sangat penting untuk membangun kepercayaan pada sistem AI.

## Prasyarat

Sebagai prasyarat, silakan ikuti "Responsible AI Principles" Learn Path dan tonton video di bawah ini tentang topik tersebut:

Pelajari lebih lanjut tentang AI yang Bertanggung Jawab dengan mengikuti [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Pendekatan Microsoft terhadap AI yang Bertanggung Jawab](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Pendekatan Microsoft terhadap AI yang Bertanggung Jawab")

> 🎥 Klik gambar di atas untuk video: Pendekatan Microsoft terhadap AI yang Bertanggung Jawab

## Keadilan

Sistem AI harus memperlakukan semua orang secara adil dan menghindari memengaruhi kelompok orang yang serupa dengan cara yang berbeda. Misalnya, ketika sistem AI memberikan panduan tentang pengobatan medis, aplikasi pinjaman, atau pekerjaan, mereka harus memberikan rekomendasi yang sama kepada semua orang dengan gejala, keadaan finansial, atau kualifikasi profesional yang serupa. Setiap dari kita sebagai manusia membawa bias yang diwarisi yang memengaruhi keputusan dan tindakan kita. Bias ini dapat terlihat dalam data yang kita gunakan untuk melatih sistem AI. Manipulasi semacam itu terkadang terjadi secara tidak sengaja. Sering kali sulit untuk secara sadar mengetahui kapan Anda memperkenalkan bias dalam data.

**“Ketidakadilan”** mencakup dampak negatif, atau “kerugian”, bagi sekelompok orang, seperti yang didefinisikan dalam hal ras, gender, usia, atau status disabilitas. Kerugian utama terkait keadilan dapat diklasifikasikan sebagai:

- **Distribusi**, jika gender atau etnis tertentu, misalnya, lebih diuntungkan dibandingkan yang lain.
- **Kualitas layanan**. Jika Anda melatih data untuk satu skenario spesifik tetapi kenyataannya jauh lebih kompleks, ini menghasilkan layanan yang berkinerja buruk. Misalnya, dispenser sabun tangan yang tidak dapat mendeteksi orang dengan kulit gelap. [Referensi](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Pelecehan**. Mengkritik atau melabeli sesuatu atau seseorang secara tidak adil. Misalnya, teknologi pelabelan gambar yang secara keliru melabeli gambar orang berkulit gelap sebagai gorila.
- **Representasi berlebihan atau kurang**. Ide bahwa kelompok tertentu tidak terlihat dalam profesi tertentu, dan layanan atau fungsi apa pun yang terus mempromosikan hal itu berkontribusi pada kerugian.
- **Stereotip**. Mengaitkan atribut yang telah ditentukan sebelumnya dengan kelompok tertentu. Misalnya, sistem penerjemahan bahasa antara Inggris dan Turki mungkin memiliki ketidakakuratan karena kata-kata dengan asosiasi stereotip terhadap gender.

![terjemahan ke Turki](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> terjemahan ke Turki

![terjemahan kembali ke Inggris](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> terjemahan kembali ke Inggris

Saat merancang dan menguji sistem AI, kita perlu memastikan bahwa AI adil dan tidak diprogram untuk membuat keputusan yang bias atau diskriminatif, yang juga dilarang dilakukan oleh manusia. Menjamin keadilan dalam AI dan machine learning tetap menjadi tantangan sosial-teknis yang kompleks.

### Keandalan dan keamanan

Untuk membangun kepercayaan, sistem AI perlu dapat diandalkan, aman, dan konsisten dalam kondisi normal maupun tak terduga. Penting untuk mengetahui bagaimana sistem AI akan berperilaku dalam berbagai situasi, terutama ketika mereka berada di luar batas normal. Saat membangun solusi AI, perlu ada fokus yang substansial pada bagaimana menangani berbagai keadaan yang mungkin dihadapi oleh solusi AI tersebut. Misalnya, mobil tanpa pengemudi harus mengutamakan keselamatan orang. Akibatnya, AI yang menggerakkan mobil harus mempertimbangkan semua kemungkinan skenario yang dapat dihadapi mobil, seperti malam hari, badai petir atau badai salju, anak-anak berlari di jalan, hewan peliharaan, konstruksi jalan, dll. Seberapa baik sistem AI dapat menangani berbagai kondisi secara andal dan aman mencerminkan tingkat antisipasi yang dipertimbangkan oleh ilmuwan data atau pengembang AI selama desain atau pengujian sistem.

> [🎥 Klik di sini untuk video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusivitas

Sistem AI harus dirancang untuk melibatkan dan memberdayakan semua orang. Saat merancang dan mengimplementasikan sistem AI, ilmuwan data dan pengembang AI mengidentifikasi dan mengatasi hambatan potensial dalam sistem yang dapat secara tidak sengaja mengecualikan orang. Misalnya, ada 1 miliar orang dengan disabilitas di seluruh dunia. Dengan kemajuan AI, mereka dapat mengakses berbagai informasi dan peluang dengan lebih mudah dalam kehidupan sehari-hari mereka. Dengan mengatasi hambatan tersebut, tercipta peluang untuk berinovasi dan mengembangkan produk AI dengan pengalaman yang lebih baik yang menguntungkan semua orang.

> [🎥 Klik di sini untuk video: inklusivitas dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Keamanan dan privasi

Sistem AI harus aman dan menghormati privasi orang. Orang memiliki kepercayaan yang lebih rendah pada sistem yang membahayakan privasi, informasi, atau kehidupan mereka. Saat melatih model machine learning, kita mengandalkan data untuk menghasilkan hasil terbaik. Dalam melakukannya, asal dan integritas data harus dipertimbangkan. Misalnya, apakah data dikirimkan oleh pengguna atau tersedia secara publik? Selanjutnya, saat bekerja dengan data, sangat penting untuk mengembangkan sistem AI yang dapat melindungi informasi rahasia dan tahan terhadap serangan. Seiring dengan semakin meluasnya penggunaan AI, melindungi privasi dan mengamankan informasi pribadi serta bisnis yang penting menjadi semakin kritis dan kompleks. Masalah privasi dan keamanan data memerlukan perhatian khusus untuk AI karena akses ke data sangat penting bagi sistem AI untuk membuat prediksi dan keputusan yang akurat dan terinformasi tentang orang.

> [🎥 Klik di sini untuk video: keamanan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sebagai industri, kita telah membuat kemajuan signifikan dalam privasi & keamanan, yang didorong secara signifikan oleh regulasi seperti GDPR (General Data Protection Regulation).
- Namun, dengan sistem AI, kita harus mengakui adanya ketegangan antara kebutuhan akan data pribadi yang lebih banyak untuk membuat sistem lebih personal dan efektif – dan privasi.
- Sama seperti dengan lahirnya komputer yang terhubung dengan internet, kita juga melihat peningkatan besar dalam jumlah masalah keamanan terkait AI.
- Pada saat yang sama, kita telah melihat AI digunakan untuk meningkatkan keamanan. Sebagai contoh, sebagian besar pemindai antivirus modern didukung oleh heuristik AI saat ini.
- Kita perlu memastikan bahwa proses Data Science kita berpadu harmonis dengan praktik privasi dan keamanan terbaru.

### Transparansi

Sistem AI harus dapat dipahami. Bagian penting dari transparansi adalah menjelaskan perilaku sistem AI dan komponennya. Meningkatkan pemahaman tentang sistem AI membutuhkan para pemangku kepentingan untuk memahami bagaimana dan mengapa mereka berfungsi sehingga mereka dapat mengidentifikasi potensi masalah kinerja, kekhawatiran keamanan dan privasi, bias, praktik eksklusif, atau hasil yang tidak diinginkan. Kami juga percaya bahwa mereka yang menggunakan sistem AI harus jujur dan terbuka tentang kapan, mengapa, dan bagaimana mereka memilih untuk menerapkannya, serta keterbatasan sistem yang mereka gunakan. Misalnya, jika sebuah bank menggunakan sistem AI untuk mendukung keputusan pemberian pinjaman konsumen, penting untuk memeriksa hasilnya dan memahami data mana yang memengaruhi rekomendasi sistem. Pemerintah mulai mengatur AI di berbagai industri, sehingga ilmuwan data dan organisasi harus menjelaskan apakah sistem AI memenuhi persyaratan regulasi, terutama ketika ada hasil yang tidak diinginkan.

> [🎥 Klik di sini untuk video: transparansi dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Karena sistem AI sangat kompleks, sulit untuk memahami cara kerjanya dan menafsirkan hasilnya.
- Kurangnya pemahaman ini memengaruhi cara sistem ini dikelola, dioperasikan, dan didokumentasikan.
- Lebih penting lagi, kurangnya pemahaman ini memengaruhi keputusan yang dibuat menggunakan hasil yang dihasilkan oleh sistem ini.

### Akuntabilitas

Orang-orang yang merancang dan menerapkan sistem AI harus bertanggung jawab atas cara sistem mereka beroperasi. Kebutuhan akan akuntabilitas sangat penting dengan teknologi yang sensitif seperti pengenalan wajah. Baru-baru ini, ada permintaan yang meningkat untuk teknologi pengenalan wajah, terutama dari organisasi penegak hukum yang melihat potensi teknologi ini dalam penggunaan seperti menemukan anak-anak yang hilang. Namun, teknologi ini berpotensi digunakan oleh pemerintah untuk membahayakan kebebasan fundamental warganya, misalnya, dengan memungkinkan pengawasan terus-menerus terhadap individu tertentu. Oleh karena itu, ilmuwan data dan organisasi perlu bertanggung jawab atas bagaimana sistem AI mereka memengaruhi individu atau masyarakat.

[![Peneliti AI Terkemuka Memperingatkan Pengawasan Massal Melalui Pengenalan Wajah](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Pendekatan Microsoft terhadap AI yang Bertanggung Jawab")

> 🎥 Klik gambar di atas untuk video: Peringatan Pengawasan Massal Melalui Pengenalan Wajah

Pada akhirnya, salah satu pertanyaan terbesar untuk generasi kita, sebagai generasi pertama yang membawa AI ke masyarakat, adalah bagaimana memastikan bahwa komputer tetap bertanggung jawab kepada manusia dan bagaimana memastikan bahwa orang-orang yang merancang komputer tetap bertanggung jawab kepada semua orang.

## Penilaian Dampak

Sebelum melatih model machine learning, penting untuk melakukan penilaian dampak untuk memahami tujuan sistem AI; apa penggunaan yang dimaksudkan; di mana sistem akan diterapkan; dan siapa yang akan berinteraksi dengan sistem tersebut. Hal ini membantu peninjau atau penguji mengevaluasi sistem untuk mengetahui faktor-faktor yang perlu dipertimbangkan saat mengidentifikasi risiko potensial dan konsekuensi yang diharapkan.

Berikut adalah area fokus saat melakukan penilaian dampak:

* **Dampak buruk pada individu**. Menyadari adanya pembatasan atau persyaratan, penggunaan yang tidak didukung, atau keterbatasan yang diketahui yang menghambat kinerja sistem sangat penting untuk memastikan bahwa sistem tidak digunakan dengan cara yang dapat merugikan individu.
* **Persyaratan data**. Memahami bagaimana dan di mana sistem akan menggunakan data memungkinkan peninjau untuk mengeksplorasi persyaratan data apa yang perlu diperhatikan (misalnya, regulasi data GDPR atau HIPPA). Selain itu, periksa apakah sumber atau jumlah data cukup untuk pelatihan.
* **Ringkasan dampak**. Kumpulkan daftar potensi kerugian yang dapat timbul dari penggunaan sistem. Sepanjang siklus hidup ML, tinjau apakah masalah yang diidentifikasi telah diatasi atau diminimalkan.
* **Tujuan yang berlaku** untuk masing-masing dari enam prinsip inti. Tinjau apakah tujuan dari setiap prinsip telah terpenuhi dan apakah ada celah.

## Debugging dengan AI yang Bertanggung Jawab

Seperti debugging aplikasi perangkat lunak, debugging sistem AI adalah proses yang diperlukan untuk mengidentifikasi dan menyelesaikan masalah dalam sistem. Ada banyak faktor yang dapat memengaruhi model tidak berkinerja seperti yang diharapkan atau secara bertanggung jawab. Sebagian besar metrik kinerja model tradisional adalah agregat kuantitatif dari kinerja model, yang tidak cukup untuk menganalisis bagaimana model melanggar prinsip AI yang bertanggung jawab. Selain itu, model machine learning adalah kotak hitam yang membuatnya sulit untuk memahami apa yang mendorong hasilnya atau memberikan penjelasan ketika terjadi kesalahan. Nanti dalam kursus ini, kita akan belajar bagaimana menggunakan dashboard AI yang Bertanggung Jawab untuk membantu debugging sistem AI. Dashboard ini menyediakan alat holistik bagi ilmuwan data dan pengembang AI untuk melakukan:

* **Analisis kesalahan**. Untuk mengidentifikasi distribusi kesalahan model yang dapat memengaruhi keadilan atau keandalan sistem.
* **Ikhtisar model**. Untuk menemukan di mana terdapat disparitas dalam kinerja model di berbagai kelompok data.
* **Analisis data**. Untuk memahami distribusi data dan mengidentifikasi potensi bias dalam data yang dapat menyebabkan masalah keadilan, inklusivitas, dan keandalan.
* **Interpretabilitas model**. Untuk memahami apa yang memengaruhi atau memengaruhi prediksi model. Ini membantu menjelaskan perilaku model, yang penting untuk transparansi dan akuntabilitas.

## 🚀 Tantangan

Untuk mencegah kerugian diperkenalkan sejak awal, kita harus:

- memiliki keragaman latar belakang dan perspektif di antara orang-orang yang bekerja pada sistem
- berinvestasi dalam dataset yang mencerminkan keragaman masyarakat kita
- mengembangkan metode yang lebih baik di seluruh siklus hidup machine learning untuk mendeteksi dan memperbaiki AI yang bertanggung jawab saat terjadi

Pikirkan skenario kehidupan nyata di mana ketidakpercayaan model terlihat dalam pembangunan dan penggunaan model. Apa lagi yang harus kita pertimbangkan?

## [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Dalam pelajaran ini, Anda telah mempelajari beberapa dasar konsep keadilan dan ketidakadilan dalam machine learning.
Tonton lokakarya ini untuk mendalami topik berikut:

- Mengejar AI yang bertanggung jawab: Menerapkan prinsip ke dalam praktik oleh Besmira Nushi, Mehrnoosh Sameki, dan Amit Sharma

[![Responsible AI Toolbox: Kerangka kerja open-source untuk membangun AI yang bertanggung jawab](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Kerangka kerja open-source untuk membangun AI yang bertanggung jawab")

> 🎥 Klik gambar di atas untuk menonton video: RAI Toolbox: Kerangka kerja open-source untuk membangun AI yang bertanggung jawab oleh Besmira Nushi, Mehrnoosh Sameki, dan Amit Sharma

Selain itu, baca:

- Pusat sumber daya RAI Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kelompok penelitian FATE Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repositori GitHub Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Baca tentang alat Azure Machine Learning untuk memastikan keadilan:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tugas

[Eksplorasi RAI Toolbox](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.