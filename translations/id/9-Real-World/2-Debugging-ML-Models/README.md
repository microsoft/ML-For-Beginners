<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T19:27:38+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "id"
}
-->
# Postscript: Debugging Model dalam Pembelajaran Mesin menggunakan Komponen Dasbor AI yang Bertanggung Jawab

## [Kuis pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Pendahuluan

Pembelajaran mesin memengaruhi kehidupan kita sehari-hari. AI mulai digunakan dalam beberapa sistem paling penting yang memengaruhi kita sebagai individu maupun masyarakat, seperti di bidang kesehatan, keuangan, pendidikan, dan pekerjaan. Misalnya, sistem dan model terlibat dalam tugas pengambilan keputusan sehari-hari, seperti diagnosis kesehatan atau mendeteksi penipuan. Akibatnya, kemajuan AI yang pesat dan adopsinya yang semakin meluas dihadapkan pada ekspektasi masyarakat yang terus berkembang serta regulasi yang semakin ketat. Kita sering melihat area di mana sistem AI gagal memenuhi ekspektasi; mereka memunculkan tantangan baru; dan pemerintah mulai mengatur solusi AI. Oleh karena itu, penting untuk menganalisis model-model ini agar dapat memberikan hasil yang adil, dapat diandalkan, inklusif, transparan, dan bertanggung jawab bagi semua orang.

Dalam kurikulum ini, kita akan melihat alat-alat praktis yang dapat digunakan untuk menilai apakah sebuah model memiliki masalah AI yang bertanggung jawab. Teknik debugging pembelajaran mesin tradisional cenderung didasarkan pada perhitungan kuantitatif seperti akurasi agregat atau rata-rata error loss. Bayangkan apa yang bisa terjadi jika data yang Anda gunakan untuk membangun model ini kekurangan demografi tertentu, seperti ras, gender, pandangan politik, agama, atau secara tidak proporsional mewakili demografi tersebut. Bagaimana jika output model diinterpretasikan untuk menguntungkan beberapa demografi? Hal ini dapat menyebabkan representasi berlebihan atau kurang dari kelompok fitur sensitif ini, yang mengakibatkan masalah keadilan, inklusivitas, atau keandalan dari model tersebut. Faktor lainnya adalah, model pembelajaran mesin sering dianggap sebagai kotak hitam, yang membuatnya sulit untuk memahami dan menjelaskan apa yang mendorong prediksi model. Semua ini adalah tantangan yang dihadapi oleh ilmuwan data dan pengembang AI ketika mereka tidak memiliki alat yang memadai untuk debugging dan menilai keadilan atau kepercayaan model.

Dalam pelajaran ini, Anda akan belajar tentang debugging model Anda menggunakan:

- **Analisis Kesalahan**: mengidentifikasi di mana dalam distribusi data Anda model memiliki tingkat kesalahan yang tinggi.
- **Ikhtisar Model**: melakukan analisis komparatif di berbagai kelompok data untuk menemukan disparitas dalam metrik kinerja model Anda.
- **Analisis Data**: menyelidiki di mana mungkin ada representasi berlebihan atau kurang dari data Anda yang dapat membuat model Anda cenderung menguntungkan satu demografi data dibandingkan yang lain.
- **Pentingnya Fitur**: memahami fitur mana yang mendorong prediksi model Anda pada tingkat global atau lokal.

## Prasyarat

Sebagai prasyarat, silakan tinjau [Alat AI yang Bertanggung Jawab untuk Pengembang](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif tentang Alat AI yang Bertanggung Jawab](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analisis Kesalahan

Metrik kinerja model tradisional yang digunakan untuk mengukur akurasi sebagian besar adalah perhitungan berdasarkan prediksi yang benar vs salah. Misalnya, menentukan bahwa sebuah model akurat 89% dengan error loss sebesar 0.001 dapat dianggap sebagai kinerja yang baik. Kesalahan sering kali tidak terdistribusi secara merata dalam dataset Anda. Anda mungkin mendapatkan skor akurasi model 89% tetapi menemukan bahwa ada wilayah data tertentu di mana model gagal 42% dari waktu. Konsekuensi dari pola kegagalan ini dengan kelompok data tertentu dapat menyebabkan masalah keadilan atau keandalan. Sangat penting untuk memahami area di mana model berkinerja baik atau tidak. Wilayah data di mana terdapat banyak ketidakakuratan dalam model Anda mungkin ternyata merupakan demografi data yang penting.

![Analisis dan debugging kesalahan model](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Komponen Analisis Kesalahan pada dasbor RAI menggambarkan bagaimana kegagalan model terdistribusi di berbagai kelompok dengan visualisasi pohon. Ini berguna untuk mengidentifikasi fitur atau area di mana terdapat tingkat kesalahan tinggi dalam dataset Anda. Dengan melihat dari mana sebagian besar ketidakakuratan model berasal, Anda dapat mulai menyelidiki akar penyebabnya. Anda juga dapat membuat kelompok data untuk melakukan analisis. Kelompok data ini membantu dalam proses debugging untuk menentukan mengapa kinerja model baik di satu kelompok, tetapi salah di kelompok lain.

![Analisis Kesalahan](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Indikator visual pada peta pohon membantu menemukan area masalah dengan lebih cepat. Misalnya, semakin gelap warna merah pada node pohon, semakin tinggi tingkat kesalahannya.

Peta panas adalah fungsi visualisasi lain yang dapat digunakan pengguna untuk menyelidiki tingkat kesalahan menggunakan satu atau dua fitur untuk menemukan penyebab kesalahan model di seluruh dataset atau kelompok.

![Peta Panas Analisis Kesalahan](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Gunakan analisis kesalahan ketika Anda perlu:

* Memahami secara mendalam bagaimana kegagalan model terdistribusi di seluruh dataset dan di berbagai dimensi input dan fitur.
* Memecah metrik kinerja agregat untuk secara otomatis menemukan kelompok yang salah guna menginformasikan langkah-langkah mitigasi yang ditargetkan.

## Ikhtisar Model

Mengevaluasi kinerja model pembelajaran mesin membutuhkan pemahaman menyeluruh tentang perilakunya. Hal ini dapat dicapai dengan meninjau lebih dari satu metrik seperti tingkat kesalahan, akurasi, recall, presisi, atau MAE (Mean Absolute Error) untuk menemukan disparitas di antara metrik kinerja. Satu metrik kinerja mungkin terlihat bagus, tetapi ketidakakuratan dapat terungkap dalam metrik lain. Selain itu, membandingkan metrik untuk disparitas di seluruh dataset atau kelompok membantu memberikan wawasan tentang di mana model berkinerja baik atau tidak. Hal ini sangat penting untuk melihat kinerja model di antara fitur sensitif vs tidak sensitif (misalnya, ras pasien, gender, atau usia) untuk mengungkap potensi ketidakadilan yang mungkin dimiliki model. Misalnya, menemukan bahwa model lebih sering salah di kelompok yang memiliki fitur sensitif dapat mengungkap potensi ketidakadilan yang mungkin dimiliki model.

Komponen Ikhtisar Model pada dasbor RAI membantu tidak hanya dalam menganalisis metrik kinerja dari representasi data dalam kelompok, tetapi juga memberikan pengguna kemampuan untuk membandingkan perilaku model di berbagai kelompok.

![Kelompok dataset - ikhtisar model dalam dasbor RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Fungsi analisis berbasis fitur dari komponen ini memungkinkan pengguna untuk mempersempit subkelompok data dalam fitur tertentu untuk mengidentifikasi anomali pada tingkat yang lebih rinci. Misalnya, dasbor memiliki kecerdasan bawaan untuk secara otomatis menghasilkan kelompok untuk fitur yang dipilih pengguna (misalnya, *"time_in_hospital < 3"* atau *"time_in_hospital >= 7"*). Hal ini memungkinkan pengguna untuk mengisolasi fitur tertentu dari kelompok data yang lebih besar untuk melihat apakah fitur tersebut merupakan pengaruh utama dari hasil yang salah pada model.

![Kelompok fitur - ikhtisar model dalam dasbor RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Komponen Ikhtisar Model mendukung dua kelas metrik disparitas:

**Disparitas dalam kinerja model**: Serangkaian metrik ini menghitung disparitas (perbedaan) dalam nilai metrik kinerja yang dipilih di seluruh subkelompok data. Berikut beberapa contohnya:

* Disparitas dalam tingkat akurasi
* Disparitas dalam tingkat kesalahan
* Disparitas dalam presisi
* Disparitas dalam recall
* Disparitas dalam mean absolute error (MAE)

**Disparitas dalam tingkat seleksi**: Metrik ini berisi perbedaan dalam tingkat seleksi (prediksi yang menguntungkan) di antara subkelompok. Contohnya adalah disparitas dalam tingkat persetujuan pinjaman. Tingkat seleksi berarti fraksi titik data dalam setiap kelas yang diklasifikasikan sebagai 1 (dalam klasifikasi biner) atau distribusi nilai prediksi (dalam regresi).

## Analisis Data

> "Jika Anda menyiksa data cukup lama, data akan mengaku apa saja" - Ronald Coase

Pernyataan ini terdengar ekstrem, tetapi benar bahwa data dapat dimanipulasi untuk mendukung kesimpulan apa pun. Manipulasi semacam itu terkadang terjadi secara tidak sengaja. Sebagai manusia, kita semua memiliki bias, dan sering kali sulit untuk secara sadar mengetahui kapan kita memperkenalkan bias dalam data. Menjamin keadilan dalam AI dan pembelajaran mesin tetap menjadi tantangan yang kompleks.

Data adalah titik buta besar untuk metrik kinerja model tradisional. Anda mungkin memiliki skor akurasi tinggi, tetapi ini tidak selalu mencerminkan bias data yang mendasari yang mungkin ada dalam dataset Anda. Misalnya, jika dataset karyawan memiliki 27% wanita di posisi eksekutif di sebuah perusahaan dan 73% pria di tingkat yang sama, model AI periklanan pekerjaan yang dilatih pada data ini mungkin menargetkan sebagian besar audiens pria untuk posisi pekerjaan tingkat senior. Ketidakseimbangan dalam data ini membuat prediksi model cenderung menguntungkan satu gender. Hal ini mengungkapkan masalah keadilan di mana terdapat bias gender dalam model AI.

Komponen Analisis Data pada dasbor RAI membantu mengidentifikasi area di mana terdapat representasi berlebihan dan kurang dalam dataset. Ini membantu pengguna mendiagnosis akar penyebab kesalahan dan masalah keadilan yang diperkenalkan dari ketidakseimbangan data atau kurangnya representasi dari kelompok data tertentu. Hal ini memberikan pengguna kemampuan untuk memvisualisasikan dataset berdasarkan hasil prediksi dan aktual, kelompok kesalahan, dan fitur tertentu. Terkadang menemukan kelompok data yang kurang terwakili juga dapat mengungkap bahwa model tidak belajar dengan baik, sehingga tingkat ketidakakuratan tinggi. Memiliki model dengan bias data bukan hanya masalah keadilan tetapi juga menunjukkan bahwa model tidak inklusif atau dapat diandalkan.

![Komponen Analisis Data pada Dasbor RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Gunakan analisis data ketika Anda perlu:

* Mengeksplorasi statistik dataset Anda dengan memilih filter berbeda untuk membagi data Anda ke dalam berbagai dimensi (juga dikenal sebagai kelompok).
* Memahami distribusi dataset Anda di berbagai kelompok dan grup fitur.
* Menentukan apakah temuan Anda terkait dengan keadilan, analisis kesalahan, dan kausalitas (yang berasal dari komponen dasbor lainnya) adalah hasil dari distribusi dataset Anda.
* Memutuskan di area mana untuk mengumpulkan lebih banyak data guna mengurangi kesalahan yang berasal dari masalah representasi, kebisingan label, kebisingan fitur, bias label, dan faktor serupa.

## Interpretabilitas Model

Model pembelajaran mesin cenderung menjadi kotak hitam. Memahami fitur data utama mana yang mendorong prediksi model bisa menjadi tantangan. Penting untuk memberikan transparansi mengapa model membuat prediksi tertentu. Misalnya, jika sistem AI memprediksi bahwa seorang pasien diabetes berisiko dirawat kembali di rumah sakit dalam waktu kurang dari 30 hari, sistem tersebut harus dapat memberikan data pendukung yang menyebabkan prediksinya. Memiliki indikator data pendukung memberikan transparansi untuk membantu dokter atau rumah sakit membuat keputusan yang terinformasi dengan baik. Selain itu, mampu menjelaskan mengapa model membuat prediksi untuk pasien individu memungkinkan akuntabilitas dengan regulasi kesehatan. Ketika Anda menggunakan model pembelajaran mesin dengan cara yang memengaruhi kehidupan manusia, sangat penting untuk memahami dan menjelaskan apa yang memengaruhi perilaku model. Penjelasan dan interpretabilitas model membantu menjawab pertanyaan dalam skenario seperti:

* Debugging model: Mengapa model saya membuat kesalahan ini? Bagaimana saya dapat meningkatkan model saya?
* Kolaborasi manusia-AI: Bagaimana saya dapat memahami dan mempercayai keputusan model?
* Kepatuhan regulasi: Apakah model saya memenuhi persyaratan hukum?

Komponen Pentingnya Fitur pada dasbor RAI membantu Anda untuk debugging dan mendapatkan pemahaman yang komprehensif tentang bagaimana model membuat prediksi. Ini juga merupakan alat yang berguna bagi profesional pembelajaran mesin dan pengambil keputusan untuk menjelaskan dan menunjukkan bukti fitur yang memengaruhi perilaku model untuk kepatuhan regulasi. Selanjutnya, pengguna dapat mengeksplorasi penjelasan global dan lokal untuk memvalidasi fitur mana yang mendorong prediksi model. Penjelasan global mencantumkan fitur utama yang memengaruhi prediksi keseluruhan model. Penjelasan lokal menampilkan fitur mana yang menyebabkan prediksi model untuk kasus individu. Kemampuan untuk mengevaluasi penjelasan lokal juga berguna dalam debugging atau audit kasus tertentu untuk lebih memahami dan menginterpretasikan mengapa model membuat prediksi yang akurat atau tidak akurat.

![Komponen Pentingnya Fitur pada dasbor RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Penjelasan global: Misalnya, fitur apa yang memengaruhi perilaku keseluruhan model rawat inap diabetes?
* Penjelasan lokal: Misalnya, mengapa seorang pasien diabetes berusia di atas 60 tahun dengan rawat inap sebelumnya diprediksi akan dirawat kembali atau tidak dirawat kembali dalam waktu 30 hari di rumah sakit?

Dalam proses debugging untuk memeriksa kinerja model di berbagai kelompok, Pentingnya Fitur menunjukkan tingkat dampak fitur di seluruh kelompok. Ini membantu mengungkap anomali saat membandingkan tingkat pengaruh fitur dalam mendorong prediksi model yang salah. Komponen Pentingnya Fitur dapat menunjukkan nilai mana dalam fitur yang secara positif atau negatif memengaruhi hasil model. Misalnya, jika model membuat prediksi yang tidak akurat, komponen ini memberikan kemampuan untuk mengeksplorasi lebih dalam dan mengidentifikasi fitur atau nilai fitur mana yang mendorong prediksi tersebut. Tingkat detail ini tidak hanya membantu dalam debugging tetapi juga memberikan transparansi dan akuntabilitas dalam situasi audit. Akhirnya, komponen ini dapat membantu Anda mengidentifikasi masalah keadilan. Sebagai ilustrasi, jika fitur sensitif seperti etnis atau gender sangat berpengaruh dalam mendorong prediksi model, ini bisa menjadi tanda bias ras atau gender dalam model.

![Pentingnya fitur](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Gunakan interpretabilitas ketika Anda perlu:

* Menentukan seberapa dapat dipercaya prediksi sistem AI Anda dengan memahami fitur mana yang paling penting untuk prediksi tersebut.
* Mendekati debugging model Anda dengan terlebih dahulu memahaminya dan mengidentifikasi apakah model menggunakan fitur yang sehat atau hanya korelasi palsu.
* Mengungkap potensi sumber ketidakadilan dengan memahami apakah model mendasarkan prediksi pada fitur sensitif atau fitur yang sangat berkorelasi dengan fitur sensitif.
* Membangun kepercayaan pengguna pada keputusan model Anda dengan menghasilkan penjelasan lokal untuk menggambarkan hasilnya.
* Menyelesaikan audit regulasi sistem AI untuk memvalidasi model dan memantau dampak keputusan model pada manusia.

## Kesimpulan

Semua komponen dasbor RAI adalah alat praktis untuk membantu Anda membangun model pembelajaran mesin yang lebih aman dan lebih dapat dipercaya oleh masyarakat. Ini meningkatkan pencegahan ancaman terhadap hak asasi manusia; diskriminasi atau pengecualian kelompok tertentu terhadap peluang hidup; dan risiko cedera fisik atau psikologis. Ini juga membantu membangun kepercayaan pada keputusan model Anda dengan menghasilkan penjelasan lokal untuk menggambarkan hasilnya. Beberapa potensi kerugian dapat diklasifikasikan sebagai:

- **Distribusi**, jika gender atau etnis misalnya lebih diuntungkan dibandingkan yang lain.
- **Kualitas layanan**. Jika Anda melatih data untuk satu skenario spesifik tetapi kenyataannya jauh lebih kompleks, hal ini menyebabkan layanan yang berkinerja buruk.
- **Stereotip**. Mengasosiasikan kelompok tertentu dengan atribut yang telah ditentukan sebelumnya.
- **Pelecehan**. Mengkritik dan melabeli sesuatu atau seseorang secara tidak adil.
- **Representasi berlebihan atau kurang**. Ide utamanya adalah bahwa kelompok tertentu tidak terlihat dalam profesi tertentu, dan setiap layanan atau fungsi yang terus mempromosikan hal tersebut berkontribusi pada kerugian.

### Azure RAI dashboard

[Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) dibangun menggunakan alat sumber terbuka yang dikembangkan oleh institusi akademik dan organisasi terkemuka, termasuk Microsoft. Alat ini sangat membantu ilmuwan data dan pengembang AI untuk lebih memahami perilaku model, menemukan, dan mengatasi masalah yang tidak diinginkan dari model AI.

- Pelajari cara menggunakan berbagai komponen dengan melihat [dokumentasi RAI dashboard.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Lihat beberapa [notebook contoh RAI dashboard](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) untuk debugging skenario AI yang lebih bertanggung jawab di Azure Machine Learning.

---
## 🚀 Tantangan

Untuk mencegah bias statistik atau data diperkenalkan sejak awal, kita harus:

- memiliki keragaman latar belakang dan perspektif di antara orang-orang yang bekerja pada sistem
- berinvestasi dalam dataset yang mencerminkan keragaman masyarakat kita
- mengembangkan metode yang lebih baik untuk mendeteksi dan memperbaiki bias saat terjadi

Pikirkan skenario kehidupan nyata di mana ketidakadilan terlihat dalam pembangunan dan penggunaan model. Apa lagi yang harus kita pertimbangkan?

## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)
## Tinjauan & Studi Mandiri

Dalam pelajaran ini, Anda telah mempelajari beberapa alat praktis untuk mengintegrasikan AI yang bertanggung jawab dalam pembelajaran mesin.

Tonton lokakarya ini untuk mendalami topik lebih lanjut:

- Responsible AI Dashboard: Satu tempat untuk mengoperasionalkan RAI dalam praktik oleh Besmira Nushi dan Mehrnoosh Sameki

[![Responsible AI Dashboard: Satu tempat untuk mengoperasionalkan RAI dalam praktik](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Satu tempat untuk mengoperasionalkan RAI dalam praktik")


> 🎥 Klik gambar di atas untuk video: Responsible AI Dashboard: Satu tempat untuk mengoperasionalkan RAI dalam praktik oleh Besmira Nushi dan Mehrnoosh Sameki

Referensi materi berikut untuk mempelajari lebih lanjut tentang AI yang bertanggung jawab dan cara membangun model yang lebih dapat dipercaya:

- Alat RAI dashboard Microsoft untuk debugging model ML: [Sumber daya alat AI yang bertanggung jawab](https://aka.ms/rai-dashboard)

- Jelajahi toolkit AI yang bertanggung jawab: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Pusat sumber daya RAI Microsoft: [Sumber Daya AI yang Bertanggung Jawab – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kelompok penelitian FATE Microsoft: [FATE: Keadilan, Akuntabilitas, Transparansi, dan Etika dalam AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Tugas

[Jelajahi RAI dashboard](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.