<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T19:23:21+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "id"
}
-->
# Postscript: Pembelajaran Mesin di Dunia Nyata

![Ringkasan pembelajaran mesin di dunia nyata dalam bentuk sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dalam kurikulum ini, Anda telah mempelajari banyak cara untuk mempersiapkan data untuk pelatihan dan membuat model pembelajaran mesin. Anda telah membangun serangkaian model klasik seperti regresi, clustering, klasifikasi, pemrosesan bahasa alami, dan deret waktu. Selamat! Sekarang, Anda mungkin bertanya-tanya untuk apa semua ini... apa aplikasi dunia nyata dari model-model ini?

Meskipun minat industri banyak tertuju pada AI yang biasanya memanfaatkan pembelajaran mendalam, masih ada aplikasi berharga untuk model pembelajaran mesin klasik. Anda bahkan mungkin menggunakan beberapa aplikasi ini hari ini! Dalam pelajaran ini, Anda akan mengeksplorasi bagaimana delapan industri dan domain keahlian yang berbeda menggunakan jenis model ini untuk membuat aplikasi mereka lebih efektif, andal, cerdas, dan bernilai bagi pengguna.

## [Kuis pra-pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Keuangan

Sektor keuangan menawarkan banyak peluang untuk pembelajaran mesin. Banyak masalah di area ini dapat dimodelkan dan diselesaikan menggunakan ML.

### Deteksi Penipuan Kartu Kredit

Kita telah mempelajari tentang [k-means clustering](../../5-Clustering/2-K-Means/README.md) sebelumnya dalam kursus ini, tetapi bagaimana cara menggunakannya untuk menyelesaikan masalah terkait penipuan kartu kredit?

K-means clustering berguna dalam teknik deteksi penipuan kartu kredit yang disebut **deteksi outlier**. Outlier, atau penyimpangan dalam pengamatan terhadap sekumpulan data, dapat memberi tahu kita apakah kartu kredit digunakan secara normal atau ada sesuatu yang tidak biasa. Seperti yang ditunjukkan dalam makalah yang ditautkan di bawah ini, Anda dapat mengelompokkan data kartu kredit menggunakan algoritma k-means clustering dan menetapkan setiap transaksi ke dalam cluster berdasarkan seberapa besar outlier-nya. Kemudian, Anda dapat mengevaluasi cluster yang paling berisiko untuk transaksi penipuan versus transaksi yang sah.
[Referensi](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Manajemen Kekayaan

Dalam manajemen kekayaan, individu atau perusahaan mengelola investasi atas nama klien mereka. Tugas mereka adalah mempertahankan dan meningkatkan kekayaan dalam jangka panjang, sehingga penting untuk memilih investasi yang berkinerja baik.

Salah satu cara untuk mengevaluasi bagaimana suatu investasi berkinerja adalah melalui regresi statistik. [Regresi linear](../../2-Regression/1-Tools/README.md) adalah alat yang berharga untuk memahami bagaimana suatu dana berkinerja relatif terhadap tolok ukur tertentu. Kita juga dapat menyimpulkan apakah hasil regresi tersebut signifikan secara statistik, atau seberapa besar pengaruhnya terhadap investasi klien. Anda bahkan dapat memperluas analisis Anda menggunakan regresi berganda, di mana faktor risiko tambahan dapat diperhitungkan. Untuk contoh bagaimana ini akan bekerja untuk dana tertentu, lihat makalah di bawah ini tentang mengevaluasi kinerja dana menggunakan regresi.
[Referensi](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Pendidikan

Sektor pendidikan juga merupakan area yang sangat menarik untuk penerapan ML. Ada masalah menarik yang dapat diatasi seperti mendeteksi kecurangan pada tes atau esai, atau mengelola bias, baik yang disengaja maupun tidak, dalam proses koreksi.

### Memprediksi Perilaku Siswa

[Coursera](https://coursera.com), penyedia kursus online terbuka, memiliki blog teknologi yang hebat di mana mereka membahas banyak keputusan teknik. Dalam studi kasus ini, mereka memplot garis regresi untuk mencoba mengeksplorasi korelasi antara rating NPS (Net Promoter Score) yang rendah dan retensi atau penurunan kursus.
[Referensi](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mengurangi Bias

[Grammarly](https://grammarly.com), asisten penulisan yang memeriksa kesalahan ejaan dan tata bahasa, menggunakan sistem [pemrosesan bahasa alami](../../6-NLP/README.md) yang canggih di seluruh produknya. Mereka menerbitkan studi kasus menarik di blog teknologi mereka tentang bagaimana mereka menangani bias gender dalam pembelajaran mesin, yang telah Anda pelajari dalam [pelajaran pengantar tentang keadilan](../../1-Introduction/3-fairness/README.md).
[Referensi](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Ritel

Sektor ritel dapat sangat diuntungkan dari penggunaan ML, mulai dari menciptakan perjalanan pelanggan yang lebih baik hingga mengelola inventaris secara optimal.

### Personalisasi Perjalanan Pelanggan

Di Wayfair, sebuah perusahaan yang menjual barang-barang rumah tangga seperti furnitur, membantu pelanggan menemukan produk yang sesuai dengan selera dan kebutuhan mereka adalah hal yang utama. Dalam artikel ini, para insinyur dari perusahaan tersebut menjelaskan bagaimana mereka menggunakan ML dan NLP untuk "menampilkan hasil yang tepat bagi pelanggan". Secara khusus, Query Intent Engine mereka telah dibangun untuk menggunakan ekstraksi entitas, pelatihan klasifikasi, ekstraksi aset dan opini, serta penandaan sentimen pada ulasan pelanggan. Ini adalah contoh klasik bagaimana NLP bekerja dalam ritel online.
[Referensi](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Manajemen Inventaris

Perusahaan inovatif dan gesit seperti [StitchFix](https://stitchfix.com), layanan kotak yang mengirimkan pakaian kepada konsumen, sangat bergantung pada ML untuk rekomendasi dan manajemen inventaris. Tim styling mereka bekerja sama dengan tim merchandising mereka, bahkan: "salah satu ilmuwan data kami bereksperimen dengan algoritma genetik dan menerapkannya pada pakaian untuk memprediksi apa yang akan menjadi pakaian yang sukses yang belum ada saat ini. Kami membawa itu ke tim merchandise dan sekarang mereka dapat menggunakannya sebagai alat."
[Referensi](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Kesehatan

Sektor kesehatan dapat memanfaatkan ML untuk mengoptimalkan tugas penelitian dan juga masalah logistik seperti readmisi pasien atau menghentikan penyebaran penyakit.

### Manajemen Uji Klinis

Toksisitas dalam uji klinis adalah perhatian utama bagi pembuat obat. Seberapa banyak toksisitas yang dapat ditoleransi? Dalam studi ini, menganalisis berbagai metode uji klinis menghasilkan pendekatan baru untuk memprediksi peluang hasil uji klinis. Secara khusus, mereka dapat menggunakan random forest untuk menghasilkan [classifier](../../4-Classification/README.md) yang mampu membedakan antara kelompok obat.
[Referensi](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Manajemen Readmisi Rumah Sakit

Perawatan rumah sakit mahal, terutama ketika pasien harus readmisi. Makalah ini membahas sebuah perusahaan yang menggunakan ML untuk memprediksi potensi readmisi menggunakan algoritma [clustering](../../5-Clustering/README.md). Cluster ini membantu analis untuk "menemukan kelompok readmisi yang mungkin memiliki penyebab yang sama".
[Referensi](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Manajemen Penyakit

Pandemi baru-baru ini telah menyoroti cara-cara pembelajaran mesin dapat membantu menghentikan penyebaran penyakit. Dalam artikel ini, Anda akan mengenali penggunaan ARIMA, kurva logistik, regresi linear, dan SARIMA. "Pekerjaan ini adalah upaya untuk menghitung tingkat penyebaran virus ini dan dengan demikian memprediksi kematian, pemulihan, dan kasus yang dikonfirmasi, sehingga dapat membantu kita mempersiapkan diri dengan lebih baik dan bertahan."
[Referensi](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologi dan Teknologi Hijau

Alam dan ekologi terdiri dari banyak sistem sensitif di mana interaksi antara hewan dan alam menjadi fokus. Penting untuk dapat mengukur sistem ini secara akurat dan bertindak dengan tepat jika sesuatu terjadi, seperti kebakaran hutan atau penurunan populasi hewan.

### Manajemen Hutan

Anda telah mempelajari tentang [Reinforcement Learning](../../8-Reinforcement/README.md) dalam pelajaran sebelumnya. Ini bisa sangat berguna saat mencoba memprediksi pola di alam. Secara khusus, ini dapat digunakan untuk melacak masalah ekologi seperti kebakaran hutan dan penyebaran spesies invasif. Di Kanada, sekelompok peneliti menggunakan Reinforcement Learning untuk membangun model dinamika kebakaran hutan dari citra satelit. Menggunakan "spatially spreading process (SSP)" yang inovatif, mereka membayangkan kebakaran hutan sebagai "agen di setiap sel dalam lanskap." "Set tindakan yang dapat diambil oleh api dari suatu lokasi pada titik waktu tertentu termasuk menyebar ke utara, selatan, timur, atau barat atau tidak menyebar.

Pendekatan ini membalikkan pengaturan RL biasa karena dinamika Proses Keputusan Markov (MDP) yang sesuai adalah fungsi yang diketahui untuk penyebaran kebakaran langsung." Baca lebih lanjut tentang algoritma klasik yang digunakan oleh kelompok ini di tautan di bawah ini.
[Referensi](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sensor Gerak Hewan

Meskipun pembelajaran mendalam telah menciptakan revolusi dalam melacak gerakan hewan secara visual (Anda dapat membuat [pelacak beruang kutub](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) Anda sendiri di sini), ML klasik masih memiliki tempat dalam tugas ini.

Sensor untuk melacak gerakan hewan ternak dan IoT memanfaatkan jenis pemrosesan visual ini, tetapi teknik ML yang lebih dasar berguna untuk memproses data awal. Misalnya, dalam makalah ini, postur domba dipantau dan dianalisis menggunakan berbagai algoritma klasifikasi. Anda mungkin mengenali kurva ROC di halaman 335.
[Referensi](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Manajemen Energi

Dalam pelajaran kami tentang [peramalan deret waktu](../../7-TimeSeries/README.md), kami mengangkat konsep meteran parkir pintar untuk menghasilkan pendapatan bagi kota berdasarkan pemahaman tentang penawaran dan permintaan. Artikel ini membahas secara rinci bagaimana clustering, regresi, dan peramalan deret waktu digabungkan untuk membantu memprediksi penggunaan energi di masa depan di Irlandia, berdasarkan meteran pintar.
[Referensi](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Asuransi

Sektor asuransi adalah sektor lain yang menggunakan ML untuk membangun dan mengoptimalkan model keuangan dan aktuaria yang layak.

### Manajemen Volatilitas

MetLife, penyedia asuransi jiwa, terbuka tentang cara mereka menganalisis dan mengurangi volatilitas dalam model keuangan mereka. Dalam artikel ini Anda akan melihat visualisasi klasifikasi biner dan ordinal. Anda juga akan menemukan visualisasi peramalan.
[Referensi](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Seni, Budaya, dan Sastra

Dalam seni, misalnya dalam jurnalisme, ada banyak masalah menarik. Mendeteksi berita palsu adalah masalah besar karena telah terbukti memengaruhi opini publik dan bahkan mengguncang demokrasi. Museum juga dapat memanfaatkan ML dalam segala hal mulai dari menemukan hubungan antar artefak hingga perencanaan sumber daya.

### Deteksi Berita Palsu

Mendeteksi berita palsu telah menjadi permainan kucing dan tikus dalam media saat ini. Dalam artikel ini, para peneliti menyarankan bahwa sistem yang menggabungkan beberapa teknik ML yang telah kita pelajari dapat diuji dan model terbaik diterapkan: "Sistem ini didasarkan pada pemrosesan bahasa alami untuk mengekstrak fitur dari data dan kemudian fitur-fitur ini digunakan untuk pelatihan pengklasifikasi pembelajaran mesin seperti Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), dan Logistic Regression (LR)."
[Referensi](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Artikel ini menunjukkan bagaimana menggabungkan berbagai domain ML dapat menghasilkan hasil menarik yang dapat membantu menghentikan penyebaran berita palsu dan menciptakan kerusakan nyata; dalam kasus ini, dorongan utamanya adalah penyebaran rumor tentang pengobatan COVID yang memicu kekerasan massa.

### Museum ML

Museum berada di ambang revolusi AI di mana pengatalogan dan digitalisasi koleksi serta menemukan hubungan antar artefak menjadi lebih mudah seiring kemajuan teknologi. Proyek seperti [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) membantu membuka misteri koleksi yang tidak dapat diakses seperti Arsip Vatikan. Namun, aspek bisnis museum juga mendapat manfaat dari model ML.

Misalnya, Art Institute of Chicago membangun model untuk memprediksi apa yang diminati oleh audiens dan kapan mereka akan menghadiri pameran. Tujuannya adalah menciptakan pengalaman pengunjung yang dipersonalisasi dan dioptimalkan setiap kali pengguna mengunjungi museum. "Selama tahun fiskal 2017, model tersebut memprediksi kehadiran dan penerimaan dengan akurasi hingga 1 persen, kata Andrew Simnick, wakil presiden senior di Art Institute."
[Referensi](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Pemasaran

### Segmentasi Pelanggan

Strategi pemasaran yang paling efektif menargetkan pelanggan dengan cara yang berbeda berdasarkan berbagai pengelompokan. Dalam artikel ini, penggunaan algoritma Clustering dibahas untuk mendukung pemasaran yang berbeda. Pemasaran yang berbeda membantu perusahaan meningkatkan pengenalan merek, menjangkau lebih banyak pelanggan, dan menghasilkan lebih banyak uang.
[Referensi](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Tantangan

Identifikasi sektor lain yang mendapat manfaat dari beberapa teknik yang telah Anda pelajari dalam kurikulum ini, dan temukan bagaimana sektor tersebut menggunakan ML.
## [Kuis setelah kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Belajar Mandiri

Tim data science Wayfair memiliki beberapa video menarik tentang bagaimana mereka menggunakan ML di perusahaan mereka. Ada baiknya [melihatnya](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tugas

[Perburuan ML](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.