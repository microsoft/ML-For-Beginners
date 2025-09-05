<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T19:23:57+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "ms"
}
-->
# Postscript: Pembelajaran Mesin di Dunia Sebenar

![Ringkasan pembelajaran mesin di dunia sebenar dalam sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dalam kurikulum ini, anda telah mempelajari pelbagai cara untuk menyediakan data bagi latihan dan mencipta model pembelajaran mesin. Anda telah membina siri model regresi klasik, pengelompokan, klasifikasi, pemprosesan bahasa semula jadi, dan siri masa. Tahniah! Kini, anda mungkin tertanya-tanya apa tujuan semua ini... apakah aplikasi dunia sebenar untuk model-model ini?

Walaupun AI yang biasanya menggunakan pembelajaran mendalam telah menarik banyak perhatian dalam industri, masih terdapat aplikasi yang bernilai untuk model pembelajaran mesin klasik. Anda mungkin sudah menggunakan beberapa aplikasi ini hari ini! Dalam pelajaran ini, anda akan meneroka bagaimana lapan industri dan domain subjek yang berbeza menggunakan jenis model ini untuk menjadikan aplikasi mereka lebih berprestasi, boleh dipercayai, pintar, dan bernilai kepada pengguna.

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Kewangan

Sektor kewangan menawarkan banyak peluang untuk pembelajaran mesin. Banyak masalah dalam bidang ini sesuai untuk dimodelkan dan diselesaikan menggunakan ML.

### Pengesanan penipuan kad kredit

Kami telah mempelajari tentang [k-means clustering](../../5-Clustering/2-K-Means/README.md) sebelum ini dalam kursus, tetapi bagaimana ia boleh digunakan untuk menyelesaikan masalah berkaitan penipuan kad kredit?

K-means clustering berguna dalam teknik pengesanan penipuan kad kredit yang dipanggil **pengesanan outlier**. Outlier, atau penyimpangan dalam pemerhatian tentang satu set data, boleh memberitahu kita sama ada kad kredit sedang digunakan secara normal atau jika terdapat sesuatu yang luar biasa. Seperti yang ditunjukkan dalam kertas kerja yang dipautkan di bawah, anda boleh menyusun data kad kredit menggunakan algoritma k-means clustering dan menetapkan setiap transaksi kepada kelompok berdasarkan sejauh mana ia kelihatan sebagai outlier. Kemudian, anda boleh menilai kelompok yang paling berisiko untuk transaksi penipuan berbanding transaksi sah.
[Rujukan](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Pengurusan kekayaan

Dalam pengurusan kekayaan, individu atau firma mengendalikan pelaburan bagi pihak pelanggan mereka. Tugas mereka adalah untuk mengekalkan dan meningkatkan kekayaan dalam jangka masa panjang, jadi adalah penting untuk memilih pelaburan yang berprestasi baik.

Salah satu cara untuk menilai bagaimana sesuatu pelaburan berprestasi adalah melalui regresi statistik. [Regresi linear](../../2-Regression/1-Tools/README.md) adalah alat yang berguna untuk memahami bagaimana sesuatu dana berprestasi berbanding penanda aras tertentu. Kita juga boleh membuat kesimpulan sama ada hasil regresi itu signifikan secara statistik, atau sejauh mana ia akan mempengaruhi pelaburan pelanggan. Anda juga boleh memperluaskan analisis anda menggunakan regresi berganda, di mana faktor risiko tambahan boleh diambil kira. Untuk contoh bagaimana ini berfungsi untuk dana tertentu, lihat kertas kerja di bawah tentang menilai prestasi dana menggunakan regresi.
[Rujukan](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Pendidikan

Sektor pendidikan juga merupakan bidang yang sangat menarik di mana ML boleh digunakan. Terdapat masalah menarik untuk ditangani seperti mengesan penipuan dalam ujian atau esei atau menguruskan bias, sama ada sengaja atau tidak, dalam proses pembetulan.

### Meramalkan tingkah laku pelajar

[Coursera](https://coursera.com), penyedia kursus terbuka dalam talian, mempunyai blog teknologi yang hebat di mana mereka membincangkan banyak keputusan kejuruteraan. Dalam kajian kes ini, mereka memplotkan garis regresi untuk cuba meneroka sebarang korelasi antara penilaian NPS (Net Promoter Score) yang rendah dan pengekalan atau penurunan kursus.
[Rujukan](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mengurangkan bias

[Grammarly](https://grammarly.com), pembantu penulisan yang memeriksa kesalahan ejaan dan tatabahasa, menggunakan sistem [pemprosesan bahasa semula jadi](../../6-NLP/README.md) yang canggih dalam produknya. Mereka menerbitkan kajian kes yang menarik dalam blog teknologi mereka tentang bagaimana mereka menangani bias jantina dalam pembelajaran mesin, yang anda pelajari dalam [pelajaran pengenalan tentang keadilan](../../1-Introduction/3-fairness/README.md).
[Rujukan](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Runcit

Sektor runcit sememangnya boleh mendapat manfaat daripada penggunaan ML, dengan segala-galanya daripada mencipta perjalanan pelanggan yang lebih baik kepada pengurusan inventori secara optimum.

### Memperibadikan perjalanan pelanggan

Di Wayfair, sebuah syarikat yang menjual barangan rumah seperti perabot, membantu pelanggan mencari produk yang sesuai dengan citarasa dan keperluan mereka adalah sangat penting. Dalam artikel ini, jurutera dari syarikat tersebut menerangkan bagaimana mereka menggunakan ML dan NLP untuk "menampilkan hasil yang tepat untuk pelanggan". Khususnya, Query Intent Engine mereka telah dibina untuk menggunakan pengekstrakan entiti, latihan pengklasifikasi, pengekstrakan aset dan pendapat, serta penandaan sentimen pada ulasan pelanggan. Ini adalah contoh klasik bagaimana NLP berfungsi dalam runcit dalam talian.
[Rujukan](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Pengurusan inventori

Syarikat inovatif dan tangkas seperti [StitchFix](https://stitchfix.com), perkhidmatan kotak yang menghantar pakaian kepada pengguna, sangat bergantung pada ML untuk cadangan dan pengurusan inventori. Pasukan penggayaan mereka bekerjasama dengan pasukan merchandising mereka, sebenarnya: "salah seorang saintis data kami bermain-main dengan algoritma genetik dan menerapkannya pada pakaian untuk meramalkan apa yang akan menjadi pakaian yang berjaya yang tidak wujud hari ini. Kami membawa itu kepada pasukan merchandise dan kini mereka boleh menggunakannya sebagai alat."
[Rujukan](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Penjagaan Kesihatan

Sektor penjagaan kesihatan boleh memanfaatkan ML untuk mengoptimumkan tugas penyelidikan dan juga masalah logistik seperti kemasukan semula pesakit atau menghentikan penyebaran penyakit.

### Pengurusan ujian klinikal

Ketoksikan dalam ujian klinikal adalah kebimbangan utama kepada pembuat ubat. Berapa banyak ketoksikan yang boleh diterima? Dalam kajian ini, menganalisis pelbagai kaedah ujian klinikal membawa kepada pembangunan pendekatan baharu untuk meramalkan kemungkinan hasil ujian klinikal. Khususnya, mereka dapat menggunakan random forest untuk menghasilkan [pengklasifikasi](../../4-Classification/README.md) yang mampu membezakan antara kumpulan ubat.
[Rujukan](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Pengurusan kemasukan semula hospital

Penjagaan hospital adalah mahal, terutamanya apabila pesakit perlu dimasukkan semula. Kertas kerja ini membincangkan sebuah syarikat yang menggunakan ML untuk meramalkan potensi kemasukan semula menggunakan algoritma [pengelompokan](../../5-Clustering/README.md). Kelompok ini membantu penganalisis untuk "menemui kumpulan kemasukan semula yang mungkin berkongsi punca yang sama".
[Rujukan](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Pengurusan penyakit

Pandemik baru-baru ini telah menonjolkan cara pembelajaran mesin boleh membantu menghentikan penyebaran penyakit. Dalam artikel ini, anda akan mengenali penggunaan ARIMA, lengkung logistik, regresi linear, dan SARIMA. "Kerja ini adalah usaha untuk mengira kadar penyebaran virus ini dan dengan itu meramalkan kematian, pemulihan, dan kes yang disahkan, supaya ia dapat membantu kita untuk bersedia dengan lebih baik dan bertahan."
[Rujukan](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologi dan Teknologi Hijau

Alam semula jadi dan ekologi terdiri daripada banyak sistem sensitif di mana interaksi antara haiwan dan alam semula jadi menjadi fokus. Adalah penting untuk dapat mengukur sistem ini dengan tepat dan bertindak dengan sewajarnya jika sesuatu berlaku, seperti kebakaran hutan atau penurunan populasi haiwan.

### Pengurusan hutan

Anda telah mempelajari tentang [Pembelajaran Pengukuhan](../../8-Reinforcement/README.md) dalam pelajaran sebelumnya. Ia boleh sangat berguna apabila cuba meramalkan corak dalam alam semula jadi. Khususnya, ia boleh digunakan untuk menjejaki masalah ekologi seperti kebakaran hutan dan penyebaran spesies invasif. Di Kanada, sekumpulan penyelidik menggunakan Pembelajaran Pengukuhan untuk membina model dinamik kebakaran hutan daripada imej satelit. Menggunakan "proses penyebaran spatial (SSP)" yang inovatif, mereka membayangkan kebakaran hutan sebagai "agen di mana-mana sel dalam landskap." "Set tindakan yang boleh diambil oleh kebakaran dari lokasi pada bila-bila masa termasuk menyebar ke utara, selatan, timur, atau barat atau tidak menyebar.

Pendekatan ini membalikkan persediaan RL biasa kerana dinamik Proses Keputusan Markov (MDP) yang sepadan adalah fungsi yang diketahui untuk penyebaran kebakaran segera." Baca lebih lanjut tentang algoritma klasik yang digunakan oleh kumpulan ini di pautan di bawah.
[Rujukan](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Pengesanan pergerakan haiwan

Walaupun pembelajaran mendalam telah mencipta revolusi dalam menjejaki pergerakan haiwan secara visual (anda boleh membina [penjejak beruang kutub](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) anda sendiri di sini), ML klasik masih mempunyai tempat dalam tugas ini.

Sensor untuk menjejaki pergerakan haiwan ternakan dan IoT menggunakan jenis pemprosesan visual ini, tetapi teknik ML yang lebih asas berguna untuk memproses data awal. Sebagai contoh, dalam kertas kerja ini, postur kambing biri-biri dipantau dan dianalisis menggunakan pelbagai algoritma pengklasifikasi. Anda mungkin mengenali lengkung ROC pada halaman 335.
[Rujukan](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Pengurusan Tenaga

Dalam pelajaran kami tentang [ramalan siri masa](../../7-TimeSeries/README.md), kami menyebut konsep meter parkir pintar untuk menjana pendapatan bagi sebuah bandar berdasarkan pemahaman tentang penawaran dan permintaan. Artikel ini membincangkan secara terperinci bagaimana pengelompokan, regresi, dan ramalan siri masa digabungkan untuk membantu meramalkan penggunaan tenaga masa depan di Ireland, berdasarkan meter pintar.
[Rujukan](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Insurans

Sektor insurans adalah satu lagi sektor yang menggunakan ML untuk membina dan mengoptimumkan model kewangan dan aktuari yang berdaya maju.

### Pengurusan Volatiliti

MetLife, penyedia insurans hayat, terbuka dengan cara mereka menganalisis dan mengurangkan volatiliti dalam model kewangan mereka. Dalam artikel ini, anda akan melihat visualisasi klasifikasi binari dan ordinal. Anda juga akan menemui visualisasi ramalan.
[Rujukan](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Seni, Budaya, dan Kesusasteraan

Dalam seni, contohnya dalam kewartawanan, terdapat banyak masalah menarik. Mengesan berita palsu adalah masalah besar kerana ia telah terbukti mempengaruhi pendapat orang ramai dan bahkan menjatuhkan demokrasi. Muzium juga boleh mendapat manfaat daripada menggunakan ML dalam segala-galanya daripada mencari hubungan antara artifak kepada perancangan sumber.

### Pengesanan berita palsu

Mengesan berita palsu telah menjadi permainan kucing dan tikus dalam media hari ini. Dalam artikel ini, penyelidik mencadangkan bahawa sistem yang menggabungkan beberapa teknik ML yang telah kami pelajari boleh diuji dan model terbaik digunakan: "Sistem ini berdasarkan pemprosesan bahasa semula jadi untuk mengekstrak ciri daripada data dan kemudian ciri-ciri ini digunakan untuk latihan pengklasifikasi pembelajaran mesin seperti Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD), dan Logistic Regression (LR)."
[Rujukan](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Artikel ini menunjukkan bagaimana menggabungkan pelbagai domain ML boleh menghasilkan hasil menarik yang boleh membantu menghentikan penyebaran berita palsu dan mencipta kerosakan sebenar; dalam kes ini, dorongan adalah penyebaran khabar angin tentang rawatan COVID yang mencetuskan keganasan massa.

### ML Muzium

Muzium berada di ambang revolusi AI di mana katalog dan pendigitalan koleksi serta mencari hubungan antara artifak menjadi lebih mudah apabila teknologi maju. Projek seperti [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) membantu membuka misteri koleksi yang tidak dapat diakses seperti Arkib Vatican. Tetapi, aspek perniagaan muzium mendapat manfaat daripada model ML juga.

Sebagai contoh, Art Institute of Chicago membina model untuk meramalkan apa yang penonton minati dan bila mereka akan menghadiri pameran. Matlamatnya adalah untuk mencipta pengalaman pelawat yang diperibadikan dan dioptimumkan setiap kali pengguna melawat muzium. "Semasa tahun fiskal 2017, model itu meramalkan kehadiran dan kemasukan dengan ketepatan dalam 1 peratus, kata Andrew Simnick, naib presiden kanan di Art Institute."
[Rujukan](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Pemasaran

### Segmentasi pelanggan

Strategi pemasaran yang paling berkesan menyasarkan pelanggan dengan cara yang berbeza berdasarkan pelbagai kumpulan. Dalam artikel ini, penggunaan algoritma pengelompokan dibincangkan untuk menyokong pemasaran yang berbeza. Pemasaran yang berbeza membantu syarikat meningkatkan pengiktirafan jenama, mencapai lebih ramai pelanggan, dan menjana lebih banyak wang.
[Rujukan](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Cabaran

Kenal pasti sektor lain yang mendapat manfaat daripada beberapa teknik yang anda pelajari dalam kurikulum ini, dan temui bagaimana ia menggunakan ML.
## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Pasukan sains data Wayfair mempunyai beberapa video menarik tentang bagaimana mereka menggunakan ML di syarikat mereka. Ia berbaloi untuk [dilihat](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tugasan

[Perburuan ML](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.