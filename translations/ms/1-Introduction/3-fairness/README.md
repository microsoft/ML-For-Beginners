<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T19:33:25+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "ms"
}
-->
# Membina Penyelesaian Pembelajaran Mesin dengan AI yang Bertanggungjawab

![Ringkasan AI yang bertanggungjawab dalam Pembelajaran Mesin dalam bentuk sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuiz Pra-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Pengenalan

Dalam kurikulum ini, anda akan mula meneroka bagaimana pembelajaran mesin boleh dan sedang memberi kesan kepada kehidupan seharian kita. Pada masa kini, sistem dan model terlibat dalam tugas membuat keputusan harian seperti diagnosis kesihatan, kelulusan pinjaman, atau pengesanan penipuan. Oleh itu, adalah penting bahawa model ini berfungsi dengan baik untuk memberikan hasil yang boleh dipercayai. Sama seperti mana-mana aplikasi perisian, sistem AI mungkin tidak memenuhi jangkaan atau menghasilkan hasil yang tidak diingini. Itulah sebabnya penting untuk memahami dan menjelaskan tingkah laku model AI.

Bayangkan apa yang boleh berlaku apabila data yang anda gunakan untuk membina model ini kekurangan demografi tertentu seperti bangsa, jantina, pandangan politik, agama, atau secara tidak seimbang mewakili demografi tersebut. Bagaimana pula apabila output model ditafsirkan untuk memihak kepada sesetengah demografi? Apakah akibatnya kepada aplikasi tersebut? Selain itu, apa yang berlaku apabila model menghasilkan hasil yang buruk dan membahayakan orang? Siapa yang bertanggungjawab terhadap tingkah laku sistem AI? Ini adalah beberapa soalan yang akan kita terokai dalam kurikulum ini.

Dalam pelajaran ini, anda akan:

- Meningkatkan kesedaran tentang kepentingan keadilan dalam pembelajaran mesin dan bahaya berkaitan keadilan.
- Mengenali amalan meneroka outlier dan senario luar biasa untuk memastikan kebolehpercayaan dan keselamatan.
- Memahami keperluan untuk memperkasakan semua orang dengan mereka bentuk sistem yang inklusif.
- Meneroka betapa pentingnya melindungi privasi dan keselamatan data serta individu.
- Melihat kepentingan pendekatan kotak kaca untuk menjelaskan tingkah laku model AI.
- Berhati-hati tentang bagaimana akauntabiliti adalah penting untuk membina kepercayaan dalam sistem AI.

## Prasyarat

Sebagai prasyarat, sila ambil Laluan Pembelajaran "Prinsip AI yang Bertanggungjawab" dan tonton video di bawah mengenai topik ini:

Ketahui lebih lanjut tentang AI yang Bertanggungjawab dengan mengikuti [Laluan Pembelajaran](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Pendekatan Microsoft terhadap AI yang Bertanggungjawab](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Pendekatan Microsoft terhadap AI yang Bertanggungjawab")

> 🎥 Klik imej di atas untuk video: Pendekatan Microsoft terhadap AI yang Bertanggungjawab

## Keadilan

Sistem AI harus melayan semua orang dengan adil dan mengelakkan memberi kesan yang berbeza kepada kumpulan orang yang serupa. Sebagai contoh, apabila sistem AI memberikan panduan tentang rawatan perubatan, permohonan pinjaman, atau pekerjaan, ia harus memberikan cadangan yang sama kepada semua orang dengan simptom, keadaan kewangan, atau kelayakan profesional yang serupa. Setiap daripada kita sebagai manusia membawa bias yang diwarisi yang mempengaruhi keputusan dan tindakan kita. Bias ini boleh kelihatan dalam data yang kita gunakan untuk melatih sistem AI. Manipulasi sedemikian kadang-kadang berlaku secara tidak sengaja. Selalunya sukar untuk secara sedar mengetahui bila anda memperkenalkan bias dalam data.

**“Ketidakadilan”** merangkumi kesan negatif, atau “bahaya”, kepada sekumpulan orang, seperti yang ditakrifkan dalam istilah bangsa, jantina, umur, atau status kecacatan. Bahaya berkaitan keadilan utama boleh diklasifikasikan sebagai:

- **Peruntukan**, jika jantina atau etnik, sebagai contoh, lebih diutamakan berbanding yang lain.
- **Kualiti perkhidmatan**. Jika anda melatih data untuk satu senario tertentu tetapi realiti jauh lebih kompleks, ia membawa kepada perkhidmatan yang kurang berprestasi. Sebagai contoh, dispenser sabun tangan yang tidak dapat mengesan orang dengan kulit gelap. [Rujukan](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Pencemaran nama baik**. Mengkritik dan melabel sesuatu atau seseorang secara tidak adil. Sebagai contoh, teknologi pelabelan imej secara terkenal melabelkan imej orang berkulit gelap sebagai gorila.
- **Perwakilan berlebihan atau kurang**. Idea bahawa kumpulan tertentu tidak dilihat dalam profesion tertentu, dan mana-mana perkhidmatan atau fungsi yang terus mempromosikan itu menyumbang kepada bahaya.
- **Stereotaip**. Mengaitkan kumpulan tertentu dengan atribut yang telah ditetapkan. Sebagai contoh, sistem terjemahan bahasa antara Bahasa Inggeris dan Bahasa Turki mungkin mempunyai ketidaktepatan disebabkan oleh perkataan dengan kaitan stereotaip kepada jantina.

![terjemahan ke Bahasa Turki](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> terjemahan ke Bahasa Turki

![terjemahan kembali ke Bahasa Inggeris](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> terjemahan kembali ke Bahasa Inggeris

Apabila mereka bentuk dan menguji sistem AI, kita perlu memastikan bahawa AI adalah adil dan tidak diprogramkan untuk membuat keputusan yang berat sebelah atau diskriminasi, yang juga dilarang untuk dibuat oleh manusia. Menjamin keadilan dalam AI dan pembelajaran mesin kekal sebagai cabaran sosio-teknikal yang kompleks.

### Kebolehpercayaan dan keselamatan

Untuk membina kepercayaan, sistem AI perlu boleh dipercayai, selamat, dan konsisten di bawah keadaan biasa dan luar jangka. Adalah penting untuk mengetahui bagaimana sistem AI akan berkelakuan dalam pelbagai situasi, terutamanya apabila ia adalah outlier. Apabila membina penyelesaian AI, perlu ada fokus yang besar pada cara menangani pelbagai keadaan yang mungkin dihadapi oleh penyelesaian AI. Sebagai contoh, kereta pandu sendiri perlu meletakkan keselamatan orang sebagai keutamaan utama. Akibatnya, AI yang menggerakkan kereta perlu mempertimbangkan semua senario yang mungkin dihadapi oleh kereta seperti waktu malam, ribut petir atau salji, kanak-kanak berlari melintasi jalan, haiwan peliharaan, pembinaan jalan, dan sebagainya. Sejauh mana sistem AI boleh menangani pelbagai keadaan dengan kebolehpercayaan dan keselamatan mencerminkan tahap jangkaan yang dipertimbangkan oleh saintis data atau pembangun AI semasa mereka bentuk atau menguji sistem.

> [🎥 Klik di sini untuk video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusiviti

Sistem AI harus direka untuk melibatkan dan memperkasakan semua orang. Apabila mereka bentuk dan melaksanakan sistem AI, saintis data dan pembangun AI mengenal pasti dan menangani halangan yang berpotensi dalam sistem yang boleh secara tidak sengaja mengecualikan orang. Sebagai contoh, terdapat 1 bilion orang kurang upaya di seluruh dunia. Dengan kemajuan AI, mereka boleh mengakses pelbagai maklumat dan peluang dengan lebih mudah dalam kehidupan seharian mereka. Dengan menangani halangan, ia mencipta peluang untuk berinovasi dan membangunkan produk AI dengan pengalaman yang lebih baik yang memberi manfaat kepada semua orang.

> [🎥 Klik di sini untuk video: inklusiviti dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Keselamatan dan privasi

Sistem AI harus selamat dan menghormati privasi orang. Orang kurang percaya kepada sistem yang meletakkan privasi, maklumat, atau nyawa mereka dalam risiko. Apabila melatih model pembelajaran mesin, kita bergantung pada data untuk menghasilkan hasil terbaik. Dalam melakukannya, asal usul data dan integriti mesti dipertimbangkan. Sebagai contoh, adakah data dihantar oleh pengguna atau tersedia secara umum? Seterusnya, semasa bekerja dengan data, adalah penting untuk membangunkan sistem AI yang boleh melindungi maklumat sulit dan menahan serangan. Apabila AI menjadi lebih meluas, melindungi privasi dan mengamankan maklumat peribadi dan perniagaan yang penting menjadi semakin kritikal dan kompleks. Isu privasi dan keselamatan data memerlukan perhatian yang sangat teliti untuk AI kerana akses kepada data adalah penting untuk sistem AI membuat ramalan dan keputusan yang tepat dan bermaklumat tentang orang.

> [🎥 Klik di sini untuk video: keselamatan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sebagai industri, kita telah membuat kemajuan yang ketara dalam privasi & keselamatan, yang didorong secara signifikan oleh peraturan seperti GDPR (Peraturan Perlindungan Data Umum).
- Namun dengan sistem AI, kita mesti mengakui ketegangan antara keperluan untuk lebih banyak data peribadi untuk menjadikan sistem lebih peribadi dan berkesan – dan privasi.
- Sama seperti kelahiran komputer yang disambungkan dengan internet, kita juga melihat peningkatan besar dalam jumlah isu keselamatan yang berkaitan dengan AI.
- Pada masa yang sama, kita telah melihat AI digunakan untuk meningkatkan keselamatan. Sebagai contoh, kebanyakan pengimbas anti-virus moden didorong oleh heuristik AI hari ini.
- Kita perlu memastikan bahawa proses Sains Data kita bergabung secara harmoni dengan amalan privasi dan keselamatan terkini.

### Ketelusan

Sistem AI harus dapat difahami. Bahagian penting ketelusan adalah menjelaskan tingkah laku sistem AI dan komponennya. Meningkatkan pemahaman tentang sistem AI memerlukan pihak berkepentingan memahami bagaimana dan mengapa ia berfungsi supaya mereka dapat mengenal pasti isu prestasi yang berpotensi, kebimbangan keselamatan dan privasi, bias, amalan pengecualian, atau hasil yang tidak diingini. Kami juga percaya bahawa mereka yang menggunakan sistem AI harus jujur dan terbuka tentang bila, mengapa, dan bagaimana mereka memilih untuk menggunakannya. Serta batasan sistem yang mereka gunakan. Sebagai contoh, jika sebuah bank menggunakan sistem AI untuk menyokong keputusan pemberian pinjaman kepada pengguna, adalah penting untuk memeriksa hasilnya dan memahami data mana yang mempengaruhi cadangan sistem. Kerajaan mula mengawal selia AI merentasi industri, jadi saintis data dan organisasi mesti menjelaskan jika sistem AI memenuhi keperluan peraturan, terutamanya apabila terdapat hasil yang tidak diingini.

> [🎥 Klik di sini untuk video: ketelusan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Oleh kerana sistem AI sangat kompleks, sukar untuk memahami bagaimana ia berfungsi dan mentafsir hasilnya.
- Kekurangan pemahaman ini mempengaruhi cara sistem ini diuruskan, dioperasikan, dan didokumentasikan.
- Kekurangan pemahaman ini lebih penting lagi mempengaruhi keputusan yang dibuat menggunakan hasil yang dihasilkan oleh sistem ini.

### Akauntabiliti

Orang yang mereka bentuk dan menggunakan sistem AI mesti bertanggungjawab terhadap cara sistem mereka beroperasi. Keperluan untuk akauntabiliti amat penting dengan teknologi penggunaan sensitif seperti pengecaman wajah. Baru-baru ini, terdapat permintaan yang semakin meningkat untuk teknologi pengecaman wajah, terutamanya daripada organisasi penguatkuasaan undang-undang yang melihat potensi teknologi dalam kegunaan seperti mencari kanak-kanak yang hilang. Walau bagaimanapun, teknologi ini berpotensi digunakan oleh kerajaan untuk meletakkan kebebasan asas rakyat mereka dalam risiko dengan, sebagai contoh, membolehkan pengawasan berterusan terhadap individu tertentu. Oleh itu, saintis data dan organisasi perlu bertanggungjawab terhadap bagaimana sistem AI mereka memberi kesan kepada individu atau masyarakat.

[![Penyelidik AI Terkemuka Memberi Amaran tentang Pengawasan Massa Melalui Pengecaman Wajah](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Pendekatan Microsoft terhadap AI yang Bertanggungjawab")

> 🎥 Klik imej di atas untuk video: Amaran tentang Pengawasan Massa Melalui Pengecaman Wajah

Akhirnya, salah satu soalan terbesar untuk generasi kita, sebagai generasi pertama yang membawa AI kepada masyarakat, adalah bagaimana memastikan komputer akan kekal bertanggungjawab kepada manusia dan bagaimana memastikan orang yang mereka bentuk komputer kekal bertanggungjawab kepada semua orang.

## Penilaian Impak

Sebelum melatih model pembelajaran mesin, adalah penting untuk menjalankan penilaian impak untuk memahami tujuan sistem AI; apa kegunaan yang dimaksudkan; di mana ia akan digunakan; dan siapa yang akan berinteraksi dengan sistem tersebut. Ini berguna untuk pengulas atau penguji yang menilai sistem untuk mengetahui faktor apa yang perlu dipertimbangkan semasa mengenal pasti risiko yang berpotensi dan akibat yang dijangkakan.

Berikut adalah kawasan fokus semasa menjalankan penilaian impak:

* **Kesan buruk terhadap individu**. Menyedari sebarang sekatan atau keperluan, penggunaan yang tidak disokong atau sebarang batasan yang diketahui yang menghalang prestasi sistem adalah penting untuk memastikan sistem tidak digunakan dengan cara yang boleh menyebabkan bahaya kepada individu.
* **Keperluan data**. Memahami bagaimana dan di mana sistem akan menggunakan data membolehkan pengulas meneroka sebarang keperluan data yang perlu anda perhatikan (contohnya, peraturan data GDPR atau HIPPA). Selain itu, periksa sama ada sumber atau kuantiti data adalah mencukupi untuk latihan.
* **Ringkasan impak**. Kumpulkan senarai potensi bahaya yang boleh timbul daripada menggunakan sistem. Sepanjang kitaran hayat ML, semak sama ada isu yang dikenal pasti telah dikurangkan atau ditangani.
* **Matlamat yang boleh digunakan** untuk setiap enam prinsip teras. Nilai sama ada matlamat daripada setiap prinsip telah dicapai dan jika terdapat sebarang jurang.

## Debugging dengan AI yang Bertanggungjawab

Sama seperti debugging aplikasi perisian, debugging sistem AI adalah proses yang diperlukan untuk mengenal pasti dan menyelesaikan isu dalam sistem. Terdapat banyak faktor yang boleh mempengaruhi model tidak berprestasi seperti yang diharapkan atau bertanggungjawab. Kebanyakan metrik prestasi model tradisional adalah agregat kuantitatif prestasi model, yang tidak mencukupi untuk menganalisis bagaimana model melanggar prinsip AI yang bertanggungjawab. Tambahan pula, model pembelajaran mesin adalah kotak hitam yang menjadikannya sukar untuk memahami apa yang mendorong hasilnya atau memberikan penjelasan apabila ia membuat kesilapan. Kemudian dalam kursus ini, kita akan belajar bagaimana menggunakan papan pemuka AI yang Bertanggungjawab untuk membantu debugging sistem AI. Papan pemuka menyediakan alat holistik untuk saintis data dan pembangun AI untuk melaksanakan:

* **Analisis ralat**. Untuk mengenal pasti pengedaran ralat model yang boleh menjejaskan keadilan atau kebolehpercayaan sistem.
* **Gambaran keseluruhan model**. Untuk menemui di mana terdapat perbezaan dalam prestasi model merentasi kohort data.
* **Analisis data**. Untuk memahami pengedaran data dan mengenal pasti sebarang potensi bias dalam data yang boleh membawa kepada isu keadilan, inklusiviti, dan kebolehpercayaan.
* **Kebolehinterpretasian model**. Untuk memahami apa yang mempengaruhi atau mempengaruhi ramalan model. Ini membantu dalam menjelaskan tingkah laku model, yang penting untuk ketelusan dan akauntabiliti.

## 🚀 Cabaran

Untuk mengelakkan bahaya daripada diperkenalkan sejak awal, kita harus:

- mempunyai kepelbagaian latar belakang dan perspektif di kalangan orang yang bekerja pada sistem
- melabur dalam set data yang mencerminkan kepelbagaian masyarakat kita
- membangunkan kaedah yang lebih baik sepanjang kitaran hayat pembelajaran mesin untuk mengesan dan membetulkan AI yang bertanggungjawab apabila ia berlaku

Fikirkan tentang senario kehidupan sebenar di mana ketidakpercayaan model jelas dalam pembinaan dan penggunaan model. Apa lagi yang perlu kita pertimbangkan?

## [Kuiz Pasca-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Dalam pelajaran ini, anda telah mempelajari beberapa asas konsep keadilan dan ketidakadilan dalam pembelajaran mesin.
Tonton bengkel ini untuk mendalami topik:

- Dalam usaha AI yang bertanggungjawab: Membawa prinsip ke amalan oleh Besmira Nushi, Mehrnoosh Sameki dan Amit Sharma

[![Responsible AI Toolbox: Kerangka sumber terbuka untuk membina AI yang bertanggungjawab](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Kerangka sumber terbuka untuk membina AI yang bertanggungjawab")

> 🎥 Klik imej di atas untuk video: RAI Toolbox: Kerangka sumber terbuka untuk membina AI yang bertanggungjawab oleh Besmira Nushi, Mehrnoosh Sameki, dan Amit Sharma

Juga, baca:

- Pusat sumber RAI Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kumpulan penyelidikan FATE Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repositori GitHub Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Baca tentang alat Azure Machine Learning untuk memastikan keadilan:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tugasan

[Terokai RAI Toolbox](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.