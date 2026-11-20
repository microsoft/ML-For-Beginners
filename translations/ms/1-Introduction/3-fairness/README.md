# Membangun solusi Pembelajaran Mesin dengan AI yang bertanggungjawab
 
![Ringkasan AI bertanggungjawab dalam Pembelajaran Mesin dalam sketchnote](../../../../translated_images/ms/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Kuiz pra-ceramah](https://ff-quizzes.netlify.app/en/ml/)
 
## Pengenalan

Dalam kurikulum ini, anda akan mula menemui bagaimana pembelajaran mesin boleh dan sedang memberi impak kepada kehidupan seharian kita. Malah sekarang, sistem dan model terlibat dalam tugas membuat keputusan harian, seperti diagnosis penjagaan kesihatan, kelulusan pinjaman atau pengesanan penipuan. Oleh itu, adalah penting supaya model-model ini berfungsi dengan baik untuk memberikan hasil yang boleh dipercayai. Sama seperti mana-mana aplikasi perisian, sistem AI mungkin gagal mencapai jangkaan atau memberikan hasil yang tidak diingini. Oleh itu, adalah penting untuk dapat memahami dan menjelaskan tingkah laku model AI.

Bayangkan apa yang boleh berlaku apabila data yang anda gunakan untuk membina model ini tidak mempunyai sesetengah demografi, seperti bangsa, jantina, pandangan politik, agama, atau mewakili demografi tersebut secara tidak seimbang. Bagaimana pula apabila output model ditafsirkan untuk memihak kepada sesetengah demografi? Apakah akibatnya kepada aplikasi? Selain itu, apa yang berlaku apabila model mempunyai hasil negatif dan membahayakan orang? Siapakah yang bertanggungjawab ke atas tingkah laku sistem AI? Ini adalah beberapa soalan yang akan kita terokai dalam kurikulum ini.

Dalam pelajaran ini, anda akan:

- Meningkatkan kesedaran anda tentang kepentingan keadilan dalam pembelajaran mesin dan bahaya berkaitan keadilan.
- Membiasakan diri dengan amalan meneroka outlier dan senario luar biasa untuk memastikan kebolehpercayaan dan keselamatan.
- Memahami keperluan untuk memberdayakan semua orang dengan mereka bentuk sistem inklusif.
- Meneroka betapa pentingnya melindungi privasi dan keselamatan data serta orang.
- Melihat betapa pentingnya pendekatan kotak kaca untuk menjelaskan tingkah laku model AI.
- Berhati-hati tentang bagaimana akauntabiliti adalah penting untuk membina kepercayaan dalam sistem AI.

## Prasyarat

Sebagai prasyarat, sila ambil Jejak Pembelajaran "Prinsip AI Bertanggungjawab" dan tonton video di bawah mengenai topik ini:

Ketahui lebih lanjut tentang AI Bertanggungjawab dengan mengikuti [Jalur Pembelajaran](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Pendekatan Microsoft terhadap AI Bertanggungjawab](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Pendekatan Microsoft terhadap AI Bertanggungjawab")

> 🎥 Klik imej di atas untuk video: Pendekatan Microsoft terhadap AI Bertanggungjawab

## Keadilan

Sistem AI harus melayan semua orang dengan adil dan mengelak memberi kesan berbeza kepada kumpulan orang yang serupa. Sebagai contoh, apabila sistem AI memberikan panduan mengenai rawatan perubatan, permohonan pinjaman, atau pekerjaan, ia harus membuat cadangan yang sama kepada semua orang dengan simptom, keadaan kewangan, atau kelayakan profesional yang serupa. Setiap kita sebagai manusia membawa bias yang diwarisi yang mempengaruhi keputusan dan tindakan kita. Bias ini boleh jelas dalam data yang kita gunakan untuk melatih sistem AI. Manipulasi sebegini kadangkala berlaku tanpa sengaja. Selalunya sukar untuk sedar bila anda memperkenalkan bias dalam data.

**“Ketidakadilan”** merangkumi kesan negatif, atau “bahaya”, kepada kumpulan orang, seperti yang ditakrifkan dari segi bangsa, jantina, umur, atau status ketidakupayaan. Bahaya berkaitan keadilan utama boleh diklasifikasikan sebagai:

- **Peruntukan**, jika jantina atau etnik contohnya diutamakan berbanding yang lain.
- **Kualiti perkhidmatan**. Jika anda melatih data untuk satu senario tertentu tetapi realitinya jauh lebih kompleks, ia membawa kepada perkhidmatan yang berkualiti rendah. Contohnya, dispenser sabun tangan yang tidak dapat mengesan orang dengan kulit gelap. [Rujukan](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Penghinaan**. Mengkritik dan melabel sesuatu atau seseorang secara tidak adil. Sebagai contoh, teknologi pelabelan imej terkenal kerana melabel imej orang berkulit gelap sebagai gorila.
- **Representasi berlebihan atau kurang**. Idea ialah bahawa kumpulan tertentu tidak dilihat dalam profesion tertentu, dan mana-mana perkhidmatan atau fungsi yang terus mempromosikan itu menyumbang kepada bahaya.
- **Pengstereotipan**. Mengaitkan satu kumpulan dengan atribut yang telah ditetapkan. Sebagai contoh, sistem terjemahan bahasa antara Inggeris dan Turki mungkin mempunyai ketidaktepatan disebabkan perkataan dengan asosiasi stereotaip kepada jantina.

![terjemahan ke Turki](../../../../translated_images/ms/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> terjemahan ke Turki

![terjemahan kembali ke Inggeris](../../../../translated_images/ms/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> terjemahan kembali ke Inggeris

Apabila mereka bentuk dan menguji sistem AI, kita perlu memastikan bahawa AI adalah adil dan tidak diprogramkan untuk membuat keputusan berat sebelah atau diskriminasi, yang juga dilarang kepada manusia. Menjamin keadilan dalam AI dan pembelajaran mesin masih merupakan cabaran sosio-teknikal yang kompleks.

### Kebolehpercayaan dan keselamatan

Untuk membina kepercayaan, sistem AI perlu boleh dipercayai, selamat, dan konsisten dalam keadaan biasa dan tidak dijangka. Penting untuk mengetahui bagaimana sistem AI akan berkelakuan dalam pelbagai situasi, terutamanya apabila mereka adalah outlier. Apabila membina solusi AI, perlu ada tumpuan yang ketara bagaimana untuk menangani pelbagai keadaan yang bakal ditemui oleh solusi AI tersebut. Contohnya, kereta memandu sendiri perlu meletakkan keselamatan orang sebagai keutamaan utama. Oleh itu, AI yang menggerakkan kereta perlu mempertimbangkan semua senario yang mungkin dihadapi oleh kereta seperti malam, ribut petir atau ribut salji, kanak-kanak berlari melintasi jalan, haiwan peliharaan, pembinaan jalan dan lain-lain. Seberapa baik sistem AI dapat menangani pelbagai keadaan dengan boleh dipercayai dan selamat mencerminkan tahap ramalan yang dipertimbangkan oleh saintis data atau pembangun AI semasa mereka bentuk atau menguji sistem tersebut.

> [🎥 Klik di sini untuk video: Kebolehpercayaan dan keselamatan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusif

Sistem AI harus direka untuk melibatkan dan memberdayakan semua orang. Apabila mereka bentuk dan melaksanakan sistem AI, saintis data dan pembangun AI mengenal pasti dan mengatasi halangan dalam sistem yang boleh secara tidak sengaja mengecualikan orang. Contohnya, terdapat 1 bilion orang dengan kecacatan di seluruh dunia. Dengan kemajuan AI, mereka dapat mengakses pelbagai maklumat dan peluang dengan lebih mudah dalam kehidupan seharian mereka. Dengan mengatasi halangan tersebut, ia mencipta peluang untuk berinovasi dan membangunkan produk AI dengan pengalaman yang lebih baik yang memberi manfaat kepada semua orang.

> [🎥 Klik di sini untuk video: Inklusif dalam AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Keselamatan dan privasi

Sistem AI harus selamat dan menghormati privasi orang. Orang kurang percaya pada sistem yang meletakkan privasi, maklumat, atau nyawa mereka berisiko. Apabila melatih model pembelajaran mesin, kita bergantung pada data untuk menghasilkan keputusan terbaik. Dalam melakukannya, asal usul data dan integritinya mesti dipertimbangkan. Contohnya, adakah data dihantar oleh pengguna atau tersedia secara umum? Seterusnya, semasa bekerja dengan data, adalah penting untuk membangunkan sistem AI yang dapat melindungi maklumat sulit dan menahan serangan. Apabila AI menjadi lebih meluas, melindungi privasi dan mengamankan maklumat penting peribadi dan perniagaan menjadi semakin kritikal dan kompleks. Isu privasi dan keselamatan data memerlukan perhatian yang sangat khusus kerana akses kepada data adalah penting untuk sistem AI membuat ramalan dan keputusan yang tepat dan bermaklumat mengenai orang.

> [🎥 Klik di sini untuk video: Keselamatan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sebagai sebuah industri, kita telah membuat kemajuan ketara dalam Privasi & keselamatan, yang secara signifikan dipacu oleh peraturan seperti GDPR (Peraturan Perlindungan Data Umum).
- Namun dengan sistem AI, kita mesti mengakui ketegangan antara keperluan untuk lebih banyak data peribadi untuk menjadikan sistem lebih peribadi dan berkesan – dan privasi.
- Sama seperti apabila komputer bersambung dengan internet diperkenalkan, kita juga menyaksikan peningkatan besar dalam bilangan isu keselamatan yang berkaitan dengan AI.
- Pada masa yang sama, kita telah melihat AI digunakan untuk meningkatkan keselamatan. Contohnya, kebanyakan pengimbas anti-virus moden hari ini dikuasakan oleh heuristik AI.
- Kita perlu memastikan bahawa proses Sains Data kita bersatu padu secara harmoni dengan amalan privasi dan keselamatan terkini.


### Ketelusan
Sistem AI harus boleh difahami. Bahagian penting dalam ketelusan adalah menerangkan tingkah laku sistem AI dan komponennya. Meningkatkan pemahaman tentang sistem AI memerlukan agar pihak berkepentingan memahami bagaimana dan mengapa sistem berfungsi supaya mereka boleh mengenalpasti isu prestasi, kebimbangan keselamatan dan privasi, bias, amalan pengecualian, atau hasil yang tidak diingini. Kami juga percaya bahawa mereka yang menggunakan sistem AI harus jujur dan terbuka tentang bila, mengapa, dan bagaimana mereka memilih untuk menggunakannya. Serta had sistem yang mereka gunakan. Contohnya, jika sebuah bank menggunakan sistem AI untuk menyokong keputusan pinjaman pengguna, adalah penting untuk meneliti hasil dan memahami data mana yang mempengaruhi cadangan sistem. Kerajaan mula mengawal AI merentasi industri, jadi saintis data dan organisasi mesti menjelaskan sama ada sistem AI memenuhi keperluan peraturan, terutamanya apabila terdapat hasil yang tidak diingini.

> [🎥 Klik di sini untuk video: Ketelusan dalam AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Kerana sistem AI sangat kompleks, sukar untuk memahami bagaimana ia berfungsi dan mentafsirkan hasilnya.
- Kekurangan pemahaman ini mempengaruhi cara sistem ini diurus, dioperasikan, dan didokumentasikan.
- Kekurangan pemahaman ini yang lebih penting mempengaruhi keputusan yang dibuat menggunakan hasil yang dihasilkan oleh sistem ini.

### Akauntabiliti
 
Orang yang mereka bentuk dan melaksanakan sistem AI mesti bertanggungjawab terhadap bagaimana sistem mereka beroperasi. Keperluan akauntabiliti adalah sangat penting terutamanya dengan teknologi penggunaan sensitif seperti pengecaman wajah. Baru-baru ini, terdapat permintaan yang semakin meningkat untuk teknologi pengecaman wajah, terutamanya daripada organisasi penguatkuasaan undang-undang yang melihat potensi teknologi tersebut dalam kegunaan seperti mencari kanak-kanak hilang. Walau bagaimanapun, teknologi ini berpotensi digunakan oleh kerajaan untuk meletakkan kebebasan asas warganegara mereka berisiko dengan, sebagai contoh, membenarkan pengawasan berterusan individu tertentu. Oleh itu, saintis data dan organisasi perlu bertanggungjawab terhadap bagaimana sistem AI mereka memberi impak kepada individu atau masyarakat.

[![Penyelidik AI Terkemuka Memberi Amaran tentang Pengawasan Massa Melalui Pengecaman Wajah](../../../../translated_images/ms/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Pendekatan Microsoft terhadap AI Bertanggungjawab")

> 🎥 Klik imej di atas untuk video: Amaran pengawasan massa melalui pengecaman wajah

Akhirnya salah satu soalan terbesar untuk generasi kita, sebagai generasi pertama yang membawa AI ke masyarakat, adalah bagaimana memastikan komputer akan kekal bertanggungjawab kepada manusia dan bagaimana memastikan orang yang mereka bentuk komputer kekal bertanggungjawab kepada semua orang lain.

## Penilaian impak

Sebelum melatih model pembelajaran mesin, adalah penting untuk menjalankan penilaian impak untuk memahami tujuan sistem AI; apa kegunaan yang dimaksudkan; di mana ia akan digunakan; dan siapa yang akan berinteraksi dengan sistem tersebut. Ini membantu penilai atau penguji sistem untuk mengetahui faktor yang perlu diambil kira apabila mengenal pasti potensi risiko dan akibat yang dijangkakan.

Berikut adalah kawasan tumpuan apabila menjalankan penilaian impak:

* **Kesan buruk kepada individu**. Menyedari sebarang sekatan atau keperluan, penggunaan tidak disokong atau sebarang had yang diketahui yang menghalang prestasi sistem adalah penting untuk memastikan sistem tidak digunakan dengan cara yang boleh membahayakan individu.
* **Keperluan data**. Memperoleh pemahaman tentang bagaimana dan di mana sistem akan menggunakan data membenarkan penilai meneroka sebarang keperluan data yang perlu diambil perhatian (contohnya, peraturan data GDPR atau HIPAA). Selain itu, periksa sama ada sumber atau kuantiti data mencukupi untuk latihan.
* **Ringkasan impak**. Kumpul senarai potensi bahaya yang boleh timbul daripada menggunakan sistem. Sepanjang kitaran hayat ML, semak sama ada isu yang dikenal pasti telah dikurangkan atau diatasi.
* **Matlamat yang terpakai** untuk setiap enam prinsip teras. Nilai sama ada matlamat dari setiap prinsip tercapai dan jika terdapat sebarang kekurangan.


## Pengesan ralat dengan AI bertanggungjawab

Serupa dengan mengesan ralat aplikasi perisian, mengesan ralat sistem AI adalah proses perlu untuk mengenal pasti dan menyelesaikan isu dalam sistem. Terdapat banyak faktor yang boleh menyebabkan model tidak berprestasi seperti yang dijangka atau bertanggungjawab. Kebanyakan metrik prestasi model tradisional adalah agregat kuantitatif bagi prestasi model, yang tidak mencukupi untuk menganalisis bagaimana model melanggar prinsip AI bertanggungjawab. Selain itu, model pembelajaran mesin adalah kotak hitam yang menyukarkan untuk memahami apa yang mendorong hasilnya atau memberikan penjelasan apabila ia melakukan kesilapan. Kemudian dalam kursus ini, kita akan belajar bagaimana menggunakan papan pemuka AI Bertanggungjawab untuk membantu mengesan ralat sistem AI. Papan pemuka memberikan alat holistik untuk saintis data dan pembangun AI untuk melaksanakan:

* **Analisis ralat**. Untuk mengenal pasti taburan ralat model yang boleh menjejaskan keadilan atau kebolehpercayaan sistem.
* **Gambaran keseluruhan model**. Untuk menemui di mana terdapat perbezaan dalam prestasi model merentasi kohort data.
* **Analisis data**. Untuk memahami taburan data dan mengenal pasti sebarang bias potensi dalam data yang boleh membawa kepada isu keadilan, inklusif, dan kebolehpercayaan.
* **Interpretabiliti model**. Untuk memahami apa yang mempengaruhi atau memberi kesan kepada ramalan model. Ini membantu menjelaskan tingkah laku model, yang penting untuk ketelusan dan akauntabiliti.


## 🚀 Cabaran
 
Untuk mengelakkan bahaya daripada diperkenalkan pada mulanya, kita harus:

- mempunyai kepelbagaian latar belakang dan perspektif dalam kalangan orang yang bekerja pada sistem
- melabur dalam set data yang mencerminkan kepelbagaian masyarakat kita
- membangunkan kaedah yang lebih baik sepanjang kitaran hayat pembelajaran mesin untuk mengesan dan membetulkan AI yang tidak bertanggungjawab apabila ia berlaku

Fikirkan tentang senario kehidupan sebenar di mana ketidakpercayaan model jelas semasa pembinaan model dan penggunaannya. Apa lagi yang harus kita pertimbangkan?

## [Kuiz pasca-ceramah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri
 
Dalam pelajaran ini, anda telah mempelajari beberapa asas konsep keadilan dan ketidakadilan dalam pembelajaran mesin.
 
Tonton bengkel ini untuk menyelami lebih dalam topik:

- Dalam mengejar AI bertanggungjawab: Membawa prinsip ke amalan oleh Besmira Nushi, Mehrnoosh Sameki dan Amit Sharma

[![Kotak Alat AI Bertanggungjawab: Rangka kerja sumber terbuka untuk membina AI bertanggungjawab](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Kotak Alat RAI: Rangka kerja sumber terbuka untuk membina AI bertanggungjawab")

> 🎥 Klik imej di atas untuk video: Kotak Alat RAI: Rangka kerja sumber terbuka untuk membina AI bertanggungjawab oleh Besmira Nushi, Mehrnoosh Sameki, dan Amit Sharma

Juga, baca:

- Pusat sumber RAI Microsoft: [Sumber AI Bertanggungjawab – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kumpulan penyelidikan FATE Microsoft: [FATE: Keadilan, Akauntabiliti, Ketelusan, dan Etika dalam AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

Kotak Alat RAI:

- [Repositori GitHub Kotak Alat AI Bertanggungjawab](https://github.com/microsoft/responsible-ai-toolbox)

Baca mengenai alat Azure Machine Learning untuk memastikan keadilan:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tugasan

[Terokai Kotak Alat RAI](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk ketepatan, sila ambil maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang sahih. Untuk maklumat penting, terjemahan oleh manusia profesional adalah disyorkan. Kami tidak bertanggungjawab terhadap sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->