<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T19:28:22+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "ms"
}
-->
# Postscript: Penyahpepijatan Model dalam Pembelajaran Mesin menggunakan komponen papan pemuka AI Bertanggungjawab

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Pengenalan

Pembelajaran mesin memberi kesan kepada kehidupan harian kita. AI semakin digunakan dalam beberapa sistem paling penting yang mempengaruhi kita sebagai individu dan masyarakat, seperti penjagaan kesihatan, kewangan, pendidikan, dan pekerjaan. Sebagai contoh, sistem dan model terlibat dalam tugas membuat keputusan harian, seperti diagnosis penjagaan kesihatan atau pengesanan penipuan. Akibatnya, kemajuan AI bersama dengan penerimaan yang dipercepatkan telah bertemu dengan jangkaan masyarakat yang berkembang dan peraturan yang semakin ketat sebagai tindak balas. Kita sering melihat kawasan di mana sistem AI terus gagal memenuhi jangkaan; mereka mendedahkan cabaran baharu; dan kerajaan mula mengawal selia penyelesaian AI. Oleh itu, adalah penting untuk menganalisis model ini bagi menyediakan hasil yang adil, boleh dipercayai, inklusif, telus, dan bertanggungjawab untuk semua orang.

Dalam kurikulum ini, kita akan melihat alat praktikal yang boleh digunakan untuk menilai sama ada model mempunyai isu AI yang bertanggungjawab. Teknik penyahpepijatan pembelajaran mesin tradisional cenderung berdasarkan pengiraan kuantitatif seperti ketepatan agregat atau purata kehilangan ralat. Bayangkan apa yang boleh berlaku apabila data yang anda gunakan untuk membina model ini kekurangan demografi tertentu, seperti bangsa, jantina, pandangan politik, agama, atau mewakili demografi tersebut secara tidak seimbang. Bagaimana pula apabila output model ditafsirkan untuk memihak kepada beberapa demografi? Ini boleh memperkenalkan perwakilan berlebihan atau kurang bagi kumpulan ciri sensitif ini, yang mengakibatkan isu keadilan, keterangkuman, atau kebolehpercayaan daripada model. Faktor lain ialah model pembelajaran mesin dianggap sebagai kotak hitam, yang menjadikannya sukar untuk memahami dan menjelaskan apa yang mendorong ramalan model. Semua ini adalah cabaran yang dihadapi oleh saintis data dan pembangun AI apabila mereka tidak mempunyai alat yang mencukupi untuk menyahpepijat dan menilai keadilan atau kebolehpercayaan model.

Dalam pelajaran ini, anda akan belajar tentang penyahpepijatan model anda menggunakan:

- **Analisis Ralat**: mengenal pasti di mana dalam taburan data anda model mempunyai kadar ralat yang tinggi.
- **Gambaran Keseluruhan Model**: melakukan analisis perbandingan merentasi pelbagai kohort data untuk menemui ketidaksamaan dalam metrik prestasi model anda.
- **Analisis Data**: menyiasat di mana mungkin terdapat perwakilan berlebihan atau kurang dalam data anda yang boleh mempengaruhi model anda untuk memihak kepada satu demografi data berbanding yang lain.
- **Kepentingan Ciri**: memahami ciri-ciri yang mendorong ramalan model anda pada tahap global atau tempatan.

## Prasyarat

Sebagai prasyarat, sila semak [Alat AI Bertanggungjawab untuk pembangun](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif tentang Alat AI Bertanggungjawab](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Analisis Ralat

Metrik prestasi model tradisional yang digunakan untuk mengukur ketepatan kebanyakannya adalah pengiraan berdasarkan ramalan betul vs salah. Sebagai contoh, menentukan bahawa model adalah tepat 89% masa dengan kehilangan ralat sebanyak 0.001 boleh dianggap sebagai prestasi yang baik. Ralat selalunya tidak diedarkan secara seragam dalam set data asas anda. Anda mungkin mendapat skor ketepatan model 89% tetapi mendapati bahawa terdapat kawasan berbeza dalam data anda di mana model gagal 42% masa. Akibat daripada corak kegagalan ini dengan kumpulan data tertentu boleh membawa kepada isu keadilan atau kebolehpercayaan. Adalah penting untuk memahami kawasan di mana model berprestasi baik atau tidak. Kawasan data di mana terdapat sejumlah besar ketidaktepatan dalam model anda mungkin ternyata menjadi demografi data yang penting.

![Menganalisis dan menyahpepijat ralat model](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Komponen Analisis Ralat pada papan pemuka RAI menggambarkan bagaimana kegagalan model diedarkan merentasi pelbagai kohort dengan visualisasi pokok. Ini berguna untuk mengenal pasti ciri atau kawasan di mana terdapat kadar ralat yang tinggi dengan set data anda. Dengan melihat dari mana kebanyakan ketidaktepatan model datang, anda boleh mula menyiasat punca utama. Anda juga boleh mencipta kohort data untuk melakukan analisis. Kohort data ini membantu dalam proses penyahpepijatan untuk menentukan mengapa prestasi model baik dalam satu kohort tetapi salah dalam yang lain.

![Analisis Ralat](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Penunjuk visual pada peta pokok membantu dalam mencari kawasan masalah dengan lebih cepat. Sebagai contoh, semakin gelap warna merah pada nod pokok, semakin tinggi kadar ralat.

Peta haba adalah satu lagi fungsi visualisasi yang boleh digunakan oleh pengguna dalam menyiasat kadar ralat menggunakan satu atau dua ciri untuk mencari penyumbang kepada ralat model merentasi keseluruhan set data atau kohort.

![Peta Haba Analisis Ralat](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Gunakan analisis ralat apabila anda perlu:

* Mendapatkan pemahaman mendalam tentang bagaimana kegagalan model diedarkan merentasi set data dan merentasi beberapa dimensi input dan ciri.
* Memecahkan metrik prestasi agregat untuk secara automatik menemui kohort yang salah bagi memaklumkan langkah mitigasi yang disasarkan.

## Gambaran Keseluruhan Model

Menilai prestasi model pembelajaran mesin memerlukan pemahaman holistik tentang tingkah lakunya. Ini boleh dicapai dengan menyemak lebih daripada satu metrik seperti kadar ralat, ketepatan, ingatan, ketepatan, atau MAE (Mean Absolute Error) untuk mencari ketidaksamaan antara metrik prestasi. Satu metrik prestasi mungkin kelihatan hebat, tetapi ketidaktepatan boleh didedahkan dalam metrik lain. Selain itu, membandingkan metrik untuk ketidaksamaan merentasi keseluruhan set data atau kohort membantu menjelaskan di mana model berprestasi baik atau tidak. Ini amat penting dalam melihat prestasi model di kalangan ciri sensitif vs tidak sensitif (contohnya, bangsa pesakit, jantina, atau umur) untuk mendedahkan potensi ketidakadilan yang mungkin ada pada model. Sebagai contoh, mendapati bahawa model lebih salah dalam kohort yang mempunyai ciri sensitif boleh mendedahkan potensi ketidakadilan yang mungkin ada pada model.

Komponen Gambaran Keseluruhan Model pada papan pemuka RAI membantu bukan sahaja dalam menganalisis metrik prestasi perwakilan data dalam kohort, tetapi ia memberi pengguna keupayaan untuk membandingkan tingkah laku model merentasi kohort yang berbeza.

![Kohort dataset - gambaran keseluruhan model dalam papan pemuka RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Fungsi analisis berasaskan ciri komponen membolehkan pengguna memperincikan subkumpulan data dalam ciri tertentu untuk mengenal pasti anomali pada tahap yang lebih terperinci. Sebagai contoh, papan pemuka mempunyai kecerdasan terbina dalam untuk secara automatik menjana kohort untuk ciri yang dipilih oleh pengguna (contohnya, *"time_in_hospital < 3"* atau *"time_in_hospital >= 7"*). Ini membolehkan pengguna mengasingkan ciri tertentu daripada kumpulan data yang lebih besar untuk melihat sama ada ia adalah pengaruh utama kepada hasil yang salah model.

![Kohort ciri - gambaran keseluruhan model dalam papan pemuka RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Komponen Gambaran Keseluruhan Model menyokong dua kelas metrik ketidaksamaan:

**Ketidaksamaan dalam prestasi model**: Set metrik ini mengira ketidaksamaan (perbezaan) dalam nilai metrik prestasi yang dipilih merentasi subkumpulan data. Berikut adalah beberapa contoh:

* Ketidaksamaan dalam kadar ketepatan
* Ketidaksamaan dalam kadar ralat
* Ketidaksamaan dalam ketepatan
* Ketidaksamaan dalam ingatan
* Ketidaksamaan dalam mean absolute error (MAE)

**Ketidaksamaan dalam kadar pemilihan**: Metrik ini mengandungi perbezaan dalam kadar pemilihan (ramalan yang menguntungkan) di kalangan subkumpulan. Contoh ini ialah ketidaksamaan dalam kadar kelulusan pinjaman. Kadar pemilihan bermaksud pecahan titik data dalam setiap kelas yang diklasifikasikan sebagai 1 (dalam klasifikasi binari) atau taburan nilai ramalan (dalam regresi).

## Analisis Data

> "Jika anda menyeksa data cukup lama, ia akan mengaku apa sahaja" - Ronald Coase

Kenyataan ini kedengaran ekstrem, tetapi benar bahawa data boleh dimanipulasi untuk menyokong sebarang kesimpulan. Manipulasi sedemikian kadangkala boleh berlaku secara tidak sengaja. Sebagai manusia, kita semua mempunyai bias, dan sering sukar untuk mengetahui secara sedar apabila anda memperkenalkan bias dalam data. Menjamin keadilan dalam AI dan pembelajaran mesin kekal sebagai cabaran yang kompleks.

Data adalah titik buta besar untuk metrik prestasi model tradisional. Anda mungkin mempunyai skor ketepatan yang tinggi, tetapi ini tidak selalu mencerminkan bias data asas yang mungkin ada dalam set data anda. Sebagai contoh, jika set data pekerja mempunyai 27% wanita dalam jawatan eksekutif di sebuah syarikat dan 73% lelaki pada tahap yang sama, model AI pengiklanan pekerjaan yang dilatih pada data ini mungkin menyasarkan kebanyakan penonton lelaki untuk jawatan pekerjaan peringkat kanan. Mempunyai ketidakseimbangan ini dalam data mempengaruhi ramalan model untuk memihak kepada satu jantina. Ini mendedahkan isu keadilan di mana terdapat bias jantina dalam model AI.

Komponen Analisis Data pada papan pemuka RAI membantu mengenal pasti kawasan di mana terdapat perwakilan berlebihan dan kurang dalam set data. Ia membantu pengguna mendiagnosis punca utama ralat dan isu keadilan yang diperkenalkan daripada ketidakseimbangan data atau kekurangan perwakilan kumpulan data tertentu. Ini memberi pengguna keupayaan untuk memvisualisasikan set data berdasarkan hasil yang diramalkan dan sebenar, kumpulan ralat, dan ciri tertentu. Kadangkala penemuan kumpulan data yang kurang diwakili juga boleh mendedahkan bahawa model tidak belajar dengan baik, maka ketidaktepatan yang tinggi. Mempunyai model yang mempunyai bias data bukan sahaja isu keadilan tetapi menunjukkan bahawa model tidak inklusif atau boleh dipercayai.

![Komponen Analisis Data pada Papan Pemuka RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Gunakan analisis data apabila anda perlu:

* Meneroka statistik set data anda dengan memilih penapis berbeza untuk membahagikan data anda kepada dimensi berbeza (juga dikenali sebagai kohort).
* Memahami taburan set data anda merentasi kohort dan kumpulan ciri yang berbeza.
* Menentukan sama ada penemuan anda berkaitan dengan keadilan, analisis ralat, dan sebab-akibat (yang diperoleh daripada komponen papan pemuka lain) adalah hasil daripada taburan set data anda.
* Memutuskan di kawasan mana untuk mengumpul lebih banyak data bagi mengurangkan ralat yang datang daripada isu perwakilan, bunyi label, bunyi ciri, bias label, dan faktor serupa.

## Kebolehinterpretasian Model

Model pembelajaran mesin cenderung menjadi kotak hitam. Memahami ciri data utama yang mendorong ramalan model boleh menjadi mencabar. Adalah penting untuk menyediakan ketelusan tentang mengapa model membuat ramalan tertentu. Sebagai contoh, jika sistem AI meramalkan bahawa pesakit diabetes berisiko dimasukkan semula ke hospital dalam masa kurang daripada 30 hari, ia sepatutnya dapat menyediakan data sokongan yang membawa kepada ramalannya. Mempunyai penunjuk data sokongan membawa ketelusan untuk membantu doktor atau hospital membuat keputusan yang tepat. Selain itu, dapat menjelaskan mengapa model membuat ramalan untuk pesakit individu membolehkan akauntabiliti dengan peraturan kesihatan. Apabila anda menggunakan model pembelajaran mesin dengan cara yang memberi kesan kepada kehidupan manusia, adalah penting untuk memahami dan menjelaskan apa yang mempengaruhi tingkah laku model. Kebolehjelasan dan kebolehinterpretasian model membantu menjawab soalan dalam senario seperti:

* Penyahpepijatan model: Mengapa model saya membuat kesilapan ini? Bagaimana saya boleh memperbaiki model saya?
* Kerjasama manusia-AI: Bagaimana saya boleh memahami dan mempercayai keputusan model?
* Pematuhan peraturan: Adakah model saya memenuhi keperluan undang-undang?

Komponen Kepentingan Ciri pada papan pemuka RAI membantu anda menyahpepijat dan mendapatkan pemahaman yang komprehensif tentang bagaimana model membuat ramalan. Ia juga merupakan alat yang berguna untuk profesional pembelajaran mesin dan pembuat keputusan untuk menjelaskan dan menunjukkan bukti ciri yang mempengaruhi tingkah laku model untuk pematuhan peraturan. Seterusnya, pengguna boleh meneroka penjelasan global dan tempatan untuk mengesahkan ciri mana yang mendorong ramalan model. Penjelasan global menyenaraikan ciri utama yang mempengaruhi ramalan keseluruhan model. Penjelasan tempatan memaparkan ciri mana yang membawa kepada ramalan model untuk kes individu. Keupayaan untuk menilai penjelasan tempatan juga berguna dalam menyahpepijat atau mengaudit kes tertentu untuk lebih memahami dan mentafsir mengapa model membuat ramalan yang tepat atau tidak tepat.

![Komponen Kepentingan Ciri pada papan pemuka RAI](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Penjelasan global: Sebagai contoh, ciri apa yang mempengaruhi tingkah laku keseluruhan model kemasukan semula hospital diabetes?
* Penjelasan tempatan: Sebagai contoh, mengapa pesakit diabetes berumur lebih 60 tahun dengan kemasukan hospital sebelumnya diramalkan untuk dimasukkan semula atau tidak dimasukkan semula dalam masa 30 hari ke hospital?

Dalam proses penyahpepijatan untuk memeriksa prestasi model merentasi kohort yang berbeza, Kepentingan Ciri menunjukkan tahap pengaruh ciri merentasi kohort. Ia membantu mendedahkan anomali apabila membandingkan tahap pengaruh ciri dalam mendorong ramalan yang salah model. Komponen Kepentingan Ciri boleh menunjukkan nilai mana dalam ciri yang mempengaruhi hasil model secara positif atau negatif. Sebagai contoh, jika model membuat ramalan yang tidak tepat, komponen ini memberi anda keupayaan untuk memperincikan dan mengenal pasti ciri atau nilai ciri yang mendorong ramalan tersebut. Tahap perincian ini bukan sahaja membantu dalam penyahpepijatan tetapi menyediakan ketelusan dan akauntabiliti dalam situasi pengauditan. Akhirnya, komponen ini boleh membantu anda mengenal pasti isu keadilan. Sebagai ilustrasi, jika ciri sensitif seperti etnik atau jantina sangat mempengaruhi ramalan model, ini boleh menjadi tanda bias kaum atau jantina dalam model.

![Kepentingan ciri](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Gunakan kebolehinterpretasian apabila anda perlu:

* Menentukan sejauh mana ramalan sistem AI anda boleh dipercayai dengan memahami ciri mana yang paling penting untuk ramalan tersebut.
* Mendekati penyahpepijatan model anda dengan memahaminya terlebih dahulu dan mengenal pasti sama ada model menggunakan ciri yang sihat atau hanya korelasi palsu.
* Mendedahkan potensi sumber ketidakadilan dengan memahami sama ada model membuat ramalan berdasarkan ciri sensitif atau ciri yang sangat berkorelasi dengannya.
* Membina kepercayaan pengguna terhadap keputusan model anda dengan menjana penjelasan tempatan untuk menggambarkan hasilnya.
* Melengkapkan audit peraturan sistem AI untuk mengesahkan model dan memantau kesan keputusan model terhadap manusia.

## Kesimpulan

Semua komponen papan pemuka RAI adalah alat praktikal untuk membantu anda membina model pembelajaran mesin yang kurang berbahaya dan lebih dipercayai oleh masyarakat. Ia meningkatkan pencegahan ancaman kepada hak asasi manusia; mendiskriminasi atau mengecualikan kumpulan tertentu daripada peluang hidup; dan risiko kecederaan fizikal atau psikologi. Ia juga membantu membina kepercayaan terhadap keputusan model anda dengan menjana penjelasan tempatan untuk menggambarkan hasilnya. Beberapa potensi bahaya boleh diklasifikasikan sebagai:

- **Peruntukan**, jika jantina atau etnik sebagai contoh lebih disukai berbanding yang lain.
- **Kualiti perkhidmatan**. Jika anda melatih data untuk satu senario tertentu tetapi realitinya jauh lebih kompleks, ia membawa kepada perkhidmatan yang kurang berprestasi.
- **Stereotaip**. Mengaitkan kumpulan tertentu dengan atribut yang telah ditetapkan.
- **Pencemaran nama baik**. Mengkritik dan melabel sesuatu atau seseorang secara tidak adil.
- **Perwakilan berlebihan atau kurang**. Idea di sini adalah bahawa kumpulan tertentu tidak kelihatan dalam profesion tertentu, dan sebarang perkhidmatan atau fungsi yang terus mempromosikan perkara ini menyumbang kepada kemudaratan.

### Papan Pemuka Azure RAI

[Papan Pemuka Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) dibina menggunakan alat sumber terbuka yang dikembangkan oleh institusi akademik dan organisasi terkemuka termasuk Microsoft, yang sangat membantu saintis data dan pembangun AI untuk memahami tingkah laku model, mengenal pasti dan mengurangkan isu yang tidak diingini daripada model AI.

- Ketahui cara menggunakan komponen yang berbeza dengan melihat [dokumentasi papan pemuka RAI.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Lihat beberapa [notebook contoh papan pemuka RAI](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) untuk menyelesaikan senario AI yang lebih bertanggungjawab dalam Azure Machine Learning.

---
## 🚀 Cabaran

Untuk mengelakkan bias statistik atau data daripada diperkenalkan sejak awal, kita harus:

- mempunyai kepelbagaian latar belakang dan perspektif di kalangan orang yang bekerja pada sistem
- melabur dalam set data yang mencerminkan kepelbagaian masyarakat kita
- membangunkan kaedah yang lebih baik untuk mengesan dan membetulkan bias apabila ia berlaku

Fikirkan tentang senario kehidupan sebenar di mana ketidakadilan jelas dalam pembinaan dan penggunaan model. Apa lagi yang perlu kita pertimbangkan?

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)
## Ulasan & Kajian Kendiri

Dalam pelajaran ini, anda telah mempelajari beberapa alat praktikal untuk mengintegrasikan AI yang bertanggungjawab dalam pembelajaran mesin.

Tonton bengkel ini untuk mendalami topik:

- Papan Pemuka AI Bertanggungjawab: Pusat sehenti untuk mengoperasikan RAI dalam amalan oleh Besmira Nushi dan Mehrnoosh Sameki

[![Papan Pemuka AI Bertanggungjawab: Pusat sehenti untuk mengoperasikan RAI dalam amalan](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Papan Pemuka AI Bertanggungjawab: Pusat sehenti untuk mengoperasikan RAI dalam amalan")

> 🎥 Klik imej di atas untuk video: Papan Pemuka AI Bertanggungjawab: Pusat sehenti untuk mengoperasikan RAI dalam amalan oleh Besmira Nushi dan Mehrnoosh Sameki

Rujuk bahan berikut untuk mengetahui lebih lanjut tentang AI yang bertanggungjawab dan cara membina model yang lebih dipercayai:

- Alat papan pemuka RAI Microsoft untuk menyelesaikan masalah model ML: [Sumber alat AI bertanggungjawab](https://aka.ms/rai-dashboard)

- Terokai kit alat AI bertanggungjawab: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Pusat sumber RAI Microsoft: [Sumber AI Bertanggungjawab – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Kumpulan penyelidikan FATE Microsoft: [FATE: Keadilan, Akauntabiliti, Ketelusan, dan Etika dalam AI - Penyelidikan Microsoft](https://www.microsoft.com/research/theme/fate/)

## Tugasan

[Terokai papan pemuka RAI](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat penting, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.