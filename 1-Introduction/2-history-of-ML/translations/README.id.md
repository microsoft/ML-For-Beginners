# Sejarah Machine Learning

![Ringkasan dari Sejarah Machine Learning dalam sebuah catatan sketsa](../../../sketchnotes/ml-history.png)
> Catatan sketsa oleh [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz Pra-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/3/)

Dalam pelajaran ini, kita akan membahas tonggak utama dalam sejarah Machine Learning dan Artificial Intelligence. 

Sejarah Artifical Intelligence, AI, sebagai bidang terkait dengan sejarah Machine Learning, karena algoritma dan kemajuan komputasi yang mendukung ML dimasukkan ke dalam pengembangan AI. Penting untuk diingat bahwa, meski bidang-bidang ini sebagai bidang-bidang penelitian yang berbeda mulai terbentuk pada 1950-an, [algoritmik, statistik, matematik, komputasi dan penemuan teknis](https://wikipedia.org/wiki/Timeline_of_machine_learning) penting sudah ada sebelumnya, dan saling tumpang tindih di era ini. Faktanya, orang-orang telah memikirkan pertanyaan-pertanyaan ini selama [ratusan tahun](https://wikipedia.org/wiki/History_of_artificial_intelligence): artikel ini membahas dasar-dasar intelektual historis dari gagasan 'mesin yang berpikir'. 

## Penemuan penting

- 1763, 1812 [Bayes Theorem](https://wikipedia.org/wiki/Bayes%27_theorem) dan para pendahulu. Teorema ini dan penerapannya mendasari inferensi, mendeskripsikan kemungkinan suatu peristiwa terjadi berdasarkan pengetahuan sebelumnya. 
- 1805 [Least Square Theory](https://wikipedia.org/wiki/Least_squares) oleh matematikawan Perancis Adrien-Marie Legendre. Teori ini yang akan kamu pelajari di unit Regresi, ini membantu dalam *data fitting*. 
- 1913 [Markov Chains](https://wikipedia.org/wiki/Markov_chain) dinamai dengan nama matematikawan Rusia, Andrey Markov, digunakan untuk mendeskripsikan sebuah urutan dari kejadian-kejadian yang mungkin terjadi berdasarkan kondisi sebelumnya.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) adalah sebuah tipe dari *linear classifier* yang ditemukan oleh psikolog Amerika, Frank Rosenblatt, yang mendasari kemajuan dalam *Deep Learning*.
- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) adalah sebuah algoritma yang pada awalnya didesain untuk memetakan rute. Dalam konteks ML, ini digunakan untuk mendeteksi berbagai pola.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) digunakan untuk melatih [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network). 
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) adalah *artificial neural networks* yang berasal dari *feedforward neural networks* yang membuat grafik sementara. 

✅ Lakukan sebuah riset kecil. Tanggal berapa lagi yang merupakan tanggal penting dalam sejarah ML dan AI?
## 1950: Mesin yang berpikir

Alan Turing, merupakan orang luar biasa yang terpilih oleh [publik di tahun 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) sebagai ilmuwan terhebat di abad 20, diberikan penghargaan karena membantu membuat fondasi dari sebuah konsep 'mesin yang bisa berpikir', Dia berjuang menghadapi orang-orang yang menentangnya dan keperluannya sendiri untuk bukti empiris dari konsep ini dengan membuat [Turing Test](https://www.bbc.com/news/technology-18475646), yang mana akan kamu jelajahi di pelajaran NLP kami.

## 1956: Proyek Riset Musim Panas Dartmouth

"Proyek Riset Musim Panas Dartmouth pada *artificial intelligence* merupakan sebuah acara penemuan untuk *artificial intelligence* sebagai sebuah bidang," dan dari sinilah istilah '*artificial intelligence*' diciptakan ([sumber](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Setiap aspek pembelajaran atau fitur kecerdasan lainnya pada prinsipnya dapat dideskripsikan dengan sangat tepat sehingga sebuah mesin dapat dibuat untuk mensimulasikannya. 

Ketua peneliti, profesor matematika John McCarthy, berharap "untuk meneruskan dasar dari dugaan bahwa setiap aspek pembelajaran atau fitur kecerdasan lainnya pada prinsipnya dapat dideskripsikan dengan sangat tepat sehingga mesin dapat dibuat untuk mensimulasikannya." Marvin Minsky, seorang tokoh terkenal di bidang ini juga termasuk sebagai peserta penelitian.

Workshop ini dipuji karena telah memprakarsai dan mendorong beberapa diskusi termasuk "munculnya metode simbolik, sistem yang berfokus pada domain terbatas (sistem pakar awal), dan sistem deduktif versus sistem induktif." ([sumber](https://wikipedia.org/wiki/Dartmouth_workshop)).

## 1956 - 1974: "Tahun-tahun Emas"

Dari tahun 1950-an hingga pertengahan 70-an, optimisme memuncak dengan harapan bahwa AI dapat memecahkan banyak masalah. Pada tahun 1967, Marvin Minsky dengan yakin menyatakan bahwa "Dalam satu generasi ... masalah menciptakan '*artificial intelligence*' akan terpecahkan secara substansial." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Penelitian *natural language processing* berkembang, pencarian disempurnakan dan dibuat lebih *powerful*, dan konsep '*micro-worlds*' diciptakan, di mana tugas-tugas sederhana diselesaikan menggunakan instruksi bahasa sederhana. 

Penelitian didanai dengan baik oleh lembaga pemerintah, banyak kemajuan dibuat dalam komputasi dan algoritma, dan prototipe mesin cerdas dibangun. Beberapa mesin tersebut antara lain: 

* [Shakey the robot](https://wikipedia.org/wiki/Shakey_the_robot), yang bisa bermanuver dan memutuskan bagaimana melakukan tugas-tugas secara 'cerdas'. 

    ![Shakey, an intelligent robot](../images/shakey.jpg)
    > Shakey pada 1972

* Eliza, sebuah 'chatterbot' awal, dapat mengobrol dengan orang-orang dan bertindak sebagai 'terapis' primitif. Kamu akan belajar lebih banyak tentang Eliza dalam pelajaran NLP. 

    ![Eliza, a bot](../images/eliza.png)
    > Sebuah versi dari Eliza, sebuah *chatbot*

* "Blocks world" adalah contoh sebuah *micro-world* dimana balok dapat ditumpuk dan diurutkan, dan pengujian eksperimen mesin pengajaran untuk membuat keputusan dapat dilakukan. Kemajuan yang dibuat dengan *library-library* seperti [SHRDLU](https://wikipedia.org/wiki/SHRDLU) membantu mendorong kemajuan pemrosesan bahasa.

    [![blocks world dengan SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world dengan SHRDLU")
    
    > 🎥 Klik gambar diatas untuk menonton video: Blocks world with SHRDLU

## 1974 - 1980: "Musim Dingin AI"

Pada pertengahan 1970-an, semakin jelas bahwa kompleksitas pembuatan 'mesin cerdas' telah diremehkan dan janjinya, mengingat kekuatan komputasi yang tersedia, telah dilebih-lebihkan. Pendanaan telah habis dan kepercayaan dalam bidang ini menurun. Beberapa masalah yang memengaruhi kepercayaan diri termasuk: 

- **Keterbatasan**. Kekuatan komputasi terlalu terbatas.
- **Ledakan kombinatorial**. Jumlah parameter yang perlu dilatih bertambah secara eksponensial karena lebih banyak hal yang diminta dari komputer, tanpa evolusi paralel dari kekuatan dan kemampuan komputasi. 
- **Kekurangan data**. Adanya kekurangan data yang menghalangi proses pengujian, pengembangan, dan penyempurnaan algoritma.
- **Apakah kita menanyakan pertanyaan yang tepat?**. Pertanyaan-pertanyaan yang diajukan pun mulai dipertanyakan kembali. Para peneliti mulai melontarkan kritik tentang pendekatan mereka 
  - Tes Turing mulai dipertanyakan, di antara ide-ide lain, dari 'teori ruang Cina' yang mengemukakan bahwa, "memprogram komputer digital mungkin membuatnya tampak memahami bahasa tetapi tidak dapat menghasilkan pemahaman yang sebenarnya." ([sumber](https://plato.stanford.edu/entries/chinese-room/))
  - Tantangan etika ketika memperkenalkan kecerdasan buatan seperti si "terapis" ELIZA ke dalam masyarakat. 

Pada saat yang sama, berbagai aliran pemikiran AI mulai terbentuk. Sebuah dikotomi didirikan antara praktik ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). Lab _Scruffy_ mengubah program selama berjam-jam sampai mendapat hasil yang diinginkan. Lab _Neat_ "berfokus pada logika dan penyelesaian masalah formal". ELIZA dan SHRDLU adalah sistem _scruffy_ yang terkenal. Pada tahun 1980-an, karena perkembangan permintaan untuk membuat sistem ML yang dapat direproduksi, pendekatan _neat_ secara bertahap menjadi yang terdepan karena hasilnya lebih dapat dijelaskan.

## 1980s Sistem Pakar

Seiring berkembangnya bidang ini, manfaatnya bagi bisnis menjadi lebih jelas, dan begitu pula dengan menjamurnya 'sistem pakar' pada tahun 1980-an. "Sistem pakar adalah salah satu bentuk perangkat lunak artificial intelligence (AI) pertama yang benar-benar sukses."  ([sumber](https://wikipedia.org/wiki/Expert_system)).

Tipe sistem ini sebenarnya adalah _hybrid_, sebagian terdiri dari mesin aturan yang mendefinisikan kebutuhan bisnis, dan mesin inferensi yang memanfaatkan sistem aturan untuk menyimpulkan fakta baru. 

Pada era ini juga terlihat adanya peningkatan perhatian pada jaringan saraf. 

## 1987 - 1993: AI 'Chill'

Perkembangan perangkat keras sistem pakar terspesialisasi memiliki efek yang tidak menguntungkan karena menjadi terlalu terspesialiasasi. Munculnya komputer pribadi juga bersaing dengan sistem yang besar, terspesialisasi, dan terpusat ini. Demokratisasi komputasi telah dimulai, dan pada akhirnya membuka jalan untuk ledakan modern dari *big data*. 

## 1993 - 2011

Pada zaman ini memperlihatkan era baru bagi ML dan AI untuk dapat menyelesaikan beberapa masalah yang sebelumnya disebabkan oleh kurangnya data dan daya komputasi. Jumlah data mulai meningkat dengan cepat dan tersedia secara luas, terlepas dari baik dan buruknya, terutama dengan munculnya *smartphone* sekitar tahun 2007. Daya komputasi berkembang secara eksponensial, dan algoritma juga berkembang saat itu. Bidang ini mulai mengalami kedewasaan karena hari-hari yang tidak beraturan di masa lalu mulai terbentuk menjadi disiplin yang sebenarnya. 

## Sekarang

Saat ini, *machine learning* dan AI hampir ada di setiap bagian dari kehidupan kita. Era ini menuntut pemahaman yang cermat tentang risiko dan efek potensi dari berbagai algoritma yang ada pada kehidupan manusia. Seperti yang telah dinyatakan oleh Brad Smith dari Microsoft, "Teknologi informasi mengangkat isu-isu yang menjadi inti dari perlindungan hak asasi manusia yang mendasar seperti privasi dan kebebasan berekspresi. Masalah-masalah ini meningkatkan tanggung jawab bagi perusahaan teknologi yang menciptakan produk-produk ini. Dalam pandangan kami, mereka juga menyerukan peraturan pemerintah yang bijaksana dan untuk pengembangan norma-norma seputar penggunaan yang wajar" ([sumber](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

Kita masih belum tahu apa yang akan terjadi di masa depan, tetapi penting untuk memahami sistem komputer dan perangkat lunak serta algoritma yang dijalankannya. Kami berharap kurikulum ini akan membantu kamu untuk mendapatkan pemahaman yang lebih baik sehingga kamu dapat memutuskan sendiri. 

[![Sejarah Deep Learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Sejarah Deep Learning")
> 🎥 Klik gambar diatas untuk menonton video: Yann LeCun mendiskusikan sejarah dari Deep Learning dalam pelajaran ini

---
## 🚀Tantangan

Gali salah satu momen bersejarah ini dan pelajari lebih lanjut tentang orang-orang di baliknya. Ada karakter yang menarik, dan tidak ada penemuan ilmiah yang pernah dibuat dalam kekosongan budaya. Apa yang kamu temukan? 

## [Quiz Pasca-Pelajaran](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/4/)

## Ulasan & Belajar Mandiri

Berikut adalah item untuk ditonton dan didengarkan: 

[Podcast dimana Amy Boyd mendiskusikan evolusi dari AI](http://runasradio.com/Shows/Show/739)

[![Sejarah AI oleh Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Sejarah AI oleh Amy Boyd")

## Tugas

[Membuat sebuah *timeline*](assignment.id.md)
