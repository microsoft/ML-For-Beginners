# Pengenalan kepada pemprosesan bahasa semulajadi

Pelajaran ini merangkumi sejarah ringkas dan konsep penting dalam *pemprosesan bahasa semulajadi*, satu cabang daripada *linguistik komputasi*.

## [Kuiz pra-ceramah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Pengenalan

NLP, seperti yang biasa dikenali, adalah salah satu bidang yang paling terkenal di mana pembelajaran mesin telah digunakan dan diterapkan dalam perisian pengeluaran.

✅ Bolehkah anda memikirkan perisian yang anda gunakan setiap hari yang mungkin mempunyai beberapa NLP di dalamnya? Bagaimana dengan program pemprosesan kata atau aplikasi mudah alih yang anda gunakan secara kerap?

Anda akan belajar tentang:

- **Idea tentang bahasa**. Bagaimana bahasa berkembang dan apakah bidang kajian utama.
- **Definisi dan konsep**. Anda juga akan belajar definisi dan konsep tentang bagaimana komputer memproses teks, termasuk penguraian, tatabahasa, dan mengenal pasti kata nama dan kata kerja. Terdapat beberapa tugas pengekodan dalam pelajaran ini, dan beberapa konsep penting diperkenalkan yang anda akan belajar untuk kod kemudian dalam pelajaran seterusnya.

## Linguistik komputasi

Linguistik komputasi adalah bidang penyelidikan dan pembangunan selama beberapa dekad yang mengkaji bagaimana komputer boleh bekerja dengan, dan bahkan memahami, menterjemah, dan berkomunikasi dengan bahasa. Pemprosesan bahasa semulajadi (NLP) adalah bidang berkaitan yang memberi tumpuan kepada bagaimana komputer boleh memproses bahasa 'semulajadi', atau bahasa manusia.

### Contoh - pendiktean telefon

Jika anda pernah mendikte kepada telefon anda daripada menaip atau bertanya kepada pembantu maya soalan, ucapan anda telah ditukar kepada bentuk teks dan kemudian diproses atau *diuraikan* dari bahasa yang anda gunakan. Kata kunci yang dikesan kemudian diproses ke dalam format yang telefon atau pembantu boleh faham dan bertindak balas.

![pemahaman](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.ms.png)
> Pemahaman linguistik sebenar adalah sukar! Imej oleh [Jen Looper](https://twitter.com/jenlooper)

### Bagaimana teknologi ini dibuat mungkin?

Ini mungkin kerana seseorang menulis program komputer untuk melakukannya. Beberapa dekad yang lalu, beberapa penulis fiksyen sains meramalkan bahawa orang akan kebanyakannya bercakap dengan komputer mereka, dan komputer akan sentiasa memahami dengan tepat apa yang mereka maksudkan. Malangnya, ia ternyata menjadi masalah yang lebih sukar daripada yang dibayangkan oleh ramai, dan walaupun ia adalah masalah yang lebih difahami hari ini, terdapat cabaran yang ketara dalam mencapai pemprosesan bahasa semulajadi yang 'sempurna' apabila ia berkaitan dengan memahami makna ayat. Ini adalah masalah yang sangat sukar apabila ia berkaitan dengan memahami humor atau mengesan emosi seperti sindiran dalam ayat.

Pada ketika ini, anda mungkin mengingati kelas sekolah di mana guru meliputi bahagian tatabahasa dalam ayat. Di sesetengah negara, pelajar diajar tatabahasa dan linguistik sebagai subjek khusus, tetapi di banyak negara, topik-topik ini dimasukkan sebagai sebahagian daripada pembelajaran bahasa: sama ada bahasa pertama anda di sekolah rendah (belajar membaca dan menulis) dan mungkin bahasa kedua di sekolah menengah. Jangan risau jika anda bukan pakar dalam membezakan kata nama daripada kata kerja atau kata keterangan daripada kata sifat!

Jika anda bergelut dengan perbezaan antara *masa kini mudah* dan *masa kini progresif*, anda tidak bersendirian. Ini adalah perkara yang mencabar bagi ramai orang, bahkan penutur asli bahasa. Berita baiknya adalah bahawa komputer sangat baik dalam menerapkan peraturan formal, dan anda akan belajar untuk menulis kod yang boleh *menguraikan* ayat serta manusia. Cabaran yang lebih besar yang anda akan kaji kemudian ialah memahami *makna*, dan *sentimen*, sesuatu ayat.

## Prasyarat

Untuk pelajaran ini, prasyarat utama adalah dapat membaca dan memahami bahasa pelajaran ini. Tiada masalah matematik atau persamaan untuk diselesaikan. Walaupun pengarang asal menulis pelajaran ini dalam bahasa Inggeris, ia juga diterjemahkan ke dalam bahasa lain, jadi anda mungkin sedang membaca terjemahan. Terdapat contoh di mana beberapa bahasa yang berbeza digunakan (untuk membandingkan peraturan tatabahasa yang berbeza dari bahasa yang berbeza). Ini *tidak* diterjemahkan, tetapi teks penjelasan diterjemahkan, jadi maknanya harus jelas.

Untuk tugas pengekodan, anda akan menggunakan Python dan contoh-contohnya menggunakan Python 3.8.

Dalam bahagian ini, anda akan memerlukan, dan menggunakan:

- **Pemahaman Python 3**. Pemahaman bahasa pengaturcaraan dalam Python 3, pelajaran ini menggunakan input, gelung, pembacaan fail, array.
- **Visual Studio Code + sambungan**. Kami akan menggunakan Visual Studio Code dan sambungan Python. Anda juga boleh menggunakan IDE Python pilihan anda.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) adalah perpustakaan pemprosesan teks yang dipermudahkan untuk Python. Ikuti arahan di laman TextBlob untuk memasangnya pada sistem anda (pasang juga korpora, seperti yang ditunjukkan di bawah):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Anda boleh menjalankan Python secara langsung dalam persekitaran VS Code. Semak [dokumentasi](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) untuk maklumat lanjut.

## Bercakap dengan mesin

Sejarah mencuba untuk membuat komputer memahami bahasa manusia kembali beberapa dekad, dan salah seorang saintis terawal yang mempertimbangkan pemprosesan bahasa semulajadi adalah *Alan Turing*.

### Ujian 'Turing'

Ketika Turing sedang meneliti *kecerdasan buatan* pada tahun 1950-an, dia mempertimbangkan jika ujian perbualan boleh diberikan kepada seorang manusia dan komputer (melalui surat-menyurat yang ditaip) di mana manusia dalam perbualan itu tidak pasti sama ada mereka sedang berbual dengan manusia lain atau komputer.

Jika, selepas tempoh perbualan tertentu, manusia tidak dapat menentukan bahawa jawapan itu dari komputer atau tidak, maka bolehkah komputer dikatakan *berfikir*?

### Inspirasi - 'permainan tiruan'

Idea untuk ini datang dari permainan pesta yang dipanggil *The Imitation Game* di mana seorang penyiasat berada sendirian dalam bilik dan ditugaskan untuk menentukan siapa daripada dua orang (di bilik lain) adalah lelaki dan wanita masing-masing. Penyiasat boleh menghantar nota, dan mesti cuba memikirkan soalan di mana jawapan bertulis mendedahkan jantina orang misteri itu. Sudah tentu, pemain di bilik lain cuba menipu penyiasat dengan menjawab soalan dengan cara yang mengelirukan atau mengelirukan penyiasat, sambil memberikan penampilan menjawab dengan jujur.

### Membangunkan Eliza

Pada tahun 1960-an, seorang saintis MIT bernama *Joseph Weizenbaum* membangunkan [*Eliza*](https://wikipedia.org/wiki/ELIZA), seorang 'terapis' komputer yang akan menanyakan soalan kepada manusia dan memberikan penampilan memahami jawapan mereka. Walau bagaimanapun, walaupun Eliza boleh menguraikan ayat dan mengenal pasti beberapa struktur tatabahasa dan kata kunci untuk memberikan jawapan yang munasabah, ia tidak boleh dikatakan *memahami* ayat itu. Jika Eliza diberi ayat yang mengikuti format "**I am** <u>sedih</u>" ia mungkin menyusun semula dan menggantikan kata-kata dalam ayat untuk membentuk jawapan "Berapa lama **anda telah** <u>sedih</u>". 

Ini memberikan kesan bahawa Eliza memahami kenyataan itu dan menanyakan soalan susulan, sedangkan sebenarnya, ia hanya mengubah masa dan menambah beberapa kata. Jika Eliza tidak dapat mengenal pasti kata kunci yang ia mempunyai jawapan untuk, ia akan memberikan jawapan rawak yang boleh digunakan untuk banyak kenyataan yang berbeza. Eliza boleh dengan mudah ditipu, contohnya jika pengguna menulis "**Anda adalah** sebuah <u>basikal</u>" ia mungkin menjawab dengan "Berapa lama **saya telah** sebuah <u>basikal</u>?", bukannya jawapan yang lebih beralasan.

[![Berbual dengan Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Berbual dengan Eliza")

> 🎥 Klik imej di atas untuk video tentang program ELIZA asal

> Nota: Anda boleh membaca keterangan asal [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) yang diterbitkan pada tahun 1966 jika anda mempunyai akaun ACM. Sebagai alternatif, baca tentang Eliza di [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Latihan - mengekod bot perbualan asas

Bot perbualan, seperti Eliza, adalah program yang meminta input pengguna dan kelihatan memahami dan memberi respons dengan bijak. Tidak seperti Eliza, bot kita tidak akan mempunyai beberapa peraturan yang memberikan penampilan mempunyai perbualan pintar. Sebaliknya, bot kita hanya akan mempunyai satu kebolehan, iaitu untuk meneruskan perbualan dengan respons rawak yang mungkin berfungsi dalam hampir mana-mana perbualan remeh.

### Rancangan

Langkah-langkah anda semasa membina bot perbualan:

1. Cetak arahan yang menasihati pengguna cara berinteraksi dengan bot
2. Mulakan gelung
   1. Terima input pengguna
   2. Jika pengguna meminta untuk keluar, maka keluar
   3. Proses input pengguna dan tentukan respons (dalam kes ini, respons adalah pilihan rawak daripada senarai kemungkinan respons generik)
   4. Cetak respons
3. kembali ke langkah 2

### Membina bot

Mari kita buat bot seterusnya. Kita akan mulakan dengan mendefinisikan beberapa frasa.

1. Buat bot ini sendiri dalam Python dengan respons rawak berikut:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Berikut adalah beberapa output contoh untuk panduan anda (input pengguna pada baris yang bermula dengan `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Satu penyelesaian yang mungkin untuk tugas ini adalah [di sini](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Berhenti dan pertimbangkan

    1. Adakah anda fikir respons rawak akan 'menipu' seseorang untuk berfikir bahawa bot sebenarnya memahami mereka?
    2. Apakah ciri-ciri yang perlu ada pada bot untuk menjadi lebih berkesan?
    3. Jika bot benar-benar boleh 'memahami' makna ayat, adakah ia perlu 'mengingati' makna ayat-ayat sebelumnya dalam perbualan juga?

---

## 🚀Cabaran

Pilih salah satu elemen "berhenti dan pertimbangkan" di atas dan sama ada cuba melaksanakannya dalam kod atau tulis penyelesaian di atas kertas menggunakan pseudokod.

Dalam pelajaran seterusnya, anda akan belajar tentang beberapa pendekatan lain untuk menguraikan bahasa semulajadi dan pembelajaran mesin.

## [Kuiz pasca-ceramah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Kajian Semula & Kajian Sendiri

Lihat rujukan di bawah sebagai peluang bacaan lanjut.

### Rujukan

1. Schubert, Lenhart, "Linguistik Komputasi", *The Stanford Encyclopedia of Philosophy* (Edisi Musim Bunga 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tugasan 

[Cari bot](assignment.md)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.