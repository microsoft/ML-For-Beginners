<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T20:37:22+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "ms"
}
-->
# Pengenalan kepada pemprosesan bahasa semula jadi

Pelajaran ini merangkumi sejarah ringkas dan konsep penting tentang *pemprosesan bahasa semula jadi*, satu cabang daripada *linguistik komputer*.

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Pengenalan

NLP, seperti yang biasa dikenali, adalah salah satu bidang yang paling terkenal di mana pembelajaran mesin telah diterapkan dan digunakan dalam perisian pengeluaran.

✅ Bolehkah anda memikirkan perisian yang anda gunakan setiap hari yang mungkin mempunyai elemen NLP? Bagaimana pula dengan program pemprosesan kata atau aplikasi mudah alih yang anda gunakan secara berkala?

Anda akan belajar tentang:

- **Idea tentang bahasa**. Bagaimana bahasa berkembang dan apakah bidang utama kajian.
- **Definisi dan konsep**. Anda juga akan mempelajari definisi dan konsep tentang bagaimana komputer memproses teks, termasuk penguraian, tatabahasa, dan mengenal pasti kata nama dan kata kerja. Terdapat beberapa tugas pengekodan dalam pelajaran ini, dan beberapa konsep penting diperkenalkan yang akan anda pelajari untuk kod dalam pelajaran seterusnya.

## Linguistik komputer

Linguistik komputer adalah bidang penyelidikan dan pembangunan selama beberapa dekad yang mengkaji bagaimana komputer boleh bekerja dengan, dan bahkan memahami, menterjemah, serta berkomunikasi dengan bahasa. Pemprosesan bahasa semula jadi (NLP) adalah bidang berkaitan yang memberi tumpuan kepada bagaimana komputer boleh memproses bahasa 'semula jadi', atau bahasa manusia.

### Contoh - pendikte telefon

Jika anda pernah mendikte kepada telefon anda daripada menaip atau bertanya kepada pembantu maya soalan, ucapan anda telah ditukar kepada bentuk teks dan kemudian diproses atau *diuraikan* daripada bahasa yang anda gunakan. Kata kunci yang dikesan kemudian diproses ke dalam format yang telefon atau pembantu boleh fahami dan bertindak balas.

![pemahaman](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Pemahaman linguistik sebenar adalah sukar! Imej oleh [Jen Looper](https://twitter.com/jenlooper)

### Bagaimana teknologi ini menjadi mungkin?

Ini menjadi mungkin kerana seseorang telah menulis program komputer untuk melakukannya. Beberapa dekad yang lalu, beberapa penulis fiksyen sains meramalkan bahawa manusia akan lebih banyak bercakap dengan komputer mereka, dan komputer akan sentiasa memahami dengan tepat apa yang dimaksudkan. Malangnya, ia ternyata menjadi masalah yang lebih sukar daripada yang dibayangkan oleh ramai orang, dan walaupun ia adalah masalah yang lebih difahami hari ini, terdapat cabaran besar dalam mencapai pemprosesan bahasa semula jadi yang 'sempurna' apabila ia berkaitan dengan memahami maksud ayat. Ini adalah masalah yang sangat sukar apabila ia berkaitan dengan memahami humor atau mengesan emosi seperti sindiran dalam ayat.

Pada ketika ini, anda mungkin teringat kelas sekolah di mana guru mengajar bahagian tatabahasa dalam ayat. Di beberapa negara, pelajar diajar tatabahasa dan linguistik sebagai subjek khusus, tetapi di banyak negara, topik ini termasuk sebagai sebahagian daripada pembelajaran bahasa: sama ada bahasa pertama anda di sekolah rendah (belajar membaca dan menulis) dan mungkin bahasa kedua di sekolah menengah. Jangan risau jika anda bukan pakar dalam membezakan kata nama daripada kata kerja atau kata keterangan daripada kata sifat!

Jika anda bergelut dengan perbezaan antara *present simple* dan *present progressive*, anda tidak keseorangan. Ini adalah perkara yang mencabar bagi ramai orang, termasuk penutur asli sesuatu bahasa. Berita baiknya ialah komputer sangat baik dalam menerapkan peraturan formal, dan anda akan belajar menulis kod yang boleh *menguraikan* ayat sebaik manusia. Cabaran yang lebih besar yang akan anda kaji kemudian ialah memahami *maksud* dan *sentimen* sesuatu ayat.

## Prasyarat

Untuk pelajaran ini, prasyarat utama adalah dapat membaca dan memahami bahasa pelajaran ini. Tiada masalah matematik atau persamaan untuk diselesaikan. Walaupun penulis asal menulis pelajaran ini dalam bahasa Inggeris, ia juga diterjemahkan ke dalam bahasa lain, jadi anda mungkin sedang membaca terjemahan. Terdapat contoh di mana beberapa bahasa yang berbeza digunakan (untuk membandingkan peraturan tatabahasa yang berbeza bagi bahasa yang berbeza). Ini *tidak* diterjemahkan, tetapi teks penjelasan diterjemahkan, jadi maksudnya seharusnya jelas.

Untuk tugas pengekodan, anda akan menggunakan Python dan contoh-contohnya menggunakan Python 3.8.

Dalam bahagian ini, anda akan memerlukan, dan menggunakan:

- **Pemahaman Python 3**. Pemahaman bahasa pengaturcaraan dalam Python 3, pelajaran ini menggunakan input, gelung, pembacaan fail, dan array.
- **Visual Studio Code + sambungan**. Kami akan menggunakan Visual Studio Code dan sambungan Pythonnya. Anda juga boleh menggunakan IDE Python pilihan anda.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) adalah pustaka pemprosesan teks yang dipermudahkan untuk Python. Ikuti arahan di laman TextBlob untuk memasangnya pada sistem anda (pasang korpora juga, seperti yang ditunjukkan di bawah):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Petua: Anda boleh menjalankan Python secara langsung dalam persekitaran VS Code. Semak [dokumen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) untuk maklumat lanjut.

## Berbicara dengan mesin

Sejarah usaha untuk membuat komputer memahami bahasa manusia telah berlangsung selama beberapa dekad, dan salah seorang saintis terawal yang mempertimbangkan pemprosesan bahasa semula jadi ialah *Alan Turing*.

### Ujian 'Turing'

Apabila Turing sedang menyelidik *kecerdasan buatan* pada tahun 1950-an, beliau mempertimbangkan sama ada ujian perbualan boleh diberikan kepada manusia dan komputer (melalui korespondensi bertulis) di mana manusia dalam perbualan itu tidak pasti sama ada mereka sedang berbual dengan manusia lain atau komputer.

Jika, selepas tempoh perbualan tertentu, manusia tidak dapat menentukan sama ada jawapan itu daripada komputer atau tidak, maka bolehkah komputer itu dikatakan *berfikir*?

### Inspirasi - 'permainan tiruan'

Idea ini datang daripada permainan parti yang dipanggil *The Imitation Game* di mana seorang penyiasat berada sendirian di dalam bilik dan ditugaskan untuk menentukan siapa di antara dua orang (di bilik lain) adalah lelaki dan perempuan masing-masing. Penyiasat boleh menghantar nota, dan mesti cuba memikirkan soalan di mana jawapan bertulis mendedahkan jantina orang misteri. Sudah tentu, pemain di bilik lain cuba mengelirukan penyiasat dengan menjawab soalan sedemikian rupa untuk mengelirukan atau mengelirukan penyiasat, sambil juga memberikan penampilan menjawab dengan jujur.

### Membangunkan Eliza

Pada tahun 1960-an, seorang saintis MIT bernama *Joseph Weizenbaum* membangunkan [*Eliza*](https://wikipedia.org/wiki/ELIZA), seorang 'terapis' komputer yang akan bertanya soalan kepada manusia dan memberikan penampilan memahami jawapan mereka. Walau bagaimanapun, walaupun Eliza boleh menguraikan ayat dan mengenal pasti struktur tatabahasa tertentu dan kata kunci untuk memberikan jawapan yang munasabah, ia tidak boleh dikatakan *memahami* ayat tersebut. Jika Eliza diberikan ayat dengan format "**Saya** <u>sedih</u>", ia mungkin menyusun semula dan menggantikan kata-kata dalam ayat untuk membentuk jawapan "Berapa lama **anda telah** <u>sedih</u>". 

Ini memberikan gambaran bahawa Eliza memahami kenyataan itu dan sedang bertanya soalan susulan, sedangkan sebenarnya, ia hanya menukar masa dan menambah beberapa perkataan. Jika Eliza tidak dapat mengenal pasti kata kunci yang mempunyai jawapan, ia sebaliknya akan memberikan jawapan rawak yang boleh digunakan untuk banyak kenyataan yang berbeza. Eliza boleh dengan mudah dikelirukan, contohnya jika pengguna menulis "**Anda adalah** sebuah <u>basikal</u>", ia mungkin menjawab "Berapa lama **saya telah** sebuah <u>basikal</u>?", bukannya jawapan yang lebih munasabah.

[![Berbual dengan Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Berbual dengan Eliza")

> 🎥 Klik imej di atas untuk video tentang program ELIZA asal

> Nota: Anda boleh membaca penerangan asal tentang [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) yang diterbitkan pada tahun 1966 jika anda mempunyai akaun ACM. Sebagai alternatif, baca tentang Eliza di [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Latihan - pengekodan bot perbualan asas

Bot perbualan, seperti Eliza, adalah program yang mendapatkan input pengguna dan kelihatan memahami serta bertindak balas dengan bijak. Tidak seperti Eliza, bot kita tidak akan mempunyai beberapa peraturan yang memberikan penampilan perbualan yang bijak. Sebaliknya, bot kita hanya akan mempunyai satu keupayaan, iaitu meneruskan perbualan dengan jawapan rawak yang mungkin berfungsi dalam hampir mana-mana perbualan remeh.

### Rancangan

Langkah-langkah anda semasa membina bot perbualan:

1. Cetak arahan yang menasihati pengguna cara berinteraksi dengan bot
2. Mulakan gelung
   1. Terima input pengguna
   2. Jika pengguna meminta keluar, maka keluar
   3. Proses input pengguna dan tentukan jawapan (dalam kes ini, jawapan adalah pilihan rawak daripada senarai kemungkinan jawapan generik)
   4. Cetak jawapan
3. Kembali ke langkah 2

### Membina bot

Mari kita bina bot seterusnya. Kita akan mulakan dengan mendefinisikan beberapa frasa.

1. Cipta bot ini sendiri dalam Python dengan jawapan rawak berikut:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Berikut adalah beberapa output contoh untuk panduan anda (input pengguna adalah pada baris yang bermula dengan `>`):

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

    1. Adakah anda fikir jawapan rawak akan 'mengelirukan' seseorang untuk berfikir bahawa bot sebenarnya memahami mereka?
    2. Apakah ciri yang diperlukan oleh bot untuk menjadi lebih berkesan?
    3. Jika bot benar-benar boleh 'memahami' maksud ayat, adakah ia perlu 'mengingati' maksud ayat sebelumnya dalam perbualan juga?

---

## 🚀Cabaran

Pilih salah satu elemen "berhenti dan pertimbangkan" di atas dan cuba melaksanakannya dalam kod atau tulis penyelesaian di atas kertas menggunakan pseudokod.

Dalam pelajaran seterusnya, anda akan belajar tentang beberapa pendekatan lain untuk menguraikan bahasa semula jadi dan pembelajaran mesin.

## [Kuiz pasca-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Lihat rujukan di bawah sebagai peluang pembacaan lanjut.

### Rujukan

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tugasan 

[Carian bot](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.