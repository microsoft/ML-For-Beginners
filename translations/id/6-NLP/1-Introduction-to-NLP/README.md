<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T20:36:53+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "id"
}
-->
# Pengantar Pemrosesan Bahasa Alami

Pelajaran ini mencakup sejarah singkat dan konsep penting dari *pemrosesan bahasa alami*, sebuah cabang dari *linguistik komputasional*.

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Pengantar

NLP, sebagaimana biasa disebut, adalah salah satu bidang yang paling dikenal di mana pembelajaran mesin telah diterapkan dan digunakan dalam perangkat lunak produksi.

✅ Bisakah Anda memikirkan perangkat lunak yang Anda gunakan setiap hari yang mungkin memiliki beberapa NLP di dalamnya? Bagaimana dengan program pengolah kata atau aplikasi seluler yang Anda gunakan secara teratur?

Anda akan belajar tentang:

- **Ide tentang bahasa**. Bagaimana bahasa berkembang dan apa saja area studi utama.
- **Definisi dan konsep**. Anda juga akan mempelajari definisi dan konsep tentang bagaimana komputer memproses teks, termasuk parsing, tata bahasa, dan mengidentifikasi kata benda dan kata kerja. Ada beberapa tugas pemrograman dalam pelajaran ini, dan beberapa konsep penting diperkenalkan yang akan Anda pelajari untuk diprogram di pelajaran berikutnya.

## Linguistik Komputasional

Linguistik komputasional adalah bidang penelitian dan pengembangan selama beberapa dekade yang mempelajari bagaimana komputer dapat bekerja dengan, bahkan memahami, menerjemahkan, dan berkomunikasi dengan bahasa. Pemrosesan bahasa alami (NLP) adalah bidang terkait yang berfokus pada bagaimana komputer dapat memproses bahasa 'alami', atau bahasa manusia.

### Contoh - dikte telepon

Jika Anda pernah mendikte ke telepon Anda daripada mengetik atau bertanya kepada asisten virtual sebuah pertanyaan, ucapan Anda diubah menjadi bentuk teks dan kemudian diproses atau *diurai* dari bahasa yang Anda ucapkan. Kata kunci yang terdeteksi kemudian diproses ke dalam format yang dapat dipahami dan digunakan oleh telepon atau asisten.

![pemahaman](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Pemahaman linguistik yang sebenarnya itu sulit! Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

### Bagaimana teknologi ini bisa terjadi?

Hal ini dimungkinkan karena seseorang menulis program komputer untuk melakukannya. Beberapa dekade yang lalu, beberapa penulis fiksi ilmiah memprediksi bahwa orang akan lebih sering berbicara dengan komputer mereka, dan komputer akan selalu memahami apa yang mereka maksud. Sayangnya, ternyata masalah ini lebih sulit daripada yang dibayangkan banyak orang, dan meskipun masalah ini jauh lebih dipahami saat ini, ada tantangan signifikan dalam mencapai pemrosesan bahasa alami yang 'sempurna' dalam hal memahami makna sebuah kalimat. Ini adalah masalah yang sangat sulit terutama dalam memahami humor atau mendeteksi emosi seperti sarkasme dalam sebuah kalimat.

Pada titik ini, Anda mungkin mengingat pelajaran sekolah di mana guru membahas bagian-bagian tata bahasa dalam sebuah kalimat. Di beberapa negara, siswa diajarkan tata bahasa dan linguistik sebagai mata pelajaran khusus, tetapi di banyak negara, topik-topik ini dimasukkan sebagai bagian dari pembelajaran bahasa: baik bahasa pertama Anda di sekolah dasar (belajar membaca dan menulis) dan mungkin bahasa kedua di sekolah menengah. Jangan khawatir jika Anda bukan ahli dalam membedakan kata benda dari kata kerja atau kata keterangan dari kata sifat!

Jika Anda kesulitan dengan perbedaan antara *present simple* dan *present progressive*, Anda tidak sendirian. Ini adalah hal yang menantang bagi banyak orang, bahkan penutur asli suatu bahasa. Kabar baiknya adalah bahwa komputer sangat baik dalam menerapkan aturan formal, dan Anda akan belajar menulis kode yang dapat *mengurai* sebuah kalimat sebaik manusia. Tantangan yang lebih besar yang akan Anda pelajari nanti adalah memahami *makna* dan *sentimen* dari sebuah kalimat.

## Prasyarat

Untuk pelajaran ini, prasyarat utama adalah kemampuan membaca dan memahami bahasa pelajaran ini. Tidak ada masalah matematika atau persamaan yang harus diselesaikan. Meskipun penulis asli menulis pelajaran ini dalam bahasa Inggris, pelajaran ini juga diterjemahkan ke dalam bahasa lain, sehingga Anda mungkin sedang membaca terjemahan. Ada contoh di mana sejumlah bahasa berbeda digunakan (untuk membandingkan aturan tata bahasa yang berbeda dari berbagai bahasa). Bahasa-bahasa ini *tidak* diterjemahkan, tetapi teks penjelasannya diterjemahkan, sehingga maknanya harus jelas.

Untuk tugas pemrograman, Anda akan menggunakan Python dan contoh-contohnya menggunakan Python 3.8.

Dalam bagian ini, Anda akan membutuhkan, dan menggunakan:

- **Pemahaman Python 3**. Pemahaman bahasa pemrograman dalam Python 3, pelajaran ini menggunakan input, loop, pembacaan file, array.
- **Visual Studio Code + ekstensi**. Kami akan menggunakan Visual Studio Code dan ekstensi Python-nya. Anda juga dapat menggunakan IDE Python pilihan Anda.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) adalah pustaka pemrosesan teks yang disederhanakan untuk Python. Ikuti instruksi di situs TextBlob untuk menginstalnya di sistem Anda (instal juga korpora, seperti yang ditunjukkan di bawah):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Anda dapat menjalankan Python langsung di lingkungan VS Code. Periksa [dokumen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) untuk informasi lebih lanjut.

## Berbicara dengan Mesin

Sejarah mencoba membuat komputer memahami bahasa manusia sudah berlangsung selama beberapa dekade, dan salah satu ilmuwan pertama yang mempertimbangkan pemrosesan bahasa alami adalah *Alan Turing*.

### Tes 'Turing'

Ketika Turing meneliti *kecerdasan buatan* pada tahun 1950-an, ia mempertimbangkan apakah tes percakapan dapat diberikan kepada manusia dan komputer (melalui korespondensi tertulis) di mana manusia dalam percakapan tersebut tidak yakin apakah mereka sedang berbicara dengan manusia lain atau komputer.

Jika, setelah percakapan berlangsung cukup lama, manusia tidak dapat menentukan apakah jawaban berasal dari komputer atau tidak, maka dapatkah komputer dikatakan *berpikir*?

### Inspirasi - 'permainan imitasi'

Ide ini berasal dari permainan pesta yang disebut *Permainan Imitasi* di mana seorang interogator berada sendirian di sebuah ruangan dan ditugaskan untuk menentukan siapa dari dua orang (di ruangan lain) yang masing-masing adalah laki-laki dan perempuan. Interogator dapat mengirim catatan, dan harus mencoba memikirkan pertanyaan di mana jawaban tertulis mengungkapkan jenis kelamin orang misterius tersebut. Tentu saja, para pemain di ruangan lain mencoba menipu interogator dengan menjawab pertanyaan sedemikian rupa sehingga menyesatkan atau membingungkan interogator, sambil tetap memberikan kesan menjawab dengan jujur.

### Mengembangkan Eliza

Pada tahun 1960-an, seorang ilmuwan MIT bernama *Joseph Weizenbaum* mengembangkan [*Eliza*](https://wikipedia.org/wiki/ELIZA), seorang 'terapis' komputer yang akan mengajukan pertanyaan kepada manusia dan memberikan kesan memahami jawaban mereka. Namun, meskipun Eliza dapat mengurai sebuah kalimat dan mengidentifikasi konstruksi tata bahasa tertentu dan kata kunci untuk memberikan jawaban yang masuk akal, Eliza tidak dapat dikatakan *memahami* kalimat tersebut. Jika Eliza diberikan sebuah kalimat dengan format "**Saya merasa** <u>sedih</u>", ia mungkin akan menyusun ulang dan mengganti kata-kata dalam kalimat tersebut untuk membentuk respons "Sudah berapa lama **Anda merasa** <u>sedih</u>".

Ini memberikan kesan bahwa Eliza memahami pernyataan tersebut dan mengajukan pertanyaan lanjutan, padahal sebenarnya ia hanya mengubah bentuk kata kerja dan menambahkan beberapa kata. Jika Eliza tidak dapat mengidentifikasi kata kunci yang memiliki respons, ia akan memberikan respons acak yang dapat berlaku untuk banyak pernyataan berbeda. Eliza dapat dengan mudah ditipu, misalnya jika pengguna menulis "**Anda adalah** sebuah <u>sepeda</u>", ia mungkin akan merespons dengan "Sudah berapa lama **saya menjadi** sebuah <u>sepeda</u>?", alih-alih respons yang lebih masuk akal.

[![Berbicara dengan Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Berbicara dengan Eliza")

> 🎥 Klik gambar di atas untuk video tentang program ELIZA asli

> Catatan: Anda dapat membaca deskripsi asli tentang [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) yang diterbitkan pada tahun 1966 jika Anda memiliki akun ACM. Alternatifnya, baca tentang Eliza di [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Latihan - membuat bot percakapan dasar

Bot percakapan, seperti Eliza, adalah program yang meminta input pengguna dan tampaknya memahami serta merespons dengan cerdas. Tidak seperti Eliza, bot kita tidak akan memiliki beberapa aturan yang memberikan kesan percakapan yang cerdas. Sebaliknya, bot kita hanya akan memiliki satu kemampuan, yaitu melanjutkan percakapan dengan respons acak yang mungkin cocok dalam percakapan sepele.

### Rencana

Langkah-langkah Anda saat membuat bot percakapan:

1. Cetak instruksi yang memberi tahu pengguna cara berinteraksi dengan bot
2. Mulai loop
   1. Terima input pengguna
   2. Jika pengguna meminta keluar, maka keluar
   3. Proses input pengguna dan tentukan respons (dalam hal ini, respons adalah pilihan acak dari daftar kemungkinan respons generik)
   4. Cetak respons
3. Kembali ke langkah 2

### Membuat bot

Mari kita buat bot berikutnya. Kita akan mulai dengan mendefinisikan beberapa frasa.

1. Buat bot ini sendiri dalam Python dengan respons acak berikut:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Berikut adalah beberapa output contoh untuk panduan Anda (input pengguna ada di baris yang dimulai dengan `>`):

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

    Salah satu solusi yang mungkin untuk tugas ini ada [di sini](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Berhenti dan pertimbangkan

    1. Apakah menurut Anda respons acak akan 'menipu' seseorang untuk berpikir bahwa bot benar-benar memahami mereka?
    2. Fitur apa yang dibutuhkan bot agar lebih efektif?
    3. Jika bot benar-benar dapat 'memahami' makna sebuah kalimat, apakah ia juga perlu 'mengingat' makna kalimat sebelumnya dalam percakapan?

---

## 🚀Tantangan

Pilih salah satu elemen "berhenti dan pertimbangkan" di atas dan coba terapkan dalam kode atau tulis solusi di atas kertas menggunakan pseudocode.

Dalam pelajaran berikutnya, Anda akan belajar tentang sejumlah pendekatan lain untuk mengurai bahasa alami dan pembelajaran mesin.

## [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Lihat referensi di bawah ini sebagai peluang bacaan lebih lanjut.

### Referensi

1. Schubert, Lenhart, "Linguistik Komputasional", *The Stanford Encyclopedia of Philosophy* (Edisi Musim Semi 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "Tentang WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tugas 

[Cari bot](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.