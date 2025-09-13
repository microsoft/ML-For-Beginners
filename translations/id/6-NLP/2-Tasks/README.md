<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T20:26:44+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "id"
}
-->
# Tugas dan Teknik Pemrosesan Bahasa Alami yang Umum

Untuk sebagian besar *pemrosesan bahasa alami*, teks yang akan diproses harus dipecah, diperiksa, dan hasilnya disimpan atau dibandingkan dengan aturan dan kumpulan data. Tugas-tugas ini memungkinkan programmer untuk mendapatkan _makna_ atau _niat_ atau hanya _frekuensi_ istilah dan kata dalam sebuah teks.

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

Mari kita pelajari teknik-teknik umum yang digunakan dalam pemrosesan teks. Dikombinasikan dengan pembelajaran mesin, teknik-teknik ini membantu Anda menganalisis sejumlah besar teks secara efisien. Namun, sebelum menerapkan ML pada tugas-tugas ini, mari kita pahami masalah yang dihadapi oleh spesialis NLP.

## Tugas Umum dalam NLP

Ada berbagai cara untuk menganalisis teks yang sedang Anda kerjakan. Ada tugas-tugas yang dapat Anda lakukan, dan melalui tugas-tugas ini Anda dapat memahami teks dan menarik kesimpulan. Biasanya, Anda melakukan tugas-tugas ini secara berurutan.

### Tokenisasi

Mungkin hal pertama yang harus dilakukan sebagian besar algoritma NLP adalah memecah teks menjadi token, atau kata-kata. Meskipun ini terdengar sederhana, memperhitungkan tanda baca dan pembatas kata serta kalimat dalam berbagai bahasa dapat membuatnya menjadi rumit. Anda mungkin perlu menggunakan berbagai metode untuk menentukan batasan.

![tokenisasi](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisasi sebuah kalimat dari **Pride and Prejudice**. Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

### Embedding

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) adalah cara untuk mengonversi data teks Anda secara numerik. Embedding dilakukan sedemikian rupa sehingga kata-kata dengan makna serupa atau kata-kata yang sering digunakan bersama akan berkelompok.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings untuk sebuah kalimat dari **Pride and Prejudice**. Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

✅ Coba [alat menarik ini](https://projector.tensorflow.org/) untuk bereksperimen dengan word embeddings. Klik pada satu kata untuk melihat kelompok kata-kata serupa: 'toy' berkelompok dengan 'disney', 'lego', 'playstation', dan 'console'.

### Parsing & Tagging Bagian dari Ucapan

Setiap kata yang telah di-tokenisasi dapat diberi tag sebagai bagian dari ucapan - seperti kata benda, kata kerja, atau kata sifat. Kalimat `the quick red fox jumped over the lazy brown dog` mungkin diberi tag POS sebagai fox = kata benda, jumped = kata kerja.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing sebuah kalimat dari **Pride and Prejudice**. Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

Parsing adalah mengenali kata-kata yang saling terkait dalam sebuah kalimat - misalnya `the quick red fox jumped` adalah urutan kata sifat-kata benda-kata kerja yang terpisah dari urutan `lazy brown dog`.

### Frekuensi Kata dan Frasa

Prosedur yang berguna saat menganalisis teks dalam jumlah besar adalah membangun kamus dari setiap kata atau frasa yang menarik dan seberapa sering kata atau frasa tersebut muncul. Frasa `the quick red fox jumped over the lazy brown dog` memiliki frekuensi kata 2 untuk kata "the".

Mari kita lihat contoh teks di mana kita menghitung frekuensi kata. Puisi Rudyard Kipling berjudul The Winners mengandung ayat berikut:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Karena frekuensi frasa dapat bersifat tidak sensitif terhadap huruf besar atau sensitif terhadap huruf besar sesuai kebutuhan, frasa `a friend` memiliki frekuensi 2, `the` memiliki frekuensi 6, dan `travels` memiliki frekuensi 2.

### N-grams

Teks dapat dipecah menjadi urutan kata dengan panjang tertentu, satu kata (unigram), dua kata (bigram), tiga kata (trigram), atau sejumlah kata (n-grams).

Misalnya, `the quick red fox jumped over the lazy brown dog` dengan skor n-gram 2 menghasilkan n-grams berikut:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Mungkin lebih mudah untuk memvisualisasikannya sebagai kotak geser di atas kalimat. Berikut ini adalah untuk n-grams dengan 3 kata, n-gram ditampilkan dalam huruf tebal di setiap kalimat:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![jendela geser n-grams](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Nilai N-gram 3: Infografis oleh [Jen Looper](https://twitter.com/jenlooper)

### Ekstraksi Frasa Kata Benda

Dalam sebagian besar kalimat, terdapat kata benda yang menjadi subjek atau objek kalimat. Dalam bahasa Inggris, kata benda sering kali dapat dikenali dengan adanya 'a', 'an', atau 'the' di depannya. Mengidentifikasi subjek atau objek kalimat dengan 'mengekstraksi frasa kata benda' adalah tugas umum dalam NLP saat mencoba memahami makna sebuah kalimat.

✅ Dalam kalimat "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", bisakah Anda mengidentifikasi frasa kata benda?

Dalam kalimat `the quick red fox jumped over the lazy brown dog` terdapat 2 frasa kata benda: **quick red fox** dan **lazy brown dog**.

### Analisis Sentimen

Sebuah kalimat atau teks dapat dianalisis untuk sentimen, atau seberapa *positif* atau *negatif* teks tersebut. Sentimen diukur dalam *polaritas* dan *objektivitas/subjektivitas*. Polaritas diukur dari -1.0 hingga 1.0 (negatif hingga positif) dan 0.0 hingga 1.0 (paling objektif hingga paling subjektif).

✅ Nanti Anda akan belajar bahwa ada berbagai cara untuk menentukan sentimen menggunakan pembelajaran mesin, tetapi salah satu caranya adalah dengan memiliki daftar kata dan frasa yang dikategorikan sebagai positif atau negatif oleh seorang ahli manusia dan menerapkan model tersebut pada teks untuk menghitung skor polaritas. Bisakah Anda melihat bagaimana cara ini bekerja dalam beberapa situasi dan kurang efektif dalam situasi lainnya?

### Infleksi

Infleksi memungkinkan Anda mengambil sebuah kata dan mendapatkan bentuk tunggal atau jamak dari kata tersebut.

### Lematisasi

*Lemma* adalah akar atau kata dasar untuk sekumpulan kata, misalnya *flew*, *flies*, *flying* memiliki lemma dari kata kerja *fly*.

Ada juga basis data yang berguna untuk peneliti NLP, terutama:

### WordNet

[WordNet](https://wordnet.princeton.edu/) adalah basis data kata, sinonim, antonim, dan banyak detail lainnya untuk setiap kata dalam berbagai bahasa. Basis data ini sangat berguna saat mencoba membangun terjemahan, pemeriksa ejaan, atau alat bahasa apa pun.

## Perpustakaan NLP

Untungnya, Anda tidak perlu membangun semua teknik ini sendiri, karena ada pustaka Python yang sangat baik yang membuatnya jauh lebih mudah diakses oleh pengembang yang tidak mengkhususkan diri dalam pemrosesan bahasa alami atau pembelajaran mesin. Pelajaran berikutnya mencakup lebih banyak contoh pustaka ini, tetapi di sini Anda akan mempelajari beberapa contoh berguna untuk membantu Anda dengan tugas berikutnya.

### Latihan - menggunakan pustaka `TextBlob`

Mari kita gunakan pustaka bernama TextBlob karena pustaka ini memiliki API yang berguna untuk menangani jenis tugas ini. TextBlob "berdiri di atas bahu raksasa [NLTK](https://nltk.org) dan [pattern](https://github.com/clips/pattern), dan bekerja dengan baik dengan keduanya." Pustaka ini memiliki sejumlah besar ML yang tertanam dalam API-nya.

> Catatan: Panduan [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) yang berguna tersedia untuk TextBlob dan direkomendasikan untuk pengembang Python berpengalaman.

Saat mencoba mengidentifikasi *noun phrases*, TextBlob menawarkan beberapa opsi ekstraktor untuk menemukan frasa kata benda.

1. Lihatlah `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Apa yang terjadi di sini? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) adalah "Ekstraktor frasa kata benda yang menggunakan chunk parsing yang dilatih dengan korpus pelatihan ConLL-2000." ConLL-2000 mengacu pada Konferensi Pembelajaran Bahasa Alami Komputasional tahun 2000. Setiap tahun konferensi ini mengadakan lokakarya untuk menangani masalah NLP yang sulit, dan pada tahun 2000 masalahnya adalah chunking kata benda. Model dilatih pada Wall Street Journal, dengan "bagian 15-18 sebagai data pelatihan (211727 token) dan bagian 20 sebagai data uji (47377 token)". Anda dapat melihat prosedur yang digunakan [di sini](https://www.clips.uantwerpen.be/conll2000/chunking/) dan [hasilnya](https://ifarm.nl/erikt/research/np-chunking.html).

### Tantangan - meningkatkan bot Anda dengan NLP

Dalam pelajaran sebelumnya, Anda membuat bot Q&A yang sangat sederhana. Sekarang, Anda akan membuat Marvin sedikit lebih simpatik dengan menganalisis input Anda untuk sentimen dan mencetak respons yang sesuai dengan sentimen tersebut. Anda juga perlu mengidentifikasi `noun_phrase` dan menanyakan tentangnya.

Langkah-langkah Anda saat membangun bot percakapan yang lebih baik:

1. Cetak instruksi yang memberi tahu pengguna cara berinteraksi dengan bot
2. Mulai loop 
   1. Terima input pengguna
   2. Jika pengguna meminta keluar, maka keluar
   3. Proses input pengguna dan tentukan respons sentimen yang sesuai
   4. Jika frasa kata benda terdeteksi dalam sentimen, ubah menjadi bentuk jamak dan tanyakan lebih lanjut tentang topik tersebut
   5. Cetak respons
3. Kembali ke langkah 2

Berikut adalah cuplikan kode untuk menentukan sentimen menggunakan TextBlob. Perhatikan bahwa hanya ada empat *gradasi* respons sentimen (Anda dapat menambahkan lebih banyak jika Anda mau):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Berikut adalah beberapa output sampel untuk panduan Anda (input pengguna ada di baris yang dimulai dengan >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Salah satu solusi untuk tugas ini dapat ditemukan [di sini](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Pemeriksaan Pengetahuan

1. Apakah menurut Anda respons simpatik dapat 'menipu' seseorang untuk berpikir bahwa bot benar-benar memahami mereka?
2. Apakah mengidentifikasi frasa kata benda membuat bot lebih 'meyakinkan'?
3. Mengapa mengekstraksi 'frasa kata benda' dari sebuah kalimat merupakan hal yang berguna untuk dilakukan?

---

Implementasikan bot dalam pemeriksaan pengetahuan sebelumnya dan uji pada teman. Bisakah bot tersebut menipu mereka? Bisakah Anda membuat bot Anda lebih 'meyakinkan'?

## 🚀Tantangan

Ambil tugas dalam pemeriksaan pengetahuan sebelumnya dan coba implementasikan. Uji bot pada teman. Bisakah bot tersebut menipu mereka? Bisakah Anda membuat bot Anda lebih 'meyakinkan'?

## [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Dalam beberapa pelajaran berikutnya, Anda akan mempelajari lebih lanjut tentang analisis sentimen. Teliti teknik menarik ini dalam artikel seperti yang ada di [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tugas 

[Buat bot berbicara kembali](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.