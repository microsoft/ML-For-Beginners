<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T20:27:18+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "ms"
}
-->
# Tugas Pemprosesan Bahasa Semula Jadi dan Teknik-Tekniknya

Untuk kebanyakan *tugas pemprosesan bahasa semula jadi*, teks yang akan diproses mesti dipecahkan, diperiksa, dan hasilnya disimpan atau dirujuk silang dengan peraturan dan set data. Tugas-tugas ini membolehkan pengaturcara mendapatkan _makna_ atau _niat_ atau hanya _kekerapan_ istilah dan perkataan dalam teks.

## [Kuiz Pra-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

Mari kita terokai teknik-teknik biasa yang digunakan dalam pemprosesan teks. Digabungkan dengan pembelajaran mesin, teknik-teknik ini membantu anda menganalisis sejumlah besar teks dengan cekap. Sebelum menerapkan ML kepada tugas-tugas ini, mari kita fahami masalah yang dihadapi oleh pakar NLP.

## Tugas-Tugas Biasa dalam NLP

Terdapat pelbagai cara untuk menganalisis teks yang sedang anda kerjakan. Terdapat tugas-tugas yang boleh anda laksanakan dan melalui tugas-tugas ini anda dapat memahami teks dan membuat kesimpulan. Biasanya, anda melaksanakan tugas-tugas ini secara berurutan.

### Tokenisasi

Kemungkinan besar perkara pertama yang perlu dilakukan oleh kebanyakan algoritma NLP ialah memecahkan teks kepada token, atau perkataan. Walaupun ini kedengaran mudah, mengambil kira tanda baca dan pemisah perkataan serta ayat dalam pelbagai bahasa boleh menjadi rumit. Anda mungkin perlu menggunakan pelbagai kaedah untuk menentukan sempadan.

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisasi ayat daripada **Pride and Prejudice**. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

### Embedding

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) adalah cara untuk menukar data teks anda kepada bentuk numerik. Embedding dilakukan dengan cara supaya perkataan yang mempunyai makna serupa atau perkataan yang digunakan bersama-sama berkumpul bersama.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings untuk ayat dalam **Pride and Prejudice**. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

✅ Cuba [alat menarik ini](https://projector.tensorflow.org/) untuk bereksperimen dengan word embeddings. Klik pada satu perkataan menunjukkan kumpulan perkataan serupa: 'toy' berkumpul dengan 'disney', 'lego', 'playstation', dan 'console'.

### Parsing & Tagging Bahagian Ucapan

Setiap perkataan yang telah ditokenkan boleh ditandai sebagai bahagian ucapan - kata nama, kata kerja, atau kata sifat. Ayat `the quick red fox jumped over the lazy brown dog` mungkin ditandai POS sebagai fox = kata nama, jumped = kata kerja.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing ayat daripada **Pride and Prejudice**. Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

Parsing ialah mengenal pasti perkataan yang berkaitan antara satu sama lain dalam ayat - contohnya `the quick red fox jumped` ialah urutan kata sifat-kata nama-kata kerja yang berasingan daripada urutan `lazy brown dog`.

### Kekerapan Perkataan dan Frasa

Prosedur yang berguna semasa menganalisis sejumlah besar teks ialah membina kamus setiap perkataan atau frasa yang menarik dan berapa kerap ia muncul. Frasa `the quick red fox jumped over the lazy brown dog` mempunyai kekerapan perkataan sebanyak 2 untuk the.

Mari kita lihat contoh teks di mana kita mengira kekerapan perkataan. Puisi Rudyard Kipling, The Winners, mengandungi ayat berikut:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Oleh kerana kekerapan frasa boleh sensitif huruf besar atau tidak sensitif huruf besar seperti yang diperlukan, frasa `a friend` mempunyai kekerapan sebanyak 2 dan `the` mempunyai kekerapan sebanyak 6, dan `travels` ialah 2.

### N-grams

Teks boleh dipecahkan kepada urutan perkataan dengan panjang tertentu, satu perkataan (unigram), dua perkataan (bigram), tiga perkataan (trigram) atau sebarang bilangan perkataan (n-grams).

Sebagai contoh, `the quick red fox jumped over the lazy brown dog` dengan skor n-gram sebanyak 2 menghasilkan n-grams berikut:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Ia mungkin lebih mudah untuk menggambarkannya sebagai kotak gelongsor di atas ayat. Berikut ialah n-grams untuk 3 perkataan, n-gram ditunjukkan dalam huruf tebal dalam setiap ayat:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Nilai N-gram sebanyak 3: Infografik oleh [Jen Looper](https://twitter.com/jenlooper)

### Ekstraksi Frasa Kata Nama

Dalam kebanyakan ayat, terdapat kata nama yang menjadi subjek atau objek ayat. Dalam bahasa Inggeris, ia sering dapat dikenalpasti dengan 'a', 'an', atau 'the' yang mendahuluinya. Mengenalpasti subjek atau objek ayat dengan 'mengekstrak frasa kata nama' adalah tugas biasa dalam NLP apabila cuba memahami makna ayat.

✅ Dalam ayat "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", bolehkah anda mengenalpasti frasa kata nama?

Dalam ayat `the quick red fox jumped over the lazy brown dog` terdapat 2 frasa kata nama: **quick red fox** dan **lazy brown dog**.

### Analisis Sentimen

Satu ayat atau teks boleh dianalisis untuk sentimen, atau betapa *positif* atau *negatif*nya ia. Sentimen diukur dalam *polariti* dan *objektiviti/subjektiviti*. Polariti diukur dari -1.0 hingga 1.0 (negatif ke positif) dan 0.0 hingga 1.0 (paling objektif ke paling subjektif).

✅ Nanti anda akan belajar bahawa terdapat pelbagai cara untuk menentukan sentimen menggunakan pembelajaran mesin, tetapi satu cara ialah mempunyai senarai perkataan dan frasa yang dikategorikan sebagai positif atau negatif oleh pakar manusia dan menerapkan model itu kepada teks untuk mengira skor polariti. Bolehkah anda melihat bagaimana ini berfungsi dalam beberapa keadaan dan kurang berfungsi dalam keadaan lain?

### Infleksi

Infleksi membolehkan anda mengambil satu perkataan dan mendapatkan bentuk tunggal atau jamak perkataan tersebut.

### Lematisasi

*Lemma* ialah akar atau kata dasar untuk satu set perkataan, contohnya *flew*, *flies*, *flying* mempunyai lemma kata kerja *fly*.

Terdapat juga pangkalan data berguna yang tersedia untuk penyelidik NLP, terutamanya:

### WordNet

[WordNet](https://wordnet.princeton.edu/) ialah pangkalan data perkataan, sinonim, antonim dan banyak butiran lain untuk setiap perkataan dalam pelbagai bahasa. Ia sangat berguna apabila cuba membina terjemahan, pemeriksa ejaan, atau alat bahasa dari sebarang jenis.

## Perpustakaan NLP

Nasib baik, anda tidak perlu membina semua teknik ini sendiri, kerana terdapat perpustakaan Python yang sangat baik tersedia yang menjadikannya lebih mudah diakses oleh pembangun yang tidak pakar dalam pemprosesan bahasa semula jadi atau pembelajaran mesin. Pelajaran seterusnya termasuk lebih banyak contoh ini, tetapi di sini anda akan belajar beberapa contoh berguna untuk membantu anda dengan tugas seterusnya.

### Latihan - menggunakan perpustakaan `TextBlob`

Mari gunakan perpustakaan bernama TextBlob kerana ia mengandungi API yang berguna untuk menangani jenis tugas ini. TextBlob "berdiri di atas bahu gergasi [NLTK](https://nltk.org) dan [pattern](https://github.com/clips/pattern), dan berfungsi dengan baik dengan kedua-duanya." Ia mempunyai sejumlah besar ML yang tertanam dalam API-nya.

> Nota: Panduan [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) yang berguna tersedia untuk TextBlob yang disyorkan untuk pembangun Python berpengalaman 

Apabila cuba mengenalpasti *noun phrases*, TextBlob menawarkan beberapa pilihan ekstraktor untuk mencari frasa kata nama. 

1. Lihat `ConllExtractor`.

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

    > Apa yang sedang berlaku di sini? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) ialah "Ekstraktor frasa kata nama yang menggunakan chunk parsing yang dilatih dengan korpus latihan ConLL-2000." ConLL-2000 merujuk kepada Persidangan Pembelajaran Bahasa Semula Jadi Komputasi pada tahun 2000. Setiap tahun persidangan itu mengadakan bengkel untuk menangani masalah NLP yang sukar, dan pada tahun 2000 ia adalah noun chunking. Model dilatih pada Wall Street Journal, dengan "bahagian 15-18 sebagai data latihan (211727 token) dan bahagian 20 sebagai data ujian (47377 token)". Anda boleh melihat prosedur yang digunakan [di sini](https://www.clips.uantwerpen.be/conll2000/chunking/) dan [hasilnya](https://ifarm.nl/erikt/research/np-chunking.html).

### Cabaran - meningkatkan bot anda dengan NLP

Dalam pelajaran sebelumnya anda telah membina bot Q&A yang sangat ringkas. Sekarang, anda akan menjadikan Marvin lebih bersimpati dengan menganalisis input anda untuk sentimen dan mencetak respons yang sesuai dengan sentimen tersebut. Anda juga perlu mengenalpasti `noun_phrase` dan bertanya mengenainya.

Langkah-langkah anda semasa membina bot perbualan yang lebih baik:

1. Cetak arahan yang menasihati pengguna cara berinteraksi dengan bot
2. Mulakan gelung 
   1. Terima input pengguna
   2. Jika pengguna meminta untuk keluar, maka keluar
   3. Proses input pengguna dan tentukan respons sentimen yang sesuai
   4. Jika frasa kata nama dikesan dalam sentimen, jadikan ia bentuk jamak dan minta input lanjut mengenai topik tersebut
   5. Cetak respons
3. Kembali ke langkah 2

Berikut ialah petikan kod untuk menentukan sentimen menggunakan TextBlob. Perhatikan bahawa terdapat hanya empat *gradien* respons sentimen (anda boleh mempunyai lebih banyak jika anda mahu):

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

Berikut ialah beberapa output contoh untuk panduan anda (input pengguna berada pada baris yang bermula dengan >):

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

Satu penyelesaian yang mungkin untuk tugas ini adalah [di sini](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Pemeriksaan Pengetahuan

1. Adakah anda fikir respons yang bersimpati akan 'menipu' seseorang untuk berfikir bahawa bot itu benar-benar memahami mereka?
2. Adakah mengenalpasti frasa kata nama menjadikan bot lebih 'boleh dipercayai'?
3. Mengapa mengekstrak 'frasa kata nama' daripada ayat adalah perkara yang berguna untuk dilakukan?

---

Laksanakan bot dalam pemeriksaan pengetahuan sebelumnya dan uji pada rakan. Bolehkah ia menipu mereka? Bolehkah anda menjadikan bot anda lebih 'boleh dipercayai'?

## 🚀Cabaran

Ambil satu tugas dalam pemeriksaan pengetahuan sebelumnya dan cuba laksanakannya. Uji bot pada rakan. Bolehkah ia menipu mereka? Bolehkah anda menjadikan bot anda lebih 'boleh dipercayai'?

## [Kuiz Pasca-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri

Dalam beberapa pelajaran seterusnya anda akan belajar lebih lanjut tentang analisis sentimen. Kajilah teknik menarik ini dalam artikel seperti di [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tugasan 

[Make a bot talk back](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.