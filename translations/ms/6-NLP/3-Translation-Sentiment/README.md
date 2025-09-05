<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T20:41:32+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "ms"
}
-->
# Terjemahan dan analisis sentimen dengan ML

Dalam pelajaran sebelumnya, anda telah belajar cara membina bot asas menggunakan `TextBlob`, sebuah perpustakaan yang menggabungkan ML di belakang tabir untuk melaksanakan tugas NLP asas seperti pengekstrakan frasa kata nama. Satu lagi cabaran penting dalam linguistik komputer ialah _terjemahan_ yang tepat bagi ayat daripada satu bahasa lisan atau tulisan kepada bahasa lain.

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

Terjemahan adalah masalah yang sangat sukar kerana terdapat ribuan bahasa dan setiap satu mempunyai peraturan tatabahasa yang sangat berbeza. Satu pendekatan ialah menukar peraturan tatabahasa formal untuk satu bahasa, seperti Bahasa Inggeris, kepada struktur yang tidak bergantung kepada bahasa, dan kemudian menterjemahkannya dengan menukar kembali kepada bahasa lain. Pendekatan ini bermaksud anda akan mengambil langkah berikut:

1. **Pengenalpastian**. Kenal pasti atau tag perkataan dalam bahasa input sebagai kata nama, kata kerja, dan sebagainya.
2. **Buat terjemahan**. Hasilkan terjemahan langsung bagi setiap perkataan dalam format bahasa sasaran.

### Contoh ayat, Bahasa Inggeris ke Bahasa Ireland

Dalam 'Bahasa Inggeris', ayat _I feel happy_ terdiri daripada tiga perkataan dalam susunan:

- **subjek** (I)
- **kata kerja** (feel)
- **kata sifat** (happy)

Namun, dalam bahasa 'Ireland', ayat yang sama mempunyai struktur tatabahasa yang sangat berbeza - emosi seperti "*happy*" atau "*sad*" dinyatakan sebagai sesuatu yang *berada pada* anda.

Frasa Bahasa Inggeris `I feel happy` dalam Bahasa Ireland akan menjadi `Tá athas orm`. Terjemahan *literal* adalah `Happy is upon me`.

Seorang penutur Bahasa Ireland yang menterjemah ke Bahasa Inggeris akan mengatakan `I feel happy`, bukan `Happy is upon me`, kerana mereka memahami maksud ayat tersebut, walaupun perkataan dan struktur ayatnya berbeza.

Susunan formal untuk ayat dalam Bahasa Ireland adalah:

- **kata kerja** (Tá atau is)
- **kata sifat** (athas, atau happy)
- **subjek** (orm, atau upon me)

## Terjemahan

Program terjemahan yang naif mungkin hanya menterjemah perkataan, tanpa mengambil kira struktur ayat.

✅ Jika anda telah belajar bahasa kedua (atau ketiga atau lebih) sebagai orang dewasa, anda mungkin bermula dengan berfikir dalam bahasa asal anda, menterjemah konsep perkataan demi perkataan dalam kepala anda kepada bahasa kedua, dan kemudian menyebut terjemahan anda. Ini serupa dengan apa yang dilakukan oleh program terjemahan komputer yang naif. Penting untuk melepasi fasa ini untuk mencapai kefasihan!

Terjemahan naif membawa kepada terjemahan yang buruk (dan kadang-kadang lucu): `I feel happy` diterjemahkan secara literal kepada `Mise bhraitheann athas` dalam Bahasa Ireland. Ini bermaksud (secara literal) `me feel happy` dan bukan ayat Bahasa Ireland yang sah. Walaupun Bahasa Inggeris dan Bahasa Ireland adalah bahasa yang dituturkan di dua pulau yang berdekatan, mereka adalah bahasa yang sangat berbeza dengan struktur tatabahasa yang berbeza.

> Anda boleh menonton beberapa video tentang tradisi linguistik Ireland seperti [yang ini](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Pendekatan pembelajaran mesin

Setakat ini, anda telah belajar tentang pendekatan peraturan formal untuk pemprosesan bahasa semula jadi. Satu lagi pendekatan ialah mengabaikan maksud perkataan, dan _sebaliknya menggunakan pembelajaran mesin untuk mengesan corak_. Ini boleh berfungsi dalam terjemahan jika anda mempunyai banyak teks (sebuah *corpus*) atau teks (*corpora*) dalam kedua-dua bahasa asal dan sasaran.

Sebagai contoh, pertimbangkan kes *Pride and Prejudice*, sebuah novel Bahasa Inggeris terkenal yang ditulis oleh Jane Austen pada tahun 1813. Jika anda merujuk buku itu dalam Bahasa Inggeris dan terjemahan manusia bagi buku itu dalam *Bahasa Perancis*, anda boleh mengesan frasa dalam satu yang diterjemahkan secara _idiomatik_ ke dalam yang lain. Anda akan melakukannya sebentar lagi.

Sebagai contoh, apabila frasa Bahasa Inggeris seperti `I have no money` diterjemahkan secara literal ke Bahasa Perancis, ia mungkin menjadi `Je n'ai pas de monnaie`. "Monnaie" adalah 'false cognate' Bahasa Perancis yang rumit, kerana 'money' dan 'monnaie' bukan sinonim. Terjemahan yang lebih baik yang mungkin dibuat oleh manusia adalah `Je n'ai pas d'argent`, kerana ia lebih baik menyampaikan maksud bahawa anda tidak mempunyai wang (daripada 'duit syiling' yang merupakan maksud 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Imej oleh [Jen Looper](https://twitter.com/jenlooper)

Jika model ML mempunyai cukup terjemahan manusia untuk membina model, ia boleh meningkatkan ketepatan terjemahan dengan mengenal pasti corak biasa dalam teks yang telah diterjemahkan sebelum ini oleh penutur manusia pakar kedua-dua bahasa.

### Latihan - terjemahan

Anda boleh menggunakan `TextBlob` untuk menterjemah ayat. Cuba baris pertama yang terkenal dari **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` melakukan kerja yang cukup baik dalam terjemahan: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Boleh dikatakan bahawa terjemahan TextBlob jauh lebih tepat, sebenarnya, daripada terjemahan Bahasa Perancis tahun 1932 bagi buku itu oleh V. Leconte dan Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Dalam kes ini, terjemahan yang dimaklumkan oleh ML melakukan kerja yang lebih baik daripada penterjemah manusia yang secara tidak perlu meletakkan kata-kata dalam mulut pengarang asal untuk 'kejelasan'.

> Apa yang sedang berlaku di sini? dan mengapa TextBlob begitu baik dalam terjemahan? Sebenarnya, ia menggunakan Google Translate di belakang tabir, AI yang canggih yang mampu menganalisis berjuta-juta frasa untuk meramalkan rentetan terbaik untuk tugas yang diberikan. Tiada apa-apa yang manual berlaku di sini dan anda memerlukan sambungan internet untuk menggunakan `blob.translate`.

✅ Cuba beberapa ayat lagi. Mana yang lebih baik, terjemahan ML atau manusia? Dalam kes mana?

## Analisis sentimen

Satu lagi bidang di mana pembelajaran mesin boleh berfungsi dengan sangat baik ialah analisis sentimen. Pendekatan bukan ML untuk sentimen adalah dengan mengenal pasti perkataan dan frasa yang 'positif' dan 'negatif'. Kemudian, diberikan teks baharu, hitung nilai keseluruhan perkataan positif, negatif dan neutral untuk mengenal pasti sentimen keseluruhan. 

Pendekatan ini mudah ditipu seperti yang mungkin anda lihat dalam tugas Marvin - ayat `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` adalah ayat sentimen negatif yang sarkastik, tetapi algoritma mudah mengesan 'great', 'wonderful', 'glad' sebagai positif dan 'waste', 'lost' dan 'dark' sebagai negatif. Sentimen keseluruhan dipengaruhi oleh perkataan yang bercanggah ini.

✅ Berhenti sejenak dan fikirkan bagaimana kita menyampaikan sarkasme sebagai penutur manusia. Nada suara memainkan peranan besar. Cuba ucapkan frasa "Well, that film was awesome" dengan cara yang berbeza untuk mengetahui bagaimana suara anda menyampaikan maksud.

### Pendekatan ML

Pendekatan ML adalah dengan mengumpulkan secara manual teks negatif dan positif - tweet, ulasan filem, atau apa sahaja di mana manusia telah memberikan skor *dan* pendapat bertulis. Kemudian teknik NLP boleh digunakan pada pendapat dan skor, supaya corak muncul (contohnya, ulasan filem positif cenderung mempunyai frasa 'Oscar worthy' lebih daripada ulasan filem negatif, atau ulasan restoran positif mengatakan 'gourmet' jauh lebih banyak daripada 'disgusting').

> ⚖️ **Contoh**: Jika anda bekerja di pejabat seorang ahli politik dan terdapat undang-undang baharu yang sedang dibahaskan, pengundi mungkin menulis kepada pejabat dengan e-mel yang menyokong atau menentang undang-undang baharu tertentu. Katakan anda ditugaskan membaca e-mel dan menyusunnya dalam 2 timbunan, *menyokong* dan *menentang*. Jika terdapat banyak e-mel, anda mungkin berasa terbeban untuk membaca semuanya. Bukankah lebih baik jika bot boleh membaca semuanya untuk anda, memahaminya dan memberitahu anda dalam timbunan mana setiap e-mel patut diletakkan? 
> 
> Satu cara untuk mencapai itu adalah dengan menggunakan Pembelajaran Mesin. Anda akan melatih model dengan sebahagian daripada e-mel *menentang* dan sebahagian daripada e-mel *menyokong*. Model akan cenderung mengaitkan frasa dan perkataan dengan pihak menentang dan pihak menyokong, *tetapi ia tidak akan memahami sebarang kandungan*, hanya bahawa perkataan dan corak tertentu lebih cenderung muncul dalam e-mel *menentang* atau *menyokong*. Anda boleh mengujinya dengan beberapa e-mel yang anda tidak gunakan untuk melatih model, dan lihat jika ia sampai kepada kesimpulan yang sama seperti anda. Kemudian, setelah anda berpuas hati dengan ketepatan model, anda boleh memproses e-mel masa depan tanpa perlu membaca setiap satu.

✅ Adakah proses ini terdengar seperti proses yang telah anda gunakan dalam pelajaran sebelumnya?

## Latihan - ayat sentimental

Sentimen diukur dengan *polarity* dari -1 hingga 1, bermaksud -1 adalah sentimen paling negatif, dan 1 adalah sentimen paling positif. Sentimen juga diukur dengan skor 0 - 1 untuk objektiviti (0) dan subjektiviti (1).

Lihat semula *Pride and Prejudice* oleh Jane Austen. Teks tersedia di sini di [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Contoh di bawah menunjukkan program pendek yang menganalisis sentimen ayat pertama dan terakhir dari buku tersebut dan memaparkan skor polariti dan subjektiviti/objektiviti sentimennya.

Anda harus menggunakan perpustakaan `TextBlob` (dijelaskan di atas) untuk menentukan `sentiment` (anda tidak perlu menulis kalkulator sentimen anda sendiri) dalam tugas berikut.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Anda melihat output berikut:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Cabaran - periksa polariti sentimen

Tugas anda adalah untuk menentukan, menggunakan polariti sentimen, sama ada *Pride and Prejudice* mempunyai lebih banyak ayat yang benar-benar positif daripada yang benar-benar negatif. Untuk tugas ini, anda boleh menganggap bahawa skor polariti 1 atau -1 adalah benar-benar positif atau negatif masing-masing.

**Langkah-langkah:**

1. Muat turun salinan [Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) dari Project Gutenberg sebagai fail .txt. Buang metadata di awal dan akhir fail, tinggalkan hanya teks asal
2. Buka fail dalam Python dan ekstrak kandungannya sebagai string
3. Buat TextBlob menggunakan string buku
4. Analisis setiap ayat dalam buku dalam gelung
   1. Jika polariti adalah 1 atau -1 simpan ayat dalam array atau senarai mesej positif atau negatif
5. Pada akhirnya, cetak semua ayat positif dan negatif (secara berasingan) dan jumlah setiap satu.

Berikut adalah [penyelesaian](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) sebagai contoh.

✅ Pemeriksaan Pengetahuan

1. Sentimen berdasarkan perkataan yang digunakan dalam ayat, tetapi adakah kod *memahami* perkataan tersebut?
2. Adakah anda fikir polariti sentimen adalah tepat, atau dengan kata lain, adakah anda *bersetuju* dengan skor tersebut?
   1. Khususnya, adakah anda bersetuju atau tidak bersetuju dengan polariti **positif** mutlak bagi ayat berikut?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Tiga ayat berikut telah diberi skor dengan sentimen positif mutlak, tetapi setelah dibaca dengan teliti, mereka bukan ayat positif. Mengapa analisis sentimen menganggap mereka ayat positif?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Adakah anda bersetuju atau tidak bersetuju dengan polariti **negatif** mutlak bagi ayat berikut?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Mana-mana peminat Jane Austen akan memahami bahawa dia sering menggunakan bukunya untuk mengkritik aspek yang lebih tidak masuk akal dalam masyarakat Inggeris Regency. Elizabeth Bennett, watak utama dalam *Pride and Prejudice*, adalah pemerhati sosial yang tajam (seperti pengarangnya) dan bahasanya sering sangat bernuansa. Malah Mr. Darcy (watak cinta dalam cerita) mencatatkan penggunaan bahasa Elizabeth yang suka bermain dan menggoda: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Cabaran

Bolehkah anda menjadikan Marvin lebih baik dengan mengekstrak ciri lain daripada input pengguna?

## [Kuiz pasca-kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Ulasan & Kajian Kendiri
Terdapat banyak cara untuk mengekstrak sentimen daripada teks. Fikirkan tentang aplikasi perniagaan yang mungkin menggunakan teknik ini. Fikirkan bagaimana ia boleh menjadi tidak tepat. Baca lebih lanjut mengenai sistem perusahaan canggih yang menganalisis sentimen seperti [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Uji beberapa ayat daripada Pride and Prejudice di atas dan lihat sama ada ia dapat mengesan nuansa.

## Tugasan 

[Lesen puitis](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.