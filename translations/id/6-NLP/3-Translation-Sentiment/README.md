<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T20:40:17+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "id"
}
-->
# Analisis Sentimen dan Terjemahan dengan ML

Dalam pelajaran sebelumnya, Anda telah belajar cara membangun bot dasar menggunakan `TextBlob`, sebuah pustaka yang mengintegrasikan ML di balik layar untuk melakukan tugas NLP dasar seperti ekstraksi frasa kata benda. Tantangan penting lainnya dalam linguistik komputasi adalah _terjemahan_ yang akurat dari satu bahasa lisan atau tulisan ke bahasa lain.

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

Terjemahan adalah masalah yang sangat sulit karena ada ribuan bahasa, masing-masing dengan aturan tata bahasa yang sangat berbeda. Salah satu pendekatan adalah mengubah aturan tata bahasa formal dari satu bahasa, seperti bahasa Inggris, menjadi struktur yang tidak bergantung pada bahasa, lalu menerjemahkannya dengan mengubah kembali ke bahasa lain. Pendekatan ini melibatkan langkah-langkah berikut:

1. **Identifikasi**. Identifikasi atau tandai kata-kata dalam bahasa input sebagai kata benda, kata kerja, dll.
2. **Buat terjemahan**. Hasilkan terjemahan langsung dari setiap kata dalam format bahasa target.

### Contoh kalimat, Inggris ke Irlandia

Dalam bahasa 'Inggris', kalimat _I feel happy_ terdiri dari tiga kata dengan urutan:

- **subjek** (I)
- **kata kerja** (feel)
- **kata sifat** (happy)

Namun, dalam bahasa 'Irlandia', kalimat yang sama memiliki struktur tata bahasa yang sangat berbeda - emosi seperti "*happy*" atau "*sad*" diekspresikan sebagai sesuatu yang *ada pada* Anda.

Frasa bahasa Inggris `I feel happy` dalam bahasa Irlandia menjadi `Tá athas orm`. Terjemahan *harfiah* adalah `Happy is upon me`.

Seorang penutur bahasa Irlandia yang menerjemahkan ke bahasa Inggris akan mengatakan `I feel happy`, bukan `Happy is upon me`, karena mereka memahami makna kalimat tersebut, meskipun kata-kata dan struktur kalimatnya berbeda.

Urutan formal untuk kalimat dalam bahasa Irlandia adalah:

- **kata kerja** (Tá atau is)
- **kata sifat** (athas, atau happy)
- **subjek** (orm, atau upon me)

## Terjemahan

Program terjemahan yang naif mungkin hanya menerjemahkan kata-kata, tanpa memperhatikan struktur kalimat.

✅ Jika Anda telah belajar bahasa kedua (atau ketiga atau lebih) sebagai orang dewasa, Anda mungkin memulai dengan berpikir dalam bahasa asli Anda, menerjemahkan konsep kata demi kata di kepala Anda ke bahasa kedua, lalu mengucapkan terjemahan Anda. Ini mirip dengan apa yang dilakukan program terjemahan komputer yang naif. Penting untuk melewati fase ini untuk mencapai kefasihan!

Terjemahan naif menghasilkan terjemahan yang buruk (dan kadang-kadang lucu): `I feel happy` diterjemahkan secara harfiah menjadi `Mise bhraitheann athas` dalam bahasa Irlandia. Itu berarti (secara harfiah) `me feel happy` dan bukan kalimat bahasa Irlandia yang valid. Meskipun bahasa Inggris dan Irlandia adalah bahasa yang digunakan di dua pulau yang berdekatan, mereka adalah bahasa yang sangat berbeda dengan struktur tata bahasa yang berbeda.

> Anda dapat menonton beberapa video tentang tradisi linguistik Irlandia seperti [yang satu ini](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Pendekatan Machine Learning

Sejauh ini, Anda telah belajar tentang pendekatan aturan formal untuk pemrosesan bahasa alami. Pendekatan lain adalah mengabaikan makna kata-kata, dan _sebaliknya menggunakan machine learning untuk mendeteksi pola_. Ini dapat bekerja dalam terjemahan jika Anda memiliki banyak teks (sebuah *corpus*) atau teks (*corpora*) dalam bahasa asal dan target.

Misalnya, pertimbangkan kasus *Pride and Prejudice*, sebuah novel bahasa Inggris terkenal yang ditulis oleh Jane Austen pada tahun 1813. Jika Anda membaca buku tersebut dalam bahasa Inggris dan terjemahan manusia dari buku tersebut dalam bahasa *Prancis*, Anda dapat mendeteksi frasa dalam satu bahasa yang diterjemahkan secara _idiomatik_ ke bahasa lain. Anda akan melakukannya sebentar lagi.

Misalnya, ketika frasa bahasa Inggris seperti `I have no money` diterjemahkan secara harfiah ke bahasa Prancis, itu mungkin menjadi `Je n'ai pas de monnaie`. "Monnaie" adalah 'false cognate' Prancis yang rumit, karena 'money' dan 'monnaie' tidak sinonim. Terjemahan yang lebih baik yang mungkin dibuat oleh manusia adalah `Je n'ai pas d'argent`, karena lebih baik menyampaikan makna bahwa Anda tidak memiliki uang (daripada 'uang receh' yang merupakan arti dari 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Gambar oleh [Jen Looper](https://twitter.com/jenlooper)

Jika model ML memiliki cukup banyak terjemahan manusia untuk membangun model, ia dapat meningkatkan akurasi terjemahan dengan mengidentifikasi pola umum dalam teks yang sebelumnya telah diterjemahkan oleh penutur manusia ahli dari kedua bahasa.

### Latihan - terjemahan

Anda dapat menggunakan `TextBlob` untuk menerjemahkan kalimat. Cobalah kalimat pembuka terkenal dari **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` melakukan pekerjaan yang cukup baik dalam terjemahan: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Dapat dikatakan bahwa terjemahan TextBlob jauh lebih tepat, bahkan dibandingkan dengan terjemahan Prancis tahun 1932 dari buku tersebut oleh V. Leconte dan Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Dalam kasus ini, terjemahan yang diinformasikan oleh ML melakukan pekerjaan yang lebih baik daripada penerjemah manusia yang secara tidak perlu menambahkan kata-kata ke dalam teks asli penulis untuk 'kejelasan'.

> Apa yang terjadi di sini? dan mengapa TextBlob sangat baik dalam terjemahan? Nah, di balik layar, ia menggunakan Google Translate, AI canggih yang mampu menganalisis jutaan frasa untuk memprediksi string terbaik untuk tugas yang sedang dilakukan. Tidak ada yang manual di sini dan Anda memerlukan koneksi internet untuk menggunakan `blob.translate`.

✅ Cobalah beberapa kalimat lagi. Mana yang lebih baik, terjemahan ML atau manusia? Dalam kasus apa?

## Analisis Sentimen

Area lain di mana machine learning dapat bekerja dengan sangat baik adalah analisis sentimen. Pendekatan non-ML untuk sentimen adalah mengidentifikasi kata-kata dan frasa yang 'positif' dan 'negatif'. Kemudian, diberikan teks baru, hitung nilai total kata-kata positif, negatif, dan netral untuk mengidentifikasi sentimen keseluruhan. 

Pendekatan ini mudah tertipu seperti yang mungkin Anda lihat dalam tugas Marvin - kalimat `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` adalah kalimat sentimen negatif yang sarkastik, tetapi algoritma sederhana mendeteksi 'great', 'wonderful', 'glad' sebagai positif dan 'waste', 'lost' dan 'dark' sebagai negatif. Sentimen keseluruhan dipengaruhi oleh kata-kata yang saling bertentangan ini.

✅ Berhenti sejenak dan pikirkan bagaimana kita menyampaikan sarkasme sebagai penutur manusia. Intonasi nada memainkan peran besar. Cobalah mengucapkan frasa "Well, that film was awesome" dengan berbagai cara untuk menemukan bagaimana suara Anda menyampaikan makna.

### Pendekatan ML

Pendekatan ML adalah secara manual mengumpulkan teks negatif dan positif - tweet, atau ulasan film, atau apa pun di mana manusia memberikan skor *dan* opini tertulis. Kemudian teknik NLP dapat diterapkan pada opini dan skor, sehingga pola muncul (misalnya, ulasan film positif cenderung memiliki frasa 'Oscar worthy' lebih sering daripada ulasan film negatif, atau ulasan restoran positif mengatakan 'gourmet' jauh lebih sering daripada 'disgusting').

> ⚖️ **Contoh**: Jika Anda bekerja di kantor seorang politisi dan ada undang-undang baru yang sedang diperdebatkan, konstituen mungkin menulis email ke kantor tersebut untuk mendukung atau menentang undang-undang baru tersebut. Misalkan Anda ditugaskan membaca email dan menyortirnya ke dalam 2 tumpukan, *mendukung* dan *menentang*. Jika ada banyak email, Anda mungkin kewalahan mencoba membaca semuanya. Bukankah akan menyenangkan jika bot dapat membaca semuanya untuk Anda, memahaminya, dan memberi tahu Anda di tumpukan mana setiap email berada? 
> 
> Salah satu cara untuk mencapai itu adalah dengan menggunakan Machine Learning. Anda akan melatih model dengan sebagian email *menentang* dan sebagian email *mendukung*. Model cenderung mengasosiasikan frasa dan kata dengan sisi menentang dan sisi mendukung, *tetapi tidak akan memahami konten apa pun*, hanya bahwa kata-kata dan pola tertentu lebih mungkin muncul dalam email *menentang* atau *mendukung*. Anda dapat mengujinya dengan beberapa email yang belum Anda gunakan untuk melatih model, dan melihat apakah model tersebut sampai pada kesimpulan yang sama seperti Anda. Kemudian, setelah Anda puas dengan akurasi model, Anda dapat memproses email di masa depan tanpa harus membaca setiap email.

✅ Apakah proses ini terdengar seperti proses yang telah Anda gunakan dalam pelajaran sebelumnya?

## Latihan - kalimat sentimental

Sentimen diukur dengan *polaritas* dari -1 hingga 1, yang berarti -1 adalah sentimen paling negatif, dan 1 adalah sentimen paling positif. Sentimen juga diukur dengan skor 0 - 1 untuk objektivitas (0) dan subjektivitas (1).

Lihat kembali *Pride and Prejudice* karya Jane Austen. Teks tersedia di [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Contoh di bawah ini menunjukkan program pendek yang menganalisis sentimen dari kalimat pertama dan terakhir dari buku tersebut dan menampilkan skor polaritas dan subjektivitas/objektivitas sentimennya.

Anda harus menggunakan pustaka `TextBlob` (dijelaskan di atas) untuk menentukan `sentiment` (Anda tidak perlu menulis kalkulator sentimen Anda sendiri) dalam tugas berikut.

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

## Tantangan - periksa polaritas sentimen

Tugas Anda adalah menentukan, menggunakan polaritas sentimen, apakah *Pride and Prejudice* memiliki lebih banyak kalimat yang benar-benar positif daripada yang benar-benar negatif. Untuk tugas ini, Anda dapat mengasumsikan bahwa skor polaritas 1 atau -1 adalah benar-benar positif atau negatif.

**Langkah-langkah:**

1. Unduh [salinan Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) dari Project Gutenberg sebagai file .txt. Hapus metadata di awal dan akhir file, sisakan hanya teks asli
2. Buka file di Python dan ekstrak kontennya sebagai string
3. Buat TextBlob menggunakan string buku
4. Analisis setiap kalimat dalam buku dalam sebuah loop
   1. Jika polaritas adalah 1 atau -1, simpan kalimat tersebut dalam array atau daftar pesan positif atau negatif
5. Di akhir, cetak semua kalimat positif dan negatif (secara terpisah) dan jumlah masing-masing.

Berikut adalah [solusi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) sebagai contoh.

✅ Pengetahuan yang Diperiksa

1. Sentimen didasarkan pada kata-kata yang digunakan dalam kalimat, tetapi apakah kode *memahami* kata-kata tersebut?
2. Apakah Anda pikir polaritas sentimen itu akurat, atau dengan kata lain, apakah Anda *setuju* dengan skor tersebut?
   1. Secara khusus, apakah Anda setuju atau tidak setuju dengan polaritas **positif** absolut dari kalimat berikut?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Tiga kalimat berikut diberi skor dengan sentimen positif absolut, tetapi setelah membaca dengan cermat, mereka bukan kalimat positif. Mengapa analisis sentimen menganggap mereka sebagai kalimat positif?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Apakah Anda setuju atau tidak setuju dengan polaritas **negatif** absolut dari kalimat berikut?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Setiap penggemar Jane Austen akan memahami bahwa dia sering menggunakan bukunya untuk mengkritik aspek-aspek yang lebih konyol dari masyarakat Inggris pada masa Regency. Elizabeth Bennett, karakter utama dalam *Pride and Prejudice*, adalah pengamat sosial yang tajam (seperti penulisnya) dan bahasanya sering kali sangat bernuansa. Bahkan Mr. Darcy (tokoh cinta dalam cerita) mencatat penggunaan bahasa Elizabeth yang penuh permainan dan menggoda: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Tantangan

Bisakah Anda membuat Marvin lebih baik dengan mengekstraksi fitur lain dari input pengguna?

## [Kuis Pasca-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri
Ada banyak cara untuk mengekstrak sentimen dari teks. Pikirkan aplikasi bisnis yang mungkin menggunakan teknik ini. Pikirkan juga bagaimana teknik ini bisa salah. Baca lebih lanjut tentang sistem canggih yang siap digunakan oleh perusahaan untuk menganalisis sentimen seperti [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Uji beberapa kalimat dari Pride and Prejudice di atas dan lihat apakah sistem tersebut dapat mendeteksi nuansa.

## Tugas 

[Lisensi puitis](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.