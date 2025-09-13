<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T20:32:48+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "ms"
}
-->
# Analisis Sentimen dengan Ulasan Hotel - Memproses Data

Dalam bahagian ini, anda akan menggunakan teknik yang dipelajari dalam pelajaran sebelumnya untuk melakukan analisis data eksploratori pada set data yang besar. Setelah anda memahami kegunaan pelbagai kolum dengan baik, anda akan belajar:

- cara membuang kolum yang tidak diperlukan
- cara mengira data baru berdasarkan kolum sedia ada
- cara menyimpan set data yang dihasilkan untuk digunakan dalam cabaran akhir

## [Kuiz Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

### Pengenalan

Setakat ini, anda telah mempelajari bagaimana data teks sangat berbeza daripada jenis data numerik. Jika ia adalah teks yang ditulis atau diucapkan oleh manusia, ia boleh dianalisis untuk mencari pola dan frekuensi, sentimen, dan makna. Pelajaran ini membawa anda ke dalam set data sebenar dengan cabaran sebenar: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** yang termasuk [lesen CC0: Domain Awam](https://creativecommons.org/publicdomain/zero/1.0/). Data ini diambil dari Booking.com daripada sumber awam. Pencipta set data ini ialah Jiashen Liu.

### Persediaan

Anda akan memerlukan:

* Keupayaan untuk menjalankan notebook .ipynb menggunakan Python 3
* pandas
* NLTK, [yang perlu anda pasang secara tempatan](https://www.nltk.org/install.html)
* Set data yang tersedia di Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Saiznya sekitar 230 MB selepas dinyahzip. Muat turun ke folder root `/data` yang berkaitan dengan pelajaran NLP ini.

## Analisis Data Eksploratori

Cabaran ini mengandaikan bahawa anda sedang membina bot cadangan hotel menggunakan analisis sentimen dan skor ulasan tetamu. Set data yang akan anda gunakan termasuk ulasan 1493 hotel yang berbeza di 6 bandar.

Menggunakan Python, set data ulasan hotel, dan analisis sentimen NLTK, anda boleh mengetahui:

* Apakah kata-kata dan frasa yang paling kerap digunakan dalam ulasan?
* Adakah *tag* rasmi yang menerangkan hotel berkorelasi dengan skor ulasan (contohnya, adakah ulasan lebih negatif untuk hotel tertentu oleh *Keluarga dengan anak kecil* berbanding *Pengembara solo*, mungkin menunjukkan ia lebih sesuai untuk *Pengembara solo*?)
* Adakah skor sentimen NLTK 'bersetuju' dengan skor numerik pengulas hotel?

#### Set Data

Mari kita terokai set data yang telah anda muat turun dan simpan secara tempatan. Buka fail ini dalam editor seperti VS Code atau Excel.

Header dalam set data adalah seperti berikut:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Berikut adalah pengelompokan yang mungkin lebih mudah untuk diperiksa: 
##### Kolum Hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Menggunakan *lat* dan *lng*, anda boleh memplot peta dengan Python yang menunjukkan lokasi hotel (mungkin dengan kod warna untuk ulasan negatif dan positif)
  * Hotel_Address tidak begitu berguna kepada kita, dan kita mungkin menggantikannya dengan negara untuk memudahkan pengisihan & pencarian

**Kolum Meta-Ulasan Hotel**

* `Average_Score`
  * Menurut pencipta set data, kolum ini adalah *Skor Purata hotel, dikira berdasarkan ulasan terbaru dalam setahun terakhir*. Ini kelihatan seperti cara yang tidak biasa untuk mengira skor, tetapi ia adalah data yang diambil, jadi kita mungkin menerimanya buat masa ini.
  
  ✅ Berdasarkan kolum lain dalam data ini, bolehkah anda memikirkan cara lain untuk mengira skor purata?

* `Total_Number_of_Reviews`
  * Jumlah ulasan yang diterima oleh hotel ini - tidak jelas (tanpa menulis kod) sama ada ini merujuk kepada ulasan dalam set data.
* `Additional_Number_of_Scoring`
  * Ini bermaksud skor ulasan diberikan tetapi tiada ulasan positif atau negatif ditulis oleh pengulas

**Kolum Ulasan**

- `Reviewer_Score`
  - Ini adalah nilai numerik dengan maksimum 1 tempat perpuluhan antara nilai minimum dan maksimum 2.5 dan 10
  - Tidak dijelaskan mengapa 2.5 adalah skor terendah yang mungkin
- `Negative_Review`
  - Jika pengulas tidak menulis apa-apa, medan ini akan mempunyai "**No Negative**"
  - Perhatikan bahawa pengulas mungkin menulis ulasan positif dalam kolum ulasan negatif (contohnya, "tiada apa yang buruk tentang hotel ini")
- `Review_Total_Negative_Word_Counts`
  - Jumlah kata negatif yang lebih tinggi menunjukkan skor yang lebih rendah (tanpa memeriksa sentimen)
- `Positive_Review`
  - Jika pengulas tidak menulis apa-apa, medan ini akan mempunyai "**No Positive**"
  - Perhatikan bahawa pengulas mungkin menulis ulasan negatif dalam kolum ulasan positif (contohnya, "tiada apa yang baik tentang hotel ini sama sekali")
- `Review_Total_Positive_Word_Counts`
  - Jumlah kata positif yang lebih tinggi menunjukkan skor yang lebih tinggi (tanpa memeriksa sentimen)
- `Review_Date` dan `days_since_review`
  - Ukuran kesegaran atau keusangan mungkin diterapkan pada ulasan (ulasan lama mungkin tidak seakurat ulasan baru kerana pengurusan hotel berubah, atau renovasi telah dilakukan, atau kolam renang ditambah dll.)
- `Tags`
  - Ini adalah deskriptor pendek yang mungkin dipilih oleh pengulas untuk menerangkan jenis tetamu mereka (contohnya, solo atau keluarga), jenis bilik yang mereka miliki, tempoh penginapan dan bagaimana ulasan dikirimkan.
  - Malangnya, menggunakan tag ini bermasalah, lihat bahagian di bawah yang membincangkan kegunaannya

**Kolum Pengulas**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ini mungkin menjadi faktor dalam model cadangan, contohnya, jika anda dapat menentukan bahawa pengulas yang lebih prolifik dengan ratusan ulasan lebih cenderung negatif daripada positif. Walau bagaimanapun, pengulas mana-mana ulasan tertentu tidak dikenal pasti dengan kod unik, dan oleh itu tidak dapat dihubungkan dengan satu set ulasan. Terdapat 30 pengulas dengan 100 atau lebih ulasan, tetapi sukar untuk melihat bagaimana ini dapat membantu model cadangan.
- `Reviewer_Nationality`
  - Sesetengah orang mungkin berpendapat bahawa kebangsaan tertentu lebih cenderung memberikan ulasan positif atau negatif kerana kecenderungan kebangsaan. Berhati-hati membina pandangan anekdot seperti ini ke dalam model anda. Ini adalah stereotaip kebangsaan (dan kadangkala perkauman), dan setiap pengulas adalah individu yang menulis ulasan berdasarkan pengalaman mereka. Ia mungkin telah ditapis melalui banyak lensa seperti penginapan hotel sebelumnya, jarak perjalanan, dan temperamen peribadi mereka. Berfikir bahawa kebangsaan mereka adalah sebab skor ulasan adalah sukar untuk dibenarkan.

##### Contoh

| Skor Purata | Jumlah Ulasan | Skor Pengulas | Ulasan <br />Negatif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Ulasan Positif                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Ini bukan hotel tetapi tapak pembinaan Saya diganggu dari awal pagi dan sepanjang hari dengan bunyi pembinaan yang tidak dapat diterima semasa berehat selepas perjalanan jauh dan bekerja di bilik Orang bekerja sepanjang hari dengan alat pemecah di bilik bersebelahan Saya meminta pertukaran bilik tetapi tiada bilik senyap tersedia Untuk membuat keadaan lebih buruk saya dikenakan bayaran berlebihan Saya keluar pada waktu petang kerana saya perlu meninggalkan penerbangan awal dan menerima bil yang sesuai Sehari kemudian hotel membuat caj lain tanpa persetujuan saya melebihi harga yang ditempah Tempat yang mengerikan Jangan hukum diri anda dengan menempah di sini | Tiada apa Tempat yang mengerikan Jauhkan diri | Perjalanan perniagaan Pasangan Bilik Double Standard Menginap 2 malam |

Seperti yang anda lihat, tetamu ini tidak mempunyai pengalaman yang baik di hotel ini. Hotel ini mempunyai skor purata yang baik iaitu 7.8 dan 1945 ulasan, tetapi pengulas ini memberikannya 2.5 dan menulis 115 kata tentang betapa negatifnya penginapan mereka. Jika mereka tidak menulis apa-apa dalam kolum Ulasan_Positive, anda mungkin menganggap tiada apa yang positif, tetapi mereka menulis 7 kata amaran. Jika kita hanya mengira kata-kata tanpa makna, atau sentimen kata-kata, kita mungkin mempunyai pandangan yang berat sebelah terhadap niat pengulas. Anehnya, skor mereka 2.5 membingungkan, kerana jika penginapan hotel itu sangat buruk, mengapa memberikan sebarang mata? Menyelidiki set data dengan teliti, anda akan melihat bahawa skor terendah yang mungkin adalah 2.5, bukan 0. Skor tertinggi yang mungkin adalah 10.

##### Tags

Seperti yang disebutkan di atas, pada pandangan pertama, idea untuk menggunakan `Tags` untuk mengkategorikan data masuk akal. Malangnya, tag ini tidak diseragamkan, yang bermaksud bahawa dalam hotel tertentu, opsinya mungkin *Single room*, *Twin room*, dan *Double room*, tetapi dalam hotel berikutnya, ia adalah *Deluxe Single Room*, *Classic Queen Room*, dan *Executive King Room*. Ini mungkin perkara yang sama, tetapi terdapat begitu banyak variasi sehingga pilihannya menjadi:

1. Cuba menukar semua istilah kepada satu standard, yang sangat sukar, kerana tidak jelas apa laluan penukaran dalam setiap kes (contohnya, *Classic single room* dipetakan kepada *Single room* tetapi *Superior Queen Room with Courtyard Garden or City View* jauh lebih sukar untuk dipetakan)

1. Kita boleh mengambil pendekatan NLP dan mengukur frekuensi istilah tertentu seperti *Solo*, *Business Traveller*, atau *Family with young kids* seperti yang berlaku pada setiap hotel, dan faktor itu ke dalam cadangan  

Tags biasanya (tetapi tidak selalu) adalah satu medan yang mengandungi senarai 5 hingga 6 nilai yang dipisahkan koma yang sejajar dengan *Jenis perjalanan*, *Jenis tetamu*, *Jenis bilik*, *Bilangan malam*, dan *Jenis peranti ulasan dikirimkan*. Walau bagaimanapun, kerana sesetengah pengulas tidak mengisi setiap medan (mereka mungkin meninggalkan satu kosong), nilai-nilai tidak selalu dalam urutan yang sama.

Sebagai contoh, ambil *Jenis kumpulan*. Terdapat 1025 kemungkinan unik dalam medan ini dalam kolum `Tags`, dan malangnya hanya sebahagian daripadanya merujuk kepada kumpulan (sesetengah adalah jenis bilik dll.). Jika anda menapis hanya yang menyebut keluarga, hasilnya mengandungi banyak hasil jenis *Family room*. Jika anda memasukkan istilah *with*, iaitu mengira nilai *Family with*, hasilnya lebih baik, dengan lebih daripada 80,000 daripada 515,000 hasil mengandungi frasa "Family with young children" atau "Family with older children".

Ini bermakna kolum tags tidak sepenuhnya tidak berguna kepada kita, tetapi ia memerlukan usaha untuk menjadikannya berguna.

##### Skor Purata Hotel

Terdapat beberapa keanehan atau percanggahan dengan set data yang saya tidak dapat memahami, tetapi digambarkan di sini supaya anda menyedarinya semasa membina model anda. Jika anda memahaminya, sila maklumkan kepada kami dalam bahagian perbincangan!

Set data mempunyai kolum berikut yang berkaitan dengan skor purata dan jumlah ulasan: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel tunggal dengan jumlah ulasan terbanyak dalam set data ini ialah *Britannia International Hotel Canary Wharf* dengan 4789 ulasan daripada 515,000. Tetapi jika kita melihat nilai `Total_Number_of_Reviews` untuk hotel ini, ia adalah 9086. Anda mungkin mengandaikan bahawa terdapat banyak lagi skor tanpa ulasan, jadi mungkin kita harus menambah nilai kolum `Additional_Number_of_Scoring`. Nilai itu adalah 2682, dan menambahnya kepada 4789 memberikan kita 7,471 yang masih kekurangan 1615 daripada `Total_Number_of_Reviews`. 

Jika anda mengambil kolum `Average_Score`, anda mungkin mengandaikan ia adalah purata ulasan dalam set data, tetapi deskripsi dari Kaggle adalah "*Skor Purata hotel, dikira berdasarkan ulasan terbaru dalam setahun terakhir*". Itu tidak kelihatan begitu berguna, tetapi kita boleh mengira purata kita sendiri berdasarkan skor ulasan dalam set data. Menggunakan hotel yang sama sebagai contoh, skor purata hotel diberikan sebagai 7.1 tetapi skor yang dikira (purata skor pengulas *dalam* set data) adalah 6.8. Ini hampir sama, tetapi bukan nilai yang sama, dan kita hanya boleh mengandaikan bahawa skor yang diberikan dalam ulasan `Additional_Number_of_Scoring` meningkatkan purata kepada 7.1. Malangnya tanpa cara untuk menguji atau membuktikan andaian itu, sukar untuk menggunakan atau mempercayai `Average_Score`, `Additional_Number_of_Scoring` dan `Total_Number_of_Reviews` apabila ia berdasarkan, atau merujuk kepada, data yang kita tidak miliki.

Untuk merumitkan lagi, hotel dengan jumlah ulasan kedua tertinggi mempunyai skor purata yang dikira sebanyak 8.12 dan `Average_Score` dalam set data adalah 8.1. Adakah skor ini kebetulan atau adakah hotel pertama adalah percanggahan? 

Dengan kemungkinan bahawa hotel ini mungkin adalah outlier, dan mungkin kebanyakan nilai sesuai (tetapi beberapa tidak atas sebab tertentu), kita akan menulis program pendek seterusnya untuk meneroka nilai dalam set data dan menentukan penggunaan yang betul (atau tidak) nilai-nilai tersebut.
> 🚨 Nota penting  
>  
> Apabila bekerja dengan dataset ini, anda akan menulis kod yang mengira sesuatu daripada teks tanpa perlu membaca atau menganalisis teks itu sendiri. Inilah intipati NLP, mentafsirkan makna atau sentimen tanpa memerlukan manusia melakukannya. Walau bagaimanapun, ada kemungkinan anda akan membaca beberapa ulasan negatif. Saya menggesa anda untuk tidak melakukannya, kerana anda tidak perlu. Sebahagian daripadanya adalah remeh, atau ulasan negatif hotel yang tidak relevan, seperti "Cuaca tidak bagus", sesuatu yang di luar kawalan hotel, atau sesiapa pun. Tetapi ada sisi gelap kepada beberapa ulasan juga. Kadang-kadang ulasan negatif bersifat perkauman, seksis, atau diskriminasi umur. Ini adalah sesuatu yang malang tetapi dijangka dalam dataset yang diambil dari laman web awam. Sesetengah pengulas meninggalkan ulasan yang mungkin anda anggap tidak menyenangkan, tidak selesa, atau menyedihkan. Lebih baik biarkan kod mengukur sentimen daripada membacanya sendiri dan merasa terganggu. Walau bagaimanapun, hanya sebilangan kecil yang menulis perkara sedemikian, tetapi mereka tetap wujud.
## Latihan - Penerokaan Data
### Muatkan Data

Cukup sudah memeriksa data secara visual, sekarang anda akan menulis kod dan mendapatkan jawapan! Bahagian ini menggunakan pustaka pandas. Tugas pertama anda adalah memastikan anda boleh memuatkan dan membaca data CSV. Pustaka pandas mempunyai pemuat CSV yang pantas, dan hasilnya diletakkan dalam dataframe, seperti dalam pelajaran sebelumnya. CSV yang kita muatkan mempunyai lebih setengah juta baris, tetapi hanya 17 lajur. Pandas memberikan banyak cara yang berkuasa untuk berinteraksi dengan dataframe, termasuk keupayaan untuk melakukan operasi pada setiap baris.

Mulai dari sini dalam pelajaran ini, akan ada potongan kod dan beberapa penjelasan tentang kod serta perbincangan tentang apa maksud hasilnya. Gunakan _notebook.ipynb_ yang disertakan untuk kod anda.

Mari kita mulakan dengan memuatkan fail data yang akan anda gunakan:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Sekarang data telah dimuatkan, kita boleh melakukan beberapa operasi ke atasnya. Simpan kod ini di bahagian atas program anda untuk bahagian seterusnya.

## Terokai Data

Dalam kes ini, data sudah *bersih*, yang bermaksud ia sedia untuk digunakan dan tidak mempunyai aksara dalam bahasa lain yang mungkin mengganggu algoritma yang hanya mengharapkan aksara bahasa Inggeris.

✅ Anda mungkin perlu bekerja dengan data yang memerlukan pemprosesan awal untuk memformatnya sebelum menerapkan teknik NLP, tetapi tidak kali ini. Jika anda perlu, bagaimana anda akan menangani aksara bukan bahasa Inggeris?

Luangkan masa untuk memastikan bahawa setelah data dimuatkan, anda boleh menerokainya dengan kod. Sangat mudah untuk ingin fokus pada lajur `Negative_Review` dan `Positive_Review`. Lajur-lajur ini dipenuhi dengan teks semula jadi untuk algoritma NLP anda proseskan. Tetapi tunggu! Sebelum anda melompat ke NLP dan analisis sentimen, anda harus mengikuti kod di bawah untuk memastikan nilai yang diberikan dalam dataset sepadan dengan nilai yang anda kira menggunakan pandas.

## Operasi Dataframe

Tugas pertama dalam pelajaran ini adalah memeriksa sama ada penegasan berikut adalah betul dengan menulis beberapa kod yang memeriksa dataframe (tanpa mengubahnya).

> Seperti banyak tugas pengaturcaraan, terdapat beberapa cara untuk menyelesaikannya, tetapi nasihat yang baik adalah melakukannya dengan cara yang paling mudah dan senang, terutamanya jika ia akan lebih mudah difahami apabila anda kembali kepada kod ini di masa depan. Dengan dataframe, terdapat API yang komprehensif yang sering mempunyai cara untuk melakukan apa yang anda mahu dengan cekap.

Anggap soalan berikut sebagai tugas pengkodan dan cuba jawab tanpa melihat penyelesaiannya.

1. Cetak *bentuk* dataframe yang baru sahaja anda muatkan (bentuk adalah bilangan baris dan lajur).
2. Kira kekerapan untuk kebangsaan pengulas:
   1. Berapa banyak nilai yang berbeza untuk lajur `Reviewer_Nationality` dan apakah nilai-nilai tersebut?
   2. Kebangsaan pengulas mana yang paling biasa dalam dataset (cetak negara dan bilangan ulasan)?
   3. Apakah 10 kebangsaan yang paling kerap ditemui seterusnya, dan kiraan kekerapan mereka?
3. Apakah hotel yang paling kerap diulas untuk setiap 10 kebangsaan pengulas teratas?
4. Berapa banyak ulasan per hotel (kekerapan ulasan hotel) dalam dataset?
5. Walaupun terdapat lajur `Average_Score` untuk setiap hotel dalam dataset, anda juga boleh mengira skor purata (mengambil purata semua skor pengulas dalam dataset untuk setiap hotel). Tambahkan lajur baru ke dataframe anda dengan tajuk lajur `Calc_Average_Score` yang mengandungi purata yang dikira itu.
6. Adakah mana-mana hotel mempunyai `Average_Score` dan `Calc_Average_Score` yang sama (dibundarkan kepada 1 tempat perpuluhan)?
   1. Cuba tulis fungsi Python yang mengambil Siri (baris) sebagai argumen dan membandingkan nilai-nilai tersebut, mencetak mesej apabila nilai-nilai tidak sama. Kemudian gunakan kaedah `.apply()` untuk memproses setiap baris dengan fungsi tersebut.
7. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Negative_Review` "No Negative".
8. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive".
9. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive" **dan** nilai lajur `Negative_Review` "No Negative".

### Jawapan Kod

1. Cetak *bentuk* dataframe yang baru sahaja anda muatkan (bentuk adalah bilangan baris dan lajur).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Kira kekerapan untuk kebangsaan pengulas:

   1. Berapa banyak nilai yang berbeza untuk lajur `Reviewer_Nationality` dan apakah nilai-nilai tersebut?
   2. Kebangsaan pengulas mana yang paling biasa dalam dataset (cetak negara dan bilangan ulasan)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Apakah 10 kebangsaan yang paling kerap ditemui seterusnya, dan kiraan kekerapan mereka?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Apakah hotel yang paling kerap diulas untuk setiap 10 kebangsaan pengulas teratas?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Berapa banyak ulasan per hotel (kekerapan ulasan hotel) dalam dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Nama_Hotel                 | Jumlah_Ulasan_Hotel | Jumlah_Ulasan_Ditemui |
   | :----------------------------------------: | :------------------: | :-------------------: |
   | Britannia International Hotel Canary Wharf |         9086         |         4789          |
   |    Park Plaza Westminster Bridge London    |         12158        |         4169          |
   |   Copthorne Tara Hotel London Kensington   |         7105         |         3578          |
   |                    ...                     |          ...         |          ...          |
   |       Mercure Paris Porte d Orleans        |          110         |          10           |
   |                Hotel Wagner                |          135         |          10           |
   |            Hotel Gallitzinberg             |          173         |           8           |

   Anda mungkin perasan bahawa hasil *dikira dalam dataset* tidak sepadan dengan nilai dalam `Total_Number_of_Reviews`. Tidak jelas sama ada nilai ini dalam dataset mewakili jumlah ulasan hotel yang sebenar, tetapi tidak semua telah diambil, atau pengiraan lain. `Total_Number_of_Reviews` tidak digunakan dalam model kerana ketidakjelasan ini.

5. Walaupun terdapat lajur `Average_Score` untuk setiap hotel dalam dataset, anda juga boleh mengira skor purata (mengambil purata semua skor pengulas dalam dataset untuk setiap hotel). Tambahkan lajur baru ke dataframe anda dengan tajuk lajur `Calc_Average_Score` yang mengandungi purata yang dikira itu. Cetak lajur `Hotel_Name`, `Average_Score`, dan `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Anda mungkin juga tertanya-tanya tentang nilai `Average_Score` dan mengapa ia kadang-kadang berbeza daripada skor purata yang dikira. Oleh kerana kita tidak dapat mengetahui mengapa sesetengah nilai sepadan, tetapi yang lain mempunyai perbezaan, adalah lebih selamat dalam kes ini untuk menggunakan skor ulasan yang kita ada untuk mengira purata sendiri. Walau bagaimanapun, perbezaannya biasanya sangat kecil, berikut adalah hotel dengan penyimpangan terbesar daripada purata dataset dan purata yang dikira:

   | Perbezaan_Skor_Purata | Skor_Purata | Skor_Purata_Dikira |                                  Nama_Hotel |
   | :--------------------: | :---------: | :----------------: | --------------------------------------------: |
   |          -0.8          |     7.7     |        8.5         |                  Best Western Hotel Astoria |
   |          -0.7          |     8.8     |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |          -0.7          |     7.5     |        8.2         |               Mercure Paris Porte d Orleans |
   |          -0.7          |     7.9     |        8.6         |             Renaissance Paris Vendome Hotel |
   |          -0.5          |     7.0     |        7.5         |                         Hotel Royal Elys es |
   |          ...           |     ...     |        ...         |                                         ... |
   |          0.7           |     7.5     |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |          0.8           |     7.1     |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |          0.9           |     6.8     |        5.9         |                               Villa Eugenie |
   |          0.9           |     8.6     |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |          1.3           |     7.2     |        5.9         |                          Kube Hotel Ice Bar |

   Dengan hanya 1 hotel mempunyai perbezaan skor lebih daripada 1, ini bermakna kita mungkin boleh mengabaikan perbezaan dan menggunakan skor purata yang dikira.

6. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Negative_Review` "No Negative".

7. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive".

8. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive" **dan** nilai lajur `Negative_Review` "No Negative".

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Cara Lain

Cara lain untuk mengira item tanpa Lambdas, dan gunakan sum untuk mengira baris:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Anda mungkin perasan bahawa terdapat 127 baris yang mempunyai kedua-dua nilai "No Negative" dan "No Positive" untuk lajur `Negative_Review` dan `Positive_Review` masing-masing. Ini bermakna pengulas memberikan hotel skor numerik, tetapi enggan menulis ulasan positif atau negatif. Nasib baik ini adalah jumlah baris yang kecil (127 daripada 515738, atau 0.02%), jadi ia mungkin tidak akan mempengaruhi model atau hasil kita dalam arah tertentu, tetapi anda mungkin tidak menjangkakan dataset ulasan mempunyai baris tanpa ulasan, jadi ia berbaloi untuk meneroka data untuk menemui baris seperti ini.

Sekarang anda telah meneroka dataset, dalam pelajaran seterusnya anda akan menapis data dan menambah analisis sentimen.

---
## 🚀Cabaran

Pelajaran ini menunjukkan, seperti yang kita lihat dalam pelajaran sebelumnya, betapa pentingnya memahami data anda dan keanehannya sebelum melakukan operasi ke atasnya. Data berasaskan teks, khususnya, memerlukan pemeriksaan yang teliti. Terokai pelbagai dataset yang berat dengan teks dan lihat jika anda boleh menemui kawasan yang boleh memperkenalkan bias atau sentimen yang tidak seimbang ke dalam model.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Kajian & Pembelajaran Kendiri

Ambil [Laluan Pembelajaran ini tentang NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) untuk menemui alat yang boleh dicuba semasa membina model yang berat dengan ucapan dan teks.

## Tugasan 

[NLTK](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.