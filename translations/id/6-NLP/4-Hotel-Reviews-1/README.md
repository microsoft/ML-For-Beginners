<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T20:31:46+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "id"
}
-->
# Analisis Sentimen dengan Ulasan Hotel - Memproses Data

Di bagian ini, Anda akan menggunakan teknik yang telah dipelajari di pelajaran sebelumnya untuk melakukan analisis data eksplorasi pada dataset besar. Setelah memahami kegunaan berbagai kolom, Anda akan belajar:

- cara menghapus kolom yang tidak diperlukan
- cara menghitung data baru berdasarkan kolom yang ada
- cara menyimpan dataset hasil untuk digunakan dalam tantangan akhir

## [Kuis Pra-Pelajaran](https://ff-quizzes.netlify.app/en/ml/)

### Pendahuluan

Sejauh ini Anda telah mempelajari bahwa data teks sangat berbeda dengan data numerik. Jika teks tersebut ditulis atau diucapkan oleh manusia, teks tersebut dapat dianalisis untuk menemukan pola dan frekuensi, sentimen, dan makna. Pelajaran ini membawa Anda ke dalam dataset nyata dengan tantangan nyata: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** yang mencakup [lisensi CC0: Domain Publik](https://creativecommons.org/publicdomain/zero/1.0/). Dataset ini diambil dari Booking.com dari sumber publik. Pembuat dataset ini adalah Jiashen Liu.

### Persiapan

Anda akan membutuhkan:

* Kemampuan untuk menjalankan notebook .ipynb menggunakan Python 3
* pandas
* NLTK, [yang harus Anda instal secara lokal](https://www.nltk.org/install.html)
* Dataset yang tersedia di Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ukurannya sekitar 230 MB setelah diekstrak. Unduh ke folder root `/data` yang terkait dengan pelajaran NLP ini.

## Analisis Data Eksplorasi

Tantangan ini mengasumsikan bahwa Anda sedang membangun bot rekomendasi hotel menggunakan analisis sentimen dan skor ulasan tamu. Dataset yang akan Anda gunakan mencakup ulasan dari 1493 hotel berbeda di 6 kota.

Menggunakan Python, dataset ulasan hotel, dan analisis sentimen NLTK, Anda dapat menemukan:

* Apa kata dan frasa yang paling sering digunakan dalam ulasan?
* Apakah *tag* resmi yang menggambarkan hotel berkorelasi dengan skor ulasan (misalnya, apakah ulasan lebih negatif untuk hotel tertentu oleh *Keluarga dengan anak kecil* dibandingkan oleh *Pelancong solo*, mungkin menunjukkan bahwa hotel tersebut lebih cocok untuk *Pelancong solo*)?
* Apakah skor sentimen NLTK 'sesuai' dengan skor numerik dari pengulas hotel?

#### Dataset

Mari kita eksplorasi dataset yang telah Anda unduh dan simpan secara lokal. Buka file tersebut di editor seperti VS Code atau bahkan Excel.

Header dalam dataset adalah sebagai berikut:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Berikut adalah pengelompokan yang mungkin lebih mudah untuk diperiksa: 
##### Kolom Hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Dengan menggunakan *lat* dan *lng*, Anda dapat membuat peta dengan Python yang menunjukkan lokasi hotel (mungkin diberi kode warna untuk ulasan negatif dan positif)
  * Hotel_Address tidak terlalu berguna bagi kita, dan kemungkinan akan diganti dengan negara untuk mempermudah penyortiran & pencarian

**Kolom Meta-Ulasan Hotel**

* `Average_Score`
  * Menurut pembuat dataset, kolom ini adalah *Skor Rata-rata hotel, dihitung berdasarkan komentar terbaru dalam satu tahun terakhir*. Ini tampaknya cara yang tidak biasa untuk menghitung skor, tetapi ini adalah data yang diambil, jadi kita mungkin menerimanya apa adanya untuk saat ini.
  
  ✅ Berdasarkan kolom lain dalam data ini, dapatkah Anda memikirkan cara lain untuk menghitung skor rata-rata?

* `Total_Number_of_Reviews`
  * Jumlah total ulasan yang diterima hotel ini - tidak jelas (tanpa menulis kode) apakah ini merujuk pada ulasan dalam dataset.
* `Additional_Number_of_Scoring`
  * Ini berarti skor ulasan diberikan tetapi tidak ada ulasan positif atau negatif yang ditulis oleh pengulas

**Kolom Ulasan**

- `Reviewer_Score`
  - Ini adalah nilai numerik dengan maksimal 1 tempat desimal antara nilai minimum dan maksimum 2.5 dan 10
  - Tidak dijelaskan mengapa 2.5 adalah skor terendah yang mungkin
- `Negative_Review`
  - Jika pengulas tidak menulis apa pun, kolom ini akan memiliki "**No Negative**"
  - Perhatikan bahwa pengulas mungkin menulis ulasan positif di kolom ulasan negatif (misalnya, "tidak ada yang buruk tentang hotel ini")
- `Review_Total_Negative_Word_Counts`
  - Jumlah kata negatif yang lebih tinggi menunjukkan skor yang lebih rendah (tanpa memeriksa sentimen)
- `Positive_Review`
  - Jika pengulas tidak menulis apa pun, kolom ini akan memiliki "**No Positive**"
  - Perhatikan bahwa pengulas mungkin menulis ulasan negatif di kolom ulasan positif (misalnya, "tidak ada yang baik tentang hotel ini sama sekali")
- `Review_Total_Positive_Word_Counts`
  - Jumlah kata positif yang lebih tinggi menunjukkan skor yang lebih tinggi (tanpa memeriksa sentimen)
- `Review_Date` dan `days_since_review`
  - Ukuran kesegaran atau keusangan mungkin diterapkan pada ulasan (ulasan yang lebih lama mungkin tidak seakurat ulasan yang lebih baru karena manajemen hotel berubah, atau renovasi telah dilakukan, atau kolam renang ditambahkan, dll.)
- `Tags`
  - Ini adalah deskriptor pendek yang mungkin dipilih pengulas untuk menggambarkan jenis tamu mereka (misalnya, solo atau keluarga), jenis kamar yang mereka miliki, lama menginap, dan bagaimana ulasan dikirimkan.
  - Sayangnya, menggunakan tag ini bermasalah, lihat bagian di bawah yang membahas kegunaannya

**Kolom Pengulas**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ini mungkin menjadi faktor dalam model rekomendasi, misalnya, jika Anda dapat menentukan bahwa pengulas yang lebih produktif dengan ratusan ulasan lebih cenderung negatif daripada positif. Namun, pengulas dari ulasan tertentu tidak diidentifikasi dengan kode unik, dan oleh karena itu tidak dapat dikaitkan dengan satu set ulasan. Ada 30 pengulas dengan 100 atau lebih ulasan, tetapi sulit untuk melihat bagaimana ini dapat membantu model rekomendasi.
- `Reviewer_Nationality`
  - Beberapa orang mungkin berpikir bahwa kebangsaan tertentu lebih cenderung memberikan ulasan positif atau negatif karena kecenderungan nasional. Berhati-hatilah membangun pandangan anekdotal seperti itu ke dalam model Anda. Ini adalah stereotip nasional (dan terkadang rasial), dan setiap pengulas adalah individu yang menulis ulasan berdasarkan pengalaman mereka. Ulasan tersebut mungkin telah difilter melalui banyak lensa seperti pengalaman hotel sebelumnya, jarak yang ditempuh, dan temperamen pribadi mereka. Berpikir bahwa kebangsaan mereka adalah alasan untuk skor ulasan sulit untuk dibenarkan.

##### Contoh

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Ini saat ini bukan hotel tetapi situs konstruksi Saya diteror sejak pagi hari dan sepanjang hari dengan kebisingan bangunan yang tidak dapat diterima saat beristirahat setelah perjalanan panjang dan bekerja di kamar Orang-orang bekerja sepanjang hari dengan alat berat di kamar sebelah Saya meminta perubahan kamar tetapi tidak ada kamar yang tenang tersedia Untuk membuat keadaan lebih buruk saya dikenakan biaya berlebih Saya check out di malam hari karena saya harus pergi penerbangan sangat pagi dan menerima tagihan yang sesuai Sehari kemudian hotel membuat biaya tambahan tanpa persetujuan saya melebihi harga yang dipesan Tempat yang mengerikan Jangan menyiksa diri dengan memesan di sini | Tidak ada Tempat yang mengerikan Jauhi | Perjalanan bisnis Pasangan Kamar Double Standar Menginap 2 malam |

Seperti yang Anda lihat, tamu ini tidak memiliki pengalaman menginap yang menyenangkan di hotel ini. Hotel ini memiliki skor rata-rata yang baik yaitu 7.8 dan 1945 ulasan, tetapi pengulas ini memberikan skor 2.5 dan menulis 115 kata tentang betapa negatifnya pengalaman mereka. Jika mereka tidak menulis apa pun di kolom Positive_Review, Anda mungkin menyimpulkan bahwa tidak ada yang positif, tetapi ternyata mereka menulis 7 kata peringatan. Jika kita hanya menghitung kata-kata tanpa memperhatikan makna atau sentimen kata-kata tersebut, kita mungkin memiliki pandangan yang bias tentang maksud pengulas. Anehnya, skor mereka 2.5 membingungkan, karena jika pengalaman menginap di hotel itu sangat buruk, mengapa memberikan poin sama sekali? Dengan menyelidiki dataset secara mendalam, Anda akan melihat bahwa skor terendah yang mungkin adalah 2.5, bukan 0. Skor tertinggi yang mungkin adalah 10.

##### Tags

Seperti disebutkan di atas, sekilas, ide untuk menggunakan `Tags` untuk mengkategorikan data masuk akal. Sayangnya, tag ini tidak distandarisasi, yang berarti bahwa di hotel tertentu, opsinya mungkin *Single room*, *Twin room*, dan *Double room*, tetapi di hotel berikutnya, opsinya adalah *Deluxe Single Room*, *Classic Queen Room*, dan *Executive King Room*. Ini mungkin hal yang sama, tetapi ada begitu banyak variasi sehingga pilihannya menjadi:

1. Mencoba mengubah semua istilah menjadi satu standar, yang sangat sulit, karena tidak jelas apa jalur konversi dalam setiap kasus (misalnya, *Classic single room* dipetakan ke *Single room* tetapi *Superior Queen Room with Courtyard Garden or City View* jauh lebih sulit untuk dipetakan)

1. Kita dapat mengambil pendekatan NLP dan mengukur frekuensi istilah tertentu seperti *Solo*, *Business Traveller*, atau *Family with young kids* saat mereka berlaku untuk setiap hotel, dan memasukkan itu ke dalam rekomendasi  

Tags biasanya (tetapi tidak selalu) merupakan satu bidang yang berisi daftar 5 hingga 6 nilai yang dipisahkan koma yang sesuai dengan *Jenis perjalanan*, *Jenis tamu*, *Jenis kamar*, *Jumlah malam*, dan *Jenis perangkat ulasan dikirimkan*. Namun, karena beberapa pengulas tidak mengisi setiap bidang (mereka mungkin meninggalkan satu kosong), nilai-nilai tersebut tidak selalu dalam urutan yang sama.

Sebagai contoh, ambil *Jenis grup*. Ada 1025 kemungkinan unik di bidang ini dalam kolom `Tags`, dan sayangnya hanya beberapa di antaranya yang merujuk pada grup (beberapa adalah jenis kamar, dll.). Jika Anda memfilter hanya yang menyebutkan keluarga, hasilnya berisi banyak hasil tipe *Family room*. Jika Anda menyertakan istilah *dengan*, yaitu menghitung nilai *Family with*, hasilnya lebih baik, dengan lebih dari 80.000 dari 515.000 hasil yang berisi frasa "Family with young children" atau "Family with older children".

Ini berarti kolom tags tidak sepenuhnya tidak berguna bagi kita, tetapi akan membutuhkan beberapa pekerjaan untuk membuatnya berguna.

##### Skor Rata-rata Hotel

Ada sejumlah keanehan atau ketidaksesuaian dengan dataset yang tidak dapat saya pahami, tetapi diilustrasikan di sini agar Anda menyadarinya saat membangun model Anda. Jika Anda memahaminya, beri tahu kami di bagian diskusi!

Dataset memiliki kolom berikut yang berkaitan dengan skor rata-rata dan jumlah ulasan: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel tunggal dengan jumlah ulasan terbanyak dalam dataset ini adalah *Britannia International Hotel Canary Wharf* dengan 4789 ulasan dari 515.000. Tetapi jika kita melihat nilai `Total_Number_of_Reviews` untuk hotel ini, nilainya adalah 9086. Anda mungkin menyimpulkan bahwa ada banyak skor tanpa ulasan, jadi mungkin kita harus menambahkan nilai kolom `Additional_Number_of_Scoring`. Nilai tersebut adalah 2682, dan menambahkannya ke 4789 menghasilkan 7471 yang masih kurang 1615 dari `Total_Number_of_Reviews`. 

Jika Anda mengambil kolom `Average_Score`, Anda mungkin menyimpulkan bahwa itu adalah rata-rata dari ulasan dalam dataset, tetapi deskripsi dari Kaggle adalah "*Skor Rata-rata hotel, dihitung berdasarkan komentar terbaru dalam satu tahun terakhir*". Itu tampaknya tidak terlalu berguna, tetapi kita dapat menghitung rata-rata kita sendiri berdasarkan skor ulasan dalam dataset. Menggunakan hotel yang sama sebagai contoh, skor rata-rata hotel diberikan sebagai 7.1 tetapi skor yang dihitung (rata-rata skor pengulas *dalam* dataset) adalah 6.8. Ini mendekati, tetapi bukan nilai yang sama, dan kita hanya dapat menebak bahwa skor yang diberikan dalam ulasan `Additional_Number_of_Scoring` meningkatkan rata-rata menjadi 7.1. Sayangnya tanpa cara untuk menguji atau membuktikan asumsi tersebut, sulit untuk menggunakan atau mempercayai `Average_Score`, `Additional_Number_of_Scoring`, dan `Total_Number_of_Reviews` ketika mereka didasarkan pada, atau merujuk pada, data yang tidak kita miliki.

Untuk memperumit masalah lebih lanjut, hotel dengan jumlah ulasan tertinggi kedua memiliki skor rata-rata yang dihitung sebesar 8.12 dan dataset `Average_Score` adalah 8.1. Apakah skor ini benar adalah kebetulan atau apakah hotel pertama adalah ketidaksesuaian? 

Dengan kemungkinan bahwa hotel-hotel ini mungkin merupakan outlier, dan bahwa mungkin sebagian besar nilai sesuai (tetapi beberapa tidak karena alasan tertentu), kita akan menulis program pendek berikutnya untuk mengeksplorasi nilai-nilai dalam dataset dan menentukan penggunaan yang benar (atau tidak digunakan) dari nilai-nilai tersebut.
> 🚨 Catatan penting
>
> Saat bekerja dengan dataset ini, Anda akan menulis kode yang menghitung sesuatu dari teks tanpa harus membaca atau menganalisis teks itu sendiri. Inilah inti dari NLP, menafsirkan makna atau sentimen tanpa harus melibatkan manusia secara langsung. Namun, ada kemungkinan Anda akan membaca beberapa ulasan negatif. Saya menyarankan Anda untuk tidak melakukannya, karena Anda tidak perlu. Beberapa ulasan tersebut mungkin konyol atau tidak relevan, seperti ulasan negatif tentang hotel yang menyebutkan "Cuacanya tidak bagus", sesuatu yang berada di luar kendali hotel, atau bahkan siapa pun. Tetapi ada sisi gelap dari beberapa ulasan juga. Kadang-kadang ulasan negatif mengandung unsur rasis, seksis, atau diskriminasi usia. Hal ini sangat disayangkan tetapi dapat dipahami mengingat dataset ini diambil dari situs web publik. Beberapa pengulas meninggalkan ulasan yang mungkin Anda anggap tidak menyenangkan, membuat tidak nyaman, atau bahkan mengganggu. Lebih baik biarkan kode yang mengukur sentimen daripada membacanya sendiri dan merasa terganggu. Meski demikian, hanya sebagian kecil yang menulis hal-hal semacam itu, tetapi mereka tetap ada.
## Latihan - Eksplorasi Data
### Memuat Data

Cukup sudah memeriksa data secara visual, sekarang saatnya menulis kode dan mendapatkan jawaban! Bagian ini menggunakan pustaka pandas. Tugas pertama Anda adalah memastikan bahwa Anda dapat memuat dan membaca data CSV. Pustaka pandas memiliki loader CSV yang cepat, dan hasilnya ditempatkan dalam dataframe, seperti yang telah dipelajari di pelajaran sebelumnya. CSV yang kita muat memiliki lebih dari setengah juta baris, tetapi hanya 17 kolom. Pandas memberikan banyak cara yang kuat untuk berinteraksi dengan dataframe, termasuk kemampuan untuk melakukan operasi pada setiap baris.

Mulai dari sini dalam pelajaran ini, akan ada potongan kode dan beberapa penjelasan tentang kode serta diskusi tentang apa arti hasilnya. Gunakan _notebook.ipynb_ yang disertakan untuk kode Anda.

Mari kita mulai dengan memuat file data yang akan Anda gunakan:

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

Setelah data dimuat, kita dapat melakukan beberapa operasi pada data tersebut. Simpan kode ini di bagian atas program Anda untuk bagian berikutnya.

## Eksplorasi Data

Dalam kasus ini, data sudah *bersih*, artinya data siap untuk digunakan dan tidak memiliki karakter dalam bahasa lain yang dapat mengganggu algoritma yang hanya mengharapkan karakter dalam bahasa Inggris.

✅ Anda mungkin harus bekerja dengan data yang memerlukan beberapa pemrosesan awal untuk memformatnya sebelum menerapkan teknik NLP, tetapi tidak kali ini. Jika Anda harus melakukannya, bagaimana Anda akan menangani karakter non-Inggris?

Luangkan waktu untuk memastikan bahwa setelah data dimuat, Anda dapat mengeksplorasinya dengan kode. Sangat mudah untuk ingin fokus pada kolom `Negative_Review` dan `Positive_Review`. Kolom-kolom tersebut diisi dengan teks alami untuk diproses oleh algoritma NLP Anda. Tapi tunggu! Sebelum Anda melompat ke NLP dan analisis sentimen, Anda harus mengikuti kode di bawah ini untuk memastikan apakah nilai-nilai yang diberikan dalam dataset sesuai dengan nilai-nilai yang Anda hitung dengan pandas.

## Operasi Dataframe

Tugas pertama dalam pelajaran ini adalah memeriksa apakah pernyataan berikut benar dengan menulis beberapa kode yang memeriksa dataframe (tanpa mengubahnya).

> Seperti banyak tugas pemrograman, ada beberapa cara untuk menyelesaikannya, tetapi saran yang baik adalah melakukannya dengan cara yang paling sederhana dan mudah, terutama jika akan lebih mudah dipahami saat Anda kembali ke kode ini di masa depan. Dengan dataframe, ada API yang komprehensif yang sering kali memiliki cara untuk melakukan apa yang Anda inginkan secara efisien.

Anggap pertanyaan berikut sebagai tugas pemrograman dan coba jawab tanpa melihat solusinya.

1. Cetak *shape* dari dataframe yang baru saja Anda muat (shape adalah jumlah baris dan kolom).
2. Hitung frekuensi untuk kebangsaan reviewer:
   1. Berapa banyak nilai yang berbeda untuk kolom `Reviewer_Nationality` dan apa saja nilainya?
   2. Kebangsaan reviewer mana yang paling umum dalam dataset (cetak negara dan jumlah ulasan)?
   3. Apa 10 kebangsaan yang paling sering ditemukan berikutnya, dan hitungan frekuensinya?
3. Hotel mana yang paling sering diulas untuk masing-masing dari 10 kebangsaan reviewer teratas?
4. Berapa banyak ulasan per hotel (hitungan frekuensi hotel) dalam dataset?
5. Meskipun ada kolom `Average_Score` untuk setiap hotel dalam dataset, Anda juga dapat menghitung skor rata-rata (mengambil rata-rata dari semua skor reviewer dalam dataset untuk setiap hotel). Tambahkan kolom baru ke dataframe Anda dengan header kolom `Calc_Average_Score` yang berisi rata-rata yang dihitung tersebut.
6. Apakah ada hotel yang memiliki `Average_Score` dan `Calc_Average_Score` yang sama (dibulatkan ke 1 tempat desimal)?
   1. Coba tulis fungsi Python yang mengambil Series (baris) sebagai argumen dan membandingkan nilainya, mencetak pesan saat nilainya tidak sama. Kemudian gunakan metode `.apply()` untuk memproses setiap baris dengan fungsi tersebut.
7. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Negative_Review` berupa "No Negative".
8. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Positive_Review` berupa "No Positive".
9. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Positive_Review` berupa "No Positive" **dan** nilai kolom `Negative_Review` berupa "No Negative".

### Jawaban Kode

1. Cetak *shape* dari dataframe yang baru saja Anda muat (shape adalah jumlah baris dan kolom).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Hitung frekuensi untuk kebangsaan reviewer:

   1. Berapa banyak nilai yang berbeda untuk kolom `Reviewer_Nationality` dan apa saja nilainya?
   2. Kebangsaan reviewer mana yang paling umum dalam dataset (cetak negara dan jumlah ulasan)?

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

   3. Apa 10 kebangsaan yang paling sering ditemukan berikutnya, dan hitungan frekuensinya?

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

3. Hotel mana yang paling sering diulas untuk masing-masing dari 10 kebangsaan reviewer teratas?

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

4. Berapa banyak ulasan per hotel (hitungan frekuensi hotel) dalam dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   Anda mungkin menyadari bahwa hasil *counted in the dataset* tidak sesuai dengan nilai di `Total_Number_of_Reviews`. Tidak jelas apakah nilai ini dalam dataset mewakili jumlah total ulasan yang dimiliki hotel, tetapi tidak semuanya di-scrape, atau perhitungan lainnya. `Total_Number_of_Reviews` tidak digunakan dalam model karena ketidakjelasan ini.

5. Meskipun ada kolom `Average_Score` untuk setiap hotel dalam dataset, Anda juga dapat menghitung skor rata-rata (mengambil rata-rata dari semua skor reviewer dalam dataset untuk setiap hotel). Tambahkan kolom baru ke dataframe Anda dengan header kolom `Calc_Average_Score` yang berisi rata-rata yang dihitung tersebut. Cetak kolom `Hotel_Name`, `Average_Score`, dan `Calc_Average_Score`.

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

   Anda mungkin juga bertanya-tanya tentang nilai `Average_Score` dan mengapa kadang-kadang berbeda dari skor rata-rata yang dihitung. Karena kita tidak dapat mengetahui mengapa beberapa nilai cocok, tetapi yang lain memiliki perbedaan, yang paling aman dalam kasus ini adalah menggunakan skor ulasan yang kita miliki untuk menghitung rata-rata sendiri. Namun demikian, perbedaannya biasanya sangat kecil, berikut adalah hotel dengan deviasi terbesar dari rata-rata dataset dan rata-rata yang dihitung:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Dengan hanya 1 hotel yang memiliki perbedaan skor lebih besar dari 1, ini berarti kita mungkin dapat mengabaikan perbedaan tersebut dan menggunakan skor rata-rata yang dihitung.

6. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Negative_Review` berupa "No Negative".

7. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Positive_Review` berupa "No Positive".

8. Hitung dan cetak berapa banyak baris yang memiliki nilai kolom `Positive_Review` berupa "No Positive" **dan** nilai kolom `Negative_Review` berupa "No Negative".

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

Cara lain untuk menghitung item tanpa Lambda, dan menggunakan sum untuk menghitung baris:

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

   Anda mungkin menyadari bahwa ada 127 baris yang memiliki nilai "No Negative" dan "No Positive" untuk kolom `Negative_Review` dan `Positive_Review` masing-masing. Artinya, reviewer memberikan skor numerik kepada hotel, tetapi tidak menulis ulasan positif maupun negatif. Untungnya ini adalah jumlah baris yang kecil (127 dari 515738, atau 0,02%), sehingga kemungkinan besar tidak akan memengaruhi model atau hasil kita ke arah tertentu, tetapi Anda mungkin tidak mengharapkan dataset ulasan memiliki baris tanpa ulasan, jadi ini layak untuk dieksplorasi.

Setelah Anda mengeksplorasi dataset, dalam pelajaran berikutnya Anda akan memfilter data dan menambahkan analisis sentimen.

---
## 🚀Tantangan

Pelajaran ini menunjukkan, seperti yang kita lihat di pelajaran sebelumnya, betapa pentingnya memahami data Anda dan kekurangannya sebelum melakukan operasi pada data tersebut. Data berbasis teks, khususnya, memerlukan pengamatan yang cermat. Telusuri berbagai dataset yang kaya teks dan lihat apakah Anda dapat menemukan area yang dapat memperkenalkan bias atau sentimen yang menyimpang ke dalam model.

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tinjauan & Studi Mandiri

Ikuti [Learning Path tentang NLP ini](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) untuk menemukan alat yang dapat dicoba saat membangun model yang kaya teks dan ucapan.

## Tugas 

[NLTK](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.