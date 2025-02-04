# Analisis Sentimen dengan Ulasan Hotel - Memproses Data

Dalam bagian ini, Anda akan menggunakan teknik yang telah dipelajari dalam pelajaran sebelumnya untuk melakukan analisis data eksplorasi pada dataset besar. Setelah Anda memiliki pemahaman yang baik tentang kegunaan berbagai kolom, Anda akan belajar:

- bagaimana menghapus kolom yang tidak diperlukan
- bagaimana menghitung beberapa data baru berdasarkan kolom yang ada
- bagaimana menyimpan dataset yang dihasilkan untuk digunakan dalam tantangan akhir

## [Kuis Pra-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Pengenalan

Sejauh ini Anda telah belajar tentang bagaimana data teks sangat berbeda dengan jenis data numerik. Jika itu adalah teks yang ditulis atau diucapkan oleh manusia, dapat dianalisis untuk menemukan pola dan frekuensi, sentimen, dan makna. Pelajaran ini membawa Anda ke dalam dataset nyata dengan tantangan nyata: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** dan termasuk [lisensi CC0: Domain Publik](https://creativecommons.org/publicdomain/zero/1.0/). Dataset ini diambil dari Booking.com dari sumber publik. Pembuat dataset ini adalah Jiashen Liu.

### Persiapan

Anda akan membutuhkan:

* Kemampuan untuk menjalankan notebook .ipynb menggunakan Python 3
* pandas
* NLTK, [yang harus Anda instal secara lokal](https://www.nltk.org/install.html)
* Dataset yang tersedia di Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ukurannya sekitar 230 MB setelah diekstrak. Unduh ke folder root `/data` yang terkait dengan pelajaran NLP ini.

## Analisis data eksplorasi

Tantangan ini mengasumsikan bahwa Anda sedang membangun bot rekomendasi hotel menggunakan analisis sentimen dan skor ulasan tamu. Dataset yang akan Anda gunakan mencakup ulasan dari 1493 hotel berbeda di 6 kota.

Menggunakan Python, dataset ulasan hotel, dan analisis sentimen NLTK Anda bisa mengetahui:

* Apa kata dan frasa yang paling sering digunakan dalam ulasan?
* Apakah *tag* resmi yang menggambarkan hotel berkorelasi dengan skor ulasan (misalnya apakah ulasan lebih negatif untuk hotel tertentu untuk *Keluarga dengan anak kecil* dibandingkan dengan *Pelancong Solo*, mungkin menunjukkan bahwa hotel tersebut lebih baik untuk *Pelancong Solo*?)
* Apakah skor sentimen NLTK 'setuju' dengan skor numerik ulasan hotel?

#### Dataset

Mari kita jelajahi dataset yang telah Anda unduh dan simpan secara lokal. Buka file dalam editor seperti VS Code atau bahkan Excel.

Header dalam dataset adalah sebagai berikut:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Berikut ini mereka dikelompokkan dengan cara yang mungkin lebih mudah untuk diperiksa:
##### Kolom Hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Menggunakan *lat* dan *lng* Anda bisa memetakan lokasi hotel dengan Python (mungkin diberi kode warna untuk ulasan negatif dan positif)
  * Hotel_Address tidak jelas bermanfaat bagi kita, dan kita mungkin akan menggantinya dengan negara untuk memudahkan pengurutan & pencarian

**Kolom Meta-ulasan Hotel**

* `Average_Score`
  * Menurut pembuat dataset, kolom ini adalah *Skor Rata-rata hotel, dihitung berdasarkan komentar terbaru dalam setahun terakhir*. Ini tampaknya cara yang tidak biasa untuk menghitung skor, tetapi ini adalah data yang diambil sehingga kita mungkin menerimanya apa adanya untuk saat ini.
  
  ✅ Berdasarkan kolom lain dalam data ini, dapatkah Anda memikirkan cara lain untuk menghitung skor rata-rata?

* `Total_Number_of_Reviews`
  * Jumlah total ulasan yang diterima hotel ini - tidak jelas (tanpa menulis beberapa kode) apakah ini mengacu pada ulasan dalam dataset.
* `Additional_Number_of_Scoring`
  * Ini berarti skor ulasan diberikan tetapi tidak ada ulasan positif atau negatif yang ditulis oleh pengulas

**Kolom Ulasan**

- `Reviewer_Score`
  - Ini adalah nilai numerik dengan paling banyak 1 tempat desimal antara nilai minimum dan maksimum 2.5 dan 10
  - Tidak dijelaskan mengapa 2.5 adalah skor terendah yang mungkin
- `Negative_Review`
  - Jika seorang pengulas tidak menulis apa-apa, kolom ini akan berisi "**No Negative**"
  - Perhatikan bahwa seorang pengulas mungkin menulis ulasan positif di kolom Ulasan Negatif (misalnya "tidak ada yang buruk tentang hotel ini")
- `Review_Total_Negative_Word_Counts`
  - Jumlah kata negatif yang lebih tinggi menunjukkan skor yang lebih rendah (tanpa memeriksa sentimen)
- `Positive_Review`
  - Jika seorang pengulas tidak menulis apa-apa, kolom ini akan berisi "**No Positive**"
  - Perhatikan bahwa seorang pengulas mungkin menulis ulasan negatif di kolom Ulasan Positif (misalnya "tidak ada yang baik tentang hotel ini sama sekali")
- `Review_Total_Positive_Word_Counts`
  - Jumlah kata positif yang lebih tinggi menunjukkan skor yang lebih tinggi (tanpa memeriksa sentimen)
- `Review_Date` dan `days_since_review`
  - Sebuah ukuran kesegaran atau ketidakaktualan dapat diterapkan pada ulasan (ulasan yang lebih lama mungkin tidak seakurat yang lebih baru karena manajemen hotel berubah, atau renovasi telah dilakukan, atau kolam renang ditambahkan, dll.)
- `Tags`
  - Ini adalah deskriptor pendek yang mungkin dipilih pengulas untuk menggambarkan jenis tamu mereka (misalnya solo atau keluarga), jenis kamar yang mereka miliki, lama menginap dan bagaimana ulasan diajukan.
  - Sayangnya, menggunakan tag ini bermasalah, lihat bagian di bawah yang membahas kegunaannya

**Kolom Pengulas**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ini mungkin menjadi faktor dalam model rekomendasi, misalnya, jika Anda dapat menentukan bahwa pengulas yang lebih produktif dengan ratusan ulasan lebih cenderung negatif daripada positif. Namun, pengulas dari ulasan tertentu tidak diidentifikasi dengan kode unik, dan oleh karena itu tidak dapat dikaitkan dengan satu set ulasan. Ada 30 pengulas dengan 100 atau lebih ulasan, tetapi sulit untuk melihat bagaimana ini dapat membantu model rekomendasi.
- `Reviewer_Nationality`
  - Beberapa orang mungkin berpikir bahwa kebangsaan tertentu lebih cenderung memberikan ulasan positif atau negatif karena kecenderungan nasional. Berhati-hatilah membangun pandangan anekdotal seperti itu ke dalam model Anda. Ini adalah stereotip nasional (dan terkadang rasial), dan setiap pengulas adalah individu yang menulis ulasan berdasarkan pengalaman mereka. Mungkin telah disaring melalui banyak lensa seperti pengalaman hotel sebelumnya, jarak yang ditempuh, dan temperamen pribadi mereka. Berpikir bahwa kebangsaan mereka adalah alasan untuk skor ulasan sulit untuk dibenarkan.

##### Contoh

| Skor Rata-rata | Total Jumlah Ulasan | Skor Pengulas | Ulasan <br />Negatif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Ulasan Positif                   | Tag                                                                                       |
| -------------- | ------------------- | ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                | 2.5           | Saat ini ini bukan hotel tetapi situs konstruksi Saya diteror dari pagi hari dan sepanjang hari dengan kebisingan bangunan yang tidak dapat diterima saat beristirahat setelah perjalanan panjang dan bekerja di kamar Orang-orang bekerja sepanjang hari dengan palu di kamar yang bersebelahan Saya meminta untuk pindah kamar tetapi tidak ada kamar yang tenang tersedia Untuk memperburuk keadaan saya dikenakan biaya lebih Saya check out di malam hari karena saya harus berangkat sangat pagi penerbangan dan menerima tagihan yang sesuai Sehari kemudian hotel membuat biaya lain tanpa persetujuan saya melebihi harga yang dipesan Ini tempat yang mengerikan Jangan menghukum diri Anda dengan memesan di sini | Tidak ada tempat yang mengerikan | Perjalanan bisnis Pasangan Kamar Double Standar Menginap 2 malam |

Seperti yang Anda lihat, tamu ini tidak memiliki pengalaman menginap yang menyenangkan di hotel ini. Hotel ini memiliki skor rata-rata yang baik yaitu 7.8 dan 1945 ulasan, tetapi pengulas ini memberikan skor 2.5 dan menulis 115 kata tentang betapa negatifnya pengalaman mereka. Jika mereka tidak menulis apa pun di kolom Ulasan_Positive, Anda mungkin menyimpulkan bahwa tidak ada yang positif, tetapi sayangnya mereka menulis 7 kata peringatan. Jika kita hanya menghitung kata-kata alih-alih makna, atau sentimen dari kata-kata tersebut, kita mungkin memiliki pandangan yang miring tentang niat pengulas. Anehnya, skor mereka yang 2.5 membingungkan, karena jika pengalaman menginap di hotel itu sangat buruk, mengapa memberikan poin sama sekali? Menyelidiki dataset dengan cermat, Anda akan melihat bahwa skor terendah yang mungkin adalah 2.5, bukan 0. Skor tertinggi yang mungkin adalah 10.

##### Tag

Seperti disebutkan di atas, pada pandangan pertama, ide untuk menggunakan `Tags` untuk mengkategorikan data masuk akal. Sayangnya, tag ini tidak distandarisasi, yang berarti bahwa di hotel tertentu, opsinya mungkin *Kamar Single*, *Kamar Twin*, dan *Kamar Double*, tetapi di hotel berikutnya, mereka adalah *Kamar Single Deluxe*, *Kamar Queen Klasik*, dan *Kamar King Eksekutif*. Ini mungkin hal yang sama, tetapi ada begitu banyak variasi sehingga pilihannya menjadi:

1. Mencoba mengubah semua istilah menjadi satu standar, yang sangat sulit, karena tidak jelas apa jalur konversinya dalam setiap kasus (misalnya *Kamar single klasik* dipetakan ke *Kamar single* tetapi *Kamar Queen Superior dengan Pemandangan Taman atau Kota* jauh lebih sulit untuk dipetakan)

1. Kita dapat mengambil pendekatan NLP dan mengukur frekuensi istilah tertentu seperti *Solo*, *Pelancong Bisnis*, atau *Keluarga dengan anak kecil* karena berlaku untuk setiap hotel, dan memasukkan itu ke dalam rekomendasi  

Tag biasanya (tetapi tidak selalu) merupakan satu bidang yang berisi daftar 5 hingga 6 nilai yang dipisahkan dengan koma yang sesuai dengan *Jenis perjalanan*, *Jenis tamu*, *Jenis kamar*, *Jumlah malam*, dan *Jenis perangkat yang digunakan untuk mengirimkan ulasan*. Namun, karena beberapa pengulas tidak mengisi setiap bidang (mereka mungkin meninggalkan satu kosong), nilainya tidak selalu dalam urutan yang sama.

Sebagai contoh, ambil *Jenis grup*. Ada 1025 kemungkinan unik dalam bidang ini di kolom `Tags`, dan sayangnya hanya beberapa dari mereka yang merujuk pada grup (beberapa adalah jenis kamar, dll.). Jika Anda memfilter hanya yang menyebutkan keluarga, hasilnya mengandung banyak hasil tipe *Kamar keluarga*. Jika Anda memasukkan istilah *dengan*, yaitu menghitung nilai *Keluarga dengan*, hasilnya lebih baik, dengan lebih dari 80.000 dari 515.000 hasil yang mengandung frasa "Keluarga dengan anak kecil" atau "Keluarga dengan anak yang lebih tua".

Ini berarti kolom tag tidak sepenuhnya tidak berguna bagi kita, tetapi akan membutuhkan beberapa pekerjaan untuk membuatnya berguna.

##### Skor rata-rata hotel

Ada sejumlah keanehan atau ketidaksesuaian dengan dataset yang tidak bisa saya pahami, tetapi diilustrasikan di sini sehingga Anda menyadarinya saat membangun model Anda. Jika Anda mengetahuinya, beri tahu kami di bagian diskusi!

Dataset ini memiliki kolom berikut yang berkaitan dengan skor rata-rata dan jumlah ulasan:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel tunggal dengan ulasan terbanyak dalam dataset ini adalah *Britannia International Hotel Canary Wharf* dengan 4789 ulasan dari 515.000. Tetapi jika kita melihat nilai `Total_Number_of_Reviews` untuk hotel ini, nilainya adalah 9086. Anda mungkin menyimpulkan bahwa ada banyak skor tanpa ulasan, jadi mungkin kita harus menambahkan nilai kolom `Additional_Number_of_Scoring`. Nilai itu adalah 2682, dan menambahkannya ke 4789 memberi kita 7.471 yang masih 1615 kurang dari `Total_Number_of_Reviews`.

Jika Anda mengambil kolom `Average_Score`, Anda mungkin menyimpulkan bahwa itu adalah rata-rata dari ulasan dalam dataset, tetapi deskripsi dari Kaggle adalah "*Skor Rata-rata hotel, dihitung berdasarkan komentar terbaru dalam setahun terakhir*". Itu tampaknya tidak terlalu berguna, tetapi kita dapat menghitung rata-rata kita sendiri berdasarkan skor ulasan dalam dataset. Menggunakan hotel yang sama sebagai contoh, skor rata-rata hotel yang diberikan adalah 7.1 tetapi skor yang dihitung (rata-rata skor pengulas *dalam* dataset) adalah 6.8. Ini dekat, tetapi bukan nilai yang sama, dan kita hanya bisa menebak bahwa skor yang diberikan dalam ulasan `Additional_Number_of_Scoring` meningkatkan rata-rata menjadi 7.1. Sayangnya, tanpa cara untuk menguji atau membuktikan pernyataan tersebut, sulit untuk menggunakan atau mempercayai `Average_Score`, `Additional_Number_of_Scoring` dan `Total_Number_of_Reviews` ketika mereka didasarkan pada, atau merujuk pada, data yang tidak kita miliki.

Untuk memperumit masalah lebih lanjut, hotel dengan jumlah ulasan tertinggi kedua memiliki skor rata-rata yang dihitung sebesar 8.12 dan dataset `Average_Score` adalah 8.1. Apakah skor yang benar ini kebetulan atau apakah hotel pertama adalah ketidaksesuaian?

Dengan kemungkinan bahwa hotel ini mungkin merupakan outlier, dan mungkin sebagian besar nilai cocok (tetapi beberapa tidak karena alasan tertentu) kita akan menulis program singkat berikutnya untuk mengeksplorasi nilai-nilai dalam dataset dan menentukan penggunaan yang benar (atau tidak- penggunaan) dari nilai-nilai tersebut.

> 🚨 Sebuah catatan peringatan
>
> Saat bekerja dengan dataset ini, Anda akan menulis kode yang menghitung sesuatu dari teks tanpa harus membaca atau menganalisis teks sendiri. Ini adalah esensi dari NLP, menafsirkan makna atau sentimen tanpa harus ada manusia yang melakukannya. Namun, ada kemungkinan Anda akan membaca beberapa ulasan negatif. Saya akan menyarankan Anda untuk tidak melakukannya, karena Anda tidak perlu. Beberapa dari mereka konyol, atau ulasan hotel negatif yang tidak relevan, seperti "Cuacanya tidak bagus", sesuatu yang di luar kendali hotel, atau siapa pun. Tetapi ada sisi gelap dari beberapa ulasan juga. Terkadang ulasan negatif bersifat rasis, seksis, atau ageis. Ini tidak menyenangkan tetapi diharapkan dalam dataset yang diambil dari situs web publik. Beberapa pengulas meninggalkan ulasan yang Anda anggap tidak menyenangkan, tidak nyaman, atau mengganggu. Lebih baik biarkan kode yang mengukur sentimen daripada membaca sendiri dan merasa tidak nyaman. Yang mengatakan, itu adalah minoritas yang menulis hal-hal seperti itu, tetapi mereka tetap ada.

## Latihan - Eksplorasi Data
### Memuat data

Itu cukup memeriksa data secara visual, sekarang Anda akan menulis beberapa kode dan mendapatkan beberapa jawaban! Bagian ini menggunakan perpustakaan pandas. Tugas pertama Anda adalah memastikan Anda dapat memuat dan membaca data CSV. Perpustakaan pandas memiliki pemuat CSV yang cepat, dan hasilnya ditempatkan dalam dataframe, seperti pada pelajaran sebelumnya. CSV yang kita muat memiliki lebih dari setengah juta baris, tetapi hanya 17 kolom. Pandas memberi Anda banyak cara yang kuat untuk berinteraksi dengan dataframe, termasuk kemampuan untuk melakukan operasi pada setiap baris.

Dari sini di pelajaran ini, akan ada cuplikan kode dan beberapa penjelasan tentang kode dan beberapa diskusi tentang apa arti hasilnya. Gunakan _notebook.ipynb_ yang disertakan untuk kode Anda.

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

Sekarang data sudah dimuat, kita bisa melakukan beberapa operasi pada data tersebut. Simpan kode ini di bagian atas program Anda untuk bagian selanjutnya.

## Jelajahi data

Dalam hal ini, datanya sudah *bersih*, artinya siap untuk digunakan, dan tidak memiliki karakter dalam bahasa lain yang mungkin mengganggu algoritma yang hanya mengharapkan karakter bahasa Inggris.

✅ Anda mungkin harus bekerja dengan data yang memerlukan beberapa pemrosesan awal untuk memformatnya sebelum menerapkan teknik NLP, tetapi tidak kali ini. Jika Anda harus melakukannya, bagaimana Anda akan menangani karakter non-Inggris?

Luangkan waktu sejenak untuk memastikan bahwa setelah data dimuat, Anda dapat menjelajahinya dengan kode. Sangat mudah untuk ingin fokus pada kolom `Negative_Review` dan `Positive_Review`. Mereka penuh dengan teks alami untuk diproses oleh algoritma NLP Anda. Tapi tunggu! Sebelum Anda terjun ke NLP dan sentimen, Anda harus mengikuti kode di bawah ini untuk memastikan bahwa nilai yang diberikan dalam dataset sesuai dengan nilai yang Anda hitung dengan pandas.

## Operasi Dataframe

Tugas pertama dalam pelajaran ini adalah memeriksa apakah pernyataan berikut benar dengan menulis beberapa kode yang memeriksa data frame (tanpa mengubahnya).

> Seperti banyak tugas pemrograman, ada beberapa cara untuk menyelesaikannya, tetapi saran yang baik adalah melakukannya dengan cara yang paling sederhana dan termudah yang Anda bisa, terutama jika akan lebih mudah untuk dipahami saat Anda kembali ke kode ini di masa mendatang. Dengan dataframes, ada API yang komprehensif yang sering kali memiliki cara untuk melakukan apa yang Anda inginkan secara efisien.
Perlakukan pertanyaan-pertanyaan berikut sebagai tugas pemrograman dan cobalah menjawabnya tanpa melihat solusinya. 1. Cetak *shape* dari data frame yang baru saja Anda muat (shape adalah jumlah baris dan kolom) 2. Hitung frekuensi untuk kebangsaan pengulas: 1. Berapa banyak nilai berbeda yang ada untuk
baris mempunyai nilai lajur `Positive_Review` "No Positive" 9. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive" **dan** nilai lajur `Negative_Review` "No Negative" ### Jawapan Kod 1. Cetak *bentuk* data frame yang baru sahaja dimuatkan (bentuk adalah bilangan baris dan lajur) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Kira jumlah kekerapan untuk kewarganegaraan pengulas: 1. Berapa banyak nilai yang berbeza ada untuk lajur `Reviewer_Nationality` dan apakah ia? 2. Apakah kewarganegaraan pengulas yang paling umum dalam dataset (cetak negara dan bilangan ulasan)? ```python
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
   ``` 3. Apakah 10 kewarganegaraan yang paling kerap ditemui seterusnya, dan jumlah kekerapan mereka? ```python
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
      ``` 3. Apakah hotel yang paling kerap diulas untuk setiap 10 kewarganegaraan pengulas teratas? ```python
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
   ``` 4. Berapa banyak ulasan terdapat bagi setiap hotel (jumlah kekerapan hotel) dalam dataset? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | Anda mungkin perasan bahawa hasil *yang dikira dalam dataset* tidak sepadan dengan nilai dalam `Total_Number_of_Reviews`. Tidak jelas jika nilai dalam dataset ini mewakili jumlah ulasan yang hotel ada, tetapi tidak semua dikikis, atau beberapa pengiraan lain. `Total_Number_of_Reviews` tidak digunakan dalam model kerana ketidakjelasan ini. 5. Walaupun terdapat lajur `Average_Score` untuk setiap hotel dalam dataset, anda juga boleh mengira skor purata (mendapatkan purata semua skor pengulas dalam dataset untuk setiap hotel). Tambah lajur baru kepada dataframe anda dengan tajuk lajur `Calc_Average_Score` yang mengandungi purata yang dikira itu. Cetak lajur `Hotel_Name`, `Average_Score`, dan `Calc_Average_Score`. ```python
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
   ``` Anda mungkin juga tertanya-tanya tentang nilai `Average_Score` dan mengapa ia kadang-kadang berbeza daripada skor purata yang dikira. Oleh kerana kita tidak tahu mengapa beberapa nilai sepadan, tetapi yang lain mempunyai perbezaan, adalah lebih selamat dalam kes ini untuk menggunakan skor ulasan yang kita ada untuk mengira purata sendiri. Walau bagaimanapun, perbezaan biasanya sangat kecil, berikut adalah hotel dengan perbezaan terbesar dari purata dataset dan purata yang dikira: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | Dengan hanya 1 hotel yang mempunyai perbezaan skor lebih daripada 1, ini bermakna kita mungkin boleh mengabaikan perbezaan tersebut dan menggunakan skor purata yang dikira. 6. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Negative_Review` "No Negative" 7. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive" 8. Kira dan cetak berapa banyak baris yang mempunyai nilai lajur `Positive_Review` "No Positive" **dan** nilai lajur `Negative_Review` "No Negative" ```python
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
   ``` ## Cara lain Cara lain untuk mengira item tanpa Lambdas, dan menggunakan sum untuk mengira baris: ```python
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
   ``` Anda mungkin perasan bahawa terdapat 127 baris yang mempunyai kedua-dua nilai "No Negative" dan "No Positive" untuk lajur `Negative_Review` dan `Positive_Review` masing-masing. Ini bermakna pengulas memberikan skor numerik kepada hotel, tetapi enggan menulis ulasan positif atau negatif. Nasib baik ini adalah jumlah baris yang kecil (127 daripada 515738, atau 0.02%), jadi mungkin tidak akan menjejaskan model atau keputusan kita dalam mana-mana arah tertentu, tetapi anda mungkin tidak menjangkakan dataset ulasan mempunyai baris tanpa ulasan, jadi ia patut meneroka data untuk menemui baris seperti ini. Sekarang setelah anda meneroka dataset, dalam pelajaran seterusnya anda akan menapis data dan menambah beberapa analisis sentimen. --- ## 🚀Cabaran Pelajaran ini menunjukkan, seperti yang kita lihat dalam pelajaran sebelumnya, betapa pentingnya untuk memahami data anda dan kekurangannya sebelum melakukan operasi ke atasnya. Data berasaskan teks, khususnya, memerlukan pemeriksaan yang teliti. Selidiki pelbagai set data yang berat dengan teks dan lihat jika anda dapat menemui kawasan yang boleh memperkenalkan bias atau sentimen yang menyimpang ke dalam model. ## [Kuiz selepas kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Kajian & Pembelajaran Kendiri Ambil [Laluan Pembelajaran ini mengenai NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) untuk menemui alat yang boleh dicuba semasa membina model yang berat dengan ucapan dan teks. ## Tugasan [NLTK](assignment.md)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.