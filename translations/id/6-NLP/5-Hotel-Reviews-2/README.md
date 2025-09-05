<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T20:45:25+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "id"
}
-->
# Analisis Sentimen dengan Ulasan Hotel

Setelah Anda menjelajahi dataset secara mendetail, sekarang saatnya untuk memfilter kolom dan menggunakan teknik NLP pada dataset untuk mendapatkan wawasan baru tentang hotel.

## [Kuis Pra-Kuliah](https://ff-quizzes.netlify.app/en/ml/)

### Operasi Pemfilteran & Analisis Sentimen

Seperti yang mungkin sudah Anda perhatikan, dataset memiliki beberapa masalah. Beberapa kolom diisi dengan informasi yang tidak berguna, sementara yang lain tampak tidak benar. Jika benar, tidak jelas bagaimana mereka dihitung, dan jawabannya tidak dapat diverifikasi secara independen melalui perhitungan Anda sendiri.

## Latihan: Pemrosesan Data Lebih Lanjut

Bersihkan data sedikit lebih banyak. Tambahkan kolom yang akan berguna nanti, ubah nilai di kolom lain, dan hapus beberapa kolom sepenuhnya.

1. Pemrosesan kolom awal

   1. Hapus `lat` dan `lng`

   2. Ganti nilai `Hotel_Address` dengan nilai berikut (jika alamat mengandung nama kota dan negara yang sama, ubah menjadi hanya kota dan negara).

      Berikut adalah satu-satunya kota dan negara dalam dataset:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Sekarang Anda dapat melakukan query data tingkat negara:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Proses kolom Meta-review Hotel

   1. Hapus `Additional_Number_of_Scoring`

   2. Ganti `Total_Number_of_Reviews` dengan jumlah total ulasan untuk hotel tersebut yang benar-benar ada dalam dataset 

   3. Ganti `Average_Score` dengan skor yang dihitung sendiri

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Proses kolom ulasan

   1. Hapus `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date`, dan `days_since_review`

   2. Biarkan `Reviewer_Score`, `Negative_Review`, dan `Positive_Review` seperti apa adanya
     
   3. Biarkan `Tags` untuk sementara waktu

     - Kita akan melakukan beberapa operasi pemfilteran tambahan pada tag di bagian berikutnya, lalu tag akan dihapus

4. Proses kolom reviewer

   1. Hapus `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Biarkan `Reviewer_Nationality`

### Kolom Tag

Kolom `Tag` bermasalah karena merupakan daftar (dalam bentuk teks) yang disimpan di kolom. Sayangnya, urutan dan jumlah sub bagian dalam kolom ini tidak selalu sama. Sulit bagi manusia untuk mengidentifikasi frasa yang benar untuk diperhatikan, karena ada 515.000 baris, dan 1427 hotel, dan masing-masing memiliki opsi yang sedikit berbeda yang dapat dipilih oleh seorang reviewer. Di sinilah NLP sangat berguna. Anda dapat memindai teks dan menemukan frasa yang paling umum, lalu menghitungnya.

Sayangnya, kita tidak tertarik pada kata tunggal, tetapi frasa multi-kata (misalnya *Perjalanan bisnis*). Menjalankan algoritma distribusi frekuensi multi-kata pada data sebanyak itu (6762646 kata) bisa memakan waktu yang luar biasa lama, tetapi tanpa melihat data, tampaknya itu adalah pengeluaran yang diperlukan. Di sinilah analisis data eksplorasi menjadi berguna, karena Anda telah melihat sampel tag seperti `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, Anda dapat mulai bertanya apakah mungkin untuk sangat mengurangi pemrosesan yang harus Anda lakukan. Untungnya, itu mungkin - tetapi pertama-tama Anda perlu mengikuti beberapa langkah untuk memastikan tag yang relevan.

### Memfilter Tag

Ingatlah bahwa tujuan dataset adalah untuk menambahkan sentimen dan kolom yang akan membantu Anda memilih hotel terbaik (untuk diri sendiri atau mungkin untuk klien yang meminta Anda membuat bot rekomendasi hotel). Anda perlu bertanya pada diri sendiri apakah tag tersebut berguna atau tidak dalam dataset akhir. Berikut adalah satu interpretasi (jika Anda membutuhkan dataset untuk alasan lain, tag yang berbeda mungkin tetap masuk/keluar dari seleksi):

1. Jenis perjalanan relevan, dan itu harus tetap
2. Jenis grup tamu penting, dan itu harus tetap
3. Jenis kamar, suite, atau studio tempat tamu menginap tidak relevan (semua hotel pada dasarnya memiliki kamar yang sama)
4. Perangkat tempat ulasan dikirimkan tidak relevan
5. Jumlah malam tamu menginap *mungkin* relevan jika Anda mengaitkan masa tinggal yang lebih lama dengan mereka menyukai hotel lebih banyak, tetapi itu agak dipaksakan, dan mungkin tidak relevan

Singkatnya, **pertahankan 2 jenis tag dan hapus yang lainnya**.

Pertama, Anda tidak ingin menghitung tag sampai mereka dalam format yang lebih baik, jadi itu berarti menghapus tanda kurung siku dan tanda kutip. Anda dapat melakukan ini dengan beberapa cara, tetapi Anda ingin yang tercepat karena ini bisa memakan waktu lama untuk memproses banyak data. Untungnya, pandas memiliki cara mudah untuk melakukan setiap langkah ini.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Setiap tag menjadi seperti: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Selanjutnya kita menemukan masalah. Beberapa ulasan, atau baris, memiliki 5 kolom, beberapa 3, beberapa 6. Ini adalah hasil dari bagaimana dataset dibuat, dan sulit untuk diperbaiki. Anda ingin mendapatkan hitungan frekuensi dari setiap frasa, tetapi mereka berada dalam urutan yang berbeda di setiap ulasan, sehingga hitungan mungkin salah, dan sebuah hotel mungkin tidak mendapatkan tag yang layak untuknya.

Sebaliknya, Anda akan menggunakan urutan yang berbeda untuk keuntungan kita, karena setiap tag adalah multi-kata tetapi juga dipisahkan oleh koma! Cara termudah untuk melakukan ini adalah dengan membuat 6 kolom sementara dengan setiap tag dimasukkan ke dalam kolom yang sesuai dengan urutannya dalam tag. Anda kemudian dapat menggabungkan 6 kolom menjadi satu kolom besar dan menjalankan metode `value_counts()` pada kolom yang dihasilkan. Dengan mencetaknya, Anda akan melihat ada 2428 tag unik. Berikut adalah sampel kecil:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Beberapa tag umum seperti `Submitted from a mobile device` tidak berguna bagi kita, jadi mungkin bijaksana untuk menghapusnya sebelum menghitung kemunculan frasa, tetapi ini adalah operasi yang sangat cepat sehingga Anda dapat membiarkannya dan mengabaikannya.

### Menghapus Tag Durasi Menginap

Menghapus tag ini adalah langkah pertama, ini sedikit mengurangi jumlah total tag yang harus dipertimbangkan. Perhatikan bahwa Anda tidak menghapusnya dari dataset, hanya memilih untuk menghapusnya dari pertimbangan sebagai nilai untuk dihitung/disimpan dalam dataset ulasan.

| Durasi menginap | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

Ada berbagai macam kamar, suite, studio, apartemen, dan sebagainya. Semuanya memiliki arti yang kurang lebih sama dan tidak relevan bagi Anda, jadi hapus dari pertimbangan.

| Jenis kamar                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Akhirnya, dan ini menyenangkan (karena tidak memerlukan banyak pemrosesan sama sekali), Anda akan mendapatkan tag *berguna* berikut:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

Anda bisa berargumen bahwa `Travellers with friends` kurang lebih sama dengan `Group`, dan itu adil untuk menggabungkan keduanya seperti di atas. Kode untuk mengidentifikasi tag yang benar ada di [notebook Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Langkah terakhir adalah membuat kolom baru untuk masing-masing tag ini. Kemudian, untuk setiap baris ulasan, jika kolom `Tag` cocok dengan salah satu kolom baru, tambahkan 1, jika tidak, tambahkan 0. Hasil akhirnya adalah hitungan berapa banyak reviewer yang memilih hotel ini (secara agregat) untuk, misalnya, bisnis vs rekreasi, atau untuk membawa hewan peliharaan, dan ini adalah informasi yang berguna saat merekomendasikan hotel.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Simpan File Anda

Akhirnya, simpan dataset seperti sekarang dengan nama baru.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operasi Analisis Sentimen

Di bagian terakhir ini, Anda akan menerapkan analisis sentimen pada kolom ulasan dan menyimpan hasilnya dalam dataset.

## Latihan: memuat dan menyimpan data yang telah difilter

Perhatikan bahwa sekarang Anda memuat dataset yang telah difilter yang disimpan di bagian sebelumnya, **bukan** dataset asli.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Menghapus Stop Words

Jika Anda menjalankan Analisis Sentimen pada kolom ulasan Negatif dan Positif, itu bisa memakan waktu lama. Diuji pada laptop uji yang kuat dengan CPU cepat, itu memakan waktu 12 - 14 menit tergantung pada perpustakaan sentimen yang digunakan. Itu adalah waktu yang (relatif) lama, jadi layak untuk diselidiki apakah itu bisa dipercepat. 

Menghapus stop words, atau kata-kata umum dalam bahasa Inggris yang tidak mengubah sentimen sebuah kalimat, adalah langkah pertama. Dengan menghapusnya, analisis sentimen harus berjalan lebih cepat, tetapi tidak menjadi kurang akurat (karena stop words tidak memengaruhi sentimen, tetapi mereka memperlambat analisis). 

Ulasan negatif terpanjang adalah 395 kata, tetapi setelah menghapus stop words, menjadi 195 kata.

Menghapus stop words juga merupakan operasi yang cepat, menghapus stop words dari 2 kolom ulasan di lebih dari 515.000 baris memakan waktu 3,3 detik pada perangkat uji. Itu bisa memakan waktu sedikit lebih lama atau lebih cepat tergantung pada kecepatan CPU perangkat Anda, RAM, apakah Anda memiliki SSD atau tidak, dan beberapa faktor lainnya. Relatif singkatnya operasi berarti bahwa jika itu meningkatkan waktu analisis sentimen, maka itu layak dilakukan.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Melakukan Analisis Sentimen

Sekarang Anda harus menghitung analisis sentimen untuk kolom ulasan negatif dan positif, dan menyimpan hasilnya dalam 2 kolom baru. Uji sentimen adalah membandingkannya dengan skor reviewer untuk ulasan yang sama. Misalnya, jika sentimen menganggap ulasan negatif memiliki sentimen 1 (sentimen sangat positif) dan sentimen ulasan positif 1, tetapi reviewer memberikan hotel skor terendah yang mungkin, maka teks ulasan tidak sesuai dengan skor, atau analis sentimen tidak dapat mengenali sentimen dengan benar. Anda harus mengharapkan beberapa skor sentimen benar-benar salah, dan sering kali itu dapat dijelaskan, misalnya ulasan bisa sangat sarkastik "Tentu saja saya SUKA tidur di kamar tanpa pemanas" dan analis sentimen menganggap itu sentimen positif, meskipun manusia yang membacanya akan tahu itu adalah sarkasme.
NLTK menyediakan berbagai analyzer sentimen untuk dipelajari, dan Anda dapat menggantinya serta melihat apakah analisis sentimen menjadi lebih atau kurang akurat. Analisis sentimen VADER digunakan di sini.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, Juni 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Nantinya dalam program Anda, ketika Anda siap untuk menghitung sentimen, Anda dapat menerapkannya pada setiap ulasan seperti berikut:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Proses ini memakan waktu sekitar 120 detik di komputer saya, tetapi akan bervariasi di setiap komputer. Jika Anda ingin mencetak hasilnya dan melihat apakah sentimen sesuai dengan ulasan:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Hal terakhir yang harus dilakukan dengan file sebelum menggunakannya dalam tantangan adalah menyimpannya! Anda juga sebaiknya mempertimbangkan untuk mengatur ulang semua kolom baru Anda agar lebih mudah digunakan (untuk manusia, ini adalah perubahan kosmetik).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Anda harus menjalankan seluruh kode untuk [notebook analisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (setelah Anda menjalankan [notebook penyaringan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) untuk menghasilkan file Hotel_Reviews_Filtered.csv).

Untuk merangkum, langkah-langkahnya adalah:

1. File dataset asli **Hotel_Reviews.csv** dieksplorasi dalam pelajaran sebelumnya dengan [notebook eksplorasi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv disaring oleh [notebook penyaringan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) menghasilkan **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv diproses oleh [notebook analisis sentimen](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) menghasilkan **Hotel_Reviews_NLP.csv**
4. Gunakan Hotel_Reviews_NLP.csv dalam Tantangan NLP di bawah ini

### Kesimpulan

Ketika Anda memulai, Anda memiliki dataset dengan kolom dan data tetapi tidak semuanya dapat diverifikasi atau digunakan. Anda telah mengeksplorasi data, menyaring apa yang tidak Anda perlukan, mengubah tag menjadi sesuatu yang berguna, menghitung rata-rata Anda sendiri, menambahkan beberapa kolom sentimen, dan semoga, mempelajari beberapa hal menarik tentang pemrosesan teks alami.

## [Kuis setelah pelajaran](https://ff-quizzes.netlify.app/en/ml/)

## Tantangan

Sekarang setelah dataset Anda dianalisis untuk sentimen, coba gunakan strategi yang telah Anda pelajari dalam kurikulum ini (mungkin clustering?) untuk menentukan pola terkait sentimen.

## Tinjauan & Studi Mandiri

Ambil [modul Learn ini](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) untuk mempelajari lebih lanjut dan menggunakan alat yang berbeda untuk mengeksplorasi sentimen dalam teks.

## Tugas 

[Coba dataset yang berbeda](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.