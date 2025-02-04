# Analisis Sentimen dengan Ulasan Hotel

Sekarang setelah anda menjelajahi dataset secara detail, saatnya untuk menyaring kolom-kolom dan kemudian menggunakan teknik NLP pada dataset untuk mendapatkan wawasan baru tentang hotel-hotel tersebut.
## [Kuis Pra-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operasi Penyaringan & Analisis Sentimen

Seperti yang mungkin sudah anda perhatikan, dataset ini memiliki beberapa masalah. Beberapa kolom diisi dengan informasi yang tidak berguna, yang lain tampaknya tidak benar. Jika mereka benar, tidak jelas bagaimana mereka dihitung, dan jawabannya tidak dapat diverifikasi secara independen oleh perhitungan anda sendiri.

## Latihan: sedikit lagi pemrosesan data

Bersihkan data sedikit lagi. Tambahkan kolom yang akan berguna nanti, ubah nilai di kolom lain, dan hapus beberapa kolom sepenuhnya.

1. Pemrosesan kolom awal

   1. Hapus `lat` dan `lng`

   2. Ganti nilai `Hotel_Address` dengan nilai berikut (jika alamat berisi nama kota dan negara, ubah menjadi hanya kota dan negara).

      Ini adalah satu-satunya kota dan negara dalam dataset:

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

      Sekarang anda bisa meng-query data tingkat negara:

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

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` dengan skor yang dihitung sendiri

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Proses kolom ulasan

   1. Hapus `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, anda bisa mulai bertanya apakah mungkin untuk mengurangi pemrosesan yang harus dilakukan. Untungnya, itu mungkin - tetapi pertama-tama anda perlu mengikuti beberapa langkah untuk memastikan tag yang diminati.

### Penyaringan tag

Ingat bahwa tujuan dataset ini adalah untuk menambahkan sentimen dan kolom yang akan membantu anda memilih hotel terbaik (untuk diri sendiri atau mungkin untuk tugas klien yang meminta anda membuat bot rekomendasi hotel). Anda perlu bertanya pada diri sendiri apakah tag tersebut berguna atau tidak dalam dataset akhir. Berikut adalah satu interpretasi (jika anda membutuhkan dataset untuk alasan lain, tag yang berbeda mungkin tetap masuk/keluar dari seleksi):

1. Jenis perjalanan itu relevan, dan itu harus tetap
2. Jenis kelompok tamu itu penting, dan itu harus tetap
3. Jenis kamar, suite, atau studio tempat tamu menginap tidak relevan (semua hotel pada dasarnya memiliki kamar yang sama)
4. Perangkat yang digunakan untuk mengirim ulasan tidak relevan
5. Jumlah malam yang dihabiskan pengulas *mungkin* relevan jika anda mengaitkan masa tinggal yang lebih lama dengan mereka yang lebih menyukai hotel tersebut, tetapi itu sedikit berlebihan, dan mungkin tidak relevan

Singkatnya, **pertahankan 2 jenis tag dan hapus yang lain**.

Pertama, anda tidak ingin menghitung tag sampai mereka dalam format yang lebih baik, jadi itu berarti menghapus tanda kurung siku dan kutipan. Anda bisa melakukannya dengan beberapa cara, tetapi anda ingin yang tercepat karena bisa memakan waktu lama untuk memproses banyak data. Untungnya, pandas memiliki cara mudah untuk melakukan masing-masing langkah ini.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Setiap tag menjadi sesuatu seperti: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

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

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
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

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

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

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` kolom cocok dengan salah satu kolom baru, tambahkan 1, jika tidak, tambahkan 0. Hasil akhirnya adalah hitungan berapa banyak pengulas yang memilih hotel ini (secara agregat) untuk, misalnya, bisnis vs liburan, atau untuk membawa hewan peliharaan, dan ini adalah informasi yang berguna saat merekomendasikan hotel.

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

### Simpan file anda

Akhirnya, simpan dataset sebagaimana adanya sekarang dengan nama baru.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operasi Analisis Sentimen

Di bagian akhir ini, anda akan menerapkan analisis sentimen pada kolom ulasan dan menyimpan hasilnya dalam dataset.

## Latihan: memuat dan menyimpan data yang sudah disaring

Perhatikan bahwa sekarang anda memuat dataset yang sudah disaring yang disimpan di bagian sebelumnya, **bukan** dataset asli.

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

### Menghapus kata-kata stop

Jika anda menjalankan Analisis Sentimen pada kolom Ulasan Negatif dan Positif, itu bisa memakan waktu lama. Diuji pada laptop uji yang kuat dengan CPU cepat, itu memakan waktu 12 - 14 menit tergantung pada perpustakaan sentimen yang digunakan. Itu adalah waktu yang (relatif) lama, jadi patut diselidiki apakah itu bisa dipercepat.

Menghapus kata-kata stop, atau kata-kata umum dalam bahasa Inggris yang tidak mengubah sentimen suatu kalimat, adalah langkah pertama. Dengan menghapusnya, analisis sentimen harus berjalan lebih cepat, tetapi tidak kurang akurat (karena kata-kata stop tidak mempengaruhi sentimen, tetapi mereka memperlambat analisis).

Ulasan negatif terpanjang adalah 395 kata, tetapi setelah menghapus kata-kata stop, itu menjadi 195 kata.

Menghapus kata-kata stop juga merupakan operasi yang cepat, menghapus kata-kata stop dari 2 kolom ulasan lebih dari 515.000 baris memakan waktu 3,3 detik pada perangkat uji. Itu bisa memakan waktu sedikit lebih atau kurang untuk anda tergantung pada kecepatan CPU perangkat anda, RAM, apakah anda memiliki SSD atau tidak, dan beberapa faktor lainnya. Relatif singkatnya operasi ini berarti bahwa jika itu meningkatkan waktu analisis sentimen, maka itu layak dilakukan.

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

### Melakukan analisis sentimen

Sekarang anda harus menghitung analisis sentimen untuk kolom ulasan negatif dan positif, dan menyimpan hasilnya dalam 2 kolom baru. Uji sentimen akan membandingkannya dengan skor pengulas untuk ulasan yang sama. Misalnya, jika sentimen berpikir ulasan negatif memiliki sentimen 1 (sentimen sangat positif) dan sentimen ulasan positif 1, tetapi pengulas memberi hotel skor terendah yang mungkin, maka teks ulasan tidak cocok dengan skor, atau analisis sentimen tidak dapat mengenali sentimen dengan benar. Anda harus mengharapkan beberapa skor sentimen sepenuhnya salah, dan sering kali itu bisa dijelaskan, misalnya ulasan bisa sangat sarkastis "Tentu saja saya SANGAT MENYUKAI tidur di kamar tanpa pemanas" dan analisis sentimen berpikir itu adalah sentimen positif, meskipun manusia yang membacanya akan tahu itu adalah sarkasme.

NLTK menyediakan berbagai analis sentimen untuk dipelajari, dan anda bisa menggantinya dan melihat apakah sentimen lebih atau kurang akurat. Analisis sentimen VADER digunakan di sini.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Model Berbasis Aturan yang Parsimonious untuk Analisis Sentimen Teks Media Sosial. Konferensi Internasional Kedelapan tentang Weblogs dan Media Sosial (ICWSM-14). Ann Arbor, MI, Juni 2014.

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

Nantinya dalam program anda ketika anda siap menghitung sentimen, anda bisa menerapkannya pada setiap ulasan sebagai berikut:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ini memakan waktu sekitar 120 detik di komputer saya, tetapi akan bervariasi di setiap komputer. Jika anda ingin mencetak hasilnya dan melihat apakah sentimen sesuai dengan ulasan:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Hal terakhir yang harus dilakukan dengan file sebelum menggunakannya dalam tantangan adalah menyimpannya! Anda juga harus mempertimbangkan untuk mengatur ulang semua kolom baru anda agar mudah digunakan (untuk manusia, ini adalah perubahan kosmetik).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Anda harus menjalankan seluruh kode untuk [notebook analisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (setelah anda menjalankan [notebook penyaringan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) untuk menghasilkan file Hotel_Reviews_Filtered.csv).

Untuk meninjau, langkah-langkahnya adalah:

1. File dataset asli **Hotel_Reviews.csv** dieksplorasi dalam pelajaran sebelumnya dengan [notebook penjelajah](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv disaring oleh [notebook penyaringan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) menghasilkan **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv diproses oleh [notebook analisis sentimen](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) menghasilkan **Hotel_Reviews_NLP.csv**
4. Gunakan Hotel_Reviews_NLP.csv dalam Tantangan NLP di bawah ini

### Kesimpulan

Ketika anda mulai, anda memiliki dataset dengan kolom dan data tetapi tidak semuanya dapat diverifikasi atau digunakan. Anda telah menjelajahi data, menyaring apa yang tidak anda butuhkan, mengubah tag menjadi sesuatu yang berguna, menghitung rata-rata anda sendiri, menambahkan beberapa kolom sentimen dan semoga, mempelajari beberapa hal menarik tentang pemrosesan teks alami.

## [Kuis Pasca-Kuliah](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Tantangan

Sekarang setelah anda menganalisis dataset untuk sentimen, lihat apakah anda bisa menggunakan strategi yang telah anda pelajari dalam kurikulum ini (klastering, mungkin?) untuk menentukan pola di sekitar sentimen.

## Tinjauan & Studi Mandiri

Ikuti [modul Learn ini](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) untuk mempelajari lebih lanjut dan menggunakan alat yang berbeda untuk menjelajahi sentimen dalam teks.
## Tugas 

[Coba dataset yang berbeda](assignment.md)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.