<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T20:46:05+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ms"
}
-->
# Analisis Sentimen dengan Ulasan Hotel

Sekarang setelah anda meneroka dataset dengan terperinci, tiba masanya untuk menapis kolum dan menggunakan teknik NLP pada dataset untuk mendapatkan wawasan baru tentang hotel.

## [Kuiz pra-kuliah](https://ff-quizzes.netlify.app/en/ml/)

### Operasi Penapisan & Analisis Sentimen

Seperti yang mungkin anda perasan, dataset ini mempunyai beberapa isu. Beberapa kolum dipenuhi dengan maklumat yang tidak berguna, sementara yang lain kelihatan tidak tepat. Jika ia betul sekalipun, tidak jelas bagaimana ia dikira, dan jawapan tidak dapat disahkan secara bebas melalui pengiraan anda sendiri.

## Latihan: Pemprosesan Data Tambahan

Bersihkan data sedikit lagi. Tambahkan kolum yang akan berguna kemudian, ubah nilai dalam kolum lain, dan buang beberapa kolum sepenuhnya.

1. Pemprosesan kolum awal

   1. Buang `lat` dan `lng`

   2. Gantikan nilai `Hotel_Address` dengan nilai berikut (jika alamat mengandungi nama bandar dan negara yang sama, ubah kepada hanya bandar dan negara).

      Berikut adalah satu-satunya bandar dan negara dalam dataset:

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

      Kini anda boleh membuat pertanyaan data peringkat negara:

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

2. Proses kolum Meta-ulasan Hotel

   1. Buang `Additional_Number_of_Scoring`

   2. Gantikan `Total_Number_of_Reviews` dengan jumlah ulasan sebenar untuk hotel tersebut yang terdapat dalam dataset 

   3. Gantikan `Average_Score` dengan skor yang dikira sendiri

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Proses kolum ulasan

   1. Buang `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` dan `days_since_review`

   2. Kekalkan `Reviewer_Score`, `Negative_Review`, dan `Positive_Review` seperti sedia ada
     
   3. Kekalkan `Tags` buat masa ini

     - Kita akan melakukan beberapa operasi penapisan tambahan pada tag dalam bahagian seterusnya dan kemudian tag akan dibuang

4. Proses kolum pengulas

   1. Buang `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Kekalkan `Reviewer_Nationality`

### Kolum Tag

Kolum `Tag` bermasalah kerana ia adalah senarai (dalam bentuk teks) yang disimpan dalam kolum. Malangnya, susunan dan bilangan sub bahagian dalam kolum ini tidak selalu sama. Sukar bagi manusia untuk mengenal pasti frasa yang betul untuk diperhatikan, kerana terdapat 515,000 baris, dan 1427 hotel, dan setiap satu mempunyai pilihan yang sedikit berbeza yang boleh dipilih oleh pengulas. Di sinilah NLP berguna. Anda boleh mengimbas teks dan mencari frasa yang paling biasa, serta mengiranya.

Malangnya, kita tidak berminat dengan perkataan tunggal, tetapi frasa berbilang perkataan (contohnya *Perjalanan perniagaan*). Menjalankan algoritma pengedaran kekerapan frasa berbilang perkataan pada data sebanyak itu (6762646 perkataan) boleh mengambil masa yang luar biasa, tetapi tanpa melihat data, nampaknya itu adalah perbelanjaan yang perlu. Di sinilah analisis data eksplorasi berguna, kerana anda telah melihat sampel tag seperti `[' Perjalanan perniagaan  ', ' Pengembara solo ', ' Bilik Single ', ' Menginap 5 malam ', ' Dihantar dari peranti mudah alih ']`, anda boleh mula bertanya sama ada mungkin untuk mengurangkan pemprosesan yang perlu dilakukan. Nasib baik, ia boleh - tetapi pertama anda perlu mengikuti beberapa langkah untuk memastikan tag yang menarik.

### Penapisan tag

Ingat bahawa matlamat dataset adalah untuk menambah sentimen dan kolum yang akan membantu anda memilih hotel terbaik (untuk diri sendiri atau mungkin untuk tugas pelanggan yang meminta anda membuat bot cadangan hotel). Anda perlu bertanya kepada diri sendiri sama ada tag berguna atau tidak dalam dataset akhir. Berikut adalah satu tafsiran (jika anda memerlukan dataset untuk tujuan lain, tag yang berbeza mungkin kekal/keluar daripada pemilihan):

1. Jenis perjalanan adalah relevan, dan itu harus kekal
2. Jenis kumpulan tetamu adalah penting, dan itu harus kekal
3. Jenis bilik, suite, atau studio yang tetamu menginap tidak relevan (semua hotel pada dasarnya mempunyai bilik yang sama)
4. Peranti yang ulasan dihantar tidak relevan
5. Bilangan malam pengulas menginap *mungkin* relevan jika anda mengaitkan penginapan yang lebih lama dengan mereka menyukai hotel lebih banyak, tetapi ini agak tidak relevan

Secara ringkas, **kekalkan 2 jenis tag dan buang yang lain**.

Pertama, anda tidak mahu mengira tag sehingga ia berada dalam format yang lebih baik, jadi itu bermakna membuang kurungan dan tanda petik. Anda boleh melakukannya dengan beberapa cara, tetapi anda mahukan cara yang paling pantas kerana ia boleh mengambil masa yang lama untuk memproses banyak data. Nasib baik, pandas mempunyai cara mudah untuk melakukan setiap langkah ini.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Setiap tag menjadi sesuatu seperti: `Perjalanan perniagaan, Pengembara solo, Bilik Single, Menginap 5 malam, Dihantar dari peranti mudah alih`. 

Seterusnya kita menemui masalah. Beberapa ulasan, atau baris, mempunyai 5 kolum, ada yang 3, ada yang 6. Ini adalah hasil daripada cara dataset dicipta, dan sukar untuk diperbaiki. Anda mahu mendapatkan kiraan kekerapan setiap frasa, tetapi ia berada dalam susunan yang berbeza dalam setiap ulasan, jadi kiraan mungkin tidak tepat, dan hotel mungkin tidak mendapat tag yang sepatutnya.

Sebaliknya, anda akan menggunakan susunan yang berbeza untuk kelebihan kita, kerana setiap tag adalah berbilang perkataan tetapi juga dipisahkan oleh koma! Cara paling mudah untuk melakukan ini adalah dengan mencipta 6 kolum sementara dengan setiap tag dimasukkan ke dalam kolum yang sepadan dengan susunannya dalam tag. Anda kemudian boleh menggabungkan 6 kolum menjadi satu kolum besar dan menjalankan kaedah `value_counts()` pada kolum yang terhasil. Apabila mencetaknya, anda akan melihat terdapat 2428 tag unik. Berikut adalah sampel kecil:

| Tag                            | Kiraan |
| ------------------------------ | ------ |
| Perjalanan santai              | 417778 |
| Dihantar dari peranti mudah alih | 307640 |
| Pasangan                       | 252294 |
| Menginap 1 malam               | 193645 |
| Menginap 2 malam               | 133937 |
| Pengembara solo                | 108545 |
| Menginap 3 malam               | 95821  |
| Perjalanan perniagaan          | 82939  |
| Kumpulan                       | 65392  |
| Keluarga dengan anak kecil     | 61015  |
| Menginap 4 malam               | 47817  |
| Bilik Double                   | 35207  |
| Bilik Double Standard          | 32248  |
| Bilik Double Superior          | 31393  |
| Keluarga dengan anak besar     | 26349  |
| Bilik Double Deluxe            | 24823  |
| Bilik Double atau Twin         | 22393  |
| Menginap 5 malam               | 20845  |
| Bilik Double atau Twin Standard | 17483  |
| Bilik Double Classic           | 16989  |
| Bilik Double atau Twin Superior | 13570  |
| 2 bilik                        | 12393  |

Beberapa tag biasa seperti `Dihantar dari peranti mudah alih` tidak berguna kepada kita, jadi mungkin bijak untuk membuangnya sebelum mengira kejadian frasa, tetapi ia adalah operasi yang sangat pantas sehingga anda boleh membiarkannya dan mengabaikannya.

### Membuang tag tempoh penginapan

Membuang tag ini adalah langkah pertama, ia mengurangkan jumlah tag yang perlu dipertimbangkan sedikit. Perhatikan bahawa anda tidak membuangnya daripada dataset, hanya memilih untuk membuangnya daripada pertimbangan sebagai nilai untuk dikira/disimpan dalam dataset ulasan.

| Tempoh penginapan | Kiraan |
| ------------------ | ------ |
| Menginap 1 malam   | 193645 |
| Menginap 2 malam   | 133937 |
| Menginap 3 malam   | 95821  |
| Menginap 4 malam   | 47817  |
| Menginap 5 malam   | 20845  |
| Menginap 6 malam   | 9776   |
| Menginap 7 malam   | 7399   |
| Menginap 8 malam   | 2502   |
| Menginap 9 malam   | 1293   |
| ...                | ...    |

Terdapat pelbagai jenis bilik, suite, studio, apartmen dan sebagainya. Semuanya bermaksud perkara yang hampir sama dan tidak relevan kepada anda, jadi buang mereka daripada pertimbangan.

| Jenis bilik                  | Kiraan |
| ---------------------------- | ------ |
| Bilik Double                 | 35207  |
| Bilik Double Standard        | 32248  |
| Bilik Double Superior        | 31393  |
| Bilik Double Deluxe          | 24823  |
| Bilik Double atau Twin       | 22393  |
| Bilik Double atau Twin Standard | 17483 |
| Bilik Double Classic         | 16989  |
| Bilik Double atau Twin Superior | 13570 |

Akhirnya, dan ini menggembirakan (kerana ia tidak memerlukan banyak pemprosesan sama sekali), anda akan tinggal dengan tag berikut yang *berguna*:

| Tag                                           | Kiraan |
| --------------------------------------------- | ------ |
| Perjalanan santai                             | 417778 |
| Pasangan                                      | 252294 |
| Pengembara solo                               | 108545 |
| Perjalanan perniagaan                         | 82939  |
| Kumpulan (digabungkan dengan Pengembara dengan rakan) | 67535  |
| Keluarga dengan anak kecil                    | 61015  |
| Keluarga dengan anak besar                    | 26349  |
| Dengan haiwan peliharaan                      | 1405   |

Anda boleh berhujah bahawa `Pengembara dengan rakan` adalah sama dengan `Kumpulan` lebih kurang, dan itu adalah wajar untuk menggabungkan kedua-duanya seperti di atas. Kod untuk mengenal pasti tag yang betul adalah [notebook Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Langkah terakhir adalah mencipta kolum baru untuk setiap tag ini. Kemudian, untuk setiap baris ulasan, jika kolum `Tag` sepadan dengan salah satu kolum baru, tambahkan 1, jika tidak, tambahkan 0. Hasil akhirnya adalah kiraan berapa ramai pengulas memilih hotel ini (secara agregat) untuk, contohnya, perniagaan vs santai, atau untuk membawa haiwan peliharaan, dan ini adalah maklumat berguna apabila mencadangkan hotel.

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

### Simpan fail anda

Akhirnya, simpan dataset seperti sekarang dengan nama baru.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operasi Analisis Sentimen

Dalam bahagian terakhir ini, anda akan menggunakan analisis sentimen pada kolum ulasan dan menyimpan hasilnya dalam dataset.

## Latihan: muat dan simpan data yang ditapis

Perhatikan bahawa sekarang anda memuat dataset yang ditapis yang disimpan dalam bahagian sebelumnya, **bukan** dataset asal.

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

### Membuang kata-kata biasa

Jika anda menjalankan Analisis Sentimen pada kolum ulasan Negatif dan Positif, ia boleh mengambil masa yang lama. Diuji pada komputer riba ujian yang berkuasa dengan CPU pantas, ia mengambil masa 12 - 14 minit bergantung pada perpustakaan sentimen yang digunakan. Itu adalah masa yang (relatif) lama, jadi berbaloi untuk menyiasat jika ia boleh dipercepatkan. 

Membuang kata-kata biasa, atau kata-kata Inggeris yang biasa yang tidak mengubah sentimen ayat, adalah langkah pertama. Dengan membuangnya, analisis sentimen sepatutnya berjalan lebih pantas, tetapi tidak kurang tepat (kerana kata-kata biasa tidak mempengaruhi sentimen, tetapi ia memperlahankan analisis). 

Ulasan negatif yang paling panjang adalah 395 perkataan, tetapi selepas membuang kata-kata biasa, ia menjadi 195 perkataan.

Membuang kata-kata biasa juga merupakan operasi yang pantas, membuang kata-kata biasa daripada 2 kolum ulasan dalam 515,000 baris mengambil masa 3.3 saat pada peranti ujian. Ia mungkin mengambil masa sedikit lebih atau kurang untuk anda bergantung pada kelajuan CPU peranti anda, RAM, sama ada anda mempunyai SSD atau tidak, dan beberapa faktor lain. Kependekan relatif operasi ini bermakna jika ia meningkatkan masa analisis sentimen, maka ia berbaloi untuk dilakukan.

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

Sekarang anda harus mengira analisis sentimen untuk kedua-dua kolum ulasan negatif dan positif, dan menyimpan hasilnya dalam 2 kolum baru. Ujian sentimen adalah untuk membandingkannya dengan skor pengulas untuk ulasan yang sama. Sebagai contoh, jika sentimen menganggap ulasan negatif mempunyai sentimen 1 (sentimen yang sangat positif) dan sentimen ulasan positif juga 1, tetapi pengulas memberikan hotel skor terendah yang mungkin, maka sama ada teks ulasan tidak sepadan dengan skor, atau penganalisis sentimen tidak dapat mengenali sentimen dengan betul. Anda harus menjangkakan beberapa skor sentimen yang benar-benar salah, dan sering kali itu dapat dijelaskan, contohnya ulasan mungkin sangat sarkastik "Sudah tentu saya SUKA tidur di bilik tanpa pemanas" dan penganalisis sentimen menganggap itu adalah sentimen positif, walaupun manusia yang membacanya akan tahu ia adalah sindiran.
NLTK menyediakan pelbagai penganalisis sentimen untuk dipelajari, dan anda boleh menggantikannya serta melihat sama ada sentimen lebih atau kurang tepat. Analisis sentimen VADER digunakan di sini.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Model Berasaskan Peraturan yang Ringkas untuk Analisis Sentimen Teks Media Sosial. Persidangan Antarabangsa Kelapan mengenai Weblogs dan Media Sosial (ICWSM-14). Ann Arbor, MI, Jun 2014.

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

Kemudian dalam program anda apabila anda bersedia untuk mengira sentimen, anda boleh menerapkannya pada setiap ulasan seperti berikut:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Ini mengambil masa kira-kira 120 saat pada komputer saya, tetapi ia akan berbeza pada setiap komputer. Jika anda ingin mencetak hasilnya dan melihat sama ada sentimen sepadan dengan ulasan:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Perkara terakhir yang perlu dilakukan dengan fail sebelum menggunakannya dalam cabaran adalah menyimpannya! Anda juga harus mempertimbangkan untuk menyusun semula semua lajur baru anda supaya ia mudah digunakan (untuk manusia, ini adalah perubahan kosmetik).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Anda harus menjalankan keseluruhan kod untuk [notebook analisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (selepas anda menjalankan [notebook penapisan anda](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) untuk menghasilkan fail Hotel_Reviews_Filtered.csv).

Untuk mengulas semula, langkah-langkahnya adalah:

1. Fail dataset asal **Hotel_Reviews.csv** diterokai dalam pelajaran sebelumnya dengan [notebook penerokaan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv ditapis oleh [notebook penapisan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) menghasilkan **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv diproses oleh [notebook analisis sentimen](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) menghasilkan **Hotel_Reviews_NLP.csv**
4. Gunakan Hotel_Reviews_NLP.csv dalam Cabaran NLP di bawah

### Kesimpulan

Apabila anda bermula, anda mempunyai dataset dengan lajur dan data tetapi tidak semuanya boleh disahkan atau digunakan. Anda telah meneroka data, menapis apa yang tidak diperlukan, menukar tag kepada sesuatu yang berguna, mengira purata anda sendiri, menambah beberapa lajur sentimen dan diharapkan, mempelajari beberapa perkara menarik tentang pemprosesan teks semula jadi.

## [Kuiz selepas kuliah](https://ff-quizzes.netlify.app/en/ml/)

## Cabaran

Sekarang setelah anda menganalisis dataset anda untuk sentimen, cuba gunakan strategi yang telah anda pelajari dalam kurikulum ini (pengelompokan, mungkin?) untuk menentukan corak sekitar sentimen.

## Ulasan & Kajian Kendiri

Ambil [modul pembelajaran ini](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) untuk mempelajari lebih lanjut dan menggunakan alat yang berbeza untuk meneroka sentimen dalam teks.

## Tugasan 

[Cuba dataset yang berbeza](assignment.md)

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.