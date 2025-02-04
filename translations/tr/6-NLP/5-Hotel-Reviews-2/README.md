# Otel yorumları ile duygu analizi

Artık veri setini ayrıntılı bir şekilde incelediğinize göre, sütunları filtreleyip veri seti üzerinde NLP tekniklerini kullanarak oteller hakkında yeni bilgiler edinmenin zamanı geldi.
## [Ders öncesi sınav](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Filtreleme ve Duygu Analizi İşlemleri

Muhtemelen fark etmişsinizdir, veri setinde bazı sorunlar var. Bazı sütunlar gereksiz bilgilerle dolu, diğerleri ise yanlış görünüyor. Doğru olsalar bile, nasıl hesaplandıkları belirsiz ve cevaplar kendi hesaplamalarınızla bağımsız olarak doğrulanamıyor.

## Egzersiz: biraz daha veri işleme

Verileri biraz daha temizleyin. Daha sonra kullanışlı olacak sütunlar ekleyin, diğer sütunlardaki değerleri değiştirin ve bazı sütunları tamamen kaldırın.

1. İlk sütun işlemleri

   1. `lat` ve `lng`'i kaldırın

   2. `Hotel_Address` değerlerini aşağıdaki değerlerle değiştirin (adres şehir ve ülke ismini içeriyorsa, sadece şehir ve ülke olarak değiştirin).

      Veri setinde sadece bu şehirler ve ülkeler var:

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

      Artık ülke düzeyinde veri sorgulayabilirsiniz:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Otel_Adresi             | Otel_Adı   |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Otel Meta-inceleme sütunlarını işleyin

  1. `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` kendi hesapladığımız skorla kaldırın

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. İnceleme sütunlarını işleyin

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' İş seyahati  ', ' Yalnız gezgin ', ' Tek Kişilik Oda ', ' 5 gece kaldı ', ' Mobil cihazdan gönderildi ']` sütunlarını kaldırın, işlemi büyük ölçüde azaltmanın mümkün olup olmadığını sormaya başlayabilirsiniz. Neyse ki mümkün - ancak önce ilgi çekici etiketleri belirlemek için birkaç adımı izlemeniz gerekiyor.

### Etiketleri filtreleme

Veri setinin amacının, en iyi oteli seçmenize yardımcı olacak duygu ve sütunlar eklemek olduğunu unutmayın (kendiniz için veya belki size bir otel öneri botu yapma görevi veren bir müşteri için). Etiketlerin nihai veri setinde yararlı olup olmadığını kendinize sormanız gerekiyor. İşte bir yorum (eğer veri setine başka nedenlerle ihtiyacınız varsa, farklı etiketler seçime dahil olabilir veya dışarıda kalabilir):

1. Seyahat türü önemlidir ve kalmalıdır
2. Misafir grubu türü önemlidir ve kalmalıdır
3. Misafirin kaldığı oda, süit veya stüdyo türü önemsizdir (tüm otellerde temelde aynı odalar vardır)
4. İncelemenin gönderildiği cihaz önemsizdir
5. İncelemecinin kaldığı gece sayısı, eğer daha uzun kalmalarını oteli daha çok sevmeleriyle ilişkilendirirseniz, önemli olabilir, ancak bu biraz zorlayıcıdır ve muhtemelen önemsizdir

Özetle, **2 tür etiketi tutun ve diğerlerini kaldırın**.

İlk olarak, etiketleri daha iyi bir formata getirmeden saymak istemezsiniz, bu da köşeli parantezleri ve tırnak işaretlerini kaldırmak anlamına gelir. Bunu birkaç şekilde yapabilirsiniz, ancak en hızlı yolu istersiniz çünkü çok fazla veriyi işlemek uzun sürebilir. Neyse ki, pandas bu adımların her birini kolayca yapmanın bir yolunu sunar.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Her etiket şu şekilde olur: `İş seyahati, Yalnız gezgin, Tek Kişilik Oda, 5 gece kaldı, Mobil cihazdan gönderildi`. 

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

Some of the common tags like `Mobil cihazdan gönderildi` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

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

You could argue that `Arkadaşlarla seyahat edenler` is the same as `Grup` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Etiket` sütunu yeni sütunlardan biriyle eşleşiyorsa, 1 ekleyin, değilse 0 ekleyin. Sonuç, bu oteli (toplamda) iş veya eğlence için veya bir evcil hayvanla getirmek için kaç incelemecinin seçtiğini saymak olacak ve bu, bir otel önerirken yararlı bir bilgidir.

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

### Dosyanızı kaydedin

Son olarak, veri setini şu anki haliyle yeni bir adla kaydedin.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Duygu Analizi İşlemleri

Bu son bölümde, inceleme sütunlarına duygu analizi uygulayacak ve sonuçları bir veri setinde kaydedeceksiniz.

## Egzersiz: filtrelenmiş verileri yükleyin ve kaydedin

Artık filtrelenmiş veri setini yüklediğinizi, **orijinal** veri setini değil, unutmayın.

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

### Stop kelimelerini kaldırma

Negatif ve Pozitif inceleme sütunlarında Duygu Analizi çalıştırırsanız, uzun sürebilir. Hızlı bir CPU'ya sahip güçlü bir test dizüstü bilgisayarında test edildiğinde, kullanılan duygu kütüphanesine bağlı olarak 12 - 14 dakika sürdü. Bu (nispeten) uzun bir süre, bu nedenle hızlandırılıp hızlandırılamayacağını araştırmaya değer.

Stop kelimeleri, yani bir cümlenin duygusunu değiştirmeyen yaygın İngilizce kelimeleri kaldırmak ilk adımdır. Onları kaldırarak, duygu analizi daha hızlı çalışmalı, ancak daha az doğru olmamalıdır (çünkü stop kelimeleri duyguyu etkilemez, ancak analizi yavaşlatır).

En uzun negatif inceleme 395 kelimeydi, ancak stop kelimeleri kaldırıldıktan sonra 195 kelime oldu.

Stop kelimeleri kaldırmak da hızlı bir işlemdir, 515.000 satırda 2 inceleme sütunundan stop kelimelerini kaldırmak test cihazında 3.3 saniye sürdü. Cihazınızın CPU hızına, RAM'e, SSD'ye sahip olup olmamanıza ve bazı diğer faktörlere bağlı olarak sizin için biraz daha uzun veya kısa sürebilir. İşlemin nispi kısalığı, duygu analizi süresini iyileştiriyorsa, yapmaya değer olduğu anlamına gelir.

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

### Duygu analizi gerçekleştirme

Şimdi negatif ve pozitif inceleme sütunları için duygu analizini hesaplamalı ve sonucu 2 yeni sütunda saklamalısınız. Duygu testinin, aynı inceleme için incelemecinin puanıyla karşılaştırılması olacaktır. Örneğin, duygu analizinin negatif incelemenin 1 (son derece pozitif duygu) ve pozitif inceleme duygu analizinin 1 olduğunu düşündüğünü varsayalım, ancak incelemeci otele mümkün olan en düşük puanı verdiyse, inceleme metni puanla eşleşmiyor olabilir veya duygu analizörü duyguyu doğru tanıyamamış olabilir. Bazı duygu puanlarının tamamen yanlış olmasını beklemelisiniz ve bu genellikle açıklanabilir olacaktır, örneğin inceleme son derece alaycı olabilir "Tabii ki ısıtma olmayan bir odada uyumayı SEVDİM" ve duygu analizörü bunun pozitif bir duygu olduğunu düşünebilir, ancak bunu okuyan bir insan bunun alaycı olduğunu bilir.

NLTK, öğrenmek için farklı duygu analizörleri sağlar ve bunları değiştirebilir ve duygu analizinin daha doğru olup olmadığını görebilirsiniz. Burada VADER duygu analizi kullanılmıştır.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Sosyal Medya Metni için Basit Kurallara Dayalı Bir Model. Sekizinci Uluslararası Webloglar ve Sosyal Medya Konferansı (ICWSM-14). Ann Arbor, MI, Haziran 2014.

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

Programınızın ilerleyen bölümlerinde duygu analizi yapmaya hazır olduğunuzda, her incelemeye aşağıdaki gibi uygulayabilirsiniz:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Bu, bilgisayarımda yaklaşık 120 saniye sürüyor, ancak her bilgisayarda değişecektir. Sonuçları yazdırmak ve duygunun incelemeye uyup uymadığını görmek isterseniz:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Dosyayı kullanmadan önce yapmanız gereken son şey, onu kaydetmektir! Ayrıca, yeni sütunlarınızı yeniden düzenlemeyi düşünmelisiniz, böylece çalışmak daha kolay olur (bir insan için, bu kozmetik bir değişikliktir).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[analiz defterinin](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) tamamını çalıştırmalısınız (Hotel_Reviews_Filtered.csv dosyasını oluşturmak için [filtreleme defterinizi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) çalıştırdıktan sonra).

Gözden geçirmek için adımlar:

1. Orijinal veri seti dosyası **Hotel_Reviews.csv** önceki derste [keşif defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ile incelenmiştir
2. Hotel_Reviews.csv [filtreleme defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ile filtrelenir ve **Hotel_Reviews_Filtered.csv** elde edilir
3. Hotel_Reviews_Filtered.csv [duygu analizi defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ile işlenir ve **Hotel_Reviews_NLP.csv** elde edilir
4. Aşağıdaki NLP Challenge'da Hotel_Reviews_NLP.csv dosyasını kullanın

### Sonuç

Başladığınızda, sütunlar ve veriler içeren bir veri setiniz vardı, ancak hepsi doğrulanabilir veya kullanılabilir değildi. Verileri incelediniz, ihtiyacınız olmayanları filtrelediniz, etiketleri faydalı bir şeye dönüştürdünüz, kendi ortalamalarınızı hesapladınız, bazı duygu sütunları eklediniz ve umarım doğal metni işlemede ilginç şeyler öğrendiniz.

## [Ders sonrası sınav](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Zorluk

Artık veri setinizin duygu analizi yapıldığına göre, bu müfredatta öğrendiğiniz stratejileri (belki kümeleme?) kullanarak duygu etrafında kalıplar belirleyip belirleyemeyeceğinizi görün.

## İnceleme ve Kendi Kendine Çalışma

Bu [Learn modülünü](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) alarak daha fazla bilgi edinin ve metinlerde duygu keşfetmek için farklı araçlar kullanın.
## Ödev

[Farklı bir veri seti deneyin](assignment.md)

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Orijinal belgenin kendi dilindeki hali yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.