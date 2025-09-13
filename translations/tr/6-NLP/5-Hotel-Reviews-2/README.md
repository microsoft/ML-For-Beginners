<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T08:09:17+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "tr"
}
-->
# Otel Yorumları ile Duygu Analizi

Artık veri setini detaylı bir şekilde incelediğinize göre, sütunları filtreleme zamanı geldi. Daha sonra veri seti üzerinde NLP tekniklerini kullanarak oteller hakkında yeni içgörüler elde edeceksiniz.

## [Ders Öncesi Testi](https://ff-quizzes.netlify.app/en/ml/)

### Filtreleme ve Duygu Analizi İşlemleri

Muhtemelen fark etmişsinizdir, veri setinde birkaç sorun var. Bazı sütunlar gereksiz bilgilerle dolu, diğerleri ise yanlış görünüyor. Doğru olsalar bile, nasıl hesaplandıkları belirsiz ve kendi hesaplamalarınızla bağımsız olarak doğrulanamıyor.

## Alıştırma: Biraz Daha Veri İşleme

Veriyi biraz daha temizleyin. Daha sonra faydalı olacak sütunlar ekleyin, diğer sütunlardaki değerleri değiştirin ve bazı sütunları tamamen kaldırın.

1. İlk sütun işlemleri

   1. `lat` ve `lng` sütunlarını kaldırın.

   2. `Hotel_Address` değerlerini aşağıdaki değerlere değiştirin (adres şehir ve ülke adını içeriyorsa, sadece şehir ve ülke olarak değiştirin).

      Veri setindeki tek şehir ve ülkeler şunlardır:

      Amsterdam, Hollanda

      Barselona, İspanya

      Londra, Birleşik Krallık

      Milano, İtalya

      Paris, Fransa

      Viyana, Avusturya 

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

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Hollanda    |    105     |
      | Barselona, İspanya     |    211     |
      | Londra, Birleşik Krallık |    400     |
      | Milano, İtalya         |    162     |
      | Paris, Fransa          |    458     |
      | Viyana, Avusturya      |    158     |

2. Otel Meta-yorum sütunlarını işleme

   1. `Additional_Number_of_Scoring` sütununu kaldırın.

   2. `Total_Number_of_Reviews` sütununu veri setinde gerçekten bulunan otel yorumlarının toplam sayısıyla değiştirin.

   3. `Average_Score` sütununu kendi hesapladığınız skorla değiştirin.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Yorum sütunlarını işleme

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ve `days_since_review` sütunlarını kaldırın.

   2. `Reviewer_Score`, `Negative_Review` ve `Positive_Review` sütunlarını olduğu gibi tutun.

   3. `Tags` sütununu şimdilik tutun.

      - Bir sonraki bölümde etiketler üzerinde ek filtreleme işlemleri yapacağız ve ardından etiketler kaldırılacak.

4. Yorumcu sütunlarını işleme

   1. `Total_Number_of_Reviews_Reviewer_Has_Given` sütununu kaldırın.
  
   2. `Reviewer_Nationality` sütununu tutun.

### Etiket Sütunları

`Tag` sütunu sorunlu çünkü sütunda bir liste (metin biçiminde) saklanıyor. Ne yazık ki, bu sütundaki alt bölümlerin sırası ve sayısı her zaman aynı değil. 515.000 satır ve 1427 otel olduğu için, bir insanın ilgilenmesi gereken doğru ifadeleri belirlemesi zor. Her bir yorumcunun seçebileceği seçenekler biraz farklı. İşte burada NLP devreye giriyor. Metni tarayabilir, en yaygın ifadeleri bulabilir ve bunları sayabilirsiniz.

Ne yazık ki, tek kelimelerle değil, çok kelimeli ifadelerle ilgileniyoruz (örneğin, *İş seyahati*). Bu kadar veri üzerinde çok kelimeli bir sıklık dağılım algoritması çalıştırmak (6762646 kelime) olağanüstü bir zaman alabilir, ancak veriye bakmadan bunun gerekli bir masraf olduğu düşünülebilir. İşte burada keşif veri analizi faydalı oluyor, çünkü etiketlerin bir örneğini gördünüz: `[' İş seyahati  ', ' Tek başına seyahat eden ', ' Tek kişilik oda ', ' 5 gece kaldı ', ' Mobil cihazdan gönderildi ']`. Bu, işlemi büyük ölçüde azaltmanın mümkün olup olmadığını sormaya başlamanızı sağlar. Neyse ki mümkün - ancak önce ilgilenilecek etiketleri belirlemek için birkaç adımı takip etmeniz gerekiyor.

### Etiketleri Filtreleme

Unutmayın, veri setinin amacı duygu ve sütunlar ekleyerek en iyi oteli seçmenize yardımcı olmaktır (kendiniz için veya belki bir otel öneri botu oluşturmanızı isteyen bir müşteri için). Etiketlerin nihai veri setinde faydalı olup olmadığını kendinize sormanız gerekiyor. İşte bir yorum (veri setine başka nedenlerle ihtiyaç duyulursa farklı etiketler seçilebilir):

1. Seyahat türü önemlidir ve kalmalıdır.
2. Misafir grubunun türü önemlidir ve kalmalıdır.
3. Misafirin kaldığı oda, süit veya stüdyo türü önemsizdir (tüm otellerde temelde aynı odalar vardır).
4. Yorumun gönderildiği cihaz önemsizdir.
5. Misafirin kaldığı gece sayısı *önemli* olabilir, eğer daha uzun süre kalanların oteli daha çok sevdiğini düşünürseniz, ancak bu zayıf bir bağlantıdır ve muhtemelen önemsizdir.

Özetle, **2 tür etiketi tutun ve diğerlerini kaldırın**.

İlk olarak, etiketleri daha iyi bir formata getirmeden saymak istemezsiniz, bu da köşeli parantezleri ve tırnak işaretlerini kaldırmak anlamına gelir. Bunu birkaç şekilde yapabilirsiniz, ancak en hızlı yöntemi seçmek istersiniz çünkü çok fazla veri işlemek uzun sürebilir. Neyse ki, pandas bu adımların her birini kolayca yapmanın bir yolunu sunar.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Her bir etiket şu şekilde olur: `İş seyahati, Tek başına seyahat eden, Tek kişilik oda, 5 gece kaldı, Mobil cihazdan gönderildi`.

Sonra bir sorunla karşılaşıyoruz. Bazı yorumlar veya satırlar 5 sütuna, bazıları 3'e, bazıları 6'ya sahip. Bu, veri setinin nasıl oluşturulduğunun bir sonucu ve düzeltmesi zor. Her bir ifadeyi sıklıkla saymak istiyorsunuz, ancak her yorumda farklı sırada oldukları için sayım yanlış olabilir ve bir otel hak ettiği bir etiketi alamayabilir.

Bunun yerine, farklı sıralamayı avantajımıza kullanacağız çünkü her bir etiket çok kelimeli ancak aynı zamanda bir virgülle ayrılmış! Bunun en basit yolu, her bir etiketi sırasına karşılık gelen sütuna yerleştirerek 6 geçici sütun oluşturmaktır. Daha sonra bu 6 sütunu tek bir büyük sütunda birleştirip `value_counts()` yöntemini çalıştırabilirsiniz. Bunu yazdırdığınızda, 2428 benzersiz etiket olduğunu göreceksiniz. İşte küçük bir örnek:

| Etiket                          | Sayı   |
| ------------------------------- | ------ |
| Tatil seyahati                  | 417778 |
| Mobil cihazdan gönderildi       | 307640 |
| Çift                            | 252294 |
| 1 gece kaldı                    | 193645 |
| 2 gece kaldı                    | 133937 |
| Tek başına seyahat eden         | 108545 |
| 3 gece kaldı                    | 95821  |
| İş seyahati                     | 82939  |
| Grup                            | 65392  |
| Küçük çocuklu aile              | 61015  |
| 4 gece kaldı                    | 47817  |
| Çift kişilik oda                | 35207  |
| Standart çift kişilik oda       | 32248  |
| Üst düzey çift kişilik oda      | 31393  |
| Büyük çocuklu aile              | 26349  |
| Lüks çift kişilik oda           | 24823  |
| Çift veya ikiz kişilik oda      | 22393  |
| 5 gece kaldı                    | 20845  |
| Standart çift veya ikiz kişilik oda | 17483  |
| Klasik çift kişilik oda         | 16989  |
| Üst düzey çift veya ikiz kişilik oda | 13570 |
| 2 oda                           | 12393  |

`Mobil cihazdan gönderildi` gibi yaygın etiketler bizim için hiçbir işe yaramaz, bu yüzden bunları saymadan önce kaldırmak akıllıca olabilir, ancak bu kadar hızlı bir işlem olduğu için onları bırakabilir ve görmezden gelebilirsiniz.

### Kalış Süresi Etiketlerini Kaldırma

Bu etiketleri kaldırmak ilk adımdır, dikkate alınacak toplam etiket sayısını biraz azaltır. Not: Bunları veri setinden kaldırmazsınız, sadece inceleme veri setinde değer olarak saymayı/dahil etmeyi seçmezsiniz.

| Kalış süresi   | Sayı   |
| -------------- | ------ |
| 1 gece kaldı   | 193645 |
| 2 gece kaldı   | 133937 |
| 3 gece kaldı   | 95821  |
| 4 gece kaldı   | 47817  |
| 5 gece kaldı   | 20845  |
| 6 gece kaldı   | 9776   |
| 7 gece kaldı   | 7399   |
| 8 gece kaldı   | 2502   |
| 9 gece kaldı   | 1293   |
| ...            | ...    |

Çok çeşitli odalar, süitler, stüdyolar, daireler vb. vardır. Hepsi kabaca aynı şeyi ifade eder ve sizin için önemli değildir, bu yüzden bunları dikkate almaktan çıkarın.

| Oda türü                     | Sayı   |
| ---------------------------- | ------ |
| Çift kişilik oda             | 35207  |
| Standart çift kişilik oda    | 32248  |
| Üst düzey çift kişilik oda   | 31393  |
| Lüks çift kişilik oda        | 24823  |
| Çift veya ikiz kişilik oda   | 22393  |
| Standart çift veya ikiz kişilik oda | 17483 |
| Klasik çift kişilik oda      | 16989  |
| Üst düzey çift veya ikiz kişilik oda | 13570 |

Son olarak, ve bu harika (çünkü çok fazla işlem gerektirmedi), aşağıdaki *faydalı* etiketlerle kalacaksınız:

| Etiket                                       | Sayı   |
| ------------------------------------------- | ------ |
| Tatil seyahati                              | 417778 |
| Çift                                        | 252294 |
| Tek başına seyahat eden                     | 108545 |
| İş seyahati                                 | 82939  |
| Grup (Arkadaşlarla seyahat edenlerle birleştirildi) | 67535  |
| Küçük çocuklu aile                          | 61015  |
| Büyük çocuklu aile                          | 26349  |
| Evcil hayvanla                              | 1405   |

`Arkadaşlarla seyahat edenler`in `Grup` ile aynı olduğu söylenebilir ve bu iki etiketin birleştirilmesi mantıklı olur. Doğru etiketleri belirlemek için kod [Etiketler not defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) dosyasındadır.

Son adım, bu etiketlerin her biri için yeni sütunlar oluşturmaktır. Daha sonra, her bir yorum satırı için, `Tag` sütunu yeni sütunlardan biriyle eşleşirse, 1 ekleyin, eşleşmezse 0 ekleyin. Sonuç olarak, bir oteli iş veya tatil için seçen yorumcuların toplam sayısını (toplamda) elde edeceksiniz veya bir evcil hayvanla seyahat edenler için ve bu, bir otel önerirken faydalı bir bilgi olacaktır.

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

### Dosyanızı Kaydedin

Son olarak, veri setini şu anki haliyle yeni bir adla kaydedin.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Duygu Analizi İşlemleri

Bu son bölümde, yorum sütunlarına duygu analizi uygulayacak ve sonuçları bir veri setinde kaydedeceksiniz.

## Alıştırma: Filtrelenmiş Veriyi Yükleyin ve Kaydedin

Artık önceki bölümde kaydedilen filtrelenmiş veri setini, **orijinal veri setini değil**, yüklediğinizi unutmayın.

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

### Durdurma Kelimelerini Kaldırma

Negatif ve Pozitif yorum sütunlarında Duygu Analizi çalıştıracak olsaydınız, bu uzun sürebilirdi. Hızlı bir CPU'ya sahip güçlü bir test dizüstü bilgisayarında test edildiğinde, kullanılan duygu kütüphanesine bağlı olarak 12 - 14 dakika sürdü. Bu (nispeten) uzun bir süre, bu yüzden hızlandırılıp hızlandırılamayacağını araştırmaya değer.

Durdurma kelimelerini, yani bir cümlenin duygusunu değiştirmeyen yaygın İngilizce kelimeleri kaldırmak ilk adımdır. Bunları kaldırarak, duygu analizi daha hızlı çalışmalı, ancak daha az doğru olmamalıdır (çünkü durdurma kelimeleri duyguya etki etmez, ancak analizi yavaşlatır). 

En uzun negatif yorum 395 kelimeydi, ancak durdurma kelimeleri kaldırıldıktan sonra 195 kelimeye düştü.

Durdurma kelimelerini kaldırmak da hızlı bir işlemdir; 515.000 satırda 2 yorum sütunundan durdurma kelimelerini kaldırmak test cihazında 3.3 saniye sürdü. Cihazınızın CPU hızı, RAM, SSD olup olmaması ve diğer bazı faktörlere bağlı olarak biraz daha fazla veya az sürebilir. İşlemin nispeten kısa olması, duygu analizi süresini iyileştiriyorsa yapmaya değer olduğu anlamına gelir.

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

### Duygu Analizi Yapma

Şimdi hem negatif hem de pozitif yorum sütunları için duygu analizini hesaplamalı ve sonucu 2 yeni sütunda saklamalısınız. Duygunun testi, aynı yorum için yorumcunun verdiği puanla karşılaştırmak olacaktır. Örneğin, duygu analizi negatif yorumun duygu puanını 1 (son derece olumlu duygu) ve pozitif yorumun duygu puanını 1 olarak hesaplıyorsa, ancak yorumcu otele mümkün olan en düşük puanı verdiyse, o zaman ya yorum metni puanla eşleşmiyor ya da duygu analizcisi duyguyu doğru bir şekilde tanıyamamış olabilir. Bazı duygu puanlarının tamamen yanlış olmasını beklemelisiniz ve genellikle bu açıklanabilir olacaktır, örneğin yorum son derece alaycı olabilir: "Tabii ki ısıtması olmayan bir odada uyumayı ÇOK SEVDİM" ve duygu analizcisi bunun olumlu bir duygu olduğunu düşünebilir, ancak bir insan bunu okuduğunda bunun alaycı olduğunu anlayacaktır.
NLTK, farklı duygu analiz araçları sunar ve bunları değiştirerek duygu analizinin daha doğru olup olmadığını görebilirsiniz. Burada VADER duygu analizi kullanılmıştır.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Sosyal Medya Metinlerinin Duygu Analizi için Basit ve Kural Tabanlı Bir Model. Sekizinci Uluslararası Web Günlükleri ve Sosyal Medya Konferansı (ICWSM-14). Ann Arbor, MI, Haziran 2014.

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

Programınızda duygu analizi yapmaya hazır olduğunuzda, bunu her bir incelemeye şu şekilde uygulayabilirsiniz:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Bu işlem bilgisayarımda yaklaşık 120 saniye sürüyor, ancak her bilgisayarda farklılık gösterebilir. Sonuçları yazdırmak ve duygu analizinin incelemeyle eşleşip eşleşmediğini görmek isterseniz:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Dosyayı zorlukta kullanmadan önce yapmanız gereken son şey, onu kaydetmektir! Ayrıca, yeni sütunlarınızı yeniden sıralamayı düşünmelisiniz, böylece çalışması daha kolay olur (insan için, bu sadece kozmetik bir değişikliktir).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[Analiz defterinin](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) tüm kodunu çalıştırmalısınız (Hotel_Reviews_Filtered.csv dosyasını oluşturmak için [filtreleme defterinizi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) çalıştırdıktan sonra).

Adımları gözden geçirmek gerekirse:

1. Orijinal veri seti dosyası **Hotel_Reviews.csv**, önceki derste [keşif defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ile incelenmiştir.
2. Hotel_Reviews.csv, [filtreleme defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) tarafından filtrelenerek **Hotel_Reviews_Filtered.csv** dosyasına dönüştürülmüştür.
3. Hotel_Reviews_Filtered.csv, [duygu analizi defteri](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) tarafından işlenerek **Hotel_Reviews_NLP.csv** dosyasına dönüştürülmüştür.
4. Aşağıdaki NLP Zorluğunda **Hotel_Reviews_NLP.csv** dosyasını kullanın.

### Sonuç

Başladığınızda, sütunlar ve veriler içeren bir veri setiniz vardı, ancak bunların hepsi doğrulanabilir veya kullanılabilir değildi. Verileri incelediniz, ihtiyacınız olmayanları filtrelediniz, etiketleri işe yarar bir şeye dönüştürdünüz, kendi ortalamalarınızı hesapladınız, bazı duygu sütunları eklediniz ve umarım doğal metin işleme hakkında ilginç şeyler öğrendiniz.

## [Ders sonrası test](https://ff-quizzes.netlify.app/en/ml/)

## Zorluk

Artık veri setinizin duygu analizini yaptığınıza göre, bu müfredatta öğrendiğiniz stratejileri (belki kümeleme?) kullanarak duygu etrafındaki kalıpları belirlemeye çalışın.

## Gözden Geçirme ve Kendi Kendine Çalışma

[Duygu analizi hakkında daha fazla bilgi edinmek ve metindeki duyguları keşfetmek için farklı araçlar kullanmak üzere bu Learn modülünü](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) alın.

## Ödev 

[Farklı bir veri seti deneyin](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.