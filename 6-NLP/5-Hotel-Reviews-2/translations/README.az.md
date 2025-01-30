# Otel rəyləri ilə fikir analizi

Sən artıq dataseti ətraflı kəşf etmisən və indi sütunları filtrasiya edərək yeni məlumatlar toplamaq üçün NLP texnikalarını tətbiq edəcəksən.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/?loc=az)

### Filtrasiya və fikir analizi əməliyyatları

Sən yəqin datasetin bəzi problemləri olduğuna şahid olmusan. Bəzi sütunlar mənasız və düzgün olmayan məlumatlarla doldurulub. Əgər onlar doğrudurlarsa, onları necə hesablamalı olduğumuz aydın deyil və azad şəkildə öz hesablamalarımızla cavab əldə edə bilməyəcəyik.

## Tapşırıq: datanı biraz emal et

Datanı biraz təmizlə. Sonradan faydalı ola biləcək sütunlar əlavə et, digər sütunlardakı məlumatları dəyiş və bəzi sütunları ümumiyyətlə ləğv et.

1. İlkin sütun emalı

   1. `lat` və `lng` sütunlarını silin

   2. `Hotel_Address` dəyərlərini aşağıdakı dəyərlərlə dəyiş (əgər ünvanda seçilmiş şəhər və ölkə adından biri varsa, onları sadəcə şəhər və ölkə adı birləşməsinə dəyiş).

      Datasetdə yalnız bu şəhər və ölkələr vardır:

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

      # Bütün ünvanları daha qısa və yararlı dəyərlərlə əvəzlə
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # value_counts() cəmi yekun rəy sayına bərabər olmalıdır
      print(df["Hotel_Address"].value_counts())
      ```

      Artıq ölkə səviyyəsində sorğu yarada bilərsən:

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

2. Otel rəyləri ilə əlaqəli sütunları emal et

  1. `Additional_Number_of_Scoring` sütununu sil

  1. `Total_Number_of_Reviews` sütunundakı dəyərləri otelin cari datasetdəki rəylərin ümumi sayı ilə əvəz et

  1. `Average_Score` öz hesabladığımız orta bal ilə əvəz et

  ```python
  # `Additional_Number_of_Scoring` sil
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # `Total_Number_of_Reviews` və `Average_Score` öz hesabladığımız dəyərlə əvəzlə
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Rəy sütunlarını emal et

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` və `days_since_review` sütunlarını sil

   2. `Reviewer_Score`, `Negative_Review`, və `Positive_Review` olduğu kimi saxla,

   3. `Tags` hələlik saxla

     - Biz növbəti fikirədə tqqlar üzərində bəzi filtrasiya əməliyyatları aparacağıq və spnra tqqları siləcəyik

4. Rəybildirən sütununu sil

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` sil

  2. `Reviewer_Nationality` saxla

### Teq sütunları

`Tag` sütunu daxilində mətnləri siyahı kimi saxladığı üçün problemlidir. Təəssüf ki, bu sütun daxilindəki dəyərlərin sırası və sayı həmişə eyni olmur. Bu halda insan tərəfindən ona maraqlı olan və düzgün sözü tapmaq çətinləşir, çünki burada 515,000 sətir, 1427 otel və hər birində rəybildirənin seçimləri çox az fərqlənir. Burada NLP öz sözünü deyir. Sən mətnləri skan edə və ən çox istifadə olunan sözləri seçə və saya bilərsən.

Təəssüf ki, biz 1 söz yerinə çoxlu söz birləşməsi ilə maraqlanırıq (misal üçün, *Biznes səyahəti*). Çoxlu söz birləşmələrinin istifadə tezliyinin paylanma alqoritmini bu qədər böyük data (6762646 söz) üzərində icra etmək ağlasığmaz qədər vaxt aparar, lakin bütün data üzərindən keçmədən bunu etmək də mümkün deyil. Burada kəşfiyyatçı data analizi köməyimizə çata bilər, çünki bizim əlimizdə artıq `[' Biznes səyahəti  ', ' Yalnız səyahətçi ', ' Tək otaq ', ' 5 günlük qalmaq ', ' Mobil cihazdan sorğulanıb ']` kimi nümunə teqlər var və bunlar bizim emal üçün lazım olan zamanı kifayət qədər aşağı salacaq. Bunu bildiyimiz üçün şanslıyıq, lakin bizə maraqlı olan teqlərı təyin etmək üçün bəzi addımları izləməliyik.

### Teqlərin filtrasiyası

Yadda saxlayaq ki, datasetin məqsədi sənin ən yaxşı oteli seçməyinə kömək etməsi üçün fikir və sütunlar əlavə etməkdir (özün və ya müştərin üçün otel tövsiyyəsi edəcək bot hazırlamaq kimi tapşırığın var). Özünə hansı teqlərin yekun datasetdə mənalı olub-olmayacağı barədə sual ver. Nümunə bir ssenari (əgər sənə başqa məqsəqlər üçün dataset lazımdırsa, hansı teqləri əlavə edə və ya çıxarmağın öz əlindədir):

1. Hansı növ səyahət uyğundursa, onlar qalmalıdır
2. Qonaq qrupunun tipi vacibdirsə, onlar qalmalıdır
3. Qonağın qaldığı məkanın növü (otaq, ev, studio) maraqlı deyil (bütün otellərdə, demək olar ki, eyni otaqlar var)
4. Hansı cihazdan rəy bildirməsi maraqlı deyil
5. Əgər uzun müddətli səyahətlə maraqlanırsansa, rəybildirənin neçə gecə qaldığı maraqlı *ola bilər*, lakin bu əlavə məsələlərdir, çox güman ki, uyğun olmayacaq

Yekun olaraq **2 teq növünü saxla və digərlərini sil**.

İlkin olaraq teqlərin daha yaxşı formata salmamış saymaq istəməyəcəksən, yəni mötərizələri və dırnaq işarələrini silməyin lazımdır. Bunu bir neçə üsulla edə bilərsən, lakin sənə bu qədər çox datanı çox vaxt sərf etmədən sürətli həll etmək lazımdır. Pandas kitabxanasında bu addımların hər birini icra etmək üçün asan yollar var.

```Python
# Açılan və bağlanan mötərizələri sil
df.Tags = df.Tags.str.strip("[']")
# Həmçinin dırnaqları da sil
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Hər teq buna bənzər bir hala çevriləcək: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

İndi başqa bir bir problemlə üzləşirik. Rəylərin (sətirlərin) bəzilərində 5, bəzilərində 3, digərlərində 6 sütun var. Dataset özü belə yaradılıb və bunu həll etmək çətindir. Hər bir sözün istifadə tezliyini saymaq istəyirsən, lakin onlar hər rəydə fərqli sıradadır. Sıralamanın standart olmaması otellərə daha layiqli olduğu teqləri mənsub etməkdə çətinlik yaradır.

Bunun yerinə biz sıralamanı özümüzə sərf edən vəziyyətə çevirə bilərik. Belə ki, hər teq söz birləşməsindən ibarət olsa da vergüllə ayrılır! Ən sadə yolu müvəqqəti olaraq 6 sütun yaradıb hər birinə teq əlavə edə bilərik. Bundan sonra biz 6 sütunu böyük bir sütun içinə birləşdirə və `value_counts()` funksiyası ilə saya bilərik. Bunu icra etdikdən sonra 2428 vahid teq olduğunu görəcəksən. Kiçik nümunə:

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

`Submitted from a mobile device` kimi ümumi teqlərin bəziləri bizə əhəmiyyətli deyil. Belə teqləri silmək daha ağıllı bir addım kimi görünsə də, əməlliyyatlarımızın sürətinin çox olması bunu etməyi gərəksiz edir.

### Qonaqlama günlərini silmək

Bu teqləri silmək birinci addımdır və bizə lazım olan teqlərin sayını azaltmaqda kömək edəcək. Diqqətdə saxlamaq lazımdır ki, biz bu məlumatlarını datasetdən silmirik, sadəcə emal etdiyimiz rəylər datasetində ancaq saymaq üçün lazım olanları saxlayırıq.

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

Burada müxtəlif sayda otaqlar, suitlər, evlər, studiyalar və s. var. Bunların hamısı demək olar ki, eyni məna daşıyırlar, buna görə onları da silək.

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

Yekun olaraq bu zövqlü emaldan sonra bizə aşağıdakı *faydalı* teqlər qaldı:

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

`Travellers with friends` və `Group` teqləri bir-birinə çox yaxındır düşünə və ədalətli olması üçün bunları birləşdirə də bilərsən. Düzgün teqlərin təyin olması üçün yazılmış kodu [Teqlər notbukundan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) əldə edə bilərsən.

Son addımımız hər teq üçün yeni sütunun yaradılması olacaq. Bundan sonra hər bir rəy sətirində `Tag` sütununa uyğun gəldiyi halda 1 əlavə edəcəksən, gəlmədiyi zaman isə 0. Yekun nəticədə bu saylar istifadəçilərə otelin hansı məqsəd üçün daha uyğun olduğunu təyin etməyə faydalı olacaq, misal üçün daha işgüzar yoxsa istirahət səfərləri üçün uyğun olduğunu və ya ev heyvanına icazə verildiyini biləcəklər.

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

### Faylı yadda saxla

Nəhayət ki, dataseti yadda saxlayıb yeni oyuna başlaya bilərsən

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# yeni datanın hesablanmış sütunlarla yadda saxlanılması
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Fikir analizi əməliyyatları

Bu son bölmədə rəy sütunlarına fikir analizi tətbiq edib nəticələri datasetdə saxlayacağıq.

## Tapşırıq: filtrlənmiş datanı yüklə və yadda saxla

Nəzərə al ki, sən indi son bölmədən əldə olunmuş filtrlənmiş dataseti yükləyirsən, orijinal dataseti **yox**.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# CSV faylından filtlənmiş otel rəylərini yüklə
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# Sənin kodun bura daxil edilməlidir


# Son olaraq xatırla ki, otel rəylərini yeni NLP data əlavə edib yadda saxlamalısan
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Stop sözlərin silinməsi

Əgər sən Mənfi və Müsbət rəy sütunları üzərində Fikir Analizi aparırsansa, bu çox çəkə bilər. Güclü notbukların sürətli CPU komponenti ilə test edilib ki, istifadə olunan fikir analizi kitabxanasından asılı olaraq bu 12-14 dəqiqə çəkə bilir. Bu (nisbətən) çox zamandır, buna görə əməliyyatları necə sürətləndirə biləcəyimiz barədə düşünməyə dəyər.

Stop sözləri və ya cümlənin məqsədini dəyişdirməyən ümumi İnglis sözlərini silmək birinci addım olacaq. Bunları yığışdırmaqla fikir analizimiz daha sürətli icra olunacaq və dəqiqlikdə geri qalmayacaq (stop sözlər cümlənin əsas fikrini dəyişmir, sadəcə analizin sürətini yavaşladır).

Ən uzun mənfi rəy 395 sözdür, lakin stop sözləri yığışdırdıqdan sonra bu 195 oldu.

Stop sözləri silmək özü sürətli əməliyyatdır, 515,000 sətirlik data içində 2 rəy sütunundan stop sözlərin silinməsi test komputerində 3.3 saniyə çəkdi. Bu sənin cihazının CPU sürətindən, RAM və SSD olub-olmamasından və digər faktorlardan asılı olaraq aşağı-yuxarı cüzi olaraq fərq edə bilər. Əməliyyatın nisbi qısalığı fikir analizinin daha da yaxşılaşdırdığını və bunu etməyə dəydiyini göstərir.

```python
from nltk.corpus import stopwords

# CSV fayldan otel rəylərini yüklə
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Stop sözləri sil - çox sözlərə görə yavaş ola bilər!
# Ryan Xanın (ryanxjhan Kaggle-da) stop sözlərin silinməsində müxtəlif yanaşmaların performans fərqləri barədə dəyərli məqaləsi vardır
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # Ryanın tövsiyyəsinə uyğun olaraq bu yanaşmanı tətbiq edirik
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Stop sözləri hər iki sütundan sil
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Fikir analizinin aparılması

Artıq sən həm mənfi, həm də müsbət rəylərdə fikir analizi apara və nəticələri 2 yeni sütunda saxlaya bilərsən. Fikirlərin yoxlanılması rəy verənin həmin rəyə verdiyi bala uyğunluğunu müqayisə üçün istifadə olunacaq. Misal üçün, əgər fikir analizi nəticəsində mənfi rəyin fikrini 1 kimi qiymətləndirsə (çox müsbət fikir kimi) və ya müsbət rəyin fikrini 1 kimi qiymətləndirsə, lakin rəy verən otelə mümkün ən aşağı balı veribsə, deməli ya rəy verənin mətni verdiyi bala uyğun gəlmir, ya da bizim fikir analizimiz düzgün icra olunmayıb. Sən fikir analizindən çıxan nəticənin tam səhv ola biləcəyi ehtimalını qəbul etməlisən və bunu izah da etmək mümkündür. Misal üçün, rəyin özü çox kinayəli ola bilər - "Təbii ki mən istilik sistemi olmayan otaqda yatmağı ÇOX sevdim" və fikir analizimiz bunu təcrübəli insandan fərqli olaraq müsbət rəy kimi qəbul edə bilər.

NLTK müxtəlif fikir analizi edən funksiyalar təmin edir və sən onların nəticələnin nə qədər yaxşı və ya pis olduğunuz yoxlayaraq öyrənə bilərsən. Burada VADER fikir analiz üsulu istifadə olunmuşdur.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Sosial Media Mətninin Fikir Təhlili üçün Parsimon Qaydaya əsaslanan Model. Vebloqlar və Sosial Media üzrə Səkkizinci Beynəlxalq Konfrans (ICWSM-14). Ann Arbor, MI, İyun 2014.

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
Daha sonra fikir analizi hesablamasını hər rəy üçün aşağıdakı kimi tətbiq edə bilərsən:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Bu bizim kompüterimizdə təxmini 120 saniyə çəkdi, lakin bu hər cihaza uyğun dəyişə bilər. Əgər nəticələri çap edib rəylərin fikirlərlə uyğunlaşmasını yoxlamaq istəyirsənsə, aşağıdakı kodu tətbiq edə bilərsən:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```
Bu faylı tapşırıqda istifadə etməzdən əvvəl ediləcək ən son şey bunun yadda saxlamaqdır! Sən həmçinin bütün yeni sütunları daha rahat istifadə edilə bilməsi üçün yerlərini dəyişə bilərsən (bu yalnız insanlar üçün olan kosmetik dəyişiklikdir).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Sən bütün kodu [analiz notbukunda](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) icra etməlisən (yalnız [filtrasiya notbukunu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) Hotel_Reviews_Filtered.csv faylını yaratmaq üçün icra etdikdən sonra).

İzləmək üçün düzgün sıralanmış addımlar aşağıdakı kimidir:

1. Original dataset faylı **Hotel_Reviews.csv** əvvəlki dərsdə [kəşfetmə notbukunda](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) analiz olunub
2. Hotel_Reviews.csv faylı [filtrasiya notbuku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ilə filtrlənib və nəticə **Hotel_Reviews_Filtered.csv** faylına yazılıb
3. Hotel_Reviews_Filtered.csv faylı [fikir analizi notbuku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ilə emal olunub və nəticə **Hotel_Reviews_NLP.csv** faylına yazılıb
4. Hotel_Reviews_NLP.csv faylını aşağldakı NLP məşğələsində istifadə olunacaq

### Nəticə

Başlayanda sənin sütunlarla datasetin var idi və onların hamısının nə doğruluğunu yoxlamaq mümkün idi, nə də istifadəsi. Sən datanı kəşf etdin, lazımsız hissələri filtrlədin, taqları faydalı formaya saldın, ortalama nəticəni hesabladın, fikirlərin analizini bəzi sütunlara yazdın və ümid edirik ki, təbii mətnləri emal etməklə bağlı maraqlı nələrsə öyrəndin.

---
## 🚀 Məşğələ

Artıq sən bu datasetdə fikir analizini yerinə yetirdin, indi isə fikir ətrafında oxşarlıqları (modelləri) təyin etmək üçün bu kursda öyrəndiyin başqa hansı strategiyaları (klasterləşdirmə bəlkə?) tətbiq edə biləcəyini düşün.

## [Müharizə sonrası quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/?loc=az)

## Təkrarlayın və özünüz öyrənin

[Bu öyrənmə modulunu](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) götür və mətnlərdə fikir analizi üçün istifadə olunan müxtəlif alətləri öyrən.

## Tapşırıq

[Başqa dataset yoxla](assignment.az.md)
