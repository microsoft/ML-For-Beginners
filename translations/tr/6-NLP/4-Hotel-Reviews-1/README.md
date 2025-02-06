# Otel Yorumlarıyla Duygu Analizi - Verilerin İşlenmesi

Bu bölümde, önceki derslerde öğrendiğiniz teknikleri kullanarak büyük bir veri seti üzerinde keşifsel veri analizi yapacaksınız. Çeşitli sütunların faydasını iyi anladığınızda, şunları öğreneceksiniz:

- Gereksiz sütunları nasıl kaldıracağınızı
- Mevcut sütunlara dayanarak bazı yeni verileri nasıl hesaplayacağınızı
- Sonuçta oluşan veri setini nihai zorlukta kullanmak için nasıl kaydedeceğinizi

## [Ders Öncesi Quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Giriş

Şu ana kadar metin verilerinin sayısal veri türlerinden oldukça farklı olduğunu öğrendiniz. Eğer bu metin bir insan tarafından yazılmış veya söylenmişse, desenleri ve frekansları, duygu ve anlamı bulmak için analiz edilebilir. Bu ders sizi gerçek bir veri seti ve gerçek bir zorlukla tanıştıracak: **[515K Avrupa'daki Otel Yorumları Verisi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** ve bir [CC0: Kamu Malı lisansı](https://creativecommons.org/publicdomain/zero/1.0/) içerir. Bu veri seti Booking.com'dan kamuya açık kaynaklardan toplanmıştır. Veri setinin yaratıcısı Jiashen Liu'dur.

### Hazırlık

İhtiyacınız olacaklar:

* Python 3 kullanarak .ipynb not defterlerini çalıştırma yeteneği
* pandas
* NLTK, [yerel olarak yüklemeniz gereken](https://www.nltk.org/install.html)
* Kaggle'da bulunan veri seti [515K Avrupa'daki Otel Yorumları Verisi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Yaklaşık 230 MB açılmış haliyle. Bu NLP dersleriyle ilişkili kök `/data` klasörüne indirin.

## Keşifsel veri analizi

Bu zorluk, duygu analizi ve misafir yorum puanlarını kullanarak bir otel öneri botu oluşturduğunuzu varsayar. Kullanacağınız veri seti, 6 şehirdeki 1493 farklı otelin yorumlarını içerir.

Python, otel yorumları veri seti ve NLTK'nın duygu analizini kullanarak şunları bulabilirsiniz:

* Yorumlarda en sık kullanılan kelimeler ve ifadeler nelerdir?
* Bir oteli tanımlayan resmi *etiketler* yorum puanlarıyla ilişkilendiriliyor mu (örneğin, belirli bir otel için *Küçük çocuklu aile* için daha olumsuz yorumlar mı var, belki de *Yalnız gezginler* için daha iyi olduğunu gösteriyor?)
* NLTK duygu puanları otel yorumcunun sayısal puanıyla 'uyuşuyor' mu?

#### Veri Seti

İndirdiğiniz ve yerel olarak kaydettiğiniz veri setini keşfedelim. Dosyayı VS Code veya hatta Excel gibi bir editörde açın.

Veri setindeki başlıklar şu şekildedir:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

İncelemesi daha kolay olacak şekilde gruplandırılmışlardır:
##### Otel Sütunları

* `Hotel_Name`, `Hotel_Address`, `lat` (enlem), `lng` (boylam)
  * *lat* ve *lng* kullanarak otel konumlarını gösteren bir harita çizebilirsiniz (belki olumsuz ve olumlu yorumlar için renk kodlu olarak)
  * Hotel_Address bize açıkça yararlı değil ve muhtemelen daha kolay sıralama ve arama için bir ülkeyle değiştireceğiz

**Otel Meta-yorum Sütunları**

* `Average_Score`
  * Veri seti yaratıcısına göre, bu sütun *otel puanının, son yılın en son yorumuna dayalı olarak hesaplanan ortalama puanı* anlamına gelir. Bu puanı hesaplamak için alışılmadık bir yol gibi görünüyor, ancak şimdilik bu veriyi yüzeyde kabul edebiliriz.
  
  ✅ Bu verideki diğer sütunlara dayanarak, ortalama puanı hesaplamak için başka bir yol düşünebilir misiniz?

* `Total_Number_of_Reviews`
  * Bu otelin aldığı toplam yorum sayısı - bu veri setindeki yorumlara mı atıfta bulunuyor (kod yazmadan) net değil.
* `Additional_Number_of_Scoring`
  * Bu, bir yorum puanı verildiği, ancak yorumcunun olumlu veya olumsuz bir yorum yazmadığı anlamına gelir.

**Yorum Sütunları**

- `Reviewer_Score`
  - Bu, min ve max değerleri 2.5 ile 10 arasında olan en fazla 1 ondalık basamağa sahip bir sayısal değerdir
  - Neden 2.5'in en düşük puan olduğu açıklanmamış
- `Negative_Review`
  - Bir yorumcu hiçbir şey yazmazsa, bu alan "**No Negative**" olacaktır
  - Bir yorumcu olumsuz yorum sütununda olumlu bir yorum yazabilir (örneğin, "bu otelde kötü bir şey yok")
- `Review_Total_Negative_Word_Counts`
  - Yüksek olumsuz kelime sayıları daha düşük bir puanı gösterir (duygusallığı kontrol etmeden)
- `Positive_Review`
  - Bir yorumcu hiçbir şey yazmazsa, bu alan "**No Positive**" olacaktır
  - Bir yorumcu olumlu yorum sütununda olumsuz bir yorum yazabilir (örneğin, "bu otelde hiç iyi bir şey yok")
- `Review_Total_Positive_Word_Counts`
  - Yüksek olumlu kelime sayıları daha yüksek bir puanı gösterir (duygusallığı kontrol etmeden)
- `Review_Date` ve `days_since_review`
  - Bir yorumun tazeliği veya bayatlığı ölçüsü uygulanabilir (daha eski yorumlar, otel yönetimi değiştiği veya yenilemeler yapıldığı için, veya bir havuz eklendiği için, daha yeni yorumlar kadar doğru olmayabilir)
- `Tags`
  - Bunlar, bir yorumcunun misafir türünü (örneğin, yalnız veya aile) tanımlamak için seçebileceği kısa tanımlayıcılardır, sahip oldukları oda türü, kalış süresi ve yorumun nasıl gönderildiği.
  - Ne yazık ki, bu etiketleri kullanmak sorunludur, kullanışlılıklarını tartışan aşağıdaki bölüme bakın

**Yorumcu Sütunları**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Bu, bir öneri modelinde bir faktör olabilir, örneğin, yüzlerce yorumu olan daha üretken yorumcuların daha olumsuz olmaktan ziyade daha olumlu olma olasılığını belirleyebilseydiniz. Ancak, belirli bir yorumun yorumcusu benzersiz bir kodla tanımlanmadığından, bir yorum setiyle ilişkilendirilemez. 100 veya daha fazla yorumu olan 30 yorumcu var, ancak bunun öneri modeline nasıl yardımcı olabileceğini görmek zor.
- `Reviewer_Nationality`
  - Bazı insanlar, belirli milliyetlerin ulusal bir eğilim nedeniyle olumlu veya olumsuz bir yorum yapma olasılığının daha yüksek olduğunu düşünebilir. Bu tür anekdot görüşleri modellerinize dahil ederken dikkatli olun. Bunlar ulusal (ve bazen ırksal) klişelerdir ve her yorumcu, deneyimlerine dayalı olarak bir yorum yazan bireydi. Önceki otel konaklamaları, seyahat edilen mesafe ve kişisel mizacı gibi birçok mercekten filtrelenmiş olabilir. Bir yorum puanının nedeni olarak milliyetlerini düşünmek zor.

##### Örnekler

| Ortalama Puan | Toplam Yorum Sayısı | Yorumcu Puanı | Olumsuz <br />Yorum                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Olumlu Yorum                      | Etiketler                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Bu şu anda bir otel değil, bir inşaat alanı Uzun bir yolculuktan sonra dinlenirken ve odada çalışırken sabah erken saatlerde ve tüm gün boyunca kabul edilemez inşaat gürültüsü ile terörize edildim İnsanlar tüm gün boyunca bitişik odalarda matkaplarla çalışıyordu Oda değişikliği talep ettim ama sessiz bir oda mevcut değildi Durumu daha da kötüleştirmek için fazla ücret alındım Akşam erken bir uçuşa çıkmak zorunda olduğum için akşam çıkış yaptım ve uygun bir fatura aldım Bir gün sonra otel benim rızam olmadan rezervasyon fiyatının üzerinde başka bir ücret aldı Berbat bir yer Kendinize burada rezervasyon yaparak ceza vermeyin | Hiçbir şey Berbat bir yer Uzak durun | İş gezisi                                Çift Standart Çift Kişilik Oda 2 gece kaldı |

Gördüğünüz gibi, bu misafir otelde mutlu bir konaklama yapmamış. Otelin 7.8 gibi iyi bir ortalama puanı ve 1945 yorumu var, ancak bu yorumcu 2.5 puan vermiş ve konaklamalarının ne kadar olumsuz olduğuna dair 115 kelime yazmış. Pozitif_Yorum sütununda hiçbir şey yazmadılarsa, hiçbir şeyin pozitif olmadığını varsayabilirsiniz, ama ne yazık ki 7 kelimelik bir uyarı yazmışlar. Yalnızca kelimeleri sayarsak, yorumcunun niyetinin anlamı veya duygusu yerine, çarpık bir görüşe sahip olabiliriz. Garip bir şekilde, 2.5 puanı kafa karıştırıcı, çünkü otel konaklaması bu kadar kötüyse, neden hiç puan verilsin ki? Veri setini yakından incelediğinizde, en düşük olası puanın 2.5 olduğunu, 0 olmadığını göreceksiniz. En yüksek olası puan ise 10.

##### Etiketler

Yukarıda belirtildiği gibi, ilk bakışta `Tags` kullanarak verileri kategorize etme fikri mantıklı geliyor. Ne yazık ki bu etiketler standartlaştırılmamış, bu da belirli bir otelde seçeneklerin *Tek kişilik oda*, *İkiz oda* ve *Çift kişilik oda* olabileceği, ancak bir sonraki otelde *Deluxe Tek Kişilik Oda*, *Klasik Kraliçe Odası* ve *Executive King Odası* olabileceği anlamına gelir. Bunlar aynı şeyler olabilir, ancak o kadar çok varyasyon var ki seçim şu şekilde olur:

1. Tüm terimleri tek bir standarda dönüştürmeye çalışmak, bu çok zor çünkü her durumda dönüşüm yolunun ne olacağı net değil (örneğin, *Klasik tek kişilik oda* *Tek kişilik oda*ya eşlenebilir, ancak *Avlu Bahçesi veya Şehir Manzaralı Superior Kraliçe Odası* çok daha zor eşlenir)

2. Bir NLP yaklaşımı benimseyebiliriz ve her otelde *Yalnız*, *İş Seyahatçisi* veya *Küçük çocuklu aile* gibi belirli terimlerin sıklığını ölçebiliriz ve bunu öneriye dahil edebiliriz

Etiketler genellikle (ama her zaman değil) *Gezi türü*, *Misafir türü*, *Oda türü*, *Gece sayısı* ve *Yorumun gönderildiği cihaz türü* ile uyumlu 5 ila 6 virgülle ayrılmış değer içeren tek bir alan içerir. Ancak, bazı yorumcular her alanı doldurmazsa (birini boş bırakabilirler), değerler her zaman aynı sırada olmaz.

Bir örnek olarak, *Grup türü* alın. `Tags` sütununda bu alanda 1025 benzersiz olasılık vardır ve ne yazık ki bunların yalnızca bazıları bir grubu ifade eder (bazıları oda türüdür vb.). Yalnızca aileden bahsedenleri filtrelerseniz, sonuçlar birçok *Aile odası* türü sonuç içerir. *ile* terimini dahil ederseniz, yani *Küçük çocuklu aile* veya *Büyük çocuklu aile* ifadelerini sayarsanız, sonuçlar daha iyi olur ve 515.000 sonucun 80.000'inden fazlası "Küçük çocuklu aile" veya "Büyük çocuklu aile" ifadesini içerir.

Bu, etiketler sütununun tamamen işe yaramaz olmadığı anlamına gelir, ancak işe yarar hale getirmek için biraz çalışma gerekecektir.

##### Ortalama otel puanı

Veri setiyle ilgili anlayamadığım bazı tuhaflıklar veya tutarsızlıklar var, ancak modellerinizi oluştururken bunların farkında olmanız için burada gösterilmiştir. Çözerseniz, lütfen tartışma bölümünde bize bildirin!

Veri seti, ortalama puan ve yorum sayısıyla ilgili aşağıdaki sütunlara sahiptir:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Bu veri setindeki en fazla yoruma sahip tek otel *Britannia International Hotel Canary Wharf* 4789 yorumla 515.000'den. Ancak bu otel için `Total_Number_of_Reviews` değerine bakarsak, 9086'dır. Belki daha fazla yorumsuz puan olduğunu varsayabilirsiniz, bu yüzden `Additional_Number_of_Scoring` sütun değerini eklemeliyiz. Bu değer 2682'dir ve 4789'a eklediğimizde 7471 olur, bu da hala `Total_Number_of_Reviews`'dan 1615 eksiktir.

`Average_Score` sütunlarını alırsanız, bunun veri setindeki yorumların ortalaması olduğunu varsayabilirsiniz, ancak Kaggle açıklaması "*otel puanının, son yılın en son yorumuna dayalı olarak hesaplanan ortalama puanı*"dır. Bu pek yararlı görünmüyor, ancak veri setindeki yorum puanlarına dayalı olarak kendi ortalamamızı hesaplayabiliriz. Aynı oteli örnek olarak kullanarak, otelin ortalama puanı 7.1 olarak verilir, ancak hesaplanan puan (veri setindeki ortalama yorumcu puanı) 6.8'dir. Bu yakın, ancak aynı değer değil ve `Additional_Number_of_Scoring` yorumlarında verilen puanların ortalamayı 7.1'e yükselttiğini tahmin edebiliriz. Ne yazık ki, bu iddiayı test etmenin veya kanıtlamanın bir yolu olmadığından, `Average_Score`, `Additional_Number_of_Scoring` ve `Total_Number_of_Reviews`'ya dayanan veya atıfta bulunan verileri kullanmak veya güvenmek zor.

İşleri daha da karmaşık hale getirmek için, en fazla yoruma sahip ikinci otel hesaplanan ortalama puanı 8.12'dir ve veri setindeki `Average_Score` 8.1'dir. Bu doğru puan bir tesadüf mü yoksa ilk otel bir tutarsızlık mı?

Bu otelin bir aykırı değer olabileceği ve belki de çoğu değerin (bazıları bir nedenle değil) tutarlı olduğu varsayımıyla, veri setindeki değerleri keşfetmek ve değerlerin doğru kullanımını (veya kullanılmamasını) belirlemek için bir sonraki kısa programı yazacağız.

> 🚨 Bir uyarı notu
>
> Bu veri setiyle çalışırken, metni kendiniz okumadan veya analiz etmeden bir şeyler hesaplayan kod yazacaksınız. Bu, NLP'nin özü, bir insanın yapmasına gerek kalmadan anlam veya duyguyu yorumlamak. Ancak, bazı olumsuz yorumları okumanız mümkün. Okumanız gerekmediği için size bunu yapmamanızı tavsiye ederim. Bazıları saçma veya alakasız olumsuz otel yorumlarıdır, örneğin "Hava iyi değildi", otelin veya herhangi birinin kontrolü dışında bir şey. Ancak, bazı yorumların karanlık bir tarafı da var. Bazen olumsuz yorumlar ırkçı, cinsiyetçi veya yaşçı olabilir. Bu, bir kamu web sitesinden kazınan bir veri setinde beklenebilir. Bazı yorumcular, hoşlanmayacağınız, rahatsız edici veya üzücü bulacağınız yorumlar bırakır. Kodun duyguyu ölçmesine izin vermek daha iyidir, kendiniz okuyup üzülmektense. Bu, bu tür şeyleri yazanların azınlıkta olduğu anlamına gelir, ancak yine de varlar.

## Alıştırma - Veri keşfi
### Veriyi yükleyin

Veriyi görsel olarak incelemek yeterli, şimdi biraz kod yazacak ve bazı cevaplar alacaksınız! Bu bölüm pandas kütüphanesini kullanır. İlk göreviniz, CSV verilerini yükleyip okuyabileceğinizden emin olmaktır. Pandas kütüphanesi hızlı bir CSV yükleyiciye sahiptir ve sonuç, önceki derslerde olduğu gibi bir veri çerçevesine yerleştirilir. Yüklediğimiz CSV, yarım milyondan fazla satır içerir, ancak sadece 17 sütun vardır. Pandas, bir veri çerçevesiyle etkileşimde bulunmak için birçok güçlü yol sunar, her satırda işlemler yapma yeteneği de dahil.

Bu dersten itibaren, kod parçacıkları ve kodun bazı açıklamaları ve sonuçların ne anlama geldiği hakkında bazı tartışmalar olacaktır. Kodunuz için dahil edilen _notebook.ipynb_ dosyasını kullanın.

Kullanacağınız veri dos
rows have column `Positive_Review` değerleri "No Positive" 9. Sütun `Positive_Review` değerleri "No Positive" **ve** `Negative_Review` değerleri "No Negative" olan kaç satır olduğunu hesaplayın ve yazdırın ### Kod cevapları 1. Yeni yüklediğiniz veri çerçevesinin *şeklini* yazdırın (şekil satır ve sütun sayısıdır) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Yorumcu milliyetlerinin frekans sayısını hesaplayın: 1. `Reviewer_Nationality` sütunu için kaç farklı değer var ve bunlar nelerdir? 2. Veri setinde en yaygın olan yorumcu milliyeti nedir (ülke ve yorum sayısını yazdırın)? ```python
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
   ``` 3. En sık bulunan bir sonraki 10 milliyet ve frekans sayıları nelerdir? ```python
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
      ``` 3. İlk 10 yorumcu milliyetinin her biri için en sık yorumlanan otel hangisiydi? ```python
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
   ``` 4. Veri setinde otel başına kaç yorum var (otel frekans sayısı)? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | Veri setinde *sayılmış* sonuçların `Total_Number_of_Reviews` değerleriyle eşleşmediğini fark edebilirsiniz. Bu değerin veri setinde otelin sahip olduğu toplam yorum sayısını temsil edip etmediği veya hepsinin kazınmamış olup olmadığı veya başka bir hesaplama olup olmadığı belirsizdir. Bu belirsizlik nedeniyle `Total_Number_of_Reviews` modelde kullanılmamaktadır. 5. Veri setindeki her otel için bir `Average_Score` sütunu olmasına rağmen, her otel için tüm yorumcu puanlarının ortalamasını alarak bir ortalama puan da hesaplayabilirsiniz. Veri çerçevenize `Calc_Average_Score` başlıklı yeni bir sütun ekleyin ve bu hesaplanmış ortalamayı içeren sütunu ekleyin. `Hotel_Name`, `Average_Score` ve `Calc_Average_Score` sütunlarını yazdırın. ```python
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
   ``` `Average_Score` değerinin ve hesaplanan ortalama puanın neden bazen farklı olduğunu merak edebilirsiniz. Bazı değerlerin neden eşleştiğini, ancak diğerlerinde neden bir fark olduğunu bilemediğimiz için, bu durumda kendimiz ortalamayı hesaplamak en güvenli yoldur. Bununla birlikte, farklar genellikle çok küçüktür, işte veri seti ortalaması ile hesaplanan ortalama arasındaki en büyük sapma olan oteller: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | Sadece 1 otelin puan farkının 1'den büyük olması, farkı görmezden gelip hesaplanan ortalama puanı kullanabileceğimiz anlamına gelir. 6. Sütun `Negative_Review` değerleri "No Negative" olan kaç satır olduğunu hesaplayın ve yazdırın 7. Sütun `Positive_Review` değerleri "No Positive" olan kaç satır olduğunu hesaplayın ve yazdırın 8. Sütun `Positive_Review` değerleri "No Positive" **ve** `Negative_Review` değerleri "No Negative" olan kaç satır olduğunu hesaplayın ve yazdırın ```python
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
   ``` ## Başka bir yol Lambdas kullanmadan öğeleri saymanın başka bir yolu ve satırları saymak için sum kullanın: ```python
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
   ``` `Negative_Review` ve `Positive_Review` sütunları için sırasıyla "No Negative" ve "No Positive" değerlerine sahip 127 satır olduğunu fark etmiş olabilirsiniz. Bu, yorumcunun otele bir sayısal puan verdiği, ancak olumlu veya olumsuz bir yorum yazmaktan kaçındığı anlamına gelir. Neyse ki bu küçük bir satır miktarıdır (515738'den 127, yani %0.02), bu yüzden modelimizi veya sonuçlarımızı belirli bir yöne çekmeyecektir, ancak bir yorum veri setinin yorum içermeyen satırlara sahip olmasını beklemeyebilirsiniz, bu yüzden bu tür satırları keşfetmek için verileri keşfetmeye değer. Veri setini keşfettiğinize göre, bir sonraki derste verileri filtreleyecek ve bazı duygu analizleri ekleyeceksiniz. --- ## 🚀Meydan okuma Bu ders, önceki derslerde gördüğümüz gibi, verilerinizi ve özelliklerini anlamanın ne kadar kritik derecede önemli olduğunu gösterir. Özellikle metin tabanlı veriler dikkatli bir inceleme gerektirir. Çeşitli metin ağırlıklı veri setlerini inceleyin ve bir modele önyargı veya çarpık duygu ekleyebilecek alanları keşfedip edemeyeceğinizi görün. ## [Ders sonrası sınav](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## İnceleme ve Kendi Kendine Çalışma [Bu NLP Öğrenme Yolunu](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) alın ve konuşma ve metin ağırlıklı modeller oluştururken denemek için araçları keşfedin. ## Ödev [NLTK](assignment.md)

**Feragatname**:
Bu belge, makine tabanlı AI çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluğa özen göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal diliyle yazılmış hali, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilmektedir. Bu çevirinin kullanılmasından kaynaklanabilecek herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.