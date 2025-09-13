<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T08:06:28+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "tr"
}
-->
# Otel Yorumlarıyla Duygu Analizi - Verilerin İşlenmesi

Bu bölümde, önceki derslerde öğrendiğiniz teknikleri kullanarak büyük bir veri seti üzerinde keşifsel veri analizi yapacaksınız. Çeşitli sütunların faydasını iyi bir şekilde anladıktan sonra şunları öğreneceksiniz:

- Gereksiz sütunların nasıl kaldırılacağını
- Mevcut sütunlara dayanarak yeni verilerin nasıl hesaplanacağını
- Ortaya çıkan veri setinin son zorlukta kullanılmak üzere nasıl kaydedileceğini

## [Ders Öncesi Testi](https://ff-quizzes.netlify.app/en/ml/)

### Giriş

Şimdiye kadar, metin verilerinin sayısal veri türlerinden oldukça farklı olduğunu öğrendiniz. Eğer bu metin bir insan tarafından yazılmış veya söylenmişse, desenler, sıklıklar, duygular ve anlamlar bulmak için analiz edilebilir. Bu ders sizi gerçek bir veri seti ve gerçek bir zorlukla tanıştırıyor: **[Avrupa'daki 515K Otel Yorumları Verisi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**. Bu veri seti [CC0: Kamu Malı Lisansı](https://creativecommons.org/publicdomain/zero/1.0/) ile lisanslanmıştır ve Booking.com'dan kamuya açık kaynaklardan toplanmıştır. Veri setinin yaratıcısı Jiashen Liu'dur.

### Hazırlık

İhtiyacınız olacaklar:

* Python 3 kullanarak .ipynb not defterlerini çalıştırma yeteneği
* pandas
* NLTK, [yerel olarak yüklemeniz gereken](https://www.nltk.org/install.html)
* Kaggle'dan indirilebilen veri seti [Avrupa'daki 515K Otel Yorumları Verisi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Sıkıştırılmış haliyle yaklaşık 230 MB boyutundadır. Bu NLP dersleriyle ilişkili kök `/data` klasörüne indirin.

## Keşifsel Veri Analizi

Bu zorluk, duygu analizi ve misafir yorum puanlarını kullanarak bir otel öneri botu oluşturduğunuzu varsayar. Kullanacağınız veri seti, 6 şehirdeki 1493 farklı otelin yorumlarını içerir.

Python, bir otel yorumları veri seti ve NLTK'nin duygu analizi kullanılarak şunları öğrenebilirsiniz:

* Yorumlarda en sık kullanılan kelime ve ifadeler nelerdir?
* Bir oteli tanımlayan resmi *etiketler*, yorum puanlarıyla ilişkili mi (örneğin, *Küçük çocuklu aile* için bir otelin daha olumsuz yorumları, *Yalnız gezgin* için olanlardan daha fazla mı, bu da otelin *Yalnız gezginler* için daha uygun olduğunu gösterebilir mi)?
* NLTK duygu puanları, otel yorumcunun sayısal puanıyla 'uyumlu' mu?

#### Veri Seti

Yerel olarak indirdiğiniz ve kaydettiğiniz veri setini keşfedelim. Dosyayı VS Code veya Excel gibi bir editörde açın.

Veri setindeki başlıklar şunlardır:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

İncelemesi daha kolay olması için şu şekilde gruplandırılmıştır:
##### Otel Sütunları

* `Hotel_Name`, `Hotel_Address`, `lat` (enlem), `lng` (boylam)
  * *lat* ve *lng* kullanarak, otel konumlarını gösteren bir harita oluşturabilirsiniz (belki olumsuz ve olumlu yorumlar için renk kodlaması yaparak)
  * Hotel_Address bizim için açıkça faydalı görünmüyor ve daha kolay sıralama ve arama için bunu bir ülke ile değiştirebiliriz.

**Otel Meta-Yorum Sütunları**

* `Average_Score`
  * Veri seti yaratıcısına göre, bu sütun *Otelin Ortalama Puanı, son bir yıldaki en son yorumlara dayanarak hesaplanmıştır*. Bu, puanı hesaplamak için alışılmadık bir yol gibi görünüyor, ancak bu veri kazındığı için şimdilik olduğu gibi kabul edebiliriz.
  
  ✅ Bu verilerdeki diğer sütunlara dayanarak, ortalama puanı hesaplamak için başka bir yol düşünebilir misiniz?

* `Total_Number_of_Reviews`
  * Bu otelin aldığı toplam yorum sayısı - bu, veri setindeki yorumlara mı atıfta bulunuyor (kod yazmadan) net değil.
* `Additional_Number_of_Scoring`
  * Bu, bir puan verildiği ancak yorumcunun olumlu veya olumsuz bir yorum yazmadığı anlamına gelir.

**Yorum Sütunları**

- `Reviewer_Score`
  - Bu, 2.5 ile 10 arasında en fazla 1 ondalık basamağa sahip bir sayısal değerdir.
  - Neden 2.5'in mümkün olan en düşük puan olduğu açıklanmamıştır.
- `Negative_Review`
  - Eğer bir yorumcu hiçbir şey yazmadıysa, bu alan "**No Negative**" içerecektir.
  - Bir yorumcu, Olumsuz yorum sütununda olumlu bir yorum yazabilir (örneğin, "bu otelde kötü bir şey yok").
- `Review_Total_Negative_Word_Counts`
  - Daha yüksek olumsuz kelime sayıları, daha düşük bir puanı gösterir (duygusallık kontrol edilmeden).
- `Positive_Review`
  - Eğer bir yorumcu hiçbir şey yazmadıysa, bu alan "**No Positive**" içerecektir.
  - Bir yorumcu, Olumlu yorum sütununda olumsuz bir yorum yazabilir (örneğin, "bu otelde hiç iyi bir şey yok").
- `Review_Total_Positive_Word_Counts`
  - Daha yüksek olumlu kelime sayıları, daha yüksek bir puanı gösterir (duygusallık kontrol edilmeden).
- `Review_Date` ve `days_since_review`
  - Bir yorum için tazelik veya bayatlık ölçüsü uygulanabilir (daha eski yorumlar, otel yönetimi değiştiği, yenilemeler yapıldığı veya bir havuz eklendiği için daha yeni yorumlar kadar doğru olmayabilir).
- `Tags`
  - Bunlar, bir yorumcunun kendilerini tanımlamak için seçebileceği kısa tanımlayıcılardır (örneğin, yalnız veya aile, oda türü, kalış süresi ve yorumun nasıl gönderildiği).
  - Ne yazık ki, bu etiketleri kullanmak sorunludur, aşağıdaki bölümde faydaları tartışılmıştır.

**Yorumcu Sütunları**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Bu, bir öneri modelinde bir faktör olabilir, örneğin, yüzlerce yorumu olan daha üretken yorumcuların olumsuz olmaktan çok olumlu olma olasılığının daha yüksek olduğunu belirleyebilirseniz. Ancak, herhangi bir belirli yorumun yorumcusu benzersiz bir kodla tanımlanmadığından, bir dizi yorumla ilişkilendirilemez. 100 veya daha fazla yorumu olan 30 yorumcu vardır, ancak bunun öneri modeline nasıl yardımcı olabileceğini görmek zordur.
- `Reviewer_Nationality`
  - Bazı insanlar, belirli milletlerin ulusal eğilimleri nedeniyle olumlu veya olumsuz bir yorum yapma olasılığının daha yüksek olduğunu düşünebilir. Modellerinize bu tür anekdot görüşleri dahil ederken dikkatli olun. Bunlar ulusal (ve bazen ırksal) stereotiplerdir ve her yorumcu, deneyimlerine dayanarak bir yorum yazan bireylerdir. Bu, önceki otel konaklamaları, seyahat edilen mesafe ve kişisel mizaç gibi birçok mercekten filtrelenmiş olabilir. Yorum puanının nedeni olarak milliyetlerini düşünmek zor bir iddiadır.

##### Örnekler

| Ortalama Puan | Toplam Yorum Sayısı | Yorumcu Puanı | Olumsuz <br />Yorum                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Olumlu Yorum                     | Etiketler                                                                                  |
| -------------- | ------------------- | ------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                | 2.5           | Bu şu anda bir otel değil, bir inşaat alanı. Uzun bir yolculuktan sonra dinlenirken ve odada çalışırken sabah erken saatlerden itibaren ve tüm gün boyunca kabul edilemez inşaat gürültüsüyle terörize edildim. İnsanlar tüm gün boyunca, yani bitişik odalarda matkaplarla çalışıyordu. Oda değişikliği talep ettim ancak sessiz bir oda mevcut değildi. Daha da kötüsü, fazla ücret alındım. Akşam saatlerinde erken bir uçuş için ayrılmam gerektiği için çıkış yaptım ve uygun bir fatura aldım. Bir gün sonra otel, rezervasyon fiyatını aşan bir tutarı izinsiz olarak tekrar tahsil etti. Bu korkunç bir yer. Kendinize ceza vermeyin, burada rezervasyon yapmayın. | Hiçbir şey. Korkunç bir yer. Uzak durun. | İş gezisi, Çift, Standart Çift Kişilik Oda, 2 gece konaklama |

Gördüğünüz gibi, bu misafir otelde mutlu bir konaklama geçirmemiş. Otelin 7.8 gibi iyi bir ortalama puanı ve 1945 yorumu var, ancak bu yorumcu 2.5 puan vermiş ve konaklamalarının ne kadar olumsuz olduğuna dair 115 kelime yazmış. Eğer Olumlu_Yorum sütununda hiçbir şey yazmamış olsaydı, hiçbir olumlu şey olmadığını varsayabilirdiniz, ancak ne yazık ki 7 kelimelik bir uyarı yazmışlar. Eğer kelimeleri saymak yerine kelimelerin anlamını veya duygusunu hesaba katmazsak, yorumcunun niyetine dair çarpık bir görüşe sahip olabiliriz. Garip bir şekilde, 2.5 puanı kafa karıştırıcı çünkü bu otel konaklaması bu kadar kötüydü, neden hiç puan versin? Veri setini yakından incelediğinizde, mümkün olan en düşük puanın 2.5, sıfır olmadığını göreceksiniz. En yüksek puan ise 10.

##### Etiketler

Yukarıda belirtildiği gibi, ilk bakışta `Tags` sütununu kullanarak verileri kategorize etmek mantıklı görünüyor. Ne yazık ki, bu etiketler standartlaştırılmamış, bu da şu anlama geliyor: bir otelde seçenekler *Tek kişilik oda*, *İkiz oda* ve *Çift kişilik oda* olabilirken, bir sonraki otelde *Deluxe Tek Kişilik Oda*, *Klasik Queen Oda* ve *Executive King Oda* olabilir. Bunlar aynı şeyler olabilir, ancak o kadar çok varyasyon var ki seçenek şu hale gelir:

1. Tüm terimleri tek bir standarda dönüştürmeye çalışmak, bu çok zordur çünkü her durumda dönüşüm yolunun ne olacağı açık değildir (örneğin, *Klasik tek kişilik oda* *Tek kişilik oda*ya eşlenebilir, ancak *Avlu Bahçesi veya Şehir Manzaralı Superior Queen Oda* eşlemesi çok daha zordur).

1. Bir NLP yaklaşımı benimseyip, her otel için *Yalnız*, *İş Seyahatinde*, veya *Küçük çocuklu aile* gibi belirli terimlerin sıklığını ölçebilir ve bunu öneri modeline dahil edebiliriz.

Etiketler genellikle (ancak her zaman değil) *Seyahat türü*, *Misafir türü*, *Oda türü*, *Gece sayısı* ve *Yorumun gönderildiği cihaz türü* ile hizalanan 5 ila 6 virgülle ayrılmış değerden oluşan tek bir alan içerir. Ancak, bazı yorumcular her alanı doldurmadığı için (birini boş bırakabilirler), değerler her zaman aynı sırada değildir.

Örneğin, *Grup türü* alın. Bu sütunda `Tags` alanında 1025 benzersiz olasılık vardır ve ne yazık ki bunların yalnızca bir kısmı bir gruba atıfta bulunur (bazıları oda türüdür vb.). Yalnızca aileyi belirtenleri filtrelerseniz, sonuçlar birçok *Aile odası* türü sonucu içerir. *ile* terimini dahil ederseniz, yani *Küçük çocuklu aile* veya *Büyük çocuklu aile* ifadelerini sayarsanız, sonuçlar daha iyidir ve 515.000 sonucun 80.000'inden fazlası bu ifadeleri içerir.

Bu, etiketler sütununun tamamen işe yaramaz olmadığını, ancak işe yarar hale getirmek için biraz çaba gerektirdiğini gösterir.

##### Ortalama Otel Puanı

Veri setiyle ilgili çözemedim ancak modellerinizi oluştururken farkında olmanız için burada gösterilen birkaç tuhaflık veya tutarsızlık vardır. Eğer çözerseniz, lütfen tartışma bölümünde bizimle paylaşın!

Veri seti, ortalama puan ve yorum sayısıyla ilgili şu sütunlara sahiptir:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Bu veri setindeki en fazla yoruma sahip tek otel, *Britannia International Hotel Canary Wharf* olup 515.000 yorumdan 4789'una sahiptir. Ancak, bu otel için `Total_Number_of_Reviews` değeri 9086'dır. Belki de birçok daha fazla puanın yorum içermediğini varsayabilirsiniz, bu yüzden belki `Additional_Number_of_Scoring` sütun değerini eklemeliyiz. Bu değer 2682'dir ve 4789'a eklenmesi 7471 eder, bu da hala `Total_Number_of_Reviews` değerinden 1615 eksiktir.

`Average_Score` sütununu alırsanız, bunun veri setindeki yorumların ortalaması olduğunu varsayabilirsiniz, ancak Kaggle açıklaması "*Otelin Ortalama Puanı, son bir yıldaki en son yorumlara dayanarak hesaplanmıştır*" şeklindedir. Bu çok kullanışlı görünmüyor, ancak veri setindeki yorum puanlarına dayanarak kendi ortalamamızı hesaplayabiliriz. Aynı oteli örnek olarak kullanırsak, otelin ortalama puanı 7.1 olarak verilmiştir, ancak veri setindeki hesaplanan puan (yorumcu puanlarının ortalaması) 6.8'dir. Bu yakın, ancak aynı değer değil ve yalnızca `Additional_Number_of_Scoring` yorumlarında verilen puanların ortalamayı 7.1'e yükselttiğini varsayabiliriz. Ne yazık ki, bu iddiayı test etmenin veya kanıtlamanın bir yolu olmadığından, `Average_Score`, `Additional_Number_of_Scoring` ve `Total_Number_of_Reviews` değerlerini kullanmak veya güvenmek zordur.

Durumu daha da karmaşık hale getirmek için, ikinci en yüksek yorum sayısına sahip otelin hesaplanan ortalama puanı 8.12 ve veri setindeki `Average_Score` 8.1'dir. Bu doğru puan bir tesadüf mü yoksa ilk otel bir tutarsızlık mı?

Bu otelin bir aykırı değer olabileceği ve belki de çoğu değerin uyumlu olduğu (ancak bazı nedenlerle bazıları uyumsuz) olasılığı üzerine, veri setindeki değerleri keşfetmek ve değerlerin doğru kullanımını (veya kullanılmamasını) belirlemek için bir sonraki adımda kısa bir program yazacağız.
> 🚨 Bir uyarı notu
>
> Bu veri setiyle çalışırken, metni kendiniz okumadan veya analiz etmeden metinden bir şeyler hesaplayan kod yazacaksınız. Bu, NLP'nin özüdür: bir insanın yapmasına gerek kalmadan anlamı veya duyguyu yorumlamak. Ancak, bazı olumsuz yorumları okumanız mümkün olabilir. Bunu yapmamanızı tavsiye ederim, çünkü buna gerek yok. Bazıları saçma veya alakasız olumsuz otel yorumlarıdır, örneğin "Hava güzel değildi" gibi, otelin veya herhangi birinin kontrolü dışında olan bir şey. Ancak bazı yorumların karanlık bir tarafı da vardır. Bazen olumsuz yorumlar ırkçı, cinsiyetçi veya yaş ayrımcı olabilir. Bu talihsiz bir durumdur, ancak halka açık bir web sitesinden alınmış bir veri setinde beklenebilir. Bazı yorumcular, hoş olmayan, rahatsız edici veya üzücü bulabileceğiniz yorumlar bırakabilir. Duyguyu kodun ölçmesine izin vermek, onları kendiniz okuyup üzülmekten daha iyidir. Bununla birlikte, bu tür şeyler yazanlar azınlıktadır, ancak yine de varlar.
## Alıştırma - Veri Keşfi
### Veriyi Yükleme

Veriyi görsel olarak incelemek yeterli, şimdi biraz kod yazıp bazı sorulara cevap bulacaksınız! Bu bölümde pandas kütüphanesi kullanılacak. İlk göreviniz, CSV verisini yükleyip okuyabildiğinizden emin olmak. Pandas kütüphanesi hızlı bir CSV yükleyiciye sahiptir ve sonuç, önceki derslerde olduğu gibi bir dataframe'e yerleştirilir. Yüklediğimiz CSV dosyası yarım milyondan fazla satır içeriyor, ancak sadece 17 sütun var. Pandas, bir dataframe ile etkileşim kurmak için birçok güçlü yöntem sunar, bunlar arasında her satırda işlemler yapma yeteneği de bulunur.

Bu dersten itibaren, kod parçacıkları ve kodun açıklamaları ile sonuçların ne anlama geldiği hakkında bazı tartışmalar yer alacak. Kodunuzu yazmak için _notebook.ipynb_ dosyasını kullanın.

Hadi kullanacağınız veri dosyasını yüklemekle başlayalım:

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

Veri yüklendikten sonra, üzerinde bazı işlemler yapabiliriz. Bu kodu programınızın bir sonraki bölümü için en üstte tutun.

## Veriyi Keşfetme

Bu durumda, veri zaten *temiz* durumda, yani üzerinde çalışmaya hazır ve yalnızca İngilizce karakterler bekleyen algoritmaları zorlayabilecek diğer dillerdeki karakterlere sahip değil.

✅ NLP tekniklerini uygulamadan önce veriyi biçimlendirmek için bazı ilk işlemler yapmanız gerekebilir, ancak bu sefer gerek yok. Eğer gerekseydi, İngilizce olmayan karakterleri nasıl ele alırdınız?

Veri yüklendikten sonra, kodla keşfedebildiğinizden emin olun. `Negative_Review` ve `Positive_Review` sütunlarına odaklanmak çok kolaydır. Bu sütunlar, NLP algoritmalarınızın işlemesi için doğal metinlerle doludur. Ama durun! NLP ve duygu analizine geçmeden önce, aşağıdaki kodu takip ederek veri setindeki değerlerin pandas ile hesapladığınız değerlerle eşleşip eşleşmediğini kontrol etmelisiniz.

## Dataframe İşlemleri

Bu dersteki ilk görev, aşağıdaki varsayımların doğru olup olmadığını kontrol etmek için dataframe'i inceleyen kod yazmaktır (değiştirmeden).

> Birçok programlama görevinde olduğu gibi, bunu tamamlamanın birkaç yolu vardır, ancak iyi bir tavsiye, özellikle gelecekte bu koda geri döndüğünüzde anlaması daha kolay olacaksa, en basit ve en kolay yolu seçmektir. Dataframe'lerle çalışırken, genellikle istediğiniz şeyi verimli bir şekilde yapmanın bir yolunu sunan kapsamlı bir API vardır.

Aşağıdaki soruları kodlama görevleri olarak ele alın ve çözümü görmeden cevaplamaya çalışın.

1. Yüklediğiniz dataframe'in *şeklini* yazdırın (şekil, satır ve sütun sayısıdır).
2. İnceleyenlerin milliyetleri için frekans sayımını hesaplayın:
   1. `Reviewer_Nationality` sütununda kaç farklı değer var ve bunlar neler?
   2. Veri setinde en yaygın inceleyen milliyeti hangisi (ülke ve inceleme sayısını yazdırın)?
   3. En sık bulunan sonraki 10 milliyet ve frekans sayıları nelerdir?
3. En sık bulunan 10 inceleyen milliyeti için en çok incelenen otel hangisiydi?
4. Veri setindeki otel başına kaç inceleme var (otel frekans sayımı)?
5. Veri setindeki her otel için tüm inceleyen puanlarının ortalamasını alarak bir ortalama puan hesaplayabilirsiniz. Dataframe'inize `Calc_Average_Score` başlıklı yeni bir sütun ekleyin ve bu hesaplanan ortalamayı içerir. 
6. Herhangi bir otelin `Average_Score` ve `Calc_Average_Score` değerleri (1 ondalık basamağa yuvarlanmış) aynı mı?
   1. Bir Series (satır) argümanını alan ve değerleri karşılaştıran, eşit olmadığında bir mesaj yazdıran bir Python fonksiyonu yazmayı deneyin. Ardından `.apply()` yöntemini kullanarak her satırı bu fonksiyonla işleyin.
7. `Negative_Review` sütununda "No Negative" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.
8. `Positive_Review` sütununda "No Positive" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.
9. `Positive_Review` sütununda "No Positive" **ve** `Negative_Review` sütununda "No Negative" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.

### Kod Cevapları

1. Yüklediğiniz dataframe'in *şeklini* yazdırın (şekil, satır ve sütun sayısıdır).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. İnceleyenlerin milliyetleri için frekans sayımını hesaplayın:

   1. `Reviewer_Nationality` sütununda kaç farklı değer var ve bunlar neler?
   2. Veri setinde en yaygın inceleyen milliyeti hangisi (ülke ve inceleme sayısını yazdırın)?

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

   3. En sık bulunan sonraki 10 milliyet ve frekans sayıları nelerdir?

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

3. En sık bulunan 10 inceleyen milliyeti için en çok incelenen otel hangisiydi?

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

4. Veri setindeki otel başına kaç inceleme var (otel frekans sayımı)?

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
   
   Veri setinde *sayılan* sonuçların `Total_Number_of_Reviews` değerleriyle eşleşmediğini fark edebilirsiniz. Bu değerin otelin toplam inceleme sayısını temsil ettiği, ancak hepsinin kazınmadığı veya başka bir hesaplama yapıldığı açık değildir. Bu belirsizlik nedeniyle `Total_Number_of_Reviews` modelde kullanılmaz.

5. Veri setindeki her otel için tüm inceleyen puanlarının ortalamasını alarak bir ortalama puan hesaplayabilirsiniz. Dataframe'inize `Calc_Average_Score` başlıklı yeni bir sütun ekleyin ve bu hesaplanan ortalamayı içerir. `Hotel_Name`, `Average_Score` ve `Calc_Average_Score` sütunlarını yazdırın.

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

   `Average_Score` değerinin neden bazen hesaplanan ortalama puandan farklı olduğunu merak edebilirsiniz. Bazı değerlerin eşleştiğini, ancak diğerlerinde bir fark olduğunu bilmediğimiz için, bu durumda en güvenli yol, sahip olduğumuz inceleme puanlarını kullanarak ortalamayı kendimiz hesaplamaktır. Bununla birlikte, farklar genellikle çok küçüktür, işte veri seti ortalaması ile hesaplanan ortalama arasındaki en büyük sapmaya sahip oteller:

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

   Sadece 1 otelin puan farkı 1'den büyük olduğundan, farkı görmezden gelip hesaplanan ortalama puanı kullanabiliriz.

6. `Negative_Review` sütununda "No Negative" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.

7. `Positive_Review` sütununda "No Positive" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.

8. `Positive_Review` sütununda "No Positive" **ve** `Negative_Review` sütununda "No Negative" değerine sahip kaç satır olduğunu hesaplayın ve yazdırın.

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

## Başka Bir Yol

Lambda kullanmadan öğeleri saymanın başka bir yolu ve satırları saymak için toplamı kullanmak:

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

   `Negative_Review` ve `Positive_Review` sütunlarında sırasıyla "No Negative" ve "No Positive" değerine sahip 127 satır olduğunu fark etmiş olabilirsiniz. Bu, inceleyen kişinin otele bir sayısal puan verdiği, ancak ne olumlu ne de olumsuz bir inceleme yazmayı reddettiği anlamına gelir. Neyse ki bu, küçük bir satır miktarıdır (127 satırdan 515738'e, yani %0.02), bu nedenle modelimizi veya sonuçlarımızı belirli bir yönde çarpıtmayacaktır, ancak bir inceleme veri setinde hiç inceleme olmayan satırların olmasını beklememiş olabilirsiniz, bu nedenle bu tür satırları keşfetmek için veriyi incelemek önemlidir.

Artık veri setini keşfettiğinize göre, bir sonraki derste veriyi filtreleyecek ve bazı duygu analizleri ekleyeceksiniz.

---
## 🚀Meydan Okuma

Bu ders, önceki derslerde gördüğümüz gibi, verinizi ve tuhaflıklarını anlamanın ne kadar kritik olduğunu gösteriyor. Özellikle metin tabanlı veriler dikkatli bir inceleme gerektirir. Çeşitli metin ağırlıklı veri setlerini inceleyin ve bir modele önyargı veya çarpık duygu ekleyebilecek alanları keşfedip edemeyeceğinizi görün.

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## İnceleme ve Kendi Kendine Çalışma

[Bu NLP Öğrenme Yolunu](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) alarak konuşma ve metin ağırlıklı modeller oluştururken denenecek araçları keşfedin.

## Ödev 

[NLTK](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.