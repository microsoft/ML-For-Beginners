<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-06T07:50:57+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "tr"
}
-->
# Kümeleme Giriş

Kümeleme, bir veri kümesinin etiketlenmediğini veya girdilerin önceden tanımlanmış çıktılarla eşleştirilmediğini varsayan bir tür [Denetimsiz Öğrenme](https://wikipedia.org/wiki/Unsupervised_learning) yöntemidir. Bu yöntem, etiketlenmemiş verileri çeşitli algoritmalarla analiz ederek, verideki desenlere göre gruplamalar sağlar.

[![PSquare'dan No One Like You](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "PSquare'dan No One Like You")

> 🎥 Yukarıdaki görsele tıklayarak bir video izleyebilirsiniz. Kümeleme ile makine öğrenimi çalışırken, 2014'te PSquare tarafından yayımlanan bu yüksek puanlı Nijerya Dance Hall şarkısının keyfini çıkarın.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

### Giriş

[Kümeleme](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124), veri keşfi için oldukça faydalıdır. Nijeryalı dinleyicilerin müzik tüketim alışkanlıklarındaki eğilimleri ve desenleri keşfetmeye yardımcı olup olamayacağını görelim.

✅ Kümelemenin kullanım alanlarını düşünmek için bir dakikanızı ayırın. Gerçek hayatta, kümeleme çamaşır yığınını ayırıp aile üyelerinizin kıyafetlerini düzenlemeniz gerektiğinde gerçekleşir 🧦👕👖🩲. Veri biliminde ise, bir kullanıcının tercihlerini analiz etmeye veya etiketlenmemiş herhangi bir veri kümesinin özelliklerini belirlemeye çalışırken kümeleme yapılır. Kümeleme, bir anlamda, bir çorap çekmecesi gibi kaosu anlamlandırmaya yardımcı olur.

[![ML'ye Giriş](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Kümelemeye Giriş")

> 🎥 Yukarıdaki görsele tıklayarak bir video izleyebilirsiniz: MIT'den John Guttag kümelemeyi tanıtıyor.

Profesyonel bir ortamda, kümeleme pazar segmentasyonu belirlemek, örneğin hangi yaş gruplarının hangi ürünleri satın aldığını anlamak için kullanılabilir. Bir başka kullanım alanı ise, kredi kartı işlemleri veri kümesinden dolandırıcılığı tespit etmek gibi anomali tespitidir. Ya da bir grup tıbbi taramada tümörleri belirlemek için kümeleme kullanılabilir.

✅ Bankacılık, e-ticaret veya iş dünyasında 'doğada' kümelemeyle nasıl karşılaşmış olabileceğinizi bir dakika düşünün.

> 🎓 İlginç bir şekilde, kümeleme analizi 1930'larda Antropoloji ve Psikoloji alanlarında ortaya çıkmıştır. Sizce o zamanlar nasıl kullanılmış olabilir?

Alternatif olarak, arama sonuçlarını gruplamak için kullanılabilir - örneğin alışveriş bağlantıları, görseller veya incelemeler şeklinde. Kümeleme, büyük bir veri kümesini azaltmak ve daha ayrıntılı analiz yapmak istediğinizde faydalıdır, bu nedenle diğer modeller oluşturulmadan önce veri hakkında bilgi edinmek için kullanılabilir.

✅ Verileriniz kümelere organize edildikten sonra, onlara bir küme kimliği atarsınız ve bu teknik, bir veri kümesinin gizliliğini korumak için faydalı olabilir; bir veri noktasına daha açıklayıcı tanımlayıcı veriler yerine küme kimliğiyle atıfta bulunabilirsiniz. Küme kimliğine diğer küme unsurlarından ziyade neden başvurabileceğinize dair başka nedenler düşünebilir misiniz?

Kümeleme teknikleri hakkındaki bilginizi bu [Öğrenme Modülü](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott) ile derinleştirin.

## Kümelemeye Başlangıç

[Scikit-learn geniş bir yöntem yelpazesi sunar](https://scikit-learn.org/stable/modules/clustering.html) ve seçiminiz kullanım durumunuza bağlıdır. Belgelerine göre, her yöntemin çeşitli avantajları vardır. İşte Scikit-learn tarafından desteklenen yöntemlerin ve uygun kullanım durumlarının basitleştirilmiş bir tablosu:

| Yöntem Adı                  | Kullanım Durumu                                                        |
| :-------------------------- | :--------------------------------------------------------------------- |
| K-Means                     | genel amaçlı, tümevarımsal                                             |
| Affinity propagation        | çok sayıda, düzensiz kümeler, tümevarımsal                             |
| Mean-shift                  | çok sayıda, düzensiz kümeler, tümevarımsal                             |
| Spectral clustering         | az sayıda, düzenli kümeler, tümdengelimsel                            |
| Ward hierarchical clustering| çok sayıda, kısıtlı kümeler, tümdengelimsel                           |
| Agglomerative clustering    | çok sayıda, kısıtlı, Öklid dışı mesafeler, tümdengelimsel             |
| DBSCAN                      | düz olmayan geometri, düzensiz kümeler, tümdengelimsel                |
| OPTICS                      | düz olmayan geometri, değişken yoğunluklu düzensiz kümeler, tümdengelimsel |
| Gaussian mixtures           | düz geometri, tümevarımsal                                             |
| BIRCH                       | büyük veri kümesi, aykırı değerler, tümevarımsal                       |

> 🎓 Kümeleri nasıl oluşturduğumuz, veri noktalarını gruplara nasıl topladığımızla yakından ilgilidir. Şimdi bazı terimleri açıklayalım:
>
> 🎓 ['Tümdengelimsel' vs. 'Tümevarımsal'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Tümdengelimsel çıkarım, belirli test durumlarına eşlenen gözlemlenmiş eğitim durumlarından türetilir. Tümevarımsal çıkarım ise genel kurallara eşlenen eğitim durumlarından türetilir ve bu kurallar daha sonra test durumlarına uygulanır.
> 
> Örnek: Kısmen etiketlenmiş bir veri kümeniz olduğunu hayal edin. Bazı şeyler 'plak', bazıları 'cd' ve bazıları boş. Göreviniz, boşlara etiket vermektir. Eğer tümevarımsal bir yaklaşım seçerseniz, 'plak' ve 'cd' arayan bir model eğitirsiniz ve bu etiketleri etiketlenmemiş verilere uygularsınız. Bu yaklaşım, aslında 'kaset' olan şeyleri sınıflandırmada zorluk çeker. Tümdengelimsel bir yaklaşım ise, bu bilinmeyen verileri daha etkili bir şekilde ele alır çünkü benzer öğeleri gruplamaya çalışır ve ardından bir gruba etiket uygular. Bu durumda, kümeler 'yuvarlak müzik şeyleri' ve 'kare müzik şeyleri' gibi görünebilir.
> 
> 🎓 ['Düz olmayan' vs. 'Düz' geometri](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Matematiksel terminolojiden türetilen düz olmayan vs. düz geometri, noktalar arasındaki mesafelerin 'düz' ([Öklid](https://wikipedia.org/wiki/Euclidean_geometry)) veya 'düz olmayan' (Öklid dışı) geometrik yöntemlerle ölçülmesini ifade eder.
>
>'Düz' bu bağlamda Öklid geometrisini ifade eder (bir kısmı 'düzlem' geometrisi olarak öğretilir) ve düz olmayan, Öklid dışı geometriyi ifade eder. Geometri ile makine öğreniminin ne ilgisi var? İki alan da matematiğe dayandığından, kümelerdeki noktalar arasındaki mesafeleri ölçmek için ortak bir yol olmalıdır ve bu, verinin doğasına bağlı olarak 'düz' veya 'düz olmayan' bir şekilde yapılabilir. [Öklid mesafeleri](https://wikipedia.org/wiki/Euclidean_distance), iki nokta arasındaki bir doğru parçasının uzunluğu olarak ölçülür. [Öklid dışı mesafeler](https://wikipedia.org/wiki/Non-Euclidean_geometry) ise bir eğri boyunca ölçülür. Verileriniz görselleştirildiğinde bir düzlemde değilmiş gibi görünüyorsa, bunu ele almak için özel bir algoritma kullanmanız gerekebilir.
>
![Düz vs Düz Olmayan Geometri Bilgilendirme Görseli](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Bilgilendirme görseli: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Mesafeler'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Kümeler, örneğin noktalar arasındaki mesafeler olan mesafe matrisiyle tanımlanır. Bu mesafe birkaç şekilde ölçülebilir. Öklid kümeleri, nokta değerlerinin ortalamasıyla tanımlanır ve bir 'merkez' veya merkez noktası içerir. Mesafeler, bu merkeze olan mesafeyle ölçülür. Öklid dışı mesafeler ise 'clustroid' olarak adlandırılan, diğer noktalara en yakın olan nokta ile tanımlanır. Clustroid'ler çeşitli şekillerde tanımlanabilir.
> 
> 🎓 ['Kısıtlı'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Kısıtlı Kümeleme](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf), bu denetimsiz yönteme 'yarı denetimli' öğrenme ekler. Noktalar arasındaki ilişkiler 'bağlanamaz' veya 'bağlanmalı' olarak işaretlenir, böylece veri kümesine bazı kurallar zorlanır.
>
>Örnek: Bir algoritma, etiketlenmemiş veya kısmen etiketlenmiş bir veri kümesi üzerinde serbest bırakılırsa, ürettiği kümeler düşük kaliteli olabilir. Yukarıdaki örnekte, kümeler 'yuvarlak müzik şeyleri', 'kare müzik şeyleri', 'üçgen şeyler' ve 'kurabiyeler' olarak gruplandırılabilir. Eğer algoritmaya bazı kısıtlamalar veya kurallar verilirse ("öğe plastikten yapılmış olmalı", "öğe müzik üretebilmeli"), bu algoritmanın daha iyi seçimler yapmasına yardımcı olabilir.
> 
> 🎓 'Yoğunluk'
> 
> 'Gürültülü' olarak kabul edilen veriler 'yoğun' olarak değerlendirilir. Kümelerindeki noktalar arasındaki mesafeler, inceleme sırasında daha yoğun veya daha az yoğun olabilir ve bu nedenle bu veriler uygun kümeleme yöntemiyle analiz edilmelidir. [Bu makale](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html), düzensiz küme yoğunluğuna sahip gürültülü bir veri kümesini keşfetmek için K-Means kümeleme ve HDBSCAN algoritmalarını kullanmanın farkını göstermektedir.

## Kümeleme Algoritmaları

100'den fazla kümeleme algoritması vardır ve kullanımları eldeki verinin doğasına bağlıdır. İşte başlıca olanlardan bazıları:

- **Hiyerarşik kümeleme**. Bir nesne, daha uzak bir nesne yerine yakınındaki bir nesneye göre sınıflandırıldığında, kümeler üyelerinin diğer nesnelere olan mesafelerine göre oluşturulur. Scikit-learn'ün agglomerative clustering yöntemi hiyerarşiktir.

   ![Hiyerarşik Kümeleme Bilgilendirme Görseli](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Bilgilendirme görseli: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Merkez kümeleme**. Bu popüler algoritma, oluşturulacak 'k' veya küme sayısının seçilmesini gerektirir, ardından algoritma bir kümenin merkez noktasını belirler ve verileri bu noktanın etrafında toplar. [K-means kümeleme](https://wikipedia.org/wiki/K-means_clustering), merkez kümeleme türünün popüler bir versiyonudur. Merkez, en yakın ortalama ile belirlenir, bu nedenle adı buradan gelir. Kümeden olan kare mesafesi minimize edilir.

   ![Merkez Kümeleme Bilgilendirme Görseli](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Bilgilendirme görseli: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Dağılım tabanlı kümeleme**. İstatistiksel modellemeye dayalı olan dağılım tabanlı kümeleme, bir veri noktasının bir kümeye ait olma olasılığını belirlemeye ve buna göre atamaya odaklanır. Gaussian karışım yöntemleri bu türe aittir.

- **Yoğunluk tabanlı kümeleme**. Veri noktaları, yoğunluklarına veya birbirleri etrafındaki gruplanmalarına göre kümelere atanır. Gruptan uzak olan veri noktaları aykırı değerler veya gürültü olarak kabul edilir. DBSCAN, Mean-shift ve OPTICS bu tür kümelemeye aittir.

- **Izgara tabanlı kümeleme**. Çok boyutlu veri kümeleri için bir ızgara oluşturulur ve veri, ızgaranın hücreleri arasında bölünerek kümeler oluşturulur.

## Egzersiz - Verilerinizi Kümeleyin

Kümeleme, doğru görselleştirme ile büyük ölçüde desteklenir, bu yüzden müzik verilerimizi görselleştirerek başlayalım. Bu egzersiz, bu verinin doğasına en uygun kümeleme yöntemlerini belirlememize yardımcı olacaktır.

1. Bu klasördeki [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) dosyasını açın.

1. İyi bir veri görselleştirme için `Seaborn` paketini içe aktarın.

    ```python
    !pip install seaborn
    ```

1. [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) dosyasından şarkı verilerini ekleyin. Şarkılar hakkında bazı verilerle bir dataframe yükleyin. Kütüphaneleri içe aktararak ve verileri dökerek bu veriyi keşfetmeye hazırlanın:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Verinin ilk birkaç satırını kontrol edin:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Dataframe hakkında bilgi almak için `info()` çağırın:

    ```python
    df.info()
    ```

   Çıktı şu şekilde görünecek:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Null değerleri kontrol etmek için `isnull()` çağırın ve toplamın 0 olduğunu doğrulayın:

    ```python
    df.isnull().sum()
    ```

    Her şey yolunda görünüyor:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Veriyi tanımlayın:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Eğer kümeleme ile çalışıyorsak, etiketli verilere ihtiyaç duymayan bir denetimsiz yöntem, neden bu veriyi etiketlerle gösteriyoruz? Veri keşfi aşamasında faydalı olabilirler, ancak kümeleme algoritmalarının çalışması için gerekli değiller. Sütun başlıklarını kaldırabilir ve veriye sütun numaralarıyla da başvurabilirsiniz.

Verinin genel değerlerine bakın. Popülerlik '0' olabilir, bu da sıralaması olmayan şarkıları gösterir. Bunları kısa süre içinde çıkaralım.

1. En popüler türleri bulmak için bir barplot kullanın:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Daha fazla üst değer görmek isterseniz, üst `[:5]` değerini daha büyük bir değere değiştirin veya tümünü görmek için kaldırın.

Not: En üst tür 'Missing' olarak tanımlandığında, bu Spotify'ın onu sınıflandırmadığı anlamına gelir, bu yüzden bunu çıkaralım.

1. Eksik verileri filtreleyerek çıkarın:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Şimdi türleri tekrar kontrol edin:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Açık ara, en üst üç tür bu veri setine hakim. `afro dancehall`, `afropop` ve `nigerian pop` üzerine yoğunlaşalım, ayrıca veri setini 0 popülerlik değerine sahip olanları (veri setinde popülerlik ile sınıflandırılmamış ve bizim amaçlarımız için gürültü olarak kabul edilebilir) çıkarmak için filtreleyelim:

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Verinin herhangi bir şekilde güçlü bir korelasyon gösterip göstermediğini hızlıca test edin:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Tek güçlü korelasyon `energy` ve `loudness` arasında, ki bu çok şaşırtıcı değil, çünkü yüksek sesli müzik genellikle oldukça enerjik olur. Bunun dışında korelasyonlar nispeten zayıf. Bu veriden bir kümeleme algoritmasının ne çıkarabileceğini görmek ilginç olacak.

    > 🎓 Korelasyonun nedensellik anlamına gelmediğini unutmayın! Korelasyonun kanıtı var, ancak nedenselliğin kanıtı yok. [Eğlenceli bir web sitesi](https://tylervigen.com/spurious-correlations) bu noktayı vurgulayan görseller sunuyor.

Bu veri setinde bir şarkının algılanan popülerliği ve dans edilebilirliği etrafında bir yakınsama var mı? Bir FacetGrid, türden bağımsız olarak hizalanan eşmerkezli daireler olduğunu gösteriyor. Nijeryalı zevklerin bu tür için belirli bir dans edilebilirlik seviyesinde birleştiği olabilir mi?

✅ Farklı veri noktalarını (energy, loudness, speechiness) ve daha fazla veya farklı müzik türlerini deneyin. Neler keşfedebilirsiniz? Verilerin genel dağılımını görmek için `df.describe()` tablosuna göz atın.

### Egzersiz - veri dağılımı

Bu üç tür, popülerliklerine göre dans edilebilirlik algısında önemli ölçüde farklı mı?

1. En üst üç türümüzün popülerlik ve dans edilebilirlik için veri dağılımını belirli bir x ve y ekseni boyunca inceleyin.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Genel bir yakınsama noktası etrafında eşmerkezli daireler keşfedebilirsiniz, bu da noktaların dağılımını gösterir.

    > 🎓 Bu örnek, birden fazla dağılımla çalışırken veriyi yorumlamamıza olanak tanıyan sürekli bir olasılık yoğunluğu eğrisi kullanan bir KDE (Kernel Density Estimate) grafiği kullanır.

    Genel olarak, üç tür popülerlik ve dans edilebilirlik açısından gevşek bir şekilde hizalanır. Bu gevşek hizalanmış veride kümeleri belirlemek bir zorluk olacaktır:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Bir scatter plot oluşturun:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Aynı eksenlerin scatterplot'u benzer bir yakınsama modeli gösterir.

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Genel olarak, kümeleme için scatterplot'ları kullanarak veri kümelerini gösterebilirsiniz, bu nedenle bu tür görselleştirmeyi öğrenmek çok faydalıdır. Bir sonraki derste, bu filtrelenmiş veriyi alıp k-means kümeleme kullanarak ilginç şekillerde örtüşen grupları keşfedeceğiz.

---

## 🚀Meydan Okuma

Bir sonraki ders için hazırlık olarak, üretim ortamında keşfedebileceğiniz ve kullanabileceğiniz çeşitli kümeleme algoritmaları hakkında bir grafik oluşturun. Kümeleme hangi tür problemleri çözmeye çalışıyor?

## [Ders sonrası quiz](https://ff-quizzes.netlify.app/en/ml/)

## İnceleme ve Kendi Kendine Çalışma

Kümeleme algoritmalarını uygulamadan önce, öğrendiğimiz gibi, veri setinizin doğasını anlamak iyi bir fikirdir. Bu konu hakkında daha fazla bilgi edinin [burada](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Bu faydalı makale](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) farklı veri şekilleri göz önüne alındığında çeşitli kümeleme algoritmalarının nasıl davrandığını anlatıyor.

## Ödev

[Kümeleme için diğer görselleştirmeleri araştırın](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluğu sağlamak için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.