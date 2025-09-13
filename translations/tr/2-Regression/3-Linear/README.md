<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "40e64f004f3cb50aa1d8661672d3cd92",
  "translation_date": "2025-09-06T07:44:07+00:00",
  "source_file": "2-Regression/3-Linear/README.md",
  "language_code": "tr"
}
-->
# Scikit-learn ile regresyon modeli oluşturma: dört farklı regresyon yöntemi

![Doğrusal ve polinomial regresyon infografiği](../../../../2-Regression/3-Linear/images/linear-polynomial.png)
> İnfografik: [Dasani Madipalli](https://twitter.com/dasani_decoded)
## [Ders öncesi sınav](https://ff-quizzes.netlify.app/en/ml/)

> ### [Bu ders R dilinde de mevcut!](../../../../2-Regression/3-Linear/solution/R/lesson_3.html)
### Giriş 

Şimdiye kadar, bu derste kullanacağımız kabak fiyatlandırma veri setinden toplanan örnek verilerle regresyonun ne olduğunu keşfettiniz. Ayrıca bunu Matplotlib kullanarak görselleştirdiniz.

Artık ML için regresyonu daha derinlemesine incelemeye hazırsınız. Görselleştirme, verileri anlamlandırmanıza olanak tanırken, Makine Öğrenimi'nin gerçek gücü _modelleri eğitmekten_ gelir. Modeller, veri bağımlılıklarını otomatik olarak yakalamak için geçmiş verilere dayanarak eğitilir ve modelin daha önce görmediği yeni veriler için sonuçları tahmin etmenize olanak tanır.

Bu derste, regresyonun iki türü hakkında daha fazla bilgi edineceksiniz: _temel doğrusal regresyon_ ve _polinomial regresyon_, bu tekniklerin altında yatan bazı matematiksel kavramlarla birlikte. Bu modeller, farklı giriş verilerine bağlı olarak kabak fiyatlarını tahmin etmemize olanak tanıyacak.

[![Başlangıç seviyesinde ML - Doğrusal Regresyonu Anlamak](https://img.youtube.com/vi/CRxFT8oTDMg/0.jpg)](https://youtu.be/CRxFT8oTDMg "Başlangıç seviyesinde ML - Doğrusal Regresyonu Anlamak")

> 🎥 Doğrusal regresyonun kısa bir video özeti için yukarıdaki görsele tıklayın.

> Bu müfredat boyunca, matematik bilgisi minimum düzeyde varsayılmaktadır ve diğer alanlardan gelen öğrenciler için erişilebilir hale getirmeyi amaçlıyoruz. Bu nedenle, notlar, 🧮 matematiksel açıklamalar, diyagramlar ve diğer öğrenme araçlarına dikkat edin.

### Ön Koşul

Şimdiye kadar, incelediğimiz kabak verilerinin yapısına aşina olmalısınız. Bu dersin _notebook.ipynb_ dosyasında önceden yüklenmiş ve temizlenmiş olarak bulabilirsiniz. Dosyada, kabak fiyatı yeni bir veri çerçevesinde bushel başına gösterilmektedir. Bu not defterlerini Visual Studio Code'daki çekirdeklerde çalıştırabildiğinizden emin olun.

### Hazırlık

Bu verileri yüklediğinizi hatırlatmak isteriz, böylece sorular sorabilirsiniz.

- Kabak almak için en iyi zaman ne zaman?
- Mini kabakların bir kutusunun fiyatı ne kadar olabilir?
- Kabakları yarım bushel sepetlerde mi yoksa 1 1/9 bushel kutularında mı almalıyım?
Bu verileri daha fazla incelemeye devam edelim.

Önceki derste, bir Pandas veri çerçevesi oluşturdunuz ve orijinal veri setinin bir kısmını bushel başına fiyatlandırmayı standartlaştırarak doldurdunuz. Ancak bunu yaparak, yalnızca sonbahar ayları için yaklaşık 400 veri noktası toplayabildiniz.

Bu dersin eşlik eden not defterinde önceden yüklenmiş verilere bir göz atın. Veriler önceden yüklenmiş ve ay verilerini göstermek için ilk bir saçılım grafiği çizilmiştir. Belki verileri daha fazla temizleyerek verilerin doğası hakkında biraz daha ayrıntı elde edebiliriz.

## Doğrusal regresyon çizgisi

1. Derste öğrendiğiniz gibi, doğrusal regresyon çalışmasının amacı bir çizgi çizmek ve:

- **Değişken ilişkilerini göstermek**. Değişkenler arasındaki ilişkiyi göstermek
- **Tahminler yapmak**. Yeni bir veri noktasının bu çizgiyle ilişkili olarak nerede yer alacağını doğru bir şekilde tahmin etmek.

Bu tür bir çizgi çizmek için **En Küçük Kareler Regresyonu** kullanılması yaygındır. 'En küçük kareler' terimi, regresyon çizgisinin etrafındaki tüm veri noktalarının karelerinin alınması ve ardından toplanması anlamına gelir. İdeal olarak, bu son toplamın mümkün olduğunca küçük olması gerekir, çünkü düşük hata sayısı veya `en küçük kareler` istiyoruz.

Bunu yapmamızın nedeni, tüm veri noktalarımızdan en az toplam mesafeye sahip bir çizgi modellemek istememizdir. Ayrıca terimleri toplamadan önce karelerini alırız çünkü yönünden ziyade büyüklüğüyle ilgileniyoruz.

> **🧮 Matematiği göster**
> 
> Bu çizgi, _en iyi uyum çizgisi_ olarak adlandırılır ve [bir denklemle](https://en.wikipedia.org/wiki/Simple_linear_regression) ifade edilebilir: 
> 
> ```
> Y = a + bX
> ```
>
> `X` 'açıklayıcı değişken'dir. `Y` 'bağımlı değişken'dir. Çizginin eğimi `b` ve `a` y-kesişimidir, bu da `X = 0` olduğunda `Y` değerine karşılık gelir.
>
>![eğimi hesapla](../../../../2-Regression/3-Linear/images/slope.png)
>
> İlk olarak, eğim `b` hesaplanır. İnfografik: [Jen Looper](https://twitter.com/jenlooper)
>
> Başka bir deyişle, kabak verilerinin orijinal sorusuna atıfta bulunarak: "aylara göre bushel başına kabak fiyatını tahmin et", `X` fiyatı ifade ederken `Y` satış ayını ifade eder.
>
>![denklemi tamamla](../../../../2-Regression/3-Linear/images/calculation.png)
>
> `Y` değerini hesaplayın. Eğer yaklaşık 4 dolar ödüyorsanız, bu Nisan olmalı! İnfografik: [Jen Looper](https://twitter.com/jenlooper)
>
> Çizgiyi hesaplayan matematik, çizginin eğimini göstermelidir, bu da aynı zamanda y-kesişimine bağlıdır, yani `X = 0` olduğunda `Y`'nin konumlandığı yer.
>
> Bu değerlerin hesaplama yöntemini [Math is Fun](https://www.mathsisfun.com/data/least-squares-regression.html) web sitesinde gözlemleyebilirsiniz. Ayrıca, sayıların değerlerinin çizgiyi nasıl etkilediğini görmek için [bu En Küçük Kareler hesaplayıcısını](https://www.mathsisfun.com/data/least-squares-calculator.html) ziyaret edin.

## Korelasyon

Anlamanız gereken bir diğer terim, verilen X ve Y değişkenleri arasındaki **Korelasyon Katsayısı**dır. Bir saçılım grafiği kullanarak bu katsayıyı hızlıca görselleştirebilirsiniz. Veri noktalarının düzgün bir çizgide dağıldığı bir grafik yüksek korelasyona sahiptir, ancak X ve Y arasında her yerde dağılmış veri noktalarına sahip bir grafik düşük korelasyona sahiptir.

İyi bir doğrusal regresyon modeli, En Küçük Kareler Regresyonu yöntemiyle bir regresyon çizgisi kullanarak 1'e yakın (0'dan uzak) bir Korelasyon Katsayısına sahip olan modeldir.

✅ Bu dersin eşlik eden not defterini çalıştırın ve Ay ile Fiyat arasındaki saçılım grafiğine bakın. Kabak satışları için Ay ile Fiyat arasındaki veri, saçılım grafiğine göre görsel yorumunuza göre yüksek veya düşük korelasyona sahip gibi görünüyor mu? Bu durum, `Ay` yerine daha ince bir ölçüm kullanırsanız, örneğin *yılın günü* (yılın başlangıcından itibaren geçen gün sayısı) değişir mi?

Aşağıdaki kodda, verileri temizlediğimizi ve aşağıdaki gibi bir veri çerçevesi elde ettiğimizi varsayacağız:

ID | Ay | YılınGünü | Çeşit | Şehir | Paket | Düşük Fiyat | Yüksek Fiyat | Fiyat
---|----|-----------|-------|-------|-------|-------------|--------------|------
70 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364
71 | 9 | 267 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
72 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 18.0 | 18.0 | 16.363636
73 | 10 | 274 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 17.0 | 17.0 | 15.454545
74 | 10 | 281 | PIE TYPE | BALTIMORE | 1 1/9 bushel cartons | 15.0 | 15.0 | 13.636364

> Verileri temizleme kodu [`notebook.ipynb`](../../../../2-Regression/3-Linear/notebook.ipynb) dosyasında mevcuttur. Önceki derste yapılan aynı temizleme adımlarını uyguladık ve aşağıdaki ifadeyi kullanarak `YılınGünü` sütununu hesapladık:

```python
day_of_year = pd.to_datetime(pumpkins['Date']).apply(lambda dt: (dt-datetime(dt.year,1,1)).days)
```

Artık doğrusal regresyonun arkasındaki matematiği anladığınıza göre, bir Regresyon modeli oluşturarak hangi kabak paketinin en iyi kabak fiyatlarına sahip olacağını tahmin edip edemeyeceğimizi görelim. Bir tatil kabak bahçesi için kabak satın alan biri, bahçe için kabak paketlerini optimize etmek amacıyla bu bilgiye ihtiyaç duyabilir.

## Korelasyon Arayışı

[![Başlangıç seviyesinde ML - Korelasyon Arayışı: Doğrusal Regresyonun Anahtarı](https://img.youtube.com/vi/uoRq-lW2eQo/0.jpg)](https://youtu.be/uoRq-lW2eQo "Başlangıç seviyesinde ML - Korelasyon Arayışı: Doğrusal Regresyonun Anahtarı")

> 🎥 Korelasyonun kısa bir video özeti için yukarıdaki görsele tıklayın.

Önceki dersten muhtemelen farklı aylar için ortalama fiyatın şu şekilde göründüğünü gördünüz:

<img alt="Ay bazında ortalama fiyat" src="../2-Data/images/barchart.png" width="50%"/>

Bu, bir korelasyon olması gerektiğini ve `Ay` ile `Fiyat` veya `YılınGünü` ile `Fiyat` arasındaki ilişkiyi tahmin etmek için doğrusal regresyon modeli eğitmeye çalışabileceğimizi gösteriyor. İşte ikinci ilişkiyi gösteren saçılım grafiği:

<img alt="Fiyat vs. Yılın Günü saçılım grafiği" src="images/scatter-dayofyear.png" width="50%" /> 

`corr` fonksiyonunu kullanarak bir korelasyon olup olmadığını görelim:

```python
print(new_pumpkins['Month'].corr(new_pumpkins['Price']))
print(new_pumpkins['DayOfYear'].corr(new_pumpkins['Price']))
```

Görünüşe göre korelasyon oldukça küçük, `Ay` için -0.15 ve `YılınGünü` için -0.17, ancak başka önemli bir ilişki olabilir. Farklı kabak çeşitlerine karşılık gelen farklı fiyat kümeleri var gibi görünüyor. Bu hipotezi doğrulamak için, her kabak kategorisini farklı bir renkle çizelim. `scatter` çizim fonksiyonuna bir `ax` parametresi geçirerek tüm noktaları aynı grafikte çizebiliriz:

```python
ax=None
colors = ['red','blue','green','yellow']
for i,var in enumerate(new_pumpkins['Variety'].unique()):
    df = new_pumpkins[new_pumpkins['Variety']==var]
    ax = df.plot.scatter('DayOfYear','Price',ax=ax,c=colors[i],label=var)
```

<img alt="Fiyat vs. Yılın Günü saçılım grafiği" src="images/scatter-dayofyear-color.png" width="50%" /> 

Araştırmamız, çeşidin genel fiyat üzerinde satış tarihinden daha fazla etkisi olduğunu öne sürüyor. Bunu bir çubuk grafikle görebiliriz:

```python
new_pumpkins.groupby('Variety')['Price'].mean().plot(kind='bar')
```

<img alt="Çeşide göre fiyat çubuk grafiği" src="images/price-by-variety.png" width="50%" /> 

Şimdi bir süreliğine yalnızca bir kabak çeşidine, 'pie type' çeşidine odaklanalım ve tarihin fiyat üzerindeki etkisini görelim:

```python
pie_pumpkins = new_pumpkins[new_pumpkins['Variety']=='PIE TYPE']
pie_pumpkins.plot.scatter('DayOfYear','Price') 
```
<img alt="Fiyat vs. Yılın Günü saçılım grafiği" src="images/pie-pumpkins-scatter.png" width="50%" /> 

Şimdi `corr` fonksiyonunu kullanarak `Fiyat` ile `YılınGünü` arasındaki korelasyonu hesaplasak, yaklaşık `-0.27` gibi bir değer elde ederiz - bu da tahmin edici bir model eğitmenin mantıklı olduğunu gösterir.

> Doğrusal regresyon modeli eğitmeden önce, verilerimizin temiz olduğundan emin olmak önemlidir. Doğrusal regresyon eksik değerlerle iyi çalışmaz, bu nedenle tüm boş hücrelerden kurtulmak mantıklıdır:

```python
pie_pumpkins.dropna(inplace=True)
pie_pumpkins.info()
```

Bir diğer yaklaşım, bu boş değerleri ilgili sütunun ortalama değerleriyle doldurmak olabilir.

## Basit Doğrusal Regresyon

[![Başlangıç seviyesinde ML - Scikit-learn ile Doğrusal ve Polinomial Regresyon](https://img.youtube.com/vi/e4c_UP2fSjg/0.jpg)](https://youtu.be/e4c_UP2fSjg "Başlangıç seviyesinde ML - Scikit-learn ile Doğrusal ve Polinomial Regresyon")

> 🎥 Doğrusal ve polinomial regresyonun kısa bir video özeti için yukarıdaki görsele tıklayın.

Doğrusal Regresyon modelimizi eğitmek için **Scikit-learn** kütüphanesini kullanacağız.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
```

Başlangıçta, giriş değerlerini (özellikler) ve beklenen çıktıyı (etiket) ayrı numpy dizilerine ayırıyoruz:

```python
X = pie_pumpkins['DayOfYear'].to_numpy().reshape(-1,1)
y = pie_pumpkins['Price']
```

> Giriş verilerinde `reshape` işlemi yapmamız gerektiğini unutmayın, böylece Doğrusal Regresyon paketi bunu doğru şekilde anlayabilir. Doğrusal Regresyon, her bir satırın giriş özelliklerinin bir vektörüne karşılık geldiği bir 2D-dizi bekler. Bizim durumumuzda, yalnızca bir girişimiz olduğu için, N×1 şekline sahip bir diziye ihtiyacımız var, burada N veri setinin boyutudur.

Daha sonra, verileri eğitim ve test veri setlerine ayırmamız gerekiyor, böylece modelimizi eğittikten sonra doğrulayabiliriz:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

Son olarak, gerçek Doğrusal Regresyon modelini eğitmek yalnızca iki satır kod alır. `LinearRegression` nesnesini tanımlarız ve `fit` yöntemiyle verilerimize uyarlarız:

```python
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
```

`LinearRegression` nesnesi, `fit` işleminden sonra regresyonun tüm katsayılarını içerir ve bunlara `.coef_` özelliği ile erişilebilir. Bizim durumumuzda, yalnızca bir katsayı vardır ve bu yaklaşık `-0.017` olmalıdır. Bu, fiyatların zamanla biraz düştüğünü, ancak çok fazla olmadığını, günde yaklaşık 2 sent olduğunu gösterir. Ayrıca, regresyonun Y ekseniyle kesişim noktasına `lin_reg.intercept_` kullanarak erişebiliriz - bu bizim durumumuzda yaklaşık `21` olacaktır, yılın başındaki fiyatı gösterir.

Modelimizin ne kadar doğru olduğunu görmek için test veri setinde fiyatları tahmin edebilir ve ardından tahminlerimizin beklenen değerlere ne kadar yakın olduğunu ölçebiliriz. Bu, beklenen ve tahmin edilen değerler arasındaki tüm kare farklarının ortalaması olan ortalama kare hata (MSE) metriği kullanılarak yapılabilir.

```python
pred = lin_reg.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')
```
Hatalarımız yaklaşık %17 oranında, yani pek iyi değil. Model kalitesinin bir diğer göstergesi **determinasyon katsayısıdır** ve şu şekilde elde edilebilir:

```python
score = lin_reg.score(X_train,y_train)
print('Model determination: ', score)
```
Eğer değer 0 ise, bu modelin girdi verilerini dikkate almadığı ve *en kötü doğrusal tahmin edici* olarak hareket ettiği anlamına gelir; bu da sonuçların basit bir ortalamasıdır. Değer 1 olduğunda, tüm beklenen çıktıları mükemmel bir şekilde tahmin edebileceğimiz anlamına gelir. Bizim durumumuzda, katsayı yaklaşık 0.06 civarında, bu da oldukça düşük.

Regresyonun bizim durumumuzda nasıl çalıştığını daha iyi görmek için test verilerini regresyon çizgisiyle birlikte görselleştirebiliriz:

```python
plt.scatter(X_test,y_test)
plt.plot(X_test,pred)
```

<img alt="Doğrusal regresyon" src="images/linear-results.png" width="50%" />

## Polinom Regresyon

Doğrusal Regresyon'un bir diğer türü Polinom Regresyon'dur. Bazen değişkenler arasında doğrusal bir ilişki olabilir - örneğin, kabak hacmi büyüdükçe fiyatın artması - ancak bazen bu ilişkiler bir düzlem veya doğru olarak çizilemez.

✅ İşte [Polinom Regresyon](https://online.stat.psu.edu/stat501/lesson/9/9.8) kullanabilecek veri türlerine dair bazı örnekler.

Tarih ve Fiyat arasındaki ilişkiye tekrar bakın. Bu dağılım grafiği mutlaka bir doğru ile analiz edilmeli mi? Fiyatlar dalgalanamaz mı? Bu durumda polinom regresyonu deneyebilirsiniz.

✅ Polinomlar, bir veya daha fazla değişken ve katsayı içerebilen matematiksel ifadelerdir.

Polinom regresyon, doğrusal olmayan veriye daha iyi uyum sağlamak için eğri bir çizgi oluşturur. Bizim durumumuzda, girdi verilerine kare `DayOfYear` değişkenini eklersek, verilerimizi yıl içinde belirli bir noktada minimuma sahip olan parabolik bir eğri ile uyumlu hale getirebiliriz.

Scikit-learn, veri işleme adımlarını bir araya getirmek için kullanışlı bir [pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html?highlight=pipeline#sklearn.pipeline.make_pipeline) içerir. **Pipeline**, bir dizi **tahmin ediciden** oluşur. Bizim durumumuzda, önce modelimize polinom özellikler ekleyen ve ardından regresyonu eğiten bir pipeline oluşturacağız:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())

pipeline.fit(X_train,y_train)
```

`PolynomialFeatures(2)` kullanmak, girdi verilerinden tüm ikinci derece polinomları dahil edeceğimiz anlamına gelir. Bizim durumumuzda bu sadece `DayOfYear`<sup>2</sup> anlamına gelir, ancak iki girdi değişkeni X ve Y verildiğinde, bu X<sup>2</sup>, XY ve Y<sup>2</sup> ekleyecektir. Daha yüksek dereceli polinomlar kullanmak istersek bunu da yapabiliriz.

Pipeline'lar, orijinal `LinearRegression` nesnesi gibi kullanılabilir, yani pipeline'ı `fit` edebilir ve ardından tahmin sonuçlarını almak için `predict` kullanabiliriz. İşte test verilerini ve yaklaşık eğriyi gösteren grafik:

<img alt="Polinom regresyon" src="images/poly-results.png" width="50%" />

Polinom Regresyon kullanarak biraz daha düşük MSE ve daha yüksek determinasyon elde edebiliriz, ancak fark çok büyük değil. Diğer özellikleri de dikkate almamız gerekiyor!

> Kabak fiyatlarının minimum seviyede olduğu zamanın Cadılar Bayramı civarında olduğunu görebilirsiniz. Bunu nasıl açıklarsınız?

🎃 Tebrikler, kabak turtası fiyatını tahmin etmeye yardımcı olabilecek bir model oluşturdunuz. Muhtemelen aynı prosedürü tüm kabak türleri için tekrarlayabilirsiniz, ancak bu oldukça zahmetli olur. Şimdi modelimizde kabak çeşitlerini nasıl dikkate alacağımızı öğrenelim!

## Kategorik Özellikler

İdeal bir dünyada, aynı modeli kullanarak farklı kabak çeşitlerinin fiyatlarını tahmin edebilmek isteriz. Ancak `Variety` sütunu, `Month` gibi sütunlardan biraz farklıdır, çünkü sayısal olmayan değerler içerir. Bu tür sütunlara **kategorik** denir.

[![Başlangıç Seviyesi ML - Doğrusal Regresyon ile Kategorik Özellik Tahminleri](https://img.youtube.com/vi/DYGliioIAE0/0.jpg)](https://youtu.be/DYGliioIAE0 "Başlangıç Seviyesi ML - Doğrusal Regresyon ile Kategorik Özellik Tahminleri")

> 🎥 Yukarıdaki görsele tıklayarak kategorik özelliklerin kullanımına dair kısa bir video izleyebilirsiniz.

Burada ortalama fiyatın çeşitliliğe bağlı olarak nasıl değiştiğini görebilirsiniz:

<img alt="Çeşide göre ortalama fiyat" src="images/price-by-variety.png" width="50%" />

Çeşidi dikkate almak için önce bunu sayısal bir forma dönüştürmemiz, yani **kodlamamız** gerekir. Bunu yapmanın birkaç yolu vardır:

* Basit **sayısal kodlama**, farklı çeşitlerin bir tablosunu oluşturur ve ardından çeşit adını bu tablodaki bir indeksle değiştirir. Bu, doğrusal regresyon için en iyi fikir değildir, çünkü doğrusal regresyon indeksin gerçek sayısal değerini alır ve bunu bir katsayı ile çarparak sonuca ekler. Bizim durumumuzda, indeks numarası ile fiyat arasındaki ilişki açıkça doğrusal değildir, indekslerin belirli bir şekilde sıralandığından emin olsak bile.
* **One-hot kodlama**, `Variety` sütununu dört farklı sütunla değiştirir, her biri bir çeşit için. Her sütun, ilgili satırın belirli bir çeşide ait olup olmadığını göstermek için `1` veya `0` içerir. Bu, doğrusal regresyonda her kabak çeşidi için bir katsayı oluşturur ve bu katsayı o çeşidin "başlangıç fiyatı" (veya "ek fiyatı") için sorumludur.

Aşağıdaki kod, bir çeşidi nasıl one-hot kodlayabileceğimizi gösterir:

```python
pd.get_dummies(new_pumpkins['Variety'])
```

 ID | FAIRYTALE | MINIATURE | MIXED HEIRLOOM VARIETIES | PIE TYPE
----|-----------|-----------|--------------------------|----------
70 | 0 | 0 | 0 | 1
71 | 0 | 0 | 0 | 1
... | ... | ... | ... | ...
1738 | 0 | 1 | 0 | 0
1739 | 0 | 1 | 0 | 0
1740 | 0 | 1 | 0 | 0
1741 | 0 | 1 | 0 | 0
1742 | 0 | 1 | 0 | 0

One-hot kodlanmış çeşidi girdi olarak kullanarak doğrusal regresyonu eğitmek için, sadece `X` ve `y` verilerini doğru şekilde başlatmamız gerekir:

```python
X = pd.get_dummies(new_pumpkins['Variety'])
y = new_pumpkins['Price']
```

Kodun geri kalanı, yukarıda Doğrusal Regresyonu eğitmek için kullandığımız kodla aynıdır. Eğer denerseniz, ortalama kare hata (MSE) yaklaşık aynı kalır, ancak determinasyon katsayısı çok daha yüksek (~%77) olur. Daha doğru tahminler elde etmek için daha fazla kategorik özelliği ve `Month` veya `DayOfYear` gibi sayısal özellikleri dikkate alabiliriz. Daha büyük bir özellik dizisi oluşturmak için `join` kullanabiliriz:

```python
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']
```

Burada ayrıca `City` ve `Package` türünü de dikkate alıyoruz, bu da bize MSE 2.84 (%10) ve determinasyon 0.94 sağlar!

## Hepsini Bir Araya Getirmek

En iyi modeli oluşturmak için yukarıdaki örnekten birleştirilmiş (one-hot kodlanmış kategorik + sayısal) verileri Polinom Regresyon ile birlikte kullanabiliriz. İşte tüm kodun tamamı:

```python
# set up training data
X = pd.get_dummies(new_pumpkins['Variety']) \
        .join(new_pumpkins['Month']) \
        .join(pd.get_dummies(new_pumpkins['City'])) \
        .join(pd.get_dummies(new_pumpkins['Package']))
y = new_pumpkins['Price']

# make train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# setup and train the pipeline
pipeline = make_pipeline(PolynomialFeatures(2), LinearRegression())
pipeline.fit(X_train,y_train)

# predict results for test data
pred = pipeline.predict(X_test)

# calculate MSE and determination
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

score = pipeline.score(X_train,y_train)
print('Model determination: ', score)
```

Bu, bize yaklaşık %97'lik en iyi determinasyon katsayısını ve MSE=2.23 (~%8 tahmin hatası) sağlar.

| Model | MSE | Determinasyon |
|-------|-----|---------------|
| `DayOfYear` Doğrusal | 2.77 (%17.2) | 0.07 |
| `DayOfYear` Polinom | 2.73 (%17.0) | 0.08 |
| `Variety` Doğrusal | 5.24 (%19.7) | 0.77 |
| Tüm özellikler Doğrusal | 2.84 (%10.5) | 0.94 |
| Tüm özellikler Polinom | 2.23 (%8.25) | 0.97 |

🏆 Tebrikler! Bu derste dört farklı Regresyon modeli oluşturdunuz ve model kalitesini %97'ye çıkardınız. Regresyon ile ilgili son bölümde, kategorileri belirlemek için Lojistik Regresyonu öğreneceksiniz.

---
## 🚀Meydan Okuma

Bu not defterinde farklı değişkenleri test ederek korelasyonun model doğruluğuyla nasıl ilişkili olduğunu inceleyin.

## [Ders sonrası test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Bu derste Doğrusal Regresyon hakkında bilgi edindik. Regresyonun diğer önemli türleri de vardır. Stepwise, Ridge, Lasso ve Elasticnet tekniklerini okuyun. Daha fazla bilgi edinmek için iyi bir kurs [Stanford İstatistiksel Öğrenme kursu](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning) olabilir.

## Ödev 

[Bir Model Oluşturun](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul edilmez.