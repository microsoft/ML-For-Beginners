<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-06T07:52:56+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "tr"
}
-->
# Postscript: Gerçek Dünyada Makine Öğrenimi

![Gerçek dünyada makine öğreniminin özetini içeren bir sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Bu müfredatta, verileri eğitim için hazırlamanın ve makine öğrenimi modelleri oluşturmanın birçok yolunu öğrendiniz. Klasik regresyon, kümeleme, sınıflandırma, doğal dil işleme ve zaman serisi modellerinden oluşan bir dizi model oluşturdunuz. Tebrikler! Şimdi, tüm bunların ne için olduğunu merak ediyor olabilirsiniz... Bu modellerin gerçek dünyadaki uygulamaları nelerdir?

Sanayide genellikle derin öğrenimden yararlanan yapay zeka büyük ilgi görse de, klasik makine öğrenimi modellerinin hala değerli uygulamaları bulunmaktadır. Bugün bile bu uygulamalardan bazılarını kullanıyor olabilirsiniz! Bu derste, sekiz farklı sektör ve konu alanının bu tür modelleri nasıl daha performanslı, güvenilir, akıllı ve kullanıcılar için değerli hale getirdiğini keşfedeceksiniz.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finans

Finans sektörü, makine öğrenimi için birçok fırsat sunar. Bu alandaki birçok problem, ML kullanılarak modellenebilir ve çözülebilir.

### Kredi Kartı Dolandırıcılığı Tespiti

[K-means kümeleme](../../5-Clustering/2-K-Means/README.md) hakkında daha önce kursta öğrendik, ancak kredi kartı dolandırıcılığıyla ilgili problemleri çözmek için nasıl kullanılabilir?

K-means kümeleme, **aykırı değer tespiti** adı verilen bir kredi kartı dolandırıcılığı tespit tekniğinde işe yarar. Aykırı değerler veya bir veri kümesi hakkındaki gözlemlerdeki sapmalar, bir kredi kartının normal bir şekilde mi yoksa olağandışı bir şekilde mi kullanıldığını bize gösterebilir. Aşağıdaki bağlantıda yer alan makalede gösterildiği gibi, k-means kümeleme algoritması kullanılarak kredi kartı verilerini sıralayabilir ve her işlemi ne kadar aykırı göründüğüne göre bir kümeye atayabilirsiniz. Daha sonra, en riskli kümeleri dolandırıcılık ve meşru işlemler açısından değerlendirebilirsiniz.
[Referans](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Varlık Yönetimi

Varlık yönetiminde, bir birey veya firma müşterileri adına yatırımları yönetir. Amaç, uzun vadede serveti korumak ve büyütmektir, bu nedenle iyi performans gösteren yatırımları seçmek çok önemlidir.

Belirli bir yatırımın nasıl performans gösterdiğini değerlendirmek için istatistiksel regresyon kullanılabilir. [Doğrusal regresyon](../../2-Regression/1-Tools/README.md), bir fonun belirli bir ölçütle nasıl performans gösterdiğini anlamak için değerli bir araçtır. Ayrıca regresyon sonuçlarının istatistiksel olarak anlamlı olup olmadığını veya bir müşterinin yatırımlarını ne kadar etkileyebileceğini çıkarabiliriz. Analizinizi daha da genişletmek için ek risk faktörlerini hesaba katabileceğiniz çoklu regresyon kullanabilirsiniz. Belirli bir fon için bunun nasıl çalışacağına dair bir örnek için aşağıdaki makaleye göz atabilirsiniz.
[Referans](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Eğitim

Eğitim sektörü, ML'nin uygulanabileceği çok ilginç bir alandır. Testlerde veya makalelerde hile yapmayı tespit etmek veya düzeltme sürecindeki kasıtlı veya kasıtsız önyargıyı yönetmek gibi ilginç problemler ele alınabilir.

### Öğrenci Davranışını Tahmin Etme

[Coursera](https://coursera.com), çevrimiçi bir açık kurs sağlayıcısı, birçok mühendislik kararını tartıştığı harika bir teknoloji bloguna sahiptir. Bu vaka çalışmasında, düşük NPS (Net Promoter Score) puanı ile kurs devamlılığı veya bırakma arasında bir korelasyon olup olmadığını keşfetmek için bir regresyon çizgisi çizdiler.
[Referans](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Önyargıyı Azaltma

[Grammarly](https://grammarly.com), yazım ve dilbilgisi hatalarını kontrol eden bir yazma asistanı, ürünlerinde sofistike [doğal dil işleme sistemleri](../../6-NLP/README.md) kullanır. Teknoloji bloglarında, makine öğreniminde cinsiyet önyargısını nasıl ele aldıklarını tartıştıkları ilginç bir vaka çalışması yayınladılar. Bu konuya [adalet dersi](../../1-Introduction/3-fairness/README.md) girişimizde değinmiştiniz.
[Referans](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Perakende

Perakende sektörü, müşteri yolculuğunu iyileştirmekten envanteri optimal bir şekilde stoklamaya kadar ML'nin kullanımından kesinlikle faydalanabilir.

### Müşteri Yolculuğunu Kişiselleştirme

Ev eşyaları satan bir şirket olan Wayfair'de, müşterilerin zevklerine ve ihtiyaçlarına uygun ürünleri bulmalarına yardımcı olmak çok önemlidir. Bu makalede, şirketin mühendisleri ML ve NLP'yi "müşteriler için doğru sonuçları ortaya çıkarmak" için nasıl kullandıklarını anlatıyor. Özellikle, Sorgu Niyet Motorları müşteri incelemelerinde varlık çıkarımı, sınıflandırıcı eğitimi, varlık ve görüş çıkarımı ve duygu etiketleme kullanılarak oluşturulmuştur. Bu, çevrimiçi perakendede NLP'nin nasıl çalıştığına dair klasik bir kullanım örneğidir.
[Referans](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Envanter Yönetimi

[StitchFix](https://stitchfix.com) gibi yenilikçi ve çevik şirketler, öneriler ve envanter yönetimi için ML'ye büyük ölçüde güveniyor. Stil ekipleri, ürün ekipleriyle birlikte çalışıyor: "Bir veri bilimcimiz genetik bir algoritma üzerinde çalıştı ve bunu bugün var olmayan başarılı bir kıyafet parçasını tahmin etmek için giyim üzerine uyguladı. Bunu ürün ekibine sunduk ve şimdi bunu bir araç olarak kullanabiliyorlar."
[Referans](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sağlık Hizmetleri

Sağlık sektörü, araştırma görevlerini ve hastaların yeniden hastaneye yatması veya hastalıkların yayılmasını durdurma gibi lojistik problemleri optimize etmek için ML'den yararlanabilir.

### Klinik Denemeleri Yönetme

Klinik denemelerde toksisite, ilaç üreticileri için büyük bir endişe kaynağıdır. Ne kadar toksisite tolere edilebilir? Bu çalışmada, çeşitli klinik deneme yöntemlerini analiz etmek, klinik deneme sonuçlarının olasılıklarını tahmin etmek için yeni bir yaklaşımın geliştirilmesine yol açtı. Özellikle, rastgele orman kullanarak gruplar arasındaki ilaçları ayırt edebilen bir [sınıflandırıcı](../../4-Classification/README.md) üretebildiler.
[Referans](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Hastane Yeniden Kabul Yönetimi

Hastane bakımı maliyetlidir, özellikle hastalar yeniden hastaneye yatırılmak zorunda kaldığında. Bu makale, [kümeleme](../../5-Clustering/README.md) algoritmalarını kullanarak yeniden kabul potansiyelini tahmin eden bir şirketi tartışıyor. Bu kümeler, analistlerin "ortak bir nedeni paylaşabilecek yeniden kabul gruplarını keşfetmesine" yardımcı olur.
[Referans](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Hastalık Yönetimi

Son pandemi, makine öğreniminin hastalık yayılmasını durdurmaya nasıl yardımcı olabileceğine dair parlak bir ışık tuttu. Bu makalede, ARIMA, lojistik eğriler, doğrusal regresyon ve SARIMA'nın kullanımını tanıyacaksınız. "Bu çalışma, bu virüsün yayılma hızını hesaplamaya ve böylece ölümleri, iyileşmeleri ve doğrulanmış vakaları tahmin etmeye çalışarak daha iyi hazırlanmamıza ve hayatta kalmamıza yardımcı olmayı amaçlamaktadır."
[Referans](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekoloji ve Yeşil Teknoloji

Doğa ve ekoloji, hayvanlar ve doğa arasındaki etkileşimlerin odak noktası olduğu birçok hassas sistemden oluşur. Bu sistemleri doğru bir şekilde ölçmek ve bir şeyler olduğunda, örneğin bir orman yangını veya hayvan popülasyonunda bir düşüş, uygun şekilde hareket etmek önemlidir.

### Orman Yönetimi

Önceki derslerde [Pekiştirmeli Öğrenme](../../8-Reinforcement/README.md) hakkında bilgi edindiniz. Doğadaki kalıpları tahmin etmeye çalışırken çok faydalı olabilir. Özellikle, orman yangınları ve istilacı türlerin yayılması gibi ekolojik problemleri izlemek için kullanılabilir. Kanada'da bir grup araştırmacı, uydu görüntülerinden orman yangını dinamik modelleri oluşturmak için Pekiştirmeli Öğrenme kullandı. Yenilikçi bir "mekansal yayılma süreci (SSP)" kullanarak, bir orman yangınını "manzaradaki herhangi bir hücredeki ajan" olarak hayal ettiler. "Yangının herhangi bir noktada bir konumdan alabileceği eylemler arasında kuzeye, güneye, doğuya veya batıya yayılma veya yayılmama yer alır."

Bu yaklaşım, ilgili Markov Karar Süreci (MDP) dinamiklerinin bilinen bir işlev olduğu için, genellikle RL kurulumunu tersine çevirir. Aşağıdaki bağlantıda bu grubun kullandığı klasik algoritmalar hakkında daha fazla bilgi edinin.
[Referans](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Hayvanların Hareket Algılaması

Derin öğrenme, hayvan hareketlerini görsel olarak izleme konusunda bir devrim yaratmış olsa da (kendi [kutup ayısı izleyicinizi](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) burada oluşturabilirsiniz), klasik ML bu görevde hala bir yere sahiptir.

Çiftlik hayvanlarının hareketlerini izlemek için sensörler ve IoT bu tür görsel işlemeyi kullanır, ancak daha temel ML teknikleri veri ön işleme için faydalıdır. Örneğin, bu makalede, koyun duruşları çeşitli sınıflandırıcı algoritmalar kullanılarak izlenmiş ve analiz edilmiştir. Sayfa 335'te ROC eğrisini tanıyabilirsiniz.
[Referans](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Enerji Yönetimi

[Zaman serisi tahmini](../../7-TimeSeries/README.md) derslerimizde, bir kasaba için arz ve talebi anlamaya dayalı olarak gelir elde etmek için akıllı park sayaçları kavramını ele aldık. Bu makale, İrlanda'daki gelecekteki enerji kullanımını tahmin etmek için kümeleme, regresyon ve zaman serisi tahmininin nasıl birleştirildiğini ayrıntılı olarak tartışıyor. Tahminler, akıllı sayaçlardan elde edilen verilere dayanıyor.
[Referans](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Sigorta

Sigorta sektörü, ML'yi uygulanabilir finansal ve aktüeryal modeller oluşturmak ve optimize etmek için kullanan bir başka sektördür.

### Volatilite Yönetimi

MetLife, bir hayat sigortası sağlayıcısı, finansal modellerindeki volatiliteyi analiz etme ve azaltma yöntemlerini açıkça paylaşmaktadır. Bu makalede ikili ve sıralı sınıflandırma görselleştirmelerini göreceksiniz. Ayrıca tahmin görselleştirmelerini keşfedeceksiniz.
[Referans](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Sanat, Kültür ve Edebiyat

Sanatta, örneğin gazetecilikte, birçok ilginç problem bulunmaktadır. Sahte haberleri tespit etmek büyük bir problemdir çünkü insanların görüşlerini etkilediği ve hatta demokrasileri devirdiği kanıtlanmıştır. Müzeler de ML'den, eserler arasındaki bağlantıları bulmaktan kaynak planlamasına kadar birçok alanda faydalanabilir.

### Sahte Haber Tespiti

Sahte haberleri tespit etmek, günümüz medyasında bir kedi-fare oyununa dönüşmüştür. Bu makalede, araştırmacılar, çalıştığımız ML tekniklerinden birkaçını birleştiren bir sistemin test edilip en iyi modelin uygulanabileceğini öneriyor: "Bu sistem, verilerden özellikler çıkarmak için doğal dil işleme temellidir ve ardından bu özellikler, Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) ve Logistic Regression (LR) gibi makine öğrenimi sınıflandırıcılarının eğitimi için kullanılır."
[Referans](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Bu makale, farklı ML alanlarını birleştirmenin, sahte haberlerin yayılmasını durdurmaya ve gerçek zararı önlemeye yardımcı olabilecek ilginç sonuçlar üretebileceğini göstermektedir; bu durumda, COVID tedavileri hakkında yayılan söylentilerin şiddet olaylarını tetiklemesi bir motivasyondu.

### Müze ML

Müzeler, koleksiyonları kataloglama ve dijitalleştirme ile eserler arasındaki bağlantıları bulmayı kolaylaştıran bir AI devriminin eşiğindedir. [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) gibi projeler, Vatikan Arşivleri gibi erişilemeyen koleksiyonların sırlarını açığa çıkarmaya yardımcı oluyor. Ancak, müzelerin iş yönü de ML modellerinden faydalanmaktadır.

Örneğin, Chicago Sanat Enstitüsü, ziyaretçilerin neyle ilgilendiğini ve sergilere ne zaman katılacaklarını tahmin etmek için modeller oluşturdu. Amaç, her ziyaretçinin müzeyi ziyaret ettiğinde bireyselleştirilmiş ve optimize edilmiş bir deneyim yaratmaktır. "2017 mali yılında, model katılım ve girişleri yüzde 1 doğrulukla tahmin etti, diyor Andrew Simnick, Chicago Sanat Enstitüsü'nde kıdemli başkan yardımcısı."
[Referans](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Pazarlama

### Müşteri Segmentasyonu

En etkili pazarlama stratejileri, müşterileri çeşitli gruplara göre farklı şekillerde hedefler. Bu makalede, farklı pazarlama stratejilerini desteklemek için Kümeleme algoritmalarının kullanımı tartışılmaktadır. Farklılaştırılmış pazarlama, şirketlerin marka bilinirliğini artırmasına, daha fazla müşteriye ulaşmasına ve daha fazla para kazanmasına yardımcı olur.
[Referans](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Zorluk

Bu müfredatta öğrendiğiniz tekniklerden faydalanan başka bir sektörü belirleyin ve bu sektörün ML'yi nasıl kullandığını keşfedin.
## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Wayfair veri bilimi ekibinin, şirketlerinde makine öğrenimini nasıl kullandıklarına dair birkaç ilginç videosu var. [Göz atmaya değer](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Ödev

[Bir ML hazine avı](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.