# Makine öğreniminin tarihi

![Bir taslak-notta makine öğrenimi geçmişinin özeti](../../../sketchnotes/ml-history.png)
> [Tomomi Imura](https://www.twitter.com/girlie_mac) tarafından hazırlanan taslak-not

## [Ders öncesi test](https://white-water-09ec41f0f.azurestaticapps.net/quiz/3?loc=tr)

Bu derste, makine öğrenimi ve yapay zeka tarihindeki önemli kilometre taşlarını inceleyeceğiz.

Bir alan olarak yapay zekanın (AI) tarihi, makine öğreniminin tarihi ile iç içedir, çünkü makine öğrenimini destekleyen algoritmalar ve bilgi-işlem kapasitesindeki ilerlemeler, yapay zekanın gelişimini beslemektedir. Ayrı bilim alanlanları olarak bu alanlar 1950'lerde belirginleşmeye başlarken, önemli [algoritmik, istatistiksel, matematiksel, hesaplamalı ve teknik keşiflerin](https://wikipedia.org/wiki/Timeline_of_machine_learning) bir kısmı bu dönemden önce gelmiş ve bir kısmı da bu dönem ile örtüşmüştür. Aslında, insanlar [yüzlerce yıldır](https://wikipedia.org/wiki/History_of_artificial_intelligence) bu soruları düşünüyorlar: bu makale bir 'düşünen makine' fikrinin tarihsel entelektüel temellerini tartışıyor.

## Önemli keşifler

- 1763, 1812 - [Bayes Teoremi](https://tr.wikipedia.org/wiki/Bayes_teoremi) ve öncülleri. Bu teorem ve uygulamaları, önceki bilgilere dayalı olarak meydana gelen bir olayın olasılığını tanımlayan çıkarımın temelini oluşturur.
- 1805 - [En Küçük Kareler Teorisi](https://tr.wikipedia.org/wiki/En_k%C3%BC%C3%A7%C3%BCk_kareler_y%C3%B6ntemi), Fransız matematikçi Adrien-Marie Legendre tarafından bulunmuştur. Regresyon ünitemizde öğreneceğiniz bu teori, makine öğrenimi modelini veriye uydurmada yardımcı olur.
- 1913 - Rus matematikçi Andrey Markov'un adını taşıyan [Markov Zincirleri](https://tr.wikipedia.org/wiki/Markov_zinciri), önceki bir duruma dayalı olası olaylar dizisini tanımlamak için kullanılır.
- 1957 - [Algılayıcı (Perceptron)](https://tr.wikipedia.org/wiki/Perceptron), derin öğrenmedeki ilerlemelerin temelini oluşturan Amerikalı psikolog Frank Rosenblatt tarafından icat edilen bir tür doğrusal sınıflandırıcıdır.
- 1967 - [En Yakın Komşu](https://wikipedia.org/wiki/Nearest_neighbor), orijinal olarak rotaları haritalamak için tasarlanmış bir algoritmadır. Bir ML bağlamında kalıpları tespit etmek için kullanılır.
- 1970 - [Geri Yayılım](https://wikipedia.org/wiki/Backpropagation), [ileri beslemeli sinir ağlarını](https://wikipedia.org/wiki/Feedforward_neural_network) eğitmek için kullanılır.
- 1982 - [Tekrarlayan Sinir Ağları](https://wikipedia.org/wiki/Recurrent_neural_network), zamansal grafikler oluşturan ileri beslemeli sinir ağlarından türetilen yapay sinir ağlarıdır.

✅ Biraz araştırma yapın. Makine öğrenimi ve yapay zeka tarihinde önemli olan başka hangi tarihler öne çıkıyor?

## 1950: Düşünen makineler

[2019'da halk tarafından](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) 20. yüzyılın en büyük bilim adamı seçilen gerçekten dikkate değer bir kişi olan Alan Turing'in, 'düşünebilen makine' kavramının temellerini attığı kabul edilir. Kendisine karşı çıkanlara yanıt olması için ve bu kavramın deneysel kanıtlarını bulma ihtiyacı sebebiyle, NLP derslerimizde keşfedeceğiniz [Turing Testi'ni](https://www.bbc.com/news/technology-18475646) oluşturdu.

## 1956: Dartmouth Yaz Araştırma Projesi

"Yapay zeka üzerine Dartmouth Yaz Araştırma Projesi", bir alan olarak yapay zeka için çığır açan bir olaydı ve burada 'yapay zeka' terimi ortaya çıktı ([kaynak](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Öğrenmenin her yönü veya zekanın diğer herhangi bir özelliği, prensipte o kadar kesin bir şekilde tanımlanabilir ki, onu simüle etmek için bir makine yapılabilir.

Baş araştırmacı, matematik profesörü John McCarthy, "öğrenmenin her yönünün veya zekanın diğer herhangi bir özelliğinin prensipte oldukça kesin bir şekilde tanımlanabileceği varsayımına dayanarak, onu simüle etmek için bir makine yapılabileceği" varsayımının doğru olmasını umarak ilerliyordu. Katılımcılar arasında bu alanın bir diğer önderi olan Marvin Minsky de vardı.

Çalıştay, "sembolik yöntemlerin yükselişi, sınırlı alanlara odaklanan sistemler (ilk uzman sistemler) ve tümdengelimli sistemlere karşı tümevarımlı sistemler" dahil olmak üzere çeşitli tartışmaları başlatmış ve teşvik etmiştir. ([kaynak](https://tr.wikipedia.org/wiki/Dartmouth_Konferans%C4%B1)).

## 1956 - 1974: "Altın yıllar"

1950'lerden 70'lerin ortalarına kadar, yapay zekanın birçok sorunu çözebileceği umuduyla iyimserlik arttı. 1967'de Marvin Minsky kendinden emin bir şekilde "Bir nesil içinde... 'yapay zeka' yaratma sorunu büyük ölçüde çözülecek" dedi. (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Doğal dil işleme araştırmaları gelişti, aramalar iyileştirildi ve daha güçlü hale getirildi, ve basit görevlerin sade dil talimatları kullanılarak tamamlandığı 'mikro dünyalar' kavramı yaratıldı.

Araştırmalar, devlet kurumları tarafından iyi finanse edildi, hesaplamalar ve algoritmalarda ilerlemeler kaydedildi ve akıllı makinelerin prototipleri yapıldı. Bu makinelerden bazıları şunlardır:

* [Robot Shakey](https://wikipedia.org/wiki/Shakey_the_robot), manevra yapabilir ve görevleri 'akıllıca' nasıl yerine getireceğine karar verebilir.

    ![Shakey, akıllı bir robot](../images/shakey.jpg)
    > 1972'de Shakey

* Erken bir 'sohbet botu' olan Eliza, insanlarla sohbet edebilir ve ilkel bir 'terapist' gibi davranabilirdi. NLP derslerinde Eliza hakkında daha fazla bilgi edineceksiniz.

    ![Eliza, bir bot](../images/eliza.png)
    > Bir sohbet robotu olan Eliza'nın bir versiyonu

* "Dünya Blokları", blokların üst üste koyulabilecekleri, sıralanabilecekleri ve karar vermeyi öğreten makinelerdeki deneylerin test edilebileceği bir mikro dünyaya örnekti. [SHRDLU](https://wikipedia.org/wiki/SHRDLU) gibi kütüphanelerle oluşturulan gelişmeler, dil işlemeyi ilerletmeye yardımcı oldu.

    [![SHRDLU ile Dünya Blokları](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "SHRDLU ile Dünya Blokları" )
    
    > 🎥 Video için yukarıdaki resme tıklayın: SHRDLU ile Dünya Blokları

## 1974 - 1980: "Yapay Zekâ Kışı"

1970'lerin ortalarına gelindiğinde, 'akıllı makineler' yapmanın karmaşıklığının hafife alındığı ve mevcut hesaplama gücü göz önüne alındığında, verilen vaatlerin abartıldığı ortaya çıktı. Finansman kurudu ve alana olan güven azaldı. Güveni etkileyen bazı sorunlar şunlardı:

- **Kısıtlıklar**. Hesaplama gücü çok sınırlıydı.
- **Kombinasyonel patlama**. Hesaplama gücü ve yeteneğinde paralel bir evrim olmaksızın, bilgisayarlardan daha fazla soru istendikçe, eğitilmesi gereken parametre miktarı katlanarak arttı.
- **Veri eksikliği**. Algoritmaları test etme, geliştirme ve iyileştirme sürecini engelleyen bir veri kıtlığı vardı.
- **Doğru soruları mı soruyoruz?**. Sorulan sorular sorgulanmaya başlandı. Araştırmacılar mevcut yaklaşımları eleştirmeye başladı:
  - Turing testleri, diğer fikirlerin yanı sıra, "Çin odası teorisi" aracılığıyla sorgulanmaya başlandı. Bu teori, "dijital bir bilgisayar, programlanarak dili anlıyormuş gibi gösterilebilir fakat gerçek bir dil anlayışı elde edilemez" savını öne sürmektedir. ([kaynak](https://plato.stanford.edu/entries/chinese-room/)
  - "Terapist" ELIZA gibi yapay zekaların topluma tanıtılmasının etiğine meydan okundu.

Aynı zamanda, çeşitli yapay zekâ düşünce okulları oluşmaya başladı. ["dağınık" ile "düzenli AI"](https://wikipedia.org/wiki/Neats_and_scruffies) uygulamaları arasında bir ikilem kuruldu. _Dağınık_ laboratuvarlar, istenen sonuçları elde edene kadar programlar üzerinde saatlerce ince ayar yaptı. _Düzenli_ laboratuvarlar "mantık ve biçimsel problem çözmeye odaklandı". ELIZA ve SHRDLU, iyi bilinen _dağınık_ sistemlerdi. 1980'lerde, ML sistemlerinin sonuçlarını tekrarlanabilir hale getirmek için talep ortaya çıktıkça, sonuçları daha açıklanabilir olduğu için _düzenli_ yaklaşım yavaş yavaş ön plana çıktı.

## 1980'ler: Uzman sistemler

Alan büyüdükçe, şirketlere olan faydası daha net hale geldi ve 1980'lerde 'uzman sistemlerin' yaygınlaşması da bu şekilde meydana geldi. "Uzman sistemler, yapay zeka (AI) yazılımlarının gerçek anlamda başarılı olan ilk formları arasındaydı." ([kaynak](https://tr.wikipedia.org/wiki/Uzman_sistemler)).

Bu sistem türü aslında kısmen iş gereksinimlerini tanımlayan bir kural aracından ve yeni gerçekleri çıkarmak için kurallar sisteminden yararlanan bir çıkarım aracından oluşan bir _melezdir_.

Bu çağda aynı zamanda sinir ağlarına artan ilgi de görülmüştür.

## 1987 - 1993: Yapay Zeka 'Soğuması'

Özelleşmiş uzman sistem donanımının yaygınlaşması, talihsiz bir şekilde bunları aşırı özelleşmiş hale getirdi. Kişisel bilgisayarların yükselişi de bu büyük, özelleşmiş, merkezi sistemlerle rekabet etti. Bilgisayarın demokratikleşmesi başlamıştı ve sonunda modern büyük veri patlamasının yolunu açtı.

## 1993 - 2011

Bu çağ, daha önce veri ve hesaplama gücü eksikliğinden kaynaklanan bazı sorunları çözebilmek için ML ve AI için yeni bir dönemi getirdi. Veri miktarı hızla artmaya başladı ve özellikle 2007'de akıllı telefonun ortaya çıkmasıyla birlikte iyisiyle kötüsüyle daha yaygın bir şekilde ulaşılabilir hale geldi. Hesaplama gücü katlanarak arttı ve algoritmalar da onunla birlikte gelişti. Geçmişin başıboş günleri gitmiş, yerine giderek olgunlaşan gerçek bir disipline dönüşüm başlamıştı.

## Şimdi

Günümüzde makine öğrenimi ve yapay zeka hayatımızın neredeyse her alanına dokunuyor. Bu çağ, bu algoritmaların insan yaşamı üzerindeki risklerinin ve potansiyel etkilerinin dikkatli bir şekilde anlaşılmasını gerektirmektedir. Microsoft'tan Brad Smith'in belirttiği gibi, "Bilgi teknolojisi, gizlilik ve ifade özgürlüğü gibi temel insan hakları korumalarının kalbine giden sorunları gündeme getiriyor. Bu sorunlar, bu ürünleri yaratan teknoloji şirketlerinin sorumluluğunu artırıyor. Bizim açımızdan bakıldığında, düşünceli hükümet düzenlemeleri ve kabul edilebilir kullanımlar etrafında normların geliştirilmesi için de bir çağrı niteliği taşıyor." ([kaynak](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/) )).

Geleceğin neler getireceğini birlikte göreceğiz, ancak bu bilgisayar sistemlerini ve çalıştırdıkları yazılım ve algoritmaları anlamak önemlidir. Bu müfredatın, kendi kararlarınızı verebilmeniz için daha iyi bir anlayış kazanmanıza yardımcı olacağını umuyoruz.

[![Derin öğrenmenin tarihi](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Derin öğrenmenin tarihi")
> 🎥 Video için yukarıdaki resme tıklayın: Yann LeCun bu derste derin öğrenmenin tarihini tartışıyor

---
## 🚀Meydan okuma

Bu tarihi anlardan birine girin ve arkasındaki insanlar hakkında daha fazla bilgi edinin. Büyüleyici karakterler var ve kültürel bir boşlukta hiçbir bilimsel keşif yaratılmadı. Ne keşfedersiniz?

## [Ders sonrası test](https://white-water-09ec41f0f.azurestaticapps.net/quiz/4?loc=tr)

## İnceleme ve Bireysel Çalışma

İşte izlenmesi ve dinlenmesi gerekenler:

[Amy Boyd'un yapay zekanın evrimini tartıştığı bu podcast](http://runasradio.com/Shows/Show/739)

[![Amy Boyd ile Yapay Zekâ'nın tarihi](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Amy Boyd ile Yapay Zekâ'nın tarihi")

## Ödev

[Bir zaman çizelgesi oluşturun](assignment.tr.md)