<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-06T08:08:24+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "tr"
}
-->
# Çeviri ve Duygu Analizi ile Makine Öğrenimi

Önceki derslerde, temel NLP görevlerini gerçekleştirmek için sahne arkasında ML kullanan bir kütüphane olan `TextBlob` ile basit bir bot oluşturmayı öğrendiniz. Hesaplamalı dilbilimin bir diğer önemli zorluğu, bir cümleyi bir konuşulan veya yazılı dilden diğerine doğru bir şekilde _çevirme_ işlemidir.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

Çeviri, binlerce dilin varlığı ve her birinin çok farklı dilbilgisi kurallarına sahip olabilmesi nedeniyle oldukça zor bir problemdir. Bir yaklaşım, bir dilin (örneğin İngilizce) resmi dilbilgisi kurallarını dil bağımsız bir yapıya dönüştürmek ve ardından başka bir dile çevirerek geri dönüştürmektir. Bu yaklaşım şu adımları içerir:

1. **Tanımlama**. Girdi dilindeki kelimeleri isim, fiil vb. olarak tanımlayın veya etiketleyin.
2. **Çeviri oluşturma**. Hedef dil formatında her kelimenin doğrudan çevirisini üretin.

### Örnek cümle, İngilizceden İrlandacaya

İngilizcede _I feel happy_ cümlesi üç kelimeden oluşur ve sıralaması şu şekildedir:

- **özne** (I)
- **fiil** (feel)
- **sıfat** (happy)

Ancak, İrlandaca dilinde aynı cümle çok farklı bir dilbilgisi yapısına sahiptir - "*mutlu*" veya "*üzgün*" gibi duygular *üzerinde* olma durumu olarak ifade edilir.

İngilizce `I feel happy` ifadesi İrlandacada `Tá athas orm` olur. *Kelime kelime* çeviri `Mutluluk benim üzerimde` şeklinde olur.

Bir İrlandaca konuşan kişi İngilizceye çeviri yaparken `Happy is upon me` yerine `I feel happy` der, çünkü cümlenin anlamını anlar, kelimeler ve cümle yapısı farklı olsa bile.

İrlandaca cümle için resmi sıralama şu şekildedir:

- **fiil** (Tá veya is)
- **sıfat** (athas veya happy)
- **özne** (orm veya upon me)

## Çeviri

Basit bir çeviri programı yalnızca kelimeleri çevirir ve cümle yapısını görmezden gelir.

✅ Eğer bir yetişkin olarak ikinci (veya üçüncü ya da daha fazla) bir dil öğrenmişseniz, muhtemelen ana dilinizde düşünerek, bir kavramı kelime kelime kafanızda ikinci dile çevirerek ve ardından çevirinizi konuşarak başlamış olabilirsiniz. Bu, basit çeviri bilgisayar programlarının yaptığına benzer. Akıcılık kazanmak için bu aşamayı geçmek önemlidir!

Basit çeviri kötü (ve bazen komik) yanlış çevirilere yol açar: `I feel happy` İrlandacaya kelime kelime çevrildiğinde `Mise bhraitheann athas` olur. Bu, kelime kelime `ben hissediyorum mutluluk` anlamına gelir ve geçerli bir İrlandaca cümle değildir. İngilizce ve İrlandaca, birbirine yakın iki adada konuşulan diller olmasına rağmen, çok farklı dilbilgisi yapısına sahip dillerdir.

> İrlanda dil gelenekleri hakkında [bu video](https://www.youtube.com/watch?v=mRIaLSdRMMs) gibi bazı videolar izleyebilirsiniz.

### Makine öğrenimi yaklaşımları

Şimdiye kadar, doğal dil işleme için resmi kurallar yaklaşımını öğrendiniz. Bir diğer yaklaşım ise kelimelerin anlamını görmezden gelmek ve _makine öğrenimini kullanarak kalıpları tespit etmektir_. Bu, hem kaynak hem de hedef dillerde çok fazla metin (*corpus*) veya metinler (*corpora*) varsa çeviri için işe yarayabilir.

Örneğin, Jane Austen tarafından 1813 yılında yazılmış ünlü İngiliz romanı *Pride and Prejudice* (Gurur ve Önyargı) durumunu ele alalım. Kitabı İngilizce olarak ve kitabın *Fransızca* insan çevirisini incelerseniz, bir dildeki ifadelerin diğerine _deyimsel_ olarak çevrildiğini tespit edebilirsiniz. Bunu birazdan yapacaksınız.

Örneğin, İngilizce `I have no money` ifadesi Fransızcaya kelime kelime çevrildiğinde `Je n'ai pas de monnaie` olabilir. "Monnaie" Fransızca'da yanıltıcı bir 'false cognate'dir, çünkü 'money' ve 'monnaie' eş anlamlı değildir. İnsan çevirmen tarafından yapılabilecek daha iyi bir çeviri `Je n'ai pas d'argent` olur, çünkü bu, paranızın olmadığını (bozuk para anlamına gelen 'monnaie' yerine) daha iyi ifade eder.

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Görsel: [Jen Looper](https://twitter.com/jenlooper)

Eğer bir ML modeli, her iki dilde de uzman insan konuşmacılar tarafından daha önce çevrilmiş metinlerdeki ortak kalıpları tespit etmek için yeterli insan çevirisine sahipse, çevirilerin doğruluğunu artırabilir.

### Alıştırma - çeviri

Cümleleri çevirmek için `TextBlob` kullanabilirsiniz. **Pride and Prejudice**'ın ünlü ilk cümlesini deneyin:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` çeviriyi oldukça iyi yapar: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

TextBlob'ın çevirisinin, kitabın 1932 Fransızca çevirisi olan V. Leconte ve Ch. Pressoir tarafından yapılan çeviriden çok daha kesin olduğu söylenebilir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Bu durumda, ML tarafından bilgilendirilen çeviri, gereksiz yere orijinal yazarın ağzına kelimeler koyan insan çevirmeninden daha iyi bir iş çıkarır.

> Burada neler oluyor? Ve neden TextBlob çeviride bu kadar iyi? Aslında, sahne arkasında Google Translate kullanıyor, milyonlarca ifadeyi analiz edebilen ve görev için en iyi dizeleri tahmin edebilen sofistike bir yapay zeka. Burada manuel bir işlem yok ve `blob.translate` kullanmak için bir internet bağlantısına ihtiyacınız var.

✅ Daha fazla cümle deneyin. Hangisi daha iyi, ML mi yoksa insan çevirisi mi? Hangi durumlarda?

## Duygu Analizi

Makine öğreniminin çok iyi çalışabileceği bir diğer alan duygu analizidir. Duyguya yönelik bir ML olmayan yaklaşım, 'pozitif' ve 'negatif' olan kelimeleri ve ifadeleri tanımlamaktır. Ardından, yeni bir metin verildiğinde, genel duyguyu belirlemek için pozitif, negatif ve nötr kelimelerin toplam değerini hesaplayın.

Bu yaklaşım, Marvin görevinde gördüğünüz gibi kolayca yanıltılabilir - `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` cümlesi alaycı, negatif bir duygu cümlesidir, ancak basit algoritma 'great', 'wonderful', 'glad' kelimelerini pozitif ve 'waste', 'lost' ve 'dark' kelimelerini negatif olarak algılar. Genel duygu bu çelişkili kelimelerle etkilenir.

✅ Bir insan konuşmacı olarak alaycılığı nasıl ilettiğimizi bir saniye durup düşünün. Ses tonlaması büyük bir rol oynar. "Well, that film was awesome" ifadesini farklı şekillerde söyleyerek sesinizin anlamı nasıl ilettiğini keşfetmeye çalışın.

### ML Yaklaşımları

ML yaklaşımı, negatif ve pozitif metin gruplarını - tweetler, film incelemeleri veya bir insanın bir puan *ve* yazılı bir görüş verdiği herhangi bir şeyi - manuel olarak toplamak olacaktır. Ardından, görüşlere ve puanlara NLP teknikleri uygulanabilir, böylece kalıplar ortaya çıkar (örneğin, pozitif film incelemelerinde 'Oscar worthy' ifadesi negatif film incelemelerine göre daha sık görülür veya pozitif restoran incelemelerinde 'gourmet' kelimesi 'disgusting' kelimesinden çok daha fazla kullanılır).

> ⚖️ **Örnek**: Bir politikacının ofisinde çalıştığınızı ve tartışılan yeni bir yasa olduğunu varsayalım. Vatandaşlar, belirli yeni yasayı destekleyen veya karşı çıkan e-postalar yazabilir. Diyelim ki, e-postaları okuyup iki yığın halinde sıralamakla görevlisiniz: *destekleyen* ve *karşı çıkan*. Çok fazla e-posta varsa, hepsini okumaya çalışırken bunalmış hissedebilirsiniz. Tüm e-postaları sizin için okuyabilecek, anlayabilecek ve her bir e-postanın hangi yığında olması gerektiğini söyleyebilecek bir botun olması güzel olmaz mıydı? 
> 
> Bunu başarmanın bir yolu Makine Öğrenimi kullanmaktır. Modeli, *karşı çıkan* e-postaların bir kısmı ve *destekleyen* e-postaların bir kısmı ile eğitirsiniz. Model, karşı çıkan taraf ve destekleyen taraf ile belirli ifadeleri ve kelimeleri ilişkilendirme eğiliminde olur, *ancak içeriği anlamaz*, yalnızca belirli kelimelerin ve kalıpların bir *karşı çıkan* veya *destekleyen* e-postada daha sık ortaya çıkma olasılığı olduğunu bilir. Modeli, eğitmek için kullanmadığınız bazı e-postalarla test edebilir ve sizinle aynı sonuca ulaşıp ulaşmadığını görebilirsiniz. Ardından, modelin doğruluğundan memnun olduğunuzda, gelecekteki e-postaları her birini okumak zorunda kalmadan işleyebilirsiniz.

✅ Bu süreç, önceki derslerde kullandığınız süreçlere benziyor mu?

## Alıştırma - duygusal cümleler

Duygu, -1 ile 1 arasında bir *polarite* ile ölçülür, bu da -1'in en negatif duygu, 1'in ise en pozitif duygu olduğu anlamına gelir. Duygu ayrıca 0 - 1 arasında bir nesnellik (0) ve öznelik (1) puanı ile ölçülür.

Jane Austen'ın *Pride and Prejudice* kitabına tekrar bir göz atın. Metin, [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) adresinde mevcuttur. Aşağıdaki örnek, kitabın ilk ve son cümlelerinin duygusunu analiz eden ve duygu polaritesini ve öznelik/nesnellik puanını gösteren kısa bir programı göstermektedir.

Bu görevde `TextBlob` kütüphanesini (yukarıda açıklanmıştır) kullanarak `sentiment` belirlemeniz gerekir (kendi duygu hesaplayıcınızı yazmanız gerekmez).

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Aşağıdaki çıktıyı görürsünüz:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Zorluk - duygu polaritesini kontrol etme

Göreviniz, duygu polaritesini kullanarak *Pride and Prejudice* kitabında kesinlikle pozitif cümlelerin kesinlikle negatif cümlelerden daha fazla olup olmadığını belirlemektir. Bu görev için, polarite puanı 1 veya -1 olan bir cümlenin kesinlikle pozitif veya negatif olduğunu varsayabilirsiniz.

**Adımlar:**

1. [Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) kitabının bir kopyasını Project Gutenberg'den .txt dosyası olarak indirin. Dosyanın başındaki ve sonundaki metadataları kaldırarak yalnızca orijinal metni bırakın.
2. Dosyayı Python'da açın ve içeriği bir string olarak çıkarın.
3. Kitap stringi kullanarak bir TextBlob oluşturun.
4. Kitaptaki her cümleyi bir döngüde analiz edin.
   1. Eğer polarite 1 veya -1 ise cümleyi pozitif veya negatif mesajlar listesine kaydedin.
5. Sonunda, tüm pozitif cümleleri ve negatif cümleleri (ayrı ayrı) ve her birinin sayısını yazdırın.

İşte bir örnek [çözüm](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Bilgi Kontrolü

1. Duygu, cümlede kullanılan kelimelere dayanır, ancak kod *kelimeleri anlıyor mu*?
2. Duygu polaritesinin doğru olduğunu düşünüyor musunuz, başka bir deyişle, puanlarla *aynı fikirde misiniz*?
   1. Özellikle aşağıdaki cümlelerin kesinlikle **pozitif** polaritesiyle aynı fikirde misiniz veya değil misiniz?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Aşağıdaki 3 cümle kesinlikle pozitif bir duygu ile puanlanmış, ancak dikkatli bir okuma yapıldığında pozitif cümleler değildir. Duygu analizi neden bu cümleleri pozitif olarak değerlendirdi?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Aşağıdaki cümlelerin kesinlikle **negatif** polaritesiyle aynı fikirde misiniz veya değil misiniz?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Jane Austen'ın herhangi bir hayranı, yazarın kitaplarını İngiliz Regency toplumunun daha saçma yönlerini eleştirmek için sıklıkla kullandığını anlayacaktır. *Pride and Prejudice* kitabının ana karakteri Elizabeth Bennett, keskin bir sosyal gözlemcidir (yazar gibi) ve dili genellikle oldukça nüanslıdır. Hatta Mr. Darcy (hikayenin aşk ilgisi) Elizabeth'in dilini eğlenceli ve alaycı bir şekilde kullanmasını fark eder: "Sizinle tanışma zevkine yeterince uzun süredir sahibim ve ara sıra kendi görüşlerinizi ifade etmekten büyük keyif aldığınızı biliyorum."

---

## 🚀Zorluk

Kullanıcı girdisinden diğer özellikleri çıkararak Marvin'i daha da geliştirebilir misiniz?

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma
Metinden duygu çıkarımı yapmanın birçok yolu vardır. Bu tekniği kullanabilecek iş uygulamalarını düşünün. Bunun nasıl yanlış sonuçlar verebileceğini düşünün. Duygu analizi yapan, gelişmiş ve kurumsal kullanıma hazır sistemler hakkında daha fazla bilgi edinin, örneğin [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Yukarıdaki Pride and Prejudice cümlelerinden bazılarını test edin ve nüansı algılayıp algılayamadığını görün.

## Ödev 

[Şairane özgürlük](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalardan sorumlu değiliz.