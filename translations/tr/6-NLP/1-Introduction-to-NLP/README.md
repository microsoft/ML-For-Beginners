<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-06T08:07:48+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "tr"
}
-->
# Doğal Dil İşlemeye Giriş

Bu ders, *doğal dil işleme* alanının kısa bir tarihçesini ve önemli kavramlarını kapsar. Doğal dil işleme, *hesaplamalı dilbilim* alt alanlarından biridir.

## [Ders Öncesi Test](https://ff-quizzes.netlify.app/en/ml/)

## Giriş

NLP (Doğal Dil İşleme), genellikle bilindiği gibi, makine öğreniminin uygulandığı ve üretim yazılımlarında kullanıldığı en bilinen alanlardan biridir.

✅ Her gün kullandığınız ve muhtemelen içinde NLP bulunan bir yazılım düşünebilir misiniz? Peki ya düzenli olarak kullandığınız kelime işlem programları veya mobil uygulamalar?

Bu derste şunları öğreneceksiniz:

- **Dillerin fikri**. Dillerin nasıl geliştiği ve başlıca çalışma alanlarının neler olduğu.
- **Tanım ve kavramlar**. Bilgisayarların metni nasıl işlediği, ayrıştırma, dilbilgisi ve isim-fiil tanımlama gibi konular hakkında tanımlar ve kavramlar öğreneceksiniz. Bu derste bazı kodlama görevleri bulunuyor ve sonraki derslerde kodlamayı öğreneceğiniz birkaç önemli kavram tanıtılıyor.

## Hesaplamalı Dilbilim

Hesaplamalı dilbilim, bilgisayarların dillerle nasıl çalışabileceğini, hatta anlayabileceğini, çevirebileceğini ve iletişim kurabileceğini inceleyen, onlarca yıllık bir araştırma ve geliştirme alanıdır. Doğal dil işleme (NLP), bilgisayarların 'doğal' yani insan dillerini nasıl işleyebileceğine odaklanan ilgili bir alandır.

### Örnek - Telefon Dikte

Telefonunuza yazmak yerine dikte ettiyseniz veya bir sanal asistana soru sorduysanız, konuşmanız metin formuna dönüştürülmüş ve ardından konuştuğunuz dilde *ayrıştırılmıştır*. Algılanan anahtar kelimeler, telefonun veya asistanın anlayabileceği ve işlem yapabileceği bir formata dönüştürülmüştür.

![anlama](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Gerçek dilbilimsel anlama zordur! Görsel: [Jen Looper](https://twitter.com/jenlooper)

### Bu teknoloji nasıl mümkün hale geliyor?

Bu, birinin bu işlemi gerçekleştiren bir bilgisayar programı yazması sayesinde mümkün hale geliyor. Birkaç on yıl önce, bazı bilim kurgu yazarları insanların çoğunlukla bilgisayarlarıyla konuşacağını ve bilgisayarların her zaman tam olarak ne demek istediklerini anlayacağını öngörmüştü. Ne yazık ki, bu birçok kişinin hayal ettiğinden daha zor bir problem olduğu ortaya çıktı ve bugün çok daha iyi anlaşılan bir problem olmasına rağmen, bir cümlenin anlamını anlamada 'mükemmel' doğal dil işlemeyi başarmada önemli zorluklar bulunmaktadır. Özellikle bir cümlede mizahı anlamak veya alay gibi duyguları tespit etmek söz konusu olduğunda bu oldukça zor bir problemdir.

Bu noktada, okulda öğretmenin bir cümledeki dilbilgisi bölümlerini ele aldığı dersleri hatırlıyor olabilirsiniz. Bazı ülkelerde, öğrencilere dilbilgisi ve dilbilim ayrı bir ders olarak öğretilir, ancak birçok ülkede bu konular bir dil öğrenmenin bir parçası olarak öğretilir: ya ilkokulda birinci dilinizi (okuma ve yazmayı öğrenme) ya da ortaokul veya lisede ikinci bir dili öğrenirken. İsimleri fiillerden veya zarfları sıfatlardan ayırmada uzman değilseniz endişelenmeyin!

Eğer *basit geniş zaman* ile *şimdiki zamanın hikayesi* arasındaki farkı anlamakta zorlanıyorsanız, yalnız değilsiniz. Bu, birçok kişi için, hatta bir dilin ana konuşmacıları için bile zor bir şeydir. İyi haber şu ki, bilgisayarlar resmi kuralları uygulamada gerçekten iyidir ve bir cümleyi bir insan kadar iyi *ayrıştırabilen* kod yazmayı öğreneceksiniz. Daha sonra inceleyeceğiniz daha büyük zorluk ise bir cümlenin *anlamını* ve *duygusunu* anlamaktır.

## Ön Koşullar

Bu ders için ana ön koşul, bu dersin dilini okuyup anlayabilmektir. Çözülecek matematik problemleri veya denklemler yoktur. Orijinal yazar bu dersi İngilizce yazmış olsa da, ders diğer dillere de çevrilmiştir, dolayısıyla bir çeviri okuyor olabilirsiniz. Farklı dillerin dilbilgisi kurallarını karşılaştırmak için kullanılan bazı örnekler vardır. Bu örnekler *çevirilmez*, ancak açıklayıcı metin çevrilir, bu nedenle anlam açık olmalıdır.

Kodlama görevleri için Python kullanacaksınız ve örnekler Python 3.8 ile yapılmıştır.

Bu bölümde ihtiyacınız olanlar ve kullanacaklarınız:

- **Python 3 anlama**. Python 3 programlama dilini anlama, bu derste giriş, döngüler, dosya okuma, diziler kullanılıyor.
- **Visual Studio Code + eklenti**. Visual Studio Code ve Python eklentisini kullanacağız. Ayrıca tercih ettiğiniz bir Python IDE'sini de kullanabilirsiniz.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob), Python için basitleştirilmiş bir metin işleme kütüphanesidir. TextBlob sitesindeki talimatları izleyerek sisteminize kurun (aşağıda gösterildiği gibi corpusları da yükleyin):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 İpucu: Python'u doğrudan VS Code ortamlarında çalıştırabilirsiniz. Daha fazla bilgi için [dokümanlara](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) göz atın.

## Makinelerle Konuşmak

Bilgisayarların insan dilini anlamasını sağlama çabalarının tarihi onlarca yıl öncesine dayanır ve doğal dil işlemeyi düşünen ilk bilim insanlarından biri *Alan Turing* idi.

### 'Turing Testi'

Turing, 1950'lerde *yapay zeka* araştırmaları yaparken, bir insan ve bilgisayara (yazılı iletişim yoluyla) bir konuşma testi verilip, konuşmadaki insanın başka bir insanla mı yoksa bir bilgisayarla mı konuştuğundan emin olamadığı bir durumu düşündü.

Eğer belirli bir uzunlukta bir konuşmadan sonra insan, cevapların bir bilgisayardan mı yoksa bir insandan mı geldiğini belirleyemezse, bilgisayarın *düşündüğü* söylenebilir mi?

### İlham - 'Taklit Oyunu'

Bu fikir, bir sorgulayıcının bir odada yalnız olduğu ve diğer odadaki iki kişinin sırasıyla erkek ve kadın olduğunu belirlemeye çalıştığı bir parti oyunu olan *Taklit Oyunu*ndan geldi. Sorgulayıcı notlar gönderebilir ve yazılı cevapların gizemli kişinin cinsiyetini ortaya çıkaracağı sorular düşünmeye çalışmalıdır. Tabii ki, diğer odadaki oyuncular sorgulayıcıyı yanıltmak veya kafa karıştırmak için soruları yanıltıcı bir şekilde cevaplamaya çalışırken aynı zamanda dürüstçe cevap veriyormuş gibi görünmeye çalışırlar.

### Eliza'yı Geliştirmek

1960'larda MIT'den bir bilim insanı olan *Joseph Weizenbaum*, insanlara sorular soran ve onların cevaplarını anlıyormuş gibi görünen bir bilgisayar 'terapisti' olan [*Eliza*](https://wikipedia.org/wiki/ELIZA)'yı geliştirdi. Ancak, Eliza bir cümleyi ayrıştırıp belirli dilbilgisi yapıları ve anahtar kelimeleri tanımlayarak makul bir cevap verebilse de, cümleyi *anladığı* söylenemezdi. Eliza'ya "**Ben** <u>üzgün</u>üm" formatında bir cümle sunulursa, cümledeki kelimeleri yeniden düzenleyip yerine koyarak "Ne zamandır **sen** <u>üzgün</u>sün?" şeklinde bir cevap oluşturabilirdi.

Bu, Eliza'nın ifadeyi anladığı ve takip eden bir soru sorduğu izlenimini veriyordu, oysa gerçekte sadece zamanı değiştiriyor ve bazı kelimeler ekliyordu. Eliza, yanıt verebileceği bir anahtar kelimeyi tanımlayamazsa, bunun yerine birçok farklı ifadeye uygulanabilecek rastgele bir yanıt verirdi. Örneğin, bir kullanıcı "**Sen** bir <u>bisiklet</u>sin" yazarsa, "Ne zamandır **ben** bir <u>bisiklet</u>im?" şeklinde bir yanıt verebilirdi, daha mantıklı bir yanıt yerine.

[![Eliza ile Sohbet](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Eliza ile Sohbet")

> 🎥 Yukarıdaki görsele tıklayarak orijinal ELIZA programı hakkında bir video izleyebilirsiniz.

> Not: [Eliza'nın](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) 1966'da yayınlanan orijinal açıklamasını bir ACM hesabınız varsa okuyabilirsiniz. Alternatif olarak, Eliza hakkında [wikipedia](https://wikipedia.org/wiki/ELIZA)'dan bilgi edinebilirsiniz.

## Egzersiz - Temel Bir Konuşma Botu Kodlama

Eliza gibi bir konuşma botu, kullanıcı girdisini alan ve anlamış gibi görünerek akıllıca yanıt veren bir programdır. Eliza'nın aksine, botumuzun akıllı bir konuşma yapıyormuş gibi görünmesini sağlayan birkaç kuralı olmayacak. Bunun yerine, botumuz yalnızca bir yeteneğe sahip olacak: rastgele yanıtlarla neredeyse herhangi bir basit konuşmayı sürdürmek.

### Plan

Bir konuşma botu oluştururken adımlarınız:

1. Kullanıcıya botla nasıl etkileşim kuracağına dair talimatları yazdırın
2. Bir döngü başlatın
   1. Kullanıcı girdisini kabul edin
   2. Kullanıcı çıkmak istediğini belirtirse çıkın
   3. Kullanıcı girdisini işleyin ve yanıtı belirleyin (bu durumda yanıt, olası genel yanıtlar listesinden rastgele bir seçimdir)
   4. Yanıtı yazdırın
3. Adım 2'ye geri dönün

### Botu Oluşturma

Şimdi botu oluşturalım. Öncelikle bazı ifadeleri tanımlayarak başlayacağız.

1. Aşağıdaki rastgele yanıtlarla Python'da bu botu kendiniz oluşturun:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    İşte size rehberlik edecek bazı örnek çıktılar (kullanıcı girdisi `>` ile başlayan satırlarda):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Görev için olası bir çözüm [burada](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Dur ve düşün

    1. Rastgele yanıtların birini botun gerçekten onları anladığına inandırabileceğini düşünüyor musunuz?
    2. Botun daha etkili olması için hangi özelliklere ihtiyacı olurdu?
    3. Bir bot gerçekten bir cümlenin anlamını 'anlayabilseydi', önceki cümlelerin anlamını bir konuşmada 'hatırlaması' gerekir miydi?

---

## 🚀Meydan Okuma

Yukarıdaki "dur ve düşün" unsurlarından birini seçin ve ya kodda uygulamaya çalışın ya da kağıt üzerinde bir çözüm yazın (sözde kod kullanarak).

Bir sonraki derste, doğal dili ayrıştırma ve makine öğrenimi için bir dizi başka yaklaşımı öğreneceksiniz.

## [Ders Sonrası Test](https://ff-quizzes.netlify.app/en/ml/)

## Gözden Geçirme ve Kendi Kendine Çalışma

Aşağıdaki referanslara göz atarak daha fazla okuma fırsatları değerlendirin.

### Referanslar

1. Schubert, Lenhart, "Hesaplamalı Dilbilim", *Stanford Felsefe Ansiklopedisi* (Bahar 2020 Baskısı), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton Üniversitesi "WordNet Hakkında." [WordNet](https://wordnet.princeton.edu/). Princeton Üniversitesi. 2010. 

## Ödev 

[Bir bot arayın](assignment.md)

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.