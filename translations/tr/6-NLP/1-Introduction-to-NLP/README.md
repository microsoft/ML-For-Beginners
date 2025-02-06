# Doğal Dil İşlemeye Giriş

Bu ders, *hesaplamalı dilbilim* alt alanı olan *doğal dil işleme*nin kısa bir tarihini ve önemli kavramlarını kapsar.

## [Ders Öncesi Testi](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Giriş

Genellikle NLP olarak bilinen doğal dil işleme, makine öğreniminin uygulandığı ve üretim yazılımlarında kullanılan en bilinen alanlardan biridir.

✅ Her gün kullandığınız ve muhtemelen içinde biraz NLP barındıran bir yazılım düşünebilir misiniz? Peki ya düzenli olarak kullandığınız kelime işlem programları veya mobil uygulamalar?

Öğrenecekleriniz:

- **Dillerin fikri**. Dillerin nasıl geliştiği ve ana çalışma alanlarının neler olduğu.
- **Tanım ve kavramlar**. Bilgisayarların metni nasıl işlediğine dair tanımlar ve kavramlar, cümle çözümleme, dilbilgisi ve isim ve fiilleri tanımlama dahil. Bu derste bazı kodlama görevleri var ve sonraki derslerde kodlamayı öğreneceğiniz birkaç önemli kavram tanıtılıyor.

## Hesaplamalı Dilbilim

Hesaplamalı dilbilim, bilgisayarların dillerle nasıl çalışabileceğini, hatta anlayabileceğini, çevirebileceğini ve iletişim kurabileceğini araştıran ve geliştiren bir alandır. Doğal dil işleme (NLP), bilgisayarların 'doğal' veya insan dillerini nasıl işleyebileceğine odaklanan ilgili bir alandır.

### Örnek - telefon dikte

Telefonunuza yazmak yerine dikte ettiyseniz veya sanal bir asistana soru sorduysanız, konuşmanız bir metin formuna dönüştürülmüş ve ardından konuştuğunuz dilden *çözümleme* yapılmıştır. Algılanan anahtar kelimeler, telefonun veya asistanın anlayabileceği ve işlem yapabileceği bir formata dönüştürülmüştür.

![anlama](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.tr.png)
> Gerçek dilsel anlama zordur! Görsel [Jen Looper](https://twitter.com/jenlooper) tarafından

### Bu teknoloji nasıl mümkün hale geliyor?

Bu, birinin bunu yapmak için bir bilgisayar programı yazması sayesinde mümkündür. Birkaç on yıl önce, bazı bilim kurgu yazarları, insanların çoğunlukla bilgisayarlarıyla konuşacağını ve bilgisayarların her zaman ne demek istediklerini tam olarak anlayacağını öngörmüştü. Ne yazık ki, bu birçok kişinin hayal ettiğinden daha zor bir problem olduğu ortaya çıktı ve bugün çok daha iyi anlaşılan bir problem olmasına rağmen, bir cümlenin anlamını anlamak söz konusu olduğunda 'mükemmel' doğal dil işlemeye ulaşmakta önemli zorluklar vardır. Özellikle bir cümledeki mizahı anlamak veya alay gibi duyguları tespit etmek söz konusu olduğunda bu zor bir problemdir.

Bu noktada, öğretmenin bir cümledeki dilbilgisi bölümlerini ele aldığı okul derslerini hatırlayabilirsiniz. Bazı ülkelerde, öğrenciler dilbilgisi ve dilbilimi ayrı bir konu olarak öğretilirken, birçok ülkede bu konular bir dil öğrenmenin bir parçası olarak dahil edilir: ya ilkokulda ana dilinizi (okumayı ve yazmayı öğrenmek) ya da ortaokul veya lisede ikinci bir dili öğrenmek. İsimleri fiillerden veya zarfları sıfatlardan ayırt etme konusunda uzman değilseniz endişelenmeyin!

*Geniş zaman* ile *şimdiki zaman* arasındaki farkla mücadele ediyorsanız, yalnız değilsiniz. Bu, birçok insan için, hatta bir dilin ana konuşmacıları için bile zor bir şeydir. İyi haber şu ki, bilgisayarlar resmi kuralları uygulamada gerçekten iyidir ve bir cümleyi bir insan kadar iyi *çözümleyecek* kod yazmayı öğreneceksiniz. Daha sonra inceleyeceğiniz daha büyük zorluk, bir cümlenin *anlamını* ve *duygusunu* anlamaktır.

## Ön Koşullar

Bu ders için ana ön koşul, bu dersin dilini okuyabilmek ve anlayabilmektir. Çözülecek matematik problemleri veya denklemler yoktur. Orijinal yazar bu dersi İngilizce yazmış olsa da, başka dillere de çevrilmiştir, bu yüzden bir çeviri okuyabilirsiniz. Birkaç farklı dilin kullanıldığı örnekler vardır (farklı dillerin dilbilgisi kurallarını karşılaştırmak için). Bu diller *çevirilmemiştir*, ancak açıklayıcı metin çevrilmiştir, bu yüzden anlam net olmalıdır.

Kodlama görevleri için Python kullanacaksınız ve örnekler Python 3.8 kullanılarak yapılmıştır.

Bu bölümde, ihtiyacınız olacak ve kullanacaksınız:

- **Python 3 anlama**. Python 3'te programlama dili anlama, bu ders girdi, döngüler, dosya okuma, diziler kullanır.
- **Visual Studio Code + eklenti**. Visual Studio Code ve Python eklentisini kullanacağız. Ayrıca tercih ettiğiniz bir Python IDE'sini de kullanabilirsiniz.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob), Python için basitleştirilmiş bir metin işleme kütüphanesidir. TextBlob sitesindeki talimatları izleyerek sisteminize yükleyin (aşağıda gösterildiği gibi corpusları da yükleyin):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 İpucu: Python'u doğrudan VS Code ortamlarında çalıştırabilirsiniz. Daha fazla bilgi için [belgelere](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) göz atın.

## Makinelerle Konuşmak

Bilgisayarların insan dilini anlamasını sağlamaya yönelik çalışmalar on yıllar öncesine dayanır ve doğal dil işlemeyi düşünen en erken bilim insanlarından biri *Alan Turing* idi.

### 'Turing testi'

Turing, 1950'lerde *yapay zeka* araştırmaları yaparken, bir insana ve bilgisayara (yazılı iletişim yoluyla) bir konuşma testi verilse, insanın konuşmada başka bir insanla mı yoksa bir bilgisayarla mı konuştuğundan emin olamaması durumunu düşündü.

Belirli bir konuşma süresinden sonra, insan cevapların bir bilgisayardan mı yoksa başka bir insandan mı geldiğini belirleyemezse, bilgisayarın *düşündüğü* söylenebilir mi?

### İlham - 'taklit oyunu'

Bu fikir, bir sorgulayıcının bir odada yalnız olduğu ve başka bir odadaki iki kişinin cinsiyetini belirlemeye çalıştığı bir parti oyunundan geldi. Sorgulayıcı notlar gönderebilir ve yazılı cevapların gizemli kişinin cinsiyetini ortaya çıkaracak sorular düşünmeye çalışmalıdır. Tabii ki, diğer odadaki oyuncular, soruları yanıltıcı veya kafa karıştırıcı şekilde cevaplayarak sorgulayıcıyı yanıltmaya çalışırken, aynı zamanda dürüstçe cevap veriyormuş gibi görünmeye çalışır.

### Eliza'yı geliştirmek

1960'larda MIT'den bir bilim insanı olan *Joseph Weizenbaum*, [*Eliza*](https://wikipedia.org/wiki/ELIZA) adında bir bilgisayar 'terapisti' geliştirdi. Eliza, insana sorular sorar ve cevaplarını anlıyormuş gibi görünürdü. Ancak, Eliza bir cümleyi çözümleyip belirli dilbilgisi yapıları ve anahtar kelimeleri tanımlayarak makul bir cevap verebilirken, cümleyi *anladığı* söylenemezdi. Eliza, "**Ben** <u>üzgün</u>üm" formatındaki bir cümleye karşılık, cümledeki kelimeleri yeniden düzenleyip yerine koyarak "Ne kadar süredir **üzgün** <u>olduğunuzu</u> hissediyorsunuz" şeklinde yanıt verebilirdi.

Bu, Eliza'nın ifadeyi anladığı ve bir takip sorusu sorduğu izlenimini verirken, gerçekte, zamanı değiştirip bazı kelimeler ekliyordu. Eliza, yanıt verebileceği bir anahtar kelimeyi tanımlayamazsa, bunun yerine birçok farklı ifadeye uygulanabilecek rastgele bir yanıt verirdi. Eliza kolayca kandırılabilirdi, örneğin bir kullanıcı "**Sen** bir <u>bisiklet</u>sin" yazarsa, "Ne kadar süredir **ben** bir <u>bisiklet</u>im?" şeklinde yanıt verebilirdi, mantıklı bir yanıt yerine.

[![Eliza ile Sohbet](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Eliza ile Sohbet")

> 🎥 Yukarıdaki görüntüye tıklayarak orijinal ELIZA programı hakkında bir video izleyebilirsiniz

> Not: Bir ACM hesabınız varsa, 1966'da yayınlanan [Eliza'nın](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) orijinal tanımını okuyabilirsiniz. Alternatif olarak, Eliza hakkında [wikipedia](https://wikipedia.org/wiki/ELIZA)'dan bilgi edinin

## Alıştırma - Temel Bir Konuşma Botu Kodlama

Eliza gibi bir konuşma botu, kullanıcı girdilerini alan ve anlamış gibi görünen ve akıllıca yanıt veren bir programdır. Eliza'nın aksine, botumuz akıllı bir konuşma izlenimi veren birkaç kurala sahip olmayacak. Bunun yerine, botumuzun tek bir yeteneği olacak, neredeyse her sıradan konuşmada işe yarayabilecek rastgele yanıtlarla konuşmayı sürdürmek.

### Plan

Bir konuşma botu oluştururken adımlarınız:

1. Kullanıcıya botla nasıl etkileşime geçeceğini anlatan talimatları yazdırın
2. Bir döngü başlatın
   1. Kullanıcı girdisini kabul edin
   2. Kullanıcı çıkmak isterse, çıkın
   3. Kullanıcı girdisini işleyin ve yanıtı belirleyin (bu durumda, yanıt olası genel yanıtlar listesinden rastgele bir seçimdir)
   4. Yanıtı yazdırın
3. Adım 2'ye geri dönün

### Botu Oluşturma

Şimdi botu oluşturalım. Öncelikle bazı ifadeleri tanımlayarak başlayacağız.

1. Aşağıdaki rastgele yanıtlarla bu botu kendiniz Python'da oluşturun:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    İşte size rehberlik etmesi için bazı örnek çıktılar (kullanıcı girdisi `>` ile başlayan satırlarda):

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

    Göreve olası bir çözüm [burada](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Düşünün ve durun

    1. Rastgele yanıtların birinin botun gerçekten anladığını düşündüreceğini düşünüyor musunuz?
    2. Botun daha etkili olması için hangi özelliklere ihtiyaç duyardı?
    3. Bir bot gerçekten bir cümlenin anlamını anlayabilseydi, önceki cümlelerin anlamını da 'hatırlaması' gerekir miydi?

---

## 🚀Meydan Okuma

Yukarıdaki "düşünün ve durun" unsurlarından birini seçin ve bunu kodda uygulamaya çalışın veya bir çözümü kağıt üzerinde sahte kod kullanarak yazın.

Bir sonraki derste, doğal dili çözümleme ve makine öğrenimine yönelik başka yaklaşımlar hakkında bilgi edineceksiniz.

## [Ders Sonrası Testi](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## İnceleme ve Kendi Kendine Çalışma

Aşağıdaki referanslara daha fazla okuma fırsatı olarak göz atın.

### Referanslar

1. Schubert, Lenhart, "Hesaplamalı Dilbilim", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "WordNet Hakkında." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Ödev 

[Bir bot arayın](assignment.md)

**Feragatname**:
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba sarf etsek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dilindeki hali yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi tavsiye edilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.