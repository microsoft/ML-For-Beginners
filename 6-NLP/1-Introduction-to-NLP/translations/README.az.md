# Təbii dil emalına giriş

Bu dərs *komputer dilçiliyinin* alt bölməsi olan *təbii dil emalının* qısa tarixçəsini və vacib anlayışları əhatə edir.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/?loc=az)

## Giriş

NLP (Natural Language Processing - Təbii Dil Emalı) maşın öyrənməsinin tətbiq olunduğu və proqramlarda istifadə olunan ən yaxşı bilinən sahədir.


✅ Gündəlik istifadə etdiyin hansı proqramların daxilində NLP olması ilə bağlı təxmin edə bilərsən? Mütəmadi olaraq söz emalı edən proqramlar və ya mobil tətbiqlər istifadə edirsənmi?

Bunlar barədə öyrənəcəksən:

- **Dillər ideyası**. Dillər necə inkişaf edib və hansı sahələrdə vacib araşdırlmalar aparılıb.
- **Tərif və anlayışlar**. Sən həmçinin komputerlərin mətnləri necə emal etməsi (qrammatikanı anlamaq, isimləri və felləri seçmək daxil olmaqla) barədə tərifləri və anlayışları öyrənəcəksən. Bu dərsdə bəzi kodlaşdırma tapşırıqları var və növbəti dərslərdə digər vacib anlayışlarla bunun davamı gələcək.

## Komputer dilçiliyi

Komputer dilçiliyi on illərdir araşdırma və inkişaf etdirilən bir sahədir. Burada komputerlərin dillərlə necə işləyə işləyə biləcəyi, hətta anlaya, tərcümə edə və əlaqə qura biləcəyi öyrənilir. NLP bununla əlaqəli sahədir və burada komputerlərin insanların istifadə etdiyi təbii dili necə emal edəcəyi diqqət mərkəzindədir.

### Nümanə - telefon imlası

Əgər indiyə kimi telefonunda yazmaq əvəzinə imla demisənsə, və ya virtual köməkçiyə sual vermisənsə, sənin nitqin mətn formasına çevrilib və sonra emal olunub. Tapılmış açar sözlər yenidən telefonun və ya köməkçinin başa düşəcəyi, işini görə biləcəyi formaya çevrilir.

![anlama](../images/comprehension.png)
> Həqiqi dilçiliyi anlamaq çətindir! [Jen Looper](https://twitter.com/jenlooper) təfədindən təsvir.

### Bu texnologiya necə mümkündür?

Biz bunu etməsi üçün komputer proqramı yazdığımıza görə mümkündür. Bir neçə onilliklər əvvəl bəzi elmi fantastika yazıçıları biz çox insanın komputerlə danışacağını, və komputerin onları həmişə doğru anlayacağını təxmin etmişdilər. Təəssüflər olsun ki, bu təsəvvür ediləndən daha çətin bir problem olduğu anlaşıldı, 'həqiqi' təbii dili emal etmək və cümlənin ifadə etdiyi mənanı başa düşmək üçün çox çətinliklər mövcuddur. Daxilində yumor və sarkazm daşıyan cümlələri başa düşmək xüsusilə çətin məsələdir.

Biz burada məktəb dərslərində müəllimlərin cümlənin daxilində istifadə olunan qrammatikanı əhatə etməsini xatırlayırıq. Bəzi ölkələrdə tələbələr qrammatika və dilçilik mövzularını xüsusi bir fənn kimi keçirlər, və ya bu mövzular onların öyrənmə kurikulumuna daxil edilirlər: ya ibtidai sinifdə oxuma və yazmaq dərsi kimi, ya orta məktəbdə ikinci dil kimi, ya da ali məktəbdə. İsimləri fellərdən və ya zərfləri sifətlərdən ayırd etməkdə ekspert deyilsənsə, buna görə narahat olma!

Əgər sən *indiki zaman* və *indiki davamedici* arasında fərqi anlamaqda çətinlik çəkirsənsə, tək deyilsən. Bu məsələ öz doğma dilində danışan bir çox insan üçün də çətindir. Yaxşı xəbər budur ki, komputerlər formal qaydaları tətbiq etməkdə çox yaxşıdırlar və sən cümləni insan kimi *oxmağı* bacaran kod yazmağı öyrənəcəksən. Əsas çətinliyin cümlənin *mənası* və *hissini* anlamaqda olduğunu görəcəksən.

## İlkin şərtlər

Bu dərsdə əsas tələb dərsin dilini oxuya və başa düşə bilməkdir. Burada həll olunmalı riyazi məsələ və ya tənliklər yoxdur. İlkin müəllif bu dərsi inglis dilində yazmış olsa da, bu həmçinin başqa dillərə tərcümə olunmuş ola bilər və sən tərcüməni oxumuş ola bilərsən. Burada müxtəlif dillərdə də istifadə olunan nümunələr vardır (müxtəlif dillərin qrammatika qaydalarını müqayisə etmək üçün). İzahedici mətnlərdən başqaları aydın başa düşülsün deyə tərcümə *olunmayıblar*.

Kodlaşdırma tapşırığında Python 3.8 proqramlaşdırma dili və nümunə mətnlər istifadə edəcəksən.

Bu bölmədə sənə lazım olacaq və istifadə edəcəksən:

- **Python 3 anlamaq**. Python 3 dilində bəzi proqramlaşdırma paradiqmlərini anlamaq lazımdır. Bu dərsdə giriş (input), dövrlər (loops), fayl oxuma (file reading) və massivlər (arrays) istifadə olunmuşdur.
- **Visual Studio Code + extension**. Biz Visual Studio Code proqramını və Python əlavəsini (extension) istifadə edəcəyik. İstədiyin başqa Python IDE də istifadə edə bilərsən.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) Python dilində sadələşdirilmiş mətn emalı üçün kitabxanadır. TextBlob veb saytındakı təlimatları izləyərək proqramı öz sisteminə yüklə (corpora proqramını da aşağıda qeyd olunduğu kimi yüklə):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Məsləhət: Sən Python proqramını birbaşa VS Code mühitində də icra edə bilərsən. Əlavə məlumat üçün [sənədlərə](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) baxa bilərsən.

## Maşınlarla danışmaq

Komputerlərin insan dilini anlaması tarixçəsi bir neçə onilliklərə gedir və ilk təbii dil emalını reallaşdıran alim *Alan Turinq* hesab olunur.

### 'Turinq testi'

Turinq 1950-ci illərdə *süni intellekt* haqqında araşdırma apararkən insanın yazı vasitəsi ilə ünsiyyət qurduğu tərəfin başqa bir insan və ya komputer olduğunu fərqində olmamasının mümkünlüyü düşünürdü.

Əgər müəyyən bir müddət söhbətdən sonra insan cavabları komputerdən, yoxsa insandan aldığını fərqləndirə bilməzsə, biz hesab bilərik ki komputer *düşünə bilir*?

### İlham - 'təqlid oyunu'

Bu ideya qrup daxili oynanılan *Təqlid oyunundan* götürülüb. Oyun zamanı otaqda olan bir şəxs (sorğu aparan) digər otaqda olan iki nəfərin uyğun olaraq oğlan və ya qız olduğunu tapması tələb olunur.
Sorğu aparan şəxs qeydlər göndərə bilər və yazılı cavablardan sirli şəxsin cinsini aşkar edə biləcək sualları düşünməyə çalışmalıdır. Təbii ki, digər otaqdakı oyunçular sorğuçunu səmimi cavablandırdıqlarına inandırmaqla çaşdırmağa da çalışırlar.

### Elizanın yaradılması

1960-cı illərdə *Cosept Veizenbaum* adlı MIT alimi insanlara sual verməklə onların cavabını başa düşdüyü görünüşü verən [*Eliza*](https://wikipedia.org/wiki/ELIZA) adlı komputer 'terapisti' yaratdı. Lakin Elizanın cümləndə bəzi açar sözləri və qrammatika quruluşunu oxuyaraq cavab verə bilməsi, onun cümləni həqiqətən *anladığı* demək olmaz. Əgər Elizaya "**Mən** <u>pisəm</u>" kimi cümlə daxil edilsə, o sözlərin yerini dəyişəcək və yeni cavab formalaşdıracaq - "**Sən** nə qədər müddətdir <u>pissən</u>?".

Bu Elizanın cümləni anladığı hissi verib açıq suallarla davam etsə də, reallıqda o sadəcə cümlənin zamanını dəyişir və əlavə sözlər daxil edir. Əgər Eliza cavab verə biləcək açar söz tapa bilməsə, o fərqli mənalara gətirə biləcək təsadüfi cavab verəcək. Eliza rahatlıqla aldadıla bilər, misal üçün istifadəçi əgər "**Sən** <u>velosipedsən</u>" yazsa, o mənalı cavab yerinə "Nə qədər müddətdir **mən** <u>velosipedəm</u>?" kimi cavab verəcək.

[![Eliza ilə danışmaq](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Eliza ilə danışmaq")

> 🎥 Orijinal ELIZA proqramı barədə videoya baxmaq üçün yuxarıdakı şəkilə kliklə

> Qeyd: Əgər ACM hesabın varsa, 1966-cı ildə nəşr edilmiş [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) haqqında orijinal məqaləni
oxuya bilərsən. Əlavə olaraq Eliza haqqında [wikipediada](https://wikipedia.org/wiki/ELIZA) da oxuya bilərsən.

## Tapşırıq - sadə danışıq botu kodlaması

Eliza kimi danışıq botu istifadəçi daxiletməsini qəbul edən, ağıllı şəkildə başa düşən və cavab verə bilən proqramdır. Elizadan fərqli olaraq bizim botun ağıllı söhbət apardığı görünüşü yaraqmaq üçün qaydaları olmayacaq. Bunun əvəzinə bizim botumuz hər bir sadə söhbətdə işlədilə biləcək təsadüfi cavabları verməklə söhbəti davam etdirməyə çalışacaq bir qabiliyyətə sahib olacaq.

### Plan

Danışıq botu yaratmaq üçün addımlar:

1. İstifadəçinin botla necə əlaqə qura biləcəyi barədə təlimatı ekrana çıxar
2. Dövr (loop) başlat
   1. İstifadəçi daxiletməsini qəbul et
   2. İstifadəçi çıxmaq istəsə, tətbiqdən çıx
   3. İstifadəçi daxiletməsini emal et və cavab seç (bu halda cavab mümkün ola biləcək ümumi cavablar siyahısından birinin təsadüfi seçilməsidir)
   4. Cavabı çap et
3. Dövrün başlanğıcına geri dön (2-ci addım)

### Bot yaratmaq

Gəlin indi botu yaradaq. Bəzi ifadələri təyin etməklə başlayacağıq.

1. Aşağıdakı təsadüfi cavablarla Python-da bu botu yaradın:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Burada kömək üçün bir neçə cavabla (output) tanış ola bilərsən (istifadəçi daxiletmələri olan sətirlər `>` simvolu ilə başlayır):

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

    Tapşırıq üçün həllərdən biri ilə [buradan](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py) tanış ola bilərsən.

    ✅ Dayan və düşün

    1. Düşünürsən ki, təsadüfi cavablar istifadəçini 'aldadaraq' botun onu həqiqətən anladığına inandıra bilər?
    2. Bot daha effektiv olması üçün hansı funksiyalara ehtiyac vardır?
    3. Əgər bot həqiqətən cümlənin mənasını 'başa düşsə', onun söhbətdəki əvvəlki cümlələrdəki mənanı da 'xatırlaması' lazımdırmı?

---

## 🚀 Məşğələ

Yuxarıdakı "dayan və düşün" elementlərindən birini seç və onu kodda tətbiq etməyə və ya həlli kağızda kod planı (pseudocode) kimi yazmağa çalış.

Növbəti dərsdə təbii dilin emalı və maşın öyrənməsi üçün bir neçə yanaşma öyrənəcəksən.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/?loc=az)

## Təkrarlayın və özünüz öyrənin

Əlavə oxumaq materialları üçün aşağıdakı istinadlara baxın.

### İstinadlar

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## Tapşırıq

[Bot axtarışı](assignment.az.md)
