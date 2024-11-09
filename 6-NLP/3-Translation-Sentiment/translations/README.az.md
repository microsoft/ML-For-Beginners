# Tərcümə və mətn analizi ilə ML

Əvvəlki dərslərdə əsas nitq birləşmələrinin çıxarılması kimi əsas NLP vəzifələrini yerinə yetirmək üçün səhnə arxasında ML istifadə edən TextBlob kitabxanası ilə bot yaratmağı öyrəndiniz. Hesablama dilçiliyində digər mühüm bir məsələ isə bir cümlənin danışıq və ya yazılı dildən digərinə dəqiq tərcümə edilməsidir.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/?loc=az)

Tərcümə çox çətin bir məsələdir, çünki minlərlə dil var və hər birinin çox fərqli qrammatika qaydaları ola bilər. Bir yanaşma bir dili, məsələn, İngilis dilinin rəsmi qrammatika qaydalarını dilin özündən asılı olmayan bir struktura çevirmək və sonra başqa bir dilə tərcümə edərək geri çevirməkdir. Bu yanaşma aşağıdakı addımları atmağınızı tələb edir:

1. **İdentifikasiya**. Daxil olan dildəki sözləri isim, fel və s. kimi müəyyənləşdirin və ya işarələyin.
2. **Tərcümə yaradın**. Hədəf dil formatında hər sözün birbaşa tərcüməsini hazırlayın.

### Nümunə cümlə, İngilis dilindən İrland dilinə

'İngilis', dilində cümlə _I feel happy_ üç ardıcıl sözdən ibarətdir:

- **isim** (I)
- **feil** (feel)
- **sifət** (happy)

Lakin, 'İrland' dilində eyni cümlə çox fərqli qrammatik quruluşa malikdir - "*xoşbəxt*" və ya "**kədərli" kimi hisslər sanki sənin üzərində imiş kimi ifadə edilir.

İngilis dilindəki `I feel happy` ifadəsi İrland dilində `Tá athas orm` formasında olardı. *literal* tərcüməsi `Xoşbəxtlik mənim üstümdədir`. olaraq anlaşılır.

İrland dilində danışan biri İngilis dilinə tərcümə edərkən `I feel happy` deyər, `Happy is upon me` (Xoşbəxtlik mənim üzərimdədir) deməz, çünki cümlənin mənasını fərqli formada başa düşülər, hətta sözlər və cümlə quruluşu fərqli olsa belə.

İrland dilindəki cümlə quruluşu:

- **fel** (Tá və ya is)
- **sifət** (athas, və ya xoşbəxt)
- **isim** (orm, or mənim üzərimdə)

## Tərcümə

Sadə tərcümə proqramları yalnız sözləri tərcümə edə bilir, cümlə strukturunu nəzərə almır.

✅ Əgər sən yetişkin biri kimi ikinci (üçüncü və ya daha çox) dil öyrənmisənsə, çox güman ki, birinci öz doğma dilində fikirləşməyə başlamısan, fikrini beynində ikinci dilə sözbəsöz tərcümə etmisən və sonra öz tərcüməni səsləndirmisən. Tərcümə kompüter proqramları da bunun bənzərini edir. Səmərəliliyə nail olmaq üçün bu mərhələni keçmək vacibdir!

Sadə tərcümə yanlış tərcümələrə (bəzən möhtəşəmlərinə) gətirib çıxara bilir: `I feel happy` (mən yaxşı hiss edirəm) irland dilinə hərfi olaraq `Mise bhraitheann athas` kimi tərcümə olunur. Bu (hərfi olaraq) `me feel happy` (mən yaxşı hiss edirəm) kimi mənaya gəlir, lakin irland dilində bu düzgün bir cümlə deyil. Baxmayaraq ki, inglis və irland dilləri iki yaxın adada danışılır, onlar çox fərqli qrammatik strukturları ilə çox fərqli dillərdir.

> İrland dilçiliyi ənənələri barədə [bunun kimi](https://www.youtube.com/watch?v=mRIaLSdRMMs) bəzi videolara baxa bilərsən.

### Maşın öyrənməsi yanaşmaları

İndiyə kimi təbii dil emalına rəsmi qaydalarla yanaşmalarını öyrənmisən. Başqa yanaşma sözlərin mənasını nəzərə almamaq və _onun yerinə maşın öyrənmə istifadə etməklə modeli tapmaqdır_. Əgər həm başlanğıc, həm də təyinat dillərdə çox sayda söz (*corpus*) və ya sözləri (*corpora*) tərcümə etmək istəyirsənsə, bu metod yaxşı işləyir.

Misal olaraq 1813-cü ildə Ceyn Austin tərəfindən yazılan məşhur inglis hekayəsi *Qürur və qərəzi* nəzərə alaq. Əgər sən kitabı inglis dilində analiz etsən və insan tərəfindən *fransız* dilinə tərcümə olunmuş versiyasını oxusan orada bəzi söz birləşmələrinin _idiomatik olaraq_ digər dilinə tərcümə olunduğuna şahid olacaqsan. Tezliklə bunu görəcəksən.

Misal üçün inglis dilində `I have no money`(mənim pulum yoxdur) ifadəsi hərfi olaraq fransız dilinə tərcümə olunanda `Je n'ai pas de monnaie` çevrilir. "Monnaie" sözü fransız dilində 'yanlış qohum' sözdür, belə ki 'money' və 'monnaie' sözləri sinonim deyillər. İnsan tərəfindən daha yaxşı tərcümə olsunsa `Je n'ai pas d'argent` kimi olardı, çünki burda daha aydın şəkildə pulun olmadığı mənası verir (nəinki 'monnaie' sözünün mənası 'boş dəyişiklik' mənasını verir).

![monnaie](../images/monnaie.png)

> [Cen Luper](https://twitter.com/jenlooper) tərəfindən şəkil

Əgər ML modeli kifayət qədər insan tərcümələrinə sahib olsa, o tərcübəli və hər iki dildə danışan insanların əvvəlki tərcümələri üzərindən ümumi qaydaları taparaq dəqiqliyini daha da inkişaf etdirə bilər.

### Məşğələ - tərcümə

Sən cümlələri tərcümə etmək üçün `TextBlob` istifadə edə bilərsən. **Qürur və qərəz**-dən olan ilk məşhur cümləni yoxla:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` tərcümədə yaxşı nəticə göstərir: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

TextBlob-un tərcüməsinin dəqiqlikdən biraz uzaq olduğu müzakirə oluna bilər, fakt olaraq V. Lakont və Ç. Pressorun 1932-ci kitabının fransızca tərcüməsi belədir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Bu vəziyyətdə ML tərəfindən tərcümə tapşırığı daha yaxşı yerinə yetirir, nəinki insan tərcüməçi müəllifin original sözləri arasında 'dəqiqliyi' artırmaq üçün lazımsız sözlər daxil edir.

> Burada nə baş verir? Nə üçün TextBlob tərcümədə çox yaxşıdır? O arxa planda tapşırığı yerinə yetirmək üçün ən yaxşı sətirləri proqnozlaşdırmaq üçün milyonlarla ifadəni təhlil edə bilən mürəkkəb AI - Google tərcümə proqramını istifadə edir. Burada heç bir iş əllə görülmür və sənin `blob.translate` istifadə edə bilməyin üçün internet əlaqəsinə ehtiyacın var.

✅ Müxtəlif cümlələr yoxla. Ml yoxsa insan tərcüməsi daha yaxşıdır? Hansı hallarda?

## Hiss analizi

Maşın öyrənmənin çox yaxşı işləyə biləcəyi başqa bir sahə hiss analizidir. Hissələrə qeyri-ML yanaşma "müsbət" və "mənfi" olan söz və ifadələri müəyyən etməkdir. Sonra, yeni bir mətn parçası verilərkən ümumi əhval-ruhiyyəni müəyyən etmək üçün müsbət, mənfi və neytral sözlərin ümumi dəyərini hesablayın.

Bu yanaşma Marvin tapşırığında gördüyümüz cümlə kimi çox rahatlıqla aldadıla bilər, `Əla, bu gözəl vaxt itkisi idi, bu qaranlıq yolda itdiyimiz üçün şadam` cümləsi sarkazm və neqativ hiss daşıyır, lakin sadə alqoritm 'əla', 'gözəl', 'şad' sözlərini müsbət, 'itki', 'itmiş' və 'qaranlıq' is mənfi kimi təyin edəcək. Ümumi hiss bu qarşılıqlı sözlərə görə qeyri-dəqiq olacaq.

✅ Bir saniyə dayanın və insan kimi sarkazmı necə çatdırdığımızı düşünün. Ton oynaması böyük rol oynayır. Səsinizin mənanı necə çatdırdığını görmək üçün müxtəlif yollarla "Yaxşı, o film möhtəşəm idi" ifadəsini deməyə çalışın.

### ML yanaşmaları

ML yanaşması mətnin mənfi və müsbət hissələrini - tvitləri, film rəylərini və ya insanın xal *və* yazılı rəy verdiyi hər şeyi əl ilə toplamaq olardı. Sonra NLP texnikaları fikirlərə və xallara tətbiq oluna bilər, beləliklə nümunələr meydana çıxa bilər (məsələn, müsbət film rəylərində mənfi film rəylərindən daha çox "Oskara layiqdir" ifadəsi olur və ya müsbət restoran rəyləri "iyrənc"dən daha çox "gurme" sözü istifadə olunur.).


> ⚖️ **Nümunə**: Əgər siz siyasətçinin ofisində işləmisinizsə və yeni qanun müzakirə olunurdusa, seçicilər xüsusi yeni qanunu dəstəkləyən və ya əleyhinə olan e-poçtların daxil olduğunu görə bilərdiniz. Deyək ki, sizə e-poçtları oxumaq və onları *lehinə* və *əleyhinə* olmaqla 2 qrupa çeşidləmək tapşırılıb. Əgər çox e-məktublar olsaydı, onların hamısını oxumağa cəhd etmək sizə yorucu gələ bilər. Bir bot sizin üçün onların hamısını oxuyub başa düşsə və hər bir e-poçtun hansı qrupa aid olduğunu sizə bildirsə, gözəl olmazdımı?
>
> Buna nail olmağın bir yolu Maşın Öyrənməsindən istifadə etməkdir. Siz modeli e-poçtların bir hissəsi ilə *leyihə* və digər hissəsi ilə *əleyhinə* olaraq öyrədərdiniz. Model ifadələri və sözləri əleyhinə və lehinə tərəf ilə əlaqələndirməyə meylli olardı, *lakin o, məzmunun heç birini başa düşməyəcək*, yalnız müəyyən söz və nümunələri öz ehtimallarına əsasən *əleyhinə* və ya *leyinə* e-məktublarda olacaqdır. Siz modeli öyrətmək üçün istifadə etmədiyiniz bəzi e-poçt məktubları ilə sınaqdan keçirə və onun sizinlə eyni nəticəyə gəlib-gəlmədiyini görə bilərsiniz. Bundan sonra modelin dəqiqliyindən məmnun ola və oxumadan növbəti e-poçtların hər birini emal edə bilərsiniz.

✅ Bu proses əvvəlki dərslərdə istifadə etdiyimiz proseslərə oxşar gəldimi?

## Tapşırıq - sentimental cümlələr

Hiss -1-dən 1-ə kimi olan *şkala* üzərində ölçülür, -1 ən mənfi hissi, 1 isə ən müsbəti bildirir. Hiss həçminin 0 - 1 arasında obyektivlik (0) və subyektivlik (1) üzrə də ölçülür.

Ceyn Austinin *Qürur və qərəz* əsərinə yenidən baxaq. Mətn [Qutenberq layihəsində](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) də yerləşdirilib. Aşağıdakı nümunədə qısa proqram kitabdan birinci və son cümlələrini hiss analizi edir, hiss şkalası və subtektivlik/obyektivlik qiymətlərini çap edir.

`TextBlob` kitabxanasını (yuxarıda qeyd olunan) `hissi` təyin etmək üçün (öz hiss hesablayıcını yazmağa ehtiyac olmamalıdır) aşağıdakı tapşırıqda istifadə etməlisən.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Aşağıdakı nəticəni görməlisən:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Məşğələ - hiss şkalasını yoxla

Sənin tapşırığın hiss şkalaları istifadə etməklə *Qürur və Qərəz* mütləq sayda mənfidən daha çox müsbət cümlələrin olduğu halı təyin etməkdir. Bu tapşırıq üçün sən şkalanın 1 və ya -1 dəyərlərini müvafiq olaraq mütləq müsbət və ya mənfi kimi saya bilərsən.

**Addımlar:**

1. [Qürur və Qərəzin nüxsəsini](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) Qutenberq layihəsindən .txt faylı kimi yükləyin. Faylın əvvəlində və sonunda metadatanı silin, yalnız orijinal mətni saxla
2. Faylı Python-da açın və məzmunu string kimi çıxar
3. Kitab string dəyərini istifadə edərək TextBlob yarat
4. Kitabın hər cümləsini dövr içində analiz et
   1. Cümlənin müsbət və mənfi mənasına əsasən şkala dəyəri 1 və ya -1 kimi çoxluq içində saxla
5. Sonda bütün müsbət və mənfi cümlələri (ayrı-ayrı) və hər birinin sayını çap et

Nümunə [həll](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb) burdadadır.

✅ Bilik yoxlaması

1. Hiss cümlədə istifadə olunan sözlərə əsaslanır, lakin kod sözləri *anlayırmı*?
2. Sizcə, hiss şkalası dəqiqdirmi, başqa sözlə, xallarla *razısan*?
   1. Xüsusilə aşağıdaki cümlələrin mütləq **müsbət** şkala ilə qiymətləndirilməsinə razısan, yoxsa razı deyilsən?
      * “Nə gözəl atanız var, qızlar!” qapı bağlananda dedi.
      * “Cənab Darsi ilə bağlı imtahanınız bitdi, məncə,” Miss Binqli dedi; "Və dua et, nəticəsi nə olacaq?" “Mən buna tam əmin oldum ki, cənab Darsinin heç bir qüsuru yoxdur”.
      * Bu cür şeylər necə də gözəl baş verir!
      * Dünyada bu cür şeylərə ən böyük nifrətim var.
      * Seyə bilərəm ki, Şarlotta əla menecerdir.
      * “Bu, həqiqətən də ləzzətlidir!“
      * Mən çox xoşbəxtəm!
      * Ponilər haqqında təsəvvürünüz ləzzətlidir.
   2. Növbəti 3 cümlə mütləq müsbət əhval-ruhiyyə ilə qiymətləndirildi, lakin yaxından oxuduqda onlar müsbət cümlələr deyil. Nə üçün hisslərin təhlili onların müsbət cümlələr olduğunu düşündü?
      * “Onun Niderfilddə qalması başa çatanda xoşbəxt olacağam!” Elizabet cavab verdi: “Kaş ki, sənə təsəlli verəcək bir şey deyə biləydim”; “Ancaq bu, mənim gücümdən tamamilə kənardır”.
      * Səni xoşbəxt görə bilsəydim!
      * Bizim əziyyətimiz, əzizim Lizzy, çox böyükdür.
   3. Aşağıdakı cümlələrin mütləq **mənfi** şkalası ilə razısan, yoxsa razı deyilsən?
      - Onun qürurundan hamı iyrənir.
      - "Mən onun yad insanlar arasında necə davrandığını bilmək istərdim." "Onu eşidəcəksən, amma özünü çox qorxunc bir şeyə hazırla."
      - Fasil Elizabetin hissləri üçün qorxunc idi.
      - Dəhşətli olardı!

✅ Jane Austenin hər hansı bir həvəskarı başa düşəcək ki, o, tez-tez kitablarından İngilis Regency cəmiyyətinin daha gülməli aspektlərini tənqid etmək üçün istifadə edir. *Qürur və qərəz* əsərinin baş qəhrəmanı Elizabet Bennett (müəllif kimi) diqqətli sosial müşahidəçidir və onun danışığı tez-tez ağır sözlərlə dolu olur. Hətta cənab Darsi (hekayədəki sevgi marağı) Elizabetin oynaq və zəhlətökən dildən istifadə etdiyini qeyd edir: “Sizinlə uzun müddətli tanışlığımızdan həzz alıram və sizin hərdən özünüzə aid olmayan fikirləri söyləməkdən böyük həzz aldığınızı da bilirəm."

---

## 🚀 Məşğələ

İstifadəçi məlumatlarından digər xüsusiyyətləri çıxarmaqla Marvin-i daha da yaxşılaşdıra bilərsinizmi?

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/?loc=az)

## Təkrarlayın və özünüz öyrənin

Mətndən hissləri çıxarmağın bir çox yolu var. Bu texnikadan istifadə edə biləcək biznes tətbiqlərini düşünün. Bunun necə pisləşə biləcəyini düşünün. [Azure mətn təhlili](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott) kimi korporativ istifadəyə hazır hiss analizi edən sistemlər barədə oxuyun. Yuxarıdakı Qurur və Qərəz əsərindən olan bəzi cümlələri yoxlayın və hissi düzgün təyin edə bildiyini müşahidə edin.


## Tapşırıq

[Poetik lisenziya](assignment.az.md)