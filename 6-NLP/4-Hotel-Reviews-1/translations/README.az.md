# Otel rəyləri ilə duyğu analizi - datanın emalı

Bu bölmədə siz böyük verilənlər toplusunun kəşfiyyat xarakterli məlumat təhlilini aparmaq üçün əvvəlki dərslərdəki üsullardan istifadə edəcəksiniz. Müxtəlif sütunların necə faydalı olduğunu yaxşı başa düşdükdən sonra öyrənəcəksiniz:

- lazımsız sütunları necə çıxarmağı
- mövcud sütunlar əsasında bəzi yeni məlumatları necə hesablamağı
- nəticədə əldə edilən dataseti son problemdə istifadə etmək üçün necə yadda saxlamağı

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/?loc=az)

### Giriş

İndiyə qədər siz mətn tipli datanın ədədi data növlərindən tamamilə fərqli olduğunu öyrəndiniz. Əgər bu data insan tərəfindən yazılmış və ya danışılan mətndirsə, nümunələri və tezlikləri, duyğuları və mənanı tapmaq üçün təhlil edilə bilər. Bu dərs sizə real bir məsələ ilə bağlı və [CC0: İctimai Domen lisenziyası](https://creativecommons.org/publicdomain/zero/1.0/) olan real dataset (**[Avropada 515 min otel rəyi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**) təqdim edir. Bu dataset Booking.com saytındakı açıq mənbələrdən götürülüb və Jiashen Liu tərəfindən yaradılmışdır.

### Hazırlıq

Sizə lazım olacaq:

* Python 3-dən istifadə edərək .ipynb noutbuklarını idarə etmək imkanı
* pandas
* [Cihazınızda quraşdırmalı olduğunuz](https://www.nltk.org/install.html) NLTK
* Kaggle-dan əldə edə biləcəyiniz [Avropada 515 min otel rəyi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe) dataseti. Datanın orijinal halında ölçüsü təxminən 230 MB-dır. Datanı NLP dərsləri ilə bağlı olan `/data` qovluğuna endirin.

## Kəşfiyyat xarakterli məlumat təhlili

Bu məşğələ güman edir ki, siz duyğu analizi və qonaq rəylərindən istifadə edərək bir otel tövsiyə botu yaradırsınız. İstifadə edəcəyiniz datasetə 6 şəhərdə yerləşən 1493 müxtəlif otel haqqında rəylər daxildir.

Python-dan, otel rəyləri üçün olan datasetdən və NLTK-nın duyğu analizindən istifadə edərək siz aşağıdakıları öyrənə bilərsiniz:

* Rəylərdə ən çox istifadə olunan söz və ifadələr hansılardır?
* Oteli təsvir edən rəsmi *teqlər* rəy qiymətləri ilə əlaqələndirilirmi (məsələn, hansısa otel *Solo səyahət edənlər* ilə müqayisədə *Azyaşlı uşaqlı ailələr* tərəfindən daha mənfi qiymətləndirilib, bəlkə də, bu, otelin *Solo səyahət edənlər* üçün daha yaxşı olduğunu göstərir?)
  * NLTK duyğu analizinin qiymətləndirməsi ilə otel rəylərindəki ədədi qiymətləndirmə arasında uyğunluq varmı?

#### Dataset

Gəlin cihazınıza endirdiyiniz və yaddaşda saxladığınız dataseti araşdıraq. Faylı VS Code redaktorunda və ya elə Excel-də açın.

Datasetdəki başlıqlar aşağıdakı kimidir:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Burada onlar yoxlanması daha asan ola biləcək şəkildə qruplaşdırılıb:

##### Oteli təsvir edən sütunlar

* `Hotel_Name`, `Hotel_Address`, `lat` (en dairəsi), `lng` (uzunluq dairəsi)
  * *lat* və *lng* sütunlarını istifadə edərək Python ilə otellərin olduğu yerləri göstərən xəritə çəkə bilərsiniz (mənfi və müsbət rəyli otelləri ayırmaq üçün seçilmiş rənglər də istifadə etmək olar)
  * Hotel_Address sütunu bizim üçün çox da faydalı olmadığından yəqin ki, daha asan çeşidləmə və axtarış üçün bunu ölkə ilə əvəz edəcəyik

**Otelin əsas rəy sütunları**

* `Average_Score`
  * Datasetin yaradıcısına görə bu sütun *Son bir ildəki ən son şərhə əsasən hesablanmış otelin orta qiymətidir*. Bu üsul qiyməti hesablamaq üçün olan ənənəvi üsullardan olmasa da, datanı birbaşa bu şəkildə əldə etdiyimizə görə biz onu hələlik nominal dəyər kimi qəbul edə bilərik.

  ✅ Bu datadakı digər sütunlara əsaslanaraq ortalama qiyməti hesablamaq üçün başqa yol düşünə bilərsinizmi?

* `Total_Number_of_Reviews`
  * Bu otel üçün olan rəylərin ümumi sayı - bu sayın datasetdəki rəylərə aid olub-olmadığını qısa kodlama etmədən müəyyən etmək olmur.
* `Additional_Number_of_Scoring`
  * Bu o deməkdir ki, rəy qiyməti verilib, lakin qonaq tərəfindən müsbət və ya mənfi rəy yazılmayıb.

**Rəy sütunları**

- `Reviewer_Score`
  - Bu, 2,5 (minimum) və 10 (maksimum) dəyərləri arasında ən çoxu 1 onluq kəsr hissəsi olan ədədi dəyərdir.
  - 2,5-in niyə mümkün olan ən aşağı bal olduğu izah edilməyib
- `Negative_Review`
  - Əgər qonaq rəy bildirməyibsə, bu xanada "**No Negative**" dəyəri olacaq
  - Nəzərə alın ki, rəy sahibinin yazdığı müsbət rəy _mənfi rəy_ sütununda ola bilər (məs. "bu otel ilə bağlı pis heç nə yoxdur")
- `Review_Total_Negative_Word_Counts`
  - Daha yüksək sayda mənfi sözlərin olmağı daha aşağı bala işarədir (əgər sentimentallığı yoxlamasaq)
- `Positive_Review`
  - Əgər qonaq rəy bildirməyibsə, bu xanada "**No Positive**" dəyəri olacaq
  - Nəzərə alın ki, rəy sahibinin yazdığı mənfi rəy _müsbət rəy_ sütununda ola bilər (məs. "bu otel ilə bağlı yaxşı heç nə yoxdur")
- `Review_Total_Positive_Word_Counts`
  - Daha yüksək sayda müsbət sözlərin olmağı daha yüksək bala işarədir (əgər sentimentallığı yoxlamasaq)
- `Review_Date` və `days_since_review`
  - Rəylərə yenilik və ya köhnəlik ölçü meyarları tətbiq oluna bilər (köhnə rəylər yeniləri qədər dəqiq olmaya bilər, çünki otel rəhbərliyi dəyişib, təmir işləri aparılıb və ya hovuz əlavə edilib və s.)
- `Tags`
  - Teqlər rəy sahibinin olduğu qonaq növünü (məsələn, solo və ya ailə), qonaq olduğu otağın növünü, qalma müddətini və rəyin necə təqdim olunduğunu təsvir etmək üçün seçə biləcəyi qısa təsvirlərdir.
  - Təəssüf ki, bu teqlərdən istifadə problemlidir, onların faydalılığını müzakirə edən aşağıdakı bölməni nəzərdən keçirin.

**Rəy sahibləri ilə bağlı sütunlar**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Əgər daha çox rəy yazıb yüzlərlə rəyi olan rəy sahiblərinə məxsus olan rəylərin müsbət deyil, mənfi olma ehtimalının daha yüksək olduğunu müəyyən edə bilsəniz, bu, tövsiyyə modelində amillərdən biri ola bilər. Bununla belə, hər hansı bir rəyin sahibi unikal kodla müəyyən edilmir və buna görə də digər rəylər toplusu ilə əlaqələndirilə bilməz. 100 və ya daha çox rəyi olan 30 rəy sahibi var, lakin bunun tövsiyə modelinə necə kömək edə biləcəyini görmək çətindir.
- `Reviewer_Nationality`
  - Bəzi insanlar düşünə bilər ki, müəyyən millətlərin milli meylinə görə müsbət və ya mənfi rəy vermə ehtimalı daha yüksəkdir. Modellərinizdə bu cür subyektiv fikirləri nəzərə alarkən diqqətli olun. Bunlar milli (və bəzən irqi) stereotiplərdir və hər bir rəy sahibi öz təcrübəsinə əsaslanaraq rəy yazan bir fərddir. Ola bilsin ki, bu, onların əvvəlki otel qonaqlamaları, qət etdikləri məsafə və şəxsi temperamentləri kimi bir çox amillərdən asılı olub. Onların milli mənsubiyyətinin rəy qiymətləndirmələrinə istiqamət verdiklərini düşünməyə haqq qazandırmaq çətindir.

##### Nümunələr

| Ortalama Bal | Ümumi rəy sayı | Rəy Sahibinin Qiymətləndirməsi | Mənfi <br />Rəy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Müsbət   Rəy                 | Teqlər                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Bu hazırda otel deyil, tikinti sahəsidir. Uzun bir səfərdən sonra istirahət edərkən və otaqda işləyərkən səhər tezdən və bütün günü qəbuledilməz bina səs-küyü ilə dəhşətə gəldim. İnsanlar bütün günü bitişik otaqlarda çəkiclərlə işləyirdilər. Mən otağı dəyişmək istədim, amma səssiz otaq yox idi. Bütün bunların üzərinə məndən əlavə ödəniş də tutuldu. Növbəti gün uçuşum erkən olduğu üçün oteli gecədən tərk etdim və bunun üçün müvafiq ödəməni aldım. Bir gün sonra otel mənim razılığım olmadan rezervasiya edilmiş qiymətdən artıq ödəniş tutdu. Çox pis yerdir. Burada otaq bron etməklə özünüzə pislik etməyin | Söz yoxdur  Çox pis yerdir Uzaq durun | İşgüzar səyahət Cütlük Standart İkili  Otaq 2 gecə |

Gördüyünüz kimi bu qonaq bu oteldən razı qalmayıb. Otelin yaxşı orta balı (7,8) və haqqında 1945 rəy var, lakin bu qonaq otelə 2,5 qiymət verib və onların qonaqlamasının nə qədər mənfi olduğuna dair 115 söz yazıb. Əgər onlar Positive_Review sütununda heç nə yazmasaydılar, pozitiv heç nə olmadığını təxmin etmək olardı, amma təəssüf ki, xəbərdarlıq üçün 7 söz yazıblar. Əgər sözlərin mənası və ya duyğusu əvəzinə sadəcə sözləri saysaq, rəy sahibinin niyyəti ilə bağlı yanlış istiqamətdə fikirlərə sahib ola bilərdik. Qəribədir ki, onların verdiyi 2,5 bal çaşqınlıq yaradır, çünki əgər o oteldə qalmaq belə pis idisə, niyə ümumiyyətlə, ona xal verirsiniz? Verilənlər toplusunu daha diqqətlə araşdırdıqda, mümkün olan ən aşağı balın 0 deyil, 2,5 olduğunu görəcəksiniz. Mümkün olan ən yüksək bal isə 10-dur.

##### Teqlər

Yuxarıda qeyd edildiyi kimi, ilk baxışdan məlumatları kateqoriyalara ayırmaq üçün `Teqlər`dən istifadə etmək fikri məntiqlidir. Təəssüf ki, bu etiketlər standartlaşdırılmayıb, yəni müəyyən bir oteldə seçimlər *Single room*, *Twin room*, və *Double room*, amma digər bir oteldə eyni otaqlar *Deluxe Single Room*, *Classic Queen Room*, və *Executive King Room* kimi fərqli adlandırılıb. Bunlar eyni şeylər ola bilər, lakin o qədər çox variasiya var ki, seçim belə olur:

1. Bütün şərtləri vahid standarta dəyişdirməyə cəhd etmək, lakin bu çox çətindir, çünki bu halda müxtəlif növlər arasında uyğunluq bəlli deyil. (məs. *Classic single room*, *Single room* seçiminə uyğun gəlir, lakin *Superior Queen Room with Courtyard Garden or City View* seçimini standartlaşdırmaq çox daha çətindir)

2. Biz NLP yanaşması ilə başlaya və *Solo*, *Business Traveller*, ya da *Family with young kids* kimi bəzi ifadələrin istifadə tezliyini ölçə bilər və onları tövsiyyə modelində amil olaraq nəzərə ala bilərik, çünki bunlar bütün otellər üçün uyğundur.

Teqlər adətən (lakin həmişə deyil) *Səyahətin növü*, *Qonaqların növü*, *Otaq növü*, *Gecə sayı* və *Rəyin təqdim olunduğu cihazın növü* kimi kateqoriyalara uyğunlaşdırılan 5-6 vergüllə ayrılmış dəyərlərin siyahısından ibarət sahədir. Bununla belə bəzi rəy sahibləri hər bir sahəni doldurmadığına görə (bəzilərini doldurmaya bilirlər) dəyərlər həmişə eyni ardıcıllıqla olmur.

Nümunə olaraq *Qrupun növü* kateqoriyasını götürək. `Teqlər` sütununda bu kateqoriya üçün 1025 unikal dəyər var və təəssüf ki, onlardan yalnız bəziləri qrup növünə istinad edir (bəziləri otaq növüdür və s.). Qrup növü olaraq yalnız ailə qeyd edənləri filtrləsəniz, nəticələrdə çoxlu *Ailə otağı* tipli otaq növü dəyərləri də olacaq. Əgər ailənin növünü də filtrə daxil etsəniz, bu zaman nəticə daha yaxşı olacaq, çünki 515 000 nəticədən 80 000-dən çoxunda "Gənc uşaqlı ailə" və ya "Yaşlı uşaqlı ailə" ifadəsi var.

Bu o deməkdir ki, teqlər sütunu bizim üçün tamamilə yararsız deyil, lakin onu faydalı etmək üçün bir az iş lazımdır.

##### Otelin ortalama balı

Dataset ilə bağlı bir sıra qəribəliklər və ya uyğunsuzluqlar var ki, mən onları anlaya bilmirəm, lakin modellərinizi qurarkən onlardan xəbərdar olmanız üçün burada təsvir edilmişdir. Əgər başa düşsəniz, zəhmət olmasa müzakirə bölməsində bizə bildirin!

Datasetdə ortalama bal və rəylərin sayı ilə bağlı aşağıdakı sütunlar var:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score

Bu datasetdə ən çox rəyi olan tək otel 515.000 rəydən 4789 rəylə *Britannia International Hotel Canary Wharf*-dır. Amma bu otel üçün `Total_Number_of_Reviews` dəyərinə baxsaq görərik ki, bu 9086-dır. Güman edə bilərsiniz ki, rəyi olmayan çoxlu qiymətləndirmə var, ona görə də bəlkə də `Additional_Number_of_Scoring` sütunundakı dəyəri bu dəyərə əlavə etmək lazımdır. Bu dəyər 2682-dir və onu 4789-a əlavə etməklə 7,471 əldə edirik ki, bu da hələ də `Total_Number_of_Reviews` sütun dəyərindən 1615 azdır.

Əgər `Average_Score` sütun dəyərlərinə baxsaq, bunun datasetdəki balların ortalaması olduğunu güman edə bilərsiniz, lakin Kaggle-da verilən təsvir "*Otelin son bir ildəki ən son şərhə əsasən hesablanmış orta balı*" şəklindədir. Bu, o qədər də faydalı görünmür, lakin biz datasetdəki ballara əsaslanaraq öz ortalamamızı hesablaya bilərik. Nümunə olaraq eyni oteldən istifadə edərək orta otel balı 7,1 kimi verilir, lakin hesablanmış bal (datasetdə rəy sahibinin ortalama qiymətləndirməsi) 6,8-dir. Bu yaxındır, lakin eyni dəyər deyil və biz yalnız təxmin edə bilərik ki, `Additional_Number_of_Scoring` sütununda olan bal dəyərləri ortalama göstəricisini 7.1-ə yüksəldib. Təəssüf ki, bu iddianı sınamaq və ya sübut etmək üçün heç bir yol olmadığından `Average_Score`, `Additional_Number_of_Scoring` və `Total_Number_of_Reviews` sütun dəyərləri bizdə olmayan dataya əsaslandıqda və ya onlara istinad etdikdə onlardan istifadə etmək və ya onların doğruluğuna inanmaq çətindir.

Bütün bunların üzərinə, ən çox rəy alan ikinci otelin hesablanmış orta balı 8,12 və datasetdəki `Average_Score` sütun dəyəri isə 8.1-dir. Bu düzgün hesab təsadüfdür yoxsa ilk otel datası doğru deyil?

Bu otellərin istisna olmaları ehtimalı və bəlkə də dəyərlərin əksəriyyətinin üst-üstə düşməsi (niyəsə bəziləri üst-üstə düşmür) ehtimalı üzərinə datasetdəki dəyərlərin düzgün istifadə edilməsini (və ya edilməməsini) araşdırmaq üçün qısa proqram yazacağıq.

> 🚨 Qeyd
>
> Bu verilənlər toplusu ilə işləyərkən siz mətni özünüz oxumadan və ya təhlil etmədən mətndən nəyisə hesablayan kod yazacaqsınız. NLP-nin mahiyyəti də elə bundan ibarətdir, insandan asılı olmadan məna və ya duyğuları analiz və şərh edir. Bununla belə, bəzi mənfi rəyləri oxumağınız mümkündür. Mən oxumamağınızın tərəfdarıyam, çünki buna məcbur deyilsiniz. Onlardan bəziləri mənasız və ya aidiyyatı olmayan mənfi otel rəyləridir, məsələn, "Hava əla deyildi", hansı ki, oteldən və ya həqiqətən də hər hansı birindən asılı olmadan baş verən bir şeydir. Ancaq bəzi rəylərin qaranlıq tərəfi də var. Bəzən mənfi rəylər irqçi, cinsiyyətçi və ya yaş ilə bağlı ayrı-seçkilik etməyə yönəlib. Bu təəssüf doğurur, lakin ictimai vebsaytdan əldə edilmiş verilənlər bazasında gözləniləndir. Bəzi rəy sahibləri xoşagəlməz, narahat və ya əsəbi hesab edə biləcəyiniz rəylər yazırlar. Kodunuzun duyğu analizi etməsinə icazə vermək, onları özünüz oxuyub üzülməkdən daha yaxşıdır. Bununla belə, bu tip rəylər azlıq təşkil edir, lakin həmişə mövcud olduğunu bilməkdə fayda var.

## Məşğələ -  Datanın təhlili
### Datanı yükləyin

Datanı vizual olaraq bu qədər yoxlamaq kifayətdir, artıq biraz kod yazacaq və bəzi cavablar alacaqsınız! Bu hissə pandas kitabxanasını istifadə edir. İlk tapşırığınız CSV məlumatlarını yükləyə və oxuya bildiyinizə əmin olmaqdır. Pandas kitabxanasının sürətli CSV yükləyicisi var və əvvəlki dərslərdə olduğu kimi nəticə datafreymə yerləşdiriləcək. Yüklədiyimiz CSV-də yarım milyondan çox sətir, lakin cəmi 17 sütun var. Pandas sizə datafreym ilə qarşılıqlı əməliyyatlar aparmaq üçün çoxlu faydalı metodlar, o cümlədən hər sətirdə əməliyyatlar yerinə yetirmək imkanı verir.

Bu hissədən etibarən bu dərs koddan fraqmentlər, kodun ümumi analizi və nəticələrin nə mənaya gəldiyi üzərinə aparılan müzəkirələrdən ibarət olacaq. Kodunuz üçün _notebook.ipynb_ noutbukunu istifadə edin.

İstifadə edəcəyiniz data faylını yükləməklə başlayaq:

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

Artıq data yükləndiyinə görə biz onun üzərində bəzi əməliyyatları yerinə yetirə bilərik. Növbəti hissə üçün bu kodu proqramınızın yuxarı hissəsində saxlayın.

## Datanı təhlil edin

Artıq data *təmizdir*, bu isə o deməkdir ki, data onunla işlənilməyə hazırdır və yalnız ingilis simvollarını gözləyən alqoritmləri poza biləcək başqa simvollar yoxdur.

✅ NLP üsullarını tətbiq etməzdən əvvəl onu formatlaşdırmaq üçün bəzi ilkin emal tələb edən data ilə işləməli ola bilərdiniz, lakin bu dəfə buna ehtiyac yoxdur. Əgər olsaydı, ingiliscədə istifadə olunmayan simvolların öhdəsindən necə gələrdiniz?

Data yükləndikdən sonra onu yazdığınız kodu istifadə edərək araşdıra biləcəyinizdən əmin olun. İlk baxışdan `Negative_Review` və `Positive_Review` sütunlarına diqqəti cəmləmək daha asan görünür. Həmin sütunlar sizin NLP alqoritmlərinizin rahatlıqla işləyə biləcəyi, təbii dildə yazılmış mətnlərdən ibarətdir. Lakin gözləyin! NLP və duyğu analizinə başlamazdan əvvəl, ilk addım kimi, aşağıdakı kodu istifadə edərək datasetdəki dəyərlər ilə pandas kitabxanasını istifadə edərək hesabladığınız dəyərlərin uyğunlaşdığına əmin olun.

## Datafreym əməliyyatları

Bu dərsdəki ilk tapşırıq datafreymi (dəyişdirmədən) yoxlayan kod yazaraq aşağıdakı iddiaların düzgün olub-olmadığını yoxlamaqdır.

> Bir çox proqramlaşdırma tapşırıqları kimi bu tapşırığı da başa çatdırmağın bir neçə yolu var, lakin məsləhət odur ki, bunu edə biləcəyiniz ən sadə, ən asan şəkildə edəsiniz, çünki gələcəkdə bu koda qayıtdığınız zaman onu başa düşmək daha asan olacaq. Burda istədiyinizə bir çox hallarda səmərəli şəkildə nail olmağınıza kömək ola biləcək datafreym üçün olan API-lar var.

Aşağıdakı suallara kodlaşdırma tapşırıqları kimi baxın və həllinə baxmadan onlara cavab verməyə çalışın.

1. İndicə yüklədiyiniz datafreymin *formasını* çap edin (burada forma deyəndə sətir və sütunların sayı nəzərdə tutulur)
2. Müxtəlif milliyətlərdən olan rəy sahibləri üçün tezliyi hesablayın:
   1. `Reviewer_Nationality` sütunu üçün neçə fərqli dəyər var və hansılardır?
   2. Datasetdə ən çox hansı milliyyətdən olan rəy sahibi var (ölkəni və rəylərin sayını çap edin)?
   3. Növbəti 10 milliyət hansıdır və tezlikləri nə qədərdir?
3. Bu 10 milliyyətdən olan rəy sahiblərinin ən çox qiymətləndirdiyi otel hansıdır?
4. Datasetdə hər bir otel üçün neçə rəy var (otellərin rəy sıxlığı necədir)?
5. Baxmayaraq ki, datasetdə hər bir otel üçün `Average_Score` sütunu var, siz də ortalama balı hesablaya bilərsiniz (bunun üçün hər bir otelin datasetdəki rəy qiymətləndirmələrinin ortalamasını hesablamaq lazımdır). Datafreymə `Calc_Average_Score` başlıqlı və hesablanmış ortalamanı göstərən yeni sütun əlavə edin.
6. Hansısa otelin eyni `Average_Score` və `Calc_Average_Score` (kəsr hissəsi onluqlara yuvarlaqlaşdırılmış) dəyəri varmı?
   1. Giriş olaraq sətir götürən və dəyərləri müqayisə edən bir Python funksiyası yazmağa çalışın və dəyərlər eyni olmadıqda mesaj çap edin. Daha sonra isə `.apply()` funksiyasını istifadə edərək yazdığınız funksiyanı digər sətirlərə də tətbiq edin.
7. `Negative_Review` sütununda neçə sətrin "No Negative" dəyəri olduğunu hesablayın və nəticəni çap edin.
8. `Positive_Review` sütununda neçə sətrin "No Positive" dəyəri olduğunu hesablayın və nəticəni çap edin.
9. Neçə sətrin `Positive_Review` sütun və "No Positive" dəyəri, **və** neçəsinin `Negative_Review` sütun və "No Negative" dəyəri olduğunu hesablayın və nəticəni çap edin.

### Kodlaşdırma tapşırıqlarına cavablar

1. İndicə yüklədiyiniz datafreymin *formasını* çap edin (burada forma deyəndə sətir və sütunların sayı nəzərdə tutulur)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Müxtəlif milliyətlərdən olan rəy sahibləri üçün tezliyi hesablayın:

   1. `Reviewer_Nationality` sütunu üçün neçə fərqli dəyər var və hansılardır?
   2. Datasetdə ən çox hansı milliyyətdən olan rəy sahibi var (ölkəni və rəylərin sayını çap edin)?

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

   3. Növbəti 10 milliyət hansıdır və tezlikləri nə qədərdir?

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

3. Bu 10 milliyyətdən olan rəy sahiblərinin ən çox qiymətləndirdiyi otel hansıdır?

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

4. Datasetdə hər bir otel üçün neçə rəy var (otellərin rəy sıxlığı necədir)?

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

   `Total_Number_of_Reviews` sütunundakı dəyərlərin *datasetdə tapılan* nəticələr ilə uyğunlaşmadığını görə bilərsiniz. Bizə məlum deyil ki, bu dəyər otelə aid olan bütün rəylərin sayını göstərir (lakin bütün rəylər saytdan əldə edilməmişdir), yoxsa başqa hesablamanın nəticəsidir. Bu qeyri-müəyyənliyə görə `Total_Number_of_Reviews` sütunu modeldə istifadə edilməmişdir.

5. Baxmayaraq ki datasetdə hər bir otel üçün `Average_Score` sütunu var, siz də ortalama balı hesablaya bilərsiniz (bunun üçün hər bir otelin datasetdəki rəy qiymətləndirmələrinin ortalamasını hesablamaq lazımdır). Datafreymə `Calc_Average_Score` başlıqlı və hesablanmış ortalamanı göstərən yeni sütun əlavə edin. `Hotel_Name`, `Average_Score` və `Calc_Average_Score` sütunlarını çap edin.

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

   Sizə `Average_Score` sütunundakı dəyər və bu dəyərin bəzən niyə hesablanmış ortalama baldan fərqli olduğu maraqlı ola bilər. Niyə bəzi dəyərlərin üst-üstə düşüb, digərlərinin isə fərqləndiyini bilmədiyimiz üçün ortalama balını özümüzün hesablamalı olduğumuz rəy qiymətləndirmələrini istifadə etmək ən doğrusudur. Bununla belə, fərqliliklər adətən çox kiçikdir və aşağıda datasetdəki ortalama rəy dəyəri ilə hesablanmış ortalama rəy dəyəri ən çox fərqlənən otellər göstərilib:

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

   Sadəcə bir oteldə bu fərq 1-dən böyük olduğundan fərqi nəzərə almayıb hesablanmış ortalama qiymətləndirməni istifadə edə bilərik.

6. `Negative_Review` sütununda neçə sətrin "No Negative" dəyəri olduğunu hesablayın və nəticəni çap edin.

7. `Positive_Review` sütununda neçə sətrin "No Positive" dəyəri olduğunu hesablayın və nəticəni çap edin.

8. Neçə sətrin `Positive_Review` sütun və "No Positive" dəyəri, **və** neçəsinin `Negative_Review` sütun və "No Negative" dəyəri olduğunu hesablayın və nəticəni çap edin.

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

## Başqa üsul

Digər bir üsul Lambdas istifadə etmədən sözügedən dəyərləri saymaq və cəmləməklə sətirləri saymaqdır:

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

   127 sətr üçün `Negative_Review` və `Positive_Review` sütunlarında müvafiq olaraq həm "No Negative", həm də "No Positive" dəyərlərinin olduğunu görmüş ola bilərsiniz. Bu o deməkdir ki, qonaq oteli ədədi olaraq qiymətləndirib, lakin nə mənfi nə də müsbət rəy yazıb. Xoşbəxtlikdən belə sətirlərin sayı azdır (127/515738, hansı ki 0.02% təşkil edir), ona görə də böyük ehtimalla bu sətirlər bizim modeli hansısa yanlış istiqamətdə yönləndirməyəcək. Bununla belə, rəylərin datasetində rəy olmayan sətirlərin olması təbii olaraq gözlənilməz olduğundan, datanı təhlil etmək həmişə vacibdir.

Datasetin təhlilini bitirdiyinizə görə artıq növbəti dərsdə datanı filtrləyəcək və data üzərində duyğu analizi aparacaqsınız.

---
## 🚀 Məşğələ

Bu dərs, əvvəlki dərslərdə də gördüyümüz kimi data üzərində əməliyyatlar etməzdən əvvəl onun özünü və çatışmazlıqlarını başa düşməyin nə qədər vacib olduğunu nümayiş etdirir. Xüsusilə də mətnə əsaslanan data diqqətlə araşdırılır. Müxtəlif mətn ağırlıqlı datasetləri araşdırın və modeldə qərəzli və ya başqa istiqamətə yönəldən duyğu yarada biləcək hissələri aşkar edə bilib-bilmədiyinizə baxın.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/?loc=az)

## Təkrarlayın və özünüz öyrənin

Nitq və mətn ağırlıqlı modellər yaradarkən müxtəlif alətlərlə tanış olmaq üçün [NLP üzrə olan bu təlim toplusunu](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) sınayın.

## Tapşırıq

[NLTK](assignment.az.md)
