# Məsuliyyətli AI ilə maşın öyrənməsi həlləri tapmaq

![Eskizdə Maşın öyrənməsində məsuliyyətli AI xülasəsi](../../../sketchnotes/ml-fairness.png)
> [Tomomi Imura](https://www.twitter.com/girlie_mac) tərəfindən eskiz

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/?loc=az)

## Giriş

Bu kurrikulumda siz maşın öyrənməsinin gündəlik həyatımıza necə təsir edə biləcəyini və etdiyini kəşf etməyə başlayacaqsınız. Hətta indi də sistemlər və modellər səhiyyə diaqnozları, kreditlərin təsdiqlənməsi və ya dələduzluğun aşkarlanması kimi gündəlik qərar qəbuletmə işlərində iştirak edirlər. Buna görə də, etibarlı nəticələr əldə etmək üçün bu modellərin yaxşı işləməsi vacibdir. Hər hansı bir proqram tətbiqi kimi, AI sistemləri də gözləntilərə çatmayacaq və ya arzuolunmaz nəticə ilə üzləşəcək. Buna görə də AI modelinin davranışını başa düşmək və izah etmək vacibdir.

Təsəvvür edin ki, bu modelləri yaratmaq üçün istifadə etdiyiniz data irq, cins, siyasi görüş, din kimi müəyyən demoqrafik göstəricilərə malik olmadıqda və ya qeyri-mütənasib şəkildə belə demoqrafik göstəriciləri təmsil edərsə nələr baş verə bilər. Modelin çıxışı bəzi demoqrafik göstəricilərə üstünlük vermək üçün şərh edilməsinin nəticəsi nə olacaq? Tətbiq üçün nəticəsi nə olacaq? Bundan əlavə, modelin mənfi nəticəsi olduqda və insanlar üçün zərərli olduqda nə baş verəcək? AI sistemlərinin davranışına görə kim cavabdehdir? Bunlar bu kurrikulumda araşdıracağımız bəzi suallardır.

Bu dərsdə siz:

- Maşın öyrənməsində ədalətin əhəmiyyəti və ədalətlə əlaqəli zərərlər haqqında məlumatlılığınızı artıracaqsınız
- Etibarlılıq və təhlükəsizliyi təmin etmək üçün kənar göstəriciləri və qeyri-adi ssenariləri araşdırmaq təcrübəsi ilə tanış olacaqsınız
- İnklüziv sistemlərin dizayn edilməsi ilə hər kəsin səlahiyyətləndirilməsi ehtiyacı haqqında anlayış əldə edəcəksiniz
- Məlumatların və insanların məxfiliyinin və təhlükəsizliyinin qorumasının nə qədər vacib olduğunu araşdıracaqsınız
- Süni intellekt modellərinin davranışını izah etmək üçün şüşə qutu yanaşmasının vacibliyini görəcəksiniz
- Süni intellekt sistemlərinə inam yaratmaq üçün məsuliyyətliliyin necə vacib olduğunu nəzərə alacaqsınız

## Tələb olunanlar

İlkin olaraq, zəhmət olmasa, "Məsuliyyətli AI Prinsipləri" Təlim Yoluna başlayın və mövzu ilə bağlı aşağıdakı videoya baxın:

Bu [Təlim Yolunu](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) izləməklə Məhsul AI haqqında daha çox məlumat əldə edin

[![Microsoft-un Məsul AI yanaşması](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft-un Məsul AI yanaşması")

> 🎥 Video üçün yuxarıdakı şəklə klikləyin: Microsoft-un Məsul AI yanaşması

## Ədalətlilik

Süni intellekt sistemləri hər kəsə ədalətli davranmalı və oxşar insan qruplarına müxtəlif yollarla təsir etməməlidir. Məsələn, süni intellekt sistemləri tibbi müalicə, kredit müraciətləri və ya işə qəbulla bağlı təlimat verdikdə, oxşar simptomları, maliyyə vəziyyəti və ya peşəkar keyfiyyətləri olan hər kəsə eyni tövsiyələri verməlidir. İnsan olaraq hər birimiz qərarlarımıza və hərəkətlərimizə təsir edən irsi qərəzlər daşıyırıq. Bu qərəzlər AI sistemlərini öyrətmək üçün istifadə etdiyimiz məlumatlarda aydın görünə bilər. Belə manipulyasiya bəzən istəmədən baş verə bilər. Məlumatlarda qərəzliliyi nə vaxt tətbiq etdiyinizi şüurlu şəkildə bilmək çox vaxt çətindir.

**“Haqsızlıq”** irq, cins, yaş və ya əlillik statusu kimi xüsusiyyətləti baxımından bir qrup insan üçün mənfi təsirləri və ya “zərərləri” əhatə edir. Ədalətlə əlaqəli əsas zərərlər aşağıdakı kimi təsnif edilə bilər:

- **Ayrı-seçkilik**, məsələn, cins və ya etnik mənsubiyyət digərindən üstündürsə.

- **Xidmət keyfiyyəti**. Əgər məlumatı konkret bir ssenari üçün öyrədirsinizsə, lakin reallıq daha mürəkkəbdirsə, bu, xidmətin keyfiyyətsiz olmasına gətirib çıxarır. Məsələn, qara dərili insanları hiss edə bilməyən əl sabunu dispenseri. [İstinad](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)

- **Təhqir**. Bir şeyi və ya kimisə haqsız yerə tənqid etmək, etiketləmək. Məsələn, bir şəkil etiketləmə texnologiyası qara dərili insanların şəkillərini qorilla kimi yanlış etiketlədi.

- **Həddindən artıq və ya az təmsil olunma**. İdeya ondan ibarətdir ki, müəyyən bir qrup müəyyən bir peşədə kifayət qədər təmsil olunmur və zərər verən hər hansı bir xidmət və ya funksiya bunu təbliğ etməyə davam edir.

- **Stereotipləşdirmə**. Müəyyən bir qrupun ön yarğılı fikirlərlə əlaqələndirilməsi. Məsələn, ingilis və türk dilləri arasında dil tərcümə sistemində cins ilə stereotipik əlaqəsi olan sözlərə görə səhvlər ola bilər.

![translation to Turkish](../images/gender-bias-translate-en-tr.png)
> türk dilinə tərcümə

![translation back to English](../images/gender-bias-translate-tr-en.png)
> ingilis dilinə geri tərcümə

Süni intellekt sistemlərinin dizayn edilməsi və test olunması zamanı biz süni intellektin ədalətli olmasını və qərəzli və ya ayrı-seçkilik xarakterli qərarlar qəbul etmək üçün proqramlaşdırılmamasını təmin etməliyik, hansı ki insanların da qəbul etməsi qadağandır. Süni intellekt və maşın öyrənməsində ədalətin təmin edilməsi mürəkkəb sosial texniki problem olaraq qalır.

### Etibarlılıq və təhlükəsizlik

Güvən yaratmaq üçün AI sistemləri həm normal, həm də gözlənilməz şəraitdə etibarlı, təhlükəsiz və düzgün olmalıdır. Süni intellekt sistemlərinin müxtəlif situasiyalarda necə davranacağını bilmək vacibdir, xüsusən də onlar normal şəraitdən kənara çıxdıqda. Süni intellekt həllərini qurarkən süni intellektin qarşılaşacağı müxtəlif vəziyyətlərin necə idarə olunacağına böyük diqqət yetirilməlidir. Məsələn, özünü idarə edən avtomobil insanların təhlükəsizliyini əsas prioritet kimi qoymalıdır. Nəticədə, avtomobili idarə edən süni intellekt avtomobilin qarşılaşa biləcəyi bütün mümkün ssenariləri nəzərə almalıdır, məsələn, gecə, tufan və ya çovğun, küçədə qaçan uşaqlar, ev heyvanları, yol tikintiləri və s. Bir AI sisteminin nə qədər yaxşı olduğu bir sıra fərqli şəraitdə nə dərəcədə etibarlı və təhlükəsiz işlədiyindən asılıdır, bu, sistem dizaynı və testi zamanı data mühəndisi və ya AI proqramçısı tərəfindən nəzərə alınan proqnoz səviyyəsini göstərir.

> [🎥 Video üçün bura klikləyin: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Daxil edilmə

AI sistemləri hər kəslə əlaqədə olacaq və fayda verəcək şəkildə formalaşdırılmalıdır. AI sistemlərinin dizayn və icra mərhələrində data mühəndisləri və proqramçılar sistemin istəmədən kimlərisə xaric etməsi ehtimalarını təyin etməyə və qarşısını almağa çalışırlar. Misal üçün, dünyada müxtəlif əngəlləri olan 1 milyard insan var. AI inkişafı nəticəsində onlar geniş miqyasda məlumatlara və fürsətlərə gündəlik həyatlarında asanlıqla çata bilirlər. Əngəllər barədə öncədən düşünmək AI məhsullarını hər kəs tərəfindən faydalana biləcəyi daha yaxşı təcrübəni yaratmağa fürsətlər yaradır.

> [🎥 Video üçün bura klikləyin: AI-da daxil edilmə](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Təhlükəsizlik və məxfilik

AI sistemləri təhlükəsiz olmalı və insanlarım məxfiliyinə hörmət etməlidir. İnsanlar onların məxfiliyini, məlumatlarını və ya həyatlarını risk altında qoyan sistemlərə az güvənirlər. Maşın öyrənməsi modellərini sazladığımız zaman ən yaxşı nəticəni verən datalara etibar edirik. Bunu etdiyimiz zaman datanın mənbəsini və düzgün üsulla əldə olduğunu da nəzərə almalıyıq. Misal üçün, bu məlumat istifadəçi tərəfindən təqdim olunub və ya ictimai olaraq əlçatandırmı? Data ilə işləyərkən nəzərə alınmalı növbəti məsələ, AI sistemlərinin məxfi məlumatlarını qoruyacağı və hücumlara davamlı olacağı şəkildə hazırlanması vacibdir. AI istifadəsi artıqca vacib şəxsi və biznes məlumatlarının məxfiliyinin qorunması daha vacib və qəliz hala gəlir. Məxfilik və data təhlükəsizliyi problemləri AI üçün daha çox diqqət tələb edir, çünki bu məlumatlar AI sistemlərinin insanlar barədə dəqiq təxminlər və qərarlar verməsi üçün lazımdır.

> [🎥 Video üçün bura klikləyin: AI-da təhlükəsizlik](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Cari sənayedə Məxfilik və təhlükəsizlik sahəsində əhəmiyyətli irəliləyişlər etmişik, GDPR (Ümumi Məlumatların Qorunması Qaydası) kimi qaydalarla diqqəti daha da artırmışıq
- Buna baxmayaraq biz AI sistemlərində effektiv fərdiləşdirilmə üçün şəxsi məlumatlara olan ehtiyac ilə təhlükəsizlik arasında olan gərginliyi başa düşməliyik.
- Kompüterlərin internetə qoşulması ilə təhlükəsizlik problemlərinin yaranmasındakı böyük sıçrayışı indi AI ilə əlaqəli sistemlərində görürük.
- Eyni zamanda biz AI-ın təhlükəsizlik tərəfdən inkişaf etdiyini də görürük. Misal olaraq, bugünkü müasir anti-virus proqramları AI sistemlərinin iştirakı ilə həyata keçirir.
- Bizim Data elmi proseslərinin ən son məxfilik və təhlükəsizlik təcrübələrini özündə birləşdiyinə əmin olmağımız lazımdır.

### Şəffaflıq

AI sistemləri başa düşülən olmalıdır. Şəffaflığın ən vacib hissəsi AI sistemlərinin davranışını və komponentlərini izah etməkdir. AI sistemlərinin anlaşıqlığını artırmaqla biz mümkün performans problemlərini, təhlükəsizlik və məxfilik məsələlərini, qərəzləri, çıxdaş edilmə hallarını və ya istənilməyən nəticələri daha tez tapa bilərik. Biz həmçinin inanırıq ki, insanlar AI sistemlərini nə zaman, nə üçün və necə istifadə edəcəklərini seçməkləri barədə səmimi, məsuliyyətli olmalıdırlar. Əlavə olaraq, işlətdikləri sistemin limitləri barədə də məlumatlı olmalıdırlar. Misal üçün, əgər bank AI sistemini müştərilərə verəcək kredit təklifini hazırlamaqda köməkçi kimi istifadə edirsə, bu sistemin verdiyi təklifləri ən çox hansı parametrlərin təsir etdiyini araşdırması vacibdir. Hökümətlər AI-ın sənayelərdə istifadəsinə qaydalar tətbiq etməyə başlayır, belə ki data mühəndisləri və şirkətləri AI sisteminin verdiyi qərarların (xüsusilə arzuolunmaz nəticələrin) tənzimləyici tələblərə cavab verdiyini izah edə bilməlidirlər.

> [🎥 Video üçün bura klikləyin: AI-da şəffaflıq](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- AI sistemləri qəliz olduğu üçün onun işləmə qaydasını və verdiyi qərarları anlamaq çətindir.
- Bu anlaşılmada əksiklik həmin sistemlərin idarə edilməsinə, istifadə edilməsinə və sənədləşməsinə təsir edir.
- Bu anlaşılmadakı əksiklik ən vacib olaraq sistemin səbəb olduğunu nəticəyə gətirən qərarlara təsir edir.

### Məsuliyyət

AI tərtib edən və işə salan şəxslər sistemin necə işlədiyi ilə bağlı məsuliyyət daşımalıdırlar. Üz tanınması kimi həssas texnologiyaların istifadəsində məsuliyyət ehtiyacı daha əhəmiyyətlidir. Son zamanlar, itmiş uşaqları tapmaq kimi istifadəyə yararlı üz tanıma texnologiyalarına tələb hüquq təşkilatlarından tərəfindən kəskin artmışdır. Lakin bu texnologiyalar dövlət tərəfindən istifadə olunaraq vətəndaşları fundamental azadlıq risklərinə gətirib çıxara bilər. Misal üçün onlar seçilmiş fərdləri daim müşahidə edə bilərlər. Buna görə də data mühəndisləri və təşkilatlar AI sistemlərinin fərdlərə və cəmiyyətə necə təsir etməsi ilə bağlı məsuliyyətli olmalıdırlar.

[![Tanınmış AI tədqiqatçısı üz tanınması tətbiqi ilə ilə kütləvi izləmə barədə xəbərdarlıq edir edir](../images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft-un məsuliyyətli AI-a yanaşması")

> 🎥 Video üçün yuxarıdakı şəkilə klikləyin: Üz tanınması tətbiqi ilə ilə kütləvi izləmə barədə xəbərdarlıq

AI-ı cəmiyyətimizə gətirmiş ilk nəsil olaraq bizə ünvanlanmış ən böyük suallardan biri, kompüterlərin insanlara məlusiyyətli qalacağına və kompüterləri tərtib edən insanların digər hər kəsə məsuliyyətli qalacağına necə əmin ola bilərik.

## Təsirin qiymətləndirilməsi

Maşın öyrənmə modelini öyrətməzdən əvvəl AI sisteminin məqsədini anlamaq üçün mümkün ola biləcək təsirləri qiymətləndirməyimiz vacibdir. Sistemin istifadəsində məqsəd nədir, harada tətbiq olunacaq və bununla kim işləyəcək. Bu məqamlar mümkün ola biləcək riskləri və gözlənilən nəticələri nəzərə almaq üçün sistemi yoxlayan və ya test edən şəxslər üçün faydalı olacaqdır.

Aşağıda qeyd olunanlar təsirin qiymətləndirilməsi zamanı nəzərə alınacaq sahələrdir:

* **Fərdlərə mənfi təsir**. Tələbləri və məhdudiyyətləri, nəzərdə tutulmayan istifadə yeri və sistemin performans limitlərini bilməklə bu sistemin başqa fərdlərə hansısa yolla zərər vurmayacağına əmin olmalısınız.
* **Məlumat tələbləri**. Yoxlayan şəxslər sistemin dataları necə və harada istifadə edəcəyini öyrənməklə məlumat saxlanmasındakı tələblərdə nələrə diqqət etməli olacağını biləcəklər (misal üçün GDPR və ya HIPPA data qaydaları). Əlavə olaraq, datanın mənbəyinin və miqdarının öyrənmə üçün kifayət edəcəyi də yoxlayın.
* **Təsirin xülasəsi**. Sistemin istifadəsindən yarana biləcək bütün mümkün təhlükələri siyahı formasında topla. ML prosesi boyunca təyin olunmuş problemləri necə həll edəcəyini nəzərdən keçir.
* **Uyğun məqsədlər** 6 əsas prinsipin hər birinin məqsədinə çatıla bildiyini yoxla və mümkün boşluqları düşün.


## Məsul AI-ı izləmək

Proqram təminatlarının izlənildiyi kimi AI sistemlərində də problemlərin tapılması və həlli üçün izləmə mütləqdir. Modelin gözlənildiyi kimi və ya məsuliyyətlə işləməməsinə təsir edən bir çox amil var. Ənənəvi model performans göstəricilərinin əksəriyyəti modelin performansının kəmiyyət aqreqatlarıdır və modelin məsul AI prinsiplərini necə pozduğunu təhlil etmək üçün kifayət deyil. Bundan əlavə, maşın öyrənmə modelləri qara qutudur və onun nəticəyə necə gəldiyini başa düşmək, səhvlərini izah etmək qəlizdir. Bu kursun davamında Məsul AI panelini necə qura və sistemi izləyə biləcəyimizi öyrənəcəyik. İdarə paneli data tədqiqatçıları və AI proqramçıları üçün aşağıdakıları yerinə yetirmək üçün vahid bir alət təqdim edir:

* **Xəta analizi**. Sistemin ədalətliliyinə və ya etibarlılığına təsir edə biləcək modelin statistik xəta paylanmasını müəyyən etmək.
* **Modelin təsviri**. Datasetlər arasında modelin performansında uyğunsuzluğun harada olduğunu aşkar etmək.
* **Data analizi**. Məlumatların paylanmasını başa düşmək və məlumatlarda ədalətlilik, əhatəlilik və etibarlılıq problemlərinə səbəb ola biləcək hər hansı potensial qərəzliyi müəyyən etmək.
* **Model tətbiq qabiliyyəti**. Modelin proqnozlarına nəyin təsir etdiyini başa düşmək. Bu modelin davranışını izah etməyə kömək edir, şəffaflıq və cavabdehlik üçün vacibdir.

## 🚀 Məşğələ

Zərərin baş verməsinin qabağını almaq üçün ilk növbədə biz:

- sistem üzərində işləyən insanların müxtəlif təcrübə və istiqamətlərdən gəldiyinə əmin olaq
- cəmiyyətimizin fərqliliyini özündə əks etdirən data toplusunu yığmağa sərmayə qoyaq
- problem olduqda cavabdeh AI-ı tapmağa və düzəltməyə qadir olan daha yaxşı maşın öyrənməsi metodları tapaq

Modelin qurulmasında və istifadəsində etibarsızlığın aşkar olduğu real həyat ssenariləri haqqında düşünün. Başqa nələri nəzərə almalıyıq?

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/?loc=az)

## Təkrarlayın və özünüz öyrənin

Bu dərsdə siz maşın öyrənməsində ədalət və ədalətsizlik anlayışlarının bəzi əsaslarını öyrəndiniz.

Mövzuları daha dərindən öyrənmək üçün bu seminara baxın:

- Məsuliyyətli AI axtarışında: Besmira Nushi, Mehrnoosh Sameki və Amit Sharma tərəfindən praktikaya prinsiplərin gətirilməsi

[![Məsuliyyətli AI alətləri: Məsul AI yaratmaq üçün açıq mənbəli çərçivə](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI (Məsuliyyətli AI) Toolbox: Məsuliyyətli AI yaratmaq üçün açıq mənbəli çərçivə")

> 🎥 Videoya baxmaq üçün yuxarıdakı şəkilə klikləyin: RAI (Məsuliyyətli AI) Toolbox: Besmira Nushi, Mehrnoosh Sameki və Amit Sharma tərəfindən məsuliyyətli AI yaratmaq üçün açıq mənbə çərçivəsi

Həmçinin oxuyun:

- Microsoft-un RAI məlumat mərkəzi: [Məsuliyyətli AI məlumat mərkəzi – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft-un FATE tədqiqat qrupu: [FATE: Süni intellektdə ədalət, cavabdehlik, şəffaflıq və etika - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Məsuliyyətli AI Toolbox GitHub reposu](https://github.com/microsoft/responsible-ai-toolbox)

Ədalətliliyi təmin etmək üçün Azure Machine Learning alətləri haqqında oxuyun:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tapşırıq

[RAI Toolboxu kəşf edin](assignment.az.md)
