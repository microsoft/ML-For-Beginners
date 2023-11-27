# Məhsul Sİ ilə maşın öyrənməsi həlləri tapmaq.
 
![Eskizdə Maşın Öyrənməsində məhsul Sİ-in xülasəsi](../../sketchnotes/ml-fairness.png)
> [Tomomi Imura](https://www.twitter.com/girlie_mac) tərəfindən eskiz

## [Mühazirə öncəsi test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)
 
## Giriş

Bu kurrikulumda siz maşın öyrənməsinin gündəlik həyatımıza necə təsir edə biləcəyini və necə təsir etdiyini kəşf etməyə başlayacaqsınız. Hətta indi də sistemlər və modellər səhiyyə diaqnozları, kreditlərin təsdiqlənməsi və ya dələduzluğun aşkarlanması kimi gündəlik qərar qəbuletmə işlərində iştirak edirlər. Buna görə də, etibarlı nəticələr əldə etmək üçün bu modellərin yaxşı işləməsi vacibdir. Hər hansı bir proqram tətbiqi kimi, Sİ sistemləri də gözləntilərə çatmayacaq və ya arzuolunmaz nəticə ilə üzləşəcək. Buna görə də Sİ modelinin davranışını başa düşmək və izah etmək vacibdir.

Təsəvvür edin ki, bu modelləri yaratmaq üçün istifadə etdiyiniz məlumatlar irq, cins, siyasi görüş, din kimi müəyyən demoqrafik göstəricilərə malik olmadıqda və ya qeyri-mütənasib şəkildə bu cür demoqrafik göstəriciləri təmsil edərsə nə baş verə bilər. Modelin çıxışı bəzi demoqrafik göstəricilərə üstünlük vermək üçün şərh edilməsinin nəticəsi nə olacaq? Tətbiq üçün nəticəsi nə olacaq? Bundan əlavə, modelin mənfi nəticəsi olduqda və insanlar üçün zərərli olduqda nə baş verəcək? Sİ sistemlərinin davranışına görə kim cavabdehdir? Bunlar bu kurrikulumda araşdıracağımız bəzi suallardır.

Bu dərsdə siz: 

- Maşın öyrənməsində ədalətin əhəmiyyəti və ədalətlə əlaqəli zərərlər haqqında məlumatlılığınızı artıracaqsınız
- Etibarlılıq və təhlükəsizliyi təmin etmək üçün kənar göstəriciləri və qeyri-adi ssenariləri araşdırmaq təcrübəsi ilə tanış olacaqsınız
- İnklüziv sistemlərin dizayn edilməsi ilə hər kəsin səlahiyyətləndirilməsi ehtiyacı haqqında anlayış əldə edəcəksiniz
- Məlumatların və insanların məxfiliyinin və təhlükəsizliyinin qorumasının nə qədər vacib olduğunu araşdıracaqsınız
- Süni intellekt modellərinin davranışını izah etmək üçün şüşə qutu yanaşmasının vacibliyini görəcəksiniz
- Süni intellekt sistemlərinə inam yaratmaq üçün məsuliyyətliliyin necə vacib olduğunu nəzərə alacaqsınız

## Tələb olunanlar

İlkin olaraq, zəhmət olmasa, "Məsul Sİ Prinsipləri" Təlim Yolunu başlayın və mövzu ilə bağlı aşağıdakı videoya baxın:

Bu [Təlim Yolunu](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott) izləməklə Məhsul Sİ haqqında daha çox məlumat əldə edin 

[![Microsoft-un Məsul Sİ-ə yanaşması](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft-un Məsul Sİ-ə yanaşması")

> 🎥 Video üçün yuxarıdakı şəklə klikləyin: Microsoft-un Məsul Sİ-ə yanaşması

## Ədalətlilik

Süni intellekt sistemləri hər kəsə ədalətli davranmalı və oxşar insan qruplarına müxtəlif yollarla təsir etməməlidir. Məsələn, süni intellekt sistemləri tibbi müalicə, kredit müraciətləri və ya işə qəbulla bağlı təlimat verdikdə, oxşar simptomları, maliyyə vəziyyəti və ya peşəkar keyfiyyətləri olan hər kəsə eyni tövsiyələri verməlidir. İnsan olaraq hər birimiz qərarlarımıza və hərəkətlərimizə təsir edən irsi qərəzləri daşıyırıq. Bu qərəzlər AI sistemlərini öyrətmək üçün istifadə etdiyimiz məlumatlarda aydın görünə bilər. Belə manipulyasiya bəzən istəmədən baş verə bilər. Məlumatlarda qərəzliliyi nə vaxt tətbiq etdiyinizi şüurlu şəkildə bilmək çox vaxt çətindir.

**“Haqsızlıq”** irq, cins, yaş və ya əlillik statusu kimi xüsusiyyətləti baxımından bir qrup insan üçün mənfi təsirləri və ya “zərərləri” əhatə edir. Ədalətlə əlaqəli əsas zərərlər aşağıdakı kimi təsnif edilə bilər:

- **Ayrı-seçkilik**, məsələn, cins və ya etnik mənsubiyyət digərindən üstündürsə.

- **Xidmət keyfiyyəti**. Əgər məlumatı konkret bir ssenari üçün öyrədirsinizsə, lakin reallıq daha mürəkkəbdirsə, bu, xidmətin keyfiyyətsiz olmasına gətirib çıxarır. Məsələn, qara dərili insanları hiss edə bilməyən əl sabunu dispenseri. [İstinad](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)

- **Təhqir**. Bir şeyi və ya kimisə haqsız yerə tənqid etmək, etiketləmək. Məsələn, bir şəkil etiketləmə texnologiyası qara dərili insanların şəkillərini qorilla kimi yanlış etiketlədi.

- **Həddindən artıq və ya az təmsil olunma**. İdeya ondan ibarətdir ki, müəyyən bir qrup müəyyən bir peşədə kifayət qədər təmsil olunmur və zərər verən hər hansı bir xidmət və ya funksiya bunu təbliğ etməyə davam edir.

- **Stereotipləşdirmə**. Müəyyən bir qrupun ön yarğılı fikirlər əlaqələndirilməsi. Məsələn, ingilis və türk dilləri arasında dil tərcümə sistemində cins ilə stereotipik əlaqəsi olan sözlərə görə səhvlər ola bilər.

![translation to Turkish](images/gender-bias-translate-en-tr.png)
> türk dilinə tərcümə

![translation back to English](images/gender-bias-translate-tr-en.png)
> ingilis dilinə geri tərcümə

Süni intellekt sistemlərinin dizayn edilməsi və test olunması zamanı biz süni intellektin ədalətli olmasını və qərəzli və ya ayrı-seçkilik xarakterli qərarlar qəbul etmək üçün proqramlaşdırılmamasını təmin etməliyik, hansı ki insanların da qəbul etməsi qadağandır. Süni intellekt və maşın öyrənməsində ədalətin təmin edilməsi mürəkkəb sosial texniki problem olaraq qalır.

### Etibarlılıq və təhlükəsizlik

Güvən yaratmaq üçün AI sistemləri həm normal həm də gözlənilməz şəraitdə etibarlı, təhlükəsiz və düzgün olmalıdır. Süni intellekt sistemlərinin müxtəlif situasiyalarda necə davranacağını bilmək vacibdir, xüsusən də onlar normal şəraitdən kənara çıxdıqda. Süni intellekt həllərini qurarkən, süni intellektin qarşılaşacağı müxtəlif vəziyyətlərin necə idarə olunacağına böyük diqqət yetirilməlidir. Məsələn, özünü idarə edən avtomobil insanların təhlükəsizliyini əsas prioritet kimi qoymalıdır. Nəticədə, avtomobili idarə süni intellekt avtomobilin qarşılaşa biləcəyi bütün mümkün ssenariləri nəzərə almalıdır, məsələn, gecə, tufan və ya çovğun, küçədə qaçan uşaqlar, ev heyvanları, yol tikintiləri və s. Bir Sİ sisteminin nə qədər yaxşı olduğu bir sıra fərqli şəraitdə nə dərəcə etibarlı və təhlükəsiz işlədiyindən asılıdır, bu, sistem dizaynı və testi zamanı data alimi və ya Sİ təminatçısı tərəfindən nəzərə alınan proqnoz səviyyəsini göstərir. 

> [🎥 Video üçün bura klikləyin:: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

