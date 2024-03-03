# Məhsul Sİ ilə maşın öyrənməsi həlləri tapmaq.

![Eskizdə Maşın Öyrənməsində məhsul Sİ-in xülasəsi](../../../sketchnotes/ml-fairness.png)
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

![translation to Turkish](../images/gender-bias-translate-en-tr.png)
> türk dilinə tərcümə

![translation back to English](../images/gender-bias-translate-tr-en.png)
> ingilis dilinə geri tərcümə

Süni intellekt sistemlərinin dizayn edilməsi və test olunması zamanı biz süni intellektin ədalətli olmasını və qərəzli və ya ayrı-seçkilik xarakterli qərarlar qəbul etmək üçün proqramlaşdırılmamasını təmin etməliyik, hansı ki insanların da qəbul etməsi qadağandır. Süni intellekt və maşın öyrənməsində ədalətin təmin edilməsi mürəkkəb sosial texniki problem olaraq qalır.

### Etibarlılıq və təhlükəsizlik

Güvən yaratmaq üçün AI sistemləri həm normal həm də gözlənilməz şəraitdə etibarlı, təhlükəsiz və düzgün olmalıdır. Süni intellekt sistemlərinin müxtəlif situasiyalarda necə davranacağını bilmək vacibdir, xüsusən də onlar normal şəraitdən kənara çıxdıqda. Süni intellekt həllərini qurarkən, süni intellektin qarşılaşacağı müxtəlif vəziyyətlərin necə idarə olunacağına böyük diqqət yetirilməlidir. Məsələn, özünü idarə edən avtomobil insanların təhlükəsizliyini əsas prioritet kimi qoymalıdır. Nəticədə, avtomobili idarə süni intellekt avtomobilin qarşılaşa biləcəyi bütün mümkün ssenariləri nəzərə almalıdır, məsələn, gecə, tufan və ya çovğun, küçədə qaçan uşaqlar, ev heyvanları, yol tikintiləri və s. Bir Sİ sisteminin nə qədər yaxşı olduğu bir sıra fərqli şəraitdə nə dərəcə etibarlı və təhlükəsiz işlədiyindən asılıdır, bu, sistem dizaynı və testi zamanı data alimi və ya Sİ təminatçısı tərəfindən nəzərə alınan proqnoz səviyyəsini göstərir.

> [🎥 Video üçün bura klikləyin: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Daxil edilmə

Sİ sistemləri hər kəslə əlaqədə olacaq və fayda verəcək şəkildə formalaşdırılmalıdır. Sİ sistemlərinin dizayn və icra mərhələrində data mühəndisləri və proqramçılar sistemin istəmədən kimlərisə xaric etməsi ehtimalarını təyin etməyə və qarşısını almağa çalışırlar. Misal üçün dünyada müxtəlif əngəlləri olan 1 milyard insan var. Sİ inkişafı nəticəsində onlar geniş miqyasda məlumatlara və fürsətlərə gündəlik həyatlarında asanlıqla çata bilirlər. Əngəllər barədə öncədən düşünmək Sİ məhsullarını hər kəs tərəfindən faydalana biləcəyi daha yaxşı təcrübəni yaratmağa fürsətlər yaradır.

> [🎥 Video üçün bura klikləyin: Sİ-də daxil edilmə](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Təhlükəsizlik və məxfilik

Sİ sistemləri təhlükəsiz olmalı və insanlarım məxfiliyinə hörmət etməlidir. İnsanlar onların məxfiliyini, məlumatlarını və ya həyatlarını risk altında qoyan sistemlərə az güvənirlər. Maşın öyrənməsi modellərini sazladığımız zaman ən yaxşı nəticəni verən datalara etibar edirik. Bunu etdiyimiz zaman datanın mənbəsini və düzgün üsulla əldə olduğunu da nəzərə almalıyıq. Misal üçün, bu məlumat istifadəçi tərəfindən təqdim olunub və ya ictimai olaraq əlçatandırmı? Data ilə işləyərkən nəzərə alınmalı növbəti məsələ, Sİ sistemlərinin məxfi məlumatlarını qoruyacağı və hücumlara davamlı olacağı şəkildə hazırlanması vacibdir. Sİ istifadəsi artıqca vacib şəxsi və biznes məlumatlarının məxfiliyinin qorunması daha vacib və qəliz hala gəlir. Məxfilik və data təhlükəsizliyi problemləri Sİ üçün daha çox diqqət tələb edir, çünki bu məlumatlar Sİ sistemlərinin insanlar barədə dəqiq təxminlər və qərarlar verməsi üçün lazımdır.

> [🎥 Video üçün bura klikləyin: Sİ-də təhlükəsizlik](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Cari sənayedə Məxfilik və təhlükəsizlik sahəsində əhəmiyyətli irəliləyişlər etmişik, GDPR (Ümumi Məlumatların Qorunması Qaydası) kimi qaydalarla diqqəti daha da artırmışıq
- Buna baxmayaraq biz Sİ sistemlərində effektiv fərdiləşdirilmə üçün şəxsi məlumatlara olan ehtiyac ilə təhlükəsizlik arasında olan gərginliyi başa düşməliyik.
- Kompüterlərin internetə qoşulması ilə təhlükəsizlik problemlərinin yaranmasındakı böyük sıçrayışı indi Sİ ilə əlaqəli sistemlərində görürük.
- Eyni zamanda biz Sİ təhlükəsizlik tərəfdən inkişaf etdiyini də görürük. Misal olaraq, bugünkü müasir anti-virus proqramları Sİ sistemlərinin iştirakı ilə həyata keçirir.
- Bizim Data elmi proseslərinin ən son məxfilik və təhlükəsizlik təcrübələrini özündə birləşdiyinə əmin olmağımız lazımdır.

### Şəffaflıq

Sİ sistemləri başa düşülən olmalıdır. Şəffaflığın ən vacib hissəsi Sİ sistemlərinin davranışını və komponentlərini izah etməkdir. Sİ sistemlərinin anlaşıqlığını artırmaqla biz mümkün performans problemlərini, təhlükəsizlik və məxfilik məsələlərini, qərəzləri, çıxdaş edilmə hallarını və ya istənilməyən nəticələri daha tez tapa bilərik. Biz həmçinin inanırıq ki, insanlar Sİ sistemlərini nə zaman, nə üçün və necə istifadə edəcəklərini seçməkləri barədə səmimi, məsuliyyətli olmalıdırlar. Əlavə olaraq işlətdikləri sistemin limitləri barədə də məlumatlı olmalıdırlar. Misal üçün, əgər bank Sİ sistemini müştərilərə verəcək kredit təklifini hazırlamaqda köməkçi kimi istifadə edirsə, bu sistemin verdiyi təklifləri ən çox hansı parametrlərin təsir etdiyini araşdırması vacibdir. Hökümətlər Sİ-in sənayelərdə istifadəsinə qaydalar tətbiq etməyə başlayır, belə ki data mühəndisləri və şirkətləri Sİ sisteminin verdiyi qərarların (xüsusilə arzuolunmaz nəticələrin) tənzimləyici tələblərə cavab verdiyini izah edə bilməlidirlər.

> [🎥 Video üçün bura klikləyin: Sİ-də şəffaflıq](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Sİ sistemləri qəliz olduğu üçün onun işləmə qaydasını və verdiyi qərarları anlamaq çətindir.
- Bu anlaşılmada əksiklik həmin sistemlərin idarə edilməsinə, istifadə edilməsinə və sənədləşməsinə təsir edir.
- Bu anlaşılmadakı əksiklik ən vacib olaraq sistemin səbəb olduğunu nəticəyə gətirən qərarlara təsir edir.

### Accountability

The people who design and deploy AI systems must be accountable for how their systems operate. The need for accountability is particularly crucial with sensitive use technologies like facial recognition. Recently, there has been a growing demand for facial recognition technology, especially from law enforcement organizations who see the potential of the technology in uses like finding missing children. However, these technologies could potentially be used by a government to put their citizens’ fundamental freedoms at risk by, for example, enabling continuous surveillance of specific individuals. Hence, data scientists and organizations need to be responsible for how their AI system impacts individuals or society.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Click the image above for a video: Warnings of Mass Surveillance Through Facial Recognition

Ultimately one of the biggest questions for our generation, as the first generation that is bringing AI to society, is how to ensure that computers will remain accountable to people and how to ensure that the people that design computers remain accountable to everyone else.

## Impact assessment

Before training a machine learning model, it is important to conduct an impact assessmet to understand the purpose of the AI system; what the intended use is; where it will be deployed; and who will be interacting with the system.  These are helpful for reviewer(s) or testers evaluating the system to know what factors to take into consideration when identifying potential risks and expected consequences.

The following are areas of focus when conducting an impact assessment:

* **Adverse impact on individuals**.  Being aware of any restriction or requirements, unsupported use or any known limitations hindering the system's performance is vital to ensure that the system is not used in a way that could cause harm to individuals.
* **Data requirements**.  Gaining an understanding of how and where the system will use data enables reviewers to explore any data requirements you would need to be mindful of (e.g., GDPR or HIPPA data regulations).  In addition, examine whether the source or quantity of data is substantial for training.
* **Summary of impact**.  Gather a list of potential harms that could  arise from using the system.  Throughout the ML lifecycle, review if the issues identified are mitigated or addressed.
* **Applicable goals** for each of the six core principles.  Assess if the goals from each of the principles are met and if there are any gaps.


## Debugging with responsible AI

Similar to debugging a software application, debugging an AI system is a necessary process of identifying and resolving issues in the system.  There are many factors that would affect a model not performing as expected or responsibly.  Most traditional model performance metrics are quantitative aggregates of a model's performance, which are not sufficient to analyze how a model violates the responsible AI principles. Furthermore, a machine learning model is a black box that makes it difficult to understand what drives its outcome or provide explanation when it makes a mistake.  Later in this course, we will learn how to use the Responsible AI dashboard to help debug AI systems.  The dashboard provides a holistic tool for data scientists and AI developers to perform:

* **Error analysis**.  To identify the error distribution of the model that can affect the system's fairness or reliability.
* **Model overview**. To discover where there are disparities in the model's performance across data cohorts.
* **Data analysis**.  To understand the data distribution and identify any potential bias in the data that could lead to fairness, inclusiveness, and reliability issues.
* **Model interpretability**. To understand what affects or influences the model's predictions. This helps in explaining the model's behavior, which is important for transparency and accountability.


## 🚀 Challenge

To prevent harms from being introduced in the first place, we should:

- have a diversity of backgrounds and perspectives among the people working on systems
- invest in datasets that reflect the diversity of our society
- develop better methods throughout the machine learning lifecycle for detecting and correcting responible AI when it occurs

Think about real-life scenarios where a model's untrustworthiness is evident in model-building and usage. What else should we consider?

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Review & Self Study

In this lesson, you have learned some basics of the concepts of fairness and unfairness in machine learning.

Watch this workshop to dive deeper into the topics:

- In pursuit of responsible AI: Bringing principles to practice by Besmira Nushi, Mehrnoosh Sameki and Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Click the image above for a video: RAI Toolbox: An open-source framework for building responsible AI by Besmira Nushi, Mehrnoosh Sameki, and Amit Sharma

Also, read:

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft’s FATE research group: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Read about Azure Machine Learning's tools to ensure fairness:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Assignment

[Explore RAI Toolbox](assignment.md)
