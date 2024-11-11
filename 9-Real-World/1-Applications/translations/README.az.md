# Postskript: Real həyatda maşın öyrənməsi

![Maşın öyrənməsinin real həyatda tətbiqinin ümumiləşdirilmiş eskizi](../../../sketchnotes/ml-realworld.png)
> [Tomomi Imura](https://www.twitter.com/girlie_mac) tərəfindən eskiz

Bu kurikulumda, sən öyrətmə datasının bir neçə hazırlama üsulunu və maşın öyrənməsi modelini qurmağı öyrənmisən. Sən bir neçə klassik reqressiya, klasterləşdirmə, qruplaşdırma, təbii dil emalı və zaman seriyası modelləri qurmusan. Təbriklər! İndi sənə maraqlı gələ bilər ki, bu modellərin hamısı real həyatda necə tətbiq olunurlar?

Sənayedə daha çox maraq AI, dərin öyrənmə ətrafında toplansa da, hələ də klassik maşın öyrənmə modelləri üçün çoxlu dəyərli tətbiqlər var. Sən bu tətbiqlərin hətta bəzilərini bugün istifadə etmisən! Bu dərsdə sən 8 fərqli sənayədə və ixtisaslaşmış sahədə bu modelləri istifadə etməklə tətbiqləri istifadəçilərə necə daha məhsuldar, etibarlı, ağıllı və dəyərli edildiyini kəşf edəcəksən.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/?loc=az)

## 💰 Maliyyə

Maliyyə sektoru maşın öyrənməsi üçün çox fürsətlər verir. Bu sahədəki çox problemlər özləri modelləşdirilib ML vasitəsilə həll olunmağa meyillidir.

### Bank kartlarında fırıldaqçılığıq təyini

Biz əvvəlki dərslərdə [k-ortalama klasterləşməsi](../../../5-Clustering/2-K-Means/translations/README.az.md) barədə öyrənmişik, bəs bunu bank kartı fırıldaqçılığı problemini həll etmək üçün necə istifadə edə bilərik?

K-ortalama klasterləşməsi bank kartı fırıldaqçılığını təyin edilməsi üçün **kənarlaşmanın təyini** adlanan texnikanın istifadəsində çox yararlı olur. Kənarlaşanlar və ya müşahidə olunan data qrupundan fərqlənənlər bizə bank kartının normal şəraitdə istifadə olunduğunu və ya şübhəli bir əməliyyatın icra olunduğunu deyə bilir. Aşağıdakı məqalədə göstərildiyi kimi siz bank kartı datalarını k-ortalama klasterləşməsi alqoritmi ilə sıralaya bilər və hər əməliyyatı bir klasterə aid edərək onun nə qədər kənarlaşdığını təyin edə bilərsiniz. Sonra siz fırıldaqçılığa yaxın ən riskli klasterləri normal əməliyyatlarla müqayisədə fərqləndirə bilərsiniz.
[İstinad](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Sərvətin idarə olunması

Sərvətin idarə edilməsində fərd və ya şirkət müştərilərinin adından sərmayələrin icrasında cavabdehdirlər. Onların işi şərvəti qorumaq və uzun müddətdə inkişaf etdirməkdir, buna görə də onlara sərmayəni düzgün seçərək məhsuldar nəticə göstərmək çox vacibdir.

Müəyyən bir sərmayənin necə işlədiyini ölçməyin yollarından biri statistik reqressiyadır. [Xətti reqressiya](../../../2-Regression/1-Tools/translations/README.az.md) fondun başqa etalonlarla müqayisədə necə işlədiyini başa düşmək üçün dəyərli bir alətdir. Biz həmçinin reqressiyanın nəticəsinin statistik olaraq nə qədər təsirli olduğunu və ya müştərinin sərmayəsinə nə qədər təsir edə biləcəyini öyrənə bilərik. Analizini əlavə olaraq risk faktorlarının nəzərə alan müxtəlif reqressiya üsulları ilə genişləndirə bilərsən. Misal üçün bunun müəyyən fonda necə təsir edə biləcəyini və onu necə ölçə biləcəyini aşağıdakı məqalədən öyrənə bilərsən.
[İstinad](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Təhsil

Təhsil sektoru da ML tətbiq oluna biləcək çox maraqlı sahədir. Burada imtahan və ya esselərdə fırıldağın, hansısa tərəfin şüurlu və ya şüursuz tutulmasını təyin olunması kimi maraqlı problemlər var.

### Tələbələrin davranışının proqnozlaşdırılması

[Coursera](https://coursera.com) online açıq kurs təminatçısıdır və onların çoxlu mühəndislik qərarlarının müzakirə olunduğu möhtəşəm bloqları var. Bir araşdırmada onlar aşağı NPS (Başqasına Tövsiyyə Dərəcəsi) ilə kursun baxılma və ya yarımçıq qoyulma halları arasında əlaqəni öyrənmək üçün reqressiya xətti çəkərək öyrənmək istəyiblər.
[İstinad](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Tərəf tutmanın qarşısının alınması

[Grammarly](https://grammarly.com) yazmaq üçün köməkçidir və öz məhsullarında xüsusi [təbii dil emalı sistemi](../../../6-NLP/translations/README.az.md) istifadə edərək yazı və qrammatika xətalarını tapır. Onlar texnoloji bloqlarında bizim [ədalətli sistemlərə giriş dərsimizdə](../../../1-Introduction/3-fairness/translations/README.az.md) öyrəndiyimiz kimi genderlər arasında tərəf tutmanın maşın öyrənməsi ilə necə həll etdikləri ilə bağlı maraqlı bir araşdırma paylaşıblar.
[İstinad](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Pərakəndə satış

Pərakəndə satış sektoru yaxşı müştəri təcrübəsi yaratmaqdan inventarda məhsulların optimal sayda stoklanmasına kimi ML istifadəsindən çox faydalar götürür.

### Müştəri təcrübəsini fərdiləşdirmək

Ev əşyaları və mebellər satışı edən Wayfair şirkəsi müştərilərinin zövqlərinə və ehtiyaclarına uyğun doğru məhsulu tapmaqda kömək edir. Bu məqalədə şirkətin mühəndisləri ML və NPL istifadə etməklə "müştərilərə doğru nəticələrin çıxardılması" necə etdiklərini paylaşıblar. Onlar məlumatın filtrlənməsi, qruplaşdırılması, məhsul və fikrin müştəri rəylərindən təyin olunmasını istifadə edərək sorğuların məqsədini təyin edən mühərrik (Query Intent Engine) hazırlayıblar. Bu NLP-nin onlayn pərakəndə sahəsində istifadə olunan klassik tətbiqidir.
[İstinad](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### İnventarın idarə edilməsi

İstehlakçılara geyin qutularının çatdırılması ilə məşğul olan [StitchFix](https://stitchfix.com) kimi innovativ və çevik şirkətlər məhsul tövsiyyəsi və inventarın idarə olunmasında ML-dən çox asılıdırlar. Onların moda komandaları ticarət komandaları ilə birlikdə çalışaraq belə bir nəticəyə gəliblər: "bizim data mühəndislərimizdən biri paltarlar üzərində genetika algoritmi tətbiq edərək indiyə kimi mövcud olmayan və uğurlu olacaq paltar nümunələri proqnozlaşdırmışdır. Biz bunu ticarət komandasına gətirdik və onu artıq alət kimi işlərində istifadə edirlər."
[İstinad](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Səhiyyə

Səhiyyə sektorunda araşdırmaları optimallaşdırılması, xəstələrin yenidən qəbulu zamanı logistika problemlərini həll etməyə və ya xəstəliyin yayılmasının qarşısını almaq üçün ML-dən istifadə oluna bilər.

### Tibbi təcrübələrin idarə edilməsi

Dərman hazırlayanlar üçün tibbi təcrübələrdəki zəhərlənmə halları ən narahat olduqları məqamdır. Nə qədər zəhərlənmə qəbul oluna bilər? Bu araşdırmada tibbi təcrübələrin nəticələrinin proqnozlaşdırılması üçün yeni üsulların yaradılmasında müxtəlif üsullar analiz olunub. Onlar xüsusi olaraq təsadüfi parametrlər istifadə edərək [qruplaşdırıcının](../../../4-Classification/translations/README.az.md) köməyi ilə dərman qruplarını ayırd edə biliblər.
[İstinad](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Xəstəxanada yenidən qəbulunun idarə edilməsi

Xəstəxana xidmədi çox bahalıdır, xüsusən də xəstələrin yenidən yerləşdirilməsi lazım olduqda. Bu araşdırmada ML [klasterləşdirmə](../../../5-Clustering/translations/README.az.md) alqoritmləri istifadə etməklə yenidən yerləşdirilməsi lazım olacaq xəstələri təxmin edən şirkət haqqında danışılır. Bu klasterlər "yenidən yerləşdirilməsi lazım olan qruplarda oxşar xüsusiyyətləri" analiz etməkdə kömək edir.
[İstinad](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Xəstəliklərin idarə olunması

Ən son pandemiya bizə maşın öyrənməsinin köməyi ilə xəstəliyin yayılmasının qarşısı alınması üsullarını aydınlatdı. Bu məqalədə ARIMA-nın, logistika əyrilərinin, xətti reqressiyanın və SARIMA-nın istifadəsini görəcəksən. "Bu araşdırma virusun yayılma sürətini hesablamağa, ölüm və sağalma ehtimallarını proqnozlaşdrmağa çalışıb bizim gələcəyə daha yaxşı hazırlıqlı olmağa və sağ qalmağa kömək edəcək."
[İstinad](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologiya və yaşıl texnologiya

Təbiət və ekologiya heyvanlar və təbiət arasında əlaqədə olan bir çox həssas sistemlərdən ibarətdir və daim diqqəti özünə cəlb edir. Bu sistemləri dəqiq ölçə bilmək çox vacibdir və meşə yanğınları və ya heyvanların artımında ani dəyişikləri analiz edib edib müvafiq addımlar atılmalıdır.

### Meşələrin idarə edilməsi

Əvvəlki dərslərdə [Gücləndirici öyrənmə (RL)](../../../8-Reinforcement/translations/README.az.md) haqqında öyrənmisən. Bu üsul təbiətdəki hərəkətləri proqnozlaşdırmaqda çox faydalıdır. Xüsusi ilə meşə yanğınları və ziyanverici həşaratlar kimi ekoloji problemləri izləməkdə istifadə oluna bilinir. Kanadada bir qrup tədqiqatçı Gücləndirici öyrənmə üsulu tətbiq edərək peyk şəkillərindən meşə yanğınlarının dinamika modelini qurublar. Onlar innovativ "məkanda yayılma prosesini(SSP)" istifadə etməklə meşə yanğınını "müstəvi üzərində istənilən mövqedə olan agentə" bənzədiblər. "Yanğının olduğu məkandan verilən zamandan sonra edə biləcəyi hərəkətlər yalnız şimala, cənuba, şərqə və qərbə yayılmaq və ya heç yayılmamaqdır.

Bu yanaşma sadə RL tətbiqini yanğının ani yayılmasının dinamikasına daha uyğun olan məşhur Markov Qərarvermə Prosesinə (MDP) çevirir." Aşağıdakı linkdən klassik alqoritmlərin bu qrupa aid problemlərdə necə istifadə olunması barədə oxuya bilərsiniz.
[İstinad](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Heyvanların hərəkətini hiss etmək

Dərin öyrənmə heyvanların hərəkətinin vizual olaraq izlənməsində inqilab yaratsa da (qütb ayısını izləmək üçün proqramı [burada](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) hazırlaya bilərsən), klassik ML-in bu tapşırıqda hələ də yeri var.

Ferma heyvanlarının hərəkətlərini izləyən sensorlar və IoT (Əşyaların interneti) cihazları vizual emal üçün istifadə olunur, lakin datanın əvvəldən emal olunması üçün sadə ML texnikalarını istifadə etmək daha faydalıdır. Misal üçün bu məqalədə qoyunların duruşları müxtəlif qruplaşdırıcı alqoritmlərlə müşahidə və analiz olunur. 335-ci səhifədə ROC əyrisini tanıyacaqsan.
[İstinat](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Enerjinin idarə edilməsi

[Zaman seriyasının proqnoqlaşdırılması](../../../7-TimeSeries/translations/README.az.md) dərsimizdə tələb və təklifi başa düşərək kəndə gəlir gətirəcək ağıllı parklama sayğacı konseptini qurmuşduq. Bu məqalədə ağıllı sayğaclar istifadə edərək klasterləşmə, reqressiya və zaman seriyasının proqnozlaşdırılmasının birgə tətbiqi ilə gələcəkdə İrlandiyada enerji istehlakını proqnozlaşdırılmasına necə kömək etdiyindən danışılır.
[İstinad](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Sığorta

Sığorta sektoru ML istifadə etməklə sərfəli maliyyə və risk modellərinin qurulmasının tətbiq olunduğu başqa bir sahədir.

### Dəyişkənliyin idarə edilməsi

Həyat sığortası təklif edən şirkət Metlife öz maliyyə modellərində dəyişkənlik risklərinin qarşısını almaq üçün istifadə etdiyi üsullardan bəhs etməyə dəyər. Bu məqalədə ikili və çoxsaylı qruplaşdırıcı vizuallardan istifadə olunduğunu görəcəksən. Sən həmçinin proqnozlaşdırma qrafiklərini də kəşf edəcəksən.
[İstinad](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 İncəsənət, Mədəniyyət, və Ədəbiyyat

İncəsənət sahəsində, misal olaraq jurnalistikada çoxlu sayda maraqlı problemlər var. Saxta xəbərlərin insanların fikirlərinə təsir etməsi, hətta demokratiyanın pozulmasına yol açması çox böyük bir problemdir. Muzeylər də həmçinin ML istifadə etməklə artefaktlar arasında əlaqələr tapmaqdan resurs planlamasına kimi çox şeydə faydalana bilirlər.

### Saxta xəbərlərin aşkarlanması

İndiki mediyada saxta xəbərlərin aşkarlanması siçan və pişik oynununa çevrilib. Bu məqalədə tədqiqatçılar yoxlamalarına əsasən bir neçə fərqli ML texnikalarını birləşdirərək ən yaxşı modeli yaratdıqlarını düşünürlər: "Bu sistem təbii dil emalı istifadə edərək datadan faktları çıxarır və həmin faktlar Naive Bayes, Dəstək Vektor Maşını (SVM), Random Forest (RF), Stokastik Qradient Enişi (SGD) və Logistik Reqressiyası (LR) kimi maşın öyrənməsi alqoritmlərində öyrətmə datası kimi istifadə olunur".
[İstinad](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Bu məqalə müxtəlif ML sahələrini birləşdirməklə saxta xəbərlərin yayılmasını və həqiqi ziyan verməsini dayandırmaqda kömək edən maraqlı nəticələrin əldə olunduğunu göstərir, misal üçün COVID müalicəsi ilə bağlı xəbərlərin yayılmasında təkan insanlar arasında izdihama və zorakılığa səbəb ola bilər.

### Muzeylərdə ML

Muzeylər AI inqilabı ilə tarixi zirvəyə qalxıb, belə ki kolleksiyaların rəqəmsallaşdırılması, kataloqlaşdırılması və artefaktların arasında əlaqələrin tapılması texnologiyanın köməyi ilə çox asanlaşıb. [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources) kimi layihələr Vatikan Arxivləri kimi əlçatmaz kolleksiyaların sirrlərini açmaqda kömək ediblər. Muzeylər ML modelləridən həmçinin biznes tərəfdən də faydalanırlar.

Misal üçün Çikaqo İncəsənət İnstitutu modellər qurmaqla onun proqnozları əsasında tamaşaçıların marağına və iştirakına əsaslanan sərgilər hazırlayırlar. Burada əsas məqsəd muzeyi ziyarət edən qonaqlara hər dəfə fərdiləşdirilmiş və optimallaşdırılmış təcrübə yaşatmaqdır. "İncəsənət İnstitutunun böyük vitse prezidenti Andryu Siminkin sözlərinə görə 2017-ci ildə model dəqiqlikdə 1 faiz xəta ilə iştirak və satışları proqnozlaşdırmışdır".
[İstinad](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketinq

### Müştəri seqmentasiyası

Marketingin ən effektiv strategiyası müştəriləri fərqli qruplar halında hədəfləməkdir. Bu məqalədə klasterləşmə alqoritmi tətbiq olunmaqla fərqləndirilən marketinqə necə dəstək olunması barədə danışılır. Fərqləndirilən marketinq şirlətlərə brendlərini daha yaxşı tanıtmağa, daha çox müştəriyə çıxmağa və daha çox pul qazanmağa kömək edir.
[İstinad](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Məşğələ

Bu kurikulumda öyrəndiyin texnikaların istifadə oluna biləcəyi başqa bir sahəni tap və ML-in necə istifadə olunmasını araşdır.

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/?loc=az)

## Təkrarlayın və özünüz öyrənin

Wayfair data elmi komandasının öz şirkətlərində ML-in necə istifadə etdikləri ilə bağlı çoxlu maraqlı videoları var. [Onlara baxmağa](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos) dəyər!

## Tapşırıq

[ML iməcliyi](assignment.az.md)
