# Maşın öyrənmə texnikaları

Maşın öyrənmə modellərinin və onların istifadə etdiyi məlumatların qurulması, istifadəsi və saxlanılması prosesi bir çox digər proqramlaşdırma proseslərindən çox fərqli bir prosesdir. Bu dərsdə biz prosesin gizli tərəflərini aydınlaşdırıb bilməli olduğunuz əsas texnikaları təsvir edəcəyik. Öyrənəcəklərin:

- Yüksək səviyyədə maşın öyrənməsinin əsasını təşkil edən prosesləri başa düşmək.
- "Modellər", "proqnozlar" və "təlim məlumatları" kimi əsas anlayışları kəşf etmək.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/?loc=az)

[![Yeni başlayanlar üçün ML- Maşın öyrənmə texnikaları](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "Yeni başlayanlar üçün ML- Maşın öyrənmə texnikaları")

> 🎥 Bu dərsin üzərindən keçən qısa video üçün yuxarıdakı şəkilin üzərinə klikləyin.

## Giriş

Yüksək səviyyədə maşın öyrənməsi (ML) prosesləri yaratmaq sənəti bir sıra addımlardan ibarətdir:

1. **Sualla bağlı qərar verin**. Əksər ML prosesləri sadə şərti proqram və ya qaydalara əsaslanan mühərrik tərəfindən cavablandırıla bilinməyən bir sual verməklə başlayır. Bu suallar çox vaxt məlumat toplusuna əsaslanan proqnozlar ətrafında fırlanır.
2. **Məlumatları toplayın və hazırlayın**. Sualınıza cavab vermək üçün sizə data lazımdır. Məlumatlarınızın keyfiyyəti və bəzən kəmiyyəti ilkin sualınıza nə qədər yaxşı cavab verə biləcəyinizi müəyyən edəcək. Məlumatların vizuallaşdırılması bu mərhələnin vacib aspektidir. Bu mərhələ həmçinin model yaratmaq üçün datanın öyrətmə və test qrupuna bölünməsini də əhatə edir.
3. **Təlim üsulunu seçin**. Sualınızdan və məlumatların xarakterindən asılı olaraq məlumatlarınızı ən yaxşı şəkildə əks etdirmək və ona qarşı dəqiq proqnozlar vermək üçün modeli necə öyrətmək istədiyinizi seçməlisiniz. Bu, ML prosesinizin xüsusi təcrübə və çox vaxt xeyli miqdarda təcrübə tələb edən hissəsidir.
4. **Modeli öyrədin**. Öyrətmə datadan istifadə edərək siz məlumatlar daxilində qanunauyğunluqları tapmaq üçün müxtəlif alqoritmlər istifadə edəcəksiniz. Model daha yaxşı bir model yaratmaq üçün məlumatların müəyyən hissələrini digərləri üzərində imtiyaz vermək üçün tənzimlənə bilən daxili çəkilərdən istifadə edə bilər.
5. **Modeli qiymətləndirin**. Modelin necə işlədiyini görmək üçün datasetdən əvvəl istifadə edilməmiş hissələri (test datanı) istifadə edirsiniz.
6. **Parametrlərin tənzimlənməsi**. Modelinizin performansına əsaslanaraq, modeli daha yaxşı öyrətmək üçün istifadə olunan alqoritmlərin parametrlərini dəyişərək prosesi yenidən təkrar edə bilərsiniz.
7. **Proqnoz vermək**. Modelinizin dəqiqliyini yoxlamaq üçün yeni giriş datalardan istifadə edin.

## Hansı sualı vermək lazımdır

Kompüterlər məlumatlarda gizli modelləri aşkar etməkdə xüsusilə bacarıqlıdırlar. Bu yardım proqramı şərti əsaslı qaydalar mühərriki yaratmaqla asanlıqla cavablandırıla bilinməyən müəyyən bir mövzular haqqında sualları olan tədqiqatçılar üçün çox faydalıdır. Məsələn, bir tapşırığı nəzərə alsaq, data mühəndisi siqaret çəkənlərə qarşı siqaret çəkməyənlərin ölümü ilə bağlı özü yaratdığı model qura bilər.

Bir çox digər dəyişənlər tənliyə gətirildikdə ML modeli keçmiş sağlamlıq tarixinə əsaslanaraq gələcək ölüm ehtimalını proqnozlaşdırmaq üçün daha səmərəli ola bilər. Daha pozitiv bir nümunə, enlik, uzunluq, iqlim dəyişikliyi, okeana yaxınlıq, cərəyanların axınının nümunələri və s. daxil olan məlumatlara əsaslanaraq müəyyən bir yerdə aprel ayı üçün hava proqnozları verilə bilər.

✅ [Bu təqdimatda](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) ML istifadə edilərək keçmiş hava məlumatları analizi istifadə edilərək hava modelinin qurulmasını təsvir edilir.

## Model qurmaqdan əvvəl tapşırıqlar

Modelinizi qurmağa başlamazdan əvvəl yerinə yetirməli olduğunuz bir neçə tapşırıq var. Sualınızı yoxlamaq və modelin proqnozlarına əsaslanan fərziyyə formalaşdırmaq üçün bir neçə elementi müəyyən edib sazlamalısınız.

### Data

Sualınıza hər cür əminliklə cavab verə bilmək üçün sizə düzgün tipdə və çox miqdarda məlumat lazımdır. Bu nöqtədə etməli olduğunuz iki şey var:

- **Data toplayın**. Datanın təhlilində ədalətlə bağlı əvvəlki dərsi nəzərə alaraq ehtiyatla toplayın. Bu datanın mənbələrində malik ola biləcək hər hansı bir qərəzdən xəbərdar olun və mənşəyini sənədləşdirin.

- **Data hazırlayın**. Datanın hazırlanması prosesində bir neçə addım var. Müxtəlif mənbələrdən gəldiyi üçün datanı toplamaq və normallaşdırmaq lazım ola bilər. Verilənlərin keyfiyyətini və kəmiyyətini müxtəlif üsullarla yaxşılaşdıra bilərsiniz, misal üçün söz kimi yazılmış ədədləri rəqəmlərə keçirmək ([Klasterləşdirmə](../../../5-Clustering/1-Visualize/translations/README.az.md) bölməsində etdiyimiz kimi). Siz həmçinin əldə olunmuş məlumatlardan yeni məlumatlar da yarada bilərsiniz. ([Qruplaşdırma](../../../4-Classification/1-Introduction/translations/README.az.md) bölməsində etdiyimiz kimi). Datanı təmizləyə və dəyişə bilərsiniz ([Veb tətbiq](../../../3-Web-App/translations/README.az.md) dərsində etdiyimiz kimi). Yekun olaraq siz öyrətmə texnikasından asılı olaraq məlumatları təsadüfi seçimlərlə əvəzləyə və ya yerlərini qarışdıra bilərsiniz.

✅ Datanı yığdıqdan və emal etdikdən sonra vaxt ayırıb fikrinizdə tutduğunuz suala uyğun işləri gördüyünüzdən əmin olun. [Klasterləşdirmə](../../../5-Clustering/1-Visualize/translations/README.az.md) dərslərində öyrəndiyimiz kimi bəzən yığdığımız məlumatlar bizim tapşırığımız üçün yaxşı nəticələr göstərməyə bilər!

### Xüsusiyyətlər və hədəf

[Xüsusiyyət](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) datanızın ölçülə bilən parametridir. Bir çox verilənlər bazasında o, 'tarix' 'ölçü' və ya 'rəng' adları ilə sütun başlığı kimi ifadə edilir. Kodda adətən `X` kimi təqdim olunan xüsusiyyət dəyişəniniz modeli öyrətmək üçün istifadə olunacaq giriş dəyişənini təmsil edir.

Hədəf, proqnozlaşdırmağa çalışdığınız bir şeydir. Kodda adətən `y` kimi təqdim olunan hədəf məlumatlarınız haqqında soruşmağa çalışdığınız sualın cavabını təmsil edir: dekabr ayında hansı **rəng** balqabaq ən ucuz olacaq? San Fransiskoda hansı məhəllələrdə ən yaxşı daşınmaz əmlakın **qiyməti** neçə olacaq? Bəzən hədəfə etiket atributu da deyilir.

### Xüsusiyyət üçün dəyişən seçmək

🎓 **Xüsusiyyət seçimi və xüsusiyyətlərin çıxarılması**. Model qurarkən hansı dəyişəni seçmək lazım olduğunu necə bilirsiniz? Çox güman ki, ən effektiv model üçün düzgün dəyişənləri seçmək üçün **xüsusiyyət seçimi** və ya **xüsusiyyət çıxarılması** prosesindən keçəcəksiniz. Bununla belə, onlar eyni şey deyillər: "**Xüsusiyyətlərin çıxarılması** orijinal xüsusiyyətlərin funksiyalarından yeni xüsusiyyətlər yaradır, halbuki **xüsusiyyət seçimi** xüsusiyyətlərin alt dəstini qaytarır." ([mənbə](https://wikipedia.org/wiki/Feature_selection))

### Data vizuallaşdırılması

Data mühəndisinin alət dəstinin mühüm cəhəti Seaborn və ya MatPlotLib kimi bir neçə mükəmməl kitabxanadan istifadə edərək məlumatları vizuallaşdırmaq gücüdür. Verilənlərinizi vizual şəkildə təmsil etmək sizə istifadə edə biləcəyiniz gizli uyğunluqları aşkar etməyə imkan verə bilər. Vizuallaşdırmalarınız sizə qərəzli və ya balanssız məlumatları da aşkar etməyə kömək edə bilər ([Qruplaşdırma](../../../4-Classification/1-Introduction/translations/README.az.md) bölməsində öyrəndiyimiz kimi).


### Dataseti bölün

Təlimdən əvvəl verilənlər bazanızı məlumatları yaxşı təmsil etməsini ödəyən qeyri-bərabər ölçülü iki və ya daha çox hissəyə bölmək lazımdır.

- **Öyrətmə**. Modelinizi öyrətmək üçün uyğun seçilmiş məlumat dəstinin bir hissəsidir. Bu dəst orijinal datasetin əksəriyyətini təşkil edir.
- **Test**. Test verilənlər toplusu qurulmuş modelin performansını təsdiqləmək üçün istifadə etdiyiniz, adətən orijinal məlumatlardan seçilən müstəqil datasetdir.
- **Doğrulama**. Doğrulama dəsti modeli təkmilləşdirmək üçün modelin hiperparametrlərini və ya arxitekturasını tənzimləmək üçün istifadə etdiyiniz daha kiçik müstəqil nümunələr qrupudur. Məlumatınızın ölçüsündən və verdiyiniz sualdan asılı olaraq bu üçüncü dəsti qurmağa ehtiyacınız olmaya bilər ([Zaman seriyalarının proqnozlaşdırılması](../../../7-TimeSeries/1-Introduction/translations/README.az.md) bölməsində qeyd etdiyimiz kimi).

## Model qurmaq

Sizin hədəfiniz öyrətmə datasından və ya datanın statistik təsvirindən istifadə edərək model qurmaq və onu müxtəlif alqoritmlərlə **öyrətməkdir**. Modelin öyrədilməsi onun məlumatlara əlçatan edir və imkan verir ki, başa düşdüyü və kəşf etdiyi qanunauyğunları yoxlasın, qəbul etsin və ya imtina etsin.

### Öyrənmə üsulunu seçin

Sualınızdan və məlumatlarınızın xarakterindən asılı olaraq, siz onu öyrətmək üçün bir üsul seçəcəksiniz. Bu kursda istifadə etdiyimiz [Scikit-learn's sənədləri](https://scikit-learn.org/stable/user_guide.html) ilə addımlayaraq siz bir modeli öyrətməyin bir çox yolunu araşdıra bilərsiniz. Təcrübənizdən asılı olaraq ən yaxşı modeli yaratmaq üçün bir neçə fərqli metodu sınamalı ola bilərsiniz. Çox güman ki, siz data mühəndislərinin modellərin performansını ölçdükləri yollardan keçəcəksiniz. Bunlara modeli istifadə edilməmiş datalarla öyrətmə, dəqiqliyin yoxlama, qərəz və digər keyfiyyəti aşağı sala biləcək problemləri aşkarlama yolları aiddir. Bunların sonunda isə siz ən uyğun öyrənmə metodunu seçmiş olacaqsınız.

### Modeli öyrətmək

Öyrətmə datanızı hazır tutmaqla siz modelinizi "uyğunlaşdırmağa" hazırsınız. Siz görəcəksiniz ki, bir çox ML kitabxanalarında 'model.fit' funksiyası var - məhz burada siz xüsusiyyət dəyişənlərini məlumat toplusu kimi (adətən 'X') və hədəf dəyişənini (adətən 'y') ötürəcəksiniz.

### Modeli qiymətləndirin

Öyrətmə prosesi başa çatdıqdan sonra (böyük modeli hazırlamaq üçün bir çox təkrarlamalar və ya “epoxalar” tələb oluna bilər), siz modelin keyfiyyətini ölçmək üçün test məlumatlarından istifadə edə biləcəksiniz. Bu məlumat modelin əvvəllər təhlil etmədiyi ilkin məlumatların bir hissəsidir. Siz modelinizin keyfiyyətinə dair göstəricilər cədvəlini çap edə bilərsiniz.

🎓 **Modelin uyğunlaşdırılması**

Maşın öyrənməsi kontekstində model uyğunluğu modelin əsas funksiyasının dəqiqliyinə istinad edir. Çünki burada model tanış olmayan məlumatları təhlil etməyə çalışır.

🎓 **Az uyğunlaşdırılma** və **çox uyğunlaşdırılma** modelin keyfiyyətini aşağı salan ümumi problemlərdir, burada model ya çox uyğundur, ya da kifayət qədər deyildir. Bunlar modelin təxminlərini ya əvvəlki məlumalarla çox yaxın etmətə çalışır, ya da təlim məlumatlarından uzaq edir. Çox uyğunlaşdırılmış modellər təlim məlumatlarının xüsusiyyətlərini və xətalı hissələri çox yaxşı öyrənmişdir. Az uyğunlaşdırılmış modelin dəqiqliyi isə həm təlim məlumatlarını az dəqiqliklə öyrənmiş qədər, həm də heç öyrənməmiş qədər az olur.

![çox uyğunlaşdırılmış model](../images/overfitting.png)
> [Jen Looper](https://twitter.com/jenlooper) tərəfindən infoqrafik

## Parametrlərin tənzimlənməsi

İlkin öyrətmə tamamlandıqdan sonra modelin keyfiyyətini müşahidə edin və onun “hiperparametrlərini” tənzimləməklə təkmilləşdirməyi düşünün. Proses haqqında bu [sənədlərdən](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott) ətraflı oxuya bilərsiniz.

## Proqnoz

Bu anda siz modelin dəqiqliyini yoxlamaq üçün tamamilə yeni məlumatlardan istifadə edə bilərsiniz. "Tətbiqi" ML konfiqurasiyalarında siz veb mühitdə istifadəçilərdən giriş məlumatlarını alaraq dəyişənlər kimi modelinizə daxil edə və nəticəni hesablaya bilərsiniz.

Bu dərslərdə siz data mühəndisinin istifadə etdiyi jestlərin hamısını - hazırlama, qurma, sınaqdan keçirmə, qiymətləndirmə və proqnozlaşdırma üçün bu addımlardan necə istifadə edəcəyinizi kəşf edəcəksiniz və bu sizin ML mühəndisi olmaq üçün öyrənmə səyahətinizdə kömək edəcək.

---

## 🚀 Məşğələ

ML təcrübəçisinin addımlarını əks etdirən proses sxemini çəkin. Hazırda prosesin gedişində özünüzü harada görürsünüz? Harada çətinlik çəkəcəyinizi proqnozlaşdırırsınız? Sizə asan görünən nədir?

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/?loc=az)

## Təkrarlayın və özünüz öyrənin

İnternetdə data mühəndisi ilə gündəlik iş həyatları müzakirə edilən müsahibələrini axtarın. [Burada](https://www.youtube.com/watch?v=Z3IjgbbCEfs) biri var.

## Tapşırıq

[Data mühəndisi ilə müsahibə](assignment.az.md)