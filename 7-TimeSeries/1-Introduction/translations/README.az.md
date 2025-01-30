# Zaman seriyalarının proqnozlaşdırılmasına giriş

![Zaman seriyalarının eskiz üzərində icmalı](../../../sketchnotes/ml-timeseries.png)

> [Tomomi Imura](https://www.twitter.com/girlie_mac) tərəfindən çəkilmiş eskiz

Bu və sonrakı dərsdə siz ML mühəndisinin repertuarının digər mövzulardan bir qədər az tanınan maraqlı və dəyərli hissəsi olan zaman seriyalarının proqnozlaşdırılması haqqında öyrənəcəksiniz. Bu proqnozlaşdırma bir növ 'kristal kürə'-yə bənzəyir: məsələn, qiymət dəyişəninin keçmiş performansına əsaslanaraq, onun gələcək potensial dəyərini təxmin edə bilərsiniz.

[![Zaman seriyalarının proqnozlaşdırılmasına giriş](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Zaman seriyalarının proqnozlaşdırılmasına giriş")

> 🎥 Zaman seriyalarının proqnozlaşdırılması haqqında olan video üçün yuxarıdakı şəkilə klikləyin.

## [Mühazirədən əvvəl test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41/?loc=az)

Zaman seriyalarının proqnozlaşdırılması qiymət, inventar və təchizat zənciri problemlərinə birbaşa tətbiqi nəzərə alınmaqla biznes üçün real dəyərə malik faydalı və maraqlı sahədir. Gələcək performansı daha yaxşı proqnozlaşdırmaq üçün daha çox məlumat əldə etmək üçün dərin öyrənmə üsullarından istifadə edilməyə başlansa da, zaman seriyalarının proqnozlaşdırılması klassik ML texnikalarına əsaslanan bir sahə olaraq qalır.

> Penn State-in faydalı zaman seriyası kurrikulumuna [burada](https://online.stat.psu.edu/stat510/lesson/1) baxa bilərsiniz.

# Giriş

Tutaq ki, siz onların nə qədər tez-tez istifadə edildiyi və zamanla nə qədər müddətə istifadə edildiyi barədə məlumat verən bir seriya ağıllı parkomatlara sahibsiniz.

> Əgər tələb və təklif qaydalarına uyğun olaraq sayğacın keçmiş performansına əsaslanıb onun gələcək dəyərini proqnozlaşdıra bilsəydiniz, bunanla nə edərdiniz?

Məqsədinizə çatmaq üçün nə vaxt hərəkət edəcəyinizi dəqiq proqnozlaşdırmaq zaman seriyalarının proqnozlaşdırılması ilə həll edilə bilən bir problemdir. Pik saatlarda dayanacaq yeri axtaran insanlardan daha çox ödəniş almaq onları sevindirməz, lakin bu, küçələri təmizləmək üçün gəlir əldə etməyin etibarlı yolu olardı!

Gəlin zaman seriyaları alqoritmlərinin bəzi növlərini araşdıraq və notbuk yaratmaqla bəzi məlumatları təmizləmək və hazırlamağa başlayaq. Təhlil edəcəyiniz data GEFCom2014 proqnozlaşdırma müsabiqəsindən götürülmüşdür. Həmin data, 2012 və 2014-cü illər arasında 3 illik, saatlıq elektrik yükü və temperatur dəyərlərindən ibarətdir. Elektrik yükünün və temperaturun tarixi nümunələrini nəzərə alaraq elektrik yükünün gələcək dəyərlərini proqnozlaşdıra bilərsiniz.

Bu nümunədə siz yalnız keçmiş yükləmə datasından istifadə edərək bir addım irəlini proqnozlaşdırmağı öyrənəcəksiniz. Başlamazdan əvvəl ekran arxasında nələrin baş verdiyini anlamaqda fayda var.

## Bəzi təriflər

"Zaman seriyaları" termini ilə qarşılaşarkən onun bir neçə fərqli kontekstdə istifadəsini başa düşməlisiniz.

🎓 **Zaman seriyaları**

Riyaziyyatda "zaman seriyası vaxt sırasına görə indeksləşdirilmiş(və ya siyahıya alınmış, yaxud da qrafikləşdirilmiş) data nöqtələrinin sırasıdır. Ən çox yayılmış zaman seriyası zamanın ardıcıl bərabər məsafəli nöqtələrində götürülən ardıcıllıqdır." Zaman seriyasına misal olaraq [Dow Jones Sənaye Ortalamasının](https://wikipedia.org/wiki/Time_series) gündəlik bağlanış dəyərini göstərmək olar. Zaman seriyalarının qrafiklərindən və statistik modelləşdirməsindən istifadəyə tez-tez siqnalların işlənməsi, hava proqnozu, zəlzələnin proqnozlaşdırılması və hadisələrin baş verdiyi müddətdə məlumat nöqtələrinin zamanla tərtib oluna biləcəyi digər sahələrdə rast gəlinir.

🎓 **Zaman seriyalarının təhlili**

Zaman seriyalarının təhlili yuxarıda qeyd olunan zaman seriyası məlumatlarının təhlilidir. Zaman seriyası dataları müxtəlif formalarda ola bilər. Onlardan biri fasilə verən hadisədən əvvəl və sonra bir zaman seriyasındakı qanunauyğunluqları aşkarlayan "kəsintili (diskret) zaman seriyası"-dır. Zaman seriyası üçün lazım olan analiz növü məlumatların xarakterindən asılıdır. Məlumatların özü isə nömrələr və ya simvollar seriyası formasında ola bilər.

İcrası nəzərdə tutulan təhlildə tezlik-domen və zaman-domen, xətti və qeyri-xətti və s. daxil olmaqla müxtəlif üsullardan istifadə edilir. Bu tip dataların analiz üsulları barədə ətraflı məlumatı [buradan](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) öyrənə bilərsiniz.

🎓 **Zaman seriyaları proqnozu**

Zaman seriyalarının proqnozlaşdırılması, keçmişdə toplanmış dataların yaratdığı qanunauyğunluğa əsasən gələcək dəyərləri proqnozlaşdırmaq üçün hansısa bir modeldən istifadə etməkdir. Zaman seriyası datalarını tədqiq etmək üçün reqressiya modellərindən istifadə etmək mümkün olsa da, belə məlumatlar xüsusi model tiplərindən istifadə etməklə ən yaxşı şəkildə təhlil edilir.

Zaman seriyası datası xətti reqressiya ilə təhlil edilə bilən məlumatlardan fərqli olaraq ardıcıl müşahidələrin siyahısıdır. Ən çox yayılmışı ARIMA-dır, açılışı "Autoregressive Integrated Moving Average" deməkdir.

[ARIMA modelləri](https://online.stat.psu.edu/stat510/lesson/1/1.1) "seriyanın indiki dəyərini keçmiş dəyərlər və keçmiş proqnoz xətaları ilə əlaqələndirir." Onlar məlumatların zamanla seriyalandığı zaman-domen məlumatlarını təhlil etmək üçün ən uyğun variantdırlar.

> ARIMA modellərinin bir neçə növü var ki, onlar haqqında [burada](https://people.duke.edu/~rnau/411arim.htm) öyrənə bilərsiniz. Növbəti dərsdə onlar barədə öyrənəcəksiniz.

Növbəti dərsdə siz dəyərini zamanla yenilənən bir dəyişənə fokuslanan [Univariate Time Series](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm) istifadə edərək ARIMA modeli quracaqsınız. Bu tip məlumatlara misal olaraq Mauna Loa Rəsədxanasında aylıq karbon qazı konsentrasiyasını qeyd edən [bu verilənlər bazası](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) ola bilər:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Bu data setində zamanla yenilənən dəyişəni müəyyən edin.

## Nəzərə alınmalı Zaman seriyası data xarakteristikaları

Zaman seriyası məlumatlarına baxarkən onların daha yaxşı başa düşmək üçün nəzərə almalı və azaltmalı olduğunuz [müəyyən xüsusiyyətlərə](https://online.stat.psu.edu/stat510/lesson/1/1.1) malik olduğunu görə bilərsiniz. Əgər siz zaman seriyası datalarını analiz etmək istədiyiniz potensial 'sinqal' ötürücüsü olaraq görürsünüzə, bu xüsusiyyətlər “səs-küy” kimi düşünülə bilər. Siz tez-tez müxtəlif statistik üsullardan istifadə edərək həmin xüsusiyyətlərin bəzilərini əvəz etməklə 'səs-küyü' azaltmalı olacaqsınız.

Zaman seriyaları ilə işləyə bilmək üçün bilməli olduğunuz bəzi anlayışlar bunlardır:

🎓 **Trendlər**

Trendlər zamanla ölçülə bilən artımlar və azalmalar kimi tərif edilir. [Ətraflı oxu](https://machinelearningmastery.com/time-series-trends-in-python). Zaman seriyaları kontekstində isə trendlər, onları necə istifadə etmək və lazım gələrsə, zaman seriyalarınızdan silmək haqqındadır.

🎓 **[Mövsümilik](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Mövsümilik dedikdə, məsələn, satışlara təsir edə biləcək bayram ərəfəsi kimi dövri dalğalanmalar nəzərdə tutulur. Fərqli qrafiklərin datada mövsümiliyi necə göstərdiyinə [diqqət yetirin](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm).

🎓 **Kənar göstəricilər**

Kənar göstəricilər standart məlumat fərqindən çox uzaqdır.

🎓 **Uzun müddətli dövr**

Mövsümilikdən asılı olmayaraq məlumatlar bir ildən çox davam edən iqtisadi tənəzzül kimi uzunmüddətli dövrü göstərə bilər.

🎓 **Daimi fərq**

Zaman keçdikcə bəzi məlumatlar gecə və gündüz enerji istifadəsi kimi daimi dalğalanmaları göstərir.

🎓 **Kəskin dəyişikliklər**

Verilənlər əlavə təhlilə ehtiyac duyan qəfil dəyişikliklər göstərə bilər. Məsələn, COVID səbəbiylə müəssisələrin qəfil bağlanması məlumatların dəyişməsinə səbəb oldu.

✅ Burada bir neçə il ərzində, günlük oyundaxili satışları göstərən bir [nümunə zaman seriyası qrafiki](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) verilmişdir. Göstərilən datada yuxarıda sadalanan xüsusiyyətlərdən hansılarını sezə bilərsiniz?

![Oyundaxili satış xərcləri](../images/currency.png)

## Tapşırıq - enerji istifadəsi datasının istifadəsinə başlamaq

Keçmiş istifadəni nəzərə alaraq gələcək enerji istifadəsini proqnozlaşdırmaq üçün zaman seriyası modelini yaratmağa başlayaq.

> Bu nümunədəki datalar GEFCom2014 proqnozlaşdırma müsabiqəsindən götürülmüşdür. 2012-2014-cü illər arasında 3 illik, saatlıq elektrik yükü və temperatur dəyərlərindən ibarətdir.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli və Rob J. Hyndman, "Ehtimallı enerji proqnozlaşdırılması: Qlobal Enerji Proqnozlaşdırma Müsabiqəsi 2014 və sonrası", Beynəlxalq Proqnozlaşdırma Jurnalı, cild 32, №3, səh 896 -913, iyul-sentyabr, 2016.

1. Bu dərsin `working` qovluğundakı _notebook.ipynb_ faylını açın. Dataları yükləməyə və vizuallaşdırmağa kömək edəcək kitabxanalar əlavə etməklə başlayın

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Nəzərə alın ki, sizin tərtibat mühitinizi quran və məlumatların endirilməsini idarə edən `common` qovluğundakı fayllardan istifadə edirsiniz.

2. Növbəti addım olaraq, `load_data()` və `head()` funksiyalarını çağıraraq dataları datafreym olaraq yoxlayın:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Tarixi və yükü təmsil edən iki sütun olduğunu görə bilərsiniz:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. İndi isə `plot()` funksiyasını çağıraraq datanın qrafikini qurun:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![enerji qrafiki](../images/energy-plot.png)

4. İndi 2014-cü ilin iyul ayının ilk həftəsini, `[tarixdən]: [tarixə]` formasında `energy`-ə giriş dəyişəni formasında təqdim edərək qrafiki qurun:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![iyul](../images/july-2014.png)

    Gözəl qrafik alındı! Bu qrafiklərə nəzər yetirin və yuxarıda sadalanan xüsusiyyətlərdən hər hansı birini müəyyən edə bildiyinizə baxın. Verilənləri vizuallaşdırmaqla nəyi təxmin edə bilərik?

Növbəti dərsdə bəzi proqnozlar yaratmaq üçün ARIMA modeli yaradacaqsınız.

---

## 🚀 Məşğələ

Zaman seriyalarının proqnozlaşdırılmasından faydalanacağını düşünə biləcəyiniz bütün sənaye və araşdırma sahələrinin siyahısını tərtib edin. Bu texnikaların incəsənətdə tətbiqi barədə düşünə bilərsinizmi? Ekonometrikada? Ekologiya? Pərakəndə satış? Sənaye? Maliyyə? Başqa harada?

## [Mühazirə sonrası test](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42/?loc=az)

## Təkrarlayın və özünüz öyrənin

Onları burada əhatə etməsək də, neyroşəbəkələr bəzən zaman seriyalarının proqnozlaşdırılmasının klassik üsullarını təkmilləşdirmək üçün istifadə olunur. Onlar haqqında daha ətraflı [bu məqalədə](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412) oxuyun.

## Tapşırıq

[Daha bir neçə zaman seriyasını vizuallaşdırın](assignment.az.md)