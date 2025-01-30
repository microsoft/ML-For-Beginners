# Daha Real Dünya

Bizim vəziyyətimizdə Piter, demək olar ki, yorulmadan və ac qalmadan hərəkət edə bilirdi. Daha realist bir dünyada isə vaxt aşırı oturub dincəlməli, həm də qidalanmalı oluruq. Aşağıdakı qaydaları həyata keçirməklə dünyamızı daha real edək:

1. Bir yerdən başqa yerə köçməklə Piter **enerjisini** itirir və bir qədər **yorğunluq** qazanır.
2. Piter alma yeməklə daha çox enerji qazana bilər.
3. Piter ağacın altında və ya çəmənlikdə dincəlməklə yorğunluqdan qurtula bilər (yəni ağac və ya ot olan bir xanaya getmək - yaşıl sahə)
4. Piter canavarı tapıb öldürməlidir
5. Canavarı öldürmək üçün Piterin müəyyən enerji və yorğunluq səviyyəsinə malik olması lazımdır, əks halda o, döyüşü uduzur.

## Təlimatlar

Həlliniz üçün başlanğıc nöqtəsi kimi orijinal [notebook.ipynb](../notebook.ipynb) notbukundan istifadə edin.

Yuxarıdakı mükafat funksiyasını oyunun qaydalarına uyğun olaraq dəyişdirin. Oyunda qalib gəlmək üçün lazım olan ən yaxşı strategiyanı öyrənmək üçün isə gücləndirici öyrənmə alqoritmini işlədin və təsadüfi gedişin nəticələrini qazandığınız və itirdiyiniz oyunların sayına görə alqoritminizlə müqayisə edin.

> **Qeyd**: Yeni dünyanızda insanın mövqeyindən əlavə yorğunluq və enerji səviyyələri də olduğuna görə vəziyyət daha mürəkkəbdir. Siz vəziyyəti bir qrup kimi təqdim etməyi seçə bilərsiniz(Board, enerji, yorğunluq) və ya vəziyyət üçün bir sinif yarada bilə(onu `Board`-dan da alt sinif olaraq yaratmağınız mümkündür) və ya [rlboard.py](../rlboard.py) içərisindəki orijinal `Board` sinfini dəyişdirə bilərsiniz.

Həllinizdə, lütfən, kodu təsadüfi gediş strategiyasına cavab verəcək formada saxlayın və ən sonda alqoritminizin nəticələrini təsadüfi gedişlə müqayisə edin.

> **Qeyd**: Onun işləməsi üçün hiperparametrləri, xüsusən də dövrlərin sayını tənzimləməlisiniz. Oyunun uğuru(canavarla mübarizə) nadir bir hadisə olduğundan daha uzun məşq vaxtı gözləmək olar.

## Rubrika

| Meyarlar | Nümunəvi | Adekvat | İnkişaf Etdirilməli Olan |
| -------- | -------- | ------- | ------------------------ |
|          | Yeni dünya qaydalarının tərifi, Q-Öyrənməsi alqoritmi və bəzi mətn izahatları ilə notebook təqdim olunub. Q-Öyrənməsi təsadüfi gedişlə müqayisədə nəticələri əhəmiyyətli dərəcədə yaxşılaşdıra bilib. | Notbuk təqdim olunub, Q-Öyrənməsi həyata keçirilib və təsadüfi gedişlə müqayisədə nəticələri yaxşılaşdırsa, əhəmiyyətli dərəcədə deyil; və ya notbuk zəif sənədləşdirilib və kod yaxşı strukturlaşdırılmayıb. | Dünyanın qaydalarını yenidən müəyyən etmək üçün bəzi cəhdlər edilib, lakin Q-Öyrənməsi alqoritmi işləmir və ya mükafat funksiyası tam müəyyən edilməyib. |