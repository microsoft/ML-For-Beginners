# Dağ maşınını öyrədin

[Openai Gym](http://gym.openai.com),elə hazırlanıb ki, bütün mühitlər eyni API ilə bizi təmin edir - məsələn, `reset`, `step` və `render` kimi eyni metodlar, **fəaliyyət sahəsi** və **müşahidə sahəsinin** eyni abstrakt formaları. Beləliklə minimal kod dəyişiklikləri ilə eyni gücləndirmə öyrənməsi alqoritmlərini fərqli mühitlərə uyğunlaşdırmaq mümkündür.

## Dağ maşını mühiti

[Dağ maşını mühiti](https://gym.openai.com/envs/mountaincar-v0/) bir dərədə ilişib qalmış bir avtomobilin olduğu mühitdir:

<img src="../images/mountaincar.png" width="300"/>

Məqsəd hər addımda aşağıdakı hərəkətlərdən birini icra edərək vadidən çıxmaq və bayrağı tutmaqdır:

| Dəyər | Mənası                    |
| ----- | ------------------------- |
| 0     | Sola doğru sürəti artırın |
| 1     | Sürətlənməyin             |
| 2     | Sağa doğru sürəti artırın |

Bu problemin əsas hiyləsi avtomobilin mühərrikinin bir təkanda təpəni aşmaq üçün kifayət qədər güclü olmamasıdır. Buna görə də müvəffəq olmağın yeganə yolu təcil yaratmaq üçün geri və irəli sürməkdir.

Müşahidə məkanı yalnız iki dəyərdən ibarətdir:

| Nömrə | Müşahidə            | Min   | Max  |
| ----- | --------------------| ------| ---- |
| 0     | Avtomobilin mövqeyi | -1.2  | 0.6  |
| 1     | Avtomobilin sürəti  | -0.07 | 0.07 |

Dağ avtomobili üçün mükafat sistemi olduqca çətindir:

 * Agentə dağın üstündəki bayrağa(mövqeyi = 0.5) çatdığı təqdirdə 0 mükafat verilir.
 * Agentin mövqeyi 0,5-dən az olduqda -1 ilə mükafatlandırılır.

Gedişat avtomobil mövqeyi 0,5-dən çox olduqda və ya icra uzunluğu 200-dən çox olarsa, sona çatır.

## Təlimat

Dağ avtomobil problemini həll etmək üçün gücləndirmə öyrənməsi alqoritmimizi indiki problemə uyğunlaşdırın. Mövcud [notebook.ipynb](../notebook.ipynb) kodu ilə başlayın, cari mühiti yeni mühitlə əvəz edin, vəziyyətin diskretləşdirilməsi funksiyalarını dəyişdirin və minimal kod dəyişiklikləri ilə təlim keçmək üçün lazım olan alqoritmi qurmağa çalışın. Hiperparameterləri tənzimləməklə nəticəni optimallaşdırın.

> **Qeyd**: Alqoritmin yaxınlaşmasını təmin etmək üçün hiperparametrləri sazlamaq lazım ola bilər.

## Rubrika

| Meyarlar | Nümunəvi | Adekvat | İnkişaf Etdirilməli Olan |
| -------- | -------- | ------- | ------------------------ |
|          | Q-Learning alqoritmi minimal kod dəyişiklikləri ilə CartPole nümunəsi uğurla uyğunlaşdırılıb və 200-dən az addımla bayrağın tutulması problemi həll edilib. | Q-Öyrənməsi alqoritmi internetdən götürülüb, amma yaxşı sənədləşdirilib; və ya mövcud alqoritm qəbul edilmişdir, amma istənilən nəticələr əldə edilməmişdir. | Tələbə heç bir alqoritmi uğurla uyğunlaşdıra bilməyib, amma həlli istiqamətində əhəmiyyətli addımlar atıb(vəziyyətin diskretləşməsini tətbiq edib, Q-Cədvəl data strukturunu qurub və s.) |