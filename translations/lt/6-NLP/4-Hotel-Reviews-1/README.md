<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T08:05:02+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "lt"
}
-->
# Sentimentų analizė su viešbučių apžvalgomis - duomenų apdorojimas

Šiame skyriuje naudosite ankstesnėse pamokose išmoktas technikas, kad atliktumėte didelio duomenų rinkinio tyrimą. Kai gerai suprasite įvairių stulpelių naudingumą, išmoksite:

- kaip pašalinti nereikalingus stulpelius
- kaip apskaičiuoti naujus duomenis remiantis esamais stulpeliais
- kaip išsaugoti gautą duomenų rinkinį, kad galėtumėte jį naudoti galutiniame iššūkyje

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

### Įvadas

Iki šiol sužinojote, kad tekstiniai duomenys labai skiriasi nuo skaitinių duomenų tipų. Jei tekstą parašė ar pasakė žmogus, jį galima analizuoti, kad būtų nustatyti modeliai, dažniai, sentimentai ir prasmė. Ši pamoka nukelia jus į realų duomenų rinkinį su realiu iššūkiu: **[515K viešbučių apžvalgos Europoje](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, kuris turi [CC0: viešosios domeno licenciją](https://creativecommons.org/publicdomain/zero/1.0/). Duomenys buvo surinkti iš Booking.com viešų šaltinių. Duomenų rinkinio kūrėjas yra Jiashen Liu.

### Pasiruošimas

Jums reikės:

* Galimybės paleisti .ipynb užrašų knygeles naudojant Python 3
* pandas
* NLTK, [kurį turėtumėte įdiegti lokaliai](https://www.nltk.org/install.html)
* Duomenų rinkinio, kuris yra prieinamas Kaggle [515K viešbučių apžvalgos Europoje](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Jo dydis yra apie 230 MB išpakuotas. Atsisiųskite jį į šaknies `/data` aplanką, susijusį su šiomis NLP pamokomis.

## Duomenų tyrimas

Šis iššūkis daro prielaidą, kad kuriate viešbučių rekomendacijų botą, naudodami sentimentų analizę ir svečių apžvalgų įvertinimus. Duomenų rinkinys, kurį naudosite, apima 1493 skirtingų viešbučių apžvalgas iš 6 miestų.

Naudodami Python, viešbučių apžvalgų duomenų rinkinį ir NLTK sentimentų analizę, galite sužinoti:

* Kokie yra dažniausiai naudojami žodžiai ir frazės apžvalgose?
* Ar oficialios *žymos*, apibūdinančios viešbutį, koreliuoja su apžvalgų įvertinimais (pvz., ar tam tikro viešbučio *Šeima su mažais vaikais* apžvalgos yra labiau neigiamos nei *Kelionė vienam*, galbūt nurodant, kad jis geriau tinka *Kelionėms vienam*)?
* Ar NLTK sentimentų įvertinimai „sutampa“ su viešbučio apžvalgininko skaitiniu įvertinimu?

#### Duomenų rinkinys

Išnagrinėkime duomenų rinkinį, kurį atsisiuntėte ir išsaugojote lokaliai. Atidarykite failą redaktoriuje, pvz., VS Code arba net Excel.

Duomenų rinkinio antraštės yra tokios:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Čia jos suskirstytos į grupes, kad būtų lengviau jas analizuoti:  
##### Viešbučių stulpeliai

* `Hotel_Name`, `Hotel_Address`, `lat` (platuma), `lng` (ilguma)
  * Naudodami *lat* ir *lng* galite sukurti žemėlapį su Python, rodantį viešbučių vietas (galbūt spalvotai koduotas pagal neigiamas ir teigiamas apžvalgas)
  * Hotel_Address nėra akivaizdžiai naudingas mums, ir greičiausiai jį pakeisime šalimi, kad būtų lengviau rūšiuoti ir ieškoti

**Viešbučių meta-apžvalgų stulpeliai**

* `Average_Score`
  * Pasak duomenų rinkinio kūrėjo, šis stulpelis yra *Vidutinis viešbučio įvertinimas, apskaičiuotas remiantis naujausiu komentaru per pastaruosius metus*. Tai atrodo neįprastas būdas apskaičiuoti įvertinimą, tačiau tai yra surinkti duomenys, todėl kol kas galime juos priimti kaip faktą. 
  
  ✅ Remiantis kitais šio duomenų rinkinio stulpeliais, ar galite sugalvoti kitą būdą apskaičiuoti vidutinį įvertinimą?

* `Total_Number_of_Reviews`
  * Bendras apžvalgų skaičius, kurį šis viešbutis gavo - nėra aišku (be kodo rašymo), ar tai reiškia apžvalgas duomenų rinkinyje.
* `Additional_Number_of_Scoring`
  * Tai reiškia, kad buvo pateiktas apžvalgos įvertinimas, bet apžvalgininkas neparašė teigiamos ar neigiamos apžvalgos

**Apžvalgų stulpeliai**

- `Reviewer_Score`
  - Tai skaitinė reikšmė su daugiausia 1 dešimtainiu skaičiumi tarp minimalių ir maksimalių reikšmių 2.5 ir 10
  - Nėra paaiškinta, kodėl mažiausias galimas įvertinimas yra 2.5
- `Negative_Review`
  - Jei apžvalgininkas nieko neparašė, šiame lauke bus "**No Negative**"
  - Atkreipkite dėmesį, kad apžvalgininkas gali parašyti teigiamą apžvalgą neigiamų apžvalgų stulpelyje (pvz., "šiame viešbutyje nėra nieko blogo")
- `Review_Total_Negative_Word_Counts`
  - Didesnis neigiamų žodžių skaičius rodo žemesnį įvertinimą (neatsižvelgiant į sentimentus)
- `Positive_Review`
  - Jei apžvalgininkas nieko neparašė, šiame lauke bus "**No Positive**"
  - Atkreipkite dėmesį, kad apžvalgininkas gali parašyti neigiamą apžvalgą teigiamų apžvalgų stulpelyje (pvz., "šiame viešbutyje nėra nieko gero")
- `Review_Total_Positive_Word_Counts`
  - Didesnis teigiamų žodžių skaičius rodo aukštesnį įvertinimą (neatsižvelgiant į sentimentus)
- `Review_Date` ir `days_since_review`
  - Galima taikyti šviežumo ar senumo matą apžvalgai (senesnės apžvalgos gali būti ne tokios tikslios kaip naujesnės, nes viešbučio valdymas pasikeitė, buvo atlikti renovacijos darbai, pridėtas baseinas ir pan.)
- `Tags`
  - Tai trumpi aprašymai, kuriuos apžvalgininkas gali pasirinkti, kad apibūdintų, kokio tipo svečias jis buvo (pvz., vienas ar šeima), kokio tipo kambarį turėjo, kiek laiko viešėjo ir kaip pateikė apžvalgą. 
  - Deja, šių žymų naudojimas yra problematiškas, žr. skyrių žemiau, kuriame aptariamas jų naudingumas

**Apžvalgininkų stulpeliai**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Tai gali būti veiksnys rekomendacijų modelyje, pavyzdžiui, jei galėtumėte nustatyti, kad produktyvesni apžvalgininkai, turintys šimtus apžvalgų, dažniau būna neigiami nei teigiami. Tačiau bet kurioje konkrečioje apžvalgoje apžvalgininkas nėra identifikuojamas unikaliu kodu, todėl jo negalima susieti su apžvalgų rinkiniu. Yra 30 apžvalgininkų, turinčių 100 ar daugiau apžvalgų, tačiau sunku suprasti, kaip tai gali padėti rekomendacijų modeliui.
- `Reviewer_Nationality`
  - Kai kurie žmonės gali manyti, kad tam tikros tautybės yra labiau linkusios pateikti teigiamą ar neigiamą apžvalgą dėl nacionalinio polinkio. Būkite atsargūs, kurdami tokius anekdotinius požiūrius į savo modelius. Tai yra nacionaliniai (ir kartais rasiniai) stereotipai, o kiekvienas apžvalgininkas buvo individas, kuris rašė apžvalgą remdamasis savo patirtimi. Ji galėjo būti filtruota per daugelį lęšių, tokių kaip ankstesni viešbučių apsilankymai, kelionės atstumas ir asmeninis temperamentas. Manyti, kad jų tautybė buvo apžvalgos įvertinimo priežastis, yra sunku pateisinti.

##### Pavyzdžiai

| Vidutinis Įvertinimas | Bendras Apžvalgų Skaičius | Apžvalgininko Įvertinimas | Neigiama <br />Apžvalga                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Teigiama Apžvalga                 | Žymos                                                                                      |
| --------------------- | ------------------------ | ------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8                  | 1945                    | 2.5                      | Šiuo metu tai nėra viešbutis, o statybvietė. Buvau terorizuotas nuo ankstyvo ryto ir visą dieną nepriimtinu statybų triukšmu, ilsėdamasis po ilgos kelionės ir dirbdamas kambaryje. Žmonės dirbo visą dieną, t. y. su gręžtuvais gretimuose kambariuose. Paprašiau pakeisti kambarį, bet tylus kambarys nebuvo prieinamas. Dar blogiau, man buvo per daug apmokestinta. Išsiregistravau vakare, nes turėjau ankstyvą skrydį ir gavau tinkamą sąskaitą. Kitą dieną viešbutis be mano sutikimo padarė dar vieną mokestį, viršijantį užsakymo kainą. Tai siaubinga vieta. Nepunykite savęs užsisakydami čia. | Nieko. Siaubinga vieta. Venkite. | Verslo kelionė, Pora, Standartinis dvivietis kambarys, Viešėjo 2 naktis |

Kaip matote, šis svečias neturėjo laimingos viešnagės šiame viešbutyje. Viešbutis turi gerą vidutinį įvertinimą - 7.8 ir 1945 apžvalgas, tačiau šis apžvalgininkas suteikė jam 2.5 ir parašė 115 žodžių apie tai, kaip neigiama buvo jų viešnagė. Jei jie nieko neparašė teigiamų apžvalgų stulpelyje, galite manyti, kad nebuvo nieko teigiamo, tačiau jie parašė 7 įspėjimo žodžius. Jei tiesiog skaičiuotume žodžius, o ne jų prasmę ar sentimentus, galėtume turėti iškreiptą apžvalgininko ketinimų vaizdą. Keista, jų įvertinimas 2.5 yra painus, nes jei viešnagė buvo tokia bloga, kodėl suteikti bet kokius taškus? Išnagrinėjus duomenų rinkinį atidžiai, matysite, kad mažiausias galimas įvertinimas yra 2.5, o ne 0. Didžiausias galimas įvertinimas yra 10.

##### Žymos

Kaip minėta aukščiau, iš pirmo žvilgsnio idėja naudoti `Tags` duomenims kategorizuoti atrodo prasminga. Deja, šios žymos nėra standartizuotos, o tai reiškia, kad tam tikrame viešbutyje pasirinkimai gali būti *Vienvietis kambarys*, *Dviejų lovų kambarys* ir *Dvivietis kambarys*, tačiau kitame viešbutyje jie yra *Deluxe vienvietis kambarys*, *Klasikinis karalienės kambarys* ir *Executive karaliaus kambarys*. Tai gali būti tie patys dalykai, tačiau yra tiek daug variantų, kad pasirinkimas tampa:

1. Bandymas pakeisti visus terminus į vieną standartą, kuris yra labai sunkus, nes neaišku, koks būtų konversijos kelias kiekvienu atveju (pvz., *Klasikinis vienvietis kambarys* atitinka *Vienvietis kambarys*, tačiau *Superior Queen Room with Courtyard Garden or City View* yra daug sunkiau susieti)

1. Galime taikyti NLP metodą ir matuoti tam tikrų terminų, pvz., *Solo*, *Verslo keliautojas* arba *Šeima su mažais vaikais*, dažnį, kaip jie taikomi kiekvienam viešbučiui, ir įtraukti tai į rekomendaciją  

Žymos paprastai (bet ne visada) yra vienas laukas, kuriame yra 5–6 kableliais atskirtos reikšmės, atitinkančios *Kelionės tipą*, *Svečių tipą*, *Kambario tipą*, *Naktų skaičių* ir *Įrenginį, kuriuo pateikta apžvalga*. Tačiau kadangi kai kurie apžvalgininkai neužpildo kiekvieno lauko (jie gali palikti vieną tuščią), reikšmės ne visada yra ta pačia tvarka.

Pavyzdžiui, paimkime *Grupės tipą*. Šiame lauke `Tags` stulpelyje yra 1025 unikalios galimybės, ir, deja, tik kai kurios iš jų nurodo grupę (kai kurios yra kambario tipas ir pan.). Jei filtruojate tik tuos, kurie mini šeimą, rezultatai apima daugybę *Šeimos kambario* tipo rezultatų. Jei įtraukiate terminą *su*, t. y. skaičiuojate *Šeima su* reikšmes, rezultatai yra geresni, nes daugiau nei 80 000 iš 515 000 rezultatų turi frazę "Šeima su mažais vaikais" arba "Šeima su vyresniais vaikais".

Tai reiškia, kad žymų stulpelis nėra visiškai nenaudingas mums, tačiau reikės šiek tiek darbo, kad jis būtų naudingas.

##### Vidutinis viešbučio įvertinimas

Duomenų rinkinyje yra keletas keistenybių ar neatitikimų, kurių negaliu suprasti, tačiau jie iliustruojami čia, kad būtumėte informuoti apie juos, kai kuriate savo modelius. Jei suprasite, prašome pranešti mums diskusijų skyriuje!

Duomenų rinkinyje yra šie stulpeliai, susiję su vidutiniu įvertinimu ir apžvalgų skaičiumi:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Viešbutis, turintis daugiausiai apžvalgų šiame duomenų rinkinyje, yra *Britannia International Hotel Canary Wharf* su 4789 apžvalgomis iš 515 000. Tačiau jei pažvelgsime į `Total_Number_of_Reviews` reikšmę šiam viešbučiui, ji yra 9086. Galite manyti, kad yra daug daugiau įvertinimų be apžvalgų, todėl galbūt turėtume pridėti `Additional_Number_of_Scoring` stulpelio reikšmę. Ta reikšmė yra 2682, ir pridėjus ją prie 4789 gauname 7471, kuris vis dar yra 1615 mažesnis nei `Total_Number_of_Reviews`.

Jei paimsite `Average_Score` stulpelį, galite manyti, kad tai yra vidurkis apžvalgų duomenų rinkinyje, tačiau Kaggle aprašymas yra "*Vidutinis viešbučio įvertinimas, apskaičiuotas remiantis naujausiu komentaru per pastaruosius metus*". Tai neatrodo labai naudinga, tačiau galime apskaičiuoti savo vidurk
> 🚨 Pastaba dėl atsargumo
>
> Dirbant su šiuo duomenų rinkiniu, jūs rašysite kodą, kuris apskaičiuos kažką iš teksto, nereikalaujant, kad jūs patys skaitytumėte ar analizuotumėte tekstą. Tai yra NLP esmė – interpretuoti prasmę ar nuotaiką, nereikalaujant žmogaus įsikišimo. Tačiau gali būti, kad jūs perskaitysite kai kurias neigiamas apžvalgas. Rekomenduoju to nedaryti, nes to nereikia. Kai kurios iš jų yra kvailos arba nesvarbios neigiamos viešbučių apžvalgos, pavyzdžiui, „Oras nebuvo geras“, kas yra už viešbučio ar bet kieno kontrolės ribų. Tačiau kai kurios apžvalgos turi ir tamsiąją pusę. Kartais neigiamos apžvalgos yra rasistinės, seksistinės ar diskriminuojančios pagal amžių. Tai yra apmaudu, bet tikėtina, kai duomenų rinkinys surinktas iš viešos svetainės. Kai kurie apžvalgininkai palieka atsiliepimus, kurie gali būti nemalonūs, nepatogūs ar sukelti neigiamas emocijas. Geriau leisti kodui įvertinti nuotaiką, nei skaityti juos pačiam ir jaustis prastai. Vis dėlto, tokių apžvalgų yra mažuma, tačiau jos vis tiek egzistuoja.
## Pratimai - Duomenų tyrinėjimas
### Duomenų įkėlimas

Užteks vizualiai nagrinėti duomenis, dabar parašysite šiek tiek kodo ir gausite atsakymus! Šiame skyriuje naudojama pandas biblioteka. Pirmoji užduotis – įsitikinti, kad galite įkelti ir perskaityti CSV duomenis. Pandas biblioteka turi greitą CSV įkėlimo funkciją, o rezultatas patalpinamas į duomenų rėmelį, kaip ir ankstesnėse pamokose. CSV, kurį įkeliame, turi daugiau nei pusę milijono eilučių, bet tik 17 stulpelių. Pandas suteikia daug galingų būdų sąveikauti su duomenų rėmeliu, įskaitant galimybę atlikti operacijas kiekvienoje eilutėje.

Nuo šios pamokos dalies bus pateikiami kodo fragmentai, paaiškinimai apie kodą ir diskusijos apie tai, ką reiškia rezultatai. Naudokite pridėtą _notebook.ipynb_ savo kodui.

Pradėkime nuo duomenų failo įkėlimo, kurį naudosite:

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

Kai duomenys jau įkelti, galime atlikti tam tikras operacijas su jais. Laikykite šį kodą programos viršuje kitai daliai.

## Duomenų tyrinėjimas

Šiuo atveju duomenys jau yra *švarūs*, tai reiškia, kad jie paruošti darbui ir neturi simbolių kitomis kalbomis, kurie galėtų sutrikdyti algoritmus, tikinčius, kad duomenys yra tik anglų kalba.

✅ Gali tekti dirbti su duomenimis, kuriems reikalingas pradinis apdorojimas prieš taikant NLP technikas, bet ne šį kartą. Jei reikėtų, kaip tvarkytumėte ne angliškus simbolius?

Skirkite akimirką, kad įsitikintumėte, jog įkėlus duomenis galite juos tyrinėti naudodami kodą. Labai lengva susitelkti į `Negative_Review` ir `Positive_Review` stulpelius. Jie užpildyti natūraliu tekstu, kurį apdoros jūsų NLP algoritmai. Bet palaukite! Prieš pasinerdami į NLP ir sentimentų analizę, turėtumėte sekti žemiau pateiktą kodą, kad patikrintumėte, ar duomenų rinkinyje pateiktos reikšmės atitinka tas, kurias apskaičiuojate naudodami pandas.

## Duomenų rėmelio operacijos

Pirmoji užduotis šioje pamokoje yra patikrinti, ar šie teiginiai teisingi, parašant kodą, kuris nagrinėja duomenų rėmelį (jo nekeisdami).

> Kaip ir daugelis programavimo užduočių, yra keletas būdų tai atlikti, tačiau gera praktika yra tai daryti kuo paprasčiau ir lengviau, ypač jei tai bus lengviau suprasti, kai vėliau grįšite prie šio kodo. Dirbant su duomenų rėmeliais, yra išsamus API, kuris dažnai turi efektyvų būdą atlikti tai, ko norite.

Traktuokite šiuos klausimus kaip programavimo užduotis ir pabandykite atsakyti į juos nesinaudodami sprendimu.

1. Išspausdinkite duomenų rėmelio *formą* (forma – tai eilučių ir stulpelių skaičius).
2. Apskaičiuokite apžvalgininkų tautybių dažnio skaičiavimą:
   1. Kiek skirtingų reikšmių yra stulpelyje `Reviewer_Nationality` ir kokios jos?
   2. Kokia apžvalgininkų tautybė yra dažniausia duomenų rinkinyje (išspausdinkite šalį ir apžvalgų skaičių)?
   3. Kokios yra kitos 10 dažniausiai pasitaikančių tautybių ir jų dažnio skaičiavimas?
3. Koks buvo dažniausiai apžvelgtas viešbutis kiekvienai iš 10 dažniausiai pasitaikančių apžvalgininkų tautybių?
4. Kiek apžvalgų yra kiekvienam viešbučiui (viešbučių dažnio skaičiavimas) duomenų rinkinyje?
5. Nors duomenų rinkinyje yra stulpelis `Average_Score` kiekvienam viešbučiui, taip pat galite apskaičiuoti vidutinį balą (apskaičiuodami visų apžvalgininkų balų vidurkį duomenų rinkinyje kiekvienam viešbučiui). Pridėkite naują stulpelį prie savo duomenų rėmelio su stulpelio pavadinimu `Calc_Average_Score`, kuriame yra apskaičiuotas vidurkis.
6. Ar yra viešbučių, kurių (suapvalinus iki 1 dešimtainės vietos) `Average_Score` ir `Calc_Average_Score` yra vienodi?
   1. Pabandykite parašyti Python funkciją, kuri priima Series (eilutę) kaip argumentą ir palygina reikšmes, išspausdindama pranešimą, kai reikšmės nesutampa. Tada naudokite `.apply()` metodą, kad apdorotumėte kiekvieną eilutę su funkcija.
7. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Negative_Review` reikšmes "No Negative".
8. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Positive_Review` reikšmes "No Positive".
9. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Positive_Review` reikšmes "No Positive" **ir** stulpelio `Negative_Review` reikšmes "No Negative".

### Kodo atsakymai

1. Išspausdinkite duomenų rėmelio *formą* (forma – tai eilučių ir stulpelių skaičius).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Apskaičiuokite apžvalgininkų tautybių dažnio skaičiavimą:

   1. Kiek skirtingų reikšmių yra stulpelyje `Reviewer_Nationality` ir kokios jos?
   2. Kokia apžvalgininkų tautybė yra dažniausia duomenų rinkinyje (išspausdinkite šalį ir apžvalgų skaičių)?

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

   3. Kokios yra kitos 10 dažniausiai pasitaikančių tautybių ir jų dažnio skaičiavimas?

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

3. Koks buvo dažniausiai apžvelgtas viešbutis kiekvienai iš 10 dažniausiai pasitaikančių apžvalgininkų tautybių?

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

4. Kiek apžvalgų yra kiekvienam viešbučiui (viešbučių dažnio skaičiavimas) duomenų rinkinyje?

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
   
   Galite pastebėti, kad *skaičiuojami duomenų rinkinyje* rezultatai nesutampa su `Total_Number_of_Reviews` reikšme. Neaišku, ar ši reikšmė duomenų rinkinyje atspindėjo bendrą viešbučio apžvalgų skaičių, bet ne visos buvo surinktos, ar tai buvo kitas skaičiavimas. `Total_Number_of_Reviews` nėra naudojamas modelyje dėl šio neaiškumo.

5. Nors duomenų rinkinyje yra stulpelis `Average_Score` kiekvienam viešbučiui, taip pat galite apskaičiuoti vidutinį balą (apskaičiuodami visų apžvalgininkų balų vidurkį duomenų rinkinyje kiekvienam viešbučiui). Pridėkite naują stulpelį prie savo duomenų rėmelio su stulpelio pavadinimu `Calc_Average_Score`, kuriame yra apskaičiuotas vidurkis. Išspausdinkite stulpelius `Hotel_Name`, `Average_Score` ir `Calc_Average_Score`.

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

   Galite taip pat stebėtis `Average_Score` reikšme ir kodėl ji kartais skiriasi nuo apskaičiuoto vidutinio balo. Kadangi negalime žinoti, kodėl kai kurios reikšmės sutampa, o kitos turi skirtumą, saugiausia šiuo atveju naudoti apžvalgų balus, kuriuos turime, kad patys apskaičiuotume vidurkį. Vis dėlto skirtumai paprastai yra labai maži, štai viešbučiai su didžiausiu nukrypimu nuo duomenų rinkinio vidurkio ir apskaičiuoto vidurkio:

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

   Kadangi tik 1 viešbutis turi balų skirtumą, didesnį nei 1, tai reiškia, kad galime ignoruoti skirtumą ir naudoti apskaičiuotą vidutinį balą.

6. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Negative_Review` reikšmes "No Negative".

7. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Positive_Review` reikšmes "No Positive".

8. Apskaičiuokite ir išspausdinkite, kiek eilučių turi stulpelio `Positive_Review` reikšmes "No Positive" **ir** stulpelio `Negative_Review` reikšmes "No Negative".

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

## Kitas būdas

Kitas būdas skaičiuoti elementus be Lambdas ir naudoti sumą eilučių skaičiavimui:

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

   Galbūt pastebėjote, kad yra 127 eilutės, kurios turi tiek "No Negative", tiek "No Positive" reikšmes stulpeliuose `Negative_Review` ir `Positive_Review`. Tai reiškia, kad apžvalgininkas suteikė viešbučiui skaitinį balą, bet atsisakė rašyti tiek teigiamą, tiek neigiamą apžvalgą. Laimei, tai yra nedidelis eilučių skaičius (127 iš 515738, arba 0,02%), todėl tai greičiausiai nesukels modelio ar rezultatų iškraipymo, tačiau galbūt nesitikėjote, kad apžvalgų duomenų rinkinyje bus eilučių be apžvalgų, todėl verta tyrinėti duomenis, kad atrastumėte tokias eilutes.

Dabar, kai ištyrėte duomenų rinkinį, kitoje pamokoje filtruosite duomenis ir pridėsite sentimentų analizę.

---
## 🚀Iššūkis

Ši pamoka parodo, kaip matėme ankstesnėse pamokose, kaip svarbu kritiškai suprasti savo duomenis ir jų ypatybes prieš atliekant operacijas su jais. Teksto pagrindu sukurti duomenys, ypač, reikalauja atidaus nagrinėjimo. Peržiūrėkite įvairius tekstui skirtus duomenų rinkinius ir pažiūrėkite, ar galite atrasti sritis, kurios galėtų įvesti šališkumą ar iškraipytą sentimentą į modelį.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Pasinaudokite [šiuo NLP mokymosi keliu](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), kad atrastumėte įrankius, kuriuos galite išbandyti kurdami kalbos ir teksto pagrindu sukurtus modelius.

## Užduotis

[NLTK](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, atkreipiame dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudotis profesionalių vertėjų paslaugomis. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus aiškinimus, kylančius dėl šio vertimo naudojimo.