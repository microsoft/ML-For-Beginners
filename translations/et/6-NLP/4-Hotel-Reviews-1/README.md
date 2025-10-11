<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-10-11T11:35:04+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "et"
}
-->
# Sentimentianalüüs hotelliarvustustega - andmete töötlemine

Selles osas kasutad eelmistes tundides õpitud tehnikaid, et teha suure andmestiku uurivat andmeanalüüsi. Kui oled saanud hea ülevaate erinevate veergude kasulikkusest, õpid:

- kuidas eemaldada mittevajalikud veerud
- kuidas arvutada uusi andmeid olemasolevate veergude põhjal
- kuidas salvestada tulemuseks saadud andmestik, et seda kasutada lõppväljakutses

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

### Sissejuhatus

Siiani oled õppinud, et tekstandmed erinevad oluliselt numbrilistest andmetest. Kui tekst on inimese kirjutatud või räägitud, saab seda analüüsida, et leida mustreid ja sagedusi, sentimenti ja tähendust. See tund viib sind reaalse andmestiku ja väljakutse juurde: **[515K hotelliarvustuste andmed Euroopas](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, millel on [CC0: Public Domain litsents](https://creativecommons.org/publicdomain/zero/1.0/). Andmed on kogutud Booking.com-i avalikest allikatest. Andmestiku looja on Jiashen Liu.

### Ettevalmistus

Sul on vaja:

* Võimalust käivitada .ipynb märkmikke Python 3 abil
* pandas
* NLTK, [mille peaksid kohapeal installima](https://www.nltk.org/install.html)
* Andmestikku, mis on saadaval Kaggle'is [515K hotelliarvustuste andmed Euroopas](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Selle lahtipakkimata suurus on umbes 230 MB. Laadi see alla NLP tundidega seotud juurkataloogi `/data`.

## Uuriv andmeanalüüs

Selles väljakutses eeldatakse, et ehitad hotellisoovituste boti, kasutades sentimentianalüüsi ja külaliste arvustuste hindeid. Kasutatav andmestik sisaldab 1493 erineva hotelli arvustusi 6 linnas.

Pythonit, hotelliarvustuste andmestikku ja NLTK sentimentianalüüsi kasutades võiksid välja selgitada:

* Millised on arvustustes kõige sagedamini kasutatavad sõnad ja fraasid?
* Kas hotelli ametlikud *sildid* korreleeruvad arvustuste hinnetega (nt kas negatiivseid arvustusi on rohkem *Noorte lastega perede* puhul kui *Üksikreisijate* puhul, mis võib viidata sellele, et hotell sobib paremini *Üksikreisijatele*)?
* Kas NLTK sentimentihinded "nõustuvad" hotelli arvustaja numbrilise hindega?

#### Andmestik

Uurime andmestikku, mille oled alla laadinud ja kohapeal salvestanud. Ava fail redaktoris nagu VS Code või isegi Excelis.

Andmestiku päised on järgmised:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Siin on need rühmitatud viisil, mis võib olla lihtsam uurimiseks:  
##### Hotelli veerud

* `Hotel_Name`, `Hotel_Address`, `lat` (laiuskraad), `lng` (pikkuskraad)
  * Kasutades *lat* ja *lng* võiksid Pythoniga koostada kaardi, mis näitab hotelli asukohti (võib-olla värvikoodiga negatiivsete ja positiivsete arvustuste jaoks)
  * Hotel_Address ei tundu meile otseselt kasulik, ja tõenäoliselt asendame selle riigiga, et lihtsustada sortimist ja otsingut

**Hotelli meta-arvustuste veerud**

* `Average_Score`
  * Andmestiku looja sõnul on see veerg *Hotelli keskmine hinne, arvutatud viimase aasta viimase kommentaari põhjal*. See tundub ebatavaline viis hinde arvutamiseks, kuid kuna andmed on kogutud, võtame selle praegu tõe pähe. 
  
  ✅ Kas saad mõelda teisele viisile keskmise hinde arvutamiseks, tuginedes andmestiku teistele veergudele?

* `Total_Number_of_Reviews`
  * Hotelli saadud arvustuste koguarv - pole selge (ilma koodi kirjutamata), kas see viitab andmestikus olevatele arvustustele.
* `Additional_Number_of_Scoring`
  * See tähendab, et arvustaja andis hinde, kuid ei kirjutanud positiivset ega negatiivset arvustust

**Arvustuste veerud**

- `Reviewer_Score`
  - See on numbriline väärtus, millel on maksimaalselt 1 kümnendkoht vahemikus 2.5 kuni 10
  - Pole selgitatud, miks 2.5 on madalaim võimalik hinne
- `Negative_Review`
  - Kui arvustaja ei kirjutanud midagi, on selles väljas "**No Negative**"
  - Pane tähele, et arvustaja võib kirjutada positiivse arvustuse negatiivse arvustuse veergu (nt "selles hotellis pole midagi halba")
- `Review_Total_Negative_Word_Counts`
  - Suurem negatiivsete sõnade arv viitab madalamale hindele (ilma sentimenti kontrollimata)
- `Positive_Review`
  - Kui arvustaja ei kirjutanud midagi, on selles väljas "**No Positive**"
  - Pane tähele, et arvustaja võib kirjutada negatiivse arvustuse positiivse arvustuse veergu (nt "selles hotellis pole üldse midagi head")
- `Review_Total_Positive_Word_Counts`
  - Suurem positiivsete sõnade arv viitab kõrgemale hindele (ilma sentimenti kontrollimata)
- `Review_Date` ja `days_since_review`
  - Võiks rakendada värskuse või vananemise mõõdet arvustusele (vanemad arvustused ei pruugi olla nii täpsed kui uuemad, kuna hotelli juhtkond on muutunud, tehtud on renoveerimistöid või lisatud bassein jne)
- `Tags`
  - Need on lühikesed kirjeldused, mille arvustaja võib valida, et kirjeldada külalise tüüpi (nt üksik või pere), toa tüüpi, peatumise pikkust ja kuidas arvustus esitati. 
  - Kahjuks on nende siltide kasutamine problemaatiline, vaata allpool olevat jaotist, mis arutleb nende kasulikkuse üle

**Arvustaja veerud**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - See võib olla tegur soovitusmudelis, näiteks kui suudad kindlaks teha, et produktiivsemad arvustajad, kellel on sadu arvustusi, olid tõenäolisemalt negatiivsed kui positiivsed. Kuid konkreetse arvustuse arvustajat ei ole identifitseeritud unikaalse koodiga ja seetõttu ei saa teda siduda arvustuste kogumiga. Andmestikus on 30 arvustajat, kellel on 100 või rohkem arvustust, kuid on raske näha, kuidas see võiks soovitusmudelit aidata.
- `Reviewer_Nationality`
  - Mõned inimesed võivad arvata, et teatud rahvused on tõenäolisemalt positiivse või negatiivse arvustuse andjad rahvusliku kalduvuse tõttu. Ole ettevaatlik selliste anekdootlike vaadete lisamisel oma mudelitesse. Need on rahvuslikud (ja mõnikord rassilised) stereotüübid ning iga arvustaja oli individuaalne, kes kirjutas arvustuse oma kogemuse põhjal. See võis olla filtreeritud läbi mitme prisma, nagu nende varasemad hotellipeatused, läbitud vahemaa ja isiklik temperament. Raske on õigustada arvamust, et nende rahvus oli arvustuse hinde põhjus.

##### Näited

| Keskmine hinne | Arvustuste koguarv | Arvustaja hinne | Negatiivne <br />arvustus                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positiivne arvustus                 | Sildid                                                                                      |
| -------------- | ------------------ | --------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945               | 2.5             | See ei ole praegu hotell, vaid ehitusplats. Mind terroriseeriti varahommikust ja kogu päeva jooksul vastuvõetamatu ehitusmüraga, kui puhkasin pärast pikka reisi ja töötasin toas. Inimesed töötasid kogu päeva, näiteks naelutajatega kõrvalruumides. Palusin toa vahetust, kuid vaikset tuba polnud saadaval. Asja hullemaks muutmiseks võeti minult ülehinnatud tasu. Lahkusin õhtul, kuna pidin varahommikul lennule minema, ja sain sobiva arve. Päev hiljem tegi hotell minu nõusolekuta veel ühe tasu, mis ületas broneeritud hinna. See on kohutav koht. Ära karista ennast, broneerides siin. | Mitte midagi. Kohutav koht. Hoia eemale. | Ärirännak                                Paar Standard Double Room Peatus 2 ööd |

Nagu näha, ei olnud sellel külalisel hotellis meeldiv peatus. Hotellil on hea keskmine hinne 7.8 ja 1945 arvustust, kuid see arvustaja andis sellele 2.5 ja kirjutas 115 sõna, kuidas negatiivne nende peatus oli. Kui nad ei kirjutanud midagi Positiivne arvustus veergu, võiks arvata, et midagi positiivset polnud, kuid siiski kirjutasid nad 7 hoiatavat sõna. Kui loeksime ainult sõnu, mitte nende tähendust või sentimenti, võiksime saada arvustaja kavatsusest moonutatud pildi. Kummalisel kombel on nende hinne 2.5 segadust tekitav, sest kui hotellipeatus oli nii halb, miks anda sellele üldse punkte? Andmestikku lähemalt uurides näed, et madalaim võimalik hinne on 2.5, mitte 0. Kõrgeim võimalik hinne on 10.

##### Sildid

Nagu eespool mainitud, esmapilgul tundub idee kasutada `Tags` veergu andmete kategoriseerimiseks mõistlik. Kahjuks ei ole need sildid standardiseeritud, mis tähendab, et ühes hotellis võivad valikud olla *Single room*, *Twin room* ja *Double room*, kuid järgmises hotellis on need *Deluxe Single Room*, *Classic Queen Room* ja *Executive King Room*. Need võivad olla samad asjad, kuid variatsioone on nii palju, et valikuks jääb:

1. Püüda muuta kõik terminid üheks standardiks, mis on väga keeruline, kuna pole selge, milline oleks teisendusteekond igal juhul (nt *Classic single room* vastab *Single room*-ile, kuid *Superior Queen Room with Courtyard Garden or City View* on palju raskem kaardistada)

1. Võime võtta NLP lähenemise ja mõõta teatud terminite nagu *Solo*, *Business Traveller* või *Family with young kids* sagedust, kui need kehtivad iga hotelli kohta, ja arvestada seda soovituses  

Sildid on tavaliselt (kuid mitte alati) üks väli, mis sisaldab 5–6 komaga eraldatud väärtust, mis vastavad *Reisi tüübile*, *Külaliste tüübile*, *Toa tüübile*, *Ööde arvule* ja *Seadmele, millelt arvustus esitati*. Kuid kuna mõned arvustajad ei täida iga välja (nad võivad jätta ühe tühjaks), ei ole väärtused alati samas järjekorras.

Näiteks võtame *Grupi tüüp*. Selles veerus `Tags` on 1025 unikaalset võimalust ja kahjuks viitavad ainult mõned neist grupile (mõned on toa tüüp jne). Kui filtreerid ainult need, mis mainivad peret, sisaldavad tulemused palju *Family room* tüüpi tulemusi. Kui lisad termini *with*, st loendad *Family with* väärtused, on tulemused paremad, kus üle 80 000 515 000 tulemusest sisaldavad fraasi "Family with young children" või "Family with older children".

See tähendab, et siltide veerg ei ole meile täiesti kasutu, kuid selle kasulikuks muutmiseks on vaja tööd.

##### Hotelli keskmine hinne

Andmestikus on mitmeid veidrusi või lahknevusi, mida ma ei suuda välja selgitada, kuid need on siin illustreeritud, et oleksid neist teadlik, kui ehitad oma mudeleid. Kui suudad selle välja selgitada, anna meile teada arutelusektsioonis!

Andmestikus on järgmised veerud, mis on seotud keskmise hinde ja arvustuste arvuga:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Andmestiku hotell, millel on kõige rohkem arvustusi, on *Britannia International Hotel Canary Wharf* 4789 arvustusega 515 000-st. Kuid kui vaatame selle hotelli `Total_Number_of_Reviews` väärtust, on see 9086. Võiks arvata, et on palju rohkem hindeid ilma arvustusteta, seega võiksime lisada `Additional_Number_of_Scoring` veeru väärtuse. See väärtus on 2682 ja kui lisada see 4789-le, saame 7471, mis on siiski 1615 vähem kui `Total_Number_of_Reviews`.

Kui võtad `Average_Score` veeru, võiks arvata, et see on andmestikus olevate arvustuste keskmine, kuid Kaggle'i kirjeldus on "*Hotelli keskmine hinne, arvutatud viimase aasta viimase kommentaari põhjal*". See ei tundu eriti kasulik, kuid saame arvutada oma keskmise, tuginedes andmestikus olevatele arvustuste hinnetele. Kasutades sama hotelli näitena, on hotelli keskmine hinne antud kui 7.1, kuid arvutatud hinne (keskmine arvustaja hinne *andmestikus*) on 6.8. See on lähedane, kuid mitte sama väärtus, ja võime ainult arvata, et `Additional_Number_of_Scoring` arvustuste hinded tõstsid keskmise 7.1-ni. Kahjuks, kuna puudub viis seda testida või tõestada, on raske kasutada või usaldada `Average_Score`, `Additional_Number_of_Scoring` ja `Total_Number_of_Reviews`, kui need põhinevad andmetel, mida meil pole.

Asja veelgi keerulisemaks muutmiseks on hotell, millel on andmestikus arvustuste arvult teine koht, arvutatud keskmise hindega 8.12 ja andmestiku `Average_Score` on 8.1. Kas see õige hinne on juhus või on esimene hotell lahknevus? 
Võimalik, et see hotell on erandlik ja enamik väärtusi klapivad (kuid mõned mingil põhjusel ei klapi). Järgmises osas kirjutame lühikese programmi, et uurida andmestiku väärtusi ja määrata kindlaks nende õige kasutus (või mittekasutus).

> 🚨 Hoiatus
>
> Selle andmestikuga töötades kirjutate koodi, mis arvutab midagi tekstist, ilma et peaksite ise teksti lugema või analüüsima. See on NLP olemus – tähenduse või meeleolu tõlgendamine ilma inimese sekkumiseta. Siiski on võimalik, et loete mõningaid negatiivseid arvustusi. Soovitan tungivalt seda mitte teha, sest te ei pea. Mõned neist on rumalad või ebaolulised negatiivsed hotelliarvustused, näiteks "Ilm polnud suurepärane", mis on midagi, mida hotell või keegi teine ei saa kontrollida. Kuid mõnel arvustusel on ka tumedam külg. Mõnikord on negatiivsed arvustused rassistlikud, seksistlikud või vanuselist diskrimineerimist sisaldavad. See on kahetsusväärne, kuid oodatav, kui andmestik on kogutud avalikult veebisaidilt. Mõned arvustajad jätavad arvustusi, mis võivad olla ebameeldivad, ebamugavad või häirivad. Parem on lasta koodil meeleolu mõõta, kui ise neid lugeda ja häiritud olla. Seda öeldes on selliseid arvustusi kirjutavaid inimesi vähemuses, kuid nad eksisteerivad siiski.

## Harjutus – Andmete uurimine
### Andmete laadimine

Visuaalsest andmete uurimisest piisab, nüüd kirjutate koodi ja saate vastuseid! Selles osas kasutatakse pandas teeki. Teie esimene ülesanne on tagada, et suudate CSV-andmed laadida ja lugeda. Pandas teekil on kiire CSV-laadur, mille tulemus paigutatakse andmeraami, nagu varasemates tundides. Laaditav CSV sisaldab üle poole miljoni rea, kuid ainult 17 veergu. Pandas pakub palju võimsaid viise andmeraamiga töötamiseks, sealhulgas võimalust teha operatsioone igal real.

Alates sellest punktist selles tunnis on koodinäited, koodi selgitused ja arutelu tulemuste tähenduse üle. Kasutage kaasasolevat _notebook.ipynb_-faili oma koodi jaoks.

Alustame andmefaili laadimisest, mida hakkate kasutama:

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

Nüüd, kui andmed on laaditud, saame nendega operatsioone teha. Hoidke see kood oma programmi alguses järgmise osa jaoks.

## Andmete uurimine

Antud juhul on andmed juba *puhastatud*, mis tähendab, et need on valmis töötlemiseks ja ei sisalda teistes keeltes märke, mis võiksid algoritme segadusse ajada, kui need eeldavad ainult ingliskeelseid märke.

✅ Võimalik, et peate töötama andmetega, mis vajavad algset töötlemist enne NLP-tehnikate rakendamist, kuid mitte seekord. Kui peaksite, siis kuidas käsitleksite mitte-ingliskeelseid märke?

Veenduge, et pärast andmete laadimist saate neid koodiga uurida. On väga lihtne keskenduda `Negative_Review` ja `Positive_Review` veergudele. Need on täidetud loomuliku tekstiga, mida teie NLP-algoritmid saavad töödelda. Kuid oodake! Enne NLP ja meeleolu analüüsi hüppamist peaksite järgima allolevat koodi, et veenduda, kas andmestikus antud väärtused vastavad pandasiga arvutatud väärtustele.

## Andmeraami operatsioonid

Selle tunni esimene ülesanne on kontrollida, kas järgmised väited on õiged, kirjutades koodi, mis uurib andmeraami (seda muutmata).

> Nagu paljude programmeerimisülesannete puhul, on mitmeid viise selle täitmiseks, kuid hea nõuanne on teha seda kõige lihtsamal ja kergemini mõistetaval viisil, eriti kui see muudab koodi tulevikus lihtsamini arusaadavaks. Andmeraamidega töötades on olemas ulatuslik API, mis sageli pakub tõhusat viisi soovitud toimingu tegemiseks.

Käsitlege järgmisi küsimusi kui programmeerimisülesandeid ja proovige neile vastata ilma lahendust vaatamata.

1. Väljasta just laaditud andmeraami *kuju* (kuju tähendab ridade ja veergude arvu).
2. Arvutage arvustajate rahvuste sagedus:
   1. Kui palju erinevaid väärtusi on veerus `Reviewer_Nationality` ja millised need on?
   2. Milline arvustaja rahvus on andmestikus kõige levinum (väljasta riik ja arvustuste arv)?
   3. Millised on järgmised 10 kõige sagedamini esinevat rahvust ja nende sagedus?
3. Milline hotell sai kõige rohkem arvustusi iga 10 kõige sagedamini esineva rahvuse arvustajate seas?
4. Kui palju arvustusi on igal hotellil (hotelli sagedus andmestikus)?
5. Kuigi andmestikus on iga hotelli jaoks veerg `Average_Score`, saate arvutada ka keskmise skoori (arvutades iga hotelli arvustajate skooride keskmise andmestikus). Lisage oma andmeraamile uus veerg pealkirjaga `Calc_Average_Score`, mis sisaldab arvutatud keskmist. 
6. Kas mõnel hotellil on sama (ümardatud ühe kümnendkohani) `Average_Score` ja `Calc_Average_Score`?
   1. Proovige kirjutada Python-funktsioon, mis võtab argumendiks Series (rea) ja võrdleb väärtusi, väljastades sõnumi, kui väärtused ei ole võrdsed. Seejärel kasutage `.apply()` meetodit, et töödelda iga rida funktsiooniga.
7. Arvutage ja väljasta, kui paljudel ridadel on veeru `Negative_Review` väärtus "No Negative".
8. Arvutage ja väljasta, kui paljudel ridadel on veeru `Positive_Review` väärtus "No Positive".
9. Arvutage ja väljasta, kui paljudel ridadel on veeru `Positive_Review` väärtus "No Positive" **ja** veeru `Negative_Review` väärtus "No Negative".

### Koodi vastused

1. Väljasta just laaditud andmeraami *kuju* (kuju tähendab ridade ja veergude arvu).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Arvutage arvustajate rahvuste sagedus:

   1. Kui palju erinevaid väärtusi on veerus `Reviewer_Nationality` ja millised need on?
   2. Milline arvustaja rahvus on andmestikus kõige levinum (väljasta riik ja arvustuste arv)?

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

   3. Millised on järgmised 10 kõige sagedamini esinevat rahvust ja nende sagedus?

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

3. Milline hotell sai kõige rohkem arvustusi iga 10 kõige sagedamini esineva rahvuse arvustajate seas?

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

4. Kui palju arvustusi on igal hotellil (hotelli sagedus andmestikus)?

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
   
   Võite märgata, et *andmestikus loendatud* tulemused ei vasta väärtusele `Total_Number_of_Reviews`. Ei ole selge, kas see väärtus andmestikus esindas hotelli arvustuste koguarvu, kuid mitte kõiki ei kogutud, või mõnda muud arvutust. `Total_Number_of_Reviews` ei kasutata mudelis, kuna see on ebaselge.

5. Kuigi andmestikus on iga hotelli jaoks veerg `Average_Score`, saate arvutada ka keskmise skoori (arvutades iga hotelli arvustajate skooride keskmise andmestikus). Lisage oma andmeraamile uus veerg pealkirjaga `Calc_Average_Score`, mis sisaldab arvutatud keskmist. Väljasta veerud `Hotel_Name`, `Average_Score` ja `Calc_Average_Score`.

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

   Võite ka imestada `Average_Score` väärtuse üle ja miks see mõnikord erineb arvutatud keskmisest skoorist. Kuna me ei saa teada, miks mõned väärtused klapivad, kuid teised erinevad, on antud juhul kõige ohutum kasutada arvustuste skoori, et keskmine ise arvutada. Seda öeldes on erinevused tavaliselt väga väikesed, siin on hotellid, millel on suurim erinevus andmestiku keskmise ja arvutatud keskmise vahel:

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

   Kuna ainult ühel hotellil on skoori erinevus suurem kui 1, tähendab see, et tõenäoliselt võime erinevuse ignoreerida ja kasutada arvutatud keskmist skoori.

6. Arvutage ja väljasta, kui paljudel ridadel on veeru `Negative_Review` väärtus "No Negative".

7. Arvutage ja väljasta, kui paljudel ridadel on veeru `Positive_Review` väärtus "No Positive".

8. Arvutage ja väljasta, kui paljudel ridadel on veeru `Positive_Review` väärtus "No Positive" **ja** veeru `Negative_Review` väärtus "No Negative".

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

## Teine viis

Teine viis loendamiseks ilma Lambdasita ja ridade loendamiseks summa abil:

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

   Võite märgata, et on 127 rida, millel on veergudes `Negative_Review` ja `Positive_Review` vastavalt väärtused "No Negative" ja "No Positive". See tähendab, et arvustaja andis hotellile numbrilise skoori, kuid keeldus kirjutamast kas positiivset või negatiivset arvustust. Õnneks on selliste ridade arv väike (127 rida 515738-st ehk 0,02%), seega see tõenäoliselt ei mõjuta meie mudelit ega tulemusi mingis suunas. Kuid te ei pruugi oodata, et arvustuste andmestikus on ridu, kus pole arvustusi, seega tasub andmeid uurida, et selliseid ridu avastada.

Nüüd, kui olete andmestikku uurinud, filtreerite järgmises tunnis andmeid ja lisate meeleolu analüüsi.

---
## 🚀Väljakutse

See tund näitab, nagu nägime varasemates tundides, kui kriitiliselt oluline on mõista oma andmeid ja nende eripärasid enne nende töötlemist. Tekstipõhised andmed vajavad eriti hoolikat uurimist. Uurige erinevaid tekstimahukaid andmestikke ja vaadake, kas suudate avastada valdkondi, mis võivad mudelisse kallutatust või moonutatud meeleolu tuua.

## [Loengu järgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õppimine

Võtke [see NLP õppeprogramm](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), et avastada tööriistu, mida proovida kõne- ja tekstimahukate mudelite loomisel.

## Ülesanne 

[NLTK](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.