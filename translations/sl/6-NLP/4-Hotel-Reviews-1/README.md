<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T14:00:43+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "sl"
}
-->
# Analiza sentimenta s hotelskimi ocenami - obdelava podatkov

V tem poglavju boste uporabili tehnike iz prejšnjih lekcij za izvedbo raziskovalne analize velikega nabora podatkov. Ko boste pridobili dobro razumevanje uporabnosti različnih stolpcev, se boste naučili:

- kako odstraniti nepotrebne stolpce
- kako izračunati nove podatke na podlagi obstoječih stolpcev
- kako shraniti nastali nabor podatkov za uporabo v končnem izzivu

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

Do sedaj ste se naučili, da so besedilni podatki precej drugačni od numeričnih podatkov. Če gre za besedilo, ki ga je napisal ali izgovoril človek, ga je mogoče analizirati za iskanje vzorcev, frekvenc, sentimenta in pomena. Ta lekcija vas popelje v resničen nabor podatkov z resničnim izzivom: **[515K Hotelske ocene v Evropi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, ki vključuje [CC0: javno domeno](https://creativecommons.org/publicdomain/zero/1.0/). Podatki so bili pridobljeni iz Booking.com iz javnih virov. Ustvarjalec nabora podatkov je Jiashen Liu.

### Priprava

Potrebovali boste:

* Zmožnost izvajanja .ipynb zvezkov z uporabo Python 3
* pandas
* NLTK, [ki ga morate namestiti lokalno](https://www.nltk.org/install.html)
* Nabor podatkov, ki je na voljo na Kaggle [515K Hotelske ocene v Evropi](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Velikost datoteke po razpakiranju je približno 230 MB. Prenesite jo v korensko mapo `/data`, povezano s temi lekcijami NLP.

## Raziskovalna analiza podatkov

Ta izziv predvideva, da gradite hotelskega priporočilnega bota z uporabo analize sentimenta in ocen gostov. Nabor podatkov, ki ga boste uporabili, vključuje ocene 1493 različnih hotelov v 6 mestih.

Z uporabo Pythona, nabora hotelskih ocen in analize sentimenta NLTK lahko ugotovite:

* Katere so najpogosteje uporabljene besede in fraze v ocenah?
* Ali se uradne *oznake* hotela ujemajo z ocenami (npr. ali so bolj negativne ocene za določen hotel pri *Družinah z majhnimi otroki* kot pri *Samostojnih popotnikih*, kar morda nakazuje, da je hotel bolj primeren za *Samostojne popotnike*)?
* Ali se ocene sentimenta NLTK 'strinjajo' s številčno oceno hotelskega ocenjevalca?

#### Nabor podatkov

Raziskujmo nabor podatkov, ki ste ga prenesli in shranili lokalno. Odprite datoteko v urejevalniku, kot je VS Code ali celo Excel.

Naslovi stolpcev v naboru podatkov so naslednji:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Tukaj so razvrščeni na način, ki je morda lažji za pregled:  
##### Stolpci hotela

* `Hotel_Name`, `Hotel_Address`, `lat` (geografska širina), `lng` (geografska dolžina)
  * Z uporabo *lat* in *lng* lahko s Pythonom narišete zemljevid, ki prikazuje lokacije hotelov (morda barvno kodirane za negativne in pozitivne ocene)
  * Hotel_Address ni očitno uporaben za nas, zato ga bomo verjetno zamenjali z državo za lažje razvrščanje in iskanje

**Stolpci meta-ocen hotela**

* `Average_Score`
  * Po navedbah ustvarjalca nabora podatkov ta stolpec predstavlja *Povprečno oceno hotela, izračunano na podlagi najnovejšega komentarja v zadnjem letu*. To se zdi nenavaden način izračuna ocene, vendar so to pridobljeni podatki, zato jih za zdaj sprejmemo kot takšne. 
  
  ✅ Glede na druge stolpce v teh podatkih, ali lahko pomislite na drug način za izračun povprečne ocene?

* `Total_Number_of_Reviews`
  * Skupno število ocen, ki jih je hotel prejel - ni jasno (brez pisanja kode), ali se to nanaša na ocene v naboru podatkov.
* `Additional_Number_of_Scoring`
  * To pomeni, da je bila podana ocena, vendar ocenjevalec ni napisal pozitivne ali negativne ocene.

**Stolpci ocen**

- `Reviewer_Score`
  - To je številčna vrednost z največ 1 decimalnim mestom med minimalno in maksimalno vrednostjo 2.5 in 10
  - Ni pojasnjeno, zakaj je najnižja možna ocena 2.5
- `Negative_Review`
  - Če ocenjevalec ni napisal ničesar, bo to polje vsebovalo "**No Negative**"
  - Upoštevajte, da lahko ocenjevalec napiše pozitivno oceno v stolpec Negative review (npr. "ni nič slabega glede tega hotela")
- `Review_Total_Negative_Word_Counts`
  - Višje število negativnih besed nakazuje nižjo oceno (brez preverjanja sentimenta)
- `Positive_Review`
  - Če ocenjevalec ni napisal ničesar, bo to polje vsebovalo "**No Positive**"
  - Upoštevajte, da lahko ocenjevalec napiše negativno oceno v stolpec Positive review (npr. "ni nič dobrega glede tega hotela")
- `Review_Total_Positive_Word_Counts`
  - Višje število pozitivnih besed nakazuje višjo oceno (brez preverjanja sentimenta)
- `Review_Date` in `days_since_review`
  - Svežino ali zastarelost ocene bi lahko uporabili kot merilo (starejše ocene morda niso tako natančne kot novejše, ker se je upravljanje hotela spremenilo, izvedene so bile prenove, dodan je bil bazen itd.)
- `Tags`
  - To so kratki opisi, ki jih ocenjevalec lahko izbere za opis vrste gosta (npr. samostojni ali družinski), vrste sobe, dolžine bivanja in načina oddaje ocene.
  - Na žalost je uporaba teh oznak problematična, preverite spodnji razdelek, ki obravnava njihovo uporabnost.

**Stolpci ocenjevalca**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - To bi lahko bil dejavnik v priporočilnem modelu, na primer, če bi lahko ugotovili, da so bolj plodni ocenjevalci s stotinami ocen bolj verjetno negativni kot pozitivni. Vendar ocenjevalec posamezne ocene ni identificiran z edinstveno kodo, zato ga ni mogoče povezati z naborom ocen. Obstaja 30 ocenjevalcev s 100 ali več ocenami, vendar je težko videti, kako bi to lahko pomagalo priporočilnemu modelu.
- `Reviewer_Nationality`
  - Nekateri ljudje morda mislijo, da so določene narodnosti bolj verjetno podale pozitivno ali negativno oceno zaradi nacionalne nagnjenosti. Bodite previdni pri vključevanju takšnih anekdotičnih pogledov v svoje modele. To so nacionalni (in včasih rasni) stereotipi, vsak ocenjevalec pa je bil posameznik, ki je napisal oceno na podlagi svoje izkušnje. Ta je bila morda filtrirana skozi številne leče, kot so njihova prejšnja bivanja v hotelih, prepotovana razdalja in njihova osebna temperamentnost. Težko je upravičiti razmišljanje, da je bila njihova narodnost razlog za oceno.

##### Primeri

| Povprečna ocena | Skupno število ocen | Ocena ocenjevalca | Negativna <br />ocena                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pozitivna ocena                 | Oznake                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Trenutno to ni hotel, ampak gradbišče. Terorizirali so me od zgodnjega jutra in ves dan z nesprejemljivim gradbenim hrupom, medtem ko sem počival po dolgem potovanju in delal v sobi. Ljudje so delali ves dan, npr. z udarnimi kladivi v sosednjih sobah. Prosil sem za zamenjavo sobe, vendar ni bilo na voljo tihe sobe. Da bi bilo še huje, so mi zaračunali preveč. Odjavil sem se zvečer, saj sem moral zgodaj zjutraj na let in prejel ustrezen račun. Dan kasneje je hotel brez mojega soglasja izvedel dodatno bremenitev, ki je presegala ceno rezervacije. To je grozno mesto. Ne kaznujte se z rezervacijo tukaj. | Nič. Grozno mesto. Izogibajte se. | Poslovno potovanje                                Par Standardna dvoposteljna soba Bivanje 2 noči |

Kot lahko vidite, gost ni imel prijetnega bivanja v tem hotelu. Hotel ima dobro povprečno oceno 7.8 in 1945 ocen, vendar mu je ta ocenjevalec dal oceno 2.5 in napisal 115 besed o tem, kako negativno je bilo njegovo bivanje. Če v stolpcu Positive_Review ni napisal ničesar, bi lahko sklepali, da ni bilo nič pozitivnega, vendar je napisal 7 opozorilnih besed. Če bi šteli samo besede namesto pomena ali sentimenta besed, bi lahko imeli izkrivljen pogled na namen ocenjevalca. Nenavadno je, da je njihova ocena 2.5 zmedena, saj če je bilo bivanje v hotelu tako slabo, zakaj sploh dati kakršne koli točke? Pri podrobnem pregledu nabora podatkov boste videli, da je najnižja možna ocena 2.5, ne 0. Najvišja možna ocena je 10.

##### Oznake

Kot je bilo omenjeno zgoraj, se na prvi pogled zdi ideja uporabe `Tags` za kategorizacijo podatkov smiselna. Na žalost te oznake niso standardizirane, kar pomeni, da so v določenem hotelu možnosti *Enoposteljna soba*, *Dvoposteljna soba* in *Zakonska soba*, v naslednjem hotelu pa *Deluxe enoposteljna soba*, *Klasična soba Queen* in *Izvrstna soba King*. To so morda iste stvari, vendar je toliko različic, da izbira postane:

1. Poskus spremeniti vse izraze v en sam standard, kar je zelo težko, ker ni jasno, kakšna bi bila pot pretvorbe v vsakem primeru (npr. *Klasična enoposteljna soba* se preslika v *Enoposteljna soba*, vendar *Superior Queen soba z vrtom ali pogledom na mesto* je veliko težje preslikati)

1. Lahko uporabimo pristop NLP in izmerimo pogostost določenih izrazov, kot so *Samostojni*, *Poslovni popotnik* ali *Družina z majhnimi otroki*, kot se nanašajo na vsak hotel, ter to vključimo v priporočilo  

Oznake so običajno (vendar ne vedno) eno polje, ki vsebuje seznam 5 do 6 vrednosti, ločenih z vejicami, ki se nanašajo na *Vrsto potovanja*, *Vrsto gostov*, *Vrsto sobe*, *Število nočitev* in *Vrsto naprave, na kateri je bila oddana ocena*. Ker pa nekateri ocenjevalci ne izpolnijo vsakega polja (morda pustijo eno prazno), vrednosti niso vedno v istem vrstnem redu.

Na primer, vzemimo *Vrsto skupine*. V tem polju v stolpcu `Tags` je 1025 edinstvenih možnosti, na žalost pa se le nekatere nanašajo na skupino (nekatere so vrsta sobe itd.). Če filtrirate samo tiste, ki omenjajo družino, rezultati vsebujejo veliko rezultatov tipa *Družinska soba*. Če vključite izraz *z*, tj. štejete vrednosti *Družina z*, so rezultati boljši, saj več kot 80.000 od 515.000 rezultatov vsebuje frazo "Družina z majhnimi otroki" ali "Družina z starejšimi otroki".

To pomeni, da stolpec oznak ni popolnoma neuporaben za nas, vendar bo potrebno nekaj dela, da ga naredimo uporabnega.

##### Povprečna ocena hotela

Obstaja nekaj nenavadnosti ali neskladnosti z naborom podatkov, ki jih ne morem razložiti, vendar so tukaj prikazane, da se jih zavedate pri gradnji svojih modelov. Če jih razložite, nam to prosim sporočite v razdelku za razpravo!

Nabor podatkov ima naslednje stolpce, povezane s povprečno oceno in številom ocen:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel z največ ocenami v tem naboru podatkov je *Britannia International Hotel Canary Wharf* s 4789 ocenami od 515.000. Če pa pogledamo vrednost `Total_Number_of_Reviews` za ta hotel, je 9086. Lahko bi sklepali, da obstaja veliko več ocen brez besedilnih ocen, zato bi morda morali dodati vrednost stolpca `Additional_Number_of_Scoring`. Ta vrednost je 2682, in če jo dodamo k 4789, dobimo 7471, kar je še vedno 1615 manj od `Total_Number_of_Reviews`. 

Če vzamete stolpec `Average_Score`, bi lahko sklepali, da gre za povprečje ocen v naboru podatkov, vendar opis iz Kaggle pravi: "*Povprečna ocena hotela, izračunana na podlagi najnovejšega komentarja v zadnjem letu*". To se ne zdi zelo uporabno, vendar lahko izračunamo svoje povprečje na podlagi ocen v naboru podatkov. Če vzamemo isti hotel kot primer, je podana povprečna ocena hotela 7.1, vendar je izračunana ocena (povprečna ocena ocenjevalca *v* naboru podatkov) 6.8. To je blizu, vendar ni enaka vrednost, in lahko le ugibamo, da so ocene podane v ocenah `Additional_Number_of_Scoring` povečale povprečje na 7.1. Na žalost brez načina za testiranje ali dokazovanje te trditve je težko uporabiti ali zaupati `Average_Score`, `Additional_Number_of_Scoring` in `Total_Number_of_Reviews`, ko temeljijo na podatkih, ki jih nimamo.

Da bi stvari še bolj zapletli, ima hotel z drugim največjim številom ocen izračunano povprečno oceno 8.12, medtem ko je podana povprečna ocena v naboru podatkov 8.1. Ali je ta pravilna ocena naključje ali je prvi hotel neskladje? 

Ob možnosti, da bi ti hoteli lahko bili odstopajoči, in da morda večina vrednosti ustreza (nekateri pa ne iz nekega razloga), bomo v naslednjem koraku napisali kratek program za raziskovanje vrednosti v naboru podatkov in določitev pravilne uporabe (ali neuporabe) vrednosti.
> 🚨 Opozorilo

> Pri delu s tem naborom podatkov boste pisali kodo, ki izračuna nekaj iz besedila, ne da bi morali sami prebrati ali analizirati besedilo. To je bistvo NLP-ja: interpretacija pomena ali sentimenta brez potrebe po človeškem posredovanju. Vendar pa obstaja možnost, da boste prebrali nekatere negativne ocene. Svetoval bi vam, da tega ne počnete, saj ni potrebno. Nekatere od teh ocen so nesmiselne ali nepomembne negativne ocene hotelov, kot na primer: "Vreme ni bilo dobro," kar je zunaj nadzora hotela ali kogarkoli drugega. Toda obstaja tudi temna plat nekaterih ocen. Včasih so negativne ocene rasistične, seksistične ali starostno diskriminatorne. To je žalostno, vendar pričakovano v naboru podatkov, pridobljenem z javne spletne strani. Nekateri ocenjevalci pustijo ocene, ki bi jih lahko dojemali kot neprijetne, nelagodne ali vznemirljive. Bolje je, da koda izmeri sentiment, kot da jih sami preberete in se ob tem vznemirite. Kljub temu gre za manjšino, ki piše takšne stvari, vendar vseeno obstajajo.
## Vaja - Raziskovanje podatkov
### Nalaganje podatkov

Dovolj je bilo vizualnega pregledovanja podatkov, zdaj boste napisali nekaj kode in dobili odgovore! Ta del uporablja knjižnico pandas. Vaša prva naloga je zagotoviti, da lahko naložite in preberete podatke iz datoteke CSV. Knjižnica pandas ima hiter nalagalnik za CSV, rezultat pa je shranjen v dataframe, kot v prejšnjih lekcijah. CSV, ki ga nalagamo, ima več kot pol milijona vrstic, vendar le 17 stolpcev. Pandas vam ponuja veliko zmogljivih načinov za interakcijo z dataframe, vključno z možnostjo izvajanja operacij na vsaki vrstici.

Od tu naprej v tej lekciji bodo vključeni odlomki kode, razlage kode in razprave o tem, kaj rezultati pomenijo. Uporabite priloženi _notebook.ipynb_ za svojo kodo.

Začnimo z nalaganjem datoteke s podatki, ki jo boste uporabljali:

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

Ko so podatki naloženi, lahko na njih izvajamo operacije. To kodo obdržite na vrhu svojega programa za naslednji del.

## Raziskovanje podatkov

V tem primeru so podatki že *čisti*, kar pomeni, da so pripravljeni za delo in ne vsebujejo znakov v drugih jezikih, ki bi lahko povzročili težave algoritmom, ki pričakujejo samo angleške znake.

✅ Morda boste morali delati s podatki, ki zahtevajo začetno obdelavo, da jih formatirate, preden uporabite tehnike NLP, vendar tokrat ne. Če bi morali, kako bi obravnavali ne-angleške znake?

Vzemite si trenutek, da se prepričate, da lahko po nalaganju podatkov z njimi raziskujete s kodo. Zelo enostavno je osredotočiti se na stolpca `Negative_Review` in `Positive_Review`. Napolnjena sta z naravnim besedilom, ki ga lahko obdelajo vaši NLP algoritmi. Ampak počakajte! Preden se lotite NLP in sentimenta, sledite spodnji kodi, da ugotovite, ali vrednosti v podatkovnem naboru ustrezajo vrednostim, ki jih izračunate s pandas.

## Operacije na dataframe

Prva naloga v tej lekciji je preveriti, ali so naslednje trditve pravilne, tako da napišete kodo, ki preučuje dataframe (brez spreminjanja).

> Kot pri mnogih programerskih nalogah obstaja več načinov za izvedbo, vendar je dobro priporočilo, da to storite na najpreprostejši in najlažji način, še posebej, če bo to lažje razumeti, ko se boste kasneje vrnili k tej kodi. Pri dataframe obstaja obsežen API, ki pogosto ponuja učinkovit način za dosego želenega.

Obravnavajte naslednja vprašanja kot naloge kodiranja in jih poskusite rešiti brez gledanja rešitve.

1. Izpišite *obliko* dataframe, ki ste ga pravkar naložili (oblika je število vrstic in stolpcev).
2. Izračunajte frekvenčno število za narodnosti ocenjevalcev:
   1. Koliko različnih vrednosti je v stolpcu `Reviewer_Nationality` in katere so?
   2. Katera narodnost ocenjevalcev je najpogostejša v podatkovnem naboru (izpišite državo in število ocen)?
   3. Katere so naslednjih 10 najpogostejših narodnosti in njihovo frekvenčno število?
3. Kateri hotel je bil najpogosteje ocenjen za vsako od 10 najpogostejših narodnosti ocenjevalcev?
4. Koliko ocen je na hotel (frekvenčno število ocen na hotel) v podatkovnem naboru?
5. Čeprav obstaja stolpec `Average_Score` za vsak hotel v podatkovnem naboru, lahko izračunate tudi povprečno oceno (pridobite povprečje vseh ocen ocenjevalcev v podatkovnem naboru za vsak hotel). Dodajte nov stolpec v svoj dataframe z naslovom stolpca `Calc_Average_Score`, ki vsebuje to izračunano povprečje.
6. Ali imajo hoteli enako (zaokroženo na 1 decimalno mesto) vrednost `Average_Score` in `Calc_Average_Score`?
   1. Poskusite napisati funkcijo v Pythonu, ki sprejme Series (vrstico) kot argument in primerja vrednosti, pri čemer izpiše sporočilo, ko vrednosti niso enake. Nato uporabite metodo `.apply()`, da obdelate vsako vrstico s funkcijo.
7. Izračunajte in izpišite, koliko vrstic ima stolpec `Negative_Review` z vrednostjo "No Negative".
8. Izračunajte in izpišite, koliko vrstic ima stolpec `Positive_Review` z vrednostjo "No Positive".
9. Izračunajte in izpišite, koliko vrstic ima stolpec `Positive_Review` z vrednostjo "No Positive" **in** stolpec `Negative_Review` z vrednostjo "No Negative".

### Odgovori s kodo

1. Izpišite *obliko* dataframe, ki ste ga pravkar naložili (oblika je število vrstic in stolpcev).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Izračunajte frekvenčno število za narodnosti ocenjevalcev:

   1. Koliko različnih vrednosti je v stolpcu `Reviewer_Nationality` in katere so?
   2. Katera narodnost ocenjevalcev je najpogostejša v podatkovnem naboru (izpišite državo in število ocen)?

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

   3. Katere so naslednjih 10 najpogostejših narodnosti in njihovo frekvenčno število?

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

3. Kateri hotel je bil najpogosteje ocenjen za vsako od 10 najpogostejših narodnosti ocenjevalcev?

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

4. Koliko ocen je na hotel (frekvenčno število ocen na hotel) v podatkovnem naboru?

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
   
   Morda boste opazili, da se rezultati *štetja v podatkovnem naboru* ne ujemajo z vrednostjo v `Total_Number_of_Reviews`. Ni jasno, ali ta vrednost v podatkovnem naboru predstavlja skupno število ocen, ki jih je hotel imel, vendar niso bile vse pridobljene, ali pa je bil uporabljen kakšen drug izračun. `Total_Number_of_Reviews` se v modelu ne uporablja zaradi te nejasnosti.

5. Čeprav obstaja stolpec `Average_Score` za vsak hotel v podatkovnem naboru, lahko izračunate tudi povprečno oceno (pridobite povprečje vseh ocen ocenjevalcev v podatkovnem naboru za vsak hotel). Dodajte nov stolpec v svoj dataframe z naslovom stolpca `Calc_Average_Score`, ki vsebuje to izračunano povprečje. Izpišite stolpce `Hotel_Name`, `Average_Score` in `Calc_Average_Score`.

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

   Morda se sprašujete o vrednosti `Average_Score` in zakaj se včasih razlikuje od izračunanega povprečja. Ker ne moremo vedeti, zakaj se nekatere vrednosti ujemajo, druge pa imajo razlike, je v tem primeru najvarneje uporabiti ocene, ki jih imamo, za izračun povprečja sami. Kljub temu so razlike običajno zelo majhne, tukaj so hoteli z največjim odstopanjem med povprečjem iz podatkovnega niza in izračunanim povprečjem:

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

   Ker ima le 1 hotel razliko v oceni večjo od 1, to pomeni, da lahko verjetno ignoriramo razliko in uporabimo izračunano povprečno oceno.

6. Izračunajte in izpišite, koliko vrstic ima stolpec `Negative_Review` z vrednostjo "No Negative".

7. Izračunajte in izpišite, koliko vrstic ima stolpec `Positive_Review` z vrednostjo "No Positive".

8. Izračunajte in izpišite, koliko vrstic ima stolpec `Positive_Review` z vrednostjo "No Positive" **in** stolpec `Negative_Review` z vrednostjo "No Negative".

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

## Drug način

Drug način za štetje elementov brez Lambdas in uporaba funkcije sum za štetje vrstic:

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

   Morda ste opazili, da je 127 vrstic, ki imajo vrednosti "No Negative" in "No Positive" v stolpcih `Negative_Review` in `Positive_Review`. To pomeni, da je ocenjevalec hotelu dal numerično oceno, vendar se je odločil, da ne napiše niti pozitivne niti negativne ocene. Na srečo je to majhno število vrstic (127 od 515738, ali 0,02%), zato verjetno ne bo vplivalo na naš model ali rezultate v nobeni smeri, vendar morda ne bi pričakovali, da bo podatkovni nabor ocen vseboval vrstice brez ocen, zato je vredno raziskati podatke, da odkrijete takšne vrstice.

Zdaj, ko ste raziskali podatkovni nabor, boste v naslednji lekciji filtrirali podatke in dodali analizo sentimenta.

---
## 🚀Izziv

Ta lekcija prikazuje, kot smo videli v prejšnjih lekcijah, kako izjemno pomembno je razumeti svoje podatke in njihove posebnosti, preden na njih izvajate operacije. Besedilni podatki, zlasti, zahtevajo skrbno preučitev. Prebrskajte različne podatkovne nize, bogate z besedilom, in preverite, ali lahko odkrijete področja, ki bi lahko v model vnesla pristranskost ali izkrivljen sentiment.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Vzemite [to učno pot o NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott), da odkrijete orodja, ki jih lahko preizkusite pri gradnji modelov, bogatih z govorom in besedilom.

## Naloga 

[NLTK](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da se zavedate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.