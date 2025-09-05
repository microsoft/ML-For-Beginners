<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T13:59:21+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "hr"
}
-->
# Analiza sentimenta s recenzijama hotela - obrada podataka

U ovom dijelu koristit ćete tehnike iz prethodnih lekcija za provođenje istraživačke analize velikog skupa podataka. Kada steknete dobar uvid u korisnost različitih stupaca, naučit ćete:

- kako ukloniti nepotrebne stupce
- kako izračunati nove podatke na temelju postojećih stupaca
- kako spremiti dobiveni skup podataka za korištenje u završnom izazovu

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

### Uvod

Do sada ste naučili kako se tekstualni podaci značajno razlikuju od numeričkih podataka. Ako je tekst napisan ili izgovoren od strane čovjeka, može se analizirati kako bi se pronašli obrasci, učestalosti, sentiment i značenje. Ova lekcija vas uvodi u stvarni skup podataka sa stvarnim izazovom: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** koji uključuje [CC0: Public Domain licencu](https://creativecommons.org/publicdomain/zero/1.0/). Podaci su prikupljeni s Booking.com-a iz javnih izvora. Autor skupa podataka je Jiashen Liu.

### Priprema

Trebat će vam:

* Mogućnost pokretanja .ipynb bilježnica koristeći Python 3
* pandas
* NLTK, [koji biste trebali instalirati lokalno](https://www.nltk.org/install.html)
* Skup podataka dostupan na Kaggleu [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Skup podataka ima oko 230 MB kada se raspakira. Preuzmite ga u korijensku mapu `/data` povezanu s ovim NLP lekcijama.

## Istraživačka analiza podataka

Ovaj izazov pretpostavlja da gradite bot za preporuku hotela koristeći analizu sentimenta i ocjene gostiju. Skup podataka koji ćete koristiti uključuje recenzije 1493 različita hotela u 6 gradova.

Koristeći Python, skup podataka recenzija hotela i NLTK-ovu analizu sentimenta, mogli biste otkriti:

* Koje su najčešće korištene riječi i fraze u recenzijama?
* Jesu li službeni *tagovi* koji opisuju hotel povezani s ocjenama recenzija (npr. postoje li negativnije recenzije za određeni hotel od strane *Obitelji s malom djecom* nego od *Solo putnika*, što možda ukazuje da je hotel bolji za *Solo putnike*)?
* Slažu li se NLTK ocjene sentimenta s numeričkom ocjenom recenzenta?

#### Skup podataka

Istražimo skup podataka koji ste preuzeli i spremili lokalno. Otvorite datoteku u uređivaču poput VS Codea ili čak Excela.

Naslovi u skupu podataka su sljedeći:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Ovdje su grupirani na način koji bi mogao biti lakši za pregled: 
##### Stupci hotela

* `Hotel_Name`, `Hotel_Address`, `lat` (geografska širina), `lng` (geografska dužina)
  * Koristeći *lat* i *lng* mogli biste nacrtati kartu u Pythonu koja prikazuje lokacije hotela (možda obojene prema negativnim i pozitivnim recenzijama)
  * Hotel_Address nije očito koristan za nas, i vjerojatno ćemo ga zamijeniti državom radi lakšeg sortiranja i pretraživanja

**Meta-recenzijski stupci hotela**

* `Average_Score`
  * Prema autoru skupa podataka, ovaj stupac predstavlja *Prosječnu ocjenu hotela, izračunatu na temelju najnovijeg komentara u posljednjih godinu dana*. Ovo se čini neobičnim načinom izračuna ocjene, ali to su podaci prikupljeni pa ih za sada možemo uzeti zdravo za gotovo.
  
  ✅ Na temelju drugih stupaca u ovom skupu podataka, možete li smisliti drugi način za izračun prosječne ocjene?

* `Total_Number_of_Reviews`
  * Ukupan broj recenzija koje je hotel primio - nije jasno (bez pisanja koda) odnosi li se to na recenzije u skupu podataka.
* `Additional_Number_of_Scoring`
  * Ovo znači da je ocjena dana, ali nije napisana pozitivna ili negativna recenzija od strane recenzenta.

**Stupci recenzija**

- `Reviewer_Score`
  - Ovo je numerička vrijednost s najviše jednom decimalom između minimalne i maksimalne vrijednosti 2.5 i 10
  - Nije objašnjeno zašto je 2.5 najniža moguća ocjena
- `Negative_Review`
  - Ako recenzent nije ništa napisao, ovo polje će sadržavati "**No Negative**"
  - Imajte na umu da recenzent može napisati pozitivnu recenziju u stupcu Negative review (npr. "nema ništa loše u ovom hotelu")
- `Review_Total_Negative_Word_Counts`
  - Veći broj negativnih riječi ukazuje na nižu ocjenu (bez provjere sentimentalnosti)
- `Positive_Review`
  - Ako recenzent nije ništa napisao, ovo polje će sadržavati "**No Positive**"
  - Imajte na umu da recenzent može napisati negativnu recenziju u stupcu Positive review (npr. "u ovom hotelu nema ništa dobro")
- `Review_Total_Positive_Word_Counts`
  - Veći broj pozitivnih riječi ukazuje na višu ocjenu (bez provjere sentimentalnosti)
- `Review_Date` i `days_since_review`
  - Može se primijeniti mjera svježine ili zastarjelosti recenzije (starije recenzije možda nisu toliko točne kao novije jer se uprava hotela promijenila, izvršene su renovacije, dodan je bazen itd.)
- `Tags`
  - Ovo su kratki opisi koje recenzent može odabrati kako bi opisao vrstu gosta (npr. solo ili obitelj), vrstu sobe koju su imali, duljinu boravka i način na koji je recenzija poslana.
  - Nažalost, korištenje ovih tagova je problematično, pogledajte odjeljak u nastavku koji raspravlja o njihovoj korisnosti.

**Stupci recenzenta**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Ovo bi moglo biti faktor u modelu preporuke, na primjer, ako možete utvrditi da su plodniji recenzenti s desecima recenzija skloniji negativnim nego pozitivnim ocjenama. Međutim, recenzent bilo koje pojedine recenzije nije identificiran jedinstvenim kodom, i stoga se ne može povezati s nizom recenzija. Postoji 30 recenzenata s 100 ili više recenzija, ali teško je vidjeti kako to može pomoći modelu preporuke.
- `Reviewer_Nationality`
  - Neki bi mogli pomisliti da su određene nacionalnosti sklonije davanju pozitivnih ili negativnih recenzija zbog nacionalne sklonosti. Budite oprezni pri uključivanju takvih anegdotalnih pogleda u svoje modele. Ovo su nacionalni (a ponekad i rasni) stereotipi, a svaki recenzent je bio pojedinac koji je napisao recenziju na temelju svog iskustva. To iskustvo moglo je biti filtrirano kroz mnoge leće poput njihovih prethodnih boravaka u hotelima, udaljenosti koju su putovali i njihove osobne naravi. Teško je opravdati mišljenje da je njihova nacionalnost bila razlog za ocjenu recenzije.

##### Primjeri

| Prosječna ocjena | Ukupan broj recenzija | Ocjena recenzenta | Negativna <br />recenzija                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pozitivna recenzija               | Tagovi                                                                                      |
| ----------------- | --------------------- | ----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8              | 1945                 | 2.5              | Ovo trenutno nije hotel već gradilište. Teroriziran sam od ranog jutra i cijeli dan neprihvatljivom bukom građevinskih radova dok sam se odmarao nakon dugog puta i radio u sobi. Ljudi su radili cijeli dan, npr. s bušilicama u susjednim sobama. Zatražio sam promjenu sobe, ali nije bilo tihe sobe. Da stvar bude gora, naplaćeno mi je više nego što je trebalo. Odjavio sam se navečer jer sam morao rano na let i dobio odgovarajući račun. Dan kasnije hotel je napravio dodatnu naplatu bez mog pristanka, iznad dogovorene cijene. Strašno mjesto. Nemojte se kažnjavati rezervacijom ovdje. | Ništa. Strašno mjesto. Klonite se. | Poslovno putovanje, Par, Standardna dvokrevetna soba, Boravak 2 noći |

Kao što možete vidjeti, ovaj gost nije imao sretan boravak u ovom hotelu. Hotel ima dobru prosječnu ocjenu od 7.8 i 1945 recenzija, ali ovaj recenzent dao je ocjenu 2.5 i napisao 115 riječi o tome koliko je njihov boravak bio negativan. Ako nisu ništa napisali u stupcu Positive_Review, mogli biste zaključiti da nije bilo ništa pozitivno, ali ipak su napisali 7 riječi upozorenja. Ako bismo samo brojali riječi umjesto značenja ili sentimenta riječi, mogli bismo imati iskrivljen pogled na namjeru recenzenta. Zanimljivo, njihova ocjena od 2.5 je zbunjujuća, jer ako je boravak u hotelu bio tako loš, zašto uopće dati bodove? Istražujući skup podataka pažljivo, vidjet ćete da je najniža moguća ocjena 2.5, a ne 0. Najviša moguća ocjena je 10.

##### Tagovi

Kao što je gore spomenuto, na prvi pogled ideja korištenja `Tags` za kategorizaciju podataka ima smisla. Nažalost, ovi tagovi nisu standardizirani, što znači da u određenom hotelu opcije mogu biti *Single room*, *Twin room* i *Double room*, ali u sljedećem hotelu, one su *Deluxe Single Room*, *Classic Queen Room* i *Executive King Room*. Ovo bi moglo biti isto, ali postoji toliko varijacija da izbor postaje:

1. Pokušati promijeniti sve izraze u jedan standard, što je vrlo teško, jer nije jasno koji bi put konverzije bio u svakom slučaju (npr. *Classic single room* mapira se na *Single room*, ali *Superior Queen Room with Courtyard Garden or City View* je mnogo teže mapirati)

1. Možemo pristupiti NLP-u i mjeriti učestalost određenih izraza poput *Solo*, *Business Traveller* ili *Family with young kids* kako se primjenjuju na svaki hotel, i to uključiti u preporuku  

Tagovi su obično (ali ne uvijek) jedno polje koje sadrži popis od 5 do 6 vrijednosti odvojenih zarezima koje se odnose na *Vrstu putovanja*, *Vrstu gostiju*, *Vrstu sobe*, *Broj noćenja* i *Vrstu uređaja na kojem je recenzija poslana*. Međutim, budući da neki recenzenti ne popunjavaju svako polje (mogu ostaviti jedno prazno), vrijednosti nisu uvijek u istom redoslijedu.

Na primjer, uzmite *Vrstu grupe*. Postoji 1025 jedinstvenih mogućnosti u ovom polju u stupcu `Tags`, i nažalost samo neki od njih odnose se na grupu (neki su vrsta sobe itd.). Ako filtrirate samo one koji spominju obitelj, rezultati sadrže mnoge rezultate tipa *Family room*. Ako uključite izraz *with*, tj. brojite vrijednosti *Family with*, rezultati su bolji, s preko 80,000 od 515,000 rezultata koji sadrže frazu "Family with young children" ili "Family with older children".

To znači da stupac tagova nije potpuno beskoristan za nas, ali će zahtijevati određeni rad kako bi bio koristan.

##### Prosječna ocjena hotela

Postoji niz neobičnosti ili neslaganja u skupu podataka koje ne mogu objasniti, ali su ovdje ilustrirane kako biste bili svjesni njih prilikom izrade svojih modela. Ako ih uspijete razjasniti, javite nam u odjeljku za raspravu!

Skup podataka ima sljedeće stupce koji se odnose na prosječnu ocjenu i broj recenzija:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hotel s najviše recenzija u ovom skupu podataka je *Britannia International Hotel Canary Wharf* s 4789 recenzija od ukupno 515,000. Ali ako pogledamo vrijednost `Total_Number_of_Reviews` za ovaj hotel, ona iznosi 9086. Mogli biste zaključiti da postoji mnogo više ocjena bez recenzija, pa možda trebamo dodati vrijednost stupca `Additional_Number_of_Scoring`. Ta vrijednost iznosi 2682, a dodavanjem na 4789 dobivamo 7471, što je još uvijek 1615 manje od `Total_Number_of_Reviews`.

Ako uzmete stupac `Average_Score`, mogli biste zaključiti da je to prosjek recenzija u skupu podataka, ali opis s Kagglea je "*Prosječna ocjena hotela, izračunata na temelju najnovijeg komentara u posljednjih godinu dana*". To se ne čini korisnim, ali možemo izračunati vlastiti prosjek na temelju ocjena recenzenata u skupu podataka. Koristeći isti hotel kao primjer, prosječna ocjena hotela je navedena kao 7.1, ali izračunata ocjena (prosječna ocjena recenzenata *u* skupu podataka) je 6.8. Ovo je blizu, ali nije ista vrijednost, i možemo samo pretpostaviti da su ocjene dane u recenzijama `Additional_Number_of_Scoring` povećale prosjek na 7.1. Nažalost, bez načina da testiramo ili dokažemo tu tvrdnju, teško je koristiti ili vjerovati `Average_Score`, `Additional_Number_of_Scoring` i `Total_Number_of_Reviews` kada se temelje na podacima koje nemamo.

Da dodatno zakompliciramo stvari, hotel s drugim najvećim brojem recenzija ima izračunatu prosječnu ocjenu od 8.12, dok je `Average_Score` u skupu podataka 8.1. Je li ova točna ocjena slučajnost ili je prvi hotel neslaganje?

S obzirom na mogućnost da bi ovi hoteli mogli biti iznimke, i da možda većina vrijednosti odgovara (ali neke ne iz nekog razloga), sljedeće ćemo napisati kratki program za istraživanje vrijednosti u skupu podataka i određivanje ispravne upotrebe (ili neupotrebe) vrijednosti.
> 🚨 Napomena upozorenja
>
> Kada radite s ovim skupom podataka, pisat ćete kod koji izračunava nešto iz teksta bez potrebe da sami čitate ili analizirate tekst. Ovo je suština NLP-a, interpretiranje značenja ili sentimenta bez potrebe da to radi čovjek. Međutim, moguće je da ćete pročitati neke od negativnih recenzija. Preporučujem da to ne radite, jer nije potrebno. Neke od njih su glupe ili nebitne negativne recenzije hotela, poput "Vrijeme nije bilo dobro", što je izvan kontrole hotela, ili bilo koga drugog. No, postoji i mračna strana nekih recenzija. Ponekad su negativne recenzije rasističke, seksističke ili diskriminiraju na temelju dobi. To je nesretno, ali očekivano u skupu podataka prikupljenom s javne web stranice. Neki recenzenti ostavljaju recenzije koje biste mogli smatrati neukusnima, neugodnima ili uznemirujućima. Bolje je pustiti kod da izmjeri sentiment nego da ih sami čitate i uzrujate se. Ipak, manjina je onih koji pišu takve stvari, ali oni ipak postoje.
## Vježba - Istraživanje podataka
### Učitajte podatke

Dosta je bilo vizualnog pregleda podataka, sada ćete napisati malo koda i dobiti odgovore! Ovaj dio koristi biblioteku pandas. Vaš prvi zadatak je osigurati da možete učitati i pročitati CSV podatke. Biblioteka pandas ima brz učitavač za CSV, a rezultat se smješta u dataframe, kao u prethodnim lekcijama. CSV koji učitavamo ima više od pola milijuna redaka, ali samo 17 stupaca. Pandas vam pruža mnogo moćnih načina za rad s dataframeovima, uključujući mogućnost izvođenja operacija na svakom retku.

Od sada nadalje u ovoj lekciji, bit će isječaka koda, objašnjenja koda i rasprava o tome što rezultati znače. Koristite priloženi _notebook.ipynb_ za svoj kod.

Počnimo s učitavanjem datoteke s podacima koju ćete koristiti:

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

Sada kada su podaci učitani, možemo izvoditi neke operacije na njima. Držite ovaj kod na vrhu svog programa za sljedeći dio.

## Istražite podatke

U ovom slučaju, podaci su već *čisti*, što znači da su spremni za rad i ne sadrže znakove na drugim jezicima koji bi mogli zbuniti algoritme koji očekuju samo engleske znakove.

✅ Možda ćete morati raditi s podacima koji zahtijevaju početnu obradu kako bi se formatirali prije primjene NLP tehnika, ali ne ovaj put. Da morate, kako biste se nosili s ne-engleskim znakovima?

Odvojite trenutak da osigurate da, nakon što su podaci učitani, možete ih istražiti pomoću koda. Vrlo je lako poželjeti se usredotočiti na stupce `Negative_Review` i `Positive_Review`. Oni su ispunjeni prirodnim tekstom za vaše NLP algoritme. Ali pričekajte! Prije nego što se upustite u NLP i analizu sentimenta, trebali biste slijediti kod u nastavku kako biste utvrdili odgovaraju li vrijednosti u skupu podataka vrijednostima koje izračunate pomoću pandas.

## Operacije s dataframeovima

Prvi zadatak u ovoj lekciji je provjeriti jesu li sljedeće tvrdnje točne pisanjem koda koji ispituje dataframe (bez mijenjanja istog).

> Kao i kod mnogih programerskih zadataka, postoji nekoliko načina za njihovo rješavanje, ali dobar savjet je učiniti to na najjednostavniji i najlakši način, posebno ako će biti lakše razumjeti kada se vratite ovom kodu u budućnosti. Kod dataframeova postoji sveobuhvatan API koji često ima način da učinkovito postignete ono što želite.

Tretirajte sljedeća pitanja kao zadatke kodiranja i pokušajte ih riješiti bez gledanja rješenja.

1. Ispišite *oblik* dataframea koji ste upravo učitali (oblik je broj redaka i stupaca).
2. Izračunajte učestalost pojavljivanja nacionalnosti recenzenata:
   1. Koliko različitih vrijednosti postoji za stupac `Reviewer_Nationality` i koje su to vrijednosti?
   2. Koja je nacionalnost recenzenata najčešća u skupu podataka (ispisati državu i broj recenzija)?
   3. Koje su sljedećih 10 najčešće pronađenih nacionalnosti i njihova učestalost?
3. Koji je hotel najčešće recenziran za svaku od 10 najčešćih nacionalnosti recenzenata?
4. Koliko recenzija ima po hotelu (učestalost recenzija po hotelu) u skupu podataka?
5. Iako postoji stupac `Average_Score` za svaki hotel u skupu podataka, možete također izračunati prosječnu ocjenu (dobivanjem prosjeka svih ocjena recenzenata u skupu podataka za svaki hotel). Dodajte novi stupac u svoj dataframe s naslovom stupca `Calc_Average_Score` koji sadrži taj izračunati prosjek.
6. Imaju li neki hoteli iste (zaokružene na jednu decimalu) vrijednosti `Average_Score` i `Calc_Average_Score`?
   1. Pokušajte napisati Python funkciju koja uzima Series (redak) kao argument i uspoređuje vrijednosti, ispisujući poruku kada vrijednosti nisu jednake. Zatim koristite metodu `.apply()` za obradu svakog retka pomoću funkcije.
7. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Negative_Review` kao "No Negative".
8. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Positive_Review` kao "No Positive".
9. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Positive_Review` kao "No Positive" **i** `Negative_Review` kao "No Negative".

### Odgovori u kodu

1. Ispišite *oblik* dataframea koji ste upravo učitali (oblik je broj redaka i stupaca).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Izračunajte učestalost pojavljivanja nacionalnosti recenzenata:

   1. Koliko različitih vrijednosti postoji za stupac `Reviewer_Nationality` i koje su to vrijednosti?
   2. Koja je nacionalnost recenzenata najčešća u skupu podataka (ispisati državu i broj recenzija)?

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

   3. Koje su sljedećih 10 najčešće pronađenih nacionalnosti i njihova učestalost?

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

3. Koji je hotel najčešće recenziran za svaku od 10 najčešćih nacionalnosti recenzenata?

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

4. Koliko recenzija ima po hotelu (učestalost recenzija po hotelu) u skupu podataka?

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

   Možda ćete primijetiti da se *broj recenzija u skupu podataka* ne podudara s vrijednošću u `Total_Number_of_Reviews`. Nije jasno predstavlja li ova vrijednost ukupan broj recenzija koje je hotel imao, ali nisu sve prikupljene, ili neki drugi izračun. `Total_Number_of_Reviews` se ne koristi u modelu zbog ove nejasnoće.

5. Iako postoji stupac `Average_Score` za svaki hotel u skupu podataka, možete također izračunati prosječnu ocjenu (dobivanjem prosjeka svih ocjena recenzenata u skupu podataka za svaki hotel). Dodajte novi stupac u svoj dataframe s naslovom stupca `Calc_Average_Score` koji sadrži taj izračunati prosjek. Ispišite stupce `Hotel_Name`, `Average_Score` i `Calc_Average_Score`.

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

   Možda ćete se zapitati o vrijednosti `Average_Score` i zašto je ponekad različita od izračunate prosječne ocjene. Kako ne možemo znati zašto se neke vrijednosti podudaraju, a druge imaju razliku, najsigurnije je u ovom slučaju koristiti ocjene recenzenata koje imamo za izračun prosjeka. Ipak, razlike su obično vrlo male, evo hotela s najvećim odstupanjem između prosječne ocjene iz skupa podataka i izračunate prosječne ocjene:

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

   S obzirom na to da samo 1 hotel ima razliku u ocjeni veću od 1, to znači da vjerojatno možemo zanemariti razliku i koristiti izračunatu prosječnu ocjenu.

6. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Negative_Review` kao "No Negative".

7. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Positive_Review` kao "No Positive".

8. Izračunajte i ispišite koliko redaka ima vrijednosti stupca `Positive_Review` kao "No Positive" **i** `Negative_Review` kao "No Negative".

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

## Drugi način

Drugi način brojanja stavki bez Lambdi, koristeći sum za brojanje redaka:

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

   Možda ste primijetili da postoji 127 redaka koji imaju i "No Negative" i "No Positive" vrijednosti za stupce `Negative_Review` i `Positive_Review`. To znači da je recenzent dao hotelu numeričku ocjenu, ali je odbio napisati pozitivnu ili negativnu recenziju. Srećom, ovo je mali broj redaka (127 od 515738, ili 0,02%), tako da vjerojatno neće utjecati na naš model ili rezultate u bilo kojem smjeru, ali možda niste očekivali da skup podataka sadrži retke bez recenzija, pa vrijedi istražiti podatke kako biste otkrili takve retke.

Sada kada ste istražili skup podataka, u sljedećoj lekciji filtrirat ćete podatke i dodati analizu sentimenta.

---
## 🚀Izazov

Ova lekcija pokazuje, kao što smo vidjeli u prethodnim lekcijama, koliko je kritično važno razumjeti svoje podatke i njihove specifičnosti prije izvođenja operacija na njima. Tekstualni podaci, posebno, zahtijevaju pažljivo ispitivanje. Pregledajte različite skupove podataka bogate tekstom i provjerite možete li otkriti područja koja bi mogla unijeti pristranost ili iskrivljeni sentiment u model.

## [Post-lecture kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Prođite [ovaj put učenja o NLP-u](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) kako biste otkrili alate koje možete isprobati prilikom izrade modela temeljenih na govoru i tekstu.

## Zadatak 

[NLTK](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.