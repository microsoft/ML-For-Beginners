<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-08-29T14:27:24+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "tl"
}
-->
# Sentiment analysis gamit ang mga review ng hotel - pagproseso ng data

Sa seksyong ito, gagamitin mo ang mga teknik mula sa mga nakaraang aralin upang magsagawa ng exploratory data analysis sa isang malaking dataset. Kapag nakuha mo na ang tamang pag-unawa sa kahalagahan ng iba't ibang mga column, matutunan mo:

- kung paano alisin ang mga hindi kinakailangang column
- kung paano kalkulahin ang bagong data batay sa mga umiiral na column
- kung paano i-save ang resulta ng dataset para magamit sa huling hamon

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Panimula

Sa ngayon, natutunan mo na kung paano ang text data ay lubos na naiiba sa numerical na uri ng data. Kung ito ay text na isinulat o sinabi ng tao, maaari itong suriin upang makita ang mga pattern, dalas, damdamin, at kahulugan. Ang araling ito ay magdadala sa iyo sa isang tunay na dataset na may tunay na hamon: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** na may kasamang [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). Ang data ay kinuha mula sa Booking.com mula sa mga pampublikong mapagkukunan. Ang tagalikha ng dataset ay si Jiashen Liu.

### Paghahanda

Kakailanganin mo:

* Kakayahang magpatakbo ng .ipynb notebooks gamit ang Python 3
* pandas
* NLTK, [na dapat mong i-install nang lokal](https://www.nltk.org/install.html)
* Ang dataset na makukuha sa Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ito ay humigit-kumulang 230 MB kapag na-unzip. I-download ito sa root `/data` folder na nauugnay sa mga araling NLP na ito.

## Exploratory data analysis

Ang hamon na ito ay nagpapalagay na ikaw ay gumagawa ng hotel recommendation bot gamit ang sentiment analysis at mga score ng review ng mga bisita. Ang dataset na gagamitin mo ay naglalaman ng mga review ng 1493 iba't ibang hotel sa 6 na lungsod.

Gamit ang Python, isang dataset ng mga review ng hotel, at ang sentiment analysis ng NLTK, maaari mong tukuyin:

* Ano ang mga pinakaginagamit na salita at parirala sa mga review?
* Ang mga opisyal na *tags* ba na naglalarawan sa isang hotel ay may kaugnayan sa mga score ng review (halimbawa, mas negatibo ba ang mga review para sa isang partikular na hotel mula sa *Family with young children* kaysa sa *Solo traveller*, na maaaring magpahiwatig na mas angkop ito para sa *Solo travellers*?)
* Ang mga sentiment score ng NLTK ba ay "sumasang-ayon" sa numerical score ng reviewer ng hotel?

#### Dataset

Suriin natin ang dataset na na-download mo at na-save nang lokal. Buksan ang file sa isang editor tulad ng VS Code o kahit Excel.

Ang mga header sa dataset ay ang mga sumusunod:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Narito ang mga ito na naka-grupo sa paraang mas madaling suriin: 
##### Mga column ng hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Gamit ang *lat* at *lng* maaari kang mag-plot ng mapa gamit ang Python na nagpapakita ng mga lokasyon ng hotel (maaaring naka-color code para sa negatibo at positibong review)
  * Ang Hotel_Address ay hindi mukhang kapaki-pakinabang sa atin, at malamang na papalitan natin ito ng bansa para sa mas madaling pag-uuri at paghahanap

**Mga Meta-review column ng hotel**

* `Average_Score`
  * Ayon sa tagalikha ng dataset, ang column na ito ay ang *Average Score ng hotel, na kinakalkula batay sa pinakabagong komento sa nakaraang taon*. Mukhang hindi pangkaraniwang paraan ito ng pagkalkula ng score, ngunit ito ang data na nakuha kaya maaaring tanggapin natin ito sa ngayon. 
  
  ✅ Batay sa iba pang mga column sa data na ito, makakaisip ka ba ng ibang paraan upang kalkulahin ang average score?

* `Total_Number_of_Reviews`
  * Ang kabuuang bilang ng mga review na natanggap ng hotel na ito - hindi malinaw (nang hindi nagsusulat ng code) kung ito ay tumutukoy sa mga review sa dataset.
* `Additional_Number_of_Scoring`
  * Nangangahulugan ito na may score na ibinigay ngunit walang positibo o negatibong review na isinulat ng reviewer

**Mga column ng review**

- `Reviewer_Score`
  - Ito ay isang numerical na halaga na may pinakamataas na 1 decimal place sa pagitan ng min at max na halaga na 2.5 at 10
  - Hindi ipinaliwanag kung bakit 2.5 ang pinakamababang posibleng score
- `Negative_Review`
  - Kung walang isinulat ang reviewer, ang field na ito ay magkakaroon ng "**No Negative**"
  - Tandaan na maaaring magsulat ang reviewer ng positibong review sa Negative review column (halimbawa, "walang masama sa hotel na ito")
- `Review_Total_Negative_Word_Counts`
  - Mas mataas na bilang ng negatibong salita ay nagpapahiwatig ng mas mababang score (nang hindi sinusuri ang damdamin)
- `Positive_Review`
  - Kung walang isinulat ang reviewer, ang field na ito ay magkakaroon ng "**No Positive**"
  - Tandaan na maaaring magsulat ang reviewer ng negatibong review sa Positive review column (halimbawa, "walang maganda sa hotel na ito")
- `Review_Total_Positive_Word_Counts`
  - Mas mataas na bilang ng positibong salita ay nagpapahiwatig ng mas mataas na score (nang hindi sinusuri ang damdamin)
- `Review_Date` at `days_since_review`
  - Maaaring mag-apply ng freshness o staleness measure sa isang review (ang mas lumang review ay maaaring hindi kasing-accurate ng mas bago dahil nagbago ang pamamahala ng hotel, o may mga renovation na ginawa, o may idinagdag na pool, atbp.)
- `Tags`
  - Ito ay mga maikling descriptor na maaaring piliin ng reviewer upang ilarawan ang uri ng bisita sila (halimbawa, solo o pamilya), ang uri ng kwarto na kanilang tinirhan, ang haba ng pananatili, at kung paano naisumite ang review. 
  - Sa kasamaang palad, ang paggamit ng mga tag na ito ay may problema, tingnan ang seksyon sa ibaba na naglalarawan ng kanilang kahalagahan

**Mga column ng reviewer**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Maaaring maging salik ito sa isang recommendation model, halimbawa, kung matutukoy mo na ang mas prolific na reviewer na may daan-daang review ay mas malamang na maging negatibo kaysa positibo. Gayunpaman, ang reviewer ng anumang partikular na review ay hindi nakilala gamit ang isang natatanging code, at samakatuwid ay hindi maaaring maiugnay sa isang set ng mga review. Mayroong 30 reviewer na may 100 o higit pang mga review, ngunit mahirap makita kung paano ito makakatulong sa recommendation model.
- `Reviewer_Nationality`
  - Ang ilang tao ay maaaring mag-isip na ang ilang nasyonalidad ay mas malamang na magbigay ng positibo o negatibong review dahil sa isang pambansang hilig. Mag-ingat sa pagbuo ng ganitong anecdotal na pananaw sa iyong mga modelo. Ito ay mga pambansa (at kung minsan ay lahi) na stereotype, at bawat reviewer ay isang indibidwal na nagsulat ng review batay sa kanilang karanasan. Maaaring na-filter ito sa maraming lens tulad ng kanilang mga nakaraang pananatili sa hotel, ang distansya ng paglalakbay, at ang kanilang personal na ugali. Ang pag-iisip na ang kanilang nasyonalidad ang dahilan ng score ng review ay mahirap patunayan.

##### Mga Halimbawa

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Ito ay kasalukuyang hindi isang hotel kundi isang construction site Ako ay ginising mula sa maagang umaga at buong araw ng hindi katanggap-tanggap na ingay ng konstruksyon habang nagpapahinga pagkatapos ng mahabang biyahe at nagtatrabaho sa kwarto Ang mga tao ay nagtatrabaho buong araw gamit ang jackhammers sa mga katabing kwarto Humiling ako ng pagbabago ng kwarto ngunit walang tahimik na kwarto na magagamit Upang gawing mas masama, ako ay siningil ng sobra Nag-check out ako sa gabi dahil kailangan kong umalis ng maagang flight at nakatanggap ng tamang bill Isang araw pagkatapos, ang hotel ay gumawa ng isa pang singil nang walang pahintulot na mas mataas sa nakabook na presyo Napakasamang lugar Huwag parusahan ang sarili sa pag-book dito | Wala  Napakasamang lugar Lumayo | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Makikita mo, ang bisitang ito ay hindi nagkaroon ng masayang pananatili sa hotel na ito. Ang hotel ay may magandang average score na 7.8 at 1945 review, ngunit ang reviewer na ito ay nagbigay ng 2.5 at nagsulat ng 115 salita tungkol sa kung gaano ka-negatibo ang kanilang pananatili. Kung wala silang isinulat sa Positive_Review column, maaari mong ipalagay na walang positibo, ngunit isinulat nila ang 7 salita ng babala. Kung bibilangin lang natin ang mga salita sa halip na ang kahulugan o damdamin ng mga salita, maaaring magkaroon tayo ng maling pananaw sa intensyon ng reviewer. Nakakagulat, ang kanilang score na 2.5 ay nakakalito, dahil kung napakasama ng pananatili sa hotel, bakit pa magbigay ng anumang puntos? Sa pagsisiyasat sa dataset nang malapitan, makikita mo na ang pinakamababang posibleng score ay 2.5, hindi 0. Ang pinakamataas na posibleng score ay 10.

##### Tags

Tulad ng nabanggit sa itaas, sa unang tingin, ang ideya na gamitin ang `Tags` upang i-categorize ang data ay may katuturan. Sa kasamaang palad, ang mga tag na ito ay hindi standardized, na nangangahulugan na sa isang hotel, ang mga opsyon ay maaaring *Single room*, *Twin room*, at *Double room*, ngunit sa susunod na hotel, ang mga ito ay *Deluxe Single Room*, *Classic Queen Room*, at *Executive King Room*. Maaaring pareho ang mga ito, ngunit napakaraming variation na ang pagpipilian ay:

1. Subukang baguhin ang lahat ng termino sa isang pamantayan, na napakahirap, dahil hindi malinaw kung ano ang conversion path sa bawat kaso (halimbawa, *Classic single room* ay tumutugma sa *Single room* ngunit *Superior Queen Room with Courtyard Garden or City View* ay mas mahirap i-map)

1. Maaari tayong gumamit ng NLP approach at sukatin ang dalas ng ilang termino tulad ng *Solo*, *Business Traveller*, o *Family with young kids* habang ito ay naaangkop sa bawat hotel, at isama iyon sa rekomendasyon  

Ang mga tag ay karaniwang (ngunit hindi palaging) isang solong field na naglalaman ng listahan ng 5 hanggang 6 na comma-separated values na naaayon sa *Uri ng biyahe*, *Uri ng bisita*, *Uri ng kwarto*, *Bilang ng gabi*, at *Uri ng device kung saan naisumite ang review*. Gayunpaman, dahil ang ilang reviewer ay hindi pinupunan ang bawat field (maaaring iwanan nila ang isa na blangko), ang mga value ay hindi palaging nasa parehong pagkakasunod-sunod.

Halimbawa, kunin ang *Uri ng grupo*. Mayroong 1025 natatanging posibilidad sa field na ito sa `Tags` column, at sa kasamaang palad, ilan lamang sa mga ito ang tumutukoy sa isang grupo (ang ilan ay uri ng kwarto, atbp.). Kung i-filter mo lang ang mga nagbabanggit ng pamilya, ang mga resulta ay naglalaman ng maraming *Family room* na uri ng resulta. Kung isasama mo ang terminong *with*, i.e. bilangin ang *Family with* values, mas maganda ang mga resulta, na may higit sa 80,000 sa 515,000 resulta na naglalaman ng pariralang "Family with young children" o "Family with older children".

Ibig sabihin, ang tags column ay hindi ganap na walang silbi sa atin, ngunit kakailanganin ng kaunting trabaho upang maging kapaki-pakinabang.

##### Average hotel score

Mayroong ilang mga kakaibang bagay o hindi pagkakapare-pareho sa dataset na hindi ko maipaliwanag, ngunit inilalarawan dito upang malaman mo ang mga ito kapag gumagawa ng iyong mga modelo. Kung ma-figure out mo ito, mangyaring ipaalam sa amin sa discussion section!

Ang dataset ay may mga sumusunod na column na nauugnay sa average score at bilang ng mga review:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Ang nag-iisang hotel na may pinakamaraming review sa dataset na ito ay *Britannia International Hotel Canary Wharf* na may 4789 review mula sa 515,000. Ngunit kung titingnan natin ang `Total_Number_of_Reviews` value para sa hotel na ito, ito ay 9086. Maaari mong ipalagay na may mas maraming score na walang review, kaya marahil dapat nating idagdag ang `Additional_Number_of_Scoring` column value. Ang value na iyon ay 2682, at kapag idinagdag sa 4789 ay makakakuha tayo ng 7,471 na kulang pa rin ng 1615 sa `Total_Number_of_Reviews`.

Kung kukunin mo ang `Average_Score` column, maaari mong ipalagay na ito ang average ng mga review sa dataset, ngunit ang paglalarawan mula sa Kaggle ay "*Average Score ng hotel, na kinakalkula batay sa pinakabagong komento sa nakaraang taon*". Mukhang hindi ito kapaki-pakinabang, ngunit maaari nating kalkulahin ang sarili nating average batay sa mga review score sa dataset. Gamit ang parehong hotel bilang halimbawa, ang average hotel score ay ibinigay bilang 7.1 ngunit ang calculated score (average reviewer score *sa* dataset) ay 6.8. Malapit ito, ngunit hindi pareho ang halaga, at maaari lamang nating ipalagay na ang mga score na ibinigay sa `Additional_Number_of_Scoring` reviews ay nagtaas ng average sa 7.1. Sa kasamaang palad, nang walang paraan upang subukan o patunayan ang assertion na iyon, mahirap gamitin o pagkatiwalaan ang `Average_Score`, `Additional_Number_of_Scoring` at `Total_Number_of_Reviews` kapag ang mga ito ay batay sa, o tumutukoy sa, data na wala tayo.

Upang gawing mas kumplikado ang mga bagay, ang hotel na may pangalawang pinakamataas na bilang ng mga review ay may calculated average score na 8.12 at ang dataset `Average_Score` ay 8.1. Ang tamang score ba na ito ay isang coincidence o ang unang hotel ay isang discrepancy?
Sa posibilidad na ang mga hotel na ito ay maaaring isang outlier, at na karamihan sa mga halaga ay tumutugma (ngunit ang ilan ay hindi sa ilang kadahilanan), magsusulat tayo ng maikling programa upang suriin ang mga halaga sa dataset at tukuyin ang tamang paggamit (o hindi paggamit) ng mga halaga.

> 🚨 Isang paalala ng pag-iingat
>
> Kapag nagtatrabaho sa dataset na ito, magsusulat ka ng code na nagkakalkula ng isang bagay mula sa teksto nang hindi kinakailangang basahin o suriin ang teksto mismo. Ito ang esensya ng NLP, ang pag-unawa sa kahulugan o damdamin nang hindi kinakailangang gawin ito ng tao. Gayunpaman, posible na mabasa mo ang ilan sa mga negatibong review. Hinihikayat kitang huwag gawin ito, dahil hindi mo naman kailangang gawin. Ang ilan sa mga ito ay walang kabuluhan o hindi mahalagang negatibong review ng hotel, tulad ng "Hindi maganda ang panahon," isang bagay na wala sa kontrol ng hotel, o kahit sino. Ngunit may madilim na bahagi rin sa ilang mga review. Minsan ang mga negatibong review ay rasista, seksista, o may diskriminasyon sa edad. Ito ay hindi kanais-nais ngunit inaasahan sa isang dataset na kinuha mula sa isang pampublikong website. Ang ilang mga reviewer ay nag-iiwan ng mga review na maaaring makita mong hindi kanais-nais, hindi komportable, o nakakabahala. Mas mabuting hayaan ang code na sukatin ang damdamin kaysa basahin mo ito at maapektuhan. Gayunpaman, ito ay isang maliit na bahagi lamang na nagsusulat ng ganitong mga bagay, ngunit umiiral pa rin sila.

## Ehersisyo - Paggalugad ng Data
### I-load ang Data

Sapat na ang pagsusuri sa data nang biswal, ngayon ay magsusulat ka ng code at makakakuha ng mga sagot! Ang seksyong ito ay gumagamit ng pandas library. Ang iyong unang gawain ay tiyaking ma-load at mabasa mo ang CSV data. Ang pandas library ay may mabilis na CSV loader, at ang resulta ay inilalagay sa isang dataframe, tulad ng sa mga nakaraang aralin. Ang CSV na ating ilo-load ay may higit sa kalahating milyong mga hilera, ngunit 17 kolum lamang. Ang pandas ay nagbibigay ng maraming makapangyarihang paraan upang makipag-ugnayan sa isang dataframe, kabilang ang kakayahang magsagawa ng mga operasyon sa bawat hilera.

Mula dito sa araling ito, magkakaroon ng mga snippet ng code at ilang paliwanag ng code at talakayan tungkol sa kung ano ang ibig sabihin ng mga resulta. Gamitin ang kasamang _notebook.ipynb_ para sa iyong code.

Simulan natin sa pag-load ng data file na gagamitin mo:

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

Ngayon na ang data ay na-load, maaari tayong magsagawa ng ilang mga operasyon dito. Panatilihin ang code na ito sa itaas ng iyong programa para sa susunod na bahagi.

## Galugarin ang Data

Sa kasong ito, ang data ay *malinis* na, ibig sabihin ay handa na itong gamitin, at walang mga karakter sa ibang wika na maaaring magdulot ng problema sa mga algorithm na umaasa lamang sa mga karakter sa Ingles.

✅ Maaaring kailanganin mong magtrabaho sa data na nangangailangan ng paunang pagproseso upang ma-format ito bago mag-apply ng mga NLP technique, ngunit hindi sa pagkakataong ito. Kung kailangan mo, paano mo haharapin ang mga karakter na hindi Ingles?

Maglaan ng sandali upang tiyakin na kapag na-load na ang data, maaari mo itong galugarin gamit ang code. Napakadaling mag-focus sa mga kolum na `Negative_Review` at `Positive_Review`. Ang mga ito ay puno ng natural na teksto para sa iyong mga NLP algorithm upang iproseso. Ngunit sandali! Bago ka tumalon sa NLP at damdamin, sundin ang code sa ibaba upang tiyakin kung ang mga halagang ibinigay sa dataset ay tumutugma sa mga halagang kinakalkula mo gamit ang pandas.

## Mga Operasyon sa Dataframe

Ang unang gawain sa araling ito ay tiyakin kung tama ang mga sumusunod na pahayag sa pamamagitan ng pagsulat ng code na sinusuri ang dataframe (nang hindi ito binabago).

> Tulad ng maraming gawain sa programming, may ilang paraan upang makumpleto ito, ngunit ang magandang payo ay gawin ito sa pinakasimple at pinakamadaling paraan na kaya mo, lalo na kung mas madali itong maunawaan kapag binalikan mo ang code na ito sa hinaharap. Sa mga dataframe, mayroong isang komprehensibong API na madalas may paraan upang gawin ang gusto mo nang mahusay.

Ituring ang mga sumusunod na tanong bilang mga gawain sa coding at subukang sagutin ang mga ito nang hindi tinitingnan ang solusyon.

1. I-print ang *hugis* ng dataframe na kakaload mo lang (ang hugis ay ang bilang ng mga hilera at kolum).
2. Kalkulahin ang bilang ng mga paglitaw ng bawat reviewer nationality:
   1. Ilan ang natatanging halaga para sa kolum na `Reviewer_Nationality` at ano-ano ang mga ito?
   2. Aling reviewer nationality ang pinakakaraniwan sa dataset (i-print ang bansa at bilang ng mga review)?
   3. Ano ang susunod na nangungunang 10 pinakakaraniwang nationality, at ang kanilang bilang?
3. Ano ang pinakamaraming na-review na hotel para sa bawat isa sa nangungunang 10 reviewer nationality?
4. Ilan ang mga review bawat hotel (bilang ng mga review ng bawat hotel) sa dataset?
5. Habang mayroong kolum na `Average_Score` para sa bawat hotel sa dataset, maaari mo ring kalkulahin ang average score (kinukuha ang average ng lahat ng reviewer scores sa dataset para sa bawat hotel). Magdagdag ng bagong kolum sa iyong dataframe na may header na `Calc_Average_Score` na naglalaman ng kalkuladong average. 
6. Mayroon bang mga hotel na may parehong (niround-off sa 1 decimal place) `Average_Score` at `Calc_Average_Score`?
   1. Subukang magsulat ng Python function na tumatanggap ng isang Series (row) bilang argumento at ikinukumpara ang mga halaga, nagpi-print ng mensahe kapag hindi magkapareho ang mga halaga. Pagkatapos gamitin ang `.apply()` method upang iproseso ang bawat hilera gamit ang function.
7. Kalkulahin at i-print kung ilang hilera ang may halagang "No Negative" sa kolum na `Negative_Review`.
8. Kalkulahin at i-print kung ilang hilera ang may halagang "No Positive" sa kolum na `Positive_Review`.
9. Kalkulahin at i-print kung ilang hilera ang may halagang "No Positive" sa kolum na `Positive_Review` **at** halagang "No Negative" sa kolum na `Negative_Review`.

### Mga Sagot sa Code

1. I-print ang *hugis* ng dataframe na kakaload mo lang (ang hugis ay ang bilang ng mga hilera at kolum).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Kalkulahin ang bilang ng mga paglitaw ng bawat reviewer nationality:

   1. Ilan ang natatanging halaga para sa kolum na `Reviewer_Nationality` at ano-ano ang mga ito?
   2. Aling reviewer nationality ang pinakakaraniwan sa dataset (i-print ang bansa at bilang ng mga review)?

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

   3. Ano ang susunod na nangungunang 10 pinakakaraniwang nationality, at ang kanilang bilang?

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

3. Ano ang pinakamaraming na-review na hotel para sa bawat isa sa nangungunang 10 reviewer nationality?

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

4. Ilan ang mga review bawat hotel (bilang ng mga review ng bawat hotel) sa dataset?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Pangalan ng Hotel                 | Kabuuang Bilang ng Review | Kabuuang Nahanap na Review |
   | :-----------------------------------------------: | :-----------------------: | :-----------------------: |
   | Britannia International Hotel Canary Wharf        |          9086            |        4789              |
   | Park Plaza Westminster Bridge London              |         12158            |        4169              |
   | Copthorne Tara Hotel London Kensington            |          7105            |        3578              |
   |                    ...                            |           ...            |         ...              |
   | Mercure Paris Porte d Orleans                     |           110            |         10               |
   | Hotel Wagner                                      |           135            |         10               |
   | Hotel Gallitzinberg                               |           173            |          8               |

   Mapapansin mo na ang mga *nabilang sa dataset* na resulta ay hindi tumutugma sa halaga sa `Total_Number_of_Reviews`. Hindi malinaw kung ang halagang ito sa dataset ay kumakatawan sa kabuuang bilang ng mga review na mayroon ang hotel, ngunit hindi lahat ay na-scrape, o ibang kalkulasyon. Ang `Total_Number_of_Reviews` ay hindi ginagamit sa modelo dahil sa kawalang-kalinawan na ito.

5. Habang mayroong kolum na `Average_Score` para sa bawat hotel sa dataset, maaari mo ring kalkulahin ang average score (kinukuha ang average ng lahat ng reviewer scores sa dataset para sa bawat hotel). Magdagdag ng bagong kolum sa iyong dataframe na may header na `Calc_Average_Score` na naglalaman ng kalkuladong average. I-print ang mga kolum na `Hotel_Name`, `Average_Score`, at `Calc_Average_Score`.

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

   Maaari mo ring mapansin ang `Average_Score` na halaga at kung bakit ito minsan ay naiiba sa kalkuladong average score. Dahil hindi natin alam kung bakit ang ilang mga halaga ay tumutugma, ngunit ang iba ay may pagkakaiba, pinakaligtas sa kasong ito na gamitin ang mga review scores na mayroon tayo upang kalkulahin ang average natin. Gayunpaman, ang mga pagkakaiba ay karaniwang napakaliit, narito ang mga hotel na may pinakamalaking pagkakaiba mula sa dataset average at kalkuladong average:

   | Pagkakaiba sa Average_Score | Average_Score | Calc_Average_Score |                                  Pangalan ng Hotel |
   | :-------------------------: | :-----------: | :----------------: | -----------------------------------------------: |
   |           -0.8              |      7.7      |        8.5         |                  Best Western Hotel Astoria      |
   |           -0.7              |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery      |
   |           -0.7              |      7.5      |        8.2         |               Mercure Paris Porte d Orleans      |
   |           -0.7              |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel      |
   |           -0.5              |      7.0      |        7.5         |                         Hotel Royal Elys es      |
   |           ...               |      ...      |        ...         |                                         ...      |
   |           0.7               |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre      |
   |           0.8               |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur      |
   |           0.9               |      6.8      |        5.9         |                               Villa Eugenie      |
   |           0.9               |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux      |
   |           1.3               |      7.2      |        5.9         |                          Kube Hotel Ice Bar      |

   Sa iisang hotel lamang na may pagkakaiba ng score na higit sa 1, nangangahulugan ito na maaari nating balewalain ang pagkakaiba at gamitin ang kalkuladong average score.

6. Kalkulahin at i-print kung ilang hilera ang may halagang "No Negative" sa kolum na `Negative_Review`.

7. Kalkulahin at i-print kung ilang hilera ang may halagang "No Positive" sa kolum na `Positive_Review`.

8. Kalkulahin at i-print kung ilang hilera ang may halagang "No Positive" sa kolum na `Positive_Review` **at** halagang "No Negative" sa kolum na `Negative_Review`.

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

## Isa pang Paraan

Isa pang paraan upang bilangin ang mga item nang walang Lambdas, at gamitin ang sum upang bilangin ang mga hilera:

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

   Mapapansin mo na mayroong 127 hilera na may parehong "No Negative" at "No Positive" na mga halaga para sa mga kolum na `Negative_Review` at `Positive_Review` ayon sa pagkakabanggit. Nangangahulugan ito na ang reviewer ay nagbigay ng numerikal na score sa hotel, ngunit tumangging magsulat ng positibo o negatibong review. Sa kabutihang palad, ito ay maliit na bilang ng mga hilera (127 sa 515738, o 0.02%), kaya malamang na hindi nito maimpluwensyahan ang ating modelo o mga resulta sa anumang partikular na direksyon, ngunit maaaring hindi mo inaasahan na ang isang dataset ng mga review ay may mga hilera na walang review, kaya't sulit na galugarin ang data upang matuklasan ang mga hilera tulad nito.

Ngayon na na-explore mo na ang dataset, sa susunod na aralin ay ifi-filter mo ang data at magdadagdag ng sentiment analysis.

---
## 🚀Hamunin

Ipinapakita ng araling ito, tulad ng nakita natin sa mga nakaraang aralin, kung gaano kahalaga ang maunawaan ang iyong data at ang mga kakaibang aspeto nito bago magsagawa ng mga operasyon dito. Ang mga text-based na data, partikular, ay nangangailangan ng masusing pagsusuri. Halukayin ang iba't ibang text-heavy datasets at tingnan kung makakahanap ka ng mga lugar na maaaring magdulot ng bias o skewed sentiment sa isang modelo.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## Pagsusuri at Pag-aaral sa Sarili

Kunin ang [Learning Path na ito sa NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) upang matuklasan ang mga tool na maaaring subukan kapag gumagawa ng mga modelo na mabigat sa pagsasalita at teksto.

## Takdang-Aralin

[NLTK](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.