<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T18:25:22+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "tl"
}
-->
# Sentiment analysis gamit ang mga review ng hotel - pagproseso ng data

Sa seksyong ito, gagamitin mo ang mga teknik na natutunan sa mga nakaraang aralin upang magsagawa ng exploratory data analysis sa isang malaking dataset. Kapag nagkaroon ka ng mas malalim na pag-unawa sa kahalagahan ng iba't ibang column, matutunan mo:

- kung paano alisin ang mga hindi kinakailangang column
- kung paano kalkulahin ang bagong data batay sa mga umiiral na column
- kung paano i-save ang resulta ng dataset para magamit sa huling hamon

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Panimula

Sa ngayon, natutunan mo na ang text data ay lubos na naiiba sa numerical data. Kung ito ay text na isinulat o sinabi ng tao, maaari itong suriin upang matukoy ang mga pattern, dalas, damdamin, at kahulugan. Ang araling ito ay magdadala sa iyo sa isang tunay na dataset na may tunay na hamon: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** na may kasamang [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). Ang data ay kinuha mula sa Booking.com mula sa mga pampublikong mapagkukunan. Ang tagalikha ng dataset ay si Jiashen Liu.

### Paghahanda

Kakailanganin mo:

* Kakayahang magpatakbo ng .ipynb notebooks gamit ang Python 3
* pandas
* NLTK, [na dapat mong i-install nang lokal](https://www.nltk.org/install.html)
* Ang dataset na makukuha sa Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ang laki nito ay humigit-kumulang 230 MB kapag na-unzip. I-download ito sa root `/data` folder na nauugnay sa mga araling NLP na ito.

## Exploratory data analysis

Ang hamon na ito ay nagpapalagay na ikaw ay bumubuo ng isang hotel recommendation bot gamit ang sentiment analysis at mga score ng review ng mga bisita. Ang dataset na gagamitin mo ay naglalaman ng mga review ng 1493 iba't ibang hotel sa 6 na lungsod.

Gamit ang Python, isang dataset ng mga review ng hotel, at ang sentiment analysis ng NLTK, maaari mong tukuyin:

* Ano ang mga pinakaginagamit na salita at parirala sa mga review?
* Ang mga opisyal na *tags* ba na naglalarawan sa hotel ay may kaugnayan sa mga score ng review (halimbawa, mas negatibo ba ang mga review para sa isang partikular na hotel mula sa *Family with young children* kaysa sa *Solo traveller*, na maaaring magpahiwatig na mas angkop ito para sa *Solo travellers*?)
* Ang mga sentiment score ng NLTK ba ay "sumasang-ayon" sa numerical score ng reviewer ng hotel?

#### Dataset

Suriin natin ang dataset na na-download mo at na-save nang lokal. Buksan ang file sa isang editor tulad ng VS Code o kahit Excel.

Ang mga header sa dataset ay ang mga sumusunod:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Narito ang mga ito na naka-grupo sa paraang mas madaling suriin:  
##### Mga column ng hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Gamit ang *lat* at *lng*, maaari kang mag-plot ng mapa gamit ang Python na nagpapakita ng mga lokasyon ng hotel (maaaring naka-color code para sa mga negatibo at positibong review)
  * Ang Hotel_Address ay hindi mukhang kapaki-pakinabang sa atin, at malamang na papalitan natin ito ng bansa para sa mas madaling pag-uuri at paghahanap

**Mga column ng meta-review ng hotel**

* `Average_Score`
  * Ayon sa tagalikha ng dataset, ang column na ito ay ang *Average Score ng hotel, na kinakalkula batay sa pinakabagong komento sa nakaraang taon*. Mukhang hindi pangkaraniwang paraan ito ng pagkalkula ng score, ngunit ito ang data na nakuha kaya maaaring tanggapin natin ito sa ngayon.

  ✅ Batay sa iba pang column sa data na ito, makakaisip ka ba ng ibang paraan upang kalkulahin ang average score?

* `Total_Number_of_Reviews`
  * Ang kabuuang bilang ng mga review na natanggap ng hotel na ito - hindi malinaw (nang hindi nagsusulat ng code) kung ito ay tumutukoy sa mga review sa dataset.
* `Additional_Number_of_Scoring`
  * Nangangahulugan ito na may score na ibinigay ngunit walang positibo o negatibong review na isinulat ng reviewer.

**Mga column ng review**

- `Reviewer_Score`
  - Ito ay isang numerical value na may pinakamataas na 1 decimal place sa pagitan ng minimum at maximum na halaga na 2.5 at 10
  - Hindi ipinaliwanag kung bakit 2.5 ang pinakamababang posibleng score
- `Negative_Review`
  - Kung walang isinulat ang reviewer, ang field na ito ay magkakaroon ng "**No Negative**"
  - Tandaan na maaaring magsulat ang reviewer ng positibong review sa Negative review column (halimbawa, "walang masama sa hotel na ito")
- `Review_Total_Negative_Word_Counts`
  - Mas mataas na bilang ng negatibong salita ang nagpapahiwatig ng mas mababang score (nang hindi sinusuri ang damdamin)
- `Positive_Review`
  - Kung walang isinulat ang reviewer, ang field na ito ay magkakaroon ng "**No Positive**"
  - Tandaan na maaaring magsulat ang reviewer ng negatibong review sa Positive review column (halimbawa, "walang maganda sa hotel na ito")
- `Review_Total_Positive_Word_Counts`
  - Mas mataas na bilang ng positibong salita ang nagpapahiwatig ng mas mataas na score (nang hindi sinusuri ang damdamin)
- `Review_Date` at `days_since_review`
  - Maaaring mag-apply ng freshness o staleness measure sa isang review (ang mas lumang review ay maaaring hindi kasing-accurate ng mas bago dahil nagbago ang pamamahala ng hotel, may mga renovation, o may idinagdag na pool, atbp.)
- `Tags`
  - Ito ay mga maikling descriptor na maaaring piliin ng reviewer upang ilarawan ang uri ng bisita sila (halimbawa, solo o pamilya), ang uri ng kuwarto na kanilang tinirhan, ang haba ng pananatili, at kung paano naisumite ang review.
  - Sa kasamaang-palad, ang paggamit ng mga tag na ito ay may problema, tingnan ang seksyon sa ibaba na naglalarawan ng kanilang kahalagahan.

**Mga column ng reviewer**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Maaaring maging salik ito sa isang recommendation model, halimbawa, kung matutukoy mo na ang mas prolific na reviewer na may daan-daang review ay mas malamang na maging negatibo kaysa positibo. Gayunpaman, ang reviewer ng anumang partikular na review ay hindi nakilala gamit ang isang natatanging code, kaya't hindi ito maaaring maiugnay sa isang set ng mga review. Mayroong 30 reviewer na may 100 o higit pang review, ngunit mahirap makita kung paano ito makakatulong sa recommendation model.
- `Reviewer_Nationality`
  - Ang ilang tao ay maaaring mag-isip na ang ilang nasyonalidad ay mas malamang na magbigay ng positibo o negatibong review dahil sa isang pambansang hilig. Mag-ingat sa pagbuo ng ganitong anecdotal na pananaw sa iyong mga modelo. Ito ay mga pambansa (at kung minsan ay lahi) na stereotype, at ang bawat reviewer ay isang indibidwal na nagsulat ng review batay sa kanilang karanasan. Maaaring na-filter ito sa maraming lens tulad ng kanilang mga nakaraang pananatili sa hotel, ang distansya ng kanilang paglalakbay, at ang kanilang personal na ugali. Mahirap patunayan na ang kanilang nasyonalidad ang dahilan ng score ng review.

##### Mga Halimbawa

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Ito ay kasalukuyang hindi isang hotel kundi isang construction site Ako ay ginising mula sa maagang umaga at buong araw ng hindi katanggap-tanggap na ingay ng konstruksyon habang nagpapahinga pagkatapos ng mahabang biyahe at nagtatrabaho sa kuwarto Ang mga tao ay nagtatrabaho buong araw gamit ang jackhammers sa mga katabing kuwarto Humiling ako ng palit ng kuwarto ngunit walang tahimik na kuwarto na magagamit Upang gawing mas masama, ako ay siningil ng sobra Nag-check out ako sa gabi dahil kailangan kong umalis para sa maagang flight at nakatanggap ng tamang bill Isang araw pagkatapos, ang hotel ay gumawa ng karagdagang singil nang walang pahintulot na mas mataas sa nakabook na presyo Napakasamang lugar Huwag parusahan ang sarili mo sa pag-book dito | Wala  Napakasamang lugar Lumayo | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Makikita mo, ang bisitang ito ay hindi nagkaroon ng masayang pananatili sa hotel na ito. Ang hotel ay may magandang average score na 7.8 at 1945 review, ngunit binigyan ito ng reviewer ng 2.5 at nagsulat ng 115 salita tungkol sa kung gaano ka-negatibo ang kanilang pananatili. Kung wala silang isinulat sa Positive_Review column, maaari mong ipalagay na walang positibo, ngunit isinulat nila ang 7 salita ng babala. Kung bibilangin lang natin ang mga salita sa halip na ang kahulugan o damdamin ng mga salita, maaaring magkaroon tayo ng maling pananaw sa intensyon ng reviewer. Nakakagulat, ang kanilang score na 2.5 ay nakakalito, dahil kung napakasama ng pananatili sa hotel, bakit pa magbigay ng anumang puntos? Sa pagsisiyasat sa dataset nang malapitan, makikita mo na ang pinakamababang posibleng score ay 2.5, hindi 0. Ang pinakamataas na posibleng score ay 10.

##### Tags

Tulad ng nabanggit sa itaas, sa unang tingin, ang ideya na gamitin ang `Tags` upang i-categorize ang data ay may katuturan. Sa kasamaang-palad, ang mga tag na ito ay hindi standardized, na nangangahulugan na sa isang hotel, ang mga opsyon ay maaaring *Single room*, *Twin room*, at *Double room*, ngunit sa susunod na hotel, ang mga ito ay *Deluxe Single Room*, *Classic Queen Room*, at *Executive King Room*. Maaaring pareho ang mga ito, ngunit napakaraming variation na ang pagpipilian ay:

1. Subukang baguhin ang lahat ng termino sa isang pamantayan, na napakahirap, dahil hindi malinaw kung ano ang conversion path sa bawat kaso (halimbawa, *Classic single room* ay tumutugma sa *Single room* ngunit *Superior Queen Room with Courtyard Garden or City View* ay mas mahirap i-map)

1. Maaari tayong gumamit ng NLP approach at sukatin ang dalas ng ilang termino tulad ng *Solo*, *Business Traveller*, o *Family with young kids* habang ito ay naaangkop sa bawat hotel, at isama ito sa rekomendasyon.

Ang mga tag ay karaniwang (ngunit hindi palaging) isang field na naglalaman ng listahan ng 5 hanggang 6 na comma-separated values na tumutugma sa *Uri ng biyahe*, *Uri ng bisita*, *Uri ng kuwarto*, *Bilang ng gabi*, at *Uri ng device kung saan naisumite ang review*. Gayunpaman, dahil ang ilang reviewer ay hindi pinupunan ang bawat field (maaaring iwanan nila ang isa na blangko), ang mga value ay hindi palaging nasa parehong pagkakasunod-sunod.

Halimbawa, kunin ang *Uri ng grupo*. Mayroong 1025 natatanging posibilidad sa field na ito sa `Tags` column, at sa kasamaang-palad, ilan lamang sa mga ito ang tumutukoy sa isang grupo (ang ilan ay uri ng kuwarto, atbp.). Kung i-filter mo lamang ang mga nagbabanggit ng pamilya, ang mga resulta ay naglalaman ng maraming *Family room* na uri ng resulta. Kung isasama mo ang terminong *with*, i.e. bilangin ang *Family with* values, mas maganda ang mga resulta, na may higit sa 80,000 sa 515,000 resulta na naglalaman ng pariralang "Family with young children" o "Family with older children".

Ibig sabihin, ang tags column ay hindi ganap na walang silbi sa atin, ngunit kakailanganin ng kaunting trabaho upang maging kapaki-pakinabang ito.

##### Average hotel score

Mayroong ilang mga kakaibang bagay o hindi pagkakapare-pareho sa dataset na hindi ko maipaliwanag, ngunit inilalarawan dito upang malaman mo ang mga ito kapag bumubuo ng iyong mga modelo. Kung ma-figure out mo ito, mangyaring ipaalam sa amin sa discussion section!

Ang dataset ay may mga sumusunod na column na nauugnay sa average score at bilang ng mga review:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Ang hotel na may pinakamaraming review sa dataset na ito ay *Britannia International Hotel Canary Wharf* na may 4789 review mula sa 515,000. Ngunit kung titingnan natin ang `Total_Number_of_Reviews` value para sa hotel na ito, ito ay 9086. Maaari mong ipalagay na may mas maraming score na walang review, kaya marahil dapat nating idagdag ang `Additional_Number_of_Scoring` column value. Ang value na iyon ay 2682, at kapag idinagdag sa 4789, makakakuha tayo ng 7,471 na kulang pa rin ng 1615 sa `Total_Number_of_Reviews`.

Kung kukunin mo ang `Average_Score` column, maaari mong ipalagay na ito ang average ng mga review sa dataset, ngunit ang paglalarawan mula sa Kaggle ay "*Average Score ng hotel, na kinakalkula batay sa pinakabagong komento sa nakaraang taon*". Mukhang hindi ito kapaki-pakinabang, ngunit maaari nating kalkulahin ang sarili nating average batay sa mga review score sa dataset. Gamit ang parehong hotel bilang halimbawa, ang average hotel score ay ibinigay bilang 7.1 ngunit ang calculated score (average reviewer score *sa* dataset) ay 6.8. Malapit ito, ngunit hindi pareho ang value, at maaari lamang nating ipalagay na ang mga score na ibinigay sa `Additional_Number_of_Scoring` reviews ay nagtaas ng average sa 7.1. Sa kasamaang-palad, nang walang paraan upang subukan o patunayan ang assertion na iyon, mahirap gamitin o pagkatiwalaan ang `Average_Score`, `Additional_Number_of_Scoring`, at `Total_Number_of_Reviews` kapag ang mga ito ay batay sa, o tumutukoy sa, data na wala tayo.

Upang gawing mas kumplikado ang mga bagay, ang hotel na may pangalawang pinakamataas na bilang ng mga review ay may calculated average score na 8.12 at ang dataset `Average_Score` ay 8.1. Coincidence ba ang tamang score na ito o discrepancy ang unang hotel?

Sa posibilidad na ang mga hotel na ito ay maaaring isang outlier, at na marahil ang karamihan sa mga value ay tumutugma (ngunit ang ilan ay hindi sa ilang kadahilanan), magsusulat tayo ng maikling programa sa susunod upang suriin ang mga value sa dataset at tukuyin ang tamang paggamit (o hindi paggamit) ng mga value.
> 🚨 Isang paalala ng pag-iingat  
>  
> Kapag nagtatrabaho sa dataset na ito, magsusulat ka ng code na nagkakalkula ng isang bagay mula sa teksto nang hindi kinakailangang basahin o suriin ang teksto mismo. Ito ang diwa ng NLP, ang pag-unawa sa kahulugan o damdamin nang hindi kinakailangang gawin ito ng tao. Gayunpaman, posible na mabasa mo ang ilan sa mga negatibong review. Hinihikayat kita na huwag gawin ito, dahil hindi mo naman kailangan. Ang ilan sa mga ito ay walang saysay, o hindi mahalagang negatibong review ng hotel, tulad ng "Hindi maganda ang panahon," isang bagay na wala sa kontrol ng hotel, o kahit sino. Ngunit may madilim na bahagi rin sa ilang review. Minsan ang mga negatibong review ay rasista, seksista, o may diskriminasyon sa edad. Ito ay nakakalungkot ngunit inaasahan sa isang dataset na kinuha mula sa isang pampublikong website. Ang ilang mga reviewer ay nag-iiwan ng mga review na maaaring makita mong hindi kaaya-aya, hindi komportable, o nakakagambala. Mas mabuting hayaan ang code na sukatin ang damdamin kaysa basahin mo ang mga ito at maapektuhan. Gayunpaman, ito ay isang maliit na bahagi lamang na nagsusulat ng ganitong mga bagay, ngunit umiiral pa rin sila.
## Ehersisyo - Paggalugad ng Data
### I-load ang data

Tama na ang pagsusuri ng data nang biswal, oras na para magsulat ng code at makakuha ng mga sagot! Ang seksyong ito ay gumagamit ng pandas library. Ang iyong unang gawain ay tiyaking ma-load at mabasa mo ang CSV data. Ang pandas library ay may mabilis na CSV loader, at ang resulta ay inilalagay sa isang dataframe, tulad ng sa mga nakaraang aralin. Ang CSV na ating ilo-load ay may higit sa kalahating milyong rows, ngunit may 17 columns lamang. Ang pandas ay nagbibigay ng maraming makapangyarihang paraan upang makipag-ugnayan sa isang dataframe, kabilang ang kakayahang magsagawa ng mga operasyon sa bawat row.

Mula dito sa araling ito, magkakaroon ng mga code snippets, ilang paliwanag tungkol sa code, at talakayan tungkol sa kahulugan ng mga resulta. Gamitin ang kasamang _notebook.ipynb_ para sa iyong code.

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

Ngayon na na-load na ang data, maaari na tayong magsagawa ng ilang operasyon dito. Panatilihin ang code na ito sa itaas ng iyong programa para sa susunod na bahagi.

## Galugarin ang data

Sa kasong ito, ang data ay *malinis* na, ibig sabihin handa na itong gamitin, at walang mga karakter sa ibang wika na maaaring magdulot ng problema sa mga algorithm na inaasahan lamang ang mga karakter sa Ingles.

✅ Maaaring kailanganin mong magtrabaho sa data na nangangailangan ng paunang pagproseso upang ma-format ito bago mag-apply ng mga teknik sa NLP, ngunit hindi sa pagkakataong ito. Kung kailangan mo, paano mo haharapin ang mga karakter na hindi Ingles?

Maglaan ng sandali upang tiyakin na kapag na-load na ang data, magagawa mo itong galugarin gamit ang code. Napakadaling mag-focus sa mga column na `Negative_Review` at `Positive_Review`. Ang mga ito ay puno ng natural na teksto para sa iyong NLP algorithms na iproseso. Ngunit sandali! Bago ka sumabak sa NLP at sentiment analysis, sundin mo ang code sa ibaba upang matiyak kung ang mga halaga sa dataset ay tumutugma sa mga halagang kinakalkula mo gamit ang pandas.

## Mga operasyon sa Dataframe

Ang unang gawain sa araling ito ay suriin kung tama ang mga sumusunod na assertions sa pamamagitan ng pagsulat ng code na sinusuri ang dataframe (nang hindi ito binabago).

> Tulad ng maraming gawain sa programming, may iba't ibang paraan upang makumpleto ito, ngunit ang magandang payo ay gawin ito sa pinakasimple at pinakamadaling paraan na maaari mong gawin, lalo na kung mas madali itong maunawaan kapag binalikan mo ang code na ito sa hinaharap. Sa mga dataframe, mayroong komprehensibong API na madalas may paraan upang gawin ang gusto mo nang mahusay.

Ituring ang mga sumusunod na tanong bilang mga coding task at subukang sagutin ang mga ito nang hindi tinitingnan ang solusyon.

1. I-print ang *shape* ng dataframe na kakaload mo lang (ang shape ay ang bilang ng rows at columns).
2. Kalkulahin ang frequency count para sa mga nationality ng reviewer:
   1. Ilan ang mga natatanging halaga para sa column na `Reviewer_Nationality` at ano ang mga ito?
   2. Anong nationality ng reviewer ang pinakakaraniwan sa dataset (i-print ang bansa at bilang ng mga review)?
   3. Ano ang susunod na nangungunang 10 pinakakaraniwang nationality, at ang kanilang frequency count?
3. Ano ang pinakamaraming na-review na hotel para sa bawat isa sa nangungunang 10 nationality ng reviewer?
4. Ilan ang mga review bawat hotel (frequency count ng hotel) sa dataset?
5. Bagama't may column na `Average_Score` para sa bawat hotel sa dataset, maaari mo ring kalkulahin ang average score (kinukuha ang average ng lahat ng reviewer scores sa dataset para sa bawat hotel). Magdagdag ng bagong column sa iyong dataframe na may header na `Calc_Average_Score` na naglalaman ng calculated average. 
6. Mayroon bang mga hotel na may parehong (rounded to 1 decimal place) `Average_Score` at `Calc_Average_Score`?
   1. Subukang magsulat ng Python function na tumatanggap ng Series (row) bilang argumento at ikinukumpara ang mga halaga, nagpi-print ng mensahe kapag ang mga halaga ay hindi magkapareho. Pagkatapos gamitin ang `.apply()` method upang iproseso ang bawat row gamit ang function.
7. Kalkulahin at i-print kung ilang rows ang may column na `Negative_Review` na may halagang "No Negative".
8. Kalkulahin at i-print kung ilang rows ang may column na `Positive_Review` na may halagang "No Positive".
9. Kalkulahin at i-print kung ilang rows ang may column na `Positive_Review` na may halagang "No Positive" **at** `Negative_Review` na may halagang "No Negative".

### Mga sagot sa code

1. I-print ang *shape* ng dataframe na kakaload mo lang (ang shape ay ang bilang ng rows at columns).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Kalkulahin ang frequency count para sa mga nationality ng reviewer:

   1. Ilan ang mga natatanging halaga para sa column na `Reviewer_Nationality` at ano ang mga ito?
   2. Anong nationality ng reviewer ang pinakakaraniwan sa dataset (i-print ang bansa at bilang ng mga review)?

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

   3. Ano ang susunod na nangungunang 10 pinakakaraniwang nationality, at ang kanilang frequency count?

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

3. Ano ang pinakamaraming na-review na hotel para sa bawat isa sa nangungunang 10 nationality ng reviewer?

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

4. Ilan ang mga review bawat hotel (frequency count ng hotel) sa dataset?

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
   
   Mapapansin mo na ang *counted in the dataset* na resulta ay hindi tumutugma sa halaga sa `Total_Number_of_Reviews`. Hindi malinaw kung ang halagang ito sa dataset ay kumakatawan sa kabuuang bilang ng mga review na mayroon ang hotel, ngunit hindi lahat ay na-scrape, o iba pang kalkulasyon. Ang `Total_Number_of_Reviews` ay hindi ginagamit sa model dahil sa kawalang-linaw na ito.

5. Bagama't may column na `Average_Score` para sa bawat hotel sa dataset, maaari mo ring kalkulahin ang average score (kinukuha ang average ng lahat ng reviewer scores sa dataset para sa bawat hotel). Magdagdag ng bagong column sa iyong dataframe na may header na `Calc_Average_Score` na naglalaman ng calculated average. I-print ang mga column na `Hotel_Name`, `Average_Score`, at `Calc_Average_Score`.

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

   Maaaring magtaka ka rin tungkol sa halaga ng `Average_Score` at kung bakit ito minsan ay naiiba sa calculated average score. Dahil hindi natin alam kung bakit ang ilang mga halaga ay tumutugma, ngunit ang iba ay may pagkakaiba, pinakaligtas sa kasong ito na gamitin ang mga review scores na mayroon tayo upang kalkulahin ang average mismo. Gayunpaman, ang mga pagkakaiba ay karaniwang napakaliit, narito ang mga hotel na may pinakamalaking deviation mula sa dataset average at sa calculated average:

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

   Sa iisang hotel lamang na may pagkakaiba ng score na higit sa 1, nangangahulugan ito na maaari nating balewalain ang pagkakaiba at gamitin ang calculated average score.

6. Kalkulahin at i-print kung ilang rows ang may column na `Negative_Review` na may halagang "No Negative".

7. Kalkulahin at i-print kung ilang rows ang may column na `Positive_Review` na may halagang "No Positive".

8. Kalkulahin at i-print kung ilang rows ang may column na `Positive_Review` na may halagang "No Positive" **at** `Negative_Review` na may halagang "No Negative".

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

## Isa pang paraan

Isa pang paraan upang magbilang ng mga item nang walang Lambdas, at gamitin ang sum upang bilangin ang mga rows:

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

   Mapapansin mo na mayroong 127 rows na may parehong "No Negative" at "No Positive" na mga halaga para sa mga column na `Negative_Review` at `Positive_Review` ayon sa pagkakabanggit. Nangangahulugan ito na ang reviewer ay nagbigay ng numerical score sa hotel, ngunit tumangging magsulat ng positibo o negatibong review. Sa kabutihang palad, ito ay maliit na bilang ng rows (127 sa 515738, o 0.02%), kaya malamang na hindi nito maapektuhan ang ating model o mga resulta sa anumang partikular na direksyon, ngunit maaaring hindi mo inaasahan na ang isang dataset ng mga review ay may mga rows na walang review, kaya't sulit na galugarin ang data upang matuklasan ang mga ganitong rows.

Ngayon na na-explore mo na ang dataset, sa susunod na aralin ay ifi-filter mo ang data at magdadagdag ng sentiment analysis.

---
## 🚀Hamunin

Ipinapakita ng araling ito, tulad ng nakita natin sa mga nakaraang aralin, kung gaano kahalaga ang maunawaan ang iyong data at ang mga kahinaan nito bago magsagawa ng mga operasyon dito. Ang data na batay sa teksto, partikular, ay nangangailangan ng maingat na pagsusuri. Maghukay sa iba't ibang text-heavy datasets at tingnan kung makakahanap ka ng mga lugar na maaaring magpakilala ng bias o skewed sentiment sa isang model.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pag-aaral sa Sarili

Kunin ang [Learning Path na ito sa NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) upang matuklasan ang mga tool na maaaring subukan kapag gumagawa ng mga model na batay sa speech at text.

## Takdang Aralin

[NLTK](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.