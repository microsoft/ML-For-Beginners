# Uchambuzi wa hisia na maoni ya hoteli - kuchakata data

Katika sehemu hii utatumia mbinu zilizojifunza katika masomo yaliyopita kufanya uchambuzi wa data wa seti kubwa ya data. Mara baada ya kuelewa vizuri umuhimu wa safu mbalimbali, utajifunza:

- jinsi ya kuondoa safu zisizo za lazima
- jinsi ya kuhesabu data mpya kulingana na safu zilizopo
- jinsi ya kuhifadhi seti ya data iliyotokana kwa matumizi katika changamoto ya mwisho

## [Maswali kabla ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Utangulizi

Hadi sasa umejifunza kuhusu jinsi data ya maandishi inavyotofautiana na aina za data za nambari. Ikiwa ni maandishi yaliyoandikwa au kusemwa na binadamu, yanaweza kuchambuliwa ili kupata mifumo na marudio, hisia na maana. Somo hili linakupeleka kwenye seti ya data halisi na changamoto halisi: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** na inajumuisha [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). Iliandikwa kutoka vyanzo vya umma vya Booking.com. Muundaji wa seti ya data ni Jiashen Liu.

### Maandalizi

Utahitaji:

* Uwezo wa kuendesha daftari za .ipynb kwa kutumia Python 3
* pandas
* NLTK, [ambayo unapaswa kusakinisha ndani ya kompyuta yako](https://www.nltk.org/install.html)
* Seti ya data inayopatikana kwenye Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ina takriban 230 MB baada ya kufunguliwa. Pakua kwenye folda ya mzizi `/data` inayohusishwa na masomo haya ya NLP.

## Uchambuzi wa data

Changamoto hii inadhani kuwa unajenga roboti ya mapendekezo ya hoteli kwa kutumia uchambuzi wa hisia na alama za maoni ya wageni. Seti ya data unayotumia inajumuisha maoni ya hoteli 1493 tofauti katika miji 6.

Kwa kutumia Python, seti ya data ya maoni ya hoteli, na uchambuzi wa hisia za NLTK unaweza kugundua:

* Maneno na misemo inayotumika mara kwa mara katika maoni ni yapi?
* Je, *lebo* rasmi zinazofafanua hoteli zinahusiana na alama za maoni (mfano, je, maoni hasi zaidi kwa hoteli fulani ni kwa *Familia yenye watoto wadogo* kuliko *Msafiri wa peke yake*, labda ikionyesha ni bora kwa *Wasafiri wa peke yao*?)
* Je, alama za hisia za NLTK 'zinakubaliana' na alama za nambari za mtoa maoni wa hoteli?

#### Seti ya data

Hebu tuangalie seti ya data uliyopakua na kuhifadhi ndani ya kompyuta yako. Fungua faili katika mhariri kama VS Code au hata Excel.

Vichwa vya habari katika seti ya data ni kama ifuatavyo:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hapa vimepangwa kwa njia ambayo inaweza kuwa rahisi kuchunguza:
##### Safu za hoteli

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Kwa kutumia *lat* na *lng* unaweza kuchora ramani kwa kutumia Python inayoonyesha maeneo ya hoteli (labda kwa rangi tofauti kwa maoni hasi na chanya)
  * Hotel_Address sio ya wazi kwa matumizi yetu, na labda tutaibadilisha na nchi kwa urahisi wa kupanga na kutafuta

**Safu za Maoni ya Meta ya Hoteli**

* `Average_Score`
  * Kulingana na muundaji wa seti ya data, safu hii ni *Alama ya wastani ya hoteli, iliyohesabiwa kulingana na maoni ya hivi karibuni katika mwaka uliopita*. Hii inaonekana kama njia isiyo ya kawaida ya kuhesabu alama, lakini ni data iliyopatikana hivyo tunaweza kuichukua kwa thamani ya uso kwa sasa.
  
  ✅ Kulingana na safu nyingine katika data hii, unaweza kufikiria njia nyingine ya kuhesabu alama ya wastani?

* `Total_Number_of_Reviews`
  * Jumla ya maoni ambayo hoteli hii imepokea - haijulikani (bila kuandika baadhi ya misimbo) ikiwa hii inahusu maoni katika seti ya data.
* `Additional_Number_of_Scoring`
  * Hii inamaanisha alama ya maoni ilitolewa lakini hakuna maoni chanya au hasi yaliyoandikwa na mtoa maoni

**Safu za maoni**

- `Reviewer_Score`
  - Hii ni thamani ya nambari yenye nafasi 1 ya desimali kati ya thamani za chini na za juu 2.5 na 10
  - Haijaelezewa kwa nini 2.5 ndio alama ya chini kabisa inayowezekana
- `Negative_Review`
  - Ikiwa mtoa maoni hakuandika chochote, sehemu hii itakuwa na "**Hakuna Hasi**"
  - Kumbuka kuwa mtoa maoni anaweza kuandika maoni chanya katika safu ya maoni hasi (mfano, "hakuna kitu kibaya kuhusu hoteli hii")
- `Review_Total_Negative_Word_Counts`
  - Hesabu ya maneno hasi zaidi inaonyesha alama ya chini zaidi (bila kuangalia hisia)
- `Positive_Review`
  - Ikiwa mtoa maoni hakuandika chochote, sehemu hii itakuwa na "**Hakuna Chanya**"
  - Kumbuka kuwa mtoa maoni anaweza kuandika maoni hasi katika safu ya maoni chanya (mfano, "hakuna kitu kizuri kabisa kuhusu hoteli hii")
- `Review_Total_Positive_Word_Counts`
  - Hesabu ya maneno chanya zaidi inaonyesha alama ya juu zaidi (bila kuangalia hisia)
- `Review_Date` na `days_since_review`
  - Kipimo cha upya au uchakavu kinaweza kutumika kwa maoni (maoni ya zamani yanaweza kuwa sio sahihi kama ya karibuni kwa sababu usimamizi wa hoteli umebadilika, au ukarabati umefanyika, au bwawa limeongezwa n.k.)
- `Tags`
  - Hizi ni maelezo mafupi ambayo mtoa maoni anaweza kuchagua kufafanua aina ya mgeni waliokuwa (mfano, msafiri wa peke yake au familia), aina ya chumba walichokuwa nacho, muda wa kukaa na jinsi maoni yalivyowasilishwa.
  - Kwa bahati mbaya, kutumia lebo hizi ni shida, angalia sehemu hapa chini inayojadili umuhimu wake

**Safu za mtoa maoni**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Hii inaweza kuwa sababu katika mfano wa mapendekezo, kwa mfano, ikiwa unaweza kubaini kuwa waandishi wa maoni wenye maoni mengi zaidi ya mia moja walikuwa na uwezekano mkubwa wa kuwa na maoni hasi badala ya chanya. Hata hivyo, mtoa maoni wa maoni yoyote maalum hajatajwa na msimbo wa kipekee, na hivyo hawezi kuunganishwa na seti ya maoni. Kuna waandishi wa maoni 30 wenye maoni 100 au zaidi, lakini ni vigumu kuona jinsi hii inaweza kusaidia mfano wa mapendekezo.
- `Reviewer_Nationality`
  - Watu wengine wanaweza kufikiria kwamba utaifa fulani una uwezekano mkubwa wa kutoa maoni chanya au hasi kwa sababu ya mwelekeo wa kitaifa. Kuwa makini kujenga maoni kama haya ya kidokezo katika mifano yako. Hizi ni mifano ya kitaifa (na wakati mwingine ya kikabila), na kila mtoa maoni alikuwa mtu binafsi ambaye aliandika maoni kulingana na uzoefu wao. Inawezekana kuwa yalichujwa kupitia lensi nyingi kama vile kukaa kwao kwa hoteli za awali, umbali waliotembea, na tabia yao binafsi. Kufikiria kuwa utaifa wao ulikuwa sababu ya alama ya maoni ni vigumu kuthibitisha.

##### Mifano

| Alama ya Wastani | Jumla ya Maoni   | Alama ya Mtoa Maoni | Maoni Hasi <br />                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Maoni Chanya                 | Lebo                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Hii kwa sasa sio hoteli lakini ni eneo la ujenzi niliteswa kutoka asubuhi na mapema na siku nzima na kelele za ujenzi zisizokubalika wakati wa kupumzika baada ya safari ndefu na kufanya kazi chumbani Watu walikuwa wakifanya kazi siku nzima i e na nyundo za umeme katika vyumba vya jirani niliomba kubadilisha chumba lakini hakuna chumba kimya kilichopatikana Ili kufanya mambo kuwa mabaya zaidi nilitozwa zaidi niliondoka jioni kwa kuwa nilihitaji kuondoka mapema sana kwa ndege na kupokea bili inayofaa Siku moja baadaye hoteli ilifanya malipo mengine bila idhini yangu zaidi ya bei iliyowekwa Ni mahali pabaya Usijiteshe kwa kujibook hapa | Hakuna  Mahali pabaya Kaa mbali | Safari ya kikazi                                Wenzi Chumba cha kawaida cha mara mbili  Kaa usiku 2 |

Kama unavyoona, mgeni huyu hakuwa na kukaa vizuri katika hoteli hii. Hoteli ina alama nzuri ya wastani ya 7.8 na maoni 1945, lakini mtoa maoni huyu alitoa alama 2.5 na kuandika maneno 115 kuhusu jinsi kukaa kwao kulivyokuwa hasi. Ikiwa hawakuandika chochote kabisa katika safu ya Positive_Review, unaweza kudhani hakuna kitu chanya, lakini walakini waliandika maneno 7 ya onyo. Ikiwa tungehisabu maneno badala ya maana, au hisia za maneno, tunaweza kuwa na mtazamo potofu wa nia ya mtoa maoni. Kwa kushangaza, alama yao ya 2.5 inachanganya, kwa sababu ikiwa kukaa kwao katika hoteli hiyo kulikuwa kibaya sana, kwa nini wape alama yoyote kabisa? Kuchunguza seti ya data kwa karibu, utaona kuwa alama ya chini kabisa inayowezekana ni 2.5, sio 0. Alama ya juu kabisa inayowezekana ni 10.

##### Lebo

Kama ilivyotajwa hapo juu, kwa mtazamo wa kwanza, wazo la kutumia `Tags` kuainisha data linaonekana kuwa na maana. Kwa bahati mbaya lebo hizi hazijasanifishwa, ambayo inamaanisha kuwa katika hoteli fulani, chaguzi zinaweza kuwa *Chumba kimoja*, *Chumba pacha*, na *Chumba mara mbili*, lakini katika hoteli nyingine, ni *Chumba cha Deluxe Single*, *Chumba cha Classic Queen*, na *Chumba cha Executive King*. Hizi zinaweza kuwa vitu sawa, lakini kuna tofauti nyingi sana kwamba chaguo linakuwa:

1. Jaribu kubadilisha maneno yote kuwa kiwango kimoja, ambayo ni ngumu sana, kwa sababu haijulikani njia ya kubadilisha itakuwa nini katika kila kesi (mfano, *Chumba kimoja cha classic* kinaenda kwa *Chumba kimoja* lakini *Chumba cha Superior Queen na Mtazamo wa Bustani ya Ua au Jiji* ni ngumu zaidi kubadilisha)

1. Tunaweza kuchukua njia ya NLP na kupima marudio ya maneno fulani kama *Solo*, *Msafiri wa Biashara*, au *Familia yenye watoto wadogo* jinsi zinavyotumika kwa kila hoteli, na kuingiza hiyo katika mapendekezo

Lebo kawaida (lakini si mara zote) ni uwanja mmoja unaojumuisha orodha ya maadili 5 hadi 6 yaliyotenganishwa kwa koma yanayolingana na *Aina ya safari*, *Aina ya wageni*, *Aina ya chumba*, *Idadi ya usiku*, na *Aina ya kifaa maoni yalivyowasilishwa*. Hata hivyo, kwa sababu waandishi wa maoni wengine hawajazi kila uwanja (wanaweza kuacha moja wazi), maadili hayako katika mpangilio sawa kila wakati.

Kwa mfano, chukua *Aina ya kikundi*. Kuna uwezekano wa kipekee 1025 katika uwanja huu katika safu ya `Tags`, na kwa bahati mbaya ni baadhi tu yao yanayohusu kikundi (baadhi ni aina ya chumba n.k.). Ikiwa unachuja zile zinazotaja familia pekee, matokeo yana maadili mengi ya aina ya *Chumba cha familia*. Ikiwa unajumuisha neno *na*, yaani hesabu maadili ya *Familia na*, matokeo ni bora, na zaidi ya 80,000 ya matokeo 515,000 yanajumuisha maneno "Familia na watoto wadogo" au "Familia na watoto wakubwa".

Hii inamaanisha safu ya lebo sio bure kabisa kwetu, lakini itachukua kazi kuifanya iwe na maana.

##### Alama ya wastani ya hoteli

Kuna mambo kadhaa yasiyoeleweka au tofauti na seti ya data ambayo siwezi kuelewa, lakini yanaonyeshwa hapa ili uweze kujua wakati wa kujenga mifano yako. Ikiwa utaelewa, tafadhali tujulishe katika sehemu ya majadiliano!

Seti ya data ina safu zifuatazo zinazohusiana na alama ya wastani na idadi ya maoni:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hoteli moja yenye maoni mengi zaidi katika seti hii ya data ni *Britannia International Hotel Canary Wharf* yenye maoni 4789 kati ya 515,000. Lakini ikiwa tutaangalia thamani ya `Total_Number_of_Reviews` kwa hoteli hii, ni 9086. Unaweza kudhani kuwa kuna alama nyingi zaidi bila maoni, hivyo labda tunapaswa kuongeza thamani ya safu ya `Additional_Number_of_Scoring`. Thamani hiyo ni 2682, na kuiongeza kwa 4789 inatufikisha 7,471 ambayo bado ni 1615 pungufu ya `Total_Number_of_Reviews`.

Ikiwa unachukua safu za `Average_Score`, unaweza kudhani ni wastani wa maoni katika seti ya data, lakini maelezo kutoka Kaggle ni "*Alama ya wastani ya hoteli, iliyohesabiwa kulingana na maoni ya hivi karibuni katika mwaka uliopita*". Hiyo haionekani kuwa na maana, lakini tunaweza kuhesabu wastani wetu kulingana na alama za maoni katika seti ya data. Kwa kutumia hoteli hiyo hiyo kama mfano, alama ya wastani ya hoteli inatolewa kama 7.1 lakini alama iliyohesabiwa (wastani wa alama za mtoa maoni *katika* seti ya data) ni 6.8. Hii ni karibu, lakini sio thamani sawa, na tunaweza kudhani tu kuwa alama zilizotolewa katika maoni ya `Additional_Number_of_Scoring` ziliongeza wastani hadi 7.1. Kwa bahati mbaya bila njia ya kujaribu au kuthibitisha dhana hiyo, ni vigumu kutumia au kuamini `Average_Score`, `Additional_Number_of_Scoring` na `Total_Number_of_Reviews` wakati zinategemea, au zinarejelea, data ambayo hatuna.

Kuchanganya mambo zaidi, hoteli yenye idadi ya pili ya juu zaidi ya maoni ina alama ya wastani iliyohesabiwa ya 8.12 na seti ya data ya `Average_Score` ni 8.1. Je, alama sahihi ni bahati nasibu au hoteli ya kwanza ni tofauti?

Kwa uwezekano kwamba hoteli hizi zinaweza kuwa ni kipekee, na kwamba labda maadili mengi yanafikia (lakini baadhi hayafanyi kwa sababu fulani) tutaandika programu fupi ijayo ili kuchunguza maadili katika seti ya data na kubaini matumizi sahihi (au yasiyo ya matumizi) ya maadili.

> 🚨 Tahadhari
>
> Unapofanya kazi na seti hii ya data utaandika misimbo inayohesabu kitu kutoka kwa maandishi bila kuwa na haja ya kusoma au kuchambua maandishi wewe mwenyewe. Hii ndio kiini cha NLP, kutafsiri maana au hisia bila kuwa na binadamu kufanya hivyo. Hata hivyo, inawezekana kwamba utasoma baadhi ya maoni hasi. Ningekushauri usifanye hivyo, kwa sababu huna haja ya kufanya hivyo. Baadhi yao ni ya kipuuzi, au maoni hasi yasiyohusiana na hoteli, kama "Hali ya hewa haikuwa nzuri", jambo ambalo liko nje ya uwezo wa hoteli, au hata mtu yeyote. Lakini kuna upande mweusi kwa baadhi ya maoni pia. Wakati mwingine maoni hasi ni ya kibaguzi, ya kijinsia, au ya umri. Hii ni bahati mbaya lakini inatarajiwa katika seti ya data iliyopatikana kutoka tovuti ya umma. Waandishi wa maoni wengine wanaacha maoni ambayo ungeona yasiyopendeza, yasiyofurahisha, au yenye kusikitisha. Bora kuacha msimbo kupima hisia kuliko kuyasoma mwenyewe na kusikitika. Hata hivyo, ni wachache wanaoandika vitu kama hivyo
rows have column `Positive_Review` values of "No Positive" 9. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ### Code answers 1. Print out the *shape* of the data frame you have just loaded (the shape is the number of rows and columns) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calculate the frequency count for reviewer nationalities: 1. How many distinct values are there for the column `Reviewer_Nationality` and what are they? 2. What reviewer nationality is the most common in the dataset (print country and number of reviews)? ```python
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
   ``` 3. What are the next top 10 most frequently found nationalities, and their frequency count? ```python
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
      ``` 3. What was the most frequently reviewed hotel for each of the top 10 most reviewer nationalities? ```python
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
   ``` 4. How many reviews are there per hotel (frequency count of hotel) in the dataset? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | You may notice that the *counted in the dataset* results do not match the value in `Total_Number_of_Reviews`. It is unclear if this value in the dataset represented the total number of reviews the hotel had, but not all were scraped, or some other calculation. `Total_Number_of_Reviews` is not used in the model because of this unclarity. 5. While there is an `Average_Score` column for each hotel in the dataset, you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel). Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. Print out the columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`. ```python
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
   ``` You may also wonder about the `Average_Score` value and why it is sometimes different from the calculated average score. As we can't know why some of the values match, but others have a difference, it's safest in this case to use the review scores that we have to calculate the average ourselves. That said, the differences are usually very small, here are the hotels with the greatest deviation from the dataset average and the calculated average: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | With only 1 hotel having a difference of score greater than 1, it means we can probably ignore the difference and use the calculated average score. 6. Calculate and print out how many rows have column `Negative_Review` values of "No Negative" 7. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" 8. Calculate and print out how many rows have column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative" ```python
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
   ``` ## Another way Another way count items without Lambdas, and use sum to count the rows: ```python
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
   ``` You may have noticed that there are 127 rows that have both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review` respectively. That means that the reviewer gave the hotel a numerical score, but declined to write either a positive or negative review. Luckily this is a small amount of rows (127 out of 515738, or 0.02%), so it probably won't skew our model or results in any particular direction, but you might not have expected a data set of reviews to have rows with no reviews, so it's worth exploring the data to discover rows like this. Now that you have explored the dataset, in the next lesson you will filter the data and add some sentiment analysis. --- ## 🚀Challenge This lesson demonstrates, as we saw in previous lessons, how critically important it is to understand your data and its foibles before performing operations on it. Text-based data, in particular, bears careful scrutiny. Dig through various text-heavy datasets and see if you can discover areas that could introduce bias or skewed sentiment into a model. ## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Review & Self Study Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models. ## Assignment [NLTK](assignment.md)

**Kanusho**: 
Hati hii imetafsiriwa kwa kutumia huduma za tafsiri za AI za mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwepo kwa usahihi. Hati ya asili katika lugha yake ya asili inapaswa kuzingatiwa kama chanzo chenye mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.