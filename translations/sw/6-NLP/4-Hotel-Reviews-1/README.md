<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T16:52:54+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "sw"
}
-->
# Uchambuzi wa hisia kwa maoni ya hoteli - kuchakata data

Katika sehemu hii utatumia mbinu ulizojifunza katika masomo ya awali kufanya uchambuzi wa data wa awali kwenye seti kubwa ya data. Mara utakapopata uelewa mzuri wa umuhimu wa safu mbalimbali, utajifunza:

- jinsi ya kuondoa safu zisizo za lazima
- jinsi ya kuhesabu data mpya kulingana na safu zilizopo
- jinsi ya kuhifadhi seti ya data iliyosindikwa kwa matumizi katika changamoto ya mwisho

## [Jaribio la kabla ya somo](https://ff-quizzes.netlify.app/en/ml/)

### Utangulizi

Hadi sasa umejifunza jinsi data ya maandishi inavyotofautiana na aina za data za nambari. Ikiwa ni maandishi yaliyoandikwa au kusemwa na binadamu, yanaweza kuchambuliwa ili kupata mifumo na marudio, hisia na maana. Somo hili linakuletea seti halisi ya data na changamoto halisi: **[Maoni ya Hoteli 515K Barani Ulaya](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** ambayo ina [Leseni ya CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Data hii ilikusanywa kutoka Booking.com kupitia vyanzo vya umma. Muundaji wa seti ya data ni Jiashen Liu.

### Maandalizi

Utahitaji:

* Uwezo wa kuendesha .ipynb notebooks kwa kutumia Python 3
* pandas
* NLTK, [ambayo unapaswa kusakinisha kwa ndani](https://www.nltk.org/install.html)
* Seti ya data inayopatikana kwenye Kaggle [Maoni ya Hoteli 515K Barani Ulaya](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ina ukubwa wa takriban 230 MB baada ya kufunguliwa. Pakua kwenye folda ya mizizi `/data` inayohusiana na masomo haya ya NLP.

## Uchambuzi wa data wa awali

Changamoto hii inadhani kuwa unajenga roboti ya mapendekezo ya hoteli kwa kutumia uchambuzi wa hisia na alama za maoni ya wageni. Seti ya data unayotumia inajumuisha maoni ya hoteli 1493 katika miji 6.

Kwa kutumia Python, seti ya data ya maoni ya hoteli, na uchambuzi wa hisia wa NLTK unaweza kugundua:

* Ni maneno na misemo gani inayotumika mara nyingi zaidi katika maoni?
* Je, *lebo* rasmi zinazofafanua hoteli zinahusiana na alama za maoni (kwa mfano, je, maoni hasi zaidi kwa hoteli fulani yanatoka kwa *Familia yenye watoto wadogo* kuliko *Msafiri peke yake*, labda ikionyesha kuwa ni bora kwa *Wasafiri peke yao*)?
* Je, alama za hisia za NLTK 'zinakubaliana' na alama ya nambari ya mtoa maoni wa hoteli?

#### Seti ya data

Hebu tuchunguze seti ya data uliyopakua na kuhifadhi kwa ndani. Fungua faili katika mhariri kama VS Code au hata Excel.

Vichwa vya safu katika seti ya data ni kama ifuatavyo:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Hapa vimepangwa kwa njia ambayo inaweza kuwa rahisi kuchunguza:  
##### Safu za hoteli

* `Hotel_Name`, `Hotel_Address`, `lat` (latitudo), `lng` (longitudo)
  * Kwa kutumia *lat* na *lng* unaweza kuchora ramani kwa Python inayoonyesha maeneo ya hoteli (labda kwa rangi tofauti kwa maoni hasi na chanya)
  * Hotel_Address haionekani kuwa na manufaa kwetu, na labda tutaibadilisha na nchi kwa urahisi wa kupanga na kutafuta

**Safu za meta-maoni ya hoteli**

* `Average_Score`
  * Kulingana na muundaji wa seti ya data, safu hii ni *Alama ya wastani ya hoteli, iliyohesabiwa kulingana na maoni ya hivi karibuni katika mwaka uliopita*. Hii inaonekana kama njia isiyo ya kawaida ya kuhesabu alama, lakini ni data iliyokusanywa kwa hivyo tunaweza kuichukua kama ilivyo kwa sasa. 
  
  ✅ Kulingana na safu nyingine katika data hii, unaweza kufikiria njia nyingine ya kuhesabu alama ya wastani?

* `Total_Number_of_Reviews`
  * Jumla ya idadi ya maoni ambayo hoteli hii imepokea - haijabainishwa (bila kuandika msimbo) ikiwa hii inahusu maoni katika seti ya data.
* `Additional_Number_of_Scoring`
  * Hii inamaanisha alama ya maoni ilitolewa lakini hakuna maoni chanya au hasi yaliyoandikwa na mtoa maoni

**Safu za maoni**

- `Reviewer_Score`
  - Hii ni thamani ya nambari yenye desimali moja kati ya thamani ya chini na ya juu 2.5 na 10
  - Haijafafanuliwa kwa nini 2.5 ndiyo alama ya chini inayowezekana
- `Negative_Review`
  - Ikiwa mtoa maoni hakuandika chochote, safu hii itakuwa na "**No Negative**"
  - Kumbuka kuwa mtoa maoni anaweza kuandika maoni chanya katika safu ya maoni hasi (kwa mfano, "hakuna kitu kibaya kuhusu hoteli hii")
- `Review_Total_Negative_Word_Counts`
  - Idadi kubwa ya maneno hasi inaonyesha alama ya chini (bila kuangalia hisia)
- `Positive_Review`
  - Ikiwa mtoa maoni hakuandika chochote, safu hii itakuwa na "**No Positive**"
  - Kumbuka kuwa mtoa maoni anaweza kuandika maoni hasi katika safu ya maoni chanya (kwa mfano, "hakuna kitu kizuri kuhusu hoteli hii kabisa")
- `Review_Total_Positive_Word_Counts`
  - Idadi kubwa ya maneno chanya inaonyesha alama ya juu (bila kuangalia hisia)
- `Review_Date` na `days_since_review`
  - Kipimo cha ukale au upya kinaweza kutumika kwa maoni (maoni ya zamani yanaweza kuwa si sahihi kama ya hivi karibuni kwa sababu usimamizi wa hoteli umebadilika, au ukarabati umefanyika, au bwawa limeongezwa n.k.)
- `Tags`
  - Hizi ni maelezo mafupi ambayo mtoa maoni anaweza kuchagua kuelezea aina ya mgeni aliyekuwa (kwa mfano, peke yake au familia), aina ya chumba alichokuwa nacho, muda wa kukaa na jinsi maoni yalivyowasilishwa. 
  - Kwa bahati mbaya, kutumia lebo hizi ni changamoto, angalia sehemu hapa chini inayojadili umuhimu wake

**Safu za mtoa maoni**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Hii inaweza kuwa sababu katika modeli ya mapendekezo, kwa mfano, ikiwa unaweza kubaini kuwa watoa maoni wenye maoni mengi zaidi ya mamia walikuwa na uwezekano mkubwa wa kuwa hasi kuliko chanya. Hata hivyo, mtoa maoni wa maoni yoyote maalum hajabainishwa kwa msimbo wa kipekee, na kwa hivyo hawezi kuunganishwa na seti ya maoni. Kuna watoa maoni 30 wenye maoni 100 au zaidi, lakini ni vigumu kuona jinsi hii inaweza kusaidia modeli ya mapendekezo.
- `Reviewer_Nationality`
  - Watu wengine wanaweza kufikiria kuwa mataifa fulani yana uwezekano mkubwa wa kutoa maoni chanya au hasi kwa sababu ya mwelekeo wa kitaifa. Kuwa makini kujenga maoni ya aina hii ya kimazungumzo katika modeli zako. Hizi ni dhana za kitaifa (na wakati mwingine za kikabila), na kila mtoa maoni alikuwa mtu binafsi aliyeandika maoni kulingana na uzoefu wake. Inaweza kuwa imechujwa kupitia lenzi nyingi kama vile kukaa kwake kwa hoteli za awali, umbali aliosafiri, na tabia yake binafsi. Kufikiria kuwa utaifa wao ndio sababu ya alama ya maoni ni vigumu kuthibitisha.

##### Mifano

| Alama ya Wastani | Jumla ya Maoni | Alama ya Mtoa Maoni | Maoni Hasi                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Maoni Chanya                 | Lebo                                                                                      |
| ---------------- | -------------- | ------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8              | 1945           | 2.5                 | Hii kwa sasa si hoteli bali ni eneo la ujenzi Niliteswa kutoka asubuhi na mapema na siku nzima na kelele za ujenzi zisizokubalika wakati wa kupumzika baada ya safari ndefu na kufanya kazi katika chumba Watu walikuwa wakifanya kazi siku nzima kwa kutumia mashine za kuchimba katika vyumba vya karibu Niliomba kubadilisha chumba lakini hakuna chumba kimya kilichopatikana Ili kufanya mambo kuwa mabaya zaidi nilitozwa zaidi Niliondoka jioni kwa kuwa nilikuwa na safari ya mapema sana na kupokea bili inayofaa Siku moja baadaye hoteli ilifanya malipo mengine bila idhini yangu zaidi ya bei iliyowekwa Ni mahali pabaya Usijiteshe kwa kuweka nafasi hapa | Hakuna kitu kibaya Mahali pabaya Epuka | Safari ya kibiashara                                Wenzi Chumba cha kawaida cha kitanda mara mbili Kukaa usiku 2 |

Kama unavyoona, mgeni huyu hakuwa na kukaa kwa furaha katika hoteli hii. Hoteli ina alama nzuri ya wastani ya 7.8 na maoni 1945, lakini mtoa maoni huyu aliipa 2.5 na kuandika maneno 115 kuhusu jinsi kukaa kwake kulivyokuwa hasi. Ikiwa hakuandika chochote kabisa katika safu ya Maoni_Chanya, unaweza kudhani hakukuwa na kitu chanya, lakini kwa bahati mbaya aliandika maneno 7 ya onyo. Ikiwa tungehesabu maneno badala ya maana, au hisia za maneno, tunaweza kuwa na mtazamo uliopotoshwa wa nia ya mtoa maoni. Kwa kushangaza, alama yake ya 2.5 inachanganya, kwa sababu ikiwa kukaa kwake katika hoteli hiyo kulikuwa mbaya sana, kwa nini kutoa alama yoyote? Kuchunguza seti ya data kwa karibu, utaona kuwa alama ya chini kabisa inayowezekana ni 2.5, si 0. Alama ya juu kabisa inayowezekana ni 10.

##### Lebo

Kama ilivyotajwa hapo juu, kwa mtazamo wa kwanza, wazo la kutumia `Tags` kuainisha data lina mantiki. Kwa bahati mbaya lebo hizi hazijasanifishwa, ambayo inamaanisha kuwa katika hoteli fulani, chaguo zinaweza kuwa *Chumba cha mtu mmoja*, *Chumba cha mapacha*, na *Chumba cha kitanda mara mbili*, lakini katika hoteli nyingine, ni *Chumba cha mtu mmoja cha kifahari*, *Chumba cha kifalme cha kawaida*, na *Chumba cha kifalme cha mtendaji*. Hizi zinaweza kuwa vitu sawa, lakini kuna tofauti nyingi sana kwamba chaguo linakuwa:

1. Jaribu kubadilisha maneno yote kuwa kiwango kimoja, ambayo ni ngumu sana, kwa sababu haijabainishwa njia ya ubadilishaji itakuwa nini katika kila kesi (kwa mfano, *Chumba cha kawaida cha mtu mmoja* kinafanana na *Chumba cha mtu mmoja* lakini *Chumba cha kifalme cha kifahari na bustani ya ua au mwonekano wa jiji* ni ngumu zaidi kuoanisha)

1. Tunaweza kuchukua mbinu ya NLP na kupima marudio ya maneno fulani kama *Peke yake*, *Msafiri wa kibiashara*, au *Familia yenye watoto wadogo* jinsi yanavyotumika kwa kila hoteli, na kuingiza hilo katika mapendekezo  

Lebo kwa kawaida (lakini si kila mara) ni safu moja inayojumuisha orodha ya thamani 5 hadi 6 zilizotenganishwa kwa koma zinazolingana na *Aina ya safari*, *Aina ya wageni*, *Aina ya chumba*, *Idadi ya usiku*, na *Aina ya kifaa maoni yalivyowasilishwa*. Hata hivyo, kwa sababu watoa maoni wengine hawajazijaza kila sehemu (wanaweza kuacha moja tupu), thamani hazipo kila mara kwa mpangilio sawa.

Kwa mfano, chukua *Aina ya kikundi*. Kuna uwezekano 1025 wa kipekee katika safu hii ya `Tags`, na kwa bahati mbaya ni baadhi tu yao yanayorejelea kikundi (baadhi ni aina ya chumba n.k.). Ikiwa utachuja tu zile zinazotaja familia, matokeo yanajumuisha aina nyingi za *Chumba cha familia*. Ikiwa utajumuisha neno *na*, yaani, hesabu thamani za *Familia na*, matokeo ni bora, na zaidi ya 80,000 kati ya matokeo 515,000 yanajumuisha maneno "Familia yenye watoto wadogo" au "Familia yenye watoto wakubwa".

Hii inamaanisha safu ya lebo si bure kabisa kwetu, lakini itahitaji kazi ili iwe na manufaa.

##### Alama ya wastani ya hoteli

Kuna idadi ya mambo ya ajabu au tofauti na seti ya data ambayo siwezi kuelewa, lakini yameonyeshwa hapa ili uweze kuyatambua unapojenga modeli zako. Ikiwa utayagundua, tafadhali tujulishe katika sehemu ya majadiliano!

Seti ya data ina safu zifuatazo zinazohusiana na alama ya wastani na idadi ya maoni: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Hoteli moja yenye idadi kubwa zaidi ya maoni katika seti hii ya data ni *Britannia International Hotel Canary Wharf* yenye maoni 4789 kati ya 515,000. Lakini ikiwa tutaangalia thamani ya `Total_Number_of_Reviews` kwa hoteli hii, ni 9086. Unaweza kudhani kuwa kuna alama nyingi zaidi bila maoni, kwa hivyo labda tunapaswa kuongeza thamani ya safu ya `Additional_Number_of_Scoring`. Thamani hiyo ni 2682, na kuiongeza kwa 4789 inatufikisha 7471 ambayo bado ni pungufu ya 1615 kutoka kwa `Total_Number_of_Reviews`. 

Ikiwa utachukua safu ya `Average_Score`, unaweza kudhani ni wastani wa maoni katika seti ya data, lakini maelezo kutoka Kaggle ni "*Alama ya wastani ya hoteli, iliyohesabiwa kulingana na maoni ya hivi karibuni katika mwaka uliopita*". Hii haionekani kuwa na manufaa, lakini tunaweza kuhesabu wastani wetu wenyewe kulingana na alama za maoni katika seti ya data. Kwa kutumia hoteli hiyo hiyo kama mfano, alama ya wastani ya hoteli inatolewa kama 7.1 lakini alama iliyohesabiwa (wastani wa alama za mtoa maoni *katika* seti ya data) ni 6.8. Hii ni karibu, lakini si thamani sawa, na tunaweza tu kudhani kuwa alama zilizotolewa katika maoni ya `Additional_Number_of_Scoring` ziliongeza wastani hadi 7.1. Kwa bahati mbaya bila njia ya kujaribu au kuthibitisha dhana hiyo, ni vigumu kutumia au kuamini `Average_Score`, `Additional_Number_of_Scoring` na `Total_Number_of_Reviews` wakati zinategemea, au zinarejelea, data ambayo hatuna.

Ili kufanya mambo kuwa magumu zaidi, hoteli yenye idadi ya pili ya juu ya maoni ina alama ya wastani iliyohesabiwa ya 8.12 na `Average_Score` ya seti ya data ni 8.1. Je, alama hii sahihi ni bahati mbaya au hoteli ya kwanza ni tofauti? 

Kwa uwezekano kwamba hoteli hizi zinaweza kuwa tofauti, na kwamba labda thamani nyingi zinahesabu (lakini baadhi hazihesabu kwa sababu fulani) tutaandika programu fupi ijayo kuchunguza thamani katika seti ya data na kubaini matumizi sahihi (au kutotumia) ya thamani.
> 🚨 Tahadhari muhimu  
>  
> Unapofanya kazi na seti hii ya data, utaandika msimbo unaohesabu kitu kutoka kwa maandishi bila kulazimika kusoma au kuchambua maandishi mwenyewe. Hii ndiyo kiini cha NLP, kutafsiri maana au hisia bila kumhusisha binadamu moja kwa moja. Hata hivyo, kuna uwezekano kwamba utaona baadhi ya maoni hasi. Ningekushauri usifanye hivyo, kwa sababu huna haja ya kufanya hivyo. Baadhi ya maoni hayo ni ya kipuuzi, au hayana umuhimu, kama vile maoni hasi kuhusu hoteli yanayosema "Hali ya hewa haikuwa nzuri," jambo ambalo liko nje ya uwezo wa hoteli, au mtu yeyote. Lakini kuna upande wa giza kwa baadhi ya maoni pia. Wakati mwingine maoni hasi yanaweza kuwa ya kibaguzi, ya kijinsia, au ya umri. Hili ni jambo la kusikitisha lakini linaweza kutarajiwa katika seti ya data iliyokusanywa kutoka tovuti ya umma. Baadhi ya watu huacha maoni ambayo unaweza kuyachukia, kukosa raha, au kuhuzunika. Ni bora kuruhusu msimbo kupima hisia badala ya kuyasoma mwenyewe na kuhisi huzuni. Hata hivyo, ni wachache wanaoandika mambo kama hayo, lakini bado yapo.
## Zoezi - Uchunguzi wa Takwimu
### Pakia Takwimu

Hiyo inatosha kuangalia takwimu kwa macho, sasa utaandika msimbo na kupata majibu! Sehemu hii inatumia maktaba ya pandas. Kazi yako ya kwanza ni kuhakikisha unaweza kupakia na kusoma data ya CSV. Maktaba ya pandas ina kipakiaji cha haraka cha CSV, na matokeo huwekwa kwenye dataframe, kama ilivyo kwenye masomo ya awali. CSV tunayoipakia ina zaidi ya nusu milioni ya safu, lakini ina safu 17 tu. Pandas inakupa njia nyingi zenye nguvu za kuingiliana na dataframe, ikiwa ni pamoja na uwezo wa kufanya operesheni kwenye kila safu.

Kuanzia hapa katika somo hili, kutakuwa na vipande vya msimbo na maelezo ya msimbo pamoja na majadiliano kuhusu maana ya matokeo. Tumia _notebook.ipynb_ iliyojumuishwa kwa msimbo wako.

Tuanzie kwa kupakia faili ya data utakayotumia:

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

Sasa data imepakuliwa, tunaweza kufanya operesheni kadhaa juu yake. Weka msimbo huu juu ya programu yako kwa sehemu inayofuata.

## Chunguza Takwimu

Katika hali hii, data tayari ni *safi*, maana yake ni kwamba iko tayari kufanyiwa kazi, na haina herufi katika lugha nyingine ambazo zinaweza kusababisha matatizo kwa algorithimu zinazotarajia herufi za Kiingereza pekee.

✅ Unaweza kulazimika kufanya kazi na data inayohitaji usindikaji wa awali ili kuipanga kabla ya kutumia mbinu za NLP, lakini si wakati huu. Ikiwa ungetakiwa, ungewezaje kushughulikia herufi zisizo za Kiingereza?

Chukua muda kuhakikisha kwamba mara data inapopakuliwa, unaweza kuichunguza kwa msimbo. Ni rahisi kutaka kuzingatia safu za `Negative_Review` na `Positive_Review`. Zimejaa maandishi ya asili kwa algorithimu zako za NLP kuchakata. Lakini subiri! Kabla ya kuingia kwenye NLP na hisia, unapaswa kufuata msimbo hapa chini ili kuthibitisha kama maadili yaliyotolewa kwenye seti ya data yanalingana na maadili unayohesabu kwa pandas.

## Operesheni za Dataframe

Kazi ya kwanza katika somo hili ni kuangalia kama madai yafuatayo ni sahihi kwa kuandika msimbo unaochunguza dataframe (bila kuibadilisha).

> Kama ilivyo kwa kazi nyingi za programu, kuna njia kadhaa za kukamilisha hili, lakini ushauri mzuri ni kufanya kwa njia rahisi na rahisi unavyoweza, hasa ikiwa itakuwa rahisi kuelewa unapokuja tena kwenye msimbo huu siku zijazo. Kwa dataframes, kuna API ya kina ambayo mara nyingi itakuwa na njia ya kufanya unachotaka kwa ufanisi.

Chukulia maswali yafuatayo kama kazi za kuandika msimbo na jaribu kuyajibu bila kuangalia suluhisho.

1. Chapisha *umbo* la dataframe uliyopakia (umbo ni idadi ya safu na safu wima)
2. Hesabu idadi ya mara kwa mara ya uraia wa wakaguzi:
   1. Kuna maadili mangapi tofauti kwa safu wima `Reviewer_Nationality` na ni yapi?
   2. Uraia gani wa wakaguzi ni wa kawaida zaidi kwenye seti ya data (chapisha nchi na idadi ya ukaguzi)?
   3. Ni uraia gani 10 wa juu zaidi unaopatikana mara kwa mara, na idadi yao ya mara kwa mara?
3. Hoteli ipi ilikaguliwa mara nyingi zaidi kwa kila moja ya uraia 10 wa juu wa wakaguzi?
4. Kuna ukaguzi wangapi kwa kila hoteli (idadi ya mara kwa mara ya hoteli) kwenye seti ya data?
5. Ingawa kuna safu wima ya `Average_Score` kwa kila hoteli kwenye seti ya data, unaweza pia kuhesabu wastani wa alama (kupata wastani wa alama zote za wakaguzi kwenye seti ya data kwa kila hoteli). Ongeza safu mpya kwenye dataframe yako yenye kichwa cha safu wima `Calc_Average_Score` inayojumuisha wastani uliokokotwa.
6. Je, kuna hoteli yoyote yenye `Average_Score` na `Calc_Average_Score` sawa (imezungushwa hadi sehemu moja ya desimali)?
   1. Jaribu kuandika kazi ya Python inayochukua Series (safu) kama hoja na kulinganisha maadili, ikichapisha ujumbe wakati maadili hayalingani. Kisha tumia njia ya `.apply()` kuchakata kila safu kwa kazi hiyo.
7. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Negative_Review` ya "No Negative"
8. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Positive_Review` ya "No Positive"
9. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Positive_Review` ya "No Positive" **na** `Negative_Review` ya "No Negative"

### Majibu ya Msimbo

1. Chapisha *umbo* la dataframe uliyopakia (umbo ni idadi ya safu na safu wima)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Hesabu idadi ya mara kwa mara ya uraia wa wakaguzi:

   1. Kuna maadili mangapi tofauti kwa safu wima `Reviewer_Nationality` na ni yapi?
   2. Uraia gani wa wakaguzi ni wa kawaida zaidi kwenye seti ya data (chapisha nchi na idadi ya ukaguzi)?

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

   3. Ni uraia gani 10 wa juu zaidi unaopatikana mara kwa mara, na idadi yao ya mara kwa mara?

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

3. Hoteli ipi ilikaguliwa mara nyingi zaidi kwa kila moja ya uraia 10 wa juu wa wakaguzi?

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

4. Kuna ukaguzi wangapi kwa kila hoteli (idadi ya mara kwa mara ya hoteli) kwenye seti ya data?

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
   
   Unaweza kugundua kuwa matokeo ya *kuhesabiwa kwenye seti ya data* hayalingani na thamani katika `Total_Number_of_Reviews`. Haijulikani kama thamani hii kwenye seti ya data iliwakilisha idadi ya ukaguzi hoteli iliyo nayo, lakini si zote zilichanganuliwa, au hesabu nyingine. `Total_Number_of_Reviews` haitumiki kwenye modeli kwa sababu ya kutokueleweka huku.

5. Ingawa kuna safu wima ya `Average_Score` kwa kila hoteli kwenye seti ya data, unaweza pia kuhesabu wastani wa alama (kupata wastani wa alama zote za wakaguzi kwenye seti ya data kwa kila hoteli). Ongeza safu mpya kwenye dataframe yako yenye kichwa cha safu wima `Calc_Average_Score` inayojumuisha wastani uliokokotwa. Chapisha safu wima `Hotel_Name`, `Average_Score`, na `Calc_Average_Score`.

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

   Unaweza pia kushangaa kuhusu thamani ya `Average_Score` na kwa nini wakati mwingine ni tofauti na wastani uliokokotwa. Kwa kuwa hatuwezi kujua kwa nini baadhi ya maadili yanalingana, lakini mengine yana tofauti, ni salama katika hali hii kutumia alama za ukaguzi tulizo nazo kuhesabu wastani wenyewe. Hata hivyo, tofauti hizo kwa kawaida ni ndogo sana, hapa kuna hoteli zenye tofauti kubwa zaidi kati ya wastani wa seti ya data na wastani uliokokotwa:

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

   Kwa kuwa hoteli moja tu ina tofauti ya alama kubwa zaidi ya 1, ina maana tunaweza kupuuza tofauti hiyo na kutumia wastani uliokokotwa.

6. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Negative_Review` ya "No Negative" 

7. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Positive_Review` ya "No Positive"

8. Hesabu na chapisha idadi ya safu zenye maadili ya safu wima `Positive_Review` ya "No Positive" **na** `Negative_Review` ya "No Negative"

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

## Njia Nyingine

Njia nyingine ya kuhesabu vitu bila Lambdas, na kutumia sum kuhesabu safu:

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

   Unaweza kugundua kuwa kuna safu 127 zenye maadili ya "No Negative" na "No Positive" kwa safu wima `Negative_Review` na `Positive_Review` mtawalia. Hii ina maana kwamba mkaguzi alitoa hoteli alama ya nambari, lakini alikataa kuandika ukaguzi chanya au hasi. Kwa bahati nzuri, hii ni idadi ndogo ya safu (127 kati ya 515738, au 0.02%), kwa hivyo labda haitapotosha modeli yetu au matokeo kwa mwelekeo wowote, lakini huenda hukutarajia seti ya data ya ukaguzi kuwa na safu bila ukaguzi, kwa hivyo ni vyema kuchunguza data ili kugundua safu kama hizi.

Sasa kwa kuwa umechunguza seti ya data, katika somo linalofuata utachuja data na kuongeza uchambuzi wa hisia.

---
## 🚀Changamoto

Somo hili linaonyesha, kama tulivyoona katika masomo ya awali, jinsi ilivyo muhimu sana kuelewa data yako na kasoro zake kabla ya kufanya operesheni juu yake. Data inayotegemea maandishi, hasa, inahitaji uchunguzi wa makini. Chunguza seti mbalimbali za data zenye maandishi mengi na uone kama unaweza kugundua maeneo ambayo yanaweza kuingiza upendeleo au hisia zilizopotoka kwenye modeli.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Mapitio & Kujifunza Mwenyewe

Chukua [Njia hii ya Kujifunza kuhusu NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) ili kugundua zana za kujaribu wakati wa kujenga modeli zenye maandishi na sauti nyingi.

## Kazi 

[NLTK](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutokuelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.