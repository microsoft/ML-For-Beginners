<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T17:06:23+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "sw"
}
-->
# Uchambuzi wa Hisia kwa Maoni ya Hoteli

Sasa kwa kuwa umechunguza seti ya data kwa undani, ni wakati wa kuchuja safu na kisha kutumia mbinu za NLP kwenye seti ya data ili kupata maarifa mapya kuhusu hoteli.

## [Maswali ya awali ya somo](https://ff-quizzes.netlify.app/en/ml/)

### Uendeshaji wa Kuchuja na Uchambuzi wa Hisia

Kama ulivyogundua, seti ya data ina masuala kadhaa. Baadhi ya safu zimejaa taarifa zisizo na maana, nyingine zinaonekana kuwa si sahihi. Ikiwa ni sahihi, haijulikani jinsi zilivyohesabiwa, na majibu hayawezi kuthibitishwa kwa uhuru kupitia hesabu zako mwenyewe.

## Zoezi: Usindikaji wa data zaidi kidogo

Safisha data kidogo zaidi. Ongeza safu ambazo zitakuwa muhimu baadaye, badilisha maadili katika safu nyingine, na futa safu fulani kabisa.

1. Usindikaji wa awali wa safu

   1. Futa `lat` na `lng`

   2. Badilisha maadili ya `Hotel_Address` na maadili yafuatayo (ikiwa anwani ina jina la jiji na nchi, ibadilishe kuwa jiji na nchi tu).

      Haya ndiyo majiji na nchi pekee katika seti ya data:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Sasa unaweza kuuliza data ya kiwango cha nchi:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Usindikaji wa safu za Meta-review za Hoteli

  1. Futa `Additional_Number_of_Scoring`

  1. Badilisha `Total_Number_of_Reviews` na idadi ya jumla ya maoni ya hoteli hiyo ambayo kwa kweli yako kwenye seti ya data 

  1. Badilisha `Average_Score` na alama yetu iliyohesabiwa wenyewe

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Usindikaji wa safu za maoni

   1. Futa `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` na `days_since_review`

   2. Weka `Reviewer_Score`, `Negative_Review`, na `Positive_Review` kama zilivyo,
     
   3. Weka `Tags` kwa sasa

     - Tutafanya baadhi ya operesheni za kuchuja za ziada kwenye tags katika sehemu inayofuata na kisha tags zitafutwa

4. Usindikaji wa safu za mtoa maoni

  1. Futa `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Weka `Reviewer_Nationality`

### Safu za Tag

Safu ya `Tag` ni tatizo kwa kuwa ni orodha (katika mfumo wa maandishi) iliyohifadhiwa kwenye safu. Kwa bahati mbaya, mpangilio na idadi ya sehemu ndogo katika safu hii si sawa kila wakati. Ni vigumu kwa binadamu kutambua misemo sahihi ya kuvutiwa nayo, kwa sababu kuna safu 515,000, na hoteli 1427, na kila moja ina chaguo tofauti kidogo ambazo mtoa maoni anaweza kuchagua. Hapa ndipo NLP inang'aa. Unaweza kuchanganua maandishi na kupata misemo ya kawaida zaidi, na kuhesabu.

Kwa bahati mbaya, hatuvutiwi na maneno moja, bali misemo ya maneno mengi (mfano *Safari ya Biashara*). Kuendesha algoriti ya usambazaji wa maneno mengi kwenye data nyingi (6762646 maneno) inaweza kuchukua muda mwingi sana, lakini bila kuangalia data, inaonekana kuwa ni gharama ya lazima. Hapa ndipo uchambuzi wa data wa uchunguzi unakuwa muhimu, kwa sababu umeona sampuli ya tags kama `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, unaweza kuanza kuuliza ikiwa inawezekana kupunguza sana usindikaji unaohitaji kufanya. Kwa bahati nzuri, inawezekana - lakini kwanza unahitaji kufuata hatua chache ili kuthibitisha tags za kuvutiwa nazo.

### Kuchuja tags

Kumbuka kuwa lengo la seti ya data ni kuongeza hisia na safu ambazo zitakusaidia kuchagua hoteli bora (kwa ajili yako mwenyewe au labda mteja anayekuagiza kutengeneza bot ya mapendekezo ya hoteli). Unahitaji kujiuliza ikiwa tags ni muhimu au la katika seti ya data ya mwisho. Hapa kuna tafsiri moja (ikiwa unahitaji seti ya data kwa sababu nyingine tags tofauti zinaweza kubaki/kutolewa):

1. Aina ya safari ni muhimu, na hiyo inapaswa kubaki
2. Aina ya kikundi cha wageni ni muhimu, na hiyo inapaswa kubaki
3. Aina ya chumba, suite, au studio ambayo mgeni alikaa si muhimu (hoteli zote zina vyumba vya msingi sawa)
4. Kifaa ambacho maoni yalitumwa nacho si muhimu
5. Idadi ya usiku mtoa maoni alikaa *inaweza* kuwa muhimu ikiwa unahusisha kukaa kwa muda mrefu na wao kupenda hoteli zaidi, lakini ni jambo la kubahatisha, na labda si muhimu

Kwa muhtasari, **weka aina 2 za tags na ondoa zingine**.

Kwanza, hutaki kuhesabu tags hadi ziwe katika muundo bora, kwa hivyo hiyo inamaanisha kuondoa mabano ya mraba na nukuu. Unaweza kufanya hivi kwa njia kadhaa, lakini unataka njia ya haraka zaidi kwa kuwa inaweza kuchukua muda mrefu kusindika data nyingi. Kwa bahati nzuri, pandas ina njia rahisi ya kufanya kila moja ya hatua hizi.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Kila tag inakuwa kitu kama: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Kisha tunakutana na tatizo. Baadhi ya maoni, au safu, zina safu 5, nyingine 3, nyingine 6. Hii ni matokeo ya jinsi seti ya data ilivyoundwa, na ni vigumu kurekebisha. Unataka kupata hesabu ya mara kwa mara ya kila msemo, lakini ziko katika mpangilio tofauti katika kila maoni, kwa hivyo hesabu inaweza kuwa si sahihi, na hoteli inaweza kukosa tag ambayo ilistahili.

Badala yake utatumia mpangilio tofauti kwa faida yetu, kwa sababu kila tag ni maneno mengi lakini pia imetenganishwa na koma! Njia rahisi ya kufanya hivi ni kuunda safu 6 za muda na kila tag ikijazwa kwenye safu inayolingana na mpangilio wake katika tag. Kisha unaweza kuunganisha safu 6 kuwa safu moja kubwa na kuendesha njia ya `value_counts()` kwenye safu inayotokana. Ukiichapisha, utaona kulikuwa na tags 2428 za kipekee. Hapa kuna sampuli ndogo:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Baadhi ya tags za kawaida kama `Submitted from a mobile device` hazina maana kwetu, kwa hivyo inaweza kuwa jambo la busara kuziondoa kabla ya kuhesabu mara za msemo, lakini ni operesheni ya haraka sana unaweza kuziacha na kuzipuuza.

### Kuondoa tags za muda wa kukaa

Kuondoa tags hizi ni hatua ya kwanza, inapunguza idadi ya tags zinazozingatiwa kidogo. Kumbuka huziondoi kutoka kwenye seti ya data, unachagua tu kuziondoa kutoka kwa kuzingatiwa kama maadili ya kuhesabu/kubaki kwenye seti ya data ya maoni.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

Kuna aina mbalimbali za vyumba, suites, studio, vyumba vya ghorofa na kadhalika. Vyote vinamaanisha takriban kitu kimoja na si muhimu kwako, kwa hivyo ondoa kutoka kwa kuzingatiwa.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Hatimaye, na hili ni jambo la kufurahisha (kwa sababu halikuchukua usindikaji mwingi hata kidogo), utabaki na tags zifuatazo *muhimu*:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

Unaweza kusema kwamba `Travellers with friends` ni sawa na `Group` kwa kiasi fulani, na hilo lingekuwa jambo la haki kuunganisha mbili kama ilivyo hapo juu. Nambari ya kutambua tags sahihi iko [katika daftari la Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Hatua ya mwisho ni kuunda safu mpya kwa kila moja ya tags hizi. Kisha, kwa kila safu ya maoni, ikiwa safu ya `Tag` inalingana na moja ya safu mpya, ongeza 1, ikiwa sivyo, ongeza 0. Matokeo ya mwisho yatakuwa hesabu ya ni watoa maoni wangapi walichagua hoteli hii (kwa jumla) kwa, tuseme, biashara dhidi ya burudani, au kuleta mnyama, na hii ni taarifa muhimu wakati wa kupendekeza hoteli.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Hifadhi faili yako

Hatimaye, hifadhi seti ya data kama ilivyo sasa na jina jipya.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Uendeshaji wa Uchambuzi wa Hisia

Katika sehemu hii ya mwisho, utatumia uchambuzi wa hisia kwenye safu za maoni na kuhifadhi matokeo kwenye seti ya data.

## Zoezi: pakia na hifadhi data iliyochujwa

Kumbuka kuwa sasa unapakia seti ya data iliyochujwa ambayo ilihifadhiwa katika sehemu iliyopita, **si** seti ya data ya awali.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Kuondoa maneno ya kawaida

Ikiwa ungeendesha Uchambuzi wa Hisia kwenye safu za maoni ya Negativity na Positivity, inaweza kuchukua muda mrefu. Imejaribiwa kwenye kompyuta ya majaribio yenye CPU ya haraka, ilichukua dakika 12 - 14 kulingana na ni maktaba gani ya hisia iliyotumika. Huo ni muda mrefu (kwa kiasi), kwa hivyo inafaa kuchunguza ikiwa inaweza kuharakishwa. 

Kuondoa maneno ya kawaida, au maneno ya Kiingereza ya kawaida ambayo hayabadilishi hisia za sentensi, ni hatua ya kwanza. Kwa kuyaondoa, uchambuzi wa hisia unapaswa kuendeshwa haraka, lakini usiwe na usahihi mdogo (kwa kuwa maneno ya kawaida hayabadilishi hisia, lakini yanapunguza kasi ya uchambuzi). 

Maoni ya Negativity yaliyo marefu zaidi yalikuwa na maneno 395, lakini baada ya kuondoa maneno ya kawaida, yanakuwa na maneno 195.

Kuondoa maneno ya kawaida pia ni operesheni ya haraka, kuondoa maneno ya kawaida kutoka safu 2 za maoni zaidi ya safu 515,000 ilichukua sekunde 3.3 kwenye kifaa cha majaribio. Inaweza kuchukua muda kidogo zaidi au kidogo kwako kulingana na kasi ya CPU ya kifaa chako, RAM, ikiwa una SSD au la, na baadhi ya mambo mengine. Ushorti wa operesheni unamaanisha kwamba ikiwa inaboresha muda wa uchambuzi wa hisia, basi inafaa kufanya.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Kufanya uchambuzi wa hisia

Sasa unapaswa kuhesabu uchambuzi wa hisia kwa safu za maoni ya Negativity na Positivity, na kuhifadhi matokeo kwenye safu mpya 2. Jaribio la hisia litakuwa kulinganisha na alama ya mtoa maoni kwa maoni hayo hayo. Kwa mfano, ikiwa hisia zinafikiri maoni ya Negativity yalikuwa na hisia ya 1 (hisia chanya sana) na hisia ya maoni ya Positivity ya 1, lakini mtoa maoni alitoa hoteli alama ya chini kabisa, basi ama maandishi ya maoni hayalingani na alama, au mchanganuzi wa hisia haukuweza kutambua hisia kwa usahihi. Unapaswa kutarajia baadhi ya alama za hisia kuwa si sahihi kabisa, na mara nyingi hilo litakuwa linaelezeka, mfano maoni yanaweza kuwa ya kejeli sana "Bila shaka NILIPENDA kulala kwenye chumba bila joto" na mchanganuzi wa hisia anafikiri hiyo ni hisia chanya, ingawa binadamu anayesoma angejua ni kejeli.
NLTK hutoa wachambuzi mbalimbali wa hisia za kujifunza, na unaweza kubadilisha na kuona kama hisia ni sahihi zaidi au chini. Uchambuzi wa hisia wa VADER umetumika hapa.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Mfano Rahisi wa Kanuni kwa Uchambuzi wa Hisia za Maandishi ya Mitandao ya Kijamii. Mkutano wa Nane wa Kimataifa wa Weblogs na Mitandao ya Kijamii (ICWSM-14). Ann Arbor, MI, Juni 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Baadaye katika programu yako, unapokuwa tayari kuhesabu hisia, unaweza kuitumia kwa kila tathmini kama ifuatavyo:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Hii inachukua takriban sekunde 120 kwenye kompyuta yangu, lakini itatofautiana kwa kila kompyuta. Ikiwa unataka kuchapisha matokeo na kuona kama hisia zinalingana na tathmini:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Jambo la mwisho kabisa kufanya na faili kabla ya kuitumia katika changamoto ni kuihifadhi! Unapaswa pia kuzingatia kupanga upya safu zako mpya zote ili ziwe rahisi kufanya kazi nazo (kwa binadamu, ni mabadiliko ya muonekano).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Unapaswa kuendesha msimbo mzima wa [daftari la uchambuzi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (baada ya kuendesha [daftari lako la kuchuja](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ili kuzalisha faili ya Hotel_Reviews_Filtered.csv).

Kwa muhtasari, hatua ni:

1. Faili ya dataset ya awali **Hotel_Reviews.csv** ilichunguzwa katika somo lililopita na [daftari la uchunguzi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv ilichujwa na [daftari la kuchuja](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) na kusababisha **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv ilichakatwa na [daftari la uchambuzi wa hisia](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) na kusababisha **Hotel_Reviews_NLP.csv**
4. Tumia Hotel_Reviews_NLP.csv katika Changamoto ya NLP hapa chini

### Hitimisho

Ulipoanza, ulikuwa na dataset yenye safu na data lakini si yote iliyoweza kuthibitishwa au kutumika. Umechunguza data, ukachuja kile usichohitaji, ukabadilisha vitambulisho kuwa kitu cha maana, ukahesabu wastani wako mwenyewe, ukaongeza safu za hisia na pengine, umejifunza mambo ya kuvutia kuhusu uchakataji wa maandishi asilia.

## [Jaribio la baada ya somo](https://ff-quizzes.netlify.app/en/ml/)

## Changamoto

Sasa kwa kuwa dataset yako imechambuliwa kwa hisia, jaribu kutumia mikakati uliyojifunza katika mtaala huu (labda clustering?) ili kubaini mifumo inayohusiana na hisia.

## Mapitio na Kujisomea

Chukua [moduli hii ya Kujifunza](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) ili kujifunza zaidi na kutumia zana tofauti kuchunguza hisia katika maandishi.

## Kazi

[Jaribu dataset tofauti](assignment.md)

---

**Kanusho**:  
Hati hii imetafsiriwa kwa kutumia huduma ya kutafsiri ya AI [Co-op Translator](https://github.com/Azure/co-op-translator). Ingawa tunajitahidi kuhakikisha usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokuwa sahihi. Hati ya asili katika lugha yake ya awali inapaswa kuzingatiwa kama chanzo cha mamlaka. Kwa taarifa muhimu, tafsiri ya kitaalamu ya binadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.