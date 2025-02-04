# Uchambuzi wa Hisia na Maoni ya Hoteli

Sasa kwa kuwa umechunguza kwa kina seti ya data, ni wakati wa kuchuja safu na kisha kutumia mbinu za NLP kwenye seti ya data ili kupata maarifa mapya kuhusu hoteli.
## [Jaribio la kabla ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Uchujaji na Shughuli za Uchambuzi wa Hisia

Kama ulivyoweza kugundua, seti ya data ina masuala kadhaa. Baadhi ya safu zimejazwa na taarifa zisizo na maana, zingine zinaonekana kuwa si sahihi. Ikiwa ni sahihi, haijulikani jinsi zilivyohesabiwa, na majibu hayawezi kuthibitishwa kwa uhuru kwa hesabu zako mwenyewe.

## Zoezi: usindikaji zaidi wa data

Safisha data kidogo zaidi. Ongeza safu ambazo zitakuwa na manufaa baadaye, badilisha thamani katika safu nyingine, na acha baadhi ya safu kabisa.

1. Usindikaji wa awali wa safu

   1. Acha `lat` na `lng`

   2. Badilisha thamani za `Hotel_Address` na thamani zifuatazo (ikiwa anwani ina jina la mji na nchi, badilisha iwe tu mji na nchi).

      Hizi ni miji na nchi pekee katika seti ya data:

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

  1. Acha `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` na hesabu yetu wenyewe

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Usindikaji wa safu za maoni

   1. Acha `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, unaweza kuanza kujiuliza kama inawezekana kupunguza kwa kiasi kikubwa usindikaji unaohitaji kufanya. Kwa bahati nzuri, inawezekana - lakini kwanza unahitaji kufuata hatua chache ili kubaini lebo za umuhimu.

### Kuchuja lebo

Kumbuka kwamba lengo la seti ya data ni kuongeza hisia na safu ambazo zitakusaidia kuchagua hoteli bora (kwa ajili yako au labda mteja anayekuagiza kutengeneza bot ya mapendekezo ya hoteli). Unahitaji kujiuliza kama lebo ni muhimu au la katika seti ya data ya mwisho. Hapa kuna tafsiri moja (ikiwa unahitaji seti ya data kwa sababu nyingine tofauti lebo zinaweza kubaki/kuondolewa kwenye uteuzi):

1. Aina ya safari ni muhimu, na hiyo inapaswa kubaki
2. Aina ya kikundi cha wageni ni muhimu, na hiyo inapaswa kubaki
3. Aina ya chumba, suite, au studio ambayo mgeni alikaa haina umuhimu (hoteli zote zina vyumba vya kimsingi sawa)
4. Kifaa ambacho maoni yalitumwa hakina umuhimu
5. Idadi ya usiku mtoa maoni alikaa inaweza kuwa muhimu ikiwa utahusisha kukaa kwa muda mrefu na wao kupenda hoteli zaidi, lakini ni jambo la mbali, na labda halina umuhimu

Kwa muhtasari, **weka aina 2 za lebo na ondoa nyingine**.

Kwanza, hutaki kuhesabu lebo mpaka ziwe katika muundo bora, hivyo inamaanisha kuondoa mabano na nukuu. Unaweza kufanya hivi kwa njia kadhaa, lakini unataka njia ya haraka zaidi kwani inaweza kuchukua muda mrefu kusindika data nyingi. Kwa bahati nzuri, pandas ina njia rahisi ya kufanya kila moja ya hatua hizi.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Kila lebo inakuwa kama: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

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

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

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

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

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

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

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

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` safu inafanana na moja ya safu mpya, ongeza 1, ikiwa haifanani, ongeza 0. Matokeo ya mwisho yatakuwa hesabu ya ni watoa maoni wangapi walichagua hoteli hii (kwa jumla) kwa mfano, biashara dhidi ya burudani, au kuleta mnyama, na hii ni taarifa muhimu wakati wa kupendekeza hoteli.

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

## Shughuli za Uchambuzi wa Hisia

Katika sehemu hii ya mwisho, utatumia uchambuzi wa hisia kwenye safu za maoni na kuhifadhi matokeo katika seti ya data.

## Zoezi: pakia na hifadhi data iliyochujwa

Kumbuka kwamba sasa unapakia seti ya data iliyochujwa ambayo iliokolewa katika sehemu iliyopita, **si** seti ya data asilia.

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

### Kuondoa maneno yasiyo na maana

Ikiwa ungeendesha Uchambuzi wa Hisia kwenye safu za maoni ya Negativity na Positivity, inaweza kuchukua muda mrefu. Imejaribiwa kwenye laptop yenye CPU ya kasi, ilichukua dakika 12 - 14 kulingana na maktaba ya hisia iliyotumiwa. Huo ni muda mrefu (kwa kiasi fulani), hivyo ni vyema kuchunguza kama inaweza kuharakishwa. 

Kuondoa maneno yasiyo na maana, au maneno ya kawaida ya Kiingereza ambayo hayabadilishi hisia ya sentensi, ni hatua ya kwanza. Kwa kuyaondoa, uchambuzi wa hisia unapaswa kuendeshwa haraka zaidi, lakini sio kuwa na usahihi mdogo (kwa kuwa maneno yasiyo na maana hayabadilishi hisia, lakini yanapunguza kasi ya uchambuzi). 

Maoni ya kirefu zaidi ya negativity yalikuwa na maneno 395, lakini baada ya kuondoa maneno yasiyo na maana, ni maneno 195.

Kuondoa maneno yasiyo na maana pia ni operesheni ya haraka, kuondoa maneno yasiyo na maana kutoka kwenye safu 2 za maoni zaidi ya mistari 515,000 ilichukua sekunde 3.3 kwenye kifaa cha majaribio. Inaweza kuchukua muda kidogo zaidi au kidogo kwako kulingana na kasi ya CPU ya kifaa chako, RAM, ikiwa una SSD au la, na baadhi ya mambo mengine. Ufupi wa operesheni hii inamaanisha kwamba ikiwa inaboresha muda wa uchambuzi wa hisia, basi inafaa kufanya.

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

Sasa unapaswa kuhesabu uchambuzi wa hisia kwa safu zote za maoni ya negativity na positivity, na kuhifadhi matokeo katika safu mpya 2. Jaribio la hisia litakuwa kulinganisha na alama ya mtoa maoni kwa maoni sawa. Kwa mfano, ikiwa hisia zinafikiri maoni ya negativity yalikuwa na hisia ya 1 (hisia chanya sana) na maoni ya positivity hisia ya 1, lakini mtoa maoni alitoa hoteli alama ya chini kabisa, basi ama maandishi ya maoni hayalingani na alama, au mchanganuzi wa hisia hakuweza kutambua hisia kwa usahihi. Unapaswa kutarajia baadhi ya alama za hisia kuwa si sahihi kabisa, na mara nyingi hiyo itaelezeka, kwa mfano maoni yanaweza kuwa na kejeli kali "Bila shaka NILIPENDA kulala katika chumba bila joto" na mchanganuzi wa hisia anafikiri hiyo ni hisia chanya, ingawa binadamu akisoma angejua ni kejeli. 

NLTK inatoa wachambuzi wa hisia tofauti wa kujifunza nao, na unaweza kuzibadilisha na kuona kama hisia ni sahihi zaidi au chini. Uchambuzi wa hisia wa VADER umetumika hapa.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

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

Baadaye katika programu yako wakati uko tayari kuhesabu hisia, unaweza kuitumia kwa kila maoni kama ifuatavyo:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Hii inachukua takriban sekunde 120 kwenye kompyuta yangu, lakini itatofautiana kwenye kila kompyuta. Ikiwa unataka kuchapisha matokeo na kuona kama hisia zinaendana na maoni:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Jambo la mwisho kabisa kufanya na faili kabla ya kuitumia kwenye changamoto, ni kuihifadhi! Unapaswa pia kuzingatia kupanga upya safu zako zote mpya ili ziwe rahisi kufanya kazi nazo (kwa binadamu, ni mabadiliko ya vipodozi).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Unapaswa kuendesha msimbo mzima kwa [notebook ya uchambuzi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (baada ya kuendesha [notebook yako ya kuchuja](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ili kuzalisha faili ya Hotel_Reviews_Filtered.csv).

Ili kukagua, hatua ni:

1. Faili asilia ya seti ya data **Hotel_Reviews.csv** inachunguzwa katika somo lililopita na [notebook ya uchunguzi](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv inachujwa na [notebook ya kuchuja](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) na kusababisha **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv inasindikwa na [notebook ya uchambuzi wa hisia](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) na kusababisha **Hotel_Reviews_NLP.csv**
4. Tumia Hotel_Reviews_NLP.csv katika Changamoto ya NLP hapa chini

### Hitimisho

Ulipoanza, ulikuwa na seti ya data yenye safu na data lakini si yote inaweza kuthibitishwa au kutumika. Umechunguza data, umechambua kile usichohitaji, umebadilisha lebo kuwa kitu muhimu, umehesabu wastani wako mwenyewe, umeongeza safu za hisia na kwa matumaini, umejifunza mambo ya kuvutia kuhusu usindikaji wa maandishi ya asili.

## [Jaribio la baada ya somo](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Changamoto

Sasa kwa kuwa una seti yako ya data iliyochambuliwa kwa hisia, angalia kama unaweza kutumia mikakati uliyojifunza katika mtaala huu (labda kugrupu?) ili kubaini mifumo kuzunguka hisia. 

## Mapitio na Kujisomea

Chukua [moduli hii ya Kujifunza](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) kujifunza zaidi na kutumia zana tofauti kuchunguza hisia katika maandishi.
## Kazi 

[Jaribu seti tofauti ya data](assignment.md)

**Kanusho**: 
Hati hii imetafsiriwa kwa kutumia huduma za tafsiri za AI za mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kuwa tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokamilika. Hati asilia katika lugha yake ya asili inapaswa kuzingatiwa kama chanzo rasmi. Kwa taarifa muhimu, tafsiri ya kitaalamu ya kibinadamu inapendekezwa. Hatutawajibika kwa kutoelewana au tafsiri zisizo sahihi zinazotokana na matumizi ya tafsiri hii.