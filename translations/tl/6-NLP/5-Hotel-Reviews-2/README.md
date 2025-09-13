<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T18:28:06+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "tl"
}
-->
# Sentiment analysis gamit ang mga review ng hotel

Ngayon na napag-aralan mo nang mabuti ang dataset, oras na para i-filter ang mga column at gamitin ang mga teknik ng NLP sa dataset upang makakuha ng mga bagong insight tungkol sa mga hotel.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Mga Operasyon sa Pag-filter at Sentiment Analysis

Tulad ng napansin mo, may ilang isyu ang dataset. Ang ilang mga column ay puno ng walang kwentang impormasyon, ang iba naman ay tila mali. Kung tama man ang mga ito, hindi malinaw kung paano ito kinalkula, at ang mga sagot ay hindi maaaring independiyenteng ma-verify gamit ang sarili mong mga kalkulasyon.

## Ehersisyo: Karagdagang pagproseso ng data

Linisin ang data nang kaunti pa. Magdagdag ng mga column na magiging kapaki-pakinabang sa hinaharap, baguhin ang mga halaga sa ibang mga column, at tuluyang tanggalin ang ilang mga column.

1. Paunang pagproseso ng mga column

   1. Tanggalin ang `lat` at `lng`

   2. Palitan ang mga halaga ng `Hotel_Address` gamit ang mga sumusunod na halaga (kung ang address ay naglalaman ng pangalan ng lungsod at bansa, palitan ito ng lungsod at bansa lamang).

      Narito ang mga lungsod at bansa sa dataset:

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

      Ngayon maaari kang mag-query ng data sa antas ng bansa:

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

2. Proseso ng mga column ng Hotel Meta-review

  1. Tanggalin ang `Additional_Number_of_Scoring`

  1. Palitan ang `Total_Number_of_Reviews` gamit ang kabuuang bilang ng mga review para sa hotel na aktwal na nasa dataset 

  1. Palitan ang `Average_Score` gamit ang sarili nating kalkuladong score

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Proseso ng mga column ng review

   1. Tanggalin ang `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` at `days_since_review`

   2. Panatilihin ang `Reviewer_Score`, `Negative_Review`, at `Positive_Review` sa kanilang kasalukuyang anyo,
     
   3. Panatilihin ang `Tags` pansamantala

     - Magkakaroon tayo ng karagdagang mga operasyon sa pag-filter sa mga tag sa susunod na seksyon at pagkatapos ay tatanggalin ang mga tag

4. Proseso ng mga column ng reviewer

  1. Tanggalin ang `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Panatilihin ang `Reviewer_Nationality`

### Mga column ng Tag

Ang column na `Tag` ay may problema dahil ito ay isang listahan (sa anyong text) na nakaimbak sa column. Sa kasamaang palad, ang pagkakasunod-sunod at bilang ng mga sub-seksyon sa column na ito ay hindi palaging pareho. Mahirap para sa tao na tukuyin ang tamang mga parirala na dapat pagtuunan ng pansin, dahil mayroong 515,000 na mga row, at 1427 na mga hotel, at bawat isa ay may bahagyang magkakaibang mga opsyon na maaaring piliin ng reviewer. Dito nagiging kapaki-pakinabang ang NLP. Maaari mong i-scan ang teksto at hanapin ang mga pinakakaraniwang parirala, at bilangin ang mga ito.

Sa kasamaang palad, hindi tayo interesado sa mga salitang mag-isa, kundi sa mga parirala na binubuo ng maraming salita (hal. *Business trip*). Ang pagpapatakbo ng isang algorithm para sa pamamahagi ng dalas ng mga parirala sa ganitong kalaking data (6762646 na mga salita) ay maaaring tumagal ng napakahabang oras, ngunit nang hindi tinitingnan ang data, tila ito ay isang kinakailangang gastusin. Dito nagiging kapaki-pakinabang ang exploratory data analysis, dahil nakita mo ang sample ng mga tag tulad ng `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, maaari kang magsimulang magtanong kung posible bang lubos na bawasan ang pagproseso na kailangan mong gawin. Sa kabutihang palad, posible ito - ngunit kailangan mo munang sundin ang ilang hakbang upang matukoy ang mga tag na mahalaga.

### Pag-filter ng mga tag

Tandaan na ang layunin ng dataset ay magdagdag ng sentiment at mga column na makakatulong sa iyo na pumili ng pinakamahusay na hotel (para sa iyong sarili o marahil sa isang kliyente na nag-aatas sa iyo na gumawa ng hotel recommendation bot). Kailangan mong tanungin ang iyong sarili kung ang mga tag ay kapaki-pakinabang o hindi sa panghuling dataset. Narito ang isang interpretasyon (kung kailangan mo ang dataset para sa ibang mga layunin, maaaring manatili/matanggal ang iba't ibang mga tag):

1. Ang uri ng biyahe ay mahalaga, at dapat manatili
2. Ang uri ng grupo ng bisita ay mahalaga, at dapat manatili
3. Ang uri ng kwarto, suite, o studio na tinuluyan ng bisita ay hindi mahalaga (lahat ng hotel ay may halos parehong mga kwarto)
4. Ang device na ginamit sa pagsusumite ng review ay hindi mahalaga
5. Ang bilang ng mga gabi na tinuluyan ng reviewer *maaaring* mahalaga kung iugnay mo ang mas mahabang pananatili sa mas mataas na kasiyahan sa hotel, ngunit malabo ito, at marahil hindi mahalaga

Sa kabuuan, **panatilihin ang 2 uri ng mga tag at tanggalin ang iba pa**.

Una, ayaw mong bilangin ang mga tag hangga't hindi sila nasa mas maayos na format, kaya nangangahulugan ito ng pag-aalis ng mga square bracket at mga quote. Maaari mong gawin ito sa iba't ibang paraan, ngunit gusto mo ang pinakamabilis dahil maaaring tumagal ng mahabang oras ang pagproseso ng maraming data. Sa kabutihang palad, may madaling paraan ang pandas para gawin ang bawat isa sa mga hakbang na ito.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Ang bawat tag ay nagiging ganito: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Susunod, makakakita tayo ng problema. Ang ilang mga review, o mga row, ay may 5 column, ang iba ay may 3, ang iba naman ay may 6. Ito ay resulta ng kung paano nilikha ang dataset, at mahirap ayusin. Gusto mong makakuha ng frequency count ng bawat parirala, ngunit ang mga ito ay nasa iba't ibang pagkakasunod-sunod sa bawat review, kaya maaaring mali ang bilang, at maaaring hindi ma-assign sa isang hotel ang tag na nararapat dito.

Sa halip, gagamitin mo ang iba't ibang pagkakasunod-sunod sa iyong kalamangan, dahil ang bawat tag ay multi-word ngunit hiwalay din ng isang comma! Ang pinakasimpleng paraan para gawin ito ay lumikha ng 6 na pansamantalang column kung saan ang bawat tag ay ipapasok sa column na tumutugma sa pagkakasunod-sunod nito sa tag. Pagkatapos ay maaari mong pagsamahin ang 6 na column sa isang malaking column at patakbuhin ang `value_counts()` method sa resulting column. Kapag na-print mo ito, makikita mo na mayroong 2428 natatanging mga tag. Narito ang isang maliit na sample:

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

Ang ilan sa mga karaniwang tag tulad ng `Submitted from a mobile device` ay walang silbi sa atin, kaya maaaring matalinong tanggalin ang mga ito bago bilangin ang paglitaw ng parirala, ngunit ito ay napakabilis na operasyon kaya maaari mo silang iwanan at huwag pansinin.

### Pagtanggal ng mga tag ng haba ng pananatili

Ang pagtanggal ng mga tag na ito ay hakbang 1, binabawasan nito ang kabuuang bilang ng mga tag na dapat isaalang-alang. Tandaan na hindi mo sila tinatanggal mula sa dataset, kundi pinipili lamang na huwag isama ang mga ito bilang mga halaga na bibilangin/panatilihin sa dataset ng mga review.

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

Mayroong napakaraming uri ng mga kwarto, suite, studio, apartment, at iba pa. Lahat ng mga ito ay halos pareho ang ibig sabihin at hindi mahalaga sa iyo, kaya tanggalin ang mga ito mula sa konsiderasyon.

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

Sa wakas, at ito ay kahanga-hanga (dahil hindi ito nangangailangan ng masyadong maraming pagproseso), maiiwan ka sa mga sumusunod na *kapaki-pakinabang* na mga tag:

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

Maaari mong sabihin na ang `Travellers with friends` ay halos pareho sa `Group`, at makatarungan na pagsamahin ang dalawa tulad ng nasa itaas. Ang code para sa pagtukoy ng tamang mga tag ay nasa [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Ang huling hakbang ay lumikha ng mga bagong column para sa bawat isa sa mga tag na ito. Pagkatapos, para sa bawat review row, kung ang column na `Tag` ay tumutugma sa isa sa mga bagong column, magdagdag ng 1, kung hindi, magdagdag ng 0. Ang resulta ay magiging bilang ng kung ilang reviewer ang pumili sa hotel na ito (sa kabuuan) para sa, halimbawa, negosyo kumpara sa leisure, o para magdala ng alagang hayop, at ito ay kapaki-pakinabang na impormasyon kapag nagrerekomenda ng hotel.

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

### I-save ang iyong file

Sa wakas, i-save ang dataset sa kasalukuyang anyo nito gamit ang bagong pangalan.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Mga Operasyon sa Sentiment Analysis

Sa huling seksyon na ito, mag-a-apply ka ng sentiment analysis sa mga column ng review at i-save ang mga resulta sa dataset.

## Ehersisyo: I-load at i-save ang na-filter na data

Tandaan na ngayon ay ilo-load mo ang na-filter na dataset na na-save sa nakaraang seksyon, **hindi** ang orihinal na dataset.

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

### Pagtanggal ng mga stop word

Kung magpapatakbo ka ng Sentiment Analysis sa mga column ng Negative at Positive review, maaaring tumagal ito ng mahabang oras. Sa pagsubok gamit ang isang makapangyarihang test laptop na may mabilis na CPU, tumagal ito ng 12 - 14 minuto depende sa kung aling sentiment library ang ginamit. Ito ay isang (relatibong) mahabang oras, kaya sulit na imbestigahan kung maaari itong mapabilis. 

Ang pagtanggal ng mga stop word, o mga karaniwang salitang Ingles na hindi nagbabago sa sentiment ng isang pangungusap, ang unang hakbang. Sa pamamagitan ng pagtanggal ng mga ito, ang sentiment analysis ay dapat tumakbo nang mas mabilis, ngunit hindi magiging mas mababa ang accuracy (dahil ang mga stop word ay hindi nakakaapekto sa sentiment, ngunit pinapabagal nila ang analysis). 

Ang pinakamahabang negative review ay may 395 na salita, ngunit pagkatapos tanggalin ang mga stop word, ito ay naging 195 na salita.

Ang pagtanggal ng mga stop word ay isang mabilis na operasyon din, ang pagtanggal ng mga stop word mula sa 2 review column sa mahigit 515,000 na mga row ay tumagal ng 3.3 segundo sa test device. Maaaring tumagal ito ng bahagyang mas matagal o mas mabilis para sa iyo depende sa bilis ng CPU ng iyong device, RAM, kung mayroon kang SSD o wala, at ilang iba pang mga salik. Ang relatibong ikli ng operasyon ay nangangahulugan na kung mapapabilis nito ang sentiment analysis, sulit itong gawin.

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

### Pagganap ng sentiment analysis

Ngayon ay dapat mong kalkulahin ang sentiment analysis para sa parehong negative at positive review column, at i-store ang resulta sa 2 bagong column. Ang pagsubok ng sentiment ay ihahambing ito sa score ng reviewer para sa parehong review. Halimbawa, kung ang sentiment ay nag-iisip na ang negative review ay may sentiment na 1 (sobrang positibong sentiment) at ang positive review sentiment ay 1, ngunit binigyan ng reviewer ang hotel ng pinakamababang score na posible, maaaring hindi tumutugma ang review text sa score, o maaaring hindi nakilala ng sentiment analyser ang sentiment nang tama. Dapat mong asahan na ang ilang sentiment scores ay ganap na mali, at madalas na maipapaliwanag ito, halimbawa, ang review ay maaaring sobrang sarcastic "Siyempre NAGUSTUHAN ko ang pagtulog sa kwarto na walang heating" at iniisip ng sentiment analyser na positibo ang sentiment, kahit na alam ng tao na ito ay sarcasm.
NLTK ay nagbibigay ng iba't ibang sentiment analyzers na maaaring pag-aralan, at maaari mong palitan ang mga ito upang makita kung mas tumpak o hindi ang sentiment analysis. Ang VADER sentiment analysis ang ginamit dito.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Isang Simpleng Rule-based Model para sa Sentiment Analysis ng Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, Hunyo 2014.

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

Sa iyong programa, kapag handa ka nang kalkulahin ang sentiment, maaari mo itong i-apply sa bawat review tulad ng sumusunod:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Tumatagal ito ng humigit-kumulang 120 segundo sa aking computer, ngunit mag-iiba ito depende sa bawat computer. Kung nais mong i-print ang mga resulta at tingnan kung tumutugma ang sentiment sa review:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Ang huling bagay na dapat gawin sa file bago ito gamitin sa challenge ay ang i-save ito! Dapat mo ring isaalang-alang ang pag-aayos ng lahat ng iyong bagong columns upang mas madali itong gamitin (para sa tao, ito ay isang cosmetic na pagbabago).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Dapat mong patakbuhin ang buong code para sa [analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (pagkatapos mong patakbuhin ang [filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) upang makabuo ng Hotel_Reviews_Filtered.csv file).

Para sa pagsusuri, ang mga hakbang ay:

1. Ang orihinal na dataset file **Hotel_Reviews.csv** ay sinuri sa nakaraang aralin gamit ang [explorer notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Ang Hotel_Reviews.csv ay na-filter gamit ang [filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) na nagresulta sa **Hotel_Reviews_Filtered.csv**
3. Ang Hotel_Reviews_Filtered.csv ay na-proseso gamit ang [sentiment analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) na nagresulta sa **Hotel_Reviews_NLP.csv**
4. Gamitin ang Hotel_Reviews_NLP.csv sa NLP Challenge sa ibaba

### Konklusyon

Sa simula, mayroon kang dataset na may mga columns at data ngunit hindi lahat ay maaaring ma-verify o magamit. Sinuri mo ang data, tinanggal ang hindi mo kailangan, binago ang mga tags sa mas kapaki-pakinabang na anyo, kinalkula ang sarili mong mga average, nagdagdag ng ilang sentiment columns, at sana, natutunan mo ang mga kawili-wiling bagay tungkol sa pagproseso ng natural na teksto.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Hamon

Ngayon na na-analyze mo na ang dataset para sa sentiment, subukan mong gamitin ang mga estratehiyang natutunan mo sa kurikulum na ito (clustering, marahil?) upang tukuyin ang mga pattern sa paligid ng sentiment.

## Pagsusuri at Pag-aaral sa Sarili

Kunin ang [Learn module na ito](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) upang matuto pa at gumamit ng iba't ibang tools para suriin ang sentiment sa teksto.

## Takdang Aralin

[Subukan ang ibang dataset](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.