<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-11-18T18:28:59+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "pcm"
}
-->
# Sentiment analysis wit hotel reviews

Now wey you don check di dataset well well, na time to filter di columns and use NLP techniques for di dataset to get new gist about di hotels.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Filtering & Sentiment Analysis Operations

As you don notice, di dataset get some wahala. Some columns dey full wit tori wey no make sense, others dey look wrong. Even if dem dey correct, e no clear how dem take calculate am, and you no fit confirm di answers wit your own calculation.

## Exercise: small data processing

Make di data clean small. Add columns wey go dey useful later, change di values for some columns, and comot some columns completely.

1. First column processing

   1. Comot `lat` and `lng`

   2. Change `Hotel_Address` values to dis kind values (if di address get di name of di city and di country, change am to just di city and di country).

      Na only dis cities and countries dey di dataset:

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

      Now you fit query country level data:

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

2. Process Hotel Meta-review columns

  1. Comot `Additional_Number_of_Scoring`

  1. Change `Total_Number_of_Reviews` to di total number of reviews for dat hotel wey dey di dataset 

  1. Change `Average_Score` to di score wey we calculate by ourselves

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Process review columns

   1. Comot `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Leave `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as dem be,
     
   3. Leave `Tags` for now

     - We go do some extra filtering operations for di tags for di next section and then we go comot di tags

4. Process reviewer columns

  1. Comot `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Leave `Reviewer_Nationality`

### Tag columns

Di `Tag` column get wahala because e be list (inside text form) wey dem store for di column. Di order and number of sub sections for dis column no dey always di same. E hard for person to sabi di correct phrases wey dem suppose focus on, because di dataset get 515,000 rows, and 1427 hotels, and each one get small small different options wey reviewer fit choose. Na here NLP go help. You fit scan di text and find di most common phrases, and count dem.

Di wahala be say we no dey interested in single words, but multi-word phrases (e.g. *Business trip*). To run multi-word frequency distribution algorithm for dis plenty data (6762646 words) fit take plenty time, but if you no look di data, e go seem like say na wetin you suppose do. Na here exploratory data analysis go help, because you don see sample of di tags like `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, you fit begin ask if e possible to reduce di processing wey you suppose do. Luckily, e dey possible - but first you need follow some steps to sabi di tags wey dey important.

### Filtering tags

Remember say di goal of di dataset na to add sentiment and columns wey go help you choose di best hotel (for yourself or maybe client wey dey ask you to make hotel recommendation bot). You need ask yourself if di tags dey useful or not for di final dataset. Dis na one way to look am (if you need di dataset for other reasons, di tags wey go stay or comot fit dey different):

1. Di type of trip dey important, e suppose stay
2. Di type of guest group dey important, e suppose stay
3. Di type of room, suite, or studio wey di guest stay no dey important (all hotels get di same kind rooms)
4. Di device wey dem take submit di review no dey important
5. Di number of nights wey reviewer stay *fit* dey important if you think say longer stays mean dem like di hotel more, but e no sure, e fit no dey important

To summarize, **keep 2 kinds of tags and comot di others**.

First, you no go wan count di tags until dem dey better format, so dat mean say you go comot di square brackets and quotes. You fit do dis in different ways, but you go wan use di fastest way because e fit take long time to process plenty data. Luckily, pandas get easy way to do each of dis steps.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Each tag go turn something like: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we go see one wahala. Some reviews, or rows, get 5 columns, some get 3, some get 6. Na di way dem take create di dataset cause dis, and e hard to fix. You go wan get frequency count of each phrase, but dem dey different order for each review, so di count fit no correct, and hotel fit no get tag wey e suppose get.

Instead you go use di different order to your advantage, because each tag na multi-word but e dey separate by comma! Di simplest way na to create 6 temporary columns wey each tag go enter di column wey match di order for di tag. You fit then join di 6 columns into one big column and run di `value_counts()` method for di column wey result. If you print am out, you go see say e get 2428 unique tags. Dis na small sample:

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

Some of di common tags like `Submitted from a mobile device` no dey useful to us, so e go make sense to comot dem before you count di phrase occurrence, but e dey fast to do so you fit leave dem and just ignore dem.

### Removing di length of stay tags

To comot dis tags na di first step, e go reduce di total number of tags wey you go consider small. Note say you no go comot dem from di dataset, just choose to comot dem from di values wey you go count/keep for di reviews dataset.

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

E get plenty variety of rooms, suites, studios, apartments and so on. All of dem mean di same thing and no dey relevant to you, so comot dem from consideration.

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

Finally, and dis one sweet (because e no take plenty processing at all), you go remain wit di following *useful* tags:

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

You fit talk say `Travellers with friends` na di same as `Group` more or less, and e go make sense to join di two as above. Di code for identifying di correct tags dey [di Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Di final step na to create new columns for each of dis tags. Then, for every review row, if di `Tag` column match one of di new columns, add 1, if e no match, add 0. Di end result go be count of how many reviewers choose dis hotel (together) for, say, business vs leisure, or to bring pet come, and dis na useful gist when you dey recommend hotel.

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

### Save your file

Finally, save di dataset as e dey now wit new name.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentiment Analysis Operations

For dis final section, you go use sentiment analysis for di review columns and save di results for di dataset.

## Exercise: load and save di filtered data

Note say now you dey load di filtered dataset wey you save for di previous section, **no be** di original dataset.

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

### Removing stop words

If you run Sentiment Analysis for di Negative and Positive review columns, e fit take long time. Tested for one strong laptop wit fast CPU, e take 12 - 14 minutes depending on di sentiment library wey dem use. Dat na (relatively) long time, so e go make sense to check if e fit dey faster. 

To comot stop words, or common English words wey no dey change di sentiment of sentence, na di first step. If you comot dem, di sentiment analysis suppose run faster, but e no go dey less accurate (because di stop words no dey affect sentiment, but dem dey slow down di analysis). 

Di longest negative review na 395 words, but after you comot di stop words, e go be 195 words.

To comot di stop words na fast operation, to comot di stop words from 2 review columns for 515,000 rows take 3.3 seconds for di test device. E fit take small more or less time for you depending on your device CPU speed, RAM, whether you get SSD or not, and some other things. Di short time wey e take mean say if e go make di sentiment analysis faster, e go make sense to do am.

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

### Performing sentiment analysis

Now you suppose calculate di sentiment analysis for both negative and positive review columns, and store di result for 2 new columns. Di test of di sentiment go be to compare am wit di reviewer's score for di same review. For example, if di sentiment think say di negative review get sentiment of 1 (extremely positive sentiment) and di positive review sentiment na 1, but di reviewer give di hotel di lowest score wey e fit give, then e mean say di review text no match di score, or di sentiment analyser no fit recognize di sentiment well. You suppose expect some sentiment scores to dey completely wrong, and sometimes e go dey explainable, e.g. di review fit dey extremely sarcastic "Of course I LOVED sleeping in a room with no heating" and di sentiment analyser go think say na positive sentiment, even though person wey read am go sabi say na sarcasm. 
NLTK get different sentiment analyzer wey you fit learn with, and you fit change am to see if the sentiment dey more or less correct. Na VADER sentiment analysis dem use here.

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

Later for your program when you wan calculate sentiment, you fit use am for each review like this:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

E go take like 120 seconds for my computer, but e fit dey different for other computers. If you wan print the results and check whether the sentiment match the review:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

The last thing wey you go do with the file before you use am for the challenge na to save am! You fit also think about how you go arrange all your new columns so e go dey easy to work with (for person, na just cosmetic change).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

You suppose run the whole code for [the analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (after you don run [your filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) to generate the Hotel_Reviews_Filtered.csv file).

To summarize, the steps na:

1. Original dataset file **Hotel_Reviews.csv** wey dem explore for the previous lesson with [the explorer notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv wey dem filter with [the filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) wey result in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv wey dem process with [the sentiment analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) wey result in **Hotel_Reviews_NLP.csv**
4. Use Hotel_Reviews_NLP.csv for the NLP Challenge below

### Conclusion

When you start, you get dataset wey get columns and data but no be all of am fit verify or use. You don explore the data, remove wetin you no need, change tags to something wey dey useful, calculate your own averages, add some sentiment columns and hopefully, you don learn some interesting things about how to process natural text.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Challenge

Now wey you don analyze your dataset for sentiment, try use the strategies wey you don learn for this curriculum (like clustering, maybe?) to find patterns around sentiment.

## Review & Self Study

Take [this Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) to learn more and use different tools to explore sentiment for text.
## Assignment 

[Try a different dataset](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis docu don dey translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am accurate, abeg sabi say automated translations fit get mistake or no dey 100% correct. Di original docu for di language wey dem write am first na di main correct source. For important information, e better make una use professional human translation. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because of dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->