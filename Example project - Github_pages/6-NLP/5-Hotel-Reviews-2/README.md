# Sentiment analysis with hotel reviews

Now that you have explored the dataset in detail, it's time to filter the columns and then use NLP techniques on the dataset to gain new insights about the hotels.
## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Filtering & Sentiment Analysis Operations

As you've probably noticed, the dataset has a few issues. Some columns are filled with useless information, others seem incorrect. If they are correct, it's unclear how they were calculated, and answers cannot be independently verified by your own calculations.

## Exercise: a bit more data processing

Clean the data just a bit more. Add columns that will be useful later, change the values in other columns, and drop certain columns completely.

1. Initial column processing

   1. Drop `lat` and `lng`

   2. Replace `Hotel_Address` values with the following values (if the address contains the same of the city and the country, change it to just the city and the country).

      These are the only cities and countries in the dataset:

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

      Now you can query country level data:

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

  1. Drop `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` with our own calculated score

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Process review columns

   1. Drop `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` , you can begin to ask if it's possible to greatly reduce the processing you have to do. Luckily, it is - but first you need to follow a few steps to ascertain the tags of interest.

### Filtering tags

Remember that the goal of the dataset is to add sentiment and columns that will help you choose the best hotel (for yourself or maybe a client tasking you to make a hotel recommendation bot). You need to ask yourself if the tags are useful or not in the final dataset. Here is one interpretation (if you needed the dataset for other reasons different tags might stay in/out of the selection):

1. The type of trip is relevant, and that should stay
2. The type of guest group is important, and that should stay
3. The type of room, suite, or studio that the guest stayed in is irrelevant (all hotels have basically the same rooms)
4. The device the review was submitted on is irrelevant
5. The number of nights reviewer stayed for *could* be relevant if you attributed longer stays with them liking the hotel more, but it's a stretch, and probably irrelevant

In summary, **keep 2 kinds of tags and remove the others**.

First, you don't want to count the tags until they are in a better format, so that means removing the square brackets and quotes. You can do this several ways, but you want the fastest as it could take a long time to process a lot of data. Luckily, pandas has an easy way to do each of these steps.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Each tag becomes something like: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

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

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` column matches one of the new columns, add a 1, if not, add a 0. The end result will be a count of how many reviewers chose this hotel (in aggregate) for, say, business vs leisure, or to bring a pet to, and this is useful information when recommending a hotel.

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

Finally, save the dataset as it is now with a new name.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentiment Analysis Operations

In this final section, you will apply sentiment analysis to the review columns and save the results in a dataset.

## Exercise: load and save the filtered data

Note that now you are loading the filtered dataset that was saved in the previous section, **not** the original dataset.

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

If you were to run Sentiment Analysis on the Negative and Positive review columns, it could take a long time. Tested on a powerful test laptop with fast CPU,it took 12 - 14 minutes depending on which sentiment library was used. That's a (relatively) long time, so worth investigating if that can be speeded up. 

Removing stop words, or common English words that do not change the sentiment of a sentence, is the first step. By removing them, the sentiment analysis should run faster, but not be less accurate (as the stop words do not affect sentiment, but they do slow down the analysis). 

The longest negative review was 395 words, but after removing the stop words, it is 195 words.

Removing the stop words is also a fast operation, removing the stop words from 2 review columns over 515,000 rows took 3.3 seconds on the test device. It could take slightly more or less time for you depending on your device CPU speed, RAM, whether you have an SSD or not, and some other factors. The relative shortness of the operation means that if it improves the sentiment analysis time, then it is worth doing.

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

Now you should calculate the sentiment analysis for both negative and positive review columns, and store the result in 2 new columns. The test of the sentiment will be to compare it to the reviewer's score for the same review. For instance, if the sentiment thinks the negative review had a sentiment of 1 (extremely positive sentiment) and a positive review sentiment of 1, but the reviewer gave the hotel the lowest score possible, then either the review text doesn't match the score, or the sentiment analyser could not recognize the sentiment correctly. You should expect some sentiment scores to be completely wrong, and often that will be explainable, e.g. the review could be extremely sarcastic "Of course I LOVED sleeping in a room with no heating" and the sentiment analyser thinks that's positive sentiment, even though a human reading it would know it was sarcasm. 

NLTK supplies different sentiment analyzers to learn with, and you can substitute them and see if the sentiment is more or less accurate. The VADER sentiment analysis is used here.

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

Later in your program when you are ready to calculate sentiment, you can apply it to each review as follows:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

This takes approximately 120 seconds on my computer, but it will vary on each computer. If you want to print of the results and see if the sentiment matches the review:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

The very last thing to do with the file before using it in the challenge, is to save it! You should also consider reordering all your new columns so they are easy to work with (for a human, it's a cosmetic change).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

You should run the entire code for [the analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (after you've run [your filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) to generate the Hotel_Reviews_Filtered.csv file).

To review, the steps are:

1. Original dataset file **Hotel_Reviews.csv** is explored in the previous lesson with [the explorer notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv is filtered by [the filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) resulting in **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv is processed by [the sentiment analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) resulting in **Hotel_Reviews_NLP.csv**
4. Use Hotel_Reviews_NLP.csv in the NLP Challenge below

### Conclusion

When you started, you had a dataset with columns and data but not all of it could be verified or used. You've explored the data, filtered out what you don't need, converted tags into something useful, calculated your own averages, added some sentiment columns and hopefully, learned some interesting things about processing natural text.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Challenge

Now that you have your dataset analyzed for sentiment, see if you can use strategies you've learned in this curriculum (clustering, perhaps?) to determine patterns around sentiment. 

## Review & Self Study

Take [this Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) to learn more and use different tools to explore sentiment in text.
## Assignment 

[Try a different dataset](assignment.md)
