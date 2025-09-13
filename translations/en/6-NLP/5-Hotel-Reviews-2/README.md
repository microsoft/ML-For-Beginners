<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T11:03:14+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "en"
}
-->
# Sentiment analysis with hotel reviews

Now that you've explored the dataset in detail, it's time to filter the columns and apply NLP techniques to gain new insights about the hotels.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Filtering & Sentiment Analysis Operations

As you may have noticed, the dataset has some issues. Certain columns contain irrelevant information, while others seem inaccurate. Even if they are accurate, it's unclear how the values were calculated, making it impossible to verify them independently.

## Exercise: Additional Data Processing

Clean the data further by adding useful columns, modifying values in others, and removing unnecessary columns.

1. Initial column processing

   1. Remove `lat` and `lng`.

   2. Replace `Hotel_Address` values with the following format: if the address contains the city and country, simplify it to just the city and country.

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

      Now you can query data at the country level:

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

   1. Remove `Additional_Number_of_Scoring`.

   2. Replace `Total_Number_of_Reviews` with the actual total number of reviews for each hotel in the dataset.

   3. Replace `Average_Score` with a score calculated by you.

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Process review columns

   1. Remove `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date`, and `days_since_review`.

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are.

   3. Retain `Tags` for now.

      - Additional filtering will be applied to the tags in the next section, after which they will be removed.

4. Process reviewer columns

   1. Remove `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Retain `Reviewer_Nationality`.

### Tag Columns

The `Tag` column is problematic because it contains a list (in text form) stored in the column. The order and number of subsections in this column are inconsistent. With 515,000 rows and 1,427 hotels, each with slightly different options for reviewers, it's difficult for a human to identify the relevant phrases. This is where NLP excels. By scanning the text, you can identify the most common phrases and count their occurrences.

However, we are not interested in single words but rather multi-word phrases (e.g., *Business trip*). Running a multi-word frequency distribution algorithm on such a large dataset (6,762,646 words) could take a significant amount of time. Without analyzing the data, it might seem like a necessary step. But exploratory data analysis can help reduce the processing time. For example, based on a sample of tags like `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, you can determine whether it's possible to simplify the process. Fortunately, it is—but you'll need to follow a few steps to identify the relevant tags.

### Filtering Tags

The goal of the dataset is to add sentiment and columns that help you choose the best hotel (for yourself or for a client who needs a hotel recommendation bot). You need to decide which tags are useful for the final dataset. Here's one interpretation (though different goals might lead to different choices):

1. The type of trip is relevant and should be retained.
2. The type of guest group is important and should be retained.
3. The type of room, suite, or studio the guest stayed in is irrelevant (most hotels offer similar rooms).
4. The device used to submit the review is irrelevant.
5. The number of nights stayed *might* be relevant if longer stays indicate satisfaction, but it's probably not significant.

In summary, **keep two types of tags and discard the rest**.

First, you need to reformat the tags before counting them. This involves removing square brackets and quotes. There are several ways to do this, but you'll want the fastest method since processing a large dataset can be time-consuming. Fortunately, pandas provides an efficient way to handle these steps.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Each tag will look like this: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

Next, you'll encounter a problem: some reviews have 5 tags, others have 3, and some have 6. This inconsistency is due to how the dataset was created and is difficult to fix. While this could lead to inaccurate counts, you can use the varying order of tags to your advantage. Since each tag is multi-word and separated by commas, the simplest solution is to create 6 temporary columns, each containing one tag based on its position. You can then merge these columns into one and use the `value_counts()` method to count occurrences. This will reveal 2,428 unique tags. Here's a small sample:

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

Some tags, like `Submitted from a mobile device`, are irrelevant and can be ignored. However, since counting phrases is a fast operation, you can leave them in and disregard them later.

### Removing Length-of-Stay Tags

The first step is to remove length-of-stay tags from consideration. This slightly reduces the total number of tags. Note that you are not removing these tags from the dataset, just excluding them from the analysis.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed 2 nights  | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed 4 nights  | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed 6 nights  | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed 8 nights  | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

Similarly, there are many types of rooms, suites, and studios. These are all essentially the same and irrelevant to your analysis, so exclude them as well.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard Double Room          | 32248 |
| Superior Double Room          | 31393 |
| Deluxe Double Room            | 24823 |
| Double or Twin Room           | 22393 |
| Standard Double or Twin Room  | 17483 |
| Classic Double Room           | 16989 |
| Superior Double or Twin Room  | 13570 |

Finally, after minimal processing, you'll be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo traveler                                 | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family with older children                    | 26349  |
| With a pet                                    | 1405   |

You might combine `Travellers with friends` and `Group` into one category, as shown above. The code for identifying the correct tags can be found in [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. For every review row, if the `Tag` column matches one of the new columns, assign a value of 1; otherwise, assign 0. This will allow you to count how many reviewers chose a hotel for reasons like business, leisure, or bringing a pet—valuable information for hotel recommendations.

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

### Save Your File

Finally, save the updated dataset with a new name.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Sentiment Analysis Operations

In this final section, you'll apply sentiment analysis to the review columns and save the results in the dataset.

## Exercise: Load and Save the Filtered Data

Make sure to load the filtered dataset saved in the previous section, **not** the original dataset.

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

### Removing Stop Words

Running sentiment analysis on the Negative and Positive review columns can take a long time. On a high-performance test laptop, it took 12–14 minutes depending on the sentiment library used. This is relatively slow, so it's worth exploring ways to speed up the process.

Removing stop words—common English words that don't affect sentiment—is the first step. By eliminating these words, sentiment analysis should run faster without losing accuracy. Stop words don't influence sentiment but do slow down processing.

For example, the longest negative review in the dataset was 395 words. After removing stop words, it was reduced to 195 words.

Removing stop words is a quick operation. On the test device, it took 3.3 seconds to process the stop words in two review columns across 515,000 rows. The exact time may vary depending on your device's CPU, RAM, storage type (SSD or HDD), and other factors. Given the short processing time, it's worth doing if it speeds up sentiment analysis.

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

### Performing Sentiment Analysis

Now calculate sentiment scores for both the negative and positive review columns, storing the results in two new columns. You can test the sentiment analysis by comparing the sentiment scores to the reviewer's score for the same review. For example, if the sentiment analysis assigns a positive score to both the negative and positive reviews, but the reviewer gave the hotel the lowest possible score, there may be a mismatch between the review text and the score. Alternatively, the sentiment analyzer might have misinterpreted the sentiment.

Some sentiment scores will inevitably be incorrect. For instance, sarcasm in a review—e.g., "Of course I LOVED sleeping in a room with no heating"—might be interpreted as positive sentiment by the analyzer, even though a human reader would recognize the sarcasm.
NLTK offers various sentiment analyzers to experiment with, allowing you to swap them out and assess whether the sentiment analysis becomes more or less accurate. In this example, the VADER sentiment analysis tool is utilized.

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

Later in your program, when you're ready to calculate sentiment, you can apply it to each review as shown below:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

On my computer, this process takes roughly 120 seconds, though the time may vary depending on the machine. If you'd like to print the results and check whether the sentiment aligns with the review:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

The final step before using the file in the challenge is to save it! Additionally, you might want to reorder all your new columns to make them more user-friendly (this is purely a cosmetic adjustment for easier handling).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

You should execute the complete code from [the analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (after running [the filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) to generate the Hotel_Reviews_Filtered.csv file).

To summarize, the steps are:

1. The original dataset file **Hotel_Reviews.csv** is explored in the previous lesson using [the explorer notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb).
2. **Hotel_Reviews.csv** is filtered using [the filtering notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), resulting in **Hotel_Reviews_Filtered.csv**.
3. **Hotel_Reviews_Filtered.csv** is processed using [the sentiment analysis notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), producing **Hotel_Reviews_NLP.csv**.
4. Use **Hotel_Reviews_NLP.csv** in the NLP Challenge below.

### Conclusion

At the beginning, you had a dataset with columns and data, but not all of it was usable or verifiable. You've explored the data, filtered out unnecessary parts, transformed tags into meaningful information, calculated averages, added sentiment columns, and hopefully gained valuable insights into processing natural text.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Challenge

Now that your dataset has been analyzed for sentiment, try applying strategies you've learned in this curriculum (such as clustering) to identify patterns related to sentiment.

## Review & Self Study

Take [this Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) to deepen your understanding and explore sentiment analysis using different tools.

## Assignment

[Experiment with a different dataset](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we aim for accuracy, please note that automated translations may include errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is advised. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.