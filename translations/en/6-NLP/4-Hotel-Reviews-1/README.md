<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-06T11:01:06+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "en"
}
-->
# Sentiment analysis with hotel reviews - processing the data

In this section, you'll apply techniques from previous lessons to perform exploratory data analysis on a large dataset. Once you understand the relevance of the various columns, you'll learn:

- How to remove unnecessary columns
- How to calculate new data based on existing columns
- How to save the resulting dataset for use in the final challenge

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

Up to this point, you've learned that text data is quite different from numerical data. If the text is written or spoken by humans, it can be analyzed to uncover patterns, frequencies, sentiment, and meaning. This lesson introduces you to a real dataset with a real challenge: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, which is licensed under [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). The dataset was scraped from Booking.com using public sources and created by Jiashen Liu.

### Preparation

You will need:

- The ability to run .ipynb notebooks using Python 3
- pandas
- NLTK, [which you should install locally](https://www.nltk.org/install.html)
- The dataset, available on Kaggle: [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). It is approximately 230 MB unzipped. Download it to the root `/data` folder associated with these NLP lessons.

## Exploratory data analysis

This challenge assumes you're building a hotel recommendation bot using sentiment analysis and guest review scores. The dataset includes reviews of 1493 hotels across 6 cities.

Using Python, the hotel reviews dataset, and NLTK's sentiment analysis, you could explore:

- What are the most frequently used words and phrases in reviews?
- Do the official *tags* describing a hotel correlate with review scores (e.g., are negative reviews more common for *Family with young children* than for *Solo traveler*, possibly indicating the hotel is better suited for *Solo travelers*?)
- Do the NLTK sentiment scores align with the numerical scores given by hotel reviewers?

#### Dataset

Let's examine the dataset you've downloaded and saved locally. Open the file in an editor like VS Code or Excel.

The dataset headers are as follows:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Here they are grouped for easier analysis:

##### Hotel columns

- `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  - Using *lat* and *lng*, you could plot a map with Python showing hotel locations, perhaps color-coded for negative and positive reviews.
  - `Hotel_Address` is not particularly useful and could be replaced with a country for easier sorting and searching.

**Hotel Meta-review columns**

- `Average_Score`
  - According to the dataset creator, this column represents the *Average Score of the hotel, calculated based on the latest comment in the last year*. While this calculation method seems unusual, we'll take it at face value for now.
  
  ✅ Can you think of another way to calculate the average score using the other columns in this dataset?

- `Total_Number_of_Reviews`
  - The total number of reviews the hotel has received. It's unclear (without writing code) whether this refers to the reviews in the dataset.
- `Additional_Number_of_Scoring`
  - Indicates a review score was given without a positive or negative review being written.

**Review columns**

- `Reviewer_Score`
  - A numerical value with at most one decimal place, ranging between 2.5 and 10.
  - It's unclear why 2.5 is the lowest possible score.
- `Negative_Review`
  - If no negative review was written, this field contains "**No Negative**."
  - Sometimes, reviewers write positive comments in the Negative_Review column (e.g., "there is nothing bad about this hotel").
- `Review_Total_Negative_Word_Counts`
  - Higher negative word counts generally indicate a lower score (without sentiment analysis).
- `Positive_Review`
  - If no positive review was written, this field contains "**No Positive**."
  - Sometimes, reviewers write negative comments in the Positive_Review column (e.g., "there is nothing good about this hotel at all").
- `Review_Total_Positive_Word_Counts`
  - Higher positive word counts generally indicate a higher score (without sentiment analysis).
- `Review_Date` and `days_since_review`
  - You could apply a freshness or staleness measure to reviews (e.g., older reviews might be less accurate due to changes in hotel management, renovations, or new amenities like a pool).
- `Tags`
  - Short descriptors selected by reviewers to describe their type of stay (e.g., solo or family), room type, length of stay, and how the review was submitted.
  - Unfortunately, these tags are problematic; see the section below discussing their usefulness.

**Reviewer columns**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - This could be a factor in a recommendation model. For example, prolific reviewers with hundreds of reviews might tend to be more negative. However, reviewers are not identified with unique codes, so their reviews cannot be linked. There are 30 reviewers with 100 or more reviews, but it's unclear how this could aid the recommendation model.
- `Reviewer_Nationality`
  - Some might assume certain nationalities are more likely to give positive or negative reviews due to cultural tendencies. Be cautious about incorporating such anecdotal views into your models. These are stereotypes, and each reviewer is an individual whose review reflects their personal experience, filtered through factors like previous hotel stays, travel distance, and temperament. It's hard to justify attributing review scores solely to nationality.

##### Examples

| Average Score | Total Number Reviews | Reviewer Score | Negative Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive Review                 | Tags                                                                                      |
| -------------- | -------------------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                 | 2.5            | This is currently not a hotel but a construction site. I was terrorized from early morning and all day with unacceptable building noise while resting after a long trip and working in the room. People were working all day, i.e., with jackhammers in the adjacent rooms. I asked for a room change, but no silent room was available. To make things worse, I was overcharged. I checked out in the evening since I had to leave for an early flight and received an appropriate bill. A day later, the hotel made another charge without my consent in excess of the booked price. It's a terrible place. Don't punish yourself by booking here. | Nothing. Terrible place. Stay away. | Business trip, Couple, Standard Double Room, Stayed 2 nights |

As you can see, this guest had a very negative experience. The hotel has a decent average score of 7.8 and 1945 reviews, but this reviewer gave it a 2.5 and wrote 115 words about their dissatisfaction. They wrote only 7 words in the Positive_Review column, warning others to avoid the hotel. If we only counted words instead of analyzing their sentiment, we might misinterpret the reviewer's intent. Interestingly, the lowest possible score is 2.5, not 0, which raises questions about the scoring system.

##### Tags

At first glance, using `Tags` to categorize data seems logical. However, these tags are not standardized. For example, one hotel might use *Single room*, *Twin room*, and *Double room*, while another uses *Deluxe Single Room*, *Classic Queen Room*, and *Executive King Room*. These might refer to the same types of rooms, but the variations make standardization difficult.

Options for handling this:

1. Attempt to standardize all terms, which is challenging due to unclear mappings (e.g., *Classic single room* maps to *Single room*, but *Superior Queen Room with Courtyard Garden or City View* is harder to map).
2. Use an NLP approach to measure the frequency of terms like *Solo*, *Business Traveller*, or *Family with young kids* and factor them into the recommendation.

Tags usually contain 5-6 comma-separated values, including *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, and *Type of device used to submit the review*. However, since some reviewers leave fields blank, the values are not always in the same order.

For example, filtering for *Family with* yields over 80,000 results containing phrases like "Family with young children" or "Family with older children." This shows the `Tags` column has potential but requires effort to make it useful.

##### Average hotel score

There are some oddities in the dataset that are worth noting:

The dataset includes the following columns related to average scores and reviews:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

The hotel with the most reviews in the dataset is *Britannia International Hotel Canary Wharf*, with 4789 reviews out of 515,000. However, its `Total_Number_of_Reviews` value is 9086. Adding the `Additional_Number_of_Scoring` value (2682) to 4789 gives 7471, which is still 1615 short of the `Total_Number_of_Reviews`.

The `Average_Score` column description from Kaggle states it is "*Average Score of the hotel, calculated based on the latest comment in the last year*." This calculation method seems unhelpful, but we can calculate our own average based on the review scores in the dataset. For example, the average score for this hotel is listed as 7.1, but the calculated average (based on reviewer scores in the dataset) is 6.8. This discrepancy might be due to scores from `Additional_Number_of_Scoring` reviews, but there's no way to confirm this.

To further complicate matters, the hotel with the second-highest number of reviews has a calculated average score of 8.12, while the dataset lists it as 8.1. Is this coincidence or a discrepancy?

Given these inconsistencies, we'll write a short program to explore the dataset and determine the best way to use (or not use) these values.
> 🚨 A note of caution
>
> When working with this dataset, you'll be writing code to calculate something based on the text without needing to read or analyze the text yourself. This is the core of NLP—interpreting meaning or sentiment without requiring human intervention. However, you might come across some negative reviews. I strongly recommend avoiding reading them, as it's unnecessary. Some of these reviews are trivial or irrelevant, like "The weather wasn't great," which is beyond the hotel's control—or anyone's, for that matter. But there is also a darker side to some reviews. Occasionally, negative reviews may contain racist, sexist, or ageist remarks. This is unfortunate but not surprising, given that the dataset is scraped from a public website. Some reviewers leave comments that you might find offensive, uncomfortable, or upsetting. It's better to let the code assess the sentiment rather than exposing yourself to potentially distressing content. That said, such reviews are written by a minority, but they do exist nonetheless.
## Exercise - Data exploration
### Load the data

Enough of visually examining the data—now it's time to write some code and get answers! This section uses the pandas library. Your first task is to ensure you can load and read the CSV data. The pandas library has a fast CSV loader, and the result is stored in a dataframe, as you've seen in previous lessons. The CSV file we are loading contains over half a million rows but only 17 columns. Pandas provides many powerful ways to interact with a dataframe, including the ability to perform operations on every row.

From this point onward in the lesson, you'll encounter code snippets, explanations of the code, and discussions about the results. Use the included _notebook.ipynb_ for your code.

Let's start by loading the data file you'll be working with:

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

Now that the data is loaded, you can perform some operations on it. Keep this code at the top of your program for the next part.

## Explore the data

In this case, the data is already *clean*, meaning it is ready to work with and does not contain characters in other languages that might confuse algorithms expecting only English characters.

✅ You might encounter data that requires initial processing to format it before applying NLP techniques, but not this time. If you had to, how would you handle non-English characters?

Take a moment to ensure that once the data is loaded, you can explore it with code. It's tempting to focus on the `Negative_Review` and `Positive_Review` columns, as they contain natural text for your NLP algorithms to process. But wait! Before diving into NLP and sentiment analysis, follow the code below to verify whether the values in the dataset match the values you calculate using pandas.

## Dataframe operations

Your first task in this lesson is to check whether the following assertions are correct by writing code to examine the dataframe (without modifying it).

> As with many programming tasks, there are multiple ways to approach this, but a good rule of thumb is to choose the simplest and easiest method, especially if it will be easier to understand when revisiting the code later. With dataframes, the comprehensive API often provides efficient ways to achieve your goals.

Treat the following questions as coding tasks and try to answer them without looking at the solution.

1. Print the *shape* of the dataframe you just loaded (the shape refers to the number of rows and columns).
2. Calculate the frequency count for reviewer nationalities:
   1. How many distinct values exist in the `Reviewer_Nationality` column, and what are they?
   2. Which reviewer nationality is the most common in the dataset (print the country and the number of reviews)?
   3. What are the next top 10 most frequently found nationalities, along with their frequency counts?
3. What is the most frequently reviewed hotel for each of the top 10 reviewer nationalities?
4. How many reviews are there per hotel (frequency count of hotels) in the dataset?
5. While the dataset includes an `Average_Score` column for each hotel, you can also calculate an average score (by averaging all reviewer scores in the dataset for each hotel). Add a new column to your dataframe called `Calc_Average_Score` that contains this calculated average.
6. Do any hotels have the same (rounded to 1 decimal place) `Average_Score` and `Calc_Average_Score`?
   1. Try writing a Python function that takes a Series (row) as an argument and compares the values, printing a message when the values are not equal. Then use the `.apply()` method to process every row with the function.
7. Calculate and print how many rows have `Negative_Review` values of "No Negative."
8. Calculate and print how many rows have `Positive_Review` values of "No Positive."
9. Calculate and print how many rows have `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative."

### Code answers

1. Print the *shape* of the dataframe you just loaded (the shape refers to the number of rows and columns).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calculate the frequency count for reviewer nationalities:

   1. How many distinct values exist in the `Reviewer_Nationality` column, and what are they?
   2. Which reviewer nationality is the most common in the dataset (print the country and the number of reviews)?

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

   3. What are the next top 10 most frequently found nationalities, along with their frequency counts?

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

3. What is the most frequently reviewed hotel for each of the top 10 reviewer nationalities?

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

4. How many reviews are there per hotel (frequency count of hotels) in the dataset?

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
   
   You may notice that the *counted in the dataset* results do not match the value in `Total_Number_of_Reviews`. It is unclear whether this value represents the total number of reviews the hotel received but not all were scraped, or some other calculation. `Total_Number_of_Reviews` is not used in the model due to this ambiguity.

5. While the dataset includes an `Average_Score` column for each hotel, you can also calculate an average score (by averaging all reviewer scores in the dataset for each hotel). Add a new column to your dataframe called `Calc_Average_Score` that contains this calculated average. Print the columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`.

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

   You may also wonder about the `Average_Score` value and why it sometimes differs from the calculated average score. Since we can't determine why some values match while others differ, it's safest to use the review scores we have to calculate the average ourselves. That said, the differences are usually very small. Here are the hotels with the greatest deviation between the dataset average and the calculated average:

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

   With only one hotel having a score difference greater than 1, we can likely ignore the difference and use the calculated average score.

6. Calculate and print how many rows have `Negative_Review` values of "No Negative."

7. Calculate and print how many rows have `Positive_Review` values of "No Positive."

8. Calculate and print how many rows have `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative."

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

## Another way

Another way to count items without using Lambdas is to use the sum function to count rows:

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

   You may have noticed that there are 127 rows with both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review`, respectively. This means the reviewer gave the hotel a numerical score but chose not to write either a positive or negative review. Fortunately, this is a small number of rows (127 out of 515,738, or 0.02%), so it likely won't skew the model or results significantly. However, you might not have expected a dataset of reviews to include rows without any reviews, so it's worth exploring the data to uncover such anomalies.

Now that you've explored the dataset, the next lesson will focus on filtering the data and adding sentiment analysis.

---
## 🚀Challenge

This lesson demonstrates, as we've seen in previous lessons, how critically important it is to understand your data and its quirks before performing operations on it. Text-based data, in particular, requires careful scrutiny. Explore various text-heavy datasets and see if you can identify areas that might introduce bias or skewed sentiment into a model.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models.

## Assignment 

[NLTK](assignment.md)

---

**Disclaimer**:  
This document has been translated using the AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). While we strive for accuracy, please note that automated translations may contain errors or inaccuracies. The original document in its native language should be regarded as the authoritative source. For critical information, professional human translation is recommended. We are not responsible for any misunderstandings or misinterpretations resulting from the use of this translation.