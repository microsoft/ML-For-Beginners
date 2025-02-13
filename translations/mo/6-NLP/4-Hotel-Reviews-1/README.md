# Sentiment analysis with hotel reviews - processing the data

In this section, you will apply the techniques from previous lessons to perform exploratory data analysis on a large dataset. Once you grasp the significance of the various columns, you will learn:

- how to eliminate unnecessary columns
- how to compute new data based on existing columns
- how to save the resulting dataset for the final challenge

## [Pre-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introduction

So far, you've learned that text data differs significantly from numerical data types. If it's text written or spoken by a human, it can be analyzed to uncover patterns, frequencies, sentiments, and meanings. This lesson introduces you to a real dataset with a real challenge: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, which comes with a [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). The data was scraped from Booking.com from public sources, and the dataset was created by Jiashen Liu.

### Preparation

You will need:

* The ability to run .ipynb notebooks using Python 3
* pandas
* NLTK, [which you should install locally](https://www.nltk.org/install.html)
* The dataset available on Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). It is approximately 230 MB when unzipped. Download it to the root `/data` folder associated with these NLP lessons.

## Exploratory data analysis

This challenge assumes that you are developing a hotel recommendation bot using sentiment analysis and guest review scores. The dataset you will be using consists of reviews from 1493 different hotels in 6 cities.

Using Python, a dataset of hotel reviews, and NLTK's sentiment analysis, you could discover:

* What are the most frequently used words and phrases in reviews?
* Do the official *tags* describing a hotel correlate with review scores (e.g., are the more negative reviews for a particular hotel from *Families with young children* rather than *Solo travelers*, possibly indicating it is better suited for *Solo travelers*?)
* Do the NLTK sentiment scores 'agree' with the numerical scores given by hotel reviewers?

#### Dataset

Let's explore the dataset you've downloaded and saved locally. Open the file in an editor like VS Code or even Excel.

The headers in the dataset are as follows:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Here they are grouped in a way that might be easier to examine: 
##### Hotel columns

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Using *lat* and *lng*, you could plot a map with Python showing the hotel locations (perhaps color-coded for negative and positive reviews).
  * Hotel_Address may not be very useful to us, and we will likely replace it with a country for easier sorting & searching.

**Hotel Meta-review columns**

* `Average_Score`
  * According to the dataset creator, this column represents the *Average Score of the hotel, calculated based on the latest comment in the last year*. This seems like an unusual method to calculate the score, but it is the data scraped, so we may take it at face value for now. 
  
  ✅ Based on the other columns in this data, can you think of another way to calculate the average score?

* `Total_Number_of_Reviews`
  * This indicates the total number of reviews this hotel has received - it is not clear (without writing some code) if this refers to the reviews in the dataset.
* `Additional_Number_of_Scoring`
  * This means a review score was given, but no positive or negative review was written by the reviewer.

**Review columns**

- `Reviewer_Score`
  - This is a numerical value with at most 1 decimal place between the min and max values of 2.5 and 10.
  - It is not explained why 2.5 is the lowest score possible.
- `Negative_Review`
  - If a reviewer wrote nothing, this field will show "**No Negative**".
  - Note that a reviewer may write a positive review in the Negative review column (e.g., "there is nothing bad about this hotel").
- `Review_Total_Negative_Word_Counts`
  - Higher negative word counts indicate a lower score (without checking the sentimentality).
- `Positive_Review`
  - If a reviewer wrote nothing, this field will show "**No Positive**".
  - Note that a reviewer may write a negative review in the Positive review column (e.g., "there is nothing good about this hotel at all").
- `Review_Total_Positive_Word_Counts`
  - Higher positive word counts indicate a higher score (without checking the sentimentality).
- `Review_Date` and `days_since_review`
  - A freshness or staleness measure might be applied to a review (older reviews might not be as accurate as newer ones because hotel management changed, renovations have been made, or a pool was added, etc.).
- `Tags`
  - These are short descriptors that a reviewer may select to describe the type of guest they were (e.g., solo or family), the type of room they had, the length of stay, and how the review was submitted. 
  - Unfortunately, using these tags is problematic; check the section below which discusses their usefulness.

**Reviewer columns**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - This might be a factor in a recommendation model, for instance, if you could determine that more prolific reviewers with hundreds of reviews were more likely to be negative rather than positive. However, the reviewer of any particular review is not identified with a unique code, and therefore cannot be linked to a set of reviews. There are 30 reviewers with 100 or more reviews, but it's hard to see how this can aid the recommendation model.
- `Reviewer_Nationality`
  - Some people might think that certain nationalities are more likely to give a positive or negative review because of a national inclination. Be cautious about incorporating such anecdotal views into your models. These are national (and sometimes racial) stereotypes, and each reviewer was an individual who wrote a review based on their experience. Their review might have been influenced by various factors such as previous hotel stays, the distance traveled, and their personal temperament. Assuming their nationality was the reason for a review score is hard to justify.

##### Examples

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is currently not a hotel but a construction site. I was terrorized from early morning and all day with unacceptable building noise while resting after a long trip and working in the room. People were working all day with jackhammers in the adjacent rooms. I asked for a room change, but no silent room was available. To make things worse, I was overcharged. I checked out in the evening since I had to leave for a very early flight and received an appropriate bill. A day later, the hotel made another charge without my consent in excess of the booked price. It's a terrible place. Don't punish yourself by booking here. | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double Room Stayed 2 nights |

As you can see, this guest did not have a pleasant stay at this hotel. The hotel has a good average score of 7.8 and 1945 reviews, but this reviewer gave it 2.5 and wrote 115 words about how negative their stay was. If they wrote nothing at all in the Positive_Review column, you might assume there was nothing positive, but alas, they wrote 7 words of warning. If we just counted words instead of the meaning or sentiment of the words, we might have a skewed view of the reviewer's intent. Strangely, their score of 2.5 is perplexing because if that hotel stay was so bad, why give it any points at all? Investigating the dataset closely, you'll see that the lowest possible score is 2.5, not 0. The highest possible score is 10.

##### Tags

As mentioned above, at first glance, the idea of using `Tags` to categorize the data makes sense. Unfortunately, these tags are not standardized, which means that in a given hotel, the options might be *Single room*, *Twin room*, and *Double room*, but in the next hotel, they might be *Deluxe Single Room*, *Classic Queen Room*, and *Executive King Room*. These might be the same things, but there are so many variations that the choice becomes:

1. Attempt to change all terms to a single standard, which is very difficult because it is not clear what the conversion path would be in each case (e.g., *Classic single room* maps to *Single room*, but *Superior Queen Room with Courtyard Garden or City View* is much harder to map).

2. We can take an NLP approach and measure the frequency of certain terms like *Solo*, *Business Traveller*, or *Family with young kids* as they apply to each hotel, and factor that into the recommendation.  

Tags are usually (but not always) a single field containing a list of 5 to 6 comma-separated values aligning to *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, and *Type of device review was submitted on*. However, because some reviewers don't fill in each field (they might leave one blank), the values are not always in the same order.

As an example, take *Type of group*. There are 1025 unique possibilities in this field in the `Tags` column, and unfortunately, only some of them refer to a group (some are the type of room, etc.). If you filter only the ones that mention family, the results contain many *Family room* type results. If you include the term *with*, i.e., count the *Family with* values, the results are better, with over 80,000 of the 515,000 results containing the phrase "Family with young children" or "Family with older children".

This means the tags column is not entirely useless to us, but it will take some work to make it useful.

##### Average hotel score

There are a number of oddities or discrepancies with the dataset that I can't figure out, but are illustrated here so you are aware of them when building your models. If you figure it out, please let us know in the discussion section!

The dataset has the following columns relating to the average score and number of reviews: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

The single hotel with the most reviews in this dataset is *Britannia International Hotel Canary Wharf* with 4789 reviews out of 515,000. But if we look at the `Total_Number_of_Reviews` value for this hotel, it is 9086. You might surmise that there are many more scores without reviews, so perhaps we should add in the `Additional_Number_of_Scoring` column value. That value is 2682, and adding it to 4789 gets us 7,471, which is still 1615 short of the `Total_Number_of_Reviews`. 

If you take the `Average_Score` columns, you might think it is the average of the reviews in the dataset, but the description from Kaggle is "*Average Score of the hotel, calculated based on the latest comment in the last year*". That doesn't seem very useful, but we can calculate our own average based on the reviewer scores in the dataset. Using the same hotel as an example, the average hotel score is given as 7.1, but the calculated score (average reviewer score *in* the dataset) is 6.8. This is close but not the same value, and we can only guess that the scores given in the `Additional_Number_of_Scoring` reviews increased the average to 7.1. Unfortunately, with no way to test or prove that assertion, it is difficult to use or trust `Average_Score`, `Additional_Number_of_Scoring`, and `Total_Number_of_Reviews` when they are based on, or refer to, data we do not have.

To complicate things further, the hotel with the second highest number of reviews has a calculated average score of 8.12, and the dataset `Average_Score` is 8.1. Is this correct score a coincidence, or is the first hotel a discrepancy? 

On the possibility that these hotels might be outliers, and that maybe most of the values tally up (but some do not for some reason), we will write a short program next to explore the values in the dataset and determine the correct usage (or non-usage) of the values.

> 🚨 A note of caution
>
> When working with this dataset, you will write code that calculates something from the text without having to read or analyze the text yourself. This is the essence of NLP, interpreting meaning or sentiment without requiring human intervention. However, it is possible that you will encounter some negative reviews. I would advise against reading them, as you don't have to. Some of them are trivial, or irrelevant negative hotel reviews, such as "The weather wasn't great," something beyond the control of the hotel, or indeed, anyone. But there is a darker side to some reviews too. Sometimes, negative reviews are racist, sexist, or ageist. This is unfortunate but to be expected in a dataset scraped from a public website. Some reviewers leave comments that you might find distasteful, uncomfortable, or upsetting. It is better to let the code measure the sentiment than to read them yourself and be distressed. That said, it is a minority that write such things, but they exist nonetheless.

## Exercise - Data exploration
### Load the data

That's enough visual examination of the data; now you'll write some code to get some answers! This section uses the pandas library. Your very first task is to ensure you can load and read the CSV data. The pandas library has a fast CSV loader, and the result is placed in a dataframe, as in previous lessons. The CSV we are loading has over half a million rows, but only 17 columns. Pandas provides many powerful ways to interact with a dataframe, including the ability to perform operations on every row.

From here on in this lesson, there will be code snippets and some explanations of the code, along with discussions about what the results mean. Use the included _notebook.ipynb_ for your code.

Let's start by loading the data file you will be using:

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

Now that the data is loaded, we can perform some operations on it. Keep this code at the top of your program for the next part.

## Explore the data

In this case, the data is already *clean*, meaning that it is ready to work with and does not contain characters from other languages that might confuse algorithms expecting only English characters.

✅ You might have to work with data that requires some initial processing to format it before applying NLP techniques, but not this time. If you had to, how would you handle non-English characters?

Take a moment to ensure that once the data is loaded, you can explore it with code. It's very tempting to focus on the `Negative_Review` and `Positive_Review` columns. They are filled with natural text for your NLP algorithms to process. But wait! Before you dive into the NLP and sentiment analysis, you should follow the code below to check if the values given in the dataset match the values you calculate with pandas.

## Dataframe operations

The first task in this lesson is to verify if the following assertions are correct by writing some code that examines the dataframe (without altering it).

> Like many programming tasks, there are several ways to complete this, but a good practice is to do it in the simplest, most straightforward way you can, especially if it will be easier to understand when you revisit this code in the future. With dataframes, there is a comprehensive API that will often have a way to achieve what you want efficiently.
Treat the following questions as coding tasks and attempt to answer them without looking at the solution. 1. Print out the *shape* of the dataframe you have just loaded (the shape is the number of rows and columns) 2. Calculate the frequency count for reviewer nationalities: 1. How many distinct values are there for the column `Reviewer_Nationality` and what are they? 2. What reviewer nationality is the most common in the dataset (print country and number of reviews)? 3. What are the next top 10 most frequently found nationalities, and their frequency count? 3. What was the most frequently reviewed hotel for each of the top 10 most reviewer nationalities? 4. How many reviews are there per hotel (frequency count of hotel) in the dataset? 5. While there is an `Average_Score` column for each hotel in the dataset, you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel). Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. 6. Do any hotels have the same (rounded to 1 decimal place) `Average_Score` and `Calc_Average_Score`?
   1. Try writing a Python function that takes a Series (row) as an argument and compares the values, printing out a message when the values are not equal. Then use the `.apply()` method to process every row with the function. 7. Calculate and print out how many rows have column `Negative_Review` values of "No Negative" 8. Calculate and print out how many
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
   ``` You may have noticed that there are 127 rows that have both "No Negative" and "No Positive" values for the columns `Negative_Review` and `Positive_Review` respectively. That means that the reviewer gave the hotel a numerical score, but declined to write either a positive or negative review. Luckily this is a small amount of rows (127 out of 515738, or 0.02%), so it probably won't skew our model or results in any particular direction, but you might not have expected a data set of reviews to have rows with no reviews, so it's worth exploring the data to discover rows like this. Now that you have explored the dataset, in the next lesson you will filter the data and add some sentiment analysis. --- ## 🚀Challenge This lesson demonstrates, as we saw in previous lessons, how critically important it is to understand your data and its foibles before performing operations on it. Text-based data, in particular, bears careful scrutiny. Dig through various text-heavy datasets and see if you can discover areas that could introduce bias or skewed sentiment into a model. ## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Review & Self Study Take [this Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to discover tools to try when building speech and text-heavy models. ## Assignment [NLTK](assignment.md) Please write the output from left to right.

I'm sorry, but I can't provide a translation to "mo" as it is not a recognized language code. If you meant a specific language, please clarify which language you'd like the text translated into, and I'd be happy to help!