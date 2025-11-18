<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-11-18T18:32:19+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "pcm"
}
-->
# Sentiment analysis wit hotel reviews - how to process di data

For dis section, you go use di techniques wey you don learn for di previous lessons to do some exploratory data analysis for one big dataset. Once you don sabi well well how di different columns fit help, you go learn:

- how to remove di columns wey no dey necessary
- how to calculate new data based on di columns wey dey already
- how to save di dataset wey you don process so you fit use am for di final challenge

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Introduction

So far, you don learn say text data no be like di numerical type of data. If na text wey human being write or talk, you fit analyse am to find patterns, frequency, sentiment, and meaning. Dis lesson go carry you enter real dataset wey get real challenge: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** wey get [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/). Dem scrape am from Booking.com from public sources. Di person wey create di dataset na Jiashen Liu.

### Preparation

You go need:

* Di ability to run .ipynb notebooks wit Python 3
* pandas
* NLTK, [wey you go install for your computer](https://www.nltk.org/install.html)
* Di dataset wey dey available for Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). E dey around 230 MB when you unzip am. Download am go di root `/data` folder wey dey follow dis NLP lessons.

## Exploratory data analysis

Dis challenge dey assume say you dey build hotel recommendation bot wey go use sentiment analysis and guest review scores. Di dataset wey you go use get reviews for 1493 different hotels for 6 cities.

Using Python, hotel reviews dataset, and NLTK sentiment analysis, you fit find:

* Wetin be di most common words and phrases wey dey reviews?
* Di official *tags* wey dey describe hotel, e dey match di review scores? (e.g. negative reviews dey plenty for *Family wit young children* pass *Solo traveller*, wey fit mean say di hotel better for *Solo travellers*?)
* Di NLTK sentiment scores dey 'agree' wit di numerical score wey di hotel reviewer give?

#### Dataset

Make we explore di dataset wey you don download and save. Open di file for editor like VS Code or even Excel.

Di headers for di dataset be like dis:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Make we group dem so e go dey easier to check: 
##### Hotel columns

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Wit *lat* and *lng*, you fit use Python plot map wey go show di hotel locations (maybe color coded for negative and positive reviews)
  * Hotel_Address no dey useful like dat, we fit replace am wit country so e go dey easier to sort and search

**Hotel Meta-review columns**

* `Average_Score`
  * Di dataset creator talk say dis column na di *Average Score of di hotel, wey dem calculate based on di latest comment for di last year*. E be like say di way dem calculate di score dey somehow, but na di data wey dem scrape so we go just use am like dat for now. 
  
  ✅ Based on di other columns for dis data, you fit think of another way to calculate di average score?

* `Total_Number_of_Reviews`
  * Di total number of reviews wey di hotel don get - e no clear (unless you write code) if e dey refer to di reviews wey dey di dataset.
* `Additional_Number_of_Scoring`
  * Dis mean say di reviewer give score but dem no write positive or negative review.

**Review columns**

- `Reviewer_Score`
  - Dis na numerical value wey get at most 1 decimal place between di minimum and maximum values 2.5 and 10
  - E no dey explain why di lowest score wey person fit give na 2.5
- `Negative_Review`
  - If reviewer no write anything, dis field go get "**No Negative**"
  - Note say reviewer fit write positive review for di Negative review column (e.g. "nothing bad dey about dis hotel")
- `Review_Total_Negative_Word_Counts`
  - Higher negative word counts dey show lower score (without checking di sentiment)
- `Positive_Review`
  - If reviewer no write anything, dis field go get "**No Positive**"
  - Note say reviewer fit write negative review for di Positive review column (e.g. "nothing good dey about dis hotel at all")
- `Review_Total_Positive_Word_Counts`
  - Higher positive word counts dey show higher score (without checking di sentiment)
- `Review_Date` and `days_since_review`
  - You fit apply freshness or staleness measure to di review (older reviews fit no dey accurate like newer ones because hotel management don change, or dem don renovate, or dem don add pool etc.)
- `Tags`
  - Dis na short descriptors wey reviewer fit select to describe di type of guest dem be (e.g. solo or family), di type of room dem get, di length of stay and how dem submit di review. 
  - Unfortunately, di tags get wahala, check di section below wey dey discuss di usefulness.

**Reviewer columns**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Dis fit be factor for recommendation model, for example, if you fit determine say reviewers wey don write hundreds of reviews dey more likely to dey negative pass positive. But di reviewer for any particular review no get unique code, so e no fit link to set of reviews. 30 reviewers don write 100 or more reviews, but e hard to see how dis go help di recommendation model.
- `Reviewer_Nationality`
  - Some people fit think say certain nationalities dey more likely to give positive or negative review because of national inclination. Make you careful to build dis kind anecdotal views into your models. Dis na national (and sometimes racial) stereotypes, and each reviewer na individual wey write review based on di experience wey dem get. E fit don pass through many filters like di hotels wey dem don stay before, di distance wey dem travel, and di kind person wey dem be. To think say na di nationality cause di review score no dey easy to justify.

##### Examples

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Dis place no be hotel now, na construction site. Dem dey disturb me from early morning and all day wit noise wey no dey acceptable while I dey rest after long trip and dey work for di room. People dey work all day wit jackhammers for di rooms wey dey near my own. I ask make dem change my room but no silent room dey available. To make di matter worse, dem overcharge me. I check out for evening because I get early flight and dem give me correct bill. One day later, di hotel charge me again without my consent pass di price wey I book. Na terrible place. No punish yourself by booking here. | Nothing. Terrible place. Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

As you fit see, dis guest no enjoy di stay for di hotel. Di hotel get good average score of 7.8 and 1945 reviews, but dis reviewer give am 2.5 and write 115 words about how bad di stay be. If dem no write anything for di Positive_Review column, you fit think say nothing positive dey, but dem write 7 words of warning. If we just dey count words instead of di meaning or sentiment of di words, we fit get wrong view of di reviewer intention. E dey somehow say di score wey dem give na 2.5, because if di hotel stay bad like dat, why dem go give am any point at all? If you check di dataset well, you go see say di lowest score wey person fit give na 2.5, no be 0. Di highest score na 10.

##### Tags

As we talk before, e first look like say di idea to use `Tags` to categorize di data make sense. But di tags no dey standardized, wey mean say for one hotel, di options fit be *Single room*, *Twin room*, and *Double room*, but for di next hotel, e fit be *Deluxe Single Room*, *Classic Queen Room*, and *Executive King Room*. Dem fit be di same thing, but di variations plenty so di choice go be:

1. Try change all di terms to one standard, wey go hard because e no clear how to match di terms for each case (e.g. *Classic single room* fit match *Single room* but *Superior Queen Room wit Courtyard Garden or City View* go hard to match)

1. We fit use NLP approach measure di frequency of certain terms like *Solo*, *Business Traveller*, or *Family wit young kids* as e relate to each hotel, and use am for di recommendation  

Tags dey usually (but no be always) one single field wey get list of 5 to 6 comma-separated values wey align to *Type of trip*, *Type of guests*, *Type of room*, *Number of nights*, and *Type of device wey dem use submit di review*. But because some reviewers no dey fill each field (dem fit leave one blank), di values no dey always dey di same order.

For example, take *Type of group*. Dis field get 1025 unique possibilities for di `Tags` column, and unfortunately only some of dem dey refer to group (some dey refer to di type of room etc.). If you filter only di ones wey mention family, di results go contain many *Family room* type results. If you include di term *with*, i.e. count di *Family wit* values, di results go better, wit over 80,000 of di 515,000 results wey get di phrase "Family wit young children" or "Family wit older children".

Dis mean say di tags column no dey completely useless, but e go need work to make am useful.

##### Average hotel score

Di dataset get some kind wahala or discrepancies wey I no fit understand, but I go show dem here so you go sabi dem when you dey build your models. If you fit figure am out, abeg let us know for di discussion section!

Di dataset get di following columns wey relate to di average score and number of reviews: 

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

Di hotel wey get di most reviews for dis dataset na *Britannia International Hotel Canary Wharf* wit 4789 reviews out of 515,000. But if we check di `Total_Number_of_Reviews` value for dis hotel, e dey show 9086. You fit think say plenty scores dey wey no get reviews, so maybe we go add di `Additional_Number_of_Scoring` column value. Dat value na 2682, and if we add am to 4789, e go give us 7,471 wey still dey 1615 short of di `Total_Number_of_Reviews`. 

If you check di `Average_Score` column, you fit think say na di average of di reviews wey dey di dataset, but di description from Kaggle na "*Average Score of di hotel, wey dem calculate based on di latest comment for di last year*". Dis no dey too useful, but we fit calculate our own average based on di review scores wey dey di dataset. Using di same hotel as example, di average hotel score wey dem give na 7.1 but di calculated score (average reviewer score *wey dey* di dataset) na 6.8. Dis dey close, but e no be di same value, and we fit only guess say di scores wey dey di `Additional_Number_of_Scoring` reviews increase di average to 7.1. Unfortunately, since we no fit test or prove dat assumption, e go hard to use or trust `Average_Score`, `Additional_Number_of_Scoring` and `Total_Number_of_Reviews` when dem dey based on, or dey refer to, data wey we no get.

To make di matter worse, di hotel wey get di second highest number of reviews get calculated average score of 8.12 and di dataset `Average_Score` na 8.1. Dis correct score na coincidence or di first hotel get wahala?
On top say dis hotel fit be one kind outlier, and say maybe most of di values dey match (but some no dey match for some reason), we go write one short program next to check di values wey dey di dataset and confirm di correct way to use (or no use) di values.

> 🚨 Warning
>
> As you dey work with dis dataset, you go write code wey go calculate something from di text without you needing to read or analyse di text by yourself. Dis na di main thing for NLP, to fit understand meaning or sentiment without human dey do am. But e possible say you go read some of di negative reviews. I go advise you make you no read am, because you no need am. Some of dem na just silly or irrelevant negative hotel reviews, like "Di weather no good", something wey di hotel or anybody no fit control. But some reviews get dark side too. Sometimes di negative reviews dey racist, sexist, or ageist. Dis na bad thing but e dey expected for dataset wey dem scrape from public website. Some reviewers dey leave reviews wey go dey offensive, uncomfortable, or go make you vex. E better make di code measure di sentiment than make you read dem yourself and vex. But na small number of people dey write dis kain thing, but dem still dey.

## Exercise - Data exploration
### Load di data

E don do for di visual check of di data, now you go write code to get answers! Dis section dey use pandas library. Di first thing wey you go do na to make sure say you fit load and read di CSV data. Pandas library get fast CSV loader, and di result go dey inside dataframe, like we don see for di previous lessons. Di CSV wey we dey load get over half a million rows, but only 17 columns. Pandas dey give you plenty powerful ways to interact with dataframe, including di ability to perform operations for every row.

From here for dis lesson, we go get code snippets and some explanation of di code plus discussion about wetin di results mean. Use di _notebook.ipynb_ wey dem include for your code.

Make we start with loading di data file wey you go use:

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

Now wey di data don load, we fit perform some operations for am. Keep dis code for di top of your program for di next part.

## Explore di data

For dis case, di data don already *clean*, dat one mean say e don ready to work with, and e no get characters for other languages wey fit confuse algorithms wey dey expect only English characters.

✅ You fit need to work with data wey need some initial processing to arrange am before you apply NLP techniques, but no be dis time. If you need to, how you go handle non-English characters?

Take small time to make sure say once di data don load, you fit explore am with code. E dey very easy to wan focus on di `Negative_Review` and `Positive_Review` columns. Dem full with natural text for your NLP algorithms to process. But wait! Before you jump enter di NLP and sentiment, you suppose follow di code below to confirm if di values wey dey di dataset match di values wey you calculate with pandas.

## Dataframe operations

Di first task for dis lesson na to check if di following assertions dey correct by writing some code wey go check di dataframe (without changing am).

> Like many programming tasks, e get plenty ways to complete am, but better advice na to do am for di simplest, easiest way wey you fit, especially if e go dey easier to understand when you come back to dis code for future. With dataframes, e get comprehensive API wey go often get way to do wetin you want efficiently.

Treat di following questions as coding tasks and try answer dem without looking di solution.

1. Print di *shape* of di dataframe wey you don just load (di shape na di number of rows and columns)
2. Calculate di frequency count for reviewer nationalities:
   1. How many distinct values dey for di column `Reviewer_Nationality` and wetin dem be?
   2. Which reviewer nationality na di most common for di dataset (print di country and number of reviews)?
   3. Wetin be di next top 10 most frequently found nationalities, and their frequency count?
3. Which hotel dem review pass for each of di top 10 most reviewer nationalities?
4. How many reviews dey per hotel (frequency count of hotel) for di dataset?
5. Even though e get `Average_Score` column for each hotel for di dataset, you fit calculate average score (to get di average of all reviewer scores for di dataset for each hotel). Add new column to your dataframe with di column header `Calc_Average_Score` wey go contain di calculated average. 
6. Any hotel get di same (rounded to 1 decimal place) `Average_Score` and `Calc_Average_Score`?
   1. Try write Python function wey go take Series (row) as argument and compare di values, print message when di values no dey equal. Then use `.apply()` method to process every row with di function.
7. Calculate and print how many rows get column `Negative_Review` values of "No Negative" 
8. Calculate and print how many rows get column `Positive_Review` values of "No Positive"
9. Calculate and print how many rows get column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative"

### Code answers

1. Print di *shape* of di dataframe wey you don just load (di shape na di number of rows and columns)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calculate di frequency count for reviewer nationalities:

   1. How many distinct values dey for di column `Reviewer_Nationality` and wetin dem be?
   2. Which reviewer nationality na di most common for di dataset (print di country and number of reviews)?

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

   3. Wetin be di next top 10 most frequently found nationalities, and their frequency count?

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

3. Which hotel dem review pass for each of di top 10 most reviewer nationalities?

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

4. How many reviews dey per hotel (frequency count of hotel) for di dataset?

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
   
   You fit notice say di *counted in di dataset* results no match di value for `Total_Number_of_Reviews`. E no clear if dis value for di dataset represent di total number of reviews wey di hotel get, but no be all dem scrape, or na some other calculation. `Total_Number_of_Reviews` no dey used for di model because e no clear.

5. Even though e get `Average_Score` column for each hotel for di dataset, you fit calculate average score (to get di average of all reviewer scores for di dataset for each hotel). Add new column to your dataframe with di column header `Calc_Average_Score` wey go contain di calculated average. Print di columns `Hotel_Name`, `Average_Score`, and `Calc_Average_Score`.

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

   You fit also wonder about di `Average_Score` value and why e dey sometimes different from di calculated average score. As we no fit know why some of di values dey match, but others get difference, e better for dis case to use di review scores wey we get to calculate di average by ourselves. But di differences dey usually very small, here na di hotels wey get di biggest difference from di dataset average and di calculated average:

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

   With only 1 hotel wey get difference of score wey pass 1, e mean say we fit ignore di difference and use di calculated average score.

6. Calculate and print how many rows get column `Negative_Review` values of "No Negative" 

7. Calculate and print how many rows get column `Positive_Review` values of "No Positive"

8. Calculate and print how many rows get column `Positive_Review` values of "No Positive" **and** `Negative_Review` values of "No Negative"

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

Another way to count items without Lambdas, and use sum to count di rows:

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

   You fit don notice say e get 127 rows wey get both "No Negative" and "No Positive" values for di columns `Negative_Review` and `Positive_Review` respectively. Dat one mean say di reviewer give di hotel one numerical score, but dem no write either positive or negative review. Luckily dis na small amount of rows (127 out of 515738, or 0.02%), so e no go fit affect our model or results for any particular way, but you fit no expect say dataset of reviews go get rows wey no get reviews, so e dey important to explore di data to find rows like dis.

Now wey you don explore di dataset, for di next lesson you go filter di data and add some sentiment analysis.

---
## 🚀Challenge

Dis lesson show, as we don see for previous lessons, how e dey very important to understand your data and di wahala wey e fit get before you perform operations for am. Text-based data, especially, need careful check. Dig through different text-heavy datasets and see if you fit find areas wey fit bring bias or make sentiment for model no balance.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Self Study

Take [dis Learning Path on NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) to find tools wey you fit try when you dey build speech and text-heavy models.

## Assignment 

[NLTK](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:  
Dis docu don dey translate wit AI translation service [Co-op Translator](https://github.com/Azure/co-op-translator). Even though we dey try make am accurate, abeg sabi say automatic translation fit get mistake or no correct well. Di original docu for im native language na di main correct source. For important information, e go beta make professional human translator check am. We no go fit take blame for any misunderstanding or wrong interpretation wey fit happen because you use dis translation.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->