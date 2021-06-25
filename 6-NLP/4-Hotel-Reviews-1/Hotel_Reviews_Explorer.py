# EDA
import pandas as pd
import time

def get_difference_review_avg(row):
    return row["Average_Score"] - row["Calc_Average_Score"]

# Load the hotel reviews from CSV
print("Loading data file now, this could take a while depending on file size")
start = time.time()
df = pd.read_csv('Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")

# What shape is the data (rows, columns)?
print("The shape of the data (rows, cols) is " + str(df.shape))

# value_counts() creates a Series object that has index and values
#                in this case, the country and the frequency they occur in reviewer nationality
nationality_freq = df["Reviewer_Nationality"].value_counts()

# What reviewer nationality is the most common in the dataset?
print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")

# What is the top 10 most common nationalities and their frequencies?
print("The top 10 highest frequency reviewer nationalities are:")
print(nationality_freq[0:10].to_string())

# How many unique nationalities are there?
print("There are " + str(nationality_freq.index.size) + " unique nationalities in the dataset")

# What was the most frequently reviewed hotel for the top 10 nationalities - print the hotel and number of reviews
for nat in nationality_freq[:10].index:
   # First, extract all the rows that match the criteria into a new dataframe
   nat_df = df[df["Reviewer_Nationality"] == nat]   
   # Now get the hotel freq
   freq = nat_df["Hotel_Name"].value_counts()
   print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 

# How many reviews are there per hotel (frequency count of hotel) and do the results match the value in `Total_Number_of_Reviews`?
# First create a new dataframe based on the old one, removing the uneeded columns
hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
# Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
# Get rid of all the duplicated rows
hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
print()
print(hotel_freq_df.to_string())
print(str(hotel_freq_df.shape))

# While there is an `Average_Score` for each hotel according to the dataset, 
# you can also calculate an average score (getting the average of all reviewer scores in the dataset for each hotel)
# Add a new column to your dataframe with the column header `Calc_Average_Score` that contains that calculated average. 
df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
# Add a new column with the difference between the two average scores
df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
# Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
# Sort the dataframe to find the lowest and highest average score difference
review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
print(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
# Do any hotels have the same (rounded to 1 decimal place) `Average_Score` and `Calc_Average_Score`?
