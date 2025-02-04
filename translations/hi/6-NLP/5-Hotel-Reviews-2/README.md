# होटल समीक्षा के साथ भावना विश्लेषण

अब जब आपने डेटासेट का विस्तार से अन्वेषण कर लिया है, तो समय आ गया है कि आप कॉलम को फ़िल्टर करें और फिर होटल के बारे में नई अंतर्दृष्टि प्राप्त करने के लिए एनएलपी तकनीकों का उपयोग करें।
## [पूर्व-व्याख्यान क्विज़](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### फ़िल्टरिंग और भावना विश्लेषण संचालन

जैसा कि आपने शायद देखा होगा, डेटासेट में कुछ समस्याएं हैं। कुछ कॉलम बेकार की जानकारी से भरे हुए हैं, जबकि अन्य गलत लगते हैं। अगर वे सही भी हैं, तो यह स्पष्ट नहीं है कि उनकी गणना कैसे की गई थी, और आपके अपने गणनाओं द्वारा उत्तरों को स्वतंत्र रूप से सत्यापित नहीं किया जा सकता।

## अभ्यास: थोड़ा और डेटा प्रोसेसिंग

डेटा को थोड़ा और साफ करें। उन कॉलम को जोड़ें जो बाद में उपयोगी होंगे, अन्य कॉलम में मान बदलें, और कुछ कॉलम को पूरी तरह से हटा दें।

1. प्रारंभिक कॉलम प्रोसेसिंग

   1. `lat` और `lng` को हटा दें

   2. `Hotel_Address` मानों को निम्नलिखित मानों के साथ बदलें (यदि पता शहर और देश का नाम शामिल करता है, तो इसे केवल शहर और देश में बदलें)।

      ये डेटासेट में केवल शहर और देश हैं:

      एम्स्टर्डम, नीदरलैंड्स

      बार्सिलोना, स्पेन

      लंदन, यूनाइटेड किंगडम

      मिलान, इटली

      पेरिस, फ्रांस

      वियना, ऑस्ट्रिया 

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

      अब आप देश स्तर का डेटा क्वेरी कर सकते हैं:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | होटल_पता              | होटल_नाम |
      | :--------------------- | :--------: |
      | एम्स्टर्डम, नीदरलैंड्स |    105     |
      | बार्सिलोना, स्पेन       |    211     |
      | लंदन, यूनाइटेड किंगडम |    400     |
      | मिलान, इटली           |    162     |
      | पेरिस, फ्रांस          |    458     |
      | वियना, ऑस्ट्रिया        |    158     |

2. होटल मेटा-रिव्यू कॉलम प्रोसेस करें

  1. `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` को हमारे अपने गणना किए गए स्कोर के साथ हटा दें

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. समीक्षा कॉलम प्रोसेस करें

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` को हटा दें, आप पूछ सकते हैं कि क्या प्रोसेसिंग को काफी हद तक कम करना संभव है। सौभाग्य से, यह संभव है - लेकिन पहले आपको यह सुनिश्चित करने के लिए कुछ कदम उठाने की आवश्यकता है कि कौन से टैग प्रासंगिक हैं।

### टैग फ़िल्टरिंग

याद रखें कि डेटासेट का लक्ष्य भावना और कॉलम जोड़ना है जो आपको सर्वश्रेष्ठ होटल चुनने में मदद करेगा (अपने लिए या शायद किसी क्लाइंट के लिए जो आपसे होटल सिफारिश बॉट बनाने का काम सौंप रहा है)। आपको खुद से पूछना होगा कि क्या टैग अंतिम डेटासेट में उपयोगी हैं या नहीं। यहां एक व्याख्या है (यदि आपको डेटासेट अन्य कारणों से चाहिए तो विभिन्न टैग चयन में रह सकते हैं/नहीं रह सकते):

1. यात्रा का प्रकार प्रासंगिक है, और इसे रहना चाहिए
2. अतिथि समूह का प्रकार महत्वपूर्ण है, और इसे रहना चाहिए
3. अतिथि ने जिस प्रकार के कमरे, सुइट या स्टूडियो में ठहराव किया वह अप्रासंगिक है (सभी होटलों में मूल रूप से समान कमरे होते हैं)
4. समीक्षा जिस डिवाइस से सबमिट की गई वह अप्रासंगिक है
5. समीक्षक ने कितनी रातें ठहराई यह प्रासंगिक हो सकता है अगर आप मानते हैं कि लंबे ठहराव का मतलब है कि उन्हें होटल अधिक पसंद आया, लेकिन यह एक खिंचाव है, और शायद अप्रासंगिक

संक्षेप में, **2 प्रकार के टैग रखें और अन्य को हटा दें**।

पहले, आप टैग की गिनती तब तक नहीं करना चाहेंगे जब तक वे बेहतर प्रारूप में न हों, इसलिए इसका मतलब है कोष्ठकों और उद्धरणों को हटाना। आप इसे कई तरीकों से कर सकते हैं, लेकिन आप सबसे तेज़ तरीका चाहते हैं क्योंकि बहुत सारा डेटा प्रोसेस करने में बहुत समय लग सकता है। सौभाग्य से, पांडा के पास इन चरणों में से प्रत्येक को करने का एक आसान तरीका है।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

प्रत्येक टैग कुछ इस प्रकार बन जाता है: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

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

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` कॉलम नए कॉलम में से एक से मेल खाता है, तो 1 जोड़ें, यदि नहीं, तो 0 जोड़ें। अंतिम परिणाम यह होगा कि कितने समीक्षकों ने इस होटल को (कुल मिलाकर) व्यापार बनाम अवकाश के लिए चुना, या पालतू जानवर को लाने के लिए चुना, और यह होटल की सिफारिश करते समय उपयोगी जानकारी है।

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

### अपनी फ़ाइल सहेजें

अंत में, अब जैसा है वैसा ही डेटासेट एक नए नाम से सहेजें।

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## भावना विश्लेषण संचालन

इस अंतिम अनुभाग में, आप समीक्षा कॉलम पर भावना विश्लेषण लागू करेंगे और परिणामों को एक डेटासेट में सहेजेंगे।

## अभ्यास: फ़िल्टर किया गया डेटा लोड और सहेजें

ध्यान दें कि अब आप वह फ़िल्टर किया गया डेटासेट लोड कर रहे हैं जिसे पिछले अनुभाग में सहेजा गया था, **मूल** डेटासेट नहीं।

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

### स्टॉप शब्द हटाना

यदि आप नकारात्मक और सकारात्मक समीक्षा कॉलम पर भावना विश्लेषण चलाते हैं, तो इसमें बहुत समय लग सकता है। एक शक्तिशाली परीक्षण लैपटॉप पर तेज़ सीपीयू के साथ परीक्षण किया गया, इसमें 12 - 14 मिनट लगे, जो भावना पुस्तकालय पर निर्भर करता है। यह (सापेक्ष) लंबा समय है, इसलिए यह जांचने योग्य है कि क्या इसे तेज किया जा सकता है।

स्टॉप शब्दों, या सामान्य अंग्रेजी शब्दों को हटाना जो वाक्य की भावना को नहीं बदलते, पहला कदम है। उन्हें हटाकर, भावना विश्लेषण को तेज़ी से चलाना चाहिए, लेकिन कम सटीक नहीं होना चाहिए (क्योंकि स्टॉप शब्द भावना को प्रभावित नहीं करते, लेकिन वे विश्लेषण को धीमा कर देते हैं)।

सबसे लंबी नकारात्मक समीक्षा 395 शब्दों की थी, लेकिन स्टॉप शब्दों को हटाने के बाद, यह 195 शब्दों की हो गई।

स्टॉप शब्दों को हटाना भी एक तेज़ ऑपरेशन है, 515,000 पंक्तियों में से 2 समीक्षा कॉलम से स्टॉप शब्दों को हटाने में परीक्षण डिवाइस पर 3.3 सेकंड लगे। आपके लिए इसमें थोड़ा अधिक या कम समय लग सकता है, जो आपके डिवाइस के सीपीयू गति, रैम, एसएसडी होने या न होने और कुछ अन्य कारकों पर निर्भर करता है। ऑपरेशन की सापेक्ष कम समय की वजह से अगर यह भावना विश्लेषण समय में सुधार करता है, तो यह करने लायक है।

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

### भावना विश्लेषण करना

अब आपको नकारात्मक और सकारात्मक समीक्षा कॉलम के लिए भावना विश्लेषण की गणना करनी चाहिए, और परिणाम को 2 नए कॉलम में सहेजना चाहिए। भावना की परीक्षा यह होगी कि इसे समीक्षक के स्कोर से तुलना की जाए। उदाहरण के लिए, यदि भावना विश्लेषण नकारात्मक समीक्षा को 1 (अत्यधिक सकारात्मक भावना) और सकारात्मक समीक्षा भावना को 1 मानता है, लेकिन समीक्षक ने होटल को सबसे कम संभव स्कोर दिया, तो या तो समीक्षा पाठ स्कोर से मेल नहीं खाता, या भावना विश्लेषक भावना को सही से पहचान नहीं पाया। आपको उम्मीद करनी चाहिए कि कुछ भावना स्कोर पूरी तरह से गलत होंगे, और अक्सर यह समझाने योग्य होगा, जैसे कि समीक्षा अत्यधिक व्यंग्यात्मक हो सकती है "बेशक मुझे बिना हीटिंग वाले कमरे में सोना बहुत पसंद आया" और भावना विश्लेषक सोचता है कि यह सकारात्मक भावना है, जबकि एक इंसान इसे पढ़कर जानता होगा कि यह व्यंग्य है।

एनएलटीके विभिन्न भावना विश्लेषक प्रदान करता है, और आप उन्हें बदल सकते हैं और देख सकते हैं कि भावना अधिक या कम सटीक है। यहां VADER भावना विश्लेषण का उपयोग किया गया है।

> हुट्टो, सी.जे. और गिल्बर्ट, ई.ई. (2014)। VADER: सोशल मीडिया टेक्स्ट के भावना विश्लेषण के लिए एक सरल नियम-आधारित मॉडल। आठवीं अंतर्राष्ट्रीय सम्मेलन पर वेबलॉग्स और सोशल मीडिया (आईसीडब्ल्यूएसएम-14)। एन आर्बर, एमआई, जून 2014।

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

बाद में अपने प्रोग्राम में जब आप भावना की गणना करने के लिए तैयार हों, तो आप इसे प्रत्येक समीक्षा पर इस प्रकार लागू कर सकते हैं:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

यह मेरे कंप्यूटर पर लगभग 120 सेकंड लेता है, लेकिन यह प्रत्येक कंप्यूटर पर भिन्न होगा। यदि आप परिणामों को प्रिंट करना चाहते हैं और देखना चाहते हैं कि क्या भावना समीक्षा से मेल खाती है:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

फाइल का उपयोग करने से पहले आखिरी चीज जो करनी है, वह इसे सहेजना है! आपको अपने सभी नए कॉलम को फिर से क्रमबद्ध करने पर भी विचार करना चाहिए ताकि वे काम करने में आसान हों (एक इंसान के लिए, यह एक सौंदर्य परिवर्तन है)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

आपको [विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) के लिए पूरा कोड चलाना चाहिए (जैसा कि आपने [अपनी फ़िल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) चलाने के बाद Hotel_Reviews_Filtered.csv फाइल बनाने के लिए किया था)।

समीक्षा करने के लिए, चरण हैं:

1. मूल डेटासेट फाइल **Hotel_Reviews.csv** को पिछले पाठ में [एक्सप्लोरर नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) के साथ अन्वेषण किया गया है
2. Hotel_Reviews.csv को [फ़िल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) द्वारा फ़िल्टर किया गया है जिसके परिणामस्वरूप **Hotel_Reviews_Filtered.csv** बनता है
3. Hotel_Reviews_Filtered.csv को [भावना विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) द्वारा प्रोसेस किया गया है जिसके परिणामस्वरूप **Hotel_Reviews_NLP.csv** बनता है
4. नीचे दिए गए एनएलपी चैलेंज में Hotel_Reviews_NLP.csv का उपयोग करें

### निष्कर्ष

जब आपने शुरू किया, तो आपके पास कॉलम और डेटा के साथ एक डेटासेट था, लेकिन इसका सारा हिस्सा सत्यापित या उपयोग नहीं किया जा सकता था। आपने डेटा का अन्वेषण किया, जो आवश्यक नहीं था उसे फ़िल्टर किया, टैग को उपयोगी चीजों में परिवर्तित किया, अपने औसत की गणना की, कुछ भावना कॉलम जोड़े और उम्मीद है, प्राकृतिक पाठ प्रोसेसिंग के बारे में कुछ रोचक चीजें सीखी हैं।

## [पोस्ट-व्याख्यान क्विज़](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## चुनौती

अब जब आपने अपने डेटासेट का भावना के लिए विश्लेषण कर लिया है, तो देखें कि क्या आप इस पाठ्यक्रम में सीखी गई रणनीतियों (शायद क्लस्टरिंग?) का उपयोग करके भावना के आसपास पैटर्न निर्धारित कर सकते हैं।

## समीक्षा और स्व-अध्ययन

[इस लर्न मॉड्यूल](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) को लें ताकि आप अधिक जान सकें और पाठ में भावना का पता लगाने के लिए विभिन्न उपकरणों का उपयोग कर सकें।
## असाइनमेंट 

[एक अलग डेटासेट आज़माएं](assignment.md)

**अस्वीकरण**:
इस दस्तावेज़ का अनुवाद मशीन आधारित एआई अनुवाद सेवाओं का उपयोग करके किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवादों में त्रुटियां या अशुद्धियाँ हो सकती हैं। इसकी मूल भाषा में मूल दस्तावेज़ को प्रामाणिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम जिम्मेदार नहीं हैं।