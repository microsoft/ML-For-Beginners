<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T18:38:29+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "mr"
}
-->
# हॉटेल पुनरावलोकनांसह भावना विश्लेषण

आता तुम्ही डेटासेटचा सविस्तर अभ्यास केला आहे, आता स्तंभ फिल्टर करण्याची आणि हॉटेल्सबद्दल नवीन अंतर्दृष्टी मिळवण्यासाठी डेटासेटवर NLP तंत्र वापरण्याची वेळ आली आहे.
## [पूर्व-व्याख्यान प्रश्नमंजुषा](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### फिल्टरिंग आणि भावना विश्लेषण प्रक्रिया

तुम्ही कदाचित लक्षात घेतले असेल की डेटासेटमध्ये काही समस्या आहेत. काही स्तंभ निरुपयोगी माहितीने भरलेले आहेत, तर काही चुकीचे वाटतात. जर ते योग्य असतील, तरीही ते कसे मोजले गेले हे अस्पष्ट आहे, आणि तुमच्या स्वतःच्या गणनांद्वारे उत्तरांची स्वतंत्रपणे पडताळणी करता येत नाही.

## व्यायाम: डेटा प्रक्रिया थोडी अधिक

डेटा थोडा अधिक स्वच्छ करा. अशा स्तंभांची भर घाला जे नंतर उपयुक्त ठरतील, इतर स्तंभांमधील मूल्ये बदला आणि काही स्तंभ पूर्णपणे काढून टाका.

1. प्रारंभिक स्तंभ प्रक्रिया

   1. `lat` आणि `lng` काढा

   2. `Hotel_Address` मूल्ये खालीलप्रमाणे बदला (जर पत्त्यात शहर आणि देशाचा उल्लेख असेल, तर ते फक्त शहर आणि देशात बदला).

      डेटासेटमध्ये फक्त खालील शहरं आणि देश आहेत:

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

      आता तुम्ही देश स्तरावर डेटा क्वेरी करू शकता:

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

2. हॉटेल मेटा-रिव्ह्यू स्तंभ प्रक्रिया

  1. `Additional_Number_of_Scoring` काढा

  1. `Total_Number_of_Reviews` त्या हॉटेलसाठी डेटासेटमध्ये प्रत्यक्षात असलेल्या पुनरावलोकनांच्या एकूण संख्येसह बदला 

  1. `Average_Score` आमच्याच गणनेनुसार बदला

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. पुनरावलोकन स्तंभ प्रक्रिया

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` आणि `days_since_review` काढा

   2. `Reviewer_Score`, `Negative_Review`, आणि `Positive_Review` तसेच ठेवा,
     
   3. `Tags` सध्या ठेवा

     - पुढील विभागात टॅग्सवर काही अतिरिक्त फिल्टरिंग ऑपरेशन्स केले जातील आणि नंतर टॅग्स काढले जातील

4. पुनरावलोकनकर्ता स्तंभ प्रक्रिया

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` काढा
  
  2. `Reviewer_Nationality` ठेवा

### टॅग स्तंभ

`Tag` स्तंभ समस्यात्मक आहे कारण तो एक यादी (मजकूर स्वरूपात) स्तंभात संग्रहित आहे. दुर्दैवाने, या स्तंभातील उपविभागांचा क्रम आणि संख्या नेहमी समान नसते. 515,000 ओळी, 1427 हॉटेल्स आणि प्रत्येक पुनरावलोकनकर्त्याने निवडलेल्या पर्यायांमध्ये थोडासा फरक असल्यामुळे योग्य वाक्यांश ओळखणे मानवी दृष्टिकोनातून कठीण आहे. येथे NLP उपयुक्त ठरते. तुम्ही मजकूर स्कॅन करू शकता आणि सर्वात सामान्य वाक्यांश शोधू शकता आणि त्यांची गणना करू शकता.

दुर्दैवाने, आम्हाला एकाच शब्दांमध्ये रस नाही, तर बहु-शब्द वाक्यांशांमध्ये (उदा. *Business trip*). इतक्या मोठ्या डेटावर बहु-शब्द वारंवारता वितरण अल्गोरिदम चालवणे (6762646 शब्द) खूप वेळ घेऊ शकते, परंतु डेटा न पाहता, ते आवश्यक खर्च वाटते. येथे एक्सप्लोरेटरी डेटा अॅनालिसिस उपयुक्त ठरते, कारण तुम्ही टॅग्सचा नमुना पाहिला आहे जसे की `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, तुम्ही विचार करू शकता की प्रक्रिया मोठ्या प्रमाणात कमी करणे शक्य आहे का. सुदैवाने, ते शक्य आहे - परंतु प्रथम तुम्हाला टॅग्सची खात्री करण्यासाठी काही चरणांचे अनुसरण करणे आवश्यक आहे.

### टॅग्स फिल्टर करणे

लक्षात ठेवा की डेटासेटचा उद्देश भावना आणि स्तंभ जोडणे आहे जे तुम्हाला सर्वोत्तम हॉटेल निवडण्यास मदत करतील (तुमच्यासाठी किंवा कदाचित तुम्हाला हॉटेल शिफारस करणारा बॉट तयार करण्याचे काम देणाऱ्या क्लायंटसाठी). तुम्हाला विचार करणे आवश्यक आहे की टॅग्स अंतिम डेटासेटमध्ये उपयुक्त आहेत की नाहीत. येथे एक व्याख्या दिली आहे (जर तुम्हाला डेटासेट इतर कारणांसाठी आवश्यक असेल तर वेगळे टॅग्स निवड/काढले जाऊ शकतात):

1. प्रवासाचा प्रकार संबंधित आहे, आणि तो ठेवला पाहिजे
2. पाहुण्यांच्या गटाचा प्रकार महत्त्वाचा आहे, आणि तो ठेवला पाहिजे
3. पाहुण्यांनी ज्या खोलीत, सूटमध्ये किंवा स्टुडिओमध्ये राहिले त्याचा प्रकार अप्रासंगिक आहे (सर्व हॉटेल्समध्ये मूलतः समान खोली असते)
4. पुनरावलोकन ज्या डिव्हाइसवर सबमिट केले गेले ते अप्रासंगिक आहे
5. पुनरावलोकनकर्त्याने किती रात्री राहिले *संबंधित* असू शकते जर तुम्ही दीर्घकालीन राहण्याचे श्रेय हॉटेलला अधिक आवडले असेल, परंतु ते तर्कसंगत आहे आणि कदाचित अप्रासंगिक आहे

सारांश, **2 प्रकारचे टॅग ठेवा आणि इतर काढा**.

प्रथम, तुम्हाला टॅग्स मोजायचे नाहीत जोपर्यंत ते चांगल्या स्वरूपात नसतील, त्यामुळे चौकोनी कंस आणि उद्धरण काढणे आवश्यक आहे. तुम्ही हे अनेक प्रकारे करू शकता, परंतु तुम्हाला सर्वात जलद पद्धत हवी आहे कारण खूप डेटा प्रक्रिया करणे वेळखाऊ ठरू शकते. सुदैवाने, पॅंडासमध्ये प्रत्येक चरण करण्याचा सोपा मार्ग आहे.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

प्रत्येक टॅग असे काहीतरी बनते: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

पुढे एक समस्या आढळते. काही पुनरावलोकनांमध्ये, किंवा ओळींमध्ये, 5 स्तंभ असतात, काहींमध्ये 3, काहींमध्ये 6. हे डेटासेट कसे तयार केले गेले याचा परिणाम आहे आणि दुरुस्त करणे कठीण आहे. तुम्हाला प्रत्येक वाक्यांशाची वारंवारता मोजायची आहे, परंतु ते प्रत्येक पुनरावलोकनात वेगळ्या क्रमाने आहेत, त्यामुळे गणना चुकीची असू शकते आणि हॉटेलला त्याला पात्र असलेला टॅग दिला जाऊ शकत नाही.

त्याऐवजी तुम्ही वेगळ्या क्रमाचा फायदा घेणार आहात, कारण प्रत्येक टॅग बहु-शब्द आहे परंतु अल्पविरामाने वेगळा आहे! याचे सर्वात सोपे मार्ग म्हणजे 6 तात्पुरते स्तंभ तयार करणे ज्यामध्ये प्रत्येक टॅग त्याच्या क्रमाशी संबंधित स्तंभात घातला जातो. तुम्ही नंतर 6 स्तंभ एका मोठ्या स्तंभात विलीन करू शकता आणि परिणामी स्तंभावर `value_counts()` पद्धत चालवू शकता. ते प्रिंट केल्यावर, तुम्हाला दिसेल की 2428 अद्वितीय टॅग्स होते. येथे एक छोटासा नमुना आहे:

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
| Superior Double or Twin Room   | 13570 |
| 2 rooms                        | 12393 |

`Submitted from a mobile device` सारखे काही सामान्य टॅग्स आपल्याला उपयोगाचे नाहीत, त्यामुळे ते मोजण्यापूर्वी काढून टाकणे स्मार्ट गोष्ट असू शकते, परंतु ही प्रक्रिया इतकी जलद आहे की तुम्ही ते ठेवू शकता आणि दुर्लक्ष करू शकता.

### राहण्याच्या कालावधीचे टॅग्स काढणे

हे टॅग्स काढणे हा पहिला टप्पा आहे, यामुळे विचारात घेण्याच्या टॅग्सची एकूण संख्या थोडी कमी होते. लक्षात ठेवा तुम्ही ते डेटासेटमधून काढत नाही, फक्त पुनरावलोकन डेटासेटमध्ये मोजण्याच्या/ठेवल्याच्या मूल्यांमधून काढता.

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

खोली, सूट, स्टुडिओ, अपार्टमेंट्स इत्यादींच्या प्रकारांची विविधता खूप मोठी आहे. ते सर्व साधारणपणे समान अर्थ दर्शवतात आणि आपल्यासाठी संबंधित नाहीत, त्यामुळे विचारात घेण्यापासून काढून टाका.

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

शेवटी, आणि हे आनंददायक आहे (कारण यासाठी फारशी प्रक्रिया करावी लागली नाही), तुम्ही खालील *उपयुक्त* टॅग्ससह राहाल:

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

तुम्ही `Travellers with friends` हे `Group` सारखेच आहे असे म्हणू शकता, आणि ते वाजवी आहे की वर नमूद केल्याप्रमाणे दोन्ही एकत्र करा. योग्य टॅग्स ओळखण्यासाठी कोड [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) मध्ये आहे.

अंतिम टप्पा म्हणजे प्रत्येक टॅगसाठी नवीन स्तंभ तयार करणे. मग, प्रत्येक पुनरावलोकन ओळीसाठी, जर `Tag` स्तंभ नवीन स्तंभांपैकी एखाद्या स्तंभाशी जुळत असेल, तर 1 जोडा, अन्यथा 0 जोडा. अंतिम परिणाम असा असेल की किती पुनरावलोकनकर्त्यांनी हॉटेल निवडले (एकत्रितपणे) व्यवसाय विरुद्ध विश्रांतीसाठी, किंवा पाळीव प्राणी आणण्यासाठी, आणि हॉटेलची शिफारस करताना ही उपयुक्त माहिती असेल.

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

### तुमची फाईल सेव्ह करा

शेवटी, डेटासेट सध्याच्या स्वरूपात नवीन नावाने सेव्ह करा.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## भावना विश्लेषण प्रक्रिया

या अंतिम विभागात, तुम्ही पुनरावलोकन स्तंभांवर भावना विश्लेषण लागू कराल आणि परिणाम डेटासेटमध्ये सेव्ह कराल.

## व्यायाम: फिल्टर केलेला डेटा लोड करा आणि सेव्ह करा

लक्षात ठेवा की आता तुम्ही मागील विभागात सेव्ह केलेला फिल्टर केलेला डेटासेट लोड करत आहात, **मूळ डेटासेट नाही**.

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

### स्टॉप शब्द काढून टाकणे

जर तुम्ही नकारात्मक आणि सकारात्मक पुनरावलोकन स्तंभांवर भावना विश्लेषण चालवले, तर त्याला खूप वेळ लागू शकतो. वेगवान CPU असलेल्या शक्तिशाली टेस्ट लॅपटॉपवर चाचणी केली असता, वापरलेल्या भावना लायब्ररीवर अवलंबून 12 - 14 मिनिटे लागली. हा (सापेक्ष) दीर्घ कालावधी आहे, त्यामुळे तो वेगवान होऊ शकतो का याचा तपास घेणे योग्य आहे.

सामान्य इंग्रजी शब्द, जे वाक्याच्या भावनेत बदल घडवत नाहीत, काढून टाकणे हा पहिला टप्पा आहे. ते काढून टाकल्याने भावना विश्लेषण वेगवान चालेल, परंतु कमी अचूक होणार नाही (कारण स्टॉप शब्द भावना प्रभावित करत नाहीत, परंतु ते विश्लेषण मंद करतात). 

सर्वात लांब नकारात्मक पुनरावलोकन 395 शब्दांचे होते, परंतु स्टॉप शब्द काढून टाकल्यानंतर ते 195 शब्दांचे होते.

स्टॉप शब्द काढून टाकणे ही देखील जलद प्रक्रिया आहे, 2 पुनरावलोकन स्तंभांमधून 515,000 ओळींवरून स्टॉप शब्द काढून टाकणे टेस्ट डिव्हाइसवर 3.3 सेकंद घेतले. तुमच्या डिव्हाइसच्या CPU गती, RAM, SSD आहे की नाही, आणि इतर काही घटकांवर अवलंबून तुम्हाला थोडा अधिक किंवा कमी वेळ लागू शकतो. ऑपरेशनचा तुलनेने कमी कालावधी म्हणजे जर ते भावना विश्लेषणाचा वेळ सुधारत असेल, तर ते करण्यासारखे आहे.

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

### भावना विश्लेषण करणे
आता तुम्हाला नकारात्मक आणि सकारात्मक पुनरावलोकन स्तंभांसाठी भावना विश्लेषणाची गणना करायची आहे आणि निकाल दोन नवीन स्तंभांमध्ये संग्रहित करायचा आहे. भावना विश्लेषणाची चाचणी म्हणजे तीच भावना पुनरावलोकनासाठी पुनरावलोकनकर्त्याच्या स्कोअरशी तुलना करणे. उदाहरणार्थ, जर भावना विश्लेषकाला वाटले की नकारात्मक पुनरावलोकनाची भावना 1 (अत्यंत सकारात्मक भावना) आहे आणि सकारात्मक पुनरावलोकनाची भावना 1 आहे, पण पुनरावलोकनकर्त्याने हॉटेलला शक्य तितका कमी स्कोअर दिला, तर कदाचित पुनरावलोकन मजकूर स्कोअरशी जुळत नाही, किंवा भावना विश्लेषक भावना योग्यरित्या ओळखू शकला नाही. तुम्हाला काही भावना स्कोअर पूर्णपणे चुकीचे असण्याची अपेक्षा असू शकते, आणि बऱ्याच वेळा ते समजण्याजोगे असेल, उदा. पुनरावलोकन अत्यंत उपरोधिक असू शकते "अर्थातच मला हीटिंग नसलेल्या खोलीत झोपायला खूप आवडलं" आणि भावना विश्लेषकाला वाटेल की ती सकारात्मक भावना आहे, जरी माणूस वाचताना जाणेल की ती उपरोध आहे.

NLTK विविध भावना विश्लेषक पुरवते ज्यांच्याशी शिकता येते, आणि तुम्ही त्यांना बदलून पाहू शकता की भावना अधिक अचूक आहे की नाही. येथे VADER भावना विश्लेषण वापरले आहे.

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

तुमच्या प्रोग्राममध्ये जेव्हा तुम्ही भावना मोजण्यासाठी तयार असाल, तेव्हा तुम्ही प्रत्येक पुनरावलोकनावर खालीलप्रमाणे लागू करू शकता:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

हे माझ्या संगणकावर सुमारे 120 सेकंद घेतं, पण प्रत्येक संगणकावर वेगळं असेल. जर तुम्हाला निकाल छापून पाहायचे असतील आणि भावना पुनरावलोकनाशी जुळते का ते पाहायचं असेल:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

चॅलेंजमध्ये वापरण्यापूर्वी फाईलसह शेवटची गोष्ट म्हणजे ती जतन करणे! तुम्ही तुमचे नवीन स्तंभ पुन्हा क्रमबद्ध करण्याचा विचार देखील करू शकता जेणेकरून ते वापरण्यास सोपे होतील (माणसासाठी, हा एक सौंदर्यात्मक बदल आहे).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

तुम्ही [विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) साठी संपूर्ण कोड चालवला पाहिजे (तुम्ही [फिल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) चालवल्यानंतर Hotel_Reviews_Filtered.csv फाईल तयार करण्यासाठी).

पुनरावलोकन करण्यासाठी, पायऱ्या अशा आहेत:

1. मूळ डेटासेट फाईल **Hotel_Reviews.csv** मागील धडात [एक्सप्लोरर नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) सह शोधली गेली आहे.
2. Hotel_Reviews.csv [फिल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) द्वारे फिल्टर केली जाते, परिणामी **Hotel_Reviews_Filtered.csv** तयार होते.
3. Hotel_Reviews_Filtered.csv [भावना विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) द्वारे प्रक्रिया केली जाते, परिणामी **Hotel_Reviews_NLP.csv** तयार होते.
4. NLP चॅलेंजमध्ये खाली **Hotel_Reviews_NLP.csv** वापरा.

### निष्कर्ष

जेव्हा तुम्ही सुरुवात केली, तेव्हा तुमच्याकडे स्तंभ आणि डेटा असलेला डेटासेट होता, पण त्यातील सर्व काही सत्यापित किंवा वापरले जाऊ शकत नव्हते. तुम्ही डेटा शोधला, गरज नसलेले फिल्टर केले, टॅग्स उपयुक्त गोष्टींमध्ये रूपांतरित केले, तुमची स्वतःची सरासरी गणना केली, काही भावना स्तंभ जोडले आणि कदाचित, नैसर्गिक मजकूर प्रक्रिया करण्याबद्दल काही मनोरंजक गोष्टी शिकल्या.

## [पाठानंतर प्रश्नमंजुषा](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## चॅलेंज

आता तुमचं डेटासेट भावना विश्लेषणासाठी तयार आहे, तुम्ही या अभ्यासक्रमात शिकलेल्या रणनीती (क्लस्टरिंग, कदाचित?) वापरून भावना संबंधित नमुने शोधू शकता का ते पाहा.

## पुनरावलोकन आणि स्व-अभ्यास

[हा Learn मॉड्यूल](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) घ्या, अधिक जाणून घ्या आणि मजकूरातील भावना शोधण्यासाठी वेगवेगळ्या साधनांचा वापर करा.

## असाइनमेंट

[वेगळ्या डेटासेटचा प्रयत्न करा](assignment.md)

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी कृपया लक्षात ठेवा की स्वयंचलित भाषांतरे त्रुटी किंवा अचूकतेच्या अभावाने युक्त असू शकतात. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून उद्भवलेल्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.