<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:58:43+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "hi"
}
-->
# होटल समीक्षाओं के साथ भाव विश्लेषण

अब जब आपने डेटासेट को विस्तार से देखा है, तो यह समय है कि कॉलम्स को फ़िल्टर करें और फिर डेटासेट पर NLP तकनीकों का उपयोग करके होटलों के बारे में नई जानकारियाँ प्राप्त करें।  
## [प्री-लेक्चर क्विज़](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### फ़िल्टरिंग और भाव विश्लेषण संचालन

जैसा कि आपने शायद देखा होगा, डेटासेट में कुछ समस्याएँ हैं। कुछ कॉलम्स में बेकार जानकारी भरी हुई है, जबकि अन्य गलत लगते हैं। अगर वे सही भी हैं, तो यह स्पष्ट नहीं है कि उन्हें कैसे गणना की गई, और आपके अपने गणनाओं से उत्तर स्वतंत्र रूप से सत्यापित नहीं किए जा सकते।

## अभ्यास: डेटा प्रोसेसिंग को थोड़ा और बेहतर बनाना

डेटा को थोड़ा और साफ करें। ऐसे कॉलम्स जोड़ें जो बाद में उपयोगी होंगे, अन्य कॉलम्स में मान बदलें, और कुछ कॉलम्स को पूरी तरह से हटा दें।

1. प्रारंभिक कॉलम प्रोसेसिंग

   1. `lat` और `lng` को हटा दें।

   2. `Hotel_Address` के मानों को निम्नलिखित मानों से बदलें (यदि पता शहर और देश का नाम शामिल करता है, तो इसे केवल शहर और देश में बदल दें)।

      डेटासेट में केवल ये शहर और देश हैं:

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

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | एम्स्टर्डम, नीदरलैंड्स |    105     |
      | बार्सिलोना, स्पेन       |    211     |
      | लंदन, यूनाइटेड किंगडम   |    400     |
      | मिलान, इटली            |    162     |
      | पेरिस, फ्रांस           |    458     |
      | वियना, ऑस्ट्रिया        |    158     |

2. होटल मेटा-रिव्यू कॉलम्स को प्रोसेस करें

   1. `Additional_Number_of_Scoring` को हटा दें।

   2. `Total_Number_of_Reviews` को उस होटल के लिए डेटासेट में वास्तव में मौजूद समीक्षाओं की कुल संख्या से बदलें।

   3. `Average_Score` को हमारे द्वारा गणना किए गए स्कोर से बदलें।

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. रिव्यू कॉलम्स को प्रोसेस करें

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` और `days_since_review` को हटा दें।

   2. `Reviewer_Score`, `Negative_Review`, और `Positive_Review` को जैसा है वैसा रखें।

   3. `Tags` को अभी के लिए रखें।

      - हम अगले सेक्शन में टैग्स पर कुछ अतिरिक्त फ़िल्टरिंग संचालन करेंगे और फिर टैग्स को हटा देंगे।

4. समीक्षक कॉलम्स को प्रोसेस करें

   1. `Total_Number_of_Reviews_Reviewer_Has_Given` को हटा दें।

   2. `Reviewer_Nationality` को रखें।

### टैग कॉलम्स

`Tag` कॉलम समस्या पैदा करता है क्योंकि यह एक सूची (टेक्स्ट फॉर्म में) के रूप में कॉलम में संग्रहीत है। दुर्भाग्य से, इस कॉलम में उपखंडों की संख्या और क्रम हमेशा समान नहीं होता। 515,000 पंक्तियों और 1427 होटलों के साथ, यह मानव के लिए सही वाक्यांशों की पहचान करना कठिन है। यही वह जगह है जहाँ NLP उपयोगी है। आप टेक्स्ट को स्कैन कर सकते हैं, सबसे सामान्य वाक्यांशों को ढूंढ सकते हैं, और उनकी गिनती कर सकते हैं।

दुर्भाग्य से, हम एकल शब्दों में रुचि नहीं रखते, बल्कि बहु-शब्द वाक्यांशों (जैसे *Business trip*) में रुचि रखते हैं। इतने बड़े डेटा (6762646 शब्द) पर एक बहु-शब्द आवृत्ति वितरण एल्गोरिदम चलाना असाधारण रूप से समय लेने वाला हो सकता है। लेकिन डेटा को देखे बिना, यह आवश्यक खर्च लग सकता है। यही वह जगह है जहाँ खोजपूर्ण डेटा विश्लेषण उपयोगी होता है। उदाहरण के लिए, आपने टैग्स का एक नमूना देखा होगा जैसे `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, आप यह पूछना शुरू कर सकते हैं कि क्या आपको प्रोसेसिंग को काफी हद तक कम करना संभव है। सौभाग्य से, यह संभव है - लेकिन पहले आपको रुचि के टैग्स का पता लगाने के लिए कुछ चरणों का पालन करना होगा।

### टैग्स को फ़िल्टर करना

याद रखें कि डेटासेट का उद्देश्य भाव और ऐसे कॉलम्स जोड़ना है जो आपको सबसे अच्छा होटल चुनने में मदद करें (अपने लिए या शायद किसी क्लाइंट के लिए जो आपसे होटल सिफारिश बॉट बनाने का काम करवा रहा हो)। आपको खुद से पूछना होगा कि क्या टैग्स अंतिम डेटासेट में उपयोगी हैं या नहीं। यहाँ एक व्याख्या है (यदि आपको अन्य कारणों से डेटासेट की आवश्यकता होती, तो अलग टैग्स चयन में रह सकते थे/निकल सकते थे):

1. यात्रा का प्रकार प्रासंगिक है, और इसे रखना चाहिए।  
2. अतिथि समूह का प्रकार महत्वपूर्ण है, और इसे रखना चाहिए।  
3. अतिथि ने जिस प्रकार के कमरे, सुइट, या स्टूडियो में ठहराव किया, वह अप्रासंगिक है (सभी होटलों में मूल रूप से समान कमरे होते हैं)।  
4. जिस डिवाइस से समीक्षा सबमिट की गई, वह अप्रासंगिक है।  
5. समीक्षक ने कितनी रातें ठहरीं, यह *प्रासंगिक हो सकता है यदि आप लंबे ठहराव को होटल को अधिक पसंद करने से जोड़ते हैं, लेकिन यह एक खिंचाव है और शायद अप्रासंगिक है।  

सारांश में, **2 प्रकार के टैग्स रखें और बाकी को हटा दें**।

पहले, आप तब तक टैग्स की गिनती नहीं करना चाहते जब तक वे बेहतर प्रारूप में न हों, इसलिए इसका मतलब है वर्ग कोष्ठक और उद्धरण चिह्न हटाना। आप इसे कई तरीकों से कर सकते हैं, लेकिन आप सबसे तेज़ तरीका चाहते हैं क्योंकि यह बहुत सारे डेटा को प्रोसेस करने में समय ले सकता है। सौभाग्य से, पांडा में इन चरणों में से प्रत्येक को करने का एक आसान तरीका है।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

प्रत्येक टैग कुछ इस तरह बन जाता है: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`।  

इसके बाद एक समस्या सामने आती है। कुछ समीक्षाओं या पंक्तियों में 5 कॉलम्स होते हैं, कुछ में 3, कुछ में 6। यह डेटासेट के निर्माण के तरीके का परिणाम है और इसे ठीक करना कठिन है। आप प्रत्येक वाक्यांश की आवृत्ति गिनना चाहते हैं, लेकिन वे प्रत्येक समीक्षा में अलग क्रम में हैं, इसलिए गिनती गलत हो सकती है, और किसी होटल को वह टैग नहीं मिल सकता जो उसे मिलना चाहिए था।

इसके बजाय आप अलग क्रम का लाभ उठाएँगे, क्योंकि प्रत्येक टैग बहु-शब्द है लेकिन एक अल्पविराम से भी अलग है! इसका सबसे सरल तरीका यह है कि 6 अस्थायी कॉलम्स बनाएँ और प्रत्येक टैग को उस कॉलम में डालें जो टैग के क्रम से मेल खाता हो। फिर आप 6 कॉलम्स को एक बड़े कॉलम में मर्ज कर सकते हैं और परिणामी कॉलम पर `value_counts()` विधि चला सकते हैं। इसे प्रिंट करने पर, आप देखेंगे कि 2428 अद्वितीय टैग्स थे। यहाँ एक छोटा नमूना है:

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

कुछ सामान्य टैग्स जैसे `Submitted from a mobile device` हमारे लिए उपयोगी नहीं हैं, इसलिए उन्हें वाक्यांश की घटना की गिनती करने से पहले हटाना एक स्मार्ट कदम हो सकता है, लेकिन यह इतना तेज़ ऑपरेशन है कि आप उन्हें छोड़ सकते हैं और अनदेखा कर सकते हैं।

### ठहराव की अवधि वाले टैग्स को हटाना

इन टैग्स को हटाना पहला कदम है, यह विचार किए जाने वाले टैग्स की कुल संख्या को थोड़ा कम कर देता है। ध्यान दें कि आप उन्हें डेटासेट से नहीं हटाते, बस उन्हें समीक्षाओं के डेटासेट में गिनने/रखने के लिए विचार से हटा देते हैं।

| ठहराव की अवधि   | Count  |
| --------------- | ------ |
| Stayed 1 night  | 193645 |
| Stayed 2 nights | 133937 |
| Stayed 3 nights | 95821  |
| Stayed 4 nights | 47817  |
| Stayed 5 nights | 20845  |
| Stayed 6 nights | 9776   |
| Stayed 7 nights | 7399   |
| Stayed 8 nights | 2502   |
| Stayed 9 nights | 1293   |
| ...             | ...    |

कमरे, सुइट्स, स्टूडियो, अपार्टमेंट्स आदि की एक बड़ी विविधता है। वे सभी लगभग एक ही चीज़ का मतलब रखते हैं और आपके लिए प्रासंगिक नहीं हैं, इसलिए उन्हें विचार से हटा दें।

| कमरे का प्रकार                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard Double Room          | 32248 |
| Superior Double Room          | 31393 |
| Deluxe Double Room            | 24823 |
| Double or Twin Room           | 22393 |
| Standard Double or Twin Room  | 17483 |
| Classic Double Room           | 16989 |
| Superior Double or Twin Room  | 13570 |

अंत में, और यह प्रसन्नता की बात है (क्योंकि इसमें ज्यादा प्रोसेसिंग नहीं लगी), आपके पास निम्नलिखित *उपयोगी* टैग्स बचेंगे:

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

आप यह तर्क दे सकते हैं कि `Travellers with friends` लगभग `Group` के समान है, और इसे ऊपर की तरह संयोजित करना उचित होगा। सही टैग्स की पहचान करने के लिए कोड [Tags नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) में है।

अंतिम चरण में इन टैग्स के लिए नए कॉलम्स बनाएँ। फिर, प्रत्येक समीक्षा पंक्ति के लिए, यदि `Tag` कॉलम नए कॉलम्स में से किसी एक से मेल खाता है, तो 1 जोड़ें, अन्यथा 0 जोड़ें। अंतिम परिणाम यह होगा कि कितने समीक्षकों ने इस होटल को (सामूहिक रूप से) व्यवसाय बनाम अवकाश के लिए चुना, या पालतू जानवर लाने के लिए, और यह होटल की सिफारिश करते समय उपयोगी जानकारी है।

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

अंत में, डेटासेट को अब एक नए नाम के साथ सहेजें।

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## भाव विश्लेषण संचालन

इस अंतिम सेक्शन में, आप समीक्षा कॉलम्स पर भाव विश्लेषण लागू करेंगे और परिणामों को एक डेटासेट में सहेजेंगे।

## अभ्यास: फ़िल्टर किए गए डेटा को लोड और सहेजें

ध्यान दें कि अब आप पिछले सेक्शन में सहेजे गए फ़िल्टर किए गए डेटासेट को लोड कर रहे हैं, **मूल** डेटासेट को नहीं।

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

### स्टॉप वर्ड्स हटाना

यदि आप नकारात्मक और सकारात्मक समीक्षा कॉलम्स पर भाव विश्लेषण चलाते हैं, तो इसमें काफी समय लग सकता है। एक शक्तिशाली टेस्ट लैपटॉप पर तेज़ CPU के साथ परीक्षण किया गया, तो इसमें 12-14 मिनट लगे, यह इस बात पर निर्भर करता है कि कौन सा भाव लाइब्रेरी उपयोग किया गया। यह (सापेक्ष रूप से) लंबा समय है, इसलिए यह जांचने लायक है कि इसे तेज़ किया जा सकता है या नहीं।

स्टॉप वर्ड्स, या सामान्य अंग्रेजी शब्द जो वाक्य के भाव को नहीं बदलते, को हटाना पहला कदम है। इन्हें हटाने से भाव विश्लेषण तेज़ होना चाहिए, लेकिन कम सटीक नहीं (क्योंकि स्टॉप वर्ड्स भाव को प्रभावित नहीं करते, लेकिन वे विश्लेषण को धीमा कर देते हैं)।

सबसे लंबी नकारात्मक समीक्षा 395 शब्दों की थी, लेकिन स्टॉप वर्ड्स हटाने के बाद यह 195 शब्दों की हो गई।

स्टॉप वर्ड्स हटाना भी एक तेज़ ऑपरेशन है। 515,000 पंक्तियों पर 2 समीक्षा कॉलम्स से स्टॉप वर्ड्स हटाने में टेस्ट डिवाइस पर 3.3 सेकंड लगे। यह आपके डिवाइस की CPU गति, RAM, SSD की उपस्थिति, और अन्य कारकों पर निर्भर करता है कि इसमें थोड़ा अधिक या कम समय लग सकता है। ऑपरेशन की सापेक्ष छोटी अवधि का मतलब है कि यदि यह भाव विश्लेषण समय में सुधार करता है, तो यह करने लायक है।

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

### भाव विश्लेषण करना
अब आपको नकारात्मक और सकारात्मक समीक्षा कॉलम के लिए भाव विश्लेषण की गणना करनी चाहिए और परिणाम को 2 नए कॉलम में संग्रहीत करना चाहिए। भाव का परीक्षण समीक्षक के स्कोर के साथ तुलना करना होगा जो उसी समीक्षा के लिए दिया गया है। उदाहरण के लिए, यदि भाव विश्लेषण यह मानता है कि नकारात्मक समीक्षा का भाव 1 (अत्यधिक सकारात्मक भाव) है और सकारात्मक समीक्षा का भाव भी 1 है, लेकिन समीक्षक ने होटल को सबसे कम स्कोर दिया है, तो या तो समीक्षा का टेक्स्ट स्कोर से मेल नहीं खाता है, या भाव विश्लेषक भाव को सही ढंग से पहचानने में असमर्थ हो सकता है। आपको उम्मीद करनी चाहिए कि कुछ भाव स्कोर पूरी तरह से गलत होंगे, और अक्सर इसका कारण समझाया जा सकता है, जैसे कि समीक्षा अत्यधिक व्यंग्यात्मक हो सकती है "बिल्कुल, मुझे बिना हीटिंग वाले कमरे में सोना बहुत पसंद आया" और भाव विश्लेषक इसे सकारात्मक भाव मानता है, जबकि एक इंसान इसे पढ़कर समझ जाएगा कि यह व्यंग्य है।

NLTK विभिन्न भाव विश्लेषकों को सीखने के लिए प्रदान करता है, और आप उन्हें बदल सकते हैं और देख सकते हैं कि भाव अधिक या कम सटीक है। यहां VADER भाव विश्लेषण का उपयोग किया गया है।

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

बाद में जब आप अपने प्रोग्राम में भाव की गणना करने के लिए तैयार हों, तो आप इसे प्रत्येक समीक्षा पर इस प्रकार लागू कर सकते हैं:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

यह मेरे कंप्यूटर पर लगभग 120 सेकंड लेता है, लेकिन यह प्रत्येक कंप्यूटर पर अलग-अलग हो सकता है। यदि आप परिणामों को प्रिंट करना चाहते हैं और देखना चाहते हैं कि भाव समीक्षा से मेल खाता है या नहीं:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

फाइल का उपयोग करने से पहले अंतिम काम यह है कि इसे सेव करें! आपको अपने नए कॉलम को पुनः व्यवस्थित करने पर भी विचार करना चाहिए ताकि वे काम करने में आसान हों (यह एक सौंदर्यात्मक बदलाव है)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

आपको [विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) के पूरे कोड को चलाना चाहिए (उसके बाद जब आपने [फिल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) को चलाया है ताकि Hotel_Reviews_Filtered.csv फाइल उत्पन्न हो सके)।

समीक्षा करने के लिए, चरण हैं:

1. मूल डेटा सेट फाइल **Hotel_Reviews.csv** को पिछले पाठ में [एक्सप्लोरर नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) के साथ एक्सप्लोर किया गया है।
2. Hotel_Reviews.csv को [फिल्टरिंग नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) द्वारा फिल्टर किया गया है, जिससे **Hotel_Reviews_Filtered.csv** प्राप्त होता है।
3. Hotel_Reviews_Filtered.csv को [भाव विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) द्वारा प्रोसेस किया गया है, जिससे **Hotel_Reviews_NLP.csv** प्राप्त होता है।
4. NLP चैलेंज में नीचे **Hotel_Reviews_NLP.csv** का उपयोग करें।

### निष्कर्ष

जब आपने शुरुआत की थी, आपके पास कॉलम और डेटा वाला एक डेटा सेट था लेकिन इसका सारा डेटा सत्यापित या उपयोग नहीं किया जा सकता था। आपने डेटा को एक्सप्लोर किया, जो आवश्यक नहीं था उसे फिल्टर किया, टैग्स को उपयोगी चीजों में बदला, अपने औसत की गणना की, कुछ भाव कॉलम जोड़े और उम्मीद है कि प्राकृतिक टेक्स्ट को प्रोसेस करने के बारे में कुछ दिलचस्प बातें सीखी होंगी।

## [पोस्ट-लेक्चर क्विज़](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## चैलेंज

अब जब आपने अपने डेटा सेट का भाव विश्लेषण कर लिया है, तो देखें कि क्या आप इस पाठ्यक्रम में सीखी गई रणनीतियों (शायद क्लस्टरिंग?) का उपयोग करके भाव के आसपास पैटर्न निर्धारित कर सकते हैं।

## समीक्षा और स्व-अध्ययन

[इस Learn मॉड्यूल](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) को लें ताकि आप अधिक सीख सकें और टेक्स्ट में भाव को एक्सप्लोर करने के लिए विभिन्न टूल्स का उपयोग कर सकें।

## असाइनमेंट 

[एक अलग डेटा सेट आज़माएं](assignment.md)

---

**अस्वीकरण**:  
यह दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियां या अशुद्धियां हो सकती हैं। मूल भाषा में उपलब्ध मूल दस्तावेज़ को आधिकारिक स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।  