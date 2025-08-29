<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T18:39:29+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ne"
}
-->
# होटल समीक्षाहरूको साथ भावना विश्लेषण

अब तपाईंले डेटासेटलाई विस्तारमा अन्वेषण गरिसकेपछि, स्तम्भहरू फिल्टर गर्ने र त्यसपछि होटलहरूको बारेमा नयाँ जानकारी प्राप्त गर्न डेटासेटमा NLP प्रविधिहरू प्रयोग गर्ने समय आएको छ।  
## [पूर्व-व्याख्यान प्रश्नोत्तरी](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### फिल्टरिङ र भावना विश्लेषण कार्यहरू

जसरी तपाईंले सम्भवतः ध्यान दिनुभएको छ, डेटासेटमा केही समस्या छन्। केही स्तम्भहरू अनावश्यक जानकारीले भरिएका छन्, अरू केही गलत देखिन्छन्। यदि ती सही छन् भने पनि, ती कसरी गणना गरिएका थिए भन्ने स्पष्ट छैन, र तपाईंको आफ्नै गणनाहरूले उत्तरहरू स्वतन्त्र रूपमा प्रमाणित गर्न सकिँदैन।

## अभ्यास: अझै केही डेटा प्रशोधन

डेटालाई अझै थोरै सफा गर्नुहोस्। पछि उपयोगी हुने स्तम्भहरू थप्नुहोस्, अन्य स्तम्भहरूमा रहेका मानहरू परिवर्तन गर्नुहोस्, र केही स्तम्भहरू पूर्ण रूपमा हटाउनुहोस्।

1. प्रारम्भिक स्तम्भ प्रशोधन

   1. `lat` र `lng` हटाउनुहोस्।

   2. `Hotel_Address` का मानहरू निम्न मानहरूसँग प्रतिस्थापन गर्नुहोस् (यदि ठेगानामा शहर र देशको नाम समावेश छ भने, त्यसलाई केवल शहर र देशमा परिवर्तन गर्नुहोस्)।

      डेटासेटमा रहेका एकमात्र शहर र देशहरू यिनीहरू हुन्:

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

      अब तपाईं देश स्तरको डेटा सोधपुछ गर्न सक्नुहुन्छ:

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

2. होटल मेटा-समीक्षा स्तम्भहरू प्रशोधन गर्नुहोस्

   1. `Additional_Number_of_Scoring` हटाउनुहोस्।

   2. `Total_Number_of_Reviews` लाई डेटासेटमा वास्तवमै भएका समीक्षाहरूको कुल संख्याले प्रतिस्थापन गर्नुहोस्।

   3. `Average_Score` लाई हाम्रो आफ्नै गणना गरिएको स्कोरले प्रतिस्थापन गर्नुहोस्।

      ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. समीक्षा स्तम्भहरू प्रशोधन गर्नुहोस्

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` र `days_since_review` हटाउनुहोस्।

   2. `Reviewer_Score`, `Negative_Review`, र `Positive_Review` जस्ताको तस्तै राख्नुहोस्।

   3. `Tags` लाई अहिलेका लागि राख्नुहोस्।

      - हामी अर्को खण्डमा ट्यागहरूमा केही अतिरिक्त फिल्टरिङ कार्यहरू गर्नेछौं र त्यसपछि ट्यागहरू हटाइनेछ।

4. समीक्षक स्तम्भहरू प्रशोधन गर्नुहोस्

   1. `Total_Number_of_Reviews_Reviewer_Has_Given` हटाउनुहोस्।

   2. `Reviewer_Nationality` राख्नुहोस्।

### ट्याग स्तम्भहरू

`Tag` स्तम्भ समस्याग्रस्त छ किनभने यो स्तम्भमा सूची (पाठ रूपमा) भण्डारण गरिएको छ। दुर्भाग्यवश, यस स्तम्भमा उपविभागहरूको क्रम र संख्या सधैं एउटै हुँदैन। 515,000 पङ्क्तिहरू, 1427 होटलहरू, र प्रत्येक समीक्षकले छनोट गर्न सक्ने विकल्पहरू अलिकति फरक हुने हुँदा, सही वाक्यांशहरू पहिचान गर्न मानिसका लागि यो कठिन हुन्छ। यहीं NLP उपयोगी हुन्छ। तपाईं पाठ स्क्यान गर्न सक्नुहुन्छ र सबैभन्दा सामान्य वाक्यांशहरू फेला पार्न सक्नुहुन्छ, र तिनीहरूको गणना गर्न सक्नुहुन्छ।

दुर्भाग्यवश, हामीलाई एकल शब्दहरूमा होइन, बहु-शब्द वाक्यांशहरूमा रुचि छ (जस्तै *Business trip*)। यति धेरै डेटामा (6762646 शब्दहरू) बहु-शब्द आवृत्ति वितरण एल्गोरिदम चलाउनु असाधारण रूपमा धेरै समय लाग्न सक्छ, तर डेटा नहेरीकन, यो आवश्यक खर्च जस्तो देखिन्छ। यहीं अन्वेषणात्मक डेटा विश्लेषण उपयोगी हुन्छ, किनभने तपाईंले ट्यागहरूको नमूना देख्नुभएको छ जस्तै `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, तपाईंले सोध्न थाल्न सक्नुहुन्छ कि तपाईंले गर्नुपर्ने प्रशोधनलाई धेरै कम गर्न सम्भव छ कि छैन। भाग्यवश, यो सम्भव छ - तर पहिले तपाईंले केही चरणहरू पालना गर्न आवश्यक छ।

### ट्यागहरू फिल्टर गर्दै

याद गर्नुहोस् कि डेटासेटको लक्ष्य भावना र स्तम्भहरू थप्नु हो जसले तपाईंलाई उत्कृष्ट होटल छनोट गर्न मद्दत गर्दछ (आफ्नो लागि वा सायद तपाईंलाई होटल सिफारिस बोट बनाउन कार्य दिने ग्राहकको लागि)। तपाईंले सोध्न आवश्यक छ कि ट्यागहरू अन्तिम डेटासेटमा उपयोगी छन् कि छैनन्। यहाँ एउटा व्याख्या छ (यदि तपाईंलाई अन्य कारणहरूको लागि डेटासेट चाहिएको भए, फरक ट्यागहरू चयनमा रहन/नहुन सक्छन्):

1. यात्राको प्रकार सान्दर्भिक छ, र यो रहनुपर्छ।  
2. पाहुनाको समूहको प्रकार महत्त्वपूर्ण छ, र यो रहनुपर्छ।  
3. पाहुनाले बसेको कोठा, सुइट, वा स्टुडियोको प्रकार अप्रासंगिक छ (सबै होटलहरूमा प्रायः उस्तै कोठाहरू हुन्छन्)।  
4. समीक्षाहरू पठाउन प्रयोग गरिएको उपकरण अप्रासंगिक छ।  
5. समीक्षकले बसेको रातहरूको संख्या *महत्त्वपूर्ण* हुन सक्छ यदि तपाईंले लामो बसाइलाई होटल मन पराउनेसँग सम्बन्धित ठान्नुभयो भने, तर यो अलिकति तन्काइएको कुरा हो, र सम्भवतः अप्रासंगिक छ।  

सारांशमा, **2 प्रकारका ट्यागहरू राख्नुहोस् र अरू हटाउनुहोस्।**

पहिले, तपाईंले ट्यागहरू राम्रो स्वरूपमा नआएसम्म गणना गर्न चाहनुहुन्न, त्यसैले यसको मतलब वर्ग कोष्ठक र उद्धरणहरू हटाउनु हो। तपाईंले यो धेरै तरिकाहरूले गर्न सक्नुहुन्छ, तर तपाईंलाई सबैभन्दा छिटो चाहिन्छ किनभने धेरै डेटा प्रशोधन गर्न लामो समय लाग्न सक्छ। भाग्यवश, pandas ले यी चरणहरू गर्न सजिलो तरिका प्रदान गर्दछ।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

प्रत्येक ट्याग यस्तो हुन्छ: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`।  

अर्को समस्या देखिन्छ। केही समीक्षाहरू, वा पङ्क्तिहरू, 5 स्तम्भहरू छन्, केही 3, केही 6। यो डेटासेट कसरी सिर्जना गरियो भन्ने परिणाम हो, र समाधान गर्न गाह्रो छ। तपाईं प्रत्येक वाक्यांशको आवृत्ति गणना गर्न चाहनुहुन्छ, तर तिनीहरू प्रत्येक समीक्षामा फरक क्रममा छन्, त्यसैले गणना गलत हुन सक्छ, र कुनै होटलले आफूले योग्य ट्याग प्राप्त नगर्न सक्छ।  

यसको सट्टा तपाईंले फरक क्रमलाई आफ्नो फाइदामा प्रयोग गर्नुहुनेछ, किनभने प्रत्येक ट्याग बहु-शब्द हो तर अल्पविरामले पनि छुट्याइएको छ! सबैभन्दा सरल तरिका भनेको 6 अस्थायी स्तम्भहरू सिर्जना गर्नु हो, प्रत्येक ट्यागलाई यसको क्रम अनुसार स्तम्भमा राख्नु। त्यसपछि तपाईं 6 स्तम्भहरूलाई एक ठूला स्तम्भमा मर्ज गर्न सक्नुहुन्छ र `value_counts()` विधि चलाउन सक्नुहुन्छ। यसलाई प्रिन्ट गर्दा, तपाईंले देख्नुहुनेछ कि 2428 अद्वितीय ट्यागहरू थिए। यहाँ एउटा सानो नमूना छ:

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

केही सामान्य ट्यागहरू जस्तै `Submitted from a mobile device` हाम्रो लागि उपयोगी छैनन्, त्यसैले तिनीहरूलाई वाक्यांशको आवृत्ति गणना गर्नु अघि हटाउनु बुद्धिमानी हुन सक्छ, तर यो यति छिटो कार्य हो कि तपाईं तिनीहरूलाई राख्न सक्नुहुन्छ र बेवास्ता गर्न सक्नुहुन्छ।

### बसाइको अवधि ट्यागहरू हटाउँदै

यी ट्यागहरू हटाउनु पहिलो चरण हो, यसले विचार गर्नुपर्ने ट्यागहरूको कुल सङ्ख्या थोरै घटाउँछ। ध्यान दिनुहोस् कि तपाईंले तिनीहरूलाई डेटासेटबाट हटाउनुहुन्न, केवल समीक्षाहरूको डेटासेटमा मानहरू गणना/राख्न विचार नगर्नुहोस्।

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

कोठाहरू, सुइटहरू, स्टुडियोहरू, अपार्टमेन्टहरू आदि जस्ता कोठाहरूको विविधता धेरै छ। तिनीहरू सबैले लगभग उस्तै कुरा जनाउँछन् र तपाईंका लागि सान्दर्भिक छैनन्, त्यसैले तिनीहरूलाई विचारबाट हटाउनुहोस्।

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

अन्ततः, र यो रमाइलो छ (किनभने यसले धेरै प्रशोधन लिएको छैन), तपाईं निम्न *उपयोगी* ट्यागहरूसँग बाँकी रहनेछ:

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

तपाईंले `Travellers with friends` लाई `Group` जस्तै हो भनेर तर्क गर्न सक्नुहुन्छ, र माथिको रूपमा दुईलाई संयोजन गर्नु उचित हुनेछ। सही ट्यागहरू पहिचान गर्ने कोड [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) मा छ।

अन्तिम चरण भनेको यी प्रत्येक ट्यागहरूको लागि नयाँ स्तम्भहरू सिर्जना गर्नु हो। त्यसपछि, प्रत्येक समीक्षा पङ्क्तिका लागि, यदि `Tag` स्तम्भ नयाँ स्तम्भहरूसँग मेल खान्छ भने, 1 थप्नुहोस्, यदि मेल खाँदैन भने, 0 थप्नुहोस्। अन्तिम परिणामले यो जानकारी दिनेछ कि कति समीक्षकहरूले यो होटल (कुलमा) व्यवसाय बनाम मनोरञ्जनका लागि, वा पालतू जनावर ल्याउन छनोट गरे, र यो होटल सिफारिस गर्दा उपयोगी जानकारी हो।

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

### आफ्नो फाइल सुरक्षित गर्नुहोस्

अन्ततः, डेटासेटलाई अहिलेको रूपमा नयाँ नाममा सुरक्षित गर्नुहोस्।

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## भावना विश्लेषण कार्यहरू

यस अन्तिम खण्डमा, तपाईं समीक्षा स्तम्भहरूमा भावना विश्लेषण लागू गर्नुहुनेछ र परिणामहरू डेटासेटमा सुरक्षित गर्नुहुनेछ।

## अभ्यास: फिल्टर गरिएको डेटा लोड गर्नुहोस् र सुरक्षित गर्नुहोस्

ध्यान दिनुहोस् कि अब तपाईंले अघिल्लो खण्डमा सुरक्षित गरिएको फिल्टर गरिएको डेटासेट लोड गर्दै हुनुहुन्छ, **मूल** डेटासेट होइन।

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

### स्टप शब्दहरू हटाउँदै

यदि तपाईंले नकारात्मक र सकारात्मक समीक्षा स्तम्भहरूमा भावना विश्लेषण चलाउनुभयो भने, यसले धेरै समय लिन सक्छ। शक्तिशाली परीक्षण ल्यापटपमा परीक्षण गर्दा, यसले प्रयोग गरिएको भावना पुस्तकालयमा निर्भर गर्दै 12 - 14 मिनेट लियो। यो (सापेक्ष रूपमा) लामो समय हो, त्यसैले यसलाई छिटो बनाउन सकिन्छ कि भनेर अनुसन्धान गर्न लायक छ।  

सामान्य अंग्रेजी शब्दहरू, जसले वाक्यको भावनालाई परिवर्तन गर्दैनन्, हटाउनु पहिलो चरण हो। तिनीहरू हटाउँदा, भावना विश्लेषण छिटो चल्नुपर्छ, तर कम सटीक हुनु हुँदैन (किनभने स्टप शब्दहरूले भावना प्रभावित गर्दैनन्, तर तिनीहरूले विश्लेषणलाई ढिलो बनाउँछन्)।  

सबैभन्दा लामो नकारात्मक समीक्षा 395 शब्दको थियो, तर स्टप शब्दहरू हटाएपछि, यो 195 शब्दको भयो।  

स्टप शब्दहरू हटाउनु पनि छिटो कार्य हो, 515,000 पङ्क्तिहरूमा 2 समीक्षा स्तम्भहरूबाट स्टप शब्दहरू हटाउन परीक्षण उपकरणमा 3.3 सेकेन्ड लाग्यो। तपाईंको उपकरणको CPU गति, RAM, SSD छ कि छैन, र अन्य केही कारकहरूमा निर्भर गर्दै यसले तपाईंलाई अलिकति बढी वा कम समय लिन सक्छ। कार्यको सापेक्ष छोटो समयले यो भावना विश्लेषण समय सुधार्छ भने, यो गर्न लायक छ।

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

### भावना विश्लेषण गर्दै
अब तपाईंले नकारात्मक र सकारात्मक समीक्षा स्तम्भहरूको भावना विश्लेषण गणना गर्नुपर्छ, र परिणामलाई २ नयाँ स्तम्भहरूमा भण्डारण गर्नुपर्छ। भावना परीक्षण समीक्षकको स्कोरसँग तुलना गरेर गरिनेछ। उदाहरणका लागि, यदि भावना विश्लेषकले नकारात्मक समीक्षालाई १ (अत्यन्त सकारात्मक भावना) र सकारात्मक समीक्षालाई १ को भावना दिएको छ, तर समीक्षकले होटललाई सबैभन्दा कम स्कोर दिएको छ भने, समीक्षा पाठ स्कोरसँग मेल खाँदैन वा भावना विश्लेषकले सही रूपमा भावना चिन्न सकेन। तपाईंले केही भावना स्कोरहरू पूर्ण रूपमा गलत हुने अपेक्षा गर्नुपर्छ, र प्रायः यो व्याख्या गर्न सकिनेछ, जस्तै समीक्षा अत्यन्त व्यंग्यात्मक हुन सक्छ "पक्कै पनि म बिना तातोको कोठामा सुत्न मन पराउँछु" र भावना विश्लेषकले यसलाई सकारात्मक भावना ठान्छ, यद्यपि मानिसले पढ्दा यो व्यंग्य हो भन्ने थाहा पाउँछ।

NLTK ले सिक्नका लागि विभिन्न भावना विश्लेषकहरू प्रदान गर्दछ, र तपाईंले तिनीहरूलाई प्रतिस्थापन गर्न सक्नुहुन्छ र भावना बढी वा कम सही छ कि छैन हेर्न सक्नुहुन्छ। यहाँ VADER भावना विश्लेषण प्रयोग गरिएको छ।

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: सामाजिक मिडिया पाठको भावना विश्लेषणको लागि एक सरल नियम-आधारित मोडेल। आठौं अन्तर्राष्ट्रिय सम्मेलन वेब्लग्स र सामाजिक मिडिया (ICWSM-14)। एन आर्बर, MI, जून 2014।

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

तपाईंको प्रोग्राममा भावना गणना गर्न तयार हुँदा, तपाईं यसलाई प्रत्येक समीक्षामा निम्नानुसार लागू गर्न सक्नुहुन्छ:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

यसले मेरो कम्प्युटरमा लगभग १२० सेकेन्ड लिन्छ, तर प्रत्येक कम्प्युटरमा फरक हुनेछ। यदि तपाईं परिणामहरू प्रिन्ट गर्न चाहनुहुन्छ र भावना समीक्षासँग मेल खाँदछ कि छैन हेर्न चाहनुहुन्छ भने:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

चुनौतीमा प्रयोग गर्नु अघि फाइलसँग गर्नुपर्ने अन्तिम कुरा भनेको यसलाई बचत गर्नु हो! तपाईंले सबै नयाँ स्तम्भहरू पुन: क्रमबद्ध गर्न विचार गर्नुपर्छ ताकि तिनीहरू काम गर्न सजिलो होस् (मानिसका लागि, यो एक सौन्दर्यात्मक परिवर्तन हो)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

तपाईंले [विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) को सम्पूर्ण कोड चलाउनुपर्छ (पछाडि [फिल्टरिङ नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) चलाएपछि Hotel_Reviews_Filtered.csv फाइल उत्पन्न गर्न)।

समीक्षा गर्न, चरणहरू हुन्:

1. मूल डेटासेट फाइल **Hotel_Reviews.csv** अघिल्लो पाठमा [एक्सप्लोरर नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) द्वारा अन्वेषण गरिएको छ।
2. Hotel_Reviews.csv [फिल्टरिङ नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) द्वारा फिल्टर गरिएको छ जसको परिणामस्वरूप **Hotel_Reviews_Filtered.csv** प्राप्त हुन्छ।
3. Hotel_Reviews_Filtered.csv [भावना विश्लेषण नोटबुक](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) द्वारा प्रशोधन गरिएको छ जसको परिणामस्वरूप **Hotel_Reviews_NLP.csv** प्राप्त हुन्छ।
4. NLP चुनौतीमा **Hotel_Reviews_NLP.csv** प्रयोग गर्नुहोस्।

### निष्कर्ष

जब तपाईंले सुरु गर्नुभयो, तपाईंसँग स्तम्भहरू र डाटा भएको डेटासेट थियो तर सबैलाई प्रमाणित वा प्रयोग गर्न सकिने थिएन। तपाईंले डाटालाई अन्वेषण गर्नुभयो, आवश्यक नभएको कुरा फिल्टर गर्नुभयो, ट्यागहरू उपयोगी चीजमा रूपान्तरण गर्नुभयो, आफ्नै औसतहरू गणना गर्नुभयो, केही भावना स्तम्भहरू थप्नुभयो र आशा छ, प्राकृतिक पाठ प्रशोधनको बारेमा केही रोचक कुरा सिक्नुभयो।

## [पाठ-पछिको क्विज](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## चुनौती

अब तपाईंको डेटासेटलाई भावना विश्लेषणका लागि विश्लेषण गरिएको छ, तपाईंले यस पाठ्यक्रममा सिकेका रणनीतिहरू (शायद क्लस्टरिङ?) प्रयोग गरेर भावना वरिपरि ढाँचाहरू निर्धारण गर्न सक्नुहुन्छ।

## समीक्षा र आत्म अध्ययन

[यो सिक्ने मोड्युल](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) लिनुहोस् थप जान्न र पाठमा भावना अन्वेषण गर्न विभिन्न उपकरणहरू प्रयोग गर्न।

## असाइनमेन्ट 

[अर्को डेटासेट प्रयास गर्नुहोस्](assignment.md)

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।