<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-10-11T11:31:51+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "ta"
}
-->
# ஹோட்டல் விமர்சனங்களுடன் உணர்வு பகுப்பாய்வு

நீங்கள் தரவுத்தொகுப்பை விரிவாக ஆராய்ந்த பிறகு, களங்களை வடிகட்டவும், பின்னர் ஹோட்டல்களைப் பற்றிய புதிய தகவல்களைப் பெறுவதற்காக NLP தொழில்நுட்பங்களைப் பயன்படுத்தவும் நேரம் வந்துள்ளது.

## [முன்-வகுப்பு வினாடி வினா](https://ff-quizzes.netlify.app/en/ml/)

### வடிகட்டல் மற்றும் உணர்வு பகுப்பாய்வு செயல்பாடுகள்

நீங்கள் கவனித்திருப்பீர்கள் போல, தரவுத்தொகுப்பில் சில சிக்கல்கள் உள்ளன. சில களங்கள் பயனற்ற தகவல்களால் நிரப்பப்பட்டுள்ளன, மற்றவை தவறாக தோன்றுகின்றன. அவை சரியாக இருந்தால், அவை எவ்வாறு கணக்கிடப்பட்டன என்பது தெளிவாக இல்லை, மேலும் உங்கள் சொந்த கணக்கீடுகளால் பதில்களை சுயமாக சரிபார்க்க முடியாது.

## பயிற்சி: மேலும் சில தரவுகளை செயல்படுத்துதல்

தரவை மேலும் சுத்தமாக்கவும். பின்னர் பயன்படக்கூடிய களங்களைச் சேர்க்கவும், பிற களங்களில் உள்ள மதிப்புகளை மாற்றவும், குறிப்பிட்ட களங்களை முழுமையாக நீக்கவும்.

1. ஆரம்ப கள செயல்பாடு

   1. `lat` மற்றும் `lng` களை நீக்கவும்

   2. `Hotel_Address` மதிப்புகளை பின்வரும் மதிப்புகளுடன் மாற்றவும் (முகவரியில் நகரம் மற்றும் நாடு பெயர் உள்ளதானால், அதை நகரம் மற்றும் நாடு மட்டுமாக மாற்றவும்).

      தரவுத்தொகுப்பில் உள்ள நகரங்கள் மற்றும் நாடுகள் இவை மட்டுமே:

      ஆம்ஸ்டர்டாம், நெதர்லாந்து

      பார்சிலோனா, ஸ்பெயின்

      லண்டன், ஐக்கிய இராச்சியம்

      மிலன், இத்தாலி

      பாரிஸ், பிரான்ஸ்

      வியன்னா, ஆஸ்திரியா 

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

      இப்போது நீங்கள் நாடு மட்டத்திலான தரவுகளை கேட்கலாம்:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | ஆம்ஸ்டர்டாம், நெதர்லாந்து |    105     |
      | பார்சிலோனா, ஸ்பெயின்       |    211     |
      | லண்டன், ஐக்கிய இராச்சியம் |    400     |
      | மிலன், இத்தாலி           |    162     |
      | பாரிஸ், பிரான்ஸ்          |    458     |
      | வியன்னா, ஆஸ்திரியா        |    158     |

2. ஹோட்டல் மெட்டா-விமர்சன களங்களை செயல்படுத்துதல்

  1. `Additional_Number_of_Scoring` களை நீக்கவும்

  1. `Total_Number_of_Reviews` களை தரவுத்தொகுப்பில் உண்மையில் உள்ள ஹோட்டலின் மொத்த விமர்சனங்களுடன் மாற்றவும் 

  1. `Average_Score` ஐ நாங்கள் கணக்கிட்ட மதிப்புடன் மாற்றவும்

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. விமர்சன களங்களை செயல்படுத்துதல்

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` மற்றும் `days_since_review` களை நீக்கவும்

   2. `Reviewer_Score`, `Negative_Review`, மற்றும் `Positive_Review` களை அப்படியே வைத்திருக்கவும்
     
   3. `Tags` களை தற்காலிகமாக வைத்திருக்கவும்

     - அடுத்த பகுதியில் சில கூடுதல் வடிகட்டல் செயல்பாடுகளை `Tags` களில் செய்ய வேண்டும், பின்னர் `Tags` களை நீக்கப்படும்

4. விமர்சகர் களங்களை செயல்படுத்துதல்

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` களை நீக்கவும்
  
  2. `Reviewer_Nationality` களை வைத்திருக்கவும்

### Tag களங்கள்

`Tag` களம் ஒரு பட்டியல் (உரை வடிவத்தில்) களத்தில் சேமிக்கப்பட்டுள்ளது என்பதால் சிக்கலாக உள்ளது. துரதிருஷ்டவசமாக, இந்த களத்தில் உள்ள துணை பிரிவுகளின் வரிசையும் எண்ணிக்கையும் எப்போதும் ஒரே மாதிரியானவை அல்ல. ஒரு மனிதனுக்கு சரியான சொற்றொடர்களை அடையாளம் காண்பது கடினமாக இருக்கும், ஏனெனில் 515,000 வரிசைகள் மற்றும் 1427 ஹோட்டல்கள் உள்ளன, மேலும் ஒவ்வொன்றும் விமர்சகர் தேர்வு செய்யக்கூடிய slightly வேறுபட்ட விருப்பங்களைக் கொண்டுள்ளது. இங்கு NLP உதவுகிறது. நீங்கள் உரையை ஸ்கேன் செய்து மிகவும் பொதுவான சொற்றொடர்களைக் கண்டறிந்து அவற்றை எண்ணலாம்.

துரதிருஷ்டவசமாக, தனித்த சொற்களில் நாங்கள் ஆர்வமாக இல்லை, ஆனால் பல சொற்களைக் கொண்ட சொற்றொடர்கள் (எ.கா. *Business trip*). அந்த அளவிலான தரவுகளில் (6762646 சொற்கள்) ஒரு பல சொற்களுக்கான அதிரடி விநியோகக் குருவை இயக்குவது மிகவும் அதிக நேரத்தை எடுத்துக்கொள்ளக்கூடும், ஆனால் தரவுகளைப் பார்க்காமல், அது ஒரு தேவையான செலவாகத் தோன்றுகிறது. இது ஆராய்ச்சி தரவுப் பகுப்பாய்வு பயனுள்ளதாக இருக்கும் இடம், ஏனெனில் நீங்கள் `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` போன்ற `Tags` களின் மாதிரியைப் பார்த்துள்ளீர்கள், நீங்கள் அதைச் செய்ய முடியும் என்று கேட்கத் தொடங்கலாம். செயலாக்கத்தை மிகவும் குறைக்க முடியும். அதிர்ஷ்டவசமாக, அது முடியும் - ஆனால் முதலில் நீங்கள் ஆர்வமான `Tags` களை உறுதிப்படுத்த சில படிகளைப் பின்பற்ற வேண்டும்.

### Tag களை வடிகட்டுதல்

தரவுத்தொகுப்பின் நோக்கம் உங்களுக்காக அல்லது ஹோட்டல் பரிந்துரை பாட்டை உருவாக்க உங்களை பணியமர்த்தும் ஒரு வாடிக்கையாளருக்காக சிறந்த ஹோட்டலைத் தேர்ந்தெடுக்க உதவும் உணர்வு மற்றும் களங்களைச் சேர்ப்பது என்பதை நினைவில் கொள்ளுங்கள். `Tags` கள் இறுதி தரவுத்தொகுப்பில் பயனுள்ளதாக உள்ளதா என்பதை நீங்கள் உங்களுக்கே கேட்க வேண்டும். இதோ ஒரு விளக்கம் (உங்களுக்கு தரவுத்தொகுப்பு வேறு காரணங்களுக்காக தேவைப்பட்டால், வெவ்வேறு `Tags` உள்ள/வெளியே தேர்ந்தெடுக்கப்படலாம்):

1. பயணத்தின் வகை தொடர்புடையது, அது இருக்க வேண்டும்
2. விருந்தினரின் குழுவின் வகை முக்கியமானது, அது இருக்க வேண்டும்
3. விருந்தினர் தங்கிய அறை, ஸ்யூட் அல்லது ஸ்டுடியோவின் வகை தொடர்புடையது அல்ல (அனைத்து ஹோட்டல்களிலும் அடிப்படையில் ஒரே மாதிரியான அறைகள் உள்ளன)
4. விமர்சனம் சமர்ப்பிக்கப்பட்ட சாதனம் தொடர்புடையது அல்ல
5. விமர்சகர் தங்கிய இரவுகளின் எண்ணிக்கை *தொடர்புடையதாக* இருக்கலாம், ஆனால் அது ஒரு stretch ஆகும், மற்றும் பொதுவாக தொடர்புடையது அல்ல

சுருக்கமாக, **2 வகையான `Tags` களை வைத்திருக்கவும், மற்றவற்றை நீக்கவும்**.

முதலில், `Tags` களை சிறந்த வடிவத்தில் மாற்றும் வரை நீங்கள் அவற்றை எண்ண விரும்பவில்லை, அதாவது சதுர கோடுகள் மற்றும் மேற்கோள்களை நீக்க வேண்டும். இதைச் செய்ய பல வழிகள் உள்ளன, ஆனால் நீங்கள் விரைவான முறையை விரும்புகிறீர்கள், ஏனெனில் அதிக அளவிலான தரவுகளை செயல்படுத்த இது நீண்ட நேரம் எடுத்துக்கொள்ளக்கூடும். அதிர்ஷ்டவசமாக, pandas இவற்றில் ஒவ்வொன்றையும் செய்ய எளிய வழியை வழங்குகிறது.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ஒவ்வொரு `Tag` களும் இவ்வாறு மாறும்: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

அடுத்ததாக ஒரு சிக்கல் வருகிறது. சில விமர்சனங்கள் அல்லது வரிசைகள் 5 களங்கள் கொண்டவை, சில 3, சில 6. இது தரவுத்தொகுப்பு உருவாக்கப்பட்ட விதத்தின் விளைவு, மற்றும் சரிசெய்வது கடினம். ஒவ்வொரு சொற்றொடரின் அதிரடி எண்ணிக்கையைப் பெற விரும்புகிறீர்கள், ஆனால் அவை ஒவ்வொரு விமர்சனத்திலும் வெவ்வேறு வரிசையில் உள்ளன, எனவே எண்ணிக்கை தவறாக இருக்கலாம், மேலும் ஒரு ஹோட்டலுக்கு அது தகுதியுடைய `Tag` களை வழங்க முடியாது.

அதற்குப் பதிலாக, நீங்கள் வெவ்வேறு வரிசையை நன்மையாகப் பயன்படுத்துவீர்கள், ஏனெனில் ஒவ்வொரு `Tag` களும் பல சொற்களைக் கொண்டது, ஆனால் கமாவால் பிரிக்கப்பட்டுள்ளது! இதைச் செய்ய மிக எளிமையான வழி, ஒவ்வொரு `Tag` ஐ அதன் வரிசைக்கு ஏற்ப களத்தில் சேர்த்து 6 தற்காலிக களங்களை உருவாக்குவது. பின்னர் நீங்கள் 6 களங்களை ஒரு பெரிய களத்தில் இணைத்து, resulting களத்தில் `value_counts()` முறை இயக்கலாம். அதை அச்சிடும்போது, 2428 தனித்த `Tags` கள் இருந்தன என்பதை நீங்கள் காண்பீர்கள். இதோ ஒரு சிறிய மாதிரி:

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

`Submitted from a mobile device` போன்ற பொதுவான `Tags` கள் எங்களுக்கு எந்த பயனும் இல்லை, எனவே அவற்றை phrase occurrence ஐ எண்ணுவதற்கு முன் நீக்குவது புத்திசாலித்தனமானது, ஆனால் இது மிகவும் விரைவான செயல்பாடு என்பதால் அவற்றை உள்ளே வைத்திருக்கலாம் மற்றும் அவற்றை புறக்கணிக்கலாம்.

### தங்கிய இரவுகளின் `Tags` களை நீக்குதல்

இந்த `Tags` களை நீக்குவது முதல் படி, இது கருதப்படும் `Tags` களின் மொத்த எண்ணிக்கையை சற்று குறைக்கிறது. நீங்கள் அவற்றை தரவுத்தொகுப்பில் இருந்து நீக்கவில்லை என்பதை கவனிக்கவும், விமர்சனங்கள் தரவுத்தொகுப்பில் மதிப்புகளாக எண்ண/வைத்திருக்க தேர்வு செய்யவில்லை.

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

அறைகள், ஸ்யூட்கள், ஸ்டுடியோக்கள், அபார்ட்மெண்ட்கள் மற்றும் பலவற்றின் வகைகள் மிகவும் பரந்தவை. அவை அனைத்தும் அடிப்படையில் ஒரே பொருளைத் தருகின்றன மற்றும் உங்களுக்கு தொடர்புடையவை அல்ல, எனவே அவற்றை கருதலிலிருந்து நீக்கவும்.

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

இறுதியாக, இது மகிழ்ச்சியானது (ஏனெனில் இது அதிக செயலாக்கத்தை எடுத்துக்கொள்ளவில்லை), நீங்கள் பின்வரும் *பயனுள்ள* `Tags` களுடன் இருக்கும்:

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

`Travellers with friends` என்பது `Group` உடன் ஒரே மாதிரியானது என்று நீங்கள் வாதிடலாம், மேலும் இரண்டு ஒன்றாக இணைக்கப்பட்டிருப்பது நியாயமானது. சரியான `Tags` களை அடையாளம் காணும் குறியீடு [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ஆகும்.

இறுதி படி, இந்த புதிய `Tags` களுக்கான புதிய களங்களை உருவாக்குவது. பின்னர், ஒவ்வொரு விமர்சன வரிசைக்கும், `Tag` களம் புதிய களங்களில் ஒன்றுடன் பொருந்தினால், 1 ஐச் சேர்க்கவும், பொருந்தவில்லை என்றால், 0 ஐச் சேர்க்கவும். இறுதி முடிவாக, இந்த ஹோட்டலை (மொத்தத்தில்) வணிகம் vs பொழுதுபோக்கு அல்லது ஒரு செல்லப்பிராணியுடன் கொண்டு வருவதற்காக தேர்ந்தெடுத்த விமர்சகர்களின் எண்ணிக்கை இருக்கும், மேலும் இது ஹோட்டலை பரிந்துரைக்கும் போது பயனுள்ள தகவலாக இருக்கும்.

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

### உங்கள் கோப்பை சேமிக்கவும்

இறுதியாக, தரவுத்தொகுப்பை தற்போது புதிய பெயருடன் சேமிக்கவும்.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## உணர்வு பகுப்பாய்வு செயல்பாடுகள்

இந்த இறுதி பகுதியில், நீங்கள் விமர்சன களங்களில் உணர்வு பகுப்பாய்வைச் செயல்படுத்தி, முடிவுகளை தரவுத்தொகுப்பில் சேமிக்க வேண்டும்.

## பயிற்சி: வடிகட்டப்பட்ட தரவை ஏற்றவும் மற்றும் சேமிக்கவும்

இப்போது நீங்கள் ஏற்றும் தரவுத்தொகுப்பு, முந்தைய பகுதியில் சேமிக்கப்பட்ட வடிகட்டப்பட்ட தரவுத்தொகுப்பு, **அசல் தரவுத்தொகுப்பு அல்ல** என்பதை கவனிக்கவும்.

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

### Stop words ஐ நீக்குதல்

நீங்கள் Negative மற்றும் Positive விமர்சன களங்களில் உணர்வு பகுப்பாய்வை இயக்கினால், இது நீண்ட நேரம் எடுத்துக்கொள்ளக்கூடும். வேகமான CPU கொண்ட சக்திவாய்ந்த டெஸ்ட் லேப்டாபில் சோதிக்கப்பட்டது, இது பயன்படுத்தப்பட்ட உணர்வு நூலகத்தைப் பொறுத்து 12 - 14 நிமிடங்கள் எடுத்தது. இது (ஒப்பீட்டளவில்) நீண்ட நேரம் ஆகும், எனவே அதை வேகமாக்க முடியுமா என்பதை ஆராய்வது மதிப்புமிக்கது.

Stop words, அல்லது ஒரு வாக்கியத்தின் உணர்வை மாற்றாத பொதுவான ஆங்கில சொற்களை நீக்குவது முதல் படி. அவற்றை நீக்குவதன் மூலம், உணர்வு பகுப்பாய்வு வேகமாக இயங்க வேண்டும், ஆனால் குறைவான துல்லியமாக இருக்காது (Stop words உணர்வை பாதிக்காது, ஆனால் அவை பகுப்பாய்வை மெதுவாக்குகின்றன). 

நீண்ட Negative விமர்சனம் 395 சொற்கள் கொண்டது, ஆனால் Stop words ஐ நீக்கிய பிறகு, இது 195 சொற்கள்.

Stop words ஐ நீக்குவது ஒரு விரைவான செயல்பாடாகும், 2 விமர்சன களங்களில் இருந்து Stop words ஐ 515,000 வரிசைகளில் நீக்குவது டெஸ்ட் சாதனத்தில் 3.3 விநாடிகள் எடுத்தது. உங்கள் சாதனத்தின் CPU வேகம், RAM, SSD உள்ளதா இல்லையா, மற்றும் சில பிற காரணிகளைப் பொறுத்து இது சற்று அதிகமாக அல்லது குறைவாக நேரம் எடுத்துக்கொள்ளலாம். செயல்பாட்டின் ஒப்பீட்டளவில் குறுகிய காலம் உணர்வு பகுப்பாய்வு நேரத்தை மேம்படுத்தினால், அதைச் செய்ய மதிப்புமிக்கது.

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

### உணர்வு பகுப்பாய்வைச் செயல்படுத்துதல்

இப்போது நீங்கள் Negative மற்றும் Positive விமர்சன களங்களுக்கு உணர்வு பகுப்பாய்வை கணக்கிட வேண்டும், மேலும் முடிவுகளை 2 புதிய களங்களில் சேமிக்க வேண்டும். உணர்வின் சோதனை, அதே விமர்சனத்திற்கான விமர்சகரின் மதிப்புடன் அதை ஒப்பிடுவது. உதாரணமாக, Negative விமர்சனம் 1 (மிகவும் நேர்மறையான உணர்வு) என்ற உணர்வைக் கொண்டது மற்றும் Positive விமர்சன உணர்வு 1, ஆனால் விமர்சகர் ஹோட்டலுக்கு மிகக் குறைந்த மதிப்பை அளித்தார் என்றால், விமர்சன உரை மதிப்புடன் பொருந்தவில்லை அல்லது உணர்வு பகுப்பாய்வாளர் உணர்வை சரியாக அடையாளம் காண முடியவில்லை. சில உணர்வு மதிப்புகள் முற்றிலும் தவறாக இருக்கும் என்று நீங்கள் எதிர்பார்க்க வேண்டும், மேலும் அது அடிக்கடி விளக்கத்தக்கதாக இருக்கும், உதாரணமாக, விமர்சனம் மிகவும் கிண்டலாக இருக்கலாம் "நிச்சயமாக நான் வெப்பமில்ல
NLTK பல்வேறு உணர்வு பகுப்பாய்வு கருவிகளை வழங்குகிறது, அவற்றைப் பயன்படுத்தி கற்றுக்கொள்ளலாம், மேலும் அவற்றை மாற்றி உணர்வு எவ்வளவு துல்லியமாக இருக்கிறது என்பதைப் பார்க்கலாம். இங்கு VADER உணர்வு பகுப்பாய்வு பயன்படுத்தப்படுகிறது.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: சமூக ஊடக உரையின் உணர்வு பகுப்பாய்வுக்கான ஒரு எளிய விதி அடிப்படையிலான முறை. வலைப்பதிவுகள் மற்றும் சமூக ஊடகங்கள் பற்றிய எட்டாவது சர்வதேச மாநாடு (ICWSM-14). அன் ஆர்பர், MI, ஜூன் 2014.

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

உங்கள் நிரலின் பின்னர், உணர்வுகளை கணக்கிட தயாராக இருக்கும் போது, ஒவ்வொரு விமர்சனத்திற்கும் அதை இவ்வாறு பயன்படுத்தலாம்:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

இது என் கணினியில் சுமார் 120 விநாடிகள் ஆகும், ஆனால் ஒவ்வொரு கணினியிலும் மாறுபடும். முடிவுகளை அச்சிட்டு, உணர்வு விமர்சனத்துடன் பொருந்துகிறதா என்பதைப் பார்க்க விரும்பினால்:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

சவாலில் பயன்படுத்துவதற்கு முன் கோப்புடன் செய்ய வேண்டிய கடைசி விஷயம், அதை சேமிக்க வேண்டும்! உங்கள் புதிய நெடுவரிசைகளை மீண்டும் வரிசைப்படுத்துவது பற்றி நீங்கள் கவனம் செலுத்த வேண்டும், அவை மனிதர்களுக்கு எளிதாக வேலை செய்யும் வகையில் இருக்க வேண்டும் (இது ஒரு அழகியல் மாற்றம்).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

[அனாலிசிஸ் நோட்புக்](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) முழு குறியீட்டையும் இயக்க வேண்டும் (முன்னதாக [உங்கள் வடிகட்டல் நோட்புக்](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) இயக்கி Hotel_Reviews_Filtered.csv கோப்பை உருவாக்கிய பிறகு).

மீளாய்வு செய்ய, படிகள்:

1. முதன்மை தரவுத்தொகுப்பு கோப்பு **Hotel_Reviews.csv** [ஆராய்ச்சியாளர் நோட்புக்](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) மூலம் முந்தைய பாடத்தில் ஆராயப்பட்டது.
2. Hotel_Reviews.csv [வடிகட்டல் நோட்புக்](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) மூலம் வடிகட்டப்பட்டது, இதன் முடிவாக **Hotel_Reviews_Filtered.csv** கிடைத்தது.
3. Hotel_Reviews_Filtered.csv [உணர்வு பகுப்பாய்வு நோட்புக்](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) மூலம் செயலாக்கப்பட்டது, இதன் முடிவாக **Hotel_Reviews_NLP.csv** கிடைத்தது.
4. கீழே உள்ள NLP சவாலில் Hotel_Reviews_NLP.csv பயன்படுத்தவும்.

### முடிவு

நீங்கள் தொடங்கிய போது, உங்கள் தரவுத்தொகுப்பில் நெடுவரிசைகள் மற்றும் தரவுகள் இருந்தன, ஆனால் அவற்றில் அனைத்தையும் சரிபார்க்க முடியாது அல்லது பயன்படுத்த முடியாது. நீங்கள் தரவுகளை ஆராய்ந்துள்ளீர்கள், தேவையற்றவற்றை வடிகட்டியுள்ளீர்கள், குறிச்சொற்களை பயனுள்ளதாக்க மாற்றியுள்ளீர்கள், உங்கள் சொந்த சராசரிகளை கணக்கிட்டுள்ளீர்கள், சில உணர்வு நெடுவரிசைகளைச் சேர்த்துள்ளீர்கள், மற்றும் இயற்கை உரையை செயலாக்குவது பற்றி சில சுவாரஸ்யமான விஷயங்களை கற்றுக்கொண்டிருப்பீர்கள் என்று நம்புகிறோம்.

## [பாடத்திற்குப் பிந்தைய வினாடி வினா](https://ff-quizzes.netlify.app/en/ml/)

## சவால்

இப்போது உங்கள் தரவுத்தொகுப்பை உணர்வுக்கான பகுப்பாய்வுடன் ஆய்வு செய்துள்ளீர்கள், இந்த பாடத்திட்டத்தில் நீங்கள் கற்றுக்கொண்ட உத்திகளை (குழுவாக்கம், உதாரணமாக?) பயன்படுத்தி உணர்வைச் சுற்றியுள்ள முறைமைகளை கண்டறிய முயற்சிக்கவும்.

## மீளாய்வு & சுயபடிப்பு

உரையில் உணர்வுகளை ஆராய்வதற்கான பல்வேறு கருவிகளைப் பற்றி மேலும் அறிய [இந்த Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) எடுத்துக்கொள்ளவும்.

## பணிக்கட்டளை

[வேறு ஒரு தரவுத்தொகுப்பை முயற்சிக்கவும்](assignment.md)

---

**குறிப்பு**:  
இந்த ஆவணம் [Co-op Translator](https://github.com/Azure/co-op-translator) என்ற AI மொழிபெயர்ப்பு சேவையைப் பயன்படுத்தி மொழிபெயர்க்கப்பட்டுள்ளது. நாங்கள் துல்லியத்திற்காக முயற்சிக்கின்றோம், ஆனால் தானியங்கி மொழிபெயர்ப்புகளில் பிழைகள் அல்லது தவறான தகவல்கள் இருக்கக்கூடும் என்பதை கவனத்தில் கொள்ளவும். அதன் தாய்மொழியில் உள்ள மூல ஆவணம் அதிகாரப்பூர்வ ஆதாரமாக கருதப்பட வேண்டும். முக்கியமான தகவல்களுக்கு, தொழில்முறை மனித மொழிபெயர்ப்பு பரிந்துரைக்கப்படுகிறது. இந்த மொழிபெயர்ப்பைப் பயன்படுத்துவதால் ஏற்படும் எந்த தவறான புரிதல்கள் அல்லது தவறான விளக்கங்களுக்கு நாங்கள் பொறுப்பல்ல.