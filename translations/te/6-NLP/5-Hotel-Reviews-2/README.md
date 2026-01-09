<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-12-19T14:38:20+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "te"
}
-->
# హోటల్ సమీక్షలతో భావ విశ్లేషణ

ఇప్పుడు మీరు డేటాసెట్‌ను వివరంగా పరిశీలించినందున, కాలమ్స్‌ను ఫిల్టర్ చేసి, ఆపై డేటాసెట్‌పై NLP సాంకేతికతలను ఉపయోగించి హోటల్స్ గురించి కొత్త అవగాహనలను పొందే సమయం వచ్చింది.

## [పూర్వ-లెక్చర్ క్విజ్](https://ff-quizzes.netlify.app/en/ml/)

### ఫిల్టరింగ్ & భావ విశ్లేషణ ఆపరేషన్లు

మీరు గమనించినట్లయితే, డేటాసెట్‌లో కొన్ని సమస్యలు ఉన్నాయి. కొన్ని కాలమ్స్ అనవసరమైన సమాచారంతో నిండిపోయాయి, మరికొన్ని తప్పుగా కనిపిస్తున్నాయి. అవి సరైనవైతే, అవి ఎలా లెక్కించబడ్డాయో స్పష్టంగా లేదు, మరియు మీ స్వంత లెక్కింపులతో సమాధానాలను స్వతంత్రంగా ధృవీకరించలేరు.

## వ్యాయామం: కొంతమంది డేటా ప్రాసెసింగ్

డేటాను కొంచెం మరింత శుభ్రం చేయండి. తర్వాత ఉపయోగకరమైన కాలమ్స్‌ను జోడించండి, ఇతర కాలమ్స్‌లో విలువలను మార్చండి, మరియు కొన్ని కాలమ్స్‌ను పూర్తిగా తొలగించండి.

1. ప్రారంభ కాలమ్ ప్రాసెసింగ్

   1. `lat` మరియు `lng` తొలగించండి

   2. `Hotel_Address` విలువలను క్రింది విలువలతో మార్చండి (ఒక చిరునామాలో నగరం మరియు దేశం రెండూ ఉంటే, దాన్ని కేవలం నగరం మరియు దేశంగా మార్చండి).

      డేటాసెట్‌లో ఉన్న నగరాలు మరియు దేశాలు ఇవే:

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
      
      # అన్ని చిరునామాలను సంక్షిప్తమైన, మరింత ఉపయోగకరమైన రూపంతో మార్చండి
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # value_counts() యొక్క మొత్తం సమీక్షల మొత్తం సంఖ్యకు చేరాలి
      print(df["Hotel_Address"].value_counts())
      ```

      ఇప్పుడు మీరు దేశ స్థాయి డేటాను ప్రశ్నించవచ్చు:

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

2. హోటల్ మెటా-రివ్యూ కాలమ్స్ ప్రాసెస్ చేయండి

  1. `Additional_Number_of_Scoring` తొలగించండి

  2. `Total_Number_of_Reviews` విలువను ఆ హోటల్‌కు డేటాసెట్‌లో వాస్తవంగా ఉన్న సమీక్షల మొత్తం సంఖ్యతో మార్చండి

  3. `Average_Score` ను మన స్వంత లెక్కించిన స్కోర్‌తో మార్చండి

  ```python
  # `Additional_Number_of_Scoring` ను తొలగించండి
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # `Total_Number_of_Reviews` మరియు `Average_Score` ను మన స్వంత గణన విలువలతో మార్చండి
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. సమీక్ష కాలమ్స్ ప్రాసెస్ చేయండి

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` మరియు `days_since_review` తొలగించండి

   2. `Reviewer_Score`, `Negative_Review`, మరియు `Positive_Review` ను అలాగే ఉంచండి,
     
   3. ప్రస్తుతానికి `Tags` ను ఉంచండి

     - తదుపరి విభాగంలో ట్యాగ్స్‌పై మరింత ఫిల్టరింగ్ ఆపరేషన్లు చేస్తాము, ఆ తర్వాత ట్యాగ్స్ తొలగించబడతాయి

4. సమీక్షకుల కాలమ్స్ ప్రాసెస్ చేయండి

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` తొలగించండి
  
  2. `Reviewer_Nationality` ను ఉంచండి

### ట్యాగ్ కాలమ్స్

`Tag` కాలమ్ సమస్యాత్మకం ఎందుకంటే అది కాలమ్‌లో నిల్వ ఉన్న ఒక జాబితా (పాఠ్య రూపంలో). దురదృష్టవశాత్తు, ఈ కాలమ్‌లో ఉప విభాగాల క్రమం మరియు సంఖ్య ఎప్పుడూ ఒకేలా ఉండదు. మానవుడు సరైన పదబంధాలను గుర్తించడం కష్టం, ఎందుకంటే 515,000 వరుసలు, 1427 హోటల్స్ ఉన్నాయి, మరియు ప్రతి ఒక్కరిలో సమీక్షకుడు ఎంచుకునే ఎంపికలు కొంచెం భిన్నంగా ఉంటాయి. ఇక్కడ NLP ప్రకాశిస్తుంది. మీరు పాఠ్యాన్ని స్కాన్ చేసి అత్యంత సాధారణ పదబంధాలను కనుగొని, వాటిని లెక్కించవచ్చు.

దురదృష్టవశాత్తు, మేము ఒక్కో పదాలను కాకుండా, బహుళ పదబంధాలను (ఉదా: *Business trip*) ఆసక్తి కలిగి ఉన్నాము. ఆంతరంగిక పదబంధ ఫ్రీక్వెన్సీ పంపిణీ అల్గోరిథం అమలు చేయడం (6762646 పదాలు) చాలా సమయం తీసుకోవచ్చు, కానీ డేటాను చూడకుండానే అది అవసరమైన ఖర్చు అనిపిస్తుంది. ఇక్కడ అన్వేషణాత్మక డేటా విశ్లేషణ ఉపయోగకరం, ఎందుకంటే మీరు ట్యాగ్స్ యొక్క నమూనాను చూసారు, ఉదా: `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, మీరు ప్రాసెసింగ్‌ను గణనీయంగా తగ్గించగలరా అని అడగవచ్చు. అదృష్టవశాత్తు, అవును - కానీ ముందుగా మీరు ఆసక్తి కలిగిన ట్యాగ్స్‌ను నిర్ధారించడానికి కొన్ని దశలను అనుసరించాలి.

### ట్యాగ్స్ ఫిల్టరింగ్

డేటాసెట్ యొక్క లక్ష్యం భావాన్ని మరియు కాలమ్స్‌ను జోడించడం, ఇవి ఉత్తమ హోటల్‌ను ఎంచుకోవడంలో సహాయపడతాయి (మీ కోసం లేదా క్లయింట్ కోసం హోటల్ సిఫార్సు బాట్ తయారుచేయడానికి). మీరు ట్యాగ్స్ ఉపయోగకరమా లేదా కాదా అని అడగాలి. ఇక్కడ ఒక వ్యాఖ్యానం ఉంది (మీరు డేటాసెట్‌ను ఇతర కారణాల కోసం అవసరం అయితే, వేరే ట్యాగ్స్ ఎంపికలో ఉండవచ్చు లేదా ఉండకపోవచ్చు):

1. ప్రయాణ రకం సంబంధితది, అది ఉండాలి
2. అతిథి గుంపు రకం ముఖ్యమైనది, అది ఉండాలి
3. అతిథి ఉన్న గది, సూట్ లేదా స్టూడియో రకం సంబంధం లేదు (అన్ని హోటల్స్‌లో ప్రాథమికంగా అదే గదులు ఉంటాయి)
4. సమీక్ష సమర్పించిన పరికరం సంబంధం లేదు
5. సమీక్షకుడు ఎంత రాత్రులు ఉన్నాడో *సంబంధం ఉండవచ్చు* (వారు ఎక్కువ కాలం ఉంటే హోటల్ ఇష్టపడతారని భావిస్తే), కానీ అది కొంతవరకు మాత్రమే, సాధారణంగా సంబంధం లేదు

సారాంశంగా, **2 రకాల ట్యాగ్స్‌ను ఉంచి మిగతా వాటిని తొలగించండి**.

మొదట, మీరు ట్యాగ్స్‌ను లెక్కించాలనుకుంటే, అవి మెరుగైన ఫార్మాట్‌లో ఉండాలి, అంటే చతురస్ర కోట్స్ మరియు కోట్స్ తొలగించాలి. మీరు దీన్ని అనేక విధాల చేయవచ్చు, కానీ మీరు వేగంగా చేయగలిగే విధానాన్ని కోరుకుంటారు, ఎందుకంటే చాలా డేటాను ప్రాసెస్ చేయడానికి ఎక్కువ సమయం పడుతుంది. అదృష్టవశాత్తు, pandas ఈ దశలను సులభంగా చేయగలదు.

```Python
# ప్రారంభ మరియు ముగింపు కోట్స్ తీసివేయండి
df.Tags = df.Tags.str.strip("[']")
# అన్ని కోట్స్ కూడా తీసివేయండి
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ప్రతి ట్యాగ్ ఇలా మారుతుంది: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

తర్వాత ఒక సమస్య వస్తుంది. కొన్ని సమీక్షలు లేదా వరుసలు 5 కాలమ్స్ కలిగి ఉంటాయి, కొన్ని 3, కొన్ని 6. ఇది డేటాసెట్ సృష్టి విధానం కారణంగా, సరిచేయడం కష్టం. మీరు ప్రతి పదబంధం యొక్క ఫ్రీక్వెన్సీ లెక్కించాలనుకుంటున్నారు, కానీ అవి ప్రతి సమీక్షలో వేరే క్రమంలో ఉన్నందున, లెక్క తప్పు కావచ్చు, మరియు హోటల్‌కు అది అర్హమైన ట్యాగ్ కేటాయించబడకపోవచ్చు.

దీనికి బదులుగా, మీరు వేరే క్రమాన్ని మన లాభానికి ఉపయోగిస్తారు, ఎందుకంటే ప్రతి ట్యాగ్ బహుళ పదబంధం అయినప్పటికీ, కామాతో వేరుచేయబడింది! దీని సులభమైన మార్గం 6 తాత్కాలిక కాలమ్స్ సృష్టించడం, ప్రతి ట్యాగ్‌ను దాని క్రమంలో ఉన్న కాలమ్‌లో చేర్చడం. ఆ తర్వాత ఆ 6 కాలమ్స్‌ను ఒక పెద్ద కాలమ్‌గా విలీనం చేసి, ఆ కాలమ్‌పై `value_counts()` పద్ధతిని అమలు చేయవచ్చు. ప్రింట్ చేస్తే, 2428 ప్రత్యేక ట్యాగ్స్ ఉన్నట్లు కనిపిస్తుంది. ఇక్కడ చిన్న నమూనా:

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

కొన్ని సాధారణ ట్యాగ్స్, ఉదా: `Submitted from a mobile device` మనకు ఉపయోగం లేదు, కాబట్టి వాటిని లెక్కించే ముందు తొలగించడం మంచిది, కానీ ఇది చాలా వేగంగా జరిగే ఆపరేషన్ కాబట్టి వాటిని ఉంచి పక్కన పెట్టవచ్చు.

### ఉండే కాలం ట్యాగ్స్ తొలగించడం

ఈ ట్యాగ్స్ తొలగించడం మొదటి దశ, ఇది మొత్తం ట్యాగ్స్ సంఖ్యను కొంచెం తగ్గిస్తుంది. గమనించండి, మీరు వాటిని డేటాసెట్ నుండి తొలగించరు, కేవలం సమీక్షల డేటాసెట్‌లో లెక్కించడానికి/ఉంచడానికి పరిగణన నుండి తీసివేస్తారు.

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

గదులు, సూట్లు, స్టూడియోలు, అపార్ట్‌మెంట్లు విభిన్న రకాలు ఉన్నాయి. అవి సారాంశంగా ఒకే అర్థం కలిగి ఉంటాయి మరియు మీకు సంబంధం లేదు, కాబట్టి వాటిని పరిగణన నుండి తీసివేయండి.

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

చివరగా, ఇది ఆనందదాయకం (ఎందుకంటే చాలా ప్రాసెసింగ్ అవసరం కాలేదు), మీరు ఈ క్రింది *ఉపయోగకరమైన* ట్యాగ్స్‌తో మిగిలిపోతారు:

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

`Travellers with friends` అనేది `Group` తో సమానమని మీరు వాదించవచ్చు, మరియు పై విధంగా వాటిని కలపడం సరైనది. సరైన ట్యాగ్స్ గుర్తించడానికి కోడ్ [Tags నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) లో ఉంది.

చివరి దశ ప్రతి ట్యాగ్ కోసం కొత్త కాలమ్స్ సృష్టించడం. ఆపై, ప్రతి సమీక్ష వరుసకు, `Tag` కాలమ్ కొత్త కాలమ్‌లలో ఒకదానికి సరిపోతే 1 జోడించండి, లేకపోతే 0 జోడించండి. ఫలితం, ఉదా: వ్యాపార ప్రయాణం vs విశ్రాంతి కోసం ఈ హోటల్ ఎన్ని సమీక్షకులు ఎంచుకున్నారు అనే లెక్క, ఇది హోటల్ సిఫార్సు చేయడంలో ఉపయోగకరమైన సమాచారం.

```python
# ట్యాగ్‌లను కొత్త కాలమ్స్‌గా ప్రాసెస్ చేయండి
# Hotel_Reviews_Tags.py ఫైల్, అత్యంత ముఖ్యమైన ట్యాగ్‌లను గుర్తిస్తుంది
# విశ్రాంతి ప్రయాణం, జంట, ఒంటరి ప్రయాణికుడు, వ్యాపార ప్రయాణం, మిత్రులతో ప్రయాణికులతో కలిపిన గ్రూప్,
# చిన్న పిల్లలతో కుటుంబం, పెద్ద పిల్లలతో కుటుంబం, పెంపుడు జంతువుతో
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### మీ ఫైల్‌ను సేవ్ చేయండి

చివరగా, డేటాసెట్‌ను ఇప్పుడు ఉన్నట్లుగా కొత్త పేరుతో సేవ్ చేయండి.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# లెక్కించబడిన కాలమ్స్‌తో కొత్త డేటా ఫైల్‌ను సేవ్ చేస్తోంది
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## భావ విశ్లేషణ ఆపరేషన్లు

ఈ చివరి విభాగంలో, మీరు సమీక్ష కాలమ్స్‌పై భావ విశ్లేషణను అమలు చేసి, ఫలితాలను డేటాసెట్‌లో సేవ్ చేస్తారు.

## వ్యాయామం: ఫిల్టర్ చేసిన డేటాను లోడ్ చేసి సేవ్ చేయండి

గమనించండి, ఇప్పుడు మీరు గత విభాగంలో సేవ్ చేసిన ఫిల్టర్ చేసిన డేటాసెట్‌ను లోడ్ చేస్తున్నారు, **మూల డేటాసెట్ కాదు**.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# ఫిల్టర్ చేసిన హోటల్ సమీక్షలను CSV నుండి లోడ్ చేయండి
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# మీ కోడ్ ఇక్కడ జోడించబడుతుంది


# చివరగా, కొత్త NLP డేటా జోడించిన హోటల్ సమీక్షలను సేవ్ చేయడం మర్చిపోకండి
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### స్టాప్ వర్డ్స్ తొలగించడం

మీరు నెగటివ్ మరియు పాజిటివ్ సమీక్ష కాలమ్స్‌పై భావ విశ్లేషణను అమలు చేస్తే, అది చాలా సమయం తీసుకోవచ్చు. శక్తివంతమైన టెస్ట్ ల్యాప్‌టాప్‌లో వేగవంతమైన CPUతో పరీక్షించినప్పుడు, ఇది 12 - 14 నిమిషాలు పట్టింది, ఉపయోగించిన భావ లైబ్రరీపై ఆధారపడి. ఇది (సాపేక్షంగా) ఎక్కువ సమయం, కాబట్టి వేగవంతం చేయగలమా అని పరిశీలించవలసి ఉంటుంది.

స్టాప్ వర్డ్స్, లేదా సాధారణ ఇంగ్లీష్ పదాలు, వాక్య భావాన్ని మార్చవు, తొలగించడం మొదటి దశ. వాటిని తీసివేస్తే, భావ విశ్లేషణ వేగంగా నడుస్తుంది, కానీ తక్కువ ఖచ్చితత్వం ఉండదు (స్టాప్ వర్డ్స్ భావాన్ని ప్రభావితం చేయవు, కానీ విశ్లేషణను మందగింపజేస్తాయి).

అతి పొడవైన నెగటివ్ సమీక్ష 395 పదాలు, కానీ స్టాప్ వర్డ్స్ తీసివేసిన తర్వాత 195 పదాలు మాత్రమే.

స్టాప్ వర్డ్స్ తొలగించడం కూడా వేగవంతమైన ఆపరేషన్, 2 సమీక్ష కాలమ్స్ నుండి 515,000 వరుసలపై స్టాప్ వర్డ్స్ తీసివేయడం టెస్ట్ పరికరంలో 3.3 సెకన్లు పట్టింది. మీ పరికరం CPU వేగం, RAM, SSD ఉన్నా లేకపోయినా, మరియు ఇతర కారణాలపై కొంత తేడా ఉండవచ్చు. ఆపరేషన్ తక్కువ సమయం కావడం వల్ల, భావ విశ్లేషణ సమయం మెరుగుపడితే, ఇది చేయడం విలువైనది.

```python
from nltk.corpus import stopwords

# CSV నుండి హోటల్ సమీక్షలను లోడ్ చేయండి
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# స్టాప్ వర్డ్స్ తొలగించండి - చాలా టెక్స్ట్ కోసం ఇది నెమ్మదిగా ఉండవచ్చు!
# ర్యాన్ హాన్ (ryanxjhan కాగుల్ లో) వివిధ స్టాప్ వర్డ్స్ తొలగింపు పద్ధతుల పనితీరును కొలిచే గొప్ప పోస్ట్ కలిగి ఉన్నారు
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # ర్యాన్ సూచించిన పద్ధతిని ఉపయోగించడం
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# రెండు కాలమ్స్ నుండి స్టాప్ వర్డ్స్ తొలగించండి
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### భావ విశ్లేషణ నిర్వహణ

ఇప్పుడు మీరు నెగటివ్ మరియు పాజిటివ్ సమీక్ష కాలమ్స్ కోసం భావ విశ్లేషణను లెక్కించి, ఫలితాన్ని 2 కొత్త కాలమ్స్‌లో నిల్వ చేయాలి. భావ పరీక్ష సమీక్షకుడి స్కోర్‌తో పోల్చడం ద్వారా జరుగుతుంది. ఉదాహరణకు, భావ విశ్లేషణ నెగటివ్ సమీక్షకు 1 (అత్యంత పాజిటివ్ భావం) మరియు పాజిటివ్ సమీక్షకు 1 అని భావిస్తే, కానీ సమీక్షకుడు హోటల్‌కు అత్యల్ప స్కోర్ ఇచ్చినట్లయితే, సమీక్ష పాఠ్యం స్కోర్‌కు సరిపోలకపోవచ్చు లేదా భావ విశ్లేషకుడు భావాన్ని సరిగ్గా గుర్తించలేకపోయినట్టవుతుంది. కొన్ని భావ స్కోర్లు పూర్తిగా తప్పు ఉండవచ్చు, మరియు తరచుగా అది వివరణాత్మకం, ఉదా: సమీక్ష చాలా వ్యంగ్యంగా ఉండవచ్చు "Of course I LOVED sleeping in a room with no heating" మరియు భావ విశ్లేషకుడు దాన్ని పాజిటివ్ భావంగా భావిస్తాడు, కానీ మానవుడు చదివితే అది వ్యంగ్యం అని తెలుసుకుంటాడు.
NLTK వివిధ భావోద్వేగ విశ్లేషకులను నేర్చుకోవడానికి అందిస్తుంది, మీరు వాటిని మార్చి భావోద్వేగం ఎక్కువ లేదా తక్కువ ఖచ్చితంగా ఉందో చూడవచ్చు. ఇక్కడ VADER భావోద్వేగ విశ్లేషణ ఉపయోగించబడింది.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# వాడర్ సెంటిమెంట్ విశ్లేషకాన్ని సృష్టించండి (మీరు ప్రయత్నించగల NLTKలో ఇతరులు కూడా ఉన్నారు)
vader_sentiment = SentimentIntensityAnalyzer()
# హుట్టో, సి.జె. & గిల్బర్ట్, ఈ.ఈ. (2014). VADER: సోషల్ మీడియా టెక్స్ట్ సెంటిమెంట్ విశ్లేషణ కోసం ఒక సరళమైన నియమాధారిత మోడల్. ఎనిమిదవ అంతర్జాతీయ వెబ్‌లాగ్స్ మరియు సోషల్ మీడియా కాన్ఫరెన్స్ (ICWSM-14). ఆన్ ఆర్బర్, MI, జూన్ 2014.

# సమీక్షకు 3 ఇన్‌పుట్ అవకాశాలు ఉన్నాయి:
# ఇది "నెగటివ్ లేదు" కావచ్చు, అప్పుడు 0 ను తిరిగి ఇవ్వండి
# ఇది "పాజిటివ్ లేదు" కావచ్చు, అప్పుడు 0 ను తిరిగి ఇవ్వండి
# ఇది ఒక సమీక్ష కావచ్చు, అప్పుడు సెంటిమెంట్‌ను లెక్కించండి
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

మీ ప్రోగ్రామ్‌లో మీరు భావోద్వేగాన్ని లెక్కించడానికి సిద్ధంగా ఉన్నప్పుడు, మీరు దీన్ని ప్రతి సమీక్షకు క్రింది విధంగా వర్తింపజేయవచ్చు:

```python
# ఒక నెగటివ్ భావోద్వేగం మరియు పాజిటివ్ భావోద్వేగం కాలమ్‌ను జోడించండి
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

ఇది నా కంప్యూటర్‌లో సుమారు 120 సెకన్లు పడుతుంది, కానీ ప్రతి కంప్యూటర్‌లో ఇది మారవచ్చు. మీరు ఫలితాలను ముద్రించి భావోద్వేగం సమీక్షకు సరిపోతుందో లేదో చూడాలనుకుంటే:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

సవాలు కోసం ఫైల్‌ను ఉపయోగించే ముందు చేయవలసిన చివరి విషయం, దాన్ని సేవ్ చేయడం! మీరు మీ కొత్త కాలమ్స్‌ను సులభంగా పని చేయడానికి (మానవునికి ఇది ఒక రూపకల్పన మార్పు) పునఃక్రమీకరించడాన్ని కూడా పరిగణించాలి.

```python
# కాలమ్స్‌ను పునఃక్రమించండి (ఇది రూపకల్పన సంబంధమైనది, కానీ తరువాత డేటాను సులభంగా అన్వేషించడానికి)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

మీరు మొత్తం కోడ్‌ను [విశ్లేషణ నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) కోసం నడపాలి (మీరు [ఫిల్టరింగ్ నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) నడిపి Hotel_Reviews_Filtered.csv ఫైల్‌ను సృష్టించిన తర్వాత).

సమీక్షించడానికి, దశలు:

1. అసలు డేటాసెట్ ఫైల్ **Hotel_Reviews.csv** ను గత పాఠంలో [ఎక్స్‌ప్లోరర్ నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) తో పరిశీలించారు
2. Hotel_Reviews.csv ను [ఫిల్టరింగ్ నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ద్వారా ఫిల్టర్ చేసి **Hotel_Reviews_Filtered.csv** ను పొందారు
3. Hotel_Reviews_Filtered.csv ను [భావోద్వేగ విశ్లేషణ నోట్‌బుక్](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ద్వారా ప్రాసెస్ చేసి **Hotel_Reviews_NLP.csv** ను పొందారు
4. క్రింద ఉన్న NLP సవాలులో Hotel_Reviews_NLP.csv ను ఉపయోగించండి

### ముగింపు

మీరు ప్రారంభించినప్పుడు, మీ వద్ద కాలమ్స్ మరియు డేటాతో కూడిన డేటాసెట్ ఉంది కానీ అందులోని అన్ని డేటాను ధృవీకరించలేకపోయారు లేదా ఉపయోగించలేకపోయారు. మీరు డేటాను పరిశీలించారు, అవసరం లేని వాటిని ఫిల్టర్ చేశారు, ట్యాగ్‌లను ఉపయోగకరమైన వాటిగా మార్చారు, మీ స్వంత సగటులను లెక్కించారు, కొన్ని భావోద్వేగ కాలమ్స్ జోడించారు మరియు సహజ భాషా ప్రాసెసింగ్ గురించి కొన్ని ఆసక్తికర విషయాలు నేర్చుకున్నారు.

## [పోస్ట్-లెక్చర్ క్విజ్](https://ff-quizzes.netlify.app/en/ml/)

## సవాలు

ఇప్పుడు మీరు మీ డేటాసెట్‌ను భావోద్వేగం కోసం విశ్లేషించారు, మీరు ఈ పాఠ్యాంశంలో నేర్చుకున్న వ్యూహాలను (క్లస్టరింగ్, కావచ్చు?) ఉపయోగించి భావోద్వేగం చుట్టూ నమూనాలను గుర్తించగలరా చూడండి.

## సమీక్ష & స్వీయ అధ్యయనం

భావోద్వేగాన్ని మరింత తెలుసుకోవడానికి మరియు వేర్వేరు సాధనాలను ఉపయోగించి భావోద్వేగాన్ని అన్వేషించడానికి [ఈ లెర్న్ మాడ్యూల్](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) తీసుకోండి.

## అసైన్‌మెంట్

[వేరే డేటాసెట్ ప్రయత్నించండి](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**అస్పష్టత**:  
ఈ పత్రాన్ని AI అనువాద సేవ [Co-op Translator](https://github.com/Azure/co-op-translator) ఉపయోగించి అనువదించబడింది. మేము ఖచ్చితత్వానికి ప్రయత్నించినప్పటికీ, ఆటోమేటెడ్ అనువాదాల్లో పొరపాట్లు లేదా తప్పిదాలు ఉండవచ్చు. మూల పత్రం దాని స్వదేశీ భాషలో అధికారిక మూలంగా పరిగణించాలి. ముఖ్యమైన సమాచారానికి, ప్రొఫెషనల్ మానవ అనువాదం సిఫార్సు చేయబడుతుంది. ఈ అనువాదం వాడకంలో ఏర్పడిన ఏవైనా అపార్థాలు లేదా తప్పుదారులు కోసం మేము బాధ్యత వహించము.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->