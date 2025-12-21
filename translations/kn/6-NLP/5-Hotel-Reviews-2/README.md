<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-12-19T14:43:57+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "kn"
}
-->
# ಹೋಟೆಲ್ ವಿಮರ್ಶೆಗಳೊಂದಿಗೆ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ

ನೀವು ಈಗಾಗಲೇ ಡೇಟಾಸೆಟ್ ಅನ್ನು ವಿವರವಾಗಿ ಅನ್ವೇಷಿಸಿದ್ದೀರಿ, ಈಗ ಕಾಲಮ್‌ಗಳನ್ನು ಫಿಲ್ಟರ್ ಮಾಡಿ ನಂತರ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ NLP ತಂತ್ರಗಳನ್ನು ಬಳಸಿಕೊಂಡು ಹೋಟೆಲ್‌ಗಳ ಬಗ್ಗೆ ಹೊಸ洞察ಗಳನ್ನು ಪಡೆಯುವ ಸಮಯವಾಗಿದೆ.

## [ಪೂರ್ವ-ಲೇಕ್ಚರ್ ಕ್ವಿಜ್](https://ff-quizzes.netlify.app/en/ml/)

### ಫಿಲ್ಟರಿಂಗ್ ಮತ್ತು ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ಕಾರ್ಯಾಚರಣೆಗಳು

ನೀವು ಗಮನಿಸಿದ್ದಂತೆ, ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಕೆಲವು ಸಮಸ್ಯೆಗಳಿವೆ. ಕೆಲವು ಕಾಲಮ್‌ಗಳು ಅರ್ಥವಿಲ್ಲದ ಮಾಹಿತಿಯಿಂದ ತುಂಬಿವೆ, ಇತರವು ತಪ್ಪಾಗಿವೆ ಎಂದು ತೋರುತ್ತದೆ. ಅವು ಸರಿಯಾಗಿದ್ದರೆ, ಅವು ಹೇಗೆ ಲೆಕ್ಕಿಸಲ್ಪಟ್ಟಿವೆ ಎಂಬುದು ಸ್ಪಷ್ಟವಿಲ್ಲ, ಮತ್ತು ಉತ್ತರಗಳನ್ನು ನಿಮ್ಮ ಸ್ವಂತ ಲೆಕ್ಕಾಚಾರಗಳಿಂದ ಸ್ವತಂತ್ರವಾಗಿ ಪರಿಶೀಲಿಸಲಾಗುವುದಿಲ್ಲ.

## ವ್ಯಾಯಾಮ: ಸ್ವಲ್ಪ ಹೆಚ್ಚು ಡೇಟಾ ಪ್ರಕ್ರಿಯೆ

ಡೇಟಾವನ್ನು ಸ್ವಲ್ಪ ಹೆಚ್ಚು ಸ್ವಚ್ಛಗೊಳಿಸಿ. ನಂತರ ಉಪಯುಕ್ತವಾಗುವ ಕಾಲಮ್‌ಗಳನ್ನು ಸೇರಿಸಿ, ಇತರ ಕಾಲಮ್‌ಗಳ ಮೌಲ್ಯಗಳನ್ನು ಬದಲಿಸಿ, ಮತ್ತು ಕೆಲವು ಕಾಲಮ್‌ಗಳನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ತೆಗೆದುಹಾಕಿ.

1. ಪ್ರಾಥಮಿಕ ಕಾಲಮ್ ಪ್ರಕ್ರಿಯೆ

   1. `lat` ಮತ್ತು `lng` ಅನ್ನು ತೆಗೆದುಹಾಕಿ

   2. `Hotel_Address` ಮೌಲ್ಯಗಳನ್ನು ಕೆಳಗಿನ ಮೌಲ್ಯಗಳಿಂದ ಬದಲಿಸಿ (ವಿಳಾಸದಲ್ಲಿ ನಗರ ಮತ್ತು ದೇಶದ ಹೆಸರು ಇದ್ದರೆ, ಅದನ್ನು ಕೇವಲ ನಗರ ಮತ್ತು ದೇಶಕ್ಕೆ ಬದಲಿಸಿ).

      ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಇವು ಮಾತ್ರ ನಗರಗಳು ಮತ್ತು ದೇಶಗಳು:

      ಆಂಸ್ಟರ್ಡ್ಯಾಮ್, ನೆದರ್ಲ್ಯಾಂಡ್ಸ್

      ಬಾರ್ಸಿಲೋನಾ, ಸ್ಪೇನ್

      ಲಂಡನ್, ಯುನೈಟೆಡ್ ಕಿಂಗ್‌ಡಮ್

      ಮಿಲಾನ್, ಇಟಲಿ

      ಪ್ಯಾರಿಸ್, ಫ್ರಾನ್ಸ್

      ವಿಯೆನ್ನಾ, ಆಸ್ಟ್ರಿಯಾ 

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
      
      # ಎಲ್ಲಾ ವಿಳಾಸಗಳನ್ನು ಸಂಕ್ಷಿಪ್ತ, ಹೆಚ್ಚು ಉಪಯುಕ್ತ ರೂಪದಲ್ಲಿ ಬದಲಾಯಿಸಿ
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # value_counts() ನ ಮೊತ್ತವು ವಿಮರ್ಶೆಗಳ ಒಟ್ಟು ಸಂಖ್ಯೆಗೆ ಸೇರಬೇಕು
      print(df["Hotel_Address"].value_counts())
      ```

      ಈಗ ನೀವು ದೇಶ ಮಟ್ಟದ ಡೇಟಾವನ್ನು ಪ್ರಶ್ನಿಸಬಹುದು:

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

2. ಹೋಟೆಲ್ ಮೆಟಾ-ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ

  1. `Additional_Number_of_Scoring` ಅನ್ನು ತೆಗೆದುಹಾಕಿ

  2. `Total_Number_of_Reviews` ಅನ್ನು ಆ ಹೋಟೆಲ್‌ಗೆ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ನಿಜವಾಗಿಯೂ ಇರುವ ವಿಮರ್ಶೆಗಳ ಒಟ್ಟು ಸಂಖ್ಯೆಯಿಂದ ಬದಲಿಸಿ

  3. `Average_Score` ಅನ್ನು ನಮ್ಮ ಸ್ವಂತ ಲೆಕ್ಕಿಸಿದ ಅಂಕದಿಂದ ಬದಲಿಸಿ

  ```python
  # `Additional_Number_of_Scoring` ಅನ್ನು ತೆಗೆದುಹಾಕಿ
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # `Total_Number_of_Reviews` ಮತ್ತು `Average_Score` ಅನ್ನು ನಮ್ಮ ಸ್ವಂತ ಲೆಕ್ಕಹಾಕಿದ ಮೌಲ್ಯಗಳಿಂದ ಬದಲಾಯಿಸಿ
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ಮತ್ತು `days_since_review` ಅನ್ನು ತೆಗೆದುಹಾಕಿ

   2. `Reviewer_Score`, `Negative_Review`, ಮತ್ತು `Positive_Review` ಅನ್ನು ಹಾಗೆಯೇ ಇಡಿರಿ,
     
   3. ಈಗಿಗೆ `Tags` ಅನ್ನು ಇಡಿರಿ

     - ಮುಂದಿನ ವಿಭಾಗದಲ್ಲಿ ಟ್ಯಾಗ್‌ಗಳ ಮೇಲೆ ಕೆಲವು ಹೆಚ್ಚುವರಿ ಫಿಲ್ಟರಿಂಗ್ ಕಾರ್ಯಾಚರಣೆಗಳನ್ನು ಮಾಡಲಾಗುವುದು ಮತ್ತು ನಂತರ ಟ್ಯಾಗ್‌ಗಳನ್ನು ತೆಗೆದುಹಾಕಲಾಗುವುದು

4. ವಿಮರ್ಶಕರ ಕಾಲಮ್‌ಗಳನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` ಅನ್ನು ತೆಗೆದುಹಾಕಿ
  
  2. `Reviewer_Nationality` ಅನ್ನು ಇಡಿರಿ

### ಟ್ಯಾಗ್ ಕಾಲಮ್‌ಗಳು

`Tag` ಕಾಲಮ್ ಸಮಸ್ಯೆಯಾಗಿದೆ ಏಕೆಂದರೆ ಅದು ಕಾಲಮ್‌ನಲ್ಲಿ ಪಠ್ಯ ರೂಪದಲ್ಲಿ ಸಂಗ್ರಹಿಸಲಾದ ಪಟ್ಟಿ. ದುರದೃಷ್ಟವಶಾತ್, ಈ ಕಾಲಮ್‌ನ ಉಪ ವಿಭಾಗಗಳ ಕ್ರಮ ಮತ್ತು ಸಂಖ್ಯೆ ಯಾವಾಗಲೂ ಒಂದೇ ರೀತಿಯಲ್ಲ. ಮಾನವನಿಗೆ ಸರಿಯಾದ ವಾಕ್ಯಗಳನ್ನು ಗುರುತಿಸುವುದು ಕಷ್ಟ, ಏಕೆಂದರೆ 515,000 ಸಾಲುಗಳು, 1427 ಹೋಟೆಲ್‌ಗಳು ಇವೆ, ಮತ್ತು ಪ್ರತಿಯೊಂದು ವಿಮರ್ಶಕನು ಆಯ್ಕೆಮಾಡಬಹುದಾದ ಸ್ವಲ್ಪ ವಿಭಿನ್ನ ಆಯ್ಕೆಗಳು ಇವೆ. ಇಲ್ಲಿ NLP ಪ್ರಭಾವಶಾಲಿ. ನೀವು ಪಠ್ಯವನ್ನು ಸ್ಕ್ಯಾನ್ ಮಾಡಿ ಸಾಮಾನ್ಯ ವಾಕ್ಯಗಳನ್ನು ಕಂಡುಹಿಡಿದು ಅವುಗಳನ್ನು ಎಣಿಸಬಹುದು.

ದುರದೃಷ್ಟವಶಾತ್, ನಾವು ಏಕಪದಗಳಲ್ಲಿ ಆಸಕ್ತಿ ಹೊಂದಿಲ್ಲ, ಆದರೆ ಬಹುಪದ ವಾಕ್ಯಗಳಲ್ಲಿ (ಉದಾ: *Business trip*). ಇಷ್ಟು ದೊಡ್ಡ ಡೇಟಾದ ಮೇಲೆ ಬಹುಪದ ಫ್ರೀಕ್ವೆನ್ಸಿ ವಿತರಣಾ ಆಲ್ಗೋರಿದಮ್ ಅನ್ನು ನಡೆಸುವುದು (6762646 ಪದಗಳು) ಬಹಳ ಸಮಯ ತೆಗೆದುಕೊಳ್ಳಬಹುದು, ಆದರೆ ಡೇಟಾವನ್ನು ನೋಡದೆ ಇದನ್ನು ಅಗತ್ಯ ವೆಚ್ಚವೆಂದು ಭಾಸವಾಗುತ್ತದೆ. ಇಲ್ಲಿ ಅನ್ವೇಷಣಾತ್ಮಕ ಡೇಟಾ ವಿಶ್ಲೇಷಣೆ ಉಪಯುಕ್ತವಾಗುತ್ತದೆ, ಏಕೆಂದರೆ ನೀವು ಟ್ಯಾಗ್‌ಗಳ ಮಾದರಿಯನ್ನು ನೋಡಿದ್ದೀರಿ ಉದಾ: `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` , ನೀವು ಪ್ರಕ್ರಿಯೆಯನ್ನು ಬಹಳ ಕಡಿಮೆ ಮಾಡಲು ಸಾಧ್ಯವಿದೆಯೇ ಎಂದು ಕೇಳಬಹುದು. ಅದೃಷ್ಟವಶಾತ್, ಸಾಧ್ಯ - ಆದರೆ ಮೊದಲು ನೀವು ಆಸಕ್ತಿಯ ಟ್ಯಾಗ್‌ಗಳನ್ನು ಖಚಿತಪಡಿಸಲು ಕೆಲವು ಹಂತಗಳನ್ನು ಅನುಸರಿಸಬೇಕು.

### ಟ್ಯಾಗ್‌ಗಳನ್ನು ಫಿಲ್ಟರ್ ಮಾಡುವುದು

ಡೇಟಾಸೆಟ್‌ನ ಗುರಿ ಭಾವನೆ ಮತ್ತು ಕಾಲಮ್‌ಗಳನ್ನು ಸೇರಿಸುವುದು, ಇದು ನಿಮಗೆ ಅತ್ಯುತ್ತಮ ಹೋಟೆಲ್ ಆಯ್ಕೆ ಮಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ (ನಿಮಗಾಗಿ ಅಥವಾ ಗ್ರಾಹಕನಿಗೆ ಹೋಟೆಲ್ ಶಿಫಾರಸು ಬಾಟ್ ಮಾಡಲು). ನೀವು ಟ್ಯಾಗ್‌ಗಳು ಅಂತಿಮ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಉಪಯುಕ್ತವೋ ಇಲ್ಲವೋ ಎಂದು ಕೇಳಿಕೊಳ್ಳಬೇಕು. ಇಲ್ಲಿದೆ ಒಂದು ವ್ಯಾಖ್ಯಾನ (ನೀವು ಬೇರೆ ಕಾರಣಗಳಿಗಾಗಿ ಡೇಟಾಸೆಟ್ ಬೇಕಾದರೆ ವಿಭಿನ್ನ ಟ್ಯಾಗ್‌ಗಳು ಒಳಗಾಗಬಹುದು/ಬಾಹ್ಯವಾಗಬಹುದು):

1. ಪ್ರಯಾಣದ ಪ್ರಕಾರ ಸಂಬಂಧಿಸಿದೆ, ಮತ್ತು ಅದು ಉಳಿಯಬೇಕು
2. ಅತಿಥಿ ಗುಂಪಿನ ಪ್ರಕಾರ ಮುಖ್ಯವಾಗಿದೆ, ಮತ್ತು ಅದು ಉಳಿಯಬೇಕು
3. ಅತಿಥಿ ಉಳಿದ ಕೊಠಡಿ, ಸೂಟ್ ಅಥವಾ ಸ್ಟುಡಿಯೋ ಪ್ರಕಾರ ಅಸಂಬಂಧಿತ (ಎಲ್ಲಾ ಹೋಟೆಲ್‌ಗಳಲ್ಲಿಯೂ ಮೂಲತಃ ಒಂದೇ ರೀತಿಯ ಕೊಠಡಿಗಳು ಇವೆ)
4. ವಿಮರ್ಶೆ ಸಲ್ಲಿಸಿದ ಸಾಧನ ಅಸಂಬಂಧಿತ
5. ವಿಮರ್ಶಕನು ಉಳಿದ ರಾತ್ರಿ ಸಂಖ್ಯೆ *ಸಂಬಂಧಿಸಬಹುದು* ಆದರೆ ಅದು ದೂರದೃಷ್ಟಿ ಮತ್ತು ಬಹುಶಃ ಅಸಂಬಂಧಿತ

ಸಾರಾಂಶವಾಗಿ, **2 ವಿಧದ ಟ್ಯಾಗ್‌ಗಳನ್ನು ಉಳಿಸಿ ಮತ್ತು ಇತರವನ್ನು ತೆಗೆದುಹಾಕಿ**.

ಮೊದಲು, ನೀವು ಟ್ಯಾಗ್‌ಗಳನ್ನು ಉತ್ತಮ ಸ್ವರೂಪದಲ್ಲಿ ಇರಿಸುವವರೆಗೆ ಎಣಿಸಲು ಬಯಸುವುದಿಲ್ಲ, ಅಂದರೆ ಚದರ ಕೊಠಡಿಗಳು ಮತ್ತು ಉಲ್ಲೇಖಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು. ನೀವು ಇದನ್ನು ಹಲವಾರು ರೀತಿಯಲ್ಲಿ ಮಾಡಬಹುದು, ಆದರೆ ವೇಗವಾಗಿ ಮಾಡಬೇಕಾಗುತ್ತದೆ ಏಕೆಂದರೆ ಬಹಳ ಡೇಟಾ ಪ್ರಕ್ರಿಯೆಗೆ ಸಮಯ ಬೇಕಾಗಬಹುದು. ಅದೃಷ್ಟವಶಾತ್, pandas ಇದನ್ನು ಸುಲಭವಾಗಿ ಮಾಡುತ್ತದೆ.

```Python
# ತೆರೆಯುವ ಮತ್ತು ಮುಚ್ಚುವ ಕೋಷ್ಠಕಗಳನ್ನು ತೆಗೆದುಹಾಕಿ
df.Tags = df.Tags.str.strip("[']")
# ಎಲ್ಲಾ ಉಲ್ಲೇಖಗಳನ್ನು ಕೂಡ ತೆಗೆದುಹಾಕಿ
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ಪ್ರತಿ ಟ್ಯಾಗ್ ಹೀಗೆ ಆಗುತ್ತದೆ: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

ಮುಂದೆ ಸಮಸ್ಯೆ ಕಂಡುಬರುತ್ತದೆ. ಕೆಲವು ವಿಮರ್ಶೆಗಳು ಅಥವಾ ಸಾಲುಗಳು 5 ಕಾಲಮ್‌ಗಳಿವೆ, ಕೆಲವು 3, ಕೆಲವು 6. ಇದು ಡೇಟಾಸೆಟ್ ರಚನೆಯ ಪರಿಣಾಮ, ಮತ್ತು ಸರಿಪಡಿಸಲು ಕಷ್ಟ. ನೀವು ಪ್ರತಿಯೊಂದು ವಾಕ್ಯದ ಫ್ರೀಕ್ವೆನ್ಸಿ ಎಣಿಕೆ ಪಡೆಯಲು ಬಯಸುತ್ತೀರಿ, ಆದರೆ ಅವು ಪ್ರತಿ ವಿಮರ್ಶೆಯಲ್ಲಿ ವಿಭಿನ್ನ ಕ್ರಮದಲ್ಲಿವೆ, ಆದ್ದರಿಂದ ಎಣಿಕೆ ತಪ್ಪಾಗಬಹುದು, ಮತ್ತು ಹೋಟೆಲ್‌ಗೆ ಅದು ಅರ್ಹವಾದ ಟ್ಯಾಗ್ ನೀಡಲಾಗದಿರಬಹುದು.

ಬದಲಿಗೆ ನೀವು ವಿಭಿನ್ನ ಕ್ರಮವನ್ನು ನಮ್ಮ ಲಾಭಕ್ಕೆ ಬಳಸುತ್ತೀರಿ, ಏಕೆಂದರೆ ಪ್ರತಿಯೊಂದು ಟ್ಯಾಗ್ ಬಹುಪದವಾಗಿದೆ ಆದರೆ ಕಾಮಾ ಮೂಲಕ ವಿಭಜಿಸಲಾಗಿದೆ! ಸರಳ ವಿಧಾನವೆಂದರೆ 6 ತಾತ್ಕಾಲಿಕ ಕಾಲಮ್‌ಗಳನ್ನು ರಚಿಸಿ, ಪ್ರತಿಯೊಂದು ಟ್ಯಾಗ್ ಅನ್ನು ಅದರ ಕ್ರಮಕ್ಕೆ ಹೊಂದಿಕೊಂಡು ಕಾಲಮ್‌ಗೆ ಸೇರಿಸಿ. ನಂತರ ಆ 6 ಕಾಲಮ್‌ಗಳನ್ನು ಒಂದು ದೊಡ್ಡ ಕಾಲಮ್‌ಗೆ ಮರ್ಜ್ ಮಾಡಿ `value_counts()` ವಿಧಾನವನ್ನು ಆ ಕಾಲಮ್ ಮೇಲೆ ಚಾಲನೆ ಮಾಡಬಹುದು. ಅದನ್ನು ಮುದ್ರಿಸಿದಾಗ, 2428 ವಿಭಿನ್ನ ಟ್ಯಾಗ್‌ಗಳಿವೆ. ಇಲ್ಲಿದೆ ಸಣ್ಣ ಮಾದರಿ:

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

ಕೆಲವು ಸಾಮಾನ್ಯ ಟ್ಯಾಗ್‌ಗಳು `Submitted from a mobile device` ನಮಗೆ ಉಪಯೋಗವಿಲ್ಲ, ಆದ್ದರಿಂದ ಅವುಗಳನ್ನು ಎಣಿಸುವ ಮೊದಲು ತೆಗೆದುಹಾಕುವುದು ಬುದ್ಧಿವಂತಿಕೆ, ಆದರೆ ಇದು ವೇಗವಾದ ಕಾರ್ಯವಾಗಿರುವುದರಿಂದ ಅವುಗಳನ್ನು ಉಳಿಸಿ ನಿರ್ಲಕ್ಷಿಸಬಹುದು.

### ಉಳಿದಿರುವ ಟ್ಯಾಗ್‌ಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು

ಈ ಟ್ಯಾಗ್‌ಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು ಹಂತ 1, ಇದು ಪರಿಗಣಿಸಬೇಕಾದ ಟ್ಯಾಗ್‌ಗಳ ಒಟ್ಟು ಸಂಖ್ಯೆಯನ್ನು ಸ್ವಲ್ಪ ಕಡಿಮೆ ಮಾಡುತ್ತದೆ. ಗಮನಿಸಿ ನೀವು ಅವುಗಳನ್ನು ಡೇಟಾಸೆಟ್‌ನಿಂದ ತೆಗೆದುಹಾಕುವುದಿಲ್ಲ, ಕೇವಲ ವಿಮರ್ಶೆಗಳ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಎಣಿಕೆ/ಉಳಿಸುವ ಮೌಲ್ಯಗಳ ಪರಿಗಣನೆಗೆ ತೆಗೆದುಹಾಕುತ್ತೀರಿ.

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

ಕೊಠಡಿಗಳ, ಸೂಟ್‌ಗಳ, ಸ್ಟುಡಿಯೋಗಳ, ಅಪಾರ್ಟ್‌ಮೆಂಟ್‌ಗಳ ಬಹುಮತ variety ಇದೆ. ಅವು ಎಲ್ಲವೂ ಅಂದಾಜು ಮಾಡಬಹುದಾದ ಅರ್ಥ ಹೊಂದಿವೆ ಮತ್ತು ನಿಮಗೆ ಸಂಬಂಧವಿಲ್ಲ, ಆದ್ದರಿಂದ ಅವುಗಳನ್ನು ಪರಿಗಣನೆಗೆ ತೆಗೆದುಹಾಕಿ.

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

ಕೊನೆಗೆ, ಮತ್ತು ಇದು ಸಂತೋಷದಾಯಕ (ಏಕೆಂದರೆ ಬಹಳ ಪ್ರಕ್ರಿಯೆ ಬೇಕಾಗಲಿಲ್ಲ), ನೀವು ಕೆಳಗಿನ *ಉಪಯುಕ್ತ* ಟ್ಯಾಗ್‌ಗಳನ್ನು ಹೊಂದಿರುತ್ತೀರಿ:

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

ನೀವು ವಾದಿಸಬಹುದು `Travellers with friends` ಮತ್ತು `Group` ಬಹುಶಃ ಒಂದೇ, ಮತ್ತು ಮೇಲಿನಂತೆ ಅವುಗಳನ್ನು ಸಂಯೋಜಿಸುವುದು ನ್ಯಾಯಸಮ್ಮತ. ಸರಿಯಾದ ಟ್ಯಾಗ್‌ಗಳನ್ನು ಗುರುತಿಸುವ ಕೋಡ್ [Tags ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ನಲ್ಲಿ ಇದೆ.

ಕೊನೆಯ ಹಂತವು ಈ ಟ್ಯಾಗ್‌ಗಳಿಗಾಗಿ ಹೊಸ ಕಾಲಮ್‌ಗಳನ್ನು ರಚಿಸುವುದು. ನಂತರ, ಪ್ರತಿಯೊಂದು ವಿಮರ್ಶೆ ಸಾಲಿಗೆ, `Tag` ಕಾಲಮ್ ಹೊಸ ಕಾಲಮ್‌ಗಳಲ್ಲಿ ಒಂದೊಂದರೊಂದಿಗೆ ಹೊಂದಿದರೆ, 1 ಸೇರಿಸಿ, ಇಲ್ಲದಿದ್ದರೆ 0 ಸೇರಿಸಿ. ಅಂತಿಮ ಫಲಿತಾಂಶವು ಎಷ್ಟು ವಿಮರ್ಶಕರು ಈ ಹೋಟೆಲ್ ಆಯ್ಕೆಮಾಡಿದ್ದಾರೆ ಎಂಬ ಎಣಿಕೆ ಆಗಿರುತ್ತದೆ, ಉದಾ: ವ್ಯವಹಾರ ಅಥವಾ ವಿಶ್ರಾಂತಿ, ಅಥವಾ ಪಶುಪಾಲನೆಗಾಗಿ, ಮತ್ತು ಇದು ಹೋಟೆಲ್ ಶಿಫಾರಸು ಮಾಡುವಾಗ ಉಪಯುಕ್ತ ಮಾಹಿತಿ.

```python
# ಟ್ಯಾಗ್‌ಗಳನ್ನು ಹೊಸ ಕಾಲಮ್ಗಳಾಗಿ ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಿ
# Hotel_Reviews_Tags.py ಫೈಲ್, ಅತ್ಯಂತ ಪ್ರಮುಖ ಟ್ಯಾಗ್‌ಗಳನ್ನು ಗುರುತಿಸುತ್ತದೆ
# ವಿಶ್ರಾಂತಿ ಪ್ರವಾಸ, ಜೋಡಿ, ಏಕಾಂಗ ಪ್ರವಾಸಿ, ವ್ಯವಹಾರಿಕ ಪ್ರವಾಸ, ಸ್ನೇಹಿತರೊಂದಿಗೆ ಪ್ರಯಾಣಿಕರೊಂದಿಗೆ ಗುಂಪು
# ಯುವ ಮಕ್ಕಳೊಂದಿಗೆ ಕುಟುಂಬ, ಹಿರಿಯ ಮಕ್ಕಳೊಂದಿಗೆ ಕುಟುಂಬ, ಪಶುವೊಂದರೊಂದಿಗೆ
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### ನಿಮ್ಮ ಫೈಲ್ ಉಳಿಸಿ

ಕೊನೆಗೆ, ಡೇಟಾಸೆಟ್ ಅನ್ನು ಈಗಿನ ಸ್ಥಿತಿಯಲ್ಲಿ ಹೊಸ ಹೆಸರಿನಿಂದ ಉಳಿಸಿ.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# ಲೆಕ್ಕ ಹಾಕಲಾದ ಕಾಲಮ್‌ಗಳೊಂದಿಗೆ ಹೊಸ ಡೇಟಾ ಫೈಲ್ ಉಳಿಸಲಾಗುತ್ತಿದೆ
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ಕಾರ್ಯಾಚರಣೆಗಳು

ಈ ಕೊನೆಯ ವಿಭಾಗದಲ್ಲಿ, ನೀವು ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳಿಗೆ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಅನ್ವಯಿಸಿ ಫಲಿತಾಂಶಗಳನ್ನು ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಉಳಿಸುವಿರಿ.

## ವ್ಯಾಯಾಮ: ಫಿಲ್ಟರ್ ಮಾಡಿದ ಡೇಟಾವನ್ನು ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಉಳಿಸಿ

ಈಗ ನೀವು ಹಿಂದಿನ ವಿಭಾಗದಲ್ಲಿ ಉಳಿಸಿದ ಫಿಲ್ಟರ್ ಮಾಡಿದ ಡೇಟಾಸೆಟ್ ಅನ್ನು ಲೋಡ್ ಮಾಡುತ್ತಿದ್ದೀರಿ, **ಮೂಲ ಡೇಟಾಸೆಟ್ ಅಲ್ಲ**.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# CSV ನಿಂದ ಫಿಲ್ಟರ್ ಮಾಡಲಾದ ಹೋಟೆಲ್ ವಿಮರ್ಶೆಗಳನ್ನು ಲೋಡ್ ಮಾಡಿ
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# ನಿಮ್ಮ ಕೋಡ್ ಅನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಲಾಗುವುದು


# ಕೊನೆಗೆ ಹೊಸ NLP ಡೇಟಾ ಸೇರಿಸಿದ ಹೋಟೆಲ್ ವಿಮರ್ಶೆಗಳನ್ನು ಉಳಿಸಲು ಮರೆಯಬೇಡಿ
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### ನಿಲ್ಲಿಸುವ ಪದಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು

ನೀವು ನೆಗೆಟಿವ್ ಮತ್ತು ಪಾಸಿಟಿವ್ ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳಲ್ಲಿ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ನಡೆಸಿದರೆ, ಅದು ಬಹಳ ಸಮಯ ತೆಗೆದುಕೊಳ್ಳಬಹುದು. ಶಕ್ತಿಶಾಲಿ ಟೆಸ್ಟ್ ಲ್ಯಾಪ್‌ಟಾಪ್‌ನಲ್ಲಿ ವೇಗವಾದ CPU ಇದ್ದು, 12 - 14 ನಿಮಿಷಗಳವರೆಗೆ ತೆಗೆದುಕೊಂಡಿತು, ಯಾವ ಭಾವನೆ ಗ್ರಂಥಾಲಯ ಬಳಸಿದೆಯೋ ಅವನ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ. ಇದು (ಸಾಪೇಕ್ಷವಾಗಿ) ದೀರ್ಘ ಸಮಯ, ಆದ್ದರಿಂದ ಅದನ್ನು ವೇಗಗೊಳಿಸಲು ಸಾಧ್ಯವಿದೆಯೇ ಎಂದು ಪರಿಶೀಲಿಸುವುದು ಸೂಕ್ತ.

ನಿಲ್ಲಿಸುವ ಪದಗಳು ಅಥವಾ ಸಾಮಾನ್ಯ ಇಂಗ್ಲಿಷ್ ಪದಗಳು, ಅವು ವಾಕ್ಯದ ಭಾವನೆಯನ್ನು ಬದಲಾಯಿಸುವುದಿಲ್ಲ, ಅವುಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು ಮೊದಲ ಹಂತ. ಅವುಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದರಿಂದ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ವೇಗವಾಗಿ ನಡೆಯುತ್ತದೆ, ಆದರೆ ಕಡಿಮೆ ನಿಖರವಾಗುವುದಿಲ್ಲ (ನಿಲ್ಲಿಸುವ ಪದಗಳು ಭಾವನೆಗೆ ಪ್ರಭಾವ ಬೀರುವುದಿಲ್ಲ, ಆದರೆ ವಿಶ್ಲೇಷಣೆಯನ್ನು ನಿಧಾನಗೊಳಿಸುತ್ತವೆ).

ಅತ್ಯಂತ ದೀರ್ಘ ನೆಗೆಟಿವ್ ವಿಮರ್ಶೆ 395 ಪದಗಳಿತ್ತು, ಆದರೆ ನಿಲ್ಲಿಸುವ ಪದಗಳನ್ನು ತೆಗೆದುಹಾಕಿದ ನಂತರ ಅದು 195 ಪದಗಳಾಗಿದೆ.

ನಿಲ್ಲಿಸುವ ಪದಗಳನ್ನು ತೆಗೆದುಹಾಕುವುದು ಕೂಡ ವೇಗವಾದ ಕಾರ್ಯ, 2 ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳಿಂದ 515,000 ಸಾಲುಗಳಲ್ಲಿ 3.3 ಸೆಕೆಂಡುಗಳಲ್ಲಿ ತೆಗೆದುಕೊಂಡಿತು. ನಿಮ್ಮ ಸಾಧನದ CPU ವೇಗ, RAM, SSD ಇದ್ದೇ ಇಲ್ಲ, ಮತ್ತು ಇತರ ಕೆಲವು ಅಂಶಗಳ ಮೇಲೆ ಸ್ವಲ್ಪ ಹೆಚ್ಚು ಅಥವಾ ಕಡಿಮೆ ಸಮಯ ತೆಗೆದುಕೊಳ್ಳಬಹುದು. ಕಾರ್ಯದ ಸಾಪೇಕ್ಷ ಸಣ್ಣತೆ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆಯ ಸಮಯವನ್ನು ಸುಧಾರಿಸಿದರೆ, ಅದನ್ನು ಮಾಡುವುದು ಲಾಭದಾಯಕ.

```python
from nltk.corpus import stopwords

# CSV ನಿಂದ ಹೋಟೆಲ್ ವಿಮರ್ಶೆಗಳನ್ನು ಲೋಡ್ ಮಾಡಿ
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# ಸ್ಟಾಪ್ ಪದಗಳನ್ನು ತೆಗೆದುಹಾಕಿ - ಹೆಚ್ಚಿನ ಪಠ್ಯಕ್ಕೆ ಇದು ನಿಧಾನವಾಗಬಹುದು!
# ರಯಾನ್ ಹ್ಯಾನ್ (Kaggle ನಲ್ಲಿ ryanxjhan) ವಿವಿಧ ಸ್ಟಾಪ್ ಪದಗಳ ತೆಗೆದುಹಾಕುವ ವಿಧಾನಗಳ ಕಾರ್ಯಕ್ಷಮತೆಯನ್ನು ಅಳೆಯುವ ಉತ್ತಮ ಪೋಸ್ಟ್ ಹೊಂದಿದ್ದಾರೆ
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # ರಯಾನ್ ಶಿಫಾರಸು ಮಾಡುವ ವಿಧಾನವನ್ನು ಬಳಸುವುದು
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# ಎರಡೂ ಕಾಲಮ್‌ಗಳಿಂದ ಸ್ಟಾಪ್ ಪದಗಳನ್ನು ತೆಗೆದುಹಾಕಿ
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### ಭಾವನೆ ವಿಶ್ಲೇಷಣೆ ನಡೆಸುವುದು

ಈಗ ನೀವು ನೆಗೆಟಿವ್ ಮತ್ತು ಪಾಸಿಟಿವ್ ವಿಮರ್ಶೆ ಕಾಲಮ್‌ಗಳಿಗೆ ಭಾವನೆ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಲೆಕ್ಕಿಸಿ, 2 ಹೊಸ ಕಾಲಮ್‌ಗಳಲ್ಲಿ ಫಲಿತಾಂಶವನ್ನು ಸಂಗ್ರಹಿಸಬೇಕು. ಭಾವನೆ ಪರೀಕ್ಷೆ ವಿಮರ್ಶಕರ ಅಂಕದೊಂದಿಗೆ ಹೋಲಿಕೆ ಮಾಡುವುದು. ಉದಾಹರಣೆಗೆ, ಭಾವನೆ ವಿಶ್ಲೇಷಕನು ನೆಗೆಟಿವ್ ವಿಮರ್ಶೆಗೆ 1 (ಅತ್ಯಂತ ಧನಾತ್ಮಕ ಭಾವನೆ) ಮತ್ತು ಪಾಸಿಟಿವ್ ವಿಮರ್ಶೆಗೆ 1 ಎಂದು ಅಂದಾಜಿಸಿದರೆ, ಆದರೆ ವಿಮರ್ಶಕನು ಹೋಟೆಲ್‌ಗೆ ಕನಿಷ್ಠ ಅಂಕ ನೀಡಿದ್ದರೆ, ವಿಮರ್ಶೆ ಪಠ್ಯ ಮತ್ತು ಅಂಕ ಹೊಂದಿಕೆಯಾಗುತ್ತಿಲ್ಲ ಅಥವಾ ಭಾವನೆ ವಿಶ್ಲೇಷಕನು ಭಾವನೆಯನ್ನು ಸರಿಯಾಗಿ ಗುರುತಿಸಲಿಲ್ಲ. ಕೆಲವು ಭಾವನೆ ಅಂಕಗಳು ಸಂಪೂರ್ಣ ತಪ್ಪಾಗಿರಬಹುದು, ಮತ್ತು ಬಹುಶಃ ಅದು ವಿವರಿಸಬಹುದಾಗಿದೆ, ಉದಾ: ವಿಮರ್ಶೆ ಅತ್ಯಂತ ವ್ಯಂಗ್ಯಾತ್ಮಕವಾಗಿರಬಹುದು "ನಾನು ಬಿಸಿಲಿಲ್ಲದ ಕೊಠಡಿಯಲ್ಲಿ ನಿದ್ರೆ ಮಾಡಿದ್ದೇನೆ ಎಂದು ಖಂಡಿತವಾಗಿ ಪ್ರೀತಿಸಿದೆ" ಮತ್ತು ಭಾವನೆ ವಿಶ್ಲೇಷಕನು ಅದನ್ನು ಧನಾತ್ಮಕ ಭಾವನೆ ಎಂದು ಭಾವಿಸುತ್ತದೆ, ಆದರೆ ಮಾನವ ಓದಿದರೆ ಅದು ವ್ಯಂಗ್ಯ ಎಂದು ತಿಳಿಯುತ್ತದೆ.
NLTK ವಿಭಿನ್ನ ಭಾವನಾತ್ಮಕ ವಿಶ್ಲೇಷಕರನ್ನು ಕಲಿಯಲು ಒದಗಿಸುತ್ತದೆ, ಮತ್ತು ನೀವು ಅವುಗಳನ್ನು ಬದಲಾಯಿಸಿ ಭಾವನೆ ಹೆಚ್ಚು ಅಥವಾ ಕಡಿಮೆ ನಿಖರವಾಗಿದೆಯೇ ಎಂದು ನೋಡಬಹುದು. ಇಲ್ಲಿ VADER ಭಾವನಾತ್ಮಕ ವಿಶ್ಲೇಷಣೆ ಬಳಸಲಾಗಿದೆ.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ವಾಡರ್ ಭಾವನಾತ್ಮಕ ವಿಶ್ಲೇಷಕವನ್ನು ರಚಿಸಿ (ನೀವು ಪ್ರಯತ್ನಿಸಬಹುದಾದ NLTK ನಲ್ಲಿ ಇತರರೂ ಇದ್ದಾರೆ)
vader_sentiment = SentimentIntensityAnalyzer()
# ಹುಟ್ಟೋ, ಸಿ.ಜೆ. & ಗಿಲ್ಬರ್ಟ್, ಇ.ಇ. (2014). ವಾಡರ್: ಸಾಮಾಜಿಕ ಮಾಧ್ಯಮ ಪಠ್ಯದ ಭಾವನಾತ್ಮಕ ವಿಶ್ಲೇಷಣೆಗೆ ನಿಯಮಾಧಾರಿತ ಸರಳ ಮಾದರಿ. ಎಂಟನೇ ಅಂತಾರಾಷ್ಟ್ರೀಯ ವೆಬ್ಲಾಗ್ ಮತ್ತು ಸಾಮಾಜಿಕ ಮಾಧ್ಯಮ ಸಮ್ಮೇಳನ (ICWSM-14). ಆನ್ ಅರ್ಬರ್, MI, ಜೂನ್ 2014.

# ವಿಮರ್ಶೆಗೆ 3 ಇನ್ಪುಟ್ ಸಾಧ್ಯತೆಗಳಿವೆ:
# ಅದು "ನಕಾರಾತ್ಮಕವಿಲ್ಲ" ಆಗಿರಬಹುದು, ಆ ಸಂದರ್ಭದಲ್ಲಿ 0 ಅನ್ನು ಹಿಂತಿರುಗಿಸಿ
# ಅದು "ಧನಾತ್ಮಕವಿಲ್ಲ" ಆಗಿರಬಹುದು, ಆ ಸಂದರ್ಭದಲ್ಲಿ 0 ಅನ್ನು ಹಿಂತಿರುಗಿಸಿ
# ಅದು ವಿಮರ್ಶೆಯಾಗಿರಬಹುದು, ಆ ಸಂದರ್ಭದಲ್ಲಿ ಭಾವನೆಯನ್ನು ಲೆಕ್ಕಿಸಿ
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

ನಿಮ್ಮ ಪ್ರೋಗ್ರಾಂನಲ್ಲಿ ನಂತರ ನೀವು ಭಾವನೆಯನ್ನು ಲೆಕ್ಕಹಾಕಲು ಸಿದ್ಧರಾಗಿದ್ದಾಗ, ನೀವು ಪ್ರತಿಯೊಂದು ವಿಮರ್ಶೆಗೆ ಇದನ್ನು ಹೀಗೆ ಅನ್ವಯಿಸಬಹುದು:

```python
# ನಕಾರಾತ್ಮಕ ಭಾವನೆ ಮತ್ತು ಧನಾತ್ಮಕ ಭಾವನೆ ಕಾಲಮ್ ಸೇರಿಸಿ
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

ಇದು ನನ್ನ ಕಂಪ್ಯೂಟರ್‌ನಲ್ಲಿ ಸುಮಾರು 120 ಸೆಕೆಂಡುಗಳು ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ, ಆದರೆ ಪ್ರತಿ ಕಂಪ್ಯೂಟರ್‌ನಲ್ಲಿ ಇದು ಬದಲಾಗಬಹುದು. ನೀವು ಫಲಿತಾಂಶಗಳನ್ನು ಮುದ್ರಿಸಿ ಭಾವನೆ ವಿಮರ್ಶೆಗೆ ಹೊಂದಿಕೆಯಾಗಿದೆಯೇ ಎಂದು ನೋಡಲು ಬಯಸಿದರೆ:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

ಚಾಲೆಂಜ್‌ನಲ್ಲಿ ಬಳಸುವ ಮೊದಲು ಫೈಲ್‌ನೊಂದಿಗೆ ಮಾಡಬೇಕಾದ ಕೊನೆಯ ಕೆಲಸ, ಅದನ್ನು ಉಳಿಸುವುದು! ನೀವು ನಿಮ್ಮ ಎಲ್ಲಾ ಹೊಸ ಕಾಲಮ್‌ಗಳನ್ನು ಮರುಕ್ರಮಗೊಳಿಸುವುದನ್ನು ಪರಿಗಣಿಸಬೇಕು, ಇದರಿಂದ ಅವುಗಳನ್ನು ಕೆಲಸ ಮಾಡಲು ಸುಲಭವಾಗುತ್ತದೆ (ಮಾನವನಿಗೆ ಇದು ಒಂದು ಸೌಂದರ್ಯ ಬದಲಾವಣೆ).

```python
# ಕಾಲಮ್ಗಳನ್ನು ಮರುಕ್ರಮಗೊಳಿಸಿ (ಇದು ಸೌಂದರ್ಯಕ್ಕಾಗಿ, ಆದರೆ ನಂತರ ಡೇಟಾವನ್ನು ಅನ್ವೇಷಿಸಲು ಸುಲಭವಾಗಿಸಲು)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

ನೀವು ಸಂಪೂರ್ಣ ಕೋಡ್ ಅನ್ನು [ವಿಶ್ಲೇಷಣಾ ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb)ಗಾಗಿ ಚಲಾಯಿಸಬೇಕು (ನೀವು [ನಿಮ್ಮ ಫಿಲ್ಟರಿಂಗ್ ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ಅನ್ನು ಚಲಾಯಿಸಿ Hotel_Reviews_Filtered.csv ಫೈಲ್ ಅನ್ನು ರಚಿಸಿದ ನಂತರ).

ಪುನಃ ಪರಿಶೀಲಿಸಲು, ಹಂತಗಳು ಇವು:

1. ಮೂಲ ಡೇಟಾಸೆಟ್ ಫೈಲ್ **Hotel_Reviews.csv** ಅನ್ನು ಹಿಂದಿನ ಪಾಠದಲ್ಲಿ [ಎಕ್ಸ್‌ಪ್ಲೋರರ್ ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ಮೂಲಕ ಅನ್ವೇಷಿಸಲಾಗಿದೆ
2. Hotel_Reviews.csv ಅನ್ನು [ಫಿಲ್ಟರಿಂಗ್ ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ಮೂಲಕ ಫಿಲ್ಟರ್ ಮಾಡಿ **Hotel_Reviews_Filtered.csv** ಅನ್ನು ಪಡೆದಿದ್ದಾರೆ
3. Hotel_Reviews_Filtered.csv ಅನ್ನು [ಭಾವನಾತ್ಮಕ ವಿಶ್ಲೇಷಣಾ ನೋಟ್ಬುಕ್](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ಮೂಲಕ ಪ್ರಕ್ರಿಯೆ ಮಾಡಿ **Hotel_Reviews_NLP.csv** ಅನ್ನು ಪಡೆದಿದ್ದಾರೆ
4. ಕೆಳಗಿನ NLP ಚಾಲೆಂಜ್‌ನಲ್ಲಿ Hotel_Reviews_NLP.csv ಅನ್ನು ಬಳಸಿ

### ಸಮಾರೋಪ

ನೀವು ಪ್ರಾರಂಭಿಸಿದಾಗ, ನಿಮಗೆ ಕಾಲಮ್‌ಗಳು ಮತ್ತು ಡೇಟಾ ಇರುವ ಡೇಟಾಸೆಟ್ ಇತ್ತು ಆದರೆ ಅದರಲ್ಲಿ ಎಲ್ಲವೂ ಪರಿಶೀಲಿಸಲಾಗಲಿಲ್ಲ ಅಥವಾ ಬಳಸಲಾಗಲಿಲ್ಲ. ನೀವು ಡೇಟಾವನ್ನು ಅನ್ವೇಷಿಸಿ, ಬೇಕಾಗದವನ್ನೆಲ್ಲಾ ಫಿಲ್ಟರ್ ಮಾಡಿ, ಟ್ಯಾಗ್‌ಗಳನ್ನು ಉಪಯುಕ್ತವಾದುದಾಗಿ ಪರಿವರ್ತಿಸಿ, ನಿಮ್ಮ ಸ್ವಂತ ಸರಾಸರಿಗಳನ್ನು ಲೆಕ್ಕಹಾಕಿ, ಕೆಲವು ಭಾವನಾತ್ಮಕ ಕಾಲಮ್‌ಗಳನ್ನು ಸೇರಿಸಿ ಮತ್ತು ಸಹಜ ಪಠ್ಯವನ್ನು ಪ್ರಕ್ರಿಯೆ ಮಾಡುವ ಬಗ್ಗೆ ಕೆಲವು ಆಸಕ್ತಿದಾಯಕ ವಿಷಯಗಳನ್ನು ಕಲಿತಿದ್ದೀರಿ.

## [ಪಾಠೋತ್ತರ ಕ್ವಿಜ್](https://ff-quizzes.netlify.app/en/ml/)

## ಚಾಲೆಂಜ್

ನೀವು ಈಗ ನಿಮ್ಮ ಡೇಟಾಸೆಟ್ ಅನ್ನು ಭಾವನೆಗಾಗಿ ವಿಶ್ಲೇಷಿಸಿದ್ದೀರಿ, ಈ ಪಠ್ಯಕ್ರಮದಲ್ಲಿ ನೀವು ಕಲಿತ ತಂತ್ರಗಳನ್ನು (ಶ್ರೇಣೀಕರಣ, ಬಹುಶಃ?) ಬಳಸಿಕೊಂಡು ಭಾವನೆಯ ಸುತ್ತಲೂ ಮಾದರಿಗಳನ್ನು ನಿರ್ಧರಿಸಲು ಪ್ರಯತ್ನಿಸಿ.

## ವಿಮರ್ಶೆ ಮತ್ತು ಸ್ವಯಂ ಅಧ್ಯಯನ

[ಈ ಲರ್ನ್ ಮೋಡ್ಯೂಲ್](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) ಅನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ ಮತ್ತು ಪಠ್ಯದಲ್ಲಿ ಭಾವನೆಯನ್ನು ಅನ್ವೇಷಿಸಲು ವಿಭಿನ್ನ ಸಾಧನಗಳನ್ನು ಬಳಸಿ.

## ನಿಯೋಜನೆ

[ಬೇರೊಂದು ಡೇಟಾಸೆಟ್ ಪ್ರಯತ್ನಿಸಿ](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ಅಸ್ವೀಕಾರ**:  
ಈ ದಸ್ತಾವೇಜು AI ಅನುವಾದ ಸೇವೆ [Co-op Translator](https://github.com/Azure/co-op-translator) ಬಳಸಿ ಅನುವಾದಿಸಲಾಗಿದೆ. ನಾವು ನಿಖರತೆಯಿಗಾಗಿ ಪ್ರಯತ್ನಿಸುತ್ತಿದ್ದರೂ, ಸ್ವಯಂಚಾಲಿತ ಅನುವಾದಗಳಲ್ಲಿ ದೋಷಗಳು ಅಥವಾ ಅಸತ್ಯತೆಗಳು ಇರಬಹುದು ಎಂದು ದಯವಿಟ್ಟು ಗಮನಿಸಿ. ಮೂಲ ಭಾಷೆಯಲ್ಲಿರುವ ಮೂಲ ದಸ್ತಾವೇಜನ್ನು ಅಧಿಕೃತ ಮೂಲವಾಗಿ ಪರಿಗಣಿಸಬೇಕು. ಮಹತ್ವದ ಮಾಹಿತಿಗಾಗಿ, ವೃತ್ತಿಪರ ಮಾನವ ಅನುವಾದವನ್ನು ಶಿಫಾರಸು ಮಾಡಲಾಗುತ್ತದೆ. ಈ ಅನುವಾದ ಬಳಕೆಯಿಂದ ಉಂಟಾಗುವ ಯಾವುದೇ ತಪ್ಪು ಅರ್ಥಮಾಡಿಕೊಳ್ಳುವಿಕೆ ಅಥವಾ ತಪ್ಪು ವಿವರಣೆಗಳಿಗೆ ನಾವು ಹೊಣೆಗಾರರಾಗುವುದಿಲ್ಲ.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->