<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-06T07:22:48+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "pa"
}
-->
# ਹੋਟਲ ਰਿਵਿਊਜ਼ ਨਾਲ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ

ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ ਡਾਟਾਸੈਟ ਨੂੰ ਵਿਸਤਾਰ ਵਿੱਚ ਖੰਗਾਲ ਲਿਆ ਹੈ, ਤਾਂ ਕਾਲਮਾਂ ਨੂੰ ਫਿਲਟਰ ਕਰਨ ਅਤੇ ਹੋਟਲਾਂ ਬਾਰੇ ਨਵੇਂ ਅੰਤਰਦ੍ਰਿਸ਼ਟੀ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਡਾਟਾਸੈਟ 'ਤੇ NLP ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦਾ ਸਮਾਂ ਹੈ।

## [ਪ੍ਰੀ-ਲੈਕਚਰ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

### ਫਿਲਟਰਿੰਗ ਅਤੇ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਕਾਰਵਾਈਆਂ

ਜਿਵੇਂ ਤੁਸੀਂ ਸ਼ਾਇਦ ਧਿਆਨ ਦਿੱਤਾ ਹੋਵੇਗਾ, ਡਾਟਾਸੈਟ ਵਿੱਚ ਕੁਝ ਸਮੱਸਿਆਵਾਂ ਹਨ। ਕੁਝ ਕਾਲਮ ਬੇਕਾਰ ਜਾਣਕਾਰੀ ਨਾਲ ਭਰੇ ਹੋਏ ਹਨ, ਹੋਰ ਗਲਤ ਲੱਗਦੇ ਹਨ। ਜੇ ਉਹ ਸਹੀ ਹਨ, ਤਾਂ ਇਹ ਸਪਸ਼ਟ ਨਹੀਂ ਹੈ ਕਿ ਉਹ ਕਿਵੇਂ ਗਣਨਾ ਕੀਤੇ ਗਏ ਸਨ, ਅਤੇ ਤੁਹਾਡੇ ਆਪਣੇ ਗਣਨਾਵਾਂ ਦੁਆਰਾ ਜਵਾਬਾਂ ਨੂੰ ਸਵੈ-ਨਿਰਧਾਰਿਤ ਤੌਰ 'ਤੇ ਪ੍ਰਮਾਣਿਤ ਨਹੀਂ ਕੀਤਾ ਜਾ ਸਕਦਾ।

## ਅਭਿਆਸ: ਥੋੜਾ ਹੋਰ ਡਾਟਾ ਪ੍ਰੋਸੈਸਿੰਗ

ਡਾਟਾ ਨੂੰ ਹੋਰ ਸਾਫ਼ ਕਰੋ। ਉਹ ਕਾਲਮ ਸ਼ਾਮਲ ਕਰੋ ਜੋ ਬਾਅਦ ਵਿੱਚ ਲਾਭਦਾਇਕ ਹੋਣਗੇ, ਹੋਰ ਕਾਲਮਾਂ ਵਿੱਚ ਮੁੱਲਾਂ ਨੂੰ ਬਦਲੋ, ਅਤੇ ਕੁਝ ਕਾਲਮਾਂ ਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਹਟਾਓ।

1. ਸ਼ੁਰੂਆਤੀ ਕਾਲਮ ਪ੍ਰੋਸੈਸਿੰਗ

   1. `lat` ਅਤੇ `lng` ਨੂੰ ਹਟਾਓ

   2. `Hotel_Address` ਮੁੱਲਾਂ ਨੂੰ ਹੇਠਾਂ ਦਿੱਤੇ ਮੁੱਲਾਂ ਨਾਲ ਬਦਲੋ (ਜੇ ਪਤਾ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਦੇ ਨਾਮ ਨੂੰ ਸ਼ਾਮਲ ਕਰਦਾ ਹੈ, ਤਾਂ ਇਸਨੂੰ ਸਿਰਫ਼ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਵਿੱਚ ਬਦਲੋ):

      ਇਹ ਡਾਟਾਸੈਟ ਵਿੱਚ ਸਿਰਫ਼ ਇਹ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਹਨ:

      ਐਮਸਟਰਡਮ, ਨੀਦਰਲੈਂਡ

      ਬਾਰਸਲੋਨਾ, ਸਪੇਨ

      ਲੰਡਨ, ਯੂਨਾਈਟਡ ਕਿੰਗਡਮ

      ਮਿਲਾਨ, ਇਟਲੀ

      ਪੈਰਿਸ, ਫਰਾਂਸ

      ਵੀਅਨਾ, ਆਸਟਰੀਆ 

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

      ਹੁਣ ਤੁਸੀਂ ਦੇਸ਼ ਪੱਧਰ ਦਾ ਡਾਟਾ ਪੁੱਛ ਸਕਦੇ ਹੋ:

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

2. ਹੋਟਲ ਮੈਟਾ-ਰਿਵਿਊ ਕਾਲਮਾਂ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰੋ

  1. `Additional_Number_of_Scoring` ਨੂੰ ਹਟਾਓ

  1. `Total_Number_of_Reviews` ਨੂੰ ਉਸ ਹੋਟਲ ਲਈ ਡਾਟਾਸੈਟ ਵਿੱਚ ਅਸਲ ਵਿੱਚ ਮੌਜੂਦ ਰਿਵਿਊਜ਼ ਦੀ ਕੁੱਲ ਗਿਣਤੀ ਨਾਲ ਬਦਲੋ 

  1. `Average_Score` ਨੂੰ ਸਾਡੇ ਆਪਣੇ ਗਣਨਾ ਕੀਤੇ ਸਕੋਰ ਨਾਲ ਬਦਲੋ

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. ਰਿਵਿਊ ਕਾਲਮਾਂ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰੋ

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ਅਤੇ `days_since_review` ਨੂੰ ਹਟਾਓ

   2. `Reviewer_Score`, `Negative_Review`, ਅਤੇ `Positive_Review` ਨੂੰ ਜਿਵੇਂ ਦੇ ਤਿਵੇਂ ਰੱਖੋ,
     
   3. `Tags` ਨੂੰ ਅਜੇ ਲਈ ਰੱਖੋ

     - ਅਗਲੇ ਭਾਗ ਵਿੱਚ ਟੈਗਾਂ 'ਤੇ ਕੁਝ ਵਾਧੂ ਫਿਲਟਰਿੰਗ ਕਾਰਵਾਈਆਂ ਕੀਤੀਆਂ ਜਾਣਗੀਆਂ ਅਤੇ ਫਿਰ ਟੈਗਾਂ ਨੂੰ ਹਟਾ ਦਿੱਤਾ ਜਾਵੇਗਾ

4. ਰਿਵਿਊਅਰ ਕਾਲਮਾਂ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰੋ

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` ਨੂੰ ਹਟਾਓ
  
  2. `Reviewer_Nationality` ਨੂੰ ਰੱਖੋ

### ਟੈਗ ਕਾਲਮ

`Tag` ਕਾਲਮ ਸਮੱਸਿਆਜਨਕ ਹੈ ਕਿਉਂਕਿ ਇਹ ਇੱਕ ਸੂਚੀ (ਟੈਕਸਟ ਰੂਪ ਵਿੱਚ) ਹੈ ਜੋ ਕਾਲਮ ਵਿੱਚ ਸਟੋਰ ਕੀਤੀ ਗਈ ਹੈ। ਦੁਖਦਾਈ ਗੱਲ ਇਹ ਹੈ ਕਿ ਇਸ ਕਾਲਮ ਵਿੱਚ ਉਪ-ਵਿਭਾਗਾਂ ਦੀ ਕ੍ਰਮ ਅਤੇ ਗਿਣਤੀ ਹਮੇਸ਼ਾ ਇੱਕੋ ਜਿਹੀ ਨਹੀਂ ਹੁੰਦੀ। ਇੱਕ ਮਨੁੱਖ ਲਈ ਸਹੀ ਵਾਕਾਂਸ਼ਾਂ ਦੀ ਪਛਾਣ ਕਰਨਾ ਮੁਸ਼ਕਲ ਹੈ, ਕਿਉਂਕਿ 515,000 ਪੰਕਤਾਂ ਹਨ, ਅਤੇ 1427 ਹੋਟਲ ਹਨ, ਅਤੇ ਹਰ ਇੱਕ ਵਿੱਚ ਰਿਵਿਊਅਰ ਦੁਆਰਾ ਚੁਣੇ ਗਏ ਵਿਕਲਪਾਂ ਵਿੱਚ ਥੋੜ੍ਹਾ ਅੰਤਰ ਹੁੰਦਾ ਹੈ। ਇਹ ਜਿੱਥੇ NLP ਮਦਦਗਾਰ ਹੁੰਦਾ ਹੈ। ਤੁਸੀਂ ਟੈਕਸਟ ਨੂੰ ਸਕੈਨ ਕਰ ਸਕਦੇ ਹੋ ਅਤੇ ਸਭ ਤੋਂ ਆਮ ਵਾਕਾਂਸ਼ਾਂ ਨੂੰ ਲੱਭ ਸਕਦੇ ਹੋ, ਅਤੇ ਉਨ੍ਹਾਂ ਦੀ ਗਿਣਤੀ ਕਰ ਸਕਦੇ ਹੋ।

ਦੁਖਦਾਈ ਗੱਲ ਇਹ ਹੈ ਕਿ ਸਾਨੂੰ ਸਿਰਫ਼ ਇੱਕ-ਸ਼ਬਦ ਵਾਲੇ ਟੈਗਾਂ ਵਿੱਚ ਦਿਲਚਸਪੀ ਨਹੀਂ ਹੈ, ਸਗੋਂ ਬਹੁ-ਸ਼ਬਦ ਵਾਲੇ ਵਾਕਾਂਸ਼ਾਂ ਵਿੱਚ (ਜਿਵੇਂ *Business trip*)। ਇਸ ਕਦਰ ਡਾਟਾ (6762646 ਸ਼ਬਦ) 'ਤੇ ਬਹੁ-ਸ਼ਬਦ ਵਾਲੇ ਫ੍ਰੀਕਵੈਂਸੀ ਡਿਸਟ੍ਰੀਬਿਊਸ਼ਨ ਐਲਗੋਰਿਦਮ ਨੂੰ ਚਲਾਉਣਾ ਬਹੁਤ ਜ਼ਿਆਦਾ ਸਮਾਂ ਲੈ ਸਕਦਾ ਹੈ, ਪਰ ਡਾਟਾ ਨੂੰ ਦੇਖਣ ਤੋਂ ਬਿਨਾਂ, ਇਹ ਲਗਦਾ ਹੈ ਕਿ ਇਹ ਇੱਕ ਜ਼ਰੂਰੀ ਖਰਚਾ ਹੈ। ਇਹ ਜਿੱਥੇ ਖੋਜੀ ਡਾਟਾ ਵਿਸ਼ਲੇਸ਼ਣ ਲਾਭਦਾਇਕ ਹੁੰਦਾ ਹੈ, ਕਿਉਂਕਿ ਤੁਸੀਂ ਟੈਗਾਂ ਦਾ ਨਮੂਨਾ ਦੇਖਿਆ ਹੈ ਜਿਵੇਂ `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, ਤੁਸੀਂ ਪੁੱਛਣਾ ਸ਼ੁਰੂ ਕਰ ਸਕਦੇ ਹੋ ਕਿ ਕੀ ਇਹ ਸੰਭਵ ਹੈ ਕਿ ਤੁਸੀਂ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਪ੍ਰੋਸੈਸਿੰਗ ਨੂੰ ਬਹੁਤ ਘਟਾ ਸਕਦੇ ਹੋ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ, ਇਹ ਸੰਭਵ ਹੈ - ਪਰ ਪਹਿਲਾਂ ਤੁਹਾਨੂੰ ਦਿਲਚਸਪੀ ਵਾਲੇ ਟੈਗਾਂ ਨੂੰ ਪਛਾਣ ਕਰਨ ਲਈ ਕੁਝ ਕਦਮਾਂ ਦੀ ਪਾਲਣਾ ਕਰਨ ਦੀ ਜ਼ਰੂਰਤ ਹੈ।

### ਟੈਗਾਂ ਨੂੰ ਫਿਲਟਰ ਕਰਨਾ

ਯਾਦ ਰੱਖੋ ਕਿ ਡਾਟਾਸੈਟ ਦਾ ਉਦੇਸ਼ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਅਤੇ ਕਾਲਮਾਂ ਨੂੰ ਸ਼ਾਮਲ ਕਰਨਾ ਹੈ ਜੋ ਤੁਹਾਨੂੰ ਸਭ ਤੋਂ ਵਧੀਆ ਹੋਟਲ ਚੁਣਨ ਵਿੱਚ ਮਦਦ ਕਰੇਗਾ (ਤੁਹਾਡੇ ਲਈ ਜਾਂ ਸ਼ਾਇਦ ਇੱਕ ਕਲਾਇੰਟ ਜੋ ਤੁਹਾਨੂੰ ਹੋਟਲ ਰਿਕਮੈਂਡੇਸ਼ਨ ਬੋਟ ਬਣਾਉਣ ਲਈ ਕਹਿ ਰਿਹਾ ਹੈ)। ਤੁਹਾਨੂੰ ਆਪਣੇ ਆਪ ਨੂੰ ਪੁੱਛਣ ਦੀ ਜ਼ਰੂਰਤ ਹੈ ਕਿ ਕੀ ਟੈਗਾਂ ਅੰਤਮ ਡਾਟਾਸੈਟ ਵਿੱਚ ਲਾਭਦਾਇਕ ਹਨ ਜਾਂ ਨਹੀਂ। ਇੱਥੇ ਇੱਕ ਵਿਆਖਿਆ ਹੈ (ਜੇ ਤੁਹਾਨੂੰ ਹੋਰ ਕਾਰਨਾਂ ਲਈ ਡਾਟਾਸੈਟ ਦੀ ਜ਼ਰੂਰਤ ਹੋਵੇ ਤਾਂ ਵੱਖ-ਵੱਖ ਟੈਗਾਂ ਨੂੰ ਚੁਣਿਆ ਜਾ ਸਕਦਾ ਹੈ):

1. ਯਾਤਰਾ ਦੀ ਕਿਸਮ ਸਬੰਧਿਤ ਹੈ, ਅਤੇ ਇਹ ਰਹੇਗਾ
2. ਮਹਿਮਾਨ ਸਮੂਹ ਦੀ ਕਿਸਮ ਮਹੱਤਵਪੂਰਨ ਹੈ, ਅਤੇ ਇਹ ਰਹੇਗਾ
3. ਕਮਰੇ, ਸੂਟ, ਜਾਂ ਸਟੂਡੀਓ ਦੀ ਕਿਸਮ ਜਿਸ ਵਿੱਚ ਮਹਿਮਾਨ ਰਹੇ, ਅਸੰਬੰਧਿਤ ਹੈ (ਸਾਰੇ ਹੋਟਲਾਂ ਵਿੱਚ ਮੁਢਲੀ ਤੌਰ 'ਤੇ ਇੱਕੋ ਜਿਹੇ ਕਮਰੇ ਹੁੰਦੇ ਹਨ)
4. ਜਿਹੜੇ ਡਿਵਾਈਸ 'ਤੇ ਰਿਵਿਊ ਸਬਮਿਟ ਕੀਤਾ ਗਿਆ ਸੀ, ਅਸੰਬੰਧਿਤ ਹੈ
5. ਰਾਤਾਂ ਦੀ ਗਿਣਤੀ ਜਿਨ੍ਹਾਂ ਲਈ ਰਿਵਿਊਅਰ ਰਿਹਾ *ਸ਼ਾਇਦ* ਸਬੰਧਿਤ ਹੋ ਸਕਦਾ ਹੈ ਜੇ ਤੁਸੀਂ ਲੰਬੇ ਰਹਿਣ ਨੂੰ ਹੋਟਲ ਨੂੰ ਵਧੇਰੇ ਪਸੰਦ ਕਰਨ ਨਾਲ ਜੋੜਦੇ ਹੋ, ਪਰ ਇਹ ਇੱਕ ਖਿੱਚ ਹੈ, ਅਤੇ ਸ਼ਾਇਦ ਅਸੰਬੰਧਿਤ

ਸੰਖੇਪ ਵਿੱਚ, **2 ਕਿਸਮਾਂ ਦੇ ਟੈਗਾਂ ਨੂੰ ਰੱਖੋ ਅਤੇ ਹੋਰਾਂ ਨੂੰ ਹਟਾਓ**।

ਸਭ ਤੋਂ ਪਹਿਲਾਂ, ਤੁਸੀਂ ਟੈਗਾਂ ਦੀ ਗਿਣਤੀ ਨਹੀਂ ਕਰਨਾ ਚਾਹੁੰਦੇ ਜਦੋਂ ਤੱਕ ਉਹ ਬਿਹਤਰ ਫਾਰਮੈਟ ਵਿੱਚ ਨਾ ਹੋਣ, ਇਸ ਲਈ ਇਸਦਾ ਮਤਲਬ ਹੈ ਚੌਰਸ ਬ੍ਰੈਕਟ ਅਤੇ ਕੋਟਸ ਨੂੰ ਹਟਾਉਣਾ। ਤੁਸੀਂ ਇਹ ਕਈ ਤਰੀਕਿਆਂ ਨਾਲ ਕਰ ਸਕਦੇ ਹੋ, ਪਰ ਤੁਸੀਂ ਸਭ ਤੋਂ ਤੇਜ਼ ਤਰੀਕੇ ਦੀ ਚਾਹਤ ਰੱਖਦੇ ਹੋ ਕਿਉਂਕਿ ਇਹ ਬਹੁਤ ਸਾਰੇ ਡਾਟਾ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰਨ ਵਿੱਚ ਬਹੁਤ ਸਮਾਂ ਲੈ ਸਕਦਾ ਹੈ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ, pandas ਵਿੱਚ ਇਹਨਾਂ ਕਦਮਾਂ ਵਿੱਚੋਂ ਹਰ ਇੱਕ ਨੂੰ ਕਰਨ ਦਾ ਇੱਕ ਆਸਾਨ ਤਰੀਕਾ ਹੈ।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ਹਰ ਟੈਗ ਕੁਝ ਇਸ ਤਰ੍ਹਾਂ ਬਣ ਜਾਂਦਾ ਹੈ: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

ਅਗਲੇ ਕਦਮਾਂ ਵਿੱਚ, ਤੁਸੀਂ ਟੈਗਾਂ ਨੂੰ ਸਹੀ ਫਾਰਮੈਟ ਵਿੱਚ ਲਿਆਉਣ, ਗਿਣਤੀ ਕਰਨ ਅਤੇ ਹੋਟਲਾਂ ਲਈ ਸਿਫਾਰਸ਼ੀ ਜਾਣਕਾਰੀ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ ਕਾਰਵਾਈਆਂ ਕਰਦੇ ਹੋ।
NLTK ਵੱਖ-ਵੱਖ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਕ ਉਪਲਬਧ ਕਰਵਾਉਂਦਾ ਹੈ, ਅਤੇ ਤੁਸੀਂ ਇਹਨਾਂ ਨੂੰ ਬਦਲ ਸਕਦੇ ਹੋ ਅਤੇ ਦੇਖ ਸਕਦੇ ਹੋ ਕਿ ਭਾਵਨਾ ਜ਼ਿਆਦਾ ਜਾਂ ਘੱਟ ਸਹੀ ਹੈ। ਇੱਥੇ VADER ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਵਰਤਿਆ ਗਿਆ ਹੈ।

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

ਆਪਣੇ ਪ੍ਰੋਗਰਾਮ ਵਿੱਚ ਜਦੋਂ ਤੁਸੀਂ ਭਾਵਨਾ ਦੀ ਗਣਨਾ ਕਰਨ ਲਈ ਤਿਆਰ ਹੋ, ਤਾਂ ਤੁਸੀਂ ਇਸਨੂੰ ਹਰੇਕ ਸਮੀਖਿਆ 'ਤੇ ਇਸ ਤਰ੍ਹਾਂ ਲਾਗੂ ਕਰ ਸਕਦੇ ਹੋ:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

ਇਹ ਮੇਰੇ ਕੰਪਿਊਟਰ 'ਤੇ ਲਗਭਗ 120 ਸਕਿੰਟ ਲੈਂਦਾ ਹੈ, ਪਰ ਇਹ ਹਰ ਕੰਪਿਊਟਰ 'ਤੇ ਵੱਖ-ਵੱਖ ਹੋਵੇਗਾ। ਜੇ ਤੁਸੀਂ ਨਤੀਜੇ ਪ੍ਰਿੰਟ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਅਤੇ ਦੇਖਣਾ ਚਾਹੁੰਦੇ ਹੋ ਕਿ ਭਾਵਨਾ ਸਮੀਖਿਆ ਨਾਲ ਮਿਲਦੀ ਹੈ:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

ਫਾਈਲ ਨਾਲ ਆਖਰੀ ਕੰਮ, ਇਸਨੂੰ ਚੁਣੌਤੀ ਵਿੱਚ ਵਰਤਣ ਤੋਂ ਪਹਿਲਾਂ, ਇਸਨੂੰ ਸੇਵ ਕਰਨਾ ਹੈ! ਤੁਹਾਨੂੰ ਆਪਣੇ ਨਵੇਂ ਕਾਲਮਾਂ ਨੂੰ ਦੁਬਾਰਾ ਕ੍ਰਮਬੱਧ ਕਰਨ ਬਾਰੇ ਵੀ ਸੋਚਣਾ ਚਾਹੀਦਾ ਹੈ ਤਾਂ ਕਿ ਇਹਨਾਂ ਨਾਲ ਕੰਮ ਕਰਨਾ ਆਸਾਨ ਹੋਵੇ (ਇਹ ਸਿਰਫ਼ ਦ੍ਰਿਸ਼ਟੀਕੋਣ ਦਾ ਬਦਲਾਅ ਹੈ)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

ਤੁਹਾਨੂੰ [ਵਿਸ਼ਲੇਸ਼ਣ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ਲਈ ਪੂਰਾ ਕੋਡ ਚਲਾਉਣਾ ਚਾਹੀਦਾ ਹੈ (ਜਦੋਂ ਤੁਸੀਂ [ਫਿਲਟਰਿੰਗ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ਚਲਾਇਆ ਹੈ ਤਾਂ Hotel_Reviews_Filtered.csv ਫਾਈਲ ਬਣਾਉਣ ਲਈ)।

ਸੰਖੇਪ ਵਿੱਚ, ਕਦਮ ਇਹ ਹਨ:

1. ਮੂਲ ਡਾਟਾਸੈੱਟ ਫਾਈਲ **Hotel_Reviews.csv** ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ [ਐਕਸਪਲੋਰਰ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ਨਾਲ ਖੋਜੀ ਗਈ ਹੈ।
2. Hotel_Reviews.csv ਨੂੰ [ਫਿਲਟਰਿੰਗ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ਦੁਆਰਾ ਫਿਲਟਰ ਕੀਤਾ ਗਿਆ ਹੈ ਜਿਸਦਾ ਨਤੀਜਾ **Hotel_Reviews_Filtered.csv** ਹੈ।
3. Hotel_Reviews_Filtered.csv ਨੂੰ [ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ਦੁਆਰਾ ਪ੍ਰੋਸੈਸ ਕੀਤਾ ਗਿਆ ਹੈ ਜਿਸਦਾ ਨਤੀਜਾ **Hotel_Reviews_NLP.csv** ਹੈ।
4. Hotel_Reviews_NLP.csv ਨੂੰ ਹੇਠਾਂ ਦਿੱਤੀ NLP ਚੁਣੌਤੀ ਵਿੱਚ ਵਰਤੋ।

### ਨਿਸਕਰਸ਼

ਜਦੋਂ ਤੁਸੀਂ ਸ਼ੁਰੂ ਕੀਤਾ, ਤੁਹਾਡੇ ਕੋਲ ਕਾਲਮਾਂ ਅਤੇ ਡਾਟਾ ਵਾਲਾ ਡਾਟਾਸੈੱਟ ਸੀ ਪਰ ਇਸ ਵਿੱਚੋਂ ਸਾਰਾ ਡਾਟਾ ਪ੍ਰਮਾਣਿਤ ਜਾਂ ਵਰਤਣਯੋਗ ਨਹੀਂ ਸੀ। ਤੁਸੀਂ ਡਾਟਾ ਦੀ ਖੋਜ ਕੀਤੀ, ਜੋ ਲੋੜੀਂਦਾ ਨਹੀਂ ਸੀ ਉਸਨੂੰ ਹਟਾਇਆ, ਟੈਗਾਂ ਨੂੰ ਕੁਝ ਉਪਯੋਗ ਵਿੱਚ ਬਦਲਿਆ, ਆਪਣੇ ਆਪ ਦੇ ਔਸਤ ਗਣੇ, ਕੁਝ ਭਾਵਨਾ ਕਾਲਮ ਜੋੜੇ ਅਤੇ ਸ਼ਾਇਦ ਕੁਝ ਦਿਲਚਸਪ ਗੱਲਾਂ ਸਿੱਖੀਆਂ ਜੋ ਕੁਦਰਤੀ ਟੈਕਸਟ ਪ੍ਰੋਸੈਸਿੰਗ ਨਾਲ ਸੰਬੰਧਿਤ ਹਨ।

## [ਪਾਠ ਬਾਅਦ ਕਵਿਜ਼](https://ff-quizzes.netlify.app/en/ml/)

## ਚੁਣੌਤੀ

ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ ਆਪਣੇ ਡਾਟਾਸੈੱਟ ਨੂੰ ਭਾਵਨਾ ਲਈ ਵਿਸ਼ਲੇਸ਼ਿਤ ਕਰ ਲਿਆ ਹੈ, ਦੇਖੋ ਕਿ ਤੁਸੀਂ ਇਸ ਪਾਠਕ੍ਰਮ ਵਿੱਚ ਸਿੱਖੀਆਂ ਰਣਨੀਤੀਆਂ (ਸ਼ਾਇਦ ਕਲੱਸਟਰਿੰਗ?) ਵਰਤ ਕੇ ਭਾਵਨਾ ਦੇ ਆਸਪਾਸ ਪੈਟਰਨਾਂ ਦੀ ਪਛਾਣ ਕਰ ਸਕਦੇ ਹੋ।

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

[ਇਹ Learn ਮੋਡਿਊਲ](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) ਲਓ ਤਾਂ ਜੋ ਹੋਰ ਸਿੱਖਿਆ ਜਾ ਸਕੇ ਅਤੇ ਵੱਖ-ਵੱਖ ਟੂਲਾਂ ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਟੈਕਸਟ ਵਿੱਚ ਭਾਵਨਾ ਦੀ ਖੋਜ ਕੀਤੀ ਜਾ ਸਕੇ।

## ਅਸਾਈਨਮੈਂਟ

[ਇੱਕ ਵੱਖਰਾ ਡਾਟਾਸੈੱਟ ਅਜ਼ਮਾਓ](assignment.md)

---

**ਅਸਵੀਕਾਰਨਾ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦੀ ਕੋਸ਼ਿਸ਼ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁਣੀਕਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਮੂਲ ਦਸਤਾਵੇਜ਼, ਜੋ ਇਸਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਹੈ, ਨੂੰ ਅਧਿਕਾਰਤ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤ ਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।