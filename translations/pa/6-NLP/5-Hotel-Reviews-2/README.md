<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T18:41:30+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "pa"
}
-->
# ਹੋਟਲ ਰਿਵਿਊਜ਼ ਨਾਲ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ

ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ ਡਾਟਾਸੈੱਟ ਨੂੰ ਵਿਸਤਾਰ ਵਿੱਚ ਖੋਜ ਲਿਆ ਹੈ, ਤਾਂ ਕਾਲਮਾਂ ਨੂੰ ਫਿਲਟਰ ਕਰਨ ਅਤੇ ਹੋਟਲਾਂ ਬਾਰੇ ਨਵੇਂ ਅਨੁਮਾਨ ਪ੍ਰਾਪਤ ਕਰਨ ਲਈ NLP ਤਕਨੀਕਾਂ ਦੀ ਵਰਤੋਂ ਕਰਨ ਦਾ ਸਮਾਂ ਹੈ।  
## [ਪ੍ਰੀ-ਲੈਕਚਰ ਪ੍ਰਸ਼ਨੋਤਰੀ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### ਫਿਲਟਰਿੰਗ ਅਤੇ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਾਰਵਾਈਆਂ

ਜਿਵੇਂ ਤੁਸੀਂ ਸ਼ਾਇਦ ਨੋਟ ਕੀਤਾ ਹੋਵੇਗਾ, ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਕੁਝ ਸਮੱਸਿਆਵਾਂ ਹਨ। ਕੁਝ ਕਾਲਮ ਬੇਕਾਰ ਜਾਣਕਾਰੀ ਨਾਲ ਭਰੇ ਹੋਏ ਹਨ, ਹੋਰ ਗਲਤ ਲੱਗਦੇ ਹਨ। ਜੇ ਉਹ ਸਹੀ ਹਨ, ਤਾਂ ਇਹ ਸਪਸ਼ਟ ਨਹੀਂ ਹੈ ਕਿ ਉਹ ਕਿਵੇਂ ਗਣਨਾ ਕੀਤੇ ਗਏ ਸਨ, ਅਤੇ ਤੁਹਾਡੇ ਆਪਣੇ ਹਿਸਾਬ ਨਾਲ ਉਨ੍ਹਾਂ ਦੇ ਜਵਾਬਾਂ ਦੀ ਸਵੈ-ਪੁਸ਼ਟੀ ਨਹੀਂ ਕੀਤੀ ਜਾ ਸਕਦੀ।

## ਅਭਿਆਸ: ਹੋਰ ਡਾਟਾ ਪ੍ਰੋਸੈਸਿੰਗ

ਡਾਟਾ ਨੂੰ ਹੋਰ ਸਾਫ਼ ਕਰੋ। ਉਹ ਕਾਲਮ ਸ਼ਾਮਲ ਕਰੋ ਜੋ ਬਾਅਦ ਵਿੱਚ ਲਾਭਦਾਇਕ ਹੋਣਗੇ, ਹੋਰ ਕਾਲਮਾਂ ਵਿੱਚ ਮੁੱਲ ਬਦਲੋ, ਅਤੇ ਕੁਝ ਕਾਲਮਾਂ ਨੂੰ ਪੂਰੀ ਤਰ੍ਹਾਂ ਹਟਾਓ।

1. ਸ਼ੁਰੂਆਤੀ ਕਾਲਮ ਪ੍ਰੋਸੈਸਿੰਗ

   1. `lat` ਅਤੇ `lng` ਨੂੰ ਹਟਾਓ

   2. `Hotel_Address` ਦੇ ਮੁੱਲਾਂ ਨੂੰ ਹੇਠਾਂ ਦਿੱਤੇ ਮੁੱਲਾਂ ਨਾਲ ਬਦਲੋ (ਜੇ ਪਤਾ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਦੇ ਨਾਮ ਨੂੰ ਸ਼ਾਮਲ ਕਰਦਾ ਹੈ, ਤਾਂ ਇਸਨੂੰ ਸਿਰਫ਼ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਵਿੱਚ ਬਦਲੋ)।

      ਇਹ ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਸਿਰਫ਼ ਇਹ ਸ਼ਹਿਰ ਅਤੇ ਦੇਸ਼ ਹਨ:

      ਐਮਸਟਰਡਮ, ਨੀਦਰਲੈਂਡ

      ਬਾਰਸਿਲੋਨਾ, ਸਪੇਨ

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

      ਹੁਣ ਤੁਸੀਂ ਦੇਸ਼ ਪੱਧਰੀ ਡਾਟਾ ਕਵੈਰੀ ਕਰ ਸਕਦੇ ਹੋ:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | ਐਮਸਟਰਡਮ, ਨੀਦਰਲੈਂਡ    |    105     |
      | ਬਾਰਸਿਲੋਨਾ, ਸਪੇਨ         |    211     |
      | ਲੰਡਨ, ਯੂਨਾਈਟਡ ਕਿੰਗਡਮ   |    400     |
      | ਮਿਲਾਨ, ਇਟਲੀ             |    162     |
      | ਪੈਰਿਸ, ਫਰਾਂਸ            |    458     |
      | ਵੀਅਨਾ, ਆਸਟਰੀਆ          |    158     |

2. ਹੋਟਲ ਮੈਟਾ-ਰਿਵਿਊ ਕਾਲਮ ਪ੍ਰੋਸੈਸ ਕਰੋ

  1. `Additional_Number_of_Scoring` ਨੂੰ ਹਟਾਓ

  1. `Total_Number_of_Reviews` ਨੂੰ ਉਸ ਹੋਟਲ ਲਈ ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਮੌਜੂਦ ਕੁੱਲ ਰਿਵਿਊਜ਼ ਨਾਲ ਬਦਲੋ

  1. `Average_Score` ਨੂੰ ਸਾਡੇ ਆਪਣੇ ਗਣਨਾ ਕੀਤੇ ਸਕੋਰ ਨਾਲ ਬਦਲੋ

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. ਰਿਵਿਊ ਕਾਲਮ ਪ੍ਰੋਸੈਸ ਕਰੋ

   1. `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` ਅਤੇ `days_since_review` ਨੂੰ ਹਟਾਓ

   2. `Reviewer_Score`, `Negative_Review`, ਅਤੇ `Positive_Review` ਨੂੰ ਜਿਵੇਂ ਦੇ ਤਿਵੇਂ ਰੱਖੋ
     
   3. `Tags` ਨੂੰ ਅਜੇ ਲਈ ਰੱਖੋ

     - ਅਗਲੇ ਭਾਗ ਵਿੱਚ ਟੈਗਸ 'ਤੇ ਕੁਝ ਹੋਰ ਫਿਲਟਰਿੰਗ ਕਾਰਵਾਈਆਂ ਕੀਤੀਆਂ ਜਾਣਗੀਆਂ ਅਤੇ ਫਿਰ ਟੈਗਸ ਨੂੰ ਹਟਾ ਦਿੱਤਾ ਜਾਵੇਗਾ

4. ਰਿਵਿਊਅਰ ਕਾਲਮ ਪ੍ਰੋਸੈਸ ਕਰੋ

  1. `Total_Number_of_Reviews_Reviewer_Has_Given` ਨੂੰ ਹਟਾਓ
  
  2. `Reviewer_Nationality` ਨੂੰ ਰੱਖੋ

### ਟੈਗ ਕਾਲਮ

`Tag` ਕਾਲਮ ਸਮੱਸਿਆਜਨਕ ਹੈ ਕਿਉਂਕਿ ਇਹ ਇੱਕ ਸੂਚੀ (ਪਾਠ ਰੂਪ ਵਿੱਚ) ਹੈ ਜੋ ਕਾਲਮ ਵਿੱਚ ਸਟੋਰ ਕੀਤੀ ਗਈ ਹੈ। ਦੁਖਦਾਈ ਗੱਲ ਇਹ ਹੈ ਕਿ ਇਸ ਕਾਲਮ ਵਿੱਚ ਉਪ-ਵਿਭਾਗਾਂ ਦੀ ਗਿਣਤੀ ਅਤੇ ਕ੍ਰਮ ਹਮੇਸ਼ਾ ਇੱਕੋ ਜਿਹਾ ਨਹੀਂ ਹੁੰਦਾ। ਇੱਕ ਮਨੁੱਖ ਲਈ ਸਹੀ ਵਾਕਾਂਸ਼ਾਂ ਦੀ ਪਛਾਣ ਕਰਨਾ ਮੁਸ਼ਕਲ ਹੈ, ਕਿਉਂਕਿ 515,000 ਕਤਾਰਾਂ ਹਨ, ਅਤੇ 1427 ਹੋਟਲ ਹਨ, ਅਤੇ ਹਰ ਇੱਕ ਦੇ ਕੋਲ ਰਿਵਿਊਅਰ ਦੁਆਰਾ ਚੁਣੇ ਗਏ ਥੋੜ੍ਹੇ ਜਿਹੇ ਵੱਖਰੇ ਵਿਕਲਪ ਹਨ। ਇਹ ਥਾਂ NLP ਦੀ ਮਹੱਤਤਾ ਦਿਖਾਉਂਦੀ ਹੈ। ਤੁਸੀਂ ਪਾਠ ਨੂੰ ਸਕੈਨ ਕਰ ਸਕਦੇ ਹੋ ਅਤੇ ਸਭ ਤੋਂ ਆਮ ਵਾਕਾਂਸ਼ ਲੱਭ ਸਕਦੇ ਹੋ ਅਤੇ ਉਨ੍ਹਾਂ ਦੀ ਗਿਣਤੀ ਕਰ ਸਕਦੇ ਹੋ।

ਦੁਖਦਾਈ ਗੱਲ ਇਹ ਹੈ ਕਿ ਅਸੀਂ ਸਿਰਫ਼ ਇੱਕ-ਸ਼ਬਦ ਵਾਲੇ ਟੈਗਸ ਵਿੱਚ ਦਿਲਚਸਪੀ ਨਹੀਂ ਰੱਖਦੇ, ਸਗੋਂ ਬਹੁ-ਸ਼ਬਦ ਵਾਲੇ ਵਾਕਾਂਸ਼ਾਂ ਵਿੱਚ (ਜਿਵੇਂ ਕਿ *Business trip*)। ਇਸ ਤਰ੍ਹਾਂ ਦੇ ਡਾਟੇ (6762646 ਸ਼ਬਦ) 'ਤੇ ਬਹੁ-ਸ਼ਬਦ ਫ੍ਰਿਕਵੈਂਸੀ ਡਿਸਟ੍ਰੀਬਿਊਸ਼ਨ ਐਲਗੋਰਿਦਮ ਚਲਾਉਣਾ ਬਹੁਤ ਜ਼ਿਆਦਾ ਸਮਾਂ ਲੈ ਸਕਦਾ ਹੈ, ਪਰ ਡਾਟੇ ਨੂੰ ਦੇਖਣ ਤੋਂ ਬਿਨਾਂ, ਇਹ ਲੱਗਦਾ ਹੈ ਕਿ ਇਹ ਲਾਜ਼ਮੀ ਖਰਚਾ ਹੈ। ਇਹ ਥਾਂ ਖੋਜੀ ਡਾਟਾ ਵਿਸ਼ਲੇਸ਼ਣ ਲਾਭਦਾਇਕ ਹੁੰਦਾ ਹੈ, ਕਿਉਂਕਿ ਤੁਸੀਂ ਟੈਗਸ ਦਾ ਨਮੂਨਾ ਦੇਖਿਆ ਹੈ ਜਿਵੇਂ ਕਿ `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` , ਤੁਸੀਂ ਪੁੱਛਣਾ ਸ਼ੁਰੂ ਕਰ ਸਕਦੇ ਹੋ ਕਿ ਕੀ ਇਹ ਸੰਭਵ ਹੈ ਕਿ ਤੁਹਾਨੂੰ ਕਰਨ ਵਾਲੀ ਪ੍ਰੋਸੈਸਿੰਗ ਨੂੰ ਬਹੁਤ ਘਟਾਇਆ ਜਾ ਸਕੇ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ, ਇਹ ਸੰਭਵ ਹੈ - ਪਰ ਪਹਿਲਾਂ ਤੁਹਾਨੂੰ ਦਿਲਚਸਪੀ ਵਾਲੇ ਟੈਗਸ ਦੀ ਪਛਾਣ ਕਰਨ ਲਈ ਕੁਝ ਕਦਮਾਂ ਦੀ ਪਾਲਣਾ ਕਰਨ ਦੀ ਲੋੜ ਹੈ।

### ਟੈਗਸ ਨੂੰ ਫਿਲਟਰ ਕਰਨਾ

ਯਾਦ ਰੱਖੋ ਕਿ ਡਾਟਾਸੈੱਟ ਦਾ ਉਦੇਸ਼ ਭਾਵਨਾ ਅਤੇ ਕਾਲਮ ਸ਼ਾਮਲ ਕਰਨਾ ਹੈ ਜੋ ਤੁਹਾਨੂੰ ਸਭ ਤੋਂ ਵਧੀਆ ਹੋਟਲ ਚੁਣਨ ਵਿੱਚ ਮਦਦ ਕਰੇਗਾ (ਆਪਣੇ ਲਈ ਜਾਂ ਸ਼ਾਇਦ ਕਿਸੇ ਕਲਾਇੰਟ ਲਈ ਜੋ ਤੁਹਾਨੂੰ ਹੋਟਲ ਦੀ ਸਿਫਾਰਸ਼ ਕਰਨ ਵਾਲਾ ਬੋਟ ਬਣਾਉਣ ਲਈ ਕਹਿ ਰਿਹਾ ਹੈ)। ਤੁਹਾਨੂੰ ਆਪਣੇ ਆਪ ਤੋਂ ਪੁੱਛਣ ਦੀ ਲੋੜ ਹੈ ਕਿ ਕੀ ਟੈਗਸ ਅੰਤਮ ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਲਾਭਦਾਇਕ ਹਨ ਜਾਂ ਨਹੀਂ। ਇੱਥੇ ਇੱਕ ਵਿਆਖਿਆ ਹੈ (ਜੇ ਤੁਹਾਨੂੰ ਹੋਰ ਕਾਰਨਾਂ ਲਈ ਡਾਟਾਸੈੱਟ ਦੀ ਲੋੜ ਹੋਵੇ ਤਾਂ ਵੱਖਰੇ ਟੈਗਸ ਚੋਣ/ਬਾਹਰ ਰਹਿ ਸਕਦੇ ਹਨ):

1. ਯਾਤਰਾ ਦੀ ਕਿਸਮ ਸਬੰਧਤ ਹੈ, ਅਤੇ ਇਹ ਰਹਿਣੀ ਚਾਹੀਦੀ ਹੈ
2. ਮਹਿਮਾਨ ਸਮੂਹ ਦੀ ਕਿਸਮ ਮਹੱਤਵਪੂਰਨ ਹੈ, ਅਤੇ ਇਹ ਰਹਿਣੀ ਚਾਹੀਦੀ ਹੈ
3. ਮਹਿਮਾਨ ਜਿਸ ਕਿਸਮ ਦੇ ਕਮਰੇ, ਸੂਟ ਜਾਂ ਸਟੂਡੀਓ ਵਿੱਚ ਰਹਿੰਦਾ ਸੀ, ਉਹ ਗੈਰ-ਲਾਭਦਾਇਕ ਹੈ (ਸਾਰੇ ਹੋਟਲ ਬੁਨਿਆਦੀ ਤੌਰ 'ਤੇ ਇੱਕੋ ਜਿਹੇ ਕਮਰੇ ਰੱਖਦੇ ਹਨ)
4. ਜਿਹੜੇ ਡਿਵਾਈਸ ਤੋਂ ਰਿਵਿਊ ਸਬਮਿਟ ਕੀਤਾ ਗਿਆ ਸੀ, ਉਹ ਗੈਰ-ਲਾਭਦਾਇਕ ਹੈ
5. ਮਹਿਮਾਨ ਨੇ ਕਿੰਨੇ ਰਾਤਾਂ ਲਈ ਰਹਿਣਾ ਕੀਤਾ *ਲਾਭਦਾਇਕ* ਹੋ ਸਕਦਾ ਹੈ ਜੇ ਤੁਸੀਂ ਲੰਬੇ ਰਹਿਣ ਨੂੰ ਹੋਟਲ ਨੂੰ ਪਸੰਦ ਕਰਨ ਨਾਲ ਜੋੜਦੇ ਹੋ, ਪਰ ਇਹ ਇੱਕ ਅਨੁਮਾਨ ਹੈ, ਅਤੇ ਸ਼ਾਇਦ ਗੈਰ-ਲਾਭਦਾਇਕ ਹੈ

ਸਾਰ ਵਿੱਚ, **2 ਕਿਸਮ ਦੇ ਟੈਗਸ ਰੱਖੋ ਅਤੇ ਹੋਰਾਂ ਨੂੰ ਹਟਾਓ**।

ਸਭ ਤੋਂ ਪਹਿਲਾਂ, ਤੁਸੀਂ ਟੈਗਸ ਦੀ ਗਿਣਤੀ ਨਹੀਂ ਕਰਨਾ ਚਾਹੁੰਦੇ ਜਦ ਤੱਕ ਉਹ ਇੱਕ ਵਧੀਆ ਫਾਰਮੈਟ ਵਿੱਚ ਨਾ ਹੋਣ, ਇਸ ਲਈ ਇਸਦਾ ਮਤਲਬ ਹੈ ਚੌਰਸ ਬ੍ਰੈਕਟ ਅਤੇ ਕੋਟਸ ਨੂੰ ਹਟਾਉਣਾ। ਤੁਸੀਂ ਇਹ ਕਈ ਤਰੀਕਿਆਂ ਨਾਲ ਕਰ ਸਕਦੇ ਹੋ, ਪਰ ਤੁਸੀਂ ਸਭ ਤੋਂ ਤੇਜ਼ ਤਰੀਕਾ ਚਾਹੁੰਦੇ ਹੋ ਕਿਉਂਕਿ ਇਹ ਬਹੁਤ ਸਾਰੇ ਡਾਟੇ ਨੂੰ ਪ੍ਰੋਸੈਸ ਕਰਨ ਵਿੱਚ ਲੰਮਾ ਸਮਾਂ ਲੈ ਸਕਦਾ ਹੈ। ਖੁਸ਼ਕਿਸਮਤੀ ਨਾਲ, pandas ਵਿੱਚ ਹਰ ਇੱਕ ਕਦਮ ਕਰਨ ਦਾ ਇੱਕ ਆਸਾਨ ਤਰੀਕਾ ਹੈ।

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

ਹਰ ਟੈਗ ਕੁਝ ਇਸ ਤਰ੍ਹਾਂ ਬਣ ਜਾਂਦਾ ਹੈ: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`। 

ਅਗਲਾ ਸਮੱਸਿਆ ਆਉਂਦੀ ਹੈ। ਕੁਝ ਰਿਵਿਊਜ਼, ਜਾਂ ਕਤਾਰਾਂ, ਵਿੱਚ 5 ਕਾਲਮ ਹਨ, ਕੁਝ ਵਿੱਚ 3, ਕੁਝ ਵਿੱਚ 6। ਇਹ ਇਸ ਤਰ੍ਹਾਂ ਹੈ ਕਿ ਡਾਟਾਸੈੱਟ ਕਿਵੇਂ ਬਣਾਇਆ ਗਿਆ ਸੀ, ਅਤੇ ਇਸਨੂੰ ਠੀਕ ਕਰਨਾ ਮੁਸ਼ਕਲ ਹੈ। ਤੁਸੀਂ ਹਰ ਵਾਕਾਂਸ਼ ਦੀ ਗਿਣਤੀ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ, ਪਰ ਉਹ ਹਰ ਰਿਵਿਊ ਵਿੱਚ ਵੱਖਰੇ ਕ੍ਰਮ ਵਿੱਚ ਹਨ, ਇਸ ਲਈ ਗਿਣਤੀ ਗਲਤ ਹੋ ਸਕਦੀ ਹੈ, ਅਤੇ ਇੱਕ ਹੋਟਲ ਨੂੰ ਉਹ ਟੈਗ ਨਹੀਂ ਮਿਲ ਸਕਦਾ ਜਿਸਦਾ ਇਹ ਹੱਕਦਾਰ ਸੀ।

ਇਸਦੀ ਬਜਾਏ ਤੁਸੀਂ ਵੱਖਰੇ ਕ੍ਰਮ ਨੂੰ ਆਪਣੇ ਲਾਭ ਲਈ ਵਰਤੋਂਗੇ, ਕਿਉਂਕਿ ਹਰ ਟੈਗ ਬਹੁ-ਸ਼ਬਦ ਵਾਲਾ ਹੈ ਪਰ ਇੱਕ ਕਾਮਾ ਨਾਲ ਵੀ ਵੱਖਰਾ ਕੀਤਾ ਗਿਆ ਹੈ! ਇਸਦਾ ਸਭ ਤੋਂ ਆਸਾਨ ਤਰੀਕਾ ਇਹ ਹੈ ਕਿ 6 ਅਸਥਾਈ ਕਾਲਮ ਬਣਾਏ ਜਾਣ ਜਿੱਥੇ ਹਰ ਟੈਗ ਨੂੰ ਉਸਦੇ ਕ੍ਰਮ ਦੇ ਅਨੁਸਾਰ ਕਾਲਮ ਵਿੱਚ ਸ਼ਾਮਲ ਕੀਤਾ ਜਾਵੇ। ਫਿਰ ਤੁਸੀਂ 6 ਕਾਲਮਾਂ ਨੂੰ ਇੱਕ ਵੱਡੇ ਕਾਲਮ ਵਿੱਚ ਮਿਲਾ ਸਕਦੇ ਹੋ ਅਤੇ resulting ਕਾਲਮ 'ਤੇ `value_counts()` ਵਿਧੀ ਚਲਾ ਸਕਦੇ ਹੋ। ਇਸਨੂੰ ਪ੍ਰਿੰਟ ਕਰਦੇ ਹੋਏ, ਤੁਸੀਂ ਦੇਖੋਗੇ ਕਿ 2428 ਵਿਲੱਖਣ ਟੈਗਸ ਸਨ। ਇੱਥੇ ਇੱਕ ਛੋਟਾ ਨਮੂਨਾ ਹੈ:

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

ਕੁਝ ਆਮ ਟੈਗਸ ਜਿਵੇਂ ਕਿ `Submitted from a mobile device` ਸਾਡੇ ਲਈ ਕਿਸੇ ਕੰਮ ਦੇ ਨਹੀਂ ਹਨ, ਇਸ ਲਈ ਇਹਨਾਂ ਨੂੰ ਗਿਣਤੀ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਹਟਾਉਣਾ ਸਮਝਦਾਰ ਗੱਲ ਹੋ ਸਕਦੀ ਹੈ, ਪਰ ਇਹ ਇੱਕ ਤੇਜ਼ ਕਾਰਵਾਈ ਹੈ ਇਸ ਲਈ ਤੁਸੀਂ ਇਹਨਾਂ ਨੂੰ ਰੱਖ ਸਕਦੇ ਹੋ ਅਤੇ ਇਹਨਾਂ ਨੂੰ ਅਣਡਿੱਠਾ ਕਰ ਸਕਦੇ ਹੋ।

### ਰਹਿਣ ਦੀ ਮਿਆਦ ਵਾਲੇ ਟੈਗਸ ਨੂੰ ਹਟਾਉਣਾ

ਇਹ ਟੈਗਸ ਨੂੰ ਹਟਾਉਣਾ ਪਹਿਲਾ ਕਦਮ ਹੈ, ਇਹ ਗਿਣਤੀ ਕਰਨ ਵਾਲੇ ਟੈਗਸ ਦੀ ਕੁੱਲ ਗਿਣਤੀ ਨੂੰ ਥੋੜ੍ਹਾ ਘਟਾਉਂਦਾ ਹੈ। ਨੋਟ ਕਰੋ ਕਿ ਤੁਸੀਂ ਇਹਨਾਂ ਨੂੰ ਡਾਟਾਸੈੱਟ ਤੋਂ ਹਟਾਉਂਦੇ ਨਹੀਂ ਹੋ, ਸਿਰਫ਼ ਇਹਨਾਂ ਨੂੰ ਰਿਵਿਊਜ਼ ਡਾਟਾਸੈੱਟ ਵਿੱਚ ਗਿਣਤੀ/ਰੱਖਣ ਲਈ ਮੁਲਾਂਕਣ ਤੋਂ ਹਟਾਉਂਦੇ ਹੋ।

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

ਕਮਰੇ, ਸੂਟ, ਸਟੂਡੀਓ, ਅਪਾਰਟਮੈਂਟ ਆਦਿ ਦੀਆਂ ਕਿਸਮਾਂ ਦੀ ਇੱਕ ਵੱਡੀ ਕਿਸਮ ਹੈ। ਇਹ ਸਾਰੇ ਲਗਭਗ ਇੱਕੋ ਜਿਹੇ ਹਨ ਅਤੇ ਸਾਡੇ ਲਈ ਲਾਭਦਾਇਕ ਨਹੀਂ ਹਨ, ਇਸ ਲਈ ਇਹਨਾਂ ਨੂੰ ਮੁਲਾਂਕਣ ਤੋਂ ਹਟਾਓ।

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

ਅੰਤ ਵਿੱਚ, ਅਤੇ ਇਹ ਖੁਸ਼ੀ ਦੀ ਗੱਲ ਹੈ (ਕਿਉਂਕਿ ਇਸ ਵਿੱਚ ਬਹੁਤ ਘੱਟ ਪ੍ਰੋਸੈਸਿੰਗ ਲੱਗੀ), ਤੁਸੀਂ ਹੇਠਾਂ ਦਿੱਤੇ *ਲਾਭਦਾਇਕ* ਟੈਗਸ ਨਾਲ ਰਹਿ ਜਾਵੋਗੇ:

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

ਤੁਸੀਂ ਦਲੀਲ ਕਰ ਸਕਦੇ ਹੋ ਕਿ `Travellers with friends` ਲਗਭਗ `Group` ਦੇ ਬਰਾਬਰ ਹੈ, ਅਤੇ ਇਹ ਦੋਨੋਂ ਨੂੰ ਉਪਰੋਕਤ ਤਰ੍ਹਾਂ ਮਿਲਾਉਣਾ ਠੀਕ ਹੋਵੇਗਾ। ਸਹੀ ਟੈਗਸ ਦੀ ਪਛਾਣ ਕਰਨ ਲਈ ਕੋਡ [Tags ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ਵਿੱਚ ਹੈ।

ਅੰਤਮ ਕਦਮ ਹਰ ਇੱਕ ਟੈਗ ਲਈ ਨਵੇਂ ਕਾਲਮ ਬਣਾਉਣਾ ਹੈ। ਫਿਰ, ਹਰ ਰਿਵਿਊ ਕਤਾਰ ਲਈ, ਜੇ `Tag` ਕਾਲਮ ਨਵੇਂ ਕਾਲਮਾਂ ਵਿੱਚੋਂ ਕਿਸੇ ਇੱਕ ਨਾਲ ਮੇਲ ਖਾਂਦਾ ਹੈ, ਤਾਂ 1 ਸ਼ਾਮਲ ਕਰੋ, ਨਹੀਂ ਤਾਂ 0 ਸ਼ਾਮਲ ਕਰੋ। ਅੰਤਮ ਨਤੀਜਾ ਇਹ ਹੋਵੇਗਾ ਕਿ ਕਿੰਨੇ ਰਿਵਿਊਅਰਾਂ ਨੇ ਇਸ ਹੋਟਲ ਨੂੰ (ਕੁੱਲ ਮਿਲਾ ਕੇ) ਕਾਰੋਬਾਰ ਵਿਰੁੱਧ ਮਨੋਰੰਜਨ ਲਈ ਚੁਣਿਆ, ਜਾਂ ਪਾਲਤੂ ਜਾਨਵਰ ਲਿਆਉਣ ਲਈ, ਅਤੇ ਇਹ ਜਾਣਕਾਰੀ ਹੋਟਲ ਦੀ ਸਿਫਾਰਸ਼ ਕਰਨ ਸਮੇਂ ਲਾਭਦਾਇਕ ਹੈ।

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

### ਆਪਣੀ ਫਾਈਲ ਸੇਵ ਕਰੋ

ਅੰਤ ਵਿੱਚ, ਡਾਟਾਸੈੱਟ ਨੂੰ ਹੁਣ ਦੇ ਰੂਪ ਵਿੱਚ ਇੱਕ ਨਵੇਂ ਨਾਮ ਨਾਲ ਸੇਵ ਕਰੋ।

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਾਰਵਾਈਆਂ

ਇਸ ਅੰਤਮ ਭਾਗ ਵਿੱਚ, ਤੁਸੀਂ ਰਿਵਿਊ ਕਾਲਮਾਂ 'ਤੇ ਭਾਵਨਾ ਵਿਸ਼ਲੇਸ਼ਣ ਲਾਗੂ ਕਰੋਗੇ ਅਤੇ ਨਤੀਜਿਆਂ ਨੂੰ ਡਾਟਾਸੈੱਟ ਵਿੱਚ
ਹੁਣ ਤੁਹਾਨੂੰ ਨਕਾਰਾਤਮਕ ਅਤੇ ਸਕਾਰਾਤਮਕ ਸਮੀਖਿਆ ਕਾਲਮਾਂ ਲਈ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਦੀ ਗਣਨਾ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ ਅਤੇ ਨਤੀਜੇ ਨੂੰ 2 ਨਵੇਂ ਕਾਲਮਾਂ ਵਿੱਚ ਸਟੋਰ ਕਰਨਾ ਚਾਹੀਦਾ ਹੈ। ਭਾਵਨਾਤਮਕ ਟੈਸਟ ਇਹ ਹੋਵੇਗਾ ਕਿ ਇਸਨੂੰ ਸਮੀਖਿਆਕਾਰ ਦੇ ਅੰਕਾਂ ਨਾਲ ਤੁਲਨਾ ਕਰਨਾ। ਉਦਾਹਰਣ ਲਈ, ਜੇ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਇਹ ਸੋਚਦਾ ਹੈ ਕਿ ਨਕਾਰਾਤਮਕ ਸਮੀਖਿਆ ਦਾ ਭਾਵ 1 ਹੈ (ਬਹੁਤ ਹੀ ਸਕਾਰਾਤਮਕ ਭਾਵਨਾ) ਅਤੇ ਸਕਾਰਾਤਮਕ ਸਮੀਖਿਆ ਦਾ ਭਾਵ 1 ਹੈ, ਪਰ ਸਮੀਖਿਆਕਾਰ ਨੇ ਹੋਟਲ ਨੂੰ ਸਭ ਤੋਂ ਘੱਟ ਅੰਕ ਦਿੱਤੇ ਹਨ, ਤਾਂ ਇਸਦਾ ਮਤਲਬ ਹੈ ਕਿ ਸਮੀਖਿਆ ਦਾ ਪਾਠ ਅੰਕਾਂ ਨਾਲ ਮੇਲ ਨਹੀਂ ਖਾਂਦਾ ਜਾਂ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਕ ਭਾਵਨਾ ਨੂੰ ਸਹੀ ਤਰੀਕੇ ਨਾਲ ਪਛਾਣ ਨਹੀਂ ਸਕਿਆ। ਤੁਹਾਨੂੰ ਕੁਝ ਭਾਵਨਾਤਮਕ ਅੰਕਾਂ ਦੀ ਉਮੀਦ ਕਰਨੀ ਚਾਹੀਦੀ ਹੈ ਕਿ ਉਹ ਪੂਰੀ ਤਰ੍ਹਾਂ ਗਲਤ ਹੋਣਗੇ, ਅਤੇ ਅਕਸਰ ਇਹ ਸਮਝਾਇਆ ਜਾ ਸਕਦਾ ਹੈ, ਜਿਵੇਂ ਕਿ ਸਮੀਖਿਆ ਬਹੁਤ ਹੀ ਵਿਅੰਗਪੂਰਨ ਹੋ ਸਕਦੀ ਹੈ "ਜ਼ਰੂਰ ਮੈਨੂੰ ਬਹੁਤ ਮਜ਼ਾ ਆਇਆ ਇੱਕ ਕਮਰੇ ਵਿੱਚ ਸੌਣ ਦਾ ਜਿਸ ਵਿੱਚ ਹੀਟਿੰਗ ਨਹੀਂ ਸੀ" ਅਤੇ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਕ ਸੋਚਦਾ ਹੈ ਕਿ ਇਹ ਸਕਾਰਾਤਮਕ ਭਾਵਨਾ ਹੈ, ਹਾਲਾਂਕਿ ਇੱਕ ਮਨੁੱਖ ਜੋ ਇਸਨੂੰ ਪੜ੍ਹਦਾ ਹੈ ਉਹ ਜਾਣਦਾ ਹੈ ਕਿ ਇਹ ਵਿਅੰਗ ਹੈ।

NLTK ਵੱਲੋਂ ਵੱਖ-ਵੱਖ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਕ ਪ੍ਰਦਾਨ ਕੀਤੇ ਜਾਂਦੇ ਹਨ ਜਿਨ੍ਹਾਂ ਨਾਲ ਸਿੱਖਿਆ ਜਾ ਸਕਦੀ ਹੈ, ਅਤੇ ਤੁਸੀਂ ਉਨ੍ਹਾਂ ਨੂੰ ਬਦਲ ਸਕਦੇ ਹੋ ਅਤੇ ਦੇਖ ਸਕਦੇ ਹੋ ਕਿ ਕੀ ਭਾਵਨਾ ਹੋਰ ਜ਼ਿਆਦਾ ਜਾਂ ਘੱਟ ਸਹੀ ਹੈ। ਇੱਥੇ VADER ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਵਰਤੀ ਗਈ ਹੈ।

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

ਆਪਣੇ ਪ੍ਰੋਗਰਾਮ ਵਿੱਚ ਬਾਅਦ ਵਿੱਚ ਜਦੋਂ ਤੁਸੀਂ ਭਾਵਨਾ ਦੀ ਗਣਨਾ ਕਰਨ ਲਈ ਤਿਆਰ ਹੋਵੋਗੇ, ਤਾਂ ਤੁਸੀਂ ਇਸਨੂੰ ਹਰੇਕ ਸਮੀਖਿਆ 'ਤੇ ਇਸ ਤਰ੍ਹਾਂ ਲਾਗੂ ਕਰ ਸਕਦੇ ਹੋ:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

ਇਹ ਮੇਰੇ ਕੰਪਿਊਟਰ 'ਤੇ ਲਗਭਗ 120 ਸਕਿੰਟ ਲੈਂਦਾ ਹੈ, ਪਰ ਇਹ ਹਰ ਕੰਪਿਊਟਰ 'ਤੇ ਵੱਖਰਾ ਹੋਵੇਗਾ। ਜੇ ਤੁਸੀਂ ਨਤੀਜੇ ਪ੍ਰਿੰਟ ਕਰਨਾ ਚਾਹੁੰਦੇ ਹੋ ਅਤੇ ਦੇਖਣਾ ਚਾਹੁੰਦੇ ਹੋ ਕਿ ਕੀ ਭਾਵਨਾ ਸਮੀਖਿਆ ਨਾਲ ਮੇਲ ਖਾਂਦੀ ਹੈ:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

ਫਾਈਲ ਨਾਲ ਚੁਣੌਤੀ ਵਿੱਚ ਵਰਤਣ ਤੋਂ ਪਹਿਲਾਂ ਆਖਰੀ ਕੰਮ ਇਹ ਹੈ ਕਿ ਇਸਨੂੰ ਸੇਵ ਕਰੋ! ਤੁਹਾਨੂੰ ਆਪਣੇ ਨਵੇਂ ਕਾਲਮਾਂ ਨੂੰ ਦੁਬਾਰਾ ਕ੍ਰਮਬੱਧ ਕਰਨ ਬਾਰੇ ਵੀ ਸੋਚਣਾ ਚਾਹੀਦਾ ਹੈ ਤਾਂ ਜੋ ਇਹ ਵਰਤਣ ਲਈ ਆਸਾਨ ਹੋਣ (ਇੱਕ ਮਨੁੱਖ ਲਈ, ਇਹ ਸਿਰਫ਼ ਇੱਕ ਦ੍ਰਿਸ਼ਟੀਗਤ ਬਦਲਾਅ ਹੈ)।

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

ਤੁਹਾਨੂੰ [ਵਿਸ਼ਲੇਸ਼ਣ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ਲਈ ਪੂਰਾ ਕੋਡ ਚਲਾਉਣਾ ਚਾਹੀਦਾ ਹੈ (ਜਦੋਂ ਤੁਸੀਂ [ਫਿਲਟਰਿੰਗ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ਚਲਾਇਆ ਹੋਵੇ) Hotel_Reviews_Filtered.csv ਫਾਈਲ ਤਿਆਰ ਕਰਨ ਲਈ।

ਸੰਖੇਪ ਵਿੱਚ, ਕਦਮ ਇਹ ਹਨ:

1. ਮੂਲ ਡਾਟਾਸੈੱਟ ਫਾਈਲ **Hotel_Reviews.csv** ਨੂੰ ਪਿਛਲੇ ਪਾਠ ਵਿੱਚ [ਐਕਸਪਲੋਰਰ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb) ਨਾਲ ਖੋਜਿਆ ਗਿਆ ਸੀ।
2. Hotel_Reviews.csv ਨੂੰ [ਫਿਲਟਰਿੰਗ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) ਦੁਆਰਾ ਫਿਲਟਰ ਕੀਤਾ ਗਿਆ ਜਿਸ ਨਾਲ **Hotel_Reviews_Filtered.csv** ਬਣੀ।
3. Hotel_Reviews_Filtered.csv ਨੂੰ [ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਨੋਟਬੁੱਕ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) ਦੁਆਰਾ ਪ੍ਰਕਿਰਿਆ ਕੀਤੀ ਗਈ ਜਿਸ ਨਾਲ **Hotel_Reviews_NLP.csv** ਬਣੀ।
4. NLP ਚੁਣੌਤੀ ਵਿੱਚ Hotel_Reviews_NLP.csv ਦੀ ਵਰਤੋਂ ਕਰੋ।

### ਨਿਸਕਰਸ਼

ਜਦੋਂ ਤੁਸੀਂ ਸ਼ੁਰੂ ਕੀਤਾ ਸੀ, ਤੁਹਾਡੇ ਕੋਲ ਕਾਲਮਾਂ ਅਤੇ ਡਾਟਾ ਵਾਲਾ ਡਾਟਾਸੈੱਟ ਸੀ ਪਰ ਇਸ ਵਿੱਚੋਂ ਸਾਰਾ ਡਾਟਾ ਪ੍ਰਮਾਣਿਤ ਜਾਂ ਵਰਤਣਯੋਗ ਨਹੀਂ ਸੀ। ਤੁਸੀਂ ਡਾਟੇ ਦੀ ਖੋਜ ਕੀਤੀ, ਜੋ ਲੋੜੀਂਦਾ ਨਹੀਂ ਸੀ ਉਸਨੂੰ ਹਟਾਇਆ, ਟੈਗਾਂ ਨੂੰ ਕੁਝ ਉਪਯੋਗ ਵਿੱਚ ਬਦਲਿਆ, ਆਪਣੇ ਆਪਣੇ ਔਸਤ ਦੀ ਗਣਨਾ ਕੀਤੀ, ਕੁਝ ਭਾਵਨਾਤਮਕ ਕਾਲਮ ਸ਼ਾਮਲ ਕੀਤੇ ਅਤੇ ਉਮੀਦ ਹੈ ਕਿ ਕੁਦਰਤੀ ਪਾਠ ਨੂੰ ਪ੍ਰਕਿਰਿਆ ਕਰਨ ਬਾਰੇ ਕੁਝ ਦਿਲਚਸਪ ਗੱਲਾਂ ਸਿੱਖੀਆਂ।

## [ਪਾਠ-ਪ੍ਰਸ਼ਨੋਤਰੀ](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## ਚੁਣੌਤੀ

ਹੁਣ ਜਦੋਂ ਤੁਸੀਂ ਆਪਣੇ ਡਾਟਾਸੈੱਟ ਦਾ ਭਾਵਨਾਤਮਕ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰ ਲਿਆ ਹੈ, ਤਾਂ ਦੇਖੋ ਕਿ ਕੀ ਤੁਸੀਂ ਇਸ ਪਾਠਕ੍ਰਮ ਵਿੱਚ ਸਿੱਖੀਆਂ ਰਣਨੀਤੀਆਂ (ਸ਼ਾਇਦ ਕਲੱਸਟਰਿੰਗ?) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਭਾਵਨਾ ਦੇ ਆਧਾਰ 'ਤੇ ਪੈਟਰਨ ਪਛਾਣ ਸਕਦੇ ਹੋ।

## ਸਮੀਖਿਆ ਅਤੇ ਸਵੈ ਅਧਿਐਨ

ਇਹ [Learn module](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) ਲਓ ਤਾਂ ਜੋ ਹੋਰ ਸਿੱਖਿਆ ਜਾ ਸਕੇ ਅਤੇ ਪਾਠ ਵਿੱਚ ਭਾਵਨਾ ਦੀ ਖੋਜ ਕਰਨ ਲਈ ਵੱਖ-ਵੱਖ ਟੂਲ ਵਰਤੇ ਜਾ ਸਕਣ।

## ਅਸਾਈਨਮੈਂਟ

[ਇੱਕ ਵੱਖਰਾ ਡਾਟਾਸੈੱਟ ਅਜ਼ਮਾਓ](assignment.md)

---

**ਅਸਵੀਕਰਤੀ**:  
ਇਹ ਦਸਤਾਵੇਜ਼ AI ਅਨੁਵਾਦ ਸੇਵਾ [Co-op Translator](https://github.com/Azure/co-op-translator) ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਅਨੁਵਾਦ ਕੀਤਾ ਗਿਆ ਹੈ। ਜਦੋਂ ਕਿ ਅਸੀਂ ਸਹੀ ਹੋਣ ਦਾ ਯਤਨ ਕਰਦੇ ਹਾਂ, ਕਿਰਪਾ ਕਰਕੇ ਧਿਆਨ ਦਿਓ ਕਿ ਸਵੈਚਾਲਿਤ ਅਨੁਵਾਦਾਂ ਵਿੱਚ ਗਲਤੀਆਂ ਜਾਂ ਅਸੁੱਛਤਾਵਾਂ ਹੋ ਸਕਦੀਆਂ ਹਨ। ਇਸ ਦੀ ਮੂਲ ਭਾਸ਼ਾ ਵਿੱਚ ਮੌਜੂਦ ਮੂਲ ਦਸਤਾਵੇਜ਼ ਨੂੰ ਪ੍ਰਮਾਣਿਕ ਸਰੋਤ ਮੰਨਿਆ ਜਾਣਾ ਚਾਹੀਦਾ ਹੈ। ਮਹੱਤਵਪੂਰਨ ਜਾਣਕਾਰੀ ਲਈ, ਪੇਸ਼ੇਵਰ ਮਨੁੱਖੀ ਅਨੁਵਾਦ ਦੀ ਸਿਫਾਰਸ਼ ਕੀਤੀ ਜਾਂਦੀ ਹੈ। ਇਸ ਅਨੁਵਾਦ ਦੀ ਵਰਤੋਂ ਤੋਂ ਪੈਦਾ ਹੋਣ ਵਾਲੇ ਕਿਸੇ ਵੀ ਗਲਤਫਹਿਮੀ ਜਾਂ ਗਲਤ ਵਿਆਖਿਆ ਲਈ ਅਸੀਂ ਜ਼ਿੰਮੇਵਾਰ ਨਹੀਂ ਹਾਂ।  