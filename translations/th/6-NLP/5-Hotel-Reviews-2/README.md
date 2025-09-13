<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T22:26:32+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "th"
}
-->
# การวิเคราะห์ความรู้สึกด้วยรีวิวโรงแรม

หลังจากที่คุณได้สำรวจชุดข้อมูลอย่างละเอียดแล้ว ถึงเวลาในการกรองคอลัมน์และใช้เทคนิค NLP กับชุดข้อมูลเพื่อค้นหาแนวคิดใหม่เกี่ยวกับโรงแรม

## [แบบทดสอบก่อนการบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

### การกรองข้อมูลและการดำเนินการวิเคราะห์ความรู้สึก

คุณอาจสังเกตเห็นว่าชุดข้อมูลมีปัญหาบางอย่าง เช่น คอลัมน์บางคอลัมน์มีข้อมูลที่ไม่มีประโยชน์ บางคอลัมน์ดูเหมือนจะไม่ถูกต้อง หรือหากถูกต้อง ก็ไม่ชัดเจนว่าคำนวณมาอย่างไร และคำตอบไม่สามารถตรวจสอบได้ด้วยการคำนวณของคุณเอง

## แบบฝึกหัด: การประมวลผลข้อมูลเพิ่มเติมเล็กน้อย

ทำความสะอาดข้อมูลเพิ่มเติมอีกเล็กน้อย เพิ่มคอลัมน์ที่มีประโยชน์ในภายหลัง เปลี่ยนค่าของคอลัมน์อื่น และลบคอลัมน์บางส่วนออกไป

1. การประมวลผลคอลัมน์เบื้องต้น

   1. ลบ `lat` และ `lng`

   2. แทนที่ค่าของ `Hotel_Address` ด้วยค่าต่อไปนี้ (หากที่อยู่มีชื่อเมืองและประเทศเดียวกัน ให้เปลี่ยนเป็นแค่ชื่อเมืองและประเทศ)

      เมืองและประเทศในชุดข้อมูลมีดังนี้:

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

      ตอนนี้คุณสามารถเรียกดูข้อมูลระดับประเทศได้:

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

2. การประมวลผลคอลัมน์รีวิวเมตาของโรงแรม

  1. ลบ `Additional_Number_of_Scoring`

  1. แทนที่ `Total_Number_of_Reviews` ด้วยจำนวนรีวิวทั้งหมดของโรงแรมที่มีอยู่จริงในชุดข้อมูล 

  1. แทนที่ `Average_Score` ด้วยคะแนนที่คำนวณขึ้นเอง

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. การประมวลผลคอลัมน์รีวิว

   1. ลบ `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` และ `days_since_review`

   2. เก็บ `Reviewer_Score`, `Negative_Review` และ `Positive_Review` ไว้ตามเดิม
     
   3. เก็บ `Tags` ไว้ชั่วคราว

     - เราจะทำการกรองเพิ่มเติมในส่วนของแท็กในส่วนถัดไป และจากนั้นจะลบแท็กออก

4. การประมวลผลคอลัมน์ผู้รีวิว

  1. ลบ `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. เก็บ `Reviewer_Nationality` ไว้

### คอลัมน์แท็ก

คอลัมน์ `Tag` มีปัญหาเนื่องจากเป็นรายการ (ในรูปแบบข้อความ) ที่ถูกเก็บไว้ในคอลัมน์ น่าเสียดายที่ลำดับและจำนวนส่วนย่อยในคอลัมน์นี้ไม่เหมือนกันเสมอไป การระบุวลีที่น่าสนใจอาจเป็นเรื่องยากสำหรับมนุษย์ เนื่องจากมีแถว 515,000 แถว และโรงแรม 1427 แห่ง และแต่ละแห่งมีตัวเลือกที่แตกต่างกันเล็กน้อยที่ผู้รีวิวสามารถเลือกได้ นี่คือจุดที่ NLP มีประโยชน์ คุณสามารถสแกนข้อความและค้นหาวลีที่พบบ่อยที่สุดและนับจำนวนได้

น่าเสียดายที่เราไม่ได้สนใจคำเดี่ยว แต่สนใจวลีที่มีหลายคำ (เช่น *Business trip*) การรันอัลกอริธึมการแจกแจงความถี่ของวลีหลายคำในข้อมูลจำนวนมาก (6762646 คำ) อาจใช้เวลานานมาก แต่หากไม่ดูข้อมูล อาจดูเหมือนว่าเป็นสิ่งจำเป็น นี่คือจุดที่การวิเคราะห์ข้อมูลเชิงสำรวจมีประโยชน์ เพราะคุณได้เห็นตัวอย่างของแท็ก เช่น `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` คุณสามารถเริ่มถามได้ว่ามันเป็นไปได้ที่จะลดการประมวลผลที่คุณต้องทำหรือไม่ โชคดีที่เป็นไปได้ แต่ก่อนอื่นคุณต้องทำตามขั้นตอนบางอย่างเพื่อระบุแท็กที่น่าสนใจ

### การกรองแท็ก

จำไว้ว่าเป้าหมายของชุดข้อมูลคือการเพิ่มความรู้สึกและคอลัมน์ที่จะช่วยให้คุณเลือกโรงแรมที่ดีที่สุด (สำหรับตัวคุณเองหรืออาจเป็นงานที่ลูกค้าขอให้คุณสร้างบอทแนะนำโรงแรม) คุณต้องถามตัวเองว่าแท็กมีประโยชน์หรือไม่ในชุดข้อมูลสุดท้าย นี่คือการตีความหนึ่ง (หากคุณต้องการชุดข้อมูลด้วยเหตุผลอื่น แท็กที่เลือกอาจแตกต่างออกไป):

1. ประเภทของการเดินทางมีความเกี่ยวข้อง และควรเก็บไว้
2. ประเภทของกลุ่มผู้เข้าพักมีความสำคัญ และควรเก็บไว้
3. ประเภทของห้อง สวีท หรือสตูดิโอที่ผู้เข้าพักพักอยู่ไม่มีความเกี่ยวข้อง (โรงแรมทั้งหมดมีห้องพื้นฐานเหมือนกัน)
4. อุปกรณ์ที่ใช้ส่งรีวิวไม่มีความเกี่ยวข้อง
5. จำนวนคืนที่ผู้รีวิวพัก *อาจ* มีความเกี่ยวข้องหากคุณเชื่อมโยงการพักนานขึ้นกับการชอบโรงแรมมากขึ้น แต่ก็เป็นการคาดเดา และอาจไม่มีความเกี่ยวข้อง

สรุปคือ **เก็บแท็ก 2 ประเภทและลบประเภทอื่นออก**

ก่อนอื่น คุณไม่ต้องการนับแท็กจนกว่าพวกมันจะอยู่ในรูปแบบที่ดีกว่า ซึ่งหมายถึงการลบวงเล็บเหลี่ยมและเครื่องหมายคำพูด คุณสามารถทำได้หลายวิธี แต่คุณต้องการวิธีที่เร็วที่สุดเนื่องจากอาจใช้เวลานานในการประมวลผลข้อมูลจำนวนมาก โชคดีที่ pandas มีวิธีง่าย ๆ ในการทำแต่ละขั้นตอนเหล่านี้

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

แต่ละแท็กจะกลายเป็นบางอย่างเช่น: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

จากนั้นเราพบปัญหา บางรีวิวหรือแถวมี 5 คอลัมน์ บางแถวมี 3 คอลัมน์ บางแถวมี 6 คอลัมน์ นี่เป็นผลมาจากวิธีการสร้างชุดข้อมูล และแก้ไขได้ยาก คุณต้องการนับความถี่ของแต่ละวลี แต่พวกมันอยู่ในลำดับที่แตกต่างกันในแต่ละรีวิว ดังนั้นการนับอาจผิดพลาด และโรงแรมอาจไม่ได้รับแท็กที่สมควรได้รับ

แทนที่จะใช้ลำดับที่แตกต่างกันให้เป็นประโยชน์ เพราะแต่ละแท็กมีหลายคำแต่ก็แยกกันด้วยเครื่องหมายจุลภาค! วิธีที่ง่ายที่สุดในการทำเช่นนี้คือการสร้างคอลัมน์ชั่วคราว 6 คอลัมน์ โดยแต่ละแท็กจะถูกแทรกลงในคอลัมน์ที่ตรงกับลำดับของมัน จากนั้นคุณสามารถรวมคอลัมน์ทั้ง 6 เข้าด้วยกันเป็นคอลัมน์ใหญ่หนึ่งคอลัมน์และรันเมธอด `value_counts()` บนคอลัมน์ที่ได้ ผลลัพธ์ที่พิมพ์ออกมาจะแสดงว่ามีแท็กที่ไม่ซ้ำกัน 2428 รายการ นี่คือตัวอย่างเล็ก ๆ:

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

แท็กทั่วไปบางอย่าง เช่น `Submitted from a mobile device` ไม่มีประโยชน์สำหรับเรา ดังนั้นอาจเป็นความคิดที่ดีที่จะลบออกก่อนนับการเกิดของวลี แต่เนื่องจากเป็นการดำเนินการที่รวดเร็ว คุณสามารถปล่อยไว้และเพิกเฉยได้

### การลบแท็กที่เกี่ยวกับระยะเวลาการเข้าพัก

การลบแท็กเหล่านี้เป็นขั้นตอนแรก ซึ่งช่วยลดจำนวนแท็กที่ต้องพิจารณาลงเล็กน้อย โปรดทราบว่าคุณไม่ได้ลบแท็กเหล่านี้ออกจากชุดข้อมูล เพียงแค่เลือกที่จะไม่พิจารณาเป็นค่าที่จะนับ/เก็บไว้ในชุดข้อมูลรีวิว

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

มีความหลากหลายของห้อง สวีท สตูดิโอ อพาร์ตเมนต์ และอื่น ๆ มากมาย ทั้งหมดนี้มีความหมายคล้ายกันและไม่มีความเกี่ยวข้องกับคุณ ดังนั้นให้ลบออกจากการพิจารณา

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

สุดท้าย และนี่เป็นเรื่องน่ายินดี (เพราะไม่ต้องใช้การประมวลผลมากนัก) คุณจะเหลือแท็กที่ *มีประโยชน์* ดังนี้:

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

คุณอาจโต้แย้งว่า `Travellers with friends` เหมือนกับ `Group` มากหรือน้อย และนั่นจะเป็นการรวมทั้งสองเข้าด้วยกันตามที่แสดงไว้ด้านบน โค้ดสำหรับการระบุแท็กที่ถูกต้องอยู่ใน [Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb)

ขั้นตอนสุดท้ายคือการสร้างคอลัมน์ใหม่สำหรับแต่ละแท็กเหล่านี้ จากนั้นสำหรับทุกแถวรีวิว หากคอลัมน์ `Tag` ตรงกับหนึ่งในคอลัมน์ใหม่ ให้เพิ่มค่า 1 หากไม่ตรง ให้เพิ่มค่า 0 ผลลัพธ์สุดท้ายจะเป็นจำนวนผู้รีวิวที่เลือกโรงแรมนี้ (ในภาพรวม) สำหรับการเดินทางเพื่อธุรกิจหรือพักผ่อน หรือเพื่อพาสัตว์เลี้ยงมาด้วย และนี่เป็นข้อมูลที่มีประโยชน์เมื่อแนะนำโรงแรม

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

### บันทึกไฟล์ของคุณ

สุดท้าย บันทึกชุดข้อมูลในรูปแบบปัจจุบันด้วยชื่อใหม่

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## การดำเนินการวิเคราะห์ความรู้สึก

ในส่วนสุดท้ายนี้ คุณจะใช้การวิเคราะห์ความรู้สึกกับคอลัมน์รีวิวและบันทึกผลลัพธ์ในชุดข้อมูล

## แบบฝึกหัด: โหลดและบันทึกข้อมูลที่กรองแล้ว

โปรดทราบว่าตอนนี้คุณกำลังโหลดชุดข้อมูลที่กรองแล้วซึ่งถูกบันทึกไว้ในส่วนก่อนหน้า **ไม่ใช่** ชุดข้อมูลต้นฉบับ

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

### การลบคำหยุด

หากคุณรันการวิเคราะห์ความรู้สึกในคอลัมน์รีวิวเชิงลบและเชิงบวก อาจใช้เวลานาน ทดสอบบนแล็ปท็อปที่มี CPU เร็ว ใช้เวลา 12 - 14 นาที ขึ้นอยู่กับไลบรารีการวิเคราะห์ความรู้สึกที่ใช้ นั่นเป็นเวลาที่ค่อนข้างนาน ดังนั้นจึงควรตรวจสอบว่ามีวิธีเร่งความเร็วหรือไม่ 

การลบคำหยุด หรือคำภาษาอังกฤษทั่วไปที่ไม่เปลี่ยนแปลงความรู้สึกของประโยค เป็นขั้นตอนแรก โดยการลบคำเหล่านี้ การวิเคราะห์ความรู้สึกควรทำงานเร็วขึ้น แต่ไม่ลดความแม่นยำ (เนื่องจากคำหยุดไม่ส่งผลต่อความรู้สึก แต่ทำให้การวิเคราะห์ช้าลง) 

รีวิวเชิงลบที่ยาวที่สุดมี 395 คำ แต่หลังจากลบคำหยุดแล้ว เหลือ 195 คำ

การลบคำหยุดเป็นการดำเนินการที่รวดเร็ว การลบคำหยุดจาก 2 คอลัมน์รีวิวใน 515,000 แถวใช้เวลา 3.3 วินาทีบนอุปกรณ์ทดสอบ อาจใช้เวลามากหรือน้อยกว่านี้เล็กน้อยขึ้นอยู่กับความเร็ว CPU ของอุปกรณ์ RAM ว่ามี SSD หรือไม่ และปัจจัยอื่น ๆ ความสั้นของการดำเนินการนี้หมายความว่าหากมันช่วยปรับปรุงเวลาการวิเคราะห์ความรู้สึก ก็ถือว่าคุ้มค่าที่จะทำ

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

### การดำเนินการวิเคราะห์ความรู้สึก

ตอนนี้คุณควรคำนวณการวิเคราะห์ความรู้สึกสำหรับทั้งคอลัมน์รีวิวเชิงลบและเชิงบวก และเก็บผลลัพธ์ไว้ในคอลัมน์ใหม่ 2 คอลัมน์ การทดสอบความรู้สึกจะเปรียบเทียบกับคะแนนของผู้รีวิวสำหรับรีวิวเดียวกัน ตัวอย่างเช่น หากการวิเคราะห์ความรู้สึกคิดว่าความรู้สึกของรีวิวเชิงลบมีค่า 1 (ความรู้สึกเชิงบวกอย่างมาก) และความรู้สึกของรีวิวเชิงบวกมีค่า 1 แต่ผู้รีวิวให้คะแนนโรงแรมต่ำที่สุดเท่าที่จะเป็นไปได้ นั่นหมายความว่าข้อความรีวิวไม่ตรงกับคะแนน หรือเครื่องมือวิเคราะห์ความรู้สึกไม่สามารถรับรู้ความรู้สึกได้อย่างถูกต้อง คุณควรคาดหวังว่าคะแนนความรู้สึกบางส่วนจะผิดพลาดอย่างสิ้นเชิง และมักจะสามารถอธิบายได้ เช่น รีวิวอาจมีการประชดประชันอย่างมาก "แน่นอนว่าฉันชอบนอนในห้องที่ไม่มีเครื่องทำความร้อน" และเครื่องมือวิเคราะห์ความรู้สึกคิดว่านั่นเป็นความรู้สึกเชิงบวก แม้ว่ามนุษย์ที่อ่านจะรู้ว่ามันเป็นการประชดประชัน
NLTK มีตัววิเคราะห์ความรู้สึกหลายแบบให้เลือกใช้ และคุณสามารถเปลี่ยนไปใช้ตัวอื่นเพื่อดูว่าการวิเคราะห์ความรู้สึกนั้นแม่นยำมากขึ้นหรือน้อยลง ตัววิเคราะห์ความรู้สึก VADER ถูกนำมาใช้ในที่นี้

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

ในโปรแกรมของคุณ เมื่อคุณพร้อมที่จะคำนวณความรู้สึก คุณสามารถนำไปใช้กับแต่ละรีวิวได้ดังนี้:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

กระบวนการนี้ใช้เวลาประมาณ 120 วินาทีบนคอมพิวเตอร์ของฉัน แต่เวลาที่ใช้จะขึ้นอยู่กับแต่ละเครื่อง หากคุณต้องการพิมพ์ผลลัพธ์ออกมาและดูว่าความรู้สึกตรงกับรีวิวหรือไม่:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

สิ่งสุดท้ายที่ต้องทำกับไฟล์ก่อนนำไปใช้ในความท้าทายคือการบันทึกไฟล์! คุณควรพิจารณาจัดเรียงคอลัมน์ใหม่ทั้งหมดเพื่อให้ง่ายต่อการใช้งาน (สำหรับมนุษย์ มันเป็นการเปลี่ยนแปลงเชิงความสวยงาม)

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

คุณควรเรียกใช้โค้ดทั้งหมดใน [notebook การวิเคราะห์](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (หลังจากที่คุณเรียกใช้ [notebook การกรอง](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) เพื่อสร้างไฟล์ Hotel_Reviews_Filtered.csv)

เพื่อสรุป ขั้นตอนคือ:

1. ไฟล์ชุดข้อมูลต้นฉบับ **Hotel_Reviews.csv** ถูกสำรวจในบทเรียนก่อนหน้าด้วย [notebook การสำรวจ](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv ถูกกรองโดย [notebook การกรอง](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) และได้ผลลัพธ์เป็น **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv ถูกประมวลผลโดย [notebook การวิเคราะห์ความรู้สึก](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) และได้ผลลัพธ์เป็น **Hotel_Reviews_NLP.csv**
4. ใช้ Hotel_Reviews_NLP.csv ในความท้าทาย NLP ด้านล่าง

### สรุป

เมื่อคุณเริ่มต้น คุณมีชุดข้อมูลที่มีคอลัมน์และข้อมูล แต่ไม่ใช่ทั้งหมดที่สามารถตรวจสอบหรือใช้งานได้ คุณได้สำรวจข้อมูล กรองสิ่งที่ไม่จำเป็น เปลี่ยนแท็กให้เป็นสิ่งที่มีประโยชน์ คำนวณค่าเฉลี่ยของคุณเอง เพิ่มคอลัมน์ความรู้สึก และหวังว่าคุณจะได้เรียนรู้สิ่งที่น่าสนใจเกี่ยวกับการประมวลผลข้อความธรรมชาติ

## [แบบทดสอบหลังการบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

## ความท้าทาย

ตอนนี้คุณได้วิเคราะห์ชุดข้อมูลเพื่อความรู้สึกแล้ว ลองใช้กลยุทธ์ที่คุณได้เรียนรู้ในหลักสูตรนี้ (เช่น การจัดกลุ่ม) เพื่อค้นหารูปแบบเกี่ยวกับความรู้สึก

## ทบทวนและศึกษาด้วยตนเอง

ลองเรียน [โมดูลนี้](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) เพื่อเรียนรู้เพิ่มเติมและใช้เครื่องมือที่แตกต่างกันในการสำรวจความรู้สึกในข้อความ

## งานที่ได้รับมอบหมาย

[ลองใช้ชุดข้อมูลอื่น](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามนุษย์ที่เป็นมืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้