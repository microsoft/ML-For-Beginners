<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T22:15:16+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "th"
}
-->
# การวิเคราะห์ความรู้สึกด้วยรีวิวโรงแรม - การประมวลผลข้อมูล

ในส่วนนี้ คุณจะใช้เทคนิคที่เรียนรู้ในบทเรียนก่อนหน้าเพื่อทำการวิเคราะห์ข้อมูลเบื้องต้นในชุดข้อมูลขนาดใหญ่ เมื่อคุณเข้าใจถึงความมีประโยชน์ของคอลัมน์ต่าง ๆ แล้ว คุณจะได้เรียนรู้:

- วิธีลบคอลัมน์ที่ไม่จำเป็น
- วิธีคำนวณข้อมูลใหม่จากคอลัมน์ที่มีอยู่
- วิธีบันทึกชุดข้อมูลที่ได้เพื่อใช้ในความท้าทายสุดท้าย

## [แบบทดสอบก่อนเรียน](https://ff-quizzes.netlify.app/en/ml/)

### บทนำ

จนถึงตอนนี้ คุณได้เรียนรู้ว่าข้อมูลข้อความนั้นแตกต่างจากข้อมูลประเภทตัวเลขอย่างมาก หากข้อความนั้นถูกเขียนหรือพูดโดยมนุษย์ ข้อมูลนั้นสามารถถูกวิเคราะห์เพื่อค้นหารูปแบบ ความถี่ ความรู้สึก และความหมาย บทเรียนนี้จะนำคุณเข้าสู่ชุดข้อมูลจริงพร้อมความท้าทายจริง: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** ซึ่งมี [CC0: Public Domain license](https://creativecommons.org/publicdomain/zero/1.0/) ข้อมูลนี้ถูกดึงมาจาก Booking.com จากแหล่งข้อมูลสาธารณะ ผู้สร้างชุดข้อมูลคือ Jiashen Liu

### การเตรียมตัว

สิ่งที่คุณต้องมี:

* ความสามารถในการรันไฟล์ .ipynb โดยใช้ Python 3
* pandas
* NLTK, [ซึ่งคุณควรติดตั้งในเครื่อง](https://www.nltk.org/install.html)
* ชุดข้อมูลที่มีอยู่ใน Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe) ขนาดประมาณ 230 MB เมื่อคลายซิปแล้ว ดาวน์โหลดไปยังโฟลเดอร์ `/data` ที่เกี่ยวข้องกับบทเรียน NLP เหล่านี้

## การวิเคราะห์ข้อมูลเบื้องต้น

ความท้าทายนี้สมมติว่าคุณกำลังสร้างบอทแนะนำโรงแรมโดยใช้การวิเคราะห์ความรู้สึกและคะแนนรีวิวของผู้เข้าพัก ชุดข้อมูลที่คุณจะใช้ประกอบด้วยรีวิวของโรงแรม 1493 แห่งใน 6 เมือง

โดยใช้ Python, ชุดข้อมูลรีวิวโรงแรม และการวิเคราะห์ความรู้สึกของ NLTK คุณสามารถค้นหา:

* คำและวลีที่ถูกใช้บ่อยที่สุดในรีวิวคืออะไร?
* *แท็ก* อย่างเป็นทางการที่อธิบายโรงแรมมีความสัมพันธ์กับคะแนนรีวิวหรือไม่ (เช่น รีวิวที่เป็นลบมากขึ้นสำหรับโรงแรมที่ระบุว่าเหมาะสำหรับ *ครอบครัวที่มีเด็กเล็ก* มากกว่า *นักเดินทางคนเดียว* อาจบ่งบอกว่าโรงแรมเหมาะสำหรับ *นักเดินทางคนเดียว* มากกว่า)
* คะแนนความรู้สึกของ NLTK 'สอดคล้อง' กับคะแนนตัวเลขของผู้รีวิวหรือไม่?

#### ชุดข้อมูล

มาสำรวจชุดข้อมูลที่คุณดาวน์โหลดและบันทึกไว้ในเครื่อง เปิดไฟล์ในโปรแกรมแก้ไข เช่น VS Code หรือแม้แต่ Excel

หัวข้อในชุดข้อมูลมีดังนี้:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

นี่คือการจัดกลุ่มในรูปแบบที่อาจง่ายต่อการตรวจสอบ:
##### คอลัมน์โรงแรม

* `Hotel_Name`, `Hotel_Address`, `lat` (ละติจูด), `lng` (ลองจิจูด)
  * โดยใช้ *lat* และ *lng* คุณสามารถสร้างแผนที่ด้วย Python เพื่อแสดงตำแหน่งโรงแรม (อาจใช้สีเพื่อแสดงรีวิวที่เป็นลบและบวก)
  * Hotel_Address อาจไม่เป็นประโยชน์ต่อเรา และเราน่าจะเปลี่ยนเป็นประเทศเพื่อให้ง่ายต่อการจัดเรียงและค้นหา

**คอลัมน์เมตารีวิวของโรงแรม**

* `Average_Score`
  * ตามที่ผู้สร้างชุดข้อมูลระบุ คอลัมน์นี้คือ *คะแนนเฉลี่ยของโรงแรมที่คำนวณจากความคิดเห็นล่าสุดในปีที่ผ่านมา* วิธีการคำนวณคะแนนนี้ดูแปลก แต่เนื่องจากเป็นข้อมูลที่ดึงมา เราอาจยอมรับตามนั้นในตอนนี้
  
  ✅ จากคอลัมน์อื่น ๆ ในข้อมูลนี้ คุณคิดวิธีอื่นในการคำนวณคะแนนเฉลี่ยได้หรือไม่?

* `Total_Number_of_Reviews`
  * จำนวนรีวิวทั้งหมดที่โรงแรมนี้ได้รับ - ไม่ชัดเจน (โดยไม่เขียนโค้ด) ว่าหมายถึงรีวิวในชุดข้อมูลหรือไม่
* `Additional_Number_of_Scoring`
  * หมายถึงมีการให้คะแนนรีวิว แต่ไม่มีการเขียนรีวิวที่เป็นบวกหรือเป็นลบโดยผู้รีวิว

**คอลัมน์รีวิว**

- `Reviewer_Score`
  - เป็นค่าตัวเลขที่มีทศนิยมไม่เกิน 1 ตำแหน่ง ระหว่างค่าต่ำสุดและสูงสุด 2.5 และ 10
  - ไม่ได้อธิบายว่าทำไมคะแนนต่ำสุดที่เป็นไปได้คือ 2.5
- `Negative_Review`
  - หากผู้รีวิวไม่ได้เขียนอะไร คอลัมน์นี้จะมี "**No Negative**"
  - โปรดทราบว่าผู้รีวิวอาจเขียนรีวิวที่เป็นบวกในคอลัมน์ Negative review (เช่น "ไม่มีอะไรแย่เกี่ยวกับโรงแรมนี้")
- `Review_Total_Negative_Word_Counts`
  - จำนวนคำที่เป็นลบสูงขึ้นบ่งชี้คะแนนที่ต่ำลง (โดยไม่ตรวจสอบความรู้สึก)
- `Positive_Review`
  - หากผู้รีวิวไม่ได้เขียนอะไร คอลัมน์นี้จะมี "**No Positive**"
  - โปรดทราบว่าผู้รีวิวอาจเขียนรีวิวที่เป็นลบในคอลัมน์ Positive review (เช่น "ไม่มีอะไรดีเกี่ยวกับโรงแรมนี้เลย")
- `Review_Total_Positive_Word_Counts`
  - จำนวนคำที่เป็นบวกสูงขึ้นบ่งชี้คะแนนที่สูงขึ้น (โดยไม่ตรวจสอบความรู้สึก)
- `Review_Date` และ `days_since_review`
  - อาจมีการใช้มาตรการความสดใหม่หรือความเก่าของรีวิว (รีวิวที่เก่าอาจไม่แม่นยำเท่ากับรีวิวใหม่ เนื่องจากการเปลี่ยนแปลงการบริหารโรงแรม การปรับปรุง หรือการเพิ่มสิ่งอำนวยความสะดวก เช่น สระว่ายน้ำ)
- `Tags`
  - เป็นคำอธิบายสั้น ๆ ที่ผู้รีวิวอาจเลือกเพื่ออธิบายประเภทของแขกที่พวกเขาเป็น (เช่น เดินทางคนเดียวหรือครอบครัว) ประเภทของห้องที่พวกเขาเข้าพัก ระยะเวลาการเข้าพัก และวิธีการส่งรีวิว
  - น่าเสียดายที่การใช้แท็กเหล่านี้มีปัญหา ดูส่วนด้านล่างที่กล่าวถึงความมีประโยชน์ของแท็กเหล่านี้

**คอลัมน์ผู้รีวิว**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - อาจเป็นปัจจัยในโมเดลแนะนำ เช่น หากคุณสามารถกำหนดได้ว่าผู้รีวิวที่มีรีวิวหลายร้อยรายการมีแนวโน้มที่จะเป็นลบมากกว่าบวก อย่างไรก็ตาม ผู้รีวิวของรีวิวใด ๆ ไม่ได้ถูกระบุด้วยรหัสเฉพาะ และดังนั้นจึงไม่สามารถเชื่อมโยงกับชุดรีวิวได้ มีผู้รีวิว 30 คนที่มีรีวิว 100 รายการขึ้นไป แต่ยากที่จะเห็นว่าข้อมูลนี้จะช่วยโมเดลแนะนำได้อย่างไร
- `Reviewer_Nationality`
  - บางคนอาจคิดว่าชาติบางชาติมีแนวโน้มที่จะให้รีวิวที่เป็นบวกหรือลบมากกว่าเนื่องจากลักษณะประจำชาติ โปรดระวังการสร้างมุมมองเชิงเล่าเรื่องแบบนี้ในโมเดลของคุณ สิ่งเหล่านี้เป็นภาพลักษณ์ประจำชาติ (และบางครั้งก็เป็นภาพลักษณ์เชิงเชื้อชาติ) และผู้รีวิวแต่ละคนเป็นบุคคลที่เขียนรีวิวตามประสบการณ์ของพวกเขา อาจถูกกรองผ่านเลนส์หลายแบบ เช่น การเข้าพักโรงแรมครั้งก่อน ระยะทางที่เดินทาง และอารมณ์ส่วนตัว การคิดว่าชาติของพวกเขาเป็นเหตุผลสำหรับคะแนนรีวิวเป็นเรื่องยากที่จะพิสูจน์

##### ตัวอย่าง

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | ขณะนี้ไม่ใช่โรงแรมแต่เป็นไซต์ก่อสร้าง ฉันถูกรบกวนตั้งแต่เช้าตรู่และตลอดทั้งวันด้วยเสียงก่อสร้างที่ไม่สามารถยอมรับได้ขณะพักผ่อนหลังจากการเดินทางไกลและทำงานในห้อง คนงานทำงานตลอดทั้งวัน เช่น ใช้เครื่องเจาะในห้องข้างเคียง ฉันขอเปลี่ยนห้องแต่ไม่มีห้องเงียบให้ เพื่อให้แย่ลงไปอีก ฉันถูกเรียกเก็บเงินเกินราคา ฉันเช็คเอาท์ในตอนเย็นเนื่องจากต้องออกเดินทางแต่เช้าตรู่และได้รับใบเรียกเก็บเงินที่เหมาะสม วันต่อมาโรงแรมเรียกเก็บเงินอีกครั้งโดยไม่ได้รับความยินยอมในราคาที่เกินกว่าที่จองไว้ เป็นสถานที่ที่แย่มาก อย่าทำร้ายตัวเองด้วยการจองที่นี่ | ไม่มีอะไร สถานที่แย่มาก หลีกเลี่ยง | ทริปธุรกิจ คู่รัก ห้องมาตรฐานเตียงคู่ พัก 2 คืน |

ดังที่คุณเห็น แขกคนนี้ไม่ได้มีความสุขกับการเข้าพักที่โรงแรมนี้ โรงแรมมีคะแนนเฉลี่ยที่ดีที่ 7.8 และรีวิว 1945 รายการ แต่ผู้รีวิวนี้ให้คะแนน 2.5 และเขียน 115 คำเกี่ยวกับความลบของการเข้าพัก หากพวกเขาไม่ได้เขียนอะไรเลยในคอลัมน์ Positive_Review คุณอาจสันนิษฐานว่าไม่มีอะไรเป็นบวก แต่พวกเขาเขียนคำเตือน 7 คำ หากเรานับคำแทนที่จะดูความหมายหรือความรู้สึกของคำ เราอาจมีมุมมองที่ผิดเพี้ยนเกี่ยวกับเจตนาของผู้รีวิว แปลกที่คะแนน 2.5 ของพวกเขานั้นน่าสับสน เพราะหากการเข้าพักโรงแรมแย่มาก ทำไมถึงให้คะแนนใด ๆ เลย? การตรวจสอบชุดข้อมูลอย่างใกล้ชิด คุณจะเห็นว่าคะแนนต่ำสุดที่เป็นไปได้คือ 2.5 ไม่ใช่ 0 คะแนนสูงสุดที่เป็นไปได้คือ 10

##### แท็ก

ดังที่กล่าวไว้ข้างต้น ในแวบแรก แนวคิดในการใช้ `Tags` เพื่อจัดหมวดหมู่ข้อมูลดูเหมือนสมเหตุสมผล น่าเสียดายที่แท็กเหล่านี้ไม่ได้มาตรฐาน ซึ่งหมายความว่าในโรงแรมหนึ่ง ตัวเลือกอาจเป็น *Single room*, *Twin room*, และ *Double room* แต่ในโรงแรมถัดไป ตัวเลือกอาจเป็น *Deluxe Single Room*, *Classic Queen Room*, และ *Executive King Room* สิ่งเหล่านี้อาจเป็นสิ่งเดียวกัน แต่มีความหลากหลายมากจนตัวเลือกกลายเป็น:

1. พยายามเปลี่ยนคำศัพท์ทั้งหมดให้เป็นมาตรฐานเดียว ซึ่งยากมาก เพราะไม่ชัดเจนว่าเส้นทางการแปลงจะเป็นอย่างไรในแต่ละกรณี (เช่น *Classic single room* แปลงเป็น *Single room* แต่ *Superior Queen Room with Courtyard Garden or City View* ยากที่จะแปลง)

1. เราสามารถใช้วิธี NLP และวัดความถี่ของคำบางคำ เช่น *Solo*, *Business Traveller*, หรือ *Family with young kids* ที่ใช้กับแต่ละโรงแรม และนำมาพิจารณาในโมเดลแนะนำ  

แท็กมักจะ (แต่ไม่เสมอไป) เป็นฟิลด์เดียวที่มีรายการค่าคั่นด้วยเครื่องหมายจุลภาค 5 ถึง 6 ค่า ซึ่งสอดคล้องกับ *ประเภทของการเดินทาง*, *ประเภทของแขก*, *ประเภทของห้อง*, *จำนวนคืน*, และ *ประเภทของอุปกรณ์ที่ใช้ส่งรีวิว* อย่างไรก็ตาม เนื่องจากผู้รีวิวบางคนไม่ได้กรอกแต่ละฟิลด์ (อาจปล่อยว่างไว้หนึ่งฟิลด์) ค่าจึงไม่ได้อยู่ในลำดับเดียวกันเสมอไป

ตัวอย่างเช่น *ประเภทของกลุ่ม* มีความเป็นไปได้ที่ไม่ซ้ำกัน 1025 รายการในฟิลด์นี้ในคอลัมน์ `Tags` และน่าเสียดายที่มีเพียงบางส่วนเท่านั้นที่อ้างถึงกลุ่ม (บางส่วนเป็นประเภทของห้อง ฯลฯ) หากคุณกรองเฉพาะค่าที่กล่าวถึงครอบครัว ผลลัพธ์จะมีค่าประเภท *Family room* จำนวนมาก หากคุณรวมคำว่า *with* เช่น นับค่าที่มี *Family with* ผลลัพธ์จะดีกว่า โดยมีมากกว่า 80,000 จาก 515,000 ผลลัพธ์ที่มีวลี "Family with young children" หรือ "Family with older children"

นี่หมายความว่าคอลัมน์แท็กไม่ได้ไร้ประโยชน์ต่อเรา แต่จะต้องใช้ความพยายามในการทำให้มีประโยชน์

##### คะแนนเฉลี่ยของโรงแรม

มีความแปลกหรือความไม่สอดคล้องกันในชุดข้อมูลที่ฉันไม่สามารถหาคำตอบได้ แต่ได้แสดงไว้ที่นี่เพื่อให้คุณทราบเมื่อสร้างโมเดลของคุณ หากคุณหาคำตอบได้ โปรดแจ้งให้เราทราบในส่วนการอภิปราย!

ชุดข้อมูลมีคอลัมน์ต่อไปนี้ที่เกี่ยวข้องกับคะแนนเฉลี่ยและจำนวนรีวิว:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

โรงแรมเดียวที่มีรีวิวมากที่สุดในชุดข้อมูลนี้คือ *Britannia International Hotel Canary Wharf* โดยมีรีวิว 4789 รายการจาก 515,000 รายการ แต่หากเราดูค่าของ `Total_Number_of_Reviews` สำหรับโรงแรมนี้ จะเป็น 9086 คุณอาจสันนิษฐานว่ามีคะแนนมากกว่าที่ไม่มีรีวิว ดังนั้นเราอาจเพิ่มค่าคอลัมน์ `Additional_Number_of_Scoring` ค่านั้นคือ 2682 และเมื่อเพิ่มเข้าไปใน 4789 จะได้ 7471 ซึ่งยังคงขาดไป 1615 จาก `Total_Number_of_Reviews`

หากคุณใช้คอลัมน์ `Average_Score` คุณอาจสันนิษฐานว่ามันคือค่าเฉลี่ยของรีวิวในชุดข้อมูล แต่คำอธิบายจาก Kaggle คือ "*คะแนนเฉลี่ยของโรงแรมที่คำนวณจากความคิดเห็นล่าสุดในปีที่ผ่านมา*" ซึ่งดูเหมือนจะไม่เป็นประโยชน์นัก แต่เราสามารถคำนวณค่าเฉลี่ยของเราเองจากคะแนนรีวิวในชุดข้อมูล โดยใช้โรงแรมเดียวกันเป็นตัวอย่าง คะแนนเฉลี่ยของโรงแรมที่ให้ไว้คือ 7.1 แต่คะแนนที่คำนวณได้ (คะแนนเฉลี่ยของผู้รีวิว *ใน* ชุดข้อมูล) คือ 6.8 ซึ่งใกล้เคียงกัน แต่ไม่ใช่ค่าเดียวกัน และเราสามารถเดาได้ว่าคะแนนที่ให้ไว้ในรีวิว `Additional_Number_of_Scoring` เพิ่มค่าเฉลี่ยเป็น 7.1 น่าเสียดายที่ไม่มีวิธีทดสอบหรือพิสูจน์ข้อสันนิษฐานนี้ จึงยากที่จะใช้หรือเชื่อถือ `Average_Score`, `Additional_Number_of_Scoring` และ `Total_Number_of_Reviews` เมื่อพวกมันอ้างอิงถึงข้อมูลที่เราไม่มี

เพื่อทำให้เรื่องซับซ้อนขึ้น โรงแรมที่มีจำนวนรีวิวมากที่สุดอันดับสองมีคะแนนเฉลี่ยที่คำนวณได้คือ 8.12 และคะแนน `Average_Score` ในชุดข้อมูลคือ 8.1 นี่เป็นคะแนนที่ถูกต้องโดยบังเอิญหรือโรงแรมแรกเป็นความไม่สอดคล้องกัน?

ในความเป็นไปได้ที่โรงแรมเหล่านี้อาจเป็นค่าผิดปกติ และอาจเป็นไปได้ว่าค่าที่เหลือส่วนใหญ่สอดคล้องกัน (แต่บางค่าที่ไม่สอดคล้องกันด้วยเหตุผลบางประการ) เราจะเขียนโปรแกรมสั้น ๆ ต่อไปเพื่อสำรวจค่าต่าง ๆ ในชุดข้อมูลและกำหนดการใช้งานที่ถูกต้อง (หรือไม่ใช้งาน) ของค่าต่าง ๆ
> 🚨 หมายเหตุสำคัญ  
>  
> เมื่อทำงานกับชุดข้อมูลนี้ คุณจะเขียนโค้ดที่คำนวณบางสิ่งจากข้อความโดยไม่จำเป็นต้องอ่านหรือวิเคราะห์ข้อความด้วยตัวเอง นี่คือแก่นแท้ของ NLP ซึ่งคือการตีความความหมายหรืออารมณ์โดยไม่ต้องให้มนุษย์ทำเอง อย่างไรก็ตาม มีความเป็นไปได้ที่คุณจะอ่านรีวิวเชิงลบบางส่วน ขอแนะนำว่าอย่าทำ เพราะคุณไม่จำเป็นต้องอ่าน บางรีวิวอาจดูไร้สาระ หรือเป็นรีวิวเชิงลบที่ไม่เกี่ยวข้องกับโรงแรม เช่น "อากาศไม่ดี" ซึ่งเป็นสิ่งที่อยู่นอกเหนือการควบคุมของโรงแรม หรือแม้แต่ใครก็ตาม แต่ก็มีด้านมืดของรีวิวบางส่วนเช่นกัน บางครั้งรีวิวเชิงลบอาจมีเนื้อหาเหยียดเชื้อชาติ เหยียดเพศ หรือเหยียดอายุ ซึ่งเป็นเรื่องน่าเสียดายแต่ก็เป็นสิ่งที่คาดการณ์ได้ในชุดข้อมูลที่ดึงมาจากเว็บไซต์สาธารณะ บางคนเขียนรีวิวที่คุณอาจรู้สึกไม่พอใจ ไม่สบายใจ หรือสะเทือนใจ ดังนั้นควรปล่อยให้โค้ดวัดอารมณ์แทนที่จะอ่านด้วยตัวเองและรู้สึกไม่ดี อย่างไรก็ตาม มีเพียงส่วนน้อยที่เขียนสิ่งเหล่านี้ แต่ก็ยังมีอยู่
## แบบฝึกหัด - การสำรวจข้อมูล
### โหลดข้อมูล

พอแล้วกับการตรวจสอบข้อมูลด้วยสายตา ตอนนี้คุณจะเขียนโค้ดเพื่อหาคำตอบ! ส่วนนี้จะใช้ไลบรารี pandas งานแรกของคุณคือการตรวจสอบว่าคุณสามารถโหลดและอ่านข้อมูล CSV ได้หรือไม่ ไลบรารี pandas มีตัวโหลด CSV ที่รวดเร็ว และผลลัพธ์จะถูกเก็บไว้ใน dataframe เช่นเดียวกับบทเรียนก่อนหน้า ไฟล์ CSV ที่เรากำลังโหลดมีมากกว่าครึ่งล้านแถว แต่มีเพียง 17 คอลัมน์เท่านั้น Pandas มีวิธีการที่ทรงพลังมากมายในการโต้ตอบกับ dataframe รวมถึงความสามารถในการดำเนินการกับทุกแถว

จากนี้ไปในบทเรียนนี้ จะมีตัวอย่างโค้ดและคำอธิบายบางส่วนเกี่ยวกับโค้ด รวมถึงการอภิปรายเกี่ยวกับความหมายของผลลัพธ์ ใช้ _notebook.ipynb_ ที่ให้มาเพื่อเขียนโค้ดของคุณ

เริ่มต้นด้วยการโหลดไฟล์ข้อมูลที่คุณจะใช้:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

เมื่อข้อมูลถูกโหลดแล้ว เราสามารถดำเนินการบางอย่างกับมันได้ เก็บโค้ดนี้ไว้ที่ด้านบนของโปรแกรมสำหรับส่วนถัดไป

## สำรวจข้อมูล

ในกรณีนี้ ข้อมูลได้รับการ *ทำความสะอาด* แล้ว หมายความว่าพร้อมใช้งานและไม่มีตัวอักษรในภาษาอื่นที่อาจทำให้อัลกอริทึมที่คาดหวังเฉพาะตัวอักษรภาษาอังกฤษเกิดปัญหา

✅ คุณอาจต้องทำงานกับข้อมูลที่ต้องการการประมวลผลเบื้องต้นเพื่อจัดรูปแบบก่อนที่จะใช้เทคนิค NLP แต่ไม่ใช่ในครั้งนี้ หากคุณต้องจัดการกับตัวอักษรที่ไม่ใช่ภาษาอังกฤษ คุณจะจัดการอย่างไร?

ใช้เวลาสักครู่เพื่อให้แน่ใจว่าเมื่อข้อมูลถูกโหลดแล้ว คุณสามารถสำรวจข้อมูลด้วยโค้ดได้ เป็นเรื่องง่ายที่จะมุ่งเน้นไปที่คอลัมน์ `Negative_Review` และ `Positive_Review` ซึ่งเต็มไปด้วยข้อความธรรมชาติสำหรับอัลกอริทึม NLP ของคุณในการประมวลผล แต่เดี๋ยวก่อน! ก่อนที่คุณจะกระโดดเข้าสู่ NLP และการวิเคราะห์ความรู้สึก คุณควรทำตามโค้ดด้านล่างเพื่อยืนยันว่าค่าที่ให้มาในชุดข้อมูลตรงกับค่าที่คุณคำนวณด้วย pandas หรือไม่

## การดำเนินการกับ Dataframe

งานแรกในบทเรียนนี้คือการตรวจสอบว่าข้อความต่อไปนี้ถูกต้องหรือไม่โดยการเขียนโค้ดเพื่อตรวจสอบ dataframe (โดยไม่เปลี่ยนแปลงมัน)

> เช่นเดียวกับงานโปรแกรมมิ่งหลายๆ อย่าง มีหลายวิธีในการทำสิ่งนี้ให้สำเร็จ แต่คำแนะนำที่ดีคือทำในวิธีที่ง่ายที่สุดและเข้าใจง่ายที่สุด โดยเฉพาะอย่างยิ่งถ้ามันจะง่ายต่อการเข้าใจเมื่อคุณกลับมาดูโค้ดนี้ในอนาคต ด้วย dataframe มี API ที่ครอบคลุมซึ่งมักจะมีวิธีการทำสิ่งที่คุณต้องการอย่างมีประสิทธิภาพ

ปฏิบัติต่อคำถามต่อไปนี้เป็นงานเขียนโค้ดและพยายามตอบโดยไม่ดูคำตอบ

1. พิมพ์ *shape* ของ dataframe ที่คุณเพิ่งโหลด (shape คือจำนวนแถวและคอลัมน์)
2. คำนวณจำนวนครั้งที่ปรากฏของสัญชาติผู้รีวิว:
   1. มีค่าที่แตกต่างกันกี่ค่าในคอลัมน์ `Reviewer_Nationality` และมีค่าอะไรบ้าง?
   2. สัญชาติผู้รีวิวใดที่พบมากที่สุดในชุดข้อมูล (พิมพ์ชื่อประเทศและจำนวนรีวิว)?
   3. สัญชาติที่พบมากที่สุด 10 อันดับถัดไปและจำนวนครั้งที่พบคืออะไร?
3. โรงแรมที่ถูกรีวิวมากที่สุดสำหรับแต่ละสัญชาติผู้รีวิว 10 อันดับแรกคืออะไร?
4. มีรีวิวกี่รายการต่อโรงแรม (จำนวนครั้งที่โรงแรมถูกรีวิว) ในชุดข้อมูล?
5. แม้ว่าจะมีคอลัมน์ `Average_Score` สำหรับแต่ละโรงแรมในชุดข้อมูล คุณยังสามารถคำนวณคะแนนเฉลี่ย (โดยการหาค่าเฉลี่ยของคะแนนผู้รีวิวทั้งหมดในชุดข้อมูลสำหรับแต่ละโรงแรม) เพิ่มคอลัมน์ใหม่ใน dataframe ของคุณโดยใช้ชื่อคอลัมน์ `Calc_Average_Score` ที่มีค่าเฉลี่ยที่คำนวณได้
6. มีโรงแรมใดบ้างที่มีค่า `Average_Score` และ `Calc_Average_Score` เท่ากัน (ปัดเศษทศนิยม 1 ตำแหน่ง)?
   1. ลองเขียนฟังก์ชัน Python ที่รับ Series (แถว) เป็นอาร์กิวเมนต์และเปรียบเทียบค่า โดยพิมพ์ข้อความเมื่อค่าต่างกัน จากนั้นใช้เมธอด `.apply()` เพื่อประมวลผลทุกแถวด้วยฟังก์ชัน
7. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Negative_Review` เป็น "No Negative"
8. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Positive_Review` เป็น "No Positive"
9. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Positive_Review` เป็น "No Positive" **และ** `Negative_Review` เป็น "No Negative"

### คำตอบโค้ด

1. พิมพ์ *shape* ของ dataframe ที่คุณเพิ่งโหลด (shape คือจำนวนแถวและคอลัมน์)

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. คำนวณจำนวนครั้งที่ปรากฏของสัญชาติผู้รีวิว:

   1. มีค่าที่แตกต่างกันกี่ค่าในคอลัมน์ `Reviewer_Nationality` และมีค่าอะไรบ้าง?
   2. สัญชาติผู้รีวิวใดที่พบมากที่สุดในชุดข้อมูล (พิมพ์ชื่อประเทศและจำนวนรีวิว)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. สัญชาติที่พบมากที่สุด 10 อันดับถัดไปและจำนวนครั้งที่พบคืออะไร?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. โรงแรมที่ถูกรีวิวมากที่สุดสำหรับแต่ละสัญชาติผู้รีวิว 10 อันดับแรกคืออะไร?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. มีรีวิวกี่รายการต่อโรงแรม (จำนวนครั้งที่โรงแรมถูกรีวิว) ในชุดข้อมูล?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |
   
   คุณอาจสังเกตเห็นว่าผลลัพธ์ *ที่นับในชุดข้อมูล* ไม่ตรงกับค่าที่อยู่ใน `Total_Number_of_Reviews` ไม่ชัดเจนว่าค่านี้ในชุดข้อมูลแสดงถึงจำนวนรีวิวทั้งหมดที่โรงแรมมี แต่ไม่ได้ถูกดึงข้อมูลทั้งหมด หรือเป็นการคำนวณอื่น `Total_Number_of_Reviews` ไม่ได้ถูกใช้ในโมเดลเนื่องจากความไม่ชัดเจนนี้

5. แม้ว่าจะมีคอลัมน์ `Average_Score` สำหรับแต่ละโรงแรมในชุดข้อมูล คุณยังสามารถคำนวณคะแนนเฉลี่ย (โดยการหาค่าเฉลี่ยของคะแนนผู้รีวิวทั้งหมดในชุดข้อมูลสำหรับแต่ละโรงแรม) เพิ่มคอลัมน์ใหม่ใน dataframe ของคุณโดยใช้ชื่อคอลัมน์ `Calc_Average_Score` ที่มีค่าเฉลี่ยที่คำนวณได้ พิมพ์คอลัมน์ `Hotel_Name`, `Average_Score` และ `Calc_Average_Score`

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   คุณอาจสงสัยเกี่ยวกับค่าของ `Average_Score` และเหตุใดจึงแตกต่างจากคะแนนเฉลี่ยที่คำนวณได้ในบางครั้ง เนื่องจากเราไม่สามารถทราบได้ว่าทำไมบางค่าจึงตรงกัน แต่บางค่ามีความแตกต่าง ในกรณีนี้ปลอดภัยที่สุดที่จะใช้คะแนนรีวิวที่เรามีเพื่อคำนวณค่าเฉลี่ยด้วยตัวเอง อย่างไรก็ตาม ความแตกต่างมักจะเล็กน้อย นี่คือโรงแรมที่มีความแตกต่างมากที่สุดระหว่างค่าเฉลี่ยในชุดข้อมูลและค่าเฉลี่ยที่คำนวณได้:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   มีเพียงโรงแรมเดียวที่มีความแตกต่างของคะแนนมากกว่า 1 หมายความว่าเราสามารถมองข้ามความแตกต่างนี้และใช้คะแนนเฉลี่ยที่คำนวณได้

6. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Negative_Review` เป็น "No Negative"

7. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Positive_Review` เป็น "No Positive"

8. คำนวณและพิมพ์จำนวนแถวที่มีค่าคอลัมน์ `Positive_Review` เป็น "No Positive" **และ** `Negative_Review` เป็น "No Negative"

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## อีกวิธีหนึ่ง

อีกวิธีหนึ่งในการนับรายการโดยไม่ใช้ Lambda และใช้ sum เพื่อคำนวณจำนวนแถว:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   คุณอาจสังเกตเห็นว่ามี 127 แถวที่มีค่า "No Negative" และ "No Positive" สำหรับคอลัมน์ `Negative_Review` และ `Positive_Review` ตามลำดับ นั่นหมายความว่าผู้รีวิวให้คะแนนตัวเลขแก่โรงแรม แต่ปฏิเสธที่จะเขียนรีวิวเชิงบวกหรือเชิงลบ โชคดีที่นี่เป็นจำนวนแถวที่น้อยมาก (127 จาก 515738 หรือ 0.02%) ดังนั้นจึงไม่น่าจะทำให้โมเดลหรือผลลัพธ์ของเราผิดเพี้ยนไปในทิศทางใด แต่คุณอาจไม่คาดคิดว่าชุดข้อมูลรีวิวจะมีแถวที่ไม่มีรีวิวเลย ดังนั้นจึงควรสำรวจข้อมูลเพื่อค้นหาแถวแบบนี้

ตอนนี้คุณได้สำรวจชุดข้อมูลแล้ว ในบทเรียนถัดไปคุณจะกรองข้อมูลและเพิ่มการวิเคราะห์ความรู้สึก

---
## 🚀ความท้าทาย

บทเรียนนี้แสดงให้เห็นว่า เช่นเดียวกับที่เราเห็นในบทเรียนก่อนหน้า การทำความเข้าใจข้อมูลและข้อบกพร่องของมันมีความสำคัญอย่างยิ่งก่อนที่จะดำเนินการใดๆ กับมัน ข้อมูลที่เป็นข้อความโดยเฉพาะต้องได้รับการตรวจสอบอย่างละเอียด ลองสำรวจชุดข้อมูลที่มีข้อความจำนวนมากและดูว่าคุณสามารถค้นพบพื้นที่ที่อาจแนะนำอคติหรือความรู้สึกที่ผิดเพี้ยนในโมเดลได้หรือไม่

## [แบบทดสอบหลังการบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

ลอง [เส้นทางการเรียนรู้เกี่ยวกับ NLP นี้](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) เพื่อค้นหาเครื่องมือที่คุณสามารถลองใช้เมื่อสร้างโมเดลที่เน้นข้อความและเสียงพูด

## การบ้าน

[NLTK](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลโดยอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความผิดที่เกิดจากการใช้การแปลนี้