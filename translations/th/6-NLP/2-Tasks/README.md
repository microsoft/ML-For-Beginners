<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T22:12:26+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "th"
}
-->
# งานและเทคนิคทั่วไปในด้านการประมวลผลภาษาธรรมชาติ

สำหรับงาน *การประมวลผลภาษาธรรมชาติ* ส่วนใหญ่ ข้อความที่ต้องการประมวลผลจะต้องถูกแยกส่วน ตรวจสอบ และจัดเก็บผลลัพธ์ หรืออ้างอิงกับกฎและชุดข้อมูล งานเหล่านี้ช่วยให้โปรแกรมเมอร์สามารถสกัด _ความหมาย_ หรือ _เจตนา_ หรือเพียงแค่ _ความถี่_ ของคำและคำศัพท์ในข้อความได้

## [แบบทดสอบก่อนการบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

มาสำรวจเทคนิคทั่วไปที่ใช้ในการประมวลผลข้อความกัน เทคนิคเหล่านี้เมื่อรวมกับการเรียนรู้ของเครื่องจะช่วยให้คุณวิเคราะห์ข้อความจำนวนมากได้อย่างมีประสิทธิภาพ อย่างไรก็ตาม ก่อนที่จะนำ ML ไปใช้กับงานเหล่านี้ เรามาทำความเข้าใจปัญหาที่ผู้เชี่ยวชาญ NLP มักพบเจอกันก่อน

## งานทั่วไปใน NLP

มีวิธีการวิเคราะห์ข้อความที่คุณกำลังทำงานอยู่หลายวิธี มีงานที่คุณสามารถทำได้ และผ่านงานเหล่านี้คุณจะสามารถเข้าใจข้อความและสรุปผลได้ โดยปกติคุณจะดำเนินการงานเหล่านี้ตามลำดับ

### Tokenization

สิ่งแรกที่อัลกอริทึม NLP ส่วนใหญ่ต้องทำคือการแบ่งข้อความออกเป็นโทเค็นหรือคำ แม้ว่าจะฟังดูง่าย แต่การต้องคำนึงถึงเครื่องหมายวรรคตอนและตัวแบ่งคำและประโยคในภาษาต่าง ๆ อาจทำให้ซับซ้อนได้ คุณอาจต้องใช้วิธีการต่าง ๆ เพื่อกำหนดจุดแบ่ง

![tokenization](../../../../6-NLP/2-Tasks/images/tokenization.png)
> การแบ่งประโยคจาก **Pride and Prejudice** อินโฟกราฟิกโดย [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) เป็นวิธีการแปลงข้อมูลข้อความของคุณให้อยู่ในรูปแบบตัวเลข โดยการทำ embeddings จะทำในลักษณะที่คำที่มีความหมายคล้ายกันหรือคำที่ใช้ร่วมกันจะจัดกลุ่มอยู่ใกล้กัน

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Word embeddings สำหรับประโยคใน **Pride and Prejudice** อินโฟกราฟิกโดย [Jen Looper](https://twitter.com/jenlooper)

✅ ลองใช้ [เครื่องมือที่น่าสนใจนี้](https://projector.tensorflow.org/) เพื่อทดลองกับ word embeddings การคลิกที่คำหนึ่งจะแสดงกลุ่มคำที่คล้ายกัน เช่น 'toy' จะอยู่ในกลุ่มเดียวกับ 'disney', 'lego', 'playstation', และ 'console'

### Parsing & Part-of-speech Tagging

ทุกคำที่ถูกแบ่งเป็นโทเค็นสามารถถูกแท็กเป็นส่วนของคำพูด เช่น คำนาม คำกริยา หรือคำคุณศัพท์ ประโยค `the quick red fox jumped over the lazy brown dog` อาจถูกแท็ก POS เป็น fox = noun, jumped = verb

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> การวิเคราะห์ประโยคจาก **Pride and Prejudice** อินโฟกราฟิกโดย [Jen Looper](https://twitter.com/jenlooper)

การวิเคราะห์โครงสร้างประโยคคือการระบุว่าคำใดเกี่ยวข้องกันในประโยค เช่น `the quick red fox jumped` เป็นลำดับคำคุณศัพท์-คำนาม-คำกริยา ที่แยกออกจากลำดับ `lazy brown dog`

### Word and Phrase Frequencies

กระบวนการที่มีประโยชน์เมื่อวิเคราะห์ข้อความจำนวนมากคือการสร้างพจนานุกรมของทุกคำหรือวลีที่สนใจและความถี่ที่ปรากฏ วลี `the quick red fox jumped over the lazy brown dog` มีความถี่ของคำว่า the เท่ากับ 2

มาดูตัวอย่างข้อความที่เรานับความถี่ของคำกัน บทกวี The Winners ของ Rudyard Kipling มีบทดังนี้:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

เนื่องจากความถี่ของวลีสามารถไม่คำนึงถึงตัวพิมพ์ใหญ่หรือตัวพิมพ์เล็กได้ตามต้องการ วลี `a friend` มีความถี่เท่ากับ 2 และ `the` มีความถี่เท่ากับ 6 และ `travels` มีความถี่เท่ากับ 2

### N-grams

ข้อความสามารถถูกแบ่งออกเป็นลำดับของคำที่มีความยาวกำหนด เช่น คำเดียว (unigram), สองคำ (bigrams), สามคำ (trigrams) หรือจำนวนคำใด ๆ (n-grams)

ตัวอย่างเช่น `the quick red fox jumped over the lazy brown dog` ด้วยคะแนน n-gram เท่ากับ 2 จะได้ n-grams ดังนี้:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

อาจจะง่ายขึ้นถ้าจินตนาการว่าเป็นกล่องเลื่อนผ่านประโยค นี่คือตัวอย่างสำหรับ n-grams ที่มี 3 คำ โดย n-gram จะถูกเน้นในแต่ละประโยค:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grams sliding window](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> ค่า n-gram เท่ากับ 3: อินโฟกราฟิกโดย [Jen Looper](https://twitter.com/jenlooper)

### Noun phrase Extraction

ในประโยคส่วนใหญ่จะมีคำนามที่เป็นประธานหรือกรรมของประโยค ในภาษาอังกฤษมักจะสามารถระบุได้จากการมี 'a' หรือ 'an' หรือ 'the' นำหน้า การระบุประธานหรือกรรมของประโยคโดยการ 'สกัดวลีคำนาม' เป็นงานทั่วไปใน NLP เมื่อพยายามทำความเข้าใจความหมายของประโยค

✅ ในประโยค "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun." คุณสามารถระบุวลีคำนามได้หรือไม่?

ในประโยค `the quick red fox jumped over the lazy brown dog` มีวลีคำนาม 2 วลี: **quick red fox** และ **lazy brown dog**

### Sentiment analysis

ประโยคหรือข้อความสามารถถูกวิเคราะห์เพื่อดูอารมณ์ หรือว่า *เป็นบวก* หรือ *เป็นลบ* อารมณ์ถูกวัดในรูปแบบ *polarity* และ *objectivity/subjectivity* โดย polarity วัดจาก -1.0 ถึง 1.0 (ลบถึงบวก) และ 0.0 ถึง 1.0 (วัตถุประสงค์ที่สุดถึงอัตวิสัยที่สุด)

✅ ในภายหลังคุณจะได้เรียนรู้ว่ามีวิธีการต่าง ๆ ในการกำหนดอารมณ์โดยใช้การเรียนรู้ของเครื่อง แต่หนึ่งในวิธีคือการมีรายการคำและวลีที่ถูกจัดหมวดหมู่เป็นบวกหรือลบโดยผู้เชี่ยวชาญ และนำโมเดลนั้นไปใช้กับข้อความเพื่อคำนวณคะแนน polarity คุณเห็นไหมว่าวิธีนี้สามารถทำงานได้ในบางสถานการณ์และไม่ดีในบางสถานการณ์?

### Inflection

Inflection ช่วยให้คุณสามารถนำคำมาเปลี่ยนเป็นรูปเอกพจน์หรือพหูพจน์ได้

### Lemmatization

*Lemma* คือรากหรือคำหลักสำหรับชุดคำ เช่น *flew*, *flies*, *flying* มี lemma เป็นคำกริยา *fly*

ยังมีฐานข้อมูลที่มีประโยชน์สำหรับนักวิจัย NLP โดยเฉพาะ:

### WordNet

[WordNet](https://wordnet.princeton.edu/) เป็นฐานข้อมูลของคำ คำพ้อง คำตรงข้าม และรายละเอียดอื่น ๆ สำหรับทุกคำในหลายภาษา เป็นเครื่องมือที่มีประโยชน์มากเมื่อพยายามสร้างการแปล ตัวตรวจสอบการสะกด หรือเครื่องมือภาษาทุกประเภท

## ไลบรารี NLP

โชคดีที่คุณไม่จำเป็นต้องสร้างเทคนิคเหล่านี้ด้วยตัวเอง เพราะมีไลบรารี Python ที่ยอดเยี่ยมที่ทำให้การพัฒนาสำหรับผู้ที่ไม่ได้เชี่ยวชาญใน NLP หรือการเรียนรู้ของเครื่องง่ายขึ้นมาก ในบทเรียนถัดไปจะมีตัวอย่างเพิ่มเติม แต่ที่นี่คุณจะได้เรียนรู้ตัวอย่างที่มีประโยชน์เพื่อช่วยคุณในงานถัดไป

### การฝึกฝน - ใช้ไลบรารี `TextBlob`

มาลองใช้ไลบรารีที่ชื่อว่า TextBlob ซึ่งมี API ที่มีประโยชน์สำหรับจัดการงานประเภทนี้ TextBlob "สร้างขึ้นบนพื้นฐานของ [NLTK](https://nltk.org) และ [pattern](https://github.com/clips/pattern) และทำงานร่วมกันได้ดี"

> หมายเหตุ: มี [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) ที่มีประโยชน์สำหรับ TextBlob ซึ่งแนะนำสำหรับนักพัฒนา Python ที่มีประสบการณ์

เมื่อพยายามระบุ *noun phrases* TextBlob มีตัวเลือกหลายตัวสำหรับการสกัดวลีคำนาม

1. ลองดูที่ `ConllExtractor`

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > เกิดอะไรขึ้นที่นี่? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) คือ "ตัวสกัดวลีคำนามที่ใช้ chunk parsing ที่ฝึกด้วยชุดข้อมูล ConLL-2000" ConLL-2000 หมายถึงการประชุมปี 2000 เกี่ยวกับ Computational Natural Language Learning ซึ่งในปีนั้นมีการจัด workshop เพื่อแก้ปัญหา NLP ที่ยากลำบาก และในปี 2000 คือ noun chunking โมเดลถูกฝึกด้วย Wall Street Journal โดยใช้ "sections 15-18 เป็นข้อมูลการฝึก (211727 tokens) และ section 20 เป็นข้อมูลทดสอบ (47377 tokens)" คุณสามารถดูขั้นตอนที่ใช้ [ที่นี่](https://www.clips.uantwerpen.be/conll2000/chunking/) และ [ผลลัพธ์](https://ifarm.nl/erikt/research/np-chunking.html)

### ความท้าทาย - ปรับปรุงบอทของคุณด้วย NLP

ในบทเรียนก่อนหน้านี้คุณได้สร้างบอท Q&A แบบง่าย ๆ ตอนนี้คุณจะทำให้ Marvin มีความเห็นอกเห็นใจมากขึ้นโดยการวิเคราะห์ข้อความที่คุณป้อนเพื่อดูอารมณ์และพิมพ์คำตอบที่เหมาะสมกับอารมณ์นั้น คุณยังต้องระบุ `noun_phrase` และถามเกี่ยวกับหัวข้อนั้นด้วย

ขั้นตอนของคุณเมื่อสร้างบอทสนทนาที่ดีขึ้น:

1. พิมพ์คำแนะนำเพื่อแนะนำผู้ใช้วิธีการโต้ตอบกับบอท
2. เริ่มลูป 
   1. รับข้อมูลจากผู้ใช้
   2. หากผู้ใช้ขอออก ให้หยุด
   3. ประมวลผลข้อมูลผู้ใช้และกำหนดคำตอบที่เหมาะสมกับอารมณ์
   4. หากตรวจพบ noun phrase ในอารมณ์ ให้เปลี่ยนเป็นรูปพหูพจน์และถามข้อมูลเพิ่มเติมเกี่ยวกับหัวข้อนั้น
   5. พิมพ์คำตอบ
3. กลับไปที่ขั้นตอน 2

นี่คือตัวอย่างโค้ดสำหรับการกำหนดอารมณ์โดยใช้ TextBlob สังเกตว่ามีเพียงสี่ *ระดับ* ของการตอบสนองต่ออารมณ์ (คุณสามารถเพิ่มได้หากต้องการ):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

นี่คือตัวอย่างผลลัพธ์เพื่อเป็นแนวทาง (ข้อมูลผู้ใช้เริ่มต้นด้วย >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

หนึ่งในวิธีแก้ปัญหาที่เป็นไปได้สำหรับงานนี้คือ [ที่นี่](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ ตรวจสอบความรู้

1. คุณคิดว่าการตอบสนองที่เห็นอกเห็นใจจะ 'หลอก' ให้คนคิดว่าบอทเข้าใจพวกเขาจริง ๆ ได้หรือไม่?
2. การระบุ noun phrase ทำให้บอทดู 'น่าเชื่อถือ' มากขึ้นหรือไม่?
3. ทำไมการสกัด 'noun phrase' จากประโยคถึงเป็นสิ่งที่มีประโยชน์?

---

ลองสร้างบอทในแบบตรวจสอบความรู้ข้างต้นและทดสอบกับเพื่อนของคุณ มันสามารถหลอกพวกเขาได้หรือไม่? คุณสามารถทำให้บอทของคุณดู 'น่าเชื่อถือ' มากขึ้นได้หรือไม่?

## 🚀ความท้าทาย

ลองทำงานในแบบตรวจสอบความรู้ข้างต้นและพยายามนำไปใช้ ทดสอบบอทกับเพื่อนของคุณ มันสามารถหลอกพวกเขาได้หรือไม่? คุณสามารถทำให้บอทของคุณดู 'น่าเชื่อถือ' มากขึ้นได้หรือไม่?

## [แบบทดสอบหลังการบรรยาย](https://ff-quizzes.netlify.app/en/ml/)

## ทบทวนและศึกษาด้วยตนเอง

ในบทเรียนถัดไปคุณจะได้เรียนรู้เพิ่มเติมเกี่ยวกับการวิเคราะห์อารมณ์ ค้นคว้าเทคนิคที่น่าสนใจนี้ในบทความ เช่น บทความใน [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## งานที่ได้รับมอบหมาย 

[ทำให้บอทตอบกลับ](assignment.md)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้องมากที่สุด แต่โปรดทราบว่าการแปลโดยอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้