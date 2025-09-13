<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T21:24:57+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "th"
}
-->
# โมเดลการจัดกลุ่มสำหรับการเรียนรู้ของเครื่อง

การจัดกลุ่มเป็นงานในด้านการเรียนรู้ของเครื่องที่มุ่งค้นหาวัตถุที่มีความคล้ายคลึงกันและจัดกลุ่มเหล่านี้ให้อยู่ในกลุ่มที่เรียกว่า "คลัสเตอร์" สิ่งที่แตกต่างระหว่างการจัดกลุ่มกับวิธีการอื่นในด้านการเรียนรู้ของเครื่องคือกระบวนการเกิดขึ้นโดยอัตโนมัติ ในความเป็นจริง อาจกล่าวได้ว่ามันตรงกันข้ามกับการเรียนรู้แบบมีผู้สอน

## หัวข้อเฉพาะภูมิภาค: โมเดลการจัดกลุ่มสำหรับรสนิยมทางดนตรีของผู้ฟังชาวไนจีเรีย 🎧

ผู้ฟังชาวไนจีเรียมีรสนิยมทางดนตรีที่หลากหลาย การใช้ข้อมูลที่ดึงมาจาก Spotify (ได้รับแรงบันดาลใจจาก [บทความนี้](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) เรามาดูเพลงที่ได้รับความนิยมในไนจีเรียกัน ชุดข้อมูลนี้ประกอบด้วยข้อมูลเกี่ยวกับคะแนน 'danceability', 'acousticness', ความดัง, 'speechiness', ความนิยม และพลังงานของเพลงต่าง ๆ จะน่าสนใจมากหากเราสามารถค้นพบรูปแบบในข้อมูลนี้!

![เครื่องเล่นแผ่นเสียง](../../../5-Clustering/images/turntable.jpg)

> ภาพถ่ายโดย <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> บน <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
ในบทเรียนชุดนี้ คุณจะได้ค้นพบวิธีใหม่ ๆ ในการวิเคราะห์ข้อมูลโดยใช้เทคนิคการจัดกลุ่ม การจัดกลุ่มมีประโยชน์อย่างยิ่งเมื่อชุดข้อมูลของคุณไม่มีป้ายกำกับ หากมีป้ายกำกับ การใช้เทคนิคการจำแนกประเภท เช่นที่คุณได้เรียนรู้ในบทเรียนก่อนหน้า อาจมีประโยชน์มากกว่า แต่ในกรณีที่คุณต้องการจัดกลุ่มข้อมูลที่ไม่มีป้ายกำกับ การจัดกลุ่มเป็นวิธีที่ยอดเยี่ยมในการค้นหารูปแบบ

> มีเครื่องมือแบบ low-code ที่มีประโยชน์ซึ่งสามารถช่วยคุณเรียนรู้เกี่ยวกับการทำงานกับโมเดลการจัดกลุ่ม ลองใช้ [Azure ML สำหรับงานนี้](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## บทเรียน

1. [แนะนำการจัดกลุ่ม](1-Visualize/README.md)
2. [การจัดกลุ่มแบบ K-Means](2-K-Means/README.md)

## เครดิต

บทเรียนเหล่านี้เขียนขึ้นด้วย 🎶 โดย [Jen Looper](https://www.twitter.com/jenlooper) พร้อมการตรวจสอบที่เป็นประโยชน์จาก [Rishit Dagli](https://rishit_dagli) และ [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)

ชุดข้อมูล [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ได้มาจาก Kaggle โดยดึงข้อมูลจาก Spotify

ตัวอย่าง K-Means ที่มีประโยชน์ซึ่งช่วยในการสร้างบทเรียนนี้ ได้แก่ [การสำรวจดอกไอริส](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [สมุดบันทึกเบื้องต้น](https://www.kaggle.com/prashant111/k-means-clustering-with-python), และ [ตัวอย่าง NGO สมมติ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาต้นทางควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้