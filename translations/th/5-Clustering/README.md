# แบบจำลองการจัดกลุ่มสำหรับการเรียนรู้ของเครื่อง

การจัดกลุ่มเป็นงานการเรียนรู้ของเครื่องที่มุ่งหาวัตถุที่คล้ายคลึงกันและจัดกลุ่มเหล่านี้เป็นกลุ่มที่เรียกว่ากลุ่มจัดกลุ่ม สิ่งที่แตกต่างของการจัดกลุ่มจากวิธีการอื่นในเรียนรู้ของเครื่อง คือสิ่งต่าง ๆ เกิดขึ้นโดยอัตโนมัติ จริง ๆ แล้วสามารถกล่าวได้ว่านี่เป็นสิ่งที่ตรงข้ามกับการเรียนรู้ที่มีผู้ดูแล

## หัวข้อท้องถิ่น: แบบจำลองการจัดกลุ่มสำหรับรสนิยมเพลงของผู้ชมชาวไนจีเรีย 🎧

ผู้ชมที่หลากหลายของไนจีเรียมีรสนิยมทางดนตรีที่หลากหลาย ใช้ข้อมูลที่ขูดจาก Spotify (ได้รับแรงบันดาลใจจาก [บทความนี้](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) มาดูเพลงที่ได้รับความนิยมในไนจีเรีย ชุดข้อมูลนี้ประกอบด้วยข้อมูลเกี่ยวกับคะแนน 'danceability' 'acousticness' ความดัง 'speechiness' ความนิยมและพลังงานของเพลงต่าง ๆ น่าสนใจที่จะค้นหารูปแบบในข้อมูลนี้!

![แผ่นเสียง](../../../translated_images/th/turntable.f2b86b13c53302dc.webp)

> ภาพโดย <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> บน <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
ในชุดบทเรียนนี้ คุณจะค้นพบวิธีใหม่ในการวิเคราะห์ข้อมูลโดยใช้เทคนิคการจัดกลุ่ม การจัดกลุ่มมีประโยชน์เป็นพิเศษเมื่อชุดข้อมูลของคุณไม่มีป้ายกำกับ หากมีป้ายกำกับ เทคนิคการจำแนกประเภทเช่นที่คุณเรียนในบทเรียนก่อนหน้าอาจมีประโยชน์มากกว่า แต่ในกรณีที่คุณต้องการจัดกลุ่มข้อมูลที่ไม่มีป้ายกำกับ การจัดกลุ่มเป็นวิธีที่ดีในการค้นหารูปแบบ

> มีเครื่องมือที่ใช้โค้ดน้อยที่มีประโยชน์ซึ่งช่วยให้คุณเรียนรู้เกี่ยวกับการทำงานกับแบบจำลองการจัดกลุ่ม ลองใช้ [Azure ML สำหรับงานนี้](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## บทเรียน

1. [แนะนำการจัดกลุ่ม](1-Visualize/README.md)
2. [การจัดกลุ่ม K-Means](2-K-Means/README.md)

## เครดิต

บทเรียนเหล่านี้เขียนด้วยเสียงเพลง 🎶 โดย [Jen Looper](https://www.twitter.com/jenlooper) พร้อมบทวิจารณ์ที่เป็นประโยชน์จาก [Rishit Dagli](https://rishit_dagli/) และ [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan)

ชุดข้อมูล [เพลงของไนจีเรีย](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) ได้มาจาก Kaggle โดยขูดข้อมูลจาก Spotify

ตัวอย่าง K-Means ที่มีประโยชน์ซึ่งช่วยในการสร้างบทเรียนนี้รวมถึง [การสำรวจดอกไอริส](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [สมุดจดแนะนำ](https://www.kaggle.com/prashant111/k-means-clustering-with-python), และ [ตัวอย่าง NGO สมมุติ](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**ปฏิเสธความรับผิดชอบ**:
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) ขณะที่เราพยายามให้ความถูกต้อง โปรดทราบว่าการแปลโดยอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาต้นทางควรถูกพิจารณาเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ แนะนำให้ใช้การแปลโดยมนุษย์มืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดที่เกิดขึ้นจากการใช้การแปลนี้
<!-- CO-OP TRANSLATOR DISCLAIMER END -->