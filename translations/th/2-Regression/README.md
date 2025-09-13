<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-05T21:04:38+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "th"
}
-->
# โมเดลการถดถอยสำหรับการเรียนรู้ของเครื่อง
## หัวข้อภูมิภาค: โมเดลการถดถอยสำหรับราคาฟักทองในอเมริกาเหนือ 🎃

ในอเมริกาเหนือ ฟักทองมักถูกแกะสลักเป็นหน้าตาน่ากลัวสำหรับวันฮาโลวีน มาค้นพบเพิ่มเติมเกี่ยวกับผักที่น่าสนใจเหล่านี้กันเถอะ!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> ภาพถ่ายโดย <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> บน <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## สิ่งที่คุณจะได้เรียนรู้

[![Introduction to Regression](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Regression Introduction video - Click to Watch!")
> 🎥 คลิกที่ภาพด้านบนเพื่อดูวิดีโอแนะนำบทเรียนนี้อย่างรวดเร็ว

บทเรียนในส่วนนี้ครอบคลุมประเภทของการถดถอยในบริบทของการเรียนรู้ของเครื่อง โมเดลการถดถอยสามารถช่วยกำหนด _ความสัมพันธ์_ ระหว่างตัวแปรต่าง ๆ โมเดลประเภทนี้สามารถทำนายค่าต่าง ๆ เช่น ความยาว อุณหภูมิ หรืออายุ และช่วยเปิดเผยความสัมพันธ์ระหว่างตัวแปรในขณะที่วิเคราะห์จุดข้อมูล

ในชุดบทเรียนนี้ คุณจะค้นพบความแตกต่างระหว่างการถดถอยเชิงเส้นและการถดถอยแบบลอจิสติก และเรียนรู้ว่าเมื่อใดควรเลือกใช้แบบใด

[![ML for beginners - Introduction to Regression models for Machine Learning](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML for beginners - Introduction to Regression models for Machine Learning")

> 🎥 คลิกที่ภาพด้านบนเพื่อดูวิดีโอสั้น ๆ แนะนำโมเดลการถดถอย

ในกลุ่มบทเรียนนี้ คุณจะได้เตรียมตัวเริ่มต้นงานการเรียนรู้ของเครื่อง รวมถึงการตั้งค่า Visual Studio Code เพื่อจัดการโน้ตบุ๊ก ซึ่งเป็นสภาพแวดล้อมทั่วไปสำหรับนักวิทยาศาสตร์ข้อมูล คุณจะได้ค้นพบ Scikit-learn ซึ่งเป็นไลบรารีสำหรับการเรียนรู้ของเครื่อง และคุณจะสร้างโมเดลแรกของคุณ โดยเน้นไปที่โมเดลการถดถอยในบทนี้

> มีเครื่องมือที่ใช้โค้ดน้อยที่มีประโยชน์ซึ่งสามารถช่วยให้คุณเรียนรู้เกี่ยวกับการทำงานกับโมเดลการถดถอย ลองใช้ [Azure ML สำหรับงานนี้](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### บทเรียน

1. [เครื่องมือที่ใช้ในงาน](1-Tools/README.md)
2. [การจัดการข้อมูล](2-Data/README.md)
3. [การถดถอยเชิงเส้นและพหุนาม](3-Linear/README.md)
4. [การถดถอยแบบลอจิสติก](4-Logistic/README.md)

---
### เครดิต

"ML with regression" เขียนด้วย ♥️ โดย [Jen Looper](https://twitter.com/jenlooper)

♥️ ผู้ร่วมสร้างแบบทดสอบ ได้แก่: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) และ [Ornella Altunyan](https://twitter.com/ornelladotcom)

ชุดข้อมูลฟักทองแนะนำโดย [โปรเจกต์นี้บน Kaggle](https://www.kaggle.com/usda/a-year-of-pumpkin-prices) และข้อมูลมาจาก [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) ที่เผยแพร่โดยกระทรวงเกษตรของสหรัฐอเมริกา เราได้เพิ่มจุดข้อมูลเกี่ยวกับสีตามชนิดเพื่อปรับการกระจายให้เป็นมาตรฐาน ข้อมูลนี้อยู่ในโดเมนสาธารณะ

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ แนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้