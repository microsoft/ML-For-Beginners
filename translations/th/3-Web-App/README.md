<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T21:46:06+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "th"
}
-->
# สร้างเว็บแอปเพื่อใช้งานโมเดล ML ของคุณ

ในส่วนนี้ของหลักสูตร คุณจะได้เรียนรู้หัวข้อการประยุกต์ใช้ ML: วิธีการบันทึกโมเดล Scikit-learn เป็นไฟล์ที่สามารถนำไปใช้ในการทำนายภายในเว็บแอปพลิเคชัน เมื่อโมเดลถูกบันทึกแล้ว คุณจะได้เรียนรู้วิธีการใช้งานโมเดลในเว็บแอปที่สร้างด้วย Flask คุณจะเริ่มต้นด้วยการสร้างโมเดลโดยใช้ข้อมูลเกี่ยวกับการพบเห็น UFO! จากนั้น คุณจะสร้างเว็บแอปที่ช่วยให้คุณสามารถป้อนจำนวนวินาที พร้อมกับค่าละติจูดและลองจิจูด เพื่อทำนายว่าประเทศใดรายงานการพบเห็น UFO

![UFO Parking](../../../3-Web-App/images/ufo.jpg)

ภาพถ่ายโดย <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> บน <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## บทเรียน

1. [สร้างเว็บแอป](1-Web-App/README.md)

## เครดิต

"สร้างเว็บแอป" เขียนด้วย ♥️ โดย [Jen Looper](https://twitter.com/jenlooper)

♥️ แบบทดสอบเขียนโดย Rohan Raj

ชุดข้อมูลมาจาก [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)

สถาปัตยกรรมเว็บแอปได้รับการแนะนำบางส่วนจาก [บทความนี้](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) และ [repo นี้](https://github.com/abhinavsagar/machine-learning-deployment) โดย Abhinav Sagar

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้องมากที่สุด แต่โปรดทราบว่าการแปลโดยอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้