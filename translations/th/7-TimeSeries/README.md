<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "61342603bad8acadbc6b2e4e3aab3f66",
  "translation_date": "2025-09-05T21:17:27+00:00",
  "source_file": "7-TimeSeries/README.md",
  "language_code": "th"
}
-->
# การแนะนำการพยากรณ์อนุกรมเวลา

การพยากรณ์อนุกรมเวลาคืออะไร? มันคือการคาดการณ์เหตุการณ์ในอนาคตโดยการวิเคราะห์แนวโน้มในอดีต

## หัวข้อภูมิภาค: การใช้ไฟฟ้าทั่วโลก ✨

ในสองบทเรียนนี้ คุณจะได้เรียนรู้เกี่ยวกับการพยากรณ์อนุกรมเวลา ซึ่งเป็นพื้นที่ที่ค่อนข้างไม่ค่อยมีคนรู้จักในด้านการเรียนรู้ของเครื่อง แต่มีคุณค่ามากสำหรับการใช้งานในอุตสาหกรรมและธุรกิจ รวมถึงสาขาอื่นๆ แม้ว่าเครือข่ายประสาทเทียมสามารถนำมาใช้เพื่อเพิ่มประสิทธิภาพของโมเดลเหล่านี้ได้ แต่เราจะศึกษาในบริบทของการเรียนรู้ของเครื่องแบบดั้งเดิม เนื่องจากโมเดลช่วยคาดการณ์ประสิทธิภาพในอนาคตโดยอ้างอิงจากข้อมูลในอดีต

หัวข้อภูมิภาคของเราคือการใช้ไฟฟ้าทั่วโลก ซึ่งเป็นชุดข้อมูลที่น่าสนใจสำหรับการเรียนรู้เกี่ยวกับการพยากรณ์การใช้พลังงานในอนาคตโดยอ้างอิงจากรูปแบบการใช้ในอดีต คุณจะเห็นว่าการพยากรณ์ประเภทนี้สามารถเป็นประโยชน์อย่างมากในสภาพแวดล้อมทางธุรกิจ

![electric grid](../../../7-TimeSeries/images/electric-grid.jpg)

ภาพถ่ายโดย [Peddi Sai hrithik](https://unsplash.com/@shutter_log?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) ของเสาไฟฟ้าบนถนนในรัฐราชสถานบน [Unsplash](https://unsplash.com/s/photos/electric-india?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## บทเรียน

1. [การแนะนำการพยากรณ์อนุกรมเวลา](1-Introduction/README.md)
2. [การสร้างโมเดล ARIMA สำหรับอนุกรมเวลา](2-ARIMA/README.md)
3. [การสร้าง Support Vector Regressor สำหรับการพยากรณ์อนุกรมเวลา](3-SVR/README.md)

## เครดิต

"การแนะนำการพยากรณ์อนุกรมเวลา" เขียนด้วย ⚡️ โดย [Francesca Lazzeri](https://twitter.com/frlazzeri) และ [Jen Looper](https://twitter.com/jenlooper) โน้ตบุ๊กปรากฏออนไลน์ครั้งแรกใน [Azure "Deep Learning For Time Series" repo](https://github.com/Azure/DeepLearningForTimeSeriesForecasting) ซึ่งเขียนโดย Francesca Lazzeri บทเรียน SVR เขียนโดย [Anirban Mukherjee](https://github.com/AnirbanMukherjeeXD)

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่แม่นยำ เอกสารต้นฉบับในภาษาต้นทางควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษาจากผู้เชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความที่ผิดพลาดซึ่งเกิดจากการใช้การแปลนี้