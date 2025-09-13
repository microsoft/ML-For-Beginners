<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T19:44:20+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "vi"
}
-->
# Xây dựng ứng dụng web để sử dụng mô hình ML của bạn

Trong phần này của chương trình học, bạn sẽ được giới thiệu về một chủ đề ML ứng dụng: cách lưu mô hình Scikit-learn của bạn dưới dạng tệp để có thể sử dụng để dự đoán trong một ứng dụng web. Sau khi mô hình được lưu, bạn sẽ học cách sử dụng nó trong một ứng dụng web được xây dựng bằng Flask. Đầu tiên, bạn sẽ tạo một mô hình sử dụng một số dữ liệu liên quan đến các lần nhìn thấy UFO! Sau đó, bạn sẽ xây dựng một ứng dụng web cho phép bạn nhập số giây cùng với giá trị vĩ độ và kinh độ để dự đoán quốc gia nào đã báo cáo nhìn thấy UFO.

![Bãi đỗ UFO](../../../3-Web-App/images/ufo.jpg)

Ảnh của <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> trên <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Bài học

1. [Xây dựng ứng dụng web](1-Web-App/README.md)

## Tín dụng

"Xây dựng ứng dụng web" được viết với ♥️ bởi [Jen Looper](https://twitter.com/jenlooper).

♥️ Các câu đố được viết bởi Rohan Raj.

Bộ dữ liệu được lấy từ [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Kiến trúc ứng dụng web được gợi ý một phần bởi [bài viết này](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) và [repo này](https://github.com/abhinavsagar/machine-learning-deployment) của Abhinav Sagar.

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.