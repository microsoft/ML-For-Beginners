# Mô hình phân cụm cho học máy

Phân cụm là một nhiệm vụ học máy nhằm tìm các đối tượng giống nhau và nhóm chúng lại thành các nhóm gọi là cụm. Điều khác biệt của phân cụm so với các phương pháp khác trong học máy là mọi thứ diễn ra tự động, thực tế có thể nói đây là điều ngược lại với học có giám sát.

## Chủ đề khu vực: mô hình phân cụm cho sở thích âm nhạc của khán giả Nigeria 🎧

Khán giả đa dạng của Nigeria có sở thích âm nhạc đa dạng. Sử dụng dữ liệu lấy từ Spotify (lấy cảm hứng từ [bài viết này](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), chúng ta hãy xem xét một số bài hát phổ biến tại Nigeria. Bộ dữ liệu này bao gồm các thông tin về điểm 'danceability' (khả năng khiêu vũ), 'acousticness' (độ acoustic), âm lượng, 'speechiness' (độ lời nói), độ phổ biến và năng lượng của các bài hát khác nhau. Sẽ rất thú vị khi khám phá các mẫu trong dữ liệu này!

![Một bàn xoay đĩa](../../../translated_images/vi/turntable.f2b86b13c53302dc.webp)

> Ảnh bởi <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> trên <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Trong chuỗi bài học này, bạn sẽ khám phá những cách mới để phân tích dữ liệu bằng kỹ thuật phân cụm. Phân cụm đặc biệt hữu ích khi bộ dữ liệu của bạn không có nhãn. Nếu có nhãn, thì các kỹ thuật phân loại như bạn đã học trong các bài trước có thể sẽ hữu dụng hơn. Nhưng trong trường hợp bạn muốn nhóm dữ liệu chưa được gán nhãn, phân cụm là cách tuyệt vời để khám phá các mẫu.

> Có những công cụ low-code hữu ích có thể giúp bạn học cách làm việc với các mô hình phân cụm. Hãy thử [Azure ML cho nhiệm vụ này](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Các bài học

1. [Giới thiệu về phân cụm](1-Visualize/README.md)
2. [Phân cụm K-Means](2-K-Means/README.md)

## Tác giả

Các bài học này được viết cùng với âm nhạc 🎶 bởi [Jen Looper](https://www.twitter.com/jenlooper) với sự góp ý hữu ích từ [Rishit Dagli](https://rishit_dagli/) và [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Bộ dữ liệu [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) được lấy từ Kaggle, thu thập từ Spotify.

Các ví dụ K-Means hữu ích hỗ trợ trong việc tạo bài học này bao gồm ví dụ khám phá [hoa iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), sổ tay [giới thiệu](https://www.kaggle.com/prashant111/k-means-clustering-with-python) và ví dụ [tổ chức phi chính phủ giả định](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Tuyên bố miễn trừ trách nhiệm**:
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng bản dịch tự động có thể chứa lỗi hoặc sai sót. Tài liệu gốc bằng ngôn ngữ gốc nên được coi là nguồn tin chính thức. Đối với thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm về bất kỳ hiểu lầm hoặc giải thích sai nào phát sinh từ việc sử dụng bản dịch này.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->