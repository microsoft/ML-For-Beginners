<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T19:10:23+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "vi"
}
-->
# Các mô hình phân cụm trong học máy

Phân cụm là một nhiệm vụ trong học máy, nơi nó tìm cách xác định các đối tượng giống nhau và nhóm chúng lại thành các nhóm gọi là cụm. Điều làm phân cụm khác biệt so với các phương pháp khác trong học máy là mọi thứ diễn ra tự động, thực tế có thể nói rằng nó hoàn toàn trái ngược với học có giám sát.

## Chủ đề khu vực: các mô hình phân cụm cho sở thích âm nhạc của khán giả Nigeria 🎧

Khán giả đa dạng của Nigeria có sở thích âm nhạc phong phú. Sử dụng dữ liệu được thu thập từ Spotify (lấy cảm hứng từ [bài viết này](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), hãy cùng xem một số bài hát phổ biến ở Nigeria. Bộ dữ liệu này bao gồm thông tin về điểm 'danceability', 'acousticness', độ lớn âm thanh, 'speechiness', mức độ phổ biến và năng lượng của các bài hát. Sẽ rất thú vị khi khám phá các mẫu trong dữ liệu này!

![Một bàn xoay](../../../5-Clustering/images/turntable.jpg)

> Ảnh của <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> trên <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Trong loạt bài học này, bạn sẽ khám phá những cách mới để phân tích dữ liệu bằng các kỹ thuật phân cụm. Phân cụm đặc biệt hữu ích khi bộ dữ liệu của bạn không có nhãn. Nếu nó có nhãn, thì các kỹ thuật phân loại như những gì bạn đã học trong các bài học trước có thể hữu ích hơn. Nhưng trong trường hợp bạn muốn nhóm dữ liệu không có nhãn, phân cụm là một cách tuyệt vời để khám phá các mẫu.

> Có những công cụ low-code hữu ích có thể giúp bạn tìm hiểu cách làm việc với các mô hình phân cụm. Hãy thử [Azure ML cho nhiệm vụ này](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Các bài học

1. [Giới thiệu về phân cụm](1-Visualize/README.md)
2. [Phân cụm K-Means](2-K-Means/README.md)

## Tín dụng

Các bài học này được viết với 🎶 bởi [Jen Looper](https://www.twitter.com/jenlooper) cùng với các đánh giá hữu ích từ [Rishit Dagli](https://rishit_dagli) và [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Bộ dữ liệu [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) được lấy từ Kaggle, được thu thập từ Spotify.

Các ví dụ K-Means hữu ích hỗ trợ việc tạo bài học này bao gồm [khám phá iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [notebook giới thiệu](https://www.kaggle.com/prashant111/k-means-clustering-with-python), và [ví dụ giả định về NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm về bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.