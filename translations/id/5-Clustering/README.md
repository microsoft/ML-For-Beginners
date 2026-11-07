# Model klasterisasi untuk pembelajaran mesin

Klasterisasi adalah tugas pembelajaran mesin yang bertujuan untuk menemukan objek yang mirip satu sama lain dan mengelompokkannya ke dalam grup yang disebut klaster. Yang membedakan klasterisasi dari pendekatan lain dalam pembelajaran mesin, adalah bahwa prosesnya terjadi secara otomatis, bahkan bisa dikatakan ini kebalikan dari pembelajaran terawasi. 

## Topik regional: model klasterisasi untuk selera musik audiens di Nigeria 🎧

Beragamnya audiens di Nigeria juga memiliki selera musik yang beragam. Dengan menggunakan data yang diambil dari Spotify (terinspirasi dari [artikel ini](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), mari kita lihat beberapa musik yang populer di Nigeria. Dataset ini mencakup data tentang skor 'danceability', 'acousticness', loudness, 'speechiness', popularitas, dan energi dari berbagai lagu. Akan menarik untuk menemukan pola di data ini!

![Turntable](../../../translated_images/id/turntable.f2b86b13c53302dc.webp)

> Foto oleh <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> di <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Dalam rangkaian pelajaran ini, Anda akan menemukan cara baru untuk menganalisis data menggunakan teknik klasterisasi. Klasterisasi sangat berguna ketika dataset Anda tidak memiliki label. Jika dataset memiliki label, maka teknik klasifikasi seperti yang Anda pelajari dalam pelajaran sebelumnya mungkin lebih berguna. Tetapi dalam kasus di mana Anda ingin mengelompokkan data tidak berlabel, klasterisasi adalah cara yang bagus untuk menemukan pola.

> Ada alat low-code yang berguna yang dapat membantu Anda belajar bekerja dengan model klasterisasi. Coba [Azure ML untuk tugas ini](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Pelajaran

1. [Pengenalan klasterisasi](1-Visualize/README.md)
2. [Pengelompokan K-Means](2-K-Means/README.md)

## Kredit

Pelajaran ini ditulis dengan 🎶 oleh [Jen Looper](https://www.twitter.com/jenlooper) dengan ulasan yang membantu oleh [Rishit Dagli](https://rishit_dagli/) dan [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dataset [Lagu Nigeria](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) diperoleh dari Kaggle yang diambil dari Spotify.

Contoh K-Means yang berguna yang membantu dalam pembuatan pelajaran ini termasuk [eksplorasi iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [notebook pengantar](https://www.kaggle.com/prashant111/k-means-clustering-with-python), dan contoh [NGO hipotetis](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Penafian**:
Dokumen ini telah diterjemahkan menggunakan layanan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk mencapai akurasi, harap diketahui bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang sah. Untuk informasi penting, disarankan menggunakan terjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->