<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T19:10:34+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "id"
}
-->
# Model Clustering untuk Pembelajaran Mesin

Clustering adalah tugas pembelajaran mesin yang bertujuan untuk menemukan objek yang mirip satu sama lain dan mengelompokkannya ke dalam kelompok yang disebut cluster. Yang membedakan clustering dari pendekatan lain dalam pembelajaran mesin adalah bahwa prosesnya terjadi secara otomatis. Faktanya, bisa dikatakan bahwa ini adalah kebalikan dari pembelajaran terawasi.

## Topik regional: model clustering untuk selera musik audiens Nigeria 🎧

Audiens Nigeria yang beragam memiliki selera musik yang beragam pula. Dengan menggunakan data yang diambil dari Spotify (terinspirasi oleh [artikel ini](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), mari kita lihat beberapa musik yang populer di Nigeria. Dataset ini mencakup data tentang skor 'danceability', 'acousticness', tingkat keras suara (loudness), 'speechiness', popularitas, dan energi dari berbagai lagu. Akan menarik untuk menemukan pola dalam data ini!

![Turntable](../../../5-Clustering/images/turntable.jpg)

> Foto oleh <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> di <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Dalam rangkaian pelajaran ini, Anda akan menemukan cara baru untuk menganalisis data menggunakan teknik clustering. Clustering sangat berguna ketika dataset Anda tidak memiliki label. Jika dataset Anda memiliki label, maka teknik klasifikasi seperti yang telah Anda pelajari dalam pelajaran sebelumnya mungkin lebih berguna. Namun, dalam kasus di mana Anda ingin mengelompokkan data yang tidak berlabel, clustering adalah cara yang bagus untuk menemukan pola.

> Ada alat low-code yang berguna untuk membantu Anda mempelajari cara bekerja dengan model clustering. Cobalah [Azure ML untuk tugas ini](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Pelajaran

1. [Pengantar clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Kredit

Pelajaran ini ditulis dengan 🎶 oleh [Jen Looper](https://www.twitter.com/jenlooper) dengan ulasan yang bermanfaat dari [Rishit Dagli](https://rishit_dagli) dan [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dataset [Lagu Nigeria](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) diambil dari Kaggle sebagai hasil scraping dari Spotify.

Contoh K-Means yang berguna yang membantu dalam membuat pelajaran ini termasuk [eksplorasi iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), [notebook pengantar](https://www.kaggle.com/prashant111/k-means-clustering-with-python), dan [contoh hipotetis NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.